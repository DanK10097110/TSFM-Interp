"""Evaluate four-view source-only codes, real target deletions, and paired middle decoder directions."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import causal_scores as causal
from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder import directions, evaluate as original, evaluation_scores as dose_score
from research.activation_transfer.moment.crosscoder import interventions as live, patch_probe
from research.activation_transfer.moment.crosscoder.input_sensitivity.score import summary as crossed_features
from research.activation_transfer.moment.token_paths.geometry import load

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
METHODS=('aligned','shuffled_depth','masked')
NEW=METHODS[:2]
DOMAINS=('daily','monash_traffic')


def label(method,seed):
    return method+str(cd.SEEDS.index(seed)+1)


def read(method,seed,d):
    tag=label(method,seed)
    if method=='masked':
        m4,weather,blind=original.representation(tag,d)
        with np.load(cd.CACHE/f'{method}_{seed}.npz') as f:
            return m4,weather,blind,f['decoder_small'],f['decoder_large']
    with np.load(OUT/f'{method}_{seed}.npz') as f:
        for k,x in [('m4_ids',d['m4_ids']),('station_ids',d['ids']),('dates',d['dates']),('blind_ids',d['blind_ids']),('blind_domains',d['blind_domain'])]:
            np.testing.assert_array_equal(f[k],x)
        return f['m4_source'],f['noaa_source'],f['blind_source'],f['decoder_small'],f['decoder_large']


def fit(method,seed):
    tag=label(method,seed); d=original.raw_data()
    m4,weather,blind,_,_=read(method,seed,d)
    w,b,cut,meta=original.calibrate(m4,d['y'])
    vr,vd=original.apply(m4[768:],w,b,cut);rr,dose=original.apply(weather,w,b,cut)
    br,bd=original.apply(blind,w,b,cut)
    np.testing.assert_allclose(((vd-d['y'][768:])**2).sum(0),meta['validation_sse'])
    if method=='masked':
        with np.load(original.OUT/f'{tag}_doses.npz') as f:
            np.testing.assert_allclose(dose,f['dose'],rtol=1e-6,atol=1e-5)
            np.testing.assert_array_equal(dose>0,f['dose']>0)
        return
    original.write_npz(OUT/f'{tag}_doses.npz',ids=d['ids'],dates=d['dates'],atoms=d['atoms'],raw=rr,dose=dose,
        **{tag:dose},validation_raw=vr,validation_dose=vd,validation_ids=d['m4_ids'][768:],
        blind_raw=br,blind_dose=bd,blind_ids=d['blind_ids'],blind_domains=d['blind_domain'])
    original.write_json(OUT/f'{tag}_readout.json',meta)


def physical(method,seed):
    if method=='masked':raise ValueError('Existing two-view finite waves are archived; avoid rerunning the controls.')
    tag=label(method,seed)
    live.run_arm(tag,OUT/f'{tag}_doses.npz',OUT,16,live.material(),load('large'))


def directional(method,seed):
    tag=label(method,seed);d=original.raw_data()
    _,_,codes,ds,dt=read(method,seed,d)
    ds,dt=ds.astype(float),dt.astype(float)
    x,ids,domain,_,jac=directions.read_views()
    np.testing.assert_array_equal(ids,d['blind_ids'])
    assert np.all((codes>0).sum(1)==32)
    atom=np.flatnonzero(codes>0).reshape(len(codes),32)%codes.shape[1]
    dose=np.take_along_axis(codes,atom,axis=1)
    dense=ds@directions.ordinary_map()
    dense*=np.linalg.norm(dt,axis=1,keepdims=True)/np.linalg.norm(dense,axis=1,keepdims=True)
    np.testing.assert_allclose(np.linalg.norm(dt,axis=1),np.linalg.norm(dense,axis=1),atol=1e-6)
    with np.load(patch_probe.DEST/'probe_all.npz') as f:
        source_obs=ds@f['small_weight'];target_obs=dt@f['large_weight'];mapped_obs=dense@f['large_weight']
    patch=patch_probe.compare(source_obs,target_obs,atom)
    patch_dense=patch_probe.compare(source_obs,mapped_obs,atom)
    waves={}
    for arm,axes,name in [('small',ds,'small'),('large',dt,'large'),('dense',dense,'large')]:
        waves[arm]=-dose[:,:,None]*(axes[atom]@jac[name].transpose(0,2,1))
    forecast=directions.compare(waves)
    metrics=dict(patch=patch['shape'],patch_other=patch['other_shape'],patch_margin=patch['shape']-patch['other_shape'],
                 patch_dense=patch_dense['shape'],patch_minus_dense=patch['shape']-patch_dense['shape'],
                 patch_level=patch['level_sign'],patch_level_other=patch['other_level_sign'],
                 forecast=forecast['paired'],forecast_other=forecast['within_history'],forecast_margin=forecast['paired']-forecast['within_history'],
                 forecast_dense=forecast['dense'],forecast_minus_dense=forecast['paired']-forecast['dense'])
    if method=='masked':
        with np.load(directions.OUT/f'{method}_{seed}.npz') as f:
            np.testing.assert_array_equal(f['ids'],ids);np.testing.assert_array_equal(f['atom'],atom)
            np.testing.assert_allclose(metrics['forecast'],f['paired'],atol=2e-6)
            np.testing.assert_allclose(metrics['forecast_other'],f['within_history'],atol=2e-6)
        with np.load(patch_probe.DEST/f'all_{method}_{seed}.npz') as f:
            np.testing.assert_array_equal(f['ids'][:len(ids)],ids)
            np.testing.assert_array_equal(f['atoms'][:len(ids)],atom)
            np.testing.assert_allclose(f['source'],source_obs,atol=2e-6)
    report={}
    for pop in DOMAINS:
        ix=domain==pop;assert ix.sum()==128
        report[pop]=crossed_features({k:v[ix] for k,v in metrics.items()},atom[ix])
        # Celsius is not common across these source domains. Use per-history SD for native derivatives.
        report[pop]['native_local_rms_history_sd']={}
        for k,z in waves.items():
            sig=(z[ix]-z[ix].mean(-1,keepdims=True))/x[ix].std(1)[:,None,None]
            report[pop]['native_local_rms_history_sd'][k]=np.quantile(np.sqrt(np.mean(sig**2,-1)),[.1,.5,.9]).tolist()
    original.write_npz(OUT/(tag+'_directions.npz'),ids=ids,domain=domain,atoms=atom,**metrics)
    original.write_json(OUT/(tag+'_directions.json'),report)


def interval(v):
    return dict(point=float(v[0]),paired95=np.nanquantile(v[1:],[.025,.975]).tolist())


def pairs(draws):
    combined=dict(draws)
    for method in METHODS:
        combined[method+'_mean']=(draws[label(method,cd.SEEDS[0])]+draws[label(method,cd.SEEDS[1])])/2
    asks=[('aligned_mean','shuffled_depth_mean'),('aligned_mean','masked_mean'),('shuffled_depth_mean','masked_mean')]
    for seed in cd.SEEDS:
        asks.extend((label(a,seed),label(b,seed)) for a,b in [('aligned','shuffled_depth'),('aligned','masked'),('shuffled_depth','masked')])
    for other in ('raw','map_all_exactgate'):
        if other in combined:
            asks += [(method+'_mean',other) for method in METHODS]+[(label(m,s),other) for m in NEW for s in cd.SEEDS]
    return {a+'_minus_'+b:combined[a]-combined[b] for a,b in asks}


def dose_summary():
    d=original.raw_data()
    with np.load(original.EXTERNAL/'donor.npz') as f:
        truth=f['dose'];np.testing.assert_array_equal(f['atoms'],d['atoms'])
        np.testing.assert_array_equal(f['ids'],d['ids']);np.testing.assert_array_equal(f['dates'],d['dates'])
    weights=dose_score.resample((35,11))
    report={};scores={};aps={}
    for method,seed in [(m,s) for m in METHODS for s in cd.SEEDS]+[('raw',None)]:
        tag=label(method,seed) if seed else 'raw'
        loc=OUT if method in NEW else original.OUT
        with np.load(loc/f'{tag}_doses.npz') as f:
            np.testing.assert_array_equal(f['ids'],d['ids']);np.testing.assert_array_equal(f['dates'],d['dates']);np.testing.assert_array_equal(f['atoms'],d['atoms'])
            a,report[tag]=dose_score.score(truth,f['dose'],f['raw'],weights)
        scores[tag],aps[tag]=a['skill'],a['ap']
    for k,vals in [('dose_skill',scores),('average_precision',aps)]:
        report[k+'_paired']={n:interval(v) for n,v in pairs(vals).items()}
    original.write_json(OUT/'dose_scores.json',report)


def direction_summary():
    report={}
    for pop in DOMAINS:
        weights=dose_score.resample((128,),draws=3000,seed=22112445)
        out={}
        for metric in ('patch_margin','forecast_margin','patch','forecast','patch_minus_dense','forecast_minus_dense'):
            draws={}
            for method in METHODS:
                for seed in cd.SEEDS:
                    tag=label(method,seed)
                    with np.load(OUT/f'{tag}_directions.npz') as f:
                        use=f['domain']==pop;values=f[metric][use].mean(1)
                        if not draws:ids=f['ids'][use]
                        else:np.testing.assert_array_equal(f['ids'][use],ids)
                    draws[tag]=(weights@values)/128
            out[metric]={k:interval(v) for k,v in pairs(draws).items()}
        report[pop]=out
    original.write_json(OUT/'direction_contrasts.json',report)


def physical_summary():
    controls=['raw','masked1','masked2','map_all_exactgate']
    tags=controls+[label(m,s) for m in NEW for s in cd.SEEDS]
    # Let the production scorer consume the common-case set without copying unchanged historical waves.
    with tempfile.TemporaryDirectory(prefix='moment_multilayer_score_') as temp:
        staging=Path(temp)
        for tag in tags:
            base=original.OUT if tag in controls else OUT
            (staging/f'causal_{tag}.npz').symlink_to((base/f'causal_{tag}.npz').resolve())
        causal.run(staging)
        full=json.loads((staging/'causal_summary.json').read_text())
    with np.load(original.EXTERNAL/'donor.npz') as f:
        sd=f['sd'];ref=causal.normalized_shape(f['wave'],sd);on=f['dose']>0
    rng=np.random.default_rng(22112397)
    sw=np.vstack([np.ones(35),rng.multinomial(35,np.full(35,1/35),3000)])
    tw=np.vstack([np.ones(11),rng.multinomial(11,np.full(11,1/11),3000)])
    total=lambda v:causal.crossed_sum(v,sw,tw)
    complete=np.all(total(np.sum(ref*ref,axis=-1))>1e-16,axis=1)
    draws={}
    for tag in tags:
        loc=original.OUT if tag in controls else OUT
        with np.load(loc/f'causal_{tag}.npz') as f:
            estimated=causal.normalized_shape(f['wave'],sd)
        score,_,_=causal.score_opportunities(ref,estimated,on,total)
        draws[tag]=np.nanmean(score,axis=1)
        np.testing.assert_allclose(draws[tag][0],full['arms'][tag]['all']['value'])
    comparisons={}
    for k,v in pairs(draws).items():
        comparisons[k]=dict(**interval(v),complete_four95=np.quantile(v[1:][complete[1:]],[.025,.975]).tolist())
    full['multilayer_paired']=comparisons
    original.write_json(OUT/'native_scores.json',full)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fit','physical','directions','summarize'))
    parser.add_argument('--method',choices=METHODS);parser.add_argument('--seed',choices=cd.SEEDS,type=int)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='summarize':dose_summary();direction_summary();physical_summary()
    else:{'fit':fit,'physical':physical,'directions':directional}[args.stage](args.method,args.seed)
