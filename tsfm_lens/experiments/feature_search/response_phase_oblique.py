"""Prepare unseen oblique phase inputs on the existing16 transfer backgrounds.

No predictors are fitted: zero/scalar/complex/fullH are copied byte-for-byte from
the old-only predictor artifact. Frozen transfer capture is reused with a300s cap.
Cardinal-versus-oblique comparisons are paired by source, not by phase row.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search import response_phase_transfer as capture_driver

ANGLES = np.pi/8 + np.arange(8)*np.pi/4
CARDINAL = np.arange(4)*np.pi/2
BASELINES = ('zero','scalar_bounded','complex_linear','full_H')
PERIOD = 19.2


def prediction(h,amplitude,angles):
    """Apply the fixed old map in the original forecast-origin sine/cosine coordinates."""
    t=np.arange(32);b=np.column_stack([np.sin(2*np.pi*t/PERIOD),np.cos(2*np.pi*t/PERIOD)])
    c=np.column_stack([np.cos(angles),np.sin(angles)])
    return amplitude*c@np.asarray(h,float).T@b.T


def score(delta,amplitude,angles,h):
    """Phase-pooled error and exact opposite-pair odd/even residual decomposition."""
    delta=np.asarray(delta,float);n=len(angles);half=n//2
    if delta.shape!=(n,32) or n not in (4,8):raise ValueError('Need full cardinal or oblique opposite pairs')
    ideal=prediction(np.eye(2),amplitude,angles);pred=prediction(h,amplitude,angles)
    denominator=float(np.sum(ideal**2));error=delta-pred
    odd=(delta[:half]-delta[half:])/2;even=delta[:half]+delta[half:]
    predicted_odd=(pred[:half]-pred[half:])/2
    odd_nmse=float(2*np.sum((odd-predicted_odd)**2)/denominator)
    even_nmse=float(np.sum(even**2)/(2*denominator))
    nmse=float(np.sum(error**2)/denominator)
    if abs(nmse-odd_nmse-even_nmse)>1e-10:raise ValueError('Opposite-pair energy closure failed')
    return dict(nmse=nmse,odd_residual_nmse=odd_nmse,even_nmse=even_nmse,
                per_angle_nmse=(np.sum(error**2,axis=1)/np.sum(ideal**2,axis=1)).tolist(),
                per_angle_squared_error=np.sum(error**2,axis=1).tolist(),
                per_angle_ideal_energy=np.sum(ideal**2,axis=1).tolist()),dict(delta=delta,prediction=pred,ideal=ideal,odd=odd,even=even)


def assert_background_parity(new,old):
    """The same observed background must have exactly the same forecast in both cohorts."""
    if not np.array_equal(new,old):raise ValueError('Cardinal/oblique background forecast parity failed')


def bank_from_prior(inputs,groups):
    """Reuse exact saved backgrounds; add only eight previously unevaluated phase histories."""
    bank=[];nodes=[];newgroups=[]
    selected=sorted([g for g in groups if g['center']==128/PERIOD and g['offset']==0],key=lambda g:g['source_id'])
    if len(selected)!=16 or [g['source_id'] for g in selected]!=list(range(16)):raise ValueError('Need existing16 center groups')
    worst=0.;closure=0.;distances=[]
    for g in selected:
        sid=g['source_id'];x=inputs[g['base']].copy();base=len(bank);bank.append(x)
        nodes.append(dict(index=base,source_id=sid,source=g['source'],kind='background'))
        indices=[];a=g['amplitude'];t=np.arange(-128,0)
        nominal=np.stack([a*np.sin(2*np.pi*t/PERIOD+angle) for angle in ANGLES])
        for k,tone in enumerate(nominal):
            idx=len(bank);indices.append(idx);bank.append((x.astype(float)+tone).astype(np.float32))
            nodes.append(dict(index=idx,source_id=sid,source=g['source'],kind='oblique',root=base,angle_id=k,angle=float(ANGLES[k])))
        observed=np.stack([bank[i].astype(float)-x for i in indices])
        worst=max(worst,float(np.sqrt(np.mean((observed-nominal)**2))/a))
        closure=max(closure,float(np.sqrt(np.mean((observed[:4]+observed[4:])**2))/a))
        distances.extend((np.sqrt(np.mean(observed**2,axis=1))/g['input_scale']).tolist())
        newgroups.append(dict(source=g['source'],source_id=sid,background_type=g['background_type'],base=base,indices=indices,
            cardinal_base=g['base'],cardinal_indices=g['indices'],amplitude=a,input_scale=g['input_scale']))
    if worst>1e-5 or closure>2e-5:raise ValueError('Nominal oblique input/closure gate failed')
    return np.stack(bank),nodes,newgroups,dict(nominal_rms_over_peak_max=worst,antipodal_rms_over_peak_max=closure,
        actual_input_rms_over_background_sd_min=min(distances),actual_input_rms_over_background_sd_max=max(distances))


def source_draws(types):
    """Common stratified source draws; eight angles remain dependent within each source."""
    rng=np.random.default_rng(4954500);types=np.asarray(types)
    return np.concatenate([rng.choice(np.flatnonzero(types==k),size=(16384,np.sum(types==k))) for k in sorted(set(types))],axis=1)


def interval(values,draws):
    """Paired source intervals; family4 applies only to the four declared native contrasts."""
    values=np.asarray(values,float);means=values[draws].mean(axis=1)
    return dict(mean=float(values.mean()),positive_sources=int(np.sum(values>0)),source_differences=values.tolist(),
                percentile95=np.quantile(means,[.025,.975]).tolist(),family4_percentile=np.quantile(means,[.00625,.99375]).tolist())


def prepare(out,prior):
    """Freeze144 inputs and old predictor bindings; do not create a native slot grant."""
    previous=capture_driver.validate(prior)
    if read(prior/'knowledge.json')['status']!='complete_audited':raise ValueError('Prior transfer cohort must be audited')
    execution=read(prior/'execution.json')
    if not all(execution[m]['valid'] for m in ('bolt','timesfm')):raise ValueError('Prior native execution invalid')
    inputs=np.load(prior/'inputs.npy');original_groups=read(prior/'groups.json')
    bank,nodes,groups,gates=bank_from_prior(inputs,original_groups)
    for g in groups:
        for idx in g['indices']:
            if any(np.array_equal(bank[idx],inputs[old]) for old in g['cardinal_indices']):raise ValueError('Oblique input duplicates cardinal input')
    out.mkdir(parents=True,exist_ok=False);np.save(out/'inputs.npy',bank)
    for name,value in [('nodes',nodes),('groups',groups),('sources',read(prior/'sources.json'))]:write(out/f'{name}.json',value)
    (out/'frozen_predictors.json').write_bytes((prior/'frozen_predictors.json').read_bytes())
    np.save(out/'bootstrap_source_indices.npy',source_draws([g['background_type'] for g in groups]))
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__),root/'experiments/feature_search/response_phase_oblique_audit.py',
           root/'experiments/feature_search/response_phase_oblique_plants.py',root/'tests/test_response_phase_oblique.py']
    source_hashes=dict(previous['source_hashes']);source_hashes.update({str(f.resolve()):sha(f) for f in files})
    names=['inputs.npy','nodes.json','groups.json','sources.json','frozen_predictors.json','bootstrap_source_indices.npy']
    prior_names=['inputs.npy','groups.json','sources.json','frozen_predictors.json','protocol.json','execution.json','audit.json']
    for model in ('bolt','timesfm'):
        worker=read(prior/f'{model}_worker.json')
        if any(worker['gates'].values()) or not worker['batch_replay_exact']:raise ValueError('Prior native gates failed')
        for policy in ('native','locked'):
            name=f'{model}_{policy}_forecast.npy'
            if sha(prior/name)!=worker['hashes'][name]:raise ValueError('Prior forecast binding changed')
            prior_names.append(name)
    budget={m:dict(seconds=1.1*(20+execution[m]['seconds']*144/400),formula='1.1*(20s allowance + actual400-node elapsed *144/400); estimate only') for m in ('bolt','timesfm')}
    if any(v['seconds']>=300 for v in budget.values()):raise ValueError('Prospective budget exceeds cap')
    protocol=dict(question='Does old frozen phase-response prediction transfer from cardinal to previously unseen oblique phases on the same16 backgrounds?',
        stage='prepared_capture_held',source_count=16,nodes=144,perturbed_inputs=128,backgrounds=16,
        period=PERIOD,angles=ANGLES.tolist(),cardinal_angles=CARDINAL.tolist(),context=128,horizon=32,
        source_role='Reuse exact16 fresh-transfer backgrounds4954300..4954315. NO new independent sources claimed. Eight new oblique input phases per source; phases never inflate n.',
        coefficients='Byte-identical old-only frozen_predictors.json copied from transfer4954300. Only center19.2 zero, bounded scalar, complexlinear and fullH. Same finite-horizon19.2 basis. NO fits, frequency shifts, per-source oracle, or new predictor selection.',
        primary='Native: for each model (1) oblique scalar NMSE minus fullH NMSE; (2) oblique fullH NMSE minus cardinal fullH NMSE. Four paired-source contrasts,16384 common type-stratified source bootstrap draws seed4954500; nominal95 and family4 Bonferroni98.75 percentile intervals. No equivalence claim from nonsignificance, no automatic threshold.',
        descriptive='Locked policy; each baseline zero/scalar/complex/fullH; each of8 angles per source retained. Cardinal4 and oblique8 scores each pool actual waveform error over all angles before dividing by total ideal signal energy. No per-angle ratio averaging for primary.',
        odd_even='Opposite pairs i,i+4 for oblique and i,i+2 for cardinal. Save raw odd differences and even sums relative to background; each frozen predictor error splits into odd residual plus even energy. No new fit. Input antipodal FP32 closure is measured.',
        controls='Previous frozen capture repeat, normalizer endpoint-state parity, self-patch and same-batch replay; after capture require exact new-vs-saved cardinal background forecast parity, all four model/policy settings. Checkpoint/code/input/old-predictor hashes; finite checks.',
        limitations='A finite eight-angle holdout can expose cardinal aliasing but cannot establish a globally linear phase law; higher harmonics can alias even this grid. Conditional on same16 synthetic sources, fixed amplitude.1 and32-step horizon. Output-frequency distortion may contribute shared error; no forecasting repair, absent-information or patch-causality claim.',
        native_capture='Reuse frozen response_phase_transfer.run/worker/capture_bank, forecasts only with transient endpoint-state identity checks.',
        cpu_threads=1,worker_cap=3*2**30,seconds_per_model=300,budget=budget,capture_hold=True,
        grant_contract='No grant created by prepare. Native entry points require explicit external coordinator slot grant bound to this exact protocol; queue behind Plato32freshcausal and Popper64electricity.',
        checkpoints=previous['checkpoints'],prior=str(prior.resolve()),prior_hashes={name:sha(prior/name) for name in prior_names},
        source_hashes=source_hashes,hashes={name:sha(out/name) for name in names})
    write(out/'protocol.json',protocol)
    write(out/'knowledge.json',dict(status='prepared_native_capture_held',question=protocol['question'],design=protocol,
        exact_findings=dict(input_gates=gates,nodes=144,new_sources=0,unseen_phase_inputs=128,native_results=None),
        controls=protocol['controls'],limitations=protocol['limitations'],evidence_class='Frozen prospective within-source phase holdout; no native outcomes',
        failed_instrumentation_runs=[],next_decision='HOLD native. Await coordinator slot grant after queued studies; no grant requested or created.'))


def analyze(out):
    """Saved-array comparisons without fitting any coefficients or selecting phase methods."""
    p=capture_driver.validate(out);prior=Path(p['prior']);execution=read(out/'execution.json')
    if (out/'results.json').exists():raise ValueError('Refuse overwrite analysis')
    for name,digest in p['prior_hashes'].items():
        if sha(prior/name)!=digest:raise ValueError('Prior cardinal/predictor binding changed')
    groups=read(out/'groups.json');predictors=read(out/'frozen_predictors.json');draws=np.load(out/'bootstrap_source_indices.npy')
    rows=[];arrays={}
    for model in ('bolt','timesfm'):
        if not execution[model]['valid']:raise ValueError('Invalid oblique capture')
        worker=read(out/f'{model}_worker.json')
        if any(worker['gates'].values()) or not worker['batch_replay_exact']:raise ValueError('Oblique native gates failed')
        for policy in ('native','locked'):
            name=f'{model}_{policy}_forecast.npy'
            if sha(out/name)!=worker['hashes'][name]:raise ValueError('Oblique forecast hash changed')
            forecast=np.load(out/name);cardinal=np.load(prior/name)
            maps=next(r['matrices'] for r in predictors if r['model']==model and r['policy']==policy and r['center']==128/PERIOD)
            for g in groups:
                assert_background_parity(forecast[g['base']],cardinal[g['cardinal_base']])
                scores={}
                for cohort,angles,y in [('oblique',ANGLES,forecast[g['indices']].astype(float)-forecast[g['base']].astype(float)),
                                        ('cardinal',CARDINAL,cardinal[g['cardinal_indices']].astype(float)-cardinal[g['cardinal_base']].astype(float))]:
                    scores[cohort]={}
                    for baseline in BASELINES:
                        metrics,raw=score(y,g['amplitude'],angles,maps[baseline]);scores[cohort][baseline]=metrics
                        prefix=f'{model}_{policy}_{g["source_id"]}_{cohort}_{baseline}'
                        arrays.update({f'{prefix}_{k}':v for k,v in raw.items()})
                rows.append(dict(g,model=model,policy=policy,scores=scores))
    primary=[];descriptive=[]
    for model in ('bolt','timesfm'):
        for policy in ('native','locked'):
            selected=[r for r in rows if r['model']==model and r['policy']==policy]
            contrast={
                'oblique_scalar_minus_fullH':[r['scores']['oblique']['scalar_bounded']['nmse']-r['scores']['oblique']['full_H']['nmse'] for r in selected],
                'oblique_minus_cardinal_fullH':[r['scores']['oblique']['full_H']['nmse']-r['scores']['cardinal']['full_H']['nmse'] for r in selected]}
            records=[dict(model=model,policy=policy,contrast=name,**interval(v,draws)) for name,v in contrast.items()]
            if policy=='native':primary.extend(records)
            for baseline in BASELINES:
                descriptive.append(dict(model=model,policy=policy,baseline=baseline,
                    mean_scores={cohort:{field:float(np.mean([r['scores'][cohort][baseline][field] for r in selected])) for field in ('nmse','odd_residual_nmse','even_nmse')} for cohort in ('cardinal','oblique')},
                    oblique_minus_cardinal=interval([r['scores']['oblique'][baseline]['nmse']-r['scores']['cardinal'][baseline]['nmse'] for r in selected],draws)))
    np.savez_compressed(out/'raw_arithmetic.npz',**arrays);write(out/'source_rows.json',rows)
    results=dict(primary=primary,descriptive=descriptive,background_parity_exact=True,predictors_refitted=False)
    write(out/'results.json',results);k=read(out/'knowledge.json');k.update(status='computed_pending_audit',exact_findings=results,
        next_decision='Release native slot after capture, independently audit saved parity/phase/error arithmetic, then interpret within-source phase holdout.')
    write(out/'knowledge.json',k)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','worker','analyze'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prior',type=Path,default=Path('runs/response_phase_transfer4954300'))
    parser.add_argument('--model',choices=['bolt','timesfm']);parser.add_argument('--grant',type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare(args.out,args.prior)
    elif args.command in ('run','worker'):
        if args.model is None:parser.error('--model required')
        (capture_driver.run if args.command=='run' else capture_driver.worker)(args.out,args.model,args.grant)
    else:
        try:analyze(args.out)
        except Exception as error:
            k=read(args.out/'knowledge.json');k['status']='failed_instrumentation';k['failed_instrumentation_runs'].append(dict(stage='analysis',error=repr(error)));write(args.out/'knowledge.json',k);raise
