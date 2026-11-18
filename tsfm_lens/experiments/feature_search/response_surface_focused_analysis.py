"""Source-grouped focused-v2 analysis, roundoff diagnostics and independent arithmetic audit.

Forecast semantics, hidden geometry and four-point interactions remain separate.
All source and phase rows survive; numerical-resolution flags never select cases.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha
from experiments.feature_search.response_surface_metrics import rms,difference,ratio,local_response,interaction,known_response,summarize


def rounding_proxy(arrays,weights):
    """FP32 half-ULP accumulation proxy; not a bound on the model's forward computation."""
    widths=[abs(np.spacing(np.asarray(a,np.float32))).astype(float)/2 for a in arrays]
    return rms(sum(abs(w)*d for w,d in zip(weights,widths)))


def score_arrays(values,inputs,paths,pairs,signals,model,policy,cut):
    """Compute all local and interaction rows in FP64, retaining measured denominator sizes."""
    local=[];mixed=[];responses=[];saved={}
    first=[];second=[];quartets=[];response_vectors=[];truth_vectors=[]
    for row in paths:
        ym,y0,yp,mid=[values[row[k]] for k in ('minus','zero','plus','midpoint')]
        result=local_response(ym,y0,yp,mid,inputs[row['minus']],inputs[row['zero']],inputs[row['plus']],row['step'],row['input_scale'])
        storage=rounding_proxy([yp,ym,mid],[1,1,-2])
        path_rounding=rms(inputs[row['plus']].astype(float)+inputs[row['minus']]-2*inputs[row['midpoint']].astype(float))/row['input_scale']
        input_proxy=(result['distance_sensitivity'] or 0.)*path_rounding
        local.append(dict(row,**result,model=model,policy=policy,cut=cut,storage_rounding_proxy=storage,
                          input_rounding_proxy=input_proxy,above_roundoff_proxy=bool(result['chord_second_rms']>10*(storage+input_proxy))))
        if cut=='forecast':
            first.append(difference(yp,ym)/(2*row['step']))
            second.append(yp.astype(float)+ym.astype(float)-2*mid.astype(float))
    for row in pairs:
        y0,ya,yb,yab=[values[row[k]] for k in ('base','a','b','ab')]
        result=interaction(y0,ya,yb,yab,difference(inputs[row['a']],inputs[row['base']]),
                           difference(inputs[row['b']],inputs[row['base']]),row['input_scale'])
        storage=rounding_proxy([yab,ya,yb,y0],[1,-1,-1,1])
        closure=rms(inputs[row['ab']].astype(float)-inputs[row['a']]-inputs[row['b']]+inputs[row['base']])/row['input_scale']
        gain=max(result['effect_a_rms']/result['input_a'],result['effect_b_rms']/result['input_b'])
        input_proxy=closure*gain
        mixed.append(dict(row,**result,model=model,policy=policy,cut=cut,storage_rounding_proxy=storage,
                          input_rounding_proxy=input_proxy,above_roundoff_proxy=bool(result['interaction_rms']>10*(storage+input_proxy))))
        if cut=='forecast':quartets.append(yab.astype(float)-ya.astype(float)-yb.astype(float)+y0.astype(float))
    if cut=='forecast':
        for row in signals:
            effect=difference(values[row['zero']],values[row['base']])
            truth=row['amplitude']*row['input_scale']*np.sin(2*np.pi*row['frequency']*np.arange(32)/128+row['phase'])
            responses.append(dict(model=model,policy=policy,**row,**known_response(effect,truth)))
            response_vectors.append(effect);truth_vectors.append(truth)
        saved=dict(local_first=np.asarray(first),local_chord_second=np.asarray(second),interaction=np.asarray(quartets),
                   response=np.asarray(response_vectors),known_signal=np.asarray(truth_vectors))
    return local,mixed,responses,saved


def convergence(records):
    """Average dependent phase/frequency conditions inside each source before shrinkage counts."""
    grouped={}
    for r in records:
        key=tuple(r[k] for k in ('model','policy','cut','axis','source'))
        grouped.setdefault(key,{}).setdefault(r['radius'],[]).append(r)
    source_rows=[]
    for key,levels in grouped.items():
        if set(levels)!={.002,.005,.01}:raise ValueError('Incomplete radius triple')
        mean=lambda radius,field:float(np.mean([r[field] for r in levels[radius] if r[field] is not None])) if any(r[field] is not None for r in levels[radius]) else None
        small,medium,large=[mean(radius,'chord_relative') for radius in (.002,.005,.01)]
        ss,sl=mean(.002,'distance_sensitivity'),mean(.01,'distance_sensitivity')
        source_rows.append(dict(zip(('model','policy','cut','axis','source'),key),
            small_curvature=small,medium_curvature=medium,large_curvature=large,
            curvature_decrease=None if small is None or large is None else large-small,
            shrinks=None if small is None or large is None else small<large,
            sensitivity_ratio=None if ss is None or sl is None else ratio(ss,sl),
            small_above_roundoff_fraction=float(np.mean([r['above_roundoff_proxy'] for r in levels[.002]])),
            dependent_cells_per_radius=len(levels[.002])))
    output=[]
    for key in sorted(set(tuple(r[k] for k in ('model','policy','cut','axis')) for r in source_rows)):
        own=[r for r in source_rows if tuple(r[k] for k in ('model','policy','cut','axis'))==key]
        valid=[r for r in own if r['shrinks'] is not None]
        item=dict(zip(('model','policy','cut','axis'),key),n_sources=len(own),n_valid_sources=len(valid),
                  shrink_source_count=sum(r['shrinks'] for r in valid))
        for field in ('small_curvature','medium_curvature','large_curvature','curvature_decrease','sensitivity_ratio','small_above_roundoff_fraction'):
            xs=[r[field] for r in own if r[field] is not None]
            item[field]=dict(mean=float(np.mean(xs)),median=float(np.median(xs)),min=float(min(xs)),max=float(max(xs))) if xs else None
        output.append(item)
    return source_rows,output


def independent_audit(out,model,local,mixed,responses):
    """Replay reported arithmetic with separate direct norm/dot formulas and saved arrays."""
    arrays={};inputs=np.load(out/'inputs.npy',mmap_mode='r');checks=0;maximum=0.
    def values(row):
        """Load one raw endpoint by policy/cut."""
        cut=row.get('cut','forecast');key=(row['policy'],cut)
        if key not in arrays:
            suffix='forecast' if cut=='forecast' else f'hidden_{cut}'
            arrays[key]=np.load(out/f'{model}_{key[0]}_{suffix}.npy',mmap_mode='r')
        return arrays[key]
    def norm(a):
        """Direct L2 per-element norm, independent of the scoring helper."""
        return float(np.linalg.norm(np.ravel(a))/np.sqrt(np.size(a)))
    def check(a,b):
        """Assert consistent finite arithmetic without silently flooring null denominators."""
        nonlocal checks,maximum
        if a is None or b is None:assert a is b
        else:
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11);maximum=max(maximum,abs(a-b))
        checks+=1
    for r in local:
        y=values(r);minus,plus,mid=[y[r[k]].astype(float) for k in ('minus','plus','midpoint')]
        dx=norm(inputs[r['plus']].astype(float)-inputs[r['minus']])/2/r['input_scale']
        chord=norm(plus+minus-2*mid);effects=norm(plus-mid)+norm(minus-mid)
        check(r['input_radius'],dx);check(r['distance_sensitivity'],norm(plus-minus)/(2*dx))
        check(r['chord_second_rms'],chord);check(r['chord_relative'],chord/effects if effects else None)
    for r in mixed:
        y=values(r);base,a,b,ab=[y[r[k]].astype(float) for k in ('base','a','b','ab')]
        residual=norm(ab-a-b+base);denom=norm(a-base)+norm(b-base)
        check(r['interaction_rms'],residual);check(r['relative_sum'],residual/denom if denom else None)
    for r in responses:
        y=values(r);effect=y[r['zero']].astype(float)-y[r['base']].astype(float)
        truth=r['amplitude']*r['input_scale']*np.sin(2*np.pi*r['frequency']*np.arange(32)/128+r['phase'])
        check(r['nmse'],float(np.dot(effect-truth,effect-truth)/np.dot(truth,truth)))
    paths=read(out/'paths.json');pairs=read(out/'interactions.json')
    for policy in ('native','locked'):
        raw=np.load(out/f'{model}_{policy}_forecast.npy',mmap_mode='r');saved=np.load(out/f'{model}_{policy}_arithmetic.npz')
        for i,r in enumerate(paths):
            direct=raw[r['plus']].astype(float)+raw[r['minus']].astype(float)-2*raw[r['midpoint']].astype(float)
            np.testing.assert_array_equal(saved['local_chord_second'][i],direct);checks+=1
        for i,r in enumerate(pairs):
            direct=raw[r['ab']].astype(float)-raw[r['a']].astype(float)-raw[r['b']].astype(float)+raw[r['base']].astype(float)
            np.testing.assert_array_equal(saved['interaction'][i],direct);checks+=1
    return dict(passed=True,checks=checks,maximum_absolute_discrepancy=maximum)


def analyze(out,model):
    """Analyze only a completed valid worker and distill every successful or failed analysis."""
    from experiments.feature_search.response_surface_focused import validate
    p=validate(out);execution=read(out/'execution.json')
    models=[model] if model else [m for m,r in execution.items() if r['valid']]
    if not models:raise ValueError('No completed valid native workers')
    for model in models:
        try:
            if not execution.get(model,{}).get('valid'):raise ValueError('Invalid worker; cannot infer')
            if (out/f'{model}_analysis.json').exists():raise ValueError('Refuse overwrite completed analysis')
            worker=read(out/f'{model}_worker.json')
            for name,digest in worker['hashes'].items():
                if sha(out/name)!=digest:raise ValueError('Raw artifact changed')
            inputs=np.load(out/'inputs.npy',mmap_mode='r')
            paths,pairs,signals=[read(out/f'{name}.json') for name in ('paths','interactions','signals')]
            types={s['source']:s['background_type'] for s in read(out/'sources.json')}
            for records in (paths,pairs,signals):
                for r in records:r['background_type']=types[r['source']]
            local=[];mixed=[];responses=[]
            for policy in ('native','locked'):
                for cut in ['forecast']+worker['cuts']:
                    suffix='forecast' if cut=='forecast' else f'hidden_{cut}'
                    values=np.load(out/f'{model}_{policy}_{suffix}.npy',mmap_mode='r')
                    a,b,c,saved=score_arrays(values,inputs,paths,pairs,signals,model,policy,cut)
                    local.extend(a);mixed.extend(b);responses.extend(c)
                    if saved:np.savez_compressed(out/f'{model}_{policy}_arithmetic.npz',**saved)
            source_curves,curves=convergence(local)
            audit=independent_audit(out,model,local,mixed,responses)
            summary=dict(model=model,source_count=p['source_count'],curvature=curves,
                local=summarize(local,['model','policy','cut','axis','radius'],['distance_sensitivity','chord_relative','above_roundoff_proxy']),
                interactions=summarize(mixed,['model','policy','cut','radius'],['interaction_rms','relative_sum','relative_a','relative_b','per_input_product','above_roundoff_proxy']),
                known_response=summarize(responses,['model','policy','frequency'],['nmse','gain']),
                known_response_by_type=summarize(responses,['model','policy','frequency','background_type'],['nmse','gain']),
                audit=audit,evidence='Descriptive fresh synthetic source-grouped geometry and behavioral response; no global smoothness, causal semantics or training-prior claim',
                resolution='Roundoff proxy flags are descriptive only; all rows retained. Shared amplitude midpoint has saved FP32 input closure; matched input radii are measured after quantization.')
            for suffix,value in [('local_rows',local),('interaction_rows',mixed),('response_rows',responses),('source_curves',source_curves),('analysis',summary)]:
                write(out/f'{model}_{suffix}.json',value)
            knowledge=read(out/'knowledge.json');knowledge.setdefault('model_results',{})[model]=summary
            knowledge['status']='analyzed_available_models';knowledge['evidence_class']=summary['evidence']
            knowledge['next_decision']='Review source-level curves, semantic responses and numerical floors. No expansion or additional native worker without a coordinator slot grant.'
            write(out/'knowledge.json',knowledge)
            manifest={f.name:sha(f) for f in out.glob(f'{model}_*') if f.is_file() and not f.name.endswith('_analysis_manifest.json')}
            write(out/f'{model}_analysis_manifest.json',manifest);print(dict(model=model,audit=audit),flush=True)
        except Exception as error:
            knowledge=read(out/'knowledge.json');knowledge['failed_instrumentation_runs'].append(dict(stage='analysis',model=model,error=repr(error)))
            knowledge['status']='failed_analysis_instrumentation';knowledge['next_decision']='Preserve raw outputs and diagnose the analysis; do not infer from failed summaries.'
            write(out/'knowledge.json',knowledge);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',choices=['bolt','timesfm']);args=parser.parse_args();analyze(args.out,args.model)
