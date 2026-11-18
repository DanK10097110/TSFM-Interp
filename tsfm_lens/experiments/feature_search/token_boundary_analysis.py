"""Independently audit saved boundary surfaces and flag precision-limited ratios."""
from pathlib import Path
import argparse
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def precision_eligible(left,middle,right):
    """Conservative FP32-relative resolution guard for a measured endpoint chord.

This is a numerical diagnostic threshold, not a confidence interval or a proof
of rounding-error magnitude. Near-uniform random logits can make probability
curvature ratios meaningless despite exact deterministic repeats.
"""
    x=[np.asarray(v,float) for v in (left,middle,right)]
    scale=max(float(np.sqrt(np.mean(v*v))) for v in x)
    span=float(np.sqrt(np.mean((x[2]-x[0])**2)))
    floor=8*np.finfo(np.float32).eps*scale
    return bool(span>floor),float(floor),span


def run(source,out):
    """Replay raw-vector arithmetic and summarize only numerically resolved curves."""
    out.mkdir(parents=True,exist_ok=False)
    protocol=read(source/'protocol.json');groups=[];summary=[];checks=0;largest=0.
    for condition in ('pretrained','random'):
        saved=read(source/f'{condition}_results.json');a=np.load(source/f'{condition}_arrays.npz')
        curve=[];bound=[]
        for index,seed in enumerate(protocol['seeds']):
            row=next(r for r in saved['rows'] if r['surface']=='boundary' and r['seed']==seed)
            hidden={}
            for cut in saved['cuts']:
                key=f'hidden_{cut}';v0=a[f'{index}_raw0_{key}'];v1=a[f'{index}_raw1_{key}'];v2=a[f'{index}_raw2_{key}']
                if not np.array_equal(v0,v1):raise ValueError('Saved same-token gate fails')
                value=float(np.linalg.norm((v2-v1).astype(float))/np.linalg.norm(v1.astype(float)))
                largest=max(largest,abs(value-row['relative_hidden_jump'][key]));checks+=1
                np.testing.assert_allclose(value,row['relative_hidden_jump'][key],rtol=1e-12,atol=1e-12)
                hidden[key]=value
            p0=a[f'{index}_raw0_probability'];p1=a[f'{index}_raw1_probability'];p2=a[f'{index}_raw2_probability']
            if not np.array_equal(p0,p1):raise ValueError('Probability same-token gate fails')
            tv=float(np.sum(abs(p2.astype(float)-p1.astype(float)))/2)
            np.testing.assert_allclose(tv,row['probability_tv'],rtol=2e-6,atol=1e-12);checks+=1
            bound.append(dict(seed=seed,raw_distance=row['raw_rms_distance'],changed_tokens=row['changed_tokens'],probability_tv=tv,
                              embedding_relative_jump=row['embedding_relative_jump'],relative_hidden_jump=hidden))
            for radius in protocol['radii']:
                for surface in [f'hidden_{cut}' for cut in saved['cuts']]+['probability']:
                    left,mid,right=[a[f'{index}_{radius}_{tag}_{surface}'].astype(float) for tag in ('left','middle','right')]
                    original=next(r for r in saved['rows'] if r['surface']==surface and r['seed']==seed and r['radius']==radius)
                    bend=float(np.linalg.norm(mid-(left+right)/2)/np.sqrt(mid.size))
                    eligible,floor,span=precision_eligible(left,mid,right)
                    np.testing.assert_allclose(bend,original['bend_rms'],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(span,original['span_rms'],rtol=1e-12,atol=1e-12);checks+=2
                    curve.append(dict(seed=seed,radius=radius,surface=surface,precision_eligible=eligible,
                                      resolution_floor=floor,span_rms=span,bend_rms=bend,ratio=None if not eligible else bend/span))
        for surface in [f'hidden_{cut}' for cut in saved['cuts']]+['probability']:
            for radius in protocol['radii']:
                selected=[r for r in curve if r['surface']==surface and r['radius']==radius]
                values=[r['ratio'] for r in selected if r['precision_eligible']]
                summary.append(dict(condition=condition,surface=surface,radius=radius,eligible_sources=len(values),total_sources=len(selected),
                                    mean_ratio=None if not values else float(np.mean(values))))
        groups.append(dict(condition=condition,boundaries=bound,curves=curve,
            means=dict(raw_distance=float(np.mean([r['raw_distance'] for r in bound])),
                       probability_tv=float(np.mean([r['probability_tv'] for r in bound])),
                       relative_hidden_jump={key:float(np.mean([r['relative_hidden_jump'][key] for r in bound])) for key in bound[0]['relative_hidden_jump']}),
            seconds=saved['seconds'],peak_rss_bytes=saved['peak_rss_bytes']))
    write(out/'results.json',groups);write(out/'summary.json',summary)
    write(out/'verification.json',dict(independent_checks=checks,largest_hidden_jump_discrepancy=largest,
        bindings={f.name:sha(f) for f in source.iterdir() if f.name.endswith(('_results.json','_arrays.npz'))},code_sha256=sha(__file__)))
    write(out/'knowledge.json',dict(status='complete',question=protocol['question'],checkpoint=protocol['checkpoint'],n_sources=len(protocol['seeds']),
        exact_findings=dict(boundary_means=[dict(condition=g['condition'],**g['means']) for g in groups],curve_summary=summary),
        controls='Exact same-token/ID-embedding/native repeats; FP64 saved-vector replay; conservative8-FP32-epsilon chord-resolution guard',
        limitations='Boundary-selected synthetic points; off-manifold embedding interpolation; one initialization per architecture; random probability ratios may be precision-limited; no multi-step forecasting claim',
        interpretation='Raw-input discontinuity has a known tokenizer cause; smooth embedding-coordinate behavior and trained depth contraction are distinct questions. Small random probability changes are not evidence of better forecasting.',
        next_decision='If the boundary/continuous-path distinction is clear, replicate on a second cached model size with fresh paired backgrounds.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source,a.out)
