"""Independent saved-array arithmetic for the closed native response PoC."""
from pathlib import Path
import numpy as np
from experiments.feature_search.dictionary_response_poc import read,write,sha


def audit(root):
    """Recompute endpoint errors via flattened dot products, without native calls."""
    rows=read(root/'results.json');summary=read(root/'summary.json');p=read(root/'protocol.json')
    forecasts=np.load(root/'forecasts.npz');checks=0;domains={};comparisons={}
    def close(a,b):
        nonlocal checks
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14);checks+=1
    def energy(x):
        flat=np.asarray(x,float).ravel()
        return float(np.dot(flat,flat)/len(flat))
    for path,digest in p['files'].items():
        assert sha(path)==digest;checks+=1
    assert len(rows)==20 and len({r['source'] for r in rows})==20;checks+=1
    assert sum(bool(r['response']) for r in rows)==15;checks+=1
    for row in rows:
        i=row['row'];sd=row['history_sd']
        base=forecasts[f'{i}_base_native'].astype(float)
        for method in ('lambda0','lambda1','pca'):
            rebuilt=forecasts[f'{i}_base_{method}'].astype(float)
            error=energy(rebuilt-base)
            close(error,row['baseline'][method]['mse'])
            close(error/sd**2,row['baseline'][method]['history_variance_nmse'])
            for direction,values in row['response'].items():
                actual=forecasts[f'{i}_{direction}_1_native'].astype(float)-forecasts[f'{i}_{direction}_-1_native'].astype(float)
                decoded=forecasts[f'{i}_{direction}_1_{method}'].astype(float)-forecasts[f'{i}_{direction}_-1_{method}'].astype(float)
                target=energy(actual);error=energy(decoded-actual);r=values[method]
                close(target,r['native_response_energy']);close(error,r['error_energy'])
                close(1e-12*sd**2,r['denominator_floor'])
                assert r['small_response']==(target<=1e-12*sd**2);checks+=1
                if r['nmse'] is not None:close(error/target,r['nmse'])
                close(error/max(target,1e-12*sd**2),r['floored_nmse'])
                close(error/sd**2,r['error_history_variance'])
    for key in forecasts.files:
        if key.endswith('_independent'):
            np.testing.assert_array_equal(forecasts[key],forecasts[key.removesuffix('_independent')]);checks+=1
    for method in ('lambda0','lambda1','pca'):
        close(np.mean([r['baseline'][method]['history_variance_nmse'] for r in rows]),summary[method]['baseline_mean'])
        for direction in ('sinusoid','noise'):
            scores=[r['response'][direction][method]['nmse'] for r in rows if direction in r['response'] and r['response'][direction][method]['nmse'] is not None]
            close(np.mean(scores),summary[method]['responses'][direction]['mean_source_nmse'])
        domains[method]={}
        for domain in sorted({r['domain'] for r in rows}):
            subset=[r for r in rows if r['domain']==domain]
            domains[method][domain]=dict(baseline_n_sources=len(subset),baseline_mean=float(np.mean([r['baseline'][method]['history_variance_nmse'] for r in subset])),
                responses={d:dict(n_sources=sum(d in r['response'] for r in subset),mean_source_nmse=float(np.mean([r['response'][d][method]['nmse'] for r in subset if d in r['response'] and r['response'][d][method]['nmse'] is not None]))) for d in ('sinusoid','noise')})
    for reference in ('lambda0','pca'):
        comparisons[reference]={}
        for metric in ('baseline','sinusoid','noise'):
            if metric=='baseline':
                values=[(r['baseline'][reference]['history_variance_nmse'],r['baseline']['lambda1']['history_variance_nmse']) for r in rows]
            else:
                values=[(r['response'][metric][reference]['nmse'],r['response'][metric]['lambda1']['nmse']) for r in rows if metric in r['response'] and r['response'][metric][reference]['nmse'] is not None and r['response'][metric]['lambda1']['nmse'] is not None]
            before,after=np.asarray(values).T
            comparisons[reference][metric]=dict(n_sources=len(values),lambda1_improved_sources=int(np.sum(after<before)),relative_mean_improvement=float(1-after.mean()/before.mean()),mean_paired_gain=float(np.mean(before-after)))
    record=dict(passed=True,checks=checks,independent_forecast_replays=9,by_domain=domains,lambda1_comparisons=comparisons,analysis_code_sha256=sha(__file__),scope='saved-only independent arithmetic; source means, no CIs')
    write(root/'arithmetic_audit.json',record)
    knowledge=read(root/'knowledge.json')
    knowledge['exact_results'].update(table=summary,execution=read(root/'execution.json'),independent_arithmetic='arithmetic_audit.json',arithmetic_checks=checks,lambda1_comparisons=comparisons)
    write(root/'knowledge.json',knowledge)
    print(dict(summary=summary,comparisons=comparisons,checks=checks))


if __name__=='__main__':
    audit(Path('runs/dictionary_response_native4965000'))
