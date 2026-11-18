"""Saved-array-only decomposition of a closed native response mean reversal.

Source means remain primary. Influence deletions, pooled energies and hidden/
output associations are descriptive diagnostics, never selections or tuning.
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.feature_search.dictionary_response_poc import read,write,sha


def decompose(actual,decoded):
    """Orthogonal projection of OUTPUT ERROR onto actual native response."""
    import numpy as np
    a=np.asarray(actual,float).ravel();b=np.asarray(decoded,float).ravel();e=b-a
    target=float(a@a/len(a));error=float(e@e/len(a))
    if target<=0:
        raise ValueError('Zero native response; decomposition not identifiable')
    alpha=float(e@a/(a@a));parallel=alpha*a;orth=e-parallel
    return dict(native_energy=target,error_energy=error,nmse=error/target,
        error_parallel_energy=float(parallel@parallel/len(a)),error_orthogonal_energy=float(orth@orth/len(a)),
        error_parallel_signed_gain=alpha,decoded_parallel_gain=1+alpha,
        parallel_orthogonal_inner_product=float(parallel@orth/len(a)),
        native_error_inner_product=float(a@e/len(a)),decoded_energy=float(b@b/len(a)))


def influence(rows):
    """Predeclared same-source removal: largest positive lambda1-lambda0 NMSE."""
    import numpy as np
    order=sorted(rows,key=lambda r:(-(r['lambda1_nmse']-r['lambda0_nmse']),r['source']))
    total_positive=sum(max(0,r['lambda1_nmse']-r['lambda0_nmse']) for r in rows)
    output=[]
    for count in (0,1,2):
        removed=order[:count];kept=[r for r in rows if r['source'] not in {s['source'] for s in removed}]
        mean0=float(np.mean([r['lambda0_nmse'] for r in kept]));mean1=float(np.mean([r['lambda1_nmse'] for r in kept]))
        output.append(dict(removed_count=count,removed_sources=[r['source'] for r in removed],n_sources=len(kept),lambda0_mean=mean0,lambda1_mean=mean1,
            lambda1_minus_lambda0=mean1-mean0,removed_positive_deterioration_share=sum(max(0,r['lambda1_nmse']-r['lambda0_nmse']) for r in removed)/total_positive if total_positive else None,
            status='PRIMARY_ALL_SOURCES' if count==0 else 'DESCRIPTIVE_SENSITIVITY_NOT_REPLACEMENT'))
    return output


def association(x,y):
    """Descriptive Pearson and average-tie-rank correlation, no inference."""
    import numpy as np
    def ranks(v):
        order=np.argsort(v,kind='stable');result=np.empty(len(v),float)
        for value in np.unique(v):
            pos=np.flatnonzero(np.asarray(v)[order]==value);result[order[pos]]=float(pos.mean())
        return result
    def corr(a,b):
        return None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
    return dict(n_sources=len(x),pearson=corr(x,y),spearman=corr(ranks(x),ranks(y)))


def aggregate(rows):
    """Retain source-equal means/medians alongside both pooled-energy variants."""
    import numpy as np
    return dict(n_sources=len(rows),mean_source_nmse=float(np.mean([r['nmse'] for r in rows])),median_source_nmse=float(np.median([r['nmse'] for r in rows])),
        pooled_raw_energy_ratio=sum(r['error_energy'] for r in rows)/sum(r['native_energy'] for r in rows),
        pooled_history_normalized_energy_ratio=sum(r['error_energy_history_variance'] for r in rows)/sum(r['native_energy_history_variance'] for r in rows),
        means={key:float(np.mean([r[key] for r in rows])) for key in ('native_energy','error_energy','native_energy_history_variance','error_energy_history_variance','error_parallel_energy','error_orthogonal_energy','parallel_nmse','orthogonal_nmse')},
        median_native_energy_history_variance=float(np.median([r['native_energy_history_variance'] for r in rows])),
        parallel_error_share=sum(r['error_parallel_energy'] for r in rows)/sum(r['error_energy'] for r in rows))


def prepare(out,source):
    """Freeze diagnostic ranking, projection, aggregation and inputs before reading arrays."""
    files=[source/n for n in ('forecasts.npz','states.npz','results.json','summary.json','protocol.json','closure.json')]
    files += [Path(__file__),Path('tests/test_dictionary_response_influence.py'),Path('experiments/feature_search/dictionary_response_poc.py')]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',dict(source=str(source.resolve()),files={str(p.resolve()):sha(p) for p in files},threads=1,seconds=30,
        methods=['lambda0','lambda1','pca'],directions=['sinusoid','noise'],units='15 independent sources per direction,3/domain; signs and methods paired within source',
        influence='For each direction, rank sources descending lambda1_nmse-lambda0_nmse (ties source ID ascending). Delete top1 and top2 from BOTH arms only for prespecified descriptive sensitivity; all15 mean stays primary.',
        decomposition='e=decoded forecast difference-native forecast difference; e_parallel=dot(e,a)/dot(a,a)*a; e_orth=e-e_parallel. Audit E(e)=Eparallel+Eorth and zero cross term; Edecoded=Enative+2mean(a*e)+Eerror.',
        denominator_diagnostic='Raw and history-variance-normalized energies; same source denominator across methods. Rank native normalized energy and paired absolute error-energy change; retain every source. No denominator-based filtering.',
        hidden='Use saved ACTUAL singleton native and decoded first-block states. Compare hidden response NMSE/error energy with output error across sources, plus paired arm differences; descriptive correlations only, no Jacobian or causal inference.',
        aggregation='source mean primary; median, raw pooled-energy ratio and history-normalized pooled-energy ratio secondary. No CI, fitting, selection or native calls.',
        limits=['single seed and previously inspected15 sources','top-removal results are influence diagnostics, not an improved result','finite response differences do not estimate a downstream Jacobian'],closure='objective branch remains CLOSED'))
    write(out/'knowledge.json',dict(question='What drives the mean native-response reversal despite most-source improvement?',status='frozen_ready',exact_results={'protocol':'protocol.json'},controls=['saved arrays only; fixed ranking before diagnostic'],limitations=read(out/'protocol.json')['limits'],next_step='Execute one bounded saved-only diagnostic and close.'))


def worker(out):
    """No model imports, forwards or fits; exact stored-array algebra only."""
    import numpy as np
    p=read(out/'protocol.json');source=Path(p['source'])
    for path,digest in p['files'].items():
        if sha(path)!=digest:raise ValueError('Changed frozen diagnostic input: '+path)
    old=read(source/'results.json');forecasts=np.load(source/'forecasts.npz');states=np.load(source/'states.npz')
    rows=[];checks=0;max_identity=0.;max_cross=0.
    def check(a,b):
        nonlocal checks
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=1e-13);checks+=1
    for r in old:
        i=r['row'];variance=r['history_sd']**2
        for direction in r['response']:
            plus=f'{i}_{direction}_1';minus=f'{i}_{direction}_-1'
            actual=forecasts[plus+'_native'].astype(float)-forecasts[minus+'_native'].astype(float)
            hidden=states[plus].astype(float)-states[minus].astype(float)
            for method in p['methods']:
                decoded=forecasts[plus+'_'+method].astype(float)-forecasts[minus+'_'+method].astype(float)
                values=decompose(actual,decoded)
                check(values['error_energy'],values['error_parallel_energy']+values['error_orthogonal_energy'])
                check(values['parallel_orthogonal_inner_product'],0.)
                check(values['decoded_energy'],values['native_energy']+2*values['native_error_inner_product']+values['error_energy'])
                check(values['native_energy'],r['response'][direction][method]['native_response_energy'])
                check(values['error_energy'],r['response'][direction][method]['error_energy'])
                check(values['nmse'],r['response'][direction][method]['nmse'])
                max_identity=max(max_identity,abs(values['error_energy']-values['error_parallel_energy']-values['error_orthogonal_energy']))
                max_cross=max(max_cross,abs(values['parallel_orthogonal_inner_product']))
                rebuilt=states[plus+'_'+method].astype(float)-states[minus+'_'+method].astype(float)
                he=float(np.mean((rebuilt-hidden)**2));ht=float(np.mean(hidden**2))
                if ht<=0:raise ValueError('Zero hidden response energy')
                row=dict(row=i,source=r['source'],domain=r['domain'],direction=direction,method=method,history_variance=variance,**values,
                    native_energy_history_variance=values['native_energy']/variance,error_energy_history_variance=values['error_energy']/variance,
                    parallel_nmse=values['error_parallel_energy']/values['native_energy'],orthogonal_nmse=values['error_orthogonal_energy']/values['native_energy'],
                    denominator_floor=1e-12*variance,denominator_floor_multiple=values['native_energy']/(1e-12*variance),
                    hidden_native_response_energy=ht,hidden_response_error_energy=he,hidden_response_nmse=he/ht)
                rows.append(row)
    tables=[];correlations=[];paired=[];influences={}
    for direction in p['directions']:
        for method in p['methods']:
            own=[r for r in rows if r['direction']==direction and r['method']==method]
            tables.append(dict(direction=direction,method=method,domain='ALL',**aggregate(own)))
            for domain in sorted({r['domain'] for r in own}):
                tables.append(dict(direction=direction,method=method,domain=domain,**aggregate([r for r in own if r['domain']==domain])))
            correlations.append(dict(direction=direction,method=method,
                hidden_vs_output_nmse=association([r['hidden_response_nmse'] for r in own],[r['nmse'] for r in own]),
                hidden_error_vs_history_normalized_output_error=association([r['hidden_response_error_energy'] for r in own],[r['error_energy_history_variance'] for r in own])))
        own0=sorted([r for r in rows if r['direction']==direction and r['method']=='lambda0'],key=lambda r:r['source'])
        own1=sorted([r for r in rows if r['direction']==direction and r['method']=='lambda1'],key=lambda r:r['source'])
        median_native=float(np.median([r['native_energy_history_variance'] for r in own0]));group=[]
        for a,b in zip(own0,own1):
            assert a['source']==b['source'];check(a['native_energy'],b['native_energy'])
            delta=b['error_energy']-a['error_energy'];nmse_delta=b['nmse']-a['nmse']
            check(delta/a['native_energy'],nmse_delta)
            item=dict(source=a['source'],domain=a['domain'],direction=direction,lambda0_nmse=a['nmse'],lambda1_nmse=b['nmse'],delta_nmse=nmse_delta,
                delta_error_energy=delta,delta_error_energy_history_variance=delta/a['history_variance'],
                native_energy=a['native_energy'],native_energy_history_variance=a['native_energy_history_variance'],native_energy_relative_to_direction_median=a['native_energy_history_variance']/median_native,
                lambda0_error_energy_history_variance=a['error_energy_history_variance'],lambda1_error_energy_history_variance=b['error_energy_history_variance'],
                lambda0_parallel_nmse=a['parallel_nmse'],lambda1_parallel_nmse=b['parallel_nmse'],lambda0_orthogonal_nmse=a['orthogonal_nmse'],lambda1_orthogonal_nmse=b['orthogonal_nmse'],
                delta_hidden_nmse=b['hidden_response_nmse']-a['hidden_response_nmse'],delta_hidden_error_energy=b['hidden_response_error_energy']-a['hidden_response_error_energy'])
            group.append(item)
        for item in group:
            item['native_energy_ascending_rank']=1+sum(v['native_energy_history_variance']<item['native_energy_history_variance'] for v in group)
            item['error_deterioration_descending_rank']=1+sum(v['delta_error_energy_history_variance']>item['delta_error_energy_history_variance'] for v in group)
        paired+=group;influences[direction]=influence(group)
        correlations.append(dict(direction=direction,method='paired_lambda1_minus_lambda0',
            hidden_vs_output_nmse=association([r['delta_hidden_nmse'] for r in group],[r['delta_nmse'] for r in group]),
            hidden_error_vs_history_normalized_output_error=association([r['delta_hidden_error_energy'] for r in group],[r['delta_error_energy_history_variance'] for r in group])))
    write(out/'source_table.json',rows)
    with (out/'source_table.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'domain_summary.json',tables);write(out/'paired_contributions.json',paired);write(out/'influence.json',influences);write(out/'hidden_output_associations.json',correlations)
    write(out/'energy_identity_audit.json',dict(passed=True,checks=checks,output_rows=len(rows),max_absolute_energy_closure=max_identity,max_absolute_parallel_orthogonal_inner_product=max_cross,
        identities=['Eerror=Eparallel+Eorthogonal','mean(parallel*orthogonal)=0','Edecoded=Enative+2mean(native*error)+Eerror','deltaNMSE=deltaErrorEnergy/sameNativeEnergy'],tolerance='FP64 rtol2e-12 atol1e-13; algebraically exact identities checked numerically'))
    write(out/'knowledge.json',dict(question='Do denominator weighting or actual output errors explain mean reversal?',status='complete_objective_closed',native_calls=0,fit_steps=0,
        exact_results=dict(source_table='source_table.csv',domain_summary='domain_summary.json',paired_contributions='paired_contributions.json',influence='influence.json',hidden_associations='hidden_output_associations.json',audit='energy_identity_audit.json',checks=checks),
        controls=['all saved sources retained in primary','fixed same-source top1/top2 removal rule','parallel/orthogonal and energy identities checked','actual singleton hidden/reconstructed states, no cached proxies'],limitations=p['limits'],next_step='Objective branch CLOSED; no optimization, fitting or native work.'))
    print(dict(checks=checks,rows=len(rows),influence=influences),flush=True)


def launch(out):
    """Single-thread external30s cap; never invokes a model worker."""
    start=time.monotonic();reason=None
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',DICTIONARY_SAVED_DIAGNOSTIC='1')
    with (out/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-m','experiments.feature_search.dictionary_response_influence','worker','--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-start>30:reason='30s cap';os.killpg(child.pid,9);child.wait();break
            time.sleep(.02)
    result=dict(returncode=child.returncode,seconds=time.monotonic()-start,reason=reason,threads=1,native_calls=0,fit_steps=0)
    write(out/'execution.json',result)
    if child.returncode:
        write(out/'knowledge.json',dict(question='Mean reversal saved-only diagnostic',status='failed_closed',exact_results=result,controls=['30s monitor'],limitations=['incomplete diagnostic'],next_step='Branch closed; no automatic retry.'))
    print(result)


def main():
    """Freeze before executing one saved-only diagnostic."""
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','worker'));parser.add_argument('--out',type=Path,default=Path('runs/dictionary_response_influence4965000'));a=parser.parse_args()
    if a.command=='prepare':prepare(a.out,Path('runs/dictionary_response_native4965000'))
    elif a.command=='run':launch(a.out)
    else:
        if os.environ.get('DICTIONARY_SAVED_DIAGNOSTIC')!='1':parser.error('Use bounded supervisor')
        worker(a.out)


if __name__=='__main__':main()
