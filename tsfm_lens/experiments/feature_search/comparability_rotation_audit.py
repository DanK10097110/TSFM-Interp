"""Diagnose actual-state basis sensitivity without inspecting TEST prediction risk.

Preserve the failed-run evidence and compare all fixed candidates, separating
coordinatewise standardization from global-scale PCA and CV tie selection.
"""
from experiments.feature_search.comparability_pilot import (ROOT, load_data, make_banks,
    write, sha, ENDPOINTS)
from experiments.feature_search.comparability_core import center_gram, pca_kernel, weights, ridge, ALPHAS
import os
import resource
import time
import numpy as np
from scipy.linalg import eigvalsh


def audit(out):
    """Measure real cached rotations, PCA boundary gaps and every candidate prediction."""
    start=time.monotonic()
    rows,x,hidden,targets,_,_=load_data()
    records=[]
    for cohort,keep in [('synthetic',lambda r:r['domain']=='synthetic'),('real_natural',lambda r:r['case']=='real'),('real_injected',lambda r:r['case']=='real_injected')]:
        ids=np.array([i for i,r in enumerate(rows) if keep(r)])
        rs=[rows[i] for i in ids];tr=np.array([i for i,r in enumerate(rs) if r['split']=='fit'])
        for a,b in [('bolt','timesfm'),('timesfm','bolt')]:
            h=hidden[a][ids];v=np.random.default_rng(4951001).normal(size=h.shape[1]);v/=np.linalg.norm(v)
            rotated=h-2*np.outer(h@v,v)
            k,kr=h@h.T,rotated@rotated.T
            banks,final=make_banks(x[ids],{'A_hidden':k},rs,tr)
            rb,rf=make_banks(x[ids],{'A_hidden':kr},rs,tr)
            checks=[]
            for name,train,bank,rotbank in [('final',tr,final,rf)]+[(f'fold{j}',t,bank,rr[2]) for j,((t,va,bank),rr) in enumerate(zip(banks,rb))]:
                w=weights(rs,train);sw=np.sqrt(w);cap=max(1,len({rs[i]['source'] for i in train})-1)
                kc=center_gram(k,train,w);scale=w@np.diag(kc)[train]
                spectrum=eigvalsh(kc[np.ix_(train,train)]*sw[:,None]*sw[None,:]/scale)[::-1]
                pred_error=0.;kernel_error=0.
                for candidate,kk in bank['input+A_hidden'].items():
                    rk=rotbank['input+A_hidden'][candidate]
                    kernel_error=max(kernel_error,float(abs(kk-rk).max()))
                    for alpha in ALPHAS:
                        for ep in ENDPOINTS:
                            yy=targets[b][ep][ids]
                            pp=ridge(kk,yy,train,np.arange(len(rs)),w,alpha)
                            rp=ridge(rk,yy,train,np.arange(len(rs)),w,alpha)
                            pred_error=max(pred_error,float(abs(pp-rp).max()))
                scaled=[]
                for hh in (h,rotated):
                    mean=w@hh[train];sd=np.sqrt(w@((hh[train]-mean)**2));sd[sd<1e-8]=1
                    z=(hh-mean)/sd
                    scaled.append(pca_kernel(z@z.T,train,w,min(8,cap)))
                bad_diff=float(np.max(abs(scaled[0]-scaled[1])))
                gaps={str(r):float(spectrum[min(r,cap)-1]-spectrum[min(r,cap)]) if min(r,cap)<len(spectrum) else None for r in (8,16)}
                checks.append(dict(part=name,n_train_sources=cap+1,rank_boundary_gaps=gaps,
                    eigenvalues=spectrum.tolist(),max_fixed_candidate_kernel_error=kernel_error,
                    max_fixed_candidate_prediction_error=pred_error,
                    deliberately_coordinatewise_scaled_hidden_kernel_error=bad_diff))
                assert pred_error<1e-7, 'Fixed-candidate algorithm fails unchanged rotation gate'
            records.append(dict(cohort=cohort,A=a,B=b,relative_raw_gram_error=float(abs(k-kr).max()/abs(k).max()),checks=checks))
    result=dict(records=records,source_sha256=sha(__file__),seconds=time.monotonic()-start,
        diagnosis='Hidden preprocessing uses FIT-weighted centering, global trace scale and dual PCA. No coordinatewise hidden StandardScaler. All fixed-candidate predictions must retain original1e-7 gate. Finite-rank PCA boundary gaps and spectra recorded. Deliberately coordinatewise-standardized hidden branch is a known-bad preprocessing control, never used in predictions.',
        failure='Initial actual-state real_natural Bolt->TimesFM log_failure selected pca16+pca8 before rotation and descriptor_ridge+pca8 after. CV losses differed at roundoff1e-17 while final predictions differed0.054029673. CV folds with <=16 rank can tie full and truncated input kernels, while final FIT has20 sources so truncation changes the final model. This is numerical instability of model selection, not substantive fixed-candidate coordinate sensitivity.',
        amendment='Deterministic first-candidate rule for CV scores within1e-12*max(1,abs(minimum)); unchanged rotation tolerance1e-7. No TEST-risk tuning.',
        failed_run_log_sha256=sha(out/'analysis.log'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'rotation_audit.json',result)
    print({k:result[k] for k in ['seconds','peak_rss_bytes']},flush=True)
    print('max fixed prediction error',max(c['max_fixed_candidate_prediction_error'] for r in records for c in r['checks']),flush=True)


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(3*2**30,3*2**30))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    audit(ROOT/'runs/comparability_pilot4951000')
