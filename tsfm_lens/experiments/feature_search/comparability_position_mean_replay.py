"""Independent source-first and float64 reconstruction replay of fixed mean audit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import math
import resource
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/comparability_position_mean20261116'
SOURCE=ROOT/'runs/selective_replication4942000'
DICTIONARY=ROOT/'runs/dictionary_pilot4951000/recovery_exit_monitor'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,v):
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,indent=2,allow_nan=False)


def scalar_mean(v):
    a=np.asarray(v).ravel()
    return math.fsum(map(float,a))/len(a)


def source_average(values,rows,ids):
    sources=sorted({rows[i]['source'] for i in ids})
    return {s:scalar_mean([values[i] for i in ids if rows[i]['source']==s]) for s in sources}


def run():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    result=read(OUT/'results.json');metadata=read(SOURCE/'rows.json')
    fit=[i for i,r in enumerate(metadata) if r['split']=='fit']
    test=[i for i,r in enumerate(metadata) if r['split']=='test'];rows=[metadata[i] for i in test]
    fit_sources=sorted({metadata[i]['source'] for i in fit})
    audit=dict(passed=True,models={},metric_comparisons=0,max_metric_relative_error=0.,
        method='No production fit/weight/scoring/PCA functions imported. Source-first FIT means; float64 coefficient projection with independent argpartition selection; compensated scalar arithmetic from saved reconstructions.',
        pca_reconstruction_tolerance_in_fit_scale=1e-4,metric_relative_tolerance=2e-12)
    for model in ('bolt','timesfm'):
        with np.load(SOURCE/f'{model}_states.npz') as cache:
            h=np.stack([cache[f'{i}_full'][0] for i in range(len(metadata))]).astype(float)
        with np.load(OUT/f'{model}_saved.npz') as f:saved={k:f[k] for k in f.files}
        per_source=[np.mean(h[[i for i in fit if metadata[i]['source']==s]],axis=0) for s in fit_sources]
        position=np.mean(per_source,axis=0);global_mean=position.mean(axis=0)
        np.testing.assert_allclose(position,saved['position_mean'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(global_mean,saved['global_mean'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(saved['test_row_ids'],test)
        ma=dict(fit_position_mean_max_error=float(np.max(abs(position-saved['position_mean']))),pca={})
        for k in (8,16):
            path=DICTIONARY/f'fit_{model}_cut0_pca_k{k}_s4951000/dictionary.npz'
            with np.load(path) as f:artifact={key:f[key] for key in f.files}
            x=(h[test]-artifact['center'].astype(float))/float(artifact['scale'])
            coefficient=x@artifact['basis'].astype(float)
            order=np.argpartition(abs(coefficient),-k,axis=-1)[...,-k:]
            sparse=np.zeros_like(coefficient)
            np.put_along_axis(sparse,order,np.take_along_axis(coefficient,order,axis=-1),axis=-1)
            rec=(sparse@artifact['basis'].astype(float).T)*float(artifact['scale'])+artifact['center']
            error=float(np.max(abs(rec-saved[f'pca_k{k}']))/float(artifact['scale']))
            assert error<1e-4,(model,k,error)
            ma['pca'][f'k{k}']=dict(max_absolute_reconstruction_error_in_fit_scale=error)
        for domain,cell in result['results'][model].items():
            ids=[i for i,r in enumerate(rows) if domain=='ALL' or r['domain']==domain]
            sources=sorted({rows[i]['source'] for i in ids})
            center=np.mean([np.mean(h[[test[i] for i in ids if rows[i]['source']==s]],axis=(0,1)) for s in sources],axis=0)
            varrows=np.array([scalar_mean((value-center)**2) for value in h[test]])
            variance=scalar_mean(list(source_average(varrows,rows,ids).values()))
            risks={}
            for arm in cell['mse']:
                loss=np.array([scalar_mean((value-prediction)**2) for value,prediction in zip(h[test],saved[arm])])
                per=source_average(loss,rows,ids);risks[arm]=scalar_mean(list(per.values()))
                assert len(per)==cell['n_sources']
                for s,v in per.items():
                    assert abs(v-cell['per_source_mse'][arm][s])<=2e-12*max(1,abs(v))
            comparisons=[(variance,cell['test_variance_about_domain_test_global_mean'])]
            for arm,value in risks.items():
                comparisons += [(value,cell['mse'][arm]),(value/variance,cell['mse_over_test_variance'][arm]),
                    (value/risks['global_fit_mean'],cell['mse_over_global_fit_mean_error'][arm]),
                    (1-value/risks['global_fit_mean'],cell['reduction_over_global_fit_mean'][arm])]
            for actual,stored in comparisons:
                relative=abs(actual-stored)/max(1,abs(actual))
                assert relative<=2e-12,(model,domain,actual,stored)
                audit['metric_comparisons']+=1;audit['max_metric_relative_error']=max(audit['max_metric_relative_error'],relative)
        audit['models'][model]=ma
    write(OUT/'arithmetic_replay.json',audit)
    protocol=read(OUT/'protocol.json')
    knowledge=dict(question=protocol['question'],design=protocol,status='complete_saved_only',exact_findings=result,
        controls=dict(validation=read(OUT/'validation.json'),independent_replay=audit),
        limitations=['Position-conditioned means may encode shared content, normalization or token roles; no causal position-embedding attribution.',
            'Static means respond identically zero to fixed-shape input perturbations; global compression is distinct from input-information or weak-response preservation.',
            'Four independent TEST sources per real domain; sixteen synthetic. Source-weighted global score gives synthetic half the mass.',
            'TEST variance is a scoring denominator only; negative reductions on individual domains are possible.',
            'Raw hidden-state MSE units are not comparable between models. PCA controls use input-dependent sparse coefficients and far more stored parameters.',
            'Post-hoc descriptive analysis on reused cohort, without confidence intervals, tuning or new model inference.'],
        evidence_class=protocol['evidence_class'],next_decision='Use this baseline to qualify global compression claims; it neither proves nor disproves interpretability. No native or dictionary-training expansion.',
        provenance=dict(protocol_sha256=sha(OUT/'protocol.json'),results_sha256=sha(OUT/'results.json'),replay_code_sha256=sha(Path(__file__))))
    write(OUT/'knowledge.json',knowledge)
    lines=['# Saved position-conditioned mean audit','',
        'Fixed source-weighted FIT means versus frozen PCA reconstructions; no dictionary training or native inference.',
        'Ratios below use global FIT mean TEST error as the common denominator. Source means precede aggregation. Lower is better.','',
        '| Model | Domain | TEST sources | Position mean / global mean error | PCA8 / global | PCA16 / global |',
        '|---|---|---:|---:|---:|---:|']
    for model,cells in result['results'].items():
        for domain,cell in cells.items():
            ratios=cell['mse_over_global_fit_mean_error']
            lines.append(f"| {model} | {domain} | {cell['n_sources']} | {ratios['position_conditioned_fit_mean']:.9g} | {ratios['pca_k8']:.9g} | {ratios['pca_k16']:.9g} |")
    lines += ['', 'Exact raw MSE, held-out variance, both normalized ratios, per-source scores and parameter counts are in results.json.',
        '', 'Position-conditioned means cannot respond to input changes at fixed token geometry. Their success identifies static shared structure, which may include common content; it is not causal positional-embedding attribution. PCA coefficients remain input-dependent. This qualifies global compression rather than disproving interpretability.',
        '',f"Validation:5 clean tests,5/5 isolated planted regressions detected; independent replay checked{audit['metric_comparisons']} scalar metrics plus per-source losses and reconstructions. Maximum metric relative error{audit['max_metric_relative_error']!r}."]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'completion_manifest.json',dict(status='complete_saved_only',hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(dict(audit=audit,results=result['results'],parameters=result['parameter_counts']),indent=2))


if __name__=='__main__':
    run()
