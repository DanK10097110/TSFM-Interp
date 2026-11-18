"""Saved-state position-conditioned means versus frozen sparse PCA reconstruction.

No dictionary fitting or native inference. Position means can include shared
content and cannot establish causal positional-embedding attribution.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import resource
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/selective_replication4942000'
DICTIONARY=ROOT/'runs/dictionary_pilot4951000/recovery_exit_monitor'
OUT=ROOT/'runs/comparability_position_mean20261116'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def fit_indices(rows):
    """Whole-source FIT/TEST membership; never allow source overlap."""
    fit={r['source'] for r in rows if r['split']=='fit'}
    test={r['source'] for r in rows if r['split']=='test'}
    if fit & test:
        raise ValueError('Source overlap')
    return np.array([i for i,r in enumerate(rows) if r['split']=='fit'])


def weights(rows):
    names=[r['source'] for r in rows]
    counts={s:names.count(s) for s in set(names)}
    return np.array([1/len(counts)/counts[s] for s in names])


def fit_means(states,rows):
    """Equal source mass then equal case/token mass; only FIT can set means."""
    ids=fit_indices(rows)
    w=weights([rows[i] for i in ids])
    position=np.einsum('n,ntd->td',w,states[ids].astype(float))
    global_mean=position.mean(axis=0)
    return global_mean,position


def sparse_pca(states,artifact):
    """Use existing full PCA basis and token-adaptive magnitude TopK, no refit."""
    x=(states-artifact['center'])/float(artifact['scale'])
    coefficients=x@artifact['basis']
    order=np.argsort(-abs(coefficients),axis=-1,kind='stable')[...,:int(artifact['k'])]
    z=np.zeros_like(coefficients)
    np.put_along_axis(z,order,np.take_along_axis(coefficients,order,axis=-1),axis=-1)
    rec=z@artifact['basis'].T
    return rec*float(artifact['scale'])+artifact['center'],x,rec


def source_scores(values,rows):
    """Retain source losses rather than counting dependent tokens as samples."""
    return {s:float(np.mean([v for v,r in zip(values,rows) if r['source']==s]))
            for s in sorted({r['source'] for r in rows})}


def summarize(states,predictions,rows):
    """TEST variance is a scoring denominator, never a fitted predictor."""
    w=weights(rows)
    center=np.einsum('n,nd->d',w,states.astype(float).mean(1))
    variance=float(w@np.mean((states-center)**2,axis=(1,2)))
    losses={arm:source_scores(np.mean((states.astype(float)-prediction)**2,axis=(1,2)),rows)
            for arm,prediction in predictions.items()}
    means={arm:float(np.mean(list(v.values()))) for arm,v in losses.items()}
    baseline=means['global_fit_mean']
    return dict(n_sources=len({r['source'] for r in rows}),rows=len(rows),
        test_variance_about_domain_test_global_mean=variance,mse=means,
        mse_over_test_variance={a:v/variance for a,v in means.items()},
        mse_over_global_fit_mean_error={a:v/baseline for a,v in means.items()},
        reduction_over_global_fit_mean={a:1-v/baseline for a,v in means.items()},
        per_source_mse=losses)


def prepare():
    """Freeze questions, denominators and artifact bindings before scoring."""
    OUT.mkdir(exist_ok=False)
    files=[SOURCE/'rows.json',DICTIONARY/'protocol.json',Path(__file__),ROOT/'tests/test_comparability_position_mean.py']
    for model in ('bolt','timesfm'):
        files += [SOURCE/f'{model}_states.npz',SOURCE/f'{model}_worker.json']
        for k in (8,16):
            stage=DICTIONARY/f'fit_{model}_cut0_pca_k{k}_s4951000'
            files += [stage/'dictionary.npz',stage/'results.json']
    protocol=dict(question='How much held-out state variation does a fixed position-conditioned FIT mean explain, compared with global FIT mean and existing PCAk8/16?',
        evidence_class='Post-hoc descriptive saved-only compression diagnostic; no new confirmatory claim',
        source_cohort='Frozen32FIT/32TEST sources,112rows each; same block-output cut0 and full tokens as dictionary pilot.',
        fitting='Single pooled FIT global vector and token-position matrix; equal weight per source then case then token. No domain-specific fitting.',
        scoring='Per-model/domain and ALL source-weighted state MSE, normalized by domain TEST variance about its own global feature mean and separately by globalFITmean error. TEST means are denominators only.',
        interpretation='Position-conditioned mean, not causal positional-embedding attribution; shared structure may include common content. Fixed means have identically zero response to input perturbations at fixed shape.',
        parameters='Global mean D; position mean T*D; PCA stored D*D basis+D center+1 scale (k is fixed metadata), dynamic up-to-T*k coefficients and indices per input. PCA is full-basis token-adaptive TopK, not fixed rank-k projection.',
        resource_limits=dict(threads=1,address_space_bytes=3*1024**3,native_calls=0,dictionary_training=0),
        checks='Known-answer, TEST poison, unequal-source weighting, overlap, PCA selection tests; isolated mutation subprocesses before real scoring; independent saved reconstruction/source arithmetic replay.',
        files={str(p.relative_to(ROOT)):sha(p) for p in files})
    write(OUT/'protocol.json',protocol)
    write(OUT/'knowledge_prepared.json',dict(question=protocol['question'],design=protocol,status='prepared_unscored',next_decision='Run clean tests and isolated plants before scoring.'))


def score():
    """Read native caches and frozen PCA artifacts only, then retain predictions."""
    protocol=read(OUT/'protocol.json')
    assert all(sha(ROOT/p)==h for p,h in protocol['files'].items())
    assert read(OUT/'validation.json')['passed']
    rows=read(SOURCE/'rows.json');fit=fit_indices(rows)
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test']);testrows=[rows[i] for i in test]
    assert len(fit)==len(test)==112 and len({r['source'] for r in testrows})==32
    declared=read(DICTIONARY/'protocol.json')
    assert fit.tolist()==declared['fit_rows'] and test.tolist()==declared['test_rows']
    results={};gates={};counts={}
    for model in ('bolt','timesfm'):
        assert sha(SOURCE/f'{model}_states.npz')==declared['hashes'][f'{model}_states.npz']
        with np.load(SOURCE/f'{model}_states.npz') as archive:
            states=np.stack([archive[f'{i}_full'][0] for i in range(len(rows))])
        global_mean,position=fit_means(states,rows);nt,d=position.shape
        predictions=dict(global_fit_mean=np.broadcast_to(global_mean,states[test].shape),
                         position_conditioned_fit_mean=np.broadcast_to(position,states[test].shape))
        gates[model]={}
        for k in (8,16):
            stage=DICTIONARY/f'fit_{model}_cut0_pca_k{k}_s4951000'
            with np.load(stage/'dictionary.npz') as data:artifact={a:data[a] for a in data.files}
            assert sha(stage/'dictionary.npz')==read(stage/'results.json')['dictionary_sha256']
            prediction,x,rec=sparse_pca(states[test],artifact);predictions[f'pca_k{k}']=prediction.astype(float)
            w=weights(testrows)
            scaled_mse=float(w@np.mean((rec-x)**2,axis=(1,2)))
            scaled_energy=float(w@np.mean(x*x,axis=(1,2)))
            nmse=scaled_mse/scaled_energy
            recorded=read(stage/'results.json')['metrics']['test']['nmse']
            assert abs(nmse-recorded)<2e-6
            np.testing.assert_allclose(global_mean,artifact['center'],rtol=1e-6,atol=1e-6)
            gates[model][f'pca_k{k}']=dict(replayed_native_scaled_nmse=nmse,recorded_nmse=recorded,absolute_difference=abs(nmse-recorded))
        results[model]={}
        for domain in ['ALL']+sorted({r['domain'] for r in testrows}):
            ids=np.array([i for i,r in enumerate(testrows) if domain=='ALL' or r['domain']==domain])
            results[model][domain]=summarize(states[test][ids],{a:v[ids] for a,v in predictions.items()},[testrows[i] for i in ids])
        counts[model]=dict(tokens=nt,hidden_width=d,global_mean_parameters=d,position_mean_parameters=nt*d,
                           pca_numeric_parameters=d*d+d+1,pca_dynamic_coefficients_per_row={str(k):nt*k for k in (8,16)},
                           pca_dynamic_indices_per_row={str(k):nt*k for k in (8,16)},trainable_optimizer_parameters=0)
        np.savez_compressed(OUT/f'{model}_saved.npz',global_mean=global_mean,position_mean=position,test_row_ids=test,
                            **predictions)
    write(OUT/'results.json',dict(results=results,parameter_counts=counts,gates=gates))
    write(OUT/'knowledge_scored.json',dict(question=protocol['question'],design=protocol,status='scored_pending_independent_replay',exact_findings=results,controls=gates,limitations=[protocol['interpretation']],evidence_class=protocol['evidence_class'],next_decision='Independently replay means, reconstructions and source arithmetic; no native.'))


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','score'))
    {'prepare':prepare,'score':score}[parser.parse_args().stage]()
