"""Independent inverse/arithmetic replay and knowledge distillation for both followups.

Uses saved decoder outputs and separately written Fourier least-squares/inverse
arithmetic; does not import the experiment's projection, inverse or scoring code.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha


def coefficients(x, spans):
    """Independent direct least-squares projection with a separate DC column."""
    results=[]
    for columns in np.array_split(np.arange(128),spans):
        t=columns-128
        basis=np.asarray([np.sin(2*np.pi*t/24),np.cos(2*np.pi*t/24),
                          np.sin(2*np.pi*t/19.2),np.cos(2*np.pi*t/19.2),np.ones(len(t))]).T
        result=np.linalg.lstsq(basis,np.asarray(x,float)[:,columns].T,rcond=None)[0].T
        results.append(result[:,:4])
    return np.stack(results,1)


def verify_hashes(directory,protocol):
    """Verify immutable sources and all bound artifacts."""
    for name,digest in protocol['source_hashes'].items():
        if sha(name)!=digest:raise AssertionError(f'Frozen source changed: {name}')
    if 'prior_hashes' in protocol:
        for name,digest in protocol['prior_hashes'].items():
            if sha(Path(protocol['prior'])/name)!=digest:raise AssertionError(f'Prior artifact changed: {name}')
    for name,digest in protocol.get('hashes',{}).items():
        if sha(directory/name)!=digest:raise AssertionError(f'Bound artifact changed: {name}')


def audit_normalization(out):
    """Replay each inverse and physical error, including every source-held-out fold."""
    p=read(out/'protocol.json');verify_hashes(out,p);prior=Path(p['prior'])
    raw=np.load(prior/'inputs.npy',mmap_mode='r');rows=read(prior/'decoder_rows.json')
    background=np.asarray([raw[r['base']] for r in rows],float)
    query=np.asarray([raw[r['index']] for r in rows],float)
    truth=coefficients(query-background,1)[:,0]
    metadata=np.load(out/'normalization_metadata.npz');results=read(out/'results.json')
    max_inverse=0.;max_score=0.;max_target=0.;checks=0
    for row in results['rows']:
        key='_'.join(row[k] for k in ('model','policy','route','feature','cut'))
        saved=np.load(out/f'{key}.npz');pred=saved['prediction'];route=row['route']
        prefix=f"{row['model']}_{row['policy']}"
        sq=metadata[prefix+'_query_sigma'];sb=metadata[prefix+'_base_sigma'];spans=sq.shape[1]
        if route.startswith('physical'):
            inverted=pred;target=truth
        elif route=='normalized_pair':
            paired=pred.reshape(len(rows),2,spans,4)
            inverted=np.mean(paired[:,0]*sq[:,:,None]-paired[:,1]*sb[:,:,None],axis=1)
            target=np.stack([coefficients(metadata[prefix+'_normalized_query'],spans),
                             coefficients(metadata[prefix+'_normalized_base'],spans)],axis=1).reshape(len(rows),-1)
        else:
            normalized=pred.reshape(len(rows),spans,4)
            cb=coefficients(background,spans)
            inverted=(normalized*sq[:,:,None]+cb*(sq/sb-1)[:,:,None]).mean(1)
            target=coefficients(metadata[prefix+'_normalized_query']-metadata[prefix+'_normalized_base'],spans).reshape(len(rows),-1)
        inverse_error=float(np.max(abs(inverted-saved['physical_prediction'])))
        target_error=float(np.max(abs(target-saved['target'])))
        max_inverse=max(max_inverse,inverse_error);max_target=max(max_target,target_error)
        np.testing.assert_allclose(inverted,saved['physical_prediction'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(target,saved['target'],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(truth,saved['physical_truth'],rtol=1e-11,atol=1e-11)
        groups=saved['source_ids'];mean=np.zeros_like(truth)
        for group in set(groups):mean[groups==group]=truth[groups!=group].mean(0)
        denominator=float(np.mean((truth-mean)**2))
        for label,name in [('physical','physical_prediction'),('shuffled','physical_shuffled'),
                           ('inverse_mean_control','physical_mean_control'),('inverse_zero_control','physical_zero_control')]:
            error=(saved[name]-truth)**2
            value=float(error.mean()/denominator)
            max_score=max(max_score,abs(value-row[label]['nmse']))
            np.testing.assert_allclose(value,row[label]['nmse'],rtol=1e-11,atol=1e-11)
            for group in set(groups):
                np.testing.assert_allclose(float(error[groups==group].mean()),row[label]['source_mse'][str(group)],rtol=1e-11,atol=1e-11)
            checks+=1+len(set(groups))
        for fold in row['folds']:
            assert fold['source_id'] not in fold['train_sources']
            assert len(fold['train_sources'])==7
            checks+=2
        checks+=3
    summary=[]
    def get(model,route,feature,cut):
        """Select a prespecified native first-block or matched control result."""
        return next(r for r in results['rows'] if r['model']==model and r['policy']=='native' and
                    r['route']==route and r['feature']==feature and r['cut']==cut)
    for model in ('bolt','timesfm'):
        direct=get(model,'physical_delta','hidden','0');corrected=get(model,'normalized_delta','hidden','0')
        summary.append(dict(model=model,direct_physical_nmse=direct['physical']['nmse'],
            direct_physical_hidden_metadata_nmse=get(model,'physical_delta','hidden_metadata','0')['physical']['nmse'],
            normalized_delta_restored_nmse=corrected['physical']['nmse'],
            normalized_delta_input_nmse=get(model,'normalized_delta','input','input')['physical']['nmse'],
            normalized_delta_metadata_only_nmse=get(model,'normalized_delta','metadata_only','metadata')['physical']['nmse'],
            normalized_delta_inverse_mean_nmse=corrected['inverse_mean_control']['nmse'],
            normalized_delta_inverse_zero_nmse=corrected['inverse_zero_control']['nmse'],
            normalized_delta_shuffled_nmse=corrected['shuffled']['nmse'],
            paired_endpoint_hidden_nmse=get(model,'normalized_pair','hidden','0')['physical']['nmse'],
            paired_endpoint_input_nmse=get(model,'normalized_pair','input','input')['physical']['nmse'],
            improved_source_count=sum(corrected['physical']['source_mse'][g]<direct['physical']['source_mse'][g] for g in direct['physical']['source_mse']),
            sources=8))
    audit=dict(passed=True,decoder_fits=len(results['rows']),checks=checks,max_inverse_discrepancy=max_inverse,
               max_target_discrepancy=max_target,max_nmse_discrepancy=max_score,source_hashes_verified=True,
               audit_source_sha256=sha(__file__))
    write(out/'audit.json',audit)
    knowledge=read(out/'knowledge.json')
    knowledge.update(distilled_findings=summary,
        interpretation='Omitted native normalization is a substantial measurement confound in direct physical-coefficient decoding. Corrected decoding supports accessibility with specified observation/inverse side information; it does not demonstrate hidden-only physical information or causal feature semantics.',
        feature_label_note="The result feature='hidden' describes the ridge covariates only. For normalized_delta, physical restoration ALSO uses observed background Fourier coefficients and actual endpoint native scales. These are history-only side information, supplied identically to every input/metadata/mean/zero/shuffled control. No generating coefficients, amplitudes, phases or source IDs are covariates.",
        endpoint_qualification='Paired normalized endpoints avoid the background-coefficient correction, but corrected hidden and matched-input errors are similar. No blanket hidden-over-input dominance is claimed; direct physical input differences remain a much stronger accessible baseline.',
        independent_audit=audit,next_decision='Coordinator may ledger audited findings. Fresh32-source crossed-phase design is prepared; HOLD native capture while dictionary/comparability own slots.')
    write(out/'knowledge.json',knowledge)
    print(dict(normalization_audit=audit,summary=summary),flush=True)


def audit_crossed(out):
    """Independently verify every quantized radius and the full source-phase membership."""
    p=read(out/'protocol.json');verify_hashes(out,p);inputs=np.load(out/'inputs.npy',mmap_mode='r')
    paths=read(out/'paths.json');pairs=read(out/'interactions.json');sources=read(out/'sources.json')
    assert len(sources)==32 and [r['seed'] for r in sources]==list(range(4954000,4954032))
    max_radius_error=0.;max_interaction_error=0.
    for source in sources:
        own=[r for r in paths if r['source']==source['source']]
        assert len(own)==144 and set(r['phase_id'] for r in own)==set(range(4))
        assert len([r for r in pairs if r['source']==source['source']])==48
    for r in paths:
        minus,plus=[inputs[r[k]].astype(float) for k in ('minus','plus')]
        radius=float(np.linalg.norm(plus-minus)/np.sqrt(128)/2/r['input_scale'])
        error=abs(radius-r['radius']);max_radius_error=max(max_radius_error,error);assert error<1e-7
        np.testing.assert_array_equal(inputs[r['midpoint']],((plus+minus)/2).astype(np.float32))
    for r in pairs:
        base=inputs[r['base']].astype(float)
        for k in ('a','b'):
            actual=float(np.linalg.norm(inputs[r[k]].astype(float)-base)/np.sqrt(128)/r['input_scale'])
            error=abs(actual-r['radius']);max_interaction_error=max(max_interaction_error,error);assert error<1e-7
    assert p['capture_hold'] and not list(out.glob('*_forecast.npy')) and not list(out.glob('*_hidden_*.npy'))
    record=dict(passed=True,paths_verified=len(paths),interaction_quartets_verified=len(pairs),sources=32,
                max_radius_error=max_radius_error,max_interaction_radius_error=max_interaction_error,
                capture_started=False,source_hashes_verified=True,audit_source_sha256=sha(__file__))
    write(out/'audit.json',record);knowledge=read(out/'knowledge.json');knowledge['independent_audit']=record
    knowledge['next_decision']='HOLD native capture. Current native slots belong to dictionary/comparability; no native process has been started by this followup.'
    write(out/'knowledge.json',knowledge);print(dict(crossed_preparation_audit=record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--normalization',type=Path,required=True)
    parser.add_argument('--crossed',type=Path,required=True);args=parser.parse_args()
    audit_normalization(args.normalization);audit_crossed(args.crossed)
