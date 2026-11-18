"""Independent saved-forecast replay and portable Stage49 scientific report.

No native models or experimental analysis helpers are imported. Replay derives
all 22 contrasts from exact saved rows and verifies the existing results. Portable
rendering only displays recorded values; it does not create new scientific tests.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path):
    """Read an explicitly selected scientific artifact."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    """Hash exact bytes, without claiming a timestamp or weight identity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_followups(phase,recoverability,homogeneous):
    """Display complete registered decisions; partial/negative cases cannot pass."""
    models=('timesfm','bolt','chronos2');pairs=('p32_16','p40_20')
    expected={f'{a}-{b}/{p}' for a,b in [('timesfm','bolt'),('timesfm','chronos2'),('bolt','chronos2')] for p in pairs}
    if set(phase['comparisons'])!=expected:raise ValueError('Retain all six phase comparisons')
    comparisons={};complete=False
    for key,value in phase['comparisons'].items():
        names,pair=key.split('/');first,second=names.split('-')
        responses=[phase['responses'][m][pair] for m in (first,second)]
        stable=all(r['all_four_low_order_stable'] for r in responses)
        adequate=all(r['adequate_full_response_coverage'] for r in responses)
        status=('pilot_negative_closed' if not value['pilot_numeric_gate'] else 'new_native_control_failed' if not phase['new_native_controls_pass']
                else 'unresolved_or_aliased' if not stable else 'stable_low_order_partial_coverage' if not adequate else 'stable_and_adequate')
        if status!=value['status']:raise ValueError('Phase status disagrees with full stability/coverage gates')
        comparisons[key]=status;complete=complete or status=='stable_and_adequate'
    recovery={}
    for model in ('bolt','timesfm'):
        fresh=recoverability['models'][model]['datasets']['fresh']
        passed=fresh['output_readout']['mean']<=.01 and fresh['output_minus_head_excess']['mean']<=.01
        if passed!=recoverability['models'][model]['fresh_gate_pass']:raise ValueError('Recoverability point gate mismatch')
        recovery[model]=dict(fresh=fresh,passed=bool(passed))
    if not homogeneous['calibration']['passed']:raise ValueError('Homogeneous source replay did not pass calibration')
    shares={k:v['aggregate_signed_projection_share'] for k,v in homogeneous['attribution']['components'].items()}
    return dict(phase=dict(status='completed_partial_closed' if not complete else 'completed_full_response_gate',
        any_eligible_complete_response=bool(complete),comparisons=comparisons,
        response_matrix={m:{p:phase['responses'][m][p]['status'] for p in pairs} for m in models},
        hidden_capture_authorized=False),recoverability=recovery,
        homogeneous=dict(calibration_passed=homogeneous['calibration']['passed'],
            maximum_replay_RMS=homogeneous['calibration']['maximum_common_scale_waveform_RMS'],
            signed_projection_shares=shares,energies=homogeneous['mean_normalized_energies'],
            vector_closure=homogeneous['attribution']['vector_closure_max'],energy_closure=homogeneous['attribution']['energy_closure_max'],
            n_bundles=len(homogeneous['attribution']['native_normalized_energy'])),
        proposed_intervention=dict(status='not_run',scope='Anchored composition restoration is proposed; no intervention outcomes included or inferred.'))


def portable_assets(root):
    """Preserve original producer/figure bytes and package the linked phase report."""
    public=root/'examples/feature_search_stage49'
    names=['crossover_summary.png','crossover_summary.pdf','producer_results.json','protocol.json','producer_provenance.json']
    for directory in ('phase_orbit','restoration'):
        if (public/directory).is_dir():names += [str(p.relative_to(public)) for p in (public/directory).iterdir() if p.is_file()]
    assets={name:dict(sha256=sha(public/name),base64=base64.b64encode((public/name).read_bytes()).decode('ascii')) for name in names}
    audit=root/'tsfm_lens/runs/feature_search_stage49_restoration3700000_independent_audit'
    for public_name,source_name in [('restoration_audit.json','audit.json'),('restoration_exact_p.json','exact_p_verification.json'),('restoration_exceptions.json','descriptive_exceptions.json')]:
        if (audit/source_name).is_file():assets[public_name]=dict(sha256=sha(audit/source_name),base64=base64.b64encode((audit/source_name).read_bytes()).decode('ascii'))
    compact_audit=root/'tsfm_lens/runs/feature_search_stage49_compactness3600000_independent_audit/audit.json'
    if compact_audit.is_file():assets['compactness_audit.json']=dict(sha256=sha(compact_audit),base64=base64.b64encode(compact_audit.read_bytes()).decode('ascii'))
    return assets


def write_assets(data,out):
    for name,record in data['portable_assets'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe portable asset path')
        content=base64.b64decode(record['base64'],validate=True)
        if hashlib.sha256(content).hexdigest()!=record['sha256']:raise ValueError('Portable asset checksum mismatch')
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)


def summarize_restoration(original,corrected,fresh,validity):
    """Fresh selected-cut confirmation requires full64 and authoritative resources."""
    cap=int(1.5*2**30)
    if not original.get('resource_limit_exceeded') or corrected.get('effective_earliest_cut')!=15:
        raise ValueError('Retain failed original and qualified corrected discovery separately')
    equivalence=corrected['execution_equivalence']
    if not (equivalence['all_arrays_bitwise_equal'] and equivalence['array_count']==29 and equivalence['all_scientific_fields_exact'] and corrected['numerical_checks']['passed'] and corrected['peak_resident_bytes']<=cap):
        raise ValueError('Corrected discovery validation failed')
    if fresh['status']!='complete' or fresh['candidate_cut']!=15 or fresh['late_control_cut']!=20 or fresh['family_size']!=2:
        raise ValueError('Fresh restoration requires fixed15/full Holm2 completion')
    ids=fresh['per_seed']['seed_ids']
    if ids!=list(range(3700000,3700064)) or any(type(v) is not int for v in ids):raise ValueError('Fresh64 exact generating-seed identities required')
    records=validity['validity'];controller=fresh['controller_peak_resident_bytes']
    if not validity['all_four_valid'] or len(records)!=4 or not validity['models_exited_before_analysis']:
        raise ValueError('All four shards must pass before pooled inference')
    for i,r in enumerate(records):
        if not (r['valid'] and r['shard_index']==i and r['seed_ids']==list(range(3700000+16*i,3700016+16*i))
                and r['numerical_checks']['passed'] and r['primary15_random_geometry_pass'] and r['model_released_before_return']
                and r['peak_resident_bytes']+controller<=cap and r['wall_seconds']<=180):
            raise ValueError('Fresh shard validity/resource failure')
    if fresh['maximum_simultaneous_task_RSS']>cap or fresh['peak_resident_bytes']+controller>cap or fresh['total_wall_seconds']>180:
        raise ValueError('Fresh total resource failure overrides scientific outcome')
    claims=fresh['claims']
    if set(claims)!={'native_gain','random_advantage'}:raise ValueError('One complete Holm2 family required')
    per=fresh['per_seed'];vectors={k:np.asarray(per[k],float) for k in ('native_loss','component_loss','edited_loss','random_mean_loss','native_gain','random_advantage')}
    if any(v.shape!=(64,) or not np.isfinite(v).all() for v in vectors.values()):raise ValueError('All64 finite seed outcomes required')
    np.testing.assert_allclose(vectors['native_gain'],vectors['native_loss']-vectors['edited_loss'],atol=1e-12)
    np.testing.assert_allclose(vectors['random_advantage'],vectors['random_mean_loss']-vectors['edited_loss'],atol=1e-12)
    for name,key in [('native','native_loss'),('component_reference','component_loss'),('anchored','edited_loss'),('random_mean','random_mean_loss')]:
        np.testing.assert_allclose(fresh['mean_losses'][name],vectors[key].mean(),atol=1e-12)
    ps={};passed=True
    for key,minimum in [('native_gain',.1),('random_advantage',.05)]:
        v=vectors[key];positive=int((v>0).sum());negative=int((v<0).sum());n=positive+negative
        ps[key]=float(binomtest(positive,n,.5,alternative='two-sided').pvalue) if n else 1.
        r=claims[key]
        if r['positive']!=positive or r['negative']!=negative or r['n_seeds']!=64 or r['ties']!=64-n:raise ValueError('Fresh seed sign counts mismatch')
        np.testing.assert_allclose(r['mean'],v.mean(),atol=1e-12)
        passed=passed and positive>negative and v.mean()>=minimum
    running=0.
    for rank,(key,p) in enumerate(sorted(ps.items(),key=lambda row:row[1])):
        running=max(running,min(1.,(2-rank)*p));np.testing.assert_allclose(claims[key]['p_holm'],running,rtol=1e-12,atol=0)
        passed=passed and running<.05
    gap=float(np.mean(vectors['native_loss']-vectors['component_loss']))
    recovery=None if gap<.1 else float(vectors['native_gain'].mean()/gap)
    np.testing.assert_allclose(fresh['gates']['reference_gap'],gap,atol=1e-12)
    if recovery is not None:np.testing.assert_allclose(fresh['gates']['recovery_unclipped'],recovery,atol=1e-12)
    passed=bool(passed and gap>=.1 and recovery is not None and recovery>=.75 and fresh['gates']['mean_suffix_energy']<=.05)
    if passed!=fresh['qualified'] or passed!=fresh['gates']['passed']:raise ValueError('Fresh full qualification disagrees with point/sign/resource gates')
    return dict(status='completed_fresh_replication',qualified=passed,n_seeds=64,cut=15,claims=claims,mean_losses=fresh['mean_losses'],gates=fresh['gates'],
        controls=fresh['descriptive_controls'],original_resource_failure_retained=True,
        original_peak_bytes=original['peak_resident_bytes'],corrected_peak_bytes=corrected['peak_resident_bytes'],
        fresh_max_worker_peak=max(r['peak_resident_bytes'] for r in records),fresh_conservative_task_peak=max(r['peak_resident_bytes'] for r in records)+controller,
        fresh_max_simultaneous_RSS=fresh['maximum_simultaneous_task_RSS'],
        total_wall_seconds=fresh['total_wall_seconds'],interpretation='Off-manifold background-anchored suffix compatibility;not canonical feature identification,harmful-layer origin,or deployable repair')


def summarize_compactness(result,protocol,discovery):
    """A fixed reused16 screen cannot establish broad dimension impossibility."""
    if protocol['ranks']!=[1,2,4] or protocol['cut']!=15 or protocol['seed_ids']!=list(range(3600000,3600016)):
        raise ValueError('Preserve the frozen rank/seed/cut screen')
    if protocol['gates']!=dict(recovery=.8,mean_loss=.06,random_advantage=.05,mean_only_advantage=.05) or set(result['ranks'])!={'1','2','4'}:
        raise ValueError('Preserve all fixed compactness gates and ranks')
    if result['status']!='complete' or not result['execution_valid']:raise ValueError('Compactness execution is not valid')
    if result['total_wall_seconds']>180 or any(r['returncode']!=0 or r['conservative_peak_sum']>int(1.5*2**30) for r in result['execution_phases'].values()):
        raise ValueError('Compactness resource failure')
    full=float(np.mean(discovery['pilot']['descriptive']['15']['anchor_loss']));native=float(np.mean(discovery['pilot']['native_loss']))
    np.testing.assert_allclose(result['full_edit_mean_improvement'],native-full,atol=1e-12)
    ranks={};eligible=[]
    for rank in (1,2,4):
        value=result['ranks'][str(rank)];g=protocol['gates']
        np.testing.assert_allclose(value['recovery'],(native-value['mean_loss'])/(native-full),atol=1e-12)
        np.testing.assert_allclose(value['mean_only_advantage'],result['mean_only_mean_loss']-value['mean_loss'],atol=1e-12)
        passed=bool(value['valid_all16'] and value['recovery']>=g['recovery'] and value['mean_loss']<=g['mean_loss']
            and value['random_advantage']>=g['random_advantage'] and value['mean_only_advantage']>=g['mean_only_advantage'])
        if passed!=value['passed']:raise ValueError('Compactness fixed gate mismatch')
        if passed:eligible.append(rank)
        ranks[str(rank)]=value
    if result['smallest_passing_rank']!=(min(eligible) if eligible else None):raise ValueError('Compactness rank selection mismatch')
    return dict(status='closed_negative' if not eligible else 'exploratory_candidate_only',ranks=ranks,full_edit_mean_loss=full,
        native_mean_loss=native,mean_only_mean_loss=result['mean_only_mean_loss'],gates=protocol['gates'],
        scope='Prespecified uncentered SVD ranks1/2/4 on reused16,crossfit8/8;no p-values or fresh confirmation;does not disprove all low-dimensional/nonlinear mechanisms')


def bootstrap(values,config):
    """Independently replay the registered pointwise seed-mean interval."""
    draws=np.random.default_rng(config['bootstrap_seed']).integers(0,len(values),size=(config['n_boot'],len(values)))
    return np.quantile(np.asarray(values)[draws].mean(1),[.025,.975])


def replay(prepared,run):
    """Verify source IDs/hashes, direct losses, full Gram closure and all 22 tests."""
    frozen=read(prepared/'protocol.json')
    result=read(run/'results.json')
    config=frozen['config']
    assert result['protocol']==config
    assert (prepared/'protocol.json').read_bytes()==(run/'protocol.json').read_bytes()
    provenance=read(run/'provenance.json')
    saved=np.load(run/'seed_metrics.npz',allow_pickle=False)
    vectors,claims={},{}
    checks=0
    maximum_error=0.
    maximum_closure=0.
    for case,source in frozen['sources'].items():
        assert sha(prepared/f'{case}.npz')==source['corpus_sha256']
        assert sha(prepared/f'{case}.json')==source['metadata_sha256']
        with np.load(prepared/f'{case}.npz',allow_pickle=False) as corpus:
            data={k:corpus[k] for k in corpus.files}
        metadata=read(prepared/f'{case}.json')
        periods=source['periods']
        context=data['context'].shape[1]
        ids=np.asarray([f"{m['generator_seed']}:{periods[0]}:{periods[1]}:{context}:{m['condition']}" for m in metadata])
        np.testing.assert_array_equal(ids,data['row_ids'])
        assert len(set(ids))==len(ids)==config['n_seeds']*6
        lookup={(m['seed'],m['condition']):i for i,m in enumerate(metadata)}
        times=np.arange(config['horizon'])
        design=np.column_stack([np.ones(len(times))]+[fn(2*np.pi*times/p) for p in periods for fn in (np.sin,np.cos)])
        assert np.linalg.matrix_rank(design)==5 and np.linalg.cond(design)<1.72
        truth=data['clean_future']
        for index,m in enumerate(metadata):
            assert m['generator_seed']==config['seed_base']+m['seed']
            expected=m['offset']+data['component_future'][index].sum(0)
            np.testing.assert_allclose(truth[index],expected,atol=1e-12)
            if m['condition']=='background':
                assert np.count_nonzero(data['component_future'][index])==0
                np.testing.assert_allclose(data['context'][index],m['offset'],atol=1e-6)
        vectors[case]={}
        for model in config['models']:
            name=model['name']
            path=run/f'{name}_{case}.npz'
            assert sha(path)==provenance[name]['cases'][case]['forecast_sha256']
            with np.load(path,allow_pickle=False) as artifact:
                np.testing.assert_array_equal(artifact['row_ids'],ids)
                assert str(artifact['corpus_sha256'].item())==source['corpus_sha256']
                pred=artifact['prediction']
            assert pred.shape==truth.shape and np.isfinite(pred).all()
            values={key:[] for key in ('L','S','G','pure_phi','pure_psi','background','J','off_target_energy')}
            for seed in range(config['n_seeds']):
                ix={c:lookup[seed,c] for c in config['conditions']}
                m=metadata[ix['baseline']]
                energy=(m['a']**2+m['b']**2)/2
                losses={c:np.mean(np.square(pred[i]-truth[i]))/energy for c,i in ix.items()}
                composed=pred[ix['pure_phi']]+pred[ix['pure_psi']]-pred[ix['background']]
                residual=np.mean(np.square(composed-truth[ix['baseline']]))/energy
                values['L'].append(losses['baseline'])
                values['S'].append(residual)
                values['G'].append(losses['baseline']-residual)
                values['J'].append(losses['baseline']-(losses['pure_phi']+losses['pure_psi'])/2)
                for condition in ('pure_phi','pure_psi','background'):values[condition].append(losses[condition])
                derivative=np.asarray((pred[ix['plus']]-pred[ix['minus']])/(2*config['delta']),dtype=np.float64)
                coefficients=np.linalg.solve(design.T@design,design.T@derivative)
                pieces=[design[:,1:3]@coefficients[1:3],design[:,3:5]@coefficients[3:5],
                        np.full(len(times),coefficients[0]),derivative-design@coefficients]
                cross=sum(2*np.mean(a*b) for i,a in enumerate(pieces) for b in pieces[i+1:])
                closure=np.mean(np.square(derivative))-sum(np.mean(np.square(a)) for a in pieces)-cross
                maximum_closure=max(maximum_closure,abs(closure)/(m['b']**2/2))
                values['off_target_energy'].append(np.mean(np.square(pieces[0]))/(m['b']**2/2))
            values={key:np.asarray(value) for key,value in values.items()}
            vectors[case][name]=values
            for key,value in values.items():
                original=saved[f'{case}/{name}/{key}']
                maximum_error=max(maximum_error,float(np.max(abs(value-original))))
                np.testing.assert_allclose(value,original,atol=1e-10,rtol=1e-10)
                checks+=len(value)
            descriptive=result['descriptive'][case][name]
            ratio=values['S'].sum()/values['L'].sum()
            np.testing.assert_allclose(ratio,descriptive['residual_fraction']['value'],atol=1e-12)
            draws=np.random.default_rng(config['bootstrap_seed']).integers(0,config['n_seeds'],size=(config['n_boot'],config['n_seeds']))
            ratios=values['S'][draws].sum(1)/values['L'][draws].sum(1)
            np.testing.assert_allclose(np.quantile(ratios,[.025,.975]),[descriptive['residual_fraction'][k] for k in ('lo','hi')],atol=1e-12)
    for context in config['contexts']:
        first=vectors[f'p32_16_c{context}'];second=vectors[f'p40_20_c{context}']
        def gap(case,metric):return case['timesfm'][metric]-case['bolt'][metric]
        g32,g40=gap(first,'L'),gap(second,'L')
        c32,c40=gap(first,'G'),gap(second,'G')
        o32,o40=gap(first,'off_target_energy'),gap(second,'off_target_energy')
        differences=dict(g32=g32,g40=g40,g40_minus_g32=g40-g32,C32=c32,C40=c40,C40_minus_C32=c40-c32,
                         o32=o32,o40=o40,o40_minus_o32=o40-o32,G_timesfm32=first['timesfm']['G'],G_bolt40=second['bolt']['G'])
        for key,difference in differences.items():
            label=f'c{context}/{key}'
            positive=int(np.sum(difference>0));negative=int(np.sum(difference<0));n=positive+negative
            test=binomtest(positive,n,.5,alternative='two-sided')
            interval=test.proportion_ci(.95,method='exact')
            value=dict(value=positive/n,p=test.pvalue,positive=positive,negative=negative,ties=len(difference)-n,
                       lo=interval.low,hi=interval.high,mean=float(difference.mean()))
            stored=result['claims'][label]
            for field in ('value','p','lo','hi','positive','negative','ties'):
                np.testing.assert_allclose(value[field],stored[field],atol=1e-12)
            np.testing.assert_allclose(value['mean'],stored['mean']['value'],atol=1e-10)
            np.testing.assert_allclose(bootstrap(difference,config),[stored['mean'][k] for k in ('lo','hi')],atol=1e-10)
            claims[label]=value
    assert set(claims)==set(result['claims']) and len(claims)==22
    running=0.
    for rank,(key,value) in enumerate(sorted(claims.items(),key=lambda item:item[1]['p'])):
        running=max(running,min(1.,(len(claims)-rank)*value['p']))
        np.testing.assert_allclose(running,result['claims'][key]['p_holm'],atol=1e-15)
        stored=result['claims'][key]
        passed=running<config['alpha'] and abs(value['mean'])>=stored['minimum_absolute_mean'] and np.sign(value['mean'])==np.sign(value['value']-.5)
        assert bool(passed)==stored['passed']
    def flip(context):
        return all(result['claims'][f'c{context}/{key}']['passed'] and np.sign(claims[f'c{context}/{key}']['mean'])==sign
                   for key,sign in [('g32',1),('g40',-1),('g40_minus_g32',-1)])
    assert flip(128)==result['primary_ranking_flip'] is False
    assert flip(512)==result['sensitivity_ranking_flip'] is True
    assert maximum_closure<1e-10
    return dict(passed=True,claims_replayed=22,per_seed_values_checked=checks,
                maximum_per_seed_absolute_error=maximum_error,maximum_full_gram_energy_closure=maximum_closure,
                methods='Independent direct saved-forecast losses; Gram normal equations; scipy exact binomial/CP; sorted monotone Holm; paired bootstrap replay',
                source_sha256=sha(__file__),prepared_protocol_sha256=sha(prepared/'protocol.json'),results_sha256=sha(run/'results.json'))


def collect(root):
    """Select already reported numerical evidence, not new exploratory endpoints."""
    run=root/'tsfm_lens/runs/feature_search_stage49_crossover3400000'
    prepared=root/'tsfm_lens/runs/feature_search_stage49_crossover3400000_prepared'
    verification=replay(prepared,run)
    rbf_path=root/'tsfm_lens/runs/feature_search_20261111_stage49_rbf_challenge/results.json'
    clamp_path=root/'tsfm_lens/runs/stage49_timesfm_residual_update3100000_retry1/results.json'
    rbf,clamp=read(rbf_path),read(clamp_path)
    paths=dict(phase=root/'tsfm_lens/runs/feature_search_stage49_phase_alias3500000',
        recoverability=root/'tsfm_lens/runs/feature_search_stage49_forecast_recoverability',
        homogeneous=root/'tsfm_lens/runs/feature_search_stage49_homogeneous_pilot')
    followup_artifacts={key:dict(results=read(path/'results.json'),protocol=read(path/'protocol.json')) for key,path in paths.items()}
    recovery=followup_artifacts['recoverability']
    if recovery['results']['protocol_sha256']!=sha(paths['recoverability']/'protocol.json'):raise ValueError('Recoverability protocol binding mismatch')
    homogeneous=followup_artifacts['homogeneous']
    if homogeneous['results']['calibration']['maximum_common_scale_waveform_RMS']>homogeneous['protocol']['replay_normalized_waveform_RMS']:
        raise ValueError('Homogeneous replay exceeds its registered calibration bound')
    summary=summarize_followups(*(followup_artifacts[key]['results'] for key in ('phase','recoverability','homogeneous')))
    restoration_paths=dict(original=root/'tsfm_lens/runs/feature_search_stage49_restoration3600000',
        discovery=root/'tsfm_lens/runs/feature_search_stage49_restoration3600000_memorysafe',
        replication=root/'tsfm_lens/runs/feature_search_stage49_restoration3700000')
    restoration={key:dict(results=read(path/'results.json'),protocol=read(path/'protocol.json')) for key,path in restoration_paths.items()}
    fresh_path=restoration_paths['replication'];restoration['controller_validity']=read(fresh_path/'controller_validity.json')
    if restoration['replication']['results']['protocol_sha256']!=sha(fresh_path/'protocol.json'):raise ValueError('Fresh restoration protocol binding mismatch')
    summary['restoration']=summarize_restoration(restoration['original']['results'],restoration['discovery']['results'],restoration['replication']['results'],restoration['controller_validity'])
    summary['proposed_intervention']=dict(status='completed_fresh_replication',scope='Anchored restoration completed; future low-rank compactness screen remains not run in this snapshot.')
    audit_path=root/'tsfm_lens/runs/feature_search_stage49_restoration3700000_independent_audit'
    restoration['independent_audit']=read(audit_path/'audit.json');restoration['exact_p_verification']=read(audit_path/'exact_p_verification.json')
    if restoration['independent_audit']['status']!='independent_audit_passed' or not restoration['independent_audit']['formal_replication_gate_verified']:
        raise ValueError('Independent restoration audit has not passed')
    for name,value in restoration['independent_audit']['bindings'].items():
        if sha(fresh_path/name)!=value:raise ValueError('Independent audit producer binding mismatch')
    compact_path=root/'tsfm_lens/runs/feature_search_stage49_compactness3600000'
    compactness=dict(results=read(compact_path/'results.json'),protocol=read(compact_path/'protocol.json'))
    if compactness['results']['source_hashes']['protocol.json']!=sha(compact_path/'protocol.json'):raise ValueError('Compactness protocol binding mismatch')
    summary['compactness']=summarize_compactness(compactness['results'],compactness['protocol'],restoration['discovery']['results'])
    compact_audit=root/'tsfm_lens/runs/feature_search_stage49_compactness3600000_independent_audit/audit.json'
    compactness['independent_audit']=read(compact_audit)
    if compactness['independent_audit']['status']!='independent_audit_passed_screen_closed':raise ValueError('Compactness audit not passed')
    for name,value in compactness['independent_audit']['hashes'].items():
        if sha(compact_path/name)!=value:raise ValueError('Compactness audit producer binding mismatch')
    summary['proposed_intervention']=dict(status='completed_restoration_and_closed_compactness',scope='No higher ranks or fresh compactness inference authorized')
    sources={str(path.relative_to(root)):sha(path) for path in [run/'results.json',run/'protocol.json',run/'provenance.json',prepared/'protocol.json',rbf_path,clamp_path]}
    sources.update({str((path/name).relative_to(root)):sha(path/name) for path in paths.values() for name in ('results.json','protocol.json')})
    sources.update({str((path/name).relative_to(root)):sha(path/name) for path in restoration_paths.values() for name in ('results.json','protocol.json')})
    sources.update({str(path.relative_to(root)):sha(path) for path in [fresh_path/'controller_validity.json',audit_path/'audit.json',audit_path/'exact_p_verification.json']})
    sources.update({str((compact_path/name).relative_to(root)):sha(compact_path/name) for name in ('results.json','protocol.json')})
    sources[str(compact_audit.relative_to(root))]=sha(compact_audit)
    return dict(results=read(run/'results.json'),protocol=read(run/'protocol.json'),provenance=read(run/'provenance.json'),
        replay=verification,source_files=sources,phase_orbit=summary['phase'],followups=summary,
        followup_artifacts=followup_artifacts,restoration_evidence=restoration,compactness_evidence=compactness,portable_assets=portable_assets(root),
        negatives=dict(rbf=dict(input_loss=rbf['models']['input_rbf']['evaluation_prediction_mse'],
            joint_loss=rbf['models']['joint_rbf']['evaluation_prediction_mse'],
            primary_gain=rbf['contrasts']['joint_vs_input']['mean_prediction_loss_gain'],
            passed=rbf['contrasts']['joint_vs_input']['passes_frozen_robustness_gate']),
            residual_clamp=dict(sites=clamp['sites'],final_holm=clamp['holm8']['19_vs_native']['p_holm'],
                final_positive=clamp['holm8']['19_vs_native']['positive'],n_seeds=16)),
        scope='Display and independent verification of saved scientific results; no native inference or new scientific claim family')


def plot(data,out):
    """Draw four panels with primary/sensitivity and mean/sign distinctions visible."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=data['results'];desc=result['descriptive'];claims=result['claims']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13.4,9.8))
    colors=['#375f9d','#b4c8df','#c3d7be','#d88a3f']
    left=axes[0,0]
    labels=[]
    for group,(pair,model) in enumerate([(32,'timesfm'),(32,'bolt'),(40,'timesfm'),(40,'bolt')]):
        metrics=desc[f'p{pair}_{pair//2}_c128'][model]['metrics']
        for offset,key in enumerate(['L','pure_phi','pure_psi','S']):
            left.bar(group+(offset-1.5)*.19,metrics[key]['value'],.18,color=colors[offset],label={'L':'Native mixed','pure_phi':'Pure first','pure_psi':'Pure second','S':'Actual-background sum'}[key] if group==0 else None)
        labels.append(f'{"TF" if model=="timesfm" else "Bolt"}\n{pair}/{pair//2}')
    left.set_xticks(range(4),labels);left.set_yscale('log');left.set_ylim(3e-5,2.)
    left.set_ylabel('MSE / common theoretical mixed energy');left.set_title('A  Different failure patterns · primary context 128',loc='left',fontsize=11)
    left.legend(fontsize=8,ncol=2,loc='upper left');left.grid(axis='y',alpha=.18)
    ax=axes[0,1]
    for y,(pair,model) in enumerate([(32,'timesfm'),(32,'bolt'),(40,'timesfm'),(40,'bolt')]):
        value=desc[f'p{pair}_{pair//2}_c128'][model]['residual_fraction']
        color='#28659c' if model=='timesfm' else '#c46a28'
        ax.errorbar(value['value'],3-y,xerr=[[value['value']-value['lo']],[value['hi']-value['value']]],fmt='o',color=color,capsize=4)
        ax.annotate(f"{value['value']:.3f}",(value['value'],3-y),xytext=(6,6),textcoords='offset points',fontsize=9)
    ax.set_yticks(range(4),['Bolt 40/20','TF 40/20','Bolt 32/16','TF 32/16']);ax.axvline(1,color='#777',ls='--',lw=1)
    ax.set_xlim(-.05,1.1);ax.set_ylim(-.5,3.65);ax.set_xlabel('S/L: ratio of mean normalized losses (pointwise 95% CI)')
    ax.set_title('B  Remaining error after privileged component sum',loc='left',fontsize=11);ax.grid(axis='x',alpha=.18)
    ax=axes[1,0]
    keys=['g32','g40','g40_minus_g32','o32','o40','o40_minus_o32']
    for y,key in enumerate(keys):
        value=claims[f'c128/{key}'];q=value['value']
        ax.errorbar(q,5-y,xerr=[[q-value['lo']],[value['hi']-q]],fmt='o',color='#8b3b3b' if not value['passed'] else '#286d59',capsize=3)
        ax.text(1.07,5-y,f"{value['positive']}+/{value['negative']}−; Holm22={value['p_holm']:.2g}",fontsize=8,va='center')
    ax.axvline(.5,color='#777',ls='--',lw=1);ax.set_xlim(-.04,1.04);ax.set_ylim(-.7,5.65)
    ax.set_yticks(range(6),['o40−o32','o40','o32','g40−g32','g40','g32']);ax.set_xlabel('q = P(TF−Bolt contrast > 0 | non-ties); exact CP 95% CI')
    ax.set_title('C  Off-response reversal; accuracy-flip gate FAILED',loc='left',fontsize=11);ax.grid(axis='x',alpha=.18)
    ax=axes[1,1]
    for i,(label,context) in enumerate([('Primary128',128),('Sensitivity512',512)]):
        for j,key in enumerate(['G_timesfm32','G_bolt40']):
            value=claims[f'c{context}/{key}'];mean=value['mean'];x=i+(j-.5)*.22
            ax.errorbar(x,mean['value'],yerr=[[mean['value']-mean['lo']],[mean['hi']-mean['value']]],fmt='o',color=['#28659c','#c46a28'][j],capsize=4,label=['TF 32/16','Bolt 40/20'][j] if i==0 else None)
            ax.annotate(f"q={value['value']:.3f}\nHolm22={value['p_holm']:.2g}",(x,mean['value']),xytext=(8,0 if j==0 else 8),textcoords='offset points',fontsize=8)
    ax.axhline(0,color='#777',ls='--',lw=1);ax.set_xticks([0,1],['Primary context 128','Sensitivity context 512']);ax.set_xlim(-.4,1.65)
    ax.set_ylabel('G = native minus actual-sum normalized MSE');ax.set_title('D  Sensitivity cannot rescue primary decisions',loc='left',fontsize=11)
    ax.legend(fontsize=8,loc='upper right');ax.grid(axis='y',alpha=.18)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fresh 64-seed crossover replication · composition harm differs from pure-frequency weakness',fontsize=14,y=.99)
    fig.text(.02,.015,'TF = TimesFM; g = normalized native-loss gap; o = off-frequency response gap (both TF minus Bolt).\nAll 22 claims across both contexts receive one global Holm correction. Exact two-sided q tests concern conditional signs, not mean superiority.\nA: direct waveform losses retain non-orthogonal cross terms. B/D: approximate pointwise paired-seed bootstrap intervals. C: exact pointwise Clopper–Pearson intervals.\nComponent sum uses oracle-separated inputs and actual f(background); not a deployable repair. Primary native accuracy flip FAILED.\nPure-frequency weakness and composition effects can coexist; nonsignificant G is not absence. No hidden-feature cause established.',fontsize=8,va='bottom')
    fig.tight_layout(rect=[0,.09,1,.95],h_pad=2,w_pad=3.5)
    fig.savefig(out/'crossover_summary.png',dpi=220)
    fig.savefig(out/'crossover_summary.pdf')
    plt.close(fig)


def followup_html(data):
    """Compact completed/partial/closed decisions with their scientific limits."""
    if 'followups' not in data:return '<p>Legacy snapshot: phase outcomes were not yet included.</p>'
    f=data['followups'];phase=f['phase'];escape=html.escape
    names={'timesfm':'TimesFM','bolt':'Bolt','chronos2':'Chronos-2'}
    labels={'unresolved_or_aliased':'Unresolved / aliased','stable_and_adequate':'Stable and adequate',
            'stable_low_order_partial_coverage':'Stable low-order; partial coverage'}
    matrix=''.join('<tr><td>'+names[m]+'</td>'+''.join('<td>'+labels[phase['response_matrix'][m][p]]+'</td>' for p in ('p32_16','p40_20'))+'</tr>' for m in names)
    recovery=''
    for model,row in f['recoverability'].items():
        r=row['fresh']
        recovery+='<tr><td>'+names[model]+'</td>'+''.join(f'<td>{r[key]["mean"]:.7g}</td>' for key in ('native','output_readout','frozen_head_readout','output_minus_head_excess'))+'<td>'+('Pass' if row['passed'] else 'Failed')+'</td></tr>'
    h=f['homogeneous'];shares=h['signed_projection_shares']
    pending=(compactness_html(f['compactness']) if 'compactness' in f else '''<h2>Prospective low-rank compactness screen: not yet run in this snapshot</h2><p>A small structured-direction screen is being prepared. Full-state restoration does not establish a compact or isolated feature; no low-rank outcome is inferred from implementation status.</p>''' if 'restoration' in f else '''<h2>Proposed intervention: not yet run in this report snapshot</h2><p>Anchored composition restoration is a proposed follow-up. No outcome is inferred from source development or included as evidence here.</p>''')
    return '''<h2>Phase-resolution audit: completed, partial response only</h2>
<p>No eligible comparison passed the complete feature-expansion gate. Bolt 40/20 has stable low-order patterns but fails held-out waveform coverage; TimesFM and Bolt 32/16 fail the required stability check. Both pilot-negative comparisons remain closed. No finer grid, higher order, or hidden capture is supported by this branch.</p>
<table><tr><th>Model</th><th>32/16</th><th>40/20</th></tr>'''+matrix+'''</table>
<p><a href="phase_orbit/index.html">Full partial-response report, mode tables and bound figure/data</a>. Only stable 40/20 mode identities are described there. Native preprocessing varies with phase on finite windows; these patterns do not assign a normalization or internal-feature cause. Original pilot control failures remain recorded; separate FP64 replay and fixed-B8 diagnostics do not retroactively relabel them. <a href="followup_results.json">Completed follow-up producer results</a> and <a href="followup_protocols.json">their bound protocols</a> remain available.</p>
<h2>Forecast-output recoverability: tested linear gate closed</h2>
<p>This separate 32/16 task uses context 128 and horizon 32, rather than the horizon-64 crossover above. Readout fitting uses 96 old training bundles and 48 validation bundles; the table evaluates 128 previously inspected generating bundles. Task-specific supervised recovery from the forecast output is substantially worse than the frozen head-input readout. Both models fail the descriptive point gate: output reconstruction MSE ≤0.01 and output-minus-head excess ≤0.01. These are exploratory cached-cohort diagnostics, not new held-out confirmation or an information-destruction proof.</p>
<table><tr><th>Model</th><th>Native MSE</th><th>Output readout</th><th>Frozen head readout</th><th>Output−head excess</th><th>Point gate</th></tr>'''+recovery+'''</table>
<p>Losses use common theoretical mixture energy. Extra semantic labels, a known harmonic basis, and external context mean/scale privilege both readouts; the input oracle is near zero. This is not a deployable forecast repair. Exact intervals, old/fresh cohort results, and the legacy cache row-order limitation remain in the portable source data.</p>
<h2>Homogeneous normalizer: calibrated source replay, descriptive decomposition</h2>
<p>On the TimesFM 32/16 task, context 128 and horizon 64, identical normalized 32-step patches route identical attention value vectors; the collapsed tokenwise path reproduces the inspected native forecasts. This explains the homogeneous context-invariance boundary, not composition error or general irrelevance of attention routing. The 16-bundle replay maximum normalized waveform RMS is '''+f'{h["maximum_replay_RMS"]:.7g}'+'''.</p>
<p>The exact radial decomposition has signed projection shares '''+f'{shares["fixed_normalizer"]:.7g}'+''' for the fixed-normalizer term and '''+f'{shares["radial"]:.7g}'+''' for the radial remainder. These are signed vector projections, not additive energy fractions or causal blame; cross terms and replay residual remain recorded. Fixed-mixture-normalizer component inputs are off the native unit-RMS manifold. No fitting, population replication, or repair claim follows.</p>
'''+pending


def restoration_html(data):
    """Lead with the controlled improvement and its limits before statistics."""
    r=data.get('followups',{}).get('restoration')
    if r is None:return '<h1>Different frequencies expose different failures</h1>'
    if not r['qualified']:return '<h1>Anchored restoration: replication did not qualify</h1>'
    losses=r['mean_losses'];native=r['claims']['native_gain'];random=r['claims']['random_advantage']
    return '''<h1>Composing component states improves fresh mixture forecasts</h1>
<p>The component forecasts are accurate, but TimesFM struggles when those components are mixed. On 64 new mixtures, amplitude- and background-aware state composition after 15 of 20 transformer blocks reduced the mean forecast loss and beat geometry-matched random edits. Those available component representations can support a much better continuation under this controlled intervention.</p>
<p class="notice"><strong>Fresh selected-cut replication passed.</strong> Native normalized MSE '''+f'{losses["native"]:.6g}'+''' → anchored '''+f'{losses["anchored"]:.6g}'+'''; matched-random mean '''+f'{losses["random_mean"]:.6g}'+'''. Improvement occurs in '''+str(native['positive'])+'''/64 seeds; '''+str(native['negative'])+''' seeds become worse. The actual component-sum reference is '''+f'{losses["component_reference"]:.6g}'+'''. This uses known separated components and an off-manifold state edit, not a deployable forecasting fix.</p>
<p><a href="restoration/index.html">Loss chart, discovery history and all controls</a> · <a href="restoration/source_data.json">Portable restoration source data</a> · <a href="restoration_audit.json">Independent model-free audit</a> · <a href="restoration_evidence.json">Bound producer results and protocols</a></p>
<figure><img src="restoration/restoration_summary.png" alt="Restoration discovery and fresh selected-cut replication losses with geometry controls and resource history"><figcaption>TimesFM 32/16, context 128, horizon 64; fresh generating seeds 3700000–3700063. Cut 15 means after 15 blocks, index 14. Cut 20 and raw-input reconstruction are controls, not additional candidates. Known component decomposition is privileged.</figcaption></figure>
<h2>What replicated—and what it does not establish</h2>
<p>Native-minus-anchored gain is '''+f'{native["mean"]:.6g}'+''' ('''+str(native['positive'])+'''/64 favorable signs; global Holm2 p='''+f'{native["p_holm"]:.6g}'+'''). Mean-random-minus-anchored advantage is '''+f'{random["mean"]:.6g}'+''' ('''+str(random['positive'])+'''/64; Holm2 p='''+f'{random["p_holm"]:.6g}'+'''). Both registered empirical size gates and the conditional-majority tests pass. Exact sign tests do not establish population mean-superiority.</p>
<p>The unclipped recovery ratio is '''+f'{r["gates"]["recovery_unclipped"]:.6g}'+''' of the <em>native-to-component-reference gap</em>, not the fraction of native error removed and not a bounded ceiling score. Suffix-remainder energy is '''+f'{r["gates"]["mean_suffix_energy"]:.6g}'+'''. The component sum is a reference, not a performance bound.</p>
<p>This supports compatibility of a background-anchored state composition with the remaining suffix. It does not identify the origin of harmful computation, a unique or isolated feature, or a generally usable repair. Cut 15 was selected from a coarse registered discovery grid; it is not claimed to be the earliest possible site. Geometry controls use two fixed random draws per seed, averaged within seed, not 128 independent samples.</p>
<p>The original 16-seed discovery <strong>failed its memory cap</strong> and remains failed. A resource-only rerun reproduced all 29 saved matrices bitwise and the entire scientific dictionary exactly, then passed; that is execution correction, not replication. The separate fresh 64-seed run used four serial workers, released every model before pooled statistics, passed the 1.5 GiB cap, and finished in '''+f'{r["total_wall_seconds"]:.2f}'+''' seconds.</p>
<p>The independent audit verifies all 64 saved outcomes, identities, hashes, exact binomial/Holm2 values and gate/resource overrides. Its geometry check verifies frozen diagnostics and random-draw hashes; hidden states were not persisted, so it does not regenerate those rotations from hidden states. <a href="restoration_exact_p.json">Bit-exact p-value verification</a> · <a href="restoration_exceptions.json">Three negative cases, descriptive only</a>.</p>'''


def compactness_html(data):
    if data['status']!='closed_negative':return '<h2>Low-rank screen: exploratory candidate only</h2><p>No fresh confirmation is implied.</p>'
    rows=''.join(f"<tr><td>{rank}</td><td>{value['mean_loss']:.7g}</td><td>{100*value['recovery']:.3g}%</td><td>{value['random_advantage']:.7g}</td><td>{value['mean_only_advantage']:.7g}</td><td>{'Pass' if value['passed'] else 'Failed'}</td></tr>" for rank,value in data['ranks'].items())
    return '''<h2>Low-rank compactness screen: closed negative</h2>
<p>A separate exploratory screen reused the 16 discovery seeds, at fixed cut 15, with an uncentered SVD channel basis. Opposite generating-seed parity folds fit on 8 bundles and apply to the other 8; no truth or forecast loss enters the basis fit. All prespecified ranks 1, 2, and 4 fail the frozen screen.</p>
<table><tr><th>Rank</th><th>Mean normalized MSE</th><th>Full-edit gain recovered</th><th>Random-loss advantage</th><th>Mean-only advantage</th><th>Screen</th></tr>'''+rows+'''</table>
<p>Full-state edit loss on these 16 seeds is '''+f'{data["full_edit_mean_loss"]:.7g}'+'''; the mean-only control is '''+f'{data["mean_only_mean_loss"]:.7g}'+'''. Rank 4 recovers only '''+f'{100*data["ranks"]["4"]["recovery"]:.3g}'+'''% of the <em>full-edit improvement</em>, below the required 80%, and loss remains above 0.06. This recovery denominator differs from the native-to-component-reference gap used in the fresh full-state replication.</p>
<p>No higher-rank search or fresh compactness inference follows. This closes the tiny prespecified uncentered-SVD screen; it does <strong>not</strong> disprove all low-dimensional or nonlinear mechanisms. The mean-only edit is a training-fold channel mean broadcast across positions, not every possible position template. Random controls preserve edit geometry but are not matched-rank or capacity-equivalent controls. There are no p-values or confirmatory claims on these reused seeds. <a href="compactness_evidence.json">Bound compactness results and protocol</a> · <a href="compactness_audit.json">Independent audit</a> recomputes the held-out projectors, all 96 keyed random edits and their geometry from saved states.</p>'''


def render(data,out):
    """Make a self-contained narrative plus linked local scientific figure/data."""
    result=data['results'];claims=result['claims'];escape=html.escape
    rows=''.join('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in [key,
        f"{claim['mean']['value']:.6g}",f"{claim['value']:.6g}",f"{claim['p_holm']:.6g}",claim['passed'],claim['scope'].replace('rescue128','rescue 128')])+'</tr>'
        for key,claim in claims.items())
    rbf=data['negatives']['rbf'];clamp=data['negatives']['residual_clamp']
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Frequency composition: fresh replication</title><style>
body{font:17px/1.55 system-ui,sans-serif;max-width:1140px;margin:40px auto;padding:0 24px;color:#192738;background:#fafbfd}h1{line-height:1.15}img{width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:.9rem;color:#475569}a{color:#0755a2}table{border-collapse:collapse;width:100%;font-size:.85rem}td,th{border:1px solid #cbd5e1;padding:8px;text-align:left}th{background:#e7eef6}.notice{background:#fff3dc;border-left:4px solid #b76b16;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.75rem}</style>
'''+restoration_html(data)+'''
<h2>Frequency crossover: the primary native accuracy flip remains failed</h2>
<p>TimesFM 32/16 shows strong error reduction when known components are forecast separately. Bolt 40/20 already struggles with pure frequencies, and most mixed error remains after component summation. The native off-response reversal replicated; the <strong>primary native accuracy ranking flip FAILED</strong>.</p>
<p><a href="../feature_search_20261111/index.html">Earlier study and diagnostic tools</a> · <a href="source_data.json">Portable scientific source data</a> · <a href="crossover_summary.pdf">Figure PDF</a> · <a href="producer_results.json">All saved results</a> · <a href="protocol.json">Frozen producer protocol</a> · <a href="producer_provenance.json">Native source and resource provenance</a></p>
<p class="notice">Context 128 is primary. Its 32/16 TF−Bolt mean gap is positive, but q=0.5625 and Holm22 p=1: no confirmed ranking flip. Context 512 passes the separate sensitivity flip and cannot rescue the primary result. A mean interval excluding zero is not a significant exact sign result.</p>
<figure><img src="crossover_summary.png" alt="Four panels comparing native and component losses, remaining error, exact sign effects and separate context sensitivity"><figcaption>64 fresh matched seeds 3400000–3400063, periods 32/16 and 40/20, horizon 64, six input conditions, three models. The figure focuses on the two selected models; all three remain in the portable results. Confidence intervals are pointwise. Native sums use actual model background forecasts and oracle-separated component inputs, not a deployable repair.</figcaption></figure>
<h2>Decision boundaries</h2><ul>
<li>TimesFM 32/16: positive native-minus-component-sum gain passes the registered sign and empirical size gates. This is an input-level functional composition diagnostic, including native normalization.</li>
<li>Pure-frequency weakness and composition effects can coexist; nonsignificant G is not absence.</li><li>Bolt 40/20 at 128: pure-frequency errors are large and the empirical component-sum gain is positive, but its sign test does not pass. This does not establish absence of composition effects or a pure-only mechanism.</li>
<li>Bolt 40/20 at 512: the detected gain is negative; the component sum is worse. This sensitivity finding is not a positive improvement.</li>
<li>Off-response contrasts reverse between frequency pairs, without identifying a unique hidden feature or proving patch aliasing causes the task error.</li></ul>
<h2>Closed branches remain visible</h2><p>Stronger input-only RBF control: prediction loss '''+f"{rbf['input_loss']:.8g}"+''' versus '''+f"{rbf['joint_loss']:.8g}"+''' after adding derivative features; primary gain '''+f"{rbf['primary_gain']:.8g}"+'''. The robustness gate fails. This closes the tested incremental prediction claim, not every possible feature predictor.</p>
<p>Residual-update clamp: no site advances. Final block 19 has '''+str(clamp['final_positive'])+'''/16 favorable signs, Holm8 p='''+str(clamp['final_holm'])+'''; native controls pass but the full intervention gate fails. This is not forecast repair.</p>
'''+followup_html(data)+'''
<h2>All 22 registered effects</h2><p>g is the TF−Bolt native mixed-loss gap; C is the difference in native-minus-actual-sum gains; o is the full-Gram off-target derivative energy gap. Common theoretical mixed energy normalizes every pure/mixed/background loss. Two-sided detection alone is not a predicted-direction or mean-superiority claim. Legacy mixed-minus-average-pure interaction remains descriptive: an additive biased forecaster can have nonzero interaction at zero actual-sum gain.</p>
<table><thead><tr><th>Contrast</th><th>Signed empirical mean</th><th>q positive/non-tie</th><th>Global Holm22 p</th><th>Size/sign gate</th><th>Scope</th></tr></thead><tbody>'''+rows+'''</tbody></table>
<p>q tests concern conditional majority among non-tied matched seeds. Mean and ratio intervals use approximate paired whole-seed bootstrap; no confidence bound establishes the practical margin. Native normalization remains part of each model function. Significant associations or input interventions do not establish hidden mediation.</p>
<details><summary>Independent CPU replay and source provenance</summary><pre>'''+escape(json.dumps(dict(replay=data['replay'],source_files=data['source_files'],producer_protocol=data['protocol'],producer_provenance=data['provenance']),indent=2))+'''</pre></details>
<p>Rebuild without model weights or run stores, from repository root:</p><pre>python tsfm_lens/experiments/feature_search/stage49_report.py --source-data examples/feature_search_stage49/source_data.json --out examples/feature_search_stage49_rebuild</pre></html>'''
    if not (out/'../feature_search_20261111/index.html').is_file():
        text=text.replace('<a href="../feature_search_20261111/index.html">Earlier study and diagnostic tools</a>', 'Earlier study link unavailable in this output location')
    (out/'index.html').write_text(text,encoding='utf-8')


def main():
    """Use repo sources once, or rebuild entirely from portable scientific JSON."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path)
    parser.add_argument('--source-data',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if bool(args.repo_root)==bool(args.source_data):parser.error('Choose repo-root or source-data')
    data=read(args.source_data) if args.source_data else collect(args.repo_root)
    args.out.mkdir(parents=True,exist_ok=False)
    if 'portable_assets' in data:write_assets(data,args.out)
    else:plot(data,args.out)
    render(data,args.out)
    (args.out/'source_data.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    for name,key in [('producer_results.json','results'),('protocol.json','protocol'),('producer_provenance.json','provenance')]:
        if (args.out/name).exists():
            if read(args.out/name)!=data[key]:raise ValueError('Original producer asset disagrees with portable data')
        else:(args.out/name).write_text(json.dumps(data[key],indent=2,allow_nan=False),encoding='utf-8')
    if 'followup_artifacts' in data:
        for name,key in [('followup_results.json','results'),('followup_protocols.json','protocol')]:
            (args.out/name).write_text(json.dumps({study:artifact[key] for study,artifact in data['followup_artifacts'].items()},indent=2,allow_nan=False),encoding='utf-8')
    if 'restoration_evidence' in data:
        (args.out/'restoration_evidence.json').write_text(json.dumps(data['restoration_evidence'],indent=2,allow_nan=False),encoding='utf-8')
    if 'compactness_evidence' in data:
        (args.out/'compactness_evidence.json').write_text(json.dumps(data['compactness_evidence'],indent=2,allow_nan=False),encoding='utf-8')
    (args.out/'render_provenance.json').write_text(json.dumps(dict(renderer_sha256=sha(__file__),source_data_sha256=sha(args.out/'source_data.json'),scope='Rendering only; independent replay source hash remains a separate computation record'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
