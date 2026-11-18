"""Saved-state edit reach audit. No fitting, native imports, or output inference.

Base states have an exact singleton replay gate. Cached perturbed states came
from paired inference; their norms cannot certify unsaved singleton branch edits.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
import resource
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/dictionary_pilot4951000/recovery_exit_monitor'
OUT = ROOT/'runs/comparability_dictionary_edit_audit20261116'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(hidden, artifact):
    """Reproduce frozen NumPy ReLU plus stable magnitude TopK, without torch."""
    x = (hidden-artifact['center'])/float(artifact['scale'])
    pre = np.maximum((x-artifact['b_dec'])@artifact['W_enc']+artifact['b_enc'], 0)
    order = np.argsort(-abs(pre), axis=-1, kind='stable')
    code = np.zeros_like(pre)
    ids = order[..., :int(artifact['k'])]
    np.put_along_axis(code, ids, np.take_along_axis(pre, ids, axis=-1), axis=-1)
    return code


def norms(hidden, selected_code, decoder, scale):
    """Distinguish latent activity, proposed edit, injected edit and null geometry."""
    delta = -(selected_code@decoder)*float(scale)
    injected = np.asarray(hidden+delta, np.float32).astype(float)-hidden.astype(float)
    rows, edits = hidden.reshape(-1, hidden.shape[-1]).astype(float), delta.reshape(-1, delta.shape[-1]).astype(float)
    _, singular, right = np.linalg.svd(rows, full_matrices=False)
    tolerance = np.finfo(float).eps*max(rows.shape)*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > tolerance))
    basis = right[:rank]
    perpendicular = edits-(edits@basis.T)@basis
    strength = np.linalg.svd(perpendicular, compute_uv=False)
    residual_rank = int(np.sum(strength > max(1e-10, np.finfo(float).eps*max(edits.shape)*(strength[0] if len(strength) else 0.))))
    return dict(selected_nonzero_coefficients=int(np.count_nonzero(selected_code)),
                selected_code_l2=float(np.linalg.norm(selected_code.astype(float))),
                proposed_edit_l2=float(np.linalg.norm(edits)), proposed_edit_rms=float(np.sqrt(np.mean(edits**2))),
                proposed_exact_zero=bool(np.all(delta == 0)),
                injected_edit_l2=float(np.linalg.norm(injected)), injected_exact_zero=bool(np.all(injected == 0)),
                perpendicular_edit_l2=float(np.linalg.norm(perpendicular)), randomizable_rank=residual_rank,
                deterministic_geometry_failure=bool(residual_rank == 0 or rows.shape[1]-rank < residual_rank))


def calibration():
    """Known-answer decoys prevent interpreting geometry failure as a zero edit."""
    h=np.array([[1., 0., 0.]], dtype=np.float32)
    zero=norms(h,np.array([[0.]],np.float32),np.array([[1.,0.,0.]],np.float32),1.)
    parallel=norms(h,np.array([[1.]],np.float32),np.array([[1.,0.,0.]],np.float32),1.)
    cancelled=norms(h,np.array([[1.,1.]],np.float32),np.array([[0.,1.,0.],[0.,-1.,0.]],np.float32),1.)
    rounded=norms(h,np.array([[1e-10]],np.float32),np.array([[1.,0.,0.]],np.float32),1.)
    assert zero['proposed_exact_zero'] and zero['selected_nonzero_coefficients']==0
    assert parallel['deterministic_geometry_failure'] and parallel['proposed_edit_l2']==1 and not parallel['proposed_exact_zero']
    assert cancelled['selected_nonzero_coefficients']==2 and cancelled['proposed_exact_zero']
    assert not rounded['proposed_exact_zero'] and rounded['injected_exact_zero']
    plants=[]
    for name, bad, expected in [('invalid_implies_zero', parallel['deterministic_geometry_failure'], parallel['proposed_exact_zero']),
                                ('latent_active_implies_nonzero_edit', cancelled['selected_nonzero_coefficients']>0, not cancelled['proposed_exact_zero']),
                                ('proposed_equals_injected_zero', rounded['proposed_exact_zero'], rounded['injected_exact_zero'])]:
        assert bad != expected, name
        plants.append(dict(plant=name,detected=True))
    return dict(known_answer_cases=dict(zero=zero,parallel_nonzero=parallel,latent_cancellation=cancelled,rounding=rounded),isolated_logical_plants=plants)


def run():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    controls=calibration()
    protocol=read(RUN/'protocol.json'); source=Path(protocol['source'])
    code_files=[Path(p) for p in protocol['code_hashes']]
    assert all(sha(p)==protocol['code_hashes'][str(p)] for p in code_files)
    state_file=source/'timesfm_states.npz'
    assert sha(state_file)==protocol['hashes']['timesfm_states.npz']
    details=[]; provenance={str(state_file.relative_to(ROOT)):sha(state_file)}
    with np.load(state_file) as cache:
        for seed in (4951000,4951001,4951002):
            fit=RUN/f'fit_timesfm_cut0_topk_k8_s{seed}'
            native=RUN/f'native_timesfm_cut0_topk_k8_s{seed}'
            result=read(fit/'results.json'); observations=read(native/'results.json')
            with np.load(fit/'dictionary.npz') as f:artifact={k:f[k] for k in f.files}
            assert sha(fit/'dictionary.npz')==result['dictionary_sha256']
            for path in (fit/'dictionary.npz',fit/'results.json',native/'results.json'):
                provenance[str(path.relative_to(ROOT))]=sha(path)
            for observation in observations:
                row=observation['row']
                assert observation['exact_gates']
                states={'base':cache[f'{row}_full'][0:1]}
                for direction in ('sinusoid','noise'):
                    pair=cache[f'{row}_{direction}_0.1_locked'][0]
                    for index,sign in enumerate(('plus','minus')):states[f'{direction}_{sign}']=pair[index:index+1]
                for label,hidden in states.items():
                    z=encode(hidden,artifact)
                    for budget in ('1','4'):
                        selected=result['perturbations'][budget]['selected']
                        metrics=norms(hidden,z[...,selected],artifact['W_dec'][selected],artifact['scale'])
                        if label=='base':
                            saved=observation['measurements']['ablate'+budget]
                            output=dict(kind='base_forecast_MSE',value=saved['mse'],random_valid=saved['random_valid'])
                        else:
                            direction=label.split('_')[0]
                            saved=observation['causal_selectivity'][direction]['budgets'][budget]
                            output=dict(kind='pair_response_change_NMSE_not_individual_branch_effect',value=saved['response_change_nmse'],random_valid=saved['random_valid'],natural_response_reach=observation['causal_selectivity'][direction]['reach'])
                        details.append(dict(seed=seed,row=row,source=observation['source'],state=label,budget=budget,selected=selected,
                            state_evidence='Exact-replayed native singleton base' if label=='base' else 'Earlier saved paired-inference state; fresh native singleton perturbed state was not saved or equality-gated',
                            norms=metrics,saved_output=output))
    summary=[]
    for seed in (4951000,4951001,4951002):
        for label in ('base','sinusoid_plus','sinusoid_minus','noise_plus','noise_minus'):
            for budget in ('1','4'):
                cells=[r for r in details if (r['seed'],r['state'],r['budget'])==(seed,label,budget)]
                summary.append(dict(seed=seed,state=label,budget=budget,rows=len(cells),
                    zero_latent_rows=sum(c['norms']['selected_nonzero_coefficients']==0 for c in cells),
                    zero_proposed_edits=sum(c['norms']['proposed_exact_zero'] for c in cells),
                    zero_injected_edits=sum(c['norms']['injected_exact_zero'] for c in cells),
                    max_proposed_edit_l2=max(c['norms']['proposed_edit_l2'] for c in cells),
                    max_injected_edit_l2=max(c['norms']['injected_edit_l2'] for c in cells),
                    nonzero_edit_sources=[c['source'] for c in cells if not c['norms']['proposed_exact_zero']]))
    OUT.mkdir(exist_ok=False)
    record=dict(question='Were selected activation edits actually zero, rather than merely output-invisible or invalid random geometry?',
        status='complete_saved_only',design='Reconstruct selected code contributions and float32 injection norms using exact frozen artifacts; TimesFM TopKk8 three seeds,20 receivers,budgets1/4,base and saved paired +/-states. No fits or native calls.',
        exact_findings=summary,details=details,controls=controls,provenance_sha256=provenance,
        evidence_class='Saved-array instrumentation audit; base states exactly bound to native runs, perturbed cached states have weaker provenance for singleton edits.',
        limitations=['Zero output central-response change can coexist with nonzero activation edits, including cancelling branch effects.',
            'Invalid null can mean zero edit, edit wholly in fixed receiver span, insufficient complement rank, or numerical geometry failure; it does not by itself establish zero edit.',
            'Native singleton perturbed states/latents were not saved; earlier paired-inference caches do not certify exact singleton branch-edit norms.',
            'No causal irrelevance conclusion. No new native evidence.'],
        next_decision='Correct wording to distinguish confirmed zero base edits, cached perturbation edit norms, and unsaved native branch edits. Hume may incorporate this into saved-only validity audit; no native authorization.',
        code_sha256=sha(Path(__file__)))
    (OUT/'knowledge.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
