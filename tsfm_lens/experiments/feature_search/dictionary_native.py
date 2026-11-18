"""Queued native preservation/selectivity and actual FFN expansion-hook pilots.

Every forecast uses singleton geometry. Residual output replacement and FFN
down-projection-input replacement are separate hook contracts and artifacts.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np

from experiments.feature_search import dictionary_core as core
from experiments.feature_search.dictionary_pilot import read, write, sha, verify, knowledge


def exact_gate(reference, value, name):
    """Reject any arithmetic discrepancy; do not silently relax identity gates."""
    if not np.array_equal(reference, value):
        raise ValueError(f'Exact {name} gate failed: {float(np.max(abs(reference-value)))}')


def receiver_indices(rows, split='test', per_domain=4):
    """Deterministic one case per source; no selection using features or forecasts."""
    result = []
    for domain in sorted({r['domain'] for r in rows}):
        sources = sorted({r['source'] for r in rows if r['domain']==domain and r['split']==split})
        for source in sources[:per_domain]:
            result.append(next(i for i, r in enumerate(rows) if r['source']==source and r['split']==split))
    return result


def load_native(source, model):
    """Verify local frozen weights before constructing an offline native adapter."""
    from pathlib import Path
    from experiments.feature_search.real_background_poc import Native
    cp = read(source/'protocol.json')['checkpoints'][model]
    for name, expected in cp['weights'].items():
        if sha(Path(cp['path'])/name) != expected:
            raise ValueError('Checkpoint changed')
    return Native(model, cp['path'])


def ffn_projection(native, cut):
    """Down-projection INPUT is the actual FFN expansion, never a residual output.

    TimesFM ff1 consumes activation(ff0(...)); Bolt wo consumes its activated
    (possibly gated) expansion. Neither is labeled PatchTST post-GELU.
    """
    if native.name == 'timesfm':
        return native.model.stacked_xf[int(cut)].ff1, 'activated_ffn_expansion_before_ff1'
    block = native.model.encoder.block[int(cut)]
    ff = block.layer[-1].DenseReluDense
    return ff.wo, 'activated_or_gated_ffn_expansion_before_wo'


@contextmanager
def expansion_hook(module, captured, replacement=None):
    """Capture/replace one projection input with strict geometry and hook count."""
    import torch
    def hook(_module, args):
        hidden = args[0]
        captured.append(hidden.detach().cpu().numpy().copy())
        if replacement is not None:
            value = torch.as_tensor(replacement, dtype=hidden.dtype, device=hidden.device)
            if value.shape != hidden.shape:
                raise ValueError('FFN expansion geometry mismatch')
            return (value,)+args[1:]
    handle = module.register_forward_pre_hook(hook)
    try:
        yield
        if len(captured) != 1:
            raise ValueError('Expected exactly one FFN projection call')
    finally:
        handle.remove()


def forecast_metrics(prediction, baseline, scale):
    """Report absolute/history-normalized distortion and native-output variation units."""
    error = float(np.mean((prediction-baseline)**2))
    energy = float(np.var(baseline))
    return dict(mse=error, history_variance_nmse=error/scale**2,
                native_forecast_variance_nmse=None if energy <= 1e-12 else error/energy)


def native_stage(args, dest):
    """Native reconstruction and FIT-selected feature removal with matched nulls."""
    from experiments.feature_search.selective_study import blocks, capture, patched
    from experiments.feature_search.composition_restoration import random_edits
    from experiments.feature_search.coherence_study import normalization
    p, source = verify(args.out, args.model)
    trained = args.out/f'fit_{args.model}_cut{args.cut}_{args.method}_k{args.k}_s{args.seed}'
    fit_result = read(trained/'results.json')
    if fit_result['protocol_sha256'] != sha(args.out/'protocol.json') or fit_result['dictionary_sha256'] != sha(trained/'dictionary.npz'):
        raise ValueError('Dictionary provenance mismatch')
    artifact = dict(np.load(trained/'dictionary.npz'))
    rows = read(source/'rows.json')
    indices = receiver_indices(rows)
    x = np.load(source/'inputs.npz')['history']
    native = load_native(source, args.model)
    site = blocks(native)[args.cut]
    c = read(source/f'{args.model}_worker.json')['cuts'].index(args.cut)
    result = []
    rng = np.random.default_rng(args.seed)
    with np.load(source/f'{args.model}_states.npz') as cache:
        for i in indices:
            query = x[i:i+1]
            sd = float(query.std())
            if sd <= 1e-8:
                raise ValueError('Zero physical output scale can hide invalid replay')
            baseline, captured = capture(native, query, {args.cut: site})
            h = captured[args.cut]
            exact_gate(baseline, native.predict(query)[0], 'repeat')
            exact_gate(h, cache[f'{i}_full'][c:c+1], 'saved singleton state')
            exact_gate(baseline, patched(native, query, site, h), 'self-patch')
            z = core.encode((h-artifact['center'])/float(artifact['scale']), artifact)
            reconstruction = core.reconstruct(z, artifact)*float(artifact['scale'])+artifact['center']
            variants = {'reconstruction': reconstruction-h}
            for budget in (1, 4):
                selected = fit_result['perturbations'][str(budget)]['selected']
                variants[f'ablate{budget}'] = -(z[..., selected] @ core.decoder(artifact)[selected])*float(artifact['scale'])
            measurements = {}
            for name, delta in variants.items():
                control, invalid, diagnostics = random_edits(h, delta, rng, count=2, return_diagnostics=True)
                pred = patched(native, query, site, h+delta)
                random_predictions = [patched(native, query, site, h+value) for value in control]
                measurements[name] = dict(**forecast_metrics(pred, baseline, sd),
                    random_valid=not bool(invalid[0]), random_geometry=diagnostics,
                    random_metrics=[forecast_metrics(v, baseline, sd) for v in random_predictions],
                    prediction=pred[0].tolist(), random_predictions=[v[0].tolist() for v in random_predictions])
            causal = {}
            for direction in ('sinusoid', 'noise'):
                raw = (np.sin(2*np.pi*5.35*np.arange(-128, 0)/128+.3) if direction=='sinusoid'
                       else np.random.default_rng(4944000+i).normal(size=128))
                raw = raw/np.sqrt(np.mean(raw**2))*sd
                natural, removed, randomized = [], {1: [], 4: []}, {1: [], 4: []}
                random_valid = {1: True, 4: True}
                for sign in (1, -1):
                    with normalization(native, query, 'background'):
                        pred, states = capture(native, query+sign*.1*raw, {args.cut: site})
                    hs = states[args.cut]
                    oracle = patched(native, query, site, hs)
                    # Shared normalization is tested numerically, not inferred from reach.
                    exact_gate(pred, oracle, 'natural counterfactual suffix')
                    natural.append(pred)
                    zs = core.encode((hs-artifact['center'])/float(artifact['scale']), artifact)
                    for budget in (1, 4):
                        selected = fit_result['perturbations'][str(budget)]['selected']
                        delta = -(zs[..., selected] @ core.decoder(artifact)[selected])*float(artifact['scale'])
                        null, invalid = random_edits(hs, delta, rng, count=1)
                        random_valid[budget] &= not bool(invalid[0])
                        removed[budget].append(patched(native, query, site, hs+delta))
                        randomized[budget].append(patched(native, query, site, hs+null[0]))
                response = natural[0]-natural[1]
                energy = float(np.mean(response**2))
                causal[direction] = dict(response_energy=energy, reach=bool(energy > 1e-12*sd**2),
                    budgets={str(b): dict(random_valid=random_valid[b],
                        response_change_nmse=None if energy <= 1e-12*sd**2 else float(np.mean((removed[b][0]-removed[b][1]-response)**2)/energy),
                        random_response_change_nmse=None if energy <= 1e-12*sd**2 else float(np.mean((randomized[b][0]-randomized[b][1]-response)**2)/energy)) for b in (1, 4)})
            result.append(dict(row=i, source=rows[i]['source'], domain=rows[i]['domain'],
                baseline=baseline[0].tolist(), exact_gates=True, measurements=measurements, causal_selectivity=causal))
            write(dest/'partial.json', result)
    write(dest/'results.json', result)
    preservation = core.source_summary([r['measurements']['reconstruction']['history_variance_nmse'] for r in result], result)
    random_invalid = sum(not r['measurements']['reconstruction']['random_valid'] for r in result)
    knowledge(dest, 'Does reconstruction preserve native forecasts, and does feature removal affect sinusoid more than nuisance?',
              dict(results_file='results.json', sha256=sha(dest/'results.json'), rows=len(result), exact_identity_gates=True,
                   preservation=preservation, reconstruction_random_invalid=random_invalid),
              ['exact singleton self/repeat/saved gates', 'same-cut counterfactual oracle', 'two Gram-matched reconstruction/edit random controls', 'FIT-selected feature IDs'],
              p['limitations']+['fixed perturbation phase; no semantic uniqueness or learned-concept verdict',
                               'random sign-branch controls are independent, so response specificity null is conservative'],
              'Coordinator reviews preservation, reach, null validity and target-v-noise effects jointly; fresh crossed-phase test before usefulness claim.', 'causal_pilot_complete_no_automatic_claim')


def ffn_stage(args, dest):
    """Missing-hook pilot: actual expansion states and fresh quadrature perturbations."""
    from experiments.feature_search.coherence_study import normalization
    p, source = verify(args.out, args.model)
    rows = read(source/'rows.json')
    x = np.load(source/'inputs.npz')['history']
    native = load_native(source, args.model)
    module, kind = ffn_projection(native, args.cut)
    indices = receiver_indices(rows, 'fit', 2)+receiver_indices(rows, 'test', 2)
    states, audit = {}, []
    for i in indices:
        query = x[i:i+1]
        captured = []
        with expansion_hook(module, captured):
            baseline = native.predict(query)[0]
        h = captured[0]
        exact_gate(baseline, native.predict(query)[0], 'FFN repeat')
        with expansion_hook(module, [], replacement=h):
            exact_gate(baseline, native.predict(query)[0], 'FFN self-patch')
        states[f'{i}_full'] = h[0]
        sd = float(query.std())
        for phase in (0., np.pi/2):
            raw = np.sin(2*np.pi*5.35*np.arange(-128, 0)/128+phase)
            raw = raw/np.sqrt(np.mean(raw**2))*sd
            pair = []
            for sign in (1, -1):
                values = []
                with normalization(native, query, 'background'), expansion_hook(module, values):
                    native.predict(query+sign*.1*raw)
                pair.append(values[0][0])
            states[f'{i}_phase{phase}'] = np.stack(pair)
        audit.append(dict(row=i, source=rows[i]['source'], split=rows[i]['split'],
                          kind=kind, full_shape=list(h.shape), exact_self_patch=True))
    np.savez_compressed(dest/'ffn_states.npz', **states)
    write(dest/'rows.json', audit)
    knowledge(dest, 'Can actual FFN expansion neurons be captured and replayed at the chosen cut?',
              dict(rows=audit, sha256=sha(dest/'ffn_states.npz')), ['exact singleton FFN self/repeat gates', 'FIT/TEST sources separate', 'two quadrature phases'],
              p['limitations']+['hook pilot only; this is NOT residual-coordinate evidence or PatchTST post-GELU replication'],
              'Fit matched native/rotation few-feature selectivity on FIT expansion states, then TEST; add noise/background and initialization controls.', 'instrumentation_complete')
