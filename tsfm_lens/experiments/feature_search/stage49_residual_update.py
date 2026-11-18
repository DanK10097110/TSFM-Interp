"""Stage49: exploratory natural-perturbation residual-update mediation.

No semantic feature, additive attribution, or general repair claim follows.
This standalone pilot does not alter model weights or older experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

SITES = (0, 9, 15, 19)
CONDITIONS = [('base', 0., 0)] + [(f'psi_{sign}_{delta}', delta, sign)
                                                for delta in (.01, .02) for sign in (1, -1)]


def checkpoint_path(path):
    """Keep the HF snapshot parent containing config; weight files are symlinks."""
    return Path(path).absolute()


def corpus():
    contexts, futures, metadata = [], [], []
    t = np.arange(160)
    for seed in range(16):
        rng = np.random.default_rng(3100000 + seed)
        a, b = rng.uniform(.5, 2, 2)
        offset = rng.uniform(-2, 2)
        phi, psi = rng.uniform(-np.pi, np.pi, 2)
        for label, delta, sign in CONDITIONS:
            angle = psi + sign * delta
            x = offset + a*np.sin(2*np.pi*t/32+phi) + b*np.sin(2*np.pi*t/16+angle)
            contexts.append(x[:128]); futures.append(x[128:])
            metadata.append(dict(seed=seed, generator_seed=3100000+seed, condition=label,
                                 a=a, b=b, offset=offset, phi=phi, psi=angle,
                                 scale=float(np.sqrt((a*a+b*b)/2))))
    return np.asarray(contexts, np.float32), np.asarray(futures), metadata


def primary(output):
    if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[0], torch.Tensor):
        raise ValueError('Expected TimesFM (hidden, live decode_cache) tuple')
    return output[0]


def replacement(live_input, live_output, base_input, base_output, mode, direction=None):
    """Preserve live tuple tail; anchored arithmetic gives exact self identity."""
    y = primary(live_output)
    if not all(v.shape == y.shape for v in (live_input, base_input, base_output)):
        raise ValueError('Every position and channel must match the SAME baseline row')
    anchored = base_output + (live_input - base_input)
    if mode == 'update':
        result = anchored
    elif mode == 'output':
        result = base_output.clone()
    elif mode == 'random':
        if direction is None or direction.shape != y.shape:
            raise ValueError('Need a fixed per-seed all-position random direction')
        axes = tuple(range(1, y.ndim))
        norm = torch.linalg.vector_norm(y-anchored, dim=axes, keepdim=True)
        unit = direction / torch.linalg.vector_norm(direction, dim=axes, keepdim=True)
        result = y + norm * unit
    else:
        raise ValueError(mode)
    return (result,) + live_output[1:]


@contextmanager
def block_hook(module, baseline=None, mode=None, direction=None):
    """Clone BEFORE forward to protect the residual input from inplace blocks."""
    state = dict(inputs=[], outputs=[])
    def pre(_, args):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError('Expected positional hidden-state input')
        state['inputs'].append(args[0].detach().clone())
    def post(_, args, output):
        y = primary(output)
        state['outputs'].append(y.detach().clone())
        if baseline is not None:
            return replacement(state['inputs'][-1], output, *baseline, mode, direction)
    handles = [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


def prepare(out, checkpoint=None):
    import timesfm.torch.transformer as source
    import timesfm.timesfm_2p5.timesfm_2p5_torch as decode_source
    protocol = dict(stage=49, model='timesfm', seed_base=3100000, n_seeds=16,
        context=128, horizon=32, periods=[32,16], conditions=CONDITIONS, sites=SITES,
        primary_delta=.01, consistency_delta=.02, batch_size=8,
        intervention='y_base+(x_live-x_base), all positions; live tuple cache preserved',
        random='one fixed Gaussian direction per seed/site, seed 4931000+100*site+seed; matched per-row full edit Frobenius norm; same direction across signs/deltas; no selection',
        controls='native repeat/permutation/capture, baseline self-clamp and same-condition self-clamp exact zero; one call per block; whole-output ceiling suppresses both derivatives',
        metrics='signed four spectral coefficient derivatives; off energy / aggregate analytic own strength; intended derivative relative RMS error; full future scaled MSE',
        practical=dict(relative_off_reduction=.25, absolute_off_reduction=.05,
                       max_intended_relative_RMS_error_worsening=.05, both_delta_directions=True),
        statistical='Eight two-sided exact sign tests: each site benefit vs native and vs fixed random, global Holm8; both positive majority and adjusted p<.05; exploratory site selection only',
        uncertainty='2000 paired-seed bootstrap draws, fixed seed 4931016; ratio of aggregate energies; pointwise intervals',
        interpretation='Composite-block context-dependent update sensitivity, not isolated semantic feature or general repair; downstream effects need not add',
        source_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        decode_source_sha256=hashlib.sha256(Path(decode_source.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if checkpoint is not None:
        checkpoint = checkpoint_path(checkpoint)
        protocol['checkpoint'] = str(checkpoint)
        protocol['checkpoint_sha256'] = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
    x, future, metadata = corpus()
    if out.exists():
        raise ValueError('Use a fresh output directory; no frozen artifacts overwritten')
    out.mkdir(parents=True)
    np.savez_compressed(out/'corpus.npz', context=x, clean_future=future)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    protocol['corpus_sha256'] = hashlib.sha256((out/'corpus.npz').read_bytes()).hexdigest()
    protocol['metadata_sha256'] = hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    return protocol


def coefficient_derivative(prediction, delta, metadata):
    """Global future cos/sin ordering: phi cos,sin; psi cos,sin."""
    rows = prediction.reshape(16, 5, 32)
    pair = (1, 2) if delta == .01 else (3, 4)
    derivative = (rows[:, pair[0]] - rows[:, pair[1]]) / (2*delta)
    t = np.arange(128, 160)
    design = np.stack([np.cos(2*np.pi*t/32), np.sin(2*np.pi*t/32),
                       np.cos(2*np.pi*t/16), np.sin(2*np.pi*t/16), np.ones(32)], axis=1)
    scales = np.asarray([metadata[i*5]['scale'] for i in range(16)])
    return (np.linalg.lstsq(design, derivative.T, rcond=None)[0][:4].T
            / scales[:, None])


def analyze(predictions, future, metadata, controls):
    from scipy.stats import binomtest
    from tsfm_lens.analysis.stats import holm
    rng = np.random.default_rng(4931016)
    samples = rng.integers(0, 16, (2000, 16))
    b = np.asarray([metadata[i*5]['b']/metadata[i*5]['scale'] for i in range(16)])
    psi = np.asarray([metadata[i*5]['psi'] for i in range(16)])
    ideal = np.stack([b*np.cos(psi), -b*np.sin(psi)], axis=1)
    strength = b*b
    def ratio(numerator):
        values = numerator[samples].sum(axis=1)/strength[samples].sum(axis=1)
        return dict(value=float(numerator.sum()/strength.sum()),
                    lo=float(np.quantile(values, .025)), hi=float(np.quantile(values, .975)))
    def sign(d):
        p, n = int((d > 0).sum()), int((d < 0).sum())
        return dict(positive=p, negative=n, ties=int((d == 0).sum()),
                    p=float(binomtest(p, p+n, .5).pvalue) if p+n else 1.)
    results = dict(scope='exploratory composite-update mediation; no isolated-feature or repair claim',
                   controls=controls, deltas={}, coefficient_order=['phi_cos','phi_sin','psi_cos','psi_sin'])
    family = {}
    for delta in (.01, .02):
        derivatives = {k:coefficient_derivative(v, delta, metadata) for k,v in predictions.items()}
        energies = {k:np.sum(v[:, :2]**2, axis=1) for k,v in derivatives.items()}
        own_errors = {k:float(np.sqrt(np.sum((v[:, 2:]-ideal)**2)/strength.sum()))
                      for k,v in derivatives.items()}
        table = {}
        for key, value in derivatives.items():
            scale = np.asarray([r['scale'] for r in metadata])
            per_row_mse = np.mean((predictions[key]-future)**2, axis=1)/scale**2
            table[key] = dict(off_energy=ratio(energies[key]),
                intended_relative_RMS_error=own_errors[key],
                signed_derivative_mean=value.mean(axis=0).tolist(),
                intended_projection=float(np.sum(value[:, 2:]*ideal)/strength.sum()),
                scaled_future_MSE=float(per_row_mse.mean()),
                scaled_future_MSE_by_condition={condition[0]:float(per_row_mse[i::5].mean()) for i,condition in enumerate(CONDITIONS)},
                signed_derivative_change_from_native_mean=(value-derivatives['native']).mean(axis=0).tolist(),
                per_seed_signed_coefficients=value.tolist())
        for site in SITES:
            key = f'update_{site}'
            native_benefit = energies['native']-energies[key]
            random_benefit = energies[f'random_{site}']-energies[key]
            table[key]['native_benefit'] = ratio(native_benefit)
            table[key]['random_benefit'] = ratio(random_benefit)
            table[key]['relative_off_reduction'] = float(native_benefit.sum()/energies['native'].sum())
            table[key]['intended_error_worsening'] = own_errors[key]-own_errors['native']
            if delta == .01:
                family[f'{site}_vs_native'] = sign(native_benefit)
                family[f'{site}_vs_random'] = sign(random_benefit)
        results['deltas'][str(delta)] = table
    adjusted = holm({k:v['p'] for k,v in family.items()})
    for key in family:
        family[key]['p_holm'] = adjusted[key]
    results['holm8'] = family
    results['sites'] = {}
    for site in SITES:
        tables = [results['deltas'][str(d)][f'update_{site}'] for d in (.01,.02)]
        primary = tables[0]
        practical = (primary['relative_off_reduction'] >= .25 and primary['native_benefit']['value'] >= .05
                     and primary['intended_error_worsening'] <= .05
                     and all(t['native_benefit']['value'] > 0 and t['random_benefit']['value'] > 0 for t in tables))
        statistical = all(family[f'{site}_vs_{control}']['positive'] > family[f'{site}_vs_{control}']['negative']
                          and family[f'{site}_vs_{control}']['p_holm'] < .05 for control in ('native','random'))
        ceiling = all(np.linalg.norm(coefficient_derivative(predictions[f'output_{site}'],d,metadata))
                      / np.sqrt(strength.sum()) <= .01 for d in (.01,.02))
        results['sites'][str(site)] = dict(practical_pass=bool(practical), statistical_pass=bool(statistical),
                                          ceiling_pass=bool(ceiling),
                                          advance=bool(practical and statistical and ceiling and controls['all_pass']))
    return results


def run(out):
    """Single offline GPU worker; the caller must separately authorize its slot."""
    from tsfm_lens.config import ModelConfig, DataConfig
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    import time
    import timesfm.torch.transformer as transformer_source
    import timesfm.timesfm_2p5.timesfm_2p5_torch as decode_source
    total_started = time.monotonic()
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['driver_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Code changed after prospective protocol freeze')
    for key, module in [('source_sha256',transformer_source),('decode_source_sha256',decode_source)]:
        if protocol[key] != hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest():
            raise ValueError('Installed source changed after prospective freeze')
    if (out/'predictions.npz').exists():
        raise ValueError('Never overwrite a completed or partial run')
    if protocol['corpus_sha256'] != hashlib.sha256((out/'corpus.npz').read_bytes()).hexdigest():
        raise ValueError('Corpus binding failed')
    if protocol['metadata_sha256'] != hashlib.sha256((out/'metadata.json').read_bytes()).hexdigest():
        raise ValueError('Metadata binding failed')
    checkpoint = Path(protocol['checkpoint'])
    if protocol['checkpoint_sha256'] != hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest():
        raise ValueError('Checkpoint binding failed')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(1., 6*2**30/torch.cuda.get_device_properties(0).total_memory))
    adapter = TimesFMAdapter(ModelConfig(name='timesfm', checkpoint=str(checkpoint.parent)),
        DataConfig(context_len=128, horizon=32), torch.device('cuda:0'), torch.float32)
    adapter.ensure_loaded()
    blocks = dict(adapter.module.named_modules())
    if len(adapter.module.stacked_xf) != 20:
        raise ValueError('Frozen site selection requires exactly 20 blocks')
    arrays = np.load(out/'corpus.npz'); x = arrays['context']; future = arrays['clean_future']
    metadata = json.loads((out/'metadata.json').read_text())
    controls = dict(native_repeat=0., permutation=0., capture=0., self_clamp=0.,
                    base_clamp=0., random_edit_relative_norm_error=0.)
    caches = {site:[[],[]] for site in SITES}
    native = []
    from contextlib import ExitStack
    started = time.monotonic()
    edits = {}
    saved_directions = {}
    def predict(batch):
        return adapter.predict(batch, 32, [.5])['point']
    with torch.inference_mode():
        for start in range(0, len(x), 8):
            batch = x[start:start+8]
            expected = predict(batch)
            controls['native_repeat'] = max(controls['native_repeat'], float(abs(expected-predict(batch)).max()))
            perm = np.random.default_rng(4931000+start).permutation(len(batch))
            controls['permutation'] = max(controls['permutation'], float(abs(expected-predict(batch[perm])[np.argsort(perm)]).max()))
            with ExitStack() as stack:
                states = {site:stack.enter_context(block_hook(blocks[f'stacked_xf.{site}'])) for site in SITES}
                observed = predict(batch)
            controls['capture'] = max(controls['capture'], float(abs(expected-observed).max()))
            for site,state in states.items():
                if len(state['inputs']) != 1 or len(state['outputs']) != 1:
                    raise ValueError('Exactly one prefill block call required')
                for index,key in enumerate(('inputs','outputs')):
                    if tuple(state[key][0].shape) != (len(batch),4,1280):
                        raise ValueError('Unexpected full physical token shape')
                    caches[site][index].append(state[key][0].cpu())
            native.append(expected)
        caches = {site:tuple(torch.cat(v) for v in values) for site,values in caches.items()}
        predictions = dict(native=np.concatenate(native))
        for site in SITES:
            module = blocks[f'stacked_xf.{site}']
            directions = torch.from_numpy(np.stack([np.random.default_rng(4931000+100*site+seed).normal(size=(4,1280)).astype(np.float32) for seed in range(16)]))
            saved_directions[str(site)] = directions.numpy()
            for mode in ('update','random','output','self'):
                outputs = []
                for start in range(0,len(x),8):
                    rows = np.arange(start,min(start+8,len(x)))
                    base_rows = rows if mode == 'self' else (rows//5)*5
                    baseline = tuple(v[base_rows].to(adapter.device) for v in caches[site])
                    direction = directions[rows//5].to(adapter.device)
                    with block_hook(module, baseline, 'update' if mode == 'self' else mode, direction) as state:
                        output = predict(x[rows])
                    if len(state['inputs']) != 1:
                        raise ValueError('Hook fired an unexpected number of times')
                    if mode == 'random':
                        y = state['outputs'][0]
                        # Reconstruct in the SAME anchored arithmetic as the actual hook.
                        anchored = baseline[1]+(state['inputs'][0]-baseline[0])
                        edited = replacement(state['inputs'][0], (y,None), *baseline, 'random',direction)[0]
                        expected_norm = torch.linalg.vector_norm(y-anchored,dim=(1,2))
                        actual_norm = torch.linalg.vector_norm(edited-y,dim=(1,2))
                        error = ((actual_norm-expected_norm).abs()/expected_norm.clamp_min(1e-12)).max().item()
                        controls['random_edit_relative_norm_error'] = max(controls['random_edit_relative_norm_error'],error)
                        edits[f'{site}_{start}'] = dict(expected=expected_norm.cpu().tolist(),actual=actual_norm.cpu().tolist())
                    outputs.append(output)
                joined = np.concatenate(outputs)
                if mode == 'self':
                    controls['self_clamp'] = max(controls['self_clamp'],float(abs(joined-predictions['native']).max()))
                else:
                    predictions[f'{mode}_{site}'] = joined
                    controls['base_clamp'] = max(controls['base_clamp'],float(abs(joined[::5]-predictions['native'][::5]).max()))
    controls['all_pass'] = bool(all(controls[k] == 0 for k in ('native_repeat','permutation','capture','self_clamp','base_clamp')) and controls['random_edit_relative_norm_error'] <= 1e-5)
    controls['walltime_seconds'] = time.monotonic()-started
    controls['total_including_load_walltime_seconds'] = time.monotonic()-total_started
    controls['peak_GPU_allocated_bytes'] = torch.cuda.max_memory_allocated()
    controls['peak_GPU_reserved_bytes'] = torch.cuda.max_memory_reserved()
    np.savez_compressed(out/'predictions.npz', **predictions)
    np.savez_compressed(out/'random_directions.npz', **saved_directions)
    (out/'random_edit_norms.json').write_text(json.dumps(edits,indent=2))
    np.savez_compressed(out/'block_cache.npz', **{f'{site}_{name}':value.numpy() for site,pair in caches.items() for name,value in zip(('input','output'),pair)})
    result = analyze(predictions,future,metadata,controls)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(controls=controls,sites=result['sites']),indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare,args.checkpoint)
    else:
        run(args.run)
