"""History-only normalizer metadata and fair coefficient decoding from saved states.

No model weights are loaded and no native inference is performed. Bolt's scalar
normalizer and TimesFM's four cumulative patch normalizers are applied with the
installed native kernels. Targets, covariates and inverse maps stay separate.
"""
from __future__ import annotations
import argparse
import inspect
from pathlib import Path
import time
import numpy as np
from experiments.feature_search.coherence_study import read, write, sha

CENTERS = (128/24, 128/19.2)
ROUTES = ('physical_delta', 'physical_pair', 'normalized_delta', 'normalized_pair')


def project(values, spans):
    """Project each normalization span onto two fixed sine/cos pairs plus a DC nuisance."""
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != 128 or spans not in (1, 4):
        raise ValueError('Expected128 samples and1 or4 normalization spans')
    output = []
    for indices in np.array_split(np.arange(128), spans):
        t = indices-128
        design = np.column_stack([v for f in CENTERS for v in
                                  (np.sin(2*np.pi*f*t/128), np.cos(2*np.pi*f*t/128))]+[np.ones(len(t))])
        output.append((values[:, indices]@np.linalg.pinv(design).T)[:, :4])
    return np.stack(output, axis=1)


def native_metadata(history, model):
    """Compute actual native FP32 history-only stats without instantiating a forecaster."""
    import torch
    torch.set_num_threads(1)
    x = torch.as_tensor(np.asarray(history, np.float32).copy())
    if model == 'bolt':
        from chronos.chronos_bolt import InstanceNorm
        z, (mu, sigma) = InstanceNorm()(x)
    elif model == 'timesfm':
        from timesfm.torch.util import update_running_stats, revin
        patches = x.reshape(-1, 4, 32)
        n = torch.zeros(len(x)); mean = torch.zeros_like(n); scale = torch.zeros_like(n)
        mus, sigmas = [], []
        for i in range(4):
            (n, mean, scale), _ = update_running_stats(n, mean, scale, patches[:, i], torch.zeros_like(patches[:, i], dtype=torch.bool))
            mus.append(mean); sigmas.append(scale)
        mu, sigma = torch.stack(mus, 1), torch.stack(sigmas, 1)
        z = revin(patches, mu, sigma).reshape(-1, 128)
    else: raise ValueError('Unknown model')
    if torch.any(sigma < 1e-6): raise ValueError('Degenerate normalization: inverse would be asymmetric')
    return z.numpy().astype(float), mu.numpy().astype(float), sigma.numpy().astype(float)


def normalize_with(history, mu, sigma):
    """Replay fixed metadata on its original FP32 input, matching native broadcast arithmetic."""
    import torch
    x = torch.as_tensor(np.asarray(history, np.float32).copy()).reshape(len(history), mu.shape[1], -1)
    return ((x-torch.as_tensor(mu, dtype=torch.float32)[..., None])/
            torch.as_tensor(sigma, dtype=torch.float32)[..., None]).reshape(-1, 128).numpy().astype(float)


def restore(prediction, route, sigma_q, sigma_b, background_coeff):
    """Invert normalized predictions with history-only metadata, never generating labels.

For normalized differences, the known background projection is necessary:
sq*(Zq-Zb)+(sq/sb-1)*Cb. Paired endpoint decoding avoids this observed-coefficient
correction entirely: sq*Zq-sb*Zb. All baselines share the same inverse map.
"""
    prediction = np.asarray(prediction, float)
    if route.startswith('physical'): return prediction
    spans = sigma_q.shape[1]
    if route == 'normalized_delta':
        normalized = prediction.reshape(-1, spans, 4)
        physical = sigma_q[..., None]*normalized+(sigma_q/sigma_b-1)[..., None]*background_coeff
    elif route == 'normalized_pair':
        normalized = prediction.reshape(-1, 2, spans, 4)
        physical = sigma_q[..., None]*normalized[:, 0]-sigma_b[..., None]*normalized[:, 1]
    else: raise ValueError('Unknown inverse route')
    return physical.mean(axis=1)


def ridge_grouped(blocks, targets, groups, alpha=.01):
    """Frozen ridge with train-only block scaling and whole-source holdouts.

Equal trace-scale block kernels prevent a tiny metadata block from disappearing
beside thousands of hidden coordinates. A training-row permutation of the first
block is an incremental feature control; all other blocks remain aligned.
"""
    blocks = [np.asarray(block, float) for block in blocks]
    targets, groups = np.asarray(targets, float), np.asarray(groups)
    prediction = np.zeros_like(targets); shuffled = np.zeros_like(targets); means = np.zeros_like(targets)
    folds = []
    for source in sorted(set(groups)):
        test = groups == source; train = ~test
        gram = np.zeros((train.sum(), train.sum())); cross = np.zeros((test.sum(), train.sum()))
        gram_null = np.zeros_like(gram); cross_null = np.zeros_like(cross)
        rng = np.random.default_rng(4953000+int(source)); order = rng.permutation(train.sum())
        scales = []
        for i, block in enumerate(blocks):
            mean = block[train].mean(0); x = block[train]-mean; xt = block[test]-mean
            scale = float(np.sqrt(np.mean(x*x)))
            if scale == 0: scale = 1.
            scales.append(scale); x, xt = x/scale, xt/scale
            g, c = x@x.T/x.shape[1], xt@x.T/x.shape[1]
            gram += g/len(blocks); cross += c/len(blocks)
            if i == 0: g, c = g[order][:, order], c[:, order]
            gram_null += g/len(blocks); cross_null += c/len(blocks)
        ym = targets[train].mean(0); yc = targets[train]-ym
        prediction[test] = cross@np.linalg.solve(gram+alpha*np.eye(train.sum()), yc)+ym
        shuffled[test] = cross_null@np.linalg.solve(gram_null+alpha*np.eye(train.sum()), yc)+ym
        means[test] = ym
        folds.append(dict(source_id=int(source), train_sources=[int(g) for g in sorted(set(groups[train]))],
                          rows=int(test.sum()), block_scales=scales))
    return prediction, shuffled, means, folds


def scores(prediction, truth, physical_mean, groups):
    """Evaluate every route in identical physical-coefficient units and source groups."""
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    error = (prediction-truth)**2; reference = (physical_mean-truth)**2
    return dict(mse=float(error.mean()), nmse=float(error.mean()/reference.mean()),
                source_mse={str(g):float(error[np.asarray(groups) == g].mean()) for g in sorted(set(groups))})


def prepare(out, prior):
    """Freeze all reused arrays, source-exact native normalization code and decoder choices."""
    from chronos import chronos_bolt
    from timesfm.torch import util
    from timesfm.timesfm_2p5 import timesfm_2p5_torch
    files = [Path(__file__), Path(inspect.getfile(chronos_bolt)), Path(inspect.getfile(util)),
             Path(inspect.getfile(timesfm_2p5_torch)), Path(__file__).resolve().parents[2]/'tests/test_response_surface_normalization.py']
    inputs = ['protocol.json', 'inputs.npy', 'decoder_rows.json', 'rows.json', 'execution.json']
    for model in ('bolt', 'timesfm'):
        worker = read(prior/f'{model}_worker.json')
        inputs.append(f'{model}_worker.json')
        inputs.extend(f'{model}_{policy}_hidden_{cut}.npy' for policy in ('native', 'locked') for cut in worker['cuts'])
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(question='Does omitted native-normalizer information explain native physical-coefficient decoding error?',
        prior=str(prior.resolve()), rows=120, sources=8, alpha=.01, centers=list(CENTERS), routes=list(ROUTES),
        normalization='Actual installed native FP32 kernels: Bolt whole128; TimesFM cumulative32/64/96/128 prefix stats, applied patchwise. No weights or forward passes.',
        target='Physical injection coefficients derived by joint fixed-frequency Fourier projection of query-minus-background with DC nuisance. Normalized targets project actual normalized endpoint differences or paired endpoints within normalization spans.',
        inverse='normalized_delta: sq*pred+(sq/sb-1)*Cb with observed past-only background coefficients. normalized_pair: sq*predq-sb*predb, no background-coefficient correction. Average restored patch coefficients.',
        features='Only observed raw/normalized histories, saved hidden states, and endpoint native means/scales. No generating amplitude, phase, source identity, condition labels or coefficient labels as covariates.',
        controls='Matched input delta/pair with same target and inverse; metadata-only; hidden plus metadata; shuffled first feature block with metadata unchanged; training-mean and zero normalized predictions restored identically.',
        cv='Leave-one-source-out; all dependent rows held out together. Fixed alpha=.01; train-only scalar RMS per block; equal block-kernel weight; no tuning or p values.',
        primary='Native first-block physical_delta versus normalized_delta and normalized_pair, hidden-only versus hidden+metadata, with matched inputs and inverse-only controls; all other cuts and locked policy descriptive.',
        limits='Reused8 instrumentation sources; phase/source confounding persists. Decoder failure cannot establish missing information; a corrected decoder does not establish causal semantics or training attribution.',
        source_hashes={str(f.resolve()):sha(f) for f in files}, prior_hashes={name:sha(prior/name) for name in inputs},
        native_capture='NONE: no model checkpoint or native forecaster instantiated')
    write(out/'protocol.json', protocol)
    write(out/'knowledge.json', dict(status='prepared', question=protocol['question'], design=protocol,
          exact_findings=[], controls=protocol['controls'], limitations=protocol['limits'],
          evidence_class='Prospective descriptive source-held-out decoder comparison', next_decision='Run saved-array-only analysis; native capture remains held.'))


def run(out):
    """Reuse native arrays, persist all predictions, and distill results immediately."""
    started = time.monotonic(); p = read(out/'protocol.json'); prior = Path(p['prior'])
    for name, expected in p['source_hashes'].items():
        if sha(name) != expected: raise ValueError(f'Frozen source changed {name}')
    for name, expected in p['prior_hashes'].items():
        if sha(prior/name) != expected: raise ValueError(f'Bound prior changed {name}')
    if (out/'results.json').exists(): raise ValueError('Refuse overwrite completed experiment')
    rows = read(prior/'decoder_rows.json'); inputs = np.load(prior/'inputs.npy', mmap_mode='r')
    ids = np.asarray([r['index'] for r in rows]); base_ids = np.asarray([r['base'] for r in rows]); groups = np.asarray([r['source_id'] for r in rows])
    query, base = inputs[ids].astype(float), inputs[base_ids].astype(float)
    physical = project(query-base, 1)[:, 0]
    reference = np.asarray([np.asarray(r['coefficient'])*r['scale'] for r in rows])
    label_check = float(abs(physical-reference).max())
    if label_check > 2e-6: raise ValueError('History-derived physical coefficients disagree with known signal')
    physical_mean = np.zeros_like(physical)
    for group in sorted(set(groups)): physical_mean[groups == group] = physical[groups != group].mean(0)
    results, artifacts, gates = [], {}, dict(physical_label_max_error=label_check, normalized_replay_max=0., inverse_oracle_max=0.)
    for model in ('bolt', 'timesfm'):
        zq_native, mq_native, sq_native = native_metadata(query, model)
        zb, mb, sb = native_metadata(base, model)
        spans = sq_native.shape[1]; background_coeff = project(base, spans)
        for policy in ('native', 'locked'):
            mq, sq = (mq_native, sq_native) if policy == 'native' else (mb, sb)
            zq = normalize_with(query, mq, sq)
            if policy == 'native':
                replay = float(abs(zq-zq_native).max()); gates['normalized_replay_max'] = max(gates['normalized_replay_max'], replay)
                if replay != 0: raise ValueError('Native FP32 normalizer replay differs')
            cq, cb = project(zq, spans), project(zb, spans)
            metadata = np.concatenate([mq, sq, mb, sb], axis=1)
            targets = dict(physical_delta=physical, physical_pair=physical,
                           normalized_delta=(cq-cb).reshape(len(rows), -1),
                           normalized_pair=np.stack([cq, cb], 1).reshape(len(rows), -1))
            input_features = dict(physical_delta=query-base, physical_pair=np.concatenate([query, base], 1),
                                  normalized_delta=zq-zb, normalized_pair=np.concatenate([zq, zb], 1))
            artifacts.update({f'{model}_{policy}_{name}':value for name,value in
                [('query_mu',mq),('query_sigma',sq),('base_mu',mb),('base_sigma',sb),('normalized_query',zq),('normalized_base',zb)]})
            hidden = {}
            for cut in read(prior/f'{model}_worker.json')['cuts']:
                values = np.load(prior/f'{model}_{policy}_hidden_{cut}.npy', mmap_mode='r')
                hq, hb = values[ids].astype(float).reshape(len(rows), -1), values[base_ids].astype(float).reshape(len(rows), -1)
                hidden[cut] = dict(delta=hq-hb, pair=np.concatenate([hq, hb], 1))
            for route, target in targets.items():
                oracle = restore(target, route, sq, sb, background_coeff)
                error = float(abs(oracle-physical).max()); gates['inverse_oracle_max'] = max(gates['inverse_oracle_max'], error)
                if error > 5e-6: raise ValueError(f'Inverse target closure failed {error}')
                feature_sets = [('input', 'input', [input_features[route]]),
                                ('input_metadata', 'input', [input_features[route], metadata]),
                                ('metadata_only', 'metadata', [metadata])]
                suffix = route.split('_')[1]
                for cut in hidden:
                    feature_sets.extend([('hidden', cut, [hidden[cut][suffix]]),
                                         ('hidden_metadata', cut, [hidden[cut][suffix], metadata])])
                for feature, cut, blocks in feature_sets:
                    pred, shuffled, mean, folds = ridge_grouped(blocks, target, groups, p['alpha'])
                    restored = [restore(v, route, sq, sb, background_coeff) for v in (pred, shuffled, mean, np.zeros_like(pred))]
                    key = f'{model}_{policy}_{route}_{feature}_{cut}'
                    np.savez_compressed(out/f'{key}.npz', prediction=pred, target=target, physical_prediction=restored[0],
                        physical_truth=physical, physical_shuffled=restored[1], physical_mean_control=restored[2],
                        physical_zero_control=restored[3], source_ids=groups, row_ids=ids, base_ids=base_ids)
                    result = dict(model=model, policy=policy, route=route, feature=feature, cut=cut,
                        physical=scores(restored[0],physical,physical_mean,groups), shuffled=scores(restored[1],physical,physical_mean,groups),
                        inverse_mean_control=scores(restored[2],physical,physical_mean,groups), inverse_zero_control=scores(restored[3],physical,physical_mean,groups),
                        normalized_target_mse=float(np.mean((pred-target)**2)), folds=folds)
                    results.append(result)
        print(dict(model=model, completed_fits=len(results), seconds=time.monotonic()-started), flush=True)
    np.savez_compressed(out/'normalization_metadata.npz', **artifacts)
    write(out/'results.json', dict(gates=gates, rows=results, seconds=time.monotonic()-started))
    primary = [{k:r[k] for k in ('model','policy','route','feature','cut','physical','shuffled','inverse_mean_control','inverse_zero_control')}
               for r in results if r['cut'] in ('0','input','metadata')]
    write(out/'knowledge.json', dict(status='complete_saved_array_followup',question=p['question'],design=p,
        exact_findings=primary,controls=dict(gates=gates,description=p['controls']),limitations=p['limits'],
        evidence_class='Descriptive8-source held-out linear decodability with history-only inverse normalization; not causal feature semantics',
        failed_instrumentation_runs=[],next_decision='Review direct/normalized and endpoint/difference routes against matched input and inverse-only controls. Prepare crossed-phase smaller-radius design; HOLD native capture.',
        seconds=time.monotonic()-started,native_capture_performed=False))
    write(out/'manifest.json', {f.name:sha(f) for f in out.iterdir() if f.suffix in ('.json','.npz') and f.name != 'manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--prior', type=Path, default=Path('runs/response_surface_pilot4951000'))
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out,args.prior)
    else:
        try: run(args.out)
        except Exception as error:
            old = read(args.out/'knowledge.json'); old.update(status='failed_instrumentation',failed_instrumentation_runs=[dict(error=repr(error))],next_decision='Repair only new followup source; preserve failed artifacts and prior frozen sources.')
            write(args.out/'knowledge.json',old)
            raise
