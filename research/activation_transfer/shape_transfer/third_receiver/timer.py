"""Collect Timer's last native token and its eight-lead post-block Jacobian."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import HUB
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.forecast_requests.run import MODELS
from research.activation_transfer.replication.data import CACHE
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.models import build_adapter

HERE = Path(__file__).resolve().parent
SCRATCH = Path('/tmp/tsfm_sae_timer_third_20261120')
LAYER = 'model.layers.4'


def adapter_and_selector(first):
    kind, repo, revision = MODELS['timer']
    checkpoint = HUB / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
    config = ModelConfig(name='timer', adapter=kind, checkpoint=str(checkpoint),
                         kwargs={'auto_class': 'AutoModelForCausalLM'}, batch_size=32)
    adapter = build_adapter(config, DataConfig(context_len=480, horizon=8), torch.device('cpu'), torch.float32)
    adapter.ensure_loaded()
    adapter.predict(first[:1], 8, [.5])
    strategies = adapter.describe_strategies()
    assert strategies['forecast'] == 'forward_field' and strategies['quantiles'] == 'degenerate_point'
    spans = np.asarray(adapter.token_time_spans())
    np.testing.assert_array_equal(spans, np.column_stack((np.arange(0,480,96), np.arange(96,481,96))))
    layers = adapter.all_layer_names()
    assert layers[len(layers)//2] == LAYER
    return adapter, last_window_slice(adapter, np.array([len(spans)-1])), spans


def capture(adapter, x, batch):
    tokens = [capture_raw_tokens(adapter, x[i:i+batch], [LAYER])[LAYER][:,-1:].numpy()
              for i in range(0, len(x), batch)]
    result = np.concatenate(tokens)
    assert result.shape == (len(x), 1, 1024)
    return result


def gradient(adapter, selector, x, tokens, batch):
    gradients, outputs = [], []
    for i in range(0, len(x), batch):
        h = torch.from_numpy(tokens[i:i+batch])
        delta = torch.zeros_like(h[:, 0], requires_grad=True)
        with torch.enable_grad(), token_patch(adapter.module, LAYER, selector, h + delta[:, None]):
            point = adapter.predict_differentiable(x[i:i+len(h)], 8)
            rows = [torch.autograd.grad(point[:, t].sum(), delta, retain_graph=t < 7)[0] for t in range(8)]
        gradients.append(torch.stack(rows, dim=1).detach().numpy())
        outputs.append(point.detach().numpy())
    return np.concatenate(gradients), np.concatenate(outputs)


def collect(cache, scratch, out, batch):
    with np.load(cache/'data.npz') as a:
        x, ids, labels = a['x'], a['ids'], a['split']
    adapter, selector, spans = adapter_and_selector(x)
    print('resolved', LAYER, spans[-1].tolist(), adapter.describe_strategies()['forecast'], flush=True)
    tokens = capture(adapter, x, batch)
    train = labels == 'train'
    assert train.sum() == 1024
    radius = np.median(np.linalg.norm(tokens[train, 0], axis=1))
    x, tokens, ids, labels = x[~train], tokens[~train], ids[~train], labels[~train]
    jac, patched = gradient(adapter, selector, x, tokens, batch)
    ordinary = change_forecasts(adapter, LAYER, x, tokens, None, selector, batch)
    np.testing.assert_allclose(patched, ordinary, atol=8e-5, rtol=2e-5)
    checks = np.concatenate([np.flatnonzero(labels == s)[:8] for s in ('cal', 'daily', 'machines')])
    rng = np.random.default_rng(8903)
    u = rng.normal(size=(len(checks), tokens.shape[-1])).astype('f4')
    u *= (.002 * radius/np.linalg.norm(u, axis=1))[:, None]
    high, low = [change_forecasts(adapter, LAYER, x[checks], tokens[checks], sign*u, selector, batch)
                 for sign in (1, -1)]
    measured = (high-low)/2
    predicted = np.einsum('nhd,nd->nh', jac[checks], u)
    fd_error = np.linalg.norm(measured-predicted)/np.linalg.norm(measured)
    assert np.isfinite(jac).all() and fd_error < .08, fd_error
    scratch.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scratch/'timer_jac.npz', ids=ids, split=labels, jac=jac, tokens=tokens,
                        radius=radius, forecast=ordinary)
    np.savez_compressed(out/'timer_validation.npz', ids=ids, split=labels, ordinary=ordinary, fd_ids=ids[checks],
                        measured=measured, predicted=predicted)
    meta = dict(checkpoint=MODELS['timer'], site=LAYER, site_side='output', token_spans=spans.tolist(),
                patch_index=4, selected_span=spans[-1].tolist(), radius=float(radius), cap=float(.2*radius),
                patch_replay_max=float(abs(ordinary-patched).max()), fd_relative_error=float(fd_error),
                n_jacobians=len(x), horizon=8, width=jac.shape[2], patch_count=1,
                point_strategy=adapter.describe_strategies()['forecast'])
    (out/'timer.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('central-difference error', fd_error, 'radius', radius, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--scratch', type=Path, default=SCRATCH)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    collect(args.cache, args.scratch, args.out, args.batch)
