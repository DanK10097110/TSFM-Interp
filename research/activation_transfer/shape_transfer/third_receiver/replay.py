"""Independently solve and add 24 writes to live Timer activations, one item at a time."""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.replication.data import CACHE
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as DONOR_JAC
from research.activation_transfer.shape_transfer.third_receiver.timer import HERE, LAYER, SCRATCH, adapter_and_selector
from tsfm_lens.sae.train import load_sae_checkpoint


@contextmanager
def live_add(adapter, displacement):
    def callback(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        assert hidden.shape == (1, 5, 1024)
        edited = torch.cat([hidden[:, :-1], hidden[:, -1:] + displacement.reshape(1,1,-1)], dim=1)
        return (edited,*output[1:]) if isinstance(output, tuple) else edited
    handle = dict(adapter.module.named_modules())[LAYER].register_forward_hook(callback)
    try:
        yield
    finally:
        handle.remove()


def replay(cache, donors, scratch, out):
    with np.load(cache/'data.npz') as data:
        by_id = dict(zip(data['ids'], data['x']))
    with np.load(scratch/'timer_jac.npz') as data:
        stored = dict(zip(data['ids'], data['jac']))
    adapter, _, _ = adapter_and_selector(next(iter(by_id.values()))[None])
    report = []
    for donor in ('bolt', 'timesfm'):
        with np.load(out/(donor+'.npz')) as f:
            result = dict(f)
        with np.load(donors/(donor+'_jac.npz')) as d:
            train = d['split'] == 'cal'
            sj = d['jac'][~train].astype(float)
            np.testing.assert_array_equal(d['ids'][~train], result['ids'])
            mean_sj = d['jac'][train].astype(float).mean(axis=0)
        decoder = load_sae_checkpoint(str(dictionary_path(donor))).W_dec.detach().numpy()
        for domain in ('daily', 'machines'):
            active = result['active'] & (result['split'] == domain)[:,None]
            for i in np.flatnonzero(active.any(axis=1))[:2]:
                j = np.flatnonzero(active[i])[0]
                item = result['ids'][i]
                x = by_id[item][None]
                u = torch.zeros(1024, requires_grad=True)
                with live_add(adapter, u):
                    point = adapter.predict_differentiable(x, 8)[0]
                    jac = np.stack([torch.autograd.grad(point[k], u, retain_graph=k<7)[0].numpy() for k in range(8)])
                np.testing.assert_allclose(jac, stored[item], atol=2e-5, rtol=.0001)
                ordinary = adapter.predict(x, 8, [.5])['point'][0]
                atom = decoder[result['atoms'][j]].astype(float)
                weight = result['shape_weight']
                # Orthogonal constant/shape projector reproduces weighting without the production transform.
                dc = np.ones((8,8))/8
                w = dc + weight * (np.eye(8)-dc)
                for arm in ('adaptive', 'mean_donor', 'wrong_leads'):
                    order = [3,6,1,7,0,2,5,4] if arm == 'wrong_leads' else list(range(8))
                    design = w @ jac[order]
                    request = w @ ((mean_sj if arm == 'mean_donor' else sj[i]) @ atom)
                    gram = design @ design.T
                    ridge = .001 * np.linalg.svd(design, compute_uv=False)[0]**2
                    delta = -result['codes'][i,j]*(design.T @ np.linalg.solve(gram + ridge*np.eye(8), request))
                    norm = np.linalg.norm(delta)
                    capped = norm > .2*result['radius']
                    assert bool(capped) == bool(result[arm+'_clipped'][i,j])
                    if capped:
                        delta *= .2*result['radius']/norm
                    with live_add(adapter, torch.tensor(delta, dtype=torch.float32)):
                        actual = adapter.predict(x, 8, [.5])['point'][0] - ordinary
                    expected = result[arm][i,j]
                    np.testing.assert_allclose(actual, expected, atol=6e-5, rtol=3e-3)
                    report.append(dict(donor=donor, domain=domain, id=str(item), atom=int(result['atoms'][j]),
                                       arm=arm, max_abs=float(abs(expected-actual).max()),
                                       jac_max_abs=float(abs(stored[item]-jac).max()), capped=bool(capped)))
    assert len(report) == 24
    (out/'replay.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--donor-jac', type=Path, default=DONOR_JAC)
    parser.add_argument('--timer-jac', type=Path, default=SCRATCH)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    args = parser.parse_args()
    torch.set_num_threads(1)
    replay(args.cache, args.donor_jac, args.timer_jac, args.out)
