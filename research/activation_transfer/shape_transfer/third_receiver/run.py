"""Translate the two donor dictionaries' physical removals into Timer without fitting Timer atoms."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import dictionary_path
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as DONOR_JAC
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape, solve_adaptive
from research.activation_transfer.shape_transfer.third_receiver.timer import HERE, LAYER, SCRATCH, adapter_and_selector
from tsfm_lens.sae.train import load_sae_checkpoint

ARMS = ('adaptive', 'wrong_leads', 'mean_donor')
ORDER = np.array([3, 6, 1, 7, 0, 2, 5, 4])


def requests(donor_jac, atoms, cal):
    local = donor_jac @ atoms
    avg_level = donor_jac[cal].mean(axis=1, keepdims=True)
    weight = np.sqrt(np.mean(avg_level**2) / np.mean((donor_jac[cal]-avg_level)**2))
    return local[~cal], local[cal].mean(axis=0), weight


def directions(arm, target_jac, local, average, weight):
    if arm not in ARMS:
        raise ValueError(arm)
    requested = np.broadcast_to(average, local.shape) if arm == 'mean_donor' else local
    operator = target_jac[:, ORDER] if arm == 'wrong_leads' else target_jac
    return solve_adaptive(boost_shape(operator, weight), boost_shape(requested, weight))


def run(source, cache, donor_jac, target_jac, out, batch):
    with np.load(REPLICATION/'results'/(source+'_seed0.npz')) as f:
        ids, labels, codes, atoms, truth = f['ids'], f['split'], f['source_codes'], f['atoms'], f['native']
    with np.load(cache/'data.npz') as f:
        use = f['split'] != 'train'
        np.testing.assert_array_equal(ids, f['ids'][use])
        np.testing.assert_array_equal(labels, f['split'][use])
        x = f['x'][use]
    cal = labels == 'cal'
    with np.load(target_jac/'timer_jac.npz') as f:
        np.testing.assert_array_equal(ids, f['ids'])
        np.testing.assert_array_equal(labels, f['split'])
        tj, ordinary, tokens, radius = f['jac'][~cal].astype(float), f['forecast'][~cal], f['tokens'][~cal], float(f['radius'])
    with np.load(donor_jac/(source+'_jac.npz')) as f:
        np.testing.assert_array_equal(ids, f['ids'])
        np.testing.assert_array_equal(labels, f['split'])
        sj = f['jac'].astype(float)
    decoder = load_sae_checkpoint(str(dictionary_path(source))).W_dec.detach().numpy()[atoms].astype(float).T
    local, average, weight = requests(sj, decoder, cal)
    adapter, selector, _ = adapter_and_selector(x)
    observed = codes[~cal]
    record = dict(ids=ids[~cal], split=labels[~cal], atoms=atoms, codes=observed, active=observed > 0,
                  donor_reference=truth[~cal], shape_weight=weight, radius=radius)
    for arm in ARMS:
        writes = directions(arm, tj, local, average, weight)
        effects, clips = apply(adapter, LAYER, selector, x[~cal], tokens, ordinary, observed, writes, radius, batch)
        record[arm] = effects
        record[arm+'_clipped'] = clips
        print(source, arm, int(clips.sum()), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/(source+'.npz'), **record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('timesfm','bolt'))
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--donor-jac', type=Path, default=DONOR_JAC)
    parser.add_argument('--timer-jac', type=Path, default=SCRATCH)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.source, args.cache, args.donor_jac, args.timer_jac, args.out, args.batch)
