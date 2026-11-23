"""Transfer frozen donor removals through two independently trained receiving dictionaries."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.groups import pursuit
from research.activation_transfer.moment.dictionary_transfer.match import dictionary, encode, hidden
from research.activation_transfer.moment.dictionary_transfer.more_seeds.fit import HERE, OUT, SCRATCH, SEEDS
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle, obtain
from research.activation_transfer.moment.dictionary_transfer.weather.study import CACHE as DEFAULT_WEATHER_CACHE
from research.activation_transfer.moment.dictionary_transfer.weather.study import OUT as WEATHER_OUT
from research.activation_transfer.moment.dictionary_transfer.weather.study import file_hash, histories, live
from research.activation_transfer.moment.token_paths.geometry import load
from tsfm_lens.extraction.hooks import ActivationCatcher, token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

SOURCE = 22112201
ORIGINAL = 22112202
ARMS = ('corr_own', 'corr_forced', 'group16')
WEATHER_CACHE = Path(os.environ.get('TSFM_MOMENT_MORE_SEEDS_WEATHER_CACHE', DEFAULT_WEATHER_CACHE))


def receiver(seed):
    if seed == ORIGINAL:
        return dictionary('large', seed)
    return load_sae_checkpoint(str(SCRATCH / f'sae_{seed}.pt'))


def original_data():
    with np.load(WEATHER_OUT / f'{SOURCE}.npz') as f:
        return dict(f)


def prepare(seed):
    OUT.mkdir(exist_ok=True, parents=True)
    original = original_data()
    atoms = original['atoms']
    with np.load(HERE.parents[1] / f'horizon_identity/results/large_{SOURCE}_dictionary.npz') as f:
        np.testing.assert_array_equal(atoms, f['ids'])
        axes = f['decoder'].astype(float)
    src, target = dictionary('large', SOURCE), receiver(seed)
    training = hidden('large')[:, -1]
    src_codes = encode(src, training)[:, atoms].astype(float)
    tgt_codes = encode(target, training).astype(float)
    eligible = np.flatnonzero((tgt_codes > 0).sum(0) >= 8)
    a = src_codes - src_codes.mean(0)
    b = tgt_codes[:, eligible]
    b -= b.mean(0)
    denom = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)
    corr = np.divide(a.T @ b, denom, out=np.zeros_like(denom), where=denom > 0)
    cols = corr.argmax(1)
    ids = eligible[cols]
    dec = target.W_dec.detach().numpy().astype(float)
    np.testing.assert_allclose(np.linalg.norm(dec, axis=1), 1, atol=1e-4)
    eligible_dec = dec[eligible]
    groups = [pursuit(axis, eligible_dec, eligible)[16] for axis in axes]
    g_ids, g_coeff, g_vec = map(np.stack, zip(*groups))
    np.testing.assert_allclose(g_vec, np.einsum('nk,nkd->nd', g_coeff, dec[g_ids]), atol=1e-6)
    # The original weather sweep encoded weather with one PyTorch CPU thread.
    torch.set_num_threads(1)
    with np.load(WEATHER_CACHE / 'clean.npz') as f:
        np.testing.assert_array_equal(f['ids'], original['ids'])
        np.testing.assert_allclose(encode(src, f['middle'])[:, atoms], original['z'], atol=1e-6)
        weather_code = encode(target, f['middle'])[:, ids]
    result = dict(atoms=atoms, matched_ids=ids, corr=corr[np.arange(len(atoms)), cols], target_z=weather_code,
                  group_ids=g_ids, group_coeff=g_coeff, group_vector=g_vec, single_vector=dec[ids],
                  group_r2=1 - np.sum((axes-g_vec)**2, axis=1)/np.sum(axes**2, axis=1),
                  cos_at_corr=np.einsum('ij,ij->i', axes, dec[ids]) / np.linalg.norm(axes, axis=1),
                  recipient_seed=seed, eligible_receiver_ids=eligible)
    if seed == ORIGINAL:
        for k in ('matched_ids', 'group_ids', 'atoms'):
            np.testing.assert_array_equal(result[k], original[k])
        for k in ('corr', 'target_z', 'group_coeff', 'group_r2', 'cos_at_corr'):
            np.testing.assert_allclose(result[k], original[k], atol=1e-6)
    else:
        result['recipient_sae_sha256'] = file_hash(SCRATCH / f'sae_{seed}.pt')
    np.savez_compressed(OUT / f'match_{seed}.npz', **result)
    print('Matched', seed, len(eligible), flush=True)


def displacements(original, match):
    r, c = original['rows'], original['cols']
    np.testing.assert_equal((r, c), np.nonzero(original['z'] > 0))
    single = match['single_vector'][c]
    return {'corr_own': -match['target_z'][r, c, None] * single,
            'corr_forced': -original['donor_norm'][:, None] * single / np.linalg.norm(single, axis=1, keepdims=True),
            'group16': -original['z'][r, c, None] * match['group_vector'][c]}


def independent_native_check(model, seed, original, match, arm, x, waves):
    src, tgt = dictionary('large', SOURCE), receiver(seed)
    site = str(original['site'])
    r, c = original['rows'], original['cols']
    on = match['target_z'][r, c] > 0
    # Systematic positions span the station roster; the two gate checks cover native dosage.
    checks = np.unique(np.r_[0, len(on)//2, len(on)-1, np.flatnonzero(on)[:1], np.flatnonzero(~on)[:1]])
    gaps = []
    for i in checks:
        row, col = int(r[i]), int(c[i])
        atom, target = int(match['atoms'][col]), int(match['matched_ids'][col])
        sample = x[row:row+1]
        with ActivationCatcher(model.module, [site]) as capture:
            base = model.predict(sample, 16, [.5])['point'][0].astype(float)
            state = capture.collect()[site]
        np.testing.assert_allclose(base, original['base'][row], atol=.00015, rtol=1e-5)
        slot = model.token_slice(state.shape[1]).stop - 1
        assert slot == int(original['pos'])
        old = state[:, slot]
        with torch.inference_mode():
            sz = src.encode(old)[0, atom].item()
            tz = tgt.encode(old)[0, target].item()
        np.testing.assert_allclose(sz, original['z'][row, col], atol=3e-4, rtol=3e-5)
        np.testing.assert_allclose(tz, match['target_z'][row, col], atol=3e-4, rtol=3e-5)
        axis = tgt.W_dec[target].detach().numpy().astype(float)
        if arm == 'corr_own':
            delta = -tz * axis
        elif arm == 'corr_forced':
            norm = sz * np.linalg.norm(src.W_dec[atom].detach().numpy().astype(float))
            delta = -norm * axis / np.linalg.norm(axis)
        else:
            axes = tgt.W_dec.detach().numpy()[match['group_ids'][col]].astype(float)
            delta = -sz * (match['group_coeff'][col] @ axes)
        fixed = (old + torch.from_numpy(delta.astype('f4')))[:, None]
        with token_patch(model.module, site, lambda n: slice(slot, slot+1), fixed):
            patched = model.predict(sample, 16, [.5])['point'][0].astype(float)
        observed = patched-base
        np.testing.assert_allclose(observed, waves[i], atol=.00025, rtol=.002)
        gaps.append(dict(event=int(i), station_date=str(original['ids'][row]), atom=atom, target=target,
                         local_target_active=bool(on[i]), max_abs_celsius=float(np.max(np.abs(observed-waves[i])))))
    return gaps


def physical(seed, arms):
    orig = original_data()
    with np.load(OUT / f'match_{seed}.npz') as f:
        match = dict(f)
    with np.load(WEATHER_CACHE / 'clean.npz') as f:
        clean = dict(f)
    x, ids, *_ = histories()
    np.testing.assert_array_equal(ids, orig['ids'])
    site, pos = str(orig['site']), int(orig['pos'])
    model = load('large')
    sig = sha256(np.ascontiguousarray(clean['middle']).tobytes() + str(clean['checkpoint_sha256']).encode()
                 + str(clean['context_sha256']).encode()).hexdigest()
    # Avoid touching the original study's lock file. The existing prefix has a signed provenance key.
    prefix_path = WEATHER_CACHE / 'ordinary_prefix.npz'
    if prefix_path.exists():
        with np.load(prefix_path) as f:
            assert str(f['signature']) == sig
            prefix = f['states']
    else:
        prefix = obtain(model, x, site, pos, SCRATCH, sig, 24)
    np.testing.assert_allclose(prefix[:, pos], clean['middle'], atol=3e-5, rtol=1e-6)
    r = orig['rows']
    moves = displacements(orig, match)
    batch = 24
    for arm in arms:
        d = moves[arm].astype('f4')
        waves, shadow = [], []
        for start in range(0, len(r), batch):
            which = slice(start, start+batch)
            rows = r[which]
            with cached_middle(model, prefix[rows], site):
                y, final, observed, sd = live(model, x[rows], d[which])
            np.testing.assert_allclose(observed, clean['middle'][rows], atol=5e-4, rtol=3e-5)
            np.testing.assert_allclose(sd, clean['scale'][rows], rtol=2e-6)
            waves.append(y.astype(float) - clean['base'][rows] - orig['noop'][which])
            read = (final.astype(float)-clean['final'][rows]) @ clean['head'].astype(float).T
            shadow.append(read.reshape(-1, 16) * sd[:, None] - orig['noop_head'][which])
            if start % (batch * 8) == 0:
                print('Physical', seed, arm, start, '/', len(r), flush=True)
        waves, shadow = np.concatenate(waves), np.concatenate(shadow)
        np.testing.assert_allclose(waves, shadow, atol=.00015, rtol=.001)
        check = independent_native_check(model, seed, orig, match, arm, x, waves)
        np.savez_compressed(OUT / f'{seed}_{arm}.npz', effect=waves, rows=r, cols=orig['cols'],
                            hidden_norm=np.linalg.norm(d.astype(float), axis=1), shadow=shadow)
        (OUT / f'check_{seed}_{arm}.json').write_text(json.dumps(check, indent=2) + '\n')
        print('Complete', seed, arm, max(v['max_abs_celsius'] for v in check), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('seed', type=int, choices=(ORIGINAL, *SEEDS))
    p.add_argument('stage', choices=('prepare', 'all', *ARMS))
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'prepare':
        prepare(args.seed)
    else:
        # Match the original weather baseline, including floating-point reduction order.
        torch.set_num_threads(1)
        physical(args.seed, ARMS if args.stage == 'all' else [args.stage])
