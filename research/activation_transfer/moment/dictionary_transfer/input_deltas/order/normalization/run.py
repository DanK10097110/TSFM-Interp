"""Freeze clean, masked RevIN moments when encoding the two ordered temperature edits."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates import run as prior
from research.activation_transfer.moment.dictionary_transfer.input_deltas.order.run import OUT as ORDER_OUT, PERMUTATION

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
CACHE = Path('/tmp/tsfm_moment_fixed_norm_20261122')
ORDINARY = prior.WORK / 'physical_ramp_confirmation'


def inputs():
    ids, dates, clean = prior.panel()
    assert clean.shape == (35*11, 480)
    reverse = clean.copy()
    reverse[:, -8:] += np.linspace(-2, 2, 8)[PERMUTATION]
    return ids, dates, {'base': clean, 'ramp': prior.modify(clean, 'ramp'), 'permuted': reverse}


def capture_fixed(model, clean, changed, full_forecast=False):
    x, mask = model.prepare(clean)
    edited, actual_mask = model.prepare(changed)
    torch.testing.assert_close(mask, actual_mask, rtol=0, atol=0)
    normalizer = model.module.normalizer
    clean_norm = normalizer(x=x, mask=mask, mode='norm').clone()
    mu, sd = normalizer.mean.clone(), normalizer.stdev.clone()
    normalizer(x=edited, mask=mask, mode='norm')
    actual_sd = normalizer.stdev.clone()
    got = []
    def fix(values, mask):
        assert torch.equal(values, edited) and torch.equal(mask, actual_mask)
        normalizer.mean, normalizer.stdev = mu, sd
    def check(module, args, kwargs, observed):
        if kwargs['mode'] == 'denorm':
            assert full_forecast and torch.equal(module.mean, mu) and torch.equal(module.stdev, sd)
            return
        assert kwargs['mode'] == 'norm'
        expect = (edited - mu) / sd
        if normalizer.affine:
            expect = expect * normalizer.affine_weight + normalizer.affine_bias
        torch.testing.assert_close(observed, expect, rtol=0, atol=0)
        untouched = mask[:, None, :].bool() & torch.eq(x, edited)
        assert torch.all(untouched.sum(-1) >= 472)
        torch.testing.assert_close(observed[untouched], clean_norm[untouched], rtol=0, atol=0)
        got.append(True)
    h = normalizer.register_forward_hook(check, with_kwargs=True)
    try:
        with patch.object(normalizer, '_get_statistics', fix):
            states = model.predict(changed, 16, [.5])['point'] if full_forecast else prior.capture(model, changed)
    finally:
        h.remove()
    assert got == [True], 'Expected the frozen normalizer in the actual encoder path.'
    return states, mu.numpy().ravel(), sd.numpy().ravel(), actual_sd.numpy().ravel()


def original_state(size, label, values):
    if size == 'large' and label == 'base':
        with np.load(prior.EXTERNAL / 'ordinary.npz') as f:
            assert str(f['input_sha256']) == sha256(values.tobytes()).hexdigest()
            return f['last']
    with np.load(ORDINARY / f'{size}_{label}.npz') as f:
        assert str(f['sig']) == sha256(values.tobytes()).hexdigest()
        return f['arr']


def capture_size(size):
    ids, dates, arrays = inputs()
    model = prior.load(size)
    states = {}
    stats = {}
    for label, vals in arrays.items():
        values = [capture_fixed(model, arrays['base'][i:i+prior.BATCH], vals[i:i+prior.BATCH])
                  for i in range(0, len(vals), prior.BATCH)]
        state, mean, clean_sd, native_sd = map(np.concatenate, zip(*values))
        ordinary = original_state(size, label, vals)
        if label == 'base':
            np.testing.assert_array_equal(state, ordinary)
        states[label] = state
        stats[label+'_mean'], stats[label+'_clean_scale'], stats[label+'_native_scale'] = mean, clean_sd, native_sd
        print('Captured', size, label, state.shape, flush=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    OUT.mkdir(exist_ok=True, parents=True)
    np.savez(CACHE / f'{size}.npz', **states)
    np.savez_compressed(OUT / f'{size}_normalizer.npz', ids=ids, dates=dates, **stats)


def physical(size):
    ids, dates, arrays = inputs()
    x = arrays['base']
    with np.load(CACHE / f'{size}.npz') as f:
        moves = {k:f[k]-f['base'] for k in ['ramp','permuted']}
    arm = 'reference' if size == 'large' else 'mapped'
    if size == 'small':
        with np.load(prior.MATCH/'small_to_large_map.npz') as f:
            w = f['w'].astype(float)
        moves = {k: v@w for k,v in moves.items()}
    with np.load(prior.EXTERNAL/'ordinary.npz') as f:
        np.testing.assert_array_equal(ids,f['ids']); np.testing.assert_array_equal(dates,f['dates'])
        base, last = f['base'], f['last']
    full = np.load(prior.WORK/'live_hidden.npy',mmap_mode='r')
    np.testing.assert_array_equal(full[:,61], last)
    model = prior.load('large')
    prior.finite_waves(model,x,full,base,{k+'_'+arm:v for k,v in moves.items()},OUT,ids,dates)


def clean_forecast_check(size):
    ids, dates, arrays = inputs()
    ix = np.linspace(0, len(arrays['base'])-1, 12, dtype=int)
    values = arrays['base'][ix]
    model = prior.load(size)
    ordinary = model.predict(values,16,[.5])['point']
    frozen,*_ = capture_fixed(model,values,values,full_forecast=True)
    np.testing.assert_array_equal(ordinary,frozen)
    (OUT / f'{size}_clean_forecast.json').write_text(json.dumps(dict(indices=ix.tolist(),
        max_abs_C=float(np.max(np.abs(ordinary-frozen)))),indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('size',choices=('small','large')); p.add_argument('stage',choices=('capture','physical','clean'))
    a=p.parse_args(); torch.set_num_threads(2)
    {'capture':capture_size,'physical':physical,'clean':clean_forecast_check}[a.stage](a.size)
