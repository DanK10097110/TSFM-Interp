"""Measure small-model dose predictions by deleting a known feature in the live large model."""
import argparse
import fcntl
import tempfile
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import OUT as DETECTOR
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates import run as original
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle
from research.activation_transfer.moment.token_paths.geometry import layers, load

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
EXPECTED_HASH = 'dae56b90fada336391aef9a4d27f7d8b100fbba13aa823e698543196ca37be45'
ATOMS = np.array([187, 1236, 1867, 1235])
COUNTS = np.array([252, 161, 93, 62])


def digest(x):
    return sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def material():
    ids, dates, x = original.panel()
    dates = np.asarray(dates)
    with np.load(original.OUT/'ordinary.npz') as f:
        last, base, sd = f['last'], f['base'], f['sd']
        np.testing.assert_array_equal(f['ids'], ids)
        np.testing.assert_array_equal(f['dates'], dates)
        assert str(f['input_sha256']) == digest(x) == EXPECTED_HASH
    with np.load(original.OUT/'donor.npz') as f:
        np.testing.assert_array_equal(f['ids'], ids)
        np.testing.assert_array_equal(f['dates'], dates)
        np.testing.assert_array_equal(f['atoms'], ATOMS)
        np.testing.assert_array_equal(f['sd'], sd)
        dose, truth = f['dose'], f['wave']
        source_fit_id = str(f['fit_sha256'])
    np.testing.assert_array_equal(x.std(1), sd)
    np.testing.assert_array_equal((dose > 0).sum(axis=0), COUNTS)
    np.testing.assert_array_equal(truth[dose == 0], 0.)
    with np.load(DETECTOR/'fit.npz') as fit:
        np.testing.assert_array_equal(ATOMS, fit['atoms'])
        fresh, directions = original.policy('donor', last, fit)
    np.testing.assert_array_equal(fresh > 0, dose > 0)
    np.testing.assert_allclose(fresh, dose, atol=1e-5, rtol=1e-6)
    prefix_path = original.WORK/'live_hidden.npy'
    if not prefix_path.exists():
        from tsfm_lens.extraction.hooks import ActivationCatcher
        model = load('large')
        middle = layers(model)['middle']
        pieces = []
        for start in range(0, len(x), 16):
            with ActivationCatcher(model.module, [middle]) as cap:
                model.predict(x[start:start+16], 16, [.5])
                pieces.append(cap.collect()[middle].numpy().copy())
        full = np.concatenate(pieces)
    else:
        full = np.load(prefix_path, mmap_mode='r')
    assert full.shape == (385, 64, 1024)
    np.testing.assert_array_equal(full[:, 61], last)
    return dict(x=x, ids=ids, dates=dates, base=base, prefix=full, sd=sd, truth=truth,
                dose=dose, directions=directions, original_fit_sha=source_fit_id)


def read_doses(path, arm, m):
    if not re.fullmatch('[a-zA-Z][a-zA-Z0-9_]*', arm) or arm.startswith('oracle'):
        raise ValueError(f'Not a source-only arm name: {arm}')
    with np.load(path) as f:
        np.testing.assert_array_equal(f['ids'], m['ids'])
        np.testing.assert_array_equal(f['dates'], m['dates'])
        np.testing.assert_array_equal(f['atoms'], ATOMS)
        values = f[arm]
    if values.shape != m['dose'].shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f'Malformed source-only doses, {arm}')
    return values.astype(np.float32)


def pick_checks(pred, natural):
    # Make the first representatives deterministic for each crossing of true and predicted gates.
    ix = []
    for p in (False, True):
        for t in (False, True):
            pairs = np.argwhere(((pred > 0) == p) & ((natural > 0) == t))
            if len(pairs):
                ix.extend(pairs[np.unique([0, len(pairs)//2, len(pairs)-1])])
    return np.array(ix)


def run_arm(arm, input_file, out, batch, m, model):
    out.mkdir(parents=True, exist_ok=True)
    pred = read_doses(input_file, arm, m)
    sig = digest(pred)
    dest = out/f'causal_{arm}.npz'
    if dest.exists():
        with np.load(dest) as prev:
            assert digest(prev['dose']) == sig and str(prev['input_sha']) == EXPECTED_HASH
        print(arm, 'already complete', flush=True)
        return
    truth = m['dose']
    waves = np.zeros((*pred.shape, 16), dtype='f4')
    measured = np.zeros(pred.shape, dtype=bool)
    # Reuse only exactly equal float32 doses at the same fixed (history, atom).
    same_native = (pred == truth) & (pred > 0)
    waves[same_native] = m['truth'][same_native]
    measured[same_native] = True
    for f in sorted(out.glob('causal_*.npz')):
        with np.load(f) as other:
            assert str(other['input_sha']) == EXPECTED_HASH
            np.testing.assert_array_equal(other['atoms'], ATOMS)
            same = (pred > 0) & ~measured & (other['dose'] == pred)
            if same.any():
                waves[same] = other['wave'][same]
                measured[same] = True
    row, atom = np.nonzero((pred > 0) & ~measured)
    middle = layers(model)['middle']
    assert model.patch_layout(480,16)[0].stop-1 == 61
    print(arm,'live',len(row),'unique writes, reused',int(measured.sum()),flush=True)
    for start in range(0, len(row), batch):
        rr = row[start:start+batch]; aa = atom[start:start+batch]
        moves = -pred[rr, aa, None]*m['directions'][aa]
        with cached_middle(model, m['prefix'][rr], middle), addition(model, moves):
            y = model.predict(m['x'][rr], 16, [.5])['point']
        waves[rr, aa] = y-m['base'][rr]
    np.testing.assert_array_equal(waves[pred == 0], 0.)
    assert np.isfinite(waves).all()
    checks = pick_checks(pred, truth)
    rr, aa = checks.T
    plain = model.predict(m['x'][rr], 16, [.5])['point']
    with addition(model, -pred[rr, aa, None]*m['directions'][aa]):
        full = model.predict(m['x'][rr], 16, [.5])['point']-plain
    with addition(model, -truth[rr, aa, None]*m['directions'][aa]):
        replay_truth = model.predict(m['x'][rr], 16, [.5])['point']-plain
    error = np.max(abs(full-waves[rr, aa]),axis=1)
    true_error = np.max(abs(replay_truth-m['truth'][rr, aa]),axis=1)
    assert max(error.max(), true_error.max()) < .00015
    tmp = dest.with_suffix('.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f, ids=m['ids'], dates=m['dates'], atoms=ATOMS, dose=pred, wave=waves,
                            sd=m['sd'], input_sha=EXPECTED_HASH, directions_sha=digest(m['directions']),
                            dose_sha=sig, original_fit_sha=m['original_fit_sha'], full_checks=checks,
                            full_errors_c=error, native_reference_errors_c=true_error,
                            live_writes=len(row), reused_identical=int(measured.sum()))
    tmp.replace(dest)
    print(arm, 'archived', str(dest), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('arms',nargs='+')
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--inputs',type=Path,default=OUT/'predictions.npz')
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads)
    m=material();model=load('large')
    for arm in args.arms:
        lock_id = sha256(str((args.out/f'causal_{arm}.npz').resolve()).encode()).hexdigest()
        lock_path = Path(tempfile.gettempdir())/('tsfm_cc_forecast_'+lock_id+'.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            run_arm(arm,args.inputs,args.out,args.batch,m,model)
