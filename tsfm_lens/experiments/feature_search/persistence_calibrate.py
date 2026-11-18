"""Calibrate exact native identity across singleton calls and batch positions.

No tolerance gate is substituted. Full forecasts and exact/absolute/relative
differences distinguish stable per-position arithmetic from changing predictions
on repeated identical calls. This is instrumentation, not a forecast study.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from experiments.feature_search.persistence_protocol import ENV, finite, read, sha, write
from experiments.feature_search.persistence_study import bound, target_guard
from experiments.feature_search.persistence_execution import execute


def difference(a, b):
    """Report exact equality and errors relative to the reference forecast RMS."""
    a, b = finite(a).astype(float), finite(b).astype(float)
    delta = a-b
    scale = max(float(np.sqrt(np.mean(b*b))), 1e-12)
    return dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.max(abs(delta))),
                rms_abs=float(np.sqrt(np.mean(delta*delta))), reference_rms=scale,
                max_abs_over_reference_rms=float(np.max(abs(delta))/scale),
                rms_relative=float(np.sqrt(np.mean(delta*delta))/scale))


def prepare(out, pilot):
    """Freeze instrument code and source histories before calibration inference."""
    p = bound(pilot)
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name('persistence_execution.py')]
    p.update(pilot=str(pilot.resolve()), query_ids=[0, 1, 4, 5],
             sources=dict(p['sources'], **{str(f.resolve()): sha(f) for f in paths}),
             tests=['singleton_repeat', 'duplicate_batch_repeat', 'duplicate_positions',
                    'singleton_to_duplicate_position0', 'singleton_to_duplicate_position1',
                    'mixed_batch_repeat', 'mixed_position_change', 'singleton_to_mixed_position0'],
             policy='No tolerances; diagnose original exact gate failure. No target reads.')
    write(out/'protocol.json', p)
    write(out/'seal.json', dict(protocol_sha256=sha(out/'protocol.json')))


def worker(out, model):
    """Collect fixed queries under repeated and reordered native batch layouts."""
    import os
    p = read(out/'protocol.json')
    if sha(out/'protocol.json') != read(out/'seal.json')['protocol_sha256']:
        raise ValueError('Calibration protocol changed')
    pilot = Path(p['pilot'])
    sys.addaudithook(target_guard(pilot/'evaluation'))
    bound(pilot)
    for path, digest in p['sources'].items():
        if sha(path) != digest:
            raise ValueError('Calibration source changed')
    for name, value in ENV.items():
        if os.environ.get(name) != value:
            raise ValueError('Environment mismatch')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from experiments.feature_search.real_background_poc import Native
    native = Native(model, p['checkpoints'][model]['path'])
    x = np.load(pilot/'inference/inputs.npz')['history']
    saved, comparisons = {}, []
    for i in p['query_ids']:
        query = x[i:i+1]; other = x[(i+1) % len(x):(i+1) % len(x)+1]
        singleton = native.predict(query)[0][0]
        singleton_repeat = native.predict(query)[0][0]
        duplicate = native.predict(np.repeat(query, 2, axis=0))[0]
        duplicate_repeat = native.predict(np.repeat(query, 2, axis=0))[0]
        mixed = native.predict(np.concatenate([query, other]))[0]
        mixed_repeat = native.predict(np.concatenate([query, other]))[0]
        reversed_mixed = native.predict(np.concatenate([other, query]))[0]
        values = dict(singleton=singleton, singleton_repeat=singleton_repeat,
                      duplicate=duplicate, duplicate_repeat=duplicate_repeat,
                      mixed=mixed, mixed_repeat=mixed_repeat, reversed_mixed=reversed_mixed)
        saved.update({f'q{i}_{k}': finite(v) for k, v in values.items()})
        pairs = [(singleton_repeat, singleton), (duplicate_repeat, duplicate),
                 (duplicate[1], duplicate[0]), (duplicate[0], singleton), (duplicate[1], singleton),
                 (mixed_repeat, mixed), (reversed_mixed[1], mixed[0]), (mixed[0], singleton)]
        comparisons.extend(dict(query=i, comparison=name, **difference(a, b))
                           for name, (a, b) in zip(p['tests'], pairs))
    np.savez_compressed(out/f'{model}_forecasts.npz', **saved)
    write(out/f'{model}_calibration.json', comparisons)
    print(comparisons, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--model', choices=['bolt', 'timesfm'], default='bolt')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.pilot)
    elif args.command == 'run':
        execute(args.out, 'experiments.feature_search.persistence_calibrate', args.model)
    else:
        worker(args.out, args.model)
