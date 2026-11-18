"""Frozen singleton-geometry recovery of the exact-gated persistence pilot.

The original native implementation and failed run stay untouched. Calibration
found bit-exact repeats but batch-position errors up to4.76837158203125e-06 in
Bolt. Every native call here uses batch size1, including independent calls in
the duplicate-history control. No equality gate or tolerance is changed.
"""
import argparse
from pathlib import Path

import numpy as np

from experiments.feature_search.persistence_protocol import prepare as prepare_original
from experiments.feature_search.persistence_protocol import exact, read, sha, write
from experiments.feature_search.persistence_execution import execute


def singleton_class(base):
    """Preserve native math, weights and patching while fixing batch geometry."""
    class Singleton(base):
        def predict(self, x, replacement=None):
            """Execute genuinely separate singleton calls, then concatenate results."""
            values = [super(Singleton, self).predict(
                x[i:i+1], replacement=None if replacement is None else replacement[i:i+1])
                for i in range(len(x))]
            return tuple(np.concatenate([value[j] for value in values]) for j in [0, 1])
    return Singleton


def prepare(out, failed, calibration):
    """Freeze an unchanged corpus with a separately hashed singleton wrapper."""
    prepare_original(out, failed.parent/'selective_replication4942000')
    for part in ['inference/inputs.npz', 'evaluation/targets.npz']:
        a, b = np.load(out/part), np.load(failed/part)
        if set(a.files) != set(b.files):
            raise ValueError('Recovery corpus schema changed')
        for key in a.files:
            exact(a[key], b[key], 'unchanged recovery corpus')
    p = read(out/'inference/protocol.json')
    paths = [Path(__file__), Path(__file__).with_name('persistence_execution.py')]
    p['sources'].update({str(f.resolve()): sha(f) for f in paths})
    p['native_geometry'] = 'All calls batch size1, including both independent duplicate-history calls'
    p['entrypoint'] = 'experiments.feature_search.persistence_singleton'
    write(out/'inference/protocol.json', p)
    protocol = read(out/'protocol.json')
    protocol['sealed']['inference/protocol.json'] = sha(out/'inference/protocol.json')
    protocol['recovery'] = dict(failed=str(failed.resolve()), original_protocol_sha256=sha(failed/'protocol.json'),
                                calibration=str(calibration.resolve()),
                                calibration_protocol_sha256=sha(calibration/'protocol.json'),
                                reason='Exact duplicate-batch identity failed; calibration isolates stable position arithmetic',
                                adjustment=p['native_geometry'], gate='Every original exact gate unchanged; no tolerance')
    write(out/'protocol.json', protocol)
    write(out/'seal.json', dict(protocol_sha256=sha(out/'protocol.json')))


def worker(out, model):
    """Install only the declared geometry wrapper, then run frozen original worker."""
    from experiments.feature_search import real_background_poc, persistence_study
    real_background_poc.Native = singleton_class(real_background_poc.Native)
    persistence_study.worker(out, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--failed', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--model', choices=['bolt', 'timesfm'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.failed, args.calibration)
    elif args.command == 'run':
        execute(args.out, 'experiments.feature_search.persistence_singleton', args.model)
        execution = {m: read(args.out/f'{m}_execution.json') for m in ['bolt', 'timesfm']
                     if (args.out/f'{m}_execution.json').exists()}
        write(args.out/'execution.json', execution)
    else:
        worker(args.out, args.model)
