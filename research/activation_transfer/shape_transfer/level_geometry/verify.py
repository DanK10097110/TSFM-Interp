"""Check numeric equivalence of separate Jacobian and target-intervention reruns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def array_gap(left, right):
    x, y = np.asarray(left), np.asarray(right)
    if x.dtype.kind in 'OUSBb' or not x.size:
        np.testing.assert_array_equal(x, y)
        return 0.0
    np.testing.assert_allclose(x, y, atol=3e-5, rtol=1e-5)
    return float(np.max(np.abs(x.astype(float) - y.astype(float))))


def compare(left, right):
    with np.load(left) as a, np.load(right) as b:
        assert set(a.files) == set(b.files)
        return {key: array_gap(a[key], b[key]) for key in a.files}


def json_gap(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        return max((json_gap(value, right[key]) for key, value in left.items()), default=0.)
    if isinstance(left, list):
        assert len(left) == len(right)
        return max((json_gap(a, b) for a, b in zip(left, right)), default=0.)
    return array_gap(left, right)


def read_json(path):
    return json.loads(path.read_text())


def checks(original, replica, original_jac, replica_jac):
    result = {}
    for model in ('bolt', 'timesfm'):
        geometry = read_json(original / f'{model}_geometry.json')
        replay_geometry = read_json(replica / f'{model}_geometry.json')
        # A zip container digest can differ with member timestamps even for identical arrays.
        geometry.pop('jac_sha256')
        replay_geometry.pop('jac_sha256')
        result[model] = {
            'jac': compare(original_jac / f'{model}_jac.npz', replica_jac / f'{model}_jac.npz'),
            'interventions': compare(original / f'{model}_effects.npz', replica / f'{model}_effects.npz'),
            'geometry_max_difference': json_gap(geometry, replay_geometry),
        }
    gap = json_gap(read_json(original / 'scores.json'), read_json(replica / 'scores.json'))
    assert gap < 1e-4
    result['max_heldout_score_or_interval_difference'] = gap
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['original', 'replica', 'original-jac', 'replica-jac', 'out']:
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    differences = checks(args.original, args.replica, args.original_jac, args.replica_jac)
    args.out.write_text(json.dumps(differences, indent=2) + '\n')
