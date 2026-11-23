"""Independently replay recorded event effects with all native encoder blocks live."""
import json

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.more_seeds.fit import OUT, SCRATCH, SEEDS
from research.activation_transfer.moment.dictionary_transfer.more_seeds.run import ARMS, SOURCE, independent_native_check, original_data
from research.activation_transfer.moment.dictionary_transfer.weather.study import file_hash, histories
from research.activation_transfer.moment.token_paths.geometry import CACHE, ROOT, load


def audit():
    torch.set_num_threads(1)
    original = original_data()
    x, ids, *_ = histories()
    np.testing.assert_array_equal(ids, original['ids'])
    assert file_hash(ROOT / 'large/model.safetensors') == str(original['checkpoint_sha256'])
    assert file_hash(CACHE / 'large' / f'sae_{SOURCE}.pt') == str(original['source_sae_sha256'])
    model = load('large')
    for seed in SEEDS:
        with np.load(OUT / f'match_{seed}.npz') as f:
            match = dict(f)
        assert file_hash(SCRATCH / f'sae_{seed}.pt') == str(match['recipient_sae_sha256'])
        for arm in ARMS:
            with np.load(OUT / f'{seed}_{arm}.npz') as f:
                np.testing.assert_array_equal(f['rows'], original['rows'])
                gaps = independent_native_check(model, seed, original, match, arm, x, f['effect'])
            (OUT / f'check_{seed}_{arm}.json').write_text(json.dumps(gaps, indent=2)+'\n')


if __name__ == '__main__':
    audit()
