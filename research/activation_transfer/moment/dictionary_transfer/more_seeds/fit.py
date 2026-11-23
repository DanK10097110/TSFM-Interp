"""Fit one registered large MOMENT receiver without altering source artifacts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.horizon_identity.train import tensor_hash
from research.activation_transfer.moment.token_paths.geometry import CACHE
from tsfm_lens.sae.train import SAETrainConfig, save_sae, train_sae

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get('TSFM_MOMENT_MORE_SEEDS_CACHE', '/tmp/tsfm_moment_more_seeds_20261122'))
OUT = Path(os.environ.get('TSFM_MOMENT_MORE_SEEDS_OUT', HERE / 'results'))
SEEDS = (22112203, 22112204)


def fit(seed):
    torch.set_num_threads(2)
    cache = CACHE / 'large'
    states = np.load(cache / 'train_observed.npy', mmap_mode='r')
    assert states.shape == (1024, 60, 1024)
    train = np.asarray(states.reshape(-1, states.shape[-1]))
    expected = json.loads((HERE.parents[1] / 'horizon_identity/results/large_22112201_dictionary.json').read_text())
    digest = tensor_hash(train)
    assert digest == expected['train_sha256']
    config = SAETrainConfig(dict_size_mult=4, k=32, lr=.001, epochs=6, batch_size=512, seed=seed)
    sae, history = train_sae(train, config, torch.device('cpu'), context=f'MOMENT large, observed middle H16, seed {seed}')
    SCRATCH.mkdir(exist_ok=True, parents=True)
    save_sae(sae, SCRATCH / f'sae_{seed}.pt')
    metadata = dict(seed=seed, train_sha256=digest, loss=history, train_meta=sae.train_meta)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / f'fit_{seed}.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, choices=SEEDS)
    fit(parser.parse_args().seed)
