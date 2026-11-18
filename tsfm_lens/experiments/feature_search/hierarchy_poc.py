"""Bounded fresh-series comparison using a surviving, layer-matched SAE.

Run from tsfm_lens with PYTHONPATH=.; only public fresh synthetic rows enter.
The first SAE is a historical dictionary, so this tests transfer organization,
not whether fresh training would discover a hierarchy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from hierarchy import (cosine_hierarchy, decoder_hierarchy, parent_responses,
                       predict_labels, promotion_gate, random_hierarchy)
from tsfm_lens.sae.models import TopKSAE
from tsfm_lens.experiments.feature_search.statistics import paired_effect


def run(data_dir: Path, checkpoint: Path, layer: str) -> dict:
    """Fit memberships on train-active atoms and score the untouched test split."""
    torch.set_num_threads(1)
    metadata = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    tokens = np.load(data_dir / 'bolt.npz')[layer]
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if tokens.shape[-1] != saved['d_in']:
        raise ValueError('checkpoint hidden dimension mismatch')
    model = TopKSAE(saved['d_in'], saved['dict_size'], saved['k'])
    model.load_state_dict(saved['state_dict'])
    window_activations = tokens.reshape(len(tokens), 4, 2, tokens.shape[-1]).mean(2)
    with torch.no_grad():
        codes = model.encode(torch.from_numpy(window_activations.astype(np.float32))).mean(1).numpy()
    split = np.array([x['split'] for x in metadata])
    train, test = split == 'train', split == 'test'
    frequency = (codes[train] > 0).mean(0)
    active = np.flatnonzero(frequency > 0)
    subset = active[np.argsort(-frequency[active], kind='stable')[:256]]
    if len(subset) < 8:
        return {'status': 'defer', 'reason': 'fewer than eight train-active atoms'}
    atoms = saved['state_dict']['W_dec'][subset].numpy()
    codes = codes[:, subset]
    baseline = cosine_hierarchy(atoms, 4)
    diagnostics = [{} for _ in range(3)]
    memberships = [decoder_hierarchy(atoms, 4, seed=s, steps=100, diagnostics=diagnostics[s])
                   for s in range(3)]
    arms = {'cosine': baseline, 'random': random_hierarchy(baseline, 0),
            'uncompressed': np.eye(len(subset)),
            **{f'decoder_random_{s}': random_hierarchy(m, s) for s, m in enumerate(memberships)},
            **{f'decoder_{s}': m for s, m in enumerate(memberships)}}
    labels = {'coarse': np.array([x['dose'] for x in metadata]),
              'fine': np.array([x['period'] for x in metadata])}
    results, correctness = {}, {}
    groups = np.array([x['seed'] for x in metadata])
    for task, y in labels.items():
        mask = np.ones(len(y), dtype=bool) if task == 'coarse' else labels['coarse'] == 1
        tr, te = train & mask, test & mask
        correctness[task] = {}
        results[task] = {}
        for name, membership in arms.items():
            response = parent_responses(codes, membership)
            pred = predict_labels(response[tr], y[tr], response[te])
            _, inverse, counts = np.unique(y[te], return_inverse=True, return_counts=True)
            score = (pred == y[te]) * len(pred) / (len(counts) * counts[inverse])
            correctness[task][name] = score
            results[task][name] = float(score.mean())
        results[task]['decoder_vs_cosine'] = [paired_effect(
            correctness[task][f'decoder_{s}'], correctness[task]['cosine'],
            groups[te], n_boot=999, n_permutations=999, seed=s) for s in range(3)]
        results[task]['cosine_vs_random'] = paired_effect(
            correctness[task]['cosine'], correctness[task]['random'], groups[te],
            n_boot=999, n_permutations=999, seed=0)
    assignments = [np.where(m.sum(1) > 0, m.argmax(1), -1) for m in memberships]
    ari = [float(adjusted_rand_score(assignments[i], assignments[j]))
           for i, j in [(0, 1), (0, 2), (1, 2)]]
    gains = {task: [results[task][f'decoder_{s}']-results[task]['cosine']
                    for s in range(3)] for task in labels}
    return {'status': 'exploratory', 'layer': layer, 'checkpoint': str(checkpoint),
            'n_train_active': len(active), 'n_selected': len(subset),
            'activation_frequency_coverage': float(frequency[subset].sum()/frequency.sum()),
            'level2_seed_ari': ari, 'scores': results,
            'optimization': diagnostics,
            'pooling': '32-step means (two 16-step Bolt tokens), encode then series mean',
            'distribution_shift': 'historical SAE context480/horizon64; fresh context128/horizon32',
            'unassigned_atoms': [int((m.sum(1)==0).sum()) for m in memberships],
            'promote': bool(all(np.all(m.sum(1) > 0) for m in memberships)
                            and promotion_gate(gains['coarse'], gains['fine'], ari)),
            'scope': 'historical SAE; capped train-active atoms; frozen level1; predictive only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--layer', default='encoder.block.3')
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.checkpoint, args.layer), indent=2))
