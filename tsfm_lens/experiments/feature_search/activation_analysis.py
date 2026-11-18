"""Frozen validation-selected direct probes and cross-model contrast translation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from activation_probe import input_features


def fit_predict(x, y, train, valid, method, seed=0):
    """Select train-only representation and validation-only ridge regularization."""
    a = x[train]
    if method == 'rotated_neuron':
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1], x.shape[1])))
        x = x @ q; a = x[train]
    if method in ['neuron', 'rotated_neuron']:
        ac = a - a.mean(0)
        covariance = ac.T @ (y[train] - y[train].mean(0))
        score = abs(covariance) / np.maximum(np.sqrt((ac ** 2).sum(0)), 1e-12)
        x = x[:, [int(np.argmax(score))]]
    if method == 'pca16':
        pca = PCA(n_components=min(16, sum(train)-1, x.shape[1]), svd_solver='full').fit(x[train])
        x = pca.transform(x)
    scale = StandardScaler().fit(x[train]); x = scale.transform(x)
    fits = [Ridge(alpha=alpha).fit(x[train], y[train]) for alpha in [.01, 1, 100]]
    losses = [float(np.mean((fit.predict(x[valid])-y[valid])**2)) for fit in fits]
    selected = int(np.argmin(losses))
    return fits[selected].predict(x), losses[selected], [.01, 1, 100][selected]


def analyze(root: Path):
    """Write frozen-selection test predictions; no sealed or private data access."""
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    arrays = np.load(root/'corpus.npz')
    split = np.array([m['split'] for m in metadata])
    dose = np.array([m['dose'] for m in metadata])
    inputs = input_features(arrays['context'])
    models = {name: np.load(root/f'{name}.npz') for name in ['bolt','timesfm']}
    summary, predictions = {}, {}
    for target in ['presence', 'period_present']:
        mask = np.ones(len(dose), dtype=bool) if target == 'presence' else dose == 1
        y = dose.astype(float) if target == 'presence' else np.array([m['period'] for m in metadata],dtype=float)
        tr, va, te = [(split == s) & mask for s in ['train','validation','test']]
        denom = float(np.var(y[tr]))
        baseline, valid_loss, alpha = fit_predict(inputs, y, tr, va, 'ridge')
        entries = dict(input=dict(test_loss=float(np.mean((baseline[te]-y[te])**2)/denom),alpha=alpha))
        predictions[target+'_input'] = baseline[te]
        predictions[target+'_truth'] = y[te]
        predictions[target+'_seed'] = np.array([m['seed'] for m in metadata])[te]
        for name,data in models.items():
            layers = [k for k in data.files if k != 'prediction']
            for method in ['neuron','rotated_neuron','pca16','ridge']:
                candidates = [(layer, *fit_predict(data[layer].mean(1).astype(float), y, tr, va, method)) for layer in layers]
                layer, prediction, loss, alpha = min(candidates,key=lambda c:c[2])
                entries[name+'_'+method] = dict(layer=layer,alpha=alpha,validation_loss=loss/denom,
                                                test_loss=float(np.mean((prediction[te]-y[te])**2)/denom))
                predictions[target+'_'+name+'_'+method] = prediction[te]
        summary[target] = entries
    translation = {}
    pair_split = split[::2]
    tr, va, te = [pair_split == s for s in ['train','validation','test']]
    input_diff = inputs[1::2]-inputs[::2]
    labels = np.array([[m[k] for k in ['period','amplitude','phase','slope','noise']] for m in metadata[1::2]])
    final = {'bolt':'encoder.block.5','timesfm':'stacked_xf.19'}
    deltas = {}
    for name,data in models.items():
        pooled = data[final[name]].mean(1).astype(float)
        delta = pooled[1::2]-pooled[::2]
        pca = PCA(n_components=16,svd_solver='full').fit(delta[tr])
        deltas[name] = pca.transform(delta)
    for source,destination in [('bolt','timesfm'),('timesfm','bolt')]:
        y = deltas[destination]
        denom = float(np.mean((y[te]-y[tr].mean(0))**2))
        results = {}
        for feature,x in [('activation',deltas[source]),('input',input_diff),('oracle_generator_labels',labels)]:
            pred, loss, alpha = fit_predict(x,y,tr,va,'ridge')
            results[feature] = dict(test_nmse=float(np.mean((pred[te]-y[te])**2)/denom),alpha=alpha)
            predictions[f'translation_{source}_{destination}_{feature}'] = pred[te]
        predictions[f'translation_{source}_{destination}_truth'] = y[te]
        translation[source+'_to_'+destination] = results
    summary['translation'] = translation
    summary['evidence'] = 'correlational decoding and linear translation; no uniqueness or causality claim'
    (root/'probe_analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(root/'probe_predictions.npz',**predictions)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    analyze(parser.parse_args().root)
