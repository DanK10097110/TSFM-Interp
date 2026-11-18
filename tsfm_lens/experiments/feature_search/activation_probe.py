"""Small direct-activation probe; all results are correlational, never causal.

Fresh synthetic generating series are split before any probe selection. This
standalone experiment compares raw coordinates, PCA, supervised ridge and a
rotated coordinate control without fitting an SAE or touching sealed corpora.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def make_series(n: int, length: int, seed: int):
    """Generate independent series with orthogonal frequency/trend nuisance draws."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, length)
    cycles = rng.uniform(2, 10, n)
    trend = rng.uniform(-2, 2, n)
    phase = rng.uniform(-np.pi, np.pi, n)
    amplitude = rng.uniform(.5, 2, n)
    x = (amplitude[:, None] * np.sin(np.pi * cycles[:, None] * (t + 1)
         + phase[:, None]) + trend[:, None] * t)
    x += rng.normal(0, .15, x.shape)
    return x.astype(np.float32), np.column_stack([cycles, trend])


def input_features(x):
    """Return a declared strong input baseline: waveform, spectrum and summaries."""
    centered = x - x.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1))
    return np.column_stack([x, spectrum, x.mean(axis=1), x.std(axis=1)])


def _ridge(train, valid, test, ytrain, yvalid):
    """Fit scaling on train and choose regularization using validation only."""
    scale = StandardScaler().fit(train)
    a, b, c = [scale.transform(z) for z in (train, valid, test)]
    fits = [Ridge(alpha=alpha).fit(a, ytrain) for alpha in (0.01, 1, 100)]
    best = max(fits, key=lambda fit: r2_score(yvalid, fit.predict(b)))
    return best.predict(c)


def compare_probes(train, valid, test, ytrain, yvalid, ytest, seed=0):
    """Score train-selected probes on untouched test rows, including rotation.

    Rotation preserves geometry but changes coordinates. Ridge and full PCA
    should survive it; a neuron result is meaningful only relative to this
    coordinate control. High decoding does not show unique or causal content.
    """
    center = train.mean(axis=0)
    a, b, c = [np.asarray(z, dtype=np.float64) - center
               for z in (train, valid, test)]
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(a.shape[1], a.shape[1])))
    scores = {}
    for label, aa, bb, cc in [('neuron', a, b, c), ('rotated_neuron', a @ q, b @ q, c @ q)]:
        covariance = (aa - aa.mean(0)).T @ (ytrain - ytrain.mean())
        denom = np.sqrt(np.sum((aa - aa.mean(0)) ** 2, axis=0))
        index = int(np.argmax(np.abs(covariance) / np.maximum(denom, 1e-12)))
        pred = _ridge(aa[:, [index]], bb[:, [index]], cc[:, [index]], ytrain, yvalid)
        scores[label] = float(r2_score(ytest, pred))
    pca = PCA(n_components=min(16, len(a) - 1, a.shape[1])).fit(a)
    scores['pca16'] = float(r2_score(ytest, _ridge(
        pca.transform(a), pca.transform(b), pca.transform(c), ytrain, yvalid)))
    scores['supervised_ridge'] = float(r2_score(ytest, _ridge(a, b, c, ytrain, yvalid)))
    return scores
