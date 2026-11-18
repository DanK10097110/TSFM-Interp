"""Small source-weighted dictionary controls, independent of native forecasting.

Residual coordinates are signed directions, not FFN neurons. A sparse probe is
an information diagnostic, never evidence that pretraining learned a concept.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tsfm_lens.sae.models import TopKSAE


def source_weights(sources):
    """Equal mass per source; repeated cases/tokens cannot inflate its weight."""
    _, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def split_indices(rows):
    """Refuse mixed source membership before any fitting or normalization."""
    membership = {}
    for row in rows:
        if row['split'] not in ('fit', 'test'):
            raise ValueError('Unknown split')
        key = row['source']
        if key in membership and membership[key] != row['split']:
            raise ValueError('Source leakage')
        membership[key] = row['split']
    indices = tuple(np.array([i for i, r in enumerate(rows) if r['split'] == s])
                    for s in ('fit', 'test'))
    if any(len(i) == 0 for i in indices):
        raise ValueError('Both FIT and TEST are required')
    return indices


def fit_scaler(x, weights):
    """FIT-only vector center and ONE scalar preserve native coordinate geometry."""
    x = np.asarray(x, dtype=np.float64)
    center = weights @ x
    scale = float(np.sqrt(weights @ np.mean((x - center) ** 2, axis=1)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError('Degenerate FIT states')
    return center.astype(np.float32), scale


def signed_topk(x, k):
    """Keep signed largest magnitudes, with scalar or per-token active budgets."""
    x = np.asarray(x)
    counts = np.broadcast_to(np.asarray(k, dtype=int), x.shape[:-1]).reshape(-1)
    if np.any(counts < 0) or np.any(counts > x.shape[-1]):
        raise ValueError('Invalid activity budget')
    flat = x.reshape(-1, x.shape[-1])
    order = np.argsort(-abs(flat), axis=1, kind='stable')
    selected = np.arange(x.shape[-1])[None] < counts[:, None]
    result = np.zeros_like(flat)
    np.put_along_axis(result, order, np.take_along_axis(flat, order, axis=1)*selected, axis=1)
    return result.reshape(x.shape)


def fit_basis(x, weights, method, seed):
    """Full orthonormal bases; PCA uses FIT covariance, rotation uses only seed."""
    d = x.shape[1]
    if method == 'native':
        return np.eye(d, dtype=np.float32)
    if method == 'rotation':
        q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
        return (q*np.where(np.diag(r) < 0, -1., 1.)).astype(np.float32)
    if method == 'pca':
        _, vectors = np.linalg.eigh((x.T * weights) @ x)
        return vectors[:, ::-1].astype(np.float32)
    raise ValueError(method)


class NestedTopK(TopKSAE):
    """Matryoshka-inspired: global TopK code, shared decoder, d/2,d prefixes.

    Equal prefix losses; no BatchTopK, AuxK, revival, temporal regularizer or
    forecast-consistency training. This is not a paper reproduction.
    """

    def reconstruction_loss(self, x):
        """One sparse code, two truncated reconstructions; no prefix re-ranking."""
        z = self.encode(x)
        widths = sorted(set((self.dict_size//2, self.dict_size)))
        return torch.stack([((z[:, :n] @ self.W_dec[:n] + self.b_dec - x)**2).mean()
                            for n in widths]).mean()


def train_sae(x, weights, method, seed, k, width, steps=500, batch=64):
    """Same initialization, sampled tokens and optimizer steps for both recipes."""
    if method not in ('topk', 'nested') or not 1 <= width <= 2*x.shape[1]:
        raise ValueError('Invalid bounded SAE')
    if not 1 <= steps <= 500 or not 1 <= batch <= 64 or k > width:
        raise ValueError('Training cap')
    cls = NestedTopK if method == 'nested' else TopKSAE
    model = cls(x.shape[1], width, k, generator=torch.Generator().manual_seed(seed))
    data = torch.as_tensor(x, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        indices = rng.choice(len(x), batch, replace=True, p=weights)
        values = data[indices]
        loss = (model.reconstruction_loss(values) if method == 'nested'
                else ((model(values)[0]-values)**2).mean())
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_decoder_()
        losses.append(float(loss.detach()))
    model.eval()
    arrays = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    return arrays, dict(steps=steps, batch=batch, sampled_tokens=steps*batch,
                       equivalent_uniform_token_passes=steps*batch/len(x),
                       trainable_parameters=sum(p.numel() for p in model.parameters()),
                       reconstruction_terms=2 if method == 'nested' else 1,
                       forward_dense_matmul_flops_per_step=2*batch*x.shape[1]*(2*width+(width//2 if method=='nested' else 0)),
                       flop_scope='analytical dense forward matmuls only; backward/TopK/Adam excluded; nested 1.25x forward cost at even width',
                       first_loss=losses[0], final_loss=losses[-1], aux_k=False)


def encode(x, artifact, counts=None):
    """Frozen inference; neither the batch composition nor TEST sets any scalar."""
    k = int(artifact['k'])
    if 'basis' in artifact:
        return signed_topk(x @ artifact['basis'], k if counts is None else counts)
    pre = np.maximum((x-artifact['b_dec']) @ artifact['W_enc']+artifact['b_enc'], 0)
    return signed_topk(pre, k)


def decoder(artifact):
    """Return feature rows in residual units before the common scalar inverse."""
    return artifact['basis'].T if 'basis' in artifact else artifact['W_dec']


def reconstruct(z, artifact):
    """Decode in scaled coordinates, including learned decoder bias if present."""
    return z @ decoder(artifact) + artifact.get('b_dec', 0)


def health(x, z, reconstruction, weights):
    """Weighted fidelity/activity and strict empirical never-active fraction."""
    l0 = np.count_nonzero(z, axis=1)
    freq = weights @ (z != 0)
    error = np.mean((x-reconstruction)**2, axis=1)
    denom = float(weights @ np.mean(x*x, axis=1))
    return dict(nmse=float(weights @ error/max(denom, 1e-20)),
                mean_l0=float(weights @ l0), min_l0=int(l0.min()), max_l0=int(l0.max()),
                dead_fraction=float(np.mean(freq == 0)), activity_frequency=freq.tolist())


def sparse_probe(train_x, train_y, weights, test_x, budget):
    """FIT correlation screen then fixed ridge, with exactly 1 or 4 candidates."""
    mean = weights @ train_x
    scale = np.sqrt(weights @ (train_x-mean)**2).clip(1e-8)
    a = (train_x-mean)/scale
    ymean = float(weights @ train_y)
    covariance = (a.T*weights) @ (train_y-ymean)
    selected = np.argsort(-abs(covariance), kind='stable')[:budget]
    design = a[:, selected]
    coef = np.linalg.solve((design.T*weights) @ design + .01*np.eye(len(selected)),
                           (design.T*weights) @ (train_y-ymean))
    prediction = ((test_x[:, selected]-mean[selected])/scale[selected]) @ coef+ymean
    return prediction, selected.tolist()


def conditional_probe(train_x, train_y, weights, test_x, budget, train_metadata, test_metadata):
    """Select few features on FIT residuals, giving both arms the SAME metadata."""
    mean = weights @ train_metadata
    scale = np.sqrt(weights @ (train_metadata-mean)**2).clip(1e-8)
    m = (train_metadata-mean)/scale
    mt = (test_metadata-mean)/scale
    xm, ym = weights @ train_x, float(weights @ train_y)
    gram = (m.T*weights) @ m+.01*np.eye(m.shape[1])
    cx = np.linalg.solve(gram, (m.T*weights) @ (train_x-xm))
    cy = np.linalg.solve(gram, (m.T*weights) @ (train_y-ym))
    residual, selected = sparse_probe(train_x-xm-m @ cx, train_y-ym-m @ cy,
                                      weights, test_x-xm-mt @ cx, budget)
    return ym+mt @ cy+residual, ym+mt @ cy, selected


def stability(a, b, active_a, active_b):
    """One-to-one absolute cosine matching, excluding empirically dead atoms."""
    a, b = a[np.asarray(active_a)], b[np.asarray(active_b)]
    if min(len(a), len(b)) == 0:
        return dict(matched=0, mean_abs_cosine=None)
    a = a/np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    b = b/np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    scores = abs(a @ b.T)
    i, j = linear_sum_assignment(-scores)
    return dict(matched=len(i), mean_abs_cosine=float(scores[i, j].mean()),
                unmatched_a=len(a)-len(i), unmatched_b=len(b)-len(j))


def source_summary(values, rows):
    """Retain independent-source scores; domain means never count tokens as n."""
    result = {}
    for domain in sorted({r['domain'] for r in rows}):
        sources = sorted({r['source'] for r in rows if r['domain'] == domain})
        per_source = {s: float(np.mean([v for v, r in zip(values, rows) if r['source'] == s]))
                      for s in sources}
        result[domain] = dict(n_sources=len(sources), mean=float(np.mean(list(per_source.values()))),
                              per_source=per_source)
    return result
