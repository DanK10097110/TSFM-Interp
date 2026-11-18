"""Source-balanced, coordinate-invariant predictive comparisons without inference.

Dual FIT-only PCA preserves full-token Gram geometry; no cross-model coordinate
subtraction is used. All selection, centering and bandwidths use training sources.
"""
import hashlib
import numpy as np
from scipy.linalg import eigh, solve


ALPHAS = (0.001, 0.01, 0.1, 1.0)


def stable_argmin(scores):
    """Resolve machine-precision CV ties by the preregistered candidate order."""
    values = np.asarray(scores)
    minimum = values.min()
    return int(np.flatnonzero(values <= minimum + 1e-12 * max(1., abs(minimum)))[0])


def weights(rows, ids):
    """Give each source equal total weight, regardless of repeated conditions."""
    names = [rows[i]['source'] for i in ids]
    counts = {s: names.count(s) for s in set(names)}
    return np.array([1 / counts[s] / len(counts) for s in names])


def folds(rows, ids, count=4):
    """Stratify deterministic whole-source inner folds by domain."""
    assignment = {}
    for domain in sorted({rows[i]['domain'] for i in ids}):
        sources = sorted({rows[i]['source'] for i in ids if rows[i]['domain'] == domain},
                         key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        assignment.update({s: j % count for j, s in enumerate(sources)})
    return [(np.array([i for i in ids if assignment[rows[i]['source']] != f]),
             np.array([i for i in ids if assignment[rows[i]['source']] == f])) for f in range(count)]


def center_gram(k, train, w):
    """Center every row relative to the weighted training feature mean."""
    mean = k[:, train] @ w
    return k - mean[:, None] - mean[None, :] + w @ k[np.ix_(train, train)] @ w


def pca_kernel(k, train, w, rank):
    """Return truncated PCA Gram using only FIT rows; scalar scaling keeps rotations."""
    k = center_gram(k, train, w)
    scale = max(float(w @ np.diag(k)[train]), 1e-14)
    k = k / scale
    sw = np.sqrt(w)
    vals, vecs = eigh(k[np.ix_(train, train)] * sw[:, None] * sw[None, :])
    keep = np.flatnonzero(vals > max(vals[-1] * 1e-10, 1e-14))[-rank:]
    if not len(keep):
        return np.zeros_like(k)
    z = (k[:, train] * sw) @ (vecs[:, keep] / np.sqrt(vals[keep]))
    return z @ z.T


def input_kernels(x, train, w, rank_cap):
    """Strong past-only linear and nonlinear controls with FIT-only standardization."""
    mean = w @ x[train]
    sd = np.sqrt(w @ ((x[train] - mean) ** 2))
    sd[sd < 1e-8] = 1
    z = (x - mean) / sd
    raw = z @ z.T
    result = {f'pca{r}': pca_kernel(raw, train, w, min(r, rank_cap)) for r in (8, 16)}
    for name, values in [('history_ridge', x[:, :128]), ('descriptor_ridge', z)]:
        full = center_gram(values @ values.T, train, w)
        result[name] = full / max(float(w @ np.diag(full)[train]), 1e-14)
    dist = np.maximum(np.diag(raw)[:, None] + np.diag(raw)[None, :] - 2 * raw, 0)
    td = dist[np.ix_(train, train)]
    bandwidth = max(float(np.median(td[np.triu_indices(len(train), 1)])), 1e-10)
    for gamma in (0.5, 2.0):
        result[f'rbf{gamma}'] = center_gram(np.exp(-gamma * dist / bandwidth), train, w)
    return result


def kernel_bank(x, grams, rows, train):
    """Build candidate kernels once per FIT fold, independent of every target label."""
    w = weights(rows, train)
    cap = max(1, len({rows[i]['source'] for i in train}) - 1)
    ins = input_kernels(x, train, w, cap)
    bank = {'input': ins}
    for name, gram in grams.items():
        hidden = {f'pca{r}': pca_kernel(gram, train, w, min(r, cap)) for r in (8, 16)}
        bank[name] = hidden
        bank['input+' + name] = {f'{a}+{b}': (ka + kb) / 2 for a, ka in ins.items() for b, kb in hidden.items()}
    return bank


def ridge(k, y, train, evaluate, w, alpha):
    """Weighted kernel ridge with a training-only intercept and normalized objective."""
    sw = np.sqrt(w)
    mean = w @ y[train]
    matrix = k[np.ix_(train, train)] * sw[:, None] * sw[None, :]
    coef = solve(matrix + alpha * np.eye(len(train)), (y[train] - mean) * sw[:, None], assume_a='pos')
    return mean + (k[np.ix_(evaluate, train)] * sw) @ coef


def select_predict(banks, final_bank, y, rows, train, evaluate, arm, binary=False):
    """Choose kernel/regularization by whole-source inner FIT risk, then refit FIT."""
    candidates = [(name, alpha) for name in final_bank[arm] for alpha in ALPHAS]
    scores = []
    for name, alpha in candidates:
        losses, source_names = [], []
        for tr, va, bank in banks:
            pred = ridge(bank[arm][name], y, tr, va, weights(rows, tr), alpha)
            if binary:
                pred = pred.clip(0, 1)
            losses.extend(np.mean((pred - y[va]) ** 2, axis=1))
            source_names.extend(rows[i]['source'] for i in va)
        scores.append(float(np.mean([np.mean([v for v, s in zip(losses, source_names) if s == source]) for source in sorted(set(source_names))])))
    chosen = stable_argmin(scores)
    name, alpha = candidates[chosen]
    pred = ridge(final_bank[arm][name], y, train, evaluate, weights(rows, train), alpha)
    if binary:
        pred = pred.clip(0, 1)
    return pred, dict(kernel=name, alpha=alpha, cv_loss=scores[chosen])


def shuffled_indices(rows, train, seed):
    """Break FIT hidden/target pairing with whole-source permutations within strata.

    Preserve cohort, domain and condition signatures; TEST identities never move.
    Fixed points are permitted for a genuine uniform permutation.
    """
    mapping = np.arange(len(rows))
    groups = {}
    for i in train:
        groups.setdefault(rows[i]['source'], []).append(i)
    strata = {}
    for source, ids in groups.items():
        ordered = sorted(ids, key=lambda i: rows[i]['case'])
        key = (rows[ids[0]]['run'], rows[ids[0]]['domain'], tuple(rows[i]['case'] for i in ordered))
        strata.setdefault(key, []).append(ordered)
    rng = np.random.default_rng(seed)
    for bundles in strata.values():
        for dest, j in zip(bundles, rng.permutation(len(bundles))):
            mapping[dest] = bundles[j]
    return mapping


def group_values(values, rows, ids):
    """Aggregate repeated cases before every uncertainty or headline calculation."""
    return {s: float(np.mean([values[j] for j, i in enumerate(ids) if rows[i]['source'] == s])) for s in sorted({rows[i]['source'] for i in ids})}


def interval(values, seed=4951000, coverage=0.95):
    """Conditional source bootstrap; predictions stay fixed, training uncertainty omitted."""
    v = np.asarray(values, float)
    draws = np.random.default_rng(seed).choice(v, size=(4000, len(v)), replace=True).mean(1)
    q = (1 - coverage) / 2
    return dict(mean=float(v.mean()), low=float(np.quantile(draws, q)), high=float(np.quantile(draws, 1-q)), n_sources=len(v))


def descriptors(history, plans):
    """Past-only raw trajectory, spectra, autocorrelations and saved tone descriptors."""
    x = np.asarray(history, float)
    mean, sd = x.mean(1), x.std(1)
    if np.any(sd <= 1e-10):
        raise ValueError('Degenerate context scale')
    z = (x - mean[:, None]) / sd[:, None]
    acf = np.stack([np.mean(z[:, lag:] * z[:, :-lag], axis=1) for lag in range(1, 33)], axis=1)
    tone = np.zeros((len(x), 3 * 7 + 1))
    for i, plan in enumerate(plans):
        tone[i, -1] = len(plan['details'])
        for j, d in enumerate(plan['details'][:3]):
            tone[i, 7*j:7*j+7] = [1, d['frequency']/64, *list(np.asarray(d['coefficients'])/sd[i]), d['stability'], d['explained'], d['confidence']]
    return np.column_stack([z, np.abs(np.fft.rfft(z)) / np.sqrt(x.shape[1]), acf,
                            np.quantile(z, np.linspace(0, 1, 9), axis=1).T,
                            mean/sd, np.log(sd), tone])
