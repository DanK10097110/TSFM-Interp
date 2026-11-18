"""Small held-out predictive comparison tools for feature discovery.

Generating-series seeds, including every perturbation, are the independent
units. Discovery and validation choose a ridge; evaluation never chooses it.
Intervals condition on fitted predictions. Predictivity is not causality or
proof that an undetected representation is absent (see FINDINGS MN-28).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ...analysis.stats import holm, mean_ci


def _groups(groups):
    """Require scalar, nonmissing numeric or string generating-seed labels."""
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.dtype.kind not in "biufUS" or (groups.dtype.kind in "f" and not np.isfinite(groups).all()):
        raise ValueError("Need one-dimensional finite numeric or string generating seeds")
    if groups.dtype.kind in "US" and np.any(groups == ""):
        raise ValueError("Generating seed labels cannot be empty")
    return groups


@dataclass(frozen=True)
class GroupedSplit:
    """Row indices whose generating seeds are disjoint across partitions."""

    train: np.ndarray
    validation: np.ndarray
    evaluation: np.ndarray


def grouped_split(groups, *, seed=0, fractions=(0.6, 0.2, 0.2)):
    """Partition generating seeds reproducibly, keeping perturbations together."""
    groups = _groups(groups)
    fractions = np.asarray(fractions, dtype=float)
    if groups.ndim != 1 or fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Need one-dimensional seeds and three positive fractions summing to one")
    units = np.unique(groups)
    if len(units) < 5:
        raise ValueError("Need at least five independent generating seeds")
    units = np.random.default_rng(seed).permutation(units)
    n_train = max(1, int(len(units) * fractions[0]))
    n_valid = max(1, int(len(units) * fractions[1]))
    if n_train + n_valid >= len(units):
        raise ValueError("Fractions leave no evaluation seeds")
    return GroupedSplit(*(np.flatnonzero(np.isin(groups, part)) for part in
                          (units[:n_train], units[n_train:n_train+n_valid], units[n_train+n_valid:])))


def _arrays(x, y):
    """Validate finite design matrices and scalar or multivariate targets."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim not in (1, 2) or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need finite aligned matrix X and vector/matrix y")
    return x, y


def fit_ridge(x, y, split, *, alphas=(0.01, 0.1, 1.0, 10.0)):
    """Choose alpha on validation, then fit scaling and ridge on discovery only.

    Discovery consists of train plus validation. Evaluation targets and feature
    values never enter fitting or alpha selection. The caller must preserve the
    generating-seed partition when supplying a custom split.
    """
    x, y = _arrays(x, y)
    parts = [np.asarray(getattr(split, name)) for name in ("train", "validation", "evaluation")]
    joined = np.concatenate(parts)
    if any(len(p) == 0 for p in parts) or len(np.unique(joined)) != len(joined) or np.any(joined < 0) or np.any(joined >= len(x)):
        raise ValueError("Split indices must be nonempty, disjoint and in range")
    alphas = tuple(alphas)
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Ridge alphas must be finite and positive")
    scores = []
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[split.train], y[split.train])
        scores.append(float(np.mean((model.predict(x[split.validation]) - y[split.validation]) ** 2)))
    alpha = alphas[int(np.argmin(scores))]
    discovery = np.concatenate((split.train, split.validation))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x[discovery], y[discovery])
    return model, {"alpha": float(alpha), "validation_mse": min(scores)}


def paired_effect(a, b, groups, *, n_boot=1000, n_permutations=999, seed=0, ci=0.95):
    """Estimate mean a-b with equal-seed weighting, cluster CI and sign-flip p.

    The two-sided permutation null requires exchangeable paired labels (or
    symmetric seed-level differences). It is not a bootstrap sign-count p.
    Every perturbation of a seed shares its sign. The plus-one correction
    bounds attainable precision by 1/(n_permutations+1).
    """
    a, b, groups = np.asarray(a, float), np.asarray(b, float), _groups(groups)
    if a.ndim != 1 or a.shape != b.shape or a.shape != groups.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need finite paired vectors and one generating seed per row")
    units = np.unique(groups)
    if len(units) < 3 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in (n_boot, n_permutations)) or not 0 < ci < 1:
        raise ValueError("Need at least three seeds, positive resamples and 0<ci<1")
    differences = np.array([(a-b)[groups == unit].mean() for unit in units])
    result = mean_ci(differences, n_boot=n_boot, seed=seed, ci=ci, unit="generating_series_seed")
    rng = np.random.default_rng(seed)
    observed = abs(differences.mean())
    exceed = sum(abs(np.mean(differences * rng.choice((-1, 1), len(units)))) >= observed
                 for _ in range(n_permutations))
    result.update(p=(1 + exceed) / (n_permutations + 1), n_seeds=len(units),
                  n_permutations=n_permutations, ci=ci, inference="fixed_predictions")
    return result


def compare_effects(comparisons, groups, **kwargs):
    """Holm-adjust a prespecified complete family of named paired comparisons."""
    results = {name: paired_effect(a, b, groups, **kwargs) for name, (a, b) in comparisons.items()}
    adjusted = holm({name: result["p"] for name, result in results.items()})
    for name, result in results.items():
        result["p_holm"] = adjusted[name]
    return results


def evidence_verdict(effect, *, equivalence_margin, alpha=0.05):
    """Separate detected effects, interval equivalence and unresolved absence.

    Equivalence is descriptive interval containment, not a calibrated TOST or
    multiplicity-adjusted equivalence claim. 'Shared' additionally requires
    positive transfer evidence on both sides; this function cannot infer it.
    """
    if equivalence_margin <= 0 or not 0 < alpha < 1:
        raise ValueError("Need a positive practical margin and 0<alpha<1")
    if effect["lo"] > -equivalence_margin and effect["hi"] < equivalence_margin:
        return "interval_within_margin"
    if effect["p_holm"] <= alpha and (effect["lo"] > 0 or effect["hi"] < 0):
        return "detected"
    return "not_detected"


def incremental_predictivity(inputs, internals, gap, groups, *, split=None, seed=0, alphas=(0.01, 0.1, 1.0, 10.0), **resampling):
    """Evaluate added prediction of a forecast-error gap beyond input features.

    Both models use the same untouched evaluation seeds. Positive effect means
    lower squared error after adding internals. This strict holdout alternative
    to cross-fitting avoids choosing features on the confirmation rows.
    """
    inputs, gap = _arrays(inputs, gap)
    internals, _ = _arrays(internals, gap)
    groups = _groups(groups)
    if groups.shape != (len(gap),) or gap.ndim != 1:
        raise ValueError("Need scalar error gaps and aligned generating seeds")
    split = grouped_split(groups, seed=seed) if split is None else split
    seed_sets = [set(groups[getattr(split, part)]) for part in ("train", "validation", "evaluation")]
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Generating seeds cross partition boundaries")
    baseline, base_fit = fit_ridge(inputs, gap, split, alphas=alphas)
    full_x = np.column_stack((inputs, internals))
    full, full_fit = fit_ridge(full_x, gap, split, alphas=alphas)
    idx = split.evaluation
    base_loss = (gap[idx] - baseline.predict(inputs[idx])) ** 2
    full_loss = (gap[idx] - full.predict(full_x[idx])) ** 2
    effect = paired_effect(base_loss, full_loss, groups[idx], seed=seed, **resampling)
    return {"effect": effect, "baseline_fit": base_fit, "full_fit": full_fit,
            "evaluation_indices": idx, "evidence_class": "held_out_predictive"}
