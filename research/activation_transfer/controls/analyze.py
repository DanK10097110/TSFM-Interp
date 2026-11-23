"""Predict a destination's signed effects from source black-box responses on held-out series."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def fit_clean(train_x, response, test_x, use_stats=True):
    """Fit one multivariate ridge regression with an unpenalized intercept."""
    n = len(train_x)
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-10] = 1
    train_x, test_x = (train_x-center)/scale, (test_x-center)/scale
    mu = response.mean(axis=0)
    shape = (n, *response.shape[1:])
    if not use_stats:
        return np.broadcast_to(mu, shape), np.broadcast_to(mu, (len(test_x), *response.shape[1:]))
    solution = np.linalg.solve(train_x.T @ train_x + np.eye(train_x.shape[1])*n/10,
                               train_x.T @ (response-mu).reshape(n, -1))
    return ((train_x @ solution).reshape(shape) + mu,
            (test_x @ solution).reshape(len(test_x), *response.shape[1:]) + mu)


def by_horizon(x, component):
    if component == "full":
        return x
    mean = x.mean(axis=-1, keepdims=True)
    return x-mean if component == "shape" else mean


def gains(base_loss, new_loss):
    den = base_loss.mean(axis=0)
    if np.any(den < 1e-14):
        return None
    return 1-new_loss.mean(axis=0)/den


def stratified_draws(x, rng, count):
    """Preserve roughness and the entire repeated-measures block of each donor."""
    group = np.digitize(x[:, 8], np.quantile(x[:, 8], [.25, .5, .75]))
    strata = [np.flatnonzero(group == g) for g in np.unique(group)]
    for _ in range(count):
        mapping = np.arange(len(x))
        for ids in strata:
            mapping[ids] = rng.permutation(ids)
        yield mapping


def compare(train_x, test_x, src, dst, src_obs, dst_obs, *, component, use_stats, seed,
            draws, permutations):
    """Fit a regression in calibration only; recompute error ratios on resampled series."""
    src, dst, src_obs, dst_obs = (by_horizon(y, component) for y in (src, dst, src_obs, dst_obs))
    src_fit, src_test = fit_clean(train_x, src, test_x, use_stats)
    dst_fit, dst_test = fit_clean(train_x, dst, test_x, use_stats)
    a, b = src-src_fit, dst-dst_fit
    num = np.sum(a*b, axis=(0,2))
    denom = np.sum(a*a, axis=(0,2))
    coef = np.maximum(0, np.divide(num, denom, out=np.zeros_like(num), where=denom>1e-14))
    correction = (src_obs-src_test)*coef[None, :, None]
    residual = dst_obs-dst_test
    base = (residual**2).mean(axis=-1)
    new = ((residual-correction)**2).mean(axis=-1)
    gain = gains(base, new)
    if gain is None:
        return {"valid": False, "reason": "component has negligible baseline error"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(base), size=(draws, len(base)))
    den = base[idx].mean(axis=1)
    boot = 1-new[idx].mean(axis=1)/np.maximum(den, 1e-14)
    intervals = np.quantile(boot, [.025, .975], axis=0).T
    mean = float(gain.mean())
    output = {
        "valid": True, "n": len(base), "gain": mean,
        "ci95": np.quantile(boot.mean(axis=1), [.025, .975]).tolist(),
        "per_edit_gain": gain.tolist(), "per_edit_ci95": intervals.tolist(),
        "calibration_scalar": coef.tolist(), "per_edit_sse_base": base.sum(axis=0).tolist(),
        "per_edit_sse_augmented": new.sum(axis=0).tolist(),
        "mean_paired_win_fraction": float(((base > new) + .5*(base == new)).mean()),
    }
    if permutations:
        null = np.empty(permutations)
        for i, p in enumerate(stratified_draws(test_x, rng, permutations)):
            permuted = ((residual-correction[p])**2).mean(axis=-1)
            null[i] = gains(base, permuted).mean()
        output.update(null_mean=float(null.mean()),
                      null_interval95=np.quantile(null, [.025, .975]).tolist(),
                      perm_p=(1+int((null >= mean).sum()))/(1+len(null)))
    return output


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adjusted = np.empty_like(ps)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(ps[order]*np.arange(len(ps), 0, -1)))
    return adjusted


def alignment(a, b):
    norms = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    valid = norms > 1e-10
    values = np.sum(a*b, axis=-1)[valid]/norms[valid]
    return {"n": len(values), "median_cosine": float(np.median(values)) if len(values) else None}


def analyze(path, draws=2000, permutations=1999):
    with np.load(path, allow_pickle=False) as raw:
        d = {k: raw[k] for k in raw.files}
    split = d["split"]
    train = split == "train"
    domains = ("m4_daily", "m4_weekly", "LOOP_SEATTLE")
    models = ("bolt", "timesfm")
    x = d["summaries"]
    edits = d["edits"].tolist()
    results = dict(primary={}, template={}, components={}, fixed_moments={}, diagnostics={},
                   edits=edits, bootstrap_draws=draws, permutation_draws=permutations)
    primaries = []
    for j, domain in enumerate(domains):
        test = split == domain
        diag = {}
        for source in models:
            center = d[source][test, 4, :].astype(np.float64)
            diag[source+"_translation"] = {"max_abs_error": float(np.max(abs(center-.35))),
                                          "median_abs_error":float(np.median(abs(center-.35)))}
            endpoint_change = np.array([.35, .35*np.sin(2*np.pi*31/8), .35, 0, .35, 0])
            y = d[source][test].astype(np.float64)
            energy = np.square(y).sum(axis=(0, 2))
            level_only = y.mean(axis=-1, keepdims=True)
            diag[source+"_endpoint_persistence"] = {
                "recovery_vs_zero": (1 - np.square(y-endpoint_change[None, :, None]).sum((0,2))/energy).tolist(),
                "horizon_level_energy_fraction": (1 - np.square(y-level_only).sum((0,2))/energy).tolist(),
                "note": "Descriptive check added after the primary analysis. Endpoints follow the known raw edit."
            }
        for i, edit in enumerate(edits):
            diag[edit+"_alignment"] = alignment(d["bolt"][test, i], d["timesfm"][test, i])
        diag["fixed_moments_input_dose_range"] = np.quantile(d["rms"][test, 5], [0,.5,1]).tolist()
        results["diagnostics"][domain] = diag
        for k, source in enumerate(models):
            dest = models[1-k]
            name = f"{domain}/{source}_to_{dest}"
            a, b = [d[model].astype(np.float64) for model in (source, dest)]
            seed = 97313 + 100*j + 10*k
            indices = slice(0,4)
            arrays = (x[train], x[test], a[train, indices], b[train, indices], a[test, indices], b[test, indices])
            primary = compare(*arrays, component="full", use_stats=True, seed=seed, draws=draws,
                              permutations=permutations)
            results["primary"][name] = primary
            primaries.append(primary)
            results["template"][name] = compare(*arrays, component="full", use_stats=False,
                                                seed=seed, draws=draws, permutations=0)
            results["components"][name] = {kind: compare(*arrays, component=kind, use_stats=True,
                                                        seed=seed, draws=draws, permutations=0)
                                           for kind in ("level", "shape")}
            rms = d["rms"][:, 5]
            tt, te = train & (rms>1e-8), test & (rms>1e-8)
            norm = np.maximum(1e-8, rms)[:, None, None]
            a_d, b_d = a[:, 5:6, :]/norm, b[:, 5:6, :]/norm
            arrays = (x[tt], x[te], a_d[tt], b_d[tt], a_d[te], b_d[te])
            results["fixed_moments"][name] = compare(*arrays, component="full", use_stats=True,
                                                    seed=seed, draws=draws, permutations=0)
    if permutations:
        adj = holm([row["perm_p"] for row in primaries])
        for row, val in zip(primaries, adj, strict=True):
            row["perm_p_holm"] = float(val)
            row["passes_registered_rule"] = bool(row["gain"]>0 and val<.05)
    results["responses_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=HERE/"responses.npz")
    parser.add_argument("--out", type=Path, default=HERE/"results.json")
    args = parser.parse_args()
    result = analyze(args.responses)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for name, row in result["primary"].items():
        print(name, row["gain"], row["ci95"], row["perm_p_holm"])
