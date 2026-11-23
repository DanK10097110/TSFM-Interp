"""Patch low-rank approximations to an alignment map fitted only on clean training states."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from research.activation_transfer.transport.analyze import fit_bridge, metrics, scaled_like
from research.activation_transfer.transport.capture import REVISIONS, load_adapter, signed_response
from research.activation_transfer.transport.natural import SHIFTS, SOURCES, STUDY, contexts
from research.activation_transfer.transport.prepare import DEFAULT_CACHE


RANKS = (1, 4, 16, 64)
SEEDS = (2026112081, 2026112082, 2026112083)


def fitted_basis(model, training_states):
    """Training-side reduced-rank regression directions, ordered by fitted state variance."""
    fitted = model.predict(training_states) - model.target_mean
    _, singular, basis = svd(fitted, full_matrices=False, check_finite=False)
    rank = np.count_nonzero(singular > 1e-8 * singular[0])
    return singular[:rank], basis[:rank]


def projected(x, basis):
    """Orthogonal projection in destination hidden coordinates, also for batches of contrasts."""
    return (x @ basis.T) @ basis


def alternatives(x, singular, basis):
    """Yield principal and equally long random projections with fixed nested axes."""
    orientation = {}
    for kind in ("uniform", "covariance"):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            g = rng.standard_normal((len(singular), max(RANKS)))
            if kind == "covariance":
                g *= singular[:, None]
            q, _ = np.linalg.qr(g, mode="reduced")
            orientation[kind, seed] = q.T @ basis
    for rank in RANKS:
        principal = projected(x, basis[:rank])
        yield f"principal_{rank}", principal
        for (kind, seed), axes in orientation.items():
            random = projected(x, axes[:rank])
            yield f"{kind}_{rank}_{seed}", scaled_like(random, principal)


def energy_stats(prediction, reference, weights, centered):
    """One dot/squared norm per resample; recompute the mean for each bootstrap."""
    n = len(reference)
    p = prediction.reshape(n, -1).astype(float)
    t = reference.reshape(n, -1).astype(float)
    xy = weights @ np.einsum("ij,ij->i", p, t)
    xx = weights @ np.einsum("ij,ij->i", p, p)
    yy = weights @ np.einsum("ij,ij->i", t, t)
    if centered:
        ps, ts = weights @ p, weights @ t
        xy -= np.einsum("ij,ij->i", ps, ts) / n
        xx -= np.einsum("ij,ij->i", ps, ps) / n
        yy -= np.einsum("ij,ij->i", ts, ts) / n
    return np.column_stack([xy, np.maximum(yy, 0), np.maximum(xx, 0)])


def summarize_curves(predictions, target, full):
    """Compare rank curves to controls by paired series bootstrap with a frozen rank grid."""
    n = len(target)
    rng = np.random.default_rng(2026112084)
    weights = rng.multinomial(n, np.full(n, 1. / n), size=3000).astype(float)
    weights = np.vstack((np.ones(n), weights))
    def ci(v):
        return dict(value=float(v[0]), ci95=np.quantile(v[1:], [.025, .975]).tolist())
    views = {}
    for view, hcenter, scenter in (("full", False, False), ("shape", True, False),
                                   ("background", False, True), ("background_shape", True, True)):
        def transform(v):
            return v - v.mean(axis=-1, keepdims=True) if hcenter else v
        ref = transform(target)
        full_transformed = transform(full)
        scores = {}
        full_recovery = {}
        for name, data in {"full": full, **predictions}.items():
            val = transform(data)
            scores[name] = metrics(energy_stats(val, ref, weights, scenter))["recovery"]
            if name.startswith("principal"):
                full_recovery[name] = metrics(energy_stats(val, full_transformed, weights, scenter))["recovery"]
        curves = {}
        for rank in RANKS:
            key = f"principal_{rank}"
            value = scores[key]
            comparison = {}
            for kind in ("uniform", "covariance"):
                nulls = np.stack([scores[f"{kind}_{rank}_{seed}"] for seed in SEEDS])
                comparison[kind] = dict(average=ci(nulls.mean(axis=0)),
                                        seed_values=nulls[:, 0].tolist(),
                                        paired_difference=ci(value - nulls.mean(axis=0)))
            curves[str(rank)] = dict(score=ci(value), versus_full_response=ci(full_recovery[key]),
                                     principal_minus_full=ci(value - scores["full"]), controls=comparison)
        views[view] = dict(full_map=ci(scores["full"]), rank=curves)
    return views


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", choices=REVISIONS)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=STUDY / "results")
    p.add_argument("--batch-size", type=int, default=48)
    args = p.parse_args()
    target_name = args.target
    source_name = next(name for name in REVISIONS if name != target_name)
    train = np.load(args.cache / "data.npz")["split"] == "train"
    x_tr = np.load(args.cache / (source_name + ".npz"))["clean"][train]
    y_tr = np.load(args.cache / (target_name + ".npz"))["clean"][train]
    bridge, _ = fit_bridge(x_tr, y_tr)
    singular, basis = fitted_basis(bridge, x_tr)
    nt = dict(np.load(args.cache / "natural_data.npz"))
    x = contexts(nt["segment"])[0]
    d = bridge.transport(np.load(args.cache / ("natural_" + source_name + ".npz"))["delta"])
    native = np.load(args.cache / ("natural_" + target_name + ".npz"))["native"]
    np.testing.assert_allclose(projected(d, basis), d, atol=2e-5, rtol=2e-5)
    prefix = source_name + "_to_" + target_name
    adapter = load_adapter(target_name)
    responses = {}
    progress = args.cache / (prefix + "_rank")
    progress.mkdir(exist_ok=True)
    for key, edit in alternatives(d, singular, basis):
        # Resume a given condition only for this exact transported edit and input matrix.
        import hashlib
        fingerprint = hashlib.sha256(nt["ids"].tobytes() + x.tobytes() + edit.tobytes()).hexdigest()
        file = progress / (key + "_" + fingerprint[:16] + ".npz")
        if file.is_file():
            responses[key] = np.load(file)["response"]
            continue
        print(prefix, "natural rank", key, flush=True)
        responses[key] = signed_response(adapter, x, edit, batch_size=args.batch_size)
        np.savez_compressed(file, response=responses[key])
    full_archive = dict(np.load(args.out / (prefix + "_natural.npz")))
    np.testing.assert_array_equal(nt["ids"], full_archive["ids"])
    np.savez_compressed(args.out / (prefix + "_rank.npz"), ids=nt["ids"], group=nt["group"],
                        shifts=SHIFTS, native=native, full=full_archive["bridge"], **responses)
    out = dict(source=source_name, target=target_name, train_rank=len(singular),
               fraction_clean_fitted_variance={str(r): float(np.sum(singular[:r] ** 2) / np.sum(singular ** 2))
                                               for r in RANKS}, groups={})
    for name, _, _, _ in SOURCES:
        rows = nt["group"] == name
        out["groups"][name] = summarize_curves({k: v[rows] for k, v in responses.items()},
                                                native[rows], full_archive["bridge"][rows])
    (args.out / (prefix + "_rank.json")).write_text(json.dumps(out, separators=(",", ":"),
                                                               allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
