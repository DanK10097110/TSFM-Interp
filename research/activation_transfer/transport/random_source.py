"""Train bridges from architecture-matched random source features to a frozen pretrained target."""

import argparse
import gc
import hashlib
import json

import numpy as np
import torch
from scipy.linalg import eigvalsh

from research.activation_transfer.transport.analyze import fit_bridge, metrics, sufficient
from research.activation_transfer.transport.capture import REVISIONS, load_adapter, run_pass, signed_response, site
from research.activation_transfer.transport.natural import SOURCES, STUDY, contexts
from research.activation_transfer.transport.prepare import CONTEXT, DEFAULT_CACHE, HORIZON
from research.activation_transfer.transport.rank import energy_stats

SEEDS = (2026112091, 2026112092)


def geometry(x):
    """Describe feature strength and the variance spectrum independently of hidden width."""
    rows = x.reshape(-1, x.shape[-1]).astype(np.float64)
    centered = rows - rows.mean(axis=0)
    kernel = centered @ centered.T if len(rows) < rows.shape[1] else centered.T @ centered
    eigen = np.maximum(eigvalsh(kernel), 0)[::-1]
    positive = eigen[eigen > max(eigen[0] * 1e-14, 1e-30)]
    if not len(positive):
        return {"rms": float(np.sqrt(np.mean(rows**2))), "stable_rank": 0.0, "effective_rank": 0.0}
    p = positive / positive.sum()
    return dict(rms=float(np.sqrt(np.mean(rows**2))), centered_rms=float(np.sqrt(np.mean(centered**2))),
                stable_rank=float(1 / p[0]), effective_rank=float(np.exp(-np.sum(p * np.log(p)))),
                rank_90=int(np.searchsorted(np.cumsum(p), .90) + 1),
                rank_99=int(np.searchsorted(np.cumsum(p), .99) + 1),
                numerical_rank=len(positive), top_variance=p[:12].tolist())


def state_fingerprint(module):
    h = hashlib.sha256()
    for name, values in module.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(values.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def block_change(adapter, x):
    """Measure first-to-middle residual-stream change on a train-only batch."""
    layer_names = adapter.all_layer_names()
    layer, tokens = site(adapter)
    states = {}
    def record(name):
        def save(module, args, kwargs):
            v = args[0] if args else kwargs["hidden_states"]
            states[name] = v.detach()[:, tokens, :].flatten(1).cpu().numpy().copy()
        return save
    handles = [adapter.module.get_submodule(name).register_forward_pre_hook(record(name), with_kwargs=True)
               for name in [layer_names[0], layer]]
    try:
        with torch.inference_mode():
            adapter.predict(x, HORIZON, [.5])
    finally:
        for h in handles:
            h.remove()
    first = states[layer_names[0]]
    change = states[layer] - first
    return {"relative_l2": float(np.linalg.norm(change) / max(np.linalg.norm(first), 1e-12)),
            "max_absolute": float(np.max(np.abs(change)))}


def capture_random(name, mode, seed, fixed, data, cache):
    """Produce only source-side states. Each random architecture is independently constructed."""
    torch.manual_seed(seed)
    adapter = load_adapter(name, random_init=True)
    norms_changed = 0
    if mode == "active":
        if name != "timesfm":
            raise ValueError("the active-normalization variant applies only to TimesFM")
        from timesfm.torch.normalization import RMSNorm
        with torch.no_grad():
            for module in adapter.module.modules():
                if isinstance(module, RMSNorm):
                    if torch.count_nonzero(module.scale):
                        raise ValueError("active variant assumes zero norm scales at factory initialization")
                    module.scale.fill_(1)
                    norms_changed += 1
        if not norms_changed:
            raise ValueError("no RMSNorm modules found")
    x = fixed["x"][fixed["split"] == "train"]
    metadata = {"name": name, "seed": seed, "mode": mode, "norms_changed": norms_changed,
                "fingerprint": state_fingerprint(adapter.module), "block_change": block_change(adapter, x[:16])}
    clean = run_pass(adapter, x)[0]
    _, hi, lo = contexts(data["segment"])
    pos = run_pass(adapter, hi.reshape(-1, CONTEXT))[0]
    neg = run_pass(adapter, lo.reshape(-1, CONTEXT))[0]
    delta = ((pos - neg) / 2).reshape(len(hi), hi.shape[1], -1)
    if not np.isfinite(delta).all() or not np.isfinite(clean).all():
        raise ValueError("random state contains nonfinite coordinates")
    metadata["train_clean_geometry"] = geometry(clean)
    metadata["delta_geometry"] = {s: geometry(delta[data["group"] == s]) for s, _, _, _ in SOURCES}
    file = cache / f"random_{name}_{mode}_{seed}.npz"
    np.savez_compressed(file, clean=clean, delta=delta, metadata=json.dumps(metadata))
    print(f"{name} {mode} seed {seed} captured", flush=True)
    return file


def summarize_group(all_predictions, truth, modes=("factory", "active")):
    """Average seed-specific score estimates and contrast with the paired trained-source score."""
    n = len(truth)
    rng = np.random.default_rng(2026112093)
    weights = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1/n), size=4000)]).astype(float)
    def summarize(v):
        return {"value": float(v[0]), "ci95": np.quantile(v[1:], [.025, .975]).tolist()}
    views = {}
    for name, hcenter, scenter in (("full", False, False), ("shape", True, False),
                                   ("background", False, True), ("background_shape", True, True)):
        def transform(v):
            return v - v.mean(axis=-1, keepdims=True) if hcenter else v
        target = transform(truth)
        scores = {k: metrics(energy_stats(transform(v), target, weights, scenter))["recovery"]
                  for k, v in all_predictions.items()}
        out = {k: summarize(v) for k, v in scores.items()}
        for mode in modes:
            keys = [f"{mode}_{seed}" for seed in SEEDS]
            if not all(k in scores for k in keys):
                continue
            avg = np.mean([scores[k] for k in keys], axis=0)
            out[mode + "_average"] = summarize(avg)
            out["trained_minus_" + mode] = summarize(scores["trained"] - avg)
            out[mode + "_minus_raw"] = summarize(avg - scores["raw"])
        views[name] = out
    return views


def main():
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=REVISIONS)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=STUDY / "results")
    args = p.parse_args()
    source_name = args.source
    target_name = next(name for name in REVISIONS if name != source_name)
    fixed = dict(np.load(args.cache / "data.npz"))
    data = dict(np.load(args.cache / "natural_data.npz"))
    x = contexts(data["segment"])[0]
    modes = ("factory", "active") if source_name == "timesfm" else ("factory",)
    files = {}
    for mode in modes:
        for seed in SEEDS:
            q = args.cache / f"random_{source_name}_{mode}_{seed}.npz"
            if not q.exists():
                capture_random(source_name, mode, seed, fixed, data, args.cache)
            files[mode, seed] = q
            gc.collect()
    target = load_adapter(target_name)
    source = load_adapter(source_name)
    train = fixed["split"] == "train"
    y = np.load(args.cache / (target_name + ".npz"))["clean"][train]
    source_tr = np.load(args.cache / (source_name + ".npz"))["clean"][train]
    ref = dict(np.load(args.cache / ("natural_" + target_name + ".npz")))
    src_delta = np.load(args.cache / ("natural_" + source_name + ".npz"))["delta"]
    source_metadata = dict(block_change=block_change(source, fixed["x"][train][:16]),
                           train_clean_geometry=geometry(source_tr),
                           delta_geometry={s: geometry(src_delta[data["group"] == s]) for s, _, _, _ in SOURCES})
    del source
    gc.collect()
    stem = source_name + "_to_" + target_name
    original = dict(np.load(args.out / (stem + "_natural.npz")))
    np.testing.assert_array_equal(original["ids"], data["ids"])
    effects = {"trained": original["bridge"], "raw": original["raw"], "shuffle": original["shuffle"]}
    info = {}
    for (mode, seed), path in files.items():
        random = dict(np.load(path))
        label = f"{mode}_{seed}"
        model, settings = fit_bridge(random["clean"], y)
        pred = model.transport(random["delta"])
        effects[label] = signed_response(target, x, pred)
        info[label] = dict(metadata=json.loads(str(random["metadata"])), fit=settings,
                           hidden_metrics={s: {k: float(v) for k, v in metrics(sufficient(
                               pred[data["group"] == s], ref["delta"][data["group"] == s]).sum(axis=0)).items()}
                               for s, _, _, _ in SOURCES})
        print(stem, label, "random-source patch complete", flush=True)
    results = {"trained_geometry": source_metadata, "randoms": info, "groups": {}}
    for s, _, _, _ in SOURCES:
        sub = data["group"] == s
        results["groups"][s] = summarize_group({k: v[sub] for k, v in effects.items()}, ref["native"][sub])
        print(stem, s, {k: round(v["value"], 3) for k, v in results["groups"][s]["full"].items()}, flush=True)
    np.savez_compressed(args.out / (stem + "_random_source.npz"), ids=data["ids"], group=data["group"],
                        native=ref["native"], **effects)
    (args.out / (stem + "_random_source.json")).write_text(json.dumps(results, separators=(",", ":"),
                                                                       allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
