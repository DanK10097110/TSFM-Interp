"""Direct source-referenced transfer of fixed PCA and temporal-contrast interventions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.linalg import svd

from research.activation_transfer.transport.analyze import fit_bridge, metrics, scaled_like
from research.activation_transfer.transport.capture import REVISIONS, load_adapter, signed_response
from research.activation_transfer.transport.natural import STUDY
from research.activation_transfer.transport.prepare import CONCEPTS, CONTEXT, DEFAULT_CACHE, HORIZON, REVISION
from research.activation_transfer.transport.rank import energy_stats

NAMES = ("pc1", "pc4", "pc16", "pc64", *CONCEPTS)
COHORTS = [("m4_daily", "m4_daily", 128, 2026112121), ("SZ_TAXI", "SZ_TAXI/15T", 96, 2026112122)]


def material(cache, out, data_root):
    excluded = set(json.loads((out / "manifest.json").read_text())["items"])
    natural = json.loads((out / "natural_manifest.json").read_text())
    excluded.update(id_ for v in natural.values() for id_ in v["items"])
    controls = STUDY.parent / "controls"
    excluded.update(i["item"] for i in json.loads((controls / "items.json").read_text())["rows"])
    shape = json.loads((controls / "shape_items.json").read_text())
    excluded.update(i["item"] for v in shape.values() for i in v["rows"])
    atoms = json.loads((STUDY.parent / "atoms/results/manifest.json").read_text())["split"]
    excluded.update("m4_" + i for i in atoms if i.startswith("daily/"))
    ids, x, groups, manifest = [], [], [], {}
    seen = set()
    for key, loc, n, seed in COHORTS:
        path = data_root / loc / "data-00000-of-00001.arrow"
        with pa.memory_map(str(path)) as source:
            table = pa.ipc.open_stream(source).read_all()
        rows = []
        for i in np.random.default_rng(seed).permutation(len(table)):
            raw_id = str(table["item_id"][int(i)].as_py())
            identity = f"{key}/{raw_id}"
            if identity in excluded or (key == "m4_daily" and int(raw_id) < 2000):
                continue
            vals = table["target"][int(i)].values
            if len(vals) < CONTEXT + HORIZON:
                continue
            hist = vals.slice(len(vals) - HORIZON - CONTEXT, CONTEXT).to_numpy(zero_copy_only=False).astype(float)
            if not np.isfinite(hist).all() or hist.std() < 1e-8:
                continue
            hist = ((hist - hist.mean()) / hist.std()).astype(np.float32)
            fingerprint = hashlib.sha256(hist.tobytes()).digest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            excluded.add(identity)
            x.append(hist)
            ids.append(identity)
            rows.append(identity)
            groups.append(key)
            if len(rows) == n:
                break
        if len(rows) != n:
            raise RuntimeError(f"only {len(rows)} series available for {key}")
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        manifest[key] = dict(sha256=checksum, ids=rows, seed=seed)
    np.savez_compressed(cache / "axes_data.npz", ids=ids, group=groups, x=x)
    (out / "axes_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({k: len(v["ids"]) for k, v in manifest.items()}, flush=True)


def training_directions(clean, delta):
    """Four empirical-input axes and four input-labeled mean contrasts, fitted only in source space."""
    h = clean.astype(float) - clean.mean(axis=0, dtype=np.float64)
    _, s, basis = svd(h, full_matrices=False, check_finite=False)
    i = np.array([0, 3, 15, 63])
    dirs = basis[i] * (0.5 * s[i] / np.sqrt(len(clean) - 1))[:, None]
    signs = np.sign(np.take_along_axis(dirs, np.argmax(np.abs(dirs), axis=1)[:,None], axis=1))
    return np.vstack([dirs * signs, delta.mean(axis=0, dtype=float)])


def static_response(adapter, x, vectors):
    """Patch precisely the same source-trained vector on every evaluation history."""
    return signed_response(adapter, x, np.broadcast_to(vectors, (len(x), *vectors.shape)))


def native(model, cache):
    fixed = dict(np.load(cache / "data.npz"))
    original = dict(np.load(cache / (model + ".npz")))
    train = fixed["split"] == "train"
    d = training_directions(original["clean"][train], original["delta"][train])
    adapter = load_adapter(model)
    x = np.load(cache / "axes_data.npz")["x"]
    source = static_response(adapter, x, d)
    calib = static_response(adapter, fixed["x"][train][:96], d)
    np.savez_compressed(cache / ("axes_" + model + ".npz"), directions=d, source=source, calibration=calib)
    print(model, "fixed source effects captured", flush=True)


def summarize(predictions, source, ids):
    """Balance the source-energy-normalized scores of each of the four distinct axes."""
    n = len(ids)
    rng = np.random.default_rng(2026112123)
    weights = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1/n), size=3000)]).astype(float)
    def ci(x):
        return dict(value=float(x[0]), ci95=np.quantile(x[1:], [.025,.975]).tolist())
    result = {}
    for view, hc, bc in (("full", False, False), ("shape", True, False),
                          ("background", False, True)):
        trans = lambda z: z-z.mean(-1,keepdims=True) if hc else z
        target = trans(source)
        accum = {}
        for aug, pred in predictions.items():
            p = trans(pred)
            accum[aug] = {m: [] for m in ("recovery", "cosine")}
            for i in range(len(NAMES)):
                all_metrics = metrics(energy_stats(p[:,i], target[:,i], weights, bc))
                for m in accum[aug]:
                    accum[aug][m].append(all_metrics[m])
            accum[aug] = {m: np.stack(rows) for m,rows in accum[aug].items()}
        accum["wrong_mean"] = {m: np.mean([accum[f"wrong_{s}"][m] for s in (1,2,3)],axis=0)
                               for m in ("recovery", "cosine")}
        summarized = {}
        for label, span in [("pca", slice(0,4)), ("named", slice(4,8))]:
            estimates = {k: {m: a[span].mean(0) for m,a in detail.items()} for k,detail in accum.items()}
            summary = {k: {m: ci(a) for m,a in detail.items()} for k,detail in estimates.items()}
            summary["bridge_minus"] = {k: {m: ci(estimates["bridge"][m]-detail[m]) for m in detail}
                                       for k,detail in estimates.items() if k != "bridge"}
            summarized[label] = summary
        summarized["per_axis"] = {name: {k:{m:ci(a[i]) for m,a in detail.items()}
                                        for k,detail in accum.items()} for i,name in enumerate(NAMES)}
        result[view] = summarized
    result["source_rms"] = np.sqrt(np.mean(source.astype(float)**2,axis=(0,2))).tolist()
    return result


def evaluate(name, cache, out):
    dest = next(k for k in REVISIONS if k != name)
    data = dict(np.load(cache / "axes_data.npz"))
    native_data = dict(np.load(cache / ("axes_" + name + ".npz")))
    train = np.flatnonzero(np.load(cache / "data.npz")["split"] == "train")
    h, y = [np.load(cache / (m + ".npz"))["clean"] for m in (name, dest)]
    trained, _ = fit_bridge(h[train], y[train])
    perm = np.random.default_rng(2026112003).permutation(train)
    sham, _ = fit_bridge(h[train], y[perm])
    d = native_data["directions"]
    bridge = trained.transport(d)
    vectors = dict(bridge=bridge, shuffle=scaled_like(sham.transport(d), bridge))
    for s in (1,2,3):
        wrong = np.r_[np.roll(np.arange(4), -s), np.roll(np.arange(4,8), -s)]
        vectors[f"wrong_{s}"] = scaled_like(bridge[wrong], bridge)
    adapter = load_adapter(dest)
    preds = {}
    for method, edits in vectors.items():
        print(name, "->", dest, method, flush=True)
        preds[method] = static_response(adapter,data["x"],edits)
    preds["template"] = np.broadcast_to(native_data["calibration"].mean(0, dtype=float),
                                         native_data["source"].shape)
    stem = name + "_to_" + dest + "_axes"
    np.savez_compressed(out / (stem + ".npz"), source=native_data["source"], ids=data["ids"], groups=data["group"],
                        axes=NAMES, **preds)
    result = dict(source=name,target=dest,axes=NAMES,source_norm=np.linalg.norm(d,axis=1).tolist(),
                  mapped_norm=np.linalg.norm(bridge,axis=1).tolist(),groups={})
    for key, _, _, _ in COHORTS:
        idx = data["group"] == key
        row = summarize({k:v[idx] for k,v in preds.items()},native_data["source"][idx],data["ids"][idx])
        result["groups"][key] = row
        print(key, {k:round(row["full"][k]["bridge"]["recovery"]["value"],3) for k in ("pca","named")}, flush=True)
    (out / (stem + ".json")).write_text(json.dumps(result, separators=(",",":"), allow_nan=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare","native","evaluate"))
    p.add_argument("source", nargs="?", choices=REVISIONS)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out", type=Path, default=STUDY/"results")
    p.add_argument("--data-root", type=Path, default=Path.home()/".cache/huggingface/hub"/
                   "datasets--Salesforce--GiftEval/snapshots"/REVISION)
    args = p.parse_args()
    if args.stage == "prepare":
        material(args.cache,args.out,args.data_root)
    elif args.stage == "native":
        native(args.source,args.cache)
    else:
        evaluate(args.source,args.cache,args.out)


if __name__ == "__main__":
    main()
