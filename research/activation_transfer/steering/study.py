"""Move fixed source cue directions through maps learned only from paired clean histories."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from scipy.linalg import cho_factor, cho_solve

from tsfm_lens.config import load_config
from tsfm_lens.pipeline import Context
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite, _leaves

ROOT = Path(__file__).resolve().parents[3]
REVISION = "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
DATA_ROOT = Path.home() / ".cache/huggingface/hub/datasets--Salesforce--GiftEval/snapshots" / REVISION
MODELS = ["Chronos-Bolt", "TimesFM"]
CHECKPOINTS = {
    "Chronos-Bolt": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    "TimesFM": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
}
CUES = ["recent_reversal", "period_8"]
DOMAINS = ["daily", "weekly"]
DEPTHS = [0.25, 0.5, 0.75]
HORIZON = 8


def load_histories(dataset, count, seed, root=DATA_ROOT, context=64, dedupe=None):
    """Sample one buffered, standardized context per distinct eligible item."""
    path = root / ("m4_" + dataset) / "data-00000-of-00001.arrow"
    with pa.memory_map(str(path)) as f:
        table = pa.ipc.open_stream(f).read_all()
    ids, histories, seen = [], [], set()
    for row in np.random.default_rng(seed).permutation(len(table)):
        item = str(table["item_id"][int(row)].as_py())
        raw = np.asarray(table["target"][int(row)].as_py(), dtype=np.float64)[-(context + HORIZON):-HORIZON]
        if item in seen or raw.shape != (context,) or not np.isfinite(raw).all() or raw.std() < 1e-7:
            continue
        normalized = ((raw - raw.mean()) / raw.std()).astype("f4")
        if dedupe is not None:
            digest = sha256(normalized.tobytes()).digest()
            if digest in dedupe:
                continue
            dedupe.add(digest)
        seen.add(item)
        ids.append(item)
        histories.append(normalized)
        if len(ids) == count:
            break
    if len(ids) != count:
        raise ValueError(f"Only {len(ids)} eligible items in {dataset}")
    return np.asarray(ids), np.asarray(histories, dtype=np.float32)


def cues():
    """Signed cues in standardized input units, both zero at the endpoint."""
    recent = np.r_[np.linspace(0, -1, 16), np.linspace(-1, 0, 16)]
    periodic = np.sin(2 * np.pi * (np.arange(32) - 31) / 8)
    return 0.35 * np.pad(np.stack([recent, periodic]), [(0, 0), (32, 0)]).astype("f4")


def sites_and_states(model, layers, inputs, batch_size):
    """Capture only the final vector at each chosen layer on the live steering forecast path."""
    sites, output = {}, {layer: [] for layer in layers}
    def capture(layer):
        def hook(module, args, kwargs):
            path, value = next((p, v) for p, v in _leaves(dict(args=args, kwargs=kwargs))
                               if v.ndim == 3 and v.is_floating_point())
            sites[layer] = InterventionSite(layer, list(path), 0, list(value.shape[1:]), "last")
            output[layer].append(value[:, -1, :].detach().float().cpu().numpy().copy())
        return hook
    with ExitStack() as stack:
        for layer in layers:
            handle = model.module.get_submodule(layer).register_forward_pre_hook(capture(layer), with_kwargs=True)
            stack.callback(handle.remove)
        for start in range(0, len(inputs), batch_size):
            model.forecast(inputs[start:start + batch_size], HORIZON)
    states = {layer: np.concatenate(chunks) for layer, chunks in output.items()}
    if any(len(v) != len(inputs) for v in states.values()):
        raise ValueError("Sites must execute exactly once per input")
    return sites, states


def learn_directions(source, target, cue_vectors, perms, penalty=0.01):
    """Dual ridge displacement map and dose-matched correspondence shams."""
    x = source.astype(float) - source.mean(axis=0, dtype=float)
    y = target.astype(float) - target.mean(axis=0, dtype=float)
    gram = x @ x.T
    ridge = penalty * np.linalg.eigvalsh(gram)[-1]
    if ridge <= 0:
        raise ValueError("Source states have no variance")
    weights = cho_solve(cho_factor(gram + ridge * np.eye(len(x))), x @ cue_vectors.T).T
    mapped = weights @ y
    shams = np.stack([weights @ y[p] for p in perms], axis=1)
    norms = np.linalg.norm(shams, axis=2, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Degenerate shuffled edit")
    shams *= np.linalg.norm(mapped, axis=1)[:, None, None] / norms
    return mapped, shams


def patch_effect(model, site, contexts, state_norms, directions, batch_size):
    """Central difference per history and direction, using the actual SteeringModel edit."""
    directions = np.asarray(directions, dtype=float)
    n, k = len(contexts), len(directions)
    flat = np.arange(n * k * 2)
    out = []
    for start in range(0, len(flat), batch_size):
        idx = flat[start:start + batch_size]
        row, arm, sign = idx // (2 * k), (idx // 2) % k, 1 - 2 * (idx % 2)
        coef = np.zeros((len(row), k))
        # The hook multiplies by the pre-edit state's norm. Undo it for a fixed, absolute translation.
        coef[np.arange(len(row)), arm] = sign / np.maximum(state_norms[row], 1e-6)
        out.append(model.evaluate(contexts[row], HORIZON, site, directions, coef)[..., 0])
    out = np.concatenate(out).reshape(n, k, 2, HORIZON)
    return (out[:, :, 0] - out[:, :, 1]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Fresh or resumable intermediate directory")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    batch = args.batch_size
    torch.set_num_threads(4)
    daily_ids, daily = load_histories("daily", 704, 202611201, args.data_root)
    weekly_ids, weekly = load_histories("weekly", 96, 202611202, args.data_root)
    contexts = np.concatenate([daily[608:], weekly])
    fit_cue = daily[512:608]
    input_sets = [daily[:512], contexts]
    for pattern in cues():
        input_sets.extend([fit_cue + pattern, fit_cue - pattern])
    data = np.concatenate(input_sets)
    cfg = load_config(ROOT / "examples/steering/all_types.yaml")
    for spec in cfg.models:
        repo = "models--" + spec.checkpoint.replace("/", "--")
        spec.checkpoint = str(Path.home()/".cache/huggingface/hub"/repo/"snapshots"/CHECKPOINTS[spec.name])
    ctx = Context(cfg)
    models, caches, sites = {}, {}, {}
    for name in MODELS:
        adapter = ctx.hub.get(name)
        model = SteeringModel.from_adapter(adapter, quantiles=[0.5])
        all_layers = adapter.all_layer_names()
        layers = [all_layers[int(frac * len(all_layers))] for frac in DEPTHS]
        cache = args.out / (name + "_states.npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                if not np.array_equal(z["data_digest"], sha256(data.tobytes()).hexdigest()):
                    raise ValueError("Cache inputs changed")
                states = {layer: z[layer] for layer in layers}
            site, singleton = sites_and_states(model, layers, data[:1], 1)
            for layer in layers:
                np.testing.assert_allclose(singleton[layer][0], states[layer][0], atol=2e-4, rtol=1e-4)
        else:
            site, states = sites_and_states(model, layers, data, batch)
            np.savez_compressed(cache, **states, data_digest=sha256(data.tobytes()).hexdigest())
        models[name], sites[name], caches[name] = model, list(site[l] for l in layers), list(states[l] for l in layers)
        print("captured", name, layers, [v.shape for v in states.values()], flush=True)
    rng = np.random.default_rng(202611203)
    perms = [rng.permutation(512) for _ in range(8)]
    metadata = {"models": MODELS, "cues": CUES, "depths": DEPTHS, "series_ids": {"clean_fit":daily_ids[:512].tolist(),
                "cue_fit":daily_ids[512:608].tolist(),"daily":daily_ids[608:].tolist(),"weekly":weekly_ids.tolist()},
                "data_revision": REVISION, "input_sha256":sha256(data.tobytes()).hexdigest(), "sites":{}, "vectors":[],
                "checkpoint_revision":{}, "torch":str(torch.__version__)}
    for name in MODELS:
        metadata["checkpoint_revision"][name] = CHECKPOINTS[name]
        metadata["sites"][name] = [dict(layer=s.module, path=s.path, shape=s.shape) for s in sites[name]]
    for j, depth in enumerate(DEPTHS):
        native = {}
        for name in MODELS:
            states = caches[name][j]
            signed = states[704:].reshape(len(CUES),2,96,-1).astype(float)
            native[name] = (signed[:,0] - signed[:,1]).mean(1)/2
        for target in MODELS:
            source = next(m for m in MODELS if m != target)
            xs, ys = caches[source][j][:512], caches[target][j][:512]
            mapped, shams = learn_directions(xs, ys, native[source], perms)
            model, site = models[target], sites[target][j]
            norms = np.linalg.norm(caches[target][j][512:704], axis=1)
            for c, cue in enumerate(CUES):
                reference = native[target][c]
                vec = mapped[c]
                directions = np.vstack([reference, vec, shams[c]])
                fingerprint = sha256(directions.tobytes() + contexts.tobytes()).hexdigest()
                row = dict(target=target, source=source, depth=depth, cue=cue,
                           cosine=float(vec @ reference / (np.linalg.norm(vec)*np.linalg.norm(reference))),
                           mapped_to_native_norm=float(np.linalg.norm(vec) / np.linalg.norm(reference)),
                           site=site.module, fingerprint=fingerprint)
                out = args.out / f"d{j}_{target}_{cue}.npz"
                if out.exists():
                    with np.load(out) as saved:
                        if str(saved["fingerprint"]) != fingerprint:
                            raise ValueError("Stored directions or data do not match")
                else:
                    ordinary = model.forecast(contexts[:3],HORIZON)
                    with_zero = model.evaluate(contexts[:3],HORIZON,site,directions[:1],np.zeros((3,1)))
                    np.testing.assert_allclose(ordinary, with_zero, atol=2e-5, rtol=2e-5)
                    effects = patch_effect(model, site, contexts, norms, directions, batch)
                    singleton = patch_effect(model, site, contexts[:1], norms[:1], directions[:2], 1)
                    np.testing.assert_allclose(effects[:1,:2], singleton, atol=2e-4, rtol=0.002)
                    np.savez_compressed(out, effects=effects.astype("f4"), fingerprint=fingerprint)
                metadata["vectors"].append(row)
                print("evaluated", depth, source, "->", target, cue, flush=True)
        (args.out/"metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
