"""Confirm cross-size transport of a named physical ramp on untouched station/date combinations."""
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import OUT as EXTERNAL, WORK, panel
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import EndPrefix, cached_middle
from research.activation_transfer.moment.input_paths.run import modify
from research.activation_transfer.moment.token_paths.geometry import layers, load

OUT = Path(__file__).resolve().parent / "results"
BATCH = 16


def capture(model, x):
    chunks = []
    def take(_, inputs, outputs):
        chunks.append(outputs[0][:, 61].detach().cpu().numpy().copy())
        raise EndPrefix()

    handle = model.module.get_submodule(layers(model)["middle"]).register_forward_hook(take)
    try:
        for i in range(0, len(x), BATCH):
            try:
                model.predict(x[i:i+BATCH], 16, [.5])
            except EndPrefix:
                continue
            raise RuntimeError("Missing middle-layer capture")
    finally:
        handle.remove()
    return np.concatenate(chunks)


def score(waves, n, t):
    rng = np.random.default_rng(22112418)
    sw = rng.multinomial(n, np.full(n, 1/n), 3000)
    tw = rng.multinomial(t, np.full(t, 1/t), 3000)
    def totals(arr):
        v = (arr**2).sum(1).reshape(n, t)
        return np.r_[v.sum(), np.einsum("bs,st,bt->b", sw, v, tw)]
    def describe(v):
        return [float(v[0]), *np.quantile(v[1:], [.025, .975]).tolist()]
    out = {}
    for side, sl in (("early", slice(8)), ("late", slice(8,16))):
        for mode in ("shape", "full"):
            def select(z):
                a = z[:, sl].astype(float)
                return a - a.mean(1, keepdims=True) if mode == "shape" else a
            truth = select(waves["reference"])
            den = totals(truth)
            scores = {k: 1 - totals(select(z)-truth)/den for k, z in waves.items() if k != "reference"}
            for arm in ("wrong_edit", "other_station"):
                scores["mapped_minus_"+arm] = scores["mapped"] - scores[arm]
            out[side+"_"+mode] = {k: describe(v) for k,v in scores.items()}
            out[side+"_"+mode]["reference_RMS_C"] = float(np.sqrt(np.mean(truth**2)))
    return out


def saved_state(model, cache, label, values):
    sig = sha256(values.tobytes()).hexdigest()
    path = cache / (label + ".npz")
    if not path.exists():
        arr = capture(model, values)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, arr=arr, sig=sig)
        tmp.replace(path)
    with np.load(path) as f:
        assert str(f["sig"]) == sig
        return f["arr"]


def equal_length(candidate, reference):
    lengths = np.linalg.norm(reference, axis=1)
    z = candidate * (lengths / np.linalg.norm(candidate, axis=1))[:,None]
    np.testing.assert_allclose(np.linalg.norm(z,axis=1), lengths, rtol=1e-9)
    return z


def finite_waves(model, x, full, base, arms, out_dir, ids, dates):
    site = layers(model)["middle"]
    sel = np.unique(np.linspace(0, len(x)-1, 12, dtype=int))
    clean = model.predict(x[sel], 16, [.5])["point"]
    waves, replay = {}, {}
    for arm, delta in arms.items():
        armfile = out_dir / (arm + ".npz")
        sig = sha256(delta.tobytes() + x.tobytes()).hexdigest()
        if not armfile.exists():
            effects = []
            for i in range(0, len(x), BATCH):
                sl = slice(i, i+BATCH)
                with cached_middle(model, full[sl].copy(), site), addition(model, delta[sl]):
                    effects.append(model.predict(x[sl], 16, [.5])["point"] - base[sl])
            wave = np.concatenate(effects)
            with addition(model, delta[sel]):
                actual = model.predict(x[sel], 16, [.5])["point"] - clean
            err = float(np.max(abs(actual-wave[sel])))
            assert err < .00015, err
            tmp = armfile.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, wave=wave, native_error_C=err, signature=sig, ids=ids, dates=dates)
            tmp.replace(armfile)
        with np.load(armfile) as f:
            assert str(f["signature"]) == sig
            waves[arm] = f["wave"]
            replay[arm] = float(f["native_error_C"])
        print(arm, flush=True)
    return waves, replay


def run():
    ids, dates, x = panel()
    with np.load(EXTERNAL / "ordinary.npz") as f:
        large_base, base = f["last"], f["base"]
        np.testing.assert_array_equal(ids, f["ids"])
        np.testing.assert_array_equal(dates, f["dates"])
        assert str(f["input_sha256"]) == sha256(x.tobytes()).hexdigest()
    mapfile = MATCH / "small_to_large_map.npz"
    with np.load(mapfile) as f:
        w = f["w"].astype(float)
    cache = WORK / "physical_ramp_confirmation"
    cache.mkdir(exist_ok=True)
    small = load("small")
    sbase = saved_state(small, cache, "small_base", x)
    mapped = (saved_state(small, cache, "small_ramp", modify(x, "ramp")) - sbase) @ w
    wrong = (saved_state(small, cache, "small_recent", modify(x, "recent")) - sbase) @ w
    del small
    large = load("large")
    reference = saved_state(large, cache, "large_ramp", modify(x, "ramp")) - large_base
    order = np.random.default_rng(22112418).permutation(len(ids))
    partner = np.empty(len(ids), int)
    partner[order] = np.roll(order, 1)
    assert np.all(partner != np.arange(len(ids)))
    ix = (len(dates)*partner[:,None] + np.arange(len(dates))).ravel()
    arms = dict(reference=reference, mapped=mapped, other_station=equal_length(mapped[ix], mapped), wrong_edit=equal_length(wrong, mapped))
    full = np.load(WORK / "live_hidden.npy", mmap_mode="r")
    np.testing.assert_array_equal(full[:,61], large_base)
    waves, replay = finite_waves(large, x, full, base, arms, OUT, ids, dates)
    out = dict(stations=len(ids), dates=len(dates), station_ids=ids.tolist(), origins=list(dates),
               native_max_C=replay, map_sha256=sha256(mapfile.read_bytes()).hexdigest(), other_station_index=partner.tolist(),
               outcomes=score(waves, len(ids), len(dates)))
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
