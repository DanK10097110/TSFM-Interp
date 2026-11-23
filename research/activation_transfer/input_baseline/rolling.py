"""Compare the frozen raw-history and donor-state estimators on observed window shifts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.input_baseline.run import DEFAULT_CACHE, fit, load_adapter, pooled_evaluation, raw_state, run_pass, signed_response
from research.activation_transfer.transport.natural import contexts


def run(target, cache, output):
    source = "bolt" if target == "timesfm" else "timesfm"
    with np.load(cache / "data.npz") as original:
        train = original["split"] == "train"
        raw_training = original["x"][train]
    with np.load(cache / "natural_data.npz") as data:
        center, high, low = contexts(data["segment"])
        ids, groups = data["ids"], data["group"]
    with np.load(cache / (source + ".npz")) as s, np.load(cache / (target + ".npz")) as t:
        source_training, target_training = s["clean"][train], t["clean"][train]
    with np.load(cache / ("natural_" + target + ".npz")) as stored:
        native = stored["native"]
    source_model = load_adapter(source)
    source_states = []
    for windows in (high, low):
        states, _ = run_pass(source_model, windows.reshape(-1, windows.shape[-1]))
        source_states.append(states.reshape(*windows.shape[:2], -1))
    del source_model
    with np.load(cache / ("natural_" + source + ".npz")) as stored:
        np.testing.assert_allclose((source_states[0] - source_states[1])/2, stored["delta"],
                                   atol=6e-4, rtol=1e-3)
    target_model = load_adapter(target)
    rolling = np.stack([high, low])
    candidates = {"raw": (raw_state(raw_training), raw_state(rolling)),
                  "raw_recent": (raw_state(raw_training, True), raw_state(rolling, True)),
                  "source": (source_training, np.stack(source_states))}
    original_fits = json.loads((output / (target + ".json")).read_text())
    effects = {}
    for name, (x, z) in candidates.items():
        model, selection = fit(x, target_training)
        if selection["family"] != original_fits["fit"][name]["family"]:
            raise ValueError("Clean-training choice changed")
        np.testing.assert_allclose(selection["grid"], original_fits["fit"][name]["grid"])
        pair = model.predict(z.reshape(-1, z.shape[-1])).reshape(*z.shape[:-1], -1)
        effects[name] = signed_response(target_model, center, (pair[0] - pair[1])/2)
    metadata = dict(source=source, target=target, raw_choice=original_fits["raw_choice"], splits={})
    for group in np.unique(groups):
        mask = groups == group
        n = mask.sum()
        predictions = {k: v[mask] for k, v in effects.items()}
        predictions.update(bridge=predictions["source"], best_raw=predictions[metadata["raw_choice"]])
        draws = np.random.default_rng(2026112025).integers(n, size=(4000, n))
        metadata["splits"][str(group)] = pooled_evaluation(predictions, native[mask], draws)
    np.savez_compressed(output / ("rolling_" + target + ".npz"), ids=ids, groups=groups, native=native,
                        **{k: v.astype("f4") for k, v in effects.items()})
    (output / ("rolling_" + target + ".json")).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["bolt", "timesfm"])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    torch.set_num_threads(3)
    run(args.target, args.cache, args.out)
