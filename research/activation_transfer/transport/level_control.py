"""Test whether a public-input or training-template level improves transferred forecast shapes."""

import json

import numpy as np

from research.activation_transfer.transport.analyze import metrics, sufficient
from research.activation_transfer.transport.natural import STUDY


def summarize(archive, group_key):
    """Retain the bridge's horizon deviations and substitute another predictor's mean per item."""
    result = {}
    groups = archive[group_key]
    for group in np.unique(groups):
        rows = groups == group
        n = rows.sum()
        bridge, truth = archive["bridge"][rows], archive["native"][rows]
        reference = sufficient(bridge, truth)
        rng = np.random.default_rng(2026112097)
        w = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=4000)]).astype(float)
        baseline = metrics(w @ reference)["recovery"]
        levels = {}
        for name in ("raw", "forecast_template"):
            if name not in archive:
                continue
            level = archive[name][rows].mean(axis=-1, keepdims=True)
            hybrid = bridge - bridge.mean(axis=-1, keepdims=True) + level
            score = metrics(w @ sufficient(hybrid, truth))["recovery"]
            improvement = score - baseline
            frac_error = improvement / (1 - baseline)
            levels[name] = {"score": float(score[0]), "gain": float(improvement[0]),
                            "gain_ci95": np.quantile(improvement[1:], [.025,.975]).tolist(),
                            "fraction_bridge_error_removed": float(frac_error[0]),
                            "error_fraction_ci95": np.quantile(frac_error[1:], [.025,.975]).tolist()}
        result[str(group)] = {"n": int(n), "bridge": float(baseline[0]), "hybrids": levels}
    return result


def main():
    out = STUDY / "results"
    results = {}
    for prefix in ("timesfm_to_bolt", "bolt_to_timesfm"):
        original = dict(np.load(out / (prefix + ".npz")))
        controls = dict(np.load(out / (prefix + "_controls.npz")))
        np.testing.assert_array_equal(original["ids"], controls["ids"])
        original["forecast_template"] = controls["forecast_template"]
        natural = dict(np.load(out / (prefix + "_natural.npz")))
        results[prefix] = {"fixed": summarize(original, "split"), "natural": summarize(natural, "group")}
    (out / "level_control.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
