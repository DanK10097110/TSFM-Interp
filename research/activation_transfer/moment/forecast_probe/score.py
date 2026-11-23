"""Compare true prospective forecasts at matched unit, independently clustered by item."""
import json

import numpy as np

from research.activation_transfer.moment.forecast_probe.capture import HEADS, MOMENT, OUT, QUALITY, WEATHER, rows
from research.activation_transfer.moment.forecast_quality.score import boot_weights, item_metrics, percentile


def masked(group, ids):
    stub = "test16" if group == "test" else "external16"
    with np.load(HEADS / "large_0" / (stub + ".npz")) as f:
        shape = (len(f["ids"]), 2 if group == "test" else 4, 16)
        h, mu, sd, pub = f["h"], f["mu"], f["sd"], f["pub"]
        order = {str(item): idx for idx, item in enumerate(f["ids"])}
        sel = [order[str(item)] for item in ids]
    with np.load(MOMENT / "head_test/results/large_0_heads.npz") as f:
        raw = (h @ f["terminal_hidden_weight"] + f["terminal_hidden_bias"]) * sd[:, None, None] + mu[:, None, None]
    return dict(masked_terminal=raw.reshape(shape)[sel], masked_published=pub.reshape(shape)[sel])


def forecasts(group):
    ids, context, truth, domain = rows(group)
    with np.load(OUT / f"{group}_pred.npz") as f:
        np.testing.assert_array_equal(ids, f["ids"])
        pred = {key: f[key] for key in f.files if key not in ("ids", "domain")}
    pred.update(masked(group, ids))
    pred["last"] = np.broadcast_to(context[:, :, -1:], truth.shape)
    if group == "test":
        for basename, keys in {"classical": {"history_select": "history_select_16"},
                               "bolt": {"bolt": "raw_16"}, "timesfm": {"timesfm": "raw_16"}}.items():
            with np.load(QUALITY / f"{basename}.npz") as f:
                mapping = {str(k): i for i, k in enumerate(f["ids"])}
                idx = [mapping[str(k)] for k in ids]
                for tag, key in keys.items():
                    pred[tag] = f[key][idx]
        with np.load(MOMENT / "head_test/results/input_pred.npz") as f:
            mapping = {str(k): i for i, k in enumerate(f["ids"])}
            idx = [mapping[str(k)] for k in ids]
            gap = {arm:float(np.max(abs((pred[arm + "_residual"] - f[arm + "_16"][idx]) /
                                         (context.std(2) + 1e-5)[:, :, None]))) for arm in ("lag32", "lag464")}
        assert max(gap.values()) < 1e-4, gap
    else:
        with np.load(WEATHER / "panel.npz") as f:
            np.testing.assert_array_equal(ids, f["ids"])
            pred.update(climatology=f["clim"], climate_anomaly=f["anomaly"])
        gap = {}
    return pred, ids, context, truth, domain, gap


def run():
    summary = {}
    for group in ("test", "weather"):
        pred, ids, context, truth, domain, gap = forecasts(group)
        unit = domain if group == "test" else np.repeat("NOAA", len(ids))
        metric = "mase1" if group == "test" else "mae"
        accum, relative_to_last, entries = {}, {}, {}
        for d in np.unique(unit):
            keep = unit == d
            assert sum(keep) == (96 if group == "test" else 46)
            weights = boot_weights(d, sum(keep))
            joint = {k:weights @ item_metrics(values[keep], truth[keep], context[keep])[metric] for k, values in pred.items()}
            refs = [("hidden_direct", k) for k in joint if k != "hidden_direct"]
            refs.extend(("hidden_residual", k) for k in joint if k not in ("hidden_residual", "hidden_direct"))
            entries[d] = dict(n_items=int(sum(keep)), scores={k:percentile(v) for k,v in joint.items()},
                              vs_last={k:percentile(v / joint["last"]) for k,v in joint.items()},
                              paired_ratio={}, paired_difference={})
            for name, values in joint.items():
                relative_to_last.setdefault(name, []).append(values / joint["last"])
            for a, b in refs:
                label = a + "_vs_" + b
                ratio = joint[a] / joint[b]
                entries[d]["paired_ratio"][label] = percentile(ratio)
                entries[d]["paired_difference"][label] = percentile(joint[a] - joint[b])
                accum.setdefault(label, []).append(ratio)
        # Independent scalar implementation of the published unit of analysis.
        check = {}
        for name in ("hidden_direct", "hidden_residual"):
            err = 0.0
            for d in np.unique(unit):
                measurements = []
                for item in np.flatnonzero(unit == d):
                    by_origin = []
                    for origin in range(context.shape[1]):
                        denom = 1 if group == "weather" else np.mean(np.abs(np.diff(context[item, origin])))
                        by_origin.append(np.sum(np.abs(pred[name][item, origin] - truth[item, origin])) / (16 * denom))
                    measurements.append(sum(by_origin) / len(by_origin))
                err = max(err, abs(float(np.mean(measurements)) - entries[d]["scores"][name]["value"]))
            assert err < 1e-10
            check[name] = err
        summary[group] = dict(metric=metric, domains=entries, independent_item_loop_max_abs_gap=check,
                              prior_residual_lag_max_sd_gap=gap)
        if group == "test":
            summary[group]["equal_domain_mean_ratio_to_persistence"] = {
                k:percentile(np.mean(v, axis=0)) for k,v in relative_to_last.items()}
            summary[group]["equal_domain_mean_paired_ratio"] = {
                k:percentile(np.mean(v, axis=0)) for k,v in accum.items()}
    (OUT / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    run()
