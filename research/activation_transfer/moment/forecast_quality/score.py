"""Compare forecasts in native units and on a per-series scale, with paired whole-item uncertainty."""
import argparse
import json
from hashlib import sha256

import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT, hash_array

MODELS = ("moment_small", "moment_base", "moment_large", "bolt", "timesfm")


def boot_weights(domain, n):
    key = int(sha256(("moment-quality-bootstrap|" + domain).encode()).hexdigest()[:16], 16)
    sampled = np.random.default_rng(key).multinomial(n, np.full(n, 1/n), size=4000)
    return np.vstack([np.ones(n), sampled]) / n


def percentile(values):
    return dict(value=float(values[0]), ci95=np.quantile(values[1:], [.025, .975]).tolist())


def item_metrics(prediction, target, contexts):
    """One value per independent item. First average each origin's scaled error across its own leads."""
    residual = prediction - target
    absolute = np.abs(residual).mean(2)
    scale = np.abs(np.diff(contexts, axis=2)).mean(2)
    sd = np.std(contexts, axis=2)
    return dict(mase1=(absolute / scale).mean(1),
                nmae_sd=(absolute / sd).mean(1),
                mae=absolute.mean(1), mse=np.square(residual).mean((1, 2)), bias=residual.mean((1, 2)))


def comparisons(methods):
    pairs = []
    for name in MODELS:
        if name not in methods:
            continue
        pairs.extend((name, base) for base in ("last", "history_select", "season1", "bolt", "timesfm")
                     if base in methods and base != name)
        pairs.append((name + "_z", name))
    for model, reference in (("moment_large", "moment_small"), ("moment_large", "moment_base"),
                             ("moment_base", "moment_small")):
        if model in methods and reference in methods:
            pairs.append((model, reference))
    return pairs


def run(partial=False):
    with np.load(OUT / "panel.npz") as source:
        x = source["context"]
        y = source["future"]
        items = source["id"]
        domains = source["domain"]
        splits = source["split"]
    registry = json.loads((OUT / "manifest.json").read_text())
    np.testing.assert_array_equal(items, registry["ids"])
    if hash_array(x) != registry["input_sha256"] or hash_array(y) != registry["target_sha256"]:
        raise ValueError("Historical evaluation arrays no longer match the frozen registry")
    models = {}
    for name in MODELS:
        path = OUT / (name + ".npz")
        if partial and not path.exists():
            continue
        with np.load(path) as file:
            np.testing.assert_array_equal(file["ids"], items)
            assert str(file["context_sha256"]) == hash_array(x)
            models[name] = dict(file)
    with np.load(OUT / "classical.npz") as file:
        np.testing.assert_array_equal(file["ids"], items)
        assert str(file["context_sha256"]) == hash_array(x)
        classic = dict(file)
    results = {}
    for split in ("holdout", "diagnostic"):
        split_scores = {}
        consistency = {}
        for model, archive in models.items():
            gaps = np.abs(archive["raw_8"] - archive["raw_16"][:, :, :8]) / x.std(2, keepdims=True)
            consistency[model] = {}
            for domain in np.unique(domains):
                rows = (domains == domain) & (splits == split)
                values = gaps[rows]
                consistency[model][domain] = dict(
                    mean=float(values.mean()), median=float(np.median(values)),
                    p95=float(np.quantile(values, .95)), max=float(values.max()),
                )
        for endpoint, h, sl in (("h8",8,slice(8)),("h16",16,slice(16)),
                                ("h16_first8",16,slice(8)),("h16_last8",16,slice(8,16))):
            absolute = {}
            relative = {}
            across = {}
            for domain in np.unique(domains):
                use = (domains == domain) & (splits == split)
                weight = boot_weights(domain + "/" + split, int(use.sum()))
                target = y[use, :, sl]
                wave = {}
                for key, values in classic.items():
                    if key.endswith("_" + str(h)) and not key.startswith("selected"):
                        v = values[use, :, sl]
                        if np.isfinite(v).all():
                            wave[key.rsplit("_", 1)[0]] = v
                for model, archive in models.items():
                    wave[model] = archive[f"raw_{h}"][use, :, sl]
                    wave[model + "_z"] = archive[f"standardized_{h}"][use, :, sl]
                per_item = {key: item_metrics(v, target, x[use]) for key, v in wave.items()}
                per_domain = {}
                for key, score in per_item.items():
                    per_domain[key] = {metric: percentile(weight @ score[metric])
                                       for metric in ("mae","mase1","nmae_sd","bias")}
                    per_domain[key]["rmse"] = percentile(np.sqrt(weight @ score["mse"]))
                    per_domain[key]["median_item_mae"] = float(np.median(score["mae"]))
                    per_domain[key]["median_item_mase1"] = float(np.median(score["mase1"]))
                errors = {}
                for model, baseline in comparisons(wave):
                    mine = per_item[model]["mase1"]
                    ref = per_item[baseline]["mase1"]
                    ratio = (weight @ mine) / (weight @ ref)
                    tied = np.isclose(mine, ref, rtol=1e-12, atol=1e-14)
                    win = np.where(tied, .5, mine < ref).astype(float)
                    name = model + "_vs_" + baseline
                    errors[name] = dict(ratio=percentile(ratio), win_frequency=percentile(weight @ win),
                                        paired_mase1_difference=percentile(weight @ (mine-ref)))
                    across.setdefault(name, []).append(ratio)
                absolute[domain] = per_domain
                relative[domain] = errors
            macro = {key: percentile(np.mean(ratios, axis=0)) for key, ratios in across.items()}
            split_scores[endpoint] = dict(absolute=absolute, comparisons=relative, equal_domain_ratios=macro)
        choices = {}
        for h in (8,16):
            choices[h] = {}
            for domain in np.unique(domains):
                k, count = np.unique(classic[f"selected_{h}"][(domains == domain) & (splits == split)], return_counts=True)
                choices[h][domain] = {name:int(n) for name,n in zip(k,count)}
        results[split] = dict(scores=split_scores, shared_horizon_difference_sd=consistency,
                              classical_winner_by_origin=choices)
    dst = OUT / ("partial_summary.json" if partial else "summary.json")
    dst.write_text(json.dumps(results, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    run(parser.parse_args().partial)
