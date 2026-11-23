"""Evaluate both coverage and physical response, resampling whole histories jointly."""
import json

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.feature_groups.data import OUT
from research.activation_transfer.shape_transfer.sparse_stability.score import interval, rate

NAMES = ("learned", "fixed", "level", "donor_shuffle", "matched0", "matched1", "matched2", "matched3", "local")
SHAMS = NAMES[4:8]


def weights(n):
    return np.vstack([np.ones(n), resampling_counts(n)])


def clean_metric(x):
    return [float(v) if np.isfinite(v) else None for v in x]


def recover(active, ref, pred, draws, pooled=False):
    signal = np.square(ref).sum(-1) * active
    error = np.square(ref - pred).sum(-1) * active
    if pooled:
        den, loss = draws @ signal.sum(1), draws @ error.sum(1)
        return np.divide(den-loss, den, out=np.full_like(den, np.nan), where=den > 1e-20)
    count = draws @ active.astype(float)
    den, loss = draws @ signal, draws @ error
    ok = (count > 0) & (den > 8e-10 * count)
    ratios = np.divide(den-loss, den, out=np.full_like(den, np.nan), where=ok)
    keep = ok[0]
    if not keep.any():
        return np.full(len(draws), np.nan)
    n = np.isfinite(ratios[:, keep]).sum(1)
    return np.divide(np.nansum(ratios[:, keep], 1), n, out=np.full(len(n), np.nan), where=n > 0)


def transform(x, part):
    x = np.array(x, dtype=float)
    if part == "shape":
        return x - x.mean(-1, keepdims=True)
    if part == "level":
        return np.repeat(x.mean(-1, keepdims=True), 8, axis=-1)
    return x


def group_measures(data, active, features, draws):
    usable = active & features[None, :]
    result = {}
    for key in NAMES:
        taken = data[key + "_id"] >= 0
        same = taken & (data[key + "_id"] == data["local_id"])
        result[key] = {"available":rate(usable & taken, usable, draws),
                       "captures_local":rate(usable & same, usable, draws)}
    return result


def scores(data, mask, cols, draws, names, add_averaged=False, per_atom=False):
    rec = {"atoms":data["atoms"][cols].tolist(), "cases":int(mask[:, cols].sum()), "n_atoms":int(cols.sum())}
    if not cols.any():
        return rec
    active = mask[:, cols]
    for part in ("full","shape","level"):
        ref = transform(data["reference"][:, cols], part)
        arrays = {}
        for key in names:
            estimate = transform(data[key + "_effect"][:, cols], part)
            arrays[key] = recover(active, ref, estimate, draws)
        if add_averaged:
            arrays["matched_mean"] = np.mean([arrays[key] for key in SHAMS], axis=0)
        out = {name:interval(series) for name, series in arrays.items()}
        out["learned_minus"] = {k:interval(arrays["learned"]-v) for k,v in arrays.items() if k!="learned"}
        rec[part] = out
        if per_atom and part in ("full", "shape"):
            points = {}
            for k in ("learned","level","donor_shuffle","local"):
                pred = transform(data[k + "_effect"][:, cols], part)
                points[k] = clean_metric([recover(active[:,j:j+1], ref[:,j:j+1], pred[:,j:j+1],
                                                 draws[:1])[0] for j in range(cols.sum())])
            rec[part+"_per_atom"] = points
    return rec


def slice_rows(arch, mask):
    return {k:(v if k in ("atoms","unique_physical_calls") else v[mask]) for k,v in arch.items()}


def fresh(source):
    groups = json.loads((OUT / (source + "_lists.json")).read_text())["groups"]
    same_length = np.array([len(a) == len(b) for a,b in zip(groups["learned"], groups["level"])])
    with np.load(OUT / (source + "_fresh.npz")) as f:
        raw = dict(f)
    result = {}
    for domain in np.unique(raw["split"]):
        z = slice_rows(raw, raw["split"] == domain)
        active = z["codes"] > 0
        cols = active.sum(0) >= 8
        draw = weights(len(active))
        entry = dict(firing=active.sum(0).tolist(), n_series=len(active),
                     primary=scores(z, active, cols, draw, NAMES, True, True),
                     coverage=group_measures(z, active, cols, draw))
        entry["same_level_list_length"] = scores(z, active, cols & same_length, draw, ("learned","level"))
        entry["opportunity_matched"] = {}
        for key in ("level", "donor_shuffle", *SHAMS):
            mask = active & (z["learned_id"] >= 0) & (z[key + "_id"] >= 0)
            mutual = mask.sum(0) >= 8
            entry["opportunity_matched"][key] = scores(z, mask, mutual, draw, ("learned",key))
        pooled = {}
        for part in ("full","shape"):
            ref = transform(z["reference"], part)
            arrays = {k:recover(active, ref, transform(z[k+"_effect"], part), draw, True) for k in NAMES}
            arrays["matched_mean"] = np.mean([arrays[k] for k in SHAMS], axis=0)
            pooled[part] = {k:interval(v) for k,v in arrays.items()}
        entry["pooled_all_firings"] = pooled
        result[str(domain)] = entry
    return result


def stability(source):
    groups = json.loads((OUT / (source + "_lists.json")).read_text())["groups"]
    with np.load(OUT / (source + "_stability_clean.npz")) as c, np.load(OUT / (source + "_noise.npz")) as p:
        clean, noise = dict(c), dict(p)
    n = len(clean["ids"])
    draws = weights(n)
    result = {}
    for dose in sorted(set(noise["split"])):
        z = slice_rows(noise, noise["split"] == dose)
        assert list(z["ids"]) == [str(dose) + "/" + i for i in clean["ids"]]
        active = (clean["codes"] > 0) & (z["codes"] > 0)
        c, p = clean["local_id"], z["local_id"]
        changed = active & (c != p)
        r = {"switch_rate":rate(changed, active, draws), "groups":{}}
        containment = {}
        for name, lists in groups.items():
            pre = np.column_stack([np.isin(c[:,j], g) for j,g in enumerate(lists)])
            post = np.column_stack([np.isin(p[:,j], g) for j,g in enumerate(lists)])
            containment[name] = (changed & pre & post).astype(float)
            a, b = clean[name+"_id"], z[name+"_id"]
            both = active & (a>=0) & (b>=0)
            r["groups"][name] = {"both_winners_in_list":rate(changed & pre & post, changed, draws),
                                 "kept_in_list_given_original_inside":rate(changed & pre & post, changed & pre, draws),
                                 "one_or_both_inside":rate(changed & (pre|post), changed, draws),
                                 "list_available_on_both":rate(both, active, draws),
                                 "restricted_winner_switched":rate(both & (a != b), both, draws)}
        containment["matched_mean"] = np.mean([containment[k] for k in SHAMS], axis=0)
        denom = draws @ changed.sum(1)
        r["learned_minus_switch_containment"] = {}
        for other in ("level", "donor_shuffle", "matched_mean"):
            delta = draws @ (containment["learned"] - containment[other]).sum(1)
            r["learned_minus_switch_containment"][other] = interval(np.divide(delta, denom, out=np.full_like(delta, np.nan), where=denom > 0))
        if dose == "0.02":
            cols = active.sum(0) >= 8
            r["clean_actual"] = scores(clean, active, cols, draws, ("learned","local"))
            r["perturbed_actual"] = scores(z, active, cols, draws, ("learned","local"))
        result[dose] = r
    return result


if __name__ == "__main__":
    out = {s:dict(fresh=fresh(s), stability=stability(s)) for s in ("bolt","timesfm")}
    (OUT / "summary.json").write_text(json.dumps(out,allow_nan=False,separators=(",",":")) + "\n")
