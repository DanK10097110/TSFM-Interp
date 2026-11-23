"""Raw histories, signed patch statistics and training-only nomination for two candidate descriptors."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.stats import rankdata
import torch

from research.activation_transfer.moment.dictionary_transfer.match import dictionary, encode
from research.activation_transfer.moment.horizon_identity.train import HERE as DICTS
from research.activation_transfer.moment.token_paths.geometry import CACHE, histories

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent
AT = TRANSFER.parents[1]
OUT = HERE / "results"
SEED = 22112201
NAMES = ("level", "trend")


def measurements(raw):
    x = np.asarray(raw, float)
    q = x[:, -8:]
    t = np.arange(8, dtype=float) - 3.5
    level = q.mean(1) - x.mean(1)
    trend = 7 * (q @ t) / (t @ t)
    physical = np.column_stack((level, trend))
    return physical / (x.std(1)[:, None] + 1e-5), physical, q.mean(1)


def source_windows(ids, standardized):
    manifest = json.loads((AT / "shape_transfer/blind_frequency/results/manifest.json").read_text())["domains"]
    domains = np.array([s.split("/", 1)[0] for s in ids])
    ans = np.empty_like(standardized, dtype=float)
    for d in np.unique(domains):
        with pa.memory_map(manifest[d]["source"]) as source:
            table = pa.ipc.open_stream(source).read_all()
            key = "item_id" if "item_id" in table.column_names else "id"
            lookup = {str(k):i for i,k in enumerate(table[key].to_pylist())}
            for r in np.where(domains == d)[0]:
                item = str(ids[r]).split("/", 1)[1]
                seq = table["target"][lookup[item]].values
                ans[r] = seq.slice(len(seq)-488, 480).to_numpy(zero_copy_only=False)
    normalized = ((ans - ans.mean(1, keepdims=True)) / ans.std(1, keepdims=True)).astype("f4")
    np.testing.assert_array_equal(normalized, standardized)
    return ans


def materials(train):
    standardized, ids, domain = histories(train)
    with np.load(CACHE / "large" / ("train.npz" if train else "eval.npz")) as f:
        np.testing.assert_array_equal(ids, f["ids"])
        assert f["positions"][0] == 61
        h = f["middle"][:, 0]
    with np.load(DICTS / "results" / f"large_{SEED}_dictionary.npz") as f:
        atoms = f["ids"]
    codes = encode(dictionary("large", SEED), h)[:, atoms].astype(float)
    return ids, source_windows(ids, standardized), codes, atoms, domain


def auc(y, score):
    y = np.asarray(y, bool)
    n = int(y.sum())
    if n == 0 or n == len(y):
        return np.nan
    ranks = rankdata(score)
    return (ranks[y].sum() - n*(n+1)/2) / (n*(len(y)-n))


def fit_line(stat, y):
    cx = stat - stat.mean()
    slope = float(cx @ (y-y.mean()) / (cx @ cx))
    return slope, float(y.mean() - stat.mean()*slope)


def predicted(stat, fit):
    return np.maximum(0, fit["intercept"] + fit["slope"] * stat)


def firing(z):
    return dict(firing=int(np.count_nonzero(z>0)), n=len(z), prevalence=float(np.mean(z>0)))


def nominate(statistics, codes, atoms, fit_ix, val_ix):
    candidate, winners, used = {}, {}, set()
    for idx, kind in enumerate(NAMES):
        raw = statistics[:, idx]
        rec = []
        for j, atom in enumerate(atoms):
            y = codes[:, j]
            slope, intercept = fit_line(raw[fit_ix], y[fit_ix])
            orient = 1 if slope >= 0 else -1
            tr, va = y[fit_ix], y[val_ix]
            qualified = min(np.count_nonzero(tr), np.count_nonzero(tr==0),
                            np.count_nonzero(va), np.count_nonzero(va==0)) >= 8
            fit_auc, val_auc = auc(tr>0, raw[fit_ix]*orient), auc(va>0, raw[val_ix]*orient)
            rec.append(dict(id=int(atom), column=j, slope=slope, intercept=intercept, polarity=orient,
                            fit=dict(**firing(tr), auc=float(fit_auc) if np.isfinite(fit_auc) else None),
                            validation=dict(**firing(va), auc=float(val_auc) if np.isfinite(val_auc) else None),
                            eligible=bool(qualified)))
        rec.sort(key=lambda item:(-(item["validation"]["auc"] or 0),item["id"]))
        candidates = [r for r in rec if r["eligible"] and r["id"] not in used]
        if candidates:
            winners[kind] = candidates[0]
            used.add(candidates[0]["id"])
        candidate[kind] = rec
    return winners, candidate


def main():
    torch.set_num_threads(1)
    ids, raw, codes, atoms, _ = materials(True)
    assert len(ids) == 1024 and len(set(ids)) == len(ids) and len(atoms) == 16
    order = np.random.default_rng(22112291).permutation(len(ids))
    train, val = order[:768], order[768:]
    winners, ranked = nominate(measurements(raw)[0], codes, atoms, train, val)
    rec = dict(winners=winners, all_candidates=ranked, training_ids=ids[train].tolist(),
               validation_ids=ids[val].tolist(), source_dictionary_seed=SEED)
    (OUT / "nomination.json").write_text(json.dumps(rec, indent=2, allow_nan=False)+"\n")
    print(winners,flush=True)


if __name__ == "__main__":
    main()
