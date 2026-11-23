"""Disjoint-calibration refits of exact personalized and global receiver IDs."""
import json

import numpy as np
import torch

from research.activation_transfer.shape_transfer.feature_groups.data import OUT, dictionaries, load
from research.activation_transfer.shape_transfer.feature_groups.fit import gain_tables, greedy


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def describe(values):
    return dict(mean=float(np.mean(values)), split_p10_p90=np.quantile(values, [.1,.9]).tolist())


def split_fits(gains, members, histories, half):
    groups, firing = [], []
    normalized = []
    for g, cols, rows in zip(gains,members,histories):
        keep = half[rows]
        sub = g[keep]
        firing.append(int(keep.sum()))
        groups.append(cols[greedy(sub)].tolist())
        den = sub.max(1).sum() if len(sub) else 0.
        if den > 0:
            normalized.append((sub / den, cols))
    universe = np.unique(np.concatenate([x[1] for x in normalized]))
    dense = np.zeros((sum(len(x[0]) for x in normalized),len(universe)))
    start = 0
    for v, cols in normalized:
        dense[start:start+len(v),np.searchsorted(universe,cols)] = v
        start += len(v)
    global_set = universe[greedy(dense)].tolist()
    return groups,np.array(firing),global_set


def run(source):
    dic = dictionaries(source)
    data = load(source, dic, "cal")
    rows = [np.flatnonzero(data["codes"][:, j] > 0) for j in range(len(dic.atoms))]
    members, gains = [], []
    for gain in gain_tables(data, dic, False):
        cols = np.flatnonzero(gain.max(0) > 0)
        gains.append(gain[:,cols])
        members.append(cols)
    rng = np.random.default_rng(2026112719)
    records = []
    for _ in range(40):
        a = np.zeros(96, dtype=bool)
        a[rng.choice(96,48,replace=False)] = True
        left,lcount,g_left = split_fits(gains,members,rows,a)
        right,rcount,g_right = split_fits(gains,members,rows,~a)
        record = dict(half_a=np.flatnonzero(a).tolist(), global_lists=[g_left,g_right],
                      global_jaccard=jac(g_left,g_right), global_first=float(g_left[0]==g_right[0]), subsets={})
        for minimum in (2,4):
            ids = np.flatnonzero((lcount >= minimum) & (rcount >= minimum))
            other, own = [], []
            for j in ids:
                eligible = [k for k in ids if k!=j and len(right[k]) == len(right[j])]
                if eligible:
                    other.append(np.mean([jac(left[j],right[k]) for k in eligible]))
                    own.append(jac(left[j],right[j]))
            record["subsets"][str(minimum)] = dict(
                n=len(ids), atoms=dic.atoms[ids].tolist(), first=np.mean([left[j][0]==right[j][0] for j in ids]),
                same=np.mean([jac(left[j],right[j]) for j in ids]), other=np.mean(other),
                own_minus_other=np.mean(np.array(own)-np.array(other)))
        records.append(record)
    summary = {m:{k:describe([r["subsets"][m][k] for r in records]) for k in
                  ("n","first","same","other","own_minus_other")} for m in ("2","4")}
    summary.update(global_jaccard=describe([r["global_jaccard"] for r in records]),
                   global_first=describe([r["global_first"] for r in records]))
    return dict(summary=summary, splits=records)


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = {s:run(s) for s in ("bolt","timesfm")}
    (OUT/"fit_stability.json").write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n")
