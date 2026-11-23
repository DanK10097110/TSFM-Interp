"""Learn compact sets on calibration-only signed response profiles."""
import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from research.activation_transfer.shape_transfer.feature_groups.data import OUT, dictionaries, load
from research.activation_transfer.shape_transfer.sparse_writes.run import HERE as SPARSE, weight_leads

K = 4
SHAMS = 4


def greedy(gains):
    """Facility-location objective: maximize each example's best available gain."""
    best = np.zeros(len(gains))
    chosen = []
    for _ in range(K):
        increments = np.maximum(gains - best[:, None], 0.).sum(0)
        atom = int(increments.argmax())
        if increments[atom] <= 0:
            break
        chosen.append(atom)
        best = np.maximum(best, gains[:, atom])
    return np.array(chosen, dtype=int)


def gain_tables(data, dic, level_only):
    codes, support = data["codes"], data["support"]
    for donor in range(codes.shape[1]):
        rows = np.flatnonzero(codes[:, donor] > 0)
        gains = np.zeros((len(rows), len(dic.decoder)))
        for r, i in enumerate(rows):
            active = support[i]
            profiles = dic.decoder[active] @ data["tj"][i].T
            request = -codes[i, donor] * (dic.donor[donor] @ data["sj"][i].T)
            if level_only:
                profiles = profiles.mean(1, keepdims=True)
                request = np.array([request.mean()])
            else:
                profiles = weight_leads(profiles, dic.weight)
                request = weight_leads(request, dic.weight)
            energy, dot = np.square(profiles).sum(1), profiles @ request
            bounds = data["strengths"][i, active]
            coef = np.clip(dot / np.maximum(energy, 1e-24), -bounds, bounds)
            gains[r, active] = np.maximum(0., 2 * coef * dot - coef**2 * energy)
        yield gains


def greedy_sets(data, dic, level_only):
    return [greedy(gains).tolist() for gains in gain_tables(data, dic, level_only)]


def availability(support, cols, conditional):
    """Coverage and sorted member frequencies on donor firings and all histories."""
    if not len(cols):
        return np.array([0., 0.])
    view = support[:, cols]
    return np.r_[view[conditional].any(1).mean(), view.any(1).mean(),
                 np.sort(view[conditional].mean(0)), np.sort(view.mean(0))]


def shams_for_list(present, rows, members, rng):
    k = len(members)
    if not k:
        return [[] for _ in range(SHAMS)]
    observed = present[rows].any(0)
    observed[members] = False
    candidates = np.flatnonzero(observed)
    if len(candidates) < k:
        raise ValueError("Too few disjoint atoms available for matched controls")
    wanted = availability(present, members, rows)
    marginal = np.stack([present[rows].mean(0), present.mean(0)], axis=1)
    neighborhoods = []
    for atom in members:
        cost = np.square(marginal[candidates] - marginal[atom]).sum(1)
        # Include all ties at the 32nd nearest neighbor.
        neighbors = candidates[cost <= np.sort(cost)[min(31, len(cost)-1)]]
        neighborhoods.append(neighbors)
    shams = []
    for _ in range(SHAMS):
        n = 1600
        proposals = np.column_stack([rng.choice(neighbors, n) for neighbors in neighborhoods])
        unique = np.ones(n, dtype=bool)
        for i in range(k):
            for j in range(i):
                unique &= proposals[:, i] != proposals[:, j]
        proposals = proposals[unique]
        if not len(proposals):
            proposals = np.array([rng.choice(candidates, k, replace=False) for _ in range(n)])
        costs = []
        for batch in np.array_split(proposals, max(1, len(proposals)//200)):
            bits = present[:, batch]
            by_case = bits.any(2)
            stats = np.column_stack([by_case[rows].mean(0), by_case.mean(0), np.sort(bits[rows].mean(0)),
                                     np.sort(bits.mean(0))])
            costs.extend(np.square(stats - wanted).sum(1))
        shams.append(proposals[int(np.argmin(costs))].tolist())
    return shams


def reassign(groups, present, codes):
    owners = [-1] * len(groups)
    for k in sorted(set(map(len, groups))):
        js = [j for j, group in enumerate(groups) if len(group) == k]
        if len(js) < 2 or k == 0:
            continue
        cost = np.zeros((len(js), len(js)))
        for a, j in enumerate(js):
            firing = codes[:, j] > 0
            want = availability(present, groups[j], firing)
            for b, other in enumerate(js):
                delta = availability(present, groups[other], firing) - want
                cost[a, b] = float(delta @ delta) if a != b else 1e8
        rr, cc = linear_sum_assignment(cost)
        for a, b in zip(rr, cc):
            assert a != b
            owners[js[a]] = js[b]
    return owners


def fit(source):
    dic = dictionaries(source)
    data = load(source, dic, "cal")
    assert len(data["ids"]) == 96 and data["ref"] is None
    codes = data["codes"]
    learned, level = [greedy_sets(data, dic, flag) for flag in (False, True)]
    with np.load(SPARSE / "results" / (source + "_identity.npz")) as archived:
        np.testing.assert_array_equal([g[0] if g else -1 for g in learned], archived["fixed"])
    present = data["strengths"] > 0
    rng = np.random.default_rng(2026112707 if source == "bolt" else 2026112708)
    randoms = [shams_for_list(present, codes[:, j] > 0, g, rng) for j, g in enumerate(learned)]
    owners = reassign(learned, present, codes)
    groups = dict(learned=learned, fixed=[g[:1] for g in learned], level=level,
                  donor_shuffle=[learned[i] if i >= 0 else [] for i in owners])
    for r in range(SHAMS):
        groups[f"matched{r}"] = [g[r] for g in randoms]
    stats = {name:[availability(present, group, codes[:, j] > 0).tolist() for j, group in enumerate(lst)]
             for name, lst in groups.items()}
    result = dict(calibration_ids=data["ids"].tolist(), atoms=dic.atoms.tolist(), firing=(codes>0).sum(0).tolist(),
                  groups=groups, assigned_donor=owners, cal_availability=stats)
    (OUT / (source + "_lists.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(source, 'trained', [len(g) for g in learned], 'firing', result['firing'], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["bolt", "timesfm"])
    torch.set_num_threads(2)
    fit(parser.parse_args().source)
