"""Select within frozen ID lists and measure physical single-atom write responses."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.shape_transfer.feature_groups.data import OUT, dictionaries, load
from research.activation_transfer.shape_transfer.sparse_stability.prepare import OUT as STABILITY
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads


def choose(gain, ids, allowed):
    """Use the best positive predicted gain among the currently allowed IDs."""
    permitted = np.arange(len(ids)) if allowed is None else np.flatnonzero(np.isin(ids, allowed))
    if len(permitted):
        best = permitted[gain[permitted].argmax()]
        if gain[best] > 0.:
            return int(best)
    return -1


def evaluate(source, cohort, global_only=False, unit_weight=False):
    dic = dictionaries(source)
    label = "_global" if global_only else ""
    output_label = "_unit" if unit_weight else label
    weight = 1. if unit_weight else dic.weight
    if unit_weight and (source != "bolt" or cohort != "fresh" or global_only):
        raise ValueError("Unit-weight diagnostic was registered for Bolt-source fresh cases only")
    fit = json.loads((OUT / (source + label + "_lists.json")).read_text())
    data = load(source, dic, cohort)
    ids, codes = data["ids"], data["codes"]
    assert set(fit["calibration_ids"]).isdisjoint(ids)
    np.testing.assert_array_equal(fit["atoms"], dic.atoms)
    groups = dict(fit["groups"])
    if not global_only:
        groups["local"] = None
    if global_only and cohort != "fresh":
        groups = {"global_functional":groups["global_functional"]}
    if unit_weight:
        groups = {"learned":groups["learned"], "local":None}
    out = dict(ids=ids, split=data["labels"], atoms=dic.atoms, codes=codes, reference=data["ref"])
    for key in groups:
        out[key + "_id"] = np.full(codes.shape, -1, dtype=int)
        out[key + "_coef"] = np.zeros(codes.shape, dtype="f4")
    # All methods use the same local objective and dose for a given (history, donor, receiver ID).
    write_keys, index = [], {}
    payload, rows = [], []
    write_arms = list(groups) if cohort == "fresh" or global_only else ["learned", "local"]
    for i, active in enumerate(data["support"]):
        profiles = weight_leads(dic.decoder[active] @ data["tj"][i].T, weight)
        bounds = data["strengths"][i, active]
        energy = np.square(profiles).sum(1)
        for j in np.flatnonzero(codes[i] > 0):
            goal = weight_leads(-codes[i, j] * (dic.donor[j] @ data["sj"][i].T), weight)
            dot = profiles @ goal
            coef = np.clip(dot / np.maximum(energy, 1e-24), -bounds, bounds)
            gain = 2 * coef * dot - coef**2 * energy
            for name, lists in groups.items():
                winner = choose(gain, active, None if lists is None else lists[j])
                if winner < 0:
                    continue
                atom = int(active[winner])
                movement = coef[winner] * dic.decoder[atom]
                cap = min(1., .2 * dic.radius / max(np.linalg.norm(movement), 1e-20))
                out[name + "_id"][i, j] = atom
                out[name + "_coef"][i, j] = cap * coef[winner]
                execute = name in write_arms and (cohort != "noise" or data["labels"][i] == "0.02")
                pair = (i, j, atom)
                if execute and pair not in index:
                    index[pair] = len(payload)
                    rows.append(i)
                    payload.append((cap * movement).astype("f4"))
                    write_keys.append(pair)
    model = adapter_for(dic.target)
    select = last_window_slice(model, data["positions"])
    layer = MODELS[dic.target][-1]
    actual = []
    for start in range(0, len(rows), 32):
        ix = rows[start:start+32]
        y = change_forecasts(model, layer, data["x"][ix], data["tokens"][ix],
                             np.array(payload[start:start+32]), select, 32)
        actual.extend(y - data["ordinary"][ix])
    for name in write_arms:
        result = np.zeros((*codes.shape, 8), dtype="f4")
        selected = out[name + "_id"]
        if cohort == "noise":
            result[data["labels"] == "0.001"] = np.nan
        for i, j in np.argwhere(selected >= 0):
            if cohort == "noise" and data["labels"][i] != "0.02":
                continue
            result[i, j] = actual[index[i, j, selected[i, j]]]
        out[name + "_effect"] = result
    if cohort in ("stability_clean", "noise") and not global_only:
        with np.load(STABILITY / (source + ".npz")) as prior:
            clean = cohort == "stability_clean"
            # Previous noise study scored only cases with a persisting donor.
            prev = prior["clean_winner" if clean else "winner"]
            prev_y = prior["clean_effect" if clean else "actual_reselect"]
            ok = prev >= 0
            np.testing.assert_array_equal(out["local_id"][ok], prev[ok])
            if not clean:
                ok &= (data["labels"] == "0.02")[:, None]
            np.testing.assert_allclose(out["local_effect"][ok], prev_y[ok], atol=8e-5, rtol=5e-4)
    out["unique_physical_calls"] = np.array(len(write_keys))
    np.savez_compressed(OUT / (source + output_label + "_" + cohort + ".npz"), **out)
    print(source, cohort, 'physical forecast cases', len(write_keys), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["bolt", "timesfm"])
    parser.add_argument("cohort", choices=["fresh", "stability_clean", "noise"])
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--unit-weight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.source, args.cohort, args.global_only, args.unit_weight)
