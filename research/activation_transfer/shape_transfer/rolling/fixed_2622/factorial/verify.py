"""Recompute scalar choices independently and replay donor plus recipient actions at each offset."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import N, OUT as ORIGINAL
from research.activation_transfer.shape_transfer.rolling.fixed_2622.factorial.data import CACHE, OFFSETS, OUT
from research.activation_transfer.shape_transfer.rolling.fixed_2622.fit import DONOR, PRIMARY
from research.activation_transfer.shape_transfer.rolling.verify import replay, weighted
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes


def verify():
    with np.load(OUT / "effects.npz") as f:
        effects = dict(f)
    with np.load(CACHE / "data.npz") as f:
        raw = dict(f)
    dic = dictionaries("timesfm")
    query = list(dic.atoms).index(DONOR)
    spec = json.loads((ORIGINAL / "frozen.json").read_text())
    st, sp, sids = read_materials(CACHE, "timesfm")
    tt, tp, tids = read_materials(CACHE, "bolt")
    np.testing.assert_array_equal(sids, tids)
    src = adapter_for("timesfm")
    tgt = adapter_for("bolt")
    sselect, tselect = last_window_slice(src, sp), last_window_slice(tgt, tp)
    support, strengths = active_codes(dic.sae, tt)
    with np.load(CACHE / "timesfm_geometry.npz") as f:
        sj = f["jac"].astype(float)
    with np.load(CACHE / "bolt_geometry.npz") as f:
        tj = f["jac"].astype(float)
    cap = .2 * dic.radius
    errors = []
    cases = []
    rng = np.random.default_rng(20261121430)
    for cell, offset in enumerate(OFFSETS):
        for sensor in rng.choice(N, 3, replace=False):
            i = cell * N + sensor
            old = 2 * cell * N + sensor
            now = old + N
            assert tids[now] == effects["ids"][i]
            history = raw["x"][now:now+1]
            source_change = -effects["source_now"][i] * dic.donor[query]
            goal = weighted(sj[now] @ source_change, dic.weight)
            actual = replay(src, MODELS["timesfm"][-1], sselect, history, st[now:now+1], source_change)
            errors.append(float(np.max(np.abs(actual - effects["native"][i]))))
            budget = strengths[old, PRIMARY] if strengths[old, PRIMARY] > 0 else spec["fallback_activation"][0]
            budget = min(budget, cap / np.linalg.norm(dic.decoder[PRIMARY]))
            sham = 1355
            old_goal = weighted(-effects["source_anchor"][i] * (sj[old] @ dic.donor[query]), dic.weight)
            profile = weighted(tj[old] @ dic.decoder[sham], dic.weight)
            sham_coef = (1 if old_goal @ profile >= 0 else -1) * budget * np.linalg.norm(dic.decoder[PRIMARY]) / np.linalg.norm(dic.decoder[sham])
            choices = []
            for atom in support[now]:
                profile = weighted(tj[now] @ dic.decoder[atom], dic.weight)
                dose = float(np.clip((profile @ goal) / max(profile @ profile, 1e-24), -strengths[now, atom], strengths[now, atom]))
                choices.append((float(np.square(dose * profile - goal).sum()), atom, dose))
            loss, selected, dose = min(choices)
            if loss < np.square(goal).sum():
                dose *= min(1., cap / max(abs(dose) * np.linalg.norm(dic.decoder[selected]), 1e-20))
            else:
                selected, dose = -1, 0.
            for arm, atom, amount in (("fixed", PRIMARY, budget), ("fresh", selected, dose), ("sham", sham, sham_coef)):
                assert effects[arm + "_id"][i] == atom
                np.testing.assert_allclose(effects[arm + "_coef"][i], amount, rtol=2e-6, atol=1e-7)
                patched = replay(tgt, MODELS["bolt"][-1], tselect, history, tt[now:now+1], amount * dic.decoder[atom])
                errors.append(float(np.max(np.abs(patched - effects[arm + "_effect"][i]))))
            cases.append(str(tids[now]))
    if max(errors) > 8e-5:
        raise ValueError(f"Maximum real replay discrepancy {max(errors)}")
    result = dict(sensors_and_origins=cases, actual_interventions=len(errors), max_abs_error=max(errors))
    result["geometry_checks"] = {
        model: json.loads((CACHE / (model + "_checks.json")).read_text()) for model in ("bolt", "timesfm")
    }
    (OUT / "replay.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    verify()
