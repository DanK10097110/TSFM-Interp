"""Compare the fixed positive Bolt write, a norm-matched sham and current best active feature."""
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.feature_groups.data import dictionaries
from research.activation_transfer.shape_transfer.rolling.fixed_2622.data import N, OUT as ORIGINAL
from research.activation_transfer.shape_transfer.rolling.fixed_2622.factorial.data import CACHE, OFFSETS, OUT
from research.activation_transfer.shape_transfer.rolling.fixed_2622.fit import DONOR, PRIMARY
from research.activation_transfer.shape_transfer.rolling.intervene import optimal
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads

ARMS = ("fixed", "fresh", "sham")


def run():
    spec = json.loads((ORIGINAL / "frozen.json").read_text())
    sham = spec["candidates"][1]
    if spec["candidates"][:2] != [PRIMARY, 1355]:
        raise ValueError("Preregistered target IDs differ from frozen calibration")
    dic = dictionaries("timesfm")
    query = list(dic.atoms).index(DONOR)
    source = dict(np.load(CACHE / "timesfm_geometry.npz"))
    target = dict(np.load(CACHE / "bolt_geometry.npz"))
    native = dict(np.load(CACHE / "timesfm_native.npz"))
    raw = dict(np.load(CACHE / "data.npz"))
    tokens, positions, ids = read_materials(CACHE, "bolt")
    for material in (raw, native, source, target):
        np.testing.assert_array_equal(material["ids"], ids)
    np.testing.assert_array_equal(native["atoms"], dic.atoms)
    decoder_norms = np.linalg.norm(dic.decoder, axis=1)
    cap = .2 * dic.radius
    support, strengths = active_codes(dic.sae, tokens)
    anchor = np.concatenate([np.arange(i * 2 * N, i * 2 * N + N) for i in range(len(OFFSETS))])
    current = anchor + N
    source_code = native["codes"][:, query]
    donor_profile = weight_leads(
        -source_code[:, None] * (source["jac"].astype(float) @ dic.donor[query]), dic.weight,
    )
    anchor_bound = np.minimum(
        np.where(strengths[anchor, PRIMARY] > 0, strengths[anchor, PRIMARY], spec["fallback_activation"][0]),
        cap / decoder_norms[PRIMARY],
    )
    writes = dict(
        ids=ids[current], offsets=raw["split"][current].astype(int), anchor_ids=ids[anchor],
        source_anchor=source_code[anchor], source_now=source_code[current], donor=DONOR,
        receiver_anchor=strengths[anchor, PRIMARY], receiver_now=strengths[current, PRIMARY],
        native=native["native"][current, query], anchor_bound=anchor_bound, radius=dic.radius,
    )
    writes["fixed_coef"] = anchor_bound
    writes["fixed_id"] = np.full(len(anchor), PRIMARY, dtype=int)
    sham_profile = weight_leads(target["jac"][anchor].astype(float) @ dic.decoder[sham], dic.weight)
    preferred = np.einsum("nh,nh->n", donor_profile[anchor], sham_profile)
    writes["sham_coef"] = np.where(preferred >= 0., 1., -1.) * anchor_bound * decoder_norms[PRIMARY] / decoder_norms[sham]
    writes["sham_id"] = np.full(len(anchor), sham, dtype=int)
    writes["fresh_coef"] = np.zeros(len(anchor))
    writes["fresh_id"] = np.full(len(anchor), -1, dtype=int)
    for r, row in enumerate(current):
        choices = support[row]
        profiles = weight_leads(dic.decoder[choices] @ target["jac"][row].astype(float).T, dic.weight)
        atom, amount = optimal(choices, profiles, strengths[row, choices], donor_profile[row], None, decoder_norms, cap)
        writes["fresh_id"][r] = atom
        writes["fresh_coef"][r] = amount

    model = adapter_for("bolt")
    select = last_window_slice(model, positions)
    for arm in ARMS:
        amounts = writes[arm + "_coef"]
        chosen = writes[arm + "_id"]
        live = np.flatnonzero(amounts)
        effects = np.zeros((len(current), 8), dtype="f4")
        for start in range(0, len(live), 32):
            rows = live[start:start + 32]
            now = current[rows]
            deltas = (amounts[rows, None] * dic.decoder[chosen[rows]]).astype("f4")
            effects[rows] = change_forecasts(
                model, MODELS["bolt"][-1], raw["x"][now], tokens[now], deltas, select, 32,
            ) - target["forecast"][now]
        writes[arm + "_effect"] = effects
    np.testing.assert_allclose(
        np.abs(writes["sham_coef"]) * decoder_norms[sham],
        writes["fixed_coef"] * decoder_norms[PRIMARY],
    )
    np.savez_compressed(OUT / "effects.npz", **writes)


if __name__ == "__main__":
    torch.set_num_threads(2)
    run()
