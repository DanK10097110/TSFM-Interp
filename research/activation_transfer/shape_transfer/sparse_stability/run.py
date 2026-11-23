"""Replay the original positive TopK support and signed write in perturbed histories."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as OLD_GEOMETRY
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL
from research.activation_transfer.shape_transfer.sparse_stability.prepare import DOSES, N, OUT, SCRATCH
from research.activation_transfer.shape_transfer.sparse_writes.run import HERE as SPARSE, pursuit, weight_leads
from tsfm_lens.sae.train import load_sae_checkpoint


def indices(ids, keys):
    by_id = {key: row for row, key in enumerate(ids)}
    return np.array([by_id[key] for key in keys])


def active_codes(sae, tokens):
    with torch.inference_mode():
        z = sae.encode(torch.from_numpy(tokens.mean(1))).numpy()
    rows = [np.flatnonzero(values > 0) for values in z]
    if not all(len(r) == 32 for r in rows):
        raise ValueError("Expected 32 positive target atoms per history")
    return np.stack(rows), z


def one_edit(valid, profiles, bounds, goal, decoder, cap, carry=None):
    """Choose one permitted atom or refit an explicitly carried identity."""
    choices = np.arange(len(valid)) if carry is None else np.flatnonzero(valid == carry)
    if len(choices):
        idx, coefs = pursuit(profiles[choices], bounds[choices], goal, 1)
        if len(idx):
            atom = valid[choices[idx[0]]]
            coef = coefs[0]
            coef *= min(1., cap / max(abs(coef) * np.linalg.norm(decoder[atom]), 1e-20))
            return atom, coef
    return -1, 0.


def choose(support, z, jac, source_jac, decoder, donors, codes, mask, weight, radius, carried=None):
    picked = np.full(mask.shape, -1, dtype=int)
    coefficients = np.zeros(mask.shape, dtype=float)
    for i, valid in enumerate(support):
        profiles = weight_leads(decoder[valid] @ jac[i].T, weight)
        for j in np.flatnonzero(mask[i]):
            goal = weight_leads(-codes[i, j] * (donors[j] @ source_jac[i].T), weight)
            old = None if carried is None else carried[i, j]
            picked[i, j], coefficients[i, j] = one_edit(
                valid, profiles, z[i, valid], goal, decoder, .2 * radius, old)
    return picked, coefficients


def physical_response(model, layer, selector, x, tokens, ordinary, chosen, coefficients, decoder):
    rows, cols = np.nonzero(chosen >= 0)
    responses = np.zeros((*chosen.shape, 8), dtype="f4")
    for start in range(0, len(rows), 32):
        ix, j = rows[start:start + 32], cols[start:start + 32]
        moves = (coefficients[ix, j, None] * decoder[chosen[ix, j]]).astype("f4")
        y = change_forecasts(model, layer, x[ix], tokens[ix], moves, selector, 32)
        responses[ix, j] = y - ordinary[ix]
    return responses


def run(source):
    target = next(k for k in MODELS if k != source)
    ids = np.array(json.loads((OUT / "manifest.json").read_text())["ids"])
    t_sae = load_sae_checkpoint(str(dictionary_path(target)))
    s_sae = load_sae_checkpoint(str(dictionary_path(source)))
    decoder = t_sae.W_dec.detach().numpy().astype(float)
    with np.load(REPLICATION / "results" / (source + "_seed0.npz")) as old:
        sel = indices(old["ids"], ids)
        source_codes, atoms = old["source_codes"][sel], old["atoms"]
        clean_native = old["native"][sel]
    donors = s_sae.W_dec.detach().numpy()[atoms].astype(float)
    h, _, row_ids = read_materials(CACHE, target)
    clean_support, clean_z = active_codes(t_sae, h[indices(row_ids, ids)])
    with np.load(FULL / "results" / (source + ".npz")) as old:
        weight = float(old["shape_weight"])
    with np.load(OLD_GEOMETRY / (target + "_jac.npz")) as t, np.load(OLD_GEOMETRY / (source + "_jac.npz")) as s:
        tj, sj = [v["jac"][indices(v["ids"], ids)].astype(float) for v in (t, s)]
        radius = float(t["radius"])
    clean_winner, clean_amount = choose(clean_support, clean_z, tj, sj, decoder, donors, source_codes,
                                        source_codes > 0, weight, radius)
    with np.load(SPARSE / "results" / (source + ".npz")) as original:
        sel = indices(original["ids"], ids)
        np.testing.assert_array_equal(clean_winner, original["one_signed_ids"][sel, :, 0])
        np.testing.assert_allclose(clean_amount, original["one_signed_coefficients"][sel, :, 0],
                                   rtol=3e-5, atol=1e-5)
        clean_effect = original["one_signed"][sel]
    tokens, positions, combined = read_materials(SCRATCH, target)
    with np.load(SCRATCH / "data.npz") as data:
        x = data["x"]
        np.testing.assert_array_equal(combined, data["ids"])
        np.testing.assert_array_equal(np.repeat(DOSES, N), data["split"])
    with np.load(SCRATCH / (source + "_native.npz")) as s:
        np.testing.assert_array_equal(s["ids"], combined)
        np.testing.assert_array_equal(s["atoms"], atoms)
        codes, native = s["codes"], s["native"]
    with np.load(SCRATCH / (target + "_geometry.npz")) as t, np.load(SCRATCH / (source + "_geometry.npz")) as s:
        for archive in (t, s):
            np.testing.assert_array_equal(archive["ids"], combined)
        jac, source_jac = t["jac"].astype(float), s["jac"].astype(float)
        base = t["forecast"]
    support, z = active_codes(t_sae, tokens)
    mask = np.tile(source_codes > 0, (2, 1)) & (codes > 0)
    orig = np.tile(clean_winner, (2, 1))
    chosen, amount = choose(support, z, jac, source_jac, decoder, donors, codes, mask, weight, radius)
    carried, refit = choose(support, z, jac, source_jac, decoder, donors, codes, mask, weight, radius, orig)
    same = mask & (orig == chosen)
    np.testing.assert_array_equal(chosen[same], carried[same])
    np.testing.assert_allclose(amount[same], refit[same], rtol=1e-9, atol=1e-10)
    # On every unchanged identity, both prescribed writes are identical. Replay only changed nonzero carries.
    model = adapter_for(target)
    layer = MODELS[target][-1]
    selector = last_window_slice(model, positions)
    response = physical_response(model, layer, selector, x, tokens, base, chosen, amount, decoder)
    changed = np.where(mask & ~same, carried, -1)
    alt = physical_response(model, layer, selector, x, tokens, base, changed, refit, decoder)
    alt[same] = response[same]
    np.testing.assert_array_equal(alt[mask & (carried == -1)], 0)
    positive = (-codes[:, :, None] * (donors @ source_jac.transpose(0, 2, 1))).astype("f4")
    np.savez_compressed(OUT / (source + ".npz"), ids=ids, atoms=atoms, doses=DOSES, clean_support=clean_support,
                        clean_winner=clean_winner, clean_coefficients=clean_amount, clean_codes=source_codes,
                        clean_native=clean_native, clean_effect=clean_effect, perturbed_support=support,
                        perturbed_codes=codes, winner=chosen, carry=carried, coefficients=amount,
                        carry_coefficients=refit, actual_native=native, actual_reselect=response,
                        actual_carry=alt, source_linear=positive)
    print(source, "original signed identity replay and both perturbed executions complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=MODELS)
    torch.set_num_threads(2)
    run(parser.parse_args().source)
