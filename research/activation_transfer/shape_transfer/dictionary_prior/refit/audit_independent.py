"""Audit both SAE seeds, held-out masks, projection privileges, physical actions and reported endpoint scores."""
import json
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.atoms.vectors import choose_atoms
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL
from research.activation_transfer.shape_transfer.quantiles.geometry import material
from research.activation_transfer.shape_transfer.response_subspace.run import CHECKPOINTS
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

OUT = Path(__file__).resolve().parent / "results"
ARMS = ("natural", "attenuated", "corrected", "dense", "hidden_single", "hidden_span")


def direct_arms(profile, target_codes, decoder, goal, weight, donor):
    weights = weight * np.eye(8) + np.full((8, 8), (1. - weight) / 8)
    weighted = weights @ profile
    want = weights @ goal
    signature = -target_codes[:, None] * (decoder @ weighted.T)
    full_losses, scaled_losses, fractions = [], [], []
    for candidate in signature:
        full_losses.append(np.linalg.norm(want - candidate))
        fraction = float(np.clip(np.dot(want, candidate) / max(np.dot(candidate, candidate), 1e-24), 0., 1.))
        fractions.append(fraction)
        scaled_losses.append(np.linalg.norm(want - fraction * candidate))
    natural = int(np.argmin(full_losses))
    attenuated = int(np.argmin(scaled_losses))
    prior = -fractions[attenuated] * target_codes[attenuated] * decoder[attenuated]

    u, values, vh = np.linalg.svd(weighted, full_matrices=False)
    lift = (vh.T * (values / (values**2 + .001 * values[0]**2))) @ u.T
    unit = decoder / np.linalg.norm(decoder, axis=1, keepdims=True)
    angles = unit @ donor
    best = int(np.argmax(np.abs(angles)))
    _, svals, basis = np.linalg.svd(decoder, full_matrices=False)
    rows = basis[svals > svals.max() * 1e-7]
    return (natural, attenuated, best), dict(
        natural=-target_codes[natural] * decoder[natural], attenuated=prior,
        corrected=prior + lift @ (want - weighted @ prior), dense=lift @ want,
        hidden_single=angles[best] * unit[best], hidden_span=(rows @ donor) @ rows,
    )


def score_raw(rec, domain, summary):
    keep = rec["split"] == domain
    cols = np.flatnonzero(rec["active"][keep].sum(0) >= 8)
    errors = []
    for arm in ARMS:
        for name, span in (("early", slice(8)), ("late", slice(8, 16))):
            for part in ("full", "shape"):
                vals = []
                for col in cols:
                    cases = keep[rec["rows"]] & (rec["atom_index"] == col)
                    true = rec["source"][cases, span].astype(float)
                    pred = rec[arm][cases, span].astype(float)
                    if part == "shape":
                        true -= true.mean(1, keepdims=True)
                        pred -= pred.mean(1, keepdims=True)
                    energy = np.vdot(true, true)
                    if energy > 8e-10 * len(true):
                        vals.append(1. - np.vdot(pred - true, pred - true) / energy)
                expected = summary[name + "_" + part][arm]["value"]
                errors.append(abs(float(np.mean(vals)) - expected))
    return errors


def audit(model, summary):
    all_ids, all_x, labels, all_tokens, positions = material(model)
    train = labels == "train"
    take = np.isin(labels, ("daily", "extra_daily"))
    ids, x, tokens = all_ids[take], all_x[take], all_tokens[take]
    assert len(set(ids)) == 320 and np.count_nonzero(train) > 0 and not set(ids).intersection(all_ids[train])

    layer_num = 3 if model == "bolt" else 10
    with np.load(DEPTH / f"{model}_{layer_num}.npz") as file:
        rows = np.isin(file["split"], ("daily", "extra_daily"))
        np.testing.assert_array_equal(file["ids"][rows], ids)
        np.testing.assert_array_equal(file["positions"], positions)
        assert str(file["layer"]) == MODELS[model][-1]
        state_error = float(np.max(np.abs(file["states"][rows] - tokens)))
        np.testing.assert_allclose(file["states"][rows], tokens, atol=8e-5, rtol=3e-5)
        jacs = file["jac"][rows].astype(float)
        radius = float(file["radius"])
    radius_independent = float(np.median(np.linalg.norm(all_tokens[train].mean(1), axis=1)))
    np.testing.assert_allclose(radius, radius_independent, rtol=2e-6)
    with np.load(FULL / "results" / (model + ".npz")) as file:
        weight = float(file["shape_weight"])
    adapter = adapter_for(model)
    spans = adapter.token_time_spans()[positions]
    assert spans[0, 0] == 448 and spans[-1, 1] == 480
    select = last_window_slice(adapter, positions)
    layer = MODELS[model][-1]
    rng = np.random.default_rng(20261121448)
    meta, detectors, codes, reported = {}, {}, {}, {}
    for seed in (1, 2):
        file = CHECKPOINTS / f"seed{seed}_{model}.pt"
        meta[seed] = json.loads(file.with_suffix(".json").read_text())
        with file.open("rb") as handle:
            fingerprint = file_digest(handle, "sha256").hexdigest()
        assert fingerprint == meta[seed]["sha256"] and meta[seed]["seed"] == seed
        detector = load_sae_checkpoint(str(file))
        detectors[seed] = detector.W_dec.detach().numpy().astype(float)
        with torch.inference_mode():
            all_codes = detector.encode(torch.from_numpy(all_tokens.mean(1))).numpy()
        codes[seed] = all_codes[take]
        assert np.all((codes[seed] > 0).sum(1) == 32) and detector.k == 32 and len(detectors[seed]) == 6144
        with np.load(REPLICATION / "results" / f"{model}_seed{seed}.npz") as old:
            np.testing.assert_array_equal(choose_atoms(all_codes[train], 16, 2026112070 + seed), old["atoms"])
            assert str(old["source_checkpoint_hash"]) == fingerprint
            reported[seed] = old["atoms"]
    for field in ("train_item_sha256", "train_state_sha256", "n_rows", "training"):
        assert meta[1][field] == meta[2][field]
    assert meta[1]["sha256"] != meta[2]["sha256"]

    cases, score_errors, replay_errors, source_errors = [], [], [], []
    diagonals = {}
    for donor_seed in (1, 2):
        for receiver_seed in (donor_seed, 3 - donor_seed):
            rec = dict(np.load(OUT / f"{model}_{donor_seed}_{receiver_seed}.npz"))
            np.testing.assert_array_equal(rec["ids"], ids)
            np.testing.assert_array_equal(rec["atoms"], reported[donor_seed])
            native_codes = codes[donor_seed][:, rec["atoms"]]
            np.testing.assert_array_equal(rec["active"], native_codes > 0)
            row_ids, atom_cols = np.nonzero(native_codes > 0)
            np.testing.assert_array_equal(rec["rows"], row_ids)
            np.testing.assert_array_equal(rec["atom_index"], atom_cols)
            for key in ("natural_id", "attenuated_id", "hidden_id"):
                assert np.all(codes[receiver_seed][row_ids, rec[key]] > 0)
            if donor_seed == receiver_seed:
                diagonals[donor_seed] = (rec["source"], rec["dense"])
            else:
                np.testing.assert_array_equal(rec["source"], diagonals[donor_seed][0])
                np.testing.assert_allclose(rec["dense"], diagonals[donor_seed][1], atol=3e-5)
            for domain in ("daily", "extra_daily"):
                row = summary[f"{model}_donor{donor_seed}"][domain]["seed"][str(receiver_seed)]
                score_errors.extend(score_raw(rec, domain, row))
                capable = np.flatnonzero(rec["split"][row_ids] == domain)
                clipped = capable[rec["natural_capped"][capable]]
                selected = [int(rng.choice(capable))]
                remaining = np.setdiff1d(clipped if len(clipped) > 1 else capable, selected)
                if not len(remaining):
                    remaining = np.setdiff1d(capable, selected)
                selected.append(int(rng.choice(remaining)))
                for i in selected:
                    r, atom = rec["rows"][i], rec["atoms"][rec["atom_index"][i]]
                    donor = -float(codes[donor_seed][r, atom]) * detectors[donor_seed][atom]
                    pool = np.flatnonzero(codes[receiver_seed][r] > 0)
                    choices, moves = direct_arms(
                        jacs[r], codes[receiver_seed][r, pool].astype(float), detectors[receiver_seed][pool],
                        rec["source"][i, :8].astype(float), weight, donor,
                    )
                    for k, name in enumerate(("natural_id", "attenuated_id", "hidden_id")):
                        assert pool[choices[k]] == rec[name][i]
                    for key, arm in (("single_fraction", "hidden_single"), ("span_fraction", "hidden_span")):
                        fraction = float(moves[arm] @ moves[arm]) / float(donor @ donor)
                        np.testing.assert_allclose(fraction, rec[key][i], rtol=2e-6, atol=1e-8)
                    clean = adapter.predict(x[r:r+1], 16, [.5])["quantiles"][0, :, 0]
                    for key, delta in {**moves, "native": donor}.items():
                        if key == "native":
                            expected = rec["source"][i]
                        else:
                            scale = min(1., .2 * radius / max(np.linalg.norm(delta), 1e-18))
                            assert rec[key + "_capped"][i] == (scale < 1.)
                            delta = scale * delta
                            expected = rec[key][i]
                        replacement = torch.from_numpy(tokens[r:r+1] + delta.astype("f4")[None, None, :])
                        with token_patch(adapter.module, layer, select, replacement):
                            new = adapter.predict(x[r:r+1], 16, [.5])["quantiles"][0, :, 0]
                        error = float(np.max(np.abs(new - clean - expected)))
                        (source_errors if key == "native" else replay_errors).append(error)
                    cases.append(dict(donor=donor_seed, receiver=receiver_seed, series=str(ids[r]), atom=int(atom)))
    if max(replay_errors + source_errors) > 2e-4 or max(score_errors) > 1e-9:
        raise ValueError("Independent checkpoint replay or squared-error calculation disagreed")
    return dict(
        cases=cases, receiver_replays=len(replay_errors), source_replays=len(source_errors),
        receiver_max_error=max(replay_errors), source_max_error=max(source_errors), max_capture_error=state_error,
        independent_macro_scores=len(score_errors), macro_max_error=max(score_errors),
        model_layer=layer, observed_span=spans.tolist(), radius=radius_independent,
        training_rows=meta[1]["n_rows"], seed1=meta[1]["sha256"], seed2=meta[2]["sha256"],
        training_state_hash=meta[1]["train_state_sha256"], training_item_hash=meta[1]["train_item_sha256"],
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    summary = json.loads((OUT / "summary.json").read_text())
    report = {name: audit(name, summary) for name in ("bolt", "timesfm")}
    (OUT / "independent_replay.json").write_text(json.dumps(report, indent=2) + "\n")
