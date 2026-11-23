"""Independently re-encode selected endpoints from checkpoint weights and replay the five physical writes."""
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.quantiles.geometry import material
from research.activation_transfer.shape_transfer.response_subspace.run import CHECKPOINTS
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

OUT = Path(__file__).resolve().parent / "results"


def hard_code(pre, k):
    positive = np.maximum(pre, 0.)
    active = np.argsort(positive)[-k:]
    result = np.zeros_like(positive)
    result[active] = positive[active]
    return result


def check_subgroup_scores(raw, published):
    arms = ("true", "hard", "locked", "hard_eighth", "locked_eighth")
    errors = []
    for name, include in (
        ("all", np.ones(len(raw["rows"]), dtype=bool)), ("crossed", raw["switch"] > 0),
        ("stable", raw["switch"] == 0),
    ):
        donors = np.unique(raw["atom_index"][include], return_counts=True)
        keep = donors[0][donors[1] >= 8]
        assert len(keep) == published[name]["features"]
        if not len(keep):
            continue
        for arm in arms:
            for label, leads in (("early", slice(8)), ("late", slice(8, 16))):
                for part in ("shape", "full"):
                    values = []
                    for donor in keep:
                        events = include & (raw["atom_index"] == donor)
                        a = raw["source"][events, leads].astype(float)
                        b = raw[arm][events, leads].astype(float)
                        if part == "shape":
                            a -= a.mean(1, keepdims=True)
                            b -= b.mean(1, keepdims=True)
                        power = np.vdot(a, a)
                        if power > len(a) * 8e-10:
                            values.append(1. - np.vdot(a - b, a - b) / power)
                    expected = published[name][label + "_" + part][arm]["value"]
                    errors.append(abs(float(np.mean(values)) - expected))
    return errors


def verify(model):
    all_ids, all_x, labels, tokens, positions = material(model)
    use = labels == "extra_daily"
    radius = float(np.median(np.linalg.norm(tokens[labels == "train"].mean(1), axis=1)))
    ids, inputs, tokens = all_ids[use], all_x[use], tokens[use]
    adapter = adapter_for(model)
    select = last_window_slice(adapter, positions)
    layer = MODELS[model][-1]
    rng = np.random.default_rng(20261121449)
    errors, row_reports, score_errors, endpoint_counts = [], [], [], {}
    published = json.loads((OUT / "encoding.json").read_text())
    for donor in (1, 2):
        raw = dict(np.load(OUT / f"encode_{model}_{donor}.npz"))
        np.testing.assert_array_equal(ids, raw["ids"])
        dst = load_sae_checkpoint(str(CHECKPOINTS / f"seed{3-donor}_{model}.pt"))
        src = load_sae_checkpoint(str(CHECKPOINTS / f"seed{donor}_{model}.pt"))
        bias = dst.b_dec.detach().numpy().astype(float)
        bias_enc = dst.b_enc.detach().numpy().astype(float)
        encoder = dst.W_enc.detach().numpy().astype(float)
        decoder = dst.W_dec.detach().numpy().astype(float)
        with torch.inference_mode():
            z = src.encode(torch.from_numpy(tokens.mean(1))).numpy()
        np.testing.assert_array_equal(z[:, raw["atoms"]] > 0, raw["active"])
        score_errors.extend(check_subgroup_scores(raw, published[f"{model}_donor{donor}"]))
        endpoint_counts[donor] = {}
        row = raw["rows"]
        atom = raw["atoms"][raw["atom_index"]]
        change = -z[row, atom, None] * src.W_dec.detach().numpy()[atom]
        with torch.inference_mode():
            for fraction, name in ((1., "switch"), (.125, "switch_eighth")):
                starts = torch.from_numpy(tokens[row].mean(1))
                stops = starts + fraction * torch.from_numpy(change)
                before, after = (dst.encode(states).numpy() > 0 for states in (starts, stops))
                entered = (after & ~before).sum(1)
                exited = (before & ~after).sum(1)
                np.testing.assert_array_equal(entered, raw[name])
                np.testing.assert_array_equal(entered, exited)
                assert np.all(after.sum(1) == 32) and np.all(before.sum(1) == 32)
                endpoint_counts[donor][name] = dict(events=len(entered), switched=int(np.count_nonzero(entered)))
        # Draw a switching and stable case when both exist. Keep strata fixed before the independent forwards.
        candidates = []
        for changed in (False, True):
            group = np.flatnonzero((raw["switch"] > 0) == changed)
            if len(group):
                candidates.append(int(rng.choice(group)))
        for i in candidates:
            r = raw["rows"][i]
            atom = raw["atoms"][raw["atom_index"][i]]
            change = -float(z[r, atom]) * src.W_dec[atom].detach().numpy().astype(float)
            state = tokens[r].mean(0).astype(float)
            clean = hard_code((state - bias) @ encoder + bias_enc, dst.k)
            moves = {"true": change}
            for fraction, suffix in ((1., ""), (.125, "_eighth")):
                pre = (state + fraction * change - bias) @ encoder + bias_enc
                hard = hard_code(pre, dst.k)
                locked = np.maximum(pre, 0.) * (clean > 0)
                assert np.count_nonzero((hard > 0) & (clean == 0)) == raw["switch" + suffix][i]
                for key, code in (("hard", hard), ("locked", locked)):
                    moves[key + suffix] = ((code - clean) @ decoder) / fraction
            ordinary = adapter.predict(inputs[r:r+1], 16, [.5])["quantiles"][0, :, 0]
            for name, move in moves.items():
                norm = np.linalg.norm(move)
                amount = min(1., .2 * radius / max(norm, 1e-20))
                assert (amount < 1.) == raw[name + "_capped"][i]
                np.testing.assert_allclose(norm / np.linalg.norm(change), raw[name + "_length_ratio"][i],
                                           rtol=2e-4, atol=1e-5)
                dx = amount * move
                np.testing.assert_allclose(np.linalg.norm(change - dx) / np.linalg.norm(change),
                                           raw[name + "_hidden_error"][i], rtol=6e-4, atol=1e-4)
                h = torch.from_numpy(tokens[r:r+1] + dx.astype("f4")[None, None])
                with token_patch(adapter.module, layer, select, h):
                    edited = adapter.predict(inputs[r:r+1], 16, [.5])["quantiles"][0, :, 0]
                errors.append(float(np.max(np.abs(edited - ordinary - raw[name][i]))))
            row_reports.append(dict(seed=donor, series=str(ids[r]), source_atom=int(atom), entrants=int(raw["switch"][i])))
    if max(errors) > 3e-4 or max(score_errors) > 1e-9:
        raise ValueError(f"Independent endpoint replay differs by {max(errors)}")
    return dict(cases=row_reports, real_writes=len(errors), max_abs_error=max(errors),
                endpoint_counts=endpoint_counts, independent_macro_scores=len(score_errors),
                macro_max_error=max(score_errors))


if __name__ == "__main__":
    torch.set_num_threads(2)
    results = {name: verify(name) for name in ("bolt", "timesfm")}
    (OUT / "independent_encoding.json").write_text(json.dumps(results, indent=2) + "\n")
