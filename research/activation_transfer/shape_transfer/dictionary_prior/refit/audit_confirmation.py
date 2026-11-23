"""Rebuild the locked panel from raw Arrow rows, remeasure physical writes and check primary contrasts."""
import json
from hashlib import file_digest, sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from research.activation_transfer.atoms.materials import CACHE as ATOMS, MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE as REPLICATION
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.depth.geometry import SCRATCH as DEPTH
from research.activation_transfer.shape_transfer.dictionary_prior.refit.confirm_overlap import OUT
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as EXTRA
from research.activation_transfer.shape_transfer.rolling.data import CACHE as ROLLING
from research.activation_transfer.shape_transfer.response_subspace.run import CHECKPOINTS
from tsfm_lens.extraction.extract import capture_raw_tokens
from tsfm_lens.extraction.hooks import token_patch
from tsfm_lens.sae.train import load_sae_checkpoint

CACHED = Path("/tmp/tsfm_gate_confirm_20261121")


def raw_inputs():
    manifest = json.loads((OUT / "confirm_manifest.json").read_text())
    previous_hashes = set()
    for cache in (ATOMS, REPLICATION, EXTRA, BLIND, ROLLING):
        with np.load(cache / "data.npz") as arch:
            previous_hashes.update(sha256(x.tobytes()).digest() for x in arch["x"])
    inputs, ids, counts = [], [], {}
    for domain, seed in (("daily", 1102189), ("weekly", 1102191)):
        material = manifest["domains"][domain]
        assert material["seed"] == seed
        path = Path(material["source"])
        with path.open("rb") as file:
            digest = file_digest(file, "sha256").hexdigest()
        with pa.memory_map(str(path)) as file:
            table = pa.ipc.open_stream(file).read_all()
        key = "item_id" if "item_id" in table.column_names else "id"
        items = table[key].to_pylist()
        assert len(items) == len(set(items))
        exclude = set(material["excluded"])
        local = previous_hashes.copy()
        eligible = []
        skip = dict(short=0, invalid=0, reserved_id=0, normalized_duplicate=0)
        for i in np.random.default_rng(seed).permutation(len(table)):
            item = str(items[i])
            target = table["target"][int(i)].values.to_numpy(zero_copy_only=False)
            if len(target) < 488:
                skip["short"] += 1
                continue
            current = np.asarray(target[-488:-8], dtype=np.float64)
            if not np.isfinite(current).all() or current.std() < 1e-8:
                skip["invalid"] += 1
                continue
            if item in exclude:
                skip["reserved_id"] += 1
                continue
            window = ((current - np.mean(current)) / np.std(current)).astype("f4")
            hashed = sha256(window.tobytes()).digest()
            if hashed in local:
                skip["normalized_duplicate"] += 1
                continue
            local.add(hashed)
            eligible.append((item, window, hashed))
        chosen = eligible[:96]
        assert [row[0] for row in chosen] == material["items"]
        previous_hashes.update(row[2] for row in chosen)
        ids.extend(domain + "/" + row[0] for row in chosen)
        inputs.extend(row[1] for row in chosen)
        counts[domain] = dict(total_raw=len(items), exclusions=len(exclude), drop=skip, eligible=len(eligible),
                              taken=len(chosen), source_sha256=digest)
        assert len(eligible) + sum(skip.values()) == len(items)
    assert ids == manifest["ids"]
    raw = np.asarray(inputs)
    assert sha256(raw.tobytes()).hexdigest() == manifest["input_sha256"]
    with np.load(CACHED / "data.npz") as file:
        np.testing.assert_array_equal(ids, file["ids"])
        np.testing.assert_array_equal(raw, file["x"])
    return np.asarray(ids), raw, counts


def codes(pre):
    """Construct TopK and shoulder codes directly from sorted rectified preactivations."""
    pos = np.maximum(pre, 0.)
    order = np.argsort(-pos, kind="stable")
    sparse = np.zeros_like(pos)
    sparse[order[:32]] = pos[order[:32]]
    floor = pos[order[32]]
    width = max(pos[order[15]] - floor, 1e-8)
    shoulder = pos * np.minimum(np.maximum(pos - floor, 0.) / width, 1.)
    return sparse, shoulder


def contrasts(data, summary, allowed=None):
    result = {}
    for domain in np.unique(data["split"]):
        include = data["split"] == domain
        if allowed is not None:
            include &= allowed
        rows = np.flatnonzero(include)
        cols = np.flatnonzero(data["active"][rows].sum(0) >= 8)
        assert summary[domain]["features"] == len(cols)
        if not len(cols):
            result[domain] = dict(eligible=0)
            continue
        take = np.isin(data["rows"], rows) & np.isin(data["atom_index"], cols)
        assert int(take.sum()) == summary[domain]["events"]
        error_gain = np.zeros((len(rows), len(cols)))
        power = np.zeros_like(error_gain)
        rmap = dict(zip(rows, range(len(rows))))
        cmap = dict(zip(cols, range(len(cols))))
        for i in np.flatnonzero(take):
            r, c = rmap[data["rows"][i]], cmap[data["atom_index"][i]]
            target, hard, locked = [data[k][i, 8:].astype(float) for k in ("source", "hard", "locked")]
            target -= target.mean()
            hard -= hard.mean()
            locked -= locked.mean()
            power[r, c] = np.vdot(target, target)
            error_gain[r, c] = np.vdot(hard - target, hard - target) - np.vdot(locked - target, locked - target)
        measured = power.sum(0) > 8e-10 * data["active"][np.ix_(rows, cols)].sum(0)
        point = float(np.mean(error_gain[:, measured].sum(0) / power[:, measured].sum(0)))
        expected = summary[domain]["late_shape"]["locked_minus_hard"]
        np.testing.assert_allclose(point, expected["value"], rtol=1e-11, atol=1e-11)
        hist = np.random.default_rng(2026112021).multinomial(len(rows), np.full(len(rows), 1/len(rows)), size=4000)
        feat = np.random.default_rng(26112106).multinomial(len(cols), np.full(len(cols), 1/len(cols)), size=4000)
        draw_energy = hist @ power
        draw_gain = hist @ error_gain
        keep = draw_energy > 8e-10 * (hist @ data["active"][np.ix_(rows, cols)].astype(float))
        keep &= measured[None, :]
        fraction = np.divide(draw_gain, draw_energy, out=np.zeros_like(draw_gain), where=keep)
        weight = feat * keep
        result_draws = np.divide((weight * fraction).sum(1), weight.sum(1),
                                 out=np.full(len(weight), np.nan), where=weight.sum(1) > 0)
        lower = float(np.nanquantile(result_draws, .05/8))
        np.testing.assert_allclose(lower, expected["family_one_sided_lower"], rtol=1e-10, atol=1e-10)
        result[domain] = dict(eligible=len(cols), events=int(take.sum()), point=point, family_lower=lower)
    return result


def physical(model, ids, raw, reported, strict, unseen):
    adapter = adapter_for(model)
    layer = MODELS[model][-1]
    spans = adapter.token_time_spans()
    positions = np.flatnonzero(spans[:, 0] >= 448)
    assert spans[positions[0], 0] == 448 and spans[positions[-1], 1] == 480
    select = last_window_slice(adapter, positions)
    prefix = "3" if model == "bolt" else "10"
    with np.load(DEPTH / f"{model}_{prefix}.npz") as geom:
        radius = float(geom["radius"])
    with np.load(CACHED / f"{model}.npz") as archived:
        np.testing.assert_array_equal(ids, archived["ids"])
        np.testing.assert_array_equal(positions, archived["positions"])
        old_h = archived["tokens"]
    cases, max_error, evals, restricted = [], 0., {}, {}
    for seed in (1, 2):
        src = load_sae_checkpoint(str(CHECKPOINTS / f"seed{seed}_{model}.pt"))
        dst = load_sae_checkpoint(str(CHECKPOINTS / f"seed{3-seed}_{model}.pt"))
        with np.load(OUT / f"confirm_{model}_{seed}.npz") as arch:
            data = dict(arch)
        np.testing.assert_array_equal(ids, data["ids"])
        evals[seed] = contrasts(data, reported[f"{model}_donor{seed}"])
        restricted[seed] = contrasts(data, strict[f"{model}_donor{seed}"], unseen)
        first = np.random.default_rng(20261121452 + seed)
        src_encoder, dst_encoder = [s.W_enc.detach().numpy().astype(float) for s in (src, dst)]
        sdec, tdec = [s.W_dec.detach().numpy().astype(float) for s in (src, dst)]
        for domain in ("weekly", "daily"):
            fired = np.flatnonzero(data["split"][data["rows"]] == domain)
            i = int(first.choice(fired))
            row, atom = data["rows"][i], data["atoms"][data["atom_index"][i]]
            obs = raw[row:row+1]
            h = capture_raw_tokens(adapter, obs, [layer])[layer][:, positions].numpy()
            np.testing.assert_allclose(h, old_h[row:row+1], rtol=6e-5, atol=8e-5)
            state = h[0].mean(0).astype(float)
            zs, _ = codes((state - src.b_dec.detach().numpy()) @ src_encoder + src.b_enc.detach().numpy())
            v = -zs[atom] * sdec[atom]
            assert zs[atom] > 0
            pre = (state - dst.b_dec.detach().numpy()) @ dst_encoder + dst.b_enc.detach().numpy()
            start, shoulder_start = codes(pre)
            end_pre = pre + v @ dst_encoder
            end, shoulder_end = codes(end_pre)
            assert np.count_nonzero((end > 0) & (start == 0)) == data["switch"][i]
            moves = dict(source=v, true=v, hard=(end-start) @ tdec,
                         locked=(np.maximum(end_pre, 0.) * (start > 0) - start) @ tdec,
                         shoulder=(shoulder_end - shoulder_start) @ tdec)
            for name in ("hard", "locked", "shoulder"):
                moves[name + "_equal"] = moves[name] * (np.linalg.norm(v) / max(np.linalg.norm(moves[name]),1e-12))
            ordinary = adapter.predict(obs, 16, [.5])["quantiles"][0, :, 0]
            for name, vec in moves.items():
                scalar = 1. if name == "source" else min(1., .2*radius/max(np.linalg.norm(vec),1e-12))
                assert data[name + "_capped"][i] == (scalar < 1.)
                replacement = torch.from_numpy(h + (vec*scalar).astype("f4")[None,None])
                with token_patch(adapter.module, layer, select, replacement):
                    out = adapter.predict(obs, 16, [.5])["quantiles"][0, :, 0]
                error = float(np.max(np.abs(out - ordinary - data[name][i])))
                max_error = max(error, max_error)
            cases.append(dict(series=str(ids[row]), seed=seed, source_atom=int(atom)))
    assert max_error < 3e-4, max_error
    return dict(cases=cases, writes=8*len(cases), max_abs_error=max_error, primary=evals,
                without_audited_prior_use=restricted)


if __name__ == "__main__":
    torch.set_num_threads(2)
    ids, raw, counts = raw_inputs()
    report = json.loads((OUT / "confirmation.json").read_text())
    strict = json.loads((OUT / "confirmation_disjoint.json").read_text())
    overlap = json.loads((OUT / "confirm_overlap.json").read_text())
    np.testing.assert_array_equal(overlap["ids"], ids)
    unseen = np.array(overlap["frozen_order_mask"])
    result = dict(inputs=counts, replay={model: physical(model, ids, raw, report, strict, unseen)
                                        for model in ("bolt","timesfm")})
    (OUT / "confirm_independent.json").write_text(json.dumps(result, indent=2) + "\n")
