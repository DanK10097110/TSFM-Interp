"""Physically install the locally fitted weights, compare forecasts, and challenge padding and sites."""
from argparse import Namespace
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch
from safetensors import safe_open
from transformers import T5Config, T5EncoderModel

from research.activation_transfer.moment.forecast_probe.capture import (
    CACHE, MODEL, OUT, model, native, rows, terminal_split,
)
from research.activation_transfer.moment.forecast_quality.score import item_metrics


def install(m, heads, arm):
    lin = m.head.linear
    lin.weight.copy_(torch.from_numpy(heads["hidden_" + arm + "_w"].T))
    lin.bias.copy_(torch.from_numpy(heads["hidden_" + arm + "_b"]))


def forward(m, lag, residual=False, kind="edge"):
    out = []
    for i in range(0, len(lag), 8):
        s = lag[i:i+8]
        y = m(**native(s, kind)).forecast[:, 0].detach().numpy().astype(float)
        if residual:
            y += (s[:, -1] - m.normalizer.mean.flatten().numpy())[:, None]
        out.append(y)
    return np.concatenate(out)


def native_identity(m):
    released = MODEL / "model.safetensors"
    with released.open("rb") as handle:
        archive_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    assert archive_sha == "a56928052ac6f5d09b97c3834bea6ce3aef9f02b513b5fac98954e7377801572"
    live = m.state_dict()
    checked = 0
    with safe_open(released, framework="pt") as f:
        assert set(live) == set(f.keys())
        for k in f.keys():
            if not k.startswith("head."):
                assert torch.equal(live[k], f.get_tensor(k)), k
                checked += 1
    return {"safetensors_sha256":archive_sha, "all_exact_nonhead_state_keys":checked,
            "new_weight_shape":list(live["head.linear.weight"].shape),
            "requires_gradient_parameters":sum(p.requires_grad for p in m.parameters())}


def original_graph(m, lag):
    source = MODEL.parent / "research_src"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "3ab637e413f35f2c317573c0ace280d825c558de"
    sys.path.insert(0, str(source))
    from moment.models.moment import MOMENT as ResearchModel
    from moment.utils.data import upsample_timeseries
    checked_lengths = []
    for length in (464, 480):
        raw = lag[:, -length:]
        batch = native(raw)
        for i, row in enumerate(raw):
            values, mask = upsample_timeseries(row, 512, sampling_type="pad", direction="backward", mode="edge")
            np.testing.assert_array_equal(values.astype("float32"), batch["x_enc"][i, 0])
            np.testing.assert_array_equal(mask, batch["input_mask"][i])
        checked_lengths.append(length)
    canonical = T5Config.from_pretrained(str(MODEL.parent / "flan_large"), local_files_only=True)
    cfg = json.loads((MODEL / "config.json").read_text())
    cfg.update(task_name="short-horizon-forecasting", finetuning_mode="linear-probing",
               forecast_horizon=16, d_model=canonical.d_model)
    def backbone(name):
        assert name == cfg["transformer_backbone"]
        return T5EncoderModel(canonical)
    with patch.object(T5EncoderModel, "from_pretrained", side_effect=backbone):
        original = ResearchModel(Namespace(**cfg)).eval().requires_grad_(False)
    original.load_state_dict(m.state_dict(), strict=True)
    public = forward(m, lag)
    native_pred = m.forecast(**native(lag)).forecast[:, 0].numpy()
    research = original(**native(lag)).forecast[:, 0].numpy()
    research_explicit = original.long_forecast(**native(lag)).forecast[:, 0].numpy()
    np.testing.assert_array_equal(research, research_explicit)
    sd = lag.std(-1, keepdims=True) + 1e-5
    gap_public = float(np.max(abs((public - native_pred) / sd)))
    gap_original = float(np.max(abs((public - research) / sd)))
    assert max(gap_original, gap_public) < 3e-5
    return dict(paper_revision=commit, exact_original_padding_lengths=checked_lengths, paper_head=type(original.head).__name__, public_dispatch_max_sd=gap_public,
                paper_long_forecast_max_sd=gap_original)


def interventions(m, lag):
    base = forward(m, lag)
    sd = m.normalizer.stdev.flatten().numpy().copy()
    linear = m.head.linear.weight.detach().numpy().astype(float).reshape(16, 64, 1024)
    records = []
    for slot in (0, 63):
        # One unit hidden norm in the positive gradient direction for forecast lead 1.
        direction = linear[0, slot].copy()
        direction /= np.linalg.norm(direction)
        delta = torch.from_numpy(direction).float()
        def add(_head, args):
            modified = args[0].clone()
            modified[:, :, slot] += delta
            return (modified,)
        hook = m.head.register_forward_pre_hook(add)
        try:
            patched = forward(m, lag)
        finally:
            hook.remove()
        observed = (patched - base) / sd[:, None]
        expected = linear[:, slot] @ direction
        gap = float(abs(observed - expected).max())
        assert gap < 3e-5 and expected[0] > 1e-6
        records.append(dict(slot=slot, observed_first_lead_sd=float(observed[:, 0].mean()),
                            analytic_first_lead_sd=float(expected[0]), max_gap_sd=gap))
    squared = np.sum(linear**2, axis=(0, 2))
    return dict(actual_unit_write=records, head_squared_weight_fraction_64slots=(squared/squared.sum()).tolist())


def masking(m, test_lag):
    ids, hist, target, _ = rows("train")
    raw = np.concatenate((hist[:2], target[:2]), axis=1)
    contaminated = raw.copy()
    contaminated[:, -16:] += 1e7
    def capture(data, kind):
        all_states = []
        def record(_module, args):
            all_states.append(args[0].numpy().copy())
        hook = m.head.register_forward_pre_hook(record)
        try:
            output = forward(m, data, kind=kind)
        finally:
            hook.remove()
        return np.concatenate(all_states)[:, 0], output
    real, pred = capture(terminal_split(raw)[0], "edge")
    changed, cpred = capture(terminal_split(contaminated)[0], "edge")
    np.testing.assert_array_equal(real, changed)
    np.testing.assert_array_equal(pred, cpred)
    states, out_edge = capture(test_lag, "edge")
    alternative, out_zero = capture(test_lag, "constant")
    dstate = alternative - states
    valid_from = (512 - test_lag.shape[1]) // 8
    pad_delta = float(np.max(abs(dstate[:, :valid_from])))
    visible_delta = float(np.max(abs(dstate[:, valid_from:])))
    assert pad_delta > .01 and visible_delta < 1e-5
    operator = m.head.linear.weight.numpy().reshape(16, 64, 1024)
    transform = np.einsum("bpd,hpd->bh", dstate[:, :valid_from].astype(float), operator[:, :valid_from])
    gap = float(np.max(abs((out_zero - out_edge) / m.normalizer.stdev.numpy().reshape(-1, 1) - transform)))
    assert gap < 3e-5
    return dict(corrupt_label_fit_ids=ids[:2].tolist(), contaminated_label_increment=1e7,
                altered_training_state_or_forecast_coordinates=0, padding_switch_max_hidden_change=pad_delta,
                observed_slot_max_hidden_change=visible_delta, padding_only_linear_replay_max_sd=gap)


def run():
    torch.set_num_threads(1)
    m = model()
    identity = native_identity(m)
    print("Verified all released nonhead state tensors", flush=True)
    with np.load(CACHE / "weights.npz") as f:
        ws = {k:f[k] for k in f.files if k.startswith("hidden")}
    cases, variations = {}, {}
    for cohort in ("test", "weather"):
        ids, lag, target, domain = rows(cohort)
        sub = (np.concatenate([np.where(domain == k)[0][:8] for k in np.unique(domain)]) if cohort == "test" else np.arange(8))
        raw = lag[sub].reshape(-1, lag.shape[-1])
        scale = raw.std(axis=1, keepdims=True) + 1e-5
        scratch = CACHE / f"audit_{cohort}.json"
        if scratch.exists():
            saved = json.loads(scratch.read_text())
            assert saved["case"]["ids"] == ids[sub].tolist()
            cases[cohort] = saved["case"]
            variations.update(saved["sensitivity"])
            continue
        cases[cohort] = dict(ids=ids[sub].tolist(), n_origins=len(raw))
        primary_direct = None
        with np.load(OUT / f"{cohort}_pred.npz") as f:
            for arm in ("direct", "residual"):
                install(m, ws, arm)
                physical = forward(m, raw, residual=arm == "residual")
                if arm == "direct":
                    primary_direct = physical
                cached = f["hidden_" + arm][sub].reshape(-1, 16)
                err = float(abs((physical - cached) / scale).max())
                assert err < 3e-4, (cohort, arm, err)
                cases[cohort][arm + "_max_sd"] = err
                print(cohort, arm, "physical parity", err, flush=True)
        install(m, ws, "direct")
        outputs = {"edge480":primary_direct, "edge464":forward(m, raw[:, -464:]),
                   "zero480":forward(m, raw, kind="constant")}
        groups = domain[sub] if cohort == "test" else np.repeat("NOAA",len(sub))
        for group in np.unique(groups):
            keep = groups == group
            yy, xx = target[sub][keep], lag[sub][keep]
            metric = "mase1" if cohort == "test" else "mae"
            entry = {}
            for k, v in outputs.items():
                yhat = v.reshape(len(sub), lag.shape[1], -1)[keep]
                delta = ((v-outputs["edge480"]) / scale).reshape(len(sub), lag.shape[1], -1)[keep]
                entry[k] = dict(error=float(item_metrics(yhat, yy, xx)[metric].mean()),
                                mean_absolute_change_history_sd=float(abs(delta).mean()))
            variations[str(group)] = entry
        scratch.write_text(json.dumps(dict(case=cases[cohort],
                                           sensitivity={str(k):variations[str(k)] for k in np.unique(groups)})))
        print(cohort, "padding and cropped replay complete", flush=True)
    install(m, ws, "direct")
    _, all_lag, _, _ = rows("test")
    sample = all_lag[[0, 99], 0]
    summary = dict(versions={k:importlib.metadata.version(k) for k in ("torch", "transformers", "momentfm", "numpy")},
                   checkpoint=identity, production_replay=cases, padding_length_sensitivity=variations,
                   site=interventions(m, sample), masking=masking(m, sample), original_public_graph=original_graph(m, sample))
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    with torch.inference_mode():
        run()
