"""Exchange clean and edited token states while the rest of the forecast stays live."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import DATA_ROOT, MODELS, TEST, adapter_for, data
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite, _get, _replace

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1] / "forecast_requests" / "results"
NAMES = ("global_offset", "recent_step", "recent_ramp")


def stimuli(width):
    edits = np.zeros((3, width), dtype="f4")
    edits[0] = .05
    edits[1, -32:] = .05
    ramp = np.linspace(-1, 1, 32)
    edits[2, -32:] = .05 * ramp / np.sqrt(np.mean(ramp**2))
    return edits


def forward(model, site, histories, replacement=None, mode="all"):
    captured = []

    def callback(module, args, kwargs):
        packed = dict(args=args, kwargs=kwargs)
        states = _get(packed, site.path)
        if tuple(states.shape) != (len(histories), *site.shape):
            raise ValueError("Registered block geometry changed")
        captured.append(states.detach().clone())
        if replacement is None:
            return
        if replacement.shape != states.shape:
            raise ValueError("Replacement geometry changed")
        if mode == "all":
            edited = replacement
        elif mode == "last":
            edited = states.clone()
            edited[:, -1] = replacement[:, -1]
        else:
            raise ValueError(mode)
        packed = _replace(packed, site.path, edited)
        return packed["args"], packed["kwargs"]

    block = model.module.get_submodule(site.module)
    with block.register_forward_pre_hook(callback, with_kwargs=True):
        forecast = model.forecast(histories, 8)[..., 0]
    if len(captured) != 1:
        raise ValueError("The block must execute once")
    return forecast, captured[0]


def run(name, out, batch):
    all_ids, histories = data(DATA_ROOT)
    ids, histories = all_ids[TEST], histories[TEST]
    meta = json.loads((PARENT / (name + ".json")).read_text(encoding="utf-8"))
    np.testing.assert_array_equal(ids, meta["test_ids"])
    np.testing.assert_array_equal(MODELS[name], meta["source_fits"][name]["checkpoint"])
    site = InterventionSite(**meta["source_fits"][name]["site"][0])
    model = SteeringModel.from_adapter(adapter_for(name), quantiles=[.5])
    parts = {}
    def store(key, value):
        parts.setdefault(key, []).append(value)
    for start in range(0, len(ids), batch):
        x = histories[start:start + batch]
        original, h0 = forward(model, site, x)
        restored, _ = forward(model, site, x, h0)
        np.testing.assert_allclose(restored, original, atol=1e-7, rtol=0)
        for label, edit in zip(NAMES, stimuli(x.shape[1])):
            y = x + edit
            f1, h1 = forward(model, site, y)
            store(label + "_total", f1 - original)
            for mode in ("last", "all"):
                from_clean, _ = forward(model, site, x, h1, mode)
                from_edit, _ = forward(model, site, y, h0, mode)
                store(label + "_" + mode + "_sufficient", from_clean - original)
                store(label + "_" + mode + "_remaining", from_edit - original)
                a = h0[:, -1] if mode == "last" else h0
                b = h1[:, -1] if mode == "last" else h1
                movement = (b-a).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)
                store(label + "_" + mode + "_movement", movement.cpu().numpy())
    out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out / (name + ".npz"), ids=ids, **{k: np.concatenate(v) for k,v in parts.items()})


def analyze(out):
    rows = []
    for name in MODELS:
        with np.load(out / (name + ".npz")) as z:
            for domain in ("daily", "weekly"):
                keep = np.char.startswith(z["ids"], domain + "/")
                for label in NAMES:
                    total = z[label + "_total"][keep]
                    energy = np.sum(total**2)
                    row = dict(model=name, domain=domain, input=label, n=int(keep.sum()),
                               output_rms=float(np.sqrt(np.mean(total**2))),
                               maximum_lift_error=float(abs(total - .05).max()) if label == "global_offset" else None)
                    for mode in ("last", "all"):
                        row[mode + "_median_relative_movement"] = float(np.median(z[label + "_" + mode + "_movement"][keep]))
                        for intervention in ("sufficient", "remaining"):
                            response = z[label + "_" + mode + "_" + intervention][keep]
                            stem = mode + "_" + intervention
                            row[stem + "_gain"] = float(np.sum(response * total) / energy)
                            row[stem + "_recovery"] = float(1 - np.sum((response-total)**2) / energy)
                    rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[*MODELS, "analyze"])
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.task == "analyze":
        analyze(args.out)
    else:
        run(args.task, args.out, args.batch)
