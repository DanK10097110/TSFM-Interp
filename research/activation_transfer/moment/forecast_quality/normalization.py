"""Check unit changes, context-only standardization, batching and native horizon dependence on real histories."""
import argparse
import json

import numpy as np
import torch

from research.activation_transfer.moment.forecast_quality.data import OUT, hash_array
from research.activation_transfer.moment.forecast_quality.predict import NAMES, adapter, cached_call, checkpoint_metadata, save_npz

TRANSFORMS = {
    "moderate_scale_shift": (3.7, 11.2), "large_units": (1e4, -1e3),
    "near_epsilon": (1e-5, 0.), "below_epsilon": (1e-7, 0.),
    "large_offset": (1., 1e4), "float32_resolution": (1., 1e7),
    "sign_reversal": (-1., 0.),
}


def discrepancy(error):
    mark = np.asarray(error, dtype=float)
    rms = np.sqrt(np.square(mark).mean(1))
    absolute = np.abs(mark)
    return dict(mean_abs=float(absolute.mean()), max_abs=float(absolute.max()),
                median_item_rms=float(np.median(rms)), p95_item_rms=float(np.quantile(rms, .95)),
                rows_above_1e_minus_4=int((absolute.max(1) > 1e-4).sum()),
                rows_above_1e_minus_2=int((absolute.max(1) > 1e-2).sum()))


def run(model):
    with np.load(OUT / "panel.npz") as file:
        domains, split = file["domain"], file["split"]
        rows = np.concatenate([np.flatnonzero((domains == d) & (split == "diagnostic"))[:8]
                               for d in np.unique(domains)])
        ids = file["id"][rows]
        x = file["context"][rows, 1, :]
    sd = x.std(1, keepdims=True)
    mean = x.mean(1, keepdims=True)
    z = (x - mean) / sd
    enc = adapter(model)
    _, hashes, versions = checkpoint_metadata(model)
    def forecast(inputs, horizon, batch):
        digest = hash_array(inputs)
        signature = json.dumps([model, digest, hashes, versions, horizon, batch, torch.get_num_threads()], sort_keys=True)
        cache = OUT / ".passes" / f"{model}_normalization_{horizon}_{batch}_{digest}.npz"
        return cached_call(enc, inputs, horizon, batch, cache, signature)[0]
    report = dict(ids=ids.tolist(), observed_context=480, origin="later", transforms=TRANSFORMS)
    arrays = dict(ids=ids, domain=domains[rows])
    order = np.random.default_rng(6429912).permutation(len(rows))
    for horizon in (8,16):
        ref = forecast(z, horizon, 16)
        result = {}
        arrays[f"h{horizon}_standard_input"] = ref
        for name, (a,b) in TRANSFORMS.items():
            obs = a*z + b
            recovered_input = (obs.astype("f4").astype(float) - b) / a
            after = (forecast(obs, horizon, 16) - b) / a
            result[name] = dict(forecast=discrepancy(after - ref), input_roundtrip=discrepancy(recovered_input-z))
            arrays[f"h{horizon}_" + name] = after
        raw = forecast(x, horizon, 16)
        external = ref * sd + mean
        result["observed_raw_vs_external_z"] = discrepancy((raw - external)/sd)
        result["observed_by_domain"] = {
            str(d):discrepancy(((raw - external)/sd)[domains[rows] == d]) for d in np.unique(domains)
        }
        # Each row receives the same input repeatedly, including alongside very different companions.
        single = forecast(z, horizon, 1)
        reordered = forecast(z[order], horizon, 16)[np.argsort(order)]
        companion = np.empty((3*len(z),z.shape[1]))
        companion[0::3], companion[1::3], companion[2::3] = z, 1e4*z[order]+1e5, np.zeros_like(z)
        mixed = forecast(companion, horizon, 12)[::3]
        result["batch"] = {"single_vs_16": discrepancy(single-ref), "permute": discrepancy(reordered-ref),
                           "extreme_companions":discrepancy(mixed-ref)}
        constant = np.array([-7., 0., 1e-8, 7., 1e5])[:,None]
        cy = forecast(np.repeat(constant, z.shape[1], axis=1), horizon, 5)
        result["constant_max_abs_native_units"] = np.abs(cy-constant).max(1).tolist()
        report[str(horizon)] = result
    (OUT / (model + "_normalization.json")).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_npz(OUT / (model + "_normalization.npz"), **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",choices=NAMES)
    torch.set_num_threads(2)
    run(parser.parse_args().model)
