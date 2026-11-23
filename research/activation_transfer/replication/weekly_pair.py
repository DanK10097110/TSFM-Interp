"""Check the frozen daily SAE pair under the same causal intervention on weekly histories."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.boundary.analyze import load_study
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import HERE
from tsfm_lens.sae.train import load_sae_checkpoint


def evaluate(scratch, output):
    captures = load_study(scratch, "weekly")
    arrays = dict(ids=captures["bolt"].ids, recent_mean=captures["bolt"].raw[:, -32:].mean(1))
    report = {}
    for name, index in (("timesfm", 742), ("bolt", 5432)):
        cap = captures[name]
        sae = load_sae_checkpoint(str(dictionary_path(name)))
        with torch.inference_mode():
            z = sae.encode(torch.from_numpy(cap.clean)).numpy()[:, index]
            v = sae.W_dec[index].numpy()
        enc = adapter_for(name)
        layer = MODELS[name][-1]
        select = last_window_slice(enc, cap.positions)
        base = change_forecasts(enc, layer, cap.raw, cap.tokens, None, select, 16)
        zero = change_forecasts(enc, layer, cap.raw, cap.tokens, np.zeros_like(cap.clean), select, 16)
        np.testing.assert_allclose(zero, base, atol=1e-5, rtol=1e-5)
        active = z > 0
        wave = np.zeros_like(base)
        if active.any():
            altered = change_forecasts(enc, layer, cap.raw[active], cap.tokens[active], -z[active,None]*v,
                                       select, 16)
            wave[active] = altered - base[active]
        arrays[name+"_z"] = z
        arrays[name+"_native"] = wave
        report[name] = dict(feature=index, site=layer, firings=int(active.sum()),
            max_zero_drift=float(np.abs(zero-base).max()), recent_mean_auc=float(roc_auc_score(active,arrays["recent_mean"]))
            if np.unique(active).size==2 else None)
    report["n"] = len(arrays["ids"])
    report["joint"] = int(np.sum((arrays["timesfm_z"] > 0) & (arrays["bolt_z"] > 0)))
    draws = resampling_counts(report["n"])
    for source, other in (("timesfm","bolt"),("bolt","timesfm")):
        active = arrays[source+"_z"] > 0
        if not active.any():
            continue
        observed = draws @ active.astype(int) > 0
        for kind in ("full", "shape"):
            reference, prediction = (arrays[key+"_native"][:,None].astype(float) for key in (source, other))
            if kind == "shape":
                reference -= reference.mean(axis=-1,keepdims=True)
                prediction -= prediction.mean(axis=-1,keepdims=True)
            _, point, boot = aggregate(active[:,None], prediction, reference, draws[observed])
            report[source][kind] = {k:ci(point[k], boot[k]) for k in ("cosine","skill","gain")}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"weekly_pair.npz", **arrays)
    (output/"weekly_pair.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    p.add_argument("--output", type=Path, default=HERE/"results")
    a = p.parse_args()
    torch.set_num_threads(2)
    evaluate(a.scratch, a.output)
