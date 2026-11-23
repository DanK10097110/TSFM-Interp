"""Verify the named reciprocal SAE matches from freshly captured reserve histories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.boundary.patch_forecast import change_forecasts, last_window_slice
from research.activation_transfer.replication.data import HERE
from tsfm_lens.sae.train import load_sae_checkpoint


def replay(model, scratch, results):
    """Apply each actual native feature where it fires, independently of the matched model."""
    other = "timesfm" if model == "bolt" else "bolt"
    expected = {}
    with np.load(results / f"{model}_seed0.npz") as primary, np.load(results / f"{model}_coactive.npz") as direct:
        np.testing.assert_array_equal(primary["ids"], direct["ids"])
        ids = primary["ids"]
        info = json.loads((results / f"{model}_coactive.json").read_text(encoding="utf-8"))
        j = info["selected_index"]
        native_id = int(primary["atoms"][j])
        expected[native_id] = [("selected_source", primary["source_codes"][:,j], primary["native"][:,j])]
    with np.load(results / f"{other}_coactive.npz") as reverse:
        np.testing.assert_array_equal(ids, reverse["ids"])
        info = json.loads((results / f"{other}_coactive.json").read_text(encoding="utf-8"))
        j = info["selected_index"]
        reciprocal_id = int(reverse["target_atoms"][j])
        expected.setdefault(reciprocal_id, []).append(("selected_target", reverse["native_target_codes"][:,j],
                                                       reverse["selected_target_native"]))
    with np.load(scratch / "data.npz") as f:
        keep = f["split"] != "train"
        np.testing.assert_array_equal(ids, f["ids"][keep])
        raw = f["x"][keep]
    with np.load(scratch / f"{model}.npz") as f:
        np.testing.assert_array_equal(ids, f["ids"][keep])
        tokens, positions = f["tokens"][keep], f["positions"]
    sae = load_sae_checkpoint(str(dictionary_path(model)))
    with torch.inference_mode():
        codes = sae.encode(torch.from_numpy(tokens.mean(axis=1))).numpy()
    enc = adapter_for(model)
    site = MODELS[model][-1]
    selector = last_window_slice(enc, positions)
    base = change_forecasts(enc, site, raw, tokens, None, selector, 16)
    output = {}
    for atom, references in expected.items():
        z = codes[:,atom]
        rows = z > 0
        direction = sae.W_dec[atom].detach().numpy()
        edit = -z[rows,None] * direction[None]
        changed = change_forecasts(enc, site, raw[rows], tokens[rows], edit, selector, 16)
        actual = np.zeros_like(base)
        actual[rows] = changed - base[rows]
        record = {}
        for name, old_codes, area in references:
            np.testing.assert_array_equal(z > 0, old_codes > 0)
            np.testing.assert_allclose(z, old_codes, rtol=1e-5, atol=3e-5)
            np.testing.assert_allclose(actual, area, rtol=1e-5, atol=1e-5)
            record[name] = dict(n_series=len(ids), firings=int(rows.sum()),
                                max_activation_change=float(np.max(np.abs(z-old_codes))),
                                max_forecast_change=float(np.max(np.abs(actual-area))))
        output[str(atom)] = record
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    scores = {model: replay(model, a.scratch, a.results) for model in MODELS}
    a.output.write_text(json.dumps(scores, indent=2)+"\n", encoding="utf-8")
