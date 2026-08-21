"""Is the ~95% dead SAE dictionary a symptom of an under-diverse corpus?
(ROADMAP.md sec 23.2 A1.)

`run_sae_capacity_sweep.py` varies the *dictionary* against a fixed corpus.
This script does the complementary thing: it holds the trained dictionary
fixed and varies the *data* it is measured on, which is the only way to
separate "the benchmark corpus does not excite enough directions" from "the
dictionary is too large for the training budget."

The comparison is against real, established, externally-sourced data --
Monash via `sae/real_data.py`, the same pool `sae.real_data_enabled` already
augments training with -- at the same layer, through the same checkpoint. It
reports, per target:

  * participation-ratio effective dimensionality of the benchmark activations
    and of the real-data activations, which bounds how many dictionary atoms
    could carry distinct structure at all;
  * the production SAE's dead-feature rate measured on benchmark rows (the
    number `sae/meta.json` records) and, separately, on real rows.

If the corpus were the binding constraint, real data would show a markedly
higher effective dimensionality and revive a large share of the atoms the
benchmark leaves dead. If it is not, both numbers move together and the fix
lies in the dictionary/training budget instead.

No training happens here and nothing is written into the run's store; the
checkpoints read are the ones `sae/train.py::run_sae` already saved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.analysis.internals import _participation_ratio
from tsfm_lens.config import load_config
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.eval import dead_feature_rate, reconstruction_fidelity
from tsfm_lens.sae.real_data import extract_real_activations
from tsfm_lens.sae.train import (load_all_windows, load_sae_checkpoint, sanitize,
                                 _sample_real_contexts)
from tsfm_lens.utils import log, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="", help="JSON output (default <run>/sae/corpus_diversity_check.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = cfg.run_dir()
    device = torch.device(cfg.run.device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, getattr(torch, cfg.run.dtype))

    real_contexts = _sample_real_contexts(cfg)
    log.info(f"sampled {real_contexts.shape} real context windows from {cfg.sae.real_data_source}")

    results = {}
    for target in cfg.sae.targets:
        model, layer = target["model"], target["layer"]
        key = f"{model}/{layer}"
        adapter = hub.get(model)
        bench = load_all_windows(store, model, layer)
        real = extract_real_activations(adapter, layer, real_contexts,
                                        cfg.alignment.window, cfg.sae.batch_size, device)
        ed_b = _participation_ratio(torch.from_numpy(bench).to(device).float())
        ed_r = _participation_ratio(torch.from_numpy(real).to(device).float())

        row = {"d_in": int(bench.shape[1]),
               "n_benchmark_rows": int(bench.shape[0]), "n_real_rows": int(real.shape[0]),
               "eff_dim_benchmark": ed_b, "eff_dim_real": ed_r,
               "eff_dim_union": _participation_ratio(
                   torch.from_numpy(np.concatenate([bench, real], axis=0)).to(device).float())}

        ckpt = run_dir / "sae" / sanitize(model) / f"{sanitize(layer)}.pt"
        if ckpt.exists():
            sae = load_sae_checkpoint(str(ckpt)).to(device)
            row.update(dict_size=int(sae.dict_size), k=int(sae.k),
                       dead_on_benchmark=dead_feature_rate(sae, bench, device),
                       dead_on_real=dead_feature_rate(sae, real, device),
                       dead_on_union=dead_feature_rate(
                           sae, np.concatenate([bench, real], axis=0), device),
                       fidelity_on_benchmark=reconstruction_fidelity(sae, bench, device),
                       fidelity_on_real=reconstruction_fidelity(sae, real, device))
        else:
            row["checkpoint_missing"] = str(ckpt)

        results[key] = row
        log.info(f"{key}: eff_dim bench={ed_b:.2f} real={ed_r:.2f} union={row['eff_dim_union']:.2f}"
                 + (f" | dead bench={row['dead_on_benchmark']:.4f} real={row['dead_on_real']:.4f}"
                    f" union={row['dead_on_union']:.4f}" if "dead_on_benchmark" in row else ""))
        hub.release(model)

    out = Path(args.out) if args.out else run_dir / "sae" / "corpus_diversity_check.json"
    save_json(out, results)
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
