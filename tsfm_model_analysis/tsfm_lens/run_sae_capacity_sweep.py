"""Why is the production SAE dictionary ~95% dead? (ROADMAP.md sec 23.2 A1.)

The recorded condition is stark: `runs/medium_run_chronos_base`'s two pinned
targets train to a 95.9% / 97.4% dead-feature rate, while the crosscoder's
Stage 0 gate reached 3.2% dead on the *same two checkpoints*. Three
explanations were live, and they call for different fixes:

  (a) the corpus is not diverse enough to excite a large dictionary,
  (b) the dictionary is too large for the number of training rows,
  (c) the training procedure never revives an atom once it stops firing.

This script decides between (b) and (c) by measurement, holding the corpus
and the activations fixed: it reads an already-extracted run's store and
trains one `TopKSAE` per grid cell, varying only dictionary size, AuxK and
epoch budget. Nothing is re-extracted and no checkpoint is loaded, so a cell
costs seconds -- which is the point, since the production stage's own dead
rate has never been measured against anything but its one committed setting.

(a) is answered by artifacts that already exist rather than by this grid:
`configs/medium_run_chronos_base.yaml` runs with `real_data_enabled: true`,
so 4000 of its 8608 training rows are real Monash series, and it is 95.9%
dead anyway. Corpus diversity is not the free variable it looks like.

The dead rate here is measured over the benchmark activations only, exactly
as `sae/train.py::run_sae` does, so a number from this grid is comparable to
a number in `sae/meta.json`. Real-data augmentation is *off* in every cell
(it needs a live checkpoint), which makes this grid the `real_data: false`
arm of the same comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.sae.eval import dead_feature_rate, reconstruction_fidelity
from tsfm_lens.sae.train import SAETrainConfig, load_all_windows, train_sae
from tsfm_lens.utils import log, save_json


def sweep_target(acts: np.ndarray, dict_sizes, aux_ks, epochs_list, k: int,
                 lr: float, batch_size: int, resample_every: int,
                 seed: int, device: torch.device) -> list:
    rows = []
    for dict_size in dict_sizes:
        for aux_k in aux_ks:
            for epochs in epochs_list:
                cfg = SAETrainConfig(k=k, lr=lr, epochs=epochs, batch_size=batch_size,
                                     seed=seed, resample_dead_every_epochs=resample_every,
                                     aux_k=aux_k, dict_size=dict_size)
                sae, _ = train_sae(acts, cfg, device)
                dead = dead_feature_rate(sae, acts, device)
                fid = reconstruction_fidelity(sae, acts, device)
                row = {"dict_size": dict_size, "aux_k": aux_k, "epochs": epochs, "k": k,
                       "n_rows": int(acts.shape[0]),
                       "rows_per_atom": round(acts.shape[0] / dict_size, 3),
                       "dead_feature_rate": dead, "n_alive": int(round(dict_size * (1 - dead))),
                       "reconstruction_fidelity": fid}
                rows.append(row)
                log.info(f"  dict={dict_size:>6} aux_k={aux_k:>3} epochs={epochs:>4} "
                         f"-> dead={dead:.4f} alive={row['n_alive']:>5} fid={fid:.4f}")
                del sae
                torch.cuda.empty_cache() if device.type == "cuda" else None
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="existing run directory with an activations.zarr")
    ap.add_argument("--targets", required=True,
                    help="comma-separated model/layer pairs, e.g. 'TimesFM/stacked_xf.18,Chronos-T5-Base/encoder.block.6'")
    ap.add_argument("--dict-sizes", default="", help="comma-separated; default = production 8*d_in, plus 4096,2048,1024,512")
    ap.add_argument("--aux-ks", default="0,64")
    ap.add_argument("--epochs", default="60,240")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--resample-dead-every-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="", help="JSON output path (default <run>/sae/capacity_sweep.json)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    device = torch.device(args.device)
    aux_ks = [int(x) for x in args.aux_ks.split(",") if x != ""]
    epochs_list = [int(x) for x in args.epochs.split(",") if x != ""]

    results = {}
    for target in args.targets.split(","):
        model, layer = target.split("/", 1)
        acts = load_all_windows(store, model, layer)
        d_in = int(acts.shape[1])
        if args.dict_sizes:
            dict_sizes = [int(x) for x in args.dict_sizes.split(",")]
        else:
            dict_sizes = sorted({8 * d_in, 4096, 2048, 1024, 512}, reverse=True)
        log.info(f"{target}: {acts.shape[0]} rows x {d_in} dims; dict sizes {dict_sizes}")
        results[target] = {"d_in": d_in, "n_rows": int(acts.shape[0]),
                           "grid": sweep_target(acts, dict_sizes, aux_ks, epochs_list, args.k,
                                                args.lr, args.batch_size,
                                                args.resample_dead_every_epochs, args.seed, device)}

    out = Path(args.out) if args.out else run_dir / "sae" / "capacity_sweep.json"
    save_json(out, results)
    log.info(f"wrote {out}")
    print(json.dumps({t: r["grid"] for t, r in results.items()}, indent=2)[:400])


if __name__ == "__main__":
    main()
