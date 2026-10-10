"""Wall-clock of the ablation battery per feature at two designs on ONE real target (ROADMAP.md
sec 42 decision 2), read-only from an existing run.

The run directory is opened only for reading: its SAE checkpoint, activation store and corpus
are loaded, the candidate features are read from the target's existing `*_ablation.json`, and
every output goes to `--out`, a different directory. Nothing is written into the run.

    python run_powered_timing.py --run <runs/panel7_v3_dev> --model Chronos-Bolt \
        --layer encoder.block.4 --out runs/powered_timing --n-features 8

Each design is run under the run's own dual null (`profile_matched_cov` primary,
`profile_matched`), with the powered test off and on, on the same candidates; the table gives
seconds per feature. The first measurement of a model on a GPU pays for CUDA warm-up, so one
discarded warm-up run precedes the timed ones.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.ablation_run import ablation_path, run_ablation_target
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, setup_logging


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-features", type=int, default=8)
    ap.add_argument("--corpus", default=None, help="override data.path (absolute)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--designs", nargs="+", default=["8x16", "32x64"])
    args = ap.parse_args(argv)
    setup_logging()
    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    if args.corpus:
        cfg.data.path = args.corpus
    if args.device:
        cfg.run.device = args.device
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)
    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)
    existing = load_json(ablation_path(run_dir, args.model, args.layer))
    cands = [{"feature": int(c["feature"]), "rules": c.get("rules", [])}
             for c in existing["candidates"] if c.get("scorable")][:args.n_features]
    log.info("timing: %s/%s on %d candidates (read-only from %s)", args.model, args.layer,
             len(cands), run_dir)

    def once(top_k, n_null, powered):
        t0 = time.time()
        art = run_ablation_target(
            cfg, run_dir, hub, data, store, device, args.model, args.layer,
            top_k_series=top_k, n_null_directions=n_null, max_series=cfg.concepts.max_series,
            keep_forecasts=0, candidates=cands, powered_test=powered)
        return time.time() - t0, art

    once(8, 16, False)
    rows = []
    for design in args.designs:
        top_k, n_null = (int(v) for v in design.lower().split("x"))
        for powered in (False, True):
            secs, art = once(top_k, n_null, powered)
            scorable = [c for c in art.get("candidates", []) if c.get("scorable")]
            rows.append({"design": design, "powered_test": powered, "seconds": secs,
                         "n_features": len(cands), "n_scorable": len(scorable),
                         "seconds_per_feature": secs / len(cands),
                         "n_chunks": art.get("n_chunks"),
                         "series_per_chunk_cap": art.get("series_per_chunk_cap"),
                         "rows_scored_median": sorted(c["n_rows_scored"] for c in scorable)[len(scorable) // 2]
                         if scorable else None,
                         "ablation_nulls": art.get("ablation_nulls"),
                         "batch_size": hub.get(args.model).cfg.batch_size,
                         **({"powered_summary": art["powered"],
                             "candidates_powered": [
                                 {"feature": c["feature"], "legacy_n_channels_clearing": c["n_channels_clearing"],
                                  "p_holm_t": c["powered"]["p_holm_t"] if c["powered"]["scored"] else None,
                                  "p_holm_rank": c["powered"]["p_holm_rank"] if c["powered"]["scored"] else None,
                                  "causal": c["powered"]["causal"]} for c in scorable]}
                            if powered and scorable else {})})
            log.info("timing: %s powered=%s: %.2fs (%.3fs/feature)", design, powered, secs,
                     secs / len(cands))
    out = {"run": str(run_dir), "model": args.model, "layer": args.layer, "device": str(device),
           "features": [c["feature"] for c in cands], "rows": rows}
    save_json(Path(args.out) / "timing.json", out)
    return out


if __name__ == "__main__":
    main()
