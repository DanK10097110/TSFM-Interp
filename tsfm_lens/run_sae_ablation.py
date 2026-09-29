"""ROADMAP.md sec 27: what each SAE feature causally does on the series it
actually fires on -- the ABLATION fingerprint.

Component A (`run_stage2_response_fingerprint.py`) asks whether INJECTING a
feature's direction at +-k*sigma moves the forecast more than an arbitrary
direction does, measured over a representative family-stratified sample.
That is the right question for "is this atom causally real at all". It is
the wrong question for "what does this atom DO", for two reasons this pass
exists to fix:

  1. Most series in a representative sample do not fire the atom at all
     (TopK sparsity), so a real effect is diluted by rows where there was
     nothing to remove.
  2. Injecting a direction into series that never carry it measures the
     decoder direction's generic push, not the feature's role in the
     computation the model actually performs when it fires.

So this pass ABLATES (zeroes the atom out of the reconstruction) on the
atom's OWN top-firing series. Both passes share the reach gate, the
baseline convention (full token-level reconstruction, so the SAE's own
reconstruction cost is not charged to the feature) and the same nine
channels, which is what makes their two answers comparable rather than
merely different.

Candidates are read from this target's existing Stage 2 artifact when one
is present, so the correlational and causal views describe the SAME
features -- that pairing is the point of the whole exercise (two features
can fire on the same series for different reasons, and only the causal
half can tell them apart).

The driver body lives in `tsfm_lens/sae/ablation_run.py`, shared with the
`concepts` pipeline stage (ROADMAP.md sec 37.4) so the two cannot drift.

Example:
    python run_sae_ablation.py --run runs/full_report_run_4model --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.extraction.store import ActivationStore
from tsfm_lens.models import ModelHub
from tsfm_lens.sae.ablation_run import ablation_targets, run_ablation_all
from tsfm_lens.utils import resolve_device, resolve_dtype, set_seed, setup_logging


def _targets_for(run_dir: Path, args) -> list:
    if args.all:
        return ablation_targets(run_dir)
    if not args.model or not args.layer:
        raise SystemExit("pass --model and --layer, or --all")
    return [(args.model, args.layer)]


def main() -> None:
    ap = argparse.ArgumentParser(description="ROADMAP.md sec 27: SAE feature ablation fingerprints")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--top-k-series", type=int, default=8,
                    help="how many of each atom's own strongest-firing series to ablate on")
    ap.add_argument("--n-null-directions", type=int, default=16)
    ap.add_argument("--max-series", type=int, default=64)
    ap.add_argument("--keep-forecasts", type=int, default=3,
                    help="per-series with/without forecast pairs kept for the report overlay")
    ap.add_argument("--n-features-per-rule", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)
    run_dir = Path(args.run)
    cfg = load_config(str(run_dir / "config_resolved.yaml"))
    cfg.run.name, cfg.run.seed = run_dir.name, args.seed
    device = resolve_device(cfg.run.device)
    dtype = resolve_dtype(cfg.run.dtype, device)

    store = ActivationStore(run_dir / "activations.zarr", mode="r")
    hub = ModelHub(cfg.models, cfg.data, device, dtype)
    data = load_benchmark(cfg.data, cfg.run.seed)

    written = run_ablation_all(
        cfg, run_dir, hub, data, store, device, _targets_for(run_dir, args),
        top_k_series=args.top_k_series, n_null_directions=args.n_null_directions,
        max_series=args.max_series, keep_forecasts=args.keep_forecasts,
        n_features_per_rule=args.n_features_per_rule)
    for out_path in written:
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
