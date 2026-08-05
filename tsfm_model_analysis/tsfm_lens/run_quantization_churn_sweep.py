"""ROADMAP.md §7 follow-up: test the "continuous embedding vs. quantization"
hypothesis behind `run_noise_snr_sweep.py`'s noise-SNR x depth sign flip.

That sweep found the cross-model depth-profile rank agreement under additive
noise flips sign between SNR=20dB (rho=+0.40) and SNR=12dB (rho=-0.44), then
plateaus at strong anti-correlation (rho ~ -0.76 to -0.85) from 6dB down
through -3dB -- but didn't test *why*, naming a direct tokenizer inspection
as the concrete next step (the same method `CLAUDE.md` §11.16 used to find
that `impulse_alignment_check`'s hardcoded impulse amplitude was silently
miscalibrated for Chronos's context-adaptive quantization tokenizer).

This script reuses the *exact* series and noise draws `run_l3`/
`run_noise_snr_sweep.py` used (same row-sampling rng, `corrupt_noise` from
`analysis/l3_perturbation.py` with the same rng derivation, so points line
up 1:1 against an existing `noise_snr_sweep.json`) and, for whichever model
in the comparison pair exposes a quantization tokenizer (skips with a log,
per `CLAUDE.md` §2.5, for continuous-embedding models like TimesFM -- there
is nothing analogous to measure there), tokenizes clean vs. corrupted
contexts at each SNR and measures per-series token-ID churn: what fraction
of tokens land in a different quantization bin, and how large the jump is
when they do.

Reading the result: if churn stays near zero through the sign-flip point and
only ramps up once the sweep enters the anti-correlation plateau, that's
direct evidence the depth-profile disagreement tracks a quantization
instability, not just "more noise" in the abstract. If churn ramps smoothly
across the whole range with no relation to the sign-flip's location, that
weakens the quantization hypothesis (the effect would need a different
explanation).

Reuses an already-extracted run's store/models -- no re-extraction, no new
downloads, and no forward passes at all (`prepare()` only tokenizes; it does
not run the encoder), so this is far cheaper than the sensitivity sweep it's
following up on.

Example:
    python run_quantization_churn_sweep.py --run runs/medium_run_chronos_base \\
        --snr 20,12,6,3,0,-3 --max-series 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.l3_perturbation import corrupt_noise
from tsfm_lens.analysis.quantization_churn import token_churn_stats
from tsfm_lens.config import load_config
from tsfm_lens.data import load_benchmark
from tsfm_lens.models import ModelHub
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, save_json, set_seed, setup_logging


def _has_quantization_tokenizer(adapter) -> bool:
    """True for adapters (e.g. Chronos-T5) whose `pipeline.tokenizer` maps
    context values to discrete bin IDs via `context_input_transform` --
    false for continuous-embedding (TimesFM) or patch-based (Chronos-Bolt)
    adapters, which have nothing analogous for this probe to measure.
    """
    pipeline = getattr(adapter, "pipeline", None)
    tokenizer = getattr(pipeline, "tokenizer", None)
    return callable(getattr(tokenizer, "context_input_transform", None))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md §7: quantization-churn probe for the noise-SNR sign flip")
    parser.add_argument("--run", required=True, help="an already-extracted run directory")
    parser.add_argument("--snr", default="20,12,6,3,0,-3",
                        help="comma-separated SNR_dB values; should match run_noise_snr_sweep.py's")
    parser.add_argument("--max-series", type=int, default=96,
                        help="must match the max_series used for the noise_snr_sweep.json being "
                             "compared against, else a different row sample is drawn")
    parser.add_argument("--sweep-json", default=None,
                        help="existing noise_snr_sweep.json to juxtapose against "
                             "(default: <run's parent>/noise_snr_sweep.json)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    base_cfg = load_config(run_dir / "config_resolved.yaml")
    set_seed(base_cfg.run.seed)
    device = resolve_device(base_cfg.run.device)
    dtype = resolve_dtype(base_cfg.run.dtype, device)
    hub = ModelHub(base_cfg.models, base_cfg.data, device, dtype)
    data = load_benchmark(base_cfg.data, base_cfg.run.seed)
    model_a, model_b = base_cfg.comparison_pair()

    # Exact same row sample `run_l3` draws (`analysis/l3_perturbation.py`),
    # so the noise realizations line up 1:1 with an existing
    # `noise_snr_sweep.json`'s already-published rho values.
    rng_rows = np.random.default_rng(base_cfg.run.seed + 3)
    n_rows = min(data.n, args.max_series)
    rows = np.sort(rng_rows.choice(data.n, size=n_rows, replace=False))
    contexts = data.contexts()[rows]

    target = None
    for mcfg in (model_a, model_b):
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        if _has_quantization_tokenizer(adapter):
            target = (mcfg.name, adapter)
            break
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)
    if target is None:
        log.warning("neither %s nor %s exposes a quantization tokenizer "
                    "(pipeline.tokenizer.context_input_transform); nothing for this probe "
                    "to measure, skipping", model_a.name, model_b.name)
        return
    model_name, adapter = target
    log.info("quantization-churn probe: %s (the quantized-tokenizer model in this pair)", model_name)

    snrs = [float(s) for s in args.snr.split(",")]
    points = []
    for snr in snrs:
        # Same rng derivation `run_l3` uses for the sole "noise" corruption
        # (index 0 in cfg.l3.corruptions): a fresh Generator from the same
        # seed reproduces the identical standard-normal draw at every SNR,
        # so only snr_db's scaling differs between sweep points -- common
        # random numbers, matching run_noise_snr_sweep.py's own design.
        rng = np.random.default_rng(base_cfg.run.seed + 100)
        corrupted = corrupt_noise(contexts.copy(), rng, snr_db=snr)
        ids_clean, mask_clean = adapter.prepare(contexts)
        ids_corrupt, mask_corrupt = adapter.prepare(corrupted)
        if not np.array_equal(mask_clean.cpu().numpy(), mask_corrupt.cpu().numpy()):
            raise ValueError(
                f"attention_mask differs between clean/corrupted contexts at snr_db={snr}; "
                "token-position bookkeeping assumption violated -- fix before trusting churn numbers")
        stats = token_churn_stats(ids_clean.cpu().numpy(), ids_corrupt.cpu().numpy(),
                                  mask_clean.cpu().numpy())
        stats["snr_db"] = snr
        points.append(stats)
        log.info("snr_db=%g churn_frac=%.4f [%.4f, %.4f] mean_jump=%.2f",
                 snr, stats["churn_frac"]["value"], stats["churn_frac"]["lo"],
                 stats["churn_frac"]["hi"], stats["mean_abs_id_jump_when_changed"])

    if not base_cfg.run.keep_models_loaded:
        hub.release(model_name)

    sweep_path = Path(args.sweep_json) if args.sweep_json else run_dir.parent / "noise_snr_sweep.json"
    rho_by_snr = {}
    if sweep_path.exists():
        sweep = load_json(sweep_path)
        rho_by_snr = {p["snr_db"]: p["agreement_rho"]["value"] for p in sweep["points"]}
    else:
        log.warning("no existing noise_snr_sweep.json found at %s; churn will be reported alone, "
                    "without the rho comparison", sweep_path)

    result = {"run": str(run_dir), "model": model_name, "n_series": int(n_rows),
             "sweep_json": str(sweep_path) if sweep_path.exists() else None, "points": points}
    out = Path(args.out) if args.out else run_dir.parent / "quantization_churn_sweep.json"
    save_json(out, result)

    print(f"\nwrote {out}\n")
    print(f"{'SNR_dB':>8}  {'churn_frac [CI]':>26}  {'mean_jump':>10}  {'cross_model_rho':>16}")
    for p in points:
        cf = p["churn_frac"]
        cf_str = f"{cf['value']:.4f} [{cf['lo']:.4f},{cf['hi']:.4f}]"
        rho = rho_by_snr.get(p["snr_db"])
        rho_str = f"{rho:+.3f}" if rho is not None else "n/a"
        print(f"{p['snr_db']:>8g}  {cf_str:>26}  {p['mean_abs_id_jump_when_changed']:>10.2f}  {rho_str:>16}")


if __name__ == "__main__":
    main()
