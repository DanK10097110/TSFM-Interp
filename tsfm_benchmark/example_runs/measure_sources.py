"""Measure candidate ``chronos_datasets`` subsets as real-derived sources (ROADMAP sec 41, V3-A).

For each subset, load a seeded row sample through ``sources._load_chronos_datasets_config`` (the
loader the builder itself uses), cut one window per surviving series, and report the statistics
the flatness investigation (BM-06) used to decide which sources are predictable enough: median
zero fraction, median lag-1 autocorrelation, fraction periodic, and the naive-flat proxy
(lag-1 < 0.2 and not periodic). "Periodic" here is `stats.dominant_period`'s autocorrelation
peak (min_lag 4) with autocorrelation at that lag >= 0.3; BM-06's own periodicity cutoff is not
recorded in the repo, so this definition is stated rather than inherited.

    python tsfm_benchmark/example_runs/measure_sources.py --subsets ercot,solar_1h --out stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.sources import _load_chronos_datasets_config  # noqa: E402


def _acf(x: np.ndarray, lag: int) -> float:
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 1e-12 or lag >= len(x):
        return 0.0
    return float((xc[:-lag] * xc[lag:]).sum() / denom)


def window_stats(x: np.ndarray, min_period_lag: int = 4, periodic_acf: float = 0.3) -> dict:
    """Zero fraction, lag-1, periodicity and the naive-flat proxy for one window."""
    xc = x - x.mean()
    ac = np.correlate(xc, xc, mode="full")[len(xc) - 1:]
    ac = ac / (ac[0] + 1e-12)
    hi = max(min_period_lag + 1, len(xc) // 2)
    period = min_period_lag + int(np.argmax(ac[min_period_lag:hi]))
    lag1 = _acf(x, 1)
    periodic = bool(ac[period] >= periodic_acf)
    return {"zero_frac": float((x == 0).mean()), "lag1": lag1, "periodic": periodic,
            "flat_proxy": bool(lag1 < 0.2 and not periodic), "constant": bool(np.ptp(x) < 1e-9)}


def measure_subset(subset: str, limit: int, window: int, min_length: int, seed: int,
                   windows_per_row: int = 1, finite_windows: bool = False, field_name: str = "target") -> dict:
    """Statistics for one subset over a seeded sample of at most ``limit`` rows."""
    items = _load_chronos_datasets_config(
        dataset_name="autogluon/chronos_datasets", subset=subset, field_name=field_name,
        split="train", limit=limit, seed=seed, max_length=window, license="unknown",
        windows_per_row=windows_per_row, finite_windows=finite_windows)
    from datasets import load_dataset
    lengths = np.array([len(x) for x in load_dataset("autogluon/chronos_datasets", subset, split="train")[field_name]])
    kept = [v for _, v in items if len(v) >= min_length]
    rows = [window_stats(v[:window]) for v in kept]
    out = {"subset": subset, "n_rows": int(len(lengths)), "n_rows_ge_min_length": int((lengths >= min_length).sum()),
           "median_row_length": float(np.median(lengths)), "n_loaded": len(items), "n_passing_min_length": len(rows)}
    if rows:
        out.update(
            median_length=float(np.median([len(v) for v in kept])),
            zero_frac_median=float(np.median([r["zero_frac"] for r in rows])),
            lag1_median=float(np.median([r["lag1"] for r in rows])),
            frac_periodic=float(np.mean([r["periodic"] for r in rows])),
            frac_flat_proxy=float(np.mean([r["flat_proxy"] for r in rows])),
            frac_lag1_below_0p2=float(np.mean([r["lag1"] < 0.2 for r in rows])),
            frac_constant=float(np.mean([r["constant"] for r in rows])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--window", type=int, default=576)
    ap.add_argument("--min-length", type=int, default=576)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows-per-row", type=int, default=1)
    ap.add_argument("--finite-windows", action="store_true")
    ap.add_argument("--field-name", default="target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = []
    for subset in args.subsets.split(","):
        try:
            r = measure_subset(subset, args.limit, args.window, args.min_length, args.seed,
                               args.windows_per_row, args.finite_windows, args.field_name)
        except Exception as ex:
            r = {"subset": subset, "error": f"{type(ex).__name__}: {ex}"}
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
