"""Run the validation report pipeline on a benchmark.

With --corpus, validates a sealed corpus directory produced by the generation
pipeline (reads corpus.jsonl directly, no dependency on the generator package).
Without it, builds a small demo benchmark with planted near-duplicates so the
pipeline can be exercised standalone. Either way it runs algorithmic matching,
catch22 features, a 3D UMAP embedding, and feature-space diversity metrics, then
writes a JSON report and an interactive 3D HTML plot.

    # validate a real sealed corpus
    PYTHONPATH=. python3 examples/run_validation.py --corpus /path/to/public_dev --out outputs

    # standalone demo (needs the generator package on the path)
    PYTHONPATH=../tsfm_benchmark:. python3 examples/run_validation.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tsfm_benchmark.benchmark_validation as bv


def make_demo_records():
    """Build a diverse demo benchmark plus planted duplicates (lazy generator import)."""
    from tsfm_benchmark import BenchmarkBuilder, LeakageAuditor, TaskSpec

    specs = [
        TaskSpec("pure_seasonal", "parametric", 40, {"length": 400, "seasonalities": [{"period": 24, "amplitude": 1.0}], "noise_scale": 0.1}),
        TaskSpec("multi_seasonal_trend", "parametric", 40, {"length": 400, "trend": {"order": 2, "scale": 1.0}, "seasonalities": [{"period": 24, "amplitude": 1.0}, {"period": 168, "amplitude": 0.5}], "ar_coeffs": [0.5], "noise_scale": 0.15}),
        TaskSpec("regime_switching", "parametric", 40, {"length": 400, "seasonalities": [{"period": 50, "amplitude": 1.2}], "noise_scale": 0.2, "n_changepoints": 5}),
        TaskSpec("noise_dominated", "parametric", 40, {"length": 400, "ar_coeffs": [0.9], "noise_scale": 1.0}),
        TaskSpec("spiky_anomalous", "parametric", 40, {"length": 400, "seasonalities": [{"period": 12, "amplitude": 0.6}], "noise_scale": 0.1, "n_anomalies": 12, "anomaly_magnitude": 8.0}),
    ]
    builder = BenchmarkBuilder(auditor=LeakageAuditor(metric="dtw", threshold=0.0))
    result = builder.build(specs, seed=0, epoch=0)
    records = bv.from_samples(result.public_dev)

    base = records[0].values
    rng = np.random.default_rng(1)
    for k in range(5):
        dup = base + rng.normal(0, 0.01 * base.std(), size=len(base))
        records.append(bv.loaders.SeqRecord(f"planted_dup_{k}", dup, "planted_dup"))
    return records


def main():
    ap = argparse.ArgumentParser(description="Validate a benchmark for redundancy and diversity.")
    ap.add_argument("--corpus", default=None, help="sealed corpus directory to validate; omit for the demo")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--method", choices=["xcorr", "dtw"], default="xcorr")
    ap.add_argument("--catch24", action="store_true", help="use catch24 (adds mean and std)")
    ap.add_argument("--redundancy-threshold", type=float, default=0.97)
    ap.add_argument("--max-sequences", type=int, default=None, help="subsample for the O(n^2) matcher")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.corpus:
        records = bv.from_sealed(args.corpus)
        print(f"loaded {len(records)} sequences from {args.corpus}")
    else:
        records = make_demo_records()
        print(f"loaded {len(records)} demo sequences (incl. 5 planted near-duplicates)")

    match = bv.match_all(records, method=args.method, redundancy_threshold=args.redundancy_threshold, max_sequences=args.max_sequences)
    fm = bv.extract_features(records, catch24=args.catch24)
    coords, embed_method = bv.embed_3d(fm.features_scaled, method="umap", seed=0)
    diversity = bv.diversity_metrics(fm)

    report = bv.build_report(match, fm, diversity, embed_method)
    json_path = bv.save_report(report, os.path.join(args.out, "validation_report.json"))
    html_path = bv.plot_embedding(coords, fm, embed_method, os.path.join(args.out, "feature_space_3d.html"))

    bv.print_summary(report)
    print(f"\nwrote {json_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
