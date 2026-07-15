"""Run the validation report pipeline on a benchmark.

Validates a sealed corpus directory produced by the generation pipeline
(reads corpus.jsonl directly, no dependency on the generator package). By
default this looks for ./benchmark_out/public_dev -- the default --out of
run_full.py -- so running this with no arguments after a default run_full.py
build validates the real thing. It never silently substitutes fabricated
data: if no corpus is found and --demo isn't passed, it exits with an error
rather than falling back to the demo, since that's what happens when
"why does my report only show a couple hundred points" actually means
"this validated the demo, not your benchmark."

Runs algorithmic matching (DTW by default -- see benchmark_validation.matching
for why), catch22 features, a 3D UMAP embedding, corpus-wide and per-group
diversity metrics, and per-group redundancy, then writes a JSON report plus
ten standalone interactive HTML plots: the 3D feature embedding, corpus
composition by task/tier, real-domain composition (when the corpus has
real-derived provenance), example raw sequences per generator, raw
length/scale distributions, the catch22 feature-variance ranking, the raw
series behind the most extreme feature-value outliers, the pairwise-redundancy
histogram, the top redundant pairs themselves (overlaid with full creation
provenance), and per-group diversity comparison.

    # validate a real sealed corpus (default --corpus is ./benchmark_out/public_dev)
    PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_validation.py --out outputs

    # validate a specific corpus directory
    PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_validation.py --corpus /path/to/public_dev --out outputs

    # standalone demo, explicit opt-in (never runs by accident)
    PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_validation.py --demo
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tsfm_benchmark.benchmark_validation as bv
from tsfm_benchmark.logging_utils import StepTimer, setup_logging

DEFAULT_CORPUS = "./benchmark_out/public_dev"
LOG_DIR = REPO_ROOT / "logs"

logger = logging.getLogger("tsfm_benchmark.example_runs.run_validation")


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
    ap.add_argument("--corpus", default=None, help=f"sealed corpus directory to validate (default: {DEFAULT_CORPUS} if it exists)")
    ap.add_argument("--demo", action="store_true", help="run the standalone demo benchmark instead of a real corpus (never the default)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--method", choices=["xcorr", "dtw"], default="dtw", help="pairwise matcher: dtw (default, tolerates local time warping, batched via dtaidistance) or the cheaper shift-only xcorr")
    ap.add_argument("--catch24", action="store_true", help="use catch24 (adds mean and std)")
    ap.add_argument("--redundancy-threshold", type=float, default=0.97)
    ap.add_argument("--max-sequences", type=int, default=None, help="subsample for the O(n^2) matcher (recommended above a few thousand sequences)")
    ap.add_argument("--group-by", default="task", choices=["task", "tier", "group", "archetype"], help="grouping key for per-group diversity/example-sequence breakdowns")
    ap.add_argument("--debug", action="store_true", help=f"write comprehensive DEBUG-level logs to a timestamped file under {LOG_DIR}")
    args = ap.parse_args()

    log_path = setup_logging(args.debug, LOG_DIR, "run_validation")
    if log_path:
        logger.info("debug logging enabled, writing full log to %s", log_path)

    os.makedirs(args.out, exist_ok=True)
    run_start = time.perf_counter()

    if args.demo:
        with StepTimer(logger, "building DEMO benchmark"):
            records = make_demo_records()
        logger.info("loaded %d DEMO sequences (incl. 5 planted near-duplicates) -- this is NOT a real benchmark", len(records))
    else:
        corpus = args.corpus or DEFAULT_CORPUS
        if not os.path.isdir(corpus) or not os.path.isfile(os.path.join(corpus, "corpus.jsonl")):
            logger.error("no sealed corpus found at '%s' (expected a corpus.jsonl there).", corpus)
            logger.error("Build one first, e.g.:")
            logger.error("    PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_full.py --config configs/example.yaml")
            logger.error("or pass --corpus /path/to/public_dev, or --demo to run the standalone demo instead.")
            sys.exit(1)
        with StepTimer(logger, f"loading sealed corpus from {corpus}"):
            records = bv.from_sealed(corpus)
        logger.info("loaded %d sequences from %s", len(records), corpus)

    with StepTimer(logger, f"matching all pairs (method={args.method})"):
        match = bv.match_all(records, method=args.method, redundancy_threshold=args.redundancy_threshold, max_sequences=args.max_sequences)
    with StepTimer(logger, f"extracting catch{'24' if args.catch24 else '22'} features"):
        fm = bv.extract_features(records, catch24=args.catch24)
    with StepTimer(logger, "computing 3D embedding"):
        coords, embed_method = bv.embed_3d(fm.features_scaled, method="umap", seed=0)
    with StepTimer(logger, "computing corpus-wide diversity metrics"):
        diversity = bv.diversity_metrics(fm)

    group_labels = [getattr(r, args.group_by) for r in records]
    with StepTimer(logger, f"computing per-group diversity metrics (group-by={args.group_by})"):
        by_group = bv.diversity_metrics_by_group(fm, group_labels)
    if not by_group:
        logger.warning("no group had enough sequences to report per-group diversity for --group-by %s", args.group_by)

    id_to_tier = {r.seq_id: r.tier for r in records}
    if len(set(id_to_tier.values())) > 1:
        with StepTimer(logger, "splitting redundancy by tier"):
            redundancy_split = bv.redundancy_by_group(match, id_to_tier)
    else:
        redundancy_split = None

    with StepTimer(logger, "building report"):
        report = bv.build_report(
            match, fm, diversity, embed_method,
            records=records,
            diversity_by_group=by_group,
            diversity_by_group_key=args.group_by,
            redundancy_by_group_result=redundancy_split,
        )
    json_path = bv.save_report(report, os.path.join(args.out, "validation_report.json"))
    logger.info("wrote %s", json_path)
    bv.print_summary(report)

    with StepTimer(logger, "rendering plots"):
        written = [json_path]
        written.append(bv.plot_embedding(coords, fm, embed_method, os.path.join(args.out, "feature_space_3d.html")))
        written.append(bv.plot_group_composition(records, os.path.join(args.out, "composition.html")))
        written.append(bv.plot_example_sequences(records, os.path.join(args.out, "example_sequences.html"), key=args.group_by))
        written.append(bv.plot_value_length_distribution(records, os.path.join(args.out, "length_scale_distribution.html")))
        written.append(bv.plot_feature_variance(diversity, os.path.join(args.out, "feature_variance.html")))
        written.append(bv.plot_feature_anomalies(fm, records, os.path.join(args.out, "feature_anomalies.html")))
        written.append(bv.plot_redundancy_histogram(match, os.path.join(args.out, "redundancy_histogram.html")))
        written.append(bv.plot_top_redundant_pairs(match, records, os.path.join(args.out, "top_redundant_pairs.html")))
        if by_group:
            written.append(bv.plot_diversity_by_group(by_group, os.path.join(args.out, "diversity_by_group.html")))
        domain_path = bv.plot_domain_composition(records, os.path.join(args.out, "domain_composition.html"))
        if domain_path:
            written.append(domain_path)
        else:
            logger.info("no real-derived domain provenance found; skipped domain_composition.html")

    for path in written:
        logger.info("wrote %s", path)
    logger.info("run_validation complete in %.2fs", time.perf_counter() - run_start)


if __name__ == "__main__":
    main()
