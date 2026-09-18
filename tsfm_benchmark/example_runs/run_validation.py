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
import json
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
    ap.add_argument("--blocked", action="store_true", help="approximate O(n*k) matcher (sec 15 A17): bucket by shape, score within/adjacent buckets only -- for corpora too large for the exact O(n^2) pass even with --max-sequences")
    ap.add_argument("--n-blocks", type=int, default=None, help="number of shape-clusters for --blocked (default: ~sqrt(n/2))")
    ap.add_argument("--group-by", default="task", choices=["task", "tier", "group", "archetype"], help="grouping key for per-group diversity/example-sequence breakdowns")
    ap.add_argument("--split-role", default="auto", choices=["public", "private", "auto"],
                     help="ROADMAP.md sec 34 item B2.2. 'auto' (default) infers the role from the sealed "
                          "manifest's own 'visibility' field, never from the directory name. 'private' "
                          "redacts the one per-sample-identifying field this report can carry "
                          "(matching.top_redundant_pairs), skips the UMAP embedding and every per-sample "
                          "plot (example sequences, feature-anomaly gallery, top-redundant-pairs plot), "
                          "and logs loudly -- per item B2.1's argument, the corpus-construction statistics "
                          "this script computes involve no model and cannot leak model-comparative "
                          "information, but a per-sample anomaly gallery on the private split would leak "
                          "individualized detail the one-shot discipline (CLAUDE.md sec 6.7) protects.")
    ap.add_argument("--compare-splits", nargs=2, metavar=("PUBLIC_DIR", "PRIVATE_DIR"), default=None,
                     help="ROADMAP.md sec 34 items B2.3/B3: instead of validating one corpus, compare the "
                          "public and private splits against each other (cross-split near-duplicate "
                          "fraction, composition equality, feature-space geometry, and a distributional "
                          "equivalence test) and write cross_split.json. Model-free by construction -- see "
                          "benchmark_validation/cross_split.py's module docstring.")
    ap.add_argument("--persist-into-private", action="store_true",
                     help="with --compare-splits, ALSO write cross_split.json into PRIVATE_DIR itself (a "
                          "plain sibling file next to corpus.jsonl/manifest.json -- it does not touch the "
                          "sealed manifest or its global_digest, so load_sealed(verify=True) is unaffected) "
                          "so tsfm_lens's confirm stage can find and report the exchangeability verdict "
                          "(ROADMAP.md sec 34 item B3.4/B5). Off by default: mutating the private "
                          "directory, even with a file that touches no sealed content, is a deliberate act.")
    ap.add_argument("--n-perm", type=int, default=2000,
                     help="permutations for the cross-split energy-distance test (sec 6.6: 24 features "
                          "against a 1/n_perm floor needs n_perm >= ~480 at minimum; default 2000)")
    ap.add_argument("--margin-sd", type=float, default=0.2,
                     help="TOST equivalence margin in pooled-robust-SD units (a stated convention, not a "
                          "measurement -- ROADMAP.md sec 34 item B3.2)")
    ap.add_argument("--debug", action="store_true", help=f"write comprehensive DEBUG-level logs to a timestamped file under {LOG_DIR}")
    ap.add_argument("--enforce-gates", action="store_true",
                     help="exit nonzero if any diversity/redundancy gate fails (ROADMAP.md sec 16 E23). "
                          "Off by default -- the gate table always prints, but thresholds are a "
                          "first-pass calibration against one demo-mode reference point, not yet "
                          "validated enough to block a build on by default.")
    args = ap.parse_args()

    log_path = setup_logging(args.debug, LOG_DIR, "run_validation")
    if log_path:
        logger.info("debug logging enabled, writing full log to %s", log_path)

    os.makedirs(args.out, exist_ok=True)
    run_start = time.perf_counter()

    if args.compare_splits:
        public_dir, private_dir = args.compare_splits
        logger.warning("cross-split comparison: reading the PRIVATE split at %s -- this is model-free "
                        "(sec 34 item B2.1) but is still a read of the consumable private corpus; "
                        "logged loudly the way `confirm` does.", private_dir)
        for d in (public_dir, private_dir):
            if not os.path.isfile(os.path.join(d, "corpus.jsonl")):
                logger.error("no sealed corpus found at '%s' (expected a corpus.jsonl there).", d)
                sys.exit(1)
        priv_manifest = bv.read_manifest(private_dir)
        priv_role = bv.infer_split_role(priv_manifest)
        if priv_role != "private":
            logger.error("PRIVATE_DIR='%s' manifest visibility is '%s', not 'private' -- refusing to "
                          "run the cross-split comparison against a corpus that isn't actually sealed "
                          "private (ROADMAP.md sec 34 item B2.2 requires this be inferred from the "
                          "manifest, not the directory name).", private_dir, priv_role)
            sys.exit(1)
        with StepTimer(logger, f"loading public split from {public_dir}"):
            public_records = bv.from_sealed(public_dir)
        with StepTimer(logger, f"loading private split from {private_dir}"):
            private_records = bv.from_sealed(private_dir)
        logger.info("loaded %d public, %d private sequences", len(public_records), len(private_records))
        with StepTimer(logger, f"cross-split comparison (method={args.method}, blocked={args.blocked})"):
            cross_report = bv.build_cross_split_report(
                public_records, private_records, method=args.method,
                near_dup_threshold=args.redundancy_threshold, blocked=args.blocked,
                n_blocks=args.n_blocks, max_sequences=args.max_sequences, catch24=args.catch24,
                n_perm=args.n_perm, margin_sd=args.margin_sd,
            )
        # Carried alongside the report (not into build_cross_split_report's
        # own return contract, which knows nothing about manifests) so a
        # downstream reader -- `tsfm_lens/analysis/confirm.py`'s B3.4 wiring
        # -- can refuse a `cross_split.json` that doesn't match the private
        # corpus it's sitting beside, mirroring C1.3's validation_report
        # digest cross-check (`corpus_card.py::_read_validation_report`)
        # rather than trusting the sibling file by path alone.
        cross_report["private_corpus_digest"] = (priv_manifest or {}).get("global_digest")
        cross_path = bv.save_cross_split_report(cross_report, os.path.join(args.out, "cross_split.json"))
        logger.info("wrote %s", cross_path)
        bv.print_cross_split_summary(cross_report)
        if args.persist_into_private:
            priv_path = os.path.join(private_dir, "cross_split.json")
            logger.warning("--persist-into-private: writing %s (a plain sibling file; does not touch "
                            "manifest.json or its sealed global_digest)", priv_path)
            bv.save_cross_split_report(cross_report, priv_path)
        logger.info("run_validation --compare-splits complete in %.2fs", time.perf_counter() - run_start)
        return

    corpus_digest = None
    split_role = None
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
        # ROADMAP.md sec 34 item C1.3: read the sealed manifest's own
        # global_digest directly (plain json, no tsfm_benchmark import) so a
        # downstream tsfm_lens corpus card can cross-check this report against
        # the corpus it actually ran on, without this script gaining a
        # dependency on the generator package (this module's own stated
        # design principle, see the module docstring).
        manifest_path = os.path.join(corpus, "manifest.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    corpus_digest = json.load(fh).get("global_digest")
            except Exception as exc:  # noqa: BLE001 -- degrade loudly, never block validation on this
                logger.warning("could not read global_digest from %s (%s); validation_report.json "
                                "will carry corpus_digest=null", manifest_path, exc)
        else:
            logger.warning("no manifest.json at %s; validation_report.json will carry corpus_digest=null", corpus)

        manifest = bv.read_manifest(corpus)
        split_role = bv.infer_split_role(manifest, args.split_role)
        if split_role == "private":
            logger.warning("split-role=private (%s): this run is model-free (ROADMAP.md sec 34 item "
                            "B2.1) but will REDACT per-sample identifiers from validation_report.json "
                            "and skip the UMAP embedding and every per-sample plot. Logged loudly the "
                            "way `confirm` does.",
                            "inferred from manifest" if args.split_role == "auto" else "explicit --split-role")
        elif args.split_role == "auto" and split_role == "unknown":
            logger.warning("could not infer split-role from manifest at %s (missing/unrecognized "
                            "'visibility'); treating as public -- pass --split-role private explicitly "
                            "if this IS the private split.", corpus)

    with StepTimer(logger, f"matching all pairs (method={args.method})"):
        match = bv.match_all(records, method=args.method, redundancy_threshold=args.redundancy_threshold,
                             max_sequences=args.max_sequences, blocked=args.blocked, n_blocks=args.n_blocks)
    with StepTimer(logger, f"extracting catch{'24' if args.catch24 else '22'} features"):
        fm = bv.extract_features(records, catch24=args.catch24)
    if split_role == "private":
        logger.info("split-role=private: skipping UMAP embedding (a per-sample visualization -- item "
                     "B2.3's own failure-mode list rules this out for the private split explicitly)")
        coords, embed_method = None, "skipped (private split, sec 34 item B2.3)"
    else:
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
            corpus_digest=corpus_digest,
            split_role=split_role,
        )
    json_path = bv.save_report(report, os.path.join(args.out, "validation_report.json"))
    logger.info("wrote %s", json_path)
    bv.print_summary(report)

    gate_report = bv.check_diversity_gates(report)
    bv.print_gate_summary(gate_report)
    if args.enforce_gates and not gate_report["passed"]:
        logger.error("diversity gates failed with --enforce-gates set; see the gate table above")
        sys.exit(3)

    with StepTimer(logger, "rendering plots"):
        written = [json_path]
        if split_role == "private":
            logger.info("split-role=private: skipping every per-sample plot (embedding, example "
                         "sequences, feature-anomaly gallery, top-redundant-pairs) -- item B2.1's "
                         "redaction rule")
        else:
            written.append(bv.plot_embedding(coords, fm, embed_method, os.path.join(args.out, "feature_space_3d.html")))
            written.append(bv.plot_example_sequences(records, os.path.join(args.out, "example_sequences.html"), key=args.group_by))
            written.append(bv.plot_feature_anomalies(fm, records, os.path.join(args.out, "feature_anomalies.html")))
            written.append(bv.plot_top_redundant_pairs(match, records, os.path.join(args.out, "top_redundant_pairs.html")))
        written.append(bv.plot_group_composition(records, os.path.join(args.out, "composition.html")))
        written.append(bv.plot_value_length_distribution(records, os.path.join(args.out, "length_scale_distribution.html")))
        written.append(bv.plot_feature_variance(diversity, os.path.join(args.out, "feature_variance.html")))
        written.append(bv.plot_redundancy_histogram(match, os.path.join(args.out, "redundancy_histogram.html")))
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
