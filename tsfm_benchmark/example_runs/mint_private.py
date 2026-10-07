"""Mint a private split for a corpus whose private split was deliberately not built (ROADMAP sec 41).

Run this only after the hypotheses are registered; a private split that anything has looked at
is consumed (CLAUDE.md sec 6.6). It generates the epoch's private split from the same task specs
as the dev split, gates it with the unchanged leakage auditor, and refuses to write anything when
a sample hash equals one in any existing sealed split under ``--root``, which is how v2's first
private epoch was caught reusing v1's. On success it seals ``<out>/corpus.jsonl`` + ``manifest.json``
and writes ``mint_audit_counts.json`` (candidates / rejected / accepted) beside them.

v3 private split, epoch 1 (seed 7000003 keeps every v3 sample seed disjoint from v1/v2's seed-0 splits):

    cd "TSFM Interp"
    PYTHONPATH=. python3 tsfm_benchmark/example_runs/mint_private.py \
        --config tsfm_benchmark/configs/benchmark_v3.yaml --seed 7000003 --epoch 1 \
        --out benchmark_v3/private_epoch1 --references monash --reference-limit 120 --root .

Use a fresh ``--epoch`` (2, 3, ...) for every further private split; never reuse one.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline import BenchmarkBuilder, LeakageAuditor  # noqa: E402
from tsfm_benchmark.build_pipeline import seal as seal_mod  # noqa: E402
from tsfm_benchmark.build_pipeline.overlap import discover_splits, overlap_counts  # noqa: E402
from tsfm_benchmark.example_runs.run_full import load_references, load_specs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--root", default=".", help="directory whose benchmark_* corpora are checked for hash overlap")
    ap.add_argument("--references", choices=["none", "monash"], default="monash")
    ap.add_argument("--reference-limit", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--max-count", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists and is not empty; a private split is minted once")
    if args.epoch < 1:
        sys.exit("--epoch must be >= 1: epoch 0 is the dev split's epoch")

    specs = load_specs(args.config, args.max_count)
    with open(args.config, encoding="utf-8") as fh:
        assign_roles = bool(yaml.safe_load(fh).get("assign_roles", False))
    auditor = LeakageAuditor(metric="dtw", threshold=args.threshold)
    load_references(auditor, args.references, args.reference_limit)
    builder = BenchmarkBuilder(auditor=auditor, assign_roles=assign_roles)
    _, private_base = builder._epoch_seeds(args.seed, args.epoch)
    private, rejected = builder._build_split(specs, private_base, args.epoch)

    hashes = {s.content_hash() for s in private}
    counts = overlap_counts(hashes, discover_splits(args.root))
    shared = {k: v for k, v in counts.items() if v}
    if shared:
        sys.exit(f"REFUSING to seal: {sum(shared.values())} sample hashes already appear in {shared}")

    seal_mod.seal_corpus(private, str(out), args.epoch, "private",
                         extra={"note": "minted after registration; held-out, do not publish",
                                "seed": args.seed, "overlap_checked_against": sorted(counts)})
    audit = {"candidates": len(private) + len(rejected), "rejected": len(rejected), "accepted": len(private),
             "overlap_checked_against": counts}
    (out / "mint_audit_counts.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
