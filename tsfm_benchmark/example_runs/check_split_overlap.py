"""Report shared sample hashes between one sealed split and every other sealed split under a root.

    PYTHONPATH=. python3 tsfm_benchmark/example_runs/check_split_overlap.py \
        --split benchmark_v3/public_dev --root . --out overlap.json

Exits 1 when any other split shares a hash with ``--split``.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.overlap import discover_splits, manifest_hashes, overlap_counts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--pattern", default="benchmark_*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    target = Path(args.split).resolve()
    others = [p for p in discover_splits(args.root, args.pattern) if p.resolve() != target]
    counts = overlap_counts(manifest_hashes(target), others)
    result = {"split": str(target), "n_samples": len(manifest_hashes(target)), "shared_hashes": counts,
              "total_shared": sum(counts.values())}
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    sys.exit(1 if result["total_shared"] else 0)


if __name__ == "__main__":
    main()
