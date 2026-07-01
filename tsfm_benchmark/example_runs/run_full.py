"""Config-driven build of the leakage-safe synthetic benchmark.

Reads a YAML config of task specs, optionally loads a real reference corpus for
the leakage gate, builds one epoch, and seals the public and private splits to
disk. Real-derived tasks (mixture, bootstrap, PAR) can be enabled with
``--sources`` and will receive injected source series from a real corpus such as
Monash. Run from the package root:

    PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_full.py --config configs/example.yaml --out ./benchmark_out
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline import BenchmarkBuilder, LeakageAuditor, TaskSpec

_NEEDS_SOURCES = {"mixture", "block_bootstrap", "sequential_par"}


def load_specs(path, max_count=None, source_kind: str | None = None, source_limit: int = 100, source_subset: str = "tourism_monthly") -> list[TaskSpec]:
    """Turn each YAML task entry into a TaskSpec, injecting source series for real-derived tasks when requested."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    specs: list[TaskSpec] = []
    for entry in cfg["tasks"]:
        if max_count:
            entry = {**entry, "count": min(entry["count"], max_count)}

        spec = TaskSpec(**entry)
        if spec.generator in _NEEDS_SOURCES:
            if spec.source_config:
                sources = load_sources(spec.source_config, limit=source_limit)
            elif source_kind is None:
                print(f"skipping real-derived task '{spec.name}' (generator '{spec.generator}' needs injected sources)")
                continue
            else:
                sources = load_sources({"kind": source_kind, "dataset": source_kind, "subset": source_subset}, limit=source_limit)

            if not sources:
                print(f"no sources loaded for '{spec.name}' via '{source_kind or spec.source_config.get('kind', 'configured source')}'; skipping task")
                continue

            spec.generator_params = {**spec.generator_params, "sources": sources}

        specs.append(spec)
    return specs


def load_sources(kind: str, limit: int, subset: str = "tourism_monthly") -> list[tuple[Any, Any]]:
    """Load a pool of real source series for a real-derived generator."""
    if kind == "monash":
        try:
            from tsfm_benchmark.build_pipeline.sources import monash
        except Exception as ex:
            raise RuntimeError(f"could not load Monash sources ({ex})") from ex
        return list(monash(subset=subset, limit=limit))
    raise ValueError(f"unknown source kind '{kind}'")


def load_references(auditor, kind, limit, subset: str = "tourism_monthly"):
    """Populate the auditor's reference corpus, or warn that the gate is a no-op."""
    if kind == "monash":
        try:
            from tsfm_benchmark.build_pipeline.sources import monash
        except Exception as ex:
            print(f"could not load Monash references ({ex}); gate will pass trivially")
            return
        n = 0
        for ref, values in monash(subset=subset, limit=limit):
            auditor.add_reference(ref.corpus, ref.item_id, values)
            n += 1
        print(f"loaded {n} real reference series from Monash")
    else:
        print("no reference corpus loaded; leakage gate passes trivially (distance = inf)")


def main():
    ap = argparse.ArgumentParser(description="Build and seal a synthetic benchmark from a YAML config.")
    ap.add_argument("--config", default="configs/example.yaml")
    ap.add_argument("--out", default="./benchmark_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--references", choices=["none", "monash"], default="none")
    ap.add_argument("--reference-limit", type=int, default=100)
    ap.add_argument("--sources", choices=["none", "monash"], default="none", help="inject real source series into source-dependent generators")
    ap.add_argument("--source-limit", type=int, default=100, help="max number of source series to load for injected generators")
    ap.add_argument("--source-subset", default="tourism_monthly", help="Monash subset to load when --sources monash is used")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--max-count", type=int, default=None, help="cap per-task count for a quick run")
    args = ap.parse_args()

    source_kind = None if args.sources == "none" else args.sources
    specs = load_specs(
        args.config,
        args.max_count,
        source_kind=source_kind,
        source_limit=args.source_limit,
        source_subset=args.source_subset,
    )
    if not specs:
        print("no runnable tasks in config")
        sys.exit(1)

    auditor = LeakageAuditor(metric="dtw", threshold=args.threshold)
    load_references(auditor, args.references, args.reference_limit, subset=args.source_subset)

    builder = BenchmarkBuilder(auditor=auditor)
    pub = os.path.join(args.out, "public_dev")
    priv = os.path.join(args.out, "private_test")
    result = builder.build_and_seal(specs, public_dir=pub, private_dir=priv, seed=args.seed, epoch=args.epoch)

    print(f"\nepoch {result.epoch}")
    print(f"public_dev   : {len(result.public_dev)} sequences -> {pub}")
    print(f"private_test : {len(result.private_test)} sequences -> {priv}")
    print(f"rejected gate: {len(result.rejected)}")
    print(f"near-dupes   : {len(result.duplicates)}")
    print("files/split  : corpus.jsonl, manifest.json")


if __name__ == "__main__":
    main()
