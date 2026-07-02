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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline import BenchmarkBuilder, LeakageAuditor, TaskSpec
from tsfm_benchmark.build_pipeline.sources import load_sources

_NEEDS_SOURCES = {"mixture", "block_bootstrap", "sequential_par"}


def load_specs(
    path,
    max_count=None,
    source_kind: str | None = None,
    source_limit: int = 100,
    source_subset: str | None = None,
    source_n_domains: int = 6,
) -> list[TaskSpec]:
    """Turn each YAML task entry into a TaskSpec, injecting source series for real-derived tasks when requested.

    When ``source_subset`` is left unset, a task that needs the global
    ``--sources`` injection bootstraps a random sample of ``source_n_domains``
    domains from the whole catalog (see ``sources.bootstrap_catalog``) instead
    of defaulting to one hardcoded domain, so a bare ``--sources monash`` is
    already reasonably domain-diverse without the caller naming anything.

    A task with ``source_sample_size`` set is left with its ``source_config``
    intact rather than having a resolved ``sources`` list baked into
    ``generator_params`` here: ``BenchmarkBuilder`` loads (and caches) that
    pool itself and draws a fresh random subset per repeat, so pre-resolving
    it to one fixed list here would silently defeat the per-sample variety.
    """
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    specs: list[TaskSpec] = []
    for entry in cfg["tasks"]:
        if max_count:
            entry = {**entry, "count": min(entry["count"], max_count)}

        spec = TaskSpec(**entry)
        if spec.generator in _NEEDS_SOURCES:
            if not spec.source_config and source_kind is None:
                print(f"skipping real-derived task '{spec.name}' (generator '{spec.generator}' needs injected sources)")
                continue

            if not spec.source_config:
                spec.source_config = {"kind": source_kind, "n_domains": source_n_domains} if source_subset is None else {"kind": source_kind, "subset": source_subset}

            if spec.source_sample_size:
                specs.append(spec)
                continue

            config = spec.source_config
            effective_limit = config.get("limit", source_limit)

            try:
                sources = load_sources(config, limit=effective_limit)
            except Exception as ex:
                print(f"could not load sources for '{spec.name}' via '{config.get('kind', config)}' ({ex}); skipping task")
                continue

            if not sources:
                print(f"no sources loaded for '{spec.name}' via '{config.get('kind', config)}'; skipping task")
                continue

            spec.generator_params = {**spec.generator_params, "sources": sources}

        specs.append(spec)
    return specs


def load_references(auditor, kind, limit, subset: str | None = None, n_domains: int = 6):
    """Populate the auditor's reference corpus, or warn that the gate is a no-op."""
    if kind == "monash":
        try:
            from tsfm_benchmark.build_pipeline.sources import bootstrap_catalog, monash
        except Exception as ex:
            print(f"could not load Monash references ({ex}); gate will pass trivially")
            return
        items = monash(subset=subset, limit=limit) if subset else bootstrap_catalog(n_domains=n_domains, total_limit=limit)
        n = 0
        for ref, values in items:
            auditor.add_reference(ref.corpus, ref.item_id, values)
            n += 1
        where = f"domain '{subset}'" if subset else f"{n_domains} bootstrapped domains"
        print(f"loaded {n} real reference series from Monash ({where})")
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
    ap.add_argument("--source-subset", default=None, help="pin one Monash domain for --sources/--references; omit to bootstrap a random sample across the whole catalog")
    ap.add_argument("--source-n-domains", type=int, default=6, help="number of Monash domains to bootstrap across when --source-subset is not set")
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
        source_n_domains=args.source_n_domains,
    )
    if not specs:
        print("no runnable tasks in config")
        sys.exit(1)

    auditor = LeakageAuditor(metric="dtw", threshold=args.threshold)
    load_references(auditor, args.references, args.reference_limit, subset=args.source_subset, n_domains=args.source_n_domains)

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
