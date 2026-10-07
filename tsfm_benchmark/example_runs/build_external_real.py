"""Build and seal the ``external_real`` split from raw GIFT-Eval windows (ROADMAP sec 41, V3-A).

Reads the ``external_real:`` section of a corpus config (see ``configs/benchmark_v3.yaml``),
downloads only the listed GIFT-Eval datasets' Arrow files, selects domain-stratified windows by
sha256 rank, gates them against the dev split (and optionally Monash) with the unchanged
``LeakageAuditor``, and seals the survivors as their own split:

    PYTHONPATH=. python3 tsfm_benchmark/example_runs/build_external_real.py \
        --config tsfm_benchmark/configs/benchmark_v3.yaml --dev benchmark_v3/public_dev \
        --out benchmark_v3/external_real --references monash --reference-limit 120
"""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline import LeakageAuditor, load_sealed  # noqa: E402
from tsfm_benchmark.build_pipeline import seal as seal_mod  # noqa: E402
from tsfm_benchmark.build_pipeline.external import ReferenceGate, build_external_samples  # noqa: E402
from tsfm_benchmark.example_runs.run_full import load_references  # noqa: E402


def _hub_revision(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dev", required=True, help="sealed dev split the windows are audited against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--references", choices=["none", "monash"], default="monash")
    ap.add_argument("--reference-limit", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already exists and is not empty")

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)["external_real"]
    dev, dev_manifest = load_sealed(args.dev, verify=True)
    auditor = LeakageAuditor(metric="dtw", threshold=args.threshold)
    load_references(auditor, args.references, args.reference_limit)
    gate = ReferenceGate(dev, auditor, reference_corpus="benchmark_v3/public_dev")
    kept, selection = build_external_samples(cfg, gate=gate)
    audit = gate.block(kept)
    print(f"selected {len(kept)} windows from {len(selection['per_dataset'])} datasets; gate rejected "
          f"{audit['n_rejected']} of {audit['n_candidates_scored']} candidate windows; "
          f"{audit['n_near_duplicate_pairs_vs_reference']} near-duplicate pairs vs dev")

    selection["source_revision"] = _hub_revision(selection["repo"])
    selection["dev_global_digest"] = dev_manifest["global_digest"]
    seal_mod.seal_corpus(kept, str(out), 0, "external_real",
                         extra={"audit": audit, "selection": selection,
                                "note": "raw external windows; no generator; never used for discovery, "
                                        "registration or SAE training"})
    print(f"sealed {len(kept)} windows -> {out}")


if __name__ == "__main__":
    main()
