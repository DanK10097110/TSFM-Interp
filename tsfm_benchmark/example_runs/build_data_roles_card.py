"""Write the corpus data-roles card (data_roles.json + data_roles.md) from sealed splits.

    PYTHONPATH=. python3 tsfm_benchmark/example_runs/build_data_roles_card.py \
        --split dev=benchmark_v3/public_dev --split external_real=benchmark_v3/external_real \
        --out benchmark_v3
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline.data_roles import write_roles_card  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", action="append", required=True, help="NAME=DIRECTORY, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    splits = dict(s.split("=", 1) for s in args.split)
    card = write_roles_card(splits, args.out)
    for role, block in card["roles"].items():
        print(f"{role}: {block['count']}")


if __name__ == "__main__":
    main()
