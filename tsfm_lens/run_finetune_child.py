"""CLI for `tsfm_lens/experiments/finetune_child.py` (`ROADMAP.md` sec 23.3
D2 / sec 6.3.1 Option E): fine-tune `amazon/chronos-t5-small` on two
disjoint splits of a sealed corpus, producing two children
(`child_light`/`child_drifted`) that share initialization and nothing
else -- the known-lineage positive pair provenance detection (sec 6.3.1
Option C) has never had.

Example:
    python run_finetune_child.py --corpus ../benchmark_medium/public_dev \\
        --out runs/lineage_pair --light-steps 50 --drifted-steps 400
"""

from tsfm_lens.experiments.finetune_child import _cli

if __name__ == "__main__":
    _cli()
