"""Phase 2c (ROADMAP.md §6.3): can L1 CKA / L2's stitching-gain-over-baseline
distinguish a model pair with **documented shared pretraining lineage**
(e.g. two sizes of the same released family) from an **independently-
trained** pair with no such lineage?

Reuses `null_baseline.py`'s bootstrap-difference-of-each-run's-own-best-pair
machinery verbatim (`compare_l1_peak_cka`, `compare_l2_best_gain`,
`run_null_baseline_comparison`) — nothing in that machinery is specific to
"trained vs. untrained"; it already just bootstraps the difference between
two runs' own best-achievable L1/L2 signal, using a paired (tighter)
bootstrap when the two runs loaded the same corpus in the same row order.
This module exists only to give the *lineage* question its own named entry
point rather than overloading `null_baseline`'s "real vs. untrained-weights
null" framing with an unrelated use case.

`compare_lineage_signal(positive_run, negative_run, ...)`: `positive_run`'s
own L1/L2 numbers become the bootstrap's `a` side, `negative_run`'s become
`b` — so `diff_lo > 0` means the positive (shared-lineage) pair's signal
significantly *exceeds* the negative (independent) pair's, which is the
direction this method would need to show to be useful evidence of lineage.
"""

from __future__ import annotations

from pathlib import Path

from .null_baseline import run_null_baseline_comparison


def compare_lineage_signal(positive_run: Path, negative_run: Path, n_boot: int = 500,
                          seed: int = 0, ci: float = 0.95) -> dict:
    """`positive_run`: a two-model run pairing checkpoints with documented
    shared pretraining lineage. `negative_run`: a two-model run pairing
    independently-trained checkpoints (no shared lineage). Both must already
    have `l1`/`l2` artifacts (nothing is re-run). Returns the same
    `{"l1_peak_cka", "l2_best_gain"}` shape as `run_null_baseline_comparison`,
    each an `a`-vs-`b` bootstrap-difference dict with `a` = positive,
    `b` = negative.
    """
    return run_null_baseline_comparison(positive_run, negative_run, n_boot, seed, ci)
