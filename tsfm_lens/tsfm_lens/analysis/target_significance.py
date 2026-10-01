"""Per-target significance of the ablation battery's clears (`ROADMAP.md`
sec 38.3.4, decision of 2026-10-01).

The battery clears a cell (feature x channel) when its effect beats the
random-direction null's p95, so a target's clears are only evidence of causal
features when they exceed what its own nulls would clear by chance. The
nominal 0.05 is not that chance: the leave-one-draw-out rate each artifact
records under `empirical_chance` is 0.003-0.04 per cell on the 7-model run,
and the battery's log line, which printed 0.05 x cells, read the whole run as
0.96x chance when it was 2.4727x (sec 38.3.4).

This module is the one place the per-target test lives, so `register`,
the report and `confirm` read the same answer. For each measured target:
`c` = the artifact's `n_clearing_cells`, `n` and `r` = `empirical_chance`
`n_cells` and `rate`, `p = binom.sf(c - 1, n, r)` (the chance of at least `c`
clears among `n` cells at the empirical rate), then Benjamini-Hochberg over
ALL measured targets of the run at `q` (`scipy.stats.false_discovery_control`).

Evidence class: descriptive screen on dev data. A binomial treats cells as
independent although a feature's channels are correlated, so the p-values are
a screen for which targets deserve claims, not a calibrated tail probability.

Refusal, not fallback: a target without an `empirical_chance` rate (an
artifact written with `sae.ablation_empirical_chance` off) cannot be tested
against its own chance, and substituting the nominal 0.05 would reproduce the
exact mis-reading this gate exists to prevent, so `TargetChanceMissing` is
raised naming every such target.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom, false_discovery_control

TARGET_SIGNIFICANCE_Q = 0.05


class TargetChanceMissing(RuntimeError):
    """A measured target has no usable `empirical_chance` block (or no
    `n_clearing_cells`), so its clears cannot be tested against its own
    chance."""


def target_significance(targets, q: float = TARGET_SIGNIFICANCE_Q) -> dict:
    """Per-target binomial test against the empirical chance rate, BH over
    the targets.

    `targets` is an iterable of `(model, layer, artifact dict)` for every
    MEASURED target (withheld and skipped targets have no clears and are not
    in the family). Returns `{"q", "method", "n_targets", "n_significant",
    "targets": {"model/layer": {"model", "layer", "clearing_cells",
    "n_cells", "rate", "expected_cells", "p", "q_value", "significant"}}}`.
    Raises `TargetChanceMissing` listing every target whose artifact lacks
    `empirical_chance` (`rate` null or `n_cells` 0 included) or
    `n_clearing_cells`; it never falls back to the nominal 0.05.
    """
    rows, missing = [], []
    for model, layer, art in targets:
        block = art.get("empirical_chance") if isinstance(art, dict) else None
        rate = (block or {}).get("rate") if isinstance(block, dict) else None
        n_cells = (block or {}).get("n_cells") if isinstance(block, dict) else None
        clears = art.get("n_clearing_cells") if isinstance(art, dict) else None
        if rate is None or not n_cells or clears is None:
            missing.append(f"{model}/{layer}")
            continue
        rows.append((str(model), str(layer), int(clears), int(n_cells), float(rate)))
    if missing:
        raise TargetChanceMissing(
            f"the per-target significance gate needs each target's `empirical_chance` "
            f"block and `n_clearing_cells`, and {len(missing)} measured target(s) lack "
            f"them: {', '.join(sorted(missing))}. Rerun their ablation with "
            f"`sae.ablation_empirical_chance: true`; the nominal 0.05 is never used "
            f"in its place (ROADMAP.md sec 38.3.4).")
    if not rows:
        return {"q": float(q), "method": "binomial_vs_empirical_rate+BH", "n_targets": 0,
                "n_significant": 0, "targets": {}}
    p = np.array([float(binom.sf(c - 1, n, r)) for _m, _l, c, n, r in rows])
    qv = np.asarray(false_discovery_control(p, method="bh"), dtype=np.float64)
    out = {}
    for (model, layer, c, n, r), pv, qq in zip(rows, p, qv):
        out[f"{model}/{layer}"] = {
            "model": model, "layer": layer, "clearing_cells": c, "n_cells": n,
            "rate": r, "expected_cells": n * r, "p": float(pv), "q_value": float(qq),
            "significant": bool(qq <= q)}
    return {"q": float(q), "method": "binomial_vs_empirical_rate+BH",
            "n_targets": len(out),
            "n_significant": sum(1 for v in out.values() if v["significant"]),
            "targets": out}
