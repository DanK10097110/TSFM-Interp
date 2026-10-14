"""Feature-level chance test of a finished ablation battery, read from its artifacts.

A feature is scored on nine forecast channels, so "N features clear at least one
channel" needs a feature-level chance count. The battery records three per feature
(`sae/response.py::feature_chance_record`): a union bound, an independent-channel
product, and a draw-level rate that scores each null draw as a pseudo-feature on all
channels jointly, leave-one-draw-out. The channels move together under one random
direction, so the independent product overstates chance; the draw-level rate keeps
that dependence. This driver sums the per-feature rates per null mode and tests the
observed count against the Poisson-binomial distribution they define (upper tail for
the draw-level rate, both tails for the independent product). It reads
`sae/<model>/<layer>_ablation.json` only and writes nothing.

    python run_feature_chance_test.py runs/panel7_v3_dev runs/panel7_v3_sec42
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tsfm_lens.analysis.family_claims import candidate_for_null  # noqa: E402


def poisson_binomial_tails(ps: list, k: int) -> tuple[float, float]:
    """`(P(X >= k), P(X <= k))` for a sum of independent Bernoulli(`ps`)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return float(dist[k:].sum()), float(dist[:k + 1].sum())


def feature_chance_test(run_dir: Path) -> dict:
    """Per null mode: scorable features, observed clears, the three expected counts and the tails."""
    arts = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(run_dir, "sae").glob("*/*_ablation.json"))]
    arts = [a for a in arts if not a.get("skipped") and not a.get("withheld") and "candidates" in a]
    nulls = set()
    for a in arts:
        nulls |= {a.get("ablation_null", "mean_magnitude"), *(a.get("ablation_nulls") or [])}
    out = {}
    for null in sorted(nulls):
        obs, rates = 0, {"p_draw_level": [], "p_independent": [], "p_union_bound": []}
        for a in arts:
            for c in a["candidates"]:
                v = candidate_for_null(c, null, a.get("ablation_null", "mean_magnitude"))
                fc = v.get("feature_chance") or {}
                if not v.get("scorable") or fc.get("p_draw_level") is None:
                    continue
                obs += int((v.get("n_channels_clearing") or 0) >= 1)
                for key in rates:
                    rates[key].append(float(fc[key]))
        if not rates["p_draw_level"]:
            out[null] = {"measured": False, "reason": "no feature_chance records"}
            continue
        upper, _ = poisson_binomial_tails(rates["p_draw_level"], obs)
        ind_upper, ind_lower = poisson_binomial_tails(rates["p_independent"], obs)
        out[null] = {"measured": True, "n_scorable": len(rates["p_draw_level"]), "observed": obs,
                     "expected_draw_level": sum(rates["p_draw_level"]),
                     "ratio_draw_level": obs / sum(rates["p_draw_level"]),
                     "p_upper_draw_level": upper,
                     "expected_independent": sum(rates["p_independent"]),
                     "p_upper_independent": ind_upper, "p_lower_independent": ind_lower,
                     "expected_union_bound": sum(rates["p_union_bound"])}
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(run, json.dumps(feature_chance_test(Path(run)), indent=1))
