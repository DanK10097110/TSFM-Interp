"""Statistical power and the minimum detectable effect for L0's paired tests.

`ROADMAP.md` sec 34 item A1: `l0/summary.json` reports, per family, whether a
difference between the two models was *detected*. It never reports what size
of difference this run *could* have detected -- so a reader cannot tell "these
models are equivalent here" apart from "this family had 23 series". This
module is the fix: a simulation-based (not parametric -- the test it
characterizes is itself a bootstrap, so a parametric power formula would
describe a different test) minimum-detectable-effect (MDE) for
`analysis/stats.py::paired_bootstrap`, reused as-is rather than re-derived, so
the MDE's null-rejection rule is provably the same rule the real test uses.

Three failure modes get a *named* `None` rather than a number, because a
degenerate or absent measurement rendering as a confident one is exactly
`CLAUDE.md` sec 11.37's shape, and this module is one of the sites
`ROADMAP.md` sec 34.7 item 2 names for it directly: an unsatisfiable Holm
correction (the smallest attainable adjusted p already exceeds alpha, so no
effect of any size is detectable -- the MDE is infinite, not merely large), a
family too small to resample meaningfully, and a family whose observed deltas
have no spread at all (an MDE of exactly 0 would be arithmetically true and
substantively false).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .stats import paired_bootstrap

# A true MDE search is unbounded above; four observed standard deviations is
# the ceiling A1.1 specifies. Reported as a boundary case (`mde_at_ceiling`),
# never silently returned as if it were a converged bisection result -- a
# family whose own noise is so large that even a 4-SD shift isn't reliably
# detectable is a real, nameable state, not a number to trust at face value.
_MDE_GRID_SPAN_SD = 4.0
# The observed-effect power check and the ceiling check are each one
# `n_sim`-iteration simulation; the bisection converges in practice well under
# `n_grid` iterations because of the early-stop tolerance below, matching this
# item's own "~8 evaluations" cost estimate.
_BISECTION_REL_TOL = 0.01


def mde_paired_bootstrap(deltas: np.ndarray, alpha: float = 0.05, power: float = 0.8,
                         n_boot: int = 500, seed: int = 0, n_grid: int = 24,
                         n_sim: int = 400, min_attainable_p_holm: Optional[float] = None,
                         min_n: int = 8, floor: Optional[float] = None) -> dict:
    """Minimum true effect this (family's) paired bootstrap test could detect.

    `deltas` are the observed per-series paired differences (one `paired_
    bootstrap` call's worth) -- the same array a caller would hand to
    `paired_bootstrap` for the real test. `min_attainable_p_holm`, when
    given, is the SAME value the caller writes into `multiplicity.json`
    (`len(raw_tests) / n_boot`, sec 6.6) -- passed in, not re-derived here, so
    the two can never independently drift (`ROADMAP.md` sec 34.7 item 1's
    "the check that catches drift between two sites" is exactly this: a
    second, separately-computed `m/n_boot` here would be a second place for
    that arithmetic to go stale).

    Algorithm (sec 34 A1.1):
    1. Centre the observed deltas: `d0 = deltas - deltas.mean()` -- the
       observed noise shape with the observed effect removed, preserving
       whatever heavy-tailed/non-Gaussian spread the real data has, which a
       parametric power formula would throw away.
    2. For a candidate true shift `mu`, repeatedly draw a paired bootstrap
       resample of `d0 + mu` (resampling SERIES, never windows -- invariant
       2) and run the real `paired_bootstrap` test on it; the fraction of
       `n_sim` such simulations rejecting at `alpha` is the power at `mu`.
    3. Bisect on `mu` over `[0, 4*sd(deltas)]` for the smallest `mu` whose
       simulated power >= `power`.

    Returns a dict always carrying `n`/`alpha`/`power`/`n_boot`/`n_sim` plus
    exactly one of: `mde` (+ `mde_floor_units` when `floor` is given, +
    `achieved_power_at_observed`), or `reason` naming which of the three
    named failure states applies (`unsatisfiable_correction`,
    `n_below_minimum`, `degenerate_spread`) with `mde` left `None`.
    """
    d = np.asarray(deltas, dtype=np.float64)
    n = int(len(d))
    out = {"mde": None, "mde_floor_units": None, "achieved_power_at_observed": None,
          "mde_at_ceiling": None, "n": n, "alpha": float(alpha), "power": float(power),
          "n_boot": int(n_boot), "n_sim": int(n_sim), "reason": None}

    # Failure mode 1 (sec 34 A1.1's table, row 1): the p-floor makes the MDE
    # undefined. Checked first and without touching the data at all, because
    # this is a fact about the CORRECTION, not about this family's effect --
    # no true effect of any size can clear an unsatisfiable Holm correction.
    if min_attainable_p_holm is not None and min_attainable_p_holm > alpha:
        out["reason"] = "unsatisfiable_correction"
        out["min_attainable_p_holm"] = float(min_attainable_p_holm)
        m = min_attainable_p_holm * n_boot
        out["n_boot_needed"] = int(np.ceil(m / alpha)) if alpha > 0 else None
        return out

    # Failure mode 3 (row 3): too few series to resample meaningfully.
    if n < min_n:
        out["reason"] = "n_below_minimum"
        out["min_n"] = int(min_n)
        return out

    # Failure mode 2 (row 2): zero-spread deltas -- two deterministic models
    # on identical input, or a family where every series gives the same
    # delta. An MDE of 0.0 ("any difference at all would be detected") is
    # arithmetically true here and substantively false, so it is a named
    # state, never a computed 0.0.
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd <= 0.0:
        out["reason"] = "degenerate_spread"
        return out

    rng = np.random.default_rng(seed)
    d0 = d - d.mean()

    def _power_at(mu: float) -> float:
        rejections = 0
        for _ in range(n_sim):
            idx = rng.integers(0, n, n)
            sample = d0[idx] + mu
            res = paired_bootstrap(sample, n_boot=n_boot,
                                   seed=int(rng.integers(0, 2**31 - 1)), ci=1 - alpha)
            if res is not None and res["p"] < alpha:
                rejections += 1
        return rejections / n_sim

    out["achieved_power_at_observed"] = _power_at(abs(float(d.mean())))

    lo_mu, hi_mu = 0.0, _MDE_GRID_SPAN_SD * sd
    if _power_at(hi_mu) < power:
        # Even a 4-SD true shift isn't reliably detectable at this n -- report
        # the boundary itself, named as a boundary rather than a converged
        # answer, per this item's own scope ("[0, 4*sd(deltas)]").
        out["mde"] = hi_mu
        out["mde_at_ceiling"] = True
    else:
        out["mde_at_ceiling"] = False
        span = hi_mu - lo_mu
        for _ in range(n_grid):
            mid = (lo_mu + hi_mu) / 2.0
            if _power_at(mid) >= power:
                hi_mu = mid
            else:
                lo_mu = mid
            if (hi_mu - lo_mu) <= _BISECTION_REL_TOL * span:
                break
        out["mde"] = hi_mu

    if floor is not None and float(floor) > 0.0:
        out["mde_floor_units"] = float(out["mde"]) / float(floor)
    return out


def mde_ablation_effect(row_abs_effects: np.ndarray, null_draw_means: np.ndarray,
                        alpha: float = 0.05, m: int = 1, power: float = 0.8,
                        n_sim: int = 400, n_grid: int = 24, seed: int = 0,
                        null_p95: Optional[float] = None, min_n: int = 3) -> dict:
    """Smallest true ablation effect a registered `concept_causal` claim's
    test could detect (`ROADMAP.md` sec 38.2.3), so a non-replication reads
    as "absent" or "underpowered" rather than one undifferentiated failure.

    The test being characterized: the claim's statistic is the mean |effect|
    over its `k` top-firing private rows; its null is the same mean over
    each of `n` random-direction draws (`null_draw_means`, one value per
    draw); the exact p is `(1 + #{null >= stat}) / (n + 1)`; the claim is
    confirmed at the Holm level `alpha / m` (the smallest threshold a
    Holm-corrected family of `m` ever applies -- conservative for every rank
    but the first, and the only closed form available before the other
    claims' p-values exist).

    Simulation, in the spirit of `mde_paired_bootstrap` (a parametric formula
    would describe a different test): centre the observed per-row effects
    (`d0 = row_abs_effects - mean`) so the noise shape stays and the effect
    is removed, resample ROWS (the claim's own series, its resampling unit)
    at the observed `k`, shift the resample by a candidate true mean `mu`,
    and test it against the observed null draws. Bisect on `mu` for the
    smallest one whose rejection rate reaches `power`. `mde` is in the
    channel's own units; `mde_over_null_p95` divides by `null_p95` (the
    row-level p95 that dev's `effect / null_p95` vector uses) when given, so
    it reads on the same scale as `sae/concepts.py::ablation_vector`.

    Named `None` states instead of a number, as in `mde_paired_bootstrap`:
    `unsatisfiable_correction` (`m / (n + 1) > alpha`: no effect of any size
    can reach the Holm level), `n_below_minimum` (fewer than `min_n` rows),
    `degenerate_null` (the null draws have no spread, so any nonzero effect
    trivially "clears" it -- `CLAUDE.md` sec 11.37).
    """
    d = np.asarray(row_abs_effects, dtype=np.float64)
    d = d[np.isfinite(d)]
    null = np.asarray(null_draw_means, dtype=np.float64)
    null = null[np.isfinite(null)]
    k, n = int(d.size), int(null.size)
    out = {"mde": None, "mde_over_null_p95": None, "mde_at_ceiling": None,
          "achieved_power_at_observed": None, "k": k, "n_null": n, "m": int(m),
          "alpha": float(alpha), "power": float(power), "n_sim": int(n_sim),
          "reason": None}
    if n == 0:
        out["reason"] = "degenerate_null"
        return out
    min_p = float(m) / (n + 1)
    out["min_attainable_p_holm"] = min_p
    if min_p > alpha:
        out["reason"] = "unsatisfiable_correction"
        return out
    if k < min_n:
        out["reason"] = "n_below_minimum"
        out["min_n"] = int(min_n)
        return out
    if not (float(null.std()) > 0.0):
        out["reason"] = "degenerate_null"
        return out

    level = float(alpha) / max(1, int(m))
    sorted_null = np.sort(null)
    rng = np.random.default_rng(seed)
    d0 = d - d.mean()

    def _p(stat: float) -> float:
        count = n - int(np.searchsorted(sorted_null, stat, side="left"))
        return (1 + count) / (n + 1)

    def _power_at(mu: float) -> float:
        hits = 0
        for _ in range(n_sim):
            idx = rng.integers(0, k, k)
            if _p(float(np.mean(d0[idx] + mu))) <= level:
                hits += 1
        return hits / n_sim

    out["achieved_power_at_observed"] = _power_at(float(d.mean()))
    lo_mu, hi_mu = 0.0, max(float(null.max()) * 4.0, float(d.mean()) * 4.0, 1e-12)
    if _power_at(hi_mu) < power:
        out["mde"], out["mde_at_ceiling"] = hi_mu, True
    else:
        out["mde_at_ceiling"] = False
        span = hi_mu - lo_mu
        for _ in range(n_grid):
            mid = (lo_mu + hi_mu) / 2.0
            if _power_at(mid) >= power:
                hi_mu = mid
            else:
                lo_mu = mid
            if (hi_mu - lo_mu) <= _BISECTION_REL_TOL * span:
                break
        out["mde"] = hi_mu
    if null_p95 is not None and float(null_p95) > 0.0:
        out["mde_over_null_p95"] = float(out["mde"]) / float(null_p95)
    return out


def format_mde_sentence(mde: dict) -> str:
    """The one rendering of an MDE result every L0 table cell shares.

    Phrased as a capability of the RUN, never of the models (sec 34 A1.1's
    failure-mode table, row 4) -- "no difference was detected" is not the
    same claim as "the models are equivalent", and a reader who is not told
    the difference must read the second into the first.
    """
    reason = mde.get("reason")
    if reason == "unsatisfiable_correction":
        return (f"Not testable: the Holm correction for this many comparisons cannot "
                f"reach alpha={mde['alpha']:g} at n_boot={mde['n_boot']} (would need "
                f"n_boot>={mde.get('n_boot_needed')}) -- no effect of any size is "
                f"detectable at this setting, not just none was found.")
    if reason == "n_below_minimum":
        return (f"Not testable: only {mde['n']} series (need >= {mde.get('min_n')}) -- "
                f"too few to resample meaningfully.")
    if reason == "degenerate_spread":
        return "Not testable: every series in this family gives the same paired delta (no spread to resample)."
    if mde.get("mde") is None:
        return "Not testable."
    floor_clause = (f" ({mde['mde_floor_units']:.2f} noise floors)"
                    if mde.get("mde_floor_units") is not None else "")
    ceiling_clause = " (reached the search ceiling; the true MDE may be larger)" if mde.get("mde_at_ceiling") else ""
    return (f"No difference detected. A true difference of >= {mde['mde']:.3f} MASE"
           f"{floor_clause} would have been detected {mde['power'] * 100:.0f}% of the "
           f"time at this n{ceiling_clause}.")
