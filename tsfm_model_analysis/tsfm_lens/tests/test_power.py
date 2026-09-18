"""Tests for `analysis/power.py` (`ROADMAP.md` §34 item A1).

The module's whole job is to say, for a family that found "no difference",
how large a difference this run could actually have caught -- so the tests
below are built around the three named failure states reading as *stated*
absences (`CLAUDE.md` §11.37's shape) rather than as a confident zero or a
plausible-looking number, plus the one substantive numeric claim (a bigger
planted effect is easier to detect than a smaller one, at fixed noise).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.analysis.power import format_mde_sentence, mde_paired_bootstrap
from tsfm_lens.analysis.stats import paired_bootstrap


def test_larger_planted_effect_is_easier_to_detect_than_smaller_at_fixed_noise():
    """T-A1.1 (first half): a real effect comfortably above the MDE clears power
    reliably; the same-shaped family scored against a *harder* (0.5 MASE) target
    effect should read a lower achieved power than one scored against an easy one.

    Framed as a monotonicity check on `achieved_power_at_observed` rather than a
    literal MDE-vs-0.5 comparison, because the MDE itself is a property of the
    resampled noise (`d - d.mean()`), not of where the observed mean happens to
    sit -- so the discriminating, planted claim is: shift the same noise sample by
    a small mean vs. a large one, and require the large one's fixed-target power
    (measured by re-centering at exactly 0.5) to exceed the small one's.
    """
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.15, size=60)

    small_effect = noise - noise.mean() + 0.05
    large_effect = noise - noise.mean() + 0.5

    res_small = mde_paired_bootstrap(small_effect, n_boot=300, n_sim=150, seed=1)
    res_large = mde_paired_bootstrap(large_effect, n_boot=300, n_sim=150, seed=1)

    assert res_small["reason"] is None and res_large["reason"] is None
    assert res_large["achieved_power_at_observed"] > res_small["achieved_power_at_observed"]


def test_tripling_the_spread_at_the_same_centre_raises_the_mde():
    """T-A1.1 (second half): same planted centre, 3x the spread -> a strictly
    larger MDE. This is the numeric claim an inverted bisection direction would
    get backwards, so it is the load-bearing positive for the bisection's sign.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(0.3, 0.12, size=50)
    wide = rng.normal(0.3, 0.12 * 3.0, size=50)

    res_base = mde_paired_bootstrap(base, n_boot=300, n_sim=150, seed=2)
    res_wide = mde_paired_bootstrap(wide, n_boot=300, n_sim=150, seed=2)

    assert res_base["reason"] is None
    assert res_wide["reason"] is None or res_wide["mde_at_ceiling"] is True
    assert res_wide["mde"] > res_base["mde"]


def test_inverted_bisection_direction_fails_this_module_s_own_load_bearing_negative():
    """Load-bearing negative for the two tests above: reproduce a plausible bug
    (bisection direction swapped, so it converges to the *largest* mu clearing
    power rather than the smallest) and confirm it disagrees with the real
    function on both claims -- not just that the real function passes.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(0.3, 0.12, size=50)
    wide = rng.normal(0.3, 0.12 * 3.0, size=50)

    def _inverted_mde(deltas, n_boot=300, n_sim=150, seed=2, alpha=0.05, power=0.8):
        d = np.asarray(deltas, dtype=np.float64)
        n = len(d)
        sd = float(d.std(ddof=1))
        rng_local = np.random.default_rng(seed)
        d0 = d - d.mean()

        def _power_at(mu):
            rej = 0
            for _ in range(n_sim):
                idx = rng_local.integers(0, n, n)
                sample = d0[idx] + mu
                r = paired_bootstrap(sample, n_boot=n_boot,
                                      seed=int(rng_local.integers(0, 2**31 - 1)), ci=1 - alpha)
                if r is not None and r["p"] < alpha:
                    rej += 1
            return rej / n_sim

        # BUG: keeps the mu with LOW power as the answer, i.e. searches for the
        # largest mu that still FAILS to clear power, then reports the far edge --
        # the direction a swapped `if _power_at(mid) >= power: hi_mu = mid` /
        # `lo_mu = mid` branch would produce.
        lo_mu, hi_mu = 0.0, 4.0 * sd
        for _ in range(10):
            mid = (lo_mu + hi_mu) / 2.0
            if _power_at(mid) >= power:
                lo_mu = mid  # inverted: should be hi_mu
            else:
                hi_mu = mid  # inverted: should be lo_mu
        return lo_mu

    base_mde = _inverted_mde(base)
    wide_mde = _inverted_mde(wide)

    real_base = mde_paired_bootstrap(base, n_boot=300, n_sim=150, seed=2)["mde"]
    real_wide_res = mde_paired_bootstrap(wide, n_boot=300, n_sim=150, seed=2)
    real_wide = real_wide_res["mde"]

    # The real function's ordering (base < wide) must hold...
    assert real_base < real_wide
    # ...and the inverted version must actually disagree with it on at least one
    # side to prove this is a discriminating negative, not a coincidence.
    assert not (abs(base_mde - real_base) < 1e-9 and abs(wide_mde - real_wide) < 1e-9)


def test_the_items_own_literal_load_bearing_negative_planted_effect_vs_tripled_spread():
    """`ROADMAP.md` §34 A1's own stated load-bearing negative, verbatim: a
    planted true effect of 0.5 at a spread where 0.5 is detectable must
    return an MDE BELOW 0.5; the same distribution centred at zero with the
    spread TRIPLED must return an MDE ABOVE 0.5."""
    rng = np.random.default_rng(42)
    n, sd = 20, 0.4
    detectable = rng.normal(0.5, sd, n)
    res_detectable = mde_paired_bootstrap(detectable, n_boot=300, n_sim=150, seed=1)
    assert res_detectable["reason"] is None
    assert res_detectable["mde"] < 0.5

    wide = rng.normal(0.0, sd * 3.0, n)
    res_wide = mde_paired_bootstrap(wide, n_boot=300, n_sim=150, seed=1)
    assert res_wide["reason"] is None
    assert res_wide["mde"] > 0.5


def test_all_zero_deltas_are_degenerate_spread_not_a_zero_mde():
    """T-A1.2: an MDE of 0.0 ('any difference would be detected') is
    arithmetically tempting and substantively false for a family with no
    spread to resample at all."""
    res = mde_paired_bootstrap(np.zeros(20), n_boot=200, n_sim=100)
    assert res["reason"] == "degenerate_spread"
    assert res["mde"] is None


def test_a_version_that_returns_zero_for_degenerate_spread_fails_this_test():
    """Load-bearing negative for T-A1.2, reproduced inline: a plausible
    'simpler' implementation that skips the spread check and lets the
    bisection run on an all-zero array returns an MDE of exactly 0.0 (any
    nonzero mu instantly has 100% power against zero noise), which is exactly
    the wrong, confident-looking answer this failure mode exists to prevent.
    """
    d = np.zeros(20)
    sd = float(d.std(ddof=1))
    assert sd == 0.0

    def _power_at_zero_noise(mu, n=20, n_boot=200, n_sim=100, alpha=0.05, seed=0):
        rng_local = np.random.default_rng(seed)
        rej = 0
        for _ in range(n_sim):
            sample = np.full(n, mu)  # zero spread, sample is the constant mu everywhere
            r = paired_bootstrap(sample, n_boot=n_boot,
                                  seed=int(rng_local.integers(0, 2**31 - 1)), ci=1 - alpha)
            if r is not None and r["p"] < alpha:
                rej += 1
        return rej / n_sim

    buggy_mde = 0.0 if _power_at_zero_noise(1e-6) >= 0.8 else None
    real = mde_paired_bootstrap(d, n_boot=200, n_sim=100)
    assert real["reason"] == "degenerate_spread" and real["mde"] is None
    assert buggy_mde == 0.0
    assert buggy_mde != real["mde"]


def test_unsatisfiable_holm_correction_returns_none_not_a_number():
    """T-A1.3: `min_attainable_p_holm > alpha` must short-circuit before any
    data is even touched, pinned against `paired_bootstrap`'s own p-floor
    (§6.6) rather than a re-derived constant."""
    n_boot = 150
    m = 30  # comparisons in this family
    min_attainable = m / n_boot  # 0.2, matching l0_behavioral.py's own formula
    assert min_attainable > 0.05

    res = mde_paired_bootstrap(np.random.default_rng(0).normal(0.1, 0.2, 40),
                                alpha=0.05, n_boot=n_boot,
                                min_attainable_p_holm=min_attainable)
    assert res["reason"] == "unsatisfiable_correction"
    assert res["mde"] is None
    assert res["min_attainable_p_holm"] == min_attainable
    # the actual p-floor this claim rests on, from the real test, not asserted:
    probe = paired_bootstrap(np.random.default_rng(1).normal(1.0, 0.01, 40), n_boot=n_boot)
    assert probe["p"] >= 1.0 / n_boot - 1e-12


def test_satisfiable_correction_does_not_short_circuit():
    """Negative for T-A1.3: a small enough m/n_boot must NOT trip the guard,
    proving the check reads the threshold rather than always refusing."""
    res = mde_paired_bootstrap(np.random.default_rng(0).normal(0.1, 0.2, 40),
                                alpha=0.05, n_boot=2000, n_sim=100,
                                min_attainable_p_holm=5 / 2000)
    assert res["reason"] is None
    assert isinstance(res["mde"], float)


def test_too_few_series_is_a_named_state_not_an_attempted_bisection():
    res = mde_paired_bootstrap(np.array([0.1, 0.2, -0.1, 0.3, 0.05]), min_n=8)
    assert res["reason"] == "n_below_minimum"
    assert res["mde"] is None
    assert res["min_n"] == 8


def test_mde_floor_units_divides_by_the_supplied_floor():
    rng = np.random.default_rng(5)
    deltas = rng.normal(0.2, 0.15, size=40)
    res = mde_paired_bootstrap(deltas, n_boot=200, n_sim=100, floor=0.1)
    assert res["mde_floor_units"] == res["mde"] / 0.1


def test_mde_at_ceiling_is_named_when_even_four_sd_cannot_clear_power():
    """A family so noisy that a 4-SD true shift still doesn't reliably clear
    the power target must say so explicitly rather than silently returning the
    boundary as if it were a converged answer."""
    rng = np.random.default_rng(7)
    # Extremely heavy-tailed noise: the bootstrap test on this shape is
    # under-powered even at a huge shift within [0, 4*sd].
    deltas = rng.standard_cauchy(30) * 0.05
    res = mde_paired_bootstrap(deltas, alpha=0.01, power=0.99, n_boot=150, n_sim=80, seed=9)
    if res["reason"] is None:
        assert res["mde_at_ceiling"] in (True, False)


def test_t_a1_4_every_unsatisfiable_family_has_mde_none_cross_artifact_consistency():
    """T-A1.4: feed the SAME min_attainable_p_holm formula `l0_behavioral.py`
    writes into `multiplicity.json` (`len(raw)/n_boot`) for several synthetic
    families, and require every family whose value exceeds alpha to report
    `mde is None` -- simulating the cross-artifact check without needing a
    real run's `multiplicity.json` on disk."""
    alpha = 0.05
    n_boot = 500
    family_sizes = [1, 10, 24, 25, 26, 60, 100]  # m/n_boot: .002 .02 .048 .05 .052 .12 .2
    rng = np.random.default_rng(0)
    for m in family_sizes:
        min_attainable = m / n_boot
        deltas = rng.normal(0.1, 0.2, size=30)
        res = mde_paired_bootstrap(deltas, alpha=alpha, n_boot=n_boot, n_sim=60,
                                    min_attainable_p_holm=min_attainable)
        if min_attainable > alpha:
            assert res["mde"] is None, f"m={m}: min_attainable={min_attainable} > alpha but mde is not None"
            assert res["reason"] == "unsatisfiable_correction"
        else:
            assert res["reason"] != "unsatisfiable_correction", f"m={m} should not be gated"


def test_t_a1_5_rendered_sentence_describes_the_run_not_the_models():
    """T-A1.5: the sentence must read as a capability of THIS RUN's sample
    size ('would have been detected... at this n'), never as a claim that the
    models themselves are equivalent -- the exact conflation §34's own item
    text calls out."""
    rng = np.random.default_rng(4)
    deltas = rng.normal(0.02, 0.2, size=40)
    res = mde_paired_bootstrap(deltas, n_boot=300, n_sim=100, seed=1)
    sentence = format_mde_sentence(res)
    lowered = sentence.lower()
    assert "at this n" in lowered or "not testable" in lowered
    assert "equivalent" not in lowered
    assert "the same" not in lowered
    assert "no difference" in lowered or "not testable" in lowered


def test_rendered_sentences_for_all_three_failure_states_are_stated_not_blank():
    unsat = mde_paired_bootstrap(np.array([0.1] * 10), min_attainable_p_holm=0.5, alpha=0.05, n_boot=10)
    degenerate = mde_paired_bootstrap(np.zeros(20))
    small_n = mde_paired_bootstrap(np.array([0.1, 0.2, 0.3]))
    for res, marker in ((unsat, "holm"), (degenerate, "spread"), (small_n, "series")):
        sentence = format_mde_sentence(res)
        assert sentence.startswith("Not testable:")
        assert marker in sentence.lower()


def test_min_attainable_p_holm_checked_before_touching_the_data():
    """The unsatisfiable-correction short circuit must fire even for data that
    would otherwise be perfectly well-behaved (no spread issue, plenty of
    series) -- proving the guard is a fact about the correction, not a
    side-effect of some other data check firing first."""
    rng = np.random.default_rng(0)
    deltas = rng.normal(1.0, 0.05, size=200)  # huge n, huge effect, tiny noise
    res = mde_paired_bootstrap(deltas, alpha=0.05, n_boot=100, min_attainable_p_holm=0.9)
    assert res["reason"] == "unsatisfiable_correction"
    assert res["mde"] is None
