"""NaN / missing-timestep handling probe (`ROADMAP.md` sec 16 E17).

Does an adapter's `predict()` have any real missing-value handling path, or
does a NaN injected into the context error out, silently propagate into the
forecast, or (genuinely) get handled? This is not assumed true of any
adapter -- some tokenizers have an explicit masking path (Chronos-T5's
`MeanScaleUniformBins._input_transform` computes `attention_mask =
~isnan(context)` and excludes masked positions from its scale/token
computation), and some have none at all, in which case a NaN silently
poisons every downstream matmul (`CLAUDE.md` sec 2.5: a capability an
adapter lacks is itself the finding, recorded rather than crashed on or
assumed away).

This module holds the pure classification over an already-computed list of
per-scenario outcomes; the I/O (injecting NaN at a few positions and calling
`adapter.predict()`, catching whatever it raises) lives in
`analysis/frontend.py`, mirroring every other front-end probe's split.
"""

from __future__ import annotations


def nan_handling_stats(results: list) -> dict:
    """Classify how an adapter responds to NaN in its context, across scenarios.

    Each element of `results` is a dict with:
      - `position`: a label for where the NaN was injected (e.g. "front",
        "middle", "back") -- not used in scoring, kept for the record.
      - `raised`: bool, whether `predict()` raised an exception.
      - `error_type`/`error_msg`: set only when `raised` is True.
      - `output_has_nonfinite`: bool or None (None only when `raised` is
        True, since no output exists to check).

    Not a bootstrapped statistic (no `resample_unit`/CI here) -- these are a
    handful of deliberately engineered scenarios, not a random sample of
    series, so a confidence interval over them would claim a precision this
    probe was never designed to have.

    `verdict` is the single word this probe exists to answer:
      - `"errors"`: every scenario raised -- no missing-value handling at all.
      - `"propagates"`: nothing raised, but every scenario's forecast
        contains at least one non-finite value -- NaN silently entered and
        silently left, corrupting the forecast without any error.
      - `"handled"`: nothing raised and every forecast is fully finite --
        genuine missing-value handling.
      - `"mixed"`: behavior differs by injection position, which is itself
        worth surfacing rather than averaging away.
    """
    if not results:
        raise ValueError("need at least one injection scenario")
    n = len(results)
    n_raised = sum(1 for r in results if r["raised"])
    n_propagated = sum(1 for r in results
                       if not r["raised"] and r.get("output_has_nonfinite"))
    n_clean = n - n_raised - n_propagated

    if n_raised == n:
        verdict = "errors"
    elif n_raised == 0 and n_propagated == n:
        verdict = "propagates"
    elif n_raised == 0 and n_propagated == 0:
        verdict = "handled"
    else:
        verdict = "mixed"

    return {"n_scenarios": n, "n_raised": n_raised,
            "n_propagated_nonfinite": n_propagated, "n_clean": n_clean,
            "verdict": verdict, "per_scenario": results}
