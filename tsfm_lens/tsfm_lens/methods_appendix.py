"""ROADMAP.md sec 21 J5: the advanced methods appendix.

One always-rendered report appendix stating, per estimator, the exact form,
the resampling unit and design, what the null is and why, the known failure
modes, and the assumption that would invalidate it -- the things a
skeptical reviewer asks for and no other document collects in one place.

Generated FROM each estimator's own module docstring wherever that
docstring already states the estimator's exact form (`CLAUDE.md`'s own
module docstrings are written at exactly this level of detail, so pulling
them verbatim is both the cheapest implementation and the one least likely
to drift out of sync with the code): `build_methods_appendix()` imports the
module and reads `__doc__` directly, so an edit to e.g.
`analysis/l1_geometry.py`'s docstring changes the rendered appendix on the
next report build with no second file to remember to update.

Two entries (SAE fidelity, `relative_decoder_norm`) are hand-written
instead, per this item's own "where possible" qualifier: their source
functions carry only one-line docstrings (`reconstruction_fidelity`,
`dead_feature_rate`, `relative_decoder_norm` in `sae/eval.py`/
`sae/crosscoder.py`), too thin to state a resampling design or a failure
mode on their own. These two paragraphs are marked `generated=False` below
and cite the real, already-measured failure modes this repo found rather
than a hypothetical one (CLAUDE.md sec 13's dead-dictionary/window-broadcast
findings; ROADMAP.md sec 6.2.1's dead-atom-symmetry finding on
`relative_decoder_norm`) -- a reviewer-facing appendix that only ever
warned about failure modes nobody actually hit would be decorative.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MethodEntry:
    label: str
    module: Optional[str]  # dotted path under tsfm_lens/, or None for a hand-written entry
    text: Optional[str] = None  # only for generated=False entries


_HAND_WRITTEN = {
    "SAE fidelity and dead-feature rate": (
        "Reconstruction fidelity is fraction-of-variance-explained "
        "(`sae/eval.py::reconstruction_fidelity`) computed over held-out window "
        "rows at the store's own train/eval split; dead-feature rate is the "
        "fraction of dictionary atoms with zero activation above a threshold "
        "over the same rows. Neither carries a bootstrap CI in the current "
        "implementation -- both are point estimates over one already-large row "
        "sample, not a series-resampled statistic, so read them as descriptive "
        "rather than as having a stated uncertainty. The known, already-measured "
        "failure mode is severe: every SAE number recorded in this repo before "
        "`ROADMAP.md` sec 6.2.1's Stage 0 gate was measured on a dictionary "
        "**94.5-97.3% dead** (`CLAUDE.md` sec 13), which the same subsystem "
        "later proved is not intrinsic -- an AuxK auxiliary loss the production "
        "`sae` stage still leaves off by default reaches 3.2% dead on the same "
        "checkpoints. A second, orthogonal failure mode: `forecast_preservation`'s "
        "original window-pooled broadcast approximation is exact for a "
        "patch-tokenizing model (TimesFM) but badly lossy for a per-timestep "
        "one (Chronos-T5), producing a ΔMASE of +3.9 that a token-granularity "
        "re-measurement of the identical trained SAE resolves to +0.25 -- "
        "**the invalidating assumption is that every captured model tokenizes "
        "at the same granularity as the pooling window**, which is false by "
        "construction for any model whose token width differs from "
        "`alignment.window`."),
    "relative_decoder_norm (crosscoder shared/specific split)": (
        "`||dec_a[f]|| / (||dec_a[f]|| + ||dec_b[f]||)` per feature, on a "
        "**jointly** decoder-normalized crosscoder dictionary -- 0.5 reads as "
        "shared, 0 or 1 as source-specific. It is not resampled or bootstrapped "
        "in the current implementation, and its point value is provably "
        "sensitive to something orthogonal to the shared/specific question it "
        "is meant to answer: on a mostly-**dead** dictionary, dead atoms sit "
        "near a symmetric norm split by construction, which inflated the "
        "shared-fraction reading to roughly 98% before `ROADMAP.md` sec 6.2.1's "
        "Stage 0 gate produced a mostly-alive dictionary to measure it on "
        "properly. Even on that alive dictionary, a same-shape untrained-twin "
        "control read as *more* shared with itself than the real cross-model "
        "pair, and that control's own value proved **seed-fragile** "
        "(0.696-0.856 across 3 seeds) -- so **the invalidating assumption is "
        "that a single point estimate, or even a single-seed floor, is "
        "informative on its own**; this repo's own recorded practice is to "
        "never quote it without both an untrained-twin floor computed at "
        "multiple seeds and the dictionary's own alive-fraction stated "
        "alongside it."),
}

_ESTIMATORS = [
    MethodEntry("Linear CKA and RSA (L1 representational geometry)",
                "tsfm_lens.analysis.l1_geometry"),
    MethodEntry("Ridge stitching (L2 cross-model probes)",
                "tsfm_lens.analysis.l2_stitching"),
    MethodEntry("Forecast lens -- skip lens and tuned ridge lens",
                "tsfm_lens.analysis.lens"),
    MethodEntry("MASE / sMAPE / pinball (L0 behavioral profiling)",
                "tsfm_lens.analysis.l0_behavioral"),
    MethodEntry("Bootstrap designs -- paired, cluster, percentile",
                "tsfm_lens.analysis.stats"),
    MethodEntry("Adjusted mutual information (L4 activation clustering)",
                "tsfm_lens.analysis.clustering"),
    MethodEntry("SAE fidelity and dead-feature rate", None,
                text=_HAND_WRITTEN["SAE fidelity and dead-feature rate"]),
    MethodEntry("relative_decoder_norm (crosscoder shared/specific split)", None,
                text=_HAND_WRITTEN["relative_decoder_norm (crosscoder shared/specific split)"]),
    MethodEntry("Specification curve (analysis-knob robustness)",
                "tsfm_lens.analysis.spec_curve"),
]


def build_methods_appendix() -> list[dict]:
    """Return `[{label, module, text, generated}, ...]`.

    Raises `RuntimeError` for a module-backed entry whose module has no
    docstring or fails to import -- a silently blank appendix row is worse
    than a loud failure at report-build time (`CLAUDE.md` sec 2.5). Never
    silently falls back to placeholder text.
    """
    entries = []
    for e in _ESTIMATORS:
        if e.module is None:
            entries.append({"label": e.label, "module": None, "text": e.text,
                            "generated": False})
            continue
        mod = importlib.import_module(e.module)
        doc = (mod.__doc__ or "").strip()
        if not doc:
            raise RuntimeError(f"methods_appendix: {e.module} has no module docstring "
                                f"to render for {e.label!r}")
        entries.append({"label": e.label, "module": e.module, "text": doc,
                        "generated": True})
    return entries
