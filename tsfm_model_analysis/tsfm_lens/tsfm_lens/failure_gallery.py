"""ROADMAP.md sec 21 J6: the failure-mode gallery.

A short, permanent, always-rendered report section showing what each
analysis looks like **when it goes wrong** -- five real cases already on
record in `CLAUDE.md`/`ROADMAP.md`, each of which fooled someone in this
repo before it was diagnosed. Nothing else in the repo teaches a reader to
be suspicious of a plausible-looking plot the way a worked failure does;
every entry here is `evidence_class="illustrative"` in the same sense
`_how_to_read`'s ladder gives the Exemplars section -- concrete cases
chosen because they show a failure mode clearly, not because they are
typical, and not resampled or re-measured against the current run.

Unlike `methods_appendix.py`, these five are historical narratives about
specific past runs (several predate the fixes that make today's numbers
trustworthy), not something any current module docstring states -- so
every entry is hand-written and frozen, with an explicit citation into
`CLAUDE.md`/`ROADMAP.md` so a reader can verify the real numbers rather
than trusting a summary. Frozen deliberately: this gallery does not grow
every time a new bug is fixed (`CLAUDE.md`'s own sec 11 traps list already
does that job for developers of this repo); it stays at the handful of
cases most useful for teaching a *reader of the report* what to be
suspicious of.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GalleryEntry:
    title: str
    looked_like: str
    actually_was: str
    lesson: str
    source: str


GALLERY = [
    GalleryEntry(
        title="An alignment check that measured its own bug, not the model",
        looked_like=(
            "Chronos-T5-Small's impulse-alignment check reported a minimum "
            "diagonal-hit fraction of 0.06 -- barely above the 1-in-16 chance "
            "floor -- with a non-monotonic per-layer pattern (0.50, 0.06, "
            "0.88, 0.62, 0.44, 0.38). Read at face value this says exactly "
            "what invariant 7 warns about: the declared token-to-time map is "
            "wrong, and no cross-model number built on it should be trusted."),
        actually_was=(
            "The probe added a fixed absolute impulse (+8.0, 8x the probe "
            "signal's own amplitude) to test which pooled window's "
            "activations changed most. Chronos's tokenizer derives its "
            "quantization bin edges from the *whole sequence's* statistics, "
            "so an impulse that large shifted those edges and re-quantized "
            "472 of 513 tokens for every impulse position tested -- the test "
            "was measuring the tokenizer's sensitivity to outlier magnitude, "
            "not the correctness of the declared spans. Scaling the "
            "perturbation relative to the probe signal's own amplitude "
            "(0.25x, later calibrated per checkpoint/context-length) restored "
            "a perfect 1.00 diagonal-hit fraction at every layer."),
        lesson=(
            "A fixed-magnitude test probe tuned against one adapter can "
            "silently fail on a structurally different one -- scale a "
            "perturbation relative to the signal it perturbs, not to an "
            "absolute constant, whenever the thing being probed can itself "
            "rescale based on input statistics."),
        source="CLAUDE.md sec 11.16"),
    GalleryEntry(
        title="A \"flat\" causal-patching curve that was an averaging artifact",
        looked_like=(
            "TimesFM's whole-layer-averaged activation-patching restoration "
            "curve was essentially flat across depth -- read at face value, "
            "this says TimesFM carries corruption damage everywhere and "
            "nowhere in particular: no localized causal structure to report."),
        actually_was=(
            "Re-measuring the identical checkpoint with PER-WINDOW patching "
            "(a layer x window restoration heatmap, rather than one number "
            "per layer averaged over every window) recovered real, "
            "depth-intensifying causal structure the coarser whole-layer "
            "average had been hiding: effects that are real and "
            "spatially localized can cancel out, in aggregate, to look like "
            "no effect at all once averaged over position."),
        lesson=(
            "An aggregate statistic (a mean taken over positions, series, or "
            "windows) can manufacture \"no effect\" out of real, "
            "spatially-localized effects that point in different directions "
            "once pooled -- when a causal or behavioral curve looks flat, "
            "check whether the flatness survives a finer-grained slice "
            "before concluding the effect is genuinely absent."),
        source="CLAUDE.md intro reconciliation note (2026-08-03, third session); "
               "ROADMAP.md sec 5.4 Findings"),
    GalleryEntry(
        title="A 98%-dead SAE dictionary that still had a plausible fidelity number",
        looked_like=(
            "Every SAE number recorded in this repo before ROADMAP.md sec "
            "6.2.1's Stage 0 gate was measured on a dictionary 94.5-97.3% "
            "dead -- and yet it trained without error, had a reconstruction "
            "fidelity around 0.84-0.86, and had features a reader could open "
            "and look at, none of which by itself signals a problem."),
        actually_was=(
            "The vast majority of the dictionary's capacity was never used: "
            "any feature-level claim built on top (ground-truth alignment, "
            "shared-vs-specific splits) was drawn from a small, unrepresentative "
            "surviving fraction of atoms. A dedicated feasibility gate with an "
            "AuxK dead-atom auxiliary loss -- a knob that already existed in "
            "the codebase but was left off by default -- showed the same "
            "checkpoints reach 3.2% dead once turned on."),
        lesson=(
            "A metric that looks reasonable in isolation (fidelity ~0.85) can "
            "coexist with a dictionary, model, or feature set that is almost "
            "entirely inert -- check the fraction of capacity actually in use "
            "directly, rather than inferring it from a downstream metric "
            "that can look fine while measuring only the surviving remainder."),
        source="CLAUDE.md sec 13; ROADMAP.md sec 6.2.1 Stage 0 Findings"),
    GalleryEntry(
        title="The same depth-agreement figure, two different stories, from the axis alone",
        looked_like=(
            "Plotting TimesFM's and Chronos-T5-Base's depth-resolved curves "
            "(crystallization, CKA, fingerprint agreement) against the "
            "legacy `index/(n_captured-1)` relative-depth axis makes both "
            "curves span the same visual 0-to-1 range, inviting a reading "
            "like \"these models agree most in the second half of depth.\""),
        actually_was=(
            "Recomputed on the `block` axis (counting every architectural "
            "block, including the ones outside each model's captured layer "
            "stride/regex), Chronos-T5-Base's captured surface tops out at "
            "relative depth 0.478 -- its encoder is roughly half the model, "
            "and its entire decoder is uncaptured. \"Agreement in the second "
            "half of depth\" on the legacy axis is actually \"in the middle "
            "of Chronos's depth, near the very end of TimesFM's\" once the "
            "axis is corrected -- the two curves were never sharing the same "
            "kind of coordinate."),
        lesson=(
            "A shared 0-to-1 depth axis across two architecturally different "
            "models is a convention chosen for the plot, not a fact about "
            "either model -- and the convention itself can move where an "
            "agreement or disagreement in depth appears to be located."),
        source="CLAUDE.md sec 12 item 4; ROADMAP.md sec 18 F1 Findings"),
    GalleryEntry(
        title="A statistically-real ΔMASE that was quieter than the model's own noise",
        looked_like=(
            "L0's `mixture` family gap between TimesFM and Chronos-T5-Base "
            "measured 0.010 MASE -- a real, Holm-corrected number, computed "
            "correctly, that would read as \"TimesFM wins this family\" if "
            "quoted on its own."),
        actually_was=(
            "Chronos-T5-Base's own repeat-run noise floor -- how much its "
            "OWN MASE varies run to run with nothing about the model or data "
            "changed -- is +-0.160, roughly sixteen times larger than the "
            "gap being quoted. The 0.010 gap is not a small real difference; "
            "it is not a measurement at all, since it sits far inside the "
            "model's own measurement noise."),
        lesson=(
            "A small delta is not evidence of a small real difference -- it "
            "can be no measurement whatsoever, and the only way to tell the "
            "two apart is to have measured the model's own repeat-run noise "
            "floor first and compared the delta against it, not against zero."),
        source="ROADMAP.md sec 18 F6 Findings; docs/worked_example.md"),
]


def gallery_entries() -> list:
    """Return the frozen list of `GalleryEntry`. A plain accessor (mirroring
    `glossary.terms()`), kept so a caller does not import the module-level
    `GALLERY` list name directly and so a future change in storage (e.g. a
    sort) has one place to make it."""
    return list(GALLERY)
