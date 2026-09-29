"""Single source of truth for the recurring vocabulary this repo's output uses.

`ROADMAP.md` sec 21 J3, the companion to sec 21 J2's `stage_docs.py`. J2
answers "what does this *stage* tell you" once per stage; this module answers
"what does this *word* mean" once per term, for the ~25 terms that recur
across sections without any one section owning them.

The report's per-figure `_note()` blocks stay exactly as they are: they are
contextual, and a reader going through a section top to bottom is well served
by them. This module exists for the other reader -- the one who lands on the
Stitching-probes section from a deep link, meets "gain over the input-feature
baseline" cold, and has nowhere to look it up. Hence the `where` field on
every term: a definition with no pointer to the figure it governs sends that
reader back to searching.

Rendered by two surfaces from this one dict, the same way `stage_docs.py` is,
so they cannot drift: the HTML report (`report/report.py::_glossary_block`,
collapsed under the "How to read this report" preamble) and the README
(`render_glossary.py`, which splices a generated block and has a `--check`
mode for a stale-README CI gate).

Writing rules for entries here, so a later session extends rather than
dilutes it:

- **One sentence per definition**, and it must be the *operational* meaning
  in this repo -- not the textbook one. "Linear CKA" has a general
  definition; what a reader needs is that it is computed here between two
  models' aligned window matrices with the series as the resampling unit.
- **Say the limitation inside the definition** where the term is routinely
  over-read (`CLAUDE.md` sec 2.6). "Relative depth" without "a convention,
  not a claim that the same fraction is the same computational stage" is
  worse than no entry, because it licenses exactly the misreading the
  repo has already documented as a real problem (`CLAUDE.md` sec 12 item 4).
- **`where` names a report section or artifact**, not a source file. The
  audience is a report reader, not a maintainer.
"""

from __future__ import annotations

from dataclasses import dataclass

# ROADMAP.md sec 32.7c Item J3: the eight response-channel entries below are
# composed from `sae.describe.CHANNEL_MEANING` (itself composed from
# `sae.vocab.CHANNEL_DEFS`, sec 32.7c Item J1) rather than hand-typed here,
# so this module cannot become a THIRD, independently-drifting vocabulary
# for the same nine channels -- CLAUDE.md sec 11.53 names exactly that shape
# as a recurring defect in this area of the repo. `describe_term` supplies
# each channel's short display label the same way the report's hover text
# and legends already do.
from .sae.describe import CHANNEL_MEANING as _CHANNEL_MEANING
from .sae.vocab import describe_term as _describe_term

_CHANNEL_BATTERY_WHERE = (
    "Sparse features (SAE) -- the causal channel battery's evidence columns, "
    "and every channel name rendered from it.")


def _channel_glossary_definition(name: str) -> str:
    """Fold `CHANNEL_MEANING`'s two-sentence "what. high." into one sentence.

    `CHANNEL_MEANING` is written for a narrated sentence and a hover tooltip,
    where two short sentences read naturally; this module's own writing rule
    (above) asks for one sentence per definition, so the two clauses are
    joined rather than re-derived independently -- re-deriving them here
    would be exactly the third-vocabulary duplication this module's import
    comment already explains avoiding.
    """
    what, _, high = _CHANNEL_MEANING[name].partition(". ")
    high = high.rstrip(". ")
    # Every `CHANNEL_DEFS.high` clause reads "Feature <verb-phrase>"; recast
    # as "a higher value <verb-phrase>" keeps the same verb (singular
    # subject either way) while folding it into one clause instead of a
    # second sentence.
    assert high.startswith("Feature "), f"unexpected CHANNEL_MEANING shape for {name!r}"
    return f"{what.rstrip('. ')} -- a higher value {high[len('Feature '):]}."


@dataclass(frozen=True)
class Term:
    """One glossary entry: the word, what it means here, and where it is used.

    `where` is not decoration -- it is what turns a definition list into a
    navigation aid for someone reading the report out of order, which is the
    only reason this module exists separately from the per-figure notes.
    """
    term: str
    definition: str
    where: str


GLOSSARY: dict = {
    "window": Term(
        term="Window",
        definition=(
            "A pooled fixed-length interval of real time (32 timesteps by default, "
            "`alignment.window`) that every model's own tokens are mapped onto, so an "
            "architecture reading one timestep per token and one reading thirty-two "
            "become comparable at all; it is a unit of time, never a unit of any one "
            "model's tokenization."),
        where="Every cross-model section — the shared [series, window, dim] axis all of them operate on."),
    "relative-depth": Term(
        term="Relative depth",
        definition=(
            "A layer's position rescaled to 0–1 so models with different layer counts "
            "share one axis — a convention that makes the comparison drawable, not a "
            "claim that the same fraction is the same computational stage in two "
            "architectures."),
        where="Model internals, Forecast lens, Representational geometry, Perturbation — any curve plotted against depth."),
    "depth-axis": Term(
        term="Depth axis (`index` / `block` / `compute`)",
        definition=(
            "Which definition of \"how deep\" a figure uses: `index` counts only the "
            "layers this run captured, `block` (the default) counts every block in the "
            "whole stack including surfaces never captured, and `compute` uses "
            "cumulative measured FLOPs; an encoder-only capture surface therefore caps "
            "well below 1.0 on `block` rather than misleadingly reaching it."),
        where="The fairness card's \"Depth axis\" row, and every depth-plotted figure's own note."),
    "evidence-class": Term(
        term="Evidence class",
        definition=(
            "The rung a claim sits on — geometric, linearly-translatable, "
            "causal-within-model, descriptive, illustrative, or confirmatory — stated "
            "for every finding so that a confident-looking chart cannot be read as "
            "stronger evidence than its method supports."),
        where="\"How to read this report\", and the caveat under every finding."),
    "mase": Term(
        term="MASE",
        definition=(
            "Mean absolute error divided by the error of a naive same-scale baseline on "
            "the same series, so that 1.0 means \"no better than naive\" and series of "
            "wildly different amplitudes can be averaged together; series whose scale "
            "denominator is too small to be reliable are excluded and counted, not "
            "silently kept."),
        where="Behavioral profile (the primary forecast-quality metric), and every ΔMASE elsewhere."),
    "smape-pinball": Term(
        term="sMAPE and pinball loss",
        definition=(
            "The two companion forecast metrics: sMAPE is a symmetric percentage error "
            "(scale-free but unstable near zero), and pinball loss scores the predicted "
            "*quantiles* rather than the point forecast, which is the only one of the "
            "three sensitive to whether a model's uncertainty is honest."),
        where="Behavioral profile's per-family metric table."),
    "noise-floor": Term(
        term="Noise floor",
        definition=(
            "How much a model's own metric moves between two identical repeat runs "
            "(nonzero for a sampled decoder, zero for a deterministic one), measured so "
            "that a delta can be reported as a multiple of it rather than against an "
            "implied zero — a ΔMASE under about 2× its own floor is not interpretable."),
        where="The fairness card's determinism row; every ΔMASE in ablation, patching and SAE sections."),
    "family-archetype": Term(
        term="Family and archetype",
        definition=(
            "Two levels of benchmark grouping: the *family* is which generator produced "
            "a series (`parametric`, `random_parametric`, `mixture`, …) and the "
            "*archetype* is the structural recipe drawn within it (`seasonal_dominant`, "
            "`intermittent_bursts`, …); per-family statistics are the unit multiple-"
            "comparison correction is applied across."),
        where="Behavioral profile, Representational geometry's per-family CKA, Activation clusters."),
    "leakage": Term(
        term="Leakage (instance-level vs. distributional)",
        definition=(
            "Two different failures a benchmark can have: instance-level leakage is "
            "the exact series having been seen in a model's training data, which the "
            "leakage gate audits directly against real reference series; "
            "distributional leakage is only *that kind* of data having been seen, "
            "which is carried by construction for every `real_derived`-tier sample "
            "and cannot be audited by this or any other check."),
        where="The corpus card's leakage-gate figure and trust ladder."),
    "epoch": Term(
        term="Epoch (corpus)",
        definition=(
            "A fresh, non-overlapping seed range used to mint a brand-new private "
            "split when an existing one is suspected of having leaked, so a peeked-at "
            "private corpus is regenerated rather than reused."),
        where="The corpus card's header line; private benchmark confirmation."),
    "redundancy-fraction": Term(
        term="Redundancy fraction",
        definition=(
            "The share of all pairwise series comparisons — by cross-correlation or "
            "banded DTW shape matching, after z-normalization — that fall below a "
            "near-duplicate distance threshold; a high value can be a true finding "
            "rather than a defect, e.g. many pure-seasonal series sharing one period "
            "and differing only in noise."),
        where="The corpus card's diversity figure, read alongside effective dimensionality."),
    "near-collision": Term(
        term="Near-collision fraction",
        definition=(
            "In the catch22 feature space — never on UMAP coordinates — the fraction "
            "of series whose nearest neighbor sits closer than a fixed threshold; a "
            "distinct diversity signal from the shape-matching redundancy fraction, "
            "since the two are computed on different representations and never "
            "merged until the report."),
        where="The corpus card's diversity figure."),
    "catch22": Term(
        term="catch22 (feature space)",
        definition=(
            "The 22 (or 24, adding mean/std) general-purpose time-series features "
            "`benchmark_validation` computes per series and robustly scales (with "
            "winsorization against outlier swamping) before computing diversity, "
            "effective dimensionality, or near-collisions on them — the space every "
            "diversity number in this report is measured in, never the UMAP "
            "coordinates used only for the companion plot."),
        where="The corpus card's diversity and feature-space-map figures."),
    "tier": Term(
        term="Tier (`synthetic` / `real_derived`)",
        definition=(
            "Whether a benchmark series touched real data at all: `synthetic` series are "
            "generated from scratch and carry exact ground truth with no distributional "
            "leakage, while `real_derived` series are built from real corpora and "
            "inherit their distribution even though every instance still passes the "
            "leakage audit."),
        where="The corpus this run reads; recorded per sample in the sealed manifest."),
    "sealed-corpus": Term(
        term="Sealed corpus (public dev / private test)",
        definition=(
            "A corpus written with per-sample hashes and a manifest, split into a public "
            "dev half used for all exploration and a private test half drawn from a "
            "disjoint seed range that is opened exactly once; the private half is a "
            "consumable, and a peeked one is replaced rather than reused."),
        where="Private benchmark confirmation; the run header's dataset line."),
    "effective-dimensionality": Term(
        term="Effective dimensionality",
        definition=(
            "The participation ratio of a representation's PCA spectrum — roughly, how "
            "many directions actually carry variance — used as a descriptive profile of "
            "where a model expands or compresses, not as a quality score."),
        where="Model internals' per-layer depth profile."),
    "probe-decodability": Term(
        term="Probe decodability",
        definition=(
            "The held-out accuracy of a simple linear classifier reading a property "
            "(family, seasonality, trend) straight off a layer's activations, which "
            "establishes that the information is linearly *present* there and nothing "
            "more — never that the model uses it."),
        where="Model internals; the SAE section's ground-truth alignment."),
    "linear-cka": Term(
        term="Linear CKA",
        definition=(
            "A 0–1 similarity between two models' representations of the same series at "
            "a chosen layer pair, computed on the shared window axis and bootstrapped "
            "with the series as the resampling unit; it is correlational, and both "
            "models seeing the same input inflates it on its own."),
        where="Representational geometry (the layer-pair heatmap and its depth curve)."),
    "rsa": Term(
        term="RSA (representational similarity analysis)",
        definition=(
            "The rank-correlation companion to CKA: instead of comparing representations "
            "directly it compares each model's *matrix of series-to-series distances*, so "
            "it survives any monotone rescaling CKA would not."),
        where="Representational geometry, reported alongside CKA."),
    "stitching-gain": Term(
        term="Stitching gain",
        definition=(
            "The extra held-out R² a ridge map from one model's layer to the other's "
            "achieves *above* a hand-crafted input-feature probe on the same targets — "
            "the gain, never the raw R², is the evidence, because the models read the "
            "same input and a raw R² is therefore partly trivial."),
        where="Stitching probes (both directions, A→B and B→A, reported separately)."),
    "input-feature-baseline": Term(
        term="Input-feature baseline",
        definition=(
            "A deliberately simple probe (raw window values, FFT magnitudes, summary "
            "statistics) predicting the same targets the stitching map does, whose whole "
            "job is to absorb the part of the agreement that follows from shared input "
            "rather than shared learned structure."),
        where="Stitching probes; the quantity every reported gain is measured against."),
    "untrained-twin-floor": Term(
        term="Untrained-twin floor",
        definition=(
            "The same measurement run against a randomly-initialized copy of a model's "
            "own architecture, giving the level any two networks of that shape reach "
            "before either has learned anything; a similarity that fails to clear it is "
            "architecture, not learning."),
        where="Null-baseline comparisons for CKA and stitching gain."),
    "skip-lens": Term(
        term="Skip lens and tuned lens",
        definition=(
            "Two ways to ask what forecast a middle layer already implies: the skip lens "
            "patches that layer's states in as the final block's output and lets the "
            "model's own head decode them, while the tuned lens fits a held-out ridge "
            "probe instead, correcting the skip lens's miscalibration at early layers."),
        where="Forecast lens."),
    "crystallization-depth": Term(
        term="Crystallization depth",
        definition=(
            "The shallowest relative depth at which the skip-lens forecast is already "
            "within a set tolerance (10% by default) of the model's final forecast "
            "quality — where the answer is essentially settled, not where computation "
            "stops."),
        where="Forecast lens."),
    "fingerprint": Term(
        term="Corruption sensitivity fingerprint",
        definition=(
            "A layers × corruptions matrix of how much each layer's activations move "
            "when a specific structural property is destroyed in the input, forming a "
            "per-model signature whose *shape across depth* is what gets compared across "
            "models — never the raw magnitudes, which are not calibrated between "
            "corruptions."),
        where="Perturbation & patching's sensitivity heatmap and cross-model agreement."),
    "patching": Term(
        term="Activation patching",
        definition=(
            "Writing a model's own cached clean activations back into its corrupted "
            "forward pass, one layer (and optionally one window) at a time, so that "
            "recovered forecast quality localizes where a property is causally carried; "
            "always within one model, never transplanted between models."),
        where="Perturbation & patching's restoration curves and layer × window heatmaps."),
    "ablation": Term(
        term="Mean ablation (head / MLP)",
        definition=(
            "Replacing one attention head's or MLP's output with its average over the "
            "batch and measuring the forecast damage, which ranks components by how "
            "load-bearing they are without needing to know what they compute."),
        where="Attention analysis' head and MLP ΔMASE rankings."),
    "periodicity-head": Term(
        term="Periodicity head",
        definition=(
            "An attention head that puts more mass than a uniform-baseline share at "
            "multiples of a series' dominant seasonal lag — the time-series analogue of "
            "an induction head, identified by lag profile rather than by what it is "
            "named."),
        where="Attention analysis' head taxonomy."),
    "token-width": Term(
        term="Token width",
        definition=(
            "How many timesteps one of a model's tokens covers (about 32 for a patch "
            "tokenizer, 1 for a per-timestep quantizer), which sets the finest lag that "
            "model's attention can express and is multiplied into every plotted lag axis "
            "so a lag of 300 means the same 300 timesteps for every model."),
        where="Attention analysis' lag axes; the fairness card's finest-resolvable-lag row."),
    "diagonal-hit-fraction": Term(
        term="Diagonal-hit fraction (alignment)",
        definition=(
            "The share of probe windows where injecting an impulse at window *w* "
            "perturbs aligned window *w* most, verifying empirically that the token-to-"
            "time mapping an adapter declares is true for the installed library version; "
            "below 1.0 at late layers is normal, near zero anywhere is a broken mapping."),
        where="The Alignment & Provenance panel; run by a gate at extraction time."),
    "coverage-fraction": Term(
        term="Captured FLOP fraction (coverage)",
        definition=(
            "The share of a model's real per-forecast computation that the captured "
            "layers actually perform, measured rather than assumed — an encoder-only "
            "capture surface or a strided capture leaves the rest unobserved, and any "
            "depth-located claim about a model below 90% carries that qualifier "
            "automatically."),
        where="The fairness card; Cost and capacity; auto-appended to depth-located findings."),
    "dead-feature": Term(
        term="Dead feature",
        definition=(
            "A dictionary atom in a trained sparse autoencoder that never activates on "
            "any evaluation row, so it contributes nothing and inflates any statistic "
            "computed over \"all features\"; the dead rate is reported because a mostly-"
            "dead dictionary can otherwise look like a healthy one."),
        where="Sparse features (SAE) — dictionary health, next to reconstruction fidelity."),
    "ground-truth-alignment": Term(
        term="Ground-truth feature alignment",
        definition=(
            "The correlation between a learned sparse feature's activation and a known "
            "generator parameter (a seasonal period, a trend order, an anomaly flag), "
            "which is this repo's substitute for the subjective max-activating-example "
            "judgment usual in SAE work; because each feature is matched to its *best* "
            "of many candidate fields, it is read against a label-permutation null."),
        where="Sparse features (SAE)."),
    "channel-trend": Term(
        term=_describe_term("trend").label,
        definition=_channel_glossary_definition("trend"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-seasonal": Term(
        term=_describe_term("seasonal").label,
        definition=_channel_glossary_definition("seasonal"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-spectral-centroid": Term(
        term=_describe_term("spectral_centroid").label,
        definition=_channel_glossary_definition("spectral_centroid"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-level": Term(
        term=_describe_term("level").label,
        definition=_channel_glossary_definition("level"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-dispersion": Term(
        term=_describe_term("dispersion").label,
        definition=_channel_glossary_definition("dispersion"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-horizon-near": Term(
        term=_describe_term("horizon_shape_near").label,
        definition=_channel_glossary_definition("horizon_shape_near"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-horizon-far": Term(
        term=_describe_term("horizon_shape_far").label,
        definition=_channel_glossary_definition("horizon_shape_far"),
        where=_CHANNEL_BATTERY_WHERE),
    "channel-flatness": Term(
        term=_describe_term("flatness").label,
        definition=_channel_glossary_definition("flatness"),
        where=_CHANNEL_BATTERY_WHERE),
    # "mase" is already covered above (the general forecast-quality metric);
    # not duplicated here as a ninth "channel-mase" entry.
    "ami": Term(
        term="AMI (adjusted mutual information)",
        definition=(
            "How much two clusterings of the same series agree, corrected for the "
            "agreement expected by chance, used to ask whether two models carve the "
            "benchmark into similar groups without requiring their cluster labels to "
            "correspond."),
        where="Activation clusters."),
    "series-resampling": Term(
        term="The series as the resampling unit",
        definition=(
            "Every bootstrap and every train/test split in this repo resamples whole "
            "series, never windows, because windows within one series are strongly "
            "dependent and every model scores the same series — splitting on windows "
            "would make every interval far too narrow."),
        where="Every confidence interval and p-value in the report."),
    "holm": Term(
        term="Holm correction",
        definition=(
            "The multiple-comparison adjustment applied across families so that "
            "testing several groups at once does not manufacture a significant result; "
            "a claimed strength needs both a corrected p below alpha and a confidence "
            "interval excluding zero."),
        where="Behavioral profile; Private benchmark confirmation."),
    "registered-hypothesis": Term(
        term="Registered hypothesis",
        definition=(
            "A dev-corpus finding written down *before* the private corpus is opened; "
            "only registered hypotheses can be confirmed, and a finding discovered "
            "afterwards cannot be rescued by the confirmation stage no matter how "
            "strong it looks."),
        where="Private benchmark confirmation; the \"[exploratory]\" prefix marks everything that is not one."),
    "minimal-sufficient-set": Term(
        term="Minimal sufficient set",
        definition=(
            "The smallest set of a model's own attention heads found by a greedy search "
            "whose patching restores most of a corruption's damage — sufficient because "
            "patching it works, but never proven the unique or provably smallest such "
            "set, since a greedy search only ever finds *a* small set, not *the* "
            "smallest."),
        where="The seasonality circuit."),
    "necessity": Term(
        term="Necessity",
        definition=(
            "Whether ablating a candidate head set on an otherwise-clean forecast "
            "damages it toward the fully-corrupted level — the complement of "
            "sufficiency, and required alongside it because either alone is a weaker "
            "claim than both together."),
        where="The seasonality circuit."),
    "sufficiency": Term(
        term="Sufficiency",
        definition=(
            "Whether patching a candidate head set's clean values into an otherwise-"
            "corrupted forecast restores it toward the clean level — the complement of "
            "necessity."),
        where="The seasonality circuit."),
    "path-patching": Term(
        term="Path patching",
        definition=(
            "Decomposing one head's total causal effect into a direct part (measured "
            "with every other head in the found set frozen at its clean value) plus a "
            "part routed through each other head individually, then checking whether "
            "the parts sum back to the total — a check that can fail even for a "
            "correctly-implemented decomposition, if the heads genuinely interact "
            "nonlinearly rather than contributing independent, additive paths."),
        where="The seasonality circuit."),
    "reach": Term(
        term="Reach",
        definition=(
            "Whether a patched-in intervention (a feature direction, a layer's "
            "activations) actually changes anything the forecast head reads, checked "
            "before any effect is trusted — confirmed by two controls: patching a "
            "layer into itself must move the forecast by exactly 0.0, and patching an "
            "earlier layer into a later read must move it by something nonzero; a "
            "clean, flat, entirely plausible-looking curve can still mean the "
            "intervention never reached anywhere the model reads (`CLAUDE.md` sec "
            "11.42)."),
        where="SAE causal channel testing (Component A)."),
    "random-direction null": Term(
        term="Random-direction null",
        definition=(
            "The same causal channel battery re-run on directions drawn at random "
            "from the same activation space instead of a trained SAE feature's "
            "decoder direction, so a channel response is only quotable once it is "
            "compared against how large a response chance alone produces, not "
            "against zero."),
        where="SAE causal channel testing (Component A)."),
    "role": Term(
        term="Role",
        definition=(
            "A cluster of a single model's SAE features that share a similar "
            "structural signature and causal response profile, named from its "
            "dominant response channel and its best-matching ground-truth field — a "
            "description of that model's own feature space, not a claim that any "
            "other model has an equivalent role until cross-model matching (see "
            "response fingerprint) has actually checked."),
        where=(
            "Superseded by concept clustering (ROADMAP.md sec 30, "
            "2026-09-11) -- see the concept cards and concept universality "
            "table instead; the roles table/role cards this pointer named "
            "are no longer rendered.")),
    "response fingerprint": Term(
        term="Response fingerprint",
        definition=(
            "A feature's or role's vector of null-normalized responses across the "
            "causal channel battery, used as the primary signal for matching one "
            "model's roles against another's by cosine similarity — a match rate "
            "computed this way is never rendered without its own untrained-twin "
            "floor beside it, because two architecturally similar but never-"
            "jointly-trained models can already look substantially 'matched' by "
            "chance."),
        where="Cross-model role correspondence table (Component C)."),
    "concept": Term(
        term="Concept",
        definition=(
            "A group of SAE features from one model's dictionary whose removal "
            "moves the forecast in the same way (the same channels, the same "
            "direction), found by clustering each feature's ablation fingerprint "
            "and admitted only when every group has at least a minimum number of "
            "members, so a dictionary whose causal features do not group is "
            "reported as non-modular rather than forced into concepts."),
        where="The SAE section's concept cards; produced by the concepts stage."),
    "transfer": Term(
        term="Transfer",
        definition=(
            "Whether another model separates the same series a concept fires "
            "hardest on, scored as the best of that model's features against "
            "random series sets drawn with the same mix of series kinds, and "
            "called reciprocal only when that feature's own top series are in "
            "turn separated by the original concept."),
        where="The SAE section's concept universality table; produced by the concepts stage."),
    "concept atlas": Term(
        term="Concept atlas",
        definition=(
            "A grouping of causal SAE features pooled across every model and "
            "clustered by ablation-fingerprint similarity, so that every pair inside "
            "a group clears a minimum cosine, groups below the minimum size stay "
            "unassigned, and a group can span several models (unlike a plain "
            "'concept', which stays within one model's dictionary)."),
        where="The SAE section's concept atlas figure and table; produced by the concepts stage."),
    "sharing class": Term(
        term="Sharing class",
        definition=(
            "Whether an atlas concept's models also agree on the INPUTS that drive "
            "it, not just its effect: 'shared' when every model in it is connected "
            "through agreeing cross-model input pairs, 'partially shared' when only "
            "some pairs agree, 'convergent' when none do, and 'single-model' when "
            "only one model holds the concept at all."),
        where="Model comparison — the sharing map and every concept card; produced by `sae/concept_profiles.py`."),
    "convergent concept": Term(
        term="Convergent concept",
        definition=(
            "An atlas concept whose member models share the same causal EFFECT on "
            "the forecast but fire on different, unrelated INPUTS per model — a "
            "measured finding about this run's own data, not a weaker or failed "
            "version of a fully shared concept."),
        where="Model comparison — sharing-class column and Q1's answer box."),
    "provenance-driven": Term(
        term="Provenance-driven",
        definition=(
            "Said of a concept's part when what best explains which series it "
            "fires on is a fact about how the benchmark was BUILT (which tier or "
            "generator produced a series) rather than a structural property of the "
            "time series itself, so 'fires on sequential_par series' is read as a "
            "property of the corpus, not of time series in general."),
        where="Model comparison — every input-profile table; produced by `sae/concept_profiles.py`."),
    "evidence ladder": Term(
        term="Evidence ladder",
        definition=(
            "The six rungs a cross-model concept claim climbs, from weakest to "
            "strongest: same forecast effect, same driving inputs, reproducible "
            "across an independent SAE seed, other models' own dictionaries select "
            "the same inputs, the same causal effect measured on the same inputs, "
            "and confirmed on a sealed private corpus — a concept's verdict names "
            "the highest rung it actually reaches, and a rung nothing in this run "
            "measures renders 'not measured', never a failed rung."),
        where="Model comparison — the verdict table; produced by `report/derived.py::concept_verdicts`."),
}


def get(name: str) -> Term:
    """Look up one glossary term by key.

    Raises `KeyError` rather than returning a placeholder, for the same reason
    `stage_docs.get` does (`CLAUDE.md` sec 2.5): a term referenced somewhere
    but never defined should fail visibly at render time, not render an empty
    entry that reads as if the definition were deliberately blank.
    """
    if name in GLOSSARY:
        return GLOSSARY[name]
    raise KeyError(
        f"no glossary Term for {name!r} -- add one to "
        f"tsfm_lens/glossary.py:GLOSSARY (ROADMAP.md sec 21 J3)")


def terms() -> list:
    """Every term, alphabetized by display name rather than by dict key.

    A lookup table is read by eye, so display order is the alphabetical one a
    reader expects -- the dict's own key order groups related concepts, which
    is convenient for editing and wrong for looking something up.
    """
    return sorted(GLOSSARY.values(), key=lambda t: t.term.lower())


def render_markdown() -> str:
    """Render the whole glossary as a Markdown section.

    The single function both `render_glossary.py` (the README generator) and
    its drift-check test call, so the README text is a pure function of
    `GLOSSARY` and can never be hand-edited out of sync.
    """
    lines = ["## Glossary", "",
             "The recurring vocabulary of this repo's report, one sentence each "
             "(`ROADMAP.md` sec 21 J3), generated from `tsfm_lens/glossary.py` by "
             "`render_glossary.py` -- edit that file, not this section, and re-run "
             "the script to update it. The report renders the same entries under "
             "its \"How to read this report\" preamble.", ""]
    for t in terms():
        lines += [f"**{t.term}.** {t.definition}", "",
                  f"*Where it appears:* {t.where}", ""]
    return "\n".join(lines).rstrip() + "\n"
