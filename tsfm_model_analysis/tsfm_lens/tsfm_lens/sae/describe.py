"""Grounded narrator -- turn a measured SAE feature/role record into one
readable English sentence, with a hard guard against fabrication.

Why this exists
---------------
`sae/roles.py` derives every role's name from the numbers rather than
letting a human write one, which is the right discipline and produces
strings like `far-horizon disperser (up) . ar_coeff_sum`. Those are
faithful and close to unreadable. This module adds a *presentation* layer
on top -- never a measurement layer: it verbalizes an evidence packet that
was already computed elsewhere, and it is allowed to say nothing that is
not in that packet.

Why the guard is the substance, not the model
---------------------------------------------
A fluent sentence asserting something unmeasured is strictly worse than the
machine string it replaces, because it reads like a finding. `CLAUDE.md`
sec 2.5's "degrade gracefully and loudly" applies with unusual force here:
the failure mode of a language model is not a crash, it is a plausible
paragraph. So generation is treated as an *untrusted proposal* and
`check_text` is the acceptance test:

1. Every domain term in the output must map to at least one concept the
   packet actually carries (a channel in `Evidence.channels`, a structural
   field in `Evidence.structural_field`/`top3_structural`, or a family in
   `Evidence.exemplar_families`). `DOMAIN_TERMS` is the fixed, auditable
   term -> concept table this is checked against; a term absent from that
   table is invisible to the guard, so the table is deliberately wider than
   any one packet's vocabulary -- it exists to catch the words the model
   might reach for, not to describe the words it should.
2. Causal language is rejected outright when `Evidence.clears_null` is
   False. A feature whose every channel sat inside the random-direction
   null has a correlate, not an effect.
3. Any digit sequence must be a rendering of a number in the packet. The
   prompt asks for no numbers at all; this catches the case where it emits
   one anyway.
4. Length, markdown, scaffolding and placeholder checks.

On rejection the packet is re-sent with the rejection reason appended as a
corrective turn (which is what makes a retry under greedy decoding produce
a *different* answer -- same decode, different prompt). After
`MAX_ATTEMPTS` the deterministic, LLM-free `machine_fallback` renders
instead, and `Description.accepted` is False with the reason kept. The
fallback is built from the same gloss tables the guard reads, and
`_assert_glosses_self_consistent` checks at import time that every gloss
would itself pass the guard -- a fallback the guard would reject is a bug,
not a safety net.

Determinism
-----------
Greedy decode (`do_sample=False`, one beam), a fixed seed, a pinned
checkpoint revision, and `MODEL_ID`/`REVISION` recorded on every
`Description` so a rendered sentence is auditable back to what produced it.
`describe` is `describe_batch` of one, so the single and batch paths cannot
diverge in prompt construction or guarding; batch size itself can still
move a token under left-padding in bf16, which is measured rather than
asserted (see this module's tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _dc_field

from ..utils import log

__all__ = [
    "MODEL_ID", "REVISION", "MAX_ATTEMPTS", "CHANNEL_GLOSS", "FIELD_GLOSS",
    "DOMAIN_TERMS", "CAUSAL_TERMS", "CLEARED_CLAIM_TERMS",
    "UNTESTED_MARKERS",
    "FEW_SHOT_EXAMPLES", "SYSTEM_PROMPT",
    "Evidence", "Description", "Narrator",
    "load_narrator", "describe", "describe_batch", "machine_fallback",
    "check_text", "allowed_concepts", "render_evidence",
]


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Pinned via `huggingface_hub.model_info(MODEL_ID).sha` on 2026-09-01. A
# floating `main` would let the sentence a report renders change without
# anything in this repo changing (`CLAUDE.md` sec 11.8's shape).
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

MAX_ATTEMPTS = 3
SEED = 0
MAX_NEW_TOKENS = 64
MAX_WORDS = 40
MAX_SENTENCES = 2


# ---------------------------------------------------------------------------
# Evidence and result records.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    """Everything the narrator is permitted to mention. Anything else is fabrication.

    `channels` holds ONLY channels that cleared their random-direction null,
    keyed by `sae/response.py::CHANNELS` name, valued in multiples of that
    channel's own null p95 (signed -- the sign is the direction of the
    steered effect). `top3_structural` is `((field, rho, n), ...)`, the
    residualized structural correlates from
    `sae/ground_truth.py::best_ground_truth_matches_separated`.
    """

    kind: str
    model: str
    layer: str
    ident: str
    channels: dict = _dc_field(default_factory=dict)
    structural_field: str | None = None
    structural_rho: float | None = None
    structural_n: int | None = None
    top3_structural: tuple = ()
    n_atoms: int | None = None
    exemplar_families: tuple = ()
    exemplar_profile: tuple = ()
    """`((field, mean_over_firing_series, corpus_median), ...)`, structural only.

    Added 2026-09-04 on user review: "it is not clear what exactly each
    feature does as many have very similar names and activate for the same
    series." That was a measurement, not an impression -- within one target
    the top features routinely share a `structural_field` and overlap on
    exemplar families, so every field the narrator was licensed to mention
    was IDENTICAL across them and no honest description could distinguish
    them. This is the distinguishing evidence: for each field, what the
    series this feature actually fires hardest on measure, against what the
    corpus typically measures. Two features sharing a field still differ
    here whenever they fire on different ends of it.

    Structural fields only, and only fields `sae/ground_truth.py` did not
    refuse as inseparable from provenance (sec 11.48) -- a contrast on
    `generator_*` would say which generator wrote the series, which is
    corpus bookkeeping and exactly what sec 26 A1/A3 removed from the
    headline. Empty when the exemplar pass did not run, which keeps every
    packet built before this field existed byte-identical.
    """
    undirected_channels: frozenset = frozenset()
    """Channels that moved but whose DIRECTION the evidence does not support.

    The steering battery's per-channel `clears_null` tests an UNSIGNED
    effect (mean |delta| against the random-direction null's p95), while
    the number this packet carries is the SIGNED mean over that same null.
    The two answer different questions, and on
    `runs/full_report_run_4model` they disagree for 556 of 1394 cleared
    (feature, channel) cells -- down to a signed ratio of 0.01, where the
    sign of a near-zero mean is arbitrary. That is sec 11.54's defect at
    the feature site: a boolean saying "it moved this channel" beside a
    magnitude saying "how consistently in one direction", read as one
    claim, and the narrator duly wrote "increases the forecast's spread"
    off a signed mean 0.74 times its own null.

    A channel listed here keeps its place in `channels` -- it genuinely
    moved, and dropping it would assert the opposite falsehood (sec 11.37:
    absent is not the same as nothing) -- but its value is the UNSIGNED
    effect ratio, the quantity that actually cleared, and only the
    undirected verb is honest for it. Same treatment `CHANNEL_VERB`
    already gives a channel whose statistic has no direction; the
    difference is that this set is measured per feature rather than fixed
    per channel.
    """
    ablation_channels: dict = _dc_field(default_factory=dict)
    """Cleared channels from the ABLATION battery, in signed null units.

    `channels` above is the injection battery: a decoder direction steered
    INTO a clean forward pass, on series chosen without reference to whether
    the feature fires on them. This is the other intervention -- the feature
    zeroed OUT of the SAE's own reconstruction, on the series it fires
    hardest on -- so it answers a question the injection battery cannot: what
    this feature is contributing where it is actually active.

    🔴 SIGN CONVENTION, and it is not the artifact's. `sae/response.py`
    records `signed_effect` as the effect OF REMOVAL (ablated minus
    baseline); the value here is NEGATED, so it is the feature's own
    contribution and reads on the same scale and in the same direction as
    `channels`. Without that flip a direction word in the narrator's sentence
    would mean opposite things depending on which line licensed it, and
    nothing downstream could tell which -- the guard checks a word against a
    sign, not against a provenance.

    Empty when no ablation pass ran, which keeps every packet built before
    this field existed byte-identical.
    """
    ablation_measured: bool = False
    """True when the ablation battery ran for this target at all.

    Separate from `ablation_channels` being non-empty for exactly the reason
    `channels_measured` is separate from `clears_null` (`CLAUDE.md` sec
    11.37): a feature the battery scored and found nothing for, and a feature
    the battery never reached, are different states, and only the first
    licenses "removing it changes nothing".
    """
    clears_null: bool = False
    channels_measured: bool = True
    """False when NO causal channel battery was run for this target at all.

    Distinct from `clears_null=False`, which means the battery ran and
    nothing cleared. Collapsing the two makes an unmeasured feature claim a
    null comparison that never happened -- `CLAUDE.md` sec 11.37's rule that
    "absent" and "bad" must be different outcomes, which this field exists
    to keep separable all the way to the rendered sentence. Defaults True so
    every packet built before this field existed keeps its exact wording.
    """


@dataclass(frozen=True)
class Description:
    text: str
    accepted: bool
    reason: str
    attempts: int
    model_id: str
    revision: str


@dataclass
class Narrator:
    """Opaque handle returned by `load_narrator`. Held by the caller so the
    checkpoint is loaded once per process, not once per description."""

    tokenizer: object
    model: object
    device: str
    model_id: str = MODEL_ID
    revision: str = REVISION


# ---------------------------------------------------------------------------
# Vocabulary: the gloss tables the fallback writes with, and the term table
# the guard reads. Concepts are namespaced strings so a channel and a
# structural field that share a word stay distinguishable.
# ---------------------------------------------------------------------------

def channel_concept(name: str) -> str:
    return f"channel:{name}"


def field_concept(name: str) -> str:
    return f"field:{name}"


def family_concept(name: str) -> str:
    return f"family:{name}"


# Canonical English for each causal channel, used verbatim by
# `machine_fallback`. Each phrase must be recoverable by `DOMAIN_TERMS` as
# its own channel -- asserted at import time.
CHANNEL_GLOSS = {
    "trend": "the forecast's trend slope",
    "seasonal": "the forecast's seasonal magnitude",
    "spectral_centroid": "the forecast's spectral centroid",
    "level": "the forecast's overall level",
    "dispersion": "the forecast's spread",
    "horizon_shape_near": "the near horizon of the forecast",
    "horizon_shape_far": "the far horizon of the forecast",
    "mase": "forecast error",
    "flatness": "the forecast's flatness",
}

# `horizon_shape_*` is a mean ABSOLUTE deviation between the steered and
# baseline forecast over a third of the horizon (`sae/response.py::
# _horizon_shape`), so its sign carries "how much it moved", never "which
# way" -- "moves" is the only honest verb for it, and "increases"/
# "decreases" would be a claim the statistic cannot make.
CHANNEL_VERB = {
    "horizon_shape_near": ("moves", "moves"),
    "horizon_shape_far": ("moves", "moves"),
}
_DEFAULT_VERB = ("increases", "decreases")


def _verbs(ev, name: str) -> tuple:
    """The honest verb pair for one channel of one packet.

    Undirected either because the channel's own statistic carries no
    direction (`CHANNEL_VERB`) or because THIS packet's signed mean did not
    clear its own null (`Evidence.undirected_channels`).
    """
    if name in (getattr(ev, "undirected_channels", None) or frozenset()):
        return ("moves", "moves")
    return CHANNEL_VERB.get(name, _DEFAULT_VERB)

# Canonical English for each ground-truth field
# (`sae/ground_truth.py::_scalar_ground_truth` plus its one-hot dummies).
FIELD_GLOSS = {
    "trend_order": "trend order",
    "trend_scale": "trend scale",
    "n_seasonalities": "number of seasonal components",
    "seasonal_amplitude_max": "seasonal amplitude",
    "seasonal_period_dominant": "dominant seasonal period",
    "ar_order": "AR order",
    "ar_coeff_sum": "AR coefficient sum",
    "noise_scale": "noise scale",
    # "number of X", not bare "X": `_structural_clause` writes "a larger
    # {gloss}", so a bare plural noun reads "a larger changepoints".
    # `n_seasonalities` above already carries the article-compatible form.
    "n_changepoints": "number of changepoints",
    "n_anomalies": "number of anomalies",
    "has_random_walk": "a random walk component",
    "has_intermittency": "intermittency",
    "has_heteroskedastic": "heteroskedasticity",
    "tier_synthetic": "synthetic",
    "tier_realism_stress": "the realism-stress tier",
    "tier_real_derived": "real-derived",
    "generator_parametric": "parametric",
    "generator_random_parametric": "random parametric",
    "generator_mixture": "mixture",
    "generator_block_bootstrap": "block bootstrap",
    "generator_sequential_par": "sequential PAR",
    "archetype_trend_dominant": "trend dominant",
    "archetype_seasonal_dominant": "seasonal dominant",
    "archetype_multi_seasonal_complex": "multi seasonal complex",
    "archetype_regime_switching": "regime switching",
    "archetype_ar_colored_noise": "AR colored noise",
    "archetype_anomaly_heavy": "anomaly heavy",
    "archetype_clean_low_noise": "clean low noise",
    "archetype_noisy_chaotic": "noisy chaotic",
    "archetype_random_walk_drift": "random walk drift",
    "archetype_intermittent_bursts": "intermittent bursts",
    "archetype_amplitude_modulated": "amplitude modulated",
    "archetype_nonsinusoidal_seasonal": "nonsinusoidal seasonal",
}

# A boolean/dummy field reads as "series that HAVE this", a scalar field as
# "series with a higher/lower value of this".
_BOOLEAN_FIELD_PREFIXES = ("has_", "tier_", "generator_", "archetype_")

FAMILY_GLOSS = {
    "parametric": "parametric",
    "random_parametric": "random parametric",
    "mixture": "mixture",
    "block_bootstrap": "block bootstrap",
    "sequential_par": "sequential PAR",
}


def _terms(*pairs) -> dict:
    """Build the term table, merging concept tuples for any term that
    legitimately maps to more than one (a word like "seasonal" is a channel
    AND several structural fields; a mention of it is fabrication only when
    the packet carries NONE of them)."""
    out: dict = {}
    for term, concepts in pairs:
        prior = out.get(term, ())
        out[term] = prior + tuple(c for c in concepts if c not in prior)
    return out


_C = channel_concept
_F = field_concept
_FAM = family_concept

_SEASONAL = (_C("seasonal"), _F("n_seasonalities"), _F("seasonal_amplitude_max"),
             _F("seasonal_period_dominant"), _F("archetype_seasonal_dominant"),
             _F("archetype_multi_seasonal_complex"),
             _F("archetype_nonsinusoidal_seasonal"))
_TRENDY = (_C("trend"), _F("trend_order"), _F("trend_scale"),
           _F("archetype_trend_dominant"))
_NOISY = (_F("noise_scale"), _F("archetype_noisy_chaotic"),
          _F("archetype_clean_low_noise"), _F("archetype_ar_colored_noise"))
_WALKY = (_F("has_random_walk"), _F("archetype_random_walk_drift"))
_HETERO = (_F("has_heteroskedastic"), _F("archetype_amplitude_modulated"))
_INTERMITTENT = (_F("has_intermittency"), _F("archetype_intermittent_bursts"),
                 _C("flatness"))
_ANOMALY = (_F("n_anomalies"), _F("archetype_anomaly_heavy"))
_CHANGEPOINT = (_F("n_changepoints"), _F("archetype_regime_switching"))
_AR = (_F("ar_order"), _F("ar_coeff_sum"), _F("archetype_ar_colored_noise"))

# The fixed domain-term -> concept table the guard checks against. Terms are
# matched case-insensitively on a hyphen-normalized copy of the text, with
# word boundaries, so "far horizon" matches "far-horizon" while "seasonal"
# does not match inside "seasonality" (which has its own entry).
DOMAIN_TERMS = _terms(
    ("trend", _TRENDY),
    ("trends", _TRENDY),
    ("trending", _TRENDY),
    ("trend slope", _TRENDY),
    ("slope", (_C("trend"),)),
    ("upward drift", _TRENDY + _WALKY),
    ("seasonal", _SEASONAL),
    ("seasonality", _SEASONAL),
    ("seasonalities", _SEASONAL),
    ("seasonal magnitude", _SEASONAL),
    ("season", _SEASONAL),
    ("seasons", _SEASONAL),
    ("periodic", _SEASONAL),
    ("periodicity", _SEASONAL),
    ("period", _SEASONAL),
    ("cycle", _SEASONAL),
    ("cycles", _SEASONAL),
    ("cyclic", _SEASONAL),
    ("cyclical", _SEASONAL),
    ("spectral centroid", (_C("spectral_centroid"),)),
    ("spectral", (_C("spectral_centroid"),)),
    ("frequency", (_C("spectral_centroid"),) + _SEASONAL),
    ("frequencies", (_C("spectral_centroid"),) + _SEASONAL),
    ("high frequency", (_C("spectral_centroid"),)),
    ("low frequency", (_C("spectral_centroid"),)),
    ("level", (_C("level"),)),
    ("levels", (_C("level"),)),
    ("overall level", (_C("level"),)),
    ("mean level", (_C("level"),)),
    ("baseline level", (_C("level"),)),
    ("offset", (_C("level"),)),
    ("dispersion", (_C("dispersion"),)),
    ("spread", (_C("dispersion"), _C("horizon_shape_near"), _C("horizon_shape_far"))),
    ("disperses", (_C("dispersion"),)),
    ("disperser", (_C("dispersion"),)),
    ("widens", (_C("dispersion"), _C("horizon_shape_near"), _C("horizon_shape_far"))),
    ("widen", (_C("dispersion"), _C("horizon_shape_near"), _C("horizon_shape_far"))),
    ("narrows", (_C("dispersion"), _C("horizon_shape_near"), _C("horizon_shape_far"))),
    ("narrow", (_C("dispersion"), _C("horizon_shape_near"), _C("horizon_shape_far"))),
    ("variance", (_C("dispersion"),) + _NOISY),
    ("volatility", (_C("dispersion"),) + _HETERO),
    ("near horizon", (_C("horizon_shape_near"),)),
    ("early horizon", (_C("horizon_shape_near"),)),
    ("short term", (_C("horizon_shape_near"),)),
    ("short horizon", (_C("horizon_shape_near"),)),
    ("far horizon", (_C("horizon_shape_far"),)),
    ("late horizon", (_C("horizon_shape_far"),)),
    ("long term", (_C("horizon_shape_far"),)),
    ("long horizon", (_C("horizon_shape_far"),)),
    ("far end", (_C("horizon_shape_far"),)),
    ("horizon", (_C("horizon_shape_near"), _C("horizon_shape_far"))),
    ("mase", (_C("mase"),)),
    ("forecast error", (_C("mase"),)),
    ("error", (_C("mase"),)),
    ("errors", (_C("mase"),)),
    ("accuracy", (_C("mase"),)),
    ("flatness", (_C("flatness"),) + _INTERMITTENT),
    ("flat", (_C("flatness"),) + _INTERMITTENT),
    ("flatter", (_C("flatness"),) + _INTERMITTENT),
    ("constant", (_C("flatness"),)),
    ("trend_order", (_F("trend_order"),)),
    ("trend order", (_F("trend_order"),)),
    ("trend_scale", (_F("trend_scale"),)),
    ("trend scale", (_F("trend_scale"),)),
    ("n_seasonalities", (_F("n_seasonalities"),)),
    ("number of seasonal components", _SEASONAL),
    ("seasonal_amplitude_max", (_F("seasonal_amplitude_max"),)),
    ("seasonal amplitude", (_F("seasonal_amplitude_max"),)),
    ("amplitude", (_F("seasonal_amplitude_max"),) + _HETERO),
    ("seasonal_period_dominant", (_F("seasonal_period_dominant"),)),
    ("dominant seasonal period", (_F("seasonal_period_dominant"),)),
    ("ar_order", (_F("ar_order"),)),
    ("ar order", (_F("ar_order"),)),
    ("ar_coeff_sum", (_F("ar_coeff_sum"),)),
    ("ar coefficient sum", (_F("ar_coeff_sum"),)),
    ("ar coefficient", _AR),
    ("ar coefficients", _AR),
    ("autoregressive", _AR),
    ("ar", _AR),
    ("noise_scale", (_F("noise_scale"),)),
    ("noise scale", (_F("noise_scale"),)),
    ("noise", _NOISY),
    ("noisy", _NOISY),
    ("n_changepoints", (_F("n_changepoints"),)),
    ("changepoint", _CHANGEPOINT),
    ("changepoints", _CHANGEPOINT),
    ("level shift", _CHANGEPOINT),
    ("level shifts", _CHANGEPOINT),
    ("regime", _CHANGEPOINT),
    ("regimes", _CHANGEPOINT),
    ("regime switching", _CHANGEPOINT),
    ("n_anomalies", (_F("n_anomalies"),)),
    ("anomaly", _ANOMALY),
    ("anomalies", _ANOMALY),
    ("spike", _ANOMALY),
    ("spikes", _ANOMALY),
    ("spiky", _ANOMALY),
    ("outlier", _ANOMALY),
    ("outliers", _ANOMALY),
    ("has_random_walk", (_F("has_random_walk"),)),
    ("random walk", _WALKY),
    ("random walk component", _WALKY),
    ("random walk drift", _WALKY),
    ("unit root", _WALKY),
    ("has_intermittency", (_F("has_intermittency"),)),
    ("intermittency", _INTERMITTENT),
    ("intermittent", _INTERMITTENT),
    ("intermittent bursts", _INTERMITTENT),
    ("sparse", _INTERMITTENT),
    ("zero inflated", _INTERMITTENT),
    ("has_heteroskedastic", (_F("has_heteroskedastic"),)),
    ("heteroskedastic", _HETERO),
    ("heteroskedasticity", _HETERO),
    ("amplitude modulated", _HETERO),
    ("changing noise amplitude", _HETERO),
    ("tier_synthetic", (_F("tier_synthetic"),)),
    ("tier_realism_stress", (_F("tier_realism_stress"),)),
    ("tier_real_derived", (_F("tier_real_derived"),)),
    ("synthetic", (_F("tier_synthetic"),)),
    ("realism stress", (_F("tier_realism_stress"),)),
    ("real derived", (_F("tier_real_derived"),)),
    # "parametric" is a substring of "random parametric", so the bare word is
    # genuinely ambiguous between the two generators and maps to both -- the
    # import-time gloss check caught this by rejecting FIELD_GLOSS
    # ['generator_random_parametric'] against its own packet.
    ("parametric", (_F("generator_parametric"), _FAM("parametric"),
                    _F("generator_random_parametric"), _FAM("random_parametric"))),
    ("random parametric", (_F("generator_random_parametric"), _FAM("random_parametric"))),
    ("generator_parametric", (_F("generator_parametric"),)),
    ("generator_random_parametric", (_F("generator_random_parametric"),)),
    ("generator_mixture", (_F("generator_mixture"),)),
    ("generator_block_bootstrap", (_F("generator_block_bootstrap"),)),
    ("generator_sequential_par", (_F("generator_sequential_par"),)),
    ("mixture", (_F("generator_mixture"), _FAM("mixture"))),
    ("block bootstrap", (_F("generator_block_bootstrap"), _FAM("block_bootstrap"))),
    ("block_bootstrap", (_F("generator_block_bootstrap"), _FAM("block_bootstrap"))),
    ("sequential par", (_F("generator_sequential_par"), _FAM("sequential_par"))),
    ("sequential_par", (_F("generator_sequential_par"), _FAM("sequential_par"))),
    ("archetype_trend_dominant", (_F("archetype_trend_dominant"),)),
    ("trend dominant", (_F("archetype_trend_dominant"),)),
    ("archetype_seasonal_dominant", (_F("archetype_seasonal_dominant"),)),
    ("seasonal dominant", (_F("archetype_seasonal_dominant"),)),
    ("archetype_multi_seasonal_complex", (_F("archetype_multi_seasonal_complex"),)),
    ("multi seasonal complex", (_F("archetype_multi_seasonal_complex"),)),
    ("archetype_regime_switching", (_F("archetype_regime_switching"),)),
    ("archetype_ar_colored_noise", (_F("archetype_ar_colored_noise"),)),
    ("ar colored noise", (_F("archetype_ar_colored_noise"),)),
    ("archetype_anomaly_heavy", (_F("archetype_anomaly_heavy"),)),
    ("anomaly heavy", (_F("archetype_anomaly_heavy"),)),
    ("archetype_clean_low_noise", (_F("archetype_clean_low_noise"),)),
    ("clean low noise", (_F("archetype_clean_low_noise"),)),
    ("archetype_noisy_chaotic", (_F("archetype_noisy_chaotic"),)),
    ("noisy chaotic", (_F("archetype_noisy_chaotic"),)),
    ("chaotic", (_F("archetype_noisy_chaotic"),)),
    ("archetype_random_walk_drift", (_F("archetype_random_walk_drift"),)),
    ("archetype_intermittent_bursts", (_F("archetype_intermittent_bursts"),)),
    ("archetype_amplitude_modulated", (_F("archetype_amplitude_modulated"),)),
    ("archetype_nonsinusoidal_seasonal", (_F("archetype_nonsinusoidal_seasonal"),)),
    ("nonsinusoidal seasonal", (_F("archetype_nonsinusoidal_seasonal"),)),
)

# Words that assert a causal relationship. Licensed only when at least one
# channel cleared its random-direction null.
CAUSAL_TERMS = (
    "cause", "causes", "caused", "causing", "causal", "causally", "causality",
    "drive", "drives", "driven", "driving",
    "control", "controls", "controlling",
    "determine", "determines", "determining",
    "responsible for", "induces", "induce", "inducing",
    "forces", "governs", "dictates",
    "steers", "steer", "steering",
)

# Claims NO evidence packet can ever license, because nothing in
# `sae/response.py::CHANNELS` measures the thing being asserted. Every entry
# was OBSERVED in a real generation against
# `runs/full_report_run_large/sae/` -- this list is empirical and is
# expected to grow, which is the honest description of a blocklist: it
# cannot be complete, so it is the guard's weakest line rather than its
# first (the concept check above is what does the load-bearing work).
UNLICENSED_TERMS = (
    # A time shift. The battery has no channel for "the forecast happened
    # earlier or later" -- `horizon_shape_near`/`far` are the mean ABSOLUTE
    # change over a third of the horizon, so "shifts the forecast to the
    # future" is a claim about a quantity that was never computed.
    "to the future", "into the future", "toward the future", "in the future",
    "closer to the present", "closer to the past", "toward earlier",
    "earlier periods", "later periods", "forward in time", "backward in time",
    "time shift", "shifts time", "extends the horizon", "shortening its span",
    # Anthropomorphic or performance claims about the model itself.
    "excels", "works best", "performs best", "best at", "good at", "better at",
    "handles", "understands", "knows", "learns", "reality", "realistic",
    "real world", "burstiness", "parabolic",
    # Exclusivity over SERIES: nothing here is measured against every
    # alternative population.
    "only", "exclusively", "solely", "always", "never",
    "every series", "all series", "entirely", "purely",
    # A multiple written as a word is still a number, and the prompt asks
    # for none.
    "twice", "half", "double", "triple", "tenfold", "order of magnitude",
    "orders of magnitude",
)

# Comparative vocabulary for the `exemplar_profile` clause -- "this feature's
# own top series measure X above/below what the corpus typically shows".
#
# Needed because the concept check cannot see this claim. That check licenses
# a FIELD; the profile clause is an assertion ABOUT a field the packet may
# already license for a different reason (`other correlates`), so a narrator
# with no profile at all can compose a fully-licensed sentence claiming a
# contrast that was never measured -- observed on the first live run, where a
# packet reading "its own top series measure: not measured" was described as
# "firing on series whose seasonal swings are above and whose changes are
# fewer than typical". Every concept in that sentence was licensed. The
# comparison was invented.
#
# Same shape as `UP_VERBS`/`DOWN_VERBS` one level over: a direction word is a
# claim about a measurement's sign, so it needs a measurement pointing that
# way. Deliberately disjoint from those two sets -- see `render_evidence`.
PROFILE_ABOVE_TERMS = (
    "above", "exceeds", "exceed", "exceeding", "greater", "larger", "more",
    "longer", "stronger", "unusually high", "unusually large", "unusually many",
    "above average", "higher than typical", "higher than usual",
)
PROFILE_BELOW_TERMS = (
    "below", "fewer", "less", "smaller", "shorter", "weaker",
    "unusually low", "unusually small", "unusually few", "below average",
    "lower than typical", "lower than usual",
)

# Direction verbs. Licensed only when the packet holds a channel whose sign
# actually points that way. `horizon_shape_near`/`far` are excluded from the
# signed set by `CHANNEL_VERB` membership -- their statistic is an absolute
# deviation, so it can support "moves" and nothing else.
UP_VERBS = (
    "increase", "increases", "increasing", "raise", "raises", "raising",
    "boost", "boosts", "boosting", "amplify", "amplifies", "amplifying",
    "grow", "grows", "growing", "widen", "widens", "widening",
    "broaden", "broadens", "broadening", "steepen", "steepens", "steepening",
    "lift", "lifts", "lifting", "heighten", "heightens", "upward", "upwards",
)
DOWN_VERBS = (
    "decrease", "decreases", "decreasing", "lower", "lowers", "lowering",
    "reduce", "reduces", "reducing", "shrink", "shrinks", "shrinking",
    "narrow", "narrows", "narrowing", "dampen", "dampens", "dampening",
    "suppress", "suppresses", "suppressing", "tighten", "tightens", "tightening",
    "flatten", "flattens", "flattening", "drop", "drops", "dropping",
    "diminish", "diminishes", "diminishing", "weaken", "weakens", "weakening",
    "downward", "downwards",
)

# Exclusivity over EFFECTS -- "and nothing else moved". False whenever the
# packet lists a channel that cleared (the live run wrote "without any other
# measurable effects" for a feature with nine of them), and exactly true
# when it lists none, so this is conditional rather than banned outright.
# Banning it outright cost 8 correct sentences on the first tightened run.
NO_OTHER_EFFECT_TERMS = (
    "no other", "any other", "all measures", "in any way", "in any direction",
    "nothing but",
)

# Quality verdicts. "worse" is a real, licensed reading of `mase` going up
# and "better" of it going down -- but only when `mase` is in the packet at
# all, and only in the direction it actually moved. The live run produced
# "slightly improves the trend", which is a verdict on a channel that has no
# better or worse.
QUALITY_WORSE = ("worsens", "worsen", "worsening", "worse", "degrades",
                 "degrade", "degrading", "hurts", "hurt")
QUALITY_BETTER = ("improves", "improve", "improving", "improved", "better",
                  "enhances", "enhance", "enhancing", "helps", "help")

# Horizon nouns, for the proximity rule below.
_HORIZON_WORDS = ("near horizon", "far horizon", "early horizon", "late horizon",
                  "horizons", "horizon", "far end")
# Only ADVERBS and bare particles may follow a horizon noun in the
# reverse-order rule: a verb after the noun is usually about a different
# clause ("shifts the far and near horizons, increasing spread" is a
# legitimate sentence about dispersion), while "moves the far horizon
# upward" is not. The bare "up"/"down" entries are what let the rule see
# "pulls the near horizon down", which a live run produced three times and
# an adverb-only list let through.
_TRAILING_DIRECTION = ("upward", "upwards", "downward", "downwards", "higher",
                       "lower", "up", "down", "raising", "lowering")

# Nouns that name a SIGNED channel. A direction word sitting next to one of
# these is a claim about that channel, not about the horizon, so it is
# exempt from the reverse-order rule -- "reshapes the near horizon of the
# forecast, raising its level" is licensed whenever `level` is in the
# packet, and rejecting it would be the false-refusal failure CLAUDE.md
# sec 11.33/sec 11.35 warn is the expensive direction.
_SIGNED_CHANNEL_NOUNS = ("level", "levels", "trend", "trends", "slope",
                         "spread", "spreads", "dispersion", "range",
                         "seasonality", "seasonal", "error", "errors",
                         "centroid", "flatness")

# Denials of any effect. A contradiction of the packet when a channel did
# clear its null -- the mirror image of the causal-overreach rule.
NO_EFFECT_TERMS = (
    "no effect", "no measured effect", "no measurable", "no impact",
    "does nothing", "unchanged", "fails to move", "not move", "nowhere above",
    "without affecting", "negligible", "leaves the forecast alone",
)

# A claim that the null WAS cleared. Distinct from CAUSAL_TERMS, which are
# words that presuppose an effect ("causally", "drives"); these name the
# comparison itself and so are false whenever `clears_null` is False --
# including in the untested state, where the comparison never ran. Measured
# on real 1.5B output: an untested role drew "Not applicable, as the test
# cleared the null hypothesis", which asserts a clearing AND a verdict from
# a battery that did not execute (ROADMAP.md sec 26 C).
# The untested state is the ONE place the guard is an ALLOWLIST rather than a
# blacklist, and that is a measured decision, not a preference. The blacklist
# version (reject NO_EFFECT_TERMS when nothing was measured) was written
# against the three fabrications a real 1.5B run produced, passed on all
# three, and was then defeated on the re-measure by two fresh paraphrases --
# "does not measure any changes to the forecast" -- which assert the same
# false null through words no list had. A null can be phrased indefinitely
# many ways; the untested status can only be stated by saying it. So a
# description of an unmeasured feature has to CARRY one of these markers, and
# anything else falls back to the machine wording that always does
# (ROADMAP.md sec 26 C).
UNTESTED_MARKERS = (
    "not tested", "not been tested", "untested", "no test was", "no battery",
    "not measured whether", "was not measured", "were not measured",
    "not assessed", "not evaluated", "no measurement",
)

CLEARED_CLAIM_TERMS = (
    "cleared the null", "clears the null", "cleared its null", "clears its null",
    "beat the null", "beats the null", "above the null", "exceeds the null",
    "exceeded the null", "cleared the random-direction null",
    "rejected the null", "rejects the null", "significant effect",
)

# Scaffolding the model sometimes emits around an answer. Any of these makes
# the output unusable as a rendered sentence regardless of its content.
_SCAFFOLD_SUBSTRINGS = (
    "```", "{", "}", "[", "]", "<", ">", "|", "**", "##",
    "evidence", "description:", "answer:", "sentence:", "output:",
    "note:", "todo", "xxx", "placeholder", "n/a", "as an ai",
)
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*•]\s|\d+[.)]\s)")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<![0-9])[.!?]+(?![0-9])")


def _normalize(text: str) -> str:
    """Lowercase, hyphens and slashes to spaces, whitespace collapsed -- so
    "far-horizon" and "Far Horizon" both match the term "far horizon", while
    underscores survive because raw ground-truth field names contain them."""
    t = text.lower()
    t = re.sub(r"[-/‐‑–—]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _term_pattern(term: str):
    # The term is normalized the same way the TEXT is before matching. Every
    # call site searches `_normalize(...)` output, in which hyphens have
    # already become spaces -- so a hyphen left escaped in the pattern can
    # never match anything, and the term is silent decoration rather than a
    # guard. Found by the parametrized "every term is actually matched" test
    # on the first hyphenated term anyone added ("cleared the
    # random-direction null"); no pre-existing tuple carried one, so this
    # changes no current behavior and closes the hole for the next one.
    return re.compile(r"(?<![a-z0-9_])" + re.escape(_normalize(term)).replace(r"\ ", r"\s+")
                      + r"(?![a-z0-9_])")


_TERM_PATTERNS = {term: _term_pattern(term) for term in DOMAIN_TERMS}
_CAUSAL_PATTERNS = {term: _term_pattern(term) for term in CAUSAL_TERMS}
_CLEARED_CLAIM_PATTERNS = {term: _term_pattern(term)
                           for term in CLEARED_CLAIM_TERMS}
_UNTESTED_MARKER_PATTERNS = {term: _term_pattern(term)
                             for term in UNTESTED_MARKERS}
_UNLICENSED_PATTERNS = {t: _term_pattern(t) for t in UNLICENSED_TERMS}
_UP_PATTERNS = {t: _term_pattern(t) for t in UP_VERBS}
_DOWN_PATTERNS = {t: _term_pattern(t) for t in DOWN_VERBS}
_PROFILE_ABOVE_PATTERNS = {t: _term_pattern(t) for t in PROFILE_ABOVE_TERMS}
# Comparisons to the random-direction NULL, which are licensed by the
# channel evidence and are not claims about the corpus at all.
_NULL_COMPARISON = re.compile(
    r"\b(?:above|below|over|beyond|under)\s+"
    r"(?:the\s+)?(?:random[- ]direction\s+)?"
    r"(?:null|noise floor|chance)\b")
# A direction word is only a claim about the CORPUS when it is made against
# one. "strongest on series with a larger dominant seasonal period" is a
# claim about the SIGN OF RHO -- licensed by `structural_rho`, and the exact
# wording this module's own machine fallback writes; scanning bare direction
# words rejected it, i.e. the guard refused the sentence it falls back TO.
# So the scan runs only in a window around a corpus anchor, which is also
# the only phrasing `render_evidence` and the few-shot examples teach.
_CORPUS_ANCHOR = re.compile(r"\b(?:typical|typically|corpus|usual|usually|average)\b")
_ANCHOR_LOOKBEHIND = 80
_ANCHOR_LOOKAHEAD = 40


def _corpus_comparison_windows(text: str) -> str:
    """The parts of `text` that compare something to the corpus, joined.

    Empty when nothing is compared to the corpus at all -- which is the
    common case and must stay unscanned.
    """
    spans = []
    for hit in _CORPUS_ANCHOR.finditer(text):
        spans.append(text[max(0, hit.start() - _ANCHOR_LOOKBEHIND):
                          hit.end() + _ANCHOR_LOOKAHEAD])
    return " || ".join(spans)
_PROFILE_BELOW_PATTERNS = {t: _term_pattern(t) for t in PROFILE_BELOW_TERMS}
_NO_EFFECT_PATTERNS = {t: _term_pattern(t) for t in NO_EFFECT_TERMS}
# A raw artifact field name is correct but is not English, and it reads as
# a leak of the pipeline's internals into the prose the report renders.
# Only names that HAVE a gloss are refused, so a field this module has no
# English for still degrades to its raw name rather than to nothing.
_RAW_FIELD_PATTERNS = {f: _term_pattern(f) for f in FIELD_GLOSS if "_" in f}
_NO_OTHER_PATTERNS = {t: _term_pattern(t) for t in NO_OTHER_EFFECT_TERMS}
_QUALITY_WORSE_PATTERNS = {t: _term_pattern(t) for t in QUALITY_WORSE}
_QUALITY_BETTER_PATTERNS = {t: _term_pattern(t) for t in QUALITY_BETTER}


def _alt(words) -> str:
    return "|".join(re.escape(w).replace(r"\ ", r"\s+")
                    for w in sorted(words, key=len, reverse=True))


# A direction verb within three words of a horizon noun is a claim that the
# horizon moved UP or DOWN, which its own statistic (a mean absolute
# deviation) cannot support -- regardless of what other channels in the
# packet did. The plain sign rule above cannot see this, because a packet
# holding both `mase` up and `horizon_shape_near` will happily license
# "raises" and then let it attach to the wrong noun; that exact sentence
# ("raises the forecast's near horizon") is what the live run produced.
# `lead` is the same signed-channel-noun exemption the reverse-order rule
# uses, in the one position a lookbehind cannot express: "pulls the overall
# level downward and reshapes both horizons" puts the noun immediately
# before the direction word, and refusing it would be a false refusal.
_DIRECTION_ON_HORIZON = re.compile(
    r"(?<![a-z0-9_])(?P<lead>(?:" + _alt(_SIGNED_CHANNEL_NOUNS) + r")\s+)?"
    r"(" + _alt(UP_VERBS + DOWN_VERBS) + r")(?![a-z0-9_])"
    r"(?:\s+\S+){0,3}\s+(" + _alt(_HORIZON_WORDS) + r")(?![a-z0-9_])")
# The reverse order, over a five-word window rather than one: the live run
# put five words between the noun and the direction ("the near horizon of
# the forecast, moving it upward"). Widening it that far is only safe
# because a signed-channel noun on EITHER side of the direction word blocks
# the match -- the direction is then attached to that channel, which the
# packet may well license. Both halves of that exemption were measured
# against the 91 real packets: without the leading half, "reshapes the far
# and near horizons, pulling the overall level down" is a false refusal.
_HORIZON_THEN_DIRECTION = re.compile(
    r"(?<![a-z0-9_])(" + _alt(_HORIZON_WORDS) + r")(?![a-z0-9_])[,;:]?"
    r"(?:\s+(?!(?:" + _alt(_SIGNED_CHANNEL_NOUNS) + r")(?![a-z0-9_]))\S+){0,5}"
    r"\s+(" + _alt(_TRAILING_DIRECTION) + r")(?![a-z0-9_])"
    r"(?!(?:\s+\S+){0,2}\s+(?:" + _alt(_SIGNED_CHANNEL_NOUNS) + r")(?![a-z0-9_]))")


# ---------------------------------------------------------------------------
# The guard.
# ---------------------------------------------------------------------------

def _signed_channels(ev: Evidence) -> dict:
    """Both batteries' cleared channels, in the one shared sign convention.

    The guard's direction, quality and concept checks all ask "does the
    evidence support this word", and the answer is yes if EITHER battery
    supports it. Reading only `channels` would reject a direction the
    ablation pass measured -- a systematic rejection of correct sentences,
    which sec 11.51 records as the failure mode a narrowly-scoped guard
    produces: it rejected this module's own fallback.

    On a channel both batteries cleared, the larger magnitude wins. They are
    different interventions and can legitimately disagree in sign; the
    disagreement is a finding about the feature, not a reason to drop the
    channel, and dropping it would silently withhold the more informative
    case.
    """
    out = dict(ev.channels or {})
    for ch, v in (ev.ablation_channels or {}).items():
        if ch not in out or abs(float(v)) > abs(float(out[ch])):
            out[ch] = v
    return out


def allowed_concepts(ev: Evidence) -> set:
    """Every concept this packet licenses a mention of."""
    out = {channel_concept(c) for c in _signed_channels(ev)}
    if ev.structural_field:
        out.add(field_concept(ev.structural_field))
    for entry in (ev.top3_structural or ()):
        if entry and entry[0]:
            out.add(field_concept(entry[0]))
    for fam in (ev.exemplar_families or ()):
        out.add(family_concept(fam))
        out.add(field_concept(f"generator_{fam}"))
    for entry in (ev.exemplar_profile or ()):
        if entry and entry[0]:
            out.add(field_concept(entry[0]))
    return out


def _allowed_number_strings(ev: Evidence) -> set:
    """Renderings of every number in the packet, at the precisions any
    sentence would plausibly use. A digit sequence outside this set is a
    number the packet never contained."""
    out: set = set()

    def add(v):
        if v is None or isinstance(v, bool):
            return
        try:
            f = abs(float(v))
        except (TypeError, ValueError):
            return
        for nd in (0, 1, 2, 3):
            out.add(f"{f:.{nd}f}")
        if float(f).is_integer():
            out.add(str(int(f)))

    # BOTH batteries' raw values, not `_signed_channels`' merged view. That
    # merge keeps only the larger magnitude per channel, which is right for
    # the direction and concept checks but wrong here: on a channel both
    # batteries cleared, the smaller one is still a number the packet
    # literally renders, and this module's OWN fallback quotes it (the
    # ablation clause reports the ablation magnitude). It refused 6 of the
    # 98 feature fallbacks on `runs/full_report_run_4model` -- a false
    # refusal of a correct sentence, sec 11.35's damaging direction, and
    # sec 11.39's shape: the rule was right and reached one of its inputs.
    for src in (ev.channels, ev.ablation_channels):
        for v in (src or {}).values():
            add(v)
    add(ev.structural_rho)
    add(ev.structural_n)
    add(ev.n_atoms)
    for entry in (ev.top3_structural or ()):
        if entry and len(entry) >= 3:
            add(entry[1])
            add(entry[2])
    for entry in (ev.exemplar_profile or ()):
        if entry and len(entry) >= 3:
            add(entry[1])
            add(entry[2])
    return out


# Deliberately only the fragments that are labels and NOTHING else. The
# first draft also held "not measured" and "not tested", which deadlocks the
# module against itself: `UNTESTED_MARKERS` REQUIRES an untested packet's
# description to state that status, and those are the words it states it in
# -- an allowlist demanding a phrase a banlist forbids, so no untested
# description could ever be accepted. It produced zero refusals on
# `runs/full_report_run_4model` only because every packet there had a
# battery, i.e. the case that breaks it is the case that run cannot
# exercise (sec 11.51 lesson 3, a second time in the same module).
_EVIDENCE_LABEL_FRAGMENTS = (
    "none measured",
    "the same pattern",
)

# "these ROLES" only. The first draft also matched "these features", which
# refused 73 of this module's own 88 role fallbacks -- a role IS a cluster
# of features, so "These 12 features showed no measured effect" is both
# correct and the fallback's standard wording. The defect is a plural
# ROLE reference in a packet describing exactly one role.
_PLURAL_SELF_REFERENCE = re.compile(r"\b(?:these|those)\s+roles\b", re.IGNORECASE)


def check_text(text: str, ev: Evidence) -> str:
    """Empty string if `text` is an acceptable description of `ev`, else the
    reason it is not -- phrased so it can be handed straight back to the
    model as a correction."""
    raw = (text or "").strip()
    if not raw:
        return "the answer was empty"

    low = raw.lower()
    for bad in _SCAFFOLD_SUBSTRINGS:
        if bad in low:
            return (f"the answer contained {bad!r}; write one plain sentence with no "
                    "markdown, no labels and no brackets")
    if _LIST_MARKER.search(raw):
        return "the answer was a list; write one plain sentence"
    if "\n" in raw:
        return "the answer spanned several lines; write one plain sentence"

    words = raw.split()
    if len(words) > MAX_WORDS:
        return f"the answer was {len(words)} words; write at most {MAX_WORDS} words"
    # A decimal point is not a sentence boundary. Splitting on a bare
    # `[.!?]` read "0.56 times the null." as three sentences and rejected
    # the module's own fallback -- found by running `machine_fallback`
    # through `check_text`, which is why that round trip is a test.
    sentences = [s for s in _SENTENCE_SPLIT.split(raw) if s.strip()]
    if len(sentences) > MAX_SENTENCES:
        return (f"the answer was {len(sentences)} sentences; write at most "
                f"{MAX_SENTENCES}")

    if raw[-1] not in ".!?":
        return ("the answer did not end in a full stop, so it was cut off; write one "
                "complete sentence")

    # The evidence block's absence markers are LABELS in a machine-readable
    # block, not English noun phrases, and the model splices them in as
    # though they were: "This role fails to affect the forecast, tracking
    # none measured." reached 6 accepted descriptions on
    # `runs/full_report_run_4model`, and "tracking series with the same
    # pattern" is the same splice one field over. Nothing is factually
    # fabricated -- the sentence is simply not English, which no scan over
    # a vocabulary of CLAIMS can see, because the defect is that a label was
    # used as a phrase. The machine fallback states the same fact in words,
    # so a refusal here loses no information (sec 11.51 lesson 3: verified
    # by round-tripping every fallback through this guard).
    for label in _EVIDENCE_LABEL_FRAGMENTS:
        if label in low:
            return (f"the answer used the evidence label {label!r} as if it were "
                    "English; say it in plain words instead, or say nothing about it")

    # A packet describes exactly one feature or one role. Plural self-
    # reference ("these roles have no measurable effects") asserts a scope
    # the packet does not have, and it is the same defect as sec 11.53's
    # unguarded chunk STATE: a claim about how many things are being
    # described, which no vocabulary of channels or fields can reach.
    if _PLURAL_SELF_REFERENCE.search(low):
        return ("the answer described several features or roles; this evidence is "
                "about exactly one, so write about one")

    norm = _normalize(raw)

    for term in UNLICENSED_TERMS:
        if _UNLICENSED_PATTERNS[term].search(norm):
            return (f"the answer said {term!r}, which nothing in the evidence "
                    "measures; say only what was measured")

    if _signed_channels(ev):
        for term in NO_OTHER_EFFECT_TERMS:
            if _NO_OTHER_PATTERNS[term].search(norm):
                return (f"the answer said {term!r}, but the evidence lists more than "
                        "one channel that cleared the random-direction null")

    for fld in _RAW_FIELD_PATTERNS:
        if _RAW_FIELD_PATTERNS[fld].search(norm):
            return (f"the answer used the raw field name {fld!r}; write it in plain "
                    f"English as {FIELD_GLOSS[fld]!r}")

    # A direction verb is a claim about a channel's SIGN. `horizon_shape_*`
    # is a mean absolute deviation, so it carries no sign at all and cannot
    # license one -- which is why the signed set excludes it.
    # BOTH batteries' signs, not `_signed_channels`' merged view. That merge
    # keeps only the larger magnitude per channel, so when the two
    # interventions clear the SAME channel in OPPOSITE directions -- which
    # its own docstring calls a legitimate finding about the feature -- the
    # smaller one's sign is discarded and becomes unsayable. The tell is
    # sec 11.51 lesson 3: this module's own fallback quotes the ablation
    # clause ("decreases forecast error"), so the guard refused the sentence
    # it falls back to. Widening admits a direction word only where some
    # battery actually measured it; the check is unchanged where neither did.
    undirected = (getattr(ev, "undirected_channels", None) or frozenset())
    signed = [float(v)
              for src in (ev.channels, ev.ablation_channels)
              for ch, v in (src or {}).items()
              if ch not in CHANNEL_VERB and ch not in undirected]
    has_up = any(v > 0 for v in signed)
    has_down = any(v < 0 for v in signed)
    for term in UP_VERBS:
        if _UP_PATTERNS[term].search(norm) and not has_up:
            return (f"the answer said {term!r}, but no channel in the evidence moved "
                    "upward; a near or far horizon effect has a size, not a direction")
    for term in DOWN_VERBS:
        if _DOWN_PATTERNS[term].search(norm) and not has_down:
            return (f"the answer said {term!r}, but no channel in the evidence moved "
                    "downward; a near or far horizon effect has a size, not a direction")

    # The same rule for the `exemplar_profile` clause. Three states, not two,
    # for the reason `channels_measured` exists: a packet with no profile did
    # not measure this and must not be described as having found nothing
    # either -- it simply has no comparison to report, so ANY comparison in
    # the text is fabricated regardless of direction.
    prof = [e for e in (ev.exemplar_profile or ()) if e and len(e) >= 3]
    prof_above = any(float(e[1]) > float(e[2]) for e in prof)
    prof_below = any(float(e[1]) < float(e[2]) for e in prof)
    # "no effect ABOVE the random-direction null" is the module's own standard
    # phrasing for a feature that cleared nothing -- and it is a comparison to
    # the NULL, not to the corpus. Scanning the raw text flagged it, which
    # would have rejected the correct wording of the commonest packet state in
    # this repo. Removing the null collocations before the scan keeps a bare
    # "above" elsewhere in the sentence a fabrication, which is what the guard
    # is for; the alternative (requiring the full "above what is typical"
    # phrase) would have let the observed fabrication through, since it wrote
    # exactly the bare form. The scan is then narrowed to the windows that
    # actually compare something to the corpus -- see `_corpus_comparison_windows`,
    # without which this guard rejects the module's own machine fallback.
    prof_norm = _corpus_comparison_windows(_NULL_COMPARISON.sub(" ", norm))
    for term in PROFILE_ABOVE_TERMS:
        if _PROFILE_ABOVE_PATTERNS[term].search(prof_norm) and not prof_above:
            return (f"the answer said {term!r}, but " + (
                "the evidence lists nothing its own top series measure above "
                "what is typical" if prof else
                "nothing was measured about what its own top series score, so "
                "there is no comparison to the corpus to report at all"))
    for term in PROFILE_BELOW_TERMS:
        if _PROFILE_BELOW_PATTERNS[term].search(prof_norm) and not prof_below:
            return (f"the answer said {term!r}, but " + (
                "the evidence lists nothing its own top series measure below "
                "what is typical" if prof else
                "nothing was measured about what its own top series score, so "
                "there is no comparison to the corpus to report at all"))

    for pattern in (_DIRECTION_ON_HORIZON, _HORIZON_THEN_DIRECTION):
        for hit in pattern.finditer(norm):
            if (hit.groupdict().get("lead") or "").strip():
                continue
            return (f"the answer said {hit.group(0)!r}; a near or far horizon effect "
                    "is a size, not a direction, so say it moves or reshapes that "
                    "part of the forecast")

    mase = float(_signed_channels(ev).get("mase", 0.0))
    for term in QUALITY_WORSE:
        if _QUALITY_WORSE_PATTERNS[term].search(norm) and mase <= 0:
            return (f"the answer said {term!r}, but the evidence does not show "
                    "forecast error going up")
    for term in QUALITY_BETTER:
        if _QUALITY_BETTER_PATTERNS[term].search(norm) and mase >= 0:
            return (f"the answer said {term!r}, but the evidence does not show "
                    "forecast error going down")

    cleared_any = bool(ev.clears_null or ev.ablation_channels)
    if cleared_any:
        for term in NO_EFFECT_TERMS:
            if _NO_EFFECT_PATTERNS[term].search(norm):
                return (f"the answer said {term!r}, but the evidence lists a channel "
                        "that cleared the random-direction null")

    if not cleared_any:
        for term in CAUSAL_TERMS:
            if _CAUSAL_PATTERNS[term].search(norm):
                return (f"the answer said {term!r}, but no channel cleared the "
                        "random-direction null for this feature; describe it as a "
                        "correlate, not an effect")
        for term in CLEARED_CLAIM_TERMS:
            if _CLEARED_CLAIM_PATTERNS[term].search(norm):
                return (f"the answer said {term!r}, but the evidence records no "
                        "channel clearing the random-direction null; do not report "
                        "a comparison that came out negative as though it came out "
                        "positive")

    if not ev.channels_measured and not ev.ablation_measured:
        # The THIRD honest state, and the one the guard could not see. With no
        # battery run there is no null to have failed, so "does nothing above
        # the null" is not a cautious phrasing of ignorance -- it is a
        # measured negative reported from a measurement that never happened,
        # which is exactly the collapse `channels_measured` exists to prevent
        # (`CLAUDE.md` sec 11.37: "absent" and "bad" must be different
        # outcomes). Found by MEASURING 1.5B output rather than by reading the
        # guard: all three untested packets asserted a null and all three were
        # accepted (ROADMAP.md sec 26 C). The fallback already worded this
        # correctly -- what was missing was anything enforcing it.
        for term in NO_EFFECT_TERMS:
            if _NO_EFFECT_PATTERNS[term].search(norm):
                return (f"the answer said {term!r}, but no causal battery was run "
                        "for this one, so nothing measured whether it moves the "
                        "forecast; say it was not tested instead of saying it has "
                        "no effect")
        # ...and the blacklist above is not sufficient on its own -- see
        # UNTESTED_MARKERS. Requiring the status to be STATED is what makes
        # a paraphrase impossible rather than merely unlisted.
        if not any(pat.search(norm) for pat in _UNTESTED_MARKER_PATTERNS.values()):
            return ("no causal battery was run for this one, so the answer must say "
                    "so; write that it was not tested for an effect on the forecast, "
                    "then give only its structural correlate if it has one")

    allowed = allowed_concepts(ev)
    for term, concepts in DOMAIN_TERMS.items():
        if not _TERM_PATTERNS[term].search(norm):
            continue
        if not (set(concepts) & allowed):
            return (f"the answer mentioned {term!r}, which is not in the evidence; "
                    "mention only what the evidence lists")

    allowed_numbers = _allowed_number_strings(ev)
    for tok in _DIGITS.findall(raw):
        if tok not in allowed_numbers:
            return (f"the answer contained the number {tok!r}, which is not in the "
                    "evidence; write the sentence with no numbers at all")

    return ""


# ---------------------------------------------------------------------------
# The deterministic fallback -- what renders when the guard cannot be
# satisfied, so it has to be presentable rather than a debug string.
# ---------------------------------------------------------------------------

def _fmt(v: float, nd: int = 2) -> str:
    return f"{abs(float(v)):.{nd}f}"


def _subject(ev: Evidence) -> tuple:
    """`(subject, plural)`. The plural flag exists only so the fallback
    conjugates -- "These 16 features moves" was what the first version
    rendered, and a template sentence that renders bad grammar is not
    presentable, which is the one thing the fallback has to be."""
    if ev.kind == "role":
        n = ev.n_atoms
        if n == 1:
            return "This role's single feature", False
        if n:
            return f"These {n} features", True
        return "This role", False
    return "This feature", False


def _conjugate(verb: str, plural: bool) -> str:
    return verb[:-1] if (plural and verb.endswith("s")) else verb


def _channel_clause(ev: Evidence, plural: bool = False) -> str:
    """The strongest channel, verbalized with its own honest verb and its
    effect in multiples of the random-direction null."""
    items = sorted((ev.channels or {}).items(), key=lambda kv: -abs(float(kv[1])))
    if not items:
        return ""
    name, value = items[0]
    gloss = CHANNEL_GLOSS.get(name, name)
    up, down = _verbs(ev, name)
    verb = _conjugate(up if float(value) >= 0 else down, plural)
    clause = f"{verb} {gloss} ({_fmt(value)} times the random-direction null)"
    if len(items) > 1:
        second = CHANNEL_GLOSS.get(items[1][0], items[1][0])
        clause += f", and also {second}"
    return clause


def _ablation_clause(ev: Evidence, plural: bool = False) -> str:
    """The ablation battery's strongest channel, said where it was measured.

    Same shape as `_channel_clause`, and deliberately NOT merged with it: the
    two batteries answer different questions, and the trailing "where it
    fires" is the whole difference -- an injection effect is measured on
    series chosen without reference to the feature, this one only on the
    series it is actually active on.
    """
    items = sorted((ev.ablation_channels or {}).items(),
                   key=lambda kv: -abs(float(kv[1])))
    if not items:
        return ""
    name, value = items[0]
    gloss = CHANNEL_GLOSS.get(name, name)
    up, down = _verbs(ev, name)
    verb = _conjugate(up if float(value) >= 0 else down, plural)
    return (f"{verb} {gloss} on the series {'they fire' if plural else 'it fires'} "
            f"on ({_fmt(value)} times the random-direction null)")


def _structural_clause(ev: Evidence) -> str:
    fld = ev.structural_field
    if not fld:
        return ""
    gloss = FIELD_GLOSS.get(fld, fld)
    rho = ev.structural_rho
    n = ev.structural_n
    positive = rho is None or float(rho) >= 0
    if fld.startswith(_BOOLEAN_FIELD_PREFIXES):
        where = (f"strongest on series with {gloss}" if positive
                 else f"weakest on series with {gloss}")
    else:
        # "larger"/"smaller", never "higher"/"lower" -- "lower" is also a
        # direction VERB, and the direction check reads it as a claim about
        # a channel's sign rather than about a structural field's value.
        where = (f"strongest on series with a larger {gloss}" if positive
                 else f"strongest on series with a smaller {gloss}")
    stats = []
    if rho is not None:
        stats.append(f"rho {_fmt(rho)}")
    if n is not None:
        stats.append(f"n {int(n)}")
    return where + (f" ({', '.join(stats)})" if stats else "")


def machine_fallback(ev: Evidence) -> str:
    """A clean, deterministic, template-built sentence -- no model involved.

    Rendered whenever generation cannot satisfy the guard, so it is written
    to be presentable rather than diagnostic; it is also the correctness
    anchor for the guard itself, since `_assert_glosses_self_consistent` and
    this module's tests require the fallback's own output to pass
    `check_text` for the packet it was built from.
    """
    subject, plural = _subject(ev)
    chan = _channel_clause(ev, plural)
    abl = _ablation_clause(ev, plural)
    struct = _structural_clause(ev)

    # The ablation clause wins when both exist, rather than being appended.
    # It is the more relevant of the two -- it measures the feature where it
    # is actually active, which is the question that separates two features
    # that co-fire -- and concatenating both would routinely push the
    # sentence past MAX_WORDS, where the guard rejects the fallback itself.
    lead = abl or chan
    if lead:
        text = f"{subject} {lead}"
        if struct:
            text += f", {struct}"
        return text + "."

    if ev.ablation_measured and not ev.channels_measured:
        # The ablation battery ran and found nothing; the injection battery
        # never ran. "No measured effect above the null" is true of the one
        # that ran, so this is NOT the untested state -- but it must not be
        # reported as though both agreed.
        if struct:
            pronoun = "they are" if plural else "it is"
            return (f"{subject} changed nothing above the random-direction null "
                    f"when removed from the series {'they fire' if plural else 'it fires'} "
                    f"on; {pronoun} {struct}.")
        return (f"{subject} changed nothing above the random-direction null when "
                f"removed from the series {'they fire' if plural else 'it fires'} on.")

    if not ev.channels_measured:
        # No battery ran, so there is no null to have failed. Saying it
        # "showed no measured effect above the null" would report a
        # comparison that never happened as though it had come out negative.
        if struct:
            pronoun = "they are" if plural else "it is"
            # "causal" is in CAUSAL_TERMS and is licensed only when a
            # channel cleared, so the untested wording must avoid it -- and
            # should: an untested feature has no more claim on the word than
            # a feature that was tested and cleared nothing.
            return (f"{subject} {'were' if plural else 'was'} not tested for an "
                    f"effect on the forecast; {pronoun} {struct}.")
        # `subject` is plural for a role ("These 5 features"), so the verb has
        # to agree -- this branch read "These 5 features was not tested ... and
        # has no structural correlate", found by rendering the fallback rather
        # than by reading it.
        return (f"{subject} {'were' if plural else 'was'} not tested for an effect "
                f"on the forecast and {'have' if plural else 'has'} no structural "
                "correlate.")

    if struct:
        pronoun = "they are" if plural else "it is"
        return (f"{subject} showed no measured effect above the random-direction "
                f"null; {pronoun} {struct}.")
    return (f"{subject} showed no measured effect above the random-direction null "
            "and no structural correlate.")


# ---------------------------------------------------------------------------
# Prompting: in-context learning, with the examples as an auditable constant.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You rewrite measured statistics about one feature of a sparse autoencoder, "
    "trained on a time-series forecasting model, as a single plain-English "
    "sentence for a research report.\n"
    "Rules, most important first:\n"
    "1. Mention ONLY what the EVIDENCE block lists. Never name a forecast property, "
    "a series property or a data family that is not written there.\n"
    "2. If the evidence says the effect did not clear the random-direction null, "
    "never write causes, drives, controls, steers or responsible for. Call it a "
    "correlate.\n"
    "3. A near-horizon or far-horizon effect is a SIZE, not a direction. Say the "
    "feature moves or reshapes that part of the forecast. Never say it shifts the "
    "forecast earlier, later, into the future or toward the present, and never say "
    "it raises or lowers a horizon.\n"
    "4. Use an up or down word only for a property the evidence actually says "
    "increases or decreases.\n"
    "5. Exactly one complete sentence, at most thirty words, ending in a full stop.\n"
    "6. No numbers, no markdown, no lists, no quotation marks, no labels, and no "
    "raw field names with underscores in them.\n"
    "7. Claim nothing about how well the model performs, and never say only, "
    "always, never or nothing but.\n"
    "8. Say what patching the feature moves and, when a structural correlate is "
    "listed, on which kind of series it is strongest.\n"
    "9. When the evidence says its own top series measure some property above or "
    "below what is typical, say that too, in those words and without numbers. Do "
    "not write higher or lower there -- those describe the forecast, not the "
    "series. That clause is what tells this feature apart from others with the "
    "same correlate.\n"
    "10. Reply with the sentence and nothing else."
)

# Hand-written `(Evidence, description)` pairs. Real packets rather than
# pre-rendered text, for two reasons: the prompt's exemplar blocks are then
# produced by the same `render_evidence` a live packet goes through (so the
# model never sees a format it will not see again), and each exemplar can be
# run through `check_text` in a test -- an exemplar the guard would reject is
# a demonstration of how to be rejected, sitting in the prompt.
FEW_SHOT_EXAMPLES = (
    (
        Evidence(kind="role", model="A", layer="l", ident="1",
                 channels={"horizon_shape_far": 1.2795416842510559},
                 structural_field="has_random_walk", structural_rho=0.4606444223688049,
                 structural_n=555, n_atoms=12, clears_null=True),
        "Patching this role reshapes the far end of the forecast, most strongly on "
        "series that contain a random walk component.",
    ),
    (
        Evidence(kind="feature", model="A", layer="l", ident="5060",
                 channels={"mase": 1.8781122473440997, "horizon_shape_near": 0.42},
                 structural_field="has_intermittency", structural_rho=0.10136907104685296,
                 structural_n=555, clears_null=True),
        "Patching this feature makes the forecast worse and disturbs its early "
        "horizon, most on intermittent series.",
    ),
    (
        Evidence(kind="role", model="A", layer="l", ident="5", channels={},
                 structural_field="has_heteroskedastic", structural_rho=0.6111801597132711,
                 structural_n=555, n_atoms=5, clears_null=False),
        "This role has no effect above the random-direction null; it merely tracks "
        "heteroskedastic series.",
    ),
    (
        Evidence(kind="role", model="A", layer="l", ident="4",
                 channels={"level": -2.454884744177235},
                 n_atoms=2, exemplar_families=("parametric",), clears_null=True),
        "Patching this role pulls the whole forecast downward, with no structural "
        "correlate measured, seen on parametric series.",
    ),
    (
        # The commonest real packet shape by a wide margin on
        # `runs/full_report_run_large`: many channels at once, correlated with
        # the realism-stress tier. Added after a live run, where its absence
        # cost 15 of 26 rejections to one word -- the model glossed
        # `tier_realism_stress` as "realistic stress series", which is a claim
        # that the series ARE realistic rather than the name of a tier, and
        # the corrective retry could not talk it out of the paraphrase. An
        # exemplar could.
        Evidence(kind="feature", model="A", layer="l", ident="5501",
                 channels={"level": -2.41, "horizon_shape_far": 2.12,
                           "horizon_shape_near": 1.94, "spectral_centroid": 1.24,
                           "flatness": -1.09},
                 structural_field="tier_realism_stress", structural_rho=0.51,
                 structural_n=965, clears_null=True),
        "Patching this feature pulls the forecast's overall level down and reshapes "
        "both horizons, most strongly on series from the realism-stress tier.",
    ),
    (
        # The exemplar for `exemplar_profile` (sec 26 C, 2026-09-04). Without
        # one the model sees the new line only in the packet it is asked to
        # describe and treats it as noise -- and the line exists precisely to
        # be USED, since it is the only evidence that differs between two
        # features sharing a `structural_field`, which is most of a target's
        # top rows. Deliberately a packet whose `structural_field` is the
        # generic one those features share, so the exemplar demonstrates the
        # contrast doing the distinguishing work rather than decorating a
        # sentence that was already specific.
        Evidence(kind="feature", model="A", layer="l", ident="1568",
                 channels={}, channels_measured=False,
                 structural_field="seasonal_period_dominant",
                 structural_rho=0.27, structural_n=443,
                 exemplar_families=("random_parametric",),
                 exemplar_profile=(("seasonal_amplitude_max", 1.82, 1.0),
                                   ("ar_order", 0.0, 1.0))),
        "This feature was not tested for an effect on the forecast; it tracks the "
        "dominant seasonal period, firing on series whose seasonal swing is above "
        "and whose autoregressive order is below what is typical.",
    ),
)


def render_evidence(ev: Evidence) -> str:
    """The evidence block the model sees. Deterministic, and lossless with
    respect to everything the guard will later allow -- a packet field that
    is not rendered here could only ever be reached by guessing."""
    lines = ["EVIDENCE"]
    if ev.kind == "role":
        n = ev.n_atoms
        lines.append(f"kind: role of {n} features" if n else "kind: role")
    else:
        lines.append(f"kind: feature {ev.ident}")
    lines.append("cleared the random-direction null: "
                 + ("yes" if ev.clears_null else
                    "no" if ev.channels_measured else "not tested"))

    items = sorted((ev.channels or {}).items(), key=lambda kv: -abs(float(kv[1])))
    if items and ev.clears_null:
        for rank, (name, value) in enumerate(items):
            gloss = CHANNEL_GLOSS.get(name, name)
            up, down = _verbs(ev, name)
            verb = up if float(value) >= 0 else down
            prefix = "" if rank == 0 else "also "
            lines.append(f"{prefix}{verb}: {gloss}, {_fmt(value)} times the null")
    elif ev.channels_measured:
        lines.append("moves: nothing above the null")
    else:
        # Avoids every CAUSAL_TERMS word, including "causal" and "steering":
        # a term in the PROMPT is a term the model will echo, and the guard
        # would then reject its own evidence block's vocabulary.
        lines.append("moves: not tested -- no response battery was run")

    if ev.ablation_measured:
        # The second intervention, stated as such. The injection line above
        # says what steering this direction into a clean pass does; this says
        # what the feature is contributing where it actually fires, which is
        # the question two features that co-fire on the same series differ on
        # even when every correlational field they carry is identical.
        abl = sorted((ev.ablation_channels or {}).items(),
                     key=lambda kv: -abs(float(kv[1])))
        if abl:
            parts = []
            for name, value in abl:
                gloss = CHANNEL_GLOSS.get(name, name)
                up, down = _verbs(ev, name)
                verb = up if float(value) >= 0 else down
                parts.append(f"{verb}: {gloss}, {_fmt(value)} times the null")
            lines.append("what it contributes where it fires "
                         "(measured by removing it): " + "; ".join(parts))
        else:
            lines.append("what it contributes where it fires "
                         "(measured by removing it): nothing above the null")

    if ev.structural_field:
        gloss = FIELD_GLOSS.get(ev.structural_field, ev.structural_field)
        # State the DIRECTION here, exactly as `_structural_clause` already
        # states it in the machine fallback. Without it the block reads
        # "strongest on series with: seasonal amplitude", a bare noun the
        # model copies verbatim into "most strongly on series with seasonal
        # amplitude." -- grammatical debris in 7 accepted descriptions, and
        # a real loss of information, since the sign of rho is the whole
        # content of the correlate. This adds no fabrication surface: the
        # larger/smaller pair is already licensed vocabulary here, because
        # the fallback this guard round-trips has always used it.
        if not ev.structural_field.startswith(_BOOLEAN_FIELD_PREFIXES):
            positive = ev.structural_rho is None or float(ev.structural_rho) >= 0
            gloss = f"a larger {gloss}" if positive else f"a smaller {gloss}"
        tail = ""
        if ev.structural_rho is not None:
            tail = f" (rho {_fmt(ev.structural_rho)}"
            if ev.structural_n is not None:
                tail += f" over {int(ev.structural_n)} series"
            tail += ")"
        lines.append(f"strongest on series with: {gloss}{tail}")
    else:
        lines.append("strongest on series with: none measured")

    others = [e[0] for e in (ev.top3_structural or ())
              if e and e[0] and e[0] != ev.structural_field]
    lines.append("other correlates: "
                 + (", ".join(FIELD_GLOSS.get(f, f) for f in others) if others
                    else "none"))
    fams = [FAMILY_GLOSS.get(f, f) for f in (ev.exemplar_families or ())]
    lines.append("example families: " + (", ".join(fams) if fams else "none"))
    prof = [e for e in (ev.exemplar_profile or ()) if e and e[0]]
    if prof:
        # The line that lets two features sharing a `structural_field` be
        # told apart: what their OWN top-firing series measure, against the
        # corpus. Rendered as a comparison rather than a bare number so the
        # narrator has the direction without having to derive it.
        parts = []
        for field, mine, typical in prof:
            gloss = FIELD_GLOSS.get(field, field)
            # "above"/"below", never "higher"/"lower". A direction word is a
            # claim about a CHANNEL's sign, and `check_text` licenses one
            # only from a channel that actually moved that way. This is a
            # claim about a SERIES property, so it must not borrow that
            # vocabulary -- and "lower" is in `DOWN_VERBS` while "higher" is
            # not, so the natural pairing would also have rejected the
            # low-end half of every contrast and accepted the high-end half,
            # a systematic asymmetry rather than a guard. Extending
            # `has_down` to cover this field was the other option and was
            # rejected: it would license "patching this feature lowers the
            # forecast level" on the strength of a series property, which is
            # exactly the confusion the direction check exists to prevent.
            word = "above" if float(mine) > float(typical) else "below"
            parts.append(f"{gloss} {word} what is typical "
                         f"({_fmt(mine)} against {_fmt(typical)} typical)")
        lines.append("its own top series measure: " + "; ".join(parts))
    else:
        lines.append("its own top series measure: not measured")
    return "\n".join(lines)


def _messages(ev: Evidence, history: list) -> list:
    """System prompt, few-shot turns, this packet, then any rejected attempt
    plus its correction. `history` is `[(rejected_text, reason), ...]` --
    what makes a greedy retry produce a different answer at all."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_ev, desc in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": render_evidence(shot_ev)})
        msgs.append({"role": "assistant", "content": desc})
    msgs.append({"role": "user", "content": render_evidence(ev)})
    for rejected, reason in history:
        msgs.append({"role": "assistant", "content": rejected})
        msgs.append({"role": "user", "content":
                     f"That was rejected because {reason}. Write the sentence again, "
                     "obeying every rule."})
    return msgs


# ---------------------------------------------------------------------------
# Model handling.
# ---------------------------------------------------------------------------

def load_narrator(device: str = "cuda", dtype=None) -> Narrator:
    """Load the pinned narrator checkpoint once. Returns an opaque handle to
    hand to `describe`/`describe_batch`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    log.info("sae describe: loading %s @ %s on %s (%s)", MODEL_ID, REVISION[:12],
             device, dtype)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
                                                 dtype=dtype)
    model.to(device)
    model.eval()
    return Narrator(tokenizer=tok, model=model, device=str(device))


GEN_BATCH_SIZE = 16
"""Prompts sent to the narrator in one forward pass.

🔴 A FIXED bound, not a tuning knob, because the caller's batch size is the
number of packets a run happens to produce and that scales with the panel.
`run_sae_describe.py` hands `_generate` every packet at once: 116 on the
three-model run this was first exercised on, 192 on the four-model panel --
where it raised `torch.OutOfMemoryError` trying to allocate 4.93 GiB inside
Qwen's MLP with 682 MiB free. Nothing was wrong with either the model or the
packets; the batch was simply built from a quantity nobody bounded, so the
narrator worked until a run got one model wider and then did not.

16 is set below the largest batch measured to fit rather than at it, since
the peak depends on the longest prompt in the sub-batch and an evidence
packet's length is not bounded by anything either.
"""


def _generate(narrator: Narrator, batch_messages: list) -> list:
    """Greedy, deterministic completion for a batch of chat conversations.

    Chunked into `GEN_BATCH_SIZE` sub-batches, and on an out-of-memory error
    the offending sub-batch is halved and retried down to a single prompt.
    Both matter and they are not the same guard: the bound stops the common
    case (a panel with more roles than the last one), the backoff stops the
    case a bound cannot predict (one unusually long packet, or a GPU someone
    else is also using -- this repo's boxes are shared).

    Decoding is greedy, so a sub-batch's output does not depend on the other
    prompts in it beyond left-padding width, and the seed is re-set per
    sub-batch so a run is reproducible from the chunking alone.
    """
    import torch

    out_texts: list = []
    for start in range(0, len(batch_messages), GEN_BATCH_SIZE):
        window = batch_messages[start:start + GEN_BATCH_SIZE]
        size = len(window)
        while True:
            try:
                for off in range(0, len(window), size):
                    out_texts.extend(_generate_one(narrator, window[off:off + size]))
                break
            except torch.OutOfMemoryError:
                # Drop whatever this attempt produced before retrying, or
                # the halved retry would append duplicates of the inner
                # sub-batches that had already succeeded within this window.
                del out_texts[start:]
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                log.warning(
                    "sae describe: out of memory generating %d prompts at once; "
                    "retrying at %d. This is handled, but if it recurs every "
                    "run the packet count has outgrown GEN_BATCH_SIZE (%d) and "
                    "that constant should move, not this backoff",
                    len(window), size, GEN_BATCH_SIZE)
    return out_texts


def _generate_one(narrator: Narrator, batch_messages: list) -> list:
    """One forward pass. `_generate` owns the batching; this owns the call."""
    import torch

    tok = narrator.tokenizer
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
               for m in batch_messages]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(narrator.device) for k, v in enc.items()}
    torch.manual_seed(SEED)
    with torch.no_grad():
        out = narrator.model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(row, skip_special_tokens=True).strip() for row in gen]


def describe_batch(evs: list, narrator) -> list:
    """One `Description` per evidence packet, same order.

    Retries are batched by ROUND rather than per item: every packet still
    failing after round `r` is re-sent together in round `r+1` with its own
    rejection reason appended, so the number of forward passes is bounded by
    `MAX_ATTEMPTS` regardless of how many packets there are.
    """
    evs = list(evs)
    if not evs:
        return []

    if narrator is None:
        return [Description(text=machine_fallback(ev), accepted=False,
                            reason="no narrator loaded", attempts=0,
                            model_id=MODEL_ID, revision=REVISION) for ev in evs]

    results: list = [None] * len(evs)
    pending = list(range(len(evs)))
    histories: dict = {i: [] for i in pending}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        texts = _generate(narrator, [_messages(evs[i], histories[i]) for i in pending])
        still: list = []
        for i, text in zip(pending, texts):
            reason = check_text(text, evs[i])
            if not reason:
                results[i] = Description(text=text.strip(), accepted=True, reason="",
                                         attempts=attempt, model_id=narrator.model_id,
                                         revision=narrator.revision)
            else:
                histories[i].append((text.strip(), reason))
                still.append(i)
        pending = still

    for i in pending:
        last_reason = histories[i][-1][1] if histories[i] else "generation failed"
        results[i] = Description(text=machine_fallback(evs[i]), accepted=False,
                                 reason=last_reason, attempts=MAX_ATTEMPTS,
                                 model_id=narrator.model_id, revision=narrator.revision)
    return results


def describe(ev: Evidence, narrator) -> Description:
    """One packet. Implemented as `describe_batch` of one so the single and
    batch paths cannot diverge in prompt construction or guarding."""
    return describe_batch([ev], narrator)[0]


# ---------------------------------------------------------------------------
# Import-time self-consistency: a fallback the guard would reject is a bug.
# ---------------------------------------------------------------------------

def _assert_glosses_self_consistent() -> None:
    for channel, gloss in CHANNEL_GLOSS.items():
        ev = Evidence(kind="feature", model="m", layer="l", ident="0",
                      channels={channel: 1.0}, clears_null=True)
        reason = check_text(f"This feature moves {gloss}.", ev)
        if reason:
            raise AssertionError(
                f"CHANNEL_GLOSS[{channel!r}] = {gloss!r} does not pass its own "
                f"guard: {reason}")
    for fld, gloss in FIELD_GLOSS.items():
        ev = Evidence(kind="feature", model="m", layer="l", ident="0",
                      structural_field=fld, structural_rho=0.5, structural_n=100,
                      clears_null=False)
        reason = check_text(f"This feature tracks series with {gloss}.", ev)
        if reason:
            raise AssertionError(
                f"FIELD_GLOSS[{fld!r}] = {gloss!r} does not pass its own guard: "
                f"{reason}")


_assert_glosses_self_consistent()
