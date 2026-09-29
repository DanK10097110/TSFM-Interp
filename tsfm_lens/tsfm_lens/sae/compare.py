"""Cross-model comparison of SAE roles: the measured reduction, and a
guarded narration of it.

Inherited from the stage before: `sae/roles.py` clusters each target's
features into roles, `sae/response.py` measures what each feature does to
nine forecast channels under two batteries (injection and ablation), and
`sae/role_matching.py` matches one model's roles against another's by
response-fingerprint cosine against a within-run population null. Every
number this module reads was measured there; nothing here re-derives one.

What this module adds is the reduction those artifacts do not perform. The
report renders SAE evidence organised BY MEASUREMENT -- a dictionary-health
table, a structural heatmap, one match table per pair -- and a reader asking
"what does this model account for that this one doesn't" has to join six
surfaces by hand. `model_capability_profile` is that join, done once, per
model. `contrast_chunks` is the same join expressed as the comparable units a
narrator can take one at a time.

🔴 THE CLAIM THIS MODULE IS NOT ALLOWED TO MAKE, and the reason its guard is
larger than `describe.py`'s. "Model A accounts for something model B does
not" is a CAPABILITY claim. What the evidence supports is a CORRESPONDENCE
claim: at these two targets, under this cosine threshold, this role found no
counterpart. Those differ in three ways that all point the same direction --
only two of each model's layers were dictionaried, only the alive atoms were
scored, and the matching threshold is a convention (`cosine_threshold`, 0.5
by default) rather than a measurement. A role with no counterpart here may
have one at a layer nobody trained an SAE on. So `check_contrast_text`
refuses capability language outright (`_CAPABILITY_TERMS`), and the machine
fallbacks are written in correspondence terms so the guard and the fallback
agree -- `CLAUDE.md` sec 11.51's rule that a guard which rejects its own
module's fallback is describing something other than truthfulness.

The second claim it may not make is the aggregate one. `match_rate` sits
below both sides' untrained-twin floor on the one real pair ever checked
(`CLAUDE.md` sec 6.5), and the causal-agreement rate has no twin floor at
all yet, so no sentence here may quantify how much two models share.
Per-pair verdicts ARE quotable -- each carries its own population-null p95,
a different and available control -- which is why the chunk level is where
the narration lives and the synthesis level is deliberately vocabulary-
narrowed to what the chunks already established.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _dc_field

import numpy as np

from ..utils import log
from . import describe as _d
from .matching import role_causal_fingerprint

__all__ = [
    "Contrast", "ContrastSet",
    "model_capability_profile", "contrast_chunks",
    "render_contrast", "contrast_allowed_concepts", "check_contrast_text",
    "contrast_machine_fallback", "unscorable_clause",
    "DETERMINISTIC_BY_DESIGN", "describe_contrasts",
    "synthesis_evidence", "check_synthesis_text", "synthesis_fallback",
    "compare_models", "CONTRAST_SYSTEM_PROMPT", "SYNTHESIS_SYSTEM_PROMPT",
    "FEW_SHOT_CONTRASTS",
]


# ---------------------------------------------------------------------------
# The measured reduction.
# ---------------------------------------------------------------------------

def _channel_label(name: str) -> str:
    """The one English name for a channel, read from `describe.py` rather
    than re-listed here so the table and the narrator cannot disagree about
    what `horizon_shape_far` is called (sec 2.2)."""
    return _d.CHANNEL_GLOSS.get(name, name.replace("_", " "))


def _field_label(name: str) -> str:
    return _d.FIELD_GLOSS.get(name, name.replace("_", " "))


def _is_unsigned(channel: str) -> bool:
    """True for a channel that is a magnitude rather than a signed quantity.

    Read from `describe.CHANNEL_VERB`, whose keys are exactly the channels
    whose gloss verb is direction-free, for the same no-drift reason as
    `_channel_label`. A direction word on one of these is the defect
    `report/sae_features.py` already carries a guard for.
    """
    return channel in _d.CHANNEL_VERB


def model_capability_profile(roles_json: dict, ablation_by_target: dict,
                             cosine_threshold: float = 0.5,
                             match_table: dict | None = None) -> dict:
    """One record per model: what its dictionaries causally move, what they
    track structurally, and how much of that has a counterpart elsewhere.

    This is a pure reduction over already-written artifacts -- no model is
    loaded, nothing is re-measured -- and it holds to `report/derived.py`'s
    adaptivity contract: no model name, architecture family or positional
    index appears anywhere in it, so it transfers to a panel of checkpoints
    nobody has run.

    `match_table` is optional. Without it every role is reported with
    `correspondence` "not compared", which is the honest state for a solo
    run rather than a silent zero -- a role that was never offered a
    counterpart and a role that was offered one and found none are different
    outcomes (`CLAUDE.md` sec 11.37).
    """
    matched_roles: dict = {}
    offered: set = set()
    if match_table:
        for rec in match_table.get("pairs") or []:
            if not rec.get("comparable"):
                continue
            ta, tb = rec.get("target_a"), rec.get("target_b")
            offered.add(ta)
            offered.add(tb)
            for m in rec.get("matches") or []:
                ia, ib = m.get("role_a_index"), m.get("role_b_index")
                cos = m.get("cosine")
                if ia is None or ib is None or cos is None:
                    continue
                if float(cos) < float(cosine_threshold):
                    continue
                matched_roles.setdefault((ta, int(ia)), []).append((tb, int(ib)))
                matched_roles.setdefault((tb, int(ib)), []).append((ta, int(ia)))

    by_model: dict = {}
    for target, rec in (roles_json or {}).items():
        model = rec.get("model")
        if not model:
            continue
        prof = by_model.setdefault(model, {
            "model": model,
            "targets": [],
            "n_roles": 0,
            "n_roles_with_causal_direction": 0,
            "channels": {},
            "structural_fields": {},
            "roles_with_counterpart": 0,
            "roles_without_counterpart": 0,
            "roles_not_compared": 0,
        })
        prof["targets"].append(target)
        abl = ablation_by_target.get(target)
        for role in rec.get("roles") or []:
            prof["n_roles"] += 1
            idx = role.get("role")
            fp = role_causal_fingerprint(role, abl)
            if fp.get("available"):
                prof["n_roles_with_causal_direction"] += 1

            ch = role.get("dominant_channel")
            if ch and role.get("clears_null"):
                slot = prof["channels"].setdefault(
                    ch, {"channel": ch, "label": _channel_label(ch),
                         "signed": not _is_unsigned(ch),
                         "n_roles": 0, "targets": set(), "strengths": []})
                slot["n_roles"] += 1
                slot["targets"].add(target)
                eff = role.get("dominant_effect_null_units")
                if eff is not None:
                    slot["strengths"].append(abs(float(eff)))

            fld = role.get("structural_field")
            if fld:
                fslot = prof["structural_fields"].setdefault(
                    fld, {"field": fld, "label": _field_label(fld),
                          "n_roles": 0, "targets": set(), "rhos": []})
                fslot["n_roles"] += 1
                fslot["targets"].add(target)
                if role.get("structural_rho") is not None:
                    fslot["rhos"].append(abs(float(role["structural_rho"])))

            if not match_table or target not in offered:
                prof["roles_not_compared"] += 1
            elif matched_roles.get((target, int(idx))):
                prof["roles_with_counterpart"] += 1
            else:
                prof["roles_without_counterpart"] += 1

    for prof in by_model.values():
        prof["targets"] = sorted(prof["targets"])
        for slot in prof["channels"].values():
            slot["targets"] = sorted(slot["targets"])
            slot["n_targets"] = len(slot["targets"])
            slot["median_strength_null_units"] = (
                float(np.median(slot["strengths"])) if slot["strengths"] else None)
            slot.pop("strengths")
        for fslot in prof["structural_fields"].values():
            fslot["targets"] = sorted(fslot["targets"])
            fslot["n_targets"] = len(fslot["targets"])
            fslot["median_abs_rho"] = (
                float(np.median(fslot["rhos"])) if fslot["rhos"] else None)
            fslot.pop("rhos")

    return {"models": by_model,
            "cosine_threshold": float(cosine_threshold),
            "compared": bool(match_table),
            "scope_note": (
                "Counts are over the layers this run trained a dictionary on "
                "and the atoms that survived the dead-feature gate. A role "
                "with no counterpart here was not offered one at every layer "
                "of the other model, so this is a statement about "
                "correspondence at these targets, not about what either "
                "model can represent.")}


# ---------------------------------------------------------------------------
# The comparable units.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contrast:
    """One comparable unit, and everything a sentence about it may mention.

    `kind` is `"pair"` (two roles matched across models) or `"solo"` (a role
    with no counterpart above the threshold). The two carry different
    licences and are guarded differently, which is why they are one type
    with a discriminant rather than two loosely-related dicts: every guard
    that reads a `Contrast` is forced to decide which case it is handling.
    """

    kind: str
    model_a: str
    layer_a: str
    role_a: int
    name_a: str
    channels_a: dict = _dc_field(default_factory=dict)
    field_a: str | None = None
    model_b: str | None = None
    layer_b: str | None = None
    role_b: int | None = None
    name_b: str | None = None
    channels_b: dict = _dc_field(default_factory=dict)
    field_b: str | None = None
    cosine: float | None = None
    population_null_p95: float | None = None
    causal_verdict: str | None = None
    """One of `same causal role`, `fires together, acts differently`,
    `not scorable`, or None when no ablation battery reached this pair.

    🔴 The guard checks a sentence against THIS, not against the cosine.
    A pair can match on activation profile -- the two roles fire on the same
    series -- while the ablation battery says they push the forecast in
    opposite directions, and that disagreement is the entire reason this
    comparison exists. A sentence describing such a pair as agreement is the
    single most damaging thing this module could emit, because it is fluent,
    plausible, and inverts the finding.
    """
    causal_cosine: float | None = None
    causal_null_p95: float | None = None
    causal_reason: str = ""
    """WHY `causal_verdict` is `not scorable`, in the battery's own words.

    Empty for a scored pair. Carried because "not scorable" names a state
    without naming its cause, and the two causes measured on
    `runs/full_report_run_4model` are different findings that a reader must
    be able to tell apart: 25 of 27 unscorable cells are one side's role
    having NO member that cleared any channel's null (the role has no causal
    direction, which is a fact about that role), and 2 are the null pool
    being smaller than 20 (the p95 arithmetic of sec 6.6's p-floor in a
    different statistic, which is a fact about the run's size). Without this
    field the narrator invented a third, unmeasured cause -- "could not be
    scored due to differences in the roles' effects on the forecast" -- which
    is both unlicensed and, for the commonest case, the opposite of true.
    """
    n_atoms_a: int | None = None
    n_atoms_b: int | None = None
    forbidden_models: tuple = ()
    """Model names in this run that this chunk is NOT about.

    A panel run hands the narrator four names across one batch, and the two
    in any given chunk are a third of them; naming the wrong one produces a
    sentence that is fluent, well-formed and about a comparison that was
    never measured. Nothing else in the guard can catch it -- a model name
    is not a domain term, so the concept allowlist passes it straight
    through. Populated by `contrast_chunks` from the run's full model list.
    """


@dataclass
class ContrastSet:
    """A pair of models' chunks plus the pair-level facts a synthesis may use."""

    model_a: str
    model_b: str
    chunks: list = _dc_field(default_factory=list)
    match_rate: float | None = None
    match_rate_quotable: bool = False
    n_agree: int = 0
    n_disagree: int = 0
    n_unscorable: int = 0


def _role_channels(role: dict) -> dict:
    """The one channel a role is named for, as `{channel: signed_null_units}`.

    Roles carry a single dominant channel rather than the full cleared set --
    that is what `sae/roles.py` clusters on -- so this is a one-entry dict by
    construction, kept as a dict so the narrator's channel handling is
    identical to `describe.Evidence.channels`.
    """
    ch = role.get("dominant_channel")
    if not ch or not role.get("clears_null"):
        return {}
    eff = role.get("dominant_effect_null_units")
    if eff is None:
        return {}
    sign = role.get("sign")
    val = abs(float(eff)) * (1.0 if (sign is None or sign >= 0) else -1.0)
    return {ch: val}


def contrast_chunks(match_table: dict, roles_json: dict,
                    cosine_threshold: float = 0.5) -> list:
    """Every model pair's comparable units, one `ContrastSet` per pair.

    A chunk is a matched role pair or an unmatched role, never an arbitrary
    slice of a table: the unit the narrator sees is the unit the measurement
    was made on, so a sentence about it can be checked against exactly the
    record that produced it. Chunking by row count instead would let one
    sentence span two pairs whose verdicts disagree, and no guard could
    catch that.
    """
    all_models = sorted({(v or {}).get("model") for v in (roles_json or {}).values()
                         if (v or {}).get("model")})
    out = []
    for rec in match_table.get("pairs") or []:
        if not rec.get("comparable"):
            continue
        ta, tb = rec.get("target_a"), rec.get("target_b")
        ra = {int(r["role"]): r for r in ((roles_json.get(ta) or {}).get("roles") or [])}
        rb = {int(r["role"]): r for r in ((roles_json.get(tb) or {}).get("roles") or [])}
        ma = (roles_json.get(ta) or {}).get("model") or rec.get("model_a")
        mb = (roles_json.get(tb) or {}).get("model") or rec.get("model_b")
        la = (roles_json.get(ta) or {}).get("layer") or ""
        lb = (roles_json.get(tb) or {}).get("layer") or ""

        forbidden = tuple(m for m in all_models if m not in (ma, mb))
        cs = ContrastSet(model_a=ma, model_b=mb,
                         match_rate=rec.get("match_rate"),
                         match_rate_quotable=bool(rec.get("match_rate_quotable")))
        summary = rec.get("causal_summary") or {}
        cs.n_agree = int(summary.get("n_agree") or 0)
        cs.n_disagree = int(summary.get("n_disagree") or 0)
        cs.n_unscorable = int(summary.get("n_not_scorable") or 0)

        used_a, used_b = set(), set()
        for m in rec.get("matches") or []:
            ia, ib = m.get("role_a_index"), m.get("role_b_index")
            if ia is None or ib is None:
                continue
            ia, ib = int(ia), int(ib)
            role_a, role_b = ra.get(ia), rb.get(ib)
            if role_a is None or role_b is None:
                continue
            cos = m.get("cosine")
            if cos is not None and float(cos) < float(cosine_threshold):
                continue
            used_a.add(ia)
            used_b.add(ib)
            causal = m.get("causal") or {}
            cs.chunks.append(Contrast(
                kind="pair", model_a=ma, layer_a=la, role_a=ia,
                name_a=str(role_a.get("name") or ia),
                channels_a=_role_channels(role_a),
                field_a=role_a.get("structural_field"),
                model_b=mb, layer_b=lb, role_b=ib,
                name_b=str(role_b.get("name") or ib),
                channels_b=_role_channels(role_b),
                field_b=role_b.get("structural_field"),
                cosine=None if cos is None else float(cos),
                population_null_p95=m.get("population_null_p95"),
                causal_verdict=causal.get("verdict"),
                causal_cosine=causal.get("cosine"),
                causal_null_p95=causal.get("null_p95"),
                causal_reason=str(causal.get("reason") or ""),
                n_atoms_a=role_a.get("n_atoms"), n_atoms_b=role_b.get("n_atoms"),
                forbidden_models=forbidden))

        for idx, role, model, layer, other in (
                [(i, r, ma, la, mb) for i, r in ra.items() if i not in used_a] +
                [(i, r, mb, lb, ma) for i, r in rb.items() if i not in used_b]):
            if not _role_channels(role) and not role.get("structural_field"):
                continue
            cs.chunks.append(Contrast(
                kind="solo", model_a=model, layer_a=layer, role_a=int(idx),
                name_a=str(role.get("name") or idx),
                channels_a=_role_channels(role),
                field_a=role.get("structural_field"),
                model_b=other, n_atoms_a=role.get("n_atoms"),
                forbidden_models=forbidden))
        out.append(cs)
    return out


# ---------------------------------------------------------------------------
# What a contrast sentence may say.
# ---------------------------------------------------------------------------

AGREEMENT_TERMS = (
    "same", "alike", "agree", "agrees", "matching", "match", "matches",
    "both push", "in step", "line up", "lines up", "aligned", "consistent",
)

DISAGREEMENT_TERMS = (
    "differ", "differs", "different", "differently", "opposite", "oppose",
    "opposing", "diverge", "diverges", "apart", "unlike", "conflict",
    "conflicting", "disagree", "disagrees",
)

_AGREEMENT_ELIDE = re.compile(
    r"\b(?:the\s+)?same\s+(?:series|set\s+of\s+series|inputs?|data)\b")
"""The one collocation `AGREEMENT_TERMS` must not read as agreement.

Two roles matched by activation profile fire on the same series BY
CONSTRUCTION -- that is what the match means -- so saying so is never a
claim about whether removing them does the same thing. Deliberately narrow:
`same way`, `same direction` and `same effect` all survive it, so the
inversion this guard exists to catch is still caught.
"""

_CAPABILITY_TERMS = (
    "can", "cannot", "can't", "unable", "incapable", "lacks", "lack",
    "lacking", "missing", "absent", "does not have", "has no ability",
    "no ability", "fails to", "unavailable", "only model", "uniquely",
    "unique to", "exclusive", "exclusively",
    # Added after a live run: the BARE nouns, not just the negated forms.
    # "no ability" was refused while "differ in their ability to ..." was
    # accepted, which is the same claim with the negation moved outside the
    # phrase the scan was looking for.
    "ability", "abilities", "capable", "capability", "capabilities",
)
"""🔴 Refused outright, in both chunk kinds.

The whole prompt frames this as "what does this model account for that this
one doesn't", so a fluent model will reach for capability language on its
own; nothing in the evidence supports it. A role with no counterpart at
these targets is a correspondence fact bounded by which layers were
dictionaried and by an arbitrary cosine threshold -- see this module's own
header. `missing`/`absent` are included because they are the same claim in
the negative, and `uniquely`/`exclusive` because they are it in the
superlative.
"""

_SUPERIORITY_TERMS = (
    "outperform", "outperforms", "outperformed", "superior", "inferior",
    "stronger", "weaker", "beats", "wins", "loses", "advantage",
    "more effective", "less effective", "more accurate", "less accurate",
    # Added 2026-09-09 after reading a live run's ACCEPTED output: the
    # narrator glossed "raises the spectral centroid" as "strengthens the
    # forecast" / "makes it clearer" / "strengthens forecast clarity" in 4 of
    # 49 accepted texts. A higher spectral centroid is more high-frequency
    # content, not a better forecast -- the same fabrication as "outperforms",
    # reached through a channel word rather than a comparison word. Every
    # channel the evidence is licensed to name is its own such surface
    # (sec 11.51). Bare "clear" is deliberately NOT here: "clear null
    # channels" and "cleared its null" are this module's own wording.
    "strengthens", "strengthen", "strengthening", "clarity", "clearer",
    "sharper", "more precise", "less precise", "more reliable",
    "less reliable",
    # Added 2026-09-09, third read-the-output pass over the SAME run: an
    # accepted summary read "affects the forecast's accuracy and RELIABILITY".
    # "accuracy" is legal there and stays so -- it maps to `channel:mase` and
    # that pair's chunks did license mase ("increases forecast error"). Bare
    # "reliability" maps to NOTHING: it is in no DOMAIN_TERMS entry, so the
    # vocabulary scan never sees it, and it was in no QUALITY tuple either, so
    # neither did the quality scan. A word in no vocabulary is not "allowed",
    # it is unexamined -- which is sec 11.53's lesson stated for the gap
    # between two scans rather than inside one. 1 of 5 accepted summaries.
    "reliability", "reliable", "unreliable",
    # And the paraphrase that replaced "reliability" the moment it was
    # banned: the regenerated summary read "accuracy and PRECISION". The
    # compound forms "more precise"/"less precise" were already here and the
    # bare noun was not, which is sec 26 C's blacklist-defeated-on-re-measure
    # finding arriving at this module -- recorded rather than treated as a
    # surprise, because it is the reason `synthesis_fallback` was rewritten
    # to be worth reading instead of this list being trusted to converge.
    "precision", "precise", "imprecise",
    # Added 2026-09-09 from the same read-the-output pass: "has the strongest
    # PERFORMANCE on series with dominant seasonal periods" in 3 of 44
    # accepted texts. `phrase()` writes "strongest on series with ...", which
    # says where the feature FIRES; the added noun turns it into a claim about
    # how well the model forecasts there. Bare "strongest" is deliberately NOT
    # here -- it is this module's own wording, and the pair of sentences
    # differing only by that noun is what the boundary test pins.
    "performance", "performs", "perform", "performing",
)
"""Ranking language, refused in BOTH stages.

Distinct from `describe.QUALITY_BETTER`/`QUALITY_WORSE`, which this module
also scans: those are about a forecast getting better or worse, these are
about one model beating another. Nothing here measures that -- L0 does, in
a different section, on a different axis -- and the panel framing invites it
hard. Kept local to this module rather than added to `describe.py`, whose
own acceptance rates are recorded against the vocabulary it had.

The live run that motivated this produced the accepted sentence "TimesFM
outperforms Chronos-Bolt in these areas" for a pair whose every single role
comparison was unscorable.
"""

_SHARE_QUANTIFIERS = (
    "most", "many", "few", "majority", "minority", "half", "mostly",
    "largely", "generally", "typically", "overall", "in general",
)
"""Refused in the SYNTHESIS only.

Quantifying how much two models share is exactly the number that sits below
its own untrained-twin floor (`CLAUDE.md` sec 6.5). A per-chunk sentence has
its own population-null p95 and is quotable; a sentence saying "most roles
correspond" is the unquotable aggregate wearing a word instead of a number.
"""

_EVALUATIVE_UNLICENSED = (
    "excels", "works best", "performs best", "best at", "good at", "better at",
)
"""The evaluative members of `describe.UNLICENSED_TERMS`, named here so the
PROMPT can list them.

They are already refused by the unlicensed scan; what this tuple adds is that
the narrator is told. A test pins that every member is still in
`describe.UNLICENSED_TERMS` -- a hand-picked subset is a claim about another
module, and if `describe.py` ever drops one the prompt must stop teaching that
it is banned rather than silently going stale (sec 11.34: a list checked
nowhere is a claim checked nowhere).
"""

_PROMPT_BANNED_WORDS = tuple(sorted(set(
    _d.QUALITY_WORSE + _d.QUALITY_BETTER + _SUPERIORITY_TERMS
    + _EVALUATIVE_UNLICENSED)))
"""The exact words rule 6 forbids, BUILT FROM THE GUARD'S OWN TUPLES.

🔴 This is the fix for the single largest source of refusals. On the
four-model panel 17 of 25 rejected chunk sentences were one lexical family --
`excels` x5, `enhances`/`enhance`/`enhancing` x5, `improves`/`improving` x4,
`outperforms`/`outperform` x2, `strengthens` x1 -- and only 3 were actual
fabrications. The guard held the list of banned words and the prompt shared
none of it: rule 6 said "claim nothing about which model is better", which a
1.5B narrator does not map onto `enhances`. That is sec 11.53's lesson at the
prompt boundary -- a guard and the vocabulary it enforces are one object, and
so are a guard and the instruction that is supposed to pre-empt it.

Derived rather than re-typed so the two cannot drift; a test pins that this
tuple equals the scan's own concatenation.
"""

CONTRAST_SYSTEM_PROMPT = (
    "You rewrite one measured comparison between two sparse-autoencoder roles, "
    "each from a different time-series forecasting model, as a single plain-English "
    "sentence for a research report.\n"
    "Rules, most important first:\n"
    "1. Mention ONLY what the EVIDENCE block lists. Never name a forecast property, "
    "a series property or a model that is not written there.\n"
    "2. The VERDICT line is the finding. If it says the two act differently, say "
    "they differ; never say they agree, match or are alike. If it says same causal "
    "role, say they agree. If it says not scorable, say the comparison could not be "
    "scored and take no side. If it says no counterpart was found, describe "
    "only this role's own measured effects and say it has none there; never "
    "say it differs from, resembles or was compared against any role in the "
    "other model.\n"
    "3. Never say a model can, cannot, lacks, is missing or is unique in anything. "
    "A role with no counterpart was compared at these layers only.\n"
    "4. A near-horizon or far-horizon effect is a SIZE, not a direction. Never say "
    "it raises, lowers or shifts a horizon.\n"
    "5. Use an up or down word only for a property the evidence says increases or "
    "decreases.\n"
    "6. Claim nothing about which model is better, more accurate or stronger. "
    "Never use any of these words: " + ", ".join(_PROMPT_BANNED_WORDS) + ". "
    "Say WHAT each role does to the forecast, never how good it is.\n"
    "7. Exactly one complete sentence, at most thirty-five words, ending in a full "
    "stop.\n"
    "8. No numbers, no markdown, no lists, no quotation marks, and no raw field "
    "names with underscores in them.\n"
    "9. Reply with the sentence and nothing else."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are given several one-sentence findings, each already checked against its "
    "own measurement, comparing two time-series forecasting models. Write a short "
    "summary of what they add up to.\n"
    "Rules, most important first:\n"
    "1. Use ONLY properties named in the findings. Introduce nothing new.\n"
    "2. Never say how many or what share of roles agree or differ, and never use "
    "most, many, few, mostly or generally. Say which properties agreed and which "
    "differed.\n"
    "3. Never say a model can, cannot, lacks or is missing anything, and never say "
    "which model is better. Never use any of these words: "
    + ", ".join(_PROMPT_BANNED_WORDS) + ".\n"
    "4. At most two complete sentences, at most fifty words total.\n"
    "5. No numbers, no markdown, no lists, and no raw field names with underscores.\n"
    "6. Reply with the summary and nothing else."
)


# Hand-written `(Contrast, sentence)` pairs, mirroring
# `describe.FEW_SHOT_EXAMPLES` exactly -- real objects rather than pre-rendered
# text, so the prompt's exemplar blocks come out of the same `render_contrast`
# a live chunk goes through, and so every exemplar can be run through
# `check_contrast_text` in a test.
#
# 🔴 Why this exists at all. `describe._messages` has shown its narrator
# worked examples since it was written and accepts 96.8% of packets where a
# profile was measured; `_contrast_messages` was system prompt -> evidence ->
# retry, with no examples, and accepted 60% -- the only zero-shot narrator in
# the repo, doing the harder of the two jobs. The four exemplars cover the
# four states a chunk can be in, because the state is exactly what the
# narrator was getting wrong: three of the four verdict framings appear
# nowhere in its context otherwise.
FEW_SHOT_CONTRASTS = (
    (
        Contrast(kind="pair", model_a="A", layer_a="l", role_a=1,
                 name_a="trend tilter", channels_a={"trend": 1.5},
                 field_a="has_random_walk",
                 model_b="B", layer_b="m", role_b=2, name_b="trend tilter",
                 channels_b={"trend": -1.2}, field_b="has_random_walk",
                 causal_verdict="fires together, acts differently"),
        "Both roles fire on the same series, those with a random walk "
        "component, but removing each one moves the forecast's trend slope a "
        "different way.",
    ),
    (
        Contrast(kind="pair", model_a="A", layer_a="l", role_a=3,
                 name_a="far disperser", channels_a={"horizon_shape_far": 1.1},
                 field_a="n_seasonalities",
                 model_b="B", layer_b="m", role_b=4, name_b="far disperser",
                 channels_b={"horizon_shape_far": 1.4}, field_b="n_seasonalities",
                 causal_verdict="same causal role"),
        "Removing either role moves the far horizon of the forecast the same "
        "way, and both are strongest on series carrying several seasonal "
        "components.",
    ),
    (
        Contrast(kind="pair", model_a="A", layer_a="l", role_a=5,
                 name_a="level setter", channels_a={"level": 0.9},
                 field_a="trend_scale",
                 model_b="B", layer_b="m", role_b=6, name_b="level setter",
                 channels_b={}, field_b=None, causal_verdict="not scorable"),
        "One role raises the forecast's overall level and is strongest on "
        "series with a larger trend scale, while no channel cleared its null "
        "for the other, so the comparison could not be scored.",
    ),
    (
        Contrast(kind="solo", model_a="A", layer_a="l", role_a=7,
                 name_a="seasonal amp", channels_a={"seasonal": 1.3},
                 field_a="seasonal_period_dominant", model_b="B"),
        "Removing this role raises the forecast's seasonal magnitude, most on "
        "series with a longer dominant seasonal period, and no counterpart was "
        "found in B at the layers compared.",
    ),
)


def contrast_allowed_concepts(c: Contrast) -> set:
    """Every concept a sentence about this chunk may name."""
    out = set()
    for ch in list(c.channels_a) + list(c.channels_b):
        out.add(_d.channel_concept(ch))
    for fld in (c.field_a, c.field_b):
        if fld:
            out.add(_d.field_concept(fld))
    return out


def _signed_direction(c: Contrast) -> dict:
    """`{concept: sign}` over both sides, for the up/down-word check.

    A channel both roles move keeps a direction only when they agree on it;
    when they move it opposite ways there is no single direction a sentence
    could correctly attach, so the concept is dropped from this map and any
    direction word on it is refused.
    """
    out: dict = {}
    for side in (c.channels_a, c.channels_b):
        for ch, v in (side or {}).items():
            if _is_unsigned(ch):
                continue
            key = _d.channel_concept(ch)
            sign = 1 if float(v) > 0 else -1
            if key in out and out[key] != sign:
                out[key] = 0
            else:
                out.setdefault(key, sign)
    return {k: v for k, v in out.items() if v}


def render_contrast(c: Contrast) -> str:
    """The EVIDENCE block the narrator sees. Also what the guard is derived
    from, so a field added here without a matching guard clause is a new
    fabrication surface (`CLAUDE.md` sec 11.51)."""
    lines = []
    if c.kind == "pair":
        lines.append(f"comparison: a role in {c.model_a} against a role in {c.model_b}")
    else:
        lines.append(f"comparison: a role in {c.model_a} with no counterpart found "
                     f"in {c.model_b} at the layers compared")

    def side(model, channels, field, n_atoms):
        bits = []
        if channels:
            for ch, v in channels.items():
                verb = _d.CHANNEL_VERB.get(ch, _d._DEFAULT_VERB)[0 if v > 0 else 1]
                bits.append(f"{verb} {_channel_label(ch)}")
        else:
            bits.append("no channel cleared its null")
        if field:
            bits.append(f"strongest on series with {_field_label(field)}")
        return f"{model}: " + "; ".join(bits)

    lines.append(side(c.model_a, c.channels_a, c.field_a, c.n_atoms_a))
    if c.kind == "pair":
        lines.append(side(c.model_b, c.channels_b, c.field_b, c.n_atoms_b))
        if c.causal_verdict == "same causal role":
            lines.append("verdict: removing either one pushes the forecast the "
                         "same way, past what an arbitrary pair of roles reaches")
        elif c.causal_verdict == "fires together, acts differently":
            lines.append("verdict: they fire on the same series, but removing "
                         "each one pushes the forecast a different way")
        else:
            lines.append("verdict: the causal comparison could not be scored")
    else:
        lines.append("verdict: no counterpart was found at the layers compared")
    return "\n".join(lines)


MAX_CONTRAST_WORDS = 60
MAX_CONTRAST_SENTENCES = 2
"""What the GUARD admits, deliberately looser than what the prompt asks for.

The prompt asks for one sentence of at most thirty-five words, because that
is the length that reads well. This is the length that is *admissible*, and
it is set from a measurement rather than from taste: the longest
deterministic fallback over the four-model panel is 52 words, and a guard
tighter than its own fallback rejects the module's correct output (sec
11.51). Length is a style property, truthfulness is what this function is
for, and conflating the two costs the truthful sentence.
"""

MAX_SYNTHESIS_WORDS = 60
MAX_SYNTHESIS_SENTENCES = 2


def _strip_names(raw: str, *names) -> str:
    """`raw` with the given model names removed.

    Used only before a digit scan -- see the comment at either call site.
    Removing them anywhere else would blind the forbidden-model check, which
    has to see names to refuse them.

    Shared by BOTH stages on purpose. It was written for the chunk guard,
    the synthesis guard scanned `raw` directly, and the two then disagreed
    about whether `Chronos-2` contains a number: a live four-model run had
    two of six summaries refused for "the number '2'" that a chunk sentence
    about the same model was allowed to say. Fixing the one and leaving its
    sibling is how a repaired defect comes back (sec 11.39).
    """
    out = raw
    for name in names:
        if name:
            out = re.sub(re.escape(name), " ", out, flags=re.IGNORECASE)
    return out


def _strip_model_names(raw: str, c: Contrast) -> str:
    """`raw` with this chunk's own two model names removed."""
    return _strip_names(raw, c.model_a, c.model_b)


def _shape_reason(raw: str, max_words: int, max_sentences: int) -> str:
    """The format checks, shared by both stages.

    Lifted from `describe.check_text`'s own opening rather than re-derived:
    the scaffold list, the list-marker regex and the decimal-safe sentence
    splitter each exist because a specific wording defeated a simpler
    version, and a second implementation here would re-earn those the hard
    way (sec 2.2).
    """
    if not raw:
        return "the answer was empty"
    low = raw.lower()
    for bad in _d._SCAFFOLD_SUBSTRINGS:
        if bad in low:
            return (f"the answer contained {bad!r}; write plain prose with no "
                    "markdown, no labels and no brackets")
    if _d._LIST_MARKER.search(raw):
        return "the answer was a list; write plain prose"
    words = raw.split()
    if len(words) > max_words:
        return f"the answer was {len(words)} words; write at most {max_words}"
    sentences = [s for s in _d._SENTENCE_SPLIT.split(raw) if s.strip()]
    if len(sentences) > max_sentences:
        return (f"the answer was {len(sentences)} sentences; write at most "
                f"{max_sentences}")
    if raw[-1] not in ".!?":
        return ("the answer did not end in a full stop, so it was cut off; write "
                "complete sentences")
    return ""


_ABLATION_VERBS = ("removing", "remove", "removes", "removed", "ablating",
                   "ablate", "ablates", "ablated", "zeroing", "zeroes out")
"""Verbs naming what the second causal battery does (sec 27).

It removes ONE FEATURE from the SAE's own reconstruction. It does not remove
a model, and a sentence reading "removing either Sundial or TimesFM" says
something this repo cannot do and never measured -- found in 1 of 46 accepted
texts on a live run. The possessive is what separates the two readings, so
`_ablation_referent_reason` requires one; "removing Chronos-Bolt's null" is
correct and stays legal (also live, in the same run).
"""


def _ablation_referent_reason(raw: str, names) -> str:
    """Non-empty if an ablation verb takes a MODEL as its subject or object.

    Both voices, because only the ACTIVE one was guarded and the passive
    walked straight through it: "Sundial and TimesFM both have similar
    effects WHEN REMOVED from the forecast" was accepted on a live run,
    saying the same unmeasurable thing as "removing either Sundial or
    TimesFM" with the model promoted from object to subject. A guard written
    against one voice of a verb has declared the other voice unguarded, and
    nothing in the output distinguishes that from a verb nobody needed to
    guard (sec 11.53 practice 4, at the level of grammar rather than state).
    """
    for name in names:
        if not name:
            continue
        esc = re.escape(name)
        pat = re.compile(
            r"\b(?:%s)\s+(?:either\s+|both\s+)?%s(?!['\u2019]s\b)\b"
            % ("|".join(_ABLATION_VERBS), esc), re.IGNORECASE)
        if pat.search(raw):
            return (f"the answer said it removes {name!r}; the battery removes one "
                    "FEATURE from that model's reconstruction, not the model -- "
                    f"write {name}'s role, feature or channel")
        # Passive: the model is the thing being removed, at a distance. Scoped
        # to the sentence (sec 11.53 practice 5) -- the marker lands anywhere
        # from two to a dozen words after the name in real output -- and the
        # possessive is still what makes it legal, exactly as above.
        passive = re.compile(
            r"\b%s\b(?!['\u2019]s\b)[^.;]{0,80}?\b(?:when|after|once|if)\s+"
            r"(?:they\s+are\s+|it\s+is\s+|each\s+is\s+)?(?:%s)\b"
            % (esc, "|".join(v for v in _ABLATION_VERBS if v.endswith("ed"))),
            re.IGNORECASE)
        if passive.search(raw):
            return (f"the answer said {name!r} is what gets removed; the battery "
                    "removes one FEATURE from that model's reconstruction, not "
                    f"the model -- write {name}'s role, feature or channel")
    return ""


_UNMEASURED_PROCESS_TERMS = (
    "handle", "handles", "handling", "process", "processes", "processing",
    "deal with", "deals with", "dealing with", "manage", "manages",
    "managing", "interpret", "interprets", "interpreting",
    "understand", "understands", "understanding",
)
"""Verbs naming what a model DOES INTERNALLY with a kind of input.

Nothing here measures that. The two batteries measure which series a feature
fires on and which forecast channels removing it moves; "handles seasonal
patterns similarly" is a claim about internal processing that neither can
support, and it reached an accepted summary the moment "reliability" and the
field-referent guard closed the wordings around it. Its own category rather
than an entry in `_SUPERIORITY_TERMS`, because it is not an evaluative claim
-- it is an unmeasured-mechanism claim, and collapsing the two would make the
refusal reason say the wrong thing to the retry (sec 11.53 practice 4).
"""


_FORECAST_ACTION_VERBS = _ABLATION_VERBS + (
    "reduce", "reduces", "reducing", "increase", "increases", "increasing",
    "boost", "boosts", "boosting", "shift", "shifts", "shifting",
    "lower", "lowers", "lowering", "raise", "raises", "raising",
    "decrease", "decreases", "decreasing", "move", "moves", "moving",
    "suppress", "suppresses", "suppressing", "eliminate", "eliminates",
    "eliminating", "filter out", "smooth out",
)
"""Verbs whose object is something the intervention DOES to the forecast."""

_FIRING_PREPOSITIONS = re.compile(
    r"\b(?:on|with|for|carrying|those|containing|having|exhibit|exhibits|"
    r"exhibiting|show|shows|showing)\b[^.;]{0,25}$", re.IGNORECASE)
"""Trailing context that marks the term as a property series HAVE, not an
object the intervention acts on -- "fire on the same series, those with noise
scale" must stay legal, and it is this module's own `phrase()` wording."""

_FORECAST_POSSESSIVE = re.compile(
    r"\bforecast(?:['\u2019]s)?\s+(?:own\s+|overall\s+)?$", re.IGNORECASE)
"""A term written as a property OF THE FORECAST is a channel by construction.

The second, independent discriminator, and the one that makes this guard
safe when the allowed set is thin: "decreases the forecast's trend slope" and
"both remove noise" differ by exactly this, and only the first names
something the intervention measured. Without it the guard's correctness rests
entirely on `channel:trend` being licensed -- true on the live run and true by
design, but a guard with one leg is one evidence change from refusing this
module's own `phrase()` output (sec 11.51 lesson 3).
"""

_NOMINALIZED_ACTION = re.compile(
    r"\s*(?:removal|reduction|suppression|elimination|smoothing)\b",
    re.IGNORECASE)


def _field_as_forecast_object_reason(raw: str, allowed) -> str:
    """Non-empty if a SERIES property is used as something the model changes.

    🔴 This is the one leak the two-stage reduction's vocabulary narrowing
    provably cannot catch, and it produced the most misleading sentence on a
    live four-model panel: "Chronos-2 and TimesFM both REMOVE NOISE from time
    series data", from chunks that said only "both roles fire on the same
    series, THOSE WITH NOISE SCALE". Every word was licensed. What was
    invented is the grammatical role -- `field:noise_scale` describes the
    series a role FIRES ON, `channel:*` describes what removing it MOVES, and
    the two are different measurements the concept namespace already keeps
    apart. `synthesis_evidence` narrows WHICH concepts may be named and had
    no opinion on HOW, so a field term walked into the object slot of an
    ablation verb and read as a causal claim nothing measured (sec 11.53's
    referent shape, at the field/channel boundary rather than the model one).

    A term counts as field-only *in this evidence*: its licensed intersection
    with `allowed` must be non-empty and contain no `channel:`. So "trend" is
    untouched wherever a chunk moved `channel:trend`, and refused only where
    the evidence knows it solely as a property of the input -- the same term
    is legal or not depending on what was actually measured, which is the
    only reading that can be right for a word like "seasonal".

    Measured before it was written (sec 11.53 practice 1): on the live run it
    refuses exactly the 2 fabricating summaries and 0 of 63 chunk sentences,
    0 of the module's own machine fallbacks included.
    """
    allowed = set(allowed or ())
    if not allowed:
        return ""
    norm = _d._normalize(raw)
    for term, concepts in _d.DOMAIN_TERMS.items():
        inter = set(concepts) & allowed
        if not inter or any(c.startswith("channel:") for c in inter):
            continue
        pattern = _d._TERM_PATTERNS[term]
        for m in pattern.finditer(norm):
            before = norm[max(0, m.start() - 40):m.start()]
            after = norm[m.end():m.end() + 24]
            if _NOMINALIZED_ACTION.match(after):
                return (f"the answer made {term!r} something a model acts on; the "
                        "evidence names it only as a property of the series these "
                        "features FIRE ON, never as anything the forecast does")
            if _FIRING_PREPOSITIONS.search(before):
                continue
            if _FORECAST_POSSESSIVE.search(before):
                continue
            if any(_d._term_pattern(v).search(before)
                   for v in _FORECAST_ACTION_VERBS):
                return (f"the answer made {term!r} the object of something a model "
                        "does; the evidence names it only as a property of the "
                        "series these features FIRE ON, never as an effect on the "
                        "forecast")
    return ""


_COUNTERPART_NOUNS = ("counterpart", "corresponding role", "equivalent role",
                      "matching role", "matched role", "similar role")
"""Nouns that name a role in the OTHER model.

In a `solo` chunk there IS no such role -- that is the whole content of the
state. Yet 6 of 24 accepted solo sentences on a live four-model panel asserted
a comparison against one ("decreased the forecast's trend slope compared to
its counterpart in Chronos-Bolt", "differs from a role in Sundial by moving
the near horizon"), which is sec 11.37's shape once more: an absent baseline
producing a CONFIDENT claim rather than a cautious one, reached through a
vocabulary the guard had no branch for. `check_contrast_text` scored the
verdict terms only `if c.kind == "pair"`, so the solo state -- the
highest-acceptance state at 24 of 25 -- was scanned for everything except the
one thing that distinguishes it.
"""

_ABSENCE_MARKERS = ("no", "not", "none", "never", "without", "absent",
                    "lacks", "lacking", "unmatched", "nothing", "neither")
"""Words that turn a counterpart reference into a statement of its absence.

The boundary is a PAIR of near-identical sentences (sec 11.51 lesson 2): "no
counterpart was found in B" is this module's own fallback and must stay legal,
while "compared to its counterpart in B" must not. Scope is the SENTENCE, not
a fixed window, because the marker lands on either side of the noun in real
output -- "no such counterpart was found" puts it before, "this counterpart
was not found" puts it two words after, and "No counterpart was found at the
layers compared between a role in A and a role in B" governs a second
reference ten words downstream. A window tight enough to bind the first would
refuse the third, which is correct text.
"""


def _solo_counterpart_reason(raw: str, c: Contrast) -> str:
    """Non-empty if a SOLO sentence talks about a counterpart as if one exists.

    A residual this deliberately does not chase: a sentence carrying an
    absence marker for some OTHER reason ("does not move the near horizon,
    and its counterpart ...") passes. Sentence scope is what keeps the
    module's own fallback legal, and a guard that rejects its own fallback is
    describing something other than truthfulness (sec 11.51 lesson 3).
    """
    if c.kind != "solo":
        return ""
    ref = list(_COUNTERPART_NOUNS)
    if c.model_b:
        # "a role in Sundial" attributes a role to the other model just as
        # squarely as the noun "counterpart" does, and was the wording of 3
        # of the 6 live fabrications -- none of which contain "counterpart".
        ref.append(r"roles?\s+(?:\w+\s+){0,2}?in\s+" + re.escape(c.model_b))
    absence = re.compile(r"\b(?:%s)\b" % "|".join(_ABSENCE_MARKERS), re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+", raw):
        for term in ref:
            if not re.search(r"\b" + term + r"\b", sentence, re.IGNORECASE):
                continue
            if absence.search(sentence):
                continue
            return ("the answer compared this role against a counterpart in "
                    f"{c.model_b}; no counterpart was found there at the "
                    "layers compared, so say that it has none rather than "
                    "what it differs from")
    return ""


def check_contrast_text(text: str, c: Contrast) -> str:
    """Empty string if the sentence is admissible, else why it is not.

    Layered the same way `describe.check_text` is -- shape, then vocabulary,
    then direction, then the claims specific to this comparison -- so a
    rejection reason names one thing and a retry has something to act on.
    """
    raw = (text or "").strip()
    reason = _shape_reason(raw, max_words=MAX_CONTRAST_WORDS,
                           max_sentences=MAX_CONTRAST_SENTENCES)
    if reason:
        return reason
    norm = _d._normalize(raw)

    for other in c.forbidden_models or ():
        if _d._term_pattern(other).search(norm):
            return (f"the answer named {other!r}, which is not one of the two "
                    "models this comparison measured")

    for term in _d.UNLICENSED_TERMS:
        if _d._UNLICENSED_PATTERNS[term].search(norm):
            return (f"the answer said {term!r}, which nothing in the evidence "
                    "measures; say only what was measured")

    for term in _UNMEASURED_PROCESS_TERMS:
        if _d._term_pattern(term).search(norm):
            return (f"it said {term!r}; nothing here measures how a model "
                    "processes anything internally -- say which series the "
                    "features fire on, or which channel removing them moves")

    reason = _ablation_referent_reason(raw, (c.model_a, c.model_b))
    if reason:
        return reason

    reason = _solo_counterpart_reason(raw, c)
    if reason:
        return reason

    for fld in _d._RAW_FIELD_PATTERNS:
        if _d._RAW_FIELD_PATTERNS[fld].search(norm):
            return (f"the answer used the raw field name {fld!r}; write it in "
                    f"plain English as {_d.FIELD_GLOSS[fld]!r}")

    allowed = contrast_allowed_concepts(c)

    reason = _field_as_forecast_object_reason(raw, allowed)
    if reason:
        return reason
    for term, concepts in _d.DOMAIN_TERMS.items():
        if not _d._TERM_PATTERNS[term].search(norm):
            continue
        if not (set(concepts) & allowed):
            return (f"the answer mentioned {term!r}, which this comparison did "
                    "not measure; mention only what the evidence lists")

    for term in _CAPABILITY_TERMS:
        if _d._term_pattern(term).search(norm):
            return (f"the answer said {term!r}; roles were compared at these "
                    "layers only, which is not a statement about what a model "
                    "is able to represent")

    for term in _d.QUALITY_WORSE + _d.QUALITY_BETTER + _SUPERIORITY_TERMS:
        if _d._term_pattern(term).search(norm):
            return (f"the answer said {term!r}; this comparison says nothing "
                    "about how well either model forecasts")

    # A direction verb is a claim about a channel's sign, and here there are
    # two sides. A channel both roles move in OPPOSITE directions licenses
    # neither word -- `_signed_direction` drops it -- so the sentence cannot
    # borrow one side's direction and attach it to the comparison.
    signed = _signed_direction(c)
    has_up = any(v > 0 for v in signed.values())
    has_down = any(v < 0 for v in signed.values())
    for term in _d.UP_VERBS:
        if _d._UP_PATTERNS[term].search(norm) and not has_up:
            return (f"the answer said {term!r}, but no channel here has a "
                    "measured upward direction; a near or far horizon effect "
                    "has a size, not a direction")
    for term in _d.DOWN_VERBS:
        if _d._DOWN_PATTERNS[term].search(norm) and not has_down:
            return (f"the answer said {term!r}, but no channel here has a "
                    "measured downward direction; a near or far horizon effect "
                    "has a size, not a direction")

    for pattern in (_d._DIRECTION_ON_HORIZON, _d._HORIZON_THEN_DIRECTION):
        for hit in pattern.finditer(norm):
            if (hit.groupdict().get("lead") or "").strip():
                continue
            return (f"the answer said {hit.group(0)!r}; a near or far horizon "
                    "effect is a size, not a direction")

    # 🔴 Model names are stripped BEFORE the digit scan, not after. A
    # checkpoint called `Chronos-2` puts a digit in every honest sentence
    # about it, and the first version of this guard rejected 19 of this
    # module's own fallbacks for containing "the number '2'" -- the same
    # collision that once turned `Chronos-2` into `Chronos-N` and made 19
    # repeating report templates read as 1 (sec 24.6). The names are
    # licensed by `render_contrast`, so removing them leaves exactly the
    # digits nothing licensed.
    for tok in _d._DIGITS.findall(_strip_model_names(raw, c)):
        return (f"the answer contained the number {tok!r}; write the sentence "
                "with no numbers at all")

    # 🔴 "fire on the same series" is the LICENSED half of a disagreement
    # verdict -- the two roles do fire together, that is what makes their
    # causal disagreement worth reporting at all -- so the agreement scan
    # elides it first. Without this the guard rejected 13 of its own
    # fallbacks for saying "same", i.e. for stating the true half of the
    # finding it exists to protect. `same way`/`same direction` survive the
    # elision, which is the pair of near-identical sentences the test pins.
    verdict_norm = _AGREEMENT_ELIDE.sub(" ", norm)

    verdict = c.causal_verdict
    if c.kind == "pair":
        if verdict == "fires together, acts differently":
            for term in AGREEMENT_TERMS:
                if _d._term_pattern(term).search(verdict_norm):
                    return (f"it said {term!r}, but the measurement is that these "
                            "two act differently; that inverts the finding")
        elif verdict == "same causal role":
            for term in DISAGREEMENT_TERMS:
                if _d._term_pattern(term).search(verdict_norm):
                    return (f"it said {term!r}, but the measurement is that these "
                            "two act the same way; that inverts the finding")
        else:
            for term in AGREEMENT_TERMS + DISAGREEMENT_TERMS:
                if _d._term_pattern(term).search(verdict_norm):
                    return (f"it said {term!r}, but this comparison could not be "
                            "scored, so it must take no side")
    return ""


# The marker `describe_contrasts` records instead of a refusal reason when a
# chunk was never sent to the narrator. Read by `run_sae_compare.py` (study
# driver, dev branch) and by
# `report/derived.py::sae_contrast_chunks`, which both report three states --
# narrated, refused, and deliberately deterministic -- because collapsing the
# last two labels a run where nothing failed as a run of failures (sec 11.37).
DETERMINISTIC_BY_DESIGN = (
    "deterministic by design: an unscorable pair's only content is the "
    "measured reason it could not be scored, and across the runs measured so "
    "far the narrator supplied a different, unlicensed reason instead")


def unscorable_clause(c: Contrast, brief: bool = False) -> str:
    """Why this pair could not be scored, in the reader's terms.

    `Contrast.causal_reason` carries `sae/matching.py`'s own wording, which
    names the side as "A"/"B" -- correct inside that module and meaningless
    in a report where the two models have names. This translates it and
    falls back to the bare state, never to an invented cause.

    `brief` exists because the two consumers have different budgets and the
    long form breaks one of them: `check_contrast_text` caps a chunk
    sentence at 60 words, and the full clause pushed 11 of 63 machine
    fallbacks past it -- sec 11.51 lesson 3, a guard refusing its own
    module's fallback, this time by length rather than vocabulary. The
    report table has no such cap and renders the full wording.
    """
    r = (c.causal_reason or "").strip()
    if not r:
        return "not scorable" if brief else "the causal comparison could not be scored"
    lower = r.lower()
    side = (c.model_a if lower.startswith("side a:")
            else c.model_b if lower.startswith("side b:") else None)
    body = r.split(":", 1)[1].strip() if side is not None else r
    if brief:
        if side is not None and "cleared any channel" in body:
            return f"not scorable — no {side} feature in this role cleared its null"
        if side is not None:
            return f"not scorable — on the {side} side, {body}"
        if "cross-pairs" in body:
            return "not scorable — too few cross-model pairs to form a null"
        return "not scorable"
    if side is not None:
        return f"the causal comparison could not be scored: in {side}, {body}"
    return f"the causal comparison could not be scored: {body}"


def contrast_machine_fallback(c: Contrast) -> str:
    """The deterministic sentence used when generation is refused.

    Written in the same correspondence terms the guard enforces, so
    `check_contrast_text(contrast_machine_fallback(c), c)` is empty for every
    chunk -- pinned by a test, because a guard that rejects its own fallback
    is describing something other than truthfulness (`CLAUDE.md` sec 11.51).
    """
    def phrase(channels, field):
        bits = []
        for ch, v in (channels or {}).items():
            verb = _d.CHANNEL_VERB.get(ch, _d._DEFAULT_VERB)[0 if v > 0 else 1]
            bits.append(f"{verb} {_channel_label(ch)}")
        if not bits:
            bits.append("moves no channel past its own null")
        if field:
            bits.append(f"strongest on series with {_field_label(field)}")
        return ", ".join(bits)

    if c.kind == "solo":
        return (f"In {c.model_a} this role {phrase(c.channels_a, c.field_a)}; "
                f"no counterpart was found in {c.model_b} at the layers compared.")
    head = (f"In {c.model_a} this role {phrase(c.channels_a, c.field_a)}, and in "
            f"{c.model_b} its counterpart {phrase(c.channels_b, c.field_b)}")
    if c.causal_verdict == "same causal role":
        return head + "; removing either pushes the forecast the same way."
    if c.causal_verdict == "fires together, acts differently":
        return head + ("; they fire on the same series, yet removing each pushes "
                       "the forecast a different way.")
    return head + "; " + unscorable_clause(c, brief=True) + "."


# ---------------------------------------------------------------------------
# Stage 1: one guarded sentence per chunk.
# ---------------------------------------------------------------------------

def _contrast_messages(c: Contrast, history: list) -> list:
    msgs = [{"role": "system", "content": CONTRAST_SYSTEM_PROMPT}]
    for shot_c, sentence in FEW_SHOT_CONTRASTS:
        msgs.append({"role": "user", "content": render_contrast(shot_c)})
        msgs.append({"role": "assistant", "content": sentence})
    msgs.append({"role": "user", "content": render_contrast(c)})
    for rejected, reason in history:
        msgs.append({"role": "assistant", "content": rejected})
        msgs.append({"role": "user", "content":
                     f"That was rejected because {reason}. Write the sentence "
                     "again, obeying every rule."})
    return msgs


def _is_unscorable_pair(c: Contrast) -> bool:
    return c.kind == "pair" and c.causal_verdict == "not scorable"


def describe_contrasts(chunks: list, narrator,
                       prefer_deterministic_unscorable: bool = True) -> list:
    """One `describe.Description` per chunk, same order.

    Batched by retry ROUND exactly as `describe.describe_batch` is, so the
    number of forward passes stays bounded by `MAX_ATTEMPTS` however many
    chunks a panel produces -- a four-model run has six pairs and can reach
    well over a hundred chunks, which is the whole reason the user asked for
    chunking rather than one long prompt.

    🔴 `prefer_deterministic_unscorable` defaults True, for the same reason
    and on the same evidence as `compare_models`' own
    `prefer_deterministic_summary` (sec 28.12). An unscorable pair's packet
    carries exactly one thing the scored pairs do not: the measured reason it
    could not be scored. The narrator does not have it -- it is not in the
    prompt as anything it may quote -- so it supplies one, and on
    `runs/full_report_run_4model` every accepted unscorable sentence supplied
    the SAME invented cause ("due to differences in the roles' effects on the
    forecast"), which for the 25-of-27 majority case is the opposite of the
    truth: there was no measurable effect on one side to differ. No
    vocabulary scan can catch that -- the sentence is fluent, every term in
    it is licensed, and it is wrong about a fact the guard has no field for
    (sec 11.53's tenth instance). `unscorable_clause` states the measured
    reason instead. Pass False to restore generation for these chunks; the
    scored chunks are untouched either way, and their acceptance rate is
    therefore comparable across this change.
    """
    chunks = list(chunks)
    if not chunks:
        return []
    if narrator is None:
        return [_d.Description(text=contrast_machine_fallback(c), accepted=False,
                               reason="no narrator loaded", attempts=0,
                               model_id=_d.MODEL_ID, revision=_d.REVISION)
                for c in chunks]

    results: list = [None] * len(chunks)
    pending = []
    for i, c in enumerate(chunks):
        if prefer_deterministic_unscorable and _is_unscorable_pair(c):
            results[i] = _d.Description(
                text=contrast_machine_fallback(c), accepted=False,
                reason=DETERMINISTIC_BY_DESIGN, attempts=0,
                model_id=narrator.model_id, revision=narrator.revision)
        else:
            pending.append(i)
    histories: dict = {i: [] for i in pending}
    for attempt in range(1, _d.MAX_ATTEMPTS + 1):
        if not pending:
            break
        texts = _d._generate(
            narrator, [_contrast_messages(chunks[i], histories[i]) for i in pending])
        still = []
        for i, text in zip(pending, texts):
            reason = check_contrast_text(text, chunks[i])
            if not reason:
                results[i] = _d.Description(
                    text=text.strip(), accepted=True, reason="", attempts=attempt,
                    model_id=narrator.model_id, revision=narrator.revision)
            else:
                histories[i].append((text.strip(), reason))
                still.append(i)
        pending = still
    for i in pending:
        last = histories[i][-1][1] if histories[i] else "generation failed"
        results[i] = _d.Description(
            text=contrast_machine_fallback(chunks[i]), accepted=False, reason=last,
            attempts=_d.MAX_ATTEMPTS, model_id=narrator.model_id,
            revision=narrator.revision)
    return results


# ---------------------------------------------------------------------------
# Stage 2: synthesis over the accepted chunk sentences.
# ---------------------------------------------------------------------------

def _attributed(text: str, c: Contrast) -> str:
    """The sentence, with the model whose role it describes made explicit.

    🔴 The cause of stage 2's mis-attributions, and it is an INPUT defect
    rather than a narrator one. `phrase()` writes a solo chunk as "Removing
    this role moves the far horizon ... and no counterpart was found at the
    layers compared" -- correct and unambiguous in the per-pair table, where
    the row names the role and its model in its own columns. Handed to the
    synthesis as a bare bullet it says only that SOMEBODY's role does this,
    and 21 of 25 accepted solo sentences on a live four-model panel named no
    owner at all. Stage 2 then guessed, and guessed wrong twice in one run --
    attributing Sundial's flatness and spread effects to Chronos-2, with every
    model name and every channel individually licensed, so no vocabulary guard
    could see it.

    Attribution is restored from the `Contrast` the sentence was generated
    from, never inferred from the text, so it cannot be wrong. Only a sentence
    that does not already name its owner is prefixed; the pair chunks
    overwhelmingly do, and reprefixing them would read as a second speaker.
    """
    owner = c.model_a
    if not owner or owner in text:
        return text
    if c.kind == "solo":
        return f"In {owner}: {text}"
    return text


def synthesis_evidence(cs: ContrastSet, descriptions: list) -> dict:
    """The stage-2 input: the accepted stage-1 sentences and the concepts
    they actually used.

    🔴 The synthesis vocabulary is the UNION OF WHAT THE CHUNKS SAID, not the
    union of what they were licensed to say. That is the property that makes
    a two-stage reduction safe: stage 2 can only narrow. A concept a chunk
    was allowed to mention but did not is not available here, so the
    synthesis cannot reach past its own inputs to a fact no sentence
    established -- which is the failure mode "chunk then reason over the
    chunks" invites, and the reason the intermediate sentences are persisted
    rather than kept as hidden scratchpad.
    """
    used: set = set()
    kept = []
    for c, desc in zip(cs.chunks, descriptions):
        if not desc.accepted:
            continue
        norm = _d._normalize(desc.text)
        licensed = contrast_allowed_concepts(c)
        for term, concepts in _d.DOMAIN_TERMS.items():
            if _d._TERM_PATTERNS[term].search(norm):
                used |= (set(concepts) & licensed)
        kept.append(_attributed(desc.text.strip(), c))
    moved_a, moved_b = [], []
    for c in cs.chunks:
        for ch in (c.channels_a or {}):
            if ch not in moved_a:
                moved_a.append(ch)
        for ch in (c.channels_b or {}):
            if ch not in moved_b:
                moved_b.append(ch)
    return {"model_a": cs.model_a, "model_b": cs.model_b,
            "sentences": kept, "concepts": used,
            "n_chunks": len(cs.chunks), "n_accepted": len(kept),
            "n_agree": cs.n_agree, "n_disagree": cs.n_disagree,
            "n_unscorable": cs.n_unscorable,
            "moved_a": moved_a, "moved_b": moved_b}


def check_synthesis_text(text: str, ev: dict) -> str:
    """Admissibility for the stage-2 summary."""
    raw = (text or "").strip()
    reason = _shape_reason(raw, max_words=MAX_SYNTHESIS_WORDS,
                           max_sentences=MAX_SYNTHESIS_SENTENCES)
    if reason:
        return reason
    norm = _d._normalize(raw)

    # Model names are stripped BEFORE the digit scan here for the same
    # reason as in `check_contrast_text`, and the omission was not
    # hypothetical: on a live four-model panel this refused two of six
    # summaries for containing "the number '2'", where the 2 was the `2` in
    # `Chronos-2` and every honest sentence naming that model carries it.
    for tok in _d._DIGITS.findall(_strip_names(raw, ev.get("model_a"),
                                               ev.get("model_b"))):
        return (f"the answer contained the number {tok!r}; this summary may not "
                "quantify anything")

    reason = _ablation_referent_reason(raw, (ev.get("model_a"), ev.get("model_b")))
    if reason:
        return reason

    allowed = set(ev.get("concepts") or ())

    reason = _field_as_forecast_object_reason(raw, allowed)
    if reason:
        return reason
    for term, concepts in _d.DOMAIN_TERMS.items():
        if not _d._TERM_PATTERNS[term].search(norm):
            continue
        if not (set(concepts) & allowed):
            return (f"the answer named {term!r}, which none of the checked "
                    "findings mentioned; a summary may only narrow its inputs")
    for term in _SHARE_QUANTIFIERS:
        if _d._term_pattern(term).search(norm):
            return (f"it said {term!r}; how much two models share is the number "
                    "that has no untrained-twin floor in this run and may not be "
                    "quantified, in words or otherwise")
    for term in _CAPABILITY_TERMS:
        if _d._term_pattern(term).search(norm):
            return (f"it said {term!r}; these findings are about correspondence "
                    "at the layers compared, not about what a model can represent")

    for term in _UNMEASURED_PROCESS_TERMS:
        if _d._term_pattern(term).search(norm):
            return (f"it said {term!r}; nothing here measures how a model "
                    "processes anything internally -- say which series the "
                    "features fire on, or which channel removing them moves")
    # 🔴 `_SUPERIORITY_TERMS` is why this line changed. Stage 2 already ran a
    # capability scan and a quality scan -- it was never missing the scans,
    # it was missing the WORDS. On the first live run it accepted a summary
    # reading "differ significantly in their ability to ..." and "TimesFM
    # outperforms Chronos-Bolt" for a pair whose every role comparison was
    # unscorable, because "ability" was not in `_CAPABILITY_TERMS` (only "no
    # ability") and "outperforms" is in neither QUALITY tuple. A guard is its
    # vocabulary, and a scan over a vocabulary missing the word reads exactly
    # like no scan at all (sec 11.51).
    for term in _d.QUALITY_WORSE + _d.QUALITY_BETTER + _SUPERIORITY_TERMS:
        if _d._term_pattern(term).search(norm):
            return f"it said {term!r}; this summary says nothing about forecast quality"
    return ""


def synthesis_fallback(ev: dict) -> str:
    """Deterministic stage-2 text, composed from the MEASURED tallies.

    🔴 Rewritten 2026-09-09. It used to say only "individual role comparisons
    are listed below; no summary of them passed its own check" -- true, and
    worth nothing to a reader, which mattered more than it looks: after the
    referent guards landed, half the pairs on a live four-model panel degrade
    to this sentence, so the fallback IS the summary layer for most of the
    section. The tallies and the moved channels are already measured and
    already in `ContrastSet`, so a real summary here is a pure reduction that
    cannot be wrong -- strictly better than both the apology it replaces and
    the generated sentence it stands in for, which on this run kept reaching
    fabrications through fresh paraphrases of banned words (sec 26 C's
    blacklist-defeated-on-re-measure finding, at a second site).

    Degrading to the inputs rather than to silence is deliberate -- the
    chunk sentences are each individually checked, so showing all of them is
    strictly more evidence than showing a summary, just less readable. A
    summary layer that cannot be produced honestly must under-claim, which
    here means declining to compress rather than compressing loosely.
    """
    a, b = ev.get("model_a"), ev.get("model_b")
    n = ev.get("n_accepted") or 0
    agree = int(ev.get("n_agree") or 0)
    differ = int(ev.get("n_disagree") or 0)
    unscored = int(ev.get("n_unscorable") or 0)
    scored = agree + differ + unscored
    if not n and not scored:
        return (f"No comparison between {a} and {b} passed its own check at "
                "this pair.")

    def _channels(keys):
        gl = [_d.CHANNEL_GLOSS.get(k, k) for k in (keys or [])][:3]
        if not gl:
            return ""
        if len(gl) == 1:
            return gl[0]
        return ", ".join(gl[:-1]) + " and " + gl[-1]

    # No counts, in digits OR words. The digit ban's stated reason is that how
    # much two models share has no untrained-twin floor in this run and "may
    # not be quantified, in words or otherwise" -- and a first draft of this
    # very sentence was refused by that ban, which is sec 11.51 lesson 3
    # working as designed rather than a guard to loosen. The exact tallies are
    # in the table this sentence sits above, so the prose states the SHAPE of
    # the outcome and the table carries the numbers.
    parts = []
    if scored:
        if agree and differ:
            parts.append("Matched role pairs here are split: some acted alike "
                         "and some acted differently.")
        elif differ:
            parts.append("No matched role pair here acted alike; those that "
                         "could be scored acted differently.")
        elif agree:
            parts.append("Every matched role pair here that could be scored "
                         "acted alike.")
        else:
            parts.append("No matched role pair here could be scored either way.")
    ca, cb = _channels(ev.get("moved_a")), _channels(ev.get("moved_b"))
    if ca and cb:
        parts.append(f"{a}'s roles here move {ca}; {b}'s move {cb}.")
    elif ca or cb:
        who, what = (a, ca) if ca else (b, cb)
        parts.append(f"Only {who}'s roles moved anything here: {what}.")
    if not parts:
        return (f"Individual role comparisons between {a} and {b} are listed "
                "below; no summary of them passed its own check.")
    return " ".join(parts)


def _synthesis_messages(ev: dict, history: list) -> list:
    body = "\n".join(f"- {s}" for s in ev.get("sentences") or [])
    user = (f"findings comparing {ev.get('model_a')} and {ev.get('model_b')}:\n"
            f"{body}")
    msgs = [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    for rejected, reason in history:
        msgs.append({"role": "assistant", "content": rejected})
        msgs.append({"role": "user", "content":
                     f"That was rejected because {reason}. Write it again, "
                     "obeying every rule."})
    return msgs


def compare_models(match_table: dict, roles_json: dict, narrator=None,
                   cosine_threshold: float = 0.5,
                   prefer_deterministic_summary: bool = True) -> dict:
    """The whole two-stage pass: chunk, narrate each chunk, then synthesise.

    🔴 `prefer_deterministic_summary` defaults True, and that is a decision
    made on measurement rather than a preference. Stage 1 is reliable -- 61 of
    63 chunk sentences accepted on a live four-model panel, and every one read
    against its own evidence was faithful. Stage 2 is not, and it does not
    converge: across FOUR consecutive regenerations, each closing the previous
    run's wording, the accepted summary produced a fresh class of fabrication
    every time -- banned words and an evaluative channel gloss, then a
    paraphrase of a just-banned word, then attributing one model's measured
    effect to the other, then comparative quantification ("a greater effect
    ... more often") for a pair where nothing was scorable at all. Each class
    was individually guardable and the next one was not foreseeable from it.
    Meanwhile `synthesis_fallback` composes the same reduction from the
    measured tallies and cannot be wrong. So the deterministic sentence is
    what renders; the generated one is still produced, guarded and PERSISTED
    (`summary_generated`), because a decision recorded without the evidence it
    was made against goes stale the way sec 11.29's docstring premise did.
    Set False to render the generated summary instead.

    Returns one record per model pair carrying BOTH stages, because the
    synthesis is only trustworthy to the extent its inputs are visible --
    a reader who doubts the summary can read the checked sentences it was
    built from, and each of those names the role it came from.
    """
    sets = contrast_chunks(match_table, roles_json, cosine_threshold=cosine_threshold)
    out = {"pairs": [], "model_id": _d.MODEL_ID, "revision": _d.REVISION,
           "cosine_threshold": float(cosine_threshold)}
    for cs in sets:
        descs = describe_contrasts(cs.chunks, narrator)
        ev = synthesis_evidence(cs, descs)
        deterministic = synthesis_fallback(ev)
        summary_text, summary_ok, summary_reason = deterministic, False, "no narrator loaded"
        gen_text, gen_ok = "", False
        if narrator is not None and ev["n_accepted"]:
            history: list = []
            for _ in range(_d.MAX_ATTEMPTS):
                text = _d._generate(narrator, [_synthesis_messages(ev, history)])[0]
                reason = check_synthesis_text(text, ev)
                if not reason:
                    gen_text, gen_ok, summary_reason = text.strip(), True, ""
                    if not prefer_deterministic_summary:
                        summary_text, summary_ok = gen_text, True
                    break
                history.append((text.strip(), reason))
                summary_reason = reason
        elif narrator is not None:
            summary_reason = "no chunk sentence passed its own check"

        out["pairs"].append({
            "model_a": cs.model_a, "model_b": cs.model_b,
            "match_rate": cs.match_rate,
            "match_rate_quotable": cs.match_rate_quotable,
            "n_agree": cs.n_agree, "n_disagree": cs.n_disagree,
            "n_not_scorable": cs.n_unscorable,
            "summary": summary_text, "summary_accepted": summary_ok,
            "summary_reason": summary_reason,
            # The generated attempt stays on record whether or not it is the
            # one rendered, so the decision below is auditable against its own
            # inputs the way both stages already are.
            "summary_generated": gen_text,
            "summary_generated_accepted": gen_ok,
            "summary_is_deterministic": not summary_ok,
            "chunks": [
                {"kind": c.kind, "model_a": c.model_a, "layer_a": c.layer_a,
                 "role_a": c.role_a, "name_a": c.name_a,
                 "model_b": c.model_b, "layer_b": c.layer_b, "role_b": c.role_b,
                 "name_b": c.name_b, "cosine": c.cosine,
                 "population_null_p95": c.population_null_p95,
                 "causal_verdict": c.causal_verdict,
                 "causal_cosine": c.causal_cosine,
                 "causal_reason": c.causal_reason,
                 "text": d.text, "accepted": d.accepted, "reason": d.reason}
                for c, d in zip(cs.chunks, descs)],
        })
    log.info("sae compare: %d model pairs, %d chunks, %d summaries accepted",
             len(out["pairs"]), sum(len(p["chunks"]) for p in out["pairs"]),
             sum(1 for p in out["pairs"] if p["summary_accepted"]))
    return out
