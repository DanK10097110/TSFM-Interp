# ROADMAP.md — Where This Repo Is Going

> **Relationship to CLAUDE.md.** `CLAUDE.md` is the onboarding manual: what
> exists, why it was built that way, which invariants are load-bearing. This
> file is the *forward-looking* complement: what happens next, in what order,
> and why. `CLAUDE.md` describes the machine as built; `ROADMAP.md` describes
> where the machine is headed and tracks what's actually been done toward
> that. Read `CLAUDE.md` first for grounding, then this file for direction.
>
> **This file is living.** Every session that makes meaningful progress
> should update it: check off completed items, add findings to the relevant
> section, append an entry to the [Session Log](#14-session-log), and — this
> is important — **correct anything here that turns out to be wrong once
> tested.** A roadmap that silently drifts from reality is worse than no
> roadmap. If you discover a stated plan is a bad idea, don't just abandon it
> quietly: mark it explicitly abandoned and say why (§13 has a place for
> this), the same way `CLAUDE.md` §11 records traps instead of erasing them.
>
> **How this file is organized.** §0 is the working discipline; **§0.5 is the
> queue and the only place work is sequenced**; §1–§3 are the north star,
> principles and phase rationale; §4–§13 are the phases, each with an
> append-only Findings block that is the actual research record; §14 is the
> session log; §15–§21 are sized-but-unsequenced pools of audit and
> enhancement items; **§22 is the parked list**, with an un-park trigger per
> entry. Removed scaffolding — superseded plans, closed fix plans, the
> long-form session narrative — lives verbatim in `ROADMAP_ARCHIVE.md`, which
> needs no maintenance and is **not** a second source of truth.
>
> **Streamlining pass 2026-08-18.** This file was ~16.4k lines and held four
> competing orderings of the same ~60 items, plus a live fix plan for each of
> 19 already-fixed bugs. Every **Findings block, number, decision and
> refutation is preserved verbatim** — those are the record. What was removed
> is dead scaffolding, and it was *relocated*, not deleted, per §0.2. See §14's
> entry for the reasoning.

---

## 0. How to use this document

1. **Before starting work in a session**, read the phase you're picking up
   (§4–§9), its current status checklist, and any linked findings. Don't
   re-derive context that's already written down.
2. **Every phase section has three parts**: *Goal* (why this exists),
   *Concrete deliverables* (checklist, update in place), *Findings /
   decisions* (append-only log of what was learned — never delete a finding,
   even a "this didn't work" one; that's exactly the kind of thing `CLAUDE.md`
   §11 exists to preserve).
3. **Status tags**: `[ ]` not started · `[~]` in progress · `[x]` done ·
   `[!]` blocked (say on what) · `[-]` abandoned (say why, don't delete).
4. **Do not silently reorder the roadmap.** §3 gives an explicit dependency
   rationale for the phase ordering. If you think a different order is
   better, write the argument down in §13 and get it into a session log entry
   before deviating — the ordering exists because later phases consume
   artifacts earlier phases produce (e.g. the SAE work in Phase 2 needs the
   multi-domain runs from Phase 1 to know which layers are worth training on).
5. **This is a research repo, not just an engineering one.** Several sections
   below are hypotheses, not specs ("we *think* effective dimensionality
   predicts interpretability — go test it"). Follow `CLAUDE.md` §2.4: run the
   experiment, don't assume the answer. When a hypothesis is tested, record
   the result in that section's Findings block regardless of which way it
   came out.
6. **Every new report figure/section needs a plain-language explanation
   attached** (§8) — this is a hard requirement from here forward, not a
   nice-to-have. A reviewer who has never opened this repo should be able to
   read any axis label, colorbar, or section blurb and know what it means
   without opening the source.
7. **No hardcoded absolute paths, ever** (`CLAUDE.md` §7 invariant 11) — every
   path relative to the repo root, to the invoking file's own location, or to
   a CLI argument, so the repo runs unmodified on any machine. Grep for drive
   letters / `/home/`/`/Users/` before finishing any session that touches a
   config or script.
8. **Keep `DEPENDENCIES.md` current** (`CLAUDE.md` §7 invariant 12) — any
   session that changes a library version, adds a new dependency, or
   discovers a new version-fragility gotcha updates that file in the same
   session, not as a follow-up.
9. **Read §15 before trusting a number, and before adding a new stage.** §15
   lists the pipeline's currently-known silent-failure paths — several of them
   affect numbers already recorded in this file's Findings blocks (each such
   case says so explicitly). If you are about to cite a recorded result,
   check whether a §15 item confounds it. If you are about to write a new
   stage that subsamples series, reuses artifacts, or degrades a capability,
   §15's A3/A4/A5 fixes define the conventions that stage should follow so it
   doesn't add a twentieth instance of the same class of bug.
10. **§0.5 is the queue; §16–§21 are pools; §22 is the parked list.** Sections
   §15–§21 hold ~60 sized items (A/E/F/G/H/J) in topic order, deliberately not
   sequenced. **§0.5 is the only sequencing in the file** — if an item is not
   in it, it is either parked in §22 (with a reason and an un-park trigger) or
   nobody has argued it belongs. Do not treat a high tier in a pool section as
   permission to skip a corrective item: a one-button tool that silently
   produces a wrong number is worse than a five-command tool that refuses,
   and §22.9 preserves the argument that equal grounds is a *prerequisite*
   for automation rather than a refinement of it.
---

## 0.5. Start here — the queue (rewritten 2026-08-18)

> **Who this is for.** A session picking this repo up cold and asking "what do
> I actually do next?" The rest of this file is organized by *topic*, which is
> right for a reference and wrong for a starting point. **This is the only
> sequencing in the file.** Four competing orderings used to exist (this
> section's tiers, §16's T1–T4, §17.3, §22's waves); they are collapsed here,
> and §22 now holds everything deliberately *not* in this queue, each with an
> explicit un-park trigger.
>
> It is a **pointer list, not a second source of truth** — every entry links
> to the section that owns the work, and an item is marked done *there*.
>
> **Rules that apply to every item**, so they aren't repeated per-entry:
> anything expected to run 5+ minutes goes to a background agent
> (`CLAUDE.md` §2.8); any config edit means a new `run.name` (§15 A3); any new
> checkpoint or library bump means `--check-alignment` read **in full,
> per-layer, by a human** (invariant 7 + `CLAUDE.md` §11.22); before comparing
> two runs of "the same" config, check whether shared pipeline code moved
> between their dates (`CLAUDE.md` §11.24).

**What "v1" is** — the file has a north star (§1) but never said what
shippable looks like, which is why the backlog grew faster than the queue. On
current evidence, v1 is three things and nothing else:

1. **One honest cross-model comparison report** — the 16-stage pipeline, run on
   real checkpoints, with every cross-model number either on an axis both
   models genuinely share or rendered next to the *measured* size of the
   asymmetry (§18's rule). Most of this exists; F1 and F9 are what's missing.
2. **The crosscoder result, negative** (§6.2.1 Stage 4). A pre-registered
   decision rule that fired against the flagship variant, with a validation
   ladder proving the metric itself was sound, is a publishable result and the
   most defensible thing in the repo.
3. **A legibility layer thin enough to finish** (§21 J1–J3) so a reader who
   didn't build this can tell a geometric claim from a causal one.

Everything else is post-v1. That is the whole justification for §22's size.

---

**1. ✅ DONE 2026-08-18 — the crosscoder is finished.** §6.2.1 — **V2**
(BatchTopK) built, scored at 3 seeds against the *unchanged* Stage 1
scorecard, and **does not flip clause 2** at any seed (V2-vs-V0
`gt_alignment_margin` beats_v0=False at all 3 seeds). Stage 4's writeup is
in. Stage 3 (pipeline wiring) stays **gated shut** — confirmed, not just
provisional; V3/V5/V6 remain parked (§22.1) for the same underlying reason.
§6.2.1's Findings blocks have the full numbers, including a new
seed-fragility finding on the L-B floor itself. This item is closed; no
further crosscoder work is queued.

**2. ✅ DONE 2026-08-18 — F1's remaining half, the depth axis.** §18 F1. All
eight legacy call sites now go through the new `depth_axis_for_run` (backed
by `ActivationStore.stack_meta`, persisted at extraction time so report-only
stages need no reloaded model); figures name their axis in `_note()`. The
acceptance test ran against real TimesFM-2.5-200M/Chronos-T5-Base checkpoints:
Chronos's `block` axis caps at 0.478 (not 1.0) as predicted, TimesFM's own
axis independently revealed a `capture_layer_stride`-driven cap at 0.947,
`align_on_axis` reports a real `overlap_fraction=0.505`, and re-running §16
E9's depth-curve null tests reproduced every prior verdict bit-identically —
confirming (not just arguing) that those particular verdicts were never
axis-dependent. One real, previously-unknown bug was found and fixed along
the way: `l3_perturbation.py`'s `patching.json` reused the wrong (longer)
layer list's depth axis whenever `l3.patching.layer_stride>1`, silently
corrupting several report figures' plotted depth coordinates (mocks never
exercised this because the smoke config uses `layer_stride:1`) — fixed,
regression-tested, and re-verified against the regenerated real report's own
Plotly trace data. Full writeup in §18 F1's Findings; new trap recorded as
`CLAUDE.md` §11.32.

**3. ✅ DONE 2026-08-18 — F9, the fairness card.** §18 F9. Built scaffold-first
as its own "Cost" line specified: a always-rendered first report section
(`_sec_fairness`, `requires: []`) reading F1/F2/F4/F6's already-measured
artifacts into four real rows, with F3/F5/F7/F8 rendered as explicit "not yet
measured" rather than omitted. Verified on the mock smoke pipeline (13
sections, up from 12) and re-verified against the real TimesFM/Chronos-T5-Base
pair via a report-only rerender — every number matches what F1/F2/F4/F6's own
Findings already recorded piecemeal. 5 new tests
(`tests/test_fairness_card.py`). Full writeup in §18 F9's Findings.

**4. ✅ DONE 2026-08-18 — the encode-store seam.** §6.2.1 Stage 3d — un-gated
from the crosscoder entirely. `sae.persist_features: true` writes baseline-SAE
features back into the store (`sae`/`sae_pooled/{model}/{layer}`); the one
consumer built on top, `sae/feature_geometry.py::run_sae_feature_cka`, computes
series-level linear CKA between two models' persisted feature spaces
(`l1/cka_sae.json`) — a genuinely different measurement than L1's own
activation-space CKA. Real-checkpoint result at the pinned TimesFM/Chronos-T5-
Base SAE targets: 0.263 (CI [0.228, 0.333]). L1's own main CKA pass and
clustering still read only activation space (`space="act"`) — not changed by
this item. 11 new tests, full suite 452/452. Full writeup in §6.2.1 Stage 3d's
Findings.

**5. ✅ DONE 2026-08-18 — both remaining §15 items.** A21 (an unknown config
key now raises with a closest-match suggestion instead of silently running
at the default) and **A20** (the impulse-probe amplitude constant is now a
per-run self-calibration — `calibrate_impulse_amplitude`, sweeping down from
the historical 0.25 and picking the largest amplitude that doesn't confound
a re-quantizing tokenizer, via a new optional `ModelAdapter.token_ids()`
hook). Live-reverified against real Chronos-T5-Small@512 (calibrates to
0.05, diagonal-hit fraction still 1.00 — the historical result, reproduced
by a different amplitude than expected) and Chronos-T5-Base@448 (calibrates
to 0.05, diagonal-hit fraction rises from the pre-fix 0.50-with-zero-margin
to 0.93). §15 A20's Findings has the full numbers and one honest surprise:
0.25 turns out to have had a small, previously unmeasured tokenizer-noise
margin even at the historically "safe" setting. §15 is now fully closed —
all 21 items `[x]`.

**6. The legibility layer — E6 and J1 done, J2/J3 next.** E6's `Finding`
dataclass refactor and J1's `plain`/`caveat` extension are both
✅ **DONE 2026-08-18** — see §21 J1's own Findings and E6's detail-up
Findings for the full writeup (byte-identical HTML verified pre-J1,
`findings.json` now carries `plain`+`caveat` on all 45 smoke-run findings,
466/466 tests). J2 ("what this tells you" per stage) and J3 (glossary +
worked example) are next; J4–J6 only if those land easily. This is v1's
third leg; it is also the item most likely to be skipped as cosmetic, which
it is not — see §0 rule 6.

**Also live but unsequenced** (pick up when it fits, no dependency):
§6.3.1 **Option C** (idiosyncratic-error fingerprinting — black-box, zero new
forward passes, and it passes the architecture control by construction, which
is exactly what killed §6.3's original method); §16 **E3** (empirical span
discovery — highest generality-per-hour item in the file); §18 **F5**, **F8**
and §19 **G1** (each ~1 session, each closes a stated fairness or breadth gap).
(§16 **E7b**, install docs, is ✅ done 2026-08-18 — both **E7a** and **E7b**
are now `[x]`.)

**Not in this queue:** see **§22** — parked features with reasons and un-park
triggers, and §22.8's short list of things rejected outright.
---

## 1. North star

Build the **transformer-lens for time-series foundation models**: a library
that lets *anyone* drop in a new TSFM checkpoint and, with a config file and a
benchmark corpus, get back a rigorous, honest, richly-explained answer to
"what does this model represent, where, how causally, and how does it compare
to others." Two things must both be true at the end of this roadmap:

- **Comparative depth**: TimesFM vs. Chronos (family) is understood well
  enough — behaviorally, geometrically, causally, and via learned sparse
  features — that the comparison could support a real paper.
- **Generality**: the machinery that produced that understanding works, with
  no architecture-specific hacking, on a *third* and *fourth* model family
  the authors of this repo didn't originally have in mind. Generality is
  proven by doing it (§7), not by asserting the abstractions are clean.

A study of two models that doesn't generalize is a case study. A general tool
that was never stress-tested on a hard comparative case is untested
abstraction. This roadmap is built to produce both, in an order where the
case study *informs* what the general tool needs to support.

---

## 2. Guiding principles (additions to `CLAUDE.md` §2's doctrine)

These apply specifically to the work in this roadmap; they don't replace
`CLAUDE.md` §2, they extend it.

### 2.1 Ground-truth-verifiable interpretability is this repo's unfair advantage
Most SAE / feature-interpretability work (vision, language) validates
features by *human or LLM judgment* of max-activating examples — inherently
subjective. This repo's benchmark generators (`build_pipeline`) attach exact
ground truth to every synthetic series: trend order, seasonal periods and
phases, changepoint locations, anomaly indices, AR coefficients, noise
envelopes. **Any claim of the form "this feature/layer/direction represents
concept X" should be checked against that ground truth wherever the
benchmark can express X**, before falling back to qualitative exemplar
inspection. This is a genuine methodological edge over NLP interpretability
and should be exploited deliberately, not left implicit. See §5.4.

### 2.2 Novelty must be tested against the null, not just proposed
Any new method (a layer-selection score, an SAE variant, a distillation
detector) needs a **negative control** before it's trusted: a metric someone
could compute for free that the new thing must beat. Effective dimensionality
alone predicting "good layer to interpret" is a strong, testable claim — the
null is "it doesn't correlate with anything and layer choice barely matters."
Write the null down before running the experiment, per `CLAUDE.md` §2.4.

### 2.3 Explainability of the explainer
A repo whose entire purpose is helping people understand model internals must
not itself be inscrutable. Every plot needs: what the axes mean, what the
color/size encodes, what a "good" or "surprising" value looks like, and what
evidence class it belongs to (`CLAUDE.md` §2.6 / §6.6). This is elevated to
its own workstream, §8, because it currently lags the analysis machinery.

### 2.4 Architecture-agnostic by construction, not by convention
When Phase 4 (§7) adds new models, the test of success is that **zero lines
change outside `models/<new_model>_adapter.py`** and its registration. If
adding a model requires touching `analysis/`, `extraction/`, or `report/`,
that's a bug in the abstraction from Phase 0–3, not a special case to shrug
off — fix the abstraction, don't special-case the model.

### 2.5 Prefer one well-validated method over three shallow ones
The brief's four ideas (analysis breadth, SAEs, ablation, packaging) could
each expand indefinitely. Resist scope creep by asking, before adding a new
analysis: does this sharpen a claim we're actually going to make, or is it
analysis for its own sake? `CLAUDE.md`'s layered method (§6.1) exists because
each layer answers a *specific* objection to the layer before it — hold new
work to that same bar.

---

## 3. Phase sequencing and why

The user's four ideas, reordered by dependency (not by how they were listed):

```
Phase 0  Explainability & report hygiene fixes         (cheap, unblocks trust in everything after)
   │
Phase 1  Broad multi-domain behavioral + geometric      (breadth: many data regimes, both models,
         analysis at scale                               produces the evidence base every later
   │                                                      phase reasons over)
   ├──────────────────────────────┐
   ▼                              ▼
Phase 2a  Layer-selection         Phase 3  Ablation & bias-characterization
          metric research                  studies (partially parallel with 2a;
   │                                        feature-level ablation waits on 2b)
   ▼
Phase 2b  Novel SAE variant(s) — the "meat"
   │
   ├── (independent side-quest, any time after L2 is stable — it already is)
   ▼
Phase 2c  IP / distillation-detection research thread
   │
Phase 4  Multi-model expansion (Chronos-2, Moirai, Sundial, ...)
   │        — deliberately late: prove the method on a hard 2-model case
   │          before generalizing it, or you generalize the wrong abstraction
   ▼
Phase 5  Productization — "transformer-lens for TSFMs"
         (folds in everything learned about what mattered)
```

**Why Phase 0 first:** it's cheap (documentation + report-string work, no
new science), and every subsequent phase produces reports whose credibility
depends on it. Fixing it late means re-touching every section again.

**Why Phase 1 before Phase 2:** idea (1) from the brief — "does effective
dimensionality predict interpretability" — needs *variance to correlate
against*. One run on one corpus gives one number per layer; you cannot
correlate a metric against an outcome with n=1. Running broadly across many
data regimes first is what makes Phase 2a a real experiment instead of a
plausible-sounding guess.

**Why Phase 3 partially overlaps Phase 1/2a but not Phase 2b:** the
behavioral/bias ablation studies (sweep one generator parameter, watch
sensitivity) reuse Phase 1 infrastructure directly and don't need SAEs to
exist. Feature-level ablation (does zeroing SAE feature *f* change the
forecast) obviously needs Phase 2b's SAEs trained first.

**Why Phase 2c (distillation detection) is a side branch, not a dependency
of anything:** it reuses L2/L1 as they exist today (`CLAUDE.md` §6.5 stitching
is already implemented). It can be picked up by a session with spare capacity
at any point after Phase 0, without blocking the main spine. It's placed
after 2b in the diagram only because it's lower research priority than the
SAE work, not because it depends on it.

**Why multi-model expansion is late:** `CLAUDE.md` §2.1 and §14 already argue
this from the TransformerLens comparison — proving generality on a *third*
hard case is worth more after the abstraction has been stress-tested by two
models with maximally different architectures (decoder-only vs.
encoder-decoder) and a full SAE/ablation pass. Adding models earlier risks
generalizing an abstraction shaped by convenience rather than necessity.

---

## 4. Phase 0 — Explainability & report hygiene

**Goal.** Every existing report section is honest about evidence class
(`CLAUDE.md` §2.6 already requires this at the framing level) but is *not yet*
legible to a reader who doesn't already know the codebase. Fix that first,
cheaply, before adding anything new. This operationalizes brief item (5).

**Concrete deliverables — ✅ DONE 2026-08-04**
- [x] Add a `report/glossary.py` (or equivalent) mapping term → one-paragraph
  plain-language explanation... — **done, but by a different, already-mostly-
  landed mechanism than the one this checklist originally specified.** Before
  writing a new `glossary.py`, actually read `report/report.py` (`CLAUDE.md`
  §2.4 discipline: verify before assuming a gap exists) and found it already
  carries a `_note()` helper attached to nearly every figure with exactly the
  purpose/reading/limitations structure this item wants — evidently added in
  a session whose work never made it back into this checklist. Checked every
  specific term flagged above against that mechanism rather than assuming the
  checklist's framing was still accurate:
  - **"restore" / restoration score**, **"row frac"**, **"CKA"**,
    **"participation ratio"**, **"crystallization depth"** — all already
    have a proper `_note()` with the exact formula and 0/1 meaning. No gap.
  - **ΔMASE / ΔR² / Δact sign conventions** — already stated locally per
    section (e.g. "Positive ΔMASE (red) means removing that head hurts the
    forecast"). No gap beyond what the new preamble below adds globally.
  - **"window" / "relative depth"** — individual notes gestured at these but
    never gave one canonical definition; now fixed by the new preamble
    (below), which is arguably the *better* fix for a term that recurs
    everywhere rather than scattering it across every plot's note.
  - **Two genuine gaps found and fixed**: the "Top periodicity heads" table
    had no `_note()` at all (added one deriving the exact formula from
    `analysis/attention.py:_periodicity` — score is excess mass above a
    mask-fraction chance baseline — since the report previously never
    explained "mask-fraction baseline" anywhere); and the head-ablation
    note lacked the worked numeric example this checklist explicitly asked
    for (added: "a cell at +0.15 means mean-ablating that head made this
    model's average MASE 0.15 worse in absolute terms").
  - **Decision, not an oversight:** did *not* build a separate
    `report/glossary.py` / hover-tooltip system. `_note()` already solves the
    same problem contextually (attached to the exact plot it explains,
    rather than requiring a reader to look a term up elsewhere), and per
    `CLAUDE.md` §2.2/§2.5's doctrine of one well-tested mechanism over
    parallel ones, building a second explainability system for the same gap
    would be redundant, not additive.
- [x] Add a `report.verbose` config flag (default **`true`**) — done
  (`config.py`'s `ReportConfig.verbose`/`verbose_series`). Extended the
  `exemplars.py`-style narrated case-study pattern to **L3 patching**
  specifically, per this item's explicit ask: for up to `verbose_series`
  (default 3) series per corruption, the report now shows that series' own
  context/true-continuation/clean/corrupted/patched forecasts next to its
  own full layer×window restoration grid. Implementation note worth keeping:
  the patched forecast shown is the single (layer, window) cell that
  restored the most *on average across the whole sampled batch* for that
  corruption (found for free from the already-computed aggregate grid, at
  zero extra forward passes), not each series' own individually-best cell —
  stated explicitly in the report's own note so it can't be misread as more
  than it is. Getting a per-series value at all required `_window_restoration`
  to stop collapsing to a batch mean before returning (§11.17 below has the
  numerical-safety argument for why this doesn't change the existing
  aggregate `restoration`/`restoration_windows` statistics).
- [x] Write a short **"How to read this report"** preamble — done, rendered
  unconditionally (not gated on `report.verbose`, unlike the L3 case studies)
  first, before Findings. States the full evidence-class ladder
  (geometric → linearly-translatable → causal-within-model → descriptive →
  illustrative → confirmatory) and gives the one canonical definition each of
  "window" and "relative depth" that the per-plot notes only gestured at.
- [x] Add a `--verbose` CLI passthrough on `run.py` — done (`--verbose` /
  `--no-verbose`, overriding `report.verbose` from the config when passed).

**Findings / decisions**
- **2026-08-04 — a real, previously-undiscovered platform bug found while
  smoke-testing this phase's report changes**: `report.py`'s
  `out.write_text(html)` (and several other `write_text`/`read_text` calls
  across `utils.py`/`config.py`/`data.py`) used the platform-default text
  encoding rather than an explicit one. On native Windows Python (this
  session's `tsfmPy` conda env, not WSL/Linux) that default is `cp1252`, and
  the report template already contained non-cp1252 characters (`▸`/`▾` in
  the existing `details.note` CSS, present before this session's changes) —
  so `run_report` crashed with `UnicodeEncodeError` on the very first smoke
  run attempted in this environment, before any of this phase's own content
  was added. This is the same "smoke green" baseline claim as `CLAUDE.md`
  §11.15 turning out to have never actually been exercised in the current
  environment — evidently every prior "10 sections, 18 findings" smoke
  confirmation ran on Linux/WSL (UTF-8 locale), never on native Windows.
  **Fix:** added explicit `encoding="utf-8"` to every `write_text`/
  `read_text` call in the package (`report.py`, `utils.py`'s
  `save_json`/`load_json`, `config.py`'s `load_config`/`dump_config`,
  `data.py`'s jsonl loader, plus the smoke test's own report read) rather
  than patching just the one call that happened to crash first — `json.dumps`/
  `yaml.safe_dump` both default to ASCII-safe output so those two were latent
  rather than actively broken, but leaving them on an implicit,
  platform-dependent encoding was the same fragility waiting to trigger on
  the next tool/library upgrade. Re-ran the full smoke suite (all three
  tests) after the fix: **10 sections, 18 findings, unchanged from the
  previously-recorded baseline** — confirming this was a pure I/O-layer fix
  with no effect on any computed statistic. Worth a `CLAUDE.md` §11 trap
  entry (§11.17) since this is exactly the "wired but unverified environment
  assumption" pattern §11.6/§11.15 already describe, just for text encoding
  instead of a library API.
- **2026-08-04 — verified both gating paths explicitly, not just the
  default-on path**: with `report.verbose: false`, the L3 stage emits zero
  `verbose_*` keys in `patching.npz` (confirmed via a direct run) and the
  report's case-study section renders nothing, while the "How to read this
  report" preamble still renders (it is intentionally not gated on
  `report.verbose` — it's foundational orientation, not a verbose extra).
  With the default `true`, the smoke config's L3 stage (2 corruptions ×
  2 mock models) produced verbose case studies for all 4 corruption×model
  combinations, 3 series each, with correctly-shaped
  `[n_layers, n_windows, n_verbose]` restoration grids and matching
  `series_ids`/`families` — added explicit shape assertions to
  `test_per_window_and_lens_artifacts` and new token checks to
  `test_end_to_end` so this can't silently regress.

---

## 5. Phase 1 — Comprehensive multi-domain analysis (brief item 1)

**Goal.** Actually run the full pipeline — not the smoke config — across a
wide variety of data regimes and both real model families, producing enough
independent (domain × family × model) observations that later phases have
something to correlate against instead of anecdote. This is also simply the
deliverable the brief asked for directly: "comprehensively understand the
qualities of each [model]."

### 5.1 Corpus breadth
**Concrete deliverables**
- [x] Generate (or confirm already generated, then re-seal if stale) sealed
  corpora spanning: all 8 default archetypes, and **deliberately also** the
  4 opt-in archetypes (`random_walk_drift`, `intermittent_bursts`,
  `amplitude_modulated`, `nonsinusoidal_seasonal` — `CLAUDE.md` §4.2) via
  `configs/full_multidomain.yaml`, since these are the sharpest
  Chronos-vs-TimesFM stressors (quantized vs. continuous, unit-root drift,
  non-sinusoidal shape) and were opt-in specifically to avoid silently
  reshuffling old seeds, not because they're less interesting. — the
  synthetic backbone task in `full_multidomain.yaml` already names all 12
  explicitly (unchanged this session); confirmed still correct while
  reviewing the file for the real-derived expansion below.
- [x] Include at least one `real_derived` tier task (`mixture`,
  `block_bootstrap`, or `sequential_par`) once the "wired but unverified"
  status (`CLAUDE.md` §12) of `tsbootstrap`/SDV/Monash is confirmed working
  end to end — **done 2026-08-03, see §5.2 below: all three real-derived
  generators now run against live data**, not just `mixture` as before.
  `configs/full_multidomain.yaml` now has five real-derived tasks across
  those three generators and **two** independent real corpora (Monash TSF's
  full 32-domain catalog, and ETT — Electricity Transformer Temperature),
  widening the real-derived tier from 30% (one generator, one corpus) to
  ~26%+4%+0.4% (mixture+block_bootstrap+sequential_par, Monash+ETT). A new
  `configs/real_data_smoke.yaml` gives a fast (~1 min) end-to-end check of
  the whole real-data path without full_multidomain.yaml's cost.
- [x] Run `benchmark_validation` on every generated corpus and *read the
  redundancy fraction alongside the feature metrics* (`CLAUDE.md` §5) before
  trusting any corpus as "diverse enough" to build conclusions on. — done
  this session on the small smoke build (32+34 sequences; numbers in the
  Findings block below) and now also on a much larger, tractably-scaled real
  build (`configs/full_multidomain_run1.yaml`, 4219 public_dev sequences) —
  see the new 2026-08-03 (follow-up session) Findings entries below for the
  numbers and, notably, a new diversity-by-task finding. The literal
  full-scale `full_multidomain.yaml` (49,550/split) was not run this
  session — see that same Findings entry for why and what it would take.

### 5.2 Unblock the "wired but unverified" real-data path — ✅ DONE 2026-08-03
`CLAUDE.md` §12 flagged `block_bootstrap` (tsbootstrap), `sequential_par`
(SDV), and the Monash adapter as written against documented APIs but never
run against live network/versions. **All three now run end to end against
live data** (Monash TSF + ETT via Hugging Face `datasets`, real block
bootstrapping via tsbootstrap, real SDV PARSynthesizer fitting/sampling) —
verified via direct generator calls, the existing test suite, two new
regression tests (`tests/test_real_derived_generators.py`), and a full
`run_full.py` → `run_validation.py` build using `configs/real_data_smoke.yaml`
(see Findings below for the exact numbers). This was **not** a quick "bump a
version number" fix — every one of the three generators, plus the Monash
loader itself, turned out to be actually broken against current library
versions, each for a different reason:
- [x] Pin and verify `tsbootstrap`, `sdv`, `datasets` (Monash) versions
  actually work end to end; record working versions in
  `tsfm_benchmark/build_pipeline`'s `pyproject.toml` extras. — done; see
  Findings below for the exact pins and *why* each one is load-bearing (this
  isn't "use whatever's newest" — several of these versions are pinned
  specifically to avoid a breaking change in a later release).
- [x] If any adapter turns out broken against current library versions, fix
  or explicitly mark `[!]` blocked here with the failure mode — don't let it
  rot silently. — all three fixed in place; see Findings.

### 5.3 Model + checkpoint breadth (still within TimesFM/Chronos family)
- [x] Run more than the single default checkpoint pair
  (`google/timesfm-2.0-500m-pytorch`, `amazon/chronos-t5-base`) — at minimum
  add a size variant on each side (e.g. TimesFM 1.0 vs 2.0, chronos-t5-small
  vs -base vs -large, chronos-bolt at a comparable size) so findings can be
  checked for "is this a TimesFM-vs-Chronos finding or a
  bigger-model-vs-smaller-model finding." This directly de-confounds several
  of the tentative findings in `CLAUDE.md` §10. **Completed 2026-08-05**:
  `configs/medium_run_chronos_base.yaml` (added 2026-08-04) run to completion
  on a new, more powerful machine (8× RTX A5000, see the environment note
  below) — `runs/medium_run_chronos_base/`, 10 sections, 21 findings,
  ~32.5 min wall-clock (slower than the chronos-small run's ~10.5 min despite
  the better GPU — chronos-t5-base's 12 encoder blocks vs. small's 6 roughly
  doubles L3 per-window-patching and attention-pattern cost, which dominate
  wall-clock; see the environment note for why this took a fresh
  `--check-alignment` pass first). See §5's Findings block below for the
  full base-vs-small comparison and the headline result: the tested
  behavioral finding is **size-robust**, but representational-similarity
  metrics are **not**, and don't even move consistently with each other.
- [x] Confirm `--check-alignment` passes for every new checkpoint before
  trusting cross-model numbers from it (`CLAUDE.md` §6.3, invariant 7) — done
  for `amazon/chronos-t5-base`: perfect 1.00 diagonal-hit fraction at every
  one of its 12 encoder blocks (2026-08-04), same clean result as
  `chronos-t5-small` got in §5.4. Not yet done for any other candidate
  checkpoint (TimesFM 1.0/2.0, chronos-t5-large, chronos-bolt) — do this
  before building a config around any of them, per invariant 7.
  > ⚠️ **AUDIT (2026-08-06) — invariant 7 is enforced by human memory, not by
  > the pipeline, see §15 A2.** `alignment.sanity_check` is on in all three
  > real configs, but `extraction/extract.py:50-51` **discards** the check's
  > return value: there is no threshold, no artifact, no report line, and only
  > a stride-4 subset of layers is probed. A future checkpoint or library bump
  > that breaks token spans produces one INFO line inside a long run and a
  > full report of confident cross-model numbers — the precise scenario
  > `CLAUDE.md` §6.3 says must block trusting anything, and the one that
  > already cost a session once (§11.16). "Remember to run
  > `--check-alignment`" is not an invariant; a gate is.

### 5.4 Resolve the two `[VERIFY]`/known-risk items from `CLAUDE.md` §9/§10 first — ✅ DONE 2026-08-03
These were explicitly called out as untested against real weights. Both are
now closed out — see the Findings block below for the full story, which
turned out to include two genuine, previously-undiscovered bugs (matching
this repo's now-familiar pattern of "wired but unverified" almost always
meaning "actually broken until someone runs it for real"):
- [x] Run `--check-alignment` and a small real run to validate Chronos
  `output_attentions` and the TimesFM module scan against live checkpoints
  (`CLAUDE.md` §9). — done; both confirmed working, but only after fixing a
  real bug in the alignment-check tool itself (see Findings).
- [x] Re-run L3 per-window patching and confirm whether TimesFM's previously
  **flat** restoration curve (`CLAUDE.md` §10) resolves now that per-window
  patching exists — this was diagnosed as a method-coarseness artifact, not a
  finding about TimesFM, and needs to be confirmed fixed, not just assumed
  fixed because the mechanism was added. — confirmed: whole-layer patching
  is still flat (that part of the old finding holds), but per-window patching
  recovers real, strong, depth-dependent causal structure the whole-layer
  average was washing out. See Findings.

### 5.5 Consolidated cross-run reporting — ✅ DONE 2026-08-05
A single-run HTML report (`CLAUDE.md` §6.5) doesn't by itself answer "what
are the qualities of each model in general" — that needs aggregation *across*
runs.
- [x] Build a small cross-run aggregator (new, e.g.
  `tsfm_lens/report/meta_report.py`) that takes N run directories and
  produces one summary: per-archetype MASE gap, per-archetype crystallization
  depth, per-archetype CKA peak, stability of each finding across
  data regimes (does "TimesFM front-loads then compresses" hold on
  `intermittent_bursts` too, or only on the smooth archetypes it was observed
  on?). This is new work, not a rename of the existing per-run report. —
  **done**: `tsfm_lens/report/meta_report.py` (+ CLI `run_meta_report.py`)
  reads N run directories' existing artifacts (nothing re-run), producing a
  runs table (L0 overall MASE, L1 peak CKA, L2 best gain, clustering AMI,
  crystallization depth) and a per-**family** stability table (this repo's
  L0 summary only breaks down by top-level `family`, not by finer
  `random_parametric` archetype — a genuine per-archetype breakdown isn't
  in the current L0 artifact and would need a new field there first, noted
  as follow-up below, not silently claimed as done). Tested against the
  four existing run directories (`medium_run`, `medium_run_chronos_base`,
  `real_run`, `smoke`): correctly surfaced that `random_parametric` favors
  TimesFM **stably across all three real-checkpoint runs** (two Chronos
  sizes plus a third, independently-built corpus) — a genuine answer to
  exactly the "does this hold across regimes" question this deliverable
  exists to give, on the first real use. Two regression tests
  (`tests/test_meta_report.py`) cover the missing-stage graceful-degrade
  path and a real edge case hit live on `runs/real_run` (a model with
  `crystallization_depth: null` — never crystallized within tolerance —
  which crashed the first template draft on a bare `%.2f` format).
  **Follow-up, not done:** a true per-*archetype* (not per-family) stability
  view would need `l0/summary.json` to record `random_parametric`'s sampled
  archetype per series, which it currently doesn't — this is a small
  `analysis/l0_behavioral.py` schema addition, not an aggregator change, and
  is left for whoever next wants that finer granularity rather than
  bundled into this session unprompted (§2.5 scope discipline).
  > ⚠️ **AUDIT (2026-08-06):** now scheduled with a concrete plan as §15 A9,
  > because it turns out to block more than this aggregator view — the same
  > missing `archetype` column is why §6.2's ground-truth archetype dummies
  > have to be reconstructed from provenance at SAE-scoring time, which is
  > where §15 A10's "real-derived series silently count as archetype
  > negatives" problem comes from. One schema addition closes both.
- [x] Explicitly revisit every "Defensible" / "Underdelivered" claim in
  `CLAUDE.md` §10 against the broader evidence base and update that section
  (or better, retire it from `CLAUDE.md` and move the *current*, broader-based
  version of these findings into this file's Findings block, since `CLAUDE.md`
  should describe stable architecture, not a moving research result). —
  **took the "better" option**: `CLAUDE.md` §10 now redirects here;
  its original content is preserved verbatim as a dated entry in this
  section's Findings block below (2026-08-05, "carried forward from
  `CLAUDE.md` §10"), which also states plainly which of its claims are
  now superseded by later evidence and which are still exactly as
  provisional as when first measured.

**Findings / decisions**

- **2026-08-03 — the real-data path was broken in four independent ways, not
  one.** Actually running it (rather than trusting "wired but unverified" at
  face value) surfaced a chain of real breakages, each a different upstream
  dependency's own change invalidating a hardcoded assumption — the same
  pattern as the TimesFM break in `CLAUDE.md` §11.8, now four more times
  over:
  1. **`datasets>=3` removed script-based dataset loading entirely.**
     `Monash-University/monash_tsf` is script-backed
     (`monash_tsf.py`), and a modern `datasets` install (5.0.1 at time of
     testing) raises `RuntimeError: Dataset scripts are no longer
     supported` — not a network or `trust_remote_code` issue, a hard
     removal. **Fix:** pin `datasets>=2.18,<3` (2.21.0 verified). This is
     *the* reason Monash loading was never actually verified before now —
     anyone who tried it with a contemporary `datasets` install would have
     hit this immediately. `pyproject.toml`'s real-data extra now states the
     upper bound and why.
  2. **`tsbootstrap`'s entire public API was rewritten.** The class-based
     `MovingBlockBootstrap(n_bootstraps=..., block_length=...,
     rng=...).bootstrap(arr)` `block_bootstrap` was written against no
     longer exists in 0.7.1 at all — replaced by a functional
     `bootstrap(X, method=MovingBlock(block_length=...), n_bootstraps=...,
     random_state=...)` call returning a `BootstrapResult` whose replicate
     array comes back via `.values()` (a *method*, easy to misuse as a
     property) shaped `[n_bootstraps, n_obs]`. Fixed in
     `generators.py:block_bootstrap`.
  3. **`sequential_par` was a stub (`raise NotImplementedError`) — now
     implemented** against `sdv` 1.14.0's `PARSynthesizer`. Two non-obvious
     things worth not rediscovering: (a) giving PAR an explicit
     `sequence_index` column crashes `auto_assign_transformers` with
     `AttributeError: 'NoneType' object has no attribute
     'enforce_min_max_values'` in this version — omit it entirely and rely on
     row order instead (documented in the function's docstring); (b)
     `PARSynthesizer` exposes **no `random_state`/seed parameter anywhere**,
     so this generator's reproducibility is best-effort (global numpy/torch
     seeding), not the bit-exact guarantee the rest of the module provides —
     stated explicitly rather than silently assumed. Also changed
     `sequential_par`'s contract from `training: list[np.ndarray]` to
     `training: list[tuple[SourceRef, np.ndarray]]` (matching `mixture`) so
     it can actually record `source_refs` in provenance — it previously
     couldn't, since `builder.py` was stripping refs before passing them in.
  4. **A real, previously-undiscovered bug in `run_full.py`'s `load_specs`**:
     it unconditionally set `generator_params["sources"] = sources` for
     *every* generator in `_NEEDS_SOURCES`, but only `mixture` accepts a
     `sources` kwarg — `block_bootstrap` takes `source` (singular) and
     `sequential_par` takes `training`. This was invisible before because
     the only generator ever exercised through the CLI's eager-resolution
     path (no `source_sample_size` set) was `mixture`
     (`tests/test_real_source_injection.py` only covers `mixture_task`).
     Fixed to branch per generator, mirroring `builder.py._make_one`'s
     existing (correct) branching.

- **2026-08-03 — a real-world data edge case: `block_length` must be
  clamped to series length.** Monash's yearly-granularity domains
  (`tourism_yearly`, etc.) have series as short as ~14 points; a
  config-level `block_length: 24` (fine for the hourly/weekly/monthly
  domains bootstrapped alongside them) made `tsbootstrap` raise
  `MethodConfigError: block_length exceeds series length` the moment a short
  series was drawn. This is exactly the kind of thing that only surfaces by
  actually bootstrapping across real domains of wildly different
  granularity, not by testing on one hand-picked domain. **Fix:**
  `block_bootstrap` now clamps to `max(2, min(block_length, len(arr) // 2))`
  and records the effective value used in provenance
  (`effective_block_length`) for auditability, rather than crashing the
  whole task.

- **2026-08-03 — a genuinely new, unresolved risk found while doing this:
  the golden-hash regression test (`CLAUDE.md` invariant 1,
  `tests/test_generator_extensions.py::test_golden_hashes_pre_extension_outputs_unchanged`)
  fails against numpy 2.1.0** (the exact version `clean_reinstall.sh` pins),
  in a from-scratch environment, **without any change to `parametric` /
  `random_parametric`** (confirmed via `git diff` scope before investigating
  — this session only touched `block_bootstrap`/`sequential_par`). The
  mismatch is deterministic (reproduces identically across repeated runs in
  this environment), so it isn't nondeterminism — something about this
  numpy version's actual output differs from whatever numpy version the
  golden hashes were originally captured on. **Not fully root-caused this
  session** (ran out of a reasonable time budget to bisect against an old
  numpy install); the leading hypothesis, not yet confirmed: `parametric`
  uses `rng.choice(..., replace=False)` for changepoint and anomaly indices,
  and unlike `rng.normal()`/`rng.standard_normal()`, numpy does **not**
  guarantee `Generator.choice`'s exact algorithm is stable across numpy
  versions — only that it's stable *given a numpy version*. If confirmed,
  this means `CLAUDE.md` invariant 1 ("sealed corpora regenerate
  bit-exactly from their seed") is actually **numpy-version-fragile**, not
  purely a function of this repo's own code, which nothing before this
  session's environment work had reason to suspect (the invariant was never
  exercised in an environment where numpy version was actually being
  scrutinized). **Follow-up needed**: (a) confirm the `choice`-algorithm
  hypothesis by testing against an older numpy in an isolated env, (b) if
  confirmed, either pin numpy as part of the reproducibility contract
  (document it as a load-bearing version, the way `datasets`/`tsbootstrap`
  now are) or replace the `rng.choice(..., replace=False)` call sites with a
  construct built only from guaranteed-stable primitives.

- **2026-08-03 — a real, unrelated native-crash bug found and fixed while
  setting up the environment for this work**: conda-forge's `numpy==2.1.0`
  Windows build (compiled ~Sept 2024) crashes with an unhandled SEH
  exception (`0xc06d007f`) on **any** matrix multiplication when a
  significantly newer MKL runtime (`mkl==2026.1.0`, pulled in transitively
  by `sdv`'s `pytorch` dependency) is present in the same environment — an
  ABI mismatch between the numpy build and a much newer MKL, not a bug in
  this repo's code. Reproduced with a bare `a @ b` numpy call with no
  imports from this repo at all. **Fix:** pin `mkl==2024.2.2` (matching
  numpy 2.1.0's build era) alongside it. This would have silently broken
  `benchmark_validation`'s matching/UMAP/diversity code (all matmul-heavy)
  for anyone setting up a fresh conda-forge environment with these
  packages — worth checking for on any future environment rebuild.

- **2026-08-03 — real-data smoke build results** (`configs/real_data_smoke.yaml`,
  `run_full.py` → `run_validation.py`, live Monash + ETT + 20 Monash
  reference series): 32 public_dev + 34 private_test sequences sealed (4
  rejected by the leakage gate, 0 near-duplicates). Composition: 68.8%
  realism_stress / 31.2% synthetic at this tiny smoke scale (deliberately
  skewed toward real-derived to exercise all three generators). Real domains
  actually drawn: `tourism_monthly`, `tourism_yearly`, `solar_weekly`,
  `covid_deaths`, `weather`, `us_births`, `vehicle_trips` (Monash) plus
  `h1`/`h2`/`m1`/`m2` (ETT). `bootstrap_catalog`'s skip-and-continue behavior
  for broken Monash domains was confirmed live: `kdd_cup_2018`, `rideshare`,
  `traffic_hourly`, `oikolab_weather`, `pedestrian_counts` and others
  reliably fail with a pandas frequency-alias `KeyError` (`'h'`/`'min'`/`'ME'`/`'s'`
  — Monash's own loading script using aliases removed in newer pandas), and
  `london_smart_meters`/`kaggle_web_traffic`/`wind_farms_minutely` reliably
  time out (large non-streamed downloads) — both exactly as
  `CLAUDE.md` §12 already described, now confirmed rather than assumed.
  Diversity numbers at this n=32 scale are not meaningful on their own
  (effective dimensionality 2.7/22, near-collision fraction 0.06) — too small
  a sample to draw conclusions from; a real diversity read needs a
  full-scale `full_multidomain.yaml` build, which is the natural next step
  and was deliberately not run this session (a full sequential_par task at
  the configured `epochs: 64` against real several-hundred-point series took
  long enough on CPU in an initial attempt — killed after ~50 minutes with no
  visible progress on a single fit — that it needs either a smaller epoch
  count, a GPU, or a fit-once/sample-many cache before running at full
  scale; see the `sequential_par` cost note below).

- **2026-08-03 — `sequential_par`'s per-call cost is real and scales badly
  with real series length, worse than a small prototype suggested.** An
  isolated test (6-series cohort, length ~60, `epochs=8`) fit in ~3.5s; the
  same generator against `full_multidomain.yaml`'s real config
  (`epochs: 64`, cohorts drawn from several-hundred-point real Monash
  series) did not visibly complete a single additional fit in over 50
  minutes of CPU-only wall-clock time on this machine. **`cuda` now
  auto-detects `torch.cuda.is_available()` by default** (changed from a
  hardcoded `False` after a mid-session discussion — a hardcoded default
  would silently stay wrong in whichever environment doesn't match the one
  it was tuned against) — GPU should meaningfully help here, but no
  environment used this session has a working CUDA-enabled torch build (see
  the next finding). Until either a GPU env is provisioned or a fit-once/
  sample-many cache is added, keep `sequential_par` tasks' `epochs` low
  (~5-15) and cohort lengths short for anything beyond a smoke test.

  **Correction, same-day follow-up session:** the diagnosis above was
  wrong about *which* changed variable mattered — it conflated an epoch
  increase (8→64) with a cohort-length increase (real Monash series, not
  the ~60-point synthetic test) happening at the same time, and blamed
  epochs. A controlled calibration that held a real Monash cohort fixed
  (lengths up to 40,720 points) and varied only epochs (8 vs 16) measured
  **~1000s for both** — epoch count was nearly irrelevant. The actual
  driver: `sequential_par` fed every raw timestep of every cohort series,
  unbounded, into `PARSynthesizer.fit()`; SDV's DeepEcho RNN backend trains
  per-timestep, so a 40,720-point series costs roughly 1000x what a
  512-point one does, independent of epochs. **Fix:** `generators.py`'s
  `sequential_par` now takes `max_train_length` (default 512, matching this
  repo's typical `context_len` elsewhere) and truncates each cohort series
  to its most recent `max_train_length` points before fitting; the
  default sampled output `length` (when not explicit) is now computed from
  the truncated lengths too, so it isn't inflated by a raw 40k-point
  series either. Re-running the same calibration cohort with the fix
  (`max_train_length=512`, epochs 32 vs 64) measured **~5-6.5s per fit** —
  a ~150-200x speedup, epochs now correctly barely matter since length is
  bounded. `n_series_truncated` and `max_train_length` are recorded in
  provenance for auditability, same pattern as `block_bootstrap`'s
  `effective_block_length`. `configs/full_multidomain.yaml`'s
  `real_sequential_par` task (`count: 50`, `epochs: 64`) is now genuinely
  tractable at this scale (~50 fits × ~5-6s ≈ 5 min for both splits
  combined) — the earlier guidance to keep epochs at 5-15 no longer
  applies and is superseded by this finding, not simultaneously true with
  it. Lesson for next time (`CLAUDE.md` §2.4): when two variables change
  between "worked" and "hung," isolate them before writing down which one
  is at fault — the first guess here was plausible-sounding and wrong.

- **2026-08-03 — CUDA pytorch is not installable in the same conda-forge env
  as the numpy/scipy/validation stack without re-risking the MKL crash
  above.** Every conda-forge CUDA pytorch build (`cuda128`/`cuda130`)
  requires a recent MKL (2025+), which is the exact MKL generation that
  crashes this numpy 2.1.0 build. `sequential_par`'s GPU path is therefore
  currently unusable in the `tsfmPy` env (its pytorch resolved to a
  CPU-only `cpu_mkl` build, confirmed via `torch.cuda.is_available() ==
  False`). **Not resolved this session** — the two options discussed with
  the user were an isolated second conda-forge env with CUDA pytorch (talk
  to `sequential_par` via subprocess) or upgrading `mkl` in-place and
  re-verifying the matmul crash doesn't return; **decision deferred**, flag
  as `[!]` blocked-on-decision if a future session wants `sequential_par` to
  actually run on GPU.

  **✅ Resolved 2026-08-04 (superseded, not consistent with this entry —
  see `DEPENDENCIES.md`'s Findings-equivalent for the direct test): the
  §5.4 session's in-place torch upgrade to `2.9.1+cu130` (done for
  `tsfm_lens`, not explicitly re-checked against this concern at the time)
  already resolved this. Verified today: `numpy` 2.1.0 matmul, CUDA torch,
  and `sdv`/`ctgan`/`deepecho` imports all coexist and run with no crash in
  the current `tsfmPy` env (`mkl==2024.2.2` alongside `torch==2.9.1+cu130`),
  and `PARSynthesizer` defaults to `cuda=True`. `sequential_par`'s GPU path
  is not blocked by anything found today — this is no longer an open
  decision.** This paragraph is left in place rather than deleted per this
  file's own doctrine (§0.2) of never erasing a finding, even a superseded
  one — the correction lives right below it instead.

- **2026-08-03 — a second, independent real corpus works via the existing
  generic HF loader with zero new code**: `ETDataset/ett` (Electricity
  Transformer Temperature) loads cleanly through `sources.load_sources`
  exactly like Monash does, confirming `sources.py`'s "works against any HF
  dataset repo id" claim (`CLAUDE.md` §4.3) was actually true, not just
  documented. Added to `configs/full_multidomain.yaml` and
  `configs/real_data_smoke.yaml` specifically so the real-derived tier isn't
  Monash-only.

- **2026-08-03 (follow-up session) — the real fix for `sequential_par`'s
  cost was bounding training sequence length, not lowering epochs; the
  earlier finding above was diagnosed wrong.** A controlled calibration
  (same real Monash cohort — lengths up to 40,720 points — held fixed,
  varying only `epochs` 8 vs 16) measured **~1000s for both**, disproving
  the earlier guess that epoch count was the driver. The actual cause:
  `sequential_par` fed every raw timestep of every cohort series, completely
  unbounded, into `PARSynthesizer.fit()`; SDV's DeepEcho RNN backend trains
  per-timestep, so a 40,720-point series costs roughly 1000x what a
  512-point one does, independent of epochs. **Fix:** `generators.py`'s
  `sequential_par` gained a `max_train_length` parameter (default 512,
  matching this repo's typical `context_len` elsewhere) that truncates each
  cohort series to its most recent `max_train_length` points before
  fitting; the auto-computed sampled output `length` (when not given
  explicitly) is now derived from the truncated lengths too, so it isn't
  inflated by a raw 40k-point series either. Re-running the identical
  calibration cohort with the fix (`max_train_length=512`, epochs 32 vs 64)
  measured **~5-6.5s per fit** — a ~150-200x speedup — with epochs now
  correctly making almost no difference, confirming length was the actual
  lever. `n_series_truncated`/`max_train_length` are recorded in provenance
  for auditability (same pattern as `block_bootstrap`'s
  `effective_block_length`). A new regression test
  (`test_sequential_par_truncates_long_series_before_fitting`) covers this.
  **Lesson** (`CLAUDE.md` §2.4): the original finding changed two variables
  at once (epochs *and* moving from a ~60-point synthetic test cohort to
  several-hundred-to-tens-of-thousands-point real cohorts) and blamed the
  wrong one; a controlled single-variable calibration was needed to find
  the truth. `configs/full_multidomain.yaml`'s `real_sequential_par` task
  now names `max_train_length: 512` explicitly and its header comment is
  corrected to match.

- **2026-08-03 (follow-up session) — first tractably-scaled real
  (non-smoke) build completed, with real diversity numbers.** Before
  committing to a build size, ran a throughput calibration
  (`full_multidomain.yaml` with `--max-count 300`, both splits): 4313
  attempted samples in 16m33s (~4.3 samples/sec end-to-end, including the
  two-stage DTW leakage audit and cross-split near-duplicate detection).
  Extrapolating that rate to the literal `full_multidomain.yaml` config
  (~49,550 attempted samples per split, ~99,100 across both) would take
  roughly 6+ hours — impractical for this session, so instead of running it
  outright, built `configs/full_multidomain_run1.yaml`: a ratio-preserving
  ~10x scale-down (~5,000 attempted per split) with one deliberate
  exception — `real_sequential_par`'s count was kept at 40 (not scaled to
  ~5) since its cost is now bounded (previous finding above), so keeping it
  higher costs only a few extra minutes for a meaningfully larger
  real-derived population from that generator. Result: **4219 public_dev +
  4279 private_test sequences** (rejected by the leakage gate: 1482;
  near-dupes: 139 — **~1.4%**, not the ~28% seen in the `--max-count 300`
  calibration run, confirming that high figure was a `--max-count` sizing
  artifact — small per-task pools sampled disproportionately — and not a
  real property of the ratio-preserving config, exactly as
  `full_multidomain.yaml`'s own header already warned). Build took 15m48s
  total. The literal full-scale `full_multidomain.yaml` build remains
  future work, best run unattended (e.g. overnight) given the ~6-hour
  estimate — not attempted this session, and this file's smaller
  `_run1` config is not a replacement for it, just a first real data point.

- **2026-08-03 (follow-up session) — `benchmark_validation` run on the
  4219-sequence public_dev corpus from the build above; a real,
  large-enough-to-trust diversity read (unlike the earlier n=32 smoke
  numbers, which `CLAUDE.md` §5 already flagged as too small to draw
  conclusions from).** Corpus-wide: redundancy fraction **0.00%** (17 of
  8,897,871 pairs at or above the 0.97 shape-similarity threshold),
  effective dimensionality **4.84 of 22** catch22 features, near-collision
  fraction **0.05**, 1624/92818 scaled feature values winsorized (0
  sequences needed imputation). These corpus-wide numbers land close to the
  reference demo figures already recorded in `CLAUDE.md` §5 (redundancy
  4.8%, eff-dim 4.6/24) — a useful corroboration that those numbers
  generalize past the small demo corpus they were first measured on, not
  just a coincidence of that specific run.
  - **A new, genuine finding from the per-task breakdown**
    (`diversity_metrics_by_group`): the synthetic backbone's effective
    dimensionality (**6.065**, n=3154) is markedly *higher* than every
    real-derived task's — `real_ett_mixture` 2.868 (n=100),
    `real_block_bootstrap` 2.016 (n=48), `real_bootstrapped_multiplicative`
    1.531 (n=473), `real_bootstrapped_weighted_sum` 1.517 (n=405),
    `real_sequential_par` **1.265** (n=39, the lowest of any task). Read
    correctly, this is not "the real data is redundant junk" — it's that
    `random_parametric` is *deliberately* engineered to maximize structural
    variety (12 archetypes, randomized corruptions), while every
    real-derived task draws its raw material from a comparatively narrow
    slice of real domains (each task's `source_config` pools from Monash's
    ~13 discovered domains or ETT's 4 configs) and its *combination*
    generators (`mixture`'s weighted sums, `block_bootstrap`'s reshuffles,
    `sequential_par`'s learned cohort model) tend to preserve rather than
    diversify the underlying domains' catch22 shape signature. This is a
    genuine, actionable characterization of what "real-world data" is
    actually contributing to this benchmark's feature-space diversity —
    breadth of *domain* coverage, not breadth of *catch22-feature-space*
    coverage — worth stating honestly rather than assuming "more real data"
    automatically means "more diverse" in the sense this validation package
    measures (`CLAUDE.md` §5's own warning: read redundancy/diversity
    metrics for what they actually measure, not what sounds intuitive).
  - `CO_trev_1_num` remains the top-variance feature for 5 of 6 tasks even
    post-winsorization (`real_ett_mixture` is the exception, dominated by
    `SB_BinaryStats_diff_longstretch0` instead) — expected and *not* a
    regression of the winsorization fix (`CLAUDE.md` §5): winsorization
    pins genuinely extreme series at the clip boundary so they can't swamp
    every other metric, it doesn't and shouldn't make that feature stop
    being the most variable one in the corpus.
  - Within-group vs. across-group mean similarity are close (`realism_stress`
    0.5433, `synthetic` 0.5544, across-group 0.5265) — the two tiers aren't
    tightly self-clustered relative to each other, a mild positive signal
    that the real-derived tier isn't just a redundant island bolted onto the
    synthetic backbone.
  - **Follow-up worth doing, not done this session:** re-run this same
    per-task diversity breakdown once/if a true full-scale build exists, to
    check whether the effective-dimensionality gap between synthetic and
    real-derived tasks narrows, holds, or widens at 10x this scale — n=39-473
    per real-derived task here is enough to trust the *direction* of this
    finding but not to treat the exact numbers as final.

- **2026-08-03 (§5.4 session) — first-ever real-checkpoint run of
  `tsfm_lens`, and it surfaced two genuine bugs that had never been
  exercised before, plus resolved both open live-weights questions from
  `CLAUDE.md` §9/§10.** This machine has a previously-undocumented NVIDIA
  RTX 5070 (12GB VRAM, driver reporting CUDA 13.0) — worth recording since
  everything in `CLAUDE.md` up to now assumed CPU-only for this repo's
  environment work. Per explicit user instruction, this reused the existing
  `tsfmPy` conda env (built for `build_pipeline`/`benchmark_validation` in
  the prior session) rather than creating a second environment, upgrading
  its torch in place from the CPU-only 2.7.1 to `torch==2.9.1+cu130`
  (matching `clean_reinstall.sh`'s pin) alongside `transformers==4.57.6`,
  `timesfm==2.0.2`, `chronos-forecasting==2.3.1`, `zarr==2.18.7`. Verified
  before doing anything else that this doesn't resurrect the numpy/MKL ABI
  crash from `CLAUDE.md` §11.13 (bare CPU **and** GPU `a @ b` matmuls both
  succeed) and that `ctgan`/`deepecho` (sdv's `sequential_par` dependencies,
  §11.11) have no upper torch-version bound that this would violate. Full
  `tsfm_benchmark` test suite re-run after all of this: still **40 passed /
  1 failed** (the pre-existing, already-documented golden-hash numpy
  fragility, §5's Finding above) — no new regressions from sharing the env.
  Built a real (not smoke) two-model benchmark via
  `tsfm_benchmark/configs/medium_run.yaml` (written in a prior session but
  never actually built or run through `tsfm_lens` before now) —
  `benchmark_medium/`, 288 public_dev / 286 private_test sequences across 3
  families: `random_parametric`, `parametric`, `mixture`.

  1. **Bug found #1 — `tsfm_lens`'s zarr storage layer had never actually
     been run against its own pinned zarr version.** The very first real
     extraction attempt (even on the mock-model smoke config, once run in a
     real environment rather than whatever environment the "10 sections, 18
     findings" smoke-green baseline was last measured in) crashed
     immediately: `extraction/store.py` called `Group.create_array(...)`,
     which **does not exist** in any zarr 2.x release — `create_array` is a
     zarr-v3-only method; zarr 2.x's equivalent is `Group.create_dataset`.
     `pyproject.toml` pins `zarr>=2.16,<3` specifically because v3 changed
     this interface (`CLAUDE.md` §6.4) — meaning the pin was correct but the
     code calling it was written against the wrong version's API and this
     had apparently never been caught, because every previous "smoke test
     green" run must have been either mocked at a level that didn't reach
     this code path or run in an environment that silently had zarr 3.x
     installed despite the pin (not verified which; not worth archaeology).
     **Fix:** renamed every `create_array` call in `store.py` to
     `create_dataset` (verified zarr 2.18.7's `Group.create_dataset` accepts
     the identical `shape=`/`chunks=`/`dtype=`/`data=`/`overwrite=` kwargs,
     and that `Array.oindex` — already relied on elsewhere, `CLAUDE.md`
     §11.4 — still works). Smoke config now runs clean end-to-end again (10
     sections, 18 findings, matching the previously-recorded baseline) with
     the *actual* pinned zarr version, for what may be the first time.
  2. **Bug found #2 — `impulse_alignment_check`'s hardcoded impulse
     amplitude (`+= 8.0`) is badly miscalibrated for any tokenizer that does
     context-adaptive global rescaling, which is exactly what Chronos's
     `MeanScaleUniformBins` tokenizer does.** First `--check-alignment`
     against live `amazon/chronos-t5-small` gave a worrying result: min
     diagonal-hit fraction **0.06** (essentially the 1/16 chance floor),
     mean 0.48, oscillating non-monotonically across layers (0.50, 0.06,
     0.88, 0.62, 0.44, 0.38) — `CLAUDE.md` §6.3's own doctrine says this
     pattern ("near-zero everywhere... fix the adapter before trusting any
     cross-model number") should block trusting Chronos numbers entirely.
     Per §2.4 doctrine, measured rather than assumed a broken adapter:
     directly tokenized the impulse-test's input batch and diffed token IDs
     against the unperturbed baseline. Result: **472 of 513 tokens changed
     for every single window position tested** — not just tokens near the
     impulse. The test's single amplitude-8 spike (8x the base sine's unit
     amplitude) is large enough to shift Chronos's per-sequence,
     quantile-based bin edges, so nearly the *entire* token sequence shifts
     bins regardless of where the impulse actually sits — the test's
     diagonal-dominance signal was being drowned in this global-rescaling
     noise, not measuring a real misalignment. Confirmed by sweeping
     impulse amplitude (8.0 → 2.0 → 0.5 → 0.15, as a fraction of the base
     signal's own unit amplitude): Chronos's diagonal-hit fraction rose
     monotonically to a **perfect 1.00 at every layer at amplitude ≤0.3**,
     while TimesFM stayed at a perfect 1.00 across the *entire* sweep
     (0.15–8.0) since its patch-MLP embedding has no such global-rescaling
     sensitivity. **Fix:** `alignment.py`'s impulse is now `0.25 ×
     base.max()` (relative to the probe signal's own amplitude) instead of
     a hardcoded absolute constant, with the mechanism and the measured
     numbers recorded in a code comment so a future session doesn't have to
     rediscover this. Re-verified: **both TimesFM and Chronos-T5 now show a
     perfect 1.00 diagonal-hit fraction at every captured layer** — a much
     stronger, more trustworthy confirmation of the alignment machinery
     than either the misleading pre-fix Chronos numbers or anything tested
     before on live weights. `CLAUDE.md` invariant 7 ("alignment is verified
     empirically on every new library version") is now actually satisfied
     for both adapters against their documented checkpoints, for the first
     time.
  3. **Both `CLAUDE.md` §9 live-weights unknowns confirmed working.** Ran
     the full `tsfm_lens/configs/medium_run.yaml` pipeline (retargeted at
     the `benchmark_medium` corpus above) end-to-end against real
     `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-small`
     checkpoints on the RTX 5070 — all 12 stages, 10 report sections, 19
     findings, ~10.5 minutes wall-clock. Chronos's `output_attentions` path
     and TimesFM's SDPA-unfusing module scan (`CLAUDE.md` §6.2's corrected
     capability-matrix entry) both produced real, sane attention statistics:
     TimesFM's `future_mass` head-score is **exactly 0.0 at every layer/head
     as it must be for a causal decoder** (no attention leaks to future
     positions — a strong sanity check the whole capture mechanism is
     wired correctly, not just "doesn't crash"), and Chronos's first-step
     decoder cross-attention peaks at **lag 0 at every layer** (the decoder
     attends most to the single most recent context step when predicting
     the first forecast step — a plausible, non-degenerate real finding).
  4. **The flat TimesFM L3 patching curve: confirmed still flat at the
     whole-layer level, and confirmed resolved at the per-window level —
     both halves of `CLAUDE.md` §10's open question, answered.** Whole-layer
     (all-windows-at-once) restoration remains genuinely flat across depth
     on real weights (e.g. `level_shift`: 0.036, 0.037, 0.037, 0.037, 0.035
     across relative depths 0/0.25/0.5/0.75/1.0; `deseasonalize`: 0.013,
     0.007, 0.011, 0.015, 0.016) — so that part of the old finding was not a
     mock-model artifact, it reproduces on real checkpoints too. But the
     **per-window** restoration heatmap (`l3/patching.npz`,
     `restoration_windows_TimesFM`, shape `[corruption, layer, window]`)
     shows exactly what the whole-layer average was hiding: restoration is
     overwhelmingly concentrated in **window 15 of 16 — the context window
     immediately preceding the forecast horizon** — and for several
     corruptions that concentration *strengthens* with depth rather than
     staying flat: `noise`'s window-15 restoration climbs 0.068 → 0.223 →
     0.286 → 0.370 → 0.412 from layer 0 to layer 16, and `warp`/`dropout`
     show the same growing-with-depth pattern, while early windows'
     restoration correspondingly *shrinks* with depth (`noise` window 0:
     0.091 → 0.077 → 0.014 → 0.011 → 0.009). Chronos-T5 shows the same
     late-window concentration (e.g. `level_shift` window 15: 0.565, 0.587,
     0.256 across its 3 sampled layers). Read plainly: **the whole-layer
     flatness was exactly the method-coarseness artifact `CLAUDE.md` §10
     already suspected** — averaging 16 windows where 15 contribute close
     to nothing and 1 carries a strong, depth-growing signal produces a
     flat-looking mean by construction. Per-window patching recovers a
     genuine, interpretable causal-structure finding (recency-dominated
     forecast restoration, intensifying at greater depth for corruptions
     that touch the tail of the context) that whole-layer patching could
     not have found no matter how many layers or corruptions were tried.
     This is exploratory (n≈300, 3 families, one small checkpoint pair) —
     the per-corruption interpretation above should be treated as a
     hypothesis for a future dev-corpus run to test at scale, not a
     confirmed finding in the `confirm`-stage sense (§6.7).
  5. **First real (small) L0/L2 numbers, for context, not as final
     findings** (n=288, 3 families, `chronos-t5-small` not `-base`):
     overall MASE favors TimesFM (1.99 vs. 2.35, paired-bootstrap mean gap
     0.363, CI [0.150, 0.608], p=0.004) — directionally consistent with the
     `CLAUDE.md` §10 "TimesFM stronger on tested families" finding from the
     earlier small run, now corroborated on real checkpoints and a
     different corpus. Per-family: TimesFM significantly favored on
     `random_parametric` (Holm p=0.006); no significant gap on `parametric`
     or `mixture` (the real-derived family). L2 stitching gain over the
     input-feature baseline was substantial and clearly non-zero in **both**
     directions (TimesFM→Chronos best gain 0.287, CI [0.256, 0.320];
     Chronos→TimesFM best gain 0.335, CI [0.284, 0.382]) — a real,
     well-supported signal of shared structure beyond what raw input
     statistics would predict, exactly the kind of evidence `CLAUDE.md`
     §6.5's L2 discipline exists to produce.
  6. **Environment versions worth recording as load-bearing, same pattern
     as `datasets`/`tsbootstrap` before them:** `torch==2.9.1+cu130`,
     `transformers==4.57.6`, `timesfm==2.0.2`, `chronos-forecasting==2.3.1`,
     `zarr==2.18.7` all verified working together in `tsfmPy` alongside the
     existing `numpy==2.1.0`/`mkl==2024.2.2` pins. Installing these
     downgraded `huggingface-hub` from 1.26.0 to 0.36.2 (a `timesfm`/
     `chronos-forecasting` transitive constraint) — re-verified this didn't
     break `datasets`-based Monash/ETT loading (full `tsfm_benchmark` test
     suite still 40/1, unchanged).
  7. Also fixed a small, unrelated doc-drift item flagged in `CLAUDE.md`'s
     own reconciliation note: `configs/default.yaml`'s `attention:` block
     comment claiming TimesFM "exposes no patterns (functional attention)"
     was stale since the TimesFM 2.5 adapter rewrite added
     `attention_patterns` support (`CLAUDE.md` §6.2) — corrected now that
     this session's live run gave a concrete, dated confirmation to cite.

- **2026-08-04 — investigated a user-reported observation against the real
  `runs/medium_run` artifacts (not the smoke config): `detrend` "causes no
  action," `dropout`/`spike` "cause very little," in the L3 report. Verified
  empirically (`CLAUDE.md` §2.4) rather than assumed either "expected" or
  "a bug."** Pulled the actual `l3/sensitivity.npz`/`patching.npz` numbers
  from that run and directly measured each corruption's raw input
  perturbation magnitude on representative data (`CORRUPTIONS[name](v, ...)`
  called directly, no model involved) to separate "is the corruption itself
  weak" from "is the model robust to it":
  - **Not a bug in the corruptions.** `detrend`'s mean absolute input change
    (3.38) is comparable to `noise` (1.64) and `deseasonalize` (3.13) — it is
    not a weak perturbation. Its comparatively small effect is on the
    **model's reaction**: both TimesFM (0.151) and Chronos-T5 (0.310) showed
    their *smallest* activation-fingerprint change of any of the 9
    corruptions under `detrend`, despite it being a substantial input
    change — a genuine finding that both models are relatively insensitive
    to global linear-trend removal, consistent with forecasting from recent
    local level rather than committing to an extrapolated trend.
  - `spike` and `dropout` have small input footprints **by construction** —
    `spike` touches 3 of 512 timesteps (0.6% of points, mean|Δ|=0.14 despite
    a max|Δ| of 46.9 at the touched points), `dropout` touches ~14% — so a
    smaller *average* activation/behavioral change than global corruptions
    (`noise`, `frequency_shift`, `level_shift`) is arithmetically expected
    regardless of model, not evidence either model is specifically robust to
    them. The per-window patching heatmap (not the aggregate bar) is the
    right place to check whether the touched region's damage is still
    causally recoverable — it is: `spike`/`dropout` peak per-window
    restoration (0.14–0.29) is on par with `deseasonalize`'s (0.11–0.15).
  - **The real culprit for "looks like nothing":** `level_shift`'s configured
    strength (a permanent step of several std over the back 40% of the
    series) produces behavioral sensitivity 4–12x every other corruption's
    (TimesFM: 11.67 vs. 0.64–2.92 for everything else) — on the report's
    linear-scale "Behavioral sensitivity" bar chart, every other bar is
    visually compressed near zero by comparison, even though `detrend`
    (1.58), `warp` (1.59), and `frequency_shift` (2.92) are not small in
    absolute terms. This is a **display artifact of an uncalibrated
    corruption battery**, not a finding about the models, and the report's
    own `_note()` already said as much in prose — it just wasn't visible
    enough against the chart itself.
  - **Fix (report-only, no corruption/statistic changed):**
    `report/report.py`'s Behavioral sensitivity chart now prints each bar's
    value as on-chart text (so a reader isn't relying on bar height alone),
    the note names `level_shift`'s outsized configured magnitude as a
    concrete worked example, and a new findings entry states each model's
    least/most behaviorally-sensitive corruption in text explicitly flagged
    as not cross-corruption-comparable. Re-ran the `tsfm_lens` smoke suite
    after the change: still green (findings count 18→20, expected from the
    two new per-model callouts).
  - **Decision: kept all three corruptions in the battery, did not drop
    them.** L3's entire design (`CLAUDE.md` §6.1) is a *contrastive*
    signature across structural properties — a corruption a model is
    comparatively robust to is exactly as informative as one it's sensitive
    to, and dropping the "low-signal" end of that contrast would remove the
    ability to say "model X is comparatively less sensitive to Y than to Z."
    See `CLAUDE.md`'s L3 description for the added doctrine note capturing
    this so it isn't rediscovered as a false alarm again.
  - **Not done, left as an option, not a decision:** recalibrating
    corruption *strengths* to be mutually comparable (e.g. matching each
    corruption's configured parameter to a similar input-energy budget)
    would make the Behavioral sensitivity chart directly cross-corruption
    comparable without needing the display fix above. This is a real
    methodology change (touches every `l3.corruptions` config default across
    the repo, and would invalidate the exact numbers already recorded in
    this file and `CLAUDE.md` §10) — not attempted this session since it
    wasn't asked for and changes already-cited results; Phase 3's controlled
    parameter sweeps (§7) would be the natural place to pursue it
    deliberately if a future session wants to.
    > ⚠️ **AUDIT (2026-08-06):** now specified as §15 A12, with the
    > backwards-compatibility problem this entry correctly identified solved
    > rather than used as a reason to defer: calibration goes behind an
    > opt-in `l3.calibrate` mode (default `none`), so every number already
    > recorded in this file keeps reproducing byte-for-byte while a calibrated
    > run becomes possible for the first time. The display-only fix this
    > session shipped is the right *interim* answer, not a substitute — a
    > prose caveat under a chart cannot make two bars comparable.
- **2026-08-04 — repo-wide hardcoded-absolute-path audit** (user request,
  `CLAUDE.md` §7 invariant 11): grepped the whole repo for drive letters,
  `/home/`, `/Users/` across `.py`/`.yaml`/`.md`/`.toml`/`.sh`. Found exactly
  one violation — `tsfm_lens/configs/medium_run.yaml`'s two `data.path`/
  `confirm.path` fields were machine-specific absolute paths
  (`c:/Users/DaGuest/GitRepos/TSFM-Interp/benchmark_medium/...`) instead of
  relative to `tsfm_lens/` (this file's own documented invocation directory,
  matching every other config in the repo). Fixed to
  `../../benchmark_medium/public_dev` / `.../private_test`; verified the
  fixed paths still resolve and exist via `load_config` + `Path.resolve()`.
  Every `sys.path.insert` call already used the correct
  `Path(__file__).resolve().parents[N]` pattern — no fix needed there.
- **2026-08-04 — created `DEPENDENCIES.md` (repo root)** documenting the
  `tsfmPy` conda environment's exact verified package versions (user
  request). While building it, **resolved a previously-deferred open
  question from this file's own 2026-08-03 Findings**: "CUDA pytorch is not
  installable in the same conda-forge env as the numpy/scipy/validation
  stack without re-risking the MKL crash... decision deferred." Directly
  tested today: `numpy` 2.1.0 matmul, `torch` 2.9.1+cu130 CUDA availability,
  and `sdv`/`ctgan`/`deepecho` imports all coexist and function together in
  the single `tsfmPy` environment with no crash — this must have been
  superseded by the §5.4 session's in-place torch upgrade to the CUDA
  build, but nothing in this file said so until now. Also newly confirmed:
  SDV's `PARSynthesizer` (the `sequential_par` generator) defaults to
  `cuda=True` and runs fine in this environment — the GPU path this file
  previously flagged `[!]` blocked-on-decision (§5's "CUDA pytorch is not
  installable..." finding) is **not blocked by anything found today**; that
  finding is now stale, corrected here rather than silently left to mislead
  a future session. `tsfmPy` is confirmed to be genuinely one shared
  environment for both halves of the repo, not two.

- **2026-08-05 — a third machine/environment, same "wired but unverified
  pin" trap recurring, now fixed for a third time.** This session ran on a
  new machine never documented before in `CLAUDE.md`/`ROADMAP.md`/
  `DEPENDENCIES.md`: Linux (not Windows), an existing `cudaPy` conda env
  (not `tsfmPy`), **8× NVIDIA RTX A5000 (24 GB each)**. Checked package
  versions against `DEPENDENCIES.md` before trusting anything (per
  `CLAUDE.md` §11.15's own lesson: "pin compliance should be spot-checked,
  not assumed"), and found this env had **zarr 3.2.1** installed —
  exactly the v3 line whose `Group.create_array`/no-`create_dataset` API
  break was already diagnosed and fixed once, in §5.4/`CLAUDE.md` §11.15,
  on a *different* machine. `store.py`'s current code calls
  `create_dataset` (the v2-API fix), which does not exist on zarr 3.x's
  `Group` (confirmed directly: `dir(zarr.group())` on this env showed
  `create_array`/`create_group`/`create_hierarchy`, no `create_dataset`) —
  so extraction would have crashed immediately on this machine exactly
  the way it did before the §11.15 fix, had the version not been checked
  first. **Fix:** `pip install "zarr>=2.16,<3"` in `cudaPy`, which resolved
  to `zarr==2.18.7` — the same exact version already verified in
  `DEPENDENCIES.md` — with no other package changes needed. Also missing
  in this env (not needed for this run: `benchmark_medium/` was already
  built and present at the repo root, so no `build_pipeline` regeneration
  was required): `tsbootstrap`, `mkl` (as a Python-importable package —
  irrelevant, `mkl` is a C runtime `conda` manages, not a `pip`-importable
  module, so this "MISSING" is not a real gap). `torch` in this env is
  `2.12.0` (newer than `DEPENDENCIES.md`'s verified `2.9.1+cu130`) and
  worked with no observed issue for this run — not re-pinned or
  downgraded, since nothing broke, but flagged here rather than silently
  assumed identical to the documented version. Re-ran
  `--check-alignment` for both `TimesFM` and `Chronos-T5-Base` on this env
  before trusting the run (`CLAUDE.md` invariant 7) — both perfect 1.00
  diagonal-hit fraction at every layer, matching the prior machine's
  numbers exactly. **Lesson, now demonstrated a third time:** a "verified
  working" environment claim is scoped to the machine it was verified on;
  moving to new hardware requires re-checking the pin, not re-reading the
  doc and assuming it travels. `DEPENDENCIES.md` intentionally is not
  updated to describe this third environment in full (it documents one
  reproducible recipe, not every machine this repo has run on) but this
  finding is the record that the `zarr<3` pin is now confirmed load-bearing
  across three independent environments, not one.

- **2026-08-05 — chronos-t5-base vs. chronos-t5-small comparison (§5.3):
  the tested behavioral finding is size-robust; representational-similarity
  metrics are not, and disagree with each other about the direction of the
  size effect.** Same `benchmark_medium` corpus, same TimesFM checkpoint,
  same seed, only `chronos-t5-small` (6 encoder blocks) swapped for
  `chronos-t5-base` (12 blocks) — isolating the size confound as designed.
  - **Behavioral (L0), size-robust:** overall MASE barely moved
    (Chronos-T5 2.365 → Chronos-T5-Base 2.355, TimesFM unchanged at 1.991
    since it's the same checkpoint/corpus) — doubling Chronos's encoder
    depth bought **~0.4% overall MASE improvement**, not a proportional
    gain. Per-family, the one pre-registered, statistically significant dev
    finding (`random_parametric` favors TimesFM) is essentially identical
    at both sizes: ratio 0.776 (small) vs. 0.777 (base), Holm p=0.006 both,
    and **both replicate on the sealed private corpus** (confirm mean gap
    0.454 small / 0.455 base, p=0.002 both) — this specific finding is not
    a smaller-model artifact. `mixture`/`parametric` remain non-significant
    at both sizes.
  - **L1 CKA, grows with size:** peak CKA rose from **0.276** (small, at
    TimesFM `stacked_xf.4` / Chronos `encoder.block.4` — block 4 of 6, 80%
    relative depth) to **0.381** (base, at TimesFM `stacked_xf.4` / Chronos
    `encoder.block.10` — block 10 of 12, 91% relative depth) — a ~38%
    relative increase, both CIs well clear of their null baselines
    (~0.036-0.037 either way), and **both replicate on private data**
    (0.275 small / 0.374 base). The peak partner layer stayed at a similar
    *relative* depth on Chronos's side (late-but-not-final) even though
    the absolute layer index moved.
  - **L2 stitching gain-over-baseline, grows with size in both
    directions:** TimesFM→Chronos best gain 0.287 (small) → 0.318 (base);
    Chronos→TimesFM best gain 0.335 (small) → **0.413** (base) — a ~23%
    relative increase, the larger of the two directions' growth. Both
    remain clearly non-zero (CIs excluding zero) at both sizes — the bigger
    Chronos is more linearly stitchable to/from TimesFM, not just a bigger
    version of the same stitchability.
  - **L4 clustering AMI, *shrinks* with size — the opposite direction from
    CKA/L2:** cross-model cluster agreement dropped from **0.705** (small)
    to **0.538** (base), using the same `layer: auto` policy (each model's
    side of its own L1 peak-CKA pair) at both sizes. This is the headline
    caution of this comparison: two different "shared structure" metrics
    computed on the *same* size change moved in **opposite directions** —
    CKA/L2 say the bigger Chronos is more geometrically/translatably
    similar to TimesFM; AMI says its own actual activation partition
    agrees *less* with TimesFM's. Per `CLAUDE.md` §6.1's evidence-class
    discipline, L1/L2 are the geometric/translatable claims and L4
    clustering is explicitly "descriptive... approximate by construction"
    — this isn't a contradiction to resolve, it's a demonstration that
    these evidence classes answer different questions and **do not
    substitute for each other**, a concrete instance of why the layered
    method exists rather than picking one favorite metric.
  - **Lens crystallization depth — flagged as not directly comparable, not
    reported as a finding:** small run's lens stage used
    `lens.max_series: 32`, base run's used `24` (lowered for the larger
    Chronos's VRAM footprint, per that config's own comment) — different
    small subsets of the corpus, not the full 288/286 series L0/L1/L2 run
    over. The raw numbers (crystallization depth 0.8 small → 0.909 base;
    lens-internal final MASE 2.277 small → 1.680 base **for the same
    TimesFM checkpoint**, which cannot be a real per-model change) make
    this obvious once checked — the difference is which ~24-32 series got
    sampled, not a real lens-depth effect. Recorded here specifically so a
    future session doesn't mistake it for a real number; a real
    size-vs-crystallization-depth comparison would need matched
    `max_series` across configs.
  - **Confirm stage — both sizes tested against the same sealed private
    corpus, deliberately, not a seal violation:** `CLAUDE.md` §6.7's
    "confirm runs once" discipline concerns *mining new hypotheses* against
    private data; here, the exact same single pre-registered hypothesis
    (`random_parametric` favors TimesFM, registered on dev before this
    session) was re-tested against private data once per model
    configuration, as part of *this specific* pre-planned size-confound
    check — no new hypothesis was fished for using private results. Stated
    explicitly here rather than silently reusing the private corpus without
    comment, per §6.7's own "if peeked at, regenerate" caution — this is
    not the kind of peeking that caution is about, but it's a judgment call
    worth recording, not assuming.
  - **Net read:** the one specific dev-to-private-confirmed finding this
    corpus can currently support (`random_parametric` favors TimesFM) is
    not a size confound. But this is exactly one finding, on one corpus, at
    two sizes of one model family — it licenses "this particular result is
    size-robust," not "results in this report are generally
    size-independent." The AMI/CKA disagreement above is itself evidence
    that extrapolating any single internals metric's story across
    checkpoint sizes needs to be checked, not assumed, every time.

- **2026-08-05 — carried forward from `CLAUDE.md` §10 ("Findings from the
  one small real run"), which is now retired in place of this file per
  §5.5's own instruction** ("retire it from `CLAUDE.md` and move the
  *current*, broader-based version of these findings into this file's
  Findings block, since `CLAUDE.md` should describe stable architecture,
  not a moving research result"). This is the **verbatim historical
  record** of that section, predating both this roadmap and the §5.4
  live-checkpoint session below it in wall-clock time (it describes an
  earlier, smaller, likely mock- or first-pass-real-model run) — preserved
  rather than deleted per this file's own append-only Findings doctrine,
  just relocated to where research results belong:
  - *Defensible:* TimesFM stronger on the tested families. TimesFM
    front-loads then compresses family-relevant information; Chronos's
    encoder accumulates it. The models share learned structure beyond
    input statistics but **not** global geometry. TimesFM's activation
    space is far more family-organized (cluster purity **0.90–0.98** vs
    **0.60–0.79**; AMI **0.40**).
  - *The causal payload:* fingerprint agreement is middling overall
    (ρ=**0.46**), but per-corruption is the story — depth responses to
    **additive noise are strongly anti-correlated (ρ=−0.90)**: where one
    model's noise sensitivity grows with depth, the other's shrinks.
    Plausibly continuous-embedding vs. quantization, though the mechanism
    claim needs per-layer curves. They **agree** on deseasonalization
    (ρ=**0.76**).
  - *Underdelivered, resolved 2026-08-03 (§5.4 above):* TimesFM's patching
    restoration was flat across captured layers — diagnosed as whole-layer
    window patching being too coarse to localize anything in a
    residual-stream model, not a finding about TimesFM, and fixed with
    per-window patching. Confirmed on real checkpoints in §5.4's Findings
    above: whole-layer patching genuinely is still flat, but the
    per-window breakdown recovers real, depth-intensifying causal
    structure concentrated in the context window nearest the forecast
    horizon.
  - **What has and hasn't been superseded by broader evidence since:** the
    per-window-patching resolution is now confirmed on real weights (§5.4).
    The `random_parametric`-favors-TimesFM behavioral claim now has a
    second, independent confirmation on a third run and checkpoint size
    (`runs/real_run`, and the base-vs-small comparison directly above) —
    see the new `run_meta_report.py` output (§5.5) for the three-run
    stability table. The cluster-purity/AMI/fingerprint-ρ numbers above
    have **not** been re-measured against real checkpoints at the scale
    this file's later sessions used — they remain exactly what they were
    when first measured, an early, small, provisional read, not yet
    superseded or contradicted. Flag any future re-measurement here rather
    than silently treating the numbers above as current.

---

## 6. Phase 2 — Layer selection, novel SAEs, and IP detection (the "meat")

This is the brief's highest-value, most novel-research-shaped ask (items 1
and 3), plus a related but separable idea (item 2). Broken into three
sub-phases because they have different dependencies (§3).

### 6.1 Phase 2a — Does anything predict "this layer is worth interpreting"? `[x]`

> ✅ **First pass DONE 2026-08-05.** Superseded as a *selector* by §6.1.1;
> kept for its cross-model *finding*.

> **See §6.1.1** for a full redesign of the layer *selector*. The mechanism
> below (`recommend_layers`) turned out to be a cross-(run, model) correlation
> study that does **not** transfer as a within-single-model selection rule
> (§6.2 Findings proves it picks a model's *worst* layer). §6.1.1 specifies an
> architecture-agnostic, all-layers-fair, parsimonious replacement and the
> bake-off that chooses between candidates. Read §6.1 as the *scientific
> finding* it is; do not use it to pick SAE targets.

**Goal.** Test the brief's central hypothesis directly: *does a layer's
effective dimensionality (participation ratio, already computed in
`analysis/internals.py`) correlate with how interpretable/controllable that
layer turns out to be?* If yes, that's both a genuine finding and a practical
layer-selection rule for Phase 2b. If no, the fallback is just as valuable —
know that before spending compute training SAEs on the wrong layers.

**Concrete deliverables**
- [x] Define "interpretable/controllable" operationally, since it's not a
  single number today. Proposed composite (test each piece separately before
  trusting a composite):
  - **Family-probe decodability** at that layer (already in `internals.py`).
  - **Tuned-lens R²** against the model's own final forecast at that layer
    (already in `lens.py`) — a direct "is the answer linearly readable here"
    signal.
  - **L3 causal sensitivity magnitude** at that layer, i.e. does anything
    causally live here, or is it a representational way-station
    (`l3_perturbation.py` sensitivity fingerprints already computed).
  - **Post-hoc SAE quality**, once 2b exists: reconstruction fidelity at
    fixed sparsity, dead-feature rate, and (§2.1/§6.3 below) ground-truth
    feature-alignment score. This piece necessarily runs *after* some SAEs
    exist, so treat it as validating/falsifying the earlier proxies
    retroactively, not as required before starting 2b. — **used the three
    proxies above as-is** (all three already existed in run artifacts, no
    new computation needed); the fourth remains blocked on Phase 2b as
    documented, not attempted.
- [x] Compute effective dimensionality (and candidate alternatives — see
  below) per layer per model across every Phase 1 run, and correlate against
  each interpretability proxy above. Use the same series-level bootstrap
  discipline as the rest of the repo (`CLAUDE.md` §6.6) to put a CI on the
  correlation, not just a point estimate. — **done**:
  `tsfm_lens/analysis/layer_selection.py`, pooling one record per
  (run, model, layer) from every run directory that exists
  (`medium_run`, `medium_run_chronos_base`, `real_run`, `smoke` — literally
  "every Phase 1 run," since Phase 1 closed §5.5 immediately before this).
  The resampling unit is the **(run, model) group**, not the layer — layers
  within one model are depth-autocorrelated the same way windows within a
  series are (`CLAUDE.md` §6.6's discipline, extended to this differently-
  shaped data), so the cluster bootstrap resamples which of the 8
  (run, model) groups are included, keeping every group's layers together.
  Reused `analysis/stats.py`'s existing generic `bootstrap_ci` rather than
  writing a new bootstrap primitive. See Findings below for numbers.
- [x] Test at least these alternative candidate metrics against the same
  proxies, since the brief explicitly invites better metrics if they beat
  effective dimensionality:
  - Per-layer CKA-to-input (already computed in `internals.py`) — a layer
    that's barely moved from raw input statistics is a poor SAE target for
    different reasons than a too-high-dimensional one. — **tested**, weaker
    signal than effective dimensionality (see Findings).
  - Per-layer L3 fingerprint *entropy* across corruption types (a layer
    causally sensitive to everything vs. to one specific structural
    property might make differently-useful SAE targets). — **tested**, the
    single most robust correlation found in this whole study (see
    Findings) — a genuine case of the brief's "propose a better metric"
    invitation paying off.
  - Simple activation kurtosis / sparsity of the raw (pre-SAE) activations —
    cheap, worth ruling in or out early. — **not tested this session**:
    unlike the other three candidates, this needs the raw per-window
    activations from the zarr store (`extraction/store.py`), not an
    already-written summary artifact, which is a real (if modest) new
    plumbing cost the other three didn't have. Flagged as follow-up, not
    silently skipped.
- [x] Write up the result as a genuine finding either way: "effective
  dimensionality [does/doesn't] predict X, with these caveats" — and if it
  does, turn it into an actual `layer_select: auto` policy usable by Phase 2b
  and by `analysis/clustering.py`'s existing `layer: auto` (which today
  picks the L1 peak-CKA pair for an unrelated reason — reconcile or
  distinguish these two "auto" policies explicitly so they don't get
  confused). — **done, with a deliberate naming decision**: effective
  dimensionality *does* predict two different proxies, but in **opposite
  directions**, so there is no single "worth interpreting" score to turn
  into one `layer: auto`-style flag without hiding that disagreement. Added
  `recommend_layers(records, run, model, goal, top_k)` instead of a
  goal-less "auto" — `goal` must be `"forecast_readability"` (ranks by
  lowest effective dimensionality) or `"family_decodability"` (ranks by
  highest L3-fingerprint entropy), explicitly distinct from
  `clustering.py`'s own `layer: auto` (which answers "where do the two
  models' representations most agree," an unrelated question). Not yet
  wired into any config or Phase 2b training loop — that wiring is Phase
  2b's own concrete deliverable, not this one's.

**Findings / decisions**
- **2026-08-05 — effective dimensionality predicts two different
  interpretability proxies, in *opposite* directions; a proposed
  alternative (L3 fingerprint entropy) beats it on the proxy where they're
  comparable.** Pooled 64 (run, model, layer) records across all 4 existing
  run directories (8 (run, model) groups: TimesFM×3 real-checkpoint runs +
  3 different Chronos variants + 2 mock architectures from `smoke`).
  Every correlation below is Spearman ρ with a cluster-bootstrap 95% CI
  (n_boot=4000, resampling the 8 (run, model) groups) reported both raw and
  after linearly detrending both variables against relative depth first
  (`_add_depth_residuals`) — the depth-controlled column is the one to
  trust, since three of the four candidate/proxy metrics *themselves*
  trend significantly with depth (`l3_entropy` ρ=+0.47, `probe_decodability`
  ρ=+0.32, `tuned_r2_model` ρ=+0.37, all CIs excluding zero; `effective_dim`
  and `input_cka` do **not** trend significantly with depth, CIs straddle
  zero) — so a raw correlation between two depth-trending metrics risks
  just restating "both increase with depth," exactly the kind of
  confound-blind result §2.2's null-control discipline exists to catch.
  - **Robust, sign-flipped relationship (the headline result):**
    `effective_dim` vs. `tuned_r2_model` (tuned-lens R² against the model's
    own forecast): raw ρ=−0.661 [−0.872,−0.210], depth-controlled
    ρ=−0.686 [−0.871,−0.329] — **significant both ways, barely moved by
    depth control**, so this is a real, depth-independent relationship, not
    a depth artifact. **Lower** effective dimensionality predicts a
    **more linearly-readable forecast** at that layer.
    `effective_dim` vs. `probe_decodability` (family-probe accuracy): raw
    ρ=+0.511 [−0.030,+0.809] (not quite significant), depth-controlled
    ρ=+0.653 [+0.176,+0.859] (significant) — the *opposite* sign from the
    forecast-readability relationship. **Higher** effective dimensionality
    predicts **better family decodability**, once depth is controlled.
    Read together: effective dimensionality does not have one "is this
    layer worth interpreting" story — a layer that's easy to linearly read
    the forecast *from* tends to be a layer where family identity is
    *harder* to decode, and vice versa. This is exactly the kind of result
    §6.1's "if yes, that's a genuine finding" framing anticipated, but
    sharper and more useful than a single yes/no: **the answer depends on
    which notion of "interpretable" is meant**, which is why
    `recommend_layers` takes an explicit `goal` rather than picking one.
  - **A proposed alternative that beats effective dimensionality on the
    proxy where they're comparable:** `l3_entropy` (normalized Shannon
    entropy of a layer's per-corruption sensitivity fingerprint — high
    means the layer reacts broadly across corruption types rather than to
    one specific structural property) vs. `probe_decodability`: raw
    ρ=+0.642 [+0.216,+0.853], depth-controlled ρ=+0.674 [+0.247,+0.823] —
    **significant both ways, and a tighter CI than effective dimensionality's
    own (depth-controlled) relationship with the same proxy** (0.176–0.859
    vs. 0.247–0.823 — narrower, more confidently away from zero). Per this
    file's §2.2 doctrine ("novelty must be tested against the null, not
    just proposed") and the brief's own invitation to find a metric that
    beats effective dimensionality: **on the family-decodability proxy,
    L3 fingerprint entropy is the stronger of the two tested candidates.**
    It also shows a borderline depth-controlled relationship with
    `tuned_r2_model` (ρ=−0.523 [−0.745,−0.015], CI just clears zero) worth
    flagging as weaker/tentative rather than reported with the same
    confidence as the results above.
  - **input_cka: weaker signal than effective dimensionality throughout.**
    vs. `probe_decodability`: raw not significant, depth-controlled
    ρ=+0.526 [+0.145,+0.850] (significant but wider CI than either
    `effective_dim` or `l3_entropy`'s versions of the same test). vs.
    `tuned_r2_model`: not significant either way. Real but the weakest of
    the three tested representational candidates.
  - **`l3_mean_sensitivity` (the causal-magnitude proxy) was not
    significantly predicted by any candidate metric tested**, raw or
    depth-controlled. Read as a genuine negative result for this small
    sample, not evidence the proxy itself is uninformative — it's a
    different kind of signal (how much a layer's *activations* move under
    corruption) than the two proxies that did show structure.
  - **Kurtosis/sparsity of raw activations: not tested**, per the
    checklist note above — flagged rather than silently dropped.
  - **Honest limitation, stated plainly rather than glossed over:** `n_groups
    = 8` for every correlation above. A cluster bootstrap over 8 units gives
    real but wide CIs (visible above), and every "significant" result here
    should be read as **a genuine signal worth building on, not a settled
    fact** — Phase 1's future corpus-breadth work (§5.1's still-unbuilt
    literal-scale `full_multidomain.yaml`) and Phase 4's multi-model
    expansion would each add more (run, model) groups and meaningfully
    tighten these CIs. Re-run `run_layer_selection_study` (now a one-line
    call, `tsfm_lens/analysis/layer_selection.py`) against whatever new run
    directories exist next, rather than treating today's numbers as final.
  - Raw study output: `runs/layer_selection_study.json` (not committed —
    it's a run artifact, regenerable from existing run directories with no
    new model calls). Three regression tests added
    (`tests/test_layer_selection.py`): the cluster-bootstrap primitive
    against synthetic data with a known correlation and a known null (must
    find one, reject the other), the too-few-groups guard, and
    `recommend_layers`'s direction/goal-validation logic. Full `tsfm_lens`
    suite re-run after: 10/10 passing, no regressions.

### 6.1.1 Phase 2a-v2 — an architecture-agnostic, all-layers-fair layer selector `[x]`

> ✅ **DONE 2026-08-05** — implemented, bake-off-tested, and wired into
> production as the `layer_screen` stage (`work_bend` is the default method).

> **Status.** Design written 2026-08-05 (below), implemented and empirically
> bake-off-tested the same day (`tsfm_lens/analysis/layer_screen.py`,
> `layer_screen_bakeoff.py`, `run_layer_screen_bakeoff.py`,
> `tests/test_layer_screen.py` — 12 new unit tests, all passing, plus the full
> 33-test `tsfm_lens` suite re-run clean). The bake-off ran for real against
> live checkpoints, not mocked data — see the Findings block after Idea C
> below for the full result. **Same-day follow-up (below the bake-off
> Findings): wired into the pipeline as a default stage.** The result was
> genuinely mixed (no single method cleanly wins both models), but the user
> explicitly directed this to be made the production default rather than
> stay a standalone experiment — `work_bend` (Idea A) is the wired default
> precisely *because* it was the bake-off's best performer, not despite the
> mixed result; `coverage` and `factor_emergence` remain one config-line
> swaps for re-testing on new architectures. Treat every "should"/"expect" in
> the design sections below as the hypothesis it was when written; the two
> Findings blocks after Idea C are what actually happened, first in the
> bake-off, then in production.

**Why §6.1's method is not the final answer.** §6.1 delivered a real
*scientific finding* (effective dimensionality predicts two interpretability
proxies, in opposite directions; L3 fingerprint entropy beats it on one). It
did **not** deliver a trustworthy *layer selector*, and the difference is the
whole point of this section. Five concrete, already-logged problems:

1. **It's a cross-model correlation study wearing a selector's clothes.**
   `recommend_layers` is validated only as a pooled, depth-controlled,
   cross-(run, model) trend, and **provably fails within a single model** — on
   `runs/medium_run_chronos_base` it picked `encoder.block.0` for
   `"forecast_readability"`, the layer with the *lowest* tuned-lens R² of all
   12 (0.433), the literal opposite of intent (§6.2 Findings has the exact
   per-layer numbers, and `analysis/layer_selection.py`'s docstring now warns
   of it). A selector's entire job is to rank *one model's own* layers; the
   §6.1 mechanism can't be trusted to.
2. **It's circular and expensive — the dependency runs backwards.** Every
   input it consumes (internals' eff-dim/CKA/family-probe, lens' tuned-R²,
   L3's fingerprint) is an artifact of running the *full, expensive* pipeline
   over *all captured layers first*. Layer selection exists to say *where to
   spend expensive compute* (SAE training, §6.2). A selector that first
   demands the expensive analysis over every layer has the arrow reversed.
3. **It is not fair to every layer.** Capture runs at
   `capture_layer_stride: 2` (`configs/default.yaml` → only 10 of TimesFM's 20
   blocks are ever stored). The skipped layers are never scored — so "find
   *all* interesting layers" is impossible by construction; half of them are
   invisible to the selector before it starts.
4. **Its predictive direction is itself architecture-dependent.** The pooled
   negative eff-dim ↔ tuned-R² relationship is driven mostly by TimesFM and
   **reverses inside Chronos-T5-Base** (§6.2 Findings). Its proxies also lean
   on architecture-specific machinery — tuned-lens needs the model's own head
   *and* uniform hidden size (fails loudly otherwise, `CLAUDE.md` §6.5), and
   "family decodability" is one narrow notion of interpretable. None of this
   is "architecture-agnostic by construction" (§2.4).
5. **n_groups = 8, wide CIs** — a genuine signal, not a robust rule.

**Keep §6.1, but re-scope it.** Its study stays valuable as a *finding* ("does
eff-dim predict interpretability *across* models?"). What changes is that the
selection *mechanism* becomes the redesign below, and `recommend_layers` is
documented as the cross-model finding it actually is — not a per-model
selector. This is the same "distinguish two things that got the same name"
discipline §6.1 already applied to `clustering.py`'s unrelated `layer: auto`.

**Design requirements (from the user's framing, made concrete and testable).**
Every candidate below is judged against these four before the §6.1.1-E
bake-off even runs:

- **R1 — Architecture-agnostic by construction (§2.4).** Operate only on the
  aligned `[series, window, dim]` surface (`CLAUDE.md` §6.3 — the one
  representation guaranteed comparable across architectures), using
  dimension-invariant primitives. Linear CKA / RSA (`analysis/l1_geometry.py`)
  are already the repo's cross-architecture tools and *do not care about hidden
  size*, which sidesteps the lens' fail-loud uniform-hidden-size assumption
  outright. No dependence on the model's own head/decoder; no per-architecture
  layer-name heuristics beyond the adapter's existing `all_layer_names()`.
- **R2 — Fair to every layer.** The *screening* signal must be computed on
  **every** block at **stride 1**, not a strided subset — a new cheap
  screening-extraction mode (e.g. `layer_screen.stride: 1`, independent of the
  analysis `capture_layer_stride`). Fairness means every layer gets a score;
  nothing is invisible. (Note `config.py`'s `capture_layer_stride` *default* is
  already 1; it's `default.yaml` that sets 2 for cost — the screen must pin it
  to 1 regardless of what the analysis config uses.)

  > ⚠️ **AUDIT (2026-08-06) — R2 is violated by the production wiring, see
  > §15 A1.** The `layer_screen` stage as landed reads `store.layers(model)`,
  > i.e. whatever `capture_layer_stride` happened to capture (2 for TimesFM in
  > both `default.yaml` and `medium_run.yaml`), so in every production config
  > half of TimesFM's blocks are invisible to the screen — exactly problem 3
  > this section raises against §6.1, reintroduced. The requirement above was
  > never implemented; no `layer_screen.stride` knob exists. Not a small
  > deviation: the bake-off that chose `work_bend` was run at stride 1, so
  > production selections are not made under the conditions the method was
  > validated under.
- **R3 — Parsimonious ("find all interesting layers, but not too many").** The
  output is a *small* set sized to a compute budget, chosen so the next layer
  added buys little — an explicit coverage/redundancy objective, not a fixed
  top-k of a noisy score.
- **R4 — Cheap enough to run *before* the expensive stages,** so the pipeline
  dependency runs the right way round (this is what §6.1 violated). Ideally one
  extra stride-1 forward-extraction pass plus closed-form linear algebra over
  the stored activations — no lens, no L3, no SAE required to *select*.

---

**Idea A — Residual-trajectory "work + bend" profile (cheap, activation-only).**
*Intuition:* interesting layers are where the model actually *does work* on the
residual stream, and where the representational trajectory *bends* (one kind of
processing ends, another begins). A layer whose representation is nearly
identical to its neighbour's is a way-station, not a target.

*What it computes,* per layer, from the stored `[series·window, dim]` activation
matrix `H_l` (all layers, stride 1):
- **Change profile:** `change_l = 1 − linear_cka(H_l, H_{l+1})`
  (`analysis/l1_geometry.py:linear_cka`). CKA is scale- and dimension-invariant,
  so this is well-defined even when consecutive blocks differ in width and
  across architectures (R1). Peaks = the layers transforming the representation
  the most.
- **Trajectory curvature:** treat each layer's RSA representational-dissimilarity
  matrix (RDM) as a point in a depth-indexed curve; the local turning angle
  between successive RDM *difference* vectors flags a **regime boundary** — the
  depth where the geometry stops changing one way and starts changing another.
  High curvature = interesting even when the raw change magnitude is moderate.
- **Optional same-pass add-on:** participation-ratio effective dimensionality
  (already in `internals.py`, recomputed here at stride 1) and, more usefully,
  its *derivative across depth* — a sharp eff-dim change is a compression or
  expansion event worth flagging.

*Selection:* local maxima of the change profile ∪ curvature maxima; these are
naturally few (R3). *Cost:* one CKA/RDM per layer over already-stored
activations — O(L) linear-algebra passes, zero extra model calls beyond the one
stride-1 extraction (R4). Reuses `l1_geometry.linear_cka` and L1's RSA.
*Strength:* the most purely architecture-agnostic of the three — it makes no
reference to any ground-truth concept or downstream task, only to the geometry
of the model's own trajectory. *Weakness to test:* "does work here" ≠ "carries
interpretable structure here"; §6.1.1-E's gold ranking is exactly what checks
whether geometric bends coincide with SAE-interpretable layers.

---

**Idea B — Ground-truth factor-emergence spectroscopy (exploits §2.1's unfair advantage).**
*Intuition:* a layer is interesting when a *known generative factor* first
becomes linearly readable there (emergence) or stops being readable
(compression/discard). This is the one idea that can say *what* makes a layer
interesting, and validate it against ground truth rather than a proxy.

*What it computes:* for each ground-truth factor the benchmark records — trend
order, dominant seasonal period, `n_seasonalities`, changepoint count, anomaly
count, AR-coefficient energy, noise level, intermittency rate, random-walk
scale, heteroskedastic depth (`build_pipeline`'s `GroundTruth`, surfaced by
`sae/ground_truth.py:load_ground_truth_table`) — fit a cheap held-out probe
(ridge for continuous factors, logistic for categorical) from each layer's
window-pooled activations to that factor, on a **series-level** train/test split
(`CLAUDE.md` §6.6 — never a window split). The output is a `[layers × factors]`
decodability matrix `D` (R² / AUC).

*Three "interesting layer" definitions to test against each other:*
- **Emergence** — the first layer where factor *f* crosses a decodability
  threshold (the depth where the model *has* that information linearly).
- **Loss/compression** — the layer where *f*'s decodability drops after peaking
  (information discarded or entangled); often the *more* interesting event.
- **Transition mass** — per-layer `Σ_f |D[l+1,f] − D[l,f]|`; peaks = "the
  decodable-factor set changed most here."

*Selection:* keep each factor's emergence layer and peak layer, take the union,
dedupe (many factors share transitions → a small set, R3); optionally cover all
factors with the fewest layers (a set-cover, linking to Idea C). *Why it's the
strongest principled candidate:* it is directly ground-truth-verifiable (§2.1),
carries no circular dependence on the pipeline's own downstream proxies, is
architecture-agnostic because ground truth is model-independent and a probe
reads any `[·, dim]` (R1), and is cheap because ridge is closed-form (R4), on
all layers (R2). *Distinct from §6.1's family-probe:* that used a single coarse
"family" label as a *proxy to correlate a metric against*; this uses the *full
factor battery as the selection signal itself*, and reads *transitions*, not
levels. *Weakness to test:* it can only see interestingness that an
*expressible* generative factor captures — a layer whose important content is
something the synthetic ground truth doesn't encode is invisible to it (the
same caveat as catch22 in `CLAUDE.md` §12). §6.1.1-E's SAE-gold target is
precisely what quantifies how much interestingness this misses.

---

**Idea C — Redundancy-coverage selection (submodular; parsimony is the objective, not a knob).**
*Intuition:* stop scoring layers one at a time; instead pick the *smallest set
of layers that covers the representational diversity across depth*. This is the
only idea whose objective function *is* "all interesting, not too many."

*What it computes:* the full `L × L` layer similarity matrix via linear CKA (or
CKA on RDMs), over all layers (R2) — the map of which layers are redundant with
which. Two formulations to compare:
- **Greedy facility-location / set-cover:** the smallest set `S` such that every
  layer has CKA ≥ τ to some member of `S`. τ trades size against coverage and is
  the user's direct "how many is not too many" knob (R3). The objective is
  submodular, so greedy is near-optimal *and* the choice is interpretable
  ("this layer represents this redundant band").
- **Spectral banding:** cluster layers from the CKA affinity matrix, pick `k` by
  the eigengap, take each band's medoid — "one representative per depth-regime."

*Selection:* the cover / medoid set directly. *Cross-model extension worth
testing:* run the coverage over the **aligned union** of both models' bands so
the selected layer *pair* is representationally comparable across architectures —
which is exactly the shared layer a crosscoder needs (§6.2 item 1), making this
idea do double duty. *Cost:* one `L × L` CKA matrix (R4), all layers (R2),
CKA-based so architecture-agnostic (R1). *Weakness to test:* "non-redundant"
≠ "interesting" — a layer can be geometrically unique yet carry nothing
SAE-worth-training-on. Again, §6.1.1-E decides.

---

**§6.1.1-E — The bake-off: how the winner is chosen (this is the deliverable, per §2.2).**

The ideas above are hypotheses. None is adopted until this experiment runs.
The design mirrors `CLAUDE.md` §6.7's exploration-vs-confirmation discipline:
build an **independent gold ranking of layer interestingness**, then measure
each cheap selector against it — so no selector grades its own homework.

- **Two-stage protocol the winner plugs into.** Stage A (screen, cheap, stride 1,
  ALL layers): compute the chosen idea's signal → a candidate set ~2–3× the
  final budget. Stage B (confirm, expensive, candidates only): run
  lens/L3/SAE only on survivors → final top-k. This is what fixes §6.1's
  backwards dependency (R4) and makes R2 affordable.
- **The gold ranking (run once, expensively, and only once).** Train the
  baseline TopK SAE (§6.2) on **every** layer (stride 1) for one or two models
  on a small corpus, and score each layer by its **ground-truth
  feature-alignment mass** (`sae/ground_truth.py` — count × strength of
  features matching a ground-truth factor, §6.2's clearest positive result) and,
  as a second gold, per-window causal restoration (L3 patching). This is the
  operational definition of "worth interpreting," and it is deliberately *not*
  an input to any cheap selector.
- **Scoring each selector:** (i) rank-correlation and recall@k of its selected
  set against the gold ranking; (ii) the **parsimony curve** — fraction of total
  gold interestingness captured vs. number of layers selected — which is the
  "not too many" axis made quantitative (the winner is the method whose curve
  rises fastest).
- **Negative controls, written down before running (§2.2).** Uniform stride-k,
  random-k, and the all-layers upper bound. *A selector that does not beat
  uniform stride at equal k is not worth its complexity* — this null is the bar,
  stated up front so a plausible-looking method can't be rationalized past it.
- **Cross-architecture robustness gate (the test §6.1 fails).** The winner must
  pick sensible layers for **TimesFM and Chronos *individually***, not just
  pooled. A method that only works in aggregate is disqualified as a selector
  (R1). This is a hard gate, not a tiebreaker.
- **Deliverable shape once a winner exists:** a `layer_screen:
  {method: work_bend|factor_emergence|coverage, budget: N, stride: 1}` config
  block feeding a `select_layers()` that `sae.targets: auto` consumes; §6.1's
  `recommend_layers` stays as the cross-model *finding*, renamed/redocumented so
  the two are never confused. **Landed 2026-08-05 (see the production-wiring
  Findings block below):** `layer_screen: {enabled, method, budget_frac,
  min_budget, max_series, use_curvature, seed}` in `config.py`, a
  `layer_screen` pipeline stage (`pipeline.py`, runs right after `extract`,
  before every expensive stage), and `sae/train.py::_default_targets`
  consuming its `selection.json` whenever `sae.targets` is left empty.

**Recommended order to test (cheapest-informative first):** Idea A (pure
geometry, no probes, reuses L1 machinery) → Idea C (one CKA matrix, and it
doubles as the crosscoder's shared-layer picker) → Idea B (most principled and
most ground-truth-honest, but needs the per-factor probe battery built). Run all
three through §6.1.1-E rather than picking a favourite on intuition — the whole
reason this section exists is that the last "plausible-sounding" layer selector
(§6.1) didn't survive contact with a single real model.

**Findings — first bake-off run (2026-08-05).** Ran the whole §6.1.1-E
protocol for real, against live checkpoints, not mocks:
`configs/layer_screen_experiment.yaml` extracts `google/timesfm-2.5-200m-pytorch`
(TimesFM) and `amazon/chronos-t5-small` at `capture_layer_stride: 1` on
**both** models — the first time this repo has ever captured TimesFM's full
20 layers rather than the usual stride-2 10 — against 220 series of
`benchmark_medium/public_dev` (188 `random_parametric` + 32 `parametric`,
both ground-truth-bearing tiers). `--check-alignment`-equivalent diagonal-hit
fraction was a perfect 1.00 for both models before trusting anything
downstream (invariant 7). The gold reference
(`layer_screen_bakeoff.build_gold_ranking`) trained one small TopK SAE per
layer — all 26 (20 + 6), not a subset — scored by ground-truth
feature-alignment mass (`sae/ground_truth.py`, reused not reinvented);
L3 sensitivity-fingerprint magnitude (patching disabled for cost) served as
a cheaper secondary/cross-check gold. `run_layer_screen_bakeoff.py --config
configs/layer_screen_experiment.yaml` reproduces this from the committed
config against the already-extracted store in well under two minutes per
seed on one A5000.

- **Stability check passed before trusting anything else.** Per `CLAUDE.md`
  §2.4 ("verify empirically; distrust your first instinct"), re-ran the
  entire gold-ranking + scoring pass with an independent SAE training seed
  before reading any result as real. Gold-score Spearman agreement between
  the two seeds: **ρ = 1.0 (Chronos-T5-Small), ρ = 0.926 (TimesFM)** — every
  method's recall@budget and both null-beating verdicts were bit-for-bit
  identical across the reruns. The numbers below are from the first seed;
  the replicate is not a coincidence.
- **On Chronos-T5-Small (6 layers, budget=2 ≈ 25%): Idea A (work_bend)
  essentially matches the oracle.** Gold mass concentrates in the encoder's
  second half (`block.4`=106, `block.5`=98, `block.3`=83, descending to
  `block.0`=34). `work_bend`'s parsimony curve `[0.25, 0.48, 0.63, 0.82,
  0.92, 1.0]` is nearly indistinguishable from the oracle's `[0.25, 0.48,
  0.68, 0.82, 0.92, 1.0]` at every budget — **recall@budget = 1.00**,
  clearing both the uniform-stride null (`[0.08, 0.31, ...]`) and the random
  null (`[0.18, 0.35, ...]`) with room to spare. `coverage` (Idea C) is
  close behind (curve `[0.20, 0.43, 0.52, 0.61, 0.75, 1.0]`,
  recall@budget=0.50), also clearing both nulls. **`factor_emergence`
  (Idea B) is at or below the random null** (curve `[0.20, 0.28, ...]` vs.
  random's `[0.18, 0.35, ...]`) — recall@budget=**0.00**, failing both nulls.
  This is the clean, unambiguous win case: real headroom over the nulls
  existed, and one cheap geometric method (A) captured almost all of it.
- **On TimesFM (20 layers, budget=5 ≈ 25%): no selector beats uniform
  stride — but neither does the oracle, and that's the real finding.**
  Gold mass's two highest layers are `stacked_xf.0` (93.7, the very first
  captured layer) and `stacked_xf.19` (81.4, the very last) — an edge
  effect, plausibly tied to TimesFM's decoder-only architecture (the first
  block sits right after patch embedding, the last right before the
  read-out head), with the middle depths comparatively flat and close
  together (25–69). Checked the **full** parsimony curve, not just the
  budget point: oracle `[0.09, 0.17, 0.23, 0.30, 0.35, 0.41, ...]` tracks
  uniform-stride `[0.09, 0.17, 0.23, 0.27, 0.32, 0.33, ...]` almost exactly
  across *every* depth, because `linspace`-spaced stride picks already land
  near layer 0 and layer 19 by construction. All three selectors clear the
  random null (`[0.05, 0.10, 0.15, 0.19, 0.24, ...]`) comfortably but **none
  clears uniform-stride**: `work_bend` recall@budget=0.40, `coverage`
  recall@budget=0.40, `factor_emergence` recall@budget=0.20 (weakest of the
  three here too). Read plainly: this is **not** §6.1's failure mode
  (`recommend_layers` provably picked the *worst* layer on a real model) —
  here the *theoretical best possible selector* has almost no room to beat
  a free null on this particular model/corpus, so a real selector failing
  to beat it either is close to the ceiling, not a broken method. Whether
  this "TimesFM's interesting layers are its edges" pattern is a property of
  TimesFM specifically, of this corpus, or of the SAE-alignment-mass gold
  metric is not yet disentangled — flagged as follow-up, not asserted.
- **A genuine, diagnosed design weakness in Idea B, not a code bug.**
  Inspecting `factor_emergence`'s own `emergence`/`peak` breakdown
  (`select_factor_emergence`'s `components`) explains its underperformance:
  many of the ~18–20 kept ground-truth factors are only ever weakly
  decodable (their own peak R² is low), and `emergence`'s threshold is
  *relative to each factor's own peak* — so a weak factor's "emergence"
  layer is often just an early layer where noise first crosses a low
  relative bar, not a real signal. On TimesFM, 8 of 20 kept factors have
  their nominal "emergence" at layers 2–4 even though the *informative*
  peaks (the layers with real, higher R²) cluster at layers 9–16 — the
  count-based combined score gets diluted by noisy early hits before the
  real peaks can dominate it. **Follow-up, not fixed this session** (§2.5
  scope discipline: diagnosing the mechanism is itself the useful
  deliverable here): weight each factor's contribution by its own peak R²
  (or gate `emergence` behind an absolute floor, not just a relative one)
  before trusting Idea B again.
- **Combining did not beat the best single method on either model — a real,
  tested "no" to the "maybe combined" question.** `ensemble_vote`
  (≥2-of-3 agreement) and `ensemble_rank_average` both landed at
  recall@budget=0.50 on Chronos — *worse* than `work_bend` alone (1.00) — and
  at 0.40 on TimesFM, tied with (not better than) the best individual
  methods there. `ensemble_union` did no better. Read plainly: in this
  experiment, agreement between methods was not a useful proxy for
  correctness, and picking the single best-performing method beat every
  combination tried. This doesn't rule out ensembling being useful as a
  qualitative cross-check (e.g. "a layer 2 of 3 methods agree on" as a
  lower-stakes flag for a human or a downstream stage to weight more), but
  it is **not** a substitute for identifying which method is actually best
  on a given model, and should not be read as "combine them and skip the
  comparison."
- **The gold reference itself is more trustworthy for Chronos than for
  TimesFM.** Spearman agreement between the primary (SAE ground-truth-mass)
  and secondary (L3 sensitivity-fingerprint) golds: **ρ=0.714 for
  Chronos-T5-Small**, a reassuring cross-check that the SAE-based gold is
  tracking something causally real, not an SAE-training idiosyncrasy —
  versus **ρ=0.14–0.20 for TimesFM** (seed-dependent) — the two gold signals
  agree far less there. This weakens confidence in the TimesFM-specific
  conclusions above (including the "edges are gold-interesting" finding)
  more than it weakens the Chronos conclusions, and is a concrete argument
  for swapping in the design's originally-preferred secondary gold
  (per-window L3 *patching* restoration, disabled in this run for cost)
  before treating the TimesFM result as final.
- **Net recommendation (provisional, one corpus, one checkpoint pair):
  Idea A (`work_bend`) is the current best default** — cheapest to compute
  (pure CKA/RDM linear algebra over already-stored activations, no ground
  truth or probes, the strongest R1 architecture-agnosticism of the three),
  matched the oracle almost exactly on the one model where any selector had
  real headroom to beat the nulls, and tied for best-of-three on the model
  where headroom was scarce. **Idea C (`coverage`) is a good, cheap second
  opinion** — same cost profile, a genuinely different objective (pure
  representational coverage vs. change+bend), so agreement between A and C
  is a meaningful (if here, untested-as-a-lift) cross-check and disagreement
  would be worth a closer look. **Idea B (`factor_emergence`) is not
  recommended as currently designed** pending the peak-weighting fix above.
  **Not yet promoted to a config knob or `sae.targets: auto`**: this is a
  single (corpus, checkpoint-pair) data point, exactly the "n=1" caveat
  §6.1 itself had to learn the hard way applies to any cross-architecture
  generalization claim in this repo. Before trusting this further: (a) a
  second Chronos size (base/large) and ideally a non-Chronos/non-TimesFM
  third architecture, mirroring how §5.3's base-vs-small run de-confounded
  §6.1's own pooled study; (b) the per-window-patching secondary gold
  instead of the cheaper sensitivity-only proxy, given TimesFM's weak
  cross-check agreement above; (c) the Idea B peak-weighting fix, re-tested.
- Artifacts: `runs/layer_screen_experiment/` (extraction + internals + L3
  sensitivity), `runs/layer_screen_bakeoff.json` (seed 0, the numbers above)
  and `runs/layer_screen_bakeoff_seed1.json` (the stability replicate) — run
  artifacts, not committed (`.gitignore`), regenerable via `run.py --config
  configs/layer_screen_experiment.yaml --stages extract,l0,internals,l3`
  then `run_layer_screen_bakeoff.py --config
  configs/layer_screen_experiment.yaml --sae-epochs 50 --sae-dict-mult 6
  --sae-k 24`.

**Findings — production wiring (2026-08-05, same-day follow-up).** Per
explicit user direction, promoted `work_bend` from "provisional bake-off
winner, not wired anywhere" to the pipeline's default layer selector, ahead
of every config's `sae` stage. What changed, concretely:

- **`config.py`**: new `LayerScreenConfig` (`enabled=True`,
  `method="work_bend"`, `budget_frac=0.25`, `min_budget=2`,
  `max_series=100_000`, `use_curvature=True`, `seed=None` → falls back to
  `run.seed`), added to `PipelineConfig` and `_NESTED` so it loads from YAML
  like every other stage config.
- **`pipeline.py`**: a `layer_screen` `Stage` inserted immediately after
  `extract` and before `l0`/`internals`/every other analysis stage —
  satisfies R4 (screen before the expensive stages, not after) by
  construction, since stage execution follows `_stages()`'s fixed list
  order regardless of which subset is selected. Enabled by default, so it
  runs automatically in every full pipeline run without any YAML change;
  no hard DAG dependency was added from `sae` onto it (a config that
  disables `layer_screen` while giving `sae` explicit targets must keep
  working, not hard-fail).
- **`analysis/layer_screen.py`**: new `run_layer_screen(cfg, store, data,
  device)` stage entry point + `_uniform_fallback` (evenly-spaced indices —
  the cheap null itself, reused as the last-resort degrade path). Degrades
  gracefully and *loudly* (§2.5) in two independent ways: `method:
  factor_emergence` on a non-sealed or path-less corpus (no ground truth to
  probe against) logs and falls back to `work_bend` for that run; a
  selector that raises on a real model's activations logs and falls back to
  `_uniform_fallback` rather than crashing the whole pipeline over one
  model's screen. Writes `layer_screen/selection.json` (one entry per
  model: method, per-layer scores, selected layers/indices).
- **`sae/train.py::_default_targets`**: rewritten to consume
  `layer_screen/selection.json` when `sae.targets` is left empty — every
  layer a model's screen selected becomes its own SAE target, replacing the
  old "each model's final captured layer" default (itself as arbitrary a
  rule as anything in §6.1's disqualified `recommend_layers`, just never
  called out as such because nothing better existed yet). Falls back to the
  old final-layer rule, with a `log.warning`, only if the artifact is
  missing (stage disabled, or a `--stages sae`-only rerun skipped it) —
  never a silent behavior change.
- **`report/report.py`**: a new "Screen" section (between "L0" and
  "Profile", matching pipeline order) bar-charts each model's per-layer
  score with selected layers highlighted, and adds one `findings` line per
  model naming what was selected out of how many captured layers. Carries
  its own `_note()` stating plainly that the bar height is not comparable
  across models/methods and that the default method is a single-bake-off
  result, not a settled rule — the report should never imply more
  confidence in this section than the Findings above actually support.
- **Configs**: `default.yaml`, `medium_run.yaml` add an explicit
  `layer_screen: {enabled: true, method: work_bend, budget_frac: 0.25,
  min_budget: 2}` block (both already had `sae.enabled: false`, so this is
  purely informational there today — the report shows what *would* be
  trained on if SAE were turned on). `layer_screen_experiment.yaml` gets the
  same block with a note distinguishing it from
  `run_layer_screen_bakeoff.py`'s separate, more expensive three-selector
  comparison. `medium_run_chronos_base.yaml` — the one config with `sae`
  actually enabled — deliberately keeps its **explicit, hand-picked**
  `sae.targets` (the exact layers `§5.3`/`§6.2`'s Findings already cite
  numbers for) rather than switching to `auto`, so re-running that exact
  config still reproduces the recorded result; a comment there tells a
  future session to clear `targets: []` instead of copying those picks if
  they want the new default selector.
- **Tests**: `tests/test_smoke.py` — added `layer_screen/selection.json` to
  the expected-artifacts list, `"Layer screening"` to the report-token
  checks, and a new `test_layer_screen_stage_and_sae_auto_targets` that
  asserts the stage ran with `method="work_bend"` on both mock models and
  that `_default_targets` resolves *exactly* the layers `selection.json`
  named — the actual wiring, not just that both halves work in isolation.
  Full suite: **34/34 passing** (33 prior + 1 new). Verified live via the
  actual CLI (`run.py --config configs/smoke.yaml`), not only pytest's
  tmp-dir harness: report now has **11 sections / 22 findings** (was 10/18)
  and `runs/smoke/layer_screen/selection.json` contains real, sane
  `work_bend` scores for both mock architectures. Re-ran `compileall` and
  the hardcoded-absolute-path grep (invariant 11) clean across every touched
  file.
- **Not done, and not requested this session:** fixing Idea B's diagnosed
  peak-weighting flaw (§11.18); testing a second Chronos size or a
  non-Chronos/non-TimesFM third architecture before trusting `work_bend` as
  a cross-architecture default beyond the one bake-off's (corpus,
  checkpoint-pair); swapping the bake-off's secondary gold from L3
  sensitivity to per-window L3 patching. These stay exactly the follow-ups
  the first Findings block already named — wiring the current best-known
  method into production does not retroactively resolve them, and the
  report's own note above says so.

**Findings — Idea B peak-weighting fix (2026-08-06).** Picked up the one
item from the list directly above that was cheap, well-scoped, and did not
need a live checkpoint/GPU rerun: `factor_emergence_scores`
(`tsfm_lens/analysis/layer_screen.py`) now weights each ground-truth
factor's contribution to the per-layer `count_score` by that factor's own
peak decodability (`col.max()`) instead of a flat `+1` per emergence/peak
event, and gates the `emergence` event specifically behind a new absolute
(not peak-relative) `min_peak_score` floor (default `0.15`) — a factor
whose own peak never clears it is now recorded in a new
`no_emergence_factors` list and can still register a (heavily downweighted)
`peak` event, but never an `emergence` one. This is exactly the fix
`CLAUDE.md` §11.18 named as the concrete next step, implemented as both
halves ("weight by peak R²" *and* "gate behind an absolute floor"), not a
choice between them. A new planted-data unit test
(`tests/test_layer_screen.py::test_factor_emergence_weak_factors_no_longer_dilute_strong_late_peak`)
constructs one factor with a real, high-magnitude, late peak (layer 6 of 8)
against four factors that are uniformly noisy and never clear ~0.11
decodability anywhere — under the old unweighted/ungated scoring the four
weak factors' spurious early "emergence" events (layers 0–2) would
out-vote the strong factor's single real event by raw count; the fix must
put the combined score's maximum back on the strong factor's layer 5/6
neighborhood and exclude every weak factor from `emergence` via
`no_emergence_factors`. Full `tsfm_lens` test suite re-run clean after:
**66/66 passing.** **Explicitly not done this session, per its own scope**
(no live GPU/checkpoint work was in scope for this fix): re-running the
real §6.1.1-E bake-off (`run_layer_screen_bakeoff.py` against live
TimesFM 2.5 / Chronos-T5-Small) to check whether `factor_emergence` now
actually clears the uniform-stride/random nulls it previously scored at or
below — the fix is verified against a synthetic planted answer, not yet
against the real gold ranking. That live re-verification is the natural
next step before `factor_emergence` could be reconsidered as anything more
than "no longer disqualified by a known bug," and before it could be
compared again against `work_bend`/`coverage` as a production candidate.

**Findings — the deferred live re-verification (2026-08-10) — negative
result: the fix doesn't clear the nulls, and regresses TimesFM.** Reran
`run_layer_screen_bakeoff.py` against `runs/layer_screen_experiment`'s
existing store (both models, all layers, stride 1 — unchanged from the
first bake-off, reused rather than re-extracted; confirmed byte-identical
via checksum before/after) for both seed 0 and the seed-1 replicate,
backing up the pre-fix `layer_screen_bakeoff.json`/`_seed1.json` files
before overwriting (`CLAUDE.md` §0's "correct in place, never silently
delete" discipline — both preserved as `*_pre_fix_backup.json`).

- **Chronos-T5-Small: completely unaffected.** `factor_emergence` selects
  the identical `[encoder.block.0, encoder.block.3]` and scores
  recall@budget=**0.00**, `beats_uniform=False`, `beats_random=False` in
  all four runs (both seeds, pre- and post-fix) — bit-for-bit the same
  failing selection as the original 2026-08-05 run. The fix did nothing
  here.
- **TimesFM: got measurably worse.** recall@budget dropped from **0.20**
  (pre-fix, one hit — `stacked_xf.6`) to **0.00** (post-fix, zero hits) in
  both seeds. `beats_random` was `False` before and stays `False` (no
  qualitative flip), but the quantitative regression is real and
  reproduced across both seeds, not a one-off.
- **`work_bend`/`coverage`: bit-identical pre- vs. post-fix**, both models,
  both seeds — confirms the fix stayed correctly scoped to
  `factor_emergence_scores` with no side effects elsewhere, which is the
  one part of this result that came out exactly as expected.
- **Ensembles got worse too, driven entirely by `factor_emergence`'s
  changed (not just unchanged-but-still-bad) picks.** On TimesFM,
  `vote_min2`/`rank_average` dropped from 0.40 (tied with the best single
  method, pre-fix) to 0.20 (worse than either `work_bend` or `coverage`
  alone, post-fix) — `factor_emergence`'s new picks actively pull the
  combination down rather than just failing to help it.
- **Root cause, isolated properly this time** — holding one run's actual
  stored per-factor decodability matrix fixed and rescoring it with the
  literal old (unweighted, ungated) formula vs. the new (peak-weighted,
  floor-gated) formula side by side, rather than comparing two separately
  SAE-trained (and therefore individually noisy) end-to-end runs: on real
  data, `no_emergence_factors` came back **empty every time** — the new
  absolute `min_peak_score=0.15` floor never once excluded a factor here,
  so that half of the fix is inert on this corpus. The other half — peak-
  weighting — actively backfires: this corpus/model pair's *highest*
  peak-R² factors (`has_intermittency`, `archetype_intermittent_bursts`,
  `archetype_trend_dominant`, peak weight ≈0.80–0.82) are near-input
  statistics that decode well starting at layer 0 and plateau early, so
  weighting up exactly these factors pulls the combined emergence score
  toward layer 0 — the single *worst* layer by the SAE-mass gold ranking
  (TimesFM block 0 mass 34.8 vs. the real best layer, block 4's 109.9).
  This is the **mirror image** of `CLAUDE.md` §11.18's original diagnosis
  and the synthetic test built for it (weak, noisy factors spuriously
  "emerging" early and diluting one strong factor's real *late* signal) —
  on real checkpoints it's the *strong* factors whose real, legitimate
  early decodability creates the early-layer bias, so a fix that weights
  by peak magnitude amplifies exactly the thing it was meant to correct.
  The synthetic unit test (`test_factor_emergence_weak_factors_no_longer_
  dilute_strong_late_peak`) is not wrong — it correctly tests the failure
  mode it was designed for — that failure mode simply isn't the one that
  dominates on this real corpus.
- Full `tsfm_lens` pytest suite: **203 passed, 0 failed** (matching the
  last recorded count; no production code changed this session, this was a
  verification-only rerun).
- **Verdict: `factor_emergence` remains not usable, now for a diagnosed
  structural reason rather than an open question.** `work_bend` — already
  the production default and untouched by any of this — is unaffected.
  A real fix would need to explicitly discount very-early emergence layers
  (an absolute depth floor, not only a magnitude floor) rather than reweight
  by peak decodability alone; not attempted this session, since it would be
  a second design iteration needing its own synthetic-test-first treatment
  per `CLAUDE.md` §2.4, not a quick follow-on to this verification pass.

**Findings — extending to a third architecture family (2026-08-10, same
day) — answers §6.1.1-E's own named next step, and surfaces a more
important, previously-uncontrolled-for finding along the way: the
bake-off's qualitative verdicts are not stable across independent
re-extractions.** Built `configs/layer_screen_experiment_v2.yaml` — the
original bake-off config plus Sundial (added as a fourth `ModelAdapter`
earlier this same day, §9's Findings) as a third model, all three at
`capture_layer_stride: 1`. `run_layer_screen_bakeoff.py`'s "for every model
in the run" loop worked exactly as documented for three models with no
code changes — `cfg.comparison_pair()` (used by L1/L2/L3, not the
bake-off) still only compares the first two configured models, its
existing warning fired as designed, and this only meant Sundial's
`gold_agreement` (the L3-sensitivity cross-check) came back `null`;
Sundial's primary SAE-based gold ranking and all three selectors' scores
are complete and unaffected.

- **Sundial's own leaderboard** (12 layers, budget=3; gold_score by layer
  index 0–11: `[83.00, 99.28, 122.57, 153.78, 159.85, 181.88, 197.33,
  197.84, 220.30, 214.99, 197.31, 203.54]` — a plateau across layers 6–11,
  not a sharp peak, capping how informative exact-index recall can be
  here): `work_bend` selected `[3, 7, 10]`, **beats both nulls**
  (`beats_uniform_stride=True, beats_random=True`); `coverage` selected
  `[10, 2, 7]`, also **beats both nulls**; `factor_emergence` selected
  `[0, 3, 5]` — front-loaded relative to gold's actual top layers — and
  **fails both nulls** (parsimony-curve AUC strictly below both uniform-
  stride and random at every budget checked). All three tie at
  `recall_at_budget=0.0` by the harsh exact-top-3-index metric (the
  plateau means several near-tied layers outrank the nominal top-3), but
  the AUC-based null comparison is the more meaningful signal per this
  same session's Sundial-specific gold-mass-shape observation, and by that
  signal `work_bend`/`coverage` clearly win and `factor_emergence` clearly
  loses — the same qualitative pattern as the original two-model bake-off.
- **`factor_emergence`'s diagnosed failure mode (this same day, above)
  reproduces cleanly on a third, structurally distinct architecture — if
  anything more pronounced.** Sundial's highest-weight factors are the
  same near-input "meta" fields (`tier_synthetic`, `generator_mixture`,
  etc., weight ≈0.95–0.97) and 6 of its top-7-weighted factors emerge at
  layer 0–1 despite their own peak decodability sitting much later (peaks
  at layers 5, 9, 9, 11 for several) — exactly the "strong factor,
  spurious early emergence, real peak later" pattern. `no_emergence_
  factors` correctly excludes the one genuinely weak factor
  (`n_anomalies`), matching the same behavior already seen on TimesFM/
  Chronos. **Conclusion: this failure mode is a property of the
  ground-truth factor table itself** (tier/generator identity is
  near-perfectly linearly readable from the earliest layer of *any*
  architecture, by construction of how the benchmark is built) **rather
  than anything specific to TimesFM's or Chronos's own internals** — it
  generalizes cleanly to a third, unrelated architecture family, which is
  stronger evidence for the root-cause diagnosis than either single-model
  finding alone was.
- **Net verdict on `work_bend` across all three architectures now on
  record: mixed, not a clean sweep — consistent with, not a new
  contradiction of, the original two-model finding's own already-mixed
  TimesFM result.** From this same v2 run: `work_bend` beats both nulls on
  Sundial; on TimesFM it beats random but fails uniform-stride
  (`recall=0.2`, matching the qualitative verdict already on record); on
  Chronos-T5-Small it beats uniform-stride but — in this fresh extraction
  only, see below — fails random. Three architectures, three different
  "which null does it fail, if any" answers. `work_bend` is still the best
  single method tried and remains the reasonable production default, but
  "beats both nulls cleanly on every architecture" was never actually true
  even before this session — the original bake-off's own TimesFM result
  already said so; this extends the same honest picture to a third model
  rather than newly complicating it.
- 🔴 **The more important finding: a fresh, independent re-extraction of
  the "same" config changed Chronos-T5-Small's qualitative bake-off
  verdict, which no prior session had checked for.** This v2 run's
  Chronos-T5-Small numbers (`work_bend` recall=0.0, `beats_random=False`,
  `gold_agreement` ρ=0.429) do **not** match the existing, same-day
  `runs/layer_screen_bakeoff.json` numbers for the identical model at the
  identical config values (`work_bend` recall=1.0, `beats_random=True`,
  `gold_agreement` ρ=0.714) — both independently re-verified directly
  against their JSON artifacts, not taken on either session's word.
  TimesFM's qualitative verdict *did* reproduce (fails uniform, beats
  random, in both runs) — so this isn't a uniform instability, but it is a
  real one for at least one model. **This is a different axis of variance
  than anything previously checked**: the existing "seed1 stability
  replicate" (§6.1.1's first Findings block) only re-seeds SAE training on
  a *frozen, already-extracted* store, holding the activations themselves
  fixed — it was never a test of whether a fresh extraction (new forward
  passes, same corpus/config) reproduces the same gold ranking and
  therefore the same selector verdict. This run shows it does not,
  cleanly, for Chronos-T5-Small. **Practical implication, stated plainly
  rather than downplayed:** every single-run bake-off qualitative verdict
  on record in this file — including today's own "`factor_emergence`
  decisively fails, confirmed by a controlled same-data comparison"
  conclusion above — was measured on one extraction. The controlled
  same-data comparison used there is still valid evidence that the new
  formula is worse than the old one *on that one extraction's data*; what
  this new finding adds is that "that one extraction's data" is itself not
  guaranteed representative of what a different, equally-valid extraction
  of the same config would show. Not a reason to distrust any specific
  number retroactively without cause, but a real, previously-uncharacterized
  source of noise this bake-off's own documented CIs/nulls do not capture
  (they characterize sampling/bootstrap uncertainty *within* one
  extraction's SAE training, not extraction-to-extraction variance). Added
  to §13 as a new, prominent open item rather than folded quietly into an
  existing one.
- **Timing**: 3-model extraction (extract+l0+internals+l3, 220 series,
  all-layer capture) — 1m49.3s wall-clock. Bake-off (38 total per-layer
  SAEs across 3 models) — 2m37.3s wall-clock. Full `tsfm_lens` pytest
  suite: **207 passed** (independently re-run and confirmed, matching; no
  production code changed — this was a config + live-run addition only).
- Artifacts: `configs/layer_screen_experiment_v2.yaml`,
  `runs/layer_screen_experiment_v2/` (extraction store),
  `runs/layer_screen_bakeoff_v2.json` (bake-off output) — all new, nothing
  overwritten.
- ✅ **Root-caused 2026-08-10, same day, follow-up session — the 🔴 finding
  immediately above is not unexplained variance, it is `§15 A4`'s sampling
  fix.** `data.max_series: 220` against the 288-row
  `benchmark_medium/public_dev` corpus used to be applied as a bare head
  slice (`kept[:max_series]`); `§15 A4` (fixed 2026-08-06, i.e. *between*
  the 08-05 run that produced `layer_screen_bakeoff.json` and this same
  day's 08-10 rerun) replaced it with a family-stratified `sample_rows`
  call. Verified directly: the full corpus is `{random_parametric: 188,
  mixture: 60, parametric: 40}`; the first 220 rows (the old head slice)
  are `{random_parametric: 188, parametric: 32}` — **the entire 60-row
  `mixture` family, and 8 of 40 `parametric` rows, were silently absent
  from every number `layer_screen_bakeoff.json` (08-05) ever reported**,
  because `tsfm_benchmark` writes corpora grouped by generator and
  `mixture` sorts last. The three 08-10 reruns (`_v2`/a fresh `_v3`/a
  2-model-only `retest2model`, confirmed bit-identical to each other down
  to the raw stored activations via `np.array_equal`) all correctly sample
  a `mixture`-inclusive 220-row subset instead — same seed, same
  deterministic algorithm, different (more representative) population than
  08-05 saw. This fully explains the Chronos-T5-Small verdict flip with no
  need for a GPU-nondeterminism, SAE-seed, or model-state mechanism (all
  three were checked directly in code and ruled out — see `CLAUDE.md`
  §11.24 and `ROADMAP.md` §13's now-resolved entry near line 4038 for the
  full investigation). **Practical upshot, superseding the "practical
  implication" paragraph above:** `layer_screen_bakeoff.json` (08-05) was
  never a valid same-population replicate of the 08-10 runs — it was
  measuring a `mixture`-family-blind sample, a concrete instance of
  exactly the risk `§15 A4`/the open item at ROADMAP.md line ~4085 already
  named in the abstract. Treat the 08-10 3-model numbers as the current
  best answer for this experiment, not as one of two equally-valid samples
  to average over. The narrower question of whether *this* (correctly
  sampled) result is itself stable under a second seed remains open, but
  the "is the whole methodology unreliable" framing above is retracted.
- 🔴 **The second-seed test is now done (2026-08-10, same day, third
  follow-up) — answer: no, not reliably, especially for `work_bend` (the
  production default).** Built `configs/layer_screen_experiment_v2_seed1.yaml`
  (identical to `_v2.yaml` except `run.seed: 1`), extracted fresh, and ran
  the bake-off (`runs/layer_screen_bakeoff_v2_seed1.json`) — independently
  re-verified against the raw JSON/`meta.parquet` directly, not taken on the
  delegating agent's word, and both matched exactly. **Row overlap between
  the two seeds' 220-of-288 draws: intersection 169, union 271, Jaccard
  0.6236162361623616 — 76.8% of each seed's sample reappears in the
  other's, family composition identical (`{random_parametric: 144,
  mixture: 46, parametric: 30}` both).** Despite that substantial overlap,
  qualitative verdicts moved:
  - **`work_bend` (the wired production default) flips `beats_random` on 2
    of 3 models.** TimesFM: True (seed0) → **False** (seed1), `recall_at_
    budget` unchanged at exactly `0.2` both times — the flip comes entirely
    from the random-null curve shifting under the reseed, not from the
    selector's own choice (selected layers `[xf.2,6,10,15,18]` are
    bit-identical both seeds). Chronos-T5-Small: False → **True**,
    `recall_at_budget` moving `0.0→0.5` — again with the selector's own
    choice unchanged (`[block.2, block.4]` both seeds); what moved is the
    *gold* ranking itself (top-2 by SAE mass: `[block.3,5]` seed0 vs.
    `[block.4,5]` seed1 — one slot swaps, enough to change whether
    `work_bend`'s fixed pick overlaps it). Sundial alone is fully stable —
    every number and every selected layer bit-identical across seeds.
  - **`coverage` is fully stable** on all three models (`beats_random`/
    `beats_uniform_stride` identical both seeds, selected layers
    bit-identical too).
  - **`factor_emergence`'s `beats_random` is stable (still fails on
    Chronos/Sundial, still beats on TimesFM) but `beats_uniform_stride`
    flips for Chronos** (True→False) and its selected layers change on both
    Chronos and Sundial (recall moving 0.5→0.0 and 0.0→0.333 respectively).
  - Gold-agreement ρ itself moved substantially for both models with one
    (Chronos 0.429→0.543, TimesFM 0.466→0.665), consistent with a ~23%
    different row set producing a meaningfully different SAE-mass ranking.
  **This does not reopen the just-closed A4 question — the mechanism there
  (a `mixture`-family-blind head slice) is fully confirmed and distinct from
  this.** What it establishes is a *second*, independent source of
  fragility: even under the corrected, unbiased, family-stratified sampler,
  a `recall_at_budget` statistic computed at budget=2 (Chronos) or budget=5
  (TimesFM) against a 220-row draw is high-variance enough that a routine
  reseed — not an adversarial or biased one — flips which null the
  production-default selector beats on 2 of 3 architectures. Extends
  (doesn't supersede) `CLAUDE.md` §11.23's still-open item: that finding was
  about extraction-to-extraction variance from a fresh re-extraction; this
  one shows comparable-magnitude instability purely from row-sampling
  variance at fixed extraction machinery. Independently confirmed the test
  suite is unaffected: **207 passed, 2 warnings**, cross-checked by both the
  delegating agent's own run (417.01s) and a separately-launched run in
  this same session. **Practical implication for the production
  `layer_screen` stage:** `work_bend`'s selection of *which layers* is far
  more stable than its *beats-null verdict* — the same 5 (TimesFM) / 2
  (Chronos) layers get picked regardless of seed, it's only the
  null-comparison scorecard that's noisy at this sample size. That's
  arguably the more actionable reading: the selector's actual behavior
  (what it picks) looks trustworthy; the bake-off's own evaluation
  methodology (is that pick *provably* better than chance, at n=220,
  budget≤5) does not yet have the statistical power to answer that
  reliably in a single run. A fix would need either a much larger corpus,
  a bootstrap/multi-seed CI on `recall_at_budget` itself (rather than one
  point estimate), or both — not attempted this session.
- ✅ **Fixed and re-verified against real checkpoints (2026-08-10, same day,
  fourth follow-up) — the instability above is closed, not just documented.**
  User instruction was explicit: fix it, don't just record it. Root cause,
  isolated by re-reading `layer_screen_bakeoff.py`/`layer_screen.py`
  directly rather than guessing: `select_work_bend`/`select_coverage` take
  no `seed` argument at all (confirmed by grep) and are fully deterministic
  given a fixed corpus — exactly consistent with the observation two
  paragraphs up that their *selected layers* never moved across seeds. The
  entire flip traced to `build_gold_ranking` (`analysis/layer_screen_bakeoff.py`),
  which trained **exactly one** stochastic small SAE per layer as the "gold"
  reference every selector is scored against — random dictionary init,
  minibatch shuffling, and dead-neuron resampling all vary run to run, and
  at this bake-off's small budgets (2-5 layers) a modest rank swap between
  two adjacent layers' mass is enough to change which layers count as "gold
  top-budget," flipping the boolean verdict outright even though nothing
  about the selector itself changed. Two-part fix, both landed:
  1. `build_gold_ranking` now trains `n_replicates` (default 3, `run_
     layer_screen_bakeoff.py --n-gold-replicates`) independently-seeded SAEs
     per layer and **averages** the ground-truth-alignment mass instead of
     trusting one run — the standard variance-reduction-by-replication fix,
     matching how every other statistic in this repo is already treated
     (`CLAUDE.md` §6.6). Returns `gold_score` (mean) and `gold_score_std`
     (per-layer spread across replicates) so a layer whose replicates
     disagree wildly stays visible. (A related small gap the verification
     run itself caught: `run_bakeoff_for_model` computed but never saved
     `gold_score_std`/`n_gold_replicates` into the output JSON — fixed in
     the same edit, `run_layer_screen_bakeoff.py`.)
  2. New `configs/layer_screen_experiment_v4.yaml` / `_v4_seed1.yaml`: the
     corpus-subsampling half of the original noise source (§11.24's
     A4-adjacent finding) is removed outright by omitting `data.max_series`/
     `l3.max_series` entirely — `benchmark_medium/public_dev` is only 288
     rows, small enough that capping at 220 bought negligible compute
     savings while costing real determinism. Both configs now use the full,
     fixed corpus regardless of `run.seed`.
  New unit test `tests/test_layer_screen.py::
  test_build_gold_ranking_averages_replicates_and_is_deterministic` covers
  the averaging plumbing on synthetic data (CPU, no checkpoint needed).
  Full `tsfm_lens` suite green after the fix: **208/208** (207 prior + 1 new).

  **Re-verified live against all three real checkpoints** (TimesFM 2.5,
  Chronos-T5-Small, Sundial), fresh extractions on both seed 0 (`v4`) and
  seed 1 (`v4_seed1`), full 288-row corpus, 3-replicate gold ranking. Exact
  `beats_random` comparison, seed 0 → seed 1:

  | Model | work_bend | coverage | factor_emergence |
  |---|---|---|---|
  | Chronos-T5-Small | True → True | True → True | False → False |
  | Sundial | True → True | True → True | False → False |
  | TimesFM | False → False | True → True | True → True |

  **Every `beats_random` verdict now matches across seeds, for all three
  methods on all three architectures.** `work_bend` and `coverage` also
  kept bit-identical `selected_idx` across seeds on every model (as before
  the fix — consistent with the diagnosis that they were never the noise
  source). `factor_emergence`'s own selections still move across seeds
  (`select_factor_emergence`/`factor_probe_matrix` do take a `seed` kwarg,
  unlike the other two) but this no longer changes its `beats_random`
  verdict in this run. One smaller residual instability, noted rather than
  chased further this session: `factor_emergence`'s `beats_uniform_stride`
  on Chronos-T5-Small still flips (True→False) even with `beats_random`
  holding at False→False — the fix closes the specific instability it
  targeted (the production selector's null-comparison verdict) but is not
  a claim that every secondary metric in the bake-off is now seed-invariant.
  `factor_emergence` remains not the production default either way.
  Artifacts: `runs/layer_screen_bakeoff_v4.json`, `runs/
  layer_screen_bakeoff_v4_seed1.json`.

  **Bottom line:** the concern raised two entries up — that `work_bend`'s
  own trustworthiness verdict was a coin flip at this sample size — is
  resolved, not merely characterized. The fix did not touch the selector
  the production `layer_screen` stage actually runs (`work_bend`'s own
  layer-selection code path is untouched); it fixed the *evaluation
  harness* that decides whether to trust it, which is exactly where the
  noise was isolated to.

### 6.2 Phase 2b — A TSFM-native SAE variant (the flagship research thread) `[~]`

> ✅ **Baseline (item 4) DONE 2026-08-05.** ✅ **Crosscoder (item 1)
> feasibility test DONE 2026-08-05.** The flagship crosscoder build is
> §6.2.1 — read its triage block first.

**Goal.** Train sparse dictionaries on the layers Phase 2a identifies as
worth it, but don't just port a vanilla NLP-transformer SAE recipe
uncritically — time series activations have structure (periodicity,
multi-scale trend/noise separation, phase) that a generic ReLU/TopK SAE
ignores, and this repo has infrastructure (the alignment machinery,
ground-truth generators) that most SAE work doesn't. This is where "propose
something novel" (brief item 3) should actually land.

**Candidate directions — evaluate empirically, do not assume one is best.**

1. **Crosscoders for cross-model diffing (flagship candidate).** Rather than
   training two separate single-model SAEs and post-hoc matching their
   features (which is exactly the "still needs cross-model matching" problem
   `CLAUDE.md` §13 flags for the deferred plain-SAE plan), train a single
   dictionary **jointly across both models' aligned window representations**
   at (or near) the L1 peak-CKA layer pair. The alignment machinery
   (`extraction/alignment.py`) already puts both models' activations on the
   same `[series, window, dim]` axis — that's precisely the shared input a
   crosscoder needs, and this repo is unusually well set up to build one.
   A crosscoder directly operationalizes the repo's founding research
   question ("is there an intrinsic circuit across TSFMs") instead of
   inferring it indirectly through CKA/stitching correlations: a feature that
   fires jointly in both models' dictionaries *is* shared structure, by
   construction, not by post-hoc correlation. Decompose into
   shared-vs.-model-specific dictionary components the way cross-model
   diffing crosscoder work in the broader interpretability literature does,
   and report the shared/specific split as a direct answer to the founding
   question. This reuses `token_patch` for feature ablation with no new
   intervention primitive needed (`CLAUDE.md` §6.4, §13 item 3).
2. **Frequency-aware dictionary structure.** Time-series activations are not
   token embeddings; periodicity is a first-class structural property the
   benchmark itself labels exactly (`seasonalities`, ground truth periods).
   Consider initializing or regularizing some dictionary atoms to align with
   FFT-basis directions, or adding an auxiliary loss encouraging atoms to
   correspond to specific frequency bands. Test whether this improves
   ground-truth feature alignment (§6.3) over an unconstrained dictionary of
   the same size/sparsity — don't adopt it unless it measurably helps.
3. **Multi-resolution / matryoshka-style dictionary across alignment window
   scales.** The repo already pools activations at a configurable
   `alignment.window`; a hierarchical SAE that has coarse atoms for trend/
   regime-scale structure and fine atoms for local/noise-scale structure
   might map more naturally onto time series than a single flat dictionary
   sized for token-level NLP concepts.
4. **Plain per-model TopK/JumpReLU SAE as the baseline control.** However
   novel the above get, always keep a standard single-model SAE trained the
   conventional way as the comparison point — per §2.2, novelty has to beat
   a real baseline, not an implied one.

**Concrete deliverables**
- [x] Implement `SAEAdapter` (the existing `Protocol` in `sae/interface.py`)
  for at minimum the baseline (item 4) and the crosscoder (item 1); wire the
  encode-store pass into `sae/{model}/{layer}` per the seam already
  documented there. — **baseline done** (`sae/models.py::TopKSAE`);
  **crosscoder's prerequisite feasibility test now done too (same-day
  follow-up, see Findings below)** — `sae/crosscoder.py`'s `CrosscoderSAE`
  answers §13's stability question, but does not yet implement the
  `SAEAdapter` Protocol or a report-integrated cross-model diffing section,
  which is the remaining, larger scope of "the flagship crosscoder" as a
  research deliverable rather than a feasibility check. The encode-store
  pass (writing back into `sae/{model}/{layer}` so L1/clustering could read
  a `level="sae"`) is **also not wired** — `sae/interface.py`'s docstring
  now states this explicitly as a follow-up rather than implying it's done.
- [x] Build the SAE training loop as its own module (e.g.
  `tsfm_lens/sae/train.py`) reusing the zarr store's `level="window"` reads
  directly — do not re-extract activations, the store already holds them
  (`CLAUDE.md` §6.4). — done; `load_all_windows`/`train_sae`/`save_sae`/
  `load_sae_checkpoint`, plus `run_sae` as the pipeline-stage entry point
  (registered in `pipeline.py`, gated on `sae.enabled`, depends only on
  `extract`). `sae.interface.load_sae` is now a real implementation
  (delegates to `load_sae_checkpoint`), not the `NotImplementedError` stub
  it was.
- [x] Build an SAE evaluation harness covering: reconstruction fidelity
  (fraction variance explained) vs. sparsity (L0) frontier, dead-feature
  rate, and **forecast-preservation under reconstruction** (patch the
  reconstruction back in via `token_patch` and confirm MASE is close to
  clean — this is the validity check that ablation experiments in Phase 3
  will depend on). — done (`sae/eval.py`); the first run found it failing
  for both models (a real, useful negative result), and after fixing the
  baseline's dead-feature collapse (see the second Findings entry below)
  it now **passes for TimesFM** (ΔMASE +0.047) but **still fails for
  Chronos-T5-Base** (ΔMASE +2.37) — traced to an architecture-specific
  confound in the check itself (window-broadcast granularity), not
  necessarily the SAE, documented in `eval.py`'s docstring and the
  Findings below.
  > ⚠️ **AUDIT (2026-08-06) — these two ΔMASE numbers are confounded a second
  > time, independently of the window-broadcast issue, see §15 A4 and A13.**
  > `sae/eval.py:86-88` takes `data.contexts()[:take]` with `take ≤
  > adapter.cfg.batch_size` — a **head slice** of a corpus that
  > `build_pipeline` writes grouped task-by-task, so both numbers were
  > measured on the first ~16-32 series of `benchmark_medium`, plausibly a
  > single family, not a representative sample. And with no measured
  > repeat-run noise floor (A13), +0.047 cannot yet be distinguished from
  > run-to-run variation at all. Both numbers need re-measuring after A4/A13
  > land, **before** the "TimesFM passes, so Phase 3 ablation can build on
  > it" conclusion (§7 bullet 3) is acted on. The "vs. sparsity (L0)
  frontier" half is **partial**: TopK's sparsity is architectural (exactly
  `k` per row by construction), so there's no sparsity dial to sweep the
  way a ReLU+L1 SAE would need — a genuine frontier plot would need
  multiple runs at different `k`, not done this session, noted as
  follow-up rather than silently equated with "not applicable."
  > ✅ **RE-MEASURED (2026-08-06, same-day follow-up) — the deferred
  > re-run this AUDIT called for.** Re-ran `l0` (to get a matched noise
  > floor for this exact pair) and `sae` against `configs/
  > medium_run_chronos_base.yaml`'s already-extracted store, live
  > `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-base`, now
  > running through the fixed stratified `sample_rows` (A4) with a
  > matched `l0/noise_floor.json` (A13) available to interpret the result
  > against, both closed earlier the same day. **Corrected numbers,
  > superseding the ΔMASE/rho pair quoted just above (old values kept
  > verbatim there per §0.2, not deleted):**
  > - **Forecast-preservation** (the metric A4 directly affects — its
  >   `take=24 < n=288` triggers `sample_rows`'s stratified path instead of
  >   a head slice): TimesFM ΔMASE **+0.047 → +0.175**; Chronos-T5-Base
  >   ΔMASE **+2.368 → +3.869**. Both got *worse* under a representative
  >   sample, meaning the old head-slice prefix happened to be an
  >   easier-than-representative subset for this check on **both** models,
  >   not just a confound that happened to favor one of them.
  > - **Against the newly-available, matched noise floor**
  >   (`l0/noise_floor.json`, same run, same checkpoints): TimesFM is
  >   `deterministic: true` with a floor of exactly `0.0`, so its
  >   +0.175 is still 100% real signal by construction — but the earlier
  >   "passes almost exactly" framing (a +0.047 delta against a
  >   ~2.1-MASE clean baseline, ~2.3% relative) no longer holds verbatim;
  >   the corrected relative increase is ~9.3% (1.880 → 2.055), a real but
  >   noticeably larger gap than previously reported. Chronos-T5-Base's
  >   floor is `mean 0.160 / p95 0.705 / max 0.830` (family-dependent,
  >   `mixture` 0.009 to `random_parametric` 0.193 mean) — its corrected
  >   +3.869 delta is **~4.7x its own measured maximum noise floor**,
  >   closing the "maybe this is just Chronos's inherent sampling
  >   variance" question this AUDIT block left open: it is not, the
  >   failure is real signal by a wide margin. `report.html` renders both
  >   contextualized finding strings end to end (grep-confirmed): *"TimesFM
  >   ... ΔMASE +0.175 (this model is deterministic; the delta is real
  >   signal)"* / *"Chronos-T5-Base ... ΔMASE +3.869 (repeat-run floor
  >   ±0.160 — sec 15 A13 ...)"*.
  > - **Ground-truth alignment** moved too (TimesFM ρ 0.317 → 0.338,
  >   Chronos-T5-Base 0.306 → 0.401) but — checked directly rather than
  >   assumed (§2.4) — its own `rows` array is byte-identical
  >   (`arange(288)`, the corpus's full population) in both the old and
  >   new run, because `ground_truth_max_series: 2000` exceeds this
  >   corpus's 288 series, so `sample_rows`'s stratified path degenerates
  >   to "take everything" regardless of seed. This delta is **not**
  >   attributable to the A4 fix and should not be read as one — it
  >   reflects ordinary SAE-training run-to-run variance (real-data
  >   augmentation pull, dead-neuron resampling), the same class of
  >   noise A13 measures for forecasts but that this session did not
  >   separately quantify for SAE training itself.
  > - **Revised bottom line for §7 bullet 3**: the comparative claim
  >   ("TimesFM's SAE reconstruction preserves the forecast far better
  >   than Chronos-T5-Base's does") is unchanged and now rests on firmer
  >   ground (a representative sample plus a matched noise floor ruling out
  >   the "just noise" alternative for Chronos), so Phase 3's planned
  >   feature-level ablation can still reasonably build on TimesFM's SAE.
  >   But the specific magnitude previously quoted ("passes almost
  >   exactly, ΔMASE +0.047") should not be repeated — the corrected,
  >   representative-sample value is +0.175 (~9.3% relative), a real
  >   though comparatively small effect, not a near-zero one.
  > - **Scoping note**: only `l0`+`sae` were forced (not a full
  >   re-extraction), since the store was already valid and unaffected by
  >   either fix. `train_sae`'s per-target RNG (`torch.Generator().
  >   manual_seed(cfg.run.seed)`) is local to each call, independent of
  >   how much of the pipeline ran before it, which is consistent with
  >   what was observed: Chronos-T5-Base's `reconstruction_fidelity`/
  >   `dead_feature_rate` reproduced bit-for-bit against the 2026-08-05
  >   run despite the different stage subset, while TimesFM's shifted by
  >   ~0.007/0.005 — small enough to be ordinary GPU floating-point
  >   nondeterminism between runs rather than anything RNG-state-related,
  >   but not chased further since it doesn't change any conclusion above.
  > - **Verification.** Full `tsfm_lens` suite green at 173/173 both
  >   immediately before (baseline check) and after this re-run.
  > ✅ **RE-MEASURED AGAIN (2026-08-10, third data point) — §13's open item
  > asking whether the 08-06 numbers themselves were stable.** Reran
  > `sae,report` against the same already-valid `medium_run_chronos_base`
  > store (no re-extraction needed). **Chronos-T5-Base and both models'
  > ground-truth ρ replicate closely**, confirming those numbers are stable
  > across runs: Chronos ΔMASE **3.869 → 3.8686** (effectively identical);
  > ground-truth ρ TimesFM 0.338 → 0.3487, Chronos 0.401 → 0.4015 (both
  > small, ordinary-looking training-noise moves, consistent with the
  > 08-06 entry's own characterization). **TimesFM's forecast-preservation
  > ΔMASE did not replicate** — it moved **0.175 → 0.1097**, a ~37%
  > relative shift, meaningfully larger than the ~0.007
  > reconstruction-fidelity wobble the 08-06 entry above dismissed as
  > "ordinary GPU floating-point nondeterminism... not chased further."
  > Reconstruction fidelity itself stayed put this run (TimesFM 0.8575,
  > Chronos 0.8395 — both close to the "0.86/0.84" already on record
  > elsewhere in this section), so the instability is specifically in the
  > *forecast-preservation* metric, not in the SAE's own reconstruction
  > quality — i.e. `train_sae`'s per-target seeding reproduces a similar
  > dictionary each time, but which few hundred timesteps the `token_patch`
  > validity check happens to land on evidently matters more for TimesFM's
  > deterministic decoder than previously characterized. **New, real
  > finding, not yet closed**: TimesFM's own SAE forecast-preservation
  > ΔMASE has a repeat-run noise floor of its own that nothing currently
  > measures — analogous to A13's *model-level* MASE noise floor, but for
  > this specific SAE-validity check, and distinct from it. Until that
  > floor is measured (would need several repeat `sae` reruns with fixed
  > data/model, isolating the SAE-training RNG as the only varying input),
  > neither 0.175 nor 0.1097 should be quoted as *the* TimesFM
  > forecast-preservation number — read it as "small and real, exact
  > magnitude uncertain by roughly this much," which does not change
  > §6.2's qualitative bottom line (TimesFM's SAE preserves the forecast
  > far better than Chronos-T5-Base's does — Chronos's 3.87 dwarfs either
  > TimesFM value by more than an order of magnitude) but does mean the
  > specific "~9.3% relative" framing above is itself less precise than it
  > read. Ground-truth alignment permutation nulls (new context, not
  > previously recorded in this section for this pair) both cleared
  > cleanly: TimesFM 0.3487 vs. null mean 0.182/p95 0.191; Chronos 0.4015
  > vs. null mean 0.172/p95 0.178. Verification: the launching agent's own
  > post-rerun pytest reported **215 passed, 2 warnings** (same two
  > pre-existing benign warnings) — matching this session's own,
  > independently-run full-suite result from earlier the same turn (after
  > the unrelated §16 E11 change), a real cross-check rather than a single
  > unverified report, since this rerun touched only `runs/` artifacts and
  > no source code.
- [x] Implement the **ground-truth feature-alignment score** from §2.1/§6.3:
  for each learned feature, correlate its activation across the benchmark
  against every available ground-truth component (trend order, each
  seasonality's period/phase/amplitude, changepoint proximity, anomaly
  indicator, AR coefficients, noise-envelope depth, intermittency mask) and
  report the best match and its strength. This is this repo's distinctive
  answer to "is this feature interpretable" and should be a first-class
  output, not an afterthought. — **done, at series-level scalar
  granularity, not full per-window/per-timestep granularity** (see
  `sae/ground_truth.py`'s module docstring for exactly why: `GroundTruth`
  stores scalar recipe parameters, not a pooled-to-window decomposition,
  so a window-level version needing to pool `components`' per-timestep
  arrays itself is a real but separate follow-up, not built this session).
  Anomaly-proximity and changepoint-*location* matching specifically are
  **not implemented** — only `n_anomalies`/`n_changepoints` *counts* are,
  which is coarser than "proximity to a specific changepoint" the checklist
  names; flagged rather than silently claimed as the finer version. Found
  real, strong, ground-truth-verified structure on the first real run — see
  Findings.
- [x] Feed the SAE evaluation results back into §6.1's layer-selection
  correlation study (the retroactive validation step noted there). — **done
  2026-08-07**: new `sae/layer_sweep.py` (`run_layer_sweep`) trains and
  evaluates one lightweight `TopKSAE` per *captured* layer (not just the
  one or two targets a real `sae` pipeline stage run would pin), reusing
  `train_sae`/`load_all_windows`/`reconstruction_fidelity`/
  `dead_feature_rate`/`ground_truth_alignment` verbatim — deliberately
  CPU-only and skipping `forecast_preservation`, since ground-truth
  alignment needs no live model (only the store + sealed corpus + the
  freshly-trained SAE), so a full per-layer sweep costs nothing but CPU
  time, no GPU/model-loading required. New CLI `run_sae_layer_sweep.py`
  writes `<run>/sae_layer_sweep.json`. `analysis/layer_selection.py` grew a
  fourth proxy metric, `sae_ground_truth_rho` (added to `PROXY_METRICS`),
  populated from that JSON's `ground_truth_alignment.mean_abs_rho_matched`
  per (model, layer) when present and not itself an error/zero-matched
  degrade case. See Findings below for the real numbers and a flagged,
  not-fixed-this-session zarr-v3 incompatibility that limited which runs
  could be pooled.
- [x] Verbose-mode reporting (§4): a per-feature exemplar panel — top
  activating series/windows for a feature, its ground-truth alignment score,
  and (once Phase 3 feature ablation exists) its causal effect on forecast
  when zeroed. — **done 2026-08-06, series-level and forecast-effect-on-
  zero half explicitly deferred, not silently expanded to cover it**: a new
  "Sparse feature dictionary" report section (`report/report.py::_sec_sae`
  + new `report/sae_exemplars.py`) shows, per SAE target, the summary
  stats already computed (reconstruction fidelity, dead-feature rate,
  forecast-preservation ΔMASE) plus a table of the dictionary's top
  ground-truth-matched features and their top-activating exemplar series,
  including each exemplar's own real ground-truth value of the matched
  field (so a reader can eyeball whether the ρ number is a real pattern,
  not just trust it). Window-level exemplars (only series-level was built,
  matching `sae/ground_truth.py`'s own series-level-only scope) and the
  causal-ablation-when-zeroed column both remain explicit follow-ups —
  the latter is still correctly blocked on Phase 3 feature-level ablation,
  which doesn't exist yet. See this section's new Findings entry below.

**Findings / decisions**
- **2026-08-05 — baseline TopK SAE built and run against real checkpoint
  activations for the first time; genuine ground-truth-verified structure
  found, and a genuine validity-check failure that should block Phase 3
  feature-ablation work until fixed.** Ran against
  `runs/medium_run_chronos_base`'s already-extracted store (no new
  extraction, per the deliverable's own "reuse the store" instruction) —
  edited that config to add an `sae:` block (`dict_size_mult: 8, k: 32,
  epochs: 30`) and ran `python run.py --config
  configs/medium_run_chronos_base.yaml --stages sae`, which correctly
  skipped straight to the new stage since `extract` artifacts already
  existed. Targets: `TimesFM/stacked_xf.18` and
  `Chronos-T5-Base/encoder.block.6` — chosen directly from each model's own
  observed `tuned_r2_model` peak in that run (0.622 and 0.581
  respectively), **not** from Phase 2a's `recommend_layers` policy — see
  the next Finding for why that policy would have picked the *wrong* layer
  for Chronos-T5-Base here.
  - **Reconstruction fidelity: moderate.** Fraction of variance explained
    0.629 (TimesFM), 0.567 (Chronos-T5-Base) — a real baseline, not broken,
    but not strong either at only 30 epochs / ~4600 training rows against
    dictionaries this large (see next point).
  - **Dead-feature rate: ~98% for both models** (0.982 TimesFM, 0.983
    Chronos-T5-Base). `dict_size_mult: 8` against `k: 32` and only 4608
    training rows is real overcapacity — a well-known TopK-SAE pathology
    (most dictionary atoms never win top-k against a much smaller live
    population of "directions actually used") that this session's training
    loop does **not** correct: no dead-feature resampling/reinitialization
    was implemented, which the SAE literature treats as close to standard
    practice for exactly this failure mode. Stated as a real limitation of
    this baseline as configured, not silently accepted as fine — a smaller
    `dict_size_mult`, more training rows (a bigger corpus), or periodic
    dead-neuron resampling would all plausibly help, and none were tried
    this session.
  - **Forecast-preservation: fails the validity check, informatively.**
    Patching the SAE's reconstruction (broadcast from window-pooled
    granularity back to token positions, per `sae/eval.py`'s documented
    approximation) back into a clean forward pass roughly **doubled**
    MASE for both models (TimesFM 2.096 → 3.509; Chronos-T5-Base 2.837 →
    5.480) rather than leaving it "close to clean" as the checklist hoped.
    Read plainly: at this fidelity/dead-feature-rate, **this baseline
    dictionary is not yet a safe substrate for Phase 3's planned
    feature-ablation work** — an ablation delta measured against a
    reconstruction baseline that already breaks the forecast this much
    would be confounded by the SAE's own reconstruction error, not
    isolating a feature's causal contribution. This is exactly the
    "validity check ablation experiments will depend on" the checklist
    asked for, and it correctly caught a real problem before any Phase 3
    work was built on top of it — read as a success of the check, not a
    failure of the session. Note this also folds in window-pooling
    information loss (the broadcast is coarser than true per-token
    fidelity, stated explicitly in `eval.py`'s docstring), so some of this
    gap is inherent to the pooling granularity, not only the SAE.
  - **Ground-truth feature alignment: real, strong, verified signal — the
    session's clearest positive result.** Of ~184 (TimesFM) and ~104
    (Chronos) *alive* features (the ~2% surviving the dead-feature rate
    above), 62 and 61 respectively found a significant best-match ground
    truth field — i.e. roughly a third to over half of the features that
    actually do anything are ground-truth-interpretable to a meaningful
    degree. Top matches for both models are seasonality-related and
    strong: TimesFM's best feature matches `archetype_trend_dominant`
    (ρ=0.752, n=288) and several match `n_seasonalities` (ρ up to 0.546);
    Chronos-T5-Base's best matches `n_seasonalities` (ρ=0.783, n=288) and
    `seasonal_period_dominant` (ρ=−0.610, n=174). Mean |ρ| among matched
    features: 0.381 (TimesFM), 0.386 (Chronos-T5-Base). This is a genuine,
    first-time confirmation that §2.1's "ground-truth-verifiable
    interpretability is this repo's unfair advantage" claim pays off on a
    real trained SAE, not just as an aspiration — worth pursuing further
    (larger training set, fixing the dead-feature problem above) before
    concluding anything about *how* interpretable this dictionary is
    overall, since only the alive ~2% could be scored at all.
  - **A real, checked limitation of §6.1's `recommend_layers`: it does not
    transfer as a within-model ranking rule, and this run proves it, not
    just risks it.** Before picking targets, tried
    `recommend_layers(goal="forecast_readability")` on this exact run:
    for TimesFM it correctly picked `stacked_xf.18` (its actual best layer,
    tuned_r2=0.622). For Chronos-T5-Base it picked `encoder.block.0` —
    whose own `tuned_r2_model` (0.433) is the *lowest* of any of its 12
    layers, the literal opposite of the intended ranking. Cause, checked
    directly (`analysis/layer_selection.py::collect_layer_records`'s raw
    per-layer numbers for this run): within Chronos-T5-Base specifically,
    effective dimensionality and tuned-lens R² **rise together** from
    layer 0 through the mid-layers (eff-dim 3.42→13.93, R² 0.433→0.581)
    before both softening late — the *opposite* sign from §6.1's pooled,
    depth-controlled, cross-(run,model) finding. TimesFM's own layers, by
    contrast, actually do show the pooled study's negative relationship
    within-model (eff-dim 21.92→2.66, R² 0.498→0.622 from layer 6 onward).
    **Read plainly: the pooled cross-model relationship is real (§6.1's
    CIs are genuine), but it is driven more strongly by one architecture in
    this small population than the other, and does not reliably describe
    every individual model's own layers** — exactly the kind of thing that
    only surfaces by actually trying to use a proposed policy on a real
    case, not by re-deriving the pooled statistics. `recommend_layers`'s
    docstring (`analysis/layer_selection.py`) now states this limitation
    with these exact numbers so it isn't rediscovered as a surprise.
  - Test suite: 6 new unit tests (`tests/test_sae.py` — model shapes/
    sparsity, training convergence on synthetic low-rank data, checkpoint
    roundtrip, ground-truth matching logic with a planted correlation and
    a too-few-valid guard) plus one new integration test
    (`tests/test_smoke.py::test_sae_stage_integration`, mock adapters,
    confirms `forecast_preservation` works with no real corpus and
    `ground_truth_alignment` degrades to a logged, non-crashing `error`
    key when smoke data has none — real degrade-gracefully behavior, not
    assumed). Full suite after: 17/17 passing.
  - Also refactored `analysis/l0_behavioral.py::_score`'s inline MASE
    formula into a new shared `analysis/stats.py::mase()` (byte-identical
    behavior, re-verified via the smoke suite) so the forecast-preservation
    check calls the exact same metric the rest of the repo reports, not a
    reimplementation — a small, targeted refactor surfaced by actually
    trying to reuse the existing code rather than duplicating a 2-line
    formula.

- **2026-08-05 (same-day follow-up — fixing the baseline's two real
  problems, per explicit user direction).** After reporting the baseline's
  forecast-preservation failure and ~98% dead-feature rate, checked in with
  the user on how to proceed; chosen path was "fix the baseline first,"
  specifically dead-neuron resampling and pulling more SAE training data
  from an established source (user's own suggestion: "find an established
  library and just pull from it, like something on huggingface") rather
  than only shrinking the dictionary. Both landed, and **found two
  additional real bugs building the first one** — this took three
  iterations to get right, not one, each caught by actually measuring
  rather than assuming the fix worked (`CLAUDE.md` §2.4):
  1. **Dead-neuron resampling, attempt 1: catastrophic, not just
     ineffective.** First implementation reinitialized a dead atom's
     decoder column from a normalized reconstruction-residual direction and
     its encoder column at a **fixed absolute scale (0.2)**. Measured
     immediately after implementing (not assumed working): reconstruction
     fidelity went from a positive baseline to **deeply negative**
     (TimesFM: 0.617 → **−31.2**; Chronos-T5-Base: 0.506 → **−0.77**) —
     worse than predicting the mean. Root cause, found by checking what a
     standard implementation does differently: (a) a fixed absolute scale
     is miscalibrated for activation magnitudes that vary enormously across
     models/layers — fixed by scaling to `0.2 ×` the *alive* atoms' own
     average encoder-column norm instead; (b) Adam's per-parameter moment
     estimates for the resampled weight slices were never reset, so
     `opt.step()` immediately applied stale momentum (computed against the
     old, dead, near-zero-gradient weights) to a freshly meaningful
     direction — fixed by zeroing `exp_avg`/`exp_avg_sq` for every
     resampled slice.
  2. **Attempt 2, with both of those fixed: fixed for Chronos-T5-Base
     (fidelity 0.506 → 0.697), still catastrophic for TimesFM (still
     −19 to −32).** Measured again rather than declaring victory after one
     model improved. Cause: TimesFM's dictionary was both the largest
     (10240 atoms) and the most dead (~98%), so resampling **every** dead
     atom in one shot drew ~10,000 replacement directions from only
     `batch_size × 4 = 16,384` source rows via `torch.multinomial`
     weighted by a highly skewed per-row residual-loss distribution —
     thousands of atoms collapsed onto a handful of near-duplicate
     directions, and that many simultaneously-live, nearly-identical atoms
     competing for the same top-k slots destabilized training. Fixed by
     (a) capping resampling to `max_resample_frac=0.1` of the dictionary
     per event — spreading a large dead population across several
     resample events instead of one shock — and (b) jittering each sampled
     direction with a little noise before renormalizing, so atoms drawn
     from the same source row end up distinct rather than duplicated.
  3. **Attempt 3, both fixes combined: real, substantial, and stable
     improvement**, confirmed both in a fast controlled sweep against
     already-extracted activations (no model calls needed) and in a full
     real run. Sweep (dict_size_mult=8, k=32, resample every 5 epochs,
     30→60 epochs): fidelity rose to 0.86 (TimesFM) / 0.85
     (Chronos-T5-Base) at 60 epochs — a large, clean improvement over the
     no-resampling baseline at the same epoch count, and resampling the
     *large* dictionary now clearly beat shrinking it (`dict_size_mult=2`
     alone reached only ~0.54–0.61). A regression test
     (`tests/test_sae.py::test_dead_neuron_resampling_improves_not_destroys_reconstruction`)
     now guards against reintroducing either bug.
  - **Real-data augmentation, implemented and verified live, exactly as
    requested.** `tsfm_lens/sae/real_data.py` reuses
    `tsfm_benchmark.build_pipeline.sources.bootstrap_catalog` (the same
    generic HF-dataset loader §4.3/§5.2 already verified against Monash) to
    pull real series, slice random context-length windows from them, run
    them through the model, and window-pool the resulting activations —
    concatenated with the run's own benchmark activations for SAE
    training. These series never enter the leakage-audited benchmark, are
    never sealed, and never get ground-truth labels — purely SAE-training
    augmentation, governed by nothing invariant 10 restricts. Verified live
    against `Monash-University/monash_tsf` (`real_data_pool_limit: 500`):
    pooled 152 series across 12 working domains (the rest failing with the
    same pandas frequency-alias errors §12 already documents — confirmed
    again, not a new problem), of which only a fraction reach
    `context_len=512` — real, worth stating plainly: the realized
    diversity gain is bounded by how many long-enough real series exist in
    the pool, not by `real_data_n_windows`'s nominal target. Still added
    4000 real-data rows on top of the benchmark's own 4608 for this run.
    **Fixed 2026-08-06:** `run_sae` used to call `_real_data_activations`
    (which both sampled the real context pool *and* ran it through the
    model) once per target, so a multi-target run re-fetched and re-pooled
    the same catalog redundantly (this run fetched Monash twice, once per
    model) even though sampling is model-independent. `train.py` now calls
    a new `_sample_real_contexts(cfg)` once, before the per-target loop,
    and reuses the resulting `contexts` array in every target's
    `extract_real_activations` call (the model-specific half, which still
    correctly runs once per target since activations differ per model).
    Full `tsfm_lens` suite re-run clean after: 66/66.
  - **Final numbers on the real run, both fixes combined
    (`configs/medium_run_chronos_base.yaml`, now `epochs: 60,
    resample_dead_every_epochs: 5, real_data_enabled: true`):**
    reconstruction fidelity **0.858 (TimesFM), 0.840 (Chronos-T5-Base)** —
    up from the original baseline's 0.629/0.567, a large, real
    improvement, not a marginal one. Dead-feature rate only modestly
    improved (0.959/0.974, down from 0.982/0.983) — most of the
    dictionary is still unused, but the *alive* fraction now reconstructs
    far better, which is what actually mattered for the checks below.
  - **Forecast-preservation: a genuine, informative split result, not a
    uniform pass or fail.** TimesFM: MASE 2.096 (clean) → **2.143**
    (SAE-reconstructed), ΔMASE **+0.047** — the validity check now
    essentially **passes**; the reconstruction is close enough to clean
    that Phase 3 feature-ablation work could plausibly build on this
    target. Chronos-T5-Base: MASE 2.837 → **5.205**, ΔMASE **+2.37** —
    still fails, only marginally better than before (+2.64) despite a
    similar fidelity gain. Investigated why the two models diverged this
    much given comparable fidelity improvements (§2.4: measured, didn't
    assume "Chronos's SAE is just worse") and found a real, previously
    unstated confound in the *eval method itself*, not the SAE: the
    forecast-preservation check broadcasts one window-pooled reconstructed
    vector across every raw token in that window. For TimesFM,
    `alignment.window` (32) equals its own patch width, so each window is
    exactly one token — the broadcast is lossless. For Chronos, each token
    is one timestep, so a 32-step window covers **32 distinct tokens**
    that all get forced to the same single reconstructed vector — a large,
    architecture-specific information loss the check was always going to
    incur, independent of SAE quality. `sae/eval.py::forecast_preservation`'s
    docstring now states this explicitly with these exact numbers so
    Chronos's failing number isn't mistaken for "this SAE is bad" without
    the caveat. A finer, per-token (not per-window-broadcast) version of
    this check would resolve the confound for per-step-tokenized models —
    real follow-up, not attempted this session.
  - **Ground-truth alignment: richer than the pre-fix run, and one
    striking new clean match.** TimesFM: 154/10240 features matched (up
    from 62), mean |ρ| 0.317. Chronos-T5-Base: 115/6144 matched (up from
    61), mean |ρ| 0.306, and its single best match is now
    **`has_intermittency`, ρ=0.881** — one of the cleanest single-feature
    ground-truth matches found anywhere this session, a feature that
    nearly perfectly tracks whether a series has the intermittency
    ground-truth property. Both models keep matching seasonality-related
    fields prominently (`n_seasonalities`, `seasonal_amplitude_max`,
    `seasonal_period_dominant`, and for TimesFM specific archetype labels
    like `archetype_trend_dominant`/`archetype_multi_seasonal_complex`).
  - Full `tsfm_lens` test suite after all of this: 21/21 passing (up from
    17 — 3 new resampling/stability tests plus 1 new real-data test file
    with 3 tests, all mocking the network call per the same no-live-network
    discipline `test_smoke.py` follows).

- **2026-08-07 — SAE ground-truth alignment fed back into §6.1's
  layer-selection correlation study as a fourth proxy; real signal found,
  and it independently replicates the pattern `probe_decodability` already
  showed.** Built `sae/layer_sweep.py` + `run_sae_layer_sweep.py` (CPU-only
  per-layer TopK-SAE train+eval, see the checklist item above for the
  mechanism) and ran it against three run directories: `medium_run_chronos_base`
  (22 layers: real TimesFM + real Chronos-T5-Base), `null_timesfm_random`
  (20 layers: real TimesFM + its random-init twin), `null_chronos_random`
  (24 layers: real Chronos-T5-Base + its random-init twin) — 66 records, 6
  (run, model) groups total, clearing `cluster_bootstrap_spearman`'s
  `n_groups>=3` floor for the first time since this correlation study was
  built (a single-run demo previously gave only 2 groups; see the checklist
  item's prior "not done" text).
  - **Sanity check the sweep itself passes cleanly:** every random-init twin's
    `sae_ground_truth_rho` is flat and low across all its own layers
    (TimesFM-random: 0.192–0.200 at every one of 10 layers; Chronos-T5-Base-random:
    0.162–0.167 at every one of 12) — no depth structure at all, as expected
    for an untrained architecture — while both real models show real
    depth-varying structure well above their own random twin at every layer
    (TimesFM: 0.240→0.385 range; Chronos-T5-Base: 0.264→0.382 range). This is
    the same real-vs-random-twin separation §16 E9's null baseline already
    established for CKA/stitching/probe-decodability, now confirmed for the
    SAE ground-truth-alignment proxy too.
  - **Two attempted extra sources, ruled out and flagged rather than
    silently dropped:** `runs/medium_run` and `runs/real_run` (both dated
    2026-07-20) were tried as additional (run, model) groups to pool
    without new extraction, but both fail `zarr.open_group()` with
    `GroupNotFoundError` — confirmed via direct inspection that their
    `activations.zarr` directories contain a `zarr.json` file (zarr v3
    format) rather than `.zattrs`/`.zgroup` (v2), meaning both predate this
    repo's `zarr<3` pin (CLAUDE.md §11.15) and are unreadable by the
    currently-pinned zarr 2.18.7. **Not fixed this session** — would need a
    full GPU re-extraction, out of scope for this item; flagged here as a
    fixable-later note per this file's own no-silent-dropping doctrine.
    **Correction (2026-08-10, later the same day):** `medium_run` was
    re-extracted from scratch for an unrelated reason (the §5.3/A4
    re-measurement thread; `CLAUDE.md` §11.25) and now has a valid v2
    store — it could be added to this pool on a future pass. `real_run`'s
    store is unchanged and still the stale v3 format described above; not
    re-checked this session
    (§0.2), not silently worked around.
  - **The real correlation numbers**
    (`run_layer_selection_study(['runs/medium_run_chronos_base',
    'runs/null_timesfm_random', 'runs/null_chronos_random'])`, n_records=66,
    n_groups=6; `l3_entropy` correctly produced "not enough data" since `l3`
    is enabled in only one of the three runs — an expected degrade, not a
    bug):
    - `input_cka` vs `sae_ground_truth_rho`: raw ρ=0.584 (CI
      [−0.079,+0.915], **not** significant — spans zero), but
      **depth-controlled ρ=0.623 (CI [+0.074,+0.758], p<1/2000 —
      significant, excludes zero.** Layers whose activations already
      resemble the raw input (high `input_cka`) beyond what depth alone
      predicts also tend to train SAEs whose features align better with
      ground-truth structure beyond what depth alone predicts.
    - `input_cka` vs `probe_decodability` (one of the study's original
      three proxies): raw ρ=0.583 (not significant), depth-controlled
      ρ=0.580 (CI [+0.021,+0.800], significant) — **the same shape of
      result**, found independently by a completely different proxy metric
      (family-probe linear decodability vs. SAE ground-truth-feature
      alignment). Two unrelated interpretability measures agreeing this
      closely, after the same depth control, is a meaningfully stronger
      claim than either alone — convergent validity, not just one more
      correlation.
    - `effective_dim` vs `sae_ground_truth_rho`: raw ρ=−0.386, depth-controlled
      ρ=−0.382, **neither significant** (both CIs span zero:
      [−0.810,+0.376] raw, [−0.896,+0.207] depth-controlled) — no
      relationship established either way at this n_groups, unlike
      `effective_dim`'s significant (opposite-signed, per-model) relationship
      with `tuned_r2_model` documented earlier in §6.1's Findings; this
      pooled cross-model version of the effective-dim question remains
      genuinely unresolved, not contradicted.
    - `effective_dim` vs `probe_decodability`: raw ρ=−0.173, depth-controlled
      ρ=−0.059, neither significant — consistent with the `sae_ground_truth_rho`
      null result immediately above, i.e. `effective_dim` shows no clear
      pooled cross-model relationship with *either* interpretability proxy
      at this sample.
    - `depth_trend`: `probe_decodability` (ρ=0.321, CI [0.038,0.812]) and
      `sae_ground_truth_rho` (ρ=0.319, CI [0.015,0.742]) both trend
      significantly upward with relative depth pooled across these 6
      groups; `effective_dim` (ρ=0.025) and `input_cka` (ρ=−0.153) show no
      significant depth trend — this is exactly why the depth-controlled
      correlations above matter and aren't redundant with the raw ones.
  - Full `tsfm_lens` suite green at 198/198 after `sae/layer_sweep.py` +
    the `layer_selection.py` extension (3 new tests for the sweep itself,
    2 new tests for the proxy-ingestion degrade paths).

**Findings — per-feature exemplar panel / "Sparse feature dictionary" report
section (2026-08-06).** Built the checklist item above and verified it two
ways, per `CLAUDE.md` §2.4: synthetic planted-data unit tests first
(`tests/test_sae.py`'s three new tests — ranking-by-activation, a planted
feature that must recover the right exemplar series *and* the right real
`field_value` for each, and a features-with-no-match-must-be-skipped guard),
then regenerated the real report for `runs/medium_run_chronos_base` (the
already-existing real-checkpoint SAE run this section's earlier Findings
already cite numbers from) and read the actual rendered HTML rather than
trusting the unit tests alone.
- **The real report reproduces this section's own earlier-cited numbers
  exactly**, which is itself a useful consistency check: TimesFM's top
  exemplar table surfaces feature 2872 matching `archetype_trend_dominant`
  (ρ=0.650) and feature 7945 matching `n_seasonalities` (ρ=0.640); Chronos-
  T5-Base's top row is feature 2189 matching `has_intermittency` (ρ=0.881,
  shown rounded in-report as 0.88) — the exact "cleanest single-feature
  ground-truth match found anywhere this session" cited earlier in this
  block, now visible as an actual exemplar (series `2f89e898a893a0be`,
  activation 40.09).
- **A genuine, minor observation surfaced only by looking at real exemplar
  rows, not by reading the aggregate ρ number** — exactly the kind of thing
  this panel exists to catch: TimesFM's top exemplars for the
  `archetype_trend_dominant`-matched feature include two `mixture` (real-
  derived tier) series with activation 0.0 and `field_value` 0.0, sitting
  alongside three genuinely trend-dominant `random_parametric` series with
  activation ~8 and `field_value` 1.0. This is not a bug in the new panel —
  it faithfully reflects how `sae/ground_truth.py`'s existing archetype
  dummy-encoding already treats every series with no archetype label
  (i.e. every real-derived-tier series, which by design carries an empty
  `GroundTruth`, `CLAUDE.md` §4.1) as a valid `archetype_trend_dominant = 0`
  data point, rather than an excluded/`NaN` one — the same encoding
  `best_ground_truth_matches` already used to compute the ρ=0.650 this
  section cited before today. Not changed here: doing so would silently
  alter every previously-reported archetype-related ρ number in this file
  retroactively (§0.2's own "don't silently reorder/change" doctrine), and
  deciding whether real-derived series *should* count as negative examples
  for an archetype label is a real design question, not a bug fix, that a
  future session should make deliberately rather than as a side effect of
  a reporting feature.
- Only the series-level granularity was built (matching `ground_truth.py`'s
  own existing scope, `CLAUDE.md` §11's window-level version is a separate,
  unbuilt follow-up already noted there) and the causal-effect-when-zeroed
  column is not present — both stated explicitly in the section's own
  `_note()` in the report, not just here, so a reader of the report itself
  sees the same caveat.
- Full `tsfm_lens` suite after: **69/69 passing** (was 66 — 3 new unit
  tests, plus the existing `test_sae_stage_integration` extended to call
  `run_report` and assert the section degrades to an explicit
  "no ground-truth-matched features to illustrate" message rather than an
  empty table or a crash when smoke data has no sealed corpus to draw
  ground truth from, matching this stage's already-established
  degrade-gracefully behavior).

**Findings — crosscoder feasibility test (2026-08-05, same-day follow-up).**
Per §13's own prerequisite ("needs an early small-scale test before
committing significant compute"), built the smallest thing that could
answer the stability question, not the flagship crosscoder itself.

- **Mechanism** (`tsfm_lens/sae/crosscoder.py`): `CrosscoderSAE` generalizes
  `TopKSAE` to `n_sources` inputs of independent dimension — each source
  gets its own linear encoder/decoder, contributions sum before one shared
  bias + TopK, and decoder columns are normalized *jointly* per feature
  across sources (not independently per source) so `relative_decoder_norm`
  (0 = source-B-specific, 0.5 = shared, 1 = source-A-specific — the
  standard crosscoder cross-model-diffing metric) actually carries signal
  rather than being trivially washed out. `train_crosscoder` mirrors
  `sae/train.py::train_sae`'s loop (Adam, decoder renormalization every
  step, dead-atom resampling with the same two bug classes already fixed
  once in the single-source version — stale Adam moments, uncalibrated
  resample scale — ported rather than reintroduced).
- **A real instability found and fixed before it reached the real-checkpoint
  run, exactly per `CLAUDE.md` §2.4 (verify empirically, don't assume the
  first design is right).** A synthetic test with one source's activations
  at 20x the other's raw scale reproduced precisely the failure §13 named
  as the risk: source B's reconstruction fidelity collapsed to **−9.7**
  (worse than predicting the mean) while source A's stayed at 0.95, because
  summing raw per-source MSE lets whichever source has the larger absolute
  scale dominate the joint objective — a real risk for two independently-
  initialized model checkpoints, which have no reason to share an
  activation scale. **Fixed**, not worked around: `CrosscoderSAE` now takes
  a per-source `source_scale` (each source's own global std, computed once
  in `train_crosscoder`) and divides/multiplies by it inside
  `encode`/`decode`, so the dictionary is fit in a scale-normalized space
  internally while the public contract still takes/returns activations at
  their original scale, same as `TopKSAE`. The dead-neuron resampling
  routing (which source's residual is largest) was normalized the same
  way. Re-ran the mismatched-scale test after the fix: **both sources'
  fidelity stayed within a normal, comparable range (no collapse) across
  every repeated run** — this specific instability is closed, not just
  reduced.
- **3 new unit tests** (`tests/test_crosscoder.py`): joint-normalization
  invariant, a planted-cause recovery test (sparse one-hot shared/A-only/
  B-only causes — the sparse-concept structure TopK dictionaries are
  actually suited to; a smaller exploratory check confirmed the mechanism
  does *not* cleanly disentangle overlapping *continuous* Gaussian factors,
  a real, separate limitation noted in the module docstring rather than
  hidden), and the engineered mismatched-scale stability test above. One
  genuine flakiness bug found and fixed along the way: model weight init
  uses the *global* torch RNG (matching `TopKSAE`'s own existing pattern,
  which relies on the pipeline's `set_seed()` having already been called),
  so a test calling `train_crosscoder` directly without seeding the global
  RNG first got non-reproducible dead-atom counts between runs — fixed in
  the tests (`torch.manual_seed(cfg.seed)` before training), not in
  `crosscoder.py`, since production callers already go through
  `set_seed()`. Also fixed one real device-mismatch bug surfaced only by
  the real-checkpoint run below (see next bullet).
- **Real-checkpoint run** (`run_crosscoder_feasibility.py --run
  runs/medium_run_chronos_base`): loads an already-extracted store (no
  re-extraction, no new model calls), at the L1 peak-CKA pair (TimesFM
  `stacked_xf.4` ↔ Chronos-T5-Base `encoder.block.10`, CKA=0.381, 4608
  aligned rows), trains an independent baseline `TopKSAE` per model plus
  one joint crosscoder, and compares. First real-GPU run crashed with a
  device-mismatch `RuntimeError` in dead-neuron resampling
  (`torch.multinomial`'s CPU output tensor indexing a CUDA tensor two steps
  later) — a bug the CPU-only synthetic tests couldn't have caught; fixed
  by moving the sampled index tensor to the model's device immediately
  after `multinomial`. **Core stability answer: yes, comparable, no
  domination.** Across three hyperparameter settings tried (dict sizes
  1280–7680, k 16–24, 40–150 epochs), crosscoder fidelity for both sources
  landed close to (typically ~0.05–0.15 below) their own independently-
  trained baseline's fidelity, moving together rather than one collapsing
  while the other held — e.g. at the best-tuned setting: baseline
  TimesFM=0.669/Chronos-T5-Base=0.719 vs. crosscoder
  TimesFM=0.606/Chronos-T5-Base=0.648. Neither source was left
  near-unreconstructed at any setting tried.
- **A second real finding, orthogonal to stability: this run's dictionaries
  were mostly dead regardless of crosscoder vs. baseline.** Dead-feature
  rate stayed 90–98% across every dict-size/epoch combination tried, for
  *both* the independent baselines and the crosscoder alike — this is a
  data/training-budget property of these specific 4608 rows and this
  layer pair (plausibly low effective dimensionality at this depth, per
  `internals.py`'s own eff-dim curves elsewhere in this repo), not a
  crosscoder-specific pathology, since the plain single-model baseline
  hits the same wall at matched settings. **A related metric bug found and
  fixed**: `classify_features`'s shared/specific split, applied to *every*
  dictionary atom including dead ones, read as 97–99% "shared" at every
  setting — misleading, because a dead atom's decoder columns are just
  whatever random (but jointly-normalized) init left them at, which is
  centered near the "shared" band (0.5) by symmetry regardless of any
  learned signal. Added `alive_mask` and restricted the split to atoms
  that actually fired; the alive-only split at the best-tuned setting reads
  **69% shared, 30% TimesFM-specific, <1% Chronos-specific** among the 122
  (of 1280) alive atoms — still a real, likely-noisy read at this sample
  size, but no longer diluted by atoms carrying no signal at all.
- **Net answer to §13's question:** joint crosscoder training is
  trainable and stable across TimesFM and Chronos-T5-Base at their peak-CKA
  layer pair, once source-scale mismatch is corrected for (now built into
  `CrosscoderSAE` itself, not left as a caller responsibility). **Not yet
  answered:** whether the shared/specific split reflects real shared
  structure or mostly training-budget noise, given how few atoms survive
  at any dictionary size tried here — that needs either much more
  training/data, a smaller dictionary matched to this layer pair's actual
  effective dimensionality, or both, before the flagship crosscoder (full
  `SAEAdapter` implementation, a report section, ground-truth-checked
  shared features) is worth building on top of this mechanism. Artifact:
  `runs/crosscoder_feasibility.json` (not committed, `.gitignore`,
  regenerable via the command above).

---

### 6.2.1 The flagship crosscoder — `[x]` CLOSED 2026-08-18 — negative result, written up

> **Triage 2026-08-18 — read this before touching anything in §6.2/§6.2.1.**
> This subsystem has consumed more session time than any other in the repo.
> It has also produced its answer. Three questions, answered against recorded
> numbers rather than intent:
>
> **1. Is it worth the trouble? Yes — for two more days of work, then stop.**
> The pre-registered Stage 1c decision rule has already fired, *against* the
> flagship: V1 (the joint crosscoder) scores **0.2787 vs V0's 0.3204** on the
> fixed scorecard (diff −0.0417, CI [−0.0693, −0.0137], p=0.002, 46,382 rows),
> and clause 2's escape hatch — `frac_shared` **0.845 against an
> untrained-twin floor of 0.974** — makes *"joint training buys nothing at
> this layer pair, and the shared/specific axis does not separate learned
> structure from architecture match"* the supported reading. That is a real,
> decisive, negative result of exactly the kind §6.3 already produced and the
> repo already knows how to publish. It is worth finishing **because** it is
> negative and cheap to defend, not because a positive result is still
> expected.
>
> **2. Can Sonnet handle it? Yes for execution, no for the decisions.**
> Delegatable: V2's implementation (~30 lines), running
> `run_crosscoder_ladder.py`, transcribing numbers into Findings, doc
> compression. Pre-registration (Stage 1's fixed scorecard + 1c's rule) is
> precisely what makes that delegation safe — the judgment was made in
> advance, so the remaining work is mechanical. **Not delegatable**, because
> these four are what actually consumed the sessions: (a) deciding whether an
> unsatisfiable exit criterion means "fail" or "the criterion is malformed"
> (§11.29 — the answer was the second, and it took three sessions); (b)
> diagnosing arithmetic bugs whose symptom reads as a substantive finding
> (§11.31's −72 variance-explained, §11.19's fidelity collapse); (c)
> distinguishing a negative result from a wrong-layer/row-count artifact
> (§16 E9's third follow-up is the cautionary case); (d) the §11.24 check —
> did shared infrastructure move between two "identical" runs.
>
> **3. Can it be shortened without degrading integrity? Yes, and it already
> has been.** Compute is not the cost: **the full validation ladder runs in
> 37 seconds.** The cost was *deciding*, which the Stage 1c pre-registration
> exists to remove — so honor it. Concretely: V3/V5/V6 are parked (§22.1) —
> that is *following* the pre-registered rule, not cutting a corner; Stage 3
> (pipeline wiring) is not authorized for a variant that lost; Stage 3d
> (report legibility) is un-gated and worth doing on its own; run replicate
> seeds up front, since Stage 0 finding (15) measured failures as bimodal
> (one collapsed run in five) and a single seed will occasionally blame a
> variant for a bad basin. 🔴 **The integrity line, already tested twice and
> held both times:** weakening a numeric criterion so a gate passes is the
> violation. Sizing a *control* correctly (§11.29) is not.

**Remaining work: none.** Stage 2 (V0/V1/V2/V4) and Stage 4 (the writeup) are
both `[x]` as of 2026-08-18 — see their Findings blocks below for the full
numbers, including the V2-vs-V0 result (does not flip clause 2 at any of 3
seeds) and the new L-B floor seed-fragility finding. Stage 3 stays gated
shut, confirmed, not provisional. Stage 3d (report legibility) is the one
independent, un-gated item left if this subsystem is revisited — everything
else in this section is `[x]` and kept for its Findings.

**What this section inherits, and must not re-derive.**

1. **Source-scale mismatch is solved.** `CrosscoderSAE` normalizes each
   source by its own global std internally. Do not reintroduce a raw-scale
   MSE sum, and do not "simplify" `source_scale` away — a synthetic 20×
   scale mismatch reproducibly collapses the smaller source's fidelity to
   −9.7 without it (`CLAUDE.md` §11.19).
2. **The 90–98%-dead dictionary that blocked everything is fixed** (Stage 0,
   finding (17): 3.2% dead / 991 of 1024 alive, 5 of 5 seeds, via AuxK). The
   old "69% shared / 30% TimesFM-specific" split measured over 122 atoms is
   still **not quotable** — not because the dictionary is dead now, but
   because Stage 1 measured the same quantity on an alive one and found it
   architecture-dominated. Quote `frac_shared` only with its untrained-twin
   floor beside it.

---

#### Stage 0 — `[x]` CLOSED: produce a dictionary that is actually alive

> ✅ **Closed 2026-08-13 by finding (17).** `configs/crosscoder_stage0_gate.yaml`
> (dict 1024, k 48, AuxK on, per-model baselines at their **own** sizes 576/512)
> passes every criterion at 5 of 5 seeds — worst seed: 0.185 dead, 835 alive,
> fidelity 0.724 — and both baselines pass at every seed. **AuxK (H4) was
> decisive**; H3 was refuted on inspection (resampling had been on all along),
> and H1/H2 behaved *opposite* to the plan's premise (finding (16): the
> per-model baseline is 3.2–5.4× **less** dead than the crosscoder at every
> matched size, and the two move in opposite directions as the dictionary
> grows — which is why no single shared size could satisfy all three artifacts,
> `CLAUDE.md` §11.29). The four-hypothesis table and its ranked cheap tests are
> in `ROADMAP_ARCHIVE.md`; findings (1)–(17) below are the record.

**Exit criteria as pre-registered** (all must hold, on the crosscoder *and* on
each per-model `TopKSAE` baseline at the same data/budget — **matched budget,
not matched dictionary size**, per the 2026-08-13 DECISION below):

- `dead_feature_rate ≤ 0.30`
- `≥ 500` alive atoms absolute (later refined to a per-source floor scaling
  with eff-dim, 563 — *stricter*, not looser)
- `per_source_fidelity ≥ 0.70` for **both** sources simultaneously
- The winner is a committed config file, not a CLI incantation in a session log

🔴 **None of these numbers was ever relaxed**, across three sessions of the
gate failing. That is the integrity constraint this stage exists to
demonstrate; do not soften it if a future variant fails.

**Findings — Stage 0 preparation (2026-08-12): H3 refuted on inspection, H1's
substrate built and verified, H4 implemented, plus two bugs found along the
way.** No Stage 0 sweep has been *run* yet — this block records the four
things that had to be true before one could be, and the two things that
turned out not to be.

**(a) H3 is dead on arrival — see the Correction above the exit criteria.**
Decisive evidence was two lines of an existing CLI
(`run_crosscoder_feasibility.py:79,94`), not a rerun. Cost: minutes of
reading instead of the ~5 min the table budgeted, and it removes a
hypothesis rather than answering it. Resampling was on for every recorded
90–98%-dead number.

**(b) H1's big-data store is built and verified.**
`tsfm_lens/configs/crosscoder_stage0.yaml` (new, committed — Stage 0's exit
criteria demand a config file, not a CLI incantation) extracts *only* the
two layers of the recorded L1 peak-CKA pair (TimesFM `stacked_xf.4`,
Chronos-T5-Base `encoder.block.10`) from
`tsfm_benchmark/benchmark_out_full1/public_dev`, every analysis stage
disabled. Run name `crosscoder_stage0_bigdata` (new name, not a rerun into
an existing directory — §15 A3). Wall clock **50 s** total, extract stage
**42 s**, one A5000.

- Corpus 4288 samples, **975 skipped** (shorter than `context_len +
  horizon`, or non-finite) → **3313 series, 3 families**.
- Aligned rows: **46382** at 14 windows/series, against the feasibility
  run's 4608 → **10.07×**, i.e. **36.2 rows per atom** at `dict_size` 1280,
  up from 3.6. H1 is testable as designed.
- Store verification, verbatim (loaded, not inferred from shapes on disk):
  ```
  TimesFM stacked_xf.4 (46382, 1280) float32 mean 0.09705475717782974 std 3.0355584621429443 finite True
  Chronos-T5-Base encoder.block.10 (46382, 768) float32 mean -0.45171332359313965 std 22.106319427490234 finite True
  ```
  On disk float16, shapes `(3313,14,1280)` / `(3313,14,768)`, chunks
  `(256,14,1280)` / `(256,14,768)`, 196 MB. Confirmed a genuine zarr **v2**
  store (`.zattrs` present, no `zarr.json`) — checked explicitly against
  `CLAUDE.md` §11.25's silent-empty-store trap rather than assumed.
- `context_len: 448`, not the usual 512. 3205 of this corpus's 4288
  sequences are exactly 512 points long, so `512 + 64 = 576` would have
  admitted only 108 series. This is the one departure from
  `medium_run_chronos_base.yaml` that changes what is compared, and it has
  a consequence — see (c).

**(c) A real, diagnosed alignment finding at `context_len: 448`, which is a
probe-calibration artifact and NOT an adapter bug.** The extraction's own
`sanity_check` reported a Chronos diagonal-hit fraction of **0.50**, which
passed only because `min_diagonal_frac` is exactly `0.5` — i.e. the gate did
not fail, but it had no margin at all. Measured rather than assumed
(`CLAUDE.md` §2.4), and the first hypothesis (left-padding shifting the
token↔time map) was **refuted**: at 448 the tokenizer emits 449 tokens
(448 + EOS), the attention mask is all 1s, `postprocess_tokens` returns 448,
and spans are `[0,1]…[447,448]` — the declared mapping is correct.

Depth profile, same checkpoint and session, both context lengths:

```
block:            0    1    2    3    4    5    6    7    8    9   10   11
ctx=448         0.57 0.57 0.57 0.43 0.50 0.29 0.29 0.50 0.50 0.50 0.50 0.50
ctx=512         1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
```

Impulse-amplitude sweep at 448 (amplitude as a fraction of the probe's own
max, per §11.16's fix; "unrelated tokens changed" is the mean count of token
IDs that flipped away from the impulse's own window):

| amplitude | mean unrelated tokens changed | hit fractions |
|---|---|---|
| 0.25 (current default) | 38.6 | the 448 row above |
| 0.15 | 35.1 | still degraded |
| 0.10 | 26.0 | still degraded |
| 0.05 | 1.0 | **1.00 at blocks 0–7** |
| 0.02 | 1.0 | **1.00 at every block** |

A clean cliff, at the same amplitude where global re-quantization stops.
This is §11.16's mechanism exactly — Chronos's `MeanScaleUniformBins`
tokenizer computes bin edges from whole-sequence statistics — but it shows
that §11.16's fix constant (`0.25 × base.max()`) was calibrated at one
checkpoint size and **one context length** and does not generalize to
another context length of the same checkpoint. **Conclusion: the
`crosscoder_stage0_bigdata` store is trustworthy** (the token↔time map is
correct; the 0.50 measures probe amplitude, not span error). `alignment.py`
was deliberately **not** edited — changing that constant changes every
recorded alignment number in this repo, a §2.1 downstream call that needs
its own item rather than a drive-by fix. Tracked as §15 A20 below.

**(d) H4 (`aux_k`) is implemented and wired end to end, and does not
obviously work yet.** `sae/models.py` gains `pre_activations`/`sparsify`
(split out of `encode`, which is asserted bit-identical), `forward_with_pre`
(a separate method — `forward`'s two-tuple contract is consumed in a dozen
places), and `auxiliary_dead_loss`. `sae/train.py` and `sae/crosscoder.py`
both consume it; `SAETrainConfig`/`CrosscoderTrainConfig`/`SAEConfig` each
gain `aux_k: 0` (off by default, so every recorded number stays regenerable),
`aux_coef: 1/32`, `aux_dead_steps: 20`. `history["loss"]` stays
reconstruction-only so aux runs remain comparable to every aux-free run on
record; the aux term is reported separately as `history["aux"]`.

First synthetic diagnostic bed (10 planted causes, d=20, dict=320, k=4,
n=4000, 25 epochs, `TopKSAE`) — the full table, not the winner, since the
honest answer here is "no winner":

| variant | dead | fidelity |
|---|---|---|
| plain (aux off) | 0.272 | 0.8770 |
| signed, dead_steps=4, coef=0.03125 | 0.581 | 0.8682 |
| signed, dead_steps=4, coef=0.25 | 0.553 | 0.8420 |
| signed, dead_steps=20, coef=0.03125 | **0.266** | 0.8733 |
| signed, dead_steps=20, coef=0.25 | 0.375 | 0.8660 |
| relu'd, dead_steps=4, coef=0.03125 | 0.609 | 0.8668 |
| relu'd, dead_steps=4, coef=0.25 | 0.619 | 0.8482 |
| relu'd, dead_steps=20, coef=0.03125 | 0.453 | 0.8749 |
| relu'd, dead_steps=20, coef=0.25 | 0.500 | 0.8692 |

Three readings. (1) The **signed** (no-ReLU) variant beats the ReLU'd
variant at every matched setting — vindicating that documented design
choice, which was the one most likely to look like an oversight later.
(2) `aux_dead_steps=4` is **actively harmful** (0.581 vs. 0.272 plain): four
batches without firing mislabels sporadically-firing atoms as dead, and the
aux term then optimizes them *away* from the role they were actually
filling. (3) The best setting (0.266 vs. 0.272) is a wash. **AuxK is not
yet demonstrated to help.** This bed may simply be a bad test of it — 320
atoms for 10 planted causes means most atoms *should* be dead, so there is
almost no headroom for revival to be correct. A second bed with the
cause-to-atom ratio inverted is running; its result decides whether H4 is
carried into the real sweep as a candidate or recorded as refuted.
Consequently `tests/test_aux_k.py::test_aux_k_revives_dead_atoms_in_a_topk_sae`
currently **fails** (asserts `dead_aux < dead_plain − 0.05`; measured 0.581
vs. 0.272) and is knowingly left failing rather than weakened to green until
that question is settled.

**(e) A real reproducibility bug in `train_sae`, found by a new test rather
than by inspection.** `TopKSAE._init_weight` drew its dictionary from the
**global** torch RNG, while `cfg.seed` seeded only the local batch-permutation
`Generator`. Two `train_sae` calls with identical config and identical
`cfg.seed` therefore produced different dictionaries — histories starting
`0.16660461068153382` vs. `0.17042111217975617` — with the difference
determined by whatever ambient global stream state the process happened to
be in, which inside a pipeline run means *which earlier stages ran and how
many draws they made*. `run_crosscoder_feasibility.py`'s `set_seed(...)` +
`torch.manual_seed(...)` calls were caller-side workarounds for exactly this
gap. Fixed by threading an optional `generator` through
`TopKSAE.__init__`/`_init_weight` and `CrosscoderSAE.__init__`/`_init_weight`
and passing `generator=rng` from both training loops. This is a
`CLAUDE.md` §2.4 case in its purest form: the bug was invisible to reading
the code and was surfaced by an assertion that two identical runs agree.

**Superseded 2026-08-12 by the sweep-results block below** — the harness and
the sweep it calls for both exist and have been run; the eff-dim numbers and
the "18 alive atoms per effective direction may not be reachable" tension
below are still live and are directly addressed there. Kept per §0.2.

**Still open before the Stage 0 checkbox can move:** the sweep harness
(`run_crosscoder_stage0.py`) and the sweep itself (H2 → H1 → H4, H3 having
been eliminated above), plus the matched per-model `TopKSAE` baselines the
exit criteria require. Relevant prior number for H2, read from
`runs/medium_run_chronos_base/internals/profile.json` rather than guessed
(as the H2 row instructs): effective dimensionality is **28.15** for TimesFM
`stacked_xf.4` and **13.93** for Chronos-T5-Base `encoder.block.10`. A
1280-atom dictionary is therefore ~45× the pair's larger effective
dimensionality — and the exit criteria's conjunction ("≥ 500 alive **and**
≤ 30% dead") implies roughly **18 alive atoms per linear effective
direction**, which may not be reachable at any setting. If it isn't, that is
the publishable Stage 0 finding this section's own last paragraph asks for,
not a failure to report.

**Findings — Stage 0 sweep run (2026-08-12): H4 (AuxK) is decisive by a wide
margin, H1 and H2 are both refuted as fixes, and the gate does NOT pass —
but the blocker has moved from deadness to fidelity, and one per-model
baseline passes outright.** Nine grid rows × three trainings each (one joint
crosscoder + two matched per-model `TopKSAE` baselines on the same rows at
the same budget), 60 epochs, k=16, seed 0, against
`runs/crosscoder_stage0_bigdata`'s 46382 aligned rows (TimesFM
`stacked_xf.4`, d=1280; Chronos-T5-Base `encoder.block.10`, d=768). ~5 min
wall clock on one RTX A5000. Command:
`python run_crosscoder_stage0.py --run runs/crosscoder_stage0_bigdata --grid full --epochs 60 --k 16`.
Artifacts: `runs/crosscoder_stage0_bigdata/crosscoder_stage0.{json,md}`.

**The crosscoder sweep table, verbatim** (`crosscoder_stage0.md`; `n/a*` =
dictionary smaller than the 500-alive floor, so the row could not pass
regardless of training quality):

| row | rows | dict | rows/atom | xc fid TimesFM | xc fid Chronos-T5-Base | xc dead | xc alive | xc L0 | pass |
|---|---|---|---|---|---|---|---|---|---|
| `h2:dict=4x_effdim(112)` | 46382 | 112 | 414.1 | 0.4616 | 0.5856 | 0.5446 | 51 | 16.00 | n/a* |
| `h2:dict=16x_effdim(448)` | 46382 | 448 | 103.5 | 0.4753 | 0.6008 | 0.8638 | 61 | 16.00 | n/a* |
| `h2:dict=40x_effdim(1120)` | 46382 | 1120 | 41.4 | 0.4854 | 0.6070 | 0.9393 | 68 | 16.00 | fail |
| `h1:rows=4608(feasibility)` | 4608 | 1280 | 3.6 | 0.3205 | 0.3722 | 0.9609 | 50 | 16.00 | fail |
| `h1:rows=20000` | 20000 | 1280 | 15.6 | 0.4580 | 0.5614 | 0.9359 | 82 | 16.00 | fail |
| `h1:rows=all` | 46382 | 1280 | 36.2 | 0.4913 | 0.6170 | 0.9375 | 80 | 16.00 | fail |
| `h4:aux_off` | 46382 | 1280 | 36.2 | 0.4913 | 0.6170 | 0.9375 | 80 | 16.00 | fail |
| `h4:aux_k=64,coef=0.03125` | 46382 | 1280 | 36.2 | 0.6028 | 0.6791 | 0.3477 | 835 | 16.00 | fail |
| `h4:aux_k=64,coef=0.25` | 46382 | 1280 | 36.2 | 0.5687 | 0.6543 | 0.1625 | 1072 | 16.00 | fail |

`crosscoder_rows_passing: []` — **no crosscoder row passes all three exit
criteria.** The matched per-model baselines, same rows and budget, full
precision:

| row | model | dict | fidelity | dead | alive | L0 | final MSE | passes |
|---|---|---|---|---|---|---|---|---|
| `h2:dict=4x_effdim(112)` | TimesFM | 112 | 0.53932497404855 | 0.3660714328289032 | 71 | 16.0 | 0.7405528528154408 | no |
| `h2:dict=4x_effdim(112)` | Chronos-T5-Base | 112 | 0.6982846197513244 | 0.6964285969734192 | 34 | 16.0 | 12.565651349152441 | no |
| `h2:dict=16x_effdim(448)` | TimesFM | 448 | 0.5622627384910894 | 0.7723214626312256 | 102 | 16.0 | 0.7028092962820335 | no |
| `h2:dict=16x_effdim(448)` | Chronos-T5-Base | 448 | 0.7227196082177738 | 0.8705357313156128 | 58 | 16.0 | 11.547010354723085 | no (fidelity only) |
| `h2:dict=40x_effdim(1120)` | TimesFM | 1120 | 0.6033689051699381 | 0.8803571462631226 | 134 | 16.0 | 0.6371318326669371 | no |
| `h2:dict=40x_effdim(1120)` | Chronos-T5-Base | 1120 | 0.7329605062985123 | 0.9276785850524902 | 81 | 16.0 | 11.135191526960305 | no (fidelity only) |
| `h1:rows=4608` | TimesFM | 1280 | 0.32733329705716274 | 0.93359375 | 85 | 16.0 | 1.0818790197372437 | no |
| `h1:rows=4608` | Chronos-T5-Base | 1280 | 0.5424557552927723 | 0.9671875238418579 | 42 | 16.0 | 19.147632175021702 | no |
| `h1:rows=20000` | TimesFM | 1280 | 0.5359753360595216 | 0.866406261920929 | 171 | 16.0 | 0.7450868275642395 | no |
| `h1:rows=20000` | Chronos-T5-Base | 1280 | 0.6450327434749685 | 0.9554687738418579 | 57 | 16.0 | 14.749556788635253 | no |
| `h1:rows=all` / `h4:aux_off` | TimesFM | 1280 | 0.6104473418821939 | 0.893750011920929 | 136 | 16.0 | 0.621604638008481 | no |
| `h1:rows=all` / `h4:aux_off` | Chronos-T5-Base | 1280 | 0.7291947010247736 | 0.9453125 | 70 | 16.0 | 11.245107474094128 | no (fidelity only) |
| `h4:aux_k=64,coef=0.03125` | TimesFM | 1280 | 0.7080086114967346 | 0.07500000298023224 | 1184 | 16.0 | 0.47092298487967 | ✅ **PASSES ALL THREE** |
| `h4:aux_k=64,coef=0.03125` | Chronos-T5-Base | 1280 | 0.7900891284979146 | 0.69921875 | 385 | 16.0 | 8.707635288831513 | no (fidelity only) |
| `h4:aux_k=64,coef=0.25` | TimesFM | 1280 | 0.6972009715293437 | 0.06406249850988388 | 1198 | 16.0 | 0.4891255340342904 | no (dead+alive only) |
| `h4:aux_k=64,coef=0.25` | Chronos-T5-Base | 1280 | 0.8002951914643573 | 0.5859375 | 530 | 16.0 | 8.339681685076233 | no (fidelity+alive) |

**Crosscoder full precision for the three rows that matter.** `h4:aux_off` /
`h1:rows=all`: fidelity `{TimesFM: 0.49126626478729474, Chronos-T5-Base:
0.6170109332217114}`, dead `0.9375`, alive `80`, loss `0.12138739499657784`
(per-source `[0.08870120313069803, 0.03268619252384149]`, aux `0.0`).
`h4:aux_k=64,coef=0.03125`: fidelity `{TimesFM: 0.6028074448732459,
Chronos-T5-Base: 0.6791484134622208}`, dead `0.34765625`, alive `835`, loss
`0.09693617558081961` (per-source `[0.06940870532592258,
0.02752747118385025]`, aux `0.848771290176462`).
`h4:aux_k=64,coef=0.25`: fidelity `{TimesFM: 0.568718036289822,
Chronos-T5-Base: 0.6542606601812815}`, dead `0.16249999999999998`, alive
`1072`, loss `0.1047686935232541` (per-source `[0.07522591450347417,
0.02954277934876077]`, aux `0.8658747349605385`).

**(1) H4 is the decisive hypothesis — and it was ranked last.** At a fixed
dictionary, fixed rows, and everything else identical, turning on AuxK moves
the crosscoder from **80 alive atoms (93.75% dead) to 835 (34.8%) at
`coef=0.03125` and 1072 (16.2%) at `coef=0.25`** — a 10–13× increase in
living dictionary capacity. Crucially it is **not a liveness-for-fidelity
trade at `coef=0.03125`**: crosscoder fidelity *rises* on both sources at
the same time (TimesFM 0.4913 → 0.6028, Chronos 0.6170 → 0.6791), and the
per-model TimesFM baseline goes from 0.6104/89.4% dead to 0.7080/7.5% dead.
Starved atoms were not surplus capacity correctly declining to represent
nothing; they were capacity the optimizer could not reach, and the aux
gradient reaches it. `coef=0.25` buys more liveness but starts costing
fidelity (TimesFM 0.6028 → 0.5687), so 0.03125 is the better of the two
tested and the coefficient is worth sweeping finer.

**(2) H1 (data per atom) is refuted for deadness and confirmed for
fidelity — a clean dissociation.** Going 4608 → 20000 → 46382 rows at fixed
`dict_size=1280` moves the dead rate `0.9609 → 0.9359 → 0.9375` — flat, and
not even monotonic. Alive atoms go `50 → 82 → 80`. So 10× the data buys
essentially **nothing** in liveness. The same 10× buys a great deal of
*fidelity*: TimesFM `0.3205 → 0.4580 → 0.4913`, Chronos `0.3722 → 0.5614 →
0.6170`, and on the TimesFM baseline `0.3273 → 0.5360 → 0.6104`. This is
worth stating plainly because H1 was the *most likely cause* in the ranking
and the extraction built for it (§6.2.1 Stage 0 prep (b)) was the session's
most expensive prerequisite: the extra data was necessary — every
AuxK number above sits on it and the 4608-row rows are the worst in the
table on every axis — but it is not what fixes deadness.

**(3) H2 (dictionary vs. effective dimensionality) is refuted as a fix, and
the reason is the sharpest single number in the sweep: the alive-atom count
is nearly invariant to dictionary size.** 112 / 448 / 1120 atoms yield
**51 / 61 / 68** alive atoms respectively — a 10× change in dictionary size
moves the surviving count by ~33%. The dead *rate* does fall as the
dictionary shrinks (`0.9393 → 0.8638 → 0.5446`) but only because the
denominator shrinks; nothing more is actually alive. Under plain TopK at
k=16 this layer pair supports an **absolute ceiling of roughly 50–80 living
atoms**, essentially regardless of how many are offered. That ceiling is
what makes the ≥500-alive criterion unreachable by resizing: the two smaller
H2 rows are marked `n/a*` because a 112- or 448-atom dictionary cannot host
500 alive atoms even in principle. It also directly answers the "18 alive
atoms per linear effective direction may not be reachable" tension recorded
in the preparation block above — **it is reachable, but only via AuxK**
(1072 alive against a summed effective dimensionality of 28.15 + 13.93),
never via sizing.

**(4) The gate does not pass, and the blocker is now fidelity alone.**
`h4:aux_k=64,coef=0.25` satisfies `dead_ok` (0.1625 ≤ 0.30) and `alive_ok`
(1072 ≥ 500) and fails only `fidelity_ok` (TimesFM 0.5687, Chronos 0.6543,
both < 0.70). Before this sweep every row failed all three. **Stage 0's
checkbox stays `[ ]`.**

**(5) One configuration passes outright, and it is a per-model baseline, not
the crosscoder.** TimesFM's `TopKSAE` at `aux_k=64, coef=0.03125`, 46382
rows, dict 1280: fidelity 0.7080086114967346, dead 0.07500000298023224,
1184 alive — all three criteria, the first fully-passing configuration
anywhere in this work. The joint crosscoder at the identical setting reaches
only 0.6028 on the same source. **Joint training currently costs fidelity on
both sources at every row in the table** (crosscoder < both baselines,
without exception), which is a real result about the crosscoder rather than
about deadness, and is exactly what Stage 1's scorecard exists to quantify.

**(6) Chronos's baseline reaches `fidelity_ok` at five separate rows while
70–97% dead — and that is not a bug.** At `h2:dict=4x_effdim(112)` it hits
0.6982846197513244 with **34 alive atoms**; at `h1:rows=all`,
0.7291947010247736 with 70. Its measured effective dimensionality at
`encoder.block.10` is 13.93, so a few dozen atoms genuinely suffice to
reconstruct it — the dictionary is not failing to learn, there is
comparatively little there to learn. TimesFM (eff-dim 28.15, d=1280) needs
an order of magnitude more atoms for comparable fidelity. Read the
per-source fidelity and the per-source alive count together; neither alone
distinguishes "collapsed" from "the source is low-dimensional."

**(7) Consistency checks that came out clean.** `l0_actual` is exactly
`16.00` in all 27 trainings (= k, as plain TopK requires — no silent
sparsity drift). `h4:aux_off` reproduces `h1:rows=all` to every printed
digit on the crosscoder and both baselines despite being trained in a
separate call, which independently confirms the seeded-`generator` fix from
the preparation block (e) actually made training reproducible. No fidelity
fell outside [0, 1]. The `aux` loss term is exactly `0.0` in every `aux_k=0`
row and nonzero only where enabled.

**(8) The synthetic AuxK bed did not predict this, and the tests were
written accordingly.** `tests/test_aux_k.py`'s planted-data beds measured
AuxK as a **wash** (dead 0.272 → 0.266) on one bed and provably **inert** on
a second where nothing was dead — which is why that file asserts only "does
no harm" rather than "revives atoms" (`CLAUDE.md` §2.4). On real
activations the same mechanism is a 10–13× effect. The synthetic tests were
not wrong; they were measuring a regime where the failure mode AuxK fixes
does not occur. Had the stronger claim been asserted to match the
hypothesis, it would have passed for the wrong reason on synthetic data and
told us nothing about the real case.

**Next action for Stage 0, and the specific reason:** sweep **k**, not
anything already in the grid. Every row above ran at `k=16` while TimesFM's
`stacked_xf.4` has a measured effective dimensionality of **28.15** — a
16-atom reconstruction budget cannot span a ~28-dimensional subspace, so
the fidelity ceiling of ~0.60–0.71 seen across the entire table is the
expected consequence of the sparsity budget rather than evidence about the
dictionary. `k ∈ {32, 48, 64}` at `aux_k=64, coef=0.03125` on all 46382
rows is the direct test, and it is the only untested lever that plausibly
moves fidelity above 0.70 on both sources simultaneously. Note this also
loosens the interpretability claim (higher L0 = less sparse features), so
record the smallest k that clears the gate rather than the largest tried.
Finer `aux_coef` between 0.03125 and 0.25 is the secondary lever.

**Findings — Stage 0 k sweep (2026-08-12, same session): the crosscoder gate
PASSES at k=48; the prediction above was correct and the remaining failure
is on the matched *Chronos baseline*, not the crosscoder.** Ran
`run_crosscoder_stage0.py --grid k` against the same
`runs/crosscoder_stage0_bigdata` store (46382 rows, TimesFM `stacked_xf.4` ↔
Chronos-T5-Base `encoder.block.10`, `dict_size=1280`, `aux_k=64,
aux_coef=0.03125`, 60 epochs, seed 0 — only `k` varies). Artifacts:
`crosscoder_stage0_ksweep.{json,md}` in that run directory.

| row | rows | dict | rows/atom | xc fid TimesFM | xc fid Chronos-T5-Base | xc dead | xc alive | xc L0 | pass |
|---|---|---|---|---|---|---|---|---|---|
| `k=16:aux_k=64,coef=0.03125` | 46382 | 1280 | 36.2 | 0.6028 | 0.6791 | 0.3477 | 835 | 16.00 | fail |
| `k=32:aux_k=64,coef=0.03125` | 46382 | 1280 | 36.2 | 0.6899 | 0.7048 | 0.1516 | 1086 | 32.00 | fail |
| `k=48:aux_k=64,coef=0.03125` | 46382 | 1280 | 36.2 | 0.7589 | 0.7556 | 0.0203 | 1254 | 48.00 | **PASS** |
| `k=64:aux_k=64,coef=0.03125` | 46382 | 1280 | 36.2 | 0.7485 | 0.7028 | 0.2688 | 936 | 64.00 | PASS |

**(1) k was the right lever, and the effect is monotone up to k=48 on every
axis at once.** Fidelity rises on both sources (TimesFM 0.6028 → 0.6899 →
0.7589; Chronos 0.6791 → 0.7048 → 0.7556) *while* the dead rate falls
(0.3477 → 0.1516 → 0.0203) and alive atoms rise (835 → 1086 → 1254 of 1280).
This is not a sparsity-for-fidelity trade in the k≤48 range — it is the
straightforward consequence of the diagnosis in the previous block: at
`k=16` the reconstruction budget was smaller than the subspace TimesFM's
layer actually occupies (measured eff-dim **28.15**), so atoms competed for
too few slots and most never won one.

**(2) The winner is `k=48`, recorded as the smallest k that clears the gate,
not the best-looking one.** Full precision: crosscoder fidelity
**0.7588510627220952** (TimesFM) / **0.7556082781475381** (Chronos-T5-Base),
`dead_feature_rate` **0.020312499999999956**, **1254** alive atoms of 1280,
`l0_actual` exactly 48.0, final loss 0.061717941376224925 (per-source
[0.0408388023874678, 0.02087914035793102]). All three numeric criteria hold
simultaneously with margin (0.0203 ≤ 0.30; 1254 ≥ 500; both fidelities ≥
0.70), which no configuration in the previous 9-row sweep achieved on any
axis pair.

**(3) k=64 is strictly worse than k=48 despite also passing — a real
non-monotonicity worth recording.** Going 48 → 64 *lowers* fidelity on both
sources (0.7589 → 0.7485, 0.7556 → 0.7028) and *raises* the dead rate more
than 13× (0.0203 → 0.2688, alive 1254 → 936), while making every feature
less sparse. The optimum is interior, so "more k is better" is false past
the effective dimensionality; the previous block's instruction to record the
smallest passing k turns out to also select the best one here. Not
over-read: this is one seed at one layer pair, and 0.0203 vs. 0.2688 is a
large enough gap that seed noise is an unlikely explanation, but the
*location* of the optimum between 32 and 64 is resolved only to ±16.

**(4) The gate is NOT fully cleared, and the reason has moved again — from
the crosscoder to one per-model baseline.** The exit criteria require all
three conditions on the crosscoder *and* on the matched per-model `TopKSAE`
baselines at the same budget. At k=48 the **TimesFM baseline passes
outright** (fidelity 0.8223450861413343, dead 0.01718750037252903, 1258
alive) but the **Chronos-T5-Base baseline fails `dead_ok`** — fidelity
0.871442102659578 and 576 alive (both fine), dead rate
**0.550000011920929**, well over the 0.30 bar. The same pattern holds at
every k (Chronos baseline dead 0.699 / 0.745 / 0.550 / 0.543 at k =
16/32/48/64), i.e. it is not a k problem. Stage 0's checkbox therefore stays
`[ ]`.

**(5) But the specific confound Stage 0 exists to remove is gone.** The
gate's stated purpose is that a 90–98% dead crosscoder makes
`relative_decoder_norm` read ~98% "shared" from dead-atom symmetry alone. At
k=48 the crosscoder has 1254 of 1280 atoms alive — that failure mode is
eliminated, and a shared-vs-specific split computed on *this* dictionary is
no longer symmetric-by-deadness. The residual Chronos-baseline deadness
affects the **V0 per-model-dictionaries route** (`sae/matching.py`), which
is the comparison baseline every crosscoder variant must beat in Stage 2 —
so it is a live problem for Stage 2's scorecard, not for the crosscoder's
own trainability. Recording it here rather than closing Stage 0 on a
technicality, because the distinction is the whole reason the criteria named
both objects.

**(6) The likely cause of the Chronos baseline's deadness, and why the
earlier H2 refutation does not settle it.** Chronos-T5-Base's
`encoder.block.10` has measured eff-dim **13.93**, half TimesFM's 28.15, so
a 1280-atom dictionary is ~92× its effective dimensionality (vs. ~45× for
TimesFM) — a smaller per-model dictionary is the obvious candidate fix. The
previous block refuted H2 (alive count nearly invariant to dict size:
51/61/68 alive at 112/448/1120), **but every one of those rows ran with
AuxK off**, and AuxK is what turned out to be decisive. The H2 × H4
interaction has never been tested. That is the concrete next experiment for
this residual: per-model dictionary sizing (e.g. `dict_size ≈ 20–40×
eff_dim`, so ~280–560 for Chronos) *with* `aux_k=64, coef=0.03125` on, which
is cheap — it reuses the same store and touches only the baseline arm.

**(7) The joint-training fidelity cost persists at every k, and is now
quantified where it matters.** The crosscoder is below both per-model
baselines on their own source in all four rows, without exception. At the
winning k=48: TimesFM 0.7589 (joint) vs. 0.8223 (own SAE), Chronos 0.7556
vs. 0.8714 — a cost of 0.063 and 0.116 fidelity respectively. This is a real
property of the crosscoder, not a training defect, and it is exactly the
quantity Stage 1's scorecard exists to weigh against whatever cross-model
alignment the joint dictionary buys. Note the cost is ~2× larger on Chronos,
the lower-eff-dim source — consistent with the joint dictionary being
sized/budgeted for the harder source.

**(8) Consistency checks clean.** `l0_actual` is exactly `k` in all four
crosscoder rows and all eight baselines (16.0/32.0/48.0/64.0 — no sparsity
drift). `k=16:aux_k=64,coef=0.03125` reproduces the previous sweep's
`h4:aux_k=64,coef=0.03125` row to every printed digit (fidelity 0.6028 /
0.6791, dead 0.3477, alive 835) despite being a separate invocation through
a new grid — a second independent confirmation of the seeded-`generator`
reproducibility fix, and evidence the new `k_is_swept` flag did not
perturb the shared rows. All fidelities in [0, 1]; `aux` nonzero in every
row (AuxK enabled throughout).

**(9) The winner is committed as a config, and the config was verified by
replay rather than by reading it.** `configs/crosscoder_stage0_winner.yaml`
holds the winning hyperparameters, the store and layer pair they were
measured on, and the measured result inline for diffing;
`run_crosscoder_stage0.py --params <file>` executes it as a single-row sweep
(unknown `train:` keys raise rather than being ignored, so a typo cannot
silently become a different experiment). Running it end to end reproduced
the sweep **bit-for-bit** — fidelity 0.7588510627220952 / 0.7556082781475381,
dead 0.020312499999999956, 1254 alive, and both baselines identical to
their swept values on every field. That is a stronger check than "the config
file says the right numbers": it confirms the file is a runnable definition
of the experiment rather than a transcription of one, which is the
difference the exit criterion was asking for. Pinned by
`tests/test_stage0_winner_config.py` (4 tests, no GPU), which also checks the
config's inline `expected:` block against the real artifact — the
`CLAUDE.md` §11.24 failure mode is a config whose *meaning* drifts while its
bytes stay identical, and a config recording numbers nobody re-checks is
exactly where that hides.

**Next action for Stage 0:** the winning configuration is committed as
`configs/crosscoder_stage0_winner.yaml` (the fourth exit criterion), so the
only thing standing between here and `[x]` is the per-model Chronos baseline
in (6) — one cheap sweep over `dict_size` with AuxK on. Do that before
Stage 1, since Stage 1's ladder and Stage 2's V0 baseline both consume the
per-model dictionaries this affects.

---

**Findings — Stage 0 H2 × H4: the dictionary-size sweep with AuxK on
(2026-08-12, same session).** The experiment finding (6) directly above
named, run against the same `runs/crosscoder_stage0_bigdata` store, same
seed, same k=48 / `aux_k=64, coef=0.03125` / 60 epochs, varying only
`dict_size` (`run_crosscoder_stage0.py --grid h2xh4`, artifacts
`crosscoder_stage0_h2xh4.json`/`.md`):

| row | dict | rows/atom | xc fid TimesFM | xc fid Chronos | xc dead | xc alive | pass |
|---|---|---|---|---|---|---|---|
| `h2xh4:dict=640` | 640 | 72.5 | 0.7346 | 0.7670 | 0.0203 | 627 | PASS |
| `h2xh4:dict=768` | 768 | 60.4 | 0.7356 | 0.7260 | 0.1081 | 685 | PASS |
| `h2xh4:dict=896` | 896 | 51.8 | 0.7590 | 0.7782 | 0.0112 | 886 | PASS |
| `h2xh4:dict=1024` | 1024 | 45.3 | 0.7363 | 0.7348 | 0.0322 | 991 | PASS |

**(1) H2 stays refuted, now in the regime where it could have mattered.**
Finding (6) flagged that the original H2 refutation used AuxK-off rows and
that the H2 × H4 interaction had never been tested. It has now: with AuxK
on, the crosscoder clears all three numeric criteria at **every** dictionary
size from 640 through 1280 (the k-sweep's winner). Dictionary size is not
the lever for crosscoder deadness in either regime — `k` and AuxK are, and
the earlier refutation holds up rather than being an artifact of the broken
regime it was measured in.

**(2) The best crosscoder row in this repo so far is `dict=896`, not the
committed k=48/dict=1280 winner.** Fidelity 0.7590 / 0.7782 (vs. the
winner's 0.7589 / 0.7556), dead rate **0.0112** (vs. 0.0203), 886 of 896
atoms alive. It beats the committed winner on Chronos fidelity by 0.023 and
ties on TimesFM, at 70% of the parameters. **`configs/crosscoder_stage0_
winner.yaml` was deliberately not edited** — it records the row the k sweep
selected under the stated rule (smallest passing k), the numbers in its
`expected:` block are that row's, and rewriting it to chase a 0.02 fidelity
difference found by a different sweep would break the property finding (9)
just established (a config that is a runnable definition of the experiment
it names). If Stage 1 wants dict=896, that is a new committed config with
its own replay, not an in-place edit of this one.

**(3) The Chronos-T5-Base per-model baseline fails at every dictionary size,
and the failure mode is a pinch between two criteria rather than a
threshold to tune.** Its alive-atom count is almost flat across a 2×
dictionary range, so the dead *rate* is essentially mechanical:

| dict | 640 | 768 | 896 | 1024 | 1280 |
|---|---|---|---|---|---|
| alive | 467 | 473 | 529 | 546 | 576 |
| dead rate | **0.2703** | 0.3841 | 0.4096 | 0.4668 | 0.5500 |
| `dead_ok` (≤0.30) | ✅ | ❌ | ❌ | ❌ | ❌ |
| `alive_ok` (≥500) | ❌ | ❌ | ✅ | ✅ | ✅ |
| fidelity | 0.8880 | 0.8757 | 0.8810 | 0.8739 | 0.8714 |

Doubling the dictionary buys 109 additional living atoms (467 → 576). The
two criteria therefore move in opposite directions and **cross without ever
both holding**: `dict=640` is the first size to clear the rate bar (0.2703,
comfortably under 0.30) and misses the absolute floor by 33 atoms; every
larger size clears the floor and fails the rate. Fidelity is never the
problem — it is 0.87–0.89 throughout, well above the 0.70 bar, and the
*highest* fidelity in the table sits on the *smallest* dictionary.

**(4) The candidate fix named in (6) is therefore refuted, and the residual
is now a criteria question rather than a training question.** Finding (6)
predicted "per-model dictionary sizing (~280–560 for Chronos)" would fix
this. It does not: shrinking the dictionary lowers the rate exactly as
predicted, but takes the absolute alive count down with it, because this
layer supports a bounded number of distinguishable atoms (~470–580 here) no
matter how many are offered. Extrapolating the alive-fraction curve
(0.450 / 0.533 / 0.590 / 0.616 / 0.730 at dict 1280→640) says no size in the
500–640 range reaches 500 alive atoms either, which would make the `≥500`
floor **unreachable for this model at this layer**, not merely unreached.
That extrapolation is not yet a measurement — a `dict ∈ {512, 576, 704}`
confirmation sweep is running as this is written; its result is the
deciding evidence and is written up below when it lands.

**(5) What this means for Stage 0's checkbox, stated rather than resolved.**
If (4)'s extrapolation confirms, the honest reading is that `MIN_ALIVE =
500` is a *global* constant applied to a layer whose measured effective
dimensionality is **13.93** — asking a 14-dimensional representation to
support 500 distinguishable atoms while also keeping ≥70% of them alive is
a stronger demand than the same constant makes of TimesFM's 28.15. The
constant was chosen (§6.2.1 Stage 0) to make `relative_decoder_norm`
meaningful, and 467 alive atoms serve that purpose as well as 546 do — the
dead-atom symmetry confound that motivated the whole gate is *gone* at
`dict=640` (73% alive). Two defensible resolutions: scale the floor per
model (e.g. `min_alive ≥ 20× eff_dim`, giving ~279 for Chronos and ~563 for
TimesFM), or keep the constant and record Stage 0 as passing on the
crosscoder while the Chronos baseline carries a stated, quantified caveat
into Stage 2's scorecard. **Not decided here** — it changes what a recorded
gate means, so it is a call to make deliberately rather than one to slip in
alongside a sweep result. Stage 0 stays `[ ]` pending it.

**(6) Consistency checks clean.** `l0_actual` is exactly 48.0 in all four
crosscoder rows and all eight baselines. The TimesFM baseline passes at
every size (dead 0.0000 / 0.0039 / 0.0033 / 0.0117; alive 640 / 765 / 893 /
1012; fidelity 0.795–0.807), confirming the failure in (3) is specific to
the Chronos side and not a property of the harness or the store. The
joint-training fidelity cost from finding (7) reproduces at every size
(e.g. at dict=896: 0.759 joint vs. 0.807 own-SAE on TimesFM, 0.778 vs.
0.881 on Chronos).

**(7) The pinch is confirmed by measurement, not left as an extrapolation
(`--grid pinch`, dict ∈ {512, 576, 704}, same session).** Finding (4) said
the deciding evidence was a sweep below `h2xh4`'s smallest dictionary. It
ran; artifacts `crosscoder_stage0_pinch.json`/`.md`. Chronos-T5-Base
baseline, now across the full measured range:

| dict | 512 | 576 | 640 | 704 | 768 | 896 | 1024 | 1280 |
|---|---|---|---|---|---|---|---|---|
| alive | 386 | 355 | 467 | 405 | 473 | 529 | 546 | 576 |
| dead rate | **0.2461** | 0.3837 | **0.2703** | 0.4247 | 0.3841 | 0.4096 | 0.4668 | 0.5500 |
| `dead_ok` | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `alive_ok` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |

**No dictionary size in [512, 1280] satisfies both criteria**, and the two
that clear the rate bar (512 and 640) land at 386 and 467 alive — 114 and
33 short of the floor. Finding (4)'s extrapolation is therefore confirmed
in the direction it predicted, and (5)'s criteria question is now the real
residual rather than a hypothetical one.

**(8) But the alive count is noisy enough that the smooth curve in (3) was
partly an illusion — worth knowing before anyone fits a model to it.**
Ordering by dictionary size, alive goes 386 → 355 → 467 → 405 → 473 → 529 →
546 → 576: *non-monotone*, with dict=576 below dict=512 and dict=704 below
dict=640. Every row is a single seed, so run-to-run SAE-training variance
on this layer is on the order of ±60 alive atoms — comparable to the
33-atom margin (4) turns on. Two consequences, both stated rather than
worked around: the "bounded number of distinguishable atoms (~470–580)"
band in (4) is better read as **~350–580 with substantial noise**, and any
future attempt to find a passing dictionary size by search should average
replicate seeds per size (the same fix `run_layer_screen_bakeoff.py
--n-gold-replicates` already applies to the bake-off's gold ranking, for
the same reason — see `CLAUDE.md` §9's layer-screen row). It does not
change (7)'s conclusion: the *best* rate-passing row across eight
dictionary sizes is 33 atoms short, and the noise band does not reach 500
at any size where the rate bar also holds.

**(9) The crosscoder's own passing range has a floor too, now located.**
`dict=512` and `dict=576` fail — not on fidelity for 512 (0.7103 / 0.7220,
both above the bar) but on the absolute alive floor (451 and 461 atoms);
`dict=576` additionally drops Chronos fidelity to 0.6867, below 0.70. So
the crosscoder passes for every `dict_size ≥ 640` tried and fails below it,
which brackets the committed winner comfortably rather than putting it near
an edge.

**Next action for Stage 0, revised.** The training-side search is finished:
three grids (`k`, `h2xh4`, `pinch`) across four sparsity budgets and eight
dictionary sizes establish that the crosscoder clears every criterion
comfortably and the Chronos-T5-Base per-model baseline clears none of the
size-dependent ones at any setting. **What is left is not another sweep —
it is (5)'s decision**: whether `MIN_ALIVE = 500` should be a global
constant or scale with the layer's measured effective dimensionality. Both
options are defensible and both are one-line changes to
`run_crosscoder_stage0.py`'s constants; what is *not* defensible is picking
whichever one makes the gate pass. Recommended framing for whoever makes
it: the gate exists to stop dead-atom symmetry from faking a
shared-vs-specific split, that failure mode is measured by the *fraction*
alive (73% at `dict=640`) rather than by the absolute count, and the
absolute floor was added as a crude guard against a tiny dictionary
trivially satisfying the fraction — a guard that a per-model floor of
`20× eff_dim` (279 Chronos / 563 TimesFM) serves at least as well. Stage 0
stays `[ ]` until that call is made and written down.

**(10) The replicate seeds landed, and they refute the premise the decision
was being staged on (`configs/crosscoder_stage0_replicate640.yaml`, seeds
0–4, 2026-08-12).** Finding (8) said the 33-atom margin at `dict=640` was
comparable to single-seed noise, so the criteria call should rest on
replicates rather than on that one row. It now does, and the answer is not
the one the framing anticipated: the margin is not marginal, and seed 0 —
the row every previous finding quoted — was **the most favorable of five
draws on both criteria simultaneously**. Seed 0 reproduces bit-for-bit
(crosscoder fidelity 0.7346103683566961 / 0.7669887293414291, dead
0.020312499999999956, alive 627; Chronos baseline dead 0.27031251788139343,
alive 467, fidelity 0.8880293424795604), so this is genuine seed variance,
not drift (`CLAUDE.md` §11.24 checked and cleared).

| seed | xc fid TimesFM | xc fid Chronos | xc dead | xc alive | baseline-A alive | baseline-B dead | baseline-B alive |
|---|---|---|---|---|---|---|---|
| 0 | 0.7346 | 0.7670 | 0.0203 | 627 | 640 | **0.2703** | **467** |
| 1 | 0.7197 | 0.7157 | 0.0250 | 624 | 639 | 0.3375 | 424 |
| 2 | 0.7361 | 0.7579 | 0.0219 | 626 | 635 | 0.4203 | 371 |
| 3 | 0.6966 | 0.6993 | 0.0500 | 608 | 639 | 0.3063 | 444 |
| 4 | 0.7023 | 0.6999 | 0.0563 | 604 | 633 | 0.3891 | 391 |
| **mean ± sd** | 0.7179 ± 0.0181 | 0.7279 ± 0.0323 | 0.0347 ± 0.0171 | 617.8 ± 10.9 | 637.2 ± 3.0 | 0.3447 ± 0.0607 | 419.4 ± 38.9 |

Three things follow, none of them what finding (8) predicted:

- **The Chronos baseline clears `alive ≥ 500` in 0 of 5 seeds.** The mean
  shortfall is **80.6 atoms (sd 38.9)** — about 2.1 sd from the bar, not the
  33-atom coin-flip finding (8) described. Seed 0's 467 is the *maximum* of
  the five.
- **Its `dead ≤ 0.30` pass at `dict=640` also fails to replicate** — 1 of 5
  seeds (mean 0.3447 ± 0.0607). So the "first size to clear the rate bar"
  language in finding (3) and the table in (7) describes a single lucky
  draw, and `dict=640` is better read as a **rate-bar failure** too.
  🔴 **The header comment in `configs/crosscoder_stage0_replicate640.yaml`
  asserted both of those as fact; it has been corrected in place rather than
  deleted, and no other config quotes them.**
- **TimesFM's baseline is not noisy at all** (alive 637.2 ± 3.0, fidelity
  0.7913 ± 0.0026, dead ≤0.011). The ±60-atom noise band finding (8)
  inferred from the non-monotone size curve is **Chronos-specific**, and on
  the Chronos side it is if anything larger (sd 38.9 at one fixed size)
  while on the TimesFM side it barely exists. Reading one noise band off a
  curve that mixes both models was the error.

**(11) The criteria are jointly unsatisfiable for this model, provably, at
any dictionary size — which is what actually decides (5).** Combining the
two constants: `dead ≤ 0.30` is exactly `alive ≥ 0.70 × dict`, so the pair
requires a size where `alive ≥ 500` **and** `dict ≤ alive / 0.70`. The two
bars therefore pull in opposite directions — the floor wants a *large*
dictionary (alive only ever grows with dict), the rate bar a *small* one.
Chronos-T5-Base's alive count at this layer **saturates** — across the eight
swept sizes it runs 386 / 355 / 467 / 405 / 473 / 529 / 546 / 576 while the
dictionary grows 512 → 1280, i.e. the dead *rate* climbs monotonically
(0.246 → 0.550) precisely because the alive count stops tracking the
dictionary. Once alive plateaus at a ceiling C, every dictionary above
C / 0.70 fails the rate bar permanently; with C = 576 that caps a passing
size at **823**, while the smallest size whose alive count reaches 500 is
**896**. The window is empty. **So the passing window is empty by
construction, not by bad luck or bad hyperparameters.**

> 🔴 **Corrected in place 2026-08-12 (same session), arithmetic only — the
> conclusion is unchanged.** This paragraph first stated the cap as "a
> dictionary no larger than 500/0.70 = 714". That is the bound only in the
> boundary case `alive = 500` exactly; the correct cap is `C / 0.70` for the
> model's own saturation ceiling C, which is looser (823, not 714) and so
> makes the unsatisfiability claim *harder* to establish, not easier. It
> still holds: the first size reaching 500 alive atoms is 896 > 823. Caught
> by writing the argument as a test assertion
> (`tests/test_stage0_criteria.py::test_legacy_pair_is_unsatisfiable_for_a_saturating_model`),
> which failed against the 714 framing — an instance of §2.4 in the small,
> and the reason that test encodes the measured alive-per-dict table rather
> than a summary statistic of it. TimesFM shows the
opposite: alive tracks dict almost perfectly (511 / 640 / 765 / 893 / 1012 /
1258, dead 0.000–0.012), so the same two constants are trivially satisfiable
for it. The constants encode a hidden assumption — *alive-atom count scales
with dictionary size* — that holds at eff_dim 28.15 and fails at eff_dim
13.93.

**DECISION (2026-08-12) — the floor scales per model; the rate bar does
not.** `MIN_ALIVE` becomes `max(100, round(20 × eff_dim))` per source
(279 Chronos / 563 TimesFM); `MAX_DEAD_RATE = 0.30` and `MIN_FIDELITY =
0.70` stay global and unchanged. The reasoning is (11), not (10): a
criteria *pair* that no dictionary size can satisfy is measuring the
constants' interaction rather than the SAE, and that defect is visible
independently of which side it happens to fail on. The gate's actual job —
stopping dead-atom symmetry from faking a shared-vs-specific split — is
carried by the *fraction* alive, which is exactly what the untouched
`MAX_DEAD_RATE` measures; the absolute floor was only ever a guard against
a tiny dictionary trivially satisfying that fraction, and `20 × eff_dim`
guards it while scaling with the thing that actually bounds the atom count.
The `max(100, ...)` term keeps the guard from vanishing on a very
low-dimensional layer.

🔴 **This decision does not, by itself, flip Stage 0's checkbox, and that is
the point.** Under the new floor the Chronos baseline clears `alive` at
`dict=640` (419.4 ± 38.9 vs. 279) but still fails `dead ≤ 0.30`
(0.3447 ± 0.0607) at 4 of 5 seeds — so the change is demonstrably *not*
"whichever option makes the gate pass", the thing the prior paragraph
correctly ruled out. It relocates the candidate passing size downward to
`dict=512`, whose single-seed rate (0.2461) sits ~0.9 sd under the bar and
is therefore exactly the kind of number finding (8) warned against trusting.
A 5-seed `--grid pinch` replicate (dict 512 / 576 / 704) is running as this
is written; **Stage 0 stays `[ ]` until it lands** and a size clears both
criteria across seeds. ✅ **It landed — see the pinch Findings immediately
below. No size clears the criteria at any seed, for a new and different
reason than (11), and Stage 0 stays `[ ]`.**

**Findings — Stage 0 pinch replicate (2026-08-12, same session): the
unsatisfiability recurs one level up, between models rather than within one.
5 seeds × dict {512, 576, 704}, crosscoder plus both matched per-model
`TopKSAE` baselines at each size, all against the same 46382-row store at
k=48 / aux_k=64 / aux_coef=0.03125 / 60 epochs.** Artifacts:
`runs/crosscoder_stage0_bigdata/pinch_seed{0..4}.json`. Floors in force:
563 for the crosscoder and TimesFM, 279 for Chronos (the DECISION above).

| dict | artifact | dead rate | alive | fidelity (min over sources) | passes |
|---|---|---|---|---|---|
| 512 | crosscoder | 0.0754 ± 0.0740 | 473.4 ± 37.9 | 0.6981 ± 0.0418 | **0/5** |
| 512 | TimesFM baseline | 0.0086 ± 0.0064 | 507.6 ± 3.3 | 0.7780 ± 0.0057 | **0/5** |
| 512 | Chronos baseline | 0.2387 ± 0.0313 | 389.8 ± 16.0 | 0.8786 ± 0.0051 | 5/5 |
| 576 | crosscoder | 0.0722 ± 0.0644 | 534.4 ± 37.1 | 0.7128 ± 0.0137 | **0/5** |
| 576 | TimesFM baseline | 0.0094 ± 0.0069 | 570.6 ± 4.0 | 0.7845 ± 0.0042 | 5/5 |
| 576 | Chronos baseline | 0.3063 ± 0.0476 | 399.6 ± 27.4 | 0.8740 ± 0.0080 | 3/5 |
| 704 | crosscoder | 0.0693 ± 0.0561 | 655.2 ± 39.5 | 0.7055 ± 0.0357 | 4/5 |
| 704 | TimesFM baseline | 0.0131 ± 0.0097 | 694.8 ± 6.8 | 0.7931 ± 0.0039 | 5/5 |
| 704 | Chronos baseline | 0.3716 ± 0.0531 | 442.4 ± 37.4 | 0.8703 ± 0.0071 | **0/5** |

**(12) All three artifacts pass simultaneously at 0 of 5 seeds, at every
size — and the exit criteria require exactly that** ("all must hold, on the
crosscoder *and* on the matched per-model `TopKSAE` baseline at the same
data/budget"). The blocking bar is a *different one at each end of the
range*, which is the whole point: at 512 the crosscoder and TimesFM fail the
alive floor (a 512-atom dictionary cannot hold 563 alive atoms —
`structurally_cannot_pass` is `true`, not a training failure); at 704 they
both clear it comfortably and Chronos fails the untouched dead-rate bar.

**(13) The window is empty by ~37 atoms, and the three boundaries can be
located.** Interpolating each mean alive count linearly against dictionary
size: Chronos's rate bar (`alive ≥ 0.70 × dict`) has surplus +31.4 at 512 and
−3.6 at 576, crossing at **dict ≈ 569** — its ceiling. TimesFM's alive floor
(563) is crossed at **dict ≈ 568** — its floor, essentially the same point.
The crosscoder's own floor (also 563, since one dictionary serves both
sources) is crossed at **dict ≈ 606**, because it runs ~5% deader than the
TimesFM baseline at matched settings. So a shared size must satisfy
`dict ≥ 606` **and** `dict ≤ 569`. This is structurally the same defect as
(11) — two bars pulling in opposite directions with no overlap — but the
cause has moved: in (11) the two bars were both constraints on *one* model,
and the DECISION fixed that by letting the floor scale with each model's own
eff_dim. Here the surviving conflict is between *different models sharing
one dictionary size*, which no per-model constant can resolve, because the
size itself is the shared quantity.

**(14) Per-model dictionary sizing would close it, and that is the obvious
next move — but it is a second criteria change in one session, so it is
proposed, not taken.** Under the criteria's own words the baseline must be
"matched … at the same data/budget", and the harness currently implements
*budget* as including dictionary size. Read as rows/k/epochs matched but
dict sized per model, every artifact already has a passing size on record:
Chronos baseline at 512 (5/5), TimesFM baseline at 576 and 704 (5/5 each),
crosscoder at 704 (4/5). This is defensible on the same reasoning the
DECISION used — a fixed shared size is a constant that measures the models'
differing effective dimensionality rather than the SAE — but this session
has already moved one criterion, and moving a second one *in the direction
that makes the gate pass*, in the same sitting, on the same data, is exactly
the pattern finding (5) and the DECISION both went out of their way to
avoid. **Recorded as the recommended change with its evidence; not applied.
Stage 0 stays `[ ]`.**
✅ **Taken 2026-08-13, in a later session — see the second DECISION below.
The objection above was explicitly about doing it *in the same sitting*,
and it is discharged by the calendar rather than argued away; the decision
itself is justified on new evidence (finding (16)), not on the delay.**

**(15) The crosscoder's one failing seed at 704 is bimodal, not marginal —
which matters for how (14) should be read if it is ever taken.** The 4/5 is
not "four seeds squeaking over and one just under". Seeds 0/1/3/4 cluster
tightly (dead 0.0415 ± 0.0088, alive 674.8 ± 6.2, fidelity 0.7233 ± 0.0018
TimesFM / 0.7453 ± 0.0065 Chronos); seed 2 lands in a visibly different
basin (dead 0.1804, alive 577, fidelity 0.6687 / 0.6343 — **both** sources
collapse together, below the 0.70 bar). The same shape appears at 576, where
seed 0 alone shows dead 0.1997 and fidelity 0.6991 / 0.6867 against four
seeds at 0.71–0.76. So the aggregate `0.7055 ± 0.0357` at 704 is a mixture
of a tight good mode and one bad run, and quoting its sd as a noise band
would misdescribe it. A 1-in-5 joint-collapse rate is itself a real
stability finding about `CrosscoderSAE` at this budget, and it is a
prerequisite for Stage 1's ladder: a scorecard that trains one dictionary
per variant would attribute a bad basin to the variant.

**(16) The matched baseline's stated diagnostic job is complete, and it came
back inverted — which is the new evidence (14) was missing (2026-08-13).**
The exit criteria apply to the baselines "because the feasibility run showed
the per-model baseline hitting the same wall, which is what rules out 'the
crosscoder architecture is the problem' as an explanation"
(`run_crosscoder_stage0.py`'s own docstring). That premise no longer holds.
Reading the dead-rate column of (11)'s table down the dictionary axis, at
every matched size the **crosscoder is 3.2×–5.4× less dead than the Chronos
baseline it is matched against**: 0.0754 vs 0.2387 at 512, 0.0722 vs 0.3063
at 576, 0.0693 vs 0.3716 at 704. And the two move in *opposite directions*
as the dictionary grows — the crosscoder's dead rate falls monotonically
(0.0754 → 0.0722 → 0.0693) while the Chronos baseline's rises monotonically
(0.2387 → 0.3063 → 0.3716). So the baseline is not hitting the same wall;
it is hitting a different one, in the opposite direction, and the joint
dictionary is the *healthier* of the two artifacts at every size measured.
The question the baseline exists to answer — "is the crosscoder
architecture the problem?" — is therefore answered, decisively no, and
keeping it gated on a *shared* size no longer buys diagnostic information.
What it buys instead is finding (13)'s empty window: a shared size is a
shared bar on two artifacts whose alive counts behave qualitatively
differently (TimesFM's tracks the dictionary; Chronos's saturates near 576),
so it measures that difference rather than the SAE.

**DECISION (2026-08-13) — "matched" means matched *budget*, not matched
*dictionary size*; the matched-size run is retained as a control.** Stage 0's
gate now reads: the crosscoder clears all three numeric bars on its own
dictionary, **and** each per-model `TopKSAE` baseline clears them on *its
own* dictionary, with rows, `k`, epochs and AuxK settings identical across
all three. Dictionary size becomes a per-artifact hyperparameter. The three
numeric bars themselves — `MAX_DEAD_RATE = 0.30`, `MIN_FIDELITY = 0.70`,
`min_alive_for(...)` — are **untouched**, as is the requirement that all
three artifacts pass. Justification is (16), not (13): (13) established that
the shared-size criterion is unsatisfiable, but "unsatisfiable" alone is an
argument for changing *something*, not for changing *this*; (16) is what
identifies dictionary size as the part carrying no diagnostic weight.

🔴 **What this deliberately does not do.** It does not drop the matched-size
comparison — `--baseline-dict-sizes` left unset reproduces the previous
behaviour byte for byte, and every already-recorded row keeps its meaning
(`CLAUDE.md` §11.24). That comparison still answers a real and separate
question — *does joint training cost anything at identical budget?* — and
Stage 1's scorecard should keep reporting it as attribution evidence. It is
demoted from gate to control, not deleted. It also does not lower any bar:
the sizes the gate now hands the baselines are not chosen to be easy —
`dict=512` is the **only** size in the entire recorded sweep at which the
Chronos baseline passes at all, and the Chronos five-seed mean at 640 (dead
0.3447, alive 419) still fails on the untouched dead-rate bar under the new
criterion exactly as it did under the old one. `tests/
test_stage0_baseline_sizing.py::test_per_model_sizing_does_not_relax_any_
numeric_bar` pins both of those.

🔴 **And it does not flip Stage 0's checkbox on the numbers already in
hand.** Under the new criterion the two baselines are satisfied by sizes
with 5/5 records (TimesFM at 576 or 704, Chronos at 512), so the binding
constraint is now the crosscoder alone — and its best recorded multi-seed
result is 704 at **4 of 5**, with finding (15)'s bad basin, not 5 of 5.
Accepting 4/5 would be relaxing the gate by the back door, having just
declined to relax it by the front. The right move is the measurement the
single-seed record already suggests: the crosscoder at `dict=896` (dead
0.0112, alive 886, fidelity 0.7590 / 0.7782) and `dict=1024` (dead 0.0322,
alive 991, fidelity 0.7363 / 0.7348) both beat 704 on every axis at one
seed. A 5-seed `--grid gate` replicate over those two sizes, with each
baseline at its own size, is running as this is written. **Stage 0 stays
`[ ]` until it lands and a crosscoder size passes at every seed.**

**Mechanism (landed 2026-08-13).** `run_crosscoder_stage0.py` gains
`Row.baseline_dict_sizes` (per model, in model order; empty = the
crosscoder's own size), a `--baseline-dict-sizes` flag, a `gate` grid
(crosscoder at 896/1024), and a per-run baseline memo — under per-model
sizing two rows differing only in the crosscoder's dictionary train the
*identical* baseline, so memoizing it halves the sweep. Every stored row now
records `baseline_sizing: "own" | "matched"` and each baseline's own
`dict_size`, so an artifact stays interpretable after the criteria move
again. Nine new tests in `tests/test_stage0_baseline_sizing.py`, including
the memo exercised through the real training path on tiny CPU data (a memo
that served a result trained at a different size would silently turn the
control into the gate) and the satisfiability arithmetic above stated over
the measured pinch table.

**Findings — Stage 0 gate replicate (2026-08-13, same session): the gate is
met. `dict=1024` passes at 5 of 5 seeds on every criterion, both baselines
pass at their own sizes at every seed, and the main result is that a
*larger* dictionary makes the crosscoder more stable — the opposite of what
the per-model baselines do.** Run: `--grid gate --baseline-dict-sizes
576,512` at seeds 0–4 against `runs/crosscoder_stage0_bigdata` (46382
aligned rows), 11:33:03–11:38:39 EDT, ~65–70 s/seed, all five exit 0, no
warnings or OOMs. Numbers below were read out of the five
`gate_seed*.json` artifacts directly rather than taken from the run's own
summary (`CLAUDE.md` §2.4).

**(17a) `dict=1024`, the crosscoder — 5 of 5 pass.** `l0_actual` is exactly
48.0 at every seed; the alive floor is 563.

| seed | dead rate | alive | fidelity TimesFM | fidelity Chronos-T5-Base |
|---|---|---|---|---|
| 0 | 0.0322265625 | 991 | 0.7362681170891777 | 0.7348360563535948 |
| 1 | 0.1845703125 | 835 | 0.7293552719423504 | 0.7242834675826152 |
| 2 | 0.0224609375 | 1001 | 0.7591256489070557 | 0.775859251233562 |
| 3 | 0.013671875 | 1010 | 0.7619046926234602 | 0.7829349747481011 |
| 4 | 0.0458984375 | 977 | 0.7524316570146609 | 0.7661921079765164 |

Worst seed on each axis: dead 0.1846 (bar 0.30), alive 835 (floor 563),
fidelity 0.7243 (bar 0.70) — all three clear with margin, and all three
worst cases are seed 1, so this is one comparatively bad basin rather than
three independent near-misses.

**(17b) `dict=896` — 3 of 5, and the two failures are fidelity-only.**

| seed | dead rate | alive | fid TimesFM | fid Chronos | passes |
|---|---|---|---|---|---|
| 0 | 0.011160714285714302 | 886 | 0.758954941530268 | 0.778220929083343 | ✅ |
| 1 | 0.2689732142857143 | 655 | 0.7009609799524694 | 0.6785655373388639 | ❌ fidelity |
| 2 | 0.1026785714285714 | 804 | 0.7302954598508289 | 0.7444827083669372 | ✅ |
| 3 | 0.1529017857142857 | 759 | 0.7368046102305345 | 0.7472061591243 | ✅ |
| 4 | 0.2712053571428571 | 653 | 0.7067438768307592 | 0.6592242945867786 | ❌ fidelity |

`dead_ok` and `alive_ok` are **True** in both failing rows — 896 fails on
reconstruction quality alone, not on deadness. Note the shape: it is the
same seeds (1 and 4) that come out worst at 896 and, at 1024, seed 1 that
comes out worst again — seed effects survive a dictionary-size change, so
these are basins of the joint optimization rather than noise in the
measurement.

**(17c) The main result: for the crosscoder, more atoms is more stable —
which is the opposite of the per-model baselines.** Seeds 1 and 4 fail at
896 and pass at 1024, and the whole distribution shifts (worst dead rate
0.271 → 0.185, worst fidelity 0.659 → 0.724). This inverts the intuition
every earlier finding here was built on: for the *baselines*, growing the
dictionary raises the dead rate (finding (16): Chronos 0.2387 → 0.3063 →
0.3716 across 512/576/704), which is why the shared-size criterion kept
squeezing the crosscoder downward into sizes that starved it. Two artifacts
whose failure modes move in opposite directions cannot be sized by one
number — which is (16)'s DECISION restated, now with the crosscoder's own
direction measured rather than inferred.

**(17d) Both baselines pass, at every seed, and the memo is exact.** The
baselines are bit-identical across the 896 and 1024 rows within each seed
for both models at all five seeds, confirming the memo serves only
genuinely identical training.

| model @ size | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| TimesFM @576 dead | 0.02083333395421505 | 0.0034722222480922937 | 0.0052083334885537624 | 0.0034722222480922937 | 0.013888888992369175 |
| TimesFM @576 alive | 564 | 574 | 573 | 574 | 568 |
| TimesFM @576 fidelity | 0.7867365389504536 | 0.7868663881172161 | 0.7863370802222039 | 0.7865906818637007 | 0.7761479623324176 |
| Chronos @512 dead | 0.24609375 | 0.23828125 | 0.29296875 | 0.205078125 | 0.2109375 |
| Chronos @512 alive | 386 | 390 | 362 | 407 | 404 |
| Chronos @512 fidelity | 0.8810526051531897 | 0.8697087726206614 | 0.8760318918491139 | 0.8833256664626026 | 0.8827168217321761 |

`baseline_sizing == "own"` on every row of every seed.

⚠️ **(17e) The honest margin note: TimesFM's baseline clears its alive floor
by 1–11 atoms.** At `dict=576` the floor is 563, and the measured alive
counts are 564/574/573/574/568 — seed 0 passes by a single atom. A sixth
seed could plausibly land below. This was noticed and **640 was considered
and rejected**, because moving there would be a relaxation dressed as
robustness: at 576 the 563 floor demands **97.7%** of atoms alive, at 640 it
demands only 88.0%. 576 is the smallest measured size above the floor, so it
is the *strictest* available choice, and the thin margin is the price of
that strictness rather than a sign of a lucky draw. Recorded here rather
than engineered away — if a rerun does drop TimesFM below 563 at 576, the
correct reading is "this baseline sits exactly at its bar", not "the gate
was wrong".

⚠️ **(17f) What this result is not.** One layer pair, one corpus, one store,
one `k`. It says a joint dictionary over these two models at their L1
peak-CKA layers can be trained to be alive and faithful — the blocking
question Stage 0 was created to answer. It says nothing yet about whether
its features are *shared* in any meaningful sense; per this section's Stage
0 preamble, `relative_decoder_norm` must still not be quoted until Stage 1's
scorecard measures it on a dictionary that is alive, which is only now
available.

**Stage 0's checkbox is flipped.** All four exit criteria are met: dead rate
≤ 0.30, alive ≥ `min_alive_for` (563 crosscoder / 563 TimesFM / 279
Chronos), per-source fidelity ≥ 0.70 on the crosscoder **and** both
baselines, and a committed config — `tsfm_lens/configs/
crosscoder_stage0_gate.yaml`, executable via `--params`, carrying seed 0's
full-precision values *and* the five-seed worst-case bounds so it cannot be
misread as a 1/1 result. `configs/crosscoder_stage0_winner.yaml` is kept
unchanged as the matched-size control, with a correction note replacing its
now-stale "which is why Stage 0's checkbox is still open" comment. Seven new
tests in `tests/test_stage0_gate_config.py` check the config parses into
exactly the swept row, that `baseline_dict_sizes` sits at top level (under
`train:` it would train correctly while the artifact recorded
`baseline_sizing: matched`), that the committed baseline sizes are the
strict ones, and that both the seed-0 values and the five-seed bounds match
the artifacts — including that 896's 3-of-5 failure is recorded rather than
omitted. **Stage 1 is unblocked.**

---

#### Stage 1 — `[x]` The fixed scorecard and the known-answer validation ladder

> ✅ **Mechanism complete 2026-08-17** (`sae/crosscoder_eval.py`,
> `run_crosscoder_ladder.py`, `tests/test_crosscoder_eval.py`). **Verdict: a
> negative result for V1**, not a blocker — see the Findings blocks below.
> 1a–1c are kept verbatim because they are the *pre-registration*: they were
> fixed before any variant existed, and V2 must be scored against them unchanged.

**1a. `score_variant(sae, sources, gt_table, device) -> dict`** — one
function, eight numbers, identical for every variant:

| Metric | Source | What it catches |
|---|---|---|
| `fidelity_per_source` | `crosscoder.py::per_source_fidelity` (exists) | One source being sacrificed for the other |
| `fidelity_gap` | `max − min` of the above | The §11.19 domination failure, as a single number |
| `dead_feature_rate`, `n_alive` | `crosscoder.py` (exists) | Stage 0's gate, re-checked per variant |
| `l0_actual` | mean nonzero features per row | Sanity vs. the configured `k`; a BatchTopK variant's L0 is *not* `k` |
| `shared_specific_split` | `classify_features(rel_norm[alive_mask])` (exists — **must** be alive-masked) | The founding-question answer |
| `gt_alignment_shared` | `sae/ground_truth.py`, restricted to shared atoms | Are shared atoms *interpretable*, or just numerically shared? |
| `gt_alignment_specific_a/b` | same, restricted to each specific set | What does each model uniquely encode? |
| `forecast_preservation` | `sae/eval.py::forecast_preservation(..., granularity="token")` per source | Is the reconstruction usable for causal work (E15's fix applies here too) |

Reuse every one of these from existing modules. Write no new statistics —
`CLAUDE.md` §2.2. The only new code is the restriction of ground-truth
alignment to an atom subset, which is an index mask, not an algorithm.

**1b. The validation ladder — five known-answer pairs, run for every
variant.** This is the part that makes the comparison trustworthy, and it is
this repo's unfair advantage (§2.1) applied to SAE methodology itself: every
other crosscoder evaluation in the literature has to argue that its
shared/specific split is meaningful; this repo can *construct pairs whose
answer is known in advance* and check.

| Rung | Pair | Known answer | Disqualifies a variant if |
|---|---|---|---|
| **L-A** identity | Model A layer ℓ vs. **the same activations again** | 100% shared, by construction | `frac_shared < 0.95`. The metric itself is broken; stop and fix it before reading anything else. |
| **L-B** hard null | Real model A vs. **its own `random_init` twin** at the same layer | Any "shared" atom is architecture + input statistics, *not* learned structure. Gives the **shared-fraction floor.** | A real pair's `frac_shared` does not exceed this floor. Uses the already-built `random_init` mechanism (§6.2, `configs/null_*_random.yaml`) — **zero new code.** |
| **L-C** monotone | Model A layer ℓ vs. model A layer ℓ+δ, for δ = 1, 2, 4, 8 | Shared fraction should be high at δ=1 and **decay monotonically** with δ | Non-monotone or flat. Also doubles as the cross-*layer* crosscoder the original crosscoder paper is actually about. |
| **L-D** the real question | TimesFM vs. Chronos-T5-Base at the L1 peak-CKA pair | Unknown — this is the experiment | (nothing; this is the readout) |
| **L-E** planted | Synthetic sources with a constructed shared/A-only/B-only cause structure | Exact, by construction | Extends the planted-cause test already in `tests/test_crosscoder.py` into a scored **precision/recall of shared-atom recovery**, rather than a pass/fail assertion. Runs on CPU in seconds — put it in the test suite, not just the sweep. |

> **Why L-B is the single most important rung.** §6.3's Findings below are a
> live, expensive demonstration of exactly the mistake it prevents: a
> similarity metric that looked decisive turned out to be dominated by
> architecture match, discovered only when an architecture-matched
> zero-training control was finally run. A crosscoder's shared-fraction is
> the same class of number. Do not report one without its L-B floor beside
> it.

**1c. Pre-register the decision rule now, before any variant exists.** A
variant wins if, at matched data/compute budget:

1. It passes L-A (`frac_shared ≥ 0.95`) and L-E (shared-atom recovery
   F1 ≥ 0.8) — **hard prerequisites, not tiebreakers**; and
2. On L-D it achieves a higher `gt_alignment_shared` than V0's post-hoc
   matched features at equal `n_alive`, **and** a `frac_shared` that clears
   the L-B floor by a margin whose bootstrap CI excludes zero; and
3. `fidelity_gap ≤ 0.10` and `dead_feature_rate ≤ 0.30`.

Ties break toward the simpler mechanism (§2.5). If no variant beats V0 on
(2), **that is the result** — write it up as "post-hoc matching of
independent dictionaries is sufficient; joint training buys nothing here,"
which is a genuine and useful negative finding, not a failure.

> **Findings (2026-08-13) — the scorecard and all five rungs are built and
> have run against real checkpoints. V1 does not pass its own pre-registered
> decision rule, and the reason is L-B.**
>
> **Built.** `tsfm_lens/sae/crosscoder_eval.py` (`score_variant`,
> `atom_subset_alignment`, `source_views`, `l0_actual`, `atom_buckets`,
> `shared_fraction_margin`, `planted_sources`, `shared_recovery_score`,
> `monotone_decay`, `variant_verdict`) + `run_crosscoder_ladder.py` +
> `tests/test_crosscoder_eval.py` (11 tests). Every statistic is reused from
> an existing module per §2.2 — the only genuinely new code is
> `atom_subset_alignment`'s index mask, and it reports **global** atom ids so
> a subset-local feature index can never be quoted as if it were a
> dictionary position.
>
> **The run.** `configs/crosscoder_stage0_gate.yaml`'s committed parameters
> (dict 1024, k=48, 60 epochs, aux_k 64 / aux_coef 0.03125, seed 0) against
> `runs/medium_run_chronos_base`, pair TimesFM `stacked_xf.4` ↔
> Chronos-T5-Base `encoder.block.10`, 4608 aligned rows. Wall clock **37 s**
> for all seven trainings — the ~1 h estimate in the effort table below was
> two orders of magnitude pessimistic. Artifact:
> `runs/medium_run_chronos_base/crosscoder/ladder.json`.
>
> | Rung | `frac_shared` | `n_alive` | dead | fidelity per source | gap |
> |---|---|---|---|---|---|
> | **L-A** identity | **1.0000** (679 shared, 0 specific either side) | 679 | 0.337 | 0.7473 / 0.7466 | 0.0006 |
> | **L-E** planted | 0.3333 (2 shared / 2 A / 2 B — the planted answer exactly) | 6 | 0.625 | 0.9986 / 0.9986 | 0.00001 |
> | **L-C** δ=1 (`stacked_xf.4`↔`.6`) | 0.9986 | 708 | 0.309 | 0.7514 / 0.7212 | 0.0303 |
> | **L-C** δ=2 (↔`.8`) | 0.9986 | 711 | 0.306 | 0.7424 / 0.7023 | 0.0401 |
> | **L-C** δ=4 (↔`.12`) | 0.9451 | 583 | 0.431 | 0.7163 / 0.6507 | 0.0657 |
> | **L-D** the real pair | **0.8453** (579 shared / 104 TimesFM-only / 2 Chronos-only) | 685 | 0.331 | 0.7416 / 0.7455 | 0.0039 |
> | **L-B** hard null | **0.9738** (743 / 19 / 1) | 763 | 0.255 | 0.6825 / 0.9654 | 0.2828 |
>
> **1. The metric works. L-A and L-E both pass cleanly.** L-A returns exactly
> 1.000 shared with zero specific atoms on either side — the identity rung's
> whole job is to catch a broken metric before anything else is read, and it
> does not fire. L-E recovers the planted six-cause structure at
> **precision 1.0 / recall 1.0 / F1 1.0**, well past the pre-registered
> F1 ≥ 0.8 bar. So the two hard prerequisites of clause 1 are met, and
> nothing below can be dismissed as a measurement artifact of the split
> itself.
>
> **2. 🔴 L-B fails, and it fails in the direction that matters.** The
> pre-registered rule asks whether the real pair's `frac_shared` *exceeds*
> the untrained-twin floor. It does not — it is **lower**: real 0.8453 vs.
> null 0.9738, diff **−0.1285**, bootstrap CI **[−0.1601, −0.1007]**,
> p=0.002 (500 resamples, unpaired — the two runs' atoms are not in
> correspondence, and the check records `paired: false` rather than
> pretending otherwise). TimesFM against a randomly-initialized copy of
> itself reads as **more** shared than TimesFM against Chronos. This is the
> §6.3 pattern recurring exactly as the "why L-B is the single most important
> rung" callout above predicted: architecture match dominates the similarity
> number, and a shared-fraction reported without this floor beside it would
> have read as a strong positive result. **`frac_shared = 0.845` must never
> be quoted on its own.** CLAUDE.md's standing ban on quoting
> `relative_decoder_norm` is therefore **not lifted** by this run — it is
> replaced by a stronger statement: the quantity is measurable, and at V1 it
> does not distinguish learned cross-model structure from architecture.
>
> **3. L-C decays, but is not monotone, and one rung could not be run.**
> Values 0.9986 (δ=1), 0.9986 (δ=2), 0.9451 (δ=4); `monotone: false`,
> `flat: false`, range 0.053. The non-monotonicity is a **6×10⁻⁶ inversion
> between δ=1 and δ=2** — arithmetically a tie, not a reversal, and the
> honest reading is that the shared fraction is saturated at both. The real
> decay only appears at δ=4. **δ=8 was skipped**: it exceeds TimesFM's
> captured depth at `capture_layer_stride: 2`, and this is recorded in the
> artifact as `deltas_beyond_captured_depth: [8]` plus a log line, rather
> than silently omitted (§2.5). Note the ladder's own limitation this
> exposes: L-C's δ is in *captured positions*, so at stride 2 it is a
> 2-block step, and a run wanting the full δ=1..8 range needs stride 1.
> The fidelity gap grows monotonically with δ (0.030 → 0.040 → 0.066) even
> where the shared fraction is flat, which is the more sensitive signal of
> the two.
>
> **4. `dead_feature_rate` fails clause 3, and the cause is the store, not
> the settings.** Every real rung sits at 0.25–0.43 dead against the ≤0.30
> bar, and L-D specifically at 0.331. These are the *same committed
> parameters* that cleared Stage 0's gate at **3.2% dead** — the difference
> is that Stage 0's gate ran on `runs/crosscoder_stage0_bigdata`'s **46,382**
> rows and this ladder ran on `medium_run_chronos_base`'s **4,608**, a 10×
> reduction in training data at unchanged dictionary size. **So Stage 0's
> dead-rate result does not transfer across row counts**, which is a real
> and previously unstated limit on that gate's finding — it was established
> at one data budget and is being read here as if it were a property of the
> configuration. Either the ladder should be re-run against the bigdata
> store, or the gate's numbers should carry their row count everywhere they
> are quoted. Recommend both.
>
> **5. L-D's ground-truth alignment is real but modest, and lopsided.**
> Shared atoms: 279 of 579 matched, `mean_abs_rho_matched` **0.346**;
> TimesFM-specific: 75 of 104, **0.307**; Chronos-specific: 2 of 2, **0.267**.
> Top shared feature ρ=0.930 against `tier_realism_stress`, then 0.813
> (same field) and 0.790 against `archetype_nonsinusoidal_seasonal`. Two
> cautions before reading these as interpretability wins: (a) these are
> *best-of-~30-fields* matches, and the permutation null
> (`ground_truth.py::permutation_null_alignment`, §16 E9) has **not** been
> run against them here, so the headline ρ carries an unquantified
> search-inflation component; (b) the split is extremely asymmetric — 104
> TimesFM-specific atoms against **2** Chronos-specific ones. That asymmetry
> is not obviously about the models: Chronos's captured surface is its
> encoder only (§12), and its per-source fidelity on L-D (0.745) is no worse
> than TimesFM's (0.742), so the honest statement is that V1 assigns almost
> no private capacity to Chronos and this run cannot say why.
>
> **6. `variant_verdict` reports `passes: false` with one clause unrun,
> rather than scoring what it can.** `beats_v0_gt_alignment` is `null` and
> appears in `unrun: ["beats_v0_gt_alignment"]` — V0's post-hoc-matched
> baseline has not been put through `score_variant` at equal `n_alive`, so
> clause 2's first half is unmeasured. The verdict function refuses to pass
> on an unrun clause by construction (pinned by a test), so V1's
> `passes: false` is *overdetermined*: it fails L-B outright and clause 2's
> other half is not yet measured. `forecast_preservation` is likewise
> `not_supplied` on every rung — it needs a live model, adapter, store and
> corpus, and the ladder runs on stored activations only.
>
> **What this means for Stage 2.** The decision rule's own escape clause is
> now the live hypothesis: *"If no variant beats V0 on (2), that is the
> result."* V1's shared/specific split does not clear its architecture floor,
> so the next work is not "try V2 because it is next in the list" — it is
> **(a)** run V0 through `score_variant` so clause 2 stops being unrun,
> **(b)** re-run the ladder at the bigdata row count so the dead-rate
> confound is removed, and **(c)** build **V4 (latent scaling)** ahead of V2
> and V3, since it is the published diagnostic for exactly the failure L-B
> just exhibited and is the cheapest item in Stage 2. V2's stated purpose —
> reclassifying V1's false model-specific latents as shared — would, on this
> evidence, move `frac_shared` *further toward the null*, which is the wrong
> direction to chase without V4's β test to say which specific atoms are
> genuine first.
>
> **Not done:** one seed, one layer pair, one corpus, one variant, one row
> count. No V0 scorecard, no permutation null on the gt-alignment numbers,
> no forecast-preservation on any rung, and δ=8 unmeasured. The checkbox
> stays `[ ]` — Stage 1's deliverable is the scorecard *and* a decision rule
> that has been exercised against V0, and half of clause 2 has never run.

> **Findings — 2026-08-13, next action (a): V0 is scored, and clause 2 says
> the crosscoder does not beat it.** `run_crosscoder_ladder.py --v0` against
> `runs/medium_run_chronos_base` (4608 rows, TimesFM `stacked_xf.4` ↔
> Chronos-T5-Base `encoder.block.10`, `configs/crosscoder_stage0_gate.yaml`,
> seed 0), written to `crosscoder/ladder_v0.json` beside — not over — the
> prior artifact. Every rung reproduces its earlier value bit for bit; only
> the V0 row and the verdict are new.
>
> **1. The headline.** `gt_alignment_shared`: V1 **0.3456** (279
> ground-truth-matched features among its 579 shared atoms) against V0
> **0.3658** (190 of 190). Difference **−0.0202, 95% CI [−0.0489, +0.0119],
> p=0.208** → `beats_v0: false`, and the stricter
> `beats_v0_ci_excludes_zero` is false as well. So clause 2's first half is
> now *measured and failed*, not unrun. Read it as a null rather than a
> defeat: the CI comfortably spans zero, so what this run establishes is
> that joint training buys **no detectable ground-truth-alignment advantage
> over post-hoc matching at this row count**, in either direction.
>
> **2. `variant_verdict` now reports `unrun: []`.** That was the entire point
> of this action. V1 still fails — `clears_l_b_floor: false` and
> `dead_rate_ok: false` (0.331 > the 0.30 bar) — but `passes: false` is no
> longer partly an artifact of an unevaluated clause.
>
> **3. Scoring V0 as previously published would have inverted the
> conclusion, by a selection effect alone.** `best_ground_truth_matches`
> truncated its returned `features` list to the top 50 by |ρ|, and
> `sae/matching.py` reads that list as V0's *candidate pool* — a selection on
> the exact quantity clause 2 compares. Both configurations are now recorded
> side by side: on the full pool V0 scores 0.3658 over 190 matched features;
> on the as-published top-50 pool it scores **0.5724** over 44 — a **+0.207
> (57%) inflation** that would have handed V0 an apparently decisive win over
> V1's 0.3456. Its `frac_shared` inflates the same way, 0.660 → 0.880. Fixed
> with a `top_features` parameter defaulting to the value every existing
> caller was hardcoded to, so no recorded artifact changes; V0 alone passes
> `top_features=0`. The `as_published` block stays in the output so the size
> of the effect is visible rather than argued.
>
> **4. A fallback that was invisible in the numbers flipped the verdict, and
> is why the first V0 run is not the one reported above.** `row_from_params`
> reads only a params file's `train:` block, while `baseline_dict_sizes`
> sits top level *by design* (`tests/test_stage0_gate_config.py` pins that).
> The ladder did not lift it across, so V0's first run silently trained both
> baselines at the crosscoder's shared 1024 — the matched-size confound
> §11.29 records, and precisely what `run_v0`'s own docstring claimed to
> avoid. At that sizing V0 scores 0.3243 and V1 *wins* by +0.0213 [−0.0035,
> +0.0496]; at Stage 0's committed 576/512 it scores 0.3658 and V1 *loses* by
> −0.0202. **Both CIs span zero, so neither sizing resolves clause 2** — but
> the point estimate, which is what the pre-registered rule reads, changes
> sign. Fixed, and `baseline_sizing` (`own`/`matched`, the same field Stage 0
> artifacts use) is now written into the V0 row so the two runs can never be
> confused for one another again.
>
> **5. V0's supporting numbers**, at `baseline_sizing: own`: dictionaries
> 576/512, alive **504/292** (n_alive 796 vs V1's 685), dead rate 0.125 /
> 0.430 (0.268 pooled — better than V1's 0.331), fidelity 0.742/0.829 (gap
> 0.087, inside the 0.10 bar), measured L0 48.0/48.0. Matching: 288 A-side
> candidates, 201 B-side, **190 matched** at |corr| ≥ 0.3 → `frac_shared`
> 0.660. The 98 unmatched A-side features score 0.2514, below the matched
> 0.3658 — post-hoc matching does preferentially find the better-aligned
> features, which is the sense in which V0 is a real opponent rather than a
> straw man.
>
> **6. One asymmetry to keep in view when reading (1).** V1's mean is over
> 279 of its 579 shared atoms — the 48% that found a ground-truth field —
> whereas V0's 190 are 190 of 190, because V0's candidate pool *is* the
> ground-truth-matched set. Both means are therefore over
> ground-truth-matched features and are like-for-like in kind, but V1's
> shared set contains 300 atoms that matched no field and contribute
> nothing; V0's construction cannot contain such atoms at all. This is
> inherent to V0 having no native shared/specific notion, and is why both
> analog definitions are emitted into the artifact as strings
> (`V0_FRAC_SHARED_DEF`, `V0_GT_ALIGNMENT_DEF`) rather than left to a
> reader to assume.
>
> **7. Two shared-code docstrings were wrong and are corrected.**
> `TopKSAE.sparsify`/`encode` both claimed "exactly k nonzero per row". The
> ReLU precedes the top-k, so a row with fewer than k positive
> pre-activations keeps fewer — measured on a small planted pair, one
> dictionary came back at L0 2.88 against k=3. Harmless here (both real
> dictionaries measure exactly 48.0), but it is the reason `l0_actual` is
> measured rather than copied from the config, and a variant that sets
> sparsity another way would have inherited a false invariant.
>
> **What this does and does not settle.** The decision rule's escape clause —
> *"If no variant beats V0 on (2), that is the result"* — is now the leading
> reading of the evidence, but it is **not yet callable**: one seed, one
> layer pair, one corpus, 4608 rows. Given finding (4), a single point
> estimate whose sign moves with a sizing choice and whose CI spans zero in
> both configurations is not enough to write up a negative result. Next
> actions are unchanged in order but sharper in purpose: **(b)** re-run at
> `runs/crosscoder_stage0_bigdata`'s 46382 rows — which removes the dead-rate
> confound *and* narrows this CI, the two things that currently block a
> verdict; then **(c)** V4. A seed replicate of the V0 comparison is now
> worth having too, on the same reasoning as §13's SAE noise floor: a ±0.02
> difference is well inside the seed-to-seed spread that item measured for
> single-dictionary training.
>
> **Tests.** 15 new in `tests/test_v0_scorecard.py` — that the `top_features`
> default reproduces every existing caller's list exactly, that truncating by
> |ρ| provably inflates the mean (the selection effect finding 3 defends
> against), that the headline alignment was always over the full population
> so the bootstrap resamples the same set the point estimate summarizes, that
> V0's analogs carry their definitions, that an unrun V0 still fails the rule
> closed, that a scored one resolves it in both directions, that the margin
> reports the point estimate the rule pre-registered *and* the stricter CI
> reading beside it, that `too_few_features` reports a status rather than
> fabricating a verdict, that measured L0 can sit below k, and that the
> ladder lifts the gate config's top-level baseline sizes (finding 4).
> Full `tsfm_lens` suite green at **393 passed**. The checkbox stays `[ ]`:
> clause 2 has now been *exercised*, which was action (a), but Stage 1's
> deliverable is a rule exercised at a row count that can resolve it.

> **Findings — action (b), the ladder at the bigdata row count (2026-08-13).**
> `run_crosscoder_ladder.py --params configs/crosscoder_stage0_gate.yaml --run
> runs/crosscoder_stage0_bigdata --null-run runs/null_timesfm_random --v0`,
> written to `runs/crosscoder_stage0_bigdata/crosscoder/ladder_v0.json` beside
> — not over — the 4,608-row `ladder_v0.json`. **46,382 rows**, the same store
> Stage 0's gate closed on, at the same pair (TimesFM `stacked_xf.4` ↔
> Chronos-T5-Base `encoder.block.10`) and the same committed training config
> (dict 1024, k 48, 60 epochs, aux_k 64 / coef 0.03125, seed 0). Wall clock
> **150 s** — the 10× row increase cost ~1× time, because the run is dominated
> by per-epoch dictionary work rather than by the row scan.
>
> 1. 🔴 **The dead-rate failure was entirely a row-count artifact, and it is
>    gone.** L-D `dead_feature_rate` **0.0322265625** (991 of 1024 alive)
>    against **0.331** at 4,608 rows — landing on Stage 0's own 3.2% at the
>    same store, which is the cleanest possible confirmation that the two
>    measurements now agree because they finally share a corpus size.
>    `verdict.dead_rate_ok` is `true`, and the "Stage 0's 3.2% does not
>    transfer across row counts" caveat recorded above is now explained rather
>    than merely observed: it does transfer, at the row count it was measured
>    at.
> 2. 🔴 **Clause 2 resolves — and it resolves *against* the crosscoder.** V1
>    `mean_abs_rho_matched` **0.2786869234414469** vs V0's
>    **0.32036928728167674**, diff **−0.04168236384022983**, 95% CI
>    **[−0.06928870407774981, −0.013676070745315533]**, **p = 0.002**,
>    `beats_v0: false`. At 4,608 rows this was −0.0202 with a CI spanning zero;
>    ten times the rows did exactly what (b) predicted — narrowed it — but the
>    interval moved *off* zero on the losing side. The joint dictionary's
>    ground-truth alignment is now **significantly worse** than two independent
>    per-model dictionaries matched post hoc. **This is the escape clause
>    firing**: "if no variant beats V0, that is the result." It is a real
>    negative finding about V1, not a failure of the ladder.
>    ⚠️ **Read `beats_v0_ci_excludes_zero: false` carefully** — that field asks
>    whether the CI excludes zero *on the winning side*, and here the CI
>    excludes zero on the losing side. The two are not the same claim and the
>    field name alone invites the wrong one.
>    ⚠️ **One caveat on the comparison's fairness**: clause 2 pre-registered
>    "at equal `n_alive`", and the two are **991 (V1) vs 950 (V0)** — a 4.3%
>    gap in the crosscoder's favour, so the deficit is not explained by V1
>    having fewer atoms to work with.
> 3. **L-B did not run, so Stage 1 still cannot close — for a new reason.**
>    `checks["L-B"] = {"status": "row_count_mismatch", "n_real": 46382,
>    "n_null": 4608}`; the ladder refused to pair a 46,382-row real run against
>    a 4,608-row null rather than comparing across different corpora, and said
>    so in the log. `verdict.clears_l_b_floor` is `null`, `unrun:
>    ["clears_l_b_floor"]`, `passes: false`. The tri-state convention did its
>    job — an unrun clause forces a non-pass rather than being read as
>    satisfied — but it means **the L-B failure recorded above (real 0.845 vs
>    null 0.974) has neither been reproduced nor refuted at this row count**.
>    Fixing it needs a `null_timesfm_random` extraction at 46,382 rows, which
>    is a real extraction run, not a re-analysis.
> 4. **L-C did not run either**, and this one is structural rather than
>    fixable by an extraction: `{"status": "too_few_deltas", "n_deltas": 0,
>    "deltas_beyond_captured_depth": [1, 2, 4, 8]}` — every configured layer
>    offset exceeds TimesFM's captured depth at this pair. The layer-offset
>    decay rung needs deltas that fit inside the captured stack, i.e. either a
>    stride-1 capture or smaller deltas.
> 5. **The metric itself stays sound at scale.** L-A identity `frac_shared`
>    **0.9970703125** (1024 of 1024 alive, dead 0.0, fidelity gap 9.77e-05) —
>    marginally below the 1.000 measured at 4,608 rows but still far past
>    clause 1's 0.95 bar; L-E planted recovery precision **1.0** / recall
>    **1.0** / **F1 1.0**. Clause 1's two hard prerequisites both hold.
> 6. **The selection effect finding 3 defends against gets *larger* with more
>    rows, not smaller.** V0's unrestricted pool: `frac_shared`
>    **0.5804020100502513** over 231 matched of 398 candidates,
>    `mean_abs_rho_matched` **0.32036928728167674**. The same V0 scored on the
>    as-published top-50-by-|ρ| pool: `frac_shared` **0.98** and
>    `mean_abs_rho_matched` **0.551868638801926** — a **+72%** inflation of the
>    exact quantity clause 2 compares, against +57% at 4,608 rows. Had the
>    top-50 cap not been removed, V0 would have "won" clause 2 by a margin
>    manufactured entirely by ranking on the comparison metric.
> 7. **L-D's shared fraction moved down with the row count**:
>    **0.7769929364278506** (770 shared / 206 A-specific / 15 B-specific over
>    991 alive) against 0.845 at 4,608 rows. Still unquotable without its L-B
>    floor, and finding 3 means this run supplies no such floor.
>
> **What this settles and what it does not.** Action (b) did what it was
> elevated to do: the dead-rate confound is removed (finding 1) and clause 2 is
> resolved (finding 2). Stage 1's **rule has now returned a real verdict on
> V1 — it loses to the post-hoc-matching control on ground-truth alignment,
> decisively.** What (b) did not do is close Stage 1, because L-B and L-C both
> went unrun, and a ladder with two silent rungs is not the ladder. The
> checkbox stays `[ ]`. Revised next actions: **(b2)** extract
> `null_timesfm_random` at 46,382 rows so L-B is answerable at the row count
> everything else now uses; **(b3)** give L-C deltas that fit the captured
> depth; then **(c)** V4, whose case is now stronger — finding 2 says the joint
> dictionary is losing real alignment, and latent scaling is the published
> diagnostic for exactly that.
>
> **Update (2026-08-17): (b2) and (b3) both done — the ladder now runs all
> five rungs at 46,382 rows, live checkpoints, no silent rung. ✅ Checkbox
> moves to `[x]` for "the scorecard and ladder run end-to-end and return a
> verdict" — the verdict itself stays the negative finding recorded above,
> not a blocker to close on.**
>
> **(b2).** `configs/crosscoder_stage0_null.yaml` (real `TimesFM` vs. its own
> `random_init` twin, same corpus/context_len 448/seed as
> `crosscoder_stage0.yaml`) turned out to already exist and already be built
> — `runs/crosscoder_stage0_null` has 3313 series / 46,382 aligned rows,
> matching the real run's population exactly. Verified, not assumed
> (`CLAUDE.md` §11.24's own lesson): `ActivationStore.load("TimesFM",
> "stacked_xf.4", level="window")` on the null store and on
> `runs/crosscoder_stage0_bigdata` returned `np.array_equal(...) == True` —
> bit-exact, confirming the null twin's context/corpus/seed genuinely match
> the real run rather than merely having the same row *count*.
>
> **(b3).** New `configs/crosscoder_stage0_layers.yaml`: TimesFM captured at
> five layers (`stacked_xf.{4,5,6,8,12}`) instead of one, same corpus/
> context_len/seed, Chronos unchanged at `encoder.block.10`. Captured-position
> deltas 1,2,3,4 land on real block gaps 1,2,4,8 — the geometric spacing L-C's
> design wants — which is why the run must be scored with `--deltas 1,2,3,4`,
> not the ladder's default `--deltas 1,2,4,8` (that flag would silently read
> block gaps 1,2,8 and drop a rung against this layer set — the config's own
> header names this as a `CLAUDE.md` §11.24-class trap). Extraction ran in
> under a minute (104 TimesFM batches, 139 Chronos batches); alignment checks
> came back exactly as predicted — TimesFM 1.00, Chronos 0.50 (§11.26's
> known context-length-448 calibration artifact, not a defect). Row-identity
> verified before trusting anything against it: both shared layers
> (`TimesFM/stacked_xf.4`, `Chronos-T5-Base/encoder.block.10`) came back
> `np.array_equal == True` against `runs/crosscoder_stage0_bigdata`.
>
> **The completed ladder run**
> (`run_crosscoder_ladder.py --params configs/crosscoder_stage0_gate.yaml
> --run runs/crosscoder_stage0_bigdata_layers --null-run
> runs/crosscoder_stage0_null --deltas 1,2,3,4 --v0 --seed 0`, artifact
> `runs/crosscoder_stage0_bigdata_layers/crosscoder/ladder.json`):
>
> 1. **L-B now clears the floor, decisively.** Real TimesFM↔Chronos-T5-Base
>    `frac_shared` **0.7769929364278506** (bit-identical to the single-layer
>    run's 0.777 — confirms row-identity held) against the real-vs-own-
>    `random_init`-twin null floor **0.6962699822380106**: diff
>    **+0.08072295418983999**, bootstrap CI **[0.03763937605411403,
>    0.12640397682158971]**, p=0.002, `clears_floor: true`. This is the
>    opposite direction from §6.3's TimesFM-vs-`random_init` L1/L2 finding
>    that started this whole "quote nothing without its floor" discipline —
>    here the real cross-model pair beats its own architecture-only null,
>    not the other way around. Clause 2's L-B half of the pre-registered
>    decision rule is now satisfied for V1.
> 2. **L-C ran for the first time and is real, not flat, not cleanly
>    monotone.** `frac_shared` at δ=1,2,3,4 (block gaps 1,2,4,8):
>    **0.999003984063745, 0.9990079365079365, 0.9979674796747967,
>    0.9236043095004897** — high and essentially flat through δ=3, then a
>    real drop at δ=4 (block gap 8). The script's own `monotone_decay` check
>    reports `"monotone": false, "flat": false` — neither clean pattern the
>    table's informal description ("high at δ=1 and decaying monotonically")
>    predicted holds exactly. **This does not gate `verdict.passes`** —
>    `variant_verdict()` only consumes L-A, L-E, the L-B margin, the V0
>    `gt_alignment` margin, `fidelity_gap`, and `dead_feature_rate`; L-C never
>    enters the computed verdict. That matches §6.2.1 1c's actual numbered
>    decision rule (three clauses, none naming L-C), not the ladder table's
>    looser "Disqualifies a variant if: Non-monotone or flat" column — read
>    1c's prose as the pre-registered, authoritative rule and the table's
>    column as informal framing that the code was never written to enforce.
>    Not a bug to fix; a discrepancy worth naming so a future session doesn't
>    "fix" the code to match the table instead of noticing the table
>    overstated the rule.
> 3. **L-A and L-E still hold at this row count and layer set.** L-A
>    `frac_shared` **0.997** (1024/1024 alive, dead 0.0, fidelity gap ≈0),
>    L-E precision **1.0** / recall **1.0** / F1 **1.0** — both hard
>    prerequisites in clause 1 pass cleanly.
> 4. **L-D and the V0 comparison reproduce bit-for-bit**, confirming nothing
>    about the *result* changed — only the two previously-silent rungs did.
>    L-D `frac_shared` **0.7769929364278506**, `gt_alignment_shared.
>    mean_abs_rho_matched` **0.2786869234414469**; V0's **0.32036928728167674**;
>    diff **−0.04168236384022983**, CI **[−0.06928870407774981,
>    −0.013676070745315533]**, p=0.002, `beats_v0: false`. Identical to the
>    2026-08-13 single-layer run's numbers to the last printed digit — the
>    exact bit-exactness check this session performed twice (once on the
>    null store, once on the layers store) is what makes that reproduction
>    trustworthy rather than coincidental.
> 5. **`verdict.passes` is `false`, for the same reason as before and no
>    other**: `{"l_a_identity_ok": true, "l_e_recovery_ok": true,
>    "clears_l_b_floor": true, "beats_v0_gt_alignment": false,
>    "fidelity_gap_ok": true, "dead_rate_ok": true, "unrun": []}`. Five of
>    six checks pass; `unrun` is empty for the first time — every clause the
>    ladder can evaluate, evaluated. The single failing clause is the one
>    that was already known to fail at 4,608 rows: V1 loses to V0's post-hoc
>    matching on ground-truth alignment. L-B and L-C turning out favorable
>    to V1 does not change that this is what actually decides Stage 1.
> 6. **Full `tsfm_lens` suite green after this session's changes**: 406
>    passed, 2 pre-existing unrelated warnings (a documented float32-cast
>    overflow test and a return-value pytest style warning), 0 failures —
>    no production code was touched this session, only new run artifacts, so
>    this is a verification that nothing about running the ladder for real
>    destabilized anything, not evidence of a new fix.
>
> **What this actually settles.** Stage 1's mechanism — the scorecard plus
> all five ladder rungs, including the two that were previously structurally
> unrunnable — is now fully built, fully wired, and fully exercised against
> live checkpoints with no unrun clause. That was the concrete blocker
> keeping the Stage 1 checkbox open; it is closed. The *verdict* itself is
> unchanged and is not a blocker to resolve — it is Stage 1's actual output,
> a real negative result for V1 (post-hoc-matched independent dictionaries
> beat joint training on ground-truth alignment at this layer pair, this
> row count, this dict size). Per finding 2 in the block above and per Stage
> 2's own decision rule ("if no variant beats V0 ... that is the result"),
> this is not something further ladder-running fixes.
>
> **Next action, following through on (c).** V1 having a *confirmed* (not
> merely suspected) real-alignment deficit against V0, now doubly measured
> (single-layer run and this five-layer run agree bit-for-bit), makes V4
> (latent scaling / per-source normalization tuning — the published fix for
> exactly this failure mode) the next thing to build rather than a second
> re-run of V1's own ladder. Stage 1's checkbox closing does not itself
> block Stage 2 — V0 and V1 are both already built (`sae/crosscoder.py`,
> `sae/crosscoder_eval.py`); what was missing was a trustworthy full-ladder
> readout to build V4 against, and that readout now exists.

---

#### Stage 2 — `[x]` CLOSED: V0, V1, V2 and V4 all built and scored; no variant beats V0

**Triaged 2026-08-18, closed same day.** V0, V1 and V4 are built and scored.
V3, V5 and V6 are parked (§22.1). **V2 was the only remaining build**, and the
reason it survived the cut is specific: it is the *published* fix for exactly
the two artifacts V1 exhibits, so a negative result reported without it
invites the one criticism a reviewer will certainly make — *you used the
variant the literature already says is broken for model diffing.* Closing
that hole cost ~40 lines plus three attempts at the eval-time threshold (see
the Findings below V2's spec) — the debugging, not the training compute, was
the real cost, consistent with this section's own cost summary.

✅ **Ran V2 at 3 seeds**, per the instruction below (kept verbatim — it is
what was done, not merely planned): Stage 0 finding (15) measured the
crosscoder's failures as **bimodal** — four seeds clustered tight, one
collapsed — so a single-seed score will occasionally attribute a bad basin to
a variant. Seeds are seconds here; this is cheaper than the debugging it
prevents. The 3-seed run below additionally surfaced a second, seed-fragile
metric (the L-B floor) that a single seed would have reported as settled.
**V0 · Independent dictionaries + post-hoc matching — the control `[x]` built**
- *Mechanism*: two `TopKSAE`s trained separately, matched by
  `sae/matching.py` (activation-profile correlation over the shared series
  sample). Already implemented and live-verified (§16 E16: 40 of 50
  ground-truth-matched TimesFM features found a Chronos partner).
- *Role*: this is the thing to beat. Per §2.2 and `models.py`'s own
  docstring, no crosscoder variant is adopted unless it beats this.
- *Work needed*: none, except running it through `score_variant` so its
  numbers are on the same scorecard. Note it has no native `frac_shared` —
  define its analog as *fraction of A-side ground-truth-matched features
  with a B-side partner above a fixed matching threshold*, and state that
  definition in the scorecard rather than comparing incomparable quantities.

**V1 · Acausal TopK crosscoder — the reference implementation `[x]` built**
- *Mechanism*: per-source linear encoder → summed → shared bias → one
  shared TopK → per-source linear decoder. `CrosscoderSAE` today.
- *Known weaknesses to expect on the scorecard*: (a) `frac_shared` is read
  post-hoc from a decoder-norm ratio that is centered near 0.5 for *any*
  jointly-normalized atom carrying no signal — hence the mandatory
  `alive_mask`; (b) per-row hard TopK forces exactly `k` atoms on every
  window including trivially simple ones, which wastes capacity and starves
  atoms; (c) the encoder *sums* both sources, so at inference a single
  source's features cannot be computed without inventing a value for the
  other (see Stage 3's `encode_one` problem).
- *Work needed*: run through the scorecard and all five ladder rungs. This
  is the baseline every later row is compared against.

**V2 · BatchTopK crosscoder — the recommended first new variant `[x]` built, run at 3 seeds, does not flip clause 2**
- *Change*: replace per-row `torch.topk(pre, k, dim=-1)` with a **batch-level**
  top-`k·N` over the whole flattened batch, so an information-dense window
  can use more than `k` atoms and a flat window fewer, with the average
  still `k`. At eval time, per-row TopK is not available (no batch), so
  estimate a **single global threshold θ** as the running mean of the
  `k·N`-th largest pre-activation across training batches and use a JumpReLU
  (`pre * (pre > θ)`) at inference. Persist θ in the checkpoint.
- *Why it might win*: this is the published fix for both problems V1 has.
  Dead latents fall because atoms are no longer competing under a
  hard per-row cap, and — the reason it matters *specifically for model
  diffing* — the interpretability literature on crosscoder model-diffing
  reports that sparsity-penalty crosscoders manufacture **false
  model-specific latents** through two named artifacts ("complete shrinkage"
  and "latent decoupling"), and proposes BatchTopK crosscoders as the
  remedy. ⚠️ *Verify these citations and their exact claims before relying
  on the detail* — they are recorded here as a design lead, not as
  established fact in this repo (§2.4).
- *Cost*: ~30 lines in `encode` + threshold bookkeeping in
  `train_crosscoder`. Half a day.
- *Number that must move*: `dead_feature_rate` down, and `frac_specific_*`
  **down** relative to V1 at matched fidelity — if V1's specific atoms are
  partly artifacts, V2 should reclassify them as shared.

> **Findings — V2 built, three false starts on the eval-time threshold before
> a correct one, then run at 3 seeds against the pre-registered rule
> (2026-08-17/18).** `CrosscoderSAE(topk_mode="batch")` in `crosscoder.py`:
> `sparsify`'s `batch` branch is a persisted-threshold JumpReLU, `_batch_topk`
> does the actual batch-level top-`k·N` selection during training. Getting the
> *eval-time* threshold right took three attempts, each one a real, measured
> failure rather than a hypothetical:
> 1. **EMA of the training-time threshold** (`_THRESHOLD_EMA_MOMENTUM`) —
>    inflated `dead_feature_rate` to 0.228 against per-row TopK's 0.032 on
>    the real checkpoint pair, because the encoder is unconstrained and its
>    pre-activation scale drifts over training, so an EMA blends pre- and
>    post-convergence batches and ends up biased low relative to what the
>    converged model actually needs.
> 2. **Post-training calibration** (`calibrate_batch_threshold`, averaging the
>    converged model's own batches only) — *more accurate* and *worse*:
>    dead rate rose to 0.177 (from 0.064 vs V1's contemporaneous run). A
>    single global cutoff trades "which atoms fire at least once" against
>    "the average active count is `k`" — raising it to hit the second target
>    concentrates firing onto fewer, already-strong atoms and starves the
>    ones that only occasionally cleared the old, lower bar.
> 3. **`encode_eval`: stop estimating a global cutoff at all** — every eval
>    consumer already batches rows (8192 at a time), so recompute the exact
>    batch-level selection fresh per eval batch instead of trusting any
>    persisted scalar. This is what every number below uses. Both prior
>    fixes are kept in the code (the persisted `threshold`/EMA still exists
>    and is what a genuine single-row inference call would need), but no
>    scored number depends on them.
>
> **3-seed scorecard, same real pair (TimesFM `stacked_xf.4` ↔ Chronos-T5-Base
> `encoder.block.10`), same budget as L-D/V1 (dict 1024, k 48, 60 epochs,
> aux_k 64), `run_crosscoder_ladder.py --v0 --v2 --seed {0,1,2}` against
> `runs/crosscoder_stage0_bigdata_layers` /
> `runs/crosscoder_stage0_null`:**
>
> | seed | V1 (L-D) dead / n_alive / gt_align | V2 dead / n_alive / gt_align | V0 gt_align (n_alive) | V2 vs V0 diff [CI] p | beats V0? |
> |---|---|---|---|---|---|
> | 0 | 0.032 / 991 / 0.2787 | 0.211 / 808 / 0.2960 | 0.3204 (950) | −0.0244 [−0.0541, +0.0026] p=0.088 | **No** |
> | 1 | 0.185 / 835 / 0.3194 | 0.035 / 988 / 0.2848 | 0.3355 (964) | −0.0507 [−0.0783, −0.0208] p=0.002 | **No** |
> | 2 | 0.022 / 1001 / 0.2733 | 0.174 / 846 / 0.3039 | 0.3441 (935) | −0.0402 [−0.0719, −0.0075] p=0.02 | **No** |
>
> (`V2 vs V0 diff` is `gt_alignment_margin(V2, V0)` — the identical function
> and identical unpaired-over-features bootstrap Stage 1 already used for
> V1 vs V0, just not called for V2 by `run_crosscoder_ladder.py` itself,
> which only prints the V1-vs-V0 line; computed here directly from each
> ladder artifact's stored `abs_rho_matched` arrays.)
>
> 1. 🔴 **Clause 2 does not flip. At no seed does V2 beat V0**, and two of
>    three seeds are decisively negative (CI entirely below zero); the third
>    (seed 0) is the closest call in either variant's favor and still isn't
>    one (CI spans zero, point estimate still negative). Averaged over seeds,
>    V0's gt-alignment (mean 0.334) sits comfortably above both V1's (mean
>    0.290) and V2's (mean 0.296) — the post-hoc-matched independent
>    dictionaries remain the thing to beat, and nothing built here beats it.
> 2. **V2's own stated purpose — reclassify V1's shrunk "specific" atoms as
>    shared — does not happen; if anything the opposite.** The script's own
>    per-seed delta (`V2 vs L-D ... specific_a`) is **+0.038, +0.038, +0.083**
>    at seeds 0/1/2 respectively — `frac_specific_a` goes *up* under V2 at
>    every seed, not down. `dead_feature_rate` moves inconsistently (down at
>    seed 1, up at seeds 0 and 2), so even the less specific, better-attested
>    half of V2's promise ("dead atoms fall") does not hold uniformly on this
>    checkpoint pair either. This is a second, independent line of evidence
>    against V2 beyond the gt-alignment margin: it does not do the thing the
>    literature predicts it should do, on this pair.
> 3. 🔴 **A genuinely new finding, distinct from V2: the L-B floor itself is
>    seed-fragile, and the previously-reported "L-B decisively clears" result
>    was a single seed, not a property of the pair.** `clears_l_b_floor`
>    (V1's alive-atom `frac_shared` vs. its own `random_init`-twin floor) is
>    `True` only at seed 0 (diff +0.081, CI [+0.038,+0.126], p=0.002 — the
>    number already on record). At seed 1 it is **not significant** (diff
>    −0.026, CI [−0.069,+0.017], p=0.216) and at seed 2 it is **significantly
>    negative** (diff −0.107, CI [−0.141,−0.074], p=0.002) — the real pair
>    reading as *less* shared than its own untrained-twin floor, the opposite
>    of seed 0's finding. The real pair's own `frac_shared` is comparatively
>    stable across seeds (0.749–0.777); what moves is the **floor**
>    (0.696 → 0.792 → 0.856), because a from-scratch crosscoder against a
>    random target has no signal to converge to and its shared/specific split
>    is correspondingly noisier seed to seed. This does not change
>    `verdict.passes` at any seed (clause 2 already fails on its own at every
>    seed, per finding 1), but it means the seed-0 L-B finding recorded
>    earlier in this section should be read as "clears at seed 0," not as a
>    settled property of the pair — exactly the single-seed-generalizes-too-
>    far mistake `CLAUDE.md` §16 E9's peak-pair-vs-depth-curve lesson already
>    named for a different metric. The already-recorded seed-0 number is not
>    wrong; the word "decisively" over-claimed how far it travels.
> 4. Full `tsfm_lens` suite green at **418 passed** after this work (up from
>    406 — new/adjusted crosscoder tests), 2 pre-existing unrelated warnings,
>    0 failures. `tests/test_crosscoder.py` + `tests/test_crosscoder_eval.py`:
>    26/26.
>
> **Conclusion: Stage 2 is closed.** V0, V1 and V2 have all been scored on
> the identical pre-registered scorecard; none beats V0 on ground-truth
> alignment at any tested seed. Per Stage 1c's own escape clause — *"if no
> variant beats V0, that is the result"* — this is the result. V3/V5/V6 stay
> parked (§22.1): none addresses the specific failure mode measured here
> (shrinkage/decoupling was V2's hypothesis and it did not resolve it on this
> pair), and the pre-registered rule already fired twice. Stage 3 stays
> gated shut. Stage 4 (below) is the deliverable.

**V3 · Explicit shared/private parameterization — parked (§22.1)**

Partition the dictionary into `F_shared` / `F_private_A` / `F_private_B` blocks
so the split is a modeling choice with a likelihood consequence rather than a
post-hoc read of a continuous ratio against an unjustified `(0.3, 0.7)` band.
The argument is good — it makes "is there shared structure?" answerable as a
fidelity-vs-shared-block-fraction *curve* instead of a histogram threshold. But
it is ~150 lines plus a block-size sweep, and **V4's β-confirmation already
addresses the same worry without retraining anything** (it found that most
nominally-specific atoms don't survive the check). Un-park only if V2's
scorecard says the split is still artifact-dominated. Full write-up in
`ROADMAP_ARCHIVE.md`.
**V4 · Latent scaling — a diagnostic, not a model `[x]` (recommended regardless of winner)**
- *Change*: for every atom the winning variant calls "specific to A", fit a
  single scalar β minimizing `‖x_B − β · (that atom's decoded contribution in
  B's space)‖²` over the corpus. A genuinely B-unused atom gives β ≈ 0; an
  atom that B *does* use but whose decoder norm was suppressed by training
  dynamics gives β meaningfully > 0.
- *Why*: this is the published diagnostic for separating true model-specific
  latents from the shrinkage/decoupling artifacts named in V2. It applies to
  V1, V2, and V3 alike and costs no retraining.
- *Cost*: ~40 lines + one pass over the store. Half a day. **Highest
  value-per-line item in this whole section** — it makes the headline
  shared/specific number trustworthy without changing any model.
- *Number that must move*: it produces a new number —
  `frac_specific_confirmed` (specific atoms surviving the β≈0 check). Report
  both raw and confirmed splits, always.

> **Findings — V4 built, a real math bug found and fixed on real data, then
> re-verified live (2026-08-17).** Implemented as
> `sae/crosscoder_eval.py::latent_scaling_confirm`: for each atom in a
> `buckets` dict (from `atom_buckets`'s norm-ratio split), unit-normalize
> that atom's decoder row on the *other* source
> (`dhat = W_dec[other][atom] * source_scale[other]`, normalized), project
> the other source's true activations onto it (`proj = x_other @ dhat`), fit
> a through-origin scalar `beta = sum(a*proj)/sum(a*a)` against the atom's
> own joint-encoded activation `a`, and score `variance_explained`.
> 1. **Real-data run (`--v4` against `configs/crosscoder_stage0_gate.yaml` /
>    `runs/crosscoder_stage0_bigdata_layers`, dict_size 1024, k=48) surfaced
>    deeply pathological numbers on the first live pass**: individual
>    entries' `variance_explained` as low as roughly −72, means around
>    −30 (specific_a) and −13 (specific_b) — nowhere near the "β≈0 gives a
>    small number" the docstring promised, and a strong signal (§2.4) to
>    stop and diagnose rather than report as-is.
> 2. **Root cause, confirmed by direct reproduction, not guessed at**: `beta`
>    is a through-origin (no-intercept) fit, whose actual null model is
>    "predict zero" — but the R² denominator (`total_sq`) was computed as
>    the *mean-centered* second moment of `proj`
>    (`sum((proj - proj.mean())**2)`), the baseline that only matches a
>    fit **with** an intercept. The two existing V4 unit tests never caught
>    this because both used dense atom activations (`k == dict_size`, every
>    atom fires every row); the real crosscoder is TopK-sparse (k=48 of
>    1024), so `a` — and therefore the through-origin prediction — is zero
>    on most rows, while `proj` (the other source's real, generally
>    nonzero-mean signal) is not. A mean-centered baseline then compares the
>    through-origin residual against a predictor (`proj.mean()`) the fit
>    never had access to, and for a sparse-enough `a` this baseline is
>    smaller than the residual itself, producing exactly the deep-negative
>    values observed. Isolated with a standalone numpy script *before*
>    touching the fix (hand-built sparse-atom-vs-unrelated-nonzero-mean-noise
>    data): old formula gave `ve = -93.51636`, new formula (below) gave
>    `ve = 0.048748374` on identical inputs — the same order of magnitude as
>    the real-data symptom, confirming the diagnosis rather than merely
>    being consistent with it.
> 3. **Fix**: changed `total_sq` to the **uncentered** second moment,
>    `sum(proj**2)` — the R² baseline that actually matches a through-origin
>    null model (`beta = 0` ⇒ predict zero everywhere). One line changed;
>    docstring updated to state the uncentered convention explicitly and to
>    flag that this deliberately departs from `per_source_fidelity`'s own
>    (mean-centered, intercept-implying) R² elsewhere in `crosscoder.py` —
>    the two functions' R² numbers are not meant to be compared to each
>    other.
> 4. **New regression test**
>    (`test_latent_scaling_confirm_uses_an_uncentered_baseline_for_sparse_atoms`,
>    `tests/test_crosscoder_eval.py`) constructs the sparsity the two prior
>    tests missed directly: a 3-atom, k=1 dictionary where one atom fires on
>    only ~5% of rows, matched against a B-side signal that is pure noise
>    around a nonzero mean with *no* real relationship to that atom at all.
>    Asserts `variance_explained` stays small (`< 0.2` in magnitude, not
>    deeply negative) and that `beta` is a real, non-trivial fit. All 17
>    crosscoder tests (`test_crosscoder.py` + `test_crosscoder_eval.py`)
>    pass, zero regressions.
> 5. **Re-ran the real ladder live after the fix** (same config/store, same
>    `--v4 --deltas 1,2,3,4 --v0 --seed 0`), and it now reports sane,
>    small-magnitude numbers throughout: `specific_a` — 206 atoms, 59
>    confirmed (`frac_specific_confirmed = 0.28640776699029125`),
>    `mean_variance_explained = 0.025092972518137968`, every individual
>    entry in `[0.000156, 0.10528]`; `specific_b` — 15 atoms, 5 confirmed
>    (`frac_specific_confirmed = 0.3333333333333333`),
>    `mean_variance_explained = 0.02832778304054145`, every entry in
>    `[0.0020, 0.0672]`. No entry anywhere is negative, let alone deeply
>    so — the pathology is gone.
> 6. **Every other rung reproduced bit-for-bit against the previously
>    recorded run** (L-A `frac_shared = 0.9970703125`, L-E F1 `1.0`, L-C at
>    every tested offset, L-D `frac_shared = 0.7769929364278506`, L-B
>    `frac_shared = 0.6962699822380106`, V0's full matching/alignment
>    numbers, and the unchanged `verdict.passes: false` /
>    `beats_v0_gt_alignment: false`) — the `§11.24`-class check this repo's
>    own doctrine calls for whenever a background rerun of "the same"
>    artifact follows a code change, confirming the fix's blast radius
>    stayed scoped to `latent_scaling_confirm` alone. Full `tsfm_lens` suite
>    also re-run clean: 406 passed, 2 pre-existing unrelated warnings, 0
>    failures.
> 7. **The number itself is a real finding, not just a math correction.**
>    Only **~29%** (specific_a) and **~33%** (specific_b) of the atoms the
>    norm-ratio split calls "specific to one source" survive the β≈0 check
>    at the default `ve_threshold = 0.01` — the *majority* of nominally-
>    specific atoms show a small but measurable (`variance_explained`
>    roughly 0.01–0.11) latent-scaled relationship to the other source. Read
>    together with V4's own premise, this means the raw shared/specific
>    split on this crosscoder (V1, this layer pair, this dict size)
>    materially **overstates genuine specificity** — most "specific" atoms
>    look more like shrunk-but-real shared structure once rescaled, not
>    atoms the other source truly never uses. This does not change the
>    already-recorded, separate Stage 1 result that V1 loses to V0 on
>    ground-truth alignment (`beats_v0_gt_alignment: false`) — V4 makes the
>    specific/shared split interpretable, it does not make this crosscoder
>    variant win.
>
> **A new CLAUDE.md trap entry (§11.31) records the mean-centered-vs-
> uncentered R² mismatch itself, since it is a general hazard for any
> future through-origin/no-intercept fit in this codebase, not specific to
> V4.**

**V5 · Frequency-aware dictionary, V6 · Matryoshka / multi-resolution — parked (§22.1)**

Both are real ideas with published motivation, and both are 2–3 days. Parked
because neither addresses the specific failure V1 exhibits, and the
pre-registered rule has already fired. V5 is the one to un-park first if V2
flips clause 2 — it is the only variant with a TSFM-native rationale (the
benchmark labels seasonality exactly, so a frequency-biased dictionary makes a
falsifiable prediction against E13's spectral lens) rather than an
imported-from-NLP one. Full write-ups, including V5's two independently-testable
sub-variants and V6's token-level-as-a-second-source idea, in
`ROADMAP_ARCHIVE.md`.

#### Stage 3 — `[ ]` Wire the winner into the pipeline — **gated shut, confirmed 2026-08-18: stays that way**

🔴 **Do not build this.** Stage 1's decision rule fired against V1 (clause 2,
p=0.002, CI off zero on the losing side); Stage 2's V2 run (above) fired
against V2 too, at all 3 tested seeds. **Stage 3's unlock condition — V2 flips
clause 2 — did not happen.** Wiring a variant that loses to V0 into the
pipeline, with a config block, a report section and a Protocol extension,
would be building infrastructure for a mechanism the repo's own
pre-registered rule says not to adopt. The deliverable is Stage 4's writeup
below, and nothing in 3a/3b/3c gets built. This gate is not provisional
pending a future variant — V3/V5/V6 are parked (§22.1) precisely because
none addresses the failure mode V2 already tested and did not resolve; a
future un-park would need a new argument, not just more seeds of what is
already built.

**3a. The `SAEAdapter` problem, if it becomes relevant.** `sae/interface.py`'s
Protocol is `encode(activations: [N,D]) -> [N,F]` — inherently single-source; a
crosscoder's `encode` needs *all* sources. Two options, pick one and record
which:
- **Option 1 (recommended) — extend the Protocol.** Optional
  `encode_joint(sources) -> Tensor` plus an `n_sources` attribute; `load_sae`
  dispatches on a `variant` field in the checkpoint. Single-source consumers
  check `n_sources == 1` and skip loudly.
- **Option 2 — a source view.** `CrosscoderSourceView(crosscoder, i)` satisfying
  the existing Protocol by passing zeros for the other sources. Less plumbing,
  but **it is an approximation** — the encoder sums source contributions, so
  zeroing B changes A's features. If chosen, measure how much (correlate
  `encode_one(x_A)` against joint `encode([x_A, x_B])` over the corpus) and
  print that number in the report. Do not ship it silently.

**3b/3c. Pipeline stage and report section.** A `crosscoder` stage depending on
`extract` and `layer_screen`, gated on `sae.crosscoder.enabled`, with `pairs`
defaulting to the L1 peak-CKA pair (`auto` is a lookup in `l1/meta.json`, not a
search). The report section's minimum contents, per §0 rule 6: the
shared/specific histogram **with the L-B floor drawn on it as a reference
line**, top shared atoms with their ground-truth field and ρ, top
model-specific atoms β-confirmed via V4, the fidelity-per-source pair, and one
findings sentence carrying the floor and CI.

**3d. ✅ DONE 2026-08-18 — un-gated from the crosscoder entirely.** The
encode-store seam (`sae/{model}/{layer}` written back into the store) has been
promised in `sae/interface.py`'s docstring since the SAE baseline landed and is
still unwired. It needs no crosscoder: doing it for the **baseline** SAEs
unblocks `level="sae"` reads in L1 and clustering, i.e. **CKA in feature space
instead of activation space** — a genuinely different measurement and one of
the cheapest interesting wins left in the repo. Moved to §0.5's queue as its
own item.

> **Findings — 2026-08-18.**
>
> **1. Mechanism.** `extraction/store.py` gains two purely additive zarr
> groups, `sae/{model}/{layer}` (window-level, `[N, W, F]`) and
> `sae_pooled/{model}/{layer}` (series-level, `[N, F]`), mirroring
> `act`/`pooled`'s existing shapes one axis wider (`F` = dictionary size).
> `init_sae_layer`/`write_sae_batch`/`has_sae_features` are the write-side
> API; `load(model, layer, level=..., space="act"|"sae")` is the read side
> — `space="sae"` against an absent array raises a KeyError naming
> `has_sae_features` as the check to run first (a caller bug, not a
> gracefully-degradable absence like `stack_meta`'s empty dict — most
> (model, layer) pairs never get features persisted at all, since the new
> `sae.persist_features` config flag defaults to **off**: a TopK
> dictionary is typically 8x-16x wider than its input, so this data does
> not multiply every run's disk footprint by default). No `_SCHEMA_VERSION`
> bump — purely additive, no existing key's shape/dtype/meaning changed.
> `sae/train.py::encode_and_persist_features` is the write-side caller,
> invoked once per target inside `run_sae` right after
> `reconstruction_fidelity`/`dead_feature_rate` are computed, batching over
> series (not one `[N*W, dict_size]` tensor) for the same memory reason
> `write_batch` already batches raw activations. A failure here is caught
> and logged, not fatal to the whole `sae` stage (CLAUDE.md sec 2.5);
> `results[key]["features_persisted"]` records which targets actually got
> written.
>
> **2. The consumer.** New `sae/feature_geometry.py::run_sae_feature_cka`,
> called from `run_sae`'s end once `sae.persist_features` is on: linear CKA
> (reusing `analysis/l1_geometry.py::linear_cka` and
> `analysis/stats.py::bootstrap_ci` rather than duplicating either) between
> every pair of (model_a target, model_b target) whose features were
> persisted, series-level only (not window-level — a first pass at this
> measurement multiplying L1's own memory footprint by the dictionary-size
> ratio was judged not worth doing before the series-level result is known
> to be useful). Returns `None` with a log line, not a crash, when either
> comparison-pair model has zero persisted targets. Writes
> `l1/cka_sae.json`.
>
> **3. Real-checkpoint result.** Reran `configs/medium_run_chronos_base.yaml`'s
> `sae` stage (`--stages sae --force sae`) with `persist_features: true`
> added, against an **isolated copy** of `runs/medium_run_chronos_base`'s
> already-extracted store (so the canonical run's checkpoints/numbers this
> file and `CLAUDE.md` already quote elsewhere stayed untouched — the copy
> was deleted after reading its results). Both pinned SAE targets (TimesFM
> `stacked_xf.18`, Chronos-T5-Base `encoder.block.6`, dict_size 6144,
> `real_data_enabled: true`, 60 epochs) trained and persisted cleanly:
> `sae: persisted encoded features for Chronos-T5-Base/encoder.block.6
> (288 series x 16 windows x 6144 features)` (and the TimesFM target
> identically). Feature-space CKA at that pinned pair:
> **0.2625783085823059** (95% CI **[0.22790290378034117,
> 0.3332766056060791]**, `n=288` series). For context, **not** a like-for-
> like comparison (this run's own already-recorded activation-space *peak*
> pair, over every layer combination, sits at a different pair entirely —
> `stacked_xf.4`/`encoder.block.10`, CKA 0.381, shuffled-null 0.036,
> `l1/meta.json`): the SAE targets were pinned for *forecast-readability*
> reasons (highest `tuned_r2_model`), not for peak activation-space CKA, so
> this is one new, real number at a layer pair L1 was never optimizing
> over — not yet a "feature space vs. activation space at matched layers"
> verdict. That comparison (and a null-controlled read of the 0.263 itself)
> is the natural next step if this measurement gets used for anything,
> not attempted here.
>
> **4. Tests.** 11 new: `tests/test_sae_encode_store.py` (8 — store
> round-trip at both granularities, `space="sae"` absent-array error
> message, `space="act"` unaffected, batched writes landing at the right
> offset, `encode_and_persist_features` matching a direct `sae.encode()`
> call within float16 round-trip tolerance, both single-batch and
> multi-batch `series_batch` paths) and `tests/test_sae_feature_cka.py`
> (3 — full mock-pipeline integration: `persist_features: true` populates
> the store and `l1/cka_sae.json`; `persist_features: false` (the default)
> leaves no trace in either; targets on only one comparison-pair model
> degrade `run_sae_feature_cka` to `None` + a log line rather than
> crashing the `sae` stage). Full suite rerun clean afterward: **452
> passed, 2 warnings in 475.80s** (up from F9's own post-item baseline of
> 441 — this item's 11 new tests, no regression, no pre-existing test
> changing status).
>
> **5. What this does not close.** `analysis/l1_geometry.py`'s own main CKA
> pass and `clustering.py` still read only `space="act"` — this item wires
> the seam and builds one standalone consumer, it does not change L1's or
> clustering's default behavior. Window-level feature-space CKA, and a
> null-controlled/matched-layer-pair read of the 0.263 result above, are
> both open follow-ups, not attempted here. `sae/interface.py`'s docstring
> updated to match (the "not yet wired" parenthetical removed).

#### Stage 4 — `[x]` The research deliverable — **written 2026-08-18, this is the endpoint**

Written into this Findings block, all three parts plus the methodological
result. On the evidence in hand this is a **negative result**, and the
pre-registered rule (Stage 1c) explicitly anticipated it — *"post-hoc
matching of independent dictionaries is sufficient; joint training buys
nothing here"* is exactly what the numbers below say, for both variants
built.

> **Findings — the three-part deliverable (2026-08-18).**
>
> **1. The number.** Over alive atoms of the reference crosscoder (V1, dict
> 1024, k 48, seed 0 — the seed V4's β-confirmation ran against), the raw
> shared fraction is **77.70%** (770 of 991 alive atoms). β-confirmed via V4
> (§6.2.1 finding above): of the 206 atoms the norm-ratio split calls
> "TimesFM-specific," only 59 (28.6%) survive the β≈0 check; of 15
> "Chronos-specific" atoms, only 5 (33.3%) do. Folding the *un*-confirmed
> atoms in both buckets back into "shared" (they show a real, β>0
> cross-source relationship the raw split's shrinkage/decoupling artifacts
> hid) gives a **β-confirmed shared fraction of 93.5%** (927 of 991 alive
> atoms) — the headline number moves from "three-quarters shared" to
> "the large majority of what looks specific isn't." This β-confirmed
> correction has no bootstrap CI of its own (V4 was run once, against one
> trained object); it is a point-estimate correction to a number that does
> have one (below), not a new number with the same evidentiary weight.
>
> **2. The floor — and the finding that complicates part 1.** The raw shared
> fraction against V1's own `random_init`-twin floor (`L-B`, resampled over
> atoms) is what the pre-registered rule actually gates, and it is **seed-
> fragile, not settled**: clears decisively at seed 0 (diff +0.081, CI
> [+0.038,+0.126], p=0.002), fails to clear at seed 1 (diff −0.026, CI
> [−0.069,+0.017], p=0.216), and clears in the *wrong direction* at seed 2
> (diff −0.107, CI [−0.141,−0.074], p=0.002 — the real pair reads as *less*
> shared than an untrained twin of one side). The real pair's own raw
> `frac_shared` barely moves across seeds (0.749–0.777); the floor itself
> swings from 0.696 to 0.856. **Reading part 1's headline number against
> only the seed-0 floor — as this section did until today — overstated how
> settled "clears the floor" is.** No β-confirmation exists for the floor's
> own crosscoder (V4 was run only against the real pair), so even the
> stable-seed-0 comparison compares a β-confirmed real number against an
> un-confirmed floor — a like-for-like β-confirmed floor is unbuilt and
> would be the natural next step if this metric is revisited, not attempted
> here since Stage 1c's rule already resolves the variant question without
> it (part 3 below does not depend on this floor at all).
>
> **3. The content — what the shared and specific-looking atoms actually
> are, via ground-truth alignment (V1, seed 0, 316 of 770 shared atoms
> matched to a field, i.e. this is a description of the matched subset, not
> of all shared atoms).** Shared atoms' best-matching ground-truth fields
> cluster on **generator identity and seasonal/noise structure**:
> `generator_mixture`, `generator_block_bootstrap`, `n_seasonalities`,
> `noise_scale`, and the `trend_dominant`/`seasonal_dominant`/
> `nonsinusoidal_seasonal` archetype flags are the most frequent best
> matches among the top-50 listed, with three atoms hitting ρ=1.000 exactly
> on `generator_block_bootstrap`. Atoms the split calls TimesFM-specific
> (206 matched, 158 with a field) skew toward `seasonal_period_dominant`,
> the `seasonal_dominant`/`ar_colored_noise` archetypes, and
> `has_random_walk`; Chronos-specific atoms (15 matched, 11 with a field)
> skew toward `trend_dominant`/`regime_switching`/`clean_low_noise` and
> `n_changepoints`. Read alongside part 1: since most of these "specific"
> atoms are *not* β-confirmed as genuinely specific, this reads less as
> "TimesFM encodes periodicity and Chronos encodes regime changes" and more
> as "the seasonality/period axis is where V1's shrinkage artifact
> concentrates" — a hypothesis for what a future, better-behaved variant
> would need to get right, not a settled finding about either model.
>
> **The three-part number, restated as one sentence:** on this checkpoint
> pair and layer, a jointly-trained TopK crosscoder finds a dictionary that
> is mostly-to-overwhelmingly shared by either reading (78% raw, 93.5%
> β-confirmed), organized around generator/seasonality/noise structure — but
> whether that shared-ness exceeds what two same-shaped networks share
> before either learns anything is genuinely unresolved (1 of 3 seeds says
> yes, 2 say no or the opposite), and on the one question Stage 1c actually
> pre-registered as decisive — does joint training beat independently
> trained, post-hoc-matched dictionaries on ground-truth alignment — the
> answer is **no, decisively, for both variants built, at every seed
> tested.**
>
> **The methodological result, which is the more durable half.** Two things
> this stage can state that the crosscoder literature this repo drew on does
> not: (a) a shared-fraction metric **can be validated against constructed
> known-answer pairs** before being trusted on real data — L-A identity
> returns exactly 1.000 (never below 0.997 across the 3 seeds tested here),
> L-E's planted recovery hits F1 1.0 at every seed; and (b) the same metric
> read **without** a same-seed, architecture-matched floor is not just
> incomplete but actively misleading in *either direction* — at 4,608 rows
> TimesFM read as more shared with a random copy of itself than with
> Chronos (§6.3's finding), and at 46,382 rows across 3 seeds the floor
> itself is unstable enough that a single-seed "decisively clears" claim
> (recorded, uncorrected, until this session) does not generalize even
> within the *same* pair and metric. A crosscoder shared-fraction number is
> not safe to quote without both a same-seed floor and, ideally, more than
> one seed of that floor. Both points transfer to any model-diffing work, in
> any modality, using any crosscoder variant — they do not depend on V1 or
> V2 specifically.
>
> **Then updated**: `CLAUDE.md` §13 item 3 and this file's §11/§13, which
> previously said the flagship crosscoder was not started or still open —
> see those sections for the corrections.

#### Cost summary — corrected against measurement

| Stage | Status | What it actually cost |
|---|---|---|
| 0 | `[x]` | **4 sessions** against a ~2–3 h estimate. The overrun was **not** compute — it was three rounds of arguing about exit criteria, and the root cause was an unchecked premise (`CLAUDE.md` §11.29) |
| 1 | `[x]` | **37 s of GPU** for all seven trainings, against a "~1 h" estimate — two orders of magnitude pessimistic. Plus one re-extraction for L-B at matched row count |
| 2 | `[x]` V0/V1/V2/V4 all built and scored | ~40 lines + three threshold-estimation attempts for V2 (the debugging, not the ~150s/seed of GPU, was the cost); 3-seed run for V2 adds ~7.5 min total |
| 3 | `[ ]` gated, confirmed staying gated | V2 did not flip clause 2 at any of 3 seeds — not authorized |
| 4 | `[x]` | The writeup, above. **This is the endpoint, and the section is done** |

🔴 **The single most useful number here: the ladder costs 37 seconds.** Any
statement of the form "we can't afford to check that" is false for this
subsystem. What the project could not afford was *deciding*, which is exactly
what the Stage 1c pre-registration was built to remove — so honor it.

### 6.3 Phase 2c — L2 stitching as a distillation / fine-tune detector (brief item 2) — ❌ falsified for its stated use case, DONE 2026-08-10 (a real, decisive, negative result — see Findings)

**Goal.** Test whether the repo's existing L2 stitching-gain machinery (and
L1 CKA) can distinguish "model B was fine-tuned or distilled from model A"
from "model B was trained independently," with a real IP/provenance-detection
use case in mind. This is separable from — and doesn't block — 2a/2b.

**Concrete deliverables**
- [x] Design a **ground-truth-labeled experiment**, since no public registry
  of "known-distilled TSFM pairs" is likely to exist: fine-tune (or
  distill, if a teacher-student recipe is feasible) a small model from a
  pretrained checkpoint as the **positive** case, and separately train a
  same-size, same-architecture model from scratch (or take an existing
  independently-trained checkpoint of matching size) as the **negative**
  control. Different checkpoint sizes of the *same* released family
  (e.g. `chronos-t5-small` vs. `chronos-t5-base`) are a plausible proxy
  positive if they share meaningful pretraining lineage — verify what's
  actually documented about their training relationship before assuming it,
  don't guess. — **done 2026-08-07**: verified (via `hf_hub_download` on
  `amazon/chronos-t5-base`'s actual README, not assumed) that every
  Chronos-T5 size is fine-tuned from a **separately pretrained**
  `google/t5-efficient-{tiny,mini,small,base,large}` text checkpoint — so
  the positive pair does **not** share initialization weights, this is not
  literal distillation — but all five sizes were then trained on the
  *identical* time-series corpus using the *identical* published Chronos
  procedure (arXiv:2403.07815). That's the precise, documented "meaningful
  pretraining lineage" this item asked to confirm rather than guess:
  shared training data/recipe, independent base weights. Negative control:
  `amazon/chronos-t5-base` vs `google/timesfm-2.5-200m-pytorch` (already
  on record from `medium_run_chronos_base` — real checkpoints, unrelated
  organizations, unrelated training data, no shared lineage). See the
  confound this introduces in the Findings below (same-architecture
  positive vs. cross-architecture negative) — flagged, not hidden.
- [x] Compute L1 peak-CKA and L2 stitching-gain-over-baseline for both the
  positive and negative pairs, using the exact same benchmark corpus and
  identical bootstrap discipline (`CLAUDE.md` §6.6) as the main comparison.
  — **done 2026-08-07**: new `configs/distill_positive_chronos_small_base.yaml`
  (same `benchmark_medium/public_dev` corpus, context/horizon/seed as
  `medium_run_chronos_base.yaml`) run live against real
  `amazon/chronos-t5-small` + `amazon/chronos-t5-base` checkpoints. Real
  numbers in Findings below.
- [x] The claim worth testing precisely: does the *gain over the
  input-feature baseline* (not raw CKA/R², which two same-domain-trained
  independent models could share plenty of) separate positive from negative
  pairs with a CI that excludes overlap? State the null explicitly (§2.2):
  independently-trained same-size models might already show substantial
  stitching gain just from sharing training-data statistics, in which case
  this signal is weaker evidence of lineage than of "same era, same data"
  and that distinction needs to be reported honestly, not oversold. —
  **done 2026-08-07**: new `analysis/distillation_detection.py` +
  `run_distillation_detection_test.py` (a thin, purpose-named reuse of
  `null_baseline.py`'s existing bootstrap-difference machinery — see
  Findings for why no new statistical code was needed). Real result: yes,
  cleanly, for both L1 and L2 — see Findings for exact numbers. The stated
  null (same-era/same-data rather than lineage) is exactly the confound
  named in the item above and addressed the same way: flagged as real and
  currently unresolved by this design, not resolved by it.
- [x] If the signal holds up, write this as a standalone method note (its
  own doc or a clearly separated section here) describing the intended
  use case (provenance/IP disputes) and its actual false-positive risk
  in plain terms — this has real-world stakes if anyone downstream acts on
  it, so §2.6 (`CLAUDE.md`)'s "never let a correlational number be read as
  causal" discipline applies doubly hard here: this method establishes
  *representational similarity*, not legal derivation, and any writeup must
  say so explicitly. — **answered, negatively: no standalone method note
  will be written, because the signal does not hold up (done 2026-08-10,
  see Findings).** The honest next step named at the time — a
  same-architecture negative control isolating "lineage" from "architecture
  match" — turned out not to need a second real checkpoint after all: the
  already-implemented `random_init` mechanism (§6.2) gives a same-
  architecture, *zero-training* control for free. That control's L1/L2
  signal came back significantly **higher** than the real trained positive
  pair's, not lower — meaning architecture match alone, with no training
  and no lineage whatsoever, produces more apparent "shared structure" by
  this method's own metrics than genuine documented shared-lineage
  training does. This closes the item by falsifying the method for its
  stated use case, not by validating it — a standalone provenance/IP
  method note would be actively misleading to write now, so the checklist
  is complete precisely by staying unwritten.

**Findings / decisions**
- **2026-08-07 — first real test: the signal is real and large, but the
  design has a confound that must be resolved before this method is
  trustworthy evidence of lineage specifically.** Built
  `configs/distill_positive_chronos_small_base.yaml` (lean: only
  `l0`/`l1`/`l2`/`internals`/`report`, no `l3`/`lens`/`attention`/`sae`/
  `confirm` — this deliverable needs L1/L2 only) and ran it live against
  real `amazon/chronos-t5-small` (6 encoder blocks) + `amazon/chronos-t5-base`
  (12 encoder blocks) checkpoints, same corpus as `medium_run_chronos_base`.
  Extraction + L0 + L1 + L2 + report completed in **under two minutes**
  (288 series, no L3/attention/SAE) — by far the cheapest real-checkpoint
  run in this repo's history, since this deliverable only ever needed two
  of the pipeline's ~13 stages.
  - **Point estimates**: positive pair (Chronos-T5-Small vs Chronos-T5-Base)
    L1 peak window-CKA = **0.734**; negative pair (Chronos-T5-Base vs
    TimesFM, already on record from `medium_run_chronos_base`) L1 peak =
    **0.381**. L2 best gain-over-baseline: positive = **0.637**, negative =
    **0.413**. Both point estimates already nearly double for the positive
    pair before any significance test.
  - **Significance test** (`analysis/distillation_detection.py`'s
    `compare_lineage_signal`, `n_boot=2000`, paired bootstrap — both runs
    load the identical corpus/seed, confirmed row-aligned by
    `null_baseline.py`'s existing `_series_aligned` check, not assumed):
    L1 diff = **+0.353, 95% CI [+0.307, +0.390], p=0.0005** (floored at
    1/2000 per `CLAUDE.md` §6.6); L2 diff = **+0.224, 95% CI [+0.168,
    +0.285], p=0.0005**. Both CIs are comfortably clear of zero and clear
    of each other's point estimate — this is a decisive, not marginal,
    separation for this one (positive, negative) pair.
  - **No new statistical code was needed.** `null_baseline.py`'s
    `compare_l1_peak_cka`/`compare_l2_best_gain`/
    `run_null_baseline_comparison` already implement exactly "bootstrap the
    difference between two runs' own best-achievable L1/L2 signal, paired
    when the two runs share corpus row order" — nothing in that
    implementation assumes one side is an untrained-weights null
    specifically. `distillation_detection.py` is a ~20-line named wrapper
    (`compare_lineage_signal`) that calls `run_null_baseline_comparison`
    under this question's own semantics (`a`=positive, `b`=negative) rather
    than duplicating the bootstrap logic — `CLAUDE.md` §2.2's "prefer one
    well-tested mechanism over a parallel one" applied directly.
  - ⚠️ **The confound this design does not resolve, stated plainly rather
    than glossed over**: the positive pair is **same-architecture**
    (both T5 encoder-decoder, both Chronos scalar-quantization
    tokenization) while the negative pair is **cross-architecture**
    (TimesFM decoder-only vs. Chronos encoder-decoder, different
    tokenization entirely). So this one test cannot distinguish "the
    signal tracks shared pretraining lineage" from "the signal just tracks
    shared architecture" — both are true simultaneously for the positive
    pair, and both are false simultaneously for the negative pair, by
    construction of the only two runs available. A same-architecture
    negative control (two independently-trained T5-encoder-decoder time-
    series models with no shared lineage) is the natural next step and is
    **not currently possible with any checkpoint already in this repo's
    reach** — no other public T5-based TSFM was found during this session.
    Read this Finding as: **"L1/L2's gain-over-baseline separates this
    positive pair from this negative pair, decisively"** — a real, useful,
    first result — **not** yet "L1/L2 detects shared pretraining lineage
    specifically, independent of architecture." The distinction matters
    precisely because §6.3's own intended use case (provenance/IP disputes)
    has real stakes if overclaimed — which is why the "standalone method
    note" item above is left undone rather than written prematurely.
  - Full `tsfm_lens` test suite green at **200/200** (2 new tests for
    `distillation_detection.py`, reusing `test_null_baseline.py`'s exact
    mock-pipeline fixture pattern — a same-architecture "positive" pair of
    two `mock_patch` instances and a cross-architecture "negative" pair of
    `mock_patch` vs `mock_wave`, confirming only that the comparison
    plumbing runs and labels sides correctly, not making any claim about
    mock adapters having real lineage).
- **2026-08-10 — the confound named above is now resolved, and it kills the
  method for its stated use case.** Built the same-architecture,
  zero-training control the 2026-08-07 Findings said wasn't available: not
  a second real T5-based TSFM checkpoint (still doesn't exist in this
  repo's reach), but `random_init: true` on **both** sides of the existing
  positive pair's own architectures — `configs/
  distill_negative_random_architecture.yaml` pairs a freshly, randomly
  initialized `Chronos-T5-Small` against a freshly, randomly initialized
  `Chronos-T5-Base`: same architecture family and tokenizer as the real
  positive pair, but zero shared initialization, zero training, zero
  lineage of any kind. Ran live against real checkpoint configs (weights
  discarded per §6.2's `random_init` mechanism, no new code needed — exactly
  the "prefer one well-tested mechanism" reuse `CLAUDE.md` §2.2 asks for).
  - **Point estimates**: L1 peak window-CKA = **0.878** (best pair
    `encoder.block.0` ↔ `encoder.block.8`); L2 best gain-over-baseline =
    **0.834** (Small→Base) / **0.940** (Base→Small). All three numbers
    *exceed* the real trained positive pair's own (0.734 CKA / 0.637 best
    gain) — a zero-training, architecture-matched pair looks *more*
    "linearly shared" than the actual documented-lineage trained pair does.
  - **Significance test** (`run_distillation_detection_test.py`, positive =
    `distill_positive_chronos_small_base`, "negative" = this new random-
    architecture control, `n_boot=2000`, paired — both runs load the
    identical corpus/seed): L1 diff = **−0.1431, 95% CI [−0.2173, −0.1302],
    p=0.0005**; L2 diff = **−0.3029, 95% CI [−0.3353, −0.2685], p=0.0005**.
    Both decisively favor the *random-architecture* side, the opposite
    direction the method would need to show lineage-specific signal.
  - **Reading this correctly.** This is not "the confound is still
    unresolved" — it is resolved, and the answer is that **architecture
    match alone, with no training at all, explains more of the previously-
    measured positive-vs-negative gap than lineage does.** The 2026-08-07
    result ("positive pair beats negative pair, decisively") was real and
    reproduces, but this session's control shows that gap was very likely
    driven by "same architecture" rather than "shared training lineage" —
    if anything, real training *shrinks* the stitching/CKA signal relative
    to what architecture-matched random initialization alone already
    produces on the same input corpus. A plausible mechanism (not tested
    further this session): an untrained network's representations stay
    close to a generic, architecture-determined transform of the input,
    which two same-architecture random twins share almost automatically;
    training pulls each model's representations toward its own model-
    specific, task-driven structure, which need not stay as mutually
    predictable even between models with real shared lineage.
  - **Consequence for the method as a provenance/IP detector.** Not
    usable as designed. A same-architecture pair with *zero* relationship
    of any kind already produces a bigger apparent "positive" signal by
    this method's own metrics than genuine shared training lineage does —
    so a high L1 CKA / L2 gain number cannot be read as evidence of lineage
    specifically; it may just as easily be evidence of nothing but a shared
    architecture family, which is very often already known and undisputed
    in a real provenance question. This is exactly the "never let a
    correlational number be read as causal" risk the checklist item above
    flagged in advance — caught before, not after, a method note overclaimed
    it.
  - Full `tsfm_lens` suite unaffected by this session's changes (one new
    config, no source-code edits) — not re-run as part of this finding
    since nothing testable changed; the live-checkpoint results above are
    themselves the verification.

---

### 6.3.1 Provenance detection, reopened — five replacement approaches `[ ]`

> **Added 2026-08-11 (planning only).** §6.3 above is a *method*
> falsification, not a *goal* abandonment, and the distinction was getting
> lost: the deliverables are all `[x]`, the section header says "DONE," and
> a fresh session skimming checkboxes would reasonably conclude provenance
> detection is finished and settled. It isn't. **The capability is still
> wanted** — "can you tell, from weights and behavior alone, whether model B
> descends from model A" is a real question with real commercial and legal
> demand, and this repo is unusually well-equipped to answer it (five
> adapters across four architecture families, a `random_init` null
> mechanism, a sealed corpus with exact ground truth, and paired-bootstrap
> discipline already built).
>
> What §6.3 established is a **hard constraint on any replacement**, and it
> is worth stating precisely because it is the single most useful thing that
> negative result produced:
>
> 🔴 **Architecture match alone produces a larger apparent lineage signal
> than genuine shared lineage does.** Two `random_init` twins — zero
> training, no relationship of any kind — scored L1 CKA **0.878** and L2
> gain **0.834 / 0.940**, versus the real documented-lineage pair's
> **0.734** and **0.637**. Any replacement method must therefore either
> (a) explicitly control for architecture, or (b) use a signal that
> architecture-matching cannot fake. A method that reports raw
> representational similarity is answering "are these the same
> architecture," a question that is nearly always already known and not in
> dispute in a real provenance case.
>
> Below: five approaches, each independently buildable, compared on one
> table. **Option E is a prerequisite for honestly evaluating A–D** and
> should be built first regardless of which detector is eventually adopted.

#### Comparison table

| | **A · Architecture-controlled residual** | **B · Weight-space fingerprinting** | **C · Idiosyncratic-error fingerprinting** | **D · Tokenizer/front-end fingerprinting** | **E · Build a real positive pair** |
|---|---|---|---|---|---|
| **Signal used** | Similarity *minus* the architecture-matched null floor | Permutation-aligned weight/singular-value structure | Per-series forecast **error** correlation after removing predictable-difficulty effects | Quantization grid, patch phase, scale-equivariance, NaN policy — front-end constants | (not a detector — creates the ground truth) |
| **Can architecture-matching fake it?** | **No, by construction** — the floor is subtracted | **No** — untrained twins share architecture but not weight values | **Mostly no** — error idiosyncrasies come from training data/recipe, not layer counts | **No** — front-end constants are recipe choices, not architecture | n/a |
| **Needs both models' weights?** | Activations only (grey-box) | **Yes, full weights** (white-box) | **No — black-box, API-only** ⭐ | No — black-box | Yes (to train) |
| **Works across different architectures?** | Yes, but the floor is only defined per-architecture | No — needs matched shapes | **Yes** — this is its main advantage | Partially | n/a |
| **Cost** | ~2 h (mechanism already exists) | ~1 day | ~1 day | ~half day (E17 built most of it) | **~1–2 days GPU** |
| **Ground truth needed** | A known-lineage pair (→ E) | Same | Same | Same | — it *is* the ground truth |
| **How it gets falsified** | The residual is ≤ 0 for a known-lineage pair, or > 0 for a known-unrelated one | Alignment fails to find better-than-chance permutation on a *known* fine-tune pair | Error correlation of a known-unrelated pair matches a known-lineage pair | Two independently-trained models share the same front-end constants (very possible — everyone copies the same defaults) | The fine-tune fails to converge, or converges so far from its parent that it's not a realistic positive |

---

#### Option A · Architecture-controlled residual `[ ]` — cheapest, directly repairs the falsified method

**Idea.** §6.3's method isn't wrong, it's *unnormalized*. Report not
`CKA(A, B)` but `CKA(A, B) − CKA(A, A_random_twin)` — or better, the
paired-bootstrap difference between them, which `analysis/null_baseline.py`
**already computes** (it is exactly what §16 E9's follow-ups did for L1/L2).

**Concretely.**
1. For candidate pair (A, B), also run A vs `random_init(A)` and B vs
   `random_init(B)` — configs of this exact shape already exist
   (`configs/null_timesfm_random.yaml`, `null_chronos_random.yaml`).
2. Compare **at matching layer indices**, not each run's own global peak.
   This is not a detail — §16 E9's third follow-up found that comparing
   peak-against-peak produced a *completely wrong verdict* for L2, reversed
   once matched-index comparison was used. Bake matched-index comparison
   into the method; do not leave it as a caveat.
3. The detector output is the **depth curve of the residual**, not one
   number: §16 E9's second follow-up showed real shared structure beats both
   floors at middle depths and loses at the extremes. A single scalar
   discards the part of the signal that is actually diagnostic.

**Why it might still fail, stated up front**: the null is an *untrained*
twin, and a plausible adversary's model is trained — just independently.
The right floor for "independently trained, same architecture" is a second
independently-trained real checkpoint of the same architecture, which for
most TSFM families does not exist publicly. Option A therefore likely gives
a floor that is too *low*, i.e. it may over-report lineage. Say so in any
output. **Done when**: the residual is positive with a CI excluding zero for
E's known-lineage pair and contains zero for a known-unrelated pair.

#### Option B · Weight-space / permutation-aligned fingerprinting `[ ]`

**Idea.** Fine-tuning moves weights *a little*; independent training lands in
a completely different basin. Weights carry provenance evidence that
activations wash out — but only up to the symmetries of the architecture
(neuron permutation within a layer, scale/shift redistribution across
LayerNorm boundaries). So: canonicalize, then compare.

**Concretely**, three sub-signals in increasing cost:
- **B1 — permutation-invariant spectra (cheapest, start here).** Singular
  values of each weight matrix are invariant to neuron permutation entirely.
  Compare per-layer singular-value spectra between A and B. A fine-tune
  barely moves them; independent training does not reproduce them. ~50 lines,
  no optimization, no matched-neuron problem.
- **B2 — matched-permutation distance.** Solve the layerwise assignment
  problem (Hungarian on neuron-activation correlation, or on weight-row
  cosine) to align B's neurons to A's, then measure post-alignment weight
  distance. Standard "git re-basin" style machinery.
- **B3 — anchor-weight probe.** Some parameters are near-untouched by
  fine-tuning (embedding tables, LayerNorm gains). Their exact values are a
  near-hash of the parent checkpoint.

**Why it might fail**: requires full weights (many provenance disputes are
API-only), and requires matched architecture shapes, so it says nothing about
cross-architecture distillation — which is the *interesting* case and the one
this repo's TimesFM-vs-Chronos setup is built for. **Best used as the
high-confidence white-box confirmation**, not the general detector.

#### Option C · Idiosyncratic-error fingerprinting `[ ]` — the most promising, and black-box ⭐

**Idea.** Two models trained on the same data with the same recipe make the
**same specific mistakes on the same specific series**. Two independently
trained models make errors of similar *magnitude* on the same series — because
some series are simply harder — but the *residual after removing difficulty*
is uncorrelated. This is the logic behind authorship/plagiarism detection and
behavioral model-identity work generally, and it needs nothing but forecasts.

**Concretely.**
1. `predict()` both models on the corpus — L0 already does this; reuse the
   arrays, no new forward passes.
2. Per series, compute the error vector `e = ŷ − y` over the horizon.
3. **Regress out difficulty.** The confound is that hard series are hard for
   everyone. Remove it by regressing each model's per-series error on
   series-level difficulty features (the input-feature probe from L2 already
   exists — raw window + FFT magnitudes + summary stats), and correlating the
   **residuals**. This is the load-bearing step and the direct analog of L2's
   input-feature baseline; without it, the method just re-measures "hard
   series are hard" and will produce a large signal for any two competent
   models.
4. Also correlate the **signed, shaped** error over horizon steps, not just
   its magnitude — E12 already resolves everything by horizon step, so the
   per-step error shape is free. Shared over-shooting at h=1 followed by
   shared reversion at h=20 is a far more specific fingerprint than a
   per-series scalar.
5. Statistics per invariant 2: series is the resampling unit; cluster
   bootstrap the correlation.

**Why this is the best candidate.** It is black-box (works against an API),
architecture-agnostic (works for cross-architecture distillation, the case B
cannot touch), needs no new forward passes, and — decisively — **architecture
matching cannot fake it**: two `random_init` twins have *identical*
architecture and their errors are pure noise, which is exactly the control
that killed §6.3's method and which this one passes by construction. Run that
control first as a sanity check; it costs one `predict()` pass.

**Why it might fail**: models trained on overlapping public corpora (nearly
all TSFMs) may share error idiosyncrasies from *the data* rather than from
lineage — the same "same era/data confound" §13 item 6 named. Option C
doesn't escape that confound, it just moves it somewhere more measurable:
with E's fine-tuned pair you can quantify how much of the correlation
survives when only the *recipe* is shared vs. when the *initialization* is.

#### Option D · Front-end / tokenizer fingerprinting `[ ]`

**Idea.** A model's input front-end is a set of essentially arbitrary recipe
constants — quantization bin count and range, patch width and phase
convention, scaling policy, NaN handling, context truncation behavior. A
fine-tune inherits them exactly. An independent implementation almost never
matches on all of them simultaneously.

**Concretely.** This is §16 E17's `frontend` stage used as a detector rather
than as a diagnostic — most of the measurement already exists
(`analysis/phase_sensitivity.py`, `analysis/quantization_churn.py`,
`analysis/context_scaling.py`). Produce a fixed-length **front-end
fingerprint vector** per model and compare by exact/near match.

**Why it might fail as a sole detector**: front-end constants are frequently
copied between *unrelated* projects (everyone uses patch=32 because TimesFM
did). High false-positive risk. **Best used as corroborating evidence** —
cheap, interpretable, and a *mismatch* is strong negative evidence even when
a match is weak positive evidence.

#### Option E · Build a real known-lineage pair `[ ]` 🔴 — do this first

**Why this is a prerequisite, not an option.** §6.3's positive case was
`chronos-t5-small` vs `chronos-t5-base` — which, as §6.3's own deliverable
correctly discovered, do **not** share initialization weights; they share a
training corpus and recipe. So the experiment never had a true positive case,
and — as its Findings flag — it paired a *same-architecture* positive against
a *cross-architecture* negative, confounding lineage with architecture in the
one comparison the whole method rested on. **A–D cannot be honestly evaluated
until a genuine, unambiguous positive pair exists**, and the only reliable way
to get one is to make it.

**Concretely.**
1. Take `amazon/chronos-t5-small` as the parent.
2. Fine-tune it on this repo's own corpus (`benchmark_medium/public_dev` or
   an E4 reference corpus) for a modest number of steps → **child_light**.
3. Fine-tune much harder / on a deliberately different distribution →
   **child_drifted**. This gives a *lineage-strength axis*, not a single
   positive point, which is what turns "does the detector fire" into "how far
   can a descendant drift before the detector stops firing" — a far more
   useful characterization and a genuinely publishable curve.
4. **Negative control that matters**: `chronos-t5-small` vs
   `chronos-t5-mini`, both real, same family, same architecture family,
   documented *independent* base weights — a same-architecture negative,
   which is exactly what §6.3 lacked.
5. Also keep the existing cheap null: `random_init` twins (zero training).

**Deliverable**: a small, committed fine-tuning script
(`tsfm_lens/experiments/finetune_child.py` or similar) with a recorded seed,
step count, and corpus digest, plus the resulting checkpoints stored locally
(not committed — sizes are large; record the digest and the exact command so
they are regenerable, mirroring §4.5's sealed-corpus discipline).

**Cost**: `chronos-t5-small` is ~46M parameters; a light fine-tune on a few
hundred series is well within a single GPU session. This is the cheapest
piece of *real* ground truth this whole question needs.

---

#### Recommended sequence

1. **E first** (~1–2 days). Without it, every other option is evaluated
   against a positive pair we already know isn't one.
2. **C next** (~1 day). Highest expected value: black-box,
   architecture-agnostic, no new forward passes, and passes the
   architecture-match control by construction. Run the `random_init`-twin
   sanity check *before* the real pair.
3. **A alongside C** (~2 h). Nearly free — it is a re-analysis of numbers
   already recorded — and it repairs rather than discards the falsified
   method, which is worth doing on its own terms.
4. **D** (~half day) as corroborating evidence once E17's `frontend` stage
   lands.
5. **B** (~1 day) only if the white-box case matters for the intended
   application; note it cannot address cross-architecture distillation.

**Pre-register the success bar now**: a detector is adopted if, across E's
lineage-strength axis, it separates `{child_light, child_drifted}` from
`{chronos-t5-mini, random_init twin, TimesFM}` with non-overlapping bootstrap
CIs, **and** its verdict does not flip when the architecture-matched negative
(`chronos-t5-mini`) is substituted for the cross-architecture one. That second
clause is the exact test §6.3's method failed, and no replacement should be
adopted without passing it.

**When one is adopted**: update §6.3's header from "❌ falsified" to "❌
falsified as designed; superseded by §6.3.1's Option X" — per §0.2, correct in
place, don't delete.

---

## 7. Phase 3 — Ablation and bias-characterization studies (brief item 3)

> **Status: substantially done 2026-08-05.** ✅ bullet 1 (controlled parameter
> sweeps, MASE-axis first pass) · ✅ bullet 2 (noise × depth sweep, first pass,
> plus the quantization-churn follow-up) · ✅ bullet 4 (verbose case studies —
> forecast half) · ✅ bullet 5 (bias card). Findings below.

**Goal.** Beyond the head/MLP ablation and corruption-patching already in
`CLAUDE.md` §6.5, run *designed* experiments that isolate specific biases —
"is TimesFM better at high-frequency structure, Chronos at statistical
means" is a concrete brief question (§1) that deserves a controlled answer,
not just an inference from aggregate archetype MASE.

**Concrete deliverables**
- [x] **Controlled parameter sweeps** using `build_pipeline`'s `parametric`
  generator directly (not `random_parametric`): hold every component fixed
  except one (e.g. sweep seasonal period from 4 to 256 timesteps holding
  amplitude/noise/trend constant; sweep noise SNR holding structure
  constant; sweep intermittency rate from 0 to 80%). Plot each model's MASE,
  crystallization depth, and L3 noise-fingerprint response as a function of
  the swept parameter — this directly answers "where is each model's
  sweet spot" with a dose-response curve instead of an archetype-level
  average. — **MASE axis done 2026-08-05** for all three named sweeps
  (period, noise, intermittency); see Findings below for real numbers,
  including a genuine anomaly (TimesFM's MASE spikes specifically at
  period=32, its own patch width) and a genuine methodological caveat
  (MASE's scale term gets unreliable under heavy intermittency). **Not
  done**: crystallization depth and L3 noise-fingerprint response per sweep
  point — both need the full extraction/lens machinery run per sweep point,
  not just `adapter.predict`, and are left as a follow-up rather than
  attempted partially this session.
- [x] Follow up on the one concrete causal lead already in hand (this file's
  §5 Findings, carried forward 2026-08-05 from the now-retired `CLAUDE.md`
  §10): additive-noise depth-sensitivity is **strongly
  anti-correlated** between the two models (ρ=−0.90). Design a sweep (SNR ×
  depth) specifically to characterize *why* — is it continuous embedding vs.
  quantization, as hypothesized, or something else? This is exactly the kind
  of "further understand the inner workings and biases" the brief asks for,
  and there's already a strong, specific, previously-observed effect to
  chase rather than starting from nothing. **First pass done 2026-08-05,
  same-day follow-up** — see Findings below: the anti-correlation is not a
  fixed property of one arbitrary SNR, it's a real dose-response transition
  (positive at mild noise, flipping negative and then plateauing strongly
  negative across a wide range of heavier noise) — real signal, but the
  *why* (quantization vs. something else) is still open, see Findings.
  **Quantization-churn follow-up done 2026-08-05 (same-day, second
  follow-up)**: directly tokenized Chronos under the same SNR sweep, per
  `CLAUDE.md` §11.16's method. Partial answer — token-ID jump *magnitude*
  (not raw churn fraction, which saturates immediately) plausibly explains
  the sign flip's *onset*, but not the plateau's flat shape — see Findings.
- [x] **Feature-level ablation** (depends on Phase 2b): zero individual SAE
  features (or small groups) via `token_patch` on the reconstruction and
  measure forecast impact, cross-referenced against each feature's
  ground-truth alignment score (§6.3) — do features that align well with
  "trend order" actually matter causally for trend-dominant series and nothing
  else? This is the sharpest test of whether the learned features are real
  computational structure or just descriptive correlations. **Deliberately
  skipped this session** in favor of the two fully-unblocked bullets below
  (2026-08-05, per §0.4's "explain before reordering"): per §6.2's Findings,
  the SAE's forecast-preservation validity check currently passes for
  TimesFM (ΔMASE +0.05) but still fails for Chronos-T5-Base (ΔMASE +2.37),
  so a cross-model ablation comparison built now would be one-sided and the
  Chronos side's ablation deltas would be confounded by the SAE's own
  reconstruction error rather than isolating a feature's causal
  contribution — exactly the failure mode that validity check exists to
  catch. Still the natural next pick once Chronos's forecast-preservation
  gap (a window-broadcast-granularity confound, per §6.2) is closed.

  **Findings (2026-08-11, seventeenth cron-loop firing — mechanism
  implemented and unit-tested; not yet verified against a real checkpoint.)**
  E15's token-granularity fix (above) closed the blocker, so this item is
  now unblocked and picked up per its own "natural next pick" note.
  `sae/eval.py` gained `_feature_ablated_replacement` (encode → zero one
  dictionary feature column → decode, at **token** granularity only, never
  "window" — window-broadcast would reintroduce exactly the confound E15
  just fixed, and would additionally hide a single feature's own effect
  behind that same information loss) and `feature_ablation_effects`, which
  ablates each of a caller-supplied `candidate_features` list one at a time
  and reports `mase_delta_vs_full_recon` (the causal number — compared
  against the **full** token-level reconstruction with every feature
  intact, not the raw clean forecast, so the delta isolates one feature's
  own marginal contribution rather than conflating it with the SAE's
  aggregate reconstruction error) plus `mase_delta_vs_clean` (context) and
  a per-family breakdown (via `data.families`) so a feature matched to e.g.
  "trend order" can be checked for whether its causal effect actually
  concentrates in trend-dominant series. Wired into `sae/train.py::run_sae`
  behind a new `sae.feature_ablation_enabled` flag (default `False` — costs
  one extra full forward pass per ablated feature on top of
  `forecast_preservation`'s own two); candidates are the top
  `sae.feature_ablation_top_k` (default 8) features by `|rho|` from that
  same target's already-computed `ground_truth_alignment` result, reusing
  its matches rather than re-searching. Degrades cleanly to `fa = None` with
  a logged skip when there are no ground-truth-matched features (e.g. smoke
  data, which has no sealed corpus) — verified directly, not just by
  inspection, by extending `test_smoke.py::test_sae_stage_integration` with
  `cfg.sae.feature_ablation_enabled = True` and asserting
  `entry["feature_ablation"] is None`; the live log line confirms the exact
  skip path: `sae: feature-ablation skipped for patchy/blocks.5: no
  ground-truth-matched features to ablate`. New unit test in `test_sae.py`
  (`test_feature_ablated_replacement_removes_exactly_that_features_contribution`)
  verifies the mechanism against an analytically-known linear-decoder fake
  SAE: ablating feature 0 removes exactly `outer(features[:,0], w_dec[0])`
  from the full reconstruction, and a row where that feature never fired is
  completely unaffected — both `tests/test_sae.py` (12/12) and the extended
  `test_sae_stage_integration` pass locally on mock adapters. **Not yet
  done:** a live-checkpoint run against a real sealed corpus with actual
  ground-truth matches (the only way to see `feature_ablation_effects` take
  its real, non-empty-candidates path — no existing test anywhere in the
  suite exercises `ground_truth_alignment`'s real matching-success path
  either, a pre-existing gap this session did not close) — natural next
  step is a background `Agent` run against a config like
  `medium_run_chronos_base.yaml` (real sealed corpus + already-recorded
  ground-truth matches) with `sae.feature_ablation_enabled: true`.

  **Live-checkpoint verification done 2026-08-11 (seventeenth cron-loop
  firing, same-day follow-up).** Background `Agent` run (per §2.8 —
  report-only, no doc edits) against real `google/timesfm-2.5-200m-pytorch`
  / `amazon/chronos-t5-base` activations from the already-extracted
  `runs/medium_run_chronos_base` store. Method: copied that run's
  `activations.zarr`/`meta.parquet` byte-for-byte into a new run dir
  (`runs/medium_run_chronos_base_feature_ablation_check/`, original left
  untouched), under a new config
  `configs/medium_run_chronos_base_feature_ablation_check.yaml` (exact copy
  of `medium_run_chronos_base.yaml` plus `sae.feature_ablation_enabled:
  true`, `feature_ablation_top_k: 5`, `feature_ablation_max_series: 32`),
  ran `python run.py --stages sae,report --force sae`. **Exit code 0, no
  errors/tracebacks**; the only warning was the expected, documented
  sample-cap message (`sample cap reduced: requested 32, realized 24
  (limited by batch_size)` for Chronos-T5-Base's `batch_size: 24`) — the
  feature took its real, non-empty-candidates path on both targets (154
  ground-truth-matched features for TimesFM/`stacked_xf.18`, 115 for
  Chronos-T5-Base/`encoder.block.6`), closing the exact gap named above.
  Verbatim `sae/meta.json` results:

  **TimesFM/stacked_xf.18** (`mase_clean = 2.5829224586486816`,
  `mase_full_reconstruction = 2.660618305206299`, n_series=32/32, not
  capped):
  | feature | best_field | Δ vs full-recon | Δ vs clean |
  |---|---|---|---|
  | 5109 | tier_realism_stress | −0.0003347713500261307 | 0.07736095041036606 |
  | 1457 | tier_realism_stress | 0.0024219024926424026 | 0.08011762797832489 |
  | 9419 | tier_realism_stress | 0.022360628470778465 | 0.10005635023117065 |
  | 3675 | tier_realism_stress | 0.0021889060735702515 | 0.07988462597131729 |
  | 3992 | tier_realism_stress | 0.001994139514863491 | 0.0796898603439331 |

  **Chronos-T5-Base/encoder.block.6** (`mase_clean = 2.261380910873413`,
  `mase_full_reconstruction = 2.9694175720214844`, n_series=24/32, capped by
  `batch_size`):
  | feature | best_field | Δ vs full-recon | Δ vs clean |
  |---|---|---|---|
  | 3668 | has_intermittency | 0.07875316590070724 | 0.7867897152900696 |
  | 5172 | tier_realism_stress | −0.05527125298976898 | 0.6527653336524963 |
  | 2637 | tier_realism_stress | −0.03575580567121506 | 0.6722807884216309 |
  | 263 | n_seasonalities | −0.0868653878569603 | 0.621171236038208 |
  | 3736 | tier_realism_stress | −0.3252600133419037 | 0.3827765882015228 |

  Per-family breakdowns were also returned for every row above (not
  reproduced here — see the agent's full report or rerun the config to
  regenerate `sae/meta.json`) and are non-degenerate (vary by family, not a
  flat broadcast). Every TimesFM delta is small and mixed-sign (max
  |Δ vs full-recon| ≈0.022) — consistent with a real SAE where any one of
  154 matched features carries only a sliver of the aggregate signal.
  Chronos deltas are larger and more often negative (ablating a feature
  sometimes *improves* the forecast slightly relative to the full
  reconstruction) — plausible given Chronos-T5-Base's much worse baseline
  `forecast_preservation_token` gap (ΔMASE 0.44 vs. TimesFM's 0.11 on this
  same fresh SAE retrain) leaves more room for a single feature's removal
  to move the number either way; not yet interpreted further than "the
  mechanism produces real, non-degenerate, architecture-differentiated
  numbers," which is what this verification pass was scoped to confirm.
  **One side observation, not yet investigated:** this fresh SAE retrain's
  dead-feature rates (~0.96 TimesFM / ~0.97 Chronos) read notably higher
  than the archived `medium_run_chronos_base` run's own previously-recorded
  numbers — doesn't affect whether feature-ablation itself ran correctly
  (candidate indices necessarily differ run-to-run, being drawn from a
  fresh independent SAE training), but worth a second look before trusting
  aggregate `mean_abs_rho_matched`-style claims from *this particular*
  retrained SAE specifically. New artifacts left on disk for inspection:
  `configs/medium_run_chronos_base_feature_ablation_check.yaml`,
  `runs/medium_run_chronos_base_feature_ablation_check/` (not cleaned up;
  harmless to delete, or keep as a worked reference). No code changed by
  this verification pass, so no test suite rerun was needed for it —
  separately, the full `tsfm_lens` suite was re-run this same firing after
  the unit/integration-test additions above and passed clean: **229 passed,
  0 failed, 2 pre-existing unrelated warnings (a `float16` overflow-cast
  RuntimeWarning in `test_nonfinite.py`, and a return-value
  PytestReturnNotNoneWarning in `test_smoke.py::test_end_to_end`), 292.28s**.
  This item is now closed end-to-end: mechanism implemented, unit-tested
  against a synthetic analytically-known answer, integration-tested for
  graceful degradation, and live-verified against two real checkpoints with
  real ground-truth-matched candidates. Interpreting *what* the numbers
  mean (e.g. does Chronos's larger/more-negative deltas reflect something
  real about its features vs. just its worse reconstruction baseline) is
  left as a follow-up, not a blocker on closing the implementation item.
- [x] Verbose-mode case studies for every sweep (§4): don't just report a
  dose-response curve, show 2–3 concrete series at the extremes of the sweep
  with their forecasts and lens curves side by side, narrated. — **forecast
  half done 2026-08-05** for all three sweeps (context/true-continuation/
  both-models'-forecasts, narrated, at the two extremes plus any anomaly the
  bias card flagged); see Findings below. **Not done**: per-layer skip-lens
  curves at each case-study series, mirroring `analysis/exemplars.py`'s
  existing pattern — needs the full extraction/lens machinery run per
  series, a materially larger undertaking than `adapter.predict`, left as a
  follow-up rather than attempted partially.
- [x] Consolidate bias findings into a **per-model "bias card"**: a compact,
  plain-language summary (a few sentences plus 2–3 supporting plots) of what
  each model is systematically better/worse at and under what conditions —
  this is the artifact that most directly answers the brief's original
  research questions and should be prominent in the top-level report, not
  buried in a stage-specific section. — **done 2026-08-05** as a standalone
  artifact (`runs/bias_card.html`), mirroring `report/meta_report.py`'s
  existing "cross-artifact aggregator, separate from any single run's own
  report.html" pattern rather than a new report.py section — see Findings
  below for why, and for the real generated card's content.

**Findings / decisions**
- **Controlled parameter sweeps, MASE axis (2026-08-05).** Built
  `analysis/parameter_sweep.py` (`build_recipe`/`generate_sweep_data`, pure
  and unit-tested — no GPU needed, since `parametric()` is plain numpy —
  split from the model-scoring loop the same way `sae/ground_truth.py`
  splits its pure matching function from its I/O wrapper) and
  `run_parameter_sweep.py` (reuses an existing run's model configs/
  checkpoints only; generates fresh synthetic sweep data itself, no
  benchmark corpus or store I/O — just `adapter.predict`, the same call L0
  already makes). Ran all three sweeps named in the checklist live against
  `runs/medium_run_chronos_base` (TimesFM 2.5-200M vs. Chronos-T5-Base),
  40 series per sweep point, series-level bootstrap CIs:
  - **Seasonal period (4→256, amplitude/noise/trend fixed).** TimesFM beats
    Chronos-T5-Base at every period tried except one: 4 (0.120 vs. 0.145),
    8 (0.245 vs. 0.311), 16 (0.445 vs. 0.553), 64 (0.674 vs. 0.765), 128
    (0.723 vs. 0.902), 256 (0.765 vs. 1.250) — a real, monotonically
    widening TimesFM advantage as period grows past the short end. **A
    genuine, striking anomaly at period=32**: TimesFM's MASE spikes to
    0.847 [0.757, 0.964] — *worse* than Chronos's 0.663 [0.640, 0.686] at
    that exact point, and both CIs are comfortably non-overlapping with
    their neighbors, so this isn't noise. Period=32 is exactly
    `alignment.window`/TimesFM's own patch width (`CLAUDE.md` §6.3) —
    plausibly an aliasing effect between the seasonal period and the
    patch-boundary tokenization, but **not verified as the mechanism this
    session** (would need e.g. checking whether the anomaly tracks patch
    width specifically by re-running against a TimesFM checkpoint with a
    different patch length, or phase-shifting the seasonality relative to
    the patch boundary) — flagged as a concrete, well-defined follow-up
    rather than asserted as fact.
  - **Noise scale (0.05→2.5, seasonality/trend fixed, amplitude fixed so
    this is directly an inverse-SNR sweep).** The clearest reversal in
    either sweep: at the lowest noise tried, Chronos-T5-Base is far better
    (0.188 [0.183,0.194] vs. TimesFM's 0.505 [0.473,0.538]) — the opposite
    ranking from the period sweep. Both degrade toward a similar MASE
    (~0.7–0.75) as noise grows and stay close together from noise_scale≈0.6
    onward, with TimesFM very slightly ahead at the heaviest noise tested
    (0.719 vs. 0.735 at 2.5). Read plainly: Chronos's advantage is
    concentrated specifically at *very clean* periodic signal, not a
    general noise-robustness edge — it disappears once noise_scale exceeds
    roughly 0.3 in this recipe.
  - **Intermittency rate (0→0.8, everything else fixed).** The noisiest,
    least monotonic of the three, and read with a real methodological
    caveat rather than at face value: 0 (TimesFM 0.694, Chronos 0.676), 0.1
    (0.827, 0.678), 0.2 (0.773, 0.814), 0.4 (0.837, 0.926), 0.6 (0.899,
    0.772), 0.8 (0.658, 0.581). **Caveat, not a clean finding**:
    `parametric`'s `intermittency` zeroes the *entire* series independently
    per point, including the forecast horizon itself — at rate=0.8, ~80% of
    target values are exact zeros, and `mase()`'s own scale term (mean
    absolute context step-change) also shrinks under heavy zeroing, so both
    the numerator and denominator of MASE become dominated by a floor
    effect that has little to do with either model's actual forecasting
    skill at high intermittency. This likely explains why MASE *drops* for
    both models at rate=0.8 rather than continuing to rise — predicting
    near-zero for a mostly-zero target is easy regardless of model quality.
    Not fixed this session (a scale-term redesign robust to intermittency
    is its own piece of work); stated here so the raw numbers above aren't
    mistaken for "both models get slightly better at extreme
    intermittency," which they almost certainly do not in any real sense.
    > ⚠️ **AUDIT (2026-08-06):** that "own piece of work" is now specified as
    > §15 A11, and it is bigger than this sweep — `stats.mase()` is the
    > denominator of L0's headline metric, head/MLP ablation ΔMASE, L3
    > restoration, the SAE forecast-preservation check, and every sweep, so
    > the degeneracy documented here silently propagates into all of them on
    > any zero-heavy or near-constant family. It also means
    > `intermittent_bursts` — per `CLAUDE.md` §4.2 the single sharpest
    > quantized-vs-continuous stressor in the corpus, and one of the four
    > archetypes §5.1 deliberately opted in — currently cannot be scored
    > interpretably at all.
  - 8 new unit tests (`tests/test_parameter_sweep.py`; recipe-construction
    per sweep type, sweep-data shape/label/determinism, positive-argument
    validation, and a planted-difference recovery test for the scoring
    path), all synthetic and GPU-free. Full `tsfm_lens` suite **52/52**
    after (was 44). Artifacts: `runs/param_sweep_{seasonal_period,
    noise_scale,intermittency_rate}.json` (gitignored, regenerable via the
    commands in `run_parameter_sweep.py`'s module docstring).
  - **Not done, explicit follow-up**: crystallization depth and L3
    noise-fingerprint response per sweep point (needs the full extraction/
    lens machinery run per point, a meaningfully larger undertaking than
    `adapter.predict`); verbose-mode per-sweep case studies; the per-model
    bias card consolidating this and the noise-SNR sweep below into one
    plain-language summary. Single (corpus-recipe, checkpoint-pair) result,
    same caveat as everywhere else in this file.

- **Noise-SNR × depth sweep (2026-08-05, same-day follow-up).** Built
  `run_noise_snr_sweep.py`: reuses an already-extracted run's store and
  models (no re-extraction, no new downloads) and reruns only L3's
  `noise` corruption's sensitivity fingerprint (patching disabled) at each
  of several SNR values, via the same `run_l3` the main pipeline already
  uses — cheap, since each sweep point costs exactly one forward pass per
  corruption per model over `l3.max_series` series, reusing the store's
  already-captured clean activations. Ran against
  `runs/medium_run_chronos_base` (TimesFM 2.5-200M vs. Chronos-T5-Base,
  the L1/L2/L3 comparison pair already in that run's config), 96 series,
  SNR ∈ {20, 12, 6, 3, 0, −3} dB, ~82 seconds total on one GPU.
  - **The ρ≈−0.90 anti-correlation is not a fixed property of one
    arbitrary corruption strength — it's a real dose-response transition.**
    At mild noise (SNR=20dB) the depth-profile agreement is
    **positive**, ρ=+0.397 [0.16, 0.60] (CI excludes zero). It **flips
    sign** by SNR=12dB, ρ=−0.439 [−0.60, −0.19] (CI excludes zero, opposite
    direction). From SNR=6dB down through −3dB it settles into a broad
    **plateau of strong, stable anti-correlation**: ρ=−0.852, −0.851,
    −0.825, −0.764 respectively (all CIs comfortably excluding zero,
    typically ±0.1). The main pipeline's default battery happens to use
    SNR=6dB, which this sweep shows sits squarely inside that plateau, not
    at some fragile edge case — the previously-reported number was real
    and representative of a wide noise range, not a fluke of one setting.
  - **Each model's *own* peak-sensitivity layer never moves across the
    entire SNR range** — TimesFM's noise-sensitivity fingerprint peaks at
    its very first captured layer (index 0, right after patch embedding)
    and Chronos-T5-Base's peaks at its very last captured encoder layer
    (index 11 of 12) at *every single SNR tested*, mild or heavy. Only the
    *sign/strength* of the cross-model rank agreement changes with noise
    level, not where each model's own sensitivity concentrates. Read
    together with the sign flip: at mild noise, TimesFM's early-peaking
    and Chronos's late-peaking profiles happen to still rank-correlate
    positively (both may show a broadly similar shape away from their
    respective peaks); as noise grows, the profiles increasingly diverge
    in a rank sense even though each one's own peak location is unchanged.
  - **The *why* is still open** — this sweep characterizes the effect's
    shape precisely but does not yet test the "continuous embedding vs.
    quantization" hypothesis directly. That would need a matched sweep on
    a corruption that stresses quantization specifically (e.g. comparing
    `noise` against `intermittent_bursts`-style zero-runs, or directly
    inspecting Chronos's token-ID churn under the same SNR sweep the way
    `CLAUDE.md` §11.16 did for the alignment-check tool) — named here as
    the concrete next step, not attempted this session.
  - **Single (corpus, checkpoint-pair) result** — same caveat this session
    applied everywhere else: `runs/medium_run_chronos_base` is one corpus
    against TimesFM 2.5-200M vs. Chronos-T5-Base specifically. A second
    checkpoint pair (e.g. `runs/medium_run`'s Chronos-T5-*Small*) would be
    the natural replication check, but that run's zarr store was written
    by a different environment's zarr version (v3-formatted `zarr.json`
    metadata, vs. this environment's pinned v2) and isn't readable here
    without re-extraction — flagged as a live, reproducible instance of
    exactly the environment-drift risk `CLAUDE.md` §11.15 already warns
    about, not re-litigated further this session.
  - Artifact: `runs/noise_snr_sweep.json` plus one `l3_snr_sweep_*`
    subrun directory per SNR value under `runs/` (all gitignored, not
    committed, regenerable via the command in `run_noise_snr_sweep.py`'s
    module docstring).

- **Quantization-churn probe (2026-08-05, second same-day follow-up) — the
  concrete next step named above, now attempted.** Built
  `analysis/quantization_churn.py` (`token_churn_stats`, a pure/tested
  statistic split from I/O per the same pattern `sae/ground_truth.py`
  already uses) plus `run_quantization_churn_sweep.py`, which reuses the
  *exact* 96-series row sample and `corrupt_noise` draws the sweep above
  used (same rng derivation, so points line up 1:1 against the already-
  published rho values) and tokenizes clean vs. corrupted contexts through
  whichever model in the pair exposes a quantization tokenizer
  (`pipeline.tokenizer.context_input_transform`) — skipping
  continuous-embedding models like TimesFM with a log rather than
  fabricating something to measure there (`CLAUDE.md` §2.5). No
  re-extraction, no new downloads, and — unlike every other sweep in this
  file — no forward passes at all (`prepare()` only tokenizes). Ran live
  against `runs/medium_run_chronos_base`'s Chronos-T5-Base.
  - **Churn *fraction* saturates almost immediately and does not track the
    sign flip — a real, if initially counter-intuitive, result.** Even at
    the mildest SNR tested (20dB, noise_std ≈ 0.1× signal_std), **96.4%** of
    tokens already land in a different quantization bin than their clean
    counterpart (CI [0.956, 0.971]), rising only to 99.5% by −3dB.
    `amazon/chronos-t5-base`'s tokenizer (`MeanScaleUniformBins`, 4094
    usable bins over a per-series-rescaled [−15, 15] range) is fine-grained
    enough that almost *any* perturbation reassigns some bin — churn
    fraction has essentially no dynamic range across this sweep, a ceiling
    effect, not evidence against the quantization hypothesis.
  - **Jump *magnitude* (mean |bin-index change| among changed tokens) has
    real dynamic range and partially tracks the sign flip's onset.** Mean
    jump size: 13.51 (20dB) → 32.30 (12dB) → 59.08 (6dB) → 76.99 (3dB) →
    96.88 (0dB) → 116.80 (−3dB) — monotonically increasing, as expected for
    heavier noise. The two largest *relative* jumps in mean jump size
    (20dB→12dB, ×2.4; 12dB→6dB, ×1.8) line up with the two biggest moves in
    rho from the sweep above (the sign flip +0.40→−0.44, then the
    deepening −0.44→−0.85) — a real, though correlational and only
    n=6-points-observed, alignment between "how far tokens jump when they
    do change" and "how much the cross-model depth-profile agreement
    moves."
  - **The plateau region complicates a clean quantization story — stated
    plainly rather than glossed over.** From 6dB down through −3dB, mean
    jump size keeps climbing at a *decelerating* relative rate (×1.30,
    ×1.26, ×1.20) while rho stays essentially flat (−0.852, −0.851, −0.825,
    −0.764) instead of continuing to deepen in step. If jump magnitude
    alone drove the cross-model disagreement, rho would be expected to keep
    intensifying as jump size keeps growing through this range — it
    doesn't. Read plainly: quantization jump severity is a plausible
    explanation for *why the sign flips* between mild and moderate noise,
    but not a complete explanation for the plateau's flat shape —
    something else (plausibly each model's own sensitivity fingerprint
    saturating once corruption is severe relative to the signal, a
    property that wouldn't be specific to Chronos's tokenizer at all) likely
    contributes once noise is heavy. Not decomposed further this session —
    the concrete next step if this thread is picked up again.
  - Sanity-checked the jump-size numbers against the tokenizer's own config
    rather than trusting them at face value: a mean jump of 13.5 bins at
    20dB corresponds to roughly a 0.1 z-score-unit shift (13.5 × 30/4094),
    in line with a ~0.1×-signal-std noise injection after the tokenizer's
    own per-series rescaling — the numbers are calibrated to the actual
    mechanism, not an artifact of the measurement.
  - 7 new unit tests (`tests/test_quantization_churn.py`; all synthetic —
    identity/full-churn, mask-respecting churn, jump-size accounting,
    shape/zero-valid-position validation, bootstrap-CI bounds), all against
    `token_churn_stats`'s pure logic, not a live tokenizer. Full `tsfm_lens`
    suite **44/44** after (was 37). Artifact: `runs/quantization_churn_sweep.json`
    (gitignored, not committed, regenerable via the command in
    `run_quantization_churn_sweep.py`'s module docstring).

- **Per-model bias card (2026-08-05).** Built `report/bias_card.py`
  (`compare_at_point`/`summarize_param_sweep`/`build_bias_card`, pure and
  unit-tested against synthetic sweep dicts, plus `render_bias_card_html`
  for the standalone-HTML I/O — split the same way `sae/ground_truth.py`
  and `analysis/parameter_sweep.py` already split pure logic from I/O) and
  `run_bias_card.py`, which consolidates whatever `param_sweep_*.json`
  files exist (glob default) into one plain-language summary per model.
  **Built as a standalone artifact (`runs/bias_card.html`), not a new
  `report.py` section** — a deliberate choice, not an oversight: the
  sweep JSONs are cross-artifact inputs generated outside any single run's
  own directory (same situation `report/meta_report.py` already solved for
  cross-run artifacts), so this follows that established precedent rather
  than inventing a second mechanism for the same kind of problem
  (`CLAUDE.md` §2.2/§2.5's "one well-tested mechanism over parallel ones").
  A model is only ever called "favored" at a sweep point when its bootstrap
  CI does not overlap the other model's — a raw point-estimate ratio would
  read noise as a finding.
  - Ran live against this session's three real sweep artifacts
    (`param_sweep_{seasonal_period,noise_scale,intermittency_rate}.json`,
    §7's Findings above). The generated card's verdicts corroborate and
    sharpen the hand-written Findings above, entirely mechanically (no
    numbers hand-picked for the card): **TimesFM** favored at 6/7
    confidently-different `seasonal_period` points, with the period=32
    point flagged as an **anomaly** against that sweep's own plurality
    (discovered generically — "the favored model differs from the sweep's
    own plurality winner" — not hardcoded to period=32); **Chronos-T5-Base**
    favored at 2/2 confidently-different `noise_scale` points — the
    CI-overlap requirement mechanically discovered that only the *lowest*
    two noise levels are confidently distinguishable at all, exactly
    matching the hand-written finding that Chronos's edge is concentrated
    at very clean signal and closes by moderate noise; and Chronos favored
    at 2/3 confidently-different `intermittency_rate` points (TimesFM
    anomalous at rate=0.4), both models' cards correctly carrying the
    intermittency scale-term caveat verbatim from §7's Findings above.
  - 6 new unit tests (`tests/test_bias_card.py`: CI-overlap comparison,
    plurality/anomaly detection with a planted single anomaly, a
    two-model-only guard, a no-confident-points degrade path, cross-sweep
    caveat attachment, and an HTML-render smoke test), all synthetic. Full
    `tsfm_lens` suite **58/58** after (was 52). Artifacts:
    `runs/bias_card.html` + `runs/bias_card.json` (gitignored, regenerable
    via `run_bias_card.py`).
  - **Not done**: integrating this into `report.py` itself, or a per-model
    plot beyond the existing dose-response line charts (e.g. a compact
    single "scorecard" figure); verbose-mode narrated case studies at each
    card entry's most extreme sweep point (bullet 4 — **now done, see the
    next Findings entry**). Single (corpus-recipe, checkpoint-pair) result,
    same caveat as everywhere else in this file.

- **Verbose-mode sweep case studies, forecast half (2026-08-05).** Built
  `report/sweep_case_studies.py` (`select_case_study_values`, pure and
  unit-tested — always the two sweep extremes, plus any point
  `report/bias_card.py::summarize_param_sweep` flagged as anomalous, so
  "interesting" is discovered the same generic way the bias card already
  discovers it, not hardcoded per sweep; `render_case_studies_html` for the
  I/O) and `run_sweep_case_studies.py`, which generates one fresh example
  series per selected value, runs the real models' forecasts on it, and
  renders a narrated context/true-continuation/both-models'-forecasts
  comparison — the same escalation `analysis/exemplars.py`'s own module
  docstring describes for family-level exemplars ("aggregate statistics
  answer *which* model is better; exemplars answer *what that looks
  like*"), applied to sweep points instead of families.
  - Ran live against all three of this session's real sweep artifacts
    (`runs/medium_run_chronos_base`). Each correctly selected the sweep's
    own extremes plus its already-known anomaly with no manual
    intervention: `seasonal_period` → {4, 32 (anomaly), 256}; `noise_scale`
    → {0.05, 0.6, 2.5} (no anomaly in this sweep, so the middle value fills
    the third slot, exactly the fallback the selection logic is designed
    for); `intermittency_rate` → {0, 0.4 (anomaly), 0.8}. Every case study's
    printed per-model MASE matches the aggregate sweep numbers already in
    this file's Findings above, confirming the case-study series are drawn
    from the same sweep the aggregate table describes, not an inconsistent
    parallel generation.
  - 7 new unit tests (`tests/test_sweep_case_studies.py`: extremes+middle
    selection, anomaly-priority over the middle fallback, single- and
    two-value edge cases, a max-points cap, an empty-values guard, and an
    HTML-render smoke test), all synthetic. Full `tsfm_lens` suite
    **65/65** after (was 58). Artifacts:
    `runs/sweep_case_studies_{seasonal_period,noise_scale,
    intermittency_rate}.html` (gitignored, regenerable via the command in
    `run_sweep_case_studies.py`'s module docstring).
  - **Not done, explicit follow-up**: per-layer skip-lens curves at each
    case-study series (needs `ActivationStore`/hooks run per series, not
    just `adapter.predict` — a materially larger undertaking); attention
    maps (same limitation). This closes the forecast half of bullet 4 only,
    as the checklist above now states plainly rather than marking the
    bullet fully done.

---

## 8. Cross-cutting: explainability requirement (brief item 5, expanded)

Restating as a standing requirement rather than a one-time Phase 0 task,
since every phase above adds new plots/sections:

- **No new axis, colorbar, or section may ship without a plain-language
  explanation** reachable from the report itself (tooltip, footnote, or
  linked glossary entry per §4) — not just documented in `CLAUDE.md` or this
  file. A reader without repo access is the bar.
- **State the evidence class every time** (`CLAUDE.md` §2.6/§6.6): is this
  number geometric, linearly-translatable, causal-within-model, descriptive,
  or illustrative? The existing per-section blurbs mostly do this at the
  section level; extend it to individual plots where a single section mixes
  evidence classes (e.g. the Attention section mixes descriptive lag-profile
  taxonomy with genuinely causal ablation ΔMASE — these need distinct
  framing within the same section, not one blurb covering both).
- **State sign conventions locally, every time**, not just once somewhere —
  a reader skimming won't have read the one place it was defined (§4's ΔMASE/
  ΔR²/Δact note).
- **`--verbose` is the mechanism**, not a substitute, for this — verbose
  exemplars make findings concrete, but the *aggregate* plots still need
  self-contained explanations for readers who don't have exemplar budget to
  read through.

---

## 9. Phase 4 — Multi-model expansion (brief item 4)

**Goal.** Prove the `ModelAdapter` abstraction (`CLAUDE.md` §6.2) generalizes,
deliberately late (§3) so it's stress-tested by the hard two-model case
first. Do not start this phase until Phase 2b (SAEs) and Phase 3 (ablation)
have gone through at least one full cycle on TimesFM/Chronos — the goal here
is to *reuse* that methodology on a new model, which is also the real test of
whether it's reusable.

**Candidate models, roughly in priority order (revisit before starting —
availability/licensing may have changed):**
- [x] **Chronos-2** (if released/available) — same lineage as the existing
  Chronos adapters, likely the lowest-effort addition and a good smoke test
  of the adapter-add process itself before tackling something structurally
  different. — **done 2026-08-07**: verified released (not guessed) —
  `amazon/chronos-2` exists on HF, `chronos-forecasting==2.3.1` (already
  installed) exposes `Chronos2Pipeline`, no new package needed. Turned out
  to be architecturally distinct enough to be a genuine Phase-4 stress test,
  not just a smoke test: encoder-only (no decoder), 12 blocks, 768 dim, each
  block runs TIME self-attention (within one series) then GROUP
  self-attention (across series sharing a `group_id`, along the *batch*
  axis) then feed-forward, and processes context + a `[REG]` token +
  forecast-horizon placeholders together in one non-causal encoder pass —
  no separate decode step at all. See Findings below for the real numbers
  and the one abstraction bug this surfaced.
- [ ] **Moirai (uni2ts)** — masked-encoder, multivariate-native, patch-based;
  structurally distinct from both existing adapters (handles multivariate
  series and variable patch sizes natively) — a good test of whether the
  alignment/pooling machinery (`CLAUDE.md` §6.3) holds up against a model
  whose tokenization isn't a simple fixed patch width. ⚠️ **Correction
  (2026-08-10, this section's own licensing-recheck item below): the
  "masked-encoder" description above is stale for the current release.**
  `Salesforce/moirai-2.0-R-small` (Aug 2025) moved to a **decoder-only**
  design; only the older 1.x line (`moirai-1.1-R-small`, 13.8M params) is
  the masked-encoder this bullet describes. Both are `cc-by-nc-4.0`
  (non-commercial) — fine for this repo's research use, but a real
  constraint, not previously stated. Pick 1.x specifically if the
  masked-encoder structural contrast is the point of adding this model;
  2.0 would instead be a third decoder-only architecture alongside TimesFM
  and (if added) Sundial, not a structural contrast to either.
- [x] **Sundial** — decoder-only, confirmed (2026-08-10, was open above) —
  patch length 16, **Apache-2.0** (fully permissive, no NC constraint
  unlike Moirai), pretrained on ~1 trillion time points (`thuml/
  sundial-base-128m`, ICML 2025 Oral). Its probabilistic head is
  flow-matching (`TimeFlow Loss`) — a third distinct sampling mechanism
  next to TimesFM's deterministic quantile head and Chronos's discrete-token
  sampling; `predict()`'s design will need to map flow-matching samples onto
  the existing `{"point": ..., quantiles...}` contract, mirroring how
  Chronos's `num_samples` already does this for a sampled decoder. Given
  the license difference and that Moirai 2.0 no longer offers as clean a
  structural contrast, **Sundial is now the more attractive next pick**
  of the two, ahead of where this list originally ranked it — see this
  section's Findings and §13's licensing item for the full recheck. —
  **done 2026-08-10, same day as the licensing recheck above**: adapter
  built, registered, alignment-verified, live-run confirmed against
  TimesFM. Full checklist and Findings below.
- [ ] Others worth scanning for fit against the `ModelAdapter` contract as
  they mature: Toto, Time-MoE, Tiny Time Mixers, Lag-Llama — don't commit to
  these without first checking checkpoint availability and license terms,
  the same diligence already applied to benchmark data sources
  (`CLAUDE.md` §4.3).

**Concrete deliverables per model (repeat this checklist per addition):**

**Chronos-2 — done 2026-08-07:**
- [x] Subclass `ModelAdapter`, register in `models/__init__.py`. — new
  `tsfm_lens/models/chronos2_adapter.py` (`Chronos2Adapter`), registered as
  `"chronos2"` in `models/__init__.py`'s `_register_optional`.
- [x] Run `--discover-layers`, pick `layer_regex`, run `--check-alignment`
  and confirm near-1.0 diagonal dominance at early layers before trusting
  anything (`CLAUDE.md` §6.3, invariant 7). — `--discover-layers` confirmed
  the default `encoder\.block\.\d+$` regex matches this checkpoint's 12
  blocks with no override needed; `--check-alignment Chronos-2` (via new
  `configs/chronos2_adapter_check.yaml`) showed a **perfect 1.00 diagonal-hit
  fraction at all 12 layers** — the token-span/patch-geometry math (mirrored
  from `ChronosBoltAdapter`'s own front-padding formula, since both share
  `chronos.chronos_bolt.Patch`'s convention) is correct on the first try.
- [x] Fill in the capability matrix (`CLAUDE.md` §6.2) honestly — which
  optional capabilities (`attention_patterns`, `cross_attention_patterns`,
  head/MLP info) this model supports, and confirm unsupported ones degrade
  with a log rather than breaking (`CLAUDE.md` invariant 8). — `python
  run_capability_matrix.py` auto-detected `chronos2` declaring
  `attention_info`/`mlp_info`/`attention_patterns` (all real, verified live
  below) and correctly `❌` for `cross_attention_patterns` (encoder-only, no
  decoder — never overridden, so it degrades to the base class's `None`
  automatically). **Scope note, stated not hidden**: Chronos-2 runs a
  *second* attention sub-layer per block (GROUP self-attention, across
  series along the batch axis) that this mechanism does not expose —
  `_scan_attention`/`_scan_mlp` (the shared helpers `chronos_bolt_adapter.py`
  already uses) return on the first matching submodule, which is always the
  TIME self-attention; GROUP heads are simply out of scope for head-level
  ablation/pattern capture, documented in the adapter's own docstrings
  rather than silently claimed as full coverage.
- [x] Confirm §2.4's success bar: no changes needed outside the new adapter
  file and its registration. Any exception is a bug in the abstraction — file
  it as a Phase 4 finding and fix the abstraction before moving to the next
  model. — **one exception found, filed, and fixed per this item's own
  contingency plan**: `extraction/hooks.py`'s `_primary()` crashed on the
  very first real capture (`--check-alignment`), because
  `Chronos2EncoderBlock.forward` returns an HF `ModelOutput`-style dataclass
  (dict-like, integer-indexable, but **not** an instance of `tuple` —
  `isinstance(output, tuple)` silently missed it). Fixed in the shared
  module, not the adapter — see this section's Findings for the full
  mechanism and the new regression test.
- [x] Auto-generate the capability matrix from the registry (see Phase 5)
  rather than hand-maintaining a table that will drift. — **now actually
  exercised against a real Phase-4 addition**: `run_capability_matrix.py`'s
  output above is exactly this generator, run for the first time against a
  newly-added model rather than only the pre-existing three.

**Findings / decisions**
- **2026-08-07 — Chronos-2 added; a real abstraction bug found and fixed;
  live numbers against TimesFM confirm the pipeline treats it as a genuine
  third architecture, not a Chronos-T5 clone.** `amazon/chronos-2` (120M
  params, `d_model=768`, 12 layers, 12 heads, patch=16, `use_reg_token=True`,
  `use_arcsinh=True`, verified via its own `config.json` and HF README, not
  assumed) is encoder-only: no decoder, and each block runs TIME
  self-attention → GROUP self-attention (across series sharing a
  `group_id`, along the *batch* axis, not token-token — out of scope here)
  → feed-forward. Crucially, the encoder processes **context + `[REG]` +
  forecast-horizon placeholders together in one non-causal pass** — there
  is no separate decode step at all, unlike Chronos-T5 (real
  encoder-decoder) or Chronos-Bolt (encoder + a small direct quantile
  head). `token_time_spans`/`postprocess_tokens` keep only the leading
  context-patch positions (mirroring `ChronosBoltAdapter`'s own
  trailing-`[REG]`-strip pattern, generalized to also strip the trailing
  forecast placeholders) — confirmed correct by the perfect 1.00 alignment
  check above, not just asserted.
  - **The one abstraction bug (`hooks.py`), full mechanism.** `_primary()`
    read `output[0] if isinstance(output, tuple) else output`, then called
    `.detach()` on the result. `Chronos2EncoderBlockOutput` (and every HF
    `transformers.utils.ModelOutput` subclass) is a dataclass that
    subclasses `OrderedDict` — supports `output[0]` indexing for its first
    field, but `isinstance(output, tuple)` is `False` — so `_primary`
    returned the whole dataclass unchanged, and `.detach()` raised
    `AttributeError`. Fixed by checking for `torch.Tensor` first instead of
    `tuple` (`return output if isinstance(output, torch.Tensor) else
    output[0]`) — correct for tensor, tuple, *and* any indexable dataclass
    output uniformly. A second, related gap: `token_patch`/
    `output_mean_ablate`'s hooks reconstructed a patched/ablated tuple
    output via `(patched,) + tuple(output[1:])`, which would have silently
    **degraded a dataclass output to a plain tuple** — losing attribute
    access to any other field (e.g. a block's own attention weights) for
    every later consumer of that forward pass. Fixed with a new shared
    `_rebuild(output, replacement)` helper: preserves `torch.Tensor` as-is,
    rebuilds a plain `tuple` as before, and for any `dataclasses.is_dataclass`
    output uses `dataclasses.replace(output, **{first_field: replacement})`
    to swap only the primary field, keeping every other field intact and
    the exact original type. New `tests/test_hooks_modeloutput.py` (3
    tests, all passing) exercises `ActivationCatcher`/`token_patch`/
    `output_mean_ablate` directly against a synthetic `ModelOutput`-style
    module with no real checkpoint or network access needed, so this fix
    has fast offline regression coverage independent of whether a future
    session can reach Chronos-2's weights at all. **This is exactly the
    "any exception outside the new adapter file is a bug in the
    abstraction" case this phase's own checklist was written to catch** —
    filed and fixed in the shared module, per that instruction, rather than
    worked around inside `chronos2_adapter.py`.
  - **Real numbers from a live GPU validation run**
    (`configs/chronos2_phase4_check.yaml`: Chronos-2 vs TimesFM, same
    `benchmark_medium/public_dev` corpus as `medium_run_chronos_base`, with
    `l0`/`l1`/`internals`/`l3`(**with per-window patching**, exercising the
    `token_patch` half of the hooks.py fix)/`attention`(**patterns + head/MLP
    ablation**, exercising `output_mean_ablate`/`input_slice_ablate`)/`report`
    enabled — deliberately the two mechanisms a lean L0/L1/L2-only run
    wouldn't touch). Completed clean end-to-end, 5 report sections / 17
    findings, no traceback:
    - **L0**: paired ΔMASE −0.14 [−0.21, −0.06] (favors Chronos-2 overall,
      p=0.003); Chronos-2 significantly stronger on `nonsinusoidal_seasonal`
      and `random_parametric` (Holm-corrected).
    - **Profile**: family information peaks at block 3 of 12 for Chronos-2
      (probe 0.98 vs chance 0.65) vs. the *last* captured layer for TimesFM
      (probe 0.97 vs chance 0.65, at its stride-2 layer 10 of 10) — Chronos-2
      crystallizes family identity much earlier in its (shorter) depth.
    - **L1**: peak CKA **0.43** [0.42, 0.46] at TimesFM-relative-depth 0.22
      ↔ Chronos-2-relative-depth 0.64, decisively above this run's own
      shuffled-series null (≈0.04) — real, non-trivial geometric alignment
      between TimesFM and a third, architecturally distinct model, not just
      the two flagship architectures.
    - **L3**: fingerprint agreement **ρ=−0.75** [−0.76, −0.73] overall —
      a strong *anti*-correlation, the same qualitative pattern
      `CLAUDE.md` §6.5/ROADMAP.md §5's Findings already documented between
      TimesFM and Chronos-T5 (there ρ≈−0.90), now replicated against a
      third, structurally different model; `noise` is again the most
      divergent single corruption (ρ=−0.39 [−0.51,−0.35]), consistent with
      the noise-sensitivity anti-correlation this repo has flagged as a
      concrete causal lead since §7's Phase-3 work.
    - **Attention**: Chronos-2's strongest periodicity head is L3·h5
      (excess seasonal mass 0.41, family `mixture`) and its most
      load-bearing head is L6·h9 (ΔMASE +0.070 when mean-ablated) — real,
      non-degenerate, sane numbers, confirming `attention_info`/
      `attention_patterns`/head-ablation all function correctly end-to-end
      for the new architecture, not just in the isolated alignment check.
  - Full `tsfm_lens` test suite green at **203/203** (200 before this
    session's Phase-4 work, +3 new `test_hooks_modeloutput.py` tests).
    Per this repo's established precedent (`chronos`/`chronos_bolt`/
    `timesfm` have no pytest coverage either — see
    `test_adapter_conformance.py`'s own docstring), no pytest test imports
    real Chronos-2 weights; the live numbers above are the validation,
    exactly mirroring how every other real-checkpoint adapter in this repo
    has been validated.
  - **Not done this session**: Chronos-2 is not yet part of any flagship
    comparison config (`default.yaml`/`medium_run*.yaml`), has no
    `random_init` null-baseline pair built, and its GROUP (cross-series)
    attention axis remains entirely unexplored by this pipeline — all
    reasonable next steps, not attempted here since this item's own scope
    is "prove the adapter, not run the full battery."

**Sundial — done 2026-08-10:**
- [x] Subclass `ModelAdapter`, register in `models/__init__.py`. — new
  `tsfm_lens/models/sundial_adapter.py` (`SundialAdapter`), registered as
  `"sundial"`.
- [x] Run `--discover-layers`, pick `layer_regex`, run `--check-alignment`
  and confirm near-1.0 diagonal dominance before trusting anything
  (`CLAUDE.md` §6.3, invariant 7). — module structure verified directly
  against the loaded checkpoint (not assumed from the HF card): 12
  `SundialDecoderLayer` blocks (`model.layers.{i}`), `d_model=768`, 12
  heads/64 head_dim, patch=16. `--check-alignment` found a real,
  **amplitude-dependent** diagonal-hit pattern — perfect 1.00 at every
  layer at small impulse amplitudes, decaying by mid-depth at this repo's
  default 0.25× amplitude, with zero backward/future leakage at any
  amplitude tested. Diagnosed as a genuine architectural property (plain
  residual blocks, no QK-norm) rather than a broken adapter, exactly per
  `CLAUDE.md` §11.16's precedent for not trusting a low diagonal-hit
  number without checking why first. Full mechanism now in `CLAUDE.md`
  §11.22 (a real limitation of only checking the shallowest layer, which
  is where this pattern would be invisible).
- [x] Fill in the capability matrix honestly. — `attention_info`/`mlp_info`
  supported (attention via the shared `_scan_attention`; MLP hard-coded to
  `{block}.ffn_layer` since `_scan_mlp` doesn't recognize that leaf name —
  same class of gap as Chronos-T5's `ff0`/`ff1`, same fix: hard-code rather
  than extend the shared scanner). `attention_patterns` correctly declared
  unsupported and verified to degrade with a log, not a crash — the
  checkpoint's own `output_attentions=True` path is independently broken
  upstream, and recovering weights would need a global
  `scaled_dot_product_attention` monkeypatch, judged not worth the fragility
  (same call already made for Chronos-Bolt). `cross_attention_patterns`
  correctly `n/a` (decoder-only, no encoder-decoder split).
- [x] Confirm §2.4's success bar: no changes needed outside the new adapter
  file and its registration. — **held this time**, in contrast to Chronos-2:
  Sundial's block returns a plain tensor/tuple, already handled correctly
  by `hooks.py`'s existing (post-§11.21) `_primary`/`_rebuild` logic. The
  only bugs found were in the **checkpoint's own remote code**
  (`transformers.DynamicCache` API mismatch breaking both `.generate()` and
  any cached `forward()` call), not this repo's shared infrastructure —
  worked around inside `sundial_adapter.py` by calling
  `SundialForPrediction.forward(..., use_cache=False)` directly rather than
  `.generate()`, deliberately **not** downgrading the shared `cudaPy` env's
  `transformers==4.57.6` (which would risk the `CLAUDE.md` §11.8-class
  regression of breaking the other three adapters over one new one). Full
  mechanism in `CLAUDE.md` §11.22.
- [x] Auto-generate the capability matrix from the registry rather than
  hand-maintaining a table that will drift. — `run_capability_matrix.py`
  confirmed correct output for `sundial` alongside the other four models.

**Findings — 2026-08-10.** `thuml/sundial-base-128m` (Apache-2.0, ~1
trillion time points pretraining, ICML 2025 Oral) added as a fourth real
adapter with no changes to shared infrastructure — a useful contrast with
Chronos-2's §11.21 finding, showing that finding was a real, now-closed
gap rather than "every new architecture breaks something new here."
- **Live comparison run** (`configs/sundial_phase4_check.yaml`, TimesFM vs
  Sundial, same `benchmark_medium/public_dev` corpus as `medium_run_
  chronos_base.yaml`/`chronos2_phase4_check.yaml`, l0+l1+internals+l3+
  attention, ~1 min GPU wall-clock, spot-checked against the actual
  `runs/sundial_phase4_check/l1/meta.json` artifact rather than trusted
  from the report alone): clean end-to-end, 5/5 sections, 0 failures.
  - **L0**: overall paired ΔMASE (Sundial−TimesFM) = +0.146 [−0.008,+0.306],
    p=0.06 — not significant overall. TimesFM significantly stronger on
    `mixture` specifically (+0.118 [0.090,0.148], p_holm=0.01); no
    significant family-level difference on `random_parametric`/`parametric`.
  - **L1**: peak CKA **0.381** [0.359,0.410] (TimesFM `stacked_xf.4` ↔
    Sundial `model.layers.11`) against this run's own shuffled-series null
    of **0.022** [0.020,0.023] — decisive, non-trivial shared geometry with
    a third distinct architecture, replicating the pattern Chronos-2
    already established (CKA 0.43 vs. null ≈0.04) at a comparable
    magnitude.
  - **Internals**: family-probe peak 0.974 (Sundial, final layer 11 of 12)
    vs. 0.967 (TimesFM, mid-depth `stacked_xf.10`), both far above chance
    (0.653) — Sundial crystallizes family identity latest-in-depth of any
    model in this repo so far (contrast with Chronos-2's *earliest*-in-depth
    crystallization at block 3 of 12, §9's Chronos-2 Findings above) —
    genuinely different depth-organization strategies across the four
    models now on record, not a repeated pattern.
  - **L3**: fingerprint agreement overall **ρ=0.517** [0.451,0.560] — a
    real *positive* correlation, qualitatively different from the strong
    *anti*-correlation this repo has otherwise documented for every
    TimesFM-vs-Chronos-family pairing (TimesFM/Chronos-T5 ρ≈−0.90,
    TimesFM/Chronos-2 ρ≈−0.75). `noise` is still the most divergent single
    corruption (ρ=−0.80, same qualitative lead as every other pairing so
    far), but the *overall* sign flip is a genuinely new data point: the
    anti-correlation pattern this repo had started to treat as a
    cross-architecture regularity does not hold for Sundial, worth
    revisiting once a fifth model exists to see which pairing is the
    outlier.
  - **Attention**: most load-bearing head layer.6·h1 (ΔMASE +0.059); MLP
    ablation produced sane numbers across all 12 layers; `attention`
    section correctly logged `patterns: unsupported` for Sundial with no
    crash, per the capability-matrix deliverable above.
- **Test suite**: 203 → **207 passing** (4 new tests in
  `tests/test_sundial_adapter.py`, no live weights required, matching this
  repo's established precedent for adapter tests). Independently re-run
  and confirmed (207 passed, 2 pre-existing benign warnings unrelated to
  this change) before this Findings entry was written, not just taken on
  the implementing session's word.
- **Not done this session** (same shape of gap as Chronos-2's own "not
  done" list): Sundial is not yet part of any flagship comparison config,
  no `random_init` null-baseline pair has actually been run for it
  (mechanism should already work via the existing `ModelConfig.
  random_init` flag, untested live), and `attention_patterns` stays
  unsupported by deliberate choice, not attempted further.
- Moirai/others remain **not started**. Sundial was correctly the more
  attractive of the two remaining named candidates once the licensing
  recheck (§13) surfaced Moirai's non-commercial restriction and its
  architecture drift to decoder-only in 2.0 — and, like Chronos-2, turned
  out to validate the `ModelAdapter` abstraction cleanly (this time with
  zero shared-infrastructure changes at all, a stronger form of the same
  validation).

---

## 10. Phase 5 — Productization: "transformer-lens for TSFMs"

**Goal.** Fold everything learned about what actually mattered (which
metrics predicted useful layers, which SAE variant worked, which ablations
were informative, what generalized cleanly to new models) into a library
someone outside this project can pick up and use on a model this repo's
authors have never seen.

**Concrete deliverables**
- [ ] **API/docs pass**: a quickstart that takes a new user from "I have a
  HF checkpoint" to "I have a report" in a few commands, modeled on
  `transformer-lens`'s own onboarding experience (`CLAUDE.md` §1, §14).
- [x] **Auto-generated model-zoo capability matrix** (from Phase 4) rendered
  as part of the docs, not hand-maintained. — **done 2026-08-06**: new
  `tsfm_lens/models/capability_matrix.py` + `run_capability_matrix.py`,
  see this section's checklist entry below (paired with §9's own
  "auto-generate the capability matrix from the registry" bullet, the same
  deliverable named twice from two angles). Not yet actually wired into any
  docs page (there is no docs page yet, per this phase's other still-open
  items) — the generator exists and is verified; publishing its output
  somewhere readers see it is left to whichever future session builds the
  Phase 5 docs pass.
- [ ] **Tutorial notebooks**: at minimum, (a) run the smoke config end to
  end, (b) add a new toy adapter from scratch, (c) train and interpret an
  SAE on one layer, (d) read a confirm-stage verdict correctly.
- [~] **CI**: golden-hash regression (`CLAUDE.md` §11.1) plus the smoke test
  (`CLAUDE.md` §9) on every change; a lightweight adapter-conformance test
  that any new `ModelAdapter` can be run against (discover-layers,
  check-alignment, degrade-gracefully checks) so Phase 4's per-model
  checklist becomes partly automated. — **conformance-test half done
  2026-08-06**: new `tsfm_lens/models/conformance.py`
  (`check_adapter_conformance`) automates exactly the manual checklist this
  bullet and Phase 4's per-model deliverables (§9) already describe by
  hand — layer discovery non-empty, `token_time_spans()` well-formed
  (positive-width, strictly increasing spans), `impulse_alignment_check`
  runs and returns fractions in `[0, 1]`, `predict()` returns a finite,
  correctly-shaped point forecast, and every optional capability
  (`attention_info`/`mlp_info`/`attention_patterns`/
  `cross_attention_patterns`) either returns a well-formed value or `None`
  — never raises (`CLAUDE.md` invariant 8). New
  `tests/test_adapter_conformance.py` runs it against all three registered
  mock adapters, including `mock_wave` (the third architecture that
  already existed in `models/mock.py` specifically to prove new adapters
  need no changes outside their own file, per this section's and §9's own
  success bar, but that no default config actually exercises), plus one
  test proving an adapter with none of the optional capabilities still
  passes (the "None is fine" contract) and one proving a deliberately
  broken adapter (non-monotonic `token_time_spans`) is caught, not
  silently accepted. Full suite 74/74 after adding it. **Not done, and
  correctly not attempted**: wiring this into an actual CI pipeline
  (there is no CI config in this repo yet — this bullet's "CI" framing is
  aspirational until Phase 5's packaging/CI infrastructure exists) and the
  golden-hash regression's own independent, unresolved numpy-version
  fragility (`CLAUDE.md` §11.13) — this conformance checker doesn't touch
  or fix that; it only gives the model-adapter half of this bullet an
  automated, runnable form. It also only checks mocks: real-checkpoint
  adapters still need `--check-alignment`'s own live-weights judgment call
  (`CLAUDE.md` invariant 7), which this deliberately does not replace —
  see the new module's own docstring for why.
- [x] **Flip `report.verbose` default to `false`** (§4) now that the tool is
  closer to release — verbose stays available, just opt-in. — **done
  2026-08-10**: `config.py::ReportConfig.verbose` flipped `True`→`False`.
  Only one call site anywhere in the repo relied on the implicit default
  (`tests/test_smoke.py::build_config`, whose whole purpose is exercising
  every stage/section — grepped every test file's config-building function
  for "verbose" to confirm) — updated to request `report.verbose: true`
  explicitly rather than relying on a default it needs. No production
  config (`default.yaml`, `medium_run*.yaml`, etc.) sets `report.verbose`
  explicitly, so every real run's report becomes non-verbose by default;
  `--verbose` on `run.py` remains the one-flag opt-in. Verified live: smoke
  test passed directly (not just asserted) after the change.
- [x] **Packaging**: decide on PyPI/versioning for `tsfm_lens` (currently
  only `tsfm_benchmark` has a root `pyproject.toml`); `tsfm_lens` has its own
  `pyproject.toml` one level down — reconcile these into a coherent
  install story (one repo, plausibly two installable packages, document
  which). **Decided 2026-08-11: keep two independently installable
  packages, each with its own `pyproject.toml` — not a single merged
  package.** Rationale, from actually reading both files side by side
  rather than assuming a merge is obviously better: their dependency sets
  barely overlap and differ by an order of magnitude in weight —
  `tsfm_benchmark`'s core deps are `numpy`/`scipy`/`pyyaml` with
  `real-data` as an opt-in extra (`datasets`/`tsbootstrap`/`sdv`/
  `dtaidistance`), while `tsfm_lens`'s core deps already require
  `torch`/`zarr`/`plotly`/`scikit-learn`/`jinja2` plus optional
  `chronos-forecasting`/`timesfm` extras. Someone who only wants to
  *generate and validate* benchmark corpora has no reason to install
  `torch`; someone who only wants to *run the analysis pipeline* against
  an already-built corpus has no reason to install `sdv`. A single merged
  `pyproject.toml` would force every install to pull the union of both,
  with no PyPI-extras mechanism clean enough to avoid it given the two
  packages don't share a namespace prefix (`tsfm_benchmark.*` vs.
  `tsfm_lens.*`, confirmed via each file's own `[tool.setuptools]` package
  list). This is not a new decision so much as a formalization of the
  status quo — both files already work today as independent, correctly
  scoped installs; what was actually missing was a written-down reason,
  not a structural fix. No code changed. Cross-referenced into
  `CLAUDE.md` §3's own note on this (which previously called it "not yet
  folded into one coherent install story"). If PyPI publication is ever
  pursued for either, `tsfm-benchmark` and `tsfm-lens` are the natural
  two distinct package names (already set as each `pyproject.toml`'s
  `[project.name]`) — versioning can move independently per package,
  which is the normal multi-package-monorepo convention and requires no
  extra tooling this repo doesn't already have.
- [x] Revisit and either implement or explicitly defer-with-reason the
  remaining `CLAUDE.md` §13 future-work items not already folded into this
  roadmap (Chronos decoder capture, CI diversity gates, more source
  adapters, real-corpus activation bucketing, deeper component resolution).
  **Resolved 2026-08-11 as a doc-only audit — no code changes, no test
  rerun needed.** Cross-checked all five named items against the existing
  backlog: (1) **Chronos decoder capture** — already tracked verbatim as
  §16 **E21**, no new content needed. (2) **CI diversity gates** — was
  genuinely untracked (confirmed via grep for "diversity gate"/"CI gate"/
  "redundancy fraction <"; distinct from E8/A14, which cover pipeline
  *code* correctness, not generated-*corpus* quality) — added as new §16
  **E23** above, scoped and explicitly deferred pending a first real
  threshold-calibration pass (the only reference point on record is one
  demo-mode run's numbers, §5's redundancy fraction 4.8%/eff-dim 4.6-of-24
  — one sample, not a validated threshold). (3) **More source adapters**
  (TIME/BOOM/ARFBench) — not untracked, just not cross-referenced: the
  reason (access/license unverifiable) is already stated in full at
  `CLAUDE.md` §4.3 and §12; no roadmap action needed, this note is the
  cross-reference. (4) **Real-corpus activation bucketing** — was
  genuinely untracked (confirmed via grep for "GIFT-Eval"/"LOTSA"/
  "bucket" — the only "bucket" hits were `benchmark_validation`'s
  unrelated matcher-bucketing optimization) — added as new §16 **E24**
  above, scoped and explicitly deferred behind a real large-corpus source
  and a GPU session, sequenced after the synthetic-corpus L4 work per
  `CLAUDE.md` §13 item 7's own stated rationale. (5) **Deeper component
  resolution** — already tracked verbatim as §16 **E18**, no new content
  needed. Net effect: two new backlog entries (E23, E24), no design
  decisions made about *when* to build either — both are explicitly
  deferred with a stated blocker, not silently dropped.

**Findings / decisions**
- **Adapter-conformance checker (2026-08-06).** Built
  `tsfm_lens/models/conformance.py::check_adapter_conformance` + new
  `tests/test_adapter_conformance.py` — see this section's checklist entry
  above for exactly what it checks and doesn't. The interesting design
  choice was what to raise on vs. what to accept as valid: a genuinely
  broken adapter (non-monotonic `token_time_spans`, an empty layer list, a
  malformed `attention_info` entry) raises `AssertionError` immediately
  (`CLAUDE.md` §2.5's "broken assumption fails loudly"), while an adapter
  that simply doesn't support an optional capability and correctly returns
  `None` passes cleanly — tested explicitly with a
  no-optional-capabilities stand-in subclass, not just inferred from the
  mocks (which all happen to support everything, so that path was
  previously untested by the smoke suite). Ran against `mock_wave`
  specifically because it's the one existing adapter that mirrors what a
  genuinely new Phase-4 model addition would look like (registered, never
  exercised by any default config) — passing conformance against it with
  zero changes to `models/mock.py` is a small but real confirmation of
  this section's and §9's own "no changes needed outside the new adapter
  file" success bar. Full suite 74/74 after adding it.
- **Auto-generated capability matrix (2026-08-06).** Built
  `tsfm_lens/models/capability_matrix.py` + `run_capability_matrix.py` +
  `tests/test_capability_matrix.py` — the checklist entry above covers
  what it does. Two layers, kept explicitly separate rather than one
  merged bool per capability: `declared_capabilities` is a static check
  (does the adapter *class* override `ModelAdapter`'s default for
  `attention_info`/`mlp_info`/`attention_patterns`/
  `cross_attention_patterns`?) needing no loaded model at all, and
  `verified_capabilities_from_run` reads an already-completed run's
  `attention/meta.json` + `arrays.npz` to report what actually produced
  usable output for one named model in that run — no checkpoint loaded by
  this module itself, either. Ran the CLI with no `--verify` at all and it
  reproduced `CLAUDE.md` §6.2's hand-maintained capability table exactly
  from the registry alone (all six adapters, including `chronos_bolt`'s
  declared-false on both `attention_patterns`/`cross_attention_patterns`,
  matching its documented "deliberately unsupported" status). Then ran it
  a second time with `--verify timesfm=runs/medium_run_chronos_base:TimesFM`
  and `--verify chronos=runs/medium_run_chronos_base:Chronos-T5-Base`
  (reading that already-completed real-checkpoint run's artifacts, no new
  model calls) and it reproduced the one genuinely subtle case §6.2
  documents by hand: TimesFM's `mlp_info` *declares* an override (it calls
  the shared `_scan_mlp` helper) but *verified* nothing in that real run —
  rendered as a distinct "⚠️ declared, found nothing" cell, not conflated
  with "never checked". Chronos verified fully across all four
  capabilities in the same run. Found and fixed one real bug while writing
  this: the initial renderer used `verified is True`/`is False` identity
  checks against a pandas column value, which works when a column mixes
  booleans with `pd.NA` (pandas keeps it `object`-dtype, preserving Python
  bool identity) but silently mis-renders everything as "unverified" once
  a column has no `NA` at all (pandas then stores plain `numpy.bool_`,
  which is `==True` but not `is True`) — caught by a real-run cross-check
  showing an unexpectedly-unverified cell where a verified one was
  expected, then isolated with a dedicated single-row, no-NA regression
  test (`test_render_capability_matrix_markdown_renders_verified_true_with_no_na_rows`)
  before fixing the renderer to use `pd.isna(...)`/`bool(...)` instead of
  `is`. Full suite **81/81** after both new modules. Not yet wired into
  any actual docs page, since none exists yet — see this section's
  checklist entry above.

---

## 11. What's already done (do not re-litigate, per `CLAUDE.md` §7 invariants)

> ⚠️ **Corrections, 2026-08-11 (§0.2 — correct in place, don't delete).** A
> planning pass checked this list against the live repo and found five drifts.
> The bullets below are left verbatim; read them with these corrections:
>
> 1. **"do not re-litigate" now means "this is built", not "this is audited."**
>    §15 (2026-08-06) established that invariant 7 is **not machine-enforced**
>    and invariant 8's "fail loudly" is a bare log line in twelve places.
>    Several bullets below are done in the sense that the code exists and runs,
>    with unfixed silent-failure caveats §15 names. Do not read this section as
>    a trust list — read it as a "don't rebuild this" list.
> 2. **"Full 10-stage-plus pipeline" undercounts, and its enumeration is
>    incomplete** — it is **13 stages** since `layer_screen` was added
>    2026-08-05 and enabled by default (`CLAUDE.md` §6.1). That stage is
>    missing from the arrow-chain below entirely.
> 3. **The per-window-patching bullet's "needs re-confirmation on real
>    weights" caveat is discharged** — `CLAUDE.md`'s 2026-08-03 third-session
>    note confirms the flat TimesFM curve *was* exactly the whole-layer-
>    averaging artifact it was suspected to be, and per-window patching
>    recovers real, depth-intensifying causal structure on live checkpoints.
>    Nothing outstanding here.
> 4. **"The flagship crosscoder … not started" is half stale.**
>    `sae/crosscoder.py` + `run_crosscoder_feasibility.py` landed 2026-08-05
>    with a real-checkpoint stability run (§6.2 Findings), and
>    `sae/matching.py` — post-hoc cross-model feature matching against
>    independently-trained dictionaries, i.e. the *non*-crosscoder route to
>    the same comparison — landed 2026-08-11 (§16 E16). The **flagship
>    deliverable itself is still open**; its full build plan is the new
>    §6.2.1. The encode-store-into-`sae/{model}/{layer}` seam **is** still
>    not started (§6.2.1 Stage 3d).
> 5. **"golden-hash regression" names a currently-unexplained invariant, not
>    a settled one.** That test was reported failing against numpy 2.1.0 on
>    2026-08-03 and passing (non-reproducibly, no code change) on 2026-08-06;
>    root cause is still unknown — `CLAUDE.md` §9/§11.13. Stale in the *good*
>    direction too: the TimesFM bullet's "it hasn't just 'not been done yet'"
>    live-weights re-verification **was** completed 2026-08-03, and found two
>    real bugs doing it (`CLAUDE.md` §11.15–§11.16).
> 6. **Correction (2026-08-18): item 4's "flagship deliverable itself is
>    still open" is now stale — it closed the same day.** §6.2.1 finished:
>    V2 (BatchTopK) was built and scored at 3 seeds, does not flip Stage 1c's
>    clause 2 at any seed, and Stage 4's writeup is in (§6.2.1's Findings).
>    Stage 3 (pipeline wiring) is confirmed gated shut, not provisional. The
>    encode-store-into-`sae/{model}/{layer}` seam (Stage 3d) is genuinely
>    still not started — that half of item 4 stands.

Carried forward from `CLAUDE.md` as of this roadmap's writing, so no phase
above accidentally re-proposes it as new work:

- Full 10-stage-plus pipeline (extract → L0 → internals → lens → L1 → L2 →
  L3 → attention → L4 → exemplars → confirm → report), smoke-tested green.
- Skip lens + tuned lens with crystallization depth.
- Per-window L3 activation patching (not just whole-layer) — the fix for the
  flat-TimesFM-curve issue; **needs re-confirmation on real weights**, see
  §5.4, not re-implementation.
- Attention lag taxonomy, periodicity heads, head/MLP ablation, first-step
  cross-attention.
- Exemplar case studies (the pattern §4/§7 ask to extend to L3 verbose mode).
- Confirm stage with sealed-private-corpus discipline.
- Benchmark generation with two leakage tiers, sealed corpora, epoch
  regeneration, golden-hash regression.
- Benchmark validation's fix for `CO_trev_1_num`-style outlier swamping (was
  flagged outstanding in `CLAUDE.md` as a `QuantileTransformer`/IQR-ranking
  plan; the actual landed mechanism is `RobustScaler` + winsorization with
  per-sequence anomaly diagnostics — see this file's header note and
  `CLAUDE.md` §5).
- TimesFM adapter rewritten against TimesFM 2.5's new class-based API
  (`google/timesfm-2.5-200m-pytorch`), including a newly-supported
  `attention_patterns` capability — see `CLAUDE.md` §11.8. This means the
  live-weights `--check-alignment` re-verification in §5.4 above is now
  doubly relevant: it hasn't just "not been done yet," the model underneath
  it changed since `CLAUDE.md` was last accurate.
- SAE: as of 2026-08-05, **the baseline (§6.2 item 4) is done, not just the
  contract** — a real `TopKSAE`, training loop, eval harness, and
  ground-truth alignment score all exist and have run against real
  checkpoint activations (see §6.2's Findings for what they found). **The
  flagship crosscoder (item 1) closed 2026-08-18** — see §6.2.1's Stage 4
  Findings — as a negative result (no variant beats independently-trained,
  post-hoc-matched dictionaries on ground-truth alignment). The
  encode-store-into-`sae/{model}/{layer}` seam is still exactly where it
  was — not started.

---

## 12. Non-goals (for now)

Explicitly out of scope until something above changes that judgment — listed
so a future session doesn't accidentally drift into them:

- LLM-based in-context data generation (`CLAUDE.md` §4.6) — stays dropped,
  unverifiable/non-reproducible reasoning still holds.
- Forcing symmetric Chronos-decoder capture just to match TimesFM's coverage
  (`CLAUDE.md` §13 item 2) — if pursued, it's a *separate* measurement with
  its own alignment windows, not an attempt to erase the real coverage
  asymmetry documented in `CLAUDE.md` §12.
  > **Reclassified 2026-08-11: this is a *deferred item with a design*, not a
  > clean non-goal.** §16 E21 specifies the decoder-capture surface in full
  > (teacher-forced decoder hooks, forecast-time alignment windows, and the
  > explicit rule that decoder states are **never** CKA'd against context-
  > window states). The clause above is still exactly right about *how* it
  > must be done if done — it is the design constraint E21 inherits, not a
  > reason not to schedule it. What keeps it out of scope today is cost and
  > priority ordering, not a judgment that the measurement is invalid. Track
  > it at E21; the bullet stays here so a future session doesn't read E21 as
  > licensing symmetric-coverage claims.
- Claiming cross-model causal patching (`CLAUDE.md` invariant 5) — the
  crosscoder work in §6.2 changes *what's compared* (shared features) but
  does not license transplanting activations between models; that stays
  within-model only.

---

## 13. Open questions / risks (append as they arise; mark resolved in place)

- [x] **`_per_block_flops` resolves 0 of 12 Chronos blocks by name
  (2026-08-12, §18 F4) — RESOLVED 2026-08-13, hypothesis confirmed, fix
  sketch below replaced with a sounder one.** The hypothesis was right and is
  now *verified* rather than reasoned: a synthetic two-stack probe (a `Root`
  with two same-class `Stack`s) shows `FlopCounterMode` keying its hierarchy
  from **the outermost module actually entered**, so entering `root.encoder(x)`
  yields `Stack.block.0` while entering `root(x)` yields `Root.encoder.block.0`
  — no checkpoint needed, because this is a question about torch's behavior,
  not about Chronos. **The fix sketch below (unique-suffix match) was NOT
  used, and should not be revived: it is unsound on exactly the model it
  targets.** T5's `encoder` and `decoder` are both `T5Stack`, so if both were
  ever entered as counter roots their subtrees share one key and their counts
  are *summed*; a uniqueness check on suffixes would not detect that, and
  would attribute decoder FLOPs to an encoder depth axis — the error this
  item's own last sentence says must be avoided. Implemented instead:
  `_measure_flops` registers forward pre-hooks recording which module paths
  ran, and a new `_keys_from_entry` reconstructs `ModuleTracker`'s own naming
  rule from that record (outermost entered ancestor's class name, then the
  dotted path relative to it), withholding per-block FLOPs with a WARNING
  naming the collision when two roots share a class. The pre-existing
  exact-match path is tried first and is untouched, so TimesFM's recorded
  numbers cannot drift. Live-verified on `runs/medium_run_chronos_base`
  (`--stages budget --force budget`): all 12 Chronos blocks resolve, all
  exactly **64562946048.0** FLOPs, `headline_is_upper_bound` **true → false**.
  Full numbers, including the one result worth knowing before quoting it, in
  §18 F4's "Closed" block. Three new tests in `tests/test_model_budget.py`
  cover the encoder-only shape, the same-class collision withhold, and that
  root entry still uses the exact-match path. *Original entry preserved
  below.* `analysis/model_budget.py::_per_block_flops` builds
  its lookup keys as `f"{type(adapter.module).__name__}.{block_name}"`, which
  assumes the counted call entered the adapter's `.module`. `ChronosAdapter.
  forward()` calls `self._t5.encoder(...)` directly, so `FlopCounterMode`
  most likely keys its blocks relative to *that* submodule (`T5Stack.block.0`
  rather than `<root>.encoder.block.0`) — **hypothesis, formed by reading the
  adapter, not verified.** A diagnostic script that dumps the counter's actual
  keys is written but was not run: `scratchpad/probe_keys.py` (regenerate it
  from this description; it is in a session-scoped temp dir). **Consequences,
  which are live right now:** Chronos has no per-block FLOPs, so (a) its F4
  headline is an *upper bound* rather than a measurement, (b) it has no trace
  in F2's "Compute completed by depth" chart, which currently shows TimesFM
  only, and (c) F1's proposed compute-fraction depth axis cannot be built for
  it, which matters because F1 is the item where that axis is the deliverable.
  **Fix sketch:** fall back to a suffix match, but require the candidate to be
  *unique* — a bare suffix match is ambiguous between `encoder.block.0` and
  `decoder.block.0`, and silently picking one would put decoder FLOPs on an
  encoder depth axis, which is the exact class of error F4 exists to prevent.
  Keep returning `None` when the match is not unique.
- [ ] **The F4 qualifier fires on SAE feature findings, and it is not
  obvious that it should (2026-08-12).** `_qualify_depth_claims` matches on a
  model name plus depth vocabulary, and "SAE — Chronos-T5-Base/encoder.block.6:
  feature 2189 best matches has_intermittency" contains both. Appending "at
  least ~86% of this model's forward computation is unobserved" to it is
  *true*, and the block name does locate the claim in the stack — but the
  finding is about what a feature encodes, not about where in depth something
  happens, so the qualifier is arguably noise there. Left firing deliberately:
  over-qualifying adds a true sentence, under-qualifying drops a needed one,
  and §2.5's doctrine prefers the first error. Revisit if the findings list
  gets noisy enough that readers start skimming past qualifiers — that failure
  mode would defeat the whole mechanism.

- [~] Does effective dimensionality actually predict anything useful, or is
  it a plausible-sounding metric that doesn't survive contact with data
  (§6.1)? **Partially answered (2026-08-05):** it *does* predict two
  interpretability proxies across models — but in *opposite* directions, and
  (§6.2 Findings) the relationship does **not** transfer within a single
  model, so eff-dim alone is not a usable per-model selector. A proposed
  alternative (L3 fingerprint entropy) beat it on the one comparable proxy.
  Still open at the *selector* level: which of §6.1.1's redesigned candidates
  (geometry / factor-emergence / coverage) actually wins the §6.1.1-E
  bake-off against the uniform-stride null.
- [~] Which §6.1.1 layer-selector wins, and does *any* of them beat uniform
  stride-k at equal budget (§6.1.1-E)? **Provisionally answered (2026-08-05,
  one corpus, one checkpoint pair — see §6.1.1's Findings for the full
  numbers):** Idea A (`work_bend`) essentially matched the oracle and beat
  both nulls on Chronos-T5-Small, where real headroom over uniform-stride
  existed; on TimesFM no selector beat uniform-stride, but neither did the
  oracle itself — gold interestingness there is close to evenly spread
  across depth (concentrated only at the very first/last layer), so there
  was little room for *any* selector to add value, which is a different and
  more forgiving finding than "the method failed." Idea B
  (`factor_emergence`) underperformed both nulls on Chronos, traced to a
  real, diagnosed weighting flaw — **the peak-weighting + absolute-floor
  fix was implemented and unit-tested against synthetic planted data
  (2026-08-06, `CLAUDE.md` §11.18), and has now been re-run through the
  real bake-off (2026-08-10, §6.1.1's Findings) — answered, and the answer
  is no.** `factor_emergence` still doesn't clear the nulls on
  Chronos-T5-Small (bit-identical failing selection, recall@budget=0.00,
  pre- and post-fix) and actively regressed on TimesFM (recall@budget 0.20
  → 0.00). Root cause: on real data the early-layer bias comes from
  *strong*, near-input-statistics factors decoding well from layer 0, the
  opposite of the synthetic test's weak-noisy-factor scenario the fix
  targeted — so peak-weighting amplifies rather than corrects it, and the
  new absolute floor never once fired (no factor's peak fell below it on
  this corpus). `work_bend`/`coverage` confirmed bit-identical pre- vs.
  post-fix, so the fix stayed correctly scoped with no side effects — that
  part of the original diagnosis holds. `factor_emergence` is not usable
  as a selector on the evidence gathered so far, full stop; a real fix
  would need an explicit depth floor, not a magnitude reweighting, and
  hasn't been attempted. Combining methods
  (union/vote/rank-average) did not beat the single best method on either
  model. Still open: whether this holds on a second Chronos size, ~~a third
  architecture family~~, and with the (more expensive) per-window-patching
  secondary gold instead of the cheaper sensitivity-only proxy used here.
  **The third-architecture-family question is now answered (2026-08-10,
  §6.1.1's newest Findings block, using Sundial): mixed, not a clean
  sweep.** `work_bend` beats both nulls on Sundial, but the picture across
  all three architectures now on record is "beats a different subset of
  the two nulls on each" — consistent with, not a new contradiction of,
  the original two-model finding's own already-mixed TimesFM result.
  `factor_emergence`'s diagnosed early-layer-bias failure mode (this same
  day, above) reproduces cleanly on Sundial too, and if anything more
  pronounced — strong evidence the failure is a property of the
  ground-truth factor table itself, not either specific architecture.
  **Promoted to the production default anyway, on explicit user direction
  (2026-08-05, same-day follow-up):** `layer_screen: {method: work_bend}`
  now runs by default ahead of `sae` in every config (§6.1.1's
  production-wiring Findings) — the second-Chronos-size and per-window-
  patching items above are unchanged and still worth closing, but no
  longer block using the current best-known method instead of the old
  arbitrary final-layer default.
- [x] 🔴 **New (2026-08-10, found while closing the item above): does the
  bake-off's qualitative verdict (beats a given null, yes/no) reproduce
  across independent re-extractions of the same config, or only across
  SAE-training reseeds of one frozen extraction?** — ✅ **Resolved by the
  entry immediately below**, which root-causes this to §15 A4's
  stratified-sampling fix landing between the two runs (not to any form of
  nondeterminism) and closes the residual seed-sensitivity with
  `n_gold_replicates=3`. *Checkbox added 2026-08-11 — this entry was written
  without one, so a session skimming §13 for the 🔴 marker read it as live
  when its own resolution was the next bullet down.* The existing "seed1
  stability replicate" only ever re-seeded SAE training on one already-
  extracted store — it was never a test of extraction-to-extraction
  variance. A same-day, same-config, fresh re-extraction (built to add
  Sundial as a third model, §6.1.1's newest Findings) changed
  Chronos-T5-Small's verdict outright: `work_bend` recall 1.0→0.0,
  `beats_random` True→False, gold-agreement ρ 0.714→0.429, while TimesFM's
  own qualitative verdict reproduced unchanged in the same two runs — so
  this is real, but not uniform across models. **Not yet known:** how many
  independent extractions it would take to characterize this variance
  properly (an n=2 comparison establishes that it exists, not its
  distribution), whether it's driven by corpus resampling, model-loading
  nondeterminism, SAE-training stochasticity compounding with a different
  activation draw, or something else, and — most importantly — whether
  today's `factor_emergence`-fix verdict (§6.1.1's Findings, "the fix
  doesn't clear the nulls and regresses TimesFM") would survive a second
  independent extraction the same way TimesFM's verdict just did, or flip
  the way Chronos's just did. Until this is characterized, read every
  single-run bake-off qualitative verdict in this file (there is no
  multi-extraction one yet, for any method or model) as measured on one
  sample from a distribution whose spread is now known to be non-trivial
  for at least one model, not as a fixed ground truth.
- [x] **Resolved (2026-08-10, same-day follow-up) — root-caused. It is not
  extraction/GPU/SAE-seed variance at all: it is `§15 A4`'s already-landed
  sampling fix, and the "same config" premise above was false at the code
  level even though the YAML was byte-identical.** A background agent
  ranked four candidate causes (GPU/SDPA kernel nondeterminism, SAE-init/
  seed sensitivity, global-RNG-stream reordering from adding Sundial,
  corpus-row-sampling nondeterminism) and tested them directly against real
  extractions on 8×RTX A5000 (`cudaPy`), then found something none of the
  four predicted, which a follow-up `git diff 84cdbc0 HEAD --
  tsfm_lens/data.py` independently confirmed line-for-line: **`data.py`'s
  `_assemble` used to cap `max_series` with a bare `kept[:max_series]` head
  slice; `§15 A4`'s fix (landed 2026-08-06, one day after this bake-off's
  original 08-05 run and four days before its 08-10 rerun) replaced that
  with `sample_rows(len(kept), max_series, seed, strata=families_all)`, a
  stratified random sample.** `layer_screen_experiment.yaml`/`_v2.yaml` set
  `data.max_series: 220` against a 288-row corpus
  (`benchmark_medium/public_dev`) — verified directly:
  `Counter({'random_parametric': 188, 'mixture': 60, 'parametric': 40})` for
  the full 288 rows vs. `Counter({'random_parametric': 188, 'parametric':
  32})` for the first 220 (the pre-fix head slice) — **the old head slice
  excluded the entire `mixture` family (all 60 rows) and 8 of 40
  `parametric` rows**, because `tsfm_benchmark` writes corpora grouped by
  task/generator and `mixture` happens to sort last. So the 08-05 bake-off
  (`layer_screen_bakeoff.json`) trained its gold ranking and scored every
  selector on a corpus subset with an entire real-derived family silently
  missing; the 08-10 reruns (`_v2`/`_v3`/`retest2model`, all bit-identical
  to each other) correctly used a family-proportional sample including
  `mixture`. This is fully deterministic given a fixed seed (explaining why
  `v2`/`v3`/`retest2model` are bit-for-bit identical to each other — the
  agent confirmed this directly via `np.array_equal` on the raw stored
  activations) and fully explains the "instability" without needing any
  GPU-nondeterminism, hook-leak, or model-loading mechanism — all of which
  were checked directly in the model/extraction code
  (`extraction/hooks.py`'s `ActivationCatcher.__exit__` correctly removes
  hooks even on the alignment-gate's failure path; `_primary`'s §11.21
  rewrite is provably output-preserving for tensor/tuple returns;
  `extraction/store.py`'s new `finalize_layer` only records metadata, never
  rewrites stored values) and ruled out as the mechanism. **This retracts
  the "not yet known... whether it's driven by corpus resampling..." framing
  two paragraphs up** — it is exactly that, identified precisely, not one
  candidate among several. **What this means for every number already on
  record:** the 08-05 `layer_screen_bakeoff.json` run (§6.1.1's first
  Findings block) and everything that cited it were measured on a
  `mixture`-family-blind sample — a concrete, real instance of the abstract
  risk the open item at line ~4085 below ("how much of what's already
  recorded in this file survives §15's sampling... fixes?") already named,
  now confirmed to apply here specifically. The 08-10 3-model numbers
  (§6.1.1's newest Findings) are the trustworthy ones going forward for
  this experiment; the 08-05 pairwise numbers should be read as superseded
  by them, not as an independent replication — they were never actually
  comparing the same population. **Tested (2026-08-10, same day, third
  follow-up — see §6.1.1's newest Findings for the full numbers): no, not
  reliably.** A second seed under the identical, correctly-stratified
  sampler (76.8% row overlap with seed 0, not a wildly different draw)
  still flips `work_bend`'s `beats_random` verdict on 2 of 3 architectures
  (TimesFM True→False, Chronos False→True), though `coverage` is fully
  stable and the selector's own *choice of layers* is bit-identical across
  seeds in every flip case — only the null-comparison scorecard is noisy,
  not the underlying selection. This is real statistical power, not
  systematic bias like A4 was: `recall_at_budget` at budget=2-5 over a
  220-row corpus doesn't yet have the resolution to answer "beats null,
  yes/no" with confidence in a single run. **Lesson for `CLAUDE.md` §2.8's
  own workflow,
  worth carrying forward:** a background-compute config/artifact that
  predates a shared-infrastructure bugfix is not safe to treat as "the same
  experiment, rerun" just because its YAML never changed — the fix can
  silently change what a `max_series`-style cap or any other config field
  *means*, and the only way to have caught this without the accidental
  Sundial-driven re-extraction would have been to notice the run's own
  `git log` position relative to `§15`'s fix dates. See `CLAUDE.md` §11.24
  for the write-up of this as a general trap.
  **Fixed and re-verified (2026-08-10, same day, fourth follow-up) — no
  longer an open statistical-power gap.** Per explicit user instruction not
  to leave this as a documented-but-unfixed concern: root-caused the flip
  to `build_gold_ranking`'s single stochastic per-layer SAE-training run
  (confirmed `select_work_bend`/`select_coverage` take no `seed` argument
  and are otherwise fully deterministic, so the noise had to be entering
  through the gold reference, not the selector). Fixed by averaging
  `n_replicates=3` independently-seeded SAE-training runs into the gold
  score (`analysis/layer_screen_bakeoff.py::build_gold_ranking`) and by
  dropping the corpus-subsampling cap entirely in new `configs/
  layer_screen_experiment_v4[.yaml/_seed1.yaml]` (288-row corpus is small
  enough to use in full, removing that axis of seed-dependence too).
  Re-verified live against all three real checkpoints on both seeds:
  **every `beats_random` verdict now matches across seeds for all three
  selectors on all three architectures** — `work_bend`'s TimesFM and
  Chronos-T5-Small flips are both closed. Full comparison table in
  §6.1.1's Findings block and the 2026-08-10 §14 session-log entry
  describing this fix. New unit test added; full suite green at 208/208.
- [x] Is a joint crosscoder actually trainable/stable across two
  architecturally distinct models at a shared alignment window, or does the
  representational mismatch (even at peak CKA) make joint training degrade
  into one model dominating the dictionary? **The prerequisite small-scale
  test is now done (2026-08-05, `tsfm_lens/sae/crosscoder.py` +
  `run_crosscoder_feasibility.py`, §6.2's Findings below):** yes, trainable
  and stable, with one real caveat found and fixed along the way (naive
  equal-weighted MSE across sources of different raw activation scale
  *does* let the larger-scale source dominate — fixed by per-source
  scale-normalization inside the model, not by the caller). Real-checkpoint
  run (TimesFM stacked_xf.4 ↔ Chronos-T5-Base encoder.block.10, the L1
  peak-CKA pair) shows comparable per-source fidelity to independently
  trained baselines, not a collapse.
  **Fully answered 2026-08-18 — §6.2.1, CLOSED.** The flagship deliverable
  (full shared/specific decomposition, plus a second variant, BatchTopK)
  was built, ran through the pre-registered Stage 1c decision rule at 3
  seeds, and the answer to "is it worth building" is **no, on the evidence
  gathered**: neither variant beats independently-trained, post-hoc-matched
  per-model dictionaries on ground-truth alignment (V1 diff −0.0417,
  p=0.002; V2 loses at all 3 seeds too), a decisive negative result rather
  than an inconclusive one. The dictionary being "mostly dead" was fixed
  along the way (AuxK, Stage 0) and is no longer the open issue — the open
  issue was whether joint training earns its cost over the cheaper
  alternative, and it does not, at this layer pair and corpus. See
  §6.2.1's Stage 4 Findings for the three-part writeup (the shared-fraction
  number, its seed-fragile floor, and the ground-truth content) and the
  methodological result (a shared-fraction metric needs a same-seed,
  architecture-matched floor to be readable at all — not stable even
  within one metric/pair across seeds).
- [x] Does the distillation-detection signal (§6.3) actually separate from
  "same training era, similar data" confounds, or is that confound
  unavoidable with publicly available checkpoints? May require training
  the controlled positive/negative pair from scratch to know for sure. —
  **answered 2026-08-10, and worse than the confound this question named:**
  no from-scratch training was needed — a same-architecture,
  zero-*training* control (`random_init: true` on both sides, §6.2's
  existing mechanism) showed the signal doesn't even separate from "same
  architecture, literally no training or data at all." That control's L1
  CKA (0.878) and L2 gain (0.834–0.940) both significantly *exceeded* the
  real trained positive pair's (0.734 / 0.637, p=0.0005 both). See §6.3's
  Findings for the full numbers — the method is falsified for its stated
  provenance/IP use case, not just confounded.
- [x] Licensing/access for Moirai, Sundial, and other Phase 4 candidates
  should be re-checked at the time Phase 4 actually starts, not assumed
  stable from whenever this section was written. — **checked 2026-08-10**,
  live against each model's actual HuggingFace card (not assumed from this
  file's own prior description, which turns out to be stale on
  architecture for one of the two): **Moirai** (both `Salesforce/
  moirai-1.1-R-small` and the newer `Salesforce/moirai-2.0-R-small`,
  released August 2025) is **`cc-by-nc-4.0` — non-commercial only**. That's
  fine for this repo's own research/interpretability use but is a real,
  previously-unstated constraint worth carrying forward if this project's
  scope ever shifts toward anything commercial — flagged here rather than
  silently assumed permissive, matching the diligence `CLAUDE.md` §4.3
  already applies to benchmark data sources. Also: Moirai 2.0 is a genuine
  architecture change from 1.x, not just a size bump — **decoder-only now**
  (patch embeddings + missing-value encoding into a decoder-only
  transformer), not the masked-encoder design §9's candidate list
  describes; 1.1-small is 13.8M params, 2.0-small is 11.4M. §9's "good test
  of whether alignment/pooling holds up against a model whose tokenization
  isn't a simple fixed patch width" rationale should be re-verified against
  whichever Moirai version is actually adapted — it may describe 1.x better
  than 2.0. **Sundial** (`thuml/sundial-base-128m`, ICML 2025 Oral,
  pretrained on ~1 trillion time points) is **Apache-2.0** — fully
  permissive, no constraint. Confirmed **decoder-only** (§9's "worth
  checking" is now answered, not open) with patch length 16 and a
  flow-matching (`TimeFlow Loss`) probabilistic head instead of TimesFM's
  deterministic quantile head or Chronos's discrete-token sampling — a
  third distinct way of producing multi-sample forecasts, worth naming
  explicitly when Sundial's adapter is built since `predict()`'s contract
  will need to map flow-matching sampling onto the existing
  `{"point": ..., quantiles...}` shape the way Chronos's `num_samples`
  already does. Net effect on priority: Sundial is now the **more
  attractive next pick** of the two — permissively licensed, and
  structurally further from both existing flagship architectures (a third
  decoder-only design with a genuinely different probabilistic mechanism)
  than Moirai 2.0 turned out to be. Not done this session: no adapter code
  written for either, no `predict()` design worked out — this is the
  licensing/architecture recheck only, per this item's own stated scope.
- [x] **How much of what's already recorded in this file survives §15's
  sampling and noise-floor fixes?** (Added 2026-08-06.) A4 shows several
  recorded numbers were measured on task-ordered prefixes rather than
  representative samples, and A13 shows no ΔMASE in the repo has ever been
  compared against a measured repeat-run floor. The specific numbers to
  re-measure and then supersede in place (§0.2 — correct, don't delete) are
  named in the ⚠️ AUDIT markers: §6.2's SAE forecast-preservation ΔMASE and
  ground-truth ρ values, and §5.3's lens comparison. **This is not a
  suspicion that they're wrong — it's that their error bars are currently
  unknown**, and two of them are load-bearing for a decision (whether Phase 3
  feature ablation can build on the TimesFM SAE target).
  **A concrete instance found 2026-08-10 (not one of the two named above,
  a third):** `§6.1.1`'s original `layer_screen_bakeoff.json` (08-05, one
  day before A4 landed) turned out to be exactly this failure mode —
  measured on a `max_series: 220` head slice of a 288-row corpus that
  silently excluded the entire `mixture` family — discovered by accident
  when a same-day 08-10 rerun (post-A4-fix) gave different numbers; see the
  resolution two entries below and `CLAUDE.md` §11.24. Worth a deliberate
  sweep rather than waiting for more accidents: any run directory whose
  `data.max_series` is set below its corpus's row count, and whose
  extraction predates 2026-08-06, likely has the same problem.
  **Swept (2026-08-10, same day, second follow-up):** grepped every
  `data:` block (not the many *other*, unrelated `max_series`-named knobs —
  `rsa_max_series`, `tuned_max_series`, `forecast_preservation_max_series`,
  `ground_truth_max_series`, `ablation_max_series`, l3/attention's own
  per-stage `max_series` — those are separate analysis-level subsample caps,
  already tracked by A16/the two numbers named above, not `data.py`'s
  corpus-level cap this specific bug lives in) across every config in
  `tsfm_model_analysis/tsfm_lens/configs/*.yaml`. Result: **the top-level
  `data.max_series` cap this bug affects is set** ***only*** **in the
  `layer_screen_experiment*.yaml` family** (`.yaml`, `_v2.yaml`, `_v3.yaml`,
  `_retest2model.yaml`, all `max_series: 220` against the same 288-row
  corpus). Every other real-checkpoint config that reads
  `benchmark_medium/public_dev` or `benchmark_full/public_dev`
  (`medium_run.yaml`, `medium_run_chronos_base.yaml`, `default.yaml`,
  `null_chronos_random.yaml`, `null_timesfm_random.yaml`,
  `distill_positive_chronos_small_base.yaml`, `sundial_phase4_check.yaml`,
  `chronos2_phase4_check.yaml`) sets **no** `data.max_series` at all — the
  `if cfg.max_series is not None` guard in `_assemble` never fires for any
  of them, pre- or post-fix, so they always used every row their corpus had
  regardless of which side of 2026-08-06 they ran on. **Conclusion: this
  specific bug (`data.py`'s corpus-level cap) has exactly one other victim
  beyond the two already named above, and it's the layer_screen bake-off
  just resolved — not a wider, still-undiscovered problem.** The two
  already-named §6.2/§5.3 numbers go through different call sites
  (`sae/eval.py`, `sae/ground_truth.py`, `lens.py`) that A4's fix also
  covered, and their re-measurement remains exactly as open as this item's
  original text says — this sweep doesn't change that, it just confirms
  `data.py`'s own cap isn't hiding a fourth instance anywhere else in the
  repo's existing configs.
  **Both named re-measurements are now actually done, closing this item's
  own scope (2026-08-10, later the same day).** §5.3's lens comparison: the
  `medium_run.yaml` side turned out to be blocked by an unrelated stale
  zarr-v3 store (`CLAUDE.md` §11.25), fixed via a full re-extraction; both
  sides now reproduce bit-identical TimesFM lens numbers (`final_mase:
  1.7899408340454102`) — see this file's own session log and `CLAUDE.md`
  §11.25. §6.2's SAE forecast-preservation/ground-truth numbers: it turned
  out these had *already* been re-measured once, same-day, 2026-08-06 (the
  "RE-MEASURED" entry in §6.2's own Findings block above) — this item's
  text above, written earlier that same day, went stale within hours and
  was never corrected until now (fixed in place at this file's A4 Findings
  block too). A further, independent third measurement (2026-08-10) shows
  Chronos-T5-Base's ΔMASE and both models' ground-truth ρ are stable across
  reruns, but surfaces a genuinely new, still-open sub-question: TimesFM's
  own forecast-preservation ΔMASE moved 0.175 → 0.1097 between the 08-06
  and 08-10 reruns of the *identical* config — real run-to-run SAE-training
  variance in that specific metric that nothing currently measures (see
  §6.2's Findings for the full numbers). **Spun out as its own narrower
  follow-up rather than left as a residual "still open" tag on this whole
  item**: a proper repeat-run noise floor for the SAE forecast-preservation
  check itself (several `sae`-only reruns with the extraction/model held
  fixed), analogous to A13's model-level MASE floor but scoped to this one
  metric. Until that exists, treat any single TimesFM SAE
  forecast-preservation ΔMASE as "small and real, exact magnitude
  uncertain," not a precise number.
- [x] **Is the `work_bend` production default still the bake-off winner under
  production conditions?** (Added 2026-08-06, §15 A1.) The bake-off ran at
  `capture_layer_stride: 1`; the wired stage screens whatever the analysis
  config captured, which is stride 2 for TimesFM everywhere. So the selector
  currently runs on a different input than the one it was validated on, and
  the §6.1.1 R2 requirement it was designed around is unimplemented. Closing
  A1 answers this; until then, treat production `sae.targets: auto`
  selections as unvalidated even though the *method* was validated. —
  **Resolved (2026-08-10): yes, per A1's already-shipped fix, re-confirmed
  by rereading rather than re-running.** `LayerScreenConfig`'s defaults
  (`tsfm_lens/config.py`) are `stride: 1` and `require_full_capture: True`
  — every production config already runs the dedicated stride-1 screening
  extraction A1 built, independent of whatever `capture_layer_stride` the
  main analysis config uses, unless a caller explicitly opts out. A1's own
  Findings already recorded live proof of exactly this scenario (TimesFM
  20 blocks, main store captured 10 at stride 2, the screening pass
  correctly screened all 20 and reported `fair_to_all_layers: true`) — this
  entry was just never marked resolved even though the fix it was waiting
  on had already landed and been verified. No new run needed to close
  this; the open, real caveat is still §6.1.1-E's own named next step
  (replicate on a second Chronos size / third architecture family), not
  this production-fairness question.
- [~] **Can this repo's central claims survive an untrained-weights null?**
  (Added 2026-08-06, §16 E9.) **Answered, with a genuine split verdict that
  turned out to depend heavily on comparing matched layers, not just
  matched runs (2026-08-06/07, §16 E9's Findings has every number).** SAE
  ground-truth alignment and internals' family-probe decodability survive
  cleanly — real beats untrained at every depth, for both models. **L1 and
  L2's verdicts are layer-dependent, not uniform**, and reading only each
  run's single global best pair (the first-follow-up test) understated both:
  comparing the real run's own reported peak pair against each null run's
  own *unrelated* best pair (often an early layer with a same-architecture
  self-predictability artifact, not a comparable layer) gave "L1 CKA:
  ambiguous/depends on which model; L2 gain: statistically indistinguishable
  from null (p=0.474, p=0.532)." Re-testing at the **matching layer index**
  in both runs (the depth-curve follow-ups) instead shows: **L1** decisively
  beats both floors at every middle depth (layers 6-14 of 10), losing only
  at the extremes; **L2** decisively beats both floors at the real run's own
  actual best-performing layer and every layer from there on (8 of 12
  Chronos blocks), losing only at the earliest, lowest-absolute-gain blocks.
  Still open (kept `[~]` rather than `[x]` for this reason): this is one
  direction (`Chronos->TimesFM` for L2), one corpus, one Chronos checkpoint
  size — the reverse L2 direction and cross-corpus/size replication are the
  named next step, not yet done.

**Appended 2026-08-11 (planning pass over this whole file).** Four entries
below; the first three were previously *mentioned inside other items' prose*
but never tracked as their own open questions, which is exactly how an
acknowledged follow-up gets lost.

- [x] 🔴 **How large is the repeat-run variance of a single SAE's forecast-
  preservation ΔMASE, and does it invalidate any recorded SAE number?**
  **Answered 2026-08-12: the variance is large enough to swallow the
  motivating observation whole, and it does invalidate one recorded claim
  (not the headline one). Findings below.**
  Spun out of the item above's own text, where it was named and then left
  untracked: TimesFM's SAE forecast-preservation ΔMASE moved **0.175 →
  0.1097** between two runs of an *identical* config. That is a ~40% swing
  on a headline number, and nothing in the repo currently measures the
  repeat-run noise floor for it — so every recorded SAE ΔMASE is being read
  against zero rather than against its own variance, the same class of gap
  §15 A13 fixed for behavioral ΔMASE. **Concretely**: retrain the same SAE
  target N≥5 times with different seeds against a *frozen* activation store,
  record the ΔMASE distribution, and publish it as the floor beside every
  SAE number the way A13's floor is published beside behavioral ones.
  Cheap (no extraction, no model calls beyond `forecast_preservation`'s own).
  **Blocks**: reading §6.2's SAE numbers as differences rather than as noise.

  > **Findings (2026-08-12).** New `tsfm_lens/run_sae_repeat_variance.py`
  > retrains both of `runs/medium_run_chronos_base`'s configured SAE targets
  > at seeds 0–4 against that run's **frozen, read-only** activation store
  > (nothing re-extracted; the corpus rows, checkpoints and store are held
  > fixed by construction) and records fidelity, dead-feature rate, and
  > `forecast_preservation` ΔMASE at **both** granularities per seed.
  > Artifact: `runs/medium_run_chronos_base/sae/repeat_variance.json`.
  >
  > **The frozen-store control passes exactly.** `mase_clean` — the
  > *unpatched* forecast's own MASE, which must not move when only the SAE
  > seed varies — has sd **0.0** across all five seeds for both models and
  > both granularities (TimesFM 1.8799978494644165, Chronos-T5-Base
  > 3.1090171337127686, bit-identical every time). So everything below is
  > SAE-training stochasticity and nothing else.
  >
  > | target | metric | mean | sd | min | max | range |
  > |---|---|---|---|---|---|---|
  > | TimesFM `stacked_xf.18` | fidelity | 0.8574 | 0.0064 | 0.8477 | 0.8649 | 0.0172 |
  > | | dead_rate | 0.9445 | 0.0053 | 0.9353 | 0.9481 | 0.0129 |
  > | | ΔMASE (window) | **+0.1704** | **0.1212** | +0.0650 | +0.3740 | **0.3090** |
  > | | ΔMASE (token) | +0.1704 | 0.1212 | +0.0650 | +0.3740 | 0.3090 |
  > | Chronos-T5-Base `encoder.block.6` | fidelity | 0.8318 | 0.0105 | 0.8173 | 0.8449 | 0.0277 |
  > | | dead_rate | 0.9734 | 0.0027 | 0.9691 | 0.9759 | 0.0068 |
  > | | ΔMASE (window) | **+3.8972** | 0.2420 | +3.6783 | +4.2641 | 0.5858 |
  > | | ΔMASE (token) | **+0.2457** | 0.2113 | **−0.1118** | +0.4136 | 0.5253 |
  >
  > **1. The motivating observation was noise.** TimesFM's ΔMASE moving
  > 0.175 → 0.1097 across two identical-config runs — the ~40% swing this
  > item was opened for — is a 0.065 gap against a measured seed-to-seed sd
  > of **0.1212**. Both recorded values sit comfortably inside the five-seed
  > range [0.0650, 0.3740]. There is nothing to explain: it is half a
  > standard deviation. §6.2's "did not replicate" framing (see the
  > 08-06/08-11 entries) should be read as *was never a replication test in
  > the first place*, not as an unexplained instability.
  >
  > **2. TimesFM's forecast-preservation pass/fail verdict is not resolvable
  > at n=1 seed.** The recorded ΔMASE values (+0.05, +0.1097, +0.110) all
  > come from single training runs whose own noise floor is sd 0.12 with a
  > 0.31 range. Any threshold-crossing claim in that neighbourhood — "passes
  > for TimesFM" — is a claim about one draw, not about the SAE. Quote the
  > mean and sd, or quote the number *with* this floor beside it, exactly as
  > §15 A13 requires for behavioral ΔMASE.
  >
  > **3. E15's directional conclusion is robustly confirmed; one of its
  > specific numbers is retracted.** The window-vs-token gap for
  > Chronos-T5-Base is +3.897 vs +0.246 — a ~16× reduction against a
  > seed-to-seed sd of 0.24/0.21, so the window-broadcast confound is
  > overwhelmingly the dominant driver of every previously-recorded Chronos
  > forecast-preservation failure. That holds. What does **not** hold is
  > E15's headline **−0.346** ("reconstruction is net *better* than the clean
  > forecast"): that value falls **outside** the five-seed range
  > [−0.1118, +0.4136], and the qualitative sign flip replicates in only
  > **1 of 5** seeds, at a much smaller magnitude (−0.112). E15's own
  > write-up already flagged that number as "a little surprising ... worth a
  > skeptical look before reading it as 'the SAE is simply excellent'" — the
  > skeptical look now has a measurement behind it. **Do not quote −0.346.**
  > Chronos's token-granularity ΔMASE is a small positive number, +0.246
  > ± 0.21, not a negative one.
  >
  > **4. TimesFM's window and token deltas are bit-for-bit identical at
  > every one of the five seeds** (not merely equal in the two-run
  > comparison E15 recorded) — as expected, since its token width equals the
  > alignment window, so the pooled broadcast is exact. That is now a
  > five-seed confirmation of E15's mechanism claim rather than a single
  > coincidence.
  >
  > **5. Unrelated but visible in the same artifact, and worth carrying to
  > §6.2.1:** both targets' dead-feature rates are **94.5% and 97.3%**, and
  > they are *stable* across seeds (sd 0.005 / 0.003). Every SAE number
  > recorded from this run therefore comes from a dictionary that is ~95%
  > dead — the exact condition Stage 0's gate exists to eliminate, here
  > confirmed as a reproducible property of the configuration rather than an
  > unlucky draw.
  >
  > **Not done:** the floor is measured for one run directory, one layer per
  > model, at n=5. It is not yet *published beside* every SAE number — the
  > report still renders single-seed ΔMASE with no floor next to it. That
  > rendering change is spun out as its own item below.

- [x] **Render the SAE ΔMASE noise floor in the report, the way §15 A13's
  behavioral floor is rendered.** ✅ **Done 2026-08-13** — see the Findings
  block at the end of this item; full `tsfm_lens` suite green at **378
  passed** afterward. Split out of the item above on
  2026-08-12, which measured the floor (sd 0.121 TimesFM / 0.242
  Chronos-window / 0.211 Chronos-token) but changed nothing about how the
  numbers are displayed. `report.py`'s SAE section still shows a single
  seed's ΔMASE against an implicit zero. **Concretely**: have the `sae`
  stage optionally train `n_seeds` (default 1, so nothing changes by
  default) and, when >1, record mean/sd and render "ΔMASE +0.17 ± 0.12
  (5 seeds)" plus a `_note()` stating that a single-seed delta smaller than
  the sd is not a result. Cheap — the training loop is already seed-
  parameterized and `run_sae_repeat_variance.py` is the working prototype.

  > **Findings (2026-08-13).** Implemented as specified, in five places, with
  > the default left at `n_seeds: 1` so **every already-recorded SAE number
  > stays regenerable bit-for-bit** (§2.1 — a default above 1 would multiply
  > the stage's cost and rewrite its artifact shape for every existing
  > config).
  >
  > **1. `sae/eval.py::seed_spread`** — the shared statistic (mean / sd with
  > `ddof=1` / min / max / range / n / the raw values). Placed here rather
  > than in `analysis/stats.py`, which §2.2 reserves for the *bootstrap
  > designs* that are the scientific substance; a mean/sd summary attached to
  > SAE evaluation belongs beside the other SAE evaluators. `range` is
  > reported next to `sd` because at n=5 the range is the more honest summary
  > of what one unreplicated number could have been. Non-finite entries (a
  > forecast-preservation check that raised) are dropped and `n` falls, so a
  > partial floor is visibly partial rather than NaN.
  >
  > **2. `run_sae_repeat_variance.py` now imports it** instead of carrying its
  > own private `_spread`. The prototype and the stage compute the floor with
  > one definition, so the CLI's already-recorded numbers and the stage's new
  > ones cannot drift apart. (Its now-unused `numpy`/`torch` imports went with
  > the local copy.)
  >
  > **3. `SAEConfig.n_seeds: int = 1`** — documented inline per §2.3's config
  > exception, including that only the primary seed's SAE is checkpointed and
  > put through ground-truth alignment / ablation / steering. The extra seeds
  > exist to size the floor, not to be analyzed.
  >
  > **4. `sae/train.py`** — `_train_config(cfg, seed)` builds the
  > `SAETrainConfig` (so the primary seed and every replicate provably train
  > at identical hyperparameters), `_metric_row(...)` builds one seed's row
  > (so the two paths cannot record different key sets — the first draft had
  > an inline dict literal for the primary seed and a separate builder for
  > replicates, which `seed_spread` then indexed by a hardcoded metric tuple;
  > fixed structurally rather than by test), `SEED_FLOOR_METRICS` is the
  > single list the spread is taken over, and `_repeat_metrics(...)` computes
  > a replicate's numbers. The primary seed's already-computed values are
  > **reused** as `per_seed[0]`, so `n_seeds: 5` costs four extra trainings,
  > not five. Replicates deliberately skip ground-truth alignment, ablation
  > and steering — the floor being sized attaches to the headline numbers
  > only. `mase_clean` is carried per seed as the frozen-store control the CLI
  > already uses: the store, rows and checkpoint are fixed, so it must not
  > move, and a replicate where it does means something other than the SAE
  > seed varied. Result key: `"seed_floor"` in `sae/meta.json`.
  >
  > **5. `report/report.py`** — `_seed_floor(entry, metric)` returns
  > `(suffix, resolvable)` where `resolvable` is **tri-state on purpose**, the
  > same way `_delta_phrase`'s `interpretable` is: `None` when no floor was
  > measured, `False` when `|mean| <= sd`, `True` otherwise. Collapsing `None`
  > to `False` would sprout a warning on every single-seed run that has no
  > evidence for it; collapsing to `True` would let an unresolvable delta read
  > as cleared. The SAE stats line now renders `ΔMASE +0.170 ± 0.121 over 5
  > seeds`, and when a delta is smaller than its own spread the line carries
  > `⚠ … so its sign is not established by this run` — the `_note()` the item
  > asked for, stated in the body rather than in a collapsed note, because a
  > reader who does not open the note is exactly the reader who would
  > otherwise quote the number. A new "Seed-to-seed noise floor" table
  > follows, with `_SAE_SEED_FLOOR_NOTE` covering what varies (nothing
  > upstream), how to read `sd`, that `mase_clean` is a control that must not
  > move, and — the limit worth naming — that this is **not** §15 A13's
  > behavioral repeat-run floor: a ΔMASE has to clear *both*, since they
  > measure different sources of noise (SAE-training stochasticity vs. the
  > model's own forecast nondeterminism).
  >
  > **The absent case renders a sentence, not nothing** (§2.5): a run with
  > `n_seeds: 1` gets "Not measured — this run trained one SAE per target …
  > every number above is therefore a single draw … with no floor to read it
  > against". A bare single-seed ΔMASE with nothing beside it reads exactly
  > like one that cleared a floor, which is the failure this whole item
  > exists to prevent.
  >
  > **Verified.** New `tests/test_sae_seed_floor.py`, 10 tests: the spread
  > statistics (ddof=1, dropped non-finite entries with a falling `n`, n=1
  > giving `sd: 0.0` rather than a fabricated spread, empty giving
  > `{"n": 0}`); the tri-state lookup in all four of its states; both
  > rendering paths; that `_metric_row` covers every metric `SEED_FLOOR_METRICS`
  > asks for and yields `None` for a failed check; that every configured
  > hyperparameter reaches the replicate trainings; and — mirroring
  > `test_aux_k.py`'s "the flag is not inert" pattern — that two seeds trained
  > on identical synthetic data produce **different** dictionaries while one
  > seed still reproduces exactly, since a floor built from seeds that all
  > agree would be identically zero and would license any delta at all. The
  > module docstring states what is *not* covered: the assembled `⚠` sentence
  > inside `_sec_sae`, which needs a real run directory with an activation
  > store.
  >
  > **Cross-checked against the real recorded artifacts**, not only synthetic
  > fixtures (§2.4). Feeding `runs/medium_run_chronos_base/sae/meta.json`
  > (a genuine single-seed run) through `_sae_seed_floor_block` renders the
  > "Not measured" path; feeding that run's existing
  > `sae/repeat_variance.json` through it reproduces the measured floor
  > exactly — TimesFM `stacked_xf.18` `± 0.121 over 5 seeds` (window and
  > token alike), Chronos-T5-Base `encoder.block.6` `± 0.242` (window) and
  > `± 0.211` (token), matching the numbers this item's parent recorded on
  > 2026-08-12 to three decimals. Note what the tri-state then says about
  > those real numbers: all four come back `resolvable=True`, but
  > Chronos-T5-Base's token ΔMASE clears its own spread only barely
  > (+0.2457 against sd 0.2113) — the rendering is doing exactly the job it
  > was built for on the one number in this repo that most needed it.
  >
  > **Not done:** no run has yet been executed with `n_seeds > 1`, so no
  > report on disk currently shows the table — only the "Not measured"
  > sentence. Sizing the floor for a real target still means running
  > `run_sae_repeat_variance.py` (or setting `n_seeds` and re-running the
  > stage), and the floor is per (run, target, layer, dict size, `k`) and does
  > not transfer.

- [x] **Does the L2 null-baseline verdict hold in the reverse direction
  (`TimesFM→Chronos`)?** Named as "the named next step" in the item above and
  never given its own checkbox. `l2_stitching.py` already computes both
  directions; this is a re-analysis of existing artifacts, not a rerun.
  **Done 2026-08-12** — yes, and for the same reason the forward direction
  did: real cross-model gain beats Chronos's own untrained-twin floor at
  every TimesFM depth (+0.38 to +0.49, p=0.002) and beats TimesFM's own
  floor at every layer from `stacked_xf.10` through `.18`, losing only at
  the two shallowest layers where that floor's self-predictability artifact
  is largest. The previously-recorded "0.318 real vs. 0.388 null" pessimism
  was the same wrong-layer comparison the forward direction already
  corrected. `compare_l2_depth_curve(..., direction=)` +
  `run_null_baseline_test.py --direction` added for it (2 new tests). Full
  numbers in §16 E9's fourth follow-up.
- [ ] **Does `work_bend` remain the bake-off winner on a second Chronos size
  and against the per-window-patching secondary gold?** Named inside item 2's
  resolution text as the remaining scope and never tracked separately. The
  production default currently rests on one corpus and one checkpoint pair
  per architecture.
- [x] **Should §16 E19's multivariate-axis decision be made retroactively?**
  E19 asks for the `[series, variate, window, dim]` decision to be made
  *before* Phase 4 forces an improvised one — but Phase 4 already shipped
  (Chronos-2, Sundial), and the improvisation was made: Chronos-2's GROUP
  cross-series attention axis is deliberately out of scope and
  `_scan_attention` resolves to TIME attention by first-match. The open
  question is now **"ratify or reverse"**, not "decide in advance," and E19's
  text should be updated to say so (see §16's 2026-08-11 corrections block).
  **Answered 2026-08-12: ratified.** Univariate-only is now a decided,
  documented envelope edge in `CLAUDE.md` §12 rather than an unexamined
  assumption, with Chronos-2's GROUP axis named as the concrete exclusion and
  a stated rule for future cross-series adapters (capture the time axis,
  declare the cross-series axis out of scope, say so). The unknown magnitude
  of the resulting capture-coverage gap is recorded as part of the decision,
  not hidden by it. §16 E19 is `[x]`.

---

## 14. Session log

> **The rule, restated because it was not followed.** One entry per session,
> **three lines or fewer**: what was picked up, what changed, and which
> Findings block owns the numbers. This is a changelog. The *content* of what
> was learned belongs in the relevant section's Findings block, and a number
> that exists only here is a number nobody will find.
>
> **Compressed 2026-08-18.** The log had grown to 2,743 lines across ~85
> entries — a full narrative diary, in direct contradiction of its own header
> above, and the single largest block of text in this file. Every entry is
> preserved verbatim in `ROADMAP_ARCHIVE.md`. What follows is the orientation a
> new session actually needs: what happened, in periods, with pointers.
>
> ⚠️ **Known gap:** the 2026-08-17 sessions (Stage 1's b2/b3 completion and
> V4's build + the §11.31 R² bug) are **not in the archived log** — they were
> written into §6.2.1's Findings and §0.5 but no log entry was appended. The
> Findings blocks are authoritative; the log is incomplete for that date.

### History at a glance

| Period | What happened | Numbers live in |
|---|---|---|
| **08-03** (5 sessions) | Roadmap authored. Phase 1's "wired but unverified" real-data path turned out **broken four independent ways** (`datasets>=3` removed script loading, `tsbootstrap`'s API rewritten, `sequential_par` a literal stub, a source-injection bug only `mixture` avoided) — all fixed, first real corpus built. Then the **first real-checkpoint `tsfm_lens` run ever**, which found two more real bugs (zarr v3-vs-v2, the miscalibrated impulse probe). | §5, `CLAUDE.md` §11.9–§11.16 |
| **08-04** (3 sessions) | Phase 0 (report hygiene) implemented. The user-reported "detrend/spike/dropout barely react" pattern investigated against real artifacts and found **not a bug** — sparse input footprint by construction. Checkpoint-breadth run started. | §4, §5, `CLAUDE.md` §6.5 |
| **08-05** (10 sessions, first autonomous loop) | Phase 1 closed (§5.3 size control, §5.5 cross-run aggregator). Phase 2a's layer selector **redesigned after its first version was shown to pick the worst layer**, bake-off-tested, and wired into production as `work_bend`. SAE baseline built, broken, fixed. Crosscoder feasibility test. Phase 3's four sweeps. | §5.3, §5.5, §6.1, §6.1.1, §6.2, §7 |
| **08-06** (6 sessions) | The **audit sweep**: §15 written (19 items) and then all 19 fixed in the same day, plus §16 authored. E9's untrained-weights null baseline built — now the most-cited control in the file. | §15, §16 E9 |
| **08-07** (6 sessions) | E9's eyeball comparison became a **real significance test**, and its depth-curve version *reversed* the peak-pair reading for both L1 and L2. Phase 2c (provenance detection) **falsified**. Chronos-2 added as a fourth adapter, surfacing a shared-infrastructure hook bug. | §16 E9, §6.3, §9 |
| **08-10** (10+ firings) | `CLAUDE.md` §2.8 (background delegation) written down as doctrine. Sundial added as a fifth adapter. The bake-off's **seed instability chased to root cause** — not GPU nondeterminism but A4's sampling fix landing between two "identical" runs (`CLAUDE.md` §11.24) — then fixed. A stale zarr-v3 store found reading back **silently empty** (§11.25). | §6.1.1, §9, §13 |
| **08-11** (24 firings + a planning pass) | E13 spectral lens, E2 doctor, E15 per-token SAE eval (which **retired the Chronos forecast-preservation failure** as a window-broadcast confound), E16 cross-model feature matching, E23 diversity gates. Then the user-directed pass that authored §0.5 and §6.2.1's full crosscoder build plan. | §16 E13/E2/E15/E16/E23, §6.2.1 |
| **08-12** (strategic pass + Stage 0) | §17–§22 authored (gap analysis, equal grounds, architecture adaptivity, new capabilities, two audiences). F6 and F2 implemented. Stage 0's sweeps run: **AuxK decisive, H1 and H2 refuted**, and the gate still failing on a criteria question. `CLAUDE.md` §11.27 (editing a script a background loop re-reads) self-inflicted and recorded. | §18 F2/F6, §6.2.1 findings (1)–(15) |
| **08-13** | `_per_block_flops` fixed (`CLAUDE.md` §11.28) — Chronos's coverage headline goes from upper bound to measurement **without its value changing**. F1's depth-axis module built (nothing consumes it yet). **Stage 0 closed** at 5/5 seeds, after measuring the premise the whole gate rested on and finding it **inverted** (`CLAUDE.md` §11.29). Stage 1's ladder built and run: V1 fails L-B at 4,608 rows. | §18 F1/F4, §6.2.1 findings (16)–(17), Stage 1 |
| **08-17** (⚠️ unlogged) | Stage 1 completed: L-B and L-C re-run for real at 46,382 rows — **L-B now clears its floor**, and clause 2 resolves **against V1** (loses to V0, p=0.002). V4 built; a real R² baseline bug found on live data and fixed (`CLAUDE.md` §11.31), after which most nominally-"specific" atoms turn out not to survive. | §6.2.1 Stage 1 + V4 Findings |
| **08-18** | This restructuring pass: `ROADMAP.md` 16.4k → ~5k lines, four competing orderings collapsed into §0.5's single queue, ~25 items parked with un-park triggers in §22, and the crosscoder scoped to its remaining critical path (§6.2.1's triage). No code changed, no recorded number altered. | §22, §6.2.1 |
| **08-18** (autonomous cron firing, §0.5 queue item 1) | §6.2.1 CLOSED: V2 (BatchTopK) built (already coded, not yet run), scored at 3 seeds — does not flip Stage 1c's clause 2 at any seed, and its own design goal (lower `frac_specific_a`) doesn't materialize either. New finding: the L-B untrained-twin floor is itself seed-fragile (0.696/0.792/0.856), so the prior single-seed "decisively clears" claim was overstated. Stage 4 written up (three-part deliverable + methodological result); Stage 3 confirmed gated shut. Full suite 418 passed. `CLAUDE.md` §13 item 3 corrected to match. | §6.2.1 Stage 2/4 Findings, `CLAUDE.md` §13 |

### What the log is actually evidence for

Three patterns recur often enough to be worth stating once, since they are the
best-supported claims in this file and they are claims about *method*:

1. **"Wired but unverified" has meant "broken" every single time it was
   tested.** The real-data path (four ways), the zarr storage layer, the
   alignment probe, `output_attentions`, the `sequential_par` stub. Not once
   did an untested path turn out to be fine. Treat the label as a defect
   report.
2. **Measuring the premise beats measuring the hypothesis.** Stage 0 spent four
   sessions failing a gate whose founding sentence — "the per-model baseline
   hit the same wall" — was never checked and was **backwards**
   (`CLAUDE.md` §11.29). Same shape in §11.14 (the wrong cost variable) and
   §11.3 (the wrong cause). Check the sentence the plan rests on first.
3. **A synthetic-only fix is not a fix.** `factor_emergence`'s peak-weighting
   passed a purpose-built planted test and *regressed* on real activations
   (§11.18); V4's R² bug survived two unit tests because both used dense data
   and the real crosscoder is sparse (§11.31). Re-test on real data before
   trusting a fix, not after.

### Entries from here on

Append below, three lines maximum, newest last.

- **2026-08-18 · Streamlining pass** (docs only, no code, no number changed).
  16,425 → ~10.7k lines. Four competing orderings collapsed into §0.5's single
  queue; §22 added as the parked list with an un-park trigger per entry; §15's
  19 closed fix plans became one reuse table; §14's 85-entry narrative became
  §14's history-at-a-glance. Every Findings block preserved verbatim; all
  removed text relocated to `ROADMAP_ARCHIVE.md`, not deleted (§0.2). Also
  answered the three crosscoder questions the pass was asked for — §6.2.1's
  triage block: finish V2 + Stage 4 (~2 days), Sonnet can execute against the
  pre-registration but not make the four judgment calls that consumed the
  sessions, and shortening = honoring the fired decision rule rather than
  cutting compute (the ladder runs in 37 s).

---

## 15. Audit — silent-failure paths and fragile mechanisms

> **Status: 21 closed, 0 open (2026-08-18).** A 2026-08-06 read-the-code-
> against-the-docs pass found nineteen items; all nineteen were fixed in the
> sweep that followed, and are compressed below into the conventions they
> left behind. **A21 and A20 (found later, 2026-08-12) are also now closed**
> — this section is fully reference, not open work, until a future audit
> pass adds to it.
>
> **How this section differs from `CLAUDE.md` §11.** §11 records traps that
> were *hit and fixed* and cost real time to find. This records traps found by
> *auditing* rather than by being bitten. When an item here is fixed, move its
> lesson to §11 if it was expensive, and mark the item `[x]` here — don't
> delete it (§0.2).
>
> **Priority key.** **P1** = can silently produce a wrong number that reaches
> a report or a recorded finding. **P2** = a stated invariant is unenforced, or
> a documented limitation blocks planned work. **P3** = real, worth doing,
> nothing currently depends on it.
>
> 🔴 **The one theme worth carrying forward.** Twelve of the nineteen were the
> same shape: *a mechanism that degrades quietly where the doctrine
> (`CLAUDE.md` §2.5) says it must degrade loudly, because "loudly" was
> implemented as a log line.* Logs are not loud inside a thirty-minute run that
> prints hundreds of them, and they are not loud at all once the deliverable is
> an HTML file someone opens three weeks later. The standing rule that came out
> of it, and the one to apply to any new stage: **every degrade, fallback, cap,
> and skipped check must land in a run artifact and in the report**, not only
> in stdout.

### A1–A19 — closed 2026-08-06 · the conventions they established

> **Compressed 2026-08-18.** All nineteen are `[x]`. Each carried a
> ~60-line implementation plan, which is now dead scaffolding: the code
> exists, the lesson (where it cost real time) is in `CLAUDE.md` §11, and
> the full original text — plan *and* verbatim Findings — is in
> `ROADMAP_ARCHIVE.md`. What is kept here is the part a *new* stage still
> needs: **the defect, and the mechanism that now guards against it**, since
> §0's rule 9 asks any new stage that subsamples, reuses artifacts, or
> degrades a capability to follow these rather than reinvent them.
>
> Read this list before writing a stage. Read `ROADMAP_ARCHIVE.md` only if
> you need to know why a fix was designed the way it was.

**The reuse table.** Mechanism names are real functions — call them, don't
write a second one.

| Item | The defect | Reuse this |
|---|---|---|
| **A1** P1 | `layer_screen` screened only what extraction happened to store, so at `capture_layer_stride: 2` half of TimesFM's blocks were invisible to the selector deciding where expensive analysis goes | `LayerScreenConfig.stride`/`require_full_capture`/`keep_store`; `layer_screen.py::_run_screening_extraction` runs its own stride-1 pass into a temporary store. Records `fair_to_all_layers` and reports the **model-block** denominator, not the captured count |
| **A2** P1 | The impulse alignment check ran, logged one line, and threw its result away — invariant 7 enforced by human memory | `extraction/alignment.py::run_alignment_gate` — probes every captured layer, writes the full per-layer record, and **raises** below `alignment.min_diagonal_frac` (0.5) unless `alignment.on_failure: warn` |
| **A3** P1 | Stage skipping was bare path existence, so editing a config and rerunning into the same `run.name` mixed two configs' artifacts under a complete-looking report | `manifest.py` (`resolve_config_keys`, `fingerprint_stage`, `diff_resolved`) + `config_keys` on every `Stage`. Section-wildcard keys, so a `report.title` edit does not invalidate `extract` |
| **A4** P1 | Corpora are written grouped by task and four call sites took a **head slice**, silently selecting a family-skewed prefix — this one retroactively confounded recorded numbers (see `CLAUDE.md` §11.24 for what it cost later) | `utils.sample_rows(n, k, seed, strata=None)` — deterministic, sorted, logs realized-vs-population family composition on every call. **Never head-slice a corpus** |
| **A5** P1 | The report omitted skipped *and failed* sections with no trace in the HTML — honest for whoever watched the console, invisible to the file's actual audience | Every builder carries `config_attr`, giving a real three-way `rendered` / `skipped: not enabled` / `failed` status, rendered **and** written to `report/coverage.json` |
| **A6** P2 | A mis-resolved `family_key` silently regrouped every per-family statistic; an unlabeled corpus collapsed to one family and the report still rendered "comparisons" over one group | `data.py::_resolve_family_key` (explicit key, no fallback chain) + `_check_family_key_resolution` (aggregate hit rate). Distinguishes a degenerate *comparison* from a still-meaningful *breakdown* |
| **A7** P1 | A run directory recorded the resolved config and nothing else — no git SHA, no library versions, no store schema version | `utils.run_provenance()`; store attrs carry a schema version. Found `timesfm` exposes no `__version__` at all, hence `_pkg_version`'s `importlib.metadata` fallback |
| **A8** P1 | Invariant 1 was carried as a known-red golden-hash test across five sessions, on an unconfirmed hypothesis | Confirm-before-fixing paid off: **the test passes** in the persistent environment, and a direct numpy 1.26.4-vs-2.1.0 comparison of `choice(replace=False)`/`normal()` found them bit-identical, *refuting* the hypothesis. Cause of the original report still unexplained (`CLAUDE.md` §11.13) |
| **A9** P2 | `random_parametric`'s sampled archetype never reached the analysis half, so all 12 archetypes collapsed into one family | `data.py::_assemble` carries `archetype` and `generator` columns through the same stratified subsample as A4 |
| **A10** P2 | Archetype dummies scored real-derived series (which carry an empty `GroundTruth` by design) as valid negatives, so a feature separating synthetic from real-derived scored as an archetype detector | `sae/ground_truth.py::_add_dummy_columns` uses NaN-not-applicable, not 0 |
| **A11** P2 | MASE's scale term collapses on intermittent or near-constant series, and `mase()` is the metric behind L0 headlines, ablation ΔMASE, L3 restoration and SAE preservation alike | `stats.py::_mase_scale()` + shared `dominant_period()` (deduplicated from `attention.py`'s private copy) |
| **A12** P2 | Corruption strengths were set independently, so `level_shift` dominated the sensitivity chart by construction at 4–12× everything else | `l3_perturbation.py::calibrate_corruptions` + `L3Config.calibrate: none\|input_energy`. Default stays `none`, so recorded numbers reproduce |
| **A13** P1 | Nothing measured how much a model's own MASE varies between two identical calls, so every Δ in the repo was compared against zero | `l0_behavioral.py::_measure_noise_floor()` — a cheap sub-step of L0, not a new stage. F6 then put every delta in floor units |
| **A14** P1 | No CI. Every "suite green" was a session's word for it, and five upstream breaks (`CLAUDE.md` §11.8/.9/.10/.15/.17) were each found by a human losing a session on a new machine | `.github/workflows/ci.yml`, three jobs: pinned CPU suite + `compileall` + the invariant-11 absolute-path grep + a full smoke run |
| **A15** P2 | `confirm` derived its hypotheses from a *mutable* dev artifact, and the report accumulated ~22 findings with no record of how many comparisons produced them | `analysis/hypotheses.py::build_registry(cfg)` + a frozen registered set. The repo's strongest epistemic claim is now mechanized rather than conventional |
| **A16** P2 | Stages that must fit one batch silently clamped their configured cap to `batch_size` — differently per model, since batch sizes differ | `utils.capped_take(requested, **limits)` → `n_requested`/`n_realized`/`limited_by`, and it warns when the request bound |
| **A17** P3 | `benchmark_validation` reported per-group diversity as point estimates at any group size (n=39 next to n=3154), and the O(n²) matcher blocked full-scale validation | `diversity.py::_bootstrap_group_ci` + `min_group_n`; matcher blocking |
| **A18** P3 | Absolute placeholder paths in a committed config (invariant 11 has no placeholder exemption) and a stale checkpoint id in `requirements.txt` | The A14 CI grep now flags any config path starting with `/` or a drive letter — a repo-wide grep found a **second** instance the original evidence had missed |
| **A19** P3 | `float16` activations were written with no finiteness check, so an overflowing layer entered CKA/ridge as `inf`/`NaN` and surfaced as a plausible number | `store.py::write_batch` counts non-finite elements per (model, layer) and surfaces them. Also found `store_dtype: float32` already existed as config and was silently non-functional |

### A20 — The impulse-alignment probe's amplitude constant is calibrated per (checkpoint, context length), and silently degrades at other context lengths `[x]` · **P2** · found 2026-08-12, fixed 2026-08-18

**Evidence.** `extraction/alignment.py::impulse_alignment_check` perturbs
its probe signal by `0.25 × base.max()`. That constant is §11.16's fix, and
§11.16 records it as calibrated against `amazon/chronos-t5-small` at
`context_len: 512`. Building `configs/crosscoder_stage0.yaml` at
`context_len: 448` (forced by that corpus's sequence lengths — §6.2.1's
Stage 0 Findings (b)) drops Chronos-T5-Base's diagonal-hit fraction to
**0.50**, which passes the `min_diagonal_frac: 0.5` gate with **exactly zero
margin**. The full depth profile and amplitude sweep are in §6.2.1's Stage 0
Findings (c) and are not repeated here. The token↔time map itself was
directly verified correct at 448 (449 tokens = 448 + EOS, mask all 1s,
`postprocess_tokens` → 448, spans `[0,1]…[447,448]`), and dropping the
amplitude to 0.05 restores 1.00 at blocks 0–7 and 0.02 restores 1.00 at every
block — so this measures the probe, not the adapter.

**Blast radius.** Two distinct things, and the second is the reason this is
P2 rather than P3. (1) A **false-negative gate**: at some other (checkpoint,
context length) combination the same effect will push the fraction below
0.5 and fail extraction on a store that is actually fine — or, worse, sit
just above it and be read as a real alignment warning. (2) A **false-positive
gate**, which is the dangerous direction: nothing guarantees 0.25 is
*conservative* at every combination, and a genuinely misaligned adapter
could clear 0.5 on tokenizer-rescaling noise alone. Note this interacts with
A2 (already fixed): the in-pipeline check now records its result, so a
miscalibrated probe now writes a misleading artifact rather than only
logging one.

**Fix plan.** Do **not** simply lower the constant — every recorded
alignment number in this repo was measured at 0.25, and changing it silently
rewrites what "1.00 diagonal hits" has meant historically (a §2.1 downstream
call, and exactly the §11.24 class of trap where a shared-infrastructure
change rewrites the meaning of an unchanged config). Instead: (a) make the
amplitude a parameter with the current `0.25` as its default, so nothing
recorded moves; (b) have `impulse_alignment_check` *calibrate* it — sweep
downward (0.25 → 0.15 → 0.10 → 0.05 → 0.02) and pick the largest amplitude
at which the number of unrelated tokens whose IDs change is below a small
threshold, which is the direct measurement of the confound rather than a
proxy for it; (c) record the chosen amplitude and the unrelated-token count
in the artifact A2 already writes, so a reader can tell a self-calibrated
run from a fixed-amplitude one; (d) for adapters with no input
re-quantization (TimesFM, which was 1.00 across the entire sweep in
§11.16's original measurement), the calibration is a no-op and should
terminate at the first amplitude. Then re-verify that Chronos-T5-Small at
512 still reports 1.00 at every layer, i.e. that the calibrated path
reproduces the recorded result at the recorded setting before anything new
is trusted.

**Findings — 2026-08-18, fixed exactly as planned (a)–(d), plus one honest
result the plan didn't anticipate.**
- **Mechanism.** New `extraction/alignment.py::calibrate_impulse_amplitude(
  adapter, window, candidates=(0.25,0.15,0.10,0.05,0.02), max_unrelated_frac
  =0.02)`: for adapters whose new, optional `ModelAdapter.token_ids(prepared)`
  hook returns non-None (currently `ChronosAdapter` only, backed by the
  `prepare()` call it already makes — `self.pipeline.tokenizer.
  context_input_transform`, no new tokenizer call needed), it perturbs one
  probe window at each candidate amplitude, largest first, and counts token
  ids that change *outside* that window's own declared span — the literal
  "diff tokenized ids before/after" mechanism §11.16's original ad hoc
  diagnosis used, now automatic and per-run. The first amplitude at or below
  `max_unrelated_frac` is chosen; adapters with no `token_ids()` (TimesFM/
  Sundial/Chronos-Bolt/Chronos-2 — plan point (d)) return `candidates[0]`
  (0.25) with `calibrated: False` and skip the sweep entirely, at zero extra
  cost. `impulse_alignment_check` gained an `amplitude: float = 0.25`
  parameter (plan (a) — every existing caller's behavior is unchanged unless
  it opts in); `run_alignment_gate` gained `calibrate: bool = True` and now
  calls calibration by default, recording both the chosen `amplitude` and
  the full `calibration` sweep dict in the artifact it already writes (plan
  (c)) — `calibrate=False` reproduces the exact pre-A20 record shape
  (`amplitude: 0.25, calibration: None`) for anyone needing a pre-fix
  artifact. `run.py --check-alignment` and `doctor.py`'s full preflight both
  now calibrate before probing and print/log the chosen amplitude;
  `models/conformance.py::check_adapter_conformance` (used by every mock
  adapter test) deliberately still calls the fixed-default path, since mocks
  have no `token_ids()` override and calibrating there would just add a
  no-op forward pass to a hot test path. New `AlignmentConfig.
  calibrate_amplitude: bool = True` wires the default-on behavior into
  `run_extraction`'s own `run_alignment_gate` call.
- **Unit tests (`tests/test_alignment_calibration.py`, 7 new, all
  passing).** A deterministic double (`_RequantizingAdapter`, subclassing
  `MockStepAdapter`) whose `token_ids()` shifts every token's id by a fixed
  offset whenever the input deviates from a stored clean reference by more
  than a tunable threshold — engineered so the confound is exact and
  predictable rather than approximated from real signal statistics. Covers:
  the true no-op path for adapters with no `token_ids()` (mock adapters,
  `calibrated: False`, `amplitude` stays 0.25); a confound that never fires
  reproduces 0.25 exactly (a checkpoint the historical constant was already
  safe for is unaffected); the core sweep-down-to-the-largest-safe-amplitude
  case (0.25/0.15/0.10 confound, 0.05/0.02 don't → chooses 0.05, and never
  tries 0.02 — early stopping verified directly on the sweep list); the
  all-candidates-confound fallback (returns the smallest tried, `0.02`,
  rather than raising or silently picking something unsafe);
  `impulse_alignment_check`'s default-amplitude backward-compatibility
  (explicit `amplitude=0.25` call is bit-identical to the no-argument call);
  and `run_alignment_gate`'s new `amplitude`/`calibration` record fields
  under both `calibrate=True` and `calibrate=False`.
- **Live re-verification against real checkpoints — both required checks
  from the fix plan, plus the bug-surfacing case.**
  - **Chronos-T5-Small @ context_len 512** (`configs/
    layer_screen_experiment.yaml`, the exact historically-recorded
    combination): calibration measured a *nonzero* unrelated-token churn at
    0.25/0.15/0.10 (15 tokens, 3.125% of the 480 unrelated tokens — real,
    previously unmeasured noise that happened to sit below whatever margin
    made 0.25 "look safe" before) and chose **0.05** (0 unrelated tokens
    changed). The diagonal-hit fraction at the calibrated amplitude is
    **min=1.00, mean=1.00 across all 6 probed layers** — bit-identical to
    the historically recorded result at 0.25. So the fix plan's literal
    acceptance test (does the calibrated path reproduce the recorded result
    at the recorded setting) is satisfied, but not by reproducing the
    recorded *amplitude* — the honest finding is that 0.25 was already
    slightly outside a strict calibration bar at this exact checkpoint/
    context length, and only happened not to matter for the diagonal-hit
    measurement itself. That is a real result the fix plan text didn't
    anticipate (it assumed 0.25 would calibrate back to 0.25 wherever it was
    already "safe"), recorded here rather than silently smoothed over.
  - **Chronos-T5-Base @ context_len 448** (`configs/crosscoder_stage0.yaml`,
    the exact config/checkpoint/context-length combination that surfaced
    this item — §11.26, §15 A20's own Evidence): calibration measured a much
    larger confound at 0.25/0.15 (45 of 416 unrelated tokens, 10.8%) and
    0.10 (30, 7.2%), converging to **0.05** (0 changed). At that calibrated
    amplitude the probed layer (`encoder.block.10` — this config's
    `capture_layer_stride` captures only one Chronos block) reports
    **0.93**, not the 1.00 the earlier §6.2.1 Stage 0 sweep found at blocks
    0–7 for the same amplitude. This is not a contradiction: block 10 is
    considerably deeper than blocks 0–7, and `impulse_alignment_check`'s own
    docstring already states mid/late-depth values below 1.0 are the
    *expected* signature of ordinary attention mixing, not a broken
    mapping — 0.93 at a late block is consistent with that, not evidence of
    a residual bug in this fix. The practical result is what matters most:
    **0.93 clears `min_diagonal_frac: 0.5` with a wide margin**, replacing
    the pre-fix 0.50-with-zero-margin reading this item exists to fix.
- **What this does not close.** The calibration sweep's `max_unrelated_frac
  =0.02` default and its five hardcoded candidate amplitudes are themselves
  a new calibration, chosen for this session's two real checkpoints and not
  independently validated against a third — the same class of caveat §11.16
  and §11.26 already recorded for the constant this replaces, now one level
  up. `conformance.py`'s fixed-default path means the mock-adapter test
  suite's alignment numbers are still measured at 0.25 uncalibrated, which
  is fine (mocks have no re-quantizing tokenizer to confound) but worth
  remembering if a future mock ever grows one. Full `tsfm_lens` suite green
  at **459 passed** (up from 452 before this item).

---

### A21 — An unknown key in any config section is silently dropped `[x]` · **P2** · found 2026-08-12, fixed 2026-08-18

**Evidence.** `tsfm_lens/config.py::_build` iterates the target dataclass's
own fields and copies across only the keys it recognizes:

```python
for f in dataclasses.fields(cls):
    if f.name not in data:
        continue
```

Anything else in the YAML is discarded without a word. A misspelled knob
(`capture_layer_stide`, `min_diagonal_fraction`, `verbose_series_count`)
therefore runs the whole pipeline at the *default* value while the config
file on disk — and the `config_resolved.yaml` copied next to the run —
both read as though the setting were in force. Found while writing
`tests/test_budget_stage.py` for §18 F2, when a test asserting that a
nonsense `budget:` key raises turned out to assert something the loader
does not do; the test now pins the actual behavior and points here.

**Blast radius.** Every config section, not `budget:`. This is the same
failure shape as A3 (stale artifacts) one level earlier: the run is
internally consistent and looks complete, and nothing distinguishes
"you set this" from "you meant to set this." It is worse than a stale
artifact in one respect — `config_resolved.yaml` is the artifact a future
session reads to find out what a recorded number was measured under, and
it will show the default, not the typo, so the discrepancy is invisible
even in hindsight. Note this is *not* how the two standalone experiment
loaders behave: `run_crosscoder_stage0.py --params` deliberately rejects
unknown keys (§6.2.1 Stage 0's exit criteria), so the strict behavior
already exists in this repo and is already the one judged correct where
someone thought about it.

**Fix.** In `_build`, collect `set(data) - {f.name for f in fields(cls)}`
and raise a `TypeError` naming the section, the unknown key(s), and the
closest valid field name by `difflib.get_close_matches` — a typo's whole
cost is that the right name was nearly typed. Two things to check before
landing it, both §2.1 downstream questions rather than code questions:
(a) grep every YAML under `tsfm_lens/configs/` and every config dict in
`tests/` for keys no dataclass declares, since any that exist today would
start failing loudly (that is the point, but the list should be *seen*
first, not discovered by a red suite); and (b) decide explicitly what
happens to `models[*]`, which is built through the same helper — an
adapter-specific knob passed through to one adapter would be a legitimate
reason to allow extras there and nowhere else. Test: a typo'd key in each
of a nested section, `l3.patching`, and a `models[*]` entry.

**Findings — 2026-08-18, fixed exactly as planned; the pre-landing audit
came back clean.**

1. **(a) — the audit found zero live victims.** A script exercising every
   YAML under `tsfm_lens/configs/` through `config_from_dict` (top-level
   sections, `l3.patching`, every `models[*]` entry) found **no unknown
   keys anywhere** — every already-committed config was already valid
   under the strict field set, so landing the raise broke nothing on disk.
   One thing the audit script itself had to be corrected for: three
   `configs/crosscoder_stage0_*.yaml` files use a *different*, unrelated
   config format for the standalone `run_crosscoder_stage0.py` script
   (`run: runs/some/path` as a bare string, not a `PipelineConfig`
   mapping) — the first audit pass mistakenly iterated that string's
   characters as if they were dict keys and reported 16 bogus "unknown
   keys" per file. Filtering to only YAMLs whose `run:` is itself a
   mapping (the actual `run.py`-consumable ones) resolved it. Every other
   `run_*.py` standalone script (`run_crosscoder_ladder.py`,
   `run_noise_snr_sweep.py`, `run_spectral_lens.py`, the sweep scripts,
   etc.) loads an existing run's `config_resolved.yaml` rather than
   constructing a raw dict inline — and `config_resolved.yaml` is written
   by `dump_config` via `dataclasses.asdict`, so it can only ever contain
   declared fields by construction. None of these needed touching.
2. **(b) — resolved without adding leniency anywhere.** `ModelConfig`
   already has a declared `kwargs: dict` field (already used by
   `configs/default.yaml` and consumed by `chronos_adapter.py`/
   `sundial_adapter.py` for adapter-specific knobs like `num_samples`), so
   the "legitimate reason to allow extras" the fix plan flagged is already
   satisfied by that existing field — adapter-specific knobs nest under
   `kwargs:`, they don't need to be tolerated as flat unrecognized keys on
   the model entry itself. `models[*]` therefore gets the same strict
   treatment as every other section, with no special case.
3. **Implementation.** `_build(cls, data, section=None)` gained a
   `section` parameter (threaded through every call site in
   `config_from_dict`, including `l3.patching` → `"l3.patching"` and each
   `models[i]` → `f"models[{i}]"`) purely for error messages, and now: (i)
   raises `TypeError` if `data` isn't a mapping at all (a new, incidental
   robustness improvement — the old code would have thrown a confusing
   "NoneType is not iterable"-class error deeper in for this case, not
   silently succeeded, so this wasn't a live gap, just a clearer message
   for it), then (ii) computes `set(data) - declared_fields` and raises
   naming every offending key plus its closest valid-field suggestion via
   `difflib.get_close_matches`. `key: null` (an explicit empty section,
   distinct from an unrecognized key) still falls through to `cls()`
   defaults exactly as before — only genuinely unrecognized keys are new
   errors.
4. **Test coverage.** `tests/test_budget_stage.py`'s test that explicitly
   pinned the *old* silent-drop behavior (its own comment named this exact
   item as the reason it existed) now asserts the raise instead. New
   `tests/test_config_unknown_keys.py` (8 tests) covers: a top-level
   section typo with a working suggestion, a typo with no close match, a
   typo inside `l3.patching`, a typo inside `l3`'s own fields, a typo in
   `models[1]` (confirming the index is named), a fully-valid config still
   building correctly (no false positives), `l3: null` still resolving to
   defaults, and a non-mapping section (`alignment: "not-a-mapping"`)
   raising a clear type error rather than a confusing one. Full
   `tsfm_lens` suite: **434 passed** (426 pre-existing + 8 new), zero
   regressions.
5. **Live CLI check.** Ran `python run.py --config <smoke.yaml with
   `sanity_check` typo'd to `sanity_chek`> --stages extract` through the
   real CLI (not just pytest): fails immediately at config-load time with
   `TypeError: unknown key(s) in config section 'alignment': 'sanity_chek'
   (did you mean 'sanity_check'?) -- valid fields are ['depth_axis',
   'min_diagonal_frac', 'on_failure', 'sanity_check', 'window']` — before
   any model loads or any corpus is read, exactly the "fail loudly, before
   wasting a run" behavior this item exists for.

---

## 16. Enhancement backlog — what "anyone can analyze any model with one button" actually requires

> **Added 2026-08-06 (planning only).** §1's north star is "the
> transformer-lens for TSFMs": a stranger drops in a checkpoint and gets back
> a rigorous, well-explained comparison. Measured against that bar, the repo's
> *analysis* depth is well ahead of its *adoptability*: today a new user must
> build a corpus with a second package, hand-write a `ModelAdapter`, run
> `--discover-layers`, run `--check-alignment` and judge the output, hand-edit
> a YAML, and have a GPU — before the first report. Each item below is framed
> as what a user wants, then how to build it, then what it depends on.
>
> **Tiers.** **T1** closes the gap between "a research pipeline that works" and
> "a tool someone else can use." **T2** is rigor a comparison tool has to have
> to be believed. **T3** is new analysis capability a TSFM interpretability
> user will specifically ask for (this is where transformer-lens parity and
> genuine novelty live). **T4** is breadth and scale.
>
> §15's P1 items are prerequisites for all of this, not competitors with it:
> a one-button tool that silently samples a biased prefix (A4) or skips its
> alignment gate (A2) is worse than a five-command tool, because it removes
> the human who would have noticed.

> **Status, applied 2026-08-18.** A 2026-08-11 pass found every "Depends on:
> A…" line in this section stale and wrote a corrections block; those
> corrections have now been applied to the items themselves and the block is
> archived. The three facts worth keeping:
> - **§15 blocks nothing here.** All of A2, A3, A4, A5, A7, A13, A14, A15, A16,
>   A17 are `[x]`, so every T1 item is unblocked by §15 today. Where an item
>   still lists a dependency, only the E-dependency is real.
> - **E3's premise got stronger, not weaker, when A2 landed.** A2 means the
>   in-pipeline alignment check no longer discards its result — but an
>   *enforced* check still only validates a **declared** span table, it never
>   *derives* one. Deriving one is what makes a `GenericHFAdapter` possible and
>   is E3's actual argument.
> - **E5's stated purpose was wrong.** `report/meta_report.py` does not glob run
>   directories; it takes an explicit `--runs a,b,c` list. A registry's value is
>   the append-only history and provenance keying, not "stop globbing."
>
> **Triaged 2026-08-18.** Live: E3, E9, E11, E14, E17, E20a.
> Parked with un-park triggers in §22: E1, E4, E5, E7c, E18, E20b, E21, E22,
> E24. Done: E2, E6, E7a, E7b, E8, E10, E12, E13, E15, E16, E19, E23.

### T1 — Adoptability

- [ ] **E1 · A zero-config entry point** — **parked (§22.3).** `tsfm-lens
  compare --model hf://… --model hf://… --preset standard`, resolving adapters
  from a registry by checkpoint id and defaulting to E4's corpus. The whole T1
  product bundle (E1/E4/E5) is parked together; §22.9's ordering constraint is
  the reason — automation without the legibility layer removes the expert who
  would have known not to believe an unequal axis.
- [x] **E2 · `tsfm-lens doctor` — preflight that runs automatically.** ✅
  **Implemented and live-verified 2026-08-11 (twenty-second cron-loop
  firing) — see §14's entry for the full writeup, not repeated here.** One
  command (and a default-on preflight at the start of every run, `--no-preflight`
  to skip) that checks: load-bearing pins vs. `DEPENDENCIES.md` (**zarr<3**
  first, given four recurrences), torch/CUDA availability and free VRAM vs. the
  config's estimated peak (attention's `[B,H,T,T]` and per-window patching are
  the known blowups — `CLAUDE.md` §6.5), disk headroom for the store, corpus
  seal verification, `context_len` divisibility by `alignment.window`, every
  stage's `max_series` vs. every model's `batch_size` (A16), adapter
  conformance (`models/conformance.py` — already built) and the alignment gate
  (A2) per model. Output a table of pass/warn/fail with the exact remediation
  command. This single item would have absorbed most of `CLAUDE.md` §11's
  environment traps. Depends on: A2, A7, A14.
- [ ] **E3 · Empirical token→time span discovery, and a generic HF adapter.**
  The largest generality lever in the repo. Today `token_time_spans()` is
  *declared* by an adapter author and verified by a check nobody is forced to
  run; the mapping is the one thing cross-architecture comparison rests on
  (`CLAUDE.md` §6.3). Instead **measure** it: sweep a small relative-amplitude
  impulse across every timestep (or every window, then refine), record which
  token position moves most in the first captured block, and *derive* the
  span map. The machinery exists — `impulse_alignment_check` is this
  algorithm with the answer thrown away. Deliverables: (a) `--discover-spans`
  producing a span table plus a diffuseness score; (b) a
  `GenericHFAdapter` that combines `--discover-layers`' shape-probing for
  `[B, T, D]` residual candidates with discovered spans, so a new checkpoint
  can be *attempted* with zero code; (c) an explicit refusal path — a model
  whose impulse response is diffuse (spectral tokenizers, latent-query
  models) is not time-localized, so it degrades to **L0 only** with a stated
  reason, exactly the envelope edge `CLAUDE.md` §12 already draws; (d) when a
  hand-written adapter declares spans, cross-check declared vs. discovered and
  fail on disagreement. This is also a genuinely publishable methodological
  contribution: adapter correctness becomes measured rather than asserted.
  Depends on: A2. Enables: §9 Phase 4 at a fraction of the per-model cost.
- [ ] **E4 · A bundled, versioned reference corpus** — **parked (§22.3).** Ship
  or fetch-on-first-run a small fixed corpus with a pinned digest, so two users'
  numbers are comparable. Carries an unresolved licensing decision, and E1
  depends on it.
- [ ] **E5 · A results registry** — **parked (§22.3).** Append-only history and
  provenance keying across runs, so adding a model appends to a table rather
  than requiring an explicit `--runs a,b,c` list.
- [x] **E6 · Report as a shareable product.** ✅ **DONE 2026-08-18**, against
  the item's own stated acceptance criterion (below), not against "the rest is
  cheap" list in full — confidence badges and the E5-gated diff mode are
  explicitly deferred, not forgotten (see Findings). A5's coverage panel and
  A7's provenance panel were already built and rendered before this item.
- [ ] **E7 · Docs, quickstart, notebooks, packaging** — split three ways, and
  the install-story *decision* is already made (`CLAUDE.md` §3, 2026-08-11: two
  packages, one repo, deliberately separate because their dependency sets barely
  overlap). **E7a** (console script) is `[x]`. **E7b** (install docs) is live
  and roughly an hour. **E7c** (quickstart + notebooks) is parked with E1
  (§22.3) — a quickstart written against the current five-command path would be
  rewritten by E1.
- [x] **E8 · CI.** = A14. Listed here too because "anyone can use it" is not
  credible without it. **2026-08-11: this is a pure duplicate pointer, not a
  distinct piece of work — A14 was already fixed and confirmed 2026-08-06**
  (full `.github/workflows/ci.yml` with `fast`/`pinned`/`bleeding` jobs; see
  A14's own Findings for the exact numbers). Marking `[x]` here too rather
  than leaving it as a second, permanently-open-looking checkbox for
  something already done — no new code, this is a bookkeeping fix.

### T2 — Rigor a comparison tool must have

- [~] **E9 · Untrained-weights and permutation controls as first-class
  comparison targets.** The standard null in representational-similarity work
  is a randomly-initialized model of the same architecture, and this repo
  doesn't have one. Add a `random_init: true` flag on a model config that
  loads the architecture with fresh weights, then run L1/L2/internals/probe/
  SAE against it exactly like a real model. This gives, for free: a floor for
  CKA (how much geometry is architecture and input statistics rather than
  learning), a floor for L2 stitching gain (§6.5's input-feature baseline is
  good, but an untrained-network baseline is stronger and answers a different
  objection), a floor for probe decodability, and a floor for SAE
  ground-truth alignment mass (how much of §6.2's alignment is real structure
  vs. what any random projection of a structured input yields). Add
  label-permutation nulls for every probe. Cheap, and it upgrades several
  existing results from "clearly non-zero" to "clearly above the null a
  skeptic would name." Pairs with A13's noise floor. **This is arguably the
  highest-value item in T2** — §2.2's own doctrine demands nulls, and the
  most obvious null for internals work is currently absent.
  — **2026-08-06: `random_init` implemented and unit-tested; real-checkpoint
  verification launched, not yet confirmed at time of writing — see this
  section's Findings block below for status, run directories, and (once
  complete) the actual floor numbers.** The label-permutation-null half is
  implemented for one probe (SAE ground-truth alignment, the one this item's
  own text names as "SAE ground-truth alignment mass") via
  `sae/ground_truth.py::permutation_null_alignment`; internals' family-probe
  decodability does not yet have its own permutation null (it already has a
  majority-class chance-line CI, a weaker but related control) — noted as a
  remaining follow-up, not silently folded into "done."
- [x] **E10 · Probabilistic-forecast diagnostics.** Quantiles are already
  produced and pinball loss already computed, but nothing checks
  *calibration*: PIT histograms and interval coverage per family and per
  horizon step, sharpness-vs-calibration scatter, quantile-crossing counts.
  For anyone choosing between TSFMs in practice this is a first-order
  question, it's behaviorally cheap (no new forward passes beyond what L0
  does), and it's a whole axis on which TimesFM's deterministic head and
  Chronos's sampled one differ in kind — which makes it a natural companion
  to `CLAUDE.md` §12's forecast-stochasticity-asymmetry caveat rather than
  another victim of it.
  — **✅ DONE 2026-08-10.** New `analysis/calibration.py`: a reliability
  curve (empirical vs. nominal coverage per quantile level, overall and per
  family), an approximate PIT (probability-integral-transform) histogram
  via per-position linear interpolation against each model's own discrete
  quantile levels, a quantile-crossing-rate check (a real forecast-head
  defect, independent of calibration), and outer-interval coverage +
  sharpness resolved per horizon step. All of it is a pure reduction over
  the `quantiles`/`targets` arrays `run_l0` already holds in memory after
  calling `predict()` — **zero new forward passes**, exactly as the item
  scoped it. Wired in via a new `L0Config.calibration: bool = True` flag
  (default on, silently skipped if fewer than 2 quantile levels are
  configured) and a new `l0/calibration.json` artifact
  (`l0_behavioral.py::run_l0`). Report gets a new `_calibration_block` in
  the L0 section (`report/report.py`): a reliability-curve plot, a PIT
  histogram, and a coverage/sharpness/crossing-rate table, each with a
  `_note()` purpose/reading/limitations block, plus a per-model findings
  string (`"L0 calibration — {model}: max reliability-curve gap …, outer-
  interval coverage … (nominal …), quantile-crossing rate …."`). **Verified,
  not just implemented:** 7 new synthetic-planted-answer unit tests
  (`tests/test_calibration.py`) — a well-calibrated Uniform(0,1) forecaster
  (calibrated by construction, since Uniform(0,1)'s quantile function is
  the identity) confirms the reliability curve and outer-interval coverage
  both land within 0.02 of nominal; a deliberately overconfident
  (too-narrow) forecaster shows empirical coverage far below its nominal
  claim (<0.2 vs. nominal 0.8); PIT values recover the planted identity
  mapping exactly for in-range targets; the quantile-crossing check reads
  exactly 0.0 on monotonic input and finds a single hand-planted violation
  at the exact rate `1/(n·h)` expected; horizon-resolved coverage/sharpness
  and per-family breakdown both return the right shapes/counts. All 7
  passed on first run. Live end-to-end check: ran `configs/smoke.yaml`
  (both mock architectures) and confirmed `l0/calibration.json` is written
  with real content and the report renders the new "Quantile calibration" /
  "PIT histogram" / "Interval coverage, sharpness and quantile crossing"
  sections with correctly-populated findings text — not just a passing
  pytest suite. Full `tsfm_lens` suite green at **215/215** (208 prior + 7
  new), same two pre-existing benign warnings as before, zero regressions.
- [~] **E11 · Statistical-discipline hardening.** A15's registry and
  multiplicity ledger, plus: report effect sizes with CIs everywhere a
  p-value appears (mostly done — audit for exceptions), state `n_boot` and
  the p-value floor inline where a p is shown (`CLAUDE.md` §6.6 documents the
  1/n_boot floor; the report should say it at the point of use), and make the
  cluster-bootstrap unit explicit in every artifact so a reader can check
  that the series-level rule (invariant 2) was actually followed.
  — **Partial progress 2026-08-10.** Audited every `p=` rendering in
  `report.py`: only two exist (L0's overall paired ΔMASE, CONFIRM's overall
  paired ΔMASE on private data), and both already pair the p-value with a
  `_ci_str`-rendered CI right next to it — the "report effect sizes with CIs
  everywhere a p-value appears" bullet was already satisfied, not merely
  "mostly." The registry/multiplicity-ledger bullet is also already done —
  it's A15's `hypotheses.json`, fixed 2026-08-06, this item just hadn't been
  updated to say so. What was actually missing and is now fixed: neither
  `p=` site, nor the family-level paired-tests table header, stated `n_boot`
  or the 1/n_boot floor at all — a reader had no way to tell a "genuinely
  tiny" p from a floored one without cross-referencing `CLAUDE.md` §6.6.
  `analysis/stats.py::paired_bootstrap` now returns `n_boot` in its result
  dict (`bootstrap_ci_diff` already did); `report.py` gained a small
  `_p_note(d)` helper rendering `" (n_boot=N, p floored at 1/n_boot=F)"` and
  wired it into all three sites (L0's family-tests table header, L0's
  overall-test finding, CONFIRM's overall-test blurb). Verified live, not
  just unit-tested: reran `configs/smoke.yaml`'s `l0,report` stages and
  confirmed the rendered HTML actually contains
  `"n_boot=150, p floored at 1/n_boot=0.0067"` next to the real p-value.
  Full `tsfm_lens` suite green at **215 passed, 2 warnings** (same two
  pre-existing benign warnings, zero regressions) after the change. **Not
  done:** the cluster-bootstrap-unit-explicit bullet (recording per-artifact
  that a given CI/p resampled series, not windows) — untouched this
  session, still open.

### T3 — Analysis capability a TSFM interpretability user will ask for

- [x] **E12 · Horizon-resolved everything.** Every current metric aggregates
  over the whole forecast horizon, so "where does the error come from at h=1
  vs h=64", "which layers matter for long-horizon vs short-horizon", and
  "does the forecast crystallize later in depth for later horizon steps" are
  all invisible. Concretely: MASE/pinball per horizon step (free); per-window
  patching restoration resolved by horizon step (the arrays largely exist —
  it's a reduction that's currently collapsed); crystallization depth as a
  function of horizon step. This is the most TSFM-specific analysis axis
  missing, it's cheap because the forward passes are already being paid for,
  and it directly sharpens the §1 questions about high-frequency vs.
  statistical-mean behavior.

  **Partial progress 2026-08-10 — first sub-part (MASE/pinball per horizon
  step) implemented and verified; the other two are still open.** New
  `analysis/stats.py::mase_pinball_by_horizon()` mirrors `mase()`'s exact
  per-series scale (`_mase_scale`) but keeps the horizon axis instead of
  reducing over it, returning `[n_series, horizon]` MASE and pinball arrays;
  `mase_pinball_by_horizon(...).mean(axis=1)` is verified identical to the
  existing whole-horizon `mase()` to `atol=1e-9` (a consistency test, not
  just a new formula taken on faith). New `L0Config.horizon_resolved: bool =
  True` (mirrors the `calibration` flag's pattern exactly);
  `l0_behavioral.py` computes pooled + per-family MASE/pinball curves over
  horizon step per model (`_summarize_by_horizon`) whenever `data.horizon >
  1`, written to a new `l0/horizon_resolved.json` artifact. `report.py`
  gained `_horizon_resolved_block()` (two plotly line charts — MASE by
  horizon step, pinball by horizon step, one line per model — with the same
  `_note()` purpose/reading/limitations pattern as every other report
  figure), wired into `_sec_l0` right after the existing calibration block,
  plus one new findings string per model
  (`"L0 horizon profile — {model}: MASE {h1} at h=1 vs {hN} at h={N} (ratio
  {r}x)"`). New `tests/test_horizon_resolved.py` (4 tests, all synthetic
  with a planted, known-correct answer per `CLAUDE.md`'s testing
  convention): a linearly-growing-with-horizon planted error gives a
  monotonically increasing MASE curve and a growing pinball curve; the
  per-series mean over horizon matches whole-horizon `mase()` exactly; a
  constant-offset error gives a flat curve. Verified live end-to-end, not
  just unit-tested: reran `configs/smoke.yaml`'s `l0,report` stages and
  confirmed `l0/horizon_resolved.json` has the expected `[n_series, 32]`-length
  arrays for both mock models, and that the rendered HTML actually contains
  the new findings strings verbatim (e.g. `"L0 horizon profile — patchy:
  MASE 2.731 at h=1 vs 4.042 at h=32 (ratio 1.48x)"`,
  `"...steppy: MASE 5.270 at h=1 vs 5.584 at h=32 (ratio 1.06x)"`) and the
  new chart titles ("MASE by horizon step" / pinball equivalent) — findings
  count rose from 27 to 29 (exactly the 2 new per-model findings expected),
  sections stayed at 11. Full `tsfm_lens` suite green at **219 passed, 2
  warnings** (215 prior + the 4 new tests, same two pre-existing benign
  warnings, zero regressions). **Not done, left `[~]` rather than `[x]` for
  this reason:** the other two named sub-parts — per-window L3 patching
  restoration resolved by horizon step (the arrays already exist per the
  item's own text, this needs only the reduction step) and Lens
  crystallization depth as a function of horizon step — are both
  untouched this session.

  **Second sub-part done, 2026-08-10 (twelfth cron-loop firing, same day) —
  L3 per-window patching restoration resolved by horizon step.** Confirmed
  the item's own claim was right: `_window_restoration`/`_patching`
  (`analysis/l3_perturbation.py`) already compute `f_patch`/`f_clean` per
  horizon step at every (layer, window) cell, and only `.mean(axis=1)`
  discarded the horizon axis before this — a pure reduction, no new forward
  passes. Pulled the one-liner out into a standalone, directly testable
  `restoration_by_horizon(f_patch, f_clean, damage_h)` (used at both the
  per-window and whole-context-patch call sites, so the fallback path gets
  the same treatment) instead of inlining it twice. `_patching` now also
  computes a per-horizon damage denominator (`damage_h`, mirroring the
  existing scalar `damage`) and accumulates a `[n_corruptions, n_layers,
  horizon]` array (averaged over windows, the same way the existing
  `restoration` curve is already averaged over windows) into a new
  `restoration_by_horizon` output key, gated by a new
  `PatchingConfig.horizon_resolved: bool = True` flag (`config.py`) —
  mirrors `L0Config.horizon_resolved`'s naming exactly. Saved into
  `l3/patching.npz` as `restoration_by_horizon_{model}`. `report.py` gained
  `_l3_horizon_heatmaps()` (a [relative depth x horizon step] heatmap per
  corruption, one per model, immediately below the existing per-window
  depth-x-time heatmap) plus a findings string whenever a corruption's
  best-restoring layer differs between horizon step 1 and the final step.
  New `tests/test_l3_horizon_resolved.py` (4 synthetic-planted-answer
  tests, exercising the extracted `restoration_by_horizon` function
  directly, independent of any adapter/forward pass): a perfect patch
  restores fully at every horizon step; a patch that drifts further from
  clean at later horizon steps by construction gives a monotonically
  decreasing restoration curve; a patch identical to the corrupted input
  gives exactly zero restoration everywhere; output shape matches the
  horizon axis, not the batch axis. Also extended
  `tests/test_smoke.py::test_per_window_and_lens_artifacts` with a shape/
  finiteness check on the new `restoration_by_horizon_{model}` array.
  Verified live end-to-end, not just unit-tested: reran
  `configs/smoke.yaml`'s `l3,report` stages and confirmed
  `restoration_by_horizon_{model}` arrays are finite with the expected
  `[corruptions, layers, horizon]` shape for both mock models, the new
  "per-horizon-step restoration" heatmap section and its `_note()` block
  render, and the new findings text appears verbatim (e.g. `"L3
  horizon-resolved patching — patchy/deseasonalize: the layer that best
  restores horizon step 1 (0.80 relative depth) differs from the layer
  that best restores the final horizon step (0.20)."`) — findings count
  rose from 29 to 39 (up to 6 corruptions x 2 models = 12 possible new
  findings; 10 fired, 2 corruption/model pairs happened to share the same
  best layer at both horizon extremes). Full `tsfm_lens` suite green at
  **223 passed, 2 warnings** (219 prior + the 4 new tests, same two
  pre-existing benign warnings, zero regressions). **Still open at that
  point:** only the Lens crystallization-depth-by-horizon sub-part
  remained from this item's original three.

  **Third and final sub-part done, 2026-08-11 (fourteenth cron-loop
  firing, same UTC evening) — Lens crystallization depth resolved by
  horizon step, closing E12 fully.** `_model_lens`
  (`analysis/lens.py`) already computes `lens_fc`/`final_fc` as
  `[n_layers, B, horizon]`/`[B, horizon]` arrays before collapsing the
  horizon axis with `.mean(axis=2)` for the existing whole-horizon-averaged
  curve — the same "arrays already exist, only the reduction was
  collapsed" situation as the L3 sub-part above, no new forward passes.
  Added `per_series_h` (keeps the horizon axis) and `final_mase_h`
  alongside the existing scalar versions, gated by a new
  `LensConfig.horizon_resolved: bool = True` flag (mirrors
  `PatchingConfig.horizon_resolved`'s naming exactly). Extracted the
  existing inline crystallization-crossing logic (previously duplicated
  as a one-off `next(...)` loop) into a standalone, directly-testable
  `crystallization_depths(mase_curve, final_mase, tol, depths)` that
  handles both the original 1-D (whole-horizon) case and a new 2-D
  (per-horizon-step) case with one shared implementation — verified this
  refactor is byte-for-byte behavior-preserving for the existing scalar
  path via a dedicated consistency test (single-horizon-step 2-D input
  must equal the 1-D scalar result). New `skip_mase_by_horizon` array
  (`[n_layers, horizon]`) saved to `lens/curves.npz`, and a new
  `crystallization_depth_by_horizon` list (length = horizon, `null` where
  unresolved) saved to `lens/lens.json`. `report.py` gained
  `_lens_horizon_block()`: a [relative depth x horizon step] MASE heatmap
  per model plus a crystallization-depth-vs-horizon-step line chart (one
  line per model), both with `_note()` blocks, plus a findings string
  comparing each model's horizon-step-1 vs. final-horizon-step
  crystallization depth. New `tests/test_lens_horizon_resolved.py` (4
  synthetic-planted-answer tests against `crystallization_depths`
  directly, independent of any adapter/forward pass): the scalar case
  matches a hand-computed crossing index; a curve that never crosses
  returns `None`; three independently-planted per-horizon-step columns
  (crosses early / crosses late / never crosses) all resolve to their
  correct, distinct answers in one call; and the 2-D/1-D consistency
  check above. Also extended
  `tests/test_smoke.py::test_per_window_and_lens_artifacts` with a shape/
  finiteness/list-length check on the new Lens artifacts. Verified live
  end-to-end: reran `configs/smoke.yaml`'s `lens,report` stages and
  confirmed `skip_mase_by_horizon_{model}` arrays are finite with shape
  `[n_layers, 32]` for both mock models, `crystallization_depth_by_horizon`
  is a real 32-entry list for each, and the rendered HTML contains both
  new chart titles ("skip-lens MASE by horizon step", "Crystallization
  depth by horizon step") and the new findings text verbatim (e.g.
  `"Lens horizon-resolved crystallization — patchy: horizon step 1
  crystallizes at 0.60 relative depth vs. 0.00 at the final step (32)."`)
  — findings count rose from 39 to 41 (exactly the 2 new per-model
  findings expected), sections stayed at 11. Full `tsfm_lens` suite green
  at **227 passed, 2 warnings** (223 prior + the 4 new tests, same two
  pre-existing benign warnings, zero regressions). **E12 marked `[x]`** —
  all three named sub-parts (L0 MASE/pinball, L3 per-window patching, Lens
  crystallization depth) are now implemented, tested, and live-verified,
  all resolved by horizon step with zero additional forward passes beyond
  what each stage already pays for.
- [x] **E13 · Spectral lens.** — **implemented + unit-tested 2026-08-11
  (twentieth cron-loop firing)**: `analysis/spectral_lens.py`,
  `run_spectral_lens.py`, `tests/test_spectral_lens.py` (3/3 passing on
  synthetic planted data). The time-domain skip lens says *when* the
  forecast crystallizes; a frequency-domain version says *what* crystallizes
  first. Per layer, compare the skip-lens forecast's spectrum against the
  target's, resolved by frequency band, and check band-wise error against the
  ground-truth seasonal periods the benchmark records exactly. Answers "does
  the model get the trend right early and the seasonality late (or the
  reverse)", is native to time series in a way the borrowed logit-lens idea
  isn't, reuses the existing lens forwards, and feeds §6.2's
  frequency-aware-dictionary idea with the evidence it needs to be worth
  trying.
  — **2026-08-11, same-day follow-up: the background live-checkpoint
  verification landed — zero bugs, zero code changes needed.** Ran
  `run_spectral_lens.py --run runs/medium_run_chronos_base --max-series 48`
  end to end against real TimesFM-2.5-200M / Chronos-T5-Base activations
  (`cudaPy` env — this machine has no `tsfmPy` env, contrary to what
  `CLAUDE.md`/`DEPENDENCIES.md` assume from a different machine's
  perspective; noted here since it's the third session this loop has hit
  that same stale assumption). Exit code 0, no NaN/inf, no crashes; every
  risk area named in the verification brief (ground-truth table indexing,
  `ActivationStore.layers()`, `skip_lens_forecasts`'s shape contract,
  `data.meta` columns, internal shape mismatches) checked out clean against
  the real run. **Coverage actually realized:** `--max-series 48` was
  capped by `capped_take` to **24** (Chronos-T5-Base's `batch_size: 24`,
  logged as `sample cap reduced: requested 48, realized 24 (limited by
  batch_size)`), family-stratified as `{'mixture': 5, 'parametric': 3,
  'random_parametric': 16}` out of population `{'mixture': 60, 'parametric':
  40, 'random_parametric': 188}`. Of those 24, 11 series had a finite
  ground-truth `seasonal_period_dominant`; one dropped out per-model because
  its period's FFT bin fell outside `(0, Nyquist)` for horizon=64 (the
  module's documented degrade rule, working exactly as designed — not a
  bug), leaving **`n_series_with_period=10`** for both models' band
  computations. **Exact numbers** (`tol=0.1`, the run's own
  `lens.crystallization_tol`):

  | model | trend depth | seasonal depth | residual depth | final trend err | final seasonal err | final residual err |
  |---|---|---|---|---|---|---|
  | TimesFM (10 layers) | 0.1111111111111111 | None | 0.0 | 0.1150452271103859 | 0.02738974429666996 | 0.510842502117157 |
  | Chronos-T5-Base (12 layers) | 0.9090909090909091 | 1.0 | 0.0 | 0.14323998987674713 | 0.04245809093117714 | 0.5051148533821106 |

  **Reading.** TimesFM's trend crystallizes almost immediately (depth
  0.111) while its seasonal band never crystallizes within `tol` across all
  10 captured layers — `seasonal_crystallization_depth=None` because the
  deepest layer's seasonal error (0.0567, from the raw
  `seasonal_error_curve`) is still ~2.07× the final-layer error (0.0274),
  outside the 1.1× band. Chronos-T5-Base crystallizes *both* bands very
  late and in a compressed near-final burst — trend at depth 0.909, seasonal
  at the very last layer (depth 1.0) — with seasonal only fractionally
  lagging trend rather than the wide TimesFM-style gap. Both models'
  residual band crystallizes trivially at depth 0.0, but this reads as a
  flat/no-real-depth-trend artifact (TimesFM's residual error sits in a
  narrow 0.446–0.527 band across all layers, Chronos's in 0.496–0.553) —
  layer 0 already lands within `tol` of final purely because there's no
  real depth trend to resolve, not genuine early convergence. Read together
  with `CLAUDE.md` §14's existing "TimesFM front-loads then compresses" vs.
  "Chronos accumulates" framing, this is a new, concrete, frequency-domain
  confirmation of that same asymmetry on real checkpoints for the first
  time, sharpened to a specific claim the time-domain lens couldn't make on
  its own: TimesFM gets the trend right early and never fully closes the
  seasonal gap within its captured depth, while Chronos gets both right
  only in a late, compressed burst near its final layer. **Caveat, not
  swept under the rug:** this reading rests on only 10 usable series per
  model (24 sampled, batch_size-capped, further thinned by the seasonal
  ground-truth filter) — before this becomes a number anyone cites
  elsewhere, rerun with either an explicit `batch_size` override on
  Chronos-T5-Base or a `--max-series` chosen to already respect its cap, to
  get a larger `n_series_with_period` and a real sense of how stable
  depth=0.909/1.0/None actually are under resampling. Full per-layer curves
  in `runs/spectral_lens.json` (gitignored, not committed). Full
  `tsfm_lens` test suite (including `test_spectral_lens.py`, re-run this
  same firing) still green, no regressions. **E13 marked `[x]`** — the
  live-checkpoint verification this item was left open pending has now
  landed clean; the small-sample caveat above is a follow-up refinement,
  not an open implementation gap.
- [~] **E14 · Steering / directional control.** `transformer-lens` users
  expect to *intervene*, not only observe. `token_patch` already supports
  everything needed: add `steer(layer, windows, direction, strength)` and
  measure the forecast response. The validation loop is the repo's unfair
  advantage (§2.1): take a direction associated with a known ground-truth
  factor (an SAE feature matched to `n_seasonalities`, or a probe direction
  for trend order), add it, and check whether the forecast changes *in the
  predicted direction* — spectrum shifts toward the period, trend slope
  increases. That is a far stronger claim than "this feature correlates with
  X", and it's the natural payoff of the SAE work rather than a new research
  program. Depends on: E15 for feature-level directions (probe directions
  work today).

  > **Implemented 2026-08-11 (code + synthetic unit tests; live-checkpoint
  > verification launched in background, not yet landed).** New
  > `tsfm_lens/analysis/steering.py`: pure-numpy `trend_slope(x)` (closed-
  > form OLS slope per row) and `seasonal_band_magnitude(x, periods)` (FFT
  > magnitude at each series' own ground-truth period bin, `np.nan` when
  > absent/out-of-range, mirroring `spectral_lens.py`'s degrade-gracefully
  > convention), plus `predicted_direction_metric(gt_field)` and
  > `evaluate_direction_match(rho, response_up, response_down)`. Per
  > `CLAUDE.md` §2.5, only two ground-truth fields get an unambiguous
  > "increasing this field should increase that scalar metric" mapping and
  > therefore a directional claim: `trend_scale` → trend slope,
  > `seasonal_amplitude_max` → seasonal-band magnitude at that series' own
  > `seasonal_period_dominant`. Every other matched field (`trend_order`,
  > `n_seasonalities`, `seasonal_period_dominant` itself, AR/changepoint/
  > anomaly/`has_*` flags) still gets a MASE-disruption number but
  > `predicted_direction_metric` returns `None` for it rather than forcing
  > a directional claim the field doesn't actually support.
  >
  > New `sae/eval.py::_feature_steered_replacement`/
  > `feature_steering_effects`, siblings of E15's own
  > `_feature_ablated_replacement`/`feature_ablation_effects`: for each
  > candidate feature (same top-|ρ| ground-truth-matched candidates E15's
  > ablation already selects), adds a signed `±2σ` delta (σ = that
  > feature's own clean-activation std) to the post-encode activation
  > instead of zeroing it, patches the resulting reconstruction back in via
  > `token_patch` at **token** granularity (same reasoning as E15: window-
  > broadcast would blur the exact fine-grained signal this test depends
  > on), and reports both the MASE disruption vs. the full-reconstruction
  > baseline and the raw `trend_response`/`seasonal_response` deltas for
  > both "up" and "down" steering. `feature_steering_effects` itself stays
  > agnostic of `best_field`/`rho` (mirroring how `feature_ablation_
  > effects` stays agnostic of `best_field`) — the directional verdict is
  > computed by the caller via `evaluate_direction_match`, which already
  > has those fields from `ground_truth_alignment`.
  >
  > Wired into `sae/train.py::run_sae` behind a new
  > `cfg.sae.feature_steering_enabled` flag (`config.py`, default `False`,
  > mirroring `feature_ablation_enabled`'s wiring exactly down to the same
  > try/except-log-and-continue pattern), plus `feature_steering_top_k`
  > (default 8), `feature_steering_max_series` (default 64), and
  > `feature_steering_strength_sigma` (default 2.0). Ground-truth seasonal
  > periods are loaded once per `run_sae` call via
  > `ground_truth.py::load_ground_truth_table` (the same table
  > `run_spectral_lens.py`/E13 already uses) and reused across every
  > (model, layer) target, mirroring `_sample_real_contexts`'s existing
  > once-per-call caching. Result stored as `results[key]["feature_
  > steering"]` in `sae/meta.json`, each feature entry carrying `best_field`,
  > `rho`, `predicted_metric`, and `direction_match` (`{predicted_sign,
  > matched_up, matched_down}`, each of the latter two `True`/`False`/`None`
  > — `None` when the observed response was too small to call, never a
  > forced verdict).
  >
  > New `tests/test_steering.py` (8 synthetic planted-answer tests, all
  > passing): `trend_slope` recovers a planted linear coefficient exactly
  > and stays near-zero under a pure (zero-net-trend) seasonal signal;
  > `seasonal_band_magnitude` recovers a planted sinusoid's FFT amplitude
  > (`≈ a·T/2` for amplitude `a` over `T` samples) and degrades to `NaN` for
  > a missing or out-of-range period; `predicted_direction_metric` scopes
  > correctly to just the two directional fields; `evaluate_direction_match`
  > confirms a correctly-signed response, flags a wrongly-signed one, and
  > returns `None` (not a forced verdict) for a negligible response. One
  > real test-construction bug found and fixed while writing these (same
  > class of mistake E13's own test-construction bug was): a discretely-
  > sampled sinusoid over a finite window isn't *exactly* orthogonal to a
  > linear ramp even at an integer number of periods, so the original
  > `< 1e-6` "pure seasonal has zero trend slope" assertion failed at
  > `0.0189` — loosened to `< 0.05`, still tiny relative to the seasonal
  > amplitude (3.0) and to the sibling test's planted slopes (0.5/-1.2), and
  > confirmed the *real* signal (a planted trend coefficient) still recovers
  > to 8 decimal places in the adjacent test. Full `tsfm_lens` suite
  > re-run after these changes: **248 passed** (0 failed), 2 pre-existing
  > warnings unrelated to this change (an `extract.py` float16-cast overflow
  > warning already present before this session, and a `test_smoke.py`
  > pytest-return-value style warning, also pre-existing) — no regressions.
  > **Not yet done:** live-checkpoint verification against a real SAE
  > checkpoint. A background agent for this is running as of this entry
  > against a new `configs/medium_run_chronos_base_feature_steering_check.yaml`
  > (copy of the existing `..._feature_ablation_check.yaml`, reusing
  > `medium_run_chronos_base`'s already-extracted activations, `--stages
  > sae,report` only); results follow in a later entry once it reports back.
  > (First attempt at launching this agent used `isolation: "worktree"`,
  > which was wrong and caught before any wasted GPU time: `runs/` is
  > git-ignored, so a fresh worktree has no `activations.zarr` to reuse and
  > the step-2 copy would have failed — killed and relaunched without
  > worktree isolation, directly against this working copy.)
  >
  > **Live-checkpoint verification landed 2026-08-11, same day.** The
  > relaunched run completed cleanly (exit code 0, 5m14s, `sae: complete, 2
  > target(s)` / `pipeline finished`) against real
  > `google/timesfm-2.5-200m-pytorch` (`stacked_xf.18`) and
  > `amazon/chronos-t5-base` (`encoder.block.6`) SAE checkpoints on
  > `benchmark_medium/public_dev`. Read `sae/meta.json` directly rather than
  > trusting the launching agent's own summary — it died mid-report to an
  > unrelated session-limit error, but the underlying job it started had
  > already finished successfully and left real results on disk. Mechanism
  > verified working end-to-end, no crash, sensible-scaled numbers: TimesFM
  > (`n_realized=32/32`) `mase_clean=1.741` → `mase_full_reconstruction=1.862`
  > (SAE reconstruction alone costs some fidelity, as already known from
  > E15/E17), per-feature `steering_sigma` 1.3–4.8 and `mase_delta_vs_full_
  > recon` ranging −0.012 to +0.052 across the 8 candidates; Chronos-T5-Base
  > (`n_realized=24/32`, `limited_by: ["batch_size"]` — expected, its
  > `batch_size: 24` in this config) `mase_clean=2.110` →
  > `mase_full_reconstruction=3.319`, `steering_sigma` 12.9–25.6 (correctly
  > tracking Chronos's much larger raw activation scale relative to
  > TimesFM's, exactly the kind of per-model scale-adaptation E19's
  > crosscoder work already had to solve for) and `mase_delta_vs_full_recon`
  > ranging −0.252 to +0.569.
  >
  > **A real, honestly-reportable gap surfaced by this run, not a bug:**
  > `direction_match` came back `None` for **all 16** tested features
  > (8 per model) — not because `evaluate_direction_match` failed (its own
  > 8 synthetic unit tests above already confirm that logic directly), but
  > because every one of the 16 top-|ρ| ground-truth matches on this real
  > corpus/layer pair landed on `tier_realism_stress` (7 of 8 TimesFM
  > features, 8 of 8 Chronos features) or `n_seasonalities` (1 TimesFM
  > feature) — a categorical tier flag and an integer count, neither of
  > which `predicted_direction_metric` maps to a directional claim by
  > design (§2.5: only `trend_scale`/`seasonal_amplitude_max` get one, per
  > this entry's own implementation note above). So the plumbing that
  > *would* render a directional verdict is fully exercised and produces
  > real MASE/trend/seasonal numbers, but the specific claim E14 exists to
  > test — "steering a feature matched to a continuous ground-truth
  > quantity moves the forecast in the predicted direction" — has zero
  > evaluable examples in this particular run, simply because
  > `ground_truth_alignment`'s own top-|ρ| ranking didn't surface a
  > `trend_scale`/`seasonal_amplitude_max`-matched feature in either
  > model's top 8 here. **Named next step, not attempted this session:**
  > either widen the live check's feature selection to specifically
  > include the best `trend_scale`/`seasonal_amplitude_max` matches
  > (regardless of whether they're globally top-8 by |ρ|) so the directional
  > claim actually gets tested at least once per model, or accept that on
  > this corpus those two fields are rarely any feature's *best* match and
  > report the null result as-is.
  >
  > **Widening implemented and re-verified 2026-08-11, same day (twenty-
  > second cron-loop firing).** Took the first named option: `sae/train.py`'s
  > `run_sae` now explicitly appends each field's own single best-|ρ| match
  > (`trend_scale`, `seasonal_amplitude_max`) to the steering candidate list
  > whenever `ground_truth_alignment` found one at all and it wasn't already
  > in the top-K set, logging `sae: feature-steering candidates for {key}
  > widened with directional-field match(es): [...]` when it fires. Ablation's
  > separate candidate-selection code path was deliberately left untouched.
  > Full `tsfm_lens` suite re-run: **262 passed, 0 failed**, same 2
  > pre-existing warnings — no regressions (see §14's twenty-second firing
  > entries for how a stale "248 passed" figure from an earlier check this
  > same firing was caught and corrected).
  >
  > **Re-verified live against the same real corpus — the widening logic
  > correctly did not fire, and the underlying null result persists.** A
  > background agent reran `configs/medium_run_chronos_base_feature_steering_
  > check.yaml --stages sae,report --force sae` (reusing the existing
  > `activations.zarr`). Found and fixed one real, pre-existing, unrelated
  > config bug along the way: that config's `feature_ablation_max_series`/
  > `feature_steering_max_series` were both `32`, exceeding Chronos-T5-Base's
  > `batch_size: 24` — a violation of the documented "every `max_series` must
  > be ≤ every model's `batch_size`" invariant that the new `doctor.py`
  > preflight (E2, same firing) caught immediately at startup; fixed to `24`
  > to unblock the run. After the fix, the run completed cleanly (~5 min,
  > `sae: complete, 2 target(s)`, `pipeline finished`). Read `sae/meta.json`
  > directly: grepping the full stdout for the new widening log line found
  > **zero matches** — confirmed why by reading the stored top-50
  > `ground_truth_alignment.features` list for both targets directly.
  > **Neither `trend_scale` nor `seasonal_amplitude_max` appears as any
  > feature's `best_field` anywhere in either target's top-50** (TimesFM/
  > `stacked_xf.18`'s 50 matched fields are drawn from `{ar_coeff_sum,
  > archetype_ar_colored_noise, archetype_multi_seasonal_complex,
  > archetype_random_walk_drift, archetype_seasonal_dominant,
  > archetype_trend_dominant, generator_parametric, has_heteroskedastic,
  > has_random_walk, n_seasonalities, tier_realism_stress}`; Chronos-T5-Base/
  > `encoder.block.6`'s from `{archetype_intermittent_bursts,
  > archetype_trend_dominant, generator_random_parametric,
  > n_seasonalities, noise_scale, seasonal_period_dominant,
  > tier_realism_stress}`) — so the widening code's own `field_matches`
  > list was correctly empty for both fields on both targets, and (correctly,
  > per its own logic) appended nothing and logged nothing. All 16
  > `feature_steering.features` entries (8 per target, unchanged from the
  > pre-widening run) still have `predicted_metric: null` / `direction_match:
  > null` — the same null result as before, now confirmed to persist even
  > after the fix specifically designed to rescue it.
  >
  > **A real scope limitation surfaced by this result, not previously
  > stated:** the widening code operates on `gt.get("features", [])`, which
  > is *already* `ground_truth_alignment`'s own global top-50-by-`|rho|`
  > truncation (`sae/ground_truth.py`'s `sorted(...)[:50]`), not the full
  > per-feature match list across all features (10240 for TimesFM,
  > 6144 for Chronos in this run). So the widening logic can only ever
  > rescue a `trend_scale`/`seasonal_amplitude_max` match that exists
  > somewhere in that top-50 but outside the steering stage's own top-K
  > (`feature_steering_top_k: 8`) — it cannot reach a feature whose best
  > match to either field exists only outside the top-50 overall, which is
  > exactly the case in this run (the fields don't appear anywhere in the
  > top-50 at all). This run is therefore a **true negative on an untested
  > code path**, not a confirmed pass or a disproof of the fix — it shows
  > the fix works as designed, but on this particular corpus/layer pair the
  > fix's own reach (top-50) still isn't wide enough to find a directional
  > match. **E14 stays `[~]`.** Named next step, sharper than before: either
  > widen `ground_truth_alignment` itself to compute (not just report) each
  > directional field's best match regardless of its global |ρ| rank — a
  > small, targeted change, since the per-field `rho` values are already
  > computed for all features before the top-50 truncation happens — or
  > accept that `trend_scale`/`seasonal_amplitude_max` are genuinely rarely
  > any feature's best match on this corpus/layer pair at any rank and stop
  > trying to force an evaluable example via candidate-list surgery.
- [x] **E15 · Per-token SAE evaluation, then feature-level ablation.** Fixes
  the window-broadcast confound that currently makes Chronos's
  forecast-preservation number unreadable (§6.2), which is the stated blocker
  on §7 bullet 3. Reconstruct and patch at *token* granularity rather than
  broadcasting one window-pooled vector across every token in the window;
  for per-timestep-tokenized models this is the difference between a
  meaningful validity check and an architecture-dependent artifact. Then
  feature-level ablation as §7 bullet 3 specifies, cross-referenced against
  ground-truth alignment. Depends on: A4 and A13 (so the resulting ΔMASE
  values mean something).

  **Findings (2026-08-11, sixteenth cron-loop firing — the token-granularity
  half only; feature-level ablation itself is not yet started, see below).**
  `sae/eval.py::forecast_preservation` gained `granularity: "window" |
  "token"` (default `"window"`, so no prior recorded number changed meaning);
  live-verified against `runs/medium_run_chronos_base`'s real
  TimesFM-2.5-200M / Chronos-T5-Base checkpoints via a background `Agent`
  (`--stages sae,report --force sae`, freshly retrained SAE — absolute
  values differ from the earlier ΔMASE +0.05/+2.4 pair recorded in
  `CLAUDE.md` §13 item 3 and §6.2's Findings; that's normal SAE-training
  variance, not a regression). TimesFM: `mase_delta` identical between
  granularities (`+0.10971450805664062` both ways) — expected, since its
  token width equals the alignment window, so the fix is a no-op there.
  Chronos-T5-Base: `mase_clean: 3.1090171337127686` both times;
  `mase_reconstructed` **6.9776177406311035** under `"window"` (`mase_delta:
  3.868600606918335`, badly fails) vs. **2.7632558345794678** under
  `"token"` (`mase_delta: -0.3457612991333008`, net *better* than clean).
  Decisive: the window-broadcast confound, not SAE reconstruction quality,
  was the dominant driver of every previously-recorded Chronos
  forecast-preservation failure. **Open caveat, flagged by the verifying
  agent and not yet investigated:** Chronos's negative token-granularity
  delta is itself a little surprising — n=24 series sample noise, or the
  SAE's TopK sparsity acting as a mild denoiser, are the leading guesses,
  neither confirmed. Don't read this as "the SAE is simply excellent"
  without that follow-up.

  > 🔴 **Correction (2026-08-12) — that caveat is now resolved, against
  > this number.** §13's repeat-run-variance measurement retrained this
  > exact target at five seeds on the same frozen store: Chronos-T5-Base's
  > token-granularity ΔMASE is **+0.2457 ± 0.2113**, five-seed range
  > [−0.1118, +0.4136]. The **−0.3458** recorded above falls *outside* that
  > range, and the negative sign replicates in only **1 of 5** seeds at a
  > third the magnitude. It was seed noise, not a denoising effect.
  > **Do not quote −0.346 as Chronos's token-granularity delta.** What
  > survives — and survives decisively, the gap being ~16× the seed sd — is
  > this item's actual conclusion: window **+3.897 ± 0.242** vs. token
  > **+0.246 ± 0.211**, so the window-broadcast confound really was the
  > dominant driver of every previously-recorded Chronos failure. TimesFM's
  > granularity-identity claim is likewise strengthened: the two
  > granularities are bit-for-bit identical at *every one* of the five
  > seeds, not just in this single run. Numbers in §13's Findings block. `CLAUDE.md` §13 item 3 carries the same numbers
  in a "Correction:" callout. **Feature-level ablation (the item's second
  half) is now started, 2026-08-11 (seventeenth cron-loop firing) — see
  §13's own "Feature-level ablation" bullet's new Findings note for the
  full mechanism and test status.** Implemented and unit-tested
  (`sae/eval.py::feature_ablation_effects`, wired into `run_sae` behind
  `sae.feature_ablation_enabled`). **Live-verified same day, same firing**
  against real `google/timesfm-2.5-200m-pytorch` /
  `amazon/chronos-t5-base` activations (154 / 115 ground-truth-matched
  candidate features respectively, exit code 0, no errors) — full numbers
  in §13's "Feature-level ablation" Findings note. This item is now closed
  (`[x]` in §13).
- [x] **E16 · Cross-model feature matching for per-model dictionaries.** The
  crosscoder (§6.2 item 1) is one answer to cross-model comparability; the
  complementary one — needed anyway, per `CLAUDE.md` §13 item 3 — is matching
  independently-trained dictionaries: input-space decoder correlation,
  max-activating-example overlap (Jaccard over top-k series/windows), and
  ground-truth-field agreement as a third, independent matching signal this
  repo can compute and most SAE work can't. Report the matched/unmatched
  split as a direct answer to the founding question, alongside the
  crosscoder's shared/specific split, and check whether the two methods
  agree — a genuinely informative cross-check.

  **Implemented and live-verified 2026-08-11 (eighteenth cron-loop firing).**
  New `sae/matching.py` (`matched_candidates`, `activation_profile_correlation`,
  `top_k_series`, `jaccard`, `ground_truth_agreement`,
  `match_cross_model_features`) + `tests/test_sae_matching.py` (5 synthetic,
  planted-answer tests — a true-positive planted profile pair and an
  unrelated-profile true negative, both confirmed recovered/rejected
  correctly). **One real bug caught by the planted test itself, before any
  live run**: `ground_truth_agreement`'s `bool_a == bool_b and np.sign(...)
  == np.sign(...)` returned `numpy.bool_`, not Python `bool`, so a
  downstream `is True` check failed — the exact `numpy.bool_`/`is` class of
  bug `CLAUDE.md` §9's capability-matrix note already flagged once before,
  recurring in new code. Fixed with an explicit `bool(...)` wrap; noted
  in-code as a callback to that precedent so it isn't rediscovered as a
  surprise a third time.

  **Reframing from the bullet's original wording, stated explicitly rather
  than silently substituted:** "input-space decoder correlation" is not
  literally implementable — two independently-trained SAEs over
  different-architecture models have decoder vectors in different-sized
  hidden spaces with no shared basis to correlate. The actual signal
  implemented is **activation-profile correlation**: Pearson correlation
  between two features' own per-series activation values across a *shared
  series sample* — which `ground_truth_alignment` already provides for
  free. Confirmed directly on a real run
  (`runs/medium_run_chronos_base_feature_ablation_check`): both targets'
  `ground_truth_alignment["rows"]` lists are **identical**
  (`gt_a["rows"] == gt_b["rows"]` → `True`), because both were sampled with
  the same run-level `cfg.run.seed + 12` and `cfg.sae.ground_truth_max_series`
  — a design property that was true by construction before this session but
  had never been exploited for anything cross-model. This means no new
  extraction, alignment window, or index-matching machinery was needed —
  the existing per-target ground-truth artifacts already contain a ready
  shared cross-model index. Candidates on each side are restricted to that
  target's own ground-truth-matched features (`best_ground_truth_matches`'s
  existing top-50-by-`|rho|` truncation), so matching stays a cheap ≤50×50
  search regardless of dictionary size. Matching is greedy nearest-neighbor
  (best B-side partner per A-side candidate), not an optimal bipartite
  assignment — stated as a scope limit in the module docstring, not hidden;
  a Hungarian-algorithm version is a natural but not-yet-built improvement.
  Max-activating overlap is series-level (top-k most-activating series per
  feature), matching `ground_truth_alignment`'s own already-established
  series-level (not window-level) scope for the same reason its own
  docstring already states.

  **Live-verified same day** against the real, already-trained SAE
  checkpoints in `runs/medium_run_chronos_base_feature_ablation_check/sae/`
  (TimesFM `stacked_xf.18`, dict_size 10240; Chronos-T5-Base
  `encoder.block.6`, dict_size 6144 — the exact pair the feature-ablation
  work already used, no new training or extraction needed): loaded both
  `.pt` checkpoints via the existing `load_sae_checkpoint`, re-encoded the
  shared 288-series sample from the existing zarr store via
  `store.load(..., level="series", rows=gt_a["rows"])`, and ran
  `match_cross_model_features` with `top_k=10, corr_threshold=0.3`. Real
  numbers: of TimesFM's 50 ground-truth-matched candidate features, **40
  found a Chronos-side partner clearing the correlation threshold, 10
  did not** — a genuine, non-degenerate matched/unmatched split, not
  everything-matches or nothing-matches. The single strongest match: TimesFM
  feature 2872 (`archetype_trend_dominant`, ρ=0.648) ↔ Chronos feature 2312
  (`archetype_trend_dominant`, ρ=0.527), activation-profile correlation
  0.831, series-level Jaccard 0.818, ground-truth field *and* sign agreeing
  — three independent signals lining up on the same pair is a decisively
  stronger claim than any one of them alone. Most of the top-10 matches by
  combined score are `archetype_trend_dominant` or `tier_realism_stress`
  pairs, both with `ground_truth_agree=True` — a first real answer to the
  founding "does each model learn the same features" question at the
  matching-signal level, complementary to (not yet cross-checked against)
  the crosscoder's shared/specific split from §6.2's feasibility test. Full
  `tsfm_lens` suite re-run after this addition — see §14's dated session-log
  entry for the pass count. **Not yet done, named as the natural next
  step, not hidden as a completed claim:** cross-checking this matching
  result against the crosscoder's own shared-vs-specific atom split (the
  "check whether the two methods agree" half of this bullet's original
  ask) — the crosscoder is feasibility-gate only per §6.2, no `SAEAdapter`
  or trained cross-model dictionary yet exists to compare against; and the
  Hungarian-assignment upgrade over the current greedy match.
- [~] **E17 · Input front-end diagnostics as a real stage.** Generalize the
  one-off `analysis/quantization_churn.py` script into a `frontend` stage
  characterizing what each model does to its input *before* any layer:
  quantization resolution and dynamic range (Chronos), patch-boundary **phase
  sensitivity** (shift the series by 1..patch_width and measure MASE variance
  — this is the direct test of the period=32 aliasing hypothesis §7 flagged
  and explicitly did not verify), scale-equivariance (multiply the series by
  1000 — does anything change?), context-truncation behavior, and NaN/missing
  handling. Cheap (mostly tokenizer calls and `predict`), and it explains a
  whole class of otherwise-mysterious behavioral anomalies rather than
  leaving them as flagged curiosities.
  — **2026-08-11: the named phase-sensitivity probe (the "direct test of the
  period=32 aliasing hypothesis" this item's own text calls out) is
  implemented and unit-tested** — `tsfm_lens/analysis/phase_sensitivity.py`
  (pure stats: per-series MASE coefficient of variation across context-trim
  shifts) + `run_phase_sensitivity_sweep.py` (I/O script: trims
  0..patch_width-1 points off the context's front, `adapter.predict()` at
  each trim, patch width derived per-model from `token_time_spans()` rather
  than hardcoded to 32) + `tests/test_phase_sensitivity.py` (3 synthetic
  planted-answer tests, all passing). A live-checkpoint verification against
  `runs/medium_run_chronos_base` was launched in the background same firing
  — see this item's Findings below once it lands. **Not yet done:**
  quantization resolution/dynamic range, scale-equivariance,
  context-truncation, and NaN/missing-handling diagnostics, and turning any
  of this into an actual `frontend` pipeline `Stage` (today it is a
  standalone script in the `run_quantization_churn_sweep.py`/
  `run_noise_snr_sweep.py` mold, not wired into `pipeline.py`) — this item
  stays open, just partially closed.
  — **2026-08-11, same-day follow-up: the background live-checkpoint
  verification landed, found and fixed one real bug, and produced the
  probe's first live numbers.** First attempt crashed:
  `RuntimeError: shape '[64, -1, 32]' is invalid for input of size 32704`
  inside TimesFM 2.5's own `decode()`. Root cause, diagnosed not guessed:
  `decode()` does `torch.reshape(inputs, (batch, -1, 32))`, requiring
  context length to stay an exact multiple of the 32-step patch width —
  but the script's naive `contexts[:, shift:]` trim shrinks the array to
  `context_len - shift`, which is off that multiple for every `shift` other
  than 0 or 32, and `BenchmarkData.contexts()` has no earlier history to
  extend the front into instead. **Fixed in `run_phase_sensitivity_sweep.py`
  only** (`tsfm_lens/analysis/phase_sensitivity.py` untouched, confirmed via
  `git status`): a new `_predict_point_trimmed()` keeps the tensor at the
  original, patch-aligned `context_len` and zero-fills the front `shift`
  positions while marking them `True` (padded/invalid) in TimesFM's own
  `masks` argument to `decode()` — the exact mechanism the model already
  uses for variable-length context (masked positions excluded from
  per-patch revin normalization and zeroed post-norm), read directly off
  `timesfm_2p5_torch.py::decode`. Calls `adapter.tfm.model.decode(...)`
  directly rather than widening the shared `ModelAdapter.predict()`
  contract with a mask parameter no other adapter needs. MASE's own naive-
  baseline denominator is still computed from the real trimmed context, not
  the zero-padded model-input array. A generic non-TimesFM fallback (plain
  trimmed slice) is kept for any future patch-tokenized model lacking a
  `.tfm` attribute, though unreached today since per-timestep models are
  skipped upstream as the null control. Unit tests re-confirmed passing
  after the fix. **Live numbers** (`runs/medium_run_chronos_base`, 64
  series, TimesFM-2.5-200M vs. Chronos-T5-Base): Chronos-T5-Base correctly
  skipped with the expected null-control log line
  (`patch_width=1 (no patch boundary to test)`). TimesFM (`patch_width=32`,
  shifts 0..31): corpus-mean per-series MASE coefficient of variation across
  shifts = **0.0232** (95% CI [0.0181, 0.0287], all 64 series usable, none
  excluded as near-zero-MASE), mean MASE ranging from **2.2955** (best,
  shift 1) to **2.3695** (worst, shift 30), worst-minus-best gap = **0.0740**.
  **Reading:** a real but modest effect — TimesFM's forecast MASE varies
  ~2% in aggregate purely from where an arbitrary front-trim lands relative
  to the 32-step patch grid, with the true context endpoint and forecast
  target held fixed throughout — confirming the aliasing hypothesis is real
  at the forecast level, not just plausible, though far smaller than the
  qualitative "hard ceiling" framing might suggest. Per-series variation is
  much larger than the aggregate: individual series show CV up to **0.125**,
  so the corpus-mean number understates how much any single series's
  forecast can swing with patch phase alone — a genuinely useful per-series
  diagnostic, not just a corpus-level curiosity. Full
  `runs/medium_run_chronos_base/../phase_sensitivity_sweep.json` has the
  per-shift and per-series arrays. **Scope note, unchanged:** this confirms
  the *forecast-level* phase-sensitivity sub-piece of E17 specifically — the
  item's remaining scope (quantization/dynamic-range, scale-equivariance,
  context-truncation, NaN handling, and wiring into `pipeline.py` as a real
  `frontend` `Stage`) is still open, so E17 stays `[~]`, not `[x]`.
- [ ] **E18 · Component-level attribution (path patching)** — **parked, scope it
  inside H8 (§22.7).** It is the dependency H8's seasonality circuit needs and
  has no independent consumer, so building it standalone would leave a
  capability with nothing to say.

### T4 — Breadth and scale

- [x] **E19 · Multivariate / covariate envelope.** Moirai and Chronos-2 (both
  §9 Phase 4 candidates) are multivariate-native; the whole pipeline assumes
  univariate `[series, time]`. Decide and document the axis extension *before*
  Phase 4 forces an improvised one: `[series, variate, window, dim]` with
  alignment applied per variate, cross-variate attention treated as its own
  measurement (not folded into the lag taxonomy), and an explicit statement of
  which analyses are variate-agnostic. Doing this as a design note now is
  cheap; doing it under pressure while adding a model is how abstractions get
  bent (§2.4).
  **Decided 2026-08-12 (cron loop): ratify univariate-only**, per this item's
  own reframing below and its stated recommendation. The deliverable — a
  decision paragraph — is in `CLAUDE.md` §12's "Envelope hard edges" list, not
  duplicated here. Summary of what was ratified and why: `pooling_matrix`'s
  premise is that a token maps to a contiguous time interval *within one
  series*, so a cross-series axis has no interval to pool, no place on the
  shared window axis every cross-model comparison is defined on, and no lag in
  timesteps — extending the axis means re-deriving alignment, not adding a
  dimension. Chronos-2's GROUP attention is recorded as the concrete
  exclusion (`_scan_attention` resolves to TIME by first-match), converting
  what was an improvisation at Phase 4 into a documented edge with a stated
  rule for future adapters. The decision explicitly records that the *size* of
  the resulting capture-coverage gap is unknown — nothing measures how much of
  a multivariate model's computation lives on the skipped axis — which is why
  a Moirai-class any-variate model stays scoped "gated on E19" in §19's
  landscape table rather than being called supported.
- [~] **E20 · Context-length and horizon scaling sweeps.** Reuse
  `analysis/parameter_sweep.py`'s machinery on the *config* axis rather than
  the data axis: how do MASE, crystallization depth, and attention lag
  profiles change as `context_len` grows (128 → the checkpoint's max)? Users
  choose context length in practice and have no guidance; and "does the model
  actually use long context" is answerable here with existing tools.

  **Implemented and live-verified 2026-08-11 (twenty-third cron-loop firing),
  L0 MASE axis only — first pass, matching `parameter_sweep.py`'s own
  precedent of scoping to one axis before widening.** New
  `tsfm_lens/analysis/context_scaling.py` (`context_length_values`,
  `generate_context_sweep_series`, `score_context_length_sweep`,
  `summarize_context_sweep`) + `run_context_scaling_sweep.py` (CLI, mirrors
  `run_parameter_sweep.py`'s structure: loads an existing run's
  `config_resolved.yaml` for checkpoints/device/dtype only, no
  re-extraction). Design: one fixed-length `parametric()` series per row at
  `max_context_len + horizon`; every swept context length slices trailing
  history immediately before the *same* fixed target window, so MASE vs.
  context length is a genuine dose-response curve (same series, same
  target, only the truncation point moves) rather than a comparison across
  different series per point. Context lengths are restricted to multiples
  of `window` (default 32) specifically to avoid confounding this sweep
  with E17's already-documented patch-phase sensitivity (~2% MASE swing
  from front-trim phase alone, holding true context length fixed) — a
  different question from this one. `tests/test_context_scaling.py` (7
  tests, all pure/deterministic — a fixed-target invariant test recording
  the target window seen at every sweep point via a planted `predict_fn`
  and asserting it never moves, plus a planted-threshold recovery test)
  passes 7/7; package-wide `compileall` clean; a mock-adapter dry run
  against `runs/smoke` produced sane output before attempting real
  checkpoints.

  **Live-verified same day against real checkpoints**
  (`runs/medium_run_chronos_base`'s TimesFM-2.5-200M / Chronos-T5-Base
  configs, no re-extraction — just `adapter.predict()` calls):
  `python run_context_scaling_sweep.py --run runs/medium_run_chronos_base --min-context 128 --max-context 512 --n-points 6 --n-series 64`,
  `context_lens=[128, 160, 224, 288, 384, 512]`, `n_series=64`,
  `max_context=512`, `horizon=64`, wall-clock `0m37.213s`. Exact table
  (mean per-series MASE, 95% bootstrap CI):
  ```
   context_len                TimesFM MASE        Chronos-T5-Base MASE
           128         0.964 [0.896,1.037]         0.850 [0.804,0.902]
           160         0.732 [0.707,0.760]         0.754 [0.728,0.782]
           224         0.711 [0.679,0.747]         0.697 [0.675,0.718]
           288         0.644 [0.628,0.665]         0.685 [0.666,0.705]
           384         0.644 [0.623,0.663]         0.673 [0.651,0.696]
           512         0.623 [0.608,0.637]         0.690 [0.663,0.716]
  ```
  **A genuine qualitative difference between the two models, on the first
  real run of this tool — the research question E20 was written to
  answer.** TimesFM improves (MASE decreases) monotonically all the way to
  the checkpoint's own max context (512 is its best point in this sweep;
  one flat step 288→384). Chronos-T5-Base improves sharply up to 384
  (its best point, 0.673) then gets *worse* at 512 (0.690) — additional
  context beyond ~384 slightly *hurts* Chronos-T5-Base here, on this one
  synthetic corpus/recipe, while TimesFM keeps benefiting through its own
  max. Full `tsfm_lens` suite unaffected (only the new, isolated module and
  CLI were touched; `tests/test_context_scaling.py` re-run standalone,
  7/7 passing). **Scope, stated not hidden:** one synthetic recipe
  (trend + two seasonalities + moderate noise), one seed, one corpus —
  not yet replicated across recipes/seeds, and crystallization
  depth/attention lag profile vs. context length (also named in this
  bullet) are still open, hence `[~]` not `[x]`.
- [ ] **E21 · Chronos decoder capture as a separate measurement** — **parked
  (§22.7).** The largest *measured* limitation in the repo: F4 established that
  ~86% of a Chronos-T5-Base forecast's FLOPs are unobserved. The binding design
  constraint is already settled — pooled decoder states live on *forecast* time
  and must **never** be CKA'd against context-window states; it is a second
  measurement, not a symmetry fix. Detail-up retained below.
- [ ] **E22 · Scale** — **parked (§22.4).** Split into E22a (full-scale corpus
  validation, unblocked now that A17's matcher work is `[x]`) and E22b
  (full-scale extraction + analysis). Both are wall-clock-bound and neither
  changes a conclusion.
- [x] **E23 · `benchmark_validation` diversity metrics as CI pass/fail
  gates.** `CLAUDE.md` §13 item 4, not previously cross-referenced into this
  backlog (closed the gap 2026-08-11, §11's own item revisiting `CLAUDE.md`
  §13). Turn the diversity metrics (`validation_report.json`'s redundancy
  fraction, effective dimensionality, near-collision fraction, per-group
  breakdowns from A17's fix) into explicit numeric thresholds checked
  automatically per benchmark epoch, rather than eyeballed from the report —
  e.g. fail if redundancy fraction exceeds some X, effective dimensionality
  drops below some Y, or any subgroup's near-collision fraction exceeds Z.
  Distinct from E8/A14 (test-suite CI): this is a data-quality gate on a
  *generated corpus*, not a code-correctness gate on the *pipeline*.
  **Closed 2026-08-11 (twenty-second cron-loop firing, fourth follow-up):**
  `gates.py`/`check_diversity_gates`/`--enforce-gates` were already
  implemented (see the "same firing" entry below this one, dated earlier the
  same day) but had never been calibrated against anything beyond the
  original 205-sequence synthetic demo. This firing added the two missing
  reference points named above: a second real, non-demo corpus
  (`benchmark_medium/public_dev`, 288 sequences — comfortable pass, see the
  entry below) and, closing the harder gap, three independent real FAIL
  constructions (demo mode's own planted duplicates fail
  `group[unknown].near_collision_fraction` once `--enforce-gates` is
  actually passed; a hand-built redundancy sweep flips
  `redundancy_fraction` from PASS to FAIL between 30% and 40% duplicate
  fraction in a 100-sequence corpus; an exact-duplicate isolation test
  confirms `effective_dimensionality`/`near_collision_fraction` correctly
  collapse to their degenerate extremes only under true bit-identical
  duplication, not near-duplication). Full numbers in the twenty-second
  firing's fourth-follow-up §14 entry. `DEFAULT_THRESHOLDS` was not changed
  — the finding is that the existing thresholds already correctly
  discriminate real pass/fail cases, not that they needed recalibrating.
- [ ] **E24 · Real-corpus activation bucketing** — **parked (§22.3).** Embed a
  large real corpus (Monash/GIFT-Eval/LOTSA), cluster, compare partitions
  across models via AMI. Parked because clusters on wild data are
  uninterpretable without the controlled reference points the synthetic corpus
  already supplies — which is `CLAUDE.md` §13 item 7's own stated reason for
  sequencing it last.

### Detail-up for the twelve unstarted items (added 2026-08-11)

> **Why this subsection exists.** The twelve `[ ]` items above were written as
> *arguments* for why the work matters — which is the right register for a
> backlog, and useless to someone who has to build one. Each block below adds
> the missing half: exact deliverable files, function signatures, the
> acceptance criterion that closes the item, the test that proves it, and a
> rough cost. Nothing above is superseded; this is the "how", added next to
> the existing "why". Items already `[x]` or `[~]` are not repeated here.
>
> **Read the corrections block at the top of §16 first** — six of these twelve
> list dependencies that are already satisfied.

---

**E3 · Empirical span discovery — detail-up**

*Blocker status:* **none.** A2 is `[x]`. Startable today.

*The algorithm, stated concretely* (the item describes the idea, not the
procedure):
1. Build a probe batch as `impulse_alignment_check` already does.
2. For each candidate timestep `t` (stride-`s` sweep, refine later), add an
   impulse of `amplitude_frac * base.max()` at `t` — **relative, never
   absolute**; `CLAUDE.md` §11.16 is the cost of getting this wrong.
3. Forward, capture the **first** captured block (least mixing), compute per
   token position the L2 norm of the activation delta vs. the unperturbed run.
4. `argmax` over token positions → the token responsible for `t`. Sweeping `t`
   gives a `t → token` map.
5. **Invert to contiguous spans.** The `ModelAdapter` contract requires each
   token map to a *contiguous* interval, so the raw argmax map must be
   coerced: for each token, take `[min(t), max(t)]` over the timesteps that
   chose it, then flag any token whose chosen-timestep set is non-contiguous
   (holes) or whose span overlaps another token's by more than a tolerance.
   Report the flagged fraction as the **diffuseness score** — this is the
   number that decides refusal.
6. **Sweep amplitude, do not fix it.** Sundial's alignment is amplitude-
   dependent (`CLAUDE.md` §11.22) and Chronos's tokenizer rescales globally
   (§11.16). A single amplitude gives a per-model-arbitrary answer. Run at
   ≥3 amplitudes (e.g. 0.05/0.15/0.25) and require the derived span map to
   agree across them, or report instability.

*Deliverables.*
- `tsfm_lens/extraction/span_discovery.py`:
  - `discover_spans(adapter, context_len, amplitudes, stride, device) -> SpanDiscovery`
    where `SpanDiscovery` is a dataclass carrying `spans: np.ndarray [n_tokens,2]`,
    `diffuseness: float`, `per_amplitude_agreement: float`, `flagged_tokens: list[int]`.
  - `compare_declared(adapter, discovered) -> dict` — declared-vs-discovered
    IoU per token; the cross-check E3(d) asks for.
- `run.py --discover-spans <model>` printing the table + scores.
- `models/generic_hf_adapter.py::GenericHFAdapter` — combines `--discover-layers`'
  existing `[B,T,D]` shape probing with `discover_spans` output for
  `token_time_spans()`. Must implement the **refusal path**: if
  `diffuseness > refuse_threshold`, `load()` succeeds but the adapter declares
  no time-localized spans, and the pipeline runs **L0 only** with the reason
  written into the run dir and the report (not just logged — invariant 8).

*Acceptance criterion.* For all five hand-written adapters, discovered spans
match declared spans (mean IoU ≥ 0.9) — i.e. the method reproduces answers
already known correct. Then: one checkpoint with *no* hand-written adapter runs
end-to-end through `GenericHFAdapter` to a report.

*Test.* `tests/test_span_discovery.py` against the mock adapters, whose true
spans are known exactly by construction — assert exact recovery, then assert
the refusal path fires on a deliberately non-localized mock.

*Cost.* 2 sessions. Highest generality-per-hour item in the file; it is what
turns "add a model" from an adapter-writing task into a run.

---

**E1 · Zero-config entry point, E4 · Bundled reference corpus, E5 · Results
registry — parked as a bundle (§22.3)**

The T1 adoptability product: ~4+ sessions, E1 depends on E4, E4 carries an
unresolved licensing decision. Parked because it builds for a user who does not
exist yet, and because §22.9's ordering constraint means legibility and
equal-grounds work has to land first regardless. The one genuinely reusable
piece of E1's detail-up, worth stating here so it isn't rediscovered: **there
is no checkpoint-id → adapter-name mapping anywhere in the repo today**
(`models/__init__.py` maps *name* → class), and an ordered `(regex,
adapter_name)` table with a `ValueError` naming every known pattern on no
match is that item's real content. Full detail-ups in `ROADMAP_ARCHIVE.md`.
**E6 · Report as a shareable product — detail-up**

*Blocker status:* **none.** All four A-dependencies are `[x]`, and two of E6's
own sub-deliverables are already built (coverage panel, provenance panel — see
the corrections block).

*What actually remains, and the one refactor it needs.*
🔴 `findings.json` is not a serialization task, it is a **~12-call-site
refactor**. Findings are accumulated today as a `list[str]` of prose — there is
no claim id, evidence class, or stage attached to a finding, so the file cannot
be emitted without changing how findings are created. Do this first:
- Add `@dataclass Finding` in `report/report.py`: `claim_id: str`,
  `stage: str`, `evidence_class: Literal["geometric","translatable",
  "causal_within_model","descriptive","illustrative","behavioral"]`,
  `text: str`, `registered: bool`, `cleared_noise_floor: bool | None`,
  `value: float | None`, `ci: tuple | None`.
- Change every `findings.append("...")` site to `findings.append(Finding(...))`.
  Render `.text` where prose is rendered today, so the HTML is unchanged by
  this step alone — verify that by diffing a smoke report before/after.
- `claim_id` should reuse A15's registry ids where a finding corresponds to a
  registered claim, and a generated `f"{stage}.{slug}"` otherwise.
Then the rest is cheap: `json.dump` the dataclass list; confidence badges are a
CSS class keyed on `evidence_class` + `registered`; stable anchors are
`id=f"sec-{slug}"` on each section header; the diff mode reads two runs'
`findings.json` via E5 and renders added/removed/changed by `claim_id`.

*Acceptance criterion.* Smoke report's HTML is byte-identical after the
dataclass refactor (proving it is pure plumbing), `findings.json` validates
against the dataclass on reload, and every section is deep-linkable.

*Test.* Extend `tests/test_smoke.py` to assert `findings.json` exists, parses,
has one entry per rendered finding, and that every `evidence_class` is one of
the allowed literals.

*Cost.* ~1.5 sessions, most of it the mechanical refactor. Do the refactor as
its own commit.

**Findings — 2026-08-18.** Done via a single background agent (`CLAUDE.md`
§2.8), verified independently rather than taken on the agent's word alone
(§2.4): re-ran `tests/test_smoke.py` myself after the fact (5 passed) and
hand-inspected a live `runs/smoke/report/findings.json` directly (45
findings, all `claim_id`s unique, `evidence_class` values all within the 6
allowed literals, `registered=True` on exactly the 2 `confirm`-stage
findings) rather than trusting the agent's own reported numbers unchecked.

- **The dataclass refactor.** `Finding` added at `report.py`'s top; all 36
  `findings.append(...)` call sites (not ~12 — the plan's own estimate was
  low, whether that meant call sites or functions) converted. `.text` is
  rendered wherever prose rendered before.
- **Byte-identical HTML — confirmed, but the naive check would have lied.**
  `git HEAD` (`b530337`) turned out to already be several sessions stale
  relative to the working tree's actual pre-edit `report.py` (missing a
  whole "Fairness" section and a changed `_sec_budget` arity) — diffing
  against HEAD as "before" would have manufactured a false failure entirely
  unrelated to this refactor. Fixed by reconstructing the true pre-edit file
  (reverse-applying the refactor's own edits on a scratch copy, which
  fails loudly on any mismatch) and diffing *that* against the post-refactor
  output: identical except Plotly's per-render random div UUIDs (confirmed
  pre-existing noise by rendering the same unmodified code twice). Anchors
  were verified as a second, separate diff step — dataclass-only vs.
  fully-edited output differs *only* by the added `id="sec-{slug}"`
  attributes, nothing else moved. `report/coverage.json` byte-identical
  throughout.
- **`findings.json`**: `run_dir/report/findings.json`, `{"findings": [...]}`
  of `dataclasses.asdict(f)` per entry. 45 entries on `configs/smoke.yaml`.
- **`claim_id`**: a plain per-stage counter (`f"{stage}.{n}"`), not A15's
  hypothesis-registry ids — reusing those would need a nontrivial
  text-to-hypothesis mapping the item didn't scope, so this is a deliberate
  simplification, not an oversight. Revisit if J1's `caveat` generation or
  a diff mode ever needs cross-run claim identity finer than "same stage,
  same ordinal."
- **Exploratory marking keeps both forms deliberately**: the existing
  `"[exploratory — not pre-registered] "` text prefix stays (required for
  byte-identical HTML) *and* `registered: bool` is now also a real
  structured field (`False` for exploratory, `True` for `confirm`'s two
  findings) — so a consumer that wants the flag doesn't have to parse the
  prefix back out of `.text`.
- **`evidence_class` judgment calls** (flagged for review, not silently
  decided): Attention section findings — including the ablation-based ones,
  which read as arguably causal-within-model — are `descriptive`, matching
  this item's own suggested mapping rather than CLAUDE.md §6.1's per-stage
  table verbatim; worth a second look if a future session leans on
  Attention's evidence-class tag for something load-bearing. `confirm`'s two
  findings are *not* both the same class: the family-hypothesis finding is
  `behavioral`, the CKA-replication finding is `geometric` — each tagged by
  what it's actually a claim about, not by the stage's single "gold
  standard" label.
- **Scope actually delivered vs. the detail-up's "then cheap" list**:
  confidence badges and the diff mode were **not** built this pass —
  confidence badges are left for J1 (which needs `Finding` extended with
  `plain`/`caveat` fields anyway, so styling both in one pass avoids
  touching every render site twice), and the diff mode is explicitly gated
  on E5, which is parked (§22.3). Neither is part of E6's own stated
  acceptance criterion (byte-identical HTML, `findings.json` validates,
  every section deep-linkable) — all three of which are met.
- **Unplanned but necessary fix**: converting `findings` from `list[str]` to
  `list[Finding]` broke 7 existing tests that called `_qualify_depth_claims`
  / `_sec_attention` / `_sec_budget` directly with bare-string fixtures or
  asserted `in f`/`f.count(...)` against a raw string
  (`tests/test_capture_coverage.py`, `test_floor_units.py`,
  `test_budget_stage.py`) — a real, foreseeable downstream break
  (`CLAUDE.md` §2.1), fixed by updating those call sites to operate on
  `.text` rather than leaving the suite red. First full-suite run after the
  refactor: 452 passed / 7 failed (exactly these). After the fix: **459
  passed, 0 failed** (re-verified independently, not just on the agent's
  report). `tests/test_smoke.py` extended per the item's own test plan
  (`findings.json` exists/parses/one-entry-per-finding/valid
  `evidence_class`).
- **`report/meta_report.py`** confirmed untouched and out of scope — it
  reads existing run artifacts and does not build its own findings list.

---

**E7 · Docs, quickstart, packaging — split into three**

The single checkbox bundles one blocked item with two ready ones.

- [x] **E7a · `tsfm-lens` console script.** *Unblocked.* Add
  `[project.scripts] tsfm-lens = "tsfm_lens.run:main"` to
  `tsfm_model_analysis/tsfm_lens/pyproject.toml`, which requires `run.py`'s
  argument parsing to move behind a `main()` function (it is currently
  top-level under `__main__`). Acceptance: `pip install -e .` then
  `tsfm-lens --config configs/smoke.yaml` works from any directory — which
  also exercises invariant 11 (no path may be relative to the CWD in a way
  that breaks this).

  > **Findings — done, and the item's own stated blocker was already
  > stale.** `run.py` already had a `main()` function wrapping its argument
  > parsing before this session touched anything — that half of the
  > acceptance criterion needed no work. The real gap was layout: the CLI
  > lived at the top level (`tsfm_model_analysis/tsfm_lens/run.py`, a
  > sibling of `pyproject.toml`, not inside the `tsfm_lens` importable
  > package), so `tsfm_lens.run:main` had nothing to resolve to. Fixed by
  > relocating the CLI into the package
  > (`tsfm_model_analysis/tsfm_lens/tsfm_lens/run.py`) and leaving a thin
  > shim at the historical top-level path (`tsfm_model_analysis/tsfm_lens/
  > run.py`, `from tsfm_lens.run import main` + `if __name__ ==
  > "__main__": main()`) so every existing doc command in `CLAUDE.md` §8
  > (`python run.py --config ...`, run from `tsfm_lens/`) keeps working
  > unchanged — `[project.scripts]` in `pyproject.toml` already had the
  > correct `tsfm-lens = "tsfm_lens.run:main"` line, just pointing at a
  > module that didn't exist inside the package until this move.
  > Verified, not just diffed: (1) `tsfm-lens --discover-layers` invoked via
  > the installed console script from an unrelated directory produced
  > byte-identical output to `python run.py --discover-layers` invoked the
  > old way from `tsfm_lens/`; (2) a full real end-to-end run of
  > `configs/smoke.yaml` via the console script rendered the same
  > 12-section report the shim path already produces, 45 findings; (3) the
  > full `tsfm_lens` test suite — 412 passed, 0 failures — confirming the
  > shim didn't silently break any existing `python run.py` invocation
  > invariant 11 depends on. No path in either `run.py` file is relative to
  > the CWD; both resolve via `Path(__file__)`, so the console script works
  > from any directory exactly as the acceptance criterion asks.
- [x] **E7b · Install-story docs.** ✅ **DONE 2026-08-18.** `README.md` was a
  2-line stub with no install instructions at all — this is the actual gap
  the item's "unblocked and mostly written" line undersold (the two
  sub-packages' own `README.md`s were already substantial; the top-level one
  wasn't). Rewrote it: what the repo is, the two-package structure with links
  to each half's own README, the two `pip install -e` commands with the
  dependency-weight rationale transcribed from `CLAUDE.md` §3, and a
  five-command quickstart (benchmark build → validate → smoke report) that
  was checked against `CLAUDE.md` §8's own documented CLI invocations rather
  than invented fresh. `DEPENDENCIES.md` §2's existing two install-command
  block got a short inline comment cross-referencing the same rationale
  rather than a redundant restatement. No code changed; this is docs only,
  done inline (well under 5 minutes, no background delegation needed).
- [ ] **E7c · Quickstart + four notebooks.** *Blocked on E1* (the quickstart
  is literally E1's one command) and, for the "add an adapter" notebook, on
  E3. The other three notebooks — smoke run, train/read an SAE, read a
  confirm verdict — are unblocked and can ship first.

---

**E18 · Path patching — parked, but scope it as part of H8 (§22.7)**

Path patching is the dependency H8's seasonality circuit needs and has no
independent consumer, so building it standalone would leave a capability with
nothing to say. If H8 is un-parked, E18 is its first task, not a prerequisite
to schedule separately. Detail-up in `ROADMAP_ARCHIVE.md`.
**E19 · Multivariate envelope — reframed as a retroactive decision**

Phase 4 shipped, so this is no longer "decide before Phase 4". It is now:
**ratify or reverse.** Chronos-2 is integrated with its GROUP (cross-series)
attention axis explicitly out of scope, and Sundial is univariate. The question
to answer, once, and record:
- Does the repo commit to **univariate-only** analysis, making the Chronos-2
  GROUP-axis exclusion a permanent, documented envelope edge (`CLAUDE.md` §12)?
- Or does multivariate enter the envelope, in which case the alignment premise
  needs re-derivation — `pooling_matrix` maps tokens to time intervals within
  *one* series, and a cross-series attention axis has no time interval at all.

*Deliverable is a decision paragraph in `CLAUDE.md` §12, not code.* Tracked as
its own §13 entry. Recommended: ratify univariate-only, and record the GROUP
axis as the concrete example of what that excludes.

---

**E20 · Context/horizon sweeps — split**

- [~] **E20a · Horizon sweep.** *Unblocked.* Reuses `predict()` only; no
      re-extraction. Cheap, and E12's horizon-resolved metrics already give it
      somewhere to land. **Implemented and locally verified 2026-08-11
      (twenty-fourth cron-loop firing)**, mirroring E20's own
      `context_scaling.py`/`run_context_scaling_sweep.py` pattern exactly:
      new `analysis/horizon_scaling.py` (`horizon_values` — geometrically
      spaced horizons ending exactly at `max_horizon`; `generate_horizon_
      sweep_series` — one fixed `context_len + max_horizon`-length synthetic
      series per row, context held fixed across every swept horizon;
      `score_horizon_sweep`/`summarize_horizon_sweep` — per-series MASE per
      horizon + series-bootstrap CI, reusing `stats.mase`/`mean_ci`) and
      `run_horizon_scaling_sweep.py` (CLI: loads an existing run's
      `config_resolved.yaml` for model configs only, generates its own
      synthetic sweep data, calls `adapter.predict()` per horizon — no
      benchmark corpus, no store I/O). New `tests/test_horizon_scaling.py`,
      6/6 passing (shape/determinism, positive-arg validation, a
      fixed-context invariant test confirming every swept horizon really
      does see the identical context slice, and a planted-degradation
      recovery test with a stand-in "model" that forecasts perfectly to
      step 8 then zeros — MASE at h=16/h=32 both clearly exceed h=8's exact
      zero, confirming the scoring pipeline detects induced degradation;
      note the first version of that test also asserted h=32 > h=16
      strictly, which failed against the real numbers (2.09 < 2.33) — MASE
      is scale-normalized per series, not horizon-length-normalized, so
      that stronger ordering wasn't a valid claim and was dropped rather
      than the code being bent to satisfy an over-specified test, per
      §2.4's discipline). Dry-run against the mock `runs/smoke` config
      (`patchy`/`steppy`, CPU, ~1s) completed cleanly end-to-end and showed
      the expected qualitative shape — MASE rising with horizon for both
      mock adapters. One real bug caught and fixed during the dry run,
      before any live-checkpoint attempt: the CLI's default output path
      (`run_dir.parent / "horizon_scaling_sweep.json"`) collides across
      every base run, since `run_dir.parent` is the same `runs/` directory
      regardless of which run's configs were loaded — a second sweep
      against a different base run would silently overwrite the first
      one's JSON with no error. Fixed by keying the default filename off
      `run_dir.name` (`horizon_scaling_sweep_{run_dir.name}.json`).
      ~~**Not yet done, why marked `[~]` not `[x]`:** no live real-checkpoint
      run yet~~ — **done 2026-08-12 (cron loop); E20a is now `[x]`.** Ran
      exactly the invocation this paragraph proposed (`--run
      runs/medium_run_chronos_base --min-horizon 8 --max-horizon 64
      --n-points 5 --n-series 48`, `CUDA_VISIBLE_DEVICES=1`) against live
      TimesFM-2.5-200M / Chronos-T5-Base. **29 seconds end-to-end**, both
      checkpoints loaded and released, no warnings beyond transformers'
      `torch_dtype` deprecation notice. Horizons `[8, 13, 23, 38, 64]`,
      context held at 512. Wrote
      `runs/horizon_scaling_sweep_medium_run_chronos_base.json`. Result
      (MASE, mean with series-bootstrap 95% CI):

      | horizon | TimesFM | Chronos-T5-Base |
      |---|---|---|
      | 8 | 0.613 [0.568, 0.655] | 0.690 [0.640, 0.737] |
      | 13 | 0.622 [0.583, 0.662] | 0.664 [0.622, 0.708] |
      | 23 | 0.610 [0.581, 0.638] | 0.646 [0.613, 0.678] |
      | 38 | 0.609 [0.587, 0.629] | 0.656 [0.625, 0.693] |
      | 64 | 0.620 [0.603, 0.639] | 0.688 [0.658, 0.721] |

      **Two things worth reading here, one of them a caveat on the tool
      rather than a result about the models.** (1) **MASE is essentially
      flat in horizon for both models** — TimesFM spans 0.609–0.622 and
      Chronos 0.646–0.690 across an 8× horizon range, with every CI
      overlapping its neighbours. TimesFM is lower at every horizon, and
      the two models' CIs are cleanly disjoint at h=64 ([0.603, 0.639] vs
      [0.658, 0.721]) while overlapping slightly at h=8 — i.e. what little
      the sweep separates, it separates *more* at long horizon, which is the
      one directional signal in the table. Chronos's curve is faintly
      U-shaped (best at h=23) but entirely within noise. (2) **This
      contradicts the mock dry-run's shape, and the mock was the misleading
      one.** The dry-run's "MASE rising with horizon for both mock
      adapters" was recorded above as "the expected qualitative shape" —
      real checkpoints show no such rise. The reason is the same one that
      already forced a test assertion to be dropped (see the h=32 > h=16
      note above): MASE normalizes by a **context-derived** naive scale that
      does not depend on horizon length, so a flat MASE means per-step
      absolute error is roughly horizon-independent — which is what a
      strong forecaster on the sweep's synthetic (largely periodic) series
      should do. The mock adapters degrade because they are toys, not
      because horizon degradation is the expected shape. **Do not read this
      as "horizon length doesn't matter for TSFMs"** on the strength of one
      synthetic corpus at n=48 and one horizon decade; what it establishes
      is that the sweep runs on real weights, is cheap enough to run
      routinely, and that any future claim of horizon degradation needs a
      corpus whose difficulty actually scales with horizon (E12's
      horizon-resolved metrics on the real benchmark are the better axis for
      that, and are where this should be cross-checked next).
      ⚠️ **Also surfaced this firing, not yet reconciled:** E20b's own
      framing above assumes every context-length sweep point needs a fresh
      extraction ("a different `context_len` changes the store's shape, so
      nothing is reusable"). The actual E20 implementation
      (`context_scaling.py`) does **not** do this — like E20a, it never
      touches the extraction store at all; it generates a fixed
      max-context-length synthetic series per row and slices trailing
      history per sweep point, calling only `adapter.predict()` directly
      against a live-loaded model, exactly the same mechanism this section
      uses for horizon. E20b's "expensive by nature" framing therefore
      describes a design this repo didn't end up needing — worth a closer
      look before treating E20b as blocked/expensive; it may already be
      unblocked and cheap by the same mechanism as E20a, just not yet
      relabeled as such.
**E20b · Context-length sweep — parked (§22.4).** Expensive by nature: every
context length is a fresh extraction. Detail-up in `ROADMAP_ARCHIVE.md`.
**E21 · Chronos decoder capture — detail-up**

*The single load-bearing constraint, restated so it cannot be lost:* pooled
decoder states live on **forecast time**, context states live on **context
time**. They are not the same axis. **Never CKA them against each other** — the
result would be meaningless and would look fine.

*Deliverables.* A `capture_surface: encoder|decoder` field on the Chronos model
config; teacher-forced decoding in `ChronosAdapter.forward` when
`decoder`-mode; a second span table over forecast timesteps; a separate
`activations_decoder.zarr` (not a new group in the same store — different time
axis, and mixing them invites exactly the mistake above). Analyses run over the
decoder surface must be explicitly opted into, and the report must label the
surface in every figure title.

*Acceptance criterion.* `--check-alignment` passes on the decoder surface with
its own span table, and an attempt to run L1 across the two surfaces raises
rather than producing a number.

---

**E22 · Scale, and E24 · Real-corpus activation bucketing — parked (§22.4, §22.3)**

Both are GPU/wall-clock-bound rather than thought-bound, and neither changes a
conclusion — E22 tightens CIs on conclusions already drawn at `medium_run`
scale, E24 clusters wild data whose clusters are uninterpretable without the
controlled reference points the synthetic corpus already supplies. Detail-ups
preserved in `ROADMAP_ARCHIVE.md`.
**Findings / decisions**
- *(append here — and per §2.5, before building any of these, state which
  claim it sharpens. Several T3 items are attractive precisely because they're
  interesting, which is exactly when that question needs asking.)*
- **2026-08-06 — E9 implementation session.** Picked E9 as the highest-value
  item in the backlog per this section's own text and the prior session's
  recommendation: every §15 P1 item was closed, so this was the first
  genuinely new capability rather than a fix. Sharpens: every L1 CKA, L2
  stitching-gain, internals probe-decodability, and SAE ground-truth
  alignment number already recorded in this file (`ROADMAP.md` §5's
  Findings, §6.1/§6.2's Findings) has so far been read against zero, not
  against an architecture-only floor — exactly the gap §2.2's own doctrine
  (demand nulls) and this item's text name.
  - **Design decision, checked against the actual code before writing any:**
    rather than building a new "null baseline" analysis path, `random_init`
    is implemented purely as a `ModelConfig` flag + one `if` branch inside
    each real adapter's existing `load()`. A model with `random_init: true`
    paired against its own pretrained twin is then just an ordinary
    two-model config — `l1_geometry.py`, `l2_stitching.py`, `internals.py`,
    and `sae/train.py` needed **zero changes** to produce real floor
    numbers, since they already operate generically on whatever two (or,
    for SAE, however many pinned-target) models a config names. This is a
    direct instance of `CLAUDE.md` §2.2 (prefer established mechanism over
    new code) and §2.4 (architecture-agnostic by construction).
  - **Mechanism, verified per-library rather than assumed (`CLAUDE.md`
    §2.4):** read `chronos_adapter.py`/`chronos_bolt_adapter.py`'s actual
    `load()` before writing anything, confirming `self.pipeline.model.model`
    / `self.pipeline.model` are plain `transformers.PreTrainedModel`
    instances (`T5ForConditionalGeneration` / `ChronosBoltModelForForecasting`,
    both confirmed live via `issubclass(..., PreTrainedModel)`), so
    `type(model)(model.config)` — HF's own from-config idiom, the same
    mechanism `AutoModel.from_config()` uses versus `from_pretrained()` —
    is exact and needs no custom reinitialization heuristic.
    `models/base.py::random_init_like` is this one function, used by both
    Chronos adapters. Read `timesfm` package source (not assumed) and found
    an even simpler case: `TimesFM_2p5_200M_torch.__init__` already builds
    the entire architecture (tokenizer, all 20 `stacked_xf` blocks, output
    projections) from a fixed, checkpoint-independent config *before*
    `.load_checkpoint()` is ever called, so TimesFM's `random_init` is
    simply constructing the wrapper and skipping that call — no checkpoint
    download or discard, faster than the Chronos path. Confirmed directly
    that HF's `T5LayerNorm` (used throughout Chronos/Chronos-Bolt) has **no**
    `reset_parameters()` method, which is exactly why a generic
    "`.apply(reset_parameters if present)`" heuristic was rejected in favor
    of the config-reconstruction approach — that heuristic would have
    silently left every T5LayerNorm weight at its pretrained value in a
    supposedly "untrained" model.
  - **Permutation null, scoped to what E9's own text names:** implemented
    for the SAE ground-truth alignment score specifically
    (`sae/ground_truth.py::permutation_null_alignment`), since
    `best_ground_truth_matches` picks each feature's *best* of ~30
    candidate ground-truth fields — a real multiple-comparisons inflation
    of `mean_abs_rho_matched` above zero even under pure noise, which a
    permutation null (shuffle feature-to-series correspondence, rerun the
    identical search, repeat `ground_truth_permutation_repeats` times)
    directly measures. Subsamples to `ground_truth_permutation_max_features`
    features per permutation when a dictionary is large, logged rather than
    silent (`ROADMAP.md` sec 15's no-silent-cap precedent). Internals'
    family-probe decodability was **not** given its own permutation null
    this session (it already has a weaker, related majority-class
    chance-line CI) — named explicitly as remaining scope rather than
    folded into "done."
  - **New configs**: `configs/null_timesfm_random.yaml` /
    `null_chronos_random.yaml`, each pairing a real model against its
    `random_init` twin at the exact checkpoint and SAE-target layer
    `configs/medium_run_chronos_base.yaml` already has freshly re-measured
    numbers for (this same day's earlier §6.2 re-measurement session), so
    the null floor is directly comparable to an already-recorded real
    number rather than a fresh, incomparable one. `l3`/`attention`/
    `exemplars`/`clustering`/`confirm`/`layer_screen` disabled in both —
    this run answers one question (how much of L1/L2/internals/SAE is
    architecture-only), not a full flagship re-comparison against a model
    with no learned structure to be causal about.
  - **Verification — unit tests.** New `tests/test_random_init.py` (6
    tests: `random_init_like` against a tiny fully-offline HF T5 config,
    confirming different weights and that `T5LayerNorm`'s lack of
    `reset_parameters` doesn't silently leak pretrained values; a
    no-`.config` module raises; the mock adapters' seed-offset emulation
    produces same-architecture/different-weight/reproducible pairs; a YAML
    round-trip; and a full mock-pipeline `extract`+`l1` run of a
    real-vs-random-init pair producing finite, in-range CKA) and
    `tests/test_ground_truth_permutation_null.py` (4 tests: a planted
    strong real correlation clears the permutation null's p95 while a
    pure-noise aggregate lands in the same ballpark as its own null —
    the two-scenario check that proves the null actually discriminates
    signal from search inflation, not just that it runs; `n_perm=0`
    disables cleanly; feature subsampling works). One real bug found by
    actually running the existing smoke test (not assumed passing):
    `report.py::_note()` takes exactly three positional fields, and a
    fourth item appended to `_SAE_EXEMPLAR_NOTE` collided with the
    `summary=` keyword arg (`TypeError: _note() got multiple values for
    argument 'summary'`) — fixed by folding the new explanation into the
    existing "Limitations" field instead. Full suite green at **183/183**
    (up from 173) after the fix.
  - **Verification — real checkpoints, both runs completed cleanly
    (2026-08-06).** `configs/null_timesfm_random.yaml` and
    `null_chronos_random.yaml` ran end to end against live
    `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-base`.
    Artifacts: `tsfm_model_analysis/tsfm_lens/runs/null_timesfm_random/`
    and `.../runs/null_chronos_random/`. **The result is a genuine split
    verdict, not uniform validation — read all four numbers below, not
    just the reassuring ones:**
    - **SAE ground-truth alignment survives the null cleanly, and the
      permutation-null mechanism itself checks out exactly as designed.**
      Real TimesFM (layer `stacked_xf.18`): mean |ρ| **0.358** vs. its own
      permutation null (mean 0.182, p95 0.193) — clearly above. Its
      untrained twin at the identical layer: mean |ρ| **0.195**, right at
      *its own* permutation null (p95 0.189) — i.e. an untrained model's
      SAE finds no ground-truth-aligned structure beyond pure search
      inflation, exactly as it should. Real Chronos-T5-Base (`encoder.
      block.6`): mean |ρ| **0.397** vs. null (mean 0.183, p95 0.199) —
      clearly above. Its untrained twin: mean |ρ| **0.178**, at/below its
      own null (p95 0.201). This is the cleanest result of the four and a
      strong sanity check on the whole mechanism: it distinguishes real
      from untrained cleanly in both directions it needs to.
    - **Internals' family-probe decodability also survives cleanly.** Real
      TimesFM's probe accuracy across all 10 captured depths: 0.852 →
      0.939 → 0.949 → 0.955 → 0.958 → 0.967 → 0.961 → 0.963 → 0.951 →
      0.905. Its untrained twin: **flat at 0.720 every single layer**
      (probing a network with no learned structure just decodes whatever
      linearly-readable signal survives from raw input statistics, which
      doesn't change with depth). Real Chronos-T5-Base: 0.793 → 0.825 →
      0.855 → 0.904 → 0.904 → 0.898 → 0.912 → 0.955 → 0.954 → 0.964 →
      0.984 → 0.980 across its 12 blocks; its untrained twin: **flat at
      0.665**. Real clearly and consistently beats untrained at every
      single depth, for both models.
    - **L1's CKA floor is architecture-dependent in a way that changes how
      the flagship peak-CKA number should be read.** TimesFM's own
      untrained-twin CKA is high and steeply depth-decaying: 0.599 at the
      shallowest captured layer down to 0.077 at the deepest (its patch-
      based continuous embedding carries a lot of "free" architecture-only
      geometric agreement). Chronos-T5-Base's own untrained-twin CKA is
      much lower and nearly flat: 0.123 → 0.174 (peak, block 7) → 0.152 by
      the last block (its per-timestep quantized-token embedding carries
      far less). The real cross-model peak, previously recorded this same
      day in §6.2's re-measurement session, is **CKA 0.381** at (TimesFM
      layer 4, Chronos block 10). Read against **TimesFM's own** floor at
      its matching layer 4 (0.369), the real number is barely above it —
      the reported "peak" sits almost exactly at what architecture alone
      predicts. Read against **Chronos's own** floor at its matching block
      10 (0.152), the *same* real number is clearly (~2.5x) above it. This
      asymmetry is itself the finding: which model's floor you check the
      shared peak-CKA number against changes the verdict from "barely
      distinguishable from architecture" to "clearly above it." At
      mid-to-late TimesFM depths the real cross-model curve does pull
      increasingly clear of TimesFM's own (fast-decaying) null — e.g.
      layer 8: 0.323 real vs. 0.214 null; layer 18: 0.104 vs. 0.077 — so
      the *shape* of "shared structure persisting with depth" looks real
      even where the single reported peak number does not clearly beat
      the null.
    - **L2's stitching-gain floor is the most sobering result, and
      materially qualifies `CLAUDE.md` §6.5's framing.** TimesFM's own
      untrained-twin best gain-over-baseline: **0.388** (TimesFM→random,
      CI 0.324–0.461) / **0.363** (reverse, CI 0.316–0.412). Chronos's own:
      **0.444** (CI 0.360–0.534) / **0.357** (CI 0.278–0.430). The real
      cross-model gain, previously recorded this same day: **0.318**
      (TimesFM→Chronos, CI 0.293–0.342) / **0.413** (Chronos→TimesFM, CI
      0.360–0.461). These are the same order of magnitude with
      substantially overlapping CIs — the TimesFM→Chronos real gain
      (0.318) is not just "not clearly above" but numerically *below*
      TimesFM's own untrained-twin floor (0.388) at its own best pair.
      Right now, an architecture-only untrained twin achieves a
      stitching gain over the input-feature baseline comparable to (or
      larger than) what two real, differently-trained models achieve
      against each other. This does not mean L2's real number is
      meaningless — it already carries a "not causal" evidence-class
      caveat (`CLAUDE.md` §2.6) — but it is a materially stronger
      qualification than that caveat previously implied, and should be
      read as genuinely open rather than folded quietly into "gain over
      baseline shows shared structure."
    - **What this does not (yet) establish**, stated rather than implied:
      this is one run, one corpus (`benchmark_medium`), one Chronos size
      (base), and only the single global best-pair per run has a proper
      bootstrap CI — every other depth-curve cell above is a point
      estimate. A rigorous verdict on L1/L2 needs a per-layer-pair
      bootstrap test of (real CKA/gain − matched null CKA/gain) with its
      own CI, not eyeballing two point estimates, plus replication across
      corpora and the Chronos-small size already used elsewhere in this
      file. **The single-global-best-pair version of this test is now done
      (2026-08-07 follow-up, immediately below) — the per-layer-pair grid
      version and cross-corpus/size replication remain the open part of
      this item.**
    - **Also not done this session**: wiring these floors into `report.py`
      as an automatic reference line/band the way A13's noise floor was
      wired into the L3 chart — this write-up is a manual comparison
      against two separately-run null configs, not yet a permanent,
      automatic part of every L1/L2 report section. A natural, similarly-
      scoped follow-up once the per-layer bootstrap test above exists.
  - **Follow-up (2026-08-07): the eyeball comparison above is now a real
    significance test, not two point estimates read side by side.** New
    `analysis/null_baseline.py` (`compare_l1_peak_cka`, `compare_l2_best_gain`,
    `run_null_baseline_comparison`) + `run_null_baseline_test.py` CLI +
    `stats.py::bootstrap_ci_diff` (a general paired/unpaired bootstrap CI and
    p-value for the *difference* of two statistics) + `l2_stitching.py`'s
    train/val split factored out into a reusable `series_split()`. Reads two
    already-extracted run directories — no model loaded, no checkpoint
    touched, nothing re-run — and bootstraps the difference between a real
    cross-model run's peak-pair statistic and a null run's own peak-pair
    statistic. **Paired, not independent, bootstrap**: verified (via
    `_series_aligned`, not assumed) that `medium_run_chronos_base` and both
    null runs load the identical 288-series corpus with the identical
    `run.seed`, so `l1`'s row sample and `l2`'s train/val split are
    byte-identical across runs — every bootstrap draw resamples the *same*
    series from both the real and null run simultaneously, cancelling
    shared between-series noise (the same logic `CLAUDE.md` §6.6 already
    applies to L3's cross-model fingerprint-agreement bootstrap, here
    applied cross-run instead of cross-model). Falls back to an unpaired
    bootstrap with a logged warning if alignment ever fails — exercised
    directly in `tests/test_null_baseline.py` via a deliberately
    differently-sized corpus, alongside 4 other tests (paired L1/L2 runs are
    finite and well-formed; the top-level wrapper degrades per-section, not
    wholesale, when one run is missing an artifact). Full suite green at
    **189/189** (up from 183) after adding these.
    - **Numbers reproduce the prior point-estimate reading exactly** (a
      direct correctness check on the new machinery, not just "it runs
      without crashing"): the recomputed null-run peak CKAs (0.5989 TimesFM,
      0.1739 Chronos) and best gains (0.3883 TimesFM, 0.4442 Chronos) match
      the hand-read values recorded above (0.599/0.174 and 0.388/0.444) to
      three decimal places.
    - **L1 peak CKA: the asymmetry is not just "related," it is a
      significant effect in *opposite directions* depending on which
      model's null you check the shared peak number against.** Against
      Chronos's own untrained-twin floor: real CKA 0.3812 vs. null 0.1739,
      diff **+0.207** (95% CI [+0.159, +0.217], p=0.001, paired,
      n_boot=1000) — **the real cross-model number clearly, significantly
      exceeds this floor.** Against TimesFM's own untrained-twin floor: the
      *same* real CKA 0.3812 vs. null 0.5989, diff **−0.218** (95% CI
      [−0.251, −0.180], p=0.001, paired) — **the real number is clearly,
      significantly *below* this floor.** Both are decisive (CIs nowhere
      near zero); they just decide opposite ways. This sharpens, not just
      restates, the "model-asymmetric" language from earlier this section:
      it's not that the null comparison is ambiguous, it's that it gives two
      confident, contradictory verdicts depending on which model supplies
      the architecture-only floor.
    - **L2 best gain: the null result is now confirmed, not just
      suggestive.** Against Chronos's floor: real 0.4132 vs. null 0.4442,
      diff **−0.031** (95% CI [−0.127, +0.061], p=0.474). Against TimesFM's
      floor: real 0.4132 vs. null 0.3883, diff **+0.025** (95% CI [−0.057,
      +0.102], p=0.532). Both CIs comfortably contain zero — an actual
      bootstrap test of the difference, not an eyeballed CI overlap, confirms
      neither null is statistically distinguishable from the real
      cross-model gain. `CLAUDE.md` §6.5's qualification stands as written
      and is now backed by a significance test rather than a point-estimate
      comparison.
    - **Still open, stated precisely so it isn't quietly widened into "L1/L2
      are settled":** this tests only each run's single global best pair,
      on one corpus, one Chronos size. It does *not* test every layer pair
      (the depth-curve comparison at line ~7034-7039 above is still
      point-estimate-only), and it does not replicate across corpora or the
      Chronos-small checkpoint. Extending `bootstrap_ci_diff` to a full
      per-layer-pair grid is mechanical (the function is already
      pair-agnostic); doing so and re-running against a second corpus/size
      is the next concrete step. **The depth-curve half of that step is now
      done (2026-08-07, second follow-up, immediately below); cross-corpus
      and cross-size replication is not.**
  - **Second follow-up (2026-08-07): the depth curve above is now
    significance-tested at every layer, not just the single global peak
    pair — and the result is a real, informative, non-uniform pattern, not
    a blanket "L1 fails the null."** New `analysis/null_baseline.py::
    compare_l1_depth_curve` (+ `run_null_baseline_test.py --depth-curve`):
    for each of TimesFM's 10 captured layers, take its already-computed
    best cross-model Chronos partner (`l1/cka.npz`'s row-argmax, the same
    correspondence the depth-curve point estimates above were read from),
    and bootstrap the difference against *both* models' own architecture-
    only floor **at that exact matching layer** (not the null run's own
    global peak, which — this is itself part of the finding — sits at a
    different depth than the real cross-model peak for TimesFM's side).
    2 new tests (`tests/test_null_baseline.py`), full suite green at
    **191/191**. Ran against the same already-extracted artifacts as the
    peak-pair test (`medium_run_chronos_base` vs `null_timesfm_random`/
    `null_chronos_random`), n_boot=500, still CPU-only, no re-extraction:

    | TimesFM layer | Chronos partner | real CKA | vs. TimesFM's own floor | vs. Chronos's own floor |
    |---|---|---|---|---|
    | xf.0  | block.8  | 0.181 | 0.599 — **null exceeds real** (p=.002) | 0.170 — not clearly different |
    | xf.2  | block.10 | 0.284 | 0.532 — **null exceeds real** (p=.002) | 0.152 — **real exceeds null** (p=.002) |
    | xf.4  | block.10 | 0.381 | 0.369 — not clearly different | 0.152 — **real exceeds null** (p=.002) |
    | xf.6  | block.10 | 0.353 | 0.278 — **real exceeds null** (p=.002) | 0.152 — **real exceeds null** (p=.002) |
    | xf.8  | block.10 | 0.323 | 0.215 — **real exceeds null** (p=.002) | 0.152 — **real exceeds null** (p=.002) |
    | xf.10 | block.9  | 0.309 | 0.189 — **real exceeds null** (p=.002) | 0.169 — **real exceeds null** (p=.002) |
    | xf.12 | block.9  | 0.279 | 0.163 — **real exceeds null** (p=.002) | 0.169 — **real exceeds null** (p=.002) |
    | xf.14 | block.9  | 0.238 | 0.137 — **real exceeds null** (p=.002) | 0.169 — **real exceeds null** (p=.008) |
    | xf.16 | block.8  | 0.145 | 0.105 — **real exceeds null** (p=.002) | 0.170 — **null exceeds real** (p=.002) |
    | xf.18 | block.9  | 0.104 | 0.077 — **real exceeds null** (p=.002) | 0.169 — **null exceeds real** (p=.002) |

    (Every CI is decisive — p=0.002 is the floor at n_boot=500 — except
    xf.4 vs. TimesFM's own floor and xf.0 vs. Chronos's own floor, both
    genuinely CI-spans-zero results, and xf.0 vs. TimesFM's floor / xf.16
    and xf.18 vs. Chronos's floor, which are decisive in the *null-wins*
    direction.)
    - **The headline single-peak-pair test's "not clearly different"
      verdict for TimesFM was itself an artifact of which layer got
      compared against which null value.** The peak-pair test (first
      follow-up, above) compared the real peak (layer 4) against
      TimesFM-null's own *global* peak (layer 0, CKA 0.599) and found null
      exceeds real decisively. This depth-curve test instead compares layer
      4 against TimesFM-null's value **at layer 4 specifically** (0.369) —
      a much weaker floor, giving a genuinely ambiguous (CI-spans-zero)
      result rather than a decisive loss. Both comparisons are individually
      valid; they just answer different questions ("does the real peak beat
      the null's best layer" vs. "does the real peak beat the null at the
      same depth"), and reading only one of them would have been
      misleading either way.
    - **The real, substantive finding: real cross-model structure clearly
      beats *both* architecture-only floors at every middle depth (layers
      6-14 of 10 captured), and clearly loses to at least one floor only at
      the extremes** (shallowest: layers 0/2 lose to TimesFM's own
      steeply-decaying floor; deepest: layers 16/18 lose to Chronos's own
      floor, which is nearly flat with depth and so stays a fixed ~0.17
      bar the real curve eventually decays below). This is a materially
      more informative and more positive picture for L1 than either "the
      peak doesn't survive the null" or "L1 is uniformly null-level" would
      suggest — most of the depth range shows a real, decisive,
      double-sided win over architecture alone; it's specifically the
      curve's two ends that don't.
    - **One CLI display bug found and fixed while writing this up (not a
      correctness bug in the numbers — the JSON's `diff_lo`/`diff_hi`/`p`
      were always right):** `run_null_baseline_test.py`'s print formatter
      collapsed "null decisively exceeds real" (CI entirely negative) into
      the same "NOT CLEARLY DIFFERENT" label as a genuine CI-spans-zero
      result, which would have made e.g. xf.0's clear null-exceeds-real
      result read as ambiguous from the console output alone. Fixed to a
      three-way label (`REAL EXCEEDS NULL` / `NULL EXCEEDS REAL` / `NOT
      CLEARLY DIFFERENT (CI spans zero)`); the table above uses the
      corrected classification, read directly from each row's `diff_lo`/
      `diff_hi` rather than the old printed string.
    - **Still open (at the time of writing):** L2's depth curve has no
      equivalent per-layer test yet (only its single best pair, from the
      first follow-up above); no cross-corpus or cross-checkpoint-size
      replication of any of this exists. **The L2 depth curve is now done —
      third follow-up, immediately below, and it materially changes the
      L2 verdict.**
  - **Third follow-up (2026-08-07, run via the scheduled autonomous loop):
    the L2 analog of the depth-curve test above — and it reverses, not just
    qualifies, the earlier "L2 gain is null-level" reading at the real run's
    own reported best pair.** New `analysis/null_baseline.py::
    compare_l2_depth_curve` (+ `run_null_baseline_test.py --depth-curve
    l2`): for each src layer along the real run's own best direction
    (`l2/stitching.json`'s highest-`best_gain` direction —
    `Chronos-T5-Base->TimesFM` here), bootstrap its gain (predicting its
    own best-matching dst layer) against two same-architecture floors **at
    the matching layer index** — `null_run_a` gives the src model's own
    real-vs-random-twin gain at that exact src layer, `null_run_b` gives
    the dst model's own random-twin-vs-real gain at that exact dst layer
    (mirrors `compare_l1_depth_curve`'s same-index design exactly, not each
    null run's own best pair). Required factoring `_best_direction_
    decomposition` into a lower-level `_pair_decomposition` helper so both
    the single-best-pair and depth-curve paths share one ridge-fit/baseline
    code path. 2 new tests, full suite green at **193/193**. Ran against
    the same already-extracted artifacts (`medium_run_chronos_base` vs.
    `null_chronos_random`/`null_timesfm_random`), n_boot=500, CPU-only:

    | Chronos src layer | TimesFM partner | real gain | vs. Chronos's own floor (same layer) | vs. TimesFM's own floor (same layer) |
    |---|---|---|---|---|
    | block.0  | xf.12 | 0.021 | 0.444 — **null exceeds real** (p=.002) | 0.047 — not clearly different |
    | block.1  | xf.18 | 0.077 | 0.197 — not clearly different | 0.046 — not clearly different |
    | block.2  | xf.18 | 0.138 | 0.128 — not clearly different | 0.046 — **real exceeds null** (p=.002) |
    | block.3  | xf.18 | 0.170 | 0.086 — not clearly different | 0.046 — **real exceeds null** (p=.002) |
    | block.4  | xf.18 | 0.214 | 0.078 — **real exceeds null** (p=.016) | 0.046 — **real exceeds null** (p=.002) |
    | block.5  | xf.18 | 0.277 | 0.076 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |
    | block.6  | xf.18 | 0.361 | 0.075 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |
    | block.7  | xf.18 | 0.375 | 0.073 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |
    | block.8  | xf.18 | 0.404 | 0.069 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |
    | block.9  | xf.18 | 0.409 | 0.068 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |
    | block.10 | xf.18 | **0.413** (this run's global best) | 0.062 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |
    | block.11 | xf.18 | 0.393 | 0.062 — **real exceeds null** (p=.002) | 0.046 — **real exceeds null** (p=.002) |

    - **Why this reverses, rather than merely qualifies, the earlier
      peak-pair reading.** `compare_l2_best_gain`'s single-peak-pair test
      (first follow-up above) compared the real run's best pair (block 10,
      gain 0.413) against `null_chronos_random`'s own *global* best pair —
      which this depth curve now shows sits at **block 0** (self-vs-random-
      twin gain 0.444), an entirely different, unrelated layer where an
      early Chronos block's output is still close enough to the raw input
      that it trivially predicts its own untrained twin's same-index output
      well. Comparing the real cross-model peak (block 10) against that
      irrelevant early-block artifact was comparing apples to oranges. At
      the layer the real signal actually peaks (block 10, matched
      layer-for-layer against the null), the null floor is **0.062**, not
      0.444 — and real decisively, overwhelmingly exceeds it (diff +0.351,
      95% CI [+0.265, +0.429], p=0.002), and likewise decisively exceeds
      TimesFM's own matched-layer floor (0.046, diff +0.367, CI
      [+0.316,+0.421], p=0.002). The exact same pattern that made L1's peak
      look worse than most of its depth range (§16 E9's second follow-up,
      above) made L2's peak look *worse than every layer from block 4 on*.
    - **The corrected picture: real gain decisively beats both floors at
      every layer from block 4 through block 11 (8 of 12), is ambiguous at
      blocks 1-3, and loses only at block 0** (where absolute real gain is
      smallest anyway, 0.02-0.17, and the null's self-predictability
      artifact is largest). This is the opposite shape from the original
      qualification's implication that L2's gain "does not survive the
      null" — at the specific layer pair this file already reports as the
      flagship number, it survives both null floors cleanly and by a wide
      margin. `CLAUDE.md` §6.5 corrected in place (not deleted, per this
      file's own §0.2) to reflect this.
    - **What this does not overturn:** the *other* direction
      (`TimesFM->Chronos`) and its own depth curve haven't been tested this
      way — this result is specific to the direction the real run already
      reports as its overall best, which is the one number this file's
      other sections actually cite. Cross-corpus and cross-checkpoint-size
      replication of this depth-curve result also remain undone, same as
      L1's. `run_null_baseline_test.py --depth-curve l2` makes re-running
      this against a second corpus mechanical once one is extracted.
  - **Fourth follow-up (2026-08-12, cron loop): the reverse direction
    (`TimesFM->Chronos-T5-Base`) is now measured, and it reverses the same
    way the forward one did — the third follow-up's "what this does not
    overturn" caveat is closed.** `compare_l2_depth_curve` gained a
    `direction=` parameter (and `run_null_baseline_test.py` a
    `--direction` flag) so a direction other than the run's own best-gaining
    one can be tested; L2 is not symmetric — the ridge probe maps one
    model's states onto the other's — so this is a separate measurement, not
    a re-read of the same one. Same runs, same n_boot=500, nulls supplied in
    the reverse order (`--null-run-a runs/null_timesfm_random --null-run-b
    runs/null_chronos_random`); artifact `runs/medium_run_chronos_base/
    l2_depth_curve_null_test_TimesFM_to_Chronos-T5-Base.json`.

    | TimesFM src layer | real gain | vs TimesFM's own floor (src) | vs Chronos's own floor (dst) |
    |---|---|---|---|
    | `stacked_xf.0` | 0.2132 | 0.3883 → **null exceeds** (−0.175, p=0.002) | −0.1696 → real exceeds (+0.383) |
    | `stacked_xf.2` | 0.2895 | 0.3841 → **null exceeds** (−0.095, p=0.012) | → real exceeds (+0.459) |
    | `stacked_xf.4` | 0.3106 | 0.3777 → ambiguous (−0.067, CI [−0.151,+0.009]) | → real exceeds (+0.480) |
    | `stacked_xf.6` | 0.3143 | 0.3617 → ambiguous (−0.047, p=0.332) | → real exceeds (+0.484) |
    | `stacked_xf.8` | 0.3178 | 0.3245 → ambiguous (−0.007, p=0.996) | → real exceeds (+0.487) |
    | `stacked_xf.10` | 0.3179 | 0.1179 → **real exceeds** (+0.200, p=0.002) | → real exceeds (+0.488) |
    | `stacked_xf.12` | 0.3121 | 0.0488 → **real exceeds** (+0.263, p=0.002) | → real exceeds (+0.482) |
    | `stacked_xf.14` | 0.2931 | 0.0056 → **real exceeds** (+0.288, p=0.002) | → real exceeds (+0.463) |
    | `stacked_xf.16` | 0.2695 | −0.0381 → **real exceeds** (+0.308, p=0.002) | → real exceeds (+0.439) |
    | `stacked_xf.18` | 0.2354 | −0.1045 → **real exceeds** (+0.340, p=0.002) | → real exceeds (+0.405) |

    - **Against Chronos's own untrained-twin floor the result is
      unambiguous at every depth**: that floor is *negative* at
      `encoder.block.10` (−0.1696 — a random-init Chronos twin predicts the
      real Chronos layer *worse* than the hand-crafted input-feature
      baseline does), and real cross-model gain exceeds it by +0.38 to
      +0.49 with p=0.002 at all ten TimesFM src layers. Note every src
      layer's best dst partner is the same Chronos layer, `encoder.block.10`
      — the same block the forward direction's flagship number sits on.
    - **Against TimesFM's own untrained-twin floor the shape is the same
      depth story as L1's, mirrored:** that floor decays monotonically and
      steeply with TimesFM depth (0.388 → −0.105 across layers 0→18) while
      the real cross-model gain is a shallow hump peaking at 0.318 around
      layers 8–10, so the two curves cross once. Real loses at layers 0–2,
      is statistically indistinguishable at 4–8, and **decisively wins at
      every layer from 10 through 18 (5 of 10)**.
    - **This closes the specific pessimistic claim recorded earlier in this
      block.** The "TimesFM→Chronos real gain (0.318) is numerically *below*
      TimesFM's own untrained-twin floor (0.388)" comparison used the null
      run's own *global best pair*, which — exactly as the third follow-up
      found for the other direction — sits at **layer 0**, where an
      untrained twin trivially predicts its own same-index untrained twin
      because both are still close to the raw input. At matching layer
      index that floor is 0.118 / 0.049 / 0.006 / −0.038 / −0.105 for layers
      10/12/14/16/18, and the real gain beats it decisively at each. Both
      L2 directions have now been tested the same way, and both give the
      same answer: **the wrong-layer artifact, not the L2 gain, was what
      the original single-peak-pair test measured.**
    - **Still open, unchanged:** cross-corpus and cross-checkpoint-size
      replication (both directions), and the same treatment for a third
      architecture. Nothing here is a second corpus.

---

## 17. Gap analysis — the plan as a whole, measured against the north star (added 2026-08-12)

> **Why this section exists.** §0.5 answers "what do I do next." §15 and §16
> answer "what's broken" and "what's missing." Nothing answered **"is the
> plan, taken as a whole, actually pointed at §1's north star?"** This section
> is that read, done once, honestly. It is a diagnosis, not a work list — each
> gap below hands off to the new §18/§19/§20/§21 items that fix it.
>
> **The verdict in one paragraph.** The repo's *analysis depth* is genuinely
> ahead of the published TSFM-interpretability state of the art, and its
> *statistical discipline* (series-level resampling, null baselines, sealed
> confirmation) is ahead of most interpretability work in any modality. But
> measured against "a one-button tool that lets anyone compare **any** TSFM on
> **equal grounds**", three of the four words in that sentence have unfinished
> business: **one-button** is a real, well-planned gap (§16 T1 — known and
> sized). **Any** is a narrower envelope than the plan admits: five adapters
> exist, all five are attention-based transformers over time-localized tokens,
> and the fastest-growing parts of the TSFM landscape (mixers, MoE, SSMs,
> API-only, reprogrammed vision/text models) each break an assumption nothing
> in the plan currently addresses. **Equal grounds** is the largest gap, and
> the one the plan is *least* aware of — it is treated as a set of prose
> caveats in `CLAUDE.md` §12 rather than as a measured, reported quantity, and
> at least one cross-model axis in current use is quietly comparing unlike to
> unlike.

### 17.1 The five structural gaps

**G-I 🔴 The depth axis is not comparable, and nothing says so numerically.**
`utils.py::relative_depths` is `arange(n) / (n - 1)` over each model's
**captured** layers. Two consequences nothing in the repo currently accounts
for:
- For Chronos-T5, captured layers are the **encoder only**. So Chronos's
  "relative depth 1.0" is the *middle* of its computation — the last encoder
  block, with an entire decoder still to run — while TimesFM's 1.0 is its
  actual output. Every figure that interpolates both models onto a shared
  relative-depth axis (L1's CKA depth curves, L3's cross-model fingerprint
  agreement, crystallization depth, the §16 E9 depth-curve null tests) is
  therefore comparing "80% of the way through TimesFM's whole computation"
  against "80% of the way through Chronos's *encoder*." That is not a
  rounding error; for a T5, encoder and decoder are comparable in size.
- With `capture_layer_stride: 2` the axis is over *capture points*, not
  blocks, so two configs of the same model can produce different depth
  coordinates for the same block.
`CLAUDE.md` §12 item 4 states the honest version of this ("relative-depth
interpolation is a convention, not a fact") and then the pipeline proceeds to
use it as if it were one. **Fixed by §18 F1 (candidate depth axes, empirically
compared) and F4 (coverage accounting).**

**G-II 🔴 There is no parameter, compute, or latency accounting anywhere in the
repo.** Verified by grep: no `n_params`, no FLOP estimate, no wall-clock or
peak-VRAM record in any analysis module. So every "TimesFM beats Chronos at
X" is confounded with model size, and there is no way to ask the question a
practitioner actually has — *per unit of compute*, which model wins? §5.3's
chronos-base-vs-small study is the only size control in the file and it varies
size within one family rather than normalizing across families. **Fixed by
§18 F2.**

**G-III The capability matrix degrades per-model, which silently produces
unequal comparisons.** The doctrine (`CLAUDE.md` §2.5) is right — an
unsupported capability skips and logs. But when model A supports
`attention_patterns` and model B does not, the report renders A's attention
analysis and simply omits B's, and a reader sees a rich result for one model
and nothing for the other **in a document whose entire purpose is comparison.**
There is no "compare only what both can do" mode, and no explicit separation
between the symmetric comparison and the asymmetric extras. **Fixed by §18 F3
and F9.**

**G-IV The architecture envelope is narrower than "any TSFM", and the plan
doesn't enumerate what's outside it.** All five adapters are attention
transformers over contiguous time patches. `CLAUDE.md` §12's "envelope hard
edges" names the *categories* that break (non-time-localized tokens,
non-uniform hidden size, multi-pass context) but never names a single real
model that falls into them, so there is no way to tell whether the envelope
excludes two obscure architectures or half the field. It excludes more than
the plan implies: attention-free mixers, MoE routing, state-space models,
API-only services, and vision/text-reprogrammed models are all real, published
TSFMs and all fall outside. **Fixed by §19's landscape table and adapter
tiers.**

**G-V The plan is written for its authors.** This is not a criticism of the
prose — it is dense because the subject is — but §1's north star says
*anyone*, and there is currently no on-ramp: no one-sentence "what does this
stage tell me and why would I care", no glossary independent of the report's
per-figure notes, no worked example that reads a real result end to end, and no
progressive disclosure in the report between "I want the headline" and "I want
the estimator's assumptions." Conversely there is nothing aimed at the
*advanced* reader either — no methods appendix stating each estimator's
assumptions and failure modes in one place, which is what a reviewer or a
sceptical researcher reaches for first. **Fixed by §21.**

### 17.2 What the plan gets right, and should not be "improved"

Recorded explicitly so a future pass doesn't refactor away the parts that are
load-bearing:
- **The evidence-class ladder** (L0 behavioral → L1 geometric → L2 translatable
  → L3 causal-within-model) and the refusal to ever claim cross-model
  causality. This is the single most defensible thing in the repo.
- **The L2 input-feature baseline** and the `random_init` null. Most
  representational-similarity work reports raw similarity; reporting only the
  *gain over a control* is stricter than the field norm.
- **The dev/private confirm split.** Nothing comparable exists in
  interpretability tooling.
- **The series as the resampling unit.** Easy to "simplify" into a bug.
- **`ROADMAP.md` §0.2's append-only Findings discipline.** The reason this
  file can be trusted as a research record at all.

### 17.3 Sequencing — see §0.5

This subsection used to carry its own ordering of the G-I…G-V gaps. **§0.5 is
now the only sequencing in the file**; the one argument worth keeping from here
is the one that decided §0.5's ordering and is stated there:

> G-I and G-II are cheap and **retroactively qualify numbers already recorded**
> (every depth-axis figure, every cross-model MASE comparison). They belong
> *before* the next round of cross-model claims, not after — the same argument
> §16 E9 made for the untrained-weights null, now the most-cited control in the
> file.

G-I is §18 F1 (§0.5 item 2). G-II is F2/F4/F6 — all measured; F9 renders them
(§0.5 item 3). G-III/G-V are report-layer work (§21's J items, §0.5 item 6).
G-IV (architecture breadth) is parked with a per-model un-park trigger (§22.2).
---

## 18. Equal grounds — the fairness contract, made measurable (added 2026-08-12)

> **The principle.** "Fair comparison" in this repo currently means *where
> claims are grounded* (`CLAUDE.md` §14: behavior, gains over baselines,
> within-model causal fingerprints). That principle is correct and stays. What
> it does not cover is **the axes those claims are plotted on and the budgets
> the models are given** — depth, compute, capture coverage, token resolution,
> noise floor, and training exposure. An honest grounding on an unequal axis is
> still an unequal comparison.
>
> **The rule this section establishes:** *every cross-model number must either
> be on an axis both models genuinely share, or must be rendered next to the
> measured size of the asymmetry.* No prose-only caveats. If an asymmetry
> cannot be measured, the claim is downgraded, not footnoted.
>
> **Beginner framing (used verbatim in the report):** "Comparing two forecasting
> models is like comparing two runners on different tracks. Before trusting who
> is faster, you have to know whether the tracks were the same length, whether
> both ran the whole way, and whether one was allowed to train on the course.
> This section measures all three."
>
> Items are **F1–F9**. F1, F2, F4, F6 are 🔴 — they retroactively qualify
> numbers already in this file.

### F1 🔴 The depth axis — four candidate definitions, empirically compared `[x]` — done 2026-08-18

**What's wrong.** See §17.1 G-I. `relative_depths` is index-fraction over
captured layers, so Chronos's 1.0 is mid-computation and TimesFM's is its
output; and the axis moves when `capture_layer_stride` changes.

**This is a research question with several defensible answers, so build the
alternatives and let the data choose** — the same shape as §6.1.1's
layer-selector bake-off, which is the precedent to copy including its
null-controlled scoring.

| Axis | Definition | Argument for | Argument against |
|---|---|---|---|
| **D0 index fraction** (status quo) | `i / (n_captured - 1)` | Free; what every figure uses today | Not comparable across partial capture; stride-dependent |
| **D1 block fraction of the full stack** | `block_index / (n_blocks_total - 1)`, over **all** blocks incl. uncaptured, and over the **whole model** incl. decoder | Stride-invariant; honest about truncation — Chronos's encoder ends at ~0.5, not 1.0 | Assumes every block costs the same; leaves the top half of Chronos's axis empty |
| **D2 compute fraction** | cumulative FLOPs to the end of block *i*, ÷ total forward FLOPs (needs F2) | The most defensible "how much of the computation has happened" reading; handles unequal block cost and embedding/head overhead | Needs a FLOP model per architecture; conflates compute with progress |
| **D3 functional / intrinsic axis** | order layers by a *measured* property, e.g. CKA-to-input (already computed by `internals.py`) or tuned-lens forecast-R² (already computed by `lens.py`), then align models by matching that property's value rather than by position | Aligns models by **what they've done**, not where they are — the only axis that is architecture-agnostic by construction; needs zero new measurement | Circular if the aligned property is also the thing being compared; must be used only for axes *other than* itself |

**Built already (2026-08-13, see Findings).** `analysis/depth_axis.py` supplies
all four axes plus `total_stack_size` and `align_on_axis`; `alignment.depth_axis:
block` is the configured default with `index` retained and labelled legacy;
`ModelAdapter.uncaptured_surfaces()` exists; 13 tests pass.

🔴 **What remains — this is the whole live part of F1, and it is one session.**
1. Wire the eight legacy call sites: `lens.py:141`, `spectral_lens.py:97`,
   `internals.py:89`, `l3_perturbation.py:398/684/686`, `report.py:1909/2461`.
2. Print the axis name in every cross-model depth figure title, and explain it
   in one sentence in that figure's `_note()`.
3. **Run the acceptance test** (below). Until then, every depth figure in the
   repo still renders `index`, and the built axis has moved no number.

Wiring the call sites moves recorded depth *coordinates* (not the underlying
measurements), so it wants the §2.1 treatment — one run under both axes,
diffed — not a flag flip.

**Acceptance criterion.** A cross-model depth figure under `block` shows
Chronos's curve ending at ~0.5 with the region above it visibly unmatched, and
the L1/L3 depth-agreement statistics are recomputed over the overlap only.
Then **re-run §16 E9's depth-curve null tests on the new axis and record
whether any verdict changes** — this is the test that matters, because if a
verdict flips, the old axis was materially misleading and that is itself a
publishable finding.

**Beginner explanation to ship with it.** "Two models can have a different
number of layers, and one may only be half-observable. Plotting both on '0 to
1' hides that. Under the new default axis, a model we can only see half of
occupies only half the plot, so you can see what we don't know."
**Findings — 2026-08-13, the machinery is built and tested; nothing consumes
it yet, so the item stays `[ ]`.** What landed is every deliverable above
*except* the wiring: the axis exists, is correct, is the configured default,
and is exercised by 13 tests — but no figure calls it, so no recorded number
has moved and the acceptance criterion has not been run.

1. **`tsfm_lens/analysis/depth_axis.py`** — all four axes (`index`, `block`,
   `compute`, `functional`), plus `total_stack_size` and `align_on_axis`.
   `depth_axis()` returns a `DepthAxis` dataclass rather than a bare array,
   carrying `axis`, `label`, `basis`, `covers` and `fallback_from`, because an
   axis that quietly degrades to a different axis is indistinguishable from
   the axis you asked for once it reaches a plot (§15 A1's lesson). `compute`
   without F2's per-block FLOPs and `functional` without a caller-supplied
   property both degrade to `block` **and record which axis they degraded
   from**, so the degradation travels with the numbers instead of vanishing
   into a log line.

2. **D2's denominator is the full forecast, not the capture pass** — the one
   design decision in the module that is not mechanical. Normalizing each
   model by its own capture pass gives an encoder-only model a clean 1.0 at
   its last captured block (its measured pass *is* its encoder), so the model
   whose computation is *least* observed reaches the top of the chart while a
   fully-observed model honestly stops short. That is precisely the inversion
   §18 F4's live run exposed on the "Compute completed by depth" chart
   (Chronos 1.0 vs. TimesFM 0.851), and fixing it was F1's to carry.
   `predict.flops` is used when F2 measured one; the capture-pass denominator
   is the explicitly-labelled fallback.

3. **`align_on_axis` refuses rather than clamps.** `np.interp` clamps at the
   edges, so the status quo — grid over [0, 1], interpolate both models —
   extends the shorter-spanning model's endpoint value flat across every grid
   point it never reached and then computes agreement statistics over that
   invention. On the `block` axis an encoder-only model spans roughly the
   bottom half, so roughly **half** of every such statistic was that model's
   last encoder block repeated. The new function grids the overlap only and
   returns the unmatched ranges for a figure to shade;
   `test_align_on_axis_covers_only_the_shared_range` pins the fabrication
   itself (`np.interp`'s clamped tail is asserted constant at 5.0) next to
   the correct behavior, so the test documents the defect rather than only
   the fix. Disjoint ranges return `n_grid: 0` and a note — a finding about
   the run, not a crash.

4. **`ModelAdapter.uncaptured_surfaces()`** added (defaults `{}`, no adapter
   breaks). `ChronosAdapter` and `ChronosBoltAdapter` both **count** their
   decoder blocks off the loaded module via a new shared
   `base.count_blocks_matching`, rather than returning the hardcoded
   `{"decoder": 12}` this section specified — a hardcoded count is correct
   for `chronos-t5-base` and silently wrong for small/large, which is the
   `CLAUDE.md` §11.8 failure mode. Bolt is declared separately rather than
   assumed equal to T5: its decoder is a small quantile head, so
   encoder-decoder models do **not** all lose the same fraction.

5. **`total_stack_size` keeps three losses apart**, because they have
   different fixes and collapsing them hides which is in play:
   `n_blocks_uncaptured_in_captured_surface` (stride loss — a config change
   recovers it), `n_blocks_outside_captured_surface` (architectural — no
   config recovers Chronos's decoder), and `n_blocks_captured`.

6. **A new `mock_encdec` adapter makes the acceptance criterion reproducible
   with no checkpoint** — 4 captured blocks over 4 genuinely uncaptured ones,
   `block` coordinates topping out at **3/7 ≈ 0.43** against `mock_patch`'s
   1.0. Its decoder blocks really run inside `forward` and really fall
   outside the capture regex; a test asserts that removing them changes the
   forecast, so the declaration is not a fiction `total_stack_size` would
   report identically either way. Added as a *new* adapter rather than
   retrofitted onto `MockStepAdapter` so every already-recorded mock number
   stays bit-for-bit reproducible (§2.1) — a zero-length decoder consumes no
   RNG draws, so the three existing mocks are unchanged.

7. **One real bug found by building it** (§2.4 — the mock was written to
   expose the asymmetry and immediately exposed something else). The mocks'
   `default_layer_regex` was the unanchored `blocks\.\d+$`, which `re.search`
   also matches inside `decoder_blocks.0` — so the new mock's decoder was
   being *captured*, and `total_stack_size` reported 12 blocks where 8 exist.
   Anchored to `^blocks\.\d+$`; the three pre-existing mocks match
   identically either way, since their blocks sit at the module root. The
   general shape is worth carrying: an unanchored block regex is safe only
   until a model names a second stack with the first stack's name as a
   suffix, and `search` will not tell you it happened.

8. **`config.py`: `alignment.depth_axis: "block"` is the default**, `index`
   retained and documented inline as legacy — *"what every number recorded
   before 2026-08-13 was measured on"* — so an older run's coordinates stay
   reproducible by setting one field.

**What is NOT done, and why the checkbox stays open.** The eight
`utils.relative_depths` call sites (`lens.py:141`, `spectral_lens.py:97`,
`internals.py:89`, `l3_perturbation.py:398/684/686`, `report.py:1909/2461`)
still use the legacy axis; no figure title or `_note()` names an axis; and
**the acceptance criterion has not been run** — neither the cross-model
depth figure under `block` nor the §16 E9 depth-curve null re-run that this
section calls "the acceptance test that matters". Until that re-run happens,
the honest status is: the axis is built and the old axis is still what every
number in this repo was measured on. Note also that wiring the call sites
will move recorded depth *coordinates* (not the underlying measurements),
so it wants the §2.1 treatment — a run under both axes, diffed — rather than
a flag flip.

**Suite.** `tests/test_depth_axis.py` green at **13 passed**. Pre-item
baseline independently confirmed twice at **393 passed, 2 warnings** (474.65s
and 522.27s). Full suite after this item's changes: **406 passed, 2 warnings
in 437.10s** — exactly the baseline plus this item's 13, with no pre-existing
test changing status. That is the specific thing it had to confirm: the
`^blocks\.\d+$` anchor and `_MockNet`'s new `n_decoder_layers` parameter left
the three pre-existing mocks bit-for-bit unchanged (a zero-length
`nn.ModuleList` draws no RNG, so `head`'s weights are untouched, and the
anchor is a no-op for blocks at the module root). The two warnings are the
same two the baseline carries — an intentional float16 overflow in
`test_nonfinite.py` and a `return`-instead-of-`assert` in `test_smoke.py` —
neither introduced here.

**Findings — 2026-08-18, the call sites are wired and the acceptance test
has been run against real checkpoints — `[x]`.** This closes the item.

1. **The wiring mechanism, and why it needed a new function rather than a
   flag flip at each of the eight legacy call sites.** `report.py` and
   several `analysis/` modules only ever operate against an already-written
   run directory — they never have a loaded adapter to call
   `all_layer_names()`/`uncaptured_surfaces()` on, which is exactly what
   `_block_coords` needs. Rather than special-case each of the eight call
   sites' adapter-availability, `ActivationStore.set_stack_meta`/
   `.stack_meta` (`extraction/store.py`) persist `{"all_layers": [...],
   "uncaptured_surfaces": {...}}` per model into the zarr store's attrs once,
   at extraction time (`extraction/extract.py`), when the model is loaded
   anyway. `depth_axis.py::depth_axis_for_run(axis, store, model,
   captured_layers, adapter=None, ...)` is the one function every call site
   now goes through: it reads `store.stack_meta(model)` and passes
   `all_layers`/`uncaptured_surfaces` into `depth_axis()`, falling back to a
   live `adapter` only if supplied and the store predates this metadata, and
   falling all the way back to `index` (with `fallback_from` recorded) if
   neither is available — a run extracted before this session degrades
   instead of crashing, per §2.5. Wired at `lens.py`, `spectral_lens.py`,
   `internals.py`, `l1_geometry.py`, `l3_perturbation.py` (all previously-
   named legacy call sites), and `report.py`'s cross-model depth figures,
   which now print the resolved axis name and a one-sentence explanation in
   each figure's `_note()` per this item's own item 2.

2. **Real-checkpoint acceptance test, run against `runs/medium_run_
   chronos_base` (TimesFM-2.5-200M / Chronos-T5-Base, the same pair §18 F4
   and several other Findings blocks already use).** All three acceptance
   criteria confirmed:
   - **Chronos's `block`-axis relative depth caps at 0.4782608695652174, not
     1.0** — its last captured encoder block (block 10 of 12, stride 1) sits
     just under halfway through the full encoder+decoder stack, exactly the
     "ends at ~0.5" prediction.
   - **TimesFM's own axis also revealed a previously-invisible loss**: it
     caps at 0.9473684210526315, not 1.0, because `medium_run_chronos_base
     .yaml`'s `capture_layer_stride: 2` means the last of TimesFM's 20 blocks
     is never captured (block 18 of 19 zero-indexed captured positions is the
     last one actually captured) — a genuine, config-recoverable stride loss
     that the old `index` axis's `i/(n_captured-1)` definition could not
     surface at all, since by construction it always reaches 1.0 regardless
     of how much of the real stack was skipped. This is a second, independent
     confirmation of the axis doing its job beyond the headline Chronos
     number.
   - **L1/L3 depth-agreement statistics are recomputed over the true overlap
     only**: `align_on_axis` on the real run reports
     `overlap_fraction=0.5048309178743962` between the two models' `block`
     axes — almost exactly the "roughly half" this item predicted from
     Chronos's ~0.48 cap — and the unmatched upper range is returned for the
     report to shade rather than fabricated via `np.interp`'s edge-clamping
     (the defect `test_align_on_axis_covers_only_the_shared_range` already
     pins in isolation, now confirmed to matter on real coordinates too).

3. **Re-ran §16 E9's depth-curve null tests on the corrected axis — no
   verdict flips, exactly as this item's own §16 E9 cross-reference predicted
   was possible either way.** Before spending compute on the rerun, direct
   inspection of `analysis/null_baseline.py` (`compare_l1_depth_curve`,
   `compare_l2_depth_curve`) showed neither function reads a depth axis at
   all — both match layers purely by captured-layer index/name within one
   model's own architecture, so the depth-axis correction structurally
   cannot change which layers get compared or what their bootstrap
   differences are; it only changes what x-coordinate a result gets plotted
   at. This was confirmed empirically, not just read off the code (§2.4):
   rerunning both null tests against the freshly-extracted store reproduced
   `l1_depth_curve_null_test.json`, `l2_depth_curve_null_test.json`, and
   `l2_depth_curve_null_test_TimesFM_to_Chronos-T5-Base.json` **bit-identical**
   to the previously-recorded artifacts at every layer, both L2 directions —
   no p-value, CI bound, or beats/loses verdict moved. **This is itself the
   finding the acceptance criterion asked for**: the depth-axis correction is
   real and moves recorded *coordinates*, but the specific class of
   conclusion §16 E9 draws (per-layer real-vs-null-floor comparisons) was
   never actually axis-dependent, so nothing published under that item needs
   re-reading in light of this one. A future depth-*located* claim (e.g. "the
   crossover happens at relative depth X") would be the kind that does
   depend on this axis; no such claim exists in the repo today outside the
   already-corrected coverage/compute-completed figures §18 F2/F4 cover.

4. **One real, previously-unknown bug found by running the acceptance test
   on real data rather than mocks — `l3_perturbation.py`'s `patching.json`
   silently mismatched array lengths.** `run_l3` computed one
   `depth_axis_for_run(...)` call per model from `_sensitivity`'s full
   captured-layer list and reused that same `DepthAxis` object for
   `patching.json`'s `rel_depth`/`depth_axis` fields — but `_patching`'s own
   `layers` come from a coarser `layer_stride`-subsampled slice
   (`l3.patching.layer_stride: 2` in `medium_run_chronos_base.yaml`), so
   whenever that stride is `>1` the two lists have different lengths.
   `patching.json["rel_depth"]` therefore silently carried more entries than
   `patching.json["layers"]` (10 vs. 5 for TimesFM, 12 vs. 6 for
   Chronos-T5-Base on this config) — invisible in the mock verification
   (`configs/smoke.yaml`, later `smoke_encdec.yaml`) because those configs
   use `layer_stride: 1`, where the two lists coincide by construction and
   the bug can't manifest. Plotly.js zips mismatched-length `x`/`y` (or
   `z`/`y`) arrays index-for-index rather than raising, so `report.py`'s L3
   "Activation patching: clean → corrupted restoration" curve and the
   per-window/per-horizon-step restoration heatmaps were silently plotting
   every patched layer past the first at the wrong depth coordinate, with no
   error anywhere in the pipeline. **Fixed**: a second, separately-computed
   `patching_depth_axes[model] = depth_axis_for_run(..., patching[model]
   ["layers"], ...)` now feeds `patching.json`, distinct from the
   sensitivity-list axis. **Regression test**
   (`tests/test_l3_patching_depth_axis.py`, 2 tests) confirms the fix both
   ways: `layer_stride=2` now yields `len(rel_depth) == len(layers)` for the
   patched subset, and that the values are genuinely recomputed rather than a
   truncated slice of the sensitivity axis (`rel_depth != sensitivity_rel_
   depth[:n_patched]`); a stride-1 regression guard confirms the two axes
   still agree exactly when they should. Verified the test actually catches
   the bug: temporarily reverted the fix and reran — 1 failed, as expected;
   restored the fix — 2 passed. **Verified fixed on the real checkpoint
   data, not just mocks**: re-ran `l3`+`report` for `medium_run_chronos_base`
   end to end (`python run.py --config configs/medium_run_chronos_base.yaml
   --stages l3,report --force l3,report`, exit 0, 12 sections/39 findings);
   `l3/patching.json` now shows `len(rel_depth) == len(layers)` for both
   models (TimesFM 5/5, Chronos-T5-Base 6/6), and directly parsing the live
   Plotly trace data out of the regenerated `report.html` confirms every
   affected figure's `x`/`y`/`z` arrays now match in length (patching-curve
   scatter traces: 5/5 and 6/6; per-window restoration heatmaps: `x=16,
   y=5, z=5` and `x=16, y=6, z=6`). This is the same class of defect §11.24
   warns about generalizing from — not a version pin, a checkpoint's own
   broken remote code, or a config-fingerprint gap, but a fourth shape: two
   call sites inside the *same* function computing what should be two
   distinct derived quantities from one shared variable, where only running
   against a config whose two source lists actually diverge (a real
   checkpoint's `layer_stride>1`, not any mock's `layer_stride:1`) exposes
   it. Promoted to `CLAUDE.md` §11 as its own numbered trap (§11.32) since
   it is a genuinely new failure shape, not a repeat of an existing one.

5. **Suite and CLI status after this item's changes.** Full suite: **436
   passed, 2 warnings in 844.50s** (up from the pre-item 406 — 30 new tests
   across this item's wiring work, `test_l3_patching_depth_axis.py`, and
   unrelated items landed in the same window; no regression). Live CLI run
   (`medium_run_chronos_base`, real TimesFM-2.5-200M + Chronos-T5-Base
   checkpoints, 8× RTX A5000): preflight 11/11 pass, `l3`+`report` complete
   end to end, exit 0.

6. **What this does not close.** F1's own axis choice (`block` as the
   default) is unchanged and was not re-litigated here — this item closes
   the "wire it and prove it moves nothing wrong" half, not a new bake-off
   over D0–D3. `compute` (D2) still degrades to `block` wherever F2 hasn't
   measured per-block FLOPs for a given model/config, and `functional` (D3)
   still needs a caller-supplied property — neither gap is new, both are
   already named in this item's own table.

### F2 🔴 Parameter, compute, latency and memory accounting `[x]` — done 2026-08-12

**What's wrong.** §17.1 G-II — none of this was measured anywhere in the repo,
so every quality comparison was size-confounded and the first question a
practitioner asks (*per unit of compute, which model wins?*) could not be
expressed at all.

**What landed.** `tsfm_lens/analysis/model_budget.py` (parameter census by
role, forward and full-`predict()` cost, peak VRAM, per-block cumulative
FLOPs), a `budget` pipeline stage with no dependencies (so `--stages budget` is
a valid standalone run), and a report "Cost and capacity" section re-plotting
L0 MASE against measured FLOPs and against parameters. The load-bearing design
choice: FLOPs are **measured** via `torch.utils.flop_counter.FlopCounterMode`,
not hand-derived, so the stage works on an architecture nobody has written an
adapter for yet. Full plan in `ROADMAP_ARCHIVE.md`.

#### Findings — F2 complete (2026-08-12)

The measurement half landed earlier the same session (see §14's entry above:
`analysis/model_budget.py`, 11 CPU-only tests). This block covers the other
half — config, stage, report — and what running it actually surfaced.

**What was wired.**
- `config.py::BudgetConfig` (`enabled: true` by default, `batch: 8`,
  `repeats: 5`, `warmup: 2`, `measure_predict: true`, `predict_repeats: 3`),
  registered on `PipelineConfig` and in `_NESTED`.
- `pipeline.py`: a `budget` stage keyed on `budget/model_budget.json`, placed
  immediately before `layer_screen`. **`deps=[]` deliberately** — a cost
  record needs a loaded model and nothing else, so `--stages budget` is a
  valid standalone run against a checkpoint with no activation store built
  yet; the *ordering* still puts it early so a full run reuses warm models.
  Fingerprint keys are `budget`, `data.context_len`, `data.horizon`,
  `models[*].checkpoint`, `models[*].layer_regex`,
  `models[*].capture_layer_stride` — context length above all, since
  attention cost is super-linear in it and a stale cost number under a
  changed context length is exactly §11.24's failure shape.
- `report/report.py::_sec_budget`, registered as the "Cost / Cost and
  capacity" section between L0 and Screen: a cost table, a **compute-completed-
  by-depth** curve, and the **quality-per-unit-of-compute** panel (MASE vs
  FLOPs and vs parameters, both on log axes).

**Verified live** on `configs/smoke.yaml` end to end (re-run after the
`_fmt_flops` fix below, so the checked-in `runs/smoke/report.html` matches the
code): **12 rendered / 1 skipped** sections (SAE off in that config), **44
findings**, `Cost` row `rendered`, artifact present for both mock models with
roles summing exactly to total and `flops_sanity: plausible` for each. 18
tests green (`tests/test_model_budget.py` 11 + `tests/test_budget_stage.py`
7).

**Three things the run surfaced that reading the code would not have.**

1. **A fixed `G` unit prints a small model as `0.00 GFLOPs`, which reads as
   *free* rather than as *small*.** The mock pair made this obvious
   immediately (`patchy` at 1.22 MFLOPs/series vs `steppy` at 26.75
   MFLOPs/series both rounded to `0.00 G`). Fixed with `_fmt_flops`, which
   picks T/G/M/k by magnitude and keeps two significant figures, and by
   putting both normalized-panel x-axes on **log scale** — the model sizes
   this panel exists to compare differ by orders of magnitude, and a linear
   axis renders the smaller one at the origin. Pinned by a test asserting
   `"0.00 GFLOPs" not in html`. Same class as `CLAUDE.md` §2.5: a number
   that degrades to a wrong reading rather than to no reading.
2. **A21** — the config loader silently drops unknown keys. Found because a
   test asserting a nonsense `budget:` key raises turned out to assert
   something `config.py::_build` does not do. Repo-wide, not F2-specific;
   written up as §15 A21 with a fix plan, and the test now pins the current
   behavior with a pointer there so fixing it fails loudly here.
3. **The A3 stale-artifact guard fired on the first full-config run** and
   refused to proceed — `runs/smoke` dated 2026-08-10 had an `l0` fingerprint
   mismatch from unrelated drift. Working as designed: the guard named the
   three affected stages and the upstream cause rather than silently mixing
   configs. The old run directory was moved aside (not deleted) and
   regenerated clean.

**Acceptance criteria.** Every configured model has a budget record ✅; the
report renders quality-vs-cost ✅; per-block cumulative FLOPs are recorded
from the same measured pass, so F1's D2 axis needs no second measurement ✅
(D2's *consumption* of them is F1's work, not F2's). **Not yet run against
real checkpoints** — the numbers above are mock-model numbers, and the
interesting cost result (TimesFM's deterministic single pass vs Chronos-T5's
`num_samples` × decoder passes in `predict_cost`) needs a live run to state.
That is a `medium_run` away and is the natural next F-wave action.

### F3 Capability-intersection comparison mode — parked (see §22.6) `[ ]`

A "compare only what both models can do" report mode. The asymmetry is real
(§17.1 G-III), but **F9's fairness card renders the asymmetry**, which is the
part that protects a reader; a second rendering mode refines a fix rather than
supplying one. Original write-up in `ROADMAP_ARCHIVE.md`.

### F4 🔴 Capture-coverage accounting `[x]` — done 2026-08-12

**What's wrong.** `CLAUDE.md` §12 item 2's coverage asymmetry — the single most
important caveat on every Chronos claim — existed only as prose, and
`report/coverage.json` is *report-section* coverage, not computational
coverage of each model.

**What landed.** Coverage fractions computed from F2's measured per-block
FLOPs and written to `budget/model_budget.json` under `coverage`; any
depth-located finding about a model under 90% captured FLOPs now carries an
automatic qualifier (`report.py::_qualify_depth_claims`), so the caveat no
longer depends on a reader having read a limitations list. Full plan in
`ROADMAP_ARCHIVE.md`.
**Findings (2026-08-12) — implemented, and the first live run *failed this
item's own acceptance criterion*, which is how two real bugs were found.**

`analysis/model_budget.py::capture_coverage` + `report/report.py::
_coverage_qualifiers`/`_qualify_depth_claims`, 12 tests in
`tests/test_capture_coverage.py` and 1 in `tests/test_meta_report.py`.
**Full `tsfm_lens` suite green after all F4 edits: 338 passed, 2 warnings
in 489.16s** (up from 325 pre-F4).
Measured on `runs/medium_run_chronos_base` (live TimesFM-2.5-200M +
Chronos-T5-Base, RTX A5000), numbers quoted exactly as the artifact writes
them:

| | TimesFM | Chronos-T5-Base |
|---|---|---|
| `block_fraction` | 0.5 (10 of 20, stride 2) | **1.0** (12 of 12) |
| `param_fraction` | 0.4252579280803676 | 0.42186707448695115 |
| `capture_pass_flops` | 59391344640.0 | 774755352576.0 |
| `forecast_flops` | 59391344640.0 | 5398283452416.0 |
| `flops_fraction_of_forecast` | 0.4254943502824859 | **`None`** |
| `capture_pass_fraction_of_forecast` | 1.0 | 0.14351883508993188 |
| `headline_flops_fraction` | 0.4254943502824859 | 0.14351883508993188 |
| `headline_is_upper_bound` | `False` | **`True`** |
| `depth_claims_qualified` | `True` | `True` |

**Three fractions, three different losses — and the flattering one is the
one an encoder-only model scores best on.** Chronos's `block_fraction` is a
perfect **1.0**: the layer regex matches every encoder block, so a
blocks-captured headline would report full coverage for the model this whole
item exists to qualify. `param_fraction` (0.42) and the FLOPs fractions use
whole-model denominators and do see the gap. This is why the headline is a
FLOPs fraction and why `test_encoder_only_regex_is_invisible_to_block_
fraction_but_not_to_the_others` exists.

**Bug 1 — the item's acceptance criterion failed on the model it was built
for.** First live run: the qualifier fired on **TimesFM** and not on
Chronos. `_per_block_flops` resolved **0 of 12** Chronos blocks by name
(`ChronosAdapter.forward()` calls `self._t5.encoder(...)` directly, so
`FlopCounterMode` keys blocks relative to that submodule — hypothesis, not
yet verified), so `captured_flops` was `None`, so every fraction was `None`,
so `depth_claims_qualified` was `False`. The most under-observed model in the
repo was silently exempted from its own qualifier by a missing measurement.
Fixed by adding **`capture_pass_fraction_of_forecast` = `forward.flops /
predict.flops`**, both of which the record already held. It is an *upper
bound* on observed computation (the capture pass also does embedding/norm
work belonging to no captured block, and stride may skip blocks inside it),
and an upper bound is the right instrument for a *gate*: a model failing the
90% bar even optimistically has certainly failed it. `headline_basis` and
`headline_is_upper_bound` record which reading was used, and the rendered
sentence says **"at least ~86%"**, never "~86%" — a bound rendered as a point
estimate is the one way this qualifier could overstate its own precision.

**Bug 2 — the parameter surface double-counted stride loss.** TimesFM
reported *"132.9M parameters (57%) lie outside the 20 blocks this run's layer
regex matched"* while the very next line reported *"10 of 20 matched blocks
are skipped by capture_layer_stride=2"* — 98.4M of that 132.9M **was** the
stride loss, reported twice under two different causes. The regex line's
denominator is now the *matched* blocks, not the captured ones: **34.6M
(15%)**. Each surface names a distinct loss.

**Live acceptance criterion, now met** (verbatim from `report.html`, 12
sections / 31 findings). ⚠️ **Superseded in wording 2026-08-13, not in
substance:** once per-block FLOPs resolved, this stopped being a bound, so
the re-rendered report says **"~86%"** where this quote says **"at least
~86%"**. Same number, same section count, same finding count; the hedge is
gone because the measurement replaced the bound. Kept verbatim as the record
of what the bound rendered as:

> Lens — Chronos-T5-Base: forecast crystallizes at 0.73 of depth (within 10%
> of final MASE 2.15) — within the captured surface only; **at least ~86% of
> Chronos-T5-Base's forward computation is unobserved** (116.4M parameters
> (58%) lie outside the 12 blocks this run's layer regex matched; the captured
> forward pass performs only 14% of the FLOPs of a full forecast at this run's
> decode settings; the remainder (a decoder, or repeated sampled decode steps)
> runs unobserved).

**Two things that fell out and are worth carrying into F1.** The L1 peak-pair
finding named neither model (`at L4 ↔ L10`), so the qualifier could not
attribute it — and a reader could not tell which layer belonged to which
model either. It now reads `at TimesFM L4 ↔ Chronos-T5-Base L10 (relative
depths 0.22 / 0.91)` and carries both fractions (shared preamble emitted
once, surface inventories dropped when two models are named, or the finding
becomes unreadable). **That rendering is now F1's own argument in one line:**
Chronos's "relative depth 0.91" is 0.91 of a stack that performs ≤14% of its
forecast's FLOPs. And `report/meta_report.py` inherits the caveat — it
tabulates crystallization depth and the L1 peak pair across runs, where the
difference is invisible — so its crystallization cell now annotates
*"(of the 42% of this model captured)"* for any model under 90%, with runs
predating the measurement left **unannotated** and a footnote saying that
means unmeasured rather than complete.

**Correction owed to `CLAUDE.md` §12 item 2.** *"For TimesFM we see
essentially the whole computation; for Chronos maybe half"* is now measurably
wrong in **both** halves: TimesFM observes **42.5%** of its forward FLOPs
under the default `capture_layer_stride: 2`, and Chronos's capture pass is
**≤14.4%** of a full forecast at `num_samples: 20` — not half. Corrected in
place there.

**Left open — CLOSED 2026-08-13.** `_per_block_flops`'s Chronos key-prefix
failure is fixed; see §13's now-`[x]` entry for the mechanism (the
unique-suffix fix sketch this paragraph proposed was *unsound* for T5 and was
not used). Live re-run of `--stages budget --force budget` on
`runs/medium_run_chronos_base`, ~13 s with both checkpoints cached:

- **All 12 Chronos blocks resolve**, every one at exactly
  **64562946048.0** FLOPs — a single distinct float across all 12, as 12
  identical T5 encoder blocks should be. `body_total` **774755352576.0**,
  `fraction_of_forward` **1.0**.
- 🔴 **That `1.0` is real, and it was checked rather than accepted.**
  `body_total` does not merely approach `forward.flops`, it *equals* it
  bit-for-bit, which is exactly the shape a bogus key resolution would also
  produce. Confirmed analytically, independent of the counter: at `batch=8`,
  `T=513` (512 + EOS), `d_model=768`, `d_ff=3072`, one T5 encoder block is
  QKVO 19,365,101,568 + attention score/value matmuls 6,467,641,344 + FF
  38,730,203,136 = **64,562,946,048** — the measured value to the last digit.
  `FlopCounterMode` counts only matmul-class ops, and in a T5 *encoder* every
  matmul lives inside a block (embedding is a gather; RMSNorm and dropout are
  elementwise), so a zero remainder is the correct answer, not a symptom.
  TimesFM is the contrast that makes this legible: its `body_total`
  50541363200.0 is genuinely **less than** its `forward.flops` 59391344640.0
  (fraction 0.851), because 8.85 GFLOPs of patch-embedding and output-head
  matmuls sit outside its blocks.
- **The headline number did not move: 0.14351883508993188 before and after,
  bit-for-bit.** `headline_is_upper_bound` flips **true → false**,
  `headline_basis` from *"upper bound: whole capture pass over full-forecast
  FLOPs"* to *"captured blocks over full-forecast FLOPs"*, and
  `flops_fraction_of_forecast` from `null` to that value. Read the
  no-change as the finding, not as a non-event: the old bound was **exactly
  tight**, because Chronos captures at `capture_layer_stride: 1` and no
  counted FLOPs fall outside its blocks. What changed is epistemic status —
  *"at least ~86% unobserved"* becomes *"85.6% unobserved, measured"* — which
  is precisely what the bound was designed to be honest about. A reader who
  expects the magnitude to move is right to; its not moving is the evidence.
- **No regression.** `depth_claims_qualified` stays `true` for both models
  (Chronos 0.1435, TimesFM 0.4255, both under 0.9). TimesFM is identical on
  **107 of 113** flattened keys — the 6 differences are all `timing` min/
  median/max seconds. Its params, FLOPs, all 20 `per_block` values,
  `body_total`, every coverage fraction and `peak_vram_bytes` are unchanged.
- **Warnings, as expected and nothing else.** The old *"could not be matched
  to this model's N blocks"* is gone (0 occurrences); the new same-class
  collision warning is correctly **absent** (the decoder never runs in the
  capture pass, so the encoder is the sole `T5Stack` root); and Chronos's
  qualifier line now reads `observed 14%` with the `at most ` prefix dropped,
  tracking the flag.

**The deliverable, re-rendered and checked** (`--stages report --force
report`, no model loaded). "Compute completed by depth" now carries **two**
traces where it carried one, and no code changed to make that happen — the
chart already looped over whatever models resolved:

| model | points | fraction at first block | at last block |
|---|---|---|---|
| TimesFM | 20 | 0.0425 | **0.8510** |
| Chronos-T5-Base | 12 | 0.0833 | **1.0000** |

🔴 **Read those two right-hand values together — they are F1's argument in a
picture.** Both curves are normalized by *that model's own measured forward
pass*, and for Chronos the measured forward pass **is the encoder**. So its
curve reaches a clean 1.0 at relative depth 1.0 while that point represents
≤14% of a forecast, and TimesFM's honest 0.851 (its patch-embedding and
output head sit outside its blocks) looks like the *worse-covered* model on
this axis when the coverage table says the opposite. The chart is not wrong —
its axis label and blurb both say "this model's blocks" — but it is exactly
the trap F1 exists to remove, and it is now visible rather than hypothetical.
**F1's D2 compute-fraction axis therefore needs the forecast-level
denominator** (`predict.flops`, already measured and already in the same
JSON), not this one. Also confirmed in the same render: 12 sections / 31
findings, unchanged, and the qualifier's `at least ` prefix is gone from all
3 of its occurrences (2 for TimesFM's `~57%`, which never was a bound).

**Suite: 341 passed, 2 warnings, 619.33s** — the 338 of the pre-fix baseline
plus this item's 3, no regressions, both warnings pre-existing.

### F5 Token-resolution parity `[ ]`

**What's wrong.** TimesFM resolves lags no finer than one patch (~32 steps);
Chronos-T5 resolves single steps. `CLAUDE.md` §12 item 5 is right that this is
a real ceiling, not a plotting artifact. Representation comparison already
handles it (everything pools to 32-step windows). **Attention comparison does
not** — the lag-profile taxonomy and periodicity-head analysis operate at each
model's native resolution, so a "Chronos has sharper seasonal-lag attention"
finding is partly a statement about patch size.

**Deliverables.**
- `analysis/attention.py` gains a `resolution_mode: native | matched` knob.
  Under `matched`, every model's lag axis is binned to the **coarsest**
  configured model's token width before any taxonomy or periodicity statistic
  is computed. Both are recorded; `matched` is what cross-model findings may
  cite.
- Report a per-model `finest_resolvable_lag` (= `token_width`) in the fairness
  card, and state in the attention section's `_note()` that a difference
  smaller than the coarser model's token width is not interpretable.

**Acceptance criterion.** Periodicity-head scores computed both ways on a real
run; if the cross-model ordering changes under `matched`, that is recorded as a
finding about the previous result.

**Cost.** ~1 session.

### F6 🔴 Every delta in noise-floor units `[x]` — done 2026-08-12

**What's wrong.** A13 built a repeat-run noise floor; nothing rendered any
delta *relative* to it, so every ΔMASE in the repo was implicitly compared
against zero.

**What landed.** `analysis/stats.py::in_floor_units` plus report wiring, and a
retroactive audit of every recorded ΔMASE. 8 tests
(`tests/test_floor_units.py`). Full plan in `ROADMAP_ARCHIVE.md`.

#### Findings — F6 complete (2026-08-12)

**What was built.** `analysis/stats.py::in_floor_units(delta, floor,
interpretable_ratio=2.0)` plus a companion `format_floor_units(fu)`, consumed by
`report/report.py` through one `_delta_phrase(run_dir, model, delta)` helper
rather than by each call site writing its own sentence. That indirection is the
point, not tidiness: **a delta whose floor was never measured and a delta that
sits below its floor read almost identically if each site phrases it
independently, and those are opposite claims.** So `interpretable` is a
tri-state — `None` (no floor measured for this model), `False` (measured, inside
2×), `True` — and the report's suppression gate keys on `is False` specifically,
never on falsiness. `None` must not suppress anything: suppressing a finding
because of a floor nobody measured is asserting noise on no evidence.

Call sites threaded: attention head ranking (finding **suppressed** when
uninterpretable, with the suppression stated in the section body — a silently
missing finding is the exact failure invariant 8 names), attention MLP-ablation
chart (a dotted `add_hline` at the model's floor, plus a sentence in the note,
for sampling models only), and SAE forecast preservation in both `window` and
`token` granularities. A per-run `_FLOOR_AUDIT` counter emits the audit itself as
a finding, and is reset at the top of `run_report` because the test suite calls
that function repeatedly in one process.

**Acceptance criterion — the count.** Re-rendered `runs/medium_run_chronos_base`
report-only (no stage re-run, no model loaded): **0 of 6 ΔMASE values fall at or
below their own model's repeat-run floor.** The six, verbatim from the rendered
HTML:

| value | model | floor units |
|---|---|---|
| `+0.386` | TimesFM | deterministic → real signal |
| `+0.110` | TimesFM | deterministic → real signal (window) |
| `+0.110` | TimesFM | deterministic → real signal (token) |
| `+1.094` | Chronos-T5-Base | **6.8×** its ±0.160 floor |
| `+3.869` | Chronos-T5-Base | **24.2×** (SAE window) |
| `−0.346` | Chronos-T5-Base | **2.2×** (SAE token) |

So the honest answer to F6's own question is **the record needed no retroactive
correction on this run** — nothing published from it was noise. But the number
that matters most is the marginal one: §16 E15's headline "Chronos token-
granularity ΔMASE is *negative*, reconstruction is net better than the clean
forecast" rests on `−0.346`, which is only **2.2×** the floor — barely past the
2× interpretability bar, on n=24 series. That claim was already flagged as
"worth a skeptical look" in `CLAUDE.md` §13 on intuition; F6 now puts a number on
exactly how thin it is. Read it as directionally confirming the window-broadcast
confound diagnosis (which the `+3.869` window value, at 24.2×, supports
overwhelmingly) and **not** as a measurement of SAE quality.

**Deliberate deferral — L3 patching restoration is NOT in floor units.** The
deliverable above names it, and it cannot be done from the current artifact:
`l3/patching.json` stores `layers/corruptions/rel_depth/windows/window_size/
whole_context_patch/verbose/n_requested/n_realized/limited_by` and **no damage
denominator in MASE units**. Restoration is already a fraction normalized by each
corruption's own clean-vs-corrupted damage, so converting it to floor units would
require changing the L3 stage to persist that denominator, not changing the
report. Stated in the L3 restoration note in the report itself ("read a near-zero
restoration as *not localized here*, not as *below the noise floor*") rather than
left as a silent omission. L0 is deliberately out of scope for a different
reason: its paired-bootstrap CIs are a strictly stronger treatment than a ratio
against a floor.

**Verification.** New `tests/test_floor_units.py`, 8 tests: ratio/verdict on a
sampling model in both signs, deterministic models (any nonzero delta is real,
exactly-zero is not), the tri-state under a missing *and* an empty floor dict,
the four rendered phrase shapes, and three report-level tests against a synthetic
attention fixture (below-floor suppresses the finding *and* says so in the body;
above-floor keeps the finding with its ratio; unmeasured neither suppresses nor
pretends) plus an audit-counter reset test. `tests/test_floor_units.py` +
`tests/test_noise_floor.py` green at 13 passed. Full-suite baseline immediately
before this item: **317 passed, 2 warnings (8:20)**.

### F7 Training-exposure confound — parked (see §22.6) `[ ]`

The confound is real: no cross-model claim accounts for the fact that the two
models saw different (and largely undocumented) pretraining corpora. Parked
because it is archaeology rather than measurement — §6.3's own experience
(establishing Chronos-T5's lineage required reading a model card to discover
that the five sizes share a *recipe* but not initialization) is the evidence.
**H3's memorization probe (§22.7) is the empirical version of the same
question** and is the better instrument if this becomes load-bearing. Original
write-up in `ROADMAP_ARCHIVE.md`.

### F8 Multiple comparisons across models, not just across families `[ ]`

**What's wrong.** L0 Holm-corrects across families. With N models the number of
pairwise comparisons grows as N(N−1)/2 × families × stages, and nothing
corrects across that. Phase 4 shipped five adapters; the moment three models
appear in one config, the current correction is insufficient.

**Deliverable.** Extend the A15 claim registry with a **family-wise error
budget per report**: every registered claim declares its comparison family, and
Holm (or Benjamini–Hochberg for the exploratory sections, since dev findings are
explicitly hypothesis-generating) is applied within it. Render the number of
comparisons made alongside the corrected threshold, so a reader can see the
multiplicity rather than inferring it. The `confirm` stage is unaffected — it is
one-shot by construction, which is precisely why it exists.

**Acceptance criterion.** A three-model config's L0 section states the total
comparison count and corrected alpha; a claim significant under two-model
correction but not three-model is visibly demoted.

**Cost.** ~1 session.

### F9 The fairness card — one page a reader can check `[x]` — done 2026-08-18

**The deliverable that makes this section legible.** A single, always-rendered
report section, auto-generated, listing every measured asymmetry between the
models in this run and which claims it qualifies:

| Axis | Model A | Model B | Asymmetry | Qualifies |
|---|---|---|---|---|
| Parameters / FLOPs per forward | … | … | ratio | all quality claims |
| Captured FLOP fraction (F4) | … | … | ratio | all depth-located claims |
| Depth axis used (F1) | … | … | overlap range | all cross-depth figures |
| Finest resolvable lag (F5) | … | … | ratio | all attention-lag claims |
| Forecast determinism / noise floor (F6) | … | … | ratio | all delta claims |
| Declared training exposure (F7) | … | … | overlap or "undocumented" | all behavioral claims |
| Capability intersection (F3) | … | … | list of asymmetric analyses | the asymmetric section |

Rendered **before** any result section — the fairness statement comes first,
not in an appendix — and emitted as `fairness/card.json` for E5's registry and
E6's `findings.json`. This is also the thing to show a sceptical reviewer
first, and the strongest single argument that this repo's comparisons are
honest rather than merely careful.

**Acceptance criterion.** Every row is populated from a measured artifact (no
hand-written values), and the smoke run renders it for two mock architectures.

**Cost.** ~0.5 session once F1–F7 land; **build the card's scaffold first**,
with rows marked "not yet measured", so each F item has a visible landing slot
and partial progress is legible.

**Findings — 2026-08-18, built as the scaffold-first version this item's own
"Cost" line anticipated, and closed — `[x]`.** F1, F2, F4 and F6 were the
only landed measurements when this item was picked up (F3 and F7 are parked,
§22.6; F5 and F8 remain unstarted `[ ]`), so the card renders four real rows
and three explicit "not yet measured" rows rather than waiting for all seven.

1. **Mechanism.** `report/report.py::_sec_fairness`, registered as the
   **first** entry in the `builders` list with `requires: []` — the one
   section every other section's `requires`-gated skip logic does not apart
   from the fixed "How to read this report" preamble, so it always attempts
   to render regardless of which stages ran. Each row is read straight from
   an already-written artifact: `budget/model_budget.json`'s `parameters`/
   `forward`/`coverage` fields for the Parameters, FLOPs, and Captured FLOP
   fraction rows (F2/F4); `l3/meta.json`'s `agreement.overlap_fraction` and
   `rel_depth`'s per-model last coordinate for the Depth axis row (F1);
   `l0/noise_floor.json` for the Forecast determinism/noise floor row (F6).
   A row backed by a stage that did not run (missing artifact file) renders
   `"not yet measured"` for both models rather than raising or being
   omitted — the same §2.5 degrade-loudly pattern every other builder in
   this file already follows, applied here per-row instead of per-section.
   F3, F5, F7, F8 always render `"not yet measured"` since no landed
   measurement for any of them exists anywhere in the repo yet. Also emits
   `fairness/card.json` (`{"model_a", "model_b", "rows": [...]}`) for
   tooling, per this item's own emitted-artifact requirement.

2. **Acceptance criterion, verified two ways.** "Every row is populated from
   a measured artifact (no hand-written values)" — confirmed by construction
   (every populated cell traces to a `load_json` read of an existing
   artifact key, never a literal) and by inspection of both runs below.
   "The smoke run renders it for two mock architectures" — confirmed: the
   mock `smoke.yaml` pipeline (models `patchy`/`steppy`) renders 13 sections
   (up from 12 pre-this-item) with a populated Fairness card as the first
   section, showing real measured values (Parameters "0.2M"/"0.1M", 2.65×;
   FLOPs 1.22/26.75 MFLOPs, 21.85×; Captured FLOP fraction 98.3%/99.9%;
   Depth axis both capping at 1.000 with 100.0% overlap — expected, since
   the mock adapters have no uncaptured surface; Forecast determinism both
   `"deterministic"`) for F1/F2/F4/F6 and `"not yet measured"` for F3/F5/F7/F8.

3. **Re-verified against the real checkpoint pair** (`runs/medium_run_
   chronos_base`, TimesFM-2.5-200M + Chronos-T5-Base) via a report-only
   rerender (`--stages report --force report`, no model reload, ~3s): the
   card's numbers reproduce every already-published figure for this pair
   exactly — Parameters 231.3M/201.4M (1.15×), FLOPs-per-forward 7.42/96.84
   GFLOPs (13.04×), Captured FLOP fraction 42.5%/14.4% (28.2 pt gap, §18
   F4's Findings), Depth axis caps 0.947/0.478 with 50.5% overlap (§18 F1's
   Findings), Forecast determinism `"deterministic"` vs. `"±0.160 MASE"`
   with asymmetry `"one deterministic, one sampled"` (§18 F6's Findings).
   Nothing here is a new measurement — this is exactly the point: the card
   is a read-through of numbers this file already reported piecemeal, now
   collected into one page a reader sees before any result section, which
   is the whole deliverable this item asked for.

4. **New tests** (`tests/test_fairness_card.py`, 5 tests): the full mock
   pipeline populates the four measured rows and marks the other three
   "not yet measured" rather than omitting them; every row states which
   claims it qualifies; a report-only render against a run with only
   `extract` (no `budget`/`l3`/`l0` artifacts) degrades every row to
   "not yet measured" without raising; and the Fairness section is first
   in `report/coverage.json`'s section list.

5. **Suite and CLI status.** `tests/test_smoke.py` and
   `tests/test_fairness_card.py` both green (5 + 5 passed) immediately
   after the change. Full suite rerun clean afterward: **441 passed, 2
   warnings in 352.63s** (up from F1's own post-item baseline of 436 — this
   item's 5 new tests, no regression, no pre-existing test changing
   status). The two warnings are the same two carried since before this
   item (an intentional float16 overflow in `test_nonfinite.py` and a
   `return`-instead-of-`assert` in `test_smoke.py`).

6. **What this does not close.** F5 and F8 remain genuinely unstarted, and
   F3/F7 remain deliberately parked (§22.6) — their rows are honest about
   that, not a workaround for it. If any of the four ever lands, its row
   becomes real by construction (the same artifact-read pattern extends to
   it) with no further change to this section's own code — that
   extensibility, not just the four rows shipped today, is what "build the
   scaffold first" bought.

---

## 19. Architecture adaptivity — covering the actual TSFM landscape (added 2026-08-12)

> **The goal restated concretely.** Not "support more models" — support more
> *architecture classes*, where a class is defined by **which assumption in
> `CLAUDE.md` §12's envelope it breaks.** Adding a sixth attention transformer
> over time patches costs a session and teaches nothing; adding the first
> attention-free model costs the same session and unlocks a whole family.
>
> **Sequencing rule:** one new *class* per session, chosen to break a different
> assumption, and each one must either (a) work through the existing
> abstractions unchanged — which is evidence the abstraction is right — or (b)
> reveal a shared-infrastructure gap fixed *in the shared module*, never
> per-adapter. Both outcomes already have precedent: Chronos-2 did (b)
> (`CLAUDE.md` §11.21), Sundial did (a) (§11.22). That contrast is the
> abstraction's own test suite.
>
> ⚠️ **Architecture details in the table below are from model cards and papers,
> not verified against installed checkpoints.** Marked `[VERIFY]` per this
> repo's convention. Confirm before writing any adapter — `CLAUDE.md` §11.8 is
> the cost of trusting a documented API.

### 19.1 The landscape, by assumption broken

| Model / family | Class | Breaks | Effort | Value |
|---|---|---|---|---|
| **Chronos-T5, Chronos-Bolt, Chronos-2, TimesFM, Sundial** | attention + time-local tokens | — (in envelope) | done | ✅ shipped |
| **Timer / Timer-XL** `[VERIFY]` | decoder-only patch transformer | nothing | low | low — confirms generality, no new class |
| **Toto** `[VERIFY]` | decoder-only patch, robust per-series scaling | nothing structural; a *scaler* front-end worth studying via E17 | low | medium |
| **Moment** `[VERIFY]` | masked encoder (T5 body), patch | encoder-only capture like Chronos, plus masked (non-causal) reading | low | medium — a second encoder-only point |
| **Lag-Llama** `[VERIFY]` | decoder-only over **lag features**, not contiguous patches | 🔴 **token→time span is not a contiguous interval** — a token mixes many non-adjacent lags | medium | **high** — the first real test of the pooling premise and of E3's refusal path |
| **TTM / TinyTimeMixer (Granite)** `[VERIFY]` | **MLP-mixer, no attention at all** | 🔴 the entire attention stage; head ablation; `attention_info` | medium | **highest** — see G3 |
| **TimeMoE, Moirai-MoE** `[VERIFY]` | MoE feed-forward | routing is a new axis; `mlp_info`/MLP ablation semantics change | medium | **high** — see G4 |
| **Moirai** `[VERIFY]` | masked encoder, **any-variate attention**, multi-patch-size | ~~multivariate axis (E19)~~ **E19 resolved 2026-08-12: univariate-only ratified**, so the any-variate axis is a *documented exclusion*, not a blocker — Moirai is loadable and analyzable on its time axis with its cross-variate axis unmeasured, exactly as Chronos-2 already is. What remains genuinely open is the second entry here: *variable* patch size means `token_width` is not one number | medium-high | medium — no longer gated; the residual work is variable-`token_width`, and quantifying the coverage the excluded axis costs |
| **Mamba/SSM-based forecasters** `[VERIFY]` | state-space, recurrent | patching semantics — state carries across positions, so a within-position patch does not isolate | medium | **high** — see G5 |
| **TimeGPT, hosted/API models** | closed, network-only | weights, hooks, layers — everything but `predict()` | low | **high for practitioners** — see G6 |
| **VisionTS** `[VERIFY]` | reprogrammed vision MAE; series → 2-D image | tokens are image patches over a folded time grid — localized but not in 1-D | medium | medium-high — genuinely novel envelope test |
| **Time-LLM, text-reprogrammed** `[VERIFY]` | LLM body with learned text prototypes | tokens are prototype-mixtures, not time intervals | high | medium — likely a documented refusal |
| **TabPFN-TS** `[VERIFY]` | tabular in-context learner | no time axis in the usual sense; no residual stream over time | high | medium — likely L0-only |
| **PatchTST, DLinear, N-BEATS** | task-trained baselines, not TSFMs | nothing; they are *supervised* not zero-shot | low | **high as controls** — see G7 |

### G1 Adapter capability tiers, declared and enforced `[ ]`

**Problem.** "Adding a model" is currently all-or-nothing: write the full
`ModelAdapter` or get nothing. But most of the value of the table above sits at
partial support, and a practitioner comparing a hosted model wants L0 only.

**Deliverable.** Four declared tiers on `ModelAdapter`, as a class attribute
`tier: int`, validated by `models/conformance.py` (which already exists and
already checks the contract — extend it to check *tier consistency*, i.e. a
tier-2 adapter must actually provide everything tier 1 requires):

| Tier | Requires | Unlocks |
|---|---|---|
| **0 — black box** | `predict()` only | L0, calibration (E10), horizon-resolved metrics (E12), context/horizon sweeps (E20), agreement (H4), cost (F2 partial) |
| **1 — observable** | + `forward()`, `token_time_spans()`, residual layer regex | internals, lens, L1, L2, L4, layer screen, SAE |
| **2 — steerable** | + `token_patch`-compatible blocks (single-pass context) | L3 patching, steering (E14), feature ablation |
| **3 — decomposable** | + `attention_info`/`mlp_info`/`attention_patterns` | attention taxonomy, head/MLP ablation, path patching (E18) |

Every report states each model's tier in the fairness card (F9), and F3's
intersection mode operates on tiers as its coarse-grained input.

**Acceptance criterion.** A tier-0 stub adapter (nothing but `predict`) runs
end-to-end to a report containing L0, calibration, horizon metrics and cost, and
skips everything else with a stated reason — no crash, no empty section.

**Cost.** ~1 session, and it makes G6 nearly free.

### G2 Auto-adapter — see §16 E3, not duplicated here `[ ]`

E3's span discovery plus `GenericHFAdapter` is the mechanism that makes half the
table above reachable without hand-written code, **and its refusal path is what
correctly excludes Time-LLM/TabPFN-TS rather than silently mis-analysing them.**
One addition to E3's spec from this section's perspective: `discover_spans`
should classify its result into E3's diffuseness score **plus a contiguity
verdict**, because Lag-Llama is the case where spans are *sharp but not
contiguous* — a distinct failure mode from diffuse, and one that should
degrade to tier 0 with a specific message ("tokens are time-localized but not
to contiguous intervals; the pooling premise does not hold"), not be silently
coerced into an interval.

### G3–G7 — parked (see §22.2)

All five are well-specified and none is cheap; together they are the ~8-session
"architecture breadth" investment. Parked 2026-08-18 because they build for
demand that does not exist — no specific model has been requested — while G1
above stays live precisely because it makes each of them cheaper later. Full
original write-ups in `ROADMAP_ARCHIVE.md`; un-park trigger (a real model
someone wants analyzed, one class per session, G6 first) in §22.2.

| Item | Class it unlocks | The load-bearing idea, in one line |
|---|---|---|
| **G3** | attention-free mixers (TTM/TinyTimeMixer) | Replace "read the attention matrix" with a **perturbation-measured mixing profile** — perturb window *i*, measure the change at window *j* — which is architecture-agnostic and would also give the existing five adapters a second, independent view of what their attention maps claim |
| **G4** | MoE (TimeMoE, Moirai-MoE) | Routing is a new first-class axis: which experts fire for which family, and expert-level rather than MLP-level ablation |
| **G5** | state-space / recurrent | 🔴 A within-position patch does not isolate, because state carries across positions — patching semantics must be re-derived before any L3 number means anything |
| **G6** | hosted / API-only (TimeGPT) | Nearly free once G1's tier 0 exists: `predict()`-only models get L0, calibration, horizon metrics, sweeps, H4 agreement |
| **G7** | supervised baselines (PatchTST, DLinear, N-BEATS) | Controls, **not competitors** — a task-trained model is the ceiling a zero-shot claim should be read against |

## 20. New capability proposals (added 2026-08-12; triaged 2026-08-18)

> **What's live here and what isn't.** Twelve items (H1–H12) were proposed on
> 2026-08-12, ordered by (value × cheapness) ÷ risk. The 2026-08-18 triage kept
> four in the live queue and parked the rest with un-park triggers in §22 —
> including H8, which is the strongest parked item in the file and the natural
> replacement flagship if the crosscoder ends as a negative result (§22.7).
>
> **Live:** H4 (do first), H1, H9, H12.
> **Parked:** H2, H3, H5, H6, H7, H8, H11 → §22.5 and §22.7.
> **Deduplicated:** H10 was always "see §21" — it is J4, and only J4 now.
> **Three proposals were rejected before any were written down**; that list
> moved to §22.8 so a rejection-with-a-reason lives with the parked items
> rather than in a preamble.

### H4 ⭐ Cross-model agreement as a reliability signal `[ ]` · live

**Do this first of the four — ~0.5 session, zero new forward passes.** Pure
reduction over `predict()` output L0 already holds. For each series, compute
inter-model forecast disagreement (pointwise and distributional), then test
whether disagreement predicts error. If it does, "run two models and check
whether they agree" becomes an actionable reliability heuristic for
practitioners, and a *use* for the comparison rather than only a study of it.
Report the correlation with CI, per family and per horizon step (E12's axis),
plus a calibration curve of disagreement→error so a user can read a threshold
off it.

**Acceptance criterion.** Correlation with a series-bootstrap CI, and an
explicit comparison against the obvious cheaper baseline — each model's *own*
quantile width. If self-reported uncertainty predicts error just as well,
cross-model agreement adds nothing, and that must be said.

### H1 ⭐ The model-family scaling ladder `[ ]` · live

**The cheapest genuinely novel study available in this repo.** Chronos-T5 ships
as tiny → mini → small → base → large: a five-point, same-architecture,
same-training-corpus, same-tokenizer size ladder. §5.3 used two of those points
once, as a size *control*. Used as a *ladder*, it answers a question nobody has
published for TSFMs: **which interpretability properties scale with size, and
which don't?**

Run the identical config at all five sizes and plot against parameter count:
crystallization depth (relative and absolute), effective dimensionality per
layer, family-probe decodability, L1 peak CKA against a fixed reference model,
L2 stitching gain, L4 clustering AMI, SAE dead-feature rate and ground-truth
alignment, calibration error, and MASE. Every one of these is already computed;
this is a sweep harness plus a report section, not new analysis.

**Why it's high-value.** Scaling behaviour is the most legible result type in
ML, and disagreements are the interesting part — §5's base-vs-small entry
already found L1/L2 *growing* with size while L4 clustering AMI *shrank*, on
two points. Five points turn that from a curiosity into a curve. It also
directly serves the practitioner question "is the bigger checkpoint worth it?",
in cost-normalized terms now that F2 has landed.

**Deliverables.** `configs/scaling_ladder_chronos.yaml` (five run dirs is
cleaner than five model blocks, given VRAM), a `run_scaling_ladder.py` cross-run
reducer built on the existing `report/meta_report.py` (which already aggregates
N run dirs — extend, don't duplicate), and a report section plotting
metric-vs-size with per-point CIs.

**Acceptance criterion.** Monotonicity stated per metric with CIs, and any
non-monotone metric flagged rather than smoothed. **Explicitly report which
metrics are flat** — a flat curve is the finding that a metric doesn't measure
capability.

**Cost.** 1 session of harness + 5 extraction runs (tiny/mini are cheap; large
is the constraint). Depends on nothing. Per `CLAUDE.md` §2.8 the five
extractions are background work, not foreground.

### H9 Analysis card export `[ ]` · live, after E6/F9

A one-page, citable Markdown/PDF card per analyzed model: identity and
checkpoint digest, tier, F9's fairness rows, headline L0/calibration/cost
numbers, coverage, confirmed-vs-exploratory findings, corpus digest, library
versions, and the run's git SHA. Trivially built on E6's `findings.json` +
F9's `card.json`, and it is what someone actually attaches to a paper or a
decision memo. **Cost:** ~0.5 session.

### H12 Deterministic-replay and result-provenance mode `[ ]` · live

An honesty feature aimed squarely at reproducibility problems this repo has
already paid for three times (`CLAUDE.md` §11.13's golden-hash mystery,
§11.24's config-meaning drift, §11.25's stale zarr store). One flag records,
per run: every library version, the git SHA, CUDA/driver versions, the resolved
config *after* defaults, corpus digest, every RNG seed actually drawn, and a
digest of the activation store — written to `provenance.json` and rendered in
the report. Then `run.py --verify-provenance <run_dir>` re-checks the current
environment against it and reports every difference.

**Why it earns a live slot:** §11.24 cost a full investigation session that a
single "these two runs differ in these 3 ways" diff would have closed in a
minute, and that failure mode gets *more* likely as background-agent workflows
(`CLAUDE.md` §2.8) become the norm. **Cost:** ~1 session. Composes with A3's
config fingerprint and A7's provenance record (both already built) — extend
them rather than adding a parallel mechanism.

### H2, H3, H5, H6, H7, H8, H11 — parked

Full write-ups preserved in `ROADMAP_ARCHIVE.md`; the parking reason and the
un-park trigger for each are in §22.5 (H5/H6/H7/H11) and §22.7 (H2/H3/H8). In
one line each:

| Item | What it is | Cost | Parked because |
|---|---|---|---|
| **H8** ⭐ | The seasonality circuit — a minimal sufficient head/MLP set, necessity + sufficiency | 2 sessions + E18 | Strongest parked item; the successor flagship, not a side quest (§22.7) |
| **H3** ⭐ | Memorization / verbatim-recall probing, with a matched-synthetic control | ~1.5 sessions | A different paper (§22.7) |
| **H2** ⭐ | The practitioner recommender | ~1.5 sessions | The one item that can be confidently wrong in a way a reader acts on; needs E4 (§22.7) |
| **H7** | Distribution-shift envelope — where does each model *stop* working | ~1 session | First out of the parked group; tier-0 compatible (§22.5) |
| **H5** | Temporal-position and phase representation | ~1 session | Descriptive; sharpens no existing claim (§22.5) |
| **H6** | Fine-tuning plasticity — which layers move | ~0.5 session | Free only if §6.3.1 Option E happens (§22.5) |
| **H11** | Checkpoint-trajectory analysis | ~0.5 session | Blocked on external checkpoint availability `[VERIFY]` (§22.5) |

## 21. Two audiences — a beginner on-ramp and an advanced methods spine (added 2026-08-12)

> **The problem (§17.1 G-V).** This repo's output currently assumes its reader
> already knows what CKA is, why a stitching gain over a baseline matters, and
> what "relative depth" means. That reader exists and is well served. Two others
> are not: the practitioner who wants to know which model to use and whether to
> believe the answer, and the sceptical expert who wants each estimator's
> assumptions stated in one place before reading any result.
>
> **The design principle: one document, layered — not two documents.** Two
> documents drift (this repo has already paid for that with `CLAUDE.md` vs.
> `ROADMAP.md` staleness, corrected repeatedly in both files' headers). Layering
> means every claim has a one-sentence plain reading, a normal-depth reading,
> and a methods reading, and the reader chooses the depth.

### J1 The three-layer claim contract ✅ `[x]` — DONE 2026-08-18

Extend E6's `Finding` dataclass with three text fields instead of one:
- `plain: str` — one sentence, no jargon, no numbers beyond one. *"TimesFM
  handles trending data better than Chronos here."*
- `text: str` — the current register: effect size, CI, correction, evidence
  class. (What exists today.)
- `caveat: str` — auto-composed, not hand-written: the evidence class, whether
  registered or exploratory, whether it cleared its noise floor (F6), which
  fairness rows qualify it (F9), and coverage (F4).

The report renders `plain` at headline size, `text` beneath, and `caveat` in the
existing collapsed `<details>` mechanism. **`caveat` must be generated** —
`CLAUDE.md` invariant 8's lesson is that author discipline decays while
generated text does not. **Cost:** ~1 session, together with E6.

**Findings — 2026-08-18.** Built by a background agent (`CLAUDE.md` §2.8),
independently re-verified rather than taken on its report alone (§2.4) —
and that verification caught something real: the agent's own claimed example
triples were quoted from a *stale* `runs/smoke/report/findings.json`
(timestamped ~47 minutes before its own final `report.py` edit — an earlier
iteration's artifact, not the final code's output), which on first read
looked like every finding was silently missing `plain`/`caveat` entirely.
Re-running the report stage fresh (`python run.py --config configs/smoke.yaml
--stages report --force report`) resolved it: the regenerated
`findings.json` has non-empty `plain` and `caveat` on all 45 entries, and the
example triples match what the agent quoted almost exactly (one cosmetic
difference: `l1.1`'s depth axis read `index` on the fresh run vs. the agent's
own `block`, because this store predates `ActivationStore.stack_meta` and
`depth_axis_for_run` falls back with a logged warning — expected, unrelated
to this item). Re-ran `tests/test_smoke.py` + `test_finding_caveats.py` +
`test_capture_coverage.py` directly: **24 passed**. Rendered-HTML counts
independently confirmed via direct grep on the regenerated `report.html`:
`class="finding-plain"` / `class="finding-text"` / `Caveats` each appear
**45/45/45** times, matching the finding count exactly.

- **`plain`**: all 36 call sites got a genuinely jargon-free one-sentence
  gloss (e.g. l1's peak-CKA finding → *"At their most similar layers, patchy
  and steppy organize the data closely alike."*; l2's stitching gain →
  *"You can predict what steppy is doing from what patchy is doing better
  than you could from the raw input alone — real shared structure, not just
  both models seeing the same data."*). Full list of all 36, by stage and
  `report.py` line number, is in this session's transcript rather than
  reproduced here in full — spot-checked a sample of 6 against their
  source `.text` and found no invented numbers or unsupported claims.
- **`caveat`**: a single generated post-pass, `_compose_caveats`, mirroring
  the existing `_qualify_depth_claims` pattern (a whole-list pass over
  already-built findings, not per-call-site authored text). Composes up to
  four clauses per finding: (1) a fixed sentence per `evidence_class`
  (geometric/translatable/causal_within_model/descriptive/illustrative/
  behavioral); (2) registered vs. exploratory; (3) a `cleared_noise_floor`
  clause when set (only 2 of 36 call sites populate it — the attention
  head-ablation finding and the report-level `_FLOOR_AUDIT` summary — left
  `None` elsewhere per §2.5's "don't infer what wasn't actually checked");
  (4) fairness (F9, `fairness/card.json`) and coverage (F1/F4) qualifiers on
  behavioral or depth-worded findings, reusing the same artifacts
  `_qualify_depth_claims` already reads for `.text`.
- **Two real bugs found and fixed during the agent's own verification pass**
  (both confirmed still-fixed on my independent regeneration): (1) the three
  "Noise floor —" findings (which *define* the noise floor) were getting a
  contradictory "no floor check was run for this number" clause appended
  under their own defining sentence — fixed by excluding text matching
  `"noise floor —"` from the ΔMASE-floor-missing clause; (2) a depth-worded
  finding at **100% axis overlap** (this smoke config's actual value) was
  getting a "the two models' depth axes only partially overlap (100.0%
  overlap)" clause — self-contradictory — fixed via a new `_is_full_overlap`
  helper suppressing the clause at ≥99.9% overlap. Both are the kind of bug
  only live inspection of real rendered output catches, not a spec read —
  consistent with this repo's own "verify empirically" doctrine (§2.4).
- **Full suite**: **466 passed, 0 failed** (per the agent's own completed
  run — not independently re-run in full given the ~7-minute cost and that
  the 24-test targeted re-run above already covers every file the agent
  said it touched). 3 test files needed fixing for the dataclass's new
  required fields (`test_capture_coverage.py`'s fixture helper,
  `test_smoke.py`'s HTML-block-extraction logic since the caveat's own
  nested `<div>` changed where the old "first `</div>`" split landed), plus
  one new file (`test_finding_caveats.py`, 7 tests) proving `_compose_caveats`
  is genuinely a function of structured fields, not per-instance prose (two
  `Finding`s differing only in `registered` produce different caveats, etc.).
- **Not done, correctly out of scope**: J2 (`stage_docs.py`) and J3
  (glossary/worked example) — separate queue items, next.

### J2 "What this tells you" — one page per stage `[ ]`

For each of the 13 stages, four fixed lines, written once and rendered both in
the report and in the docs: **Question** it answers · **How** in one sentence,
no formulae · **What a good/bad result looks like** · **What it cannot tell
you.** The fourth line is the one that matters and is the one most tooling
omits. Source of truth: a single `stage_docs.py` dict consumed by both the
report and the docs, so they cannot drift. **Cost:** ~1 session; mostly writing,
and much of the content already exists scattered across `_note()` blocks —
consolidate rather than re-author.

### J3 Glossary and the worked example `[ ]`

- A glossary of the ~25 recurring terms (window, relative depth, evidence class,
  crystallization depth, stitching gain, fingerprint, effective dimensionality,
  noise floor, tier, coverage fraction, dead feature, …), each in one sentence
  plus a pointer to the figure where it is used. The report's per-figure
  `_note()` blocks are contextual and stay; this is the lookup table for someone
  reading out of order.
- **One worked example, end to end**: a real report, read paragraph by paragraph
  in the docs — *this* is the number, *this* is why the CI matters here, *this*
  is the caveat that changes the conclusion, *this* is the finding I would not
  act on and why. This teaches the reading skill nothing else in the repo
  teaches, and it doubles as the regression test for whether the report is
  actually legible. **Cost:** ~1 session.

### J4 Progressive disclosure in the report (= H10) `[ ]`

A three-position control at the top: **Headline** (fairness card, L0, confirmed
findings, recommendation) · **Standard** (today's report) · **Methods**
(everything plus per-figure estimator details, seeds, sample counts, null
comparisons). Implement as CSS classes toggled by one small inline script — no
new dependency, and the file stays self-contained (an existing hard constraint).
Default to **Standard** so no current reader's experience changes. **Cost:**
~0.5 session.

### J5 The advanced methods appendix `[ ]`

One always-rendered appendix stating, per estimator, the things a reviewer asks
for and no current document collects: the estimator and its exact form (linear
CKA, ridge with which regularization selection, banded DTW with which band), the
resampling unit and design (cluster/paired/held-out-series), what the null is
and why *that* null, the known failure modes, and the assumption that would
invalidate it. Roughly one paragraph each for CKA, RSA, ridge stitching, the
tuned lens, MASE/sMAPE/pinball, the bootstrap designs, AMI, catch22 features,
SAE fidelity, and `relative_decoder_norm`.

**This is the cheapest credibility item in the whole file** — the content is
already known to whoever wrote each module, it is currently distributed across
module docstrings, and its absence is the first thing a sceptical reader
notices. Generate it *from* the module docstrings where possible so it cannot
drift. **Cost:** ~1 session.

### J6 Failure-mode gallery `[ ]`

A short, permanent section showing what each analysis looks like **when it goes
wrong**, using cases this repo already has on record: the misleading pre-fix
alignment check (`CLAUDE.md` §11.16), the flat whole-layer patching curve that
resolved under per-window patching, an SAE with a 98% dead-feature rate, a
depth-agreement figure on the old index axis vs. F1's block axis, a delta below
its own noise floor. Every one of these is a real artifact already in `runs/`.

Nothing else in the repo teaches a reader to be suspicious of a plausible-looking
plot, and every one of these fooled someone here first — which is exactly the
evidence that makes the gallery worth having. **Cost:** ~0.5 session.

---

## 22. Parked — possible future features, and why they are parked (added 2026-08-18)

> **Why this section exists.** By 2026-08-12 this file held four competing
> orderings of the same work (§0.5's tiers, §16's T1–T4, §17.3's sequencing,
> §22's six waves) across ~60 proposed items, and a session picking the repo up
> cold had no way to tell an item that corrects a recorded number from an item
> that serves a hypothetical future user. The four orderings are now collapsed
> into **one queue in §0.5**, and everything that did not earn a place in it
> lives here, with the reason.
>
> **Parked is not abandoned.** `[-]` in §0.3 means abandoned-with-a-reason and
> is not used here. Every item below is still a good idea; each is parked
> because at least one of these is true:
> - it costs multiple sessions and corrects no recorded number,
> - it serves an audience the repo does not have yet (a stranger with a
>   checkpoint), or
> - it depends on something unbuilt, unavailable, or externally undocumented.
>
> **Each entry names its un-park trigger** — the concrete event that should
> move it back into §0.5. That is the whole value of this section: it converts
> "someday" into a condition you can check.
>
> **The filter applied, stated once.** An item earns a place in the live queue
> if it (a) retroactively qualifies a number already recorded in this file, or
> (b) is cheap (≤1 session) *and* produces a statement the repo currently
> cannot make, or (c) unblocks a named next action. Everything else parks. This
> is the same test §20's own preamble applied when it rejected three proposals
> before writing any down; it is now applied to §20's own contents too.

### 22.1 Crosscoder variants beyond V2 — V3, V5, V6

**Parked:** V3 (explicit shared/private parameterization, ~150 lines + a
block-size sweep, 2–3 days), V5 (frequency-aware dictionary, V5a half a day /
V5b two days), V6 (Matryoshka / multi-resolution, 2–3 days). Full original
write-ups: `ROADMAP_ARCHIVE.md`.

**Why.** §6.2.1's pre-registered decision rule has already fired against V1,
with a bootstrap CI off zero on the *losing* side (V1 0.2787 vs V0 0.3204,
diff −0.0417, CI [−0.0693, −0.0137], p=0.002 at 46,382 rows). The rule's own
escape clause — *"if no variant beats V0, that is the result"* — is now the
reading the data supports. Each of V3/V5/V6 is an attempt to reverse a result
the repo pre-committed to accepting, at 2–3 days each, and none of the three
addresses the *specific* failure mode V1 exhibits. V2 (BatchTopK) does, which
is why V2 alone stayed in the queue.

**Un-park trigger.** V2 flips clause 2 (beats V0 on `gt_alignment_shared` with
a CI excluding zero). If a jointly-trained dictionary can be made to win at
all, the ceiling is worth chasing and V5 goes first — it is the only one of
the three with a TSFM-native rationale rather than an imported-from-NLP one,
and the benchmark labels seasonality exactly, so it is the only one this repo
is uniquely positioned to test. V3 and V6 stay parked even then unless V2's
scorecard says specifically that the split is still artifact-dominated (→ V3)
or that feature absorption is visible (→ V6).

### 22.2 Architecture breadth — G3, G4, G5, G6, G7

**Parked:** the mixing-profile abstraction for attention-free models (G3),
MoE routing as an axis (G4), state-space patching semantics (G5), black-box /
hosted models (G6), supervised baselines as controls (G7). ~8 sessions total
as sequenced in the original Wave E.

**Why.** §19's landscape table is a genuinely useful *diagnosis* — it names
the real models (TTM, Lag-Llama, SSMs, API-only services) that fall outside
the envelope instead of naming abstract categories, and it stays in §19 for
exactly that reason. But building for them is engineering against demand that
does not exist: nobody has asked this repo to analyze a mixer, and the five
integrated adapters already cover the architectures the repo's founding
research questions are about. G1 (declared-and-enforced capability tiers)
stayed in the queue because it is cheap and it makes every one of these
cheaper later; the rest is speculative platform work.

**Un-park trigger.** A specific model someone actually wants analyzed. Then
take exactly the one class it breaks, one per session, G1 first. G6
(black-box) is the cheapest and would come first on that trigger, since
tier-0 (`predict()`-only) analyses — L0, calibration, H7's shift envelope —
already work without any capture surface.

### 22.3 The adoptability product — E1, E4, E5, E24

**Parked:** the zero-config entry point (E1), the bundled versioned reference
corpus (E4, which also carries an unresolved licensing decision), the results
registry (E5), real-corpus activation bucketing (E24). ~4+ sessions as a
bundle, and E1 depends on E4.

**Why.** This is the north star's "one button", and it is real product work
for a user who does not exist yet. The repo currently has one user, who has a
working five-command path and knows what every flag does. §22's own closing
argument (preserved below in §22.9) is the strongest case *against* doing this
now: a one-button tool removes the expert who would have known not to believe
an unequal axis — so the legibility and equal-grounds work has to land first
regardless, and by the time it has, the value of automating the five commands
is much lower than the value of whatever else that session could have done.
**Kept in the live queue instead:** E7b (install docs — ✅ done 2026-08-18;
E7a's console script was already `[x]`, so both are now closed) and E6's
`findings.json` — ✅ also done 2026-08-18 (see E6's own Findings); both J1's
generated caveats and H9's analysis card can now consume it.

**Un-park trigger.** A second person tries to run the pipeline. That is not a
figure of speech — it is the event that makes E1's value real and that will
also surface, in ten minutes, which parts of E1 actually matter.

### 22.4 Expensive scale runs — E20b, E22a, E22b

**Parked:** the context-length sweep (E20b — expensive by nature: every
context length is a fresh extraction), full-scale corpus validation (E22a) and
full-scale analysis runs (E22b, the literal `full_multidomain.yaml` at ~99K
samples, ~6 h build).

**Why.** Each is GPU/wall-clock-bound rather than thought-bound, and none of
them changes a conclusion — they tighten CIs on conclusions already drawn at
`medium_run` scale. That is the lowest-value use of a session in the file:
maximum cost, zero new statements.

**Un-park trigger.** A finding whose CI is genuinely too wide to act on, or a
writeup that needs one headline number at full scale. Then run exactly that
one, unattended, per `CLAUDE.md` §2.8 — never as a session's foreground work.

### 22.5 Cheap-but-non-corrective studies — H5, H6, H7, H11

**Parked:** temporal-position/phase representation (H5, ~1 session),
fine-tuning plasticity (H6, ~0.5 session but only free *if* §6.3.1 Option E's
fine-tuned children exist), distribution-shift envelope (H7, ~1 session),
checkpoint-trajectory analysis (H11, blocked on whether any integrated family
publishes intermediate checkpoints — `[VERIFY]`, not designed-around).

**Why.** All four are well-specified and none is expensive. They park on the
filter's clause (b): each adds a new descriptive result to a report that
already has thirteen sections, and none sharpens a claim already made. H7 is
the closest call — locating where the model ordering *reverses* is genuinely
decision-relevant, and it is tier-0 compatible — so it is first out of this
group.

**Un-park trigger.** H7: any claim of the form "model A is better" that
someone wants to act on. H6: §6.3.1 Option E happening for its own reasons.
H11: a family shipping intermediate checkpoints. H5: E13's spectral lens
producing a result that phase would explain.

### 22.6 Deferred rigor items — F3, F7

**Parked:** capability-intersection comparison mode (F3), training-exposure
confound estimated rather than waved at (F7).

**Why.** F3 proposes a "compare only what both models can do" report mode. The
asymmetry it addresses is real, but F9's fairness card — which stayed in the
queue — *renders* the asymmetry, which is the part that actually protects a
reader; a second rendering mode is a refinement of a fix rather than a fix.
F7 depends on training-corpus facts the model publishers largely do not
document; §6.3's own experience (verifying Chronos-T5's lineage required
reading a README to discover the sizes share a *recipe* and not
initialization) is the evidence that this is archaeology, not measurement.
H3's memorization probe (§22.7) is the empirical version of the same question
and is the better bet if the question becomes load-bearing.

**Un-park trigger.** F3: a run where a capability asymmetry actually changes a
headline conclusion, rather than merely making one section one-sided. F7: a
publisher documenting a training corpus precisely enough to compute overlap.

### 22.7 The bigger studies — H2, H3, H8, E18, E21, §6.3.1's Options A/B/D/E

Each of these is a *project*, not a task. Parked as a group because the repo
should finish the crosscoder writeup before starting another multi-session
research thread — but two of them are the natural successors, so the reasoning
matters more here than elsewhere.

- **H8 · The seasonality circuit (2 sessions + E18).** 🔴 **This is the
  strongest parked item in the file, and the natural replacement flagship if
  the crosscoder lands as a negative result.** It asks for a *minimal
  sufficient set* of heads/MLPs whose patching restores period detection, with
  necessity and sufficiency as the acceptance pair. It is on firmer ground than
  the crosscoder was: within-model causality (no cross-model transplant, so
  invariant 5 is untouched), a ground-truth period known exactly for every
  synthetic series, a candidate set the periodicity-head taxonomy already
  supplies (so no blind search), and per-window patching already built. It
  would close `CLAUDE.md` §12 item 6 — the stated limitation that nothing is
  component-level below head/MLP granularity — which is the single biggest
  gap between this repo's mechanistic claims and transformer-lens-era work.
  **Un-park trigger:** Stage 4's writeup is done, or the crosscoder's
  negative result makes a new flagship necessary. E18 (path patching) is its
  only real dependency and should be scoped as part of it, not before it.
- **H3 · Memorization / verbatim-recall probing (~1.5 sessions).** Novel,
  cheap, and it inverts machinery the repo already owns (the leakage auditor's
  banded DTW, pointed the other way). Its matched-synthetic control is what
  makes it a measurement rather than a claim, and it is the only empirical
  handle anywhere on `CLAUDE.md` §4.1's distributional-leakage tier. Parked
  only because it is a different paper. **Un-park trigger:** F7 becoming
  load-bearing, or a reviewer asking whether the benchmark's leakage story is
  empirically supported rather than argued.
- **H2 · The practitioner recommender (~1.5 sessions).** Parked with the
  sharpest reason in this section, taken from its own risk note: it is *the
  one item in the file that can be confidently wrong in a way a reader acts
  on*. It also depends on E4's fixed corpus for comparability, which is itself
  parked (§22.3). **Un-park trigger:** E4 lands *and* there is a real user to
  be wrong at. Its refusal path is not optional — build the refusal before the
  recommendation.
- **E21 · Chronos decoder capture as a separate measurement.** The largest
  *measured* limitation in the repo: F4 established that Chronos-T5-Base's
  captured blocks perform 14.35% of a forecast's FLOPs, so ~86% of that model
  is unobserved. The design constraint is already settled and is the binding
  one: pooled decoder states live on *forecast* time and must **never** be
  CKA'd against context-window states — it is a second measurement, not a
  symmetry fix. Parked because it is a real capture-surface build and every
  current claim already carries F4's automatic qualifier. **Un-park trigger:**
  a Chronos-side claim that a reader cannot act on without the decoder, or a
  reviewer treating the coverage qualifier as disqualifying rather than
  bounding.
- **§6.3.1 Options A/B/D/E · provenance detection.** Option E (fine-tuning
  `chronos-t5-small` into labelled children) is the ground-truth prerequisite
  every other option needs, and it is real GPU training for a use case that
  was already falsified once (§6.3). **Option C stayed in the live queue** —
  black-box, architecture-agnostic, zero new forward passes, and it passes the
  architecture control by construction, which is precisely what killed §6.3's
  original method. **Un-park trigger:** Option C showing signal. Don't build
  the ground truth for a detector that doesn't detect.

### 22.8 Rejected outright (do not re-propose)

Kept from §20's original preamble, because a rejection with a reason is worth
more than the space it takes:

- **Bigger probe families (MLPs, transformers) on intermediate
  representations.** A stronger probe answers "is the information present"
  more *permissively*, which weakens every decodability claim. Linear probes
  are the right tool precisely because they are weak.
- **t-SNE / UMAP galleries of activation space.** `CLAUDE.md` §5's single most
  important rule already forbids quantitative use of embedding coordinates,
  and a gallery invites exactly that.
- **An LLM-written narrative summary of the report.** The report's value is
  that every sentence traces to an artifact. Generated prose breaks that and
  cannot be audited.

### 22.9 The one ordering constraint worth keeping (preserved verbatim)

From the original §22, because it is the argument that decided §22.3 and it
should survive the section that replaced it:

> Do not run Wave F before Wave A. A one-button tool that hands a stranger a
> depth-axis figure comparing 80%-of-TimesFM against 80%-of-Chronos's-encoder,
> with no coverage number and no noise floor, is worse than the current
> five-command tool — because it removes the expert who would have known not to
> believe it. That is §16's own argument about §15's P1 items, applied one level
> up: **equal grounds is a prerequisite for automation, not a refinement of
> it.**

Generalized, now that most of Wave A has landed: **legibility and equal
grounds gate automation, not the other way round.** F1's remaining call sites
(§18) and J5/J2/J6 (§21) come before anything in §22.3.
