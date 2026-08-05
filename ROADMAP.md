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
> **Corrections already made against the source brief this file was seeded
> from (and now also fixed in `CLAUDE.md` itself):** `benchmark_validation`
> lives at `tsfm_benchmark/benchmark_validation/` (nested, confirmed via
> `pyproject.toml`'s package list) — `CLAUDE.md`'s `[VERIFY]` guess of a
> sibling location was wrong. The `CO_trev_1_num` outlier-swamping issue
> flagged as an outstanding `QuantileTransformer`/IQR-ranking fix in
> `CLAUDE.md` §5 **has been fixed, but by a different, already-landed
> mechanism** — `RobustScaler` output winsorized to ±`clip_scaled` robust-scale
> units, with per-sequence anomaly diagnostics surfaced in `plot.py`/
> `report.py` — not the originally-proposed `QuantileTransformer`/IQR-ranking
> approach; `CLAUDE.md` §5 now describes the actual mechanism. The SAE phase
> is still exactly a stub (`sae/interface.py`, a `Protocol` and a
> `NotImplementedError` — no training code exists anywhere yet), which is
> expected: it's `Phase 2` below and hasn't started. Also newly discovered
> and fixed in `CLAUDE.md`: the TimesFM adapter has moved to TimesFM 2.5
> (different checkpoint, layer count, and a newly-supported
> `attention_patterns` capability) — see `CLAUDE.md` §11.8.

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

### 6.1 Phase 2a — Does anything predict "this layer is worth interpreting"? — ✅ first pass DONE 2026-08-05 (superseded as a *selector* by §6.1.1; kept as a cross-model *finding*)

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

### 6.1.1 Phase 2a-v2 — a better, architecture-agnostic, all-layers-fair layer selector — ✅ implemented + first bake-off run DONE 2026-08-05; not yet wired into production config

> **Status.** Design written 2026-08-05 (below), implemented and empirically
> bake-off-tested the same day (`tsfm_lens/analysis/layer_screen.py`,
> `layer_screen_bakeoff.py`, `run_layer_screen_bakeoff.py`,
> `tests/test_layer_screen.py` — 12 new unit tests, all passing, plus the full
> 33-test `tsfm_lens` suite re-run clean). The bake-off ran for real against
> live checkpoints, not mocked data — see the Findings block after Idea C
> below for the full result. **Deliberately not wired into `sae.targets:
> auto` or any config knob yet**: the result is genuinely mixed (no single
> method cleanly wins both models; see Findings), and per §2.2's own
> discipline a mixed first-pass result should be read as a real, provisional
> signal — not rushed into production the way §6.1's `recommend_layers` was
> trusted before it had actually been tried on one real model. Treat every
> "should"/"expect" in the design sections below as the hypothesis it was
> when written; the Findings block after Idea C is what actually happened.

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
  the two are never confused.

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

### 6.2 Phase 2b — A TSFM-native SAE variant (the flagship research thread) — baseline (item 4) ✅ DONE 2026-08-05; crosscoder (item 1) not started

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
  **crosscoder (item 1) not started this session** — it needs its own
  design work (joint training stability is explicitly an open question,
  §13), not a small extension of the baseline. The encode-store pass
  (writing back into `sae/{model}/{layer}` so L1/clustering could read a
  `level="sae"`) is **also not wired** — `sae/interface.py`'s docstring
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
  Findings below. The "vs. sparsity (L0)
  frontier" half is **partial**: TopK's sparsity is architectural (exactly
  `k` per row by construction), so there's no sparsity dial to sweep the
  way a ReLU+L1 SAE would need — a genuine frontier plot would need
  multiple runs at different `k`, not done this session, noted as
  follow-up rather than silently equated with "not applicable."
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
- [ ] Feed the SAE evaluation results back into §6.1's layer-selection
  correlation study (the retroactive validation step noted there). — **not
  done**; needs SAE runs across enough (run, model, layer) combinations to
  add as a fourth proxy the way §6.1's other three were pooled, which this
  session's two-target demo run doesn't yet provide at meaningful n.
- [ ] Verbose-mode reporting (§4): a per-feature exemplar panel — top
  activating series/windows for a feature, its ground-truth alignment score,
  and (once Phase 3 feature ablation exists) its causal effect on forecast
  when zeroed. — **not done**; the `sae` stage currently writes only
  `sae/meta.json` (no report-integrated section yet). Genuine follow-up,
  not attempted this session — report integration for a fundamentally new
  artifact type is its own real piece of work, distinct from getting the
  underlying numbers to exist at all (this session's actual scope).

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
    **Not fixed, minor, noted rather than silently left:** `run_sae` calls
    this once per target, so a multi-target run re-fetches and re-pools
    the same catalog redundantly (this run fetched Monash twice, once per
    model) — cheap to fix by hoisting the fetch above the per-target loop,
    not done this session.
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

### 6.3 Phase 2c — L2 stitching as a distillation / fine-tune detector (brief item 2)

**Goal.** Test whether the repo's existing L2 stitching-gain machinery (and
L1 CKA) can distinguish "model B was fine-tuned or distilled from model A"
from "model B was trained independently," with a real IP/provenance-detection
use case in mind. This is separable from — and doesn't block — 2a/2b.

**Concrete deliverables**
- [ ] Design a **ground-truth-labeled experiment**, since no public registry
  of "known-distilled TSFM pairs" is likely to exist: fine-tune (or
  distill, if a teacher-student recipe is feasible) a small model from a
  pretrained checkpoint as the **positive** case, and separately train a
  same-size, same-architecture model from scratch (or take an existing
  independently-trained checkpoint of matching size) as the **negative**
  control. Different checkpoint sizes of the *same* released family
  (e.g. `chronos-t5-small` vs. `chronos-t5-base`) are a plausible proxy
  positive if they share meaningful pretraining lineage — verify what's
  actually documented about their training relationship before assuming it,
  don't guess.
- [ ] Compute L1 peak-CKA and L2 stitching-gain-over-baseline for both the
  positive and negative pairs, using the exact same benchmark corpus and
  identical bootstrap discipline (`CLAUDE.md` §6.6) as the main comparison.
- [ ] The claim worth testing precisely: does the *gain over the
  input-feature baseline* (not raw CKA/R², which two same-domain-trained
  independent models could share plenty of) separate positive from negative
  pairs with a CI that excludes overlap? State the null explicitly (§2.2):
  independently-trained same-size models might already show substantial
  stitching gain just from sharing training-data statistics, in which case
  this signal is weaker evidence of lineage than of "same era, same data"
  and that distinction needs to be reported honestly, not oversold.
- [ ] If the signal holds up, write this as a standalone method note (its
  own doc or a clearly separated section here) describing the intended
  use case (provenance/IP disputes) and its actual false-positive risk
  in plain terms — this has real-world stakes if anyone downstream acts on
  it, so §2.6 (`CLAUDE.md`)'s "never let a correlational number be read as
  causal" discipline applies doubly hard here: this method establishes
  *representational similarity*, not legal derivation, and any writeup must
  say so explicitly.

**Findings / decisions**
- *(append here)*

---

## 7. Phase 3 — Ablation and bias-characterization studies (brief item 3)

**Goal.** Beyond the head/MLP ablation and corruption-patching already in
`CLAUDE.md` §6.5, run *designed* experiments that isolate specific biases —
"is TimesFM better at high-frequency structure, Chronos at statistical
means" is a concrete brief question (§1) that deserves a controlled answer,
not just an inference from aggregate archetype MASE.

**Concrete deliverables**
- [ ] **Controlled parameter sweeps** using `build_pipeline`'s `parametric`
  generator directly (not `random_parametric`): hold every component fixed
  except one (e.g. sweep seasonal period from 4 to 256 timesteps holding
  amplitude/noise/trend constant; sweep noise SNR holding structure
  constant; sweep intermittency rate from 0 to 80%). Plot each model's MASE,
  crystallization depth, and L3 noise-fingerprint response as a function of
  the swept parameter — this directly answers "where is each model's
  sweet spot" with a dose-response curve instead of an archetype-level
  average.
- [ ] Follow up on the one concrete causal lead already in hand (this file's
  §5 Findings, carried forward 2026-08-05 from the now-retired `CLAUDE.md`
  §10): additive-noise depth-sensitivity is **strongly
  anti-correlated** between the two models (ρ=−0.90). Design a sweep (SNR ×
  depth) specifically to characterize *why* — is it continuous embedding vs.
  quantization, as hypothesized, or something else? This is exactly the kind
  of "further understand the inner workings and biases" the brief asks for,
  and there's already a strong, specific, previously-observed effect to
  chase rather than starting from nothing.
- [ ] **Feature-level ablation** (depends on Phase 2b): zero individual SAE
  features (or small groups) via `token_patch` on the reconstruction and
  measure forecast impact, cross-referenced against each feature's
  ground-truth alignment score (§6.3) — do features that align well with
  "trend order" actually matter causally for trend-dominant series and nothing
  else? This is the sharpest test of whether the learned features are real
  computational structure or just descriptive correlations.
- [ ] Verbose-mode case studies for every sweep (§4): don't just report a
  dose-response curve, show 2–3 concrete series at the extremes of the sweep
  with their forecasts and lens curves side by side, narrated.
- [ ] Consolidate bias findings into a **per-model "bias card"**: a compact,
  plain-language summary (a few sentences plus 2–3 supporting plots) of what
  each model is systematically better/worse at and under what conditions —
  this is the artifact that most directly answers the brief's original
  research questions and should be prominent in the top-level report, not
  buried in a stage-specific section.

**Findings / decisions**
- *(append here)*

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
- [ ] **Chronos-2** (if released/available) — same lineage as the existing
  Chronos adapters, likely the lowest-effort addition and a good smoke test
  of the adapter-add process itself before tackling something structurally
  different.
- [ ] **Moirai (uni2ts)** — masked-encoder, multivariate-native, patch-based;
  structurally distinct from both existing adapters (handles multivariate
  series and variable patch sizes natively) — a good test of whether the
  alignment/pooling machinery (`CLAUDE.md` §6.3) holds up against a model
  whose tokenization isn't a simple fixed patch width.
- [ ] **Sundial** — worth checking whether it's decoder-only or
  encoder-decoder and what its probabilistic head looks like (flow-matching,
  last checked) — likely needs its own thinking about how `predict()`
  and quantiles map onto the existing contract.
- [ ] Others worth scanning for fit against the `ModelAdapter` contract as
  they mature: Toto, Time-MoE, Tiny Time Mixers, Lag-Llama — don't commit to
  these without first checking checkpoint availability and license terms,
  the same diligence already applied to benchmark data sources
  (`CLAUDE.md` §4.3).

**Concrete deliverables per model (repeat this checklist per addition):**
- [ ] Subclass `ModelAdapter`, register in `models/__init__.py`.
- [ ] Run `--discover-layers`, pick `layer_regex`, run `--check-alignment`
  and confirm near-1.0 diagonal dominance at early layers before trusting
  anything (`CLAUDE.md` §6.3, invariant 7).
- [ ] Fill in the capability matrix (`CLAUDE.md` §6.2) honestly — which
  optional capabilities (`attention_patterns`, `cross_attention_patterns`,
  head/MLP info) this model supports, and confirm unsupported ones degrade
  with a log rather than breaking (`CLAUDE.md` invariant 8).
- [ ] Confirm §2.4's success bar: no changes needed outside the new adapter
  file and its registration. Any exception is a bug in the abstraction — file
  it as a Phase 4 finding and fix the abstraction before moving to the next
  model.
- [ ] Auto-generate the capability matrix from the registry (see Phase 5)
  rather than hand-maintaining a table that will drift.

**Findings / decisions**
- *(append here)*

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
- [ ] **Auto-generated model-zoo capability matrix** (from Phase 4) rendered
  as part of the docs, not hand-maintained.
- [ ] **Tutorial notebooks**: at minimum, (a) run the smoke config end to
  end, (b) add a new toy adapter from scratch, (c) train and interpret an
  SAE on one layer, (d) read a confirm-stage verdict correctly.
- [ ] **CI**: golden-hash regression (`CLAUDE.md` §11.1) plus the smoke test
  (`CLAUDE.md` §9) on every change; a lightweight adapter-conformance test
  that any new `ModelAdapter` can be run against (discover-layers,
  check-alignment, degrade-gracefully checks) so Phase 4's per-model
  checklist becomes partly automated.
- [ ] **Flip `report.verbose` default to `false`** (§4) now that the tool is
  closer to release — verbose stays available, just opt-in.
- [ ] **Packaging**: decide on PyPI/versioning for `tsfm_lens` (currently
  only `tsfm_benchmark` has a root `pyproject.toml`); `tsfm_lens` has its own
  `pyproject.toml` one level down — reconcile these into a coherent
  install story (one repo, plausibly two installable packages, document
  which).
- [ ] Revisit and either implement or explicitly defer-with-reason the
  remaining `CLAUDE.md` §13 future-work items not already folded into this
  roadmap (Chronos decoder capture, CI diversity gates, more source
  adapters, real-corpus activation bucketing, deeper component resolution).

**Findings / decisions**
- *(append here)*

---

## 11. What's already done (do not re-litigate, per `CLAUDE.md` §7 invariants)

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
  checkpoint activations (see §6.2's Findings for what they found). The
  flagship crosscoder (item 1) and the encode-store-into-`sae/{model}/{layer}`
  seam are still exactly where they were — not started.

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
- Claiming cross-model causal patching (`CLAUDE.md` invariant 5) — the
  crosscoder work in §6.2 changes *what's compared* (shared features) but
  does not license transplanting activations between models; that stays
  within-model only.

---

## 13. Open questions / risks (append as they arise; mark resolved in place)

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
  real, diagnosed (not yet fixed) weighting flaw. Combining methods
  (union/vote/rank-average) did not beat the single best method on either
  model. Still open: whether this holds on a second Chronos size, a third
  architecture family, and with the (more expensive) per-window-patching
  secondary gold instead of the cheaper sensitivity-only proxy used here —
  not yet promoted to a config knob pending that broader check.
- [ ] Is a joint crosscoder actually trainable/stable across two
  architecturally distinct models at a shared alignment window, or does the
  representational mismatch (even at peak CKA) make joint training degrade
  into one model dominating the dictionary? Needs an early small-scale test
  before committing significant compute (§6.2).
- [ ] Does the distillation-detection signal (§6.3) actually separate from
  "same training era, similar data" confounds, or is that confound
  unavoidable with publicly available checkpoints? May require training
  the controlled positive/negative pair from scratch to know for sure.
- [ ] Licensing/access for Moirai, Sundial, and other Phase 4 candidates
  should be re-checked at the time Phase 4 actually starts, not assumed
  stable from whenever this section was written.

---

## 14. Session log

> Append one entry per session that makes non-trivial progress on this
> roadmap. Keep entries short — a few lines. This is a changelog, not a
> diary; the *content* of what was learned belongs in the relevant phase's
> Findings block above, this log just tracks when and by whom.

- **2026-08-03** — Initial roadmap authored (planning only, no implementation
  this session). Verified current repo state against `CLAUDE.md`'s
  description: corrected the `benchmark_validation` location `[VERIFY]`,
  confirmed the SAE phase is still exactly a stub. Sequenced the brief's four
  ideas plus its five specific suggestions into Phases 0–5 with an explicit
  dependency rationale (§3).
- **2026-08-03 (same day, follow-up)** — Went back and updated `CLAUDE.md`
  itself (not just this roadmap) with everything found stale during the
  verification pass above, plus a few things found on a closer look:
  the `benchmark_validation` tree location, the actual (not originally
  planned) `CO_trev_1_num` winsorization fix mechanism plus two new
  per-group diversity/redundancy functions, the TimesFM adapter's move to
  TimesFM 2.5 (new checkpoint, halved layer count, and a newly-supported
  `attention_patterns` capability — added `CLAUDE.md` §11.8 as a trap entry
  since a pinned dependency's own breaking release caused this, not a
  choice made in-repo), two new L3 corruptions (`warp`, `dropout`), and
  stale `examples/` → `example_runs/` CLI paths. This file's own
  `QuantileTransformer`/IQR wording from the initial pass was corrected to
  match, above.
- **2026-08-03 (Phase 1 §5.1/§5.2 implementation session)** — Picked up the
  roadmap's own next step: unblock and verify the "wired but unverified"
  real-data path, and widen the benchmark's real-world data beyond the one
  Monash-only `mixture` task that had ever actually been exercised. Set up
  a dedicated conda-forge environment (`tsfmPy`) for the build/validation
  half of the repo (no torch/CUDA needed here), and in the process of
  actually running things — not just reading the code — found and fixed
  four independent real breakages (a `datasets>=3` removal of script-based
  loading that made Monash entirely unloadable, a full `tsbootstrap` API
  rewrite, `sequential_par` being a literal stub, and a source-injection
  bug in `run_full.py` that only `mixture` happened to avoid), plus a
  short-real-series `block_length` crash, plus an unrelated native
  numpy/MKL ABI crash that would have broken `benchmark_validation` on any
  fresh conda-forge setup. Implemented `sequential_par` against SDV's
  `PARSynthesizer` for the first time. Widened `configs/full_multidomain.yaml`
  to use all three real-derived generators against **two** independent real
  corpora (Monash's full 32-domain catalog, not a 10-domain cap, plus ETT),
  added `configs/real_data_smoke.yaml` for fast verification, added
  `tests/test_real_derived_generators.py`, and ran a full
  `run_full.py` → `run_validation.py` build end-to-end against live data for
  the first time (see §5's Findings block for exact numbers and every
  discovered issue). Also surfaced a new, unresolved risk: the golden-hash
  bit-exact-reproducibility regression test fails against numpy 2.1.0 for
  reasons unrelated to this session's code changes — flagged as follow-up,
  not resolved. A full-scale (not smoke) real-data build, and the
  GPU-for-`sequential_par` question, are both explicitly left for the next
  session (see §5's Findings for why).
- **2026-08-03 (same-day follow-up session — §5.1's next concrete step)** —
  Picked up exactly where the previous entry left off: ran
  `benchmark_validation` on a real, non-smoke-scale build for the first
  time. Before sizing that build, calibrated instead of guessing twice: (1)
  isolated `sequential_par`'s actual per-call cost driver by holding a real
  Monash cohort fixed and varying only `epochs` — found the previous
  session's "lower epochs" diagnosis was wrong, the real cause was
  unbounded training-sequence length (a 40,720-point real series was being
  fed whole into a per-timestep RNN fit), fixed with a new
  `max_train_length` parameter (~150-200x speedup, ~1000s → ~5-6s per fit);
  (2) ran a `--max-count 300` throughput calibration on the full config
  before committing to a build size, found the literal `full_multidomain.yaml`
  would take ~6+ hours at measured throughput, and built a new
  ratio-preserving ~10x-scaled config (`configs/full_multidomain_run1.yaml`)
  instead, sized to finish in well under an hour. That build produced 4219
  public_dev + 4279 private_test sequences (near-dupes only ~1.4%, vs. the
  calibration run's ~28% — confirmed that figure was a `--max-count` sizing
  artifact, not real). Ran `benchmark_validation` on the result: corpus-wide
  numbers corroborate the earlier small-demo figures (redundancy ~0%,
  eff-dim 4.84/22), and the per-task breakdown surfaced a genuine new
  finding — the synthetic backbone's effective dimensionality is markedly
  higher than every real-derived task's, meaning the real-derived tier adds
  domain breadth but not catch22-feature-space breadth per task (see §5's
  Findings for the full numbers and the correct way to read that). The
  literal full-scale `full_multidomain.yaml` build (~99K attempted samples)
  remains future work, best done unattended/overnight given its ~6-hour
  estimate.
- **2026-08-03 (§5.4 session — first real-checkpoint run of `tsfm_lens`)** —
  Picked up §5.4, the last unstarted item in Phase 1: validate the two
  live-weights unknowns flagged in `CLAUDE.md` §9/§10 by actually running
  real TimesFM/Chronos checkpoints, something that had never been done in
  this repo before. Discovered this machine has an RTX 5070 GPU (previously
  undocumented); per user instruction, reused the existing `tsfmPy` conda
  env rather than building a second one, upgrading its torch in place to a
  CUDA build and adding `transformers`/`timesfm`/`chronos-forecasting`/
  `zarr` — verified this doesn't resurrect the `CLAUDE.md` §11.13 MKL ABI
  crash and doesn't regress the existing `tsfm_benchmark` test suite (still
  40/1, unchanged). Found and fixed two genuine, previously-undiscovered
  bugs in the process — `tsfm_lens`'s zarr storage layer called a v3-only
  API method despite being pinned to zarr v2 (would have crashed on the
  very first real extraction), and `impulse_alignment_check`'s hardcoded
  impulse amplitude was badly miscalibrated for Chronos's context-adaptive
  quantization tokenizer, producing misleadingly low (as low as 0.06)
  diagonal-alignment scores that looked like a broken adapter but were
  actually a broken test. Both fixed and re-verified: both checkpoints now
  show perfect (1.00) alignment at every layer. Ran the full
  `medium_run.yaml` pipeline end-to-end against real
  `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-small`
  checkpoints (~10.5 min on the RTX 5070) — confirmed Chronos
  `output_attentions` and TimesFM's SDPA-unfusing attention-pattern capture
  both work correctly on live weights (sane, non-degenerate outputs:
  TimesFM's future attention mass is exactly zero everywhere, as required
  for a causal decoder), and confirmed TimesFM's previously-flat L3 patching
  curve is a genuine whole-layer-averaging artifact — per-window patching
  recovers a strong, depth-intensifying causal signal concentrated in the
  context window nearest the forecast horizon. See §5's Findings block for
  full numbers, including first (exploratory, small-scale) real L0/L2
  results corroborating the earlier "TimesFM stronger" finding and
  confirming a real bidirectional L2 stitching gain. Also fixed a stale
  doc-drift comment in `configs/default.yaml` flagged in `CLAUDE.md`'s own
  reconciliation note.
- **2026-08-04 (Phase 0 session)** — Picked up Phase 0 (explainability &
  report hygiene), the roadmap's own first-listed phase and, per §3's stated
  rationale, the one every later phase's report credibility depends on —
  left entirely unstarted (all checkboxes `[ ]`) across three sessions that
  had instead continued straight into Phase 1. Before implementing anything,
  actually read `report/report.py` rather than trust the checklist's framing
  (`CLAUDE.md` §2.4): found most of the "glossary" ask already substantially
  satisfied by an existing `_note()` mechanism attached to nearly every plot,
  apparently added in a session whose work never got reflected back into
  this file — a reminder that this file can itself drift stale the same way
  `CLAUDE.md` does. Fixed the two genuine gaps that check turned up (an
  unexplained periodicity-head score, a missing worked numeric example for
  head ablation), then implemented the three items that really were
  unstarted: `report.verbose`/`--verbose`, a "How to read this report"
  preamble with the full evidence-class ladder, and — the most substantial
  piece — a new L3 per-series verbose case-study section (context + true
  continuation + clean/corrupted/patched forecasts next to that series' own
  layer x window restoration grid), extracted from data the aggregate
  computation already produces at effectively zero extra forward-pass cost.
  Also found and fixed a genuine, previously-undiscovered platform bug in
  the process: every `write_text`/`read_text` call in the package used the
  platform-default text encoding, which crashed the report stage with
  `UnicodeEncodeError` on native Windows (cp1252) the moment it actually ran
  there for the first time — the same "never verified in this actual
  environment" pattern `CLAUDE.md` §11.15 already describes. Fixed
  everywhere, not just the one call that crashed first. Re-ran the full
  smoke suite after every change (`CLAUDE.md` §11.6's own instruction) and
  added new shape/content assertions covering the new verbose artifacts and
  both the on/off gating paths, rather than eyeballing one passing run. See
  §4's Findings block for full detail.
- **2026-08-04 (housekeeping + L3 investigation session)** — Four
  user-directed tasks, none following straight from the phase sequence: (1)
  investigated a user-reported "detrend/spike/dropout barely react" pattern
  against the real `runs/medium_run` artifacts rather than guessing —
  confirmed it's not a corruption-implementation bug (direct input-magnitude
  checks show `detrend` perturbs the input as much as `noise`/
  `deseasonalize`) but a mix of a genuine finding (both real checkpoints are
  least activation-sensitive to `detrend` of any corruption — consistent
  with recency-dominated forecasting over trend-extrapolation) and a report
  display artifact (`level_shift`'s far larger configured strength dominates
  the linear-scale Behavioral sensitivity chart, visually flattening
  everything else). Fixed the display (value labels, a sharper note, a new
  least/most-sensitive findings callout) without touching the corruptions or
  their statistics, and explicitly decided to keep all three in the battery
  — dropping low-signal corruptions would remove the contrastive signal L3
  exists to produce. (2) Audited the whole repo for hardcoded absolute
  paths; found and fixed the one violation
  (`tsfm_lens/configs/medium_run.yaml`). (3) Created `DEPENDENCIES.md`,
  and in doing so resolved a previously-deferred 2026-08-03 open question:
  CUDA torch, numpy/MKL, and `sdv` all coexist fine in the single `tsfmPy`
  env today — `sequential_par`'s GPU path is not blocked, correcting a
  stale "decision deferred" note left in §5. (4) Added two new standing
  invariants to `CLAUDE.md` §7 (no hardcoded paths; keep `DEPENDENCIES.md`
  current) and mirrored them into this file's §0. Re-ran the `tsfm_lens`
  smoke suite and `tsfm_benchmark`'s test suite after all code changes
  (still 40/1 and the smoke suite green) — nothing here was left unverified.
- **2026-08-04 (checkpoint-breadth session, partial)** — Picked up §5.3, the
  next unstarted Phase 1 item: run more than the single default checkpoint
  pair so existing findings can be checked for a size confound. Added
  `configs/medium_run_chronos_base.yaml` (identical `benchmark_medium`
  corpus and TimesFM checkpoint to `configs/medium_run.yaml`, swapping only
  `chronos-t5-small` → `chronos-t5-base`), and ran `--check-alignment`
  against the new checkpoint per invariant 7 before trusting anything from
  it — perfect 1.00 diagonal-hit fraction at every one of its 12 encoder
  blocks. Started the full pipeline run, but it was **explicitly deferred
  mid-run by the user** before completing or producing a report — no
  base-vs-small comparison numbers exist yet. Left as `[~]` rather than
  `[x]` in §5.3 so this isn't mistaken for a finished comparison; the config
  and alignment check are real, the actual result is the next session's
  first step (`python run.py --config configs/medium_run_chronos_base.yaml`
  to completion, then diff against `runs/medium_run`'s existing artifacts).
- **2026-08-05 (§5.3 completion session)** — Picked up exactly where the
  prior session left off: completed the deferred
  `medium_run_chronos_base.yaml` run, this time on a new, previously-
  undocumented machine (Linux, `cudaPy` conda env, 8× RTX A5000). Checked
  package versions against `DEPENDENCIES.md` before running anything
  (`CLAUDE.md` §11.15's own lesson) and found the same zarr-v3-vs-v2
  `create_array`/`create_dataset` trap recurring on this third environment
  — fixed by pinning `zarr<3` as documented, no code change needed this
  time since the §11.15 fix already targets the v2 API correctly. Re-ran
  `--check-alignment` for both models before trusting the run (perfect 1.00
  at every layer, matching prior machines). Ran the full 12-stage pipeline
  to completion (~32.5 min; 10 sections, 21 findings) and diffed every
  major artifact against the existing chronos-small `runs/medium_run`: the
  one dev-to-private-confirmed finding this corpus supports
  (`random_parametric` favors TimesFM) is size-robust, but L1 CKA and L2
  stitching gain both *increase* with Chronos's size while L4 clustering
  AMI *decreases* — a genuine, checked demonstration that different
  evidence-class metrics don't move together across a size change, not
  just a restated doctrine point. See §5's Findings block for full numbers.
  Did not touch Phase 1 §5.1 (corpus breadth) or re-litigate §5.4 (already
  done) — this session's scope was strictly finishing §5.3's one open
  item. Next: §5.5 (cross-run aggregator + retiring `CLAUDE.md` §10's
  single-run findings into this file) is Phase 1's last unstarted item.
- **2026-08-05 (same-day follow-up — §5.5, closing out Phase 1)** — Built
  `tsfm_lens/report/meta_report.py` + CLI `run_meta_report.py`: reads N
  existing run directories' artifacts (L0/L1/L2/lens/clustering/confirm),
  degrading per-section rather than crashing when a stage was skipped in a
  given run, and renders one comparison table plus a per-family stability
  view. Tested against all four run directories that existed in the repo
  (`medium_run`, `medium_run_chronos_base`, `real_run`, `smoke`) — on first
  use it correctly surfaced that `random_parametric` favors TimesFM
  *stably* across all three real-checkpoint runs (two Chronos sizes, one
  independent corpus), directly answering the "does this hold across
  regimes" question this deliverable exists for. Found and fixed one real
  bug building it: `runs/real_run`'s TimesFM has `crystallization_depth:
  null` (never crystallized within tolerance) and the first template draft
  crashed formatting that with `%.2f` — added two regression tests
  (`tests/test_meta_report.py`) covering this and the missing-stage
  degrade path. Re-ran the full `tsfm_lens` pytest suite after — all 5
  tests (3 existing + 2 new) pass, no regressions. Then retired `CLAUDE.md`
  §10 per this section's own "better" option: moved its original
  provisional findings verbatim into this section's Findings block (dated
  2026-08-05, "carried forward"), stated which of its claims are now
  superseded by later evidence and which aren't, and replaced §10 itself
  with a pointer here — added a matching `CLAUDE.md` reconciliation note
  and updated its repo-layout tree and CLI examples for the new files.
  **Phase 1 is now fully checked off** (§5.1–§5.5 all done); Phase 2a
  (layer-selection metric research) is the next unstarted item per §3's
  dependency ordering.
- **2026-08-05 (same-day follow-up — §6.1, Phase 2a first pass)** — Picked
  up the next item per §3's dependency ordering: does effective
  dimensionality predict which layers are worth interpreting? Built
  `tsfm_lens/analysis/layer_selection.py`, pooling 64 (run, model, layer)
  records across all 4 existing run directories and correlating three
  candidate metrics (effective dimensionality, input-CKA, and a proposed
  new one — L3 fingerprint entropy) against two interpretability proxies
  (tuned-lens R², family-probe decodability), reusing `analysis/stats.py`'s
  existing bootstrap primitive with a cluster unit of (run, model) rather
  than layer (layers within one model are depth-autocorrelated the same
  way windows within a series are). Checked the obvious confound before
  trusting anything (§2.4): three of the four metrics trend with depth on
  their own, so every correlation is reported both raw and depth-controlled.
  Real result: effective dimensionality predicts the two proxies in
  **opposite directions** (lower predicts more forecast-readable, higher
  predicts more family-decodable), both significant after depth control —
  so there's no single "worth interpreting" score, which is itself the
  finding. The proposed alternative, L3 fingerprint entropy, beat effective
  dimensionality on the decodability proxy (tighter CI, same direction).
  Added `recommend_layers(goal=...)` rather than a goal-less "auto," with
  an explicit note distinguishing it from `clustering.py`'s unrelated
  `layer: auto`. Did not test kurtosis/sparsity of raw activations (would
  need new zarr-store plumbing, not just reading existing artifacts like
  the other three candidates) — flagged as follow-up. Three new regression
  tests (`tests/test_layer_selection.py`); full suite 10/10 after. See
  §6.1's Findings block for exact numbers and the honest n_groups=8
  caveat. Next: Phase 2b (SAE variants) is the flagship research thread and
  a much larger undertaking — a natural point to check in with the user on
  scope/direction before committing to a specific SAE architecture, per
  §6.2's own "evaluate empirically, do not assume one is best" instruction.
- **2026-08-05 (same-day follow-up — §6.2, Phase 2b baseline)** — Checked in
  with the user before committing to an SAE architecture given the phase's
  scale; user chose the baseline-first path (item 4: plain TopK SAE +
  eval harness + ground-truth alignment, before attempting the flagship
  crosscoder). Researched the exact integration points first (a background
  agent read `sae/interface.py`, `extraction/store.py`, `extraction/hooks.py`,
  `tsfm_benchmark`'s ground-truth schema, `config.py`, `pipeline.py` in
  detail) rather than guessing signatures. Built `sae/models.py`
  (`TopKSAE`: architectural top-k sparsity, unit-norm decoder columns),
  `sae/train.py` (training loop reading straight from the zarr store's
  window-level activations, `run_sae` as a new pipeline stage gated on
  `sae.enabled`), `sae/eval.py` (reconstruction fidelity, dead-feature
  rate, forecast-preservation reusing `token_patch` + L3's own
  per-token-to-window assignment), and `sae/ground_truth.py` (the
  ground-truth feature-alignment score, split into a pure matching
  function and an I/O wrapper specifically so the matching logic is unit-
  testable without a real corpus). Extracted `l0_behavioral.py`'s inline
  MASE formula into a shared `analysis/stats.py::mase()` along the way
  (re-verified byte-identical via the smoke suite) rather than
  reimplementing it for the new eval harness. Ran the baseline against
  `runs/medium_run_chronos_base`'s already-extracted real-checkpoint
  activations (added an `sae:` block to that config, ran `--stages sae` —
  the existing stage-skip machinery correctly reused the existing
  extraction with no new model calls needed for training/eval) and found:
  real, ground-truth-verified seasonality-linked features (the session's
  clearest positive result); a genuinely high dead-feature rate (~98%,
  `dict_size_mult: 8` was real overcapacity for ~4600 training rows with
  no dead-neuron resampling implemented); and a forecast-preservation
  check that correctly caught a real problem (patching the reconstruction
  back in roughly doubled MASE for both models) that should block Phase
  3's planned feature-ablation work on this baseline until fixed — the
  validity check did exactly its job. Also discovered, by actually trying
  to use it rather than trusting the pooled statistics, that §6.1's
  `recommend_layers` policy does **not** transfer as a within-model
  ranking rule: it correctly picked TimesFM's best layer in this run but
  picked Chronos-T5-Base's *worst* one, because that model's own layers
  trend the opposite direction from the pooled cross-model finding —
  documented with the exact numbers in both `recommend_layers`'s docstring
  and this file's §6.2 Findings. 6 new unit tests
  (`tests/test_sae.py`) plus one new mock-adapter integration test
  (`tests/test_smoke.py::test_sae_stage_integration`, confirming
  `forecast_preservation` works with no real corpus and
  `ground_truth_alignment` degrades to a logged error rather than
  crashing when one isn't available); full suite 17/17 after. See §6.2's
  Findings block for every number. Not done this session, left as
  explicit follow-up in §6.2's checklist: the crosscoder (item 1), the
  encode-store-into-`sae/{model}/{layer}` seam, feeding SAE results back
  into §6.1's correlation study, and verbose-mode report integration.
- **2026-08-05 (same-day follow-up — fixing the baseline, per user
  choice)** — User chose to fix the baseline (dead-feature collapse,
  forecast-preservation failure) before touching the crosscoder, and
  explicitly asked for more SAE training data from an established source
  ("something on huggingface") rather than only shrinking the dictionary.
  Implemented dead-neuron resampling and `sae/real_data.py` (reuses
  `tsfm_benchmark`'s existing generic HF loader, verified live against
  Monash). Dead-neuron resampling took **three iterations, not one**, each
  caught by actually measuring before declaring it fixed (§2.4): attempt 1
  (fixed absolute encoder scale, no Adam-state reset) was catastrophic —
  fidelity went deeply negative, worse than baseline, not just
  ineffective. Attempt 2 (calibrated scale + Adam-state reset) fixed
  Chronos-T5-Base but was still catastrophic for TimesFM specifically —
  root cause: resampling ~10,000 dead atoms at once from too few source
  rows collapsed most of them onto near-duplicate directions. Attempt 3
  (capped resample fraction per event + directional jitter) finally gave
  a clean, real improvement on both models, confirmed first in a fast
  controlled sweep (no model calls) before trusting it on a real run.
  Final real-run numbers: reconstruction fidelity 0.629→**0.858**
  (TimesFM), 0.567→**0.840** (Chronos-T5-Base). Forecast-preservation now
  **passes for TimesFM** (ΔMASE +0.047) but **still fails for
  Chronos-T5-Base** (ΔMASE +2.37) despite similar fidelity gains —
  investigated the discrepancy rather than shrugging at it, and found a
  real, architecture-specific confound in the check's own window-broadcast
  approximation (exact for TimesFM's patch tokenization where window ==
  token width, lossy for Chronos's per-timestep tokenization where one
  window spans 32 distinct tokens forced to the same value) — documented
  in `sae/eval.py`'s docstring, not left as an unexplained asymmetry.
  Ground-truth alignment got richer and produced the session's cleanest
  single match: Chronos-T5-Base's top feature now tracks
  `has_intermittency` at ρ=0.881. Added a regression test guarding the
  exact resampling-collapse failure mode found along the way. Full suite:
  21/21. See §6.2's Findings for every number and the full three-attempt
  story. `configs/medium_run_chronos_base.yaml`'s `sae:` block now
  reflects the working configuration (`epochs: 60,
  resample_dead_every_epochs: 5, real_data_enabled: true`), not the
  original broken-baseline one.
- **2026-08-05 (design-only session — §6.1.1, layer-selector redesign)** — No
  implementation this session, per explicit user instruction. Read `CLAUDE.md`
  and `ROADMAP.md` in full and `analysis/layer_selection.py` to characterize
  why the §6.1 layer-selection mechanism, though its limitations were already
  documented, is not the right long-term solution: it's a cross-(run, model)
  correlation study that provably fails as a within-single-model selector
  (§6.2 Findings), it's circular (needs the full expensive pipeline over all
  layers *before* it can select), it's unfair to strided-out layers, and its
  predictive direction is itself architecture-dependent. Wrote §6.1.1: three
  candidate selectors (residual-trajectory work/bend geometry;
  ground-truth factor-emergence spectroscopy; submodular redundancy-coverage),
  each mapped against four explicit design requirements (architecture-agnostic,
  all-layers-fair, parsimonious, cheap-enough-to-run-first), plus §6.1.1-E —
  the bake-off protocol (independent SAE-ground-truth gold ranking,
  parsimony curve, uniform-stride null, within-model robustness gate) that
  decides the winner rather than picking one on intuition. Re-scoped §6.1 in
  place as a *finding* not a selector, updated the §13 open question it
  partially answers, and added a new open question for the bake-off outcome.
- **2026-08-05 (same-day follow-up — §6.1.1 implementation + first bake-off
  run)** — Implemented and empirically ran the design from the session
  above, per explicit user instruction ("implement the tests, compare their
  usefulness ..., update documentation with results"). Built
  `tsfm_lens/analysis/layer_screen.py` (all three selectors:
  `select_work_bend`, `select_coverage`/`greedy_coverage_selection`,
  `select_factor_emergence`/`factor_probe_matrix`, plus the unifying
  `select_layers()` dispatcher) and `layer_screen_bakeoff.py` (§6.1.1-E's
  gold ranking via `build_gold_ranking` — reusing `sae/train.py`'s
  `train_sae` and `sae/ground_truth.py`'s matching logic rather than
  reinventing either — plus `recall_at_budget`/`parsimony_curve`/
  `null_curves`/`robustness_gate` and three ensemble combiners). Renamed
  `sae/ground_truth.py`'s `_encode_series_level` → `encode_series_level`
  (now genuinely shared between the production SAE stage and the bake-off,
  not duplicated). Added 12 new unit tests
  (`tests/test_layer_screen.py`) against synthetic data with planted,
  known-correct answers (a regime change, a redundant band, a ground-truth
  factor embedded only from a known layer onward) — full `tsfm_lens` suite
  33/33 after, no regressions. Built `configs/layer_screen_experiment.yaml`
  (the first config in this repo to capture **all 20** of TimesFM's layers,
  not the usual stride-2 10, specifically because a layer *screen* has to be
  fair to every layer by construction) and
  `run_layer_screen_bakeoff.py`, then actually ran the whole thing against
  live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small`
  checkpoints on an 8×RTX A5000 machine (extraction + gold-ranking SAE
  training for all 26 layers completed in under two minutes per seed) and
  replicated with an independent SAE seed before trusting any result
  (`CLAUDE.md` §2.4) — gold-ranking Spearman stability ρ=1.0/0.926. Headline
  result: Idea A (`work_bend`) matched the oracle almost exactly and beat
  both nulls on Chronos-T5-Small; no selector beat uniform-stride on
  TimesFM, but neither did the oracle there (a real, different finding from
  §6.1's outright selector failure); Idea B underperformed with a diagnosed,
  not-yet-fixed weighting flaw; combining methods did not beat the best
  single method on either model. Deliberately **not** wired into any config
  knob or `sae.targets: auto` yet — the result is a genuine, provisional
  first pass on one corpus/checkpoint-pair, not a settled cross-architecture
  claim. Full numbers, the exact diagnosed Idea B flaw, and the concrete
  follow-up list are in §6.1.1's Findings block and the updated §13 entry.
