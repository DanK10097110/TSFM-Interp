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
- [~] Run more than the single default checkpoint pair
  (`google/timesfm-2.0-500m-pytorch`, `amazon/chronos-t5-base`) — at minimum
  add a size variant on each side (e.g. TimesFM 1.0 vs 2.0, chronos-t5-small
  vs -base vs -large, chronos-bolt at a comparable size) so findings can be
  checked for "is this a TimesFM-vs-Chronos finding or a
  bigger-model-vs-smaller-model finding." This directly de-confounds several
  of the tentative findings in `CLAUDE.md` §10. **In progress 2026-08-04**:
  added `configs/medium_run_chronos_base.yaml` — identical corpus
  (`benchmark_medium`) and TimesFM checkpoint to `configs/medium_run.yaml`,
  swapping only `chronos-t5-small` → `chronos-t5-base` (6 → 12 encoder
  blocks), so a base-vs-small comparison isolates the size confound cleanly.
  The full pipeline run was started but **deliberately not completed/analyzed
  this session** (user explicitly deferred it mid-run) — the config and
  alignment check below are real and done; the actual base-vs-small
  comparison numbers are not yet in hand. **Next session: rerun
  `python run.py --config configs/medium_run_chronos_base.yaml` to
  completion**, then compare against the existing `runs/medium_run` (small)
  artifacts for L0 MASE gap, L1 peak CKA, L2 stitching gain, crystallization
  depth, and the L3 per-window restoration pattern (§5.4's window-15
  recency-concentration finding) before treating any of this file's or
  `CLAUDE.md`'s existing single-checkpoint findings as size-independent.
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

### 5.5 Consolidated cross-run reporting
A single-run HTML report (`CLAUDE.md` §6.5) doesn't by itself answer "what
are the qualities of each model in general" — that needs aggregation *across*
runs.
- [ ] Build a small cross-run aggregator (new, e.g.
  `tsfm_lens/report/meta_report.py`) that takes N run directories and
  produces one summary: per-archetype MASE gap, per-archetype crystallization
  depth, per-archetype CKA peak, stability of each finding across
  data regimes (does "TimesFM front-loads then compresses" hold on
  `intermittent_bursts` too, or only on the smooth archetypes it was observed
  on?). This is new work, not a rename of the existing per-run report.
- [ ] Explicitly revisit every "Defensible" / "Underdelivered" claim in
  `CLAUDE.md` §10 against the broader evidence base and update that section
  (or better, retire it from `CLAUDE.md` and move the *current*, broader-based
  version of these findings into this file's Findings block, since `CLAUDE.md`
  should describe stable architecture, not a moving research result).

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

---

## 6. Phase 2 — Layer selection, novel SAEs, and IP detection (the "meat")

This is the brief's highest-value, most novel-research-shaped ask (items 1
and 3), plus a related but separable idea (item 2). Broken into three
sub-phases because they have different dependencies (§3).

### 6.1 Phase 2a — Does anything predict "this layer is worth interpreting"?

**Goal.** Test the brief's central hypothesis directly: *does a layer's
effective dimensionality (participation ratio, already computed in
`analysis/internals.py`) correlate with how interpretable/controllable that
layer turns out to be?* If yes, that's both a genuine finding and a practical
layer-selection rule for Phase 2b. If no, the fallback is just as valuable —
know that before spending compute training SAEs on the wrong layers.

**Concrete deliverables**
- [ ] Define "interpretable/controllable" operationally, since it's not a
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
    retroactively, not as required before starting 2b.
- [ ] Compute effective dimensionality (and candidate alternatives — see
  below) per layer per model across every Phase 1 run, and correlate against
  each interpretability proxy above. Use the same series-level bootstrap
  discipline as the rest of the repo (`CLAUDE.md` §6.6) to put a CI on the
  correlation, not just a point estimate.
- [ ] Test at least these alternative candidate metrics against the same
  proxies, since the brief explicitly invites better metrics if they beat
  effective dimensionality:
  - Per-layer CKA-to-input (already computed in `internals.py`) — a layer
    that's barely moved from raw input statistics is a poor SAE target for
    different reasons than a too-high-dimensional one.
  - Per-layer L3 fingerprint *entropy* across corruption types (a layer
    causally sensitive to everything vs. to one specific structural
    property might make differently-useful SAE targets).
  - Simple activation kurtosis / sparsity of the raw (pre-SAE) activations —
    cheap, worth ruling in or out early.
- [ ] Write up the result as a genuine finding either way: "effective
  dimensionality [does/doesn't] predict X, with these caveats" — and if it
  does, turn it into an actual `layer_select: auto` policy usable by Phase 2b
  and by `analysis/clustering.py`'s existing `layer: auto` (which today
  picks the L1 peak-CKA pair for an unrelated reason — reconcile or
  distinguish these two "auto" policies explicitly so they don't get
  confused).

**Findings / decisions**
- *(append here)*

### 6.2 Phase 2b — A TSFM-native SAE variant (the flagship research thread)

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
- [ ] Implement `SAEAdapter` (the existing `Protocol` in `sae/interface.py`)
  for at minimum the baseline (item 4) and the crosscoder (item 1); wire the
  encode-store pass into `sae/{model}/{layer}` per the seam already
  documented there.
- [ ] Build the SAE training loop as its own module (e.g.
  `tsfm_lens/sae/train.py`) reusing the zarr store's `level="window"` reads
  directly — do not re-extract activations, the store already holds them
  (`CLAUDE.md` §6.4).
- [ ] Build an SAE evaluation harness covering: reconstruction fidelity
  (fraction variance explained) vs. sparsity (L0) frontier, dead-feature
  rate, and **forecast-preservation under reconstruction** (patch the
  reconstruction back in via `token_patch` and confirm MASE is close to
  clean — this is the validity check that ablation experiments in Phase 3
  will depend on).
- [ ] Implement the **ground-truth feature-alignment score** from §2.1/§6.3:
  for each learned feature, correlate its activation across the benchmark
  against every available ground-truth component (trend order, each
  seasonality's period/phase/amplitude, changepoint proximity, anomaly
  indicator, AR coefficients, noise-envelope depth, intermittency mask) and
  report the best match and its strength. This is this repo's distinctive
  answer to "is this feature interpretable" and should be a first-class
  output, not an afterthought.
- [ ] Feed the SAE evaluation results back into §6.1's layer-selection
  correlation study (the retroactive validation step noted there).
- [ ] Verbose-mode reporting (§4): a per-feature exemplar panel — top
  activating series/windows for a feature, its ground-truth alignment score,
  and (once Phase 3 feature ablation exists) its causal effect on forecast
  when zeroed.

**Findings / decisions**
- *(append here — including negative results; a crosscoder that doesn't
  outperform independent per-model SAEs at matching is itself an important
  finding about whether TimesFM and Chronos share feature-level structure,
  not just layer-level geometry)*

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
- [ ] Follow up on the one concrete causal lead already in hand
  (`CLAUDE.md` §10): additive-noise depth-sensitivity is **strongly
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
- SAE **contract only** (`Protocol`, seams documented) — no training code.
  This is exactly where Phase 2b starts from, not a head start beyond that.

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

- [ ] Does effective dimensionality actually predict anything useful, or is
  it a plausible-sounding metric that doesn't survive contact with data
  (§6.1)? Genuinely open — could go either way, and either answer is a real
  finding.
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
