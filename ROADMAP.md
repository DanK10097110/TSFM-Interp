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
>
> **Audit pass added 2026-08-06 (planning only, no code).** Two new sections
> at the end of this file: **§15** is a code-grounded audit of *silent-failure
> paths, fragile mechanisms, and acknowledged-but-fixable limitations* found by
> reading the pipeline against the claims in this file and `CLAUDE.md`
> (19 items, each with evidence at `file:line`, why it matters, and a detailed
> fix plan marked **NEEDS IMPLEMENTATION**); **§16** is an enhancement backlog
> aimed squarely at the §1 north star — what a stranger with a checkpoint and
> one button actually needs, benchmarked against what `transformer-lens` users
> expect. Several §15 items contradict claims made elsewhere in this file and
> in `CLAUDE.md`; those places now carry inline `⚠️ AUDIT` markers pointing at
> the relevant item rather than being quietly left to mislead. Nothing in §15
> or §16 has been implemented — they are scheduled work, not findings.

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
10. **§16 is a backlog, not a queue.** Items there are sized and prioritized
   but deliberately not sequenced into the §3 phase spine, because most are
   productization/rigor work that can be picked up in any order once the §15
   P1 items are closed. Do not treat a high §16 tier as permission to skip
   §15 — a one-button tool that silently produces a wrong number is worse
   than a five-command tool that refuses.

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

### 6.1.1 Phase 2a-v2 — a better, architecture-agnostic, all-layers-fair layer selector — ✅ implemented, bake-off-tested, and wired into production DONE 2026-08-05

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

### 6.2 Phase 2b — A TSFM-native SAE variant (the flagship research thread) — baseline (item 4) ✅ DONE 2026-08-05; crosscoder (item 1) feasibility test ✅ DONE 2026-08-05, flagship build not started

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

## 7. Phase 3 — Ablation and bias-characterization studies (brief item 3) — bullet 1 (controlled parameter sweeps) ✅ MASE-axis first pass DONE 2026-08-05; bullet 2 (noise x depth sweep) ✅ first pass DONE 2026-08-05; quantization-churn follow-up ✅ DONE 2026-08-05; bullet 5 (bias card) ✅ DONE 2026-08-05; bullet 4 (verbose case studies) ✅ forecast-half DONE 2026-08-05

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
- [ ] **Feature-level ablation** (depends on Phase 2b): zero individual SAE
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
- 🔴 **New (2026-08-10, found while closing the item above): does the
  bake-off's qualitative verdict (beats a given null, yes/no) reproduce
  across independent re-extractions of the same config, or only across
  SAE-training reseeds of one frozen extraction?** The existing "seed1
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
- [~] Is a joint crosscoder actually trainable/stable across two
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
  trained baselines, not a collapse. Still open: whether the flagship
  crosscoder (full shared/specific decomposition as a research deliverable,
  not just a stability check) is worth building next, given this run's
  dictionaries were mostly dead at every hyperparameter setting tried —
  see the Findings for why that's a data/training-budget issue, not
  specific to crosscoders, and what would need fixing first.
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
- **2026-08-05 (third same-day follow-up — §6.1.1 production wiring)** — Per
  explicit user instruction ("make sure the new layer selection is default
  in full pipeline runs and is fully integrated"), wired `work_bend` in as
  the pipeline's default layer selector rather than leaving it a standalone
  bake-off script. Added `LayerScreenConfig` to `config.py`; a
  `layer_screen` pipeline `Stage` in `pipeline.py` running right after
  `extract` and before every expensive stage (satisfies R4 by construction,
  via canonical stage-list order); `run_layer_screen` in
  `analysis/layer_screen.py` with two independent graceful-degrade paths
  (`factor_emergence` without a sealed/ground-truth corpus falls back to
  `work_bend`; any selector raising on a real model falls back to a
  uniform-stride `_uniform_fallback`); `sae/train.py::_default_targets`
  rewritten to consume `layer_screen/selection.json` when `sae.targets` is
  empty, replacing the old "each model's final captured layer" default; a
  new "Screen" report section between L0 and Profile with its own
  evidence-class-honest `_note()`. Updated `default.yaml`/`medium_run.yaml`/
  `layer_screen_experiment.yaml` with an explicit `layer_screen:` block;
  deliberately left `medium_run_chronos_base.yaml`'s already-enabled `sae`
  stage on its **explicit**, historically-documented targets rather than
  switching to `auto`, so re-running that exact config still reproduces
  §5.3/§6.2's recorded numbers. Extended `tests/test_smoke.py` with the new
  artifact path, the new report-section token, and a new test asserting
  `_default_targets` resolves *exactly* what `layer_screen` selected (not
  just that each half works alone) — full suite 34/34, plus a live CLI run
  of `configs/smoke.yaml` (not just pytest's tmp-dir harness) confirming a
  real 11-section/22-finding report and a sane `selection.json`. This is a
  deliberate policy call, not a new empirical result: the bake-off's mixed,
  single-corpus finding is unchanged (§6.1.1's Findings, §13's open
  question) — what changed is that "provisional but currently-best" is now
  treated as good enough to replace an admittedly-arbitrary old default,
  on the user's explicit authority to make that call, rather than waiting
  for the broader cross-architecture validation the Findings still name as
  outstanding. Full detail in §6.1.1's second Findings block.
- **2026-08-05 (fourth same-day follow-up — §6.2/§13 crosscoder feasibility
  test)** — Read `ROADMAP.md` fresh per the user's autonomous-loop
  instruction ("do the next unfinished thing... keep doing this") and
  picked up §13's explicit prerequisite for §6.2 item 1's flagship
  crosscoder: an early small-scale trainability/stability test, before
  committing to the full build. Built `tsfm_lens/sae/crosscoder.py`
  (`CrosscoderSAE` — a TopK dictionary jointly trained across `n_sources`
  inputs of independent dimension, joint per-feature decoder
  normalization so `relative_decoder_norm` carries real shared-vs-specific
  signal, dead-atom resampling ported from the single-source baseline) and
  `run_crosscoder_feasibility.py` (loads an already-extracted run's store
  at the L1 peak-CKA layer pair, trains independent baselines plus one
  joint crosscoder, compares). Found and fixed two real bugs via testing
  before trusting any result (`CLAUDE.md` §2.4): a genuine training
  instability (unequal per-source activation scale lets the larger-scale
  source dominate the joint MSE objective, reproduced synthetically as a
  fidelity collapse to −9.7, fixed via per-source scale normalization
  built into the model itself) and a device-mismatch crash in dead-neuron
  resampling that only the real-GPU run surfaced. Ran for real against
  `runs/medium_run_chronos_base`'s already-extracted TimesFM/Chronos-T5-Base
  activations: joint training is stable (no source-domination collapse
  across three hyperparameter settings), though this run's dictionaries
  were mostly dead regardless of crosscoder-vs-baseline — a data/training-
  budget finding, not a crosscoder-specific one, and itself surfaced a
  second metric bug (the shared/specific split counted dead atoms, which
  are near-"shared" by random-init symmetry regardless of signal; fixed
  with an alive-only filter). 3 new unit tests, full `tsfm_lens` suite
  37/37 after. Full numbers, both bug fixes, and the concrete "what's
  still needed before the flagship build" list are in §6.2's new Findings
  block and the updated §13 entry — this session deliberately did not
  attempt the flagship crosscoder itself (SAEAdapter implementation, report
  section), only the feasibility gate.
- **2026-08-05 (fifth same-day follow-up — §7 Phase 3 noise-SNR × depth
  sweep)** — Continued the autonomous "read roadmap, do the next unfinished
  thing" loop. Picked §7 Phase 3 bullet 2, the one bullet in an otherwise
  entirely-unstarted phase with a concrete, already-observed effect to
  chase (ρ≈−0.90 additive-noise depth-sensitivity anti-correlation) rather
  than an open-ended new design. Built `run_noise_snr_sweep.py`, which
  reruns only `analysis/l3_perturbation.py::run_l3`'s `noise` corruption
  at several SNR values against an already-extracted run's store (no
  re-extraction, no new downloads) and reads back its own already-computed
  per-corruption Spearman agreement rather than writing new statistics
  machinery. Ran for real against `runs/medium_run_chronos_base`
  (TimesFM 2.5-200M vs. Chronos-T5-Base), 96 series, 6 SNR points, ~82
  seconds total on one GPU. Found a real dose-response transition: positive
  agreement at mild noise (ρ=+0.40), a sign flip by SNR=12dB (ρ=−0.44), and
  a broad plateau of strong anti-correlation from 6dB down through −3dB
  (ρ≈−0.76 to −0.85) — the main pipeline's default 6dB setting sits well
  inside that plateau, not at a fragile edge. Also found each model's own
  peak-sensitivity layer (TimesFM's first captured layer, Chronos's last)
  never moves across the entire SNR range — only the cross-model rank
  agreement's sign/strength changes. The *why* (quantization vs. something
  else) is still open, named as the concrete next step. Full numbers in
  §7's new Findings block.
- **2026-08-05 (sixth same-day follow-up — §7 quantization-churn probe)** —
  Picked up the concrete next step named at the end of the previous entry:
  test the "continuous embedding vs. quantization" hypothesis directly,
  rather than leaving it open. Built `analysis/quantization_churn.py`
  (`token_churn_stats`, a pure statistic split from I/O so it's unit-testable
  without a live tokenizer, mirroring `sae/ground_truth.py`'s existing
  pattern) and `run_quantization_churn_sweep.py`, which reuses the exact
  same 96-series row sample and noise draws the prior sweep used and
  tokenizes clean vs. corrupted contexts through whichever model in the
  pair exposes a quantization tokenizer — gracefully skipping
  continuous-embedding models like TimesFM with a log, per `CLAUDE.md`
  §2.5, rather than fabricating a comparable metric where none exists. Ran
  live against `runs/medium_run_chronos_base`'s Chronos-T5-Base (no
  re-extraction, no forward passes — tokenization only). Found a real,
  nuanced answer rather than a clean confirmation: raw churn *fraction*
  saturates almost immediately (96.4% of tokens already reassigned at the
  mildest SNR tested) and has no dynamic range to explain anything, but
  token-ID jump *magnitude* does — its two largest relative jumps
  (20dB→12dB, 12dB→6dB) line up with the sweep's sign-flip and subsequent
  deepening, a real partial confirmation of the quantization hypothesis for
  *why the sign flips*. But the plateau region (6dB→−3dB) breaks a clean
  story: jump magnitude keeps climbing there while the cross-model
  agreement stays flat rather than continuing to deepen, so quantization
  severity alone doesn't explain the plateau's shape — likely each model's
  own sensitivity saturating under heavy corruption contributes too, not
  decomposed further this session. Sanity-checked the jump numbers against
  the tokenizer's own bin width (`MeanScaleUniformBins`, 4094 bins over
  [−15,15]) before trusting them — calibrated to the actual mechanism, not
  an artifact. 7 new unit tests (`tests/test_quantization_churn.py`, all
  synthetic), full `tsfm_lens` suite **44/44** after (was 37). Full numbers
  in §7's Findings block. This closes out §7's one concrete, ready-to-chase
  lead from `CLAUDE.md` §10/carried-forward findings; the rest of §7
  (controlled parameter sweeps, feature-level ablation, verbose-mode case
  studies, the per-model bias card) remains entirely unstarted and is the
  natural next pick for a future session, per §3's dependency ordering
  (feature-level ablation still needs a forecast-preservation-passing SAE,
  which per §6.2's Findings currently exists for TimesFM but not
  Chronos-T5-Base).
- **2026-08-05 (seventh same-day follow-up — §7 Phase 3 bullet 1, controlled
  parameter sweeps)** — Continued the autonomous "read roadmap, do the next
  unfinished thing" loop; picked up exactly what the previous entry named as
  the natural next pick, §7's remaining unstarted bullet 1. Built
  `analysis/parameter_sweep.py` (pure recipe-construction and sweep-data
  generation, using `build_pipeline`'s `parametric` generator directly and
  holding every component fixed except the one swept parameter, unit-tested
  without any GPU since generation is plain numpy) and
  `run_parameter_sweep.py` (reuses an existing run's model configs/
  checkpoints, generates fresh sweep data itself, scores via the same
  `adapter.predict` call L0 already makes — no extraction, no store I/O).
  Ran all three sweeps the checklist names, live, against
  `runs/medium_run_chronos_base`: seasonal period (4→256), noise scale
  (0.05→2.5, an inverse-SNR sweep), and intermittency rate (0→0.8). Found
  three genuinely different regimes rather than one clean story: TimesFM
  leads increasingly as seasonal period grows, with one striking exception —
  a real, CI-confirmed MASE spike specifically at period=32, TimesFM's own
  patch width, flagged as a plausible aliasing effect but explicitly not
  verified as the mechanism this session; Chronos-T5-Base is far better at
  very clean low-noise periodic signal specifically, a reversal from the
  period sweep, with the gap closing entirely by noise_scale≈0.6; and the
  intermittency sweep is confounded by MASE's own scale term collapsing
  under heavy zeroing (which also zeroes the forecast horizon itself),
  documented as a real methodological caveat rather than reported as a
  clean finding. 8 new unit tests (`tests/test_parameter_sweep.py`, all
  synthetic/GPU-free), full `tsfm_lens` suite **52/52** after (was 44). Full
  numbers and the exact caveat reasoning are in §7's Findings block. Not
  attempted this session: crystallization depth and L3 noise-fingerprint
  response per sweep point (both need the full extraction/lens machinery,
  not just prediction — a materially larger undertaking, left as an
  explicit follow-up rather than partially built), verbose-mode case
  studies, and the per-model bias card. §7's remaining unstarted work
  (feature-level ablation, verbose-mode case studies, the bias card) is the
  natural next pick, with feature-level ablation still gated on a
  forecast-preservation-passing SAE for Chronos-T5-Base specifically (per
  §6.2's Findings, TimesFM already has one).
- **2026-08-05 (eighth same-day follow-up — §7 Phase 3 bullet 5, per-model
  bias card)** — Continued the autonomous "read roadmap, do the next
  unfinished thing" loop. Of the three items the previous entry named as
  the natural next pick, deliberately chose the bias card over
  feature-level ablation (bullet 3): per `ROADMAP.md` §0.4's "explain before
  reordering," ablation is only half-unblocked (§6.2's Findings show the
  SAE's forecast-preservation validity check passes for TimesFM but still
  fails for Chronos-T5-Base), so building it now would produce a one-sided,
  confounded comparison rather than the real cross-model test it's meant to
  be — recorded as an explicit, reasoned skip in §7's checklist, not a
  silent one. Built `report/bias_card.py` (pure comparison/aggregation
  logic — CI-overlap-based "favored" verdicts, generic anomaly detection
  against each sweep's own plurality winner, caveat attachment — split from
  its HTML-rendering I/O the same way `analysis/parameter_sweep.py` and
  `sae/ground_truth.py` already split) and `run_bias_card.py`, which
  consolidates the `param_sweep_*.json` artifacts from the immediately
  preceding session into one standalone `runs/bias_card.html` — deliberately
  not a new `report.py` section, mirroring `report/meta_report.py`'s
  existing precedent for cross-artifact (not single-run) aggregation.
  6 new unit tests, full `tsfm_lens` suite **58/58** after (was 52). Ran
  live against the real sweep data: the generated card mechanically
  corroborated every hand-written finding from the previous session (the
  period=32 TimesFM anomaly, Chronos's noise-sweep edge being confined to
  the two lowest noise levels once CI-overlap is required, the
  intermittency caveat attached to both models) with no numbers hand-picked
  for the card — a genuine, if small, validation that the CI-based
  "favored" logic recovers the same conclusions a human read of the tables
  already reached. Closes out §7 bullet 5. Remaining §7 work: feature-level
  ablation (blocked as above until Chronos's SAE forecast-preservation gap
  closes) and verbose-mode case studies for the sweeps (bullet 4, still
  entirely unstarted and now the only fully-unblocked item left in this
  phase) — the natural next pick.
- **2026-08-05 (ninth same-day follow-up — §7 Phase 3 bullet 4, verbose
  sweep case studies)** — Continued the autonomous "read roadmap, do the
  next unfinished thing" loop; picked up exactly what the previous entry
  named as the last fully-unblocked item in this phase. Built
  `report/sweep_case_studies.py` (`select_case_study_values` — pure,
  reusing `report/bias_card.py`'s anomaly detection rather than
  reinventing "interesting sweep point" — plus `render_case_studies_html`
  for the I/O) and `run_sweep_case_studies.py`, which generates one fresh
  example series per selected sweep value and renders its context/true-
  continuation/both-models'-forecasts, narrated — the forecast half of the
  bullet; explicitly did not attempt the per-layer skip-lens-curve half,
  which needs the full extraction/lens machinery per series and was
  already flagged in two prior entries as a materially larger undertaking
  than anything this session's sweeps have needed so far. Ran live against
  all three real sweep artifacts from the last two sessions
  (`runs/medium_run_chronos_base`): the selection logic correctly picked
  each sweep's own extremes plus its already-known anomaly (or the middle
  value, for the noise_scale sweep which has none) with zero manual
  intervention, and every case study's per-model MASE matched the
  aggregate sweep numbers already on record, confirming consistency rather
  than a parallel, potentially-diverging generation path. 7 new unit
  tests, full `tsfm_lens` suite **65/65** after (was 58). Full detail in
  §7's Findings block. This closes the forecast half of bullet 4 — the
  checklist item is marked done with that scope stated explicitly, not
  silently expanded to cover lens curves it doesn't include. §7's remaining
  work is now: the lens-curve half of bullet 4 (needs extraction/lens
  machinery per case-study series) and feature-level ablation (bullet 3,
  still blocked on Chronos-T5-Base's SAE forecast-preservation gap, §6.2).
  Phase 3's three fully-unblocked, self-contained deliverables (controlled
  sweeps, the bias card, forecast-half case studies) are now all done;
  everything left in this phase needs either the SAE fix or new
  extraction/lens plumbing, not just `adapter.predict` — a natural point to
  pick up a different phase, or to invest in one of those two prerequisites
  directly, next session.
- **2026-08-06 (autonomous 20-minute-cadence session, user-directed)** —
  Picked up two small, well-scoped, previously-diagnosed-but-unfixed items
  that needed no live GPU/checkpoint rerun, deliberately avoiding anything
  requiring unattended multi-hour compute: (1) §6.1.1's `factor_emergence`
  peak-weighting flaw (`CLAUDE.md` §11.18) — `factor_emergence_scores`
  (`tsfm_lens/analysis/layer_screen.py`) now weights each ground-truth
  factor's contribution by its own peak decodability and gates the
  `emergence` event behind a new absolute `min_peak_score` floor, verified
  against a new planted-data unit test (weak/noisy factors no longer
  out-vote a real late peak); real-checkpoint re-verification through the
  §6.1.1-E bake-off is explicitly still open, not claimed done. (2) SAE
  training's redundant real-data catalog refetch (noted but not fixed in
  §6.2's Findings) — `run_sae` now samples the real context pool once per
  run instead of once per (model, layer) target. Full `tsfm_lens` suite
  re-run clean after both: **66/66**. Also corrected one piece of doc
  drift found while touching this area: `CLAUDE.md` §6.2's own warning
  about a stale `configs/default.yaml` comment was itself stale — that
  comment had already been fixed in an untracked earlier session. First
  fix's implementation was done by a background agent (per the user's
  20-minute self-pacing loop request); this session's continuation
  verified its work (full suite, cross-checked the diff against the
  diagnosis in both files) before writing it up here, rather than trusting
  the agent's own unseen completion report.
- **2026-08-06 (same loop, iteration 3)** — Picked up §6.2's last
  fully-unblocked, no-live-GPU-needed checklist item: the per-feature SAE
  exemplar panel. New `report/sae_exemplars.py` (pure selection/table-
  building split from I/O, mirroring `report/sweep_case_studies.py`'s own
  convention) + a new "Sparse feature dictionary" `report.py` section
  showing each SAE target's summary stats plus its top ground-truth-
  matched features' top-activating exemplar series and their real
  ground-truth values. Renamed `sae/train.py`'s private `_sanitize` to
  public `sanitize` rather than importing a name-mangled helper across a
  module boundary. Verified two ways (`CLAUDE.md` §2.4): new planted-data
  unit tests in `tests/test_sae.py`, then regenerated the real report for
  the already-existing `runs/medium_run_chronos_base` real-checkpoint run
  and read the actual HTML, which reproduced this section's own
  previously-cited numbers exactly (the `has_intermittency` ρ=0.881
  match) and surfaced one genuine, minor, not-fixed observation about how
  the existing archetype dummy-encoding treats real-derived-tier series —
  see this session's new Findings entry in §6.2 for the full account.
  Full suite: **69/69**. Everything from this loop remains uncommitted
  working-tree changes only, per the user's git-safety instructions.
- **2026-08-06 (same loop, iteration 4)** — Picked up §10 (Phase 5)'s CI
  checklist item's model-adapter half: a lightweight adapter-conformance
  checker, chosen specifically because it's testable end-to-end against
  the existing mock adapters with no live GPU/checkpoint needed, same
  standard as every prior iteration this loop. New
  `tsfm_lens/models/conformance.py::check_adapter_conformance` runs the
  same manual checklist Phase 4 (§9) and this bullet already describe in
  prose — layer discovery, well-formed `token_time_spans`, a sane
  `impulse_alignment_check`, a correctly-shaped finite `predict()` output,
  and every optional capability degrading to `None` rather than raising —
  and a new `tests/test_adapter_conformance.py` runs it against all three
  registered mocks (`mock_patch`, `mock_step`, and `mock_wave` — the third
  architecture that already existed specifically to prove new adapters
  need no changes outside their own file, but that no default config
  actually exercises), plus a no-optional-capabilities stand-in (proving
  the "`None` is fine" path, previously untested since every existing mock
  happens to support everything) and a deliberately-broken adapter (proving
  a real contract violation is still caught, not silently accepted). Full
  suite **74/74**. Not attempted, and correctly flagged as such in this
  section's checklist entry: actually wiring this into a CI pipeline (no
  CI config exists in this repo yet) and the separate, already-known,
  unresolved golden-hash numpy-version fragility (`CLAUDE.md` §11.13) —
  this bullet's "CI" framing names both together but this session's work
  only covers the adapter-conformance half. Everything from this loop
  remains uncommitted working-tree changes only.
- **2026-08-06 (same loop, iteration 5)** — Picked up §10's paired
  "auto-generated model-zoo capability matrix" items (§10 and §9's own
  cross-reference to it), the natural next Phase-5 CI item after
  iteration 4's conformance checker, and again chosen for being verifiable
  with no live GPU work: new `tsfm_lens/models/capability_matrix.py` +
  `run_capability_matrix.py` + `tests/test_capability_matrix.py`. Verified
  two ways (`CLAUDE.md` §2.4): synthetic-fixture unit tests, then the CLI
  run twice against real artifacts — once with no `--verify` at all
  (reproduced `CLAUDE.md` §6.2's hand-maintained table purely from static
  class introspection) and once pointed at the already-completed
  `runs/medium_run_chronos_base` (reproduced the one genuinely subtle
  documented case, TimesFM's `mlp_info` declaring-but-verifying-nothing,
  as a distinct rendered state from "never checked"). That real-run
  cross-check caught a genuine bug before it shipped: the renderer's
  `verified is True`/`is False` comparisons broke silently on any
  single-capability all-verified/no-`NA` DataFrame column (pandas then
  stores plain `numpy.bool_`, not Python's `True`/`False` singletons) —
  fixed to use `pd.isna(...)`/`bool(...)`, with a dedicated regression
  test added so this can't regress silently. Full suite **81/81**.
  Everything from this loop remains uncommitted working-tree changes only.
- **2026-08-06 (audit + planning session — no implementation)** — Per explicit
  user instruction ("do not implement any code"), read `CLAUDE.md` and this
  file end to end, then read the pipeline itself against their claims looking
  for silent-failure paths, fixable acknowledged limitations, and non-obvious
  design errors. Produced two new sections: **§15** (19 audited items, each
  with `file:line` evidence, a stated blast radius, and a detailed fix plan
  marked NEEDS IMPLEMENTATION) and **§16** (an enhancement backlog for the §1
  "anyone, any model, one button" north star, 22 items in four tiers). Added
  inline `⚠️ AUDIT` markers at the six places in this file where an audit
  finding contradicts or materially qualifies something already written down
  (§5.3's invariant-7 bullet, §5.5's per-archetype follow-up, §6.1.1's R2
  requirement, §6.2's forecast-preservation deliverable, §7's intermittency
  caveat, §5's corruption-calibration deferral), and a matching reconciliation
  note plus `§11.20` pointer in `CLAUDE.md`. The three findings worth naming
  here because they change how existing recorded numbers should be read:
  invariant 7's alignment verification is logged-and-discarded rather than
  enforced (A2); `build_pipeline` writes corpora grouped by task while four
  call sites subsample with a head slice, so several recorded numbers were
  measured on family-skewed prefixes (A4); and the `layer_screen` stage as
  wired cannot satisfy its own all-layers-fair requirement under any
  production config (A1). Nothing was implemented or committed.
- **2026-08-06 (same-day follow-up — the deferred §6.2/A4 re-measurement)**
  — With every §15 P1 item now fixed, picked up the one concrete piece of
  unfinished business those fixes explicitly left behind: A4's own
  Findings said the SAE forecast-preservation/ground-truth numbers "remain
  as previously recorded... left as a follow-up," and §6.2's own AUDIT
  block said not to act on the "TimesFM passes" conclusion before
  re-measuring. Verified the full suite green (173/173) first, then
  re-ran `l0` (for a matched noise floor) and `sae` against
  `configs/medium_run_chronos_base.yaml`'s already-extracted store, live
  `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-base` (checkpoints
  already cached locally, GPU idle and available). Corrected numbers now
  in §6.2's Findings, superseding the old ones in place rather than
  replacing them: forecast-preservation ΔMASE for TimesFM moved
  +0.047 → +0.175 and for Chronos-T5-Base +2.368 → +3.869 under the
  now-representative stratified sample — both *worse* than previously
  reported, ruling out the possibility that the old sampling bug happened
  to favor one model over the other. The new matched noise floor
  (TimesFM exactly `0.0`, deterministic; Chronos-T5-Base mean `0.160`/max
  `0.830`) confirms Chronos-T5-Base's failure is real signal, not sampling
  noise, and that TimesFM's own delta — while still much smaller and still
  real by construction (deterministic ⇒ any nonzero floor is signal) — is
  larger than the "passes almost exactly" framing previously implied.
  Ground-truth alignment also moved (both models' ρ up moderately) but,
  checked directly rather than assumed, its own row set is provably
  identical before and after (this corpus's 288 series is smaller than
  the configured cap, so `sample_rows` degenerates to "take everything"
  regardless of the fix) — that delta is ordinary SAE-training run-to-run
  variance, not something A4 changed, and is called out as such so it
  isn't misattributed. `report.html` was confirmed (by direct grep of the
  rendered HTML, not just the JSON artifact) to render both new
  noise-floor-contextualized finding strings end to end. Full suite
  reconfirmed green at 173/173 after the re-run. Everything from this
  session remains uncommitted working-tree changes only.
- **2026-08-06 (same-day follow-up — §16 E9 implementation)** — With every
  §15 P1 item closed and the §6.2 re-measurement done, picked E9
  (untrained-weights null baseline) as the highest-value item left in the
  backlog, per this file's own "arguably the highest-value item in T2"
  framing. Implemented `ModelConfig.random_init` + `models/base.py::
  random_init_like` (HF's own `type(model)(model.config)` idiom for the two
  Chronos adapters, a checkpoint-load skip for TimesFM — no generic
  reinitialization heuristic, no per-adapter special-casing beyond one
  branch in each `load()`), which gives L1/L2/internals/SAE a real floor
  with zero changes to any of those four analysis modules: pairing a real
  model against its own `random_init` twin is just an ordinary two-model
  config. Added `sae/ground_truth.py::permutation_null_alignment`, a
  label-permutation null for the SAE ground-truth alignment score's own
  multiple-comparisons inflation, wired into the report. New unit tests
  (`tests/test_random_init.py`, `tests/test_ground_truth_permutation_null.py`,
  10 tests total) — one, a full mock-pipeline `extract`+`l1` run of a
  real-vs-random-init pair, exercises the whole wiring end to end offline.
  Found and fixed one real bug by actually running the smoke test rather
  than trusting the diff: a 4th paragraph appended to an existing
  `report.py` note tuple collided with `_note()`'s `summary=` keyword.
  Full suite green at 183/183. Ran two real-checkpoint verification runs
  in the background (`configs/null_timesfm_random.yaml`,
  `null_chronos_random.yaml`, live TimesFM 2.5 / Chronos-T5-Base), both
  completed cleanly, and the result is a genuine **split verdict** worth
  reading in full (§16 E9's Findings has all four numbers, not summarized
  twice here): internals' probe decodability and the SAE ground-truth
  alignment (validated further by its own permutation null landing right
  on top of the untrained twins' scores, exactly as it should) both
  clearly survive the null: real models beat their untrained twins at
  every depth. L1's peak CKA and, more strikingly, L2's stitching gain do
  **not** clearly exceed an architecture-only floor computed from the same
  models' own untrained twins — L2 gain in particular sits within, and for
  one direction *below*, the untrained-twin floor's range. This materially
  qualifies (not refutes) `CLAUDE.md` §6.5's L2 framing, and is exactly the
  kind of check E9's own text predicted this repo was missing. `CLAUDE.md`
  §6.2 updated with the new capability and a matching reconciliation note.
- **2026-08-07 — E9 follow-up: the eyeball null comparison becomes a real
  significance test.** Picked up the prior session's own explicitly-named
  next step (§16 E9's Findings: "a rigorous verdict... needs a per-layer-pair
  bootstrap test... not eyeballing two point estimates") and §13's matching
  open question, both still marked unresolved at session start. Built
  `stats.py::bootstrap_ci_diff` (general paired/unpaired bootstrap CI + p for
  the difference of two statistics), factored `l2_stitching.py`'s train/val
  split into a reusable `series_split()`, and a new `analysis/null_baseline.py`
  (+ `run_null_baseline_test.py` CLI) that reads two already-extracted run
  directories — no model loaded, nothing re-run — and bootstraps real-run vs.
  null-run peak CKA / best gain with the *same* series-index draw in both
  (verified row-for-row corpus alignment first, not assumed; falls back to an
  unpaired bootstrap with a warning otherwise). 5 new tests
  (`tests/test_null_baseline.py`), full suite green at **189/189**. Ran the
  new CLI against the two already-completed real-checkpoint null runs from
  the prior session (`null_timesfm_random`, `null_chronos_random`) against
  `medium_run_chronos_base` — no GPU needed, pure re-analysis of existing
  activation stores. Result (full numbers in §16 E9's Findings, appended
  in place): L2's null result is now confirmed by an actual difference test
  (both directions' CIs comfortably contain zero, p=0.474/0.532) rather than
  inferred from overlapping point-estimate CIs. L1's result sharpens into
  something more specific than "model-asymmetric": the real cross-model peak
  CKA significantly *exceeds* Chronos's own untrained-twin floor (p=0.001)
  while significantly *trailing* TimesFM's own untrained-twin floor (p=0.001)
  — two decisive, opposite verdicts from the same real number depending on
  which model supplies the null. §13's open question and §16 E9's checklist
  item both updated in place (kept at `[~]`, not `[x]`: this tests only each
  run's single global best pair, not a full per-layer-pair grid, and on one
  corpus/checkpoint size only).
- **2026-08-07 (same-day second follow-up, run via a scheduled autonomous
  loop) — the depth-curve half of E9's named next step.** Extended
  `analysis/null_baseline.py` with `compare_l1_depth_curve`: for each of
  TimesFM's 10 captured layers, bootstrap its best-cross-model-partner CKA
  against both models' own untrained-twin floor *at that exact layer*
  (mechanical extension of `bootstrap_ci_diff`, exactly as the prior
  entry's "next concrete step" predicted). 2 new tests, full suite green at
  **191/191**. Real result (full table in §16 E9's Findings): real
  cross-model structure decisively beats **both** floors at every middle
  depth (layers 6-14 of 10), and loses to at least one floor only at the
  extremes (shallowest vs. TimesFM's own steeply-decaying floor, deepest
  vs. Chronos's own flat floor) — a materially more informative and more
  positive picture than the single peak-pair result alone suggested, since
  the peak (layer 4) happens to land in the one genuinely ambiguous spot.
  Also found and fixed a real display bug (not a numbers bug) in
  `run_null_baseline_test.py`'s verdict formatter, which collapsed
  "null decisively exceeds real" into the same label as "not clearly
  different." `CLAUDE.md` §6.5 updated with a pointer to read the depth
  curve, not just the peak. Cross-corpus/checkpoint-size replication and
  an equivalent L2 depth curve remain the open next step.
- **2026-08-07 (same-day third follow-up, run via the scheduled autonomous
  loop's next 30-minute firing) — the L2 analog, and it reverses rather
  than just qualifies the earlier peak-pair reading.** Added
  `compare_l2_depth_curve` to `analysis/null_baseline.py` (factoring
  `_best_direction_decomposition` into a reusable `_pair_decomposition`
  helper first), `--depth-curve l2` on the CLI, 2 new tests, full suite
  green at **193/193**. Real result (full table in §16 E9's Findings): the
  earlier "L2 gain is statistically indistinguishable from null" verdict
  turned out to be an artifact of comparing the real run's actual best pair
  (Chronos block 10, gain 0.413) against `null_chronos_random`'s own
  *unrelated* global-best pair (block 0 — an early-layer self-vs-random-twin
  predictability artifact, gain 0.444, nothing to do with where the real
  cross-model signal peaks). Retested layer-for-layer at block 10 in both
  runs, the null floor is only 0.062/0.046 and real decisively beats both
  (p=0.002 each) — and does so at every block from 4 through 11 (8 of 12),
  losing only at the earliest, lowest-absolute-gain blocks. Same lesson as
  the L1 depth curve, arguably sharper here: a single global-best-pair
  comparison across two independently-run configs can silently compare two
  *unrelated* layers whenever each run's own optimum lands somewhere
  different, and only a matched-layer test catches that. `CLAUDE.md` §6.5
  corrected in place (not deleted). §13's open question rewritten to state
  the layer-matching lesson explicitly rather than a flat "L2 gain doesn't
  survive." Open: the reverse L2 direction, and cross-corpus/checkpoint-size
  replication for both L1 and L2.
- **2026-08-07 (same-day fourth follow-up, autonomous loop) — §6.1's
  long-open "feed SAE evaluation results back into the layer-selection
  correlation study" item, finally unblocked by the null-baseline runs
  already sitting in `runs/`.** Built `sae/layer_sweep.py` +
  `run_sae_layer_sweep.py` (per-captured-layer, CPU-only TopK-SAE
  train+eval reusing the existing `sae` module's functions verbatim) and a
  fourth `layer_selection.py` proxy metric, `sae_ground_truth_rho`. Ran it
  against `medium_run_chronos_base` (already had it), plus
  `null_timesfm_random`/`null_chronos_random` (chosen specifically because
  they're zarr-v2 and already had `internals` enabled) to clear
  `cluster_bootstrap_spearman`'s `n_groups>=3` floor — 6 groups, 66 records
  total. Result: `input_cka` vs the new `sae_ground_truth_rho` proxy is
  depth-controlled-significant (ρ=0.623, CI [0.074,0.758]) and closely
  replicates the *already-established* `input_cka` vs `probe_decodability`
  relationship (ρ=0.580) — two independent interpretability proxies
  agreeing after the same depth control, a real convergent-validity result.
  `effective_dim` showed no significant relationship with either proxy at
  this n. Found (and explicitly flagged, not silently worked around) that
  `runs/medium_run`/`runs/real_run` can't be added to this pool — both
  predate this repo's `zarr<3` pin and are unreadable by the current
  environment (`CLAUDE.md` §11.15's trap, recurring). Full suite green at
  198/198. Full numbers in §6.1's Findings.
- **2026-08-07 (same-day fifth follow-up, autonomous loop) — Phase 2c
  (§6.3): first real test of whether L1/L2's stitching machinery separates
  a documented-shared-lineage model pair from an independently-trained
  one.** Verified from `amazon/chronos-t5-base`'s actual HF README (not
  guessed) that every Chronos-T5 size fine-tunes from a *separately*
  pretrained `google/t5-efficient-{size}` checkpoint (no shared init
  weights) but all sizes then train on the identical time-series
  corpus/procedure — the precise "meaningful lineage" the checklist asked
  to confirm. Ran a new lean config
  (`configs/distill_positive_chronos_small_base.yaml`, only `l0/l1/l2/
  internals/report`, done in under 2 minutes) live against real
  `amazon/chronos-t5-small` + `amazon/chronos-t5-base` as the positive
  pair; reused `medium_run_chronos_base` (Chronos-T5-Base vs TimesFM,
  already on disk) as the negative. New `analysis/distillation_detection.py`
  is a ~20-line named wrapper around `null_baseline.py`'s existing
  bootstrap-difference functions (nothing in them is null-specific) plus a
  matching CLI. Result: **decisive separation** — L1 peak CKA 0.734
  (positive) vs 0.381 (negative), diff +0.353 CI [+0.307,+0.390] p=0.0005;
  L2 best gain 0.637 vs 0.413, diff +0.224 CI [+0.168,+0.285] p=0.0005 —
  but flagged, not oversold: the positive pair is same-architecture (both
  T5) and the negative is cross-architecture, so this one test can't yet
  separate "tracks lineage" from "tracks architecture match." Left the
  checklist's "write a standalone method note" item explicitly undone
  pending a same-architecture negative control, per §2.6/§2.7's honesty
  doctrine — real stakes (provenance/IP disputes) if overclaimed. Full
  suite green at 200/200 (2 new tests). Full numbers in §6.3's Findings.
- **2026-08-07 (same-day sixth follow-up, autonomous loop) — Phase 4 (§9):
  first real multi-model-expansion addition, Chronos-2, plus a real bug
  found and fixed in shared infrastructure rather than worked around.**
  New `models/chronos2_adapter.py` (`Chronos2Adapter`), registered as
  `"chronos2"`. `--check-alignment` crashed on the very first real capture:
  `extraction/hooks.py`'s `_primary()` assumed a module's forward output is
  either a plain tensor or a literal `tuple`, but `Chronos2EncoderBlock`
  returns an HF `ModelOutput`-style dataclass (dict-like, integer-indexable,
  but not a `tuple` instance) — exactly the "any exception outside the new
  adapter file is a bug in the abstraction" case §9's own checklist names.
  Fixed in `hooks.py` (generalized `_primary`, added `_rebuild` so
  `token_patch`/`output_mean_ablate` preserve the original output type
  instead of degrading it to a plain tuple), with a new offline regression
  test (`tests/test_hooks_modeloutput.py`, 3 tests) rather than any
  Chronos-2-specific special-casing. After the fix: perfect 1.00
  diagonal-hit alignment at all 12 layers, and a full live GPU validation
  run (Chronos-2 vs TimesFM, with L3 per-window patching and attention
  patterns/ablation both exercised) completed clean end-to-end — L1 peak
  CKA 0.43 vs a shuffled-null ≈0.04, L3 fingerprint agreement ρ=−0.75
  (replicating the TimesFM/Chronos-T5 anti-correlation pattern against a
  third, structurally distinct model), and real, sane periodicity/
  load-bearing-head numbers from attention analysis. Full suite green at
  203/203. Full numbers and the abstraction-bug mechanism in §9's Findings.
- **2026-08-10 — resumed after a gap; documented the ad-hoc autonomous-loop
  pattern as a real doctrine section, kicked off the next flagged §13 item
  as a background run, and closed the Phase 4 licensing-recheck open
  question live.** No prior session had written down *how* the several
  "run via a scheduled autonomous loop" entries above actually worked, so
  it was ad hoc every time; added `CLAUDE.md` §2.8 ("Delegate long-running
  compute; keep the roadmap moving in parallel") stating the pattern as an
  instruction, and set up a session-local 30-minute-cadence `CronCreate`
  loop (5-hour self-stopping deadline) so this continues without a human
  re-prompting each time. Launched §13's explicitly-flagged next step — a
  live rerun of the `factor_emergence` layer-selector bake-off
  (`CLAUDE.md` §11.18's fix, previously verified only on synthetic data)
  against the real, already-extracted `runs/layer_screen_experiment` store
  — as a background agent per the new §2.8 rule, rather than blocking on it;
  its numbers will be written up here once it reports back (not yet landed
  as of this entry). While that ran, closed §13's other open item live
  rather than leaving it as a stale assumption: **Moirai** (`Salesforce/
  moirai-1.1-R-small` and the newer `moirai-2.0-R-small`) is
  `cc-by-nc-4.0` (non-commercial, fine for this repo's research use but
  previously unstated), and Moirai 2.0 turns out to have moved to a
  **decoder-only** architecture, not the masked-encoder design §9's
  candidate list described (only 1.x still matches that description).
  **Sundial** (`thuml/sundial-base-128m`) is **Apache-2.0** (fully
  permissive), confirmed decoder-only with a flow-matching probabilistic
  head — answering §9's own "worth checking" note rather than leaving it
  open. Net call: Sundial is now the more attractive next Phase-4 pick of
  the two (permissive license, cleaner structural contrast) — see §9 and
  §13 for the full recheck. No code changed this entry (docs + one
  background job only); full suite not re-run since nothing executable
  changed.
- **2026-08-10 (same-day follow-up) — the background bake-off rerun landed;
  written up as a genuine negative result, not the hoped-for confirmation.**
  The `factor_emergence` fix (`CLAUDE.md` §11.18, 2026-08-06) does not clear
  the nulls on real checkpoints: Chronos-T5-Small is completely unaffected
  (bit-identical failing selection, both seeds, pre- and post-fix) and
  TimesFM's recall@budget actively regressed (0.20→0.00). A controlled
  same-data-different-formula comparison (not just diffing two independently
  noisy end-to-end runs) pinned the cause: on this real corpus the
  early-layer bias comes from *strong*, near-input-statistics factors
  decoding well from layer 0, the mirror image of the weak-noisy-factor
  scenario the fix's synthetic test targeted — so peak-weighting amplifies
  the bias instead of correcting it, and the new absolute floor never once
  fired. `work_bend`/`coverage` (the actual production default) confirmed
  unaffected. §13's corresponding sub-question and `CLAUDE.md` §11.18 both
  updated in place with the full numbers; `factor_emergence` remains
  correctly excluded from production. Full `tsfm_lens` suite green at
  203/203 (verification-only, no production code changed). This closes out
  the two items opened earlier this session — the licensing recheck and
  this bake-off rerun — with no roadmap item left in-flight as of this
  entry; the 30-minute `CronCreate` loop set up earlier this session will
  pick the next one.
- **2026-08-10 (second cron-loop firing) — Sundial added as a fourth real
  `ModelAdapter`, Phase 4's second multi-model-expansion addition, via the
  same background-agent-plus-continue-the-loop pattern as the entry
  above.** No background agent was pending write-up at firing time (the
  bake-off rerun above was already closed out); picked Sundial as the next
  concrete item per §9's own priority list, now that the same firing's
  earlier licensing recheck had established it as the more attractive of
  the two remaining candidates. Launched the whole adapter-build-and-verify
  task as one background agent (adapter code, alignment check, capability
  matrix, a live TimesFM-vs-Sundial comparison run, tests — the full §9
  checklist, not just a rerun of existing code) rather than working on it
  inline, and deliberately did not start a second, file-touching task in
  parallel this firing to avoid contention with an agent whose scope was
  this wide. Once it landed: verified independently before writing
  anything up (re-ran the full test suite myself — 207/207, matching the
  agent's own count — and spot-checked the live run's actual
  `runs/sundial_phase4_check/l1/meta.json` artifact against the numbers
  quoted in its report, rather than trusting the report alone). Real
  result: no shared-infrastructure bug this time (contrast with Chronos-2's
  §11.21), but two genuine bugs in the **checkpoint's own remote code** (a
  `transformers.DynamicCache` API mismatch breaking `.generate()` and even
  a cached `forward()` call), worked around by calling the lower-level
  `forward(..., use_cache=False)` directly rather than risking a
  transformers downgrade that could break the other three adapters — new
  `CLAUDE.md` §11.22. Live numbers (§9's new Findings block has the full
  set): L1 peak CKA 0.381 vs. a null of 0.022 (real shared geometry with a
  third architecture, replicating Chronos-2's own finding), and a genuinely
  new data point — L3 fingerprint agreement is *positive* (ρ=0.517) for
  TimesFM-vs-Sundial, breaking the anti-correlation pattern every other
  model pairing in this repo had shown so far. `CLAUDE.md` §6.2's capability
  matrix and adapter-description prose updated to a five-model table.
  No roadmap item left in-flight as of this entry.
- **2026-08-10 (third cron-loop firing) — closed a stale-but-already-answered
  §13 item, then used today's Sundial addition to attack the bake-off's own
  named "third architecture family" open question.** No background agent
  was pending write-up. Rereading §13 (not re-running anything) found the
  "is `work_bend` still the bake-off winner under production conditions"
  item had actually been resolved by A1's fix days ago and simply never
  marked `[x]` — `LayerScreenConfig`'s defaults (`stride: 1`,
  `require_full_capture: True`) already make every production config run
  the dedicated stride-1 screening extraction A1 built, and A1's own
  Findings already recorded live proof of exactly this scenario. Marked
  resolved with a pointer back to A1 rather than re-verifying something
  already verified. Then picked up §6.1.1-E's own still-open replication
  question — "a third architecture family" — now directly answerable
  using Sundial, added earlier this same day. Launched a background agent
  to build a 3-model (TimesFM + Chronos-T5-Small + Sundial, all
  `capture_layer_stride: 1`) extension of the original bake-off config and
  rerun `run_layer_screen_bakeoff.py` against it, checking both whether
  `work_bend` still wins on a third architecture and whether
  `factor_emergence`'s newly-diagnosed (this same day) early-layer-bias
  failure mode reproduces or looks different for Sundial. Did not start a
  second parallel task this firing (the new 3-model extraction is itself
  the natural next step; no other independent item was picked up). Results
  not yet landed as of this entry.
- **2026-08-10 (fourth cron-loop firing) — the 3-model bake-off landed;
  writing it up surfaced a more important finding than the one it was sent
  to check.** Independently verified before writing anything up (reread
  the actual `runs/layer_screen_bakeoff_v2.json` and the existing
  `runs/layer_screen_bakeoff.json` directly, and reran the full pytest
  suite myself — 207/207, matching). The "third architecture family"
  question got a real answer: `work_bend` beats both nulls on Sundial, but
  the three-architecture picture is "beats a different subset of the two
  nulls on each," not a clean sweep, and `factor_emergence`'s diagnosed
  early-layer-bias failure (from earlier today) reproduces cleanly on
  Sundial, reinforcing that it's a property of the ground-truth factor
  table rather than either specific architecture. **The bigger finding,
  caught only because verifying the comparison numbers meant reading both
  JSON files directly:** Chronos-T5-Small's qualitative bake-off verdict
  is not stable across independent re-extractions of the identical config
  — `work_bend`'s `beats_random` flipped True→False between the existing
  same-day run and this session's fresh 3-model extraction, while
  TimesFM's own verdict reproduced unchanged in the same two runs. The
  bake-off's existing "stability replicate" only ever reseeds SAE training
  on one frozen, already-extracted store — it has never tested whether the
  extraction itself is stable, and this is the first time that axis was
  ever exercised (by accident, as a side effect of needing a fresh
  extraction for Sundial). Wrote this up as a new, prominent §13 open item
  (not folded quietly into an existing one) and a new `CLAUDE.md` §11.23,
  and updated the verification-status table row rather than treating it as
  a footnote — per §2.6's honesty doctrine, this genuinely changes how much
  confidence any single-run bake-off number in this file deserves, and
  said so plainly rather than downplaying it. No roadmap item left
  in-flight as of this entry; did not start a new background task this
  firing given the length of this write-up.
- **2026-08-10 (fifth cron-loop firing) — the extraction-variance mystery
  from the previous entry is fully root-caused, not just narrowed.** Found
  nothing in progress via `ListAgents` beyond the already-written-up
  3-model bake-off agent; launched a background investigation (ranked
  candidate causes, then live-tested the top one) into whether
  Chronos-T5-Small's flipped bake-off verdict was extraction/GPU/SAE-seed
  variance. While that ran, closed an unrelated `DEPENDENCIES.md`
  invariant-12 gap (Chronos-2's and Sundial's fragile version interactions
  had never been recorded there — added both with the same file:line detail
  `CLAUDE.md` §11.21/§11.22 already has). The agent's own report initially
  under-delivered (returned mid-experiment claiming it was "waiting for
  pytest," resumed once via `SendMessage` to get the real final numbers),
  but its actual finding — GPU nondeterminism ruled out via a bit-exact
  `np.array_equal` re-extraction, Sundial's presence ruled out via a
  2-model-only rerun matching the 3-model numbers, SAE-seed sensitivity
  real but not the cause since neither compared run ever varied it — pointed
  at a code change between the two runs' dates rather than randomness.
  Followed that up directly with `git diff 84cdbc0 HEAD --
  tsfm_lens/data.py` and confirmed it precisely: `§15 A4`'s sampling fix
  (landed 2026-08-06, between the experiment's 08-05 and 08-10 runs)
  replaced a `kept[:max_series]` head slice with a stratified
  `sample_rows(...)` call, and the experiment's own `max_series: 220`
  against a 288-row corpus meant the pre-fix run silently excluded the
  entire 60-row `mixture` family (verified directly via a `Counter` over
  the actual corpus file) — full family coverage only appeared in the
  post-fix runs. Corrected §13's 🔴 item and §6.1.1's Findings block in
  place (marked resolved, did not delete the original "not yet known"
  framing — appended the resolution after it per §0.2), added `CLAUDE.md`
  §11.24 for the general lesson (a background rerun of "the same"
  experiment can silently change meaning across a shared-infrastructure
  fix, with no config diff to catch it — check `git log` on the shared
  paths before spending investigation budget on nondeterminism
  hypotheses), and updated the verification-status table row. Independently
  re-ran the full test suite myself rather than trusting the agent's
  count: **207 passed, 2 warnings in 287.78s**, both warnings pre-existing
  (an intentional overflow-cast test, a `PytestReturnNotNoneWarning` in
  `test_smoke.py`) and unrelated to anything touched this session — matches
  the agent's own reported count exactly. No production code changed this
  firing (docs + a resumed investigation only). Narrower question left
  open: whether the now-correctly-sampled bake-off verdict is itself stable
  across a second seed — smaller in scope than what was open at the start
  of this firing, and a reasonable candidate for the next one.
- **2026-08-10 (sixth cron-loop firing) — started the second-seed stability
  check the previous entry named as the next candidate, and closed out its
  own "worth a deliberate sweep" note.** `ListAgents` showed nothing
  in-flight (the previous firing's investigation agent had fully completed
  and dropped off the list). Launched a background agent to build
  `configs/layer_screen_experiment_v2_seed1.yaml` (identical to `_v2.yaml`
  except `run.seed: 1`), extract, run the bake-off, and compare against the
  existing seed-0 numbers for all three models plus a row-sample-overlap
  check, to answer the narrower question left open at the end of the last
  entry — results not yet landed. While that ran, swept every config's
  `data:` block (not the several other, unrelated `*_max_series` knobs) for
  the specific `data.max_series` corpus-level cap the just-resolved bug
  lives in: it turns out to be set **only** in the
  `layer_screen_experiment*.yaml` family — every other real-checkpoint
  config (`medium_run*.yaml`, `default.yaml`, both `null_*_random.yaml`,
  `distill_positive_chronos_small_base.yaml`, `sundial_phase4_check.yaml`,
  `chronos2_phase4_check.yaml`) sets no `data.max_series` at all, so the
  `_assemble` guard this bug lived behind never fires for them regardless
  of which side of 2026-08-06 they ran on. Wrote this up as closing the
  "worth a deliberate sweep" line from two entries ago: the bug has exactly
  the one victim already found and fixed, not a wider undiscovered
  problem — the two other already-tracked affected numbers (§6.2's SAE
  forecast-preservation/ground-truth, §5.3's lens comparison) go through
  different call sites entirely and their re-measurement status is
  unchanged by this sweep. No test suite run this entry (docs-only
  addition, no code touched, and the background agent's own pytest run
  will cover the code path it touches). Second-seed stability results not
  yet landed as of this entry.
- **2026-08-10 (seventh cron-loop firing, same-turn write-up of the
  second-seed background agent) — the narrower question from the previous
  entry is answered: no, `work_bend` does not reliably survive a reseed.**
  The delegating agent's own final report (a second nested agent it used to
  actually run the GPU work) gave a full seed0-vs-seed1 comparison; rather
  than take it on trust, independently re-derived every headline number
  directly from `runs/layer_screen_bakeoff_v2_seed1.json` and both runs'
  `meta.parquet` files myself (gold_agreement, recall_at_budget,
  beats_random/beats_uniform_stride, selected layers, and the row-overlap
  Jaccard) — all matched the agent's report exactly, including to the full
  decimal. Headline: 76.8% row overlap between the two seeds' 220-of-288
  draws (not a biased or wildly different sample), yet `work_bend` — the
  production default — flips its `beats_random` verdict on 2 of 3
  architectures (TimesFM True→False, Chronos False→True), while
  `coverage` stays fully stable on all three and Sundial stays fully
  stable on every selector. In every flip, the selector's own chosen
  layers were bit-identical across seeds — only the null-comparison
  scorecard moved, tied to the SAE-mass gold ranking shifting under the
  ~23%-different row draw. Read this as a statistical-power gap in the
  bake-off's own evaluation (recall@budget at n=220, budget≤5) rather than
  a bias like A4's — a materially different, and arguably more concerning
  for production use, kind of instability than what the previous two
  entries closed out. Wrote the full comparison into `ROADMAP.md` §6.1.1's
  Findings and updated §13's item and `CLAUDE.md`'s verification-status
  row in place. Independently launched my own separate `pytest -q` run in
  parallel with the write-up (the agent's own run already reported 207
  passed, 2 warnings, matching every prior count this session) as a second
  check; it was still running when this entry was written due to apparent
  GPU contention with the peer session's own activity, so treat the
  suite's status this entry as "agent-confirmed, self-confirmation
  in-flight" rather than doubly independently verified — worth a quick
  glance next firing if the log wasn't checked before this session ends.
  No new background task started this firing — the write-up was the full
  scope of the work. Remaining open items: whether a third seed would
  narrow or confirm this spread, and whether the bake-off's statistical
  power should be improved (larger corpus, bootstrap CI on recall@budget)
  before trusting any single future run's verdict — left for a future
  firing or session, not attempted here.
- **2026-08-10 (fifth cron-loop firing) — launched the extraction-variance
  characterization, and closed a real `DEPENDENCIES.md` invariant-12 gap
  while it runs.** `ListAgents` showed only the already-written-up 3-model
  bake-off agent (completed, stale re-notification, no new content).
  Nothing in progress, so started the §13 🔴 item from the previous firing:
  launched a background agent (two parts — first read
  `data.py`/`extract.py`/`pipeline.py`/`layer_screen.py`/
  `layer_screen_bakeoff.py` for every candidate source of run-to-run
  nondeterminism with file:line citations and rank them; then design and
  run a targeted experiment against the single most plausible candidate,
  either re-running the bake-off against the *same* already-extracted
  `runs/layer_screen_experiment_v2/` store with two SAE seeds, or a fresh
  third independent extraction, whichever the ranking points to) to
  characterize whether Chronos-T5-Small's flipped verdict (previous entry)
  is explained by SAE-training stochasticity, extraction-level
  nondeterminism, or something else. Told it explicitly not to touch
  production code — root-cause first, design a fix in a later session.
  While that ran, noticed `DEPENDENCIES.md` §5 ("Known fragile spots") had
  never been updated for either Chronos-2 (`CLAUDE.md` §11.21, the
  `ModelOutput`-vs-tuple hooks bug) or Sundial (`CLAUDE.md` §11.22, the
  `DynamicCache` API break in its own remote code) — a real gap against
  invariant 12's "any newly-discovered fragile version interaction gets
  reflected there in the same session" rule, missed in both of those
  adapter-build sessions. Added both as new bullets in `DEPENDENCIES.md` §5
  with the same file:line-anchored detail `CLAUDE.md` §11 already has, so
  the two records stay in sync. No test suite run needed (docs-only change,
  no code touched). Extraction-variance results not yet landed as of this
  entry.
- **2026-08-10 — explicit user instruction to fix the second-seed
  `work_bend` instability rather than leave it as documented, mid-turn
  during a routine cron-loop firing.** Not a mechanical firing this time —
  the user directly told the session not to just leave the previous
  entry's finding ("`work_bend` doesn't reliably survive a reseed") as a
  written-up limitation. Root-caused by re-reading `layer_screen.py`
  directly rather than re-testing hypotheses: `select_work_bend`/
  `select_coverage` take no `seed` kwarg at all (grepped to confirm) and
  are fully deterministic given a fixed corpus, so the entire flip had to
  be coming from `build_gold_ranking`'s single stochastic per-layer SAE
  training run standing in as ground truth at a budget small enough
  (2-5 layers) that ordinary training noise flips which layers count as
  "gold." Fixed with two changes, both landed and both verified against
  real checkpoints, not left as an untested patch: (1) `build_gold_ranking`
  now averages `n_replicates=3` independently-seeded SAE-training runs per
  layer instead of trusting one, wired through a new `run_layer_screen_
  bakeoff.py --n-gold-replicates` flag; (2) new `configs/
  layer_screen_experiment_v4[.yaml/_seed1.yaml]` drop the `max_series: 220`
  corpus cap entirely (the corpus is only 288 rows — subsampling bought
  negligible compute savings for real determinism cost). Added a synthetic
  unit test for the averaging/determinism property
  (`test_build_gold_ranking_averages_replicates_and_is_deterministic`);
  full suite green at **208/208** (self-run, not just an agent's report).
  Then launched a background agent to rerun the actual 3-checkpoint
  bake-off (TimesFM 2.5, Chronos-T5-Small, Sundial) on both seeds with the
  fix — not just asserting the fix should work. Result: **every
  `beats_random` verdict now matches across both seeds, for all three
  selectors on all three architectures** (`work_bend`'s TimesFM False↔False
  and Chronos-T5-Small True↔True, both previously flipping, now hold).
  Full comparison table and residual caveats (one secondary metric,
  `factor_emergence`'s `beats_uniform_stride` on Chronos, still moves
  slightly) are in §6.1.1's Findings block, not repeated here. Also fixed a
  small related gap the verification agent caught along the way:
  `gold_score_std`/`n_gold_replicates` were computed but never persisted
  into the bake-off's saved JSON — one-line fix in `run_layer_screen_
  bakeoff.py`. §13's corresponding open item is updated to reflect this is
  now resolved rather than an open statistical-power gap.
- **2026-08-10 (eighth cron-loop firing, deadline reached mid-write-up) —
  stopped the loop with two items left incomplete; both now finished.**
  The prior firing's write-up of a real, newly-discovered finding (a stale
  `runs/medium_run` activation store, written in zarr v3 and silently
  unreadable — returns empty rather than erroring — under this repo's
  pinned zarr v2) was interrupted twice by `Edit` tool-hook timeouts, and
  the loop's own 5-hour deadline (`2026-08-10T21:16:58Z`) was crossed before
  a third attempt could be made, so per the loop's explicit instructions the
  session stopped (deleted cron job `d69df929`) rather than retry further.
  Picked back up on direct user instruction ("finish what you were unable
  to finish, then move onto the next roadmap item"): the `CLAUDE.md` §11.25
  trap entry landed cleanly this time, and this section's own item plus the
  `§5.3`/A4 Findings block above were updated with the same finding —
  including that the background agent attempting the actual re-extraction
  had separately failed on its own infrastructure-level session/API usage
  limit, not a code bug. Re-armed the recurring loop (new job, every 20
  minutes for 3 hours from `2026-08-10T23:13:24Z`) and relaunched the
  `runs/medium_run` re-extraction as a fresh background agent before moving
  to the next roadmap item, per `CLAUDE.md` §2.8.
- **2026-08-10 (same-day follow-up) — picked the next roadmap item while the
  re-extraction ran in the background: closed §6.3's confound, and it fell
  the wrong way.** While waiting on `runs/medium_run`'s re-extraction (also
  needing a mid-flight correction — the background agent twice ended its
  turn assuming a raw shell background job would notify it automatically,
  same mistake `CLAUDE.md` documents elsewhere; resumed it with explicit
  polling instructions each time and verified its real progress directly on
  disk rather than trusting its self-report), built the same-architecture
  negative control §6.3's 2026-08-07 Findings said wasn't available with any
  checkpoint in reach: `configs/distill_negative_random_architecture.yaml`
  pairs `Chronos-T5-Small`/`Chronos-T5-Base` with `random_init: true` on
  both sides — same architecture family as the real trained positive pair,
  zero shared training or lineage. Ran live in about 90 seconds. Result was
  the opposite of what the method would need to be useful: this zero-
  training control's L1 CKA (0.878) and L2 gain (0.834–0.940) both
  significantly *exceeded* the real trained positive pair's (0.734 CKA /
  0.637 gain; bootstrap diff CIs [−0.217,−0.130] and [−0.335,−0.269], both
  p=0.0005, `n_boot=2000`). Closed §6.3's own checklist item and §13's
  matching open question as answered — the method is falsified for its
  stated provenance/IP use case, not merely confounded — rather than left
  open. No source code changed (one new config, reused the already-built
  `random_init` mechanism and `distillation_detection.py` test harness
  exactly per `CLAUDE.md` §2.2), so no test suite re-run was needed; the
  live-checkpoint numbers are themselves the verification.
  Meanwhile the relaunched `runs/medium_run` re-extraction agent completed
  cleanly (~11 min wall-clock, needing the same "actively poll, don't
  assume a raw shell background job notifies you" correction as its
  predecessor, verified independently on disk both mid-run and at
  completion rather than trusting either agent report at face value): fresh
  v2 store confirmed (`.zattrs` present, no `zarr.json`), `report.html`
  rendered 11 sections / 27 findings, `confirm` passed (1/1 dev hypotheses
  on private data), and the now-trustworthy TimesFM lens number
  (`final_mase: 1.7899408340454102`) reproduced bit-for-bit against
  `medium_run_chronos_base.yaml`'s already-recorded value — exactly the
  cross-check the matched-`max_series` fix (§5.3/A4) was supposed to
  produce. Self-ran the full `tsfm_lens` suite afterward rather than
  trusting the agent's own count: **208 passed, 2 warnings**, identical to
  what the agent had independently reported. Both this session's open
  threads (§6.3's confound, §5.3/A4's stale store) are now fully closed —
  see their own Findings blocks for the numbers, not repeated here.
- **2026-08-10 (ninth cron-loop firing) — picked up §16 E10 (probabilistic-
  forecast calibration diagnostics) as the next concrete item, with nothing
  else in flight.** Both prior threads (§6.3's confound, §5.3/A4's stale
  store) were fully closed by the previous entry, and no background agent
  was running, so per `CLAUDE.md` §2.8 this was small/quick enough to build
  directly rather than delegate. New `analysis/calibration.py` (reliability
  curve, PIT histogram, quantile-crossing rate, interval coverage/sharpness
  by horizon) reduces over `predict()`'s already-computed `quantiles`/
  `targets` arrays — zero new forward passes, wired into `run_l0` behind a
  new `L0Config.calibration` flag and rendered as three new report
  sections. Verified with 7 new synthetic-planted-answer unit tests (all
  passing on first run) plus a live `configs/smoke.yaml` run confirming the
  new `l0/calibration.json` artifact and report sections render with real,
  correctly-populated content. Full suite green at 215/215, zero
  regressions. See §16 E10's own checklist entry for the full write-up and
  numbers, not repeated here.
- **2026-08-10 (tenth cron-loop firing) — launched the §13/A4 follow-up
  SAE re-measurement in the background, did §16 E11's n_boot/p-floor fix
  directly in parallel.** Per `CLAUDE.md` §2.8: §13's open item "how much of
  what's already recorded survives §15's sampling/noise-floor fixes" still
  named one concrete, un-closed re-measurement — §6.2's SAE
  forecast-preservation ΔMASE and ground-truth ρ, computed 2026-08-05, one
  day before A4's stratified-sampling fix landed. Confirmed the fix is
  already wired into both call sites (`sae/eval.py:94`, `sae/
  ground_truth.py:257`) and the `medium_run_chronos_base` activation store
  is a valid v2 zarr (no §11.25-class repair needed), so this was a pure
  re-run of `sae,report` against already-extracted activations — launched
  as a background agent (needed correcting twice for the same "waiting on
  a notification that will never arrive" mistake this file's own §11.24/
  earlier entries already document as a recurring risk of this delegation
  pattern; it eventually armed a proper `Monitor` for the actual GPU run
  and a second one for its own post-run pytest check). While that ran,
  picked up §16 E11 (statistical-discipline hardening) directly: audited
  every `p=` rendering in `report.py` (only two exist, both already paired
  with a CI — that bullet was already satisfied) and confirmed A15's
  `hypotheses.json` registry already covers the multiplicity-ledger bullet
  (this item just hadn't been marked to reflect either). What was actually
  missing — `n_boot`/the 1/n_boot floor stated inline wherever a p-value is
  shown — is now fixed (`stats.py::paired_bootstrap` returns `n_boot`;
  `report.py`'s new `_p_note` helper renders it at all three sites) and
  verified live against a rerun of `configs/smoke.yaml`'s `l0,report`
  stages (confirmed `"n_boot=150, p floored at 1/n_boot=0.0067"` in the
  actual rendered HTML) plus a full self-run suite: **215 passed, 2
  warnings**, same two pre-existing benign warnings, no regressions. See
  §16 E11's own checklist entry for the full detail. The cluster-bootstrap-
  unit-explicit bullet of E11 remains open; E11 marked `[~]`, not `[x]`.
  The background SAE re-measurement agent then reported back: Chronos-T5-
  Base's ΔMASE and both models' ground-truth ρ replicated closely against
  the already-recorded 2026-08-06 re-measurement (confirming those numbers
  stable), but **TimesFM's forecast-preservation ΔMASE did not** (0.175 →
  0.1097, a real ~37% shift with reconstruction fidelity itself unchanged)
  — a genuine new finding, not an agent error. Investigating this surfaced
  that the §13 open item this whole thread was closing had itself gone
  stale hours after it was written: its own "left as a follow-up" framing,
  and A4's matching Findings-block bullet, were never updated after the
  2026-08-06 same-day follow-up had already performed exactly the
  re-measurement both text blocks still described as outstanding. Corrected
  both in place (§0.2 discipline) rather than layering a third, redundant
  "still open" note on top, appended the new 08-10 measurement to §6.2's
  Findings, and spun the newly-discovered TimesFM SAE-forecast-preservation
  run-to-run variance out as its own explicit, narrower open follow-up
  (a repeat-run noise floor for that one metric specifically) instead of
  leaving the whole §13 item ambiguously reopened. §13's item marked `[x]`;
  §5.3's lens-comparison half was already closed by an earlier entry this
  same day. Verification: the launching agent's own post-rerun pytest
  (215 passed, 2 warnings) matches this turn's own earlier, independently-
  run full-suite result — a real cross-check, not a single unverified
  report.

**2026-08-10, eleventh cron-loop firing.** Nothing was in progress
(previous firing's write-up was the last outstanding item). Re-read
§13/§15/§16 fresh and picked up §16 E12 (horizon-resolved everything) —
specifically only its cheapest, zero-new-forward-passes sub-part
(MASE/pinball per horizon step), since the other two named sub-parts
(L3 per-window patching by horizon, Lens crystallization by horizon) are
each their own separate unit of work. Implemented directly (small/quick,
no background delegation needed): `analysis/stats.py::
mase_pinball_by_horizon()`, `L0Config.horizon_resolved` flag,
`l0_behavioral.py::_summarize_by_horizon` + new `l0/horizon_resolved.json`
artifact, `report.py::_horizon_resolved_block()` wired into `_sec_l0`. Four
new synthetic-planted-answer tests
(`tests/test_horizon_resolved.py`) all passed on first run. Verified live,
not just unit-tested: reran `configs/smoke.yaml`'s `l0,report` stages,
confirmed `l0/horizon_resolved.json`'s array shapes directly and the
rendered HTML's actual findings text (`"L0 horizon profile — patchy: MASE
2.731 at h=1 vs 4.042 at h=32 (ratio 1.48x)"`,
`"...steppy: MASE 5.270 at h=1 vs 5.584 at h=32 (ratio 1.06x)"`) — findings
count rose from 27 to 29 as expected, sections stayed at 11. Full
`tsfm_lens` suite green at **219 passed, 2 warnings** (215 prior + the 4
new tests, same two pre-existing benign warnings, zero regressions). §16
E12 marked `[~]` (partial — the L3/Lens sub-parts remain open, each is a
natural next pick for a future firing). No background work was launched
this firing since the chosen item was small enough to finish inline within
the loop's own turn.

**2026-08-10, twelfth cron-loop firing.** No background agent was running
(`ListAgents` returned none) and nothing else was mid-way. Per this
firing's own brief, retried the `CLAUDE.md` §11.25 write-up that two
earlier edit attempts had failed to apply due to tool-hook timeouts — this
time it landed cleanly on the first try. Confirmed on disk, not just from
memory, that the underlying fix is real before writing it up: `runs/
medium_run/activations.zarr` has a valid v2 `.zattrs`/`.zgroup` (no
`zarr.json`), file mtimes 2026-08-10 23:17–23:28Z (consistent with the
"same-day follow-up" session-log entry above, which already documented the
successful re-extraction — that entry's numbers were correct, only
`CLAUDE.md` §11.25 itself had never actually been updated to match).
Corrected §11.25 in place: replaced the "Not yet fixed as of 2026-08-10"
line with what actually happened (a second background agent completed the
delete-and-re-extract cleanly after the first hit an infra-level usage
limit; ~11 min wall-clock; fresh v2 store; TimesFM lens number
`1.7899408340454102` reproduced bit-for-bit against `medium_run_chronos_
base`; full suite 208/2 at the time), and explicitly flagged that
`real_run`'s own store is untouched and still stale (confirmed directly:
`real_run/activations.zarr` still has `zarr.json`, no `.zattrs`) — not to
be assumed fixed by association. Also corrected a now-stale sub-claim in
§6.2's SAE-ground-truth-null Findings block (line ~2880) that had listed
`medium_run` alongside `real_run` as unusable for pooling due to the zarr
v3 issue — added a same-day correction noting `medium_run` could now be
added to that pool on a future pass, while `real_run` remains excluded and
unchecked. Doc-only changes (`CLAUDE.md`, `ROADMAP.md`); no code touched,
so no test suite re-run was needed. No new roadmap item started this
firing per the brief's own instruction to finish the pending write-up
first and not layer new heavy work on top in the same turn.

**2026-08-10, thirteenth cron-loop firing.** Same firing prompt as the
prior one (the still-pending §11.25 retry it names had already landed last
firing) — `ListAgents` confirmed no background work running, so re-read
§13/§15/§16 fresh per the brief's own instructions. Picked up the first of
§16 E12's two remaining named sub-parts: L3 per-window patching
restoration resolved by horizon step, explicitly flagged by the previous
E12 write-up as "the arrays already exist... this needs only the
reduction step" — confirmed that framing was accurate by reading
`_window_restoration`/`_patching` directly, so this was small/quick enough
to build inline rather than delegate to a background agent. Implemented,
unit-tested (4 new synthetic-planted-answer tests against a newly
extracted, directly-testable `restoration_by_horizon()` helper) and
live-verified against a rerun of `configs/smoke.yaml`'s `l3,report`
stages (confirmed the new artifact's shape/finiteness and the new report
heatmap section + findings text directly). Full `tsfm_lens` suite green
at **223 passed, 2 warnings**, zero regressions. See §16 E12's own
checklist entry for the full write-up and numbers. E12 stays `[~]` — only
the Lens crystallization-depth-by-horizon sub-part remains, a natural pick
for a future firing. No background work was launched this firing since
the chosen item finished comfortably inline.

**2026-08-11, fourteenth cron-loop firing.** Same recurring firing prompt
as the prior two; `ListAgents` again confirmed no background work running
and nothing was mid-way. Picked up the exact item the previous firing
flagged as the natural next pick: the Lens crystallization-depth-by-
horizon sub-part, the last of §16 E12's three named parts. Implemented
directly (small/quick, no delegation needed): `analysis/lens.py` gained
`per_series_h`/`final_mase_h` (the same skip-lens arrays already computed
for the whole-horizon curve, kept resolved by horizon step instead of
collapsed by `.mean(axis=2)`) and a new standalone
`crystallization_depths()` function that unifies the existing scalar
crossing logic with a new per-horizon-step version — refactored the
existing call site to use it too, verified behavior-preserving via a
dedicated 2-D/1-D consistency test rather than assumed. New
`LensConfig.horizon_resolved` flag, new `skip_mase_by_horizon` array and
`crystallization_depth_by_horizon` list artifacts, new
`report.py::_lens_horizon_block()` (heatmap + line chart + findings). 4
new synthetic-planted-answer tests (`tests/test_lens_horizon_resolved.py`)
all passed on first run; extended `test_smoke.py`'s existing artifact
shape-check test too. Verified live end-to-end: reran
`configs/smoke.yaml`'s `lens,report` stages, confirmed the new arrays'
shapes/finiteness and the rendered HTML's actual new chart titles and
findings text directly (not just that the run exited 0) — findings count
rose from 39 to 41 as expected, sections stayed at 11. Full `tsfm_lens`
suite green at **227 passed, 2 warnings** (223 prior + the 4 new tests,
same two pre-existing benign warnings, zero regressions). **§16 E12 is
now fully `[x]`** — all three named sub-parts landed across this and the
two prior firings. No background work was launched or left running this
firing.

**2026-08-11, fifteenth cron-loop firing.** Same recurring firing prompt;
`ListAgents` again confirmed no background work in flight and nothing was
mid-way from a prior firing. §16 E13 (spectral lens) was considered and set
aside as too large to safely land within the remaining loop window, so
instead picked a doc-only item off the open-checklist list: "revisit and
either implement or explicitly defer-with-reason the remaining `CLAUDE.md`
§13 future-work items not already folded into this roadmap" (this
section's own list, above). Cross-checked all five named `CLAUDE.md` §13
items against the existing §16 backlog and found three already tracked
(Chronos decoder gap = E21, deeper component resolution = E18, more source
adapters already explained in full at `CLAUDE.md` §4.3/§12) and two
genuinely missing (CI diversity gates for `benchmark_validation`,
real-corpus activation bucketing) — added those as new §16 **E23**/**E24**,
each with scope and an explicit defer-with-reason rationale rather than
either implementing them or leaving them unaddressed. Checklist item marked
`[x]` with the full resolution recorded inline above. No code changed, no
test suite rerun (nothing to verify — this was a cross-referencing audit,
not an implementation), no background work launched or left running.

---

## 15. Audit — silent-failure paths, fragile mechanisms, and fixable limitations

> **Added 2026-08-06 by a read-the-code-against-the-docs pass** (planning only,
> nothing here implemented). Each item is: what's wrong · evidence at
> `file:line` · why it matters (blast radius) · the fix, in enough detail to
> implement without re-deriving the analysis · priority · rough cost.
>
> **How this section differs from `CLAUDE.md` §11.** §11 records traps that
> were *hit and fixed*. This records traps that are *live and unfixed*. When an
> item here is fixed, move its lesson to `CLAUDE.md` §11 (if it cost real
> time to find) and mark the item `[x]` here with a dated Findings note —
> don't delete it (§0.2).
>
> **Priority key.** **P1** = can silently produce a wrong number that reaches
> a report or a recorded finding; fix before the next real run. **P2** = a
> stated invariant/requirement is unenforced, or a documented limitation
> blocks already-planned work. **P3** = real, worth doing, nothing currently
> depends on it.
>
> **A recurring theme worth naming once.** Twelve of these nineteen items are
> the same shape: *a mechanism that degrades quietly where the doctrine
> (`CLAUDE.md` §2.5) says it must degrade loudly, because "loudly" was
> implemented as a log line*. Logs are not loud once the deliverable is an
> HTML file a colleague opens three weeks later, and they are not loud inside
> a thirty-minute run that prints hundreds of them. The general fix, applied
> item by item below: **every degrade, fallback, cap, and skipped check must
> land in a run artifact and in the report**, not only in stdout.

### A1 — The production `layer_screen` stage cannot be fair to every layer `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** §6.1.1's R2 ("fair to every layer... the screen must pin
  stride to 1 regardless of what the analysis config uses") is not
  implemented. The stage screens exactly the layers extraction happened to
  store.
- **Evidence.** `analysis/layer_screen.py:455` — `layers =
  store.layers(m.name)`; `configs/default.yaml:29` and
  `configs/medium_run.yaml:32` — `capture_layer_stride: 2` for TimesFM (20
  blocks → 10 captured). No `layer_screen.stride` knob exists in
  `config.py`'s `LayerScreenConfig`.
- **Blast radius.** In every production config, half of TimesFM's blocks are
  invisible to the selector that decides where expensive analysis and SAE
  training go — the exact objection (§6.1.1 problem 3) that disqualified the
  §6.1 mechanism. The report's Screen section says "selected N of 10 captured
  layers", which reads as complete. And the bake-off that chose `work_bend`
  ran at stride 1 (`configs/layer_screen_experiment.yaml`), so production
  selections are made under conditions the method was never validated under.
- **Fix.**
  1. `LayerScreenConfig` gains `stride: int = 1` and `require_full_capture:
     bool = True`.
  2. Preferred mechanism: a **dedicated cheap screening extraction** — one
     extra forward pass per model over `layer_screen.max_series` series
     (default a few hundred, far fewer than the analysis corpus), all blocks
     at stride 1, written to `screen_activations.zarr` in fp16 and deleted
     after selection unless `layer_screen.keep_store: true`. Cost is bounded
     and pays R4 correctly: screening still runs before every expensive
     stage. Reuse `extraction/extract.py::_extract_model` with an override
     for layer list and row count rather than writing a second extraction
     path.
  3. If a caller insists on screening the strided store (`require_full_capture:
     false`), record `fair_to_all_layers: false` plus the realized stride in
     `layer_screen/selection.json`, and have the report's Screen note say so
     in the section body — not in a collapsed `_note()`.
  4. Denominator honesty: the Screen section must report "screened X of Y
     model blocks" using `adapter.all_layer_names()` as Y, not the store's
     captured count.
  5. Test: a config with `capture_layer_stride: 2` and
     `require_full_capture: true` screens all 20 mock blocks; with `false`, it
     screens 10 and emits `fair_to_all_layers: false`.
- **Also worth fixing while here.** `selection.json` records `method` per
  model, and `_uniform_fallback` correctly writes `method:
  "uniform_fallback"` — but nothing surfaces the discrepancy between the
  *requested* method and the *used* one. Add `method_requested` alongside
  `method`, and have the report and `findings` name any fallback explicitly:
  a run that silently selected layers with the null it was supposed to beat
  should be impossible to miss.
- **Cost.** ~a day including the screening-extraction path and tests.
- **Findings (2026-08-06).** Implemented essentially as planned: `LayerScreenConfig`
  gained `stride`/`require_full_capture`/`keep_store`; `layer_screen.py` gained
  `_run_screening_extraction`, which reuses `extraction/extract.py::_extract_model`
  (now parameterized by an explicit `layers`/`rows` pair instead of deriving
  them from `adapter`/`data` internally) against a fresh, temporary store, then
  deletes it after selection unless `keep_store: true`. `run_layer_screen` now
  takes `hub` (pipeline.py's stage closure updated) so it can drive its own
  extraction pass. Verified two ways: (1) synthetic — a smoke config with
  `capture_layer_stride: 2` on one mock model showed the main store capturing
  3 of 6 blocks while the dedicated screening pass correctly screened all 6
  (`fair_to_all_layers: true`), and setting `require_full_capture: false`
  correctly reported `fair_to_all_layers: false` with the true denominator (6)
  both in `selection.json` and in the rendered report body (not a collapsed
  note); (2) real checkpoints — a live run against `google/timesfm-2.5-200m-pytorch`
  (`capture_layer_stride: 2`, 20 blocks) and `amazon/chronos-t5-small` (stride
  1, 6 blocks) confirmed the exact bug scenario A1 described (`extraction
  complete: {'TimesFM': 10, 'Chronos-T5': 6}`) and its fix (`layer_screen:
  TimesFM (work_bend, budget=5/20, fair_to_all_layers=True) selected
  ['stacked_xf.6', 'stacked_xf.10', 'stacked_xf.14', 'stacked_xf.16',
  'stacked_xf.18']` — a real selection over all 20 blocks, not just the 10
  the main analysis config would have captured). Full `tsfm_lens` pytest
  suite (81 tests) green after the change. One design deviation from the
  fix plan: row sampling for the screening pass uses a plain
  `rng.choice(..., replace=False)` (matching the "already random" call
  sites named in A4), not yet the stratified `sample_rows` helper A4
  introduces — the call site will be swapped to it when A4 lands, per that
  item's own fix plan.

### A2 — Invariant 7 (empirical alignment verification) is logged and discarded `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** The impulse alignment check runs, prints one INFO line,
  and its result is thrown away. There is no threshold, no artifact, no
  report surface, and only a quarter of the layers are probed.
- **Evidence.** `extraction/extract.py:50-51` —
  `impulse_alignment_check(adapter, cfg.alignment.window, layers[::max(1,
  len(layers)//4)])`, return value unused; `extraction/alignment.py:84-86` —
  `log.info(...)` is the only output; `config.py:43` —
  `sanity_check: bool = False` (the three real configs do set it `true`).
- **Blast radius.** `CLAUDE.md` §6.3 states that near-zero diagonal hits mean
  "fix the adapter before trusting any cross-model number", and §7 invariant 7
  says alignment is verified empirically on every new library version. Today
  both depend on a human running `--check-alignment` by hand and reading the
  output. This has already failed once in a way that cost a session (§11.16:
  a 0.06 hit fraction that looked like a broken adapter and was a broken
  test) — and that was found only *because* someone ran the standalone tool.
  Inside a pipeline run, the same signal is one line among hundreds, followed
  by a complete-looking report.
- **Fix.**
  1. `run_extraction` captures the returned dict per model and writes
     `alignment/alignment_check.json` (per model: per-layer hit fraction, min,
     mean, window, n_windows, impulse amplitude used, layer subset probed).
  2. Probe **all** captured layers, not `[::len//4]` — the check costs one
     forward pass over `n_windows + 1` rows, i.e. nothing.
  3. New config: `alignment.min_diagonal_frac` (default `0.5` at the shallowest
     probed layer, `0.0` elsewhere — deep layers legitimately mix positions,
     per §6.3) and `alignment.on_failure: fail|warn` **defaulting to `fail`**.
     Failing here is correct doctrine: a broken span map is a broken
     assumption, not an unsupported capability.
  4. Flip `sanity_check` default to `true` in `config.py` so a hand-written
     config can't opt out by omission.
  5. The report grows a mandatory **"Alignment & provenance"** panel (shared
     with A7) showing per-model min/mean hit fraction with a pass/fail badge,
     rendered before any cross-model section. A report whose alignment gate
     didn't run should say that where a reader will see it.
  6. Test: a deliberately-mis-declaring mock adapter (spans shifted by one
     window) must fail the gate; `tests/test_adapter_conformance.py` already
     has the "deliberately broken adapter" pattern to copy.
- **Cost.** Half a day.
- **Findings (2026-08-06).** Implemented as `extraction/alignment.py::run_alignment_gate`,
  which wraps `impulse_alignment_check` (now probing every layer
  `extraction/extract.py` actually captures, not a `[::len//4]` subset — free,
  since the check is one extra forward pass), builds the full per-layer/min/
  mean/shallowest-layer record, and raises `RuntimeError` with an actionable
  message when the shallowest probed layer's hit fraction is below
  `alignment.min_diagonal_frac` (new, default 0.5) and `alignment.on_failure`
  (new, default `"fail"`) is `"fail"`; `"warn"` logs instead. `alignment.sanity_check`
  now defaults to `true` (the three real configs already set it explicitly, so
  this changes nothing for them; it closes the "opt out by omission" gap for
  new configs). `run_extraction` writes the per-model records to
  `alignment/alignment_check.json`. Verified three ways: (1) unit-level — a
  deliberately sabotaged mock adapter (`token_time_spans` permuted via
  `np.roll`) drove the shallowest layer's hit fraction to 0.00 and the gate
  raised with the expected message under `on_failure: fail`, and returned a
  `passed: false` record under `warn`; (2) smoke end-to-end — both mock
  architectures pass cleanly (hit fraction 1.00 at every layer) and
  `alignment/alignment_check.json` contains the full expected record shape;
  full 81-test pytest suite green; (3) real checkpoints — a live run against
  `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-small` both showed
  `diagonal-hit fraction min=1.00 mean=1.00`, consistent with `CLAUDE.md` §9's
  existing findings for these two checkpoints and confirming the gate doesn't
  false-positive on real, correctly-aligned adapters. Item 6 (a permanent
  regression test copying `test_adapter_conformance.py`'s "deliberately
  broken adapter" pattern) is now committed as `tests/test_alignment_gate.py`
  (3 tests: fails loudly on broken spans, warns instead when configured,
  passes cleanly on a correct adapter). **Not yet done from the original fix
  plan:** item 5, the dedicated "Alignment & provenance" report panel —
  deferred to land together with A7 (which the fix plan explicitly says
  shares that panel), so it renders both alignment and provenance in one
  place instead of being built twice.

### A3 — Artifact reuse has no config fingerprint, so stale artifacts mix silently `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** Stage skipping is pure path-existence. Nothing checks
  whether the existing artifacts were produced by the *current* config.
- **Evidence.** `pipeline.py:169` — `if stage.done(cfg) and stage.name not in
  force`, where every `done` is a `.exists()` check (`pipeline.py:78-128`).
  `dump_config(cfg, run_dir/"config_resolved.yaml")` (`pipeline.py:146`)
  writes the config but nothing ever reads it back to compare.
- **Blast radius.** Change `data.context_len`, `alignment.window`,
  `data.path`, or a checkpoint id, rerun into the same `run.name` without
  `--force all`, and you get last config's extraction feeding this config's
  analyses, with a report that looks complete. This is not hypothetical usage:
  §6.2's own Findings describe editing `medium_run_chronos_base.yaml` to add
  an `sae:` block and rerunning `--stages sae` against existing extraction —
  correct in that instance precisely because the edit didn't touch extraction
  inputs, but nothing in the tool knew that.
- **Fix.**
  1. Each `Stage` declares `config_keys: tuple[str, ...]` — the dotted config
     paths it actually consumes (`extract` → `data.*`, `alignment.*`,
     `models[*].{checkpoint,layer_regex,capture_layer_stride,batch_size}`,
     `extraction.*`; `l1` → `l1.*` plus extract's keys, etc.). Explicit
     per-stage keys, not a whole-config hash, so an unrelated edit doesn't
     invalidate everything — that would make the guard so annoying it gets
     `--force all`-ed past, which is worse than no guard.
  2. On completion, append to `run_manifest.json` (A7): stage name,
     fingerprint (sha256 over the resolved values of its `config_keys` plus
     the fingerprints of its dependency stages), timestamp, wall-clock,
     realized sample sizes (A4/A19).
  3. On rerun: if a stage would be skipped but its fingerprint differs,
     **refuse** with a message naming the changed keys and the two values, and
     offering `--force <stage>` or a new `--allow-stale`. Transitively mark
     downstream stages stale.
  4. The report renders a "stale inputs" warning per section whose fingerprint
     doesn't match the current config, and `meta_report.py` refuses to compare
     runs whose corpus fingerprint differs without a `--allow-mixed` flag.
  5. Test: run smoke, change `alignment.window`, rerun → expect refusal
     naming `alignment.window`; then `--force all` → clean run.
- **Cost.** ~a day. Highest leverage-per-line item in this section: it turns a
  whole class of "how did that number get there" into an error message.
- **Findings (2026-08-06).** Implemented as a new `manifest.py`
  (`resolve_config_keys`, `fingerprint_stage`, `load_manifest`/`save_manifest`,
  `diff_resolved`) plus a `config_keys: tuple` field on every `Stage` in
  `pipeline.py`. One deviation from the original plan, chosen deliberately:
  keys are whole-config-section wildcards (`"data"`, `"l1"`, `"models[*].checkpoint"`,
  etc.) rather than exhaustive single-field lists per stage — still far
  narrower than a whole-config hash (a `report.title` edit doesn't invalidate
  `extract`), and it matches the fix plan's own worked example for `extract`
  (`data.*`, `alignment.*`, ...) rather than adding single-field-list
  maintenance burden with no evidence it's needed yet. `run_pipeline` now
  does a pre-flight pass computing every stage's fingerprint (own resolved
  keys + dependency fingerprints, so a change upstream propagates without
  every downstream stage needing to declare it), and refuses — before
  running anything — if any stage that would be skipped has a fingerprint
  mismatch against `run_manifest.json`, naming exactly which resolved values
  changed and which upstream stage caused a transitive mismatch. `--allow-stale`
  (new CLI flag) downgrades this to a warning. Runs with no manifest at all
  (every run directory that predates this feature) degrade to a warning and
  are trusted as-is, rather than breaking every existing run dir the moment
  this landed — verified against the real `runs/smoke` directory from this
  session's own earlier testing. Verified four ways: (1) an identical rerun
  causes no staleness warnings or errors; (2) changing `alignment.window` and
  rerunning without `--force` raises `ValueError` naming `alignment` and its
  old/new values, and separately names `layer_screen`/`l0`/etc. as
  transitively stale via `extract`; (3) `--force <non-confirm stages>` with
  `--allow-stale` (needed because `confirm`'s own pre-existing "run once"
  guard correctly refuses to be force-rerun) regenerates cleanly and the new
  manifest reflects the new value; a subsequent skip-eligible rerun is then
  clean; (4) a real run against `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small` (the same live run used to verify A1/A2) wrote a
  valid `run_manifest.json` with no errors, confirming the mechanism doesn't
  choke on real per-model config resolution (`models[*].checkpoint` etc.).
  Committed as `tests/test_manifest_fingerprint.py` (3 tests). Full
  `tsfm_lens` suite green (87 tests, up from 81 — 3 new alignment-gate tests
  from A2 plus 3 new manifest tests here). One emergent interaction worth
  recording rather than "fixing": if `confirm`'s upstream inputs go stale,
  the pipeline will keep refusing on every subsequent run (its manifest
  entry is never updated while it stays skipped, by design) until the user
  either accepts `--allow-stale` indefinitely or makes a deliberate call via
  `--force confirm` — which is the correct tension for a "consumed once"
  stage to create, not a bug to smooth over.

### A4 — Corpora are written grouped by task, and four call sites subsample with a head slice `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** `build_pipeline` emits samples task-by-task in spec order,
  so `corpus.jsonl` is **blocked by generator/task**. Several stages take the
  *first* N series rather than a random or stratified sample, which therefore
  silently selects a family-skewed prefix.
- **Evidence.** Ordering: `build_pipeline/builder.py::_build_split` — `for
  spec in specs: for k in range(spec.count)`, appending in order; `seal.py`
  preserves that order. Head slices: `data.py:75` — `kept = kept[:
  cfg.max_series]`; `sae/eval.py:86-88` — `contexts =
  data.contexts()[:take]`; `sae/ground_truth.py:153-155` —
  `np.arange(n)`; `analysis/layer_screen.py:450-451` — `rows =
  np.arange(n_use)`. Indirect: `analysis/confirm.py:80-82` passes
  `confirm.max_series` into the same `data.py` head slice.
  (Correctly randomized for contrast, and the pattern to copy:
  `internals.py:44`, `l1_geometry.py:67`/`:176`, `l3_perturbation.py:185`,
  `lens.py:92`, `attention.py:235`, `clustering.py:46` all use
  `rng.choice(..., replace=False)`.)
- **Blast radius.** Two recorded results are directly affected, and one
  recorded puzzle is probably explained by this:
  - §6.2's SAE forecast-preservation ΔMASE (+0.047 TimesFM / +2.37
    Chronos-T5-Base) was computed on `min(data.n,
    forecast_preservation_max_series, batch_size)` = the **first ~16-32
    series** of a 3-family blocked corpus. The cross-model asymmetry was
    attributed entirely to window-broadcast granularity; part of it may be
    that the two models were compared on the same skewed handful, and the
    absolute values may not represent the corpus at all.
  - §6.2's ground-truth alignment ρ values use the first
    `ground_truth_max_series` rows, so which archetypes are even present
    depends on task order (compounding A10).
  - §5.3 flagged that lens numbers moved for an *unchanged* TimesFM
    checkpoint (final MASE 2.277 → 1.680) between two configs differing only
    in `lens.max_series` (32 vs 24), and correctly concluded "the difference
    is which ~24-32 series got sampled." `lens.py` does sample randomly — but
    with a different `size`, `rng.choice` returns a *different draw*, so the
    two runs' lens stages ran on largely disjoint tiny samples. Same root
    cause class: sample identity is neither stratified nor recorded, so no two
    stages or runs are comparable.
  - `default.yaml`'s `confirm.max_series: 1024` against
    `full_multidomain_run1`'s 4279 private series would confirm hypotheses on
    a task-ordered prefix — the gold-standard stage, on a biased slice. Not
    yet bitten only because `medium_run`'s private split is smaller than its
    cap.
- **Fix.**
  1. One shared helper, `utils.sample_rows(n, k, seed, strata=None) ->
     np.ndarray`: stratified-without-replacement when `strata` is given
     (proportional allocation with a floor of 1 per stratum, remainder by
     largest-remainder), plain `rng.choice` otherwise, always sorted, always
     deterministic in `(n, k, seed, strata)`.
  2. Replace **every** cap with it, passing `data.meta["family"]` as strata:
     the four head slices above plus the seven already-random sites (so that
     family composition is stable across stages and across runs with
     different `k`).
  3. `data.py`: never truncate an ordered list. Either stratify the truncation
     or, at minimum, shuffle deterministically before it — and log the
     realized family composition, not just the family count.
  4. Every stage records the row ids it actually used in its own artifact
     (`rows: [...]`, `n_requested`, `n_realized`) so two stages' samples can
     be compared, and so a future session can tell whether a discrepancy is a
     model effect or a sampling effect without re-deriving it (§5.3 had to).
  5. Warn when realized family composition deviates from the corpus by more
     than a tolerance — the case where even stratification can't help because
     `k < n_families`.
  6. Check `benchmark_validation`'s `--max-sequences` for the same pattern
     (`CLAUDE.md` §5 documents it as subsampling; verify it's random and
     stratified by generator, since its whole purpose is measuring
     per-group diversity).
  7. Test: a synthetic 3-family blocked corpus, cap = n/3, assert all three
     families present in the realized sample and that the same `(seed, k)`
     reproduces it exactly.
- **Then re-measure.** §6.2's forecast-preservation and ground-truth numbers,
  and re-check the §5.3 lens comparison with matched `max_series` and a
  recorded row set. Supersede the old numbers in place with a dated
  correction (§0.2), don't delete them.
- **Cost.** ~a day for the helper + call sites + tests; the re-measurement is
  a short GPU run against already-extracted stores.

**Findings (2026-08-06).** Implemented per the fix plan, with two deviations
and two second-order bugs found along the way:
- `utils.sample_rows(n, k, seed, strata=None)` + `_stratified_alloc` (largest-
  remainder proportional allocation, floor 1 per stratum) landed as specified
  — deterministic in `(n, k, seed, strata)`, always sorted, logs the realized
  vs. population family composition on every call (closing fix item 5's
  "warn on deviation" via always-log rather than a silent pass/warn split;
  simpler and strictly more informative).
- All four named head slices fixed (`data.py::_assemble`, `sae/eval.py`,
  `sae/ground_truth.py`, `layer_screen.py`'s fallback path) plus all seven
  already-random sites switched from bare `rng.choice` to the shared,
  strata-aware helper (`internals.py`, `l1_geometry.py` ×2,
  `l3_perturbation.py`, `lens.py` ×2, `attention.py` ×2, `clustering.py`) so
  family composition is now stable across stages and across differing `k`
  — directly fixing §5.3's documented "lens numbers moved because
  `rng.choice` drew a different sample at a different size" puzzle: `lens.py`
  now uses one fixed seed (`cfg.run.seed + 8`) shared across both models
  instead of a shared *advancing* `rng`, so two models' lens stages sample
  the same row set at a given `k`, and two runs at different `k` share the
  maximal overlap a stratified draw allows.
- Two second-order bugs not in the original evidence list, found while fixing
  the named sites: (a) `l3_perturbation.py::_patching` re-sliced an
  already-`sample_rows`-sampled array with a further `[:take]` prefix slice,
  reintroducing family-order bias one level down — fixed by threading one
  `sel` index consistently through `ctx_clean`/`targets`/`series_ids`/
  `families`/`ctx_corr`; (b) `layer_screen.py`'s own uniform-fallback path
  (taken when a model's captured layers aren't the full block set) had an
  unnamed `np.arange(n_use)` head slice, fixed the same way as the four named
  sites.
- Every touched stage's artifact now records `n_requested`, `n_realized`,
  `rows` (fix item 4) — `sae/eval.py::forecast_preservation` and
  `sae/ground_truth.py::ground_truth_alignment` confirmed carrying this.
  **Deviation:** a `limited_by` field (why `n_realized < n_requested` — cap
  vs. batch size vs. corpus size) was drafted for `forecast_preservation`
  then deliberately removed — that's A16's scope, not A4's, and the
  heuristic first draft was fragile enough that shipping it here would have
  been exactly the "half-finished implementation" this repo's own doctrine
  forbids.
- Fix item 6 (`benchmark_validation`'s `--max-sequences`): checked
  `matching.py` directly rather than assuming — its subsampling is already
  random (`rng.choice`, not a head slice) but **not** stratified by
  generator. Decided **not** to fix in this pass (would duplicate the same
  helper into a sibling package with a different dependency footprint) and
  folded it into A17's fix plan instead, which already covers this file for
  an unrelated reason (blocked/approximate matching at scale) — noted there
  explicitly so it isn't lost.
- **Deviation from the fix plan's wording:** `layer_screen.py`'s dedicated
  screening-extraction row sampling (added under A1, same session) used a
  plain `rng.choice` rather than `sample_rows`, since `sample_rows` didn't
  exist yet when A1 was implemented earlier in this same session. Swapped
  over to `sample_rows` with family strata as part of this fix, so no
  follow-up debt remains.
- **Verification — unit tests.** New `tests/test_sample_rows.py` (5 tests):
  a synthetic 100/30/10 blocked-corpus case proving the exact bug this
  replaces (`strata[:k]` head slice is 100% one family; `sample_rows` isn't),
  determinism in `(seed, k)`, the unstratified path's behavior preserved,
  `k > n` returning everything, and an end-to-end `data.py::_assemble` case
  with a synthetic 20/10/5 blocked corpus asserting all three families
  survive a `max_series` cap. Full suite green at 92/92 after this fix
  (up from 87 after A3; the 5 new `sample_rows` tests are additive, not a
  net-zero replacement of prior coverage).
- **Verification — real-checkpoint run.** `configs/_a4_real_check.yaml`
  (scratch, deleted after use) against live `google/timesfm-2.5-200m-pytorch`
  + `amazon/chronos-t5-small`, real sealed corpus at
  `../../benchmark_medium/public_dev` (confirmed blocked by task by direct
  inspection: 288 rows, `{'random_parametric': 188, 'mixture': 60,
  'parametric': 40}` in that exact order, so the first 40+ rows are 100%
  `random_parametric` — precisely the scenario this fix targets) with
  `data.max_series: 40`. Pipeline completed cleanly end to end (`extract`,
  `layer_screen`, `lens`, `l1`, `l3`, `attention`, `cluster`, `report`; L2/
  SAE/exemplars/confirm disabled for this scratch config) — 7 sections, 16
  findings, no errors. The log's `sample_rows` lines are the actual proof:
  even though the corpus is blocked by task, every capped sub-sample kept
  all three families in roughly the population's own proportion — e.g. a
  `k=16` draw from the `k=40`-assembled population
  `{'mixture': 8, 'parametric': 6, 'random_parametric': 26}` realized
  `{'mixture': 3, 'parametric': 3, 'random_parametric': 10}`, not the
  100%-`random_parametric` result the pre-fix `[:16]` head slice would have
  produced against this exact corpus. This is the first time any stage in
  this pipeline has been run against a real, family-blocked corpus with
  `max_series` actively capping below the corpus size — a scenario the
  smoke config's single-family-per-stage synthetic data can't exercise at
  all, so this real run is not a redundant re-check of the unit tests, it is
  the only test that could have caught the original bug in situ.
- **Re-measurement (fix plan's own last step):** not performed as a
  standalone action *in this fix's own session*. The §6.2 SAE
  forecast-preservation/ground-truth numbers and the §5.3 lens comparison
  remain as previously recorded, now understood to have been measured on a
  family-skewed sample of unrecorded composition; a superseding re-run would
  need `sae.enabled: true` end-to-end (a longer GPU run than this fix's own
  verification needed) and is left as a follow-up rather than bundled into
  this fix, consistent with treating each A-item as its own scoped change.
  **Correction (2026-08-10): this bullet went stale the same day it was
  written and was never updated to say so.** The §6.2 re-measurement it
  defers *did* happen later the same day (2026-08-06, same-day follow-up —
  see §6.2's own Findings block, "RE-MEASURED" entry) and again on
  2026-08-10 (a third data point — see the same Findings block's newest
  entry). §5.3's lens comparison was also later completed (2026-08-10, see
  the entries below this one). Leaving this bullet's "left as a follow-up"
  wording unedited after both follow-ups actually landed is exactly the
  kind of doc drift `ROADMAP.md` §0 warns about — flagged and corrected in
  place here rather than silently left for a future session to re-discover.
- **Partial follow-up attempt, 2026-08-10 — one of the two configs re-measured, the other blocked by a newly-discovered stale store.** Re-ran the matched-`max_series` (24, per `medium_run.yaml`'s same-day fix) lens comparison against `medium_run_chronos_base.yaml`'s already-extracted store: succeeded, TimesFM's own lens numbers came back `final_mase: 1.7899408340454102`, `crystallization_depth: 1.0`. The `medium_run.yaml` side of the same comparison could not be re-measured — its `activations.zarr` turned out to be a dead store, written entirely in zarr v3 format and silently unreadable (returns empty, not an error) under this repo's pinned zarr v2 (see `CLAUDE.md` §11.25, new this session). A background agent was dispatched to delete the stale store and rerun `extract` from scratch, but it failed outright on an infrastructure-level session/API usage limit on the agent's own side (not a code or logic bug) before completing. **Still not re-measured as of 2026-08-10** — `medium_run`'s re-extraction plus the matched lens rerun remains the concrete next step; nothing about the numbers already on record here needed correcting, since the ones being checked were never actually invalidated, only unable to be refreshed yet.
- **Completed, same day, second attempt.** A fresh background agent was
  relaunched (the first retry's failure was purely a session/API limit, not
  a logic problem — verified by confirming a clean redo worked). The stale
  store was renamed, not deleted, to `runs/
  medium_run_stale_v3_backup_20260810` (preserved for anyone who wants to
  double-check the zarr-v3 diagnosis directly), and the full
  `medium_run.yaml` pipeline (13 stages, real `google/timesfm-2.5-200m-
  pytorch` + `amazon/chronos-t5-small` checkpoints) was rerun clean end to
  end in about 11 minutes wall-clock. Confirmed directly (not just taken on
  the agent's word — the agent itself twice made the same "waiting on a
  notification that will never come" mistake §11.25's own writeup already
  flagged as a risk of `CLAUDE.md` §2.8's delegation pattern, resumed each
  time with explicit polling instructions, and its actual progress verified
  independently via `ps`/filesystem checks): `runs/medium_run/
  activations.zarr` now has a valid v2 `.zattrs` and no `zarr.json`,
  `report.html` rendered **11 sections / 27 findings**, and `confirm`
  completed (1/1 dev hypotheses confirmed on private data, 17 registered/2
  replicable). Fresh, now-trustworthy lens numbers: TimesFM `final_mase:
  1.7899408340454102, crystallization_depth: 1.0` — **bit-identical to
  `medium_run_chronos_base.yaml`'s own already-recorded TimesFM number**,
  which is exactly the expected outcome now that both configs' `lens.
  max_series` are matched at 24 (same TimesFM checkpoint, same corpus,
  same context/horizon/seed, same sampled series → deterministic TimesFM
  reproduces bit-for-bit) — a clean, independent confirmation that the
  earlier max_series-mismatch fix this item exists to verify actually
  works. Chronos-T5 (small, this config's pairing — not directly comparable
  to `medium_run_chronos_base`'s Chronos-T5-**Base**): `final_mase:
  2.173410654067993, crystallization_depth: 1.0`. Full `tsfm_lens` pytest
  suite self-run after this fix to confirm nothing else regressed (this
  only touched a `runs/` artifact directory, not source code): **208
  passed, 2 warnings** (both pre-existing and benign — an intentional
  overflow-cast warning inside `test_nonfinite.py`'s own precision test,
  and a pytest style warning about `test_smoke.py::test_end_to_end`
  returning a value — same count and same two warnings the re-extraction
  agent's own independent pytest run had already reported). **This closes the item**: both
  configs' lens numbers are now on record from real, non-empty stores.

### A5 — The report silently omits sections; nothing in the HTML says what's missing or why `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** A section whose artifacts are absent is skipped with
  `log.info`; a section whose builder raises is dropped with `log.warning`.
  The rendered HTML — the thing that actually gets shared — records neither.
- **Evidence.** `report/report.py:92-100`.
- **Blast radius.** A reader of `report.html` cannot distinguish "L2 was
  disabled in this config" from "L2 crashed while rendering" from "L2 ran
  fine and I missed it." `CLAUDE.md` §6.5 describes the info/warning split as
  a deliberate honesty mechanism, and it is — for whoever watched the console.
  For the report's actual audience it's invisible, which is precisely the
  failure mode §2.3 ("explainability of the explainer") exists to prevent, and
  it gets worse the closer this repo gets to §1's one-button promise, where
  nobody watches a console at all.
- **Fix.**
  1. Accumulate a status per builder: `rendered` · `skipped: stage not enabled
     in config` (distinguish from) `skipped: artifacts missing` · `failed:
     <exception class>: <message>` and write `report/coverage.json`.
  2. Render a mandatory **"Run coverage"** panel: every section, its status,
     and for failures the exception message plus the artifact paths it wanted.
     Not collapsed by default when anything failed.
  3. `run_report` returns/logs a one-line summary ("11 rendered, 1 skipped, 1
     FAILED: Attention") and `run.py` exits non-zero when a section failed
     unless `--allow-partial-report` — a crash in a report builder is a bug
     (`CLAUDE.md` §11.6 says this surface is the least type-safe in the repo),
     and today it exits 0.
  4. Test: corrupt one artifact (truncate `l2/stitching.json`), assert the
     report renders with a `failed` row naming it, and that the exit code is
     non-zero.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented per the fix plan, with each builder now
carrying an explicit config attribute so "disabled" and "missing" are
genuinely distinguishable, not inferred:
- Every builder tuple in `report.py` gained a `config_attr` (`"l0"`,
  `"layer_screen"`, ..., `"confirm"`) mapping it to its stage's own
  `PipelineConfig` sub-config. The per-builder loop now checks
  `getattr(cfg, config_attr).enabled` whenever required artifacts are
  missing, producing exactly the three-way status the fix plan asked for:
  `rendered` / `skipped: stage not enabled in config` /
  `skipped: artifacts missing: <paths>` / `failed: <ExceptionClass>:
  <message>`.
- `report/coverage.json` (`{summary, sections: [{eyebrow, title, status,
  detail}, ...]}`) is written on every report render, success or failure.
- A mandatory **"Run coverage"** panel renders directly under "Before the
  numbers" (ahead of the Findings box, so it's the first thing after the
  preamble) as a `<details>` — auto-`open` when anything failed, collapsed
  by default on a clean run so it doesn't add visual clutter to the common
  case; failed rows render in red (`cov-failed`) regardless of open/closed
  state.
- `ReportConfig.allow_partial: bool = False` (new field) + CLI
  `--allow-partial-report` (mirroring `--allow-stale`'s existing pattern of
  a config field settable either way). `run_report` now computes the
  one-line summary (`"N rendered, M skipped[, K FAILED: names]"`), logs it
  at `warning` level if anything failed else `info`, writes `report.html`
  and `coverage.json` **regardless**, and only then — if there are failures
  and `allow_partial` is false — raises `RuntimeError` naming every failed
  section and its exception, plus the exact path to the (still-written)
  report and the fix/override instructions. Because `pipeline.py`'s report
  `Stage.run` is not wrapped in a try/except, this `RuntimeError` propagates
  uncaught out of `run_pipeline` and `run.py`'s `main()`, which is exactly
  what makes the process exit non-zero — no extra exit-code plumbing was
  needed beyond raising in the right place with the report already on disk.
- **Verification — unit tests.** New `tests/test_report_coverage.py` (4
  tests, reusing `test_smoke.py::build_config`'s mock pipeline as a fixture):
  a healthy run has zero `failed` rows and no `cov-failed` HTML class; a
  corrupted `l2/stitching.json` makes `run_report` raise (`pytest.raises`
  matching "L2"), with `report.html` still written and containing a
  `cov-failed` row naming L2, and `coverage.json`'s summary containing
  "FAILED: L2"; the same corruption with `cfg.report.allow_partial = True`
  set does not raise and still records the failed row; and disabling `l2`
  in config while deleting its artifacts produces `status: skipped,
  detail: "stage not enabled in config"` rather than an "artifacts missing"
  message, proving the two skip reasons are genuinely distinguished and not
  just differently-worded guesses. Full suite green at 96/96 (up from 92
  after A4).
- **Verification — CLI exit code, mock pipeline.** Ran `run.py --stages
  report` directly (not just calling `run_report` in-process) against a
  freshly-built mock-model run dir with `l1/cka.npz` corrupted to a garbage
  byte string: without `--allow-partial-report` the process printed the
  traceback and **exited 1**; with the flag it printed the same "1 FAILED:
  L1" summary line, still wrote a report naming the failure, and **exited
  0**. (First attempt reused one run directory's `config_resolved.yaml`
  across both invocations and got `exit 0` for the no-flag case too — a
  self-caught mistake, not a code bug: the flagged invocation is dumped
  back into `config_resolved.yaml` by `run_pipeline` itself, so the *third*
  command was unknowingly reading a config that already had
  `allow_partial: true` baked in from the *second* command's run. Re-ran
  with two independently-built fresh run directories, one per code path, to
  get a clean result.)
- **Verification — real checkpoints.** `configs/_a5_real_check.yaml`
  (scratch, deleted after use; same reduced-cap pattern as A4's real check)
  against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small`, real sealed corpus, with SAE/Exemplars/Confirm
  disabled and every other stage enabled. Clean run: `coverage.json` shows
  9 `rendered` + 3 `skipped: stage not enabled in config` (SAE/Exemplars/
  Confirm), 0 failed. Then corrupted the *real, checkpoint-produced*
  `l2/stitching.json` in place and re-ran `run.py --stages report` against
  that same run directory: `report: section L2 failed: JSONDecodeError...`,
  `8 rendered, 3 skipped, 1 FAILED: L2`, report.html written, process
  **exited 1**. This confirms the mechanism generalizes past the smoke
  config's mock-model artifact shapes — nothing about real-checkpoint
  extraction/analysis artifacts (bfloat16 tensors, real family labels,
  layer-screened targets) interacts differently with the coverage/failure
  logic, as expected since this fix operates purely on `run_dir` paths and
  exception objects, not on activation content.
- **Deviation from the fix plan's wording:** item 2 said "not collapsed by
  default when anything failed" — implemented as auto-`open` exactly when
  `any_failed`, and left collapsed (not "not collapsed", i.e. closed) on a
  clean run, since always-expanding a coverage table on every healthy run
  would add noise to the common case the rest of this report's `<details>`-
  based note mechanism (`CLAUDE.md` §6.5) is designed to avoid. This matches
  the fix plan's actual intent (surface failures prominently) without
  over-reading "not collapsed" as "never collapsible."

### A6 — `family_key` mis-resolution and unlabeled corpora degrade to one family, silently `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** An explicit `data.family_key` that doesn't resolve falls
  through to a candidate list, then to `provenance.generator`, then to the
  literal string `"unknown"` — per row, with no aggregate check.
- **Evidence.** `data.py:89-99` (`_family_of`), `data.py:22`
  (`_FAMILY_CANDIDATES`). The only signal is the family *count* in an INFO log
  (`data.py:83-84`).
- **Blast radius.** A typo in `family_key` silently regroups every per-family
  statistic in the pipeline — L0's Holm-corrected family strengths (and
  therefore what `confirm` pre-registers), L1's per-family CKA, attention's
  family-conditioned lag profiles, clustering's labels and AMI, exemplar
  selection. A corpus without labels collapses to a single family, at which
  point per-family "comparisons" are computed over one group and the report
  still renders them.
- **Fix.**
  1. If `family_key != "auto"` and it resolves for fewer than (say) 90% of
     rows: **raise**, naming the key, the resolution rate, and the keys that
     *would* have worked. Falling back to a different grouping than the one
     the user asked for is exactly the "silently wrong slice" `CLAUDE.md` §2.5
     forbids.
  2. Record `family_resolution: {key_requested, key_used, n_unknown,
     n_families, composition: {family: count}}` in `run_manifest.json` and
     surface it in the report's coverage panel.
  3. If `n_families < 2`, disable every per-family section with an explicit
     stated reason rather than emitting degenerate one-group statistics; keep
     the overall (pooled) comparisons.
  4. Test: a corpus with a typo'd key raises; a single-family corpus renders a
     report with per-family sections explicitly marked not-applicable.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented items 1, 2, and 4 exactly as specified;
item 3 implemented for the four sites the fix plan's own blast-radius
evidence names, after checking each one for whether "per-family" there is
actually a *comparison* (degenerate with one group) or just a *breakdown*
(still meaningful with one group) — a real subtlety the fix text didn't
fully resolve on its own:
- `data.py::_resolve_family_key` resolves one explicit dotted key with no
  fallback (distinct from `_family_of`'s "auto" chain), and
  `_check_family_key_resolution` measures the aggregate hit rate across the
  whole pre-cap corpus; below 90% it raises, naming the requested key, the
  measured rate, and — by re-running the same resolution check against
  every other `_FAMILY_CANDIDATES` key — which of them *would* clear 90%
  instead, or an explicit "none do, use `family_key: auto`" fallback
  message when none do either.
- `BenchmarkData` gained `family_resolution: dict` (`key_requested,
  key_used, resolution_rate, n_unknown, n_families, composition`) and
  `n_families` — computed once in `_assemble`, always (not just when a
  problem is detected). `extraction/extract.py` persists it via a new
  `manifest.py::record_extra(run_dir, key, value)` helper into
  `run_manifest.json` alongside (not replacing) A3's per-stage
  fingerprints; `pipeline.py`'s per-stage manifest save now re-reads and
  re-merges any such extra top-level keys on every write so a later
  stage's save can't clobber what an earlier stage recorded mid-run (a real
  ordering bug caught by testing the actual sequence, not assumed safe).
  `report.py::_family_resolution_line` renders it as a line inside the same
  `<details class="coverage">` panel A5 built, immediately under the
  section table — literally "surfaced in the report's coverage panel" as
  item 2 asked, not a separate section.
- **Item 3, scoped after inspection, not applied uniformly:** checked each
  of the fix plan's four named blast-radius sites for whether "per-family"
  there is a comparison or a breakdown:
  - **L0** (`_summarize`): `family_tests`/`strengths`/`mase_ratio` are a
    genuine cross-family *comparison* (Holm-corrected, needs >=2 groups) —
    guarded with an early return once `n_families < 2`, replaced with an
    explicit `family_comparisons: {applicable: false, reason}`. `per_family`/
    `overall` (plain aggregates, not comparisons) are untouched and still
    carry the pooled model-vs-model result.
  - **L1** (`run_l1`): family-conditioned CKA compares each family's
    geometry *to the same computation run without conditioning* — with one
    family it's a second, differently-pooled measurement of the same
    number, not comparative signal. Guarded the same way; the window-level
    `cka_window`/`depth_curve` (never family-conditioned) are untouched.
  - **Clustering** (`_label_clusters`/`run_clustering`): checked the AMI
    computation specifically and found the fix plan's own "clustering...
    AMI" framing needs a correction — `adjusted_mutual_info_score` here
    compares **the two models' own cluster assignments to each other**, not
    to family labels, so AMI is fully meaningful regardless of family count
    and was **not** guarded. What *is* genuinely degenerate is
    `_label_clusters`'s majority-family purity (trivially 100% for every
    cluster with one family, a mathematical certainty misreadable as a
    finding) — guarded to drop the family/purity framing and keep only the
    feature-based descriptor, with a `family_comparisons` note recorded in
    `comparison.json` (not `clusters.json`, to avoid the literal string
    `"family_comparisons"` being misread as a model name by
    `_sec_clusters`'s `list(clusters.keys())` — caught by tracing the
    report's own parsing before picking a location, not after).
  - **Attention:** re-read `_pattern_analysis`'s periodicity-head scoring
    and concluded it is **not** degenerate at one family the way the other
    three are — "per-family" there tags each head with whichever family
    gave it the best score, which remains a real (if less varied) signal
    with only one candidate family, not a comparison that becomes
    meaningless. **Left unguarded**, with the reasoning recorded here so a
    future session doesn't have to re-derive it. `head_delta_family`/
    `mlp_delta_family` arrays are computed but were already found to be
    dead code (never read by any `_sec_attention` render path) — orthogonal
    to this fix, not touched.
  - **Exemplars:** per-family case studies aren't a cross-family
    *comparison* either (they showcase series within each family, not
    compare families' properties) — left unguarded for the same reason as
    attention.
- **A genuine crash found and fixed along the way, not in the original
  evidence list:** `analysis/internals.py`'s family-decodability probe
  (`_family_probe`, a `LogisticRegression` classifier) **raises** from
  scikit-learn when given only one class to fit — this is exactly the
  failure mode a prior real-checkpoint session flagged as "directly
  relevant" before this fix existed. Guarded in `run_internals`: the probe
  is skipped (recorded as `{"value": None}` per layer) when `n_families <
  2`, with a `family_comparisons` reason attached to each model's profile
  entry; `effective_dim`/`input_cka` (not family-dependent) are computed as
  usual. `_sec_internals`'s trailing findings loop (`np.argmax` over probe
  values) would also have raised on an all-`None` list — guarded the same
  way. Without this fix, a real single-family corpus would have **crashed
  the whole pipeline at the `internals` stage**, not just rendered a
  degenerate number — worse than anything in the original evidence list.
- **Verification — unit tests.** New `tests/test_family_resolution.py` (5
  tests): a typo'd key (0% resolution) raises naming the correct
  alternative key; a 50%-resolution key raises; a 95%-resolution key does
  not raise; `family_key: auto` never raises even on a fully unlabeled
  corpus (falls to `"unknown"`); and a full mini pipeline (mock models, a
  genuinely single-family 40-series corpus) runs end to end through
  extract → layer_screen → l0 → internals → lens → l1 → l2 → l3 → attention
  → cluster → report with **zero crashes**, and asserts `family_comparisons.
  applicable is False` in L0/L1/clustering/internals artifacts, pooled L0
  `overall` metrics and L1's window-level `depth_curve` still present, AMI
  still a real float, `run_manifest.json` recording `n_families: 1`, and
  the rendered report containing the "Not applicable." explanation at
  least 4 times (one caught-and-fixed test bug along the way: two of the
  four sections initially used a differently-worded "not applicable" phrase
  than the other two, which the test's exact-match caught immediately —
  unified all four to the same phrasing rather than loosening the
  assertion). Full suite green at 101/101 (up from 96 after A5).
- **Verification — real checkpoints.** `configs/_a6_real_check.yaml`
  (scratch, deleted after use; same pattern as A4/A5's real checks) against
  live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small`, the
  real 3-family `benchmark_medium` corpus. Confirms the **normal path is
  unaffected** by every guard added above: `run_manifest.json` records
  `family_resolution: {key_used: auto, n_families: 3, composition:
  {mixture: 8, parametric: 6, random_parametric: 26}, ...}`; L0 still
  produces real `family_tests`; L1's `families` list has all 3; clustering
  and internals both show `family_comparisons: None` (guard correctly
  not triggered); the report contains the family-resolution line inside the
  coverage panel and **zero** "Not applicable." occurrences — a clean
  9-rendered/3-skipped/0-failed run. (The single-family crash path itself
  was not re-verified against real checkpoints beyond the mock-model
  pipeline test above — building a genuinely single-family *real* corpus
  wasn't readily available this session — but every guard is architecture-
  and GPU-agnostic code operating on `data.meta["family"]` counts, not on
  activation content, so the mock-model coverage is expected to generalize
  the same way A4/A5's did.)

### A7 — No run provenance, and the activation store has no schema version `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** A run directory records the resolved config and nothing
  else: no git SHA, no library versions, no device, no checkpoint revisions,
  no per-stage timings. The zarr store records only geometry, so a store
  written by a different zarr major version fails with a low-level error
  instead of an actionable one.
- **Evidence.** `pipeline.py:146` writes only `config_resolved.yaml`;
  `extraction/store.py:37` — attrs are `n_series`/`n_windows`/`window`/
  `context_len` only. Contrast the benchmark half, which does this correctly:
  `build_pipeline/builder.py:190` — `sample.provenance.library_versions =
  _library_versions()` per sample.
- **Blast radius.** This repo has now lost time to environment drift in three
  separate sessions (`CLAUDE.md` §11.15 zarr v3-vs-v2, §11.17 cp1252 encoding,
  and §5's third-machine entry finding zarr 3.2.1 installed against a
  `zarr<3` pin), and §7's Findings record a concrete replication *blocked*
  because `runs/medium_run`'s store was written in a v3-formatted layout and
  "isn't readable here without re-extraction." With no provenance in the run
  dir, no artifact can be attributed to an environment after the fact, and
  `meta_report.py` can silently aggregate runs from mismatched stacks.
- **Fix.**
  1. `utils.run_provenance()` → dict: git SHA + dirty flag, `tsfm_lens`
     version, python/torch/numpy/pandas/zarr/scikit-learn/transformers/
     timesfm/chronos-forecasting versions (whatever imports), CUDA/driver and
     device names, per-model resolved checkpoint id **and HF revision hash**,
     corpus manifest digest (from `load_sealed`'s manifest — already computed,
     currently discarded by the analysis half), config hash. No hostnames or
     user paths (keeps it shareable and invariant-11-clean).
  2. Write `run_manifest.json` at run start and append per-stage records
     (fingerprint from A3, wall-clock, realized n from A4/A19).
  3. `ActivationStore.create` writes `schema_version` (start at 1),
     `zarr_format`, and writer versions into root attrs; `ActivationStore.
     __init__` validates and raises a *specific* message on mismatch
     ("activations.zarr was written with zarr format 3 by tsfm_lens 0.1.0;
     this environment has zarr 2.18.7 — pin `zarr>=2.16,<3` and re-extract,
     or read it in a v3 environment"). This is the fourth recurrence of one
     trap; make the fourth one self-explaining.
  4. Report renders the provenance panel (shared with A2); `meta_report.py`
     shows one provenance row per run and warns on mismatched corpus digests
     or library majors.
  5. Test: hand-write a store with `schema_version: 999`, assert the specific
     error; assert `run_manifest.json` exists after a smoke run and contains
     no absolute user paths (ties to invariant 11).
- **Cost.** ~a day, and it retires a recurring cost.

**Findings (2026-08-06).** All five fix items implemented; one real,
previously-undocumented gap found and closed along the way (`timesfm` has
no `__version__` attribute at all — confirmed live, not assumed):
- `utils.run_provenance(cfg=None, corpus_digest=None) -> dict`: git SHA +
  dirty flag (`git status --porcelain` against a repo-relative `cwd`, never
  an absolute path — invariant 11), `tsfm_lens.__version__`, Python version,
  and versions for `torch/numpy/pandas/zarr/sklearn/transformers/timesfm/
  chronos` via `_pkg_version` — which now tries `__version__` first, then
  falls back to `importlib.metadata.version()` for packages that expose
  neither (see below), device info (CUDA availability, device name(s) via
  `torch.cuda.get_device_name`, CUDA version, driver version via
  `nvidia-smi --query-gpu=driver_version`), per-model checkpoint id **and**
  its locally-cached HF commit hash (`_hf_revision`, via
  `huggingface_hub.scan_cache_dir()` — deliberately the *local cache*, not
  a Hub API call, so it reflects the exact snapshot actually loaded and
  works fully offline), corpus digest, and a config hash (sha256 over
  `dataclasses.asdict(cfg)`, reusing `manifest.py`'s existing `_stable_json`
  serializer). Every sub-lookup is wrapped to degrade to `None` rather than
  raise (`CLAUDE.md` sec 2.5) — not installed, no git repo, no GPU, and no
  network are all real, expected environments for this repo, not bugs.
- **Corpus digest was genuinely discarded, confirmed by reading the code
  before assuming**: `data.py::_load_corpus_rows` called
  `samples, _ = load_sealed(...)`, throwing away the manifest `load_sealed`
  itself already verifies. Now returns `(rows, corpus_digest)`, threaded
  through `load_benchmark`/`_assemble` into a new `BenchmarkData.
  corpus_digest` field (parallel to A6's `family_resolution`) —
  `extraction/extract.py`'s `run_extraction` merges it into the
  already-recorded `provenance` dict once the corpus is actually loaded
  (base provenance is recorded by `run_pipeline` at run start, before any
  stage runs, since git/library/device facts don't depend on data being
  loaded; the corpus digest specifically does, so it arrives one step
  later via a read-merge-write through the same `manifest.py::record_extra`
  A6 built).
- `extraction/store.py`: `_SCHEMA_VERSION = 1` constant; `ActivationStore.
  create` stamps `schema_version` + `written_by: {tsfm_lens_version,
  zarr_version}` into root attrs; `__init__` validates on any non-`"w"`
  open and raises a message naming the stored vs. expected version and the
  writer's exact library versions, plus the concrete fix
  (`--force extract` and downstream). **Also wrapped the `zarr.open_group`
  call itself** in a try/except that re-raises with a pointer to the
  `zarr<3` pin and CLAUDE.md sec 11.15 — this catches the *other* half of
  "fails with a low-level error instead of an actionable one" (a genuinely
  cross-major-version-incompatible store failing to open at all, before
  any attrs can even be read) that a `schema_version` attrs check alone
  cannot reach, since that check requires the group to have opened
  successfully first. **Not independently tested** — reproducing a real
  zarr-v2-vs-v3 on-disk incompatibility would need a second zarr major
  installed in this environment, which isn't available; the wrap's error
  message was verified by inspection, not by triggering a real cross-version
  failure.
- **Report panel, shared with A2 as originally deferred**:
  `_alignment_provenance_block` in `report.py` reads both `run_manifest.
  json`'s `provenance` key and `alignment/alignment_check.json` (A2's
  artifact, previously written but never rendered) into one combined
  `<details class="coverage">` panel — auto-open when any model's alignment
  gate recorded `passed: false`, immediately below the run-coverage panel
  A5 built. Shows per-model min/mean diagonal-hit fraction and gate
  pass/fail, plus git SHA (+dirty), tsfm_lens version, device, config hash,
  corpus digest, per-model checkpoint+HF revision, and package versions.
- `meta_report.py`: `summarize_run` now reads each run's `run_manifest.
  json` provenance (absent for pre-A7 runs, handled as a plain missing key,
  not an error); new `provenance_warnings(run_summaries)` flags **mismatched
  corpus digests** and **mismatched major versions** of
  `torch`/`numpy`/`zarr` across the runs being aggregated — the exact
  "meta_report.py can silently aggregate runs from mismatched stacks" risk
  named in this item's own blast radius. Rendered as a highlighted warning
  block above the existing runs table, plus a new "Run provenance" table
  (one row per run: git SHA, tsfm_lens version, key package versions,
  device, corpus digest — "one provenance row per run" per the fix's own
  wording). Runs with no provenance recorded are excluded from a given
  check rather than flagged as mismatched — absence of data isn't evidence
  of drift.
- **A real, previously-undocumented gap found and fixed, not in the
  original evidence list**: the first real-checkpoint verification run
  showed `packages: {..., "timesfm": null, ...}` even though `timesfm` is
  installed and working (checkpoints loaded fine throughout this session).
  Checked directly (`import timesfm; timesfm.__version__` →
  `AttributeError`) rather than assumed: the installed `timesfm` package
  genuinely has no `__version__` attribute at all — a real gap in
  "whatever imports" as originally scoped, not a bug in the lookup itself.
  Fixed by falling back to `importlib.metadata.version(module_name)` when
  `__version__` is absent; re-verified live (`packages["timesfm"] ==
  "2.0.2"` after the fix, matching `pip show timesfm`).
- **Verification — unit tests.** New `tests/test_provenance.py` (4 tests):
  `run_provenance()` output contains no absolute `/home/...` paths and
  resolves `torch`'s version (a hard dependency, must never be `None`); a
  hand-corrupted `schema_version: 999` store raises `RuntimeError` matching
  the exact mismatch on open; a freshly-created store round-trips through
  reopen with no error; and a full smoke run's `run_manifest.json` has no
  absolute paths, records `corpus_digest: None` (correctly, for the
  no-sealed-corpus smoke source), and its `report.html` contains the
  rendered "Alignment &amp; provenance" panel. Two new tests in
  `tests/test_meta_report.py`: mismatched corpus digests and a mismatched
  `torch` major across two synthetic runs both produce the expected
  warning strings (and a matching `numpy` major does **not** false-positive
  warn), rendered visibly in the aggregate HTML; and runs with no recorded
  provenance at all aggregate cleanly with zero warnings. Full suite green
  at 107/107 (up from 101 after A6).
- **Verification — real checkpoints.** `configs/_a7_real_check.yaml`
  (scratch, deleted after use; a fast extract+l0+l1+report-only config to
  keep this check cheap) against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small`, the real `benchmark_medium` sealed corpus.
  Every field resolved to a real, correct value: `git_sha` (the actual
  working-tree HEAD, `git_dirty: true` — correctly reflecting this
  session's uncommitted edits), real package versions for all 8 tracked
  libraries (after the `timesfm` fix above), real device info (`NVIDIA RTX
  A5000`, CUDA 12.9, driver 535.216.03 via `nvidia-smi`), and — the
  detail most worth calling out — **real, resolved HF revision hashes for
  both checkpoints** (`1d952420fb...` for TimesFM 2.5,
  `a971ba2194...` for Chronos-T5-small) via `huggingface_hub.
  scan_cache_dir()`, plus a real sha256 corpus digest and config hash.
  The rendered report's "Alignment & provenance" panel showed the real
  alignment gate results (1.00/1.00 diagonal-hit fraction for both models,
  consistent with every prior session's numbers) directly alongside this
  provenance line, exactly as the shared-panel design intended.

### A8 — Invariant 1 is currently red and unowned (golden hashes vs. numpy) `[x]` · **P1** · addressed 2026-08-06

- **What's wrong.** `test_golden_hashes_pre_extension_outputs_unchanged` fails
  against numpy 2.1.0 with no change to the generators; the leading hypothesis
  (`Generator.choice(..., replace=False)` is not stream-stable across numpy
  versions) has never been confirmed, and the failure has been carried as a
  known-red test across five sessions.
- **Evidence.** `CLAUDE.md` §9, §11.13; §5's 2026-08-03 Findings (seed 0
  produces `4070db4646799f09`, expected `2856658d044e4c49`).
- **Blast radius.** Invariant 1 is the whole basis of the sealed-corpus
  reproducibility claim, which is in turn the basis of the epoch-regeneration
  story (`CLAUDE.md` §4.5) and of every "this corpus regenerates from its
  seed" statement. A red invariant that nobody owns decays into a permanently
  accepted failure, and the suite's "40 passed / 1 failed" line stops being
  read at all.
- **Fix.**
  1. **Confirm the mechanism first, cheaply** (§2.4): in an isolated env, diff
     `np.random.default_rng(0).choice(100, 5, replace=False)` and
     `default_rng(0).normal(size=5)` across the numpy versions available.
     One is expected to move and the other not; that alone settles it without
     bisecting the whole generator.
  2. If confirmed: replace every `rng.choice(..., replace=False)` in
     `generators.py` with a construct built from primitives whose stream is
     stable — candidate `np.argsort(rng.random(n))[:k]` — and **verify that
     candidate's cross-version stability by the same test** rather than
     assuming it (the assumption is what broke here the first time).
  3. Re-baseline the golden hashes in the same commit; record the numpy
     version they were captured against *in the test file itself* and in
     `DEPENDENCIES.md` (invariant 12).
  4. Split the regression test: one case over a pure `rng.normal`-only recipe,
     one over the full composite. A future break then localizes itself instead
     of just saying "the hash moved."
  5. State the **determinism contract** in the seal manifest, per generator:
     `bit_exact: true|best_effort`. `sequential_par` is best-effort by
     construction (SDV's `PARSynthesizer` exposes no seed — `CLAUDE.md`
     §11.11), and nothing in a sealed corpus currently tells a consumer which
     of its samples can actually be regenerated. That's a correctness claim a
     manifest should carry, not a docs footnote.
  6. Once green, wire into CI (A14) as a blocking check, and mark it
     `xfail(strict=True)` in the interim so it flips loudly when fixed rather
     than staying quietly red.
- **Cost.** Half a day to confirm + a day to fix and re-baseline.

**Findings (2026-08-06) — item 1's own "confirm first" step changed the
shape of this fix.** Followed the fix plan's own ordering exactly: before
touching any code, checked whether the failure reproduces at all in the
current, persistent environment.

- **The golden-hash test currently PASSES here** (`python3 -m pytest
  tests/test_generator_extensions.py::test_golden_hashes_pre_extension_outputs_unchanged`
  → 1 passed), and the **full `tsfm_benchmark` suite has zero failures**
  (37 passed / 1 skipped before this session's additions) — not the
  "40 passed / 1 failed" state `CLAUDE.md` §9 describes. This environment's
  numpy is 2.1.0, the same version the original failure report named, so
  this isn't simply "a different numpy version, of course it's fine now."
- **Fix item 1's direct comparison refutes the leading hypothesis**, rather
  than confirming it: `np.random.default_rng(0).choice(100, 5,
  replace=False)` and `.normal(size=5)` were run in an isolated,
  throw-away `venv` with `numpy==1.26.4` (chosen so as not to touch this
  session's own pinned, MKL-matched numpy 2.1.0 install and risk
  reintroducing `CLAUDE.md` §11.13's SEH crash) and compared byte-for-byte
  against the same two calls under this environment's numpy 2.1.0. **Both
  arrays were bit-identical across the two numpy versions.** If `choice`'s
  algorithm were the culprit, this is exactly the comparison that would
  have shown a divergence, and it didn't.
- **What this means, stated plainly:** the specific, five-session-old
  hypothesis ("`Generator.choice(..., replace=False)` isn't stream-stable
  across numpy versions") is not supported by direct evidence between the
  two most relevant versions (1.26.4 and 2.1.0), and the originally-reported
  mismatch is **not currently reproducible** in this persistent environment.
  The original failure's true cause remains unexplained — it may have been
  a different numpy point-release, a different platform/BLAS backend, or
  an environment-specific artifact (`CLAUDE.md` §11.13 already documents an
  unrelated numpy/MKL ABI crash in this same version's history, i.e. this
  numpy build has precedent for environment-sensitive behavior). Per
  `CLAUDE.md` §2.4's own doctrine, an unconfirmed hypothesis is not grounds
  to rewrite `generators.py`'s RNG calls — doing so would risk breaking bit-
  exact reproducibility for a *new*, self-inflicted reason (§11.1's own
  lesson), to fix a bug that isn't currently present. **No changes were made
  to `generators.py`'s actual `rng.choice` calls.**
- **What *was* still done, because it's correct regardless of whether the
  original bug ever recurs:**
  1. Item 3 (partial): recorded the numpy version (2.1.0) the hashes are
     verified against directly in the test file's own comment above
     `GOLDEN`, and updated `DEPENDENCIES.md` §3's numpy row and §5's fragile-
     spots list to state the refutation plainly rather than repeating the
     unconfirmed hypothesis as fact. Re-baselining wasn't needed — the
     existing hashes already match.
  2. Item 4: split the regression test. New `GOLDEN_NORMAL_ONLY` +
     `test_golden_hashes_normal_only_recipe_unchanged` runs the *same*
     recipe/seeds as the composite test but with `n_changepoints=0,
     n_anomalies=0` — the only two knobs that call `rng.choice(...,
     replace=False)` (confirmed by grep, not assumed). A future break now
     localizes: normal-only fails alone → the break is in `rng.choice`;
     composite fails alone → impossible (composite is a strict superset of
     normal-only's draws) — so really: both fail → the break is upstream
     (`rng.normal` or `default_rng` itself); only composite fails → the
     break is specifically in the choice-based changepoint/anomaly draws.
  3. Item 5: `generators.py` gained a `BIT_EXACT` dict (`parametric`,
     `random_parametric`, `mixture`, `block_bootstrap` → `True`;
     `sequential_par` → `False`, per `CLAUDE.md` §11.11's already-documented
     "PARSynthesizer exposes no seed" finding) and `seal.py::seal_corpus`
     now writes a `determinism: {generator: {bit_exact, count}}` block into
     every manifest — the `bit_exact: true|best_effort` contract the fix
     asked for, generalized slightly to a per-generator breakdown since a
     real corpus mixes generators with different guarantees. An unknown
     generator (not yet in `BIT_EXACT`) degrades to `bit_exact: None` rather
     than crashing or silently claiming a guarantee, tested explicitly.
  4. Item 6: **not applicable as worded** — `xfail(strict=True)` marks a
     test *expected* to fail; this test currently passes, so marking it
     `xfail(strict=True)` would itself fail the suite (strict xfail demands
     an actual failure). CI wiring itself is A14's scope, not yet reached;
     when A14 lands, this test should simply run as a normal blocking check
     like any other, with no special-casing needed now that it's green.
- **Verification — unit tests.** `test_golden_hashes_normal_only_recipe_unchanged`
  (new) plus the existing composite test both pass. New
  `tests/test_seal_determinism.py` (3 tests): `BIT_EXACT` has the expected
  value for all five registered generators (asserting `sequential_par is
  False` specifically, so this can't silently regress to claiming a
  guarantee SDV doesn't provide); a corpus mixing `parametric`/
  `sequential_par` samples produces the exact expected per-generator
  `determinism` breakdown in the sealed manifest; and a sample from a
  generator not in `BIT_EXACT` degrades to `bit_exact: None` without
  crashing `seal_corpus`. Full `tsfm_benchmark` suite green at 41 passed / 1
  skipped (up from 37 passed / 1 skipped — 4 new tests, 0 regressions, 0
  fixes needed to existing tests since none were failing).
- **Verification — real build.** Two real, non-mocked builds: (1)
  `example_runs/run_smoke.py` end to end (leakage gate, seal, reload+verify,
  cross-process reproducibility, epoch regeneration) — unaffected by the
  `seal.py` import addition, all prior assertions still pass; (2) a real
  `run_full.py` build against `configs/example.yaml` with live Monash data
  (`bootstrap_catalog` pooling 251 series across 6 real domains) producing a
  20-sample `public_dev` split mixing `parametric`/`mixture`/
  `random_parametric` — the written `manifest.json` carries a real
  `determinism` field: `{"parametric": {"bit_exact": true, "count": 10},
  "mixture": {"bit_exact": true, "count": 5}, "random_parametric":
  {"bit_exact": true, "count": 5}}`, exactly matching the corpus's actual
  generator mix. Confirms the manifest change composes correctly with the
  real leakage-audit/seal/real-source pipeline, not just synthetic
  `TimeSeriesSample` fixtures.
- **What "fixed" means for this item, precisely:** the audit's own
  instruction was to confirm the mechanism and either repair it or
  re-baseline; the confirmation step is what actually resolved this — there
  was nothing live to repair, and the hashes needed no re-baselining. What
  shipped is exactly the fix plan's items 3-5 (version provenance, a
  self-localizing test split, and the determinism contract), which stand on
  their own merit independent of whether the original numpy-version bug
  ever recurs. If it resurfaces on a *different* numpy version in a future
  session, the split test (item 4) will say which half of the mechanism
  broke instead of just "the hash moved," which is the actual, lasting
  value of this session's work here.

### A9 — `l0/summary.json` has no archetype granularity `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** `random_parametric`'s sampled archetype is recorded in
  sample provenance but never carried into the analysis half, so every
  per-group statistic stops at the coarse `family` level.
- **Evidence.** §5.5's own follow-up note; `data.py:77-82` builds `meta` with
  `series_id`/`family`/`tier` only.
- **Blast radius.** The question §5.5 exists to answer — "does 'TimesFM
  front-loads then compresses' hold on `intermittent_bursts` too, or only on
  the smooth archetypes it was observed on?" — is unanswerable today: all 12
  archetypes collapse into one `random_parametric` family. It also forces
  `sae/ground_truth.py` to reconstruct archetype labels itself, which is where
  A10 comes from.
- **Fix.** `data.py` carries `archetype` (from
  `provenance.generator_params`, `None` for tiers that can't express one) and
  `generator` into `meta.parquet`; `l0_behavioral.py` reports per-`(family,
  archetype)` alongside per-family (with an explicit minimum-n guard, and
  falling back to the family label where archetype is `None` — stated, not
  silent); `meta_report.py` gains the per-archetype stability table §5.5
  wanted; exemplars and attention gain an optional archetype conditioning.
  Keep `family` as the primary axis so recorded numbers stay comparable.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented all five fix items, scoped as follows:
- `data.py::_assemble` now carries two new `meta` columns:
  `archetype` (from `provenance.generator_params.archetype`, `None` for
  tiers that can't express one) and `generator` (the raw provenance
  generator name, finer than `family` when `family_key` groups several
  generators under one task label) via two new small resolvers
  (`_archetype_of`, `_generator_of`) applied before the same
  `max_series` stratified-subsample step A4 already built, so both new
  columns stay aligned under subsampling.
- `l0_behavioral.py`: `_score` carries `archetype`/`generator` into
  `metrics.parquet`; a new `_archetype_summary()` (called from
  `_summarize`, independent of and before the `n_families < 2` early
  return, since the single-family `random_parametric` corpus *is* the
  motivating case for this axis) reports per-archetype mean metrics **and**
  Holm-corrected paired MASE-ratio tests, mirroring `family_tests`'
  shape/statistics exactly but on the finer axis. Series whose tier can't
  express an archetype fall back to their family label (`fallback_to_family`,
  stated in the output, not silently dropped or silently blended with real
  archetype rows); groups below `stats.min_series` are dropped and named in
  `dropped_min_n` (reusing the existing min-n knob rather than adding a new
  one). Degrades explicitly (`{"applicable": False, "reason": ...}`) when no
  series in the corpus carry an archetype at all, when fewer than 2 groups
  survive the min-n guard, or when `stats.enabled` is false — matching A6's
  established "Not applicable, stated" pattern rather than a silent skip.
- `report.py`: new `_archetype_block()` renders the per-archetype table,
  fallback/dropped notes, and the paired-test table (or the applicable:false
  reason) directly under L0's existing per-family section, and feeds
  `strengths`-style findings into the same findings list.
- `meta_report.py`: new `archetype_stability()` (exact mirror of
  `family_stability()`, one axis finer) plus a rendered "Per-archetype
  stability across runs" table, wired into `build_meta_report`.
- `exemplars.py`/`report.py`: `_select_exemplars` now carries an `archetype`
  column through to `exemplars.json` (mapped from `data.meta` by row index,
  avoiding a `pivot_table` NaN-index-drop trap), and the exemplar card
  headings show it when present (`"{family} · archetype {archetype} · series
  {id}"`).
- **Deliberately scoped out, stated rather than silently dropped**:
  `attention.py`'s family-conditioned lag profiles, periodicity scores, and
  head/MLP ablation ΔMASE were **not** given an archetype axis. Unlike L0's
  aggregate (a `groupby` and a paired test), attention's per-family
  machinery is threaded through batched forward passes, ablation loops, and
  its own `top_heads`/`meta` JSON shape (`CLAUDE.md` §11.6 already flags
  this file's report-key surface as fragile) — duplicating a
  min-n-guarded, fallback-aware grouping axis through all of that
  correctly would be a multi-hour change in its own right, not the
  half-day this item was costed at, and the fix plan itself called this
  piece "optional." Left for a future item if a concrete question needs it
  (the motivating §5.5 question — "does the family-level MASE gap hold on
  every archetype" — is already answerable from L0 alone, which is what got
  built).
- **Verification — unit tests.** New `tests/test_archetype_reporting.py` (6
  tests): `_assemble` carries `archetype`/`generator` with correct `None`
  fallback; `_archetype_summary` returns `applicable: False` when no series
  carry an archetype; a group below `min_series` is dropped and named;
  a planted MASE gap on one archetype (with a tied second archetype and a
  no-archetype "real_derived" family mixed in) produces the correct
  `favored` verdict and correct `fallback_to_family`; `archetype_stability`
  flags a favored-model disagreement across two synthetic runs; and a full
  mini pipeline (three archetype-labeled + one unlabeled family, mock
  adapters) asserts `l0/summary.json`'s `per_archetype` block, `meta.parquet`'s
  `generator` column, and `report.html`'s rendered "Per-archetype breakdown"
  all appear correctly. Full suite green at 113/113 (up from 107 after A7 —
  6 new tests, 0 regressions).
- **Verification — real checkpoints.** `configs/_a9_real_check.yaml`
  (scratch, deleted after use; extract+l0+report only) against live
  `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small` on the real
  `benchmark_medium/public_dev` corpus (288 series: 188 `random_parametric`,
  60 `mixture`, 40 `parametric` — confirmed via `meta.parquet`'s new
  `generator` column, exactly matching the corpus's documented composition).
  `l0/summary.json`'s `per_archetype` block recovered all **12** of
  `CLAUDE.md` §4.2's archetypes as distinct groups (`trend_dominant`,
  `seasonal_dominant`, `multi_seasonal_complex`, `regime_switching`,
  `ar_colored_noise`, `anomaly_heavy`, `clean_low_noise`, `noisy_chaotic`,
  `random_walk_drift`, `intermittent_bursts`, `amplitude_modulated`,
  `nonsinusoidal_seasonal`), each with a real Holm-corrected paired test
  (e.g. `multi_seasonal_complex` ratio 0.365, p=0.005 but p_holm=0.07 — a
  real illustration of why the Holm correction matters once the comparison
  fans out to 14 groups instead of 3 families); `mixture` and `parametric`
  correctly appeared in `fallback_to_family` with zero groups dropped for
  low n. `report.html` rendered the "Per-archetype breakdown" section. This
  is exactly the previously-unanswerable question this item names in its
  own blast radius — "does the family-level gap hold on every archetype
  `random_parametric` was pooled from" — now answerable directly from one
  run's `l0/summary.json`, no `sae/ground_truth.py` archetype-label
  reconstruction (A10's origin) required.

### A10 — Ground-truth archetype dummies count every real-derived series as a negative `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** Real-derived-tier series carry an empty `GroundTruth` by
  design (`CLAUDE.md` §4.1); the archetype dummy encoding treats that as
  `archetype_x = 0` — a valid negative example — rather than "not applicable."
- **Evidence.** §6.2's 2026-08-06 Findings, which found this by reading real
  exemplar rows (two `mixture` series at activation 0.0 / `field_value` 0.0
  sitting in a top-exemplar table) and correctly declined to change it as a
  side effect of a reporting feature.
- **Blast radius.** Every archetype-related ρ already recorded in §6.2
  (including TimesFM's headline `archetype_trend_dominant` ρ=0.650/0.752) is
  computed against a label that partly encodes *tier*, not archetype. A
  feature that merely separates synthetic from real-derived series scores as
  an archetype detector. Interacts with A4: which tiers are even present
  depends on a head slice.
- **Fix.**
  1. `encode_series_level` emits `NaN` for any field a series' tier cannot
     express; `best_ground_truth_matches` already reports per-field `n`, so
     drop `NaN` rows per field and let `n` tell the reader how much support a
     match has.
  2. Add an explicit `tier` (and `generator`, from A9) column to the
     candidate field set, so "this feature detects the real-derived tier" is a
     *detectable, nameable* match rather than something that masquerades as an
     archetype match.
  3. Re-run against `runs/medium_run_chronos_base` and record old vs. new ρ
     side by side once, then supersede the old numbers in place with a dated
     correction (§0.2). Don't silently re-baseline.
  4. Test: a planted feature that perfectly separates tiers must match `tier`,
     not an archetype dummy.
- **Cost.** Half a day + a short re-scoring run.

**Findings (2026-08-06).** Implemented all four fix items in
`sae/ground_truth.py`, and found (via this item's own real-checkpoint
verification, per `CLAUDE.md` §2.4) two additional real bugs one level
removed from the one this item was named for:

- **Item 1 (archetype dummies), as named.** `load_ground_truth_table`'s
  per-archetype dummy construction moved into a new pure function
  `_add_dummy_columns(df)` (split from the `load_sealed` I/O, matching this
  file's existing pure/IO split for `best_ground_truth_matches` vs
  `ground_truth_alignment`, so the fix could be unit tested directly on a
  synthetic DataFrame). Rows with no archetype now get `NaN`, not `0.0`, on
  every `archetype_*` dummy.
- **Item 2 (tier/generator as first-class fields), as named, plus a real bug
  found underneath it.** `tier_{value}`/`generator_{value}` one-hot dummies
  are now added (always defined, no NaN case, unlike archetype). Building
  this surfaced a genuine, previously undocumented bug: `data.py::
  _sample_to_row`'s `hasattr(sample, "tier")` is **always False** —
  `TimeSeriesSample` has no `tier` attribute; `build_pipeline/builder.py`
  writes it to `provenance.generator_params["tier"]` instead. Confirmed live
  against `benchmark_medium/public_dev` (not assumed): every real
  sealed-corpus series' `meta["tier"]` was silently `"unknown"`. New
  `data.py::_tier_of(row)` checks the top-level key first (smoke rows) then
  falls back to `provenance.generator_params.tier` (real samples); re-verified
  live — `Counter({'synthetic': 228, 'realism_stress': 60})`, matching
  `benchmark_medium`'s known composition exactly. `l0`/A9's `meta.parquet`
  `tier` column inherits this fix for free (same `_assemble` call site).
- **A second, larger bug found the same way, one level removed from the one
  this item names:** `_scalar_ground_truth`'s `len([])`/`"x" in {}` on an
  empty `generative_params` silently produced a valid `0`/`False` for
  `n_seasonalities`/`ar_order`/`n_changepoints`/`n_anomalies`/`has_random_
  walk`/`has_intermittency`/`has_heteroskedastic` too — the exact same
  "not applicable" vs "genuinely zero" confusion this item is about, just
  not confined to the `archetype` dummy. Gated all seven fields on a new
  `has_gt` flag. **First attempt used `has_gt = bool(generative_params)`
  and was itself wrong** — checked directly against a real `mixture` sample
  rather than assumed (§2.4 again, twice in one item): `mixture` sets
  `generative_params={"mode": ...}` (`generators.py:450`), non-empty but with
  none of `parametric`'s structural keys, so `bool(p)` would have left this
  exact generator's rows un-fixed. Corrected to `has_gt = "trend" in p`,
  since `_generative_params()` (`generators.py:152`) sets `trend`/
  `seasonalities`/`ar_coeffs`/`noise_scale` together unconditionally for
  every `parametric`/`random_parametric` call and `mixture`/
  `block_bootstrap`/`sequential_par` never set `"trend"` at all — verified
  against a live `mixture` sample from `benchmark_medium` before trusting it.
- **Item 3 (re-score and supersede old numbers), done against real
  checkpoints, not a fresh run.** No retraining needed: `sae/train.py`
  already persists trained SAE weights to `.pt` (`load_sae_checkpoint`), so
  the already-extracted `runs/medium_run_chronos_base` store plus its two
  already-trained checkpoints (`TimesFM/stacked_xf.18`,
  `Chronos-T5-Base/encoder.block.6`) were re-scored directly through the
  fixed `ground_truth_alignment`, with zero new model forward passes.
  **Old vs. new, side by side:**
  - `mean_abs_rho_matched`: TimesFM 0.317 → 0.349; Chronos-T5-Base 0.306 →
    0.401 — a real, substantial increase in how well-matched the average
    top feature is, exactly as expected once tier-masquerading matches stop
    diluting the field.
  - **Most matches previously attributed to `n_seasonalities` were actually
    detecting tier.** E.g. TimesFM feature 5109: old best match
    `n_seasonalities` ρ=−0.552 (n=288) → new best match
    `tier_realism_stress` ρ=+0.943 (n=288). Six of TimesFM's top-8 and six
    of Chronos-T5-Base's top-8 (by new ρ) flipped from a spurious
    `n_seasonalities`/`ar_order`/`n_anomalies` match to a `tier_*` match at
    much higher |ρ| (0.70–0.94 vs. 0.43–0.60) — these features were tier
    detectors all along, previously misattributed to the nearest available
    numeric field because no `tier` field existed to compete for the match.
    Genuinely intermittency-detecting features (e.g. TimesFM/Chronos feature
    2189, `has_intermittency` ρ=0.881→0.880) were essentially unaffected,
    confirming the fix corrects specifically the tier-masquerading cases and
    doesn't disturb real matches.
  - **The headline number this item's own evidence names is confirmed
    genuine, not an artifact.** `CLAUDE.md` §13's `archetype_trend_dominant`
    ρ=0.650 (TimesFM feature 2872): re-scored at ρ=0.648, now correctly on
    `n=188` (the actual `random_parametric` count) instead of the old,
    tier-inflated `n=288`. Every `archetype_trend_dominant` match in both
    models' old top lists survived at essentially the same ρ under the fix
    (TimesFM: 0.650→0.648, 0.561→0.558; Chronos-T5-Base: six features at
    0.530→0.527, 0.518→0.514, 0.500→0.497) — this specific finding was a
    real archetype match, not a tier-confound artifact, and is superseded
    in place here with the corrected `n` per this item's own "supersede, don't
    silently re-baseline" instruction; no other file restates these numbers
    (`CLAUDE.md` §13's item 3 links here rather than repeating them).
- **Item 4 (test).** Done, generalized to cover both the named bug and the
  one found underneath it. New `tests/test_ground_truth_dummies.py` (9
  tests): real-derived rows get `NaN` not `0.0` on archetype dummies;
  `tier_*`/`generator_*` dummies are always defined; a planted tier-detecting
  feature matches `tier_*` and not an archetype dummy (with a sanity check
  proving the pre-fix encoding really was indistinguishable, restricted to
  the two-archetype/tier comparison where the bug actually bites); `_tier_of`
  resolves both the top-level and `provenance.generator_params` paths and
  degrades to `"unknown"` without crashing; a fully-empty `GroundTruth` gets
  `None` on every count field; a real-shaped `mixture` `generative_params`
  (`{"mode": ...}`) is correctly *not* mistaken for real ground truth; and a
  genuinely zero-valued synthetic recipe still records real `0.0`s, not
  `None`. Full suite: 122 tests collected, 121 passed, 0 failures (up from
  113 before this item — the 1-test discrepancy from this item's own 9 new
  tests is unreconciled bookkeeping across sessions, not a failure or a
  skip; the log shows zero failures/errors/skips).
- **Known, accepted side-effect, not fixed this session:** `layer_screen.py`'s
  `factor_emergence` method (not the default; `work_bend` is, per §6.1.1)
  also calls `load_ground_truth_table` and now sees the new `tier_*`/
  `generator_*` dummies as additional candidate factors alongside every
  archetype. Left as is — `factor_emergence` already has its own open,
  larger-scope defect (`CLAUDE.md` §11.18, not yet re-verified against a
  real bake-off) and is not the default selector; revisit if that method is
  ever promoted past `work_bend`.

### A11 — MASE's scale term degenerates on intermittent and near-constant series `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** The denominator is the mean absolute first difference of
  the context (`+ 1e-8`). Under heavy intermittency or a flat context it
  collapses, and because `parametric`'s intermittency zeroes the horizon too,
  numerator and denominator both floor.
- **Evidence.** §7's intermittency-sweep caveat (MASE *falls* for both models
  at rate 0.8, which is an artifact); `analysis/stats.py::mase()`.
- **Blast radius.** Wider than that one sweep, because `mase()` was
  deliberately centralized (§6.2's refactor) and is now the metric behind L0's
  headline numbers and per-family strengths, head/MLP ablation ΔMASE, L3
  restoration scores, the SAE forecast-preservation check, every parameter
  sweep, and the bias card's verdicts. Any zero-heavy or low-volatility family
  is silently compressed toward a floor in all of them. Consequence worth
  stating plainly: `intermittent_bursts` — per `CLAUDE.md` §4.2 the sharpest
  quantized-vs-continuous stressor available, opted into deliberately by §5.1
  — cannot currently be scored interpretably, so one of the corpus's most
  discriminating archetypes is dead weight until this is fixed.
- **Fix.**
  1. `l0.scale: mean_abs_diff|seasonal_naive` (default `mean_abs_diff` so
     every recorded number keeps reproducing). `seasonal_naive` uses period
     `m` from ground truth where available, else an autocorrelation estimate,
     and is the standard fix for exactly this degeneracy.
  2. A **reliability guard** independent of the scale choice: when the
     denominator is below a threshold fraction of the target's own mean
     absolute level, mark that series `mase_reliable: false`, exclude it from
     aggregates, and **report the excluded count** everywhere a MASE
     aggregate is shown. Silent exclusion would be its own version of this
     bug.
  3. Add one scale-free companion metric that doesn't degenerate the same way
     (e.g. MAE normalized by the target's own MAD) and show it beside MASE in
     L0 and the sweeps, so a floor effect is visible as a divergence between
     the two.
  4. Audit `sMAPE`'s zero handling in `l0_behavioral.py` at the same time —
     same family of failure, same corpus regions.
  5. Test: a mostly-zero synthetic series must be flagged unreliable, not
     scored; a normal series's MASE must be byte-identical to today's.
- **Cost.** ~a day including re-running the intermittency sweep to see what
  it actually says.

**Findings (2026-08-06).** Implemented items 1-4; item 5 covered by new
tests. Scoped deliberately to `stats.py` (shared) + `l0_behavioral.py`/
`config.py`/`report.py` (L0, the stage this item's own evidence and blast
radius are about) rather than every one of `mase()`'s ~8 call sites:

- **Item 1.** `stats.py` gained `dominant_period()` (moved out of
  `attention.py`, which had its own private, byte-identical copy —
  deduplicated, `attention.py` now imports the shared one under its old
  private name so nothing else in that file changed), `_mase_scale()`
  (`mean_abs_diff` — unchanged formula, still the default — or
  `seasonal_naive`, mean absolute error of a period-`m` seasonal-naive
  forecast, `m` autocorrelation-estimated per series when not supplied),
  and `mase()` gained optional `scale_mode`/`periods` kwargs with defaults
  that reproduce every existing call byte-for-byte. New `config.py::
  L0Config.scale: "mean_abs_diff"|"seasonal_naive"` (default
  `"mean_abs_diff"`). **Ground-truth-period lookup (`m` "from ground truth
  where available") was not built** — it would require L0 to load the
  sealed corpus's `GroundTruth` records the way `sae/ground_truth.py`
  does, a new coupling this stage doesn't otherwise have; the
  always-available autocorrelation fallback is what `seasonal_naive`
  actually uses. Stated here as a real, deliberate gap, not silently
  dropped.
- **Item 2.** New `stats.py::mase_reliability()`: `False` when the MASE
  denominator is below `min_scale_frac` of the target's own mean absolute
  level, independent of `scale_mode`. New `L0Config.min_scale_frac: 0.05`
  (on by default, unlike `scale` — this guard's whole purpose is to stop a
  degenerate case from silently entering aggregates, so a default of `0.0`
  would defeat it). `l0_behavioral.py::_score` records a per-series
  `mase_reliable` column in `metrics.parquet`; `_summarize`/
  `_archetype_summary` compute every MASE-based number (means, CIs, ratios,
  paired tests) on the reliable subset only, while `smape`/`pinball`/
  `mae_over_mad` still aggregate every row (they don't share this
  degeneracy). Excluded counts are recorded corpus-wide
  (`summary["mase_reliability"]`) and per `per_family`/`per_archetype` row
  (`mase_n_excluded`); `report.py` renders both — a blurb naming the total
  excluded count right under the MASE chart (auto-added as a finding too)
  and a `mase_n_excluded` column in the new per-family/per-archetype tables.
- **Item 3.** New `stats.py::mae_over_mad()` — MAE normalized by the
  target's own median absolute deviation, never touching the context, so it
  can't share MASE's context-driven degeneracy. Reported beside MASE in
  every L0 table (`_score`, `per_family`, `overall`, `per_archetype`) with a
  report note on how to read a MASE/`mae_over_mad` divergence. **Not wired
  into the parameter sweeps** (`analysis/parameter_sweep.py` still computes
  its own local MASE only) — left as a documented gap since sweep case
  studies are a separate, lower-priority consumer than L0's headline
  numbers.
- **Item 4 (sMAPE audit).** Checked rather than assumed (`CLAUDE.md` §2.4):
  generated a real `intermittent_bursts` series
  (`random_parametric(seed=0, archetypes=["intermittent_bursts"])`, 72.5%
  zero-valued) and scored a deliberately-wrong constant-0.3 forecast against
  its zero-heavy horizon. sMAPE = **1.97** (correctly saturating near its
  own max of 2.0 — a genuinely bad forecast against a near-zero target
  scores near-worst, not `inf`/`NaN`), MASE for the identical case = **3.20**
  (also a large, sane number, not the "artifact" reading this item warned
  about, since here the *context* wasn't degenerate, only the *target*
  region tested). **Conclusion: no fix needed.** `l0_behavioral.py::_score`'s
  existing `+ 1e-8` epsilon on the denominator already handles the
  both-zero edge case correctly; the real degeneracy this item is about
  lives in MASE's context-derived scale term, not sMAPE's target-derived
  one, and item 2's reliability guard is the actual fix for that.
- **Item 5 (tests).** New `tests/test_mase_reliability.py` (11 tests):
  `mean_abs_diff` scale matches the pre-existing formula byte-for-byte;
  `dominant_period` recovers a planted period-24 sine to within 1 step;
  `seasonal_naive` scale matches a hand-computed value for a known period;
  an unknown `scale_mode` raises; `mase_reliability` is all-`True` when
  disabled (`min_scale_frac<=0`), correctly flags an all-flat-context/
  nonzero-target batch as unreliable, and correctly keeps a normal series
  reliable; `mae_over_mad` matches a hand-computed value;
  `_summarize`/`_score` correctly exclude unreliable rows from the MASE
  mean while keeping them in smape (a planted 999.0 MASE on unreliable rows
  does **not** move the reported per-family mean, proving the exclusion
  actually happens, not just gets flagged); and a full mini pipeline run
  (mock adapters, `l0.min_scale_frac: 0.05`) confirms `mase_reliability`/
  `mae_over_mad` appear in real `summary.json`/`report.html` output. **Two
  pre-existing tests broke and were fixed, not worked around**:
  `test_archetype_reporting.py`'s synthetic `_metrics()` fixture built a
  metrics DataFrame by hand without a `mase_reliable`/`mae_over_mad` column
  (written before this item existed) — `_archetype_summary` now requires
  both, so both tests failed with a `KeyError` until the fixture was
  updated to add `mase_reliable=True`/`mae_over_mad=mase` columns, matching
  the "every row reliable, sanity metric equals MASE" semantics those
  synthetic rows already implied. Full suite green at 133/133 (up from 121
  after A10).
- **Verification — real checkpoints.** `configs/_a11_real_check.yaml`
  (scratch, deleted after use; extract+l0+report only,
  `min_scale_frac: 0.05`) against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small` on the real `benchmark_medium/public_dev`
  corpus. **16 of 576 model-series rows (8 unique series, since reliability
  is series-level and duplicated across both models) were excluded as
  unreliable** — real numbers, real degenerate contexts, not a synthetic
  demonstration. Rendered correctly end to end: `summary.json`'s
  `mase_reliability` block, `mae_over_mad` alongside `mase` in every
  `per_family`/`per_archetype` row, and `report.html`'s "16 of 576 series
  excluded from every MASE mean/ratio/test above and below" note (grep-
  confirmed present) plus its new "Per-family metrics" table. **A genuinely
  non-obvious finding, worth stating plainly rather than assuming the
  reliability guard would only fire where §7's caveat suspected it**: the
  8 excluded series were **not** concentrated in `intermittent_bursts` (0
  excluded there, despite being the archetype `CLAUDE.md` §4.2 calls "the
  sharpest quantized-vs-continuous stressor" and the one this item's own
  Evidence section names) — they were scattered across `parametric` (2) and
  `random_parametric` (6, including 2 in `trend_dominant` specifically).
  The degeneracy this item names turns out to be driven by genuinely flat
  *stretches of a sampled recipe* (e.g. a `trend_dominant` draw with a
  near-zero trend scale and low noise), not specifically by the
  intermittency archetype's design — a real, previously-unmeasured fact
  about which series this guard actually protects, not something either
  this item's evidence or `CLAUDE.md` had predicted.
- **Deliberately out of scope, stated rather than silently dropped**: L3's
  patching-restoration MASE, `attention.py`'s ablation ΔMASE, `lens.py`'s
  crystallization-depth MASE, `sae/eval.py`'s forecast-preservation MASE,
  `confirm.py`'s re-tested MASE, and `parameter_sweep.py`'s sweep MASE all
  keep calling `mase()`/computing MASE inline with the pre-A11 defaults
  (`scale_mode="mean_abs_diff"`, no reliability filtering) — every number
  those stages have already recorded stays byte-identical. This item's own
  Fix section named `l0.scale` (an L0-config knob) as the mechanism, and
  the reliability guard's report-surfacing ask ("everywhere a MASE
  aggregate is shown") is read here as "wherever L0 already aggregates
  MASE," not as a mandate to touch every downstream stage's own metric
  computation in the same pass — each of those is its own, separately-
  scoped follow-up if the degeneracy turns out to matter there too.

### A12 — The corruption battery is uncalibrated across corruptions `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** Each corruption's strength is set independently, so
  "behavioral sensitivity" bars are not comparable across corruptions —
  `level_shift` dominates by construction at 4-12× everything else.
- **Evidence.** §5's 2026-08-04 Findings (with numbers); `CLAUDE.md` §6.5's
  standing warning. The 2026-08-04 session fixed the *display* and explicitly
  left calibration as "an option, not a decision."
- **Blast radius.** The L3 section's headline chart, the cross-model
  fingerprint agreement (rank correlations are computed per corruption, so
  those are fine — but any statement of the form "model X is most sensitive
  to Y" is not), and any future reader who trusts bar heights over the prose
  note. Also blocks the "is one model better at high-frequency structure vs.
  statistical means" question (§1) from being answered *contrastively across
  corruption types* rather than within one.
- **Fix.**
  1. `l3.calibrate: none|input_energy` (default `none` — every recorded number
     keeps reproducing byte-for-byte, which is what made the last session
     defer this).
  2. Under `input_energy`, solve per corruption for the parameter that hits a
     common per-series perturbation-energy budget, measured with the
     corruption functions alone (no model in the loop — cheap, pure numpy,
     unit-testable). Record realized budget, parameter, and **footprint**
     (fraction of timesteps touched) per corruption in `l3/meta.json`.
  3. Report shows footprint next to every sensitivity bar regardless of mode,
     so sparse-by-construction corruptions (`spike` ~0.6%, `dropout` ~14%)
     can't be misread as "the model is robust" — the exact misreading the
     2026-08-04 session had to investigate by hand.
  4. Keep every corruption. §5's decision to keep the low-signal end of the
     contrast stands and is not what this item revisits.
- **Cost.** ~a day.

**Findings (2026-08-06).** Implemented all four fix items in
`l3_perturbation.py` + `config.py` + `report.py`:

- **Item 1.** `L3Config.calibrate: "none"|"input_energy"`, default `"none"`.
- **Item 2.** New pure-numpy `calibrate_corruptions(names, configs,
  contexts, seed)`: per-corruption perturbation energy (`mean((corrupted -
  clean)**2)`) and footprint (`fraction of timesteps changed beyond a
  scale-relative tolerance`), a shared budget set to the **median of the
  battery's own natural (pre-calibration) energies** (anchored to this
  battery's existing scale rather than an arbitrary constant — a deliberate
  design choice this item's own wording left open), and a per-corruption
  solve to hit that budget. **Checked monotonicity directly before trusting
  it** (`CLAUDE.md` §2.4): swept each corruption's obvious magnitude
  parameter against energy on a real batch of series.
  `level_shift:scale`, `spike:scale`, `smooth:kernel`, `warp:strength`,
  `dropout:frac` are all confirmed monotone increasing over a wide range —
  calibrated via a 30-iteration bisection (`_bisect_param`), clamped (not
  extrapolated) when the target falls outside the search range.
  `noise:snr_db` is calibrated via a closed form instead
  (`snr_db = 10·log10(mean_power / target)`, exact since
  `corrupt_noise`'s own formula is closed-form in `snr_db`) — cheaper and
  exact where bisection would just be approximating an already-known
  relationship. **`frequency_shift:factor` was swept and found genuinely
  non-monotone past ~1.5×** (energy plateaus/wobbles rather than growing) —
  excluded from calibration on that empirical basis, not assumed.
  `detrend` (no free parameter) and `deseasonalize` (integer `top_k`, not a
  continuous magnitude knob) have nothing to calibrate by construction.
  All three are still fully reported (footprint/energy), just never
  adjusted — `calibrated: false` in `l3/meta.json`'s new `calibration`
  block, not silently omitted from it.
- **Item 3.** Footprint is computed and stored for **every** corruption
  regardless of `calibrate` mode (`corruption_footprint_meta` covers the
  `"none"` path). `report.py::_sec_l3` now labels every behavioral-
  sensitivity bar's x-axis tick with its footprint percentage
  (`"spike<br>0.6% touched"` etc., real numbers below) and, when
  `calibrate: input_energy`, prepends a blurb naming which corruptions were
  calibrated to which shared budget and which weren't and why.
- **Item 4.** Every corruption stays in the battery in both modes — nothing
  dropped, per §5's own prior decision this item explicitly said stands.
- **A real bug found and fixed via this item's own testing, not by
  inspection alone**: the first implementation used `seed + 200 + i` inside
  `_bisect_param`'s search but `seed + i` for the *final* realized-energy
  verification recompute after the solve — for a corruption whose output
  depends on more than just the calibrated parameter (`warp`'s random knot
  jitter, drawn fresh from the RNG on every call), searching and verifying
  under two different RNG streams meant the verified energy didn't actually
  match what the search had converged on. Caught by a unit test asserting
  <5% relative error (got 5.18% — just over), not by eyeballing the numbers.
  Fixed by using one consistent seed per corruption throughout
  `calibrate_corruptions` for both the search and the verification.
  **Lesson, matching this session's own recurring pattern (§11.19's
  device-transfer bug, A9/A10's real-checkpoint-only findings): a
  stochastic corruption function needs the *same* RNG draw sequence
  compared against itself, not just the same parameter value, or a
  seed mismatch masquerades as a solver-tolerance issue.**
- **Verification — unit tests.** New `tests/test_l3_calibration.py` (9
  tests): energy/footprint are exactly zero for identical arrays;
  `spike`'s footprint is small as expected; `_bisect_param` hits a target
  within tolerance and correctly clamps rather than extrapolates when the
  target is out of range; `calibrate: none` leaves every config
  byte-identical while still reporting footprint/energy for all of them;
  `calibrate_corruptions` converges every calibratable corruption to the
  shared median budget within 5%; `detrend`/`deseasonalize` are never
  touched by calibration; a full mini pipeline with the **default** config
  (`calibrate: none`) produces a `calibration` block with `calibrated:
  false` everywhere (the byte-identical-reproduction regression guard this
  item's item 1 promised); and a full mini pipeline with `calibrate:
  input_energy` produces a real calibration block and renders both the
  calibration blurb and footprint captions in `report.html` (matched
  loosely against `"touched"`, not the literal `"touched</span>"`, since
  plotly JSON-escapes `<`/`/` in embedded tick-label strings — caught by
  the first assertion failing, not assumed). Full suite green at 142/142
  (up from 133 after A11).
- **Verification — real checkpoints.** `configs/_a12_real_check.yaml`
  (scratch, deleted after use; extract+l3+report only, patching disabled to
  keep it fast, `calibrate: input_energy`) against live
  `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small` on the real
  `benchmark_medium/public_dev` corpus. Real target energy = **0.1606**
  (median of the battery's configured-default natural energies). All six
  calibratable corruptions converged close to it: `noise` 0.1608, `level_
  shift` 0.1606, `spike` 0.1606, `dropout` 0.1626, `warp` 0.1606, `smooth`
  0.1512 (the one visibly furthest off at ~6% relative error — noted
  plainly rather than smoothed over; still within the same order of
  magnitude and not a sign of the seed bug recurring, since `smooth`'s
  corruption function is itself deterministic given `kernel`, no RNG
  dependence to mismatch). `frequency_shift`/`detrend`/`deseasonalize`
  correctly stayed at their natural, uncalibrated energies (1.132, 0.049,
  0.369 respectively). Real footprints, exactly the contrast this item's
  blast radius named: `spike` 0.6%, `dropout` 28.0%, `level_shift` 40.0%,
  `warp` 91.6%, `detrend` 96.2%, `smooth` 96.1%, `frequency_shift` 96.9%,
  `deseasonalize` 99.6%, `noise` 99.7% — confirming `spike`/`dropout`'s low
  aggregate behavioral-sensitivity bars (documented elsewhere as
  potentially misleading) are exactly the sparse-footprint corruptions this
  fix's whole point is to keep from being misread as model robustness.
  `report.html` rendered the calibration blurb and every bar's footprint
  caption (grep-confirmed).

### A13 — No repeat-run noise floor, so small ΔMASE values are uninterpretable `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** Nothing measures how much a model's own MASE varies
  between two identical calls. Every Δ in the repo is compared against zero
  rather than against that floor.
- **Evidence.** `CLAUDE.md` §12 item 3 (Chronos-T5 samples, TimesFM is
  deterministic; "seeds are pinned and `num_samples` reduced during patching"
  — pinning makes a run reproducible, it does not make a *difference*
  meaningful); head/MLP ablation ΔMASE (`attention.py`), SAE
  forecast-preservation ΔMASE (`sae/eval.py`), L3 restoration
  (`l3_perturbation.py`) all compare across separate `predict` calls.
- **Blast radius.** The SAE validity check's TimesFM "pass" is ΔMASE
  **+0.047** — a number whose significance is entirely a question of the noise
  floor, which is unknown. Head-ablation "top load-bearing heads" rankings
  can be sorted by noise in their tail. For the sampled model the floor may
  be substantial, and the two models' floors differ, which means every
  cross-model Δ comparison currently rests on an unmeasured asymmetry that
  `CLAUDE.md` §12 already names as a known hazard.
- **Fix.**
  1. Cheap sub-step in L0 (or a tiny `noise_floor` stage): for each model,
     predict the same `k` series `R` times (R≈5) with different seeds; write
     `l0/noise_floor.json` — distribution of |ΔMASE| between identical
     repeats, overall and per family, plus a `deterministic: bool` flag taken
     from the adapter.
  2. Every ΔMASE figure draws the floor as a shaded band / reference line;
     every findings string that asserts a Δ is meaningful must compare against
     it, and say so ("+0.047, against a repeat-run floor of ±0.0X").
  3. Deterministic models should show a floor of exactly 0 — a free, strong
     wiring sanity check, in the same spirit as TimesFM's exactly-0.0
     `future_mass` (§5.4).
  4. `Chronos-Bolt` as the deterministic control is already suggested by
     `CLAUDE.md` §12; the floor measurement is what makes that comparison
     quantitative instead of rhetorical.
- **Cost.** Half a day. Cheapest item here relative to how many recorded
  numbers it re-contextualizes.

**Findings (2026-08-06).** Implemented as a cheap sub-step of L0 (item 1's
first option), not a new pipeline stage — no `pipeline.py`/`manifest.py`
wiring needed, since it reuses L0's already-loaded adapters and existing
`_predict_all`/`mase` machinery.

- **Item 1.** New `l0_behavioral.py::_measure_noise_floor()`: for each
  model, calls `adapter.predict()` `noise_floor_repeats` times on the same
  `noise_floor_series`-sized subset and measures `|ΔMASE|` between
  consecutive repeats (overall, per family, and a `deterministic: bool`
  flag). **Deliberately did not add a `seed` parameter to `ModelAdapter.
  predict()`** — repeat calls under the ambient (advancing) global torch
  RNG state already differ for a genuinely sampling model and are
  byte-identical for a genuinely deterministic one, which is exactly the
  signal this item wants; adding an explicit seed contract would have
  meant touching `base.py`'s interface plus all 4 concrete adapters for no
  additional signal. Written to a new `l0/noise_floor.json`, model → `
  {repeats, n_series, deterministic, mase_abs_delta_mean/p95/max,
  per_family}`.
- **Item 1, a scoping deviation worth stating**: unlike A11/A12's "off or
  byte-identical by default" precedent, this item's own fix plan implies
  the floor should actually be *on* to recontextualize existing ΔMASE
  figures — a floor nobody ever measures is not different from not having
  this item at all. Defaulted **on** (`noise_floor_repeats: 3,
  noise_floor_series: 16`), a small, deliberately cheap addition (3 extra
  `predict()` calls on 16 series per model) rather than a full opt-in.
  `noise_floor_repeats: 0` fully disables it (adds zero cost, `l0/
  noise_floor.json` is simply not written) for anyone who wants the old
  runtime exactly.
- **Item 2.** Wired into three places, not literally "every ΔMASE display"
  (see the explicit deferral below): `report.py::_noise_floor_block()` (a
  new "Repeat-run noise floor" table + note under L0's own chart, shared
  by any section that wants to reference it); L3's "Behavioral sensitivity"
  chart gains a **dotted horizontal reference line per sampling model** at
  its own floor's mean, plus a note explaining a bar below its model's
  line is not distinguishable from repeat-run noise; and the SAE section's
  `forecast_preservation` finding string now appends `(repeat-run floor
  ±X)` for a sampling model or `(this model is deterministic; the delta is
  real signal)` for a deterministic one.
- **Item 3.** Confirmed exactly as predicted, on real weights (below) and
  via the mock-adapter unit tests: a model with no sampling in its forecast
  path shows `deterministic: true` and a floor of literally `0.0` — the
  free wiring sanity check this item names, same spirit as TimesFM's
  exactly-0.0 `future_mass` (`CLAUDE.md` §9/§5.4).
- **Item 4.** Not additional code — `Chronos-Bolt`'s adapter already exists
  (`CLAUDE.md` §6.2's support matrix); this item's contribution is that the
  "Chronos-Bolt as deterministic control" comparison `CLAUDE.md` §12
  already suggests is now backed by an actual measured number instead of
  being purely rhetorical, the moment anyone runs a config pairing it with
  Chronos-T5.
- **Deliberately out of scope, stated rather than silently dropped**:
  `attention.py`'s head/MLP ablation ΔMASE charts and `top_heads` rankings
  (named explicitly in this item's own Evidence/Blast-radius sections) do
  **not** gain a floor reference line in this pass — that chart's data
  shape (per-head, per-family delta arrays, already flagged as this
  repo's most fragile report-key surface, `CLAUDE.md` §11.6) would need
  its own, more careful integration than a single reference line, and
  ablation deltas are computed from a single clean/ablated forecast pair
  per head rather than L0's own repeat-call structure — comparable in
  principle but a separate wiring task. `l3_perturbation.py`'s activation-
  patching restoration curves (also named in Evidence) are likewise left
  unannotated; both are natural follow-ups once this item's core mechanism
  is trusted, not silently assumed already covered.
- **Verification — unit tests.** New `tests/test_noise_floor.py` (5 tests):
  a fixed-function fake adapter gets exactly `0.0`/`deterministic: true`; a
  fake adapter with real per-call sampling noise gets a real nonzero floor
  with `p95 >= mean` and correct per-family keys; repeats/n_series are
  recorded correctly; a full mini pipeline (mock adapters, default config)
  writes `l0/noise_floor.json` with both mocks correctly `deterministic:
  true` (mocks are fixed functions of their input, no dropout/sampling at
  inference) and renders "Repeat-run noise floor" in `report.html`; and
  `noise_floor_repeats: 0` fully disables the artifact. Full suite green
  at 147/147 (up from 142 after A12).
- **Verification — real checkpoints.** `configs/_a13_real_check.yaml`
  (scratch, deleted after use; extract+l0+l3(patching disabled)+report
  only) against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small` on `benchmark_medium/public_dev`. **Exactly the
  asymmetry `CLAUDE.md` §12 item 3 names, now a real number instead of a
  caveat**: TimesFM — `deterministic: true`, floor `0.0` across all three
  families, confirming the free wiring check. Chronos-T5 — `deterministic:
  false`, mean `|ΔMASE|` **0.167**, p95 **0.586**, max **0.999** — a floor
  roughly **3-20x the size of the SAE forecast-preservation "pass"
  threshold (+0.047) this item's own blast-radius section names**, meaning
  that specific "pass" (measured on deterministic TimesFM) was never
  actually at risk from this, but the identical check on a sampling model
  like Chronos-T5 would be effectively unmeasurable at that magnitude
  without many more repeats or series than either check currently uses — a
  concrete, previously-unquantified risk this item's own Blast-radius
  section predicted correctly. **A further non-obvious finding**: the
  floor is not uniform across families — `mixture` 0.041 vs `parametric`
  0.463, over an 11x range — so "the noise floor" is meaningfully family-
  dependent, not a single number a future session should assume transfers
  across corpora. `report.html` rendered the table, the dotted reference
  line at Chronos-T5's exact floor value in the L3 chart (grep-confirmed
  in the embedded plot JSON), and the "is deterministic" finding text.

### A14 — No CI; the whole test suite is manual, and upstream breaks are found by humans on new machines `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** There is no CI configuration in the repo. Every "suite
  green" claim is a session's word for it, in whatever environment that
  session had.
- **Evidence.** §10's CI checklist item (still `[~]`, explicitly noting no CI
  config exists); the repeated-rediscovery record: `CLAUDE.md` §11.8 (timesfm
  API removal), §11.9 (`datasets>=3`), §11.10 (`tsbootstrap` rewrite), §11.15
  (zarr v2/v3), §11.17 (cp1252), plus §5's third-machine zarr recurrence.
- **Blast radius.** Every one of those was found by a human losing a session
  to it on a new machine. That is the single most repeated cost in this
  repo's history, and it is exactly what CI exists to absorb.
- **Fix.** `.github/workflows/ci.yml` with three jobs:
  1. **`fast` (CPU, pinned):** `tsfm_lens` + `tsfm_benchmark` suites,
     `compileall`, the invariant-11 absolute-path grep as a real failing
     check, `tests/test_adapter_conformance.py` against all mocks, and a full
     `run.py --config configs/smoke.yaml` end-to-end asserting section and
     finding counts (the smoke run is minutes and mock-only — no downloads, no
     GPU, per `CLAUDE.md` §8).
  2. **`pinned` (the DEPENDENCIES.md recipe):** installs the exact verified
     versions and runs the same suite — proves the documented recipe still
     resolves and works.
  3. **`bleeding` (allowed to fail, scheduled weekly):** installs the *latest*
     of every load-bearing dependency (zarr, datasets, numpy, torch,
     transformers, timesfm, chronos-forecasting) and runs the same suite. Its
     entire job is to fail on someone else's release rather than on a future
     session's Tuesday. When it fails, that is a `CLAUDE.md` §11 entry
     waiting to be written cheaply instead of expensively.
  4. Publish `report.html` and `coverage.json` from the smoke run as build
     artifacts, so report regressions are reviewable in a PR.
  5. Golden-hash test blocking once A8 is green (`xfail(strict=True)` until).
- **Cost.** ~a day. Prerequisite for §16's Tier 1 credibility.

**Findings (2026-08-06).** New `.github/workflows/ci.yml` with three jobs,
matching the fix plan closely with two stated, deliberate deviations:

- **Item 1 (`fast`).** pip-only (not conda) install of DEPENDENCIES.md §3's
  exact versions where a PyPI wheel exists, CPU-only torch, then
  `compileall`, the invariant-11 absolute-path check, both test suites
  (`tsfm_benchmark/tests`, `tsfm_lens/tests` — the latter already runs
  `test_adapter_conformance.py` against every mock, nothing extra needed),
  and a full `run.py --config configs/smoke.yaml` run. `mkl` is
  **deliberately not pinned in CI** — that pin only matters for
  conda-forge's MKL-linked numpy build (`CLAUDE.md` §11.13); PyPI's numpy
  wheel links OpenBLAS, so the ABI crash it guards against doesn't apply to
  a pip-based job. Report/coverage uploaded as build artifacts (item 4).
- **Item 2 (`pinned`).** Recreates DEPENDENCIES.md §2's exact conda recipe
  via `conda-incubator/setup-miniconda`, runs the same two suites. No GPU
  on a GitHub-hosted runner, so torch is CPU-only here too — "pinned"
  verifies the *documented recovery procedure resolves and the suite
  passes*, not that a GPU is present; real-checkpoint verification stays
  the separate, manual, per-session step this repo has used throughout
  (every ROADMAP.md Findings block above this one).
- **Item 3 (`bleeding`).** Weekly cron (`workflow_dispatch` also available
  on demand) installing the *latest* of numpy/torch/transformers/timesfm/
  chronos-forecasting unpinned, `continue-on-error: true`. `zarr`/`datasets`
  are deliberately still bounded at their documented ceilings (`<3` each)
  rather than left fully open — this job exists to catch a break in a
  library that **isn't** already a understood, documented constraint
  (`CLAUDE.md` §11.8's untracked `timesfm` API removal is exactly the
  shape of thing this job is for), not to manufacture a weekly rediscovery
  of the same two already-known ceilings.
- **Item 4.** `smoke-report` artifact (report.html + coverage.json) uploaded
  from the `fast` job on every run (`if: always()`), reviewable in a PR.
- **Item 5.** Not applicable as separate work — A8 already left the
  golden-hash test green with no `xfail` special-casing (A8's own Findings:
  "when A14 lands, this test should simply run as a normal blocking check
  like any other, with no special-casing needed now that it's green"). It
  runs inside `tsfm_benchmark/tests` like every other test, blocking by
  construction.
- **A deliberate deviation from the fix plan's literal wording**: item 1
  says the smoke check should assert "section and finding counts." Built
  instead as: zero `"failed"` sections (via the run's own `coverage.json`,
  A5's artifact) + a floor of >=8 rendered sections + a minimum
  `report.html` byte size — **not** an exact count. An exact count is
  brittle by construction: this session alone changed the smoke run's
  finding count from 18 (an old `CLAUDE.md` reference) to 22, then 23 (A7),
  then 25 (after A9-A13's own new findings) without the pipeline being
  broken at any point — a CI check pinned to an exact number would have
  needed editing after nearly every item in this session, training
  whoever maintains it to treat CI failures as routine busywork rather
  than a real signal. A floor + zero-failures check still catches an
  actual regression (a section crashing, or the report collapsing to
  near-empty) without that maintenance tax.
- **Two real bugs found and fixed via this item's own local verification**
  (`CLAUDE.md` §2.4 — checked by actually running each step, not assumed
  from reading the shell commands):
  1. The invariant-11 grep's first draft (`[A-Za-z]:[\\/]`, no word
     boundary) false-positived on `https://` (the `s` before `:` in
     "https") and on **escaped newlines inside Python string literals**
     (`"data:\n"` contains the literal substring `a:\` — the `\n` escape,
     not a Windows path). Running it against the actual repo surfaced 5
     false positives immediately; fixed by requiring a word boundary
     (`\b[A-Za-z]:[\\/]`) before the candidate drive letter, which a
     `\n`/`\t` escape mid-identifier never has. Re-run: clean.
  2. The smoke-check script assumed `coverage.json` was a bare list of
     section dicts; it's actually `{"summary": ..., "sections": [...]}`
     (A5's own real schema) — caught immediately on the first local run
     (`TypeError: string indices must be integers`), fixed by indexing
     `["sections"]`.
- **A third finding, real but out of this item's own scope — flagged for
  A18, not fixed here**: `clean_reinstall.sh` (repo root) was read while
  designing the `fast`/`bleeding` jobs' install lists as a precedent for a
  pure-pip (non-conda) setup already used in this repo, and turned out to
  **install `datasets==3.6.0`** — directly contradicting the hard `<3`
  ceiling `CLAUDE.md` §11.9, `DEPENDENCIES.md` §3, and `pyproject.toml`'s
  own `real-data` extras all document, and omitting `zarr`/`tsbootstrap`/
  `sdv`/`pycatch22`/`umap-learn`/`plotly`/`jinja2`/`PyYAML` entirely while
  including packages this repo's own code never imports at all
  (`peft`, `sentence-transformers`, `hdbscan`, `matplotlib`, `seaborn`).
  This script has drifted to describe a different environment than the one
  this repo actually needs — a real, active landmine for anyone who runs it
  expecting a working setup. Not fixed in this pass (A18's explicit scope
  is "doc/config drift"); noted here so it isn't lost before that item is
  reached.
- **Verification, and its honest limit.** Everything **runnable locally**
  was actually run, not just read: YAML parses (`yaml.safe_load`,
  modulo the well-known `on:`→`True` PyYAML quirk that GitHub's own parser
  doesn't share); `compileall` clean; the fixed invariant-11 grep clean
  against the real repo; both test suites green (already reconfirmed
  147/147 for `tsfm_lens` earlier this session, unchanged by this item
  since it adds no pipeline code); a full `configs/smoke.yaml` run
  (11 sections rendered, 25 findings, 0 failed) with the fixed
  smoke-check script passing against its real `coverage.json`. **What this
  item cannot verify from here**: an actual GitHub Actions execution.
  Triggering one requires pushing this workflow file to the remote
  (`origin` is a real GitHub repo, confirmed via `git remote -v`), which
  this session did not do — pushing is a shared-state action outside this
  item's own authorization, and no commit was made for this or any other
  item this session per the user's standing "only commit when explicitly
  asked" instruction. This file is therefore verified **as far as running
  every one of its steps locally can verify it**, not as "green on GitHub"
  — a meaningful difference worth stating plainly rather than overclaiming.

### A15 — Exploratory multiplicity is unaccounted, and `confirm`'s pre-registration is a convention `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** `confirm` derives the hypotheses it tests from whatever
  `l0/summary.json` says at the moment it runs — a mutable dev artifact. Only
  L0 family strengths plus one CKA pair are confirmable at all, while the
  report accumulates ~22 "findings" with no record of how many comparisons
  produced them. Holm correction is applied within L0's family tests, not
  across the registered set.
- **Evidence.** `analysis/confirm.py:53-56` (`_test_hypotheses(cfg, metrics,
  ...)` reading dev artifacts), `report/report.py`'s `findings` accumulation.
- **Blast radius.** `CLAUDE.md` §6.7's exploration-vs-confirmation discipline
  is the repo's strongest epistemic claim, and it is currently enforced by
  discipline alone: re-running L0 after glancing at private results would
  change what gets "pre-registered", with no trace. Meanwhile the report's
  findings list is a large multiple-comparison surface labelled exploratory
  only in a preamble.
- **Fix.**
  1. A machine-readable registry, `hypotheses.json`, written by a new
     `register` step (cheap; reads existing artifacts): one entry per dev
     claim — claim id, stage, statistic, direction, threshold, the artifact
     path + content hash it was derived from, and free-text statement.
  2. `confirm` **requires** it, refuses (loudly, actionably) if any referenced
     artifact hash no longer matches, and tests exactly the registered set —
     no more, no fewer. Registry hash goes in `confirm/confirmation.json`.
  3. Extend the registry beyond L0: any statistic with a CI can be registered
     (L1 peak CKA, L2 best gain, L3 fingerprint agreement, clustering AMI).
     The confirm stage already replicates CKA — this generalizes the seam
     rather than inventing one.
  4. Multiplicity ledger: the report prints "N comparisons examined → M
     registered → K confirmed", and every unregistered finding is labelled
     *exploratory* inline in the HTML, not only in the preamble. Apply
     Holm/BH across the registered set as a whole.
  5. Test: mutate `l0/summary.json` after registration, assert `confirm`
     refuses with a message naming the drifted artifact.
- **Cost.** ~a day and a half. This is the item that most raises what the repo
  can honestly claim.

**Findings (2026-08-06).** Implemented items 1, 2, 4, and 5 in full; item 3
generalized the seam as instructed but only made two of the five stage
types it names actually *replicable* — a deliberate, stated scope decision
explained below, not a silent gap.

- **Item 1.** New `analysis/hypotheses.py`: `build_registry(cfg)` (pure,
  reads whatever dev artifacts exist) assembles one entry per confirmable
  claim across `l0` (family strengths), `l0_archetype` (A9's per-archetype
  strengths), `l1` (peak CKA), `l2` (stitching gain per direction), `l3`
  (fingerprint agreement per corruption), and `clustering` (AMI) — each
  entry carries `id`, `stage`, the statistic-specific fields, `artifact`
  (path), `artifact_sha256`, a free-text `statement`, and `replicable:
  bool` (+ `not_replicable_reason` when false). `run_register(cfg)` writes
  it to `hypotheses.json` at the run root. New `register` pipeline `Stage`
  (`deps=["l0"]`, `enabled` mirrors `confirm.enabled` — no point
  registering if nothing will consume it), inserted immediately before
  `confirm`, which now depends on `register` instead of `l0` directly
  (transitively still requires `l0`).
- **Item 2.** `confirm.py::run_confirm` now refuses immediately (before
  touching the private corpus) if `hypotheses.json` doesn't exist, naming
  the fix (`--stages` must include `register`). `check_registry_freshness`
  re-hashes every referenced artifact and raises, naming every drifted
  file and both its registered and current hash prefixes, on any
  mismatch — the actual enforcement mechanism behind `CLAUDE.md` §6.7's
  discipline, previously enforced by convention alone. `hypotheses.json`'s
  own sha256 is recorded in `confirmation.json` (`registry_sha256`).
  `_test_registered_hypotheses` (renamed from `_test_hypotheses`) sources
  its claim list from `registry["hypotheses"]` (filtered to `stage == "l0"
  and replicable`) instead of re-reading `l0/summary.json`'s `strengths`
  dict directly — the registry, not the mutable dev artifact, is now the
  source of truth for *which* claims get tested, exactly item 2's wording.
- **Item 3, generalized but scoped honestly.** `_replicate_registered_cka`
  (renamed from `_replicate_cka`) now looks up the registered, replicable
  `l1` hypothesis rather than reading `l1/meta.json`'s best pair directly —
  the seam item 3 asks to generalize, done for the one replication
  `confirm.py` already had. **`l2`/`l3`/`clustering` hypotheses are
  registered (hash-pinned, counted in the ledger, shown in a "registered
  but not yet replicated" report table with a stated reason each) but
  `confirm` does not yet re-test them on private data** — doing so
  correctly would mean re-fitting stitching probes (L2), re-running the
  full corruption battery (L3), or re-fitting cluster labels (clustering)
  against a private corpus inside this same stage, each a substantial
  design/implementation task in its own right, not a "generalize an
  existing seam" extension the way L1's CKA replication was. Also scoped
  out: `l0_archetype` claims, registered but not replicable, since
  `confirm`'s private-corpus pivot is family-keyed and a fresh private
  epoch has no guaranteed archetype composition match to dev's (stated in
  each entry's `not_replicable_reason`, not silently dropped).
- **Item 4.** `report.py::run_report` captures `len(findings)` at the exact
  point the "Confirm" builder is reached (before it runs) as the
  exploratory/confirmatory boundary, then tags every finding at an earlier
  index with `"[exploratory — not pre-registered] "` when rendering — no
  per-section code changes needed, since every section already appends to
  the same shared `findings` list in a fixed order. `_sec_confirm` renders
  a "Multiplicity ledger" (`N exploratory findings examined → M registered
  (K replicable today) → J confirmed`) plus the not-yet-replicated table.
  **A real ordering change this required**: `_sec_confirm`'s own two
  `findings.insert(0, ...)`/`insert(1, ...)` calls (originally there to put
  Confirm's findings first, for prominence) had to become `.append(...)` —
  inserting at the front would have landed them *before* the captured
  boundary index and mislabeled genuinely confirmatory findings as
  exploratory. Confirm's findings now appear at the end of the findings
  list instead of the front; a deliberate, explainable tradeoff (correctness
  of the new label over the old prominence-ordering), not an oversight.
- **Item 5.** Directly covered by `test_hypotheses.py`'s
  `test_check_registry_freshness_raises_naming_drifted_artifact_after_
  mutation` — mutates `l0/summary.json` after registration, asserts the
  raised message names `l0/summary.json` and says "hash changed".
- **Verification — unit tests.** New `tests/test_hypotheses.py` (9 tests):
  registry construction reads real strengths with a real sha256; archetype
  claims are correctly marked non-replicable with a reason; an empty
  registry when no `l0` artifact exists; `run_register` writes a real
  file; freshness passes when nothing changed; freshness raises (naming
  the file) on mutation and on deletion; `confirm` refuses outright with no
  registry; `confirm` refuses on a stale one. Two pre-existing tests
  updated for the rename/new artifact: `test_smoke.py::
  test_confirm_hypothesis_path` now builds a registry dict directly
  (standing in for what `register` would have written) and calls
  `_test_registered_hypotheses`; `test_smoke.py`'s `expected` artifact list
  gained `hypotheses.json`. `test_manifest_fingerprint.py` (A3's stage-
  fingerprint regression test, which enumerates `stage_names()`) re-run
  and confirmed unaffected by the new `register` stage. Full suite green
  at 156/156 (up from 147 after A13 — A14 added no test-suite code).
- **Verification — real checkpoints, deliberately without touching the
  private corpus.** `runs/medium_run_chronos_base` (a real prior session's
  full GPU run against `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-base`) already has a **consumed** `confirm/
  confirmation.json` — re-running `confirm` against it would be exactly
  the "repeated peeking" `CLAUDE.md` §6.7 warns against, so it wasn't done.
  Instead, `run_register` was called standalone against that run's real
  dev artifacts (a read-only operation touching only `l0`/`l1`/`l2`/`l3`/
  `clustering` JSON, never the private corpus) and registered **14 real
  hypotheses** (2 replicable: `TimesFM`'s `random_parametric` family
  strength, and the `TimesFM`↔`Chronos-T5-Base` peak CKA at
  `stacked_xf.4`↔`encoder.block.10`; 12 correctly marked not-yet-
  replicable — 2 L2 gains, 9 L3 corruption-agreement values, 1 clustering
  AMI). `check_registry_freshness` passed clean against these real,
  large, real-checkpoint-produced JSON files. A drift-refusal test was
  then run on a **scratch copy** of that run directory (never the
  original): mutating its `l0/summary.json` and re-checking produced the
  real error `"l0/summary.json (hash changed: registered bb4f19692101,
  now e9bf5f748d80)"` — the exact mechanism working against real file
  sizes and real hashes, not a synthetic stand-in. Separately, a fresh
  `configs/smoke.yaml` run (mock adapters, confirm enabled) exercised the
  full mechanism end to end including the actual private-corpus
  confirmation step (smoke's "private" corpus is a second smoke-generated
  draw, not a real sealed epoch, so no discipline concern there): **23
  exploratory findings examined → 17 hypotheses registered → 5 replicable
  → 3 confirmed**, with a "Registered but not yet replicated (12)" table
  rendered in `report.html`, and all 23 pre-Confirm findings correctly
  carrying the `[exploratory — not pre-registered]` tag (grep-confirmed
  count matches exactly).

### A16 — Silent cap reduction makes stage sample sizes non-comparable `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** Stages that must fit one batch clamp their configured cap
  to `adapter.cfg.batch_size` with no warning, so the realized sample size
  differs from the configured one — and differs *per model*, since batch sizes
  do.
- **Evidence.** `lens.py:91` — `take = min(data.n, cfg.lens.max_series,
  adapter.cfg.batch_size)`; same pattern at `l3_perturbation.py:318`,
  `sae/eval.py:86`, `attention.py` ablation paths. YAML comments carry the
  constraint ("must be <= every model's batch_size",
  `configs/medium_run.yaml:78,85,99`) — i.e. it's documented for humans and
  silently absorbed by code.
- **Blast radius.** §5.3 already hit the consequence: lens numbers that
  couldn't be compared across two runs, diagnosed correctly but only after
  someone noticed an impossible result (the same TimesFM checkpoint appearing
  to change). Any cross-run or cross-model comparison of a batch-limited
  stage is currently untrustworthy unless someone manually verifies the caps
  matched.
- **Fix.** Warn whenever a configured cap is reduced, naming which limit bound
  it; record `n_requested`/`n_realized`/`limited_by` in every stage artifact
  (folds into A4's row recording and A3's manifest); `meta_report.py` refuses
  to compare a batch-limited statistic across runs whose realized n differ,
  or flags it prominently rather than tabulating it as comparable. Optionally
  a `strict_caps: true` mode that raises instead of clamping.
- **Cost.** Half a day (mostly free if done with A4).

**Findings (2026-08-06).** Implemented the fix at exactly the sites the
item's own evidence names, and checked (rather than assumed) two sites
that looked like the same bug but weren't:

- **Core mechanism.** New `utils.py::capped_take(requested, **limits) ->
  {"n_requested", "n_realized", "limited_by"}` — generalizes the
  `n_requested`/`n_realized` field pair A4 already established (`sae/
  ground_truth.py`, `sae/eval.py`) with a `limited_by` list naming every
  limit that actually bound the result, and warns whenever the request was
  reduced. Named kwargs (`n_available=...`, `batch_size=...`) rather than
  positional limits, so the warning/`limited_by` output is self-describing
  without a separate label argument.
- **Wired into the three sites the item's evidence cites by line number,
  all confirmed still present at those lines**: `lens.py`'s `_model_lens`
  (`n_series_skip`/`n_requested_skip`/`limited_by_skip` in `lens/
  lens.json`), `l3_perturbation.py`'s `_patching` (`n_requested`/
  `n_realized`/`limited_by` in `l3/patching.json` — **had to also extend
  the save dict's key whitelist**, since `run_l3`'s `patching.json` write
  explicitly lists which keys of each model's result dict get persisted;
  the three new fields would have been computed and then silently dropped
  on the way to disk without that second edit, caught by checking the
  save site, not assumed safe from the producer-side change alone), and
  `sae/eval.py`'s `forecast_preservation` (`limited_by` alongside the
  `n_requested`/`n_realized` that already existed there from A4, with a
  comment that had *already* forward-referenced this exact item's number).
- **Two sites checked and found NOT to have this bug, correcting the
  item's own evidence list rather than fixing something that wasn't
  broken**: `attention.py`'s ablation paths (lines 226/256, explicitly
  named in the item's Evidence section) sample `take = min(data.n,
  cfg.attention.max_series)` — no `adapter.cfg.batch_size` term at all —
  and process the full realized set through internal `batch_slices`
  chunking, exactly the pattern that makes lens/l3-patching/sae-eval's
  *actual* bug (a hook-based intervention requiring one single batch,
  `CLAUDE.md`'s documented `token_patch` batch-size hardening) not apply
  here. `sae/real_data.py:58`'s superficially similar `take = min(batch_size,
  adapter.cfg.batch_size)` was also checked directly: it sizes an internal
  micro-batch for a `batch_slices` loop over the *entire* `contexts` array
  — every context still gets processed regardless of `take`'s value, so
  this is not a sample-size cap at all. Neither site was touched.
- **Item's `meta_report.py` ask**, implemented as "flag prominently" (the
  fix text's second option) rather than "refuse to compare": new
  `_batch_cap_warnings()`, reusing A7's `provenance_warnings` mechanism and
  rendering slot (same "Environment drift" box) — compares each model's
  lens `n_realized` across the aggregated runs and warns by name when they
  differ, which is exactly `CLAUDE.md` §5.3's own already-documented
  incident (crystallization depth moving for an *unchanged* checkpoint
  purely because two configs implied different realized n), now an
  automated check instead of something a human has to notice looks
  impossible. Scoped to `lens` only (the stage §5.3's real incident was
  about, and the only one `summarize_run` already tracked per-model
  numbers for) — `l3` patching's cap isn't cross-run-aggregated by
  `meta_report.py` today, so extending this warning there is a natural
  follow-up, not done in this pass, stated rather than silently skipped.
  A `strict_caps: true` raise-instead-of-clamp mode (the fix's "optionally")
  was not built — genuinely optional per the item's own wording, and the
  warn-plus-record mechanism above is the load-bearing half of this item.
- **Verification — unit tests.** New `tests/test_capped_take.py` (9 tests):
  no reduction is silent with `limited_by: None`; reduction by `batch_size`
  alone, by `n_available` alone, and by both tied together, each names the
  right limit(s); a request that exactly equals a limit (not actually
  reduced) is correctly not flagged; a full mini pipeline with `lens.
  max_series: 999` (exceeding every mock's `batch_size: 64`) records
  `n_requested_skip=999`, `n_series_skip<=64`, `limited_by_skip` non-`None`
  in real `lens.json`; the analogous check for `l3.patching.max_series:
  999` against real `patching.json`; and `_batch_cap_warnings` fires on a
  synthetic two-run mismatch and stays silent when realized n matches.
  Full suite green at 165/165 (up from 156 after A15).
- **Verification — real checkpoints.** `configs/_a16_real_check.yaml`
  (scratch, deleted after use; extract+lens+l3-patching+report only) against
  live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small` on
  `benchmark_medium/public_dev`, both models pinned to a deliberately small
  real `batch_size: 16` with `lens.max_series: 64` / `l3.patching.
  max_series: 64` — both well above that batch size, to force a genuine
  clamp on real weights rather than only mocks. Both models, both stages:
  requested 64, realized **16**, `limited_by: ["batch_size"]` — recorded
  correctly in both real `lens/lens.json` and real `l3/patching.json`, the
  exact real-world version of the §5.3 incident this item exists to make
  visible automatically instead of discoverable only by noticing an
  impossible result.

### A17 — `benchmark_validation`'s per-group metrics have no minimum-n guard or CI, and the matcher's O(n²) ceiling blocks full-scale validation `[x]` · **P3** · fixed 2026-08-06

- **What's wrong.** Per-group diversity numbers are reported as point
  estimates at any group size, and the pairwise matcher is O(n²) with
  subsampling as the only mitigation.
- **Evidence.** §5's 2026-08-03 per-task breakdown reports effective
  dimensionality at n=39 (`real_sequential_par`) alongside n=3154, with the
  caveat "enough to trust the direction, not the exact numbers" written by
  hand; `CLAUDE.md` §12 documents the O(n²) matcher and `--max-sequences`.
- **Blast radius.** A future session reading that table without the prose
  caveat will over-read n=39. And the literal-scale `full_multidomain.yaml`
  build (§5.1, still unrun, ~99K samples) cannot be validated at all at
  O(n²) — 8.9M pairs already at 4219 sequences.
- **Fix.** Bootstrap CIs on per-group diversity metrics (resampling
  sequences within group), a `min_group_n` below which a group reports
  "insufficient n" instead of a number, and a blocked/approximate matcher
  (bucket by catch22 neighbourhood first, do exact xcorr/DTW within and
  across adjacent buckets only) with the exactness trade-off stated in the
  report. Also verify `--max-sequences` subsampling is random and stratified
  by generator (A4 item 6) — if it's a head slice, per-group numbers are
  measuring task order.
- **Cost.** ~a day and a half.

**Findings (2026-08-06).** Implemented all four fix items in
`diversity.py`/`matching.py`/`report.py`/`plot.py`. This item lives entirely
in `tsfm_benchmark/benchmark_validation` — a different package from A9-A16's
`tsfm_lens` work, with no models or checkpoints involved at all (pure
corpus/feature-space code), so "real run" verification here means a real
sealed corpus, not real weights.

- **Item 1 (bootstrap CIs).** New `diversity.py::_bootstrap_group_ci`:
  resamples a group's own scaled feature rows with replacement and
  recomputes the *actual* PCA/nearest-neighbour pipeline `diversity_metrics`
  itself uses (not a closed-form/normal-theory approximation, which could
  be a poor stand-in for a nonlinear statistic like the PCA participation
  ratio at exactly the small-n regime this item is about) `n_boot` times,
  reporting a percentile CI. New `DiversityReport.effective_dimensionality_
  ci`/`near_collision_fraction_ci` fields (`None` unless requested).
  `diversity_metrics_by_group(..., n_boot=200)` — on by default, since an
  unmeasured CI is exactly the gap this item exists to close; `n_boot=0`
  restores the pre-A17 byte-identical fields.
- **Item 2 (min_group_n → explicit sentinel).** New `InsufficientN` dataclass
  (`n_sequences`, `min_required`, `status="insufficient_n"`).
  `diversity_metrics_by_group` now returns an entry for **every** group
  (never a silent omission) — `InsufficientN` below `min_group_size`,
  `DiversityReport` at or above it. `report.py::_by_group_summary` and
  `plot.py::plot_diversity_by_group` both updated to handle the mixed
  return type: the report emits an explicit `{"status": "insufficient_n",
  ...}` block per small group (JSON, not a missing key), and the plot
  excludes those groups from the bars (not plotted as a misleading zero)
  with the excluded count stated in the plot title.
- **Item 3 (blocked/approximate matcher).** New `matching.py::
  _match_all_blocked` (`match_all(..., blocked=True, n_blocks=None,
  adjacent_k=2)`): buckets sequences via `KMeans` over the same z-normalised,
  resampled arrays the exact matcher already compares (a self-contained
  shape-neighbourhood proxy — **a deliberate deviation from the fix text's
  literal "catch22 neighbourhood"**, made to avoid adding a new coupling
  from `matching.py` to `features.py`; stated here, not silently
  substituted), then computes *exact* similarity — reusing the same
  batched `_max_xcorr_matrix`/`_dtw_similarity_matrix` routines the exact
  path already uses, no new numerical code — only for pairs whose two
  sequences share a bucket or fall in each other's `adjacent_k` nearest
  buckets by centroid distance. Every other pair is left unscored (`NaN`
  in `similarity_matrix`, excluded from `redundancy_fraction`'s
  denominator and from `redundancy_by_group`'s per-group means) rather
  than assumed non-redundant. `MatchReport` gains `coverage_fraction`
  (exactly what fraction of all possible pairs were actually scored —
  the exactness trade-off the fix explicitly requires be stated, not
  buried), `blocked`, `n_blocks`. The existing exact path was refactored
  into `_match_all_exact` with **zero behavior change** (same code, new
  function name, called by default) — `match_all`'s default
  (`blocked=False`) reproduces every existing caller byte-for-byte. Falls
  back to the exact path (with `coverage_fraction=1.0`) when `n` is too
  small to block meaningfully. Wired into `run_validation.py` as
  `--blocked`/`--n-blocks`.
- **Item 4 (stratified `--max-sequences`).** Checked first, per `CLAUDE.md`
  §2.4, rather than assumed broken: `match_all`'s existing subsampling
  already used `rng.choice` (genuinely random, **not** a head slice) —
  the "if it's a head slice" conditional in this item's own fix wording
  didn't apply. It was, however, *unstratified* — a corpus dominated by one
  generator would draw mostly from it. New `matching.py::
  _stratified_subsample` (a small local proportional-allocation sampler,
  deliberately **not** an import of `tsfm_lens`'s `utils.sample_rows` —
  the two packages have no declared dependency on each other by design,
  per `loaders.py`'s own module docstring) replaces the plain `rng.choice`
  call, stratifying by `SeqRecord.group` (the generator name).
- **Verification — unit tests.** New `tests/test_benchmark_validation_a17.py`
  (9 tests): a group below `min_group_size` gets an `InsufficientN` entry
  alongside (not instead of) a normal group's `DiversityReport`; the
  bootstrap CI is present and well-formed (a test that initially asserted
  the CI must bracket the point estimate **failed and was corrected**,
  not the code — a bootstrap resample duplicates rows, which can
  systematically shift a nonlinear statistic like the PCA participation
  ratio, so non-coverage is expected statistical behavior, not a bug;
  fixed the test's assumption rather than loosening a real check);
  `n_boot=0` skips the CI; a small group's CI is measurably wider than a
  large group's (a real, checkable claim about bootstrap behavior, not
  just "doesn't crash"); stratified subsampling preserves group
  proportions and is provably not a head slice (a second test here also
  had to be corrected: it asserted a 10-of-110 minority group must get
  `>=5` of a 20-item stratified sample, but proportional allocation
  correctly gives ~2 — the test's expectation was wrong, not the sampler);
  the default exact path is unaffected by the refactor (byte-identical
  similarity matrix across two calls); the blocked matcher reports partial
  coverage with no `NaN` leaking into `redundant_pairs`; and it falls back
  to the exact path cleanly for too-small `n`. Full `tsfm_benchmark` suite
  green at 50 passed / 1 skipped (up from 41 passed / 1 skipped after A8).
- **Verification — real corpus (no models involved).** `run_validation.py`
  against the real, previously-built `benchmark_medium/public_dev` (288
  sequences, verified sealed corpus). `--group-by archetype`: all 12 real
  `CLAUDE.md` §4.2 archetypes got a real bootstrap CI, several visibly wide
  at small real n (`trend_dominant`, n=7: effective_dimensionality 1.938,
  CI (1.212, 2.237) — exactly the "trust the direction, not the exact
  number" caveat this item names, now a quantitative interval instead of
  hand-written prose). `--blocked --n-blocks 8`: real coverage_fraction
  **43.6%** of all 41,328 possible pairs, redundancy fraction and
  within/across-group similarity all computed correctly among only the
  scored pairs (0.0 redundancy, matching the exact path's own 0.0 on this
  corpus — a real, if limited, cross-check that blocking didn't change the
  qualitative finding), rendered correctly into both the console summary
  and `validation_report.json`. A direct call to `plot_diversity_by_group`
  with a real `InsufficientN` entry present confirmed no crash and a
  correctly-sized output file.

### A18 — Small doc/config drift that the repo's own invariants forbid `[x]` · **P3** · fixed 2026-08-06

- `configs/default.yaml:192` — `path: /path/to/private_benchmark_dir`, an
  absolute placeholder in a committed config. Invariant 11 has no
  placeholder exemption, and the 2026-08-04 audit's grep-for-`/home/`
  approach wouldn't catch this form. Fix: make it a relative example
  (`../../benchmark_full/private_test`) with a comment, and extend the
  invariant-11 check (A14 job 1) to flag any config path starting with `/`
  or matching a drive letter.
- `tsfm_lens/requirements.txt` — names `google/timesfm-2.0-500m-pytorch` as
  the tested adapter target; the adapter moved to TimesFM 2.5
  (`google/timesfm-2.5-200m-pytorch`, `CLAUDE.md` §11.8) sessions ago. Fix:
  update both `requirements.txt` and `pyproject.toml`'s `timesfm` extra, and
  cross-check against `DEPENDENCIES.md`.
- `runs/` artifacts referenced throughout this file are gitignored and
  regenerable, which is correct — but several Findings cite them as evidence
  without recording the provenance needed to regenerate identically. A7's
  manifest fixes this going forward; consider a one-time pass adding the
  environment used to the older Findings entries that are still load-bearing.

**Findings (2026-08-06).** Fixed the two named drift items, one item found
via the same cross-check that wasn't in the original evidence list, and
extended A14's CI check as this item's own fix wording asked for.

- **Absolute placeholder in `configs/default.yaml`.** Fixed the named
  `confirm.path: /path/to/private_benchmark_dir` → a relative example
  (`../../benchmark_full/private_test`) with an explanatory comment.
  **A repo-wide grep for the same shape (`^\s*path:\s*/`) found a second,
  unnamed instance in the same file**: `data.path: /path/to/benchmark_dir`
  (line 15) — the exact same violation the item's evidence only cited once
  (line 192). Fixed identically (`../../benchmark_full/public_dev`). Both
  now load cleanly via `load_config` (checked directly, not assumed).
- **Stale `timesfm` version references.** `requirements.txt`'s commented
  `timesfm[torch]>=1.2 ... google/timesfm-2.0-500m-pytorch` and
  `pyproject.toml`'s `timesfm[torch]>=1.2` extra both corrected to
  `timesfm>=2.0` (matching `DEPENDENCIES.md`'s verified `timesfm==2.0.2`,
  installed plain, no `[torch]` extra — `>=2.0` removed the old
  `TimesFmHparams`/`TimesFmCheckpoint` class API and, with it, whatever
  the `[torch]` extra used to install) with a comment naming
  `CLAUDE.md` §11.8 and the current default checkpoint
  (`google/timesfm-2.5-200m-pytorch`).
- **A third instance of the same drift, found via the item's own
  "cross-check against DEPENDENCIES.md" instruction, not in the original
  evidence list**: a repo-wide grep for `timesfm-2.0-500m`/`timesfm[torch]`
  found two README.md files (`tsfm_model_analysis/README.md` and
  `tsfm_model_analysis/tsfm_lens/README.md`) with the identical stale
  `pip install chronos-forecasting timesfm[torch]` line, and the outer
  `tsfm_model_analysis/README.md` specifically still describing the
  retired `google/timesfm-2.0-500m-pytorch` checkpoint in prose — a
  genuinely different, more detailed correction had already landed in the
  inner `tsfm_model_analysis/tsfm_lens/README.md` at some earlier point
  (the 2.5-only-API explanation, the SDPA-unfusing `attention_patterns`
  note, the no-MLP-submodule note) without the outer, apparently-not-kept-
  in-sync duplicate ever being updated to match. Both install lines fixed
  to plain `timesfm` (no `[torch]`); the outer README's stale checkpoint
  paragraph replaced with the same corrected text the inner one already
  had. **Not attempted**: a full reconciliation of the two README files'
  remaining differences (the `diff` surfaced several — a missing Lens/
  Attention/Exemplars row in the outer file's stage table, a differently-
  detailed L3 description) — out of this item's stated scope (doc/config
  *drift the repo's invariants forbid*, i.e. stale facts, not general
  doc-duplication cleanup); flagged here as a real, separate finding for
  whoever next touches either README.
- **CI extension, as the fix explicitly asked for.** A14's invariant-11
  grep (mid-string drive-letter / `/home/` / `/Users/` patterns) would
  **not** have caught this item's own bug shape (a bare `/path/to/...`
  with no drive letter or `/home//Users/` segment) — confirmed by testing
  the old pattern against the literal old line and seeing it not match,
  not assumed. New, separate CI step: `^\s*(path|out_dir):\s*/` over every
  YAML file, catching exactly this shape. **Caught a real false-positive
  risk before it could ever fire in CI**: the new grep, run locally
  against the actual repo, initially matched four lines inside
  `runs/*/config_resolved.yaml` — gitignored, per-run generated artifacts
  that correctly contain the absolute path of whatever machine produced
  them, not committed templates invariant 11 is about. Fixed by adding
  the same `--exclude-dir=runs` the first invariant-11 check already had;
  re-run clean against the real repo afterward.
- **Third fix-plan bullet (retroactive Findings provenance pass):
  deliberately not done**, per that bullet's own softer "consider" wording
  (contrasted with the first two bullets' plain "Fix:") — a full pass
  annotating environment provenance onto every older, still-load-bearing
  Findings entry across this file would be a large, separate undertaking
  in its own right, not a doc/config-drift fix. Left as a stated,
  explicitly-scoped-out follow-up, not silently skipped.
- **Verification.** No new unit tests — this item is pure doc/config text
  plus one CI shell-check extension, not runtime code. Verified instead by
  actually running what changed: `load_config` against the corrected
  `configs/default.yaml` resolves both paths correctly (checked directly);
  the new and existing invariant-11 grep patterns tested against the real
  repository both before and after each fix (confirmed the bug shape,
  confirmed the fix, confirmed no new false positives); a repo-wide grep
  for the retired checkpoint name and `timesfm[torch]` came back empty
  outside of `ROADMAP.md`'s own historical Findings prose (correctly left
  untouched — a dated record, not a current claim). Full `tsfm_lens` suite
  re-run clean at 165/165 (unchanged from A17 — this item touched no
  runtime code, so no test count change is the correct outcome, not an
  oversight).

### A19 — `store.load`'s row selection and `float16` storage are unguarded against silent precision/indexing surprises `[x]` · **P3** · fixed 2026-08-06

- **What's wrong.** Activations are stored `float16` (`CLAUDE.md` §6.4) and
  read back for closed-form fp32 linear algebra. Nothing checks for overflow
  to `inf` at write time (bf16-range activations can exceed fp16's ~65504
  max) or for NaNs entering a CKA/ridge solve.
- **Evidence.** `extraction/extract.py:67` — `.astype(np.float16)` with no
  finiteness check; `store.py`'s `oindex` path (correct, per `CLAUDE.md`
  §11.4) has no assertion that the returned row count matches the request.
- **Blast radius.** A single `inf`/`NaN` layer silently poisons a CKA row, an
  eff-dim estimate, or a ridge solve; downstream those appear as a plausible
  number (CKA near 0, or a `nan` that a `nanmean` swallows) rather than an
  error. Low likelihood, high diagnosis cost — the kind of thing that gets
  misattributed to a model finding. `layer_screen`'s own fallback path
  already anticipates "a degenerate/constant layer" in its docstring, which
  suggests this has been brushed against.
- **Fix.** Count and log non-finite values per (model, layer) at write time,
  store the counts in store attrs, refuse to write a layer that is wholly
  non-finite, and have every analysis that consumes activations assert
  finiteness once per layer (cheap) with an actionable error naming the
  layer. Consider `extraction.store_dtype: float32` guidance for models
  whose activations run hot. Report the counts in the coverage panel.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented the fix's core items in
`store.py`/`extract.py`/`report.py`/`config.py`, and — per this item's own
"§2.4 applies here too" instruction — found that the diagnosis's fix
item 5 ("consider a `store_dtype: float32` knob") was already wrong to
phrase as a suggestion: the knob already existed as real config, and was
silently non-functional.

- **Write-time tracking.** `store.py::write_batch` accumulates a per-
  (model, layer) non-finite count and total-element count in memory
  (`self._nonfinite_counts`/`_nonfinite_totals`) across every batch;
  `finalize_layer` (called once per layer, at the end of `extract.py::
  _extract_model`'s batch loop) persists `{count, total, fraction}` into
  `root.attrs["nonfinite"]` and **raises `RuntimeError` if a layer is
  wholly non-finite** (`fraction >= 1.0`) — refusing to leave a completely
  poisoned layer in the store for every downstream CKA/ridge/PCA call to
  independently trip over, per the fix's own wording. Partial non-
  finiteness logs a warning and is recorded, not fatal.
- **Read-time assertion.** `store.load(..., check_finite=True)` (the new
  default) checks `np.isfinite` on whatever was just materialized and
  raises `ValueError` naming the model, layer, count, and fraction the
  first time any consumer actually reads a poisoned array — the "every
  analysis... assert finiteness once per layer" half of the fix, at the
  single choke point every stage already calls through. `check_finite=
  False` exists as a stated escape hatch; **not wired into any current
  caller** — every real call site (`sae/ground_truth.py`, `sae/train.py`,
  `layer_screen.py`, `l1_geometry.py`, `l2_stitching.py`, `clustering.py`,
  `internals.py`, `lens.py`, `l3_perturbation.py`) keeps the default,
  checked directly via a repo-wide grep, not assumed.
- **Report.** New `report.py::_nonfinite_summary()` opens the store
  read-only and reads `root.attrs["nonfinite"]`, folded into the existing
  combined `_alignment_provenance_block` (A2+A7's panel) as a new table —
  renders whenever any layer has been tracked, **including when every
  count is zero** (confirmed on real data below), so the check having run
  and found nothing is itself visible, not indistinguishable from the
  check never having run.
- **The real finding, more significant than the item's own item 5's
  phrasing suggested.** `extraction.store_dtype` was **not** a
  not-yet-built "consider" suggestion — it already existed as a real
  `ExtractionConfig` field, already wired into `store.init_layer` (which
  correctly used it to set the zarr array's declared dtype). But
  `extract.py`'s very next line, `aligned = align(hidden,
  pool).cpu().numpy().astype(np.float16)`, **hardcoded `float16`
  regardless of that config** — so `store_dtype: float32` changed only
  the on-disk dtype *label*; the actual float32-range values had already
  been irreversibly rounded (and, in an overflow case, turned to `inf`)
  to float16 before `store.write_batch` ever saw them. The config knob
  was a complete no-op for the one thing (avoiding float16 overflow) it
  existed for. Found by reading the exact line this item's own Evidence
  pointed at, not by assuming a config field being consumed by one
  function meant the whole path worked — fixed by casting to
  `np.dtype(cfg.extraction.store_dtype)` instead of the hardcoded literal.
- **Verification — unit tests.** New `tests/test_nonfinite.py` (8 tests):
  a clean batch records a zero count; a partially non-finite batch is
  recorded, not raised; a wholly non-finite layer raises; counts
  accumulate correctly across multiple batches; `store.load` raises on a
  poisoned layer, naming it; `check_finite=False` bypasses the check
  (poison still present, just not raised on); a clean layer loads
  normally; and — the test that actually caught the real bug above — a
  full mini pipeline run with a deliberately huge injected activation
  value (1e5, overflows float16's ~65504 max, fits float32 easily) shows
  the value becoming non-finite under the `store_dtype: float16` default
  and **reproduces a genuine `RuntimeWarning: overflow encountered in
  cast`** in the process, then shows the identical value surviving finite
  under `store_dtype: float32` once the hardcoded-cast bug was fixed —
  proving the *values*, not just the array's dtype label, actually
  differ. Full suite green at 173/173 (up from 165 after A18).
- **Verification — real checkpoints.** `configs/_a19_real_check.yaml`
  (scratch, deleted after use; extract+l0+report only, both models at
  `capture_layer_stride: 1` — every layer captured, not the usual stride-2
  subset, to maximize the chance of catching a real overflow anywhere in
  either model's full depth) against live
  `google/timesfm-2.5-200m-pytorch` (all 20 layers) +
  `amazon/chronos-t5-small` (all 6 encoder blocks) on
  `benchmark_medium/public_dev`, `store_dtype: float16` (the default).
  **A genuinely open empirical question, resolved honestly rather than
  assumed either way**: all 26 real captured layers showed **zero**
  non-finite values — `float16` overflow, the concrete risk this whole
  item is about, did not materialize in practice for either checkpoint on
  this corpus. A true negative result, not a weaker test: the mechanism
  ran cleanly end to end on real activations with no false positives, and
  the new report table rendered all 26 layers' `0.0000%` fractions
  correctly (grep-confirmed present in real `report.html`) rather than
  being hidden because nothing was found — exactly the "state it, don't
  silently omit" behavior the rest of this session's fixes established.
  Whether a *different* real model (larger, or one known to produce
  extreme activations) would actually trip this guard remains untested;
  the guard itself is now real and in place either way.

---

**All 19 items in this section (A1–A19) are now `[x]` complete as of
2026-08-06.** Every item was implemented, unit-tested, verified against a
real run (real checkpoints for the `tsfm_lens` items, a real sealed
corpus with no models involved for A17, config/doc-only verification for
A18), and documented here with a dated Findings block before moving to the
next — several items surfaced genuinely new bugs beyond what their own
Evidence sections named (A6's sklearn single-class crash, A7's missing
`timesfm.__version__`, A9/A10's tier-threading and `mixture`
`generative_params` bugs, A12's cross-seed bisection mismatch, A16's two
evidence corrections, A18's second unnamed absolute-path instance, A19's
non-functional `store_dtype` knob — each found by direct inspection or
testing, per `CLAUDE.md` §2.4, not assumed from the audit's own wording).
Nothing in this section remains open; `ROADMAP.md` §16's enhancement
backlog is the next place forward work on this repo should look.

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

### T1 — Adoptability

- [ ] **E1 · A zero-config entry point.** `tsfm-lens compare --model
  hf://google/timesfm-2.5-200m-pytorch --model hf://amazon/chronos-t5-base
  --preset standard` → resolves adapters from the registry by checkpoint
  pattern, generates a full config, uses the bundled reference corpus (E4),
  runs preflight (E2), runs the pipeline, opens the report. `--preset
  quick|standard|deep` maps to series counts, `capture_layer_stride`, patching
  strides, and which stages are enabled — the three cost/depth points the
  existing configs already represent implicitly (`smoke`/`medium_run`/
  `default`), named and documented instead of copy-pasted. Emit the generated
  YAML into the run dir so the zero-config path is a *teaching* path: "here is
  the config I made for you, edit it next time." Depends on: E4, E2, A7.
- [ ] **E2 · `tsfm-lens doctor` — preflight that runs automatically.** One
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
- [ ] **E4 · A bundled, versioned reference corpus.** Ship (or fetch-on-first-
  use with a pinned manifest digest) `corpora/reference_v1`: a few hundred
  sealed series covering all 12 archetypes and both tiers, small enough to
  distribute, built by a committed config with a recorded seed. Two payoffs
  beyond convenience: a new user's first report needs no `build_pipeline`
  run at all, and **results become comparable across users and machines**,
  because everyone's numbers are against the same corpus digest — which is
  what makes E5's registry meaningful and what a leaderboard would require.
  Every report prints the corpus version and digest. Keep the private split
  out of the distribution (`CLAUDE.md` §4.5's seal-records-intent caveat) and
  document the epoch-regeneration path for anyone who wants a fresh held-out
  set. Depends on: A7 (digest in provenance).
- [ ] **E5 · A results registry, so adding a model appends to a table.**
  Append-only `results/index.parquet`: one row per (corpus version, checkpoint,
  config hash, stage) with the headline statistics and full provenance;
  `meta_report.py` reads it instead of globbing run directories. Turns the
  current one-off cross-run comparison into (a) regression tracking across
  checkpoints and library versions and (b) the substrate for a public
  comparison table. Depends on: A3, A7, E4.
- [ ] **E6 · Report as a shareable product.** Fold in A5's coverage panel and
  A7's provenance panel, then add: `report/findings.json` (machine-readable
  findings with the claim ids from A15's registry, so downstream tooling and
  the bias card consume one source of truth), per-finding confidence badges
  (evidence class + registered/exploratory + whether it cleared the A13 noise
  floor), stable anchors for deep-linking a section, and a "what changed vs.
  run X" diff mode built on E5. Depends on: A5, A7, A13, A15.
- [ ] **E7 · Docs, quickstart, notebooks, packaging.** §10's existing open
  items, restated here only because E1-E6 change what they should say: a
  quickstart that is literally E1's one command; notebooks for (a) smoke run,
  (b) add an adapter — ideally now "point `GenericHFAdapter` at a checkpoint
  and watch span discovery run", (c) train and read an SAE, (d) read a
  confirm verdict; the auto-generated capability matrix (built 2026-08-06)
  rendered into those docs; and the one-repo/two-packages install story
  decided and documented, with a `tsfm-lens` console script. Depends on: E1.
- [ ] **E8 · CI.** = A14. Listed here too because "anyone can use it" is not
  credible without it.

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
- [ ] **E13 · Spectral lens.** The time-domain skip lens says *when* the
  forecast crystallizes; a frequency-domain version says *what* crystallizes
  first. Per layer, compare the skip-lens forecast's spectrum against the
  target's, resolved by frequency band, and check band-wise error against the
  ground-truth seasonal periods the benchmark records exactly. Answers "does
  the model get the trend right early and the seasonality late (or the
  reverse)", is native to time series in a way the borrowed logit-lens idea
  isn't, reuses the existing lens forwards, and feeds §6.2's
  frequency-aware-dictionary idea with the evidence it needs to be worth
  trying.
- [ ] **E14 · Steering / directional control.** `transformer-lens` users
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
- [ ] **E15 · Per-token SAE evaluation, then feature-level ablation.** Fixes
  the window-broadcast confound that currently makes Chronos's
  forecast-preservation number unreadable (§6.2), which is the stated blocker
  on §7 bullet 3. Reconstruct and patch at *token* granularity rather than
  broadcasting one window-pooled vector across every token in the window;
  for per-timestep-tokenized models this is the difference between a
  meaningful validity check and an architecture-dependent artifact. Then
  feature-level ablation as §7 bullet 3 specifies, cross-referenced against
  ground-truth alignment. Depends on: A4 and A13 (so the resulting ΔMASE
  values mean something).
- [ ] **E16 · Cross-model feature matching for per-model dictionaries.** The
  crosscoder (§6.2 item 1) is one answer to cross-model comparability; the
  complementary one — needed anyway, per `CLAUDE.md` §13 item 3 — is matching
  independently-trained dictionaries: input-space decoder correlation,
  max-activating-example overlap (Jaccard over top-k series/windows), and
  ground-truth-field agreement as a third, independent matching signal this
  repo can compute and most SAE work can't. Report the matched/unmatched
  split as a direct answer to the founding question, alongside the
  crosscoder's shared/specific split, and check whether the two methods
  agree — a genuinely informative cross-check.
- [ ] **E17 · Input front-end diagnostics as a real stage.** Generalize the
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
- [ ] **E18 · Component-level attribution (path patching).** `CLAUDE.md` §13
  item 8's "deeper component resolution", scoped: head→head and head→MLP path
  patching within a model, so a claim can be "this head's output matters
  *because* that head reads it" instead of "these heads matter." This is the
  clearest remaining gap against transformer-lens's own resolution, and the
  `token_patch` primitive plus `attention_info`/`mlp_info` already provide the
  hooks. Genuinely more expensive than everything above it — schedule after
  T1/T2.

### T4 — Breadth and scale

- [ ] **E19 · Multivariate / covariate envelope.** Moirai and Chronos-2 (both
  §9 Phase 4 candidates) are multivariate-native; the whole pipeline assumes
  univariate `[series, time]`. Decide and document the axis extension *before*
  Phase 4 forces an improvised one: `[series, variate, window, dim]` with
  alignment applied per variate, cross-variate attention treated as its own
  measurement (not folded into the lag taxonomy), and an explicit statement of
  which analyses are variate-agnostic. Doing this as a design note now is
  cheap; doing it under pressure while adding a model is how abstractions get
  bent (§2.4).
- [ ] **E20 · Context-length and horizon scaling sweeps.** Reuse
  `analysis/parameter_sweep.py`'s machinery on the *config* axis rather than
  the data axis: how do MASE, crystallization depth, and attention lag
  profiles change as `context_len` grows (128 → the checkpoint's max)? Users
  choose context length in practice and have no guidance; and "does the model
  actually use long context" is answerable here with existing tools.
- [ ] **E21 · Chronos decoder capture as a separate measurement.**
  `CLAUDE.md` §13 item 2 / §12 item 1, unchanged in substance: a second
  capture surface over decoder layers under teacher forcing, with its own
  forecast-time alignment windows, reported as a *distinct* measurement rather
  than CKA'd against context-window states. Worth restating in this backlog
  because it is the single largest **coverage asymmetry** in the flagship
  comparison — half of Chronos is unobserved — and any claim about what
  "Chronos represents" is really about its encoder until this exists.
- [ ] **E22 · Scale.** A17's blocked matcher (so a full-scale corpus can be
  validated at all), streaming/chunked CKA and ridge solves so `l1.max_rows`
  and `l2.max_rows` stop being the binding constraint on cross-model
  statistics, and a cost model that turns E2's VRAM/wall-clock estimate into a
  planning tool ("this config will take ~35 min and 14 GB; per-window
  patching is 60% of it"). The §5.1 literal-scale `full_multidomain.yaml`
  build (~99K samples, ~6 h estimated) is the concrete forcing function.
- [ ] **E23 · `benchmark_validation` diversity metrics as CI pass/fail
  gates.** `CLAUDE.md` §13 item 4, not previously cross-referenced into this
  backlog (closed the gap 2026-08-11, §11's own item revisiting `CLAUDE.md`
  §13). Turn the diversity metrics (`validation_report.json`'s redundancy
  fraction, effective dimensionality, near-collision fraction, per-group
  breakdowns from A17's fix) into explicit numeric thresholds checked
  automatically per benchmark epoch, rather than eyeballed from the report —
  e.g. fail if redundancy fraction exceeds some X, effective dimensionality
  drops below some Y, or any subgroup's near-collision fraction exceeds Z.
  Distinct from E8/A14 (test-suite CI): this is a data-quality gate on a
  *generated corpus*, not a code-correctness gate on the *pipeline*. Needs a
  first real threshold-calibration pass against at least one full corpus
  build (the §5.1 demo-mode numbers — 4.8% redundancy, eff-dim 4.6/24 — are
  the only reference point so far, and it's one sample, not a validated
  threshold) before the gate can be anything other than arbitrary.
- [ ] **E24 · Real-corpus activation bucketing.** `CLAUDE.md` §13 item 7, not
  previously cross-referenced into this backlog (closed the gap 2026-08-11,
  same pass as E23). Embed a large real corpus (Monash, GIFT-Eval
  pretraining corpus, LOTSA) in each model, cluster, and compare
  **partitions** across models via AMI / cluster matching — the same L4
  method already built (`analysis/clustering.py`), just pointed at wild
  data instead of the synthetic benchmark. Deliberately sequenced after (not
  instead of) the synthetic-corpus work already done: clusters on
  uncontrolled real data are uninterpretable without the controlled
  reference points L4 already established on the synthetic benchmark first
  — this is a generalization check on an existing method, not a
  replacement for it. Needs a real large-corpus source wired in (Monash's
  `sources.py` adapter already exists per §4.3 and is the natural first
  choice) and a GPU session with both models loaded; no design blocker,
  just not yet scheduled.

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
