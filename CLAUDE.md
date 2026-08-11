# CLAUDE.md — TSFM Interpretability Pipeline

> **Purpose of this file.** Onboarding manual for a fresh session. It captures not
> just *what* exists but *why it was built that way*, which invariants are
> load-bearing, and which mistakes have already been made and fixed. Read
> §2 (Doctrine) and §7 (Invariants) before writing any code. Read §11 (Traps)
> before touching the generators or the report.
>
> **Provenance note.** This is reconstructed from the project's four design
> sessions. Where a detail is uncertain, it is marked `[VERIFY]`. Confirm
> against the actual repo before relying on it.
>
> **Reconciliation note (2026-08-03).** A pass was made to check this file's
> claims against the live repo rather than trust it blindly (per §2.4's own
> doctrine). Several things had drifted or were wrong and are now fixed
> in place below — notably the `benchmark_validation` location, the
> `CO_trev_1_num` fix (a different, already-landed mechanism than what was
> described), the TimesFM adapter's move to TimesFM 2.5 (checkpoint, layer
> count, and a new `attention_patterns` capability), and two new L3
> corruptions. This was not an exhaustive re-audit of every claim in this
> file — treat anything not called out as a fix below with the same
> "confirm before relying on it" posture as an untouched `[VERIFY]`. See
> `ROADMAP.md` in the repo root for where the project is headed next; this
> file stays the "what exists and why" reference.
>
> **Reconciliation note (2026-08-03, same-day follow-up).** Picked up
> `ROADMAP.md` Phase 1 §5.2 — actually running the "wired but unverified"
> real-data path (§12) rather than trusting that label. It was **not**
> just unverified, it was **broken in four independent ways**: `datasets>=3`
> removed script-based dataset loading (which is exactly what
> `Monash-University/monash_tsf` is) entirely; `tsbootstrap`'s public API was
> completely rewritten since `block_bootstrap` was written; `sequential_par`
> was a literal `NotImplementedError` stub; and a real bug in
> `run_full.py`'s source injection meant only `mixture` (never
> `block_bootstrap`/`sequential_par`) could actually work through the CLI.
> All four are now fixed — see §4.3, §11.9–§11.13, and §12 below, plus
> `ROADMAP.md` §5's Findings block for the full story and exact numbers from
> a live end-to-end build. Also surfaced, and **not yet resolved**: the
> golden-hash bit-exact-reproducibility test (§7 invariant 1, §9, §11.1)
> fails against numpy 2.1.0 for reasons unrelated to this session's changes —
> see §9 and §11.13.
>
> **Reconciliation note (2026-08-03, third session).** `tsfm_lens` was run
> against **real checkpoints for the first time** (`ROADMAP.md` §5.4):
> `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-small`, on a
> previously-undocumented local GPU (NVIDIA RTX 5070, 12GB). Same pattern as
> the §5.2 session above — "wired but unverified" turned out to mean
> **actually broken, twice over**, both now fixed: `extraction/store.py`
> called a zarr-**v3-only** `Group.create_array` method despite the package
> being pinned to zarr **v2** (`create_dataset` is v2's actual method — see
> §11.15), and `impulse_alignment_check`'s hardcoded impulse amplitude
> silently broke on Chronos's context-adaptive quantization tokenizer,
> producing misleadingly low alignment scores that looked like a broken
> adapter but were a broken test (§11.16). Both of `CLAUDE.md` §9's
> live-weights unknowns are now confirmed working (Chronos
> `output_attentions`, TimesFM's SDPA-unfusing `attention_patterns`), and
> §10's "flat TimesFM patching curve" is now confirmed to be exactly the
> whole-layer-averaging artifact it was suspected to be — per-window
> patching recovers real, depth-intensifying causal structure. See §9, §10,
> §11.15–§11.16 below and `ROADMAP.md` §5.4's Findings block for full numbers.
>
> **Reconciliation note (2026-08-05).** Completed `ROADMAP.md` §5.3 (the
> deferred chronos-t5-base-vs-small size-confound run), on a fourth
> previously-undocumented machine/environment (Linux, `cudaPy` conda env,
> 8× RTX A5000) — the same zarr-v3-vs-v2 trap from the third session above
> recurred and was fixed the same way (pin `zarr<3`), now confirmed
> load-bearing across three separate environments. Built the §5.5
> cross-run aggregator (`tsfm_lens/report/meta_report.py` +
> `run_meta_report.py`), which immediately surfaced a real three-run
> replication of the `random_parametric`-favors-TimesFM finding across two
> checkpoint sizes and a third, independent corpus. Per §5.5's own
> instruction, **§10 below is now retired** — its provisional single-run
> findings are preserved verbatim, just relocated to `ROADMAP.md` §5's
> Findings block alongside every later run's numbers, since a findings
> section in this file goes stale exactly the way that one did. See
> `ROADMAP.md` §5's Findings block (2026-08-05 entries) for the full
> base-vs-small comparison, including a genuine disagreement between L1/L2
> (grow with size) and L4 clustering AMI (shrinks with size) worth reading
> in full rather than summarizing here.
>
> **Reconciliation note (2026-08-05, same-day follow-up).** Implemented and
> empirically bake-off-tested `ROADMAP.md` §6.1.1's redesigned layer
> selectors (`tsfm_lens/analysis/layer_screen.py` +
> `layer_screen_bakeoff.py`, `run_layer_screen_bakeoff.py`) — the fix for
> §6.1's `recommend_layers`, which §6.2's own Findings had already shown
> provably picks the *worst* layer within a single real model. See the new
> §6.1's "Correction" callout below and §11.18 for what's new; the full
> empirical result (which of three candidate selectors actually beat cheap
> nulls, on real `google/timesfm-2.5-200m-pytorch` +
> `amazon/chronos-t5-small` checkpoints with TimesFM captured at **all 20**
> layers for the first time, not the usual stride-2 10) lives in
> `ROADMAP.md` §6.1.1's Findings block, not here, per this file's own
> "describe stable architecture, not a moving research result" doctrine
> (§5.5's precedent). **Not yet wired into any config or the SAE stage** —
> the result is a real but provisional single-corpus, single-checkpoint-pair
> first pass.
>
> **Reconciliation note (2026-08-05, third same-day follow-up).** The
> "not yet wired into any config" line directly above is now stale: per
> explicit user instruction, `layer_screen` (default method `work_bend`)
> is wired in as a real pipeline `Stage` (`pipeline.py`) that runs by
> default right after `extract`, and `sae/train.py`'s `sae.targets: auto`
> resolution now consumes its selection instead of the old arbitrary
> "final captured layer" default. This is a deliberate policy call, not a
> new empirical result — the underlying bake-off is still the single
> (corpus, checkpoint-pair) data point described above; what changed is
> that a provisional-but-currently-best method was judged good enough to
> replace an equally-unvalidated old default, on the user's authority to
> make that call. §3's repo layout, §6.1's stage table, and §9's
> verification table below are updated; full mechanism and test evidence
> are in `ROADMAP.md` §6.1.1's second Findings block, not repeated here.
>
> **Reconciliation note (2026-08-05, fourth same-day follow-up).** Picked
> up `ROADMAP.md` §13's explicit prerequisite for the flagship crosscoder
> (§6.2 item 1) — a small-scale trainability/stability test — rather than
> building the full candidate directly. New `tsfm_lens/sae/crosscoder.py`
> (`CrosscoderSAE`) + `run_crosscoder_feasibility.py`; found and fixed a
> real training instability (unequal per-source activation scale lets the
> larger-scale source dominate the joint loss, confirmed synthetically as
> a fidelity collapse) and a device-mismatch bug only a real-GPU run
> surfaced — see the new §11.19. Real-checkpoint result: joint training is
> stable once the scale fix is in (no source-domination collapse across
> three hyperparameter settings against `runs/medium_run_chronos_base`'s
> already-extracted TimesFM/Chronos-T5-Base store), but this is a
> feasibility gate only — no `SAEAdapter` implementation, pipeline stage,
> or report section exists yet. §3's repo layout and §9's verification
> table are updated; full numbers are in `ROADMAP.md` §6.2's new Findings
> block, not repeated here.
>
> **Reconciliation note (2026-08-06) — audit pass; several claims in this
> file are now qualified.** A planning-only session read this file and
> `ROADMAP.md` against the actual pipeline looking for silent-failure paths
> and fixable limitations, and wrote the result up as **`ROADMAP.md` §15**
> (19 items, each with `file:line` evidence and a detailed fix plan, all
> marked NEEDS IMPLEMENTATION) plus **§16** (an enhancement backlog for the
> "anyone, any model, one button" goal). Nothing was implemented. The four
> findings that directly qualify statements made *in this file* are marked
> inline below (§6.3, §6.4, §7 invariant 7, §8) and collected as §11.20;
> summarized once here because they change how a fresh session should read
> the rest of this document:
> 1. **Invariant 7 is not machine-enforced.** `extraction/extract.py` calls
>    `impulse_alignment_check` and *discards the result* — no threshold, no
>    artifact, no report line, and only a stride-4 subset of layers probed.
>    §6.3's "near-zero everywhere means fix the adapter before trusting any
>    cross-model number" is true and currently depends on a human running
>    `--check-alignment` by hand and reading it (`ROADMAP.md` §15 A2).
> 2. **`build_pipeline` writes corpora grouped by task, and four analysis
>    call sites subsample with a head slice**, so several already-recorded
>    numbers were measured on family-skewed prefixes rather than
>    representative samples (`ROADMAP.md` §15 A4 names which).
> 3. **Stage skipping has no config fingerprint** — editing a config and
>    rerunning into the same run directory silently mixes artifacts from two
>    configs (§15 A3). §8's "stages self-skip when artifacts exist" is
>    accurate but incomplete as guidance.
> 4. **The `layer_screen` stage cannot satisfy its own all-layers-fair
>    requirement** under any production config, because it screens the
>    strided store rather than every block (§15 A1). §6.1's Screen row and
>    §8's `layer_screen` knob description are both correct about what the
>    stage *does*; the requirement it was designed to meet is unimplemented.
>
> **Reconciliation note (2026-08-06, fifth same-day follow-up) — untrained-
> weights null baseline implemented (`ROADMAP.md` §16 E9).** New
> `ModelConfig.random_init: bool` flag (§6.2 below): a model config with
> `random_init: true` loads the same architecture (and, for Chronos, the
> same tokenizer/`chronos_config` metadata) but with freshly, randomly
> initialized weights instead of the pretrained checkpoint, via each
> library's own from-config construction path (`type(model)(model.config)`
> for the two HF-based Chronos adapters; simply not calling
> `.load_checkpoint()` for TimesFM, whose wrapper already builds its full
> architecture before any checkpoint step) — no generic reinitialization
> heuristic, and no per-adapter special-casing beyond one `if
> self.cfg.random_init` branch in each adapter's existing `load()`. Pairing
> a real model against its own `random_init` twin as an ordinary two-model
> config gives L1's CKA, L2's stitching gain, internals' probe
> decodability, and SAE ground-truth alignment a real floor with **zero
> changes to any of those four analysis modules** — the null is just
> another run of the unchanged pipeline. Also added: a label-permutation
> null for the SAE ground-truth alignment score itself
> (`sae/ground_truth.py::permutation_null_alignment`), since
> `best_ground_truth_matches` picks each feature's *best* of ~30 candidate
> fields, which inflates the headline `mean_abs_rho_matched` above zero
> from search alone even under pure noise — the permutation null shows how
> large that same number gets by chance, rendered next to the real value
> in the report. New `configs/null_timesfm_random.yaml` /
> `null_chronos_random.yaml` pair each real model (at the exact checkpoint
> and layer `configs/medium_run_chronos_base.yaml` already has recorded
> numbers for) against its random-init twin. Verified against real
> checkpoints (`ROADMAP.md` §16 E9's Findings has the numbers once the
> background run this session launched completes — see that section for
> the run directories and current status if you're reading this before it
> has). New unit tests (`tests/test_random_init.py`,
> `tests/test_ground_truth_permutation_null.py`) cover the reconstruction
> mechanism directly against a tiny offline HF T5 (no network), the mock
> adapters' seed-offset emulation of the same flag, an end-to-end
> mock-pipeline run of a real-vs-random-init pair through `extract`+`l1`,
> and the permutation null's statistical behavior on synthetic planted-
> signal-vs-pure-noise data. Full suite green after this addition (see
> `ROADMAP.md` §16 E9 for the exact count). One real bug found and fixed
> by actually running the smoke test rather than trusting the diff:
> `report.py`'s `_note()` takes exactly three positional fields, and a
> fourth appended to `_SAE_EXEMPLAR_NOTE` collided with the `summary=`
> keyword — folded into the existing "Limitations" field instead of adding
> a fourth positional argument.
>
> **Reconciliation note (2026-08-10).** No repo-content correction this
> time — added **§2.8** below, a doctrine section on *how* a session should
> work in this repo, prompted by the fact that this pattern was already
> happening ad hoc (`ROADMAP.md` §14's several "run via a scheduled
> autonomous loop" session-log entries from 2026-08-06/07) without ever
> being written down as an instruction for a fresh session to follow on
> purpose. Also picked up the next concrete open item this same session:
> `ROADMAP.md` §13's flagged-but-not-yet-run re-test of the `factor_emergence`
> layer-selector fix (`CLAUDE.md` §11.18) against the real bake-off — see
> `ROADMAP.md` §6.1.1's Findings and §13 for the outcome once that
> background run (started this session) lands.
>
> **Reconciliation note (2026-08-11) — `ROADMAP.md` planning pass; two
> claims in this file are now stale, one is unchanged-but-now-planned.** A
> user-directed session restructured `ROADMAP.md` (no code changed, no
> recorded number altered): added a **§0.5 "Start here — next actions, in
> order"** entry point, a full build plan for the flagship crosscoder
> (**§6.2.1**, with a blocking dead-feature gate, a five-rung validation
> ladder, and six candidate variants V0–V6 to be built and compared), a
> reopening of the falsified provenance work with five replacement
> approaches (**§6.3.1**), corrections blocks on §11/§12/§16, detail-ups
> for all twelve unstarted §16 items, and four previously-untracked §13
> entries. Full writeup in `ROADMAP.md` §14's entry for this date. What
> changes for *this* file:
> 1. **§13 item 4's E23 caveat is stale.** It says the diversity-gate
>    thresholds are calibrated against only the demo-mode reference point
>    and that the item "stays open" for that reason. `ROADMAP.md` §16 E23
>    is `[x]` — a real corpus build has since confirmed the thresholds and
>    no recalibration was needed.
> 2. **§13 item 3 and §6.2's "flagship crosscoder not started" is still
>    accurate but no longer the whole story** — it now has a complete,
>    stage-gated build plan at `ROADMAP.md` §6.2.1. Read that before
>    starting any crosscoder work, in particular its Stage 0 finding that
>    the 90–98% dead-feature rate makes `relative_decoder_norm` read ~98%
>    "shared" from dead-atom symmetry alone, so **no shared-vs-specific
>    split from the existing feasibility run should be quoted.**
> 3. **§12's "which of these are actually fixable" note gains one
>    correction:** the Chronos-decoder gap is no longer a clean non-goal —
>    `ROADMAP.md` §12 reclassified it as *deferred with a design* (§16
>    E21), with the never-CKA-decoder-against-context-states rule as the
>    binding constraint.

---

## 1. What this project is

A two-part research repository for **mechanistic interpretability of time-series
foundation models (TSFMs)**, built to compare **TimesFM** and **Chronos**
despite fundamentally different architectures. Usability is deliberately modeled
on `transformer-lens`.

**The central research questions:**
- Is there an intrinsic circuit/representation present across multiple TSFMs?
- What differs in what each model learns?
- Is one model better suited to high-frequency structure vs. statistical means?
- Where in depth does each model's forecast actually crystallize?

**The two halves:**

| Half | Package | Job |
|---|---|---|
| Benchmark generation | `tsfm_benchmark/build_pipeline` | Produce leakage-audited, sealed, ground-truth-labeled synthetic corpora |
| Benchmark validation | `benchmark_validation` | Prove a generated corpus is actually diverse and non-redundant |
| Model analysis | `tsfm_model_analysis/tsfm_lens` | The layered interpretability pipeline ("tsfm-lens") → one HTML report |

**The two models being compared, and why they're hard to compare:**

| | TimesFM | Chronos-T5 | Chronos-Bolt |
|---|---|---|---|
| Architecture | **Decoder-only** (one stack reads context *and* generates) | **Encoder-decoder** (T5) | Encoder-decoder, patch-based |
| Tokenization | Patches, 32 timesteps/token, MLP embedding | Scalar quantization, **1 timestep/token**, categorical vocab | Patches |
| `token_width` | ~32 | ~1 | patch size from checkpoint config |
| Decoding | Deterministic | **Sampled** (`num_samples`) | Deterministic |
| Captured surface | Essentially the whole computation | **Encoder only** — decoder largely invisible | Encoder |

> **Correction worth remembering** (an earlier session got this backwards):
> TimesFM is *decoder-only*, Chronos-T5 is the *encoder-decoder*. The asymmetry
> is therefore: TimesFM's captured stack does everything; Chronos's captured
> stack only *reads* the context.

> **A fourth architecture is now integrated at the adapter level (ROADMAP.md
> §9, Phase 4), not yet part of the flagship comparison above.** Chronos-2
> (`models/chronos2_adapter.py`) is encoder-only with no decoder at all, and
> adds a second, cross-series "group" attention axis neither model above
> has. It has real, live-checkpoint numbers on record (§6.2, §11.21) but no
> `default.yaml`/`medium_run*.yaml` config yet — the table above still
> describes the pair this repo's central research questions were built
> around and is not stale, just no longer literally "the two models this
> repo can analyze."

---

## 2. Engineering doctrine — how code is written here

This is not stylistic preference; it is the reason the repo is trustworthy.
Match it exactly.

### 2.1 Think about downstream effects before writing
Every change is evaluated against the whole pipeline, not the local function.
The canonical example: adding four new generator archetypes to a default list
looked local, but silently changed `rng.choice(len(names))` for every existing
seed, breaking bit-exact regeneration of already-sealed corpora. The fix was to
freeze the default and make new entries opt-in. **Ask "what does this break
three stages downstream?" before, not after.**

### 2.2 Prefer established libraries over new code
numpy, scipy, scikit-learn, torch, zarr, plotly, jinja2, pycatch22,
dtaidistance, umap-learn, SDV/DeepEcho, tsbootstrap. Write code only where a
library would *obscure the substance*. Two deliberate exceptions, both
documented in-repo:
- **Metrics** (MASE / sMAPE / pinball) — ~20 auditable lines in
  `analysis/l0_behavioral.py` rather than a heavy evaluation framework, so the
  formulas can be read.
- **Statistics** — ~80 auditable lines in `analysis/stats.py` on numpy, because
  the *bootstrap designs* (cluster, paired) are the scientific substance and
  generic CI libraries hide them.

### 2.3 Comment discipline
Clear docstrings at **function and module level**. Module docstrings explain
*why the module exists and what it inherits from the previous stage*. **No
per-line comments.** Config YAML is the exception — it is documented inline
because it is the user-facing surface.

### 2.4 Verify empirically; distrust your first instinct
Run the code. Test the hypothesis before asserting it. A worked example from the
validation work: the guess was that `CO_trev_1_num`'s huge variance came from raw
input amplitude; z-scoring first proved that wrong (pycatch22 z-scores
internally), and the real cause was extreme isolated spikes dominating cubed
differences. **"Good thing I tested — my first instinct was wrong and the data
proves it"** is the expected posture.

### 2.5 Degrade gracefully and loudly; never silently
- Unsupported capability → log `unsupported`, skip the analysis, render the
  report without that section.
- Wrong/unresolvable module slice → **disable the analysis with a log** rather
  than risk a silently wrong slice.
- Broken assumption (e.g. non-uniform hidden size at the skip lens) → **fail
  loudly**.
- Library-version fragility → resolve defensively, *warn when guessing*.

### 2.6 Be honest about evidence class
Every stage states what kind of evidence it produces and what limitation it
inherits. The report's own framing enforces this: L1 claims are *geometric*, L2
claims are *linear-translatability*, L3 claims are *causal within-model*. Never
let a correlational number be read as causal. Fallbacks are named in output
(`tsne_fallback`, `pca_fallback`) so a reader can never mistake one for the
primary method.

### 2.7 Push back on the framing when it's contradictory
The first session's brief ("no leakage into any model's training data" **and**
"make data similar to real data") contains a real contradiction. Rather than
paper over it, it was split into two labeled tiers. Do the same with future
briefs — surface the tension, then architect around it.

### 2.8 Delegate long-running compute; keep the roadmap moving in parallel

This repo's own history is the evidence for this section: every "autonomous
loop" session-log entry in `ROADMAP.md` §14 from 2026-08-06/07 already did
this ad hoc, and it produced more roadmap progress per session than any
single-threaded session before it. The pattern below is that same practice,
written down on purpose instead of reinvented next time.

**The problem this solves.** Real work in this repo is bimodal: most edits
(a bug fix, a new selector, a report section) are seconds of compute and
minutes of thinking; a handful of things (extraction against a live
checkpoint, SAE/crosscoder training, a layer-screen bake-off, a full
multi-stage pipeline run) are 5+ minutes of GPU/CPU time where the working
session can do nothing useful by sitting and waiting. Blocking on the
second kind wastes the first kind's opportunity cost.

**The rule.** Before starting anything expected to run 5+ minutes
(extraction, any real-checkpoint pipeline stage, SAE/crosscoder training, a
bake-off, a param sweep, a full test-suite-plus-live-run validation pass):
1. Launch it via the `Agent` tool with `run_in_background: true` rather than
   running it inline. Brief that agent exactly like a fresh colleague — it
   has no memory of this session — with: which section of `CLAUDE.md`/
   `ROADMAP.md` motivates the run, the exact commands (including env
   activation — see `DEPENDENCIES.md`), which existing artifacts to reuse
   vs. regenerate and why, and precisely what to report back (raw numbers,
   quoted exactly — not rounded or paraphrased — since these often become
   permanent `ROADMAP.md` Findings text).
2. Tell that agent to **report results, not write findings** — it should
   not edit `ROADMAP.md`/`CLAUDE.md` itself. Only the session that reads its
   report writes the findings up, so two concurrent writers never race on
   the same doc edit (see `ROADMAP.md` §0's "correct in place, never
   silently delete" discipline — that only works if one writer is doing it
   at a time).
3. While it runs, do NOT poll or wait — pick up the *next* unfinished
   roadmap item that doesn't touch the same files/artifacts the background
   run is using (a different subsystem, a docs pass, a quick synthetic-data
   test, a licensing check) and make progress on that instead. If nothing
   independent is available, it is fine to end the turn — the harness
   notifies automatically when the background agent finishes; do not
   fabricate, guess, or narrate its results before that notification
   actually arrives.
4. When the notification lands: read its report, write the findings into
   `ROADMAP.md` (append, don't overwrite — §0's discipline) and `CLAUDE.md`
   if a stated claim changed, run whatever test suite the change touches,
   and only then mark the roadmap item's checkbox done.

**Keeping a session going across a whole afternoon without a human
re-prompting it.** When asked to keep working autonomously for a stretch
(not just one background job), pair the above with a recurring wake-up —
either the `loop` skill (`/loop 30m ...`, which schedules via `CronCreate`)
or `ScheduleWakeup` inside an already-running `/loop`. Each firing should:
check whether a previously-launched background agent has since completed
(if so, write up its findings per step 4 above before doing anything else);
if nothing is in-flight, re-read `ROADMAP.md` fresh (its own state is the
source of truth on what's next, not this session's memory of it — the doc
is kept current precisely so a new firing can pick up cold) and start the
next unfinished item, applying this section's own rule recursively (long →
background + move on, short → just do it). Give the recurring prompt an
explicit stop condition (a wall-clock deadline, or "no unfinished items
remain") so it doesn't run past what was actually asked for — cron jobs in
this harness are session-only and auto-expire after 7 days regardless, but
don't rely on that as the stop condition if a shorter one was requested.

---

## 3. Repository layout

```
TSFM-Interp/
├── CLAUDE.md, ROADMAP.md, README.md   # this file, forward plan, one-line repo blurb
├── DEPENDENCIES.md                    # exact verified library versions + env recreation (§7 invariant 12)
├── pyproject.toml                     # root package = tsfm-benchmark only (see below)
├── clean_reinstall.sh                 # pinned torch 2.9.1+cu130 / numpy 2.1.0 / transformers
│                                       # env rebuild script — not part of the analysis code
├── tsfm_benchmark/
│   ├── build_pipeline/          # the generation package
│   │   ├── schema.py            # TimeSeriesSample, GroundTruth, Provenance, SourceRef, ProvenanceStep
│   │   ├── registry.py          # GENERATORS / CORRUPTIONS / SOURCES plugin registries
│   │   ├── generators.py        # parametric, random_parametric, mixture, block_bootstrap, sequential_par
│   │   ├── corruptions.py       # corruption chain primitives
│   │   ├── sources.py           # real-data adapters: generic HF loader (any repo id) +
│   │   │                        # bootstrap_catalog; monash/ett both verified live
│   │   ├── audit.py             # LeakageAuditor, find_near_duplicates, realism report
│   │   ├── builder.py           # BenchmarkBuilder: specs → audited → split → sealed
│   │   └── seal.py              # seal_corpus / load_sealed with hash manifests
│   ├── benchmark_validation/    # validation package — CORRECTED: nested here, confirmed via
│   │   │                        # pyproject.toml's package list, NOT a sibling of tsfm_benchmark
│   │   ├── matching.py          # xcorr + banded DTW shape matching, redundancy_by_group
│   │   ├── features.py          # catch22/catch24 matrix, RobustScaler + winsorization, imputation
│   │   ├── embedding.py         # 3D UMAP (+ flagged tsne/pca fallback)
│   │   ├── diversity.py         # participation-ratio eff-dim, NN tail, near-collisions,
│   │   │                        # diversity_metrics_by_group for per-subgroup collapse checks
│   │   ├── plot.py              # incl. an anomaly gallery of the most-winsorized sequences
│   │   ├── loaders.py, report.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── configs/
│   │   ├── example.yaml         # heavily inline-documented reference config
│   │   ├── medium_run.yaml      # mid-scale config between smoke-sized and full
│   │   ├── full_multidomain.yaml# all 12 archetypes + all 3 real-derived generators,
│   │   │                        # 2 real corpora (Monash full catalog + ETT); the literal
│   │   │                        # aspirational scale (~99K samples across both splits) --
│   │   │                        # not yet built, see full_multidomain_run1.yaml and §11.14
│   │   ├── full_multidomain_run1.yaml # ratio-preserving ~10x scale-down of the above,
│   │   │                        # sized from a live throughput calibration to actually
│   │   │                        # finish in a session (§11.14); first real (non-smoke)
│   │   │                        # build + validation run, ROADMAP.md §5.1
│   │   └── real_data_smoke.yaml # fast (~1 min) live real-data path check
│   ├── example_runs/            # run_full.py (build CLI), run_validation.py (validation CLI),
│   │                            # run_smoke.py (leakage/seal/reproducibility smoke), WALKTHROUGH.md
│   ├── tests/                   # 40 passed / 1 failed (golden-hash, see §9/§11.13); includes
│   │                            # test_generator_extensions.py (opt-in archetypes) and
│   │                            # test_real_derived_generators.py (block_bootstrap/sequential_par,
│   │                            # now also sequential_par's length-truncation, §11.14)
│   └── README.md
└── tsfm_model_analysis/
    └── tsfm_lens/
        ├── tsfm_lens/
        │   ├── config.py        # typed dataclasses, per-stage enables, YAML load
        │   ├── data.py          # sealed loader + jsonl fallback + smoke generator
        │   ├── utils.py         # log, save_json, batch_slices, relative_depths
        │   ├── models/          # base.py (ModelAdapter), timesfm/chronos/chronos_bolt/chronos2/mock,
        │   │                    # conformance.py (check_adapter_conformance -- ROADMAP.md
        │   │                    # §10's automated adapter-checklist, mocks only),
        │   │                    # capability_matrix.py (declared/verified capability
        │   │                    # table generator -- ROADMAP.md §10, no checkpoint load)
        │   ├── extraction/      # hooks.py, alignment.py, extract.py, store.py (zarr)
        │   ├── analysis/        # stats, l0_behavioral, internals, lens, l1_geometry,
        │   │                    # l2_stitching, l3_perturbation, attention, clustering,
        │   │                    # exemplars, confirm, layer_selection (cross-run study,
        │   │                    # ROADMAP.md §6.1 -- not a pipeline stage, reads existing
        │   │                    # run artifacts across N run dirs like report/meta_report.py),
        │   │                    # layer_screen.py (ROADMAP.md §6.1.1 -- IS a pipeline stage,
        │   │                    # runs right after extract via pipeline.py's `layer_screen`
        │   │                    # Stage; work_bend is the default method, feeding
        │   │                    # sae.targets: auto) + layer_screen_bakeoff.py (the
        │   │                    # null-controlled bake-off that chose work_bend -- not a
        │   │                    # pipeline stage, invoked via run_layer_screen_bakeoff.py
        │   │                    # like meta_report.py)
        │   ├── sae/            # interface.py (contract), models.py (TopKSAE baseline),
        │   │                    # train.py (training loop incl. dead-neuron resampling,
        │   │                    #   + pipeline-stage runner), eval.py (fidelity/dead-feature-
        │   │                    #   rate/forecast-preservation), ground_truth.py (feature-
        │   │                    #   alignment score), real_data.py (optional HF-sourced
        │   │                    #   training augmentation), crosscoder.py (CrosscoderSAE --
        │   │                    #   ROADMAP.md §13's crosscoder feasibility test; NOT an
        │   │                    #   SAEAdapter yet, NOT a pipeline stage, invoked via
        │   │                    #   run_crosscoder_feasibility.py against an already-
        │   │                    #   extracted run's store like layer_screen_bakeoff.py) --
        │   │                    #   ROADMAP.md §6.2
        │   ├── report/report.py # single-file interactive HTML (one run)
        │   ├── report/meta_report.py # cross-run aggregator (ROADMAP.md §5.5);
        │   │                    # reads N run dirs' existing artifacts, no re-run
        │   └── pipeline.py      # stage DAG, artifact skipping, dependency resolution
        ├── run.py               # CLI
        ├── run_meta_report.py   # CLI for report/meta_report.py: --runs a,b,c --out path
        ├── run_layer_screen_bakeoff.py # CLI for layer_screen_bakeoff.py (ROADMAP.md §6.1.1-E)
        ├── run_crosscoder_feasibility.py # CLI for sae/crosscoder.py (ROADMAP.md §13/§6.2)
        ├── run_noise_snr_sweep.py # reruns L3's noise corruption at several SNR values
        │                        # against an already-extracted run (ROADMAP.md §7)
        ├── run_capability_matrix.py # CLI for models/capability_matrix.py: prints/writes
        │                        # the auto-generated capability matrix; --verify
        │                        # adapter=run_dir:model_name reads an existing run's
        │                        # attention/ artifacts, no checkpoint loaded
        ├── configs/default.yaml # real pair (TimesFM vs Chronos)
        ├── configs/medium_run.yaml # mid-scale real-model config
        ├── configs/medium_run_chronos_base.yaml # size-variant control (chronos-t5-base)
        ├── configs/smoke.yaml   # two mock architectures, CPU, ~minutes
        ├── configs/layer_screen_experiment.yaml # the layer-screening bake-off config --
        │                        # TimesFM captured at ALL 20 layers (stride 1), not the
        │                        # usual stride-2 10; only extract/l0/internals/l3
        │                        # (sensitivity, no patching) stages enabled
        ├── tests/test_smoke.py  # full end-to-end + confirm hypothesis-path test
        ├── tests/test_meta_report.py # aggregator: missing-stage degrade, null-depth
        ├── tests/test_layer_screen.py # 3 selectors + bake-off scoring, all on synthetic
        │                        # data with a planted, known-correct answer
        ├── tests/test_crosscoder.py # joint-normalization invariant, planted shared/
        │                        # specific-cause recovery, engineered mismatched-scale
        │                        # stability check -- all synthetic, planted answers
        ├── tests/test_adapter_conformance.py # runs check_adapter_conformance against
        │                        # all 3 mock adapters incl. mock_wave (the unexercised-
        │                        # by-default third architecture), a no-optional-
        │                        # capabilities stand-in, and a deliberately-broken adapter
        ├── tests/test_capability_matrix.py # declared-vs-verified capability generator:
        │                        # synthetic fixtures only; the real-run cross-check that
        │                        # caught a numpy.bool_/`is` rendering bug is documented
        │                        # in ROADMAP.md, not repeated here as a pytest test
        ├── requirements.txt, pyproject.toml
        └── README.md
```

Note the two installable units: the root `pyproject.toml` packages only
`tsfm_benchmark` (+ `benchmark_validation` + `example_runs`); `tsfm_lens` has
its own separate `pyproject.toml` one level down. **Decided 2026-08-11
(`ROADMAP.md` Phase 5): this is deliberate, not unfinished** — the two
packages' dependency sets barely overlap and differ by an order of
magnitude in weight (`tsfm_benchmark`: numpy/scipy/pyyaml core, heavy real-
data deps opt-in; `tsfm_lens`: torch/zarr/plotly/scikit-learn core), so a
single merged package would force every install to pull the union of
both. Two independent installs, one repo — not a gap to close.

---

## 4. Part A — Benchmark generation (`build_pipeline`)

### 4.1 The load-bearing design decision: two leakage tiers

The brief's contradiction (§2.7) was resolved by distinguishing:
- **Instance-level leakage** — the exact series was trained on. Auditable, preventable.
- **Distributional leakage** — the model has seen *that kind* of data. Carried by
  construction by anything derived from or made similar to real corpora.

Making data realistic *increases* the second. So generation is split:

| Tier | `tier` field | Generators | Guarantee |
|---|---|---|---|
| Leakage-safe synthetic | `synthetic` | `parametric`, `random_parametric` | Touches **zero** real data; exact ground truth |
| Realism-stress (real-derived) | `real_derived` | `mixture`, `block_bootstrap`, `sequential_par` | Attribution ground truth, but inherits source distribution |

Every sample is **labeled with its tier**, and real-derived samples must still
pass the leakage audit to be admitted.

### 4.2 Generators

**`parametric`** — the backbone. Explicit additive composition, every component
recorded as ground truth:
- polynomial `trend` (`{order, scale}`)
- any number of `seasonalities` (`{period, amplitude, phase, shape}`; shapes:
  sine default, plus `square`, `sawtooth`, `triangle` with harmonics)
- `ar_coeffs` colored noise
- `n_changepoints` level shifts, `n_anomalies` point anomalies
- `random_walk_scale` (unit-root stochastic trend; component `random_walk`)
- `intermittency: {rate}` (component `intermittency_mask`)
- `heteroskedastic: {period, depth}` (component `noise_envelope`)
- unknown `shape` → `ValueError` (fails loudly)

**`random_parametric`** — redraws its *entire* structural recipe per call from an
archetype, so a task's `count` spans a genuinely varied population. Delegates
composition to `parametric` so ground truth stays exact; records the chosen
archetype and sampled recipe.

**The 12 archetypes.** The original 8 are the **frozen default**:
`trend_dominant`, `seasonal_dominant`, `multi_seasonal_complex`,
`regime_switching`, `ar_colored_noise`, `anomaly_heavy`, `clean_low_noise`,
`noisy_chaotic`.
Four are **opt-in only** (see §11.1 for why):
`random_walk_drift` (unit root), `intermittent_bursts` (30–80% zeros — the
sharpest quantized-vs-continuous stressor, i.e. Chronos vs TimesFM),
`amplitude_modulated` (heteroskedastic envelope), `nonsinusoidal_seasonal`.
Exact per-archetype sampling ranges live in `generators.py:_ARCHETYPES`.
`archetype_weights` biases the mix. `configs/full_multidomain.yaml` lists all 12
explicitly.

### 4.3 Sources
`sources.load_sources` works against **any Hugging Face dataset repo id**, not
just Monash — `kind: monash` is only a convenience alias presetting `dataset:
Monash-University/monash_tsf`. Verified live 2026-08-03 against a second,
independent corpus (`ETDataset/ett`, Electricity Transformer Temperature) with
zero new code, confirming this genericity claim was actually true and not
just documented. Real data is used **only** as (a) the leakage auditor's
reference corpus, (b) realism calibration, and (c) — for `mixture` /
`block_bootstrap` / `sequential_par` — the real-derived generators' raw
material (never copied in verbatim; see §4.1's tiers). `TIME`, `BOOM`,
`ARFBench` are *deliberately not implemented*: access paths and licenses
could not be verified. `monash`/`ett` are the templates to copy. `SourceRef`
records license per source so derived-data redistribution stays honorable.

⚠️ **Load-bearing version pin, discovered the hard way (§11.9):** `datasets`
must be `<3` — newer releases removed script-based dataset loading entirely,
and `Monash-University/monash_tsf` (like most Monash-era HF forecasting
datasets) is script-backed. `datasets==2.21.0` is verified working. This is
not a hypothetical risk; it is the reason the real-data path went unverified
for as long as it did (§12).

### 4.4 The audit gate (fires on every sample, before admission)
- **Two-stage DTW leakage gate** — cheap prefilter then Sakoe-Chiba banded DTW,
  `dtaidistance` C backend with a numpy fallback. Sample rejected below
  `--threshold` distance (default 0.35) from any reference series.
- **Near-duplicate detection** across the union of both splits, as a final
  cross-split guard.
- **Realism report** — gap in mean, std, lag-1 autocorrelation, spectral
  centroid vs a real target. Used to *tune the realism tier*, **never as a
  leakage signal**.

⚠️ `--references none` means the gate has nothing to compare against and **every
sample passes trivially.** Pass `--references monash` for the gate to do work.

### 4.5 Sealing, splits, and epochs
- Public dev and private test splits draw from **disjoint seed ranges** — same
  distribution, never a shared instance.
- `seal_corpus` writes `corpus.jsonl` + `manifest.json` (visibility, epoch,
  per-sample sha256, global digest). `load_sealed(verify=True)` re-hashes and
  raises on mismatch — catches tampering, truncation, or a public/private mix-up.
- **Epochs** (`_EPOCH_STRIDE = 100_000_000`): each epoch is a fresh
  non-overlapping seed range. `regenerate_private(epoch=N)` mints a brand-new
  held-out corpus, LiveBench-style, when the old one is suspected of having
  leaked. Every sealed epoch stays reproducible from its (private) seed.
- Seal records *intent*; it **cannot enforce filesystem permissions**. The
  private directory must be access-controlled by the surrounding deployment.

### 4.6 Not implemented by choice
LLM-based in-context generation (was in the original brief) — dropped as
unverifiable and non-reproducible.

---

## 5. Part B — Benchmark validation (`benchmark_validation`)

Answers "is this corpus actually diverse?" Two branches that **never touch until
the report**, by design, because they catch different failure modes.

**Stage 1 — shape matching (`matching.py`).** Sequences z-normalized and
resampled to equal-length windows, matched pairwise by rolling Pearson
cross-correlation over a lag band (`xcorr`, default) or localized banded DTW.
Outputs: similarity matrix, **equal-frequency bucketed histogram** of all scores,
redundancy fraction, explicit flagged pairs. O(n²) — `--max-sequences` subsamples.
The full matrix is omitted from JSON by default (O(n²)).

**Stage 2 — catch22 feature space.** `pycatch22` catch22 (or catch24, adding
mean/std), robustly scaled with degenerate-sequence imputation (counted).
Diversity metrics in `diversity.py`: **effective dimensionality** (participation
ratio of the PCA spectrum), total variance, per-feature variance ranking,
**nearest-neighbour distance tail** (mean/min/p05), **near-collision fraction**.

### 🔴 The single most important rule in this package
**Diversity is measured in the catch22 feature space, NEVER on the UMAP
coordinates.** UMAP preserves local neighbourhoods but distorts global distances
and densities, so cluster sizes and gaps in the 3D plot are *not quantitative*.
Using UMAP output to compute a diversity number is a common and serious mistake.
UMAP feeds the plot only; the plot title says so.

**Outputs:** `validation_report.json` + `feature_space_3d.html` (standalone
interactive plotly, colored by generator).

**Demo mode** (no `--corpus`): builds 5 distinct generators + 5 planted
near-duplicates. Planted duplicates must surface at similarity **1.0000** — a
quick correctness confirmation.

**Reference demo run** (205 sequences): planted duplicates at 1.0000, redundancy
fraction 4.8%, effective dimensionality 4.6 of 24. The 4.8% is a **true finding,
not a bug** — 40 pure-seasonal series share a period-24 shape and differ only in
noise, so under shape matching they are legitimately redundant. Read the
redundancy fraction *alongside* the feature metrics, never alone.

### Landed: winsorization fix for `CO_trev_1_num`-style outlier swamping

**Correction:** an earlier version of this file described the confirmed fix
as `QuantileTransformer` + ranking features by IQR, and marked it
`[VERIFY]`/possibly-not-landed. That is **not** what actually shipped. What's
in `features.py` today is a different (and, on reflection, better-documented)
fix for the same diagnosed problem — the diagnosis itself
(`CO_trev_1_num`'s tiny global IQR making a handful of extreme series scale
into the thousands and swamp every downstream L2 metric) was correct and is
preserved:

- `RobustScaler` output is **winsorized** (clipped) to ±`clip_scaled`
  robust-scale units (default `5.0`, `extract_features(..., clip_scaled=5.0)`
  in `features.py`) rather than passed through `QuantileTransformer`. This
  keeps genuinely extreme series pinned at the clip boundary — still visibly
  the most extreme thing in the corpus — without letting one feature's raw
  scale dominate every L2 (variance/PCA/nearest-neighbour) computation.
- Per-sequence **anomaly diagnostics** are now a first-class output of
  `FeatureMatrix` (`anomaly_scores`, `anomaly_features`, `anomaly_raw_values`
  — the pre-winsorization scaled value, which feature triggered it, and its
  raw value), surfaced via a gallery plot in `plot.py` and as `n_clipped` /
  `n_values_winsorized` counts in the validation report. This makes the
  outlier problem diagnosable per-sequence instead of only visible as a
  suspicious aggregate number.
- **Not landed, and not obviously still needed:** `feature_variance_ranking`
  in `diversity.py` still ranks by raw variance, not IQR. Winsorization
  addresses the same failure mode (a few extreme values dominating an L2
  metric) from a different angle, so re-ranking by IQR may now be redundant
  rather than outstanding — if a future session revisits this, check whether
  variance ranking still misleads post-winsorization before adding IQR
  ranking on top.
- Also new since this file's original writing: `diversity_metrics_by_group`
  (`diversity.py`) and `redundancy_by_group` (`matching.py`) compute the same
  diversity/redundancy metrics **per subgroup** (e.g. per generator/archetype)
  against a corpus-wide scaler, specifically so a subgroup that's collapsed
  doesn't hide behind healthy corpus-wide numbers. `report.py`'s
  `_by_group_summary` also names each group's `top_variance_feature` so a
  collapsed-looking group can be told apart from a group that's legitimately
  dominated by one real axis of variation.

---

## 6. Part C — `tsfm-lens` (the analysis half)

### 6.1 The layered method

No single measurement supports "these models share structure." Each level asks a
stronger question and exists **because of the previous level's limitation**.

| Stage | Question | Method | Limitation inherited |
|---|---|---|---|
| **L0** | Who is better, where? | MASE/sMAPE/pinball per family, paired bootstrap, Holm-corrected | Behavioral only |
| **Screen** (`layer_screen`) | Which of *this* model's own layers are worth further, expensive analysis? | Residual-trajectory work+bend geometry (default `work_bend`), or coverage/factor-emergence (§6.1.1) | Cheap proxy for interestingness, not interestingness itself |
| **Profile** (`internals`) | What is in each model? | Effective dimensionality, family-probe decodability, CKA-to-input per layer | Per-model, descriptive |
| **Lens** | Where in depth does the forecast form? | Skip lens + tuned ridge readout; crystallization depth | Depth-resolved, not component-resolved |
| **L1** | Do representations share geometry? | Linear CKA (global + per-family) w/ series-bootstrap CIs, RSA | **Correlational** |
| **L2** | Is shared geometry linearly translatable *beyond the input*? | Ridge stitching vs **input-feature baseline**; CI on the gain | Still not causal |
| **L3** | Where is structure causally carried? | Corruption sensitivity fingerprints + within-model patching (per layer **and per window**) | Within-model causality, compared across models |
| **Attention** | Which heads look where, and which matter? | Lag-profile taxonomy, periodicity heads, head/MLP mean-ablation ΔMASE, first-step cross-attention | Pattern support varies by architecture |
| **L4** (`clustering`) | How does each model organize the data? | Activation clustering + approximate labels, AMI across models | Descriptive |
| **Exemplars** | What does the difference look like? | Per-family case studies: forecasts, lens curves, attention maps | Illustrative, not statistical |
| **Confirm** | Which dev findings are real? | One-shot re-test of dev hypotheses on sealed **private** corpus | The gold standard |

Plus `extract` (upstream) and `report` (downstream). 13 stages total in the smoke
run (12 plus `layer_screen`, added 2026-08-05 and enabled by default); **11
report sections**, **22 findings** as last measured live (`configs/smoke.yaml`
via the actual CLI, not just the test suite).

**Why this ordering exists:** an early session explicitly argued down a proposal
to put SAEs first. Training good SAEs on two models × multiple layers is the
most expensive and failure-prone item available, and SAE features are *not
automatically comparable across models*. SAEs are correctly **phase 2**, after
CKA/stitching identifies which layers are worth the investment. Likewise, "is
one model better at high-frequency data" is a *behavioral* question needing zero
interpretability machinery — hence L0 exists as the hypothesis generator.

### 6.2 The `ModelAdapter` contract (`models/base.py`)

**Required:**
- `load()` / `module` property / `_release()`
- `prepare(contexts)` → model-specific prepared input
- `forward(prepared)` — runs the capture pass
- `token_time_spans()` → `[n_tokens, 2]` array mapping each token to a
  **contiguous time interval**
- `predict(contexts, horizon, quantiles)` → `{"point": ..., ...}`
- `default_layer_regex` over a residual stream emitting `[batch, tokens, dim]`

**Optional (degrade gracefully):**
- `postprocess_tokens` / `token_slice(live_len)` — for specials/padding.
  TimesFM front-pads, so its patches are the *trailing* ones; Chronos strips a
  trailing EOS.
- `attention_info()` → per-block `{block, o_proj, n_heads, head_dim}` — enables
  head ablation
- `mlp_info()` → block → MLP module name — enables MLP ablation
- `attention_patterns(prepared)` → block → `[B, H, T, T]`
- `cross_attention_patterns(prepared)` → `[L, B, H, T_enc]` (first decode step)
- `final_block_name()`, `all_layer_names()`

**Untrained-weights null baseline (`ModelConfig.random_init`, `ROADMAP.md`
§16 E9).** Any model config can set `random_init: true` to load the same
architecture (and, for Chronos, the same tokenizer/`chronos_config`) with
freshly, randomly initialized weights instead of the pretrained checkpoint.
Pair a model against its own `random_init` twin as an ordinary two-model
config (`configs/null_timesfm_random.yaml`, `null_chronos_random.yaml`) and
L1's CKA, L2's stitching gain, internals' probe decodability, and SAE
ground-truth alignment all get a real floor — how much of each is
"architecture + input statistics" rather than learning — with **no changes
to any of those four modules**; the null is just another pipeline run.
Implemented via each library's own from-config construction
(`models/base.py::random_init_like`: `type(model)(model.config)` for the
two HF-based Chronos adapters; TimesFM's wrapper already builds its full
architecture before any checkpoint step, so `random_init` there is simply
skipping that step), not a generic reinitialization heuristic — deliberate,
since e.g. HF's `T5LayerNorm` has no `reset_parameters()` and a
heuristic relying on it would silently leave that weight at its pretrained
value. `sae/ground_truth.py::permutation_null_alignment` is the
complementary label-permutation null this same item asks for, contextualizing
the SAE ground-truth alignment score's own multiple-comparisons inflation.

**Support matrix as built:**

| Capability | Chronos-T5 | TimesFM | Chronos-Bolt | Chronos-2 | Sundial |
|---|---|---|---|---|---|
| Capture / alignment / L0–L4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attention_info` / `mlp_info` | Hard-coded (`encoder.block.{i}.layer.0.SelfAttention.o`, `...layer.1.DenseReluDense`) | `attention_info` discovered via shared `_scan_attention` in `base.py`; `mlp_info` usually finds nothing (its feed-forward block is two bare `nn.Linear`s, `ff0`/`ff1`, with no wrapping MLP submodule for `_scan_mlp` to find — head ablation still works via the attention output projection) | Discovered | Discovered via the same shared `_scan_attention`/`_scan_mlp` — resolves to the block's TIME self-attention only (see below) | `attention_info` discovered via `_scan_attention` (`self_attn.o_proj`); `mlp_info` hard-coded (`{block}.ffn_layer` — `_scan_mlp` doesn't recognize that leaf name, same class of gap as Chronos-T5's `ff0`/`ff1`, worked around the same way: hard-code rather than extend the shared scanner) |
| `attention_patterns` | ✅ (`output_attentions`, EOS row/col stripped) | ✅ **(corrected — see below; was unsupported when this file was first written)** | ❌ deliberately (release fragility not worth it) | ✅ TIME self-attention only, context-patch positions only | ❌ deliberately — the checkpoint's own `output_attentions=True` path is independently broken upstream (`UnboundLocalError` in its remote code) and recovering weights would need a global `scaled_dot_product_attention` monkeypatch, judged riskier than TimesFM's local `attention_fn` swap; same reliability call as Chronos-Bolt |
| `cross_attention_patterns` | ✅ first step only | n/a | ❌ | n/a (encoder-only, no decoder) | n/a (decoder-only, no encoder-decoder split) |

> **Correction:** this file previously said TimesFM's `attention_patterns`
> was unsupported ("functional attention"). As of the TimesFM 2.5 adapter
> rewrite (`models/timesfm_adapter.py`) it **is** supported: TimesFM's
> `MultiHeadAttention` takes its dot-product implementation as a swappable
> `attention_fn(query, key, value, mask)` attribute — the default is a fused
> SDPA kernel with no exposed intermediate weights, but `attention_patterns`
> temporarily swaps in the library's own unfused dot-product math (operating
> on the same already-processed query/key/value) to recover the weights,
> then swaps the original back immediately after the one forward call this
> needs. Same computed output, just with the intermediate weights stashed
> on the way. ✅ **Checked 2026-08-06:** the stale `configs/default.yaml`
> comment this paragraph used to flag ("exposes no patterns (functional
> attention)") is no longer present — the file's `attention:` block comment
> already correctly describes `attention_patterns` support. Fixed in an
> earlier, undated session; this paragraph's own warning had gone stale.

**Sundial (`models/sundial_adapter.py`, added ROADMAP.md §9, Phase 4's
second multi-model-expansion addition, 2026-08-10).** `thuml/
sundial-base-128m`: decoder-only, 12 `SundialDecoderLayer` blocks,
`d_model=768`, 12 heads, patch=16, pretrained on ~1 trillion time points,
Apache-2.0 licensed. Uses a flow-matching ("TimeFlow Loss") head that
samples the whole forecast horizon in one shot from the last patch's
hidden state rather than autoregressive decoding — `predict()` maps its
native `(batch, num_samples, horizon)` sample output onto this repo's
`{"point": ..., quantiles...}` contract the same way `ChronosAdapter`
already does for Chronos-T5's own sampled decoding. **Two genuine upstream
breaks found and worked around, not fixed in this repo (the checkpoint's
own remote code, not shared infrastructure — see §11.22 for the
mechanism):** the HF card's documented `.generate()` path, and even a
single cached `forward()` call, both crash on any currently-supportable
transformers version because Sundial's own `prepare_inputs_for_generation`/
cache handling assumes a pre-4.41 `transformers.DynamicCache` API. Fixed by
calling `SundialForPrediction.forward(..., use_cache=False)` directly
instead of `.generate()` (sufficient since this repo's horizon, 64, is
well under Sundial's 720-token one-shot sampling limit) — **the shared
`cudaPy` env's `transformers==4.57.6` pin was left untouched**, avoiding
the §11.8-class risk of a version change silently breaking the other three
adapters. `--check-alignment` shows a real, diagnosed **amplitude-dependent**
diagonal-hit pattern (perfect 1.00 at every layer at small impulse
amplitudes, decaying by mid-depth at this repo's default 0.25× amplitude,
with zero backward/future leakage at any amplitude) — a genuine property of
Sundial's plain-residual, no-QK-norm blocks, not a broken `token_time_spans`
(§11.22 has the full diagnosis). No shared-infrastructure bug this time
(a real, useful contrast with Chronos-2 below) — Sundial's block returns a
plain tensor/tuple, already handled correctly by `hooks.py`. Live
comparison run vs. TimesFM: L1 peak CKA 0.381 (TimesFM `stacked_xf.4` ↔
Sundial `model.layers.11`) against a shuffled-series null of 0.022 —
decisive shared geometry with a third distinct architecture, replicating
the pattern Chronos-2 already established.

**Chronos-2 (`models/chronos2_adapter.py`, added ROADMAP.md §9, Phase 4's
first real multi-model-expansion addition).** Encoder-only — no decoder at
all. Each block runs TIME self-attention (within one series, the axis this
repo captures/exposes) then GROUP self-attention (across series sharing a
`group_id`, along the *batch* axis — not a token-token pattern, and
deliberately out of scope for `attention_info`/`attention_patterns` here)
then feed-forward. The encoder processes context + a `[REG]` token +
forecast-horizon placeholders **together in one non-causal pass** — there
is no separate decode step, unlike Chronos-T5's real encoder-decoder split
or Chronos-Bolt's encoder-plus-small-quantile-head design.
`token_time_spans`/`postprocess_tokens` keep only the leading context-patch
positions. `_scan_attention`'s "return on first match" behavior always
resolves to the TIME self-attention sub-layer (registered first in each
block), never GROUP — a real, stated scope limit, not a bug. Default
checkpoint `amazon/chronos-2` (120M params, `d_model=768`, 12 layers,
patch=16); `--check-alignment` shows a perfect 1.00 diagonal-hit fraction
at every layer.

**Adding a model:** subclass `ModelAdapter`, register in `models/__init__.py`,
run `--discover-layers` to pick a `layer_regex`, then `--check-alignment`.
Nothing downstream changes *in principle* — `chronos2_adapter.py` (Phase
4's first real test of that claim) needed zero adapter-specific
special-casing anywhere else, but did surface one genuine bug in shared
infrastructure (`extraction/hooks.py`, §11.21) that a Chronos-T5/Bolt/
TimesFM-only test surface had never exercised: fixed in the shared module,
not worked around per-adapter, exactly per this section's own contingency
plan for such a finding. `sundial_adapter.py` (Phase 4's second test,
§11.22) needed **zero** shared-infrastructure changes — a useful data
point that §11.21's finding was a real, now-fixed gap rather than a sign
every new architecture surfaces a new shared bug. `chronos_bolt_adapter.py`
is the compact worked example for a new adapter file itself.

### 6.3 Time alignment — the mechanism that makes cross-architecture comparison possible

The hurdle nobody escapes: TimesFM tokens = 32 timesteps, Chronos tokens = 1
timestep. **Raw token positions are not comparable.**

`extraction/alignment.py`:
- `pooling_matrix(spans, context_len, window)` → overlap-weighted, row-normalized
  `[n_windows, n_tokens]`. Raises if any window receives no tokens.
- `align(hidden, pool)` → `einsum("wt,btd->bwd")`, pooling `[B,T,D]` → `[B,W,D]`.
- **Everything cross-model operates on `[series, window, dim]`.** Default
  `window = 32` = TimesFM's `input_patch_len`, so patches map 1:1 to windows.
- Attention lag axes are multiplied by each model's `token_width` **before
  plotting**, so a plotted lag of 300 is the same physical 300 timesteps for both
  models (Chronos reaches it in ~300 tokens, TimesFM in ~9–10).

**`impulse_alignment_check`** — adapters *declare* spans; libraries *change*. An
input impulse in window *w* must perturb aligned window *w* most. Returns
per-layer diagonal-dominance fractions. Values below 1.0 at late layers are
**normal** (attention mixes positions); **near-zero everywhere means the declared
spans are wrong for your installed version — fix the adapter before trusting any
cross-model number.** `alignment.sanity_check: true` runs a cheap spot check
during extraction.

⚠️ **AUDIT (2026-08-06) — that spot check's result is thrown away; see
`ROADMAP.md` §15 A2.** `extraction/extract.py:50-51` calls the function,
ignores the returned dict, and probes only `layers[::len//4]`;
`alignment.py:84-86` logs one INFO line. There is no threshold, nothing
written to the run directory, and nothing in the report — so inside a
pipeline run the exact signal this paragraph says must block trusting a
number is one line among hundreds, followed by a complete-looking report.
The manual `--check-alignment` path *is* trustworthy (and is how §11.16 was
caught); the automatic one is decorative. Until A2 lands, treat
`--check-alignment` as mandatory-by-hand for every new checkpoint and every
library bump, exactly as invariant 7 says — the config flag does not
substitute for it. **Also: `extraction/store.py` writes activations as
`float16` with no finiteness check** (§15 A19), so an overflowing layer
enters CKA/ridge solves as `inf`/`NaN` rather than raising.

### 6.4 Extraction & storage
- `hooks.py`: `ActivationCatcher` (forward hooks) and **`token_patch`** — the
  single intervention primitive reused by L3, Lens, and (eventually) SAE feature
  ablation. It supports index functions, which is what made per-window patching
  nearly free.
  - Hardened: if the forward pass sees a **smaller batch than the replacement
    cache**, it raises with an actionable message ("the model is chunking
    internally — lower `l3.patching.max_series` to at most the model
    `batch_size`"). Patches only the replacement's rows.
- `store.py`: chunked **zarr v2** store (`zarr>=2.16,<3` — v3 changed the
  group/dataset interface). Activations stored **float16**. Integer-array row
  selection routed through zarr **orthogonal indexing** (the supported v2 path).
  `load(model, layer, level="series"|"window", rows=...)`.
- Forward passes batched under **autocast (bf16 default)**; CKA and ridge solves
  closed-form **on-device in fp32**.
- Models loaded lazily and released between stages
  (`run.keep_models_loaded: true` to keep both resident).
- VRAM knobs: `models[*].batch_size`, `capture_layer_stride`, `max_rows` caps in
  `l1`/`l2`/`internals`.

### 6.5 Stage details worth knowing

**L0** — MASE/sMAPE/pinball per family. Paired bootstrap of per-series MASE
differences, Holm-corrected across families. A "strength" requires corrected
p < alpha **and** a CI excluding zero. Ratios kept as effect sizes. Artifacts:
`l0/metrics.parquet`, `l0/summary.json`. Also computes **quantile
calibration** (`analysis/calibration.py`, `ROADMAP.md` §16 E10): reliability
curve (empirical vs. nominal coverage per quantile level, overall and per
family), an approximate PIT histogram, quantile-crossing rate, and interval
coverage/sharpness per horizon step — a pure reduction over the same
`predict()` output L0 already holds, so it costs zero extra forward passes.
`l0.calibration: true` by default; artifact `l0/calibration.json`, rendered
in the report's L0 section. This is where `CLAUDE.md` §12's forecast-
stochasticity asymmetry (Chronos-T5 sampled, TimesFM/Chronos-Bolt
deterministic) shows up on a quantile axis: sharpness differences between
models are expected from that asymmetry alone and aren't by themselves
evidence of better or worse calibration.

**Lens** (`analysis/lens.py`) — the highest-value addition of the last round;
the missing bridge between internals and L0.
1. **Skip lens** — layer-*l* token states patched in as the **final block's**
   output; the model's **own head** (for Chronos, the *entire decoder*) produces
   the forecast. Direct logit-lens translation; needs no head internals, only
   `token_patch`. Can be miscalibrated at early layers — which is what lens 2
   corrects.
2. **Tuned lens** — held-out ridge probe from stored window states to (a) the
   model's own final forecast and (b) true targets. High R² against the final
   forecast at layer *l* means the forecast is already *linearly readable* there
   even if the raw head can't decode it.
- **Crystallization depth** = first relative depth within `crystallization_tol`
  (default 10%) of final MASE.
- Assumes **uniform hidden size** across captured blocks; **fails loudly**
  otherwise. One batch per call → `max_series` must be ≤ every model's
  `batch_size`.
- Artifacts: `lens/curves.npz`, `lens/lens.json`.

**L1** — Linear CKA between **every layer pair**, global and per-family, plus
RSA. Cluster bootstrap at the peak pair. Artifacts: `l1/cka.npz`, `l1/meta.json`.

**L2** — 🔴 **The stitching baseline is load-bearing.** Both models saw the same
input, so "layer A predicts layer B" is *partly trivial*. A hand-crafted
input-feature probe (raw window + FFT magnitudes + summary stats) predicts the
same targets, and **only the gain above it is reported as evidence of shared
learned structure.** Both directions (A→B, B→A) reported. R² is
variance-weighted on a held-out **series** split — never a window split.
Artifact: `l2/stitching.json`.
⚠️ **Qualified by the untrained-weights null (2026-08-06, `ROADMAP.md` §16
E9; upgraded from an eyeball comparison to an actual significance test
2026-08-07).** The input-feature baseline controls for "both models saw the
same input"; it does **not** control for "any two networks of this shape
agree this much before either is trained." A paired bootstrap of the
*difference* between the real cross-model gain (0.413, this run's global
best pair) and each model's own untrained-twin floor confirms this is a
real null result, not just overlapping CIs: diff = −0.031 (95% CI
[−0.127, +0.061], p=0.474) against Chronos's floor, diff = +0.025 (95% CI
[−0.057, +0.102], p=0.532) against TimesFM's floor — both CIs comfortably
contain zero. **L1's peak CKA is not just "a related, model-asymmetric
version of the same gap" — the same paired-bootstrap test shows it as two
decisive verdicts pointing opposite ways**: the real cross-model CKA
(0.381) significantly *exceeds* Chronos's own untrained-twin floor (0.174,
diff +0.207, CI [+0.159,+0.217], p=0.001) while significantly *trailing*
TimesFM's own untrained-twin floor (0.599, diff −0.218, CI [−0.251,−0.180],
p=0.001) — which floor you check the one shared number against changes the
answer, cleanly, in both directions. Read "gain over baseline shows shared
learned structure" as **verdict depends on which layer of the null run you
compare against** for both L1's and L2's peak-pair numbers (see the two
depth-curve corrections immediately below — this superseded an earlier,
narrower "not yet established for L2" reading of the single-peak-pair test)
— all now backed by `analysis/null_baseline.py`'s bootstrap tests
(`ROADMAP.md` §16 E9's Findings has the full numbers), not just point
estimates. Still scoped to one corpus, one Chronos checkpoint size, and (for
L2) one direction — cross-corpus/size replication and the reverse L2
direction are the named next step. Does not apply to L0, L3, L4, attention,
or the SAE ground-truth alignment score, which the same null check
confirmed clearly survive it.
⚠️ **L1's peak-pair verdict above does not generalize to "L1 fails the
null" — read the depth curve, not just the peak (2026-08-07, `ROADMAP.md`
§16 E9's second follow-up).** Testing every TimesFM layer against both
models' own floor *at that exact depth* (not the null run's own global
peak, which sits at a different layer than the real cross-model peak)
shows real cross-model structure decisively beating **both** architecture-
only floors at every middle depth (layers 6–14 of TimesFM's 10 captured
layers), and losing to at least one floor only at the extremes — the
shallowest layers lose to TimesFM's own steeply-decaying floor, the
deepest lose to Chronos's own nearly-flat floor. The single global peak
pair (layer 4) happens to land exactly in the one genuinely ambiguous
(CI-spans-zero) spot against TimesFM's floor, which is why the peak-pair
test alone reads more pessimistically than most of the depth range
actually supports.
⚠️ **L2's single-peak-pair null result above is corrected, not just
qualified, by its own depth curve (2026-08-07, `ROADMAP.md` §16 E9's third
follow-up) — the earlier "statistically indistinguishable from null" verdict
was an artifact of comparing the real run's peak against the wrong layer of
the null run.** `compare_l2_best_gain`'s peak-pair test compared the real
run's best pair (Chronos block 10, gain 0.413) against `null_chronos_
random`'s own *global* best pair, which the depth curve reveals sits at
block 0 — an early layer where an untrained twin trivially predicts its own
same-index untrained twin (self-vs-random-twin gain 0.444) because both are
still close to the raw input, not a layer comparable to where the real
cross-model signal actually peaks. Retested at the **matching layer index**
(block 10 in both the real and null runs): the null floor there is 0.062
(Chronos-side) / 0.046 (TimesFM-side), and real decisively, overwhelmingly
exceeds both (diff +0.351 and +0.367, p=0.002 each) — real gain beats both
floors at every Chronos block from 4 through 11 (8 of 12), losing only at
the earliest blocks where absolute gain is smallest anyway. At the exact
layer pair this file already reports as the flagship L2 number, the gain
now clearly survives the null.

**L3** (`analysis/l3_perturbation.py`) — corruption battery, each targeting one
structural property: `noise` (fixed SNR dB), `detrend`, `deseasonalize`
(notch top-k spectral peaks + neighbors, DC excluded), `frequency_shift`
(resample time axis), `level_shift`, `spike`, `smooth` (uniform filter), plus
two added since this file's original writing: `warp` (local nonlinear time
warp — resamples through a smooth random monotone map, stretching/compressing
different parts of the series by different amounts to corrupt local
phase/alignment while leaving global frequency content roughly unchanged on
average, distinct from `frequency_shift`'s uniform global rescale) and
`dropout` (blanks out `n_blocks` random contiguous spans with the series'
own mean, simulating missing/intermittent data rather than any
continuous-signal distortion). Unknown corruption names raise.
1. **Sensitivity fingerprints** — per-layer relative activation change per
   corruption → `[layers × corruptions]` signature. Reuses the extraction store
   for clean activations, so each corruption costs *one* forward pass per model.
2. **Cross-model fingerprint agreement** — interpolated onto a shared
   relative-depth axis, rank-correlated per corruption.
3. **Within-model activation patching** — cached clean token states written back
   into a corrupted forward, one layer at a time; forecast restoration localizes
   where the property is causally carried. Now also **per-window**: layer×window
   restoration heatmaps, each token patched **exactly once** via span-midpoint
   assignment, sharing the full-layer damage denominator.
- Artifacts: `l3/sensitivity.npz`, `l3/meta.json`, `l3/patching.npz`,
  `l3/patching.json`.

⚠️ **The corruption battery's strengths are not calibrated to be
comparable across corruptions** (verified 2026-08-04 against a real
`medium_run` — see `ROADMAP.md` §5's Findings for the numbers). `level_shift`
is a permanent multi-standard-deviation step over the back 40% of the
series — a much larger absolute perturbation than `spike`'s few one-step
outliers or `detrend`'s slope removal — so it dominates the linear-scale
"Behavioral sensitivity" report chart regardless of which model is
"intrinsically" more sensitive; a low bar there is not evidence a
corruption "does nothing." Two distinct reasons a corruption can show low
aggregate sensitivity, and neither is a bug: (1) genuine model robustness
(`detrend` perturbs the raw input by as much as `noise`/`deseasonalize` do,
yet both real checkpoints' activations changed the *least* under it of any
corruption in the battery — consistent with forecasting from recent local
level rather than committing to an extrapolated trend), or (2) sparse input
footprint by construction (`spike` touches ~0.6% of timesteps, `dropout`
~14% — a small *average* activation/behavioral change is arithmetically
expected regardless of model, since most of the series is untouched; check
the per-window patching heatmap, not the aggregate bar, for whether the
touched region's damage is still causally recoverable). The report's own
`_note()` blocks state this, and the Behavioral sensitivity chart prints
each bar's value as text specifically so a reader isn't relying on bar
height alone. **Do not drop these corruptions for showing weak aggregate
signal** — the battery's value is contrastive (which properties a model is
comparatively robust vs. sensitive to), and a genuinely low-signal
corruption is the "null" end of that contrast, not dead weight.

**Attention** (`analysis/attention.py`) — family-conditioned lag profiles, head
taxonomy, **periodicity heads** (the induction-head analog: excess attention mass
at seasonal-lag multiples above a mask-fraction baseline), head/MLP
mean-ablation ΔMASE (overall and per family, `top_k` most load-bearing), and
first-step decoder cross-attention. `_dominant_period` via autocorrelation peak
beyond `min_lag`.
⚠️ Memory: pattern capture holds `[batch, heads, T, T]` per layer. For
chronos-t5 at T=512 that's **~2.4 GB at `batch_series: 16`** — lower
`batch_series` before lowering `max_series`.

**L4 / clustering** — activation clustering at `layer: auto` (each model's side of
the L1 peak-CKA pair), `k: auto` (number of families, clipped 2..20), PCA to
`pca_dim`, optional UMAP for the map. Cluster labels = majority family + salient
signal statistics. **Approximate by construction** — hypotheses for reading the
maps, not feature attributions. AMI across models with a series-bootstrap CI.

**Exemplars** — per-family series picked from the L0 MASE-gap distribution
(**max-gap + median-gap**), showing both forecasts, per-series lens curves, and
window-pooled attention maps.

**Confirm** — see §6.7.

**Report** (`report/report.py`) — one self-contained interactive HTML (jinja2 +
plotly). Sections are built from a `builders` list of
`(eyebrow, title, blurb, requires, build_fn)`. Each declares its **required
artifact paths**; missing artifacts → *silent* skip logged as "stage not run",
a raised exception → **warning** logged as "section failed". This distinction is
deliberate so logs stay honest. Findings are accumulated into a list and
surfaced as prose. Almost every figure carries an attached `_note()` block
(purpose / reading values / limitations) rendered as a collapsed-by-default
`<details>` directly under it — this is the report's actual glossary
mechanism, contextual per-plot rather than a separate lookup table, added
before `ROADMAP.md` existed and easy to miss if you don't open the file.
A fixed **"How to read this report"** preamble (`_how_to_read`) renders
first, unconditionally, stating the evidence-class ladder end to end and the
one canonical definition of "window" and "relative depth" that recur in
nearly every section. `report.verbose` (default `false` as of 2026-08-10, ROADMAP.md §10/§16's
"closer to release, verbose stays opt-in" call — was `true`; `run.py --verbose`/
`--no-verbose` overrides it per-run) additionally gates a narrated
single-series case-study section for **L3 patching**, matching the
`Exemplars` section's pattern: for `report.verbose_series` (default 3)
series per corruption, actual context/true-continuation/clean/corrupted/
patched forecasts next to that series' own layer×window restoration grid.
The patched forecast shown uses the single (layer, window) cell that
restored the most on average across the whole sampled batch for that
corruption — not each series' own individually-best cell, which the report
states explicitly — recovered from the same forward passes the aggregate
`l3/patching.npz` arrays already require, at no extra model-call cost.

### 6.6 Statistical discipline

🔴 **The resampling unit is always the SERIES.** Windows within a series are
strongly dependent, both models score the same series, and corruptions are
applied to the same series.

- L0: **paired** bootstrap of per-series MASE differences, Holm-corrected.
- L1: **cluster bootstrap** (resample series, keep their windows together).
- L2: CI on the best **gain**, resampling validation series with **fitted probes
  held fixed** — this is *evaluation* uncertainty, deliberately not refit
  variability.
- L3: per-series deltas retained so fingerprint agreement gets a **paired**
  cluster bootstrap (same series resample applied to both models).
- Profile: probe accuracy CIs over held-out series against a majority-class
  chance line.
- Bootstrap p-values are **approximate and floored at 1/n_boot** — decision
  aids, not precision instruments.
- `stats.n_boot` vs `stats.n_boot_heavy` cap cheap and expensive statistics
  separately. Bootstraps add analysis-time cost only, never extraction cost.

### 6.7 Exploration vs confirmation discipline

Everything on the **dev (public)** corpus is exploratory — many comparisons are
looked at, so significant results are **hypotheses**. The `confirm` stage tests
them **exactly once** on the sealed **private** corpus and reports verdicts.
That section is the evidence; the rest is context.

**The private corpus is a consumable.** The stage:
- refuses to overwrite existing confirmation artifacts,
- requires seal verification by default (`confirm.require_seal: true`),
- logs loudly when it runs.

If the private set is ever peeked at, **regenerate a fresh private epoch** rather
than reusing it. Findings not pre-registered as dev hypotheses **cannot be
rescued** by the confirm stage — by design.

⚠️ A smoke-mode confirm is *mechanically* real but **distributionally identical**
to dev (same generators, shifted seed). Genuine confirmation value requires
sealed private corpora.

---

## 7. Invariants — do not break these

1. **Sealed-corpus bit-exact regeneration.** Any change touching RNG draw order
   in generators breaks previously sealed corpora. New archetypes/options are
   **opt-in**. Golden-hash regression tests pin this (§11.1).
2. **The series is the resampling unit.** Never bootstrap or split on windows.
3. **L2 reports the gain over the input-feature baseline**, never raw R².
4. **Diversity is computed in feature space, never on UMAP coordinates.**
5. **Patching is within-model.** Cross-model activation transplants are not
   meaningful without a learned mapping. What is compared is each model's
   restoration-by-depth *curve*.
6. **`confirm` runs once**, on frozen analysis, against a sealed private corpus.
7. **Alignment is verified empirically** (`--check-alignment`) on every new
   library version before any cross-model number is trusted. ⚠️ **Enforced by
   human memory only** — the in-pipeline check discards its own result
   (`ROADMAP.md` §15 A2, and the marker in §6.3). Run the CLI check by hand,
   every time, until A2 lands.
8. **Unsupported capabilities skip and log; broken assumptions fail loudly.**
   Never silently produce a wrong slice. ⚠️ **"Loudly" is currently
   implemented as a log line in most places, which is not loud once the
   deliverable is an HTML file** — the report omits skipped and *failed*
   sections with no trace in the HTML (`ROADMAP.md` §15 A5), a mis-resolved
   `family_key` silently regroups every per-family statistic (§15 A6), a
   configured sample cap is silently clamped to the model's batch size
   (§15 A16), and `layer_screen` can fall back to the very null it was
   supposed to beat without the report saying so (§15 A1). The doctrine is
   right; twelve of §15's nineteen items are places the implementation
   doesn't meet it yet.
9. **Evidence class is stated in the report** — geometric / translatable /
   causal-within-model / descriptive / illustrative.
10. **Real data never enters the benchmark directly** — only as leakage reference
    and realism calibration.
11. **No hardcoded absolute paths, anywhere.** Every file path in code or
    config must be relative (to the repo root, to the invoking script's own
    `__file__`, or to a CLI-supplied argument) so the repo runs unmodified on
    any machine. `sys.path.insert` calls already do this correctly via
    `Path(__file__).resolve().parents[N]` — follow that pattern, not a
    literal string. Config YAML `path:`/`out_dir:` fields must be relative
    to the directory the tool is documented to be run from (e.g.
    `tsfm_lens/configs/*.yaml` paths are relative to `tsfm_lens/`, matching
    §8's documented `python run.py --config configs/...` invocation).
    Verified clean repo-wide 2026-08-04 after finding and fixing one
    violation (`tsfm_lens/configs/medium_run.yaml`'s two `path:` fields, was
    a machine-specific absolute path) — re-check with a repo-wide grep for
    drive letters / `/home/`/`/Users/` after any config or script edit.
12. **`DEPENDENCIES.md` is kept current.** Any change to a pinned library
    version (a new pin, a loosened/tightened bound, a newly-discovered
    fragile version interaction) gets reflected there in the same session —
    it is the authoritative "what's actually installed and verified working"
    record, and `CLAUDE.md` §11's traps already show how expensive it is to
    rediscover a version-fragility issue that isn't written down anywhere.

---

## 8. Configs and CLI

### `tsfm-lens`
```bash
# Smoke run: no downloads, no GPU, ~minutes. Validates every stage.
python run.py --config configs/smoke.yaml
open runs/smoke/report.html

# Real comparison
python run.py --config configs/default.yaml --check-alignment timesfm
python run.py --config configs/default.yaml --check-alignment chronos
python run.py --config configs/default.yaml

# Stage selection / reruns (stages self-skip when artifacts exist)
# WARNING: the skip is a bare path-existence check with NO config fingerprint
# (ROADMAP.md §15 A3). If you edited anything a skipped stage consumes --
# context_len, alignment.window, a checkpoint, data.path -- and rerun into the
# same run.name, you get last config's artifacts feeding this config's
# analyses, with a report that looks complete. Until A3 lands: change a config
# => change run.name, or --force all.
python run.py --config configs/default.yaml --stages extract,l1,report
python run.py --config configs/default.yaml --stages l2 --force l2

# Only after analysis is FROZEN:
python run.py --config configs/default.yaml --stages confirm,report

# Adapter development
python run.py --config configs/default.yaml --discover-layers

# Cross-run comparison (ROADMAP.md §5.5): reads N existing run directories'
# artifacts as-is, writes one comparison JSON + HTML, re-runs nothing.
python run_meta_report.py --runs runs/medium_run,runs/medium_run_chronos_base
```

**Key `default.yaml` values (corrected — TimesFM moved to 2.5 since this file
was first written; see §11.8):** `context_len: 512` (multiple of
`alignment.window`, ≤ chronos-t5's 512), `horizon: 64`, `alignment.window: 32`,
TimesFM `google/timesfm-2.5-200m-pytorch` (20 layers, `capture_layer_stride: 2`
→ **10** capture points — not the 2.0/500M checkpoint's 50 layers/25 points
this file previously stated), Chronos `amazon/chronos-t5-base` (12 encoder
blocks, stride 1, `num_samples: 20`). `family_key: auto`. Every stage has an
`enabled` flag; all knobs documented inline in the YAML.
`configs/medium_run.yaml` gives a mid-scale real-model config between the
mock-only smoke run and this full default.

Notable stage knobs: `l3.patching.per_window: true`,
`l3.patching.window_stride: 2` (every window doubles patching cost);
`lens.crystallization_tol: 0.1`; `attention.batch_series` (memory-critical);
`layer_screen.enabled: true` (default on; `method: work_bend` is the
§6.1.1 bake-off winner, `budget_frac`/`min_budget` size the per-model
selection its `selection.json` writes — ⚠️ but it screens only the layers
`capture_layer_stride` actually captured, so with the stride 2 both
`default.yaml` and `medium_run.yaml` use for TimesFM, half that model's
blocks are never scored; the bake-off ran at stride 1. `ROADMAP.md` §15 A1
has the fix; until then set `capture_layer_stride: 1` if you intend the
screen's selection to mean what its own design says it means); `sae.enabled: false` (default off;
`sae.targets: []` resolves via `layer_screen`'s selection ("auto") when
left empty, or `[{model, layer}, ...]` to pin explicit layers —
`configs/medium_run_chronos_base.yaml` has a real worked example, pinned
rather than auto so it keeps reproducing its already-documented result);
`sae.dict_size_mult`/`k`/`epochs` size the baseline `TopKSAE` when on;
`report.verbose: false` (default as of 2026-08-10, was `true`) / `report.verbose_series: 3`
(`--verbose`/`--no-verbose` on `run.py` overrides per-run — §6.5).

### `build_pipeline`
```bash
# CLI scripts live in tsfm_benchmark/example_runs/, not examples/ — run from
# tsfm_benchmark/ or adjust the path accordingly.
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml \
  --out ./benchmark_out --references monash --threshold 0.35 --epoch 0
# fast check: --max-count 15
```
Produces `benchmark_out/public_dev/` and `benchmark_out/private_test/`, each with
`corpus.jsonl` + `manifest.json`. Volume ≈ `sum(count) × 2 splits` per epoch.

**Correction:** this file previously said real-derived tasks are skipped by
`run_full.py` and must be built by calling the generators directly in code.
That's stale — a task with a `source_config` (see `configs/example.yaml`'s
fully-annotated reference and `configs/full_multidomain.yaml`) is loaded and
injected entirely from YAML; `BenchmarkBuilder` loads and caches each task's
source pool itself (`_load_cached_sources`). Verified live 2026-08-03 for all
three real-derived generators (`mixture`, `block_bootstrap`, `sequential_par`)
against Monash and ETT — see §4.3, §11.9–§11.12, `ROADMAP.md` §5. A fast
end-to-end check of the whole real-data path lives at
`configs/real_data_smoke.yaml` (~1 minute; `configs/full_multidomain.yaml` is
the real, expensive build).

### `benchmark_validation`
```bash
# run from tsfm_benchmark/ — run_validation.py lives in example_runs/, not examples/
PYTHONPATH=. python3 example_runs/run_validation.py \
  --corpus ./benchmark_out/public_dev --out outputs
# demo mode needs the generation package on the path:
PYTHONPATH=. python3 example_runs/run_validation.py
```

---

## 9. Verification status

| Suite | Status |
|---|---|
| `tsfm_benchmark/tests/` | **41 passed / 1 skipped** (see golden-hash row below — corrected 2026-08-06, was reported as 1 failing) — includes `test_real_derived_generators.py` (2026-08-03), +1 test same-day follow-up for `sequential_par`'s length-truncation fix (§11.14), +4 tests 2026-08-06 (a golden-hash split test + 3 seal-determinism tests, `ROADMAP.md` sec 15 A8) |
| `tsfm_lens/tests/test_smoke.py` | **Green end-to-end**, all 12 stages, artifact shape checks, report token checks (10 sections, 18 findings) — **corrected 2026-08-03**: this had never actually passed against the package's own pinned zarr version until this session (§11.15) |
| Golden-hash generator regression | ✅ **Currently PASSING against numpy 2.1.0** (re-verified 2026-08-06, `ROADMAP.md` sec 15 A8 — corrects this row's prior "currently FAILING" claim from 2026-08-03). The 2026-08-03 failure report is **not reproducible** in this persistent environment on the same numpy version; a direct comparison of `Generator.choice(..., replace=False)` and `.normal()` between numpy 1.26.4 and 2.1.0 in an isolated venv found them bit-identical, refuting (not confirming) the leading cross-version-instability hypothesis. Root cause of the original 2026-08-03 report remains unexplained — see §11.13's update. |
| Modularity check | Partial run (`extract,l1,report`) renders **only** the L1 section |
| `compileall` on both packages | OK |
| Confirm hypothesis-path test | Feeds synthetic dev claims + one real / one spurious effect; asserts verdicts `{trend: True, spiky: False}` |
| Real-data path (`mixture`/`block_bootstrap`/`sequential_par`, Monash + ETT) | ✅ **Verified live end-to-end 2026-08-03**, then run at ~4200-sequence scale the same day (`configs/full_multidomain_run1.yaml` + `benchmark_validation`) — see §4.3, §11.9–§11.12, §11.14, `ROADMAP.md` §5 |
| `tsfm_lens` against real checkpoints (TimesFM 2.5, Chronos-T5) | ✅ **Verified live end-to-end 2026-08-03** (third session) — full `medium_run.yaml` pipeline, 12 stages, real GPU (RTX 5070), ~10.5 min. Found and fixed two real bugs along the way (§11.15–§11.16); see `ROADMAP.md` §5.4 |
| Layer-screening bake-off (`layer_screen.py`/`layer_screen_bakeoff.py`, ROADMAP.md §6.1.1) | ✅ **First real run 2026-08-05** against live TimesFM 2.5 + Chronos-T5-Small, TimesFM captured at all 20 layers for the first time (not stride-2's usual 10); replicated across two independent SAE seeds (gold-ranking Spearman stability ρ=1.0/0.926). **Provisional result, but wired into production the same day** on explicit user direction: `work_bend` now runs as a default `layer_screen` pipeline stage ahead of `sae` in every config. Extended to a third architecture (Sundial) 2026-08-10 — `work_bend` beats both nulls there too, but the three-architecture picture is "beats a different subset of the two nulls on each," not a clean sweep; `factor_emergence`'s early-layer-bias failure reproduces cleanly on Sundial. ✅ **Root-caused same day:** a fresh re-extraction of the nominally-identical Chronos-T5-Small config had flipped its qualitative verdict (`work_bend` beats-random True→False) while TimesFM's reproduced unchanged — traced not to nondeterminism but to `ROADMAP.md` §15 A4's corpus-sampling fix landing *between* the two runs' dates: the pre-fix run's `max_series: 220` head-sliced a 288-row corpus and silently excluded the entire `mixture` family, so the two runs were never actually comparable populations despite byte-identical YAML. See §11.23–§11.24. ⚠️ **That narrower question is now answered, same day: no, not reliably.** A second sampling seed under the corrected, family-stratified sampler (76.8% row overlap with seed 0 — not a biased or adversarial draw) still flips `work_bend` — the production default — `beats_random` verdict on 2 of 3 architectures (TimesFM True→False, Chronos False→True); `coverage` stays fully stable. In every flip case the selector's own chosen layers are bit-identical across seeds — what moves is the gold-ranking/null-comparison scorecard, not the selection itself, so this reads as a statistical-power gap in the bake-off's evaluation (recall@budget over 220 rows, budget≤5) rather than a bias like A4's. ✅ **Fixed and re-verified same day, on explicit instruction not to leave this as a documented-but-unfixed gap.** Root cause was narrower than "the bake-off in general lacks power": `select_work_bend`/`select_coverage` take no `seed` argument and are fully deterministic given a fixed corpus (confirmed by reading `layer_screen.py` directly), so all the noise traced to `build_gold_ranking` (`layer_screen_bakeoff.py`) training exactly one stochastic SAE per layer as the gold reference. Fixed by (1) averaging `n_replicates=3` independently-seeded SAE-training runs per layer into the gold score instead of trusting one (`run_layer_screen_bakeoff.py --n-gold-replicates`), and (2) new `configs/layer_screen_experiment_v4[.yaml/_seed1.yaml]` dropping the `max_series: 220` corpus cap entirely (288 rows is small enough to use in full, closing the sampling axis of the noise too). Re-verified live against all three real checkpoints on both seeds: every `beats_random` verdict now matches across seeds for all three selectors on all three architectures — `work_bend`'s two prior flips (TimesFM, Chronos-T5-Small) are both closed. Full numbers in `ROADMAP.md` §6.1.1's Findings (fifth block) and §13. |
| `layer_screen` pipeline stage + `sae.targets: auto` wiring (`config.py`, `pipeline.py`, `sae/train.py::_default_targets`) | ✅ **Verified 2026-08-05** — full `tsfm_lens` suite 34/34 (33 prior + 1 new asserting `_default_targets` resolves exactly what the stage selected), plus a live CLI run of `configs/smoke.yaml` (not only pytest) confirming a real 11-section/22-finding report and a sane `layer_screen/selection.json` for both mock architectures. |
| Crosscoder feasibility test (`sae/crosscoder.py`, ROADMAP.md §13/§6.2) | ✅ **First run 2026-08-05** against live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-base` activations (an already-extracted store, no new model calls) at their L1 peak-CKA layer pair. Joint training is stable (no source-domination collapse across three hyperparameter settings) once a real, found-and-fixed scale-domination instability (§11.19) and a device-mismatch crash are corrected. **Not** an `SAEAdapter` implementation or a pipeline stage — feasibility-gate only. Full numbers in `ROADMAP.md` §6.2's Findings. |

**Golden hashes (do not let these change) — ✅ currently matching, see below:**
```
seed 0   → parametric 2856658d044e4c49  random a45664e176fbb71a  clean_low_noise
seed 7   → parametric ded6ff4ee4d08e00  random a1d7456a6daeb78e  noisy_chaotic
seed 123 → parametric 13157fdc8501e113  random 8b8395cd08aced50  trend_dominant
```
**Correction (2026-08-06, `ROADMAP.md` sec 15 A8).** The paragraph below
described a 2026-08-03 failure (seed 0 producing `4070db4646799f09` against
numpy 2.1.0) as reproducible and unfixed. Re-run 2026-08-06 in this
persistent environment (also numpy 2.1.0):
`test_golden_hashes_pre_extension_outputs_unchanged` **passes**, matching
every hash above exactly. The leading hypothesis named below was checked
directly rather than left as a guess — `Generator.choice(..., replace=
False)` and `.normal()` were compared bit-for-bit between numpy 1.26.4 and
2.1.0 in an isolated venv and found **identical** across both, which
refutes rather than confirms it. The original 2026-08-03 report's exact
cause is still unexplained (a different point-release, platform, or BLAS
backend are the remaining candidates), but as of now this invariant is
green, not red, and no code in `generators.py` was changed to make it so.
Preserved verbatim below per this file's own no-deletion doctrine (§0.2-style):

As of 2026-08-03, `test_golden_hashes_pre_extension_outputs_unchanged` fails
against numpy 2.1.0 (seed 0 produces `4070db4646799f09`, not
`2856658d044e4c49`) — reproducibly, in an environment where `parametric`/
`random_parametric` were **not touched**. Leading (unconfirmed) hypothesis:
`rng.choice(..., replace=False)` (used for changepoint/anomaly indices) isn't
guaranteed algorithm-stable across numpy versions the way `rng.normal()` is,
unlike what invariant 1 (§7) assumes. See §11.13 — not yet root-caused or
fixed, flagged here so it isn't mistaken for a currently-passing test.

✅ **Resolved 2026-08-03 (was: untested against live weights).** Chronos
`output_attentions` and TimesFM's SDPA-unfusing module scan were
compile-checked and defensive but never run on real checkpoints. Now run
live end-to-end against `amazon/chronos-t5-small` and
`google/timesfm-2.5-200m-pytorch` (`ROADMAP.md` §5.4): both produce sane,
non-degenerate outputs — TimesFM's `future_mass` head-score is exactly 0.0
at every layer/head (no attention leaks to future positions, as required
for a causal decoder — a strong sanity check beyond "doesn't crash"), and
Chronos's first-step decoder cross-attention consistently peaks at lag 0
across every layer. Getting to a trustworthy result required fixing two
real bugs first, not just running the existing code — see §11.15–§11.16.

---

## 10. Findings from real-model runs — moved to `ROADMAP.md`

**Retired from this file 2026-08-05, per `ROADMAP.md` §5.5's own instruction**
("retire it from `CLAUDE.md` ... since `CLAUDE.md` should describe stable
architecture, not a moving research result"). This section used to hold a
single small run's provisional findings; those numbers are still real and
are **not deleted** (this repo's own doctrine forbids that), just relocated
to where a growing, multi-run research record belongs — `ROADMAP.md` Phase 1
§5's Findings block, appended in chronological order alongside every later
run's numbers (the §5.4 live-checkpoint session, the §5.3 chronos-base-vs-
-small size comparison, and whatever `run_meta_report.py`'s cross-run
aggregator — `tsfm_lens/report/meta_report.py`, §5.5 — surfaces next). Read
findings there, not here: this file describes what the pipeline *is*, not
what it has *found* on any particular run, and a findings section here would
go stale exactly the way this one did.

---

## 11. Traps — mistakes already made, and what they cost

### 11.1 Growing a default list breaks sealed-corpus reproducibility 🔴
Adding 4 archetypes preserved per-archetype draw order but still broke
`random_parametric`, because `rng.choice(len(names))` changes for **every
existing seed** when the pool grows. Sealed corpora would no longer regenerate.
**Fix:** freeze the default at the original 8; new ones opt-in; golden-hash
regression test so it can't regress silently. `_DEFAULT_ARCHETYPES` carries an
explanatory comment — do not "tidy" it.

### 11.2 `hash()` is salted per process
Seed mixing originally used Python's builtin `hash()`, making builds
non-reproducible **across runs**. Now `hashlib.sha256`-based (`_stable_offset`).

### 11.3 Guessing the cause instead of measuring it
The `CO_trev_1_num` variance blowup was assumed to be input amplitude. Z-scoring
disproved it (pycatch22 z-scores internally). Real cause: extreme isolated
spikes dominating cubed differences. **Measure first.**

### 11.4 zarr v2 indexing and pandas buffers
Integer-array row selection needs zarr **orthogonal indexing**. And
`torch.from_numpy(group.index.to_numpy())` needs `.copy()` for a writable buffer.

### 11.5 Silent batch chunking during patching
Forecasting paths can pad or chunk the batch, silently misaligning the patch.
`token_patch` now raises with a fix instruction. Keep `max_series ≤ batch_size`
for every patching/lens stage.

### 11.6 Report key-name drift
A findings string referenced wrong `top_heads` keys and only surfaced in the
smoke test. Report builders are the least type-safe surface in the repo — run the
smoke test after any report edit.

### 11.7 Overstating the pipeline's shape
A draft progress update described tsfm-lens as "4 layers." It is a ~10-stage
pipeline. Describe it accurately — it's *more* progress than the shorthand
suggests.

### 11.8 A pinned model version can drop its own API out from under you
The `timesfm` PyPI package's newer releases (`>=2.0`) **removed the old
`TimesFmHparams`/`TimesFmCheckpoint`/`TimesFm` class API entirely**, and no
release of that package compatible with Python ≥3.12 still exposes it. This
wasn't a choice made in this repo — the adapter had to be rewritten
(`models/timesfm_adapter.py`) against the new `TimesFM_2p5_200M_torch`
class-based API to keep working at all, which also changed the default
checkpoint (`google/timesfm-2.5-200m-pytorch`, 20 layers, not the earlier
2.0/500M checkpoint's 50) and, as a side effect of the new library's cleaner
`attention_fn` seam, *added* a capability (`attention_patterns`) that was
previously unsupported (§6.2). **The lesson:** a dependency's own breaking
release can silently invalidate hardcoded assumptions (layer counts,
checkpoint names, capability matrices) documented elsewhere in this file
without any code in this repo changing first. After bumping *any* TSFM
library dependency, re-run `--discover-layers` and `--check-alignment`
before trusting anything downstream (`CLAUDE.md` invariant 7) — don't assume
last session's capability matrix or config values still hold. An older
`timesfm` install pinned to the pre-2.0 Hparams API would need the old
class-based loader instead of this adapter; that path is not maintained here.

### 11.9 `datasets>=3` removed script-based dataset loading entirely
`Monash-University/monash_tsf` is script-backed (`monash_tsf.py`). A modern
`datasets` install (5.0.1 at time of testing) raises `RuntimeError: Dataset
scripts are no longer supported` — not a `trust_remote_code` or network
problem, a hard removal. This is why the Monash path went unverified for as
long as it did: anyone trying it with a contemporary `datasets` would hit
this immediately. **Fix:** pin `datasets>=2.18,<3` (2.21.0 verified working,
`pyproject.toml`). Same lesson as §11.8: a dependency's own breaking release,
not a choice made in this repo.

### 11.10 `tsbootstrap`'s entire public API was rewritten
`block_bootstrap` was written against a class-based
`MovingBlockBootstrap(n_bootstraps=..., block_length=..., rng=...).bootstrap(arr)`
API that **does not exist** in tsbootstrap 0.7.1 — replaced by a functional
`bootstrap(X, method=MovingBlock(block_length=...), n_bootstraps=...,
random_state=...)` call returning a `BootstrapResult`, whose replicate array
comes back via `.values()` (a *method*, not a property — easy to silently
misuse as one) shaped `[n_bootstraps, n_obs]`. Fixed in
`generators.py:block_bootstrap`; re-verify this call site after bumping
tsbootstrap again, per the §11.8 lesson.

### 11.11 `sequential_par` was a stub; real Monash series can be shorter than a fixed `block_length`
Two related traps found implementing/exercising the real-derived generators
for the first time:
- `sequential_par` (SDV PARSynthesizer) was literally `raise
  NotImplementedError` until 2026-08-03. Implemented against sdv 1.14.0;
  giving PAR an explicit `sequence_index` column crashes
  `auto_assign_transformers` (`AttributeError: 'NoneType' object has no
  attribute 'enforce_min_max_values'`) in this version — PAR does not
  require one when each cohort series' rows are already in time order, so
  the fix omits it rather than working around the crash. `PARSynthesizer`
  also exposes **no `random_state`/seed parameter anywhere**, so this
  generator's reproducibility is best-effort (global numpy/torch seeding),
  not the bit-exact guarantee the rest of this module provides.
- Monash's yearly-granularity domains have series as short as ~14 points;
  bootstrapping across many domains at once (as `configs/full_multidomain.yaml`
  now does) means a fixed `block_length` sized for the longer domains can
  exceed a short one's length entirely — `tsbootstrap` raises
  `MethodConfigError` rather than truncating. `block_bootstrap` now clamps to
  `max(2, min(block_length, len(arr) // 2))` and records the effective value
  used in provenance, so a short real source degrades instead of crashing
  the whole task (§2.5's "degrade gracefully" doctrine, applied to real-data
  messiness this time rather than a missing capability).

### 11.12 `run_full.py`'s source injection assumed every real-derived generator takes the same kwarg
`load_specs` unconditionally set `generator_params["sources"] = sources` for
every generator in `_NEEDS_SOURCES = {"mixture", "block_bootstrap",
"sequential_par"}`, but only `mixture` accepts a `sources` list kwarg —
`block_bootstrap` takes `source` (singular, one tuple) and `sequential_par`
takes `training` (a cohort list). This was invisible because the only
generator ever exercised through this eager-resolution CLI path was
`mixture` (`tests/test_real_source_injection.py` only covers
`mixture_task`) — nobody had actually run `block_bootstrap`/`sequential_par`
through `run_full.py` before. Fixed to branch per generator, mirroring
`builder.py._make_one`'s existing (correct) branching. **Lesson:** a mocked
test that only exercises one member of a generator set can hide a real bug
in how the other members are wired — the fix here was found only by
actually building a corpus with all three generators, not by reading the
code or trusting the existing test's green checkmark.

### 11.13 An unrelated numpy/MKL ABI crash, and a newly-suspected numpy-version fragility in the golden-hash invariant
Two more things found while setting up a fresh environment for the above
(neither is a code bug in this repo, but both affect whether §7 invariant 1
and `benchmark_validation` can be trusted in a given environment):
- conda-forge's `numpy==2.1.0` Windows build (compiled ~Sept 2024) crashes
  with an unhandled SEH exception (`0xc06d007f`) on **any** matrix
  multiplication when a much newer MKL runtime (`mkl==2026.1.0`, pulled in
  transitively by `sdv`'s `pytorch` dependency) is present in the same
  environment — an ABI mismatch, reproduced with a bare `a @ b` call with no
  imports from this repo at all. This would silently break
  `benchmark_validation`'s matching/UMAP/diversity code (all matmul-heavy)
  on any fresh conda-forge setup that happens to pull a recent MKL. **Fix:**
  pin `mkl==2024.2.2` (matching numpy 2.1.0's build era) alongside it. Check
  for this on any future environment rebuild — it will not raise a Python
  exception, it hard-crashes the process.
- `test_golden_hashes_pre_extension_outputs_unchanged` (§7 invariant 1, §9)
  **fails against numpy 2.1.0** in a from-scratch environment with no change
  to `parametric`/`random_parametric` — confirmed via `git diff` scope
  before investigating. Deterministic (reproduces identically across
  repeated runs), so not nondeterminism; not yet root-caused against an old
  numpy install for comparison. Leading hypothesis: `parametric`'s
  `rng.choice(..., replace=False)` calls (changepoint/anomaly indices) rely
  on an algorithm numpy does **not** guarantee stable across versions the
  way it guarantees `rng.normal()`/`rng.standard_normal()`. If confirmed,
  invariant 1 is **numpy-version-fragile**, not purely a function of this
  repo's own code — a real gap in what "bit-exact regeneration" has actually
  been protecting against, since the golden hashes were apparently never
  checked against a *specific, pinned* numpy version before. See
  `ROADMAP.md` §5 Findings for the exact isolation steps tried and what's
  left to confirm this.

### 11.14 Diagnosed `sequential_par`'s cost against the wrong variable the first time
§11.11 recorded `sequential_par` as "expensive per series" without pinning
down *why*, and a same-day earlier attempt at a full-scale build blamed
`epochs` (64 vs. a smaller prototype's 8) after it failed to complete a
single fit in ~50 minutes. That diagnosis was never isolated and turned out
wrong: a controlled calibration in a follow-up session held one real Monash
cohort fixed (series lengths up to 40,720 points) and varied only
`epochs` (8 vs. 16) — both took ~1000s. The actual driver was that
`sequential_par` fed every raw timestep of every cohort series, completely
unbounded, into `PARSynthesizer.fit()`; SDV's DeepEcho RNN backend trains
per-timestep, so an untruncated 40k-point series costs roughly 1000x what a
512-point one does, regardless of epoch count. **Fix:** a new
`max_train_length` parameter (default 512, matching this repo's typical
`context_len` elsewhere) truncates each cohort series to its most recent
`max_train_length` points before fitting, and the auto-computed output
`length` (when not given explicitly) is now derived from the truncated
lengths too. Re-running the identical cohort with the fix
(`max_train_length=512`, epochs 32 vs. 64) measured ~5-6.5s per fit — a
~150-200x speedup, with epochs now correctly making almost no difference.
`n_series_truncated`/`max_train_length` are recorded in provenance
(mirroring `block_bootstrap`'s `effective_block_length`, §11.11).
**Lesson** (§2.4): the original attempt changed two variables at once
(epoch count *and* cohort series length, moving from a short synthetic test
to real several-hundred-to-tens-of-thousands-point Monash series) and
blamed the one that changed by a smaller factor. When something that
worked in a small test hangs at scale, isolate one variable at a time
before writing down which one is at fault — a plausible-sounding first
guess here was simply wrong. See `ROADMAP.md` §5 Findings for the exact
before/after timings and the first tractably-scaled real build
(`configs/full_multidomain_run1.yaml`) this fix unblocked.

### 11.15 `tsfm_lens`'s zarr storage layer had never actually run against its own pinned zarr version
`extraction/store.py` called `Group.create_array(shape=..., chunks=...,
dtype=..., overwrite=...)` in four places — a method that **does not exist
anywhere in the zarr 2.x line** (verified across every 2.x release up to
the latest, 2.18.7). `create_array` is zarr **v3**'s method; `pyproject.toml`
pins `zarr>=2.16,<3` specifically *because* v3 changed this interface
(`CLAUDE.md` §6.4) — meaning the version pin was right but the code calling
it was written against the wrong version's API. The very first real
extraction call (even the mock-model smoke config, run for the first time
in an environment that actually had the pinned zarr 2.x installed) crashed
immediately with `AttributeError` on `Group.__getattr__` falling through to
a `KeyError: 'create_array'`. This means the "smoke test green, 10
sections/18 findings" baseline recorded elsewhere in this file must have
been measured in an environment where zarr's version drifted above the pin
(3.x does have `create_array`), or the storage-writing code path was never
actually exercised end-to-end before — either way, this is the same "wired
but unverified actually means broken" pattern as §11.9–§11.12, just for
`tsfm_lens` instead of `build_pipeline`. **Fix:** renamed every
`create_array` call to `create_dataset` (zarr 2.x's actual `Group` method,
confirmed to accept the identical `shape=`/`chunks=`/`dtype=`/`data=`/
`overwrite=` kwargs by direct inspection, and that the returned `Array`
still supports `.oindex` orthogonal indexing, already relied on elsewhere
per §11.4). Re-ran the smoke config after the fix: clean end-to-end again,
same 10 sections / 18 findings as previously documented. **Lesson:** a
"green smoke test" claim is only as trustworthy as the environment it was
last actually run in — pin compliance should be spot-checked (e.g.
`pip show zarr`), not assumed, whenever a session inherits a "known
passing" claim from before.

### 11.16 The alignment-check tool's hardcoded impulse amplitude silently breaks on context-adaptive tokenizers
`extraction/alignment.py`'s `impulse_alignment_check` added a fixed
absolute perturbation (`+= 8.0`) to a unit-amplitude sine probe signal, at
one timestep per window, then checked which pooled window showed the
largest activation change. This works fine for TimesFM (patch-MLP
embedding, no cross-patch rescaling) but **badly miscalibrated Chronos's
result**: the first live `--check-alignment` against
`amazon/chronos-t5-small` measured a worrying **min diagonal-hit fraction of
0.06** (essentially the 1/16 chance floor) and a non-monotonic per-layer
pattern (0.50, 0.06, 0.88, 0.62, 0.44, 0.38) — exactly the "near-zero
everywhere... fix the adapter before trusting any cross-model number"
signal this file's own doctrine (§6.3) warns about. Per §2.4, measured
before concluding the adapter was broken: directly tokenizing the test's
input batch and diffing token IDs against the unperturbed baseline showed
**472 of 513 tokens changed, for every single impulse window position
tested** — not just tokens near the impulse. Chronos's `MeanScaleUniformBins`
tokenizer computes its quantization bin edges from the *whole sequence's*
statistics; an amplitude-8 spike (8x the base signal's own amplitude) is
large enough to shift those global bin edges, so nearly the entire token
sequence re-quantizes regardless of where the impulse actually sits. The
impulse test's diagonal-dominance signal was being drowned in this
global-rescaling noise — it was measuring tokenizer sensitivity to outlier
magnitude, not alignment-span correctness. Confirmed via an amplitude sweep
(8.0 → 2.0 → 0.5 → 0.15, as a fraction of the base signal's own unit
amplitude): Chronos's diagonal-hit fraction rose monotonically to a
**perfect 1.00 at every layer for amplitude ≤0.3**, while TimesFM stayed at
a perfect 1.00 across the *entire* sweep (no rescaling sensitivity to lose).
**Fix:** the impulse is now `0.25 × base.max()` — relative to the probe
signal's own amplitude, not a hardcoded absolute constant — with the
mechanism and measured numbers recorded in a code comment. Re-verified:
both TimesFM and Chronos-T5 now show a perfect 1.00 diagonal-hit fraction
at every captured layer, a far more trustworthy confirmation of the
alignment machinery than either the misleading pre-fix numbers or anything
tested before on live weights. **Lesson:** a fixed-magnitude test probe
tuned against one adapter can silently fail on a structurally different
one (here: continuous patch embedding vs. context-adaptive quantization) —
scale test perturbations relative to the signal, not to an absolute
constant, whenever the thing being probed can itself rescale based on
input statistics.

### 11.17 `write_text`/`read_text` without an explicit encoding is a platform trap
`report/report.py`'s `out.write_text(html)` (plus a handful of similar calls
in `utils.py`'s `save_json`/`load_json`, `config.py`'s
`load_config`/`dump_config`, and `data.py`'s jsonl loader) relied on
Python's platform-default text encoding rather than naming one. On Linux/WSL
that default is UTF-8, so this was invisible for as long as every "smoke
green" run happened to be exercised there. On native Windows the default is
`cp1252`, and the report template already contains non-cp1252 characters
(`▸`/`▾` in the `details.note` CSS, `Δ`/`ρ`/`×` etc. elsewhere) — so
`run_report` crashes with `UnicodeEncodeError` the moment it is actually run
on native Windows, which had apparently never happened before (`ROADMAP.md`
§4's 2026-08-04 Findings). Same "wired but unverified environment
assumption" pattern as §11.15/§11.16, just for text encoding instead of a
library API. **Fix:** every `write_text`/`read_text` call in `tsfm_lens` now
passes `encoding="utf-8"` explicitly. `json.dumps`/`yaml.safe_dump` both
default to ASCII-safe output (`ensure_ascii=True` / `allow_unicode=False`),
so those call sites were latent rather than actively broken, but were fixed
too rather than left on an implicit, platform-dependent default waiting to
break on the next library upgrade. **Lesson:** never trust `Path.write_text`/
`read_text`'s default encoding for any file that might contain non-ASCII —
name `encoding="utf-8"` explicitly, every time, regardless of what platform
today's session happens to be running on.

### 11.18 A relative-to-its-own-peak threshold lets weak signals fake "emergence" at the wrong layer
`analysis/layer_screen.py`'s Idea B (`factor_emergence_scores`) scores a
layer by how many ground-truth factors have their `emergence` (first layer
clearing a threshold) or `peak` (argmax) decodability there. `emergence`'s
threshold is defined *relative to each factor's own peak* (`threshold_frac *
col.max()`), which seemed reasonable — it lets a factor that's only ever
weakly decodable still contribute an emergence layer instead of being
swamped by strongly-decodable ones. In the first real bake-off run
(`ROADMAP.md` §6.1.1's Findings, 2026-08-05) this backfired: many of the
ground-truth factors in `benchmark_medium/public_dev` are weakly decodable
everywhere (their own peak R² is low), so a *relative* 50%-of-peak bar is
also low in absolute terms and gets cleared early by ordinary noise, not by
a real emergence event. On TimesFM, 8 of 20 kept factors nominally
"emerged" at layers 2–4 even though the *informative* peaks (the layers
with genuinely higher R²) clustered at layers 9–16 — the noisy early hits
diluted the count-based combined score before the real signal could
dominate it, and `factor_emergence` ended up scoring at or below the random
null on Chronos-T5-Small (recall@budget = 0.00). **Not fixed this
session** — the diagnosis (found by inspecting the `emergence`/`peak`
breakdown directly, not by guessing, per §2.4) is recorded as the concrete
next step: weight each factor's contribution by its own peak R² (or gate
`emergence` behind an absolute floor, not only a relative one) before
trusting this selector again. **Lesson:** a threshold defined relative to a
noisy signal's own peak doesn't distinguish "this factor has a real,
strong emergence event" from "this factor is uniformly weak and its peak is
barely above the noise floor" — an absolute floor (or a peak-magnitude
weight) is needed alongside any relative threshold, not instead of it.

**Fixed 2026-08-06 (code + synthetic-data unit test only — not yet
re-verified against the real bake-off).** `factor_emergence_scores` now
does both things the diagnosis above named: (1) each factor's contribution
to `count_score` is weighted by its own peak decodability (`col.max()`)
rather than counted as a flat `+1`, so a uniformly weak factor's spurious
early "emergence" moves the combined score proportionally less than a
strong factor's real late peak; (2) a new `min_peak_score` parameter
(default `0.15`, absolute, not peak-relative) gates the `emergence` event
specifically — a factor whose own peak never clears it is recorded in a new
`no_emergence_factors` list and contributes only a (small, weighted) `peak`
event, never an `emergence` one. A new planted-data test
(`tests/test_layer_screen.py::test_factor_emergence_weak_factors_no_longer_dilute_strong_late_peak`)
constructs one factor with a real late peak (layer 6) against four factors
that are uniformly noisy and never clear ~0.11 decodability, and asserts
the combined per-layer score's maximum lands on the strong factor's real
signal (layer 5 or 6, not the weak factors' spurious early layers 0–2) and
that every weak factor is excluded from `emergence` via
`no_emergence_factors`. Full `tsfm_lens` suite re-run clean after this fix:
66/66 passing. **What this does not close:** the fix has only been checked
against synthetic planted data, not re-run through the real §6.1.1-E
bake-off against live TimesFM/Chronos checkpoints — whether
`factor_emergence` now actually beats the
uniform-stride/random nulls on real activations (it scored at/below them
before this fix) is still open and is the natural next step before
promoting it past `work_bend` as a selector choice. See `ROADMAP.md`
§6.1.1's Findings for the same update with the exact numbers.

**Re-run against the real bake-off 2026-08-10 — the fix does not close this,
and actively regresses one of the two models.** Reran
`run_layer_screen_bakeoff.py` against the same live TimesFM 2.5 /
Chronos-T5-Small store this bug was originally diagnosed on (both seed 0
and the seed-1 replicate; old outputs backed up, not overwritten blind).
**Chronos-T5-Small: unaffected.** `factor_emergence` still selects
`[encoder.block.0, encoder.block.3]` and scores recall@budget=0.00,
`beats_random=False` in every one of the four runs (both seeds, pre- and
post-fix) — bit-for-bit the same failing selection as before the fix.
**TimesFM: got worse.** recall@budget dropped from 0.20 (pre-fix, one hit)
to 0.00 (post-fix, zero hits) in both seeds; `beats_random` was already
`False` and stays `False`, but the quantitative recall regressed. `work_bend`
and `coverage` were bit-identical pre- vs post-fix on both models/seeds,
confirming the fix stayed correctly scoped to `factor_emergence_scores`
with no side effects elsewhere — that part of the diagnosis holds.
**Root cause, isolated by holding one run's stored decodability matrix
fixed and rescoring it with the old vs. new formula (not just comparing
two separately-trained, individually-noisy runs):** on real data, `
no_emergence_factors` came back **empty** in every run — the new absolute
`min_peak_score=0.15` floor never actually excluded anything here, so that
half of the fix is inert on this corpus. The other half — weighting each
factor's contribution by its own peak decodability — backfires, because the
real factors with the *highest* peak R² on this corpus/model pair
(`has_intermittency`, `archetype_intermittent_bursts`,
`archetype_trend_dominant`, peak weight ≈0.80–0.82) are near-input
statistics whose decodability rises fast and plateaus at an early layer —
so weighting up exactly these factors pulls the combined emergence score
*toward* layer 0, which is the single worst layer by the SAE-mass gold
ranking (TimesFM block 0 mass 34.8 vs. the real peak block 4's 109.9).
This is the mirror image of the synthetic test's planted scenario (weak,
noisy factors causing a spurious *early* signal that dilutes one strong
factor's real *late* signal) — on this real corpus it is the *strong*
factors that carry the early-layer bias, so peak-weighting amplifies
rather than corrects it. The unit test this fix added still passes and is
still a real, correctly-diagnosed fix for the specific failure mode it
targets — it just isn't the failure mode that dominates on real
checkpoints, which is exactly why §2.4's discipline requires re-testing on
real data before trusting a synthetic-only fix, not after. **Bottom line,
unchanged in substance but now on firmer ground:** `factor_emergence`
remains not usable and is not the production default (`work_bend` is, and
is untouched by any of this) — but the reason has moved from "a diagnosed,
fixable weighting bug" to "a structural mismatch between what this
selector's design assumes causes early-layer bias and what actually causes
it on real activations." A future fix would need to explicitly discount
very-early emergence layers (a depth floor, not just a magnitude floor),
not reweight by peak magnitude alone — not attempted this session. Full
`tsfm_lens` suite green at 203/203 after the rerun (no production code
changed, verification-only). See `ROADMAP.md` §6.1.1's Findings and §13
for the full numbers table.

### 11.19 Summing raw per-source MSE lets the larger-scale source dominate a joint crosscoder
`sae/crosscoder.py`'s `CrosscoderSAE` trains one dictionary jointly across
`n_sources` inputs by summing each source's MSE into one loss. The first
version computed that MSE on each source's *raw* activation scale. A
synthetic test with one source at 20x the other's scale reproduced exactly
the failure `ROADMAP.md` §13 named as the risk for two independently
initialized model checkpoints (which have no reason to share an activation
scale): the smaller-scale source's reconstruction fidelity collapsed to
**−9.7** (worse than predicting the mean) while the larger-scale source
sat at 0.95 — the optimizer simply spent its capacity on whichever source
contributed more raw squared error, with no signal that the smaller
source's *relative* fit was terrible. Measured before assuming this would
be fine (§2.4) precisely because the module docstring already flagged the
risk in prose; testing it turned "should be fine" into a confirmed bug.
**Fix:** `CrosscoderSAE` now takes a per-source `source_scale` (each
source's own global std, computed once by `train_crosscoder`) and
divides/multiplies by it inside `encode`/`decode`, so the dictionary is
fit in a scale-normalized space internally while the public
`encode`/`decode`/`forward` contract still takes and returns activations
at each source's original scale — training-loss MSE and dead-neuron
resampling's "which source needs this atom most" routing were both moved
onto the same normalized residuals, not just the encoder's TopK
competition. Re-tested after the fix: fidelity for both sources stayed in
a normal, comparable range with no collapse, repeatably. A second,
unrelated bug only the real-GPU run surfaced: `torch.multinomial`'s output
stays on whatever device its input `probs` tensor was on (here, CPU, since
probabilities were moved there for the sampling call) even when every
other tensor in the function is on CUDA — indexing a CUDA tensor with that
leftover CPU index two lines later raised a device-mismatch
`RuntimeError` that no CPU-only synthetic test could have caught. Fixed by
moving the sampled index tensor to the target device immediately after
`torch.multinomial`. **Lesson:** for any newly-written multi-source loss,
test an artificially extreme scale mismatch *before* trusting a real
multi-checkpoint run — two real model checkpoints are exactly this
scenario by default, not an edge case; and a bug in device-transfer
plumbing can hide indefinitely behind CPU-only tests no matter how
thorough the synthetic coverage is, so a real-GPU run before trusting a
new training loop is not optional.

### 11.20 Open (not yet fixed) audit items — read before trusting a number 🔴
Everything above in §11 is a trap that was **hit and fixed**. As of
2026-08-06 there is also a list of traps that are **live and unfixed**:
`ROADMAP.md` §15, produced by a planning-only pass that read this file and
`ROADMAP.md` against the pipeline rather than trusting either. Nineteen
items, each with `file:line` evidence, a stated blast radius, and a detailed
fix plan; nine are marked P1 ("can silently produce a wrong number that
reaches a report or a recorded finding").

The short version, because a fresh session will otherwise re-derive it:
**twelve of the nineteen are the same shape** — a mechanism that degrades
quietly where §2.5 says it must degrade loudly, because "loudly" was
implemented as a log line. Logs are not loud inside a thirty-minute run,
and they are not loud at all once the deliverable is an HTML file someone
opens next month. The load-bearing five: the in-pipeline alignment check
discards its own result (A2); corpora are written grouped by task while four
call sites subsample with a head slice, so several already-published numbers
sit on family-skewed prefixes (A4); stage skipping has no config fingerprint
(A3); no ΔMASE anywhere has been compared against a measured repeat-run
noise floor (A13); and there is still no CI, which is why §11.8/§11.9/§11.10/
§11.15/§11.17 each cost a human a session instead of costing a build ten
minutes (A14).

Do not read this as "the results are wrong." Read it as: **the error bars on
a specific, named set of recorded numbers are currently unknown**, and
`ROADMAP.md` §15's markers say which. Two of them gate a real decision
(whether Phase 3's feature-level ablation can build on the TimesFM SAE
target). When an item is fixed, move its lesson up into §11 proper and mark
the §15 item `[x]` — don't delete it (`ROADMAP.md` §0.2).

### 11.21 A capture hook that only recognized `tuple` outputs missed HF's `ModelOutput` dataclasses
`extraction/hooks.py`'s `_primary()` (the helper every capture/patching/
ablation hook uses to pull the hidden-state tensor out of whatever a
module's `forward()` returned) checked `isinstance(output, tuple)` and
fell through to treating anything else as already-a-tensor. This was
invisible across three real adapters (Chronos-T5, Chronos-Bolt, TimesFM)
because none of their captured blocks happen to return anything but a
plain tensor or a literal tuple. Adding a fourth real adapter
(`chronos2_adapter.py`, ROADMAP.md §9, Phase 4) broke this immediately:
`Chronos2EncoderBlock.forward` returns a `Chronos2EncoderBlockOutput`, an
HF `transformers.utils.ModelOutput` dataclass — dict-like (it subclasses
`OrderedDict`), integer-indexable (`output[0]` works), but **not** an
instance of `tuple`. `_primary` returned the whole dataclass unchanged, and
the very next line's `.detach()` call crashed with `AttributeError` on the
first real `--check-alignment` run. **Fix:** check for `torch.Tensor`
first instead of `tuple` (`output if isinstance(output, torch.Tensor) else
output[0]`) — correct uniformly for a tensor, a tuple, or any
integer-indexable dataclass. A second, related gap in the same module:
`token_patch`/`output_mean_ablate`'s hooks reconstructed a patched output
via `(patched,) + tuple(output[1:])`, which would have silently degraded
any dataclass output to a plain tuple — losing attribute access to any
other field (e.g. a block's own attention weights) for whatever consumed
that forward pass next. Fixed with a new shared `_rebuild(output,
replacement)` that uses `dataclasses.replace(output, **{first_field:
replacement})` to swap only the primary field when the output is a
dataclass, preserving its exact original type. New
`tests/test_hooks_modeloutput.py` covers this directly against a synthetic
`ModelOutput`-style module, independent of any real checkpoint. **Lesson:**
"nothing downstream changes when adding a model" (`CLAUDE.md` §6.2,
ROADMAP.md §9's own success bar) is a claim about the *abstraction*, not a
guarantee that three prior real adapters have already exercised every
shape a fourth might introduce — a plain `tuple` check is exactly the kind
of assumption that looks complete until an architecture with a genuinely
different forward-output convention (HF's own `ModelOutput`, ubiquitous
across `transformers`-based models) shows up.

### 11.22 A checkpoint's own remote code can be broken against the transformers version everything else already depends on
Adding Sundial (`thuml/sundial-base-128m`, ROADMAP.md §9, Phase 4's second
model addition) surfaced two bugs — but this time in the **checkpoint's own
`trust_remote_code=True` modeling file**, not in this repo's shared
infrastructure (contrast with §11.21's Chronos-2 finding, which *was* a
shared-code bug). Sundial's HF model card's own documented usage —
`model.generate(seqs, max_new_tokens=..., num_samples=...)` — crashes with
`AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`
inside Sundial's own `prepare_inputs_for_generation`. A second, independent
break hits even a single non-generation forward call unless `use_cache=
False` is passed explicitly: `'DynamicCache' object has no attribute
'get_usable_length'` inside `SundialModel.forward`'s cache path. Both
attributes were removed from `transformers.DynamicCache` in a release
newer than the 4.40.1 the model card recommends but older than this repo's
installed 4.57.6 — the checkpoint's remote code was written against an
API surface that no longer exists in any transformers version this repo
could plausibly install today. **Not fixed by downgrading transformers**
(that would risk the exact §11.8-class regression of breaking the other
three already-working adapters over one new one) — fixed by never calling
`.generate()` at all: `SundialAdapter.predict()` calls
`SundialForPrediction.forward(..., use_cache=False)` directly, which is
sufficient because Sundial's flow-matching head samples the entire forecast
horizon in one shot from the last patch's hidden state rather than
autoregressively, and this repo's horizon (64) is far under Sundial's
720-token one-shot limit. `forward()` (the capture path) goes one level
lower still, calling `model.model(...)` to skip the flow-matching head
entirely, mirroring `ChronosAdapter`'s existing "encoder only" capture
pattern. **Lesson, distinct from §11.8-11.10's "a *pinned* library's own
release broke its API":** here the break is in a *third party's* remote
code shipped alongside the checkpoint weights, version-pinned to a
transformers release this repo doesn't (and, given the other three
adapters, can't easily) install — the fix path is "route around the
checkpoint's broken convenience method using the lower-level call it
wraps," not "pin a version," when the alternative would destabilize
everything else already depending on the installed version.

**A second, real-but-benign finding from the same session, worth recording
so it isn't mistaken for a bug later:** Sundial's `--check-alignment`
diagonal-hit fraction is **amplitude-dependent** — a perfect 1.00 at every
layer at small impulse amplitudes (0.02×–0.05× the probe signal's own
amplitude), decaying by mid-depth at this repo's default 0.25× amplitude
(§11.16's own fix value, chosen for Chronos's tokenizer). Diagnosed rather
than assumed broken (`CLAUDE.md` §2.4): zero backward/future leakage at
every window/layer/amplitude tested (causality is intact), and the decay
tracks amplitude cleanly, consistent with ordinary forward causal signal
accumulation through Sundial's plain-residual, no-QK-norm decoder blocks
— a real architectural property, not a broken `token_time_spans`.
**Directly relevant to §6.3's own audit note (A2):** the in-pipeline
automatic alignment check only probes a stride-4 subset of layers and
discards its result, and — worse for a case like this — even a *manual*
`--check-alignment` run that only glances at the shallowest layer (always
a perfect 1.00 here) would miss this pattern entirely. Reinforces, with a
concrete example, why invariant 7 means reading the check's *full
per-layer table*, not just confirming it runs without error.

### 11.23 A stability replicate that only reseeds SAE training doesn't cover extraction-to-extraction variance
`layer_screen_bakeoff.py`'s existing robustness check (ROADMAP.md
§6.1.1's first Findings block) reruns the bake-off with a different SAE
training seed against **the same, already-extracted activation store** and
checks gold-ranking Spearman stability (ρ=1.0/0.926 across the two models,
originally read as a reassuring replication). Extending the bake-off to a
third architecture (Sundial, ROADMAP.md §6.1.1's newest Findings, same
session as §11.22) required building a *fresh* extraction of the same
config values rather than reusing the frozen store, and that fresh
extraction changed Chronos-T5-Small's qualitative bake-off verdict
outright — `work_bend`'s `beats_random` flipped `True`→`False`
(recall@budget 1.0→0.0) between two nominally identical-config runs,
independently confirmed against both runs' actual JSON artifacts, not
taken on a report's word. TimesFM's own qualitative verdict reproduced
unchanged across the same two runs, so this is a real instability for at
least one model, not a uniformly broken check. **The gap:** the existing
seed-reseed replicate answers "is the bake-off's *scoring*, given fixed
activations, stable?" — yes. It has never answered "is the *activation
extraction itself* (a fresh forward pass over the same corpus/config)
stable enough that the gold ranking, and therefore the selector verdict
built on it, doesn't depend on which extraction happened to run?" — this
session's accidental natural experiment (building a third-model
extraction forced a fresh one) shows the answer, for at least
Chronos-T5-Small, is no. **Not yet root-caused** (candidates: corpus row
sampling, model-loading/dtype nondeterminism, or SAE-training stochasticity
compounding with a genuinely different activation draw) **and not yet
fixed** — flagged in `ROADMAP.md` §13 as a new, prominent open item rather
than silently absorbed into the existing seed-replicate claim. **Lesson:**
when a pipeline stage's own "robustness check" holds one expensive
upstream artifact fixed and varies only a downstream seed, that check
provides evidence about the downstream stage's stability *conditional on*
that artifact — it says nothing about whether the artifact itself would
look the same on a second, equally valid run. Don't let a partial
robustness check read as a full one.

**Root-caused 2026-08-10, same day, follow-up session — see §11.24.** This
was not GPU nondeterminism, model-loading nondeterminism, SAE-seed
sensitivity, or Sundial's presence (all four checked directly and ruled
out). It was `ROADMAP.md` §15 A4's already-landed corpus-sampling fix: the
08-05 extraction predates it, the 08-10 rerun postdates it, and the two
"identical configs" therefore sampled genuinely different, non-overlapping
row sets from the same corpus. Fully explained, not merely narrowed — see
§11.24 for the mechanism and `ROADMAP.md` §13/§6.1.1 for the corrected
Findings.

### 11.24 A background experiment's own config can silently change meaning across a shared-infrastructure fix, with no diff to catch it
§11.23 documented a real, alarming-looking symptom: two runs of
`run_layer_screen_bakeoff.py` against byte-identical YAML
(`configs/layer_screen_experiment.yaml` vs. `_v2.yaml`, diffed directly —
the only difference is a third model block for Sundial) produced
completely different Chronos-T5-Small verdicts. A dedicated background
investigation (per §2.8) ranked four plausible mechanisms — GPU/SDPA kernel
nondeterminism, SAE dictionary-init/seed sensitivity, global-RNG-stream
reordering from adding a third model to the config, and corpus-sampling
nondeterminism — and live-tested the two most GPU-dependent ones directly
(a same-store SAE-reseed test, and a from-scratch re-extraction compared
via `np.array_equal` on the raw stored activations). Both were reproducible
and deterministic (ruling out GPU/process nondeterminism outright: two
independent extractions of the same config were bit-for-bit identical), but
neither explained *why* the 08-05 and 08-10 numbers disagreed in the first
place. The actual mechanism was found by a follow-up `git diff` against the
extraction path's git history rather than by testing another hypothesis in
isolation: `data.py`'s `_assemble` used to cap `data.max_series` with a bare
`kept[:max_series]` head slice; `ROADMAP.md` §15 A4 (fixed 2026-08-06 — one
day after this experiment's original 08-05 run, four days before its
accidental 08-10 rerun) replaced that with a family-stratified
`sample_rows(len(kept), max_series, seed, strata=families_all)` call. The
experiment's own `data.max_series: 220` against the 288-row
`benchmark_medium/public_dev` corpus means the pre-fix run silently sampled
`{random_parametric: 188, parametric: 32}` — **the entire 60-row `mixture`
family, and 8 of the 40 `parametric` rows, were never seen at all** —
because `tsfm_benchmark` writes corpora grouped by generator and `mixture`
happens to sort last in this corpus. The post-fix runs correctly draw a
family-proportional 220-row sample instead. Same YAML, same seed, same
`max_series` integer, completely different actual training/scoring
population — because the code that interprets `max_series` changed
underneath it, and nothing about that shows up in a config diff. **This is
not a version-pin trap like §11.8-§11.10, and not a checkpoint's-own-code
trap like §11.22 — it's a new category**: a standalone experiment script or
config that reuses shared pipeline internals (`data.py`, `extract.py`) can
have its own meaning silently rewritten by a fix to those internals,
entirely independently of whether the experiment's own file changes at all.
`CLAUDE.md` §2.8's background-delegation workflow makes this more likely to
recur, not less — a background agent asked to "rerun the same experiment"
has no way to know a shared-infrastructure fix landed in between unless
explicitly told to check. **Lesson:** before treating two runs of a
standalone experiment config as comparable, check whether any shared
pipeline file the experiment depends on (not just the experiment's own
config/script) changed between the two runs' dates — `git log --since=<run
1 date> --until=<run 2 date> -- <shared path>` is cheap and would have
caught this immediately, and is now worth doing by default whenever a
background rerun of "the same" experiment produces a surprising result,
before spending investigation budget on nondeterminism hypotheses first.

### 11.25 A zarr store written under a different major version doesn't error at read time — it silently reads back empty
Revisiting `ROADMAP.md` §5.3's lens re-measurement (2026-08-10) required
loading `runs/medium_run`'s already-extracted `activations.zarr`, and
`ActivationStore(..., mode="r")` opened without error but returned zero
layers/rows for every model — not an `AttributeError` like §11.15's
`create_array`-vs-`create_dataset` bug, a **quiet empty store**. Inspecting
the store's own metadata directly (`runs/medium_run/activations.zarr/
zarr.json`) showed `"zarr_format": 3` — this run directory was written
entirely in zarr **v3**'s on-disk layout (a single `zarr.json` manifest per
group/array) at some point before this repo's zarr pin was enforced or
before §11.15's fix session, evidenced by file mtimes roughly two weeks
older than that session. zarr **2.18.7** (this repo's pinned version, `zarr
<3`) does not raise on `open_group` against a v3-only directory — it has no
v2 metadata (`.zattrs`/`.zgroup`) to find, so it silently treats the
directory as an empty v2 group and lets every downstream read proceed
against that emptiness rather than erroring. Confirmed this is not
universal corruption: of every run directory under `runs/`, only
`medium_run` and `real_run` lacked a valid v2 `.zattrs` — every other run
(including `medium_run_chronos_base`, used successfully in the same
session) has one.

**Fixed 2026-08-10 (`medium_run` only; `real_run` untouched — see below).**
A first attempt to delete the stale store and rerun `medium_run`'s
`extract` stage via a background agent (per §2.8) failed only because that
agent's own session hit an infrastructure-level usage limit mid-run, not
because the fix itself failed. A second, freshly-launched background agent
completed the same fix cleanly (~11 min wall-clock): the stale directory
(the original v3 `zarr.json` files plus the stray empty v2 `.zgroup` the
failed first attempt had left alongside them) was removed outright and
`extract` rerun from scratch, producing a real v2 store (`.zattrs` present,
no `zarr.json`). Verified independently on disk, not just from the agent's
own report: the full pipeline completed end to end (`report.html`, 11
sections / 27 findings at the time, `confirm` 1/1 dev hypotheses passing on
private data), and the now-trustworthy TimesFM lens number
(`final_mase: 1.7899408340454102`) reproduces bit-for-bit against
`medium_run_chronos_base.yaml`'s already-recorded value — the exact
cross-check `ROADMAP.md`'s A4 fix (family-stratified sampling) needed a
second, independent store to confirm. Full `tsfm_lens` suite green
afterward (208 passed, 2 warnings at the time). See `ROADMAP.md`'s session
log (2026-08-10, "same-day follow-up" entry after the eighth cron-loop
firing) and §5's A4 Findings block for the full numbers — not repeated
here. **`real_run` was not part of this fix and its own store's format has
not been re-checked** — the fix above only re-extracted `medium_run`
because that was the store an active re-measurement needed; treat
`real_run`'s `activations.zarr` as suspect until spot-checked the same way
(`zarr.json` present / `.zattrs` absent → stale) before trusting anything
read from it. **Lesson, distinct from §11.15's
"wrong method name for the pinned major version" bug:** a pinned major
version being *installed* correctly does not guarantee every *on-disk
store* was written under that same pin — a store's format can drift
independently of the current environment's package versions (e.g. from an
earlier environment, a different machine, or a brief unpinned install),
and the read path's degrade-to-empty behavior means this shows up as
"nothing extracted yet" rather than a version-mismatch error. Spot-check a
suspicious-looking empty store's own `zarr.json`/`.zattrs` presence
directly before assuming `extract` simply never ran.

---

## 12. Known limitations (stated, not hidden)

> **Which of these are actually fixable (added 2026-08-06).** A limitation
> being stated honestly is not the same as it being unavoidable, and the
> audit pass (`ROADMAP.md` §15/§16) found several here that have a concrete
> fix nobody had scheduled: the forecast-stochasticity asymmetry (item 3
> below) is measurable rather than merely caveat-able — nothing currently
> measures either model's repeat-run noise floor, so every ΔMASE in the repo
> is compared against zero instead of against it (§15 A13); the Chronos
> decoder gap (items 1-2) has a specified plan as §16 E21; the O(n²) matcher
> ceiling is §15 A17; and the catch22/ground-truth expressiveness caveats are
> exactly what §16 E9's untrained-weights null and E16's third matching
> signal are for. Treat the list below as accurate about *today* and
> `ROADMAP.md` §15/§16 as the record of which entries are permanent envelope
> edges versus unscheduled work.

**Architectural / conceptual**
1. **The Chronos decoder is largely invisible.** Capture is encoder-only. Any
   claim like "Chronos represents X weakly at depth d" is really a claim about
   its **encoder**. Cross-attention opens a window at **decode step one only**;
   decoder self-attention, later steps, and decoder head ablation are unobserved.
2. **Coverage asymmetry.** For TimesFM we see essentially the whole computation;
   for Chronos maybe half.
3. **Forecast stochasticity asymmetry.** Chronos-T5 samples; TimesFM and
   Chronos-Bolt are deterministic. Ablation/patching deltas sit on different
   noise floors. Seeds are pinned and `num_samples` reduced during patching;
   **Chronos-Bolt is a useful deterministic control.**
4. **Relative-depth interpolation is a convention, not a fact** — it assumes
   depth fraction is the right axis to compare a 50-layer decoder against a
   12-block encoder.
5. **TimesFM's finest resolvable lag is one patch-width (~32 steps).** Not a
   plotting artifact — a real ceiling on what that model's attention can express.
6. **Nothing is feature- or component-level below the head/MLP granularity.** You
   learn that family information *is* linearly present at layer X, not *what* is
   encoded.
7. **The L0↔internals link is correlational.** No bridging intervention
   establishes that TimesFM's early decodability *causes* its advantage. (The
   Lens stage is the closest thing and was added for exactly this reason.)

**Envelope hard edges for new architectures**
- Models whose tokens **aren't time-localized** (spectral tokenizers,
  Perceiver-style latent queries) break the pooling premise → **L0 only**.
- The skip lens assumes **uniform hidden size** across captured blocks.
- Patching assumes the context is processed in **one forward pass**.
- No `attention_patterns` → those analyses skip. No `o_proj`-style Linear → no
  head ablation. Cross-attention only fires for encoder-decoder models.

**Benchmark / validation**
- xcorr and DTW measure **shape** similarity after z-normalization — two series
  with the same shape and different noise are *intentionally* redundant. Read
  redundancy alongside feature metrics.
- catch22 is general-purpose; if a task hinges on a property catch22 doesn't
  encode, **add targeted features** rather than trusting the space.
- Matcher is **O(n²)** — past a few thousand sequences you need blocking or an
  approximate prefilter.
- `tsaug` and `timesynth` are largely unmaintained and break on recent numpy —
  the numpy parametric core exists so the pipeline doesn't depend on them.
  `tsbootstrap` is younger and had a full public-API rewrite since this was
  first written (§11.10): **pin it** (`tsbootstrap>=0.7`, current API target).
- `block_bootstrap` (tsbootstrap), `sequential_par` (SDV), and the Monash
  adapter were **wired but unverified** as of this file's earlier writing.
  **Correction (2026-08-03): now verified live end-to-end** — see §4.3,
  §9, §11.9–§11.12, and `ROADMAP.md` §5's Findings block for the full story
  (all three turned out to be actually broken against current library
  versions for different reasons, now fixed, not just "unverified").

---

## 13. Future work — in rough priority order

> **Read `ROADMAP.md` §15 and §16 alongside this list (added 2026-08-06).**
> This section is the original, pre-roadmap future-work list and is kept for
> continuity; `ROADMAP.md` is where forward work is actually tracked. §16 in
> particular covers the ground this section doesn't: what the repo needs to
> become usable by someone who didn't build it (zero-config entry point,
> preflight doctor, empirically-discovered token spans instead of declared
> ones, a bundled reference corpus, CI), plus the analysis axes a TSFM
> interpretability user asks for that nothing here mentions — horizon-resolved
> metrics, a spectral lens, forecast calibration diagnostics, steering, and
> input-front-end diagnostics. Items 2, 3, 4, 6, 7, and 8 below all have a
> §16 counterpart with a sharper scope; prefer the §16 wording where they
> disagree.

1. **Run the pipeline on real checkpoints at scale.** The highest-value next
   action. Verify the TimesFM flat-patching curve resolves under per-window
   patching, and validate `output_attentions` / the TimesFM module scan against
   live weights.
2. **Close the Chronos decoder gap** (if worth the cost). The clean extension is
   a **second capture surface over decoder layers under teacher forcing** — the
   hook machinery already supports this mechanically. The hard part: pooled
   decoder states live on **forecast time**, so they need their own alignment
   windows and **cannot be naively CKA'd** against context-window states. Treat
   as a **separate measurement**, not an attempt to force symmetry the
   architectures don't have.
3. **SAE phase — baseline landed 2026-08-05, flagship crosscoder still to
   do.** `sae.enabled: true` now runs a real baseline `TopKSAE`
   (`sae/models.py`/`train.py`/`eval.py`/`ground_truth.py`) trained straight
   from the store's window-level activations, optionally augmented with
   real (non-benchmark) activations pulled from an HF dataset
   (`sae/real_data.py`, reusing `tsfm_benchmark`'s generic loader), with a
   real evaluation harness (reconstruction fidelity, dead-feature rate,
   dead-neuron resampling, forecast-preservation via `token_patch`) and a
   real ground-truth feature-alignment score. First run found real
   seasonality-linked features but a real forecast-preservation failure
   (dead-feature collapse); fixed (two real bugs in the resampling fix
   itself along the way — see `ROADMAP.md` §6.2's Findings for the full
   story) to reconstruction fidelity 0.86/0.84 and a forecast-preservation
   check that now **passes for TimesFM** (ΔMASE +0.05) but **still fails
   for Chronos-T5-Base** (ΔMASE +2.4) — traced to an architecture-specific
   confound in the check's window-broadcast approximation (exact for
   TimesFM's patch tokenization, lossy for Chronos's per-timestep
   tokenization), not necessarily the SAE itself; documented in
   `eval.py`'s docstring. **Not yet done:** the flagship cross-model
   crosscoder (§6.2 item 1, still needs its own design work — joint-
   training stability across two architectures is an open question, §13
   below), and the encode-store pass into `sae/{model}/{layer}` this
   section used to describe as the whole seam — the baseline trains and
   evaluates but doesn't yet persist encoded features back into the store,
   so L1/clustering still can't read a `level="sae"`. Note SAE features
   **still need cross-model matching**
   (max-activating examples or input-space decoder correlations) once a
   crosscoder or per-model dictionaries on both sides exist — SAELens is
   the precedent for that being a distinct phase.

   > **Update (2026-08-11, ROADMAP.md §16 E16).** Implemented as
   > `sae/matching.py`, against independently-trained per-model dictionaries
   > (no crosscoder needed) — reframed from "input-space decoder
   > correlation" to **activation-profile correlation** (Pearson over a
   > shared series sample) since decoder vectors live in each model's own,
   > differently-sized hidden space and can't be compared directly; the
   > shared sample comes free from `ground_truth_alignment`'s existing
   > run-level row sampling, confirmed identical across targets on a real
   > run (`gt_a["rows"] == gt_b["rows"]`). Live-verified against real
   > TimesFM/Chronos-T5-Base SAE checkpoints: 40 of 50 ground-truth-matched
   > TimesFM features found a Chronos-side partner. Full numbers in
   > `ROADMAP.md` §16 E16's Findings, not repeated here.

   > **Correction (2026-08-11, ROADMAP.md §16 E15, sixteenth cron-loop
   > firing).** The "still fails for Chronos-T5-Base" claim above is now
   > superseded, not just qualified. `sae/eval.py::forecast_preservation`
   > gained a `granularity: "token"` mode (encode/decode every raw token
   > independently, no window-pooled broadcast) alongside the original,
   > now-called `"window"` mode; both are computed and recorded per target
   > (`"forecast_preservation"` / `"forecast_preservation_token"`), so no
   > prior number was overwritten. Verified live against
   > `runs/medium_run_chronos_base`'s real TimesFM-2.5-200M / Chronos-T5-
   > Base checkpoints (a freshly retrained SAE, so absolute values differ
   > from the ΔMASE +0.05/+2.4 pair quoted above — that's normal SAE-
   > training variance, not a regression): TimesFM's window and token
   > numbers are **bit-for-bit identical** (ΔMASE +0.110 both ways, exactly
   > as expected since its token width equals the alignment window), while
   > Chronos-T5-Base's ΔMASE drops from **+3.869 under "window"** (fails
   > badly) to **−0.346 under "token"** (reconstruction is net *better*
   > than the clean forecast) — decisive evidence the window-broadcast
   > confound, not SAE reconstruction quality, was the dominant driver of
   > every previously-recorded Chronos forecast-preservation failure.
   > Chronos's token-granularity delta being *negative* is itself a little
   > surprising (n=24 series noise, or TopK sparsity acting as a mild
   > denoiser, are the leading guesses) and worth a skeptical look before
   > reading it as "the SAE is simply excellent" — but directionally it
   > unambiguously confirms the confound diagnosis over the alternative
   > "genuine reconstruction quality" explanation. Feature-level ablation
   > (§7 bullet 3, `ROADMAP.md` §16 E15's second half) is now unblocked.
4. **Validation as CI gates.** Turn the diversity metrics into explicit pass/fail
   gates (redundancy fraction < X, effective dimensionality > Y, no
   near-collision cluster larger than Z) so each benchmark epoch is checked
   automatically rather than inspected manually.

   > **Update (2026-08-11, ROADMAP.md §16 E23, same firing).** Scaffolded:
   > `benchmark_validation/gates.py::check_diversity_gates` implements
   > exactly this (redundancy fraction, effective dimensionality, near-
   > collision fraction, corpus-wide and per-group), wired into
   > `run_validation.py` (always prints; `--enforce-gates` to fail the
   > exit code). Thresholds are still calibrated against only the one
   > demo-mode reference point (§5's 4.8%/4.6-of-24), so treat a fail as
   > "worth a human look" rather than a validated gate until a real corpus
   > build recalibrates them — the item stays open in `ROADMAP.md` §16 E23
   > for that reason, not because the mechanism is unbuilt.
5. **Land the QuantileTransformer / IQR fix** in validation if not already
   present (§5).
6. **More source adapters** — `TIME`, `BOOM`, `ARFBench` slot into the existing
   `SOURCES` contract once access and licenses are confirmed; `monash` is the
   template.
7. **Real-corpus activation bucketing.** Embed a large real corpus (Monash,
   GIFT-Eval pretraining corpus, LOTSA) in each model, cluster, and compare
   **partitions** across models via AMI / cluster matching. Sequenced *after*
   the synthetic work, because clusters on wild data are uninterpretable without
   controlled reference points.
8. **Deeper component resolution** — per-path causal tracing, moving toward
   TransformerLens-level component attribution.

---

## 14. Framing that has proven useful

- **tsfm-lens vs TransformerLens differ in kind, not degree.** TL is a
  single-model mechanistic *microscope* (hook every component, decompose the
  residual stream, attribute the output). This is a two-model comparative
  *telescope* (aligned representations, statistics, held-out confirmation). TL
  has nothing like the confirm stage, alignment machinery, or CI discipline;
  this has nothing like TL's component resolution. The four TL ideas worth
  borrowing were prioritized as: forecast lens → finer causal resolution →
  attention patterns → exemplars. **All four have now landed.**
- **Fairness rests on where claims are grounded**, not on architecture-blindness.
  Cross-model *conclusions* come from behavior (paired per-series MASE, identical
  statistics), from **gains over an input-feature baseline** (L2), or from
  comparing each model's **within-model** causal fingerprint (L3). Causality is
  never measured "across" architectures directly. **Call it fair with stated
  caveats — not architecture-blind.**
