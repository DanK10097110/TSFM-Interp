# Functionality Summary — what this repo does, and how usable it is

> **Scope.** A short orientation for someone deciding whether to use this repo, and
> for whom. `CLAUDE.md` is the architecture-and-doctrine reference (why things are
> built the way they are, and which invariants are load-bearing); `ROADMAP.md` is the
> forward plan and the research record. This file is neither — it is the one-page
> answer to *"what can I actually do with this today?"*
>
> Written 2026-08-28, updated 2026-08-30 after the first three-model full-feature run
> on real checkpoints. Where a number here could go stale (stage counts, test counts),
> the authoritative source is named beside it; every count below was re-measured on
> the date above rather than carried forward.

---

## 1. In one paragraph

Two independently installable halves. **`tsfm_benchmark`** generates synthetic
time-series corpora with exact ground truth, audits them for leakage against real
reference data, seals them with hash manifests, and separately validates that a
generated corpus is actually diverse rather than 5,000 copies of the same shape.
**`tsfm_lens`** takes one or more time-series foundation model checkpoints, runs
them through a 17-stage layered analysis — behavioral, cost,
representational, causal, attention, sparse-autoencoder — and renders one
self-contained interactive HTML report in which every cross-model claim is either on
an axis both models genuinely share or is printed next to the measured size of the
asymmetry. The pipeline's distinguishing property is not breadth; it is that the
report's conclusions are *derived from artifacts by printed rules* rather than
written as sentences, and that a claim which cannot be supported renders as a stated
absence rather than being quietly dropped.

## 2. What you can do with it today

| I want to… | Command | Needs |
|---|---|---|
| Kick the tyres end to end | `python run.py --config configs/smoke.yaml` | nothing — no GPU, no downloads. **29 s** cold, to a 14-section / 54-finding report |
| Analyze **one** model on its own | `--config configs/smoke_solo.yaml` | comparison stages self-drop with a stated reason |
| Compare **two** real checkpoints | `--config configs/medium_run_chronos_base.yaml` | GPU + HF checkpoints |
| Compare **three or more** (mocks) | `--config configs/smoke_panel.yaml` | all C(n,2) pairs, one designated reference |
| Compare **three real checkpoints**, every feature on | `--config configs/full_report_run_3model.yaml` | GPU; raise `n_boot` with model count (§6.6's p-floor) |
| Analyze a checkpoint **with no adapter written** | `adapter: generic_hf` + `--probe-adapter <name>` | probes 4 seams, refuses rather than guesses |
| Check an environment before a long run | `python run.py --config <cfg> --doctor` | 10 checks: version pins, VRAM, disk, store format, corpus seal, multiplicity budget, adapter conformance, alignment |
| Find the right config | `python run.py --list-configs` | prints every `configs/*.yaml` by run shape, with purpose |
| Verify a finished run is still reproducible | `python run.py --verify-provenance <run_dir>` | no model load |
| Generate a benchmark corpus | `example_runs/run_full.py --config configs/example.yaml --references monash` | CPU only |
| Prove that corpus is diverse | `example_runs/run_validation.py --corpus … [--enforce-gates]` | CPU only |
| Compare N finished runs | `python run_meta_report.py --runs a,b,c` | reads artifacts, re-runs nothing |

Plus **28** standalone `run_*.py` studies (scaling ladder, layer-screen bake-off,
crosscoder ladder, seasonality circuit, spectral lens, error fingerprinting,
agreement, parameter/horizon/context sweeps) that operate on **already-extracted
runs at zero forward passes** — that "reads existing artifacts, loads no checkpoint"
contract is the repo's most reusable design idea.

## 3. What it costs to run

Measured on the three-model full-feature run (TimesFM-2.5-200M · Chronos-T5-Base ·
Chronos-2, 288 series of 576 points, one RTX A5000). Useful because the shape is
lopsided and not where you would guess:

| stage | wall clock | note |
|---|---|---|
| `l3` | **27 min** | the corruption battery × per-window patching × 3 models — over a third of the run |
| `sae` | ~14 min | 11 dictionaries; skipped by default (`sae.enabled: false`) |
| `attention` | 7.5 min | pattern capture is the memory-critical stage, not the slow one |
| `internals` | 6.5 min | |
| `frontend` | 2.6 min | |
| `extract` | **29 s** | activations are cheap; everything downstream reuses them |
| `l0` `l1` `l2` `lens` `cluster` `budget` `layer_screen` | 16–40 s each | |
| **end to end** | **~65 min** | on one RTX A5000 (23 GB), shared box |

Three things follow. Extraction being 0.7% of the run is why the 28 standalone
studies are built to read an existing run's artifacts rather than re-extract — a new
analysis over a finished run costs seconds, not an hour. `l3.patching.per_window` and
`window_stride` are the knobs that actually move total time; nothing else is close.
And most stages scale with model count, so a two-model run should land near
two-thirds of this — *estimated from the stage shape, not measured*, which is the
distinction this file tries to keep everywhere.

## 4. The analysis stack

17 pipeline stages (`pipeline.stage_names()` is the authority). Each exists because
of the previous one's limitation, and each states its evidence class in the report:

- **`extract`** — activations captured on a shared `[series, window, dim]` axis via
  measured token→time spans, stored chunked in zarr, gated by an impulse-alignment
  check that raises rather than warns.
- **`budget` / `frontend`** — parameters, FLOPs (**measured** with torch's
  `FlopCounterMode`, so it works on unseen architectures), latency, VRAM; and what
  each tokenizer does to the input before any layer.
- **`layer_screen`** — which of a model's own layers are worth expensive analysis.
- **`l0`** — accuracy per data family, paired bootstrap, Holm-corrected across
  **every model pair jointly**, plus quantile calibration. *(behavioral)*
- **`internals` / `lens`** — effective dimensionality, probe decodability, and where
  in depth the forecast crystallizes. *(descriptive / depth-resolved)*
- **`l1`** — linear CKA between every layer pair, against a shuffled-series null.
  *(geometric — correlational)*
- **`l2`** — ridge stitching, reported only as the **gain over an input-feature
  baseline**, because both models saw the same input. *(translatable)*
- **`l3`** — corruption-sensitivity fingerprints and within-model activation
  patching, per layer and per window. *(causal within-model)*
- **`attention` / `cluster` / `sae` / `exemplars`** — head taxonomy and ablation at
  two lag resolutions, partition agreement, sparse dictionaries with ground-truth
  feature alignment, and worked case studies.
- **`register` / `confirm`** — dev findings are frozen, then tested **once** against
  a sealed private corpus. This is the only part of the report that is confirmation
  rather than exploration, and the private corpus is treated as a consumable. Three
  kinds of claim are re-tested independently — accuracy differences per family, where
  in depth each model reacts to each corruption, and the strongest representational
  similarity — and the section opens with a **"What held up (N of M)"** roll-up so the
  answer is in one place rather than assembled from three tables. Claims the stage
  cannot yet replicate are listed with the reason, grouped by stage, rather than
  silently absent. Re-running a spent confirmation requires an explicit
  `--force confirm`, which warns and marks the artifact, so a second look cannot be
  mistaken for a first one.

Five architectures have hand-written adapters (TimesFM 2.5, Chronos-T5,
Chronos-Bolt, Chronos-2, Sundial), plus a zero-code `generic_hf` probe path verified
end to end on a checkpoint nobody wrote an adapter for. Capability **tiers are
derived from what an adapter implements**, never declared, and a tier-0 black box
that only forecasts still produces a real (smaller) report.

**The benchmark half**, which the stage list above consumes rather than replaces:
twelve generator archetypes composed from explicit, recorded components (trend,
any number of seasonalities with four wave shapes, AR noise, changepoints,
anomalies, unit roots, intermittency, heteroskedasticity), so every sample carries
**exact ground truth** — which is what the SAE and probe stages score against.
Corpora are split into a public dev half and a private test half drawn from
**disjoint seed ranges**, sealed with per-sample and global hashes, and admitted only
after a two-stage banded-DTW leakage audit against real reference data. Two
deliberate tiers are labelled per sample: `synthetic` (touches zero real data) and
`real_derived` (inherits a source distribution, and says so) — the honest resolution
of a brief that asked for both no-leakage and realism. A separate validation package
then asks whether a generated corpus is *actually* diverse, in catch22 feature space
and never on the UMAP coordinates it also plots.

## 5. What a result actually looks like

The distinguishing claim above ("derived by printed rules, not written as sentences")
is easier to judge from one real row of the report's scorecard than from the claim:

> **Most load-bearing attention head, Chronos-T5-Base (`encoder.block.6` · head 8)**
> — measured **0.235 ΔMASE**, compared against **0.156 ΔMASE** (*that model's own
> repeat-run noise floor*), rule **`value ≥ 2× reference`** → verdict **does not
> clear**.

Four properties are visible at once. The reference is something *this run measured*,
not a constant. The rule is printed, so 1.508× is arithmetic you can disagree with.
The floor is **per model** — a deterministic model's floor is exactly zero, so the
same row for TimesFM reads `clears` for a structural reason the note states rather
than hides. And the verdict is computed in `Verdict.__post_init__` from the rule, so
a call site *structurally cannot* author one; a test pins that passing `verdict=` is
overwritten.

The same run's scorecard carries 16 such rows across ten stages. The row count is a
property of the *run*, not of how many layers a stage was pointed at: a model analyzed
at five SAE layers contributes one row (its weakest layer against that layer's own
permutation null, with all five in the table beneath), and a three-model run's
cross-model rows report the weakest of the three pairs rather than the first pair —
so a claim on the scorecard is one that held everywhere it was checked. Each stage
supplies its own verdict vocabulary rather than a uniform pass/fail (`clears`,
`excludes zero`, `above null`, `separated`), and where a reference does not exist the
row renders **"not comparable"** rather than a pass or a fail — which is what lets a
single-model run render honestly.

Alongside the scorecard, that run carries 77 **findings**, each rendered in three registers
so one document serves a skimmer and a specialist without writing two. A real one,
verbatim:

> **plain** — *"At their most similar layers, TimesFM and Chronos-T5-Base organize the
> data somewhat alike."*
> **text** — *"[exploratory — not pre-registered] L1 — peak similarity CKA=0.38 (95% CI
> [0.36, 0.41], series bootstrap) at TimesFM L4 ↔ Chronos-T5-Base L10 (relative depths
> 0.21 / 0.43, block axis); shuffled-series null ≈0.04 …"*
> **caveat** — *"This is a geometric similarity measure … not evidence that the models
> compute anything the same way. Exploratory: found by looking at the dev corpus, where
> many comparisons were tried … Within the captured surface only (~57% of TimesFM's
> forward computation is unobserved; ~86% of Chronos-T5-Base's)."*

The caveat is **generated, not written** — composed from the finding's evidence class,
whether it was pre-registered, whether a noise floor was checked, and the run's own
measured capture coverage. That is why it can name 57% and 86%: those are this run's
numbers, and a run with better coverage produces a weaker caveat automatically.

Findings are also **de-duplicated at render time**, which matters more than it sounds.
A stage that measures every model emits one claim per model — structurally correct,
since each is a separate measurement, and on a three-model run that made 19 of 44 claim
templates repeat verbatim except for a name. The first of each look-alike group renders
normally and the rest collapse into a *"N more of the same kind"* block: nothing is
summarized, so nothing can be summarized wrongly, and the rule applies to any future
per-model family without another edit. Likewise the **fairness card renders one
expandable card per model pair** rather than picking a reference pair, because the
report cannot know which two models a reader came to compare.

One deliberate boundary is worth naming: the report contains no reference to this
repo's own planning documents. The source deliberately cites them — that provenance is
how a number is traced back to the decision that produced it — and they are stripped at
the render boundary rather than removed from the source, keeping the actionable half of
each citation (`(<doc> §18 F1 -- enable the l3 stage)` becomes `(enable the l3 stage)`).
Stripping on the way out is the only version that stays true: the methods appendix
renders analysis modules' docstrings verbatim, so editing report strings could never
have reached all of it.

## 6. What makes it trustworthy

These are the things worth copying even if you never run the pipeline:

1. **Negative results are kept.** The flagship crosscoder lost against its own
   pre-registered decision rule and the loss is written up rather than re-searched.
   A cross-model reliability heuristic lost to a free baseline in 10 of 11 cases and
   was deliberately *not* wired in.
2. **Every headline number has a floor beside it.** Untrained-weight twins,
   shuffled-series nulls, label-permutation nulls, repeat-run noise floors,
   `p`-floors from the bootstrap itself. The report's scorecard prints the rule that
   decided each verdict, so a reader can disagree with the arithmetic rather than
   only with the English.
3. **Failures are recorded as traps with their cost.** The engineering log is **47**
   entries of "this looked fine and was wrong, here is the mechanism." Several would
   otherwise have been rediscovered repeatedly.
4. **Missing capability degrades loudly and by name.** A dropped section says *why*
   (tier, run shape, absent artifact), never "artifacts missing".
5. **The measurements that could silently measure nothing have two-sided controls.**
   The sharpest bug found to date produced no exception and no missing artifact: a
   model whose forecast head reads different positions than activation patching
   writes returned a *bit-identical* forecast at every layer, which renders as a
   clean flat depth curve — a publishable shape. The one-sided control (patch a
   layer into itself, require exactly 0) passes for a hook writing into the void.
   Adapter conformance now also requires that patching a *different* layer moves the
   forecast, and an adapter may declare itself exempt only if that second check
   genuinely fails — so the exemption cannot rot in either direction.

## 7. Honest usability assessment

**Strong:** one command to a complete report; a preflight doctor; 107 `tsfm_lens` test
modules plus 11 for the benchmark half; a glossary, per-stage "what this tells you"
docs, a worked example that is itself a regression test, and a failure-mode gallery.
CI runs the suites and the smoke pipeline under **three dependency regimes** —
pinned-by-pip, the exact `DEPENDENCIES.md` conda recipe, and *latest-of-everything,
unpinned* — on a schedule. That third job is the interesting one: nearly every
expensive trap in that engineering log was a library's own breaking release
(`datasets>=3` dropping script-based loading, `tsbootstrap`'s API rewrite, `zarr` v3
renaming a method), each of which cost a working session to rediscover. A scheduled
unpinned job converts that class of failure into a build notification.

The panel path is demonstrated rather than claimed: `full_report_run_3model.yaml`
renders 15 sections / 78 findings / 79 figures with 0 bare and 0 failed on three
real checkpoints, and the two-model numbers reproduce **bit-exactly** inside it
(L1 peak CKA `0.38115179538726807`, L2 gains 0.4132/0.3179), so adding a model
does not silently rewrite the pair's results.

**Rough edges, in the order they will bite you:**

- **Config sprawl.** ~47 configs in one flat directory, most of them frozen
  experiment records rather than starting points. `python run.py --list-configs`
  now prints them grouped by run shape with each file's own purpose line, derived
  from the files rather than from an index that could go stale. The four you
  usually want are `smoke.yaml`, `smoke_solo.yaml`, `smoke_panel.yaml`,
  `default.yaml`.
- **The full suite runs in ~14 min, not the ~3 hours previously recorded.** Measured
  2026-08-31: **945 passed, 1 skipped, 0 failed, 930.08s**, with 2 long-standing warnings.
  The repo's recorded 3:06:16 could **not** be reproduced and is left unexplained
  rather than quietly overwritten — competing load on a shared box or a cold cache are
  the untested candidates. Treat the wall clock as a range, not a constant: the same
  suite measured 551.69s earlier the same day on a smaller test count, and this box is
  shared by 30 users. (`test_smoke.py` dominates it — the smoke *pipeline* runs in 29 s,
  so that module's cost is its several additional end-to-end reruns, not the pipeline
  being slow.)
- **Thread counts are now capped for you in the test suites, but not elsewhere.**
  numpy, OpenBLAS, MKL and torch each default to *every* core and none knows about
  the others, so the suite was measured spawning **57 threads at nice 0, taking 18 of
  32 cores** on a box shared by 30 users. Both packages now ship a
  `tests/conftest.py` that caps this at 4 before numpy is imported, prints the policy
  in force (even under `-q`), and defers wholesale to any thread variable you export
  yourself. What it reliably buys is **core footprint**: peak threads drop 42 → 14 for
  `tsfm_lens` and 43 → 15 for `tsfm_benchmark`, and the suite's CPU share falls from
  18 of 32 cores to under 3. What it buys in *time* depends on the workload and should
  not be quoted as one number: a module of many small linear-algebra calls is **1.78×
  faster** capped (`test_crosscoder.py`, 17.4 s vs 31.0 s, warm both ways), while the
  whole `tsfm_benchmark` suite is unchanged (24.5 s either way across repeats). A
  full-suite A/B was attempted and abandoned without a number: both arms were
  confounded (tests were added between them, and report regenerations competed for
  the same cores), so the apparent gap is not a measurement. Read the 1.78x as one
  module's number, not the suite's. Override with `TSFM_TEST_THREADS=<n>`, disable
  with `0`, opt into renicing with `TSFM_TEST_NICE=19`.
  **The `run_*.py` studies and pipeline runs are still uncapped** — for those, prefix
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 19` by hand,
  which took the load average from 54.5 to 32.1 and this process's share to 2.9 cores.
- **Real-checkpoint runs are not one-button** — they need a GPU, a working
  `timesfm`/`chronos-forecasting`/`transformers` install, and several load-bearing
  version pins (`zarr<3`, `datasets<3`) whose absence fails in non-obvious ways.
  `DEPENDENCIES.md` and `--doctor` exist precisely for this.
- **Univariate only, by decision.** A cross-series attention axis has no time
  interval and cannot be pooled onto the shared window axis; multivariate models are
  analyzed as univariate ones with a stated unmeasured axis.
- **Chronos-T5's decoder is uncaptured** — a depth-located claim about it is a claim
  about its encoder, and the report auto-qualifies it with the measured captured-FLOP
  fraction.

**Questions it cannot answer**, as distinct from the operational edges above — these
are limits of the method, not of the implementation:

- **"Is one model *mechanistically* better?"** No stage establishes that an internal
  property *causes* a behavioral advantage. `lens` is the closest bridge and was added
  for exactly that reason; it is still depth-resolved correlation, not attribution.
- **"What does feature X encode?"** Resolution stops at head and MLP granularity. You
  learn that family information is linearly present at layer *L*, not what is encoded.
- **"Do these two models share a circuit?"** Patching is deliberately **within**-model;
  what is compared across models is each one's restoration-by-depth *curve*. The
  flagship cross-model crosscoder was built to answer this directly and **lost** to
  independently-trained per-model dictionaries under its own pre-registered rule.
- **"Which model is better in general?"** Every result is conditional on one corpus,
  one context length, one horizon. The `confirm` stage tests a frozen hypothesis on a
  sealed private split — that is the strongest claim available here, and it is a claim
  about *this* benchmark.

**Who it is for.** Someone comparing TSFM checkpoints who cares more about whether a
comparison is *fair* than about getting a number quickly. It is not a
one-line-import library, and it does not pretend to be.
