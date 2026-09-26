# CLAUDE.md — TSFM Interpretability Pipeline (short version)

> This is the working manual: architecture, doctrine, invariants, and the
> lessons that recur. It replaces a ~490 KB version that is preserved verbatim
> as **`CLAUDE_FULL.md`** (full trap narratives, every dated reconciliation
> note, verification history). Grep `CLAUDE_FULL.md` for a `§11.N` trap number
> or a module name when you need the full story; do not read it end to end.
> `ROADMAP.md` is the research log (Findings blocks, recorded numbers, plans).
> It is huge — **grep it for the section you were pointed at, never read it in
> bulk**. Active work: `ROADMAP.md` §37 (the Concept Atlas, items P0–P8).
> Section numbers below (§2, §7, §11) match `CLAUDE_FULL.md`, so references in
> code comments still resolve.

---

## 1. What this project is

Mechanistic interpretability of **time-series foundation models (TSFMs)**,
comparing models with very different architectures on equal grounds. Usability
is modeled on `transformer-lens`, but this is a *comparative* tool (aligned
representations across models, statistics, held-out confirmation), not a
single-model microscope.

**Research questions:** what interpretable concepts TSFMs learn, whether those
concepts are shared across models, whether they causally affect the forecast,
where in depth the forecast forms, and what each model is better at.

| Half | Package | Job |
|---|---|---|
| Benchmark generation | `tsfm_benchmark/build_pipeline` | Leakage-audited, sealed, ground-truth-labeled synthetic corpora |
| Benchmark validation | `tsfm_benchmark/benchmark_validation` | Prove a corpus is diverse and non-redundant |
| Model analysis | `tsfm_model_analysis/tsfm_lens` | Layered pipeline ("tsfm-lens") → one interactive HTML report |

Two separately installable packages on purpose (root `pyproject.toml` =
`tsfm_benchmark`; `tsfm_lens` has its own). Their dependencies barely overlap.

**Models with adapters:** TimesFM 2.5 (decoder-only, 32-step patches,
deterministic), Chronos-T5 (encoder-decoder, 1 step/token, *sampled* decoder,
only the encoder is captured), Chronos-Bolt (encoder + patch head,
deterministic), Chronos-2 (encoder-only, adds cross-series GROUP attention,
which is out of scope), Sundial (decoder-only, flow-matching head, *sampled*),
Lag-Llama, a zero-code `generic_hf` adapter (verified on Timer), and mocks
(`mock`, `mock_encdec`, `mock_blackbox`, `mock_wave`). TimesFM is
**decoder-only**, and Chronos-T5 is the encoder-decoder. The capture asymmetry
follows from that: TimesFM's captured stack does everything, while Chronos-T5's
captured stack only *reads* the context.

---

## 2. Engineering doctrine — how code is written here

1. **Think three stages downstream before writing.** Canonical example: adding
   archetypes to a default list changed `rng.choice(len(names))` for every
   seed and broke sealed-corpus regeneration. New behavior is **opt-in**; the
   old behavior stays reproducible behind a knob.
2. **Prefer established libraries** (numpy, scipy, sklearn, torch, zarr v2,
   plotly, jinja2, pycatch22, dtaidistance, umap-learn, tsbootstrap, SDV).
   Deliberate exceptions: the metrics in `analysis/l0_behavioral.py` and the
   bootstrap designs in `analysis/stats.py` are short auditable code, because
   the formulas *are* the substance.
3. **Comments:** docstrings at module and function level. A module docstring
   says *why the module exists and what it inherits from the previous stage*.
   **No per-line comments.** Config YAML is the exception: it is documented
   inline because it is the user-facing surface.
4. **Verify empirically; distrust your first instinct.** Run it, measure it,
   then assert it. Several documented first diagnoses here were wrong.
5. **Degrade gracefully and loudly, never silently.** An unsupported
   capability is skipped with a stated reason that is *rendered in the report*,
   not only logged. An unresolvable slice disables the analysis rather than
   guessing. A broken assumption fails loudly. A log line is not loud once the
   deliverable is an HTML file.
6. **State the evidence class.** L1 is *geometric*, L2 *linear-translatable*,
   L3 *causal within-model*; also *descriptive*, *illustrative*, *behavioral*.
   Never let a correlational number read as causal. Name fallbacks in output
   (`pca_fallback`).
7. **Push back on contradictory framing.** Surface the tension, then design
   around it (e.g. "no leakage" vs "realistic" became two labeled tiers).
8. **Delegate long compute.** Anything expected to take 5+ minutes of GPU/CPU
   runs in the background (a background agent or `run_in_background`); do other
   independent work meanwhile and do not narrate results before they arrive.
   Background agents **report**; only the orchestrating session writes
   Findings into `ROADMAP.md`. A running job re-reads source files on every
   invocation, so do not edit files a per-iteration background sweep imports.
9. **Remove scaffolding once a decision closes.** When a comparison is decided,
   delete the losing code path and record in `ROADMAP.md` what was removed and
   which measurement decided it. This applies to code only. **Never delete
   prose** in `ROADMAP.md`/`CLAUDE_FULL.md`: correct it in place. Things that
   look redundant and are *not* dead: controls, nulls and floors that gate a
   verdict; fallbacks on paths that can fail; a second granularity that answers
   a different question; standalone `run_*.py` study drivers whose JSON a human
   reads. Establish deadness by measurement, not by grep.

---

## 3. Environment and commands

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate cudaPy
cd tsfm_model_analysis/tsfm_lens            # tsfm_lens paths/configs are relative to here
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4   # shared 32-core box
nvidia-smi                                   # pick a free GPU: CUDA_VISIBLE_DEVICES=N
python -m pytest tests/<file>.py -q          # tests/conftest.py caps threads itself
python run.py --config configs/smoke.yaml    # mock models, CPU, minutes; every stage
python run.py --list-configs                 # derived from the config files themselves
python run.py --config C --stages l1,report  # stage selection
python run.py --config C --stages X --force X   # --force only affects SELECTED stages
python run.py --config C --check-alignment <model>   # on every new checkpoint or library bump
python run.py --config C --check-adapter <model>     # one-command adapter checklist
python run.py --config C --doctor                    # preflight
python run.py --verify-provenance runs/<name>        # diff a run's recorded env vs now
python render_stage_docs.py --check && python render_glossary.py --check   # after doc edits
```

- `runs/` is gitignored. There is a **51 GB home quota**, so check `quota -s`
  before large copies. `/tmp` (tmpfs) has room.
- Exact library pins are in `DEPENDENCIES.md` (invariant 12). The load-bearing
  pins: `zarr<3` (v3 stores read back **empty** under v2 with no error),
  `datasets<3` (script-backed Monash), `tsbootstrap>=0.7` API, `mkl==2024.2.2`
  with numpy 2.1.0.
- Never write into an existing `runs/<name>` to test something. Copy it to a
  scratch directory first; the recorded numbers depend on those directories.
- Benchmark CLI (run from `tsfm_benchmark/`):
  `PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml --out ./benchmark_out --references monash`.
  Validation: `example_runs/run_validation.py --corpus <dir>`.

---

## 4. Benchmark generation (`build_pipeline`)

- **Two leakage tiers.** `synthetic` (`parametric`, `random_parametric`:
  touches zero real data, exact ground truth) and `real_derived` (`mixture`,
  `block_bootstrap`, `sequential_par`: attribution ground truth, but they
  inherit the source distribution). Every sample is labeled with its tier and
  must pass the leakage audit.
- `parametric` is additive: trend, seasonalities, AR noise, changepoints,
  anomalies, random walk, intermittency, heteroskedasticity. Every component
  is recorded as ground truth.
- `random_parametric` draws a recipe from an **archetype**. The original 8 are
  a **frozen default**, and 4 more are opt-in. Never "tidy"
  `_DEFAULT_ARCHETYPES`: that breaks golden hashes.
- **Audit:** two-stage banded-DTW leakage gate (the default threshold 0.35
  requires `--references monash`; with `none` everything passes),
  cross-split near-duplicate check, and a realism report (used for tuning, not
  as a leakage signal).
- **Sealing:** dev and private splits use disjoint seed ranges.
  `seal_corpus`/`load_sealed(verify=True)` hash every sample. Epochs
  (`_EPOCH_STRIDE`) mint fresh private corpora. Seed mixing uses sha256; never
  use Python `hash()`.
- A generator that inherits source length (`block_bootstrap`) can silently
  empty a family below `context_len + horizon`. Use `min_length` in
  `source_config`. Smoke-build new corpus configs with `--max-count` and check
  the per-family length range.

## 5. Benchmark validation

Shape matching (xcorr/DTW → redundancy fraction) and catch22 feature-space
diversity (participation-ratio effective dimension, nearest-neighbour tail,
near-collisions, per-group versions) are kept separate until the report.
🔴 **Diversity is computed in feature space, never on UMAP coordinates.** UMAP
is only for the plot. RobustScaler output is winsorized at ±5. `cross_split.py`
tests dev↔private exchangeability (energy distance plus TOST, three verdict
states).

---

## 6. tsfm-lens

### 6.1 Stages (`pipeline.stage_names()` is the authority)

`corpus → extract → budget → frontend → layer_screen → l0 → internals → lens →
l1 → l2 → l3 → attention → cluster → sae → concepts → exemplars → register →
confirm → report`

| Stage | Question | Evidence / limitation |
|---|---|---|
| corpus | What is in the benchmark, and can it be trusted? | composition, audit trust ladder |
| extract | Capture residual stream → zarr store, alignment gate | — |
| budget | Params by role, FLOPs (measured via `FlopCounterMode`), latency, VRAM, captured-FLOP coverage | cost at this context/horizon only |
| frontend | Tokenizer behavior before any layer (quantization, scale equivariance, NaN handling) | input/output only |
| layer_screen | Which layers deserve expensive analysis (`work_bend` default) | cheap proxy; own stride-1 store |
| l0 | Who is better, where: MASE/sMAPE/pinball per family, paired bootstrap, Holm across all pairs, calibration | behavioral |
| internals | Per-layer effective dimension, family-probe decodability, CKA-to-input | descriptive |
| lens | Where the forecast forms: skip lens and tuned ridge lens, crystallization depth | depth-resolved |
| l1 | Linear CKA every layer pair, every model pair; RSA | **geometric / correlational** |
| l2 | Ridge stitching; report only the **gain over the input-feature baseline** | linear-translatable, not causal |
| l3 | Corruption sensitivity fingerprints plus within-model activation patching (per layer and per window) | causal within-model |
| attention | Lag profiles, periodicity heads, head/MLP ablation, native and matched lag resolution | varies by architecture |
| cluster (L4) | Activation clustering, AMI across models | descriptive |
| sae | TopK SAE per target; fidelity, dead rate, forecast preservation, ground-truth alignment vs permutation null | correlational until causal battery |
| concepts | Ablation battery → concept clustering → cross-model transfer → deterministic descriptions | see §6.8 |
| exemplars | Per-family case studies from the MASE-gap distribution | illustrative |
| register / confirm | Freeze dev hypotheses; test **once** on the sealed private corpus | the gold standard |
| report | One self-contained HTML | — |

Gates:
- **Tier gate.** `_STAGE_MIN_TIER`, with the tier derived from the adapter.
- **Run-shape gate.** `_STAGE_MIN_MODELS`: solo, pair or panel, derived from
  `len(cfg.models)`. On a solo run, comparison-only stages are dropped with a
  stated reason.
- **Routing.** A model whose measured impulse response is not time-localized
  is routed to l0/budget/report only.

Each gate writes a JSON (`tiers.json`, `shapes.json`, `routing.json`), and the
report names that gate as the skip reason.

Stages self-skip when their artifacts exist. **Per-stage config fingerprints**
(`manifest.py`) refuse a stale skip. When `report` is selected, every stage with
artifacts on disk is fingerprint-checked. Fingerprint inputs are declared in
each `Stage`'s `config_keys` (whole section, `models[*].x`, or field-level
`"sae.transfer_top_k"`). A config field that no stage artifact depends on is
declared with `metadata={"stage_input": False}`. **Adding any field to a
whole-section key marks older runs stale**; rerunning those runs needs
`allow_stale=True`.

### 6.2 `ModelAdapter` contract (`models/base.py`)

**Required.**
- `load()`, `module`, `prepare()`, `forward()`, `predict(contexts, horizon,
  quantiles) → {"point", ...}`.
- `token_time_spans()`: `[n_tokens, 2]`, each token a contiguous time interval.
- `default_layer_regex`: anchor it (`^blocks\.\d+$`). The match is a substring
  `search`.

**Optional.** `postprocess_tokens`/`token_slice`, `attention_info`, `mlp_info`,
`attention_patterns`, `cross_attention_patterns`, `token_ids` (for probe-
amplitude calibration), `time_localization`, and
`forecast_reads_patched_positions`.

**Tiers are derived from what a subclass overrides, never declared.**

| Tier | Name | Requires |
|---|---|---|
| 0 | black box | `load`, `predict` |
| 1 | observable | + `module`, `prepare`, `forward`, spans |
| 2 | steerable | + `single_pass_context` |
| 3 | decomposable | + `attention_info` and `attention_patterns` |

Missing tier-1 methods raise `CapabilityUnavailable`. That is deliberately
*not* `NotTimeLocalized`, which is a measured verdict about a model.

**Adding a model:**
1. Try `adapter: generic_hf` plus `--probe-adapter` (zero code). It refuses
   rather than guesses.
2. If you need a hand-written adapter, copy `models/TEMPLATE_adapter.py` into
   `models/contrib/<name>_adapter.py` with a module-level `ADAPTER_NAME`. It is
   discovered by an AST scan.
3. Run `--check-adapter <name>`, then `render_adapter_docs.py`.

**`random_init: true`** loads the same architecture with random weights, as a
null twin. It is a valid floor for geometry and decodability. It is **not
automatically a causal floor**:
- TimesFM's twin is an exact identity stack, because its RMSNorm scale is
  zero-initialized.
- The Chronos-2 and Chronos-Bolt twins give numerically dead forecasts.

Check reach against a twin before running a battery on it.

### 6.3 Time alignment

- Raw token positions are not comparable across models.
  `extraction/alignment.py::pooling_matrix` overlap-pools tokens into windows
  (default `alignment.window: 32`).
- **Everything cross-model runs on `[series, window, dim]`.** Attention lag
  axes are multiplied by `token_width` before plotting.
- `run_alignment_gate` runs inside extraction. It probes every layer,
  self-calibrates the impulse amplitude, gates on hits relative to the
  attainable ceiling, persists its record, and fails by default.
- `extraction/span_discovery.py` *measures* token→time spans. Its two gates
  are peak:pedestal contrast and contiguity.
- The depth axis defaults to `alignment.depth_axis: block`, measured over the
  whole stack including uncaptured parts, so Chronos-T5's encoder tops out
  around 0.48. Read cross-model depth-*location* claims with that in mind.

### 6.4 Extraction and storage

- `hooks.py::token_patch` is the single intervention primitive, used by L3,
  the lens, the SAE and the concepts stage. It raises if the model chunks the
  batch: keep `max_series ≤ batch_size`.
- `ActivationCatcher` handles tensor/tuple/`ModelOutput` outputs.
- `store.py` is a chunked **zarr v2** store holding float16 activations.
  - `load(model, layer, level="series"|"window", rows=, space="act"|"sae")`.
  - `space="sae"` exists only when `sae.persist_features: true`.
  - Check `has_sae_features` before loading it.
- Forward passes run under bf16 autocast. **`capture_raw_tokens` (the clean
  cache that gets patched back) runs with autocast OFF**, because it must match
  `predict()`'s precision.

### 6.5 Statistics

- 🔴 **The resampling unit is always the series**, never windows.
  - L0: paired bootstrap.
  - L1: cluster bootstrap.
  - L2: validation-series CI with the probes held fixed.
  - L3: paired cluster bootstrap.
- Bootstrap p-values are floored at `1/n_boot`. A Holm family of `m` tests
  cannot go below `m/n_boot`. **Raise `stats.n_boot` before adding models**:
  the report flags an unsatisfiable correction.
- ΔMASE is read against each model's repeat-run **noise floor**
  (`l0/noise_floor.json`, `in_floor_units`). A deterministic model has floor 0,
  so the floor cannot referee it.

### 6.6 Exploration vs confirmation

Everything on the dev corpus is exploratory. `confirm` tests registered
hypotheses **once** on the sealed private corpus:
- It refuses to overwrite its own artifact.
- It requires seal verification.
- `--force confirm` is recorded as `repeated_look`.

A peeked private set means minting a new epoch. Concept claims replicate
under `confirmation.json`'s `concept_replication` key (P7; epoch 1 is
`benchmark_large/private_epoch1`, consumed by
`runs/full_report_run_4model_epoch1`). Unregistered findings cannot be
rescued. L0, L1 and L3 replicate on private data.

### 6.7 Report (`report/report.py`)

- **Builders.** Sections come from a `builders` list of `(eyebrow, title,
  blurb, requires, build_fn)`. A missing artifact renders as "stage not run";
  an exception renders as a visible "section failed".
- **Findings.** Each is a `Finding` dataclass (`claim_id`, `stage`,
  `evidence_class`, `text`, `plain`, `caveat`, ...), serialized to
  `report/findings.json`. Caveats are generated, never hand-written.
- **Figures.** `_note(purpose, reading, limitations)` gives a visible caption
  plus a "What does this mean?" dropdown; `_figcap` gives the caption alone.
  Every figure needs one, and all go through `_frag`.
- **The scorecard is derived.** It is built by `report/derived.py::
  bottom_line_rows`: a pure reduction with **no model names, no architecture
  names, no `cfg.models[i]` indexing**. A `Verdict` is derived from a `Rule`
  whose text renders in the same row, and a missing reference renders "not
  comparable".
- **Citation stripping.** Internal doc citations are stripped at render time
  (`report/sanitize.py`).
- **After editing the report, verify against the RENDERED HTML**: count
  figures and captions, grep the text. Do not verify from the diff.

### 6.8 SAE and the concept chain (active area — `ROADMAP.md` §37)

1. `sae` trains a `TopKSAE` per target.
   - Targets come from `sae.targets`, or `auto` via `layer_screen`, restricted
     to captured layers.
   - The dead-feature revival recipe is `aux_k` + `dict_size_policy: search` +
     `min_train_steps`. It is on in `configs/full_report_run_large_revived.yaml`
     and later configs, and off in the default.
2. The **ablation battery** (`sae/response.py::feature_ablation_fingerprints`,
   driver `sae/ablation_run.py`) zeroes one feature out of the SAE's own
   reconstruction on the series it fires on. It measures 9 forecast channels
   (`sae/concepts.py::CHANNELS`) against a random-direction null, gated by a
   reach probe: self-patch must give exactly 0.0, and a cross-layer patch must
   give a nonzero change.
   - Output: `sae/<model>/<layer>_ablation.json`.
   - The feature vector used downstream is `signed_effect / null_p95` per
     channel (`ablation_vector`).
3. **Concepts** (`sae/concepts.py::run_concepts`) cluster causal features per
   target with KMeans. A silhouette sweep runs over k values whose smallest
   cluster is ≥ `sae.concept_min_members` (3), and the result is written to
   `sae/concepts.json`.
4. **Transfer** (`sae/transfer.py`) tests whether another model groups the
   same series, against a stratum-matched null, in both directions. Output:
   `sae/transfer.json`.
5. **Deterministic descriptions** (`sae/describe_run.py`) write
   `sae/descriptions.json`. The LLM narrator path was pruned for features; it
   remains only in `sae/compare.py`.
6. The **`concepts` stage** (`sae/concept_stage.py`, config section
   `concepts:`) chains steps 2–5, then the cross-model **atlas**
   (`sae/concept_atlas.py`, complete-linkage cosine clusters of pooled causal
   features, ≥3 members), **seed stability** (`sae/stability.py`,
   `n_sae_seeds` replicate SAEs at the primary's dict size, within-model
   ceiling) and **atlas transfer** (`run_atlas_transfer`, BH per ordered model
   pair and leg; `transfer_p_method: exact|adaptive`), then **profiles**
   (`sae/concept_profiles.py` → `sae/concept_profiles.json`, per concept
   part: what it fires on, its effect, an exemplar, and whether its model
   is better on its top series; cross-model *input agreement* is the top-k
   overlap, hypergeometric + BH, never a whole-series ρ; a `sharing_class`
   of shared / partially shared / convergent / single-model; a part is
   provenance-driven when ≥80% of its top-k comes from one real-derived
   generator), then **shared-input causal agreement**
   (`sae/shared_input_agreement.py` → `sae/shared_input_agreement.json`, the
   L5 rung: for each reciprocal-FDR atlas-transfer test, both sides are
   ablated on the same series U; each side must clear its own
   random-direction null to be scorable, and level concordance (i) and
   shape cosine (ii) must beat both sides' activation-matched
   random-feature-set floors; `acts differently` needs a statistic below
   both floors' p05, otherwise `no specific agreement`), and writes
   `sae/concept_stage.json`. It
   requires `sae.persist_features: true`, which preflight checks. It skips
   transfer with a stated reason, and deletes its own stale artifacts when it
   does not rewrite them. The preset is `configs/concept_atlas.yaml`.
7. Report modules: `report/sae_concepts.py`, `sae_concept_map.py` (a shared
   StandardScaler+PCA fit across all targets; PCA, not UMAP, because the space
   is 9-dimensional and small-sample), `sae_roles.py`, `derived.py`
   (`concept_verdicts`: the evidence ladder L1–L6 per concept).
8. **Model comparison section** (`report/model_comparison.py`, directly after
   the scorecard): three derived answer boxes (shared / unique and why /
   pair similarity), the sharing map, concept cards and verdict table.
   Pair similarity comes from `analysis/model_similarity.py` →
   `report/model_similarity.json`: 7 pair metrics, each on its own scale,
   with a rank consensus (Kendall's W) and contrasts. Metrics are never
   pooled into one score.

The reference real run is `runs/full_report_run_4model` (TimesFM, Chronos-2,
Sundial, Chronos-Bolt; 13 targets). Its current per-target concept count is
**4** (11 of 13 targets are non-modular).

---

## 7. Invariants — do not break these

1. **Sealed-corpus bit-exact regeneration.** Golden hashes are pinned
   (`tsfm_benchmark/tests`). New generator options are opt-in.
2. **The series is the resampling unit.** Never bootstrap or split on windows.
3. **L2 reports the gain over the input-feature baseline**, never raw R².
4. **Diversity is measured in feature space, never on UMAP coordinates.**
5. **Patching is within-model.** Compare restoration-by-depth *curves* across
   models, never transplant activations between models.
6. **`confirm` runs once**, against a sealed private corpus.
7. **Alignment is verified empirically** (the gate plus `--check-alignment`)
   on every new checkpoint or library version.
8. **Unsupported capabilities skip with a rendered reason; broken assumptions
   fail loudly.** Never produce a wrong slice silently.
9. **The evidence class is stated** for every reported claim.
10. **Real data never enters the benchmark directly.** It is used only as a
    leakage reference, for realism calibration, and as raw material for the
    real-derived generators.
11. **No hardcoded absolute paths.** Paths are relative to the repo, to
    `__file__`, or to a CLI argument.
12. **`DEPENDENCIES.md` is kept current** with every pin change.
13. **Artifact compatibility.** When a panel or new feature needs a new key,
    add a canonical key, leave the legacy keys byte-identical, and read
    new-then-legacy.

---

## 8. Recurring lessons (condensed from `CLAUDE_FULL.md` §11)

Each line is a mistake that was actually made. `§N` points to the full story
in `CLAUDE_FULL.md`.

**Measurement and instruments**

- **Validate a probe against a case whose answer you already know** before
  trusting it on an unknown one. Wrong PIDs gave a plausible thread count of 1
  (§11.41). A probe whose stimulus degenerates returns a clean 0 (§11.56).
- **Silent failures produce well-formed output.** A patch the head never reads
  gives a clean flat curve (§11.42). A degenerate baseline gives a *confident*
  verdict (§11.37). The free controls:
  - patch a layer into itself and require exactly 0.0, then patch another
    layer and require a nonzero change;
  - predict twice with no patch at all and require identical output (§11.49).
- **Sampled models need seeded `predict()`** whenever two calls are compared:
  Chronos-T5 and Sundial (§11.50).
- **Match capture precision to patch precision:** autocast off for the clean
  cache (§11.49).
- **Normalized statistics:**
  - Ask what the statistic is normalized *by*. A share-of-total grows with
    token count (§11.33).
  - Compute a metric's attainable ceiling before thresholding it (§11.35).
  - A fixed probe constant is calibrated per checkpoint *and* per context
    length (§11.16, §11.26).
- **Controls must be scored:**
  - A control that explains nothing looks like one that works, so score it
    out of fold (§11.36).
  - One that explains *everything* leaves a residual of rounding noise that
    Spearman happily ranks (§11.48). Bound the control from both sides.
  - A bias shared by both arms of a null is invisible in the comparison, so
    publish the shared step's own diagnostic, e.g. convergence (§11.47).
- **Ties.** A rank transform that breaks ties by position turns a constant
  column into a ramp (§11.37). Cross-fitted residualization does the same to
  a binary field (a different fold mean per fold): ρ 0.833 fell to 0.328.
- **R² baselines.** A through-origin fit's R² needs an uncentered baseline
  (§11.31).
- **Absolute epsilons.** An absolute epsilon cannot tell "no effect" from
  "numerically dead". Use relative thresholds (§11.56).
- **Residualizing on a label that encodes the answer** removes the signal.
  An archetype (or a synthetic generator) is a structural recipe, so
  residualizing structure on it flagged 48 of 55 concept parts as
  provenance-driven. Residualize only on true confounds (ROADMAP §37.11).
- **Absence of agreement is not disagreement.** Failing to beat a floor's
  p95 is not evidence that two effects differ; that needs the lower tail.
  P5b's v1 called 109 of 288 tests "acts differently" on this basis; with a
  p05 test it was 9. Also: the null that asks "does this do anything" (random
  direction) and the floor that asks "is this more specific than an equally
  active feature" (matched features) are different nulls (ROADMAP §37.8).
- **Sparse features and whole-series ρ.** Two sparse SAE features can have
  ρ 0.80 with zero shared top series, and identical top series with low ρ.
  "Same inputs" is a top-k overlap test (ROADMAP §37.11).

**Plumbing**

- **Sampling.** Corpora are written grouped by task, so never head-slice
  (`[:n]`); use `utils.sample_rows(..., strata=families)`. After fixing a
  pattern, grep the whole module for it (§11.38).
- **Making a variable conditional** breaks sites that only *mention* it: log
  lines, template arguments. Grep the symbol before running (§11.39).
- **Artifact domains.** A fix that widens what one stage emits widens what it
  hands downstream. `layer_screen` names layers the store never captured
  (§11.40). Two lists derived from one source diverge under a stride (§11.32).
- **Composed degradation.** Two components that each degrade gracefully can
  jointly empty a category (§11.46).
- **Shared infrastructure.** A standalone experiment's meaning changes when
  shared code it uses changes. Run `git log` on the shared paths between two
  "identical" runs before hypothesizing nondeterminism (§11.24).
- **Import cycles** pass tests in one import order and fail in another. Make
  the import lazy, and test the other order in a subprocess (§11.52).
- **Library keys.** `FlopCounterMode` keys modules from the module *entered*
  (§11.28). A lookup miss returns nothing and looks deliberate.
- **Library dtypes.** scipy 1.18's `rankdata` keeps a float16 input's dtype,
  and the store is float16, so rank sums and AUCs were silently quantized.
  Cast to float64 before ranking; `auc_from_ranks` now refuses float16.
- **Artifact paths.** Build a path to another stage's artifact with that
  stage's own helper (`ablation_run.ablation_path`), never by hand: real
  layer names contain dots and are `sanitize()`d on disk. Mock layer names
  have no dots, so give fixtures a dotted name.
- **Keys that collapse.** A dict keyed by an id that under-specifies the
  claim silently merges entries. 20 registered concept claims had 9 ids
  (destination model, not layer), so Holm ran over 9 p-values. Refuse
  duplicate ids where the registry is built (ROADMAP §37.10).
- **Text I/O.** Always pass `encoding="utf-8"` to `read_text`/`write_text`
  (§11.17).
- **Remedies.** `--force` only reaches selected stages. An error message that
  names a flag must be tested as a command (§11.43, §11.44).
- **Upstream breakage.** Dependency releases break APIs (timesfm, datasets,
  tsbootstrap, a checkpoint's remote code). After any bump, rerun
  `--discover-layers` and `--check-alignment` (§11.8–§11.10, §11.22). Check
  *which layer* raised before trusting a refusal (§11.58).

**Reports and generated text**

- **Adjacent fields.** Two correct fields printed side by side get read as one
  claim. Compute the cross-tab (§11.54).
- **Labels are claims.** A label naming a transformation ("null-normalized")
  must be tested like code (§11.57). A shared colour scale silently aggregates
  channels on different natural scales.
- **Guards.** A text guard is only as good as its vocabulary, and a guard that
  cannot fire on the current run has not been measured. Run guards over every
  machine fallback; a guard that rejects its own fallback is wrong. Widening
  the evidence can make an existential check vacuous (§11.51, §11.53, §11.55).
- **Plants.** A planted regression that changes nothing is indistinguishable
  from a working guard. Confirm each plant actually broke the file, and read
  pytest's summary line: `1 error` is not `N failed` (§11.53, §11.55).

---

## 9. Verification standard

- Every new test's load-bearing assertion must be **confirmed to fail against
  a planted regression**. Report which plant failed which test.
- Unit tests use synthetic data with a **planted, known answer**, and the
  fixture must contain the confusable case on purpose (a decoy).
- After any report change, check the rendered HTML. After any stage change,
  run `tests/test_smoke.py` plus that stage's tests.
- Reproduce against a real run on an **isolated copy**. Quote numbers exactly;
  do not round them, since they become permanent Findings text.
- Report outcomes faithfully: failed tests, skipped steps, and numbers that
  moved.

## 10. Known limitations

- The Chronos-T5 decoder is not captured. Claims about Chronos depth are claims
  about its encoder. Captured-FLOP coverage is TimesFM ~42% (stride 2) and
  Chronos-T5-Base ~14% of a forecast.
- Relative depth across architectures is a convention; see `depth_axis`.
- TimesFM's finest resolvable lag is one patch (~32 steps). Cite the *matched*
  attention resolution for cross-model claims.
- Univariate only, by decision. Chronos-2's cross-series GROUP axis is out of
  scope.
- Models whose tokens are not time-localized, or read disjoint lags, route to
  L0 only.
- The matcher is O(n²). catch22 may miss task-specific properties.
- Real-corpus leakage into a model's own pretraining data cannot be verified.
