# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); this
project does not yet follow strict semantic versioning across the pre-1.0
line, since the public interface (config schema, CLI flags, report contents)
is still evolving.

## [Unreleased]

## [1.0.0] - 2026-10-07

First tagged release published to PyPI as two packages, `tsfm-benchmark` and
`tsfm-lens` (both 1.0.0). The pipeline content is that of 0.1.0 plus the
additions listed here; the full list of what exists follows.

### Added before tagging

- `--bundle RUN_DIR` zips `report.html` and every small readable artifact with a
  `SHARE_MANIFEST.json` listing what was left out and why; `--prune RUN_DIR
  [--yes]` deletes the regenerable caches (SAE `*.pt`, `*.zarr` stores) of a run
  (a real 7-model run: 7.3 GB, of which about 110 MB is needed to read it). A
  pruned run keeps `pruned.json`; the pipeline then refuses every stage except
  `report`, and the report renders the same sections. A successful run logs its
  size and both commands. New module `tsfm_lens/share.py`.
- Data roles: every benchmark sample can carry one `role` (`synthetic`,
  `real_derived`, `external_real`), bound into its content hash and summarized
  by a data-roles card; a raw GIFT-Eval `external_real` slice loader; windowed
  `chronos_datasets` sources (ERCOT, KDD Cup, solar, taxi, M4 hourly and more);
  `mint_private.py` refuses any hash shared with an existing sealed split.
  Opt-in; role-free corpora hash exactly as before.
- Ablation battery: `sae.ablation_nulls` computes several nulls in one pass
  (`by_null` blocks; the first is primary), with feature-level chance estimates
  and results restricted to dictionaries that pass the admission gate and
  layers that pass the alignment gate.
- Claim types: `concept_atlas_centroid` (centroid-only, `min_members` 3), an
  external-real replication leg reported apart from the confirm verdict, and
  `family_presence` (implemented but not recommended: it fails its known-answer
  gate, FINDINGS MN-33). Claims record the null they are scored against.
- Controls: a `random_init` twin and a raw-input-statistics destination as
  negative controls for concept transfer; window-size sensitivity (64, 96) by
  exact averaging of stored windows; per-role breakdowns of accuracy, causal
  features and transfer.
- A subspace shared-input agreement test with its known-answer gate (opt-in; it
  fails the gate, FINDINGS MN-32, and is kept as the reproducer).
- `--doctor` warns instead of failing on an intentionally empty `confirm.path`.

### Packaging

- Both packages carry full metadata (authors, license, URLs, classifiers,
  keywords, readme). `tsfm-lens` exposes a `tsfm-lens` console script
  (`tsfm_lens.run:main`); `python run.py` from `tsfm_lens/` is unchanged.
- `tsfm-benchmark` now declares `scikit-learn>=1.3` as a core dependency
  (`benchmark_validation` imports it at import time; an install into an empty
  environment failed without it) and a `validation` extra (`pycatch22`,
  `umap-learn`, `plotly`). `tsfm-lens` gains a `benchmark` extra
  (`tsfm-benchmark>=1.0.0`). No existing pin changed (`DEPENDENCIES.md`).
- Wheels contain only the importable packages. Tests, configs, run outputs and
  examples stay in the repository; `tsfm-lens --list-configs` and the
  `configs/*.yaml` presets are read from a repository checkout.
- `.github/workflows/release.yml` builds both packages on a `v*` tag, publishes
  through PyPI Trusted Publishing and creates a GitHub Release. `.zenodo.json`
  carries the metadata for a Zenodo DOI. `docs/RELEASING.md` lists the owner
  steps.

### What the release contains

- `tsfm_benchmark`: parametric, random-parametric, mixture, block-bootstrap
  and sequential-PAR generators in two labeled leakage tiers; a banded-DTW
  leakage audit; sealed dev and private splits with per-sample hashes and
  epoch-based regeneration; shape-redundancy and catch22 feature-space
  validation; dev-to-private exchangeability checks.
- `tsfm_lens`: the 19 stages `corpus` through `report` (tier, run-shape and
  routing gates; per-stage config fingerprints); adapters for TimesFM 2.5,
  Chronos-T5, Chronos-Bolt, Chronos-2, Sundial and Lag-Llama, a zero-code
  `generic_hf` adapter and mock adapters; impulse-gated cross-model time
  alignment; behavioral comparison (L0), geometric (L1), linear-translatable
  (L2) and within-model causal (L3) analyses, attention analysis, activation
  clustering, TopK sparse autoencoders and the concept chain (ablation battery,
  concepts, transfer, atlas, seed stability, shared-input causal agreement).
- Held-out confirmation: `register` freezes dev hypotheses and `confirm` tests
  them once on the sealed private corpus; it refuses to overwrite its own
  artifact, requires seal verification, and records a forced rerun as
  `repeated_look`. Concept claims replicate under `concept_replication`.
- One self-contained interactive HTML report per run, with a derived scorecard
  and a model-comparison section.
- Examples: `examples/concept_atlas_v2/` and `examples/panel7_v2/` (config,
  lightweight run artifacts, rendered report), and the example configs in
  `tsfm_lens/configs/examples/`.

## [0.1.0] — first public release

First publish of both packages as a stable, documented pair.

### `tsfm_benchmark` (generation + validation)

- Two leakage tiers (`synthetic`, `real_derived`) with a two-stage banded-DTW
  leakage gate, a cross-split near-duplicate check, and a realism report.
- `parametric` (additive, exact ground truth) and `random_parametric`
  (archetype-drawn) synthetic generators; `mixture`, `block_bootstrap`, and
  `sequential_par` real-derived generators.
- Sealed public/private splits with per-sample and global integrity hashes,
  and epoch-based regeneration for a fresh, non-overlapping private split.
- A `chronos_datasets` real-data source alongside the original Monash
  adapter, rebalancing the real-derived tier away from a single dominant
  domain.
- `benchmark_validation`: shape-redundancy matching (DTW/cross-correlation)
  and catch22 feature-space diversity, kept separate until the report;
  dev↔private exchangeability checks (`cross_split.py`).

### `tsfm_lens` (the interpretability pipeline)

- A 19-stage pipeline (`corpus` → `extract` → `budget` → `frontend` →
  `layer_screen` → `l0` → `internals` → `lens` → `l1` → `l2` → `l3` →
  `attention` → `cluster` → `sae` → `concepts` → `exemplars` → `register` →
  `confirm` → `report`), with tier, run-shape, and time-localization routing
  gates that degrade a run gracefully instead of crashing it.
- Adapters for TimesFM 2.5, Chronos-T5, Chronos-Bolt, Chronos-2, Sundial, and
  Lag-Llama, plus a zero-code `generic_hf` adapter (verified on Timer) and a
  mock-architecture family for CPU-only smoke testing.
- Cross-model time alignment via overlap-weighted window pooling, verified
  empirically by an impulse alignment gate at extraction time and the
  `--check-alignment`/`--discover-spans` CLI commands.
- The evidence-class ladder: descriptive profiles, L1 (geometric, linear
  CKA/RSA), L2 (linear-translatable, ridge-stitching gain over an
  input-feature baseline), L3 (causal within-model, corruption sensitivity +
  activation patching), illustrative exemplars, and one-shot confirmation on
  a sealed private corpus.
- A sparse-autoencoder concept-discovery chain: per-target TopK SAEs, a
  causal ablation battery scored against a random-direction null, concept
  clustering, cross-model transfer testing, a cross-model concept atlas, seed
  stability, and a shared-input causal-agreement rung.
- A single self-contained interactive HTML report per run: a fairness card
  naming every measured asymmetry between compared models, a model-comparison
  section with derived answer boxes and a concept verdict table, and a
  progressive-disclosure Headline/Standard/Methods view.
- `render_stage_docs.py`, `render_glossary.py`, and `render_adapter_docs.py`
  keep the stage docs, glossary, and adapter registry documentation generated
  from source, never hand-maintained.

### Documentation

- Rewritten root, `tsfm_benchmark`, and `tsfm_lens` READMEs; a merged
  adapter-contribution guide (`tsfm_lens/docs/ADDING_A_MODEL.md`) with a
  worked case study of a real adapter bug (Sundial's input-normalization
  fix); config-system READMEs for both packages; and a curated set of example
  configs (`tsfm_lens/configs/examples/`) covering every hand-written and
  zero-code adapter family.
