# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); this
project does not yet follow strict semantic versioning across the pre-1.0
line, since the public interface (config schema, CLI flags, report contents)
is still evolving.

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
