# tsfm-lens configs

## Structure

Every config is one YAML file with one top-level section per pipeline
concern — `run`, `data`, `alignment`, `models` (a list, one entry per
checkpoint), then one section per stage (`l0`, `l1`, `l2`, `l3`, `lens`,
`attention`, `clustering`, `layer_screen`, `sae`, `concepts`, `exemplars`,
`confirm`, `report`, `stats`, `internals`, `budget`, `corpus`). A stage
section's `enabled: false` (or omitting the section) turns that stage off;
the report renders whatever subset of artifacts actually exists on disk. See
[`default.yaml`](default.yaml) for a fully-commented real-model example and
[`smoke.yaml`](smoke.yaml) for the mock-model equivalent.

**Path resolution.** Every path *inside* a config (`data.path`,
`confirm.path`, `corpus.validation_report`, ...) is relative to wherever
`run.py` is invoked from — by convention, `tsfm_lens/` — never to the config
file's own location and never an absolute path (invariant 11). Invoke every
command in this doc, and every config here, from `tsfm_lens/`.

## Fingerprints and staleness

Each stage declares which config fields its own artifacts actually depend on
(`config_keys` in `pipeline.py`, whole-section or field-level). Before
skipping a stage whose artifacts already exist, the pipeline recomputes that
stage's fingerprint from the *current* config and compares it to the one
recorded when the artifacts were written; a mismatch refuses the skip rather
than silently reusing stale output. Rerunning a run whose config changed in a
field a stage doesn't declare needs nothing special; rerunning one where it
did needs `--allow-stale` (an explicit acknowledgement, not a silent
override).

## Listing and validating

```bash
python run.py --list-configs                    # every configs/*.yaml, shape/adapters/purpose derived from its own contents
python run.py --config configs/<name>.yaml --doctor   # full preflight, incl. loading every model
```

`--list-configs` derives everything it prints from the file itself (model
count → run shape, `enabled: false` flags → which stages are off), not from a
hand-maintained description — the shape and purpose columns cannot go stale
the way a hand-written index would. It only scans `configs/*.yaml` directly
(not subdirectories), so the curated files under `configs/examples/` (below)
don't appear in its output; validate those with `--doctor` instead.

## Configs on `main`

| Config | Purpose | Models | Corpus | Runtime / GPU |
|---|---|---|---|---|
| `smoke.yaml` | CI's own smoke config; every stage, two mock architectures | 2 mock | generated in-memory | CPU, ~30s |
| `smoke_solo.yaml` | Solo run-shape acceptance (comparison stages drop via the run-shape gate) | 1 mock | generated | CPU, ~30s |
| `smoke_panel.yaml` | Panel (3+ model) run-shape acceptance | 3 mock | generated | CPU, ~30s |
| `smoke_blackbox.yaml` | Tier-0 (black-box) capability acceptance | mock | generated | CPU, ~30s |
| `smoke_three_model.yaml` | Panel shape derived from model *count*, not filename | 3 mock | generated | CPU, ~30s |
| `smoke_encdec.yaml` | Encoder-decoder `block` depth-axis behavior (mock) | 2 mock | generated | CPU, ~30s |
| `default.yaml` | The general real-model template; every knob documented inline | TimesFM + Chronos-T5 | sealed, user-provided | GPU, minutes–hours depending on corpus |
| `concept_atlas.yaml` | The Concept Atlas preset (`CLAUDE.md` §6.8) — the one-command flagship chain | 4 real models | `benchmark_large` | GPU, hours |
| `concept_atlas_v2.yaml` | Same chain, rebalanced v2 corpus | 4 real models | `benchmark_large_v2` | GPU, hours |
| `full_report_run_4model.yaml` | The reference real run (`CLAUDE.md` names `runs/full_report_run_4model` as *the* reference) | TimesFM, Chronos-2, Sundial, Chronos-Bolt | `benchmark_large` | GPU, hours |
| `generic_hf_timer.yaml` | The zero-code `generic_hf` path, worked example (Timer vs. Chronos-T5-Small) | Timer (no adapter file) + Chronos-T5-Small | `benchmark_medium` | GPU, ~tens of minutes |

## Curated examples (`configs/examples/`)

A smaller, purpose-built set for someone landing on this repo — see
[`examples/README.md`](examples/README.md) for what each one demonstrates and
how it was validated. They are not part of `--list-configs`'s scan (see
above); load them with `--config configs/examples/<name>.yaml`.

## Experiment configs (dev branch only)

These reproduce a specific, already-recorded study rather than being a
general-purpose preset. Every one is real, working, and documented in its own
header — they're just not part of the curated publish surface, and live on
the `dev` branch (grouped here by the `ROADMAP.md` item they belong to; grep
`dev`'s `ROADMAP.md` for the section number, never read it in bulk):

| Group | Files | Reproduces |
|---|---|---|
| Scaling ladder | `_ladder/ladder_{base,tiny,small,mini,large}.yaml`, `scaling_ladder_chronos.yaml` | A generated model-size sweep (`run_scaling_ladder.py --emit-configs`) |
| Crosscoder Stage 0 | `crosscoder_stage0*.yaml` (6 files) | A joint-dictionary feasibility study |
| Layer-screen bake-off | `layer_screen_experiment*.yaml` (9 files) | The comparison that chose `work_bend` as the default layer-screening method |
| Adapter-development checks | `chronos2_adapter_check.yaml`, `chronos2_phase4_check.yaml`, `sundial_adapter_check.yaml`, `sundial_phase4_check.yaml`, `smoke_lag_llama.yaml`, `lag_llama_vs_chronos.yaml` | The original per-adapter validation runs; superseded on `main` by the renamed, curated copies in `configs/examples/` |
| Historical full-report runs | `full_report_run.yaml`, `_3model`, `_large`, `_large_revived.yaml`, `medium_run.yaml`, `medium_run_chronos_base.yaml` | Dated, superseded predecessors of `full_report_run_4model.yaml` |
| Medium-run variants/checks | `medium_run_chronos_base_auxk.yaml`, `_feature_ablation_check`, `_feature_steering_check`, `_l3_input_energy.yaml`, `e9_internals_null_check.yaml`, `f5_matched_resolution.yaml`, `frontend_check.yaml` | Targeted ablation/sanity studies |
| Null / random-weight floors | `null_chronos_random.yaml`, `null_chronos_small_random.yaml`, `null_timesfm_random.yaml` | `random_init` causal-floor measurements |
| SAE revival / panel | `sae_revival.yaml`, `sae_revival_searchfix.yaml`, `sae_stage0_panel.yaml`, `sae_stage0_panel_revived.yaml` | The dead-feature-revival recipe search |
| Distillation-lineage controls | `distill_negative_random_architecture.yaml`, `distill_positive_chronos_small_base.yaml`, `lineage_pair_l0.yaml` | A model-lineage detectability study |

## `tsfm_benchmark`'s own configs

See [`../../tsfm_benchmark/configs/README.md`](../../tsfm_benchmark/configs/README.md)
for the benchmark-generation side's config table.
