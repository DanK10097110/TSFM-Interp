# Benchmark configs

Each file here is a declarative corpus definition: a list of `tasks:`, each
mapping directly onto a `build_pipeline.builder.TaskSpec` (`name`, `generator`,
`count`, `tier`, `generator_params`, an optional `corruptions:` chain, and —
for real-derived generators — `source_sample_size`/`source_config`). Every
field is documented inline in `example.yaml`; the other configs assume you've
read it first and only annotate what they change.

## How to build from one

All commands run from `tsfm_benchmark/`:

```bash
PYTHONPATH=. python3 example_runs/run_full.py --config configs/<name>.yaml --out <out-dir>
```

Paths inside a config (e.g. a `source_config` referencing a local file) are
relative to wherever you invoke the command from, not to the config file
itself — by convention that's always `tsfm_benchmark/`. See
[`../example_runs/WALKTHROUGH.md`](../example_runs/WALKTHROUGH.md) for the full
flag reference and what each run produces.

## Configs on `main`

| Config | Purpose | Approx. size | Notes |
|---|---|---|---|
| `example.yaml` | The documented quickstart/reference config | ~800 sequences/split | Field-by-field inline comments; start here |
| `medium_run.yaml` | A tsfm-lens-sized corpus (~300 series/split) | ~300/split | Sized to run against real checkpoints in minutes |
| `full_multidomain.yaml` | The full multi-domain benchmark | Large | Real build; read `example.yaml` first |
| `large_run.yaml` | ~1000 series/split, fixes a real-derived weakness in `medium_run.yaml` | ~1000/split | Builds `benchmark_large` |
| `large_run_v2.yaml` | Same corpus, rebalanced real-derived tier via `chronos_datasets` (ROADMAP D1) | ~1000/split | Builds `benchmark_large_v2`; `large_run.yaml` is untouched (opt-in, invariant 1) |

`large_run.yaml`/`large_run_v2.yaml` are what `tsfm_lens`'s flagship
multi-model configs (e.g. `full_report_run_4model.yaml`,
`concept_atlas_v2.yaml`) are built from — see
[`../../tsfm_lens/configs/README.md`](../../tsfm_lens/configs/README.md).

## Experiment configs (dev branch only)

These reproduce specific one-off studies or dev-scale checks rather than
being general-purpose corpus presets. They stay fully functional and
documented in their own headers; they are just not part of the curated
publish surface:

| Config | Reproduces |
|---|---|
| `full_multidomain_run1.yaml` | The first tractably-scaled real run of `full_multidomain.yaml`, ratio-preserved down ~10x from a live throughput calibration |
| `real_data_smoke.yaml` | A fast end-to-end check of the real-derived path (all three real-derived generators, two real corpora) without `full_multidomain.yaml`'s cost |

## Reproducibility

Every one of the configs above traces to golden-hash-pinned tests in
`tsfm_benchmark/tests`. A code change that silently alters what a config
builds is caught there before it reaches a downstream model-comparison run —
see the root README's "Reproducibility" note and `CLAUDE.md` §2.1/§4:
archetype/generator defaults are frozen, new options are opt-in.
