# TSFM Interpretability Benchmark Pipeline

A pipeline for generating a time-series benchmark whose primary value is
**known ground-truth structure for interpretability** and a **defensible
position on data leakage**, rather than realism for its own sake.

## The core tension this design takes a position on

The stated goal — "no leakage into any model's training data" combined with
"make data similar to real data" — pulls in two directions. There are two
distinct kinds of leakage:

- **Instance-level** (the exact or near-exact series was in training). Auditable
  and preventable.
- **Distributional** (the model has seen the *same kind* of data). Anything
  derived from or made similar to real corpora carries this by construction,
  and making data "similar to real data" actively increases it.

You cannot maximise realism and minimise distributional leakage at the same
time. This pipeline therefore splits generation into two tiers and labels every
sample:

- **`synthetic` (leakage-safe backbone):** built from an explicit compositional
  process, touches no real data, and records every component as ground truth.
  This is the tier interpretability claims should rest on, because it is the
  only one with exact structure to score against.
- **`realism_stress` (real-derived):** mixtures, bootstraps, and learned
  sequential models. Useful for robustness, inherits the source distribution,
  and is admitted **only after passing the leakage audit**.

A second, orthogonal defence handles *future* contamination (publishing the
benchmark gets it scraped), and combines two mechanisms:

- **A sealed, held-out private corpus** (`seal.py`): the private test split is
  materialised to an access-controlled directory that is never published, sealed
  with a manifest that hashes every sample and a global digest. Loading verifies
  those hashes, so tampering, truncation, or a public/private mix-up is caught.
  It draws from a disjoint seed range from the public split, so the two share a
  distribution but never an instance.
- **Epoch-based regeneration**: each epoch uses a fresh, non-overlapping seed
  range. When an old corpus is suspected of having leaked into training data,
  `regenerate_private` produces a brand-new sealed held-out corpus, LiveBench
  style, while every prior epoch stays reproducible from its private seed.

You publish the generator and protocol; the canonical scoring numbers come from
the sealed private corpus.

## Generators and archetypes

Two generator families back the two tiers:

- **`parametric`** (synthetic) is additive: trend, one or more seasonalities,
  AR-colored noise, changepoints, point anomalies, an optional random-walk
  component, intermittency, and heteroskedasticity. Every component's exact
  parameters are recorded as ground truth.
- **`random_parametric`** (synthetic) draws a whole recipe at once from an
  **archetype** — a named, qualitatively distinct structural regime (e.g.
  `trend_dominant`, `seasonal_dominant`, `regime_switching`,
  `noisy_chaotic`). The original 8 archetypes are a **frozen default** used
  by every sealed corpus's `rng.choice(len(names))`; 4 more
  (`random_walk_drift`, `intermittent_bursts`, `amplitude_modulated`,
  `nonsinusoidal_seasonal`) are opt-in via an explicit `archetypes: [...]`
  list in a task's `generator_params`. **Never edit the default archetype
  list** — growing it changes what index every existing seed maps to and
  silently breaks bit-exact regeneration of every sealed corpus already
  built from it.
- **`mixture`, `block_bootstrap`, `sequential_par`** (`real_derived`) inject
  real source series — a weighted/multiplicative mixture, a moving-block
  bootstrap, or a learned sequential synthesizer (SDV `PARSynthesizer`) —
  and record full attribution provenance. They inherit the source
  distribution, so every sample is admitted only after the leakage audit
  passes (see below).

A generator that inherits its source series' length (`block_bootstrap`) can
silently produce an empty family if the source is shorter than
`context_len + horizon`; pass `min_length` in `source_config` to guard
against this, and smoke-build any new corpus config with `--max-count` first
to check the realized per-family length range.

## Leakage gate

Two-stage and shift-invariant. A cheap z-normalised Euclidean nearest-neighbour
search prefilters the reference corpus to the closest `k`, then windowed **DTW**
is computed only against those and the minimum is the score. Measured here at
~3 ms per query against a 1000-series corpus versus ~2.4 ms for Euclidean alone
and ~62 ms for DTW against the full corpus, so DTW costs about a third more than
plain Euclidean rather than 26x more. DTW uses the `dtaidistance` C backend when
installed and falls back to a banded numpy implementation otherwise. DTW and
Euclidean distances live on different scales, so `threshold` must be calibrated
per metric.

## Sealing, epochs, and validation

Sealing and epoch-based regeneration are the defence against *future*
contamination (the benchmark getting scraped once published):
`seal_corpus`/`load_sealed(verify=True)` hash every sample and the whole
split, so tampering, truncation, or a public/private mix-up is caught on
load. Public and private splits are drawn from disjoint seed ranges, so they
share a distribution but never an instance. When an existing private split
is suspected of having leaked into a model's training data,
`regenerate_private` mints a fresh epoch from a new, non-overlapping seed
range — every prior epoch stays independently reproducible from its own
seed.

Once a corpus is built, **validate it** with the companion
`benchmark_validation` pipeline before trusting it for anything — shape
redundancy (cross-correlation/DTW) and catch22 feature-space diversity are
kept separate until the report, and diversity is always computed in feature
space, never on the UMAP coordinates used only for the plot. See
[`benchmark_validation/README.md`](benchmark_validation/README.md) for how
each check works, and [`example_runs/WALKTHROUGH.md`](example_runs/WALKTHROUGH.md)
for the full command-by-command walkthrough of both pipelines.

## CLI quickstart

Every command below runs from inside `tsfm_benchmark/`:

```bash
# Build a corpus from a YAML config (a few seconds with --max-count)
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml \
    --out ./benchmark_out --max-count 15

# The same build, with the leakage gate doing real work against Monash
# reference series (omitting --references leaves the gate a documented no-op)
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml \
    --out ./benchmark_out --references monash --threshold 0.35 --epoch 0

# Mint a fresh, non-overlapping private epoch (e.g. after a suspected leak)
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml \
    --out ./benchmark_out --epoch 1

# Validate the resulting public split
PYTHONPATH=. python3 example_runs/run_validation.py \
    --corpus ./benchmark_out/public_dev --out outputs
```

`run_full.py --help` and `run_validation.py --help` are the authoritative
flag lists; see [`example_runs/WALKTHROUGH.md`](example_runs/WALKTHROUGH.md)
for what each flag does and what each command produces.

## Adding a real-data source

Real data is used only as (1) a leakage-gate reference, (2) realism
calibration, and (3) raw material for the `real_derived` generators — never
copied into the benchmark directly (invariant 10). `sources.py`'s
`load_sources` works against any Hugging Face dataset repo id; `kind: monash`
is a convenience alias for `Monash-University/monash_tsf`.

`kind: chronos_datasets` is the second, newer real source
(`autogluon/chronos_datasets`), added because several long, strongly
periodic Monash domains (`electricity_hourly`, `traffic`) fail to load
through `monash_tsf`'s own script (a pandas frequency-alias bug), leaving the
real-derived tier dominated by `weather`. `chronos_datasets` mirrors the same
domains as plain parquet, sidestepping that bug entirely (and isn't
script-backed, which matters for the `datasets<3` pin). Example
`source_config`, straight out of `configs/large_run_v2.yaml`:

```yaml
tasks:
  - name: real_electricity_hourly
    generator: mixture
    generator_params: {mode: weighted_sum, weight_concentration: 0.7}
    source_sample_size: [3, 8]
    source_config:
      kind: chronos_datasets
      subset: monash_electricity_hourly   # one of chronos_datasets' own config names
      limit: 300
      seed: 11
      min_length: 576                     # >= context_len + horizon
      max_length: 4000                    # longer series are sha256-windowed, not truncated to a fixed leading slice
```

Selection and windowing are both seeded via sha256 mixing (never Python's
salted `hash()`), so the same config reproduces the same sample. Adding a
different source is the same shape: register a new `kind` in `sources.py`
that yields the canonical `(values, SourceRef)` form, record its license on
the `SourceRef`, and add it to the leakage-reference loader if it should also
serve as an audit reference. The `TIME`, `BOOM`, and `ARFBench` adapters are
left as registration points for a future contributor — their current access
paths/licenses weren't verified here.

## Reproducibility: golden hashes and opt-in generator options

Sealed corpora used across this repo's own runs are pinned by golden hashes
in `tsfm_benchmark/tests`, so a code change that silently alters what a
config builds is caught by the test suite, not discovered downstream in a
model-comparison run. This is why **new generator options are always
opt-in** (the archetype extension above is the canonical example): a
default-list change that reorders `rng.choice`, or a new corruption inserted
into an existing pipeline, changes every existing seed's output and breaks
bit-exact regeneration. If you add a knob, default it to reproduce today's
behavior exactly, and require an explicit opt-in to get the new one.

## What I deliberately recommend against

- **LLM in-context generation of series — dropped.** It is the weakest fit for a
  leakage-free interpretability benchmark: the LLM's training corpus is unknown
  and vast (so generated series can *echo* memorised data — the opposite of
  leakage-free), it yields no ground-truth structure, it is costly and
  non-reproducible, and it couples the benchmark to one model's biases. It is
  not implemented and is out of scope for this pipeline.
- **Adding math models on top of real series** muddies attribution: the trend or
  changepoint you injected is no longer the only structure present, so the
  ground truth is partial. Prefer building structure from scratch (the
  parametric generator) and use real data only as audit reference and realism
  calibration.

## Architecture

```
sources.py     real corpora -> canonical form (reference + calibration only)
generators.py  parametric (synthetic) | mixture, bootstrap, PAR (real-derived)
corruptions.py composable transforms; each records itself into provenance
audit.py       leakage NN-distance gate, near-duplicate check, realism report
builder.py     config -> build -> audit gate -> public/private split
schema.py      values + provenance + ground truth travel together, always
registry.py    string-addressable plugins for the declarative config
```

The leakage gate is the load-bearing flow: nothing enters the test set unless
its nearest-neighbour shape distance to the real corpus exceeds a threshold and
it carries complete provenance.

## Status of each component

- **Implemented and tested** (`examples/run_smoke.py`, numpy/scipy/dtaidistance):
  schema, registry, parametric generator, mixture generator, all four
  corruptions, the two-stage DTW leakage audit, near-duplicate and realism
  checks, the builder with reproducible seeds and epochs, and the sealed
  held-out private corpus with verified load. The smoke test confirms a planted
  near-leak is rejected, public and private splits never share an instance,
  regenerated epochs are disjoint, ids are reproducible across runs, and a
  tampered sealed corpus fails its integrity check.
- **Wired but unverified here** (need live network and a version pin): the
  Monash source adapter, `block_bootstrap` (tsbootstrap), `sequential_par`
  (SDV). Their APIs drift across releases, so they are explicit integration
  points rather than assertions that they run.
- **Not implemented by choice:** LLM generation (dropped); the `TIME`, `BOOM`,
  and `ARFBench` source adapters — I could not verify their current access paths
  or licenses, so I left the `monash` adapter as the template to copy once those
  are confirmed.

## Library notes (relevant to maintenance)

`sktime` and `datasets`/`SDV` are actively maintained. `tsaug` and `timesynth`
are largely unmaintained and can break on recent numpy; the numpy parametric
core exists so the pipeline does not depend on them. `tsbootstrap` is younger —
pin it. Real corpora carry licenses that may bind redistribution of derived
data; `SourceRef` records the license so this stays honourable.
```
