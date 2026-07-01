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
