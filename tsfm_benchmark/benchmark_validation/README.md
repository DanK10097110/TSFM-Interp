# Benchmark Validation Pipeline

A separate pipeline that interrogates a benchmark produced by the generation
pipeline. It answers two distinct questions with two distinct tools, because
they catch different failure modes:

1. **Is the benchmark full of redundant shapes?** Algorithmic sequence matching
   compares every pair of sequences and reports how many are near-duplicates.
2. **Does the benchmark cover a diverse range of time-series properties?** A
   catch22 feature space, summarised with quantitative diversity metrics and
   visualised in 3D, shows whether the sequences vary along many independent
   properties or collapse onto a few.

These are complementary. Two sequences can be redundant in shape (same seasonal
profile, different noise) yet separable in feature space, or distinct in shape
yet clustered on the same catch22 properties. Looking at only one view hides
half the picture.

## Stage 1 — Algorithmic sequence matching (`matching.py`)

Every sequence is first z-normalised and resampled to a common length — the
"equal sized buckets" step — so the comparison is invariant to length and
absolute scale. Then one of two matchers runs over all pairs:

- **Rolling cross-correlation:** best Pearson correlation over a band of integer
  lags. Captures linear shape similarity under small shifts; scale-invariant by
  construction, blind to nonlinear warping.
- **Localized DTW:** a Sakoe-Chiba banded dynamic time warp, distance mapped to
  a similarity. Additionally tolerant of local time warping; more expensive.

Outputs: the full similarity matrix, an **equal-frequency bucketed histogram** of
all pairwise scores (the redundancy profile of the whole set), the redundancy
fraction, and the explicit list of pairs above a redundancy threshold. The
pairwise step is O(n^2); pass `max_sequences` to score a random subset on large
benchmarks.

## Stage 2 — catch22 feature space (`features.py`, `embedding.py`, `diversity.py`)

Each sequence is passed through **catch22** (or catch24, adding mean and std),
the canonical low-redundancy feature set. Features are robustly scaled
(median/IQR) and non-finite values from degenerate sequences are imputed and
counted.

The single most important design rule here: **diversity is measured in the
feature space, never on the UMAP coordinates.** UMAP preserves local
neighbourhood structure but distorts global distances and densities, so cluster
sizes and gaps in the 3D plot are not quantitative. Using UMAP output to compute
a diversity number is a common and serious mistake; this pipeline computes every
metric on the standardised catch22 matrix and uses UMAP strictly for the eye.

Diversity metrics (`diversity.py`):
- **Effective dimensionality** — participation ratio of the PCA spectrum; how
  many independent feature axes the benchmark actually varies along.
- **Total variance** and the **per-feature variance ranking** — which catch22
  properties carry the spread.
- **Nearest-neighbour distance tail** (mean, min, 5th percentile) and the
  **near-collision fraction** — redundancy that a mean distance would hide.

Visualisation (`embedding.py`, `plot.py`): a 3D UMAP projection rendered as a
standalone interactive plotly file, coloured by the generator that produced each
sequence. A flagged PCA/t-SNE fallback keeps the plot working where UMAP is
unavailable.

## Running it

```
pip install -r requirements.txt
PYTHONPATH=../tsfm_benchmark:. python3 examples/run_validation.py
```

The demo builds a benchmark from five distinct generators plus five planted
near-duplicates, then validates it. On a sample run the planted duplicates
surface at similarity 1.0000, the pure-seasonal family shows up as genuinely
redundant under shape matching (same period-24 profile, ~5% redundant pairs),
and effective dimensionality lands near 4.6 of 24 — interpretable signals that
the test detects both planted and real redundancy.

## Honest limitations

- Cross-correlation and DTW measure *shape* similarity after z-normalisation, so
  they intentionally treat two series with the same shape and different noise as
  redundant. That is the right behaviour for a shape-redundancy test but means
  the redundancy fraction should be read alongside the feature-space metrics,
  not on its own.
- catch22 is general-purpose; if your interpretability tasks hinge on a property
  catch22 does not encode well, add targeted features rather than trusting the
  feature space to capture it.
- The matcher is O(n^2). Beyond a few thousand sequences, subsample or block.
