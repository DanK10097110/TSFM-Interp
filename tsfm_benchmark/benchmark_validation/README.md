# Benchmark Validation Pipeline

A separate pipeline that interrogates a benchmark produced by the generation
pipeline. `loaders.py` reads a sealed corpus (`corpus.jsonl`) directly, with no
import of the generator package, so everything downstream operates on the
actual samples the build pipeline wrote to disk -- not a re-derived or
idealised stand-in for them. It answers three distinct questions with three
distinct tools, because they catch different failure modes:

1. **Did the pipeline actually build what the config asked for?** Composition
   plots/report fields, computed from each sample's own provenance
   (`generator`, task name, tier, and -- for real-derived samples -- which
   real-world domains contributed), show the realized synthetic:real ratio and
   domain mix, which can (and does, once the leakage gate rejects some
   candidates) drift from the configured target.
2. **Is the benchmark full of redundant shapes?** Algorithmic sequence matching
   compares every pair of sequences and reports how many are near-duplicates.
3. **Does the benchmark cover a diverse range of time-series properties?** A
   catch22 feature space, summarised with quantitative diversity metrics and
   visualised in 3D, shows whether the sequences vary along many independent
   properties or collapse onto a few.

These are complementary. A corpus can hit its intended tier ratio while still
being full of redundant shapes, or have zero shape redundancy while still
collapsing onto a narrow slice of feature space. Two sequences can also be
redundant in shape (same seasonal profile, different noise) yet separable in
feature space, or distinct in shape yet clustered on the same catch22
properties. Looking at only one view hides most of the picture.

## Stage 0 — Composition and visual sanity check (`loaders.py`, `report.py`, `plot.py`)

Every record loaded from a sealed corpus carries whatever provenance the
generation pipeline actually recorded for it: `group` (generator name), `task`
(the specific `TaskSpec.name` that produced it -- needed because two tasks can
share a generator, e.g. two `mixture` tasks bootstrapping different Monash
domain samples), `tier`, and `domains` (the real-world Hugging Face configs
that fed a real-derived sample, extracted from its `source_refs`). Passing
`records=` to `build_report` fills in a `composition` section from these
labels, and three plots read the same data visually:

- **`plot_group_composition`** — bar chart of sequence counts per task,
  coloured by tier. Reads the *realized* synthetic:real split off the sealed
  corpus, which is not guaranteed to equal the configured one: the leakage
  gate can reject a different fraction of each tier.
- **`plot_domain_composition`** — bar chart of how many sequences drew on each
  real-world domain (only produced when the corpus has real-derived
  provenance; returns `None` for a purely synthetic benchmark).
- **`plot_example_sequences`** — a grid of actual raw series, a few per
  generator, sampled straight from the loaded records. Numeric diversity and
  redundancy metrics can both score a degenerate generator well; plotting real
  examples is the cheapest way to catch that a generator is producing garbage.

## Stage 1 — Algorithmic sequence matching (`matching.py`)

Every sequence is first z-normalised and resampled to a common length — the
"equal sized buckets" step — so the comparison is invariant to length and
absolute scale. Then one of two matchers runs over all pairs:

- **Localized DTW (default, `method="dtw"`):** a Sakoe-Chiba banded dynamic
  time warp, distance mapped to a similarity. Tolerant of local time warping
  (one regime lasting a bit longer, a seasonal period drifting), which is what
  makes it the more robust redundancy check. Scores the whole n×n matrix in
  one call to dtaidistance's parallel C `distance_matrix_fast` when installed
  (falls back to a pure-Python banded implementation, with progress logging,
  if it isn't) — batched rather than one Python-level call per pair, which is
  what makes DTW affordable as the default instead of only a fallback for
  small corpora.
- **Rolling cross-correlation (`method="xcorr"`):** best Pearson correlation
  over a band of integer lags, batched as one BLAS matmul per lag across all
  pairs at once. Cheaper than DTW but only tolerant of a uniform shift, blind
  to local warping — two sequences that are the same shape but locally
  stretched score low even though DTW would (correctly) flag them.

Outputs: the full similarity matrix, an **equal-frequency bucketed histogram** of
all pairwise scores (the redundancy profile of the whole set), the redundancy
fraction, and the explicit list of pairs above a redundancy threshold, sorted
by similarity. The pairwise step is O(n^2); pass `max_sequences` to score a
random subset on large benchmarks. `plot_redundancy_histogram` renders the
bucketed histogram; `plot_top_redundant_pairs` overlays the highest-similarity
pairs themselves (z-normalised) with each sequence's full creation provenance,
so a flagged pair can be eyeballed rather than trusted from a score alone.

## Stage 2 — catch22 feature space (`features.py`, `embedding.py`, `diversity.py`)

Each sequence is passed through **catch22** (or catch24, adding mean and std),
the canonical low-redundancy feature set. Features are robustly scaled
(median/IQR); non-finite values from degenerate sequences are imputed to the
per-feature median and counted, and *finite*-but-extreme scaled values are then
winsorized to ±5 (also counted) -- see "Honest limitations" below for why the
second step exists. `plot_feature_anomalies` plots the raw series behind
whichever sequences hit that clip hardest, so a value flagged only as a number
can be inspected as an actual shape.

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
  properties carry the spread. `plot_feature_variance` renders the ranking as
  a bar chart.
- **Nearest-neighbour distance tail** (mean, min, 5th percentile) and the
  **near-collision fraction** — redundancy that a mean distance would hide.

Visualisation (`embedding.py`, `plot.py`): a 3D UMAP projection rendered as a
standalone interactive plotly file, coloured by the generator that produced each
sequence. A flagged PCA/t-SNE fallback keeps the plot working where UMAP is
unavailable.

## Running it

```
pip install -r requirements.txt

# validate a real sealed corpus produced by the generation pipeline
PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_validation.py \
    --corpus /path/to/benchmark_out/public_dev --out outputs

# standalone demo, no real corpus needed (builds one from the generator package)
PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_validation.py --out outputs
```

Either way this writes `validation_report.json` plus standalone HTML plots to
`--out`: `composition.html`, `domain_composition.html` (real-derived corpora
only), `example_sequences.html`, `feature_space_3d.html`,
`feature_variance.html`, `feature_anomalies.html`, `redundancy_histogram.html`,
`top_redundant_pairs.html`, `length_scale_distribution.html`, and
`diversity_by_group.html` (only when some group has enough sequences to
report). For a corpus with more than a few thousand sequences, add
`--max-sequences N` -- the pairwise matcher is O(n^2) and otherwise dominates
runtime (catch22 extraction and the embedding are not subsampled by this flag
and stay comparatively cheap). Pass `--debug` to also write a full DEBUG-level
log of the run to a timestamped file under `logs/` at the repo root; without
it, the console still gets INFO-level progress for every stage.

The demo builds a benchmark from five distinct generators plus five planted
near-duplicates, then validates it. On a sample run the planted duplicates
surface at similarity 1.0000, the pure-seasonal family shows up as genuinely
redundant under shape matching (same period-24 profile, ~5% redundant pairs),
and effective dimensionality lands near 4.6 of 24 — interpretable signals that
the test detects both planted and real redundancy.

Run against an actual `configs/full_multidomain.yaml`-style build (657
synthetic + 247 real-derived sequences after the leakage gate), the
composition report reads `synthetic=657 (72.7%), realism_stress=247 (27.3%)`
against a configured 70:30 target, and `by_domain` lists all 12 Monash domains
that actually made it into the corpus with their sequence counts -- both
numbers come straight from each sample's own provenance, not from re-deriving
anything from the raw values.

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
- `diversity_metrics_by_group` scales every group against one `RobustScaler`
  fit on the *whole* corpus, so groups stay directly comparable -- but that has
  a real consequence: if a feature is nearly constant across most of the
  corpus (a tiny global IQR) while a minority group genuinely varies on it,
  that single feature can dominate the group's scaled variance and make an
  otherwise-fine group look collapsed by comparison. Observed in practice on
  the real corpus: `CO_trev_1_num` had a global IQR of ~0.008, so a handful of
  sequences with raw values around ±10-40 (near-constant runs the statistic is
  numerically unstable on) scaled to values in the *thousands* -- one column
  alone accounted for over 99% of total variance and collapsed
  `effective_dimensionality` to ~1.0 (of 22) corpus-wide, not just within one
  group. `extract_features` now winsorizes scaled values to ±5 specifically to
  cap this (see `plot_feature_anomalies` for which sequences triggered it), which
  fixed the corpus-wide collapse in that run (`effective_dimensionality` moved
  from ~1.0 to ~4.1). The clip is a mitigation, not a cure: a feature that is
  genuinely a much better discriminator for one group than the rest can still
  read as disproportionately important after clipping. This is why
  `_by_group_summary`/`plot_diversity_by_group` name the top-variance feature
  per group -- always check it before concluding a group has actually
  collapsed rather than just standing out on one feature.
