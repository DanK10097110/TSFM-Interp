# Running the pipelines — a walkthrough

This guide covers three things: running a small example to see the generation
pipeline work, running the full config-driven build, and running the
verification/validation report pipeline that lives alongside it. Every
command below has been run as written, from `tsfm_benchmark/`.

## Repository layout

```
tsfm_benchmark/
├── build_pipeline/              # generation: sources, generators, audit, sealing
├── benchmark_validation/        # verification: redundancy + diversity report
├── example_runs/                # CLI entry points for both halves (this file's home)
│   ├── run_smoke.py             #   small in-memory example (Part 1)
│   ├── run_full.py              #   full config-driven build (Part 2)
│   ├── run_validation.py        #   validation report (Part 3)
│   └── WALKTHROUGH.md           #   this file
├── configs/example.yaml         # declarative benchmark config used below
├── outputs/                     # run_validation.py's default --out target
└── requirements.txt
```

All three `example_runs/*.py` scripts are run with `PYTHONPATH=.` from inside
`tsfm_benchmark/` (so the sibling `build_pipeline`/`benchmark_validation`
packages resolve), not from `tsfm_benchmark/example_runs/` itself.

## Prerequisites

Python 3.10+. Install the package (from the repo root) plus the optional
real-data extras this walkthrough uses:

```
pip install -e .
pip install -e ".[real-data]"     # datasets, dtaidistance, tsbootstrap, sdv
```

The generation core runs on `numpy`, `scipy`, and (optionally) `dtaidistance`
(a fast DTW backend with a pure-numpy fallback). The validation pipeline
additionally needs `pycatch22`, `umap-learn`, `plotly`, and `scikit-learn`
(all in `tsfm_benchmark/requirements.txt`). Pulling real Monash/`chronos_datasets`
reference data needs `datasets` and network access to Hugging Face.

---

## Part 1 — A small example (generation)

Run the smoke test from inside the generation package:

```
cd tsfm_benchmark
PYTHONPATH=. python3 example_runs/run_smoke.py
```

This builds a tiny benchmark entirely in memory, seals the public and private
splits to temporary directories, reloads and integrity-checks the private
corpus, then prints a summary. Expected output (verified against this
checkout):

```
epoch        : 0
public_dev   : 15
private_test : 15
rejected     : 0
duplicates   : 0

sealed private reloaded+verified: 15 samples, visibility=private
...
public/private instance overlap (epoch 0): 0
reproducible across runs (same ids)      : True
intentional near-leak rejected           : True (dist 0.000)
```

The last two lines are the point of the example: the public and private splits
never share an instance, ids are reproducible, and a deliberately planted
near-duplicate of a reference series is rejected by the leakage gate.

### What "config" means here

The unit of configuration is a `TaskSpec` — one generator, repeated `count`
times, with an optional ordered chain of corruptions:

| Field | Meaning |
|---|---|
| `name` | Label for the task; also mixed into the per-sample seed. |
| `generator` | Which registered generator to call (e.g. `parametric`). |
| `count` | How many sequences this task contributes to each split. |
| `generator_params` | Keyword arguments passed straight to the generator. |
| `corruptions` | Ordered list of `{op, ...params}`; each is applied and recorded. |
| `tier` | `synthetic` (leakage-safe) or `real_derived` (real-derived; see `CLAUDE.md` §4). |

The smoke test hard-codes one `TaskSpec` in Python. Part 2 loads a list of them
from YAML instead.

---

## Part 2 — The full pipeline (config-driven build)

`configs/example.yaml` declares the benchmark. Each entry under `tasks:` maps
directly onto a `TaskSpec`. Annotated:

```yaml
tasks:
  - name: seasonal_trend_synthetic   # task label
    generator: parametric            # registered generator
    count: 200                       # 200 sequences per split
    tier: synthetic                  # leakage-safe tier
    generator_params:
      length: 512                    # points per sequence
      trend: {order: 2, scale: 0.5}  # quadratic trend, coeff spread 0.5
      seasonalities:                 # additive sinusoids
        - {period: 24, amplitude: 1.0}
        - {period: 168, amplitude: 0.4}
      ar_coeffs: [0.6, -0.2]         # AR(2) coloured noise
      noise_scale: 0.15              # base noise std
      n_changepoints: 2              # injected level shifts
      n_anomalies: 3                 # injected point anomalies
    corruptions:                     # applied in order, recorded in provenance
      - {op: jitter, sigma: 0.03}
      - {op: time_warp, strength: 0.1}
```

Run the build:

```
cd tsfm_benchmark
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml --out ./benchmark_out
```

For a quick end-to-end check without generating hundreds of series, cap the
per-task count:

```
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml --out ./benchmark_out --max-count 15
```

Flags actually accepted by `run_full.py` (run `--help` for the current,
authoritative list — this table is not exhaustive):

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/example.yaml` | Config to build from. |
| `--out` | `./benchmark_out` | Root for the sealed splits. |
| `--seed` | `0` | Base seed for the epoch. |
| `--epoch` | `0` | Epoch index; a new epoch uses a fresh, disjoint seed range. |
| `--references` | `none` | `monash` loads real reference series for the leakage gate. |
| `--reference-limit` | `100` | How many real series per Monash subset to load. |
| `--threshold` | `0.35` | Minimum leakage distance for a sample to be admitted. |
| `--max-count` | (none) | Cap per-task `count` for a fast run. |
| `--sources` | `none` | Inject real source series into source-dependent generators (`monash`). |
| `--source-limit`, `--source-subset`, `--source-n-domains` | — | Tune which/how many real series `--sources` injects. |

### What it produces

Two sealed directories, each holding a corpus and an integrity manifest:

```
benchmark_out/
├── public_dev/
│   ├── corpus.jsonl     # one sample per line: values + ground truth + provenance
│   └── manifest.json    # visibility, epoch, per-sample sha256, global digest
└── private_test/
    ├── corpus.jsonl     # the held-out split; do not publish
    └── manifest.json
```

`public_dev/` is meant to be released; `private_test/` is the held-out scoring
corpus and its directory must be access-controlled. To mint a fresh held-out
corpus for a later epoch (for example after the current one is suspected of
having leaked into training data), rerun with `--epoch 1`, which uses a
non-overlapping seed range.

### Notes on data volume and the leakage gate

Volume is set entirely by the config: roughly `sum(count) × 2 splits` per epoch.
No external data is required for the synthetic tier. Passing `--references none`
means the leakage gate has nothing to compare against and every sample passes
trivially — the sealed manifest records `extra.audit.gate.gate_effective=false`
so this is visible after the fact, not silently indistinguishable from a real
pass. Pass `--references monash` to load real series so the gate can actually
reject look-alikes. The real-derived tasks in the config (for example
`cross_domain_mixture`) are skipped by this runner because their generators need
real source series injected in code rather than declared in YAML — build those
by calling the `mixture`, `block_bootstrap`, or `sequential_par` generators
directly with loaded sources (or pass `--sources monash`), then feed the
results through the same builder.

---

## Part 3 — The verification report pipeline

This lives in `benchmark_validation/` (a subpackage of `tsfm_benchmark/`, not a
separate top-level package) and inspects a benchmark for redundancy and
diversity. Its CLI entry point is `example_runs/run_validation.py`, run from
the same `tsfm_benchmark/` directory as Parts 1 and 2. Point it at a sealed
corpus produced in Part 2:

```
cd tsfm_benchmark
PYTHONPATH=. python3 example_runs/run_validation.py --corpus ./benchmark_out/public_dev --out outputs
```

In `--corpus` mode it reads `corpus.jsonl` directly. Flags actually accepted
by `run_validation.py` (run `--help` for the current, authoritative list —
this table is not exhaustive; recent additions include `--blocked`/`--n-blocks`
for very large corpora and `--compare-splits` for the dev/private
exchangeability check in `cross_split.py`):

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | `./benchmark_out/public_dev` if it exists | Sealed corpus directory to validate. |
| `--demo` | off | Run the standalone demo benchmark instead of a real corpus (never the default). |
| `--out` | `outputs` | Where the report and plots are written. |
| `--method` | `dtw` | Matcher: `dtw` (localized, tolerates warping) or the cheaper `xcorr`. |
| `--catch24` | off | Use catch24 (adds mean and std) instead of catch22. |
| `--redundancy-threshold` | `0.97` | Similarity at/above which a pair is flagged redundant. |
| `--max-sequences` | (none) | Subsample for the O(n²) matcher on large corpora. |
| `--group-by` | `task` | Grouping key (`task`/`tier`/`group`/`archetype`) for per-group breakdowns. |

### What it produces

`--out` gets a `validation_report.json` (redundancy fraction and bucketed
similarity histogram, flagged near-duplicate pairs, catch22 feature summary,
and the feature-space diversity metrics: effective dimensionality,
nearest-neighbour distance tail, near-collision fraction, top-varying
features, per-group breakdowns and gate verdicts) plus several standalone
interactive Plotly HTML plots (composition, redundancy histogram, feature
variance, a 3D catch22/UMAP embedding, and more — the exact set is whatever
this version of the pipeline renders; treat the directory listing, not this
paragraph, as the source of truth). The 3D embedding is for visual inspection
only; every diversity number in the JSON is computed in the catch22 feature
space, never on those plotted coordinates (`CLAUDE.md` §5, invariant 4).

### Demo mode

With no `--corpus`, the runner builds its own small benchmark (five distinct
generators plus five planted near-duplicates) so the pipeline can be exercised
standalone:

```
cd tsfm_benchmark
PYTHONPATH=. python3 example_runs/run_validation.py --demo
```

The planted duplicates should surface at similarity 1.0000 in the report — a
quick confirmation that the redundancy detector works.

### End-to-end

The two pipelines chain cleanly: Part 2 writes `public_dev/`, Part 3 reads it
with `--corpus`. A full loop:

```
cd tsfm_benchmark
PYTHONPATH=. python3 example_runs/run_full.py --out ./benchmark_out --max-count 15
PYTHONPATH=. python3 example_runs/run_validation.py --corpus ./benchmark_out/public_dev --out outputs
```

---

## Appendix A — Config reference

### parametric generator params

| Key | Meaning |
|---|---|
| `length` | Points per sequence. |
| `trend` | `{order, scale}`: polynomial degree and coefficient spread. |
| `seasonalities` | List of `{period, amplitude, phase}` sinusoids. |
| `ar_coeffs` | AR coefficients for coloured noise (e.g. `[0.6, -0.2]`). |
| `noise_scale` | Base noise standard deviation. |
| `n_changepoints` | Number of injected level shifts (recorded as ground truth). |
| `n_anomalies` | Number of injected point anomalies. |
| `anomaly_magnitude` | Anomaly size, in units of `noise_scale`. |

### corruption ops

| `op` | Params | Effect |
|---|---|---|
| `jitter` | `sigma` | Add Gaussian noise scaled to the series std. |
| `scaling` | `sigma` | Multiply by a single random scalar near 1. |
| `time_warp` | `n_knots`, `strength` | Smooth random monotonic time-axis warp. |
| `dropout` | `rate` | Replace a fraction of points with forward-filled values. |

### LeakageAuditor params (set in code, or via `run_full.py --threshold`)

| Key | Default | Meaning |
|---|---|---|
| `target_len` | 256 | Common length sequences are resampled to before comparison. |
| `threshold` | 0.35 | Minimum nearest-neighbour distance to admit a sample. |
| `metric` | `dtw` | `dtw` (two-stage, shift-invariant) or `euclidean`. |
| `prefilter_k` | 10 | Euclidean prefilter size before DTW in the two-stage gate. |
| `dtw_window_frac` | 0.1 | Sakoe-Chiba band as a fraction of `target_len`. |

### validation params (`match_all`, `extract_features`, `embed_3d`)

| Key | Default | Meaning |
|---|---|---|
| `method` | `dtw` | Matcher type. |
| `length` | 256 | Equal-bucket length for matching. |
| `lag_frac` / `window_frac` | 0.1 | Cross-correlation lag band / DTW window. |
| `n_buckets` | 10 | Equal-frequency buckets for the redundancy histogram. |
| `redundancy_threshold` | 0.97 | Similarity at/above which a pair is redundant. |
| `catch24` | False | Add mean and std to the 22 catch22 features. |
| `n_neighbors` / `min_dist` | 15 / 0.1 | UMAP layout parameters (plot only). |

---

## Appendix B — The five verification checkpoints

| # | Check | When it fires | What it does |
|---|---|---|---|
| 1 | Leakage gate | Per sample, at build time | Rejects samples too close to a real reference series (two-stage DTW). |
| 2 | Near-duplicate check | Once, after both splits are built | Scans the union so nothing leaked across the public/private boundary. |
| 3 | Seal manifest | At seal time | Writes per-sample sha256 and a global digest. |
| 4 | Integrity verify | At load time (`load_sealed`) | Recomputes hashes and raises on any mismatch. |
| 5 | Validation report | Post-hoc, on the finished corpus | Redundancy and diversity; can run as a per-epoch CI gate. |
