# Running the pipelines — a walkthrough

This guide covers three things: running a small example to see the generation
pipeline work, running the full config-driven build, and running the
verification/validation report pipeline that lives in its own folder. Every
command below has been run as written.

## Repository layout

```
project_root/
├── tsfm_benchmark/              # generation pipeline
│   ├── tsfm_benchmark/          #   the package
│   ├── configs/example.yaml     #   declarative benchmark config
│   ├── examples/run_smoke.py    #   small example (Part 1)
│   ├── examples/run_full.py     #   full config-driven build (Part 2)
│   └── requirements.txt
├── benchmark_validation/        # verification report pipeline (the subfolder)
│   ├── benchmark_validation/    #   the package
│   ├── examples/run_validation.py   # validation run (Part 3)
│   ├── outputs/                 #   where reports and plots land
│   └── requirements.txt
└── WALKTHROUGH.md               # this file
```

If your copy nests `benchmark_validation/` inside `tsfm_benchmark/`, the only
thing that changes is the `PYTHONPATH` in the commands below — adjust the
relative path to point at wherever the `tsfm_benchmark` package sits.

## Prerequisites

Python 3.10+. Install each pipeline's dependencies:

```
pip install -r tsfm_benchmark/requirements.txt
pip install -r benchmark_validation/requirements.txt
```

The generation core runs on `numpy`, `scipy`, and `dtaidistance` (a fast DTW
backend with a pure-numpy fallback). `run_full.py` also needs `PyYAML`. The
validation pipeline additionally needs `pycatch22`, `umap-learn`, `plotly`, and
`scikit-learn`. Pulling real Monash reference data (optional, Part 2) needs
`datasets` and network access to HuggingFace.

---

## Part 1 — A small example (generation)

Run the smoke test from inside the generation package:

```
cd tsfm_benchmark
PYTHONPATH=. python3 examples/run_smoke.py
```

This builds a tiny benchmark entirely in memory, seals the public and private
splits to temporary directories, reloads and integrity-checks the private
corpus, then prints a summary. Expected output:

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
| `tier` | `synthetic` (leakage-safe) or `realism_stress` (real-derived). |

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
PYTHONPATH=. python3 examples/run_full.py --config configs/example.yaml --out ./benchmark_out
```

For a quick end-to-end check without generating hundreds of series, cap the
per-task count:

```
PYTHONPATH=. python3 examples/run_full.py --config configs/example.yaml --out ./benchmark_out --max-count 15
```

Flags:

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
trivially; pass `--references monash` to load real series so the gate can
actually reject look-alikes. The real-derived tasks in the config (for example
`cross_domain_mixture`) are skipped by this runner because their generators need
real source series injected in code rather than declared in YAML — build those
by calling the `mixture`, `block_bootstrap`, or `sequential_par` generators
directly with loaded sources, then feed the results through the same builder.

---

## Part 3 — The verification report pipeline

This lives in `benchmark_validation/` and inspects a benchmark for redundancy
and diversity. Point it at a sealed corpus produced in Part 2:

```
cd benchmark_validation
PYTHONPATH=. python3 examples/run_validation.py --corpus ../tsfm_benchmark/benchmark_out/public_dev --out outputs
```

In `--corpus` mode it reads `corpus.jsonl` directly, so it does **not** need the
generation package on the path. Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--corpus` | (none) | Sealed corpus directory to validate; omit for the built-in demo. |
| `--out` | `outputs` | Where the report and plot are written. |
| `--method` | `xcorr` | Matcher: `xcorr` (rolling cross-correlation) or `dtw` (localized). |
| `--catch24` | off | Use catch24 (adds mean and std) instead of catch22. |
| `--redundancy-threshold` | `0.97` | Similarity at/above which a pair is flagged redundant. |
| `--max-sequences` | (none) | Subsample for the O(n²) matcher on large corpora. |

### What it produces

Two files in `--out`:

- `validation_report.json` — the numbers: redundancy fraction and bucketed
  similarity histogram, flagged near-duplicate pairs, catch22 feature summary,
  and the feature-space diversity metrics (effective dimensionality,
  nearest-neighbour distance tail, near-collision fraction, top-varying
  features).
- `feature_space_3d.html` — a standalone interactive 3D plot of the catch22
  feature space (UMAP), colored by the generator that produced each sequence.
  Open it in any browser and rotate. The geometry is for visual inspection only;
  every diversity number in the JSON is computed in the feature space, not on
  these coordinates.

### Demo mode

With no `--corpus`, the runner builds its own small benchmark (five distinct
generators plus five planted near-duplicates) so the pipeline can be exercised
standalone. This mode imports the generation package, so put it on the path:

```
cd benchmark_validation
PYTHONPATH=../tsfm_benchmark:. python3 examples/run_validation.py
```

The planted duplicates should surface at similarity 1.0000 in the report — a
quick confirmation that the redundancy detector works.

### End-to-end

The two pipelines chain cleanly: Part 2 writes `public_dev/`, Part 3 reads it
with `--corpus`. A full loop:

```
cd tsfm_benchmark && PYTHONPATH=. python3 examples/run_full.py --out ./benchmark_out --max-count 15
cd ../benchmark_validation && PYTHONPATH=. python3 examples/run_validation.py --corpus ../tsfm_benchmark/benchmark_out/public_dev --out outputs
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
| `method` | `xcorr` | Matcher type. |
| `length` | 256 | Equal-bucket length for matching. |
| `lag_frac` / `window_frac` | 0.1 | Cross-correlation lag band / DTW window. |
| `n_buckets` | 10 | Equal-frequency buckets for the redundancy histogram. |
| `redundancy_threshold` | 0.95 | Similarity at/above which a pair is redundant. |
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
