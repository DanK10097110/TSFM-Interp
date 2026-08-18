# TSFM Interpretability Pipeline

Mechanistic interpretability tooling for time-series foundation models (TSFMs) —
built to compare architecturally different models (TimesFM, Chronos, and others)
on equal footing, through leakage-audited synthetic benchmarking, layered
representational/causal analysis, and cross-architecture comparison.

Two parts, each independently installable and independently usable:

- **[`tsfm_benchmark/`](tsfm_benchmark/)** — generates leakage-audited, sealed,
  ground-truth-labeled synthetic benchmark corpora, and validates their
  diversity (`benchmark_validation/`). CPU-only; no GPU or model checkpoints
  required. See [`tsfm_benchmark/README.md`](tsfm_benchmark/README.md).
- **[`tsfm_model_analysis/tsfm_lens/`](tsfm_model_analysis/tsfm_lens/)**
  ("tsfm-lens") — a layered interpretability pipeline that runs two (or more)
  supported TSFM checkpoints through a shared analysis stack (behavioral
  comparison, cost/FLOPs, representational geometry, causal perturbation,
  attention analysis, SAEs, ...) and renders one self-contained HTML report.
  GPU-capable (real checkpoints); also runs CPU-only against mock models for a
  fast smoke check. See
  [`tsfm_model_analysis/tsfm_lens/README.md`](tsfm_model_analysis/tsfm_lens/README.md).

For the full architecture/design-doctrine reference, read `CLAUDE.md`. For
what's being worked on next, read `ROADMAP.md`.

## Install

This repo is **two separate installable packages, on purpose** — not an
in-progress merge. The root `pyproject.toml` packages only `tsfm_benchmark`
(+ `benchmark_validation` + `example_runs`); `tsfm_lens` has its own, separate
`pyproject.toml` one level down. The two packages' dependency sets barely
overlap and differ by an order of magnitude in weight: `tsfm_benchmark`'s core
dependencies are numpy/scipy/pyyaml; `tsfm_lens`'s core dependencies (torch,
zarr, plotly, scikit-learn) are far heavier, plus optional model-loading
libraries (`transformers`, `timesfm`, `chronos-forecasting`) for real
checkpoints. A single merged package would force every install — including
someone who only wants to generate benchmark data and never touches a GPU — to
pull the union of both. So: install whichever half you need, or both, into one
shared environment.

```bash
# from the repo root: tsfm_benchmark + benchmark_validation + example_runs
pip install -e .

# tsfm_lens (only needed for the model-analysis half; separate package)
pip install -e tsfm_model_analysis/tsfm_lens
```

Both packages can share a single conda/venv environment — see
[`DEPENDENCIES.md`](DEPENDENCIES.md) for exact, verified-working library
versions, full environment recreation steps, and a list of load-bearing
version pins with why each one matters (several exist only because a looser
bound broke in a documented way — see `CLAUDE.md` §11 for the full trap log).

## Quickstart

```bash
# --- Benchmark generation (fast, no GPU, no downloads) ---
cd tsfm_benchmark
PYTHONPATH=. python3 example_runs/run_full.py --config configs/example.yaml \
    --out ./benchmark_out --references monash --threshold 0.35 --epoch 0

# --- Benchmark validation ---
PYTHONPATH=. python3 example_runs/run_validation.py \
    --corpus ./benchmark_out/public_dev --out outputs

# --- tsfm-lens smoke run (mock models, no GPU, no downloads, ~minutes) ---
cd ../tsfm_model_analysis/tsfm_lens
python run.py --config configs/smoke.yaml
# open runs/smoke/report.html in a browser
```

`CLAUDE.md` §8 has the full CLI reference, including real-checkpoint runs,
stage selection/reruns, and adapter development. See
[`tsfm_benchmark/example_runs/WALKTHROUGH.md`](tsfm_benchmark/example_runs/WALKTHROUGH.md)
and
[`tsfm_benchmark/benchmark_validation/README.md`](tsfm_benchmark/benchmark_validation/README.md)
for a deeper walkthrough of the benchmark half.
