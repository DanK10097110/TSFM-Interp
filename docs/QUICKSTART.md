# Quickstart: a first real run in about an hour

This takes you from a fresh clone to an interactive comparison report of two
small time-series foundation models (Chronos-Bolt-small and Chronos-2) on a
small synthetic corpus. It needs one CUDA GPU (about 8 GB is plenty), no
private data and no network beyond the pip and Hugging Face downloads. A
Colab version is in [`examples/quickstart.ipynb`](../examples/quickstart.ipynb).

## 1. Install

```bash
git clone https://github.com/DanK10097110/TSFM-Interp.git
cd TSFM-Interp
pip install -e .                 # tsfm_benchmark (corpus builder)
pip install -e tsfm_lens         # tsfm_lens (the analysis pipeline)
pip install chronos-forecasting  # the two checkpoints' adapter dependency
pip install umap-learn           # optional; the quickstart does not need it
```

Exact tested library pins live in [`DEPENDENCIES.md`](../DEPENDENCIES.md).
`zarr` must stay below 3 (a v3 store reads back empty with no error), which
the install above already respects.

## 2. Build the small corpus (about a minute, CPU)

From the repository root:

```bash
PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_full.py \
    --config tsfm_benchmark/configs/quickstart.yaml \
    --out ./benchmark_quickstart --references none
```

This writes 120 synthetic series to `benchmark_quickstart/public_dev` (and a
second, unused split to `private_test`). `--references none` skips the
Monash leakage gate, which is a no-op for purely synthetic data and would need
a download; the build warns about that and records it in the manifest.

## 3. Check, then run (about 25 minutes on an RTX A5000)

```bash
cd tsfm_lens
python run.py --config configs/quickstart.yaml --doctor   # preflight, ~30 s
python run.py --config configs/quickstart.yaml            # the run
```

Pick a free GPU with `nvidia-smi` and `CUDA_VISIBLE_DEVICES=N` on a shared
machine. Stages that finished are skipped on a rerun, so an interrupted run
resumes where it stopped.

## 4. Open the report

```
tsfm_lens/runs/quickstart/report.html
```

It is one self-contained HTML file. Open it in a browser.

## How to read it

Start at the scorecard at the top. Every row names the rule it was derived
from. Each claim carries an evidence class, and the classes are not
interchangeable:

| Class | Meaning | Can it say "causal"? |
|---|---|---|
| behavioral (L0) | Forecast error per family, paired bootstrap over series | No, it is about outputs |
| geometric (L1) | CKA/RSA: representations have similar shape | No, correlational |
| linear-translatable (L2) | One model's layer predicts another's, gain over an input-feature baseline | No, not causal |
| causal within-model (L3, concepts) | Corruption and activation patching, SAE feature ablation against a random-direction null | Yes, inside one model |
| held-out (confirm) | Hypotheses registered on dev data, tested once on a sealed private split | Not part of the quickstart |

## What the quickstart is not

- No `confirm` stage. Held-out confirmation needs a sealed private corpus that
  has never been looked at; the quickstart has none, so every number is
  exploratory.
- 120 series gives wide confidence intervals. Expect many "not significant"
  verdicts and few concepts.
- Synthetic data only. The full presets (`configs/concept_atlas.yaml`,
  `configs/full_report_run_4model.yaml`) use a ~1000-series corpus, four
  models and held-out confirmation. Every reduced knob in
  `tsfm_lens/configs/quickstart.yaml` states the full value in a comment.

## Where next

- Swap a model: edit `models:` in a copy of the config; see
  [`tsfm_lens/docs/ADDING_A_MODEL.md`](../tsfm_lens/docs/ADDING_A_MODEL.md).
- Read a report section by section:
  [`tsfm_lens/docs/worked_example.md`](../tsfm_lens/docs/worked_example.md).
- Re-run only some stages: `--stages l1,report`, `--force l1`.
