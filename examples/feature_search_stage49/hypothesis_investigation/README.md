# Hypothesis-driven TSFM investigation

Open [index.html](index.html) for the reviewed scientific snapshot. Its header
states whether the investigation is still active. [The session handoff](../../../tsfm_lens/docs/section49_investigation_20261116.md)
links the canonical findings and explains the distinctions needed to reuse them.

`records/` contains copied per-experiment knowledge files, including failed
attempts and preparation stages. `manifest.json` records their paths, statuses
and SHA256 hashes. `exact_tables.json` retains the unrounded figure/table data.
Raw arrays, native forecasts, source/checkpoint hashes and detailed arithmetic
audits remain under `tsfm_lens/runs/`; this report is a portable evidence summary,
not a complete raw-data or checkpoint bundle.

From the repository root, rebuild the snapshot with the existing run artifacts:

```bash
conda activate cudaPy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=tsfm_lens MPLCONFIGDIR=/tmp/section49_mpl \
python tsfm_lens/experiments/feature_search/investigation_report.py \
  --runs tsfm_lens/runs \
  --out examples/feature_search_stage49/hypothesis_investigation
```

The renderer performs no model inference or statistical selection. Use the
completed-status option only after reviewing the completed experiment records.
New experiments must use new run directories; do not overwrite the frozen runs.
The individual protocols and knowledge files document seeds, sampling units,
resource limits, native gates, failed attempts and exact reproduce pointers.

The investigation separates true-future forecasting, prediction of a model's
response, activation reconstruction, perturbation selectivity and causal effects.
Success at one does not establish the others. In particular, adaptive sparse
PCA coefficients are not a fixed low-rank representation, and a neuron that
classifies an input perturbation need not control the corresponding forecast
response. Unscorable interventions remain unscorable even when their apparent
effect is zero.
