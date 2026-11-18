# Fourier decomposition diagnostic: bounded proofs of concept

[Open the report](index.html). The adapter estimates spectral components from
history alone, forecasts their groups separately, and measures how recombination
changes fidelity. Synthetic ground-truth parameters never enter the adapter.

The run uses 16 new synthetic seeds and 24 new cached real source series,
with discovery/check halves, two frozen checkpoints, context128/horizon32,
and trees with1/2/4/8 groups. There are456 dependent condition rows, not456
independent samples. Predictions for identical tree nodes are reused:2948 model
inputs per checkpoint. One CPU thread, no GPU or downloads, no fine-tuning or SAE
training. Native worker times: Bolt55.78287084196927s, TimesFM175.95231239998247s.
Each worker stayed within its600s/3GiB cap. No expansion was launched.

## What looks worth following up

- **Distractor sensitivity:** TimesFM's known-signal response degrades as a
  distractor grows. At distractor bin17/amplitude8, check response NMSE changes
  from1.8385632140476604 native to0.004979990196191095 with K4. Bolt changes
  from1.279860610934934 to0.7478962472416841. The nearby bin6 distractor shares
  a group with the target and is not fixed by this split.
- **Response fidelity is not forecasting quality:** on electricity and pedestrian
  sources, TimesFM's K4 response improvement has the same positive mean sign in
  discovery and check, while total-future forecast error increases. M4 response
  improvement appears in check but not discovery. Bolt does not share the same
  pattern. Controlled-response consistency alone is an inadequate model ranking.
- **There is no universal helpful split count:** total forecasting often worsens
  with more groups, particularly off-grid/chirp and real inputs. Tree shape,
  detrending and phase affect the outcome. First-crossing counts are descriptive
  and depend on the declared threshold and partition.

These are exploratory candidate findings. Only8 synthetic check seeds and4 real
check sources per dataset support each primary cell. Intervals are source-level,
unadjusted, conditional descriptive bootstraps, not multiple-testing-corrected
claims. Existing real data have unknown pretraining overlap.

## What the controls do and do not establish

All input energy, DC and Nyquist are retained. Mean/linear trend is counted once.
K1 is exactly native. Both raw forecast sums and sums subtracting duplicate F(0)
bias are recorded. An affine forecaster closes exactly under the corrected sum.
Every tree reconstructs the same input, and per-node forecast changes telescope
exactly to the next frontier. Component RMS and active-group counts are retained.

For the distant on-grid distractor, the target-group input is identical across
amplitudes. The adapter's amplitude-invariance is therefore structural, not an
independent discovery; its achieved response accuracy is measured. Native
normalization and internal interference are not causally separated here.

The random grouping energy gate passes, but some sparse synthetic random trees
only move zero-energy leaves. `random_grouping_nontrivial` and effective input
change distinguish this degeneracy from a useful control. Equal nominal K is not
necessarily equal active-component count. Crest-matched phase effects are weaker
than coherent-phase effects, and the latter also change peakiness.

The frozen classical baseline fits three history-selected Fourier bins plus a
trend. It can miss the weak target under the strongest nearby distractor. A
**post-hoc model-free** periodic128 continuation control is retained separately;
it reproduces on-grid additions by construction. This neither changes any native
result nor establishes strong real-data forecasting. No future data is used to
choose frequencies or decompositions.

## Reproduction

From the repository root, using the project Python environment:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=tsfm_lens
python -m experiments.feature_search.fourier_poc prepare \
  --out tsfm_lens/runs/fourier_NEW
python -m experiments.feature_search.fourier_poc run \
  --out tsfm_lens/runs/fourier_NEW
MPLCONFIGDIR=/tmp/tsfm-matplotlib python -m experiments.feature_search.fourier_report \
  --run tsfm_lens/runs/fourier_poc4929000 \
  --out examples/feature_search_stage49/fourier_poc
python -m experiments.feature_search.fourier_audit \
  --run tsfm_lens/runs/fourier_poc4929000 \
  --report examples/feature_search_stage49/fourier_poc
```

Preparation requires cached Arrow sources and safetensors. `--cache` changes the
cache root. Existing outputs cannot be overwritten by inference. Source,
checkpoint and corpus bindings are frozen before native work. The saved bank
contains exactly the model inputs; workers do not load future arrays.

Core implementation: `fourier_adapter.py`; corpus/controller: `fourier_poc.py`;
model-free analysis/report: `fourier_report.py`; independent NumPy audit:
`fourier_audit.py`, all under `tsfm_lens/experiments/feature_search/`.

Validation:18 focused/end-to-end tests passed, seven planted regressions detected.
The existing smoke test emits PytestReturnNotNoneWarning. Independent replay
checks3456 raw/corrected losses and576 response scores per model, as well as
node replacement gains, source units, first crossings and aggregate means.
It does not certify bootstrap coverage or absence of pretraining overlap.

All input nodes, predictions, source metadata, plans and protocols remain in
`tsfm_lens/runs/fourier_poc4929000/`. The portable report contains lightweight
measurements and figures. ROADMAP §49.29–49.30; FINDINGS PM-25/PM-26.

The final effective-input audit finds nontrivial random grouping changes in
24/24 real contexts and0/16 stationary synthetic contexts. The latter control
is degenerate and cannot support a grouping-specific interpretation. An isolated
false-loss artifact is rejected by the independent audit; see
`audit_regression_check.json`.
