# Cross-model intervention on aligned temporal support

The experiment fits a ridge map between the middle-block inputs of two pretrained forecasters on ordinary daily M4 series. On held-out histories, it transports a source hidden-state contrast into the target model, applies both signs to the target model's hidden state during an ordinary forward pass, and compares the resulting forecast response with the effect of the target's own hidden contrast. Read [the findings and figure](RESULTS.md) for the outcomes and limitations, and [the protocol](PROTOCOL.md) for endpoint definitions and the role of each control.

## Reproduce

From the repository root, with its forecasting dependencies installed and the pinned public checkpoints and GIFT data cached, run the following commands. The original run used Python 3.12.13, NumPy 2.1.0, SciPy 1.18.0, PyTorch 2.12.0, PyArrow 24.0.0, Transformers 4.57.6, Chronos Forecasting 2.3.1 and TimesFM 2.0.2. The extraction and intervention scripts use CPU float32. Dataset and checkpoint revision hashes are pinned in `prepare.py` and `capture.py`. The scratch cache is shared across both directions. The summary audit can run directly against the included arrays without either forecasting model.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
study=research/activation_transfer/transport
python "$study/prepare.py"
python "$study/capture.py" bolt
python "$study/capture.py" timesfm
python "$study/analyze.py" bolt
python "$study/analyze.py" timesfm
python -m pytest -q "$study/test_transport.py"
```

The compact forecast archives in `results/` contain one row per held-out item, with item ID, split, concept names, and the signed 16-lead median forecast responses under each comparator. The JSONs report CV curves, checkpoint and patch identities, hidden-space recovery, series-bootstrap intervals on forecast metrics and pooled responses to each concept. The manifest identifies every training and evaluation item and the Arrow file hashes. The scratch `data.npz` holds standardized histories, known perturbations and IDs. Each scratch model archive holds `clean[n,width]`, `delta[n,4,width]`, `native[n,4,16]`, `input_effect[n,4,16]` and baseline `forecast[n,16]`. TimesFM stores one 1280-dimensional token. Bolt stores two consecutive 512-dimensional vectors in chronological order.

For analysis of the shipped forecast arrays alone, run `python research/activation_transfer/transport/audit.py` from the repository root. This uses NumPy and the Python standard library. It reconstructs 394 reported scalar endpoints and verifies the identity splits, the assignments of swapped histories, and the integrity of the primary response archives.

Replaying selection from the pinned Arrow data reproduced the three manifests byte for byte and every normalized history exactly. Replaying the live checkpoint interventions for the sine contrast on one held-out daily and one weekly history reproduced the saved central forecast effects within 4 × 10⁻⁷ in both architectures. The included unit suite covers the regression, splitting, bootstrap statistics, token hooks, rank and randomization controls, and the definition of the source directions.

## Follow-ups

The follow-ups preserve the original clean training series and middle-block sites. Run either model direction separately.

```bash
python "$study/context_controls.py" bolt
python "$study/context_controls.py" timesfm
python "$study/natural.py" prepare
python "$study/natural.py" capture bolt
python "$study/natural.py" capture timesfm
python "$study/natural.py" analyze bolt
python "$study/natural.py" analyze timesfm
python "$study/rank.py" bolt
python "$study/rank.py" timesfm
python "$study/random_source.py" bolt
python "$study/random_source.py" timesfm
python "$study/rewire.py" bolt
python "$study/rewire.py" timesfm
python "$study/level_control.py"
python "$study/source_axes.py" prepare
python "$study/source_axes.py" native bolt
python "$study/source_axes.py" native timesfm
python "$study/source_axes.py" evaluate bolt
python "$study/source_axes.py" evaluate timesfm
python "$study/audit.py"
python "$study/plot.py"
```

`analyze.py`, `context_controls.py`, `natural.py analyze` and `rank.py` take the name of the **target** receiving the patch. `random_source.py` and `rewire.py` take the **source** whose weights are changed. `source_axes.py native/evaluate` take the source used to define a fixed intervention direction. The random-source outputs identify both checkpoints in their filenames.

See the [background and observed-window protocol](FOLLOWUP.md), [rank protocol](RANK_PROTOCOL.md), and [random-source protocol](RANDOM_SOURCE_PROTOCOL.md). Fresh natural-window and source-axis selections read the shipped neighboring `controls/` and `atoms/results/` manifests to exclude overlapping series. Keep those manifests in place when regenerating the cohorts. The rank runner stores individual completed interventions in the scratch cache keyed by the actual input, edit and item IDs. The source-null comparison stores random-source states there as well. The level-control script is an exploratory offline diagnostic: it combines the dense bridge's horizon shape with either the train-only mean target response for that waveform or the target-patched raw-history ridge baseline's forecast level.

The [fixed source direction protocol](SOURCE_AXES_PROTOCOL.md) compares the original source intervention forecast response with the receiving model’s response after transport. It includes new daily and Shenzhen taxi cohorts. The compact source-axis result files contain all four principal directions and four named contrasts, the corresponding source and destination forecast changes, and responses to matched-norm wrong directions.
