# Reproduce the cross-model experiments

The quantities in this directory are computed from actual frozen-checkpoint forecast passes on CPU. The small NPZ and JSON records contain series IDs, directions or interventions, and per-item traces sufficient to check the reported metrics without downloading a model. The original study used Python 3.12.13, NumPy 2.1.0, SciPy 1.18.0, Torch 2.12.0, PyArrow 24.0.0, Transformers 4.57.6, chronos-forecasting 2.3.1 and TimesFM 2.0.2. Repository baseline `afce6a5` was used with the workspace’s activation-contrast steering implementation. Replay that uses the amplification API requires `tsfm_lens.steering.ActivationContrast`. [Local API](../../tsfm_lens/tsfm_lens/steering/ACTIVATION_CONTRAST.md)

## Scoring existing evidence

From the repository root, in the project’s forecasting environment:

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m pytest -q research/activation_transfer
python -m research.activation_transfer.audit
python -m research.activation_transfer.transport.audit
python -m research.activation_transfer.atoms.uncertainty
python -m research.activation_transfer.figure
```

The read-only study audit reconstructs 572 atom-transfer metrics, 168 sparse-code metrics and 504 counts of feature switches. It checks the structural zeros, held-out cohorts and calibration overlap from the shipped arrays using NumPy. The figure reads the reported primary metrics. The transport audit independently recomputes 394 quantities from the archived per-series forecasts. For an independent reanalysis of the other branches use their score programs and the included arrays. For example, `python -m research.activation_transfer.atoms.scores timesfm` scores the TimesFM **source** atom study. The [verification record](verification.json) describes the fresh-cache replays.

## Regenerating forecasts and dictionaries

The locally cached public snapshots were `amazon/chronos-bolt-small@772f3d25d38aec6d914c8949dab4462e2d46f5d8`, `google/timesfm-2.5-200m-pytorch@1d952420fba87f3c6dee4f240de0f1a0fbc790e3`, `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c` and `thuml/timer-base-84m@70077a71acce1b4c00d98332fcaabc694255d8e5`. Experiments use selected data shards from `Salesforce/GiftEval@30841734ac5cfddbd0c3bad6d09d2b6b32becbb0` and the `monash_electricity_hourly` configuration of the Chronos training datasets. Dataset hashes and exact item selections are in each branch’s manifest. Source dataset exposure during model pretraining is possible. [Bolt](https://huggingface.co/amazon/chronos-bolt-small) · [TimesFM](https://huggingface.co/google/timesfm-2.5-200m-pytorch) · [Chronos-2](https://huggingface.co/amazon/chronos-2) · [Timer](https://huggingface.co/thuml/timer-base-84m) · [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) · [Chronos training corpora](https://huggingface.co/datasets/autogluon/chronos_datasets)

Once those snapshots are cached, use `HF_HUB_OFFLINE=1`. Timer’s checkpoint registers modeling code in a writable Transformers module cache. In the shared workspace we used:

```bash
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE=/tmp/tsfm_requests_hf_modules
export MPLCONFIGDIR=/tmp/tsfm-transfer-matplotlib
```

The checkpoint-specific SAE branches additionally need the repository run `tsfm_lens/runs/panel7_v3_ablcheck`. In this workspace its `sae/` directory resolves to the saved historical weights. Fresh SAE fitting uses `activations.zarr`, `meta.parquet` and `config_resolved.yaml` from that run. Git alone does not distribute these sizeable model artifacts. The archived SAE’s 3,990 extra online Monash windows are unavailable for an exact historical retrain. The two new initializations both use the 34,515 recoverable public-development rows. The boundary archive indexes the retained local checkpoints, intermediate arrays and SHA-256 fingerprints. [Boundary archive manifest](boundary/artifacts.json)

| Research question | Driver and starting point |
|---|---|
| Does the TopK threshold distort a small intervention? Do two dictionaries on the same model agree? | [Boundary and seed study](boundary/README.md) |
| Does the clean receiver state help recover which of its own sparse features change? | [Re-encode dense displacements, then replay the decoded edits](reencode/README.md) |
| Can clean-state alignment carry an observed window change? What do shuffled, randomized and raw-input controls recover? | [Transport](transport/README.md) then [flexible raw-input comparison](input_baseline/README.md) |
| Do output patterns agree even without accessing hidden states? | [Behavioral controls](controls/README.md) |
| What happens to the forecast when a naturally firing source atom is removed? | [Atom study](atoms/README.md), [case- and dose-matched real window comparison](atoms/matched_windows/README.md), and [dictionary controls](controls/DICTIONARY_RESULTS.md) |
| Do the atom findings survive fresh initializations, new cohorts, or a level-only comparator? | [Replication, scalar calibration and output derivatives](replication/README.md) |
| Which depth or type of steering direction transfers? | [Fixed cues](steering/README.md), [state-dependent amplification](steering/amplify/README.md), [three-model output requests](forecast_requests/README.md), [32-lead replay](forecast_requests/horizon/README.md), [recipient-unit calibration](forecast_requests/units/README.md) |

Model-free reanalysis can be done from any new clone with the retained NPZ/JSON records. For heavy reruns, use a new scratch directory to recapture rather than overwriting earlier model intermediates. The initial atom study provides a single fresh-cache command:

```bash
bash research/activation_transfer/atoms/reproduce.sh /tmp/tsfm-atoms-fresh /tmp/tsfm-atoms-fresh-results
```

Scratch directories in a command are used by follow-up analyses that depend on that command’s data. Check the default cache or pass the matching `--cache` argument when reusing a nondefault directory.

If the primary atom run uses a custom scratch location, set `TSFM_ATOM_CACHE` to that directory for the replication drivers. Set `TSFM_ELECTRICITY_ARROW` to the Monash electricity Arrow stream if the local datasets cache uses another path. The source file hash recorded in the atom manifest identifies the exact input used here.


## What the replays reproduced

- Each of the four new SAE dictionaries was independently retrained a second time on the original 34,515 recoverable public-development activations for its checkpoint. All four 720-step runs reproduced every saved tensor, recorded loss and checkpoint SHA-256 exactly. This same-environment check covers the two new initializations for each model. [Training replay hashes and tensor comparisons](verification.json)
- The complete two-model atom runs used fresh captures and reproduced every stored forecast component and both score files exactly. The independent matched-window replay also reproduced both point estimates and confidence endpoints exactly. [Atom replay record](verification.json) · [Matched-window replay](atoms/matched_windows/results/replay.json)
- The independent reconstruction of the receiver’s sparse edits, starting from newly captured states, reproduced every entering and leaving ID set. Maximum weekly forecast component drift was 0.00000239 history SD. [Re-encoding replay](verification.json)
- For fixed-cue steering, the 12 archived response arrays reproduced within 0.00000346 history SD. The full **output-request refit** was more numerically sensitive. Some component comparisons exceeded the initial 0.00001 absolute/relative criterion. The largest difference in any calibrated real-map cell’s mean recovery was 0.0000154, in a calibrated sham cell 0.000317, and in an uncalibrated cell 0.0585 (a severely oversized Bolt → TimesFM lift scored −418.379 versus −418.437). The pooled daily lift changed from .734224 to .734223. One replay process ended early during the sham passes. We resumed from its independently generated fits and verified all six model/control files. Reusing the original fitted directions in the extended-horizon experiment reproduced the first eight forecast components within 0.00000334 history SD. [Actual component discrepancies, doses and scores](verification.json) · [Frozen-direction horizon replay](forecast_requests/horizon/README.md)
- All 70 study tests passed, along with 72 existing steering/SAE regression tests and five end-to-end smoke tests. Every new study test was exercised against a planted mathematical or implementation error. The small model-free scoring and figure commands were rerun from a temporary copy containing only Git-visible study materials. [Mutation report](MUTATION_CHECK.md)
