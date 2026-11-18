# Selective residual trees and internal composition

From the repository root, use the cudaPy environment, PYTHONPATH=tsfm_lens,
CUDA_VISIBLE_DEVICES empty, and OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1.
Model inference is offline using the existing cached checkpoints and real sources.

Rescore existing artifacts without redoing the main model runs:
```
python -m experiments.feature_search.selective_analysis --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_audit --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_diagnostics --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_transfer_analysis --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_strong_audit --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_report --run tsfm_lens/runs/selective_replication4942000 --out examples/feature_search_stage49/selective_composition
```

To reproduce, use NEW run directories under tsfm_lens/runs. The replication
preparer reads frozen geometry weights from selective_pilot4941000. Drivers
refuse to overwrite completed native runs.
```
python -m experiments.feature_search.selective_study prepare --out NEW_PILOT
python -m experiments.feature_search.selective_study run --out NEW_PILOT
python -m experiments.feature_search.selective_analysis --out NEW_PILOT
python -m experiments.feature_search.selective_followup prepare --out NEW_REPLICATION
python -m experiments.feature_search.selective_parallel --out NEW_REPLICATION
python -m experiments.feature_search.selective_tree_stability --out NEW_REPLICATION
python -m experiments.feature_search.selective_followup dominant --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_followup dominant --out NEW_REPLICATION --model timesfm
python -m experiments.feature_search.selective_transfer --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_transfer --out NEW_REPLICATION --model timesfm
python -m experiments.feature_search.selective_frequency_transfer --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_frequency_transfer --out NEW_REPLICATION --model timesfm
python -m experiments.feature_search.selective_strong_signal worker --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_strong_signal worker --out NEW_REPLICATION --model timesfm
```

All thresholds, splits and intervention families are in protocol.json and
ROADMAP49.35 onward. No actual future enters the native workers, extraction,
hidden-weight calibration or transfer tests. Forecast-policy selection uses FIT
futures only. TEST sources are not used for hyperparameter selection. Held-out
comparison remains exploratory rather than sealed-private confirmation.
