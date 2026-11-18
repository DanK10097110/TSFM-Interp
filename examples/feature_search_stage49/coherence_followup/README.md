# Section49 coherence follow-up

See index.html for figures, full comparison tables and limits. ROADMAP49.31–49.34 records prospective designs and findings.

From the repository root, use the cudaPy environment with PYTHONPATH=tsfm_lens,
CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1,
HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1. All cached weights and data are local.

Existing results can be rescored without inference:
```
python -m experiments.feature_search.coherence_analysis --out tsfm_lens/runs/coherence_replication4932000
python -m experiments.feature_search.coherence_audit --out tsfm_lens/runs/coherence_replication4932000
python -m experiments.feature_search.coherence_audit --training --out tsfm_lens/runs/coverage_replication493900
python -m experiments.feature_search.coherence_report --run tsfm_lens/runs/coherence_replication4932000 --training tsfm_lens/runs/coverage_replication493900 --pilot tsfm_lens/runs/coherence_pilot4931000 --normalization tsfm_lens/runs/coverage_normalization493900 --out examples/feature_search_stage49/coherence_followup
```

To reproduce inference, use a NEW output directory; drivers refuse overwrites:
```
python -m experiments.feature_search.coherence_study prepare --stage replication --out NEW_RUN
python -m experiments.feature_search.coherence_study run --out NEW_RUN
python -m experiments.feature_search.coherence_classical --out NEW_RUN
python -m experiments.feature_search.coherence_analysis --out NEW_RUN
python -m experiments.feature_search.coverage_replication --out NEW_TRAINING_RUN
python -m experiments.feature_search.coverage_normalization_control --out NEW_NORMALIZATION_RUN
```
The synthetic generator labels are used only in scoring. Fourier decomposition,
classical frequency fitting, native workers and model-selection test choices
have no access to target futures. Fixed selectors use FIT future labels only.

Data provenance: cached Chronos electricity, M4 hourly, pedestrian and traffic
series. Fresh source membership is audited; pretraining overlap remains unknown.
Figures are standalone PNGs. Complete precision is retained in JSON artifacts.
