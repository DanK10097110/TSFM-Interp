# Reproduce the follow-up experiments

The inherited repository environment, pinned model revisions, SAE weights and public datasets are described in the parent study. Commands below run from the repository root. The primary results are saved as ordinary NPZ/JSON files next to each experiment, with IDs and per-history interventions. [Environment and material preparation](../REPRODUCE.md)

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
pkg=research.activation_transfer.shape_transfer
for model in bolt timesfm timer analyze; do python -m "$pkg.input_paths.run" "$model"; done
for count in 1 2; do python -m "$pkg.token_position.run" "$count"; done
python -m "$pkg.token_position.analyze"
for stage in fit apply; do
  for model in bolt timesfm; do python -m "$pkg.local_requests.run" "$stage" "$model"; done
done
python -m "$pkg.local_requests.analyze"
for model in bolt timesfm; do python -m "$pkg.full_space.jacobians" "$model"; done
for stage in transfer leads dose access; do
  for model in bolt timesfm; do python -m "$pkg.full_space.$stage" "$model"; done
done
python -m "$pkg.full_space.summarize"
python -m "$pkg.full_space.score_followups"
for model in bolt timesfm; do
  python -m "$pkg.sparse_writes.run" "$model"
  python -m "$pkg.sparse_writes.run" "$model" --extra
done
python -m "$pkg.sparse_writes.summarize"
python -m "$pkg.sparse_writes.reasons"
python -m "$pkg.figure"
python -m pytest -q research/activation_transfer tsfm_lens/tests/test_smoke.py tsfm_lens/tests/test_contrast_steering.py
python -m "$pkg.check_plants"
```

The input, position and named-request studies reuse the original request-study splits and fits in `/tmp/tsfm_forecast_requests_20261120/`. The physical SAE experiments use `/tmp/tsfm_atom_replication_20261120/`, which contains the clean context windows, token positions and clean post-block token groups. The full local Jacobian caches are intermediate outputs. Regenerate them using the commands above when scratch data are absent. The parent study gives the original material preparation sequence. In the follow-up, both checkpoint Jacobians cover all 96 calibration and 320 subsequent histories; fixed supervised operators use only the calibration rows. Neither observational futures nor held-out ablation outcomes enter a translation.

For separate real reruns, `input_paths.run`, `token_position.run`, `local_requests.run` and `sparse_writes.run` accept `--out`; `local_requests.run` can also refit its derivatives under `--scratch`. The full-space independent audit uses fresh singleton gradients, a separate solver and direct production-model hooks on a subset, and scores the entire archive independently. The derivative-map, nuisance-level, tangent, fresh-cohort and third-receiver experiments document their own checks. [Real archive comparison](verification.py) · [Independent gradient and forecast audit](full_space/audit.py) · [Known-answer test mutations](plant_results.json) · [Generic response map](response/README.md) · [Level geometry](level_geometry/README.md) · [Input perturbation maps](tangents/README.md) · [Third receiver](third_receiver/README.md) · [Prospective sample](fresh_cohort/README.md)
