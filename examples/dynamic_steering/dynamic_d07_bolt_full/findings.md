# Dynamic steering experiment findings

{
  "n": 224,
  "successes": 197,
  "mean_error": 0.07692024919620136,
  "replay_max_abs": 0.0,
  "seconds": 269.7923247440485,
  "model_calls": 3708,
  "evaluated_rows": 36116
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | all | 8/8 | 0.0286 |
| trend (0.1) | all | 8/8 | 0.0286 |
| detrend (0.1) | all | 8/8 | 0.0286 |
| amplitude (0.1) | all | 8/8 | 0.0288 |
| seasonality (0.1) | all | 8/8 | 0.0286 |
| deseasonalize (0.1) | all | 8/8 | 0.0286 |
| smooth (0.1) | all | 8/8 | 0.0310 |
| roughness (0.1) | all | 8/8 | 0.0310 |
| uncertainty (0.1) | all | 8/8 | 0.0286 |
| asymmetry (0.1) | all | 8/8 | 0.0286 |
| quantile (0.1) | all | 8/8 | 0.0286 |
| anchor (0.1) | all | 6/8 | 0.1057 |
| mean_revert (0.1) | all | 7/8 | 0.0576 |
| bounds (0.1) | all | 0/8 | 0.8134 |
| delay (0.1) | all | 0/8 | 0.4183 |
| context_ignore (0.5) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.5) | all | 8/8 | 0.0000 |
| context_recency (0.1) | all | 8/8 | 0.0000 |
| level (0.1) | short | 8/8 | 0.0478 |
| level (0.1) | long | 8/8 | 0.0476 |
| uncertainty (-0.2) | short | 5/8 | 0.0959 |
| uncertainty (0.2) | short | 5/8 | 0.0959 |
| uncertainty (-0.2) | long | 7/8 | 0.0758 |
| uncertainty (0.2) | long | 7/8 | 0.0758 |
| context_ignore (1) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.2) | all | 8/8 | 0.0000 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
