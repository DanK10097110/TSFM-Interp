# Dynamic steering experiment findings

{
  "n": 224,
  "successes": 212,
  "mean_error": 0.06211108203172534,
  "replay_max_abs": 0.0,
  "seconds": 716.9161348519847,
  "model_calls": 3580,
  "evaluated_rows": 34620
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | all | 8/8 | 0.0221 |
| trend (0.1) | all | 8/8 | 0.0516 |
| detrend (0.1) | all | 8/8 | 0.0503 |
| amplitude (0.1) | all | 8/8 | 0.0454 |
| seasonality (0.1) | all | 8/8 | 0.0441 |
| deseasonalize (0.1) | all | 8/8 | 0.0453 |
| smooth (0.1) | all | 8/8 | 0.0503 |
| roughness (0.1) | all | 8/8 | 0.0503 |
| uncertainty (0.1) | all | 8/8 | 0.0503 |
| asymmetry (0.1) | all | 8/8 | 0.0503 |
| quantile (0.1) | all | 8/8 | 0.0503 |
| anchor (0.1) | all | 5/8 | 0.1561 |
| mean_revert (0.1) | all | 7/8 | 0.0711 |
| bounds (0.1) | all | 0/8 | 0.6323 |
| delay (0.1) | all | 8/8 | 0.0526 |
| context_ignore (0.5) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.5) | all | 8/8 | 0.0000 |
| context_recency (0.1) | all | 8/8 | 0.0000 |
| level (0.1) | short | 8/8 | 0.0441 |
| level (0.1) | long | 8/8 | 0.0441 |
| uncertainty (-0.2) | short | 8/8 | 0.0536 |
| uncertainty (0.2) | short | 8/8 | 0.0536 |
| uncertainty (-0.2) | long | 8/8 | 0.0607 |
| uncertainty (0.2) | long | 8/8 | 0.0607 |
| context_ignore (1) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.2) | all | 8/8 | 0.0000 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
