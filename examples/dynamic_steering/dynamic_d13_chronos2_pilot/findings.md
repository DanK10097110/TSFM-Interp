# Dynamic steering experiment findings

{
  "n": 5,
  "successes": 2,
  "mean_error": 0.2967107454852108,
  "replay_max_abs": 0.0,
  "seconds": 74.07909539691173,
  "model_calls": 387,
  "evaluated_rows": 2753
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | short | 0/1 | 0.6998 |
| trend (0.1) | long | 1/1 | 0.0066 |
| seasonality (0.1) | all | 0/1 | 0.5344 |
| context_ignore (0.5) | all | 1/1 | 0.0000 |
| uncertainty (0.2) | all | 0/1 | 0.2428 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
