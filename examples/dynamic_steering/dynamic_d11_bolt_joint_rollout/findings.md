# Dynamic steering experiment findings

{
  "n": 10,
  "successes": 10,
  "mean_error": 0.05169268777945625,
  "replay_max_abs": 0.0,
  "seconds": 395.46888163499534,
  "model_calls": 651,
  "evaluated_rows": 6871
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | short | 2/2 | 0.0703 |
| trend (0.1) | long | 2/2 | 0.0515 |
| seasonality (0.1) | all | 2/2 | 0.0718 |
| context_ignore (0.5) | all | 2/2 | 0.0000 |
| uncertainty (0.2) | all | 2/2 | 0.0650 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
