# Dynamic steering experiment findings

{
  "n": 5,
  "successes": 0,
  "mean_error": 0.6122646281569355,
  "replay_max_abs": 0.0,
  "seconds": 74.52047114993911,
  "model_calls": 345,
  "evaluated_rows": 3599
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | short | 0/1 | 1.0000 |
| trend (0.1) | long | 0/1 | 0.6562 |
| seasonality (0.1) | all | 0/1 | 0.2322 |
| context_ignore (0.5) | all | 0/1 | 0.1730 |
| uncertainty (0.2) | all | 0/1 | 1.0000 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
