# Dynamic steering experiment findings

{
  "n": 5,
  "successes": 4,
  "mean_error": 0.1308220004894835,
  "replay_max_abs": 0.0,
  "seconds": 51.22496699797921,
  "model_calls": 296,
  "evaluated_rows": 1712
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | short | 1/1 | 0.0511 |
| trend (0.1) | long | 1/1 | 0.0511 |
| seasonality (0.1) | all | 1/1 | 0.0511 |
| context_ignore (0.5) | all | 0/1 | 0.4497 |
| uncertainty (0.2) | all | 1/1 | 0.0511 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
