# Dynamic steering experiment findings

{
  "n": 4,
  "successes": 3,
  "mean_error": 0.07176285417778765,
  "replay_max_abs": 0.0,
  "seconds": 20.721377835958265,
  "model_calls": 286,
  "evaluated_rows": 1642
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | short | 1/1 | 0.0025 |
| trend (0.1) | long | 1/1 | 0.0025 |
| seasonality (0.1) | all | 1/1 | 0.0025 |
| context_ignore (0.5) | all | 0/1 | 0.2794 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
