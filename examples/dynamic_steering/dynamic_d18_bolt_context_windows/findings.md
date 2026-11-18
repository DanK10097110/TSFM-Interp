# Dynamic steering experiment findings

{
  "n": 28,
  "successes": 5,
  "mean_error": 0.3954840736085302,
  "replay_max_abs": 0.0,
  "seconds": 113.54066257900558,
  "model_calls": 1571,
  "evaluated_rows": 15291
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| context_ignore (0.1) | long | 2/4 | 0.1059 |
| context_ignore (0.3) | long | 0/4 | 0.5418 |
| context_ignore (1) | long | 0/4 | 0.8465 |
| context_emphasize (0.1) | short | 3/4 | 0.0914 |
| context_emphasize (0.5) | short | 0/4 | 0.3603 |
| context_focus (0.2) | long | 0/4 | 0.1852 |
| context_focus (0.2) | long | 0/4 | 0.6373 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
