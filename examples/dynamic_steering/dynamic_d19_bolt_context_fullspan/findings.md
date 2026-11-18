# Dynamic steering experiment findings

{
  "n": 28,
  "successes": 12,
  "mean_error": 0.2781368692335116,
  "replay_max_abs": 0.0,
  "seconds": 238.55399114696775,
  "model_calls": 3089,
  "evaluated_rows": 40841
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| context_ignore (0.1) | long | 4/4 | 0.0535 |
| context_ignore (0.3) | long | 0/4 | 0.2280 |
| context_ignore (1) | long | 0/4 | 0.7126 |
| context_emphasize (0.1) | short | 4/4 | 0.0459 |
| context_emphasize (0.5) | short | 1/4 | 0.1944 |
| context_focus (0.2) | long | 3/4 | 0.0795 |
| context_focus (0.2) | long | 0/4 | 0.6329 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
