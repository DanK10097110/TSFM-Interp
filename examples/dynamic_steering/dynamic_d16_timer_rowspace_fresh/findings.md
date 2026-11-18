# Dynamic steering experiment findings

{
  "n": 168,
  "successes": 159,
  "mean_error": 0.039731060861487325,
  "replay_max_abs": 0.0,
  "seconds": 65.38832406396978,
  "model_calls": 1330,
  "evaluated_rows": 6546
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | all | 8/8 | 0.0176 |
| trend (0.1) | all | 8/8 | 0.0176 |
| detrend (0.1) | all | 8/8 | 0.0176 |
| amplitude (0.1) | all | 8/8 | 0.0176 |
| seasonality (0.1) | all | 8/8 | 0.0176 |
| deseasonalize (0.1) | all | 8/8 | 0.0176 |
| smooth (0.1) | all | 8/8 | 0.0176 |
| roughness (0.1) | all | 8/8 | 0.0176 |
| anchor (0.1) | all | 7/8 | 0.0397 |
| mean_revert (0.1) | all | 8/8 | 0.0176 |
| bounds (0.1) | all | 0/8 | 0.5789 |
| delay (0.1) | all | 8/8 | 0.0223 |
| context_ignore (0.5) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.5) | all | 8/8 | 0.0000 |
| context_recency (0.1) | all | 8/8 | 0.0000 |
| level (0.1) | short | 8/8 | 0.0176 |
| level (0.1) | long | 8/8 | 0.0176 |
| context_ignore (1) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.2) | all | 8/8 | 0.0000 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
