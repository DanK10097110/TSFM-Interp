# Dynamic steering experiment findings

{
  "n": 42,
  "successes": 40,
  "mean_error": 0.042178577308566556,
  "replay_max_abs": 0.0,
  "seconds": 41.38337931595743,
  "model_calls": 636,
  "evaluated_rows": 3108
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | all | 2/2 | 0.0068 |
| trend (0.1) | all | 2/2 | 0.0068 |
| detrend (0.1) | all | 2/2 | 0.0068 |
| amplitude (0.1) | all | 2/2 | 0.0068 |
| seasonality (0.1) | all | 2/2 | 0.0068 |
| deseasonalize (0.1) | all | 2/2 | 0.0068 |
| smooth (0.1) | all | 2/2 | 0.0068 |
| roughness (0.1) | all | 2/2 | 0.0068 |
| anchor (0.1) | all | 2/2 | 0.0068 |
| mean_revert (0.1) | all | 2/2 | 0.0068 |
| bounds (0.1) | all | 0/2 | 0.7847 |
| delay (0.1) | all | 2/2 | 0.0190 |
| context_ignore (0.5) | all | 2/2 | 0.0000 |
| context_emphasize (0.5) | all | 2/2 | 0.0000 |
| context_focus (0.5) | all | 2/2 | 0.0000 |
| context_recency (0.1) | all | 2/2 | 0.0000 |
| level (0.1) | short | 2/2 | 0.0068 |
| level (0.1) | long | 2/2 | 0.0068 |
| context_ignore (1) | all | 2/2 | 0.0000 |
| context_emphasize (0.5) | all | 2/2 | 0.0000 |
| context_focus (0.2) | all | 2/2 | 0.0000 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
