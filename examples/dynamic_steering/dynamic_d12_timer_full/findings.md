# Dynamic steering experiment findings

{
  "n": 168,
  "successes": 119,
  "mean_error": 0.13319918303628261,
  "replay_max_abs": 0.0,
  "seconds": 176.90615941898432,
  "model_calls": 3318,
  "evaluated_rows": 30590
}

| Request | Forecast window | Successes | Mean relative error |
|---|---|---:|---:|
| level (0.1) | all | 7/8 | 0.0902 |
| trend (0.1) | all | 8/8 | 0.0227 |
| detrend (0.1) | all | 8/8 | 0.0228 |
| amplitude (0.1) | all | 0/8 | 0.2204 |
| seasonality (0.1) | all | 7/8 | 0.0909 |
| deseasonalize (0.1) | all | 0/8 | 0.2215 |
| smooth (0.1) | all | 8/8 | 0.0227 |
| roughness (0.1) | all | 8/8 | 0.0227 |
| anchor (0.1) | all | 0/8 | 0.4384 |
| mean_revert (0.1) | all | 2/8 | 0.2573 |
| bounds (0.1) | all | 0/8 | 0.6484 |
| delay (0.1) | all | 0/8 | 0.6404 |
| context_ignore (0.5) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.5) | all | 8/8 | 0.0000 |
| context_recency (0.1) | all | 8/8 | 0.0000 |
| level (0.1) | short | 8/8 | 0.0227 |
| level (0.1) | long | 7/8 | 0.0759 |
| context_ignore (1) | all | 8/8 | 0.0000 |
| context_emphasize (0.5) | all | 8/8 | 0.0000 |
| context_focus (0.2) | all | 8/8 | 0.0000 |

Selection used separate validation contexts. Test seeds were not used for discovery.
Success means <=10% error relative to the requested change, including all protected coordinates.
Each forecast was replayed through the complete predictor; output arrays were never replaced by targets.
Input interventions, if selected, are explicitly identified in rows.json.
Failure is bounded to this search/rank/radius and does not establish impossibility.
