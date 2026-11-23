# How much does the source checkpoint add over raw history?

Two estimators predict the change in a recipient’s hidden state. One sees the raw standardized time series, optionally giving more weight to its last 32 observations. The other sees the frozen source checkpoint’s states. Both select a linear or Gaussian kernel and ridge penalty using four-fold clean-state error on the same 384 M4 daily histories. Each resulting displacement is patched into the actual receiving model. The reference is the recipient’s independently measured native forecast response, with all comparisons in common forecast units. [Protocol and feature definitions](PROTOCOL.md)

| Input change | Population | TimesFM → Bolt, state / raw | Paired advantage [95% CI] | Bolt → TimesFM, state / raw | Paired advantage [95% CI] |
|---|---|---:|---|---:|---|
| Four fixed cues | M4 daily, 128 | .967 / .960 | .007 [.0003, .015] | .912 / .851 | .061 [.040, .085] |
| Four fixed cues | M4 weekly, 128 | .967 / .956 | .011 [.003, .020] | .925 / .855 | .071 [.050, .093] |
| Actual window shifts | M4 daily, 160 | .952 / .867 | .085 [.058, .117] | .898 / .654 | .244 [.189, .307] |
| Actual window shifts | M4 weekly, 70 | .897 / .825 | .071 [.016, .156] | .609 / .454 | .155 [.065, .274] |
| Actual window shifts | Bitbrains, 128 | .791 / .696 | .094 [.017, .187] | .519 / .076 | .443 [.269, .569] |

Recovery is one minus forecast-response SSE divided by native forecast-response energy. Item-bootstrap intervals resample both methods on the same series. The selected maps stayed frozen across all evaluation populations. The source checkpoint is a deterministic transformation of the input. A gain shows that this transformation helped with a fixed budget of paired examples and the specified estimators. [Fixed cue results into Bolt](results/bolt.json) · [Into TimesFM](results/timesfm.json) · [Rolling windows into Bolt](results/rolling_bolt.json) · [Into TimesFM](results/rolling_timesfm.json)

For inference, first create the paired clean and observed-window inputs and states with the transport scripts. Then run the two commands below for each recipient. The rolling command checks that all validation losses match the original fit. Compressed per-item responses accompany each summary. [Transport reproduction](../transport/README.md)

```bash
python -m research.activation_transfer.input_baseline.run bolt
python -m research.activation_transfer.input_baseline.rolling bolt
```
