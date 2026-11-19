# Audit the steering results

Open [the standalone report](report.html) locally to inspect the two checkpoints, all 19 single directions, and the level + trend and level + uncertainty requests. The report includes the ordinary, requested and steered forecasts, typical and worst examples, per-request confidence intervals, and an exportable predictor for each comparison. The [steering README](../../../tsfm_lens/tsfm_lens/steering/README.md) explains the mechanism, results and limits.

| Evidence | Design |
| --- | --- |
| [Matched seed](matched.json.gz) | Rank 16, four calibration and four validation histories, 24 evaluation histories, seed 731, both models, all directions and two mixtures |
| [Replication report](report.html) | Same protocol, seed 20261120 |
| [Bolt doses](bolt_doses.json.gz) and [TimesFM doses](timesfm_doses.json.gz) | Rank 16, three amounts for each of 19 directions on eight matched histories per checkpoint |
| [Mixed pilot](mixed.json) | Rank 64, two pairs and their three individual controls, eight histories per checkpoint |
| [M4 daily transfer](m4_daily.json) | Matched-seed mixed exports applied directly to the last forecast window of 24 distinct real series |
| [Radius sensitivity](bounds_radius.json) | Worst bounds miss per checkpoint on the replication set, same exported profile replayed at three radii |

To recompute counts from the saved arrays, run from the repository root with NumPy installed.

```bash
python examples/steering/validation/audit.py examples/steering/validation/report.html
python examples/steering/validation/audit.py examples/steering/validation/matched.json.gz
python examples/steering/validation/audit.py examples/steering/validation/bolt_doses.json.gz
python examples/steering/validation/audit.py examples/steering/validation/timesfm_doses.json.gz
```

The published runs used CPU float32, PyTorch 2.12.0, NumPy 2.1.0, SciPy 1.18.0, chronos-forecasting 2.3.1, TimesFM 2.0.2 and transformers 4.57.6. The cached checkpoint revisions were `772f3d25d38aec6d914c8949dab4462e2d46f5d8` for Bolt and `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` for TimesFM. Exported profiles also verify their base-weight fingerprint when loaded. Thread count matters for tight numerical replay checks. An archived TimesFM level export differed by up to 1.33e−6 on two forecast coordinates in a three-thread rerun and matched its archived output using four threads.

To rerun on the pretrained models, start in `tsfm_lens/` with both adapters and checkpoints available. Save the replication config alongside your run outputs or in a temporary file.

```bash
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
python run.py --config ../examples/steering/readout_validation.yaml --stages steering --force steering
sed -e 's/seed: 731/seed: 20261120/' -e 's/name: steering_readout_verified/name: steering_readout_replication/' ../examples/steering/readout_validation.yaml > /tmp/steering_replication.yaml
python run.py --config /tmp/steering_replication.yaml --stages steering --force steering
PYTHONPATH=. python ../examples/steering/validation/check_exports.py /tmp/steering_replication.yaml runs/steering_readout_replication/steering/report.html
```

The export check loads the ZIPs embedded in the report, reproduces the worst nonzero bounds and combined-request cases, then doubles and quadruples the inference radius for those same cases. The report's counts use the original limit. The linked compressed JSON files retain split membership, controller settings, discovery, forecasts, diagnostics and outcomes.

The [original all-type report](../all_types_report.html), [Bolt dose report](../dose_pilot_report.html), [TimesFM dose report](../dose_timesfm_report.html) and [mixed pilot](../mixed_pilot.json) record the earlier search at commit `4f92d9d`.

To repeat the M4 daily transfer, pass the local GIFT-Eval Arrow shard from dataset revision `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`. The saved result includes the source file SHA-256, item IDs, forecasts and paired losses. From `tsfm_lens/` run the command below.

```bash
PYTHONPATH=. python ../examples/steering/validation/check_transfer.py runs/steering_readout_verified/steering/report.html /path/to/m4_daily/data-00000-of-00001.arrow --out m4_daily.json
```
