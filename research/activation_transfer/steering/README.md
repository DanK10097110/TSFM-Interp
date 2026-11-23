# Depth matters differently for a reversal and a periodic cue

We fit a fixed direction for each of two paired input cues and three block depths using 96 daily calibration series. A ridge map trained on separate clean histories carried each source direction to the other model. The new edit was applied to the final model token on 96 held-out M4 daily histories and 96 weekly histories. The native reference here is the receiver's *independently fitted* direction for the same cue. An additional source-reference score checks whether the actual source forecast behavior was reproduced. [Protocol](PROTOCOL.md) · [Source, target and sham scores](results/results.json)

| Population | Quarter depth | Mid-depth | Three-quarter depth | Quarter minus three-quarter, paired 95% CI |
|---|---:|---:|---:|---:|
| Daily, both cues | .773 | .637 | .601 | +.172 [.106,.268] |
| Weekly, both cues | .828 | .651 | .571 | +.257 [.202,.329] |

The periodic cue accounts for the early advantage. Its quarter-minus-three-quarter score was +.363 [.230,.556] daily and +.529 [.419,.672] weekly. For the recent reversal the same differences were **−.019** [−.022,−.015] and **−.015** [−.017,−.012]. At all three depths both checkpoints therefore gave similarly good recipient-native recovery for the reversal. The two cues leave the final observed value unchanged. TimesFM's last token is a 32-observation data patch; Bolt's is a registration token. This comparison concerns these functional sites, and the attainable target effect itself changes with site. [Paired depth comparisons](results/depth_pairs.json)

A high receiver-native score can coexist with a source effect more than an order of magnitude larger. For the source-effect test using the new state-dependent API, which estimates a separate read direction, see the amplification experiment. [ActivationContrast experiment](amplify/README.md) · [All per-item effects](results/effects.npz)

```bash
python -m research.activation_transfer.steering.study --out /tmp/tsfm-fixed-cues
python -m research.activation_transfer.steering.analyze /tmp/tsfm-fixed-cues research/activation_transfer/steering/results
python -m research.activation_transfer.steering.depths
```
