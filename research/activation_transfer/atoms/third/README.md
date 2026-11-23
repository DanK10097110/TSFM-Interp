# TimesFM and Chronos-2 literal atom ablations

We used the archived 6,144-atom TopK dictionaries at TimesFM block 10 and Chronos-2 block 5. A source feature was removed on each history where it fired. The transformed source removal was applied in the recipient at the same final 32 time steps. A matched case and active coefficient were retained when replacing the correct direction with each of four wrong feature directions. The coactivation comparator removed the naturally active contribution of the best-correlated target feature from the original clean training pairs. [Protocol](PROTOCOL.md)

## Results

On M4 daily, each source checkpoint produced a more similar forecast change in its recipient when we used the mapped feature direction than when we substituted another feature’s direction at the same norm. Both gains in signed cosine were close to .37. Matching the size of the source’s effect was much harder, particularly when TimesFM received a Chronos-2 edit. [Chronos-2 source scores](chronos2_scores.json) · [TimesFM source scores](timesfm_scores.json)

| Source → recipient | Population | Firing source atoms | Mapped / wrong / coactive cosine | Mapped minus wrong, 95% interval | Simultaneous lower bound | Mapped full skill, 95% interval |
|---|---|---:|---:|---:|---:|---:|
| Chronos-2 → TimesFM | M4 daily | 16/16 | .514 / .143 / .404 | +.370 [.252, .537] | .212 | −9.387 [−26.293, −8.480] |
| TimesFM → Chronos-2 | M4 daily | 16/16 | .496 / .130 / .327 | +.366 [.251, .470] | .235 | +.110 [−.00031, .226] |
| Chronos-2 → TimesFM | Electricity | 4/16 | .324 / −.276 / .000 | +.600 [.280, .890] | .241 | −97.245 [−188.721, .200] |
| TimesFM → Chronos-2 | Electricity | 2/16 | .830 / .042 / .373 | +.788 [.728, .843] | .719 | +.266 [.238, .300] |

Each arm’s forecast change is compared with the same natural source ablation. Skill is one minus squared error divided by source-effect energy, pooled over active series and eight horizons within each source atom and then averaged over atoms. Zero corresponds to a forecast of no effect. On daily data, every Chronos-2 → TimesFM atom scored below zero (median −8.67). In the reverse direction, 11/16 had positive skill and its macro interval reached zero. Cosine improvements survived restriction to at least eight firing items per atom, at +.480 across 11 Chronos-2 atoms and +.359 across 14 TimesFM atoms. All four registered full-response cosine comparisons passed the one-sided multiple-comparison rule. Intervals resample series 4,000 times with maps and selected feature IDs fixed. The simultaneous lower bounds use the 1.25th percentile. [Chronos-2 source raw outcomes](chronos2_effects.npz) · [TimesFM source raw outcomes](timesfm_effects.npz)

Removing each individual response’s eight-horizon mean reduced the daily cosine advantages to +.118 [.008, .212] for Chronos-2 → TimesFM and +.121 [.021, .214] for TimesFM → Chronos-2. Centered skills were −6.583 [−14.501, −5.035] and +.046 [.013, .077]. The remaining advantage shows some correspondence in horizon shape. The larger full-vector advantage shows that the overall forecast level contributes substantially. These are secondary paired 95% intervals.

The electricity comparisons have little feature coverage. The four Chronos-2 features fired on 20, 1, 1 and 1 series, and their target coactivation matches never fired on any of those 23 cases. Both TimesFM features fired more often, on 42 and 160 series. Their Chronos-2 coactivation matches fired on 80.7% of the pooled active feature–series pairs. The two domains in this extension reuse the original study items, and the TimesFM features use the same archived dictionary and identical atom IDs. The extension establishes transfer of directional information to a third checkpoint at these sites while exposing sensitivity to the recipient and the dose. [Selection and replay metadata, Chronos-2 source](chronos2_metadata.json) · [TimesFM source](timesfm_metadata.json)

## Execution checks

Chronos-2’s live forecast contained 35 time tokens, comprising 30 context patches, a register and four forecast placeholders. The patch targeted only context slots 28 and 29, which jointly cover time indices 448–479. Separately extracted states agreed with the same live block’s output to 1.20×10⁻⁷ when running the ordinary 64-step forecast at a different batch size. For Chronos-2 the measured outcome is the first eight median predictions from that 64-step call. This matches its archived SAE activation configuration. The maximum packed zero-dose RMS across both ordered pairs was 1.61×10⁻⁶. Independently rerunning the established TimesFM source ablation on these same histories agreed with the original archive to 2.06×10⁻⁶. Raw coactivation responses retain packed-batch noise when the matched coefficient is exactly zero, and the score uses the mathematical zero change in those cases. [Live-path audit](capture.json)

## Reproduction

From the repository root set `PYTHONPATH=.:tsfm_lens`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=3`, `MKL_NUM_THREADS=3`, and `OPENBLAS_NUM_THREADS=1`. The original study’s manifest, fixed pretrained snapshots, SAE checkpoints and TimesFM source outcomes are the prerequisites. If the original `/tmp/tsfm_atoms_20261120` raw context and TimesFM captures are absent, regenerate them using `python -m research.activation_transfer.atoms.materials data` and `python -m research.activation_transfer.atoms.materials timesfm`. The third study writes its different Chronos-2 capture under `/tmp/tsfm_third_atoms_20261120`.

```bash
python -m research.activation_transfer.atoms.third.collect
python -m research.activation_transfer.atoms.third.run timesfm
python -m research.activation_transfer.atoms.third.run chronos2
python -m research.activation_transfer.atoms.third.analyze timesfm
python -m research.activation_transfer.atoms.third.analyze chronos2
python -m pytest -q research/activation_transfer/atoms/third/test_third.py
```
