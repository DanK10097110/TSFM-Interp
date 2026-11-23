# Can a shared forecast change mean that a feature helped both forecasters?

On two different daily panels, a translated feature removal usually changed eight-step mean absolute error in the same direction as its donor. Agreement was imperfect even when we replaced the translated waveform with the *exact* donor waveform. The two models' starting forecasts matter to the usefulness of the same write. [Protocol recorded before retrieving actual futures](PROTOCOL.md)

The original material builder set aside the eight observations following the standardized 480-step input. We retrieved them from the two original Arrow sources and verified every preceding standardized FP32 history against its archived model input, element by element. All 512 series had eight finite outcomes. No future entered feature selection, calibration or the intervention. We used each model's median, each active source feature's measured natural removal, and its existing physically measured eight-lead translated response. A positive change in MAE indicates that removing the feature hurt the forecast. The fraction in each table cell averages feature-wise agreement among donors with eight or more firing histories. The whole-history bootstrap preserves coactivation on each sampled series. [Outcome extraction, controls and paired scores](run.py) · [Frozen forecasts and outcomes on the focal panel](results/blind.npz)

| Source → receiver | Panel | Direction of MAE change agreed, physical | Agreement expected from each feature's sign frequencies | Exact donor edit on receiver's clean forecast | Actual translated edit on donor's clean forecast |
|---|---|---:|---:|---:|---:|
| Bolt → TimesFM | Focal daily, 128 | .839 [.771, .904] | .535 | .837 | .998 |
| TimesFM → Bolt | Focal daily, 128 | .923 [.882, .955] | .601 | .937 | .987 |
| Bolt → TimesFM | Additional daily, 128 | .842 [.772, .904] | .534 | .859 | .983 |
| TimesFM → Bolt | Additional daily, 128 | .866 [.813, .915] | .590 | .858 | .969 |
| Bolt → TimesFM | Focal traffic, 128 | .890 [.810, .951] | .666 | .895 | .995 |
| TimesFM → Bolt | Focal traffic, 128 | .775 [.641, .894] | .661 | .866 | .890 |
| Bolt → TimesFM | Additional traffic, 128 | .787 [.696, .873] | .603 | .797 | .975 |
| TimesFM → Bolt | Additional traffic, 128 | .771 [.621, .883] | .741 | .869 | .902 |

The two rightmost columns are arithmetic controls, using the clean predictions and the two measured intervention waveforms on the same history. They permit a finite change to cross the realized observation. A perfect translation can improve one model and hurt the other when their original errors point in different directions. On both daily panels, replacing the translated waveform with the donor's exact one changed agreement much less than making the two starting forecasts equal. The traffic reverse direction was different: two eligible TimesFM features dominated, and translation error also mattered appreciably. This is correspondence for these selected removals, not a policy for choosing useful edits before seeing the future.

On the focal daily panel, Bolt removals changed MAE by .0074 history SD in absolute value on average, compared with .0636 for TimesFM removals. Their *signed* mean increases were .0016 and .0455, against ordinary all-series MAEs of .156 and .159. Small changes can agree in sign while having little impact on accuracy. An independent event-by-event computation reproduced all eight primary agreement scores. A post-hoc partition by the clean forecasts' position relative to the realized future had only 6 to 13 daily histories in the most discordant stratum; its feature-specific sign estimates were unstable. These panels may overlap public pretraining, and nearby traffic sensors share conditions. [All MAE/MSE scores and denominators](results/summary.json) · [Independent recount and stratum sizes](results/strata.json)
