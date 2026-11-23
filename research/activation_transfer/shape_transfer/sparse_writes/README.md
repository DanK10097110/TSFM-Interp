# A source SAE effect can be written through a few firing target features

On the receiving checkpoint, only the 32 SAE features already firing on that history were eligible. The source intervention was its genuine one-feature removal at its natural activation. We predicted the removal's forecast signature using the source's clean-state derivative, used the receiving derivative to select an edit, and then executed that edit inside the target model. The solver did not see the post-ablation source forecast. Every signed adjustment left the target coefficient between zero and twice its initial value and preserved the state's reconstruction residual. [Protocol](PROTOCOL.md) · [Actual selection and intervention](run.py)

| Centered recovery on 192 daily histories | TimesFM → Bolt | Bolt → TimesFM |
|---|---:|---:|
| Remove a calibration-selected target feature at full strength | .005 | .000 |
| Choose a target feature for this history, then remove it at full strength | .017 | −.101 |
| Choose a target feature and partially attenuate it | .148 | .610 |
| Use a fixed target identity and adjust its signed dose per history | .032 | .215 |
| Choose one target feature and adjust its signed dose | .180 | .674 |
| Choose up to four and adjust their signed doses | **.476** | **.869** |
| Same four-feature search after reassigning decoder identities | .462 | .828 |
| Dense history-specific write | .859 | .944 |

**A failed literal knockout often has the right sign and the wrong dose.** For Bolt → TimesFM, a full target knockout improved the predicted weighted waveform error in only 13 of 520 firing feature–history cases. In the other 507, a favorable target direction always existed. Even the least excessive favorable full knockout was a median **10.83 times its own best dose**. In squared-error terms a dose beyond twice that optimum is worse than doing nothing. The actual partial-removal intervention recovered .610 of the centered source effect. In the reverse direction a full removal improved the *predicted weighted* objective in 690 of 691 cases, and actual full-waveform recovery was .827, while centered recovery was only .017. In that direction, a suitable level change was relatively easy to find and a faithful temporal pattern was harder. [Exact case counts and dose ratios](results/reasons.json)

**Identity also varies across histories.** The calibration-selected signed target features were actually available on 86.1% of TimesFM-source and 98.8% of Bolt-source daily firing cases. The locally selected single feature agreed with that fixed target ID on only 34.4% and 27.9% of cases, respectively. Access to the target's current responsiveness and a flexible dose contributed to the result. The use of bounded SAE coordinates gives the target edit a concrete sparse implementation. [Selection and per-feature counts](results/results.json)

**The dictionary control narrows the interpretation.** The four correct target entries improved daily centered recovery over one global decoder-ID permutation by .013 [−.018, .048] into Bolt and .041 [.002, .125] into TimesFM. Keeping the same number of candidates and searching them by forecast response accounts for much of the result. On the VM cohort, four correctly selected target features reached .213 into Bolt and .746 into TimesFM, with only two and four frequent source features. The latter exceeded the original full-dose dense solver's .348 point estimate. A smaller dense write had already shown strong dose sensitivity there. Neither sparse success nor the dense solve supplies a universal donor → recipient feature dictionary. [All outcomes and paired intervals](results/results.json) · [Dense dose experiment](../full_space/README.md)

We independently verified every saved coefficient against the receiver SAE's natural active set and bounds. An additive production hook reconstructed 82 interventions directly from the recorded IDs and signed amounts, including zero choices, to within 0.00000287 history standard deviations. The separate stability study then tested what happens to those IDs and actual forecast responses when the same histories are slightly perturbed. [Independent coefficient and forecast audit](results/audit.json) · [Stability under observational noise](../sparse_stability/README.md)
