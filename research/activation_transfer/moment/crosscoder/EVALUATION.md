# Predicting four native MOMENT-large SAE doses from MOMENT-small

On the 385 NOAA station/date histories, masking the large input during crosscoder training improved transfer from the small view relative to standard joint training. Breaking the historical small–large pairing during otherwise identical masked training preserved most of that gain. The matched-minus-shuffled difference was **+.033 dose skill** (95% crossed interval −.076 to .118), and **−.005 average precision** (−.059 to .043). The external data support coefficient prediction for these four selected atoms. They leave the added value of genuine cross-checkpoint correspondence unresolved. [Frozen readout design](EVALUATION_PROTOCOL.md) · [Dictionary training](DICTIONARY_PROTOCOL.md) · [Per-atom results and paired comparisons](results/evaluation.json)

| Small input representation | NOAA dose skill / AP | Held-out M4 daily dose skill / AP | Monash traffic dose skill / AP |
|---|---:|---:|---:|
| Zero dose | .000 / .369 | .000 / .275 | .000 / .441 |
| Constant historical mean | .233 / .369 | .216 / .275 | −1.112 / .441 |
| Raw hidden state | .367 / .703 | .823 / .898 | −4.593 / .675 |
| Independent native small SAE | .478 / .747 | .773 / .861 | −.724 / .581 |
| Joint crosscoder | .393 / .689 | .765 / .837 | −2.502 / .543 |
| Masked crosscoder | .546 / .733 | .799 / .871 | .005 / .817 |
| Shuffled history pairs, masked training | .513 / .739 | .793 / .866 | −.971 / .705 |
| Joint crosscoder, privileged current **small and large** | .628 / .788 | .877 / .926 | −7.585 / .603 |

Skill is one minus squared error divided by true dose energy, calculated for each target then equally averaged. AP ranks the signed ridge score before the validation-calibrated zero gate. Seeds are averaged at the metric level. Target IDs in order are 187, 1236, 1867 and 1235; their NOAA firing counts were 252, 161, 93 and 62 out of 385. The small-only joint dose skill was .467/.318 across seeds, masked .540/.551 and shuffled .526/.500. Masked-minus-standard-joint was +.153 [.056, .892], masked-minus-raw +.179 [−.023, 2.739], and masked-minus-independent-SAE +.068 [−.007, .679]. Providing both current views gained .236 [.113, .721] against ordinary joint source-only. The privileged model never enters the source-only intervention. [Fits and firing thresholds](results/predictions.json) · [Privileged fits](results/oracle_predictions.json)

The 768 M4 fitting histories gave the four targets 224/229/175/143 positive coefficients. The next 256 items selected ridge penalties and amplitude thresholds. The independent source SAE has 2,048 coordinates and was previously trained on the entire historical cohort, while the 1,024-coordinate crosscoders used the first 768 histories. The target atoms were nominated before this study using these historical data. The 2026 NOAA panel had already been used for other interventions in this project, so it is external to M4 readout calibration and familiar to the investigators. All thresholds and readouts were frozen before scoring the disjoint 128 daily, 128 traffic and external 35 × 11 NOAA histories. The small representations and correlated target labels in calibration make this supervised decoding. The separate receiving experiment measures forecast changes after physically applying these doses. [Dose arrays](results/predictions.npz) · [Physical intervention design](INTERVENTION_PROTOCOL.md)

On daily the masked-minus-shuffled difference was +.006 [−.023, .030]. Traffic illustrates the importance of retaining false alarms and reporting counts. Native positive counts were **3, 103, 119, 1**. For atom 187, raw ridge recovered all three true firings and made 107 false alarms, producing −17.739 skill. For atom 1235, the two masked fits found the lone true firing, alongside 35 and 22 false alarms. The equal-atom traffic average is highly sensitive to those two rare atoms. Among the crossed NOAA draws 2,993/3,000 had positive energy for all four atoms; among traffic draws only 1,824/3,000 did. Intervals are conditional on all four denominators being defined. They describe fixed atoms and two fixed seeds. No random-atom or unseen-domain population inference follows. [Per-atom skill, AP, firing TP/FP/FN, eligible draws](results/evaluation.json)

The ordinary large states reproduced the archived NOAA target doses within 0.000006 and reproduced every firing. Direct Cholesky solves and exhaustive gate enumeration reproduced all ten methods' published scores and doses within 0.000000001 over every validation and test item; every deployed gate agreed. Five known-answer tests passed. Five isolated wrong implementations failed their intended assertions: splitting a tied validation gate, using unscaled ridge, breaking AP ties before ranking, energy weighting instead of equal atom weighting, and treating individual station/date rows as independent. [Numerical replay](results/evaluation_audit.json) · [Audit script](audit_evaluation.py) · [Planted regressions](evaluation_mutations.py)

Run from the repository root with the repository and tsfm_lens on PYTHONPATH, the pinned study environment, and the cached dictionary and MOMENT hidden-state archives. The first command publishes the eight small-only and separate two privileged prediction arrays atomically.

```sh
export PYTHONPATH=.:tsfm_lens OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m research.activation_transfer.moment.crosscoder.evaluate raw sae joint1 joint2 masked1 masked2 shuffled1 shuffled2 oracle_joint1 oracle_joint2
python -m research.activation_transfer.moment.crosscoder.evaluation_scores
python -m research.activation_transfer.moment.crosscoder.audit_evaluation
python -m research.activation_transfer.moment.crosscoder.evaluation_mutations
python -m pytest -q research/activation_transfer/moment/crosscoder/test_evaluation.py
```
