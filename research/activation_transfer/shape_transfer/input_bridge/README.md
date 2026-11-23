# Can a shared edit to the input reproduce a natural hidden intervention?

We erased the firing coefficient of a naturally active middle-block SAE feature and observed the donor's next 16 median predictions. For the same history, we searched for one bounded edit to the final 32 observations that could reproduce the forecast change. Both models received exactly the same edited observations in their usual public forecasting interface. This asks whether an actual natural response can be represented in a small set of common input units. [Prospective protocol](PROTOCOL.md)

The twelve changeable coordinates were low-frequency DCT modes, including a constant shift of the last 32 inputs. Each of the 256 daily/traffic histories already had a common fixed normalization. An edit could use at most .25 RMS in those units over the last 32 samples. The original observed prefix and any preprocessing internal to each forecaster stayed intact.

Positive source coefficients, decoder directions and the local hidden Jacobian provided a **predicted donor knockout** at leads 1–8. We built each history's 8 × 12 public-input Jacobian with 24 forecasts per model at ±.02 tail RMS. The donor-only method fits the donor's early response and supplies its edited history directly to the recipient, without querying recipient sensitivities. Two higher-access variants fit the same goal to either the recipient's input sensitivity or both models' sensitivities jointly. A fixed signed permutation supplies a wrong set of physical modes with identical RMS. The zero reference predicts no change. The evaluation target in all cases is the separately *observed hidden knockout*, including its unrequested leads 9–16.

The four model/domain populations were 15/6 qualifying Bolt features (360/242 total firings) on daily/traffic and 15/2 TimesFM features (455/142 firings). Each feature needed eight distinct firing histories. Skill is one minus squared error over donor energy, averaging equally over eligible features. A negative score means the no-change prediction has smaller error. The paired uncertainty resamples features and histories. The full-waveform score includes each eight-lead block's mean, while the centered score evaluates changes in its shape.

## A shared short-horizon input equivalent can be constructed

On daily histories, each model could reproduce much of its own hidden knockout by changing the observations. Passing the donor's chosen input directly to the other model lost most or all of the forecast shape. **Jointly fitting one physical input edit to both models did recover shape in both**. This distinguishes existence of a shared approximate solution from being able to identify it using only the donor.

| Donor → recipient | Domain | Donor-fitted, applied to donor | Same edit, applied to recipient [crossed 95%] | Scrambled, recipient | Recipient-fitted, recipient | Jointly fitted, donor / recipient | Hidden-space recipient reference |
|---|---|---:|---:|---:|---:|---:|---:|
| Bolt → TimesFM | Daily | .900 | −.605 [−1.235, −.181] | −1.297 | .868 | .749 / .788 | .972 |
| Bolt → TimesFM | Traffic | .982 | .624 [.273, .849] | −.120 | .973 | .972 / .961 | .994 |
| TimesFM → Bolt | Daily | .635 | .140 [−.047, .293] | −.091 | .709 | .602 / .632 | .902 |
| TimesFM → Bolt | Traffic | .924 | .452 [−.347, .771] | −1.479 | .656 | .882 / .710 | .863 |

All cells above are actual centered early forecast responses. On daily, the recipient-fitted edit sent back to the donor scored −.437 for Bolt and .047 for TimesFM. The two unilateral solutions therefore selected different responses in both directions. Input sensitivity on the receiver was enough to construct a better early edit, at the cost of 24 public receiver forecasts per history for the twelve probes; the probes can be reused across all source features. The hidden-space reference has access to the full current hidden derivatives of both models. [Full estimates, uncertainty and atom results](results/summary.json)

Traffic gave stronger unilateral transfer. Every one of the six eligible Bolt atoms had positive recipient skill (.176–.955). The two eligible TimesFM atoms had 14 and 128 firings out of 128 histories, with recipient skills of .183 and .720. Across eight prespecified comparisons, Bolt → TimesFM traffic beat the zero-effect reference after adjustment (lower bound .096), and TimesFM → Bolt traffic beat its matched scramble (difference 1.931, lower bound .653). No model-domain cell cleared both controls after adjustment. The uncertainty for the two-atom cell is correspondingly wide.

## Means and unseen leads

| Donor → recipient | Domain | Donor-fitted edit, recipient early full | Recipient late full | Recipient late centered [crossed 95%] | Donor late centered | Joint fit, recipient late centered |
|---|---|---:|---:|---:|---:|---:|
| Bolt → TimesFM | Daily | .687 | .470 | −.667 [−1.249, −.326] | .022 | −.522 |
| Bolt → TimesFM | Traffic | .730 | −.083 | −.254 [−.729, .103] | −.091 | −.144 |
| TimesFM → Bolt | Daily | .514 | .586 | −.742 [−1.633, −.370] | −.193 | −.913 |
| TimesFM → Bolt | Traffic | .869 | .518 | −.125 [−3.235, .519] | .522 | −.544 |

Daily full-waveform scores retained some of the mean response despite poor centered transfer. All four unilateral and all four joint recipient estimates of previously unrequested late shape were negative. The donor itself usually lost the late shape even when early matching was accurate. TimesFM traffic was the exception, with .522 donor late centered skill on two features [−.136, .678]. These are interventions on one recent input window, and their failures leave open other physical edits, longer support or a different dose.

## Size and local linearity

For Bolt-donor removals, median donor-fitted input RMS was .010 daily and .011 traffic. **No eligible input edits in any of its four arms reached the .25 limit.** Its large daily self-to-recipient gap therefore did not arise from budget clipping. For TimesFM-donor removals those medians before capping were .090/.318; donor/recipient/joint clipping affected 62/69/86 of 455 daily cases and 110/110/126 of 142 traffic cases. The scramble had exactly the donor edit's requested norm and cap status on every firing.

For TimesFM → Bolt daily, the donor-input linear prediction of the observed donor knockout scored .842 before the cap and .833 after it; the actual edited donor scored .635. On traffic, the recipient-fitted linear prediction fell from .970 to .919 with clipping, then to .656 under the actual Bolt forward pass. Matching an infinitesimal derivative and obtaining the same finite-dose hidden response are separately measurable limitations.

## Numerical checks

An independent random direction with .01 RMS on the same 32-sample window gave public finite-difference errors of 1.61%/1.64% for Bolt and 1.47%/.75% for TimesFM on daily/traffic (8 histories each, pooled responses). The DCT orthonormality error was 1.1 × 10⁻¹⁵; maximum clean forecast differences from the earlier archive were 1.43 × 10⁻⁶ and zero. [Bolt check](results/bolt_probe.json) · [TimesFM check](results/timesfm_probe.json)

A separate solver used direct normal equations and SciPy DCT synthesis on four naturally firing examples, one per direction and domain. It repeated each underlying public input derivative with a new batch layout, removed the original feature again with an independent native hook, and replayed 32 physical donor/receiver input forecasts. Maximum absolute differences were 9.54 × 10⁻⁷ for the native knockout and 1.44 × 10⁻⁶ for the actual input effect. The independent feature-wise score calculation agreed to 4.45 × 10⁻¹⁶. Three tests passed. Five planted defects each failed a numerical assertion, covering the DCT phase, position of the edited window, per-history damping, finite input budget and signed derangement. [Independent audit](results/audit.json) · [Mutation checks](check_tests.py)

## Reproduction

Use the project's pinned offline environment and regenerate the upstream blind-frequency cohort and ordinary hidden measurements first. The upstream files supply the original standardized histories, frozen IDs, donor coefficients, actual SAE-removal effects, source sensitivity and historical shape weighting.

```sh
export PYTHONPATH=.:tsfm_lens HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
p=research.activation_transfer.shape_transfer.input_bridge
for m in bolt timesfm; do python -m "$p.probe" "$m"; done
for m in bolt timesfm; do python -m "$p.run" "$m"; done
python -m "$p.score"
python -m "$p.audit"
python -m pytest -q research/activation_transfer/shape_transfer/input_bridge/test_bridge.py
python -m "$p.check_tests"
```

The public sensitivities, centered differences and clean forecasts are archived alongside the effects in `results/*_probe.npz`. Regeneration also writes a transient copy to `/tmp/tsfm_input_bridge_20261121`. The two model result archives contain IDs, masks, doses, actual recipient and donor forecast changes and the exact input coefficients. The summary records per-feature scores, crossed and fixed-feature uncertainty, all norm statistics and all four early/late metrics. [Scoring code](score.py)
