# Does a local change of sparse-feature identity disrupt forecast control?

We perturbed the same 96 M4 daily histories twice and reran bounded, signed, one-target-atom selection in each direction between Chronos-Bolt and TimesFM. A single zero-mean observational noise vector per history was scaled to 0.1% and 2% of that history's standard deviation. The original and perturbed inputs each used their own normal model preprocessing. Both models had access to the same perturbed observations. The 96 series are the first half of the original sparse-writes daily test cohort. The experiment measures sensitivity around those histories. [Frozen protocol](PROTOCOL.md) · [Series IDs and input hashes](results/manifest.json)

For each of the 16 fixed source atoms, we compared target choices only where the source atom fired on both histories. The new selection uses its current natural intensity and the two models' current forecast derivatives. It chooses a signed adjustment to one of the target's 32 positive SAE decoder contributions. Carrying the original target identity uses exactly the same local coefficient optimization if that atom still fires, and zero otherwise. Both methods leave the adjusted coefficient between zero and twice its current value, preserve the SAE reconstruction residual, and cap the physical hidden write at 20% of the original training-state radius. We then measured the receiving model's actual eight-lead median forecast change against the source's actual ablation on the perturbed history. [Original sparse-write algorithm](../sparse_writes/README.md)

**A selected target atom can change even when the entire target TopK support stays identical.** At 0.1% observation noise, Bolt→TimesFM switched the chosen target atom on 17 of 283 continuously firing source cases. TimesFM retained all 32 active target features on every one of the 96 histories. At 2% noise, selecting the current best active atom improved the centered forecast response in this direction. In the reverse direction, switching was less common and a single target atom recovered little centered forecast energy under either policy.

### Which identities changed?

A “case” is a history and source atom. The first column of counts shows which original cases still fire; these form the denominator for switches. The target-support column counts histories on which at least one of the 32 active target features was replaced. The final columns assign every functional switch to one of three mutually exclusive outcomes.

| Source → target | Noise RMS | Source firings retained | Target support changed | Selected target ID changed, 95% interval | Old winner left | Old stayed, new winner entered | Both winners stayed |
|---|---:|---:|---:|---|---:|---:|---:|
| Bolt → TimesFM | 0.1% | 283/284 | 0/96 | 17/283 = 6.0% [3.4%, 8.9%] | 0 | 0 | 17 |
| Bolt → TimesFM | 2% | 270/284 | 21/96 | 95/270 = 35.2% [29.3%, 41.2%] | 1 | 1 | 93 |
| TimesFM → Bolt | 0.1% | 364/364 | 8/96 | 4/364 = 1.1% [0.3%, 2.3%] | 0 | 0 | 4 |
| TimesFM → Bolt | 2% | 353/364 | 80/96 | 35/353 = 9.9% [6.9%, 13.2%] | 1 | 0 | 34 |

At 2% noise the recipient retained 3,051/3,072 original active slots for TimesFM and 2,941/3,072 for Bolt. Of the 130 functional switches across both directions, 127 re-ranked two candidates that both remained active. The ranking depends on the donor's current response signature, the recipient's current feature-response derivatives and its natural coefficient bounds. This design does not isolate those three contributions. Among the 16 frozen source features, Bolt gained 1 and 9 newly firing cases at the two doses. TimesFM gained 0 and 8. [All counts and intervals](results/summary.json)

### Did reselection change real forecast control?

Recovery is one minus pooled squared error divided by pooled source ablation energy, with zero predicting no forecast change and one recovering it exactly. The centered score removes each eight-lead waveform's own mean. All 16 fixed source atoms are eligible. Consequently high-energy cases carry more weight here than in the original study's equal-atom macro score. These are 4,000 paired whole-history bootstrap intervals conditional on the two models, fixed features, noise directions and chosen series. Intervals are pointwise and the smallest switch groups contain few histories.

| Source → target | Noise RMS | Centered recovery, reselect / carry | Paired improvement [95% interval] | Full recovery, reselect / carry | Switch-only centered improvement [95% interval] |
|---|---:|---:|---|---:|---|
| Bolt → TimesFM | 0.1% | .771 / .772 | −.001 [−.007, .003] | .980 / .980 | −.018 [−.054, .112] |
| Bolt → TimesFM | 2% | .770 / .719 | +.052 [.006, .126] | .963 / .962 | +.140 [.015, .311] |
| TimesFM → Bolt | 0.1% | .026 / .029 | −.003 [−.011, .003] | .527 / .522 | −.045 [−.063, .949] |
| TimesFM → Bolt | 2% | .027 / .027 | −.001 [−.004, .002] | .521 / .522 | −.030 [−.095, .115] |

The 95 moderate-noise switches for Bolt→TimesFM occurred on 61 distinct histories. Allowing reselection recovered .679 of the centered response on those cases, compared with .539 for refitting the original identity. Considering only the 93 cases where *both* candidates survived gave a paired gain of .123 [.001, .299]. Full-response gain across all 270 cases was .001 [−.035, .038]. At the smaller dose, the 17 identity changes had little influence on aggregate skill. For TimesFM→Bolt the 35 moderate switches affected 30 histories, and either single-feature method performed poorly on centered shape. The numerical similarity in that direction is compatible with a limited shared ability to realize the source shape through one target atom. These results characterize the chosen one-atom control policy around real daily histories; they do not establish a shared atom meaning between dictionaries. [Per-case Bolt source data](results/bolt.npz) · [Per-case TimesFM source data](results/timesfm.npz)

### Checks

The clean calculation exactly recovered the archived winning ID for every source case, with all clean signed coefficients matching within tolerance. On every unchanged identity, reselection and refitting returned the same coefficient and therefore shared one physical target forecast call. Both ordinary forecast and zero-patch replay agreed within 0.0000015 normalized units. The source's first-order positive-control recovery for centered responses was .973/.947 at the two Bolt-source doses and .929/.921 at the TimesFM-source doses. The perturbed source ablation was measured in a separate real forecast call from every chosen target intervention. [Bolt numerical checks](results/bolt_checks.json) · [TimesFM numerical checks](results/timesfm_checks.json)


## Reproduction

Run the [historical sparse-writes study](../sparse_writes/README.md) and its upstream materialization to populate the fixed unperturbed selection and original full-space gradients. Perturbed activations, forecast derivatives and intermediate source interventions are stored in the scratch directory. The shipped per-pair responses are sufficient for the final analysis.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
pkg=research.activation_transfer.shape_transfer
scratch=/tmp/tsfm_sparse_stability_20261120
python -m "$pkg.sparse_stability.prepare"
for m in bolt timesfm; do
    python -m "$pkg.fresh_cohort.run" collect "$m" --cache "$scratch" --out "$scratch"
done
for m in timesfm bolt; do python -m "$pkg.sparse_stability.run" "$m"; done
python -m "$pkg.sparse_stability.score"
python -m pytest -q research/activation_transfer/shape_transfer/sparse_stability/test_stability.py
```

In an isolated mutation test, checking whether the old winner belonged to the *old* support instead of the perturbed support made the exit-versus-reranking test fail at its first assertion (one failure). Restoring the implementation passed the test. An independent loop over archived series, atom IDs and raw forecasts reproduced all 60 nonempty subgroup-by-arm skill values and their item and event counts.
