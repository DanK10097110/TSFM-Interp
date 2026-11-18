# Section49 hypothesis investigation — session handoff

Status: bounded investigation complete; experimental branches and their audits are closed.
The session used roughly three hours of wall time, cached models and CPU-only
workers. This is not a proof of universal interpretability or impossibility.
The canonical exact-results ledger is [FINDINGS.md](../../FINDINGS.md), PM-35
onward, with protocols and decisions in [ROADMAP.md](../../ROADMAP.md), §49.38–40.
The [portable report](../../examples/feature_search_stage49/hypothesis_investigation/index.html)
includes copied per-experiment `knowledge.json` files and their hashes. Full raw
arrays, checkpoint hashes, frozen protocols, failed attempts and independent
arithmetic audits are under `tsfm_lens/runs/`.

## Questions and current evidence

| Hypothesis | Reviewed evidence | What it does not establish |
| --- | --- | --- |
| Fourier gains only occur because synthetic data are excessively structured | On192 fresh public-data sources, fixed depth1 improves TimesFM electricity, hurts both models on traffic, and is inconclusive for Bolt electricity and M4. A separate October2026 holdout from eight fixed weather stations worsens both observed means. Seasonal24 remains a strong baseline. | Universal forecasting improvement. Public-source pretraining overlap is unknown; the weather observations demonstrably postdate the cached weights but locations/climates may be familiar. |
| Models simply expect structure to disappear | Matched tiny-model training demonstrates a continuation-prior mechanism. A paired MSE/absolute-loss control shows that the prediction functional also changes optimal gain, including a nonunique median at50% survival. Actual weak-tone errors contain phase and off-frequency changes; scalar attenuation and alternative marginal quantiles leave substantial errors. Visible cessation/decay cues can help. | Attribution of pretrained-model behavior to its training distribution or objective. Native outputs here are medians. |
| Internal responses are universally jagged | Bolt/TimesFM source-average local curvature decreases across tested radii. Chronos-T5 has demonstrable tokenizer discontinuities, while continuous embedding paths become more locally linear. Frozen sparse dictionaries themselves introduce much larger chord roughness than the native states in the measured paths. | Global smoothness, faithful semantics, or smoothness in every direction. Small signals can still be badly forecast. |
| Cross-model comparison is useless | Frozen probes predict other-model forecasts and fusion responses on fresh synthetic sources. Generic random features explain much of the forecast-target benefit. A separate20 FIT/32 TEST condition-shift test retains trained-feature efficiency for a fusion-response target; the advantage partly depends on the anchor, and old frozen probes extrapolate poorly. | Shared semantic coordinates, new information beyond the input, universal transfer, or cross-model causal equivalence. |
| Wrong weak-tone responses have no simple explanation | A shared sine/cosine response map explains much of Bolt's period19.2 response across backgrounds. TimesFM retains larger unexplained and source-dependent effects. | An LTI system, a universally transferable map, or corrected forecasting accuracy. |
| SAEs are necessary, or native neurons are generally monosemantic | Direct FFN selectivity is model/layer/view dependent: early Bolt native coordinates outperform rotation controls in an exploratory slice, while TimesFM shows no matching advantage. The 32-source causal follow-up does not replicate the small selected-unit advantage. PCA/TopK/nested comparisons show no uniform SAE winner. A response-aware objective improves hidden-response fidelity but worsens mean native response fidelity through adverse tails, including real series. Separate repository §50 experiments already show no-SAE dense output control can work. | General monosemanticity from a small perturbation vocabulary, residual coordinates, reconstruction quality, or successful output control alone. |

## Interpretation rules that matter for reuse

- Resample independent sources, not their phases, horizons, tokens or repeated
  forecast origins. Public series may share calendar structure; uncertainty is
  conditional on the sampled datasets.
- Distinguish forecasting the truth from predicting a model's response. A model
  can be systematically wrong and therefore easy to predict.
- Fit decomposition and selection only on available history. An oracle fit to
  a future response is an explanatory ceiling, never a deployable method.
- For normalized activations, specify target coordinates and the information
  used by the inverse. Background Fourier coefficients are input information;
  a decoder using them is not a hidden-only probe.
- Native point predictions are median quantiles for both cached checkpoints.
  A central quantile average is not an exact conditional mean; marginal
  quantiles do not define joint forecast paths.
- Compare actual sparsity and preservation, not nominal dictionary size alone.
  Residual coordinates differ from post-activation FFN neurons. Selectivity to
  a small intervention vocabulary does not prove general monosemanticity.
- A failed test of one probe or intervention does not prove information is
  absent. Random initialization and generic nonlinear features are important
  controls for decodability claims.
- Numerical curvature ratios require a signal above the precision floor.
  Random Chronos-T5 softmax controls were excluded where nearly uniform outputs
  made those ratios unreliable.

## Execution and compute discipline

The investigation uses cached checkpoints, CPU workers with one thread and
3GiB caps, at most two native/fit workers concurrently. Saved-array analyses
precede new inference. Wider grids are parked when a smaller discriminating
study answers the question. Failures and access limitations remain recorded.

All branches are closed: response-aware native preservation and tail analysis,
fixed-support dictionary geometry, and the direct64-horizon recovery. The
original horizon attempt failed a historical comparator check; its failed outputs
supplied no scientific results. The outcome-blind recovery preserves that history,
passes exact same-batch prefix gates and independently bounds rebatching drift.
At64 steps, period22 response maps improve on period19.2 maps at both weak and
strong amplitude; this is not unique identification of a frequency mechanism.

Do not read preparation/readiness records as scientific results.

Validation includes140 earlier focused tests,98 newly completed-module tests,
38 final follow-up tests, two objective-control tests and five integrated mock
pipeline tests. All passed. Native replay gates, planted regressions and independent
arithmetic audits are recorded per experiment. The smoke suite retains one
existing return-value warning. Generated stage documentation and glossary checks
match. Final report rendering and local-reference checks are recorded in
`runs/investigation_validation4963000`.

## Reuse these distinctions before launching another experiment

- **State reconstruction, response reconstruction and output preservation are
  separate targets.** The response-aware SAE improves hidden differences on
  every paired receiver yet fails on mean output-response fidelity. Keep the
  adverse real-data cases; a better median does not erase them.
- **Measure the measuring tool.** Compare hard sparse encoders with fixed linear
  projections and local fixed-support controls before attributing feature
  discontinuities to a TSFM. Adaptive sparse PCA is also nonlinear.
- **Direct neuron selectivity is not selective causal control.** The larger
  frozen four-neuron replication is negative despite nonzero interventions and
  actual output changes. Avoid rescuing it with an unregistered layer/k search.
- **Comparison is task-dependent.** Frozen response maps and comparative probes
  have bounded predictive value. Their failures under shifts limit extrapolation;
  their successes do not identify common semantic axes.
- **Normalization changes weighting as well as representation.** Raw pooled
  errors across differently scaled series can reverse a source-average result.
  Keep source-level errors, history-standardized energy and the primary metric.

## Literature distinctions

[Superposition Is Not Necessary](https://arxiv.org/abs/2605.05151) motivates
testing actual FFN neurons; its narrow PatchTST setting does not establish
monosemanticity of pretrained residual streams.
[TSAE](https://arxiv.org/abs/2610.04925) motivates temporal and variable-aware
structure; this univariate pilot is not a reproduction of its multivariate
routing. [Matryoshka SAEs](https://proceedings.mlr.press/v267/bussmann25a.html)
motivate the explicitly labeled nested-inspired control.
[BatchTopK](https://arxiv.org/abs/2412.06410) motivates measuring actual activity
with a frozen inference threshold. Existing repository §50 work already
implements AbsTopK/BatchTopK and shows that better reconstruction need not
produce better steering; preserve and reuse that work rather than duplicate it.
