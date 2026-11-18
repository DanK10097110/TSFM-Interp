# Literature informing the steering experiments

Checked 2026-11-16. These are design motivations, not replications or evidence
that the same result holds in our checkpoints. Empirical results are in
[the experiment log](steering_research_log.md).

- [Finding the Zeitgeist in Time Series Foundation Models](https://openreview.net/pdf?id=7SP3orvHPy)
  explores SAE features aligned with temporal patterns. This motivates trying
  sparse dictionaries, not assuming native TSFM neurons are monosemantic.
- [Dissecting Chronos](https://arxiv.org/abs/2603.10071) studies TopK SAEs across
  Chronos-T5-Large depth and feature ablations. Its emphasis on different causal
  roles across depth motivates testing an upstream site. Abrupt forecast
  degradation after ablation does not itself establish selective steering.
- [TSAE: Structured Sparse Autoencoders for Interpreting Time-Series Forecasting
  Models](https://arxiv.org/abs/2610.04925) adds variable structure, conditional
  temporal continuity, gated encoding and forecast-consistency training.
  Its distinctions between fidelity, coherence and physical calibration are
  directly useful here. Forecast-consistency is a promising follow-up; variable
  routing and temporal-code continuity do not apply directly to the single
  native-head vector in our first experiment.
- [TimeSAE: Sparse Decoding for Faithful Explanations of Black-Box Time Series
  Models](https://arxiv.org/abs/2601.09776) motivates checking explanations under
  distribution shift. This is distinct from the similarly named MOMENT workshop
  paper, [TimeSAE: Mechanistic Interpretability for TSFMs](https://openreview.net/pdf/5ff214e8d9bea3b66d1e2796a2e37ff01c3dd83c.pdf).
  The latter's indexed abstract reports linear steering; full PDF access was
  browser-blocked, so no detailed methodological claims are based on it here.
- [AbsTopK](https://arxiv.org/abs/2510.00404) keeps signed, largest-magnitude
  activations. Our hypothesis: seasonal polarity and rising/falling directions
  may benefit from a shared signed axis. This is an inference to test, not a
  conclusion from that paper about TSFMs.
- [BatchTopK](https://arxiv.org/abs/2412.06410) allocates sparsity over a training
  batch. Our inference uses a fixed train-calibrated threshold, allowing a
  meaningful batch-independence check and exposing test-time sparsity drift.
- [Matryoshka SAEs](https://arxiv.org/abs/2503.17547) learn nested dictionaries to
  retain coarse and fine concepts. Parked pending evidence that multi-scale
  feature absorption, rather than output response geometry, limits these tasks.
- [SAEBench](https://arxiv.org/abs/2503.09532) finds that proxy improvements need
  not translate into practical gains. Accordingly we report reconstruction,
  actual activity, full-output steering error and matched non-SAE controls
  separately.
- [Improving Steering Vectors by Targeting SAE Features](https://arxiv.org/abs/2411.02193)
  motivates learning intervention effects and minimizing collateral. Our method
  instead targets the TSFM's full quantile/time output directly; this makes it
  a task-specific control method, not a reproduction of SAE-Targeted Steering.

The sources inspected do not justify a general assumption that native TSFM
neurons are monosemantic. Raw coordinates, random dictionaries and dense
directions are therefore necessary empirical baselines. No novelty priority
claim is made for weighted cancellation, Jacobian control or these SAE variants.
