# Selective forecast steering

Steering is an opt-in Python API and research driver. It changes a frozen
model's internal activation to approximate an explicit desired forecast change.
The implemented tasks widen/narrow intervals while protecting the median,
add a specified seasonal prior while protecting interval widths, and change
one quantile while protecting every other forecast coordinate.

The default method needs **no SAE**. It measures the native head's Jacobian,
solves a norm-constrained least-squares problem, checks the actual nonlinear
forecast, and re-linearizes if needed. It adds to the original activation,
preserving any SAE reconstruction residual. All returned forecasts are verified
through the full model with an internal hook; they are not output-only edits.

```python
import numpy as np
from tsfm_lens.steering import ForecastSteerer

# [batch, context], float values, no future observations.
contexts = np.load("contexts.npy").astype(np.float32)
steerer = ForecastSteerer.from_pretrained("bolt", horizon=32, device="cpu")

wider = steerer.steer(contexts, task="interval", strength=0.2)
narrower = steerer.steer(contexts, task="interval", strength=-0.2)
seasonal = steerer.steer(contexts, task="seasonal", period=16, strength=0.3)
upper = steerer.steer(contexts, task="quantile", channel=0.9, strength=0.2)

print(wider.forecast.shape)       # [batch, horizon, 9]
print(wider.diagnostics[0])       # achieved, error, collateral, crossings, history
```

`strength` is a fractional interval-width change for `interval`, and an
amplitude/additive shift in original series units for `seasonal`/`quantile`.
Seasonal phase is fit from the context jointly with intercept and trend;
the period is an explicit user prior. Each quantile receives the same seasonal
addition. A context with no resolvable harmonic raises rather than inventing
a phase. `radius=0.2` caps the total hidden edit at20% of its original L2 norm;
`steps=6` bounds iterative effort. `achieved` means full-output relative RMS
error≤0.1 by default. Check this flag: difficult requests can remain incomplete.

For an entirely new seasonal prior, pass `phase=0.0` (or another angle in radians,
defined at the first context sample). This bypasses phase estimation and works
even when the context has no existing harmonic. Both `period` and `phase` may
be scalars or one value per input. The reported experiments used context-fitted
phase; explicit-phase support has a planted-answer test rather than a separate
population study.

Currently supported native interfaces are cached/pretrained Chronos-Bolt and
TimesFM2.5 (`model="timesfm"`). Use one native output patch: Bolt≤64 steps,
TimesFM within its point head's output patch; TimesFM contexts must be multiples
of32. The separate TimesFM long-horizon spread head and autoregressive extension
are outside this interface. Experiments so far use context128,horizon32,float32.

For a worked example and an explanation of what a verified result means, see [How forecast steering works](../tsfm_lens/steering/README.md).

## SAE combinations and alternative bases

```python
from tsfm_lens.steering import solve_direction, steer_hidden

# J: [horizon * quantiles, hidden], D: [SAE atoms, hidden].
selection = solve_direction(J, requested_change.ravel(), D,
                            max_features=16, ridge=1e-3, max_norm=norm_budget)
# Use the selected span for nonlinear correction without adding more atoms.
result = steer_hidden(native_suffix, original_hidden, requested_change,
                      basis=D[selection.selected], max_norm=norm_budget,
                      steps=6, ridge=1e-5)
```

Sparse selection is greedy, with a joint refit after each added atom. It is not
an optimal sparse-subset guarantee. Coefficients are signed decoder-direction
additions; they are not constrained to valid nonnegative SAE codes. Raw-neuron
bases, random dictionaries and data-derived bases use the same interface.
Dense full-rank SAE spans are mathematically equivalent to unrestricted hidden
control under the same hidden-norm penalty; their success does not demonstrate
an SAE advantage or monosemantic features.

`tsfm_lens.sae.modern` provides opt-in `AbsTopKSAE`, `BatchTopKSAE`, and a
matched-budget trainer alongside the existing TopK baseline. AbsTopK keeps
signed magnitude-selected activations. BatchTopK uses adaptive training activity
and a fixed train-calibrated inference threshold, so predictions do not depend
on evaluation batch composition. Report held-out L0: it can differ from training.
These research variants do not change the existing pipeline or its checkpoints.
Load a study dictionary with `load_steering_dictionary("runs/.../sae.pt")`
from `tsfm_lens.sae.modern`; this restores the correct encoder type and any
BatchTopK inference threshold. The pipeline's original checkpoint loader uses
a different format.

## Reproduce and extend

Run from `tsfm_lens/`, using the documented `cudaPy` environment. CPU is enough
for the short proofs. Cap torch/BLAS threads on shared machines.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
python run_steering.py --out runs/my_steering_pilot
python run_steering.py --out runs/my_signed_pilot --sae-kind abstopk
python -m tsfm_lens.experiments.steering_control --out runs/my_control_pilot
python -m tsfm_lens.experiments.steering_upstream --out runs/my_upstream_pilot
```

Each driver requires a new output directory and writes its protocol before
evaluation, exact measurements and intervention arrays, and `findings.md` after
completion. The [research log](steering_research_log.md) distills every experiment,
including failures, decisions and follow-ups. The [literature notes](steering_literature.md)
explain the motivation and scope of the variants.

## Interpretation limits

These are **context-adaptive output-control experiments**, not proof of a
universal semantic vector. The native-head edit occurs late in the model.
An upstream encoder challenge is reported separately. Interval width control
does not establish calibrated uncertainty, and an imposed seasonal prior need
not improve forecast accuracy. No future values are used to choose directions.
Quantiles are never sorted after steering; baseline and edited crossings are
reported explicitly. A full-output error measures both target gain and changes
orthogonal to the target, rather than hiding collateral behind a scalar score.

`trust_region_direction(..., ridge=0)` supplies the best possible *linearized*
response under a specified L2 budget. A nonzero error certifies only that local
linear problem, not global nonlinear impossibility. Failed sparse pursuit is
even weaker evidence: another support or optimizer might work.
