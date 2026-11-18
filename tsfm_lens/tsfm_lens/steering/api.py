"""User-facing steering of native forecasts with per-input success diagnostics."""

from dataclasses import dataclass

import numpy as np

from .controller import steer_hidden
from .solver import response_metrics
from .targets import interval_shift, seasonal_shift, quantile_shift


@dataclass
class SteeringBatch:
    """Forecasts and requested changes in ORIGINAL series units, plus diagnostics."""

    baseline: np.ndarray
    forecast: np.ndarray
    requested_change: np.ndarray
    hidden_delta: np.ndarray
    diagnostics: list
    quantiles: np.ndarray


class ForecastSteerer:
    """Control a frozen Bolt or TimesFM native head; no SAE training is required.

    Optional dictionary rows restrict the controller's available hidden span.
    To enforce a small number of atoms, select a support with solve_direction
    and pass those decoder rows here. Full-rank dictionaries are equivalent to
    dense control, and are not evidence of feature interpretability.
    """

    def __init__(self, head):
        self.head = head

    @classmethod
    def from_pretrained(cls, model='bolt', *, checkpoint=None, horizon=32, device='cpu'):
        """Load an explicitly supported model and verify its native head contract."""
        if model == 'bolt':
            from .native import BoltHead
            factory = BoltHead
        elif model == 'timesfm':
            from .timesfm import TimesFMHead
            factory = TimesFMHead
        else:
            raise ValueError('supported models are bolt and timesfm')
        kwargs = dict(horizon=horizon, device=device)
        if checkpoint is not None:
            kwargs['checkpoint'] = checkpoint
        return cls(factory(**kwargs))

    def steer(self, contexts, *, task='interval', strength=.2, period=None, phase=None,
              channel=.9, radius=.2, steps=6, ridge=1e-5, tolerance=.1,
              dictionary=None):
        """Request an interval, seasonal, or single-quantile edit.

        Interval strength is the fractional width change (+.2 wider, -.2
        narrower), holding the full median path fixed. Seasonal strength is
        signed amplitude in original series units; period must be supplied
        and phase is estimated from context alone unless supplied in radians
        at the first context sample. Quantile strength is an
        additive change in original units to `channel` only. Radius is the
        total hidden L2 edit limit divided by original hidden-state norm.

        Every edit is checked through full-model inference. Results include
        achieved/error/crossing diagnostics; the controller does not sort
        quantiles or silently substitute the output-only target. Interval
        control is not a calibration guarantee. This is adaptive inference
        control, not a fixed semantic steering vector or a weight update.
        """
        x = np.asarray(contexts, dtype=np.float32)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError('contexts must be nonempty finite [batch,context]')
        if task not in ('interval', 'seasonal', 'quantile') or not np.isfinite(strength):
            raise ValueError('need interval/seasonal/quantile task and finite strength')
        if not np.isfinite(radius) or radius <= 0 or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('radius must be positive and tolerance nonnegative')
        if (isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1
                or not np.isfinite(ridge) or ridge < 0):
            raise ValueError('steps must be a positive integer and ridge finite and nonnegative')
        if task == 'seasonal' and period is None:
            raise ValueError('seasonal steering requires an explicit period')
        periods = np.broadcast_to(np.asarray(period if period is not None else 0.), (len(x),))
        phases = [None]*len(x) if phase is None else np.broadcast_to(np.asarray(phase,dtype=float),(len(x),))
        cache = self.head.capture(x)
        deltas, targets, diagnostics, replays = [], [], [], []
        for i, (h, base) in enumerate(zip(cache['hidden'], cache['normalized'])):
            scale = float(cache['scale'][i, 0])
            if scale <= 0:
                raise ValueError('steering requires a positive native normalization scale')
            if task == 'interval':
                target = interval_shift(base, self.head.quantiles, strength)
            elif task == 'seasonal':
                target = seasonal_shift(x[i], self.head.horizon, self.head.quantiles,
                                        period=float(periods[i]), amplitude=strength/scale, phase=phases[i])
            else:
                target = quantile_shift(self.head.horizon, self.head.quantiles,
                                        channel=channel, amount=strength/scale)
            if np.linalg.norm(target) == 0:
                delta, replay, history = np.zeros_like(h), base.copy(), []
            else:
                result = steer_hidden(self.head.forward, self.head.tensor(h), target,
                                      max_norm=radius*float(np.linalg.norm(h)), steps=steps,
                                      ridge=ridge, tolerance=min(.005, tolerance), basis=dictionary)
                delta, replay, history = result.delta, result.forecast, result.history
            deltas.append(delta)
            targets.append(target)
            replays.append(replay)
            diagnostics.append(dict(history=history, relative_hidden_norm=float(
                np.linalg.norm(delta)/max(np.linalg.norm(h), 1e-30))))
        deltas = np.asarray(deltas)
        forecasts = self.head.predict_edited(x, deltas)
        scale = cache['scale'][:, None]
        location = cache['location'][:, None]
        baseline = cache['normalized']*scale+location
        replay = np.asarray(replays)*scale+location
        if not np.allclose(forecasts, replay, atol=1e-5, rtol=1e-5):
            raise RuntimeError('steered native replay differs from full-model intervention')
        requested = np.asarray(targets)*scale
        for i, record in enumerate(diagnostics):
            if np.linalg.norm(requested[i]) > 0:
                record.update(response_metrics(forecasts[i]-baseline[i], requested[i]))
            else:
                record.update(gain=None, collateral=0., relative_error=0., zero_target=True)
            record.update(achieved=record['relative_error'] <= tolerance,
                          crossing_fraction=float(np.mean(np.diff(forecasts[i],axis=-1)<0)),
                          baseline_crossing_fraction=float(np.mean(np.diff(baseline[i],axis=-1)<0)),
                          full_model_replay_max_abs=float(np.max(abs(forecasts[i]-replay[i]))))
        return SteeringBatch(baseline, forecasts, requested, deltas, diagnostics,
                             self.head.quantiles.copy())
