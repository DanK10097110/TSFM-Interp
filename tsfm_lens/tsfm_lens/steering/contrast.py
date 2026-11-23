"""Paired activation contrasts at one researcher-selected hidden site."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .model import SteeringModel, SteeringUnavailable, _leaves, _replace


def _run(model, row, horizon, layer, token, edit=None):
    """Capture or edit one vector on one execution of a module."""
    if model.module is None or any(m.training for m in model.module.modules()):
        raise SteeringUnavailable("contrast steering requires a frozen evaluation-mode module")
    captured = []

    def hook(module, args, kwargs):
        if captured:
            raise SteeringUnavailable("choose a layer that executes once per forecast")
        tree = {"args": args, "kwargs": kwargs}
        for path, value in _leaves(tree):
            if value.is_floating_point() and value.ndim in (2, 3) and value.shape[0] == 1:
                break
        else:
            raise SteeringUnavailable("expected a floating [1,width] or [1,tokens,width] layer input")
        if value.ndim == 2 and token not in (0, -1):
            raise ValueError("a single-vector layer has only one position")
        index = (0, token) if value.ndim == 3 else (0,)
        state = value[index].detach().float().cpu().numpy().astype(float)
        captured.append(state)
        if edit is not None:
            changed = value.clone()
            changed[index] += torch.as_tensor(edit(state), device=value.device, dtype=value.dtype)
            tree = _replace(tree, path, changed)
            return tree["args"], tree["kwargs"]

    with model.lock, torch.no_grad():
        handle = model.module.get_submodule(layer).register_forward_pre_hook(hook, with_kwargs=True)
        try:
            forecast = model.forecast(row[None], horizon)[0]
        finally:
            handle.remove()
    if not captured:
        raise SteeringUnavailable("selected layer did not execute")
    return forecast, captured[0]


@dataclass(frozen=True)
class ActivationContrast:
    """Average positive-minus-negative hidden state, bound to a model and input geometry."""
    model: SteeringModel
    layer: str
    token: int
    horizon: int
    context_length: int
    difference: np.ndarray
    midpoint: np.ndarray
    pairs: int
    fingerprint: str

    @classmethod
    def fit(cls, model, positive, negative, *, layer, prediction_length, token=-1):
        """One matched pair per row. Order fixes the sign of the learned direction."""
        positive, negative = (np.asarray(a, dtype=np.float32) for a in (positive, negative))
        if (positive.ndim != 2 or positive.shape != negative.shape or len(positive) < 2
                or not positive.shape[1] or not all(np.isfinite(a).all() for a in (positive, negative))):
            raise ValueError("supply at least two matched finite [pair, time] arrays")
        with model.lock:
            means = [np.mean([_run(model, x, prediction_length, layer, token)[1] for x in group], axis=0)
                     for group in (positive, negative)]
            difference = means[0] - means[1]
            if not np.isfinite(difference).all() or np.linalg.norm(difference) <= 1e-12:
                raise SteeringUnavailable("no measurable mean activation contrast at this site")
            return cls(model, layer, token, prediction_length, positive.shape[1], difference,
                       (means[0] + means[1]) / 2, len(positive), model.fingerprint())

    def _contexts(self, contexts):
        x = np.atleast_2d(np.asarray(contexts, dtype=np.float32))
        if x.ndim != 2 or x.shape[1] != self.context_length or not len(x) or not np.isfinite(x).all():
            raise ValueError("input must match fitted context length and be finite")
        return x

    def score(self, contexts):
        """Centered activation projection. The fitted group means differ by one score unit."""
        scale = self.difference @ self.difference
        return np.array([(_run(self.model, x, self.horizon, self.layer, self.token)[1] - self.midpoint)
                         @ self.difference / scale for x in self._contexts(contexts)])

    def predict(self, contexts, *, strength=1., mode="add"):
        """Add strength times the fixed difference, or amplify the centered projection."""
        x = self._contexts(contexts)
        if mode not in ("add", "amplify") or not np.isfinite(strength):
            raise ValueError("choose add or amplify and a finite strength")
        axis = self.difference / np.linalg.norm(self.difference)

        def edit(state):
            if state.shape != axis.shape:
                raise SteeringUnavailable("hidden geometry changed")
            if mode == "add":
                return strength * self.difference
            return strength * ((state - self.midpoint) @ axis) * axis

        return np.stack([_run(self.model, row, self.horizon, self.layer, self.token, edit)[0] for row in x])

    def save(self, path):
        metadata = {k: getattr(self, k) for k in ("layer", "token", "horizon", "context_length", "pairs", "fingerprint")}
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata), difference=self.difference, midpoint=self.midpoint)

    @classmethod
    def load(cls, path, model):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            if meta["fingerprint"] != model.fingerprint():
                raise SteeringUnavailable("checkpoint or quantiles differ from the fitted contrast")
            delta, center = data["difference"], data["midpoint"]
            if (delta.ndim != 1 or delta.shape != center.shape or not np.isfinite(center).all()
                    or not np.isfinite(delta).all() or np.linalg.norm(delta) <= 1e-12):
                raise ValueError("invalid saved activation contrast")
            return cls(model=model, difference=delta, midpoint=center, **meta)
