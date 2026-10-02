"""Error-preserving feature edits and their held-out scoring (ROADMAP.md sec 39, R0).

Why this module exists. The forecast-repair study (sec 39) asks whether editing named SAE
features, with the edit chosen on training series, improves forecasts on held-out ones.
Everything it claims rests on one primitive, the edit

    h <- h + sum_f (g_f - 1) * z_f(h) * w_f

at one layer, where `z_f(h)` is feature `f`'s code on the token's own activation `h` and
`w_f` its decoder row. The SAE reconstruction never replaces `h` (sec 39.2 rule 2): the
battery in `sae/response.py` patches the reconstruction in and measures against the
reconstruction, which absorbs the SAE's error into every comparison, so an edit applied
that way would change the forecast even at `g = 1`. Here an edit with every gain 1 is the
identity BY CONSTRUCTION (no feature is touched, the clean tokens are written back), and
`free_controls` measures that it changes the forecast by exactly 0.0, and that a real edit
at the same layer moves it by a nonzero amount (the sec 8 free controls: a patch the head
never reads gives a clean flat zero).

What it inherits. The intervention is `extraction/hooks.py::token_patch`, the repo's single
intervention primitive, written back at `predict()` precision from a clean cache captured
with autocast off (`capture_raw_tokens`, sec 11.49). Rows per forward never exceed the
model's `batch_size`, because `token_patch` raises if the model chunks the batch.

What it adds. A series-level train/test split by `utils.sample_rows` (stratified by
family, never a head slice; the series is the resampling unit, invariant 2); a gain chosen
on the training series only; and the test score of that edit, as the per-series change in
MASE (edited minus unedited, so negative is better) with a series-bootstrap CI, on named
groups of test series: where the feature fires, where it fires weakly, and every series
(the side effects of rule 6). The same numbers on the training series are returned too,
labelled `in_sample`, and are diagnostics only.

Evidence class: behavioral, held out. A held-out improvement of an edit chosen on train is
the claim; nothing here is causal on its own beyond what the edit's own forward shows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..extraction.extract import capture_raw_tokens
from ..extraction.hooks import token_patch
from ..utils import sample_rows
from .stats import mase, mean_ci

GAIN_GRID = (0.0, 0.5, 1.5, 2.0)
ZERO_TOL = 0.0
REACH_MIN_RELATIVE = 1e-5


def edit_replacement(clean_tokens: torch.Tensor, sae, device, gains: dict) -> torch.Tensor:
    """`h + sum_f (g_f - 1) z_f(h) w_f` for every token of `clean_tokens` `[B, T, D]`.

    `gains` maps feature index to gain; a gain of exactly 1.0 touches nothing, so an
    all-ones edit returns an exact copy of the clean tokens. The residual stream keeps
    the SAE's reconstruction error: only the change in the edited features' own decoded
    contribution is added.
    """
    active = {int(f): float(g) for f, g in gains.items() if float(g) != 1.0}
    if not active:
        return clean_tokens.clone()
    b, t, d = clean_tokens.shape
    flat = clean_tokens.reshape(-1, d).to(device)
    with torch.no_grad():
        codes = sae.encode(flat)
        delta = torch.zeros_like(flat)
        for f, g in active.items():
            delta = delta + (g - 1.0) * codes[:, f:f + 1] * sae.W_dec[f].detach()[None, :]
    return (flat + delta).reshape(b, t, d).cpu()


def predict_point(adapter, contexts: np.ndarray, horizon: int, quantiles: list, seed: int,
                  layer: Optional[str] = None, tokens: Optional[torch.Tensor] = None) -> np.ndarray:
    """Seeded point forecast of one batch, unpatched or with `tokens` written at `layer`.

    The seed is set before the forward in both arms, so a sampled model compares like with
    like (sec 11.50). `contexts` must hold at most `batch_size` rows.
    """
    if len(contexts) > adapter.cfg.batch_size:
        raise ValueError(f"predict_point: {len(contexts)} rows exceed batch_size "
                         f"{adapter.cfg.batch_size}; token_patch needs the batch unchunked")
    if tokens is None:
        torch.manual_seed(seed)
        return np.asarray(adapter.predict(contexts, horizon, quantiles)["point"], dtype=np.float64)
    with token_patch(adapter.module, layer, adapter.token_slice, tokens):
        torch.manual_seed(seed)
        return np.asarray(adapter.predict(contexts, horizon, quantiles)["point"], dtype=np.float64)


def predict_gains(adapter, layer: str, sae, device, contexts: np.ndarray, gains: dict,
                  horizon: int, quantiles: list, seed: int) -> tuple:
    """`(baseline, edited)` forecasts for `contexts`, in chunks of at most `batch_size` rows."""
    base, edited = [], []
    step = int(adapter.cfg.batch_size)
    for i in range(0, len(contexts), step):
        chunk = contexts[i:i + step]
        clean = capture_raw_tokens(adapter, chunk, [layer])[layer]
        base.append(predict_point(adapter, chunk, horizon, quantiles, seed))
        edited.append(predict_point(adapter, chunk, horizon, quantiles, seed, layer,
                                    edit_replacement(clean, sae, device, gains)))
    return np.concatenate(base), np.concatenate(edited)


def delta_mase(baseline: np.ndarray, edited: np.ndarray, targets: np.ndarray,
               contexts: np.ndarray) -> np.ndarray:
    """Per-series MASE change, edited minus baseline (negative is an improvement)."""
    return mase(edited, targets, contexts) - mase(baseline, targets, contexts)


def free_controls(adapter, layer: str, sae, device, contexts: np.ndarray, feature: int,
                  horizon: int, quantiles: list, seed: int, require_reach: bool = True) -> dict:
    """The two free controls of sec 8, measured on `contexts`.

    `identity_max_abs`: an edit with every gain 1 must change the forecast by exactly 0.0;
    always asserted. `reach_mean_abs`: zeroing `feature` must change it by a nonzero amount
    (a layer the head does not read gives a clean flat zero, sec 11.42). "Nonzero" is relative:
    the mean change must exceed `REACH_MIN_RELATIVE` of the forecast's mean absolute value,
    because an absolute epsilon cannot tell "no effect" from float rounding (sec 8). The reach control is
    meant for a feature known to be read, once per layer, so it is asserted only with
    `require_reach`; for an arbitrary feature, which may legitimately be inert, it is only
    recorded. Raises `RuntimeError` when an asserted control fails, because a failed control
    means every number downstream is unreadable.
    """
    base, same = predict_gains(adapter, layer, sae, device, contexts, {int(feature): 1.0},
                               horizon, quantiles, seed)
    identity = float(np.abs(same - base).max())
    if identity != ZERO_TOL:
        raise RuntimeError(f"free control failed: an all-ones edit moved the forecast by {identity!r} "
                           f"(must be exactly 0.0)")
    base, zeroed = predict_gains(adapter, layer, sae, device, contexts, {int(feature): 0.0},
                                 horizon, quantiles, seed)
    reach = float(np.abs(zeroed - base).mean())
    scale = float(np.abs(base).mean())
    if require_reach and not reach > REACH_MIN_RELATIVE * scale:
        raise RuntimeError(f"free control failed: zeroing feature {feature} at {layer} moved the forecast "
                           f"by {reach:.3g}, under {REACH_MIN_RELATIVE:g} of its scale; the head does not read this layer through this feature")
    return {"identity_max_abs": identity, "reach_mean_abs": reach,
            "forecast_scale": scale,
            "reach_relative": reach / scale if scale > 0 else None}


def split_series(n: int, families: np.ndarray, seed: int, train_fraction: float = 0.5) -> tuple:
    """`(train_rows, test_rows)`: a disjoint, family-stratified split of the series."""
    train = sample_rows(n, int(round(train_fraction * n)), int(seed), strata=families)
    test = np.setdiff1d(np.arange(n), train)
    return train, test


def group_scores(delta: np.ndarray, masks: dict, n_boot: int, seed: int) -> dict:
    """Mean per-series change, s.d. and series-bootstrap CI for each named boolean mask."""
    out = {}
    for name, mask in masks.items():
        v = np.asarray(delta, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        if v.size < 3:
            out[name] = {"n": int(v.size), "scorable": False,
                         "reason": "fewer than 3 series in this group"}
            continue
        ci = mean_ci(v, n_boot=n_boot, seed=seed)
        out[name] = {"n": int(v.size), "scorable": True, "mean": ci["value"], "sd": float(v.std(ddof=1)),
                     "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                     "ci_excludes_zero": bool(ci["lo"] > 0.0 or ci["hi"] < 0.0),
                     "resample_unit": ci["resample_unit"]}
    return out


def choose_gain(train_delta_by_gain: dict) -> float:
    """The gain with the lowest mean training change in MASE (ties go to the smaller gain)."""
    return float(min(sorted(train_delta_by_gain),
                     key=lambda g: float(np.mean(train_delta_by_gain[g]))))


def edit_feature_held_out(adapter, layer: str, sae, device, data, activations: np.ndarray,
                          feature: int, train_rows: np.ndarray, test_rows: np.ndarray,
                          horizon: int, quantiles: list, seed: int, select_on: str = "firing",
                          strong_split: Optional[float] = None, gain_grid: tuple = GAIN_GRID,
                          n_boot: int = 1000) -> dict:
    """Choose one feature's gain on the training series, score it on the test series.

    `activations` is the persisted series-level `[n_series, dict_size]` code matrix, used
    only to name the groups: `firing` is a series on which the feature's code is positive,
    `strong` and `weak` split the firing series at `strong_split` (default: the median
    positive code on the TRAINING series, so no test information chooses a group), and
    `all` is every series in the split. The gain is chosen among `gain_grid` on the
    training series of group `select_on` only. Returns the chosen gain, the training
    change for every gain, and the chosen edit's train (`in_sample`) and test scores on
    every group, the test score of the plain zero-edit (gain 0) whatever gain was chosen, and the
    free controls on a batch of test series.
    """
    acts = np.asarray(activations, dtype=np.float64)[:, int(feature)]
    train_pos = acts[train_rows][acts[train_rows] > 0.0]
    split = float(np.median(train_pos)) if strong_split is None and train_pos.size else strong_split

    def masks(rows):
        a = acts[rows]
        firing = a > 0.0
        strong = firing & (a >= split) if split is not None else np.zeros_like(firing)
        weak = firing & ~strong
        return {"firing": firing, "strong": strong, "weak": weak, "all": np.ones_like(firing)}

    contexts, targets = data.contexts(), data.targets()

    def score_rows(rows, gains):
        base, edited = predict_gains(adapter, layer, sae, device, contexts[rows], gains, horizon,
                                     quantiles, seed)
        return delta_mase(base, edited, targets[rows], contexts[rows])

    train_masks = masks(train_rows)
    sel = train_masks[select_on]
    train_by_gain = {}
    for g in gain_grid:
        d = score_rows(train_rows[sel], {int(feature): g}) if sel.any() else np.zeros(0)
        train_by_gain[float(g)] = d
    chosen = choose_gain(train_by_gain) if sel.any() else 1.0
    train_delta = score_rows(train_rows, {int(feature): chosen})
    test_delta = score_rows(test_rows, {int(feature): chosen})
    test_zero = test_delta if chosen == 0.0 else score_rows(test_rows, {int(feature): 0.0})
    test_fire = test_rows[masks(test_rows)["firing"]]
    control_rows = (test_fire if test_fire.size else test_rows)[:int(adapter.cfg.batch_size)]
    controls = free_controls(adapter, layer, sae, device, contexts[control_rows], int(feature),
                             horizon, quantiles, seed, require_reach=False)
    return {
        "feature": int(feature), "chosen_gain": chosen, "select_on": select_on,
        "strong_split": split, "n_train": int(len(train_rows)), "n_test": int(len(test_rows)),
        "train_change_by_gain": {str(g): float(np.mean(v)) if len(v) else None
                                 for g, v in train_by_gain.items()},
        "in_sample": group_scores(train_delta, train_masks, n_boot, seed),
        "test": group_scores(test_delta, masks(test_rows), n_boot, seed + 1),
        "test_at_gain_zero": group_scores(test_zero, masks(test_rows), n_boot, seed + 1),
        "free_controls": controls,
        "evidence_class": "behavioral, held out (the test series never informed the gain)",
    }
