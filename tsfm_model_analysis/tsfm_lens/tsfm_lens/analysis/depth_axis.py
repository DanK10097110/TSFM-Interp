"""Candidate definitions of "how deep is this layer" (ROADMAP.md sec 18 F1).

Every cross-model figure in this repo that interpolates two models onto a
shared depth axis has, until now, used one definition: index fraction over
each model's *captured* layers (`utils.relative_depths`). That definition
compares unlike to unlike in two ways that a reader cannot see from the
figure. Chronos-T5's captured surface is its encoder, so its "relative depth
1.0" is the middle of its computation -- last encoder block, an entire
decoder still to run -- while TimesFM's 1.0 is its actual output. And the
coordinate moves when `capture_layer_stride` changes, so the same block sits
at a different depth under stride 1 than under stride 2.

There is no single correct replacement; there are several defensible ones
that disagree, which is why this module builds four and lets a figure name
which it used rather than picking one silently:

- `index` -- the status quo, retained bit-for-bit so every already-recorded
  number stays reproducible (invariant 1's spirit applied to analysis).
- `block` -- position within the *whole* stack, including blocks the stride
  skipped and surfaces the adapter declares but never captures. This is the
  axis that makes truncation visible: an encoder-only capture surface
  occupies the bottom half of the plot and the top half is empty, which is
  the honest picture.
- `compute` -- cumulative measured FLOPs, from F2's per-block counts. The
  most defensible reading of "how much of the computation has happened",
  because it handles unequal block cost rather than assuming every block is
  one unit of progress.
- `functional` -- order by a *measured* property (CKA-to-input, tuned-lens
  R^2) and align models by the value of that property rather than by
  position. Architecture-agnostic by construction, and circular if used for
  the very axis being compared, so the caller supplies the property and
  carries that constraint.

The second deliverable here matters as much as the coordinates:
`align_on_axis` refuses to compare outside the range both models actually
span. `np.interp` clamps at the edges, so interpolating a curve that spans
[0, 0.5] onto a grid spanning [0, 1] silently extends its last value across
the entire top half and reports agreement statistics over a region where one
model contributed a constant. That is not a rendering blemish; it is a
fabricated number, and it is the specific mechanism by which the old axis
could mislead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..utils import log

AXES = ("index", "block", "compute", "functional")

_LABELS = {
    "index": "index fraction over captured layers (legacy)",
    "block": "fraction of the full stack, including uncaptured blocks",
    "compute": "fraction of measured forward FLOPs completed",
    "functional": "measured property value",
}


@dataclass
class DepthAxis:
    """Depth coordinates for one model's captured layers, plus how they were made.

    `basis` and `fallback_from` exist because an axis that quietly degrades
    to a different axis is indistinguishable from the axis you asked for once
    it reaches a plot. A figure renders `label`; a report note renders
    `basis`; `fallback_from` being non-None is a warning the reader is
    entitled to see (`CLAUDE.md` sec 2.5, and sec 15 A1's lesson that a
    silent fallback to a weaker method reads as the stronger one).
    """

    coords: np.ndarray
    axis: str
    label: str
    basis: str
    fallback_from: Optional[str] = None
    covers: tuple = (0.0, 1.0)
    detail: dict = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return self.fallback_from is not None


def adapter_uncaptured_surfaces(adapter) -> dict:
    """Blocks the adapter knows about and never captures, `{surface: n_blocks}`.

    Optional on the adapter contract and defaulting to `{}`, so no existing
    adapter breaks and a model with nothing to declare says nothing. The
    canonical entry is Chronos-T5's `{"decoder": 12}`: its layer regex covers
    the encoder only, so without this declaration nothing downstream can tell
    a 12-block model from the observed half of a 24-block one.

    **Stated positional assumption**, because the return type cannot express
    it: a declared surface is counted as running *after* the matched blocks.
    That is correct for every surface any adapter in this repo declares (a
    decoder follows its encoder). An architecture with an uncaptured surface
    running *before* the captured one would need this contract widened rather
    than reinterpreted -- its blocks would shift every captured coordinate up,
    and silently treating them as trailing would place captured blocks too
    early on the axis.
    """
    fn = getattr(adapter, "uncaptured_surfaces", None)
    if fn is None:
        return {}
    surfaces = fn() or {}
    return {str(k): int(v) for k, v in surfaces.items() if int(v) > 0}


def total_stack_size(adapter) -> dict:
    """Block accounting for one model: what exists, what was captured, what was not.

    Three losses, kept apart because they have different fixes and collapsing
    them hides which one is in play. `n_blocks_uncaptured_in_captured_surface`
    is stride loss and a config change recovers it.
    `n_blocks_outside_captured_surface` is architectural -- no config recovers
    Chronos's decoder -- and is the field that catches the asymmetry
    `CLAUDE.md` sec 12 items 1-2 have described in prose since the repo was
    written.
    """
    matched = list(adapter.all_layer_names())
    captured = list(adapter.layer_names())
    surfaces = adapter_uncaptured_surfaces(adapter)
    outside = int(sum(surfaces.values()))
    return {
        "n_blocks_total": len(matched) + outside,
        "n_blocks_matched": len(matched),
        "n_blocks_captured": len(captured),
        "n_blocks_uncaptured_in_captured_surface": len(matched) - len(captured),
        "n_blocks_outside_captured_surface": outside,
        "uncaptured_surfaces": surfaces,
    }


def _index_coords(n: int) -> np.ndarray:
    """The legacy axis, reproducing `utils.relative_depths` exactly.

    Duplicated rather than imported so a future change to one cannot silently
    move the other; a test pins them equal.
    """
    if n == 1:
        return np.array([0.5])
    return np.arange(n) / (n - 1)


def _block_coords(adapter, captured_layers: list) -> tuple:
    """Each captured block's position within the full stack, in [0, 1].

    The denominator is every block the model runs, not every block this run
    captured, which is what makes the coordinate stride-invariant: a block
    keeps its coordinate whether or not the blocks around it were captured.
    """
    stack = total_stack_size(adapter)
    matched = list(adapter.all_layer_names())
    total = stack["n_blocks_total"]
    positions = []
    for name in captured_layers:
        if name not in matched:
            raise ValueError(
                f"captured layer '{name}' is not among the layer regex's matches; "
                f"depth_axis cannot place it in the stack")
        positions.append(matched.index(name))
    if total <= 1:
        return np.array([0.5] * len(positions)), stack
    return np.asarray(positions, dtype=float) / (total - 1), stack


def _compute_coords(adapter, captured_layers: list, budget: dict) -> tuple:
    """Cumulative measured FLOPs at the end of each captured block, normalized.

    The denominator is the **full forecast** when F2 measured one, not the
    capture pass. This is not a detail. Normalizing each model by its own
    capture pass gives an encoder-only model a clean 1.0 at its last captured
    block -- because its measured pass *is* its encoder -- so the model whose
    computation is *least* observed reaches the top of the chart while a
    fully-observed model honestly stops short. That inversion is exactly what
    sec 18 F4's live run exposed, and using the forecast denominator is the
    fix F1 was asked to carry.

    Returns `(coords, basis, detail)` or `(None, reason, {})` when the inputs
    are missing, so the caller can degrade loudly rather than invent a curve.
    """
    forward = (budget or {}).get("forward") or {}
    blocks = forward.get("blocks") or {}
    cumulative = blocks.get("cumulative") or {}
    if not cumulative:
        return None, "no per-block FLOPs in this run's budget artifact", {}
    missing = [n for n in captured_layers if n not in cumulative]
    if missing:
        return None, f"{len(missing)} captured blocks absent from per-block FLOPs", {}

    predict = (budget or {}).get("predict") or {}
    denom, basis = predict.get("flops"), "cumulative FLOPs over full-forecast FLOPs"
    if not denom:
        denom, basis = forward.get("flops"), "cumulative FLOPs over capture-pass FLOPs"
    if not denom:
        return None, "no total FLOP count to normalize against", {}

    coords = np.asarray([float(cumulative[n]) for n in captured_layers]) / float(denom)
    detail = {"denominator_flops": float(denom),
              "denominator_is_forecast": bool(predict.get("flops"))}
    return coords, basis, detail


def depth_axis(adapter, captured_layers: list, axis: str = "block",
               budget: Optional[dict] = None,
               functional_values: Optional[np.ndarray] = None) -> DepthAxis:
    """Depth coordinates for `captured_layers` under one of the four definitions.

    `compute` degrades to `block` when F2's per-block FLOPs are unavailable,
    and `functional` degrades to `block` when the caller supplies no measured
    property; both record `fallback_from` so the degradation travels with the
    numbers instead of vanishing into a log line.
    """
    if axis not in AXES:
        raise ValueError(f"unknown depth axis '{axis}'; expected one of {AXES}")
    n = len(captured_layers)
    if n == 0:
        raise ValueError("depth_axis needs at least one captured layer")

    if axis == "index":
        coords = _index_coords(n)
        return DepthAxis(coords, "index", _LABELS["index"],
                         "layer position over captured layers only",
                         covers=(float(coords[0]), float(coords[-1])))

    if axis == "functional":
        if functional_values is None or len(functional_values) != n:
            log.warning("depth axis 'functional' needs one measured value per captured "
                        "layer; falling back to 'block'")
            out = depth_axis(adapter, captured_layers, "block", budget)
            out.fallback_from = "functional"
            return out
        coords = np.asarray(functional_values, dtype=float)
        return DepthAxis(coords, "functional", _LABELS["functional"],
                         "a measured property supplied by the caller",
                         covers=(float(np.min(coords)), float(np.max(coords))))

    if axis == "compute":
        coords, basis, detail = _compute_coords(adapter, captured_layers, budget or {})
        if coords is None:
            log.warning("depth axis 'compute' unavailable (%s); falling back to 'block'",
                        basis)
            out = depth_axis(adapter, captured_layers, "block", budget)
            out.fallback_from = "compute"
            out.detail["compute_unavailable_because"] = basis
            return out
        return DepthAxis(coords, "compute", _LABELS["compute"], basis,
                         covers=(float(coords[0]), float(coords[-1])), detail=detail)

    coords, stack = _block_coords(adapter, captured_layers)
    return DepthAxis(coords, "block", _LABELS["block"],
                     f"position over all {stack['n_blocks_total']} blocks this model runs",
                     covers=(float(coords[0]), float(coords[-1])), detail=stack)


def depth_coordinates(adapter, captured_layers: list, axis: str = "block",
                      budget: Optional[dict] = None,
                      functional_values: Optional[np.ndarray] = None) -> np.ndarray:
    """`depth_axis(...).coords`, for callers that want only the array."""
    return depth_axis(adapter, captured_layers, axis, budget, functional_values).coords


def align_on_axis(coords_a, values_a, coords_b, values_b, n_grid: int = 21) -> dict:
    """Interpolate two models' depth curves onto the range they genuinely share.

    `np.interp` clamps rather than extrapolates, so the status quo -- build a
    grid over [0, 1] and interpolate both models onto it -- extends the
    shorter-spanning model's endpoint value flat across every grid point it
    never reached, then computes agreement statistics over that invention. On
    the `block` axis an encoder-only model spans roughly the bottom half, so
    roughly half of every such statistic was that model's last encoder block
    repeated.

    This refuses instead. The grid covers the overlap only; everything either
    model spans alone is returned as an unmatched range for a figure to shade
    and a note to name. An empty overlap returns `n_grid: 0` and empty arrays
    rather than raising, because two models with disjoint depth ranges is a
    finding about the run, not a crash.

    `values_*` may be 1-D (one curve) or 2-D `[n_layers, n_series]`, in which
    case every column is interpolated on the same grid.
    """
    ca, cb = np.asarray(coords_a, dtype=float), np.asarray(coords_b, dtype=float)
    va, vb = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if len(ca) != va.shape[0] or len(cb) != vb.shape[0]:
        raise ValueError("align_on_axis: coordinate and value lengths disagree")

    lo = max(float(ca.min()), float(cb.min()))
    hi = min(float(ca.max()), float(cb.max()))
    unmatched = {
        "a": _unmatched_ranges(ca, lo, hi),
        "b": _unmatched_ranges(cb, lo, hi),
    }
    if hi <= lo:
        return {"grid": np.array([]), "a": np.array([]), "b": np.array([]),
                "overlap": (lo, hi), "n_grid": 0, "unmatched": unmatched,
                "overlap_fraction": 0.0,
                "note": "the two models share no depth range on this axis"}

    grid = np.linspace(lo, hi, n_grid)
    span = max(float(ca.max()), float(cb.max())) - min(float(ca.min()), float(cb.min()))
    return {
        "grid": grid,
        "a": _interp_curves(grid, ca, va),
        "b": _interp_curves(grid, cb, vb),
        "overlap": (lo, hi),
        "n_grid": n_grid,
        "unmatched": unmatched,
        "overlap_fraction": float((hi - lo) / span) if span > 0 else 1.0,
        "note": "",
    }


def _unmatched_ranges(coords: np.ndarray, lo: float, hi: float) -> list:
    """Sub-ranges of one model's span that the other model never reaches."""
    out = []
    if float(coords.min()) < lo:
        out.append((float(coords.min()), lo))
    if float(coords.max()) > hi:
        out.append((hi, float(coords.max())))
    return out


def _interp_curves(grid: np.ndarray, coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate 1-D or column-wise 2-D values, sorting by coordinate first.

    Sorting matters for the `functional` axis, whose coordinate is a measured
    property that commonly *decreases* with depth; `np.interp` requires an
    increasing x and returns silent nonsense otherwise.
    """
    order = np.argsort(coords)
    x = coords[order]
    if values.ndim == 1:
        return np.interp(grid, x, values[order])
    return np.stack([np.interp(grid, x, values[order, j])
                     for j in range(values.shape[1])], axis=1)
