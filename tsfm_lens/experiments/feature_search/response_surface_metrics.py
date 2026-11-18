"""FP64 response arithmetic, separating parameter-path bending from model curvature.

All denominators retain their absolute constituents. Hidden geometry is descriptive;
known-signal forecast accuracy is a separate behavioral endpoint.
"""
from __future__ import annotations
import numpy as np


def rms(value):
    """Root mean square after promotion, before subtraction or squaring."""
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64)**2)))


def difference(a, b):
    """Subtract saved FP32 arrays in FP64."""
    return np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)


def ratio(numerator, denominator):
    """Expose dead denominators instead of hiding them behind an absolute epsilon."""
    return float(numerator/denominator) if denominator > 0 else None


def known_response(effect, signal):
    """Measure incremental forecast against the known injected future, not real truth."""
    effect, signal = np.asarray(effect, float), np.asarray(signal, float)
    energy = float(np.mean(signal**2))
    return dict(nmse=ratio(float(np.mean((effect-signal)**2)), energy),
                gain=ratio(float(np.mean(effect*signal)), energy),
                response_rms=rms(effect), signal_rms=rms(signal))


def local_response(yminus, yzero, yplus, ymid, xminus, xzero, xplus,
                   step, input_scale):
    """Centered derivatives plus chord curvature calibrated by actual input distance.

The chord midpoint is evaluated independently. A linear map on a curved frequency
trajectory has nonzero parameter curvature but zero chord curvature.
"""
    ym, y0, yp, yc = [np.asarray(v, float) for v in (yminus, yzero, yplus, ymid)]
    xm, x0, xp = [np.asarray(v, float) for v in (xminus, xzero, xplus)]
    radius = rms((xp-xm)/2)/input_scale
    left, right = rms(x0-xm)/input_scale, rms(xp-x0)/input_scale
    secant = rms((yp-ym)/2)
    path_second = rms(yp+ym-2*y0)
    chord_second = rms(yp+ym-2*yc)
    effects = rms(yp-yc)+rms(ym-yc)
    return dict(input_radius=radius, input_left=left, input_right=right,
                input_path_bend=rms(xp+xm-2*x0)/input_scale,
                derivative_rms=rms((yp-ym)/(2*step)),
                parameter_second_rms=path_second/step**2,
                distance_sensitivity=ratio(secant, radius),
                chord_curvature_per_distance2=ratio(chord_second, radius**2),
                path_second_rms=path_second, chord_second_rms=chord_second,
                chord_relative=ratio(chord_second, effects),
                individual_effect_sum=effects)


def interaction(y0, ya, yb, yab, xa, xb, input_scale):
    """Four-point mixed difference relative to each marginal and their sum."""
    y0, ya, yb, yab = [np.asarray(v, float) for v in (y0, ya, yb, yab)]
    delta = yab-ya-yb+y0
    ea, eb, mixed = rms(ya-y0), rms(yb-y0), rms(delta)
    da, db = rms(xa)/input_scale, rms(xb)/input_scale
    return dict(interaction_rms=mixed, effect_a_rms=ea, effect_b_rms=eb,
                relative_sum=ratio(mixed, ea+eb), relative_a=ratio(mixed, ea),
                relative_b=ratio(mixed, eb), input_a=da, input_b=db,
                per_input_product=ratio(mixed, da*db))


def summarize(rows, keys, fields):
    """Average dependent conditions within source before any across-source summary."""
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, {}).setdefault(row['source'], []).append(row)
    output = []
    for key, sources in grouped.items():
        item = dict(zip(keys, key)); item['n_sources'] = len(sources)
        for field in fields:
            values = [np.mean([r[field] for r in records if r[field] is not None])
                      for records in sources.values() if any(r[field] is not None for r in records)]
            item[field] = dict(mean=float(np.mean(values)), median=float(np.median(values)),
                               min=float(np.min(values)), max=float(np.max(values))) if values else None
        output.append(item)
    return output
