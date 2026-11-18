"""Evaluate configured requests on disjoint contexts and render a portable report."""

import base64
from dataclasses import asdict
from html import escape
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from matplotlib.figure import Figure

from . import DynamicSteerer, SearchConfig, SteeringModel, SteeringRequest, SteeringUnavailable


def _sample(data, seed, limit):
    contexts = np.asarray(data.contexts(), dtype=np.float32)
    indices = []
    seen = set()
    for index in np.random.default_rng(seed).permutation(len(contexts)):
        key = contexts[index].tobytes()
        if key not in seen:
            indices.append(int(index))
            seen.add(key)
        if len(indices) == limit + 4:
            break
    if len(indices) < 4:
        raise ValueError("steering needs four distinct contexts for calibration, validation and evaluation")
    n_validation = min(2, len(indices) - 3)
    return contexts, indices[:2], indices[2:2+n_validation], indices[2+n_validation:]


def _chart(result):
    fig = Figure(figsize=(7, 3), constrained_layout=True)
    ax = fig.subplots()
    forecasts = (result.baseline[0], result.baseline[0] + result.requested_change[0],
                 result.forecast[0])
    # Show the channel with the largest requested movement, including tail-only edits.
    channel = np.argmax(np.linalg.norm(result.requested_change[0], axis=0))
    for values, label, style in zip(forecasts, ("Unsteered", "Requested", "Steered"),
                                    ("-", "--", ":")):
        ax.plot(np.arange(1, len(values)+1), values[:, channel], style, label=label)
    ax.set(xlabel="Forecast step", ylabel=f"Forecast (q={result.quantiles[channel]:g})")
    ax.legend()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return '<img alt="Unsteered, requested and steered forecasts" src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def _evaluate(steerer, request, contexts, splits, ids, horizon):
    train, valid, test = splits
    parts = []
    rows = []
    failures = []
    try:
        discovery = steerer.fit(contexts[train], request, prediction_length=horizon,
                                 validation_contexts=contexts[valid])
    except (ValueError, SteeringUnavailable) as exc:
        return f"<p>Discovery unavailable for all {len(test)} evaluation contexts. {escape(str(exc))}</p>"
    for index in test:
        x = contexts[[index]]
        try:
            result = steerer.steer(x, request, prediction_length=horizon, auto_discover=False)
            replay = steerer.replay(x, result)
        except (ValueError, SteeringUnavailable) as exc:
            failures.append(f"{ids[index]}: {exc}")
            continue
        delta = result.requested_change
        norm = np.linalg.norm(delta)
        residual = np.linalg.norm(replay - result.baseline - delta)
        relative = float(residual / norm) if norm else None
        repeatable = np.allclose(replay, result.forecast, rtol=1e-5, atol=1e-6)
        verified = relative is not None and relative <= steerer.config.tolerance and repeatable
        rows.append((index, relative, bool(repeatable), bool(verified), result))
    errors = [r[1] for r in rows if r[1] is not None]
    median = f"{np.median(errors):.1%}" if errors else "n/a"
    parts.append(f"<p><b>{sum(r[3] for r in rows)} / {len(test)} verified</b> · "
                 f"Median relative error {median} · {sum(r[2] for r in rows)} replay matches · "
                 f"{len(rows)-len(errors)} zero targets · {len(failures)} unavailable</p>")
    # Show both a typical outcome and the largest miss. Never select by success alone.
    ranked = sorted((r for r in rows if r[1] is not None), key=lambda r: r[1])
    examples = [ranked[(len(ranked)-1)//2], ranked[-1]] if ranked else rows[:1]
    shown = set()
    for index, error, _, _, result in examples:
        if index not in shown:
            shown.add(index)
            label = f"relative error {error:.1%}" if error is not None else "zero target"
            parts.append(f"<figure><figcaption>{escape(str(ids[index]))} · {label}</figcaption>"
                         + _chart(result) + "</figure>")
    parts.append("<details><summary>Construction and per-context evidence</summary><pre>")
    audit = dict(discovery=discovery, failures=failures, evaluations=[
        dict(series=str(ids[index]), relative_error=error, replay_matches=match, verified=verified,
             diagnostics=result.diagnostics[0]) for index, error, match, verified, result in rows])
    parts.append(escape(json.dumps(audit, indent=2, default=str)) + "</pre></details>")
    return "".join(parts)


def run_steering_report(cfg, hub, data):
    settings = cfg.steering
    settings.validate(cfg.data)
    contexts, train, valid, test = _sample(data, cfg.run.seed, settings.max_series)
    ids = data.meta["series_id"].to_numpy()
    search = SearchConfig(seed=cfg.run.seed)
    parts = ['<!doctype html><meta charset="utf-8"><title>Steering verification</title>'
             '<style>body{max-width:920px;margin:3em auto;padding:0 1em;font:16px/1.5 system-ui}'
             'img{max-width:100%}pre{overflow:auto}summary{cursor:pointer}</style>'
             '<h1>Steering verification</h1><p>Each request starts from the ordinary forecast. '
             'Verification measures control of a requested change in forecast units. '
             'Forecast accuracy against future observations is outside this evaluation.</p>',
             '<details><summary>How this report was constructed</summary><p>Distinct float32 '
             'contexts were sampled without replacement using the run seed. Discovery measures '
             'intervention responses on calibration contexts and selects sites on validation '
             'contexts. Full-horizon context policies directly define their target by rerunning '
             'the model with the configured input change. Their agreement checks implementation. '
             'Each evaluation replays the intervention. A verified case requires a nonzero target, '
             'relative L2 error ≤ 10% across all steps and channels (unsteered error is 100%), '
             'and replay agreement within relative tolerance 1e-5 and absolute tolerance 1e-6. '
             'Charts show the lower median and worst errors on the most targeted channel.</p><pre>',
             escape(json.dumps(dict(seed=cfg.run.seed, calibration=ids[train].tolist(),
                                    validation=ids[valid].tolist(), evaluation=ids[test].tolist(),
                                    search=asdict(search)), indent=2, default=str)), '</pre></details>']
    for spec in cfg.models:
        model = SteeringModel.from_adapter(hub.get(spec.name), quantiles=settings.quantiles)
        steerer = DynamicSteerer(model, search)
        parts.append(f"<h2>{escape(spec.name)}</h2><p>{escape(model.model_id)}</p>")
        for item in settings.requests:
            request = SteeringRequest(**item)
            dose = {"delay": f"{request.delay:g} steps",
                    "context_focus": f"outside weight {request.outside_weight:g}",
                    "bounds": f"limits {request.lower} to {request.upper}"}.get(
                        request.kind, f"strength {request.strength:g}")
            parts.append(f"<h3>{escape(request.kind)} · {dose}</h3>"
                         f"<code>{escape(json.dumps(item))}</code>")
            parts.append(_evaluate(steerer, request, contexts, (train, valid, test), ids, cfg.data.horizon))
        if not cfg.run.keep_models_loaded:
            hub.release(spec.name)
        del steerer, model
    out = cfg.run_dir() / "steering" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, delete=False) as tmp:
        tmp.write("".join(parts))
    Path(tmp.name).replace(out)
