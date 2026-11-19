"""Evaluate configured requests on disjoint contexts and render a portable report."""

import base64
from dataclasses import asdict
from html import escape
from io import BytesIO
import json
from time import perf_counter
from scipy.stats import binomtest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from matplotlib.figure import Figure

from ..utils import log
from .deployment import SteeredForecaster
from .reference_checks import report_controls
from .requests import CombinedRequest, request_from_dict, requested_change
from . import DynamicSteerer, SearchConfig, SteeringModel, SteeringUnavailable


def _sample(data, seed, limit, calibration=2, validation=2):
    contexts = np.asarray(data.contexts(), dtype=np.float32)
    indices = []
    seen = set()
    for index in np.random.default_rng(seed).permutation(len(contexts)):
        key = contexts[index].tobytes()
        if key not in seen:
            indices.append(int(index))
            seen.add(key)
        if len(indices) == limit + max(2, calibration) + validation:
            break
    if len(indices) < 4:
        raise ValueError("steering needs four distinct contexts for calibration, validation and evaluation")
    n_train = min(max(2, calibration), len(indices) - 2)
    end = n_train + min(validation, len(indices) - n_train - 1)
    return contexts, indices[:n_train], indices[n_train:end], indices[end:end+limit]


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


def _evaluate(steerer, request, contexts, splits, ids, horizon, discovery, targets):
    train, valid, test = splits
    parts = []
    rows = []
    failures = []
    try:
        if discovery is None:
            discovery = steerer.fit(contexts[train], request, prediction_length=horizon,
                                 validation_contexts=contexts[valid])
    except (ValueError, SteeringUnavailable) as exc:
        failure = dict(kind=request.kind, verified=0, total=len(test), error=str(exc))
        return f"<p>Discovery unavailable for all {len(test)} evaluation contexts. {escape(str(exc))}</p>", failure
    latencies, calls, before_mae, after_mae = [], [], [], []
    before_pinball, after_pinball = [], []
    observations = []
    for index in test:
        x = contexts[[index]]
        try:
            start = perf_counter()
            initial_calls = steerer.model.calls
            result = steerer.steer(x, request, prediction_length=horizon, auto_discover=False)
            latency = perf_counter() - start
            call_count = steerer.model.calls - initial_calls
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
        channel = list(result.quantiles).index(0.5)
        before_mae.append(float(np.mean(abs(result.baseline[0, :, channel] - targets[index]))))
        after_mae.append(float(np.mean(abs(replay[0, :, channel] - targets[index]))))
        for forecast, scores in ((result.baseline[0], before_pinball), (replay[0], after_pinball)):
            error = targets[index, :, None] - forecast
            scores.append(float(np.maximum(result.quantiles * error, (result.quantiles - 1) * error).mean()))
        observations.append(dict(series=str(ids[index]), relative_error=relative, replay_matches=bool(repeatable),
                                 verified=bool(verified), seconds=latency, calls=call_count,
                                 baseline_mae=before_mae[-1], steered_mae=after_mae[-1],
                                 baseline_pinball=before_pinball[-1], steered_pinball=after_pinball[-1],
                                 baseline=result.baseline[0].tolist(), forecast=result.forecast[0].tolist(),
                                 requested_change=delta[0].tolist(),
                                 replay=replay[0].tolist(), future=targets[index].tolist(),
                                 diagnostics=result.diagnostics[0]))
        if isinstance(request, CombinedRequest):
            components = [requested_change(result.baseline[0], x[0], result.quantiles, term)
                          for term in request.terms]
            matrix = np.column_stack([component.ravel() for component in components])
            gains, _, rank, _ = np.linalg.lstsq(matrix, (replay - result.baseline).ravel(), rcond=None)
            observations[-1].update(component_targets=[c.tolist() for c in components],
                                    component_gains=gains.tolist() if rank == len(components) else None)
        latencies.append(latency)
        calls.append(call_count)
        rows.append((index, relative, bool(repeatable), bool(verified), result))
    errors = [r[1] for r in rows if r[1] is not None]
    median = f"{np.median(errors):.1%}" if errors else "n/a"
    parts.append(f"<p><b>{sum(r[3] for r in rows)} / {len(test)} verified</b> · "
                 f"Median relative error {median} · {sum(r[2] for r in rows)} replay matches · "
                 f"{len(rows)-len(errors)} zero targets · {len(failures)} unavailable</p>")
    count = sum(r[3] for r in rows)
    interval = binomtest(count, len(test)).proportion_ci(method="wilson")
    record = dict(kind=request.kind, verified=count, total=len(test),
                  ci=[interval.low, interval.high], improved=sum(e < 1 for e in errors),
                  median_relative_error=float(np.median(errors)) if errors else None,
                  p90_relative_error=float(np.quantile(errors, .9)) if errors else None,
                  replay_matches=sum(r[2] for r in rows), completed=len(rows),
                  zero_targets=len(rows)-len(errors), unavailable=len(failures))
    missed = [row for row in observations if row["relative_error"] is not None and not row["verified"]]
    at_radius = sum(1 for row in missed if row["diagnostics"].get("radius") is not None and
                    np.isclose(row["diagnostics"]["coefficient_norm"], row["diagnostics"]["radius"]))
    record["misses_at_radius"] = at_radius
    if missed:
        parts.append(f"<p>{at_radius} of {len(missed)} nonzero misses used the full edit radius. "
                     "The construction details give each edit norm, allowed radius and local solver residual.</p>")
    if isinstance(request, CombinedRequest):
        gains = [row["component_gains"] for row in observations if row["component_gains"] is not None]
        if gains:
            record["median_component_gains"] = np.median(gains, axis=0).tolist()
            labels = [f"{term.kind} {gain:.3g}" for term, gain in zip(request.terms, record["median_component_gains"])]
            parts.append("<p>Median fitted gain for each component (1 means the requested amount) "
                         + escape(", ".join(labels)) + f". Estimated on {len(gains)} contexts with independent "
                         "target patterns. The full-forecast error also counts off-target movement.</p>")
        else:
            parts.append("<p>The component targets could not be distinguished on these contexts.</p>")
    if rows:
        diffs = np.column_stack((np.subtract(after_mae, before_mae),
                                 np.subtract(after_pinball, before_pinball)))
        rng = np.random.default_rng(steerer.config.seed)
        boot = np.array([diffs[rng.integers(len(rows), size=len(rows))].mean(axis=0) for _ in range(2000)])
        lower, upper = np.quantile(boot, [.025, .975], axis=0)
        record.update(median_seconds=float(np.median(latencies)), median_calls=float(np.median(calls)),
                      baseline_mae=float(np.mean(before_mae)), steered_mae=float(np.mean(after_mae)),
                      mae_delta_ci=[lower[0], upper[0]], baseline_pinball=float(np.mean(before_pinball)),
                      steered_pinball=float(np.mean(after_pinball)), pinball_delta_ci=[lower[1], upper[1]])
        parts.append(f"<p>Median steering time {record['median_seconds']:.3f} s · "
                     f"median predictor calls {record['median_calls']:g}. Held-out median forecast MAE "
                     f"{record['baseline_mae']:.4g} → {record['steered_mae']:.4g}. "
                     f"Paired 95% bootstrap interval for the MAE change "
                     f"[{lower[0]:.3g}, {upper[0]:.3g}]. Mean pinball loss across all quantiles "
                     f"{record['baseline_pinball']:.4g} → {record['steered_pinball']:.4g}, "
                     f"paired 95% change interval [{lower[1]:.3g}, {upper[1]:.3g}].</p>")
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
    audit = dict(statistics=record, discovery=discovery, failures=failures,
                 quantiles=steerer.model.quantiles.tolist(), evaluations=observations)
    parts.append(escape(json.dumps(audit, indent=2, default=str)) + "</pre></details>")
    return "".join(parts), record


def run_steering_report(cfg, hub, data):
    settings = cfg.steering
    settings.validate(cfg.data)
    search = SearchConfig(seed=cfg.run.seed, **settings.search)
    contexts, train, valid, test = _sample(data, cfg.run.seed, settings.max_series,
                                            search.calibration_samples, search.validation_samples)
    ids = data.meta["series_id"].to_numpy()
    parts = ['<!doctype html><meta charset="utf-8"><title>Steering verification</title>'
             '<style>table{border-collapse:collapse}th,td{text-align:left;padding:.35em .65em;border-bottom:1px solid #cad2db}th{background:#e9f1f5}body{max-width:980px;margin:3em auto;padding:0 1em;font:16px/1.5 system-ui}'
             'img{max-width:100%}pre{overflow:auto}summary{cursor:pointer}</style>'
             '<h1>Steering verification</h1><p>Each request starts from the ordinary forecast. '
             'Verification measures control of a requested change in forecast units. '
             'Observed futures are used only to score forecast errors (median MAE and quantile pinball loss).</p>',
             '<details><summary>How this report was constructed</summary><p>Distinct float32 '
             'contexts were sampled without replacement using the run seed. Discovery measures '
             'intervention responses on calibration contexts and selects sites on validation '
             'contexts. Full-horizon context policies directly define their target by rerunning '
             'the model with the configured input change. Their agreement checks implementation. '
             'Each evaluation replays the intervention. A verified case requires a nonzero target, '
             'relative L2 error within the configured tolerance across all steps and channels (unsteered error is 100%), '
             'and replay agreement within relative tolerance 1e-5 and absolute tolerance 1e-6. '
             'Charts show the lower median and worst errors on the most targeted channel.</p><pre>',
             escape(json.dumps(dict(seed=cfg.run.seed, calibration=ids[train].tolist(),
                                    validation=ids[valid].tolist(), evaluation=ids[test].tolist(),
                                    search=asdict(search)), indent=2, default=str)), '</pre></details>']
    overview = []
    requests = [request_from_dict(item) for item in settings.requests]
    for model_index, spec in enumerate(cfg.models):
        model = SteeringModel.from_adapter(hub.get(spec.name), quantiles=settings.quantiles)
        steerer = DynamicSteerer(model, search)
        parts.append(f"<h2>{escape(spec.name)}</h2><p>{escape(model.model_id)}</p>")
        try:
            discoveries = steerer.fit(contexts[train], requests, prediction_length=cfg.data.horizon,
                                      validation_contexts=contexts[valid])
        except (ValueError, SteeringUnavailable):
            discoveries = {}  # Evaluate independently if any request prevented a shared fit.
        for request_index, item in enumerate(settings.requests):
            log.info("steering: %s · %s", spec.name, item["kind"])
            request = request_from_dict(item)
            if isinstance(request, CombinedRequest):
                dose = " + ".join(term.kind for term in request.terms)
            else:
                dose = {"delay": f"{request.delay:g} steps",
                        "context_focus": f"outside weight {request.outside_weight:g}",
                        "bounds": f"limits {request.lower} to {request.upper}"}.get(
                            request.kind, f"strength {request.strength:g}")
            parts.append(f'<h3 id="m{model_index}-r{request_index}">' f"{escape(request.kind)} · {dose}</h3>"
                         f"<code>{escape(json.dumps(item))}</code>")
            html, stats = _evaluate(steerer, request, contexts, (train, valid, test), ids, cfg.data.horizon,
                                    discoveries.get(request.key(cfg.data.horizon)), data.targets())
            parts.append(html)
            overview.append((spec.name, item, stats, f"m{model_index}-r{request_index}"))
            if stats.get("completed", 0):
                predictor = SteeredForecaster(steerer, request, spec, cfg.data, cfg.run.dtype, cfg.run.seed)
                try:
                    with TemporaryDirectory() as tmp:
                        archive = predictor.export(Path(tmp) / "model.zip")
                        encoded = base64.b64encode(archive.read_bytes()).decode()
                except (OSError, ValueError, SteeringUnavailable) as exc:
                    parts.append(f"<p>Export unavailable. {escape(str(exc))}</p>")
                    continue
                filename = f"model-{model_index+1}-{request.kind}-{request_index+1}.zip"
                parts.append(f'<p><a download="{filename}" href="data:application/zip;base64,{encoded}">'
                             'Export this steering profile</a></p>')
        if not cfg.run.keep_models_loaded:
            hub.release(spec.name)
        del steerer, model
    table = ["<h2>Held-out results</h2><p>Success intervals are 95% Wilson intervals per request, "
             "assuming independent contexts. They exclude uncertainty from refitting and are unadjusted "
             "for multiple comparisons. Improved counts nonzero targets closer than the ordinary forecast. "
             "Latency excludes fitting and verification replay. Forecast loss intervals resample paired series. Smaller MAE and pinball loss are better.</p>"
             "<table><tr><th>Model</th><th>Request</th><th>Verified</th><th>95% interval</th>"
             "<th>Improved</th><th>90th percentile relative error</th></tr>"]
    for model, item, record, anchor in overview:
        lo, hi = binomtest(record["verified"], record["total"]).proportion_ci(method="wilson")
        tail = record.get("p90_relative_error")
        label = escape(", ".join(str(v) if k == "kind" else f"{k}={v}" for k,v in item.items()))
        table.append(f'<tr><td>{escape(model)}</td><td><a href="#{anchor}">{label}</a></td>'
                     f"<td>{record['verified']}/{record['total']}</td><td>{lo:.0%}–{hi:.0%}</td>"
                     f"<td>{record.get('improved', 0)}</td><td>{format(tail,'.1%') if tail is not None else 'n/a'}</td></tr>")
    parts.insert(1, report_controls() + "".join(table) + "</table>")
    evidence = json.dumps([dict(model=m, request=r, **stats) for m,r,stats,_ in overview], allow_nan=False)
    parts.append('<script type="application/json" id="steering-results">' +
                 evidence.replace("<", "\\u003c") + "</script>")
    out = cfg.run_dir() / "steering" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, delete=False) as tmp:
        tmp.write("".join(parts))
    Path(tmp.name).replace(out)
