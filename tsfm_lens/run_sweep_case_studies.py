"""ROADMAP.md §7 Phase 3, bullet 4: verbose-mode case studies for a
controlled parameter sweep (`run_parameter_sweep.py`).

Reads an existing `param_sweep_*.json`, picks a handful of "interesting"
sweep values (`report/sweep_case_studies.py::select_case_study_values` --
the two extremes plus any anomaly `report/bias_card.py` flagged), generates
one fresh example series per selected value, runs the real models' forecasts
on it, and renders a narrated context/true-continuation/forecast comparison
per value. Illustrative, not statistical (`CLAUDE.md` §2.6) -- the aggregate
dose-response numbers already in `ROADMAP.md` §7 are the evidence; this is a
concrete look into what those numbers mean for one actual series per point.

Per-layer skip-lens curves at each case-study series (mirroring
`analysis/exemplars.py`) are not built here -- that needs the full
extraction/lens machinery per series, a materially larger undertaking than
`adapter.predict`, and is a named follow-up.

Example:
    python run_sweep_case_studies.py --run runs/medium_run_chronos_base \\
        --sweep-json runs/param_sweep_seasonal_period.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsfm_lens.analysis.parameter_sweep import generate_sweep_data
from tsfm_lens.config import load_config
from tsfm_lens.models import ModelHub
from tsfm_lens.report.bias_card import summarize_param_sweep
from tsfm_lens.report.sweep_case_studies import render_case_studies_html, select_case_study_values
from tsfm_lens.utils import load_json, log, resolve_device, resolve_dtype, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md §7 Phase 3: narrated case studies for a parameter sweep")
    parser.add_argument("--run", required=True, help="an existing run directory (model configs only)")
    parser.add_argument("--sweep-json", required=True, help="an existing param_sweep_*.json")
    parser.add_argument("--max-points", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None, help="default: the base run's own run.seed, offset")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run)
    base_cfg = load_config(run_dir / "config_resolved.yaml")
    seed = (args.seed if args.seed is not None else base_cfg.run.seed) + 500  # distinct from the sweep's own draws
    set_seed(seed)
    device = resolve_device(base_cfg.run.device)
    dtype = resolve_dtype(base_cfg.run.dtype, device)
    hub = ModelHub(base_cfg.models, base_cfg.data, device, dtype)

    sweep = load_json(Path(args.sweep_json))
    param, values = sweep["param"], sweep["values"]
    anomalous_values = []
    try:
        summary = summarize_param_sweep(sweep)
        anomalous_values = [p["value"] for p in summary["points"] if p["anomalous"]]
    except ValueError as e:
        log.warning("could not compute anomalies (%s); selecting extremes only", e)

    selected = select_case_study_values(values, anomalous_values, args.max_points)
    log.info("case studies for param=%s: selected values %s (anomalous: %s)",
              param, selected, anomalous_values)

    data = generate_sweep_data(param, selected, n_per_point=1, context_len=sweep["context_len"],
                               horizon=sweep["horizon"], seed=seed)
    contexts, targets = data.contexts(), data.targets()

    forecasts = {}
    for mcfg in base_cfg.models:
        adapter = hub.get(mcfg.name)
        adapter.ensure_loaded()
        out = adapter.predict(contexts, sweep["horizon"], [0.5])
        forecasts[mcfg.name] = out["point"]
        if not base_cfg.run.keep_models_loaded:
            hub.release(mcfg.name)

    entries = []
    for row, value in enumerate(selected):
        label = f"{param}={value:g}"
        mase_by_model = {name: rows[label]["value"] for name, rows in sweep["models"].items()
                         if label in rows}
        mase_n = next(iter(sweep["models"].values())).get(label, {}).get("n")
        entries.append({
            "value": value, "anomalous": value in anomalous_values,
            "context": contexts[row].astype(float), "target": targets[row].astype(float),
            "forecasts": {name: np.asarray(f[row], dtype=float) for name, f in forecasts.items()},
            "mase": mase_by_model, "mase_n": mase_n,
        })

    out = Path(args.out) if args.out else run_dir.parent / f"sweep_case_studies_{param}.html"
    render_case_studies_html(param, entries, out)
    print(f"\nwrote {out}\n")
    for e in entries:
        tag = " (anomaly)" if e["anomalous"] else ""
        print(f"  {param}={e['value']:g}{tag}: " +
              ", ".join(f"{n} sweep-MASE={m:.3f}" for n, m in e["mase"].items()))


if __name__ == "__main__":
    main()
