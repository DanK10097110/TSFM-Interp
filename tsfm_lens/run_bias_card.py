"""ROADMAP.md §7 Phase 3, bullet 5: build the per-model "bias card."

Reads the controlled-parameter-sweep JSONs `run_parameter_sweep.py` already
wrote and consolidates them into one standalone HTML summary of what each
model is systematically better/worse at -- "the artifact that most directly
answers the brief's original research questions," per the roadmap bullet.
Re-runs nothing: this is pure synthesis over already-computed artifacts,
the same "cross-artifact aggregator, separate from any single run's own
report.html" pattern `report/meta_report.py` already established.

Example (after running `run_parameter_sweep.py` for each swept parameter):
    python run_bias_card.py --sweeps runs/param_sweep_seasonal_period.json,\\
        runs/param_sweep_noise_scale.json,runs/param_sweep_intermittency_rate.json \\
        --out runs/bias_card.html
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from tsfm_lens.report.bias_card import build_bias_card, render_bias_card_html, summarize_param_sweep
from tsfm_lens.utils import load_json, log, save_json, setup_logging

# Methodological caveats found while building each sweep (ROADMAP.md §7's
# Findings) -- attached to both models' cards so a reader can't mistake a
# confounded ranking for a clean one.
CAVEATS = {
    "intermittency_rate": (
        "parametric's intermittency zeroes the entire series independently per point, "
        "including the forecast horizon itself; MASE's own scale term (mean absolute "
        "context step-change) also shrinks under heavy zeroing, so both the numerator "
        "and denominator become dominated by a floor effect at high rates that has "
        "little to do with either model's actual forecasting skill."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROADMAP.md §7 Phase 3: consolidate parameter sweeps into a bias card")
    parser.add_argument("--sweeps", default=None,
                        help="comma-separated param_sweep_*.json paths "
                             "(default: glob runs/param_sweep_*.json)")
    parser.add_argument("--out", default="runs/bias_card.html")
    args = parser.parse_args()

    setup_logging()
    if args.sweeps:
        paths = [Path(p) for p in args.sweeps.split(",")]
    else:
        paths = sorted(Path(p) for p in glob.glob("runs/param_sweep_*.json"))
    if not paths:
        log.warning("no param_sweep_*.json files found; nothing to build a bias card from")
        return

    sweep_jsons, summaries = {}, []
    for path in paths:
        sweep = load_json(path)
        param = sweep["param"]
        sweep_jsons[param] = sweep
        try:
            summaries.append(summarize_param_sweep(sweep))
        except ValueError as e:
            log.warning("skipping %s: %s", path, e)
    if not summaries:
        log.warning("no usable two-model sweeps among %s; nothing to summarize", paths)
        return

    card = build_bias_card(summaries, caveats=CAVEATS)
    out_html = Path(args.out)
    render_bias_card_html(card, summaries, sweep_jsons, out_html)
    out_json = out_html.with_suffix(".json")
    save_json(out_json, {"card": card, "summaries": summaries})

    print(f"\nwrote {out_html} and {out_json}\n")
    for model in card["models"]:
        print(f"== {model} ==")
        for line in card["cards"][model]:
            print(f"  - {line}")
        if not card["cards"][model]:
            print("  (no confidently-different sweep points found)")


if __name__ == "__main__":
    main()
