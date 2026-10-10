"""Report panel for forecast completeness (ROADMAP.md sec 42 task C): how much of
the forecast change the top-N ranked SAE features carry, against the SAE's error
term and against activation-matched random feature sets (`sae/completeness.json`).

Reads the finished artifact only. A target that was withheld or skipped renders
as its stated reason, never as a row of zeros. The caption states the evidence
class (descriptive) and which ranking produced the features, because a fallback
prominence ranking is not an attribution ranking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..sae.completeness import FALLBACK_RANKING
from ..utils import load_json


def _fmt(v, nd: int = 4) -> str:
    return "n/a" if v is None else f"{float(v):.{nd}f}"


def _with_ci(v, ci, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    lo, hi = (ci or [None, None])
    return f"{_fmt(v, nd)} [{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


def completeness_rows(art: dict) -> tuple:
    """`-> (rows, notes)`: one row per (target, set size) of every measured target,
    and one stated reason per withheld or skipped target."""
    rows, notes = [], []
    for key, t in art.get("targets", {}).items():
        if t.get("skipped") or t.get("withheld"):
            notes.append(f"{key}: {'skipped' if t.get('skipped') else 'withheld'} - {t.get('reason')}")
            continue
        for n, r in t["per_set_size"].items():
            if not r.get("measured"):
                notes.append(f"{key} (top {n}): not measured - {r.get('reason')}")
                continue
            rows.append({
                "target": key, "top N": int(n), "series": r["n_series"],
                "feature set": _with_ci(r["delta_features"], r["delta_features_ci"]),
                "error term": _with_ci(r["delta_error"], r["delta_error_ci"]),
                "share explained": _with_ci(r["share_explained"], r["share_explained_ci"], 3),
                "median per-series share": _with_ci(r.get("share_explained_median_series"),
                                                    r.get("share_explained_median_series_ci"), 3),
                "matched random set": _with_ci(r["delta_random"], r["delta_random_ci"]),
                "excess over random": _with_ci(r["excess_over_random"],
                                               r["excess_over_random_ci"]),
                "p vs random draws": _fmt(r["p_vs_random_draws"], 3),
                "ranking": "fallback (prominence)" if t["ranking_source"] == FALLBACK_RANKING
                else t["ranking_source"]})
    return rows, notes


def completeness_section_block(cfg, run_dir, findings: list) -> tuple:
    """`-> (html, status, detail)`; skipped with a reason when no artifact exists."""
    from .report import _note, _table
    path = Path(run_dir) / "sae" / "completeness.json"
    if not path.exists():
        return "", "skipped", ("no sae/completeness.json (concepts.completeness is not set, or "
                               "the concepts stage has not run)")
    art = load_json(path)
    rows, notes = completeness_rows(art)
    html = ""
    if rows:
        html += _table(pd.DataFrame(rows))
    html += "".join(f'<p class="blurb">{n}</p>' for n in notes)
    html += _note(
        "How much of the forecast change do a few SAE features carry, compared with the SAE's "
        "error term and with random features of the same activation? Descriptive, within-model.",
        "Each number is the mean over series of the mean absolute change of the point forecast "
        "over the horizon, divided by the series' context standard deviation, with a series-"
        "bootstrap 95% interval. 'Feature set' zeroes the top N atoms out of the SAE "
        "reconstruction (error term kept); 'error term' replaces the activation by its "
        "reconstruction; 'matched random set' removes N alive atoms drawn from the same decile "
        "of mean activation. Share explained is feature set / (feature set + error term), computed on the means; the median per-series share is beside it because a few series with a large change can dominate the means. "
        "A feature set above its random baseline, with an interval that excludes zero, is "
        "what 'these atoms matter more than equally active atoms' looks like.",
        "The two changes have different baselines (reconstruction versus the clean forecast), "
        "so the share is a coverage ratio, not a decomposition of one total. A ranking labelled "
        "fallback orders atoms by mean activation times decoder norm, which is prominence, not "
        "effect: a low share under it does not bound what an attribution ranking would find. "
        "Evidence class: descriptive. Zeroing atoms moves activations off the SAE's training "
        "distribution, and a model's forecast can be insensitive to a layer for reasons that "
        "have nothing to do with the SAE.")
    return html, "rendered", ""
