"""Compact per-feature rendering for the SAE section: one row per feature, with pictures.

ROADMAP.md sec 26 B2/B3. What this replaces, and why:

The SAE section used to render, per analyzed layer, a 25-row table of
`feature | best_field | rho | series_id | family | activation | field_value`
-- five rows per feature, one per exemplar, repeating the feature's own
`best_field`/`rho` in every one. Eleven layers of that is 275 rows of
near-duplicate text, and the "top exemplar series" it offered was a bare
hex id (`a588c392da4f40fa`), which tells a reader nothing about what the
feature actually responds to. A reader cannot form a hypothesis from a
hash.

So: one row per feature, the five exemplars collapsed into five small
inline sparklines of the actual series, each clickable to enlarge. The
point of the picture is that "this feature fires on intermittent series"
is something you can SEE in three seconds and cannot see in a hex id.

Two deliberate implementation choices:

- **Inline SVG, not plotly.** A live run's report already carries 60+
  plotly figures in a ~3MB file; adding ~50 more small charts would bloat
  it badly for charts that need no interactivity beyond "make it bigger".
  An `<svg viewBox=...>` also scales into the modal with no second copy of
  the data -- the enlarge path clones the same node.
- **The modal is an overlay, not a new tab** (an explicit requirement):
  clicking a sparkline clones it into a fixed-position panel with the
  row's own numbers beside it, and Escape / click-outside closes it.

`clean`/`patched` are both accepted by `sparkline_svg` so the
with-and-without-the-feature overlay slots in unchanged once a run
produces patched forecasts; today only the series itself is free (it is
already in the corpus), so a caller that has no patched array simply omits
it and the sparkline renders the series alone rather than faking a
contrast.
"""

from __future__ import annotations

import html
from typing import Optional, Sequence

import numpy as np

from ..sae.vocab import describe_term

# Small enough to sit inside a table cell without stretching the row, large
# enough that an intermittent series is visibly different from a trending
# one at a glance.
_SPARK_W, _SPARK_H = 132, 30


# A 576-point context emitted point-for-point is ~6KB of SVG path data, and
# five exemplars on each of ~50 features would add ~1.5MB to the report --
# worse than the plotly cost inline SVG was chosen to avoid. It is also more
# resolution than a 132px-wide chart can show (4.4 points per pixel).
_MAX_POINTS = 192


def _envelope(y: np.ndarray, budget: int = _MAX_POINTS) -> np.ndarray:
    """Downsample to at most `budget` points, PRESERVING EXTREMA.

    Plain decimation (`y[::k]`) is wrong here and quietly so: an
    intermittent series is mostly zeros with a handful of one-step spikes,
    and taking every 4th point deletes most of those spikes. The sparkline
    would then show a flat line for exactly the series an intermittency or
    anomaly feature exists to detect -- a picture that contradicts the row
    it illustrates.

    So bucket the series and emit each bucket's min and max in time order.
    Every local extreme survives, at the cost of halving the effective
    horizontal resolution, which is the right trade for a chart this size.
    """
    n = y.size
    if n <= budget:
        return y
    nb = budget // 2
    edges = np.linspace(0, n, nb + 1).astype(int)
    out = np.empty(nb * 2, dtype=np.float64)
    for i in range(nb):
        seg = y[edges[i]:edges[i + 1]]
        if seg.size == 0 or bool(np.all(np.isnan(seg))):
            out[2 * i] = out[2 * i + 1] = np.nan
            continue
        lo_i, hi_i = int(np.nanargmin(seg)), int(np.nanargmax(seg))
        # Keep the two extremes in the order they actually occur, so a
        # spike-then-recover and a dip-then-recover do not look alike.
        first, second = (lo_i, hi_i) if lo_i <= hi_i else (hi_i, lo_i)
        out[2 * i], out[2 * i + 1] = seg[first], seg[second]
    return out


def sparkline_svg(series: Sequence[float], context_len: int,
                  clean: Optional[Sequence[float]] = None,
                  patched: Optional[Sequence[float]] = None,
                  title: str = "") -> str:
    """One small, self-scaling chart of a series and (optionally) two forecasts.

    The context is drawn muted and the true future in the foreground colour,
    so the boundary the model forecasts from is visible without a legend.
    `clean`/`patched`, when supplied, overlay the model's forecast with the
    feature left alone and with it steered -- the "with and without" view.
    Omitted rather than faked when a run has not produced them.
    """
    y = np.asarray(series, dtype=np.float64)
    if y.size == 0:
        return "<span class='spark-missing'>no series</span>"
    w, h = float(_SPARK_W), float(_SPARK_H)
    # Everything shares one y-scale so the overlays are comparable to the
    # series they are overlaid on.
    stack = [y] + [np.asarray(a, dtype=np.float64)
                   for a in (clean, patched) if a is not None]
    allv = np.concatenate(stack)
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    rng = (hi - lo) or 1.0

    n = y.size
    cut = max(0, min(int(context_len), n))
    x_cut = w * (cut / n) if n else w

    def _map(a: np.ndarray, x0: float, x1: float) -> str:
        """Envelope-downsample, then map onto the polyline's own x-range.

        The split at `cut` happens BEFORE downsampling and each side is
        given its own budget proportional to its screen width -- envelope
        -downsampling the concatenated series first would move the cut
        index and draw the forecast boundary in the wrong place.
        """
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return ""
        span = max(1e-9, (x1 - x0) / max(w, 1e-9))
        a = _envelope(a, max(8, int(_MAX_POINTS * span)))
        ys = 2.0 + (1.0 - (a - lo) / rng) * (h - 4.0)
        xs = np.linspace(x0, x1, a.size)
        return " ".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, ys) if np.isfinite(v))
    parts = [f"<svg class='spark' viewBox='0 0 {w:.0f} {h:.0f}' "
             f"preserveAspectRatio='none' role='img' tabindex='0' "
             f"aria-label='{html.escape(title or 'series sparkline')}'>"]
    if cut > 1:
        parts.append(f"<polyline class='spark-ctx' points='{_map(y[:cut], 0, x_cut)}'/>")
    if n - cut > 1:
        parts.append(f"<polyline class='spark-fut' points='{_map(y[cut:], x_cut, w)}'/>")
    if clean is not None and len(clean) > 1:
        parts.append(f"<polyline class='spark-clean' points='{_map(clean, x_cut, w)}'/>")
    if patched is not None and len(patched) > 1:
        parts.append(f"<polyline class='spark-patched' points='{_map(patched, x_cut, w)}'/>")
    if 0 < cut < n:
        parts.append(f"<line class='spark-cut' x1='{x_cut:.1f}' y1='0' "
                     f"x2='{x_cut:.1f}' y2='{h:.0f}'/>")
    parts.append("</svg>")
    return "".join(parts)


def exemplar_cell(exemplars: list) -> str:
    """A row's exemplar sparklines, each clickable to enlarge.

    `exemplars` is a list of dicts with `svg`, `series_id`, `family`,
    `activation`. The caption under each is deliberately the FAMILY, not
    the series id -- the family is the part a reader can reason about, and
    the id is available on the enlarged view for anyone who wants to look
    the series up.
    """
    if not exemplars:
        return "<span class='spark-missing'>none</span>"
    cells = []
    for ex in exemplars:
        meta = (f"{html.escape(str(ex.get('series_id', '')))} · "
                f"{html.escape(str(ex.get('family', '')))} · "
                f"activation {ex.get('activation', float('nan')):.2f}")
        cells.append(
            f"<figure class='spark-cell' data-meta=\"{meta}\" "
            f"title='Click to enlarge'>{ex['svg']}"
            f"<figcaption>{html.escape(str(ex.get('family', '')))}</figcaption></figure>")
    return f"<div class='spark-row'>{''.join(cells)}</div>"


def tracks_cell(field: Optional[str], rho: Optional[float], n: Optional[int]) -> str:
    """The 'what it tracks' cell: a defined human label, not a raw identifier.

    Renders the plain-English label with the raw field name and the
    definition available on hover, plus rho and its own sample size --
    a rho on 374 series and one on 965 are not the same claim, so `n`
    travels with it rather than being dropped.
    """
    if not field:
        return "<span class='muted'>no structural match above threshold</span>"
    d = describe_term(field)
    tip = html.escape(f"{field} — {d.what} High: {d.high}")
    body = (f"<abbr class='term' title=\"{tip}\">{html.escape(d.label)}</abbr>")
    if rho is not None:
        body += f" <span class='muted'>ρ={rho:+.2f}"
        if n:
            body += f", n={n}"
        body += "</span>"
    return body


def provenance_cell(field: Optional[str], rho: Optional[float]) -> str:
    """The provenance match, rendered as the caveat it is rather than as a finding.

    Always dimmed and always labelled, because this column existing at all
    is the point: sec 26 A1 found 78.6% of matched features were reported
    as tracking a `generator_*`/`tier_*` dummy, with nothing on screen
    saying that is a corpus artifact rather than a model property.
    """
    if not field:
        return "<span class='muted'>—</span>"
    d = describe_term(field)
    tip = html.escape(f"{field} — {d.what}")
    r = f" ρ={rho:+.2f}" if rho is not None else ""
    return (f"<span class='prov'><abbr class='term' title=\"{tip}\">"
            f"{html.escape(d.label)}</abbr>{r}</span>")


# The enlarge overlay. One instance per report; every sparkline clones into
# it. Kept here rather than in report.py's global CSS so the whole feature
# -- markup, style and behaviour -- is readable in one place.
MODAL_ASSETS = """
<style>
.spark { width: 132px; height: 30px; overflow: visible; }
.spark polyline { fill: none; vector-effect: non-scaling-stroke; }
.spark-ctx     { stroke: var(--muted, #9aa0a6); stroke-width: 1; opacity: .75; }
.spark-fut     { stroke: var(--fg, #202124); stroke-width: 1.2; }
.spark-clean   { stroke: #1a73e8; stroke-width: 1.2; }
.spark-patched { stroke: #d93025; stroke-width: 1.2; stroke-dasharray: 3 2; }
.spark-cut     { stroke: var(--muted, #9aa0a6); stroke-width: .6; stroke-dasharray: 2 2; opacity: .6; }
.spark-row     { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-end; }
.spark-cell    { margin: 0; cursor: zoom-in; border: 1px solid transparent; border-radius: 4px; padding: 2px; }
.spark-cell:hover, .spark-cell:focus-within { border-color: var(--muted, #9aa0a6); background: rgba(128,128,128,.06); }
.spark-cell figcaption { font-size: 10px; color: var(--muted, #5f6368); text-align: center; margin-top: 1px;
                         max-width: 132px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.spark-missing { color: var(--muted, #9aa0a6); font-style: italic; font-size: 12px; }
.prov          { color: var(--muted, #9aa0a6); font-size: 12px; }
abbr.term      { text-decoration: underline dotted; cursor: help; }
#spark-modal   { position: fixed; inset: 0; background: rgba(0,0,0,.55); display: none;
                 align-items: center; justify-content: center; z-index: 9999; }
#spark-modal.open { display: flex; }
#spark-modal .panel { background: var(--bg, #fff); color: var(--fg, #202124); border-radius: 8px;
                      padding: 18px 20px; max-width: 92vw; max-height: 88vh; overflow: auto;
                      box-shadow: 0 8px 40px rgba(0,0,0,.4); }
#spark-modal .panel svg { width: min(86vw, 900px); height: min(58vh, 380px); }
#spark-modal .panel .meta { font: 13px/1.5 system-ui, sans-serif; color: var(--muted, #5f6368);
                            margin-top: 10px; }
#spark-modal .panel .hint { font: 11px/1.4 system-ui, sans-serif; color: var(--muted, #9aa0a6); margin-top: 6px; }
</style>
<div id="spark-modal" role="dialog" aria-modal="true" aria-label="Enlarged series">
  <div class="panel"><div class="holder"></div>
  <div class="meta"></div>
  <div class="hint">Grey = context the model saw · dark = true continuation ·
   blue = forecast · red dashed = forecast with this feature steered.
   Press Escape or click outside to close.</div></div>
</div>
<script>
(function () {
  var modal = document.getElementById('spark-modal');
  if (!modal) return;
  var holder = modal.querySelector('.holder'), meta = modal.querySelector('.meta');
  function open(cell) {
    var svg = cell.querySelector('svg');
    if (!svg) return;
    holder.innerHTML = '';
    holder.appendChild(svg.cloneNode(true));
    meta.textContent = cell.getAttribute('data-meta') || '';
    modal.classList.add('open');
  }
  document.addEventListener('click', function (e) {
    var cell = e.target.closest && e.target.closest('.spark-cell');
    if (cell) { open(cell); return; }
    if (e.target === modal) modal.classList.remove('open');
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') modal.classList.remove('open');
    if ((e.key === 'Enter' || e.key === ' ') && document.activeElement &&
        document.activeElement.closest && document.activeElement.closest('.spark-cell')) {
      e.preventDefault(); open(document.activeElement.closest('.spark-cell'));
    }
  });
})();
</script>
"""


# Channels that are magnitudes, not signed quantities. Read from
# `sae/describe.py` rather than re-listed, so the narrator and the report
# cannot disagree about which channel has a direction (sec 2.2).
try:
    from ..sae.describe import CHANNEL_VERB as _CHANNEL_VERB
    _UNSIGNED_CHANNELS = frozenset(_CHANNEL_VERB)
except Exception:                                          # pragma: no cover
    _UNSIGNED_CHANNELS = frozenset({"horizon_shape_near", "horizon_shape_far"})


def ablation_effect_label(entry: Optional[dict]) -> str:
    """One phrase for what REMOVING this feature does, in null units.

    Reports the channel with the largest cleared |signed effect / null p95|,
    with a direction word -- the direction is the whole reason
    `signed_effect` is recorded beside the unsigned `effect` (two features
    whose removal moves the level in opposite directions have opposite
    causal roles). Channels whose null had no spread are excluded here as
    they are everywhere else: they cleared nothing, so they cannot be the
    largest thing that cleared.

    Returns "" when there is no ablation record at all -- the caller then
    renders nothing, rather than a sentence implying the pass ran and found
    no effect (`CLAUDE.md` sec 11.37: absent and null are different).
    """
    if not entry:
        return ""
    if not entry.get("scorable"):
        return "not measured: " + str(entry.get("reason") or "not scorable")
    best, best_ratio = None, 0.0
    for ch, rec in (entry.get("channels") or {}).items():
        if not rec.get("available") or not rec.get("clears_null"):
            continue
        p95, signed = rec.get("null_p95"), rec.get("signed_effect")
        if not p95 or signed is None:
            continue
        # The ratio printed must be the one that DECIDED the clearing --
        # `clears_null` compares the unsigned `effect` against the p95, and
        # `|signed| / p95` is a different, smaller number whenever the
        # per-series effects partly cancel. The first version printed the
        # latter, which rendered "1.0x its null" beside a cell that had
        # cleared comfortably. Direction still comes from `signed_effect`;
        # the two are read off separately because they answer separate
        # questions. Found by reading the rendered report (sec 11.48).
        ratio = float(rec.get("effect") or 0.0) / float(p95)
        if ratio > best_ratio:
            best, best_ratio = (ch, float(signed)), ratio
    if best is None:
        return "removing it moves no channel past its own null"
    ch, signed = best
    # `horizon_shape_*` is a mean ABSOLUTE deviation, so it has no sign and a
    # direction word on it is a fabrication -- the same rule
    # `sae/describe.py::CHANNEL_VERB` encodes for the narrator, and the same
    # one `check_text`'s `_DIRECTION_ON_HORIZON` guard enforces there. The
    # renderer had no equivalent and printed "raises near horizon".
    if ch in _UNSIGNED_CHANNELS:
        verb = "reshapes"
    else:
        verb = "raises" if signed > 0 else "lowers"
    return (f"removing it {verb} {describe_term(ch).label.lower()} "
            f"({best_ratio:.1f}x its null)")


def ablation_cell(entry: Optional[dict], max_series: int = 3) -> str:
    """The with-and-without-the-feature forecasts for one feature.

    Each sparkline draws that series' own context and true continuation,
    overlaid with the forecast from the SAE's FULL reconstruction (`clean`)
    and from the same reconstruction with this one atom zeroed
    (`patched`). The baseline is the full reconstruction, not the raw
    model forecast, so the gap between the two lines is the feature's own
    contribution and not the SAE's reconstruction cost -- the picture and
    the channel numbers beside it then describe the same contrast.

    Everything drawn comes from the ablation artifact itself, so this needs
    no corpus lookup and cannot pair a forecast with the wrong series.
    """
    if not entry or not entry.get("scorable"):
        return "<span class='spark-missing'>not measured</span>"
    fcs = (entry.get("forecasts") or [])[:max_series]
    if not fcs:
        return "<span class='spark-missing'>no forecasts kept</span>"
    cells = []
    for f in fcs:
        ctx, tgt = list(f.get("context") or []), list(f.get("target") or [])
        if not ctx or not tgt:
            continue
        meta = (f"{html.escape(str(f.get('series_id', '')))} · "
                f"activation {float(f.get('activation', float('nan'))):.2f} · "
                f"solid = with the feature, dashed = with it removed")
        svg = sparkline_svg(ctx + tgt, len(ctx),
                            clean=f.get("with_feature"),
                            patched=f.get("without_feature"),
                            title=f"series {f.get('series_id', '')} with and "
                                  f"without feature {entry.get('feature')}")
        cells.append(f"<figure class='spark-cell' data-meta=\"{meta}\" "
                     f"title='Click to enlarge'>{svg}"
                     f"<figcaption>with · without</figcaption></figure>")
    if not cells:
        return "<span class='spark-missing'>no forecasts kept</span>"
    return f"<div class='spark-row'>{''.join(cells)}</div>"


def feature_table_html(cards: list, series_lookup, context_len: int,
                       descriptions: Optional[dict] = None,
                       ablations: Optional[dict] = None) -> str:
    """The compact one-row-per-feature table (ROADMAP.md sec 26 B2/B3).

    `series_lookup` maps a series id to its raw values (or `None` when the
    corpus is not loadable); a row whose exemplars cannot be drawn renders
    the ids as text rather than dropping the row, so a missing corpus
    degrades the picture without deleting the measurement.

    `descriptions` optionally maps feature index -> a one-sentence
    description. It is rendered verbatim and is never required: the table
    is complete and readable without it, so a run with no narrator
    available loses a convenience, not a column of evidence.

    `ablations` optionally maps feature index -> that feature's entry in a
    `*_ablation.json` candidate list, which adds the two causal columns:
    what removing the feature does, and the forecasts with and without it.
    Both are omitted wholesale when no ablation pass has run for this
    target -- an empty column would read as "measured, no effect".
    """
    if not cards:
        return "<p class='blurb'>no features to illustrate.</p>"
    has_desc = bool(descriptions)
    has_abl = bool(ablations)
    head = ("<tr><th>Feature</th>"
            + ("<th>What it looks like it does</th>" if has_desc else "")
            + "<th>Structural correlate</th>"
            + ("<th>What removing it does</th>" if has_abl else "")
            + "<th>Top-activating series <span class='muted'>(click to enlarge)</span></th>"
            + ("<th>Forecast with · without <span class='muted'>(click to enlarge)"
               "</span></th>" if has_abl else "")
            + "<th>Corpus label</th></tr>")
    body = []
    for c in cards:
        exs = []
        for ex in c["exemplars"]:
            vals = series_lookup(ex["series_id"]) if series_lookup else None
            svg = (sparkline_svg(vals, context_len,
                                 title=f"series {ex['series_id']}")
                   if vals is not None else "")
            exs.append({**ex, "svg": svg})
        drawable = [e for e in exs if e["svg"]]
        ex_html = (exemplar_cell(drawable) if drawable else
                   "<span class='muted'>" +
                   ", ".join(html.escape(str(e["series_id"])) for e in exs) +
                   "</span>")
        row = f"<tr><td><code>#{c['feature']}</code></td>"
        if has_desc:
            d = (descriptions or {}).get(c["feature"])
            row += ("<td>" + (html.escape(d) if d else
                              "<span class='muted'>no description generated</span>") + "</td>")
        abl = (ablations or {}).get(c["feature"])
        row += f"<td>{tracks_cell(c['structural_field'], c['structural_rho'], c['structural_n'])}</td>"
        if has_abl:
            label = ablation_effect_label(abl)
            row += ("<td>" + (html.escape(label) if label else
                              "<span class='muted'>not a candidate</span>") + "</td>")
        row += f"<td>{ex_html}</td>"
        if has_abl:
            row += f"<td>{ablation_cell(abl)}</td>"
        row += (f"<td>{provenance_cell(c['provenance_field'], c['provenance_rho'])}</td></tr>")
        body.append(row)
    return (f"<table class='tbl'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def term_legend_html(names: Sequence[str], heading: str = "") -> str:
    """A definition table for whatever identifiers a figure actually rendered.

    ROADMAP.md sec 26 B1. Built from the names PRESENT in the figure rather
    than from the full vocabulary, so it cannot list a channel this run
    never probed or omit a field it did -- the same
    derive-from-the-data rule the scorecard follows.

    Covers channels and ground-truth fields with one function because the
    reader hits both in the same section (a heatmap axis says
    `spectral_centroid`; the row beside it says `has_random_walk`) and
    should not have to know which vocabulary a word came from.
    """
    seen, rows = set(), []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        d = describe_term(name)
        rows.append(
            f"<tr><td><code>{html.escape(str(name))}</code></td>"
            f"<td><b>{html.escape(d.label)}</b></td>"
            f"<td>{html.escape(d.what)}</td>"
            f"<td>{html.escape(d.high)}</td></tr>")
    if not rows:
        return ""
    return (f"{heading}<table class='tbl'><thead><tr><th>In the data</th>"
            f"<th>Name</th><th>What it measures</th>"
            f"<th>What a high value means</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")
