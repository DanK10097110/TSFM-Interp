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

# ROADMAP.md sec 32.6 Item E: the with/without difference, drawn on its OWN
# scale in a separate band beneath the main chart. §32.6 measured why this
# is necessary rather than a rescale: the two forecasts being compared are
# nearly identical (that IS the point of the SAE reconstructing well), so
# any shared axis that shows both is ~2 orders of magnitude larger than
# their difference and the gap is sub-pixel (median 0.18px, §32.6's table).
# `_DIFF_GAP` separates the two bands visually; total viewBox height grows
# from 30 to 30+2+12 = 44 exactly when a diff band is drawn, per the spec's
# own worked number.
_DIFF_GAP, _DIFF_H = 2.0, 12.0


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

    The FIRST and LAST values are then pinned back on, because preserving
    extrema does not preserve endpoints and the endpoint is the one point
    on this chart that carries a meaning of its own: `sparkline_svg` draws
    the context and the forecast as two polylines meeting at the cut, so
    the context's last drawn point is where a reader sees the forecast
    being anchored. Without this, the last bucket of an intermittent
    series emits its SPIKE as the context's final point -- measured on
    `runs/full_report_run_4model`, series `7500ab0a3b48b75f` ends at 0.0
    and is drawn ending at 20.6, moving the seam 17.6px on a 30px chart
    while the forecast's own start is off by 0.09px. The overlays then
    read as starting far below a context that does not end where it is
    drawn ending. They are appended rather than substituted, so no
    extreme this function exists to keep is traded away for them.
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
    head = y[:1] if (np.isfinite(y[0]) and out[0] != y[0]) else out[:0]
    tail = y[-1:] if (np.isfinite(y[-1]) and out[-1] != y[-1]) else out[:0]
    return np.concatenate([head, out, tail]) if head.size or tail.size else out


def sparkline_svg(series: Sequence[float], context_len: int,
                  clean: Optional[Sequence[float]] = None,
                  patched: Optional[Sequence[float]] = None,
                  unpatched: Optional[Sequence[float]] = None,
                  diff: Optional[Sequence[float]] = None,
                  title: str = "",
                  max_points: int = _MAX_POINTS,
                  viewbox_w: float = _SPARK_W,
                  viewbox_h: float = _SPARK_H) -> str:
    """One small, self-scaling chart of a series and (optionally) three forecasts.

    The context is drawn muted and the true future in the foreground colour,
    so the boundary the model forecasts from is visible without a legend.
    `clean`/`patched`, when supplied, overlay the model's forecast with the
    feature left alone and with it steered -- the "with and without" view.
    Omitted rather than faked when a run has not produced them.

    `unpatched` (ROADMAP.md sec 32.7c Item I4) is the RAW model forecast --
    no SAE reconstruction at all -- drawn as a third, muted trace beneath
    `clean`/`patched`. `clean` is already the SAE's full reconstruction, so
    the gap between `unpatched` and `clean` is the dictionary's own
    reconstruction cost, drawn on the same axes as the feature's own
    contribution (the gap between `clean` and `patched`) so a reader can see
    which gap is larger without reading two separate numbers.

    `diff` (ROADMAP.md sec 32.6 Item E) is `clean - patched`, drawn in a
    SEPARATE horizontal band beneath the main chart with its own `lo/hi` and
    a zero line -- never on the shared scale above, which sec 32.6 measured
    is the reason a real, sizeable effect (31.9% of panels move by more than
    10% of a context sd) can draw as a sub-pixel gap: the two forecasts
    being compared are nearly identical by construction (a working SAE
    reconstructs well), so any axis that shows both is ~2 orders of
    magnitude larger than their difference. Omitted (no band drawn) rather
    than faked when not supplied.
    """
    y = np.asarray(series, dtype=np.float64)
    if y.size == 0:
        return "<span class='spark-missing'>no series</span>"
    w, h = float(viewbox_w), float(viewbox_h)
    # Everything shares one y-scale so the overlays are comparable to the
    # series they are overlaid on. `diff` is deliberately NOT in this stack
    # -- it gets its own band and its own scale below.
    stack = [y] + [np.asarray(a, dtype=np.float64)
                   for a in (clean, patched, unpatched) if a is not None]
    allv = np.concatenate(stack)
    lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    rng = (hi - lo) or 1.0

    n = y.size
    cut = max(0, min(int(context_len), n))
    x_cut = w * (cut / n) if n else w

    diff_arr = None
    if diff is not None:
        d = np.asarray(diff, dtype=np.float64)
        if d.size > 1 and np.any(np.isfinite(d)):
            diff_arr = d

    total_h = h + (_DIFF_GAP + _DIFF_H if diff_arr is not None else 0.0)

    def _map_range(a: np.ndarray, x0: float, x1: float,
                  y0: float, y1: float, v_lo: float, v_hi: float) -> str:
        """Envelope-downsample, then map onto an arbitrary x-range AND
        y-band -- the shared main-chart mapping and the diff band's own
        mapping are the same operation over different bounds, so both go
        through this one function rather than risking the two silently
        drifting apart (CLAUDE.md sec 11.6-class report-key drift, one
        level down at the pixel-mapping level).

        The split at `cut` happens BEFORE downsampling and each side is
        given its own budget proportional to its screen width --
        envelope-downsampling the concatenated series first would move the
        cut index and draw the forecast boundary in the wrong place.
        """
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return ""
        span = max(1e-9, (x1 - x0) / max(w, 1e-9))
        a = _envelope(a, max(8, int(max_points * span)))
        v_rng = (v_hi - v_lo) or 1.0
        ys = y0 + 2.0 + (1.0 - (a - v_lo) / v_rng) * (y1 - y0 - 4.0)
        xs = np.linspace(x0, x1, a.size)
        return " ".join(f"{x:.1f},{v:.1f}" for x, v in zip(xs, ys) if np.isfinite(v))

    def _map(a: np.ndarray, x0: float, x1: float) -> str:
        return _map_range(a, x0, x1, 0.0, h, lo, hi)

    parts = [f"<svg class='spark' viewBox='0 0 {w:.0f} {total_h:.0f}' "
             f"preserveAspectRatio='none' role='img' tabindex='0' "
             f"aria-label='{html.escape(title or 'series sparkline')}'>"]
    if cut > 1:
        parts.append(f"<polyline class='spark-ctx' points='{_map(y[:cut], 0, x_cut)}'/>")
    if n - cut > 1:
        parts.append(f"<polyline class='spark-fut' points='{_map(y[cut:], x_cut, w)}'/>")
    # Drawn BEFORE clean/patched so it sits underneath them -- it is the
    # reconstruction-cost reference, not the contrast the two foreground
    # traces exist to show.
    if unpatched is not None and len(unpatched) > 1:
        parts.append(f"<polyline class='spark-unpatched' points='{_map(unpatched, x_cut, w)}'/>")
    if clean is not None and len(clean) > 1:
        parts.append(f"<polyline class='spark-clean' points='{_map(clean, x_cut, w)}'/>")
    if patched is not None and len(patched) > 1:
        parts.append(f"<polyline class='spark-patched' points='{_map(patched, x_cut, w)}'/>")
    if 0 < cut < n:
        parts.append(f"<line class='spark-cut' x1='{x_cut:.1f}' y1='0' "
                     f"x2='{x_cut:.1f}' y2='{total_h:.0f}'/>")
    if diff_arr is not None:
        band_y0, band_y1 = h + _DIFF_GAP, h + _DIFF_GAP + _DIFF_H
        d_lo, d_hi = float(np.nanmin(diff_arr)), float(np.nanmax(diff_arr))
        d_rng = (d_hi - d_lo) or 1.0
        zero_frac = min(1.0, max(0.0, (0.0 - d_lo) / d_rng))
        zero_y = band_y0 + 2.0 + (1.0 - zero_frac) * (band_y1 - band_y0 - 4.0)
        parts.append(f"<line class='spark-diffzero' x1='{x_cut:.1f}' "
                     f"y1='{zero_y:.1f}' x2='{w:.0f}' y2='{zero_y:.1f}'/>")
        parts.append(f"<polyline class='spark-diff' points="
                     f"'{_map_range(diff_arr, x_cut, w, band_y0, band_y1, d_lo, d_hi)}'/>")
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
.spark-unpatched { stroke: var(--muted, #9aa0a6); stroke-width: 1; opacity: .55; stroke-dasharray: 1 2; }
.spark-cut     { stroke: var(--muted, #9aa0a6); stroke-width: .6; stroke-dasharray: 2 2; opacity: .6; }
.spark-diff    { stroke: #b8860b; stroke-width: 1.1; }
.spark-diffzero { stroke: var(--muted, #9aa0a6); stroke-width: .5; stroke-dasharray: 1 1; opacity: .6; }
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
   pale dotted = raw model forecast (no SAE reconstruction) ·
   blue = forecast with this feature left alone · red dashed = with it removed.
   Press Escape or click outside to close.</div></div>
</div>
<script>
(function () {
  var modal = document.getElementById('spark-modal');
  if (!modal) return;
  var holder = modal.querySelector('.holder'), meta = modal.querySelector('.meta');
  function open(cell) {
    // ROADMAP.md sec 32.6 Item G: clone the cell's own high-resolution
    // template when one was emitted (a future addition -- no cell emits
    // one today, per the measured HTML-size cost recorded beside this
    // function), falling back to the inline SVG exactly as before when
    // absent, so a cell with no template still enlarges as it always has.
    var tmpl = cell.querySelector('template.spark-full');
    var svg = tmpl ? tmpl.content.querySelector('svg') : cell.querySelector('svg');
    if (!svg) return;
    var clone = svg.cloneNode(true);
    // The inline cell deliberately squashes its aspect ratio to fit a
    // table-row-height box ('none'); the enlarged copy should not inherit
    // that distortion.
    clone.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    holder.innerHTML = '';
    holder.appendChild(clone);
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


def _best_ablation_effect(entry: dict) -> Optional[tuple]:
    """The (channel, signed_effect, ratio) that decided this entry's headline.

    Shared by `ablation_effect_label` and `ablation_effect_html` so the
    plain-text and HTML renderers cannot name a different channel for the
    same entry. Reports the channel with the largest cleared |signed effect
    / null p95| -- the direction is the whole reason `signed_effect` is
    recorded beside the unsigned `effect` (two features whose removal moves
    the level in opposite directions have opposite causal roles). Channels
    whose null had no spread are excluded here as they are everywhere else:
    they cleared nothing, so they cannot be the largest thing that cleared.
    Returns None if nothing in the entry cleared its null.
    """
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
        return None
    return best[0], best[1], best_ratio


def ablation_effect_label(entry: Optional[dict]) -> str:
    """One phrase for what REMOVING this feature does, in null units.

    Returns "" when there is no ablation record at all -- the caller then
    renders nothing, rather than a sentence implying the pass ran and found
    no effect (`CLAUDE.md` sec 11.37: absent and null are different).
    """
    if not entry:
        return ""
    if not entry.get("scorable"):
        return "not measured: " + str(entry.get("reason") or "not scorable")
    result = _best_ablation_effect(entry)
    if result is None:
        return "removing it moves no channel past its own null"
    ch, signed, best_ratio = result
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


def ablation_effect_html(entry: Optional[dict]) -> str:
    """`ablation_effect_label`, with the named channel as an `abbr.term`.

    ROADMAP.md sec 32.7c Item J2: a channel name printed as bare text (as
    the plain-text label always has) carries its meaning nowhere -- the
    reader has to already know what "spectral centroid" means. Wrapping it
    in the same `abbr.term` hover pattern `tracks_cell`/`provenance_cell`
    already use for structural fields costs nothing per render and answers
    the complaint at this site without changing what the sentence says.
    Shares `_best_ablation_effect` with the plain-text version, so the two
    can never disagree about which channel is named -- the plain-text
    function stays the one existing tests and any non-HTML caller use.
    """
    if not entry:
        return ""
    if not entry.get("scorable"):
        return html.escape("not measured: " +
                            str(entry.get("reason") or "not scorable"))
    result = _best_ablation_effect(entry)
    if result is None:
        return html.escape("removing it moves no channel past its own null")
    ch, signed, best_ratio = result
    if ch in _UNSIGNED_CHANNELS:
        verb = "reshapes"
    else:
        verb = "raises" if signed > 0 else "lowers"
    d = describe_term(ch)
    tip = f"{d.what} {d.high}".strip()
    chan = (f"<abbr class='term' title=\"{html.escape(tip)}\">"
            f"{html.escape(d.label.lower())}</abbr>")
    return f"removing it {verb} {chan} ({best_ratio:.1f}x its null)"


def ablation_cell(entry: Optional[dict], max_series: Optional[int] = None,
                  median_hidden_norm: Optional[float] = None,
                  population: Optional[dict] = None) -> str:
    """The with-and-without-the-feature forecasts for one feature.

    Each sparkline draws that series' own context and true continuation,
    overlaid with the forecast from the SAE's FULL reconstruction (`clean`)
    and from the same reconstruction with this one atom zeroed
    (`patched`). The baseline is the full reconstruction, not the raw
    model forecast, so the gap between the two lines is the feature's own
    contribution and not the SAE's reconstruction cost -- the picture and
    the channel numbers beside it then describe the same contrast. A third,
    muted `unpatched` trace (ROADMAP.md sec 32.7c Item I4) makes that
    statement checkable: it is the raw model forecast with no SAE
    reconstruction at all, so the gap between it and `clean` is the
    dictionary's own reconstruction cost, drawn on the same axes as the
    feature's own contribution.

    Everything drawn comes from the ablation artifact itself, so this needs
    no corpus lookup and cannot pair a forecast with the wrong series.

    A cell showing fewer than 3 panels names *why*, read from the
    candidate's own `n_top_series` -- never from how many panels actually
    rendered, since that field is the only thing that can tell "this
    feature is rare" (`n_top_series < 3`, an n-of-N causal measurement)
    apart from "this run kept fewer forecasts than it found"
    (`n_top_series >= 3`, a `--keep-forecasts` truncation) -- the two are
    different facts and a reader cannot otherwise separate them
    (`ROADMAP.md` sec 32.12, Item K). A small `n=N` marker is shown on
    every row, not only sub-3 ones, so a reader can see at a glance which
    candidates rest on a small sample without opening each cell (Item K3).

    ROADMAP.md sec 32.7c Item I1: a candidate whose battery cleared NO
    channel (`n_channels_clearing == 0`) is measured, not absent -- its
    picture is still drawn, in full, in the DOM -- but it must not compete
    visually with a candidate that actually cleared something, since the
    battery already knows these are not causal (the gate scores against a
    matched-magnitude random-direction null, not against "did the forecast
    move at all"). So it renders behind a collapsed `<details>` instead of
    at full prominence. Gated STRICTLY on the artifact's own
    `n_channels_clearing` -- never on how large the drawn gap looks, which
    is the picture the measurement exists to override -- and only when that
    field is actually present as an int: an older artifact that never
    recorded it is absent information, not a zero (`CLAUDE.md` sec 11.37),
    so it renders at full prominence rather than being silently collapsed
    on a threshold this function invented.

    `population` (ROADMAP.md sec 32.7 Item H2) is `derived.flatness_
    population`'s run-wide flatness statistics, computed ONCE by the report
    section that calls this function (never here, and never re-derived per
    panel) and passed through unchanged, so every panel's clause quotes the
    same population numbers the section's own flatness table does. `None`
    (the default) renders every panel with no clause -- the caller's own
    degrade-gracefully state, not a failure of this function.
    """
    if not entry or not entry.get("scorable"):
        return "<span class='spark-missing'>not measured</span>"
    n_top_series = entry.get("n_top_series")
    n_top_series = n_top_series if isinstance(n_top_series, int) else None
    # `max_series=None` renders every pair the ablation pass kept. The old
    # hardcoded 3 silently truncated a run started with a larger
    # `--keep-forecasts`, and -- because the exemplar column beside it drew
    # its own fixed 4 -- made the two columns disagree about how many series
    # this feature was examined on (ROADMAP.md sec 28's item 7).
    fcs = list(entry.get("forecasts") or [])
    if max_series is not None:
        fcs = fcs[:max_series]
    if not fcs:
        return "<span class='spark-missing'>no forecasts kept</span>"
    # ROADMAP.md sec 32.7 Item H2: lazy per CLAUDE.md sec 11.52 -- `report.py`
    # imports this module (a two-module cycle), so `_flat_clause` is pulled
    # in here rather than at module scope.
    from .derived import ablation_panel_summary as _ablation_panel_summary
    from .report import _flat_clause
    cells = []
    for f in fcs:
        ctx, tgt = list(f.get("context") or []), list(f.get("target") or [])
        if not ctx or not tgt:
            continue
        act = float(f.get("activation", float("nan")))
        act_str = f"activation {act:.2f}"
        # ROADMAP.md sec 32.7c Item I2: the decoder column this activation
        # is the length of is unit-norm (`TopKSAE.normalize_decoder_`), so
        # the raw number means nothing without what it is a fraction OF --
        # median hidden-state norm at this target differed 55x across four
        # real targets checked, while the raw activations differed only 5x
        # in the opposite direction. Computed once per target in the SAE
        # stage (`sae/train.py`) and passed in here, never recomputed from
        # a store this function has no handle to.
        if (median_hidden_norm is not None and median_hidden_norm > 0
                and np.isfinite(act)):
            act_str += f" ({act / median_hidden_norm:.0%} of a typical hidden state)"
        # ROADMAP.md sec 32.6 Item E/F: the with/without difference, drawn on
        # its own scale and printed as a number. `None` (not zero-padding) on
        # a missing or mismatched-length pair, since a fabricated zero-filled
        # diff would draw a flat line that means "no effect measured" when
        # it should mean "not computed at all".
        with_arr = f.get("with_feature")
        without_arr = f.get("without_feature")
        diff_arr = None
        diff_html = ""
        if (with_arr is not None and without_arr is not None
                and len(with_arr) == len(without_arr) and len(with_arr) > 0):
            diff_arr = np.asarray(with_arr, dtype=np.float64) - np.asarray(without_arr, dtype=np.float64)
            ctx_sd = float(np.nanstd(np.asarray(ctx, dtype=np.float64))) if ctx else 0.0
            max_abs = float(np.nanmax(np.abs(diff_arr)))
            mean_abs = float(np.nanmean(np.abs(diff_arr)))
            if ctx_sd > 0:
                diff_html = (f" · Δ {max_abs:.3g} ({(max_abs / ctx_sd) * 100:.2g}% of context sd)")
                meta_diff = (f" · max|Δ| {max_abs:.3g} ({max_abs / ctx_sd:.2f}x context sd), "
                            f"mean|Δ| {mean_abs:.3g} ({mean_abs / ctx_sd:.2f}x context sd)")
            else:
                diff_html = f" · Δ {max_abs:.3g}"
                meta_diff = f" · max|Δ| {max_abs:.3g}, mean|Δ| {mean_abs:.3g}"
        else:
            meta_diff = ""
        # ROADMAP.md sec 32.7 Item H2: fires only when this panel's OWN
        # forecast is flat and a population was supplied -- `_flat_clause`
        # itself enforces both, so a caller that passes `population=None`
        # (the default; not every call site has it) gets every panel back
        # unchanged.
        panel = _ablation_panel_summary(f)
        flat_clause = _flat_clause(panel, population)
        meta = (f"{html.escape(str(f.get('series_id', '')))} · "
                f"{act_str} · "
                f"solid = with the feature, dashed = with it removed"
                f"{html.escape(meta_diff)}"
                + (f" · {html.escape(flat_clause)}" if flat_clause else ""))
        svg = sparkline_svg(ctx + tgt, len(ctx),
                            clean=f.get("with_feature"),
                            patched=f.get("without_feature"),
                            unpatched=f.get("unpatched"),
                            diff=diff_arr,
                            title=f"series {f.get('series_id', '')} with and "
                                  f"without feature {entry.get('feature')}")
        cells.append(f"<figure class='spark-cell' data-meta=\"{meta}\" "
                     f"title='Click to enlarge'>{svg}"
                     f"<figcaption>with · without{html.escape(diff_html)}</figcaption></figure>")
    if not cells:
        return "<span class='spark-missing'>no forecasts kept</span>"
    badge = ""
    if n_top_series is not None:
        badge = (f"<span class='muted n-badge' "
                  f"title='fires on {n_top_series} series in this sample'>"
                  f"n={n_top_series}</span> ")
    note = ""
    if len(cells) < 3:
        if n_top_series is not None and n_top_series < 3:
            clause = (f"fires on only {n_top_series} series in this sample "
                      f"— every number in this row is an "
                      f"n-of-{n_top_series}.")
        elif n_top_series is not None:
            plural = "s" if len(cells) != 1 else ""
            clause = (f"fires on {n_top_series} series in this sample; only "
                      f"{len(cells)} forecast pair{plural} kept here "
                      f"(raise --keep-forecasts to see more).")
        else:
            clause = None
        if clause:
            note = f"<p class='muted spark-note'>{html.escape(clause)}</p>"
    body = f"{badge}<div class='spark-row'>{''.join(cells)}</div>{note}"
    n_clearing = entry.get("n_channels_clearing")
    if isinstance(n_clearing, int) and n_clearing == 0:
        summary = ("1 feature whose ablation did not move any channel past "
                   "its own null — click to view its forecasts anyway")
        return (f"<details class='spark-collapsed'><summary class='muted'>"
                f"{html.escape(summary)}</summary>{body}</details>")
    return body


def feature_table_html(cards: list, series_lookup, context_len: int,
                       descriptions: Optional[dict] = None,
                       ablations: Optional[dict] = None,
                       overlay_series: Optional[int] = None,
                       median_hidden_norm: Optional[float] = None,
                       population: Optional[dict] = None) -> str:
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

    `overlay_series` caps the with/without column; `None` (the default)
    draws every pair the artifact kept. The caller sets the exemplar
    column's own count from the same number, so the two columns examine
    the same series rather than differing by however far apart their two
    hardcoded defaults happened to sit.

    `median_hidden_norm` (ROADMAP.md sec 32.7c Item I2) is this target's
    median hidden-state norm, computed once in the SAE stage and passed
    through to `ablation_cell` unchanged -- this function has no store
    handle to compute it from, and a report-only rerun has none either.

    `population` (ROADMAP.md sec 32.7 Item H2) is likewise passed through to
    `ablation_cell` unchanged -- the run-wide flatness statistics computed
    once by the calling report section, never here.
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
            label_html = ablation_effect_html(abl)
            row += ("<td>" + (label_html if label_html else
                              "<span class='muted'>not a candidate</span>") + "</td>")
        row += f"<td>{ex_html}</td>"
        if has_abl:
            row += f"<td>{ablation_cell(abl, overlay_series, median_hidden_norm, population)}</td>"
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


def metric_legend_html(columns: Sequence[str], heading: str = "") -> str:
    """A definition table for the COLUMNS a rendered table actually printed.

    Added 2026-09-11 on user review: `ΔMASE sign` and `alignment mean abs
    rho` each appeared exactly once in the whole rendered report, as a bare
    `<th>`, with no definition anywhere in the document -- and those were
    the two the reader could not act on. `term_legend_html` above could not
    serve this: its columns are "what it measures / what a high value
    means", which is the right shape for a property of the DATA and the
    wrong shape for a statistic this pipeline computed, where the missing
    half is the arithmetic.

    Built from the headers passed in, never from `METRIC_DEFS.keys()`, for
    the same reason `term_legend_html` is built from the figure's own axis:
    a legend listing a column the table did not print describes a different
    run. A column with no recorded definition is simply absent -- an
    incomplete legend is visible where a placeholder gloss is not.
    """
    from ..sae.vocab import describe_metric

    seen, rows = set(), []
    for col in columns:
        key = str(col).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        d = describe_metric(key)
        if d is None:
            continue
        rows.append(
            f"<tr><td><code>{html.escape(key)}</code></td>"
            f"<td>{html.escape(d.what)}</td>"
            f"<td>{html.escape(d.how)}</td></tr>")
    if not rows:
        return ""
    return (f"{heading}<table class='tbl'><thead><tr><th>Column</th>"
            f"<th>What it means, and why it matters</th>"
            f"<th>How it is calculated</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")
