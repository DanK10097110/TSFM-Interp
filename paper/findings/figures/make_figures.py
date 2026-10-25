"""Regenerate the data figures of the findings paper from the published seven-model run.

The figures are drawn at the paper's text width (6.0 in) so that LaTeX includes them
at scale 1, which keeps every label at its stated point size in print. Data loading,
model colors and channel labels are shared with ``paper/figures/make_figures.py``;
the labels here use the terminology of the findings paper. The geometry figure reads
``data/input_control.json``, the output of ``tsfm_lens/run_input_control_cka.py``.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures"))
import make_figures as base  # noqa: E402

OUT = Path(__file__).resolve().parent
TEXT_WIDTH = 6.0
PAD = 0.05
OK, BAD, ACCENT, INK = "#2e7d32", "#c62828", "#c0602a", "#444444"


def _style():
    """Serif text sized for print at scale 1: 9 pt body, 8.5 pt ticks."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
        "savefig.pad_inches": PAD, "pdf.fonttype": 42,
    })


def _save(fig, name):
    """Shrink the canvas until the tight bounding box is at most the text width, then save."""
    for _ in range(6):
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width
        target = TEXT_WIDTH - 2 * PAD
        if w <= target + 0.01:
            break
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(fw - (w - target) - 0.02, fh)
        if fig.get_layout_engine() is not None:
            fig.get_layout_engine().execute(fig)
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_heldout(run):
    """Development estimate against the held-out interval for each replicated claim type."""
    conf = base._load(run, "confirm/confirmation.json")
    fig = plt.figure(figsize=(TEXT_WIDTH, 5.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    tests = conf["l3_replication"]["tests"]
    y = np.arange(len(tests))[::-1]
    for yi, t in zip(y, tests):
        p = t["private"]
        col = INK if t["replicates"] else BAD
        ax.plot([p["lo"], p["hi"]], [yi, yi], color=col, lw=1.8)
        ax.plot(p["value"], yi, "o", color=col, ms=4)
        ax.plot(t["dev_rho"], yi, "x", color=ACCENT, ms=6, mew=1.4)
    ax.set_yticks(y, [t["corruption"].replace("_", " ") for t in tests])
    ax.axvline(0, color="#bbbbbb", lw=0.7)
    ax.set_xlabel("Spearman $\\rho$")
    n_rep = sum(t["replicates"] for t in tests)
    ax.set_title(f"(a) Depth profiles ({n_rep} of {len(tests)})", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    tr = conf["concept_replication"]["transfer"]["tests"]
    order = np.argsort([t["private_auc"] for t in tr])
    for i, j in enumerate(order):
        t = tr[j]
        ax.plot([t["private_null_p95"], t["private_auc"]], [i, i], color="#dddddd", lw=1.2)
        ax.plot(t["private_null_p95"], i, "|", color="#888888", ms=7, mew=1.2)
        ax.plot(t["private_auc"], i, "o", color=INK, ms=3.5)
        ax.plot(t["dev_auc"], i, "x", color=ACCENT, ms=5, mew=1.2)
    ax.set_yticks([])
    ax.set_xlim(0.5, 1.005)
    ax.set_xlabel("AUC")
    n_conf = sum(bool(t["confirmed"]) for t in tr)
    ax.set_title(f"(b) Concept transfer ({n_conf} of {len(tr)})", loc="left")

    axd = fig.add_subplot(gs[1, :])
    cr = conf["concept_replication"]
    fams = [(lab, cr[key]) for lab, key in (
        ("single-feature causal effect", "causal"),
        ("atlas concept", "atlas"),
        ("same effect on same inputs", "agreement"),
        ("concept transfer", "transfer"),
        ("atlas structure", "structure"),
        ("failure prediction", "reliability_u1"),
    ) if isinstance(cr.get(key), dict) and "n_confirmed" in cr[key]]
    y = np.arange(len(fams))[::-1]
    for yi, (_, f) in zip(y, fams):
        n_ok = f["n_confirmed"]
        n_bad = f["n_tested"] - f["n_confirmed"]
        n_nt = f.get("n_not_testable", f["n_registered"] - f["n_tested"])
        axd.barh(yi, n_ok, color=OK, height=0.65)
        axd.barh(yi, n_bad, left=n_ok, color=BAD, height=0.65)
        axd.barh(yi, n_nt, left=n_ok + n_bad, color="#d9d9d9", height=0.65, hatch="///", ec="#999999", lw=0.3)
        axd.text(n_ok + n_bad + n_nt + 0.6, yi, f"{n_ok} of {f['n_tested']}", va="center")
    axd.set_yticks(y, [f[0] for f in fams])
    axd.set_xlim(0, max(f["n_registered"] for _, f in fams) * 1.2)
    axd.set_xlabel("registered claims")
    axd.set_title("(c) Registered claims by type", loc="left")
    fig.legend(handles=[
        plt.Line2D([], [], marker="x", ls="", color=ACCENT, label="development estimate"),
        plt.Line2D([], [], marker="o", ls="-", color=INK, label="held-out estimate and 95% interval"),
        plt.Rectangle((0, 0), 1, 1, color=OK, label="confirmed"),
        plt.Rectangle((0, 0), 1, 1, color=BAD, label="not confirmed"),
        plt.Rectangle((0, 0), 1, 1, fc="#d9d9d9", ec="#999999", hatch="///", label="not testable"),
    ], loc="outside lower center", ncol=3, frameon=False)
    _save(fig, "heldout.pdf")


def fig_families(run):
    """Effect profile of each causal-effect family and its membership by model."""
    fam = base._load(run, "sae/concept_families.json")["families"]
    fam = sorted(fam, key=lambda f: -f["n_members"])
    chans = list(base.CHANNEL_LABELS)
    mat = np.array([[f["directed_profile_null_units"][c] for c in chans] for f in fam])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 4.3), layout="constrained",
                                  gridspec_kw={"width_ratios": [3.0, 1.0]})
    undirected = [j for j, c in enumerate(chans) if c.startswith("horizon_shape")]
    directed = mat.copy()
    directed[:, undirected] = np.nan
    lim = np.nanmax(np.abs(directed))
    im = ax.imshow(directed, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    dist = np.full_like(mat, np.nan)
    dist[:, undirected] = np.abs(mat[:, undirected])
    ax.imshow(dist, cmap="Greys", vmin=0, vmax=2 * np.nanmax(dist), aspect="auto")
    for i, f in enumerate(fam):
        for j, c in enumerate(chans):
            if c in f["cleared_channels"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.0))
    ax.set_xticks(range(len(chans)), [base.CHANNEL_LABELS[c].replace("\n", " ") for c in chans], rotation=45, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(fam)), [f"{f['title']} ({f['n_members']})" for f in fam])
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, aspect=25)
    cb.set_label("effect (null units)")
    ax.set_title("(a) Effect on each forecast property", loc="left")
    models = base._models_in({m for f in fam for m in f["models"]})
    left = np.zeros(len(fam))
    for m in models:
        v = np.array([f["models"].get(m, 0) / f["n_members"] for f in fam])
        ax2.barh(range(len(fam)), v, left=left, color=base._color(m, models), label=m, height=0.7)
        left += v
    ax2.set_ylim(len(fam) - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("share of members")
    ax2.set_title("(b) Membership", loc="left")
    ax2.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    fig.legend(handles=ax2.get_legend_handles_labels()[0], loc="outside lower center", ncol=4, frameon=False,
               handlelength=1.0, columnspacing=1.0)
    _save(fig, "families.pdf")


def fig_funnel(run):
    """Concept candidates at each level of evidence, and verdicts of the same-effect test."""
    st = base._load(run, "sae/concept_stage.json")
    at, sb, pr, sia = st["atlas"], st["stability"], st["profiles"], st["shared_input_agreement"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 4.3), layout="constrained")
    stages = [
        ("causal features", at["n_features"]),
        ("assigned to a cross-model concept", at["n_assigned"]),
        ("cross-model concepts", at["n_concepts"]),
        ("stable across dictionary seeds", sb["n_stable"]),
        ("members from two or more models", at["n_multi_model"]),
        ("same effect and same inputs",
         sum(v for k, v in pr["sharing_class_counts"].items() if k.startswith(("shared", "partially")))),
    ]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [s[1] for s in stages], color="#5a7d9a", height=0.65)
    for yi, s in zip(y, stages):
        ax.text(s[1] + 3, yi, str(s[1]), va="center")
    ax.set_yticks(y, [s[0] for s in stages])
    ax.set_xlim(0, at["n_features"] * 1.12)
    ax.set_title("(a) Concept candidates", loc="left")
    vc = sia["verdict_counts"]
    keys = ["same causal effect", "level only", "shape only", "no specific agreement", "acts differently", "not scorable"]
    cols = [OK, "#7cb342", "#aed581", "#bdbdbd", BAD, "#eeeeee"]
    yy = np.arange(len(keys))[::-1]
    ax2.barh(yy, [vc[k] for k in keys], color=cols, ec="#999999", lw=0.4, height=0.65)
    for yi, k in zip(yy, keys):
        ax2.text(vc[k] + 8, yi, str(vc[k]), va="center")
    ax2.set_yticks(yy, keys)
    ax2.set_xlim(0, max(vc.values()) * 1.12)
    ax2.set_xlabel("tests")
    ax2.set_title(f"(b) Same-effect test on shared inputs ({sia['n_tests']:,} tests)", loc="left")
    _save(fig, "funnel.pdf")


def _short(model):
    """Model name without the size suffix, for pair labels."""
    return model.replace("-Base", "")


def fig_geometry(path):
    """Peak CKA of each model pair before and after removing the input, and input CKA by depth."""
    d = json.loads(path.read_text(encoding="utf-8"))
    pairs = sorted(d["pairs"], key=lambda p: p["cka"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 5.0), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    for i, p in enumerate(pairs):
        z = p["zscore"]
        ax.plot([z["residual_cka"], p["cka"]], [i, i], color="#cccccc", lw=1.2, zorder=1)
        ax.plot(p["cka"], i, "o", mfc="white", mec=INK, ms=4.5, zorder=2)
        ax.plot(z["residual_cka"], i, "o", color=ACCENT, ms=4.5, zorder=3)
        ax.plot(z["residual_null_p95"], i, "|", color="#888888", ms=7, mew=1.2, zorder=2)
    ax.set_yticks(range(len(pairs)), [f"{_short(p['model_a'])} / {_short(p['model_b'])}" for p in pairs],
                  fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("CKA")
    ax.set_title("(a) Peak CKA of each model pair", loc="left")
    marks = [
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="as measured"),
        plt.Line2D([], [], marker="o", ls="", color=ACCENT, label="input removed"),
        plt.Line2D([], [], marker="|", ls="", color="#888888", mew=1.2, ms=7, label="null, 95th percentile"),
    ]

    models = base._models_in(set(d["depth_profile"]))
    for m in models:
        prof = d["depth_profile"][m]
        idx = np.array([int(r["layer"].rsplit(".", 1)[1]) for r in prof])
        ax2.plot(idx / idx.max(), [r["cka_input_zscore"] for r in prof], "-o", ms=2.5, lw=1.3,
                 color=base._color(m, models), label=_short(m))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("relative depth")
    ax2.set_ylabel("CKA with the input")
    ax2.set_title("(b) Similarity to the input", loc="left")
    fig.legend(handles=marks + ax2.get_legend_handles_labels()[0], loc="outside lower center", ncol=4,
               frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
    _save(fig, "geometry.pdf")


EXEMPLARS = {
    "shared": {"concept": 6, "series": "6400924ee89f0b4d",
               "parts": [("Chronos-2", "encoder.block.9", 3851), ("Chronos-Bolt", "encoder.block.2", 3821)]},
    "convergent": {"concept": 8,
                   "parts": [("Chronos-Bolt", "encoder.block.2", 3556, "26866501cb0bbc79"),
                             ("TimesFM", "stacked_xf.6", 1568, "0455fcd12e1372d6")]},
}


def extract_exemplars(run, out):
    """Copy the stored ablation forecasts and top-series overlaps of the two example concepts into ``out``.

    The forecasts are those the ablation battery kept for each feature, so nothing is recomputed. Needs the
    per-layer ablation records of the full run (``tsfm_lens`` run directory, not the published example).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tsfm_lens"))
    from tsfm_lens.sae import ablation_run

    def forecast(model, layer, feature, series):
        rec = json.loads(ablation_run.ablation_path(run, model, layer).read_text(encoding="utf-8"))
        cand = next(c for c in rec["candidates"] if c["feature"] == feature)
        fc = next(f for f in cand["forecasts"] if f["series_id"] == series)
        return {"model": model, "layer": layer, "feature": feature, "series_id": series,
                "context": fc["context"], "target": fc["target"],
                "with_feature": fc["with_feature"], "without_feature": fc["without_feature"]}

    prof = json.loads((run / "sae" / "concept_profiles.json").read_text(encoding="utf-8"))["concepts"]
    sh, cv = EXEMPLARS["shared"], EXEMPLARS["convergent"]
    data = {
        "shared": {"concept": prof[sh["concept"]]["name"],
                   "forecasts": [forecast(m, l, f, sh["series"]) for m, l, f in sh["parts"]],
                   "overlaps": prof[sh["concept"]]["cross_model_pairs"],
                   "top_k": len(prof[sh["concept"]]["parts"][0]["top_series_ids"])},
        "convergent": {"concept": prof[cv["concept"]]["name"],
                       "forecasts": [forecast(m, l, f, s) for m, l, f, s in cv["parts"]],
                       "overlaps": prof[cv["concept"]]["cross_model_pairs"]},
    }
    out.write_text(json.dumps(data), encoding="utf-8")


def _exemplar_panel(fig, cell, fc, title, first):
    """Context tail, outcome and the forecast with and without the feature, with the change below."""
    sub = cell.subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.08)
    ax, axd = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    ctx = np.asarray(fc["context"], dtype=float)
    sd = ctx.std()
    tail = ctx[-96:]
    t0 = np.arange(-len(tail), 0)
    h = np.arange(len(fc["target"]))
    ax.plot(t0, tail, color="#9e9e9e", lw=0.9)
    ax.plot(h, fc["target"], color="#cfcfcf", lw=0.9)
    ax.plot(h, fc["with_feature"], color=INK, lw=1.4)
    ax.plot(h, fc["without_feature"], color=ACCENT, lw=1.4, ls=(0, (3, 1.5)))
    ax.axvline(-0.5, color="#bbbbbb", lw=0.6)
    ax.set_xticks([])
    ax.set_title(title, loc="left")
    diff = (np.asarray(fc["without_feature"]) - np.asarray(fc["with_feature"])) / sd
    axd.axhline(0, color="#bbbbbb", lw=0.6)
    axd.plot(h, diff, color=ACCENT, lw=1.4)
    axd.set_xlim(ax.get_xlim())
    lim = max(0.05, 1.3 * np.abs(diff).max())
    axd.set_ylim(-lim, lim)
    if first:
        axd.set_ylabel("change\n(context SD)", fontsize=8)
    axd.set_xlabel("time step")
    return ax


def fig_exemplar(path):
    """One concept with shared inputs and one with the same effect on different inputs, from stored ablations."""
    d = json.loads(path.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(TEXT_WIDTH, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.72])
    sh, cv = d["shared"], d["convergent"]
    for j, fc in enumerate(sh["forecasts"]):
        _exemplar_panel(fig, gs[0, j], fc, f"({'ab'[j]}) {_short(fc['model'])}", j == 0)
    ax = fig.add_subplot(gs[0, 2])
    models = []
    for p in sh["overlaps"]:
        for m in (p["model_a"], p["model_b"]):
            if m not in models:
                models.append(m)
    mat = np.full((len(models), len(models)), np.nan)
    for p in sh["overlaps"]:
        i, j = models.index(p["model_a"]), models.index(p["model_b"])
        mat[i, j] = mat[j, i] = p["top_k_overlap"]
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=sh["top_k"], aspect="equal")
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center",
                        color="white" if mat[i, j] > sh["top_k"] / 2 else INK)
    labels = [_short(m).replace("Chronos-", "C-") for m in models]
    ax.set_xticks(range(len(models)), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models)), labels)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    exp = sh["overlaps"][0]["top_k_overlap_expected"]
    ax.set_title(f"(c) Shared top-{sh['top_k']} series", loc="left")
    ax.set_xlabel(f"chance: {exp:.2f}")
    for j, fc in enumerate(cv["forecasts"]):
        _exemplar_panel(fig, gs[1, j], fc, f"({'de'[j]}) {_short(fc['model'])}", j == 0)
    axl = fig.add_subplot(gs[1, 2])
    axl.axis("off")
    axl.legend(handles=[
        plt.Line2D([], [], color="#9e9e9e", label="context"),
        plt.Line2D([], [], color="#cfcfcf", label="outcome"),
        plt.Line2D([], [], color=INK, lw=1.4, label="forecast"),
        plt.Line2D([], [], color=ACCENT, lw=1.4, ls=(0, (3, 1.5)), label="feature removed"),
    ], loc="center left", frameon=False)
    _save(fig, "exemplar.pdf")


def main():
    """Write heldout.pdf, families.pdf, funnel.pdf, geometry.pdf and exemplar.pdf next to this script."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=base.DEFAULT_RUN)
    ap.add_argument("--input-control", type=Path, default=OUT / "data" / "input_control.json")
    ap.add_argument("--exemplars", type=Path, default=OUT / "data" / "exemplars.json")
    ap.add_argument("--extract-exemplars-from", type=Path, default=None,
                    help="full tsfm_lens run directory; rewrites --exemplars from its stored ablation records")
    args = ap.parse_args()
    if args.extract_exemplars_from is not None:
        extract_exemplars(args.extract_exemplars_from, args.exemplars)
    _style()
    fig_exemplar(args.exemplars)
    fig_heldout(args.run)
    fig_families(args.run)
    fig_funnel(args.run)
    fig_geometry(args.input_control)


if __name__ == "__main__":
    main()
