"""Regenerate the data figures of the findings paper from the published seven-model run.

The figures are drawn at the paper's text width (6.0 in) so that LaTeX includes them
at scale 1, which keeps every label at its stated point size in print. Data loading,
model colors and channel labels are shared with ``paper/figures/make_figures.py``;
the labels here use the terminology of the findings paper.
"""

import argparse
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


def main():
    """Write heldout.pdf, families.pdf and funnel.pdf next to this script."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=base.DEFAULT_RUN)
    args = ap.parse_args()
    _style()
    fig_heldout(args.run)
    fig_families(args.run)
    fig_funnel(args.run)


if __name__ == "__main__":
    main()
