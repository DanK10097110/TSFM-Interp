"""Regenerate every data figure in the paper from the published example run.

The paper's quantitative figures are not drawn by hand: each one is a pure
function of an artifact committed under ``examples/concept_atlas_v2/run``, so a
reader can rerun this script (or point it at their own run with ``--run``) and
obtain the same PDFs. Architecture diagrams are TikZ in ``main.tex``; report
screenshots are produced by ``screenshot_report.sh``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "examples" / "concept_atlas_v2" / "run"
OUT = Path(__file__).resolve().parent

MODEL_COLORS = {"TimesFM": "#2f6f8f", "Chronos-2": "#9b4f7f", "Sundial": "#3d7f5a", "Chronos-Bolt": "#8a6a3a"}
CHANNEL_LABELS = {
    "trend": "trend", "seasonal": "seasonal", "spectral_centroid": "spectral\ncentroid",
    "level": "level", "dispersion": "dispersion", "horizon_shape_near": "shape\n(near)",
    "horizon_shape_far": "shape\n(far)", "mase": "MASE", "flatness": "flatness",
}


def _load(run, rel):
    """Read one JSON artifact of the run."""
    return json.loads((run / rel).read_text(encoding="utf-8"))


def _style():
    """Journal-friendly defaults: small serif text, no top/right spines."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def fig_heldout(run):
    """Dev estimate vs held-out private interval for every replicated claim kind."""
    conf = _load(run, "confirm/confirmation.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.2, 1.0, 1.1]})

    ax = axes[0]
    tests = conf["l3_replication"]["tests"]
    y = np.arange(len(tests))[::-1]
    for yi, t in zip(y, tests):
        p = t["private"]
        ax.plot([p["lo"], p["hi"]], [yi, yi], color="#444", lw=1.6)
        ax.plot(p["value"], yi, "o", color="#444", ms=3.5)
        ax.plot(t["dev_rho"], yi, "x", color="#c0602a", ms=5, mew=1.3)
    ax.set_yticks(y, [t["corruption"].replace("_", " ") for t in tests])
    ax.axvline(0, color="#bbb", lw=0.6)
    ax.set_xlabel("Spearman $\\rho$ of depth profiles")
    ax.set_title("(a) L3: 9/9 replicate")

    ax = axes[1]
    tr = conf["concept_replication"]["transfer"]["tests"]
    order = np.argsort([t["private_auc"] for t in tr])
    for i, j in enumerate(order):
        t = tr[j]
        ax.plot([t["private_null_p95"], t["private_auc"]], [i, i], color="#ddd", lw=1.0)
        ax.plot(t["private_null_p95"], i, "|", color="#888", ms=5)
        ax.plot(t["private_auc"], i, "o", color="#444", ms=3)
        ax.plot(t["dev_auc"], i, "x", color="#c0602a", ms=4, mew=1.1)
    ax.set_yticks([])
    ax.set_xlim(0.5, 1.005)
    ax.set_xlabel("top-$k$ input AUC")
    ax.set_title("(b) transfer: 20/20")
    ax.text(0.52, len(tr) - 1.2, "| = null p95", fontsize=6.5, color="#666")

    ax = axes[2]
    rows = [
        ("overall $\\Delta$MASE\n(TimesFM $-$ Chronos-2)", conf["overall"]["mean"], conf["overall"]["lo"], conf["overall"]["hi"], None),
        ("mixture family $\\Delta$MASE", conf["tests"][0]["mean"], conf["tests"][0]["lo"], conf["tests"][0]["hi"], None),
        ("peak CKA\n(TimesFM xf.4, Chronos-2 b.7)", conf["cka_replication"]["private"]["value"], conf["cka_replication"]["private"]["lo"],
         conf["cka_replication"]["private"]["hi"], conf["cka_replication"]["dev_cka"]),
    ]
    for yi, (lab, m, lo, hi, dev) in zip([2, 1, 0], rows):
        ax.plot([lo, hi], [yi, yi], color="#444", lw=1.6)
        ax.plot(m, yi, "o", color="#444", ms=3.5)
        if dev is not None:
            ax.plot(dev, yi, "x", color="#c0602a", ms=5, mew=1.3)
    ax.axvline(0, color="#bbb", lw=0.6)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.yaxis.tick_right()
    ax.set_xlabel("private estimate (95% CI)")
    ax.set_title("(c) accuracy and geometry")
    fig.legend(handles=[
        plt.Line2D([], [], marker="x", ls="", color="#c0602a", label="dev (exploratory) estimate"),
        plt.Line2D([], [], marker="o", ls="-", color="#444", label="private (confirmatory) estimate and CI"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "heldout.pdf")
    plt.close(fig)


def fig_families(run):
    """What each causal-effect family does (directed, null units) and which models populate it."""
    fam = _load(run, "sae/concept_families.json")["families"]
    fam = sorted(fam, key=lambda f: -f["n_members"])
    chans = list(CHANNEL_LABELS)
    mat = np.array([[f["directed_profile_null_units"][c] for c in chans] for f in fam])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [2.6, 1.0]})
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
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=0.9))
    ax.set_xticks(range(len(chans)), [CHANNEL_LABELS[c].replace("\n", " ") for c in chans], rotation=40, ha="right")
    ax.set_yticks(range(len(fam)), [f"{f['title']} ({f['n_members']})" for f in fam])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("effect (null-p95 units)")
    ax.set_title("(a) what each family does (boxed: clears null)")
    models = list(MODEL_COLORS)
    left = np.zeros(len(fam))
    for m in models:
        v = np.array([f["models"].get(m, 0) / f["n_members"] for f in fam])
        ax2.barh(range(len(fam)), v, left=left, color=MODEL_COLORS[m], label=m, height=0.7)
        left += v
    ax2.set_ylim(len(fam) - 0.5, -0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("share of members by model")
    ax2.set_title("(b) membership")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "families.pdf")
    plt.close(fig)


def fig_funnel(run):
    """How many candidate concepts survive each rung of the evidence ladder, and the L5 verdicts."""
    st = _load(run, "sae/concept_stage.json")
    at, sb, pr, sia = st["atlas"], st["stability"], st["profiles"], st["shared_input_agreement"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.2, 1.0]})
    stages = [
        ("causal SAE features\n(clear random-direction null)", at["n_features"]),
        ("assigned to an atlas concept", at["n_assigned"]),
        ("atlas concepts ($\\geq$3 members)", at["n_concepts"]),
        ("seed-stable concepts", sb["n_stable"]),
        ("members from $\\geq$2 models", at["n_multi_model"]),
        ("same effect and same inputs\n(shared + partially shared)",
         sum(v for k, v in pr["sharing_class_counts"].items() if k.startswith(("shared", "partially")))),
    ]
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [s[1] for s in stages], color="#5a7d9a", height=0.65)
    for yi, s in zip(y, stages):
        ax.text(s[1] + 4, yi, str(s[1]), va="center", fontsize=7)
    ax.set_yticks(y, [s[0] for s in stages])
    ax.set_xlim(0, at["n_features"] * 1.15)
    ax.set_title("(a) concept candidates per evidence rung")
    vc = sia["verdict_counts"]
    keys = ["same causal effect", "level only", "shape only", "no specific agreement", "acts differently", "not scorable"]
    cols = ["#2e7d32", "#7cb342", "#aed581", "#bdbdbd", "#c62828", "#eeeeee"]
    ax2.barh(range(len(keys))[::-1], [vc[k] for k in keys], color=cols, ec="#999", lw=0.4, height=0.65)
    for yi, k in zip(range(len(keys))[::-1], keys):
        ax2.text(vc[k] + 10, yi, str(vc[k]), va="center", fontsize=7)
    ax2.set_yticks(range(len(keys))[::-1], keys)
    ax2.set_xlim(0, max(vc.values()) * 1.18)
    ax2.set_title(f"(b) shared-input causal agreement, {sia['n_tests']} tests")
    fig.tight_layout()
    fig.savefig(OUT / "funnel.pdf")
    plt.close(fig)


def fig_similarity(run):
    """Per-metric rank of every model pair: the metrics disagree, so they are never pooled."""
    sim = _load(run, "report/model_similarity.json")
    ranks = sim["consensus"]["rank_table"]
    pairs = [p["key"] for p in sim["pairs"]]
    metrics = list(ranks)
    mat = np.array([[ranks[m][p] for p in pairs] for m in metrics])
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto", vmin=1, vmax=len(pairs))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:g}", ha="center", va="center", fontsize=7,
                    color="w" if mat[i, j] < 3.5 else "k")
    abbr = {"TimesFM": "TFM", "Chronos-2": "C-2", "Sundial": "Sun", "Chronos-Bolt": "Bolt"}
    ax.set_xticks(range(len(pairs)), ["\n".join(abbr[m] for m in p.split("|")) for p in pairs])
    ax.set_yticks(range(len(metrics)), [sim["metrics"][m]["label"] for m in metrics])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("rank (1 = most similar)")
    w = sim["consensus"]["kendall_w_all"]
    ax.set_title(f"rank agreement across {w['n_metrics']} metrics: Kendall's W = {w['w']:.2f} (permutation p = {w['p']:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "similarity.pdf")
    plt.close(fig)


def main():
    """Parse ``--run`` and write every figure next to this script."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    _style()
    for fn in (fig_heldout, fig_families, fig_funnel, fig_similarity):
        fn(args.run)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
