"""Plot physical forecast recovery for the six writes on the primary daily cohort."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).resolve().parent / "results"
METHODS = [
    ("hard", "Direct SAE re-encoding", "#656a70"),
    ("projection", "Hidden projection", "#656a70"),
    ("active_same", "Active span · original length", "#126758"),
    ("random_same", "Random span · original length", "#87512e"),
    ("active_cap", "Active span · usual cap", "#126758"),
    ("random_cap", "Random span · usual cap", "#87512e"),
]


def main():
    data = json.loads((RESULTS / "span_inverse.json").read_text())["bolt"]["daily"]
    panels = [
        ("early_dense", "TimesFM reference", "Leads 1–8"),
        ("early_source", "Bolt reference", "Leads 1–8"),
        ("late_source", "Bolt reference", "Leads 9–16 · withheld"),
    ]
    fig, axes = plt.subplots(1, len(panels), sharey=True, figsize=(11.2, 4.6))
    for axis, (key, reference, leads) in zip(axes, panels):
        axis.set_title(f"{reference}\n{leads}", fontsize=11, loc="left", pad=10)
        axis.axvline(0, color="#849096", lw=1)
        axis.axvline(1, color="#849096", lw=0.9, ls=":")
        for i, (method, _, color) in enumerate(METHODS):
            result = data[key][method]
            low, high = result["crossed95"]
            y = len(METHODS) - i
            axis.hlines(y, low, high, color=color, lw=2)
            axis.scatter(result["value"], y, color=color, s=30, zorder=3)
        axis.set_xlim(-1.35, 1.10)
        axis.set_xticks([-1, -.5, 0, .5, 1])
        axis.tick_params(length=0, labelsize=9)
        axis.spines[["right", "top", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#abb1b4")
        axis.grid(axis="y", color="#eceeed", zorder=0)
    axes[0].set_yticks(np.arange(len(METHODS), 0, -1), [m[1] for m in METHODS])
    axes[0].set_ylim(.45, len(METHODS) + .55)
    fig.subplots_adjust(left=.235, right=.98, top=.75, bottom=.21, wspace=.15)
    fig.text(.025, .94, "Can the same 32 directions reproduce a steering effect?",
             fontsize=17, weight="bold")
    fig.text(.025, .88, "Chronos-Bolt → TimesFM · 360 daily firing events across 15 donor features",
             fontsize=11)
    fig.text(.235, .105, "Centered recovery     0 = unchanged forecast     1 = perfect recovery", fontsize=10)
    fig.text(.025, .035, "Bars show 95% intervals resampling histories and features. All six methods use the same events.",
             color="#4e565a", fontsize=9)
    fig.savefig(RESULTS / "span_inverse.png", dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
