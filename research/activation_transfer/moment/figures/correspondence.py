"""Plot measured daily-series forecast recovery and a selected real-temperature intervention."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
MOMENT = HERE.parent
GROUPS = MOMENT.parent / "shape_transfer" / "cross_seed_groups" / "results"


def read(path):
    return json.loads(path.read_text())


def scores():
    rows = {}
    for size in ("small", "large"):
        data = read(MOMENT / "dictionary_transfer" / "results" /
                    f"{size}_22112201_to_{size}_group_scores.json")["daily"]["native_early_shape"]
        rows["MOMENT-" + size] = [data[k]["value"] for k in ("one", "four", "sixteen")]
    for model, label in (("bolt", "Chronos-Bolt"), ("timesfm", "TimesFM")):
        data = read(GROUPS / f"{model}_daily.json")["score"]["early_centered"]["arms"]
        rows[label] = [data[k]["equal"]["skill"] for k in ("rank1", "rank4", "rank16")]
    return rows


def main():
    rows = scores()
    example = read(MOMENT / "dictionary_transfer" / "weather" / "results" / "example.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.25), layout="constrained")
    colors = ["#355c88", "#943e29", "#137568", "#78508a"]
    symbols = ["o", "s", "D", "^"]
    for (model, y), color, marker in zip(rows.items(), colors, symbols):
        left.plot([0, 1, 2], y, color=color, marker=marker, label=model, linewidth=1.85, markersize=5.5)
    left.set_xticks([0, 1, 2], ["1", "4", "16"])
    left.set(xlim=(-.13, 2.13), ylim=(0, 1), ylabel="Centered forecast-change recovery",
             xlabel="Decoder directions in the second SAE")
    left.legend(loc="upper left", ncol=2, fontsize=8.3, frameon=False, handlelength=1.35,
                columnspacing=.9)
    left.set_title("A   Same forecast model, different SAE", loc="left", fontweight="bold", fontsize=11.5)
    left.yaxis.set_major_locator(MultipleLocator(.2))
    right.axhline(0, color="#90959a", lw=.8)
    for key, label, color, line in (("donor", "Original removal", "#22262a", "-"),
                                    ("corr_own", "Matched single", "#a25d39", "--"),
                                    ("group16", "16-direction group", "#137568", "-")):
        right.plot(np.arange(1, 17), example["waveforms_celsius"][key], label=label, color=color,
                   linestyle=line, lw=2.1 if key=="donor" else 1.8, marker="o" if key=="donor" else None,
                   markersize=2.8)
    right.set(xlim=(.7,16.3), ylim=(-.061,.064), xlabel="Forecast lead (days)",
              ylabel="Change in forecast temperature (°C)")
    right.set_xticks([1,4,8,12,16])
    right.set_title("B   One MOMENT-large weather forecast", loc="left", fontweight="bold", fontsize=11.5)
    right.legend(loc="lower right", fontsize=8.2, frameon=False, ncol=1)
    for ax in (left, right):
        ax.grid(axis="y", color="#a0a8af", alpha=.24, lw=.7)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(HERE / "forecast_effect_correspondence.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
