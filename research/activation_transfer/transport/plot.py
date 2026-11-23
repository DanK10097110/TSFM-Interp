"""Visualize natural-window target response recovery and dependence on fitted subspace rank."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "results"
GROUP = ["m4_daily", "m4_weekly", "bitbrains_fast_storage"]
LABELS = ["M4 daily", "M4 weekly", "Bitbrains VMs"]
COLORS = ["#185582", "#b85b25", "#008170"]


def main():
    fig, panels = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True,
                               gridspec_kw={"height_ratios": [1,1]})
    for column, (key, title) in enumerate([("timesfm_to_bolt","TimesFM → Chronos-Bolt"),
                                           ("bolt_to_timesfm","Chronos-Bolt → TimesFM")]):
        raw = json.loads((DATA / (key+"_natural.json")).read_text())
        rank = json.loads((DATA / (key+"_rank.json")).read_text())["groups"]
        rewired = json.loads((DATA / (key+"_rewired.json")).read_text())["groups"]
        top, bottom = panels[:,column]
        top.set_title(title, fontsize=14, weight="bold", pad=9)
        for j,(g,label,color) in enumerate(zip(GROUP,LABELS,COLORS)):
            full=raw[g]["full"]["methods"]
            for pos, datum in enumerate([full["raw"]["recovery"], rewired[g]["full"]["rewired_average"],
                                         full["bridge"]["recovery"]]):
                y = j + (pos-1)*.21
                lo,hi = datum["ci95"]
                top.plot([lo,hi],[y,y],lw=1.4,color=color,alpha=.8)
                top.scatter(datum["value"],y,marker=["s","^","o"][pos],edgecolor=color,
                            facecolor=color if pos==2 else "white",s=40,zorder=3)
            r=rank[g]; levels=["1","4","16","64"]
            knots=list(range(5))
            scores=[r["full"]["rank"][v]["score"]["value"] for v in levels]+[r["full"]["full_map"]["value"]]
            shapes=[r["shape"]["rank"][v]["score"]["value"] for v in levels]+[r["shape"]["full_map"]["value"]]
            bottom.plot(knots,scores,"o-",color=color,ms=4,lw=1.8,label=label)
            bottom.plot(knots,shapes,"--",color=color,lw=1.65,alpha=.9)
        top.set_yticks(range(3),LABELS);top.set_ylim(2.55,-.55);top.set_xlim(-.94,1.12)
        top.set_xlabel("Recovery of target-native forecast effect")
        top.axvline(0,color="#777777",lw=.8)
        bottom.set_xticks(range(5),["1","4","16","64","full"]);bottom.set_ylim(-.15,1.05)
        bottom.set_xlabel("Dimensions retained from the fitted map");bottom.axhline(0,color="#888",lw=.8)
        if column==0:bottom.set_ylabel("Recovery")
        for axis in (top,bottom):
            axis.grid(axis="x" if axis is top else "y",alpha=.19)
            axis.spines[["top","right"]].set_visible(False)
    from matplotlib.lines import Line2D
    panels[0,0].legend(handles=[Line2D([0],[0],color="none",marker=m,markeredgecolor="#333333",
                                        markerfacecolor="#333333" if m=="o" else "white",label=l)
                                for m,l in [("s","Raw ridge"),("^","Rewired source"),("o","Trained source")]],
                       loc="lower left",fontsize=8, frameon=True, ncol=1)
    panels[1,0].legend(handles=[Line2D([0],[0],color="#333",ls=ls,label=label) for ls,label in
                                [("-","Full forecast"),("--","Horizon-centered shape")]],loc="lower right",fontsize=8)
    fig.align_ylabels()
    fig.savefig(DATA/"natural_transport.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
