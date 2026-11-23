"""Two directions of the same intervention, scored after removing forecast level."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def make():
    full = json.loads((HERE/"full_space/results/results.json").read_text(encoding="utf-8"))
    sparse = json.loads((HERE/"sparse_writes/results/results.json").read_text(encoding="utf-8"))
    follow = json.loads((HERE/"full_space/results/followups.json").read_text(encoding="utf-8"))
    fig,axes = plt.subplots(1,2,figsize=(12.4,5.4),gridspec_kw={"width_ratios":[1.45,1]})
    for src,label,color,offset in [("timesfm","TimesFM → Bolt","#007D77",.13),("bolt","Bolt → TimesFM","#A84D24",-.13)]:
        f = full[src]["domains"]["daily"]["subsets"]["8"]["shape"]["methods"]
        s = sparse[src]["daily"]["8"]["shape"]
        l = follow[src]["daily"]
        values = [[f["clean"]["skill"],s["local_removal"]["skill"],s["one_signed"]["skill"],
                   s["four_signed"]["skill"],f["fixed"]["skill"],f["adaptive"]["skill"]],
                  [f["adaptive"]["skill"],l["held_even"]["shape"]["odd"],
                   l["held_even"]["shape"]["odd_interpolation"],l["late"]["shape"]["eight"]]]
        for ax,series in zip(axes,values):
            for i,obs in enumerate(series):
                lo,hi=obs["ci95"]
                ax.hlines(i+offset,lo,hi,colors=color,lw=1.8)
                ax.plot(obs["value"],i+offset,"o",color=color,ms=6,label=label if i==0 and ax is axes[0] else None)
    names=[["Clean-state map","One literal target removal","One signed target coefficient",
            "Up to four signed coefficients","Fixed derivative map","Per-history derivative map"],
           ["Fit 1–8, score 1–8","Fit odd, score even","Interpolate odd, score even¹","Fit 1–8, score 9–16"]]
    for ax,labels,title in zip(axes,names,["A   How the target is edited","B   Which leads are visible"]):
        ax.set_yticks(np.arange(len(labels)),labels)
        ax.set_ylim(len(labels)-.55,-.55)
        ax.set_title(title,loc="left",fontweight="bold",fontsize=12,pad=14)
        ax.axvline(0,color="#555",lw=1)
        ax.grid(axis="x",alpha=.22)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.set_xlabel("Centered forecast recovery")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False,bbox_to_anchor=(.57,.98))
    fig.suptitle("Transferring the effect of an SAE removal",fontsize=16,fontweight="bold",y=1.02)
    fig.text(.01,.01,"192 M4 daily histories. Points average over frequent donor features. Lines are 95% series-bootstrap intervals.\n"
             "¹ Interpolation predicts a waveform without running a recipient intervention.",fontsize=9,color="#444")
    fig.tight_layout(rect=(0,.10,1,.90),w_pad=2.5)
    fig.savefig(HERE/"shape_transfer.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    make()
