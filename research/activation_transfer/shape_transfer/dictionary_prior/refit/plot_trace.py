"""Visualize an actual winner exchange and the forecast jump as the dense step approaches zero."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT=Path(__file__).resolve().parent/"results"


def main():
    data=json.loads((OUT/"boundary_trace.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(9.8,6.1),layout="constrained")
    for col,(model,rec) in enumerate(data.items()):
        ax=axes[0,col]
        x=np.array(rec["dose"])*100.
        k=rec["first"]*100.
        ax.plot(x,rec["leaving_strength"],color="#21679a",linewidth=2,label=f"Leaving feature {rec['leaving']}")
        ax.plot(x,rec["entering_strength"],color="#a34317",linewidth=2,linestyle="--",label=f"Entering feature {rec['entering']}")
        ax.axvline(k,color="#70757a",lw=.8,zorder=-1)
        ax.set_title(f"{'Chronos-Bolt' if model=='bolt' else 'TimesFM'}  ·  {rec['series']}",fontweight="bold",loc="left",fontsize=12)
        ax.set_xlabel("Percent of the donor’s natural removal")
        ax.set_ylabel("Receiving SAE activation")
        ax.legend(fontsize=8,loc="center left",framealpha=.90)
        ax=axes[1,col]
        for arm,color,line,label in (("hard","#a34317","-","Hard TopK"),("locked","#389071","--","Locked support"),("dense","#21679a",":","Dense write")):
            points=rec["arms"][arm]["crossings"]
            scale=np.array(list(map(float,points)))
            diff=[v["forecast_rms"] for v in points.values()]
            ax.loglog(scale,diff,marker="o",markersize=4,color=color,linestyle=line,label=label)
        ax.set_xlim(.02,5e-6)
        ax.set_ylim(1e-7,.085)
        ax.set_xlabel("Distance to boundary on each side (dose fraction)")
        ax.set_ylabel("Eight-lead forecast jump, RMS")
        ax.grid(axis="y",alpha=.17)
        if col==0:ax.legend(fontsize=8,frameon=False,loc="center left")
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"boundary_trace.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
