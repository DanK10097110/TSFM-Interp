"""Paired physical skill differences on the two daily cohorts, across two dictionary initializations."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.activation_transfer.shape_transfer.state_predictor.memory import HERE


def render():
    fig,axes=plt.subplots(1,2,figsize=(9.4,4.8),sharey=True)
    rows=[(m,s,c) for m in ("bolt","timesfm") for s in (0,1) for c in ("daily","extra_daily")]
    color={"bolt":"#207569","timesfm":"#965032"}
    for ax,contrast,title in zip(axes,("dense_minus_mean","support_minus_dense"),("Dense lookup minus constant mean","Binary SAE support minus dense lookup")):
        for y,(m,seed,domain) in enumerate(rows):
            p=HERE/"results"/f"{m}_seed{seed}_scores.json"
            r=json.loads(p.read_text())["domains"][domain]["physical_centered"][contrast]
            lo,hi=r["crossed95"];x=r["value"]
            ax.hlines(y,lo,hi,color=color[m],linewidth=1.5)
            ax.scatter(x,y,s=40,color=color[m],marker="o" if seed==0 else "D",zorder=3)
        ax.axvline(0,color="#383b42",lw=.9)
        ax.axhline(3.5,color="#dadde0",lw=.8)
        ax.set_title(title,fontsize=10,fontweight="bold")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x",alpha=.16)
        ax.set_xlabel("Paired change in centered forecast skill",fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(np.arange(len(rows)),[
        f"{'Bolt → TimesFM' if m=='bolt' else 'TimesFM → Bolt'},  {'archived' if s==0 else 'retrained'},  {'daily' if c=='daily' else 'extra daily'}"
        for m,s,c in rows])
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=.29,right=.98,bottom=.16,top=.88,wspace=.12)
    fig.savefig(HERE/"paired_effects.png",dpi=165,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":render()
