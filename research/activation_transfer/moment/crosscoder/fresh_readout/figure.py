"""Plot native directional specificity and actual response sizes across readout positions."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.activation_transfer.moment.crosscoder.fresh_readout.panel import OUT
from research.activation_transfer.moment.crosscoder.fresh_readout.score import SLOTS


def draw(ax,m,metric,offset,color,label,marker):
    values=np.array([m[s+"_"+metric]["estimate"] for s in SLOTS])
    interval=np.array([m[s+"_"+metric]["ci95"] for s in SLOTS])
    assert np.all((interval[:,0]<=values)&(values<=interval[:,1]))
    ax.errorbar(np.arange(4)+offset,values,yerr=np.vstack([values-interval[:,0],interval[:,1]-values]),fmt=marker,markersize=5,
                color=color,elinewidth=1.1,capsize=2.5,label=label)


def main():
    d=json.loads((OUT/"summary.json").read_text())["methods"]
    fig,(left,right)=plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    for name,dx,color,marker in (("joint",-.13,"#17517a","o"),("masked",0,"#c14823","s"),("shuffled",.13,"#686b76","^")):
        draw(left,d[name]["fixed_seed_mean"],"paired_minus_decoy",dx,color,name.capitalize(),marker)
    left.axhline(0,color="#777777",ls="--",lw=.8)
    left.set_ylabel("Matched minus distractor signed cosine")
    left.set_title("Direction agreement")
    for arm,dx,color,label,marker in (("small",-.09,"#17517a","MOMENT-small","o"),
                                     ("paired",.09,"#c14823","MOMENT-large","s")):
        draw(right,d["masked"]["fixed_seed_mean"],arm+"_centered_rms_C",dx,color,label,marker)
    right.set_ylim(bottom=0);right.set_ylabel("Centered response RMS (°C)")
    right.set_title("Response size for masked crosscoders")
    for ax in (left,right):
        ax.set_xticks(range(4),["Previous\nobserved","Edited\nobserved","Forecast\n1–8","Forecast\n9–16"])
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.16)
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(OUT/"native_readout.png",dpi=180);plt.close(fig)
    final_path=OUT/"final_summary.json"
    if final_path.exists():
        final=json.loads(final_path.read_text())["methods"]
        fig,ax=plt.subplots(figsize=(6.6,3.6),layout="constrained")
        for name,color,dx,label in (("masked","#17517a",-.06,"Paired histories"),("shuffled","#a44d26",.06,"Shuffled histories")):
            m=final[name]["fixed_seed_mean"]
            outcomes=[m["native_advantage"],m["probe_advantage"]]
            y=np.array([v["estimate"] for v in outcomes]);ci=np.array([v["ci95"] for v in outcomes])
            ax.errorbar(np.arange(2)+dx,y,yerr=np.vstack([y-ci[:,0],ci[:,1]-y]),marker="o",lw=1.5,capsize=4,
                        color=color,label=label)
        ax.axhline(0,color="#777777",ls="--",lw=.8);ax.set_xlim(-.23,1.23)
        ax.set_xticks([0,1],["Published reconstruction head","Frozen observed-value probe"])
        ax.set_ylabel("Matched minus distractor signed cosine");ax.set_title("Same finite changes to final hidden states")
        ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False);ax.grid(axis="y",alpha=.16)
        fig.savefig(OUT/"final_readout.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    main()
