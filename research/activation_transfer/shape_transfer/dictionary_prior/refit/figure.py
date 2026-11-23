"""Show the effect of changing receiver SAE fit and the independent gate replication."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
BLUE="#21679a"
ORANGE="#a34317"


def main():
    match=json.loads((OUT/"summary.json").read_text())
    gate=json.loads((OUT/"confirmation.json").read_text())
    fig,(left,right)=plt.subplots(1,2,figsize=(11.3,4.1),layout="constrained")
    methods=["bolt_donor1","bolt_donor2","timesfm_donor1","timesfm_donor2"]
    names=["Bolt, seed 1","Bolt, seed 2","TimesFM, seed 1","TimesFM, seed 2"]
    for ax in (left,right):
        ax.spines[["left","top","right"]].set_visible(False)
        ax.tick_params(axis="y",length=0,pad=5)
        ax.axvline(0,color="#858b90",linewidth=1)
        ax.set(yticks=np.arange(4)[::-1],yticklabels=names,ylim=(-.74,3.55))
    left.set_title("Can one active feature recover the later shape?",fontsize=11,loc="left",fontweight="bold",pad=11)
    right.set_title("Does fixing active membership help on replication?",fontsize=11,loc="left",fontweight="bold",pad=11)
    for i,key in enumerate(methods):
        own=int(key[-1])
        row=match[key]["extra_daily"]["seed"]
        for j,(seed,color,label) in enumerate(((own,BLUE,"Same SAE fit"),(3-own,ORANGE,"Other SAE fit"))):
            v=row[str(seed)]["late_shape"]["natural"]
            point=v["value"];lo,hi=v["crossed95"]
            y=3-i+(.12 if j==0 else -.12)
            left.errorbar(point,y,xerr=np.array([[point-lo],[hi-point]]),fmt="o",ms=5,color=color,capsize=2,
                          label=label if i==0 else None)
        for j,(freq,color) in enumerate((("daily",BLUE),("weekly",ORANGE))):
            v=gate[key][freq]["late_shape"]["locked_minus_hard"]
            point=v["value"];lo,hi=v["crossed95"]
            right.errorbar(point,3-i+(.12 if j==0 else -.12),xerr=np.array([[point-lo],[hi-point]]),fmt="o",
                           ms=5,color=color,capsize=2,label=freq.capitalize() if i==0 else None)
    left.set_xlabel("Centered recovery on leads 9–16")
    right.set_xlabel("Locked minus hard recovery on leads 9–16")
    for ax in (left,right):
        ax.legend(loc="lower left",ncol=2,frameon=False,fontsize=8,handlelength=.8)
    fig.savefig(OUT/"changing_dictionary.png",dpi=160,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
