"""Compare fixed signed group sizes, both SAE seeds and both lead blocks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent/"results"


def main():
    fig,axes=plt.subplots(1,2,figsize=(9,4.25),sharey=True,layout="constrained")
    for ax,size in zip(axes,("small","large")):
        for i,(seed,color) in enumerate(((22112201,"#106c76"),(22112202,"#a34c34"))):
            info=json.loads((OUT/f"{size}_{seed}_to_{size}_group_scores.json").read_text())["daily"]
            for j,(part,style,marker) in enumerate((("early","-","o"),("late","--","s"))):
                vals=info[f"native_{part}_shape"]
                hits=[vals[k]["value"] for k in ("one","four","sixteen")]
                x=np.arange(3)+(i-.5)*.05
                label=f"{'A to B' if i==0 else 'B to A'}, days {'1–8' if j==0 else '9–16'}"
                ax.plot(x,hits,marker=marker,ls=style,color=color,lw=1.7,ms=5,label=label)
        ax.set_xticks(range(3),["1","4","16"]);ax.set(xlabel="Receiving decoder directions",title="MOMENT-"+size)
        ax.set_ylim(0,1.02);ax.grid(axis="y",alpha=.18);ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Centered forecast-effect recovery")
    axes[1].legend(frameon=False,fontsize=8,loc="lower right")
    fig.savefig(OUT/"within_encoder_groups.png",dpi=170);plt.close(fig)


if __name__=="__main__":main()
