"""Plot comparable early and withheld donor effects from the two prospective cohorts."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from research.activation_transfer.shape_transfer.steering_budget.run import HERE

def main():
    specs=[("short_bolt","M4 daily → Bolt"),("short_timesfm","M4 daily → TimesFM"),("noaa_bolt","Weather → Bolt"),("noaa_timesfm","Weather → TimesFM")]
    colors={"sae":"#7e5487","pca":"#5f7783","response32":"#087e85"}
    fig,axes=plt.subplots(1,3,figsize=(12,4.15),sharey=True,layout="constrained")
    for ax,key,title in zip(axes,("early_dense_shape","early_source_shape","late_source_shape"),
                            ("Leads 1–8\nReceiving dense edit","Leads 1–8\nActual donor feature","Leads 9–16\nActual donor feature")):
        for i,(stem,name) in enumerate(specs):
            data=json.loads((HERE/"results"/(stem+".json")).read_text())[key]
            for off,(method,legend) in zip((-.18,0,.18),(("sae","Active SAE"),("pca","Activation PCA"),("response32","Forecast gradients"))):
                d=data[method];point=d["value"];lo,hi=d["crossed95"];y=3-i-off
                ax.plot([lo,hi],[y,y],color=colors[method],lw=1.35)
                ax.scatter(point,y,s=31,color=colors[method],label=legend if i==0 else None,zorder=3)
        ax.set_title(title,fontsize=12);ax.axvline(0,color="#50545a",lw=.8);ax.axvline(1,color="#b2b6b8",ls=":",lw=.9)
        ax.set_yticks(range(4),[s[1] for s in specs[::-1]])
        ax.grid(axis="x",alpha=.14);ax.spines[["right","top","left"]].set_visible(False)
        ax.set_xlabel("Centered recovery")
    axes[0].set_xlim(.10,1.05);axes[1].set_xlim(.04,1.06)
    # Use a wider scale on withheld leads to include all crossed intervals.
    axes[0].legend(fontsize=8.5,loc="lower left",bbox_to_anchor=(0,1.29),ncol=3,frameon=False)
    fig.savefig(HERE/"results/steering_comparison.png",dpi=175)
    plt.close(fig)
if __name__=="__main__":main()
