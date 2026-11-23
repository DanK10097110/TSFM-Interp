"""Plot observed lead placement and physical recovery on identical held-out measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def main():
    scores=json.loads((OUT/"summary.json").read_text())
    residual=json.loads((OUT/"residual.json").read_text())
    blue,orange,gray="#21679a","#a34317","#59636a"
    plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans","axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":150})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.15),gridspec_kw={"width_ratios":[1,1.3]})
    ax=axes[0]
    for row,leads in ((1,np.arange(1,9)),(0,[1,3,5,7,10,12,14,16])):
        ax.plot([1,16],[row,row],color="#c3c8cc",linewidth=1,zorder=0)
        ax.scatter(leads,[row]*8,s=63,c=blue,zorder=2)
        ax.scatter([9,11,13,15],[row]*4,s=65,edgecolor=orange,facecolor="white",marker="D",zorder=1)
    ax.set(yticks=[0,1],yticklabels=["Spread","First eight"],xlim=(.3,16.7),ylim=(-.65,1.65),
           xticks=[1,4,8,12,16],xlabel="Forecast lead")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.set_title("Where the eight measurements fall",loc="left",fontsize=11,fontweight="bold",pad=12)
    ax.legend(handles=[Line2D([],[],linestyle="",marker="o",color=blue,label="Supplied"),
                       Line2D([],[],linestyle="",marker="D",markerfacecolor="white",color=orange,label="Scored")],
              loc="lower center",ncol=2,frameon=False,fontsize=9)
    ax=axes[1]
    methods=[("First eight","first"),("Spread","spread"),("Straight interpolation","spread_interp"),
             ("Spread, beyond interpolation","curvature")]
    axispos=np.arange(len(methods))[::-1]
    ax.axvline(0,color="#6b7379",lw=.9)
    for label,color,off,pretty in (("bolt",blue,.11,"Bolt → TimesFM"),("timesfm",orange,-.11,"TimesFM → Bolt")):
        for (name,key),y in zip(methods,axispos):
            s=(residual[label]["daily"]["spread"]["physical"] if key=="curvature" else
               scores[label]["daily"]["scores"]["common_audit_shape"][key])
            lo,hi=s.get("ci95_features_and_histories",s.get("ci95_histories_features"))
            v=s["value"]
            ax.errorbar(v,y+off,xerr=np.array([[v-lo],[hi-v]]),fmt="o",capsize=2,ms=5,color=color,
                        label=pretty if y==axispos[0] else None)
    ax.set(yticks=axispos,yticklabels=[r[0] for r in methods],ylim=(-.75,3.55),xlabel="Centered recovery on leads 9, 11, 13 and 15")
    ax.tick_params(axis="y",length=0)
    ax.legend(loc="lower left",fontsize=8,ncol=2,frameon=False,handlelength=.8)
    ax.set_title("What transferred on held-out daily series",loc="left",fontsize=11,fontweight="bold",pad=12)
    fig.subplots_adjust(wspace=.74,left=.08,right=.98,bottom=.21,top=.88)
    fig.savefig(OUT/"lead_placement.png",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
