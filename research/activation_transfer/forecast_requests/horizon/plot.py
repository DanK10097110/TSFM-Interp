"""Plot target-specific lift persistence, averaging matched histories across source and depth."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {"bolt":"#7043a0","timesfm":"#16856a","timer":"#bc562c"}
LABELS = {"bolt":"Chronos-Bolt","timesfm":"TimesFM","timer":"Timer"}


def main():
    fig,axes=plt.subplots(1,2,figsize=(8,3.3),sharey=True,constrained_layout=True)
    t=np.arange(1,33)
    for ax,domain in zip(axes,("daily","weekly")):
        ax.axvspan(.5,8.5,color="#e6e9ec",alpha=.8,linewidth=0)
        ax.axhline(.05,color="#373b3e",linestyle="--",linewidth=1,label="Requested lift")
        ax.axhline(0,color="#acb2b7",linewidth=.7)
        for target,color in COLORS.items():
            with np.load(HERE/"results"/(target+".npz")) as z:
                keep=np.char.startswith(z["ids"],domain+"/")
                curves=[]
                for i in range(2):
                    names=list(z[f"{i}_arms"])
                    for source in COLORS:
                        if source!=target:
                            curves.append(z[f"{i}_effect"][keep,names.index(source)])
            items=np.mean(curves,axis=0)
            draws=np.random.default_rng(2026112401).integers(len(items),size=(4000,len(items)))
            lo,hi=np.quantile(items[draws].mean(axis=1),[.025,.975],axis=0)
            ax.fill_between(t,lo,hi,color=color,alpha=.14,linewidth=0)
            ax.plot(t,items.mean(0),color=color,linewidth=1.9,label=LABELS[target])
        ax.set(xlim=(.5,32.5),ylim=(-.003,.065),xticks=[1,8,16,24,32],xlabel="Forecast lead",
               title="M4 "+domain)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Forecast change (history SD)")
    handles,labels=axes[0].get_legend_handles_labels()
    axes[1].legend(handles,labels,loc="upper right",ncol=2,fontsize=7,framealpha=.88)
    fig.savefig(HERE/"results"/"lift_by_lead.png",dpi=175)
    plt.close(fig)


if __name__=="__main__":
    main()
