"""Plot source survival and two conditional recipient-identity trajectories."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from research.activation_transfer.shape_transfer.rolling.data import HERE, LAGS, OUT


def main():
    data=json.loads((OUT/"summary.json").read_text())
    names={"bolt":"Bolt → TimesFM","timesfm":"TimesFM → Bolt"}
    domains={"daily":"M4 daily","monash_traffic":"Hourly traffic"}
    fig, axes=plt.subplots(2,2,figsize=(8.6,5.6),sharey=True,sharex=True)
    colors=("#247459","#D68911","#613F80")
    stats=(("source_survival","Source still fires",False),
           ("prior_target_available","Old target still fires",False),
           ("switch","Old target still wins",True))
    for a,(src,title) in enumerate(names.items()):
        for b,(domain,dname) in enumerate(domains.items()):
            ax=axes[a,b]
            for color,(key,label,flip) in zip(colors,stats):
                points=[]; lowers=[]; uppers=[]
                for h in LAGS:
                    r=data[src][domain][str(h)][key]
                    v=r["value"]; lo,hi=r["ci95"]
                    if flip:v,lo,hi=1-v,1-hi,1-lo
                    points.append(v);lowers.append(lo);uppers.append(hi)
                x=np.arange(len(LAGS))
                ax.plot(x,points,label=label,color=color,marker="o",ms=3.4,linewidth=1.8)
                ax.fill_between(x,lowers,uppers,color=color,alpha=.12,linewidth=0)
            ax.set_title(title+"  ·  "+dname,fontsize=10.5)
            ax.set_xticks(range(len(LAGS)),[str(x) for x in LAGS])
            ax.set_ylim(-.03,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.grid(axis="y",alpha=.24);ax.spines[["top","right"]].set_visible(False)
            if a==1:ax.set_xlabel("Observed steps since initial origin")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.005,1,.94))
    fig.savefig(HERE/"rolling_survival.png",dpi=165,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
