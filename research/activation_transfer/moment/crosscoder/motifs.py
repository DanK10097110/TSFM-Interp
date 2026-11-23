"""Illustrative source-only nominations for three familiar within-patch input shapes."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.mechanism import OUT

DEST=OUT/"overview"


def centered_unit(x):
    z=x-x.mean(axis=-1,keepdims=True)
    return z/np.linalg.norm(z,axis=-1,keepdims=True)


def run():
    templates={"Rising ramp":np.arange(8,dtype=float),"Alternating values":np.array([-1.,1.]*4),
               "Final-sample spike":np.eye(8)[-1]}
    report={}
    for seed in cd.SEEDS:
        name=f"masked_{seed}"
        with np.load(cd.CACHE/(name+".npz")) as f:
            frequency=(f["m4_source"][:768]>0).sum(0)
        with np.load(OUT/"patch_probe"/("all_"+name+".npz")) as f:
            small,large=map(centered_unit,(f["source"],f["target"]))
        eligible=np.flatnonzero(frequency>=24)
        rec={}
        for label,goal in templates.items():
            goal=centered_unit(goal)
            atom=eligible[np.argmax(small[eligible]@goal)]
            target_scores=large[eligible]@goal
            rec[label]=dict(id=int(atom),source_firings=int(frequency[atom]),eligible=len(eligible),
                source=float(small[atom]@goal),large=float(large[atom]@goal),
                matched=float(small[atom]@large[atom]),large_rank=int(1+np.sum(target_scores>large[atom]@goal+1e-10)),
                target_template=goal.tolist(),small_pattern=small[atom].tolist(),large_pattern=large[atom].tolist())
        report[str(seed)]=rec
    (DEST/"motifs.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),sharey=True)
    for ax,(label,data) in zip(axes,report[str(cd.SEEDS[0])].items()):
        ix=np.arange(1,9)
        for k,c,style,leg in (("target_template","#9a9a9a","--","Input motif"),("small_pattern","#166e71","o-","Small"),("large_pattern","#b85426","s-","Large")):
            ax.plot(ix,data[k],style,color=c,label=leg,linewidth=1.6,markersize=3)
        ax.set_title(f"{label} · feature {data['id']}",loc="left",fontsize=10,fontweight="bold")
        ax.axhline(0,linewidth=.55,color="#9a9a9a");ax.set_xticks([1,4,8]);ax.set_xlabel("Observed patch position",fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Centered unit shape")
    fig.suptitle("Features nominated using only the small model's decoded input pattern",x=.072,ha="left",fontsize=13,fontweight="bold")
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False)
    fig.subplots_adjust(left=.075,right=.985,top=.75,bottom=.29,wspace=.13)
    fig.savefig(DEST/"motifs.png",dpi=175,facecolor="white");plt.close(fig)


if __name__=="__main__":run()
