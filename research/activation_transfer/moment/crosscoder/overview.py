"""Place matched-minus-distractor patch and forecast comparisons on identical axes for both pairs."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.mechanism import OUT
from research.activation_transfer.moment.crosscoder.patch_probe import compare,crossed
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import OUT as FAM

DEST=OUT/"overview"


def load_pair(pair,name):
    if pair=="MOMENT":
        with np.load(OUT/"directions"/(name+".npz")) as f, np.load(OUT/"patch_probe"/("all_"+name+".npz")) as p:
            ids,dom,atom,fr=f["ids"],f["domains"],f["atom"],f["paired"]-f["within_history"]
            np.testing.assert_array_equal(ids,p["ids"][:len(ids)])
            np.testing.assert_array_equal(atom,p["atoms"][:len(ids)])
            comp=compare(p["source"],p["target"],atom)
            past=comp["shape"]-comp["other_shape"]
    else:
        with np.load(FAM/(name+"_directions.npz")) as f:
            dom,atom,past,fr=f["domains"],f["atom"],f["patch32_adv"],f["forecast16_adv"]
    return dom,atom,past,fr


def collect():
    results={}
    for pair in ("MOMENT","TimesFM_Bolt"):
        results[pair]={}
        for method in cd.METHODS:
            for seed in cd.SEEDS:
                name=f"{method}_{seed}"
                dom,atom,past,fr=load_pair(pair,name)
                for d in np.unique(dom):
                    ix=dom==d
                    vals={"past":past[ix],"forecast":fr[ix],"difference":(past-fr)[ix]}
                    results[pair].setdefault(str(d),{})[name]=crossed(vals,atom[ix],None,False)
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/"comparisons.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    return results


def figure(p):
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.8),sharey=True)
    palette={"past":"#166e71","forecast":"#bb5729"}
    rows=[("daily","joint"),("daily","masked"),("daily","shuffled"),
          ("monash_traffic","joint"),("monash_traffic","masked"),("monash_traffic","shuffled")]
    for ax,(pair,title) in zip(axes,(("MOMENT","MOMENT-small → MOMENT-large"),("TimesFM_Bolt","TimesFM → Chronos-Bolt"))):
        ax.axvline(0,color="#767676",lw=.9)
        ax.axhline(2.5,color="#d3d3d3",lw=.8)
        for i,(domain,method) in enumerate(rows):
            for metric,shift in (("past",-.145),("forecast",.145)):
                for j,seed in enumerate(cd.SEEDS):
                    v=p[pair][domain][f"{method}_{seed}"][metric]
                    center=v["mean"];lo,hi=v["history_atom_95"]
                    y=i+shift+(j-.5)*.095
                    ax.plot([lo,hi],[y,y],color=palette[metric],lw=1,alpha=.75)
                    ax.scatter(center,y,c=palette[metric],marker=("o","D")[j],s=22,zorder=3,linewidths=0)
        ax.set_title(title,fontsize=13,loc="left",fontweight="bold")
        ax.set_xlim(-.28,1.04);ax.grid(axis="x",alpha=.13)
        ax.set_xlabel("Matched minus other active coordinate",fontsize=10)
        ax.spines[["right","top","left"]].set_visible(False)
        ax.tick_params(labelsize=9,axis="both",length=0)
    axes[0].set_yticks(range(6),["M4 daily · joint","M4 daily · masked","M4 daily · shuffled",
                                "Traffic · joint","Traffic · masked","Traffic · shuffled"])
    axes[0].set_ylim(5.55,-.55)
    fig.suptitle("Shared input shapes are easier to identify than shared forecast changes",fontsize=16,fontweight="bold",x=.09,ha="left",y=.985)
    fig.text(.09,.914,"Mean signed shape cosine above a same-history distractor, on 128 held-out series per domain.",fontsize=10,color="#404040")
    handles=[Line2D([0],[0],marker="o",color=c,lw=2,label=l) for c,l in [(palette["past"],"Decoded observed patch"),(palette["forecast"],"Native forecast derivative")]]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.53,.065))
    fig.text(.09,.042,"Circle and diamond show independent fits. Intervals resample histories and feature IDs. Shuffled fits use a masked encoder.",fontsize=9,color="#444444")
    fig.subplots_adjust(left=.16,right=.985,top=.83,bottom=.24,wspace=.12)
    fig.savefig(DEST/"shared_patch_forecast.png",dpi=175,facecolor="white")
    plt.close(fig)


def example():
    name="masked_22112401"
    with np.load(OUT/"directions"/("finite_"+name+".npz")) as f:
        row=int(np.flatnonzero(f["domains"]=="daily")[0]);sid=str(f["ids"][row]);atom=int(f["atom"][row]);dose=float(f["dose"][row])
        sd=float(f["history_sd"][row])
        native=[f[k][row].astype(float)/sd for k in ("small","large")]
    with np.load(OUT/"patch_probe"/("all_"+name+".npz")) as f:past=[-dose*f[k][atom] for k in ("source","target")]
    fig,ax=plt.subplots(1,2,figsize=(9.8,3.6))
    meta={}
    for a,(key,waves,title) in zip(ax,(("past",past,"Readout of the last 8 observed values"),("future",native,"Native change to the next 16 forecasts"))):
        shape=np.stack([z-z.mean() for z in waves])
        color=("#126b6d","#af4927")
        for i,s in enumerate(shape):
            a.plot(np.arange(1,len(s)+1),s,"o-",color=color[i],lw=1.8,markersize=3,label=("Small","Large")[i])
        a.axhline(0,lw=.8,color="#818181")
        a.set_xticks([1,4,8] if shape.shape[1]==8 else [1,4,8,12,16])
        a.set_title(title,fontsize=10.5,loc="left",fontweight="bold")
        a.set_xlabel("Position in readout");a.spines[["right","top"]].set_visible(False)
        meta[key]=dict(small=shape[0].tolist(),large=shape[1].tolist(),cosine=float(shape[0]@shape[1]/np.prod(np.linalg.norm(shape,axis=1))))
    ax[0].legend(loc="best",fontsize=8,frameon=False)
    ax[0].set_ylabel("Centered change (history SD)")
    fig.suptitle(f"One source-active coordinate · M4 {sid} · feature {atom}",x=.08,ha="left",fontsize=13,fontweight="bold")
    fig.subplots_adjust(left=.1,right=.98,wspace=.28,bottom=.21,top=.77)
    fig.savefig(DEST/"single_case.png",dpi=175,facecolor="white");plt.close(fig)
    (DEST/"single_case.json").write_text(json.dumps(dict(selection="first sampled daily case of first masked seed",case=sid,atom=atom,dose=dose,**meta),indent=2)+"\n")


if __name__=="__main__":
    figure(collect());example()
