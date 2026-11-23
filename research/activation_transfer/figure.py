"""Build the summary figure from the released study results without loading a forecaster."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext

HERE=Path(__file__).resolve().parent
BLUE="#16648c"
ORANGE="#b95727"
GREEN="#397452"
GREY="#525e67"


def load(name):
    return json.loads((HERE/name).read_text())


def point(ax, position, metric, color, marker, side=0):
    value=metric["value"]
    lo,hi=metric["ci95"]
    ax.plot([position+side]*2, [lo, hi], color=color, lw=1.35)
    ax.scatter(position+side,value,c=color,s=46,marker=marker,zorder=3,edgecolors="white",linewidths=.6)


def style(ax, title):
    ax.set_title(title, loc="left",fontsize=13,fontweight="bold",pad=12)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="both",length=0,pad=5)
    ax.grid(axis="y",color="#d9dee2",linewidth=.8)
    ax.set_axisbelow(True)


def legend(ax, entries, **kwargs):
    lines=[Line2D([0],[0],marker=m,color=c,lw=0,markersize=7,label=n) for n,c,m in entries]
    ax.legend(handles=lines,frameon=False,fontsize=9,handletextpad=.25,columnspacing=.8,**kwargs)


def plot():
    fig,axes=plt.subplots(2,2,figsize=(12.2,8.8),layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.16,h_pad=.14,hspace=.14,wspace=.09)
    a,b,c,d=axes.ravel()
    cross=load("boundary/crossing_bolt.json")["stages"]
    stages=[cross[str(i)] for i in [4,8,12]]
    for key,label,color,mark in [("hard","Hard TopK",ORANGE,"s"),("dense","Original state",BLUE,"o"),
                                  ("shoulder","Tapered TopK",GREEN,"^")]:
        values=[stage["norm_over_full_native"][key]["median"] for stage in stages]
        a.plot(range(3),values,marker=mark,c=color,label=label,lw=1.8,ms=6)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(LogFormatterMathtext())
    a.set_xticks(range(3),["1.88 × 10⁻³","1.17 × 10⁻⁴","7.32 × 10⁻⁶"])
    a.set_xlabel("Input distance across a feature switch (history SD)")
    a.set_ylabel("Median change / original edit norm")
    a.legend(loc="lower left",frameon=False,fontsize=9)
    style(a,"A   At a TopK boundary in Bolt")

    model_colors=[("timesfm","TimesFM → Bolt",BLUE,"o"),("bolt","Bolt → TimesFM",ORANGE,"s")]
    for j,(source,label,color,marker) in enumerate(model_colors):
        r=load(f"atoms/results/{source}_scores.json")["daily"]["source_native"]
        for i,(kind,arm) in enumerate([("full","target_dense"),("full","wrong_mean"),
                                       ("full","target_coactive"),("shape","target_dense")]):
            point(b,i,r[kind]["summary"][arm]["cosine"],color,marker,(-1 if j==0 else 1)*.07)
    b.set_xticks(range(4),["Mapped\natom","Wrong\natom","Most\ncoactive","Mapped\nshape only"])
    b.set_ylabel("Agreement with source ablation (cosine)")
    b.set_ylim(-.38,1.05)
    b.axhline(0,c=GREY,lw=1)
    legend(b,[(l,col,m) for _,l,col,m in model_colors],loc="upper right",ncol=1)
    style(b,"B   Removing a naturally firing SAE feature")

    for i,(target,domain) in enumerate([("bolt","m4_daily"),("bolt","bitbrains_fast_storage"),
                                       ("timesfm","m4_daily"),("timesfm","bitbrains_fast_storage")]):
        r=load(f"input_baseline/results/rolling_{target}.json")["splits"][domain]
        for pos,name,col,m in [(-.07,"bridge",BLUE,"o"),(.07,"best_raw",ORANGE,"s")]:
            point(c,i,r["methods"][name]["full"]["recovery"],col,m,pos)
    c.set_xticks(range(4),["Daily\n→ Bolt","VM\n→ Bolt","Daily\n→ TimesFM","VM\n→ TimesFM"])
    c.set_ylim(-.2,1.12)
    c.set_ylabel("Recovery of recipient's own forecast edit")
    c.axhline(0,c=GREY,lw=1)
    legend(c,[("Source states",BLUE,"o"),("Raw history",ORANGE,"s")],loc="lower left",ncol=2)
    style(c,"C   Advancing real input windows")

    for j,(source,label,color,marker) in enumerate(model_colors):
        target="bolt" if source=="timesfm" else "timesfm"
        methods=load(f"reencode/results/{source}_to_{target}.json")["datasets"]["weekly"]["gates"]["hard"]["methods"]
        for i,arm in enumerate(["direct","estimated","matched","raw_matched"]):
            point(d,i,methods[arm]["cosine"],color,marker,(-1 if j==0 else 1)*.07)
    d.set_xticks(range(4),["Predict code\ndirectly","Estimate\nstart","Use actual\nstart","Input cue +\nactual start"])
    d.set_ylim(0,.95)
    d.set_ylabel("Agreement with recipient SAE code change (cosine)")
    legend(d,[(l,col,m) for _,l,col,m in model_colors],loc="upper left",ncol=2)
    style(d,"D   Predicting the receiving SAE’s response")
    fig.savefig(HERE/"transfer_results.png",dpi=185,bbox_inches="tight")
    plt.close(fig)


def plot_matched_pair():
    """Show the named forward and reciprocal removals across the same 192 daily histories."""
    folder=HERE/"replication/results"
    report=load("replication/results/timesfm_coactive.json")
    j=report["selected_index"]
    with np.load(folder/"timesfm_seed0.npz") as src,np.load(folder/"timesfm_coactive.npz") as tgt:
        np.testing.assert_array_equal(src["ids"],tgt["ids"])
        rows=src["split"]=="daily"
        a=src["native"][rows,j].astype(float)
        b=tgt["selected_target_native"][rows].astype(float)
        source_active=src["active"][rows,j]
        target_active=tgt["native_target_codes"][rows,j]>0
    source_name=f"TimesFM {report['source_atoms'][j]}"
    target_name=f"Chronos-Bolt {report['target_atoms'][j]}"
    joint=source_active & target_active
    groups=[(joint,BLUE,"Both active"),(source_active & ~target_active,ORANGE,"TimesFM only"),
            (target_active & ~source_active,GREEN,"Bolt only")]
    fig,(left,right)=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    for mask,color,label in groups:
        left.scatter(a[mask].mean(1),b[mask].mean(1),s=20,c=color,alpha=.76,
                     label=f"{label} ({int(mask.sum())})",edgecolors="none")
    limits=[min(left.get_xlim()[0],left.get_ylim()[0]),max(left.get_xlim()[1],left.get_ylim()[1])]
    left.plot(limits,limits,ls="--",lw=1,color=GREY)
    left.set_xlim(limits)
    left.set_ylim(limits)
    left.set_aspect("equal",adjustable="box")
    left.set_xlabel(source_name+" mean forecast change")
    left.set_ylabel(target_name+" mean forecast change")
    left.legend(loc="upper left",frameon=False,fontsize=8)
    style(left,"A   Overall forecast shifts")
    draws=np.random.default_rng(2026112011).integers(joint.sum(),size=(4000,joint.sum()))
    leads=np.arange(1,a.shape[1]+1)
    for waveform,name,color in [(a,source_name,BLUE),(b,target_name,ORANGE)]:
        centered=waveform[joint]-waveform[joint].mean(1,keepdims=True)
        low,high=np.quantile(centered[draws].mean(1),[.025,.975],axis=0)
        right.fill_between(leads,low,high,color=color,alpha=.16,linewidth=0)
        right.plot(leads,centered.mean(0),marker="o",lw=2,ms=4,label=name,color=color)
    right.axhline(0,lw=.8,c=GREY)
    right.set_xticks(leads)
    right.set_xlabel("Forecast lead")
    right.set_ylabel("Change after removing the mean (history SD)")
    right.legend(frameon=False,loc="upper right",fontsize=9)
    style(right,"B   Horizon shape when both fire")
    fig.savefig(folder/"selected_feature_pair.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    plot()
    plot_matched_pair()
