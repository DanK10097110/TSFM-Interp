"""Compare the two evaluation questions on the same held-out domains."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA=Path(__file__).resolve().parent/"results"


def main():
    report=json.loads((DATA/"summary.json").read_text())
    fig,ax=plt.subplots(1,2,figsize=(9.1,3.8),gridspec_kw={"width_ratios":[1.43,1]},constrained_layout=True)
    rows=[("timesfm_to_bolt","daily"),("timesfm_to_bolt","weekly"),("bolt_to_timesfm","daily"),("bolt_to_timesfm","weekly")]
    labels=["TimesFM → Bolt, daily","TimesFM → Bolt, weekly","Bolt → TimesFM, daily","Bolt → TimesFM, weekly"]
    targets=[]
    for i,(pair,domain) in enumerate(rows):
        cell=report[pair][domain]
        h=cell["hidden"]["four_generic_equal_weight"]
        for arm,offset,color,label in (("clean_delta_tuned",-.11,"#6650a4","Clean, tuned on changes"),("delta",.11,"#12826d","Tangent")):
            metric=h[arm]["recovery"]
            v=metric["value"];lo,hi=metric["ci95"]
            ax[0].errorbar(v,i+offset,xerr=np.array([[v-lo],[hi-v]]),fmt="o",markersize=5,color=color,
                           capsize=2,label=label if i==0 else None)
        targets.append([cell["writes"][cue]["request"]["calibrated"]["delta"]["recovery"]["value"]
                        for cue in ("level","tilt","sine")])
    ax[0].set(yticks=np.arange(4),yticklabels=labels,ylim=(3.6,-.55),xlim=(0,1),
              xlabel="Hidden-change recovery",title="Predicting input-induced changes")
    ax[0].legend(loc="lower right",frameon=False,fontsize=8)
    ax[0].spines[["right","top"]].set_visible(False)
    arr=np.asarray(targets)
    ax[1].imshow(arr,aspect="auto",interpolation="nearest",cmap="Blues",vmin=0,vmax=1)
    for idx in np.ndindex(arr.shape):
        value=arr[idx]
        ax[1].text(idx[1],idx[0],f"{value:.3f}",ha="center",va="center",color="white" if value>.66 else "#14324b")
    ax[1].set(xticks=[0,1,2],xticklabels=["Lift","Tilt","Sine"],yticks=np.arange(4),yticklabels=[],
              xlabel="Requested forecast change",title="Transporting a native steering write")
    ax[1].tick_params(left=False)
    fig.savefig(DATA/"changes_vs_writes.png",dpi=175)
    plt.close(fig)


if __name__=="__main__":
    main()
