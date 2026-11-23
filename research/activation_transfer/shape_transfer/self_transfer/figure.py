"""Show the 3×3 held-out comparison and an input-selected decomposition of a previously named feature."""
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from research.activation_transfer.shape_transfer.depth.intervene import source_material
from research.activation_transfer.shape_transfer.self_transfer.run import HERE, NAMES


def draw():
    out=HERE/"results"
    stats=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    names=("Bolt","TimesFM","Timer")
    fig=plt.figure(figsize=(10.7,6.5),layout="constrained")
    grid=fig.add_gridspec(2,2,height_ratios=[1.,.96])
    for c,(tag,title) in enumerate((("early_shape","A   Controlled leads 1–8"),("late_shape","B   Withheld leads 9–16"))):
        ax=fig.add_subplot(grid[0,c])
        values=np.array([[stats[s]["extra_daily"]["scores"][tag][t]["value"] for t in NAMES] for s in NAMES])
        im=ax.imshow(values,cmap="RdBu",norm=TwoSlopeNorm(vmin=-1.,vcenter=0.,vmax=1.),aspect="auto")
        ax.set_xticks(range(3),names);ax.set_yticks(range(3),names)
        ax.set_xlabel("Receiving model");ax.set_ylabel("Donor model" if c==0 else "")
        ax.set_title(title,loc="left",fontweight="bold")
        for i,j in np.ndindex(values.shape):
            ax.text(j,i,f"{values[i,j]:.2f}",ha="center",va="center",
                    color="white" if abs(values[i,j])>.48 else "black",fontweight="bold")
    bar=fig.colorbar(im,ax=fig.axes[:2],fraction=.035,pad=.02)
    bar.set_label("Centered recovery")
    ids,codes,atoms,*_=source_material("timesfm")
    col=int(np.flatnonzero(atoms==742)[0])
    with np.load(out/"timesfm_to_timesfm.npz") as z:
        ii=np.array([dict(zip(ids,range(len(ids))))[v] for v in z["ids"]])
        available=np.flatnonzero((z["split"]=="extra_daily")&z["active"][:,col])
        activation=codes[ii[available],col]
        chosen=available[np.lexsort((z["ids"][available],np.abs(activation-np.median(activation))))[0]]
        ax=fig.add_subplot(grid[1,:])
        ax.axvspan(8.5,16.5,color="#eeeeee")
        for key,label,colr,style in (("native","Actual SAE removal","#151515","-"),
                                    ("projection","Part visible to the first eight derivatives","#006f9a","--"),
                                    ("null","Part invisible to the first eight derivatives","#b74932","-.")):
            yy=z[key][chosen,col]
            ax.plot(np.arange(1,17),yy,color=colr,linestyle=style,linewidth=2.1,label=label)
        ax.axhline(0,color=".6",linewidth=.8)
        ax.set_xlim(.65,16.35);ax.set_xticks(range(1,17));ax.set_xlabel("Forecast lead")
        ax.set_ylabel("Change in median forecast\n(history SD)")
        ax.set_title(f"C   One TimesFM feature (742), M4 {z['ids'][chosen]}",loc="left",fontweight="bold")
        ax.legend(loc="best",frameon=False,fontsize=8)
        row=dict(history=str(z["ids"][chosen]),atom=742,coefficient=float(codes[ii[chosen],col]),
                 eligible=int(len(available)),median_active_coefficient=float(np.median(activation)))
    fig.savefig(out/"self_transfer.png",dpi=170)
    (out/"figure_case.json").write_text(json.dumps(row,indent=2)+"\n",encoding="utf-8")
    plt.close(fig)


if __name__=="__main__":
    draw()
