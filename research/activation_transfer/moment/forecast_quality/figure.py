"""Plot native direct H16, retaining the distinction between historical archives and 2026 weather."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER

ORDER = ["moment_small","moment_base","moment_large","bolt","timesfm"]
NAMES = ["MOMENT small","MOMENT base","MOMENT large","Chronos-Bolt","TimesFM"]
DOMAINS = ["m4_daily","m4_weekly","m4_hourly","monash_traffic","monash_electricity_hourly","bitbrains_fast_storage"]


def main():
    old=json.loads((OUT/"summary.json").read_text())["holdout"]["scores"]["h16"]
    weather=json.loads((WEATHER/"summary.json").read_text())["h16"]
    ratios=np.array([[old["comparisons"][d][m+"_vs_history_select"]["ratio"]["value"] for d in DOMAINS] for m in ORDER])
    fig,(ax,wax)=plt.subplots(1,2,figsize=(12.1,4.7),gridspec_kw={"width_ratios":[1.85,1]})
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.8,wspace=.48)
    scale=np.log2(ratios)
    ax.imshow(scale,cmap="RdBu_r",norm=TwoSlopeNorm(0,-1.5,3.5),aspect="auto")
    for i in range(len(ORDER)):
        for j in range(len(DOMAINS)):
            ax.text(j,i,f"{ratios[i,j]:.2f}",ha="center",va="center",color="white" if scale[i,j]>2.25 or scale[i,j]<-1 else "#111827",fontsize=10)
    ax.set_xticks(range(6),["M4\ndaily","M4\nweekly","M4\nhourly","Traffic","Energy","Cloud\nVM"],fontsize=9)
    ax.set_yticks(range(5),NAMES,fontsize=9)
    ax.set_title("Historical panels · MASE1 ratio\nrelative to history-selected classical forecast",loc="left",fontsize=11)
    entries=[*ORDER,"climatology","climate_anomaly","last"]
    names=[*NAMES,"Climatology","Climate + anomaly","Persistence"]
    colors=["#b74f30","#bc6a32","#315f77","#345d93","#386d60","#666666","#444444","#909090"]
    for y,(m,c) in enumerate(zip(entries,colors)):
        v=weather["metrics"][m]["mae"]
        lo,hi=v["ci95"]
        wax.errorbar(v["value"],y,xerr=[[v["value"]-lo],[hi-v["value"]]],fmt="o",ms=5,capsize=2.5,color=c)
    wax.set_yticks(range(len(entries)),names,fontsize=8)
    wax.invert_yaxis()
    wax.set_xlabel("Mean absolute error (°C)",fontsize=9)
    wax.set_title("2026 weather · 46 US stations\n95% station-block intervals",loc="left",fontsize=11)
    wax.grid(True,axis="x",alpha=.25)
    wax.spines[["right","top"]].set_visible(False)
    fig.suptitle("Sixteen-step point forecasts from 480 observed values",fontsize=15,fontweight="bold",x=.1,ha="left")
    fig.savefig(OUT/"forecast_quality.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
