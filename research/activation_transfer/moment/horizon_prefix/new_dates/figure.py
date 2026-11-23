"""Show why forecast disagreement can replicate while the accuracy ordering changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent/"results"


def main():
    d=json.loads((OUT/"summary.json").read_text())
    fig,(one,two)=plt.subplots(2,1,figsize=(8,5.6),sharex=True,layout="constrained")
    months="Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov".split()
    for key,label,color,style,marker in (
        ("primary","35 original stations","#006973","-","o"),("secondary","35 additional stations","#9a4731","--","s")):
        for ax,metric in ((one,"RMS"),(two,"MAE")):
            y=d[key]["large_16_minus_direct_"+metric+"_C"]["per_date"]
            ax.plot(months,y,label=label,color=color,ls=style,marker=marker,ms=4,lw=1.6)
    one.set(title="Changing the request changes days 1–8 at every origin",ylabel="Forecast difference\n(RMS °C)",ylim=(0,None))
    two.set(title="The more accurate request varies through the year",ylabel="Error of 16-day prefix\nminus 8-day forecast (°C)",xlabel="2026 forecast origin, first day of the month")
    two.axhline(0,color=".4",lw=.9)
    one.legend(frameon=False,loc="upper right",fontsize=9)
    for ax in (one,two):
        ax.grid(axis="y",alpha=.16);ax.spines[["right","top"]].set_visible(False)
    fig.savefig(OUT/"new_weather_dates.png",dpi=165);plt.close(fig)


if __name__=="__main__":main()
