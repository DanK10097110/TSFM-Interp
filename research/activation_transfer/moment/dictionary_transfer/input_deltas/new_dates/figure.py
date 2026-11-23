"""Illustrate the confirmed ramp using the median target response, without using counterpart results for selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import panel
from research.activation_transfer.moment.input_paths.run import modify

OUT = Path(__file__).resolve().parent / "results"


def main():
    ids, dates, x = panel()
    with np.load(OUT/"reference.npz") as f:
        np.testing.assert_array_equal(ids, f["ids"])
        np.testing.assert_array_equal(dates, f["dates"])
        ref=f["wave"]
    centered = ref[:,:8].astype(float)-ref[:,:8].mean(1,keepdims=True)
    energy = np.sqrt((centered**2).mean(1))
    selected = int(np.argmin(abs(energy-np.median(energy))))
    station,date = str(ids[selected//len(dates)]),dates[selected%len(dates)]
    with np.load(OUT/"mapped.npz") as f: mapped=f["wave"][selected]
    template_path=OUT.parents[1]/"template"/"results"/"waves.npz"
    with np.load(template_path) as f:
        np.testing.assert_array_equal(f["reference"],ref)
        template = f["historical_matched"][selected]
    y=x[selected]
    changed=modify(x[selected:selected+1],"ramp")[0]
    assert abs((changed-y).sum())<1e-4
    fig,(left,right)=plt.subplots(1,2,figsize=(10.4,3.9),layout="constrained")
    left.plot(np.arange(-15,1), y[-16:], color="#2e465f",linewidth=2,marker="o",ms=3.7,label="Observed")
    left.plot(np.arange(-7,1), changed[-8:],color="#a44d2d",linewidth=2,marker="o",ms=3.7,label="After ramp")
    left.set(xticks=[-15,-12,-8,-4,0],xlabel="Days relative to last observation",ylabel="Historical temperature (°C)")
    left.legend(frameon=False,fontsize=9)
    left.set_title("A   The input edit",loc="left",fontsize=11.5,fontweight="bold")
    q=np.arange(1,17)
    right.axhline(0,color="#8b8f94",lw=.8)
    right.axvline(8.5,color="#bcc0c4",ls=":",lw=.9)
    for z,label,color,style in ((ref[selected],"Receiver's own change","#263039","-"),
                                (mapped,"Mapped donor change","#06786a","-"),
                                (template,"Historical ramp, same dose","#a95833","--")):
        right.plot(q,z,color=color,ls=style,linewidth=1.9,label=label)
    right.set(xticks=[1,4,8,12,16],xlabel="Forecast lead (days)",ylabel="Forecast change from one patch (°C)")
    right.legend(frameon=False,fontsize=8.4)
    right.set_title("B   The large receiver's forecast response",loc="left",fontsize=11.5,fontweight="bold")
    for ax in (left,right):
        ax.grid(axis="y",lw=.65,alpha=.2);ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.savefig(OUT/"ramp_in_degrees.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    arms=dict(reference=ref[selected],mapped=mapped,historical_matched=template)
    rec=dict(station=station,forecast_origin=date,selection="nearest to panel median receiving days 1–8 centered RMS, reference only",
             row=selected,median_response_RMS_C=float(np.median(energy)),case_response_RMS_C=float(energy[selected]),
             input_C=y[-16:].tolist(),ramped_C=changed[-16:].tolist(),effects_C={k:v.tolist() for k,v in arms.items()})
    (OUT/"example.json").write_text(json.dumps(rec,indent=2)+"\n")


if __name__=="__main__":
    main()
