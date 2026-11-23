"""Show an outcome-blind representative weather forecast and same-eight-day error by requested length."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from research.activation_transfer.moment.horizon_prefix.run import OUT,WEATHER,load_forecasts,HORIZONS

def main():
    with np.load(WEATHER/"panel.npz") as f:ids,x,y=f["ids"],f["context"],f["future"]
    dates=json.loads((WEATHER/"manifest.json").read_text())["origins"]
    waves={h:load_forecasts("noaa",h)[1]["published"] for h in HORIZONS}
    delta=np.sqrt(np.mean((waves[16]-waves[8])**2,axis=2))
    iid,date=np.unravel_index(np.argsort(delta.ravel(),kind="stable")[delta.size//2],delta.shape)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.25),gridspec_kw={"width_ratios":[1.65,1]},layout="constrained")
    ax,ag=axes
    ax.plot(np.arange(-15,1),x[iid,date,-16:],lw=1.6,color="#595e66",label="Observed history")
    ax.plot(np.arange(0,9),np.r_[x[iid,date,-1],y[iid,date,:8]],lw=1.8,marker="o",ms=3,color="#272b2c",label="Observed future")
    ax.plot(np.arange(1,9),waves[8][iid,date],lw=2,color="#087e85",marker="s",ms=3,label="Asked for 8 days")
    ax.plot(np.arange(1,9),waves[16][iid,date],lw=2,color="#a94e35",ls="--",marker="o",ms=3,label="Asked for 16 days")
    ax.axvline(.5,color="#9aa1a4",ls=":",lw=1)
    ax.set(xlabel="Days from last observation",ylabel="Average temperature (°C)",title="Same history and same eight future dates")
    ax.text(.01,.98,ids[iid].replace("_"," ")+" · "+dates[date],transform=ax.transAxes,va="top",fontsize=8)
    ax.legend(fontsize=8,frameon=False,loc="lower left",bbox_to_anchor=(0,-.37),ncol=2)
    with (OUT/"summary.json").open() as f:cell=json.load(f)["published"]["noaa"]
    for i,h in enumerate(HORIZONS):
        v,lo,hi=cell[str(h)]["error"]
        ag.vlines(i,lo,hi,color="#51565d",lw=1.4);ag.scatter(i,v,s=46,color="#087e85" if i==0 else "#a94e35")
        ag.annotate(f"{v:.2f}",(i,v),xytext=(5,5),textcoords="offset points",fontsize=9)
    ag.set_xticks(range(4),[str(h) for h in HORIZONS])
    ag.set(xlabel="Days requested from MOMENT-large",ylabel="First-eight-day MAE (°C)",title="46 stations, four seasonal forecasts")
    for axis in axes:
        axis.spines[["top","right"]].set_visible(False);axis.grid(axis="y",alpha=.13)
    fig.savefig(OUT/"weather_horizon.png",dpi=175)
    plt.close(fig)
    meta=dict(station=str(ids[iid]),first_forecast_date=dates[date],selection="median RMS shift, independent of observed future",
              rms_difference_C=float(delta[iid,date]))
    (OUT/"weather_example.json").write_text(json.dumps(meta,indent=2)+"\n")

if __name__=="__main__":main()
