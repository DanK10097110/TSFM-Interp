"""Render the two held-out comparisons with item-paired bootstrap intervals."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def whisker(ax, y, s, color, marker, label=None):
    v = s['mean']
    lo, hi = s['ci95']
    ax.errorbar(v, y, xerr=np.array([[v-lo],[hi-v]]), color=color, marker=marker,
                markersize=6, capsize=2.5, lw=1.6, label=label)


def figure():
    audits={v['input']:v for v in read('forecast_audit.json')}
    fixed=read('fixed.json')['datasets']['reserve']
    fixed_beh={v['source']:v for v in read('fixed_forecast_audit.json')}
    colors=['#20567A','#A54320']
    fig,(left,right)=plt.subplots(1,2,figsize=(11.4,4.1),gridspec_kw={'width_ratios':[1.4,1]})
    fig.subplots_adjust(left=.19,right=.985,top=.83,bottom=.20,wspace=.68)
    labels=[]
    for i,(dataset,title) in enumerate([('m4','Daily'),('weekly','Weekly'),('fast','Bitbrains')]):
        hidden=read(dataset+'_smooth.json')['transfer']
        for j,(a,b,caption) in enumerate([('bolt','timesfm','Bolt → TimesFM'),('timesfm','bolt','TimesFM → Bolt')]):
            y=5-2*i-j
            labels.append(title+'   '+caption)
            state=hidden[a+'_to_'+b]['shoulder_minus_code']
            forecast=audits[f'{dataset}_smooth_{b}_forecasts.json']['contrasts']['cross_shoulder_minus_cross_code']['mean_item_cosine']
            whisker(left,y+.13,state,colors[0],'o','Hidden change' if y==5 else None)
            whisker(left,y-.13,forecast,colors[1],'s','Forecast response' if y==5 else None)
    left.set_yticks(range(6),labels[::-1]); left.set_ylim(-.5,5.5)
    left.set_title('Soft shoulder minus hard TopK',loc='left',fontweight='bold')
    left.legend(loc='upper left',bbox_to_anchor=(0,-.20),ncol=2,frameon=False,fontsize=9)
    for y,(a,b) in enumerate([('timesfm','bolt'),('bolt','timesfm')]):
        h=fixed[a+'_to_'+b]['frequent_sae_minus_isotropic']
        f=fixed_beh['reserve_fixed_'+b+'_forecasts.json']['methods']['frequent_sae_minus_isotropic']['mean_item_cosine']
        whisker(right,y+.09,h,colors[0],'o'); whisker(right,y-.09,f,colors[1],'s')
    right.set_yticks([0,1],['TimesFM → Bolt','Bolt → TimesFM']); right.set_ylim(-.5,1.5)
    right.set_title('Frequent SAE axes minus isotropic axes',loc='left',fontweight='bold',fontsize=10)
    for ax in (left,right):
        ax.axvline(0,color='#73808A',lw=1,ls='--'); ax.grid(axis='x',alpha=.18)
        ax.set_xlim(-.08,.42); ax.set_xticks([0,.1,.2,.3,.4]); ax.set_xlabel('Change in mean signed cosine')
        ax.spines[['top','right','left']].set_visible(False); ax.tick_params(axis='y',length=0,labelsize=9)
    fig.savefig(HERE/'paired_intervals.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def crossing_figure():
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),sharex=True,sharey=True)
    colours={"dense":"#20567A","hard":"#A54320","shoulder":"#32835C"}
    for ax,(model,title) in zip(axes,[("bolt","Chronos-Bolt"),("timesfm","TimesFM"),("chronos2","Chronos-2")]):
        case=read("crossing_"+model+".json")
        stages=[case["stages"][n] for n in ("4","8","12")]
        x=[s["input_rms_last64"]["median"] for s in stages]
        for arm,label in [("hard","Hard TopK"),("shoulder","Smooth shoulder"),("dense","Native dense")]:
            y=[s["norm_over_full_native"][arm]["median"] for s in stages]
            ax.loglog(x,y,marker="o",markersize=4,color=colours[arm],label=label)
        ax.set_title(f"{title}   n = {len(case['ids'])}",fontsize=10,fontweight="bold")
        ax.grid(which="major",alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].invert_xaxis()
    axes[0].set_ylabel("Median displacement norm\nrelative to the full-dose native edit")
    fig.supxlabel("Actual input RMS between the two sides of a bracket (history SD)",fontsize=10,y=.13)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc="lower center",frameon=False)
    fig.subplots_adjust(bottom=.32,wspace=.15,top=.9,left=.09,right=.985)
    fig.savefig(HERE/"crossing_geometry.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__ == '__main__':
    figure()
    if (HERE/"crossing_chronos2.json").exists():
        crossing_figure()
