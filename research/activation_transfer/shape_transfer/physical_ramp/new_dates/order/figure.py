"""Physical perturbations, real recipient waveforms, and the paired order test."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.order.study import ORDER,OUT

def make(out):
    data=dict(np.load(out/'order.npz'))
    est=json.loads((out/'scores.json').read_text())['metrics']['early_centered']
    fig,axs=plt.subplots(1,3,figsize=(12.3,3.6),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.2]})
    teal='#007a73';orange='#a95232';ink='#293f49'
    ramp=np.linspace(-2,2,8);days=np.arange(-7,1)
    for delta,name,cl,style in ((ramp,'Rising ramp',teal,'-'),(ramp[ORDER],'Reversed interior',orange,'--')):
        axs[0].plot(days,delta,color=cl,ls=style,marker='o',ms=3.5,lw=2,label=name)
    axs[0].axhline(0,color='0.7',lw=.8);axs[0].set_xticks([-7,-5,-3,-1,0]);axs[0].set_xlabel('Day of the observed history (0 = last)')
    axs[0].set_ylabel('Temperature added (°C)');axs[0].legend(frameon=False,fontsize=8,loc='upper left')
    for goal,cl,ls in [('A',teal,'-'),('B',orange,'--')]:
        axs[1].plot(range(1,17),data['truth_'+goal].mean(0),color=cl,lw=2,ls=ls)
    axs[1].set_xticks([1,4,8,12,16]);axs[1].set_xlabel('Forecast lead (days)');axs[1].set_ylabel('Mean forecast change (°C)')
    for j,(key,cl) in enumerate((('advantage_A',teal),('advantage_B',orange),('balanced_advantage',ink))):
        v=est[key]['value'];lo,hi=est[key]['crossed95']
        axs[2].errorbar(v,j,xerr=[[v-lo],[hi-v]],color=cl,marker='o',capsize=3)
    axs[2].axvline(0,color='0.65',lw=.8);axs[2].set_yticks(range(3),['Rising target','Reversed target','Balanced mean'])
    axs[2].invert_yaxis();axs[2].set_ylim(2.6,-.55);axs[2].set_xlabel('Correct minus swapped shape skill')
    for a,title in zip(axs,['Two edits, same values','Bolt’s native isolated response','Early order-matching advantage']):
        a.set_title(title,loc='left',fontsize=10.5,weight='bold');a.spines[['top','right']].set_visible(False)
    fig.savefig(out/'interior_order.png',dpi=165,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);make(p.parse_args().out)
