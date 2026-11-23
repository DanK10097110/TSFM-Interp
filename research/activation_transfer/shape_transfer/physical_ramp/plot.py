"""The uncentered mean Celsius response over the complete, prespecified station sample."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.study import OUT


def plot(out):
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for source,ax,title in zip(('bolt','timesfm'),axs,('Bolt → TimesFM','TimesFM → Bolt')):
        data=dict(np.load(out/(source+'_transfer.npz')))
        day=np.arange(1,17)
        for k,label,col,line in [('truth','Receiver’s own isolated ramp','#202e37','-'),
                                 ('mapped','Correct mapped ramp','#007b77','-'),
                                 ('wrong_step','Mapped +2°C step, same norm','#b35633','--'),
                                 ('other_station','Other station ramp, same norm','#665594',':')]:
            ax.plot(day,data[k].mean(0),color=col,linestyle=line,lw=2,label=label)
        ax.axhline(0,color='#8c9494',lw=.75);ax.axvline(8.5,color='#a6aaa9',lw=.8)
        ax.spines[['right','top']].set_visible(False);ax.set_xticks([1,4,8,12,16]);ax.set_xlabel('Forecast lead (days)')
        ax.set_ylabel('Change in forecast (°C)');ax.set_title(title,loc='left',weight='bold',fontsize=11)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    fig.suptitle('Mean native forecast change from a final eight-day temperature ramp',weight='bold',fontsize=13)
    fig.savefig(out/'ramp_waveforms.png',dpi=165,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();plot(a.out)
