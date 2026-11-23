"""Plot FIT output spectra and the held-out error decomposition at the matched write budget."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def plot(root):
    scores=json.loads((root/'scores.json').read_text())
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.2,3.8),gridspec_kw={'width_ratios':[1,1.6]})
    fig.subplots_adjust(left=.085,right=.985,top=.88,bottom=.23,wspace=.36)
    for m,c in [('bolt','#1d6177'),('timesfm','#b45728')]:
        d=json.loads((root/(m+'_geometry.json')).read_text())['geometry']
        sv=np.array(d['pooled_output_rms_singular_values']);ax.semilogy(np.arange(1,9),sv/sv[0],marker='o',lw=1.6,ms=4,color=c,label={'timesfm':'TimesFM','bolt':'Bolt'}[m])
    ax.set_title('FIT forecast sensitivities',fontweight='bold');ax.set(xlabel='Output singular direction',ylabel='Sensitivity relative to the strongest',xticks=[1,2,4,6,8]);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    x=np.array([0,1,2,3.6,4.6,5.6]);dc=[];shape=[]
    for m in ['bolt','timesfm']:
        for a in ['native','mapped','corrected']:
            row=scores[m]['daily']['equal_budget']
            dc.append(np.mean([row[c][a]['dc_error']['mean'] for c in ['tilt','sine']]))
            shape.append(np.mean([row[c][a]['shape_error']['mean'] for c in ['tilt','sine']]))
    bx.bar(x,shape,color='#4c788d',width=.74,label='Centered shape error');bx.bar(x,dc,bottom=shape,width=.74,color='#d8a053',label='Level error')
    bx.axhline(1.,ls='--',color='#424242',lw=1,label='Zero-edit error');bx.set(ylabel='Squared error / requested energy',xticks=x,xticklabels=['Native','Mapped','Corrected']*2,ylim=(0,1.66))
    bx.tick_params(axis='x',labelsize=8);bx.text(1,-.12,'Into Bolt',transform=bx.get_xaxis_transform(),ha='center',fontweight='bold',fontsize=9);bx.text(4.6,-.12,'Into TimesFM',transform=bx.get_xaxis_transform(),ha='center',fontweight='bold',fontsize=9)
    bx.set_title('Daily test, equal tilt/sine weights',fontweight='bold');bx.legend(frameon=False,loc='upper right',fontsize=8,ncol=2)
    for axis in [ax,bx]:axis.spines[['top','right']].set_visible(False)
    fig.savefig(root/'level_shape.png',bbox_inches='tight',dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':plot(HERE/'results')
