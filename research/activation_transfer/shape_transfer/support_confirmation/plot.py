"""Summarize cross-receiver physical span controls and within-Bolt orthogonalization."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.activation_transfer.shape_transfer.support_confirmation.data import OUT


def plot(out):
    support=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    residual=json.loads((out/'residual.json').read_text(encoding='utf-8'))
    derivative=json.loads((out/'derivative.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    colors={'bolt':'#147675','timesfm':'#88429c'}
    def points(ax,values,labels,color,shift=0,label=None):
        y=np.arange(len(labels))[::-1]+shift
        for i,(v,yy) in enumerate(zip(values,y)):
            lo,hi=v['crossed95']
            ax.plot([lo,hi],[yy,yy],color=color,linewidth=1.7)
            ax.scatter(v['value'],yy,color=color,s=26,label=label if i==0 else None,zorder=3)
        ax.set_yticks(np.arange(len(labels))[::-1],labels)
        ax.tick_params(axis='y',length=0);ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',alpha=.19)
    labels=['Native active','Other histories','Frequent training IDs','Clean-trained PCA','Entirely nonnative IDs','Isotropic']
    ks=['native','mean_swap','frequent','pca','own_0','isotropic']
    for target,off,name in [('bolt',.115,'Chronos-Bolt'),('timesfm',-.115,'TimesFM')]:
        block=support['short'][target]['endpoints']['early_shape_dense']
        points(axes[0],[block[k] for k in ks],labels,colors[target],off,name)
    axes[0].legend(loc='lower right',frameon=False,fontsize=9);axes[0].set_title('Which 32 axes?',loc='left',weight='bold')
    labels=['Nonnative IDs','Nonnative off native','Nonnative off isotropic','Native span','Native off nonnative','PCA off native']
    block=residual['endpoints']['early_dense']
    points(axes[1],[block[k] for k in ['own_0','zero_off_native','zero_off_random','native','native_off_zero','pca_off_native']],labels,colors['bolt'])
    axes[1].set_title('Removing shared geometry in Bolt',loc='left',weight='bold')
    labels=['Both current','Other support','Other derivative','Both reassigned','Historical derivative']
    for target,off in [('bolt',.115),('timesfm',-.115)]:
        block=derivative[target]['early_shape_dense']['balanced']
        points(axes[2],[block[k] for k in ['NN','SN','NS','SS','cal_N']],labels,colors[target],off)
    axes[2].set_title('Which information must be current?',loc='left',weight='bold')
    for ax in axes:
        ax.set_xlabel('Early centered recovery')
        ax.set_xlim(.05,.9)
        ax.margins(y=.14)
    fig.suptitle('Matched-budget forecast interventions on 160 new daily histories',fontsize=16,weight='bold')
    fig.savefig(out/'support_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);plot(p.parse_args().out)
