"""Display clustered uncertainty for the independent weather histories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.activation_transfer.shape_transfer.support_confirmation.noaa.data import OUT


def run(out):
    report=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharex=True,sharey=True,layout='constrained')
    categories=['Native active 32','Most frequent 32','Zero native IDs'];positions=np.arange(3)[::-1]
    for ax,(target,title) in zip(axes,[('bolt','Chronos-Bolt'),('timesfm','TimesFM')]):
        cell=report[target]['statistics']['early_shape_dense']
        for policy,label,color,off in [('current','Current station','#126c71',.10),('wrong','Other station, same season','#a14932',-.10)]:
            for i,(name,yy) in enumerate(zip(('native','frequent','zero'),positions)):
                m=cell[name+'_'+policy];lo,hi=m['ci95']
                ax.plot([lo,hi],[yy+off,yy+off],color=color,lw=2);ax.scatter(m['value'],yy+off,color=color,s=26,label=label if i==0 else None)
        ax.set_title(title+' receiver',loc='left',weight='bold')
        ax.set_yticks(positions,categories);ax.tick_params(axis='y',length=0)
        ax.spines[['left','right','top']].set_visible(False);ax.grid(axis='x',alpha=.22);ax.margins(y=.22)
        ax.set_xlabel('Early centered recovery of dense intervention')
    fig.suptitle('Which derivative and support transfer across 46 US weather stations?',weight='bold',fontsize=13)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(out/'weather_transfer.png',dpi=165,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);run(p.parse_args().out)
