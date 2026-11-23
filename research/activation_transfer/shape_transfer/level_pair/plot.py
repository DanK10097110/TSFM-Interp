"""Compare selective code responses with actual forecast lift attributed to the named decoder term."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def plot(out):
    scores=json.loads((out/'scores.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,7.4),sharex=False)
    cues=['recent','last','slope','translation'];labels=['Recent mean','Last value','Recent slope','Translation'];y=np.arange(4)[::-1]
    for row,(model,title) in enumerate([('bolt','Chronos-Bolt 5432'),('timesfm','TimesFM 742')]):
        for col in range(2):
            ax=axes[row,col];ax.axvline(0,color='#8f969c',lw=1);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
            for domain,color,offset,label in [('daily','#12636b',.10,'M4 daily'),('monash_traffic','#9046a0',-.10,'Hourly traffic')]:
                for q,cue in enumerate(cues):
                    m=scores[model][domain][cue]['all'];metric=m['codes']['half_change'] if col==0 else m['arms']['named']['removed_level']
                    lo,hi=metric['ci95'];ax.plot([lo,hi],[y[q]+offset]*2,lw=2,color=color)
                    ax.scatter(metric['value'],y[q]+offset,s=30,color=color,label=label if not q else None,zorder=3)
            ax.set_yticks(y,labels=labels);ax.tick_params(axis='y',length=0);ax.spines[['top','right','left']].set_visible(False)
            ax.set_title(title+'  ·  '+('named activation' if col==0 else 'forecast level'),fontsize=12,loc='left',weight='bold')
            if row:ax.set_xlabel('Signed half-change in SAE activation' if col==0 else 'Lift removed by the named lock (history SD)')
    fig.suptitle('Does the recent-level feature carry the input change into the forecast?',fontsize=16,x=.055,ha='left',weight='bold')
    handles,leg=axes[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper left',bbox_to_anchor=(.05,.945),ncol=2,frameon=False)
    fig.subplots_adjust(left=.108,right=.98,top=.84,bottom=.09,hspace=.41,wspace=.28)
    fig.savefig(out/'named_pair.png',dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args();plot(a.out)
