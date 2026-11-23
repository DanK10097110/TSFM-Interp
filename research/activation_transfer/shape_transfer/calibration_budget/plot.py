"""Show the paired cost tradeoff and uncertainty on matched daily and hourly cohorts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
COLORS={'4x24':'#12636b','12x8':'#d38e21','16x24':'#12636b','48x8':'#d38e21','pilot':'#7c53a5','baseline':'#3d4449'}
LABELS={'4x24':'4 × 24','12x8':'12 × 8','16x24':'16 × 24','48x8':'48 × 8','pilot':'Two-stage','baseline':'96 × 24'}


def plot(out):
    scores=json.loads((out/'scores.json').read_text())['cohorts']
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),sharex=True,sharey=True)
    models=[('bolt','Bolt → TimesFM'),('timesfm','TimesFM → Bolt')]
    arms=tuple(COLORS);y=np.arange(len(arms))[::-1]
    for col,(source,title) in enumerate(models):
        data=scores[source]['fresh']
        for row,(domain,subset) in enumerate(data.items()):
            ax=axes[row,col];cell=subset['centered']['arms'];base=cell['baseline']['skill']['value']
            for i,arm in enumerate(arms):
                r=cell[arm]['skill'];lo,hi=r['ci95'];v=r['value'];color=COLORS[arm]
                ax.plot([lo,hi],[y[i],y[i]],color=color,lw=2)
                ax.scatter(v,y[i],color=color,s=42,zorder=3)
            ax.axvline(0,color='#bbbbbb',lw=1);ax.axvline(base,color='#bbbbbb',lw=1,ls=':');ax.grid(axis='x',alpha=.17)
            ax.set_yticks(y,labels=[LABELS[k] for k in arms]);ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(axis='y',length=0);ax.set_ylim(-.6,5.7)
            ax.set_title(title+'  ·  '+('M4 daily' if domain=='daily' else 'Hourly traffic'),loc='left',fontsize=13,weight='bold')
            if row:ax.set_xlabel('Horizon-centered recovery skill')
    fig.suptitle('How much target calibration preserves forecast shape?',fontsize=17,weight='bold',x=.057,ha='left')
    fig.text(.057,.92,'Rows show histories × directions. Bars are 95% whole-history intervals. Reduced budgets average three fits.',fontsize=10)
    fig.subplots_adjust(left=.10,right=.98,top=.845,bottom=.09,hspace=.36,wspace=.17)
    fig.savefig(out/'budget_validation.png',dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args();plot(a.out)
