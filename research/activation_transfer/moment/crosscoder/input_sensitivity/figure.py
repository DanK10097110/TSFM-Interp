"""Plot physically measured, paired-minus-alternative response agreement with crossed intervals."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.activation_transfer.moment.crosscoder.input_sensitivity.capture import OUT


def main():
    data=json.loads((OUT/'summary_0.15.json').read_text())['measurements']
    methods=[('masked','Source-masked'),('joint','Standard joint'),('shuffled','Shuffled histories')]
    colors={'masked':'#087F78','joint':'#2E5684','shuffled':'#916D54'}
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),sharex=True,sharey=True)
    for ax,(domain,title) in zip(axes,[('daily','M4 daily'),('monash_traffic','Monash traffic')]):
        ax.axvline(0,color='#505967',lw=.9,ls='--')
        for i,(method,label) in enumerate(methods):
            y=len(methods)-1-i
            for seed,offset,marker in [(22112401,.11,'o'),(22112402,-.11,'s')]:
                v=data[f'{method}_{seed}'][domain]['statistics']['margin'];mean=v['event_mean'];lo,hi=v['history_atom95']
                ax.errorbar(mean,y+offset,xerr=[[mean-lo],[hi-mean]],fmt=marker,color=colors[method],ms=5,capsize=3,lw=1.4)
        ax.set_title(title,fontsize=11,fontweight='bold');ax.set_xlim(-.15,.41);ax.set_ylim(-.45,2.45)
        ax.set_yticks([2,1,0],labels=[item[1] for item in methods]);ax.grid(axis='x',alpha=.15)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.suptitle('Input-response specificity of paired MOMENT directions',fontsize=13,fontweight='bold',x=.53,y=.99)
    fig.supxlabel('Same-coordinate cosine minus the mean across 31 other active coordinates',fontsize=10,y=.12)
    handles=[Line2D([0],[0],marker=m,color='#475164',lw=0,label=l) for m,l in [('o','Seed 1'),('s','Seed 2')]]
    fig.legend(handles=handles,ncol=2,loc='lower center',frameon=False,fontsize=9,bbox_to_anchor=(.53,-.006))
    fig.subplots_adjust(left=.20,right=.98,top=.80,bottom=.29,wspace=.14)
    fig.savefig(OUT/'native_input_sensitivity.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)


if __name__=='__main__':main()
