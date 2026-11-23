"""Print every estimate and distinguish equal-feature from response-energy weighting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
import numpy as np
from research.activation_transfer.shape_transfer.cross_seed_groups.study import ARM,OUT


def run(out):
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.6),layout='constrained')
    cm=LinearSegmentedColormap.from_list('skill',['#a64032','#fffdf9','#176e6c']);norm=Normalize(-1,1,clip=True)
    panels=[(n,p) for n in ('bolt','timesfm') for p in ('daily','monash_traffic','noaa')]
    for ax,avg,title in zip(axes,('equal','pooled'),('Each source feature weighted equally','Pooled by source waveform energy')):
        vals=[];ticks=[]
        for model,p in panels:
            r=json.loads((out/(model+'_'+p+'.json')).read_text());d=r['score']['h16_centered']['arms']
            vals.append([d[k][avg]['skill'] for k in ARM]);ticks.append(('Bolt' if model=='bolt' else 'TimesFM')+' · '+{'daily':'Daily','monash_traffic':'Traffic','noaa':'Weather'}[p])
        matrix=np.array(vals)
        ax.imshow(matrix,cmap=cm,norm=norm,aspect='auto')
        for i,row in enumerate(matrix):
            for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',size=8,color='white' if abs(v)>.67 else '#20272a')
        ax.set_xticks(range(6),('Matched\nown','Matched\nforced','Rank\n1','Rank\n4','Rank\n16','Rotated\nrank 16'),fontsize=8)
        ax.set_yticks(range(6),ticks,fontsize=8);ax.tick_params(length=0);ax.set_title(title,loc='left',size=10,weight='bold')
        ax.axhline(2.5,color='white',lw=3)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Can a second SAE reproduce a native forecast change?',fontsize=14,weight='bold')
    fig.supxlabel('Centered 16-lead skill. 1 = exact, 0 = unchanged. Colour saturates below −1.',fontsize=9)
    fig.savefig(out/'seed_groups.png',dpi=165,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.out)
