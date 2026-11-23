"""Show all eight measured domain-residual hybrid cells on matched cue values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.activation_transfer.shape_transfer.level_pair.inputs import HERE


def plot(out):
    scores=json.loads((out/'splice.json').read_text())
    figure,axes=plt.subplots(1,2,figsize=(10,4.1),sharey=True)
    for axis,model,label in zip(axes,['bolt','timesfm'],['Chronos-Bolt 5432','TimesFM 742']):
        cells=scores[model]['0']['on']['cells']
        # Columns group the source of pattern in the original early block and the energy share in the recent block.
        grid=np.array([[cells[str(a+2*b+4*c)]['fires'] for a,c in [(0,0),(1,0),(0,1),(1,1)]] for b in (0,1)])
        axis.imshow(grid,aspect='auto',vmin=0,vmax=55,cmap='Blues')
        for j in range(2):
            for i in range(4):
                axis.text(i,j,str(grid[j,i]),ha='center',va='center',
                          color='white' if grid[j,i]>29 else '#163b53',weight='bold',fontsize=14)
        axis.set_title(label,loc='left',weight='bold',fontsize=12)
        axis.set_xticks(range(4),labels=['Traffic\nTraffic','Daily\nTraffic','Traffic\nDaily','Daily\nDaily'])
        axis.set_yticks([0,1],labels=['Traffic','Daily'])
        axis.tick_params(length=0)
        for edge in axis.spines.values():edge.set_visible(False)
        axis.set_xlabel('Earlier 448 pattern (top label)\nRecent share of residual energy (bottom label)',fontsize=9)
    axes[0].set_ylabel('Final 32 pattern')
    figure.suptitle('Which input swaps recover named-feature firing?',x=.075,ha='left',weight='bold',fontsize=14,y=.99)
    figure.text(.075,.89,'Counts out of 128 pairs at identical relative recent mean, endpoint and fitted rise.',ha='left',fontsize=10)
    figure.subplots_adjust(left=.09,right=.99,top=.74,bottom=.28,wspace=.15)
    figure.savefig(out/'splice.png',dpi=160,bbox_inches='tight')
    plt.close(figure)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'results')
    plot(p.parse_args().out)
