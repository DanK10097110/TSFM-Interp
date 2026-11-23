"""Per-feature finite forecast recovery in the two directions of the same dictionary pair."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
import numpy as np

from research.activation_transfer.moment.dictionary_transfer.weather.study import OUT,SEEDS


def run(out):
    report=json.loads((out/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,6.3),layout='constrained')
    cm=LinearSegmentedColormap.from_list('wave',['#a43f2e','#fffdf9','#16706d'])
    norm=Normalize(-1,1,clip=True)
    keys=('corr_own','corr_forced','group16','geom_forced','random')
    for ax,seed,title in zip(axes,SEEDS,('SAE 1 donor → SAE 2','SAE 2 donor → SAE 1')):
        cell=report[str(seed)];atoms=[f for f in cell['features'] if f['sufficient']]
        measures=cell['statistics']['history_sd_centered']['arms']
        pos=np.arange(len(atoms))
        v=np.array([[measures[k]['per_atom'][str(row['id'])] for k in keys] for row in atoms])
        ax.imshow(v,cmap=cm,norm=norm,aspect='auto',interpolation='nearest')
        for i,row in enumerate(v):
            for j,value in enumerate(row):
                ax.text(j,i,f'{value:.2f}',ha='center',va='center',size=7.5,color='white' if abs(value)>.65 else '#202729')
        ax.set_xticks(np.arange(len(keys)),['Matched\nown dose','Matched\nforced','Fixed\ngroup','Closest\ndecoder','Random'])
        ax.set_yticks(pos,[f"{x['id']}  ({x['sites']})" for x in atoms])
        ax.set_title(title,weight='bold',loc='left')
        ax.set_ylabel('Donor feature (number of stations)')
        ax.tick_params(axis='both',length=0,labelsize=8)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Can another SAE reproduce a natural 16-day forecast change?',fontsize=14,weight='bold')
    fig.supxlabel('Centered squared-error skill. 1 = exact, 0 = unchanged. Colours saturate below −1.',size=9)
    fig.savefig(out/'per_atom_weather.png',dpi=165,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.out)
