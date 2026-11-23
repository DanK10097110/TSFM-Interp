"""Choose an illustration using only donor firing records and the donor's independently measured finite dose."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from research.activation_transfer.moment.dictionary_transfer.weather.study import OUT,SEEDS
from research.activation_transfer.moment.dictionary_transfer.weather.score import centered,matched_effect
from research.activation_transfer.shape_transfer.support_confirmation.noaa.score import eligible_mask


def select(codes,station,rows,cols,donor):
    eligible=eligible_mask(codes>0,station)
    if not eligible.any():raise ValueError('No donor atom fired at eight independent stations')
    j=int(np.argmax(np.where(eligible,(codes>0).sum(0),-1)))
    events=np.flatnonzero(cols==j)
    rms=np.sqrt(np.mean(centered(donor[events])**2,axis=1))
    i=events[np.argmin(np.abs(rms-np.median(rms)))]
    return int(i),float(np.median(rms))


def run(out):
    with np.load(out/(str(SEEDS[0])+'.npz')) as f:z=dict(f)
    orig=matched_effect(z,'donor')
    i,med=select(z['z'],z['station'],z['rows'],z['cols'],orig);r=z['rows'][i];c=z['cols'][i]
    waves={key:matched_effect(z,key)[i] for key in ('donor','corr_own','group16')}
    doc=dict(seed=SEEDS[0],event=i,item=str(z['ids'][r]),atom=int(z['atoms'][c]),recipient_id=int(z['matched_ids'][c]),
             source_fires=int((z['cols']==c).sum()),median_source_centered_rms=med,
             selected_source_centered_rms=float(np.sqrt(np.mean(centered(orig[i])**2))),source_coefficient=float(z['z'][r,c]),
             recipient_coefficient=float(z['target_z'][r,c]),history_sd_celsius=float(z['input_sd'][r]),
             waveforms_celsius={k:v.tolist() for k,v in waves.items()},
             centered_skills={k:float(1-np.sum((centered(v)-centered(waves['donor']))**2)/np.sum(centered(waves['donor'])**2))
                              for k,v in waves.items()})
    (out/'example.json').write_text(json.dumps(doc,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7.4,3.6),layout='constrained')
    styles=[('donor','Natural donor removal','#203e53','o'),('corr_own','Matched atom, own coefficient','#bb573b','s'),('group16','Fixed 16-axis group','#107b70','^')]
    for name,label,color,marker in styles:ax.plot(np.arange(1,17),waves[name],label=label,c=color,marker=marker,ms=3.8,lw=1.6)
    ax.axhline(0,color='#707777',lw=.7);ax.set_xticks([1,4,8,12,16]);ax.set_xlabel('Forecast lead (days)')
    ax.set_ylabel('Change in forecast (°C)');ax.grid(axis='y',alpha=.22)
    ax.spines[['right','top']].set_visible(False)
    display=doc['item'].replace('_',' ').replace('/',', ')
    ax.set_title(f"Removing donor atom {doc['atom']} at {display}",loc='left',fontsize=11,weight='bold')
    ax.legend(frameon=False,ncol=3,fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.36))
    fig.savefig(out/'example_weather.png',dpi=165,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.out)
