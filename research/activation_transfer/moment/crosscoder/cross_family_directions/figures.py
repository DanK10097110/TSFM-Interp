"""Conditional uncertainty and a source-only-selected physical example."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.activation_transfer.moment.crosscoder.cross_family_directions.data import OUT,METHODS,SEEDS
from research.activation_transfer.moment.crosscoder.cross_family_directions.directions import native_directions

def plots():
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
    method_labels=['Joint','Masked','Masked, shuffled pairs']
    palette=['#007f78','#a15138']
    for ax,domain,title in zip(axes,['daily','monash_traffic'],['M4 daily','Hourly traffic']):
        for mi,m in enumerate(METHODS):
            for si,s in enumerate(SEEDS):
                f=json.loads((OUT/f'{m}_{s}_clustered.json').read_text())[domain]['all_observed_coordinates']
                for feature,(k,gap) in enumerate([('patch32_adv',-.12),('forecast16_adv',.12)]):
                    avg=f[k]['mean'];low,high=f[k]['history_atom_95']
                    ax.errorbar(mi+gap+(si-.5)*.082,avg,yerr=[[avg-low],[high-avg]],
                                color=palette[feature],marker='o' if si==0 else 's',
                                markersize=4,capsize=2,linewidth=1,label=['Input patch','Forecast'][feature] if mi==si==0 else None)
        ax.axhline(0,color='0.55',lw=.8);ax.set_xticks(range(3),method_labels,rotation=12,ha='right')
        ax.set_title(title,loc='left',fontweight='bold');ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('Matching minus other active feature (cosine)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Paired directions agree more on the observed patch than on the forecast',fontweight='bold')
    fig.savefig(OUT/'patch_forecast.png',dpi=170,bbox_inches='tight');plt.close(fig)
    f=dict(np.load(OUT/'masked_22112401_precision.npz'));ix=np.flatnonzero(f['domains']=='daily')
    source=f['source'][ix];shape=source-source.mean(1,keepdims=True)
    rms=np.sqrt(np.mean(shape**2,axis=1))/f['history_sd'][ix]
    j=ix[np.argmin(np.abs(rms-np.median(rms)))]
    atom=f['feature'][j];dose=f['dose'][j]
    with np.load(OUT/'masked_22112401_dict.npz') as dictionary:s,t=native_directions(dictionary)
    with np.load(OUT/'ridge.npz') as ridge:
        patch_s=-dose*(s[atom]@ridge['timesfm_probe']);patch_t=-dose*(t[atom]@ridge['bolt_probe'])
    fig,axs=plt.subplots(1,2,figsize=(9.7,3.25),layout='constrained')
    for name,y,col in [('TimesFM',patch_s,palette[0]),('Bolt, paired',patch_t,palette[1])]:
        axs[0].plot(np.arange(-31,1),y-y.mean(),label=name,color=col,lw=2)
    axs[0].set_title('Inferred change to the 32-step patch',loc='left',fontsize=11,fontweight='bold')
    axs[0].set_xlabel('Position relative to history endpoint');axs[0].set_ylabel('Centered change (history SD)')
    colors=[palette[0],palette[1],'#62548a']
    for (key,name),color in zip([('source','TimesFM'),('paired','Bolt, paired'),('dense','Bolt, ordinary map')],colors):
        y=f[key][j]/f['history_sd'][j];axs[1].plot(np.arange(1,17),y-y.mean(),label=name,color=color,lw=2,ls='--' if key=='dense' else '-')
    axs[1].set_title('Actual change in the 16-step forecast',loc='left',fontsize=11,fontweight='bold')
    axs[1].set_xlabel('Forecast lead');axs[1].set_ylabel('Centered change (history SD)')
    for ax in axs:
        ax.axhline(0,color='0.65',lw=.7);ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'M4 {f["ids"][j]}, feature {atom}, masked crosscoder seed 1',fontweight='bold',fontsize=12)
    fig.savefig(OUT/'median_source_example.png',dpi=170,bbox_inches='tight');plt.close(fig)
    def corr(a,b):return float(np.corrcoef(a,b)[0,1])
    record=dict(id=str(f['ids'][j]),row=int(f['rows'][j]),feature=int(atom),dose=float(dose),
        selection='Median source forecast centered RMS on daily, first masked seed; no recipient outcomes used.',
        source_centered_rms_history_sd=float(rms[np.flatnonzero(ix==j)[0]]),
        input_patch_cosine=corr(patch_s,patch_t),forecast_cosine=corr(f['source'][j],f['paired'][j]),
        patch_source=patch_s.tolist(),patch_target=patch_t.tolist(),source_forecast=f['source'][j].tolist(),
        paired_forecast=f['paired'][j].tolist(),dense_forecast=f['dense'][j].tolist(),sd=float(f['history_sd'][j]))
    (OUT/'example.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':plots()
