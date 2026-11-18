"""Publication-style descriptive summary from existing result JSON only.

All fixed pure/mixed conditions and all period/context grid cells are shown.
Supervised head-input recoverability and the failed tangent retention gate are
distinguished from native zero-shot forecast performance and causal claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def build(replication,readout,boundary,tangent,branches,out,portable_source=None):
    """Render fixed stored measurements;do not load models or recompute estimates."""
    paths=dict(replication=replication,readout=readout,boundary=boundary,tangent=tangent,branches=branches)
    if portable_source is None:
        data={k:json.loads(p.read_text(encoding='utf-8')) for k,p in paths.items()};source_files={key:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for key,path in paths.items()}
    else:
        portable=json.loads(portable_source.read_text(encoding='utf-8'));data=portable['source_results'];source_files=portable['source_files']
    out.mkdir(parents=True,exist_ok=False)
    colors={'bolt':'#C46A28','timesfm':'#28659C'};labels={'bolt':'Bolt','timesfm':'TimesFM'};plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13.2,8.6));ax_a,ax_b,ax_c,ax_d=axes.ravel();conditions=['pure_phi','pure_psi','mixed'];a_values={}
    for model in ['bolt','timesfm']:
        values=[data['replication']['scaled_mse'][model][c] for c in conditions];a_values[model]=values;ax_a.plot(range(3),values,'o-',color=colors[model],label=labels[model],lw=1.7,ms=5)
    ax_a.set_xticks(range(3),['Pure 32','Pure 16','32 + 16']);ax_a.set_ylabel('Native MSE / shared mixed energy');ax_a.set_title('A  Model ranking changes with composition',loc='left',fontsize=10);ax_a.legend(frameon=False);ax_a.set_ylim(bottom=0);ax_a.grid(axis='y',alpha=.18)
    interaction=data['replication']['primary']['model_by_composition_interaction'];ax_a.text(.03,.97,f'128 matched seed bundles\nPaired interaction: {interaction["value"]:.3f}\n95% CI [{interaction["lo"]:.3f}, {interaction["hi"]:.3f}]',transform=ax_a.transAxes,va='top',fontsize=8)
    b_values={}
    for i,model in enumerate(['bolt','timesfm']):
        stored=data['readout']['models'][model]['datasets']['fresh'];metrics=[stored['native_scaled_mse'],stored['probe_scaled_mse']];xs=np.array([0,1])+(-.055 if i==0 else .055);ys=np.array([m['mean'] for m in metrics]);lo=np.array([m['lo'] for m in metrics]);hi=np.array([m['hi'] for m in metrics]);ax_b.errorbar(xs,ys,yerr=[ys-lo,hi-ys],fmt='o--',color=colors[model],capsize=3,label=labels[model],lw=1.3,ms=5);b_values[model]=metrics
    ax_b.set_yscale('log');ax_b.set_ylim(1e-4,1);ax_b.set_xticks([0,1],['Native forecast','Frozen linear probe']);ax_b.set_xlim(-.25,1.25);ax_b.set_ylabel('Mixed-series MSE / mixed energy');ax_b.set_title('B  Spectral information survives at head input',loc='left',fontsize=10);ax_b.grid(axis='y',alpha=.18)
    ax_b.text(.03,.04,'Old-trained probe; 128 fresh bundles\nKnown-frequency input oracle ≈ 2 × 10⁻¹⁶',transform=ax_b.transAxes,fontsize=8,va='bottom')
    periods=['31','32','33'];contexts=['128','512'];grid=np.empty((3,6));grid_rows=[]
    for j,(period,context) in enumerate((p,c) for p in periods for c in contexts):
        for i,condition in enumerate(conditions):
            values={model:data['boundary']['conditions'][period][model]['contexts'][context]['mse_common_theoretical_energy'][condition] for model in ['bolt','timesfm']};gap=values['timesfm']['value']-values['bolt']['value'];grid[i,j]=gap;grid_rows.append(dict(period=int(period),context=int(context),condition=condition,native_losses=values,gap_timesfm_minus_bolt=gap))
    maximum=float(abs(grid).max());cmap=LinearSegmentedColormap.from_list('model_gap',[colors['timesfm'],'#FAFAFA',colors['bolt']]);image=ax_c.imshow(grid,cmap=cmap,vmin=-maximum,vmax=maximum,aspect='auto')
    ax_c.set_xticks(range(6),[f'{p}\n{c}' for p in periods for c in contexts]);ax_c.set_yticks(range(3),['Pure period P','Pure period 16','P + 16']);ax_c.set_xlabel('Period P (top) / context steps (bottom)');ax_c.set_title('C  Context boundary depends on period',loc='left',fontsize=10)
    for i in range(3):
        for j in range(6):ax_c.text(j,i,f'{grid[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(grid[i,j])>.6*maximum else '#222222')
    for x in [1.5,3.5]:ax_c.axvline(x,color='white',lw=2)
    colorbar=fig.colorbar(image,ax=ax_c,fraction=.047,pad=.025);colorbar.set_label('Native MSE gap: TimesFM − Bolt\nnegative favors TimesFM',fontsize=8)
    for i,model in enumerate(['bolt','timesfm']):
        effect=data['branches']['models'][model]['deltas']['0.01']['branches']['skip']['signed_projection_share'];ax_d.errorbar(i,effect['value'],yerr=[[effect['value']-effect['lo']],[effect['hi']-effect['value']]],fmt='o',color=colors[model],capsize=4,ms=6);ax_d.text(i,effect['hi']+.025,f'{effect["value"]:.4f}',ha='center',fontsize=9)
    ax_d.set_xticks([0,1],['Bolt','TimesFM']);ax_d.set_xlim(-.6,1.6);ax_d.set_ylim(0,1.08);ax_d.set_ylabel('Signed projection share of native off-frequency derivative');ax_d.set_title('D  Linear head skip carries most observed mixing',loc='left',fontsize=10);ax_d.grid(axis='y',alpha=.18)
    ax_d.text(.03,.06,'128 cached bundles; exact native-head replay\nδ = 0.01; descriptive branch attribution\nNot a share of forecast error. Cross energies are retained.',transform=ax_d.transAxes,fontsize=8,va='bottom')
    for ax in [ax_a,ax_b,ax_d]:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Spectral composition: native forecasts, recoverable information, and context',fontsize=13,y=.99)
    tf_gate=data['tangent']['models']['timesfm']['targets']['psi']['deltas']['0.01']['probe_offtarget_relative_RMS'];footer=(
        'A: all pure/mixed means, common mixed-energy normalization; 128 bundles. B: 95% seed-bootstrap CIs; probes use extra synthetic labels, known periods, and external scaling—no fair zero-shot improvement claim.\n'
        'C: all predeclared period/context/condition cells, 32 paired bundles; descriptive mean gaps. Non-tiling periods retain full harmonic Gram cross terms. D: pointwise descriptive 95% seed-bootstrap intervals.\n'
        f'The clean local retained-feature comparison did not fully replicate: TimesFM off-target probe error {tf_gate:.3f} exceeded the frozen 0.10 gate. Recoverability alone is not causal feature-use proof.')
    fig.text(.015,.018,footer,fontsize=7.5,va='bottom');fig.tight_layout(rect=[0,.13,1,.95],w_pad=2,h_pad=3);fig.savefig(out/'spectral_composition_summary.png',dpi=240);fig.savefig(out/'spectral_composition_summary.pdf');plt.close(fig)
    source_data=dict(source_files=source_files,source_results=data,panel_a=dict(conditions=conditions,means=a_values,interaction=interaction,units='common mixed energy for pure and mixed conditions'),panel_b=dict(fresh_head_readout=b_values,oracle=data['readout']['models']['bolt']['datasets']['fresh']['input_oracle_scaled_mse'],caveat=data['readout']['scope']),panel_c=dict(rows=grid_rows,units=data['boundary'].get('units'),n_seeds=data['boundary']['n_seeds'],display='descriptive differences of stored means;not paired confidence intervals'),failed_tangent_gate=dict(timesfm_offtarget_relative_RMS=tf_gate,threshold=.10,replication_passes=data['tangent']['replication_passes']),branch_audit=dict(scope=data['branches']['scope'],primary_skip_signed_shares={model:data['branches']['models'][model]['deltas']['0.01']['branches']['skip']['signed_projection_share'] for model in ['bolt','timesfm']}))
    (out/'source_data.json').write_text(json.dumps(source_data,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-data',type=Path);p.add_argument('--replication',type=Path);p.add_argument('--readout',type=Path);p.add_argument('--boundary',type=Path);p.add_argument('--tangent',type=Path);p.add_argument('--branches',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.source_data is None and any(x is None for x in [a.replication,a.readout,a.boundary,a.tangent,a.branches]):p.error('Supply --source-data or all five source result paths')
    build(a.replication,a.readout,a.boundary,a.tangent,a.branches,a.out,a.source_data)
