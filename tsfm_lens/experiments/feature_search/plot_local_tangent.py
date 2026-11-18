"""Plot frozen local-head tangent replication effects directly from result JSON.

Energy effects are ratios of aggregate seed energies,with joint seed-bootstrap
intervals. Gate failures remain visible rather than being filtered from plots.
"""
import argparse
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source,out):
    """Write a standalone PNG/PDF showing model effects and the direct contrast."""
    result=json.loads((source/'results.json').read_text(encoding='utf-8'));protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=True);family=result['replication_family'];names=['bolt','timesfm'];labels=['Bolt','TimesFM'];effects=[family[name+'_psi'] for name in names];contrast=family['timesfm_minus_bolt_psi']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,4.5),gridspec_kw={'width_ratios':[1.35,1]})
    for i,(name,label,effect) in enumerate(zip(names,labels,effects)):
        passed=result['replication_passes'][name];color='#176B58' if passed else '#A54833'
        axes[0].errorbar(i,effect['value'],yerr=[[effect['value']-effect['lo']],[effect['hi']-effect['value']]],fmt='o' if passed else 'X',color=color,capsize=5,markersize=8)
    axes[0].set_xticks(range(2),labels);axes[0].set_xlim(-.6,1.6);axes[0].axhline(.03,color='#666666',ls='--',lw=1,label='Practical threshold 0.03');axes[0].set_title('Native minus frozen-probe mixing');axes[0].set_ylabel('Off-frequency derivative energy / true target strength²');axes[0].legend(loc='upper left',frameon=False,fontsize=8)
    passed=result['replication_passes']['crossmodel'];color='#315F9B' if passed else '#A54833'
    axes[1].errorbar(0,contrast['value'],yerr=[[contrast['value']-contrast['lo']],[contrast['hi']-contrast['value']]],fmt='o' if passed else 'X',color=color,capsize=5,markersize=8);axes[1].set_xticks([0],['TimesFM − Bolt']);axes[1].set_xlim(-.6,.6);axes[1].axhline(.10,color='#666666',ls='--',lw=1,label='Practical threshold 0.10');axes[1].set_title('Direct paired model contrast');axes[1].legend(loc='upper left',frameon=False,fontsize=8)
    for ax in axes:ax.axhline(0,color='#222222',lw=.8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    for ax,threshold in zip(axes,[.03,.10]):
        ax.get_legend().remove()
        ax.text(.98,threshold+.012,f'Practical threshold {threshold:.2f}',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8,color='#666666')
    gates=[]
    for name,label in zip(names,labels):
        target=result['models'][name]['targets']['psi'];gates.append(label+': '+('route/retention pass' if target['interpretation_allowed'] else '; '.join(target['gate_failure_reasons'])))
    fig.suptitle(f'Period-16 local native-head tangent replication · {protocol["n_seeds"]} fresh seed bundles',fontsize=12,y=.98)
    footer='95% joint seed-bootstrap intervals; frozen old-trained probe; δ=0.01.\n'+textwrap.fill(' | '.join(gates),width=125)+'\nCircle: full replication gate passed. X: prespecified gate failed; contrast interpretation requires both models.'
    fig.text(.02,.02,footer,fontsize=8,va='bottom');fig.tight_layout(rect=[0,.22,1,.94]);fig.savefig(out/'local_tangent_replication.png',dpi=220);fig.savefig(out/'local_tangent_replication.pdf');plt.close(fig)
    (out/'plot_data.json').write_text(json.dumps(dict(source=str(source),protocol=protocol,effects=family,gates=result['replication_passes']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();plot(args.source,args.out)
