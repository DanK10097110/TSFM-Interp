"""Portable two-panel fresh native-head replication figure from saved JSON only.

The ratio intervals are pointwise paired-seed bootstrap intervals. Exact sign
tests concern conditional majority,not mean superiority. Branch labels remain
in the portable data;energies are never stacked because cross terms matter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'chronos2':'Chronos-2','bolt':'Chronos-Bolt','timesfm':'TimesFM'}
COLORS={'chronos2':'#287A58','bolt':'#C46A28','timesfm':'#28659C'}


def plot(out,results=None,protocol=None,portable=None):
    """Render stored values and retain all source results for cache-free rebuilding."""
    if portable is None:
        data=dict(results=json.loads(results.read_text(encoding='utf-8')),protocol=json.loads(protocol.read_text(encoding='utf-8')),
            source_files={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in [('results',results),('protocol',protocol)]})
    else:data=json.loads(portable.read_text(encoding='utf-8'))
    result=data['results'];frozen=data['protocol'];delta=str(frozen['primary_delta']);models=['chronos2','bolt','timesfm'];claims=['timesfm_minus_chronos2','bolt_minus_chronos2']
    if set(result['comparisons'])!=set(claims) or frozen['seed_base']!=2225000 or frozen['n_seeds']!=128:raise ValueError('Unexpected fixed fresh replication family or cohort')
    out.mkdir(parents=True,exist_ok=True);files=['native_head_replication.png','native_head_replication.pdf','native_head_replication_source_data.json']
    if any((out/name).exists() for name in files):raise ValueError('Figure outputs already exist;choose a new destination')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,(left,right)=plt.subplots(1,2,figsize=(12.6,5.1),gridspec_kw={'width_ratios':[1,1.15]})
    for i,model in enumerate(models):
        effect=result['models'][model]['deltas'][delta]['absolute_attribution']['total_energy_ratio'];value=effect['value'];left.errorbar(i,value,yerr=[[value-effect['lo']],[effect['hi']-value]],fmt='o',color=COLORS[model],capsize=5,ms=7);left.annotate(f'{value:.4g}',(i,value),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
    left.set_yscale('log');left.set_ylim(5e-4,1.4);left.set_xlim(-.45,2.45);left.set_xticks(range(3),[NAMES[m] for m in models]);left.set_ylabel('Absolute off-frequency energy / analytic target strength²');left.set_title('A  Native response to period-16 phase changes',loc='left',fontsize=11);left.grid(axis='y',alpha=.18)
    for y,key in zip([1,0],claims):
        sign=result['comparisons'][key]['sign_test'];q=sign['q'];lo,hi=sign['q_cp95'];right.errorbar(q,y,xerr=[[q-lo],[hi-q]],fmt='o',color=COLORS[key.split('_minus_')[0]],capsize=5,ms=7);right.text(.98,y+.19,f'q = {q:.4f}; {sign["positive"]}+ / {sign["negative"]}− / {sign["ties"]} ties',ha='right',fontsize=9)
    right.axvline(.5,color='#555555',ls='--',lw=1);right.set_xlim(.45,1.015);right.set_ylim(-.5,1.5);right.set_xticks([.5,.6,.7,.8,.9,1.]);right.set_yticks([1,0],['TimesFM − Chronos-2','Chronos-Bolt − Chronos-2']);right.set_xlabel('q: positive paired energy difference, conditional on non-ties');right.set_title('B  Exact conditional-majority intervals',loc='left',fontsize=11);right.grid(axis='x',alpha=.18)
    for ax in [left,right]:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fresh native-head replication · 128 matched seed bundles',fontsize=14,y=.98)
    ptext='; '.join(f'{"TF" if key.startswith("timesfm") else "Bolt"}−C2: raw p={result["comparisons"][key]["sign_test"]["p"]:.3g}, Holm2 p={result["comparisons"][key]["p_holm"]:.3g}' for key in claims)
    caption=(
        'Known input phase counterfactual: ψ16 ± δ, primary δ=0.01; period-32 + period-16 mixture, context 128, horizon 32. No probes or isolated-feature claim.\n'
        'A: pointwise paired-seed bootstrap 95% intervals; ratio = Σ||∂(period-32 coefficients)/∂ψ16||² / Σ(b/s)², with s²=(a²+b²)/2. Ratio intervals are approximate.\n'
        'B: exact Clopper–Pearson 95% intervals and two-sided sign tests. q establishes conditional majority, not mean superiority. '+ptext)
    fig.text(.015,.02,caption,fontsize=7.6,va='bottom');fig.tight_layout(rect=[0,.22,1,.92],w_pad=2);fig.savefig(out/files[0],dpi=240);fig.savefig(out/files[1]);plt.close(fig)
    data['figure_scope']=dict(primary_delta=.01,energy_intervals='pointwise paired-seed bootstrap95%;approximate ratios',q_intervals='exact Clopper-Pearson95%;conditional majority among non-ties',no_stacked_energies=True,no_probe_or_isolated_feature_claim=True)
    (out/files[2]).write_text(json.dumps(data,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--source-data',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.source_data is None and (args.results is None or args.protocol is None):p.error('Need --source-data or both --results and --protocol')
    plot(args.out,args.results,args.protocol,args.source_data)
