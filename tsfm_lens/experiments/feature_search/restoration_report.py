"""Portable restoration report from audited saved results, with explicit pending states.

No models, scientific fits or new tests run here. Conditional discovery selection
never overrides resource/numerical failure. Incomplete fresh replication cannot
contribute confirmatory numbers. The original execution failure stays visible.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discovery_status(result):
    """Read the effective decision, preserving the resource/numerical override."""
    if not result['numerical_checks']['passed'] or result.get('resource_limit_exceeded',False):
        return dict(valid=False,effective_cut=None)
    return dict(valid=True,effective_cut=result.get('effective_earliest_cut'))


def replication_status(result=None):
    """Only a complete, valid64-seed family may expose confirmatory estimates."""
    if result is None:return dict(status='pending',complete=False,claims={},reason='Fresh outcomes not available')
    status=result.get('status','incomplete')
    ids=result.get('per_seed',{}).get('seed_ids',[])
    valid_ids=ids==list(range(3700000,3700064)) and all(type(x) is int for x in ids)
    if status!='complete' or not valid_ids or result.get('family_size')!=2 or result.get('resource_limit_exceeded',False):
        return dict(status=status if status!='complete' else 'incomplete',complete=False,claims={},reason=result.get('reason',result.get('scope','Full planned validity/completion is not established')))
    if set(result.get('claims',{}))!={'native_gain','random_advantage'}:
        return dict(status='incomplete',complete=False,claims={},reason='The frozen two-claim family is incomplete')
    return dict(status='complete',complete=True,qualified=bool(result['qualified']),claims=result['claims'],
                gates=result['gates'],mean_losses=result['mean_losses'],reason=result['scope'])


def collect(root,replication=None):
    """Package exact producer/audit records; aggregate plot numbers are already audited."""
    runs=root/'tsfm_lens/runs'
    corrected=runs/'feature_search_stage49_restoration3600000_memorysafe'
    failed=runs/'feature_search_stage49_restoration3600000'
    audit=runs/'feature_search_stage49_restoration3600000_memorysafe_independent/audit.json'
    result=read(corrected/'results.json');old=read(failed/'results.json')
    paths=[corrected/'results.json',corrected/'protocol.json',failed/'results.json',audit]
    data=dict(discovery=result,discovery_audit=read(audit),protocol=read(corrected/'protocol.json'),
        discovery_state=discovery_status(result),original_failure=dict(resource_limit_exceeded=old['resource_limit_exceeded'],
            peak_resident_bytes=old['peak_resident_bytes'],effective_cut=None,conditional_cut=old['pilot']['selection']['earliest_qualifying_cut']),
        replication=replication_status(None if replication is None else read(replication)),
        source_files={str(path.relative_to(root)):sha(path) for path in paths},
        scope='Discovery 16 plus exact execution correction; this is not independent replication. Fresh 64 is separate and pending until valid final outcomes.')
    if replication is not None:
        data['source_files'][str(replication)]=sha(replication)
        if data['replication']['complete']:
            data['fresh_result']=read(replication)
            audit_path=replication.parent.parent/(replication.parent.name+'_independent_audit')/'audit.json'
            if audit_path.is_file():
                audit=read(audit_path)
                if audit.get('status')!='independent_audit_passed' or audit['bindings']['results.json']!=sha(replication):raise ValueError('Independent audit does not bind the fresh result')
                data['fresh_independent_audit']=audit
                data['source_files'][str(audit_path)]=sha(audit_path)
            for name in ['protocol.json','controller_validity.json']:
                path=replication.parent/name
                data['source_files'][str(path)]=sha(path)
            data['fresh_protocol']=read(replication.parent/'protocol.json')
            controller=read(replication.parent/'controller_validity.json')
            data['fresh_controller_summary']={key:controller[key] for key in ['all_four_valid','completed_shards','controller_peak_bytes','maximum_simultaneous_RSS','models_exited_before_analysis']}
            data['fresh_controller_summary']['shards']=[{key:row[key] for key in ['shard_index','seed_ids','valid','primary15_random_geometry_pass','peak_resident_bytes','wall_seconds','waveform_sha256']} for row in controller['validity']]
    return data


def discovery_curve(audit):
    """An observed anchor remains visible when its random comparison is unscorable."""
    cuts=[0,5,10,15,20]
    return dict(cuts=cuts,anchor=[audit['cuts'][str(c)]['anchor_loss_mean'] for c in cuts],
        random=[None if audit['cuts'][str(c)]['control_failed_seed_indices'] else audit['cuts'][str(c)]['mean_random_loss'] for c in cuts])


def plot(data,out):
    """Render source-derived curves and leave missing/incomplete replication blank."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    audit=data['discovery_audit'];result=data['discovery'];fresh=data['replication']
    cuts=[0,5,10,15,20]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13,9.2))
    ax=axes[0,0]
    curve=discovery_curve(audit)
    anchor=curve['anchor']
    random=[float('nan') if value is None else value for value in curve['random']]
    ax.plot(cuts,anchor,'o-',color='#28659c',label='Anchored edit')
    ax.plot(cuts,random,'s--',color='#8b8b8b',label='Two matched random edits')
    ax.axhline(audit['native_mean'],color='#b65e2e',label='Native mixture')
    ax.axhline(audit['actual_sum_mean'],color='#287a58',ls=':',label='Actual component sum reference')
    ax.scatter([10],[anchor[2]],marker='x',s=100,color='#9a6525',zorder=5)
    ax.annotate('Random NA',(10,anchor[2]),xytext=(5,9),textcoords='offset points',fontsize=8,color='#9a6525')
    ax.scatter([20],[anchor[-1]],marker='o',s=120,facecolors='none',edgecolors='#333333',zorder=5)
    ax.set_yscale('log');ax.set_ylim(.002,4.5);ax.set_xticks(cuts,['Tokenizer','After 5','After 10','After 15','After 20'])
    ax.set_ylabel('MSE / common theoretical mixed energy');ax.set_title('A  Discovery curve · only four candidate cuts',loc='left',fontsize=11)
    ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.15)
    ax=axes[0,1]
    values=[audit['native_mean'],audit['cuts']['15']['anchor_loss_mean'],audit['cuts']['15']['mean_random_loss'],audit['actual_sum_mean']]
    labels=['Native','Anchored','Random mean','Component sum']
    for j,value in enumerate(values):ax.bar(j-.15,value,.28,color=['#b65e2e','#28659c','#888888','#287a58'][j])
    if fresh['complete']:
        for j,key in enumerate(['native','anchored','random_mean','component_reference']):
            ax.bar(j+.15,fresh['mean_losses'][key],.28,color=['#b65e2e','#28659c','#888888','#287a58'][j],alpha=.45,hatch='//')
        ax.legend(handles=[Patch(facecolor='#777777',label='Discovery 16'),Patch(facecolor='#bbbbbb',hatch='//',label='Fresh 64')],fontsize=8,loc='upper right')
    else:ax.text(.5,.82,'Fresh 64: '+fresh['status']+'\nNo confirmatory numbers',transform=ax.transAxes,ha='center',bbox=dict(facecolor='#fff3dc',edgecolor='none',pad=8))
    ax.set_xticks(range(4),labels,rotation=12);ax.set_yscale('log');ax.set_ylim(.002,4.5)
    ax.set_title('B  Selected cut: after 15 of 20 blocks (index 14)',loc='left',fontsize=11)
    ax.set_ylabel('Source-reported mean normalized MSE');ax.grid(axis='y',alpha=.15)
    ax=axes[1,0]
    endpoints=['native_gain','random_advantage']
    for i,key in enumerate(endpoints):
        claim=result['pilot']['selection']['claims']['15/'+key]
        q=claim['value'];ax.errorbar(q,3-i,xerr=[[q-claim['lo']],[claim['hi']-q]],fmt='o',color='#28659c',capsize=4)
        ax.text(.98,3-i+.18,f"Discovery Holm8={claim['p_holm']:.3g}",ha='right',fontsize=8)
        if fresh['complete']:
            row=fresh['claims'][key];q=row['value']
            ax.errorbar(q,1-i,xerr=[[q-row['lo']],[row['hi']-q]],fmt='s',color='#287a58',capsize=4)
            ax.text(.98,1-i+.18,f"Fresh Holm2={row['p_holm']:.3g}",ha='right',fontsize=8)
    ax.axvline(.5,color='#777',ls='--',lw=1);ax.set_xlim(.45,1.02);ax.set_ylim(-.45,3.65)
    ax.set_yticks([3,2,1,0],['Discovery: native gain','Discovery: random advantage','Fresh: native gain','Fresh: random advantage'])
    ax.set_xlabel('q: positive paired effect among non-tied seeds (CP95%)')
    ax.set_title('C  Cohorts and correction families stay separate',loc='left',fontsize=11);ax.grid(axis='x',alpha=.15)
    ax=axes[1,1];ax.axis('off')
    gate=result['pilot']['selection']['gates']['15']
    fresh_values=(
        [f"{fresh['claims']['native_gain']['mean']:.4f}",f"{fresh['claims']['random_advantage']['mean']:.4f}",
         f"{fresh['gates']['mean_suffix_energy']:.4f}",f"{100*fresh['gates']['recovery_unclipped']:.2f}%"]
        if fresh['complete'] else [fresh['status']]*4)
    rows=[['Native gain ≥0.10',f"{result['pilot']['selection']['claims']['15/native_gain']['mean']:.4f}",fresh_values[0]],
          ['Random advantage ≥0.05',f"{result['pilot']['selection']['claims']['15/random_advantage']['mean']:.4f}",fresh_values[1]],
          ['Suffix energy ≤0.05',f"{gate['suffix_energy']:.4f}",fresh_values[2]],['Gap recovery ≥75%',f"{100*gate['recovery']:.2f}%",fresh_values[3]],
          ['Cut 10 controls','Unscorable:3/16','NA; not searched'],['Cut 20','Late; ineligible','Late; random NA'],
          ['Effective cut',str(discovery_status(result)['effective_cut']),'15 only' if fresh['complete'] and fresh['qualified'] else fresh['status']]]
    table=ax.table(cellText=rows,colLabels=['Frozen gate / role','Discovery 16','Fresh 64'],colWidths=[.48,.26,.26],loc='center',cellLoc='left');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.55)
    ax.set_title('D  Point gates are not population mean guarantees',loc='left',fontsize=11)
    for ax in axes.flat:
        if ax.axison:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Anchored composition restoration · earliest qualifying TESTED cut',fontsize=14,y=.99)
    fig.text(.02,.012,'Cut 15 = AFTER 15 OF 20 TRANSFORMER BLOCKS (zero-based block index 14). Cut 10 is unscorable under its geometry gate, not a negative causal effect.\nPrivileged pure-component inputs and off-manifold state edits; this is not a deployable repair, isolated feature, or harmful-layer origin. The component sum is a reference, not an upper/lower bound.\nDiscovery 16 was rerun solely for resource/equivalence validation: all 29 arrays bitwise equal. Exact q intervals are pointwise; Holm families stay separate. Incomplete fresh outcomes show no confirmatory numbers.',fontsize=8,va='bottom')
    fig.tight_layout(rect=[0,.1,1,.95],h_pad=2.4,w_pad=3)
    fig.savefig(out/'restoration_summary.png',dpi=220);fig.savefig(out/'restoration_summary.pdf');plt.close(fig)


def render(data,out):
    """Create offline HTML preserving failed execution history and pending status."""
    escape=html.escape;fresh=data['replication'];state=discovery_status(data['discovery']);audit=data['discovery_audit']
    status=fresh['status'] if not fresh['complete'] else ('Passed on 64 new seeds' if fresh['qualified'] else 'Did not pass on 64 new seeds')
    lead=('Discovery selects the state after 15 of 20 transformer blocks (index 14) as the earliest <strong>tested</strong> cut clearing all discovery gates.' if state['valid'] and state['effective_cut']==15 else 'Discovery validation failed or no effective cut is available; no suffix qualifies.')
    fresh_text=''
    if fresh['complete']:
        losses=fresh['mean_losses'];gain=fresh['claims']['native_gain'];random=fresh['claims']['random_advantage']
        fresh_text='<h2>Fresh selected-cut result</h2><p>Native normalized MSE '+f"{losses['native']:.6f}"+' becomes '+f"{losses['anchored']:.6f}"+' under the anchored edit; matched random mean is '+f"{losses['random_mean']:.6f}"+' and actual component-sum reference is '+f"{losses['component_reference']:.6f}"+'. Native gain is positive on '+str(gain['positive'])+'/64 seeds; random advantage on '+str(random['positive'])+'/64. The registered two-claim family '+('passes' if fresh['qualified'] else 'does not pass')+' its full gates. This supports only the tested off-manifold composition-restoration hypothesis.</p>'
    if data.get('fresh_independent_audit'):
        fresh_text+='<p>Independent model-free audit passed. Its geometry audit verifies saved diagnostics, not regenerated hidden-state Gram matrices.</p>'
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Anchored composition restoration</title><style>body{font:17px/1.55 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#1b293b;background:#fafbfd}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{font-size:.9rem;color:#475569}.notice{background:#fff3dc;padding:15px;border-left:4px solid #b76b16}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.75rem}a{color:#0755a2}</style>
<h1>Anchored composition restoration</h1>
<p>'''+lead+''' Earlier cuts remain in the record; cut 10 is unscorable because random-control geometry fails on three seeds. After 20 blocks is a late control and never eligible.</p>
<p class="notice">Fresh 64 replication: <strong>'''+escape(status)+'''</strong>. '''+('No confirmatory estimates are displayed until the full planned cohort and validity gates pass.' if not fresh['complete'] else 'The fresh result has its own frozen two-claim family; it does not pool with discovery.')+' '+escape(fresh['reason'])+'''</p>
<p><a href="../index.html">Stage49 research overview</a> · <a href="source_data.json">Portable source data</a> · <a href="restoration_summary.pdf">Figure PDF</a></p>
<figure><img src="restoration_summary.png" alt="Discovery restoration curve, selected-cut task losses, separate conditional-sign effects and gate table"><figcaption>After 15 of 20 blocks means zero-based index 14. Privileged pure-component inputs and off-manifold state addition test a declared composition hypothesis; this does not identify a harmful layer or unique causal feature. The actual native component sum is a reference, not a bound, and recovery is not clipped.</figcaption></figure>
'''+fresh_text+'''<h2>Execution history remains visible</h2><p>The original 16-seed run exceeded the 1.5 GiB resident limit and could not advance. The corrected execution released model weights before statistical imports: peak '''+f"{data['discovery']['peak_resident_bytes']/2**30:.5f}"+''' GiB. All 29 saved arrays are bitwise identical and every scientific result is exact. This is an execution correction, not independent replication. The effective discovery cut is '''+escape(str(state['effective_cut']))+'''.</p>
<p>Exact sign tests concern q=P(positive paired contrast | non-ties), not population mean superiority. Empirical mean margins, suffix energy and reference-gap recovery are separate point gates. A nonlinear invertible representation can change addition compatibility while leaving the native model correct, so the curve cannot uniquely locate the origin of harm.</p>
<details><summary>Producer, audit, resource and failure provenance</summary><pre>'''+escape(json.dumps(data,indent=2))+'''</pre></details>
<p>Rebuild without checkpoints or run stores:</p><pre>python tsfm_lens/experiments/feature_search/restoration_report.py --source-data examples/feature_search_stage49/restoration/source_data.json --out examples/feature_search_stage49/restoration_rebuild</pre></html>'''
    if not (out/'../index.html').is_file():document=document.replace('<a href="../index.html">Stage49 research overview</a>','Stage49 overview unavailable at this output location')
    (out/'index.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path);parser.add_argument('--source-data',type=Path)
    parser.add_argument('--replication',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if bool(args.repo_root)==bool(args.source_data):parser.error('Choose repo-root or source-data')
    data=read(args.source_data) if args.source_data else collect(args.repo_root,args.replication)
    args.out.mkdir(parents=True,exist_ok=False);plot(data,args.out);render(data,args.out)
    (args.out/'source_data.json').write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    (args.out/'render_provenance.json').write_text(json.dumps(dict(renderer_sha256=sha(__file__),source_data_sha256=sha(args.out/'source_data.json')),indent=2),encoding='utf-8')


if __name__=='__main__':main()
