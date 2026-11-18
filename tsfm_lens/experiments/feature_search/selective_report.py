"""Scientific figures and a portable report separating hidden composition from prediction."""
from pathlib import Path
import argparse,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.feature_search.coherence_study import read,write,DOMAINS

NAMES=dict(zip(DOMAINS,['Electricity','M4 hourly','Pedestrians','Traffic']))
COLORS={'native':'#203c5a','output_free':'#d47723','output_locked':'#aa564e','hybrid':'#24867d','equal':'#7057a4','learned':'#d6487c'}


def mean(rows,field,**filters):
    """Equal source weighting for the chosen slice; missing observations stay absent."""
    values={}
    for r in rows:
        if all(r.get(k)==v for k,v in filters.items()) and r.get(field) is not None:values.setdefault(r.get('source',str(r.get('row'))),[]).append(float(r[field]))
    return float(np.mean([np.mean(v) for v in values.values()])) if values else float('nan')


def table(rows,columns):
    """Readable HTML with exact values preserved in accompanying JSON."""
    def fmt(x):return f'{x:.6g}' if isinstance(x,float) else html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(r.get(c,''))+'</td>' for c in columns)+'</tr>' for r in rows)+'</tbody></table></div>'


def closure_curve(records,plans,rows,model,case,cut,weight,field):
    """Keep the same sources across requested depths, respecting past-only early stops."""
    lookup={(r['row'],r['depth'],r['cut'],r['weight']):r for r in records if r['model']==model}
    ids=[i for i,r in enumerate(rows) if r['split']=='test' and r['case']==case and len(plans[i]['levels'])>1]
    result=[]
    for depth in [1,2,3]:
        values=[lookup[i,min(depth,len(plans[i]['levels'])-1),cut,weight][field] for i in ids]
        result.append(float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else np.nan)
    return result


def build(run,out):
    """Render a fixed set of mechanistic questions, including nulls and failed gates."""
    out.mkdir(parents=True,exist_ok=True);analysis=read(run/'analysis.json');protocol=read(run/'protocol.json');rows=read(run/'rows.json');plans=read(run/'plans.json');geometry=read(run/'marginal_geometry.json');responses=read(run/'injection_responses.json');dominance=read(run/'dominance_scores.json');tree=read(run/'tree_stability.json')
    transfers={model:read(run/f'{model}_transfer.json') for model in ['bolt','timesfm']};cuts={model:read(run/f'{model}_worker.json')['cuts'] for model in ['bolt','timesfm']}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(2,3,figsize=(14,7.6),layout='constrained')
    cases=['off_grid_far','off_grid_near','balanced_offgrid'];labels=['Off-grid / strong far distractor','Off-grid / strong near distractor','Off-grid / balanced components']
    for i,model in enumerate(['bolt','timesfm']):
        cells=[r for r in analysis['cases'] if r['model']==model and r['domain']=='synthetic']
        for j,(case,title) in enumerate(zip(cases,labels)):
            ax=axes[i,j]
            def value(config):return next(r['loss'] for r in cells if r['case']==case and r['config']==config)
            native=value(['native',0,'none','none',0.]);ax.axhline(native,color=COLORS['native'],label='native')
            for method in ['output_free','output_locked','hybrid']:
                vals=[value([method,d,'none','none',1.]) for d in [1,2,3]];ax.plot([1,2,3],vals,'o-',color=COLORS[method],label=method)
            for weight in ['equal','learned']:
                vals=[value(['fusion',d,cuts[model][0],weight,1.]) for d in [1,2,3]];ax.plot([1,2,3],vals,'o--',color=COLORS[weight],label='early fusion / '+weight)
            ax.set(title=model+' · '+title,yscale='log',xticks=[1,2,3],xlabel='Requested extraction depth',ylabel='Future MSE / past variance');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);fig.savefig(out/'depth_forecasts.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
    closure_rows=[]
    for i,model in enumerate(['bolt','timesfm']):
        for j,field in enumerate(['closure_nmse','marginal_closure_nmse']):
            ax=axes[i,j]
            for weight in ['equal','confidence','learned']:
                for cut,style in [(cuts[model][0],'-'),(cuts[model][-1],'--')]:
                    values=closure_curve(geometry,plans,rows,model,'balanced_offgrid',cut,weight,field)
                    ax.plot([1,2,3],values,'o'+style,label=weight+' / block'+cut)
                    for d,v in zip([1,2,3],values):closure_rows.append(dict(model=model,case='balanced_offgrid',cut=cut,weight=weight,depth=d,metric=field,value=v))
            ax.set(title=model+' · '+('Whole-state error' if j==0 else 'Error relative to remaining contribution'),xticks=[1,2,3],xlabel='Requested depth, with early stopping',ylabel='Hidden reconstruction NMSE',yscale='log');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);axes[1,0].legend(fontsize=8,ncol=2);fig.savefig(out/'hidden_closure.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for j,model in enumerate(['bolt','timesfm']):
        for i,metric in enumerate(['sensitivity','curvature_ratio']):
            ax=axes[i,j]
            for cut in cuts[model]:
                for policy,style in [('native','-'),('locked','--')]:
                    vals=[next(r[metric] for r in analysis['stability'] if r['model']==model and r['domain']=='synthetic' and r['cut']==cut and r['policy']==policy and r['direction']=='sinusoid' and r['epsilon']==eps) for eps in [.001,.01,.1]]
                    ax.plot([.001,.01,.1],vals,'o'+style,label='block'+cut+' / '+policy)
            ax.set(title=model,xscale='log',yscale='log',xlabel='Perturbation RMS / history SD',ylabel=metric.replace('_',' '));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);axes[0,1].legend(fontsize=8,ncol=2);fig.savefig(out/'local_stability.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,model in zip(axes,['bolt','timesfm']):
        for k,policy in enumerate(['native','output_selected','fusion_selected','geometry_selected']):
            vals=[mean(responses,'response_nmse',model=model,domain=d,policy=policy) for d in DOMAINS]
            ax.bar(np.arange(4)+(k-1.5)*.19,vals,.19,label=policy)
        ax.set(title=model,xticks=range(4),yscale='log',ylabel='Known-addition response NMSE');ax.set_xticklabels(list(NAMES.values()),rotation=15);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(out/'real_response.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    transfer_summary=[]
    for ax,model in zip(axes,['bolt','timesfm']):
        for distance,style in [('near','-'),('far','--')]:
            for metric in ['transfer_response_nmse','random_response_nmse']:
                vals=[]
                for cut in cuts[model]:
                    selected=[r for r in transfers[model] if r['cut']==cut and r['epsilon']==.1 and r['distance']==distance and r['oracle_valid'] and r['random_valid']]
                    vals.append(float(np.mean([r[metric] for r in selected])) if selected else np.nan)
                ax.plot(range(len(cuts[model])),vals,'o'+style,label=distance+' / '+('donor' if metric.startswith('transfer') else 'random'))
        ax.axhline(1,color='#333333',ls=':',label='no edit');ax.set(title=model,xticks=range(len(cuts[model])),xticklabels=['block'+c for c in cuts[model]],ylabel='Native-response transport NMSE',yscale='log');ax.grid(alpha=.2)
        for domain in DOMAINS:
            selected=[r for r in transfers[model] if r['domain']==domain]
            transfer_summary.append(dict(model=model,domain=NAMES[domain],receivers=len({r['source'] for r in selected}),rows=len(selected),oracle_pass=sum(r['oracle_valid'] for r in selected),semantic_pass=sum(r['semantic_valid'] for r in selected),random_pass=sum(r['random_valid'] for r in selected)))
    axes[0].legend(fontsize=8,ncol=2);fig.savefig(out/'transfer.png');plt.close(fig)
    frequency_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for j,model in enumerate(['bolt','timesfm']):
        datasets=[('near-daily',transfers[model]),('19.2 hours',read(run/f'{model}_transfer_period19p2.json'))]
        for k,(label,observations) in enumerate(datasets):
            for domain in DOMAINS:
                selected=[r for r in observations if r['domain']==domain and r['cut']==cuts[model][0] and r['epsilon']==.1 and r['distance']=='near']
                frequency_rows.append(dict(model=model,domain=NAMES[domain],period=label,sources=len(selected),native_signal_nmse=float(np.mean([r['native_signal_nmse'] for r in selected])),transport_nmse=float(np.mean([r['transfer_response_nmse'] for r in selected])),semantic_sources=sum(r['semantic_valid'] for r in selected)))
            for i,metric in enumerate(['native_signal_nmse','transfer_response_nmse']):
                values=[float(np.mean([r[metric] for r in observations if r['domain']==domain and r['cut']==cuts[model][0] and r['epsilon']==.1 and r['distance']=='near'])) for domain in DOMAINS]
                axes[i,j].bar(np.arange(4)+(k-.5)*.32,values,.32,label=label)
        for i in range(2):
            axes[i,j].set(title=model,xticks=range(4),ylabel='Known-signal response NMSE' if i==0 else 'Native-response transport NMSE',yscale='log');axes[i,j].set_xticklabels(list(NAMES.values()),rotation=15);axes[i,j].grid(axis='y',alpha=.2)
        axes[0,j].axhline(.5,color='#333333',ls=':',label='semantic gate')
    axes[0,0].legend(fontsize=8);fig.savefig(out/'frequency_transfer.png');plt.close(fig)
    paired_transfer=[]
    for r in read(run/'transfer_summary.json'):
        if r['cut']==cuts[r['model']][0] and r['epsilon']==.1 and r['scope']=='oracle_valid':
            near_far=r.get('near_over_far',{});random=r.get('near_random_advantage',{})
            paired_transfer.append(dict(model=r['model'],domain=NAMES[r['domain']],period=r['period'],n=r['eligible_sources'],near=r.get('near_nmse'),far=r.get('far_nmse'),near_over_far=near_far.get('mean'),lower=near_far.get('ci',[None,None])[0],upper=near_far.get('ci',[None,None])[1],near_random_advantage=random.get('mean')))
    policy_rows=[]
    for r in analysis['policies']:
        gain=r['gain_vs_native'];policy_rows.append(dict(model=r['model'],domain=NAMES.get(r['domain'],r['domain']),policy=r['policy'],choice=str(r['selected']),n=gain['n'],gain=gain['mean'],lower=gain['ci'][0],upper=gain['ci'][1],random_advantage=r.get('random_advantage',{}).get('mean','n/a'),random_valid=r.get('random_valid_fraction','n/a')))
    split_rows=[]
    for domain in ['synthetic']+list(DOMAINS):
        for case in sorted({r['case'] for r in rows if r['domain']==domain}):
            ids=[i for i,r in enumerate(rows) if r['domain']==domain and r['case']==case and r['split']=='test'];depths=[len(plans[i]['levels'])-1 for i in ids]
            split_rows.append(dict(domain=NAMES.get(domain,domain),case=case,sources=len(ids),depth0=depths.count(0),depth1=depths.count(1),depth2=depths.count(2),depth3=depths.count(3)))
    tail_rows=[r for r in read(run/'local_tail_summary.json') if r['cut']==cuts[r['model']][0] and r['policy']=='native' and r['epsilon']==.001]
    tree_rows=[]
    for domain in ['synthetic']+list(DOMAINS):
        for eps in [.001,.01,.1]:
            subset=[r for r in tree if r['domain']==domain and r['split']=='test' and r['epsilon']==eps]
            tree_rows.append(dict(domain=NAMES.get(domain,domain),radius=eps,depth_change_fraction=float(np.mean([r['depth_changed'] for r in subset])),mean_partition_rms=float(np.mean([r['matched_partition_rms'] for r in subset]))))
    weights=[]
    for model,depths in protocol['learned_weights'].items():
        for depth,cutrows in depths.items():
            for cut,r in cutrows.items():weights.append(dict(model=model,depth=depth,cut=cut,tone_weight=r['weights'][0],residual_weight=r['weights'][1],pilot_fit_rows=r['fit_rows']))
    dominance_rows=[]
    for model in ['bolt','timesfm']:
        for case in ['off_grid_far','balanced_offgrid']:
            for cut in [cuts[model][0],cuts[model][-1]]:
                dominance_rows.append(dict(model=model,case=case,cut=cut,dominant_only_closure=mean(dominance,'closure_nmse',model=model,case=case,cut=cut,split='test'),dominant_only_loss=mean(dominance,'loss',model=model,case=case,cut=cut,split='test')))
    real_rows=[]
    for model in ['bolt','timesfm']:
        for domain in DOMAINS:
            for policy in ['native','output_selected','fusion_selected','geometry_selected']:
                real_rows.append(dict(model=model,domain=NAMES[domain],policy=policy,response_nmse=mean(responses,'response_nmse',model=model,domain=domain,policy=policy),clean_loss=mean(responses,'clean_loss',model=model,domain=domain,policy=policy),injected_loss=mean(responses,'injected_loss',model=model,domain=domain,policy=policy)))
    strong=read(run/'strong_signal_scores.json');strong_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for j,model in enumerate(['bolt','timesfm']):
        choices=[('native',['native',0,'none','none',0.]),('hybrid / 1',['hybrid',1,'none','none',1.]),('hybrid / 3',['hybrid',3,'none','none',1.]),('early equal / 1',['fusion',1,cuts[model][0],'equal',1.]),('early learned / 1',['fusion',1,cuts[model][0],'learned',1.])]
        for domain in DOMAINS:
            for label,config in choices:
                subset=[r for r in strong if r['model']==model and r['domain']==domain and r['config']==config]
                strong_rows.append(dict(model=model,domain=NAMES[domain],method=label,sources=len(subset),response_nmse=mean(subset,'response_nmse'),future_loss=mean(subset,'loss_original_units'),clean_loss=mean(subset,'clean_loss')))
        for i,metric in enumerate(['response_nmse','loss_original_units']):
            values=[mean([r for r in strong if r['model']==model and r['config']==config],metric) for _,config in choices]
            axes[i,j].bar(range(len(choices)),values,color=['#203c5a','#24867d','#4aa69d','#7057a4','#d6487c'])
            axes[i,j].set(title=model,ylabel='Known-signal response NMSE' if i==0 else 'Future MSE / original past variance',yscale='log',xticks=range(len(choices)))
            axes[i,j].set_xticklabels([label for label,_ in choices],rotation=25,ha='right');axes[i,j].grid(axis='y',alpha=.2)
    fig.savefig(out/'strong_signal.png');plt.close(fig)
    narrative=read(run/'interpretation.json') if (run/'interpretation.json').exists() else dict(paragraphs=['Interpretation pending; figures and numerical tables are populated from saved results.'])
    intro=''.join('<p>'+html.escape(s)+'</p>' for s in narrative['paragraphs'])
    body=f'''<header><div class="eyebrow">SECTION 49 · INTERNAL DECOMPOSABILITY</div><h1>Selective patterns, intact residuals, and hidden-state composition</h1><p>32 fresh synthetic seeds · 32 fresh real sources · two frozen TSFMs · no neural training</p><p>224 contexts include repeated conditions within sources. Effective source count is 64, not 224.</p></header>
<section><h2>Findings and limits</h2>{intro}<p><a href="../coherence_followup/index.html">Earlier fixed-band study</a> motivated the off-grid controls; its failure did not test this selective residual tree.</p></section>
<section><h2>How earlier results shaped these tests</h2><p>PM-28's fixed-band failures motivated continuous-frequency extraction. PM-23 and earlier restoration audits motivated shared normalization, affine-baseline correction and oracle closure. SH-25 already showed retained spectral information despite poor native forecasting in another frequency regime, so a poor forecast here is not treated as proof of missing information. Earlier concept-sharing tests also had sensitivity limits and many undecided outcomes; these within-model interventions do not assume that low cross-model transfer proves different semantics.</p><p>The new tests therefore separate hidden reconstruction, the remaining small contribution, suffix behavior, prediction correctness and local stability. No single score establishes general reducibility.</p></section>
<section><h2>What was actually decomposed?</h2><p>Estimate a dominant frequency continuously between FFT bins. Split only if its past-half amplitude/phase stability, spectral prominence and explained variance pass fixed gates. Keep that tone; recurse on the remaining series, at most three times. The residual always remains intact and is always included. No generating components or future labels enter this procedure.</p>{table(split_rows,['domain','case','sources','depth0','depth1','depth2','depth3'])}<p class="caption">Accepted TEST depths. A zero is a deliberate abstention, not an unsuccessful forecast. Requested depth curves respect this same stopping rule and keep source membership fixed.</p></section>
<section><h2>Does selective extraction improve forecasting?</h2><img src="depth_forecasts.png" alt="Forecast errors across adaptive extraction depths and forecast versus activation composition"><p class="caption">TEST synthetic cases; lower is better, logarithmic axes. Output_free sums independently normalized component forecasts. Output_locked uses the original full-context normalizer. Hybrid extrapolates fitted tones classically and asks the model to forecast only the residual. Early fusion replaces the first block's state with a pure composed state. A hybrid gain alone is not evidence that internal representations compose.</p>{table(policy_rows,['model','domain','policy','choice','n','gain','lower','upper','random_advantage','random_valid'])}<p class="caption">FIT-only chosen policies, evaluated on held-out sources. Positive gain means lower future loss than native. Conditional 95% source-bootstrap intervals are exploratory, not multiplicity-corrected confirmation. Synthetic conditions average within seed. Real policy scores average original and injected contexts within source. Exact ties prefer native. Invalid random controls cannot establish directional specificity.</p></section>
<section><h2>Do the hidden representations compose?</h2><img src="hidden_closure.png" alt="Whole-state and non-dominant-contribution hidden composition errors on balanced mixtures"><p class="caption">Balanced off-grid mixtures prevent a large component from making reconstruction look trivially good. Left: composition error relative to the full signal-dependent state. Right: the same error relative to the full state minus the dominant branch state. All curves use pure composition, with no retained native-state blend. They still share the full context’s normalization statistics; this is conditional composition, not wholly independent encoding. Actual early stops are respected. Different layers and models have different coordinate systems; these are within-model reconstruction tests.</p>{table(dominance_rows,['model','case','cut','dominant_only_closure','dominant_only_loss'])}<p>States use the full context's native normalizer and all native token positions. Composition subtracts duplicated zero-context state. The remaining network is the actual native suffix. Native-state blends of 0.5 and 1 were also tested, but retaining half the original state is not itself evidence of decomposability.</p><details><summary>Weights fitted using pilot hidden states only</summary>{table(weights,['model','depth','cut','tone_weight','residual_weight','pilot_fit_rows'])}</details><p class="caption">Two bounded weights, shared across sources at each depth/cut: extracted-tone sum and residual. Fit only on pilot FIT hidden states, with no future labels. These weights were frozen before replication inference. A zero residual weight is reported, not interpreted automatically as “the model ignores noise.”</p></section>
<section><h2>Clear injected pattern, uncertain real residual</h2><img src="real_response.png" alt="Known-injected-signal response accuracy on four real datasets"><p class="caption">Four TEST sources per dataset, each paired with a known amplitude 2, period 19.2 oscillation. The model still has to handle the observed real residual. These are dependent counterfactual contexts, not new independent series. Smaller response error need not imply a better forecast of the original series.</p>{table(real_rows,['model','domain','policy','response_nmse','clean_loss','injected_loss'])}</section>
<section><h2>When the injected pattern is clearly separable</h2><img src="strong_signal.png" alt="Strong injected pattern: known response and actual future error"><p>The fixed amplitude 8 control triggers extraction on all 16 TEST real sources, compared with only 3 of 16 at amplitude 2. Gates and weights are unchanged. The model predicts the uncertain residual in the hybrid; pure early fusion instead asks the native suffix to use recombined states. Each response compares a method with its own original-context counterpart.</p>{table(strong_rows,['model','domain','method','sources','response_nmse','future_loss','clean_loss'])}<p class="caption">Exploratory follow-up on the same sources, not an independent confirmation. Figure averages all 16 sources; table keeps the four-source datasets separate. Future loss uses original-history variance to keep amplitude changes from hiding errors. No best TEST depth was selected. All tested methods, layers, depths, and valid/invalid random-control pairs remain in strong_signal_scores.json.</p></section>
<section><h2>Locally jagged, or smoothly context-dependent?</h2><img src="local_stability.png" alt="Hidden-state sensitivity and curvature across three perturbation radii"><p class="caption">Same fixed sinusoidal direction at three relative radii, averaged within each seed. Solid: native normalization. Dashed: normalization held to the original context. Sensitivity is the centered secant magnitude divided by input radius; curvature is the symmetric second difference divided by first-response magnitude. Stable sensitivity and shrinking curvature are evidence of local smoothness in these directions, not a theorem of global smoothness. A separate seeded noise direction is retained in the data. Model magnitudes are not directly comparable coordinate-wise.</p><details><summary>Small-radius first-block tails, including noise directions</summary>{table(tail_rows,['model','domain','direction','sources','contexts','sensitivity_p95','sensitivity_max','curvature_ratio_max','relative_change_max'])}<p class="caption">These are tails over observed contexts, not additional independent samples. Full layer/radius/normalizer summaries are in local_tail_summary.json.</p></details>{table(tree_rows,['domain','radius','depth_change_fraction','mean_partition_rms'])}<p class="caption">The adaptive decomposition is checked separately. Hard split thresholds can change tree structure even when the native network is smooth. Tone branches are matched before comparison so reordered peaks do not create a false instability.</p></section>
<section><h2>Does the same component effect transfer between backgrounds?</h2><img src="transfer.png" alt="Near and far background activation-effect transfer versus matched random controls"><p class="caption">Radius 0.1, TEST real receivers and nearest/farthest FIT donors within each dataset. Plot is descriptive over oracle-valid, random-valid cases; no-edit native-response error is 1. The oracle injects the receiver's actual perturbed state under its own locked normalizer. Passing that gate is required before interpreting donor transfer. Matching the native response is distinct from correctly forecasting the injected sinusoid.</p>{table(transfer_summary,['model','domain','receivers','rows','oracle_pass','semantic_pass','random_pass'])}<p class="caption">Rows repeat receivers across radii, layers and donor choices. They are not independent samples. Semantic eligibility additionally requires an accurate native known-sinusoid response. Failed eligibility stays visible; results outside that subset describe behavioral transport, not stable concept semantics. There is no cross-model activation-alignment claim.</p></section>
<section><h2>Transfer can preserve an incorrect component response</h2><img src="frequency_transfer.png" alt="Near-daily versus 19.2-hour native response correctness and activation-effect transport"><p class="caption">A fixed frequency-specificity control motivated by Bolt's near-daily transfer result. Same16 real TEST receivers, same nearest FIT donors, radius0.1 and first-block cut. Top: does the native forecast change match the actual injected oscillation? Bottom: does the transferred edit reproduce that native change? Good behavioral transport can coexist with an incorrect native response. The 19.2-hour control was declared before its execution; it is exploratory, not a frequency sweep.</p>{table(frequency_rows,['model','domain','period','sources','native_signal_nmse','transport_nmse','semantic_sources'])}<details><summary>Paired near/far contrasts on the same valid receivers</summary>{table(paired_transfer,['model','domain','period','n','near','far','near_over_far','lower','upper','near_random_advantage'])}<p class="caption">First block, radius 0.1; both donor controls must be oracle-valid and random-valid. Positive differences favor the near donor. Conditional source-bootstrap intervals are exploratory; each dataset has at most four receivers. These behavior-only contrasts do not require accurate native signal extrapolation.</p></details></section>
<section><h2>Validity, provenance and reproducibility</h2><p>Native repeat, self-replacement, anchored-normalizer identity and input reconstruction are checked. Random edits preserve receiver/edit geometry and are flagged when degenerate. Independent replay checks saved hidden geometry, local hidden sensitivity and forecast losses. Raw plus/minus forecast arrays were not retained for the initial local-sensitivity screen; those forecast-only rows are instrumented diagnostics rather than independently replayed evidence.</p><p>Real replication source IDs are fresh relative to earlier studies; pretraining overlap remains unknown. Only four TEST sources per real dataset limits generalization. FIT selection and held-out evaluation remain separate. Full curves, nulls and individual measurements are retained; no best TEST depth is presented as a deployable selection rule.</p><p><a href="analysis.json">Complete analysis</a> · <a href="marginal_geometry.json">Hidden geometry</a> · <a href="injection_responses.json">Real response pairs</a> · <a href="protocol.json">Frozen protocol</a> · <a href="independent_audit.json">Audit</a> · <a href="README.md">Reproduce</a></p></section>'''
    style='body{margin:0;background:#f4f6f8;color:#172d3e;font:16px/1.6 system-ui}header,section{max-width:1160px;margin:24px auto;padding:28px;background:white;border-radius:12px}header{background:#143547;color:white}.eyebrow{font-size:12px;letter-spacing:2px}h1{font-size:38px;line-height:1.2}h2{font-size:25px}img{width:100%;height:auto}.caption{font-size:13px;color:#526373}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #dde4ea;vertical-align:top}th{background:#eaf0f4}a{color:#206b83}details{margin:16px 0}@media(max-width:700px){header,section{padding:16px;margin:8px}h1{font-size:28px}}'
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Selective internal composition · Section49</title><style>'+style+'</style>'+body+'</html>',encoding='utf-8')
    for name in ['analysis.json','protocol.json','independent_audit.json','marginal_geometry.json','injection_responses.json','dominance_scores.json','tree_stability.json','selected_rows.json','interpretation.json','transfer_summary.json','transfer_audit.json','strong_signal_scores.json','strong_signal_audit.json','mutation_audits.json','local_tail_summary.json','resource_summary.json','environment.json','verification.json']:
        if (run/name).exists():shutil.copyfile(run/name,out/name)
    for model in ['bolt','timesfm']:
        for suffix in ['worker.json','transfer.json','transfer_execution.json','dominant_execution.json','transfer_period19p2.json','transfer_period19p2_execution.json','strong_metadata.json','strong_protocol.json','strong_execution.json']:
            shutil.copyfile(run/f'{model}_{suffix}',out/f'{model}_{suffix}')
    write(out/'closure_plot_values.json',closure_rows)
    (out/'README.md').write_text('''# Selective residual trees and internal composition

From the repository root, use the cudaPy environment, PYTHONPATH=tsfm_lens,
CUDA_VISIBLE_DEVICES empty, and OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1.
Model inference is offline using the existing cached checkpoints and real sources.

Rescore existing artifacts without redoing the main model runs:
```
python -m experiments.feature_search.selective_analysis --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_audit --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_diagnostics --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_transfer_analysis --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_strong_audit --out tsfm_lens/runs/selective_replication4942000
python -m experiments.feature_search.selective_report --run tsfm_lens/runs/selective_replication4942000 --out examples/feature_search_stage49/selective_composition
```

To reproduce, use NEW run directories under tsfm_lens/runs. The replication
preparer reads frozen geometry weights from selective_pilot4941000. Drivers
refuse to overwrite completed native runs.
```
python -m experiments.feature_search.selective_study prepare --out NEW_PILOT
python -m experiments.feature_search.selective_study run --out NEW_PILOT
python -m experiments.feature_search.selective_analysis --out NEW_PILOT
python -m experiments.feature_search.selective_followup prepare --out NEW_REPLICATION
python -m experiments.feature_search.selective_parallel --out NEW_REPLICATION
python -m experiments.feature_search.selective_tree_stability --out NEW_REPLICATION
python -m experiments.feature_search.selective_followup dominant --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_followup dominant --out NEW_REPLICATION --model timesfm
python -m experiments.feature_search.selective_transfer --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_transfer --out NEW_REPLICATION --model timesfm
python -m experiments.feature_search.selective_frequency_transfer --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_frequency_transfer --out NEW_REPLICATION --model timesfm
python -m experiments.feature_search.selective_strong_signal worker --out NEW_REPLICATION --model bolt
python -m experiments.feature_search.selective_strong_signal worker --out NEW_REPLICATION --model timesfm
```

All thresholds, splits and intervention families are in protocol.json and
ROADMAP49.35 onward. No actual future enters the native workers, extraction,
hidden-weight calibration or transfer tests. Forecast-policy selection uses FIT
futures only. TEST sources are not used for hyperparameter selection. Held-out
comparison remains exploratory rather than sealed-private confirmation.
''',encoding='utf-8')
    print(out/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.run,a.out)
