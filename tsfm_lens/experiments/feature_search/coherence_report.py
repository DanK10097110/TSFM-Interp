"""Portable figures and evidence tables for staged coherence experiments."""
from pathlib import Path
import argparse,html,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.feature_search.coherence_study import read,write,DOMAINS
from experiments.feature_search.coherence_analysis import source_values,estimate

NAMES=dict(zip(DOMAINS,['Electricity','M4 hourly','Pedestrians','Traffic']))
COLORS={'native':'#245b9b','fft':'#d66b26','shift':'#258568','bridge':'#9160ab','classical':'#444444','drop_high':'#b64343'}


def average(records,metric,**filters):
    """Average sources, not variable numbers of rows per source."""
    v=source_values(records,metric,**filters)
    return float(np.mean(list(v.values()))) if v else np.nan


def table(rows,columns):
    """Escape artifact strings in compact sortable-free HTML tables."""
    def cell(value):
        if isinstance(value,float):return f'{value:.6g}'
        return html.escape(str(value))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(r.get(c,''))+'</td>' for c in columns)+'</tr>' for r in rows)+'</tbody></table></div>'


def build(run,training,pilot,out,normalization=None):
    """Render all declared endpoints plus exploratory mechanisms with limitations."""
    out.mkdir(parents=True,exist_ok=True);analysis=read(run/'analysis.json');records=read(run/'measurements.json');protocol=read(run/'protocol.json');tr=read(training/'results.json')['results']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    pilot_records=read(pilot/'measurements.json')
    pilot_pairs=[(a,b) for a in [3.,5.,9.] for b in [a+1,2*a,3.4*a]]
    fig,axes=plt.subplots(2,1,figsize=(12,5.5),layout='constrained')
    from matplotlib.colors import LogNorm
    for ax,model in zip(axes,['bolt','timesfm']):
        values=np.asarray([[average(pilot_records,'response_nmse',model=model,method='native',domain='synthetic',target_frequency=a,distractor_frequency=b,amplitude=amp,length=128) for a,b in pilot_pairs] for amp in [.5,8.]])
        im=ax.imshow(values,aspect='auto',norm=LogNorm(vmin=.03,vmax=15),cmap='magma')
        ax.set(title=model,xticks=range(9),yticks=[0,1],yticklabels=['Distractor0.5','Distractor8'])
        ax.set_xticklabels([f'{a:g} / {b:g}' for a,b in pilot_pairs])
        for y in range(2):
            for x in range(9):ax.text(x,y,f'{values[y,x]:.2f}',ha='center',va='center',color='white',fontsize=9)
    fig.colorbar(im,ax=axes,label='Native response NMSE');axes[-1].set_xlabel('Target / distractor frequency (cycles per128 hours)')
    fig.savefig(out/'pilot_map.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    conditions=[(5.,6.),(5.,17.),(5.35,6.35),(5.35,17.35)];labels=['Near / on-grid','Far / on-grid','Near / off-grid','Far / off-grid']
    for ax,model in zip(axes,['bolt','timesfm']):
        for method in ['native','fft','shift','bridge','classical']:
            values=[average(records,'response_nmse',model='classical' if method=='classical' else model,method='native' if method=='classical' else method,domain='synthetic',target_frequency=a,distractor_frequency=b,amplitude=8.,length=128) for a,b in conditions]
            ax.plot(range(4),values,'o-',label=method,color=COLORS[method])
        ax.set(title=model,ylabel='Response NMSE (lower is better)',yscale='log',xticks=range(4));ax.set_xticklabels(labels,rotation=20,ha='right');ax.grid(alpha=.2)
    axes[1].legend(ncol=3,fontsize=9);fig.tight_layout();fig.savefig(out/'interference.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,model in enumerate(['bolt','timesfm']):
        for j,metric in enumerate(['response_nmse','mse']):
            ax=axes[i,j]
            for k,method in enumerate(['native','fft','bridge','drop_high']):
                filt=dict(model=model,split='test',length=128,method=method,target_frequency=5.)
                if metric=='mse':filt['sign']=0
                vals=[average(records,metric,domain=d,**filt) for d in DOMAINS]
                ax.bar(np.arange(4)+(k-1.5)*.19,vals,.19,label=method,color=COLORS[method])
            ax.set_yscale('log')
            ax.set(title=model+(' — injected-signal response' if j==0 else ' — actual clean future'),ylabel='Response NMSE' if j==0 else 'MSE / past variance',xticks=range(4));ax.set_xticklabels(list(NAMES.values()),rotation=15);ax.grid(axis='y',alpha=.2)
    axes[0,1].legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(out/'real_tradeoff.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    training_summary=[]
    for ax,architecture in zip(axes,['mlp','transformer']):
        for arm,color in [('withheld','#b64343'),('covered','#258568')]:
            for seed in sorted({r['seed'] for r in tr}):
                vals=[next(r['error']['heldout'] for r in tr if r['seed']==seed and r['architecture']==architecture and r['arm']==arm and r['steps']==step) for step in [400,800]]
                ax.plot([400,800],vals,color=color,alpha=.2)
            means=[]
            for step in [400,800]:
                vals=[r['error']['heldout'] for r in tr if r['architecture']==architecture and r['arm']==arm and r['steps']==step]
                means.append(np.mean(vals));training_summary.append(dict(architecture=architecture,arm=arm,steps=step,mean=float(np.mean(vals)),minimum=min(vals),maximum=max(vals),n_corpora=len(vals)))
            ax.plot([400,800],means,'o-',color=color,label=arm)
        ridge=np.mean([r['error']['heldout'] for r in tr if r['architecture']=='ridge' and r['arm']=='withheld'])
        ax.axhline(ridge,color='#555555',ls='--',label='ridge / withheld');ax.set(title=architecture,yscale='log',xlabel='Training steps',ylabel='Unseen-combination NMSE',xticks=[400,800]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training.png');plt.close(fig)
    mechanisms=[]
    for r in analysis['controls']:
        if r['domain']=='synthetic':mechanisms.append(dict(model=r.get('model','paired models'),comparison=r['contrast'],distractor=r.get('distractor_frequency'),n=r['n'],mean=r['mean'],lower=r['ci'][0],upper=r['ci'][1]))
    primary=[dict(claim=r['claim'],sources=r['n'],mean=r['mean'],lower=r['ci'][0],upper=r['ci'][1],positive=r['positive'],holm_p=r['holm_p'],passes=r['passes']) for r in analysis['primary']]
    selector=[]
    for method in dict.fromkeys(r['method'] for r in analysis['selectors']):
        row={'method':method}
        for domain in DOMAINS:row[NAMES[domain]]=next(r['loss'] for r in analysis['selectors'] if r['method']==method and r['domain']==domain)
        selector.append(row)
    final_training=[]
    for architecture in ['mlp','transformer','ridge']:
        for arm in ['withheld','covered']:
            selected=[r for r in tr if r['architecture']==architecture and r['arm']==arm and r['steps']==(0 if architecture=='ridge' else 800)]
            final_training.append(dict(architecture=architecture,arm=arm,**{g:float(np.mean([r['error'][g] for r in selected])) for g in ['single','seen','heldout']}))
    normalized_note='Normalized ridge control not run.'
    if normalization is not None:
        norm=read(normalization/'results.json');shutil.copyfile(normalization/'results.json',out/'normalized_ridge.json')
        means={arm:float(np.mean([r['error']['heldout'] for r in norm if r['arm']==arm])) for arm in ['withheld','covered']}
        normalized_note=f"A retrospective ridge control with the neural models' same past-only normalization gets held-out error {means['withheld']:.6g} without coverage and {means['covered']:.6g} with coverage. Normalization alone therefore does not make this task unsolvable for a linear predictor."
    distortion=[]
    for model in ['bolt','timesfm']:
        for distractor in [6.,17.]:
            for method in ['native','fft']:
                filt=dict(model=model,domain='synthetic',target_frequency=5.,distractor_frequency=distractor,amplitude=8.,length=128,method=method)
                distortion.append(dict(model=model,distractor=distractor,method=method,**{metric:average(records,metric,**filt) for metric in ['response_nmse','gain','gain_error','orthogonal_error']}))
    decomposition=[]
    for model in ['bolt','timesfm']:
        for domain in DOMAINS:
            filt=dict(model=model,domain=domain,method='fft',split='test',sign=0,length=128)
            cross=average(records,'error_cross',**filt);energy=average(records,'change_energy',**filt)
            decomposition.append(dict(model=model,domain=NAMES[domain],error_alignment=cross,forecast_change_energy=energy,total_loss_change=cross+energy))
    causal=[
      {'explanation':'Paired normalizer statistics create interference','test':'Lock native forward and inverse normalization to the same background','interpretation':'Compare anchored and native responses on the same 16 seeds. This tests statistic drift; it does not remove relative dynamic range or every normalization effect.'},
      {'explanation':'The target cannot be identified from the history','test':'Fit a two-tone classical estimator from history only','interpretation':'Low classical error refutes this for those tested signals. The estimator has a two-tone structural assumption; it is not a general real-series solution.'},
      {'explanation':'The adapter fails because both tones occupy one band','test':'Move one band boundary from bin5 to bin6','interpretation':'Same number of groups and lossless input reconstruction. Rescue is specific to whether the target and distractor are separated, not to decomposition count alone.'},
      {'explanation':'More evidence resolves the failure','test':'128 versus256 hours at the same forecast origin','interpretation':'Paired subset of16 seeds. Physical history and token count both change. No universal benefit is inferred.'},
      {'explanation':'Fourier truncation creates artificial patterns','test':'Off-grid tones, endpoint-bridge decomposition and high-band forecast suppression','interpretation':'Off-grid leakage and decomposition choice test scope. Dropping forecasts is a changed prediction assumption; its inputs no longer reconstruct the original signal.'},
      {'explanation':'Training lacks the relevant combinations','test':'Matched component exposures, two architectures, six fresh corpora,400/800 steps, ridge reference','interpretation':'Controlled training can establish a coverage effect in these small models. It cannot identify the hidden training histories of pretrained TSFMs.'},
      {'explanation':'Perturbation fidelity predicts practical forecasting quality','test':'Fixed input-only versus diagnostic selectors on source-disjoint real futures','interpretation':'Compare every selector with train-selected constants and past-only hindcasting. Test choices are invariant to changing test-future labels.'}]
    elapsed=sum(read(run/f'{m}_worker.json')['seconds'] for m in ['bolt','timesfm'])
    traffic=[r for r in records if r['model']=='timesfm' and r['domain']=='monash_traffic' and r['method']=='native' and r['sign']==0 and r['split']=='test']
    worst=max(traffic,key=lambda r:r['mse']);losses=[r['mse'] for r in traffic]
    traffic_note=f"Traffic has a heavy-tailed error distribution: native TimesFM mean {np.mean(losses):.6g}, median {np.median(losses):.6g}. Source {worst['source']} contributes {100*worst['mse']/sum(losses):.2f}% of its total TEST loss. It is retained under the frozen membership rule; domains are reported separately rather than pooled."
    key_filter=dict(model='timesfm',domain='synthetic',target_frequency=5.,distractor_frequency=17.,amplitude=8.,length=128)
    far_native=average(records,'response_nmse',method='native',**key_filter);far_fft=average(records,'response_nmse',method='fft',**key_filter)
    off_filter=dict(key_filter,target_frequency=5.35,distractor_frequency=17.35)
    off_native=average(records,'response_nmse',method='native',**off_filter);off_fft=average(records,'response_nmse',method='fft',**off_filter)
    train_withheld=next(r['heldout'] for r in final_training if r['architecture']=='transformer' and r['arm']=='withheld')
    train_covered=next(r['heldout'] for r in final_training if r['architecture']=='transformer' and r['arm']=='covered')
    summary_text=f"<p>TimesFM, strong far distractor: native response NMSE {far_native:.4g}, Fourier split {far_fft:.4g} on-grid; {off_native:.4g} and {off_fft:.4g} off-grid. The off-grid comparison tests whether the easy spectral separation generalizes.</p><p>Controlled transformer training: unseen-combination error at800 steps is {train_withheld:.4g} with combinations withheld and {train_covered:.4g} with coverage. All six fresh corpora favor coverage; component exposure counts and neural training budgets match.</p><p>The pedestrian response/forecast tradeoff passes the corrected primary tests. Electricity response improvement does not. Diagnostic selectors show no consistent incremental benefit over input-only selectors in this fixed screen.</p>"
    for domain in [DOMAINS[0],DOMAINS[2]]:
        gain=next(r['mean'] for r in analysis['primary'] if r['claim']=='timesfm:'+domain+':response_gain')
        harm=next(r['mean'] for r in analysis['primary'] if r['claim']=='timesfm:'+domain+':forecast_harm')
        summary_text+=f"<p>{NAMES[domain]} TEST sources: TimesFM native-minus-FFT response error {gain:.4g}; FFT-minus-native clean forecast error {harm:.4g}. Positive values mean better component response but worse actual forecasting, respectively. Full uncertainty and corrected tests appear below.</p>"
    body=f'''<header><p class="eyebrow">SECTION49 · GROUNDED FOLLOW-UP</p><h1>When component fidelity and forecast quality disagree</h1><p>Parked experiments1,7,8, followed by fresh-source replication and competing-explanation controls. Two pretrained models; {len(protocol['synthetic_seeds'])} synthetic seeds; {protocol['real_count_per_domain']*4} real sources across four datasets.</p><p class="meta">Native replication: {elapsed/60:.1f} CPU minutes · one thread/model · no GPU · offline checkpoints</p></header>
    <section><h2>Measured findings</h2>{summary_text}</section><section><h2>How to read the evidence</h2><p>Response NMSE measures whether adding a known oscillation produces the appropriate forecast change. Clean-future MSE measures actual prediction quality, in units of past variance. They answer different questions. A model can respond faithfully to an injected component and still predict the unmodified series poorly. Conversely, the off-grid controls show that decomposition itself can introduce a major failure.</p><p>Primary evidence is a fixed family of12 source-level directional tests with Holm correction and a mean-effect gate of0.1. Confidence intervals bootstrap source means, conditional on the sampled datasets. Mechanistic comparisons and model selection remain exploratory. Sources in the same dataset may be correlated; these tests do not establish broad population independence.</p></section>
    <section><h2>The parked interference map</h2><img src="pilot_map.png" alt="Pilot frequency and amplitude interference map"><p class="caption">ID1 pilot: eight fresh seeds, three target frequencies, nearby / harmonic / distant distractors, two amplitudes. This is a descriptive screen, not a corrected significance map. The replication below concentrates on target5 and5.35 and tests new off-grid cases; it does not confirm every pilot cell.</p></section>
    <section><h2>Interference, separation and identifiable signals</h2><img src="interference.png" alt="Response error by frequency separation, model and decomposition"><p class="caption">High distractor amplitude8, target amplitude0.5;64 fresh seeds, averaging paired positive/negative additions within seed. Log scale. Original FFT bands can separate far tones but group nearby tones together. Shift and bridge are shown only where actually evaluated. Classical frequencies are fitted from observed histories.</p>{table(distortion,['model','distractor','method','response_nmse','gain','gain_error','orthogonal_error'])}<p class="caption">Response error splits exactly into gain error and distortion orthogonal to the expected target waveform. The latter can include phase error and changes in other frequencies; it does not by itself identify a hidden mechanism. Mean gain alone can conceal errors that cancel across signs or seeds.</p>{table(mechanisms,['model','comparison','distractor','n','mean','lower','upper'])}<p class="caption">Exploratory paired contrasts: positive means the first method has higher response error. Normalizer/history rows use the declared16-seed subset; shifted splitting uses64. Intervals are conditional95% source bootstraps, without multiplicity correction.</p></section>
    <section><h2>Real data: response accuracy is not forecast accuracy</h2><img src="real_tradeoff.png" alt="Response and clean forecast errors on four real datasets"><p class="caption">16 untouched TEST sources per dataset; logarithmic vertical axes. Left: known bin5 oscillation, averaged over signs; right: unmodified real future. All series use past-only standardization. The full data includes an off-grid24-hour addition and endpoint-bridge controls in the downloadable tables.</p>{table(primary,['claim','sources','mean','lower','upper','positive','holm_p','passes'])}<p class="caption">All12 primary tests retained, including null or opposite results. The p-values test a directional majority, not a population mean. “Passes” additionally requires the empirical mean effect to exceed0.1.</p></section>
    <section><h2>Coverage can produce a persistent composition failure</h2><img src="training.png" alt="Matched training coverage results at400 and800 steps"><p class="caption">Six independent training corpora with paired initializations. Each arm sees exactly the same number of examples and component exposures; pair co-occurrence differs. Thin lines are corpora, thick lines their means. All checkpoints are retained. The linear ridge baseline indicates that the task is solvable without seeing the withheld pairs.</p>{table(training_summary,['architecture','arm','steps','mean','minimum','maximum','n_corpora'])}{table(final_training,['architecture','arm','single','seen','heldout'])}<p>{normalized_note}</p><p>This is a controlled finding about the small trained models. It is not evidence that any particular TSFM omitted these combinations during pretraining.</p></section>
    <section><h2>Where does the real forecast penalty come from?</h2>{table(decomposition,['model','domain','error_alignment','forecast_change_energy','total_loss_change'])}<p class="caption">Exact identity: MSE(adapter) − MSE(native) = 2 mean(native error × forecast change) + mean(forecast change²). These are observed total-future errors. The identity does not provide ground truth for individual real-data components.</p></section>
    <section><h2>Does the diagnostic help choose a forecaster?</h2>{table(selector,['method']+list(NAMES.values()))}{table([dict(domain=NAMES[r['domain']],estimator=r['estimator'],mean=r['mean'],lower=r['ci'][0],upper=r['ci'][1]) for r in analysis['selector_contrasts']],['domain','estimator','mean','lower','upper'])}<p class="caption">First table: TEST clean-future MSE; lower is better. Second: paired input-only minus diagnostic loss; positive favors diagnostics, with conditional exploratory95% intervals. Each selector trains only on64 FIT sources (16/domain), then chooses on64 TEST sources. Fixed ridge and RBF versions compare input-only descriptors against added past-only perturbation diagnostics. No method or hyperparameter is selected using TEST outcomes. Hindcast selects the model that best predicted the last32 observed points from an earlier128-point context. Full paired differences and oracle regret are in analysis.json.</p><p>{traffic_note}</p></section>
    <section><h2>Competing explanations and what the controls establish</h2>{table(causal,['explanation','test','interpretation'])}<p>“Coherence” here has several observable aspects: sensitivity to a small component, compositional behavior, stability under normalization/history changes, and actual forecast accuracy. None alone is a universal model ranking. Spectral separation can mechanically remove distractor influence without establishing correct interpretation of the original mixture.</p></section>
    <section><h2>Scope, provenance and reproducibility</h2><p>Real sources are disjoint from earlier pilots, but overlap with model pretraining is unknown. They must not be described as unseen-pretraining data. One forecast origin per source; no window-level pseudoreplication. Real component futures are unknown: the experiment scores known injected changes and the observed total future, not invented component ground truth.</p><p>Both native workers passed exact repeat and normalizer self-identity checks. Independent arithmetic audits replay saved losses, responses, primary tests and selector losses. Deliberate regression plants test the implementation guards. Resource caps are recorded in protocol.json.</p><p><a href="analysis.json">Complete analysis</a> · <a href="measurements.json">Every measurement</a> · <a href="protocol.json">Frozen protocol</a> · <a href="independent_audit.json">Independent audit</a> · <a href="training_results.json">Training results</a> · <a href="README.md">Reproduce</a></p></section>'''
    import re
    body=re.sub(r'>([^<]+)<',lambda m:'>'+re.sub(r'(?<=[A-Za-z])(?=[0-9])',' ',m[1])+'<',body)
    style='body{margin:0;background:#f4f6f8;color:#192b3b;font:16px/1.6 system-ui}header,section{max-width:1120px;margin:24px auto;padding:28px;background:white;border-radius:12px}header{background:#132b43;color:white}h1{font-size:38px;line-height:1.15}h2{font-size:24px}p{max-width:1000px}.eyebrow{letter-spacing:2px;font-size:12px}.meta,.caption{font-size:13px;opacity:.85}img{display:block;width:100%;height:auto}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce3e8;vertical-align:top}th{background:#eaf0f5}a{color:#245b9b}@media(max-width:650px){header,section{margin:8px;padding:16px}h1{font-size:28px}}'
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coherence follow-up · Section49</title><style>'+style+'</style>'+body+'</html>',encoding='utf-8')
    for name in ['analysis.json','measurements.json','protocol.json','independent_audit.json','selector_rows.json','execution.json','bolt_worker.json','timesfm_worker.json','environment.json']:shutil.copyfile(run/name,out/name)
    shutil.copyfile(training/'results.json',out/'training_results.json');shutil.copyfile(training/'independent_audit.json',out/'training_audit.json')
    shutil.copyfile(pilot/'analysis.json',out/'pilot_analysis.json')
    write(out/'training_summary.json',training_summary)
    (out/'README.md').write_text('''# Section49 coherence follow-up

See index.html for figures, full comparison tables and limits. ROADMAP49.31–49.34 records prospective designs and findings.

From the repository root, use the cudaPy environment with PYTHONPATH=tsfm_lens,
CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1,
HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1. All cached weights and data are local.

Existing results can be rescored without inference:
```
python -m experiments.feature_search.coherence_analysis --out tsfm_lens/runs/coherence_replication4932000
python -m experiments.feature_search.coherence_audit --out tsfm_lens/runs/coherence_replication4932000
python -m experiments.feature_search.coherence_audit --training --out tsfm_lens/runs/coverage_replication493900
python -m experiments.feature_search.coherence_report --run tsfm_lens/runs/coherence_replication4932000 --training tsfm_lens/runs/coverage_replication493900 --pilot tsfm_lens/runs/coherence_pilot4931000 --normalization tsfm_lens/runs/coverage_normalization493900 --out examples/feature_search_stage49/coherence_followup
```

To reproduce inference, use a NEW output directory; drivers refuse overwrites:
```
python -m experiments.feature_search.coherence_study prepare --stage replication --out NEW_RUN
python -m experiments.feature_search.coherence_study run --out NEW_RUN
python -m experiments.feature_search.coherence_classical --out NEW_RUN
python -m experiments.feature_search.coherence_analysis --out NEW_RUN
python -m experiments.feature_search.coverage_replication --out NEW_TRAINING_RUN
python -m experiments.feature_search.coverage_normalization_control --out NEW_NORMALIZATION_RUN
```
The synthetic generator labels are used only in scoring. Fourier decomposition,
classical frequency fitting, native workers and model-selection test choices
have no access to target futures. Fixed selectors use FIT future labels only.

Data provenance: cached Chronos electricity, M4 hourly, pedestrian and traffic
series. Fresh source membership is audited; pretraining overlap remains unknown.
Figures are standalone PNGs. Complete precision is retained in JSON artifacts.
''',encoding='utf-8')
    print(out/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','training','pilot','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--normalization',type=Path)
    a=p.parse_args();build(a.run,a.training,a.pilot,a.out,a.normalization)
