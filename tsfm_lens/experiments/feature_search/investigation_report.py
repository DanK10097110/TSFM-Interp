"""Portable scientific snapshot of the hypothesis-driven Section49 investigation.

This renderer does no inference or statistical selection. Exact reviewed tables
and knowledge records accompany figures, while unfinished work stays explicitly
pending. Rebuild after reviewing new experiments, then inspect rendered HTML.
"""
from pathlib import Path
import argparse,html,json,hashlib,shutil,re
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    """Read a recorded result, never infer a missing stage's outcome."""
    return json.loads(path.read_text(encoding='utf-8'))


def table(rows,columns):
    """Round displayed numbers only; downloadable JSON retains exact values."""
    def fmt(value):
        if isinstance(value,float):return f'{value:.6g}'
        return html.escape(str(value))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c.replace('_',' '))+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(row.get(c,''))+'</td>' for c in columns)+'</tr>' for row in rows)+'</tbody></table></div>'


def build(runs,out,complete=False):
    """Draw source-resampled evidence and index durable per-experiment knowledge."""
    out.mkdir(parents=True,exist_ok=True);portable=out/'records';portable.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    colors=['#176a85','#c26131','#57679c'];sections=[];exact={}
    comp=read(runs/'comparability_replication4952000/results.json');exact['comparability']=comp
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    names=[]
    for j,r in enumerate(comp):
        c=r['contrasts']['primary_gain'];den=r['losses']['input']['mean']
        ax.errorbar(100*c['mean']/den,j,xerr=[[100*(c['mean']-c['low'])/den],[100*(c['high']-c['mean'])/den]],fmt='o',capsize=4,color=colors[j])
        names.append(r['A'].title()+' → '+r['B'].title()+'\n'+r['endpoint'].replace('_',' '))
    ax.axvline(0,color='#777',lw=1);ax.set(yticks=range(3),yticklabels=names,xlabel='Absolute loss gain / observed baseline mean (%)',title='Cross-model predictive utility: 64 fresh synthetic sources')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.savefig(out/'comparability.png');plt.close(fig)
    comp_rows=[dict(Direction=r['A']+' → '+r['B'],Target=r['endpoint'],Input=r['losses']['input']['mean'],
                    Hidden=r['losses']['input+A_hidden']['mean'],Forecast=r['losses']['input+A_forecast']['mean'],
                    Both=r['losses']['input+A_forecast+A_hidden']['mean']) for r in comp]
    sections.append('<section><h2>Comparison has bounded predictive value</h2><p>All three frozen primary contrasts replicated. The intervals resample 64 source seeds and adjust for three comparisons; five conditions within each seed are dependent. The probes were fitted on 20 earlier FIT sources. Hidden means input plus the other model’s hidden states; Forecast uses its forecast instead; Both uses both.</p><img src="comparability.png" alt="Positive prediction loss reductions in three cross-model comparisons">'+table(comp_rows,['Direction','Target','Input','Hidden','Forecast','Both'])+'<p>For forecast targets, input plus source forecast outperforms input plus source hidden states. Adding hidden states to the forecast is an exploratory positive. This demonstrates finite-data predictive usefulness in these conditions, not shared feature meanings, new information beyond the exact input, or cross-model causal mediation. The exploratory generic-feature control below narrows this result substantially. The earlier pilot also had mixed or negative real-data and failure-prediction endpoints; the successful replication concerns these particular synthetic targets.</p></section>')

    norm=read(runs/'response_surface_normalization4953000/results.json')['rows']
    exact['normalization_primary']=[r for r in norm if r['policy']=='native' and r['cut'] in ('0','input','metadata')]
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    labels=['Physical\ndifference','Normalized\ndifference','Normalized\nendpoints']
    for ax,model in zip(axes,('bolt','timesfm')):
        vals=[];base=[]
        for route in ('physical_delta','normalized_delta','normalized_pair'):
            vals.append(next(r['physical']['nmse'] for r in norm if r['model']==model and r['policy']=='native' and r['route']==route and r['feature']=='hidden' and r['cut']=='0'))
            base.append(next(r['physical']['nmse'] for r in norm if r['model']==model and r['policy']=='native' and r['route']==route and r['feature']=='input'))
        ax.bar(np.arange(3)-.16,vals,.32,label='Hidden + route inverse',color=colors[0]);ax.bar(np.arange(3)+.16,base,.32,label='Input + same inverse',color=colors[1])
        ax.set(title=model.title(),yscale='log',xticks=range(3),xticklabels=labels,ylabel='Physical-coefficient NMSE');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(out/'normalization.png');plt.close(fig)
    sections.append('<section><h2>A decoder can manufacture apparent difficulty</h2><p>On eight reused pilot backgrounds, correctly normalized targets and inverse transforms greatly improve early-activation decoding. The normalized-difference inverse uses observed background Fourier coefficients and history statistics. Every matched control receives the same inverse information. This is not a hidden-only information claim.</p><img src="normalization.png" alt="Decoder error falls with correct normalization; hidden features do not universally beat matched input"><p>The endpoint route avoids the explicit background-coefficient correction and gives mixed hidden-versus-input results. Strong decodability does not establish causal semantics; poor decoding by one parameterization does not establish that information is absent.</p></section>')

    natural=read(runs/'natural_validation_replication4955000/analysis.json')
    keep=natural['primary']
    exact['natural_replication_depth1']=keep
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    domains=['monash_electricity_hourly','m4_hourly','monash_traffic'];short=['Electricity','M4 hourly','Traffic']
    for ax,model in zip(axes,('bolt','timesfm')):
        for j,d in enumerate(domains):
            r=next(r for r in keep if r['model']==model and r['domain']==d);c=r['gain_interval'];v=r['paired_mean_gain']
            ax.errorbar(j,v,yerr=[[v-c[0]],[c[1]-v]],fmt='o',capsize=4,color=colors[0])
        ax.axhline(0,color='#777',lw=1);ax.set(title=model.title(),xticks=range(3),xticklabels=short,ylabel='Native MSE − depth1 hybrid MSE');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fresh natural data: 64 sources per domain; six adjusted intervals')
    fig.savefig(out/'natural_replication.png');plt.close(fig)
    sections.append('<section><h2>Limited Fourier separation helps one real-data cell and hurts two</h2><p>The frozen depth1 rule extracts a sufficiently stable, prominent tone from history, continues it analytically, and asks the native model to forecast the residual. It can abstain. Errors are MSE after standardizing each series by its own available history. On 192 fresh sources, TimesFM electricity MSE falls from 0.552 to 0.468; its paired gain interval excludes zero after six-comparison adjustment. Bolt electricity is inconclusive. Traffic deteriorates in both models. M4 is inconclusive.</p><img src="natural_replication.png" alt="Adjusted paired source intervals: TimesFM electricity improves; both traffic cells deteriorate"><p>A seasonal24 baseline has lower electricity mean MSE (0.399), although its paired difference from the TimesFM hybrid is uncertain. Thus the positive hybrid result establishes a bounded model limitation, not a better forecasting method. The selector misses some gains but no longer rejects all of them. Source resampling is conditional on these datasets and shared calendar periods; public-source overlap with pretraining remains unknown.</p><p>Projection onto the history-fitted tone plane attributes 0.0351 of the TimesFM electricity gain to that plane and 0.0488 outside it. Residual forecasting changes more than the peeled frequency. This is an exact error partition, not identification of the true future components.</p></section>')

    weather=read(runs/'postrelease_weather_pilot202610/knowledge.json')['exact_findings']
    weather_rows=[]
    for model,r in weather['forecast_results'].items():
        weather_rows.append(dict(Model=model,Native=r['losses']['native']['equal_station_normalized_mse'],
                                 Hybrid=r['losses']['hybrid']['equal_station_normalized_mse'],
                                 Seasonal24=r['losses']['seasonal24']['equal_station_normalized_mse'],
                                 Station_wins=str(r['comparisons']['hybrid_vs_native']['station_wins'])+'/8'))
    exact['chronological_weather_primary']=weather_rows
    sections.append('<section><h2>A chronological real-data holdout gives a negative result</h2><p>October 2026 temperatures from eight predetermined airports postdate the exact cached model revisions. Thirty complete forecast windows remain after excluding two missing futures; every station receives equal weight. The frozen depth1 hybrid worsens both observed mean errors. It accepts 18 histories and abstains on 12.</p>'+table(weather_rows,['Model','Native','Hybrid','Seasonal24','Station_wins'])+'<p>These are genuinely later observations, although locations and climate patterns may be familiar to the models. Shared weather limits station independence, so intervals are descriptive. Archived observations are not a real-time vintage, and null quality flags mean unflagged rather than verified QC completion. The result limits the generality of the electricity positive; it does not show that real data are structureless.</p></section>')

    sections.append('<section><h2>Normalization does not explain the real-data split</h2><p>Keeping the full-history normalization fixed while forecasting the peeled residual retains the TimesFM electricity gain: normalized MSE is 0.551575 natively, 0.467636 with ordinary peeling and 0.469563 with fixed normalization. The small difference between the two hybrids is uncertain, not an equivalence result.</p><p>On the later weather observations, fixed normalization worsens Bolt’s hybrid mean from 0.654730 to 0.843145 and changes TimesFM’s from 0.506455 to 0.476327, an uncertain improvement still above its native 0.450347. Neither yields a general repair. Forward and inverse normalization were locked together, so their contributions are not separated. Historical batch-versus-singleton numerical drift in the electricity replay was measured and bounded; source gain signs were unchanged.</p></section>')

    sections.append('<section><h2>How to read the comparison intervals</h2><p>The first plot rescales paired absolute-gain intervals by the observed baseline mean. These are not bootstrap intervals for a percentage ratio. The underlying source-level absolute-gain contrasts determine the primary decisions.</p></section>')
    random=read(runs/'comparability_random4953000/results.json')
    exact['random_feature_controls']=random
    sections.append('<section><h2>Generic features explain part of the comparison result</h2><p>On reused replication sources, both random-feature families beat Bolt hidden states for predicting TimesFM forecasts. For predicting Bolt forecasts from TimesFM, random ReLU features are competitive while random Fourier features lose to trained states. Trained TimesFM states retain an advantage for predicting Bolt’s activation-fusion response: loss 0.002074 versus about 0.002248 for either random-feature family, conditional on input and source forecast. The controls match feature budgets and selection procedures, not architectures or FLOPs; this follow-up is exploratory.</p></section>')
    shifted=read(runs/'comparability_conditions4962000/primary_result.json')
    exact['condition_shift_primary']=shifted
    sections.append('<section><h2>A specific comparative advantage survives a fresh test</h2><p>With 20 new FIT and 32 untouched TEST sources spanning five amplitude/frequency conditions, locally refitted TimesFM hidden-state probes predict Bolt’s fusion response with loss 0.002255. The matched random Fourier and ReLU controls give 0.002963 and 0.002445. Both trained-feature reductions (23.9% and 7.75%) pass the frozen 5% threshold and positive family-adjusted source intervals.</p><p>Excluding the anchor condition descriptively reduces the trained-hidden advantage over random ReLU to 3.58%, with an interval crossing zero. Old probes transferred without refitting perform substantially worse on shifted conditions. This supports bounded feature efficiency. Predicting a fusion response also does not show that fusion improves the forecast: the fresh anchor condition deteriorates for both models.</p></section>')

    phase=read(runs/'response_phase4954200/results.json')['summary'];exact['phase_response']=phase
    phase_rows=[]
    for r in phase:
        if r['policy']=='native':
            phase_rows.append(dict(Model=r['model'],Period=r['period'],Identity=r['baselines']['identity']['mean'],
                                   Shared=r['baselines']['loso']['mean'],Oracle=r['baselines']['source_fit']['mean'],
                                   Wins=str(r['loso_win_sources'])+'/16'))
    sections.append('<section><h2>Wrong responses can still have a simple structure</h2><p>Across 16 fresh backgrounds and four crossed phases, local curvature usually shrinks with input distance. That does not make the response correct. A two-coordinate sine/cosine map fitted on the other 15 sources explains part of the weak-tone response, especially Bolt at period19.2. The per-source map is an optimistic explanatory ceiling fitted on the same phases.</p>'+table(phase_rows,['Model','Period','Identity','Shared','Oracle','Wins'])+'<p>These scores predict the model’s response, not the true future. Phase-table errors are normalized by ideal injected-sinusoid energy, not by observed model-response energy. Four phases and one amplitude cannot characterize an entire response surface. Off-frequency changes, nonlinear even responses and background dependence remain. The reused-source intervals are exploratory; source-held-out fitting is not a new confirmatory dataset.</p></section>')

    transfer=read(runs/'response_phase_transfer4954300/results.json');exact['fresh_phase_map']=transfer
    transfer_rows=[]
    for r in transfer['descriptive']:
        if r['policy']=='native' and r['offset']==0 and abs(r['center']-128/19.2)<1e-8:
            v=r['mean_nmse'];transfer_rows.append(dict(Model=r['model'],Zero=v['zero'],Scalar=v['scalar_bounded'],
                Gain_phase=v['complex_linear'],Full_map=v['full_H'],Oracle=v['oracle_same_source']))
    sections.append('<section><h2>The phase structure survives fresh backgrounds</h2><p>On 16 new backgrounds, the frozen period19.2 map predicts responses better than an old-fitted scalar attenuation model. Both models retain positive paired gain intervals after four-comparison adjustment. A simple gain-plus-phase map performs similarly to the full two-coordinate map.</p>'+table(transfer_rows,['Model','Zero','Scalar','Gain_phase','Full_map','Oracle'])+'<p>The oracle is fitted on the same fresh responses and is only an explanatory ceiling. All predictive coefficients come from old sources. Intervals condition on the four equally weighted background types; changing their mixture may change the conclusion. Substantial unexplained error remains, especially for TimesFM.</p></section>')

    sections.append('<section><h2>Phase transfer extends beyond the fitted phase axes</h2><p>Eight additional oblique phases on the same 16 transfer backgrounds retain the full-map advantage over attenuation: mean normalized error reductions are 0.2482 for Bolt and 0.1845 for TimesFM, with adjusted source intervals above zero. No predictive coefficients were refitted. These are fresh phase inputs, not new independent backgrounds.</p><p>A saved-array output-frequency comparison explains a little more of the period19.2 response. Old-source selection favors output period24 for Bolt and22 for TimesFM, but trend controls change the Bolt selection and weaken the TimesFM result. More flexible waveform templates do better still. Short windows, nuisance choices and source variation prevent identifying a unique internal oscillator or frequency-snapping mechanism.</p></section>')

    horizon=read(runs/'response_phase_horizon_recovery4954700/results.json');exact['direct64_frequency']=horizon
    horizon_rows=[dict(Horizon=r['horizon'],Amplitude=r['amplitude'],Selected_period=r['selected_period'],Same_frequency=r['same_test_nmse'],Selected_frequency=r['selected_test_nmse']) for r in horizon['rows'] if r['input_period']==19.2 and r['variant']=='tone_only']
    sections.append('<section><h2>Longer forecasts retain distortion even for strong tones</h2><p>A Bolt-only direct64-head test reuses the 16 OLD FIT and 16 transfer TEST backgrounds. OLD-source selection favors a 22-step output period for 19.2-step inputs at both weak and strong amplitude over 64 steps. Matched DC/trend controls agree there; all period24 controls retain24. Every displayed frequency-map gain has a positive family-adjusted source interval.</p>'+table(horizon_rows,['Horizon','Amplitude','Selected_period','Same_frequency','Selected_frequency'])+'<p>These errors predict model responses, normalized by ideal injected-tone energy. They are not forecast improvements. Amplitude changes normalization as well as signal dominance. Horizon-dependent selections, nonlinear even responses and the finite candidate grid prevent identifying a unique oscillator, training prior or patch mechanism.</p><p>The first capture failed a historical replay comparator that changed batch geometry; its missing numerical delta is preserved as a limitation. A four-case replay diagnostic and outcome-blind amendment precede this recovery. Current64/current32 prefixes match exactly on identical queries and use one model forward. Audited historical rebatching NMSE drift is at most 1.91×10⁻⁶, far smaller than these gains; 1,210 score/replay and 1,544 roundoff checks pass.</p></section>')

    sections.append('<section><h2>Separate frontend jumps from internal roughness</h2><p>Chronos-T5 has a discrete tokenizer: tiny raw changes can cross a token boundary. The fresh paired 16-source tiny/small-checkpoint test finds one-token jumps at about 4.84×10⁻⁹ input RMS, with finite encoder and next-token distribution changes. Within the continuous embedding space, trained probability curvature decreases from the largest to smallest tested radius on all 16 sources in both models. This identifies one source of jaggedness without claiming every internal direction is smooth. Random-model softmax curvature is below the numerical-resolution gate and is excluded.</p><p>For Bolt and TimesFM, a separate nine-quantile experiment finds that other quantiles and central summaries do not resolve the weak-tone response errors. Even a best-quantile oracle leaves large errors. That makes a median-only explanation inadequate in the tested conditions; it does not establish poor probabilistic calibration.</p></section>')

    prior=read(runs/'prior_training_pilot4951000/results.json')['records'];exact['toy_prior']=prior
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for model,color in zip(('mlp','transformer'),colors):
        for j,seed in enumerate(sorted({r['seed'] for r in prior if r['architecture']==model})):
            selected=sorted([r for r in prior if r['architecture']==model and r['seed']==seed],key=lambda r:r['training_probability'])
            ax.plot([r['training_probability'] for r in selected],[r['mean_gain'] for r in selected],'o-',color=color,alpha=.7,label=model if j==0 else None)
    ax.plot([0,1],[0,1],':',color='#333',label='Conditional-mean optimum');ax.set(xlabel='Training survival probability',ylabel='Predicted tone gain',title='Toy causal mechanism: identical histories, different futures');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'training_prior.png');plt.close(fig)
    sections.append('<section><h2>A continuation prior is possible, but insufficient as a full explanation</h2><p>Matched tiny-model training changes only future tone survival probability. Both architectures learn approximately the conditional-mean gain. This establishes a possible mechanism in toy MSE training; it does not identify pretrained-model training. The actual native outputs studied here are medians. Their API names should not be mistaken for the prediction functional.</p><img src="training_prior.png" alt="Small matched models learn tone gains corresponding to survival probabilities"><p>Random initialization already permits tone decoding, so decodability alone does not show a learned concept. In the pretrained pilots, even the best fitted conservative scalar leaves large weak-tone errors. Phase and off-frequency changes remain. Some off-frequency changes align with the background forecast, including under fixed normalization, but that template is also incomplete.</p></section>')

    objectives=read(runs/'prior_objective_control4963000/results.json');exact['objective_control']=objectives
    sections.append('<section><h2>The training loss changes the interpretation of attenuation</h2><p>In 24 matched tiny-model arms, a 25% chance of tone survival produces gains near 0.25 under squared loss but near zero under absolute loss (median training). At 75%, the corresponding gains are near 0.75 and 0.98. At exactly 50%, an interval of marginal medians is optimal; different partial gains can be similarly good. These controlled mechanisms prevent reading a survival probability directly from a pretrained median forecast.</p></section>')

    ffn=read(runs/'ffn_selectivity_pilot4961000/compact_record.json')
    direct=[]
    for r in ffn['records']:
        if r['group']=='synthetic' and r['site']=='first' and r['budget']==4 and r['view']=='rms':
            a=r['metrics']['accuracy'];c=a['controls']
            direct.append(dict(Model=r['model'],Native=a['native']['mean'],Rotation=c['rotation_average']['score']['mean'],PCA=c['PCA']['score']['mean'],Input=c['actual_input']['score']['mean']))
    exact['illustrative_direct_FFN_slice']=direct
    sections.append('<section><h2>Actual neurons can be useful without an SAE</h2><p>The direct-feature pilot captures post-activation FFN neurons, rather than treating residual coordinates as neurons. It uses five equal-energy perturbation types, FIT-only feature selection and source-held-out evaluation. This illustrative early-layer, four-feature RMS slice is selected from the full registered table; its eight synthetic TEST sources support an exploratory comparison.</p>'+table(direct,['Model','Native','Rotation','PCA','Input'])+'<p>Bolt’s native coordinates are more selective than these rotations; TimesFM shows no corresponding advantage here. Input coordinates already identify the intervention well. Different layers and pooling rules give different results. This is perturbation selectivity, not proof that neurons have one meaning generally. Rotations are structured rather than Haar, and PCA has a different candidate count. Causal testing of the selected units is a separate stage.</p></section>')

    causal=read(runs/'ffn_selectivity_causal_replication4964000/analysis.json')['primary']
    exact['direct_neuron_causal_replication']=causal
    sections.append('<section><h2>Classifier selectivity did not yield substantial selective response removal</h2><p>The small causal pilot suggested a relative advantage for four FIT-selected early Bolt neurons. A larger 32-source test crossed four frozen activity-matched control sets with every source. The advantage did not replicate: signed target-response removal differed by 0.000312, with a 95% source interval from −0.00693 to 0.00734. Selected signed removal before subtracting controls was 0.000834, also uncertain and practically small under the predeclared ±0.02 descriptive band.</p><p>All 160 selected edits were nonzero and affected outputs. Their target-edit energy after the down-projection was 2.72 times the mean control energy. Thus the result concerns aligned response removal, not a failure to intervene or an absence of all output effects. It bounds this chosen four-unit edit; it does not rule out larger groups, other layers or other features.</p></section>')

    dictionaries=read(runs/'dictionary_pilot4951000/recovery_exit_monitor/comparison_summary.json')
    exact['dictionary_comparisons']=dictionaries
    metrics=[('state_nmse','State reconstruction\n(normalized MSE)'),('sinusoid_hidden_response_nmse','Sinusoid-response reconstruction\n(normalized MSE)'),('native_point_distortion','Forecast distortion\n(history variance units)')]
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for i,model in enumerate(('bolt','timesfm')):
        for j,(metric,label) in enumerate(metrics):
            selected=[next(r for r in dictionaries['rows'] if r['model']==model and r['k']==16 and r['method']==method) for method in ('pca','topk','nested')]
            means=np.array([r[metric]['mean'] for r in selected]);lo=np.array([r[metric]['min'] for r in selected]);hi=np.array([r[metric]['max'] for r in selected])
            axes[i,j].bar(range(3),means,color=colors,yerr=[means-lo,hi-means],capsize=3)
            axes[i,j].set(xticks=range(3),xticklabels=['PCA','TopK','Nested'],title=('Bolt' if model=='bolt' else 'TimesFM')+' · k16',ylabel=label)
            axes[i,j].grid(axis='y',alpha=.2)
    fig.savefig(out/'dictionary_fidelity.png');plt.close(fig)
    sections.append('<section><h2>Good state reconstruction can hide poor weak-response fidelity</h2><p>The matched dictionary pilot compares native axes, rotations, adaptive sparse PCA, TopK and a nested-prefix variant inspired by Matryoshka SAEs. All use measured activity budgets; stochastic methods have three initializations. Error bars below show the initialization range, not confidence intervals. State error is normalized by FIT-centered state energy; hidden-response error is sourcewise response-energy-normalized. Both use 32 TEST sources, half synthetic. Forecast distortion is normalized by history variance over 20 receivers, only four synthetic. Different denominators and source weighting prevent direct comparisons between columns.</p><img src="dictionary_fidelity.png" alt="Low whole-state reconstruction errors coexist with larger normalized weak-response errors"><p>PCA is competitive and often preserves forecasts better; TimesFM k8 is an exception where TopK beats PCA. There is no uniform SAE winner. PCA uses a full FIT basis and input-dependent top-k support, not a fixed rank16 subspace. Nested training uses about 1.25 times the dense forward-matmul work at equal steps and parameters.</p><p>A static position-conditioned FIT mean explains 80.3% of Bolt’s global-mean-relative state error and 10.1% of TimesFM’s. It cannot respond to changing inputs, and it does not explain all PCA gains. Compression, responsiveness and semantic interpretability remain distinct.</p><p>The audit marks 2,048 of 3,520 causal response cells with invalid or inapplicable random controls, including zero-edit cases, as unscorable. Exact base replay finds 119 of 120 TimesFM TopK-k8 edits are zero. Cached perturbed-state proxies are kept separate from unsaved singleton branches. These cells cannot support a claim that the features are harmless. Fallback copies are excluded from null evidence. Independently sampled plus/minus edits are not calibrated smooth-response nulls, even when individual geometries are valid. The valid reconstruction and saved-difference results remain useful; a separate response-aware objective pilot tests the mismatch directly.</p></section>')

    smooth=read(runs/'dictionary_smoothness4965000/leading_findings.json');exact['dictionary_induced_geometry']=smooth
    smooth_rows=[dict(Model=g['model'],Normalization=g['policy'],Minimum=g['actual_to_raw_ratio_of_source_means_range'][0],Maximum=g['actual_to_raw_ratio_of_source_means_range'][1]) for g in smooth['groups']]
    sections.append('<section><h2>The measuring dictionary can introduce apparent jaggedness</h2><p>On 16 saved synthetic backgrounds, frozen sparse PCA, TopK and nested dictionaries at k8/16 amplify actual-path chord roughness. The table gives the ratio of source-mean decoded chord RMS to source-mean raw-state chord RMS; ranges span the six fixed method/budget choices, not uncertainty intervals.</p>'+table(smooth_rows,['Model','Normalization','Minimum','Maximum'])+'<p>Isolating each dictionary on a straight activation chord retains substantial roughness. Identity and fixed-first-k linear PCA give zero at numerical precision. Nonzero active-support switching coincides with resolved dictionary curvature; it can include ReLU zero crossings or changes in TopK membership. This first audit does not separate those contributions. All 24 sparse-arm means exceed their raw counterparts, although individual paths need not.</p><p>The isolated interpolation is off-manifold, and raw model states also have nonzero curvature. Compression changes reconstruction error and response magnitude. This is a control against attributing every feature discontinuity to the model, not proof that a smoother dictionary is more semantic. FP64 dictionary arithmetic, absolute errors and stricter resolution floors support the measured distinction.</p></section>')

    fixed=read(runs/'dictionary_smoothness_fixed_support4965100/knowledge.json')['exact_findings'];exact['fixed_support_control']=fixed
    fixed_rows=[dict(Model=r['model'],Method=r['method'],Chord_reduction_percent=100*r['actual_chord_fraction_removed'],Response_error_reduction_percent=100*r['response_error_fraction_removed']) for r in fixed['exact_table']]
    sections.append('<section><h2>Holding feature membership fixed isolates the source of those jumps</h2><p>With the same weights and locked-normalization paths, freezing the k16 selected indices at each actual midpoint removes over 99% of mean actual-path chord RMS. Leaving ReLU active or freezing its central gate gives identical saved-path metrics here, isolating changes in TopK membership.</p>'+table(fixed_rows,['Model','Method','Chord_reduction_percent','Response_error_reduction_percent'])+'<p>Response error here is hidden-state endpoint-minus-midpoint reconstruction RMS, not output or forecast error. It improves on every source in each model/method cell. State reconstruction changes little and worsens slightly in five of six means. The support is chosen locally for each path: this is a controlled diagnosis, not a deployable global encoder. No native inference or new fitting was used.</p></section>')

    response=read(runs/'dictionary_response4965000/knowledge.json')['exact_results'];exact['response_aware_dictionary']=response
    response_rows=[dict(Method=k,State=v['global_nmse'],Sinusoid=v['old_response']['sinusoid'],Noise=v['old_response']['noise'],Focused=v['focused_nmse']) for k,v in response['table'].items()]
    sections.append('<section><h2>Changing the SAE objective preserves different information</h2><p>A paired-response reconstruction loss substantially improves small-response fidelity in a matched Bolt TopK-k16 pilot. Both arms use the same initialization, batches, 500 steps and computed branches. Lambda0 uses ordinary state reconstruction; lambda1 adds FIT-scaled sinusoid/noise difference reconstruction.</p>'+table(response_rows,['Method','State','Sinusoid','Noise','Focused'])+'<p>All four columns are normalized MSE, with different denominators: State uses FIT-centered state energy; Sinusoid and Noise average sourcewise error-to-response-energy ratios; Focused averages four dependent cell ratios within each source, then sources. The columns are not directly comparable. Response gains hold on all 32 old TEST sources and all 16 focused sources with held-out directions/phases. Global reconstruction worsens. This one-seed descriptive pilot demonstrates an objective-dependent tradeoff, not monosemantic features or improved forecasting. The focused bank is excluded from fitting and selection. Native reinsertion is evaluated separately; this simplified objective is inspired by response consistency, not a reproduction of TSAE.</p></section>')

    native_response=read(runs/'dictionary_response_native4965000/knowledge.json')['exact_results'];exact['response_aware_native']=native_response
    native_response_rows=[dict(Method=k,Baseline=v['baseline_mean'],Sinusoid=v['responses']['sinusoid']['mean_source_nmse'],Noise=v['responses']['noise']['mean_source_nmse']) for k,v in native_response['table'].items()]
    sections.append('<section><h2>Better hidden-response fidelity does not guarantee better output fidelity</h2><p>Reinserting the frozen reconstructions into Bolt gives a mixed result. Baseline distortion uses 20 existing receivers and history variance as its denominator; paired-response NMSE uses 15 existing receivers and each native output response’s energy. These are different cohorts and quantities.</p>'+table(native_response_rows,['Method','Baseline','Sinusoid','Noise'])+'<p>Lambda1 improves on 12/15 sinusoid and 11/15 noise receivers, but adverse cases make the primary source-average errors 30.2% and 23.4% worse. PCA has the lowest mean in all three columns. No response denominator falls below the recorded floor.</p><p>Aggregation matters: pooling noise response energies instead of averaging sourcewise ratios reverses that contrast, from 0.537 to 0.274. This sensitivity is reported without replacing the primary metric. Baseline mean distortion also improves despite only 3/20 individual gains. Nine exact independent forecast replays, identity/restoration gates and 792 arithmetic checks pass. These results concern preservation of native behavior, not prediction accuracy against real futures.</p><p>The saved-array follow-up confirms hidden-response improvement on all 15 actual singleton sources, including the adverse output cases. Two low-response synthetic series dominate sine deterioration; two real series dominate noise deterioration. Noise error normalized by history variance nearly doubles, so its adverse tail is not merely a small-denominator artifact. Most net NMSE deterioration is orthogonal to the native response. Raw pooled energy also depends on measurement units: after history normalization, pooled noise error worsens from 0.513 to 1.010. Removing the worst cases reverses the means, but no sources are excluded from the primary result. This identifies a hidden/output fidelity mismatch without identifying a downstream Jacobian mechanism.</p></section>')

    papers=[('Superposition Is Not Necessary','https://arxiv.org/abs/2605.05151','Narrow PatchTST feed-forward-neuron study; not a blanket finding about pretrained residual streams.'),
            ('TSAE','https://arxiv.org/abs/2610.04925','Structured temporal and variable-aware SAE design; univariate pilots do not reproduce its multivariate routing.'),
            ('Matryoshka SAEs','https://proceedings.mlr.press/v267/bussmann25a.html','Nested dictionaries motivate a compact coarse/fine comparison; our simplified variant is labeled inspired-by.'),
            ('BatchTopK','https://arxiv.org/abs/2412.06410','Motivates activity-aware comparisons and a frozen inference threshold, rather than comparing nominal k alone.'),
            ('Dissecting Chronos','https://arxiv.org/abs/2603.10071','Model-specific feature interventions motivate causal controls; results on Chronos-T5 do not automatically transfer to Bolt.'),
            ('Chronos','https://arxiv.org/abs/2403.07815','Discrete scaling and quantization motivate separating frontend discontinuities from encoder behavior.')]
    sections.append('<section><h2>Direct features and modern SAE designs</h2><p>The direct-coordinate/PCA/rotation/TopK/nested-dictionary experiment tests whether an SAE adds value. Residual coordinates are not post-activation feed-forward neurons. Reconstruction, actual activity, source-held-out probes, seed stability and native forecast preservation must be considered together. Low reconstruction error alone is insufficient. The artifact index records whether each stage actually completed.</p><ul>'+''.join('<li><a href="'+url+'">'+title+'</a>: '+desc+'</li>' for title,url,desc in papers)+'</ul></section>')

    names=['prior_training_pilot4951000','prior_initial_control4951000','persistence_pilot4951000',
           'persistence_replication4953000','comparability_pilot4951000','comparability_replication4952000',
           'response_surface_pilot4951000','response_surface_normalization4953000',
           'attenuation_explanation4951000','background_modulation4956100',
           'natural_validation_pilot4951000','natural_validation_pilot4951001','natural_validation_replication4955000',
           'dictionary_pilot4951000','token_boundary_pilot4956000','quantile_response4956200',
           'prior_objective_control4963000','investigation_validation4963000',
           'comparability_dictionary_edit_audit20261116']
    for pattern in ('response_surface_crossed*','response_surface_focused*','comparability_random*',
                    'comparability_conditions*','response_phase*','token_boundary_replication*',
                    'natural_gain_location*','ffn_selectivity*','postrelease_weather*',
                    'persistence_electricity_lock*','comparability_position_mean*',
                    'dictionary_response*','dictionary_smoothness*'):
        names+=sorted(p.name for p in runs.glob(pattern) if p.is_dir())
    manifest=[];cards=[]
    for name in dict.fromkeys(names):
        root=runs/name
        if not root.exists():continue
        knowledge=sorted(root.rglob('knowledge.json'))
        for file in knowledge:
            relative=file.relative_to(runs);target=portable/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
            data=read(file);status=data.get('status',data.get('exact_findings',{}).get('status','recorded') if isinstance(data.get('exact_findings'),dict) else 'recorded')
            if data.get('phase'):status=str(data['phase'])+': '+str(status)
            digest=hashlib.sha256(file.read_bytes()).hexdigest()
            manifest.append(dict(path=str(relative),status=status,sha256=digest))
            if file.parent==root or file.parent.name in ('recovery_exit_monitor','singleton_geometry','ghcnh_oct2026'):
                cards.append('<li><a href="records/'+html.escape(str(relative))+'">'+html.escape(str(relative.parent))+'</a> — '+html.escape(str(status))+'</li>')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'exact_tables.json').write_text(json.dumps(exact,indent=2,allow_nan=False),encoding='utf-8')
    sections.append('<section><h2>Persistent experiment records</h2><p>Each completed experiment and failed stage has a durable knowledge record with its question, exact results, controls, limitations and next decision. Child records preserve recovery histories. A prepared or running record is not a scientific result. Full arrays remain in the repository runs directory.</p><ul>'+''.join(cards)+'</ul><p><a href="exact_tables.json">Exact figure tables</a> · <a href="manifest.json">Knowledge-file hashes and statuses</a> · <a href="../selective_composition/index.html">Earlier selective-composition study</a></p></section>')
    state='Completed bounded investigation' if complete else 'Ongoing investigation — interim snapshot'
    stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    toc=[]
    for index,section in enumerate(sections):
        title=re.search(r'<h2>(.*?)</h2>',section).group(1)
        sections[index]=section.replace('<section>',f'<section id="s{index}">',1)
        toc.append(f'<a href="#s{index}">{title}</a>')
    navigation='<nav aria-label="Report contents"><h2>Contents</h2><div class="toc">'+''.join(toc)+'</div></nav>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TSFM hypotheses: comparative evidence</title><style>
    body{margin:0;background:#f4f5f5;color:#20313c;font:17px/1.6 system-ui,sans-serif}header{background:#143847;color:#fff;padding:46px max(24px,calc((100vw - 1080px)/2))}header p{max-width:930px}h1{line-height:1.15;font-size:38px;margin:12px 0}main{max-width:1080px;margin:auto;padding:24px}section{background:#fff;border:1px solid #dae1e3;border-radius:10px;margin:24px 0;padding:28px}h2{font-size:25px;line-height:1.3}a{color:#126c89}img{display:block;width:100%;height:auto;margin:20px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #dae1e3}th{background:#edf3f5}li{margin:8px 0}.tag{font-size:14px;color:#b9dce5}footer{font-size:14px;color:#566b76;padding:24px} @media(max-width:600px){h1{font-size:29px}section{padding:17px}main{padding:12px}}
    nav{background:#fff;border:1px solid #dae1e3;padding:24px;border-radius:10px}.toc{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px 22px}.toc a{font-size:14px;line-height:1.4}section{scroll-margin-top:16px}
    </style><header><div class="tag">SECTION 49 · '''+state+'''</div><h1>Are TSFM representations too different or too messy to study?</h1><p>The evidence supports narrower, testable distinctions: useful cross-model prediction can coexist with different internal coordinates; normalization and readout choices can create apparent failures; smooth responses can still be non-additive or semantically wrong.</p></header><main>'''+navigation+''.join(sections)+'<footer>Generated '+stamp+'. Figures are descriptive unless their source protocol specifies a held-out contrast. No result establishes universal interpretability or universal impossibility.</footer></main></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    print(dict(out=str(out),knowledge_records=len(manifest),status=state))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--complete',action='store_true')
    args=parser.parse_args();build(args.runs,args.out,args.complete)
