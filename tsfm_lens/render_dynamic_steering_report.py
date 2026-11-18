"""Build portable evidence and static scientific figures from saved experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'tsfm_lens/runs';OUT=ROOT/'examples/dynamic_steering'
SUITES={'Bolt, H64':'dynamic_d07_bolt_full','TimesFM, H64':'dynamic_d08_timesfm_full',
        'Timer, H96':'dynamic_d16_timer_rowspace_fresh'}
KINDS=['level','trend','detrend','amplitude','seasonality','deseasonalize','smooth','roughness',
       'uncertainty','asymmetry','quantile','anchor','mean_revert','bounds','delay']


def read_rows(name):return json.loads((RUNS/name/'rows.json').read_text())


def render():
    OUT.mkdir(parents=True,exist_ok=True);manifest={};summaries={}
    for run in sorted(RUNS.glob('dynamic_d*')):
        if not (run/'findings.md').exists():continue
        dest=OUT/run.name;dest.mkdir(exist_ok=True)
        for name in ('protocol.json','discovery.json','rows.json','summary.json','findings.md','forecasts.npz'):
            source=run/name
            if source.exists():
                shutil.copy2(source,dest/name)
                manifest[f'{run.name}/{name}']=dict(bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if (run/'rows.json').exists():
            rows=read_rows(run.name)
            if rows and 'achieved' in rows[0]:
                learned=[r for r in rows if r['site']['source']!='context_policy']
                policies=[r for r in rows if r['site']['source']=='context_policy']
                summaries[run.name]=dict(learned_n=len(learned),learned_successes=sum(r['achieved'] for r in learned),
                    mean_error=float(np.mean([r['relative_error'] for r in learned])) if learned else None,
                    input_policies=len(policies),replay_max_abs=max(r['replay_max_abs'] for r in rows))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'source_data.json').write_text(json.dumps(summaries,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    means=np.full((len(KINDS),len(SUITES)),np.nan);counts=np.full(means.shape,'',dtype=object)
    for j,(label,run) in enumerate(SUITES.items()):
        rows=read_rows(run)
        for i,kind in enumerate(KINDS):
            group=[r for r in rows if r['kind']==kind and r['window']=='all']
            if group:
                means[i,j]=np.mean([r['relative_error'] for r in group])
                counts[i,j]=f"{sum(r['achieved'] for r in group)}/{len(group)}"
            else:counts[i,j]='point only'
    fig,ax=plt.subplots(figsize=(8,8));cmap=plt.get_cmap('viridis_r').copy();cmap.set_bad('#eeeeee')
    heat=ax.imshow(np.ma.masked_invalid(means),norm=LogNorm(.01,1),cmap=cmap,aspect='auto')
    for i in range(len(KINDS)):
        for j in range(len(SUITES)):
            ax.text(j,i,counts[i,j],ha='center',va='center',fontsize=9,color='white' if means[i,j]>.15 else 'black')
    ax.set_yticks(range(len(KINDS)),KINDS);ax.set_xticks(range(len(SUITES)),list(SUITES))
    ax.set_title('Automatic steering on eight fresh contexts per model\nCells: successes / cases; color: mean relative error')
    fig.colorbar(heat,ax=ax,label='Relative error (success threshold: 0.10)',shrink=.7)
    fig.tight_layout();fig.savefig(OUT/'capabilities.png',dpi=180);fig.savefig(OUT/'capabilities.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
    for row,(label,run) in enumerate(list(SUITES.items())[:2]):
        records=read_rows(run);data=np.load(RUNS/run/'forecasts.npz')
        for col,(kind,window) in enumerate([('seasonality','all'),('uncertainty','short')]):
            record=next(r for r in records if r['kind']==kind and r['window']==window and r['strength']>0)
            k=record['request_index'];base=data[f'{k}_baseline'][0];actual=data[f'{k}_forecast'][0];target=base+data[f'{k}_target'][0]
            t=np.arange(len(base));ax=axes[row,col]
            ax.fill_between(t,base[:,0],base[:,-1],color='#888888',alpha=.2,label='Original interval')
            ax.plot(t,base[:,1],color='#888888',lw=1,label='Original median')
            ax.fill_between(t,actual[:,0],actual[:,-1],color='#286a9b',alpha=.25,label='Steered interval')
            ax.plot(t,actual[:,1],color='#286a9b',lw=1.5,label='Steered median')
            for q in range(3):ax.plot(t,target[:,q],':',color='black',lw=.8,label='Requested' if q==1 else None)
            if window=='short':ax.axvspan(len(base)//2,len(base)-1,color='#cfa62b',alpha=.08)
            ax.set_title(f'{label}: {kind} ({window})');ax.set_ylabel('Original series units')
            if row==1:ax.set_xlabel('Forecast step')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',fontsize=8)
    fig.suptitle('Actual full-model interventions: first test context, selected before results')
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(OUT/'forecasts.png',dpi=180);fig.savefig(OUT/'forecasts.pdf');plt.close(fig)

    lines=['# Dynamic steering: implementation and evidence','',
        'A researcher can register a deterministic predictor and optional PyTorch module, train steering profiles,',
        'and use the same API for unfamiliar models. Layer and token locations are discovered automatically.',
        'The interface supports 19 behavioral requests, protected short/long/explicit forecast windows,',
        'and operational context relevance policies. Successful control is measured, not assumed.','',
        '- [API and custom-model registration guide](../../tsfm_lens/docs/dynamic_steering.md)',
        '- [Runnable custom-model example, no downloads](custom_model.py)',
        '- [Persistent research log, including failed experiments](../../tsfm_lens/docs/dynamic_steering_research_log.md)',
        '- [Source statistics](source_data.json), [artifact hashes](manifest.json), and [independent array audit](artifact_audit.json)','',
        '![Capabilities](capabilities.png)','',
        'Success means full-output relative error ≤0.10, including protected coordinates. All cases are retained.',
        'The three suites use eight fresh contexts, but different horizons/output contracts; this is not a ranked model benchmark.',
        'Bolt/TimesFM use rank 256; Timer uses an exact discovered Linear row space within rank 128.',
        'Large absolute bounds are a deliberate stress request and often exceed the radius 0.2 budget.',
        'Point-only Timer does not receive invented quantile/uncertainty outputs.','',
        '| Experiment | Learned/additive successes | Explicit input policies | Mean learned error |',
        '|---|---:|---:|---:|']
    for run,v in summaries.items():
        mean=f"{v['mean_error']:.4f}" if v['mean_error'] is not None else '—'
        lines.append(f"| [{run}]({run}/findings.md) | {v['learned_successes']}/{v['learned_n']} | {v['input_policies']} | {mean} |")
    lines+=['','Explicit full-horizon input policies are separated because their target is the native forecast on',
        'the gated context by definition. They demonstrate operational relevance control, not internal steering ability.',
        'For Bolt and TimesFM, all 64 ignore/focus sensitivity cases were exactly invariant to changes in removed regions.',
        'Emphasizing input deviations increased output sensitivity in 27/32 cases; nonlinear response is not universally monotonic.','',
        '![Actual forecasts](forecasts.png)','',
        'Every returned forecast in the completed steering experiments was independently replayed through the full predictor;',
        'maximum replay discrepancy was zero. The forecast outputs were never substituted with target arrays.',
        'The figures show the first preselected test context, including any residual error. Shading marks the protected half.','',
        'What changed during research:','',
        '1. A signed-effect screen rejects executed branches that do not affect the registered forecast.',
        '2. Discovery handles reused module invocations and verified expanded batches in autoregressive prediction.',
        '3. Exact Linear row spaces avoid wasting the intervention budget on invisible directions.',
        '4. New doses/requests can reuse learned response maps with explicitly labeled online selection.',
        '5. Input policies give exact ignore/focus semantics; partial-horizon policies remain measured, budget-limited interventions.','',
        'The strongest practical limit is target size and intervention geometry. Large clipping targets and some selective',
        'width/context requests fail despite other controls succeeding. Bounded-search failure does not prove model-wide',
        'impossibility. Successful late-head control also does not establish semantic neurons, a fixed reusable steering vector,',
        'better forecast accuracy, or calibrated uncertainty. Real-source checks reuse earlier public datasets, with unknown',
        'pretraining overlap. See individual protocols for exact samples, seeds and selection rules.','',
        'Reproduce from `tsfm_lens/` using the experiment entry points and each directory’s `protocol.json`:','',
        '```bash',
        'python experiments/dynamic_steering_study.py --help',
        'python experiments/dynamic_context_audit.py --help',
        'python experiments/dynamic_real_audit.py --help',
        'python experiments/dynamic_transfer_audit.py --help',
        'python render_dynamic_steering_report.py',
        'python audit_dynamic_artifacts.py','```','',
        'Use OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 for the reported CPU setup.',
        'Portable artifacts include raw forecast arrays and per-case diagnostics. Learned profile arrays remain in the local',
        '`tsfm_lens/runs/.../profiles` directories; researchers can retrain and save their own through the public API.',
        'Early pilots predate improvements recorded in the log; final code is expected to improve or change their discovery paths.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(experiments=len(summaries),artifacts=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()))))


if __name__=='__main__':render()
