"""Portable scientific figures and source-level summary for steering experiments."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'examples/steering'
RUNS=ROOT/'tsfm_lens/runs'
TASKS=['widen','narrow','season','upper']
LABELS=['Widen 20%','Narrow 20%','Seasonal prior','Upper quantile only']


def interval(values, seed=5240000):
    """Context-level bootstrap interval for a mean; not independent horizon cells."""
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(4000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),lo=float(np.quantile(means,.025)),
                hi=float(np.quantile(means,.975)),n=len(values))


def render():
    """Rebuild from persisted raw rows; examples are prospectively first test contexts."""
    OUT.mkdir(parents=True,exist_ok=True)
    source={}
    results={}
    for distribution,exp,suffix in [('fresh','e06','fresh64'),('offgrid','e07','offgrid64')]:
        for model in ['bolt','timesfm']:
            name=f'steering_{exp}_{model}_{suffix}'
            path=RUNS/name/'results.json'
            result=json.loads(path.read_text())
            results[(distribution,model)]=result
            record=dict(results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),run=name,tasks={})
            for task in TASKS:
                one={r['seed']:r for r in result['rows'] if r['task']==task and r['steps']==1}
                six={r['seed']:r for r in result['rows'] if r['task']==task and r['steps']==6}
                if set(one)!=set(six):
                    raise ValueError('paired sample IDs differ')
                keys=sorted(six)
                record['tasks'][task]=dict(error=interval([six[k]['relative_error'] for k in keys]),
                    improvement=interval([one[k]['relative_error']-six[k]['relative_error'] for k in keys]),
                    successes=sum(six[k]['relative_error']<.1 for k in keys),
                    baseline_crossing_fraction=float(np.mean([six[k]['baseline_crossing_fraction'] for k in keys])),
                    edited_crossing_fraction=float(np.mean([six[k]['crossing_fraction'] for k in keys])))
            source[f'{distribution}/{model}']=record
    (OUT/'source_data.json').write_text(json.dumps(source,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    colors={'bolt':'#245b88','timesfm':'#c46c2f'}
    for ax,distribution in zip(axes,['fresh','offgrid']):
        for offset,model in [(-.16,'bolt'),(.16,'timesfm')]:
            stats=[source[f'{distribution}/{model}']['tasks'][task]['error'] for task in TASKS]
            means=np.array([v['mean'] for v in stats])
            lower=np.array([v['lo'] for v in stats]);upper=np.array([v['hi'] for v in stats])
            ax.bar(np.arange(4)+offset,means,width=.29,color=colors[model],label=model,
                   yerr=np.stack([means-lower,upper-means]),capsize=3)
        ax.axhline(.1,color='#666666',linestyle='--',linewidth=1,label='Per-context success threshold')
        ax.set_xticks(np.arange(4),['Widen','Narrow','Season','Upper only'])
        ax.set_yscale('log');ax.set_ylim(.0003,.25)
        ax.set_title('Fresh synthetic contexts' if distribution=='fresh' else 'Fresh off-grid periods')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Mean full-output relative error (log scale)')
    axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('SAE-free adaptive steering • 64 contexts per model and distribution')
    fig.tight_layout()
    fig.savefig(OUT/'replication.png',dpi=180)
    fig.savefig(OUT/'replication.pdf')
    plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
    for row,model in enumerate(['bolt','timesfm']):
        data=np.load(RUNS/f'steering_e06_{model}_fresh64/interventions.npz')
        base=data['baseline'][0]
        t=np.arange(32)
        for col,task in enumerate(TASKS):
            ax=axes[row,col]
            key=f'0_{task}_0.2_0.2_6'
            actual=data[key+'_forecast'];desired=base+data[key+'_target']
            ax.fill_between(t,base[:,0],base[:,-1],color='#999999',alpha=.2,label='Original 10–90% band')
            ax.plot(t,base[:,4],color='#777777',linewidth=1,label='Original median')
            ax.fill_between(t,actual[:,0],actual[:,-1],color=colors[model],alpha=.25,label='Steered band')
            ax.plot(t,actual[:,4],color=colors[model],linewidth=1.5,label='Steered median')
            for channel in [0,4,8]:
                ax.plot(t,desired[:,channel],color='#222222',linestyle=':',linewidth=.8,
                        label='Requested forecast' if channel==4 else None)
            if row==0:ax.set_title(LABELS[col])
            if col==0:ax.set_ylabel(f'{model}\nNormalized value')
            if row==1:ax.set_xlabel('Forecast step')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
    fig.suptitle('Actual internal edits • first replication context (seed 5210000; no example selection)')
    fig.tight_layout(rect=(0,.05,1,.96))
    fig.savefig(OUT/'forecast_examples.png',dpi=180)
    fig.savefig(OUT/'forecast_examples.pdf')
    plt.close(fig)

    lines=['# Selective internal forecast steering','',
           'The opt-in controller works without an SAE. It measures the native response,',
           'solves for a bounded hidden edit, and verifies the resulting nonlinear forecast.',
           'It can also combine SAE decoder atoms or operate in a supplied subspace.','',
           '![Fresh-seed replication](replication.png)','',
           'All rows below use frozen settings: dose0.2, total hidden radius0.2, ridge1e-5,',
           'at most six correction steps. Success means full-output relative RMS error<0.1.',
           'Intervals are bootstrap95% intervals over contexts, not forecast cells.','',
           '| Distribution / model | Task | Mean error [95% interval] | Successes |',
           '| --- | --- | --- | ---: |']
    for name,record in source.items():
        for task,value in record['tasks'].items():
            v=value['error']
            lines.append(f'| {name} | {task} | {v["mean"]:.5f} [{v["lo"]:.5f}, {v["hi"]:.5f}] | {value["successes"]}/64 |')
    lines+=['','![Forecast examples](forecast_examples.png)','',
            'These are adaptive output-control results, not proof of a fixed semantic vector,',
            'monosemantic neurons, improved forecast accuracy, or calibrated uncertainty.',
            'Upper-only control remains harder and quantile crossings are reported separately.',
            'SAE comparisons are discovery experiments with one training seed; modern signed',
            'and batch-adaptive variants improved reconstruction but not consistently steering.','',
            '- [API and reproduction guide](../../tsfm_lens/docs/steering.md)',
            '- [Persistent experiment log, including failures](../../tsfm_lens/docs/steering_research_log.md)',
            '- [Current literature and design motivations](../../tsfm_lens/docs/steering_literature.md)',
            '- [Exact plotted statistics and source hashes](source_data.json)','',
            'Every experiment directory includes protocol.json, results.json and findings.md.',
            'Full local arrays/checkpoints are under tsfm_lens/runs/steering_*. The four',
            'replication datasets also ship their intervention arrays and independent audits here.']
    upstream_path=RUNS/'steering_e09_upstream_fresh16/results.json'
    if upstream_path.exists():
        upstream=json.loads(upstream_path.read_text())
        fig,ax=plt.subplots(figsize=(8,4))
        patterns=['reg','shared','full']
        for offset,pattern,color in zip([-.25,0,.25],patterns,['#8aa2b6','#245b88','#c46c2f']):
            values=[upstream['summary'][f'{pattern}/{task}']['error'] for task in ['widen','season','upper']]
            ax.bar(np.arange(3)+offset,values,.23,label=pattern,color=color)
        ax.axhline(.1,color='#666',linestyle='--',linewidth=1)
        ax.set_xticks(np.arange(3),['Widen','Season','Upper only'])
        ax.set_ylabel('Mean full-output relative error')
        ax.set_title('Before the Bolt decoder • 16 fresh contexts')
        ax.legend(title='Encoder token edit scope')
        fig.tight_layout();fig.savefig(OUT/'upstream.png',dpi=180);fig.savefig(OUT/'upstream.pdf');plt.close(fig)
        lines+=['','## Upstream boundary','',
                '![Encoder-output control](upstream.png)','',
                'The longer encoder-output replication used the same frozen decoder and a20%',
                'whole-state Frobenius edit budget. Widening passes15/16(REG),14/16(shared),',
                '13/16(full); seasonal control passes15/16,16/16,16/16. Upper-only passes0/16',
                'for every scope, including independent control of every encoder token.',
                'This is a restriction-specific negative, not proof of global impossibility.',
                'Head and encoder norm budgets are not directly comparable.','',
                '## Real contexts','',
                'On twelve cached real evaluation windows (four each electricity,M4,pedestrian),',
                'passes for widening/narrowing/season/upper are12/11/12/8 for Bolt and12/12/12/12',
                'for TimesFM. Settings were unchanged and all public API predictions were verified',
                'through full-model internal hooks. Sources were reused from earlier research;',
                'pretraining overlap is unknown. Future outcomes were not evaluated.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    render()
