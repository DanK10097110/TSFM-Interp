"""Independent phase-response arithmetic, source-bootstrap replay and static ellipse report.

Projection is rederived from original forecasts with normal equations. Baseline
scores and exact odd/off-frequency/even closure are verified before interpretation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from experiments.feature_search.coherence_study import read,write,sha


def audit(out):
    """Replay all128 matrices from original forecasts and all source-grouped baseline summaries."""
    from experiments.feature_search.response_phase_study import validate
    p=validate(out);prior=Path(p['prior']);rows=read(out/'matrix_rows.json');results=read(out/'results.json')
    raw=np.load(out/'raw_arithmetic.npz');draws=np.load(out/'bootstrap_source_indices.npy')
    types=np.asarray([s['background_type'] for s in read(prior/'sources.json')])
    assert draws.shape==(4096,16)
    for kind in range(4):assert np.all(np.sum(types[draws]==kind,axis=1)==4)
    forecasts={(m,policy):np.load(prior/f'{m}_{policy}_forecast.npy',mmap_mode='r') for m in ('bolt','timesfm') for policy in ('native','locked')}
    c=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    checks=0;max_error=0.;closures=[];fitted={}
    def check(actual,expected):
        """Finite comparisons use a common FP64 numerical tolerance; null denominators stay null."""
        nonlocal checks,max_error
        if actual is None or expected is None:assert actual is expected
        else:
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11)
            max_error=max(max_error,float(np.max(abs(np.asarray(actual)-np.asarray(expected)))))
        checks+=1
    for row in rows:
        f=row['frequency'];b=np.stack([np.sin(2*np.pi*f*np.arange(32)/128),np.cos(2*np.pi*f*np.arange(32)/128)],axis=1)
        y=forecasts[row['model'],row['policy']][row['indices']].astype(float)
        base=forecasts[row['model'],row['policy']][row['base']].astype(float);a=row['amplitude']
        m=np.stack([y[0]-y[2],y[1]-y[3]],axis=1)/(2*a)
        even=np.stack([y[0]+y[2]-2*base,y[1]+y[3]-2*base],axis=1)
        h=np.linalg.solve(b.T@b,b.T@m);residual=m-b@h
        fitted[row['model'],row['policy'],f,row['source_id']]=h
        fi=0 if row['period']==24. else 1;key=f"{row['model']}_{row['policy']}_{row['source_id']}_{fi}"
        for suffix,value in [('delta',y-base),('odd',m*a),('even',even),('waveform',m),('H',h),('residual',residual)]:check(raw[f'{key}_{suffix}'],value)
        check(row['H'],h);check(b.T@residual,np.zeros((2,2)))
        q=(h[0,0]+h[1,1])/2;r=(h[1,0]-h[0,1])/2
        linear=np.array([[q,-r],[r,q]]);s=h-linear
        check(row['q'],q);check(row['r'],r);check(row['complex_linear'],linear);check(row['symmetric_traceless'],s)
        check(s,s.T);check(np.trace(s),0.);check(np.sum(s*linear),0.)
        check(row['anisotropy_frobenius_fraction'],np.sum(s*s)/np.sum(h*h) if np.sum(h*h)>0 else None)
        singular=np.linalg.svd(h,compute_uv=False)
        check(row['singular_values'],singular);check(row['determinant'],np.linalg.det(h))
        check(singular,[row['complex_gain']+row['conjugate_gain'],abs(row['complex_gain']-row['conjugate_gain'])])
        d=np.column_stack([b,np.ones(32)]);dc=np.linalg.solve(d.T@d,d.T@m)
        check(row['H_dc_adjusted'],dc[:2]);check(row['dc_coefficients'],dc[2])
        off=float(np.sum(residual**2)/np.sum(b**2));ev=float(np.sum(even**2)/(4*a*a*np.sum(b**2)))
        check(row['off_frequency_nmse_contribution'],off);check(row['even_nmse_contribution'],ev)
        check(row['off_frequency_odd_energy_fraction'],np.sum(residual**2)/np.sum(m**2) if np.sum(m**2)>0 else None)
        ideal=a*c@b.T;check(raw[f'{key}_ideal'],ideal)
        for name,k in [('identity',np.eye(2)),('loso',np.asarray(row['loso_H'])),('source_fit',h)]:
            prediction=a*c@k.T@b.T
            check(raw[f'{key}_prediction_{name}'],prediction)
            direct=float(np.sum((prediction-(y-base))**2)/np.sum(ideal**2))
            decomposed=float(np.sum((b@(h-k))**2)/np.sum(b**2)+off+ev)
            check(row['losses'][name],direct);check(direct,decomposed)
            closures.append(abs(direct-decomposed))
        assert row['losses']['source_fit']<=row['losses']['identity']+1e-10
        check(row['losses']['source_fit'],off+ev)
    for row in rows:
        others=[h for (model,policy,f,source),h in fitted.items() if model==row['model'] and policy==row['policy'] and f==row['frequency'] and source!=row['source_id']]
        assert len(others)==15;check(row['loso_H'],np.mean(others,axis=0))
    for summary in results['summary']:
        selected=sorted([r for r in rows if all(r[k]==summary[k] for k in ('model','policy','frequency'))],key=lambda r:r['source_id'])
        assert [r['source_id'] for r in selected]==list(range(16))
        for name in ('identity','loso','source_fit'):
            values=np.asarray([r['losses'][name] for r in selected]);boot=values[draws].mean(1)
            check(summary['baselines'][name]['mean'],values.mean());check(summary['baselines'][name]['percentile95'],np.quantile(boot,[.025,.975]))
        gains=np.asarray([r['losses']['identity']-r['losses']['loso'] for r in selected])
        check(summary['paired_identity_minus_loso']['mean'],gains.mean())
        check(summary['paired_identity_minus_loso']['percentile95'],np.quantile(gains[draws].mean(1),[.025,.975]))
        assert summary['loso_win_sources']==int(np.sum(gains>0))
    record=dict(passed=True,checks=checks,matrices=len(rows),independent_sources=16,bootstrap_source_draws=4096,
                maximum_absolute_discrepancy=max_error,maximum_energy_closure_discrepancy=float(max(closures)),
                source_hashes_verified=True,prior_forecasts_unchanged=True,native_capture=False)
    write(out/'audit.json',record)
    plot(out,rows)
    html_report(out,results['summary'],record)
    knowledge=read(out/'knowledge.json');knowledge['status']='complete_audited';knowledge['independent_audit']=record
    knowledge['interpretation']=[
        'H is a finite-amplitude conditional odd-response map, fitted on four cardinal phases at one amplitude. It is not an LTI transfer function or a tested full phase orbit.',
        'A large symmetric-traceless component is phase-sensitive or conjugate-linear behavior in these coordinates. Pure phase conjugation can still map a circle to a circle; singular values and determinant disambiguate shape and orientation.',
        'Source-global LOSO scores measure prediction of the model response to a known intervention, not improvement of its forecasting accuracy. Identity encodes ideal continuation semantics.',
        'Per-source H is fitted on the same responses and is an optimistic in-band explanatory ceiling. Its remaining waveform error separates exactly into off-frequency odd response and even response.',
        'Primary sine/cos projection can absorb finite-window DC leakage. The DC-adjusted H and intercept are sensitivity controls, not alternate results selected after seeing outcomes.'
    ]
    knowledge['next_decision']='Coordinator reviews held-out global-response prediction versus identity and conditional response ellipses. No native capture or expansion is part of this reuse-only study.'
    write(out/'knowledge.json',knowledge)
    write(out/'manifest.json',dict(status='complete_audited',protocol_sha256=sha(out/'protocol.json'),
          hashes={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print(record,flush=True)


def plot(out,rows):
    """Static coefficient-space response ellipses; all sources retained with background-type colors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    angles=np.linspace(0,2*np.pi,200);circle=np.stack([np.cos(angles),np.sin(angles)])
    fig,axes=plt.subplots(4,2,figsize=(10,17))
    for row_index,(model,policy) in enumerate([(m,p) for m in ('bolt','timesfm') for p in ('native','locked')]):
        for col,period in enumerate((24.,19.2)):
            ax=axes[row_index,col];selected=[r for r in rows if r['model']==model and r['policy']==policy and r['period']==period]
            ax.plot(circle[0],circle[1],'k--',lw=1,label='Ideal H=I')
            for r in selected:
                h=np.asarray(r['H']);ellipse=h@circle
                ax.plot(ellipse[0],ellipse[1],color=f"C{r['background_type']}",alpha=.55,lw=1)
                ax.plot(h[0,0],h[1,0],'.',color=f"C{r['background_type']}",ms=4)
            ax.axhline(0,color='.8',lw=.5);ax.axvline(0,color='.8',lw=.5);ax.set_aspect('equal',adjustable='datalim')
            ax.set_title(f'{model} / {policy} / {period:g} h');ax.set_xlabel('Output sine coefficient / input amplitude');ax.set_ylabel('Output cosine coefficient / input amplitude')
    fig.suptitle('Conditional odd-response matrices: 16 sources per panel; colors = four background types\nDashed unit circle = identity; dot = positive-sine response. Curves interpolate fitted H, not extra model evaluations.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'response_ellipses.png',dpi=150);plt.close(fig)


def html_report(out,summary,audit_record):
    """Compact portable report makes predictive and same-source explanatory scores distinct."""
    import html
    text='<!doctype html><meta charset="utf-8"><title>Conditional phase responses</title><style>body{max-width:1150px;margin:30px auto;font:16px system-ui}td,th{padding:7px;text-align:right}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}img{width:100%}</style>'
    text+='<h1>Conditional phase-response matrices</h1><p>Reuse-only, amplitude0.1,16 independent synthetic backgrounds crossed with four phases. No model inference. Source-held-out mean H predicts intervention responses; per-source H fits those same responses and is only an optimistic explanatory ceiling.</p>'
    text+='<p>Exact finite-window sine/cos projection. The complex-linear versus symmetric-traceless split does not assume time invariance. DC-adjusted matrices and raw odd/even residuals are saved.</p>'
    text+='<p><a href="protocol.json">Frozen protocol</a> · <a href="knowledge.json">Persistent knowledge</a> · <a href="matrix_rows.json">All source matrices</a> · <a href="audit.json">Independent audit</a></p>'
    text+='<table><tr><th>Model</th><th>Policy</th><th>Period</th><th>Identity NMSE</th><th>LOSO mean H</th><th>Source-fit ceiling</th><th>LOSO wins /16</th></tr>'
    for r in summary:
        cells=[r['model'],r['policy'],r['period'],r['baselines']['identity']['mean'],r['baselines']['loso']['mean'],r['baselines']['source_fit']['mean'],r['loso_win_sources']]
        text+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in cells)+'</tr>'
    text+='</table><p>NMSE pools phase energies before division and weights sources equally. Bootstrap intervals resample sources within background type; phases are never independent resampling units. These scores differ slightly from earlier unweighted phase-NMSE averages.</p>'
    text+='<img src="response_ellipses.png" alt="Fitted conditional response ellipses by source, model and normalization"><p>Ellipse curves are algebraic interpolations of two measured opposite-phase axes. Four phases cannot validate the interpolated orbit. Identity means correct sinusoidal continuation; an accurate response predictor can reproduce incorrect semantics.</p>'
    text+='<p>Independent arithmetic checks: '+html.escape(str(audit_record))+'</p>'
    (out/'report.html').write_text(text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    try:audit(args.out)
    except Exception as error:
        k=read(args.out/'knowledge.json');k['status']='failed_independent_audit';k['failed_instrumentation_runs'].append(dict(stage='audit',error=repr(error)));write(args.out/'knowledge.json',k);raise
