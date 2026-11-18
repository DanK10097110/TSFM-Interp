"""CPU-only, source-bound summary of stable40/20 phase responses.

Partial waveform coverage precludes an internal-feature or complete mechanism
claim. No32/16 mode identities are interpreted, no new fits or tests of models.
"""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
import numpy as np
from experiments.feature_search import phase_orbit as orbit
from experiments.feature_search import phase_orbit_alias as alias
from experiments.feature_search import phase_orbit_native as native
from experiments.feature_search import phase_orbit_fp64 as fp64

MODELS=('timesfm','bolt','chronos2')
TERMS=('expected_axis_wrong_time','higher_axis','joint')


def disjoint_terms(budget):
    """Retain all low-order phase modes; no branch-energy cross-term issue."""
    terms=dict.fromkeys(TERMS,0.);mode_contributions={}
    for mode in budget['modes']:
        k,l=mode['k'],mode['l'];degree=abs(k)+abs(l)
        if degree>3 or degree==0:continue
        category=mode['category']
        if category=='expected_axis_shape':
            value=mode['physical_time_character_residual_energy'];term=TERMS[0]
        else:
            value=mode['predicted_energy'];term='joint' if k!=0 and l!=0 else 'higher_axis'
        terms[term]+=value
        mode_contributions[f'{k},{l}']=value
    if not np.isclose(sum(terms.values()),budget['primary_D'],rtol=1e-10,atol=1e-12):
        raise ValueError('Disjoint D components must close exactly to primary D')
    return terms,mode_contributions


def aggregate(budgets):
    terms=[];contributions=[]
    for budget in budgets:
        a,b=disjoint_terms(budget);terms.append(a);contributions.append(b)
    mean_terms={term:float(np.mean([row[term] for row in terms])) for term in TERMS}
    modes=[]
    for mode in budgets[0]['modes']:
        k,l=mode['k'],mode['l']
        if abs(k)+abs(l)>3:continue
        matched=[next(r for r in b['modes'] if (r['k'],r['l'])==(k,l)) for b in budgets]
        modes.append(dict(k=k,l=l,category=mode['category'],
            predicted_energy=float(np.mean([r['predicted_energy'] for r in matched])),
            error_energy=float(np.mean([r['error_energy'] for r in matched])),
            D_contribution=float(np.mean([r.get(f'{k},{l}',0.) for r in contributions]))))
    return dict(D_terms=mean_terms,D=sum(mean_terms.values()),
        scaled_native_MSE=float(np.mean([b['scaled_MSE'] for b in budgets])),
        gain_rotation_error=float(np.mean([b['expected_axis_gain_rotation_error'] for b in budgets])),
        full_error_budget={key:float(np.mean([b['error_categories'][key] for b in budgets])) for key in budgets[0]['error_categories']},
        low_order_modes=modes)


def collect(source):
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    results=json.loads((source/'results.json').read_text(encoding='utf-8'))
    if protocol['math_sha256']!=native.digest(orbit.__file__) or not results['new_native_controls_pass']:
        raise ValueError('Require unchanged math and passed amended native controls')
    pilot=Path(protocol['original_pilot']);coarse_source=Path(protocol['coarse_source']);stem='p40_20'
    rows=json.loads((source/f'{stem}_requests.json').read_text(encoding='utf-8'))
    old_rows=json.loads((coarse_source/f'{stem}.json').read_text(encoding='utf-8'))
    truth=np.load(coarse_source/f'{stem}.npz')['clean_future']
    _,fine_truth,fine_rows=orbit.make_orbit((40,20),grid=16,bundles=orbit.REFINEMENT_BUNDLES)
    lookup={(r['request_kind'],r['bundle'],r['cell']):i for i,r in enumerate(rows)}
    models={};bindings={}
    for model in MODELS:
        response=results['responses'][model][stem]
        if not response['all_four_low_order_stable']:raise ValueError('Interpret ONLY stable40/20 responses')
        coarse_path=pilot/f'{model}_{stem}_prediction.npz';new_path=source/f'{model}_{stem}_new_prediction.npz'
        if native.digest(coarse_path)!=protocol['coarse_forecast_sha256'][f'{model}/{stem}']:raise ValueError('Coarse forecast changed')
        coarse=native.load_prediction(coarse_path,model,stem,coarse_source)['prediction'].astype(np.float64)
        new=alias.load_new_prediction(new_path,model,stem,source).astype(np.float64)
        budgets=[];fine_budgets=[]
        for bundle in range(8):
            row=old_rows[bundle*64];origins=(row['origin_phi'],row['origin_psi']);scale=row['scale']
            coarse_wave=coarse[bundle*64:(bundle+1)*64].reshape(8,8,64)
            budgets.append(fp64.corrected_budget(coarse_wave,truth[bundle*64:(bundle+1)*64].reshape(8,8,64),origins,scale,40))
            if bundle in orbit.REFINEMENT_BUNDLES:
                fine=np.empty((16,16,64),np.float64)
                for i in range(16):
                    for j in range(16):
                        fine[i,j]=coarse_wave[i//2,j//2] if i%2==0 and j%2==0 else new[lookup['fine_new',bundle,i*16+j]]
                index=orbit.REFINEMENT_BUNDLES.index(bundle)
                fine_budgets.append(fp64.corrected_budget(fine,fine_truth[index*256:(index+1)*256].reshape(16,16,64),origins,scale,40))
        summary=aggregate(budgets);matched=aggregate([budgets[b] for b in orbit.REFINEMENT_BUNDLES]);fine_summary=aggregate(fine_budgets)
        off=[response['offgrid_bundles'][str(b)]['waveform_RMS_over_shared_scale'] for b in range(8)]
        summary.update(coarse_matched_four=matched,fine_four=fine_summary,
            offgrid_RMS_by_bundle=off,offgrid_mean_unexplained_energy=float(np.mean(np.square(off))),
            offgrid_pass_count=response['offgrid_pass_count_all_eight'],response_status=response['status'],
            coarse_per_bundle_terms=[disjoint_terms(b)[0] for b in budgets],
            fine_per_bundle_terms=[disjoint_terms(b)[0] for b in fine_budgets])
        ranked=sorted(summary['low_order_modes'],key=lambda r:r['D_contribution'],reverse=True)
        summary['top_descriptive_D_modes']=ranked[:6]
        models[model]=summary
        bindings[model]=dict(coarse_forecast_sha256=native.digest(coarse_path),new_forecast_sha256=native.digest(new_path))
    return dict(scope='Partial coverage; no internal feature claim. Only stable40/20 low-order identities described;32/16 shown as validation status only.',
        normalization_caveat='Finite-window phase-dependent native preprocessing can itself create modes. These are native forecast responses,not an assigned transformer/normalization cause.',
        denominator='Per-bundle theoretical shared mixture energy (a²+b²)/2;means over fixed designed amplitude bundles,not population estimates;no p-values or CIs',
        D_definition='Disjoint low-order higher-axis+joint energy plus expected-axis wrong physical-time-character energy;DC and pure gain/rotation excluded',
        uncovered_caveat='Offgrid unexplained energy measures reconstruction of native forecasts at OTHER phases;it is not an additive part of native true-future MSE.',
        models=models,status_matrix={m:{pair:results['responses'][m][pair]['status'] for pair in ('p32_16','p40_20')} for m in MODELS},
        original_pilot_failures_retained=True,original_native_ranking_reversal_failed=True,hidden_capture_authorized=False,
        source_paths=dict(alias_results=str(source/'results.json'),alias_protocol=str(source/'protocol.json')),
        source_hashes=dict(alias_results=native.digest(source/'results.json'),alias_protocol=native.digest(source/'protocol.json'),
            coarse_metadata=native.digest(coarse_source/f'{stem}.json'),new_metadata=native.digest(source/f'{stem}_requests.json'),driver=native.digest(__file__)),
        forecast_bindings=bindings)


def render(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    out.mkdir(parents=True,exist_ok=True)
    colors=dict(timesfm='#2878b5',bolt='#d66335',chronos2='#37946a')
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),gridspec_kw=dict(width_ratios=[1,1.4]))
    ax=axes[0,0];x=np.arange(3)
    for i,m in enumerate(MODELS):
        values=[max(data['models'][m]['D_terms'][term],1e-12) for term in TERMS]
        ax.bar(x+(i-1)*.23,values,width=.22,color=colors[m],label=m)
        fine=[max(data['models'][m]['fine_four']['D_terms'][term],1e-12) for term in TERMS]
        ax.scatter(x+(i-1)*.23,fine,s=20,facecolors='none',edgecolors='black',zorder=4)
        matched=[max(data['models'][m]['coarse_matched_four']['D_terms'][term],1e-12) for term in TERMS]
        ax.scatter(x+(i-1)*.23,matched,s=45,marker='_',color='black',zorder=4)
    ax.set_yscale('log');ax.set_xticks(x,['Expected axis:\nwrong time character','Higher-axis\nmodes','Joint\nphase modes'],fontsize=8)
    ax.set_ylabel('Energy / shared mixture energy');ax.set_title('A  Disjoint low-order D terms, 40/20',loc='left',fontsize=10)
    ax.legend(fontsize=8);ax.text(0,-.27,'Bars: coarse, all 8 bundles. Black dashes: matched 4.\nOpen dots: fine, same fixed 4 bundles.',transform=ax.transAxes,fontsize=8)
    ax=axes[0,1]
    for i,m in enumerate(MODELS):
        values=data['models'][m]['offgrid_RMS_by_bundle'];ax.scatter(i+np.linspace(-.12,.12,8),values,color=colors[m],s=24)
    ax.axhline(.05,color='black',ls='--',lw=1,label='Frozen coverage bound')
    ax.set_xticks(range(3),MODELS);ax.set_ylabel('Off-grid waveform RMS / shared scale')
    ax.set_title('B  Held-out response reconstruction, all 8 bundles',loc='left',fontsize=10);ax.legend(fontsize=8)
    ax=axes[1,0];ax.axis('off')
    labels={'unresolved_or_aliased':'Unresolved / aliased','stable_and_adequate':'Stable + adequate','stable_low_order_partial_coverage':'Stable; partial coverage'}
    table=ax.table(cellText=[[labels[data['status_matrix'][m][p]] for p in ('p32_16','p40_20')] for m in MODELS],rowLabels=MODELS,colLabels=['32/16','40/20'],cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2)
    ax.set_title('C  Full validation status matrix',loc='left',fontsize=10)
    ax.text(0,.05,'No 32/16 mode identities interpreted.\nBoth pilot-negative comparisons remain closed.',transform=ax.transAxes,fontsize=8)
    ax=axes[1,1];modes=data['models']['bolt']['low_order_modes'];order=[(r['k'],r['l']) for r in modes]
    matrix=np.asarray([[next(v['predicted_energy'] for v in data['models'][m]['low_order_modes'] if (v['k'],v['l'])==mode) for mode in order] for m in MODELS])
    im=ax.imshow(np.maximum(matrix,1e-10),aspect='auto',norm=LogNorm(vmin=1e-10,vmax=max(1.,matrix.max())),cmap='viridis')
    ax.set_xticks(range(len(order)),[f'{k},{l}' for k,l in order],rotation=90,fontsize=6);ax.set_yticks(range(3),MODELS,fontsize=8)
    ax.set_xlabel('All phase modes |k|+|l| ≤ 3, including DC',fontsize=8)
    ax.set_title('D  Full-waveform mode energies, coarse 40/20',loc='left',fontsize=10);fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Energy / shared mixture energy')
    fig.suptitle('Partial response description — no internal feature claim',fontsize=13,y=.99)
    bolt=data['models']['bolt']
    fig.text(.03,.015,'D excludes correct-character gain/rotation and DC; it is not total forecast error.\n'+
        f"Bolt 40/20 MSE {bolt['scaled_native_MSE']:.5f} includes {bolt['gain_rotation_error']:.5f} gain/rotation and {bolt['D_terms']['expected_axis_wrong_time']:.5f} wrong-character error.\n"+
        'Fixed designed amplitudes; descriptive only. Native normalization remains a possible contributor.\nOff-grid reconstruction error and native forecast loss are distinct. No further grids, orders, or hidden captures.',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.96));fig.savefig(out/'partial_response.png',dpi=160,metadata={'Software':'phase_orbit_partial_summary'})
    fig.savefig(out/'partial_response.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    rows=[]
    for m in MODELS:
        d=data['models'][m]
        rows.append(f"<tr><td>{m}</td><td>{d['D']:.6g}</td><td>{d['scaled_native_MSE']:.6g}</td><td>{d['offgrid_mean_unexplained_energy']:.6g}</td><td>{d['offgrid_pass_count']}/8</td></tr>")
    mode_rows=[]
    for m in MODELS:
        fine={(r['k'],r['l']):r for r in data['models'][m]['fine_four']['low_order_modes']}
        matched={(r['k'],r['l']):r for r in data['models'][m]['coarse_matched_four']['low_order_modes']}
        for r in data['models'][m]['low_order_modes']:
            key=(r['k'],r['l']);mode_rows.append(f"<tr><td>{m}</td><td>{key}</td><td>{r['category']}</td><td>{r['predicted_energy']:.6g}</td><td>{r['D_contribution']:.6g}</td><td>{matched[key]['D_contribution']:.6g}</td><td>{fine[key]['D_contribution']:.6g}</td></tr>")
    html='''<!doctype html><meta charset="utf-8"><title>Partial phase-orbit response</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}img{width:100%}.warning{background:#fff1d6;padding:14px}</style><h1>Partial phase-orbit response</h1>'''
    html+=f"<p class='warning'>{escape(data['scope'])}</p><p>{escape(data['normalization_caveat'])}</p><img src='partial_response.png' alt='Disjoint D terms, offgrid coverage, full status matrix and all loworder phase mode energies'><p>{escape(data['denominator'])}</p>"
    html+='<table><tr><th>Model</th><th>D</th><th>Native MSE</th><th>Offgrid unexplained energy</th><th>Coverage</th></tr>'+''.join(rows)+'</table>'
    html+=f"<p>D excludes correct-character gain/rotation and DC, so 90.95% of D is not 90.95% of total forecast error. Bolt 40/20 total normalized MSE is {bolt['scaled_native_MSE']:.5f}, including {bolt['gain_rotation_error']:.5f} gain/rotation error and {bolt['D_terms']['expected_axis_wrong_time']:.5f} wrong-character error.</p><p>{escape(data['uncovered_caveat'])}</p><h2>Complete native forecast error budget</h2><p>All phase orders are retained here. These four disjoint error groups sum to native true-future MSE; off-grid reconstruction error is separate.</p><table><tr><th>Model</th><th>Expected-axis shape/gain error</th><th>Higher-axis error</th><th>Joint error</th><th>Phase-DC error</th></tr>"
    for m in MODELS:
        b=data['models'][m]['full_error_budget']
        html+='<tr><td>'+m+'</td>'+''.join(f'<td>{b[k]:.6g}</td>' for k in ('expected_axis_shape','higher_axis','joint','phase_DC'))+'</tr>'
    html+='</table><h2>Disjoint D terms: matched coarse/fine confirmation</h2><table><tr><th>Model</th><th>Term</th><th>Coarse, all 8</th><th>Coarse, matched 4</th><th>Fine, same 4</th></tr>'
    for m in MODELS:
        d=data['models'][m]
        for term in TERMS:
            html+=f"<tr><td>{m}</td><td>{term}</td><td>{d['D_terms'][term]:.6g}</td><td>{d['coarse_matched_four']['D_terms'][term]:.6g}</td><td>{d['fine_four']['D_terms'][term]:.6g}</td></tr>"
    html+='</table><h2>All low-order modes, 40/20 only</h2><table><tr><th>Model</th><th>Mode</th><th>Group</th><th>Predicted energy, coarse8 bundles</th><th>D contribution, coarse8</th><th>D, coarse matched4</th><th>D, fine4</th></tr>'+''.join(mode_rows)+'</table>'
    html+="<details><summary>Source hashes and scientific provenance</summary><pre>"+escape(json.dumps({k:v for k,v in data.items() if k in ('source_paths','source_hashes','forecast_bindings')},indent=2))+"</pre></details><p><a href='source_data.json'>Portable source data</a> · <a href='partial_response.pdf'>PDF</a></p>"
    (out/'index.html').write_text(html,encoding='utf-8')
    (out/'render_provenance.json').write_text(json.dumps(dict(renderer_sha256=native.digest(__file__),
        scientific_source_data_sha256=native.digest(out/'source_data.json'),
        scope='Rendering caption update only; scientific source data and its original computation hash retained'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source',type=Path);group.add_argument('--render-json',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    data=collect(args.source) if args.source else json.loads(args.render_json.read_text(encoding='utf-8'))
    if args.out.exists():raise ValueError('Use a fresh summary destination')
    args.out.mkdir(parents=True);(args.out/'source_data.json').write_text(json.dumps(data,indent=2),encoding='utf-8');render(data,args.out)
