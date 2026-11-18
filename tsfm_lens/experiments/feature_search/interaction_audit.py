"""Rotation-invariant factorial interaction audit of existing four-cell stores.

Balanced ±1 ANOVA separates main and interaction components without learning
SAEs or directions. Hidden interaction is geometric; its association with
forecast errors is correlational. An invertible nonlinear encoding can have
large useful interaction while producing an exactly additive correct output.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def factorial_components(cells):
    """Return orthogonal main/interacting coefficients for order00,01,10,11."""
    cells=np.asarray(cells,dtype=np.float64)
    if cells.ndim<3 or cells.shape[1]!=4 or not np.isfinite(cells).all():
        raise ValueError('Need finite [bundle,4,...] cells ordered00,01,10,11')
    zero,second,first,both=[cells[:,i] for i in range(4)]
    return dict(mean=(zero+second+first+both)/4,
                main_phi=(first+both-zero-second)/4,
                main_psi=(second+both-zero-first)/4,
                interaction=(both-first-second+zero)/4)


def factorial_energy(cells):
    """Measure squared Frobenius energy per element, invariant to channel rotation."""
    parts=factorial_components(cells);axes=tuple(range(1,parts['interaction'].ndim))
    energy={key:np.mean(value**2,axis=axes) for key,value in parts.items() if key!='mean'}
    main=energy['main_phi']+energy['main_psi'];interaction=energy['interaction'];total=main+interaction
    ratio=np.divide(interaction,main,out=np.full_like(main,np.nan),where=main>0)
    share=np.divide(interaction,total,out=np.full_like(main,np.nan),where=total>0)
    return dict(**energy,main_total=main,interaction_over_main=ratio,interaction_share=share)


def corner_edit(cells,dose=.25):
    """Edit every corner independently against the original other three corners.

    Dose1 is literal additive-corner replacement, subtracting four times its
    ANOVA interaction component. Dose.25 removes exactly the ANOVA component.
    Neither dose is assumed to preserve a nonlinear decoder's useful code.
    """
    original=np.asarray(cells,dtype=float);component=factorial_components(original)['interaction']
    signs=np.array([1,-1,-1,1]).reshape((1,4)+(1,)*(original.ndim-2))
    return original-4*dose*signs*component[:,None]


def pack_cells(values,metadata,split='test'):
    """Group exact four-cell generating bundles and reject incomplete/duplicate keys."""
    seeds=sorted({m['seed'] for m in metadata if m['split']==split});lookup={}
    for i,m in enumerate(metadata):
        if m['split']!=split:continue
        key=(m['seed'],m['cell_phi'],m['cell_psi'])
        if key in lookup:raise ValueError(f'duplicate factorial cell:{key}')
        lookup[key]=i
    if not seeds:raise ValueError('No requested generating bundles')
    rows=[]
    for seed in seeds:
        keys=[(seed,a,b) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
        if any(key not in lookup for key in keys):raise ValueError(f'incomplete factorial bundle:{seed}')
        rows.append([lookup[key] for key in keys])
    return np.asarray(values)[np.array(rows)],np.array(seeds),np.array(rows)


def summarize(values):
    """Bootstrap generating-bundle means; expose undefined denominators explicitly."""
    values=np.asarray(values,dtype=float);finite=values[np.isfinite(values)]
    if not len(finite):return dict(mean=None,lo=None,hi=None,n=len(values),undefined=len(values))
    rng=np.random.default_rng(711);means=finite[rng.integers(0,len(finite),(2000,len(finite)))].mean(1)
    return dict(mean=float(finite.mean()),lo=float(np.quantile(means,.025)),hi=float(np.quantile(means,.975)),n=len(values),undefined=int(len(values)-len(finite)))


def association(a,b):
    """Describe a seed-level association without interpreting it as causal blame."""
    keep=np.isfinite(a)&np.isfinite(b)
    if keep.sum()<3 or np.ptp(a[keep])==0 or np.ptp(b[keep])==0:return dict(rho=None,p=None,reason='constant or insufficient finite seeds')
    rho,p=spearmanr(a[keep],b[keep]);return dict(rho=float(rho),p_unadjusted=float(p),n=int(keep.sum()),evidence='descriptive association;not causal error attribution')


def audit(source,out,split='test'):
    """Audit existing arrays only; never load models, weights or external corpora."""
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'));corpus=np.load(source/'corpus.npz')
    context,seeds,rows=pack_cells(corpus['context'],meta,split);truth,_,_=pack_cells(corpus['clean_future'],meta,split)
    scale=np.array([meta[i[0]]['scale']**2 for i in rows]);saved=dict(seeds=seeds)
    report=dict(protocol=dict(split=split,n_seeds=len(seeds),basis='balanced±1 factorialANOVA',coefficients='main contrasts/4;interaction Δ12/4',
                              normalization='mean square over all tokens/channels;interaction/(sum main energies)',
                              inference='seed bootstrap2000;Spearman descriptive only',causal_warning='Interaction may be useful nonlinear encoding;no causal blame'),
                controls=dict(input_interaction_max_abs=float(abs(factorial_components(context)['interaction']).max()),
                              truth_interaction_max_abs=float(abs(factorial_components(truth)['interaction']).max())),models={})
    for name in ['bolt','timesfm']:
        store=np.load(source/f'{name}_activations.npz');prediction,_,_=pack_cells(store['prediction'],meta,split)
        output=factorial_energy(prediction);error=np.mean((prediction-truth)**2,axis=(1,2))/scale
        output_scaled=output['interaction']/scale
        model=dict(output={key:summarize(value) for key,value in output.items()},
                   output_interaction_over_latent_variance=summarize(output_scaled),scaled_forecast_mse=summarize(error),
                   output_interaction_error_association=association(output_scaled,error),layers={})
        saved[name+'_forecast_error']=error;saved[name+'_output_interaction_scaled']=output_scaled
        for layer in [k for k in store.files if k!='prediction' and not k.startswith('raw__')]:
            cells,_,_=pack_cells(store[layer],meta,split);surfaces=dict(context=cells)
            raw,_,_=pack_cells(store['raw__'+layer],meta,split)
            if raw.shape[2]>cells.shape[2]:surfaces['special_tokens']=raw[:,:,cells.shape[2]:]
            for surface,c in surfaces.items():
                energy=factorial_energy(c);key=layer+'_'+surface
                model['layers'][key]=dict(shape=list(c.shape),energy={k:summarize(v) for k,v in energy.items()},
                    hidden_share_output_error_association=association(energy['interaction_share'],error),
                    hidden_share_output_interaction_association=association(energy['interaction_share'],output_scaled))
                saved[name+'_'+key+'_interaction_share']=energy['interaction_share']
                saved[name+'_'+key+'_interaction_over_main']=energy['interaction_over_main']
        report['models'][name]=model
    np.savez_compressed(out/'per_seed.npz',**saved);(out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


def ratio_of_means(numerator,denominator):
    """Bootstrap a ratio of seed means with paired resampling of both quantities."""
    rng=np.random.default_rng(711);indices=rng.integers(0,len(numerator),(2000,len(numerator)))
    ratios=numerator[indices].mean(1)/denominator[indices].mean(1)
    return dict(value=float(np.mean(numerator)/np.mean(denominator)),lo=float(np.quantile(ratios,.025)),hi=float(np.quantile(ratios,.975)))


def main_effect_frequency_audit(source,out,split='test'):
    """Separate additive wrong-frequency mappings from factorial cross-terms.

    A phi main effect can alter the period16 output while remaining strictly
    additive across the two factors. Such cross-frequency use is absent from
    the factorial interaction coefficient, so both diagnostics are needed.
    """
    from mixture_features import harmonic_fit
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'));truth=np.load(source/'corpus.npz')['clean_future'];report={};saved={}
    for name in ['bolt','timesfm']:
        prediction=np.load(source/f'{name}_activations.npz')['prediction'];cells,seeds,rows=pack_cells(prediction,meta,split);target,_,_=pack_cells(truth,meta,split);parts=factorial_components(cells)
        scale=np.array([meta[r[0]]['scale']**2 for r in rows]);error=np.mean((cells-target)**2,axis=(1,2))/scale
        interaction=np.mean(parts['interaction']**2,axis=1)/scale
        model=dict(interaction_fraction_of_total_mse=ratio_of_means(interaction,error),main_effects={})
        for main,targetamp,wrongamp in [('main_phi','a','b'),('main_psi','b','a')]:
            component=parts[main];fit=harmonic_fit(component);total=np.mean(component**2,axis=1);right=.5*fit[targetamp]**2;wrong=.5*fit[wrongamp]**2
            reconstructed=(fit['coefficients']@np.column_stack([np.sin(2*np.pi*np.arange(128,160)/32),np.cos(2*np.pi*np.arange(128,160)/32),np.sin(2*np.pi*np.arange(128,160)/16),np.cos(2*np.pi*np.arange(128,160)/16),np.ones(32)]).T)
            residual=np.mean((component-reconstructed)**2,axis=1)
            model['main_effects'][main]=dict(wrong_frequency_energy_over_latent_variance=summarize(wrong/scale),right_frequency_energy_over_latent_variance=summarize(right/scale),
                wrong_frequency_fraction_of_main_effect=ratio_of_means(wrong,total),wrong_frequency_fraction_of_total_mse=ratio_of_means(wrong/scale,error),
                offset_energy_over_latent_variance=summarize(fit['offset']**2/scale),nonharmonic_energy_over_latent_variance=summarize(residual/scale))
            saved[name+'_'+main+'_wrong_energy_scaled']=wrong/scale
        report[name]=model;saved[name+'_error']=error;saved[name+'_interaction_energy_scaled']=interaction
    report['evidence']='descriptive input/output factorial decomposition;wrong-frequency main mappings differ from nonlinear cross-term;no internal causal claim'
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez_compressed(out/'per_seed.npz',seeds=seeds,**saved);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',default='test');p.add_argument('--main-effects',action='store_true');args=p.parse_args();(main_effect_frequency_audit if args.main_effects else audit)(args.source,args.out,args.split)
