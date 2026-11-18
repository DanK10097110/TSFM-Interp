"""Task-defined independent phase orbits; no hidden features or learned probes.

Temporal z=c_sin+i*c_cos has desired POSITIVE phase mode only. Discrete
Fourier modes are grid equivalence classes until nested/offgrid validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PAIRS = ((32,16),(40,20))
RATIOS = (.5,.8,1.25,2.)
SCALES = (.8,1.4)
REFINEMENT_BUNDLES = (0,3,4,7)


def make_orbit(periods, grid=8, bundles=tuple(range(8)), offgrid=False):
    """Fixed ratio×scale bundles; grid origins/offsets independent of outcomes."""
    if tuple(periods) not in PAIRS or grid not in (8,16):
        raise ValueError('Only frozen pairs and grids are supported')
    contexts, futures, rows = [], [], []
    t = np.arange(-128,64)
    for bundle in bundles:
        if bundle not in range(8):
            raise ValueError('Unknown amplitude bundle')
        rng = np.random.default_rng(3500000+bundle)
        ratio, scale = RATIOS[bundle//2], SCALES[bundle%2]
        b = scale*np.sqrt(2/(ratio*ratio+1)); a = ratio*b
        offset = rng.uniform(-2,2); origins = rng.uniform(-np.pi,np.pi,2)
        phases = rng.uniform(-np.pi,np.pi,(16,2)) if offgrid else np.asarray(
            [(origins[0]+2*np.pi*i/grid,origins[1]+2*np.pi*j/grid)
             for i in range(grid) for j in range(grid)])
        for index,(phi,psi) in enumerate(phases):
            signal = offset+a*np.sin(2*np.pi*t/periods[0]+phi)+b*np.sin(2*np.pi*t/periods[1]+psi)
            contexts.append(signal[:128]); futures.append(signal[128:])
            rows.append(dict(bundle=bundle,generator_seed=3500000+bundle,grid=grid,
                cell=index,phi=phi,psi=psi,origin_phi=origins[0],origin_psi=origins[1],
                a=a,b=b,scale=scale,ratio=ratio,offset=offset,offgrid=offgrid))
    return np.asarray(contexts,np.float32),np.asarray(futures),rows


def temporal_components(values,periods):
    """Joint full-Gram fit at future t=0..H−1; preserve residual and offset."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Need finite row×future arrays')
    t = np.arange(values.shape[1])
    design = np.column_stack([np.ones(len(t))]+[fn(2*np.pi*t/p) for p in periods for fn in (np.sin,np.cos)])
    if np.linalg.matrix_rank(design)!=5 or np.linalg.cond(design)>100:
        raise ValueError('Unstable temporal basis')
    coefficient = np.linalg.lstsq(design,values.T,rcond=None)[0].T
    z = np.stack([coefficient[:,1]+1j*coefficient[:,2],coefficient[:,3]+1j*coefficient[:,4]],axis=1)
    residual = values-coefficient@design.T
    return dict(z=z,offset=coefficient[:,0],residual=residual,design=design,
                gram=design.T@design/len(t),coefficient=coefficient)


def orbit_fft(values,origins):
    """Dephase grid origins so coefficients share global physical phase labels."""
    values = np.asarray(values)
    n = values.shape[0]
    if values.ndim<2 or values.shape[1]!=n or n not in (8,16):
        raise ValueError('Need a complete8×8 or16×16 orbit')
    modes = np.fft.fftfreq(n,d=1/n).astype(int)
    result = np.fft.fft2(values,axes=(0,1))/(n*n)
    dephase = np.exp(-1j*(modes[:,None]*origins[0]+modes[None,:]*origins[1]))
    result *= dephase.reshape((n,n)+(1,)*(values.ndim-2))
    return result,modes


def mode_class(k,l,feature):
    """Semantic identity and physical-time equivariance are distinct labels."""
    if feature not in ('phi','psi'):
        raise ValueError('Unknown target frequency')
    desired = (1,0) if feature=='phi' else (0,1)
    if (k,l)==desired:
        semantic = 'correct_feature'
    elif k==0 and l==0:
        semantic = 'constant'
    elif k!=0 and l!=0:
        semantic = 'genuine_joint'
    else:
        semantic = 'wrong_axis_or_harmonic'
    return dict(semantic=semantic,degree=abs(k)+abs(l),
        equivariance='compatible' if k+2*l==(1 if feature=='phi' else 2) else 'violating',
        equivariance_scope='target-coefficient compatibility conditional on adequate temporal span; omitted harmonics can project into non-tiling fits; not full-waveform non-equivariance',
        correct_conjugate=(k,l)==(-desired[0],-desired[1]))


def waveform_mode_compatibility(value,k,l,base_period):
    """Full phase-mode waveform check, immune to omitted temporal basis columns."""
    value=np.asarray(value,dtype=complex)
    harmonic=np.exp(2j*np.pi*(k+2*l)*np.arange(len(value))/base_period)
    fitted=harmonic*np.vdot(harmonic,value)/np.vdot(harmonic,harmonic)
    energy=float(np.vdot(value,value).real)
    return 0. if energy==0 else float(np.vdot(value-fitted,value-fitted).real/energy)


def waveform_budget(prediction,truth,origins,scale,base_period):
    """PRIMARY exact phase×time error budget, without temporal projection."""
    prediction,truth=np.asarray(prediction),np.asarray(truth)
    if prediction.shape!=truth.shape or prediction.ndim!=3:
        raise ValueError('Need matching complete phasegrid×phasegrid×future waveforms')
    f,modes=orbit_fft(prediction,origins);g,_=orbit_fft(truth,origins);error=f-g
    categories=dict(expected_axis_shape=0.,higher_axis=0.,joint=0.,phase_DC=0.)
    entries=[];nonexpected=0.;axis_incompatible=0.
    for i,k in enumerate(modes):
        for j,l in enumerate(modes):
            expected=(abs(k)==1 and l==0) or (k==0 and abs(l)==1)
            category=('phase_DC' if k==l==0 else 'expected_axis_shape' if expected
                      else 'joint' if k!=0 and l!=0 else 'higher_axis')
            energy=float(np.mean(abs(error[i,j])**2)/scale**2)
            categories[category]+=energy
            if not expected and (k!=0 or l!=0) and abs(k)+abs(l)<=3:
                nonexpected+=float(np.mean(abs(f[i,j])**2)/scale**2)
            predicted_energy=float(np.mean(abs(f[i,j])**2)/scale**2)
            compatibility=waveform_mode_compatibility(f[i,j],int(k),int(l),base_period)
            if expected:axis_incompatible+=predicted_energy*compatibility
            entries.append(dict(k=int(k),l=int(l),category=category,degree=int(abs(k)+abs(l)),
                error_energy=energy,predicted_energy=predicted_energy,
                physical_time_character_residual_fraction=None if predicted_energy<1e-12 else compatibility,
                physical_time_character_status='negligible energy;relative fraction undefined' if predicted_energy<1e-12 else 'defined;phasealias validation pending',
                physical_time_character_residual_energy=predicted_energy*compatibility))
    mse=float(np.mean((prediction-truth)**2)/scale**2)
    return dict(scaled_MSE=mse,error_categories=categories,modes=entries,
        low_order_nonexpected_phase_energy=nonexpected,
        expected_axis_time_character_residual_energy=axis_incompatible,
        primary_D=nonexpected+axis_incompatible,
        expected_axis_gain_rotation_error=categories['expected_axis_shape']-axis_incompatible,
        error_Parseval_absolute_error=float(abs(sum(categories.values())-mse)),
        scope='Physical time-character compatibility,not architectural translation equivariance;phase aliases provisional until validation;expected±indices doNOT imply correctness;D uses disjointphasegroups,no doublecount;puregain/rotationerror separate')


def waveform_stability(coarse,fine,origins,offgrid_values,phases,scale):
    """PRIMARY nested-grid stability and coarse-only full-waveform coverage."""
    c8,m8=orbit_fft(coarse,origins);c16,m16=orbit_fft(fine,origins)
    errors=[];reference=[];high=0.
    for i,k in enumerate(m16):
        for j,l in enumerate(m16):
            if abs(k)+abs(l)<=3:
                a=int(np.flatnonzero(m8==k)[0]);b=int(np.flatnonzero(m8==l)[0])
                errors.append(c8[a,b]-c16[i,j]);reference.append(c16[i,j])
            else:high+=float(np.mean(abs(c16[i,j])**2))
    error=float(np.sqrt(np.sum(np.mean(abs(np.asarray(errors))**2,axis=-1))))
    ref=float(np.sqrt(np.sum(np.mean(abs(np.asarray(reference))**2,axis=-1))))
    relative=0. if error==0 else (error/ref if ref else float('inf'))
    absolute=error/scale
    offerror=float(np.sqrt(np.mean(abs(reconstruct(c8,m8,phases)-offgrid_values)**2))/scale)
    return dict(relative_low_coefficient_error=relative,absolute_low_coefficient_error=absolute,
        stable_low_order=bool(relative<=.05 and absolute<=.02),
        beyond_order3_energy=high/scale**2,offgrid_coarse_waveform_RMS=offerror,
        adequate_waveform_coverage=bool(high/scale**2<=.05 and offerror<=.05))


def reconstruct(coefficients,modes,phases,order=3):
    """Fixed low-order reconstruction, never select order on held-out phases."""
    output = np.zeros((len(phases),)+coefficients.shape[2:],complex)
    for i,k in enumerate(modes):
        for j,l in enumerate(modes):
            if abs(k)+abs(l)<=order:
                factor = np.exp(1j*(k*phases[:,0]+l*phases[:,1]))
                output += factor.reshape((len(phases),)+(1,)*(output.ndim-1))*coefficients[i,j]
    return output


def normalized_input(context,model):
    """Native CPU preprocessing, including TimesFM PER-PATCH running states."""
    import torch
    values = torch.as_tensor(context,dtype=torch.float32)
    if model in ('bolt','chronos2'):
        from chronos.chronos_bolt import InstanceNorm
        normalized,(mu,sigma) = InstanceNorm(use_arcsinh=model=='chronos2')(values)
        return normalized.numpy(),mu.numpy(),sigma.numpy()
    if model != 'timesfm':
        raise ValueError(model)
    from timesfm.torch.util import update_running_stats,revin
    patches = values.reshape(len(values),4,32)
    n = torch.zeros(len(values)); mu = n.clone(); sigma = n.clone(); mus=[];sigmas=[]
    for patch in patches.unbind(1):
        (n,mu,sigma),_ = update_running_stats(n,mu,sigma,patch,torch.zeros_like(patch,dtype=torch.bool))
        mus.append(mu);sigmas.append(sigma)
    mu = torch.stack(mus,1);sigma=torch.stack(sigmas,1)
    return revin(patches,mu,sigma,reverse=False).reshape(len(values),128).numpy(),mu.numpy(),sigma.numpy()


def mode_summary(values,origins,amplitude,feature):
    """No sign pooling: desired +1 and incorrect conjugate remain separate."""
    coefficients,modes = orbit_fft(values,origins)
    entries=[]
    for i,k in enumerate(modes):
        for j,l in enumerate(modes):
            entries.append(dict(k=int(k),l=int(l),energy=float(abs(coefficients[i,j])**2/amplitude**2),
                                **mode_class(int(k),int(l),feature)))
    desired=(1,0) if feature=='phi' else (0,1)
    lowwrong=sum(v['energy'] for v in entries if v['degree']<=3 and v['semantic']!='correct_feature')
    index=[int(np.flatnonzero(modes==v)[0]) for v in desired]
    correct = coefficients[tuple(index)]/amplitude
    return dict(entries=entries,wrong_low_order_energy=lowwrong,
        correct_complex_gain=[float(correct.real),float(correct.imag)],
        correct_gain_error=float(abs(correct-1)**2),
        parseval_error=float(abs(np.mean(abs(values)**2)-np.sum(abs(coefficients)**2))/amplitude**2),
        caveat='Discrete grid classes alias high phase harmonics; no native spectral interpretation before refinement/offgrid gates')


def pilot_interestingness(energy_first,energy_second):
    """Deterministic8-bundle screen, not eight independent population samples."""
    first,second=np.asarray(energy_first),np.asarray(energy_second)
    if first.shape!=(8,) or second.shape!=(8,) or not np.isfinite([first,second]).all():
        raise ValueError('Need two complete fixed8-bundle low-order wrong-energy vectors')
    forward=int(np.sum((first>=.05)&(first-second>=.02)))
    reverse=int(np.sum((second>=.05)&(second-first>=.02)))
    return dict(first_larger_count=forward,second_larger_count=reverse,
                promotes_alias_validation=bool(forward>=6 or reverse>=6),
                scope='exploratory deterministic amplitude grid; no significance or population claim')


def alias_stability(coarse,fine,origins,offgrid_values,phases,amplitude):
    """Scalar complex response: stable low modes and coverage are separate."""
    c8,m8=orbit_fft(coarse,origins);c16,m16=orbit_fft(fine,origins)
    errors=[];reference=[];high=0.
    for i,k in enumerate(m16):
        for j,l in enumerate(m16):
            if abs(k)+abs(l)<=3:
                a=int(np.flatnonzero(m8==k)[0]);b=int(np.flatnonzero(m8==l)[0])
                errors.append(c8[a,b]-c16[i,j]);reference.append(c16[i,j])
            else:high+=abs(c16[i,j])**2
    error=np.linalg.norm(errors);ref=np.linalg.norm(reference)
    relative=0. if error==0 else (float(error/ref) if ref>0 else float('inf'))
    absolute=float(error/amplitude)
    offerror=float(np.sqrt(np.mean(abs(reconstruct(c8,m8,phases)-offgrid_values)**2))/amplitude)
    return dict(relative_low_coefficient_error=relative,absolute_low_coefficient_error=absolute,
        stable_low_order=bool(relative<=.05 and absolute<=.02),
        beyond_order3_energy=float(high/amplitude**2),offgrid_coarse_complex_RMS=offerror,
        adequate_complex_coverage=bool(high/amplitude**2<=.05 and offerror<=.05))


def prepare(out):
    """Prepare only; GPU inference is intentionally unavailable in this driver."""
    if out.exists():raise ValueError('Never overwrite an existing protocol')
    out.mkdir(parents=True)
    protocol=dict(seed_base=3500000,context=128,horizon=64,pairs=PAIRS,grid=8,
        amplitude_ratios=RATIOS,shared_scales=SCALES,offset=[-2,2],bundles=8,
        models=['timesfm','bolt','chronos2'],model_context_forecasts=3072,
        complex_convention='z=c_sin+i*c_cos;correct phi(+1,0),psi(0,+1);conjugate is WRONG',
        low_order='|k|+|l|<=3 fixed',equivariance='phi k+2l=1;psi k+2l=2;independent of semantic correctness',
        equivariance_scope='fitted target-coefficient label conditional on adequate temporal span; full Gram removes included-span leakage ONLY; omitted temporal harmonics can project into40/20 H64 fits, so do not infer full-waveform non-equivariance',
        omitted_harmonic_control='exp(i3φ) temporal3ω decoy must show leakage into fittedω/2ω yet pass full phase-mode waveform compatibility at3ω',
        fixed_pilot_candidates='PRIMARY all3 unordered modelpairs ×2 periodpairs=6comparisons;reportall;direction either model,but same direction in qualifyingbundles;no statistical testing on8designedbundles',
        refinement_bundles=REFINEMENT_BUNDLES,refinement_grid=16,offgrid_cells_per_bundle=16,
        preprocessing='Record native mu/sigma and normalized-input spectra, TimesFM all4 running-patch states;C2 arcsinh retained',
        scope='Native output phase response only; no hidden captures/probes/semantic interventions; established harmonic analysis, not new theory',
        controls='native repeat and row permutation; oracle temporal fit and phase modes;full Gram and Parseval closure;no output truncation to32',
        primary='2D phaseDFT ofFULL forecast waveforms;E=DFT(prediction-truth);phase/time Parseval exact errorbudget;no temporal basis assumption',
        primary_error_categories='expected axis±(1,0)/±(0,1) gain/temporalshape error;higheraxis;joint bothnonzero;phaseDC usesE00 notrawF00;expectedindexpairs doNOT imply correct temporalcharacter',
        secondary='complex z target coefficients and all targetspan compatibility labels areSECONDARY only',
        primary_D='loworder nonexpectedphase higheraxis+joint rawF energy(excludeDC) PLUS expected±axis residualenergy after Hermitian projection onto correct exp(iω(k+2l)t);disjointgroups;puregain/rotationerror separate',
        pilot_interestingness='one fixed modelpair/period direction:D/sharedmixenergy>=.05 AND same-direction crossmodelgap>=.02 inSAME>=6/8 fixedbundles;exploratory deterministicgrid,not population samples',
        physical_character_scope='Physical time-character compatibility of forecastwaveforms,NOT architecturaltranslationequivariance;even persistencefails;finitephasealiaslabels provisionaluntilvalidation;not positionalencodingcause',
        relative_character_reporting_floor='predictedmodeenergy/sharedmixenergy<1e-12 => relativefractionundefined;absoluteenergy always retained;no seeds/modes excluded from D',
        promotion_scope='pilot promotes only mandatory alias validation;8grid alone cannot support native mode interpretation',
        stable_low_modes='PRIMARY fullwaveform8vs16 loworder relativeL2<=.05 AND absoluteL2/sharedscale<=.02 onALL4 fixed refinementbundles;timeaveragedFrobenius overmodes',
        whole_response_coverage='PRIMARY beyondorder3 fullwaveformenergy<=.05 sharedmixenergy onALL4 refinementbundles;offgrid fullwaveformRMS/sharedscale<=.05 in>=7/8 fixedbundles;zcoefficientchecksSECONDARY',
        offgrid_resolution='COARSE8grid loworder reconstruction forALL8bundles;no fine/coarse selection;full raw waveform reconstructed separately retaining offset and temporal residual',
        normalized_context_profiles='phase DFT of physical128-position vector;no future-origin temporal fit applied to context',
        validation_statuses='low-order instability=>unresolved/aliased and stop;stableloworder but failedcoverage=>reportpartialpattern+unmodeledenergy only,no hiddencaptures;bothpass=>separate approval required for captures',
        no_adaptation='no order/grid/amplitude/phase/condition changes after outcomes',
        inference_boundary='This prepare-only math driver cannot load a model or launch inference;any future runner requires separate authorization and prospective source/control binding',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    bindings={}
    for pair in PAIRS:
        x,y,rows=make_orbit(pair)
        stem=f'p{pair[0]}_{pair[1]}'
        np.savez_compressed(out/f'{stem}.npz',context=x,clean_future=y)
        (out/f'{stem}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        bindings[stem]={suffix:hashlib.sha256((out/f'{stem}.{suffix}').read_bytes()).hexdigest() for suffix in ('npz','json')}
    protocol['bindings']=bindings
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path,required=True)
    prepare(parser.parse_args().prepare)
