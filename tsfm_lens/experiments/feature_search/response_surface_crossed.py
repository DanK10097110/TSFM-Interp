"""Prepare-only fresh32-source crossed-phase, measured-radius response study.

This module intentionally has no inference command. It freezes inputs and the
capture contract while natural-data and comparability studies own native slots.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from experiments.feature_search.coherence_study import read, write, sha

CENTERS = (128/24, 128/19.2)
PHASES = tuple(float(p) for p in np.arange(4)*np.pi/2)
RADII = (.002,.005,.01)
AMPLITUDES = (.1,.5,2.)


def wave(frequency, phase):
    """Unit-peak analytic tone, with phase at forecast origin and no grid snapping."""
    return np.sin(2*np.pi*frequency*np.arange(-128,0)/128+phase)


def rms(value):
    """FP64 root mean square for measured input-distance gates."""
    return float(np.sqrt(np.mean(np.asarray(value,float)**2)))


def local_triplet(background, frequency, phase, amplitude, radius, axis):
    """Choose the parameter step to achieve the requested RMS chord half-distance."""
    x=np.asarray(background,float);scale=float(x.std());unit=wave(frequency,phase)
    zero=x+amplitude*scale*unit
    if axis=='frequency':
        def distance(step):
            return amplitude*rms((wave(frequency+step,phase)-wave(frequency-step,phase))/2)
        step=brentq(lambda d:distance(d)-radius,0.,.25,xtol=1e-14)
        minus=x+amplitude*scale*wave(frequency-step,phase)
        plus=x+amplitude*scale*wave(frequency+step,phase)
    elif axis=='amplitude':
        step=radius/rms(unit)
        if step>=amplitude:raise ValueError('Amplitude path crosses zero')
        minus=x+(amplitude-step)*scale*unit
        plus=x+(amplitude+step)*scale*unit
    else:raise ValueError('Unknown axis')
    minus,zero,plus=[v.astype(np.float32) for v in (minus,zero,plus)]
    midpoint=((minus.astype(float)+plus.astype(float))/2).astype(np.float32)
    actual=rms((plus.astype(float)-minus.astype(float))/2)/scale
    if abs(actual-radius)>1e-7:raise ValueError('Actual RMS radius gate failed after FP32 quantization')
    return minus,zero,plus,midpoint,dict(step=float(step),radius=radius,actual_radius=actual,
        radius_error=abs(actual-radius),input_scale=scale,
        input_left=rms(zero.astype(float)-minus)/scale,input_right=rms(plus.astype(float)-zero)/scale,
        chord_input_rounding=rms(plus.astype(float)+minus-2*midpoint.astype(float))/scale)


def background(seed):
    """Eight fresh seeds per mixture type; injection phase grid is independent of draws."""
    j=seed-4954000
    if not 0<=j<32:raise ValueError('Reserved seed range4954000..4954031 only')
    f,amplitude=[(17.35,.5),(17.35,8.),(6.35,.5),(6.35,8.)][j%4]
    rng=np.random.default_rng(seed);phi,psi=rng.uniform(-np.pi,np.pi,2)
    raw=.5*wave(5.35,phi)+amplitude*wave(f,psi)+rng.normal(0,.05,128)
    x=((raw-raw.mean())/raw.std()).astype(np.float32)
    return dict(source=f'syn:{seed}',seed=seed,domain='synthetic',background_type=j%4,
                target_phase=float(phi),distractor_phase=float(psi),distractor_frequency=f,
                distractor_amplitude=amplitude),x


def make_design(seeds):
    """Cross every background with all phases, both centers, amplitudes and RMS radii."""
    inputs=[];nodes=[];paths=[];pairs=[];sources=[]
    for source_id,seed in enumerate(seeds):
        meta,x=background(seed);sources.append(meta);scale=float(x.astype(float).std());lookup={}
        def add(value,kind,**details):
            """Deduplicate exact FP32 contexts only within the background source."""
            value=np.asarray(value,np.float32);key=value.tobytes()
            if key not in lookup:
                lookup[key]=len(inputs);inputs.append(value)
                nodes.append(dict(index=len(inputs)-1,source_id=source_id,source=meta['source'],kind=kind,**details))
            return lookup[key]
        root=add(x,'background')
        for phase_id,phase in enumerate(PHASES):
            for frequency in CENTERS:
                for amplitude in AMPLITUDES:
                    for axis in ('frequency','amplitude'):
                        for radius in RADII:
                            minus,zero,plus,midpoint,details=local_triplet(x,frequency,phase,amplitude,radius,axis)
                            ids=[add(v,kind,root=root) for v,kind in zip((minus,zero,plus,midpoint),('minus','center','plus','midpoint'))]
                            paths.append(dict(source=meta['source'],source_id=source_id,base=root,
                                phase_id=phase_id,phase=phase,frequency=frequency,amplitude=amplitude,axis=axis,
                                minus=ids[0],zero=ids[1],plus=ids[2],midpoint=ids[3],**details))
        for phase_a,phi in enumerate(PHASES):
            for phase_b,psi in enumerate(PHASES):
                ua,ub=wave(CENTERS[0],phi),wave(CENTERS[1],psi)
                for radius in RADII:
                    a=ua/rms(ua)*radius*scale;b=ub/rms(ub)*radius*scale
                    ia=add(x.astype(float)+a,'interaction_a',root=root)
                    ib=add(x.astype(float)+b,'interaction_b',root=root)
                    iab=add(x.astype(float)+a+b,'interaction_ab',root=root)
                    actual_a=rms(inputs[ia].astype(float)-x)/scale;actual_b=rms(inputs[ib].astype(float)-x)/scale
                    if max(abs(actual_a-radius),abs(actual_b-radius))>1e-7:raise ValueError('Interaction radius gate')
                    closure=rms(inputs[iab].astype(float)-inputs[ia]-inputs[ib]+x)/scale
                    pairs.append(dict(source=meta['source'],source_id=source_id,base=root,a=ia,b=ib,ab=iab,
                        phase_a=phase_a,phase_b=phase_b,radius=radius,input_a=actual_a,input_b=actual_b,
                        input_scale=scale,input_closure_rms=closure))
    return np.stack(inputs),nodes,paths,pairs,sources


def prepare(out,prior):
    """Freeze design and hashes without taking or waiting for a native slot."""
    inputs,nodes,paths,pairs,sources=make_design(range(4954000,4954032))
    out.mkdir(parents=True,exist_ok=False)
    np.save(out/'inputs.npy',inputs)
    for name,value in [('nodes',nodes),('paths',paths),('interactions',pairs),('sources',sources)]:write(out/f'{name}.json',value)
    repo=Path(__file__).resolve().parents[2]
    source_files=[Path(__file__),repo/'tests/test_response_surface_crossed.py',
                  repo/'experiments/feature_search/selective_study.py',repo/'experiments/feature_search/coherence_study.py',
                  repo/'experiments/feature_search/real_background_poc.py',repo/'experiments/feature_search/homogeneous_normalizer.py',
                  repo/'experiments/feature_search/response_surface_metrics.py']
    protocol=dict(stage='prepared_capture_held',question='Does response curvature converge at matched small input radii after crossing phase with context, and do local two-tone interactions persist?',
        source_count=32,seeds=list(range(4954000,4954032)),phases=list(PHASES),centers=list(CENTERS),amplitudes=list(AMPLITUDES),radii=list(RADII),
        nodes=len(nodes),local_paths=len(paths),interaction_quartets=len(pairs),context=128,horizon=32,
        design='32 independent synthetic backgrounds,8 per mixture type; every source crosses4 injection phases,2 frequency centers,3 amplitudes,2 local axes and3 measured RMS radii. Interaction crosses all4x4 phase pairs at each radius.',
        phase='Injection grid0/pi/2/pi/3pi/2 is common across backgrounds and independent of random background phases; source grouping includes every phase/path.',
        frequency_steps='Continuous root solve for RMS half-chord radius divided by fixed background SD; FP32 quantized input distance gate<=1e-7 absolute.',
        amplitude_steps='Peak amplitude increment chosen for same actual RMS radius; positive and negative endpoints preserve the same128-step background.',
        interaction='Four-point response on background x with two RMS-matched additions at24h/19.2h; report absolute mixed effect, each marginal denominator, sum denominator and input-distance product. Saved input-closure residual calibrates FP32 arithmetic contamination.',
        normalization='Matched native and actual background-locked normalizer/inverse on exactly the same saved inputs. Save full-token first/quarter/half hidden states and32-step forecasts.',
        gates='Exact repeats, background native/locked identity for both forecasts and hidden states, exact self-patch; finite arrays, source/checkpoint hashes, FP32 input radius/closure gates; no output-dependent source replacement.',
        primary='Per-source averages over dependent phase/amplitude cells: chord-relative curvature shrinkage at.01/.005/.002 RMS; input-normalized sensitivity convergence; native-versus-locked interaction and known-signal response. Preserve worst-source and phase curves.',
        inference='Exploratory fresh synthetic expansion, not sealed confirmation; resampling unit source seed, never phases or rows. No real-source generalization or model-training attribution.',
        analysis='Reuse FP64 response_surface_metrics arithmetic after capture; raw forecasts/states remain the authority. Persist knowledge.json after each execution including failures.',
        capture_hold=True,capture_status='NOT STARTED. Native slots reserved for natural-data and comparability. This driver is prepare-only.',
        prospective_execution='After explicit release: serial bounded shards of4 background sources;1CPU thread,3GiB and600 seconds per model/shard. Freeze capture-driver source in an addendum before its first launch; do not mutate this protocol.',
        expected_shards=[dict(shard=j,source_ids=list(range(4*j,4*j+4))) for j in range(8)],
        checkpoints=read(prior/'protocol.json')['checkpoints'],source_hashes={str(f.resolve()):sha(f) for f in source_files},
        hashes={name:sha(out/name) for name in ['inputs.npy','nodes.json','paths.json','interactions.json','sources.json']})
    write(out/'protocol.json',protocol)
    gates=dict(actual_local_radius_error_max=max(r['radius_error'] for r in paths),
               chord_rounding_max=max(r['chord_input_rounding'] for r in paths),
               interaction_closure_max=max(r['input_closure_rms'] for r in pairs),
               actual_interaction_radius_error_max=max(abs(r[k]-r['radius']) for r in pairs for k in ('input_a','input_b')))
    write(out/'knowledge.json',dict(status='prepared_native_capture_held',question=protocol['question'],design=protocol,
        exact_findings=dict(preparation_gates=gates,counts=dict(backgrounds=32,phases=4,nodes=len(nodes),paths=len(paths),interaction_quartets=len(pairs)),model_findings=None),
        controls=protocol['gates'],limitations=protocol['inference'],evidence_class='Frozen prospective design only; no model results',
        failed_instrumentation_runs=[],next_decision='HOLD native capture until coordinator releases a slot. No model process launched.'))
    print(dict(run=str(out.resolve()),nodes=len(nodes),paths=len(paths),interactions=len(pairs),capture='HELD',gates=gates),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior',type=Path,default=Path('runs/response_surface_pilot4951000'))
    args=parser.parse_args();prepare(args.out,args.prior)
