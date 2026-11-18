"""Prospective alias validation with immutable coarse caches and amended controls.

Original pilot control failures remain failures. This is a separately registered
follow-up using FP64 measurement and fixed-B8 independence controls. No hidden
captures, new amplitude choices, mode-order tuning, or native-ranking rescue.
"""
from __future__ import annotations
import argparse
import gc
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
from experiments.feature_search import phase_orbit as orbit
from experiments.feature_search import phase_orbit_native as native
from experiments.feature_search import phase_orbit_fp64 as fp64


def requests(pair):
    """Exactly768 new fine cells plus128 offgrid cells per period pair."""
    fine_x,fine_y,fine_rows=orbit.make_orbit(pair,grid=16,bundles=orbit.REFINEMENT_BUNDLES)
    take=[i for i,row in enumerate(fine_rows) if not (row['cell']//16%2==0 and row['cell']%16%2==0)]
    off_x,off_y,off_rows=orbit.make_orbit(pair,offgrid=True)
    rows=[]
    for i in take:rows.append(dict(fine_rows[i],request_kind='fine_new'))
    rows += [dict(row,request_kind='offgrid') for row in off_rows]
    x=np.concatenate([fine_x[take],off_x]);y=np.concatenate([fine_y[take],off_y])
    if x.shape!=(896,128) or y.shape!=(896,64):raise ValueError('Frozen request count changed')
    return x,y,rows


def ids(rows,stem):
    return np.asarray([f"{stem}:{r['generator_seed']}:{r['request_kind']}:{r['bundle']}:{r['grid']}:{r['cell']}" for r in rows])


def assess_case(coarse,new_prediction,new_rows,coarse_rows,pair):
    """ALL4 fine stability/energy checks; separate ALL8 coarse offgrid checks."""
    coarse=np.asarray(coarse,dtype=np.float64);new_prediction=np.asarray(new_prediction,dtype=np.float64)
    if coarse.shape!=(512,64) or new_prediction.shape!=(896,64):raise ValueError('Exact coarse/new forecast shapes required')
    lookup={(r['request_kind'],r['bundle'],r['cell']):i for i,r in enumerate(new_rows)}
    if len(lookup)!=896:raise ValueError('New phase-row identity duplicated')
    fine_records={};offgrid_records={}
    for bundle in range(8):
        row=coarse_rows[bundle*64];origins=(row['origin_phi'],row['origin_psi']);scale=row['scale']
        coarse_wave=coarse[bundle*64:(bundle+1)*64].reshape(8,8,64)
        off_indices=[lookup['offgrid',bundle,cell] for cell in range(16)]
        off_wave=new_prediction[off_indices]
        phases=np.asarray([[new_rows[index]['phi'],new_rows[index]['psi']] for index in off_indices])
        coefficient,modes=orbit.orbit_fft(coarse_wave,origins)
        predicted=orbit.reconstruct(coefficient,modes,phases)
        off_error=float(np.sqrt(np.mean(abs(predicted-off_wave)**2))/scale)
        # Complex target fits are secondary; full waveforms drive every gate.
        z8=orbit.temporal_components(coarse_wave.reshape(64,64),pair)['z']
        zoff=orbit.temporal_components(off_wave,pair)['z']
        secondary={}
        for feature,amplitude in enumerate((row['a'],row['b'])):
            zcoef,zmodes=orbit.orbit_fft(z8[:,feature].reshape(8,8),origins)
            secondary[('phi','psi')[feature]]=float(np.sqrt(np.mean(abs(orbit.reconstruct(zcoef,zmodes,phases)-zoff[:,feature])**2))/amplitude)
        offgrid_records[str(bundle)]=dict(waveform_RMS_over_shared_scale=off_error,passed=bool(off_error<=.05),
            secondary_complex_coefficient_RMS=secondary,offgrid_cells=16,reconstruction='coarse8 only')
        if bundle in orbit.REFINEMENT_BUNDLES:
            fine=np.empty((16,16,64),dtype=np.float64)
            for i in range(16):
                for j in range(16):
                    fine[i,j]=coarse_wave[i//2,j//2] if i%2==0 and j%2==0 else new_prediction[lookup['fine_new',bundle,i*16+j]]
            fine_records[str(bundle)]=orbit.waveform_stability(coarse_wave,fine,origins,off_wave,phases,scale)
    if set(fine_records)!={str(b) for b in (0,3,4,7)} or set(offgrid_records)!={str(b) for b in range(8)}:
        raise ValueError('Fine and offgrid bundle scopes cannot be combined or shortened')
    stable=all(r['stable_low_order'] for r in fine_records.values())
    fine_energy=all(r['beyond_order3_energy']<=.05 for r in fine_records.values())
    off_pass=sum(r['passed'] for r in offgrid_records.values())
    coverage=bool(fine_energy and off_pass>=7)
    status='unresolved_or_aliased' if not stable else ('stable_low_order_partial_coverage' if not coverage else 'stable_and_adequate')
    return dict(fine_bundles=fine_records,offgrid_bundles=offgrid_records,
        all_four_low_order_stable=bool(stable),all_four_beyond_order3_pass=bool(fine_energy),
        offgrid_pass_count_all_eight=int(off_pass),adequate_full_response_coverage=coverage,status=status)


def comparison_statuses(responses,pilot_candidates,new_native_pass):
    """Keep all six and require BOTH responses; failed pilots stay closed."""
    result={}
    for first,second in combinations(('timesfm','bolt','chronos2'),2):
        for stem in ('p32_16','p40_20'):
            key=f'{first}-{second}/{stem}'
            eligible=pilot_candidates[key]['promotes_alias_validation']
            sides=[responses[model][stem] for model in (first,second)]
            stable=all(r['all_four_low_order_stable'] for r in sides)
            adequate=all(r['adequate_full_response_coverage'] for r in sides)
            status=('pilot_negative_closed' if not eligible else 'new_native_control_failed' if not new_native_pass
                    else 'unresolved_or_aliased' if not stable else 'stable_low_order_partial_coverage' if not adequate else 'stable_and_adequate')
            result[key]=dict(pilot_numeric_gate=bool(eligible),both_model_responses_stable=bool(stable),
                both_model_responses_adequate=bool(adequate),status=status,hidden_capture_authorized=False)
    if len(result)!=6:raise ValueError('Complete family required')
    return result


def prepare(pilot,replay,control,out):
    if out.exists():raise ValueError('Never overwrite a registered follow-up')
    pp=json.loads((pilot/'protocol.json').read_text(encoding='utf-8'))
    old=json.loads((pilot/'results.json').read_text(encoding='utf-8'))
    corrected=json.loads((replay/'results.json').read_text(encoding='utf-8'))
    controls=json.loads((control/'results.json').read_text(encoding='utf-8'))
    if old['controls_pass'] or not corrected['corrected_math_pass'] or not controls['all_new_controls_pass']:
        raise ValueError('Require original failed artifact, corrected math and separately passed fixedshape control')
    source=Path(pp['core_source']);coarse_bindings={}
    for model in ('timesfm','bolt','chronos2'):
        for stem in ('p32_16','p40_20'):
            native.load_prediction(pilot/f'{model}_{stem}_prediction.npz',model,stem,source)
            coarse_bindings[f'{model}/{stem}']=native.digest(pilot/f'{model}_{stem}_prediction.npz')
    out.mkdir(parents=True);bindings={}
    for pair in orbit.PAIRS:
        stem=f'p{pair[0]}_{pair[1]}';x,y,rows=requests(pair)
        np.savez_compressed(out/f'{stem}_requests.npz',context=x,clean_future=y,phase_row_ids=ids(rows,stem))
        (out/f'{stem}_requests.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        bindings[stem]={suffix:native.digest(out/f'{stem}_requests.{suffix}') for suffix in ('npz','json')}
    protocol=dict(stage='49.10',role='New amended alias follow-up,not retroactive pilot control pass',
        original_pilot=str(pilot),original_failed_result_sha256=native.digest(pilot/'results.json'),
        original_global_controls_pass=False,original_singleton_max_abs=7.152557373046875e-7,
        fp64_replay=str(replay),fp64_replay_sha256=native.digest(replay/'results.json'),
        fixed_B8_control=str(control),fixed_B8_control_sha256=native.digest(control/'results.json'),
        models=pp['models'],installed_sources=pp['sources'],coarse_source=str(source),coarse_forecast_sha256=coarse_bindings,
        primary_rows=5376,budget='672primaryB8calls+12repeat/permutation+4C2unique/neighbor=688calls=5504contexts;allcallsB8,no padding',
        fine_bundles=(0,3,4,7),fine_grid=16,new_fine_cells_per_pair=768,offgrid_bundles=tuple(range(8)),offgrid_cells_per_bundle=16,
        offgrid_reconstruction='Coarse8 loworderONLY forALL8;no fine/coarse selection',mode_order=3,measurement_dtype='FP64',
        primary='Full waveform phaseDFT;both stability bounds onALL4;highorderenergy onALL4;coarseoffgridRMS on>=7/8 ofALL8;zcoefficientssecondary',
        thresholds=dict(relative_low_order=.05,absolute_low_order_over_scale=.02,beyond_order3_over_mix_energy=.05,offgrid_RMS_over_scale=.05),
        candidate_family='ALL6 originalcomparisons including2pilotnegatives;both model/period responses mustpass;negativepilot cannotbe rescued',
        native_controls='B8repeat/permutation exact0 eachmodel/pair;C2explicitarange andsame-slot target0 neighborreplace exact0 onactualnewcorpus;oldsingletonfailretained',
        no_expansion='No amplitude/phase/grid/order adaptation,no hidden captures,no native ranking reversal claim',
        resources=dict(gpu=3,cpu_threads=2,batch_size=8,cap_bytes=6*2**30,timeout_seconds=600),
        runner_sha256=native.digest(__file__),math_sha256=native.digest(orbit.__file__),fp64_math_sha256=native.digest(fp64.__file__),request_bindings=bindings)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')


def load_new_prediction(path,model,stem,out):
    rows=json.loads((out/f'{stem}_requests.json').read_text(encoding='utf-8'))
    with np.load(path,allow_pickle=False) as file:
        expected=dict(model=model,case=stem,request_corpus_sha256=native.digest(out/f'{stem}_requests.npz'),
            request_metadata_sha256=native.digest(out/f'{stem}_requests.json'),protocol_sha256=native.digest(out/'protocol.json'))
        for key,value in expected.items():
            if file[key].item()!=value:raise ValueError('New forecast identity binding failed')
        if not np.array_equal(file['phase_row_ids'],ids(rows,stem)) or file['prediction'].shape!=(896,64):
            raise ValueError('New forecast orderedphase/horizon mismatch')
        return file['prediction'].copy()


def run(out):
    import torch
    from tsfm_lens.config import ModelConfig,DataConfig
    from tsfm_lens.models import build_adapter
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if p['runner_sha256']!=native.digest(__file__) or p['math_sha256']!=native.digest(orbit.__file__) or p['fp64_math_sha256']!=native.digest(fp64.__file__):
        raise ValueError('Registered sources changed')
    if (out/'results.json').exists() or any(out.glob('*_new_prediction.npz')):raise ValueError('Preserve partial/completed follow-up')
    for path,value in p['installed_sources'].items():
        if native.digest(path)!=value:raise ValueError('Installed source changed')
    for field,key in [('original_pilot','original_failed_result_sha256'),('fp64_replay','fp64_replay_sha256'),('fixed_B8_control','fixed_B8_control_sha256')]:
        if native.digest(Path(p[field])/'results.json')!=p[key]:raise ValueError('Evidence anchor changed')
    cases={}
    for stem,binding in p['request_bindings'].items():
        for suffix,value in binding.items():
            if native.digest(out/f'{stem}_requests.{suffix}')!=value:raise ValueError('Requests changed')
        with np.load(out/f'{stem}_requests.npz') as file:x=file['context'].copy()
        rows=json.loads((out/f'{stem}_requests.json').read_text(encoding='utf-8'))
        cases[stem]=(x,rows)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(1.,6*2**30/torch.cuda.get_device_properties(0).total_memory))
    started=time.monotonic();responses={};telemetry={};all_pass=True;calls=0
    adapter_names=dict(timesfm='timesfm',bolt='chronos_bolt',chronos2='chronos2')
    for model,manifest in p['models'].items():
        checkpoint=Path(manifest['checkpoint'])
        if native.digest(checkpoint)!=manifest['weight_sha256'] or native.digest(checkpoint.parent/'config.json')!=manifest['config_sha256']:raise ValueError('Weights/config changed')
        model_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        adapter=build_adapter(ModelConfig(name=model,adapter=adapter_names[model],checkpoint=str(checkpoint.parent)),DataConfig(context_len=128,horizon=64),torch.device('cuda:0'),torch.float32)
        adapter.ensure_loaded();responses[model]={};checks={}
        supported=(int(adapter.tfm.model.o) if model=='timesfm' else int(adapter.pipeline.model_prediction_length)
                   if model=='bolt' else 16*adapter._num_output_patches(64))
        if supported<64:raise ValueError('Single native horizon no longer covers H64')
        def predict(batch,groups=None):
            nonlocal calls
            if time.monotonic()-started>600:raise RuntimeError('Ten-minute cap reached')
            if batch.shape!=(8,128) or not np.isfinite(batch).all():raise ValueError('Every call must use finiteB8×128')
            handle=None;hits=[]
            if groups is not None:
                def inject(_,args,kwargs):
                    hits.append(1);kwargs=dict(kwargs);kwargs['group_ids']=torch.tensor(groups,device=adapter.device,dtype=torch.long);return args,kwargs
                handle=adapter._inner.register_forward_pre_hook(inject,with_kwargs=True)
            try:
                with torch.inference_mode():value=adapter.predict(batch,64,[.5])['point']
            finally:
                if handle is not None:handle.remove()
            if groups is not None and len(hits)!=1:raise ValueError('One native group injection required')
            if value.shape!=(8,64) or not np.isfinite(value).all():raise ValueError('Output shape/nonfinite')
            calls+=1;return value
        for stem,(x,rows) in cases.items():
            forecast=np.concatenate([predict(x[i:i+8]) for i in range(0,len(x),8)])
            repeat=predict(x[:8]);perm=np.array([7,0,5,2,6,1,4,3]);shuffled=predict(x[:8][perm])[np.argsort(perm)]
            check=dict(repeat=float(abs(repeat-forecast[:8]).max()),permutation=float(abs(repeat-shuffled).max()))
            if model=='chronos2':
                explicit=predict(x[:8],np.arange(8));neighbor=x[-8:].copy();neighbor[0]=x[0]
                changed=predict(neighbor)
                check.update(default_vs_explicit=float(abs(explicit-repeat).max()),fixed_target_neighbor=float(abs(changed[0]-repeat[0]).max()))
            all_pass=all_pass and all(value==0 for value in check.values());checks[stem]=check
            path=out/f'{model}_{stem}_new_prediction.npz'
            np.savez_compressed(path,prediction=forecast,model=np.asarray(model),case=np.asarray(stem),phase_row_ids=ids(rows,stem),
                request_corpus_sha256=np.asarray(native.digest(out/f'{stem}_requests.npz')),
                request_metadata_sha256=np.asarray(native.digest(out/f'{stem}_requests.json')),protocol_sha256=np.asarray(native.digest(out/'protocol.json')))
            observed=load_new_prediction(path,model,stem,out)
            coarse_path=Path(p['original_pilot'])/f'{model}_{stem}_prediction.npz'
            if native.digest(coarse_path)!=p['coarse_forecast_sha256'][f'{model}/{stem}']:raise ValueError('Coarse forecast changed')
            source=Path(p['coarse_source']);coarse=native.load_prediction(coarse_path,model,stem,source)['prediction']
            coarse_rows=json.loads((source/f'{stem}.json').read_text(encoding='utf-8'))
            pair=tuple(int(part.lstrip('p')) for part in stem.split('_'))
            responses[model][stem]=assess_case(coarse,observed,rows,coarse_rows,pair)
        telemetry[model]=dict(wall_seconds=time.monotonic()-model_started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),native_controls=checks,native_supported_horizon=supported)
        adapter._release();del adapter;gc.collect();torch.cuda.empty_cache()
    if calls!=688:raise ValueError('Frozen call budget changed')
    pilot_candidates=json.loads((Path(p['fp64_replay'])/'results.json').read_text(encoding='utf-8'))['candidates_descriptive_only']
    result=dict(responses=responses,comparisons=comparison_statuses(responses,pilot_candidates,all_pass),new_native_controls_pass=bool(all_pass),
        telemetry=telemetry,total_wall_seconds=time.monotonic()-started,native_calls=calls,context_forecasts=calls*8,
        original_failed_controls_retained=True,original_native_ranking_reversal_failed=True,hidden_capture_authorized=False,
        scope='Amended alias validation only;stablepartialpatterns do not imply complete explanation;all6 outcomes retained')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(comparisons=result['comparisons'],new_native_controls_pass=all_pass,telemetry=telemetry),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--pilot',type=Path);parser.add_argument('--replay',type=Path);parser.add_argument('--control',type=Path);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    if a.prepare:prepare(a.pilot,a.replay,a.control,a.out)
    else:run(a.out)
