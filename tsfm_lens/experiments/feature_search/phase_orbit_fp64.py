"""CPU precision replay of unchanged saved forecasts; original failures remain.

NumPy2 can preserve FP32/complex64 in FFT. Cast both observed and analytic
waveforms explicitly before the unchanged mathematical decomposition.
"""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from experiments.feature_search import phase_orbit as orbit
from experiments.feature_search import phase_orbit_native as native


def corrected_budget(prediction,truth,origins,scale,base_period):
    return orbit.waveform_budget(np.asarray(prediction,dtype=np.float64),
        np.asarray(truth,dtype=np.float64),origins,scale,base_period)


def descriptive_candidates(records):
    if set(records)!={'timesfm','bolt','chronos2'}:raise ValueError('Retain all three models, including unresolved C2')
    result={}
    for first,second in combinations(('timesfm','bolt','chronos2'),2):
        for stem in ('p32_16','p40_20'):
            a=np.asarray([r['primary_D'] for r in records[first][stem]])
            b=np.asarray([r['primary_D'] for r in records[second][stem]])
            result[f'{first}-{second}/{stem}']=orbit.pilot_interestingness(a,b)
    if len(result)!=6:raise ValueError('Retain the complete six-comparison family')
    return result


def replay(source,out):
    if out.exists():raise ValueError('Preserve original and previous replay artifacts')
    original=json.loads((source/'results.json').read_text(encoding='utf-8'))
    producer=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    corpus=Path(producer['core_source'])
    protocol=dict(source=str(source),original_result_sha256=native.digest(source/'results.json'),
        original_protocol_sha256=native.digest(source/'protocol.json'),
        core_protocol_sha256=native.digest(corpus/'protocol.json'),
        replay_driver_sha256=native.digest(__file__),math_source_sha256=native.digest(orbit.__file__),
        original_global_controls_pass=original['controls_pass'],
        correction='Cast saved forecast AND analytic truth toFP64 beforeFFT;no new inference,native outputs unchanged',
        mathematical_gate='same absolute normalized Parseval tolerance1e-10;no threshold change',
        independent_native_failure='C2 singleton-vsB8 exact-zero control remains failed/unresolved;never reinterpret original global controls',
        eligibility_scope='all six candidate counts descriptive only;no promotion/refinement/capture authorization')
    out.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records={};bindings={};max_error=0.
    for model in ('timesfm','bolt','chronos2'):
        records[model]={}
        for stem in ('p32_16','p40_20'):
            path=source/f'{model}_{stem}_prediction.npz'
            saved=native.load_prediction(path,model,stem,corpus)
            rows=json.loads((corpus/f'{stem}.json').read_text(encoding='utf-8'))
            truth=np.load(corpus/f'{stem}.npz')['clean_future']
            period=int(stem.split('_')[0][1:]);records[model][stem]=[]
            for bundle in range(8):
                sl=slice(bundle*64,(bundle+1)*64);row=rows[bundle*64]
                result=corrected_budget(saved['prediction'][sl].reshape(8,8,64),
                    truth[sl].reshape(8,8,64),(row['origin_phi'],row['origin_psi']),row['scale'],period)
                result['bundle']=bundle;records[model][stem].append(result)
                max_error=max(max_error,result['error_Parseval_absolute_error'])
            bindings[f'{model}/{stem}']=native.digest(path)
    candidates=descriptive_candidates(records)
    result=dict(records=records,candidates_descriptive_only=candidates,
        corrected_math_Parseval_max_abs=max_error,corrected_math_pass=bool(max_error<=1e-10),
        original_global_controls_pass=original['controls_pass'],
        original_telemetry_and_failed_native_controls=original['telemetry'],
        original_candidate_outcomes_unchanged=original['candidates'],forecast_sha256=bindings,
        advance=False,scope='CPU math precision correction only;original pilot failed remains unchanged;C2 native independence unresolved;no alias validation performed')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(corrected_math_pass=result['corrected_math_pass'],
        max_Parseval_error=max_error,original_global_controls_pass=original['controls_pass'],
        candidates=candidates,advance=False),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();replay(args.source,args.out)
