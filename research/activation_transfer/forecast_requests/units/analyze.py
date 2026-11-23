"""Compare physical held-out writes using the original per-series request metrics."""
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.forecast_requests.summary import item_scores, stats

HERE=Path(__file__).resolve().parent
ARCHIVES=HERE.parent/'results'
CUES=['level','tilt','sine']


def mean_sham(effect,goal):
    scored=[item_scores(effect[:,j],goal) for j in range(effect.shape[1])]
    return {k:np.mean([m[k] for m in scored],axis=0) for k in scored[0]}


def analyze():
    rows=[]; scores={}; ids_ref=None; null_drift={}
    for target in ('timesfm','bolt'):
        source='bolt' if target=='timesfm' else 'timesfm'; key='0_'+source
        with np.load(HERE/(target+'_effects.npz')) as new, np.load(ARCHIVES/(target+'.npz')) as old, np.load(ARCHIVES/(source+'.npz')) as src, np.load(ARCHIVES/('dose_shams_'+target+'.npz')) as sham:
            ids=new['ids']; want=new['desired']; effect=new['effects']; probe=new['unit_calibration']
            for data in [old,src,sham]: np.testing.assert_array_equal(ids,data['ids'])
            if ids_ref is None: ids_ref=ids
            else: np.testing.assert_array_equal(ids_ref,ids)
            raw=old[key]; original_sham=sham[key]; target_native=old['0_native']; source_native=src['0_native']
            # At exactly zero scale, the physical change is zero. Keep raw replay noise in the NPZs.
            recorded=[]
            for values, scales in [(effect,new['scales']), (original_sham,new['original_scales'][:,1:]),
                                   (raw[:,:,1:2],new['original_scales'][:,:1])]:
                mask=scales==0
                if mask.any():
                    recorded.append(float(abs(values[:,mask]).max()))
                    values[:,mask]=0.
            null_drift[target]=max(recorded,default=0.)
        for domain in ('daily','weekly'):
            keep=np.char.startswith(ids,domain+'/');n=int(keep.sum())
            if n!=128: raise ValueError('Unexpected cohort size')
            b=np.random.default_rng(202611208).integers(n,size=(4000,n))
            for i,cue in enumerate(CUES):
                g=want[i]
                effects={'raw_direct':raw[keep,i,0],'raw_calibrated':raw[keep,i,1], 'target_native':target_native[keep,i],
                         'source_native':source_native[keep,i],'recipient_calibrated':effect[keep,i,0]}
                vals={arm:item_scores(y.astype(float),g) for arm,y in effects.items()}
                vals['raw_calibrated_shams']=mean_sham(original_sham[keep,i].astype(float),g)
                vals['recipient_calibrated_shams']=mean_sham(effect[keep,i,1:].astype(float),g)
                scored={arm:{metric:stats(v,b) for metric,v in m.items()} for arm,m in vals.items()}
                pairs=[('recipient_calibrated','raw_calibrated'),('recipient_calibrated','recipient_calibrated_shams'),
                       ('raw_calibrated','raw_calibrated_shams'),('recipient_calibrated','target_native')]
                delta={f'{a}_minus_{c}':stats(vals[a]['skill']-vals[c]['skill'],b) for a,c in pairs}
                unit_probe=item_scores(probe[:,i,0].astype(float),g)
                rows.append(dict(domain=domain,source=source,target=target,cue=cue,test_n=n,
                                 methods=scored,paired_skill_differences=delta,
                                 unit_probe_on_calibration={k:float(v.mean()) for k,v in unit_probe.items()}))
                scores[domain,target,cue]={key:item['skill'] for key,item in vals.items()}
    gaps={}
    for domain in ('daily','weekly'):
        f=scores[domain,'timesfm','level']; r=scores[domain,'bolt','level']; n=len(f['target_native'])
        b=np.random.default_rng(202611208).integers(n,size=(4000,n))
        raw=f['raw_calibrated']-r['raw_calibrated']; unit=f['recipient_calibrated']-r['recipient_calibrated']
        before=stats(raw,b); after=stats(unit,b); change=stats(unit-raw,b)
        gaps[domain]=dict(raw=before,recipient=after,paired_change=change,
                          fraction_absolute_gap_closed=float(1-abs(unit.mean())/abs(raw.mean())),
                          within_five_points=bool(after['ci95'][0]>-.05 and after['ci95'][1]<.05))
    return dict(rows=rows,level_gap_bolt_to_timesfm_minus_reverse=gaps,max_structural_zero_drift=null_drift)


if __name__=='__main__':
    (HERE/'results.json').write_text(json.dumps(analyze(),indent=2,allow_nan=False)+'\n')
