"""Equal-feature physical recovery, including false writes when the true target feature is closed."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates import run as original
from research.activation_transfer.moment.crosscoder.interventions import OUT, EXPECTED_HASH, ATOMS, digest


def crossed_sum(values, station_weights, date_weights):
    nsite=station_weights.shape[1]
    ndate=date_weights.shape[1]
    grid=values.reshape(nsite,ndate,-1)
    return np.einsum('bs,stf,bt->bf',station_weights,grid,date_weights,optimize=True)


def normalized_shape(wave,sd):
    values=wave.astype(np.float64)/sd[:,None,None]
    return values-values.mean(axis=-1,keepdims=True)


def ratio(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>1e-16)


def summary(v):
    return dict(value=float(v[0]),crossed95=np.nanquantile(v[1:],[.025,.975]).tolist())


def score_opportunities(target, predicted, open_gate, accumulator):
    """Evaluate every possible deletion; also expose harm where no natural deletion exists."""
    energy = np.sum(target**2, axis=-1)
    scale = accumulator(energy)
    error = np.sum((predicted-target)**2, axis=-1)
    harm = np.sum(predicted**2, axis=-1) * (~open_gate)
    return (1-ratio(accumulator(error),scale),
            1-ratio(accumulator(error*open_gate),scale),
            ratio(accumulator(harm),scale))


def matching_mask_pairing(seed_scores):
    """Only masked versus shuffled fixes the 50% target-view masking schedule."""
    return summary(seed_scores['masked']-seed_scores['shuffled'])


def run(out):
    with np.load(original.OUT/'donor.npz') as native:
        sites,dates,truth,sd,dose=(native[k] for k in ('ids','dates','wave','sd','dose'))
        np.testing.assert_array_equal(ATOMS,native['atoms'])
    assert (len(sites),len(dates),len(sd))==(35,11,385)
    rng=np.random.default_rng(22112397)
    sw=np.vstack([np.ones(35),rng.multinomial(35,np.full(35,1/35),3000)])
    tw=np.vstack([np.ones(11),rng.multinomial(11,np.full(11,1/11),3000)])
    per_draw=lambda a:crossed_sum(a,sw,tw)
    centered=normalized_shape(truth,sd)
    on=dose>0
    waves_c=truth.astype(float)-truth.astype(float).mean(axis=-1,keepdims=True)
    egrid=np.sum(centered**2,axis=-1).reshape(35,11,4)
    denominator=per_draw(egrid.reshape(385,4))
    all_four=np.all(denominator>1e-16,axis=1)
    filepaths=sorted(out.glob('causal_*.npz'))
    data=dict(n=385,station_count=35,dates=dates.tolist(),atoms=ATOMS.tolist(),fires=on.sum(0).tolist(),
              native_centered_rms_c=np.sqrt(np.mean(waves_c**2,axis=(0,2))).tolist(),
              native_centered_rms_when_firing_c=[float(np.sqrt(np.mean(waves_c[on[:,i],i]**2))) for i in range(4)],
              active_station_counts=(egrid.sum(1)>0).sum(0).tolist(),active_date_counts=(egrid.sum(0)>0).sum(0).tolist(),
              maximum_date_energy_share=(egrid.sum(0).max(axis=0)/egrid.sum((0,1))).tolist(),
              zero_energy_bootstraps=(denominator[1:]<=1e-16).sum(axis=0).tolist(),
              all_four_eligible_bootstraps=int(all_four[1:].sum()),arms={},paired={})
    draws={}
    for arm,path in [('zero',None)]+[(p.stem[len('causal_'):],p) for p in filepaths]:
        if path is None:
            y=np.zeros_like(truth);pred=np.zeros_like(dose);replay={}
        else:
            with np.load(path) as f:
                assert str(f['input_sha'])==EXPECTED_HASH
                np.testing.assert_array_equal(sites,f['ids']);np.testing.assert_array_equal(dates,f['dates'])
                np.testing.assert_array_equal(ATOMS,f['atoms']);np.testing.assert_array_equal(sd,f['sd'])
                pred=f['dose'];y=f['wave']
                assert digest(pred)==str(f['dose_sha'])
                replay=dict(full_count=len(f['full_checks']),max_full_error_c=float(np.max(f['full_errors_c'])),
                            max_native_target_error_c=float(np.max(f['native_reference_errors_c'])),
                            independent_writes=int(f['live_writes']),reused=int(f['reused_identical']))
        pred_on=pred>0
        np.testing.assert_array_equal(y[~pred_on],0)
        impact=normalized_shape(y,sd)
        score,open_score,harm=score_opportunities(centered,impact,on,per_draw)
        macro=np.nanmean(score,axis=1)
        open_macro=np.nanmean(open_score,axis=1);harm_macro=np.nanmean(harm,axis=1)
        np.testing.assert_allclose(open_macro-macro,harm_macro,rtol=1e-7,atol=1e-7)
        draws[arm]=macro
        raw=y.astype(float)-y.astype(float).mean(axis=-1,keepdims=True)
        raw_error=raw-waves_c
        off=~on
        fp=off & pred_on
        tp=(pred_on&on).sum(axis=0)
        data['arms'][arm]=dict(all=summary(macro),open=summary(open_macro),false_positive_energy=summary(harm_macro),
            complete_feature_bootstrap95=np.quantile(macro[1:][all_four[1:]],[.025,.975]).tolist(),
            per_atom=score[0].tolist(),per_atom_crossed95=np.nanquantile(score[1:],[.025,.975],axis=0).T.tolist(),
            per_atom_open=open_score[0].tolist(),per_atom_false_positive_energy=harm[0].tolist(),
            firing=pred_on.sum(0).tolist(),tp=tp.tolist(),fp=fp.sum(0).tolist(),fn=(on&~pred_on).sum(0).tolist(),
            centered_rms_c=np.sqrt(np.mean(raw**2,axis=(0,2))).tolist(),
            centered_error_rms_c=np.sqrt(np.mean(raw_error**2,axis=(0,2))).tolist(),
            centered_false_positive_rms_over_closed_c=[float(np.sqrt(np.mean(raw[off[:,i],i]**2))) for i in range(4)],
            centered_false_positive_rms_on_mistakes_c=[float(np.sqrt(np.mean(raw[fp[:,i],i]**2))) if fp[:,i].any() else None for i in range(4)],
            replay=replay)
    for a in draws:
        for b in draws:
            if a!=b:
                data['paired'][a+'_minus_'+b]=summary(draws[a]-draws[b])
    seed_scores={}
    for prefix in ('joint','masked','shuffled'):
        first,second=prefix+'1',prefix+'2'
        if first in draws and second in draws:
            avg=(draws[first]+draws[second])/2
            seed_scores[prefix]=avg
            data.setdefault('seed_means',{})[prefix]=summary(avg)
            for comp in draws:
                data['paired'][prefix+'_mean_minus_'+comp]=summary(avg-draws[comp])
    for a,first in seed_scores.items():
        for b,second in seed_scores.items():
            if a!=b:
                data['paired'][a+'_mean_minus_'+b+'_mean']=summary(first-second)
    if 'masked' in seed_scores and 'shuffled' in seed_scores:
        data['paired']['masked_paired_mean_minus_masked_shuffled_mean'] = matching_mask_pairing(seed_scores)
        data['controlled_pairing'] = {
            'paired_minus_shuffled_both_50pct_target_mask': matching_mask_pairing(seed_scores),
            'comparison': 'Same target-view masking. Both downstream dose readouts use correct training labels.',
        }
    if 'joint' in seed_scores and 'shuffled' in seed_scores:
        data['paired']['unmasked_paired_mean_minus_masked_shuffled_mean'] = summary(seed_scores['joint']-seed_scores['shuffled'])
        data['mixed_training_modes'] = {
            'ordinary_unmasked_paired_minus_shuffled_50pct_target_mask': summary(seed_scores['joint']-seed_scores['shuffled']),
            'comparison': 'Both pairing and masking differ.',
        }
    tmp=out/f'causal_summary.{os.getpid()}.partial'
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(out/'causal_summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    run(p.parse_args().out)
