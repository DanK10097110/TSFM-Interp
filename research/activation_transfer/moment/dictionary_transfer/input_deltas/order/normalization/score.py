"""Measure the effect of holding pre-encoder RevIN fixed, on matched real histories."""
import json

import numpy as np

from research.activation_transfer.moment.dictionary_transfer.input_deltas.order.normalization.run import CACHE, ORDER_OUT, OUT, inputs, original_state, prior


def describe(v):
    return dict(median=float(np.median(v)), mean=float(np.mean(v)), p95=float(np.quantile(v,.95)), max=float(np.max(v)))


def centered(v):
    s = v[:, :8].astype(float)
    return s-s.mean(axis=1,keepdims=True)


def paired_waveform(native, frozen, weights):
    base, changed = centered(native), centered(frozen)
    gap = changed-base
    rms = lambda x: float(np.sqrt(np.mean(x*x)))
    def totals(x):
        sw, tw = weights
        cells = np.sum(x*x,axis=1).reshape(sw.shape[1],tw.shape[1])
        return np.sum((sw@cells)*tw,axis=1)
    ratio_arr = np.sqrt(totals(gap)/totals(base))
    return dict(ordinary_rms_C=rms(base),fixed_rms_C=rms(changed),difference_rms_C=rms(gap),
                relative_rms=rms(gap)/rms(base),relative_rms_station_date_95=np.quantile(ratio_arr,[.025,.975]).tolist(),
                cosine=float(np.sum(base*changed)/(np.linalg.norm(base)*np.linalg.norm(changed))))


def score():
    ids, dates, arrays = inputs()
    counts = np.random.default_rng(22112452)
    w = (counts.multinomial(len(ids), np.full(len(ids),1/len(ids)),3000),
         counts.multinomial(len(dates),np.full(len(dates),1/len(dates)),3000))
    with np.load(prior.MATCH/'small_to_large_map.npz') as f:
        m = f['w'].astype(float)
    summary = dict(stations=len(ids),dates=list(dates),histories=len(arrays['base']))
    raw = {k:v.astype(float).std(1) for k,v in arrays.items()}
    with np.load(OUT/'small_normalizer.npz') as a, np.load(OUT/'large_normalizer.npz') as b:
        for k in ['base','ramp','permuted']:
            np.testing.assert_array_equal(a[k+'_native_scale'],b[k+'_native_scale'])
            np.testing.assert_allclose(a[k+'_native_scale'],raw[k]+1e-5,atol=2e-6,rtol=1e-6)
    summary['relative_raw_SD'] = {}
    for k,den in [('ramp','base'),('permuted','base'),('permuted','ramp')]:
        ratio = (raw[k]-raw[den])/raw[den]
        summary['relative_raw_SD'][k+'_vs_'+den] = dict(signed_median=float(np.median(ratio)),abs=describe(abs(ratio)))
    store = dict(ids=ids,dates=dates)
    summary['hidden_displacement_change']={}
    for space in ('small','large','mapped'):
        size = 'small' if space=='mapped' else space
        ordinary = {k:original_state(size,k,v) for k,v in arrays.items()}
        with np.load(CACHE/f'{size}.npz') as f:
            fixed = dict(f)
        for k in ['ramp','permuted','order_difference']:
            end,start = ('permuted','ramp') if k=='order_difference' else (k,'base')
            first, second = ordinary[end]-ordinary[start], fixed[end]-fixed[start]
            if space=='mapped':
                first,second = first@m,second@m
            a, b = first.astype(float),second.astype(float)
            den = np.linalg.norm(a,axis=1)
            assert np.all(den>0)
            rel = np.linalg.norm(b-a,axis=1)/den
            store[space+'_'+k] = rel
            summary['hidden_displacement_change'][space+'_'+k]=describe(rel)
    np.savez_compressed(OUT/'case_hidden_relative_changes.npz',**store)
    def wave(path):
        with np.load(path) as f:
            np.testing.assert_array_equal(ids,f['ids']); np.testing.assert_array_equal(dates,f['dates'])
            return f['wave'].astype(float)
    native, fixed = {}, {}
    for arm,file in [('reference','reference'),('mapped','correct')]:
        native[arm] = dict(ramp=wave(prior.OUT/f'{arm}.npz'), permuted=wave(ORDER_OUT/f'permuted_{file}.npz'))
        fixed[arm] = {k:wave(OUT/f'{k}_{arm}.npz') for k in ('ramp','permuted')}
        for family in [native,fixed]:
            family[arm]['order_difference'] = family[arm]['permuted']-family[arm]['ramp']
    summary['early_centered_waveform_change']={}
    for arm in native:
        for name in native[arm]:
            summary['early_centered_waveform_change'][arm+'_'+name]=paired_waveform(native[arm][name],fixed[arm][name],w)
    summary['early_order_contrast_recovery']={}
    for condition,cases in [('ordinary',native),('fixed',fixed)]:
        truth,est=(centered(cases[arm]['order_difference']) for arm in ('reference','mapped'))
        summary['early_order_contrast_recovery'][condition]=dict(skill=float(1-np.sum((truth-est)**2)/np.sum(truth**2)),
            cosine=float(np.sum(truth*est)/(np.linalg.norm(truth)*np.linalg.norm(est))))
    published = json.loads((ORDER_OUT/'paired_order_difference.json').read_text())['small_to_large']['early_shape']['correct']
    baseline = summary['early_order_contrast_recovery']['ordinary']
    np.testing.assert_allclose([baseline['skill'],baseline['cosine']],
                               [published['skill'][0],published['signed_cosine'][0]],atol=1e-12)
    summary['max_full_replay_C']={}
    for arm in ('reference','mapped'):
        for edit in ('ramp','permuted'):
            with np.load(OUT/f'{edit}_{arm}.npz') as f:
                summary['max_full_replay_C'][edit+'_'+arm]=float(f['native_error_C'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    score()
