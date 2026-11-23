"""Join receiver-only predictions with donor labels only after physical test writes are complete."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.activation_transfer.atoms.scores import resampling_counts,terms
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import OUT,SEED
from research.activation_transfer.moment.dictionary_transfer.weather.score import centered,infer,interval,matched_effect
from research.activation_transfer.moment.dictionary_transfer.weather.study import OUT as PRIOR


def detection(estimate,truth):
    positive=estimate>0;active=truth>0
    tp=int(np.sum(positive&active));fp=int(np.sum(positive&~active));fn=int(np.sum(~positive&active))
    energy=np.sum((truth-truth.mean())**2)
    return dict(tp=tp,fp=fp,fn=fn,tn=int(np.sum(~positive&~active)),precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,dose_r2=float(1-np.sum((estimate-truth)**2)/energy) if energy else None)


def run(out,prior):
    fit=dict(np.load(out/'fit.npz'));det=dict(np.load(out/'inferred.npz'));natural=dict(np.load(out/'natural.npz'))
    original=dict(np.load(prior/(str(SEED)+'.npz')))
    for m in (det,natural):
        np.testing.assert_equal(original['ids'],m['ids']);np.testing.assert_equal(fit['atoms'],m['atoms'])
    j=fit['donor_cols'];atoms=fit['atoms'];n=len(det['ids']);truth=original['z'][:,j]
    original_rows=original['rows'];original_cols=original['cols']
    def fill(arm):
        arr=np.zeros((n,len(j),16))
        vals=matched_effect(original,arm)
        for i,a in enumerate(j):
            pick=original_cols==a;arr[original_rows[pick],i]=vals[pick]
        return arr
    source=fill('donor');oracle=fill('group16');nat=natural['native'];rec=det['native'];station=det['station'];sd=det['input_sd']
    for i,a in enumerate(j):
        m=(original_cols==a);r=original_rows[m]
        np.testing.assert_allclose(nat[r,i],matched_effect(original,'corr_own')[m],atol=.0002,rtol=.001)
    np.testing.assert_equal(source[truth==0],0.)
    candidates=dict(natural=nat,inferred=rec,oracle=oracle,zero=np.zeros_like(source))
    sw=resampling_counts(len(set(station)),draws=3000)
    fw=np.random.default_rng(26112283).multinomial(len(j),np.full(len(j),1/len(j)),3000)
    summary=dict(cases=n,stations=len(set(station)),atoms=atoms.tolist(),training_frequencies=fit['historical_firing'][j].tolist(),
                 lambda_selected=fit['lam'].tolist(),thresholds=fit['threshold'].tolist(),validation_dose=[],detection={},metrics={})
    for i in range(len(j)):summary['validation_dose'].append(detection(fit['val_prediction'][:,i],fit['val_observed'][:,i]))
    for label,observed in [('inferred',det['dose']),('natural',natural['dose'])]:
        summary['detection'][label]={str(a):detection(observed[:,i],truth[:,i]) for i,a in enumerate(atoms)}
        summary['detection'][label]['pooled']=detection(observed.ravel(),truth.ravel())
    def sanitize(arr):
        return [float(v) if np.isfinite(v) else None for v in arr]
    for subset,mask in [('all',np.ones_like(truth,bool)),('donor_firing',truth>0)]:
        for kind in ('centered','full'):
            conv=(lambda y:centered(y/sd[:,None,None])) if kind=='centered' else (lambda y:y/sd[:,None,None])
            y=conv(source);estimates={};stats={}
            for label,raw in candidates.items():
                pred=conv(raw);moments=terms(mask,pred,y)
                pooled=np.stack([moments[station==site].sum(0) for site in np.unique(station)])
                a,b,c,per=infer(pooled,sw,fw);estimates[label]=(a,b,c)
                den=np.sum(y*y,axis=(0,2));energy=np.sum(pred*pred,axis=(0,2));wrong=np.sum(pred**2*(truth==0)[:,:,None],axis=(0,2))
                false_share=np.divide(wrong,energy,out=np.zeros_like(wrong),where=energy>0)
                fp_to_ref=np.divide(wrong,den,out=np.full_like(wrong,np.nan),where=den>0)
                stats[label]=dict(skill=a,station95=interval(b),crossed95=interval(c),per_atom=sanitize(per),
                                 false_energy_over_donor=sanitize(fp_to_ref),false_fraction_of_candidate=false_share.tolist(),
                                 false_energy_over_donor_macro=float(np.nanmean(fp_to_ref)),
                                 false_energy_over_donor_pooled=float(wrong.sum()/den.sum()),
                                 false_fraction_of_candidate_pooled=float(wrong.sum()/energy.sum()) if energy.sum()>0 else None,
                                 pooled_sd_rms=float(np.sqrt(np.mean(pred**2))),
                                 pooled_celsius_rms=float(np.sqrt(np.mean((centered(raw) if kind=='centered' else raw)[mask]**2))))
                if subset!='all':
                    for key in tuple(stats[label]):
                        if key.startswith('false_'):del stats[label][key]
                    stats[label]['pooled_sd_rms']=float(np.sqrt(np.mean(pred[mask]**2)))
            contrasts={}
            for one,two in [('inferred','natural'),('oracle','inferred'),('inferred','zero')]:
                v,b,c=(q-r for q,r in zip(estimates[one],estimates[two]))
                contrasts[one+'_minus_'+two]=dict(skill=v,station95=interval(b),crossed95=interval(c))
            summary['metrics'][subset+'_'+kind]=dict(policies=stats,contrasts=contrasts)
    np.savez_compressed(out/'evaluation.npz',ids=det['ids'],station=station,atoms=atoms,source_dose=truth,source=source,oracle=oracle,
                        natural=nat,inferred=rec,source_input_sd=sd)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--prior',type=Path,default=PRIOR);a=p.parse_args();run(a.out,a.prior)
