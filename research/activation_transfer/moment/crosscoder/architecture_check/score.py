"""Paired, cohort-specific native-dose skill and coverage, retaining closed-gate errors."""
import json

import numpy as np

from research.activation_transfer.moment.crosscoder.architecture_check.run import DOMAINS, OUT

ARMS=('state_exact','direct','explicit_gate','raw32')


def divide(numerator,denominator):
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan,dtype=float),where=denominator>0)


def result(values):
    good=values[1:][np.isfinite(values[1:])]
    return dict(point=float(values[0]) if np.isfinite(values[0]) else None,
                paired95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,finite_draws=int(len(good)))


def macro_skill(errors,reference):
    per=1-divide(errors,reference)
    return divide(np.nansum(per,axis=1),np.isfinite(per).sum(1))


def group_summary(truth,estimates,atoms,cluster):
    # Whole stations on NOAA and one unit per history on the other two cohorts.
    _,group=np.unique(cluster,return_inverse=True)
    n=group.max()+1
    weights=np.random.default_rng(22112431).multinomial(n,np.ones(n)/n,3000)
    weights=np.vstack((np.ones(n),weights))
    def totals(x):
        a=np.zeros((n,x.shape[1]),dtype=float)
        np.add.at(a,group,x)
        return weights@a
    reference=totals(truth**2)
    live=reference[0]>0
    report=dict(n_histories=len(truth),n_bootstrap_clusters=int(n),atoms=atoms.tolist(),active_atoms=int(live.sum()),
                never_fired=atoms[~live].tolist(),true_fires_per_atom=(truth>0).sum(0).tolist(),arms={},contrasts={})
    scores={}
    for arm,est in estimates.items():
        err=totals((truth-est)**2)
        macro=macro_skill(err,reference)
        micro=1-divide(err.sum(1),reference.sum(1))
        scores[arm]=(macro,micro)
        tp=int(((est>0)&(truth>0)).sum());fp=int(((est>0)&(truth==0)).sum());fn=int(((est==0)&(truth>0)).sum())
        per=1-divide(err[0],reference[0])
        report['arms'][arm]=dict(macro_skill=result(macro),pooled_skill=result(micro),tp=tp,fp=fp,fn=fn,
            pred_positive_rate=float(np.mean(est>0)),precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,rmse=float(np.sqrt(np.mean((truth-est)**2))),
            false_alarm_energy_on_never_fired=float(np.sum(est[:,~live]**2)),
            per_atom={str(j):float(v) if np.isfinite(v) else None for j,v in zip(atoms,per)})
    for a,b in [('state_exact','direct'),('explicit_gate','direct'),('state_exact','explicit_gate'),('direct','raw32'),('state_exact','raw32')]:
        report['contrasts'][a+'_minus_'+b]={key:result(x-y) for key,x,y in zip(('macro','pooled'),scores[a],scores[b])}
    return report


def score():
    report={}
    for stem in ('timesfm_to_bolt','bolt_to_timesfm'):
        with np.load(OUT/f'{stem}.npz') as f:
            z=dict(f)
        report[stem]={}
        for domain in DOMAINS:
            sub=z['domain']==domain
            output={}
            for subset in ('nominated','four_nominated','four_global'):
                atoms=z[subset]
                ix=np.array([np.where(z['atoms']==i)[0].item() for i in atoms])
                truth=z['truth'][sub][:,ix].astype(float)
                preds={arm:z[arm][sub][:,ix].astype(float) for arm in ARMS}
                preds['train_mean']=np.broadcast_to(z['truth'][:768,ix].astype(float).mean(0),truth.shape)
                output[subset]=group_summary(truth,preds,atoms,z['cluster'][sub])
            report[stem][domain]=output
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    score()
