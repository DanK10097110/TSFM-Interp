"""Station-block and feature-crossed uncertainty for a physically matched SAE replacement."""
import argparse
import json
from pathlib import Path
import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts,terms
from research.activation_transfer.moment.dictionary_transfer.weather.study import ARMS,OUT,SEEDS
from research.activation_transfer.shape_transfer.support_confirmation.noaa.score import eligible_mask


def matched_effect(data,key,shadow=False):
    name='_head' if shadow else ''
    return data[key+name]-data['noop'+name]


def centered(values):
    return values-values.mean(-1,keepdims=True)


def interval(vals):
    finite=vals[np.isfinite(vals)]
    return list(map(float,np.quantile(finite,[.025,.975]))) if len(finite) else [None,None]


def cube(v,rows,cols,shape):
    out=np.zeros((*shape,v.shape[-1]));out[rows,cols]=v;return out


def metrics(sums):
    ref,est,cross,events=np.moveaxis(sums,-1,0)
    ok=(events>0)&(ref>0)
    def divide(num,den,mask):
        return np.divide(num,den,out=np.full_like(num,np.nan),where=mask)
    return dict(skill=divide(2*cross-est,ref,ok),cosine=divide(cross,np.sqrt(ref*est),ok&(est>0)))


def infer(moments,counts,feature_draw):
    per=metrics(moments.sum(0))['skill']
    valid=np.isfinite(per)
    boot=(counts@moments.reshape(len(moments),-1)).reshape(len(counts),*moments.shape[1:])
    arr=metrics(boot)['skill']
    fac=feature_draw*np.isfinite(arr)
    crossed=np.divide(np.nansum(fac*arr,axis=1),fac.sum(1),where=fac.sum(1)>0,out=np.full(len(arr),np.nan))
    return float(per[valid].mean()),np.nanmean(arr,axis=1),crossed,per


def panel(z):
    active=z['z']>0;r,c=z['rows'],z['cols'];station=z['station'];atoms=z['atoms'];sd=z['input_sd'][r,None]
    k=eligible_mask(active,station);sel=np.flatnonzero(k);n=len(sel);eligible=k[c]
    w=resampling_counts(len(np.unique(station)),draws=3000)
    wf=np.random.default_rng(26112283).multinomial(n,np.full(n,1/n),3000) if n else None
    donor=matched_effect(z,'donor');shadow=matched_effect(z,'donor',True);none=z['noop']
    features=[];keep_precision=[]
    for j,atom in enumerate(atoms):
        pick=c==j;co=active[:,j]&(z['target_z'][:,j]>0)
        def rms(v):return float(np.sqrt(np.mean(v**2)))
        energy=rms(centered(donor[pick])) if pick.any() else None
        floor=max(rms(centered(none[pick])),rms(centered(donor[pick]-shadow[pick]))) if pick.any() else None
        passed=bool(energy is not None and energy>10*max(floor,np.finfo(float).eps))
        keep_precision.append(passed)
        record=dict(id=int(atom),match=int(z['matched_ids'][j]),corr_train=float(z['corr'][j]),
                    r2_group=float(z['group_r2'][j]),decoder_cosine=float(z['cos_at_corr'][j]),
                    sites=int(len(set(station[r[pick]]))),events=int(pick.sum()),cofires=int(co.sum()),
                    cofire_sites=int(len(set(station[co]))),sufficient=bool(k[j]),precision=passed,
                    centered_celsius_rms=energy,full_celsius_rms=rms(donor[pick]) if pick.any() else None,noise_celsius_rms=floor)
        features.append(record)
    result=dict(eligible=n,eligible_usable_precision=int(np.sum(k&keep_precision)),total_events=len(r),eligible_events=int(eligible.sum()),
                eligible_stations=int(len(set(station[r[eligible]]))),cofires=int(np.sum(eligible&(z['target_z'][r,c]>0))),
                eligible_events_by_season=[int(np.sum(eligible&(z['origin'][r]==i))) for i in range(4)],features=features,statistics={})
    if not n:return result
    fire=z['target_z'][r,c]>0
    result['hidden_dose_fraction']={}
    for arm in ARMS[1:]:
        observed=z[arm+'_norm']/z['donor_norm']
        result['hidden_dose_fraction'][arm]=dict(all=np.quantile(observed[eligible],[.1,.5,.9]).tolist(),
                                                when_cofiring=np.quantile(observed[eligible&fire],[.1,.5,.9]).tolist() if np.any(eligible&fire) else None)
    for units,raw,truth in [('history_sd',False,donor/sd),('celsius',False,donor),('head_sd',True,shadow/sd)]:
        for shape in ('centered','full'):
            transform=centered if shape=='centered' else lambda v:v
            target=transform(truth)
            goal=cube(target,r,c,active.shape)[:,sel]
            stats={};fits={}
            for key in ARMS[1:]:
                y=matched_effect(z,key,raw)
                if units!='celsius':y=y/sd
                y=transform(y);estimate=cube(y,r,c,active.shape)[:,sel]
                mat=terms(active[:,sel],estimate,goal)
                by=np.stack([mat[station==i].sum(0) for i in np.unique(station)])
                a,b,cc,per=infer(by,w,wf);fits[key]=(a,b,cc)
                metrics_all=metrics(by.sum(0))
                cos=metrics_all['cosine'];ev=eligible
                den=np.linalg.norm(target[ev]);ratio=np.linalg.norm(y[ev])/den if den>0 else np.nan
                def finite(v):return float(v) if np.isfinite(v) else None
                stats[key]=dict(skill=a,station95=interval(b),crossed95=interval(cc),per_atom={str(atoms[j]):finite(val) for j,val in zip(sel,per)},
                                cosine_macro=finite(np.nanmean(cos)) if np.isfinite(cos).any() else None,
                                cosine_measurable=int(np.isfinite(cos).sum()),pooled_rms_ratio=finite(ratio),
                                median_atom_skill=finite(np.nanmedian(per)),atoms_above_zero=int(np.sum(per>0)),
                                atoms_above_half=int(np.sum(per>.5)),atoms_measurable=int(np.isfinite(per).sum()),
                                candidate_pooled_rms=float(np.sqrt(np.mean(y[ev]**2))),
                                donor_pooled_rms=float(np.sqrt(np.mean(target[ev]**2))))
            pairs={}
            for x,y in [('corr_forced','corr_own'),('corr_forced','random'),('group16','corr_forced'),('group16','corr_own')]:
                aa,bb,cc=(v-q for v,q in zip(fits[x],fits[y]))
                pairs[x+'_minus_'+y]=dict(delta=aa,station95=interval(bb),crossed95=interval(cc))
            result['statistics'][units+'_'+shape]=dict(arms=stats,contrasts=pairs)
    ref=centered(donor/sd)
    conditional={}
    for condition,where in [('matched_on',z['target_z'][r,c]>0),('matched_off',z['target_z'][r,c]==0)]:
        vals={}
        for key in ('corr_own','corr_forced','group16'):
            y=centered(matched_effect(z,key)/sd);out=[]
            for j in sel:
                mask=(c==j)&where
                if len(set(station[r[mask]]))>=8 and np.sum(ref[mask]**2)>0:
                    out.append(1-np.sum((ref[mask]-y[mask])**2)/np.sum(ref[mask]**2))
            vals[key]=dict(eligible_atoms=len(out),macro=float(np.mean(out)) if out else None)
        conditional[condition]=dict(events=int(np.sum(eligible&where)),statistics=vals)
    result['conditional']=conditional
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    reports={}
    for seed in SEEDS:
        with np.load(a.out/(str(seed)+'.npz')) as f:reports[str(seed)]=panel(dict(f))
    (a.out/'summary.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
