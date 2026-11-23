"""Native forecast and hidden-displacement recovery, paired by the same seasonal station blocks."""
import argparse
import json
from pathlib import Path
import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.physical_ramp.study import ARM,OUT

def centered(x):
    return x-x.mean(axis=-1,keepdims=True)


def energy(arr,counts,remove_date_template):
    """Sum over reweighted station blocks; re-estimate each date's mean inside the resample."""
    mat=np.asarray(arr,dtype=float)
    square=np.sum(mat*mat,axis=(1,2))
    total=counts@square
    if remove_date_template:
        summed=counts@mat.reshape(len(mat),-1)
        total=total-np.sum(summed*summed,axis=1)/counts.sum(1)
    return np.maximum(total,0.)


def describe(v):
    return dict(value=float(v[0]),station95=np.quantile(v[1:],[.025,.975]).tolist())


def run(source,out):
    f=dict(np.load(out/(source+'_transfer.npz')));station=f['station'];origin=f['origin']
    assert f['truth'].shape==(184,16) and len(np.unique(station))==46
    np.testing.assert_equal(station,np.repeat(np.arange(46),4));np.testing.assert_equal(origin,np.tile(np.arange(4),46))
    w=np.vstack([np.ones(46),resampling_counts(46,draws=3000)])
    vec=f['target_hidden'];refnorm=np.linalg.norm(vec,axis=1);hidden={}
    for k in ARM:
        d=f[k+'_vec'];cross=np.sum(d*vec,axis=1);dn=np.linalg.norm(d,axis=1)
        denom=energy(vec.reshape(46,4,-1),w,False)
        loss=energy((vec-d).reshape(46,4,-1),w,False)
        hidden[k]=dict(skill=describe(1-loss/denom),median_cos=float(np.median(cross/(dn*refnorm))),
                       median_mapped_to_native_norm=float(np.median(dn/refnorm)),max_norm_control_error=float(np.max(abs(dn-f['budget']))))
    report=dict(source=source,target=str(f['target']),histories=184,stations=46,hidden=hidden,statistics={},reference={},zero_vectors=int(np.sum(f['budget']==0)))
    methods=(*ARM,'donor_native')
    for seg,sl in [('early',slice(0,8)),('late',slice(8,16)),('h16',slice(None))]:
        for shape in ('centered','full'):
            def transform(key):
                v=f[key][:,sl].astype(float)
                v=centered(v) if shape=='centered' else v
                return v.reshape(46,4,-1)
            goal=transform('truth');whole=transform('target_whole_input')
            den=energy(goal,w,False);ind=energy(goal,w,True)
            report['reference'][seg+'_'+shape]=dict(isolated_celsius_rms=float(np.sqrt(np.mean(goal**2))),whole_input_celsius_rms=float(np.sqrt(np.mean(whole**2))),
                                  between_station_energy_fraction=float(ind[0]/den[0]),mean_by_date=goal.mean(0).tolist(),
                                  whole_vs_isolated_pooled_cos=float(np.sum(whole*goal)/np.sqrt(np.sum(whole**2)*np.sum(goal**2))))
            for remove in (False,True):
                base=ind if remove else den;fits={};results={}
                for arm in methods:
                    v=transform(arm)
                    residual=goal-v;ratio=1-energy(residual,w,remove)/base
                    fits[arm]=ratio;results[arm]=describe(ratio)
                    if remove:v=v-v.mean(0,keepdims=True)
                    ref=goal-goal.mean(0,keepdims=True) if remove else goal
                    results[arm]['celsius_rms']=float(np.sqrt(np.mean(v*v)))
                    norm=np.linalg.norm(v)*np.linalg.norm(ref)
                    results[arm]['pooled_signed_cosine']=float(np.sum(v*ref)/norm) if norm>0 else None
                for control in ('wrong_step','other_station','donor_native'):
                    results['mapped_minus_'+control]=describe(fits['mapped']-fits[control])
                report['statistics'][seg+'_'+shape+('_between' if remove else '')]=results
    (out/(source+'_scores.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=('bolt','timesfm'));p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.model,a.out)
