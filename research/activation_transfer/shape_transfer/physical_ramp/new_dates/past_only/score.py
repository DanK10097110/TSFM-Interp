"""Paired complete-station and complete-date resampling for the same 350 waves."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.score import weights,energy,report
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.past_only.study import OUT,ARMS

def main(out):
    f=dict(np.load(out/'cases.npz'))
    sw,dw=weights(35,26112319),weights(10,26112320)
    samples=((sw,dw),(sw,np.ones_like(dw)),(np.ones_like(sw),dw))
    np.testing.assert_equal(f['station'],np.repeat(np.arange(35),10))
    np.testing.assert_equal(f['date'],np.tile(np.arange(10),35))
    r=dict(n=350,calendar_dates=f['dates'].tolist(),eligible_source_histories=f['admissible'].sum(1).tolist(),
           eligible_original_origins=(f['admissible'].sum(1)//46).tolist(),stats={})
    for name,section in [('early',slice(8)),('late',slice(8,16))]:
        for center in (True,False):
            def get(k):
                y=f[k][:,section].reshape(35,10,8)
                return y-y.mean(axis=2,keepdims=True) if center else y
            goal=get('truth');whole=get('whole')
            den=[energy(goal,s,d) for s,d in samples]
            scores={}
            for arm in (*ARMS,'past_only'):
                err=goal-get(arm)
                scores[arm]=[1-energy(err,s,d)/z for (s,d),z in zip(samples,den)]
            summary={arm:report(draws) for arm,draws in scores.items()}
            for arm in ('past_only','historical'):
                summary['mapped_minus_'+arm]=report([a-b for a,b in zip(scores['mapped'],scores[arm])])
            summary['past_minus_all_four']=report([a-b for a,b in zip(scores['past_only'],scores['historical'])])
            summary['native_isolated_rms_c']=float(np.sqrt(np.mean(goal**2)))
            summary['whole_input_rms_c']=float(np.sqrt(np.mean(whole**2)))
            summary['past_only_rms_c']=float(np.sqrt(np.mean(get('past_only')**2)))
            summary['past_only_error_rms_c']=float(np.sqrt(np.mean((get('past_only')-goal)**2)))
            r['stats'][name+('_centered' if center else '_full')]=summary
    (out/'scores.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);main(p.parse_args().out)
