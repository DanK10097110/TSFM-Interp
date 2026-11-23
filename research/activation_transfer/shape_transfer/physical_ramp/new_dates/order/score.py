"""Paired 2-by-2 target-specific shape skill on fixed station/date histories."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.order.study import OUT
from research.activation_transfer.shape_transfer.physical_ramp.new_dates.score import energy, weights, report

def balanced_advantage(scores):
    """Equal weight per requested target, each on its own reference energy."""
    return .5*((scores['AA']-scores['AB'])+(scores['BB']-scores['BA']))

def cosine(x,y):
    return np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1))

def main(out):
    f=dict(np.load(out/'order.npz'));ns=35;nd=11
    np.testing.assert_equal(f['station'],np.repeat(np.arange(ns),nd));np.testing.assert_equal(f['origin'],np.tile(np.arange(nd),ns))
    sw,dw=weights(ns,26112319),weights(nd,26112320)
    samples=((sw,dw),(sw,np.ones_like(dw)),(np.ones_like(sw),dw))
    with np.load(out/'bolt_perm.npz') as perm: assert len(perm['ids'])==len(f['ids'])
    cos=cosine(f['AA_vec'],f['BB_vec'])
    result=dict(n=385,stations=35,dates=f['dates'].tolist(),map_sha=str(f['map_sha']),context_sha=str(f['raw_input_sha']),
                hidden_mapped_cosine=dict(median=float(np.median(cos)),q10q90=np.quantile(cos,[.1,.9]).tolist()),
                median_A_to_B_mapped_norm=float(np.median(f['norm_A']/f['norm_B'])),metrics={})
    for seg,sl in (('early',slice(8)),('late',slice(8,16)),('h16',slice(16))):
        for center in (True,False):
            label=seg+('_centered' if center else '_full')
            def array(k):
                v=f[k][:,sl].reshape(ns,nd,-1)
                return v-v.mean(axis=2,keepdims=True) if center else v
            a,b=array('truth_A'),array('truth_B')
            report_by_draw=[];daily={}
            for s,d in samples:
                e={g:energy(v,s,d) for g,v in [('A',a),('B',b)]}
                vals={k:1-energy((a if k[0]=='A' else b)-array(k),s,d)/e[k[0]] for k in ('AA','AB','BA','BB')}
                vals['advantage_A']=vals['AA']-vals['AB'];vals['advantage_B']=vals['BB']-vals['BA']
                vals['balanced_advantage']=balanced_advantage(vals)
                total=(s.sum(axis=1)*d.sum(axis=1)*a.shape[2])
                vals['truth_A_rms_c']=np.sqrt(e['A']/total);vals['truth_B_rms_c']=np.sqrt(e['B']/total)
                vals['native_order_difference_rms_c']=np.sqrt(energy(a-b,s,d)/total)
                vals['whole_order_difference_rms_c']=np.sqrt(energy(array('whole_A')-array('whole_B'),s,d)/total)
                report_by_draw.append(vals)
            for code in ('AA','AB','BA','BB'):
                native=a if code[0]=='A' else b
                daily[code]=1-np.square(native-array(code)).sum(axis=(0,2))/np.square(native).sum(axis=(0,2))
            output={k:report([sample[k] for sample in report_by_draw]) for k in report_by_draw[0]}
            output['per_date_balanced_advantage']=balanced_advantage(daily).tolist()
            output['pooled_reference_cosine']=float(np.sum(a*b)/np.sqrt(np.square(a).sum()*np.square(b).sum()))
            result['metrics'][label]=output
    (out/'scores.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);main(p.parse_args().out)
