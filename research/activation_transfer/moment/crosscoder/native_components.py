"""Check level agreement, common-coordinate uncertainty and the influence of small response norms."""
import json
import numpy as np
import torch

from research.activation_transfer.moment.crosscoder import dictionaries as cd
from research.activation_transfer.moment.crosscoder.directions import OUT, read_views
from research.activation_transfer.moment.crosscoder.patch_probe import crossed


def run():
    x,ids,domains,states,jac = read_views()
    result={}
    for method in cd.METHODS:
        for seed in cd.SEEDS:
            name=f"{method}_{seed}"
            with np.load(cd.CACHE/(name+".npz")) as f:
                ds,dl=f["decoder_small"],f["decoder_large"]
            with np.load(OUT/(name+".npz")) as f:
                np.testing.assert_array_equal(ids,f["ids"])
                atoms,dose,matched,off=f["atom"],f["dose"],f["paired"],f["within_history"]
            source=ds[atoms]@jac["small"].transpose(0,2,1)
            target=dl[atoms]@jac["large"].transpose(0,2,1)
            s=np.sign(source.mean(-1)); t=np.sign(target.mean(-1))
            assert np.all(s!=0) and np.all(t!=0)
            same=(s==t).astype(float); k=atoms.shape[1]
            other=((k+s*t.sum(1,keepdims=True))/2-same)/(k-1)
            result[name]={}
            for d in np.unique(domains):
                ix=np.flatnonzero(domains==d)
                both=crossed(dict(shape=matched[ix],shape_above_distractor=(matched-off)[ix],
                    level_sign=same[ix],level_sign_above_distractor=(same-other)[ix]),
                    atoms[ix],None,False)
                scale=dose[ix,:,None]/x[ix].std(1)[:,None,None]
                sur=scale*source[ix];tar=scale*target[ix]
                shape_s=sur-sur.mean(-1,keepdims=True);shape_t=tar-tar.mean(-1,keepdims=True)
                rms=np.sqrt(np.mean(shape_s**2,axis=-1))
                bins=np.quantile(rms,[0,.25,.5,.75,1])
                by_energy=[]
                for i in range(4):
                    use=(rms>=bins[i]) & (rms<bins[i+1] if i<3 else rms<=bins[i+1])
                    by_energy.append(dict(events=int(use.sum()),paired=float(matched[ix][use].mean()),
                        paired_minus_other=float((matched-off)[ix][use].mean())))
                both["source_rms_quartiles"]=bins.tolist();both["by_source_rms"]=by_energy
                for key,a,b in (("level",sur.mean(-1),tar.mean(-1)),("shape",shape_s,shape_t)):
                    both[key+"_scale"]=dict(small_rms=float(np.sqrt(np.mean(a*a))),large_rms=float(np.sqrt(np.mean(b*b))),
                        native_skill=float(1-((a-b)**2).sum()/(a*a).sum()))
                result[name][str(d)]=both
    (OUT/"components.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2);run()
