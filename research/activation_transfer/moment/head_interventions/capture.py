"""Record raw final-token states before and after each selected MOMENT middle-layer feature removal."""
import argparse
import numpy as np
import torch

from research.activation_transfer.moment.token_paths.geometry import CACHE,histories,load,layers
from research.activation_transfer.moment.horizon_identity.train import HERE as SAE,SEEDS
from research.activation_transfer.moment.head_interventions.headbank import HERE
from research.activation_transfer.moment.head_test.capture import scaling
from tsfm_lens.extraction.hooks import ActivationCatcher,token_patch
from tsfm_lens.sae.train import load_sae_checkpoint


def run(size,seed,batch):
    adapter=load(size)
    x,ids,split=histories(False)
    cache=CACHE/size
    with np.load(cache/"eval.npz") as a:
        np.testing.assert_array_equal(a["ids"],ids)
        mid=a["middle"][:,0]
        jac=a["middle_jac"][:,:,0,:]
    sae=load_sae_checkpoint(str(cache/f"sae_{seed}.pt"))
    with np.load(SAE/"results"/f"{size}_{seed}_dictionary.npz") as f:atoms=f["ids"]
    with torch.no_grad():
        z=sae.encode(torch.from_numpy(mid))[:,atoms].numpy()
    row,col=np.nonzero(z>0)
    vec=-z[row,col,None]*sae.W_dec.detach().numpy()[atoms[col]]
    stage=layers(adapter)["middle"]
    d=mid.shape[-1]
    def physical(select,dose):
        states=[];forecasts=[];means=[];scales=[]
        for lo in range(0,len(select),batch):
            idx=select[lo:lo+batch];r=row[idx]
            hidden=(mid[r]+dose*vec[idx])[:,None]
            with token_patch(adapter.module,stage,lambda _:slice(61,62),torch.as_tensor(hidden)),ActivationCatcher(adapter.module,["encoder.final_layer_norm"]) as cap:
                f=adapter.predict(x[r],16,[.5])["point"]
                sf=cap.collect()["encoder.final_layer_norm"][:,-2:].detach().numpy()
            mu,sd=scaling(adapter.module.normalizer)
            states.append(sf);forecasts.append(f);means.append(mu);scales.append(sd)
        return np.concatenate(states),np.concatenate(forecasts),np.concatenate(means),np.concatenate(scales)
    all_events=np.arange(len(row))
    state0,f0,mu,sd=physical(all_events,0.)
    state1,f1,_,_=physical(all_events,1.)
    plus,_,_,_=physical(all_events,.01)
    minus,_,_,_=physical(all_events,-.01)
    # Float64 subtraction and division reduce extra loss of derivative precision.
    derivative=(plus.astype(float)-minus.astype(float))/.02
    analytic=np.einsum("nhd,nd->nh",jac[row],vec)
    W=adapter.module.head.linear.weight.detach().numpy()
    approx=(derivative@W.T*sd[:,None,None]).reshape(-1,16)
    fd_rel=np.linalg.norm(approx-analytic)/np.linalg.norm(analytic)
    # Some source features are weak; retain per-event discrepancies in addition to the aggregate.
    replay=np.random.default_rng(22112231).choice(all_events,min(32,len(row)),replace=False)
    plus2,_,_,_=physical(replay,.02)
    minus2,_,_,_=physical(replay,-.02)
    probe2=(plus2.astype(float)-minus2.astype(float))/.04
    target=(state1.astype(float)-state0.astype(float))@W.T*sd[:,None,None]
    np.testing.assert_allclose(target.reshape(-1,16),f1-f0,atol=1e-5,rtol=.0005)
    print(size,seed,len(row),"fd relative",fd_rel,flush=True)
    # All readouts can now be evaluated from the same physical final hidden displacement.
    np.savez_compressed(cache/f"ablation_{seed}.npz",ids=ids,domain=split,atoms=atoms,codes=z,row=row,col=col,
                        state0=state0,dstate=(state1.astype(float)-state0).astype("f4"),derivative=derivative.astype("f4"),
                        mu=mu,sd=sd,plain=f0,physical=f1-f0,analytic=analytic,fd_rel=fd_rel,
                        replay=replay,alternate_derivative=probe2.astype("f4"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size");p.add_argument("seed",type=int,choices=SEEDS)
    p.add_argument("--batch",type=int,default=16);p.add_argument("--threads",type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);run(a.size,a.seed,a.batch)
