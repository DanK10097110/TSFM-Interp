"""Hold the entire forecasting computation fixed and swap only the active SAE dictionary."""
import argparse
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS,adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.dictionary_prior.run import blind_match, correct, inverse, load_target
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import HERE as FULL
from research.activation_transfer.shape_transfer.response_subspace.run import HERE as SUBSPACE, CHECKPOINTS
from research.activation_transfer.shape_transfer.sparse_stability.run import active_codes
from research.activation_transfer.shape_transfer.sparse_writes.run import weight_leads
from tsfm_lens.sae.train import load_sae_checkpoint

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
ARMS=("natural","attenuated","corrected","dense","hidden_single","hidden_span")


def project(vector, directions):
    """Signed best single-axis and active-subspace approximations in the shared hidden units."""
    norms=(directions*directions).sum(axis=1)
    dot=directions@vector
    best=int(np.argmax(dot*dot/norms))
    single=directions[best]*(dot[best]/norms[best])
    coef=np.linalg.lstsq(directions.T, vector,rcond=1e-7)[0]
    return best,single,coef@directions


def run(model,source_seed,receiver_seed,out,batch):
    data=load_target(model)
    with np.load(SUBSPACE/"results"/f"{model}_seed{source_seed}.npz") as f:
        use=np.isin(f["split"],["daily","extra_daily"])
        np.testing.assert_array_equal(f["ids"][use],data["ids"])
        active,codes=f["active"][use],f["codes"][use]
        atoms=f["atoms"]
        native=f["native"][use,:,:,1]
    donor=load_sae_checkpoint(str(CHECKPOINTS/f"seed{source_seed}_{model}.pt"))
    receiver=load_sae_checkpoint(str(CHECKPOINTS/f"seed{receiver_seed}_{model}.pt"))
    src_d=donor.W_dec.detach().numpy()[atoms].astype(float)
    dec=receiver.W_dec.detach().numpy().astype(float)
    support,all_codes=active_codes(receiver,data["states"])
    with np.load(FULL/"results"/(model+".npz")) as f:
        weight=float(f["shape_weight"])
    row,atom=np.nonzero(active)
    n=len(row)
    rec=dict(ids=data["ids"],split=data["split"],atoms=atoms,active=active,rows=row,atom_index=atom,
             source=native[row,atom],natural_id=np.zeros(n,int),attenuated_id=np.zeros(n,int),hidden_id=np.zeros(n,int),
             single_fraction=np.zeros(n),span_fraction=np.zeros(n))
    for arm in ARMS:
        rec[arm]=np.empty((n,16),dtype="f4")
        rec[arm+"_capped"]=np.empty(n,dtype=bool)
    enc=adapter_for(model)
    sel=last_window_slice(enc,data["positions"])
    for start in range(0,n,batch):
        take=np.arange(start,min(start+batch,n))
        vectors={arm:[] for arm in ARMS}
        for i in take:
            r,k=row[i],atom[i]
            ids=support[r]
            goal=native[r,k,:8].astype(float)
            jac=data["jac"][r]
            d=dec[ids]
            strength=all_codes[r,ids].astype(float)
            full=-strength[:,None]*d
            ranks,aranks,fractions,_=blind_match(full@jac.T,goal,weight)
            best,abest=ranks[0],aranks[0]
            wj,lift=inverse(jac,weight)
            weighted_goal=weight_leads(goal,weight)
            v=-codes[r,k]*src_d[k]
            win,single,span=project(v,d)
            rec["natural_id"][i],rec["attenuated_id"][i],rec["hidden_id"][i]=ids[best],ids[abest],ids[win]
            for key,p in (("single_fraction",single),("span_fraction",span)):
                rec[key][i]=(p@p)/(v@v)
            if source_seed==receiver_seed:
                np.testing.assert_allclose(span,v,rtol=1e-7,atol=1e-7)
            prior=fractions[abest]*full[abest]
            moves=dict(natural=full[best],attenuated=prior,corrected=correct(prior,wj,lift,weighted_goal),
                       dense=lift@weighted_goal,hidden_single=single,hidden_span=span)
            for arm,v in moves.items():
                scale=min(1.,.2*data["radius"]/max(np.linalg.norm(v),1e-16))
                rec[arm+"_capped"][i]=scale<1.
                vectors[arm].append(v*scale)
        r=row[take]
        for arm,vs in vectors.items():
            y=forecast16(enc,MODELS[model][-1],data["x"][r],data["states"][r],np.array(vs,dtype="f4"),sel,batch)
            rec[arm][take]=y-data["base"][r]
    if source_seed==receiver_seed:
        ok=~rec["hidden_span_capped"]
        np.testing.assert_allclose(rec["hidden_span"][ok],rec["source"][ok],atol=4e-4,rtol=.004)
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/f"{model}_{source_seed}_{receiver_seed}.npz",**rec)
    print(model,source_seed,receiver_seed,n,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",choices=MODELS)
    p.add_argument("source",type=int,choices=(1,2))
    p.add_argument("receiver",type=int,choices=(1,2))
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.model,a.source,a.receiver,a.out,a.batch)
