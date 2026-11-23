"""Evaluate readouts on the same natural hidden removals, then translate their unseen-donor requests."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.atoms.materials import adapter_for,MODELS
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.full_space.jacobians import SCRATCH as CAL
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,entry,difference
from research.activation_transfer.moment.token_paths.geometry import CACHE,histories
from research.activation_transfer.moment.token_paths.intervene import receiver_write
from research.activation_transfer.moment.head_interventions.headbank import HERE,bank
from research.activation_transfer.moment.head_test.data import SCRATCH as READOUT,QUALITY
from research.activation_transfer.moment.forecast_quality.score import item_metrics,boot_weights,percentile


def prepare(size,seed):
    heads=bank(size)
    with np.load(CACHE/size/f"ablation_{seed}.npz") as f:d=dict(f)
    out={k:d[k] for k in ("ids","domain","atoms","codes","row","col","sd","fd_rel","analytic")}
    for name,(w,b) in heads.items():
        out[name+"_base"]=((d["state0"].astype(float)@w+b)*d["sd"][:,None,None]+d["mu"][:,None,None]).reshape(-1,16)
        out[name+"_effect"]=(d["dstate"].astype(float)@w*d["sd"][:,None,None]).reshape(-1,16)
        out[name+"_estimate"]=(d["derivative"].astype(float)@w*d["sd"][:,None,None]).reshape(-1,16)
        check=(d["alternate_derivative"].astype(float)@w*d["sd"][d["replay"],None,None]).reshape(-1,16)
        est=out[name+"_estimate"][d["replay"]]
        out[name+"_step_check"]=np.linalg.norm(check-est)/max(np.linalg.norm(est),1e-12)
    np.testing.assert_allclose(out["published_effect"],d["physical"],atol=1e-5,rtol=.0005)
    np.savez_compressed(HERE/"results"/f"{size}_{seed}_wave.npz",**out)


def score(size,seed):
    with np.load(HERE/"results"/f"{size}_{seed}_wave.npz") as z:d=dict(z)
    out={}
    for dom in np.unique(d["domain"]):
        p=Panel(d["codes"]>0,d["row"],d["col"],d["domain"]==dom)
        cell=dict(features=int(p.keep.sum()),events=int(p.events.sum()))
        for part,sl in (("first",slice(8)),("next",slice(8,16))):
            for kind in ("shape","full"):
                data={k:component(d[k+"_effect"][:,sl],kind) for k in ("published","interior","terminal","half1","half2")}
                block={}
                for a,b in (("published","interior"),("interior","published"),("half1","half2"),("half2","half1"),("terminal","interior")):
                    block[a+"_vs_"+b]=entry(p.estimate(data[a],data[b]))
                block["energy_rms"]={k:float(np.sqrt(np.mean(v[p.events]**2))) for k,v in data.items()}
                for k in data:
                    e=component(d[k+"_estimate"][:,sl],kind)
                    block[k+"_linear"]=entry(p.estimate(e,data[k]))
                cell[part+"_"+kind]=block
        out[str(dom)]=cell
    (HERE/"results"/f"{size}_{seed}_similarity.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


def ordinary(size):
    heads=bank(size)
    with np.load(READOUT/f"{size}_0/test16.npz") as f:ids,h,sd,mu=f["ids"],f["h"],f["sd"],f["mu"]
    with np.load(QUALITY/"panel.npz") as f:
        np.testing.assert_array_equal(ids,f["id"])
        x,y,domains,split=f["context"],f["future"],f["domain"],f["split"]
    waves={name:((h@w+b)*sd[:,None,None]+mu[:,None,None]).reshape(len(ids),2,16) for name,(w,b) in heads.items()}
    result={}
    for dom in np.unique(domains):
        m=(domains==dom)&(split=="holdout");boot=boot_weights("ordinary-heads/"+str(dom),m.sum())
        vals={k:item_metrics(v[m],y[m],x[m])["mase1"] for k,v in waves.items()}
        res={k:percentile(boot@v) for k,v in vals.items()}
        for a,b in (("published","interior"),("half1","half2"),("terminal","interior")):
            rms=np.sqrt(np.mean(((waves[a][m]-waves[b][m])/x[m].std(2,keepdims=True))**2,axis=2)).mean(1)
            res[a+"_minus_"+b]=percentile(boot@(vals[a]-vals[b]))
            res[a+"_vs_"+b+"_ordinary_rms"]=percentile(boot@rms)
        result[str(dom)]=res
    (HERE/"results"/f"{size}_ordinary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


def transfer(size,seed,target,batch):
    with np.load(HERE/"results"/f"{size}_{seed}_wave.npz") as f:d=dict(f)
    x,ids,split=histories(False)
    row=d["row"]
    np.testing.assert_array_equal(ids,d["ids"])
    with np.load(BLIND/(target+"_early.npz")) as f:j=f["jac"].astype(float)
    with np.load(CAL/(target+"_jac.npz")) as f:
        radius=float(f["radius"])
        cj=f["jac"][f["split"]=="cal"].astype(float)
        mu=cj.mean(1,keepdims=True)
        # Weight fixed from ordinary receiving calibration. Independent of donors and test outputs.
        weight=np.sqrt(np.mean(mu**2)/np.mean((cj-mu)**2))
    h,pos,order=read_materials(BLIND,target);np.testing.assert_array_equal(ids,order)
    adapter=adapter_for(target);site=MODELS[target][-1];sel=last_window_slice(adapter,pos)
    reference=np.zeros_like(h[row,0])
    base=forecast16(adapter,site,x[row],h[row],reference,sel,batch)
    rec=dict(row=row,col=d["col"],ids=ids,atoms=d["atoms"],codes=d["codes"],domain=split,weight=weight,cap=.2*radius)
    for head in ("published","interior","terminal","half1","half2"):
        delta,length,scale=receiver_write(j[row],d[head+"_estimate"][:,:8],.2*radius,weight)
        rec[head]=forecast16(adapter,site,x[row],h[row],delta,sel,batch)-base
        rec[head+"_scale"]=scale
        rec[head+"_source"]=d[head+"_effect"]
    outfile=HERE/"results"/f"{size}_{seed}_to_{target}.npz"
    np.savez_compressed(outfile,**rec)
    result={}
    for dom in np.unique(split):
        p=Panel(d["codes"]>0,row,d["col"],split==dom)
        base_record=dict(features=int(p.keep.sum()),events=int(p.events.sum()))
        for part,sl in (("first",slice(8)),("next",slice(8,16))):
            for kind in ("shape","full"):
                block={}
                for estimate,ref in (("published","published"),("interior","interior"),("published","interior"),
                                     ("terminal","terminal"),("half1","half2"),("half2","half2")):
                    v=p.estimate(component(rec[estimate][:,sl],kind),component(rec[ref+"_source"][:,sl],kind))
                    block[estimate+"_vs_"+ref]=entry(v)
                base_record[part+"_"+kind]=block
        result[str(dom)]=base_record
    outfile.with_suffix(".json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","score","ordinary","transfer"))
    p.add_argument("size");p.add_argument("--seed",type=int,default=22112201);p.add_argument("--target",choices=MODELS)
    p.add_argument("--batch",type=int,default=24);a=p.parse_args();torch.set_num_threads(2)
    if a.stage=="ordinary":ordinary(a.size)
    elif a.stage=="transfer":transfer(a.size,a.seed,a.target,a.batch)
    else:{"prepare":prepare,"score":score}[a.stage](a.size,a.seed)
