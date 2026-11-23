"""Compare fixed and history-specific translation in the same original steering task."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import ALIGN, DATA_ROOT, FIT, TEST, adapter_for, data, desired
from research.activation_transfer.shape_transfer.full_space.transfer import solve_fixed, solve_adaptive
from research.activation_transfer.shape_transfer.token_position.run import CACHE
from tsfm_lens.steering.model import InterventionSite, _get, _replace

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "forecast_requests"
SCRATCH = Path("/tmp/tsfm_local_requests_20261120")


def local_forecast(adapter, site, x, delta, with_grad=False):
    calls = []
    def hook(module, args, kwargs):
        packed = dict(args=args, kwargs=kwargs)
        state = _get(packed, site.path)
        if list(state.shape) != [len(x), *site.shape]:
            raise ValueError("Unexpected middle-block geometry")
        changed = state.clone()
        changed[:, -1] = state[:, -1] + delta
        packed = _replace(packed, site.path, changed)
        calls.append(True)
        return packed["args"], packed["kwargs"]
    with adapter.module.get_submodule(site.module).register_forward_pre_hook(hook, with_kwargs=True):
        if with_grad:
            y = adapter.predict_differentiable(x, 8)
        else:
            y = adapter.predict(x, 8, [.5])["quantiles"][...,0]
    if len(calls) != 1:
        raise ValueError("Expected a single middle-block invocation")
    return y


def fit(name, scratch, batch):
    ids, x = data(DATA_ROOT)
    sel = np.r_[np.arange(FIT.start,FIT.stop), np.arange(TEST.start,len(ids))]
    x = x[sel]
    adapter = adapter_for(name)
    with np.load(CACHE/(name+"_fit.npz")) as old:
        np.testing.assert_array_equal(ids,old["ids"])
        site = InterventionSite(**json.loads(str(old["metadata"]))["site"][0])
        train = old["s0"][ALIGN].astype(float)
        donor_dirs = old["d0"]
    radius = np.median(np.linalg.norm(train,axis=1))
    basis = np.linalg.svd(train-train.mean(axis=0),full_matrices=False)[2][:24]
    jac, base = [], []
    for i in range(0,len(x),batch):
        obs = x[i:i+batch]
        displace = torch.zeros((len(obs),site.width),dtype=torch.float32,requires_grad=True)
        y = local_forecast(adapter,site,obs,displace,with_grad=True)
        grad = [torch.autograd.grad(y[:,lead].sum(),displace,retain_graph=lead<7)[0] for lead in range(8)]
        jac.append(torch.stack(grad,dim=1).numpy())
        ordinary = adapter.predict(obs,8,[.5])["quantiles"][...,0]
        np.testing.assert_allclose(y.detach().numpy(),ordinary,atol=5e-5,rtol=1e-5)
        base.append(ordinary)
    jac = np.concatenate(jac)
    rng = np.random.default_rng(911)
    check = np.r_[0:4,64:68,192:196]
    u = rng.standard_normal((len(check),site.width))
    u *= (.002*radius / np.linalg.norm(u,axis=1))[:,None]
    actual = (local_forecast(adapter,site,x[check],torch.from_numpy(u).float())
              - local_forecast(adapter,site,x[check],-torch.from_numpy(u).float()))/2
    approx = np.einsum("nld,nd->nl",jac[check],u)
    error = np.linalg.norm(actual-approx) / np.linalg.norm(actual)
    if error > .08:
        raise ValueError(f"Finite difference error {error}")
    scratch.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(scratch/(name+".npz"),ids=ids[sel],jac=jac,basis=basis,
                        donor_dirs=donor_dirs,base=np.concatenate(base),radius=radius,fd_error=error)


def execute(name, scratch, out, batch):
    source = "timesfm" if name=="bolt" else "bolt"
    ids,x = data(DATA_ROOT)
    sel = np.r_[np.arange(FIT.start,FIT.stop), np.arange(TEST.start,len(ids))]
    nfit = FIT.stop-FIT.start
    with np.load(scratch/(source+".npz")) as s, np.load(scratch/(name+".npz")) as t:
        np.testing.assert_array_equal(s["ids"],t["ids"])
        np.testing.assert_array_equal(ids[sel],t["ids"])
        responses = s["jac"].astype(float) @ s["donor_dirs"].T
        jt, basis, radius = t["jac"].astype(float), t["basis"], float(t["radius"])
        base, fd_error = t["base"][nfit:], float(t["fd_error"])
    with np.load(CACHE/(name+"_fit.npz")) as old:
        site = InterventionSite(**json.loads(str(old["metadata"]))["site"][0])
    adapter = adapter_for(name)
    x = x[TEST]
    rr = responses[nfit:]
    order = [3,6,1,7,0,2,5,4]
    arms = ("fixed_pcs","fixed_full","local_pcs","local_full","local_odd","local_wrong","recipient_goal")
    records = dict(ids=ids[TEST], donor_linear=rr.transpose(0,2,1),fd_error=fd_error)
    for arm in arms:
        design = jt @ basis.T if arm.endswith("pcs") else jt
        if arm.startswith("fixed"):
            write = solve_fixed(design[:nfit],responses[:nfit])
            write = np.broadcast_to(write,(len(x),*write.shape))
        else:
            target = design[nfit:]
            if arm == "local_odd":
                write = solve_adaptive(target[:,::2],rr[:,::2])
            elif arm == "local_wrong":
                write = solve_adaptive(target[:,order],rr)
            elif arm == "recipient_goal":
                write = solve_adaptive(target,np.broadcast_to(desired().T,rr.shape))
            else:
                write = solve_adaptive(target,rr)
        if arm.endswith("pcs"):
            write = basis.T @ write
        vectors = write.transpose(0,2,1)
        norms = np.linalg.norm(vectors,axis=2)
        scale = np.minimum(1., .2*radius/np.maximum(norms,1e-15))
        vectors = (vectors*scale[:,:,None]).astype("f4")
        predictions = []
        for i in range(0,len(x)*3,batch):
            index = np.arange(i,min(i+batch,len(x)*3))
            rows, cues = np.divmod(index,3)
            y = local_forecast(adapter,site,x[rows],torch.from_numpy(vectors[rows,cues]))
            predictions.append(y-base[rows])
        records[arm] = np.concatenate(predictions).reshape(len(x),3,8)
        records[arm+"_clipped"] = scale<1.
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(name+".npz"),**records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["fit","apply"])
    p.add_argument("model",choices=["bolt","timesfm"])
    p.add_argument("--scratch",type=Path,default=SCRATCH)
    p.add_argument("--out",type=Path,default=HERE/"results")
    p.add_argument("--batch",type=int)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=="fit":
        fit(a.model,a.scratch,a.batch or 8)
    else:
        execute(a.model,a.scratch,a.out,a.batch or 32)
