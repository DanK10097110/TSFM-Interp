"""Actually apply the native writes, transported writes and observed-shift displacements."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import CUES, HERE, adapter_for, desired, live_effect, scale_factors
from research.activation_transfer.shape_transfer.tangents.capture import CACHE, NFIT, ORIG, OUT
from research.activation_transfer.shape_transfer.tangents.maps import ARMS
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite


def changing_effect(model,site,x,norms,moves):
    """One dose vector per history and arm, run on clean inputs using the original norm-scaled hook."""
    n,k,width=moves.shape
    base=np.concatenate([model.forecast(x[i:i+32],8)[...,0] for i in range(0,n,32)])
    result=[]
    for i in range(0,n*k,32):
        flat=np.arange(i,min(i+32,n*k))
        rows,arms=flat//k,flat%k
        basis=moves[rows,arms]
        coeff=np.diag(1/np.maximum(norms[rows],1e-6))
        after=model.evaluate(x[rows],8,site,basis,coeff)[...,0]
        result.append(after-base[rows])
    return np.concatenate(result).reshape(n,k,8)


def run(target,cache,original,out):
    source="timesfm" if target=="bolt" else "bolt"
    inputs=dict(np.load(cache/"inputs.npz"))
    tests=inputs["ids"][NFIT:]
    x=inputs["x"][NFIT:]
    b=dict(np.load(cache/(target+"_deltas.npz")))
    patch=dict(np.load(cache/(target+"_patches.npz")))
    fitted=dict(np.load(original/(target+"_fit.npz")))
    info=json.loads(str(fitted["metadata"]))["site"][0]
    site=InterventionSite(**info)
    target_norm=np.linalg.norm(b["clean"][NFIT:],axis=1)
    cal_norm=np.linalg.norm(b["cal_clean"],axis=1)
    model=SteeringModel.from_adapter(adapter_for(target),quantiles=[.5])
    directions=np.stack([patch["write_"+k] for k in ARMS])
    dwidth=directions.shape[-1]
    # Match the original three-request calibration batch geometry for every map.
    cal=np.stack([live_effect(model,site,inputs["cal"],cal_norm,arm,32) for arm in directions],axis=1)
    doses=np.stack([scale_factors(cal[:,i],desired()) for i in range(len(ARMS))])
    archived=json.loads((HERE/"results"/(target+".json")).read_text())
    legacy_index=ARMS.index("legacy")
    prior_doses=np.array(archived["applied_dose"]["0_"+source])
    repeated_doses=doses[legacy_index].copy()
    # This comparator replays the published policy, including the original scalar.
    doses[legacy_index]=prior_doses
    acts=np.stack([directions,directions*doses[:,:,None]],axis=1)
    flat=acts.reshape(-1,dwidth)
    all_acts=np.vstack([flat,b["direction"]])
    effects=live_effect(model,site,x,target_norm,all_acts,32)
    native=effects[:,-3:]
    measured=effects[:,:-3].reshape(len(x),len(ARMS),2,3,8)
    natural_keys=["native",*ARMS,"raw"]
    shifts=np.stack([b["delta"][NFIT:,5] if k=="native" else patch["shift_"+k] for k in natural_keys],axis=1)
    natural=changing_effect(model,site,x,target_norm,shifts)
    idx=inputs["test_index"]
    with np.load(HERE/"results"/(target+".npz")) as prior, np.load(HERE/"results"/(source+".npz")) as donor:
        np.testing.assert_array_equal(tests,prior["ids"][idx])
        np.testing.assert_array_equal(tests,donor["ids"][idx])
        key="0_"+source
        original_source=donor["0_native"][idx]
        ref=prior[key][idx,:,:2].transpose(0,2,1,3)
        errors={"native_max_abs":float(np.abs(native-prior["0_native"][idx]).max()),
                "legacy_max_abs":float(np.abs(measured[:,legacy_index]-ref).max())}
        np.testing.assert_allclose(native,prior["0_native"][idx],atol=3e-4,rtol=1e-5)
        np.testing.assert_allclose(measured[:,legacy_index],ref,atol=3e-4,rtol=1e-5)
    errors["repeated_legacy_calibration"]=repeated_doses.tolist()
    errors["archived_legacy_calibration"]=prior_doses.tolist()
    errors["legacy_dose_max_abs"]=float(np.abs(repeated_doses-prior_doses).max())
    out.mkdir(exist_ok=True,parents=True)
    stem=source+"_to_"+target
    np.savez_compressed(out/(stem+"_forecasts.npz"),ids=tests,split=inputs["split"][NFIT:],arms=ARMS,
                        cues=CUES,effects=measured.astype("f4"),target_native=native.astype("f4"),
                        source_native=original_source,requests=desired(),doses=doses,
                        natural_keys=natural_keys,natural=natural.astype("f4"),
                        calibration_effects=cal.astype("f4"),calibration_ids=inputs["cal_ids"])
    (out/(stem+"_replay.json")).write_text(json.dumps(errors,indent=2)+"\n")
    print(stem,"complete",errors,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=("bolt","timesfm"))
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--original",type=Path,default=ORIG)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    torch.set_num_threads(2)
    run(a.target,a.cache,a.original,a.out)
