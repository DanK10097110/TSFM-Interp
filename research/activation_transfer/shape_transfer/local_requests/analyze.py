"""Paired series bootstrap against the requested waveform and the donor's actual write."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.forecast_requests.run import CUES, desired
from research.activation_transfer.shape_transfer.local_requests.run import HERE, OLD


def pooled_skill(effect, reference, counts):
    sse = np.sum((effect-reference)**2,axis=1)
    energy = np.sum(reference**2,axis=1)
    if min(np.sum(energy),np.min(counts @ energy)) <= 1e-14:
        raise ValueError("Reference has insufficient energy")
    return float(1-sse.sum()/energy.sum()), 1-(counts @ sse)/(counts @ energy)


def summarize(folder):
    records = []
    for target,source in (("bolt","timesfm"),("timesfm","bolt")):
        with np.load(folder/(target+".npz")) as z, np.load(OLD/"results"/(source+".npz")) as s, np.load(
                OLD/"units"/(target+"_effects.npz")) as old:
            ids = z["ids"]
            np.testing.assert_array_equal(ids,s["ids"])
            np.testing.assert_array_equal(ids,old["ids"])
            native = s["0_native"].astype(float)
            arms = {k:z[k].astype(float) for k in ("fixed_pcs","fixed_full","local_pcs","local_full",
                                                "local_odd","local_wrong","recipient_goal","donor_linear")}
            arrays = old["effects"][:,:,0].astype(float)
            arrays[:,old["scales"][:,0]==0] = 0
            arms["clean_calibrated"] = arrays
            caps = {k:z[k+"_clipped"] for k in arms if k+"_clipped" in z.files}
        for domain in ("daily","weekly"):
            use = np.char.startswith(ids,domain+"/")
            n = int(use.sum())
            counts = np.random.default_rng(6721).multinomial(n,np.full(n,1/n),size=4000)
            for i, cue in enumerate(CUES):
                goals = np.broadcast_to(desired()[i],(n,8))
                for subset,lead in [("all",slice(None)),("even",slice(1,None,2))]:
                    for ref_name, ref in [("donor",native[use,i,lead]),("request",goals[:,lead])]:
                        for centered in ((False, True) if (cue!="level" and ref_name=="donor") else (False,)):
                            reference = ref-ref.mean(axis=1,keepdims=True) if centered else ref
                            vals,boot = {},{}
                            for arm, values in arms.items():
                                y = values[use,i,lead]
                                if centered:
                                    y = y-y.mean(axis=1,keepdims=True)
                                vals[arm],boot[arm] = pooled_skill(y,reference,counts)
                            ref_arm = "local_full" if subset=="all" else "local_odd"
                            pairs = {}
                            for arm in arms:
                                if arm!=ref_arm:
                                    pairs[ref_arm+"_minus_"+arm] = dict(value=vals[ref_arm]-vals[arm],
                                        ci95=np.quantile(boot[ref_arm]-boot[arm],[.025,.975]).tolist())
                            records.append(dict(domain=domain,target=target,cue=cue,reference=ref_name,
                                                leads=subset,centered=centered,n=n,
                                                clips={k:int(v[use,i].sum()) for k,v in caps.items()},
                                                methods={k:dict(value=v,ci95=np.quantile(boot[k],[.025,.975]).tolist())
                                                         for k,v in vals.items()},paired=pairs))
    (folder/"results.json").write_text(json.dumps(records,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=HERE/"results")
    summarize(p.parse_args().folder)
