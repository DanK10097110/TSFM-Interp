"""Recovery on constrained and unconstrained leads under two seeded architecture nulls."""
import json

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.random_receiver.run import FULL,HERE


def analyze():
    out=HERE/"results"
    with np.load(FULL/"results"/"timesfm_leads.npz") as f:
        truth,ids,active,split=f["source"],f["ids"],f["active"],f["split"]
    report={}
    for domain in ("daily","machines"):
        use=split==domain
        cols=active[use].sum(axis=0)>=8
        mask=active[use][:,cols]
        draws=resampling_counts(int(use.sum()))
        domain_summary=dict(frequent=int(cols.sum()),firing=int(mask.sum()),seeds={})
        bootstraps={}
        for twin in ("pretrained","seed7513","seed9749"):
            with np.load(out/(twin+".npz")) as f,np.load(out/(twin+"_scaled.npz")) as tiny:
                np.testing.assert_array_equal(ids,f["ids"])
                np.testing.assert_array_equal(ids,tiny["ids"])
                np.testing.assert_array_equal(active,f["active"])
                rec=dict(checkpoint_hash=str(f["state_hash"]),radius=float(f["radius"]),fd_error=float(f["fd_error"]),
                         fd_errors=f["fd_errors"].tolist(),fd_steps=f["fd_steps"].tolist(),
                         median_sensitivity=float(np.median(f["output_sensitivity"][use])),
                         cap_count=int(f["adaptive_clipped"][use][:,cols].sum()),
                         median_raw_radius=float(np.median(f["adaptive_raw_norm_fraction"][use][:,cols][mask])))
                beta=float(tiny["scale"])
                rec["scaled_goal_fraction"]=beta
                rec["scaled_clips"]=int(tiny["adaptive_clipped"][use][:,cols].sum())
                signal=beta*truth[use][:,cols,:8][mask].astype(float)
                noise=tiny["no_op"][use][:,cols,:8][mask].astype(float)
                rec["scaled_reference_rms"]=float(np.sqrt(np.mean(signal**2)))
                rec["zero_patch_rms"]=float(np.sqrt(np.mean(noise**2)))
                for arm in ("adaptive","wrong","scaled_adaptive","scaled_wrong"):
                    for subset,leads in [("early",slice(0,8)),("late",slice(8,16))]:
                        ref=truth[use][:,cols,leads].astype(float)
                        if arm.startswith("scaled_"):
                            pred=tiny[arm.removeprefix("scaled_")][use][:,cols,leads].astype(float)/beta
                        else:
                            pred=f[arm][use][:,cols,leads].astype(float)
                        for centered in (False,True):
                            label="_".join((arm,subset,"shape" if centered else "full"))
                            r=ref-ref.mean(axis=-1,keepdims=True) if centered else ref
                            y=pred-pred.mean(axis=-1,keepdims=True) if centered else pred
                            _,mean,boot=aggregate(mask,y,r,draws)
                            rec[label]=ci(mean["skill"],boot["skill"])
                            if twin=="pretrained":
                                bootstraps[label]=(mean["skill"],boot["skill"])
                            else:
                                p,b=bootstraps[label]
                                rec[label+"_minus_pretrained"]=ci(mean["skill"]-p,boot["skill"]-b)
            domain_summary["seeds"][twin]=rec
        if len({rec["checkpoint_hash"] for rec in domain_summary["seeds"].values()})!=3:
            raise ValueError("Twins must have distinct receiver weights")
        report[domain]=domain_summary
    (out/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    analyze()
