"""Score actual held-out donor and receiver forecasts on shared history bootstraps."""
import json

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.shape_transfer.continuation.probe import HERE,SCRATCH

WRONG = ["wrong"+str(i) for i in range(4)]
ARMS = ["minimum","clean_only","clean_early","paired_only","paired_early","oracle",*WRONG]
CONTRASTS = [("clean_early","minimum"),("paired_early","minimum"),("oracle","minimum"),
             ("clean_early","wrong_mean"),("clean_early","clean_only"),("paired_early","paired_only")]


def score(f,domain):
    idx = f["split"]==domain
    counts = f["active"][idx].sum(axis=0)
    cols = counts>=8
    m = f["active"][idx][:,cols]
    B = resampling_counts(len(m))
    out = dict(n=len(m),atoms=f["atoms"][cols].tolist(),firing_counts=counts[cols].tolist(),active=int(m.sum()))
    norm = {}
    for key in ARMS:
        raw = f[key+"_norm"][idx][:,cols][m]
        clipped = f[key+"_clipped"][idx][:,cols][m]
        np.testing.assert_array_equal(clipped, raw>.2)
        norm[key] = dict(clipped=int(clipped.sum()),raw_q10_50_90=np.quantile(raw,[.1,.5,.9]).tolist(),
                         actual_q10_50_90=np.quantile(np.minimum(raw,.2),[.1,.5,.9]).tolist())
    out["norm_fractions_of_radius"] = norm
    all_effects = {k:f[k][idx][:,cols].astype(float) for k in ["source",*ARMS]}
    # Post-hoc dose attribution: rescale the *observed* minimum-write effect in forecast space.
    # This is not an intervention unless the receiver is linear over that displacement.
    raw_min = f["minimum_norm"][idx][:,cols].astype(float)
    min_scale = np.minimum(1.,.2/np.maximum(raw_min,1e-20))
    shrink_controls = {}
    for arm in ("clean_early","paired_early"):
        raw = f[arm+"_norm"][idx][:,cols].astype(float)
        ratio = np.minimum(1.,.2/np.maximum(raw,1e-20))/min_scale
        shrink_controls[arm] = all_effects["minimum"]*ratio[:,:,None]
    for segment,sl in [("early",slice(0,8)),("late",slice(8,16))]:
        out[segment] = {}
        for scale in ("full","centered"):
            effects = {k:v[:,:,sl] for k,v in all_effects.items()}
            if scale=="centered":
                effects = {k:v-v.mean(axis=-1,keepdims=True) for k,v in effects.items()}
            point,draws = {},{}
            for key in ARMS:
                _,p,b = aggregate(m,effects[key],effects["source"],B)
                point[key],draws[key] = p["skill"],b["skill"]
            point["wrong_mean"] = np.mean([point[k] for k in WRONG])
            draws["wrong_mean"] = np.mean([draws[k] for k in WRONG],axis=0)
            estimates = {k:ci(v,draws[k]) for k,v in point.items()}
            for a,b in CONTRASTS:
                samples = draws[a]-draws[b]
                interval = ci(point[a]-point[b],samples)
                if segment=="late" and scale=="centered" and a in ("clean_early","paired_early") and b=="minimum":
                    interval["primary_fwer_lower"] = float(np.nanquantile(samples,.00625))
                estimates[a+"_minus_"+b] = interval
            posthoc = {}
            for arm, control in shrink_controls.items():
                baseline = control[:,:,sl]
                if scale=="centered":
                    baseline = baseline-baseline.mean(axis=-1,keepdims=True)
                _,p,b = aggregate(m,baseline,effects["source"],B)
                posthoc[arm] = dict(attenuated_minimum=ci(p["skill"],b["skill"]),
                                    actual_minus_attenuated=ci(point[arm]-p["skill"],draws[arm]-b["skill"]))
            estimates["posthoc_forecast_only_cap_rescaling"] = posthoc
            out[segment][scale] = estimates
    return out


def main():
    out = {}
    for name in ("bolt","timesfm"):
        with np.load(HERE/"results"/(name+".npz")) as f,np.load(SCRATCH/(name+"_jac16.npz")) as j:
            report = dict(weight=float(f["weight"]),radius=float(f["radius"]),min_replay=float(f["min_replay_error"]),
                          unbounded_clean_prior_replay=float(f["clean_prior_replay_error"]),
                          model_jacobian=dict(clean_max_error=float(j["clean_max_error"]),
                              early_relative_error=float(j["early_relative_error"]),fd_n=len(j["fd_indices"]),
                              early_late_fd_relative=j["fd_relative"].tolist()))
            for domain in ("daily","machines"):
                report[domain] = score(f,domain)
            out[name] = report
    (HERE/"results"/"scores.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
