"""Physical input equivalence, evaluated on actual source knockout deltas."""
import json

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.blind_frequency.score import scores
from research.activation_transfer.shape_transfer.input_bridge.probe import CAP, HERE, WINDOW
from research.activation_transfer.shape_transfer.input_bridge.run import ARMS

CONTRASTS = (("t_donor", "t_scrambled"), ("t_recipient", "t_donor"),
             ("t_joint", "t_donor"), ("s_donor", "t_donor"), ("hidden_recipient", "t_recipient"))


def interval(estimate):
    p, s, c = estimate[:3]
    return dict(value=float(p), fixed_atom_series95=np.nanquantile(s, [.025,.975]).tolist(),
                crossed95=np.nanquantile(c,[.025,.975]).tolist())


def summarize(model):
    with np.load(HERE/"results"/(model+".npz")) as f:
        r = dict(f)
    data = {}
    for domain in np.unique(r["split"]):
        rows = r["split"]==domain
        eligible = r["active"][rows].sum(0)>=8
        select = lambda a: a[rows][:, eligible]
        mask = select(r["active"])
        n = mask.shape[1]
        item = dict(n_series=int(rows.sum()), atoms=r["atoms"][eligible].tolist(),
                    n_active=int(mask.sum()), counts=mask.sum(0).tolist(), dose={})
        data[domain] = item
        if not n:continue
        for k in ARMS:
            unbounded = select(r[k+"_raw_norm"])[mask].astype(float) / np.sqrt(WINDOW)
            actual = np.linalg.norm(select(r[k+"_coeff"])[mask].astype(float), axis=-1) / np.sqrt(WINDOW)
            item["dose"][k] = dict(clipped=int((unbounded>CAP/np.sqrt(WINDOW)).sum()),
                                   requested_rms_q10_q50_q90=np.quantile(unbounded,[.1,.5,.9]).tolist(),
                                   actual_rms_q10_q50_q90=np.quantile(actual,[.1,.5,.9]).tolist())
        sw = resampling_counts(len(mask), 4000)
        fw = np.random.default_rng(20261121093).multinomial(n, np.full(n,1/n), size=len(sw))
        ref = select(r["native"]).astype(float)
        for part, slc in (("early", slice(0,8)), ("late", slice(8,16))):
            for scale in ("centered","full"):
                def view(v):
                    arr = v[..., slc]
                    return arr-arr.mean(axis=-1,keepdims=True) if scale=="centered" else arr
                truth = view(ref)
                raw = {}
                arms = [role+"_"+m for role in ("s","t") for m in ARMS]+["hidden_recipient"]
                if part=="early":
                    arms += ["requested"]+[role+"_linear_"+k for role in ("s","t") for k in ARMS]
                for arm in arms:
                    fx = select(r[arm]).astype(float)
                    raw[arm] = scores(view(fx), truth, mask, sw, fw)
                if part=="early":
                    for role,inv in (("s","donor"),("t","recipient")):
                        name=role+"_linear_"+inv
                        pre_norm=select(r[inv+"_raw_norm"]).astype(float)
                        inflate=np.maximum(1.,pre_norm/CAP)
                        fx=select(r[name]).astype(float)*inflate[...,None]
                        raw[name+"_uncapped"]=scores(view(fx),truth,mask,sw,fw)
                table = {k:interval(v) for k,v in raw.items()}
                for k,v in raw.items():
                    table[k]["by_atom"]=[float(q) if np.isfinite(q) else None for q in v[3]]
                for a,b in CONTRASTS:
                    res = [raw[a][i]-raw[b][i] for i in range(3)]
                    row = interval(res)
                    if (a,b)==CONTRASTS[0] and part=="early" and scale=="centered":
                        row["family_lower"] = float(np.nanquantile(res[2], .05/8))
                    table[a+"_minus_"+b] = row
                if part=="early" and scale=="centered":
                    table["t_donor"]["family_lower"] = float(np.nanquantile(raw["t_donor"][2], .05/8))
                item[part+"_"+scale] = table
    return dict(domain=data, donor_matched_zero_max=float(r["s_max_zero_rebatch"]),
                recipient_matched_zero_max=float(r["t_max_zero_rebatch"]), shape_weight=float(r["weight"]))


if __name__ == "__main__":
    data = {m:summarize(m) for m in ("bolt","timesfm")}
    (HERE/"results"/"summary.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
