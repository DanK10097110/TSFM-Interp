"""Feature-normalized accuracy versus natural removal and versus the actual dense recipient edit."""
import argparse
import json

import numpy as np

from research.activation_transfer.atoms.scores import resampling_counts
from research.activation_transfer.shape_transfer.blind_frequency.score import scores
from research.activation_transfer.shape_transfer.dictionary_prior.receiver_encoding.run import ARMS,HERE


def interval(raw):
    p,ss,ff=raw[:3]
    return dict(value=float(p),series95=np.nanquantile(ss,[.025,.975]).tolist(),
                crossed95=np.nanquantile(ff,[.025,.975]).tolist())


def analyze(source, matched=False):
    with np.load(HERE/"results"/(source+".npz")) as f:a=dict(f)
    measures=ARMS
    comparisons=(("locked","hard"),("shoulder","hard"),("span","hard"))
    if matched:
        with np.load(HERE/"results"/(source+"_matched.npz")) as f:
            for name in ("ids","atoms","rows","cols"):np.testing.assert_array_equal(a[name],f[name])
            a.update({k:f[k] for k in f.files if k not in a})
        measures=("dense","hard","locked","hard_equal","locked_equal")
        comparisons=(("locked_equal","hard_equal"),("hard_equal","hard"),("locked_equal","locked"),("locked","hard"))
    rows,cols=a["rows"],a["cols"]
    shape=a["active"].shape
    def fill(v):
        out=np.zeros((*shape,*v.shape[1:]),dtype=v.dtype);out[rows,cols]=v
        return out
    effects={k:fill(a[k+"_effect"]).astype(float) for k in measures}
    nenter=fill(a["enter"])
    out=dict(max_archive_difference=float(a["max_dense_reference_error"]),max_rebatch_zero=float(a["zero_rebatch_max"]),limit=float(a["cap"]),domains={})
    for domain in np.unique(a["split"]):
        histories=a["split"]==domain
        result={}
        out["domains"][domain]=result
        for subgroup,edge in (("all",a["active"]),("crossed",a["active"]&(nenter>0)),("unchanged",a["active"]&(nenter==0))):
            valid=edge[histories].sum(0)>=8
            m=edge[histories][:,valid]
            full_domain_eligible=a["active"][histories].sum(0)>=8
            item=dict(n_histories=int(histories.sum()),n_atoms=int(valid.sum()),atoms=a["atoms"][valid].tolist(),
                      n_cases=int(m.sum()),counts=m.sum(0).tolist(),
                      raw_stratum_cases_across_full_domain_eligible_atoms=int(edge[histories][:,full_domain_eligible].sum()))
            result[subgroup]=item
            if not valid.any():continue
            k=int(valid.sum())
            sw=resampling_counts(len(m),4000)
            fw=np.random.default_rng(20261121095).multinomial(k,np.full(k,1/k),size=len(sw))
            use=histories[rows]&valid[cols]&edge[rows,cols]
            d=a["dense_norm"][use]
            val=lambda x:[float(v) for v in np.quantile(x,[.1,.5,.9])]
            item["gate"]=dict(with_entries=int((a["enter"][use]>0).sum()),
                with_exits=int((a["leave"][use]>0).sum()),entry_q10_q50_q90=val(a["enter"][use]),
                exit_q10_q50_q90=val(a["leave"][use]),clean_support_q10_q50_q90=val(a["clean_count"][use]))
            item["geometry"]={}
            for key in measures:
                length=a[key+"_norm"][use]
                entry=dict(capped=int(a[key+"_cap"][use].sum()),mean_postnorm=float(np.mean(length)),
                           postnorm_q10_q50_q90=val(length),pre_norm_q10_q50_q90=val(a[key+"_pre_norm"][use]),
                           ratio_to_dense_q10_q50_q90=val(length/np.maximum(d,1e-15)),
                           cosine_to_dense_q10_q50_q90=val(a[key+"_cos"][use]))
                if key+"_zero" in a:entry["zero_directions"]=int(a[key+"_zero"][use].sum())
                if key=="dense":entry["unbounded_q10_q50_q90"]=val(a["full_dense_norm"][use])
                item["geometry"][key]=entry
            for name,y in (("donor",a["native"].astype(float)),("recipient",effects["dense"])):
                by_ref={};item[name]=by_ref
                for part,leads in (("early",slice(0,8)),("late",slice(8,16))):
                    for scale in ("full","centered"):
                        def view(v):
                            q=v[histories][:,valid,leads]
                            return q-q.mean(-1,keepdims=True) if scale=="centered" else q
                        truth=view(y)
                        energy=(m*np.sum(truth**2,axis=-1)).sum(0)
                        table=dict(reference_energy_by_atom=energy.tolist(),eligible_nondegenerate=int((energy>1e-10*m.sum(0)*8).sum()))
                        if not table["eligible_nondegenerate"]:
                            by_ref[part+"_"+scale]=table
                            continue
                        raw={key:scores(view(effects[key]),truth,m,sw,fw) for key in measures}
                        for key,rr in raw.items():
                            rt=interval(rr);rt["by_atom"]=[float(v) if np.isfinite(v) else None for v in rr[3]]
                            table[key]=rt
                        for aa,bb in comparisons:
                            diff=[raw[aa][i]-raw[bb][i] for i in range(3)]
                            stat=interval(diff)
                            if (aa,bb)==comparisons[0] and (name,part,scale,subgroup)==("recipient","early","centered","all"):
                                stat["family_lower"]=float(np.nanquantile(diff[2],.05/4))
                            table[aa+"_minus_"+bb]=stat
                        by_ref[part+"_"+scale]=table
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched",action="store_true")
    mode=parser.parse_args().matched
    summary={m:analyze(m,mode) for m in ("bolt","timesfm")}
    (HERE/"results"/("matched_summary.json" if mode else "summary.json")).write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
