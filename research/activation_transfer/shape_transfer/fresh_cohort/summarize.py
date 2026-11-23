"""Hold fixed previously selected atoms and diagnose which actually fire in the new domains."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.fresh_cohort.data import OUT
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as PREVIOUS

METHODS=("adaptive","fixed","wrong_leads","mean_donor","source_linear")


def eligible(active,minimum):
    counts=np.asarray(active,dtype=bool).sum(axis=0)
    return counts,counts>=minimum


def wave(x,shape):
    x=x.astype(float)
    return x-x.mean(-1,keepdims=True) if shape else x


def score_set(active,atoms,refs,vals,cols,draws):
    if not np.any(cols):
        return {"n_atoms":0,"atom_ids":[],"n_firings":0,"methods":{}}
    result={"n_atoms":int(cols.sum()),"atom_ids":atoms[cols].tolist(),"n_firings":int(active[:,cols].sum()),
            "n_histories_firing":int(active[:,cols].any(1).sum())}
    for shape in (False,True):
        ref=wave(refs[:,cols],shape)
        report={}
        boots={}
        active_subset=active[:,cols]
        for key,v in vals.items():
            per,pt,b=aggregate(active_subset,wave(v[:,cols],shape),ref,draws)
            boots[key]=b["skill"]
            report[key]={"skill":ci(pt["skill"],b["skill"]),"gain":ci(pt["gain"],b["gain"]),
                         "cosine":ci(pt["cosine"],b["cosine"]),
                         "per_atom_skill":{str(atom):float(val) for atom,val in zip(atoms[cols],per["skill"]) if np.isfinite(val)}}
        result["shape" if shape else "full"]={"methods":report,
            "adaptive_minus":{k:ci(report["adaptive"]["skill"]["value"]-report[k]["skill"]["value"],boots["adaptive"]-v)
                              for k,v in boots.items() if k!="adaptive"}}
    return result


def run(out):
    result={}
    for source in ("bolt","timesfm"):
        native=dict(np.load(out/(source+"_native.npz")))
        fresh=dict(np.load(out/(source+"_writes.npz")))
        prev=dict(np.load(PREVIOUS/"results"/(source+".npz")))
        old=dict(np.load(REPLICATION/"results"/(source+"_seed0.npz")))
        archive=json.loads((PREVIOUS/"results/results.json").read_text())[source]
        np.testing.assert_array_equal(native["ids"],fresh["ids"])
        np.testing.assert_array_equal(native["codes"]>0,fresh["active"])
        for z in (native,prev,old):
            np.testing.assert_array_equal(z["atoms"],fresh["atoms"])
        active=fresh["active"]; atoms=fresh["atoms"]
        report={"source":source,"atoms":atoms.tolist(),"shape_weight":float(fresh["shape_weight"]),
                "target_radius":float(fresh["old_target_radius"]),"domains":{}}
        for domain in dict.fromkeys(fresh["split"]):
            mask=fresh["split"]==domain
            sub=active[mask]
            counts=sub.sum(0)
            row={"n":int(mask.sum()),"counts":counts.tolist(),"total_firings":int(sub.sum()),
                 "histories_with_any_selected_feature":int(sub.any(1).sum()),"subsets":{}}
            for minimum in (1,8):
                _,cols=eligible(sub,minimum)
                scored=score_set(sub,atoms,native["native"][mask],{m:fresh[m][mask] for m in METHODS},
                                 cols,resampling_counts(int(mask.sum())))
                scored["cap_count"]={m:int(fresh[m+"_clipped"][mask][:,cols].sum()) for m in METHODS if m!="source_linear"}
                row["subsets"][str(minimum)]=scored
            report["domains"][str(domain)]=row
        newsel=fresh["split"]=="daily"
        oldsel=prev["split"]=="daily"
        oldnative=old["split"]=="daily"
        np.testing.assert_array_equal(prev["ids"][oldsel],old["ids"][oldnative])
        olda=prev["active"][oldsel]; newa=active[newsel]
        cols=eligible(olda,8)[1]&eligible(newa,8)[1]
        ab={"n_atoms":int(cols.sum()),"atom_ids":atoms[cols].tolist(),"old_n":int(oldsel.sum()),
            "new_n":int(newsel.sum()),"published_full":archive["domains"]["daily"]["subsets"]["8"]["full"]["methods"]["adaptive"]["skill"],
            "published_shape":archive["domains"]["daily"]["subsets"]["8"]["shape"]["methods"]["adaptive"]["skill"]}
        # The cohorts are disjoint. Sample them independently for a head-to-head interval.
        rng=np.random.default_rng(2026112603)
        draws=[rng.multinomial(len(v),np.full(len(v),1/len(v)),size=4000) for v in (olda,newa)]
        if cols.any():
            for shape in (False,True):
                samples=[]
                for sub,pred,ref,w in [(olda,prev["adaptive"][oldsel],old["native"][oldnative],draws[0]),
                                       (newa,fresh["adaptive"][newsel],native["native"][newsel],draws[1])]:
                    _,pt,bt=aggregate(sub[:,cols],wave(pred[:,cols],shape),wave(ref[:,cols],shape),w)
                    samples.append((pt["skill"],bt["skill"]))
                (u,ub),(v,vb)=samples
                ab["shape" if shape else "full"]={"old":ci(u,ub),"new":ci(v,vb),"new_minus_old":ci(v-u,vb-ub)}
        report["common_frequent_daily"]=ab
        result[source]=report
    (out/"summary.json").write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
