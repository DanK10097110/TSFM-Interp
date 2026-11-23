"""Compare true fixed and history-specific target feature edits with the dense translation."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.scores import aggregate, ci, resampling_counts
from research.activation_transfer.replication.data import HERE as REPLICATION
from research.activation_transfer.shape_transfer.sparse_writes.run import ARMS, EXTRA, FULL, HERE


def analyze(out):
    report = {}
    for source in ("timesfm","bolt"):
        with np.load(out/(source+".npz")) as new, np.load(REPLICATION/"results"/(source+"_seed0.npz")) as base, np.load(
                REPLICATION/"results"/(source+"_coactive.npz")) as old,np.load(FULL/"results"/(source+".npz")) as dense:
            ev = base["split"]!="cal"
            for a in (old,base):
                np.testing.assert_array_equal(a["ids"][ev],new["ids"])
            np.testing.assert_array_equal(new["ids"],dense["ids"])
            np.testing.assert_array_equal(new["atoms"],old["source_atoms"])
            arm = {k:new[k] for k in ARMS}
            arm.update(coactivation=old["coactive"][ev],dense=dense["adaptive"])
            reference,active,split,atoms = base["native"][ev],new["active"],new["split"],new["atoms"]
            fixed,chosen = new["fixed"],new["local_removal_ids"][:,:,0]
            n_candidates = new["candidate_count"]
            signed_choices = new["one_signed_ids"][:,:,0]
        with np.load(out/(source+"_identity.npz")) as extra:
            np.testing.assert_array_equal(extra["atoms"],atoms)
            np.testing.assert_array_equal(extra["active"],active)
            np.testing.assert_array_equal(extra["split"],split)
            arm.update({k:extra[k] for k in EXTRA})
            available = extra["fixed_signed_ids"][:,:,0]>=0
            signed_fixed = extra["fixed"]
        result = {}
        for dom in ("daily","machines"):
            use = split==dom
            draws = resampling_counts(int(use.sum()))
            result[dom] = {}
            for minimum in (1,8):
                cols = active[use].sum(axis=0)>=minimum
                firing = active[use][:,cols]
                ids = chosen[use][:,cols]
                features = atoms[cols]
                per_id = []
                for j,feat in enumerate(features):
                    choices = ids[firing[:,j],j]
                    values,counts = np.unique(choices[choices>=0],return_counts=True)
                    per_id.append(dict(source_atom=int(feat),fit_target_atom=int(fixed[cols][j]),
                                       firing=len(choices),no_op=int(np.sum(choices==-1)),distinct=len(values),
                                       matches_fixed=int(np.sum((choices>=0)&(choices==fixed[cols][j]))),
                                       modal_count=int(np.max(counts,initial=0)),
                                       modal_atom=int(values[counts.argmax()]) if len(values) else None))
                n_choice = sum(k["firing"]-k["no_op"] for k in per_id)
                cell = dict(n_atoms=len(features),n=int(use.sum()),firing=int(firing.sum()),by_atom=per_id,
                            literal_patches=n_choice,candidate_range=[int(n_candidates[use].min()),int(n_candidates[use].max())],
                            fixed_signed_target_available=float(available[use][:,cols][firing].mean()),
                            signed_choice_matches_fit=float((signed_choices[use][:,cols]==signed_fixed[cols])[firing].mean()),
                            conditional_fit_id_match=sum(k["matches_fixed"] for k in per_id)/max(n_choice,1),
                            modal_share=sum(k["modal_count"] for k in per_id)/max(n_choice,1))
                for center in (False,True):
                    ref = reference[use][:,cols].astype(float)
                    if center:
                        ref -= ref.mean(axis=-1,keepdims=True)
                    entries,bs = {},{}
                    for k,z in arm.items():
                        y = z[use][:,cols].astype(float)
                        if center:
                            y -= y.mean(axis=-1,keepdims=True)
                        per,pt,bt = aggregate(firing,y,ref,draws)
                        entries[k] = dict(skill=ci(pt["skill"],bt["skill"]),cosine=ci(pt["cosine"],bt["cosine"]))
                        bs[k] = bt["skill"]
                    for a,b in [("local_removal","fixed_removal"),("local_removal","coactivation"),
                                ("one_signed","fixed_signed"),("one_signed","partial_removal"),("partial_removal","local_removal"),
                                ("four_signed","one_signed"),("four_signed","permuted_four"),("four_signed","dense")]:
                        entries[a+"_minus_"+b] = ci(entries[a]["skill"]["value"]-entries[b]["skill"]["value"],bs[a]-bs[b])
                    cell["shape" if center else "full"] = entries
                result[dom][str(minimum)] = cell
        report[source] = result
    (out/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"results")
    analyze(p.parse_args().out)
