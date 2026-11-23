"""Independently estimate signed and scale-sensitive transfer metrics from the finite forecast archives."""
import json
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent/"results"
SEEDS=(22112201,22112202)


def summarize(estimate,reference,row,col,eligible):
    values=[]
    for i in eligible:
        mask=col==i
        x=estimate[mask].astype(float);y=reference[mask].astype(float)
        xx=np.sum(x*x);yy=np.sum(y*y);xy=np.sum(x*y)
        if yy<=1e-10*y.size:continue
        values.append((1-np.sum((x-y)**2)/yy,xy/max(np.sqrt(xx*yy),1e-18),2*xy/max(xx+yy,1e-18),
                       np.sqrt(yy/y.size),np.sqrt(xx/y.size)))
    if not values:raise ValueError("No feature above the numerical energy floor")
    sk,cos,sym,truth,candidate=np.array(values).T
    return dict(macro=float(sk.mean()),median=float(np.median(sk)),positive=int((sk>0).sum()),features=len(sk),
                median_pooled_direction_cosine=float(np.median(cos)),median_symmetric_agreement=float(np.median(sym)),
                median_source_RMS=float(np.median(truth)),median_candidate_RMS=float(np.median(candidate)))


def run():
    report={}
    for source in ("small","large"):
        for target in ("small","large"):
            for seed in SEEDS:
                stem=f"{source}_{seed}_to_{target}"
                with np.load(OUT/f"{stem}_forecast.npz") as f:d=dict(f)
                summaries=json.loads((OUT/f"{stem}_forecast.json").read_text())
                g=OUT/f"{stem}_group_waves.npz";group={}
                if source==target or seed==SEEDS[0]:
                    with np.load(g) as f:
                        np.testing.assert_array_equal(d["row"],f["row"]);np.testing.assert_array_equal(d["col"],f["col"])
                        group=dict(f)
                    group_sums=json.loads((OUT/f"{stem}_group_scores.json").read_text())
                row,col=d["row"],d["col"]
                natural=np.nonzero(d["z"]>0)
                np.testing.assert_array_equal(natural,(row,col))
                if len(np.unique(d["ids"]))!=len(d["ids"]):raise AssertionError("Repeated item ID")
                zeros=d["corr_own_length"]==0
                np.testing.assert_array_equal(d["native_corr_own"][zeros],np.zeros_like(d["native_corr_own"][zeros]))
                mat={}
                for domain in sorted(set(d["split"])):
                    mask=d["split"][row]==domain
                    eligible=np.flatnonzero(np.bincount(col[mask],minlength=d["z"].shape[1])>=8)
                    by_head={}
                    for head in ("native","interior"):
                        for part,sl in (("early",slice(8)),("late",slice(8,16))):
                            truth=d["truth_"+head][mask,sl].astype(float)
                            truth=truth-truth.mean(1,keepdims=True);rec={}
                            for k in ("dense","corr_own","corr_forced","geom_forced","random","response","one","four","sixteen","gated","dose"):
                                if head+"_"+k in d:
                                    est=d[head+"_"+k];known=summaries[str(domain)]
                                elif k+"_"+head in group:
                                    est=group[k+"_"+head];known=group_sums[str(domain)]
                                else:continue
                                est=est[mask,sl].astype(float);est-=est.mean(1,keepdims=True)
                                val=summarize(est,truth,row[mask],col[mask],eligible)
                                np.testing.assert_allclose(known[f"{head}_{part}_shape"][k]["value"],val["macro"],atol=1e-9)
                                rec[k]=val
                            by_head[head+"_"+part]=rec
                    mat[str(domain)]=by_head
                report[stem]=mat
    (OUT/"overview.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":run()
