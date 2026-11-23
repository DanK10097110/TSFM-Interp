"""Independently recompute plotted comparisons and readout differences from the saved measurements."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
SEEDS=(22112401,22112402)


def unit(x):
    z=x.astype(float)-x.astype(float).mean(-1,keepdims=True)
    return z/np.linalg.norm(z,axis=-1,keepdims=True)


def comparison(source,target,tokens):
    """Materialize each complete pairwise cosine matrix, then remove its diagonal from the control."""
    s=unit(source[tokens]); t=unit(target[tokens])
    all_pairs=s@np.swapaxes(t,-2,-1)
    match=np.diagonal(all_pairs,axis1=1,axis2=2)
    others=(all_pairs.sum(2)-match)/(tokens.shape[1]-1)
    return match-others


def run():
    summary=json.loads((OUT/"overview"/"comparisons.json").read_text())
    max_error=0.;depth_error=0.;cell_count=0;depth_count=0
    for method in ("joint","masked","shuffled"):
        for seed in SEEDS:
            name=f"{method}_{seed}"
            for pair in summary:
                if pair=="MOMENT":
                    with np.load(OUT/"directions"/(name+".npz")) as f:
                        ids,domains,tokens=f["ids"],f["domains"],f["atom"]
                        future=f["paired"]-f["within_history"]
                    with np.load(OUT/"patch_probe"/("all_"+name+".npz")) as f:
                        np.testing.assert_array_equal(f["ids"][:len(ids)],ids)
                        past=comparison(f["source"],f["target"],tokens)
                else:
                    base=HERE/"cross_family_directions"/"results"
                    with np.load(base/(name+"_dict.npz")) as m, np.load(base/"ridge.npz") as r, np.load(base/(name+"_directions.npz")) as f:
                        ids,domains,tokens=f["ids"],f["domains"],f["atom"]
                        small=(m["source_decoder"].astype(float)*m["scale"][0])@r["timesfm_probe"]
                        large=(m["target_decoder"].astype(float)*m["scale"][1])@r["bolt_probe"]
                        past=comparison(small,large,tokens)
                        future=f["forecast16_paired"]-f["forecast16_other"]
                assert len(set(ids))==256 and tokens.shape==(256,32)
                for domain in np.unique(domains):
                    keep=domains==domain
                    for k,v in (("past",past),("forecast",future),("difference",past-future)):
                        recorded=summary[pair][str(domain)][name][k]["mean"]
                        err=abs(recorded-float(v[keep].mean()));max_error=max(max_error,err)
                        assert err<2e-9,(name,pair,domain,k,err)
                        cell_count+=1
            root=OUT/"depth_trace"
            with np.load(root/(name+".npz")) as f:
                results=json.loads((root/(name+".json")).read_text())
                for domain in np.unique(f["domains"]):
                    mask=f["domains"]==domain
                    margins={}
                    for d in ("middle","upper","final","native"):
                        src=unit(f["small_"+d][mask])
                        good=unit(f["paired_"+d][mask]);bad=unit(f["decoy_"+d][mask])
                        margins[d]=np.mean(np.sum(src*(good-bad),axis=1))
                        error=abs(margins[d]-results[str(domain)][d]["advantage"][0])
                        depth_error=max(depth_error,error);assert error<2e-10;depth_count+=1
                    np.testing.assert_allclose(margins["final"]-margins["native"],
                        results[str(domain)]["final_probe_minus_native_advantage"][0],atol=2e-10)
    record=dict(figure_cells=cell_count,max_absolute_figure_difference=max_error,depth_contrasts=depth_count,
        max_absolute_depth_difference=depth_error)
    (OUT/"overview"/"audit.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    print(record)


if __name__=="__main__":run()
