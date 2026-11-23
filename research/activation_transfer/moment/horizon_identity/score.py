"""Separate position, future-token attention and sparse support in same-input horizon comparisons."""
import argparse
import json
import numpy as np

from research.activation_transfer.moment.horizon_identity.train import HERE,SEEDS
from research.activation_transfer.shape_transfer.steering_budget.score import Panel,component,difference,entry
from research.activation_transfer.atoms.scores import resampling_counts


def scalar(x,weights):
    return dict(mean=float(x.mean()),ci95=np.quantile(weights@x/weights.sum(1),[.025,.975]).tolist(),
                median=float(np.median(x)),p95=float(np.quantile(x,.95)))


def fractional(part,total,weights):
    den=np.sum(total**2,axis=1)
    num=np.sum(part*total,axis=1)
    return dict(value=float(num.sum()/den.sum()),ci95=np.quantile((weights@num)/(weights@den),[.025,.975]).tolist())


def run(size):
    out=HERE/"results";report={}
    with np.load(out/(size+"_factors.npz")) as f:g=dict(f)
    for domain in np.unique(g["domain"]):
        m=g["domain"]==domain
        w=resampling_counts(m.sum(),draws=3000)
        y={k:g[k][m,:8].astype(float) for k in ("native8","native16","position8","one_mask","both8")}
        total=y["native16"]-y["native8"]
        position=((y["one_mask"]-y["both8"])+(y["native16"]-y["position8"]))/2
        mask=((y["position8"]-y["both8"])+(y["native16"]-y["one_mask"]))/2
        interaction=y["native16"]-y["one_mask"]-y["position8"]+y["both8"]
        rms=lambda a:np.sqrt(np.mean(a*a,axis=1))
        cell=dict(n=int(m.sum()),forecast_difference_rms=scalar(rms(total),w),
                  peak_difference=scalar(np.max(np.abs(total),axis=1),w),
                  position_rms=scalar(rms(position),w),attention_rms=scalar(rms(mask),w),
                  interaction_rms=scalar(rms(interaction),w),position_share=fractional(position,total,w),
                  attention_share=fractional(mask,total,w),factorial_replay_max=float(np.max(np.abs(y["both8"]-y["native8"]))),
                  seeds={})
        for seed in SEEDS:
            filename=out/f"{size}_{seed}_ablate.npz"
            if not filename.exists():continue
            with np.load(filename) as f:d=dict(f)
            np.testing.assert_array_equal(d["ids"],g["ids"])
            anchor=d["support_native8"][m]
            gates={}
            for arm in ("native16","position8","one_mask","both8"):
                other=d["support_"+arm][m]
                changed=np.array([len(set(a)-set(b)) for a,b in zip(anchor,other)])
                gates[arm]=dict(n_switched=int(np.sum(changed>0)),mean_exits=scalar(changed,w),
                    relative_state_change=scalar(d["distance_"+arm][m]/d["norm_native8"][m],w))
            panel=Panel(d["z8"]>0,d["row"],d["col"],m)
            record=dict(gates=gates,eligible_features=int(panel.keep.sum()),events=int(panel.events.sum()))
            if panel.keep.any():
                live=panel.events;rr=d["row"][live];cc=d["col"][live]
                record["recipient_same_id_fires"]=int(np.sum(d["z16"][rr,cc]>0))
                for kind in ("full","shape"):
                    ref=component(d["native"][:,:8],kind)
                    v={arm:panel.estimate(component(d[arm][:,:8],kind),ref) for arm in ("same_vector","same_feature","wrong_position")}
                    record[kind]={arm:entry(a) for arm,a in v.items()}
                    record[kind]["same_feature_minus_vector"]=difference(v["same_feature"],v["same_vector"])
                    record[kind]["wrong_position_minus_vector"]=difference(v["wrong_position"],v["same_vector"])
                record["rms_by_arm"]={arm:float(np.sqrt(np.mean(d[arm][live,:8].astype(float)**2)))
                                      for arm in ("native","same_vector","same_feature","wrong_position")}
            cell["seeds"][seed]=record
        report[str(domain)]=cell
    (out/f"{size}_summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("size");run(p.parse_args().size)
