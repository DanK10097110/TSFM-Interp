"""Measure whether the maps predict the receiving *difference* between the two input orders."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"


def analyze(waves, n, t):
    # Both receivers are evaluated with exactly the same paired station/date draws.
    rng=np.random.default_rng(22112452)
    sw=rng.multinomial(n,np.full(n,1/n),10000)
    tw=rng.multinomial(t,np.full(t,1/t),10000)
    def total(v):
        q=v.reshape(n,t)
        return np.r_[q.sum(),((sw@q)*tw).sum(1)]
    def summarize(a):
        return [float(a[0]),*np.quantile(a[1:],[.025,.975]).tolist()]
    out={}
    for band,sl in (("early",slice(8)),("late",slice(8,16))):
        for mode in ("shape","full"):
            def select(z):
                a=z[:,sl].astype(float)
                return a-a.mean(1,keepdims=True) if mode=="shape" else a
            y=select(waves["truth_B"])-select(waves["truth_A"])
            denom=total((y*y).sum(1))
            assert np.all(denom>0)
            # Exact measured native waveforms provide a scale check for the balanced endpoint.
            native_a = select(waves["truth_A"])
            native_b = select(waves["truth_B"])
            interchange = .5*denom*(1/total((native_a**2).sum(1))+1/total((native_b**2).sum(1)))
            report={"reference_RMS_C":float(np.sqrt((y*y).mean())),
                    "correct_minus_interchanged_actual_native_waves":summarize(interchange)}
            for arm,key_a,key_b in (("correct","AA","BB"),("swapped","AB","BA")):
                z=select(waves[key_b])-select(waves[key_a])
                err=total(((z-y)**2).sum(1))
                norm=total((z*z).sum(1))
                dot=total((z*y).sum(1))
                report[arm]=dict(skill=summarize(1-err/denom),signed_cosine=summarize(dot/np.sqrt(denom*norm)),
                                 predicted_RMS_C=float(np.sqrt((z*z).mean())))
            out[band+"_"+mode]=report
    return out


def main():
    bolt=HERE.parents[3]/"shape_transfer/physical_ramp/new_dates/order/results/order.npz"
    with np.load(bolt) as f:
        ids,dates=f["stations"],f["dates"]
        bw={k:f[k] for k in ("truth_A","truth_B","AA","BB","AB","BA")}
    base=HERE.parent/"new_dates/results"
    paths=dict(truth_A=base/"reference.npz",truth_B=OUT/"permuted_reference.npz",AA=base/"mapped.npz",
               BB=OUT/"permuted_correct.npz",AB=OUT/"ramp_swapped.npz",BA=OUT/"permuted_swapped.npz")
    mw={}
    for name,path in paths.items():
        with np.load(path) as f:
            np.testing.assert_array_equal(ids,f["ids"]);np.testing.assert_array_equal(dates,f["dates"])
            mw[name]=f["wave"]
    out={k:analyze(w,len(ids),len(dates)) for k,w in (("small_to_large",mw),("timesfm_to_bolt",bw))}
    (OUT/"paired_order_difference.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
