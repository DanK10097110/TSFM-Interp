"""Crossed station/date paired comparison of the same physical ramp at different source-information levels."""
import json

import numpy as np

from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates.run import score as original_score
from research.activation_transfer.moment.dictionary_transfer.input_deltas.template.run import OUT

ARMS=("mapped","historical_fixed","historical_matched")


def describe(v):
    return dict(value=float(v[0]),ci95=np.quantile(v[1:],[.025,.975]).tolist())


def measurements(waves,n,t,dates):
    rng=np.random.default_rng(22112418)
    sw=rng.multinomial(n,np.full(n,1/n),3000)
    tw=rng.multinomial(t,np.full(t,1/t),3000)
    def totals(per_case):
        panel=per_case.reshape(n,t)
        return np.r_[per_case.sum(),np.sum((sw@panel)*tw,axis=1)]
    out={}
    for side,lead in (("early",slice(8)),("late",slice(8,16))):
        for mode in ("shape","full"):
            def select(val):
                vals=val[:,lead].astype(float)
                return vals-vals.mean(1,keepdims=True) if mode=="shape" else vals
            y=select(waves["reference"])
            energy=(y**2).sum(1)
            yss=totals(energy)
            assert (yss>0).all()
            rec={};skills={};rss={}
            for arm in ARMS:
                candidate=select(waves[arm])
                error=candidate-y
                rss[arm]=(error**2).sum(1)
                losses=totals(rss[arm])
                skills[arm]=1-losses/yss
                rec[arm]=dict(skill=describe(skills[arm]),rms_c=float(np.sqrt(np.mean(candidate**2))),
                              rmse_c=describe(np.sqrt(losses/(8*n*t))),
                              mean_abs_error_c=describe(totals(abs(error).sum(1))/(8*n*t)))
            contrasts={}
            for a,b in (("mapped","historical_matched"),("mapped","historical_fixed"),("historical_matched","historical_fixed")):
                contrasts[a+"_minus_"+b]=describe(skills[a]-skills[b])
            den=energy.reshape(n,t)
            better=(rss["historical_matched"]-rss["mapped"]).reshape(n,t)
            leave_station=(better.sum()-better.sum(1))/(den.sum()-den.sum(1))
            leave_date=(better.sum()-better.sum(0))/(den.sum()-den.sum(0))
            per_date=better.sum(0)/den.sum(0)
            out[side+"_"+mode]=dict(arms=rec,paired=contrasts,reference_rms_c=float(np.sqrt((y*y).mean())),
                mapped_minus_matched_by_date=dict(zip(map(str,dates),map(float,per_date))),
                mapped_minus_matched_leave_station_range=[float(leave_station.min()),float(leave_station.max())],
                mapped_minus_matched_leave_date_range=[float(leave_date.min()),float(leave_date.max())])
    return out


def main():
    with np.load(OUT/"waves.npz") as f:
        n,t=len(f["ids"]),len(f["dates"])
        waves={k:f[k] for k in ("reference",*ARMS)}
        result=measurements(waves,n,t,f["dates"])
        # Identical draws and denominators to the registered new-panel evaluator; only alias the labels here.
        other=original_score(dict(reference=waves["reference"],mapped=waves["mapped"],
                                  wrong_edit=waves["historical_fixed"],other_station=waves["historical_matched"]),n,t)
        for cell,new in result.items():
            for key,ref in (("mapped","mapped"),("historical_fixed","wrong_edit"),("historical_matched","other_station")):
                got=new["arms"][key]["skill"]
                np.testing.assert_allclose([got["value"],*got["ci95"]],other[cell][ref],rtol=1e-10,atol=1e-10)
            got=new["paired"]["mapped_minus_historical_matched"]
            np.testing.assert_allclose([got["value"],*got["ci95"]],other[cell]["mapped_minus_other_station"],atol=1e-10)
        def quantile(v):
            return np.quantile(v,[0,.1,.5,.9,1]).tolist()
        hidden=dict(historical_mean_norm=float(np.linalg.norm(f["historical_mean"])),
            old_individual_norm_quantiles=quantile(f["historical_norm"]),
            new_individual_norm_quantiles=quantile(f["mapped_norm"]),current_vs_template_cosine_quantiles=quantile(f["template_cosine"]))
    (OUT/"summary.json").write_text(json.dumps(dict(stations=n,dates=t,physical_cases=n*t,hidden=hidden,outcomes=result),
                                              indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
