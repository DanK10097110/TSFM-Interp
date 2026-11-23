"""Independent published-file sample and black-box ragged forecast checks on calendar M4."""
import csv
import json
from hashlib import sha256,file_digest
import numpy as np
import torch
from research.activation_transfer.moment.calendar.run import OUT,SOURCE,GROUPS
from research.activation_transfer.moment.forecast_quality.predict import adapter,checkpoint


def data():
    report={}
    for group,(H,period) in GROUPS.items():
        with np.load(OUT/f"{group}_panel.npz") as f:d=dict(f)
        manifest=json.loads((OUT/f"{group}_manifest.json").read_text())
        elig=[];lookup={k:i for i,k in enumerate(d["ids"])};matches=0
        for stage in ("train","test"):
            path=SOURCE/f"{group}-{stage}.csv"
            with path.open("rb") as f:assert file_digest(f,"sha256").hexdigest()==manifest[stage+"_sha256"]
            with path.open() as f:
                reader=csv.reader(f);next(reader)
                for row in reader:
                    key=row[0]
                    if stage=="test" and key not in lookup:continue
                    v=np.fromiter((float(x) for x in row[1:] if x),float)
                    if stage=="train":
                        if len(v)<=period or not np.isfinite(v).all() or np.array_equal(v[period:],v[:-period]):continue
                        elig.append(key)
                        if key in lookup:
                            i=lookup[key]
                            np.testing.assert_equal(v[-480:],d["x"][i][np.isfinite(d["x"][i])])
                            np.testing.assert_allclose(sum(abs(v[k]-v[k-period]) for k in range(period,len(v)))/(len(v)-period),d["scale"][i])
                    else:np.testing.assert_array_equal(d["truth"][lookup[key]],v);matches+=1
        chosen=sorted(elig,key=lambda k:sha256(("moment-calendar-20261122|"+k).encode()).hexdigest())[:256]
        np.testing.assert_array_equal(chosen,d["ids"])
        assert matches==len(chosen)
        report[group]=dict(source_histories=len(elig),traced_items=len(chosen),future_values=matches*H)
    (OUT/"source_audit.json").write_text(json.dumps(report,indent=2)+"\n")


def models(key):
    # Public APIs receive ragged original observations rather than the study's dense padded wrapper.
    if key=="timesfm":
        import timesfm
        m=adapter(key)
        cfg=timesfm.ForecastConfig(max_context=480,max_horizon=32,per_core_batch_size=4,
                   normalize_inputs=False,force_flip_invariance=False,infer_is_positive=False,
                   use_continuous_quantile_head=False,fix_quantile_crossing=False)
        m.tfm.compile(cfg)
    elif key=="bolt":m=adapter(key)
    else:
        from momentfm import MOMENTPipeline
        m=MOMENTPipeline.from_pretrained(str(checkpoint(key)[2]),strict=True,model_kwargs={"task_name":"reconstruction"}).eval()
    receipt={}
    for group,(H,period) in GROUPS.items():
        with np.load(OUT/f"{group}_panel.npz") as f:x=f["x"];ids=f["ids"]
        with np.load(OUT/f"{group}_{key}.npz") as f:saved=f["native"]
        order=np.argsort(np.isfinite(x).sum(1),kind="stable")
        selected=order[np.linspace(0,len(order)-1,12,dtype=int)]
        seq=[a[np.isfinite(a)].astype("f4") for a in x[selected]]
        with torch.inference_mode():
            if key=="timesfm":
                _,q=m.tfm.forecast(H,list(seq));pred=q[:,:,5]
            elif key=="bolt":
                q,_=m.pipeline.predict_quantiles([torch.from_numpy(s) for s in seq],prediction_length=H,quantile_levels=[.5])
                pred=q[...,0].numpy()
            else:
                # Match native 512-slot call in a different batch/loop using finite values and an explicit native mask.
                pred=[]
                for s in seq:
                    n=len(s);v=torch.zeros(1,1,512);v[0,0,-n:]=torch.from_numpy(s)
                    mask=torch.zeros(1,512,dtype=torch.long);mask[:,-n:]=1
                    pred.append(m.short_forecast(x_enc=v,input_mask=mask,forecast_horizon=H).forecast[0,0].numpy())
                pred=np.stack(pred)
        val=(pred-saved[selected])/np.array([s.std() for s in seq])[:,None]
        # Bolt batches of genuinely different physical length may change effective relative-position bias.
        receipt[group]=dict(max_difference_sd=float(np.abs(val).max()),rms_difference_sd=float(np.sqrt(np.mean(val**2))),
                            min_length=min(map(len,seq)),max_length=max(map(len,seq)),ids=ids[selected].tolist())
    (OUT/f"{key}_api_audit.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage");a=p.parse_args();torch.set_num_threads(1)
    if a.stage=="data":data()
    else:models(a.stage)
