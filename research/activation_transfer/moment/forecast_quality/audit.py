"""Independently replay raw observed/future indexing, scores and official short_forecast."""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import torch
from momentfm import MOMENTPipeline

from research.activation_transfer.moment.forecast_quality.data import OUT, SPECS
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER
from research.activation_transfer.moment.forecast_quality.predict import MOMENT_ROOT


def original_sources(panel, rows):
    manifest = json.loads((OUT/"manifest.json").read_text())
    replay = []
    for i in rows:
        domain, item = panel["id"][i].split("/",1)
        source = Path(manifest["sources"][domain]["path"])
        if not source.exists():
            source = SPECS[domain][0]
        with pa.memory_map(str(source)) as src:
            table = pa.ipc.open_stream(src).read_all()
        ids = table["item_id" if "item_id" in table.column_names else "id"].to_pylist()
        raw = np.asarray(table["target"][list(map(str,ids)).index(item)].as_py())
        if raw.ndim==2:
            raw = raw[0]
        for j, cut in enumerate(panel["origin"][i]):
            np.testing.assert_array_equal(raw[cut-480:cut],panel["context"][i,j])
            np.testing.assert_array_equal(raw[cut:cut+16],panel["future"][i,j])
        replay.append(panel["id"][i])
    return replay


def weather(path):
    manifest = json.loads((WEATHER/"manifest.json").read_text())
    with ZipFile(path) as z, np.load(WEATHER/"panel.npz") as p:
        all_count = 0
        contexts, targets = p["context"], p["future"]
        for i, st in enumerate(p["ids"]):
            vals = {}
            for year in (2024,2025,2026):
                member = f"{year}/CRND0103-{year}-{st}.txt"
                for row in z.read(member).splitlines():
                    f = row.split()
                    vals[f[1].decode()] = float(f[8])
            for j,d in enumerate(manifest["origins"]):
                o = datetime.fromisoformat(d)
                for h in range(-480,16):
                    day = o+timedelta(days=h)
                    val = vals.get(day.strftime("%Y%m%d"),-9999.)
                    if val == -9999:
                        assert h<0
                        continue
                    archive = contexts[i,j,480+h] if h<0 else targets[i,j,h]
                    if val != archive:
                        raise AssertionError((st,day,val,archive))
                    all_count +=1
    return dict(verified_measured_context_and_future_values=all_count)


def native(model):
    net = MOMENTPipeline.from_pretrained(str(MOMENT_ROOT/model.split("_")[-1]),strict=True,
                                         model_kwargs={"task_name":"reconstruction"}).eval()
    results = {}
    for target in (OUT,WEATHER):
        with np.load(target/"panel.npz") as f:
            domain = f["domain"] if "domain" in f else f["region"]
            choices = np.flatnonzero(f["split"]=="holdout") if "split" in f else np.arange(len(domain))
            sel = [choices[domain[choices]==name][-1] for name in np.unique(domain)]
            histories = f["context"][sel,-1]
            ids = f["id" if "id" in f else "ids"][sel]
        x = torch.tensor(histories,dtype=torch.float32)
        padded=torch.zeros((len(x),1,512))
        padded[:,0,-480:] = x
        mask=torch.zeros((len(x),512),dtype=torch.long)
        mask[:,-480:]=1
        replays={}
        with np.load(target/(model+".npz")) as f, torch.inference_mode():
            for h in (8,16):
                got=net.short_forecast(x_enc=padded,input_mask=mask,forecast_horizon=h).forecast[:,0].numpy()
                ref=f[f"raw_{h}"][sel,-1]
                max_error=float(np.max(abs(got-ref)/histories.std(1,keepdims=True)))
                if max_error>2e-4:
                    raise AssertionError((target,model,h,max_error))
                replays[h]=max_error
        results["six_domain" if target==OUT else "weather"]=dict(ids=ids.tolist(),max_sd_error_by_horizon=replays)
    return results


def scores():
    s = json.loads((OUT/"summary.json").read_text())["holdout"]["scores"]
    with np.load(OUT/"panel.npz") as p:
        dom=p["domain"]; split=p["split"]; x=p["context"]; future=p["future"]
    gaps=[]
    for model in ("moment_small","moment_base","moment_large","timesfm","bolt"):
        with np.load(OUT/(model+".npz")) as pred:
            for h in (8,16):
                v=pred["raw_"+str(h)]
                for d in np.unique(dom):
                    rows=np.flatnonzero((dom==d)&(split=="holdout"))
                    by_item=[]
                    for i in rows:
                        one=[]
                        for j in (0,1):
                            err=sum(abs(float(v[i,j,t])-float(future[i,j,t])) for t in range(h))/h
                            denom=sum(abs(b-a) for a,b in zip(x[i,j,:-1],x[i,j,1:]))/479
                            one.append(err/denom)
                        by_item.append(sum(one)/2)
                    gap=abs(np.mean(by_item)-s["h"+str(h)]["absolute"][d][model]["mase1"]["value"])
                    gaps.append(gap)
    if max(gaps)>1e-8:
        raise AssertionError(max(gaps))
    return dict(independently_recomputed_mase1_endpoints=len(gaps),max_absolute_error=max(gaps))


def main(snapshot):
    with np.load(OUT/"panel.npz") as p:
        d,s=p["domain"],p["split"]
        rows=np.concatenate([np.flatnonzero((d==k)&(s=="holdout"))[[1,81]] for k in np.unique(d)])
        raw=original_sources(p,rows)
    record=dict(both_origins_from_original_arrays=raw,weather=weather(snapshot),score_replay=scores())
    record["native_model_replay"]={model:native(model) for model in ("moment_small","moment_base","moment_large")}
    (OUT/"audit.json").write_text(json.dumps(record,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    torch.set_num_threads(2)
    main(parser.parse_args().snapshot)
