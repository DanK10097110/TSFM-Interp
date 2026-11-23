"""Capture paired responses to generic input edits and genuine observed window shifts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from research.activation_transfer.forecast_requests.run import ALIGN, CAL, DATA_ROOT, TEST, adapter_for, data
from research.activation_transfer.steering.study import sites_and_states
from tsfm_lens.steering import SteeringModel

CACHE=Path("/tmp/tsfm_shape_tangents_v1")
ORIG=Path("/tmp/tsfm_forecast_requests_20261120")
OUT=Path(__file__).resolve().parent/"results"
NTRAIN=512
NVALID=128
NFIT=NTRAIN+NVALID
FEATURES=("recent_bump","early_bump","smooth_trend","recent_oscillation","shift2","shift4")


def variations(item):
    seed=int.from_bytes(sha256(("2026112501/"+str(item)).encode()).digest()[:8],"little")
    rng=np.random.default_rng(seed)
    edits=[]
    for low,high in [(144,192),(0,152)]:
        width=int(rng.integers(9,33))
        start=int(rng.integers(low,high-width+1))
        wave=np.zeros(192)
        wave[start:start+width]=np.hanning(width)*rng.choice([-1,1])
        edits.append(wave)
    anchors=np.linspace(0,191,7)
    edits.append(np.interp(np.arange(192),anchors,rng.normal(size=len(anchors))))
    width=int(rng.integers(32,65))
    osc=np.zeros(192)
    osc[-width:]=np.hanning(width)*np.sin(2*np.pi*np.arange(width)/rng.uniform(5,24)+rng.uniform(-np.pi,np.pi))
    edits.append(osc)
    waves=np.stack(edits)
    return (.06*waves/np.sqrt((waves*waves).mean(axis=1,keepdims=True))).astype("f4")


def collect_inputs(cache,original,out,data_root):
    ids,x=data(data_root)
    base=np.load(original/"bolt_fit.npz")
    np.testing.assert_array_equal(ids,base["ids"])
    test_ids=ids[TEST]
    test_index=np.concatenate([np.flatnonzero(np.char.startswith(test_ids,s+"/"))[:64] for s in ("daily","weekly")])
    active=np.r_[np.arange(len(ids))[ALIGN],np.arange(len(ids))[TEST][test_index]]
    chosen=ids[active]
    rows={k:i for i,k in enumerate(chosen)}
    selected=x[active]
    plus=np.empty((len(active),6,192),"f4")
    minus=np.empty_like(plus)
    for i,(key,row) in enumerate(zip(chosen,selected)):
        waves=variations(key)
        plus[i,:4]=row+waves
        minus[i,:4]=row-waves
    seen=set()
    for domain in ("daily","weekly"):
        file=data_root/("m4_"+domain)/"data-00000-of-00001.arrow"
        with pa.memory_map(str(file)) as stream:
            table=pa.ipc.open_stream(stream).read_all()
        for k,id_ in enumerate(table["item_id"].to_pylist()):
            key=domain+"/"+str(id_)
            if key not in rows:
                continue
            i=rows[key]
            full=np.asarray(table["target"][k].as_py(),dtype=float)
            end=len(full)-8
            core=full[end-192:end]
            required=2 if i<NFIT else 4
            if end<192+required or not np.isfinite(full[end-192-required:end+required]).all():
                raise ValueError(f"Unusable observed shift for {key}")
            np.testing.assert_array_equal(((core-core.mean())/core.std()).astype("f4"),selected[i])
            plus[i,5]=minus[i,5]=selected[i]
            for j,shift in enumerate((2,) if i<NFIT else (2,4),start=4):
                plus[i,j]=(full[end-192+shift:end+shift]-core.mean())/core.std()
                minus[i,j]=(full[end-192-shift:end-shift]-core.mean())/core.std()
            seen.add(key)
    if seen!=set(chosen):
        raise ValueError("Missing real windows")
    cache.mkdir(exist_ok=True,parents=True)
    out.mkdir(exist_ok=True,parents=True)
    splits=np.array(["train"]*NTRAIN+["validation"]*NVALID+["daily"]*64+["weekly"]*64)
    np.savez_compressed(cache/"inputs.npz",ids=chosen,x=selected,plus=plus,minus=minus,
                        active=active,test_index=test_index,split=splits,cal=x[CAL],cal_ids=ids[CAL])
    meta={"ids":dict(zip(chosen,splits)),"calibration":ids[CAL].tolist(),"families":FEATURES,
          "input_sha256":sha256(selected.tobytes()+plus.tobytes()+minus.tobytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")


def capture(model_name,cache,original):
    inputs=dict(np.load(cache/"inputs.npz"))
    archive=dict(np.load(original/(model_name+"_fit.npz")))
    metadata=json.loads(str(archive["metadata"]))
    layer=metadata["site"][0]["module"]
    adapter=adapter_for(model_name)
    model=SteeringModel.from_adapter(adapter,quantiles=[.5])
    active=inputs["active"]
    clean=archive["s0"][active]
    _,baseline=sites_and_states(model,[layer],inputs["x"][[0,NFIT,-1]],3)
    np.testing.assert_allclose(baseline[layer],clean[[0,NFIT,-1]],atol=3e-4,rtol=1e-4)
    differences=[]
    for j,fam in enumerate(FEATURES):
        rows=slice(NFIT,None) if j==5 else slice(None)
        pair=np.concatenate([inputs["plus"][rows,j],inputs["minus"][rows,j]])
        _,values=sites_and_states(model,[layer],pair,32)
        hi,lo=np.split(values[layer],2)
        v=(hi-lo)/2
        if j==5:
            v=np.concatenate([np.zeros((NFIT,clean.shape[1]),"f4"),v])
        differences.append(v)
        print(model_name,fam,"captured",flush=True)
    delta=np.stack(differences,axis=1)
    np.savez_compressed(cache/(model_name+"_deltas.npz"),ids=inputs["ids"],clean=clean,delta=delta,
                        cal_clean=archive["s0"][CAL],direction=archive["d0"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","bolt","timesfm"))
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--original",type=Path,default=ORIG)
    p.add_argument("--out",type=Path,default=OUT)
    p.add_argument("--data-root",type=Path,default=DATA_ROOT)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.stage=="prepare":
        collect_inputs(a.cache,a.original,a.out,a.data_root)
    else:
        capture(a.stage,a.cache,a.original)
