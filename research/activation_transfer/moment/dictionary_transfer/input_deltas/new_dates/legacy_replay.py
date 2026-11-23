"""Repeat the seven direct interventions under the dependencies declared by upstream MOMENT."""
from importlib import metadata
import json
import sys

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates import official_replay as audit


def run():
    torch.set_num_threads(2)
    with np.load(audit.MOMENT/"horizon_prefix/new_dates/results/panel.npz") as f:
        mask=f["cohort"]=="secondary"
        panel,ids=f["x"][mask],f["ids"][mask]
    n,t,_=panel.shape
    # One date per station in archive order, cycling through all eleven origins.
    station=np.arange(n)
    dates=station%t
    rows=t*station+dates
    selected=panel[station,dates]
    with np.load(audit.HERE.parents[1]/"results/small_to_large_map.npz") as f:
        w=f["w"].astype(float)
    versions={k:metadata.version(k) for k in ("momentfm","torch","transformers","huggingface-hub","numpy")}
    assert versions["transformers"]=="4.33.3" and versions["huggingface-hub"]=="0.24.0"
    a,b=audit.native("small"),audit.native("large")
    out={k:[] for k in audit.ARMS}
    with torch.inference_mode():
        for i in range(0,n,8):
            waves=audit.one_station(ids[i:i+8],selected[i:i+8],np.linspace(-2,2,8),w,a,b)
            for k in out:
                out[k].append(waves[k])
    root=audit.HERE/"results"
    changed=audit.HERE.parent/"order/results"
    files=(root/"reference.npz",root/"mapped.npz",root/"wrong_edit.npz",changed/"ramp_swapped.npz",
           changed/"permuted_reference.npz",changed/"permuted_correct.npz",changed/"permuted_swapped.npz")
    metrics={}
    for name,path in zip(audit.ARMS,files):
        values=np.concatenate(out[name])
        with np.load(path) as f:
            np.testing.assert_array_equal(ids,f["ids"])
            diff=values.astype(float)-f["wave"][rows]
        assert diff.shape==(35,16) and np.isfinite(diff).all()
        # One thousandth of a degree was the fixed acceptable replay discrepancy.
        maximum=float(np.max(abs(diff)))
        assert maximum<.001,(name,maximum)
        metrics[name]=dict(maximum_absolute_C=maximum,rms_difference_C=float(np.sqrt((diff**2).mean())))
    result=dict(versions=versions,python=sys.version.split()[0],stations=ids.tolist(),date_indices=dates.tolist(),
                sampled_histories=n,scalar_comparisons=n*16*len(files),checks=metrics)
    (root/"legacy_replay.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    run()
