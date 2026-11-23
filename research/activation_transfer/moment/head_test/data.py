"""Choose disjoint M4 daily series for training a frozen model's reconstruction readout."""
from hashlib import sha256, file_digest
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.moment.forecast_quality.data import GIFT, OUT as QUALITY, benchmark_items,hash_array
from research.activation_transfer.moment.token_paths.geometry import histories

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
SCRATCH=Path("/tmp/tsfm_moment_nov22_heads")


def sample():
    with np.load(QUALITY/"panel.npz") as f:
        excluded={v.split("/",1)[1] for v in f["id"] if v.startswith("m4_daily/")}
        test_hashes={hash_array(x) for x in f["context"].reshape(-1,480)}
    excluded.update(benchmark_items()["m4_daily"])
    support=HERE.parents[1]/"shape_transfer/support_confirmation/results"
    for name in ("manifest","short_manifest"):
        excluded.update(map(str,json.loads((support/(name+".json")).read_text())["items"]))
    _,blind_ids,_=histories(False)
    excluded.update(v.split("/",1)[1] for v in blind_ids if v.startswith("daily/"))
    path=GIFT/"m4_daily/data-00000-of-00001.arrow"
    with pa.memory_map(str(path)) as f:table=pa.ipc.open_stream(f).read_all()
    keys=list(map(str,table["item_id"].to_pylist()))
    order=sorted(range(len(keys)),key=lambda i:sha256(("moment-recon-head-20261122|"+keys[i]).encode()).digest())
    ids=[];inputs=[];holes=[];fingerprints=set(test_hashes)
    for i in order:
        item=keys[i]
        if item in excluded:continue
        y=table["target"][i].values
        if len(y)<480:continue
        x=y.slice(len(y)-480,480).to_numpy(zero_copy_only=False).astype(float)
        digest=hash_array(x)
        if not np.isfinite(x).all() or x.std()<1e-8 or digest in fingerprints:continue
        ids.append(item);inputs.append(x);fingerprints.add(digest)
        rng=np.random.default_rng(int.from_bytes(sha256(("interior-holes|"+item).encode()).digest()[:8],"little"))
        starts=np.arange(0,60,15)+rng.integers(1,13,size=4)
        holes.append(np.column_stack([starts,starts+1]).ravel())
        if len(ids)==1152:break
    if len(ids)!=1152:raise ValueError(len(ids))
    mask=np.array(["train"]*1024+["cal"]*128)
    x=np.stack(inputs); h=np.stack(holes)
    OUT.mkdir(parents=True,exist_ok=True);SCRATCH.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(SCRATCH/"train.npz",ids=ids,context=x,holes=h,split=mask)
    with path.open("rb") as f:source_sha=file_digest(f,"sha256").hexdigest()
    result=dict(source=str(path),source_sha256=source_sha,ids=ids,split=mask.tolist(),
                context_sha256=hash_array(x),mask_sha256=hash_array(h),excluded_ids=sorted(excluded),
                n_test_context_hashes=len(test_hashes))
    (OUT/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Readout data",len(ids),len(excluded),result["context_sha256"],flush=True)


if __name__=="__main__":sample()
