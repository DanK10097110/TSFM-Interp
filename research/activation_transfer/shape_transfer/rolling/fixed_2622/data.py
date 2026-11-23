"""Hold out new sensors, then fix tail and earlier paired forecast origins."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.atoms.materials import CACHE as ATOMS
from research.activation_transfer.replication.data import CACHE as REP
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH, PUBLIC, VERSION, exclusions
from research.activation_transfer.shape_transfer.rolling.data import CACHE as DISCOVERY, OUT as OLDOUT

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
CACHE=Path("/tmp/tsfm_rolling_2622_20261121")
PHASES=("tail","earlier")
N=96
SEEDS=(20261121339,20261121340)


def pair(raw,offset):
    orig=len(raw)-8-32-480-offset
    a,b=raw[orig:orig+480].astype(float),raw[orig+32:orig+512].astype(float)
    if len(a)!=480 or len(b)!=480 or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.std(),b.std())<1e-8:
        return None
    data=np.stack([(v-a.mean())/a.std() for v in (a,b)]).astype("f4")
    fingerprints=[sha256(((v-v.mean())/v.std()).astype("f4").tobytes()).digest() for v in (a,b)]
    return data,fingerprints,dict(start=[orig,orig+32],stop=[orig+480,orig+512],anchor_mean=float(a.mean()),anchor_sd=float(a.std()),offset=offset)


def prepare():
    sets,_=exclusions()
    omit=sets["monash_traffic"].copy()
    seen=set()
    for old in (ATOMS,REP,FRESH,BLIND,DISCOVERY):
        with np.load(old/"data.npz") as f:
            for name,x in zip(f["ids"],f["x"]):
                name=str(name)
                if "monash_traffic/" in name:
                    omit.add(name.split("monash_traffic/")[-1])
                seen.add(sha256(x.tobytes()).digest())
                a=x.astype(float)
                if a.std()>0:
                    seen.add(sha256(((a-a.mean())/a.std()).astype("f4").tobytes()).digest())
    path=Path.home()/".cache/huggingface/datasets"/PUBLIC/"monash_traffic"/VERSION
    original=json.loads((OLDOUT/"manifest.json").read_text())
    with pa.memory_map(str(path)) as fp:table=pa.ipc.open_stream(fp).read_all()
    assert original["source"]["monash_traffic"]["sha256"]==sha256(path.read_bytes()).hexdigest()
    field="item_id" if "item_id" in table.column_names else "id"
    phases=np.random.default_rng(SEEDS[1]).integers(168,336,size=len(table))
    rows=[];contexts=[];used=set()
    for i in np.random.default_rng(SEEDS[0]).permutation(len(table)):
        name=str(table[field][int(i)].as_py())
        if name in omit or name in used:continue
        raw=table["target"][int(i)].values.to_numpy(zero_copy_only=False)
        if len(raw)<520+phases[i]:continue
        paired=[pair(raw,o) for o in (0,int(phases[i]))]
        if any(r is None for r in paired):continue
        hashes=[v for pair_data in paired for v in pair_data[1]]
        if len(set(hashes))<4 or seen.intersection(hashes):continue
        seen.update(hashes);used.add(name)
        rows.append(dict(series="monash_traffic/"+name,raw_length=len(raw),
                         **{phase:r[2] for phase,r in zip(PHASES,paired)}))
        contexts.append(np.stack([r[0] for r in paired]))
        if len(rows)==N:break
    assert len(rows)==N
    x=np.stack(contexts).transpose(1,2,0,3).reshape(-1,480).copy()
    ids=np.array([f"{p}/{lag}/{r['series']}" for p in PHASES for lag in (0,32) for r in rows])
    split=np.repeat(PHASES,2*N)
    CACHE.mkdir(exist_ok=True,parents=True);OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(CACHE/"data.npz",x=x,ids=ids,split=split)
    info=dict(seeds=SEEDS,prior_unique_traffic_ids=len(omit),series=rows,source=str(path),
              source_sha256=original["source"]["monash_traffic"]["sha256"],
              id_sha256=sha256("\n".join(ids).encode()).hexdigest(),input_sha256=sha256(x.tobytes()).hexdigest())
    (OUT/"manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    print(info["prior_unique_traffic_ids"],info["id_sha256"],info["input_sha256"],flush=True)


if __name__=="__main__":prepare()
