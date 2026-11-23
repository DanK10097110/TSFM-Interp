"""Write an old-panel mean input-ramp direction into the frozen large forecaster on 385 new histories."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates.run import OUT as CURRENT, BATCH
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import OUT as EXTERNAL, WORK, panel
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle
from research.activation_transfer.moment.input_paths.run import modify
from research.activation_transfer.moment.token_paths.geometry import layers, load

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
OLD=HERE.parent/"results"
TEMP=Path("/tmp/tsfm_moment_ramp_template_20261122")


def sig(arr):
    return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def generic(history, mapped):
    """Each old station/date votes once. Case-specific length is the only new scalar in arm two."""
    template=np.asarray(history,float).mean(axis=0)
    size=np.linalg.norm(template)
    if size==0:
        raise ValueError("Historical mean ramp direction has zero norm")
    lengths=np.linalg.norm(mapped,axis=1)
    return np.broadcast_to(template,mapped.shape).copy(), lengths[:,None]*(template/size)[None]


def state(label,raw):
    with np.load(WORK/"physical_ramp_confirmation"/(label+".npz")) as f:
        assert str(f["sig"])==sig(raw)
        return f["arr"]


def inputs():
    ids,dates,x=panel()
    with np.load(MATCH/"small_to_large_map.npz") as f:
        w=f["w"].astype(float)
    with np.load(OLD/"small_native.npz") as old:
        old_ids=old["ids"]
        assert len(old_ids)==46 and not set(old_ids)&set(ids)
        history=(old["ramp_state"]-old["base_state"])@w
    assert history.shape==(184,1024)
    original_manifest=json.loads((HERE.parents[2]/"forecast_quality/post_release/results/manifest.json").read_text())
    np.testing.assert_array_equal(old_ids,original_manifest["accepted"])
    old_dates=original_manifest["origins"]
    assert len(old_dates)==4 and not set(old_dates)&set(dates)
    with np.load(EXTERNAL/"ordinary.npz") as f:
        np.testing.assert_array_equal(ids,f["ids"])
        np.testing.assert_array_equal(dates,f["dates"])
        assert sig(x)==str(f["input_sha256"])
        last,baseline=f["last"],f["base"]
    delta=state("small_ramp",modify(x,"ramp"))-state("small_base",x)
    mapped=delta@w
    fixed,matched=generic(history,mapped)
    np.testing.assert_allclose(np.linalg.norm(matched,axis=1),np.linalg.norm(mapped,axis=1),rtol=1e-10)
    ref=state("large_ramp",modify(x,"ramp"))-last
    return ids,dates,x,baseline,history,dict(reference=ref,mapped=mapped,historical_fixed=fixed,historical_matched=matched)


def collect(m,site,x,base,full,delta,name):
    signature=sha256(delta.tobytes()+x.tobytes()).hexdigest()
    pieces=[]
    for i in range(0,len(x),BATCH):
        file=TEMP/f"{name}_{i}.npz"
        if file.exists():
            with np.load(file) as f:
                assert str(f["signature"])==signature
                pieces.append(f["wave"])
            continue
        sl=slice(i,i+BATCH)
        with cached_middle(m,full[sl].copy(),site),addition(m,delta[sl]):
            y=m.predict(x[sl],16,[.5])["point"]-base[sl]
        tmp=file.with_suffix(".tmp.npz")
        np.savez(tmp,signature=signature,wave=y)
        tmp.replace(file)
        pieces.append(y)
        if i%96==0:
            print(name,i+BATCH,len(x),flush=True)
    return np.concatenate(pieces)


def main():
    torch.set_num_threads(2)
    TEMP.mkdir(parents=True,exist_ok=True)
    ids,dates,x,base,history,vectors=inputs()
    full=np.load(WORK/"live_hidden.npy",mmap_mode="r")
    with np.load(EXTERNAL/"ordinary.npz") as f:
        np.testing.assert_array_equal(full[:,61],f["last"])
    m=load("large")
    site=layers(m)["middle"]
    assert site=="encoder.block.11" and m.patch_layout(480,16)[0].stop-1==61
    waves={}; proofs={}
    with torch.inference_mode():
        for name,delta in vectors.items():
            if name in ("reference","mapped"):
                with np.load(CURRENT/(name+".npz")) as f:
                    np.testing.assert_array_equal(ids,f["ids"])
                    np.testing.assert_array_equal(dates,f["dates"])
                    assert str(f["signature"])==sha256(delta.tobytes()+x.tobytes()).hexdigest()
                    waves[name]=f["wave"]
            else:
                waves[name]=collect(m,site,x,base,full,delta,name)
            print(name,"ready",flush=True)
        # One full real forward per arm, identical row shapes and an explicit zero write.
        sel=np.array([0,135,384])
        with addition(m,np.zeros((len(sel),1024))):
            zero=m.predict(x[sel],16,[.5])["point"]
        native=m.predict(x[sel],16,[.5])["point"]
        zero_error=float(abs(zero-native).max())
        assert zero_error<3e-5
        for name,v in vectors.items():
            with addition(m,v[sel]):
                effect=m.predict(x[sel],16,[.5])["point"]-zero
            gap=float(abs(effect-waves[name][sel]).max())
            assert gap<.00015,(name,gap)
            proofs[name]=gap
    np.savez_compressed(OUT/"waves.npz",ids=ids,dates=dates,input_sha256=sig(x),historical_mean=vectors["historical_fixed"][0],
                        historical_norm=np.linalg.norm(history,axis=1),mapped_norm=np.linalg.norm(vectors["mapped"],axis=1),
                        template_cosine=(vectors["mapped"]@vectors["historical_fixed"][0])/
                            (np.linalg.norm(vectors["mapped"],axis=1)*np.linalg.norm(vectors["historical_fixed"][0])),
                        **waves)
    (OUT/"replay.json").write_text(json.dumps(dict(indices=sel.tolist(),stations=ids[sel//len(dates)].tolist(),
        dates=np.array(dates)[sel%len(dates)].tolist(),native_patched_zero_vs_ordinary_max_c=zero_error,
        matched_zero_full_vs_cached_effect_max_c=proofs,context_sha256=sig(x),reference_source="physical large ramp last-state substitution",
        map_file_sha256=sha256((MATCH/"small_to_large_map.npz").read_bytes()).hexdigest(),
        historical_small_state_file_sha256=sha256((OLD/"small_native.npz").read_bytes()).hexdigest()),indent=2)+"\n")


if __name__=="__main__":
    main()
