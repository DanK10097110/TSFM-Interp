"""Transport two temporal orders of the same physical increments across MOMENT checkpoint sizes."""
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.input_deltas.new_dates import run as original

OUT = Path(__file__).resolve().parent / "results"
PERMUTATION = [0, 6, 5, 4, 3, 2, 1, 7]


def measurements(waves, stations, dates):
    rng = np.random.default_rng(22112418)
    n,t = len(stations), len(dates)
    sw = rng.multinomial(n, np.full(n,1/n), 3000)
    tw = rng.multinomial(t, np.full(t,1/t), 3000)
    def energy(v):
        cells = np.sum(v*v, axis=1).reshape(n,t)
        return np.r_[cells.sum(), np.sum((sw @ cells)*tw, axis=1)]
    def record(a):
        return [float(a[0]), *np.quantile(a[1:], [.025,.975]).tolist()]
    results = {}
    for part,sl in (("early", slice(8)), ("late", slice(8,16))):
        for mode in ("shape","full"):
            def component(v):
                a = v[:,sl].astype(float)
                return a-a.mean(1,keepdims=True) if mode=="shape" else a
            scores = {}
            margins = []
            for task in ("ramp","permuted"):
                y = component(waves[task+"_reference"])
                den = energy(y)
                assert np.all(den>0)
                estimates = {arm:1-energy(component(waves[task+"_"+arm])-y)/den for arm in ("correct","swapped")}
                diff = estimates["correct"]-estimates["swapped"]
                margins.append(diff)
                scores[task] = {**{k:record(v) for k,v in estimates.items()}, "correct_minus_swapped":record(diff),
                                "reference_RMS_C":float(np.sqrt((y*y).mean()))}
            scores["mean_correct_minus_swapped"] = record(np.mean(margins, axis=0))
            dy = component(waves["ramp_reference"])-component(waves["permuted_reference"])
            scores["reference_difference_RMS_C"] = float(np.sqrt((dy*dy).mean()))
            results[part+"_"+mode] = scores
    return results


def main():
    ids, dates, x = original.panel()
    cache = original.WORK/"physical_ramp_confirmation"
    std = original.modify(x,"ramp")
    delta = np.linspace(-2,2,8)
    shuffled = x.copy()
    shuffled[:,-8:] += delta[PERMUTATION]
    np.testing.assert_array_equal(std[:,[-8,-1]],shuffled[:,[-8,-1]])
    np.testing.assert_array_equal(np.sort(delta),np.sort(delta[PERMUTATION]))
    with np.load(original.EXTERNAL/"ordinary.npz") as f:
        base,last = f["base"],f["last"]
        np.testing.assert_array_equal(f["ids"],ids)
        np.testing.assert_array_equal(f["dates"],dates)
    with np.load(original.MATCH/"small_to_large_map.npz") as f:
        w = f["w"].astype(float)
    small = original.load("small")
    clean = original.saved_state(small,cache,"small_base",x)
    ramp = (original.saved_state(small,cache,"small_ramp",std)-clean) @ w
    perm = (original.saved_state(small,cache,"small_permuted",shuffled)-clean) @ w
    del small
    large = original.load("large")
    q = original.saved_state(large,cache,"large_permuted",shuffled)-last
    full = np.load(original.WORK/"live_hidden.npy",mmap_mode="r")
    np.testing.assert_array_equal(full[:,61],last)
    arms = dict(permuted_reference=q,permuted_correct=perm,ramp_swapped=original.equal_length(perm,ramp),
                permuted_swapped=original.equal_length(ramp,perm))
    waves,replays=original.finite_waves(large,x,full,base,arms,OUT,ids,dates)
    for name,filename in (("ramp_correct","mapped"),("ramp_reference","reference")):
        with np.load(original.OUT/(filename+".npz")) as f:
            np.testing.assert_array_equal(f["ids"],ids)
            np.testing.assert_array_equal(f["dates"],dates)
            waves[name]=f["wave"]
    out=dict(stations=len(ids),origins=list(dates),ramp=delta.tolist(),permuted=delta[PERMUTATION].tolist(),
             native_max_difference_C=replays,outcomes=measurements(waves,ids,dates))
    (OUT/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
