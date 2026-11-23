"""Keep decoder-vector length and angle fixed while changing its perpendicular forecast effect."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.moment.dictionary_transfer.functional.audit import addition
from research.activation_transfer.moment.dictionary_transfer.match import OUT as MATCH
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import OUT as FIT, SEED
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import OUT as DATA, WORK, panel
from research.activation_transfer.moment.dictionary_transfer.weather.prefix import cached_middle
from research.activation_transfer.moment.token_paths.geometry import layers, load

OUT = Path(__file__).resolve().parent / "results"
RANDOM = tuple(f"draw_{i}" for i in range(4))
ARMS = (*RANDOM, "other_feature")


def directions():
    with np.load(FIT / "fit.npz") as f:
        group = f["group"].astype(float)
        cols, atoms = f["donor_cols"], f["atoms"]
    with np.load(MATCH / f"large_{SEED}_to_large_matches.npz") as f:
        donor = f["source_decoder"][cols].astype(float)
        np.testing.assert_array_equal(f["atoms"][cols], atoms)
    unit = donor / np.linalg.norm(donor, axis=1)[:,None]
    def parallel(v):
        return np.einsum("ij,ij->i", v, unit)[:,None] * unit
    aligned = parallel(group)
    residual = group - aligned
    rng = np.random.default_rng(22112449)
    candidates = {k: rng.standard_normal(group.shape) for k in RANDOM}
    candidates["other_feature"] = np.roll(residual, -1, axis=0)
    for k, raw in candidates.items():
        ortho = raw - parallel(raw)
        assert np.all(np.linalg.norm(ortho, axis=1)>1e-9)
        v = aligned + ortho * (np.linalg.norm(residual, axis=1)/np.linalg.norm(ortho, axis=1))[:,None]
        for func in (lambda a: np.linalg.norm(a,axis=1), lambda a: (a*donor).sum(1),
                     lambda a: np.linalg.norm(a-donor,axis=1)):
            np.testing.assert_allclose(func(v), func(group), atol=2e-14, rtol=1e-12)
        candidates[k] = v
    return donor, group, atoms, candidates


def apply(batch):
    ids, dates, x = panel()
    donor, group, atoms, arms = directions()
    with np.load(DATA / "donor.npz") as f:
        dose = f["dose"]
        np.testing.assert_array_equal(f["atoms"], atoms)
        np.testing.assert_array_equal(f["ids"], ids)
    with np.load(DATA / "ordinary.npz") as f:
        base, last = f["base"], f["last"]
    full = np.load(WORK / "live_hidden.npy", mmap_mode="r")
    np.testing.assert_array_equal(full[:,61], last)
    model = load("large")
    site = layers(model)["middle"]
    r,c = np.nonzero(dose)
    assert len(r)==568 and len(ids)==35
    sel = np.linspace(0,len(r)-1,12,dtype=int)
    clean = model.predict(x[r[sel]], 16, [.5])["point"]
    # A new read of the reference and real group catches an incorrect donor sign or column ordering.
    for name, vec in (("donor",donor), ("oracle",group)):
        with addition(model, -dose[r[sel],c[sel],None]*vec[c[sel]]):
            actual = model.predict(x[r[sel]],16,[.5])["point"]-clean
        with np.load(DATA/(name+".npz")) as f:
            np.testing.assert_allclose(actual, f["wave"][r[sel],c[sel]],atol=1e-5)
    for arm, vec in arms.items():
        file = OUT / (arm+".npz")
        sig = sha256(vec.tobytes()+dose.tobytes()+x.tobytes()).hexdigest()
        if file.exists():
            with np.load(file) as f:
                assert str(f["signature"]) == sig
            continue
        wav = np.zeros((*dose.shape,16),dtype="f4")
        move = -dose[r,c,None]*vec[c]
        for o in range(0,len(r),batch):
            ix = slice(o,o+batch)
            with cached_middle(model,full[r[ix]],site), addition(model,move[ix]):
                wav[r[ix],c[ix]] = model.predict(x[r[ix]],16,[.5])["point"]-base[r[ix]]
        with addition(model,move[sel]):
            live = model.predict(x[r[sel]],16,[.5])["point"]-clean
        err = float(np.max(np.abs(live-wav[r[sel],c[sel]])))
        assert err<1e-5
        tmp=file.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, wave=wav, directions=vec, signature=sig, native_error_C=err, atoms=atoms, ids=ids, dates=dates)
        tmp.replace(file)
        print(arm, flush=True)


def score():
    ids, dates, x = panel()
    donor, group, atoms, _ = directions()
    rng = np.random.default_rng(22112449)
    n,t=len(ids),len(dates)
    sw = rng.multinomial(n,np.full(n,1/n),3000)
    tw = rng.multinomial(t,np.full(t,1/t),3000)
    def energy(a):
        v=(a**2).sum(2).reshape(n,t,len(atoms))
        return np.vstack((v.sum((0,1)),np.einsum("bs,stf,bt->bf",sw,v,tw,optimize=True)))
    def describe(z):
        return [float(z[0]),*np.nanquantile(z[1:],[.025,.975]).tolist()]
    def load_wave(path):
        with np.load(path) as f:
            np.testing.assert_array_equal(f["atoms"],atoms)
            np.testing.assert_array_equal(f["ids"],ids)
            np.testing.assert_array_equal(f["dates"],dates)
            return f["wave"].astype(float)
    ref = load_wave(DATA/"donor.npz")
    waves = {"group":load_wave(DATA/"oracle.npz")}
    waves.update({arm:load_wave(OUT/(arm+".npz")) for arm in ARMS})
    sd=x.std(1)[:,None,None]
    results=dict(atoms=atoms.tolist(),source_group_cosine=((donor*group).sum(1)/np.linalg.norm(donor,axis=1)/np.linalg.norm(group,axis=1)).tolist(),
                 raw_rms_C={k:float(np.sqrt(np.mean(v**2))) for k,v in {"reference":ref,**waves}.items()})
    for side, sl in (("h16",slice(None)),("early",slice(8)),("late",slice(8,16))):
        def center(z):
            a=z[:,:,sl]/sd
            return a-a.mean(2,keepdims=True)
        y = center(ref)
        den = energy(y)
        sc = {k:1-np.divide(energy(center(v)-y),den,out=np.full_like(den,np.nan),where=den>1e-16) for k,v in waves.items()}
        sc["random_mean"] = np.mean(np.stack([sc[k] for k in RANDOM]),axis=0)
        sc["group_minus_random"] = sc["group"]-sc["random_mean"]
        sc["group_minus_other_feature"] = sc["group"]-sc["other_feature"]
        results[side]={k:dict(equal=describe(np.nanmean(v,axis=1)),per_atom=v[0].tolist()) for k,v in sc.items()}
    (OUT/"summary.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("apply","score"))
    p.add_argument("--batch",type=int,default=16)
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.stage=="apply":
        apply(args.batch)
    else:
        score()
