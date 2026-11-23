"""Source-only tests of depth, whole-state translation and explicit TopK competition."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from tsfm_lens.extraction.hooks import ActivationCatcher
from research.activation_transfer.moment.token_paths.geometry import CACHE, histories, load, layers
from research.activation_transfer.moment.dictionary_transfer.match import dictionary, hidden
from research.activation_transfer.moment.dictionary_transfer.weather.detector.fit import train, infer_doses
from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import panel, OUT as NATIVE, WORK

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SCRATCH = Path(os.environ.get("TSFM_CROSSCODER_CACHE", "/tmp/tsfm_moment_crosscoder"))
SPLIT = 768
PENALTIES = (1e-5, 1e-4, .001, .01, .1, 1.)
SEED = 22112201


def sources():
    x, ids, _ = histories(True)
    y, held_ids, domains = histories(False)
    stations, dates, weather = panel()
    assert not set(ids) & set(held_ids) and np.isfinite(weather).all()
    assert len(stations) * len(dates) == len(weather)
    return (x, y, weather), np.r_[ids, held_ids,
        [str(i) + "/" + str(t) for i in stations for t in dates]], domains, stations, dates


def capture():
    sets, ids, _, _, _ = sources()
    a = load("small")
    names = ["encoder.block.0", layers(a)["middle"], "encoder.block." + str(len(a.module.encoder.block)-1)]
    rows = {n: [] for n in names}
    for xs in sets:
        for lo in range(0, len(xs), 32):
            with ActivationCatcher(a.module, names) as cap:
                a.predict(xs[lo:lo+32], 16, [.5])
                result = cap.collect()
            for n in names:
                rows[n].append(result[n][:, 61].numpy().copy())
    arrays = dict(zip(("first", "middle", "last"), [np.concatenate(rows[n]) for n in names]))
    with np.load(CACHE/"small"/"train.npz") as tr, np.load(CACHE/"small"/"eval.npz") as ev, np.load(
        WORK/"physical_ramp_confirmation"/"small_base.npz") as we:
        ref = np.concatenate((tr["middle"][:, 0], ev["middle"][:, 0], we["arr"]))
        np.testing.assert_array_equal(ids[:1024], tr["ids"])
        np.testing.assert_array_equal(ids[1024:1280], ev["ids"])
        assert np.max(abs(ref-arrays["middle"])) < 3e-5
    SCRATCH.mkdir(exist_ok=True, parents=True)
    np.savez(SCRATCH/"depth.npz", **arrays, ids=ids, names=names, middle_max_abs=np.max(abs(ref-arrays["middle"])))


def large_parts():
    with np.load(CACHE/"large"/"train.npz") as tr, np.load(CACHE/"large"/"eval.npz") as ev, np.load(NATIVE/"ordinary.npz") as we:
        return np.concatenate((tr["middle"][:,0], ev["middle"][:,0], we["last"]))


def targets(sae, h, atoms):
    measures, top, codes = [], [], []
    for lo in range(0, len(h), 256):
        batch = torch.tensor(h[lo:lo+256], dtype=torch.float32)
        with torch.inference_mode():
            pre = sae.pre_activations(batch)
            val, idx = torch.topk(pre.clamp_min(0), sae.k, dim=1)
            assert torch.all(val[:,-1]>0)
            measures.append(torch.cat((pre[:,atoms], val[:,-1:]), dim=1).numpy())
            codes.append(sae.sparsify(pre)[:,atoms].numpy())
            top.append(idx.numpy())
    return np.concatenate(measures), np.concatenate(top), np.concatenate(codes)


def scaled_views(parts):
    scaled = []
    for part in parts:
        x = part.astype(float)
        center = x[:SPLIT].mean(0)
        scaled.append((x-center)/np.sqrt(np.mean((x[:SPLIT]-center)**2)))
    return np.concatenate(scaled, axis=1)


def map_inputs(x_train, y_train, x_test):
    """Ridge paths on train-centered covariances. Centering chunks avoids two large float64 copies."""
    mx = np.mean(x_train, axis=0, dtype=float)
    my = np.mean(y_train, axis=0, dtype=float)
    gram = np.zeros((len(mx), len(mx)))
    cross = np.zeros((len(mx), len(my)))
    for lo in range(0, len(x_train), 1024):
        part = x_train[lo:lo+1024] - mx
        gram += part.T @ part
        cross += part.T @ (y_train[lo:lo+1024] - my)
    eig, vec = np.linalg.eigh(gram)
    cross = vec.T @ cross
    proj = (x_test - mx) @ vec
    for lam in PENALTIES:
        pred = (proj / (eig.clip(min=0)+lam*eig[-1])) @ cross + my
        yield lam, pred


def gate_predictions(values):
    return np.where(values[:,:-1] >= values[:,-1:], np.maximum(0, values[:,:-1]), 0.)


def two_readouts(x, measures, labels):
    idx = np.arange(SPLIT)
    val = np.arange(SPLIT, 1024)
    fit = train(x.astype(float), labels.astype(float), idx, val)
    _, direct = infer_doses(x,fit)
    best = np.full(labels.shape[1], np.inf)
    chosen = np.empty(labels.shape[1])
    output = np.zeros_like(labels)
    for lam, pred in map_inputs(x[idx], measures[idx], x):
        estimate = gate_predictions(pred)
        val_loss = ((estimate[val]-labels[val])**2).mean(0)
        improve = val_loss < best
        output[:, improve] = estimate[:, improve]
        chosen[improve] = lam
        best[improve] = val_loss[improve]
    return direct.astype("f4"), output, dict(gate_relative=chosen.tolist(),
        direct_penalty=fit["lam"].tolist(), direct_threshold=fit["threshold"].tolist())


def score(pred, truth):
    den = np.sum(truth.astype(float)**2,axis=0)
    per = np.divide(np.sum((pred-truth.astype(float))**2,axis=0),den,
        out=np.full(len(den),np.nan),where=den>0)
    return dict(macro=float(np.nanmean(1-per)), per_atom=[None if not np.isfinite(v) else float(1-v) for v in per],
                tp=int(((pred>0)&(truth>0)).sum()),fp=int(((pred>0)&(truth==0)).sum()),
                fn=int(((pred==0)&(truth>0)).sum()))


def run():
    sets, ids, domains, stations, dates = sources()
    sae = dictionary("large", SEED)
    with np.load(NATIVE/"donor.npz") as f:
        atoms = f["atoms"]
        np.testing.assert_array_equal(f["ids"], stations)
        np.testing.assert_array_equal(f["dates"], dates)
    large = large_parts()
    metrics, top, truth = targets(sae, large, atoms)
    np.testing.assert_array_equal(gate_predictions(metrics), truth)
    cachefile = SCRATCH/"depth.npz"
    with np.load(cachefile) as f:
        first, middle, last = [f[k] for k in ("first","middle","last")]
        np.testing.assert_array_equal(f["ids"], ids)
        report = dict(layers=f["names"].tolist(), middle_replay_error=float(f["middle_max_abs"]), atoms=atoms.tolist())
    raw = np.concatenate(sets).astype(float)
    raw = (raw-raw.mean(1,keepdims=True))/np.sqrt(raw.var(1,keepdims=True)+1e-5)
    views = {"first": first, "middle": middle, "last": last, "three_layer": scaled_views((first,middle,last)),
             "input_history": raw, "input_patch": raw[:,-8:]}
    pred, report["fits"] = {}, {}
    for k, x in views.items():
        pred[k+"_direct"], pred[k+"_gate"], report["fits"][k] = two_readouts(x, metrics, truth)
    train_ids = np.arange(SPLIT)
    validation = np.arange(SPLIT,1024)
    splits = {"validation":validation, "m4_daily":1024+np.flatnonzero(domains=="daily"),
              "traffic":1024+np.flatnonzero(domains=="monash_traffic"),"weather":np.arange(1280,len(ids))}
    bestlayer = min(("first","middle","last"),key=lambda k:-score(pred[k+"_gate"][validation],truth[validation])["macro"])
    report["validation_best_single_gate"] = bestlayer
    frequent = np.bincount(top[train_ids].ravel(), minlength=sae.dict_size).argsort(kind="stable")[-sae.k:]
    report["full_dictionary"] = {}
    for fit_on in ("last","all"):
        if fit_on=="all":
            sx, sy = (hidden(size)[:SPLIT].reshape(-1,d) for size,d in (("small",512),("large",1024)))
        else:
            sx, sy = middle[train_ids], large[train_ids]
        best_loss = np.inf
        for lam, estimate in map_inputs(sx,sy,middle):
            loss = np.mean((estimate[validation]-large[validation])**2)
            if loss < best_loss:
                best_loss, best_lam, best_hidden = loss, lam, estimate.astype("f4")
        name = "map_"+fit_on+"_exactgate"
        _, est_top, pred[name] = targets(sae,best_hidden,atoms)
        report["fits"][name] = dict(relative=best_lam, validation_hidden_mse=best_loss, training_patches=len(sx))
        top_hits = (top[:,:,None] == est_top[:,None,:]).any(2).sum(1)
        common_hits = np.isin(top, frequent).sum(1)
        per = {}
        gaps = abs(metrics[:,:-1] - metrics[:,-1:]) / metrics[:,-1:]
        for k, ix in splits.items():
            truth_energy = ((large[ix]-np.mean(sy,axis=0,dtype=float))**2).sum()
            error = ((best_hidden[ix].astype(float)-large[ix])**2).sum()
            miss = (pred[name][ix]>0)!=(truth[ix]>0)
            by_gap = {}
            for lo, hi in ((0,.05),(.05,.2),(.2,.5),(.5,np.inf)):
                sel = (gaps[ix]>=lo)&(gaps[ix]<hi)
                by_gap[str(lo)] = [int(miss[sel].sum()),int(sel.sum())]
            per[k] = dict(state_r2=float(1-error/truth_energy),top32_overlap=float(top_hits[ix].mean()/sae.k),
                frequent32_overlap=float(common_hits[ix].mean()/sae.k),gate_errors_by_relative_gap=by_gap)
        report["full_dictionary"][name]=per
    report["dose_skill"] = {k:{arm:score(p[ix], truth[ix]) for arm,p in pred.items()} for k,ix in splits.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    # Publish named predictions only after verifying the target and atom order.
    weather = splits["weather"]
    with np.load(NATIVE/"donor.npz") as f:
        np.testing.assert_array_equal(truth[weather]>0, f["dose"]>0)
        np.testing.assert_allclose(truth[weather], f["dose"], rtol=2e-6, atol=2e-6)
        report["archived_target_dose_max_abs"] = float(np.max(abs(truth[weather]-f["dose"])))
    tmp = OUT/"mechanistic_predictions.tmp.npz"
    np.savez_compressed(tmp, ids=stations, dates=dates, atoms=atoms, **{k:v[weather] for k,v in pred.items()})
    tmp.replace(OUT/"mechanistic_predictions.npz")
    np.savez_compressed(OUT/"mechanistic_panels.npz", ids=ids, domains=domains, atoms=atoms, truth=truth,
        preactivation=metrics[:,:-1], cutoff=metrics[:,-1], **pred)
    (OUT/"mechanism.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["capture","run"])
    args=p.parse_args()
    torch.set_num_threads(2)
    (capture if args.stage=="capture" else run)()
