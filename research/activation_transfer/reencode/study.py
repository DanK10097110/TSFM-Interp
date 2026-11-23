"""Predict target SAE code changes by transporting a dense displacement to different anchors."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from tsfm_lens.sae.train import load_sae_checkpoint

from research.activation_transfer.boundary.analyze import GRID, RANK, SPLITS, Ridge, load_study, train_validate_one
from research.activation_transfer.boundary.capture import DEFAULT_SCRATCH, dictionary_path
from research.activation_transfer.boundary.smooth import smooth_codes

OUT = Path(__file__).resolve().parent / "results"
CACHE = Path("/tmp/tsfm_reencode_v1")
DATASETS = ("m4", "weekly", "fast")
GATES = ("hard", "shoulder")
METHODS = ("matched", "estimated", "wrong", "direct", "raw_matched", "native_estimated", "oracle")
COUNT_NAMES = ("true_enter", "pred_enter", "correct_enter", "true_leave", "pred_leave", "correct_leave",
               "true_switch", "pred_switch", "exact", "exact_on_switch")


@torch.inference_mode()
def encode(sae, h):
    pre = sae.pre_activations(torch.as_tensor(np.ascontiguousarray(h), dtype=torch.float32))
    shoulder, _ = smooth_codes(pre, sae.k)
    return {"hard": sae.sparsify(pre).numpy(), "shoulder": shoulder["shoulder"].numpy()}


def pairs_for(n):
    if n % 2:
        raise ValueError("need an even number of series for disjoint exchanges")
    pairs = np.random.default_rng(2026112301).permutation(n).reshape(-1, 2)
    donor = np.empty(n, dtype=int)
    donor[pairs.ravel()] = pairs[:, ::-1].ravel()
    return donor, pairs


def predicted_anchor(source, target, heldout):
    end, valid = SPLITS["train"][1], SPLITS["valid"][1]
    model = Ridge(source[:end], target[:end], RANK)
    projection = model.transform(source[end:valid])
    losses = [np.mean((model.predict(projection, c)-target[end:valid])**2) for c in GRID]
    penalty = GRID[np.argmin(losses)]
    return model.predict(model.transform(heldout), penalty).astype("f4"), float(penalty)


def event_masks(z0, z1):
    base = z0[:, None] > 0
    end = z1 > 0
    return end & ~base, base & ~end


def project_positive(clean_code, predicted_delta, k):
    """Convert unconstrained direct-regression coefficients into a predicted support."""
    pred = torch.as_tensor(np.asarray(clean_code[:, None]+predicted_delta, dtype="f4")).clamp_min(0)
    val, idx = torch.topk(pred, k, dim=-1)
    return torch.zeros_like(pred).scatter_(-1, idx, val).numpy()


def compact_ids(events, k):
    """Store the IDs in each entering/leaving set, with -1 padding."""
    result = np.full((*events.shape[:-1], k), -1, dtype=np.int16)
    for index in np.ndindex(events.shape[:-1]):
        found = np.flatnonzero(events[index])
        result[index][:len(found)] = found
    return result


def counts(actual, predicted):
    a, b = actual
    p, q = predicted
    s = lambda x: x.sum(-1)
    true_switch = a.any(-1) | b.any(-1)
    pred_switch = p.any(-1) | q.any(-1)
    exact = ~((a != p).any(-1) | (b != q).any(-1))
    return np.stack([s(a), s(p), s(a & p), s(b), s(q), s(b & q), true_switch,
                     pred_switch, exact, exact & true_switch], axis=-1).astype("i2")


def energies(p, t):
    p, t = p.astype(float), t.astype(float)
    dot = (p*t).sum(-1)
    ref_sq, pred_sq = (t*t).sum(-1), (p*p).sum(-1)
    return np.stack([dot, ref_sq, pred_sq, dot/np.maximum(np.sqrt(ref_sq*pred_sq), 1e-20)], axis=-1)


def interval(x):
    return {"value": float(x[0]), "ci95": np.quantile(x[1:], [.025, .975]).tolist()}


def aggregates(measures, event_counts, w):
    # w contains item weights, with one equal weight row and a paired bootstrap afterward.
    dot, ref, pred, cos = (w@measures.sum(axis=1)).T
    summed = w@event_counts.sum(axis=1)
    c = {key: summed[:, i] for i, key in enumerate(COUNT_NAMES)}
    n = 2*w.sum(1)
    div = lambda a,b: a/np.maximum(b,1)
    scores = dict(cosine=cos/n, recovery=(2*dot-pred)/np.maximum(ref,1e-20),
                  enter_precision=div(c["correct_enter"],c["pred_enter"]),
                  enter_recall=div(c["correct_enter"],c["true_enter"]),
                  leave_precision=div(c["correct_leave"],c["pred_leave"]),
                  leave_recall=div(c["correct_leave"],c["true_leave"]),
                  exact=div(c["exact"],n), exact_when_switched=div(c["exact_on_switch"],c["true_switch"]))
    return scores, {k:int(v[0]) for k,v in c.items()}


def bootstrap(n, pairs):
    cluster = np.random.default_rng(2026112302).multinomial(len(pairs), np.full(len(pairs), 1/len(pairs)), size=2000)
    weights = np.ones((len(cluster)+1,n))
    weights[1:,pairs.ravel()] = np.repeat(cluster,2,axis=1)
    return weights


def evaluate(target, scratch, cache, out):
    source = "bolt" if target == "timesfm" else "timesfm"
    cache.mkdir(exist_ok=True, parents=True)
    out.mkdir(exist_ok=True, parents=True)
    tr = load_study(scratch, "m4")
    sae = load_sae_checkpoint(str(dictionary_path(target)))
    train_dense = tr[source].states[:,:,1]-tr[source].clean[:,None]
    tr0, tr1 = encode(sae,tr[target].clean), encode(sae,tr[target].states[:,:,1])
    dz_train = {g:tr1[g]-tr0[g][:,None] for g in GATES}
    decoder = sae.W_dec.detach().numpy()
    details = {"source":source,"target":target,"dictionary_sha256":sha256(dictionary_path(target).read_bytes()).hexdigest(),
               "training_ids":tr[source].ids[:128].tolist(),"validation_ids":tr[source].ids[128:176].tolist(),
               "datasets":{}}
    for dataset in DATASETS:
        all_data = tr if dataset=="m4" else load_study(scratch,dataset)
        select = slice(*SPLITS["test"]) if dataset=="m4" else slice(None)
        target_data, source_data = all_data[target], all_data[source]
        h = target_data.clean[select]
        edited = target_data.states[select,:,1]
        true_vector = edited-h[:,None]
        source_h = source_data.clean[select]
        source_diff = source_data.states[select,:,1]-source_h[:,None]
        ids = target_data.ids[select]
        np.testing.assert_array_equal(ids,source_data.ids[select])
        with np.load(scratch/f"{dataset}_smooth_predictions.npz") as archive:
            np.testing.assert_array_equal(ids, archive["ids"])
            dense = archive[f"{target}_cross_dense"]
            raw = archive[f"{target}_cross_input"]
            np.testing.assert_allclose(true_vector,archive[f"{target}_native_dense"],atol=2e-6)
        est, anchor_penalty = predicted_anchor(tr[source].clean,tr[target].clean,source_h)
        donor,pairs = pairs_for(len(ids))
        anchors = {"matched":h,"estimated":est,"wrong":h[donor]}
        codes = {k:encode(sae,a) for k,a in anchors.items()}
        base = codes["matched"]
        true = encode(sae,edited)
        methods = {k:(codes[k],encode(sae,a[:,None]+dense)) for k,a in anchors.items()}
        methods.update(raw_matched=(base,encode(sae,h[:,None]+raw)),
                       native_estimated=(codes["estimated"],encode(sae,est[:,None]+true_vector)),
                       oracle=(base,encode(sae,h[:,None]+true_vector)))
        zfile = {"ids":ids,"pairs":pairs,"donor_ids":ids[donor],"count_names":COUNT_NAMES}
        weekly = {"ids":ids}
        group = {"n":len(ids),"anchor_penalty":anchor_penalty,"gates":{}}
        weights = bootstrap(len(ids),pairs)
        for gate in GATES:
            expected = true[gate]-base[gate][:,None]
            events = event_masks(base[gate],true[gate])
            zfile[gate+"_true_enter"],zfile[gate+"_true_leave"] = [compact_ids(e,sae.k) for e in events]
            direct, penalty, _ = train_validate_one(train_dense,dz_train[gate],source_diff,RANK)
            direct = direct.reshape(expected.shape).astype("f4")
            metrics, output = {}, {"direct_penalty":float(penalty),"methods":{}}
            for method in METHODS:
                if method=="direct":
                    delta = direct
                    predicted_events = event_masks(base[gate],project_positive(base[gate],delta,sae.k))
                else:
                    z0,z1 = methods[method]
                    delta = z1[gate]-z0[gate][:,None]
                    predicted_events = event_masks(z0[gate],z1[gate])
                e, c = energies(delta,expected), counts(events,predicted_events)
                key = gate+"_"+method
                zfile[key+"_energy"],zfile[key+"_counts"] = e,c
                zfile[key+"_enter"],zfile[key+"_leave"] = [compact_ids(mask,sae.k) for mask in predicted_events]
                scores, absolute = aggregates(e,c,weights)
                metrics[method] = scores
                output["methods"][method] = {"counts":absolute,**{k:interval(v) for k,v in scores.items()}}
                if dataset=="weekly":
                    weekly[key] = delta@decoder
                if method=="oracle":
                    assert scores["recovery"][0]>0.99999 and absolute["exact"]==2*len(ids)
            for other in ("estimated","wrong","direct","raw_matched"):
                output["matched_minus_"+other] = {k:interval(metrics["matched"][k]-metrics[other][k]) for k in metrics["matched"]}
            group["gates"][gate]=output
        details["datasets"][dataset]=group
        np.savez_compressed(out/f"{dataset}_{target}_sufficient.npz",**zfile)
        if dataset=="weekly":
            np.savez_compressed(cache/f"weekly_{target}_decoded.npz",**weekly)
        print(source,"to",target,dataset,"done",flush=True)
    (out/f"{source}_to_{target}.json").write_text(json.dumps(details,separators=(",",":"),allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("target",choices=["bolt","timesfm"])
    p.add_argument("--scratch",type=Path,default=DEFAULT_SCRATCH)
    p.add_argument("--cache",type=Path,default=CACHE)
    p.add_argument("--out",type=Path,default=OUT)
    a=p.parse_args()
    torch.set_num_threads(2)
    evaluate(a.target,a.scratch,a.cache,a.out)
