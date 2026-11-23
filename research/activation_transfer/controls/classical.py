"""Test whether responses of simple, within-history forecasters absorb behavioral transfer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.optimize import nnls

from research.activation_transfer.controls.analyze import by_horizon, fit_clean
from research.activation_transfer.controls.collect import (CONTEXT, EDITS, HERE, HORIZON, MODELS, REV, edit_histories, forecast_responses,
                     manifest_and_histories, read_histories, source_exclusions)


def classical_response(pairs):
    """Fit local ridge and soft analogy continuations using only 128 context points per edit."""
    output = {"local_ridge": [], "soft_analog": []}
    stack = pairs.reshape(-1, CONTEXT)
    for start in range(0, len(stack), 48):
        batch = stack[start:start+48].astype(np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(batch, 32, axis=1)
        past, future = windows[:,:,:16], windows[:,:,16:]
        query = batch[:, -16:]
        px, fy = past.mean(1), future.mean(1)
        p0, f0 = past-px[:, None, :], future-fy[:, None, :]
        xtx = p0.transpose(0,2,1) @ p0
        xty = p0.transpose(0,2,1) @ f0
        coef = np.linalg.solve(xtx + .1*past.shape[1]*np.eye(16), xty)
        ridge = fy + ((query-px)[:, None, :] @ coef)[:,0]
        # Shapes can match at different levels; restore the query endpoint.
        template = past-past.mean(-1,keepdims=True)
        q_center = query-query.mean(-1,keepdims=True)
        dist = ((template-q_center[:,None,:])**2).mean(-1)
        bandwidth = np.partition(dist, 4, axis=1)[:,4:5]
        weights = np.exp(-(dist-dist.min(1,keepdims=True))/np.maximum(bandwidth,1e-12))
        weights /= weights.sum(axis=1,keepdims=True)
        analog = np.einsum("bn,bnh->bh", weights, future-past[:,:,-1:]) + query[:, -1:]
        output["local_ridge"].append(ridge)
        output["soft_analog"].append(analog)
    for name in output:
        both = np.concatenate(output[name]).reshape(*pairs.shape[:-1], 16)
        output[name] = ((both[:,:,1]-both[:,:,0])/2).astype(np.float32)
    return output


def collect_followup(args):
    previous_manifest, previous_histories = manifest_and_histories(args)
    if json.loads((args.out/"items.json").read_text()) != previous_manifest:
        raise ValueError("Calibration data fails the original item and byte-hash check")
    exclusions = source_exclusions(args.transport_manifest)
    excludes = {row["sha256"] for row in previous_manifest["rows"]}
    shard = args.data_root/"bitbrains_fast_storage/H/data-00000-of-00001.arrow"
    with pa.memory_map(str(shard)) as f:
        rows = read_histories(pa.ipc.open_stream(f).read_all(), 160, 975513, "bitbrains_fast_storage",
                              exclusions, excludes, variate=0)
    manifest = {"items": [{"item":r[0], "sha256":r[1]} for r in rows], "source": str(shard.name),
                "source_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(), "variate":0, "seed":975513}
    mf = args.out/"bitbrains_items.json"
    if mf.exists() and json.loads(mf.read_text()) != manifest:
        raise ValueError("Prior Bitbrains manifest differs")
    if not mf.exists():
        mf.write_text(json.dumps(manifest, indent=2)+"\n")
    holdout = np.stack([r[2] for r in rows])
    pairs, _, added_stats = edit_histories(holdout)
    with np.load(args.out/"responses.npz", allow_pickle=False) as pri:
        item = np.concatenate([pri["item"], [r[0] for r in rows]])
        split = np.concatenate([pri["split"], np.repeat("bitbrains_fast_storage",len(rows))])
        summaries = np.concatenate([pri["summaries"], added_stats])
        result = {"item":item, "split":split, "summaries":summaries}
        for m in MODELS:
            print(f"Bitbrains {m}", flush=True)
            result[m] = np.concatenate([pri[m][:,:4], forecast_responses(m,pairs[:,:4],args)])
    pairs, _, _ = edit_histories(np.concatenate([previous_histories, holdout]))
    result.update(classical_response(pairs[:, :4]))
    np.savez_compressed(args.out/"classical_responses.npz", **result)


def fit_predict(train_inputs, obs_inputs, target):
    """Nonnegative slopes preserve sign, fitting one shared law per input perturbation."""
    s = np.stack(train_inputs, axis=-1)
    o = np.stack(obs_inputs, axis=-1)
    coefficients = []
    for i in range(target.shape[1]):
        a = s[:, i].reshape(-1,s.shape[-1])
        b = target[:,i].reshape(-1)
        coefficients.append(nnls(a,b)[0])
    coefs = np.stack(coefficients)
    return np.einsum("nehk,ek->neh", o, coefs), coefs.tolist()


def gain(base, new, draws):
    # Each waveform contributes equally irrespective of its marginal response magnitude.
    base = (base**2).mean(axis=-1)
    new = (new**2).mean(axis=-1)
    ratios = 1-new.mean(0)/base.mean(0)
    boot = 1-new[draws].mean(1)/base[draws].mean(1)
    return {"gain":float(ratios.mean()),"ci95":np.quantile(boot.mean(1),[.025,.975]).tolist(),
            "per_edit_gain":ratios.tolist(), "per_edit_ci95":np.quantile(boot,[.025,.975],axis=0).T.tolist()}


def analyze_classical(path):
    with np.load(path,allow_pickle=False) as f:
        data = {k:f[k] for k in f.files}
    split = data["split"]
    summaries = data["summaries"]
    fit = split=="train"
    simple = ["local_ridge","soft_analog"]
    results = {}
    for j,domain in enumerate(("m4_daily","m4_weekly","LOOP_SEATTLE","bitbrains_fast_storage")):
        test = split==domain
        draws = np.random.default_rng(73439+j).integers(test.sum(), size=(2000,test.sum()))
        for component in ("full","shape","level"):
            fit_resid, test_resid = {}, {}
            for model in [*MODELS, *simple]:
                train_y = by_horizon(data[model][fit].astype(np.float64),component)
                test_y = by_horizon(data[model][test].astype(np.float64),component)
                cal_pred, test_pred = fit_clean(summaries[fit], train_y, summaries[test])
                fit_resid[model], test_resid[model] = train_y-cal_pred, test_y-test_pred
            for src,dst in [("bolt","timesfm"), ("timesfm","bolt")]:
                y = test_resid[dst]
                predictions = []
                coefficients = []
                for names in [simple,[*simple,src]]:
                    predicted, coef = fit_predict([fit_resid[a] for a in names],
                                                  [test_resid[a] for a in names], fit_resid[dst])
                    predictions.append(predicted)
                    coefficients.append(coef)
                c, ct = predictions
                results.setdefault(f"{domain}/{src}_to_{dst}",{})[component] = {
                    "n":int(test.sum()), "classical_over_stats":gain(y,y-c,draws),
                    "tsfm_over_classical":gain(y-c,y-ct,draws),
                    "coefficients": {"two_classical":coefficients[0],"two_classical_then_tsfm":coefficients[1]},
                }
    return {"response_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "edits":list(EDITS[:4]), "results":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=Path.home()/".cache/huggingface/hub")
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--transport-manifest",type=Path)
    parser.add_argument("--out",type=Path,default=HERE)
    parser.add_argument("--batch-size",type=int,default=48)
    parser.add_argument("--analyze-only",action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        if not args.transport_manifest:
            parser.error("Supply the transport split manifest to regenerate the disjoint cohorts")
        if not args.data_root:
            args.data_root=args.hub/"datasets--Salesforce--GiftEval/snapshots"/REV
        collect_followup(args)
    scores = analyze_classical(args.out/"classical_responses.npz")
    (args.out/"classical_results.json").write_text(json.dumps(scores,indent=2,allow_nan=False)+"\n")
    for k,row in scores["results"].items():
        print(k, row["full"]["classical_over_stats"]["gain"], row["full"]["tsfm_over_classical"]["gain"],
              row["full"]["tsfm_over_classical"]["ci95"])
