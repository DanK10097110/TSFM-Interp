"""Measure full-vector forecast-response error for a level-preserving source correction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from research.activation_transfer.controls.analyze import by_horizon, fit_clean, stratified_draws
from research.activation_transfer.controls.classical import classical_response, fit_predict
from research.activation_transfer.controls.collect import HERE, MODELS, REV, edit_histories, forecast_responses, read_histories, source_exclusions


def collect(args):
    excluded = source_exclusions(args.transport_manifest)
    pools = [("bitbrains_rnd/H", "bitbrains_rnd", 160, 975516, 0), ("SZ_TAXI/15T", "SZ_TAXI", 48, 975517, None)]
    groups, items, xs = [], [], []
    manifest = {}
    hashes = set()
    for directory, name, count, seed, variate in pools:
        shard = args.data_root / directory / "data-00000-of-00001.arrow"
        with pa.memory_map(str(shard)) as f:
            rows = read_histories(pa.ipc.open_stream(f).read_all(), count, seed, name, excluded, hashes, variate)
        with shard.open("rb") as f:
            digest = hashlib.file_digest(f,"sha256").hexdigest()
        manifest[name] = {"seed":seed,"sha256":digest,"variate":variate,
                          "rows":[{"item":r[0],"sha256":r[1]} for r in rows]}
        items.extend(r[0] for r in rows)
        xs.extend(r[2] for r in rows)
        groups.extend([name]*count)
    file = args.out/"shape_items.json"
    if file.exists() and json.loads(file.read_text())!=manifest:
        raise ValueError("Stored test cohort differs")
    if not file.exists():
        file.write_text(json.dumps(manifest,indent=2)+"\n")
    pairs, _, summaries = edit_histories(np.stack(xs))
    pairs = pairs[:,:4]
    result = {"split":np.array(groups),"item":np.array(items),"summaries":summaries}
    result.update(classical_response(pairs))
    for model in MODELS:
        print("Shape-transfer evaluation", model, flush=True)
        result[model]=forecast_responses(model,pairs,args)
    np.savez_compressed(args.out/"shape_responses.npz",**result)


def shape_predictions(train,test):
    """Return full median-forecast response predictions using frozen component rules."""
    simple = ["local_ridge","soft_analog"]
    output = {}
    for component in ["level","shape","full"]:
        train_residual, test_residual, observed, clean_predictions = {}, {}, {}, {}
        for name in [*MODELS,*simple]:
            observed[name] = by_horizon(test[name].astype(np.float64),component)
            train_y = by_horizon(train[name].astype(np.float64),component)
            a,b=fit_clean(train["summaries"],train_y,test["summaries"])
            train_residual[name] = train_y-a
            test_residual[name] = observed[name]-b
            clean_predictions[name] = b
        for src,dst in [("bolt","timesfm"),("timesfm","bolt")]:
            k=f"{src}_to_{dst}"
            for names,key in [(simple,"classical"),([*simple,src],"neural")]:
                delta, coefficients = fit_predict([train_residual[n] for n in names], [test_residual[n] for n in names],
                                    train_residual[dst])
                output.setdefault(k,{})[component+"_"+key]=clean_predictions[dst]+delta
                if component == "shape" and key == "neural":
                    source_coefficient = np.asarray(coefficients)[:, -1]
                    output[k]["shape_source_term"] = test_residual[src] * source_coefficient[None, :, None]
    return output


def source_donor_check(true, yhat, input_summaries, seed):
    """Permute just the source TSFM term, retaining every learned classical coefficient."""
    learned = yhat["level_classical"] + yhat["shape_neural"]
    base = yhat["level_classical"] + yhat["shape_classical"]
    term = yhat["shape_source_term"]
    scale = np.square(true-base).mean(axis=(0,2))
    observed = float(np.mean(1 - np.square(true-learned).mean(axis=(0,2))/scale))
    null = []
    rng = np.random.default_rng(seed)
    for donors in stratified_draws(input_summaries, rng, 1999):
        shuffled = learned - term + term[donors]
        null.append(float(np.mean(1 - np.square(true-shuffled).mean(axis=(0,2))/scale)))
    null = np.asarray(null)
    return {"paired_gain": observed, "mean_shuffled_gain": float(null.mean()),
            "difference": observed-float(null.mean()), "p": float((1+(null>=observed).sum())/(len(null)+1)),
            "note":"Post hoc selectivity check. Only the TSFM term is permuted in held-out roughness quartiles."}


def evaluate(true, yhat, seed):
    only_classical=yhat["level_classical"]+yhat["shape_classical"]
    schemes={"shape_only":yhat["level_classical"]+yhat["shape_neural"],
             "level_only":yhat["level_neural"]+yhat["shape_classical"],
             "both_components":yhat["level_neural"]+yhat["shape_neural"],
             "single_coefficient":yhat["full_neural"],
             "classical_single_coefficient":yhat["full_classical"]}
    baseline=((only_classical-true)**2).mean(-1)
    draws=np.random.default_rng(seed).integers(len(true), size=(4000,len(true)))
    bt=baseline[draws].mean(1)
    metrics={}
    for name,pred in schemes.items():
        loss=((pred-true)**2).mean(-1)
        ratio=1-loss.mean(0)/baseline.mean(0)
        boot=1-loss[draws].mean(1)/bt
        metrics[name]={"gain":float(np.mean(ratio)), "ci95":np.quantile(boot.mean(1),[.025,.975]).tolist(),
                       "lower_bonferroni":float(np.quantile(boot.mean(1),.0125)),
                       "per_edit_gain":ratio.tolist(),"mse_by_edit":loss.mean(0).tolist()}
    # This identity follows from the orthogonality of mean and centered trajectory.
    np.testing.assert_allclose(np.array(metrics["shape_only"]["per_edit_gain"])
                               +np.array(metrics["level_only"]["per_edit_gain"]),
                               metrics["both_components"]["per_edit_gain"],atol=1e-8)
    return {"n":len(true),"classical_split_mse_by_edit":baseline.mean(0).tolist(),"schemes":metrics}


def analyze(args):
    with np.load(args.out/"classical_responses.npz",allow_pickle=False) as archive:
        prior={k:archive[k] for k in archive.files}
    with np.load(args.out/"shape_responses.npz",allow_pickle=False) as archive:
        new={k:archive[k] for k in archive.files}
    keys=["summaries","bolt","timesfm","soft_analog","local_ridge"]
    train={k:prior[k][prior["split"]=="train"] for k in keys}
    result={}
    for data in [prior,new]:
        for j,name in enumerate(sorted(set(data["split"])-{"train"})):
            test={k:data[k][data["split"]==name] for k in keys}
            predictions=shape_predictions(train,test)
            for direction,entry in predictions.items():
                dest=direction.split("_to_")[1]
                row = evaluate(test[dest], entry, 74323+j)
                if name in ("bitbrains_rnd", "SZ_TAXI"):
                    row["source_donor_check"] = source_donor_check(test[dest], entry, test["summaries"], 8692+j)
                result[f"{name}/{direction}"] = row
    (args.out/"shape_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for k,row in result.items():
        if k.startswith(("bitbrains_rnd/","SZ_TAXI/")):
            print(k, {m:(round(x["gain"],3),round(x["lower_bonferroni"],3)) for m,x in row["schemes"].items()})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub",type=Path,default=Path.home()/".cache/huggingface/hub")
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--transport-manifest",type=Path)
    parser.add_argument("--out",type=Path,default=HERE)
    parser.add_argument("--batch-size",type=int,default=48)
    parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    if not args.analyze_only:
        if not args.transport_manifest:
            parser.error("Supply the transport split manifest")
        if not args.data_root:
            args.data_root=args.hub/"datasets--Salesforce--GiftEval/snapshots"/REV
        collect(args)
    analyze(args)
