"""Independently compare primary results to fresh-cache checkpoint and training replays."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


HERE=Path(__file__).resolve().parent


def check_bundle(expected, actual):
    """Require matching keys, shapes and discrete values; report evaluated float errors."""
    if set(expected)!=set(actual):
        raise ValueError("Replay and original keys differ")
    errors={}
    for key in expected:
        a,b=expected[key],actual[key]
        if a.shape!=b.shape:
            raise ValueError(f"Shape changed for {key}: {a.shape} to {b.shape}")
        if a.dtype.kind in "fc":
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError(f"Nonfinite forecast or direction in {key}")
            np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
            errors[key]=float(np.max(np.abs(a-b)))
        else:
            np.testing.assert_array_equal(a,b)
    return errors


def compare_models(reference,replay):
    import torch
    out={}
    for name in ("seed1_bolt","seed1_timesfm","seed2_bolt","seed2_timesfm"):
        a,b=reference/(name+".pt"),replay/(name+".pt")
        left=torch.load(a,map_location="cpu",weights_only=True)
        right=torch.load(b,map_location="cpu",weights_only=True)
        for key in ["d_in","dict_size","k"]:
            assert left[key]==right[key]
        sa,sb=left["state_dict"],right["state_dict"]
        assert set(sa)==set(sb)
        assert all(sa[k].shape==sb[k].shape and sa[k].dtype==sb[k].dtype for k in sa)
        errors={k:float((sa[k].float()-sb[k].float()).abs().max()) for k in sa}
        with a.open("rb") as r,b.open("rb") as s:
            hashes=[file_digest(f,"sha256").hexdigest() for f in (r,s)]
        orig,rerun=[json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for path in (a,b)]
        for key in ["train_state_sha256","train_item_sha256","n_rows","seed","threads","training"]:
            assert orig[key]==rerun[key]
        # A changed fit is a reproducibility result, even when the data and invocation match.
        out[name]=dict(reference_sha256=hashes[0],replay_sha256=hashes[1],tensor_max_absolute_error=errors,
                        train_state_sha256=orig["train_state_sha256"],n_rows=orig["n_rows"],
                        optimizer_steps=orig["training"]["n_optimizer_steps"],
                        mse_history_max_absolute_change=float(np.max(np.abs(np.array(orig["mse_history"])-rerun["mse_history"]))),
                        replay_seconds=rerun["wall_seconds"])
    return out


def compare_atoms(folder):
    orig=HERE/"atoms/results"
    assert json.loads((orig/"manifest.json").read_text(encoding="utf-8"))==json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    out={}
    for m in ["timesfm","bolt"]:
        name=m+"_effects.npz"
        with np.load(orig/name) as a,np.load(folder/name) as b:
            errors=check_bundle(a,b)
            out[m]=dict(n_ids=len(a["ids"]),n_atoms=a["active"].shape[1],
                        max_absolute_change=max(errors.values()),by_array=errors)
        assert json.loads((orig/(m+"_scores.json")).read_text(encoding="utf-8"))==json.loads((folder/(m+"_scores.json")).read_text(encoding="utf-8"))
    return out


def compare_steering(folder):
    with np.load(HERE/"steering/results/effects.npz") as original:
        replay={}
        for key in original:
            with np.load(folder/(key+".npz")) as f:
                replay[key]=f["effects"]
        errors=check_bundle(original,replay)
    metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    previous=json.loads((HERE/"steering/results/results.json").read_text(encoding="utf-8"))
    for key in ["series_ids","checkpoint_revision","sites","input_sha256"]:
        assert metadata[key]==previous[key]
    return dict(n_arms=len(errors),max_absolute_change=max(errors.values()),by_array=errors)


def compare_reencoding(folder):
    """Count changed feature sets after independently recapturing the model states."""
    original=HERE/"reencode/results"
    results={}
    for target in ("bolt","timesfm"):
        source="timesfm" if target=="bolt" else "bolt"
        scores_file=f"{source}_to_{target}.json"
        base_scores,rerun_scores=[json.loads((d/scores_file).read_text(encoding="utf-8"))["datasets"]
                                  for d in (original,folder)]
        cohorts={}
        for domain in ("m4","weekly","fast"):
            fname=f"{domain}_{target}_sufficient.npz"
            changes={}
            cosine_drift={}
            with np.load(original/fname) as base,np.load(folder/fname) as rerun:
                assert set(base)==set(rerun)
                for name in ("ids","pairs","donor_ids","count_names"):
                    np.testing.assert_array_equal(base[name],rerun[name])
                for gate in ("hard","shoulder"):
                    for method in ("true","matched","estimated","wrong","direct","raw_matched","native_estimated","oracle"):
                        prefix=f"{gate}_{method}"
                        enter,leave=[np.any(base[prefix+"_"+event]!=rerun[prefix+"_"+event],axis=-1)
                                     for event in ("enter","leave")]
                        changes[prefix]=int(np.count_nonzero(enter|leave))
                        if method!="true":
                            index=[s[domain]["gates"][gate]["methods"][method]["cosine"]["value"]
                                   for s in (base_scores,rerun_scores)]
                            cosine_drift[prefix]=abs(index[0]-index[1])
                cohorts[domain]=dict(edited_cases=len(base["ids"])*2,changed_enter_or_leave_sets=changes,
                                     maximum_change_in_mean_cosine=max(cosine_drift.values()))
        f=f"weekly_{target}_forecast.npz"
        with np.load(original/f) as base,np.load(folder/f) as rerun:
            assert set(base)==set(rerun)
            np.testing.assert_array_equal(base["ids"],rerun["ids"])
            errors={k:float(np.max(np.abs(base[k]-rerun[k]))) for k in base if k!="ids"}
        results[target]=dict(cohorts=cohorts,weekly_forecast_max_absolute_change=max(errors.values()),
                             weekly_forecast_change_by_arm=errors)
    return results


def request_drifts(original,replayed):
    """Keep the original 1e-5 numerical criterion visible even when a fresh fitted policy differs."""
    if set(original)!=set(replayed):
        raise ValueError("Request replay keys differ")
    result={}
    for key in original:
        a,b=original[key],replayed[key]
        assert a.shape==b.shape
        if a.dtype.kind not in "fc":
            np.testing.assert_array_equal(a,b)
            continue
        assert np.isfinite(a).all() and np.isfinite(b).all()
        delta=np.abs(a.astype(float)-b.astype(float))
        result[key]=dict(max_absolute_change=float(delta.max()),n_values=int(a.size),
                         outside_original_tolerance=int(np.count_nonzero(delta>1e-5+1e-5*np.abs(b))))
    return result


def compare_requests(folder):
    original=HERE/"forecast_requests/results"
    checks={}
    for model in ("bolt","timesfm","timer"):
        for fname in (model+".npz","dose_shams_"+model+".npz"):
            with np.load(original/fname) as a,np.load(folder/fname) as b:
                checks[fname]=request_drifts(a,b)
        manifests=[json.loads((p/(model+".json")).read_text(encoding="utf-8")) for p in (original,folder)]
        for key in ("target","calibration_ids","clean_fit_ids","source_fit_ids","test_ids","depths","requests"):
            assert manifests[0][key]==manifests[1][key]
        gains=[{k:np.asarray(v) for k,v in doc["applied_dose"].items()} for doc in manifests]
        checks[model+"_calibration_gain"]=request_drifts(*gains)
    scores=[json.loads((p/"dose_shams.json").read_text(encoding="utf-8")) for p in (original,folder)]
    grouped={}
    for domain in ("daily","weekly"):
        grouped[domain]={}
        for cue in ("level","tilt","sine"):
            grouped[domain][cue]={arm:[s["groups"][domain][cue][arm]["value"] for s in scores]
                                  for arm in ("real","sham")}
    cells=zip(scores[0]["rows"],scores[1]["rows"],strict=True)
    largest_correct=largest_sham=0.
    for a,b in cells:
        assert all(a[k]==b[k] for k in ("domain","source","target","depth","cue"))
        largest_correct=max(largest_correct,abs(a["real"]["value"]-b["real"]["value"]))
        largest_sham=max(largest_sham,abs(a["sham"]["value"]-b["sham"]["value"]))
    all_scores=[json.loads((p/"summary.json").read_text(encoding="utf-8"))["rows"] for p in (original,folder)]
    max_uncalibrated=max(abs(a["metrics"]["zero_shot"]["skill"]["value"]-b["metrics"]["zero_shot"]["skill"]["value"])
                         for a,b in zip(*all_scores,strict=True))
    return dict(component_comparisons=checks,original_and_replay_group_skills=grouped,
                max_calibrated_correct_cell_skill_change=largest_correct,
                max_calibrated_sham_cell_skill_change=largest_sham,
                max_zero_shot_cell_skill_change=max_uncalibrated)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atoms",type=Path)
    p.add_argument("--fixed-steering",type=Path)
    p.add_argument("--requests",type=Path)
    p.add_argument("--reencoding",type=Path)
    p.add_argument("--reference-seeds",type=Path)
    p.add_argument("--replayed-seeds",type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    report={}
    if args.atoms:
        report["primary_atom_replay"]=compare_atoms(args.atoms)
    if args.fixed_steering:
        report["fixed_direction_replay"]=compare_steering(args.fixed_steering)
    if args.reencoding:
        report["sparse_reencoding_replay"]=compare_reencoding(args.reencoding)
    if args.requests:
        report["output_request_replay"]=compare_requests(args.requests)
    if args.replayed_seeds:
        if not args.reference_seeds:
            p.error("Specify both original and replayed seed directories")
        report["sae_training_replay"]=compare_models(args.reference_seeds,args.replayed_seeds)
    if not report:
        p.error("Specify a replay to compare")
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n", encoding="utf-8")


if __name__=="__main__":
    main()
