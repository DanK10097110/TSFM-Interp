"""Check compact study products without a downloaded checkpoint, original Arrow sources or live hooks."""
from hashlib import sha256
from pathlib import Path
import json
import re

import numpy as np

from research.activation_transfer.moment.dictionary_transfer.overview import run as recompute_dictionary

HERE=Path(__file__).resolve().parent
SEEDS=(22112201,22112202)


def read(path):
    return json.loads(path.read_text())


def forecast_quality():
    quality=HERE/"forecast_quality/results"
    with np.load(quality/"panel.npz") as f:
        x,y,ids,domains,test=f["context"].astype(float),f["future"],f["id"],f["domain"],f["split"]=="holdout"
    study=read(quality/"summary.json")["holdout"]["scores"]
    scale=np.abs(np.diff(x,axis=2)).mean(axis=2)
    groups=np.unique(domains)
    assert len(groups)==6 and test.sum()==576 and x.shape==(720,2,480)
    errors=[]
    for model in ("moment_small","moment_base","moment_large","bolt","timesfm"):
        with np.load(quality/(model+".npz")) as f:
            np.testing.assert_array_equal(ids,f["ids"])
            for h in (8,16):
                pred=f[f"raw_{h}"].astype(float)
                assert pred.shape==(720,2,h) and np.isfinite(pred).all()
                mase=(abs(pred-y[:,:,:h]).mean(axis=2)/scale).mean(1)
                for label in groups:
                    expected=study["h"+str(h)]["absolute"][str(label)][model]["mase1"]["value"]
                    errors.append(abs(float(mase[test&(domains==label)].mean())-expected))
    assert max(errors)<1e-8,max(errors)
    parity=read(quality/"cross_replay.json")
    assert len(parity)==12 and all(v["bit_identical_at_native_float32_precision"] for v in parity.values())
    return dict(native_scores=len(errors),maximum_MASE1_error=max(errors),
                official_forecast_scalars_replayed=sum(r["number_of_forecast_values_compared"] for r in parity.values()))


def mechanisms():
    d=HERE/"horizon_prefix/results"
    z={}
    for size in ("small","large"):
        with np.load(d/f"noaa_{size}_factorial.npz") as f:
            assert len(f["ids"])==46
            err=float(np.max(abs(f["native8"][:,:8]-f["both8"][:,:8])))
            assert err<3e-5
            z[size+"_H8_reconstruction_C"]=err
    with np.load(d/"noaa_large_factorial.npz") as f:parent=dict(f)
    for pos in ("h8","h16"):
        with np.load(d/f"noaa_large_layers_{pos}.npz") as f:
            np.testing.assert_array_equal(f["cell_000"],parent["both8" if pos=="h8" else "one_mask"][:,:8])
            np.testing.assert_array_equal(f["cell_111"],parent["position8" if pos=="h8" else "native16"][:,:8])
            assert all(np.isfinite(f["cell_"+format(i,"03b")]).all() for i in range(8))
        c=read(d/f"noaa_large_layers_{pos}.json")
        total=sum(c[k]["share"][0] for k in ("early","middle","late"))
        np.testing.assert_allclose(total,1.,atol=2e-6)
    audit=read(HERE/"horizon_prefix/new_dates/results/audit.json")
    assert audit["outcomes"]==6160 and audit["observed"]+audit["imputed"]==70*11*480
    for s in ("small","large"):
        for seed in SEEDS:
            live=read(HERE/f"head_interventions/results/{s}_{seed}_replay.json")
            assert live["n"]==24 and len(live["heads"])==5
            assert max(h["max_effect_gap"] for h in live["heads"].values())<1.5e-5
        positive=read(HERE/f"token_paths/results/{s}_positive.json")
        assert positive["n"]==32 and positive["positive_rms"]>.001
        for k in ("observed_effect_max","first_to_second_max","second_to_first_max"):assert positive[k]==0
        for derivative in ("gradient","replay"):
            a=read(HERE/f"dictionary_transfer/functional/results/{s}_{derivative}_audit.json")
            if derivative=="gradient":assert max(min(e.values()) for e in a.values())<.035
            else:assert max(a.values())<2e-5
    return {**z,"temperature_observations_from_raw_archive":audit["outcomes"],"physical_readout_replays":2*2*24*5,
            "depth_factorial_cells":16,"terminal_token_known_answer_histories":64}


def transfer_controls():
    recompute_dictionary()
    base=HERE/"dictionary_transfer/results"
    grid=read(base/"overview.json");counts=0
    for source in ("small","large"):
        other="large" if source=="small" else "small"
        with np.load(base/f"{source}_{SEEDS[0]}_to_{other}_forecast.npz") as f:
            test=set(f["ids"])
        mapping=read(base/f"{source}_to_{other}_map.json")
        sets=[set(mapping[k]) for k in ("training_ids","validation_ids","blind_map_ids")]
        assert [len(s) for s in sets]==[128,64,128] and len(set.union(*sets))==320 and test.isdisjoint(set.union(*sets))
        for target in (source,other):
            for seed in SEEDS:
                stem=f"{source}_{seed}_to_{target}"
                counts+=sum(len(arms) for regime in grid[stem].values() for arms in regime.values())
                with np.load(base/f"{stem}_forecast.npz") as f:
                    if source==target:np.testing.assert_allclose(f["native_dense"],f["truth_native"],atol=2e-5,rtol=.005)
        fn=base.parent/"functional/results";stem=f"{source}_{SEEDS[0]}_to_{other}"
        with np.load(fn/f"{stem}_fit.npz") as f:
            assert len(set(f["ids"]))==128 and test.isdisjoint(f["ids"])
            assert np.all(f["permutation"]!=np.arange(96))
            for name in ("dense","sham","group"):assert np.all(np.linalg.norm(f[name],axis=1)<=f["cap"]+1e-7)
        with np.load(fn/f"{stem}_budget.npz") as f:
            np.testing.assert_allclose(f["current_matched_length"],f["wrong_matched_length"],rtol=1e-9)
            assert np.all(f["current_matched_length"]<=f["cap"]+1e-7)
            assert np.all(f["alternate"]!=np.arange(len(f["alternate"])))
    return dict(physical_direction_summaries_recomputed=counts,same_encoder_seed_pairs=4,cross_encoder_seed_pairs=4)


def new_panel_effects():
    """Rebuild confirmation point scores from finite edits without any study scorer."""
    dictionary = HERE / "dictionary_transfer"
    detector = dictionary / "weather/new_dates/results"
    ramp = dictionary / "input_deltas/new_dates/results"
    weather = HERE / "horizon_prefix/new_dates/results"
    with np.load(weather / "panel.npz") as f:
        take = f["cohort"] == "secondary"
        ids, x = f["ids"][take], f["x"][take]
    n, t, width = x.shape
    assert (n, t, width) == (35, 11, 480)
    old_panel = HERE / "forecast_quality/post_release/results/panel.npz"
    with np.load(old_panel) as f:
        assert not set(ids) & set(f["ids"])
    def read_wave(path):
        with np.load(path) as f:
            np.testing.assert_array_equal(f["ids"], ids)
            assert len(set(f["dates"])) == t and np.isfinite(f["wave"]).all()
            return f["wave"].astype(np.float64)
    def center(a):
        return a - a.mean(-1, keepdims=True)
    sd = x.reshape(n*t, width).std(1)[:,None,None]
    ref = center(read_wave(detector / "donor.npz") / sd)
    den = np.sum(ref*ref, axis=(0,2))
    scores = read(detector / "summary.json")["all_all_centered"]
    for name in ("natural", "oracle", "inferred"):
        z = center(read_wave(detector / (name+".npz")) / sd)
        per_atom = 1 - np.sum((z-ref)**2, axis=(0,2))/den
        np.testing.assert_allclose(per_atom, scores[name]["per_atom"], atol=1e-9)
    native = read_wave(ramp / "reference.npz")
    report = read(ramp / "summary.json")
    assert report["stations"] == n and report["dates"] == t
    rng = np.random.default_rng(22112418)
    sw = rng.multinomial(n,np.full(n,1/n),3000)
    tw = rng.multinomial(t,np.full(t,1/t),3000)
    def summed_energy(a):
        q = (a*a).sum(1).reshape(n,t)
        return np.r_[q.sum(), ((sw @ q) * tw).sum(1)]
    def summary(a):
        return [a[0], *np.quantile(a[1:], [.025,.975])]
    preds = {k: read_wave(ramp/(k+".npz")) for k in ("mapped","wrong_edit","other_station")}
    for part, sl in (("early",slice(8)), ("late",slice(8,16))):
        for mode, fn in (("shape",center),("full",np.asarray)):
            y=fn(native[:,sl]); den=summed_energy(y)
            estimates = {k:1-summed_energy(fn(v[:,sl])-y)/den for k,v in preds.items()}
            for k in ("wrong_edit","other_station"):
                estimates["mapped_minus_"+k] = estimates["mapped"]-estimates[k]
            for k,v in estimates.items():
                np.testing.assert_allclose(summary(v), report["outcomes"][part+"_"+mode][k], atol=1e-10)
    # Read native inputs and output arrays independently of the template model-replay script.
    tp = dictionary / "input_deltas/template/results"
    with np.load(tp / "waves.npz") as f:
        np.testing.assert_array_equal(f["reference"], native)
        np.testing.assert_array_equal(f["mapped"], preds["mapped"])
        assert f["input_sha256"].item() == sha256(x.reshape(n*t,width).tobytes()).hexdigest()
    official = read(ramp / "official_replay.json")
    arms = official["checks"]
    assert len(arms) == 7 and all(v["scalars"] == n*t*16 for v in arms.values())
    assert official["whole_graph"] and official["clean_recipients_per_arm"] == n*t
    native_error = max(v["maximum_absolute_C"] for v in arms.values())
    assert native_error < .00003
    legacy = read(ramp / "legacy_replay.json")
    assert legacy["scalar_comparisons"]==35*7*16 and legacy["sampled_histories"]==35
    assert max(v["maximum_absolute_C"] for v in legacy["checks"].values())<.001
    frozen = read(dictionary / "input_deltas/order/normalization/results/summary.json")
    assert frozen["histories"] == n*t and max(frozen["max_full_replay_C"].values()) < 1e-5
    return dict(official_complete_graph_scalars=sum(v["scalars"] for v in arms.values()),
                maximum_complete_graph_difference_C=native_error, fixed_normalization_histories=n*t,
                original_dependency_stack_scalars=legacy["scalar_comparisons"],
                weather_histories=n*t, independent_ramp_estimates_and_intervals=20,
                independently_verified_detector_feature_scores=12, template_forecasts_joined=n*t)


def external_dictionaries():
    """Recompute all three receiving-seed headline rows and the matched-angle control."""
    d = HERE / "dictionary_transfer"
    extra = d / "more_seeds/results"
    r = read(extra / "summary.json")
    with np.load(d / "weather/results/22112201.npz") as f:
        atoms, rows, cols, sd = f["atoms"], f["rows"], f["cols"], f["input_sd"]
        source = (f["donor"]-f["noop"]).astype(float)
        first = {arm:(f[arm]-f["noop"]).astype(float) for arm in ("corr_own","corr_forced","group16")}
        np.testing.assert_array_equal(f["origin"], np.tile(np.arange(4),46))
    eligible = np.where([len(np.unique(rows[cols==j]//4)) >= 8 for j in range(len(atoms))])[0]
    np.testing.assert_array_equal(atoms[eligible], r["eligible_atoms"])
    keep = np.isin(cols, eligible)
    assert keep.sum()==663
    def standardized(z):
        a = z / sd[rows,None]
        return a-a.mean(1,keepdims=True)
    ref=standardized(source)
    for seed, entry in r["receivers"].items():
        for arm, initial in first.items():
            if seed=="22112202":
                value=initial
            else:
                with np.load(extra / f"{seed}_{arm}.npz") as f:
                    value=f["effect"].astype(float)
                    np.testing.assert_array_equal(f["rows"],rows)
                    np.testing.assert_array_equal(f["cols"],cols)
            error=(standardized(value)-ref)**2
            per=[1-error[cols==j].sum()/(ref[cols==j]**2).sum() for j in eligible]
            exp=entry["metrics"]["history_sd_centered"]["arms"][arm]
            np.testing.assert_allclose(np.mean(per),exp["macro_skill"]["point"],atol=1e-9)
            np.testing.assert_allclose(per,[exp["per_donor"][str(a)] for a in atoms[eligible]],atol=1e-9)
            physical=np.sqrt(np.mean((value[keep]-source[keep])**2))
            np.testing.assert_allclose(physical,entry["metrics"]["celsius_full"]["arms"][arm]["error_rms"],atol=1e-9)
    angle = d / "angle_control/results"
    with np.load(d / "weather/detector/results/fit.npz") as f:
        group, indices = f["group"], f["donor_cols"]
    with np.load(d / "results/large_22112201_to_large_matches.npz") as f:
        donor = f["source_decoder"][indices].astype(float)
    for f in angle.glob("*.npz"):
        with np.load(f) as arm:
            vec=arm["directions"]
            np.testing.assert_allclose(np.linalg.norm(vec,axis=1),np.linalg.norm(group,axis=1),rtol=1e-11)
            np.testing.assert_allclose((vec*donor).sum(1),(group*donor).sum(1),rtol=1e-11)
            assert float(arm["native_error_C"])<1e-5
    assert len(list(angle.glob("*.npz")))==5
    return dict(evidence_cells=3*3*len(eligible), live_receiver_seeds=3,
                verified_physical_RMS_errors=9, matched_angle_vectors=5*len(indices))


def local_links():
    broken=[]
    for doc in HERE.rglob("*.md"):
        for target in re.findall(r'\]\(([^\s)]+)\)',doc.read_text()):
            if target.startswith(("https://","http://","#","mailto:","data:")):continue
            dest=target.split("#")[0]
            if dest and not (doc.parent/dest).exists():broken.append([str(doc.relative_to(HERE)),target])
    if broken:raise AssertionError(broken)
    return len(list(HERE.rglob("*.md")))


def main():
    checks=dict(forecasts=forecast_quality(),mechanisms=mechanisms(),transfer=transfer_controls(),new_panel=new_panel_effects(),external_dictionaries=external_dictionaries(),local_documents_checked=local_links())
    keys=["dictionary_transfer/results/overview.json","dictionary_transfer/weather/results/summary.json",
          "forecast_quality/results/summary.json","horizon_prefix/new_dates/results/summary.json",
          "head_test/results/summary.json","forecast_probe/results/fit.json",
          "dictionary_transfer/input_deltas/new_dates/results/summary.json","dictionary_transfer/input_deltas/template/results/summary.json",
          "dictionary_transfer/weather/new_dates/results/summary.json","review/results/review.json","confirmation.json",
          "dictionary_transfer/more_seeds/results/summary.json","dictionary_transfer/angle_control/results/summary.json",
          "dictionary_transfer/input_deltas/order/results/paired_order_difference.json",
          "../shape_transfer/physical_ramp/new_dates/results/scores.json",
          "dictionary_transfer/input_deltas/new_dates/results/official_replay.json","dictionary_transfer/input_deltas/new_dates/results/legacy_replay.json",
          "dictionary_transfer/input_deltas/order/normalization/results/summary.json",
          "../shape_transfer/physical_ramp/new_dates/past_only/results/scores.json"]
    checks["sha256"]={name:sha256((HERE/name).read_bytes()).hexdigest() for name in keys}
    (HERE/"verification.json").write_text(json.dumps(checks,indent=2)+"\n")


if __name__=="__main__":main()
