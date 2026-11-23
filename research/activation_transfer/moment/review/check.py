"""Independent reconciliation of MOMENT causal-transfer estimands and the post-release temperature panel."""
from datetime import date, timedelta
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
M = HERE.parent
D = M / "dictionary_transfer"
W = M / "forecast_quality/post_release/results"
GEOM = Path("/tmp/tsfm_moment_nov22_geometry")
BLIND = Path("/tmp/tsfm_blind_frequency_20261121/data.npz")
SNAPSHOT = Path("/tmp/tsfm_moment_quality_20261122/noaa/CRND0103-202611160850.zip")
SEEDS = (22112201, 22112202)
HASHES = {}


def hasharr(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@lru_cache
def sha(path):
    with path.open("rb") as f:
        value = hashlib.file_digest(f, "sha256").hexdigest()
    key = str(path.relative_to(M)) if path.is_relative_to(M) else str(path)
    HASHES[key] = value
    return value


def load(path):
    sha(path)
    with np.load(path) as f:
        return dict(f)


def check_equal(a,b):
    np.testing.assert_array_equal(a,b)


def summary(reference, arms, rows, cols, selected, site, centered, draws=None):
    """Loop over actual donor indices, explicitly remove waveform means and measure pooled loss."""
    limit = np.unique(site).size
    out = {}
    y = np.asarray(reference,float)
    if centered:
        y = y-y.mean(1,keepdims=True)
    selected_atoms = sorted(set(cols[selected]))
    assert all(len(set(site[rows[selected&(cols==atom)]]))>=8 for atom in selected_atoms)
    for name, cand in arms.items():
        v = np.asarray(cand,float)
        if centered:
            v = v-v.mean(1,keepdims=True)
        energy, errors = [], []
        for atom in selected_atoms:
            take = selected & (cols==atom)
            energy.append(np.sum(y[take]**2))
            errors.append(np.sum((v[take]-y[take])**2))
        energy, errors = np.array(energy), np.array(errors)
        skill = 1-errors/energy
        yy,vv = y[selected],v[selected]
        sq = np.square(vv-yy)
        r = dict(macro=float(skill.mean()), median=float(np.median(skill)),
                 energy_weighted=float(1-sq.sum()/np.square(yy).sum()),
                 l1_skill=float(1-np.abs(vv-yy).sum()/np.abs(yy).sum()),
                 mae=float(abs(vv-yy).mean()), rmse=float(np.sqrt(sq.mean())),
                 donor_rms=float(np.sqrt((yy*yy).mean())), target_rms=float(np.sqrt((vv*vv).mean())),
                 donor_mean_abs=float(abs(yy).mean()), target_mean_abs=float(abs(vv).mean()),
                 mean_signed_error=float((vv-yy).mean()))
        np.testing.assert_allclose(r["energy_weighted"], np.average(skill,weights=energy),atol=1e-12)
        if draws is not None:
            atom_goal=np.zeros((limit,len(selected_atoms)))
            atom_loss=np.zeros_like(atom_goal)
            for j,atom in enumerate(selected_atoms):
                mask=selected&(cols==atom)
                group=site[rows[mask]]
                atom_goal[:,j]=np.bincount(group,weights=(y[mask]**2).sum(1),minlength=limit)
                atom_loss[:,j]=np.bincount(group,weights=((v[mask]-y[mask])**2).sum(1),minlength=limit)
            energies=atom_goal[draws].sum(1);batch_loss=atom_loss[draws].sum(1)
            per=1-np.divide(batch_loss,energies,out=np.full_like(energies,np.nan),where=energies>0)
            r["macro_station95"]=np.quantile(np.nanmean(per,axis=1),[.025,.975]).tolist()
            r["largest_donor_energy_fraction"]=float(energy.max()/energy.sum())
            r["largest_three_donor_energy_fraction"]=float(np.sort(energy)[-3:].sum()/energy.sum())
            who=site[rows[selected]]
            losses=np.bincount(who,weights=sq.sum(1),minlength=limit)
            goals=np.bincount(who,weights=(yy**2).sum(1),minlength=limit)
            absolute=np.bincount(who,weights=abs(vv-yy).sum(1),minlength=limit)
            absgoals=np.bincount(who,weights=abs(yy).sum(1),minlength=limit)
            r["energy_weighted_station95"]=np.quantile(1-losses[draws].sum(1)/goals[draws].sum(1),[.025,.975]).tolist()
            r["l1_station95"]=np.quantile(1-absolute[draws].sum(1)/absgoals[draws].sum(1),[.025,.975]).tolist()
        out[name]=r
    return out


def eligible(rows,cols,domains,selection,cluster):
    allowed=[]
    for atom in set(cols):
        ix=selection[rows] & (cols==atom)
        if len(set(cluster[rows[ix]]))>=8:
            allowed.append(atom)
    return selection[rows]&np.isin(cols,allowed),allowed


@lru_cache
def checkpoint_arrays(size,seed):
    import torch
    file=GEOM/f"{size}/sae_{seed}.pt"
    sha(file)
    serialized=torch.load(file,map_location="cpu",weights_only=True)
    assert int(serialized["k"])==32
    return {key:v.numpy() for key,v in serialized["state_dict"].items()}


def code(h,weights):
    pre=np.maximum((h-weights["b_dec"])@weights["W_enc"]+weights["b_enc"],0)
    keep=np.argpartition(pre,-32,axis=1)[:,-32:]
    y=np.zeros_like(pre)
    np.put_along_axis(y,keep,np.take_along_axis(pre,keep,axis=1),axis=1)
    return y


@lru_cache
def ordinary_codes(size,seed,split):
    path=GEOM/size/(split+".npz");sha(path)
    with np.load(path) as f:
        assert int(f["positions"][0])==61
        return f["ids"],code(f["middle"][:,0],checkpoint_arrays(size,seed))


def climatology(values, all_dates, before):
    phase=lambda d:(date(2000,d.month,d.day)-date(2000,1,1)).days
    phase_before=np.array([phase(d) for d in before])
    src=np.array([values.get(d,-9999.) for d in before])
    valid=src!=-9999.
    sums=np.bincount(phase_before[valid],weights=src[valid],minlength=366)
    counts=np.bincount(phase_before[valid],minlength=366)
    sums=sum(np.roll(sums,k) for k in range(-7,8));counts=sum(np.roll(counts,k) for k in range(-7,8))
    curve=sums/counts;anomalies=src-curve[phase_before]
    both=valid[1:]&valid[:-1]
    rho=float(np.clip(np.sum(anomalies[1:][both]*anomalies[:-1][both])/np.sum(anomalies[:-1][both]**2),0,1))
    return {d:float(curve[phase(d)]) for d in all_dates},rho


def cuts_and_labels():
    panel=load(W/"panel.npz")
    manifest=json.loads((W/"manifest.json").read_text())
    sha(W/"manifest.json")
    assert sha(SNAPSHOT)==manifest["archive_sha256"]
    assert hasharr(panel["context"])==manifest["sha256_context"] and hasharr(panel["future"])==manifest["sha256_future"]
    check_equal(panel["ids"],manifest["accepted"])
    origins=[date.fromisoformat(k) for k in manifest["origins"]]
    assert all(s.year==2026 for s in origins)
    cases,visible,filled=0,0,0
    start=date(2008,1,1)
    before=[start+timedelta(days=i) for i in range((date(2024,1,1)-start).days)]
    all_dates=[start+timedelta(days=i) for i in range((origins[-1]-start).days+16)]
    max_impute=max_climate=max_decay=0.
    unique_targets={cut+timedelta(days=t) for cut in origins for t in range(16)}
    later_input={d for d in unique_targets if any(timedelta(0)<cut-d<=timedelta(days=480)
                                                 for cut in origins)}

    with ZipFile(SNAPSHOT) as z:
        for i,station in enumerate(panel["ids"]):
            dates={}
            for year in range(2008,2027):
                name=f"{year}/CRND0103-{year}-{station}.txt"
                if name not in z.NameToInfo:
                    continue
                for line in z.read(name).splitlines():
                    fields=line.split()
                    dt=fields[1].decode()
                    dates[date(int(dt[:4]),int(dt[4:6]),int(dt[6:8]))]=float(fields[8])
            curve,rho=climatology(dates,all_dates,before)
            for k,cut in enumerate(origins):
                end=cut-timedelta(days=1)
                assert str(end)==manifest["ranges"][k]["history_last"]
                future=[dates[cut+timedelta(days=l)] for l in range(16)]
                check_equal(future,panel["future"][i,k])
                cases+=16
                raw=np.array([dates.get(cut-timedelta(days=480-l),-9999.) for l in range(480)])
                known=raw!=-9999.
                check_equal(raw[known],panel["context"][i,k,known])
                assert int((~known).sum())==panel["imputed_days"][i,k]
                visible+=int(known.sum());filled+=int((~known).sum())
                for jj in np.flatnonzero(~known):
                    day=cut-timedelta(days=480-int(jj));last=day-timedelta(days=1)
                    while dates.get(last,-9999.)==-9999.:
                        last-=timedelta(days=1)
                    val=curve[day]+rho**((day-last).days)*(dates[last]-curve[last])
                    max_impute=max(max_impute,abs(panel["context"][i,k,jj]-val))
                endvalue=panel["context"][i,k,-1]
                for lead in range(16):
                    val=curve[cut+timedelta(days=lead)]
                    max_climate=max(max_climate,abs(val-panel["clim"][i,k,lead]))
                    anom=val+(rho**(lead+1))*(endvalue-curve[end])
                    max_decay=max(max_decay,abs(anom-panel["anomaly"][i,k,lead]))
    assert max(max_impute,max_climate,max_decay)<1e-10

    provider_file=HERE/"results/publisher_revisions.json"
    sha(provider_file)
    publisher=json.loads(provider_file.read_text())
    for name,artifact in publisher.items():
        assert date.fromisoformat(artifact["revision_last_modified"][:10])<min(origins)
        metadata=M/f"forecast_quality/results/{name}.json"
        sha(metadata)
        rec=json.loads(metadata.read_text())
        folder=Path(rec["path"]);weights=folder/"model.safetensors"
        assert sha(weights)==artifact["safetensors_lfs_sha256"]==rec["hashes"]["model.safetensors"]
        assert weights.stat().st_size==artifact["safetensors_bytes"]
        cfg=(folder/"config.json").read_bytes()
        assert hashlib.sha1(b"blob "+str(len(cfg)).encode()+b"\0"+cfg).hexdigest()==artifact["config_git_blob"]
    pred={"persistence":np.repeat(panel["context"][:,:,-1:],16,2),"climate_anomaly":panel["anomaly"]}
    for model in ("moment_small","moment_base","moment_large","timesfm","bolt"):
        f=load(W/(model+".npz"))
        check_equal(f["ids"], panel["ids"])
        assert str(f["context_sha256"])==manifest["sha256_context"]
        pred[model]=f["raw_16"]
    return panel,dict(n_stations=len(panel["ids"]),origins=manifest["origins"],snapshot_sha256=sha(SNAPSHOT),
            dates_per_station_appearing_in_a_later_history=len(later_input),unique_forecast_dates=len(unique_targets),
            independently_rebuilt_imputations_max_c=max_impute,prior_climatology_max_c=max_climate,
            prior_climate_anomaly_max_c=max_decay,exact_future_scalar_values=cases,observed_input_values=visible,prior_imputed_inputs=filled,
            point_mae_c={name:float(sum(np.mean(abs(guess-panel["future"]),axis=(1,2)))/len(panel["ids"])) for name,guess in pred.items()})


def station_panel(panel,seed):
    z=load(D/f"weather/results/{seed}.npz")
    archived=json.loads((D/"weather/results/summary.json").read_text())[str(seed)]
    dates=json.loads((W/"manifest.json").read_text())["origins"]
    check_equal(z["ids"],[f"{s}/{d}" for s in panel["ids"] for d in dates])
    r,c,st=z["rows"],z["cols"],z["station"]
    check_equal(st,np.repeat(np.arange(46),4));check_equal(z["origin"],np.tile(np.arange(4),46))
    rows,cols=np.nonzero(z["z"]>0);check_equal(r,rows);check_equal(c,cols)
    other=int(z["recipient_seed"]);assert other!=seed
    expected_match=load(D/f"results/large_{seed}_to_large_matches.npz")
    assert sha(D/f"results/large_{seed}_to_large_matches.npz")==str(z["match_sha256"])
    for s,k in ((seed,"source_sae_sha256"),(other,"recipient_sae_sha256")):
        assert sha(GEOM/f"large/sae_{s}.pt")==str(z[k])
    assert str(z["checkpoint_sha256"])==sha(Path("/tmp/tsfm_moment_native/large/model.safetensors"))
    lookup=load(D/f"results/large_{seed}_to_large_groups.npz")
    check_equal(z["atoms"],expected_match["atoms"]);check_equal(z["matched_ids"],expected_match["corr_id"])
    check_equal(z["group_ids"],lookup["sixteen_ids"]);check_equal(z["group_coeff"],lookup["sixteen_coeff"])
    wc=Path("/tmp/tsfm_moment_dictionary_weather_20261122/clean.npz")
    with np.load(wc) as ordinary:
        check_equal(ordinary["ids"],z["ids"])
        for code_name,sae_seed,atom_ids in (("z",seed,z["atoms"]),("target_z",other,z["matched_ids"])):
            independent=code(ordinary["middle"],checkpoint_arrays("large",sae_seed))[:,atom_ids]
            check_equal(independent>0,z[code_name]>0)
            np.testing.assert_allclose(independent,z[code_name],atol=1e-4,rtol=1e-5)
    dw=checkpoint_arrays("large",other)["W_dec"]
    recovered=np.einsum("ni,nid->nd",z["group_coeff"],dw[z["group_ids"]])
    np.testing.assert_allclose(recovered,lookup["sixteen_vector"],atol=1e-6)
    raw=panel["context"].reshape(-1,480)
    assert str(z["context_sha256"]) in (hasharr(raw),hasharr(raw.astype("f4")))
    np.testing.assert_allclose(z["input_sd"],raw.std(1),rtol=3e-6)
    baseline=load(W/"moment_large.npz")["raw_16"].reshape(-1,16)
    max_gap=float(np.max(np.abs(baseline-z["base"])))
    assert max_gap<1e-4
    selected,atoms=eligible(r,c,st,np.ones(len(st),bool),st)
    assert len(atoms)==archived["eligible"] and sum(selected)==archived["eligible_events"]
    raw_y=z["donor"].astype(float)-z["noop"]
    raw_est={k:z[k].astype(float)-z["noop"] for k in ("corr_own","corr_forced","geom_forced","random","group16")}
    closed=z["target_z"][r,c]==0
    assert np.max(abs(raw_est["corr_own"][closed]))<1e-5
    native_shadow=max(float(abs((z[s].astype(float)-z["noop"])-(z[s+"_head"].astype(float)-z["noop_head"])).max()) for s in ("donor",*raw_est))
    draw=np.random.default_rng(2026112231).integers(46,size=(3000,46))
    summaries={};paired={}
    for units,scale in (("celsius",1),("history_sd",z["input_sd"][r,None])):
        for center in (True,False):
            label=units+("_centered" if center else "_full")
            summary0=summary(raw_y/scale,{k:v/scale for k,v in raw_est.items()},r,c,selected,st,center,draw)
            for arm,rec in summary0.items():
                prev=archived["statistics"][label]["arms"][arm]
                np.testing.assert_allclose(rec["macro"],prev["skill"],atol=2e-5)
                np.testing.assert_allclose(rec["donor_rms"],prev["donor_pooled_rms"],rtol=2e-5)
            summaries[label]=summary0
            ref=(raw_y/scale)[selected]
            vectors={k:(v/scale)[selected] for k,v in raw_est.items()}
            if center:
                ref=ref-ref.mean(1,keepdims=True)
                vectors={k:v-v.mean(1,keepdims=True) for k,v in vectors.items()}
            who=st[r[selected]]
            totals=lambda v:np.bincount(who,weights=v.sum(1),minlength=46)
            e=totals(ref**2);abs_e=totals(abs(ref))
            group_error=vectors["group16"]-ref
            diffs={}
            for name,v in vectors.items():
                if name=="group16":
                    continue
                residual=v-ref
                dl=totals(residual**2-group_error**2)
                da=totals(abs(residual)-abs(group_error))
                diffs[name]=dict(energy_weighted_point=float(dl.sum()/e.sum()),
                    energy_weighted_station95=np.quantile(dl[draw].sum(1)/e[draw].sum(1),[.025,.975]).tolist(),
                    l1_point=float(da.sum()/abs_e.sum()),
                    l1_station95=np.quantile(da[draw].sum(1)/abs_e[draw].sum(1),[.025,.975]).tolist())
            paired[label]=diffs
    return dict(eligible_atoms=len(atoms),events=int(sum(selected)),all_firing_events=len(r),
                matched_cofires=int(((z["target_z"][r,c]>0)&selected).sum()),
                waveform_vs_head_shadow_max_c=native_shadow,independent_forecast_join_max_c=max_gap,summaries=summaries,paired_group_minus_single=paired)


def transfer(baseline,source,target,seed):
    stem=f"{source}_{seed}_to_{target}"
    z=load(D/f"results/{stem}_forecast.npz")
    group_registered=source==target or seed==SEEDS[0]
    if group_registered:
        g=load(D/f"results/{stem}_group_waves.npz")
        for k in ("ids","row","col"):check_equal(z[k],g[k])
    tr=load(GEOM/f"{source}/train.npz")
    check_equal(z["ids"],baseline["ids"]);check_equal(z["split"],baseline["split"])
    assert len(set(z["ids"])&set(tr["ids"]))==0 and len(set(z["ids"]))==256
    r,c=np.nonzero(z["z"]>0);check_equal(r,z["row"]);check_equal(c,z["col"])
    donor=load(GEOM/f"{source}/ablation_{seed}.npz")
    for a,b in (("row","row"),("col","col"),("ids","ids"),("atoms","atoms")):check_equal(z[a],donor[b])
    check_equal(z["truth_native"],donor["physical"])
    assert np.max(abs(z["z"]-donor["codes"]))<5e-5
    axes=load(D/f"results/{stem}_matches.npz")
    check_equal(axes["atoms"],z["atoms"])
    other=int(axes["recipient_seed"])
    trids,st=ordinary_codes(source,seed,"train")
    tids,tt=ordinary_codes(target,other,"train")
    check_equal(trids,tids)
    aa=st[:,z["atoms"]].astype(float);bb=tt.astype(float)
    aa-=aa.mean(0);bb-=bb.mean(0)
    freq=(tt>0).sum(0)>=8
    check_equal(freq,axes["eligible_target"])
    aa/=np.linalg.norm(aa,axis=0)
    bb[:,freq]/=np.linalg.norm(bb[:,freq],axis=0)
    corr=aa.T@bb;corr[:,~freq]=-np.inf
    check_equal(corr.argmax(1),axes["corr_id"])
    for size,chosen,atom_ids,key in ((source,seed,axes["atoms"],"z"),(target,other,axes["corr_id"],"target_codes")):
        ids,cc=ordinary_codes(size,chosen,"eval")
        check_equal(z["ids"],ids)
        check_equal(z[key]>0,cc[:,atom_ids]>0)
        np.testing.assert_allclose(z[key],cc[:,atom_ids],atol=1e-4,rtol=1e-5)
    if group_registered:
        group=load(D/f"results/{stem}_groups.npz")
        recovered=np.einsum("ni,nid->nd",group["sixteen_coeff"],checkpoint_arrays(target,other)["W_dec"][group["sixteen_ids"]])
        np.testing.assert_allclose(recovered,group["sixteen_vector"],atol=1e-6)
    if source==target:
        check_equal(axes["source_decoder"],axes["mapped"])
    else:
        m=load(D/f"results/{source}_to_{target}_map.npz")
        np.testing.assert_allclose(axes["source_decoder"].astype(float)@m["w"],axes["mapped"],atol=5e-6)
        im=json.loads((D/f"results/{source}_to_{target}_map.json").read_text())
        mapsets=[set(im[k]) for k in ("training_ids","validation_ids","blind_map_ids")]
        assert all(s.issubset(set(tr["ids"])) and not s&set(z["ids"]) for s in mapsets)
        assert all(not a&b for i,a in enumerate(mapsets) for b in mapsets[i+1:])
    yt=z["truth_native"]
    arms={"own_single":z["native_corr_own"],"forced_single":z["native_corr_forced"],
          "geometric_dense":z["native_dense"],"adaptive":z["native_response"]}
    if group_registered:
        arms["group16"]=g["sixteen_native"]
    extra={}
    rotation=D/f"results/{stem}_rotation.npz"
    if rotation.exists():
        rr=load(rotation)
        for key in ("ids","row","col"):
            check_equal(z[key],rr[key])
        arms.update(rotated16=rr["rotated"],rotated16_equal_norm=rr["rotated_equal"],state_pca16=rr["pca16"])
    if source!=target and seed==SEEDS[0]:
        fit=load(D/f"functional/results/{stem}_fit.npz")
        assert not set(fit["ids"])&set(z["ids"]) and set(fit["ids"]).issubset(set(tr["ids"]))
        gref=load(D/f"functional/results/{source}_gradient.npz")
        check_equal(gref["ids"],fit["ids"])
        assert np.count_nonzero(gref["split"]=="fit")==96 and np.count_nonzero(gref["split"]=="cal")==32
        for key in ("dense","group","sham"):
            preds=load(D/f"functional/results/{stem}_{key}.npz")
            for join in ("ids","row","col"):check_equal(z[join],preds[join])
            arms["historical_"+key]=preds["effect"]
    budget = D/f"functional/results/{stem}_budget.npz"
    if seed==SEEDS[0] and source!=target and budget.exists():
        bd=load(budget)
        for key in ("ids","row","col"):check_equal(bd[key],z[key])
        arms.update(adaptive_equal_cap=bd["current_matched"],wrong_equal_dose=bd["wrong_matched"])
        assert np.max(bd["current_matched_length"]-bd["cap"])<1e-4
        np.testing.assert_allclose(bd["current_matched_length"],bd["wrong_matched_length"],atol=2e-5,rtol=2e-5)
        assert np.all(bd["alternate"]!=np.arange(len(z["ids"])))
        check_equal(z["split"][bd["alternate"]],z["split"])
        # Recover requests directly from the source encoder's stored ordinary derivatives.
        with np.load(GEOM/f"{source}/eval.npz") as f:
            check_equal(f["ids"],z["ids"])
            jac=f["middle_jac"][:,:8,0]
        goal=np.einsum("nlh,nh->nl",jac[r].astype(float),-z["z"][r,c,None]*axes["source_decoder"][c])
        extra["max_abs_rebuilt_test_early_goal_error"]=float(abs(bd["early_goal"]-goal).max())
        assert extra["max_abs_rebuilt_test_early_goal_error"]<1e-5
    if source!=target:
        with np.load(GEOM/f"{source}/eval.npz") as sf, np.load(GEOM/f"{target}/eval.npz") as tf:
            check_equal(sf["ids"],tf["ids"])
            source_jac=sf["middle_jac"][:,:8,0]
            target_jac=tf["middle_jac"][:,:8,0]
        selected=np.linspace(0,len(r)-1,16).astype(int)
        maximum=0.
        for event in selected:
            row,col=r[event],c[event]
            direction=-float(z["z"][row,col])*axes["source_decoder"][col].astype(float)
            goal=source_jac[row].astype(float)@direction
            u,s,v=np.linalg.svd(target_jac[row].astype(float),full_matrices=False)
            candidate=v.T@((s/(s*s+.001*s[0]**2))*(u.T@goal))
            norm=np.linalg.norm(candidate)
            got=min(float(z["radius"]),norm)
            maximum=max(maximum,abs(got-float(z["response_length"][event])))
        assert maximum < 2e-4,maximum
        extra["independent_svd_16_events_max_adaptive_length_error"]=maximum
    src_json=json.loads((D/f"results/{stem}_forecast.json").read_text())
    if group_registered:
        group_json=json.loads((D/f"results/{stem}_group_scores.json").read_text())
    result={}
    use_clusters=np.arange(len(z["ids"]))
    for domain in np.unique(z["split"]):
        selected,eligible_atoms=eligible(r,c,z["split"],z["split"]==domain,use_clusters)
        entry=dict(events=int(sum(selected)),eligible_atoms=len(eligible_atoms))
        for a,sl in (("early",slice(8)),("late",slice(8,16))):
            est=summary(yt[:,sl],{k:v[:,sl] for k,v in arms.items()},r,c,selected,use_clusters,True)
            key=f"native_{a}_shape"
            np.testing.assert_allclose(est["adaptive"]["macro"],src_json[domain][key]["response"]["value"],atol=2e-5)
            if group_registered:
                np.testing.assert_allclose(est["group16"]["macro"],group_json[domain][key]["sixteen"]["value"],atol=2e-5)
            entry[a]={k:{m:n[m] for m in ("macro","energy_weighted","donor_rms","target_rms","l1_skill")} for k,n in est.items()}
        result[str(domain)]=entry
    return dict(domains=result,**extra)


def main():
    panel,labels=cuts_and_labels()
    data=load(BLIND)
    train=load(Path("/tmp/tsfm_atom_replication_20261120/data.npz"))
    include=train["split"]=="train"
    iid=set(train["ids"][include])
    fitting={hasharr(v) for v in train["x"][include]}; judgment={hasharr(v) for v in data["x"]}
    assert len(iid)==len(fitting)==1024 and len(judgment)==256
    assert not fitting&judgment and not iid&set(data["ids"])
    out=dict(training_evaluation_overlap=dict(training_distinct=1024,evaluation_distinct=256,
        duplicate_input_windows=0,duplicate_series_ids=0),post_release=labels, weather={str(s):station_panel(panel,s) for s in SEEDS},native_transfer={})
    incomplete=[]
    for source in ("small","large"):
        for target in ("small","large"):
            for seed in SEEDS:
                stem=f"{source}_{seed}_to_{target}"
                registered=["forecast.npz","forecast.json"]
                if source==target or seed==SEEDS[0]:
                    registered+=["group_waves.npz","group_scores.json"]
                missing=[v for v in registered if not (D/f"results/{stem}_{v}").exists()]
                if missing:
                    incomplete.append(dict(comparison=stem,missing=missing))
                    continue
                out["native_transfer"][stem]=transfer(data,source,target,seed)
    out["incomplete_archives_at_review"]=incomplete
    out["asset_sha256"]=HASHES
    (HERE/"results").mkdir(exist_ok=True)
    (HERE/"results/review.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
