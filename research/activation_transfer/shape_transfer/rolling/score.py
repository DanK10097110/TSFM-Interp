"""Series-clustered recovery and coverage at each relative forecast origin."""
import json
from collections import Counter
import numpy as np

from research.activation_transfer.shape_transfer.feature_groups.score import recover, transform, weights
from research.activation_transfer.shape_transfer.rolling.data import CACHE, LAGS, OUT
from research.activation_transfer.shape_transfer.rolling.intervene import ARMS
from research.activation_transfer.shape_transfer.sparse_stability.score import interval, rate, rat


def summarize_scores(ref, effects, mask, draws, macro=False):
    eligible = mask.sum(0) >= 8 if macro else mask.any(0)
    out = dict(cases=int(mask[:,eligible].sum()), eligible_columns=np.flatnonzero(eligible).tolist(),
               series=int(mask[:,eligible].any(1).sum()))
    for part in ("shape", "full"):
        y = transform(ref[:,eligible],part)
        results = {key:recover(mask[:,eligible], y, transform(effect[:,eligible],part), draws, pooled=not macro)
                   for key,effect in effects.items()}
        out[part] = {k:interval(v) for k,v in results.items()}
        for x,z in (("fresh","refit"),("fresh","stale"),("refit","bounded"),("fresh","old_signature"),
                    ("old_native_rescaled","old_native")):
            out[part][x+"_minus_"+z] = interval(results[x]-results[z])
    return out


def per_series_rms(signal, mask, draws):
    e = (np.square(signal.astype(float)).mean(-1)*mask).sum(1)
    n = mask.sum(1)
    return interval(np.sqrt(rat(draws@e,draws@n)))


def quantify(source):
    f = dict(np.load(OUT/(source+".npz")))
    with np.load(CACHE/"data.npz") as frame:
        np.testing.assert_array_equal(frame["ids"],f["ids"])
        x = frame["x"]
    series = len(f["ids"])//len(LAGS)
    out = {}
    for domain in np.unique(f["split"]):
        r0 = np.flatnonzero(f["split"][:series] == domain)
        draw = weights(len(r0))
        old = f["fresh_id"][r0]
        initial = f["codes"][r0]>0
        old_ref = f["native"][r0]
        old_mass = f["codes"][r0]
        rows = {}
        for step, lag in enumerate(LAGS):
            ix = r0+series*step
            current = f["codes"][ix]>0
            common = initial & current
            stopped = initial & ~current
            new = f["fresh_id"][ix]
            old_survives = (old[:,:,None]==f["support"][ix,None,:]).any(-1) & (old>=0)
            new_existed = (new[:,:,None]==f["support"][r0,None,:]).any(-1) & (new>=0)
            both_ranked = common & old_survives
            switch = common & (old!=new)
            assert not ((old<0)|(new<0))[common].any()
            support_overlap = (f["support"][r0,:,None]==f["support"][ix,None,:]).any(-1)
            fixed_waveform = f["old_signature_id"][ix]
            unit_ratio = np.divide(f["codes"][ix],old_mass,out=np.zeros_like(old_mass),where=old_mass>0)
            arms = {key:f[key+"_effect"][ix] for key in ARMS}
            arms.update(old_native=old_ref,old_native_rescaled=unit_ratio[:,:,None]*old_ref,
                        source_linear=f["source_linear"][ix],
                        old_signature_linear=-f["codes"][ix,:,None]*f["source_unit"][r0],
                        noop=np.zeros_like(old_ref))
            ref = f["native"][ix]
            row = dict(n=len(ix), initial=int(initial.sum()), current=int(current.sum()),
                       source_survival=rate(common,initial,draw), new_source=rate(current&~initial,current,draw),
                       prior_target_available=rate(common&old_survives,common,draw),
                       stale_amount_exceeds_current_positive_bound=rate(
                           common & old_survives & (np.abs(f["stale_coef"][ix]) > np.abs(f["bounded_coef"][ix])+1e-8),
                           common & old_survives,draw),
                       full_target_support_retention=rate(support_overlap,np.ones_like(support_overlap),draw),
                       switch=rate(switch,common,draw),
                       switch_old_left=rate(switch&~old_survives,switch,draw),
                       switch_old_survived_new_entered=rate(switch&old_survives&~new_existed,switch,draw),
                       switch_both_survived=rate(switch&old_survives&new_existed,switch,draw),
                       switched_with_old_signature=rate(both_ranked & (fixed_waveform!=old),both_ranked,draw),
                       fresh_changes_only_when_source_shape_updated=rate(switch&old_survives&(fixed_waveform==old),
                                                                       switch&old_survives,draw),
                       fresh_id_equals_old_signature_id=rate(common&(new==fixed_waveform),common,draw),
                       source_coefficient_ratio_median=float(np.median(unit_ratio[common])) if common.any() else None,
                       context_change_rms_quantiles=np.quantile(np.sqrt(np.mean((x[ix]-x[r0]).astype(float)**2,1)),
                                                               [.1,.5,.9]).tolist(),
                       original_alive_any_now_dead=rate(stopped.any(1),initial.any(1),draw),
                       all_original_firings=summarize_scores(ref,arms,initial,draw),
                       persistent=summarize_scores(ref,arms,common,draw),
                       persistent_macro=summarize_scores(ref,arms,common,draw,True))
            stale_full = arms["stale"].astype(float)
            numerator = (np.square(stale_full).sum(-1)*stopped).sum(1)
            denom = (np.square(stale_full-ref).sum(-1)*initial).sum(1)
            row["stopped_source"] = dict(cases=int(stopped.sum()), rms_stale=per_series_rms(stale_full,stopped,draw),
                        rms_bounded=per_series_rms(arms["bounded"],stopped,draw),
                        stale_error_fraction=interval(rat(draw@numerator,draw@denom)))
            if lag:
                row["target_exited"] = summarize_scores(ref,arms,common & ~old_survives,draw)
                row["target_retained"] = summarize_scores(ref,arms,common & old_survives,draw)
                pairs = Counter((int(f["atoms"][j]),int(old[i,j])) for i,j in np.argwhere(common & ~old_survives))
                row["target_exited"]["donor_original_target_pairs"] = [
                    {"donor":a,"target":b,"cases":n} for (a,b),n in pairs.most_common()]
                arriving = x[ix,-lag:].astype(float)
                innovation = np.sqrt(np.mean((arriving - x[r0,-1:])**2,axis=1))
                high = np.zeros(len(ix), dtype=bool)
                high[np.argsort(innovation,kind="stable")[len(ix)//2:]] = True
                strata = {"median_incoming_rms":float(np.median(innovation))}
                for label,gate in (("upper",high),("lower",~high)):
                    strata[label] = {"survival":rate(common & gate[:,None],initial & gate[:,None],draw),
                                     "switch":rate(switch & gate[:,None],common & gate[:,None],draw)}
                for label,num,den in (("survival",common,initial),("switch",switch,common)):
                    hi = rat(draw@(num.sum(1)*high),draw@(den.sum(1)*high))
                    lo = rat(draw@(num.sum(1)*~high),draw@(den.sum(1)*~high))
                    strata["upper_minus_lower_"+label] = interval(hi-lo)
                row["incoming_observations"] = strata
            rows[str(lag)] = row
        out[str(domain)] = rows
    return out


if __name__=="__main__":
    result = {source:quantify(source) for source in ("bolt","timesfm")}
    (OUT/"summary.json").write_text(json.dumps(result,allow_nan=False,separators=(",",":"))+"\n")
