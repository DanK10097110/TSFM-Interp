"""Build the human-readable numerical appendix from the saved, paired study outcomes."""
import argparse
from collections import Counter
import json

from research.activation_transfer.moment.forecast_quality.data import HERE, OUT
from research.activation_transfer.moment.forecast_quality.post_release.data import OUT as WEATHER

MODELS=["moment_small","moment_base","moment_large","bolt","timesfm"]
SHORT=["Small","Base","Large","Bolt","TimesFM"]
DOM={"m4_daily":"M4 daily","m4_weekly":"M4 weekly","m4_hourly":"M4 hourly","monash_traffic":"Traffic",
     "monash_electricity_hourly":"Electricity","bitbrains_fast_storage":"Bitbrains VM"}


def read(path):
    return json.loads(path.read_text())


def est(obj, dec=2):
    v=obj["value"]
    lo,hi=obj["ci95"]
    return f"{v:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"


def table(dst, columns, rows):
    dst.append("| "+" | ".join(columns)+" |")
    dst.append("|"+"|".join(["---"]*len(columns))+"|")
    for row in rows:
        dst.append("| "+" | ".join(map(str,row))+" |")
    dst.append("")


def main(core_only=False):
    summary=read(OUT/"summary.json")["holdout"]["scores"]
    weather=read(WEATHER/"summary.json")
    level=read(OUT/"level.json")
    horizons=read(OUT/"horizon.json")
    pages=["# Forecast-quality numerical appendix","",
           "All model outputs are the released direct point forecasts in the original data units. “Select” uses eight history-only validation windows to choose a classical forecast at each origin. CIs are paired 95% percentile intervals from 4,000 resamples. Complementary raw arrays and further metrics are in the results directories.",""]
    for h in ("h8","h16"):
        results=summary[h]
        pretty=h.upper()
        pages+=["## Six historical domains, "+pretty,"","The primary panel has 96 held-out series in each row and two origins per series. Each MASE1 entry normalizes the error at that origin by its observed-history mean absolute lag-one difference. Lower is better.",""]
        order=["last","history_select"]+MODELS
        cols=["Data","Last","Select"]+SHORT
        rows=[[label]+[f"{results['absolute'][d][k]['mase1']['value']:.3f}" for k in order] for d,label in DOM.items()]
        table(pages,cols,rows)
        pages+=["**Individual history-only controls, MASE1**","","Primary periods are 7 for M4 daily, 52 for weekly and 24 for hourly. Hourly series also get a 168-hour candidate. Four-cycle medians use as many complete periods as fit in the available history.",""]
        base_names=["last","drift","season1","median1","season2","median2","history_select"]
        rows=[]
        for d,label in DOM.items():
            methods=results["absolute"][d]
            rows.append([label,*[(f"{methods[m]['mase1']['value']:.3f}" if m in methods else "—") for m in base_names]])
        table(pages,["Data","Last","Drift","Primary seasonal","Primary median","168 seasonal","168 median","Select"],rows)
        pages+=["**Mean absolute error in original series units**","","M4 series mixture and Bitbrains trace magnitude make raw-unit aggregates scale-sensitive. The adjacent MASE1 table gives each series its own error denominator. Bitbrains reports CPU usage in MHz, electricity uses the released hourly demand scale and traffic values use the released occupancy scale.",""]
        rows=[[label]+[f"{results['absolute'][d][k]['mae']['value']:.4g}" for k in order] for d,label in DOM.items()]
        table(pages,cols,rows)
        pages+=["**Equal-domain mean ratio of MASE1**","","One is parity with the comparator. Ratios are formed within domain and then averaged with equal domain weight.",""]
        across=results["equal_domain_ratios"]
        rows=[[s]+[est(across[m+"_vs_"+k]) for k in ("last","history_select","bolt","timesfm")]
              for s,m in zip(SHORT[:3],MODELS[:3])]
        table(pages,["Release","÷ last","÷ classical selector","÷ Bolt","÷ TimesFM"],rows)
    pages+=["## 2026 station forecasts","","Daily temperature in degrees Celsius, four forecast origins on the same 46 stations. Intervals sample stations with all four origins together. MASE1 appears in the complete JSON.",""]
    methods=["last","lag365","climatology","climate_anomaly",*MODELS]
    names=["Last","Same day −365","Pre-2024 station climate","Climate + anomaly",*SHORT]
    rows=[[label]+[est(weather[h]["metrics"][m]["mae"]) for h in ("h8","h16","h16_last8")] for label,m in zip(names,methods)]
    table(pages,["Method","H8 MAE °C","H16 MAE °C","Leads 9–16 MAE °C"],rows)
    rows=[]
    for m,label in zip(MODELS,SHORT):
        comp=weather["h16"]["paired"][m+"_vs_climate_anomaly"]
        rows.append([label,est(comp["mae_ratio"]),est(comp["mae_difference_c"]),est(comp["region_block_mae_difference_c"])])
    table(pages,["Method","H16 ratio to climate + anomaly","Δ °C from station blocks","Δ °C from four Census region blocks"],rows)
    n=Counter(read(WEATHER/"manifest.json")["regions"])
    rows=[]
    for r,c in weather["h16"]["by_region_mae_c"].items():
        rows.append([r,n[r],*[f"{c[k]:.2f}" for k in ["climate_anomaly",*MODELS]]])
    table(pages,["Region","Stations","Climate + anomaly",*SHORT],rows)
    rows=[]
    for r,c in weather["h16"]["by_origin_mae_c"].items():
        rows.append([r,*[f"{c[k]:.2f}" for k in ["climate_anomaly",*MODELS]]])
    table(pages,["First forecast date","Climate + anomaly",*SHORT],rows)
    rows=[]
    for day,c in weather["h16"]["by_origin_mean_signed_error_c"].items():
        rows.append([day,*[f"{c[k]:+.2f}" for k in ["climate_anomaly",*MODELS]]])
    pages+=["Signed error in °C on those four H16 requests. Positive is too warm.",""]
    table(pages,["First forecast date","Climate + anomaly",*SHORT],rows)
    pages+=["## Level tracking and the native forecast horizon","","Correlations compare each predicted horizon mean against the actual future mean after subtracting each origin’s own history mean and dividing by its history SD. They describe pooled origin behavior, with no fitted adjustment. See level.json for spread, level/waveform error and the correlation with the last value.",""]
    table(pages,["Data",*SHORT],[[label]+[f"{level[d][k]['8']['corr_forecast_actual_level']:.3f}" for k in MODELS] for d,label in (*DOM.items(),("noaa","2026 stations"))])
    pages+=["Mean absolute displacement in history SD between direct H8 and H16's first eight. A different forecast at an identical lead is possible because MOMENT repositions its forecast mask patches for each request.",""]
    table(pages,["Data",*SHORT],[[label]+[f"{horizons[d][m]['mean_absolute_forecast_shift_in_past_sd']['value']:.4f}" for m in MODELS] for d,label in (*DOM.items(),("noaa","2026 stations"))])
    pages+=["## Recorded CPU wall time","","All models processed the same 1,440 six-domain windows in batches of sixteen, using two Torch threads for each process. Times are descriptive on a shared host and exclude model loading. Forecast error is assessed on 1,152 of these windows.",""]
    rows=[]
    for m,label in zip(MODELS,SHORT):
        perf=read(OUT/(m+".json"))["seconds"]
        rows.append([label,*[f"{perf[key]:.1f}" for key in ("raw_8","raw_16","standardized_8","standardized_16")]])
    table(pages,["Release","Raw H8 (s)","Raw H16 (s)","External z H8 (s)","External z H16 (s)"],rows)
    if core_only:
        (HERE/"RESULT_TABLES.md").write_text("\n".join(pages)+"\n")
        return
    pages+=["## Unit and batch checks","","Values below use the direct 16-step forecast on 48 preordered diagnostic histories. Errors are mean absolute forecast changes in units of the test trajectory’s original SD after inverting the transformation. Inputs began with unit SD; the transformation scales are relative to that value. Each constant's native absolute residual and every per-item maximum is stored in its normalization JSON.",""]
    rows=[]
    for m, label in zip(MODELS,SHORT):
        norm=read(OUT/(m+"_normalization.json"))["16"]
        vals=[norm["moderate_scale_shift"]["forecast"]["mean_abs"], norm["near_epsilon"]["forecast"]["mean_abs"],
              norm["below_epsilon"]["forecast"]["mean_abs"], norm["large_offset"]["forecast"]["mean_abs"],
              norm["batch"]["single_vs_16"]["max_abs"],norm["batch"]["extreme_companions"]["max_abs"]]
        rows.append([label,*[f"{v:.3g}" for v in vals]])
    table(pages,["Release","3.7× + 11.2","10⁻⁵×","10⁻⁷×","1× + 10⁴","Single vs 16, max","Mixed companions, max"],rows)
    pages+=["## Reconstruction contrasts","","The same eight real observations per item are evaluated on 12 diagnostic series from each of six domains. Causal paths have only the 416-point prefix. Interior conditional modes receive the specified right evidence; the visible condition also exposes all eight targets. Manual paths hold prefix normalization fixed. Entries are MAE divided by prefix SD, equally averaging the selected cases.",""]
    recon={k:read(OUT/(k+"_reconstruction.json")) for k in MODELS[:3]}
    names={"last":"Last","prefix_mean":"Prefix mean","two_sided_linear":"Two-sided interpolation","native_causal":"Official causal",
           "manual_terminal":"Independent terminal replay","interior_causal":"Interior position, prefix only",
           "interior_true_right":"Interior, true right flank","interior_wrong_right":"Interior, wrong right flank",
           "native_interior":"Official interior, both flanks","interior_visible":"Visible target patch, both flanks"}
    rows=[[label,*[f"{recon[k]['domains']['all'][method]['mae_sd']:.3f}" for k in MODELS[:3]]] for method,label in names.items()]
    table(pages,["Condition",*SHORT[:3]],rows)
    flank=read(OUT/"flank_normalization.json")
    pages+=["A follow-up decomposes level correlation in the official masked interior reconstruction. Both visible flanks contribute to the official normalization. All correlation rows use actual target level referenced to the preceding prefix, with the residual-only row subtracting the two-flank mean from the prediction. Intervals resample whole items within each of six diagnostic domains.",""]
    rows=[]
    for m,label in zip(MODELS[:3],SHORT[:3]):
        result=flank[m]
        cs=result["level_correlations"]
        rows.append([label,est(cs["native_official"]),est(cs["two_flank_mean"]),
                     est(cs["native_after_removing_flank_mean"]),est(cs["same_evidence_fixed_prefix_norm"]),
                     est(result["native_minus_flank_mean_mae_sd"])])
    table(pages,["Release","Native r","Mean-only r","Residual-only r","Fixed-prefix norm r","Native minus mean MAE/SD"],rows)
    pages+=["Small and base head Gram matrices lie within 7 × 10⁻⁷ of 1.41² I in every entry and all eight bias components are zero. The large model has learned-looking smooth adjacent-row products of 0.489–0.497 and nonzero bias. Geometry diagnoses the released tensor. It does not determine its complete training lineage.",""]
    (HERE/"RESULT_TABLES.md").write_text("\n".join(pages)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only",action="store_true")
    main(parser.parse_args().core_only)
