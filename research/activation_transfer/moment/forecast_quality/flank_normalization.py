"""Explain interior-imputation level agreement with the two observed flank means and released decoder."""
import json
import numpy as np

from research.activation_transfer.moment.forecast_quality.data import OUT
from research.activation_transfer.moment.forecast_quality.score import percentile


def correlations(a, b, draws):
    x, y = a[draws], b[draws]
    x=x-x.mean(1,keepdims=True)
    y=y-y.mean(1,keepdims=True)
    return (x*y).sum(1) / np.sqrt(np.square(x).sum(1)*np.square(y).sum(1))


def main():
    with np.load(OUT/"panel.npz") as f:
        index={item:i for i,item in enumerate(f["id"])}
        histories=f["context"][:,1]
    report={}
    for m in ("moment_small","moment_base","moment_large"):
        with np.load(OUT/(m+"_reconstruction.npz")) as q:
            ids=q["ids"]
            d=q["domain"]
            x=histories[[index[i] for i in ids]]
            prefix=x[:,:416]
            y=x[:,416:424]
            np.testing.assert_array_equal(y,q["target"])
            scale=prefix.std(1)
            initial=prefix.mean(1)
            flanks=np.concatenate([prefix,x[:,424:]],axis=1).mean(1)
            native=q["native_interior"]
            manual=q["interior_true_right"]
            visible=q["interior_visible"]
        # All six domains carry 12 cases; resample whole items within each.
        rng=np.random.default_rng(2026112287)
        groups=[np.flatnonzero(d==k) for k in np.unique(d)]
        draws=np.vstack([np.arange(len(d)),
                         np.concatenate([rng.choice(g,size=(4000,len(g)),replace=True) for g in groups],axis=1)])
        truth=(y.mean(1)-initial)/scale
        norm=(flanks-initial)/scale
        full=(native.mean(1)-initial)/scale
        decoder=(native.mean(1)-flanks)/scale
        np.testing.assert_allclose(full,norm+decoder,rtol=1e-12,atol=1e-12)
        fixed=(manual.mean(1)-initial)/scale
        scores={}
        for k,v in dict(native_official=full, two_flank_mean=norm, native_after_removing_flank_mean=decoder,
                        same_evidence_fixed_prefix_norm=fixed).items():
            scores[k]=percentile(correlations(v,truth,draws))
        scores["native_residual_vs_true_residual_from_flank_mean"]=percentile(correlations(decoder,truth-norm,draws))
        maen=(np.abs(native-y)/scale[:,None]).mean(1)
        maef=(np.abs(flanks[:,None]-y)/scale[:,None]).mean(1)
        visible_delta=((np.abs(visible-y)-np.abs(manual-y))/scale[:,None]).mean(1)
        record=dict(level_correlations=scores,
                    visible_minus_masked_same_context_mae_sd=percentile(visible_delta[draws].mean(1)),
                    visible_lower_mae_frequency=float((visible_delta<0).mean()),
                    native_mae_sd=percentile(maen[draws].mean(1)),flank_mean_mae_sd=percentile(maef[draws].mean(1)),
                    native_minus_flank_mean_mae_sd=percentile((maen-maef)[draws].mean(1)))
        report[m]=record
    (OUT/"flank_normalization.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
