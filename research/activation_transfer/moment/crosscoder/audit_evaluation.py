"""Recompute published validation gates and external reads via direct linear solves."""
import json

import numpy as np
import torch
from scipy.linalg import solve

from research.activation_transfer.moment.crosscoder.evaluate import (
    ORACLE_ARMS, OUT, SOURCE_ARMS, raw_data, representation, write_json,
)


def main():
    torch.set_num_threads(1);d=raw_data();out={}
    for arm in (*SOURCE_ARMS,*ORACLE_ARMS):
        x,test,blind=representation(arm,d)
        x=x.astype(float);xmean=x[:768].mean(0);ymean=d["y"][:768].mean(0)
        fit=x[:768]-xmean;gram=fit@fit.T
        f=json.loads((OUT/(arm+"_fit.json")).read_text())
        worst=0.;gate_gap=0.;report={}
        with np.load(OUT/(arm+"_doses.npz")) as archive:
            for j,relative in enumerate(f["lambda_relative"]):
                # Cholesky on the regularized Gram; no eigensystem or threshold routine from the fitted readout.
                alpha=relative*f["lambda_scale"]
                dual=solve(gram+np.eye(len(gram))*alpha,d["y"][:768,j]-ymean[j],assume_a="pos")
                coef=fit.T@dual
                val=(x[768:]-xmean)@coef+ymean[j];truth=d["y"][768:,j]
                pos=np.unique(val[val>0]);thresholds=np.r_[0.,(pos[1:]+pos[:-1])/2,np.inf]
                vv=np.where(val[:,None]>thresholds,val[:,None],0.)
                losses=((vv-truth[:,None])**2).sum(0)
                gate_gap=max(gate_gap,abs(losses.min()-f["validation_sse"][j]))
                cut=f["thresholds"][j];cut=np.inf if cut is None else cut
                for name,data in (("",test),("blind_",blind),("validation_",x[768:])):
                    pred=(data.astype(float)-xmean)@coef+ymean[j]
                    gated=np.where(pred>cut,pred,0.)
                    worst=max(worst,float(np.max(abs(pred-archive[name+"raw"][:,j]))),
                              float(np.max(abs(gated-archive[name+"dose"][:,j]))))
                    np.testing.assert_array_equal(gated>0,archive[name+"dose"][:,j]>0)
        assert worst<1e-6 and gate_gap<1e-6
        report.update(max_abs_prediction_error=worst,max_abs_validation_sse_disagreement=float(gate_gap))
        out[arm]=report
    write_json(OUT/"evaluation_audit.json",out)
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
