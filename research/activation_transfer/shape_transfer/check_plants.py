"""Run the unmodified tests and confirm each fails after a specific code mutation in isolation."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASES = [
    ("dictionary_prior/receiver_encoding","capacity","test_capacity","s*coord/(s*s+lam)","s*coord/(s*s+lam*s)"),
    ("dictionary_prior/receiver_encoding","span_inverse","test_span_inverse","goal=j@v[:,:,None]","goal=-j@v[:,:,None]"),
    ("forecast_loss","run","test_loss_change","(2*error+delta)*delta","(2*error-delta)*delta"),
    ("dictionary_prior/refit","run","test_refit","coef@directions","single"),
    ("dictionary_prior/refit","encoding","test_refit_encoding","(edited-clean)@sae.W_dec/fraction","(locked-clean)@sae.W_dec/fraction"),
    ("lead_budget","run","test_lead_placement","np.setdiff1d(np.arange(16),AUDIT)","np.arange(16)"),
    ("depth/feedback","last_block","test_last_block","dx - clean * radial","dx + clean * radial"),
    ("interactions","conditional","test_interior_square","4.*(high_high-high_low-low_high+low_low)","2.*(high_high-high_low-low_high+low_low)"),
    ("interactions","run","test_interactions","for v in (first,second,first+second)","for v in (first+second,)"),
    ("depth/feedback","run","test_feedback","level+weight*(diff-level)","level+(diff-level)"),
    ("quantiles","five_interp","test_five_anchors","med+NORMAL[4]*hi+right","med+NORMAL[4]*hi-right"),
    ("self_transfer","run","test_self_transfer","return proj, rest","return proj, 0*rest"),
    ("quantiles","score_interior","test_interior","deviation - FRACTIONS*tails","deviation + FRACTIONS*tails"),
    ("depth","paired_sites","test_paired_sites","+= torch.as_tensor(delta","= torch.as_tensor(delta"),
    ("depth","geometry","test_depth_capture","new[:,interval,:]=part+d[:,None,:]","new[:,interval,:]=part+2*d[:,None,:]"),
    ("quantiles","geometry","test_quantile_channels","(high-low)/2","(low-high)/2"),
    ("response_subspace","run","test_response_subspace","return parallel+(coordinates[:,perm]*signs)@complement.T",
     "return parallel+(coordinates[:,perm]*signs)@complement.T*0.5"),
    ("input_paths","run","test_input_paths","edited[:, -1] = replacement[:, -1]","edited[:, 0] = replacement[:, 0]"),
    ("full_space","transfer","test_full_space","transpose @ np.linalg.solve","-transpose @ np.linalg.solve"),
    ("full_space","leads","test_lead_holdout","calibration = source_calibration[:,leads]","calibration = source_calibration"),
    ("token_position","run","test_token_position","return live_data[-count:]","return np.arange(len(spans))[-count:]"),
    ("sparse_writes","run","test_sparse_writes","change = np.clip(inner/np.maximum(norm,1e-24),-strengths,strengths)",
     "change = inner/np.maximum(norm,1e-24)"),
]


def check():
    records = {}
    for folder, module, test_name, original, broken in CASES:
        with TemporaryDirectory(prefix="tsfm_shape_plant_") as scratch:
            work = Path(scratch)
            leaf = work / HERE.relative_to(ROOT) / folder
            leaf.mkdir(parents=True)
            source = HERE / folder / (module+".py")
            code = source.read_text(encoding="utf-8")
            if code.count(original) != 1:
                raise ValueError(f"Mutation site has changed in {source}")
            test = HERE / folder / (test_name+".py")
            (leaf/test.name).write_bytes(test.read_bytes())
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(map(str, [work,ROOT,ROOT/"tsfm_lens",ROOT/"tsfm_benchmark"]))
            for label,body,status in [("baseline",code,"1 passed"),("plant",code.replace(original,broken),"1 failed")]:
                (leaf/source.name).write_text(body,encoding="utf-8")
                result = subprocess.run([sys.executable,"-m","pytest","-q",str(leaf/test.name)],
                                        cwd=work,env=env,capture_output=True,text=True)
                if not re.search(r"\b"+status+r"\b",result.stdout) or result.returncode != int(label=="plant"):
                    raise ValueError(result.stdout + result.stderr)
            records[test_name] = dict(test=test.name,original=original,plant=broken,baseline="1 passed",mutant="1 failed")
    (HERE/"plant_results.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    check()
