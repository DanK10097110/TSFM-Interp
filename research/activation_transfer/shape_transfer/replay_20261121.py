"""Replay selected whole studies from copied source files with separately written outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PACKAGE="research.activation_transfer.shape_transfer."
TASKS={
    "pair_depth":("depth.paired_sites",["bolt"],"depth",["bolt_paired.npz"]),
    "self_timer":("self_transfer.run",["timer"],"self_transfer",[s+"_to_timer.npz" for s in ("bolt","timesfm","timer")]),
    "inner_dist":("quantiles.interior",["bolt"],"quantiles",["bolt_interior.npz"]),
    "distribution":("quantiles.normal_score",["run","timesfm","--batch","24"],"quantiles",["timesfm_normal.npz"]),
    "rotated":("response_subspace.run",["timesfm","--seed","1"],"response_subspace",["timesfm_seed1.npz"]),
    "factorial":("interactions.run",["timesfm","native"],"interactions",["timesfm_native.npz"]),
    "feedback":("depth.feedback.run",["10"],"depth/feedback",["block10.npz"]),
}
SUPPLEMENT={
    "conditional":("interactions.conditional",["blind","--batch","24"],"interactions",["blind_conditional.npz"]),
    "lead_placement":("lead_budget.run",["run","timesfm","--batch","24"],"lead_budget",["timesfm.npz"]),
    "last_block":("depth.feedback.last_block",[],"depth/feedback",["last_block.npz"]),
}
FINAL_TASKS={
    "gate_confirmation":("dictionary_prior.refit.confirmation",["run","bolt","--seed","1"],"dictionary_prior/refit",["confirm_bolt_1.npz"]),
    "gate_norms":("dictionary_prior.refit.encoding",["timesfm","2","--compare-norms"],"dictionary_prior/refit",["normgate_timesfm_2.npz"]),
    "future_outcomes":("forecast_loss.run",[],"forecast_loss",["blind.npz","extra.npz"]),
}
SPAN_TASKS={
    "decoder_span":("dictionary_prior.receiver_encoding.span_inverse",["timesfm"],
                    "dictionary_prior/receiver_encoding",["timesfm_span_inverse.npz"]),
    "span_control":("dictionary_prior.receiver_encoding.span_controls",["2301"],
                    "dictionary_prior/receiver_encoding",["span_control_2301.npz"]),
    "swap_support":("dictionary_prior.receiver_encoding.swap_support",[],
                    "dictionary_prior/receiver_encoding",["swap_support.npz"]),
}


def prepare(work):
    """Copy Python sources; make existing, immutable model/data inputs readable in the new tree."""
    ref=ROOT/"research"
    copy=work/"research"
    work.mkdir(parents=True,exist_ok=True)
    for original in ref.rglob("*.py"):
        if "__pycache__" in original.parts:
            continue
        dest=copy/original.relative_to(ref)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,dest)
    for folder in (ref/"activation_transfer").rglob("results"):
        dest=copy/folder.relative_to(ref)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(folder,target_is_directory=True)
    for name in ("tsfm_lens","tsfm_benchmark"):
        dst=work/name
        if not dst.exists():
            dst.symlink_to(ROOT/name,target_is_directory=True)
    sources=sorted(p for p in copy.rglob("*.py") if not p.is_symlink())
    digest=hashlib.sha256()
    for p in sources:
        digest.update(str(p.relative_to(copy)).encode("utf-8"));digest.update(p.read_bytes())
    (work/"source.json").write_text(json.dumps(dict(source_files=len(sources),sha256=digest.hexdigest()),indent=2)+"\n",encoding="utf-8")


def run(task,work):
    mod,argv,where,files=(TASKS | SUPPLEMENT | FINAL_TASKS | SPAN_TASKS)[task]
    out=work/"outputs"/task
    out.mkdir(exist_ok=True,parents=True)
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join(map(str,[work,ROOT/"tsfm_lens",ROOT/"tsfm_benchmark"]))
    subprocess.run([sys.executable,"-m",PACKAGE+mod,*argv,"--out",str(out)],cwd=work,env=env,check=True)
    detail={}
    for name in files:
        with np.load(HERE/where/"results"/name) as truth,np.load(out/name) as replay:
            if set(truth)!=set(replay):
                raise ValueError(f"Archive key mismatch for {name}")
            numeric,exact={},True
            for k in truth:
                a,b=truth[k],replay[k]
                if a.dtype.kind in "fc":
                    np.testing.assert_allclose(a,b,rtol=.004,atol=3e-4)
                    numeric[k]=float(np.max(np.abs(a-b),initial=0.))
                    exact &= np.array_equal(a,b)
                else:
                    np.testing.assert_array_equal(a,b)
            detail[name]=dict(identical_values=bool(exact),arrays=len(truth),max_numeric_discrepancy=max(numeric.values(),default=0.),
                              differences={k:v for k,v in numeric.items() if v>0})
    (work/(task+".json")).write_text(json.dumps(detail,indent=2)+"\n",encoding="utf-8")
    print(task,detail,flush=True)


def finish(work,stage):
    tasks={"finish":TASKS,"finish_followup":SUPPLEMENT,"finish_final":FINAL_TASKS,"finish_span":SPAN_TASKS}[stage]
    record={k:json.loads((work/(k+".json")).read_text(encoding="utf-8")) for k in tasks}
    record["copied_sources"]=json.loads((work/"source.json").read_text(encoding="utf-8"))
    name={"finish":"replay_20261121.json","finish_followup":"replay_20261121_followup.json",
          "finish_final":"replay_20261121_final.json","finish_span":"replay_20261121_span.json"}[stage]
    (HERE/name).write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("prepare","finish","finish_followup","finish_final","finish_span",*TASKS,*SUPPLEMENT,*FINAL_TASKS,*SPAN_TASKS))
    p.add_argument("--work",type=Path,default=Path("/tmp/tsfm_nov21_isolated"))
    a=p.parse_args()
    if a.task=="prepare":
        prepare(a.work)
    elif a.task.startswith("finish"):
        finish(a.work,a.task)
    else:
        run(a.task,a.work)
