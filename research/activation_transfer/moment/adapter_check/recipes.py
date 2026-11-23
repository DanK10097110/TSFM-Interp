"""Retain the small training-recipe diagnostics from ordinary pipeline outputs."""
import argparse
import json
from pathlib import Path
import yaml


def summarize(run):
    cfg=yaml.safe_load((run/"config_resolved.yaml").read_text())
    target, result=next(iter(json.loads((run/"sae/meta.json").read_text()).items()))
    return dict(target=target,series=cfg["data"]["max_series"],window=cfg["alignment"]["window"],
                dictionary=result["dict_size"],top_k=result["k"],train_series=result["holdout_split"]["n_train"],
                heldout_series=result["holdout_split"]["n_heldout"],training_windows=result["n_train_rows"],
                epochs=cfg["sae"]["epochs"],batch=cfg["sae"]["batch_size"],aux_k=cfg["sae"]["aux_k"],
                resample_dead_every_epochs=cfg["sae"]["resample_dead_every_epochs"],
                fidelity_train=result["reconstruction_fidelity"],fidelity_heldout=result["reconstruction_fidelity_heldout"],
                dead_fraction_train=result["dead_feature_rate"],dead_fraction_heldout=result["dead_feature_rate_heldout"],
                heldout_token_preservation=result["forecast_preservation_token_heldout"],
                admission=result["admission"]["passed"],reason=result["admission"]["reason"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runs",type=Path,default=Path("/tmp/tsfm_moment_native/cli"))
    args=p.parse_args()
    names=("core_small","core_large","core_v2_small","core_v2_large","core_token_small","core_token_large","core_token_early_large")
    out={name:summarize(args.runs/name) for name in names}
    Path(__file__).with_name("workflow_recipes.json").write_text(json.dumps(out,indent=2)+"\n")
