"""Give the shuffled directions exactly the target-dose supervision given to the true map."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.activation_transfer.forecast_requests.run import (
    ALIGN, CAL, CUES, DATA_ROOT, DEPTHS, HERE, MODELS, N_ALIGN, TEST, adapter_for, data,
    desired, learn_directions, live_effect, scale_factors,
)
from research.activation_transfer.forecast_requests.summary import item_scores, stats
from tsfm_lens.steering import SteeringModel
from tsfm_lens.steering.model import InterventionSite


def replay(target, cache, out, batch):
    ids, x = data(DATA_ROOT)
    fits = {m:np.load(cache / (m+"_fit.npz")) for m in MODELS}
    meta = json.loads(str(fits[target]["metadata"]))
    for archive in fits.values():
        np.testing.assert_array_equal(archive["ids"], ids)
    model = SteeringModel.from_adapter(adapter_for(target), quantiles=[.5])
    rng = np.random.default_rng(202611207)
    perms = [rng.permutation(N_ALIGN) for _ in range(4)]
    result = dict(ids=ids[TEST])
    for i in range(len(DEPTHS)):
        site = InterventionSite(**meta["site"][i])
        ts = fits[target]["s"+str(i)]
        norm = np.linalg.norm(ts, axis=1)
        for source, archive in fits.items():
            if source == target:
                continue
            _, shams = learn_directions(archive["s"+str(i)][ALIGN], ts[ALIGN], archive["d"+str(i)], perms)
            step = live_effect(model, site, x[CAL], norm[CAL], shams.reshape(-1, shams.shape[-1]), batch)
            step = step.reshape(len(x[CAL]), len(CUES), len(perms), -1)
            scales = np.stack([scale_factors(step[:,:,j], desired()) for j in range(len(perms))], axis=1)
            actions = shams * scales[:,:,None]
            effect = live_effect(model, site, x[TEST], norm[TEST], actions.reshape(-1, actions.shape[-1]), batch)
            key = f"{i}_{source}"
            result[key] = effect.reshape(len(x[TEST]), len(CUES), len(perms), -1)
            result[key+"_scales"] = scales
            print(target, key, "shams replayed", flush=True)
    np.savez_compressed(out / ("dose_shams_"+target+".npz"), **result)


def summarize(out):
    full, rows = {}, []
    for domain in ["daily", "weekly"]:
        items = []
        for target in MODELS:
            with np.load(out/(target+".npz")) as original, np.load(out/("dose_shams_"+target+".npz")) as sham:
                ids = original["ids"]
                np.testing.assert_array_equal(ids, sham["ids"])
                use = np.char.startswith(ids, domain+"/")
                n = use.sum()
                draws = np.random.default_rng(202611208).integers(n, size=(4000,n))
                for source in MODELS:
                    if source==target:
                        continue
                    for i, depth in enumerate(DEPTHS):
                        key = f"{i}_{source}"
                        real, controls = original[key][use], sham[key][use]
                        for c, name in enumerate(CUES):
                            wanted = desired()[c]
                            true_skill = item_scores(real[:,c,1], wanted)["skill"]
                            shuffled = np.mean([item_scores(controls[:,c,j], wanted)["skill"]
                                                for j in range(controls.shape[2])], axis=0)
                            gap = true_skill - shuffled
                            rows.append(dict(domain=domain, source=source, target=target, depth=depth, cue=name,
                                             real=stats(true_skill, draws), sham=stats(shuffled, draws),
                                             difference=stats(gap, draws)))
                            items.append((name, true_skill, shuffled, gap))
        groups = {}
        for c in [*CUES, "all"]:
            arrays = [values[1:] for values in items if c=="all" or values[0]==c]
            means = np.stack(arrays).mean(axis=0)
            groups[c] = {k:stats(a, draws) for k,a in zip(["real", "sham", "difference"], means)}
        full[domain] = groups
    (out/"dose_shams.json").write_text(json.dumps(dict(groups=full, rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=[*MODELS, "summary"])
    p.add_argument("--cache", type=Path, default=Path("/tmp/tsfm_forecast_requests_20261120"))
    p.add_argument("--out", type=Path, default=HERE/"results")
    p.add_argument("--batch", type=int, default=32)
    args=p.parse_args()
    torch.set_num_threads(3)
    if args.stage=="summary":
        summarize(args.out)
    else:
        replay(args.stage, args.cache, args.out, args.batch)
