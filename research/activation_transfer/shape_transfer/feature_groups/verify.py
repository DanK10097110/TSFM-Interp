"""Rebuild two independent scalar decisions per fresh-test arm and physically replay them."""
import json
import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.shape_transfer.feature_groups.data import OUT, dictionaries, load
from tsfm_lens.extraction.hooks import token_patch


def verify(source):
    dic = dictionaries(source)
    data = load(source,dic,"fresh")
    model = adapter_for(dic.target)
    layer = MODELS[dic.target][-1]
    sel = last_window_slice(model,data["positions"])
    samples = 0
    maximum = 0.
    rng = np.random.default_rng(2026112711)
    for tag in (("", "_global", "_unit") if source == "bolt" else ("", "_global")):
        list_tag = "_global" if tag == "_global" else ""
        weight = 1. if tag == "_unit" else dic.weight
        fit = json.loads((OUT/(source+list_tag+"_lists.json")).read_text())
        assert set(fit["calibration_ids"]).isdisjoint(data["ids"])
        with np.load(OUT/(source+tag+"_fresh.npz")) as f:
            groups = dict(fit["groups"])
            if not tag:
                groups["local"] = None
            elif tag == "_unit":
                groups = {"learned":groups["learned"],"local":None}
            for name, lists in groups.items():
                actual = f[name+"_effect"]
                ids = f[name+"_id"]
                assert np.all(actual[ids < 0] == 0.)
                active = data["codes"] > 0
                for dom in ("daily","monash_traffic"):
                    indices = np.argwhere(active & (data["labels"] == dom)[:,None] & (ids >= 0))
                    if not len(indices):
                        continue
                    i,j = indices[rng.integers(len(indices))]
                    allowed = list(data["support"][i])
                    if lists is not None:
                        allowed = [a for a in allowed if a in lists[j]]
                    ideal = -data["codes"][i,j] * (data["sj"][i] @ dic.donor[j])
                    def weighted(v):
                        return v.mean() + weight * (v - v.mean())
                    best, winner, dose = float(np.sum(weighted(ideal)**2)), -1, 0.
                    for atom in allowed:
                        pred = data["tj"][i] @ dic.decoder[atom]
                        q,t = weighted(pred),weighted(ideal)
                        bound = data["strengths"][i,atom]
                        c = min(bound,max(-bound, np.dot(q,t)/max(np.dot(q,q),1e-24)))
                        loss = float(np.sum((q*c-t)**2))
                        if loss < best:
                            best,winner,dose = loss,atom,c
                    assert winner == ids[i,j]
                    move = dose * dic.decoder[winner]
                    move *= min(1.,.2*dic.radius/np.linalg.norm(move))
                    x = data["x"][i:i+1]
                    ordinary = model.predict(x,8,[.5])["quantiles"][0,:8,0]
                    replacement = torch.as_tensor(data["tokens"][i:i+1]+move[None,None].astype("f4"))
                    with token_patch(model.module,layer,sel,replacement):
                        perturbed = model.predict(x,8,[.5])["quantiles"][0,:8,0]
                    error = float(np.max(np.abs(perturbed-ordinary-actual[i,j])))
                    maximum = max(maximum,error)
                    np.testing.assert_allclose(perturbed-ordinary,actual[i,j],atol=8e-5,rtol=5e-4)
                    samples += 1
    return {"independent_selection_and_physical_replay_cases":samples,"maximum_absolute_error":maximum}


if __name__ == "__main__":
    torch.set_num_threads(2)
    result = {s:verify(s) for s in ("bolt","timesfm")}
    (OUT/"replay.json").write_text(json.dumps(result,indent=2)+"\n")
