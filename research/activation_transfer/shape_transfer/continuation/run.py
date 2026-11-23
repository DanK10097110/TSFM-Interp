"""Test priors for the unconstrained continuation of a natural source SAE removal."""
import argparse
from hashlib import file_digest

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.atoms.vectors import bridge
from research.activation_transfer.boundary.patch_forecast import last_window_slice
from research.activation_transfer.replication.data import CACHE, HERE as REPLICATION
from research.activation_transfer.replication.experiment import derangements, read_materials
from research.activation_transfer.shape_transfer.continuation.probe import HERE, ORIGINAL, SCRATCH
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL
from research.activation_transfer.shape_transfer.full_space.leads import forecast16
from research.activation_transfer.shape_transfer.full_space.transfer import apply, boost_shape, solve_adaptive, solve_fixed
from tsfm_lens.sae.train import load_sae_checkpoint


def corrected(design, response, prior):
    """Solve early weighted residuals around the supplied physical decoder displacement."""
    return prior + solve_adaptive(design, response - design @ prior)


def wrong_prior(prior, perm):
    if np.any(np.asarray(perm) == np.arange(prior.shape[1])):
        raise ValueError("Wrong-atom controls must be derangements")
    norms = np.linalg.norm(prior, axis=0)
    return prior[:,perm] * (norms/norms[perm])


def run(source, batch=32):
    target = next(n for n in MODELS if n!=source)
    with np.load(CACHE/"data.npz") as data:
        ids, labels = data["ids"], data["split"]
        ev = np.isin(labels,["daily","machines"])
        x = data["x"][ev]
    train, nontrain = labels=="train", labels!="train"
    use = ev[nontrain]
    cal = labels[nontrain]=="cal"
    assert (train.sum(),cal.sum(),ev.sum()) == (1024,96,320)
    with np.load(REPLICATION/"results"/(source+"_seed0.npz")) as archive:
        np.testing.assert_array_equal(ids[nontrain],archive["ids"])
        codes, atoms = archive["source_codes"][use], archive["atoms"]
        orig_clean = archive["dense"][use]
        sha = str(archive["source_checkpoint_hash"])
    ckpt = dictionary_path(source)
    with ckpt.open("rb") as handle:
        assert sha == file_digest(handle,"sha256").hexdigest()
    sae = load_sae_checkpoint(str(ckpt))
    decoder = sae.W_dec.detach().numpy()[atoms].astype(float).T
    hs, _, si = read_materials(CACHE,source)
    ht, positions, ti = read_materials(CACHE,target)
    np.testing.assert_array_equal(ids,si)
    np.testing.assert_array_equal(ids,ti)
    with torch.inference_mode():
        actual = sae.encode(torch.from_numpy(hs[ev].mean(axis=1))).numpy()[:,atoms]
    np.testing.assert_allclose(codes,actual,atol=2e-4,rtol=1e-4)
    clean = bridge(hs[train].mean(axis=1),ht[train].mean(axis=1),decoder.T,2026112018)[0].T
    with np.load(ORIGINAL/(source+"_jac.npz")) as s, np.load(ORIGINAL/(target+"_jac.npz")) as t:
        np.testing.assert_array_equal(ids[nontrain],s["ids"])
        np.testing.assert_array_equal(ids[nontrain],t["ids"])
        js8, jt8 = s["jac"].astype(float),t["jac"][use].astype(float)
    mean = js8[cal].mean(axis=1,keepdims=True)
    weight = np.sqrt(np.mean(mean**2)/np.mean((js8[cal]-mean)**2))
    with np.load(FULL/"results"/(source+".npz")) as old:
        np.testing.assert_allclose(old["shape_weight"],weight)
    early_design = boost_shape(jt8,weight)
    early_source = boost_shape(js8[use] @ decoder,weight)
    with np.load(SCRATCH/(source+"_jac16.npz")) as s, np.load(SCRATCH/(target+"_jac16.npz")) as t:
        for store in (s,t):
            np.testing.assert_array_equal(ids[nontrain],store["ids"])
            np.testing.assert_array_equal(labels[nontrain],store["split"])
        # The fixed prior consumes calibration histories only.
        js16, jt16 = s["jac"].astype(float),t["jac"].astype(float)
        paired = solve_fixed(boost_shape(jt16[cal],weight),boost_shape(js16[cal] @ decoder,weight))
        target_forecast = t["forecast"][use]
        radius = float(t["radius"])
    adapter, layer = adapter_for(target), MODELS[target][-1]
    selector = last_window_slice(adapter,positions)
    record = dict(ids=ids[ev],split=labels[ev],active=codes>0,atoms=atoms,
                  weight=weight,radius=radius,checkpoint=sha)
    with np.load(FULL/"results"/(source+"_leads.npz")) as old:
        np.testing.assert_array_equal(ids[ev],old["ids"])
        np.testing.assert_array_equal(codes>0,old["active"])
        np.testing.assert_array_equal(atoms,old["atoms"])
        reference_min, record["source"] = old["eight"], old["source"]

    def replay(key,write):
        effect, clipped = apply(adapter,layer,selector,x,ht[ev],target_forecast,codes,
                                write,radius,batch,forecast16)
        norms = np.linalg.norm(write,axis=-2)*codes/radius
        np.testing.assert_array_equal(clipped, norms>.2)
        record[key],record[key+"_norm"],record[key+"_clipped"] = effect,norms.astype("f4"),clipped
        print(source,key,int(clipped.sum()),flush=True)
        return effect

    minimum = replay("minimum",solve_adaptive(early_design,early_source))
    np.testing.assert_allclose(minimum,reference_min,atol=3e-4,rtol=.003)
    record["min_replay_error"] = np.max(np.abs(minimum-reference_min))
    prior = replay("clean_only",clean)
    mask = (codes>0) & ~record["clean_only_clipped"]
    np.testing.assert_allclose(prior[mask,:8],orig_clean[mask],atol=3e-4,rtol=.003)
    record["clean_prior_replay_error"] = np.max(np.abs(prior[mask,:8]-orig_clean[mask]))
    replay("clean_early",corrected(early_design,early_source,clean))
    for j,perm in enumerate(derangements(len(atoms))):
        other = wrong_prior(clean,perm)
        record["wrong_perm"+str(j)] = perm
        replay("wrong"+str(j),corrected(early_design,early_source,other))
    replay("paired_only",paired)
    replay("paired_early",corrected(early_design,early_source,paired))
    # This is the only arm that reads the held-out leads of evaluation derivatives.
    oracle = solve_adaptive(boost_shape(jt16[use],weight),boost_shape(js16[use] @ decoder,weight))
    replay("oracle",oracle)
    out = HERE/"results"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/(source+".npz"),**record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",choices=MODELS)
    a = parser.parse_args()
    torch.set_num_threads(1)
    run(a.source)
