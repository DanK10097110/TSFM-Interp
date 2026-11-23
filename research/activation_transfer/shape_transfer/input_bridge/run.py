"""Apply one signed and dose-limited public history edit to both frozen models."""
import argparse
import gc

import numpy as np
import torch

from research.activation_transfer.atoms.materials import MODELS, adapter_for, dictionary_path
from research.activation_transfer.shape_transfer.blind_frequency.data import HERE as BLIND_RESULTS
from research.activation_transfer.shape_transfer.blind_frequency.data import CACHE as BLIND
from research.activation_transfer.shape_transfer.full_space.transfer import boost_shape
from research.activation_transfer.shape_transfer.input_bridge.probe import CAP, HERE, basis, edit, forecast, probe_path
from tsfm_lens.sae.train import load_sae_checkpoint

ARMS = ("donor", "scrambled", "recipient", "joint")


def minimum(design, goal):
    """Ridge inverse of one local input derivative per history, many source atom goals."""
    u, s, vh = np.linalg.svd(design, full_matrices=False)
    ridge = .001 * s[:, :1]**2
    factors = np.divide(s, s*s+ridge, out=np.zeros_like(s), where=s*s+ridge > 0)
    return ((vh.swapaxes(-1, -2) * factors[:, None, :]) @ (u.swapaxes(-1, -2) @ goal)).swapaxes(-1, -2)


def bounded(weights):
    norms = np.linalg.norm(weights, axis=-1)
    scale = np.minimum(1., CAP / np.maximum(norms, np.finfo(float).tiny))
    return weights * scale[..., None], norms


def scramble(coeff):
    """One norm-preserving wrong association between coefficients and DCT modes."""
    sign = np.where(np.arange(coeff.shape[-1])%3 == 0, 1., -1.)
    return np.roll(coeff, 5, axis=-1) * sign


def read(source):
    target = next(k for k in MODELS if k != source)
    with np.load(BLIND/"data.npz") as f:
        ids, x, labels = f["ids"], f["x"], f["split"]
    with np.load(BLIND_RESULTS/"results"/(source+"_fresh.npz")) as f:
        np.testing.assert_array_equal(ids, f["ids"])
        np.testing.assert_array_equal(labels, f["split"])
        z, atoms, weight = f["codes"], f["atoms"], float(f["weight"])
        native, higher_access = f["source"], f["minimum"]
        np.testing.assert_array_equal(z>0, f["active"])
    sae = load_sae_checkpoint(str(dictionary_path(source)))
    decoder = sae.W_dec.detach().numpy()[atoms].astype(float).T
    with np.load(BLIND/(source+"_early.npz")) as f:
        np.testing.assert_array_equal(ids, f["ids"])
        goal = (-z[:, None, :]) * (f["jac"].astype(float) @ decoder)
    probes = []
    for model in (source, target):
        with np.load(probe_path(model)) as f:
            np.testing.assert_array_equal(ids, f["ids"])
            np.testing.assert_allclose(basis(), f["axes"], atol=0, rtol=0)
            probes.append(f["jac"])
    return target, x, ids, labels, native, higher_access, z, atoms, weight, goal, probes


def run(source):
    target, x, ids, labels, native, ref, z, atoms, weight, goal, probes = read(source)
    mask = z>0
    rows, cols = np.nonzero(mask)
    designs = [boost_shape(v, weight) for v in probes]
    weighted = boost_shape(goal, weight)
    donor = minimum(designs[0], weighted)
    coefs = dict(donor=donor, scrambled=scramble(donor),
                 recipient=minimum(designs[1], weighted),
                 joint=minimum(np.concatenate(designs, axis=1), np.tile(weighted, (1, 2, 1))))
    record = dict(ids=ids, split=labels, atoms=atoms, codes=z, active=mask, native=native,
                  hidden_recipient=ref, weight=weight, requested=goal.swapaxes(1,2).astype("f4"))
    directions = {}
    for method, value in coefs.items():
        directions[method], raw_norm = bounded(value)
        record[method+"_coeff"] = directions[method].astype("f4")
        record[method+"_raw_norm"] = raw_norm.astype("f4")
        if np.any(directions[method][~mask] != 0):
            raise ValueError("Nonzero write for an inactive source atom")
    np.testing.assert_allclose(record["donor_raw_norm"], record["scrambled_raw_norm"], atol=0, rtol=0)
    nlead = native.shape[-1]
    for label, model in (("s", source), ("t", target)):
        adapter = adapter_for(model)
        zero = forecast(adapter, x[rows])
        with np.load(probe_path(model)) as f:
            record[label+"_max_zero_rebatch"] = np.max(np.abs(zero-f["base"][rows]))
        for method in ARMS:
            coeff = directions[method][rows, cols]
            y = forecast(adapter, edit(x[rows], coeff, basis()))
            change = np.zeros((*mask.shape, nlead), dtype="f4")
            change[rows, cols] = y-zero
            record[label+"_"+method] = change
            # Early derivative response of the already bounded physical write.
            record[label+"_linear_"+method] = np.einsum("nhk,nak->nah", probes[0 if label=="s" else 1], directions[method]).astype("f4")
            print(source, model, method, len(y), flush=True)
        del adapter
        gc.collect()
    np.savez_compressed(HERE/"results"/(source+".npz"), **record)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=MODELS)
    torch.set_num_threads(1)
    run(p.parse_args().source)
