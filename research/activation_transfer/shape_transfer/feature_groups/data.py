"""Load frozen evaluation cohorts and calibration geometry without using ablations for selection."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.atoms.materials import MODELS, dictionary_path
from research.activation_transfer.replication.data import CACHE as OLD, HERE as REPLICATION
from research.activation_transfer.replication.experiment import read_materials
from research.activation_transfer.shape_transfer.fresh_cohort.data import CACHE as FRESH, OUT as FRESH_OUT
from research.activation_transfer.shape_transfer.full_space.jacobians import HERE as FULL, SCRATCH as HIST
from research.activation_transfer.shape_transfer.sparse_stability.prepare import OUT as STABILITY, SCRATCH as NOISE
from research.activation_transfer.shape_transfer.sparse_stability.run import indices, active_codes
from tsfm_lens.sae.train import load_sae_checkpoint

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


@dataclass
class Dictionary:
    target: str
    atoms: np.ndarray
    donor: np.ndarray
    decoder: np.ndarray
    sae: object
    weight: float
    radius: float


def dictionaries(source):
    target = next(m for m in MODELS if m != source)
    a = load_sae_checkpoint(str(dictionary_path(source)))
    b = load_sae_checkpoint(str(dictionary_path(target)))
    with np.load(FULL / "results" / (source + ".npz")) as f:
        atoms, weight = f["atoms"], float(f["shape_weight"])
    with np.load(HIST / (target + "_jac.npz")) as f:
        radius = float(f["radius"])
    return Dictionary(target, atoms, a.W_dec.detach().numpy()[atoms].astype(float),
                      b.W_dec.detach().numpy().astype(float), b, weight, radius)


def load(source, dic, cohort):
    if cohort in ("cal", "stability_clean"):
        raw, native_dir, geo, suffix = OLD, REPLICATION / "results", HIST, "_jac.npz"
        with np.load(native_dir / (source + "_seed0.npz")) as f:
            if cohort == "cal":
                take = np.flatnonzero(f["split"] == "cal")
            else:
                keys = json.loads((STABILITY / "manifest.json").read_text())["ids"]
                take = indices(f["ids"], keys)
            ids, codes = f["ids"][take], f["source_codes"][take]
            ref = None if cohort == "cal" else f["native"][take]
            labels = np.repeat(cohort, len(ids))
            np.testing.assert_array_equal(f["atoms"], dic.atoms)
    else:
        raw, native_dir = (FRESH, FRESH_OUT) if cohort == "fresh" else (NOISE, NOISE)
        geo, suffix = raw, "_geometry.npz"
        with np.load(native_dir / (source + "_native.npz")) as f:
            ids, codes, ref, labels = f["ids"], f["codes"], f["native"], f["split"].astype(str)
            np.testing.assert_array_equal(f["atoms"], dic.atoms)
    x = None
    if cohort != "cal":
        with np.load(raw / "data.npz") as f:
            x = f["x"][indices(f["ids"], ids)]
    with np.load(geo / (source + suffix)) as s, np.load(geo / (dic.target + suffix)) as t:
        si, ti = indices(s["ids"], ids), indices(t["ids"], ids)
        sj, tj = s["jac"][si].astype(float), t["jac"][ti].astype(float)
        ordinary = t["forecast"][ti]
    all_tokens, pos, all_ids = read_materials(raw, dic.target)
    tokens = all_tokens[indices(all_ids, ids)]
    support, z = active_codes(dic.sae, tokens)
    assert len(np.unique(ids)) == len(ids)
    return dict(ids=ids, codes=codes, ref=ref, labels=labels, support=support, strengths=z,
                sj=sj, tj=tj, tokens=tokens, positions=pos, ordinary=ordinary, x=x)
