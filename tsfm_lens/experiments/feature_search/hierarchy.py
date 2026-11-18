"""Small frozen-decoder hierarchy control, inspired by arxiv:2606.16193.

Level two consumes decoder ATOMS [features, hidden], never sample codes.
This is a post-hoc ablation, not the paper's joint CSAE training procedure.
Prediction and response measurements must use separate held-out series.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def unit_atoms(decoder: np.ndarray) -> np.ndarray:
    """Validate and normalize decoder rows; refuse dead or nonfinite atoms."""
    atoms = np.asarray(decoder, dtype=np.float32)
    if atoms.ndim != 2 or min(atoms.shape) < 2 or not np.isfinite(atoms).all():
        raise ValueError("decoder must be a finite [features, hidden] matrix")
    norms = np.linalg.norm(atoms, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("remove zero decoder atoms before hierarchy fitting")
    return atoms / norms


def cosine_hierarchy(decoder: np.ndarray, n_parents: int) -> np.ndarray:
    """Cheap average-linkage cosine baseline, returning hard memberships."""
    atoms = unit_atoms(decoder)
    if not 1 < n_parents < len(atoms):
        raise ValueError("need 1 < n_parents < number of atoms")
    labels = AgglomerativeClustering(
        n_clusters=n_parents, metric="cosine", linkage="average"
    ).fit_predict(atoms)
    return np.eye(n_parents, dtype=np.float32)[labels]


def decoder_hierarchy(decoder: np.ndarray, n_parents: int, *, seed: int,
                      steps: int = 100, k: int = 1, diagnostics: dict | None = None) -> np.ndarray:
    """Fit a CPU TopK SAE to decoder rows and return atom-parent weights.

    Zero-response atoms remain unassigned and must be reported; argmax would
    otherwise silently put every such atom into parent zero. Callers control
    thread limits and must filter level-one train-inactive features first.
    """
    import torch
    from tsfm_lens.sae.models import TopKSAE

    atoms = unit_atoms(decoder)
    if not 1 < n_parents < len(atoms) or not 1 <= k <= n_parents or steps < 1:
        raise ValueError("invalid parent count, sparsity or training steps")
    model = TopKSAE(atoms.shape[1], n_parents, k,
                    generator=torch.Generator().manual_seed(seed))
    x = torch.from_numpy(atoms)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        reconstruction, _ = model(x)
        loss = (reconstruction - x).square().mean()
        losses.append(float(loss.detach()))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite decoder hierarchy loss")
        loss.backward()
        optimizer.step()
        model.normalize_decoder_()
    with torch.no_grad():
        weights = model.encode(x).numpy()
    if diagnostics is not None:
        diagnostics.update(initial_loss=losses[0], final_loss=losses[-1],
                           tail_losses=losses[-5:], steps=steps)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)


def parent_responses(codes: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Average child response per parent with identical pooling in both arms."""
    codes = np.asarray(codes, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.float64)
    if (codes.ndim != 2 or membership.ndim != 2
            or codes.shape[1] != membership.shape[0]
            or not np.isfinite(codes).all() or not np.isfinite(membership).all()
            or np.any(membership < 0)):
        raise ValueError("need finite compatible [series, atoms], [atoms, parents]")
    mass = membership.sum(axis=0)
    return (codes @ membership) / np.maximum(mass, 1e-12)


def random_hierarchy(membership: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle atom memberships while preserving parent mass and sparsity."""
    membership = np.asarray(membership)
    if membership.ndim != 2 or not np.isfinite(membership).all():
        raise ValueError("membership must be a finite matrix")
    return membership[np.random.default_rng(seed).permutation(len(membership))]


def predict_labels(train_responses: np.ndarray, train_labels: np.ndarray,
                   test_responses: np.ndarray) -> np.ndarray:
    """Fixed C=1 logistic probe, scaling fitted on training series only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
    probe.fit(train_responses, train_labels)
    return probe.predict(test_responses)


def promotion_gate(coarse_gains: list[float], fine_gains: list[float],
                   seed_ari: list[float]) -> bool:
    """Pre-fixed exploratory gate: +2pt coarse, at most 2pt fine loss, ARI .8."""
    values = [np.asarray(x, dtype=float) for x in (coarse_gains, fine_gains, seed_ari)]
    if any(x.size < 3 or not np.isfinite(x).all() for x in values):
        return False
    return bool(values[0].mean() >= .02 and values[1].min() >= -.02
                and values[2].min() >= .8)
