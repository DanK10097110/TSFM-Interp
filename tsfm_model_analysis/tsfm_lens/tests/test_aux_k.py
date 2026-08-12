"""AuxK dead-atom revival tests (ROADMAP.md §6.2.1 Stage 0, hypothesis H4).

The mechanism under test is `sae/models.py::auxiliary_dead_loss` and its two
call sites (`sae/train.py::train_sae`, `sae/crosscoder.py::train_crosscoder`).
All synthetic with a planted answer -- these are regression tests for the
mechanism, not a substitute for the real-checkpoint Stage 0 sweep whose
numbers live in ROADMAP.md §6.2.1's Findings.

Two things are asserted, and the second matters as much as the first:
revival must *work*, and it must be a genuine no-op when `aux_k: 0`, since
every crosscoder/SAE number already on record was produced without it and
would silently change meaning otherwise (`CLAUDE.md` §11.24).

Runnable directly (`python tests/test_aux_k.py`) or via pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.crosscoder import CrosscoderTrainConfig, train_crosscoder
from tsfm_lens.sae.crosscoder import alive_mask as cross_alive_mask
from tsfm_lens.sae.eval import dead_feature_rate, reconstruction_fidelity
from tsfm_lens.sae.models import TopKSAE, auxiliary_dead_loss
from tsfm_lens.sae.train import SAETrainConfig, train_sae


def _planted_activations(n=3000, d=24, n_causes=12, seed=0):
    """[N, D] built from `n_causes` sparse random directions -- a dictionary
    genuinely exists, so a dead atom is wasted capacity rather than a correct
    response to there being nothing left to represent."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_causes, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coefs = rng.exponential(size=(n, n_causes)) * (rng.random((n, n_causes)) < 0.25)
    return (coefs @ directions + 0.01 * rng.normal(size=(n, d))).astype(np.float32)


def test_encode_is_unchanged_by_the_pre_activation_refactor():
    """`pre_activations` + `sparsify` must compose to exactly the old `encode`."""
    torch.manual_seed(0)
    sae = TopKSAE(d_in=8, dict_size=32, k=4)
    x = torch.randn(50, 8)
    expected = torch.relu((x - sae.b_dec) @ sae.W_enc + sae.b_enc)
    k = min(sae.k, expected.shape[-1])
    vals, idx = torch.topk(expected, k, dim=-1)
    reference = torch.zeros_like(expected)
    reference.scatter_(-1, idx, vals)
    assert torch.equal(sae.encode(x), reference)
    recon, feats, pre = sae.forward_with_pre(x)
    assert torch.equal(feats, reference)
    assert torch.equal(recon, sae.decode(reference))
    assert pre.shape == reference.shape


def test_auxiliary_dead_loss_returns_none_when_there_is_nothing_to_revive():
    pre = torch.randn(16, 10)
    residual = torch.randn(16, 4)
    w_dec = torch.randn(10, 4)
    no_dead = torch.zeros(10, dtype=torch.bool)
    assert auxiliary_dead_loss(pre, residual, w_dec, no_dead, k_aux=4) is None
    all_dead = torch.ones(10, dtype=torch.bool)
    assert auxiliary_dead_loss(pre, residual, w_dec, all_dead, k_aux=0) is None


def test_auxiliary_dead_loss_only_touches_dead_atoms_and_is_scale_free():
    """Gradient must reach dead atoms' decoder rows and no live ones, and the
    normalization must make the loss invariant to the residual's overall scale
    -- the whole reason `aux_coef` can be one number across models/layers."""
    torch.manual_seed(0)
    pre = torch.randn(64, 12, requires_grad=True)
    residual = torch.randn(64, 5)
    w_dec = torch.randn(12, 5, requires_grad=True)
    dead = torch.zeros(12, dtype=torch.bool)
    dead[[2, 5, 9]] = True

    loss = auxiliary_dead_loss(pre, residual, w_dec, dead, k_aux=2)
    loss.backward()
    grad_rows = w_dec.grad.abs().sum(dim=1)
    assert bool((grad_rows[dead] > 0).all()), grad_rows
    assert float(grad_rows[~dead].abs().max()) == 0.0, grad_rows

    scaled = auxiliary_dead_loss(pre.detach(), residual * 100.0, w_dec.detach() * 100.0,
                                 dead, k_aux=2)
    plain = auxiliary_dead_loss(pre.detach(), residual, w_dec.detach(), dead, k_aux=2)
    assert abs(float(scaled) - float(plain)) < 1e-3, (float(scaled), float(plain))


def test_aux_k_off_is_reproducible_and_the_flag_is_not_inert():
    """Two claims that have to hold together for `aux_k: 0` to be a safe default.

    (1) The aux-free path is bit-reproducible run to run, so every number
    already recorded against it stays regenerable. (2) Turning `aux_k` on
    actually changes the trained dictionary -- without this, a green suite
    would be equally consistent with the flag being silently ignored.
    """
    x = _planted_activations(n=800, d=16, n_causes=8, seed=3)
    base = dict(dict_size_mult=4, k=4, epochs=6, batch_size=256, seed=1)
    device = torch.device("cpu")
    sae_a, hist_a = train_sae(x, SAETrainConfig(**base), device)
    sae_b, hist_b = train_sae(x, SAETrainConfig(**base), device)
    assert hist_a == hist_b, "aux-free training must be deterministic run to run"
    assert torch.equal(sae_a.W_dec, sae_b.W_dec)

    sae_c, _ = train_sae(x, SAETrainConfig(**base, aux_k=8, aux_dead_steps=1), device)
    assert not torch.equal(sae_a.W_dec, sae_c.W_dec), "aux_k had no effect at all"


def test_aux_k_at_the_default_dead_threshold_is_not_harmful_to_a_topk_sae():
    """AuxK on a `TopKSAE` must at minimum do no damage at its default settings.

    Deliberately weaker than "revives atoms", because that stronger claim is
    **not** what the data says. On this bed AuxK's best setting moves the
    dead rate from 0.272 to 0.266 -- a wash, not a revival -- and on a second
    bed sized so that no atom dies at all it is provably inert. Whether AuxK
    helps where it matters is an empirical question about *real* activations
    and is Stage 0's H4 grid to answer (`ROADMAP.md` §6.2.1); asserting a
    revival here that the measurements don't support would be a test written
    to match the hypothesis rather than the evidence (`CLAUDE.md` §2.4).
    """
    x = _planted_activations(n=4000, d=20, n_causes=10, seed=7)
    device = torch.device("cpu")
    base = dict(dict_size_mult=16, k=4, epochs=25, batch_size=512, seed=0)

    plain, _ = train_sae(x, SAETrainConfig(**base), device)
    auxed, _ = train_sae(x, SAETrainConfig(**base, aux_k=32, aux_coef=0.03125), device)

    dead_plain = dead_feature_rate(plain, x, device)
    dead_aux = dead_feature_rate(auxed, x, device)
    fid_plain = reconstruction_fidelity(plain, x, device)
    fid_aux = reconstruction_fidelity(auxed, x, device)
    assert dead_aux <= dead_plain + 0.02, (dead_plain, dead_aux)
    assert fid_aux > fid_plain - 0.05, (fid_plain, fid_aux)


def test_aux_dead_steps_too_small_is_actively_harmful():
    """Pins the measured reason `aux_dead_steps` defaults to 20, not to a
    small number: at 4 batches, atoms that fire sporadically are mislabelled
    dead and the aux term then optimizes them away from the role they were
    actually filling, roughly doubling the dead rate (0.58 vs 0.27 plain).
    This is a regression guard on the default, not an endorsement of AuxK --
    see the test above for why the stronger claim isn't asserted anywhere.
    """
    x = _planted_activations(n=4000, d=20, n_causes=10, seed=7)
    device = torch.device("cpu")
    base = dict(dict_size_mult=16, k=4, epochs=25, batch_size=512, seed=0,
                aux_k=32, aux_coef=0.03125)

    impatient, _ = train_sae(x, SAETrainConfig(**base, aux_dead_steps=4), device)
    default, _ = train_sae(x, SAETrainConfig(**base, aux_dead_steps=20), device)
    dead_impatient = dead_feature_rate(impatient, x, device)
    dead_default = dead_feature_rate(default, x, device)
    assert dead_impatient > dead_default + 0.1, (dead_impatient, dead_default)


def test_aux_k_is_inert_when_no_atom_is_dead():
    """A dictionary small enough that every atom stays alive must train to a
    bit-identical result with AuxK on or off -- `auxiliary_dead_loss` returns
    `None` rather than adding a zero term. The same behaviour was measured on
    a larger bed (96 planted causes, 512 atoms) where the plain dead rate is
    exactly 0.000 and nine different AuxK settings reproduced it exactly."""
    x = _planted_activations(n=1500, d=16, n_causes=48, seed=23)
    device = torch.device("cpu")
    base = dict(dict_size_mult=4, k=8, epochs=10, batch_size=512, seed=0)

    plain, hist_plain = train_sae(x, SAETrainConfig(**base), device)
    assert dead_feature_rate(plain, x, device) == 0.0
    auxed, hist_aux = train_sae(x, SAETrainConfig(**base, aux_k=32, aux_coef=1.0), device)
    assert hist_plain == hist_aux
    assert torch.equal(plain.W_dec, auxed.W_dec)


def test_aux_k_revives_dead_atoms_in_a_crosscoder():
    """Same claim for the joint crosscoder, where the atoms compete across
    two sources at different scales -- the case Stage 0 actually cares about."""
    xa = _planted_activations(n=3000, d=20, n_causes=10, seed=11)
    xb = (_planted_activations(n=3000, d=14, n_causes=10, seed=12) * 8.0).astype(np.float32)
    device = torch.device("cpu")
    base = dict(dict_size_mult=8, k=4, epochs=25, batch_size=512, seed=0)

    torch.manual_seed(0)
    plain, _ = train_crosscoder([xa, xb], CrosscoderTrainConfig(**base), device)
    torch.manual_seed(0)
    auxed, hist = train_crosscoder([xa, xb], CrosscoderTrainConfig(
        **base, aux_k=32, aux_coef=0.03125, aux_dead_steps=4), device)

    alive_plain = int(cross_alive_mask(plain, [xa, xb], device).sum())
    alive_aux = int(cross_alive_mask(auxed, [xa, xb], device).sum())
    assert alive_aux > alive_plain, (alive_plain, alive_aux)
    assert "aux" in hist[-1] and hist[-1]["aux"] > 0.0, hist[-1]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
