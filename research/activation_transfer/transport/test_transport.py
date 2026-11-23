"""Known-coordinate checks for transport, causal scoring, splits and matched energy."""

import numpy as np
import pytest

from research.activation_transfer.transport.analyze import fit_bridge, metrics, scaled_like, sufficient
from research.activation_transfer.transport.prepare import patterns


def test_affine_map_transports_a_new_signed_edit_without_its_offset():
    rng = np.random.default_rng(90)
    x = rng.normal(size=(180, 11))
    weights = rng.normal(size=(11, 17))
    offset = rng.normal(size=17)
    y = x @ weights + offset
    model, _ = fit_bridge(x, y)
    delta = rng.normal(size=(12, 4, 11))
    np.testing.assert_allclose(model.transport(delta), delta @ weights, rtol=.005, atol=.002)
    np.testing.assert_allclose(model.predict(x[:12]) - model.predict(x[:12] - delta[:, 0]),
                               model.transport(delta[:, 0]), atol=1e-12)
    shuffled, _ = fit_bridge(x, rng.permutation(y))
    assert np.mean((shuffled.transport(delta) - delta @ weights)**2) > 1


def test_forecast_energy_recovery_penalizes_magnitude_and_sign():
    v = np.random.default_rng(2).normal(size=(8, 4, 16))
    for factor, expected in ((0, 0.), (1, 1.), (2, 0.), (-1, -3.)):
        got = metrics(sufficient(v * factor, v).sum(axis=0))
        assert got["recovery"] == pytest.approx(expected)
        assert got["gain"] == pytest.approx(factor)


def test_local_waveforms_and_rowwise_norm_matching():
    waves = patterns()
    assert waves.shape == (4, 128)
    assert np.all(waves[:, :-32] == 0)
    assert np.all(waves.max(axis=1) == 1)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(12, 4, 23))
    reference = rng.normal(size=z.shape)
    matched = scaled_like(z, reference)
    np.testing.assert_allclose(np.linalg.norm(matched, axis=2), np.linalg.norm(reference, axis=2))
    with pytest.raises(ValueError):
        scaled_like(z * 0, reference)


def test_hook_targets_only_two_observed_patches_and_leaves_reg_token_alone():
    import torch
    from research.activation_transfer.transport.capture import run_pass, signed_response

    class ToyAdapter:
        def __init__(self):
            self.module = torch.nn.Sequential(torch.nn.Identity()).eval()

        def all_layer_names(self):
            return ["0"]

        def token_time_spans(self):
            starts = 16 * np.arange(8)
            return np.column_stack((starts, starts + 16))

        def predict(self, x, horizon, quantiles):
            h = torch.from_numpy(x.copy()).reshape(len(x), 8, 16)
            h = torch.cat((h, 50 * torch.ones(len(x), 1, 16)), dim=1)
            h = self.module(h)
            out = h[:, 6] + 2 * h[:, 7] + 5 * h[:, 8]
            return {"quantiles": out.numpy()[..., None]}

    rng = np.random.default_rng(7)
    x = rng.normal(size=(13, 128)).astype(np.float32)
    a = ToyAdapter()
    captured, f = run_pass(a, x, batch_size=5)
    np.testing.assert_allclose(captured, x[:, -32:])
    np.testing.assert_allclose(f, x[:, -32:-16] + 2 * x[:, -16:] + 250)
    d = rng.normal(size=(len(x), 4, 32)).astype(np.float32)
    expected = d[:, :, :16] + 2 * d[:, :, 16:]
    measured = signed_response(a, x, d, batch_size=5)
    np.testing.assert_allclose(measured, expected, atol=2e-5)


def test_history_derangement_and_cluster_bootstrap_preserve_pairing():
    from research.activation_transfer.transport.context_controls import bootstrap_summary, donors_and_clusters
    n = 8
    donor, pairs = donors_and_clusters(n, 19)
    assert np.all(donor != np.arange(n))
    np.testing.assert_array_equal(donor[donor], np.arange(n))
    rng = np.random.default_rng(345)
    target = rng.normal(size=(n, 2, 4)) + np.array([3, -4])[None, :, None]
    template = np.broadcast_to(target.mean(0), target.shape)
    result = bootstrap_summary({"bridge": target, "template": template}, target, pairs, n_boot=25)
    assert result["full"]["methods"]["bridge"]["recovery"]["value"] == pytest.approx(1.)
    assert result["background"]["methods"]["template"]["recovery"]["value"] == pytest.approx(0., abs=1e-10)
    assert result["full"]["methods"]["template"]["recovery"]["value"] > .8
    np.testing.assert_allclose(result["background_shape"]["methods"]["bridge"]["recovery"]["ci95"], [1, 1])


def test_observed_windows_have_same_real_time_anchor_and_lengths():
    from research.activation_transfer.transport.natural import contexts
    segment = np.arange(136)[None, :]
    x, high, low = contexts(segment)
    np.testing.assert_array_equal(x[0], np.arange(4, 132))
    np.testing.assert_array_equal(high - x[:, None, :], np.broadcast_to(np.array([1, 4])[None, :, None], high.shape))
    np.testing.assert_array_equal(x[:, None, :] - low, high - x[:, None, :])


def test_reduced_rank_recovers_planted_predictable_subspace():
    from research.activation_transfer.transport.rank import fitted_basis, projected
    rng = np.random.default_rng(55)
    x = rng.normal(size=(140, 14))
    latent = rng.normal(size=(14, 4))
    target_basis, _ = np.linalg.qr(rng.normal(size=(31, 4)))
    y = x @ latent @ target_basis.T + np.linspace(-3, 3, 31)
    model, _ = fit_bridge(x, y)
    singular, basis = fitted_basis(model, x)
    assert len(singular) == 4
    test = model.transport(rng.normal(size=(10, 2, 14)))
    np.testing.assert_allclose(projected(test, basis), test, atol=1e-11)
    assert np.linalg.norm(projected(test, basis[:1]) - test) > 1


def test_centered_bootstrap_statistics_match_direct_sample_variance():
    from research.activation_transfer.transport.rank import energy_stats
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(2, 7, 2, 4))
    index = np.array([0, 0, 3, 3, 3, 6, 4])
    weights = np.bincount(index, minlength=len(b))[None, :]
    stats = energy_stats(a, b, weights, centered=True)
    v, y = a[index], b[index]
    direct = sufficient(v-v.mean(0), y-y.mean(0)).sum(axis=0)
    np.testing.assert_allclose(stats[0], direct)


def test_random_source_rank_flags_degenerate_and_full_rank_fake_states():
    from research.activation_transfer.transport.random_source import geometry
    zero = geometry(np.ones((3, 10)))
    assert zero['stable_rank'] == 0
    v = np.arange(8.0)[:, None] * np.ones((1, 12))
    one = geometry(v)
    assert one['effective_rank'] == pytest.approx(1.)
    assert one['rank_99'] == 1


def test_rewiring_breaks_hidden_function_but_preserves_every_matrix_spectrum():
    import torch
    from research.activation_transfer.transport.rewire import scramble
    torch.manual_seed(98)
    model = torch.nn.Sequential(torch.nn.Linear(13, 18), torch.nn.GELU(), torch.nn.Linear(18, 5))
    x = torch.randn(9, 13)
    baseline = model(x).detach()
    matrices = {k: v.detach().clone() for k, v in model.named_parameters() if v.ndim == 2}
    counts = scramble(model, 11)
    assert counts[2] == 2 and not torch.allclose(baseline, model(x))
    for k, v in matrices.items():
        np.testing.assert_allclose(torch.linalg.svdvals(v).numpy(),
                                   torch.linalg.svdvals(model.get_parameter(k).detach()).numpy(),
                                   rtol=1e-5, atol=1e-6)


def test_source_axis_orientation_dose_and_group_separation():
    from research.activation_transfer.transport.source_axes import training_directions
    rng = np.random.default_rng(21)
    x = rng.normal(size=(120, 75))
    deltas = rng.normal(size=(120, 4, 75))
    dirs = training_directions(x, deltas)
    assert dirs.shape == (8, 75)
    np.testing.assert_allclose(dirs[4:], deltas.mean(0))
    normed = dirs[:4] / np.linalg.norm(dirs[:4], axis=1, keepdims=True)
    np.testing.assert_allclose(normed @ normed.T, np.eye(4), atol=1e-12)
    expected = .5 * np.std(x @ normed.T, axis=0, ddof=1)
    np.testing.assert_allclose(expected, np.linalg.norm(dirs[:4], axis=1))
