import numpy as np

from research.activation_transfer.moment.forecast_probe.capture import native, terminal_split
from research.activation_transfer.moment.forecast_probe.fit import GRID, raw_forecast, ridge_arms


def test_withheld_labels_impervious_and_backward_edge_geometry():
    source = np.arange(960, dtype=float).reshape(2, 480) + 117
    known = source[:, :464]
    before, truth = terminal_split(source.copy())
    np.testing.assert_array_equal(truth, source[:, 464:])
    source[:, 464:] *= -1e6
    after, _ = terminal_split(source)
    np.testing.assert_array_equal(before, known)
    np.testing.assert_array_equal(after, known)
    result = native(after)
    mask = result["input_mask"].numpy()
    padded = result["x_enc"].numpy()[:, 0]
    np.testing.assert_array_equal(mask.sum(-1), [464, 464])
    assert np.all(mask[:, :48] == 0) and np.all(mask[:, 48:] == 1)
    np.testing.assert_array_equal(padded[:, 48:], known)
    np.testing.assert_array_equal(padded[:, :48], np.tile(known[:, :1], (1, 48)))


def test_dual_matches_independent_primal_for_both_target_parameterizations():
    rng = np.random.default_rng(195611)
    fitting = np.arange(36) < 25
    x = rng.normal(size=(36, 5)) + np.where(fitting, 12., 27.)[:, None]
    last = x[:, 1] + rng.normal(size=36)
    y = (x - 12) @ rng.normal(size=(5, 4)) + 2 * last[:, None] + rng.normal(size=(36, 4))
    got = ridge_arms(x, fitting, y, last)
    t = x[fitting]
    tx = t.mean(0)
    gram = (t-tx).T @ (t-tx)
    scale = np.linalg.norm(gram, 2)
    for name, z in (("direct", y), ("residual", y-last[:,None])):
        ty = z[fitting].mean(0)
        ref = []
        for relative in GRID:
            w = np.linalg.solve(gram + np.eye(5) * relative * scale, (t-tx).T @ (z[fitting]-ty))
            val = (x[~fitting]-tx) @ w + ty
            ref.append((np.mean((z[~fitting]-val)**2), w))
        best = min(ref, key=lambda pair: pair[0])[1]
        np.testing.assert_allclose(got[name]["meta"]["normalized_validation_mse"], [a for a,_ in ref], rtol=1e-8)
        np.testing.assert_allclose(got[name]["w"], best, rtol=1e-8)
        np.testing.assert_allclose(got[name]["b"], ty - tx @ best, rtol=1e-8)


def test_forecast_restores_native_scale_and_only_one_last_value_skip():
    x = np.array([[1., 3.], [-2., 4.]])
    w = np.array([[1., 2.], [-3., .5]])
    b = np.array([2., 1.])
    mu, sd, last = np.array([20., -400.]), np.array([4., .25]), np.array([30., -407.])
    native = np.array([[-4., 38.], [-403., -400.25]])
    np.testing.assert_array_equal(raw_forecast(x, w, b, mu, sd, last, False), native)
    np.testing.assert_array_equal(raw_forecast(x, w, b, mu, sd, last, True), native + [[10.], [-7.]])
