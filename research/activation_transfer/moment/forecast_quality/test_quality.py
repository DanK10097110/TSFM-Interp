import numpy as np

from research.activation_transfer.moment.forecast_quality.classical import seasonal
from research.activation_transfer.moment.forecast_quality.data import origins
from research.activation_transfer.moment.forecast_quality.post_release.data import CUTS, context
from research.activation_transfer.moment.forecast_quality.score import item_metrics


def test_shortest_two_origins_are_disjoint():
    assert origins(511, "m4_daily", "T1") is None
    np.testing.assert_array_equal(origins(512, "m4_daily", "T1"), [480,496])


def test_seasonal_median_preserves_next_phase_through_wrap():
    history = np.array([99, 2,6,1, 42,4,8, 1,7,3, 9,9,5])
    actual = seasonal(history, 5, period=3, cycles=4)
    np.testing.assert_allclose(actual, [5.5,6.5,4,5.5,6.5])


def test_average_scale_free_errors_within_then_across_origins():
    contexts = np.array([[[0,1,2,3],[0,10,20,30]]])
    gold = np.zeros((1,2,2))
    prediction = np.array([[[0,2],[10,30]]])
    metrics = item_metrics(prediction, gold, contexts)
    np.testing.assert_allclose(metrics["mase1"], [1.5])
    np.testing.assert_allclose(metrics["mae"], [10.5])


def test_one_sided_observation_gap_fill():
    cut = CUTS[0]
    data = np.full(cut+1, 11.)
    data[cut-5] = 26.
    data[cut-4:cut-2] = np.nan
    data[cut-2:] = 95.
    curve = np.full(366, 10.)
    filled = context(data,curve,.5,cut)
    np.testing.assert_allclose(filled[-5:], [26.,18.,14.,95.,95.])


def test_resuming_forecasts_reuses_only_matching_pass(tmp_path, monkeypatch):
    from research.activation_transfer.moment.forecast_quality import predict

    computed = []
    def forward(_adapter, inputs, horizon, _batch):
        computed.append(len(inputs))
        return np.full((len(inputs), horizon), len(computed), dtype=float)
    monkeypatch.setattr(predict, "calls", forward)
    path = tmp_path / "one_pass.npz"
    args = (None, np.ones((3, 8)), 4, 2, path)
    first = predict.cached_call(*args, "original")
    repeated = predict.cached_call(*args, "original")
    replaced = predict.cached_call(*args, "changed")
    assert [first[2], repeated[2], replaced[2]] == [False, True, False]
    np.testing.assert_array_equal(repeated[0], first[0])
    np.testing.assert_array_equal(replaced[0], np.full((3, 4), 2.))
    assert computed == [3, 3]
