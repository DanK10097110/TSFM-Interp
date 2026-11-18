"""Confusable synthetic cases for the frozen GHCNh processing contract."""
import datetime as dt
import numpy as np
import pytest
from experiments.feature_search import postrelease_weather_data as data


def record(time, value, quality=None, station='S'):
    """Make an official-schema observation with explicit quality and station."""
    return dict(STATION=station, DATE=time, temperature=value, temperature_Quality_Code=quality)


def test_qc_units_and_identity():
    rows = [record('2026-10-08T00:00:00', 20.), record('2026-10-08T00:10:00', 99., 's'),
            record('2026-10-08T00:20:00', 88., '1'), record('2026-10-08T00:30:00', float('inf'))]
    assert list(data.aggregate(rows, 'S').values()) == [20.]
    with pytest.raises(ValueError, match='identity'):
        data.aggregate(rows, 'OTHER')


def test_duplicate_weight_and_boundary():
    rows = [record('2026-10-07T23:30:00', 10.)]*5
    rows += [record('2026-10-07T23:30:00', 20.), record('2026-10-07T23:59:59', 30.),
             record('2026-10-08T00:00:00Z', 1000.)]
    got = data.aggregate(rows, 'S')
    assert got[data.timestamp('2026-10-07T23:00:00Z')] == 22.5
    assert got[data.timestamp('2026-10-08T00:00:00Z')] == 1000.
    assert data.timestamp('2026-10-07T20:00:00-04:00') == data.timestamp('2026-10-08T00:00:00Z')


def test_causal_history_and_untouched_future():
    origin = data.timestamp('2026-10-08T00:00:00Z')
    hours = {origin+dt.timedelta(hours=i): float(i) for i in range(-128, 32)}
    del hours[origin-dt.timedelta(hours=2)]
    del hours[origin+dt.timedelta(hours=1)]
    history, future, flags = data.window(hours, origin)
    assert flags == dict(history_eligible=True, history_missing=1, future_missing=1, future_complete=False)
    assert history[-2] == -3. and np.isnan(future[1]) and future[0] == 0.
    hours[origin] = 1e6
    assert np.array_equal(data.window(hours, origin)[0], history)
    del hours[origin-dt.timedelta(hours=128)]
    assert not data.window(hours, origin)[2]['history_eligible']
    hours[origin-dt.timedelta(hours=128)] = -128.
    del hours[origin-dt.timedelta(hours=3)]
    assert data.window(hours, origin)[2]['history_eligible']
    del hours[origin-dt.timedelta(hours=4)]
    assert not data.window(hours, origin)[2]['history_eligible']
