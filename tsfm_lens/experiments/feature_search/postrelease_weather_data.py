"""Model-free GHCNh preparation under the sealed October access amendment.

Preserve scheduled windows, isolate targets, and enforce causal missingness.
No model import or forecast selection occurs here. Temperature is already Celsius.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime as dt
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from experiments.feature_search.persistence_protocol import read, write, sha
from experiments.feature_search.postrelease_weather_prepare import utcnow


def timestamp(value):
    """Interpret NOAA's documented timezone-free ISO timestamps as UTC."""
    value = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def aggregate(records, station):
    """Average distinct temperatures per timestamp, then timestamps per UTC hour."""
    exact = defaultdict(set)
    for row in records:
        if row['STATION'] != station:
            raise ValueError('Station identity mismatch')
        value = row['temperature']
        if row['temperature_Quality_Code'] not in (None, '') or value is None:
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        exact[timestamp(row['DATE'])].add(value)
    hourly = defaultdict(list)
    for time, values in exact.items():
        hour = time.replace(minute=0, second=0, microsecond=0)
        hourly[hour].append(float(np.mean(sorted(values))))
    return {hour: float(np.mean(values)) for hour, values in hourly.items()}


def window(hourly, origin):
    """Return history eligibility separately from untouched future completeness."""
    origin = timestamp(origin)
    history = np.array([hourly.get(origin+dt.timedelta(hours=k), np.nan) for k in range(-128, 0)])
    future = np.array([hourly.get(origin+dt.timedelta(hours=k), np.nan) for k in range(32)])
    missing = ~np.isfinite(history)
    eligible = bool(missing.sum() <= 2 and not missing[0])
    if eligible:
        for i in range(1, len(history)):
            if missing[i]:
                history[i] = history[i-1]
    return history, future, dict(history_eligible=eligible, history_missing=int(missing.sum()),
                                 future_missing=int((~np.isfinite(future)).sum()),
                                 future_complete=bool(np.isfinite(future).all()))


def prepare(root, output_name='prepared'):
    """Write finite history arrays and a separate target file with explicit masks."""
    stage = root/'ghcnh_oct2026'
    protocol = read(stage/'protocol.json')
    assert sha(stage/'protocol.json') == read(stage/'protocol_seal.json')['sha256']
    dest = stage/output_name
    dest.mkdir(exist_ok=False)
    histories, targets, masks, rows, inventory = [], [], [], [], []
    for airport in protocol['airports']:
        file = root/'raw'/airport['url'].split('/')[-1]
        expected = next(r for r in read(stage/'availability.json')['stations'] if r['icao'] == airport['icao'])
        assert sha(file) == expected['raw_sha256']
        records = pq.read_table(file, columns=['STATION', 'DATE', 'temperature', 'temperature_Quality_Code']).to_pylist()
        records = [r for r in records if str(r['DATE']).startswith('2026-10')]
        hourly = aggregate(records, airport['ghcnh_id'])
        inventory.append(dict(icao=airport['icao'], accepted_hours=len(hourly), missing_month_hours=744-len(hourly)))
        for origin in protocol['origins_utc']:
            history, future, flags = window(hourly, origin)
            index = len(histories) if flags['history_eligible'] else None
            rows.append(dict(icao=airport['icao'], origin=origin, input_index=index, **flags))
            if index is not None:
                histories.append(history)
            targets.append(np.where(np.isfinite(future), future, 0.))
            masks.append(np.isfinite(future))
    x = np.asarray(histories, dtype=np.float64).reshape(-1, 128)
    assert np.isfinite(x).all()
    np.savez(dest/'histories.npz', history=x)
    np.savez(dest/'targets.npz', target=np.asarray(targets), observed=np.asarray(masks))
    write(dest/'windows.json', rows)
    summary = dict(created_utc=utcnow(), scheduled_windows=len(rows), eligible_histories=len(x),
                   complete_futures=sum(r['future_complete'] for r in rows),
                   scoreable_windows=sum(r['history_eligible'] and r['future_complete'] for r in rows),
                   scoreable_stations=len({r['icao'] for r in rows if r['history_eligible'] and r['future_complete']}),
                   history_filled_bins=sum(r['history_missing'] for r in rows if r['history_eligible']),
                   station_inventory=inventory, native_launched=False,
                   target_storage='Zeros ONLY masked storage placeholders, never imputed or scored; target row index is scheduled window index.',
                   input_mapping='histories rows indexed by windows.input_index; eligibility does not depend on targets.')
    write(dest/'summary.json', summary)
    write(dest/'seal.json', dict(protocol_sha256=sha(stage/'protocol.json'),
                                files={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
                                source_sha256=sha(Path(__file__))))
    knowledge = read(stage/'knowledge.json')
    knowledge['exact_findings']['window_audit'] = summary
    knowledge['evidence_class'] = 'observational data availability and provenance; no model performance evidence'
    knowledge['next_decision'] = 'Wait for native slot grant; unchanged fixed depth1 hybrid versus native medians and seasonal24. No expansion.'
    write(dest/'knowledge.json', knowledge)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.run))
