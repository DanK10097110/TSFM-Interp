"""Form the frozen NOAA USCRN geographic and date panel from the dated official snapshot."""
import argparse
import json
import re
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SNAPSHOT = "CRND0103-202611160850.zip"
URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/snapshots/" + SNAPSHOT
ORIGINS = [date(2026, m, 10) for m in (1, 4, 7, 10)]
REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IN IL MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY CA OR WA",
}
REGION = {state: r for r, states in REGIONS.items() for state in states.split()}
START = date(2008, 1, 1)
END = ORIGINS[-1] + timedelta(days=16)
DATES = [START + timedelta(days=i) for i in range((END - START).days)]
PHASE = np.array([(date(2000, d.month, d.day) - date(2000, 1, 1)).days for d in DATES])
YEARS = np.array([d.year for d in DATES])
CUTS = np.array([(d - START).days for d in ORIGINS])
FIT_END = (date(2024, 1, 1) - START).days


def distance(a, b):
    lon, lat = np.radians(a)
    lon2, lat2 = np.radians(b)
    hav = np.sin((lat2-lat)/2)**2 + np.cos(lat)*np.cos(lat2)*np.sin((lon2-lon)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(hav))


def gather(archive, station, members):
    values = np.full(len(DATES), np.nan)
    coords = None
    for year, name in sorted(members.items()):
        if year < 2008 or year > 2026:
            continue
        for line in archive.read(name).splitlines():
            if not line.strip():
                continue
            day = line[6:14]
            y, m, d = int(day[:4]), int(day[4:6]), int(day[6:])
            idx = (date(y, m, d) - START).days
            if 0 <= idx < len(values):
                value = float(line[62:69])
                if value != -9999.0:
                    values[idx] = value
                if year == 2026:
                    coords = [float(line[22:29]), float(line[30:37])]
    if coords is None:
        raise ValueError(station)
    return values, coords


def reason(values):
    if sum(np.isfinite(values[YEARS == year]).sum() >= 300 for year in range(2008, 2024)) < 8:
        return "short_climatology"
    for t in CUTS:
        if not np.isfinite(values[t:t+16]).all():
            return "missing_future"
        missing = ~np.isfinite(values[t-480:t])
        if missing.sum() > 9 or np.convolve(missing.astype(int), np.ones(8, dtype=int), "valid").max() >= 8:
            return "missing_context"
    return None


def climate(values):
    historical = values[:FIT_END]
    p = PHASE[:FIT_END]
    available = np.isfinite(historical)
    curve = []
    for k in range(366):
        delta = abs(k-p)
        use = (np.minimum(delta, 366-delta) <= 7) & available
        curve.append(historical[use].mean())
    curve = np.array(curve)
    anomalies = historical - curve[p]
    left, right = anomalies[:-1], anomalies[1:]
    valid = np.isfinite(left) & np.isfinite(right)
    rho = np.clip(left[valid] @ right[valid] / (left[valid] @ left[valid]), 0, 1)
    return curve, float(rho)


def context(values, curve, rho, cut):
    data = values[cut-480:cut].copy()
    for j in np.flatnonzero(~np.isfinite(data)):
        k = cut - 480 + j
        prior = np.flatnonzero(np.isfinite(values[:k]))
        last = prior[-1]
        data[j] = curve[PHASE[k]] + rho**(k-last)*(values[last]-curve[PHASE[last]])
    return data


def main(path):
    OUT.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as f:
        checksum = sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            checksum.update(block)
        digest = checksum.hexdigest()
    station_map = {}
    with ZipFile(path) as archive:
        for name in archive.namelist():
            m = re.fullmatch(r"(\d{4})/CRND0103-\1-([A-Z]{2}_.+)\.txt", name)
            if m and m[2][:2] in REGION:
                station_map.setdefault(m[2], {})[int(m[1])] = name
        candidates = sorted((s for s, m in station_map.items() if 2026 in m),
                            key=lambda s: sha256(("moment-noaa-2026|" + s).encode()).digest())
        ids, coords, panel, truth, climates, decays, miss_counts = [], [], [], [], [], [], []
        log = {}
        for name in candidates:
            if len(ids) == 64:
                log[name] = "cap"
                continue
            values, coord = gather(archive, name, station_map[name])
            exclusion = reason(values)
            if exclusion is None and any(distance(coord, old) < 200 for old in coords):
                exclusion = "closer_than_200km"
            if exclusion:
                log[name] = exclusion
                continue
            curve, rho = climate(values)
            ids.append(name)
            coords.append(coord)
            histories = [context(values, curve, rho, t) for t in CUTS]
            panel.append(histories)
            truth.append([values[t:t+16] for t in CUTS])
            climates.append([curve[PHASE[t:t+16]] for t in CUTS])
            decays.append([c + rho**np.arange(1,17) * (x[-1] - curve[PHASE[t-1]])
                           for c,x,t in zip(climates[-1], histories, CUTS)])
            miss_counts.append([int(np.isnan(values[t-480:t]).sum()) for t in CUTS])
            log[name] = "accepted"
    array = np.asarray(panel)
    native = np.asarray(truth)
    metadata = dict(source=URL, archive_sha256=digest, variable="T_DAILY_AVG (degC)",
                    origins=[str(d) for d in ORIGINS],
                    ranges=[dict(history_first=str(d-timedelta(days=480)), history_last=str(d-timedelta(days=1)),
                                 target_first=str(d), target_last=str(d+timedelta(days=15))) for d in ORIGINS],
                    ordered_candidate_disposition=log, accepted=ids, coordinates_lonlat=coords,
                    regions=[REGION[s[:2]] for s in ids], filled_context_days=miss_counts,
                    sha256_context=sha256(array.tobytes()).hexdigest(),
                    sha256_future=sha256(native.tobytes()).hexdigest())
    if len(ids) < 30 or not (np.isfinite(array).all() and np.isfinite(native).all()):
        raise ValueError("Insufficient complete preselected observations")
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != metadata:
            raise ValueError("Input archive or eligibility changed the frozen station manifest")
    else:
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(OUT / "panel.npz", ids=ids, region=metadata["regions"], context=array, future=native,
                        clim=climates, anomaly=decays, imputed_days=miss_counts, coordinates=coords)
    print(len(candidates), len(ids), metadata["sha256_context"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    main(parser.parse_args().snapshot)
