"""Freeze the 24-station, 12-origin NOAA panel without inspecting future observations."""
from datetime import date
from hashlib import file_digest, sha256
from pathlib import Path
import json
import re
from zipfile import ZipFile

import numpy as np

from research.activation_transfer.moment.forecast_quality.post_release import data
from research.activation_transfer.moment.crosscoder.evaluate import write_json, write_npz

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"
DEFAULT=Path("/tmp/tsfm_moment_quality_20261122/noaa")/data.SNAPSHOT
ORIGINS=[date(2025,m,15) for m in range(1,13)]
CUTS=np.array([(d-data.START).days for d in ORIGINS])


def missing(v):
    masks=np.stack([~np.isfinite(v[t-480:t]) for t in CUTS])
    runs=np.array([np.max(np.convolve(m.astype(int),np.ones(8,int),"valid")) for m in masks])
    return masks.sum(1),runs


def main():
    previous=json.loads((data.OUT/"manifest.json").read_text())
    with DEFAULT.open("rb") as f:
        assert file_digest(f,"sha256").hexdigest()==previous["archive_sha256"]
    candidates=sorted(previous["ordered_candidate_disposition"],
                      key=lambda s:sha256(("moment-readout-month15-2025|"+s).encode()).digest())
    rows=[];log={}
    with ZipFile(DEFAULT) as archive:
        files={}
        for name in archive.namelist():
            m=re.fullmatch(r"(\d{4})/CRND0103-\1-([A-Z]{2}_.+)\.txt",name)
            if m and m[2] in candidates:
                files.setdefault(m[2],{})[int(m[1])]=name
        for s in candidates:
            if len(rows)==24:
                log[s]="not_reached";continue
            v,coords=data.gather(archive,s,files[s])
            counts,runs=missing(v)
            if sum(np.isfinite(v[data.YEARS==year]).sum()>=300 for year in range(2008,2024))<8:
                log[s]="short_climatology";continue
            if np.max(counts)>9 or np.max(runs)>=8:
                log[s]="missing_history";continue
            if any(data.distance(coords,row[1])<200 for row in rows):
                log[s]="spacing";continue
            curve,rho=data.climate(v)
            x=np.stack([data.context(v,curve,rho,t) for t in CUTS]).astype("f4")
            assert x.shape==(12,480) and np.isfinite(x).all()
            rows.append((s,coords,x,counts));log[s]="selected"
    assert len(rows)==24
    ids,coords,x,n=(np.array(a) for a in zip(*rows))
    dates=np.array([str(t) for t in ORIGINS])
    manifest=dict(archive_sha256=previous["archive_sha256"],origins=dates.tolist(),ids=ids.tolist(),disposition=log,
                  raw_input_sha256=sha256(x.tobytes()).hexdigest(),missing_filled=int(n.sum()),
                  min_distance_km=min(data.distance(a,b) for i,a in enumerate(coords) for b in coords[:i]),
                  previously_in_46_station_panel=sorted(set(ids)&set(previous["accepted"])))
    from research.activation_transfer.moment.dictionary_transfer.weather.new_dates.run import OUT as external
    with np.load(external/"ordinary.npz") as f:
        assert not set(dates)&set(f["dates"])
        manifest["previously_in_35_station_panel"]=sorted(set(ids)&set(f["ids"]))
    assert not set(dates)&set(previous["origins"])
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"manifest.json").exists():
        assert json.loads((OUT/"manifest.json").read_text())==manifest
    write_json(OUT/"manifest.json",manifest)
    write_npz(OUT/"panel.npz",ids=ids,dates=dates,x=x,missing=n,coords=coords)
    print("Station/date grid",x.shape,"minimum spacing",manifest["min_distance_km"],"fills",n.sum(),flush=True)


if __name__=="__main__":
    main()
