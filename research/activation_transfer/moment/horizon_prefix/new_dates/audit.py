"""Independently check station IDs, outcome dates, input values and paired error calculations."""
from datetime import date, timedelta
from hashlib import sha256
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

OUT=Path(__file__).resolve().parent/"results"
OLD=OUT.parents[2]/"forecast_quality/post_release/results"


def audit(archive):
    info=json.loads((OUT/"manifest.json").read_text())
    old=json.loads((OLD/"manifest.json").read_text())
    report=json.loads((OUT/"summary.json").read_text())
    assert sha256(archive.read_bytes()).hexdigest()==info["archive_sha256"]==old["archive_sha256"]
    predicted_dates=[date.fromisoformat(d) for d in info["origins"]]
    def days(origins,count):
        return {o+timedelta(days=d) for d in range(count) for o in origins}
    assert days(predicted_dates,8).isdisjoint(days([date.fromisoformat(s) for s in old["origins"]],16))
    with np.load(OUT/"panel.npz") as f:ids,truth,x,missing,g=f["ids"],f["y"],f["x"],f["missing"],f["cohort"]
    assert set(ids[g=="primary"]).issubset(old["accepted"]) and set(ids[g=="secondary"]).isdisjoint(old["accepted"])
    counts=dict(observed=0,imputed=0,outcomes=0)
    with ZipFile(archive) as z:
        for i,station in enumerate(ids):
            raw={}
            for yr in (2024,2025,2026):
                fname=f"{yr}/CRND0103-{yr}-{station}.txt"
                if fname not in z.namelist():continue
                for line in z.read(fname).splitlines():
                    day=line[6:14].decode()
                    val=float(line[62:69])
                    if val!=-9999:
                        raw[date(int(day[:4]),int(day[4:6]),int(day[6:]))]=val
            for j,origin in enumerate(predicted_dates):
                future=[raw[origin+timedelta(days=k)] for k in range(8)]
                np.testing.assert_array_equal(truth[i,j],future)
                counts["outcomes"]+=len(future)
                expected=np.array([raw.get(origin-timedelta(days=k),np.nan) for k in range(480,0,-1)])
                observed=np.isfinite(expected)
                np.testing.assert_array_equal(expected[observed],x[i,j,observed])
                assert np.sum(~observed)==missing[i,j]
                counts["imputed"]+=int(np.sum(~observed))
                counts["observed"]+=int(np.sum(observed))
    for n in ("primary","secondary"):
        keep=g==n
        for size,horizon in (("large",16),("large",32),("small",16)):
            with np.load(OUT/f"{size}_{horizon}.npz") as f:long=f["y"][keep]
            with np.load(OUT/f"{size}_8.npz") as f:short=f["y"][keep]
            for key,per in (("MAE",abs(long-truth[keep])-abs(short-truth[keep])),("RMS",(long-short)**2)):
                by_station_date=per.mean(-1)
                if key=="RMS":by_station_date=np.sqrt(by_station_date)
                check=report[n][f"{size}_{horizon}_minus_direct_{key}_C"]
                np.testing.assert_allclose(by_station_date.mean(0),check["per_date"],atol=1e-6)
                np.testing.assert_allclose(by_station_date.mean(),check["mean"],atol=2e-6)
    (OUT/"audit.json").write_text(json.dumps(counts,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,default=Path("/tmp/tsfm_moment_quality_20261122/noaa/CRND0103-202611160850.zip"))
    audit(p.parse_args().archive)
