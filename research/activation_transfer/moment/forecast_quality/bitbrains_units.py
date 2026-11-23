"""Identify the released GIFT-Eval VM target by comparison with original time-stamped Bitbrains fields."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa

from research.activation_transfer.moment.forecast_quality.data import OUT, SPECS


def run(path):
    with pa.memory_map(str(SPECS["bitbrains_fast_storage"][0])) as src:
        table=pa.ipc.open_stream(src).read_all()
    ids=table["item_id"].to_pylist()
    records={}
    with ZipFile(path) as archive:
        for uid in (580,631,764):
            name=f"fastStorage/2013-8/{uid}.csv"
            raw=archive.read(name)
            data=pd.read_csv(BytesIO(raw),sep=r";\s*",engine="python")
            # The CSV header says ms but its ten-digit Unix values encode seconds.
            times=data.iloc[:,0].to_numpy()
            if not (1e9<np.median(times)<2e9):
                raise ValueError("Unexpected original epoch units")
            data.index=pd.to_datetime(times,unit="s")
            which=ids.index(f"fastStorage_{uid}")
            reference=table["target"][which].values[0].values.to_numpy()
            first=pd.Timestamp(table["start"][which].as_py())
            dates=pd.date_range(first,periods=len(reference),freq="1h")
            stats={}
            for field in ("CPU usage [MHZ]","CPU usage [%]"):
                computed=data[field].resample("1h").mean().reindex(dates).to_numpy()
                valid=np.isfinite(computed)&np.isfinite(reference)
                delta=np.abs(computed[valid]-reference[valid])
                active=np.abs(reference[valid])>1.
                relative=delta/(1+np.abs(reference[valid]))
                stats[field]=dict(n_hours=int(valid.sum()),n_above_one_mhz=int(active.sum()),
                    median_relative_abs_error_above_one_mhz=float(np.median(relative[active])),
                    median_abs_error=float(np.median(delta)),
                    max_abs_error=float(delta.max()),median_relative_abs_error=float(np.median(delta/(1+np.abs(reference[valid])))),
                    pearson=float(np.corrcoef(computed[valid],reference[valid])[0,1]))
            if stats["CPU usage [MHZ]"]["median_relative_abs_error_above_one_mhz"]>1e-5 or stats["CPU usage [%]"]["median_relative_abs_error_above_one_mhz"]<.1:
                raise AssertionError((name,stats))
            records[str(uid)]=dict(member=name,member_sha256=sha256(raw).hexdigest(),gift_start=str(first),fields=stats)
    out=dict(source="https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip",
             selection="First, 32nd and 87th held-out Bitbrains items in preregistered order.", records=records)
    (OUT/"bitbrains_units.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_zip",type=Path)
    run(parser.parse_args().original_zip)
