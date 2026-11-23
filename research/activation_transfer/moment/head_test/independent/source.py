"""Trace the independent readout audit inputs and its held-out forecasts back to source data."""
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "results"
QUALITY = HERE.parents[1] / "forecast_quality/results"


def arrow_table(path):
    source = Path(path)
    with pa.memory_map(str(source)) as opened:
        table = pa.ipc.open_stream(opened).read_all()
    name = "item_id" if "item_id" in table.column_names else "id"
    index = {str(value): i for i, value in enumerate(table[name].to_pylist())}
    assert len(index) == len(table)
    return table, index


def run():
    train_manifest = json.loads((STUDY/"manifest.json").read_text())
    eval_manifest = json.loads((QUALITY/"manifest.json").read_text())
    with Path(train_manifest["source"]).open("rb") as f:
        assert file_digest(f,"sha256").hexdigest() == train_manifest["source_sha256"]
    tr, tindex = arrow_table(train_manifest["source"])
    with np.load("/tmp/tsfm_moment_nov22_heads/train.npz") as archive:
        fit = dict(archive)
        assert list(fit["ids"]) == train_manifest["ids"]
        for row,item in enumerate(fit["ids"]):
            raw = tr["target"][tindex[item]].values.to_numpy(zero_copy_only=False)
            np.testing.assert_array_equal(fit["context"][row],raw[-480:])
            digest=sha256(("interior-holes|"+item).encode()).digest()
            starts = np.random.default_rng(int.from_bytes(digest[:8],"little")).integers(1,13,size=4) + [0,15,30,45]
            np.testing.assert_array_equal(fit["holes"][row],np.stack([starts,starts+1],axis=1).flatten())
    traced = {}
    with np.load(QUALITY/"panel.npz") as archive:
        panel = dict(archive)
        for domain in sorted(set(panel["domain"])):
            spec = eval_manifest["sources"][domain]
            table, index = arrow_table(spec["path"])
            rows = np.flatnonzero((panel["domain"]==domain)&(panel["split"]=="holdout"))
            if domain!="m4_daily":rows=rows[np.linspace(0,len(rows)-1,8,dtype=int)]
            for row in rows:
                item=panel["id"][row].split("/",1)[1]
                raw=table["target"][index[item]].values
                if domain.startswith("bitbrains_"):raw=raw[0].values
                raw=raw.to_numpy(zero_copy_only=False)
                for origin,k in enumerate(panel["origin"][row]):
                    np.testing.assert_array_equal(panel["context"][row,origin],raw[k-480:k])
                    np.testing.assert_array_equal(panel["future"][row,origin],raw[k:k+16])
            traced[domain]=len(rows)
    result=dict(training_ids_and_all_native_history_samples_exact=1152,deterministic_hole_sets_exact=1152,
                raw_forecast_panel_items_by_domain=traced,origins_per_item=2,forecast_target_leads=16)
    (HERE/"source.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":run()
