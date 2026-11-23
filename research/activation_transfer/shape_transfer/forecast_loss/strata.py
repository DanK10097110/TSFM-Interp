"""Independently rescore event signs and describe their dependence on clean forecast disagreement."""
import argparse
import json
from pathlib import Path

import numpy as np

from research.activation_transfer.shape_transfer.forecast_loss.run import OUT, load, TOLERANCE


def run(out):
    primary = json.loads((out/"summary.json").read_text())
    record = {}
    for panel in ("extra","blind"):
        with np.load(out/(panel+".npz")) as f:
            ids,split,truth = f["ids"],f["split"],f["future"]
        record[panel] = {}
        for model,target in (("bolt","timesfm"),("timesfm","bolt")):
            atoms,active,ds,dt,bs,bt = load(panel,model,target,ids)
            sides = np.sum(np.sign(bs-truth)==np.sign(bt-truth),axis=1)
            source = np.abs(bs[:,None,:]+ds-truth[:,None,:]).mean(-1) - np.abs(bs-truth).mean(-1)[:,None]
            receiver = np.abs(bt[:,None,:]+dt-truth[:,None,:]).mean(-1) - np.abs(bt-truth).mean(-1)[:,None]
            block = {}
            for domain in np.unique(split):
                allrows = (split==domain)&np.isfinite(truth).all(-1)
                eligible = active[allrows].sum(axis=0)>=8
                row = {}
                for name,stratum in (("all",np.ones(len(ids),bool)),("at_least_six",sides>=6),("five",sides==5),("at_most_four",sides<=4)):
                    scores,hits,ncases = [],0,0
                    for j in np.flatnonzero(eligible):
                        keep = allrows & stratum & active[:,j] & (np.abs(source[:,j])>TOLERANCE) & (np.abs(receiver[:,j])>TOLERANCE)
                        if keep.any():
                            same = np.sign(source[keep,j])==np.sign(receiver[keep,j])
                            scores.append(same.mean())
                            hits+=int(same.sum())
                            ncases+=len(same)
                    row[name] = dict(histories=int((allrows&stratum).sum()),features_represented=len(scores),events=ncases,
                                     equal_feature=float(np.mean(scores)) if scores else None,
                                     event_weighted=hits/ncases if ncases else None)
                np.testing.assert_allclose(row["all"]["equal_feature"],primary[panel][model][domain]["mae"]["physical"]["agree"]["value"],atol=1e-12)
                block[str(domain)] = row
            record[panel][model] = block
    (out/"strata.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=OUT)
    run(p.parse_args().out)
