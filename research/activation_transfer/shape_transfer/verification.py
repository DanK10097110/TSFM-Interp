"""Compare complete independently regenerated real forecast archives and summary records."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCES = {
    "input_paths":("/tmp/tsfm_paths_independent_20261120",["bolt","timesfm","timer"]),
    "token_position":("/tmp/tsfm_token_position_independent_20261120",["bolt_1_patches","bolt_2_patches"]),
    "local_requests":("/tmp/tsfm_local_independent_20261120",["bolt","timesfm"]),
    "sparse_writes":("/tmp/tsfm_sparse_independent_20261120",["bolt","bolt_identity","timesfm","timesfm_identity"]),
}


def compare(first, second):
    with np.load(first) as a, np.load(second) as b:
        if set(a.files)!=set(b.files):
            raise ValueError("Mismatched result fields")
        maximum = 0.
        for k in a.files:
            left,right = a[k],b[k]
            if left.shape!=right.shape or left.dtype!=right.dtype:
                raise ValueError(f"Schema mismatch at {k}")
            if np.issubdtype(left.dtype,np.floating):
                if not np.isfinite(right).all():
                    raise ValueError(f"Nonfinite result in {k}")
                maximum = max(maximum,float(np.max(np.abs(left-right),initial=0.)))
                np.testing.assert_allclose(left,right,rtol=1e-7,atol=2e-6)
            else:
                np.testing.assert_array_equal(left,right)
        return dict(fields=len(a.files),max_abs_difference=maximum)


def main(out):
    results = {}
    for study,(scratch,names) in SOURCES.items():
        results[study] = {name:compare(HERE/study/"results"/(name+".npz"),Path(scratch)/(name+".npz"))
                          for name in names}
    out.write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=HERE/"replay.json")
    main(p.parse_args().out)
