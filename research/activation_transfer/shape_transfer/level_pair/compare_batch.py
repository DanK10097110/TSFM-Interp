"""Check a complete independent run at a different batch size, including sparse firing decisions."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify(primary,other):
    fields={};firing={}
    with np.load(primary) as first,np.load(other) as replay:
        assert set(first.files)==set(replay.files)
        for key in first.files:
            a,b=first[key],replay[key]
            if a.dtype.kind in 'USib':
                np.testing.assert_array_equal(a,b);continue
            diff=float(np.max(abs(a.astype(float)-b.astype(float))));fields[key]=diff
            if key.endswith('_z'):
                changed=int(((a>0)!=(b>0)).sum());firing[key]=changed
            np.testing.assert_allclose(a,b,atol=3e-4,rtol=.004)
    return dict(per_array_max_abs=fields,firing_disagreements=firing)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('primary',type=Path);p.add_argument('other',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();a.out.write_text(json.dumps(verify(a.primary,a.other),indent=2)+'\n')
