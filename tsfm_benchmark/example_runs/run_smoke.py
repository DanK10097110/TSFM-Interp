"""End-to-end smoke test for the leakage-safe synthetic path.

Exercises the DTW leakage gate, the held-out private corpus (seal + verified
reload), build reproducibility across processes, and epoch regeneration yielding
a disjoint held-out set. Runs with numpy, scipy, and (optionally) dtaidistance.
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tsfm_benchmark.build_pipeline import BenchmarkBuilder, LeakageAuditor, TaskSpec, load_sealed
from tsfm_benchmark.build_pipeline.generators import parametric

auditor = LeakageAuditor(metric="dtw", threshold=0.20)
for i in range(40):
    s = parametric(length=400, seed=900_000 + i, seasonalities=[{"period": 24, "amplitude": 1.0}], noise_scale=0.2)
    auditor.add_reference("fake_real", str(i), s.values)

specs = [
    TaskSpec(
        name="seasonal_with_changepoints",
        generator="parametric",
        count=15,
        generator_params={
            "length": 400,
            "trend": {"order": 2, "scale": 0.5},
            "seasonalities": [{"period": 24, "amplitude": 1.0}, {"period": 168, "amplitude": 0.4}],
            "ar_coeffs": [0.6, -0.2],
            "noise_scale": 0.15,
            "n_changepoints": 2,
            "n_anomalies": 3,
        },
        corruptions=[{"op": "jitter", "sigma": 0.03}, {"op": "time_warp", "strength": 0.1}],
    ),
]

builder = BenchmarkBuilder(auditor=auditor)

with tempfile.TemporaryDirectory() as pub, tempfile.TemporaryDirectory() as priv:
    result = builder.build_and_seal(specs, public_dir=pub, private_dir=priv, seed=0, epoch=0)
    print(f"epoch        : {result.epoch}")
    print(f"public_dev   : {len(result.public_dev)}")
    print(f"private_test : {len(result.private_test)}")
    print(f"rejected     : {len(result.rejected)}")
    print(f"duplicates   : {len(result.duplicates)}")

    loaded, manifest = load_sealed(priv, verify=True)
    print(f"\nsealed private reloaded+verified: {len(loaded)} samples, visibility={manifest['visibility']}")
    print(f"global_digest: {manifest['global_digest'][:16]}...")

    ex = result.public_dev[0]
    print(f"\nsample id    : {ex.sample_id}")
    print(f"components   : {list(ex.ground_truth.components)}")
    print(f"changepoints : {ex.ground_truth.changepoints}")
    print(f"transforms   : {[t.op for t in ex.provenance.transforms]}")
    lr = ex.leakage_report
    print(f"leakage      : metric={lr['metric']} dist={lr['nearest_distance']:.3f} >= {lr['threshold']}")

pub_ids = {s.sample_id for s in result.public_dev}
priv_ids = {s.sample_id for s in result.private_test}
print(f"\npublic/private instance overlap (epoch 0): {len(pub_ids & priv_ids)}")

with tempfile.TemporaryDirectory() as priv2:
    regen = builder.regenerate_private(specs, private_dir=priv2, seed=0, epoch=1)
    regen_ids = {s.sample_id for s in regen}
    print(f"epoch1 private vs epoch0 private overlap : {len(priv_ids & regen_ids)}")

result_again = builder.build(specs, seed=0, epoch=0)
ids_again = {s.sample_id for s in result_again.public_dev}
print(f"reproducible across runs (same ids)      : {pub_ids == ids_again}")

near_leak = parametric(length=400, seed=900_000, seasonalities=[{"period": 24, "amplitude": 1.0}], noise_scale=0.2)
rep = auditor.audit(near_leak)
print(f"\nintentional near-leak rejected           : {not rep.passed} (dist {rep.nearest_distance:.3f})")
