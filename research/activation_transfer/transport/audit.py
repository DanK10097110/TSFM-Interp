"""Recompute reported transport endpoints directly from released series-level arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def recovery(predicted, target):
    """Independent scalar implementation of the primary pooled statistic."""
    p, t = predicted.astype(float), target.astype(float)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("invalid forecast shapes or nonfinite response")
    return 1 - np.square(p - t).sum() / np.square(t).sum()


def same(a, b):
    np.testing.assert_allclose(a, b, atol=5e-6, rtol=5e-6)


def audit(out):
    """Check identity splits, donor swaps, reported endpoints and random/rank means."""
    original = json.loads((out / "manifest.json").read_text())["items"]
    natural = json.loads((out / "natural_manifest.json").read_text())
    natural_ids = [item for cohort in natural.values() for item in cohort["items"]]
    assert len(original) == 640 and len(set(natural_ids)) == 358 and not set(natural_ids).intersection(original)
    checks = 0
    for prefix in ("timesfm_to_bolt", "bolt_to_timesfm"):
        fixed_file = out / (prefix + ".npz")
        with np.load(fixed_file) as data:
            fixed = dict(data)
        main = json.loads((out / (prefix + ".json")).read_text())
        assert hashlib.sha256(fixed_file.read_bytes()).hexdigest() == main["archive_sha256"]
        assert list(map(original.get, fixed["ids"])) == list(fixed["split"])
        with np.load(out / (prefix + "_controls.npz")) as data:
            extra = dict(data)
        np.testing.assert_array_equal(fixed["ids"], extra["ids"])
        donor = dict(zip(extra["ids"], extra["donor_ids"]))
        assert all(i != j and donor[j] == i for i, j in donor.items())
        controls = json.loads((out / (prefix + "_controls.json")).read_text())
        for group, report in main["splits"].items():
            rows = fixed["split"] == group
            for key, stats in report["methods"].items():
                p, t = fixed[key][rows], fixed["native"][rows]
                same(recovery(p, t), stats["full"]["recovery"]["value"])
                same(recovery(p - p.mean(-1, keepdims=True), t - t.mean(-1, keepdims=True)),
                     stats["shape"]["recovery"]["value"])
                checks += 2
            combined = {**fixed, **extra}
            for key, stats in controls[group]["background"]["methods"].items():
                p, t = combined[key][rows], fixed["native"][rows]
                same(recovery(p.astype(float) - p.mean(0, dtype=float), t.astype(float) - t.mean(0, dtype=float)),
                     stats["recovery"]["value"])
                checks += 1
        with np.load(out / (prefix + "_natural.npz")) as data:
            z = dict(data)
        assert list(z["ids"]) == natural_ids
        reports = json.loads((out / (prefix + "_natural.json")).read_text())
        for group, report in reports.items():
            rows = z["group"] == group
            for key, stats in report["full"]["methods"].items():
                same(recovery(z[key][rows], z["native"][rows]), stats["recovery"]["value"])
                checks += 1
        for study in ("random_source", "rewired", "rank"):
            stem = prefix + "_" + study
            reports = json.loads((out / (stem + ".json")).read_text())["groups"]
            with np.load(out / (stem + ".npz")) as data:
                more = dict(data)
            np.testing.assert_array_equal(more["ids"], z["ids"])
            np.testing.assert_array_equal(more["native"], z["native"])
            for group, report in reports.items():
                rows = z["group"] == group
                t = more["native"][rows]
                if study == "rank":
                    for rank, entry in report["full"]["rank"].items():
                        s = recovery(more["principal_" + rank][rows], t)
                        same(s, entry["score"]["value"])
                        checks += 1
                        for control in ("uniform", "covariance"):
                            scores = [recovery(more[k][rows], t) for k in more
                                      if k.startswith(f"{control}_{rank}_")]
                            assert len(scores) == 3
                            same(np.mean(scores), entry["controls"][control]["average"]["value"])
                            same(s - np.mean(scores), entry["controls"][control]["paired_difference"]["value"])
                            checks += 2
                else:
                    for key in ["trained", "raw", "shuffle", *[k for k in more if k.startswith(("active_", "factory_", "rewired_"))]]:
                        same(recovery(more[key][rows], t), report["full"][key]["value"])
                        checks += 1
        direct = out / (prefix + "_axes.json")
        reports = json.loads(direct.read_text())["groups"]
        data = dict(np.load(out / (prefix + "_axes.npz")))
        manifest = json.loads((out / "axes_manifest.json").read_text())
        axes_ids = [item for cohort in manifest.values() for item in cohort["ids"]]
        assert list(data["ids"]) == axes_ids and len(set(axes_ids)) == 224
        assert not set(axes_ids).intersection(natural_ids) and not set(axes_ids).intersection(original)
        for group, reports_by_view in reports.items():
            rows = data["groups"] == group
            for family, indices in (("pca", range(4)), ("named", range(4,8))):
                for method in ("bridge", "shuffle", "wrong_1", "wrong_2", "wrong_3", "template"):
                    score = np.mean([recovery(data[method][rows, i], data["source"][rows, i]) for i in indices])
                    same(score, reports_by_view["full"][family][method]["recovery"]["value"])
                    checks += 1
    print(f"{checks} scalar endpoints reproduced from original forecast arrays; split IDs and donor pairs verified.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    audit(p.parse_args().out)
