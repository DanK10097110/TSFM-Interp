"""Recompute steering outcomes from the HTML report or its compressed numeric evidence."""
import argparse
import gzip
from html import unescape
import json
from pathlib import Path
import re

import numpy as np


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    text = path.read_text()
    blocks = [json.loads(unescape(s)) for s in re.findall(r"<pre>(.*?)</pre>", text, re.S)
              if unescape(s).lstrip().startswith("{")]
    split = next(block for block in blocks if "calibration" in block and "evaluation" in block)
    rows = json.loads(re.search(r'id="steering-results">(.*?)</script>', text).group(1))
    audits = [block for block in blocks if "evaluations" in block]
    assert len(audits) == len(rows)
    return dict(split=split, groups=[dict(summary=row, **audit) for row, audit in zip(rows, audits)])


def verify(data):
    split = data["split"]
    test = set(split["evaluation"])
    assert test.isdisjoint(split["calibration"] + split["validation"])
    assert set(split["calibration"]).isdisjoint(split["validation"])
    for group in data["groups"]:
        hits = zeros = at_radius = 0
        summary = group["summary"]
        cases = group["evaluations"]
        assert {r["series"] for r in cases} == test and len(cases) == len(test)
        for case in cases:
            base, delta, replay = [np.array(case[key]) for key in ("baseline", "requested_change", "replay")]
            matched = case["replay_matches"]
            if "forecast" in case:
                assert matched == np.allclose(replay, case["forecast"], rtol=1e-5, atol=1e-6)
            if "component_targets" in case:
                np.testing.assert_allclose(np.sum(case["component_targets"], axis=0), delta)
            diag = case["diagnostics"]
            if diag["radius"] is not None:
                assert diag["coefficient_norm"] <= diag["radius"] + 1e-8
            size = np.linalg.norm(delta)
            if size:
                error = np.linalg.norm((replay - base) - delta) / size
                np.testing.assert_allclose(error, case["relative_error"], atol=1e-9)
                passed = matched and error <= split["search"]["tolerance"]
                hits += int(passed)
                if not passed and diag["radius"] is not None:
                    at_radius += int(np.isclose(diag["coefficient_norm"], diag["radius"]))
            else:
                zeros += 1
                passed = False
            assert passed == case["verified"]
        assert hits == summary["verified"] and zeros == summary["zero_targets"]
        assert sum(c["replay_matches"] for c in cases) == summary["replay_matches"]
        print(summary["model"], summary["request"], f"{hits}/{len(cases)} verified,",
              f"{zeros} zero, {at_radius} misses at radius")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--save-evidence", type=Path, help="write the split and numeric audit blocks as gzip JSON")
    args = parser.parse_args()
    data = read(args.report)
    verify(data)
    if args.save_evidence:
        with gzip.open(args.save_evidence, "wt") as stream:
            json.dump(data, stream, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
