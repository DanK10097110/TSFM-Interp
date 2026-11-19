"""Recalculate the dose and mixed pilot tables from forecast arrays, without loading a model."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np


class EvidenceReader(HTMLParser):
    """Extract the report's JSON evidence blocks and summary from the offline HTML."""
    def __init__(self):
        super().__init__()
        self.collect = False
        self.chunks = []
        self.audits = []
        self.summary = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre" or (tag == "script" and dict(attrs).get("id") == "steering-results"):
            self.collect = True
            self.chunks.clear()

    def handle_data(self, text):
        if self.collect:
            self.chunks.append(text)

    def handle_endtag(self, tag):
        if not self.collect or tag not in ("pre", "script"):
            return
        self.collect = False
        text = "".join(self.chunks).lstrip()
        if tag == "script":
            self.summary = json.loads(text)
        elif text.startswith("{"):
            record = json.loads(text)
            if "evaluations" in record:
                self.audits.append(record)


def error_and_size(actual, target):
    size = np.linalg.norm(target)
    return (float(np.linalg.norm(actual-target)/size) if size else None, np.linalg.norm(actual))


def amount(req):
    if req["kind"] == "bounds":
        return f"±{req['upper']:g}"
    for key in ("strength", "outside_weight", "delay"):
        if key in req:
            return f"{req[key]:g}"
    raise ValueError(f"Missing dose for {req}")


def doses(path, ids):
    reader = EvidenceReader()
    reader.feed(path.read_text())
    reader.close()
    assert len(reader.audits) == len(reader.summary) == 57
    totals = {}
    for summary, audit in zip(reader.summary, reader.audits):
        cases = audit["evaluations"]
        assert [c["series"] for c in cases] == ids
        hits, sizes = 0, []
        for case in cases:
            actual = np.subtract(case["replay"], case["baseline"])
            error, size = error_and_size(actual, np.array(case["requested_change"]))
            passed = case["replay_matches"] and error is not None and error <= .1
            assert case["verified"] == passed
            hits += int(passed)
            sizes.append(size)
        assert hits == summary["verified"]
        totals.setdefault(summary["kind"], []).append((amount(summary["request"]), hits, sizes))
    return totals


def increasing(rows):
    sizes = np.array([r[2] for r in rows])
    rises = (sizes[1:] > sizes[:-1]) & ~np.isclose(sizes[1:], sizes[:-1], rtol=1e-5, atol=1e-6)
    return int(rises.all(axis=0).sum())


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, default=directory / "mixed_pilot.json")
    parser.add_argument("--bolt", type=Path, default=directory / "dose_pilot_report.html")
    parser.add_argument("--timesfm", type=Path, default=directory / "dose_timesfm_report.html")
    args = parser.parse_args()
    mixed = json.loads(args.mixed.read_text())
    ids = mixed["evaluation"]
    assert len(set(ids)) == 8 and set(ids).isdisjoint(mixed["calibration"] + mixed["validation"])
    bolt = doses(args.bolt, ids)
    timesfm = doses(args.timesfm, ids)
    assert len(bolt) == len(timesfm) == 19
    print("| Request | Amounts | Bolt verified | TimesFM verified | Bolt increasing | TimesFM increasing |")
    print("| --- | --- | --- | --- | ---: | ---: |")
    for kind, a in bolt.items():
        b = timesfm[kind]
        assert len(a) == len(b) == 3 and [x[0] for x in a] == [x[0] for x in b]
        print(f"| {kind} | {', '.join(x[0] for x in a)} | {', '.join(str(x[1]) for x in a)} | "
              f"{', '.join(str(x[1]) for x in b)} | {increasing(a)} | {increasing(b)} |")
    print()
    print("| Request | Bolt verified | Bolt median error | TimesFM verified | TimesFM median error |")
    print("| --- | ---: | ---: | ---: | ---: |")
    assert len(mixed["groups"]) == 10
    for a, b in zip(mixed["groups"][:5], mixed["groups"][5:]):
        req = a["request"]
        assert req == b["request"]
        label = req["kind"] + (" + " + req["second"]["kind"] if req.get("second") else "")
        if req["kind"] == "combined":
            label = " + ".join(term["kind"] for term in req["terms"])
        for group in (a, b):
            assert [c["series"] for c in group["cases"]] == ids
            hits, errors = 0, []
            for c in group["cases"]:
                actual = np.subtract(c["replay"], c["baseline"])
                target = np.sum(c["component_targets"], axis=0)
                error, _ = error_and_size(actual, target)
                matched = np.allclose(c["forecast"], c["replay"], rtol=1e-5, atol=1e-6)
                assert c["replay_matches"] == matched
                assert error is not None and np.isclose(error, c["relative_error"])
                hits += int(matched and error <= .1)
                errors.append(error)
            assert hits == group["verified"] and np.isclose(np.median(errors), group["median_relative_error"])
        print(f"| {label} | {a['verified']}/8 | {a['median_relative_error']:.1%} | "
              f"{b['verified']}/8 | {b['median_relative_error']:.1%} |")


if __name__ == "__main__":
    main()
