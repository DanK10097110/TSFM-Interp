"""Cross-reference actual prior item memberships, keeping exclusion registries out of the used-item union."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[2]
ROOT = SUITE.parents[1]
OUT = HERE / "results"


def read(path):
    return json.loads((SUITE / path).read_text())


def cross_reference():
    manifest = json.loads((OUT / "confirm_manifest.json").read_text())
    prior = {}
    def add(name, path, values, default=None):
        entry = prior.setdefault(name, dict(path=path, daily=set(), weekly=set()))
        for value in values:
            label = str(value).strip("/").split("/")
            domain = label[-2].removeprefix("m4_") if len(label) >= 2 else default
            if domain in ("daily", "weekly"):
                entry[domain].add(label[-1])

    benchmark = ROOT / "benchmark_v3/external_real/corpus.jsonl"
    for line in benchmark.read_text().splitlines():
        row = json.loads(line)["provenance"]["generator_params"]
        if row["dataset"] in ("m4_daily", "m4_weekly"):
            add("External real benchmark", str(benchmark.relative_to(ROOT)), [row["item_id"]], row["dataset"][3:])
    path = "replication/results/weekly_pair.npz"
    with np.load(SUITE / path) as f:
        add("Weekly SAE-pair replication", path, f["ids"], "weekly")
    path = "boundary/fixed.json"
    panel = read(path)["datasets"]
    for key, domain in (("reserve", "daily"), ("m4", "daily"), ("weekly", "weekly")):
        add("Boundary fixed-coordinate study", path, panel[key]["ids"], domain)
    path = "boundary/m4.json"
    for ids in read(path)["ids"].values():
        add("Boundary daily train, validation and test", path, ids, "daily")
    path = "boundary/weekly_seed1.json"
    add("Boundary weekly refit", path, read(path)["test_ids"], "weekly")
    path = "controls/items.json"
    add("External input-response controls", path, [r["item"] for r in read(path)["rows"]])
    path = "transport/results/manifest.json"
    add("Paired dense hidden transport", path, read(path)["items"])
    path = "transport/results/natural_manifest.json"
    for domain in ("m4_daily", "m4_weekly"):
        add("Observed-window dense transport", path, read(path)[domain]["items"])
    path = "transport/results/axes_manifest.json"
    add("Source-axis physical transport", path, read(path)["m4_daily"]["ids"], "daily")
    path = "steering/results/results.json"
    for panel, ids in read(path)["series_ids"].items():
        add("Activation contrast steering", path, ids, "weekly" if panel == "weekly" else "daily")
    path = "steering/amplify/results/manifest.json"
    add("State-dependent amplification", path, read(path)["items"])
    path = "forecast_requests/results/bolt.json"
    for key in ("clean_fit_ids", "source_fit_ids", "calibration_ids", "test_ids"):
        add("Requested forecast waveforms", path, read(path)[key])
    path = "shape_transfer/tangents/results/manifest.json"
    for key in ("ids", "calibration"):
        add("Paired input tangents", path, read(path)[key])
    path = "atoms/results/manifest.json"
    add("Original SAE atom intervention", path, read(path)["split"])
    path = "atoms/matched_windows/results/manifest.json"
    for key in ("ids", "map_train_ids"):
        add("Matched observed windows", path, read(path)[key])
    path = "replication/results/manifest.json"
    add("Independent seed donor selection", path, read(path)["sources"]["daily"]["items"], "daily")
    path = "shape_transfer/fresh_cohort/results/manifest.json"
    add("Additional daily shape transfer", path, read(path)["item_ids"])
    path = "shape_transfer/blind_frequency/results/manifest.json"
    add("Frequency-blind shape transfer", path, read(path)["domains"]["daily"]["items"], "daily")
    path = "shape_transfer/rolling/results/manifest.json"
    add("Rolling-origin shape transfer", path, [r["series"] for r in read(path)["series"]])
    path = "shape_transfer/sparse_stability/results/manifest.json"
    add("Small input noise and sparse stability", path, read(path)["ids"])

    records = []
    by_domain = {}
    for domain, meta in manifest["domains"].items():
        no_prior, shared = [], []
        for item in meta["items"]:
            labels = sorted(name for name, material in prior.items() if item in material[domain])
            records.append(dict(id=domain + "/" + item, prior_studies=labels, absent_from_audited_union=not labels))
            (shared if labels else no_prior).append(item)
        by_domain[domain] = dict(items_without_prior_membership=no_prior, items_with_prior_membership=shared)
    assert [r["id"] for r in records] == manifest["ids"]
    catalog = {}
    for name, values in prior.items():
        catalog[name] = {"path": values["path"]}
        for domain in ("daily", "weekly"):
            overlap = sorted(values[domain].intersection(manifest["domains"][domain]["items"]))
            catalog[name][domain] = {"prior_unique_ids": len(values[domain]), "overlap": overlap}
    report = dict(
        scope="Actual item memberships in the named public experimental panels. A repeated ID can have a different window, intervention or forecast horizon. Exclusion lists and datasets merely cited by other studies are not counted.",
        frozen_order_mask=[r["absent_from_audited_union"] for r in records],
        ids=manifest["ids"], rows=records, domains=by_domain, checked_studies=catalog,
    )
    (OUT / "confirm_overlap.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    for group, items in cross_reference()["domains"].items():
        print(group, len(items["items_without_prior_membership"]), len(items["items_with_prior_membership"]))
