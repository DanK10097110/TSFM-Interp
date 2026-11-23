"""Recompute central point estimates and discrete switch events from the public result archives."""
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def feature_metrics(predicted, observed, active, centered):
    """Flatten the eight forecast leads within firing items, then weight measurable features equally."""
    cosines, skills = [], []
    for atom, mask in enumerate(active.T):
        if not mask.any():
            continue
        actual = observed[mask, atom].astype(float)
        estimate = predicted[mask, atom].astype(float)
        if centered:
            actual -= actual.mean(axis=1, keepdims=True)
            estimate -= estimate.mean(axis=1, keepdims=True)
        if np.mean(actual**2) <= 1e-10:
            continue
        target_norm = np.linalg.norm(actual)
        pred_norm = np.linalg.norm(estimate)
        dot = np.sum(actual * estimate)
        cosines.append(dot / (target_norm * pred_norm) if pred_norm else 0.)
        skills.append(1 - np.sum((actual - estimate)**2) / target_norm**2)
    return np.mean(cosines), np.mean(skills)


def compare_metrics(actual, published):
    values = [published[key]["value"] for key in ("cosine", "skill")]
    np.testing.assert_allclose(actual, values, rtol=2e-6, atol=2e-6)


def atom_checks(source):
    """Check the primary study and the reserve cohorts for all three dictionaries."""
    root = HERE / "atoms/results"
    report = read_json(root / f"{source}_scores.json")
    with np.load(root / f"{source}_effects.npz") as archive:
        data = dict(archive)
    cal_ids = set(data["ids"][data["split"] == "cal"])
    primary_ids = set(data["ids"])
    assert len(primary_ids) == len(data["ids"])
    np.testing.assert_array_equal(data["active"], data["source_codes"] > 0)
    zero = data["target_match_codes"] == 0
    assert np.all(data["target_coactive"][zero] == 0)
    daily = data["split"] == "daily"
    recipient = "timesfm" if source == "bolt" else "bolt"
    shape_ratios = []
    for model, arm in ((source, "source_native"), (recipient, "target_dense")):
        noise = read_json(HERE / "controls" / f"atom_audit_{model}.json")
        floor = noise["layouts"][source]["populations"]["daily"]["max_pair_noise_rms"]
        wave = data[arm][daily].astype(float)
        centered = wave - wave.mean(axis=-1, keepdims=True)
        energy = np.sum(centered**2, axis=(0, 2))
        n_values = data["active"][daily].sum(axis=0) * centered.shape[-1]
        shape_ratios.append(float(np.min(np.sqrt(energy / n_values)) / floor))
    n_metrics = 0
    for domain in ("daily", "electricity", "cal"):
        rows = data["split"] == domain
        raw = data["source_native"][rows]
        active = data["active"][rows]
        for kind in ("full", "shape"):
            published = report[domain]["source_native"][kind]["summary"]
            for arm in ("target_dense", "source_projected", "target_sham", "target_coactive",
                        "target_omp1", "target_omp4", "target_omp16", *[f"target_wrong{i}" for i in range(4)]):
                metrics = feature_metrics(data[arm][rows], raw, active, kind == "shape")
                compare_metrics(metrics, published[arm])
                n_metrics += len(metrics)
    root = HERE / "replication/results"
    for seed in range(3):
        fit = read_json(root / f"{source}_seed{seed}.json")
        with np.load(root / f"{source}_seed{seed}.npz") as reserve:
            assert primary_ids.intersection(reserve["ids"]) == cal_ids
            np.testing.assert_array_equal(reserve["active"], reserve["source_codes"] > 0)
            for domain in ("daily", "machines"):
                rows = reserve["split"] == domain
                active = reserve["active"][rows]
                keep = active.sum(axis=0) >= 8
                info = fit["domains"][domain]["eight"]
                assert info["n"] == int(keep.sum())
                for kind in ("full", "shape"):
                    for arm in ("dense", "wrong0", "wrong1", "wrong2", "wrong3"):
                        metrics = feature_metrics(reserve[arm][rows][:, keep], reserve["native"][rows][:, keep],
                                                  active[:, keep], kind == "shape")
                        compare_metrics(metrics, info[kind][arm])
                        n_metrics += len(metrics)
    return n_metrics, min(shape_ratios)


def as_sets(ids):
    return [set(map(int, row[row >= 0])) for row in ids.reshape(-1, ids.shape[-1])]


def reencoding_checks(receiver):
    """Recover the discrete events directly from padded per-edit feature IDs."""
    source = "timesfm" if receiver == "bolt" else "bolt"
    folder = HERE / "reencode/results"
    study = read_json(folder / f"{source}_to_{receiver}.json")
    n_counts, n_metrics = 0, 0
    for domain, cohort in study["datasets"].items():
        with np.load(folder / f"{domain}_{receiver}_sufficient.npz") as data:
            for gate in ("hard", "shoulder"):
                for method, summary in cohort["gates"][gate]["methods"].items():
                    for event in ("enter", "leave"):
                        truth = as_sets(data[f"{gate}_true_{event}"])
                        predicted = as_sets(data[f"{gate}_{method}_{event}"])
                        found = sum(len(actual & estimate) for actual, estimate in zip(truth, predicted, strict=True))
                        counts = summary["counts"]
                        assert counts[f"true_{event}"] == sum(map(len, truth))
                        assert counts[f"pred_{event}"] == sum(map(len, predicted))
                        assert counts[f"correct_{event}"] == found
                        n_counts += 3
                    dot, ref, pred, _ = np.moveaxis(data[f"{gate}_{method}_energy"], -1, 0)
                    cosine = dot / np.maximum(np.sqrt(ref * pred), 1e-20)
                    np.testing.assert_allclose(cosine.mean(), summary["cosine"]["value"], atol=1e-12)
                    np.testing.assert_allclose((2 * dot - pred).sum() / ref.sum(), summary["recovery"]["value"])
                    n_metrics += 2
    return n_counts, n_metrics


def coactivation_checks(source):
    """Independently select the strongest training pair and score its natural reserve removals."""
    original = HERE / "atoms/results"
    folder = HERE / "replication/results"
    training = read_json(original / f"{source}_metadata.json")
    report = read_json(folder / f"{source}_coactive.json")
    chosen = int(np.argmax(training["target_match_train_correlation"]))
    assert chosen == report["selected_index"]
    with np.load(folder / f"{source}_coactive.npz") as matched, np.load(folder / f"{source}_seed0.npz") as base:
        np.testing.assert_array_equal(matched["ids"], base["ids"])
        np.testing.assert_array_equal(matched["source_atoms"], training["source_atoms"])
        np.testing.assert_array_equal(matched["target_atoms"], training["target_atom_matches"])
        assert np.all(matched["coactive"][matched["native_target_codes"] == 0] == 0)
        n = 0
        for domain, section in report["domains"].items():
            mask = base["split"] == domain
            active = base["active"][mask]
            reference = base["native"][mask]
            target_firing = matched["native_target_codes"][mask, chosen] > 0
            assert target_firing.sum() == section["selected_target_firings"]
            if target_firing.any():
                target_writes = matched["selected_target_native"][mask, None]
                for kind in ("full", "shape"):
                    reverse = feature_metrics(reference[:, [chosen]], target_writes, target_firing[:, None],
                                              kind == "shape")
                    compare_metrics(reverse, section[kind]["reciprocal_selected"])
                    n += len(reverse)
            for arm, waves in (("coactive", matched["coactive"]), ("mapped", base["dense"])):
                for kind in ("full", "shape"):
                    summary = section[kind][arm]
                    metrics = feature_metrics(waves[mask], reference, active, kind == "shape")
                    compare_metrics(metrics, summary["macro"])
                    n += len(metrics)
                    idx = [chosen]
                    if not active[:, idx].any():
                        assert "selected" not in summary
                        continue
                    selected = feature_metrics(waves[mask][:, idx], reference[:, idx], active[:, idx], kind == "shape")
                    compare_metrics(selected, summary["selected"])
                    n += len(selected)
    weekly = read_json(folder / "weekly_pair.json")
    other = "timesfm" if source == "bolt" else "bolt"
    with np.load(folder / "weekly_pair.npz") as f:
        live = f[source + "_z"] > 0
        assert live.sum() == weekly[source]["firings"]
        assert (live & (f[other + "_z"] > 0)).sum() == weekly["joint"]
        for kind in ("full", "shape"):
            values = feature_metrics(f[other + "_native"][:, None], f[source + "_native"][:, None],
                                     live[:, None], kind == "shape")
            compare_metrics(values, weekly[source][kind])
            n += len(values)
    return n


def main():
    features = 0
    counts = 0
    codes = 0
    shape_ratios = []
    for name in ("bolt", "timesfm"):
        n_features, ratio = atom_checks(name)
        features += n_features
        features += coactivation_checks(name)
        shape_ratios.append(ratio)
        events, points = reencoding_checks(name)
        counts += events
        codes += points
    print(f"Verified {features} atom-transfer metrics, {codes} sparse-code metrics and "
          f"{counts} switch-event counts against the per-series archives.")
    print(f"Smallest daily centered atom RMS / largest zero-dose RMS in its layout: {min(shape_ratios):.3f}.")


if __name__ == "__main__":
    main()
