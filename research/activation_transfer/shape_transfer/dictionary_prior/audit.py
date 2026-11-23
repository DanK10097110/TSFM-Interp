"""Recalculate every macro endpoint from individual series and recheck support and norm controls."""
import json

import numpy as np

from research.activation_transfer.shape_transfer.dictionary_prior.run import ARMS, OUT, SELF


def audit():
    summary = json.loads((OUT / "results.json").read_text())
    results = {}
    for key, domains in summary.items():
        donor, receiver = key.split("_to_")
        data = dict(np.load(OUT / (key + ".npz")))
        with np.load(SELF / "results" / f"{donor}_to_{donor}.npz") as old:
            mask = np.isin(old["split"], ["daily", "extra_daily"])
            for field, original in (("ids", "ids"), ("active", "active"), ("measured_source", "native")):
                np.testing.assert_array_equal(data[field], old[original][mask])
        if len(set(data["ids"])) != 320 or data["receiver_strength"].min() <= 0:
            raise ValueError("Expected 320 distinct histories with 32 positive candidates")
        if not np.all(np.diff(data["receiver_support"], axis=1) > 0):
            raise ValueError("Expected sorted unique receiver IDs")
        live = data["active"]
        for arm in ("wrong1", "wrong2"):
            np.testing.assert_allclose(data[arm + "_norm"][live], data["natural_norm"][live], rtol=1e-6)
        for name in ("natural_rank", "attenuated_rank", "level_rank"):
            ordered = np.sort(data[name][live], axis=-1)
            np.testing.assert_array_equal(ordered, np.broadcast_to(np.arange(32), ordered.shape))
        peak = 0.
        checked = 0
        magnitudes = {}
        late_per_donor = {}
        for domain, results_by_domain in domains.items():
            mask = data["split"] == domain
            actual = data["measured_source"][mask].astype(float)
            eligible = np.flatnonzero(live[mask].sum(0) >= 8)
            assert results_by_domain["macro"]["atoms"] == data["atoms"][eligible].tolist()
            late_per_donor[domain] = {"atoms": data["atoms"][eligible].tolist()}
            arms = list(ARMS) + (["known_capped"] if donor == receiver else [])
            early_native = actual[:, :, :8][live[mask]]
            native_norm = np.linalg.norm(early_native - early_native.mean(1, keepdims=True), axis=1)
            nonzero = native_norm > 1e-9
            magnitudes[domain] = {"above_1e_minus_9": int(nonzero.sum())}
            for arm in arms:
                predicted = data[arm + "_effect"][mask].astype(float)
                early_effect = predicted[:, :, :8][live[mask]]
                effect_norm = np.linalg.norm(early_effect - early_effect.mean(1, keepdims=True), axis=1)
                magnitudes[domain][arm] = np.quantile(
                    effect_norm[nonzero] / native_norm[nonzero], [.1, .5, .9],
                ).tolist()
                for name, segment in (("early", slice(8)), ("late", slice(8, 16))):
                    for output in ("full", "shape"):
                        values = []
                        by_atom = []
                        for j in eligible:
                            fired = live[mask, j]
                            want = actual[fired, j, segment]
                            got = predicted[fired, j, segment]
                            if output == "shape":
                                want = want - want.mean(1, keepdims=True)
                                got = got - got.mean(1, keepdims=True)
                            power = np.vdot(want, want)
                            if power > 8e-10 * len(want):
                                values.append(1. - np.vdot(got - want, got - want) / power)
                                by_atom.append(float(values[-1]))
                            else:
                                by_atom.append(None)
                        reference = results_by_domain["macro"][name + "_" + output][arm]["value"]
                        err = abs(np.mean(values) - reference)
                        peak = max(peak, float(err))
                        checked += 1
                        if (name, output) == ("late", "shape"):
                            late_per_donor[domain][arm] = by_atom
            if donor == receiver:
                for name in ("natural", "attenuated", "level"):
                    pos = data[name + "_rank"][mask, :, 0]
                    chosen = np.take_along_axis(data["receiver_support"][mask], pos, axis=1)
                    correct = (chosen == data["atoms"][None, :]) & live[mask]
                    assert int(correct.sum()) == results_by_domain["identification"][name]["top1"]["numerator"]
        if peak > 1e-9:
            raise ValueError(f"Independent score mismatch: {peak}")
        results[key] = dict(physical_writes=int(data["unique_physical_writes"]), macro_endpoints=checked,
                            max_score_discrepancy=peak, early_centered_norm_ratio_p10_p50_p90=magnitudes,
                            late_shape_per_donor=late_per_donor)
    (OUT / "audit.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    audit()
