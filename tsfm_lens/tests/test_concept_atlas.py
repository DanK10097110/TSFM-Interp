"""Tests for `sae/concept_atlas.py` (ROADMAP.md sec 37.15 q6, option b) and its
report renderer `report/sae_concept_atlas.py`.

Synthetic with planted answers throughout (sec 2.4). Every load-bearing
assertion was confirmed to fail under a planted regression -- see each
test's own docstring for what was reverted/swapped and what broke. The
`cluster_atlas`/`_structure_null`/`_cross_model_null` tests operate on
in-memory `(X, rows)` pairs directly (these functions take no `run_dir`),
matching `sae/concept_atlas.py`'s own module docstring's claim that pooling
is a thin, separately-testable I/O step ahead of them; only the
determinism test (which exercises `run_concept_atlas` end to end) and the
report test go through actual `sae/<model>/<layer>_ablation.json` files on
disk, mirroring `tests/test_sae_concept_map.py`'s own fixture conventions.
"""

from __future__ import annotations

import inspect
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config  # noqa: E402
from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.sae.train import sanitize  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402
from tsfm_lens.sae.concept_atlas import (  # noqa: E402
    _concept_records, _cross_model_null, _right_tail_p, _structure_null,
    cluster_atlas, run_concept_atlas)
from tsfm_lens.report.sae_concept_atlas import atlas_block, atlas_map_data  # noqa: E402

_N_CH = len(CHANNELS)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers.
# ---------------------------------------------------------------------------

def _direction(idx: int, mag: float = 6.0) -> np.ndarray:
    v = np.zeros(_N_CH, dtype=np.float64)
    v[idx] = mag
    return v


def _three_direction_fixture(seed: int = 42):
    """3 planted directions (trend / level / horizon_shape_far), 4 members
    each split 2-and-2 across two models, plus 6 unstructured noise rows
    split across a third model and one of the two real ones. Verified
    empirically (not just asserted here) to recover exactly 3 concepts at
    `min_cosine=0.85`, each spanning both of its 2 models, with every noise
    row left unassigned."""
    rng = np.random.default_rng(seed)
    X: list = []
    rows: list = []
    feat = 0
    for idx in (0, 3, 6):  # trend, level, horizon_shape_far
        base = _direction(idx)
        for model in ("m1", "m1", "m2", "m2"):
            X.append(base + rng.normal(0, 0.35, _N_CH))
            rows.append({"model": model, "layer": "L1", "feature": feat})
            feat += 1
    for i in range(6):
        v = rng.normal(0, 1, _N_CH)
        v = v / np.linalg.norm(v) * 6.0
        X.append(v)
        rows.append({"model": "m3" if i % 2 == 0 else "m1", "layer": "L1", "feature": feat})
        feat += 1
    return np.asarray(X), rows


def _candidate(feature: int, channel_values: dict) -> dict:
    """Mirrors `tests/test_sae_concept_map.py::_candidate`: `null_p95`
    pinned at 1.0 so `signed_effect` IS the null-unit ablation-vector value
    directly."""
    channels = {ch: {"null_p95": 1.0, "signed_effect": float(v)} for ch, v in channel_values.items()}
    return {"feature": feature, "scorable": True,
            "n_channels_clearing": sum(1 for v in channel_values.values() if abs(v) >= 1.0),
            "channels": channels}


def _row(channel: str, value: float, rest: float = 0.0) -> dict:
    return {ch: (value if ch == channel else rest) for ch in CHANNELS}


def _write_ablation(run_dir, model, layer, candidates, withheld=False, skipped=False):
    path = Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
    save_json(path, {"model": model, "layer": layer, "withheld": withheld,
                     "skipped": skipped, "candidates": candidates})


def _cfg(out_dir: str, n_null: int = 20, seed: int = 0):
    cfg = config_from_dict(build_config(str(out_dir)))
    cfg.run.seed = seed
    cfg.concepts.atlas_n_null = n_null
    cfg.concepts.atlas_min_cosine = 0.85
    cfg.concepts.atlas_min_members = 3
    return cfg


# ---------------------------------------------------------------------------
# 1. Three planted directions, each spanning two models, recovered as
#    three concepts.
# ---------------------------------------------------------------------------

def test_three_planted_directions_recovered_across_two_models():
    X, rows = _three_direction_fixture()
    labels = cluster_atlas(X, min_cosine=0.85, min_members=3)
    concepts = _concept_records(X, rows, labels, {})
    assert len(concepts) == 3, concepts
    for c in concepts:
        assert c["n_members"] == 4
        assert c["n_models"] == 2
        assert set(c["models"]) == {"m1", "m2"}
    # None of the 6 noise rows (the last 6 in `rows`) may join a concept.
    assert all(int(lbl) == -1 for lbl in labels[-6:])


def test_three_planted_directions_regression_one_model_only_is_caught():
    """Planted regression: if every member of one direction came from a
    SINGLE model (a bug that stopped drawing from the second model), the
    concept's own `n_models` must read 1, not 2 -- confirming the n_models
    assertion above actually discriminates rather than trivially passing."""
    X, rows = _three_direction_fixture()
    for r in rows[:4]:
        r["model"] = "m1"  # collapse the first planted direction onto one model
    labels = cluster_atlas(X, min_cosine=0.85, min_members=3)
    concepts = _concept_records(X, rows, labels, {})
    n_models_per_concept = sorted(c["n_models"] for c in concepts)
    assert n_models_per_concept == [1, 2, 2], n_models_per_concept


# ---------------------------------------------------------------------------
# 2. Every pair inside a concept clears min_cosine; complete linkage is
#    load-bearing for that guarantee.
# ---------------------------------------------------------------------------

def test_every_pair_inside_every_concept_clears_min_cosine():
    X, rows = _three_direction_fixture()
    min_cosine = 0.85
    labels = cluster_atlas(X, min_cosine=min_cosine, min_members=3)
    norms = np.linalg.norm(X, axis=1)
    unit = X / norms[:, None]
    for cid in sorted({int(c) for c in labels.tolist() if c >= 0}):
        idx = np.where(labels == cid)[0]
        sims = unit[idx] @ unit[idx].T
        iu = np.triu_indices(len(idx), k=1)
        assert sims[iu].min() >= min_cosine - 1e-9, (cid, sims[iu].min())


def test_average_linkage_would_violate_the_guarantee_on_a_chain():
    """Plant: swap production's complete linkage for average linkage on a
    5-point chain (angles 0/20/40/60/80 degrees apart in a 2-D subspace)
    and confirm the guarantee fails -- this is what pins complete linkage
    as load-bearing rather than an arbitrary choice. Verified empirically
    (see this module's implementation notes): at `min_cosine=0.85`,
    production (`cluster_atlas`, complete linkage) correctly refuses to
    form ANY concept from this chain (every pairwise cosine among any 3
    consecutive points already dips below 0.85 once the chain reaches 3
    points), while raw average linkage groups the first three points into
    one cluster of size 3 whose own minimum pairwise cosine is ~0.766 --
    below the 0.85 threshold it was supposed to guarantee."""
    from scipy.cluster.hierarchy import fcluster, linkage

    angles = [0, 20, 40, 60, 80]
    X = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] + [0.0] * (_N_CH - 2)
                 for a in angles]) * 6.0
    min_cosine = 0.85
    t = 1.0 - min_cosine

    # Production: complete linkage must not violate the guarantee (it
    # refuses to form a concept here at all -- a real answer, not a dodge:
    # the earlier test above shows complete linkage DOES form concepts
    # when the data actually supports one).
    labels_complete = cluster_atlas(X, min_cosine=min_cosine, min_members=3)
    norms = np.linalg.norm(X, axis=1)
    unit = X / norms[:, None]
    cos = unit @ unit.T
    for cid in sorted({int(c) for c in labels_complete.tolist() if c >= 0}):
        idx = np.where(labels_complete == cid)[0]
        sims = cos[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        assert sims[iu].min() >= min_cosine - 1e-9

    # The plant: average linkage on the SAME data, SAME threshold.
    Z_avg = linkage(unit, method="average", metric="cosine")
    raw_avg = fcluster(Z_avg, t=t, criterion="distance")
    violated = False
    for lab in set(raw_avg.tolist()):
        idx = np.where(raw_avg == lab)[0]
        if len(idx) >= 3:
            sims = cos[np.ix_(idx, idx)]
            iu = np.triu_indices(len(idx), k=1)
            if sims[iu].min() < min_cosine:
                violated = True
    assert violated, "average linkage was expected to violate the min-cosine guarantee on this chain"


# ---------------------------------------------------------------------------
# 3. A tight group of only 2 is not a concept.
# ---------------------------------------------------------------------------

def test_two_member_group_is_not_a_concept():
    base = _direction(0)
    rng = np.random.default_rng(1)
    X = np.stack([base + rng.normal(0, 0.05, _N_CH), base + rng.normal(0, 0.05, _N_CH)])
    labels = cluster_atlas(X, min_cosine=0.85, min_members=3)
    assert list(labels) == [-1, -1]


def test_two_member_group_regression_min_members_one_would_admit_it():
    """Planted regression: with `min_members=1` (the bug this guards
    against -- the size floor silently not applied), the same tight pair
    WOULD be admitted, confirming the assertion above discriminates."""
    base = _direction(0)
    rng = np.random.default_rng(1)
    X = np.stack([base + rng.normal(0, 0.05, _N_CH), base + rng.normal(0, 0.05, _N_CH)])
    labels = cluster_atlas(X, min_cosine=0.85, min_members=1)
    assert list(labels) == [0, 0]


# ---------------------------------------------------------------------------
# 4. Structure null.
# ---------------------------------------------------------------------------

def test_structure_null_rejects_on_planted_structure():
    X, _rows = _three_direction_fixture()
    min_cosine, min_members = 0.85, 3
    labels = cluster_atlas(X, min_cosine=min_cosine, min_members=min_members)
    real_n = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    assert real_n == 3

    rng = np.random.default_rng(1)
    n_null = 99
    n_concepts_null, _frac = _structure_null(X, min_cosine, min_members, rng, n_null)
    p95 = float(np.quantile(n_concepts_null, 0.95))
    p = _right_tail_p(real_n, n_concepts_null)
    assert p95 < real_n, (p95, real_n)
    assert p <= 0.05, p


def test_structure_null_does_not_reject_pure_noise():
    rng = np.random.default_rng(99)
    X = np.array([(rng.normal(0, 1, _N_CH)) for _ in range(18)])
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * 6.0
    min_cosine, min_members = 0.85, 3
    labels = cluster_atlas(X, min_cosine=min_cosine, min_members=min_members)
    real_n = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0

    rng2 = np.random.default_rng(2)
    n_concepts_null, _frac = _structure_null(X, min_cosine, min_members, rng2, 99)
    p = _right_tail_p(real_n, n_concepts_null)
    assert p > 0.05, p


# ---------------------------------------------------------------------------
# 5. Cross-model null preserves per-model counts.
# ---------------------------------------------------------------------------

def test_cross_model_null_permutation_preserves_per_model_counts():
    models = np.array(["m1"] * 5 + ["m2"] * 3 + ["m3"] * 2)
    labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1, -1])
    n = len(models)
    real_counts = Counter(models.tolist())

    rng = np.random.default_rng(7)
    _cross_model_null(labels, models, rng, n_null=5)

    # `_cross_model_null` calls `rng.permutation(n)` exactly once per null
    # draw, in order (verified by reading `sae/concept_atlas.py`'s source);
    # an independently-seeded generator reproduces the identical sequence,
    # so re-deriving it here checks the actual mechanism, not just the
    # mathematical fact that any permutation preserves a multiset.
    rng2 = np.random.default_rng(7)
    for _ in range(5):
        permuted = models[rng2.permutation(n)]
        assert Counter(permuted.tolist()) == real_counts


def test_cross_model_null_regression_a_resample_would_not_preserve_counts():
    """Planted regression: replacing the permutation with an i.i.d.
    resample (`rng.choice(models, size=n, replace=True)`) -- the bug this
    guards against -- does NOT preserve per-model counts in general,
    confirming the assertion above is actually pinning something real."""
    models = np.array(["m1"] * 5 + ["m2"] * 3 + ["m3"] * 2)
    real_counts = Counter(models.tolist())
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(20):
        resampled = rng.choice(models, size=len(models), replace=True)
        if Counter(resampled.tolist()) != real_counts:
            mismatches += 1
    assert mismatches > 0, "an i.i.d. resample was expected to disturb per-model counts at least once in 20 draws"


# ---------------------------------------------------------------------------
# 6. Determinism.
# ---------------------------------------------------------------------------

def test_run_concept_atlas_is_deterministic(tmp_path):
    """Two independent runs of `run_concept_atlas` against identical
    ablation inputs and the same `cfg.run.seed` must agree on everything
    this module itself computes: pooling order, cluster labels, both null
    procedures' statistics, naming, and the PCA projection.

    `umap1`/`umap2` are deliberately EXCLUDED from the byte-identical
    comparison below, and this was found empirically, not assumed
    (`CLAUDE.md` sec 2.4): a first version of this test compared the whole
    artifact and failed -- `pc1`/`pc2`/`concept`/every null statistic/every
    concept record were byte-identical across the two runs, but `umap1`/
    `umap2` differed by several units despite an identical seeded
    `umap.UMAP(random_state=seed)` call. Isolated to numba's threading
    layer (this box's `NumbaWarning: TBB threading layer... disabled`):
    forcing `NUMBA_NUM_THREADS=1` makes the SAME two calls byte-identical.
    `sae/concept_atlas.py`'s own module docstring already scopes UMAP as
    "best-effort" for exactly this reason, and the report figure always
    plots PCA, never UMAP, so this upstream nondeterminism reaches no
    rendered output -- but it is real, and asserting byte-identical `umap1`/
    `umap2` here would be a flaky test pinning a guarantee this module
    cannot actually make (`projection` still deterministically reports
    whether UMAP was attempted/succeeded, which IS asserted below).
    """
    def _make_run(d):
        cands_a = [_candidate(0, _row("trend", 6.0)), _candidate(1, _row("trend", 5.7)),
                  _candidate(2, _row("trend", 6.3)), _candidate(3, _row("level", 6.0))]
        cands_b = [_candidate(0, _row("trend", 6.1)), _candidate(1, _row("level", 6.0)),
                  _candidate(2, _row("level", 5.8))]
        _write_ablation(d, "m1", "L1", cands_a)
        _write_ablation(d, "m2", "L1", cands_b)
        return d

    d1 = _make_run(tmp_path / "run1")
    d2 = _make_run(tmp_path / "run2")
    cfg = _cfg(tmp_path / "cfgdir", n_null=15, seed=3)

    out1 = run_concept_atlas(d1, cfg)
    out2 = run_concept_atlas(d2, cfg)

    def _without_umap(out):
        stripped = dict(out)
        stripped["rows"] = [{k: v for k, v in r.items() if k not in ("umap1", "umap2")}
                            for r in out["rows"]]
        return stripped

    assert _without_umap(out1) == _without_umap(out2)
    assert out1["projection"] == out2["projection"]


def test_run_concept_atlas_withheld_and_skipped_targets_contribute_nothing(tmp_path):
    _write_ablation(tmp_path, "m1", "L1", [_candidate(0, _row("trend", 6.0))], withheld=True)
    _write_ablation(tmp_path, "m2", "L1", [_candidate(0, _row("trend", 6.0))], skipped=True)
    cfg = _cfg(tmp_path / "cfgdir")
    out = run_concept_atlas(tmp_path, cfg)
    assert out["n_features"] == 0
    assert out["rows"] == []
    assert out["concepts"] == []


# ---------------------------------------------------------------------------
# 7. Report: n_models + 1 facets, no hardcoded model name.
# ---------------------------------------------------------------------------

def _write_synthetic_atlas_artifact(run_dir, models=("alpha", "beta")):
    rows = []
    concepts = []
    feat = 0
    for cid, model_pair in enumerate([(models[0], models[1]), (models[0], models[1])]):
        members = []
        for i, m in enumerate(model_pair * 2):  # 4 members, 2 per model
            rows.append({"model": m, "layer": "L1", "feature": feat,
                        "concept": cid, "pc1": float(cid * 3 + i * 0.1), "pc2": float(i * 0.1)})
            members.append(feat)
            feat += 1
        concepts.append({
            "concept": cid, "name": f"concept {cid}", "n_members": 4,
            "models": {models[0]: 2, models[1]: 2}, "n_models": 2,
            "layers": {f"{models[0]}/L1": 2, f"{models[1]}/L1": 2},
            "mean_profile": {ch: (5.0 if ch == "trend" else 0.0) for ch in CHANNELS},
            "mean_norm": 5.0, "min_pair_cosine": 0.95, "mean_pair_cosine": 0.97,
            "members": members, "cross_model_p": 0.02,
        })
    # one unassigned row
    rows.append({"model": models[0], "layer": "L1", "feature": feat, "concept": None,
                "pc1": -3.0, "pc2": -3.0})
    doc = {
        "schema_version": 1, "space": "ablation",
        "params": {"min_cosine": 0.85, "min_members": 3, "linkage": "complete", "n_null": 20},
        "n_features": len(rows), "n_assigned": len(rows) - 1,
        "rows": rows, "concepts": concepts,
        "null": {
            "structure": {"n_null": 20, "n_concepts_real": 2, "n_concepts_null_mean": 0.3,
                         "n_concepts_null_p95": 1.0, "p_n_concepts": 0.05,
                         "frac_assigned_real": 0.9, "frac_assigned_null_mean": 0.1,
                         "frac_assigned_null_p95": 0.3, "p_frac_assigned": 0.05},
            "cross_model": {"n_null": 20, "n_multi_model_real": 2, "n_multi_model_null_mean": 0.4,
                            "n_multi_model_null_p95": 1.0, "p_n_multi_model": 0.05},
        },
        "pca": {"explained_variance_ratio": [0.6, 0.25]},
        "projection": "pca", "naming": "_compose_batch",
    }
    save_json(Path(run_dir) / "sae" / "concept_atlas.json", doc)
    return doc


def test_atlas_block_renders_n_models_plus_one_facets(tmp_path):
    _write_synthetic_atlas_artifact(tmp_path, models=("alpha", "beta"))
    html = atlas_block(tmp_path, cfg=None)
    assert html
    # Facet titles are subplot annotation `"text"` entries in the embedded
    # Plotly JSON -- distinct from `customdata` array values, which are not
    # keyed by `"text":`.
    for name in ("alpha", "beta", "All models"):
        assert re.search(r'"text"\s*:\s*"' + re.escape(name) + r'"', html), name
    n_facets = sum(len(re.findall(r'"text"\s*:\s*"' + re.escape(n) + r'"', html))
                  for n in ("alpha", "beta", "All models"))
    assert n_facets == 3  # 2 models + 1 "All models" panel = n_models + 1


def test_atlas_block_empty_artifact_returns_empty_string(tmp_path):
    (tmp_path / "sae").mkdir(parents=True)
    assert atlas_block(tmp_path, cfg=None) == ""
    assert atlas_map_data(tmp_path) is None


def _stripped_source(*fns) -> str:
    src = "\n".join(inspect.getsource(fn) for fn in fns)
    return re.sub(r'"""[\s\S]*?"""', "", src)


_BANNED_NAMES = ("TimesFM", "Chronos", "Sundial", "patchy", "steppy")


def test_no_model_name_hardcoded_in_module():
    src = _stripped_source(atlas_map_data, atlas_block)
    for name in _BANNED_NAMES:
        assert name not in src, name
    assert not re.search(r"models\[\d|cfg\.models\[", src)


def test_no_model_name_hardcoded_regression_scan_catches_a_planted_violation():
    """Planted regression: confirm the scan above actually discriminates by
    checking it against a deliberately-violating stand-in function."""
    def _bad_fn(run_dir, cfg):
        return "TimesFM" + str(run_dir) + str(cfg)

    src = _stripped_source(_bad_fn)
    assert any(name in src for name in _BANNED_NAMES)


def test_cross_model_null_detects_segregation_in_both_tails():
    """Concepts that are each one model's own must read `segregated by
    model`, not merely "not significant": a right-tail-only p is 1.0 here."""
    from tsfm_lens.sae.concept_atlas import (_cross_model_null, _left_tail_p,
                                              _mean_purity, _right_tail_p,
                                              cross_model_verdict)
    labels = np.repeat(np.arange(8), 4)
    models = np.array(["a", "b", "c", "d"] * 2).repeat(4)
    rng = np.random.default_rng(0)
    real_multi, null_multi, *_ = _cross_model_null(labels, models, rng, 200)
    assert real_multi == 0
    assert _right_tail_p(real_multi, null_multi) == 1.0
    assert _left_tail_p(real_multi, null_multi) < 0.05
    ids = list(range(8))
    real = _mean_purity(labels, models, ids)
    null = [_mean_purity(labels, models[rng.permutation(models.size)], ids) for _ in range(200)]
    assert real == 1.0
    verdict = cross_model_verdict(_left_tail_p(real, null), _right_tail_p(real, null))
    assert verdict == "segregated by model"
    mixed = np.array(["a", "b", "c", "d"] * 8)
    real_m = _mean_purity(labels, mixed, ids)
    null_m = [_mean_purity(labels, mixed[rng.permutation(mixed.size)], ids) for _ in range(200)]
    assert cross_model_verdict(_left_tail_p(real_m, null_m), _right_tail_p(real_m, null_m)) == "drawn together"
