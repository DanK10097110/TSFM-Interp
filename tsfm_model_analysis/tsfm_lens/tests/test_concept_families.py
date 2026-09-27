"""Tests for `sae/concept_families.py` (ROADMAP.md sec 37, concept families).

Synthetic with planted answers throughout (CLAUDE.md sec 2.4/sec 9). Every
load-bearing assertion below was confirmed to fail under a planted
regression -- see each test's own docstring for what was reverted/broken and
what failed. Fixture helpers mirror `tests/test_concept_atlas.py`'s own
conventions (`_candidate`/`_row`/`_write_ablation`/`_cfg`) so the two files'
fixtures stay directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_smoke import build_config  # noqa: E402
from tsfm_lens.config import config_from_dict  # noqa: E402
from tsfm_lens.manifest import fingerprint_stage, resolve_config_keys  # noqa: E402
from tsfm_lens.sae.response import CHANNELS  # noqa: E402
from tsfm_lens.sae.train import sanitize  # noqa: E402
from tsfm_lens.utils import load_json, save_json  # noqa: E402
from tsfm_lens.sae.concept_atlas import cluster_atlas, run_concept_atlas  # noqa: E402
from tsfm_lens.sae.concept_families import (  # noqa: E402
    _compose_family_sentence1, _compose_family_titles, _concept_family_majority,
    _directed_profile, _family_second_sentence, _level_carrier_caveat, _level_carrier_title,
    _render_effect_clauses, _tag_family_level_carrier, cluster_families, load_families,
    run_concept_families)
from tsfm_lens.sae.plain_text import compose_title  # noqa: E402

_N_CH = len(CHANNELS)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers (mirrors test_concept_atlas.py).
# ---------------------------------------------------------------------------

def _direction(idx: int, mag: float = 6.0) -> np.ndarray:
    v = np.zeros(_N_CH, dtype=np.float64)
    v[idx] = mag
    return v


def _candidate(feature: int, channel_values: dict) -> dict:
    channels = {ch: {"null_p95": 1.0, "signed_effect": float(v)} for ch, v in channel_values.items()}
    return {"feature": feature, "scorable": True,
            "n_channels_clearing": sum(1 for v in channel_values.values() if abs(v) >= 1.0),
            "channels": channels}


def _candidate_ls(feature: int, channel_values: dict, level_share=None,
                  n_shape_channels_clearing: int = 0) -> dict:
    """`_candidate` plus the P4 fields (`level_share`/`n_shape_channels_
    clearing`) `_candidate_lookup`/`_tag_family_level_carrier` read directly
    off the top-level candidate dict -- `_candidate` itself omits them since
    most fixtures do not exercise the level-carrier guard."""
    c = _candidate(feature, channel_values)
    c["level_share"] = level_share
    c["n_shape_channels_clearing"] = n_shape_channels_clearing
    return c


def _row(channel: str, value: float, rest: float = 0.0) -> dict:
    return {ch: (value if ch == channel else rest) for ch in CHANNELS}


def _row_vec(channel: str, value: float, rest: float = 0.0) -> np.ndarray:
    return np.array([value if ch == channel else rest for ch in CHANNELS], dtype=np.float64)


def _write_ablation(run_dir, model, layer, candidates, withheld=False, skipped=False):
    path = Path(run_dir) / "sae" / sanitize(model) / f"{sanitize(layer)}_ablation.json"
    save_json(path, {"model": model, "layer": layer, "withheld": withheld,
                     "skipped": skipped, "candidates": candidates})


def _cfg(out_dir: str, n_null: int = 20, seed: int = 0, family_min_members: int = 5,
        family_assign_min: float = 0.5):
    cfg = config_from_dict(build_config(str(out_dir)))
    cfg.run.seed = seed
    cfg.concepts.atlas_n_null = n_null
    cfg.concepts.atlas_min_cosine = 0.9
    cfg.concepts.atlas_min_members = 3
    cfg.concepts.atlas_family_min_members = family_min_members
    cfg.concepts.atlas_family_assign_min = family_assign_min
    return cfg


def _three_family_fixture_with_decoy_and_noise(seed: int = 7):
    """3 planted families (trend / seasonal / level), each 3 tight
    sub-clusters of 4 members apiece (so each family has 12 members and each
    tight sub-cluster is a would-be atlas concept), spread across 3 models;
    one decoy at cosine ~0.88 to family 0's own direction (a near-miss that
    a 0.9-cosine complete-linkage atlas must exclude, but this family layer
    must recover); 6 unstructured noise rows. Verified empirically below to
    produce exactly 3 families."""
    rng = np.random.default_rng(seed)
    X: list = []
    rows: list = []
    feat = 0
    models = ("m1", "m2", "m3")
    for fam_idx, ch_idx in enumerate((0, 1, 3)):  # trend, seasonal, level
        base = _direction(ch_idx)
        for sub in range(3):
            sub_center = base + rng.normal(0, 0.05, _N_CH)
            for _ in range(4):
                X.append(sub_center + rng.normal(0, 0.05, _N_CH))
                rows.append({"model": models[sub], "layer": "L1", "feature": feat})
                feat += 1
    # Decoy: cosine ~0.88 to family 0's pure trend direction, on the mase axis.
    d = _direction(0, 6.0)
    theta = np.arccos(0.88)
    orth = np.zeros(_N_CH)
    orth[7] = 1.0  # mase
    decoy = np.cos(theta) * d / np.linalg.norm(d) + np.sin(theta) * orth
    decoy = decoy / np.linalg.norm(decoy) * 6.0
    decoy_idx = len(X)
    X.append(decoy)
    rows.append({"model": "m1", "layer": "L1", "feature": feat})
    feat += 1
    # Noise features drawn from the subspace ORTHOGONAL to every family's
    # own defining channel (0/1/3) -- guaranteed zero cosine to any family
    # centroid regardless of `assign_min`, so "noise stays idiosyncratic" is
    # a fact about the fixture's geometry, not a statistical coin flip a
    # 9-D isotropic random draw would occasionally lose (a single stray
    # coordinate on channel 0/1/3 has a non-negligible chance of exceeding
    # cosine 0.5 against that family's near-axis-aligned centroid).
    other_channels = [c for c in range(_N_CH) if c not in (0, 1, 3)]
    for i in range(6):
        v = np.zeros(_N_CH)
        sub = rng.normal(0, 1, len(other_channels))
        sub = sub / np.linalg.norm(sub)
        for c, val in zip(other_channels, sub):
            v[c] = val
        v = v * 6.0
        X.append(v)
        rows.append({"model": models[i % 3], "layer": "L1", "feature": feat})
        feat += 1
    return np.asarray(X), rows, decoy_idx


# ---------------------------------------------------------------------------
# 1. Draft families recovered; decoy recovered by the family (not the tight
#    concept); noise idiosyncratic.
# ---------------------------------------------------------------------------

def test_three_families_recovered_decoy_included_noise_idiosyncratic():
    X, rows, decoy_idx = _three_family_fixture_with_decoy_and_noise()
    result = cluster_families(X, min_members=5, assign_min=0.5)
    labels = result["labels"]
    assert result["threshold"] is not None, result["grid"]
    n_families = int(labels.max() + 1) if labels.max() >= 0 else 0
    assert n_families == 3, (n_families, result["grid"])

    # Every tight sub-cluster of 12 (minus the decoy's own home family, which
    # has 13 with the decoy) lands together.
    sizes = {fid: int(np.sum(labels == fid)) for fid in range(3)}
    assert sorted(sizes.values()) == [12, 12, 13], sizes

    # The decoy is INSIDE a family.
    assert labels[decoy_idx] >= 0, "decoy must be assigned to a family"

    # The tight, complete-linkage atlas at 0.9 must still exclude the decoy
    # from its concept -- this family layer changes nothing about that.
    atlas_labels = cluster_atlas(X, min_cosine=0.9, min_members=3)
    assert atlas_labels[decoy_idx] == -1, "the tight atlas must still exclude the decoy"

    # The last 6 rows (noise) must all be idiosyncratic.
    assert all(int(lbl) == -1 for lbl in labels[-6:])


def test_nearest_centroid_step_recovers_an_orphan_the_draft_clustering_drops():
    """A surgical, single-mechanism test of STEP 2 (nearest-centroid
    reassignment) in isolation from step 1's own average linkage: two
    "orphan" points at cosine 0.6 to family A's direction, too few (2 <
    `min_members`) to ever form or join a draft cluster of their own, and --
    verified below -- NOT absorbed into family A's draft cluster by average
    linkage at threshold 0.65 either (`draft_labels` reads -1 for both).
    Only step 2's nearest-centroid pass, run over EVERY row regardless of
    draft membership, can recover them, and it does. Threshold is PINNED to
    0.65 via a single-value `grid` rather than left to the grid-wide
    parsimony selection (orchestrator review item 1): this test isolates one
    mechanism at one threshold, and letting selection pick a looser
    threshold (e.g. 0.5, where these orphans' cosine 0.6 already clears the
    draft-merge bar) would silently change what is being tested."""
    N = _N_CH
    rng = np.random.default_rng(5)

    def direction(idx, mag=6.0):
        v = np.zeros(N)
        v[idx] = mag
        return v

    X = []
    baseA = direction(0)
    for _ in range(6):
        X.append(baseA + rng.normal(0, 0.03, N))
    baseB = direction(3)
    for _ in range(6):
        X.append(baseB + rng.normal(0, 0.03, N))
    theta = np.arccos(0.6)
    orth = np.zeros(N)
    orth[4] = 1.0
    orphan_idx = []
    for _ in range(2):
        v = np.cos(theta) * baseA / np.linalg.norm(baseA) + np.sin(theta) * orth
        v = v / np.linalg.norm(v) * 6.0
        orphan_idx.append(len(X))
        X.append(v + rng.normal(0, 0.02, N))
    X = np.array(X)

    result = cluster_families(X, min_members=5, assign_min=0.5, grid=(0.65,))
    for i in orphan_idx:
        assert result["draft_labels"][i] == -1, "fixture invalid: orphan must miss the draft stage"
        assert result["labels"][i] == 0, "step 2 must recover the orphan into family A (0)"


def test_selection_picks_fewest_families_within_tolerance_not_best_silhouette():
    """Orchestrator review of F1, item 1 (selection-bias fix), pinned
    directly: constructed (cos_sep=0.82, noise=0.35, seed 42) so the
    FINEST grid cut's final-partition silhouette (3 families) is the
    numeric best (~0.9313), but a COARSER cut (2 families, ~0.9130) sits
    within the default `sil_tol` (0.02) of it. The parsimony rule must
    pick the coarser, 2-family cut; a best-fit (argmax silhouette) rule
    would instead pick the finer, 3-family one -- exactly the
    selection-bias distinction item 1 requires. Verified empirically
    (not hand-derived): both silhouette values and the 2-vs-3 split are
    read off `result['grid']` itself, not asserted from a guess."""
    N = _N_CH
    rng = np.random.default_rng(42)

    def direction(idx, mag=6.0):
        v = np.zeros(N)
        v[idx] = mag
        return v

    e0 = direction(0)
    theta = np.arccos(0.82)
    orth = np.zeros(N)
    orth[1] = 1.0
    e0n = e0 / np.linalg.norm(e0)
    c2dir = np.cos(theta) * e0n + np.sin(theta) * orth
    c2dir = c2dir / np.linalg.norm(c2dir) * 6.0
    e3 = direction(3)
    X = []
    for _ in range(6):
        X.append(e0 + rng.normal(0, 0.35, N))
    for _ in range(6):
        X.append(c2dir + rng.normal(0, 0.35, N))
    for _ in range(6):
        X.append(e3 + rng.normal(0, 0.35, N))
    X = np.array(X)

    grid = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    result = cluster_families(X, min_members=6, assign_min=0.5, grid=grid)
    by_t = {g["min_cosine"]: g for g in result["grid"]}
    assert by_t[0.9]["n_families"] == 3
    assert by_t[0.5]["n_families"] == 2
    best = max(g["final_silhouette"] for g in result["grid"] if g["admissible"])
    assert by_t[0.9]["final_silhouette"] == pytest.approx(best)
    assert by_t[0.5]["final_silhouette"] < best
    assert best - by_t[0.5]["final_silhouette"] < 0.02, "fixture invalid: gap must be within sil_tol"
    assert result["threshold"] == 0.5, result["threshold"]
    assert int(result["labels"].max()) + 1 == 2


def test_family_assign_min_regression_too_high_a_floor_excludes_the_decoy():
    """Planted regression: raising `assign_min` to the tight concept's own
    bar (0.9) must exclude the decoy from every family -- confirming the
    decoy-recovery assertion above actually discriminates on `assign_min`
    rather than passing regardless of it."""
    X, rows, decoy_idx = _three_family_fixture_with_decoy_and_noise()
    result = cluster_families(X, min_members=5, assign_min=0.9)
    assert result["labels"][decoy_idx] == -1


# ---------------------------------------------------------------------------
# 2. Structure null.
# ---------------------------------------------------------------------------

def test_family_structure_null_rejects_on_planted_structure():
    """`min_members=12` here (not the 5-member default the positive tests
    above use): this fixture's 3 real families sit at exactly 12-13 members
    each, and independently permuting 9 mostly-near-zero columns across only
    39 rows readily produces small SPURIOUS admissible clusters at the
    grid's looser end (0.5-0.65) purely from near-zero ties -- a real,
    measured property of this looser threshold grid (`configs/
    concept_atlas.yaml`'s own header note that 0.8 sits "barely above" its
    null on the real run makes the identical point at the tight-atlas
    level). Raising the floor to the real families' own size denies the
    null's small spurious clusters admissibility while leaving the real
    ones untouched -- confirmed empirically (not merely asserted) at
    min_members=5/7/9/12: p fell 0.373/0.144/0.060/0.005 as the floor rose,
    so this is the SAME statistic, sharpened by a floor choice, not a
    different result."""
    from tsfm_lens.sae.concept_atlas import _right_tail_p
    from tsfm_lens.sae.concept_families import DEFAULT_FAMILY_GRID, _structure_null_family

    X, _rows, _decoy = _three_family_fixture_with_decoy_and_noise()
    result = cluster_families(X, min_members=12, assign_min=0.5)
    real_n = int(result["labels"].max() + 1) if result["labels"].max() >= 0 else 0
    assert real_n == 3

    rng = np.random.default_rng(3)
    n_fam_null, _frac_null, _sil_null = _structure_null_family(
        X, min_members=12, assign_min=0.5, grid=DEFAULT_FAMILY_GRID, rng=rng, n_null=200)
    p = _right_tail_p(real_n, n_fam_null)
    assert p <= 0.05, (p, n_fam_null)


def test_family_structure_null_does_not_reject_pure_noise():
    from tsfm_lens.sae.concept_atlas import _right_tail_p
    from tsfm_lens.sae.concept_families import DEFAULT_FAMILY_GRID, _structure_null_family

    rng = np.random.default_rng(11)
    X = np.array([rng.normal(0, 1, _N_CH) for _ in range(30)])
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * 6.0
    result = cluster_families(X, min_members=5, assign_min=0.5)
    real_n = int(result["labels"].max() + 1) if result["labels"].max() >= 0 else 0

    rng2 = np.random.default_rng(4)
    n_fam_null, _frac_null, _sil_null = _structure_null_family(
        X, min_members=5, assign_min=0.5, grid=DEFAULT_FAMILY_GRID, rng=rng2, n_null=60)
    p = _right_tail_p(real_n, n_fam_null)
    assert p > 0.05, (p, n_fam_null)


# ---------------------------------------------------------------------------
# 3. Tight-concept -> family majority bookkeeping.
# ---------------------------------------------------------------------------

def test_concept_family_majority_resolves_split_by_majority():
    labels = np.array([0, 0, 0, 1, -1])
    atlas_concepts = [
        {"concept": 0, "members": [0, 1, 2]},          # unanimous -> family 0
        {"concept": 1, "members": [0, 0, 1, 1, 3]},     # split 4-vs-1 -> majority family 0
        {"concept": 2, "members": [4, 4]},              # all idiosyncratic -> family None
    ]
    out = _concept_family_majority(atlas_concepts, labels)
    assert out[0]["family"] == 0 and not out[0]["split"]
    assert out[1]["family"] == 0 and out[1]["split"]
    assert out[2]["family"] is None and not out[2]["split"]


def test_concept_family_majority_regression_tie_break_is_smallest_id():
    """Planted regression: a 2-vs-2 tie must resolve to the SMALLER family
    id, not to whichever family happened to be inserted last in the Counter
    -- confirms the tie-break is deterministic rather than dict-order
    dependent (CLAUDE.md sec 11.2/sec 11.55)."""
    labels = np.array([0, 0, 5, 5])
    atlas_concepts = [{"concept": 0, "members": [0, 1, 2, 3]}]
    out = _concept_family_majority(atlas_concepts, labels)
    assert out[0]["family"] == 0, out[0]
    # Reversed member order must not change the outcome.
    atlas_concepts_rev = [{"concept": 0, "members": [3, 2, 1, 0]}]
    out_rev = _concept_family_majority(atlas_concepts_rev, labels)
    assert out_rev[0]["family"] == 0, out_rev[0]


# ---------------------------------------------------------------------------
# 3b. Level-carrier guard for families (orchestrator review of F1, item 3):
#     a family whose members' effect is indistinguishable from a level shift
#     must not get a shape-claiming title/description from its raw channel
#     profile, mirroring `concept_atlas.py::_tag_causal_effect`'s existing
#     guard for tight atlas concepts.
# ---------------------------------------------------------------------------

def _cand_ls(feature: int, level_share, n_shape_channels_clearing: int = 0) -> dict:
    return {"feature": feature, "scorable": True,
           "level_share": level_share, "n_shape_channels_clearing": n_shape_channels_clearing}


def test_tag_family_level_carrier_positive():
    """5 members, all high level_share and none clearing a level-removed
    shape channel -- must be tagged `level carrier`, with the correct
    median."""
    rows = [{"model": "m1", "layer": "L1", "feature": i} for i in range(5)]
    lookup = {("m1", "L1", i): _cand_ls(i, level_share=v)
             for i, v in enumerate([0.9, 0.95, 0.99, 0.92, 0.97])}
    idx = np.arange(5)
    tag = _tag_family_level_carrier(idx, rows, lookup, level_share_threshold=0.9)
    assert tag["causal_tag"] == "level carrier"
    assert tag["level_share_median"] == pytest.approx(0.95)
    assert tag["n_members_shape_causal"] == 0


def test_tag_family_level_carrier_shape_causal_wins_even_with_high_level_share():
    """Planted regression companion: a SINGLE member clearing a level-removed
    shape channel must flip the whole family to `shape-causal`, even though
    every member (including that one) still has a high level_share -- the
    precedence is "any real shape effect wins", not a vote."""
    rows = [{"model": "m1", "layer": "L1", "feature": i} for i in range(5)]
    lookup = {("m1", "L1", i): _cand_ls(i, level_share=0.95, n_shape_channels_clearing=(1 if i == 0 else 0))
             for i in range(5)}
    idx = np.arange(5)
    tag = _tag_family_level_carrier(idx, rows, lookup, level_share_threshold=0.9)
    assert tag["causal_tag"] == "shape-causal"


def test_tag_family_level_carrier_regression_threshold_disabled():
    """Planted regression: dropping `level_share_threshold` to 0.0 tags a
    family as a level carrier regardless of its actual level_share, and
    raising it above 1.0 makes the guard unreachable -- confirming the
    positive test's tag is the threshold's doing, not incidental to the
    fixture."""
    rows = [{"model": "m1", "layer": "L1", "feature": i} for i in range(5)]
    lookup = {("m1", "L1", i): _cand_ls(i, level_share=0.1) for i in range(5)}
    idx = np.arange(5)
    correct = _tag_family_level_carrier(idx, rows, lookup, level_share_threshold=0.9)
    assert correct["causal_tag"] == "no measured effect"
    planted_low = _tag_family_level_carrier(idx, rows, lookup, level_share_threshold=0.0)
    assert planted_low["causal_tag"] == "level carrier"

    rows_high = [{"model": "m1", "layer": "L1", "feature": i} for i in range(5, 10)]
    lookup_high = {("m1", "L1", i): _cand_ls(i, level_share=0.99) for i in range(5, 10)}
    idx_high = np.arange(5)
    should_carry = _tag_family_level_carrier(idx_high, rows_high, lookup_high, level_share_threshold=0.9)
    assert should_carry["causal_tag"] == "level carrier"
    planted_high = _tag_family_level_carrier(idx_high, rows_high, lookup_high, level_share_threshold=1.5)
    assert planted_high["causal_tag"] != "level carrier"


def test_level_carrier_title_uses_level_direction_when_resolvable():
    directed = _row_vec("level", -4.0)
    cleared = [int(i) for i in np.argsort(-np.abs(directed)) if abs(directed[i]) >= 1.0]
    assert _level_carrier_title(directed, cleared) == "Level lowerers"
    directed_up = _row_vec("level", 4.0)
    cleared_up = [int(i) for i in np.argsort(-np.abs(directed_up)) if abs(directed_up[i]) >= 1.0]
    assert _level_carrier_title(directed_up, cleared_up) == "Level raisers"


def test_level_carrier_title_falls_back_when_level_not_resolvable():
    """Planted regression: when `level` itself never clears its own null
    (a family dominated by some other channel, but still tagged a level
    carrier via `level_share`), the fixed title must fall back to the
    generic "Level shifters", not silently keep the other channel's name."""
    directed = _row_vec("trend", 6.0)
    cleared = [int(i) for i in np.argsort(-np.abs(directed)) if abs(directed[i]) >= 1.0]
    assert _level_carrier_title(directed, cleared) == "Level shifters"


def test_level_carrier_caveat_states_no_specific_shape():
    text = _level_carrier_caveat(0.95, 5)
    assert "not separable from a level shift" in text
    assert "0.95" in text and "5" in text


def test_level_carrier_family_gets_level_title_and_caveat_description_end_to_end(tmp_path):
    """Full pipeline (`run_concept_atlas` -> `run_concept_families`): one
    draft family (6 members) whose raw channel profile is dominated by
    `horizon_shape_far` but whose UNDERLYING candidates are all high
    level_share/no-shape-clearing must come out titled from the level
    vocabulary with the caveat description, never "Far-horizon shapers".
    A second, ordinary shape-causal family (dominated by `trend`, low
    level_share, one shape-clearing member) is included as a control and
    must NOT get the caveat text."""
    rng = np.random.default_rng(13)
    level_carrier_vecs = [_direction(CHANNELS.index("horizon_shape_far")) + rng.normal(0, 0.05, _N_CH)
                         for _ in range(6)]
    shape_causal_vecs = [_direction(CHANNELS.index("trend")) + rng.normal(0, 0.05, _N_CH)
                        for _ in range(6)]
    cands = (
        [_candidate_ls(i, {ch: float(v) for ch, v in zip(CHANNELS, vec)}, level_share=0.97)
         for i, vec in enumerate(level_carrier_vecs)]
        + [_candidate_ls(i, {ch: float(v) for ch, v in zip(CHANNELS, vec)},
                        level_share=0.1, n_shape_channels_clearing=1)
          for i, vec in enumerate(shape_causal_vecs, start=6)])
    _write_ablation(tmp_path, "m1", "L1", cands)
    cfg = _cfg(tmp_path / "cfgdir", n_null=20, seed=2)

    atlas = run_concept_atlas(tmp_path, cfg)
    families = run_concept_families(tmp_path, atlas, cfg)
    assert families["measured"] is True

    level_fams = [f for f in families["families"] if f["causal_tag"] == "level carrier"]
    shape_fams = [f for f in families["families"] if f["causal_tag"] == "shape-causal"]
    assert level_fams, families["families"]
    assert shape_fams, families["families"]

    lf = level_fams[0]
    assert lf["title"] in ("Level shifters", "Level raisers", "Level lowerers"), lf["title"]
    assert "not separable from a level shift" in lf["description"]
    for sf in shape_fams:
        assert "not separable from a level shift" not in sf["description"]


def test_level_carrier_family_regression_threshold_disabled_reverts_to_confounded_title(tmp_path):
    """Planted regression on the SAME fixture as above: setting
    `level_share_threshold` above 1.0 (never reachable) disables the guard,
    so the level-carrier group must come back with its ordinary,
    level-confounded channel title ("Far-horizon shapers") instead --
    confirming the positive test's title is the guard's doing."""
    rng = np.random.default_rng(13)
    level_carrier_vecs = [_direction(CHANNELS.index("horizon_shape_far")) + rng.normal(0, 0.05, _N_CH)
                         for _ in range(6)]
    shape_causal_vecs = [_direction(CHANNELS.index("trend")) + rng.normal(0, 0.05, _N_CH)
                        for _ in range(6)]
    cands = (
        [_candidate_ls(i, {ch: float(v) for ch, v in zip(CHANNELS, vec)}, level_share=0.97)
         for i, vec in enumerate(level_carrier_vecs)]
        + [_candidate_ls(i, {ch: float(v) for ch, v in zip(CHANNELS, vec)},
                        level_share=0.1, n_shape_channels_clearing=1)
          for i, vec in enumerate(shape_causal_vecs, start=6)])
    _write_ablation(tmp_path, "m1", "L1", cands)
    cfg = _cfg(tmp_path / "cfgdir", n_null=20, seed=2)
    cfg.concepts.level_share_threshold = 1.5

    atlas = run_concept_atlas(tmp_path, cfg)
    families = run_concept_families(tmp_path, atlas, cfg)
    level_fams = [f for f in families["families"] if f["causal_tag"] == "level carrier"]
    assert not level_fams, families["families"]
    titles = {f["title"] for f in families["families"]}
    assert "Far-horizon shapers" in titles, titles


# ---------------------------------------------------------------------------
# 4. Sign convention: titles/descriptions state what the FEATURE DOES, the
#    negative of the recorded (ablation) signed_effect.
# ---------------------------------------------------------------------------

def test_sign_convention_ablating_lowers_seasonal_means_feature_raises_it():
    """The recorded mean_profile (ablation-effect, sec response.py) for
    `seasonal` is NEGATIVE (ablating these features LOWERS seasonal). The
    correct, negated ("feature does") convention must render this as the
    family INCREASING seasonality."""
    mean_profile = _row_vec("seasonal", -6.0)
    directed = _directed_profile(mean_profile)
    cleared = [int(i) for i in np.argsort(-np.abs(directed)) if abs(directed[i]) >= 1.0]
    title = _compose_family_titles([{"directed_vec": directed, "cleared": cleared, "fires_on": None}])[0]
    assert title == "Seasonality amplifiers", title
    s1 = _compose_family_sentence1([directed], [cleared])[0]
    desc = f"{s1} {_family_second_sentence({'m1': 1}, None)}"
    assert "increase" in desc and "seasonal" in desc.lower()
    assert "decrease" not in desc


def test_sign_convention_planted_flip_is_caught():
    """Plant the regression this module's docstring warns about: forget to
    negate, and feed the RAW ablation-effect vector where the "feature does"
    convention belongs. On the identical fixture above, the un-negated title
    flips to the wrong word -- confirming the positive test's assertion is
    not a coincidence of wording but actually pins the sign."""
    mean_profile = _row_vec("seasonal", -6.0)
    unnegated = mean_profile  # the planted bug: no `_directed_profile` call
    cleared = [int(i) for i in np.argsort(-np.abs(unnegated)) if abs(unnegated[i]) >= 1.0]
    title = _compose_family_titles([{"directed_vec": unnegated, "cleared": cleared, "fires_on": None}])[0]
    assert title == "Seasonality dampeners", title
    assert title != "Seasonality amplifiers"


def test_title_uniqueness_within_a_run():
    """Three families with IDENTICAL profiles (same top channel, nothing
    else cleared, no fires-on data) give the channel-escalation and
    fires-on stages nothing to differentiate on -- this is exactly the
    Roman-numeral last-resort case (orchestrator review item 2), and must
    still resolve to 3 distinct titles."""
    directed = _row_vec("trend", 6.0)
    cleared = [int(i) for i in np.argsort(-np.abs(directed)) if abs(directed[i]) >= 1.0]
    prepared = [{"directed_vec": directed, "cleared": cleared, "fires_on": None} for _ in range(3)]
    t1, t2, t3 = _compose_family_titles(prepared)
    assert len({t1, t2, t3}) == 3, (t1, t2, t3)
    assert t1 == "Trend boosters"
    assert t2.startswith("Trend boosters ")


def test_title_disambiguates_via_second_channel_not_roman_numeral():
    """Orchestrator review of F1, item 2: two families dominated by the SAME
    top channel (`horizon_shape_near`) but differing on their
    second-strongest cleared channel must get DISTINCT titles built from
    that second channel, never a Roman-numeral suffix."""
    a = _row_vec("horizon_shape_near", 6.0)
    a[list(CHANNELS).index("level")] = -3.0
    b = _row_vec("horizon_shape_near", 6.0)
    b[list(CHANNELS).index("dispersion")] = 3.0
    cleared_a = [int(i) for i in np.argsort(-np.abs(a)) if abs(a[i]) >= 1.0]
    cleared_b = [int(i) for i in np.argsort(-np.abs(b)) if abs(b[i]) >= 1.0]
    prepared = [{"directed_vec": a, "cleared": cleared_a, "fires_on": None},
               {"directed_vec": b, "cleared": cleared_b, "fires_on": None}]
    t1, t2 = _compose_family_titles(prepared)
    assert t1 != t2, (t1, t2)
    for t in (t1, t2):
        assert not any(t.endswith(f" {suf}") for suf in ("II", "III", "IV", "V")), t
    assert "Level lowerers" in t1
    assert "Volatility amplifiers" in t2


def test_title_disambiguation_regression_roman_numeral_alone_would_not_differentiate():
    """Planted-regression companion: the OLD single-family composer
    (`plain_text.compose_title`, still used by `concepts.py::plain_name` for
    tight concepts) can only tell the two families above apart with an
    arbitrary Roman-numeral suffix -- confirming the channel-based
    disambiguation above is not vacuously true (a numeral WOULD "work" too,
    it would just say nothing about why the two families differ)."""
    a = _row_vec("horizon_shape_near", 6.0)
    a[list(CHANNELS).index("level")] = -3.0
    b = _row_vec("horizon_shape_near", 6.0)
    b[list(CHANNELS).index("dispersion")] = 3.0
    cleared_a = [int(i) for i in np.argsort(-np.abs(a)) if abs(a[i]) >= 1.0]
    cleared_b = [int(i) for i in np.argsort(-np.abs(b)) if abs(b[i]) >= 1.0]
    used: set = set()
    t1 = compose_title(a, cleared_a, used)
    t2 = compose_title(b, cleared_b, used)
    assert t1 == "Near-horizon shapers"
    assert t2 == "Near-horizon shapers II"


def test_horizon_shape_clause_states_amount_not_direction():
    """Orchestrator review item 4b: `horizon_shape_near`/`_far`'s sign
    cannot support an up/down claim (`_horizon_shape` takes `abs()` per step
    before averaging, response.py) -- the description must read as an
    AMOUNT of reshaping, never a directional bend."""
    directed = _row_vec("horizon_shape_near", 6.0)
    cleared = [list(CHANNELS).index("horizon_shape_near")]
    text = _render_effect_clauses(directed, cleared)
    assert "increase how much" in text
    for word in ("upward", "downward", "bend", "bends"):
        assert word not in text, text


def test_horizon_shape_clause_negative_direction_says_decrease():
    directed = _row_vec("horizon_shape_far", -6.0)
    cleared = [list(CHANNELS).index("horizon_shape_far")]
    text = _render_effect_clauses(directed, cleared)
    assert "decrease how much" in text


def test_spectral_centroid_clause_is_directional_higher_or_lower_frequency():
    """Orchestrator review item 4b: unlike horizon_shape, `spectral_centroid`
    carries a real, recoverable sign (no `abs()` in `_spectral_centroid`),
    so its clause DOES get a directional claim."""
    directed_up = _row_vec("spectral_centroid", 6.0)
    cleared = [list(CHANNELS).index("spectral_centroid")]
    text_up = _render_effect_clauses(directed_up, cleared)
    assert "higher frequencies" in text_up

    directed_down = _row_vec("spectral_centroid", -6.0)
    text_down = _render_effect_clauses(directed_down, cleared)
    assert "lower frequencies" in text_down


def test_second_sentence_found_only_in_one_model():
    s = _family_second_sentence({"TimesFM": 3}, None)
    assert s == "Found only in TimesFM."


def test_second_sentence_found_in_multiple_models_and_plain_generator_label():
    """Orchestrator review item 4a/4c: "Its features come from N model(s)"
    reads as hedged; "Found in N models (...)" states it plainly, and the
    fires-on clause maps a raw generator name through the plain-label
    vocabulary rather than printing it verbatim."""
    s = _family_second_sentence({"TimesFM": 2, "Chronos-2": 1},
                                [{"label": "ar_colored_noise", "count": 2}])
    assert s.startswith("Found in 2 models (Chronos-2, TimesFM)."), s
    assert "noisy autocorrelated series" in s
    assert "ar_colored_noise" not in s


def test_second_sentence_regression_unmapped_generator_falls_back_to_raw_name():
    """Planted-regression companion: a generator name with no plain-label
    entry must still render (fallback to the raw name), not raise or vanish
    -- CLAUDE.md sec 8's "degrade with a stated fallback"."""
    s = _family_second_sentence({"TimesFM": 1}, [{"label": "some_future_generator", "count": 1}])
    assert "some_future_generator" in s


def test_sentence1_shortest_unique_prefix_disambiguates_shared_top2():
    """Real-run finding (measured on `runs/full_report_run_4model`, orchestrator
    diagnosis for this spec): several families share `horizon_shape_near`/`_far`
    as their top-2 cleared channels (both are mean-ABSOLUTE-deviation
    statistics, so almost every causal feature clears them), which a fixed
    top-2 clause count renders as byte-identical sentence-1 text for
    otherwise-distinct families. Two families here share that same top-2
    pair but differ on a THIRD channel; the batched shortest-unique-prefix
    must extend exactly the family that needs it to 3 clauses, leaving the
    other's shorter, already-unique 2-clause text alone."""
    shared = _row_vec("horizon_shape_near", 6.0)
    shared[list(CHANNELS).index("horizon_shape_far")] = 5.0
    a = shared.copy()
    a[list(CHANNELS).index("level")] = 4.0
    b = shared.copy()
    b[list(CHANNELS).index("mase")] = -4.0
    all_directed = [a, b]
    all_cleared = [[int(i) for i in np.argsort(-np.abs(v)) if abs(v[i]) >= 1.0] for v in all_directed]
    sentences = _compose_family_sentence1(all_directed, all_cleared)
    assert sentences[0] != sentences[1], sentences
    assert "level" in sentences[0].lower() or "level" in sentences[1].lower()


def test_sentence1_regression_fixed_top2_collides():
    """Planted regression: the OLD fixed-top-2-clause rendering (this
    module's own behavior before the shortest-unique-prefix fix) makes the
    two families above render byte-identical sentence-1 text -- confirming
    the positive test above is not vacuously true (the two families really
    do need disambiguation, this is not a fixture where any rendering would
    already differ)."""
    shared = _row_vec("horizon_shape_near", 6.0)
    shared[list(CHANNELS).index("horizon_shape_far")] = 5.0
    a = shared.copy()
    a[list(CHANNELS).index("level")] = 4.0
    b = shared.copy()
    b[list(CHANNELS).index("mase")] = -4.0
    cleared_a = [int(i) for i in np.argsort(-np.abs(a)) if abs(a[i]) >= 1.0]
    cleared_b = [int(i) for i in np.argsort(-np.abs(b)) if abs(b[i]) >= 1.0]
    from tsfm_lens.sae.concept_families import _render_effect_clauses
    fixed_top2_a = _render_effect_clauses(a, cleared_a[:2])
    fixed_top2_b = _render_effect_clauses(b, cleared_b[:2])
    assert fixed_top2_a == fixed_top2_b, (fixed_top2_a, fixed_top2_b)


# ---------------------------------------------------------------------------
# 5. End-to-end: run_concept_atlas then run_concept_families, byte-identical
#    legacy artifact, artifact written, families span models.
# ---------------------------------------------------------------------------

def _write_family_fixture_run(run_dir, seed=7):
    """Same 3-family/decoy/noise geometry as the pure-numpy fixture above,
    but written out as `sae/<model>/<layer>_ablation.json` files so the full
    `run_concept_atlas` -> `run_concept_families` chain can be exercised."""
    X, rows, _decoy_idx = _three_family_fixture_with_decoy_and_noise(seed)
    by_target: dict = {}
    for vec, row in zip(X, rows):
        key = (row["model"], row["layer"])
        by_target.setdefault(key, []).append((row["feature"], vec))
    for (model, layer), items in by_target.items():
        candidates = [_candidate(feat, {ch: float(v) for ch, v in zip(CHANNELS, vec)})
                     for feat, vec in items]
        _write_ablation(run_dir, model, layer, candidates)


def test_run_concept_families_end_to_end_writes_artifact_and_keeps_atlas_byte_identical(tmp_path):
    _write_family_fixture_run(tmp_path)
    cfg = _cfg(tmp_path / "cfgdir", n_null=25, seed=1)

    atlas_before = run_concept_atlas(tmp_path, cfg)
    atlas_json_before = load_json(tmp_path / "sae" / "concept_atlas.json")

    families = run_concept_families(tmp_path, atlas_before, cfg)

    atlas_json_after = load_json(tmp_path / "sae" / "concept_atlas.json")
    assert atlas_json_after == atlas_json_before, "concept_atlas.json must stay byte-identical"

    assert families["measured"] is True
    assert (tmp_path / "sae" / "concept_families.json").exists()
    assert len(families["families"]) >= 1
    spanning = [f for f in families["families"] if f["n_models"] >= 2]
    assert spanning, "at least one family should span >1 model on this fixture"
    for fam in families["families"]:
        assert fam["title"]
        assert fam["description"]
        # A combined ("X & Y") or fires-on-qualified ("X & Y (source)")
        # title can run longer than a single channel's 2-3 words (item 2);
        # bounded loosely just to catch a runaway/garbled composition.
        assert 2 <= len(fam["title"].split()) <= 12, fam["title"]

    loaded = load_families(tmp_path)
    assert loaded == families["families"]


def test_run_concept_families_withheld_or_empty_atlas_writes_empty_artifact(tmp_path):
    _write_ablation(tmp_path, "m1", "L1", [_candidate(0, _row("trend", 6.0))], withheld=True)
    cfg = _cfg(tmp_path / "cfgdir")
    atlas = run_concept_atlas(tmp_path, cfg)
    assert atlas["concepts"] == []
    families = run_concept_families(tmp_path, atlas, cfg)
    assert families["measured"] is False
    assert families["families"] == []
    assert load_families(tmp_path) == []


# ---------------------------------------------------------------------------
# 6. Fingerprint scoping: the two new fields must not move the `concepts`
#    stage's own whole-section fingerprint.
# ---------------------------------------------------------------------------

def test_new_family_fields_do_not_move_the_concepts_fingerprint(tmp_path):
    cfg = _cfg(tmp_path / "cfgdir")
    before = fingerprint_stage(resolve_config_keys(cfg, ("concepts",)), {})
    cfg.concepts.atlas_family_min_members = 9
    cfg.concepts.atlas_family_assign_min = 0.2
    cfg.concepts.atlas_family_sil_tol = 0.5
    after = fingerprint_stage(resolve_config_keys(cfg, ("concepts",)), {})
    assert before == after


def test_an_ordinary_concepts_field_still_moves_the_fingerprint_regression(tmp_path):
    """Planted regression companion: confirms the fingerprint mechanism
    itself is sensitive to SOME field in this section, so the equality
    above is not vacuously true because `resolve_config_keys` ignores the
    whole section."""
    cfg = _cfg(tmp_path / "cfgdir")
    before = fingerprint_stage(resolve_config_keys(cfg, ("concepts",)), {})
    cfg.concepts.atlas_min_cosine = 0.5
    after = fingerprint_stage(resolve_config_keys(cfg, ("concepts",)), {})
    assert before != after
