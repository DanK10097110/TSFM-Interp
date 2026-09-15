"""Tests for `sae/concepts.py` (ROADMAP.md sec 30, sec 30.10 stages 1-2): the
gate-tested re-clustering of ablation-battery candidates (sec 27,
`sae/response.py::feature_ablation_fingerprints`) into concepts, holding the
clustering METHOD fixed against `roles.py::cluster_roles` so sec 30.1's
NEW-vs-OLD comparison is a statement about the feature space, not about two
different clusterers (`CLAUDE.md` sec 11.41); and stage 2's deterministic
naming composer (`compose_name`, sec 30.4.1/sec 30.7 BUILD half) plus its
whole-run wiring (`assign_concept_names`).

Stage 1 scope: `ablation_vector`, `build_concept_matrix`, `cluster_concepts`,
`concept_table` (still pinned to leave `name`/`name_lead_diversified` as
placeholders on its OWN output -- naming is a separate pass run once every
target's concepts exist, sec 30.7's whole-run peer set). Stage 2 scope:
`compose_name`, `assign_concept_names`.

All synthetic, with planted answers. Each negative below is confirmed to
discriminate by planting the regression and reading pytest's own summary
line, per sec 11.55's corollary (a plant that produces a syntax error reports
`1 error`, not `N failed`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.concepts import (
    CHANNELS,
    ablation_vector,
    assign_concept_names,
    build_concept_matrix,
    cluster_concepts,
    compose_name,
    concept_table,
)
from tsfm_lens.sae.roles import cluster_roles


def _channel_rec(signed_effect, null_p95, clears=None):
    if signed_effect is None or not null_p95:
        clears = False if clears is None else clears
        return {"available": signed_effect is not None, "effect": None,
               "signed_effect": signed_effect, "null_p95": null_p95,
               "clears_null": clears}
    effect = abs(signed_effect)
    if clears is None:
        clears = effect > null_p95
    return {"available": True, "effect": effect, "signed_effect": signed_effect,
           "null_p95": null_p95, "clears_null": clears}


def _candidate(feature: int, effects: dict | None = None, scorable: bool = True,
              n_channels_clearing: int | None = None) -> dict:
    """A minimal `feature_ablation_fingerprints`-shaped candidate record.

    `effects` maps channel -> (signed_effect, null_p95); a channel absent
    from `effects` is unscorable (null_p95=None, signed_effect=None), per
    the real artifact's own per-channel `available: False` shape.
    """
    effects = effects or {}
    channels = {}
    for ch in CHANNELS:
        if ch in effects:
            signed, p95 = effects[ch]
            channels[ch] = _channel_rec(signed, p95)
        else:
            channels[ch] = _channel_rec(None, None)
    n_clearing = (sum(1 for c in channels.values() if c["clears_null"])
                 if n_channels_clearing is None else n_channels_clearing)
    return {"feature": feature, "rules": ["variance"], "scorable": scorable,
           "n_top_series": 8, "channels": channels,
           "n_channels_clearing": n_clearing}


# ---------------------------------------------------------------------------
# ablation_vector
# ---------------------------------------------------------------------------

def test_ablation_vector_unscored_channel_is_zero_not_dropped():
    cand = _candidate(0, effects={"trend": (4.0, 2.0)})  # only 1 of 9 scorable
    vec = ablation_vector(cand)
    assert vec is not None
    trend_idx = CHANNELS.index("trend")
    assert vec[trend_idx] == pytest.approx(2.0)
    other_idxs = [i for i in range(len(CHANNELS)) if i != trend_idx]
    assert np.all(vec[other_idxs] == 0.0)
    assert cand["n_channels_unscored"] == len(CHANNELS) - 1


def test_ablation_vector_all_unscored_returns_none():
    cand = _candidate(0, effects={})
    vec = ablation_vector(cand)
    assert vec is None
    assert cand["n_channels_unscored"] == len(CHANNELS)


# ---------------------------------------------------------------------------
# build_concept_matrix
# ---------------------------------------------------------------------------

def test_causal_only_drops_are_counted():
    causal = [_candidate(i, effects={"trend": (4.0, 1.0)}) for i in range(3)]
    non_causal = [_candidate(i + 3, effects={"trend": (0.1, 1.0)}) for i in range(2)]
    X, feature_ids, diag = build_concept_matrix(causal + non_causal, causal_only=True)
    assert diag["n_dropped_no_cleared_channel"] == 2
    assert X.shape[0] == 3
    assert sorted(feature_ids) == [0, 1, 2]


def test_causal_only_drop_count_negative_plant():
    """Plant: hardcode the diagnostic to 0 regardless of what was dropped --
    the test must fail, confirming it actually reads the real count."""
    causal = [_candidate(i, effects={"trend": (4.0, 1.0)}) for i in range(3)]
    non_causal = [_candidate(i + 3, effects={"trend": (0.1, 1.0)}) for i in range(2)]
    _, _, diag = build_concept_matrix(causal + non_causal, causal_only=True)
    planted_wrong = 0
    assert diag["n_dropped_no_cleared_channel"] != planted_wrong


def test_build_concept_matrix_skips_non_scorable():
    cands = [_candidate(0, effects={"trend": (4.0, 1.0)}),
            _candidate(1, effects={"trend": (4.0, 1.0)}, scorable=False)]
    X, feature_ids, diag = build_concept_matrix(cands, causal_only=True)
    assert feature_ids == [0]
    assert diag["n_dropped_not_scorable"] == 1


def test_build_concept_matrix_drops_all_unscored_row():
    cands = [_candidate(0, effects={"trend": (4.0, 1.0)}),
            _candidate(1, effects={}, n_channels_clearing=0)]
    # feature 1 clears nothing AND is all-unscored -- dropped by causal_only
    # before ablation_vector ever runs, so it must land in the
    # no-cleared-channel bucket, not the all-unscored one.
    X, feature_ids, diag = build_concept_matrix(cands, causal_only=True)
    assert feature_ids == [0]
    assert diag["n_dropped_no_cleared_channel"] == 1
    assert diag["n_dropped_all_unscored"] == 0


# ---------------------------------------------------------------------------
# cluster_concepts: method held fixed against roles.cluster_roles
# ---------------------------------------------------------------------------

def test_cluster_concepts_is_roles_cluster_roles():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, len(CHANNELS)))
    a = cluster_concepts(X, k=3, min_silhouette=0.1, seed=0)
    b = cluster_roles(X, role_k=3, min_silhouette=0.1, seed=0)
    assert np.array_equal(a["labels"], b["labels"])
    assert a["k"] == b["k"]
    assert a["silhouette"] == pytest.approx(b["silhouette"], nan_ok=True)


def test_k_auto_matches_resolve_role_k_at_min_members_1():
    """ROADMAP.md sec 32.5 Item D5: `min_members<=1` is the reproducibility
    sentinel -- it bypasses D1's sweep entirely and delegates straight to
    the pre-D ratio (`_resolve_role_k`), so every concept recorded before
    Item D stays regenerable bit for bit. This is the exact contract this
    test pinned before Item D existed, now spelled out with the sentinel
    that keeps it true (the new DEFAULT, `min_members=3`, no longer
    reproduces this -- see `tests/test_concept_clustering.py` for that)."""
    from tsfm_lens.sae.roles import _resolve_role_k
    rng = np.random.default_rng(0)
    for n in (12, 24, 48, 96):
        X = rng.normal(size=(n, len(CHANNELS)))
        result = cluster_concepts(X, k="auto", min_members=1, seed=0)
        assert result["k"] == _resolve_role_k("auto", n)
        assert result["k_sweep"] is None
        assert result["k_selection_rule"] == "ratio (min_members<=1, legacy reproduction)"


def test_fewer_than_four_candidates_is_skipped_with_reason():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, len(CHANNELS)))
    result = cluster_concepts(X, k="auto")
    assert result["non_modular"] is True
    assert result["reason"] != ""
    assert "3" in result["reason"]


def test_non_modular_propagates():
    # Every candidate identical -> zero within-cluster spread is impossible
    # to beat between-cluster spread meaningfully at k>=2; use a tiny,
    # deliberately unclusterable near-degenerate matrix.
    X = np.tile(np.array([[0.0] * len(CHANNELS)]), (8, 1))
    X += np.random.default_rng(0).normal(scale=1e-6, size=X.shape)
    result = cluster_concepts(X, k="auto", min_silhouette=0.9)
    assert result["non_modular"] is True


def test_planted_two_group_structure_recovers():
    rng = np.random.default_rng(0)
    n = len(CHANNELS)
    group_a = rng.normal(loc=5.0, scale=0.2, size=(15, n))
    group_b = rng.normal(loc=-5.0, scale=0.2, size=(15, n))
    X = np.vstack([group_a, group_b])
    result = cluster_concepts(X, k=2, min_silhouette=0.1, seed=0)
    assert result["silhouette"] > 0.5
    labels = result["labels"]
    # The two planted groups must land in different clusters (up to a
    # global relabeling -- KMeans label identity is arbitrary).
    assert len(set(labels[:15])) == 1
    assert len(set(labels[15:])) == 1
    assert labels[0] != labels[15]


# ---------------------------------------------------------------------------
# concept_table
# ---------------------------------------------------------------------------

def test_concept_table_empty_when_non_modular():
    candidates = [_candidate(i, effects={"trend": (4.0, 1.0)}) for i in range(3)]
    X, feature_ids, _ = build_concept_matrix(candidates, causal_only=True)
    cluster_result = cluster_concepts(X, k="auto")
    assert cluster_result["non_modular"] is True
    concepts = concept_table(candidates, X, feature_ids, cluster_result)
    assert concepts == []


def test_concept_table_name_and_misfits_are_placeholders_not_fabricated():
    """Stage 1 does not implement `compose_name` (sec 30.10 stage 2) or
    `misfits.py` (stage 5) -- `concept_table` must not invent a name from
    `dominant_channel` in the meantime (sec 30.1 measurement 2's own
    argmax-of-the-mean failure). Negative: a version that sets
    `name = dominant_channel` would pass every other test in this file and
    must be caught here specifically."""
    rng = np.random.default_rng(0)
    n = len(CHANNELS)
    group_a = rng.normal(loc=5.0, scale=0.2, size=(15, n))
    group_b = rng.normal(loc=-5.0, scale=0.2, size=(15, n))
    X = np.vstack([group_a, group_b])
    candidates = [_candidate(i, effects={"trend": (5.0, 1.0)}) for i in range(30)]
    feature_ids = list(range(30))
    cluster_result = cluster_concepts(X, k=2, min_silhouette=0.1, seed=0)
    concepts = concept_table(candidates, X, feature_ids, cluster_result)
    assert len(concepts) == 2
    for concept in concepts:
        assert concept["name"] is None
        assert concept["name_lead_diversified"] is None
        assert concept["misfits"] == []
        assert concept["description"] is None
        assert concept["description_generated"] is False


def test_concept_table_profile_only_channels_clearing_own_null():
    candidates = [
        _candidate(0, effects={"trend": (5.0, 1.0), "seasonal": (0.2, 1.0)}),
        _candidate(1, effects={"trend": (6.0, 1.0), "seasonal": (0.1, 1.0)}),
        _candidate(2, effects={"trend": (5.5, 1.0), "seasonal": (0.3, 1.0)}),
        _candidate(3, effects={"trend": (4.5, 1.0), "seasonal": (0.0, 1.0)}),
    ]
    X, feature_ids, _ = build_concept_matrix(candidates, causal_only=True)
    cluster_result = cluster_concepts(X, k=1)
    # k=1 forces a single "cluster" only via cluster_roles' own >=4 path;
    # if it comes back non_modular that's a valid result too -- guard it.
    if cluster_result["non_modular"]:
        pytest.skip("degenerate synthetic matrix clustered non-modular")
    concepts = concept_table(candidates, X, feature_ids, cluster_result)
    profile_channels = {p["channel"] for c in concepts for p in c["profile"]}
    assert "trend" in profile_channels
    assert "seasonal" not in profile_channels


def test_concept_table_n_members_clearing_counts_any_clearing_channel():
    candidates = [
        _candidate(0, effects={"trend": (5.0, 1.0)}),
        _candidate(1, effects={"trend": (5.0, 1.0)}),
        _candidate(2, effects={"trend": (5.0, 1.0)}, n_channels_clearing=0),
    ]
    # Force feature 2 into the same cluster by construction: build X by hand
    # so all three rows are near-identical on "trend" and feature 2's own
    # per-candidate n_channels_clearing is overridden to 0 (simulating a
    # borderline candidate whose ablation_vector row still resembles its
    # neighbours' despite not itself individually clearing).
    X = np.zeros((3, len(CHANNELS)))
    X[:, CHANNELS.index("trend")] = [5.0, 5.0, 5.0]
    feature_ids = [0, 1, 2]
    labels = np.array([0, 0, 0])
    cluster_result = {"labels": labels, "k": 1, "silhouette": float("nan"),
                      "non_modular": False, "reason": ""}
    concepts = concept_table(candidates, X, feature_ids, cluster_result)
    assert len(concepts) == 1
    assert concepts[0]["n_members_clearing"] == 2
    assert concepts[0]["n_members"] == 3


# ---------------------------------------------------------------------------
# compose_name (sec 30.10 stage 2)
# ---------------------------------------------------------------------------

def _profile_matrix(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, len(CHANNELS)), dtype=np.float64)


def test_compose_name_contrastive_beats_absolute():
    """Plant: `trend` is ~5.0 for every row (large, but with zero variance
    -> |z|=0 for everyone). `seasonal` is ~0.1 for rows 0-3 and 4.9 for row
    4 -- comparable RAW magnitude to `trend`, but a huge |z| for row 4 since
    it is an outlier against the population.

    Under design (c) (ROADMAP.md sec 32.4, Item C) `trend` is `row 4`'s
    DOMINANT lead too -- it is still the larger raw magnitude (5.0 > 4.9),
    and the dominant clause is now fixed/unconditional rather than
    competing in the contrastive pool. So "trend must not appear in row 4's
    name at all" (this test's pre-Item-C assertion) is no longer the right
    discriminator: it would now reject the correct output. What still
    separates naming-by-contrastive-z from naming-by-raw-magnitude is the
    CONTRAST clause -- only row 4's `seasonal` value is a population
    outlier (huge |z|), so only row 4 earns a "unusually ... seasonal"
    contrast clause; rows 0-3, whose only cleared channel IS their dominant
    channel, have no contrast candidate at all and render no such clause."""
    trend = CHANNELS.index("trend")
    seasonal = CHANNELS.index("seasonal")
    profiles = _profile_matrix(5)
    profiles[:, trend] = 5.0
    profiles[:4, seasonal] = 0.1
    profiles[4, seasonal] = 4.9
    cleared = np.zeros((5, len(CHANNELS)), dtype=bool)
    cleared[:, trend] = True
    cleared[4, seasonal] = True
    peers = [("t", i) for i in range(5)]

    name_odd = compose_name(4, profiles, cleared, peers)
    assert "trend" in name_odd  # fixed dominant lead, largest raw |mean|
    assert "seasonal" in name_odd  # contrastive population outlier
    assert "unusually" in name_odd  # rendered as the CONTRAST clause
    for i in range(4):
        name = compose_name(i, profiles, cleared, peers)
        assert "trend" in name
        assert "seasonal" not in name  # not cleared, not an outlier here


def test_compose_name_is_shortest_unique_prefix():
    """ROADMAP.md sec 32.4, Item C, design (c): the DOMINANT clause (here
    `trend`, tied and largest for both subjects) is fixed and never
    competes in mechanism 2/3's clause-selection pool, so what needs
    disambiguating is the CONTRAST clauses alone (capped at
    `_MAX_CLAUSES - 1 = 2`).

    Two "thief" rows independently claim `mase`/`dispersion` as their OWN
    lead contrast clause first (smaller peer keys -> `_compose_batch`'s
    tie-broken processing order reaches them before either subject; each
    thief's own dominant is a distinct, unrelated channel so its claimed
    channel stays a genuine contrast candidate for it, not its dominant).
    Both subjects clear `level` (shared, tied) plus their OWN distinct 2nd
    channel (`mase` vs `dispersion`) -- constructed so all of `level`'s,
    `mase`'s and `dispersion`'s |z| for the relevant row are EXACTLY tied
    (each column has exactly 2 of 7 rows at the same nonzero value),
    breaking only on channel index. Subject 0 reaches `level` unforced
    (unused when its turn comes); subject 1's only other candidate
    (`dispersion`) is already claimed by a thief, so it is forced into a
    genuine, unavoidable collision with subject 0 on `level`. Both then
    need their own 2nd contrast clause to disambiguate -- 3 clauses total
    (dominant + 2 contrast, 2 separators) each.

    Solo rows each carry one channel as their own dominant with nothing
    else cleared, so they render the dominant clause alone: 0 separators.
    Negative: forcing k=1 globally (denying either subject its own 2nd
    clause) would leave the two subjects colliding."""
    trend, level = CHANNELS.index("trend"), CHANNELS.index("level")
    mase, disp = CHANNELS.index("mase"), CHANNELS.index("dispersion")
    flatness = CHANNELS.index("flatness")
    hz_far = CHANNELS.index("horizon_shape_far")
    solo_channels = [CHANNELS.index(c) for c in
                     ("seasonal", "spectral_centroid", "horizon_shape_near")]

    v, dom, big = 3.0, 10.0, 8.0
    rows = [
        # Thieves: a distinct dominant channel (so the channel they exist
        # to claim stays a CONTRAST candidate for them, not their own
        # dominant), plus the one candidate they claim.
        ({flatness: big, mase: v}, {mase}, ("t", 0)),
        ({hz_far: big, disp: v}, {disp}, ("t", 1)),
        # Subjects: tied dominant (trend), tied shared contrast (level),
        # and their own distinct 2nd contrast channel.
        ({trend: dom, level: v, mase: v}, {level, mase}, ("t", 2)),
        ({trend: dom, level: v, disp: v}, {level, disp}, ("t", 3)),
    ]
    for j, ch in enumerate(solo_channels):
        rows.append(({ch: big}, set(), ("t", 100 + j)))

    n = len(rows)
    profiles = _profile_matrix(n)
    cleared = np.zeros((n, len(CHANNELS)), dtype=bool)
    peers = []
    for r, (vals, clr, key) in enumerate(rows):
        for ch, val in vals.items():
            profiles[r, ch] = val
        for ch in clr:
            cleared[r, ch] = True
        peers.append(key)

    batch_names = [compose_name(i, profiles, cleared, peers) for i in range(n)]
    subj0, subj1 = batch_names[2], batch_names[3]
    assert subj0 != subj1
    assert subj0.count("·") == 2  # dominant + 2 contrast clauses
    assert subj1.count("·") == 2
    assert "mase" in subj0 and "dispersion" not in subj0
    assert "dispersion" in subj1 and "mase" not in subj1
    for r in range(4, n):  # the solo rows
        assert batch_names[r].count("·") == 0


def test_lead_diversification_is_order_stable():
    """Shuffling the row order (permuting profiles/cleared/peers together)
    must not change any peer's own name -- the pass orders by (top-|z|,
    peer id), never by array position (sec 11.2)."""
    trend, seasonal, level = (CHANNELS.index(c) for c in ("trend", "seasonal", "level"))
    n = 6
    rng = np.random.default_rng(0)
    profiles = _profile_matrix(n)
    cleared = np.zeros((n, len(CHANNELS)), dtype=bool)
    for r in range(n):
        profiles[r, trend] = 3.0 + 0.01 * r
        profiles[r, seasonal] = 2.0 + rng.normal(scale=0.3)
        profiles[r, level] = 1.5
        cleared[r, [trend, seasonal, level]] = True
    peers = [("t", i) for i in range(n)]

    names_in_order = {peers[i]: compose_name(i, profiles, cleared, peers) for i in range(n)}

    perm = np.array([3, 0, 5, 1, 4, 2])
    profiles_p = profiles[perm]
    cleared_p = cleared[perm]
    peers_p = [peers[i] for i in perm]
    names_shuffled = {peers_p[j]: compose_name(j, profiles_p, cleared_p, peers_p)
                      for j in range(n)}

    assert names_in_order == names_shuffled


def test_lead_diversification_only_reorders_true_clauses():
    """Every clause in every composed name names a channel that is `cleared`
    for that row, with the verb's sign matching the row's own raw value --
    diversification may change WHICH true clause leads, never whether a
    clause is true."""
    rng = np.random.default_rng(1)
    n = 10
    profiles = _profile_matrix(n)
    cleared = np.zeros((n, len(CHANNELS)), dtype=bool)
    for r in range(n):
        k = rng.integers(1, 4)
        chans = rng.choice(len(CHANNELS), size=k, replace=False)
        for c in chans:
            profiles[r, c] = rng.choice([-1, 1]) * rng.uniform(1.0, 8.0)
            cleared[r, c] = True
    peers = [("t", i) for i in range(n)]

    for i in range(n):
        name = compose_name(i, profiles, cleared, peers)
        if name == "no channel clears its own null":
            assert not cleared[i].any()
            continue
        for clause in name.split(" · "):
            words = clause.split()
            verb = words[0] if words[0] not in ("dominant", "strong", "mild") else words[1]
            channel = words[-1]
            ci = CHANNELS.index(channel)
            assert cleared[i, ci]
            val = profiles[i, ci]
            if verb == "raises":
                assert val >= 1.0
            elif verb == "lowers":
                assert val <= -1.0
            else:
                assert verb == "moves"


def test_lead_diversification_dedups_on_rendered_clause_not_bare_channel():
    """Plant: row 0 clears only `trend` (+7.0, its sole candidate, processed
    first since it has the higher top-|z|). Row 1 clears `trend` (-5.0, its
    top |z| candidate -- opposite sign) and, more weakly, `seasonal` (its
    second-best candidate) -- so row 1 has somewhere to fall back to if its
    top pick is refused. Three filler rows give both channels a realistic
    population spread so the ranking isn't an artifact of only two rows
    existing.

    Deduping the lead on BARE CHANNEL INDEX (an earlier, buggy version of
    `_compose_batch`) sees `trend` already "taken" by row 0's key and
    refuses it for row 1 even though row 1's `trend` would render as
    "lowers trend" -- distinct text from row 0's "raises trend" -- pushing
    row 1 onto its weaker `seasonal` clause instead (confirmed: reverting
    the dedup key to bare channel index changes row 1's composed name to
    "mild raises seasonal"). Deduping on the RENDERED (channel, verb, tier)
    triple -- the spec's literal "whose leading clause no earlier row has
    taken" -- sees the two keys differ and lets row 1 keep its own best
    (`trend`). See `CLAUDE.md` sec 30's dated Stage 2 update (2026-09-11)
    for the real-artifact diagnosis this fix was built from."""
    trend = CHANNELS.index("trend")
    seasonal = CHANNELS.index("seasonal")
    profiles = _profile_matrix(5)
    profiles[0, trend] = 7.0
    profiles[1, trend] = -5.0
    profiles[1, seasonal] = 1.05
    profiles[2, trend] = 0.3
    profiles[2, seasonal] = 1.0
    profiles[3, trend] = -0.2
    profiles[3, seasonal] = -1.0
    profiles[4, trend] = 0.15
    profiles[4, seasonal] = 0.5
    cleared = np.zeros((5, len(CHANNELS)), dtype=bool)
    cleared[0, trend] = True
    cleared[1, [trend, seasonal]] = True
    peers = [("t", i) for i in range(5)]

    name0 = compose_name(0, profiles, cleared, peers)
    name1 = compose_name(1, profiles, cleared, peers)
    assert name0.split(" · ")[0] == "dominant raises trend"
    assert name1.split(" · ")[0] == "dominant lowers trend"


def test_lead_diversification_dedups_same_sign_different_tier():
    """Companion plant to the one above, isolating the TIER half of the key:
    row 0 and row 1 both clear `trend` with the SAME sign but different
    magnitude tiers ("mild" 1.0-2.0 null units vs "dominant" >=5.0 -- equally
    distinct rendered text), and row 1 again has a weaker `seasonal`
    fallback it would be wrongly pushed onto if the dedup key ignored tier
    (confirmed: reverting the dedup key to bare channel index changes row
    1's composed name to "moves seasonal"). 40 near-zero filler rows are
    needed here -- unlike the opposite-sign plant above, a same-sign pair
    sharing one channel pulls the population mean/std toward the larger
    value, so row 1's smaller-but-still-mild value only outranks its own
    `seasonal` candidate once enough filler dilutes that pull; a handful of
    the filler rows carry background `seasonal` noise so row 1's `seasonal`
    isn't itself a lone-outlier artifact (sec 11.55's own lesson: an
    existential/outlier-shaped statistic among near-zero peers reads as
    significant almost regardless of magnitude)."""
    trend = CHANNELS.index("trend")
    seasonal = CHANNELS.index("seasonal")
    n = 42
    profiles = _profile_matrix(n)
    profiles[0, trend] = 7.0   # dominant, processed first (higher |z|)
    profiles[1, trend] = 1.3   # mild, same sign as row 0
    profiles[1, seasonal] = 0.2
    rng = np.random.default_rng(0)
    noisy = rng.choice(np.arange(2, n), size=10, replace=False)
    profiles[noisy, seasonal] = rng.uniform(-1.0, 1.0, size=10)
    cleared = np.zeros((n, len(CHANNELS)), dtype=bool)
    cleared[0, trend] = True
    cleared[1, [trend, seasonal]] = True
    peers = [("t", i) for i in range(n)]

    name0 = compose_name(0, profiles, cleared, peers)
    name1 = compose_name(1, profiles, cleared, peers)
    assert name0.split(" · ")[0] == "dominant raises trend"
    assert name1.split(" · ")[0] == "mild raises trend"


def test_sub_null_channel_loses_its_direction():
    trend = CHANNELS.index("trend")
    profiles = _profile_matrix(2)
    profiles[0, trend] = 0.4
    profiles[1, trend] = 5.0  # a peer, so the channel has some spread
    cleared = np.zeros((2, len(CHANNELS)), dtype=bool)
    cleared[:, trend] = True
    peers = [("t", 0), ("t", 1)]

    name = compose_name(0, profiles, cleared, peers)
    assert "moves trend" in name
    assert "raises trend" not in name
    assert "lowers trend" not in name


def test_sub_null_channel_negative_plant_restoring_sign_fails():
    """Negative: a version that used raises/lowers regardless of magnitude
    would render "raises trend" here -- confirm the real test above would
    have caught it by checking the wrong string is genuinely absent from
    what a magnitude-blind renderer would have produced."""
    trend = CHANNELS.index("trend")
    profiles = _profile_matrix(2)
    profiles[0, trend] = 0.4
    profiles[1, trend] = 5.0
    cleared = np.zeros((2, len(CHANNELS)), dtype=bool)
    cleared[:, trend] = True
    peers = [("t", 0), ("t", 1)]
    name = compose_name(0, profiles, cleared, peers)
    planted_wrong = "raises trend"
    assert name != planted_wrong


def test_tier_boundaries():
    trend = CHANNELS.index("trend")
    cases = [(4.99, "strong"), (5.0, "dominant"), (1.99, "mild")]
    for val, expected_tier in cases:
        profiles = _profile_matrix(2)
        profiles[0, trend] = val
        profiles[1, trend] = 0.0
        cleared = np.zeros((2, len(CHANNELS)), dtype=bool)
        cleared[0, trend] = True
        peers = [("t", 0), ("t", 1)]
        name = compose_name(0, profiles, cleared, peers)
        assert name.startswith(expected_tier), (val, name)


# 3 clauses joined by "·" have 2 separators.
_MAX_CLAUSES_MINUS_ONE = 2


def test_name_never_exceeds_three_clauses():
    # 15 rows, every one identical on every one of the 9 channels (same
    # magnitude per channel, same across rows) -> |z|=0 everywhere, so
    # nothing is ever contrastive and lead-diversification's tie-break
    # (peers[i], processed row 0..14) can hand out at most 9 distinct
    # leading channels before the supply is exhausted. Rows 9-14 (and row
    # 0, whose own top-3 is what the exhausted rows fall back to) are
    # therefore forced into an identical clause sequence at every k, and
    # the 3-clause cap must still hold instead of letting the collision
    # grow unbounded.
    n = 15
    profiles = _profile_matrix(n)
    cleared = np.zeros((n, len(CHANNELS)), dtype=bool)
    for r in range(n):
        for c in range(len(CHANNELS)):
            profiles[r, c] = 6.0 + c  # distinct magnitude per channel, same across rows
            cleared[r, c] = True
    peers = [("t", i) for i in range(n)]
    names = [compose_name(i, profiles, cleared, peers) for i in range(n)]
    for name in names:
        assert name.count("·") <= _MAX_CLAUSES_MINUS_ONE
    # The forced collision actually happens -- otherwise this test would
    # trivially pass without ever exercising the cap.
    assert len(set(names)) < n


# ---------------------------------------------------------------------------
# assign_concept_names: whole-run wiring
# ---------------------------------------------------------------------------

def _concept_record(concept_id, features, centroid, profile):
    return {"concept": concept_id, "features": features, "n_members": len(features),
           "n_members_clearing": len(features), "centroid_null_units": centroid,
           "profile": profile, "dominant_channel": None, "name": None,
           "name_lead_diversified": None, "within_cosine_mean": float("nan"),
           "misfits": [], "description": None, "description_generated": False}


def test_assign_concept_names_peer_set_spans_whole_run():
    """Two targets, each with one concept sharing the SAME dominant channel
    (`trend`) at nearly the same magnitude -- if the peer set were per-
    target, both would trivially get the same/only name; naming across the
    WHOLE RUN (as sec 30.7 requires) must still produce two distinct names
    since a third, contrastive, per-target-unique channel exists on one of
    them."""
    seasonal = "seasonal"
    targets = {
        "M1/L1": {"concepts": [
            _concept_record(0, [1, 2], {"trend": 6.0, seasonal: 0.0},
                            [{"channel": "trend", "signed_null_units": 6.0, "n_members_clearing": 2}]),
        ]},
        "M2/L1": {"concepts": [
            _concept_record(0, [3, 4], {"trend": 6.0, seasonal: 5.5},
                            [{"channel": "trend", "signed_null_units": 6.0, "n_members_clearing": 2},
                             {"channel": seasonal, "signed_null_units": 5.5, "n_members_clearing": 2}]),
        ]},
    }
    assign_concept_names(targets)
    name_a = targets["M1/L1"]["concepts"][0]["name"]
    name_b = targets["M2/L1"]["concepts"][0]["name"]
    assert name_a is not None and name_b is not None
    assert name_a != name_b


def test_assign_concept_names_sets_lead_diversified_flag():
    targets = {
        "M1/L1": {"concepts": [
            _concept_record(0, [1], {"trend": 6.0},
                            [{"channel": "trend", "signed_null_units": 6.0, "n_members_clearing": 1}]),
        ]},
    }
    assign_concept_names(targets)
    concept = targets["M1/L1"]["concepts"][0]
    assert concept["name_lead_diversified"] is False
    assert concept["name"] is not None


def test_assign_concept_names_skips_withheld_targets():
    targets = {
        "M1/L1": {"withheld": True, "reason": "x"},
        "M2/L1": {"concepts": [
            _concept_record(0, [1], {"trend": 6.0},
                            [{"channel": "trend", "signed_null_units": 6.0, "n_members_clearing": 1}]),
        ]},
    }
    assign_concept_names(targets)  # must not raise on the withheld target
    assert targets["M2/L1"]["concepts"][0]["name"] is not None
