"""ROADMAP.md sec 37.7 P4, item 3/4: concept-level `level carrier` /
`shape-causal` / `no measured effect` tags, and their wiring into the atlas's
name composer (`sae/concept_atlas.py::_tag_causal_effect`, `_name_concepts`,
`sae/concepts.py::_compose_batch`'s `tags` parameter).

Complements `tests/test_level_removed_battery.py` (the battery-level fixture
tests sec 37.7 itself asks for) with the concept-level classification and
naming this module's docstring ("Concept tags... go on atlas concepts, and
into the concept name composer's input") also requires. Every load-bearing
assertion is confirmed against a planted regression -- see each test's own
docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_concept_atlas import _candidate, _cfg, _write_ablation  # noqa: E402
from tsfm_lens.sae.concept_atlas import (  # noqa: E402
    _candidate_lookup, _name_concepts, _tag_causal_effect, run_concept_atlas)
from tsfm_lens.sae.concepts import CHANNELS, _compose_batch  # noqa: E402
from tsfm_lens.utils import load_json  # noqa: E402

_N_CH = len(CHANNELS)
_THRESH = 0.9


def _rows(*specs):
    """`specs` is `[(model, layer, feature), ...]`."""
    return [{"model": m, "layer": l, "feature": f} for m, l, f in specs]


def _lookup(entries: dict) -> dict:
    """`entries` is `{(model, layer, feature): {"level_share": ..,
    "n_shape_channels_clearing": ..}}`."""
    return dict(entries)


# ---------------------------------------------------------------------------
# 1. `_tag_causal_effect` classification, in memory.
# ---------------------------------------------------------------------------

def test_level_carrier_requires_high_median_share_and_no_shape_clear():
    rows = _rows(("A", "L1", 0), ("A", "L1", 1), ("A", "L1", 2))
    concepts = [{"members": [0, 1, 2]}]
    lookup = _lookup({
        ("A", "L1", 0): {"level_share": 0.95, "n_shape_channels_clearing": 0},
        ("A", "L1", 1): {"level_share": 0.92, "n_shape_channels_clearing": 0},
        ("A", "L1", 2): {"level_share": 0.99, "n_shape_channels_clearing": 0},
    })
    _tag_causal_effect(concepts, rows, lookup, _THRESH)
    assert concepts[0]["causal_tag"] == "level carrier"
    assert concepts[0]["level_share_median"] == pytest.approx(0.95)
    assert concepts[0]["n_members_with_level_share"] == 3


def test_shape_causal_wins_even_with_high_level_share():
    """The priority order (sec 37.7 item 4) is checked FIRST for a shape
    clear, not for the level-share threshold -- a concept whose members are
    almost entirely level-driven but where at least one genuinely cleared a
    level-removed channel is still `shape-causal`, not `level carrier`.

    Planted regression checked: swapping the two branches' order (checking
    `median_ls >= threshold` BEFORE `any_shape_clears`) makes this test fail
    -- the concept is mistagged `level carrier` -- confirming the ordering
    in the implementation is load-bearing, not incidental.
    """
    rows = _rows(("A", "L1", 0), ("A", "L1", 1))
    concepts = [{"members": [0, 1]}]
    lookup = _lookup({
        ("A", "L1", 0): {"level_share": 0.97, "n_shape_channels_clearing": 0},
        ("A", "L1", 1): {"level_share": 0.96, "n_shape_channels_clearing": 1},
    })
    _tag_causal_effect(concepts, rows, lookup, _THRESH)
    assert concepts[0]["causal_tag"] == "shape-causal"


def test_no_measured_effect_when_neither_condition_holds():
    rows = _rows(("A", "L1", 0), ("A", "L1", 1))
    concepts = [{"members": [0, 1]}]
    lookup = _lookup({
        ("A", "L1", 0): {"level_share": 0.4, "n_shape_channels_clearing": 0},
        ("A", "L1", 1): {"level_share": 0.3, "n_shape_channels_clearing": 0},
    })
    _tag_causal_effect(concepts, rows, lookup, _THRESH)
    assert concepts[0]["causal_tag"] == "no measured effect"


def test_missing_lookup_entries_are_skipped_not_raised():
    """A member from an older ablation artifact (no `level_share` field at
    all, e.g. absent from `lookup` entirely) must not crash the pass, and
    the tag/median must be computed from whichever members DO carry it."""
    rows = _rows(("A", "L1", 0), ("A", "L1", 1))
    concepts = [{"members": [0, 1]}]
    lookup = _lookup({("A", "L1", 0): {"level_share": 0.95, "n_shape_channels_clearing": 0}})
    _tag_causal_effect(concepts, rows, lookup, _THRESH)
    assert concepts[0]["causal_tag"] == "level carrier"
    assert concepts[0]["n_members_with_level_share"] == 1


def test_undefined_level_share_never_reads_as_zero():
    """Every member's `level_share` is `None` (sec 11.37's undefined-not-zero
    case, e.g. every candidate's own denominator was exactly 0): the median
    must be `None`, not silently treated as 0.0 -- which would still
    correctly tag `no measured effect` here, but for the WRONG reason, and a
    caller reading `level_share_median` would see a fabricated number."""
    rows = _rows(("A", "L1", 0))
    concepts = [{"members": [0]}]
    lookup = _lookup({("A", "L1", 0): {"level_share": None, "n_shape_channels_clearing": 0}})
    _tag_causal_effect(concepts, rows, lookup, _THRESH)
    assert concepts[0]["level_share_median"] is None
    assert concepts[0]["n_members_with_level_share"] == 0
    assert concepts[0]["causal_tag"] == "no measured effect"


# ---------------------------------------------------------------------------
# 2. `_compose_batch`'s `tags` parameter: level carriers get a fixed,
#    honest name and free their clause slots for real peers.
# ---------------------------------------------------------------------------

def _profile_row(**channel_values) -> list:
    return [float(channel_values.get(ch, 0.0)) for ch in CHANNELS]


def test_level_carrier_named_shifts_the_level_regardless_of_raw_centroid():
    """A `level carrier` row whose RAW centroid has a large `mase` value
    (level bleed-through, sec 37.7's whole motivation) must NOT be named
    from that value -- it is unconditionally "shifts the level".
    """
    profiles = np.array([
        _profile_row(mase=8.0, level=7.5),   # level carrier -- large mase is bleed-through
        _profile_row(trend=6.0),             # ordinary peer
    ])
    cleared = np.abs(profiles) >= 1.0
    peers = [0, 1]
    tags = ["level carrier", None]
    batch = _compose_batch(profiles, cleared, peers, tags=tags)
    assert batch[0]["name"] == "shifts the level"
    assert "mase" not in batch[0]["name"]
    assert batch[1]["name"] != "shifts the level"


def test_tags_none_reproduces_pretags_naming_bit_for_bit():
    """The default (`tags=None`, every caller before sec 37.7 P4) must
    compose EXACTLY the pre-P4 name -- confirmed by comparing against the
    same call with an all-`None` tags list, which takes the identical code
    path (`is_level_carrier` all `False` either way)."""
    profiles = np.array([_profile_row(mase=8.0, level=7.5), _profile_row(trend=6.0)])
    cleared = np.abs(profiles) >= 1.0
    peers = [0, 1]
    default = _compose_batch(profiles, cleared, peers)
    explicit_none = _compose_batch(profiles, cleared, peers, tags=[None, None])
    assert default == explicit_none


def test_level_carrier_excluded_from_dominant_clause_collision():
    """Both rows have `mase` as their largest raw channel, so if the level
    carrier (row 0) were NOT excluded from the dominant-lead computation, its
    dominant clause ("dominant raises mase") would collide with row 1's OWN
    dominant clause, and mechanism 3's shortest-unique-prefix would force row
    1 to append a contrast clause ("unusually raises trend") just to stay
    distinct from a name that is about to be thrown away anyway. Excluding
    the level carrier (`dominant_idx` forced `None`, so it renders "no
    channel clears its own null" pre-override) removes that collision, so
    row 1's OWN name needs no contrast clause at all.

    Planted regression checked: reverting `dominant_idx`'s level-carrier
    exclusion (computing it from the raw centroid regardless of `tags`, i.e.
    only `contrast_candidates` stays excluded) makes row 1's name grow a
    `trend` contrast clause it does not need -- confirmed to fail this
    test's `tagged[1]["name"] == "dominant raises mase"` assertion (the
    "untagged" name it then matches instead).
    """
    profiles = np.array([
        _profile_row(mase=8.0, level=7.9),   # level carrier
        _profile_row(mase=6.0, trend=5.5),   # real peer, also dominant on mase
    ])
    cleared = np.abs(profiles) >= 1.0
    peers = [0, 1]

    tagged = _compose_batch(profiles, cleared, peers, tags=["level carrier", None])
    untagged = _compose_batch(profiles, cleared, peers, tags=[None, None])

    assert tagged[0]["name"] == "shifts the level"
    assert tagged[1]["name"] == "dominant raises mase"
    assert tagged[1]["name"] != untagged[1]["name"]


# ---------------------------------------------------------------------------
# 3. End-to-end: `run_concept_atlas` writes the new fields and the atlas's
#    `name` for a level-carrier concept.
# ---------------------------------------------------------------------------

def test_run_concept_atlas_writes_causal_tags_end_to_end(tmp_path):
    """Two models, each contributing 3 members whose 9-channel profile is a
    pure, large `level` push (so they cluster into one concept) and whose
    `level_share`/`shape_channels` mark them as level carriers on disk --
    exactly `sae/response.py`'s own artifact shape. The written
    `concept_atlas.json` must carry `causal_tag`, `level_share_median`,
    `n_members_with_level_share` and name the concept "shifts the level",
    while every field `test_run_concept_atlas_is_deterministic` already
    checks (`n_features`, `n_assigned`, cluster membership) is untouched.
    """
    cfg = _cfg(str(tmp_path), n_null=20)
    cfg.concepts.level_share_threshold = 0.9

    def _level_candidate(feature, value, level_share_val):
        cand = _candidate(feature, {"level": value})
        cand["level_share"] = level_share_val
        cand["level_share_reason"] = ""
        cand["shape_channels"] = {ch: {"null_p95": 1.0, "signed_effect": 0.0,
                                       "clears_null": False} for ch in CHANNELS}
        cand["n_shape_channels_clearing"] = 0
        return cand

    cands_a = [_level_candidate(i, 8.0, 0.97) for i in range(3)]
    cands_b = [_level_candidate(i, 7.5, 0.93) for i in range(3)]
    _write_ablation(tmp_path, "A", "L1", cands_a)
    _write_ablation(tmp_path, "B", "L1", cands_b)

    out = run_concept_atlas(tmp_path, cfg)
    assert out["n_features"] == 6
    assert len(out["concepts"]) == 1
    concept = out["concepts"][0]
    assert concept["causal_tag"] == "level carrier"
    assert concept["level_share_median"] == pytest.approx(0.95)
    assert concept["n_members_with_level_share"] == 6
    assert concept["name"] == "shifts the level"

    on_disk = load_json(tmp_path / "sae" / "concept_atlas.json")
    assert on_disk["concepts"][0]["causal_tag"] == "level carrier"
    assert on_disk["params"]["level_share_threshold"] == pytest.approx(0.9)
