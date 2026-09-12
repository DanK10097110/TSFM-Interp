"""The roles table's "also moves" column (ROADMAP.md sec 28.18).

The deterministic roles table named ONE channel per role -- the argmax --
and every surface beneath it therefore described one kind of change, while
`sae/roles.py::role_table` had measured the role's mean on all nine. On
`runs/full_report_run_4model` 55 of the 57 roles that license a channel at
all move two or more above their own null and 45 move four or more, so the
table was showing a seventh of what had been measured about the *kind* of
change each role makes -- which is exactly what the user's request for
descriptions "specific about the kind of change" is about.

Two properties are load-bearing here and each has its own negative:

  * the column WIDENS WHAT IS SHOWN AND NEVER THE TEST -- the 1.0-null-unit
    bar is the same one the dominant channel is read against, and a role
    whose own argmax is sub-null licenses nothing at all (sec 11.54's rule,
    which `_role_channels` enforces on the narrator's side and which a
    display column must not quietly contradict beside it);

  * `horizon_shape_*` gets NO DIRECTION. Its statistic is a mean absolute
    deviation (`sae/response.py::_horizon_shape`), so across every role of
    that same run both horizon channels are positive 88 of 88 times and
    negative 0 -- an up-arrow there is not merely unsupported, it is
    vacuous, and a column of uniform up-arrows reads as a finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.report.sae_roles import (  # noqa: E402
    _also_moves,
    model_roles_table,
    roles_summary_table,
)


def _role(means, dominant="trend", dom_val=3.0, **kw):
    r = {"role": 0, "name": "r", "n_atoms": 2, "features": [1, 2],
         "dominant_channel": dominant, "dominant_effect_null_units": dom_val,
         "channel_means_null_units": means, "clears_null": True}
    r.update(kw)
    return r


def test_every_other_above_null_channel_is_named_not_only_the_argmax():
    txt = _also_moves(_role({"trend": 3.0, "level": -2.0, "seasonal": 1.4}))
    assert "level" in txt and "seasonal" in txt
    # the argmax is the row's own `dominant channel` column; repeating it
    # here would double-count it as a second finding.
    assert "trend" not in txt


def test_channels_are_ordered_by_magnitude_not_by_dict_order():
    txt = _also_moves(_role({"trend": 3.0, "seasonal": 1.2, "level": -2.7}))
    assert txt.index("level") < txt.index("seasonal")


def test_a_sub_null_channel_is_excluded_even_when_others_clear():
    txt = _also_moves(_role({"trend": 3.0, "level": -2.0, "mase": 0.4}))
    assert "level" in txt and "mase" not in txt


def test_a_sub_null_dominant_licenses_nothing_however_large_the_others():
    """sec 11.54: if the role's own argmax is below its null then every
    channel is, and this column must not license one the narrator's packet
    refuses. Its wording must also not be the same as "nothing else moved"."""
    txt = _also_moves(_role({"trend": 0.4, "level": 9.9}, dom_val=0.4))
    assert "level" not in txt
    assert txt == "none above its null"


def test_argmax_clears_but_nothing_else_does_says_so_distinctly():
    txt = _also_moves(_role({"trend": 3.0, "level": 0.2}))
    assert txt == "—"
    assert txt != _also_moves(_role({"trend": 0.4, "level": 9.9}, dom_val=0.4))


def test_a_record_written_before_the_field_existed_says_not_recorded():
    """Three states, not two (sec 11.37): an empty cell for a role whose
    vector was never measured reads as "nothing else moved"."""
    r = _role({"trend": 3.0})
    r.pop("channel_means_null_units")
    assert _also_moves(r) == "not recorded"
    assert _also_moves(_role({})) == "not recorded"


def test_horizon_shape_gets_no_direction_arrow_but_signed_channels_do():
    txt = _also_moves(_role({"trend": 3.0, "horizon_shape_far": 2.2,
                             "level": -1.8}))
    assert "horizon_shape_far ↕ 2.2" in txt
    assert "horizon_shape_far ↑" not in txt and "horizon_shape_far ↓" not in txt
    # the neutral marker must be scoped to the magnitude channels, not
    # applied to every channel to be safe -- a signed channel's direction
    # is the whole reason the mean is kept signed.
    assert "level ↓ 1.8" in txt and "trend" not in txt


def test_an_undirected_channel_as_the_argmax_still_licenses_the_others():
    txt = _also_moves(_role({"horizon_shape_far": 4.0, "level": 2.0},
                            dominant="horizon_shape_far", dom_val=4.0))
    assert "level ↑ 2.0" in txt


def test_the_column_is_rendered_in_the_summary_table():
    df = roles_summary_table({"roles": [_role({"trend": 3.0, "level": -2.0})]})
    assert "also moves (× null p95)" in df.columns
    assert "level ↓ 2.0" in df["also moves (× null p95)"].iloc[0]
    # and it must sit beside, never replace, the fields sec 11.54 already
    # made legible as a trio.
    for col in ("dominant channel", "effect (× null p95)",
                "members clearing that channel"):
        assert col in df.columns


def test_the_column_is_on_the_consolidated_per_model_table():
    """`roles_summary_table` is per-TARGET; `model_roles_table` is the
    consolidated per-model table this column was fixed to also carry (sec
    11.48, and sec 26 C's own "report.py had only ever consumed the features
    half" one surface over).

    `report.py` no longer calls `model_roles_table` at all as of ROADMAP.md
    sec 30 (Stage 4, 2026-09-11): the SAE report section now reads
    `sae/concepts.json` via `report/sae_concepts.py::sae_concepts_block`
    (ablation-space clustering) instead of `sae/roles.json` via the retired
    `_sae_roles_block` (injection-space clustering), per sec 30.1's measured
    result that concepts cluster decisively better at every target checked.
    `sae_roles.py` itself is NOT retired -- `model_roles_table` stays a
    unit-tested pure table builder, exercised here directly rather than
    through a renderer that no longer calls it. See
    `tests/test_sae_concept_report.py` for the equivalent end-to-end
    coverage of the new `sae_concepts_block` renderer, and its own
    `_profile_html` for how a concept card names every channel clearing its
    own null (not only the dominant one) -- the same "also moves" concern
    this column exists for, carried into the concepts renderer directly
    rather than through this table.
    """
    rec = {"roles": [_role({"trend": 3.0, "level": -2.0})]}
    df, _dropped = model_roles_table({"M/l.0": rec}, "M", ["M/l.0"])
    assert "also moves (× null p95)" in df.columns
    assert "level ↓ 2.0" in df["also moves (× null p95)"].iloc[0]


# --- the same vacuous arrow in the role NAME (ROADMAP.md sec 28.18) -----

def test_a_role_NAME_does_not_carry_a_direction_the_statistic_cannot_have():
    """`sae/roles.py::derive_role_name` had the same defect the table's
    `also moves` column was fixed for: `horizon_shape_*` is a mean ABSOLUTE
    deviation, so its sign is 88/0 positive across the live run and an
    arrow on it reads as a finding it cannot be."""
    from tsfm_lens.sae.roles import _arrow

    assert _arrow("horizon_shape_near", 1) == "↕"
    assert _arrow("horizon_shape_far", -1) == "↕"
    # ...while a genuinely signed channel keeps its direction in both senses.
    assert _arrow("trend", 1) == "↑"
    assert _arrow("level", -1) == "↓"


def test_the_unsigned_set_comes_from_the_narrators_table_not_a_second_copy():
    """The fallback literal is byte-identical to what the import yields
    today, so a silently-failing import is invisible (sec 11.53's shape).
    This pins that the import is the live path -- if `describe` ever stops
    exporting it, this fails instead of the two tables quietly diverging."""
    from tsfm_lens.sae import roles as R

    assert hasattr(R, "_CHANNEL_VERB"), (
        "roles.py fell back to its own literal -- the narrator's "
        "CHANNEL_VERB no longer imports, so the two surfaces can now "
        "disagree about which channel has a direction")
    assert R._UNSIGNED_CHANNELS == frozenset(R._CHANNEL_VERB)
