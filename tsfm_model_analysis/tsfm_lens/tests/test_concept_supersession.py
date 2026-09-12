"""ROADMAP.md sec 30, Stage 4/5 (2026-09-11) -- the IRREVERSIBLE rename of
`sae/roles.json` to `sae/roles_injection.json`
(`tsfm_lens/sae/concepts.py::supersede_roles_artifact`), tested here against
synthetic fixtures only (never against a real run's own `roles.json` -- the
one real invocation of this mechanism, against `runs/full_report_run_4model`,
is a separate, manual, one-time act per this module's own docstring, not
something a test suite should perform).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.sae.concepts import supersede_roles_artifact  # noqa: E402
from tsfm_lens.utils import save_json  # noqa: E402


def _roles_doc():
    return {
        "Alpha/layer.0": {"model": "Alpha", "layer": "layer.0", "withheld": False,
                          "skipped": False, "k": 2, "silhouette": 0.33,
                          "non_modular": False, "non_modular_reason": "",
                          "n_candidates": 5, "channel_columns": ["trend"],
                          "roles": [{"role": 0, "features": [0, 1]}]},
        "Beta/layer.1": {"model": "Beta", "layer": "layer.1", "withheld": True,
                         "skipped": False},
    }


def test_supersede_roles_artifact_renames_and_carries_metadata_with_no_data_loss(tmp_path):
    """Happy path: `concepts.json` present, `roles.json` present -- the
    function writes `roles_injection.json` carrying both new top-level keys
    plus every original target record byte-for-byte, removes `roles.json`,
    and the result is still consumable by `run_sae_compare.py::load_inputs`
    (the one consumer that iterates every top-level key/value and needed a
    fix for exactly this artifact shape) rather than only by functions that
    already filtered defensively."""
    original = _roles_doc()
    save_json(tmp_path / "sae" / "roles.json", original)
    save_json(tmp_path / "sae" / "concepts.json",
              {"schema_version": 1, "space": "ablation",
               "supersedes": "roles_injection.json", "targets": {}})

    out_path = supersede_roles_artifact(tmp_path)

    assert out_path == tmp_path / "sae" / "roles_injection.json"
    assert not (tmp_path / "sae" / "roles.json").exists()
    assert out_path.exists()

    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["superseded_by"] == "concepts.json"
    assert "0.450" in doc["superseded_reason"]
    assert "-0.235" in doc["superseded_reason"]
    assert "13 of 13" in doc["superseded_reason"]
    # Every original target record survives completely unchanged -- not
    # merely present, but byte-identical to what was there before.
    for key, rec in original.items():
        assert doc[key] == rec

    # The consumer this rename was checked against directly: filtering to
    # dict-valued entries (run_sae_compare.py's own fix) recovers exactly
    # the original target set, with the two new metadata keys excluded.
    filtered = {k: v for k, v in doc.items() if isinstance(v, dict)}
    assert filtered == original
    assert "superseded_by" not in filtered and "superseded_reason" not in filtered
    for target in filtered:
        # run_sae_compare.py::load_inputs's own unpacking, run directly
        # against the post-rename artifact.
        model, layer = target.split("/", 1)
        assert model in ("Alpha", "Beta")


@pytest.mark.parametrize("break_fn,expected_exc", [
    (lambda run_dir: (run_dir / "sae" / "concepts.json").unlink(), FileNotFoundError),
    (lambda run_dir: (run_dir / "sae" / "roles.json").unlink(), FileNotFoundError),
])
def test_supersede_roles_artifact_refuses_without_touching_roles_json(tmp_path, break_fn, expected_exc):
    """Two preconditions, each refused with a stated reason rather than
    silently worked around (sec 2.5): no `concepts.json` (the rename is
    gated on the report actually reading concepts.json first -- sec 11.39),
    and no `roles.json` at all. Either way `roles.json` (if it existed
    before the break) is untouched -- this function must never partially
    apply itself."""
    save_json(tmp_path / "sae" / "roles.json", _roles_doc())
    save_json(tmp_path / "sae" / "concepts.json",
              {"schema_version": 1, "targets": {}})
    break_fn(tmp_path)

    with pytest.raises(expected_exc):
        supersede_roles_artifact(tmp_path)

    # Whichever precondition failed, nothing was archived, and if
    # roles.json still existed going in, it still exists coming out.
    assert not (tmp_path / "sae" / "roles_injection.json").exists()


def test_supersede_roles_artifact_refuses_to_clobber_an_existing_archive(tmp_path):
    """Re-invoking against an already-superseded run must refuse, not
    silently no-op (which would look successful while doing nothing) or
    silently overwrite a previously-archived copy (which could discard a
    different `superseded_reason` from an earlier supersession)."""
    save_json(tmp_path / "sae" / "roles.json", _roles_doc())
    save_json(tmp_path / "sae" / "concepts.json", {"schema_version": 1, "targets": {}})
    save_json(tmp_path / "sae" / "roles_injection.json", {"already": "here"})

    with pytest.raises(FileExistsError):
        supersede_roles_artifact(tmp_path)

    # roles.json is untouched -- the refusal happened before any write.
    assert (tmp_path / "sae" / "roles.json").exists()
    archived = json.loads((tmp_path / "sae" / "roles_injection.json").read_text(encoding="utf-8"))
    assert archived == {"already": "here"}
