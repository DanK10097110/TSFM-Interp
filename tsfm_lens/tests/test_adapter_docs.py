"""Tests for `adapter_docs.py`/`render_adapter_docs.py` (`ROADMAP.md` sec 34.6 Item E2).

Everything here reads the real, live registry (`tsfm_lens.models.ADAPTERS`)
rather than a synthetic stand-in, since the whole point of this item is that
the doc is derived from that exact registry and cannot drift from it
(`CLAUDE.md` sec 11.34). The one load-bearing negative (T-E2.2) registers a
real throwaway adapter, confirms `--check` correctly reports staleness, then
removes it and confirms `--check` passes again -- proving the staleness gate
is live, not merely present.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from tsfm_lens import adapter_docs
from tsfm_lens.models import ADAPTERS
from tsfm_lens.models.base import TIER_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ADAPTERS_MD = _REPO_ROOT / "tsfm_lens" / "ADAPTERS.md"


def test_every_registered_built_in_adapter_appears_with_its_derived_tier():
    """T-E2.1: the doc names every adapter in the registry, tier included."""
    rows, _ = adapter_docs.build_adapter_table()
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) >= set(ADAPTERS)
    for name, adapter_cls in ADAPTERS.items():
        row = by_name[name]
        assert row["tier"] == adapter_cls.capability_tier()
        assert row["tier_name"] == TIER_NAMES[adapter_cls.capability_tier()]
        assert row["contrib"] is False


def test_generated_markdown_contains_every_adapter_name_and_no_placeholder():
    doc = adapter_docs.render_markdown(*adapter_docs.build_adapter_table())
    for name in ADAPTERS:
        assert f"`{name}`" in doc
    assert "TEMPLATE_adapter.py" in doc
    assert "--check-adapter" in doc


def test_stages_unlocked_by_tier_is_monotone_and_matches_pipeline_min_tier():
    """T-E2.3 half one: cross-check against pipeline.py's own tier map directly,
    not by re-deriving a second copy of it by hand."""
    from tsfm_lens.pipeline import _STAGE_MIN_TIER, stage_names

    for tier in (0, 1, 2, 3):
        unlocked = set(adapter_docs.stages_unlocked_by_tier(tier))
        expected = {s for s in stage_names() if _STAGE_MIN_TIER.get(s, 1) <= tier}
        assert unlocked == expected
    # Monotone: everything unlocked at tier t is still unlocked at tier t+1.
    for t in (0, 1, 2):
        assert set(adapter_docs.stages_unlocked_by_tier(t)) <= set(
            adapter_docs.stages_unlocked_by_tier(t + 1))


def test_derived_tiers_match_the_test_suites_own_expected_tier_fixture():
    """T-E2.3 half two: cross-check against test_capability_tiers.py's
    EXPECTED_TIERS, so a drift between the two test files' assumptions
    about the registry surfaces immediately rather than only in one of them."""
    from tests.test_capability_tiers import EXPECTED_TIERS

    rows, _ = adapter_docs.build_adapter_table()
    by_name = {r["name"]: r for r in rows}
    for name, expected_tier in EXPECTED_TIERS.items():
        assert by_name[name]["tier"] == expected_tier


def test_real_adapters_declare_a_default_checkpoint_string():
    """`default_checkpoint` finds the literal every real (non-mock) adapter
    declares, and correctly reports None for adapters with no default."""
    from tsfm_lens.models.chronos_adapter import ChronosAdapter
    from tsfm_lens.models.timesfm_adapter import TimesFMAdapter
    from tsfm_lens.models.generic_hf_adapter import GenericHFAdapter
    from tsfm_lens.models.mock import MockPatchAdapter

    assert adapter_docs.default_checkpoint(ChronosAdapter) == "amazon/chronos-t5-small"
    assert adapter_docs.default_checkpoint(TimesFMAdapter) == "google/timesfm-2.5-200m-pytorch"
    assert adapter_docs.default_checkpoint(GenericHFAdapter) is None
    assert adapter_docs.default_checkpoint(MockPatchAdapter) is None


def test_check_adapter_docs_cli_reports_stale_then_fresh_around_a_real_registration():
    """T-E2.2, the load-bearing negative: `--check` must actually discriminate.

    Registers a real, throwaway contrib adapter (subclassing an existing
    working mock, exactly like `TEMPLATE_adapter.py` is designed to be
    copied), confirms the on-disk ADAPTERS.md is now stale (`--check` exits
    1 and names it), regenerates it, confirms `--check` now passes, then
    removes the throwaway file, regenerates again, and confirms the repo's
    real ADAPTERS.md is restored byte-for-byte -- so this test cannot leave
    the checked-in doc altered.
    """
    contrib_dir = _REPO_ROOT / "tsfm_lens" / "models" / "contrib"
    planted = contrib_dir / "_test_e22_throwaway_adapter.py"
    original_doc = _ADAPTERS_MD.read_text(encoding="utf-8") if _ADAPTERS_MD.exists() else None
    render_script = str(_REPO_ROOT / "render_adapter_docs.py")

    def run_check() -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, render_script, "--check"],
                              cwd=str(_REPO_ROOT), capture_output=True, text=True)

    try:
        before = run_check()
        assert before.returncode == 0, before.stdout + before.stderr

        planted.write_text(
            "from tsfm_lens.models.mock import MockPatchAdapter\n\n"
            "ADAPTER_NAME = \"_test_e22_throwaway\"\n"
            "ADAPTER_CLASS = \"ThrowawayAdapter\"\n\n\n"
            "class ThrowawayAdapter(MockPatchAdapter):\n"
            "    pass\n",
            encoding="utf-8")

        after_plant = run_check()
        assert after_plant.returncode == 1
        assert "stale" in (after_plant.stdout + after_plant.stderr).lower()

        regenerate = subprocess.run([sys.executable, render_script],
                                    cwd=str(_REPO_ROOT), capture_output=True, text=True)
        assert regenerate.returncode == 0, regenerate.stdout + regenerate.stderr
        assert "_test_e22_throwaway" in _ADAPTERS_MD.read_text(encoding="utf-8")

        after_regen = run_check()
        assert after_regen.returncode == 0, after_regen.stdout + after_regen.stderr
    finally:
        planted.unlink(missing_ok=True)
        cache = contrib_dir / "__pycache__"
        if cache.exists():
            for f in cache.glob("_test_e22_throwaway_adapter*"):
                f.unlink(missing_ok=True)
        subprocess.run([sys.executable, render_script], cwd=str(_REPO_ROOT),
                       capture_output=True, text=True)
        if original_doc is not None:
            assert _ADAPTERS_MD.read_text(encoding="utf-8") == original_doc


def test_broken_contrib_file_is_named_in_error_rows_not_silently_dropped():
    """A contrib file with an import-time error must show up by filename,
    matching Item E1's discovery-errors contract rather than vanishing."""
    contrib_dir = _REPO_ROOT / "tsfm_lens" / "models" / "contrib"
    planted = contrib_dir / "_test_e22_broken_adapter.py"
    try:
        planted.write_text(
            "ADAPTER_NAME = \"_test_e22_broken\"\n"
            "ADAPTER_CLASS = \"BrokenAdapter\"\n\n"
            "raise RuntimeError(\"deliberately broken for test_adapter_docs\")\n",
            encoding="utf-8")
        from tsfm_lens.models import contrib as contrib_pkg
        registry, discovery_errors = contrib_pkg.discover_contrib_adapters()
        assert "_test_e22_broken" in registry

        import tsfm_lens.models as models_pkg
        old_registry, old_errors = dict(models_pkg.CONTRIB_REGISTRY), list(models_pkg.discovery_errors)
        models_pkg.CONTRIB_REGISTRY.clear()
        models_pkg.CONTRIB_REGISTRY.update(registry)
        models_pkg.discovery_errors.clear()
        models_pkg.discovery_errors.extend(discovery_errors)
        try:
            _, error_rows = adapter_docs.build_adapter_table()
            names = {e["file"] for e in error_rows}
            assert "_test_e22_broken_adapter.py" in names
        finally:
            models_pkg.CONTRIB_REGISTRY.clear()
            models_pkg.CONTRIB_REGISTRY.update(old_registry)
            models_pkg.discovery_errors.clear()
            models_pkg.discovery_errors.extend(old_errors)
    finally:
        planted.unlink(missing_ok=True)


def test_adapters_md_on_disk_is_not_stale_right_now():
    """T-E2.4-adjacent acceptance check: the committed doc matches the
    registry as of this test run -- a real regression signal, not a tautology,
    since `build_adapter_table` reads live class objects while the file on
    disk is whatever text `render_adapter_docs.py` last wrote."""
    assert _ADAPTERS_MD.exists()
    current = _ADAPTERS_MD.read_text(encoding="utf-8")
    fresh = adapter_docs.render_markdown(*adapter_docs.build_adapter_table())
    assert current == fresh


def test_build_adapter_table_loads_no_checkpoint():
    """T-E2.4: this must be fast and side-effect-free -- no torch.load, no
    HF download. A generous wall-clock ceiling stands in for 'no checkpoint
    was loaded', since a real load would take vastly longer than this."""
    start = time.monotonic()
    adapter_docs.build_adapter_table()
    assert time.monotonic() - start < 5.0
