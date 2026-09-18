"""Tests for `ROADMAP.md` sec 34.6 Item E1: a home for contributed adapters.

Mutates the real, shared `tsfm_lens.models.CONTRIB_REGISTRY` / `discovery_errors`
module-level state in a few tests (planting a temp file in the real
`models/contrib/` directory and re-running discovery) -- every such test
snapshots and restores that state in a `finally` block, and deletes the
planted file, so no other test in the session observes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tsfm_lens.models as models_pkg
from tsfm_lens.config import DataConfig, ModelConfig
from tsfm_lens.models import contrib
from tsfm_lens.models.base import CapabilityUnavailable, NotTimeLocalized

CONTRIB_DIR = Path(models_pkg.__file__).resolve().parent / "contrib"


def _plant(filename: str, source: str) -> Path:
    path = CONTRIB_DIR / filename
    path.write_text(source, encoding="utf-8")
    return path


class _RegistrySnapshot:
    """Save/restore the two shared module-level dicts/lists in place, so a
    test that mutates them (by calling `models_pkg._discover_contrib()`
    again) cannot leak state into a test that runs after it."""

    def __enter__(self):
        self._registry = dict(models_pkg.CONTRIB_REGISTRY)
        self._errors = list(models_pkg.discovery_errors)
        return self

    def __exit__(self, *exc):
        models_pkg.CONTRIB_REGISTRY.clear()
        models_pkg.CONTRIB_REGISTRY.update(self._registry)
        models_pkg.discovery_errors.clear()
        models_pkg.discovery_errors.extend(self._errors)


# --- T-E1.1 -----------------------------------------------------------------

def test_import_tsfm_lens_models_imports_no_heavy_library():
    """`import tsfm_lens.models` must not pull in `chronos` or `timesfm` --
    those are real, heavy, model-library imports that stay deferred to each
    adapter's own `load()` (`CLAUDE.md`'s general discipline, extended here
    to contrib). Run in a subprocess so an unrelated test in the same pytest
    session that happens to load a real checkpoint elsewhere cannot make
    this pass for the wrong reason."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import tsfm_lens.models; "
         "heavy = [m for m in sys.modules if m == 'chronos' or m == 'timesfm' "
         "or m.startswith('chronos.') or m.startswith('timesfm.')]; "
         "assert not heavy, heavy; print('OK')"],
        cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_discovery_does_not_import_a_broken_contrib_file():
    """A contrib file that raises at import time (a bad import, or any
    top-level statement) must not prevent discovery from finding its
    ADAPTER_NAME/ADAPTER_CLASS -- discovery is a pure AST scan and never
    executes the file. Importing it (via `import_contrib_class`) is a
    separate step, and IS where the failure surfaces."""
    path = _plant("_test_e11_broken.py",
                  'ADAPTER_NAME = "e11_broken"\n'
                  'ADAPTER_CLASS = "BrokenAdapter"\n'
                  'raise RuntimeError("this file is deliberately broken at import time")\n')
    try:
        registry, errors = contrib.discover_contrib_adapters()
        assert registry["e11_broken"] == ("tsfm_lens.models.contrib._test_e11_broken",
                                          "BrokenAdapter")
        assert not any(e["file"] == "_test_e11_broken.py" for e in errors)
        with pytest.raises(RuntimeError, match="deliberately broken"):
            contrib.import_contrib_class(*registry["e11_broken"])
    finally:
        path.unlink()


# --- T-E1.2 -----------------------------------------------------------------

def test_a_contrib_file_missing_adapter_name_appears_in_discovery_errors():
    path = _plant("_test_e12_missing_name.py", 'ADAPTER_CLASS = "SomeAdapter"\n')
    try:
        registry, errors = contrib.discover_contrib_adapters()
        assert "_test_e12_missing_name.py" not in [n for n in registry]
        matches = [e for e in errors if e["file"] == "_test_e12_missing_name.py"]
        assert len(matches) == 1
        assert "ADAPTER_NAME" in matches[0]["error"]
    finally:
        path.unlink()


def test_a_contrib_file_missing_adapter_class_appears_in_discovery_errors():
    path = _plant("_test_e12_missing_class.py", 'ADAPTER_NAME = "e12_missing_class"\n')
    try:
        registry, errors = contrib.discover_contrib_adapters()
        assert "e12_missing_class" not in registry
        matches = [e for e in errors if e["file"] == "_test_e12_missing_class.py"]
        assert len(matches) == 1
        assert "ADAPTER_CLASS" in matches[0]["error"]
    finally:
        path.unlink()


def test_an_unparseable_contrib_file_appears_in_discovery_errors_by_filename():
    path = _plant("_test_e12_syntax_error.py", "def broken(:\n")
    try:
        registry, errors = contrib.discover_contrib_adapters()
        matches = [e for e in errors if e["file"] == "_test_e12_syntax_error.py"]
        assert len(matches) == 1
        assert "SyntaxError" in matches[0]["error"]
    finally:
        path.unlink()


# --- T-E1.3 -----------------------------------------------------------------

def test_a_contrib_name_colliding_with_a_built_in_is_refused_naming_both():
    """Confirmed to discriminate against a version that lets registration
    order decide silently: without the collision check, re-running discovery
    would simply overwrite CONTRIB_REGISTRY['mock_patch'], and `build_adapter`
    would return the wrong class with no error at all (`CLAUDE.md` sec
    11.34's tie-break lesson)."""
    with _RegistrySnapshot():
        path = _plant("_test_e13_collision.py",
                      'ADAPTER_NAME = "mock_patch"\n'
                      'ADAPTER_CLASS = "WouldNeverWork"\n')
        try:
            models_pkg._discover_contrib()
            assert "mock_patch" not in models_pkg.CONTRIB_REGISTRY, (
                "a contrib file must never be allowed to shadow a built-in adapter name")
            collision_errors = [e for e in models_pkg.discovery_errors
                                if e.get("collides_with") == "mock_patch"]
            assert collision_errors, "collision must be recorded in discovery_errors"
            assert "_test_e13_collision.py" in collision_errors[-1]["file"]

            # build_adapter must still resolve 'mock_patch' to the real built-in,
            # not raise and not silently return the contrib class.
            mcfg = ModelConfig(name="m", adapter="mock_patch")
            dcfg = DataConfig()
            adapter = models_pkg.build_adapter(mcfg, dcfg, torch.device("cpu"), torch.float32)
            assert type(adapter).__name__ == "MockPatchAdapter"

            # a genuinely unresolvable colliding name (removed from both
            # registries) must raise naming both the built-in and the file.
            del models_pkg.ADAPTERS["mock_patch"]
            try:
                with pytest.raises(ValueError, match="collides with a built-in"):
                    models_pkg.build_adapter(ModelConfig(name="m2", adapter="mock_patch"),
                                             dcfg, torch.device("cpu"), torch.float32)
            finally:
                from tsfm_lens.models.mock import MockPatchAdapter
                models_pkg.ADAPTERS["mock_patch"] = MockPatchAdapter
        finally:
            path.unlink()


# --- T-E1.4 -----------------------------------------------------------------

def test_template_adapter_is_importable_and_its_stubs_raise_capability_unavailable():
    from tsfm_lens.models.TEMPLATE_adapter import ADAPTER_CLASS, ADAPTER_NAME, TemplateAdapter

    assert ADAPTER_NAME and isinstance(ADAPTER_NAME, str)
    assert ADAPTER_CLASS == "TemplateAdapter"

    adapter = TemplateAdapter(ModelConfig(name="tmpl", adapter="template"),
                              DataConfig(context_len=64, horizon=8),
                              torch.device("cpu"), torch.float32)
    for call in (lambda: adapter.module,
                lambda: adapter.prepare(np.zeros((2, 64), dtype=np.float32)),
                lambda: adapter.forward(None),
                adapter.token_time_spans):
        try:
            call()
        except NotTimeLocalized:
            pytest.fail("template stub raised NotTimeLocalized, not CapabilityUnavailable -- "
                       "the two mean different things (CLAUDE.md sec 6.2)")
        except CapabilityUnavailable:
            pass
        else:
            pytest.fail("expected CapabilityUnavailable, nothing was raised")

    # As shipped it IS a working tier-0 mock (MockBlackBoxAdapter) -- real
    # forecasting, not a stub.
    assert adapter.capability_tier() == 0
    adapter.ensure_loaded()
    pred = adapter.predict(np.zeros((2, 64), dtype=np.float32), 8, [0.1, 0.5, 0.9])
    assert pred["point"].shape == (2, 8)
    assert np.all(np.isfinite(pred["point"]))
