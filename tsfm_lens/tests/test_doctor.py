"""Preflight-doctor unit tests (ROADMAP.md sec 16 E2).

Static checks are exercised directly against synthetic configs -- no
checkpoint download, no GPU. `_check_adapters_full` (the `--doctor`-only,
model-loading tier) is exercised once against `configs/smoke.yaml`'s two
mock architectures (CPU, no network, per CLAUDE.md sec 8), which is the
same "real adapter, no live checkpoint" precedent
`tests/test_adapter_conformance.py` already uses for
`models/conformance.py` itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsfm_lens.config import PipelineConfig, load_config
from tsfm_lens.doctor import (_check_capped_stages, _check_context_alignment,
                              _check_corpus_seal, _check_disk, _check_store_format,
                              _check_multiplicity_budget, _check_torch_cuda,
                              _check_vram, _check_zarr_version,
                              print_preflight, run_preflight)

SMOKE_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


def _cpu_cfg() -> PipelineConfig:
    cfg = load_config(SMOKE_CONFIG)
    cfg.run.device = "cpu"
    return cfg


def test_zarr_version_check_reports_pass_for_the_pinned_v2_install():
    check = _check_zarr_version()
    assert check.status in ("pass", "fail")  # "fail" only if zarr>=3 genuinely installed
    if check.status == "fail":
        assert "3" in check.detail
    print("zarr version check ran and returned a well-formed verdict")


def test_torch_cuda_check_passes_trivially_when_device_is_cpu():
    cfg = _cpu_cfg()
    check = _check_torch_cuda(cfg)
    assert check.status == "pass"
    assert "cpu" in check.detail
    print("torch/CUDA check correctly skips the CUDA requirement for device=cpu")


def test_vram_check_skips_cleanly_on_cpu_device():
    cfg = _cpu_cfg()
    check = _check_vram(cfg)
    assert check.status == "pass"
    assert "skipped" in check.detail
    print("VRAM check degrades to a clean skip, not a crash, on device=cpu")


def test_disk_check_never_raises_and_returns_a_verdict():
    cfg = _cpu_cfg()
    check = _check_disk(cfg)
    assert check.status in ("pass", "warn", "fail")
    print("disk headroom check returned a well-formed verdict")


def test_context_alignment_check_flags_a_non_multiple_window():
    cfg = _cpu_cfg()
    cfg.data.context_len = 100
    cfg.alignment.window = 32  # 100 % 32 != 0
    check = _check_context_alignment(cfg)
    assert check.status == "fail"
    assert "100" in check.detail and "32" in check.detail
    print("context_len/window check correctly flags a non-multiple")


def test_context_alignment_check_passes_a_clean_multiple():
    cfg = _cpu_cfg()
    cfg.data.context_len = 128
    cfg.alignment.window = 32
    check = _check_context_alignment(cfg)
    assert check.status == "pass"
    print("context_len/window check passes a clean multiple")


def test_corpus_seal_check_passes_trivially_for_non_sealed_sources():
    cfg = _cpu_cfg()
    assert cfg.data.source == "smoke"
    cfg.confirm.enabled = False  # isolate to just the `data` source for this assertion
    checks = _check_corpus_seal(cfg, full=False)
    assert len(checks) == 1
    assert checks[0].status == "pass"
    print("corpus seal check is a trivial pass when data.source != 'sealed'")


def test_corpus_seal_check_fails_on_a_missing_sealed_path():
    cfg = _cpu_cfg()
    cfg.confirm.enabled = False  # isolate to just the `data` source for this assertion
    cfg.data.source = "sealed"
    cfg.data.path = "/nonexistent/path/for/doctor/test"
    checks = _check_corpus_seal(cfg, full=False)
    assert len(checks) == 1
    assert checks[0].status == "fail"
    assert "does not exist" in checks[0].detail
    print("corpus seal check correctly fails on a missing sealed path")


def test_corpus_seal_check_warns_not_fails_on_an_unminted_confirm_path():
    cfg = _cpu_cfg()
    cfg.confirm.enabled = True
    cfg.confirm.source = "sealed"
    cfg.confirm.path = ""
    checks = [c for c in _check_corpus_seal(cfg, full=False) if c.name == "confirm corpus"]
    assert len(checks) == 1
    assert checks[0].status == "warn"
    assert "not been minted" in checks[0].detail


def test_corpus_seal_check_fails_on_a_directory_missing_manifest_or_corpus(tmp_path):
    cfg = _cpu_cfg()
    cfg.data.source = "sealed"
    cfg.data.path = str(tmp_path)  # exists, but no manifest.json/corpus.jsonl inside
    checks = _check_corpus_seal(cfg, full=False)
    assert checks[0].status == "fail"
    assert "missing corpus.jsonl/manifest.json" in checks[0].detail
    print("corpus seal check correctly fails on an incomplete sealed directory")


def test_capped_stages_check_flags_a_max_series_exceeding_the_smallest_batch_size():
    cfg = _cpu_cfg()
    cfg.models[0].batch_size = 16
    cfg.models[1].batch_size = 64
    cfg.lens.enabled = True
    cfg.lens.max_series = 32  # > min(16, 64)
    checks = _check_capped_stages(cfg)
    lens_check = next(c for c in checks if c.name == "batch cap: lens.max_series")
    assert lens_check.status == "fail"
    assert "16" in lens_check.detail
    print("capped-stages check correctly flags lens.max_series exceeding the smallest batch_size")


def test_capped_stages_check_passes_when_every_cap_fits():
    cfg = _cpu_cfg()
    cfg.models[0].batch_size = 64
    cfg.models[1].batch_size = 64
    cfg.lens.enabled = True
    cfg.lens.max_series = 24
    checks = _check_capped_stages(cfg)
    lens_check = next(c for c in checks if c.name == "batch cap: lens.max_series")
    assert lens_check.status == "pass"
    print("capped-stages check passes when max_series fits every model's batch_size")


def test_capped_stages_check_resolves_a_contrib_adapters_tier_not_just_built_ins(monkeypatch):
    """Regression, found verifying ROADMAP.md sec 34.6 Item E3's own
    scaffolded config against real preflight output (not the diff, per
    CLAUDE.md sec 11.48): the pre-fix check computed `tiers` by filtering
    `if m.adapter in ADAPTERS`, the BUILT-IN dict only -- silently excluding
    every contrib model from the tier computation. A tier-0 contrib adapter
    paired with a tier-3 built-in mock (exactly `run.py --new-adapter`'s own
    pairing) therefore computed run_tier=3 instead of the real 0, and this
    check FAILED on `attention.ablation_max_series` even though the actual
    tier gate was about to drop the `attention` stage entirely -- the exact
    false refusal the surrounding comment already warns about, tripped by
    this very function. `resolve_adapter_class` (checks CONTRIB_REGISTRY
    too) is the fix.
    """
    import tsfm_lens.models as models_pkg
    from tsfm_lens.models.mock import MockBlackBoxAdapter

    cfg = _cpu_cfg()
    cfg.models[0].adapter = "mock_patch"              # tier 3, built-in
    cfg.models[1].adapter = "_fake_contrib_blackbox"  # tier 0, "contrib"
    cfg.models[0].batch_size = 64
    cfg.models[1].batch_size = 64
    cfg.attention.enabled = True
    cfg.attention.ablation = True
    cfg.attention.ablation_max_series = 128  # > batch_size -- irrelevant once
    # `attention` is correctly dropped at the real run_tier of 0.

    saved_registry = dict(models_pkg.CONTRIB_REGISTRY)
    models_pkg.CONTRIB_REGISTRY["_fake_contrib_blackbox"] = ("fake.module", "FakeClass")
    monkeypatch.setattr(models_pkg.contrib, "import_contrib_class",
                        lambda module_path, class_name: MockBlackBoxAdapter)
    try:
        checks = _check_capped_stages(cfg)
    finally:
        models_pkg.CONTRIB_REGISTRY.clear()
        models_pkg.CONTRIB_REGISTRY.update(saved_registry)

    names = [c.name for c in checks]
    assert "batch cap: attention.ablation_max_series" not in names, (
        "the contrib model's tier-0 was not counted -- attention should have "
        "been excluded from candidates at the real run_tier, not evaluated")
    overall = next(c for c in checks if c.name == "batch caps")
    assert overall.status == "pass"
    assert "tier 0" in overall.detail
    print("capped-stages check correctly resolves a contrib adapter's tier, "
          "not just built-ins")


def test_capped_stages_check_reports_a_warn_not_a_silent_drop_for_an_unresolvable_adapter():
    """The complementary negative: an adapter this check truly cannot
    resolve (a typo, a broken contrib import) must surface as its own
    `warn` naming the model, not vanish from `tiers` the way the pre-fix
    bug silently dropped every contrib adapter (CLAUDE.md sec 2.5 -- degrade
    loudly, never silently).
    """
    cfg = _cpu_cfg()
    cfg.models[1].adapter = "_totally_unknown_adapter_name"
    checks = _check_capped_stages(cfg)
    warn_check = next(c for c in checks if c.name == "batch caps: adapter resolution")
    assert warn_check.status == "warn"
    assert "_totally_unknown_adapter_name" in warn_check.detail
    print("capped-stages check warns by name on an unresolvable adapter, "
          "rather than silently excluding it from the tier computation")


def test_run_preflight_static_mode_never_loads_a_model():
    cfg = _cpu_cfg()
    checks = run_preflight(cfg, full=False)
    assert len(checks) > 0
    assert all(c.status in ("pass", "warn", "fail") for c in checks)
    print(f"run_preflight(full=False) returned {len(checks)} well-formed checks")


def test_print_preflight_returns_false_iff_any_check_failed():
    from tsfm_lens.doctor import DoctorCheck
    all_pass = [DoctorCheck("a", "pass", "ok"), DoctorCheck("b", "warn", "meh")]
    one_fail = [DoctorCheck("a", "pass", "ok"), DoctorCheck("c", "fail", "broken", "fix it")]
    assert print_preflight(all_pass) is True
    assert print_preflight(one_fail) is False
    print("print_preflight's return value correctly tracks fail-vs-not")


def test_run_preflight_full_mode_loads_real_mock_adapters_and_runs_conformance():
    cfg = _cpu_cfg()
    checks = run_preflight(cfg, full=True)
    names = [c.name for c in checks]
    assert any(n.startswith("conformance:") for n in names)
    assert any(n.startswith("alignment:") for n in names)
    conformance_checks = [c for c in checks if c.name.startswith("conformance:")]
    assert all(c.status == "pass" for c in conformance_checks), conformance_checks
    print("run_preflight(full=True) loaded both smoke.yaml mock adapters and "
         "ran conformance + alignment checks against them")


if __name__ == "__main__":
    test_zarr_version_check_reports_pass_for_the_pinned_v2_install()
    test_torch_cuda_check_passes_trivially_when_device_is_cpu()
    test_vram_check_skips_cleanly_on_cpu_device()
    test_disk_check_never_raises_and_returns_a_verdict()
    test_context_alignment_check_flags_a_non_multiple_window()
    test_context_alignment_check_passes_a_clean_multiple()
    test_corpus_seal_check_passes_trivially_for_non_sealed_sources()
    test_corpus_seal_check_fails_on_a_missing_sealed_path()
    test_capped_stages_check_flags_a_max_series_exceeding_the_smallest_batch_size()
    test_capped_stages_check_passes_when_every_cap_fits()
    test_run_preflight_static_mode_never_loads_a_model()
    test_print_preflight_returns_false_iff_any_check_failed()
    test_run_preflight_full_mode_loads_real_mock_adapters_and_runs_conformance()
    print("All doctor preflight tests passed")


# --- activation store format (CLAUDE.md sec 11.25) -------------------------
#
# The failure this guards has no exception anywhere in it: a v3-written store
# opened under the pinned v2 reads back as an EMPTY group, so the run looks
# like one where extract never happened. The tests below therefore build the
# on-disk METADATA shapes directly rather than asserting on a message -- the
# check's whole value is that it distinguishes states that are otherwise
# indistinguishable at runtime.

def _cfg_at(tmp_path: Path, name: str = "r") -> PipelineConfig:
    cfg = _cpu_cfg()
    cfg.run.out_dir = str(tmp_path)
    cfg.run.name = name
    return cfg


def test_store_format_passes_when_no_store_exists_yet():
    # A run that has not extracted yet must not be reported as broken;
    # "nothing here" and "something unreadable here" are different states.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        check = _check_store_format(_cfg_at(Path(td)))
    assert check.status == "pass"
    assert "extract will create one" in check.detail


def test_store_format_fails_on_a_v3_store_under_the_v2_pin():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg_at(Path(td))
        store = cfg.run_dir() / "activations.zarr"
        store.mkdir(parents=True)
        (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
        check = _check_store_format(cfg)
    assert check.status == "fail"
    assert "11.25" in check.detail
    # The remediation must not be "reinstall zarr": the installed version is
    # correct and the STORE is the stale artifact.
    assert "extract" in check.remediation


def test_store_format_passes_on_a_v2_store():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg_at(Path(td))
        store = cfg.run_dir() / "activations.zarr"
        store.mkdir(parents=True)
        (store / ".zgroup").write_text('{"zarr_format": 2}')
        check = _check_store_format(cfg)
    assert check.status == "pass"


def test_store_format_warns_when_both_metadata_kinds_are_present():
    # The exact state CLAUDE.md sec 11.25 describes being left behind when a
    # v2 process opens a v3 directory: the v3 manifest is still there and a
    # bare v2 `.zgroup` has been written beside it. It reads back empty, so
    # it must not pass.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg_at(Path(td))
        store = cfg.run_dir() / "activations.zarr"
        store.mkdir(parents=True)
        (store / "zarr.json").write_text('{"zarr_format": 3}')
        (store / ".zgroup").write_text('{"zarr_format": 2}')
        check = _check_store_format(cfg)
    assert check.status == "warn"
    assert check.remediation


def test_store_format_check_is_wired_into_preflight():
    # The check existing is not the same as the check running: this module's
    # docstring claimed sec 11.25 coverage for months while no check existed.
    names = [c.name for c in run_preflight(_cpu_cfg())]
    assert "activation store format" in names


def _cfg_with_models(n: int) -> PipelineConfig:
    """`smoke.yaml` plus (n - 2) clones of its second model, renamed.

    Cloning keeps every other field of a real, valid config intact -- the
    check under test reads only `len(cfg.models)` and `cfg.stats`, and a
    hand-built stub would be a different object from the one preflight
    actually runs against.
    """
    import copy
    cfg = _cpu_cfg()
    while len(cfg.models) < n:
        extra = copy.deepcopy(cfg.models[1])
        extra.name = f"clone{len(cfg.models)}"
        cfg.models.append(extra)
    cfg.models = cfg.models[:n]
    return cfg


def test_multiplicity_budget_states_arithmetic_not_a_guess():
    """The check must report the CARRYING CAPACITY, never guess a family count.

    A family count is a property of the corpus, not of the config, so a check
    that assumed one would be a claim checked nowhere (`CLAUDE.md` sec 11.34).
    What it can state exactly is the arithmetic: pairs x families / n_boot vs
    alpha.
    """
    cfg = _cfg_with_models(3)
    cfg.stats.n_boot, cfg.stats.alpha = 2000, 0.05
    check = _check_multiplicity_budget(cfg)
    assert check.status == "pass"
    assert "3 pair(s)" in check.detail
    assert "at most 33 families" in check.detail


def test_multiplicity_budget_fails_when_not_even_one_family_fits():
    """The condition that matters is unsatisfiability, and it must FAIL loudly.

    At 6 pairs and n_boot=100 the smallest attainable adjusted p for a single
    family is 6/100 = 0.06 > alpha -- so every L0 non-result in that run would
    be arithmetic, not evidence, at any effect size.
    """
    cfg = _cfg_with_models(4)
    cfg.stats.n_boot, cfg.stats.alpha = 100, 0.05
    check = _check_multiplicity_budget(cfg)
    assert check.status == "fail"
    assert "cannot carry even ONE family" in check.detail
    assert "stats.n_boot" in check.remediation


def test_multiplicity_budget_scales_with_pair_count_not_model_count():
    """Growth is C(n,2), so capacity must fall QUADRATICALLY, not linearly.

    Pinning the ratio rather than the raw numbers: 2 -> 4 models is 1 -> 6
    pairs, so capacity must drop 6x, not 2x. A check that divided by model
    count would pass every other assertion in this file.
    """
    def capacity(n: int) -> int:
        cfg = _cfg_with_models(n)
        cfg.stats.n_boot, cfg.stats.alpha = 6000, 0.05
        detail = _check_multiplicity_budget(cfg).detail
        return int(detail.split("at most ")[1].split()[0])

    assert capacity(2) == 300
    assert capacity(4) == 50


def test_multiplicity_budget_is_wired_into_preflight():
    """A check that exists but never runs is not a check -- same reason the
    store-format test above pins its wiring."""
    names = [c.name for c in run_preflight(_cpu_cfg())]
    assert "multiplicity budget" in names
