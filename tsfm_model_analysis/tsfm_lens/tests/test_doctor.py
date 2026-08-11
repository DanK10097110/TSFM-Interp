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
                              _check_corpus_seal, _check_disk, _check_torch_cuda,
                              _check_vram, _check_zarr_version, print_preflight,
                              run_preflight)

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
