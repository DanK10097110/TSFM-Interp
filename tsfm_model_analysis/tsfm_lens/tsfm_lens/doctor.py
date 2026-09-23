"""Automated preflight checks (`ROADMAP.md` sec 16 E2).

Every trap in `CLAUDE.md` sec 11 that cost a session instead of a build --
zarr's v2/v3 split (sec 11.15), a stale on-disk store format (sec 11.25), a
batch-size cap silently exceeded (sec 11.5/A16), a corpus.jsonl that never
got sealed correctly -- is mechanically cheap to catch *before* a run starts,
not after ten minutes of extraction. This module is that catch, structured
as a fixed list of independent checks rather than one big try/except, so a
run tells you everything wrong at once instead of stopping at the first one.

Two tiers, matching the cost/frequency tradeoff `CLAUDE.md` sec 2.8 already
applies elsewhere in this repo: `run_preflight(cfg)` (the default-on check
`run.py` runs before every pipeline invocation, `--no-preflight` to skip) is
static-only -- no model load, no checkpoint download, seconds not minutes.
`run_preflight(cfg, full=True)` (`run.py --doctor`, standalone) additionally
loads every configured model and runs `models/conformance.py`'s adapter
checklist plus a real `impulse_alignment_check` -- exactly the two things
`CLAUDE.md` invariant 7 says must be checked by hand on every new checkpoint
or library bump, now runnable as one command instead of two manual ones.

A check that can't run at all (missing optional dependency, no CUDA) reports
`"warn"` with what's skipped and why, never a silent pass (`CLAUDE.md` sec
2.5) -- this module has no case where "nothing printed" means "nothing to
report."
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .utils import log


@dataclass
class DoctorCheck:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    remediation: str = ""


def _check_zarr_version() -> DoctorCheck:
    try:
        import zarr
        version = zarr.__version__
        major = int(version.split(".")[0])
    except ImportError:
        return DoctorCheck("zarr version", "fail", "zarr is not installed",
                           'pip install "zarr>=2.16,<3"')
    if major >= 3:
        return DoctorCheck(
            "zarr version", "fail",
            f"zarr {version} installed; extraction/store.py calls the zarr v2 "
            f"Group.create_dataset API, which zarr>=3 renamed (CLAUDE.md sec 11.15) "
            f"-- extraction will crash on the first write.",
            'pip install "zarr>=2.16,<3"')
    return DoctorCheck("zarr version", "pass", f"zarr {version} (<3, as pinned)")


def _check_torch_cuda(cfg: PipelineConfig) -> DoctorCheck:
    try:
        import torch
    except ImportError:
        return DoctorCheck("torch/CUDA", "fail", "torch is not installed", "pip install torch")
    if cfg.run.device != "cuda":
        return DoctorCheck("torch/CUDA", "pass",
                           f"run.device={cfg.run.device!r}; CUDA not required")
    if not torch.cuda.is_available():
        return DoctorCheck(
            "torch/CUDA", "fail",
            "run.device='cuda' but torch.cuda.is_available() is False",
            "install a CUDA-enabled torch build matching the machine's driver, "
            "or set run.device: cpu")
    return DoctorCheck("torch/CUDA", "pass", f"CUDA available: {torch.cuda.get_device_name(0)}")


def _check_vram(cfg: PipelineConfig) -> DoctorCheck:
    """Rough heads-up only -- real peak VRAM depends on each checkpoint's own
    hidden size/head count, which this check does not load anything to learn.
    """
    try:
        import torch
    except ImportError:
        return DoctorCheck("VRAM headroom", "warn", "torch not installed; cannot query VRAM")
    if cfg.run.device != "cuda" or not torch.cuda.is_available():
        return DoctorCheck("VRAM headroom", "pass", "skipped (no CUDA device in use)")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    if free_gb < 2.0:
        return DoctorCheck(
            "VRAM headroom", "fail",
            f"only {free_gb:.1f} GB free of {total_gb:.1f} GB total -- unlikely to fit "
            f"any real checkpoint",
            "free GPU memory (check nvidia-smi for other processes) or set run.device: cpu")
    warnings = []
    if cfg.attention.enabled and cfg.attention.patterns:
        n_windows = cfg.data.context_len // cfg.alignment.window
        # Order-of-magnitude only: CLAUDE.md sec 6.5's own worked example
        # (chronos-t5, T=512, batch_series=16) measured ~2.4GB held across
        # layers; this assumes a similar ~12-head, ~12-layer shape rather
        # than loading a checkpoint to know for sure.
        rough_gb = 4 * cfg.attention.batch_series * n_windows * n_windows * 12 * 12 / 1e9
        if rough_gb > free_gb * 0.5:
            warnings.append(
                f"attention.patterns at batch_series={cfg.attention.batch_series}, "
                f"{n_windows} windows is a rough order-of-magnitude {rough_gb:.1f} GB "
                f"estimate against {free_gb:.1f} GB free -- lower attention.batch_series "
                f"first if this OOMs (CLAUDE.md sec 6.5)")
    if cfg.l3.enabled and cfg.l3.patching.enabled and cfg.l3.patching.per_window:
        n_windows = cfg.data.context_len // cfg.alignment.window
        if cfg.l3.patching.max_series * n_windows > 20_000:
            warnings.append(
                f"l3.patching.per_window with max_series={cfg.l3.patching.max_series} x "
                f"{n_windows} windows multiplies patching cost substantially "
                f"(CLAUDE.md sec 6.5's per-window note) -- lower window_stride or "
                f"max_series if this OOMs or is too slow")
    if warnings:
        return DoctorCheck("VRAM headroom", "warn",
                           f"{free_gb:.1f} GB free of {total_gb:.1f} GB total; " +
                           "; ".join(warnings))
    return DoctorCheck("VRAM headroom", "pass", f"{free_gb:.1f} GB free of {total_gb:.1f} GB total")


def _check_disk(cfg: PipelineConfig) -> DoctorCheck:
    out_dir = Path(cfg.run.out_dir)
    check_path = out_dir if out_dir.exists() else (
        out_dir.parent if out_dir.parent.exists() else Path("."))
    _, _, free_bytes = shutil.disk_usage(check_path)
    free_gb = free_bytes / 1e9
    if free_gb < 2.0:
        return DoctorCheck(
            "disk headroom", "fail",
            f"only {free_gb:.1f} GB free on the filesystem holding {out_dir}",
            "free disk space or point run.out_dir at a larger volume")
    if free_gb < 10.0:
        return DoctorCheck(
            "disk headroom", "warn",
            f"{free_gb:.1f} GB free on the filesystem holding {out_dir} -- real-checkpoint "
            f"extraction stores (activations.zarr) can run several GB per model at "
            f"medium_run-sized configs",
            "free disk space, or point run.out_dir at a larger volume, before a large run")
    return DoctorCheck("disk headroom", "pass",
                       f"{free_gb:.1f} GB free on the filesystem holding {out_dir}")


def _check_store_format(cfg: PipelineConfig) -> DoctorCheck:
    """Whether an existing activation store is readable by the pinned zarr major.

    This module's own docstring has claimed since it was written that it
    catches "a stale on-disk store format (sec 11.25)". It did not -- there
    was no such check. Added 2026-08-28 after re-reading that claim as a
    claim rather than as documentation (`CLAUDE.md` sec 11.29: a premise in a
    docstring was measured once and can be wrong).

    The failure it catches is the nastiest shape in this repo, because it
    produces no error at all. `zarr.open_group` under the pinned v2 against a
    directory written by zarr **v3** finds no v2 metadata (`.zgroup`/
    `.zattrs`), treats the directory as an EMPTY v2 group, and returns it.
    Every downstream read then sees zero models and zero layers, so the run
    looks like one where `extract` simply never happened -- and because
    stages self-skip on missing artifacts, it can proceed a long way looking
    normal. `CLAUDE.md` sec 11.25 is the session that cost.

    Deliberately does NOT delete anything: a stale store is real data written
    by a real run, and which run directories are expendable is the operator's
    call, not a preflight check's (`ROADMAP.md` sec 0.5 item 13's G1 makes the
    same call about `runs/real_run`). It reports and names the fix.
    """
    store = cfg.run_dir() / "activations.zarr"
    if not store.exists():
        return DoctorCheck("activation store format", "pass",
                           f"no store at {store} yet; extract will create one")
    v3 = (store / "zarr.json").exists()
    v2 = (store / ".zgroup").exists()
    if v3 and not v2:
        return DoctorCheck(
            "activation store format", "fail",
            f"{store} is a zarr v3 store (zarr.json present, .zgroup absent), but this "
            f"package pins zarr<3. v2 does NOT raise on it -- it reads the directory "
            f"back as an EMPTY group, so every stage would report zero rows and the "
            f"run would look like one where extract never ran (CLAUDE.md sec 11.25)",
            f"delete {store} and re-run the extract stage "
            f"(--stages extract --force extract), or point run.name at a fresh run dir")
    if v3 and v2:
        return DoctorCheck(
            "activation store format", "warn",
            f"{store} carries BOTH v2 (.zgroup) and v3 (zarr.json) metadata -- most "
            f"likely a v3 store that a later v2 open partially initialized on top of, "
            f"which reads back empty rather than raising",
            f"delete {store} and re-run the extract stage; do not trust a partial read")
    if not v2:
        return DoctorCheck(
            "activation store format", "warn",
            f"{store} exists but has no zarr v2 root metadata (.zgroup); it is not a "
            f"store this package can read",
            f"delete {store} and re-run the extract stage")
    return DoctorCheck("activation store format", "pass",
                       f"{store} is a zarr v2 store, readable by the pinned zarr major")


def _check_multiplicity_budget(cfg: PipelineConfig) -> DoctorCheck:
    """Whether this run's bootstrap resolution can support its comparison count.

    `ROADMAP.md` sec 24.3 sub-item 4. A bootstrap p-value is floored at
    1/`n_boot`, so a family of `m` Holm-corrected tests cannot produce an
    adjusted p below `m / n_boot` AT ANY EFFECT SIZE. Once `m / n_boot >
    alpha` the correction is unsatisfiable and every non-result in it is
    arithmetic rather than evidence -- which is exactly how a three-model run
    once produced a textbook-looking 4-significant-families-to-0 demotion that
    was the p-floor and not multiplicity (`CLAUDE.md` sec 6.6).

    L0 now Holm-corrects across the joint (pair, family) set, so `m` grows as
    C(n,2) x families. The report already detects and reddens the
    unsatisfiable case -- AFTER the run. This says it before, which is the
    point: the fix is to raise `stats.n_boot`, and learning that at the end of
    an extraction costs the extraction.

    It does NOT guess the family count, which is a property of the corpus and
    not of the config. It reports the arithmetic instead: how many families
    this `n_boot` can carry at this pair count. An operator who knows their
    corpus can read the verdict off one number; a check that guessed would be
    a claim checked nowhere (`CLAUDE.md` sec 11.34).
    """
    n_models = len(cfg.models)
    pairs = max(1, n_models * (n_models - 1) // 2)
    if not cfg.stats.enabled:
        return DoctorCheck("multiplicity budget", "pass",
                           "stats.enabled is false; no corrected tests are run")
    n_boot, alpha = cfg.stats.n_boot, cfg.stats.alpha
    max_families = int(n_boot * alpha // pairs)
    detail = (f"{n_models} model(s) -> {pairs} pair(s); L0 Holm-corrects over "
              f"pairs x families jointly. At stats.n_boot={n_boot} and alpha={alpha} "
              f"the smallest attainable adjusted p is (pairs x families)/n_boot, so "
              f"this run can carry at most {max_families} famil"
              f"{'y' if max_families == 1 else 'ies'} before the correction becomes "
              f"unsatisfiable")
    remedy = (f"raise stats.n_boot (>= pairs x families / alpha, i.e. "
              f">= {int(pairs / alpha)} x families) before the run, not after")
    if max_families < 1:
        return DoctorCheck("multiplicity budget", "fail", detail +
                           " -- it cannot carry even ONE family: every L0 "
                           "non-result would be a p-floor artifact", remedy)
    if max_families < 4:
        return DoctorCheck("multiplicity budget", "warn", detail +
                           " -- most benchmark corpora here have 6", remedy)
    return DoctorCheck("multiplicity budget", "pass", detail)


def check_transfer_fdr_budget(n_null: int, m: int, q: float, p_method: str) -> DoctorCheck:
    """Whether `sae/transfer.py`'s exact permutation p-floor can support one
    BH family of `m` tests (`ROADMAP.md` sec 37 P3, item 7).

    Same shape as `_check_multiplicity_budget` above, one level down: a
    permutation p is floored at `1/(n_null+1)`, so `m` tests all sitting at
    that floor cannot jointly clear a Bonferroni-style bound of `q` once
    `m * floor > q` -- i.e. `m / (n_null+1) > q`, the exact condition this
    item specifies. BH is less conservative than Bonferroni (its smallest
    p can still survive on its own), so this is a SUFFICIENT-for-concern
    warning, not a proof every such family fails outright; `p_method` in
    `{"gpd_tail", "adaptive"}` routes AROUND the floor entirely (sec 37 P3
    items 2), so this only ever fires under `"exact"`.

    Unlike `_check_multiplicity_budget`, `m` here (the transfer-test count
    for one ordered model pair's one leg) is a property of how many
    concepts got trained, not of the config -- exactly the same "do not
    guess a property of the corpus" restraint that function's own docstring
    states for its own family count (`CLAUDE.md` sec 11.34). So this is
    called from `sae/concept_stage.py` AFTER `run_transfer`/
    `run_atlas_transfer` know their real per-pair, per-leg `m`, not from
    static `run_preflight` -- there is nothing to check before those tests
    exist.
    """
    detail = (f"n_null={n_null} (p-floor 1/{n_null + 1}={1.0 / (n_null + 1):.5f}), "
              f"m={m} test(s) in this BH family, q={q}, p_method={p_method!r}")
    remedy = (f"raise concepts.transfer_p_method to 'gpd_tail' or 'adaptive' "
              f"(sec 37 P3 item 2), or raise sae.transfer_n_null so that "
              f"m/(n_null+1) <= q (currently need n_null >= {int(m / max(q, 1e-12)) - 1})")
    if p_method != "exact":
        return DoctorCheck("transfer FDR floor", "pass", detail + " -- not applicable")
    if m <= 0:
        return DoctorCheck("transfer FDR floor", "pass", detail)
    if m / (n_null + 1) > q:
        return DoctorCheck("transfer FDR floor", "warn", detail +
                           " -- m/(n_null+1) exceeds q: this family's exact "
                           "p-floor cannot support a Bonferroni-tight FDR "
                           "bound at this m", remedy)
    return DoctorCheck("transfer FDR floor", "pass", detail)


def _check_corpus_seal(cfg: PipelineConfig, full: bool = False) -> list:
    checks = []
    sources = [("data", cfg.data.source, cfg.data.path)]
    if cfg.confirm.enabled:
        sources.append(("confirm", cfg.confirm.source, cfg.confirm.path))
    for label, source, path in sources:
        if source != "sealed":
            checks.append(DoctorCheck(f"{label} corpus", "pass",
                                      f"source={source!r}; seal verification not applicable"))
            continue
        p = Path(path)
        if not p.exists():
            checks.append(DoctorCheck(
                f"{label} corpus", "fail",
                f"{label}.path {path!r} does not exist (relative to the invoking cwd)",
                "check the path is relative to tsfm_lens/, per CLAUDE.md sec 8's "
                "documented `python run.py --config configs/...` invocation"))
            continue
        manifest, corpus = p / "manifest.json", p / "corpus.jsonl"
        if not manifest.exists() or not corpus.exists():
            checks.append(DoctorCheck(
                f"{label} corpus", "fail",
                f"{label}.path {path!r} exists but is missing corpus.jsonl/manifest.json",
                "point at a directory produced by tsfm_benchmark's seal_corpus"))
            continue
        if not full:
            checks.append(DoctorCheck(
                f"{label} corpus", "pass",
                f"{path!r} has manifest.json + corpus.jsonl (full hash re-verification "
                f"only with --doctor)"))
            continue
        try:
            from tsfm_benchmark import load_sealed
            samples, manifest_data = load_sealed(str(p), verify=True)
            checks.append(DoctorCheck(
                f"{label} corpus", "pass",
                f"{path!r} verified: {len(samples)} samples, epoch "
                f"{manifest_data.get('epoch')}, visibility {manifest_data.get('visibility')}"))
        except ImportError:
            checks.append(DoctorCheck(
                f"{label} corpus", "warn",
                f"tsfm_benchmark not installed; cannot verify {label} corpus seal hashes",
                "pip install -e ../../tsfm_benchmark to enable seal verification"))
        except Exception as e:
            checks.append(DoctorCheck(
                f"{label} corpus", "fail",
                f"{path!r} failed seal verification: {e}",
                "regenerate or re-seal the corpus -- do not trust it as-is"))
    return checks


def _check_context_alignment(cfg: PipelineConfig) -> DoctorCheck:
    # PipelineConfig.validate() already raises on this at load time; kept
    # here too per sec 16 E2's own checklist, and in case a caller builds a
    # PipelineConfig programmatically without calling validate().
    if cfg.data.context_len % cfg.alignment.window != 0:
        return DoctorCheck(
            "context_len / window", "fail",
            f"data.context_len={cfg.data.context_len} is not a multiple of "
            f"alignment.window={cfg.alignment.window}",
            "set data.context_len to a multiple of alignment.window")
    return DoctorCheck("context_len / window", "pass",
                       f"context_len {cfg.data.context_len} is a multiple of "
                       f"window {cfg.alignment.window}")


def _check_capped_stages(cfg: PipelineConfig) -> list:
    """Every stage that runs one forward pass per call must fit its whole
    sample in one model batch (`CLAUDE.md` sec 6.4/11.5, sec 15 A16) -- a
    cap left larger than a model's `batch_size` is a guaranteed `token_patch`
    crash once that stage actually runs, worth catching before extraction.
    """
    if not cfg.models:
        return [DoctorCheck("batch caps", "warn", "no models configured")]
    min_batch = min(m.batch_size for m in cfg.models)

    # Gate on the run's capability tier as well as on `enabled`, because a
    # stage the tier gate is about to drop cannot crash on a batch cap it
    # will never read. Failing preflight on it would be a false refusal of
    # exactly the shape `CLAUDE.md` sec 11.35 warns about -- a check that
    # reads as a considered finding while measuring something other than
    # what its name says.
    #
    # `resolve_adapter_class` (not the bare `ADAPTERS` dict) so a CONTRIB
    # adapter's tier is resolved too -- the original `if m.adapter in
    # ADAPTERS` filter silently dropped every contrib model from `tiers`
    # entirely, so a tier-0 contrib adapter paired with a tier-3 mock (the
    # exact shape `run.py --new-adapter`'s own scaffolded config produces,
    # ROADMAP.md sec 34.6 Item E3) computed run_tier=3 instead of the real 0
    # and this check then FAILED on a stage the real tier gate was about to
    # drop anyway -- the precise false refusal the comment above already
    # warns about, tripped by this very function.
    from .pipeline import _STAGE_MIN_TIER
    from .models import resolve_adapter_class
    tiers = []
    unresolvable = []
    for m in cfg.models:
        try:
            tiers.append(resolve_adapter_class(m.adapter).capability_tier())
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            unresolvable.append(f"{m.name} ({m.adapter}): {type(exc).__name__}: {exc}")
    run_tier = min(tiers) if tiers else 3

    def runs(stage: str) -> bool:
        return _STAGE_MIN_TIER.get(stage, 1) <= run_tier

    candidates = []
    if cfg.l3.enabled and cfg.l3.patching.enabled and runs("l3"):
        candidates.append(("l3.patching.max_series", cfg.l3.patching.max_series))
    if cfg.lens.enabled and runs("lens"):
        candidates.append(("lens.max_series", cfg.lens.max_series))
    if cfg.attention.enabled and cfg.attention.ablation and runs("attention"):
        candidates.append(("attention.ablation_max_series", cfg.attention.ablation_max_series))
    if cfg.sae.enabled and runs("sae"):
        candidates.append(("sae.forecast_preservation_max_series",
                           cfg.sae.forecast_preservation_max_series))
        if cfg.sae.feature_ablation_enabled:
            candidates.append(("sae.feature_ablation_max_series",
                               cfg.sae.feature_ablation_max_series))
        if cfg.sae.feature_steering_enabled:
            candidates.append(("sae.feature_steering_max_series",
                               cfg.sae.feature_steering_max_series))
    checks = []
    if unresolvable:
        # Reported, never silently absorbed into `tiers` -- an adapter this
        # check cannot resolve is exactly the shape it must not hide (sec
        # 2.5): the run_tier computed above excludes it, so say so rather
        # than let a missing model read as an agreeing one.
        checks.append(DoctorCheck(
            "batch caps: adapter resolution", "warn",
            "could not resolve a tier for: " + "; ".join(unresolvable),
            "run_tier below excludes these models -- fix the adapter name/import "
            "and rerun preflight before trusting this check"))
    if not candidates:
        checks.append(DoctorCheck("batch caps", "pass",
                                  f"no batch-per-call stages enabled at tier {run_tier}"))
        return checks
    for label, value in candidates:
        if value > min_batch:
            checks.append(DoctorCheck(
                f"batch cap: {label}", "fail",
                f"{label}={value} exceeds the smallest configured model batch_size "
                f"({min_batch}) -- this stage runs one forward pass per call and "
                f"token_patch raises if the model chunks the batch internally",
                f"set {label} to at most {min_batch}, or raise every model's batch_size"))
        else:
            checks.append(DoctorCheck(
                f"batch cap: {label}", "pass", f"{label}={value} <= min batch_size {min_batch}"))
    return checks


def _check_adapters_full(cfg: PipelineConfig) -> list:
    """Loads every configured model -- only run from `--doctor`, never the
    default preflight. Mirrors `run.py --check-alignment` plus
    `models/conformance.py::check_adapter_conformance` as one pass, and
    releases each model afterward.
    """
    from .extraction.alignment import calibrate_impulse_amplitude, impulse_alignment_check
    from .models.conformance import check_adapter_conformance
    from .pipeline import Context

    checks = []
    ctx = Context(cfg)
    for m in cfg.models:
        try:
            adapter = ctx.hub.get(m.name)
        except Exception as e:
            checks.append(DoctorCheck(
                f"adapter load: {m.name}", "fail", f"failed to load: {e}",
                "check the checkpoint name/adapter type and that the required "
                "library is installed"))
            continue
        try:
            report = check_adapter_conformance(adapter, cfg.alignment.window)
            checks.append(DoctorCheck(
                f"conformance: {m.name}", "pass",
                f"{report['n_layers']} layers, {report['n_tokens']} tokens, "
                f"predict() shape {report['predict_point_shape']}"))
        except Exception as e:
            checks.append(DoctorCheck(
                f"conformance: {m.name}", "fail", str(e),
                "see CLAUDE.md sec 6.2's ModelAdapter contract"))
        try:
            calibration = calibrate_impulse_amplitude(adapter, cfg.alignment.window)
            amplitude = calibration["amplitude"]
            alignment = impulse_alignment_check(adapter, cfg.alignment.window, amplitude=amplitude)
            min_frac = min(alignment.values())
            ok = min_frac >= cfg.alignment.min_diagonal_frac
            checks.append(DoctorCheck(
                f"alignment: {m.name}", "pass" if ok else "warn",
                f"min diagonal-hit fraction {min_frac:.2f} across {len(alignment)} "
                f"layers (threshold {cfg.alignment.min_diagonal_frac}, calibrated "
                f"impulse amplitude {amplitude:.2f} -- ROADMAP.md sec 15 A20)",
                "" if ok else "run --check-alignment for the full per-layer table "
                "before trusting any cross-model number (CLAUDE.md sec 6.3/invariant 7)"))
        except Exception as e:
            checks.append(DoctorCheck(f"alignment: {m.name}", "fail", str(e), ""))
        finally:
            ctx.hub.release(m.name)
    return checks


def run_preflight(cfg: PipelineConfig, full: bool = False) -> list:
    """Run every check; never raises -- a check that itself fails to run
    reports `"warn"` with why, per this module's own docstring.
    """
    checks = [
        _check_zarr_version(),
        _check_torch_cuda(cfg),
        _check_vram(cfg),
        _check_disk(cfg),
    ]
    checks.append(_check_store_format(cfg))
    checks.append(_check_multiplicity_budget(cfg))
    checks.extend(_check_corpus_seal(cfg, full=full))
    checks.append(_check_context_alignment(cfg))
    checks.extend(_check_capped_stages(cfg))
    if full:
        checks.extend(_check_adapters_full(cfg))
    return checks


def print_preflight(checks: list) -> bool:
    """Print a pass/warn/fail table; returns True iff nothing failed."""
    width = max((len(c.name) for c in checks), default=10)
    ok = True
    for c in checks:
        marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[c.status]
        line = f"  [{marker}] {c.name:<{width}}  {c.detail}"
        print(line)
        if c.remediation:
            print(f"           -> {c.remediation}")
        if c.status == "fail":
            ok = False
    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")
    print(f"  {len(checks)} checks: {len(checks) - n_fail - n_warn} pass, "
          f"{n_warn} warn, {n_fail} fail")
    if not ok:
        log.warning("doctor: %d preflight check(s) failed", n_fail)
    return ok
