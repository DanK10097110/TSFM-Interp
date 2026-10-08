"""Typed pipeline configuration with YAML loading and reasonable defaults.

Every analysis stage carries an `enabled` flag so users can compose exactly
the comparison they want; disabled stages are skipped and simply absent from
the final report.
"""

from __future__ import annotations

import dataclasses
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "model"
    adapter: str = "mock_patch"
    checkpoint: str = ""
    layer_regex: str = ""
    batch_size: int = 32
    capture_layer_stride: int = 1
    kwargs: dict = field(default_factory=dict)
    # Untrained-weights null baseline (ROADMAP.md sec 16 E9): same
    # architecture/checkpoint config, but randomly initialized rather than
    # pretrained. Load this model alongside its pretrained twin (same
    # `adapter`/`checkpoint`, a distinct `name`) to get a floor for CKA, L2
    # stitching gain, probe decodability, and SAE ground-truth alignment --
    # how much of each looks like "architecture + input statistics" rather
    # than learning.
    random_init: bool = False


@dataclass
class DataConfig:
    source: str = "smoke"
    path: str = ""
    context_len: int = 512
    horizon: int = 64
    max_series: Optional[int] = None
    family_key: str = "auto"
    smoke_series_per_family: int = 20


@dataclass
class AlignmentConfig:
    window: int = 32
    sanity_check: bool = True
    min_diagonal_frac: float = 0.5   # required hit fraction at the shallowest probed layer
    on_failure: str = "fail"         # fail | warn -- ROADMAP.md sec 15 A2
    # ROADMAP.md sec 15 A20: the impulse probe's amplitude used to be a
    # single hardcoded constant (0.25) calibrated against one (checkpoint,
    # context_len) pair -- it silently under- or over-shoots at others. When
    # true (default), `run_alignment_gate` self-calibrates the amplitude per
    # run via `calibrate_impulse_amplitude` before probing; the chosen value
    # and the sweep that produced it are recorded in the alignment artifact.
    # Set false to force the historical fixed 0.25 (e.g. to reproduce a
    # pre-A20 run's exact numbers).
    calibrate_amplitude: bool = True
    # Which definition of "how deep is this layer" every cross-model depth
    # figure uses (ROADMAP.md sec 18 F1, `analysis/depth_axis.py`):
    #   block    -- default. Position within the model's WHOLE stack, incl.
    #               stride-skipped blocks and surfaces the adapter declares
    #               but never captures (Chronos-T5's decoder). Truncation
    #               becomes visible: an encoder-only capture surface occupies
    #               the bottom half of the plot and the top half is empty.
    #   index    -- LEGACY, and what every number recorded before 2026-08-13
    #               was measured on. Index fraction over CAPTURED layers only,
    #               so a model's 1.0 means "last thing I captured", not "my
    #               output", and the coordinate moves with capture_layer_stride.
    #               Set this to reproduce an older run's depth coordinates.
    #   compute  -- cumulative measured FLOPs over full-forecast FLOPs, from
    #               the `budget` stage. Degrades to `block` (loudly) without it.
    #   functional -- align by a measured property rather than position;
    #               needs a caller-supplied value per layer.
    depth_axis: str = "block"


@dataclass
class ExtractionConfig:
    enabled: bool = True
    # "float16" (default, half the disk/VRAM of float32) or "float32" for a
    # model whose activations run hot enough to overflow float16's ~65504
    # max (`root.attrs["nonfinite"]`, written by `extraction/store.py::
    # finalize_layer`, sec 15 A19, says which layers actually do on a given
    # run -- check it before assuming this knob is needed for a new model).
    store_dtype: str = "float16"


@dataclass
class L0Config:
    enabled: bool = True
    quantiles: list = field(default_factory=lambda: [0.1, 0.5, 0.9])
    # MASE denominator (`ROADMAP.md` sec 15 A11): "mean_abs_diff" (default,
    # every previously-recorded number keeps reproducing) or
    # "seasonal_naive" (mean absolute error of a period-m seasonal-naive
    # forecast on the context -- the standard fix for scale degeneracy
    # under heavy intermittency or a near-flat context; period estimated
    # per series via autocorrelation when not otherwise available).
    scale: str = "mean_abs_diff"
    # Reliability guard, independent of `scale`: a series whose MASE
    # denominator is below this fraction of the target's own mean absolute
    # level is flagged `mase_reliable: false` and excluded from MASE
    # aggregates (never from smape/pinball, which don't share this
    # degeneracy). 0.0 disables the guard (matches pre-A11 behavior
    # exactly); >0 is the recommended setting for any corpus containing
    # `intermittent_bursts` or similarly zero-heavy families.
    min_scale_frac: float = 0.05
    # Repeat-run noise floor (`ROADMAP.md` sec 15 A13): how much a model's
    # own MASE varies between two calls that should, in the absence of
    # sampling, be identical -- every ΔMASE elsewhere in the repo (head/MLP
    # ablation, SAE forecast-preservation, L3 restoration) is otherwise
    # compared against zero instead of against this. `noise_floor_repeats:
    # 0` disables it (no extra predict() calls, matches pre-A13 runtime);
    # >=2 measures it on a small `noise_floor_series`-sized subset. Cheap
    # relative to what it re-contextualizes, so on by default with a small
    # repeat count (3) rather than requiring opt-in.
    noise_floor_repeats: int = 3
    noise_floor_series: int = 16
    # Probabilistic-forecast calibration diagnostics (`ROADMAP.md` sec 16
    # E10): reliability curve, PIT histogram, quantile-crossing rate, and
    # interval coverage/sharpness by horizon step -- computed from the same
    # `predict()` output `run_l0` already has in memory, so this adds no new
    # forward passes. Needs >=2 quantile levels to be meaningful; silently
    # skipped otherwise regardless of this flag.
    calibration: bool = True
    # Horizon-resolved MASE/pinball (`ROADMAP.md` sec 16 E12): every current
    # behavioral metric aggregates over the whole forecast horizon, so
    # "where does the error come from at h=1 vs h=H" is currently invisible.
    # A pure reduction over the same `predict()` output already in memory --
    # zero new forward passes -- so on by default.
    horizon_resolved: bool = True


@dataclass
class L1Config:
    enabled: bool = True
    layer_stride: int = 1
    max_rows: int = 100_000
    family_conditioned: bool = True
    min_family_series: int = 15
    rsa: bool = True
    rsa_max_series: int = 600


@dataclass
class L2Config:
    enabled: bool = True
    layer_stride: int = 2
    lambdas: list = field(default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0])
    val_frac: float = 0.25
    max_rows: int = 60_000
    input_baseline: bool = True


@dataclass
class PatchingConfig:
    enabled: bool = True
    layer_stride: int = 2
    corruptions: list = field(default_factory=lambda: [
        "deseasonalize", "noise", "level_shift", "spike", "warp", "dropout"])
    max_series: int = 64
    num_samples: int = 8
    per_window: bool = True
    window_stride: int = 1
    # ROADMAP.md sec 16 E12: resolve restoration by forecast horizon step
    # (in addition to layer/window) from the same already-computed patched
    # forecasts, no new forward passes.
    horizon_resolved: bool = True


@dataclass
class LensConfig:
    enabled: bool = True
    layer_stride: int = 1
    max_series: int = 64
    tuned: bool = True
    tuned_max_series: int = 2000
    val_frac: float = 0.25
    lambdas: list = field(default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0])
    crystallization_tol: float = 0.1
    # ROADMAP.md sec 16 E12: resolve crystallization depth per horizon step
    # (in addition to the whole-horizon-averaged number) from the same
    # already-computed skip-lens forecasts, no new forward passes.
    horizon_resolved: bool = True
    # ROADMAP.md sec 38.4 K4: number of series whose per-series convergence
    # depth is persisted to `lens/convergence.npz`. 0 keeps the legacy sample
    # (no extra forward pass); a larger value forecasts a separate stratified
    # sample in batch-sized chunks. `stage_input: False`: it only sizes that
    # additive artifact, no legacy lens artifact reads it, and fingerprinting
    # it would mark every older run stale. Change it with `--force lens`.
    depth_max_series: int = field(default=0, metadata={"stage_input": False})


@dataclass
class AttentionConfig:
    enabled: bool = True
    patterns: bool = True
    max_series: int = 128
    batch_series: int = 16
    max_lag_tokens: int = 256
    ablation: bool = True
    ablation_max_series: int = 128
    head_layer_stride: int = 1
    top_k: int = 5
    # Which lag resolution a CROSS-MODEL attention claim is allowed to cite
    # (ROADMAP.md sec 18 F5). Both are always computed and recorded: `native`
    # is each model's own token lag axis -- the historical behavior, and the
    # only meaningful axis for a within-model statement -- while `matched`
    # bins every model's lag axis to the coarsest configured model's token
    # width first, so a "sharper seasonal attention" claim is not partly a
    # statement about patch size.
    resolution_mode: str = "matched"


@dataclass
class ExemplarsConfig:
    enabled: bool = True
    per_family: int = 2
    max_families: int = 6


@dataclass
class L3Config:
    enabled: bool = True
    # `noise` and `spike` were the two weakest members of the battery under the
    # previous defaults (snr_db 6.0 / count 3, scale 6.0), at perturbation
    # energies of 0.199 and 0.167 against `level_shift`'s 2.853 and
    # `deseasonalize`'s 0.564 -- weak enough that both read as "this corruption
    # does nothing" on a shared linear axis when what was actually being
    # measured was a very small perturbation. Raised to a measured 0.790 and
    # 1.230 respectively (`spike` also from 0.58% to 1.55% of timesteps
    # touched), which puts them in the same range as the rest of the battery
    # without reaching `level_shift`'s outsized value. Measured on this repo's
    # own corpus over 200 series, not assumed; `l3.calibrate` remains the way
    # to equalize energies exactly, and these defaults are what an
    # uncalibrated run gets.
    corruptions: dict = field(default_factory=lambda: {
        "noise": {"snr_db": 0.0},
        "detrend": {},
        "deseasonalize": {"top_k": 2},
        "frequency_shift": {"factor": 2.0},
        "level_shift": {"position_frac": 0.6, "scale": 3.0},
        "spike": {"count": 8, "scale": 10.0},
        "smooth": {"kernel": 9},
        "warp": {"strength": 0.15},
        "dropout": {"frac": 0.15, "n_blocks": 3},
    })
    max_series: int = 512
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    # `ROADMAP.md` sec 15 A12: corruption strengths are set independently
    # per corruption, so raw "behavioral sensitivity" bars aren't comparable
    # across corruptions (`level_shift` dominates by construction). `"none"`
    # (default) keeps every recorded number reproducing byte-for-byte.
    # `"input_energy"` re-solves each calibratable corruption's magnitude
    # parameter (pure numpy, no model) to hit a common per-series
    # perturbation-energy budget derived from this battery's own configured
    # strengths (the median of their natural energies) -- every corruption
    # is still run and reported either way (§5's decision to keep the
    # low-signal end of the contrast stands).
    calibrate: str = "none"


@dataclass
class StatsConfig:
    enabled: bool = True
    n_boot: int = 500
    n_boot_heavy: int = 200
    ci: float = 0.95
    alpha: float = 0.05
    min_series: int = 8


@dataclass
class InternalsConfig:
    enabled: bool = True
    layer_stride: int = 1
    max_rows: int = 40_000
    probe_pca_dim: int = 50
    # Label-permutation null for family-probe decodability (ROADMAP.md sec 16
    # E9's remaining scope -- the majority-class `chance` line is a weaker,
    # related control; this reruns the identical probe fit `n_perm` times
    # with the family label shuffled at the SERIES level, mirroring
    # sae/ground_truth.py::permutation_null_alignment's "rerun the identical
    # search" pattern. `0` disables it, matching that function's convention.
    probe_permutation_repeats: int = 5
    # Iteration budget for the family probe's logistic solver. 300 was the
    # historical value and does NOT converge on a ~1000-series corpus (14
    # ConvergenceWarnings on `benchmark_large`). Raised rather than left to
    # warn on stderr: an under-converged fit UNDERSTATES probe accuracy, so
    # a too-small budget makes a model's internals look less decodable than
    # they are, and the permutation null reruns the identical fit -- so the
    # bias applies to both sides and is invisible in the real-vs-null gap
    # that the report actually renders. `probe_converged` is recorded per
    # layer so a future corpus outgrowing this budget says so in the
    # artifact instead of only in a log line nobody reads (sec 2.5).
    probe_max_iter: int = 2000


@dataclass
class ConfirmConfig:
    enabled: bool = False
    source: str = "sealed"
    path: str = ""
    max_series: int = 1024
    require_seal: bool = True
    alpha: float = 0.05
    # ROADMAP.md sec 37.10 P7 -- the null-draw count `_replicate_registered_
    # concepts` uses to recompute the registered `concept_transfer` claims'
    # stratum-matched permutation test on private data. `sae.transfer_n_null`
    # (default 200, floor 1/201) is unsatisfiable for Holm across even a
    # small registered family: with `concepts.n_registered: 20`,
    # 20/201=0.0995 > 0.05. Raised here (2000, floor 1/2001) so `m/(n_null+1)`
    # clears `alpha` -- the same fix P6a's retry applied to its own BH family
    # (ROADMAP.md sec 37.9), and `satisfiable` is recorded either way rather
    # than assumed (`CLAUDE.md` sec 6.6).
    concept_transfer_n_null: int = 2000

    # ROADMAP.md sec 38.2 (K2) -- registration and confirmation of the
    # CAUSAL concept claims (`concept_causal`, `concept_atlas`,
    # `shared_input_agreement`, `concept_structure`). Opt-in: with the
    # default `False` the registry and `confirmation.json` are byte-identical
    # to what they were before K2 existed (`CLAUDE.md` sec 2.1, invariant 13).
    #
    # Every field below carries `omit_at_default`: a default value is left out
    # of the stage fingerprint (`manifest.py`), so adding these fields does
    # not mark any older run stale (`CLAUDE.md` sec 11.51), while a
    # NON-default value fingerprints `confirm` (whole-section key) and, for
    # `register_concept_claims`, `register` (field-level key), so changing it
    # on a finished run refuses the stale skip instead of silently keeping
    # the old registry.
    register_concept_claims: bool = field(default=False, metadata={"omit_at_default": True})
    # ROADMAP.md sec 38.3.4 -- when on, claims are registered only from targets
    # whose battery clears are BH-significant against their own empirical
    # chance (`analysis/target_significance.py`); excluded claims are recorded
    # with their reason, and a target without `empirical_chance` makes
    # registration refuse. Fingerprints `register` (field-level key) and
    # `confirm` (whole section) only when on.
    register_requires_target_significance: bool = field(
        default=False, metadata={"omit_at_default": True})
    # ROADMAP.md sec 38.3.4 (L5 defined-firing rule) -- when on, an agreement
    # claim whose dev verdict is `acts differently` is registered as `differs`
    # only if the firing statistic is DEFINED (`firing_defined`: (i) needs
    # `level` to clear on both sides, (ii) needs both sides to clear a shape
    # channel); the rest are recorded under `excluded`. The rule is frozen in
    # each agreement claim and `confirm` applies it to the private verdict.
    # The dev verdicts are recomputed from `shared_input_agreement.json`, so
    # the dev agreement step is not rerun. Fingerprints `register`
    # (field-level key) and `confirm` (whole section) only when on.
    register_requires_defined_firing: bool = field(
        default=False, metadata={"omit_at_default": True})
    # ROADMAP.md sec 38.4 (K4) -- register and confirm the U1 reliability
    # claims ("do internals predict a model's per-series error beyond the free
    # baseline?"): one `reliability_u1` claim per (model, task) whose DEV gain
    # CI lower bound is > 0 in the dev K4 JSON at `reliability_dev_json`
    # (absolute, or relative to the run directory; written by
    # `run_reliability_from_internals.py`). `confirm` REFITS the frozen
    # procedure on private series. Both fields fingerprint `register`
    # (field-level key) and `confirm` only once set.
    register_reliability_claims: bool = field(default=False, metadata={"omit_at_default": True})
    reliability_dev_json: str = field(default="", metadata={"omit_at_default": True})
    # Null directions per registered causal claim on the first private pass.
    # A claim whose exact p sits on the floor 1/(n+1) is redrawn with
    # `causal_max_null` directions (the adaptive tail p, sec 37.6/P3), so the
    # ledger's attainable Holm floor is `m / (causal_max_null + 1)`.
    causal_n_null: int = field(default=200, metadata={"omit_at_default": True})
    causal_max_null: int = field(default=1000, metadata={"omit_at_default": True})
    # Random same-composition member sets per atlas claim (no forward pass).
    atlas_n_null: int = field(default=2000, metadata={"omit_at_default": True})
    # Matched random-feature-set draws per side of a shared-input agreement
    # claim. `shared_input_n_null` (50) cannot survive a Holm family of ~40
    # (40/51 > alpha), so confirm re-draws the floors at this size.
    agreement_n_null: int = field(default=1000, metadata={"omit_at_default": True})
    # Bootstrap resamples for the structure claims' private rate CI.
    structure_n_boot: int = field(default=2000, metadata={"omit_at_default": True})

    # ROADMAP.md sec 41 (V3-B) -- three opt-in claim types, each off by default
    # and left out of the stage fingerprint at its default (older runs stay
    # fresh). A non-default value fingerprints `confirm` (whole section) and,
    # for the two `register_*` flags and their thresholds, `register`.
    #
    # `register_family_presence_claims`: one `family_presence` claim per
    # (effect family, model) whose DEV count of the model's causal features
    # assigned to that family is >= `family_presence_min_dev` ("model m has
    # causal features whose effect vector lies in family F beyond chance").
    # Confirmed per claim against the model's own signed random-direction null
    # (`analysis/family_claims.py`), Holm across the registered claims, with
    # `family_presence_n_null` Monte-Carlo replicates of the null count.
    register_family_presence_claims: bool = field(default=False, metadata={"omit_at_default": True})
    family_presence_min_dev: int = field(default=3, metadata={"omit_at_default": True})
    family_presence_n_null: int = field(default=2000, metadata={"omit_at_default": True})
    # Random directions per feature in the private family-presence battery
    # (0 = `concepts.n_null_directions`). The per-feature null probability is
    # resolved at 1/(n+1), so a larger value sharpens the null.
    family_presence_n_null_directions: int = field(default=0, metadata={"omit_at_default": True})
    # ROADMAP.md sec 41.1: the ablation null a V3-B claim is registered and scored
    # against ("" = the run's own `sae.ablation_null`, today's behaviour; or one of
    # `mean_magnitude`, `profile_matched`, `profile_matched_cov`). Each claim records
    # it as `null_mode`; `confirm` runs the private battery under it and refuses when
    # the dev artifact holds no result under that null.
    primary_null: str = field(default="", metadata={"omit_at_default": True})
    # `register_atlas_centroid_claims`: one `concept_atlas_centroid` claim per
    # seed-stable multi-model atlas concept with >= `atlas_centroid_min_members`
    # (default 3, fixed before the V3 data; the >= 2 models requirement stays)
    # members: the private centroid cosine against the dev centroid, vs the
    # same-composition random-member-set null, with no pair-fraction leg. The
    # legacy `concept_atlas` rule is untouched and registered beside it.
    register_atlas_centroid_claims: bool = field(default=False, metadata={"omit_at_default": True})
    atlas_centroid_min_members: int = field(default=3, metadata={"omit_at_default": True})
    # `external_path`: a second sealed corpus (role `external_real`) on which
    # `confirm` re-runs the registered transfer and family-presence claims,
    # written under `external_replication`, never counted in the confirm
    # verdict or ledger. Refused if it overlaps the dev or private corpus by
    # sample hash; one-shot like the private confirmation.
    external_path: str = field(default="", metadata={"omit_at_default": True})
    external_source: str = field(default="sealed", metadata={"omit_at_default": True})
    external_max_series: int = field(default=1024, metadata={"omit_at_default": True})


@dataclass
class ClusteringConfig:
    enabled: bool = True
    layer: str = "auto"
    k: str = "auto"
    pca_dim: int = 50
    max_series: int = 5000
    use_umap: bool = True


@dataclass
class LayerScreenConfig:
    enabled: bool = True
    method: str = "work_bend"       # work_bend | coverage | factor_emergence
    budget_frac: float = 0.25
    min_budget: int = 2
    max_series: int = 100_000       # cap on rows fed to the screen; effectively "all" by default
    use_curvature: bool = True      # work_bend only
    min_gap: int = 1                # 1 = pure top-k. >1 forces a minimum index gap between
                                    # selected layers, which DISCARDS high-scoring adjacent
                                    # layers -- unnecessary here because work_bend already
                                    # scores cross-layer change, so two adjacent high scores
                                    # are two distinct transformations, not one counted twice
    seed: Optional[int] = None      # falls back to run.seed
    stride: int = 1                 # screening capture stride, independent of models[*].capture_layer_stride
    require_full_capture: bool = True   # run a dedicated stride-1 screening extraction (ROADMAP.md sec 15 A1)
    keep_store: bool = False        # keep screen_activations.zarr after selection instead of deleting it


@dataclass
class BudgetConfig:
    """Parameter/FLOPs/latency/VRAM accounting (ROADMAP.md sec 18 F2).

    On by default and deliberately cheap: `batch` series through a handful of
    forward passes, no activation store, no analysis. The cost is a few
    seconds per model against the alternative of every quality comparison in
    the report staying size-confounded.
    """
    enabled: bool = True
    batch: int = 8              # series per timed pass; capped at the corpus size
    repeats: int = 5            # timed calls (median reported, spread recorded)
    warmup: int = 2             # untimed calls before timing starts
    measure_predict: bool = True    # also time the full forecast path
    predict_repeats: int = 3        # fewer: a sampled decoder's pass is expensive


@dataclass
class FrontendConfig:
    """Input front-end diagnostics (ROADMAP.md sec 16 E17): what does each
    model do to its input BEFORE any layer runs? On by default and cheap --
    mostly tokenizer calls and a handful of `predict()` calls, no activation
    store, no `extract` dependency (`--stages frontend` is a valid
    standalone run against a checkpoint with nothing else built, same
    pattern as `budget`).

    Each diagnostic has its own enable flag: a model whose
    architecture makes one inapplicable (e.g. `quantization_resolution` on
    a continuous-embedding model with no re-quantizing tokenizer) degrades
    to an explicit `not_applicable` record rather than skipping the whole
    stage (`CLAUDE.md` sec 2.5).
    """
    enabled: bool = True
    max_series: int = 64            # series sampled for every sub-diagnostic below

    quantization_resolution: bool = True   # Chronos-style re-quantizing tokenizers only

    scale_equivariance: bool = True
    # Multiplicative factors applied to the raw context before predicting;
    # the forecast is divided back by the same factor before comparing
    # against the unscaled prediction. 1000x/0.001x are deliberately extreme
    # (three orders of magnitude each way) to stress internal normalization.
    scale_factors: list = field(default_factory=lambda: [1000.0, 0.001])

    context_truncation: bool = True
    context_truncation_n_points: int = 5
    # Smallest AVAILABLE (non-stale) context length tested, as a fraction of
    # data.context_len -- the "small fraction of normal" end of the sweep.
    context_truncation_min_frac: float = 0.125

    nan_handling: bool = True
    # Fraction of context timesteps set to NaN per injection scenario
    # (front/middle/back), and how many series to test it on -- deliberately
    # small since this is a mechanism probe (does it raise / propagate /
    # handle), not an effect-size estimate needing a bootstrap CI.
    nan_frac: float = 0.05
    nan_series: int = 8

    # Constant-context probe (0, 1, 1e3, 1 + 1e-7 noise): finite forecast and
    # activations? `omit_at_default` so declaring it leaves every existing run's
    # `frontend` fingerprint byte-identical; an older frontend.json simply
    # lacks the `constant_context` key and the report says so.
    constant_context: bool = field(default=True, metadata={"omit_at_default": True})


@dataclass
class SAEConfig:
    enabled: bool = False
    checkpoints: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)  # [{"model": "...", "layer": "...", "dict_size": optional int}]
    # per-target "dict_size" overrides dict_size_mult*d_in for just that target
    # (ROADMAP.md sec 13 item 9) -- e.g. to reproduce Stage 0's matched dictionary
    # sizes (576 TimesFM / 512 Chronos-T5-Base) instead of the much larger,
    # hidden-width-multiple size dict_size_mult produces by default.
    dict_size_mult: int = 8
    k: int = 32
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 4096
    resample_dead_every_epochs: int = 0
    # AuxK dead-atom revival (ROADMAP.md sec 6.2.1 Stage 0, H4). Under TopK an
    # atom that stops winning the top-k competition gets exactly zero gradient
    # forever after; `aux_k > 0` adds an auxiliary term making the currently-
    # dead atoms reconstruct the live model's residual, so they keep receiving
    # one. `0` disables it (the default, so no recorded run's numbers move).
    aux_k: int = 0
    aux_coef: float = 0.03125
    aux_dead_steps: int = 20
    # Optimizer-step budget floor (ROADMAP.md sec 23.2 A1(b)). `0` reproduces
    # today's behavior bit-for-bit -- no already-recorded number moves.
    # `epochs` is a step count in disguise: steps_per_epoch = ceil(n_rows /
    # batch_size), so `epochs: 60` is ~120 optimizer steps on this repo's
    # ~4600-row production corpus and ~720 on Stage 0's ~46000-row store --
    # the same config field, a 4-6x different training budget, with nothing
    # recording which one actually happened until now (`sae/meta.json`'s new
    # `training_budget` field). When `min_train_steps > 0`, `train_sae` runs
    # `max(epochs, ceil(min_train_steps / steps_per_epoch))` epochs instead.
    min_train_steps: int = 0
    # AuxK's dead-atom warm-up (`aux_dead_steps`) is denominated in optimizer
    # steps, so a fixed value calibrated on a many-step run silently consumes
    # a much larger fraction of a few-step run's budget (`CLAUDE.md` sec
    # 11.26's lesson, recurring: a constant reapplied outside the conditions
    # it was measured under). `0.0` keeps `aux_dead_steps` an absolute count
    # (unchanged); `> 0` instead derives
    # `aux_dead_steps = max(1, int(frac * total_optimizer_steps))`.
    aux_dead_steps_frac: float = 0.0
    # Dictionary-size policy (ROADMAP.md sec 23.2 A1(c)). "mult" (default):
    # today's `dict_size_mult * d_in` behavior, unchanged -- no recorded
    # number moves. "search": train a ladder of candidate sizes on the
    # already-cached activations (cheap -- seconds per cell, no checkpoint
    # load) and keep the LARGEST one whose dead-feature rate clears
    # `max_dead_rate`. If none clears it, keep the one with the most ALIVE
    # atoms rather than the largest dictionary outright: a saturated layer
    # can have more dead atoms in a bigger dictionary with no gain in alive
    # count (A1 finding 3 -- Chronos `encoder.block.6` plateaus at ~200-400
    # alive atoms regardless of size), so "biggest, full stop" would
    # silently prefer a worse dictionary. The full ladder is recorded in
    # `sae/meta.json` either way.
    dict_size_policy: str = "mult"
    dict_size_ladder: list = field(default_factory=lambda: [128, 256, 384, 512, 768, 1024, 1536, 2048])
    # Dead-feature-rate acceptance bar (ROADMAP.md sec 23.2 A1(d)). Doubles as
    # `dict_size_policy: "search"`'s target AND as a rendered pass/fail gate
    # on every target's FINAL dictionary regardless of policy -- recorded as
    # `dead_rate_gate` in `sae/meta.json` and rendered as a visible warning
    # (not a collapsed note) in the report's SAE section when it fails.
    # Today a 97%-dead dictionary is a number in a JSON file nothing reads;
    # this is what makes it loud instead.
    max_dead_rate: float = 0.30
    # `dict_size_policy: "search"`'s own residual gap, found by the
    # 2026-08-21 `sae_revival` validation run (ROADMAP.md sec 23.2 A1): each
    # ladder candidate was trained and scored with exactly ONE stochastic
    # draw, so a boundary-case candidate (Chronos-T5-Base's chosen size later
    # measured a 5-seed mean dead rate of 0.304 against this single draw's
    # 0.223) or a collapsed one (TimesFM's own single draw at dict_size 2048
    # scored fidelity -0.99 against a real 5-seed floor never below 0.90,
    # because the `passing` filter never looks at fidelity at all) could be
    # selected by chance. Both default to a no-op (today's single-draw,
    # dead-rate-only behavior) so no already-recorded search result moves.
    min_fidelity: float = 0.0
    dict_size_search_seeds: int = 1
    # A third gap the two knobs above don't close, found by re-running the
    # fixed search against real checkpoints (ROADMAP.md sec 23.2 A1, second
    # validation-run block, 2026-08-21): `search_dict_size`'s own selection
    # rule -- the LARGEST dict_size whose mean dead rate clears
    # `max_dead_rate` -- always lands as close to the bar as the ladder's
    # granularity allows, regardless of measurement quality, whenever dead
    # rate rises steeply with size (Chronos-T5-Base's real ladder: 128 ->
    # 0.120 ... 256 -> 0.297 ... 384 -> 0.396 ... 2048 -> 0.820 -- nothing
    # above 256 clears the bar at all). Its chosen size (256) later measured
    # a real 5-seed floor of mean 0.304 (sd 0.011) -- one sd over 0.30 --
    # while size 128 (the only other passing candidate) sat at a comfortable
    # 0.120 on the same ladder, with no larger comfortably-passing
    # alternative available (an earlier version of this comment claimed one
    # existed at 512, quoting TimesFM's ladder values by mistake; corrected
    # same day). `dict_size_search_margin > 0` tightens the passing filter to
    # `dead_feature_rate <= max_dead_rate - margin`, trading dictionary size
    # for headroom -- for Chronos-T5-Base specifically that means falling
    # back to the much-smaller 128-atom dictionary, not a "free" larger one.
    # Default 0.0 is a no-op -- no already-recorded search result moves.
    dict_size_search_margin: float = 0.0
    # Seed-to-seed noise floor for this stage's own headline numbers
    # (ROADMAP.md sec 13's SAE repeat-run-variance item). `1` trains exactly
    # one SAE per target and changes nothing -- every already-recorded number
    # stays regenerable. `>1` retrains each target at `run.seed + 0..n-1`
    # against the same (already-frozen) activations and records the spread of
    # fidelity / dead rate / forecast-preservation dMASE beside the primary
    # seed's value, so the report can render "+0.17 +/- 0.12 (5 seeds)"
    # instead of a bare delta read against zero. Only the primary seed's SAE
    # is checkpointed and put through ground-truth alignment / ablation /
    # steering -- the extra seeds exist to size the floor, not to be analyzed.
    n_seeds: int = 1
    # ------------------------------------------------------------------
    # Held-out evaluation + dictionary admission gate (2026-09-11, user-
    # requested). Until now `reconstruction_fidelity` and `dead_feature_rate`
    # were measured on the SAME rows the dictionary trained on, so a
    # dictionary that memorized the corpus and one that generalized reported
    # the identical number, and nothing anywhere refused a target on
    # reconstruction quality.
    #
    # `holdout_frac` reserves that fraction of SERIES (never windows --
    # invariant 2; windows within a series are strongly dependent) as a
    # held-out split, drawn family-stratified so the split is not a
    # task-ordered prefix (CLAUDE.md sec 11.38). The dictionary trains on the
    # remaining series only; fidelity, dead rate and forecast preservation
    # are then reported on BOTH splits. `0.0` disables the split entirely and
    # reproduces every previously-recorded SAE number bit-for-bit.
    holdout_frac: float = 0.2
    # The admission gate itself. A target clearing neither bar is not silently
    # rendered beside the ones that do -- `admission` in `sae/meta.json` says
    # which bar it missed, `layer_substitution` (below) retries another layer
    # of the SAME model, and the report refuses to draw feature-level
    # conclusions from a dictionary that never passed.
    #
    # `min_fidelity_gate` is checked against the HELD-OUT fidelity when a
    # split exists (train fidelity is what a memorizing dictionary inflates).
    # 0.85 admits every target of `runs/full_report_run_4model` except
    # Sundial's three (0.793/0.830/0.804); 0.90 would additionally reject
    # TimesFM stacked_xf.6 (0.890) and .10 (0.891) and Chronos-2 block.6
    # (0.893), which is why the default is the looser of the two the user
    # named -- it isolates the one model that actually fails.
    min_fidelity_gate: float = 0.85
    # The forecast-preservation bound, in |dMASE|. Checked at TOKEN
    # granularity, NOT window: the window number carries a per-architecture
    # broadcast confound large enough to dominate it (see
    # `eval.py::forecast_preservation`'s docstring, and CLAUDE.md sec 13's
    # correction chain). This is exactly the "high fidelity but an abnormal
    # MASE increase, which seems like a contradiction" case -- on the 4-model
    # run Chronos-Bolt `encoder.block.4` has the BEST fidelity of all 13
    # targets (0.955) and a window dMASE of +0.378, while its token dMASE is
    # -0.135. The contradiction is the confound, not the dictionary.
    # `0.0` disables the dMASE half of the gate.
    max_abs_delta_mase: float = 0.25
    # Layer substitution (2026-09-11, user-requested). When a target fails the
    # gate, retry with the next-best-scoring CAPTURED layer from that model's
    # own `layer_screen` ranking, up to this many extra attempts. A model
    # whose layers all fail is dropped from the SAE section WITH A STATED
    # REASON rather than rendered as though it had passed. `0` disables
    # substitution (a failing target is still recorded, just not retried).
    # Only applies to auto-resolved targets -- a pinned `sae.targets` entry
    # is an explicit choice and is never silently replaced.
    layer_substitution_attempts: int = 2
    forecast_preservation_max_series: int = 64
    ground_truth_max_series: int = 2000
    # Label-permutation null for ground-truth alignment (ROADMAP.md sec 16
    # E9): `best_ground_truth_matches` picks each feature's *best* of many
    # candidate fields, which inflates `mean_abs_rho_matched` above zero even
    # under pure noise (max-of-many-comparisons). Repeating the identical
    # search with feature-to-series correspondence permuted gives a real
    # floor to compare the headline number against. `0` disables it.
    ground_truth_permutation_repeats: int = 3
    ground_truth_permutation_max_features: int = 2000
    real_data_enabled: bool = False
    real_data_source: str = "Monash-University/monash_tsf"
    real_data_n_windows: int = 20_000
    real_data_pool_limit: int = 2000
    # Feature-level ablation (ROADMAP.md sec 7 bullet 3 / sec 16 E15's second
    # half): zero each of the top `feature_ablation_top_k` ground-truth-
    # matched features (by |rho|) one at a time in the token-level SAE
    # reconstruction and measure the forecast MASE impact. Off by default --
    # it costs one extra full forward pass per ablated feature on top of
    # `forecast_preservation`'s own two, and is only meaningful once that
    # check's token-granularity number shows the *intact* reconstruction
    # preserves the forecast reasonably well for this model/layer.
    feature_ablation_enabled: bool = False
    feature_ablation_top_k: int = 8
    feature_ablation_max_series: int = 64
    # Feature steering (ROADMAP.md sec 16 E14): push each of the top
    # `feature_steering_top_k` ground-truth-matched features (by |rho|) up
    # and down by `feature_steering_strength_sigma` standard deviations (of
    # that feature's own clean activation) and check whether the forecast's
    # own directional metric (trend slope / seasonal-band magnitude) moves
    # the way the feature's signed correlation with its matched field
    # predicts -- a sharper causal claim than ablation's "this feature
    # matters at all". Off by default for the same cost reason as ablation
    # (two extra forward passes per steered feature, not one).
    feature_steering_enabled: bool = False
    feature_steering_top_k: int = 8
    feature_steering_max_series: int = 64
    feature_steering_strength_sigma: float = 2.0
    # The encode-store seam (ROADMAP.md sec 6.2.1 Stage 3d): persist each
    # target's trained encoder's output back into the store
    # (`sae`/`sae_pooled/{model}/{layer}`) and, once >=2 targets span both
    # comparison-pair models, compute a feature-space CKA between them
    # (`l1/cka_sae.json`). Off by default -- a wide dictionary's window-level
    # feature array (`dict_size_mult` defaults to 8x the raw activation
    # width) can be many times the raw activation store's own size, so this
    # should not silently multiply every run's disk footprint.
    persist_features: bool = False
    # ROADMAP.md sec 25.9 Stage 3 (Component B(b)): the number of ROLES a
    # target's probed candidate features are clustered into, chosen the way
    # `clustering.py::_resolve_k` already chooses its own k -- "auto" clips
    # to a small range rather than the family count (there is no analogous
    # "family count" for a set of ~40 candidate features), an explicit int
    # pins it. `role_min_silhouette` is the bar `sae/roles.py::cluster_roles`
    # reports against, not a hard gate -- a dictionary that clusters poorly
    # at this granularity is a real finding (sec 25.13 item 2), rendered as
    # "non-modular" rather than silently forcing a partition anyway.
    role_k: str = "auto"
    role_min_silhouette: float = 0.1
    # ROADMAP.md sec 25.9 Stage 4 (Component C): explicit paths to a
    # `random_init` untrained-twin RUN DIRECTORY for one or more of this
    # run's models, keyed by the model name as configured HERE (e.g.
    # {"TimesFM": "runs/null_timesfm_random"}). There is no auto-discovery
    # mechanism anywhere in this repo (`analysis/null_baseline.py`'s own
    # convention already requires an explicit run path) -- a twin lives in a
    # SEPARATE run directory containing both the real model and its
    # `random_init: true` copy as two models of one small config, with its
    # own roles artifact built the same way as this run's -- read by
    # `sae/role_matching.py::untrained_twin_role_floor` as
    # `sae/roles_injection.json` (ROADMAP.md sec 30, Stage 4, 2026-09-11).
    # Empty (the
    # default) means no floor is available and any role-matching
    # shared-fraction number is rendered non-quotable with a stated reason
    # rather than silently omitted (sec 25.6's mandatory pairing).
    role_matching_untrained_twin_runs: dict = field(default_factory=dict)
    role_matching_cosine_threshold: float = 0.5
    role_matching_depth_tolerance: float = 0.15
    # ROADMAP.md sec 26 C, 2026-09-04, on user review: "it is not clear what
    # exactly each feature does as many have very similar names and activate
    # for the same series." When true, `run_sae_describe.py` additionally
    # measures what each feature's OWN top-firing series score on every
    # structural ground-truth field, against the corpus median and spread,
    # and hands the most unusual of those to the narrator as evidence. That
    # is what lets two features sharing a `structural_field` -- routinely
    # most of a target's top rows -- receive descriptions that differ, since
    # without it every field the guard licenses them to mention is the same.
    # Provenance fields, and any field this run's alignment refused as
    # inseparable from provenance (sec 11.48), are excluded: a contrast on
    # those reports how the benchmark was built.
    #
    # Default true because the evidence is a pure reduction over artifacts
    # the exemplar pass already loads (no extra forward pass, no extra model
    # load); `false` reproduces the pre-2026-09-04 packets exactly.
    # `stage_input: False` keeps this out of the `sae` stage's config
    # fingerprint (manifest.py). The narrator is a standalone script reading
    # finished artifacts -- this knob changes `sae/descriptions.json` and no
    # artifact the stage itself writes, so fingerprinting it would refuse
    # every existing run's report re-render over a field that stage never
    # reads. That is the false-refusal shape of CLAUDE.md sec 11.35.
    describe_from_exemplars: bool = field(
        default=True, metadata={"stage_input": False})
    # ROADMAP.md sec 30.6 -- re-clustering SAE features on their ABLATION
    # fingerprint into named CONCEPTS (`sae/concepts.py`) and testing whether
    # a concept's top-firing series are grouped the same way by another
    # model's dictionary (`sae/transfer.py`). Defaults reproduce sec 30.1/
    # 30.2's own measurements. `concept_causal_only`/`concept_k`/
    # `concept_min_silhouette` mirror `cluster_concepts`'s own parameters;
    # `concept_misfit_cosine_gap` is `misfits.py`'s relative-cohesion bar
    # (sec 30.4.3); `concept_cards_max` and `interest_weights` are report-
    # time-only (which cards render, and how they rank) and read by nothing
    # the `sae` stage itself writes, so both carry `stage_input: False` --
    # the same false-refusal shape `describe_from_exemplars` above already
    # guards against (`CLAUDE.md` sec 11.35/sec 11.51).
    concepts_enabled: bool = True
    concept_causal_only: bool = True
    concept_k: str = "auto"
    concept_min_silhouette: float = 0.1
    # ROADMAP.md sec 32.5 Item D: `cluster_concepts`'s k="auto" path sweeps
    # k in range(2, k_max+1) (k_max = min(8, n // concept_min_members, n-1))
    # instead of a single ratio-derived k, and rejects any k whose smallest
    # cluster has fewer than concept_min_members rows BEFORE comparing
    # silhouette scores -- a singleton's silhouette is maximal by
    # construction, so the size floor must gate before the score is
    # compared, not after. Default 3: fewer than 3 does not need Item D's
    # protection (a pair is already rare enough to eyeball) and 3 is what
    # the acceptance criterion (0 singleton concepts) requires as a floor.
    # `concept_min_members<=1` is a SENTINEL that bypasses the sweep
    # entirely and reproduces the pre-D ratio-based single fit bit for bit
    # (D5) -- the reproducibility knob, not a looser version of the floor.
    concept_min_members: int = 3
    concept_misfit_cosine_gap: float = 0.3
    concept_cards_max: int = field(default=24, metadata={"stage_input": False})
    # sec 30.4.2's cross-model transfer test: `transfer_top_k` is each
    # concept's/feature's top-firing series count, `transfer_n_null` the
    # matched-null draw count (the acceptance band's width is a property of
    # this value -- raising it narrows the band at linear cost, sec 30.9
    # criterion 2), `transfer_seed` the stable base seed `_seed()` mixes into
    # every per-(concept, target) draw (`CLAUDE.md` sec 11.2 -- never
    # Python's builtin `hash()`).
    transfer_enabled: bool = True
    transfer_top_k: int = 20
    transfer_n_null: int = 200
    transfer_seed: int = 0
    interest_weights: tuple = field(default=(0.5, 0.3, 0.2),
                                    metadata={"stage_input": False})
    # The random-direction null of the ablation battery (`sae/response.py::
    # feature_ablation_fingerprints`). `profile_matched` (DEFAULT since the
    # legacy null failed the known-answer gate on 5/5 seeds, FINDINGS MN-29):
    # per feature, random directions removed with that feature's own per-token
    # profile (n_null extra forwards per feature). `mean_magnitude` (legacy,
    # pin it to reproduce older runs): one uniform removal size per chunk.
    # Declared field by field on the concepts stage (`sae.ablation_null`) and
    # left out of the `sae` stage's inputs. `omit_at_value`, not
    # `omit_at_default`: the key is omitted at the LEGACY value, so a pinned
    # legacy run keeps its old fingerprint and a default run (key present)
    # reads an old legacy concepts artifact as stale.
    # `profile_matched_cov`: the same per-feature profile with the direction drawn
    # from the chunk's own token covariance (on-manifold).
    ablation_null: str = field(default="profile_matched",
                               metadata={"stage_input": False,
                                         "omit_at_value": "mean_magnitude"})
    # Opt-in: also record the battery's leave-one-draw-out empirical chance clear
    # rate (`empirical_chance` in `*_ablation.json`); default off, artifacts unchanged.
    ablation_empirical_chance: bool = field(default=False, metadata={"stage_input": False,
                                                                     "omit_at_default": True})
    # Opt-in dual-null battery (ROADMAP.md sec 41.1): every null named here is scored in
    # ONE battery pass (the real ablation forward runs once), the first is PRIMARY and
    # must equal `ablation_null` (a config that leaves `ablation_null` out gets it set
    # to the first; naming both and disagreeing is an error). The others are written
    # per candidate under `by_null[<mode>]` and as `by_null_summary`; everything
    # downstream (concepts, atlas, families, profiles, transfer) reads the primary's
    # legacy keys. Cost: n_null extra forwards per feature per extra mode. Empty
    # (default) = a single-null run, artifacts unchanged.
    ablation_nulls: tuple = field(default=(), metadata={"stage_input": False,
                                                        "omit_at_default": True})
    # Opt-in (ROADMAP.md sec 41, V3-B): keep every null draw's SIGNED per-channel mean delta
    # (`null_draw_signed_means`, which implies `keep_null_draws`) in `*_ablation.json`; the
    # family-presence claim scores a family's cosine against these. Default off, artifacts unchanged.
    keep_signed_null_draws: bool = field(default=False, metadata={"stage_input": False,
                                                                  "omit_at_default": True})


@dataclass
class ReportConfig:
    enabled: bool = True
    title: str = "TSFM Comparison Report"
    verbose: bool = False
    verbose_series: int = 3
    # A section builder raising is a bug (ROADMAP.md sec 15 A5), so the
    # default is to surface it as a hard failure (non-zero exit) rather than
    # let a report with a silently-dropped section report success. CLI
    # --allow-partial-report (or setting this true) downgrades that to a
    # loud warning, matching --allow-stale's precedent.
    allow_partial: bool = False


@dataclass
class RunConfig:
    name: str = "run"
    out_dir: str = "runs"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 0
    keep_models_loaded: bool = False


@dataclass
class CorpusConfig:
    """The corpus card (ROADMAP.md sec 34 item C1): what the benchmark actually
    is, before any model result. Needs no model and no `tsfm_benchmark`/
    `benchmark_validation` import -- it reads `data.*` (already a stage input
    to every other stage) plus, optionally, the sealed corpus's own
    `manifest.json` (written by `build_pipeline.seal.seal_corpus`, carrying
    item B1's `extra.audit` leakage/near-duplicate/realism verdict when
    present) and an already-produced `benchmark_validation` report pointed at
    by `validation_report`.

    Both fields genuinely gate what the `corpus` stage itself computes and
    writes (`enabled` whether it runs at all; `validation_report` which extra
    block the card composes), so neither needs `stage_input: False` --
    changing either legitimately invalidates `corpus/card.json` and should
    refuse a stale skip (`CLAUDE.md` sec 15 A3).
    """
    enabled: bool = True
    # Path to an existing `benchmark_validation` `validation_report.json` for
    # THIS corpus, or "" (default) to skip that section of the card. Left
    # unset rather than auto-discovered: `tsfm_lens` must not invoke
    # `benchmark_validation` itself (a separate, heavy-dependency package,
    # `CLAUDE.md` sec 3) to produce one, and guessing a path next to
    # `data.path` would silently pair a stale or unrelated report with this
    # run's corpus. The card cross-checks the report's own `corpus_digest`
    # (set by `benchmark_validation.report` alongside it, item C1.3) against
    # this run's `data.py::BenchmarkData.corpus_digest` and REFUSES to render
    # the validation summary -- rather than trust a path match -- on a
    # mismatch, naming both digests.
    validation_report: str = ""


@dataclass
class ConceptsConfig:
    """The `concepts` stage (ROADMAP.md sec 37.4 P1): ablation battery ->
    concept clustering -> cross-model transfer -> deterministic descriptions,
    over every target the `sae` stage trained.

    Off by default: it costs a patched forward pass per (candidate, series,
    null direction) per target, and it needs `sae.persist_features: true`,
    which is itself off by default. The clustering and transfer knobs stay in
    `sae:` (`concept_*`, `transfer_*`) where they were first recorded --
    moving them would change the `sae` stage's resolved fingerprint and
    refuse every existing run as stale (`CLAUDE.md` sec 11.51); this stage
    declares them as field-level fingerprint keys instead. The ablation knobs
    below are `run_sae_ablation.py`'s own CLI defaults, so a stage run and a
    script run with defaults are the same measurement.
    """
    enabled: bool = False
    top_k_series: int = 8
    n_null_directions: int = 16
    max_series: int = 64
    keep_forecasts: int = 3
    n_features_per_rule: int = 12
    describe: bool = True
    describe_top_features: int = 8

    # ROADMAP.md sec 37.15 q6 (option b) -- the cross-model CONCEPT ATLAS
    # (`sae/concept_atlas.py`): pools every target's causal ablation
    # fingerprints into one space and clusters ACROSS models, unlike
    # `concept_*` above (which clusters one target against its own peers
    # only). Lives in THIS section, not `sae:`, because unlike the
    # `concept_*`/`transfer_*` knobs it has no pre-existing `sae`-stage
    # fingerprint to protect -- it is new, so it can be declared where it is
    # actually consumed. `atlas_min_cosine: 0.9` was chosen against the
    # column-shuffle structure null on runs/full_report_run_4model (ROADMAP.md
    # sec 37.4 atlas Findings): at 0.8 the real 33 concepts sit barely above a
    # null mean of 26.04 (so most would form from shuffled channels too); at
    # 0.9 the real 19 are about twice the null mean of 9.62 (p 0.005). Lower
    # values buy coverage at the cost of concepts chance also produces.
    atlas_enabled: bool = True
    atlas_min_cosine: float = 0.9
    atlas_min_members: int = 3
    atlas_n_null: int = 200

    # ROADMAP.md sec 37 (concept FAMILIES, `sae/concept_families.py`): a
    # general, human-readable layer above the tight atlas concepts above,
    # on user review ("many concepts with only one feature, and some
    # features very close that are not in that concept ... should be more
    # interpretable and general"). Average-linkage cosine hierarchical
    # clustering, cut at a silhouette-selected threshold, THEN nearest-
    # centroid assignment of every pooled feature at `atlas_family_assign_min`
    # -- a looser, explainable-by-one-rule membership on top of
    # `atlas_min_cosine`'s tight, every-pair-clears-it concepts above (which
    # this does not change). `atlas_family_min_members` is this layer's own
    # size floor (default 5, looser than `atlas_min_members` because a
    # family is meant to be coarser than a concept).
    #
    # `stage_input: False` on both: `sae/concept_families.py::
    # run_concept_families` is the ONLY reader of these two fields, and it
    # reads the SAME already-fingerprinted ablation files `run_concept_atlas`
    # does (via `pooled_features`) -- no new forward pass, no new model load,
    # seconds of CPU. Fingerprinting them under this whole-section `concepts`
    # key (CLAUDE.md sec 6.1) would mark every existing run's `concepts`
    # stage stale for a change that cannot touch anything upstream of the
    # family layer -- `describe_from_exemplars`'s own precedent above (in
    # `SAEConfig`) for the identical false-refusal shape (CLAUDE.md sec
    # 11.35/sec 11.51). A run whose family knobs alone changed self-skips on
    # its existing `concept_stage.json`; re-deriving families from the
    # unchanged ablation files with the new knobs is a fresh call to
    # `run_concept_families`, not a full stage rerun.
    atlas_family_min_members: int = field(default=5, metadata={"stage_input": False})
    atlas_family_assign_min: float = field(default=0.5, metadata={"stage_input": False})

    # Orchestrator review of F1 (ROADMAP.md sec 37): the grid's silhouette was
    # first scored on each threshold's KEPT DRAFT members only, which rewards
    # the tightest cut by construction (fewer, better-separated survivors) --
    # measured on runs/full_report_run_4model, kept members fall from 194 at
    # 0.5 to 86 at 0.85 while draft silhouette keeps climbing, so the sweep
    # always picked the loneliest, most over-split threshold (0.85, 14
    # families). Scored on the FINAL partition instead (every row the
    # nearest-centroid step actually assigns), the real silhouette is flat
    # (~0.31-0.34) across the whole grid: the ablation-feature space is a
    # continuum, not a set of well-separated clusters, so there is no
    # "correct" k to recover -- the honest choice is the COARSEST one that
    # does not cost real separation. `atlas_family_sil_tol` is that slack:
    # among admissible thresholds, keep the fewest families whose final
    # silhouette is within this much of the best. `stage_input: False` for
    # the same reason as the two fields above (same reader, same cheap CPU
    # recompute over already-fingerprinted files).
    atlas_family_sil_tol: float = field(default=0.02, metadata={"stage_input": False})

    # ROADMAP.md sec 37 P2 -- seed stability of atlas concepts, and the
    # within-model transfer ceiling. Is an atlas concept a property of the
    # model, or of one SAE draw? For each already-trained target, this many
    # independently-seeded SAE dictionaries are trained in total (the
    # primary, seed `run.seed`, plus `n_sae_seeds - 1` replicates at
    # `run.seed + 1 .. run.seed + n_sae_seeds - 1`), each at the PRIMARY's
    # own recorded `dict_size` (never re-searched -- a replicate at a
    # different capacity would confound stability with capacity,
    # `sae/train.py::train_sae_replicate`). `1` disables replicate training
    # entirely (no stability measurement, no extra wall-clock cost). Default
    # `3`: the primary plus 2 replicates.
    n_sae_seeds: int = 3

    # ROADMAP.md sec 37 P3 -- p-values and FDR control for `sae/transfer.py`'s
    # cross-model transfer tests, plus their extension to the atlas above
    # (`run_atlas_transfer` -> `sae/atlas_transfer.json`). `transfer_p_method`
    # only changes behavior for a leg whose EXACT permutation p already hits
    # the floor `1/(sae.transfer_n_null + 1)` -- "exact" leaves it at the
    # floor (honest, coarse); "adaptive" redraws a larger null (up to
    # `transfer_max_redraw`) from the SAME seed, so a lone strong effect in a
    # large BH family can still be declared. `transfer_fdr_q` is the Benjamini-Hochberg target
    # FDR, applied separately per ordered (source model, destination model)
    # pair and per leg (never pooled across pairs -- CLAUDE.md sec 6.5's
    # per-family discipline for Holm, applied to BH here).
    transfer_p_method: str = "exact"
    transfer_fdr_q: float = 0.05
    transfer_max_redraw: int = 5000

    # ROADMAP.md sec 37 Spec A -- per-concept profiles (`sae/concept_profiles.py`):
    # what a concept fires on, whether the models that share its effect also
    # share its inputs, and whether the model that has it forecasts better
    # because of it. Additive on top of the atlas/stability/atlas-transfer
    # artifacts above; runs only when the atlas itself ran, since a profile is
    # a property of an atlas concept. `profile_n_perm` is the permutation-null
    # draw count for the search-corrected structural-field test (`concepts.py`
    # sec 37 Spec A item 2's `p_max_structural`).
    profiles_enabled: bool = True
    profile_n_perm: int = 1000

    # ROADMAP.md sec 37.8 P5a -- `analysis/response_reach.py::reach_probe`'s
    # gate, relative rather than absolute (sec 29.5 found the absolute
    # `_EPS=1e-12` admits a numerically dead model: Chronos-2's and
    # Chronos-Bolt's untrained twins, patched with a provably-different
    # (clean x1.5, fp32) replacement, moved the forecast by a relative
    # 6.5e-08 and 9.3e-08 -- four orders of magnitude above `_EPS` while
    # being, relative to the forecast, noise). `reachable` now requires
    # `cross_delta / forecast_scale > min_relative_reach`, where
    # `forecast_scale` is the mean absolute clean forecast the probe already
    # computes. `1e-3` is judgment, set between the measured populations
    # (sec 37.8's Findings): the dead Chronos-2/Chronos-Bolt twins reach at
    # most 3.47e-05 relative (29x below), and the live minimum over the 4-model
    # run's 13 SAE targets is 0.1125 (112x above, meeting the design's
    # "at least 100x below live" rule). 1e-4 left only 3x to the Bolt twin.
    # Lives in `concepts:`, not `sae:`, for the same
    # reason `atlas_min_cosine` does (no pre-existing `sae`-stage fingerprint
    # to protect); the `concepts` Stage already declares the whole
    # `concepts` section as a fingerprint key, so this field marks existing
    # `concepts` artifacts stale without any other change (`CLAUDE.md`
    # sec 6.1).
    min_relative_reach: float = 1e-3

    # ROADMAP.md sec 37.7 P4 -- causal fingerprints with the level shift
    # separated. `sae/response.py::feature_ablation_fingerprints` always
    # computes `level_share` (sec 37.3 P0's ratio) and a level-removed
    # `shape_channels` block per candidate (no extra forward pass -- both are
    # read off the SAME patched forecast already computed for `channels`), so
    # this knob is judgment only: how high a concept's members' MEDIAN
    # `level_share` must sit, with no level-removed channel clearing, before
    # the atlas tags it `level carrier` rather than `shape-causal` or `no
    # measured effect` (`sae/concept_atlas.py::_tag_causal_effect`). Sec
    # 37.3 P0 measured the pooled median at 0.535 and the per-model medians
    # at 0.687/0.542/0.467/0.297 (TimesFM/Sundial/Chronos-2/Chronos-Bolt),
    # with 20.6% of candidates above 0.9 -- 0.9 is set high enough that only
    # the candidates P0 already called "almost entirely level" qualify, not
    # the roughly half whose level share is merely majority.
    level_share_threshold: float = 0.9

    # ROADMAP.md sec 37.8 P5b -- cross-model causal agreement, measured on
    # the SAME shared series (`sae/shared_input_agreement.py`), for every
    # FDR-surviving reciprocal atlas-transfer test. `shared_input_enabled`
    # gates the whole step (it costs a patched forward pass per (real +
    # `shared_input_n_null` matched-null) feature-set ablation, per side, per
    # test); off skips with a stated reason and drops a stale artifact.
    # `shared_input_n_null` is the number of random ALIVE feature sets each
    # side draws, matched to its own real set's size and decile of mean
    # pooled activation on the shared series, to build that side's own
    # within-run floor for both agreement statistics (never an untrained-twin
    # floor -- sec 37.8's whole point is that one is obtainable for only 1 of
    # 6 pairs). Lives in `concepts:`, not `sae:`, for the same reason
    # `min_relative_reach` does: no pre-existing `sae`-stage fingerprint to
    # protect, and the `concepts` Stage already declares the whole section as
    # a fingerprint key.
    shared_input_enabled: bool = True
    shared_input_n_null: int = 50

    # ROADMAP.md sec 38.3.2 (K3) -- opt-in cap on the shared-input agreement
    # step's cost. It is the most expensive step of the concepts stage and
    # its test count grows with the number of directed model pairs (1,223
    # tests / 10,855 s on 4 models). `None` (default) scores every
    # reciprocal-FDR transfer, byte-identical to earlier runs. An integer N
    # keeps, per ordered (source model, destination model) pair, the N tests
    # with the largest reciprocal transfer margin (the smaller of the
    # forward and reverse `auc - null_p95`), ties broken by
    # (concept, source target, destination target, feature); the rest are not
    # scored, and `shared_input_agreement.json` records `agreement_cap` with
    # the per-pair kept/dropped counts. The report names the cap wherever it
    # shows agreement counts. A cap selects on transfer strength, so the
    # scored tests over-represent strong input agreement; read the verdict
    # shares as conditional on that selection. Part of the whole-section
    # `concepts` fingerprint, so setting it marks an older concepts artifact
    # stale.
    agreement_max_tests_per_pair: Optional[int] = None

    # ROADMAP.md sec 38.3.5 (L5 known-answer study) -- three opt-in variants of
    # the shared-input agreement rung, each left out of the stage fingerprint
    # at its default so every existing run and artifact is unchanged.
    # `agreement_dst_set` chooses what is ablated on the DESTINATION side:
    # "feature" (default) the transfer test's one best feature (or its atlas
    # part when that feature is in one); "concept_part" the atlas part of the
    # same concept at the destination target when it has one (else the default
    # rule); "matched_set" the destination's top-N features by forward AUC on
    # the source concept's top series, N the source set's size, which is
    # defined for every test. `agreement_k_top_series` sets the size of each
    # side's top-series set, hence of the shared series `U`; `None` keeps the
    # atlas-transfer test's own k (`sae.transfer_top_k`). `agreement_partial_
    # rung` adds a per-test `rung` and a `partial_agreement` summary that
    # counts "level only" and "shape only" as `partial`, kept apart from
    # "same".
    agreement_dst_set: str = field(default="feature", metadata={"omit_at_default": True})
    agreement_k_top_series: Optional[int] = field(default=None, metadata={"omit_at_default": True})
    agreement_partial_rung: bool = field(default=False, metadata={"omit_at_default": True})
    agreement_require_defined_firing: bool = field(default=False, metadata={"omit_at_default": True})

    # ROADMAP.md sec 40 / 41 V3-C -- the subspace agreement test
    # (`sae/subspace_agreement.py`), an opt-in redesign of the shared-input rung:
    # the whole concept is ablated as a subspace on each side, and the result is an
    # ESTIMATE (effect concordance and level-removed shape cosine with a series-bootstrap
    # CI, TOST equivalence), not a verdict from point statistics. It is not run by any
    # stage; the keys below parameterize its function and, left at their defaults, are
    # omitted from every fingerprint. `subspace_agreement_n_null` random equal-dimension
    # subspaces per side form the matched floor; `_n_boot` series-bootstrap resamples;
    # `_ci_level` the two-sided CI level (agree needs the lower bound above the floor q95,
    # differ the upper bound below the floor q05); `_margin` the TOST margin in null units.
    subspace_agreement_n_null: int = field(default=50, metadata={"omit_at_default": True})
    subspace_agreement_n_boot: int = field(default=1000, metadata={"omit_at_default": True})
    subspace_agreement_ci_level: float = field(default=0.95, metadata={"omit_at_default": True})
    subspace_agreement_margin: float = field(default=1.0, metadata={"omit_at_default": True})

    # ROADMAP.md sec 37.9 P6a -- generator-side input counterfactuals
    # (`tsfm_benchmark/build_pipeline/counterfactual.py`'s draw-neutral knobs,
    # measured by `tsfm_lens/sae/counterfactual.py`): does an atlas concept's
    # causal-feature score actually RESPOND to the structural property its
    # generator controls, rather than merely correlating with naturally
    # varying series? Off by default (`cf_enabled`): it re-runs a forward
    # pass per (series, dose) per (target, knob) on top of the already-heavy
    # `concepts` battery, and needs `tsfm_benchmark` importable (a separate
    # install; degrades with a stated reason, `{"status": "unsupported", ...}`,
    # rather than failing, when it is not). `cf_max_series` bounds the
    # per-knob series count (`utils.sample_rows`-stratified, never a head
    # slice -- CLAUDE.md sec 11.38); `cf_doses` is the multiplicative dose
    # ladder applied to each knob's own recorded value (1.0 must reproduce
    # the corpus series bit-identically -- asserted inside the measurement,
    # not only in tests); `cf_n_null` is the number of matched random
    # alive-feature sets (`shared_input_agreement.matched_null_sets`'s own
    # decile-matching logic, reused rather than re-derived) that build each
    # concept's response floor. This step measures INPUT RESPONSE only --
    # mediation and the cross-model comparison are P6b, gated on this step's
    # go/no-go result. Lives in `concepts:`, not `sae:`, for the same reason
    # `min_relative_reach`/`shared_input_enabled` do: no pre-existing
    # `sae`-stage fingerprint to protect, and the `concepts` Stage already
    # declares the whole section as a fingerprint key.
    # `cf_n_null` 2000, not 200: the null's p floor is 1/(n_null+1), and with
    # ~25 concept x knob tests per target BH cannot pass anything at 200
    # (25/201 = 0.124 > 0.05; P6a go/no-go, ROADMAP sec 37.9). The null is
    # scored on already-encoded features, so draws cost no forward passes.
    cf_enabled: bool = False
    cf_max_series: int = 64
    cf_doses: list = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    cf_n_null: int = 2000

    # Encode-check gate (P6a go/no-go, ROADMAP sec 37.9): a fresh dose=1.0
    # recompute is compared against the persisted store not by an absolute/
    # relative tolerance on raw SAE-feature VALUES (an absolute epsilon
    # cannot tell precision noise from a wrong space, CLAUDE.md sec 8 --
    # measured directly: a real run's raw activations differed by a 0.17%
    # relative max diff, bf16-scale precision noise, yet the whole feature
    # VECTOR failed a 1e-3/1e-2 allclose because sparse TopK SAEs have many
    # near-zero features an absolute atol cannot survive), but by whether
    # each atlas concept part's OWN pooled score -- the only thing this
    # module actually reads downstream -- ranks series the same way in both
    # spaces. `cf_encode_min_spearman` is a judgment call, not a derived
    # constant: 0.98 demands the fresh and stored per-series concept scores
    # be nearly rank-identical, loose enough to absorb genuine bf16 noise on
    # a smooth, well-populated feature, tight enough that a misaligned or
    # substituted array (wrong layer, wrong row order) -- which decorrelates
    # rather than merely perturbing -- cannot pass by chance. The observed
    # per-part Spearman distribution is always reported in
    # `counterfactual_response.json` so this threshold can be recalibrated
    # against real data rather than guessed twice.
    cf_encode_min_spearman: float = 0.98

    # ROADMAP.md sec 37.10 P7 -- how many `concept_transfer` claims `register`
    # freezes from dev artifacts before the private epoch is ever touched.
    # Candidates are reciprocal-FDR atlas-transfer tests
    # (`sae/atlas_transfer.json`) whose atlas concept is stable
    # (`sae/concept_stability.json`), ranked by dev AUC margin; `n_registered`
    # is the cut. More claims spend more of the one private look and tighten
    # Holm (`CLAUDE.md` sec 6.6's p-floor), so this is judgment, not derived:
    # 20, decided by the reviewing session (ROADMAP.md sec 37.15 q2).
    # `stage_input: False`: read only by `analysis/hypotheses.py::run_register`,
    # never by `run_concept_stage`, so it must not fingerprint the `concepts`
    # stage itself (the false-refusal shape `describe_from_exemplars` above
    # already guards against); `register`'s own `Stage.config_keys` declares
    # `concepts.n_registered` as a field-level key instead.
    n_registered: int = field(default=20, metadata={"stage_input": False})

    # ROADMAP.md sec 37.10 P7b -- which claim a registered `concept_transfer`
    # entry actually makes. `"search"` (default) is P7's own claim: the
    # destination's WHOLE dictionary is re-searched on private data against a
    # max-over-features null, so "confirmed" means only "the destination
    # layer has SOME feature that selects the source concept's series" (P7's
    # Findings: the private argmax feature matched dev's in just 6 of 20
    # claims). `"frozen"` is the sharper claim registered for a fresh
    # epoch: BOTH `src_features` and dev's own best `dst_feature` are frozen
    # at registration, and the private forward leg scores exactly that one
    # feature against a SINGLE-FEATURE stratum-matched null (no max over the
    # dictionary, since there is no search left to correct for) --
    # `sae/transfer.py::transfer_one_fixed_feature`. `"search"` must
    # reproduce P7's registration and replication byte-identically
    # (`test_search_mode_unchanged`); this field is read only by
    # `analysis/hypotheses.py`/`analysis/confirm.py`'s concept-transfer path,
    # so it is declared `stage_input: False` here (same reasoning as
    # `n_registered` above) and field-level on `register`'s own
    # `Stage.config_keys` instead.
    transfer_claim_mode: str = field(default="search", metadata={"stage_input": False})

    # ROADMAP.md sec 38.2 (K2) -- how many `concept_causal` claims `register`
    # freezes from the dev `*_ablation.json` artifacts (>= 4 per model, seed-
    # stable atlas members preferred), and how many of the dev "acts
    # differently" shared-input agreement verdicts join the (all) "same
    # causal effect" ones. Both are judgment counts (sec 38.2.2). Read only by
    # `analysis/hypotheses.py`, hence `stage_input: False` here and
    # field-level on `register`'s own `Stage.config_keys`; `omit_at_default`
    # keeps older runs' `register` fingerprint unchanged (`CLAUDE.md` sec
    # 11.51).
    n_registered_causal: int = field(default=32, metadata={"stage_input": False,
                                                          "omit_at_default": True})
    n_registered_agreement_differs: int = field(default=30, metadata={"stage_input": False,
                                                                      "omit_at_default": True})

    # ROADMAP.md sec 41.1 (V3 controls), all opt-in and `omit_at_default`.
    # `negative_control_runs`: `{name: run_dir}` of SOLO runs (extract + sae only) of a
    # `random_init: true` twin on this run's corpus; `sae/control_transfer.py` tests every atlas
    # concept part against the twin's dictionaries as a negative-control destination. The twin
    # lives in its own run directory so it is excluded from every other stage, routing table and
    # scorecard by construction. `input_feature_control`: also test against a destination made of
    # raw context statistics. `window_sensitivity`: coarser windows (multiples of
    # `alignment.window`) at which L1 peak CKA and the atlas-transfer pass rate are recomputed.
    # `tier_breakdown`: write and render the per-data-role breakdown.
    negative_control_runs: Optional[dict] = field(default=None, metadata={"omit_at_default": True})
    input_feature_control: bool = field(default=False, metadata={"omit_at_default": True})
    window_sensitivity: Optional[list] = field(default=None, metadata={"omit_at_default": True})
    tier_breakdown: bool = field(default=False, metadata={"stage_input": False})


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: list = field(default_factory=list)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    l0: L0Config = field(default_factory=L0Config)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    lens: LensConfig = field(default_factory=LensConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    exemplars: ExemplarsConfig = field(default_factory=ExemplarsConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    internals: InternalsConfig = field(default_factory=InternalsConfig)
    confirm: ConfirmConfig = field(default_factory=ConfirmConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    layer_screen: LayerScreenConfig = field(default_factory=LayerScreenConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    concepts: ConceptsConfig = field(default_factory=ConceptsConfig)

    def run_dir(self) -> Path:
        """Directory holding every artifact for this run."""
        return Path(self.run.out_dir) / self.run.name

    def run_shape(self) -> str:
        """Which of the three run shapes this config describes (ROADMAP.md sec 24.3).

        DERIVED from `len(self.models)`, never configured as a mode flag -- a
        hand-set mode is a claim checked nowhere (`CLAUDE.md` sec 11.34) and
        would drift the first time a model was added or removed.

        `solo` (1 model) has no cross-model measurement to make: L1, L2,
        clustering's AMI and L3's fingerprint agreement are dropped with a
        stated reason rather than rendered empty. `pair` (2) is the historical
        shape and must stay bit-for-bit unchanged. `panel` (3+) compares all
        C(n,2) pairs against a designated reference model (the first).
        """
        n = len(self.models)
        if n < 1:
            raise ValueError("config must declare at least one model")
        if n == 1:
            return "solo"
        if n == 2:
            return "pair"
        return "panel"

    def comparison_pair(self) -> tuple:
        """The two models all cross-model analyses compare (first two configured)."""
        if len(self.models) < 2:
            raise ValueError("cross-model analyses require at least two models in config")
        return self.models[0], self.models[1]

    def comparison_pairs(self) -> list:
        """Every model pair a cross-model stage may examine, reference pair first.

        The generalization of `comparison_pair()` to the three run shapes:
        `[]` for solo, exactly `[comparison_pair()]` for pair, and all C(n,2)
        pairs for panel. Pair 0 is ALWAYS `comparison_pair()`, matching the
        enumeration `l0_behavioral` already uses for its all-pairs Holm
        correction (sec 18 F8) -- so a two-model run's behavior is unchanged by
        construction, which is this seam's non-negotiable acceptance criterion
        (sec 2.1: no recorded number moves).

        Returns an empty list rather than raising for `solo`, because the
        caller that wants "the pairs to iterate over" and the caller that
        wants "the designated pair or an error" are different callers;
        `comparison_pair()` is still the one that refuses.
        """
        ms = self.models
        return [(x, y) for i, x in enumerate(ms) for y in ms[i + 1:]]

    def validate(self) -> None:
        """Fail fast on structurally invalid configurations."""
        if self.data.context_len % self.alignment.window != 0:
            raise ValueError("data.context_len must be a multiple of alignment.window")
        nulls = tuple(self.sae.ablation_nulls or ())
        if nulls:
            if len(set(nulls)) != len(nulls):
                raise ValueError(f"sae.ablation_nulls has duplicates: {list(nulls)}")
            if nulls[0] != self.sae.ablation_null:
                raise ValueError(
                    f"sae.ablation_nulls[0] ({nulls[0]!r}) is the primary null and must equal "
                    f"sae.ablation_null ({self.sae.ablation_null!r}); drop one of the two")
            from .sae.response import ABLATION_NULL_MODES
            bad = [m for m in nulls if m not in ABLATION_NULL_MODES]
            if bad:
                raise ValueError(f"sae.ablation_nulls: unknown null mode(s) {bad}; "
                                 f"expected from {list(ABLATION_NULL_MODES)}")
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        # `__`-prefixed names are reserved for L0's no-skill reference
        # pseudo-models (`__naive__`, `__seasonal_naive__`, ROADMAP.md sec 34
        # A3.2) -- a real adapter config using that namespace would collide
        # with them in `l0/metrics.parquet` and enter the per-family
        # reference table as if it were a trivial baseline.
        reserved = [n for n in names if n.startswith("__")]
        if reserved:
            raise ValueError(f"model names may not start with '__' (reserved for "
                             f"L0's no-skill pseudo-models): {reserved}")
        if len(self.models) > 2:
            import logging
            # This used to warn that every cross-model stage but L0 silently
            # compared models[0:2] and ignored the rest. That is no longer
            # true (sec 24.3 sub-item 3): L1, L2 and clustering now measure
            # every pair. The warning stays, with its content replaced rather
            # than deleted, because a panel still carries two real costs a
            # two-model run does not -- a multiplicity burden that grows as
            # C(n,2), and stages whose product remains inherently
            # pair-shaped -- and silently dropping the notice would read as
            # "a panel is just a bigger pair run".
            n_pairs = len(self.models) * (len(self.models) - 1) // 2
            logging.getLogger("tsfm_lens").warning(
                "panel run: %d models, %d pairs. L0, L1, L2 and clustering measure "
                "EVERY pair; L3's fingerprint agreement, exemplar selection and "
                "`confirm` are reported against the designated reference pair "
                "(%s vs %s) or, for exemplars, against the spread across all "
                "models. Multiplicity grows as C(n,2): a bootstrap p is floored at "
                "1/n_boot, so raise stats.n_boot (currently %d) before adding "
                "models -- see the report's multiplicity ledger",
                len(self.models), n_pairs, self.models[0].name, self.models[1].name,
                self.stats.n_boot)


def _build(cls: type, data: dict, section: str | None = None):
    """Recursively construct a dataclass from a plain dict, keeping defaults for absent keys.

    Raises on any key that isn't a declared field of `cls` (ROADMAP.md sec 15
    A21): a misspelled knob (`capture_layer_stide`) used to silently run the
    whole pipeline at the default value while `config_resolved.yaml` read as
    though the setting were in force -- the discrepancy was invisible even in
    hindsight. Naming the closest valid field (`difflib`) targets the actual
    cost of a typo: the right name was nearly typed.
    """
    if data is None:
        return cls()
    section = section or cls.__name__
    if not isinstance(data, dict):
        raise TypeError(f"config section {section!r} must be a mapping, got "
                        f"{type(data).__name__}: {data!r}")
    declared = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - declared
    if unknown:
        parts = []
        for key in sorted(unknown):
            close = difflib.get_close_matches(key, declared, n=1)
            parts.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise TypeError(f"unknown key(s) in config section {section!r}: {', '.join(parts)} "
                        f"-- valid fields are {sorted(declared)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if dataclasses.is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[f.name] = _build(f.type, v, section=f"{section}.{f.name}")
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


_NESTED = {
    "run": RunConfig, "data": DataConfig, "alignment": AlignmentConfig,
    "extraction": ExtractionConfig, "l0": L0Config, "l1": L1Config, "l2": L2Config,
    "clustering": ClusteringConfig, "layer_screen": LayerScreenConfig,
    "budget": BudgetConfig, "frontend": FrontendConfig,
    "sae": SAEConfig, "report": ReportConfig,
    "stats": StatsConfig, "internals": InternalsConfig, "confirm": ConfirmConfig,
    "lens": LensConfig, "attention": AttentionConfig, "exemplars": ExemplarsConfig,
    "corpus": CorpusConfig, "concepts": ConceptsConfig,
}


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config into a validated PipelineConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict) -> PipelineConfig:
    """Construct and validate a PipelineConfig from a plain dict."""
    cfg = PipelineConfig()
    for key, cls in _NESTED.items():
        if key in raw:
            setattr(cfg, key, _build(cls, raw[key], section=key))
    if "l3" in raw:
        l3raw = dict(raw["l3"]) if raw["l3"] is not None else {}
        patch = _build(PatchingConfig, l3raw.pop("patching", None), section="l3.patching")
        cfg.l3 = _build(L3Config, l3raw, section="l3")
        cfg.l3.patching = patch
    if "models" in raw:
        cfg.models = [_build(ModelConfig, m, section=f"models[{i}]")
                     for i, m in enumerate(raw["models"])]
    cfg.sae.ablation_nulls = tuple(cfg.sae.ablation_nulls or ())
    nulls = cfg.sae.ablation_nulls
    if nulls and "ablation_null" not in (raw.get("sae") or {}):
        cfg.sae.ablation_null = nulls[0]
    cfg.validate()
    return cfg


def dump_config(cfg: PipelineConfig, path: Path) -> None:
    """Persist the fully resolved config next to the run artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), encoding="utf-8")
