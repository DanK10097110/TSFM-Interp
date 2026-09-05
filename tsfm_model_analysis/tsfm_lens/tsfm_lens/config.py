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

    Each of the four diagnostics has its own enable flag: a model whose
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
    # own `sae`/roles.json built the same way as this run's. Empty (the
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
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
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
    cfg.validate()
    return cfg


def dump_config(cfg: PipelineConfig, path: Path) -> None:
    """Persist the fully resolved config next to the run artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), encoding="utf-8")
