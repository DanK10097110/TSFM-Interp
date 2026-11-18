# Steering research log

This is the persistent experiment-by-experiment record for the steering work
started 2026-11-16. Read this file before extending the study. Each entry records
the hypothesis, evidence, limitations, and decision; unsuccessful experiments
remain here. New experiments use new directories and never overwrite old runs.

## E00 — CPU and native-head feasibility (2026-11-16)

**Hypothesis.** Cached checkpoints and exact native-head replay make short
steering experiments possible without a working GPU.

**Experiment.** Cached `amazon/chronos-bolt-small`, float32 CPU, two torch/BLAS
threads, eight random contexts of length 128, horizon 32. Captured the input to
`output_patch_embedding`, replayed that same module and the model's own inverse
instance normalization, and compared with the full model. Timed ten batches.
Computed the autograd Jacobian of all nine quantiles over the first 32 steps
with respect to its 512-dimensional native head input. This was an environment
pilot with unseeded random contexts, not a repeatable population estimate.

**Observed.** Head input shape `[8, 1, 512]`; maximum replay discrepancy **0.0**.
Load and first capture **0.15105605125427246 s**; mean batch time
**0.053584837913513185 s**; Jacobian **0.0496058464050293 s**, shape `[288,512]`,
torch numerical rank **288**. Torch `2.12.0`; CUDA unavailable and `nvidia-smi`
cannot communicate with the driver. No packages changed.

**Interpretation.** This sampled Jacobian has no first-order row-rank obstruction
to arbitrary small changes in these output coordinates. This says nothing yet
about singular-value conditioning, sparse SAE combinations, finite changes,
earlier layers, or generalization. It is not an impossibility theorem, a learned
concept claim, or evidence of improved forecast calibration.

**Decision.** Proceed with residual-preserving, measured-response steering. Test
single atoms against sparse mixtures and dense hidden-space controls; test
local directions against directions fitted across separate contexts. Keep full
quantile/time outputs in the loss so cancellation cannot hide in scalar metrics.

## Prospective experiment sequence

1. Short Bolt proof: uncertainty widening/narrowing, phase-specific seasonal
   addition, and a single quantile-channel edit. Train a small fresh SAE using
   separate contexts. Compare individual atoms, mixtures, random dictionaries,
   and dense directions. Measure actual nonlinear responses, norm, and crossings.
2. Expand promising conditions on fresh seeds; freeze choices before evaluation.
   Examine dose response, transfer, sparse feature budget, and iterative repair.
3. Challenge the result at another intervention site or checkpoint. Distinguish
   direct output controllability from a transferable upstream concept.

Output-only edits are a transparent task oracle, not an internal mechanism.
Interval width control is not calibration control. No future values are used
to fit steering directions. Failed methods establish limits only in their
tested intervention class, norm budget, data distribution, and checkpoints.

## E01a — Pilot artifact failure (2026-11-16)

The eight-context Bolt pilot computed all interventions in 12.22 seconds but
failed while serializing a NumPy boolean in the frozen-controller clipping
flag. Directory: `tsfm_lens/runs/steering_e01_bolt_pilot`. Corpus, SAE and frozen
controller were saved; final response measurements were not. **No scientific
conclusion is drawn from this attempt.** Convert the flag to Python bool and
rerun unchanged seeds and budgets into a fresh directory. This is a plumbing
failure, not evidence against steering.

## E01b — Weighted mixtures and SAE-free control work locally (2026-11-16)

**Hypothesis.** Jointly fitted decoder atoms can cancel unwanted effects better
than one feature; a dense SAE-free edit supplies a controllability baseline.

**Protocol/artifacts.** `tsfm_lens/runs/steering_e01b_bolt_pilot/`:
`protocol.json`, `results.json`, `interventions.npz`, `findings.md`, `sae.pt`.
Command: `python run_steering.py --out runs/steering_e01b_bolt_pilot
--train 128 --test 8 --steps 200`. Bolt-small, context128, horizon32, nine native
quantiles; 128 training contexts, eight calibration contexts, eight distinct
test seeds; 512-atom TopK SAE, k16. L2 regularization0.001 on actual hidden edit,
trust radius20% of original state norm. Single,8,32-atom greedy mixtures,
raw coordinates and Gaussian atoms; local dense inverse and frozen dense
controller fitted across calibration Jacobians. All methods use original
state plus edit, never SAE reconstruction replacement. Runtime11.4761 seconds.

**Results.** Mean full-output relative RMS error, at dose0.2:

| Request | Single SAE | SAE32 | Neurons32 | Random32 | Dense local | Frozen dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| widen20% | .52829 | .10929 | .17578 | .20101 | .05402 | .14697 |
| narrow20% | .53114 | .10423 | .17527 | .19636 | .05096 | .14410 |
| seasonal amplitude0.2 | .53648 | .07105 | .32975 | .35542 | .05531 | .32825 |
| only q0.9 +0.2 | .81977 | .46137 | .41662 | .42870 | .16440 | .17665 |

Every full-output relative error counts all orthogonal collateral, not just
the targeted scalar. Dense edits pass error<0.1 on7/8 inputs for each of
widen/narrow/season; SAE32 passes5/8,5/8,6/8. Single atoms pass0/8 on all four
tasks. Native capture replay error0; edited full-model replay max absolute
error1.0728836059570312e-06. SAE centered normalized reconstruction MSE
0.0962458997964859; measured L0 exactly16; only18.164% of atoms appear in eight
test contexts (this is test-set coverage, not the training dead-feature rate).

**Interpretation.** The user's cancellation proposal works as local control in
this pilot. No SAE is needed for the best average control. That does not imply
individual neurons are monosemantic: 32-neuron controls are substantially worse
than dense ones. SAE32 helps seasonal control compared with raw/random32, but
not isolated quantile control. Frozen transfer and sparse control hit the norm
budget on hard tasks; rank alone cannot predict practical selectivity. Eight
contexts and one shallowly trained dictionary do not establish a model-wide
ranking. No uncertainty-calibration claim is made.

**Decision.** Expand dense-vs-sparse controls, test smaller/larger doses and
iterative correction, and compare better trained signed/adaptive-sparsity SAEs
at matched training budgets. Replicate at another model/site before generalizing.

## E02 — Modern dictionaries: better reconstruction is not cleaner steering

**Hypothesis.** Signed polarity (AbsTopK) or adaptive activity (BatchTopK) improves
TSFM steering at a fixed dictionary and training budget.

**Artifacts.** `tsfm_lens/runs/steering_e02_{topk,abstopk,batchtopk}/` each contains
its own protocol, exact results, interventions and findings. All use seed5170000,
512 training,16 calibration,24 new test contexts,600 training steps,512 atoms,k16,
budgets1/16/64 and doses0.05/0.2. The new TopK arm is a matched control with
train-mean decoder-bias initialization; it differs from E01's legacy trainer.
CPU runtimes67.0197,66.3443,67.2616 seconds. No variant was selected on these tests.

**Observed.** Normalized held-out reconstruction MSE: TopK0.05970584973692894,
AbsTopK0.05588775873184204, BatchTopK0.0515652634203434. Held-out mean activity:
16,16,**21.25**. BatchTopK's threshold was calibrated on training only, so its
reconstruction advantage is **not at matched realized test sparsity**.

Mean full-output relative error for64-atom mixtures at dose0.2:

| Task | TopK | AbsTopK | BatchTopK |
| --- | ---: | ---: | ---: |
| widen | .06829 | .07063 | .06992 |
| season | .06595 | .06934 | .06858 |
| only upper quantile | .29460 | .30523 | .30371 |

At dose0.05, all three seasonal64-atom variants pass24/24; none of their
single-atom controls passes any task. TopK seasonal16-atom error is0.06691 at
dose0.05, versus0.12743 at dose0.2. Local dense mean seasonal error0.0561 and
frozen dense0.2029 at dose0.2. Full-model checks pass (exact details per run).

**Interpretation.** These modern variants are useful comparison tools, but this
single-training-seed experiment provides no basis to replace TopK for steering.
Lower reconstruction error did not imply more selective output control. Small
doses are easier; higher-dose errors can involve finite-response nonlinearity
and the norm cap, rather than absence of a controllable direction. Raw64-neuron
seasonal control error0.0999 beats random64 error0.1300 but trails TopK64 error
0.0660; this task does not justify assuming monosemantic native coordinates.

**Decision.** Keep the modern variants opt-in. Prioritize adaptive nonlinear
correction and cross-model/site validation. Do not tune an inference threshold
on these test examples to repair the BatchTopK activity mismatch.

## E03a — TimesFM native-path preflight

Four deterministic contexts (seeds5180000–5180003), CPU float32, context128,
horizon32. Native final-token head state `[4,1280]`; Jacobian `[288,1280]`.
Replay uses TimesFM's own cumulative normalization routine and native point
projection quantiles, matching this repo's adapter. Baseline max discrepancy
9.5367431640625e-07; full-model additive-edit max discrepancy
1.1920928955078125e-06; total2.9218 seconds. The edit was a uniform0.0001 in each
hidden coordinate. This is a numerical path check, not steering efficacy.
The separate long-horizon spread head is outside this interface. Proceed with
the same task protocol on TimesFM; require full-model checks again in that run.

## E03b — Cross-model pilot: TimesFM supports season/width, isolated q0.9 is hard

**Hypothesis.** Native-head controllability and the benefit of combining features
extend beyond Bolt. **Artifacts:** `tsfm_lens/runs/steering_e03_timesfm_pilot/`.
Model TimesFM2.5-200M;128 train,8 calibration,8 test; seed5180000; TopK512,k16,
200 steps; doses0.05/0.2; budgets1/16/64; same radius0.2 and ridge0.001.
Runtime45.38486647605896 seconds. SAE reconstruction NMSE0.2736054062843323:
this pilot dictionary is undercomplete for1280 hidden coordinates and is not
a fair test of all possible SAEs.

Mean full-output relative error, dose0.2:

| Task | Dense | Frozen dense | SAE1 | SAE64 | Neurons64 | Random64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| widen | .0249 | .0923 | .5329 | .0365 | .0379 | .0444 |
| narrow | .0343 | .0711 | .5437 | .0406 | .0483 | .0546 |
| season | .0025 | .0092 | .4654 | .0140 | .0180 | .0396 |
| upper only | .6866 | .3258 | .8034 | .5457 | .4287 | .4358 |

Dense and64-feature variants all pass8/8 for width and season, and0/8 for
upper-only. Capture replay max2.384185791015625e-06, full-model edited replay
max3.337860107421875e-06. Comparisons are within this model; E02 used different
seeds and training budget, so this is not a paired architecture ranking.

**Interpretation/decision.** No-SAE control also works on TimesFM; its frozen
seasonal controller transfers well within this tiny distribution. Single
features remain entangled. A dense *radially clipped ridge proposal* can be
worse than a sparse proposal for q0.9; this is not a controllability paradox.
It motivates an exact linear trust-region solution and actual-response
backtracking, separating a weak optimizer from a genuine restricted-class limit.

## E04 — Nonlinear response feedback repairs most local errors

**Hypothesis.** Re-linearizing after verified improvements beats a one-shot
direction without increasing the total norm budget. **Artifacts:**
`runs/steering_e04_{bolt,timesfm}_pilot/` under `tsfm_lens/`.
Eight fresh paired contexts,seed5190000; four tasks,doses0.05/0.2,
radii0.2/0.5,1/4 iterations; ridge0.001; fixed stopping error0.005.
Runtime34.7515s Bolt and68.9238s TimesFM. All configurations reported.

Mean full-output error at dose0.2,radius0.2,one→four iterations:

| Task | Bolt | TimesFM |
| --- | ---: | ---: |
| widen | .07908 → .03209 | .03555 → .01532 |
| narrow | .06282 → .02899 | .04661 → .01597 |
| season | .07850 → .01769 | .00279 → .00279 |
| only upper | .18656 → .10021 | .58469 → .10430 |

All eight width/season tasks pass for each model after correction. Upper-only
passes6/8 Bolt and5/8 TimesFM. Native replay max9.5367431640625e-07 and
3.337860107421875e-06 respectively. Expanding radius0.2→0.5 does not improve
the four-step solution at this ridge. The ridge-zero *first-order* optimal
error for upper0.2 is0.07492 Bolt and0.09814 TimesFM at radius0.2; it is not
a bound on nonlinear edits. At smaller dose0.05, TimesFM upper error remains
0.13546 even though its local bound is essentially zero: regularization and
optimization, not local rank deficiency, remain plausible explanations.

**Decision.** Test a weaker ridge and six iterations on these discovery seeds,
then freeze before fresh64-context replication and off-grid shift. Do not call
the q0.9 problem impossible. Keep unchanged one-step and frozen-controller
results as controls rather than replacing the earlier evidence.

## E04b — Weaker ridge substantially improves isolated-quantile control

Same discovery seeds5190000–5190007; ridge1e-5,six iterations,dose0.2,
radii0.2/0.5. Artifacts `runs/steering_e04b_{bolt,timesfm}_ridge/`.
Runtime19.3279s/29.1583s. At radius0.2, errors (widen,narrow,season,upper):
Bolt **0.020271292644987497,0.0149475930707283,0.018273646549884965,
0.07081482427844576**; TimesFM **0.0020413997399784814,0.003331683670741415,
0.0007095361016165122,0.04873394225232771**. All pass8/8 except TimesFM
upper7/8. Upper requires mean relative hidden norm0.19966568925939307 Bolt,
0.1382502889822112 TimesFM. Increasing radius is not uniformly better under
finite iteration/backtracking. Full-model checks again pass.

**Important qualification.** TimesFM's native quantiles already cross on some
cells. Steered crossing fractions remain around1%; low steering error does
not imply well-ordered or calibrated quantiles. Carry baseline crossings into
replication rather than sorting outputs and concealing the issue.

**Decision frozen before replication.** Use radius0.2,ridge1e-5,six iterations,
dose0.2; compare one versus six iterations on64 new paired contexts per model,
then64 off-grid-period contexts. This is selected on discovery, not a preregistered
algorithm choice. Replication data are new synthetic seeds, not sealed benchmark
private confirmation. Retain upper-only failures rather than retuning on them.

## E05 — Upstream small-budget challenge fails, and the local bound is informative

Four new contexts,seed5200000; Bolt final encoder output (including REG),
frozen decoder downstream. Two512-dimensional intervention classes: REG only,
or a shared vector across all nine tokens normalized to preserve Frobenius edit
norm. Radius0.05 of whole encoder-state norm,ridge0.001,four iterations,dose0.2.
Artifacts `runs/steering_e05_upstream_pilot/`; runtime67.4160s.

Mean error REG (widen,season,upper):0.28654631519227836,
0.44221874118532584,0.5217495755243022. Shared:
0.1849491886811702,0.23965311972572348,0.46036981221426454. No task passes
error<0.1. First-order best possible errors under the same small norm budget
are0.3493/0.5139/0.5579(REG),0.1965/0.2840/0.4733(shared).
Actual nonlinear iteration can beat those local bounds, as expected.

**Interpretation.** This is a norm/site-specific negative, not a statement that
upstream control is impossible. Head and encoder norms have different meaning.
**Next:** challenge the negative at radius0.2,ridge1e-5,six iterations, adding
independent per-token control. Keep this result as the small-budget boundary.

## E05b — Upstream control is possible, but single-channel isolation remains weak

Same four discovery contexts as E05, radius0.2,ridge1e-5,six iterations,
plus full independent control over all9×512 encoder entries.
Artifacts `runs/steering_e05b_upstream_expanded/`;153.86499047279358 seconds;
all full-model intervention checks exactly0 discrepancy.

| Token scope | Widen error (passes/4) | Season error (passes/4) | Upper error (passes/4) |
| --- | ---: | ---: | ---: |
| REG only | .06250 (3) | .04788 (4) | .21465 (0) |
| Shared | .04944 (4) | .05539 (3) | .25129 (0) |
| Full | .08628 (3) | .04530 (4) | .20697 (0) |

Full-class first-order upper-only lower bound0.19417224975469347; actual error
0.2069713488530237. In contrast to head control, substantially changing one
quantile while protecting all other values is difficult before the decoder in
this budget. Full control does not necessarily yield a better finite-iteration
nonlinear solution, despite its better local feasible set.

**Decision.** Replicate all three scopes on16 fresh contexts,seed5230000,
same radius/ridge/steps/dose. Do not tune on that replication. This longer
experiment tests whether the upstream distinction persists, not impossibility
over arbitrary budgets, layers, or optimization methods.

## E06 — Frozen controller settings replicate on64 fresh contexts per model

Artifacts `runs/steering_e06_{bolt,timesfm}_fresh64/`; paired seed5210000–5210063.
Fixed dose0.2,radius0.2,ridge1e-5; one versus six iterations; no SAE. Runtime
74.9057s/127.1690s. Mean full-output error, one→six steps:

| Task | Bolt | TimesFM |
| --- | ---: | ---: |
| widen | .07804156 → .01906498 | .02587562 → .00157413 |
| narrow | .04730996 → .01106417 | .03146115 → .00235982 |
| season | .06507638 → .01299577 | .00127188 → .00078711 |
| upper only | .24396600 → .07500548 | .61192843 → .03220863 |

Six-step passes:64/64 on every width/season task for both models;
upper-only57/64 Bolt and63/64 TimesFM. Native edited replay errors below2e-6.
TimesFM upper-only one-step crossing fraction0.0054931640625 falls to
0.00067138671875 after correction; all crossings remain explicitly reported.

**Interpretation.** Task-specific adaptive control works without SAE features
on these fresh synthetic contexts. It does not establish monosemantic native
neurons, globally reusable semantic vectors, calibrated uncertainty, or accuracy
improvements. The earlier `neurons_N` label denotes raw **residual coordinates**,
not a separate study of MLP neurons. Single-coordinate entanglement cannot
establish semantic polysemy of every neuron class. Keep that distinction in any
future summary.

**Decision.** Carry the same frozen settings into64 off-grid contexts and
twelve cached real evaluation sources (four per dataset). No task-dependent
retuning or quantile sorting; future values remain unused.

## E07 — Off-grid periods retain width/season control; Bolt upper isolation degrades

Fixed E06 settings;64 new contexts,seed5220000; periods13.7/21.3/47.1 instead
of8/16/32. No changes to the controller, norm budget or success threshold.
Artifacts `runs/steering_e07_{bolt,timesfm}_offgrid64/`;61.9805s/122.1615s.
Mean six-step errors (widen,narrow,season,upper):

- Bolt:0.007532483041356374,0.0045750094467081875,
  0.002716605509821038,0.09301025033495691.
- TimesFM:0.0016518817296379714,0.0019212934351805799,
  0.0008516731746187663,0.03484842871917372.

Both models pass64/64 on each width/season task. Upper-only passes45/64 Bolt,
60/64 TimesFM. Bolt upper mean edit norm0.19967385172392976 nearly saturates
the0.2 budget; its first-order optimum error0.10782944291496487 is consistent
with a difficult restricted control problem, but is not a nonlinear impossibility
claim. TimesFM's actual upper error beats its original local bound0.10770399301091593,
demonstrating why that distinction matters. All native-path checks pass.

**Decision.** Retain explicit per-request failure diagnostics; do not advertise
arbitrary isolated-channel control as reliable. Width/season results justify
shipping the opt-in controller and testing real data with the same settings.

## E08 — Public API and real-context external validity

Twelve cached hourly evaluation sources from `runs/real_background_poc492306`,
four each electricity,M4,pedestrian. Select original evaluation indices, undo
their recorded normalization, use only positions128:256 as context. No future
values, source selection, or hyperparameter tuning. These are **reused** real
windows, and pretraining overlap is unknown. Seasonal prior period24 is supplied
explicitly; season/upper strength0.2 context SD, width±20%. All defaults frozen
from E04b. Artifacts `runs/steering_e08_{bolt,timesfm}_real/`;18.2705s/31.1739s.
Every request runs through `ForecastSteerer.steer`, which verifies full-model
interventions for all returned forecasts, in original data units.

Passes/12 (widen,narrow,season,upper): Bolt12,11,12,8; TimesFM12,12,12,12.
Per-dataset mean errors are in `results.json`; every failed source is retained.
Bolt's narrow pedestrian mean error0.0640823053116202 includes one failure;
Bolt upper M4/electricity/pedestrian means0.10114506084648348,
0.07585582003052742,0.09912001427245763. TimesFM's largest task/dataset mean
is0.020728867781957018 (M4 upper). Native baseline quantile crossings exist in
both models on some real sources. Small new crossing fractions can also occur;
the API reports them without sorting or hiding them behind the success flag.

**Interpretation.** This is useful external-validity evidence for adaptive
control, with twelve source windows, not a real-world generalization guarantee.
No forecast-loss/coverage claim is possible because future outcomes were not
evaluated. Keep task success and quantile ordering as separate diagnostics.

## E10 — Independent arithmetic and complete native replay audit

All **2,048** interventions from E06/E07 (two models,two64-context distributions,
four tasks,one/six iterations) were independently re-evaluated from saved
arrays, then replayed through each full pretrained model. `audit_steering.py`
does not call the producer's response-metric helper. It checks summary counts,
per-context error, hidden norm budgets and removal of hooks. Each run now has
`independent_audit.json`, binding the audit to results/array SHA256 hashes.

All pass. Maximum normalized full-model discrepancies:
E06 Bolt2.1457672119140625e-06,TimesFM2.384185791015625e-06;
E07 Bolt2.1457672119140625e-06,TimesFM2.0265579223632812e-06.
Arithmetic differences≤5.551115123125783e-17. Float32 serialization permits
tiny budget overshoots≤9.5367431640625e-07 in absolute hidden norm, below the
predeclared audit tolerance2e-5. No leaked intervention hooks.

**Decision.** The reported late-head effects are actual internal interventions,
not artifacts of suffix replay or summary arithmetic. This verifies execution;
it does not enlarge the semantic/calibration claims. Portable replication
arrays, audits, figures, source statistics, library/checkpoint provenance and
a code snapshot are retained in `examples/steering/`.

## E09 — Longer upstream replication confirms the restricted single-channel failure

Sixteen fresh contexts,seed5230000–5230015; frozen E05b settings. Runtime
**587.2643084526062 seconds** (about9.8 minutes), two CPU threads, no GPU.
Artifacts `runs/steering_e09_upstream_fresh16/`; all three token scopes tested
on every context. Native full-model checks on the first context's nine
interventions give exact0 discrepancy; all other forecasts run through the
same frozen decoder/head suffix. No exclusions or retuning.

| Scope | Widen error (passes/16) | Season error (passes/16) | Upper error (passes/16) |
| --- | ---: | ---: | ---: |
| REG only | .06293648864400642 (15) | .03323933696388885 (15) | .2570767497950713 (0) |
| Shared | .04446914917032621 (14) | .026455233029194745 (16) | .242367760175313 (0) |
| Full | .06313741179374613 (13) | .03146859625393418 (16) | .27089506942840846 (0) |

The exact original-Jacobian upper-only lower bounds average
0.4195139834384104,0.31967113879732045,0.21209865207669704. The full class
improves the local feasible set, but its finite-step nonlinear optimizer does
not beat the smaller classes. This distinguishes local reachability from
successful optimization. Width/season success before the decoder shows the
positive result is not restricted to directly editing the last projection.

**Stop/continue decision.** Close this finite-budget upstream branch as a
replicated negative for isolated-q0.9 control, with a positive width/season
result. Do not generalize to all budgets, layers, optimizers or TSFMs. Ship the
late-head controller with per-request diagnostics and keep the upstream driver
as a research instrument. A future study should compare alternative upstream
optimization and natural-state constraints on prospectively chosen new data.

## Session handoff and verification

Implemented opt-in `ForecastSteerer`, sparse weighted cancellation, exact local
trust-region solver, nonlinear acceptance, Bolt/TimesFM native interfaces,
AbsTopK/BatchTopK plus matched TopK and explicit checkpoint loading. The API also
accepts an explicit seasonal phase when the context has no harmonic; that branch
has a planted-answer test, not a population experiment.

Tests cover planted cancellation, rank obstruction, dictionary rescaling,
trust-region optimality, nonlinear budget/acceptance, original data units,
full-model path mismatch, signed gradients, batch-independent inference and
checkpoint roundtrips. Existing SAE and adapter/differentiable-forecast tests
also pass. Actual checkpoints from E01/E02 reproduce their stored reconstruction
scores after loading. See `examples/steering/verification.json` for the final
verification record. No dependency installation, model weight changes or
publication/deployment were performed.

Next-session priorities: (1) source-disjoint, multi-seed tests before replacing
TopK; (2) context-transferable small control bases versus per-input Jacobians;
(3) upstream control under natural-state constraints; (4) forecasting/coverage
utility only when a prior and evaluation outcome justify that distinct claim.
Do not rerun broad grids merely to erase the retained upper-only failures.
