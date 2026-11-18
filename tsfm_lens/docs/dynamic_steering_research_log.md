# Dynamic steering research log

Continuation requested 2026-11-16. The earlier native-head work and its limits
are in `steering_research_log.md`. Every experiment below will retain its setup,
observations, negative results and next decision.

## Prospective design

Provide the same request API for a new forecast callable, with optional PyTorch
module access. Search internal module inputs automatically using measured
finite-difference responses, so no differentiable forecast implementation is
required. If internals are unavailable, permit explicitly labeled input edits.
Do not silently edit the returned output tensor. Support train/validation
selection, profile save/load and per-input response refinement.

Nineteen behavioral requests cover level, trend, detrending, amplitude,
seasonality, deseasonalizing, smoothing, roughness, uncertainty, asymmetry,
individual quantile, anchor, mean reversion, bounds, delay, context ignoring,
context emphasis, context focus and recency. Forecast windows protect the
unselected short/long/explicit steps. Context focus is operationally defined
by an input counterfactual, not by an unverified claim about attention weights.

Hypotheses: (D1) bounded automatic site search finds useful control without
architecture-specific names; (D2) finite differences work through ordinary
no-grad prediction APIs; (D3) saved response profiles transfer to fresh contexts,
with local refinement improving poor fits; (D4) protected horizon ranges remain
stable; (D5) ignoring a window measurably reduces sensitivity to that window.
Start with planted models and small live-model pilots, then freeze successful
settings before a larger test. Failures of the bounded search do not establish
global impossibility or support for every architecture/task combination.

## D00 — planted-model integration checks

The first run passed 27/28 checks. The failed positive test used a random
square readout: the controller exhausted its 0.8 relative activation radius,
leaving 15–16% error on two new contexts. This was a reachable-span versus
bounded-norm distinction, not evidence that automatic site discovery failed:
the discovered final linear input had the lowest validation error. Preserve
bounded failure reporting; make the positive fixture's head orthogonal so
the test actually plants a well-conditioned reachable intervention. Constant
models already test honest inability. Context counterfactual independence,
all 19 target masks, reused-module invocation targeting, stochastic rejection,
profile persistence and changed-weight rejection passed in the initial run.

Revised positive fixture: 28/28 tests pass (4.63s). The change plants reachability;
it does not relax the success threshold or remove the norm bound.

## D01 — first architecture-independent live Bolt pilot

Artifact: `runs/dynamic_d01_bolt_pilot` (protocol, candidate scores, learned
profiles, raw forecasts, replay checks, findings). CPU, 16.1s total; four
candidates, 128 orthonormal directions, one calibration and one separate
validation context, five requests on one fresh context, horizon 32, radius .2.
Four of five requests achieved <=10% relative error: short level 5.55%, long
trend 4.38%, seasonality 4.38%, uncertainty 4.39%. All selected a discovered
head-input site without engine-level layer names. Exact full-predictor replay.
Recent-context half-suppression selected an input edit and failed (54.09%
error). This suggests small activation/input budgets need not reproduce large
context counterfactuals; next test other models and stronger context policies
separately. One test context is a feasibility check, not reliability evidence.

## D02 — TimesFM pilot rejects execution-order-only search

Artifact: `runs/dynamic_d02_timesfm_pilot`, 74.5s, 0/5 successes. Two late
quantile-branch sites had exactly zero measured response in the registered
forecast. They executed but their outputs were not consumed by that forecast
path. Four-candidate coverage missed the useful point/quantile readout branch.
Input control improved seasonality (23.2% error) and context suppression
(17.3%) but failed the 10% criterion. This is a search failure, not a claim of
model uncontrollability. Next change: cheaply screen up to 64 observed sites
with signed interventions; discard measured zero-response branches before
choosing expensive response-map candidates. No layer-name exceptions added.

## D03 — Timer integration boundary before steering

Artifact: `runs/dynamic_d03_timer_pilot`, failed before any steering. Existing
generic adapter block discovery requires >=2 tokens; context 128 yielded too
few patch tokens. Its own loader rejected the model. Retry with context 256
and the same generic steering engine. This is a predictor-registration limit,
not evidence that the steering algorithm cannot control Timer.

## D04 — unseen point-only Timer via generic registration

Artifact: `runs/dynamic_d04_timer_screened`, context 256, 20.7s, 3/4 successes.
Short level, long trend and seasonality each had 0.25% error; automatically
selected `lm_heads.0`. Context suppression failed at 27.94% error. Exact
full-predictor replay. No uncertainty was fabricated for this point-only model.
The longer context resolves D03's loader boundary without layer overrides.

## D05 — output-effect screening repairs TimesFM search

Artifact: `runs/dynamic_d05_timesfm_screened`, 51.2s, 4/5 successes. All four
forecast-shape requests had 5.11% error after automatic selection of an active
head-input branch. Context suppression failed at 44.97% error. This supports
measuring dependency on the requested output rather than relying on execution
order. The 64-site screen is still bounded, not an exhaustive causal graph.

Next: for full-horizon context requests, expose the counterfactual as an
explicit input relevance policy (default auto when input edits are enabled).
This guarantees the stated input gating semantics, not hidden-state control.
Keep context_mode=discover for experiments on reproducing those counterfactuals
internally, and always use discovery for partial forecast windows. Exact
ignore=1 sensitivity is tested separately. Also add automatically discovered
joint controls across repeated module invocations for autoregressive horizons.

## D06 — long-horizon extension exposes expanded-batch boundary

Artifact: `runs/dynamic_d06_bolt_long_pilot`, H=96, rank384, six candidates,
one calibration/validation/test context, 443.6s. Three of five requests met the
threshold, including the direct context policy (which is not learned control).
Seasonality 8.95%, uncertainty 6.29%; short level 14.07% and long trend 24.79%
failed. Exact replay. The run includes expensive finite differences through
Bolt's nine-branch quantile rollout, so it is a longer extension experiment.

Inspection of the installed predictor confirms later rollout expands batch B
to B×9. The initial trace admitted only first-dimension 1, hence omitted these
later internal sites. Revised discovery records observed batch multipliers up
to a configurable cap, broadcasts within each sample's branches, and validates
batched probes against singleton predictions. Joint sites retain invocation-
specific multipliers. A planted reused-readout test with 1→3 expansion passes;
next run checks the real quantile rollout. This is a predictor-geometry repair,
not a hardcoded Bolt readout rule. The package emits its own warning for H>64;
these tests evaluate control, not long-horizon accuracy.

## D09 — explicit context relevance, Bolt sensitivity audit

Artifact: `runs/dynamic_d09_bolt_context`, eight new contexts, six policies.
All 32 ignore/focus perturbation cases had exactly zero controlled forecast
sensitivity while retaining real model inference. All replays were exact.
Emphasis produced greater sensitivity in 12/16 perturbations, not all 16:
amplifying input deviations does not guarantee monotonic nonlinear forecast
sensitivity. Keep that distinction explicit in API docs and researcher claims.
These input-policy results are recorded separately from learned interventions.

## D07 — frozen larger Bolt suite, fresh eight contexts

Artifact: `runs/dynamic_d07_bolt_full`, H64, rank256, six full candidates,
two calibration and two validation rows, independent seeds81/82/83; 269.8s.
There are 224 cases: **141/168 learned/additive interventions** meet the 10%
criterion, plus 56/56 explicit input policies (not learned successes).
All full-predictor replays are exact. Basic level/trend, harmonic, smoothing,
roughness, full-horizon width/asymmetry/single-quantile and short/long level
requests pass all eight. Anchor passes6/8 and mean reversion7/8. Large absolute
bounds and one-step delay pass0/8 each; selective short/long width sometimes
fails despite full-horizon width success. The desired zero movement outside a
window consumes control capacity. Retain all failures; do not silently expand
the norm budget. Next audit on real contexts and inspect whether failures are
span-limited or norm-limited using the saved linear-solve diagnostics.

## D10 — context relevance audit on TimesFM

Artifact: `runs/dynamic_d10_timesfm_context`, same eight fresh contexts/policies
as D09. All 32 ignore/focus perturbation cases again have exact zero sensitivity
and exact replay. Emphasis increases sensitivity in15/16 cases, not universally.
Together D09/D10 support the operational ignore/focus guarantee and reject the
stronger claim that amplified inputs must monotonically amplify forecast response.

## D12 — larger Timer suite at H96, random subspaces

Artifact: `runs/dynamic_d12_timer_full`, context256, H96, rank128, four
candidates, calibration/validation/test seeds81/82/83, 176.9s. **63/112 learned
interventions** pass, plus56/56 direct input policies. Trend, detrend, smoothing,
roughness and short-level pass8/8; amplitude/deseasonalize and large anchor/
bounds/delay fail. Exact native replay. This is a useful warning against treating
full measured output rank as sufficient for practical control: a random subspace
can waste norm on directions invisible to the selected linear module.

Next hypothesis: when a discovered Linear module has output width below the
rank budget, use the exact row space of its weight. Null-space input components
cannot affect that module output, so removing them loses no possible effect
at that input site and minimizes wasted norm. Implement generically from the
module type/shape, without readout names or forecast-gradient access. First
test two matched contexts, then a new-seed replication if promising.

## D13 — Chronos2 is registered, but small-budget success is mixed

Artifact: `runs/dynamic_d13_chronos2_pilot`, H32, four candidates, rank128,
74.1s. Long trend passes (0.66% error); short level69.98%, seasonality53.44%,
uncertainty24.28% fail. The direct context policy succeeds by construction and
is separate from the 1/4 learned successes. All forecasts replay exactly.
Same engine and existing predictor wrapper, no layer override. This establishes
registration/intervention portability, not reliable control on every model.

## D14 — exact Linear row-space proof of concept

Artifact: `runs/dynamic_d14_timer_rowspace_poc`, matched first two D12 test
contexts and same rank cap/radius/training/validation settings, 41.4s.
**26/28 learned cases pass**, versus16/28 on these same contexts in D12;
14/14 direct policies remain separate. Amplitude, deseasonalization, anchor,
mean reversion and delay improve markedly. Only large bounds still fail.
Full predictor replay is exact. The discovered head uses96 row-space directions
instead of128 random directions: more effective control with fewer probes.
This is a paired exploratory comparison, not fresh confirmation. Proceed to
eight new test contexts with this basis rule frozen, and retain the matched
comparison and all negative results. No SAE or model-specific layer rule used.

## D11 — repeated-invocation discovery controls the long rollout

Artifact: `runs/dynamic_d11_bolt_joint_rollout`, H96, rank384, four candidates,
two fresh contexts (seed93), one local refinement, 395.5s. All8/8 learned cases
pass (short level, long trend, full seasonality and uncertainty), plus2/2 direct
context policies. Exact native replay. New batch-multiplier and joint-invocation
candidates were available without architecture names. This repairs D06's
coverage limitation; two contexts are still a small feasibility sample and the
native long-horizon quality warning remains relevant to forecast utility.

## D15/D17 — saved-profile migration failure before real-data evaluation

The first real-context attempts failed before forecasts: old request keys
serialized period24 as an integer, while numeric normalization now uses24.0.
Strict JSON-string comparison treated equivalent requests as different.
Repair: compare parsed request semantics, then canonicalize the restored key.
Keep failed run folders and retry with suffix b. This integration check caught
a persistence compatibility issue that same-version round-trip tests missed.

## D08 — frozen larger TimesFM suite, fresh eight contexts

Artifact: `runs/dynamic_d08_timesfm_full`, H64/rank256, six candidates, seeds
81/82/83, 716.9s. **156/168 learned/additive cases pass**, plus56 direct policies.
All moderate shape requests and all four selective short/long widening/narrowing
variants pass8/8. Large bounds fail8/8, anchor fails3/8 and mean reversion1/8.
All replays exact. Level-all selected additive input control, explicitly labeled;
other learned tasks select internal sites. Success remains conditional on the
requested target and radius. This larger run supports useful control beyond
the single-context pilots, not universality or improved forecasting accuracy.

## D16 — fresh Timer confirmation of exact row-space control

Artifact: `runs/dynamic_d16_timer_rowspace_fresh`, unchanged D14 training,
validation, radius and basis rule; eight new contexts seed103, 65.4s.
**103/112 learned cases pass**, plus56 explicit policies. All moderate tasks,
including amplitude, deseasonalize and delay, pass8/8. Large bounds fail8/8 and
anchor fails1/8. Exact full-predictor replay. This confirms the basis improvement
on new contexts; it does not make the larger-target norm constraint disappear.

## D15b — saved Bolt profiles transfer to real public-source contexts

Artifact: `runs/dynamic_d15b_bolt_real`, 12 evaluation sources from electricity,
M4-hourly and pedestrian counts; original units, no future values and no refit.
**60/72 learned cases pass**, plus24 explicit input policies. Seasonality and
short/long level pass12/12, full width11/12, short width6/12, long width7/12.
All forecasts replay exactly. These sources were used in earlier repository
research and may overlap pretraining: this is external-validity evidence, not
a sealed benchmark. Per-input response refinement remains active.

## D18 — partial-horizon context policies remain difficult at small rank

Artifact: `runs/dynamic_d18_bolt_context_windows`, H32, rank128, radius.2,
two calibration/validation and four new contexts (seed113), 113.5s. Only5/28
learned cases meet the criterion: mild recent-ignore passes2/4, mild older-
emphasis3/4; larger ignore/emphasis and focus fail. Targets protect the opposite
forecast half, so direct input gating cannot satisfy them. Exact replay.
Test full512-coordinate control next on the same contexts to isolate subspace
restriction from the norm constraint; this follow-up is a paired analysis.

## API integration checks during research

The download-free custom-model example successfully discovers a site, protects
the early forecast half, saves/reloads its learned profile and exactly replays
actual predictions. The focused suite passed56 checks before the final response-
reuse addition. Tests include unseen no-grad models, all19 request masks, exact
context invariance, inactive branches, expanded repeated invocations, root-module
row-space control, stochastic rejection, batch dependence, changed weights,
legacy key migration, corruption checks and graceful refinement budget limits.

## D17b — saved TimesFM profiles on the same real-source audit

Artifact: `runs/dynamic_d17b_timesfm_real`, 481.4s. **68/72 learned cases pass**,
plus24 direct policies. Seasonality, full-width and short/long level each pass
12/12; short and long selective width each pass10/12. Exact replay. Same reused
source caveat and no-refit protocol as D15b. This supports practical profile
portability, while explicitly retaining adaptive per-input response refinement.

## D19 — full coordinate span improves partial context control, but is insufficient

Artifact: `runs/dynamic_d19_bolt_context_fullspan`, same four contexts and
requests as D18, rank cap512, 238.6s. Success improves from5/28 to12/28:
mild recent-ignore4/4, mild older-emphasis4/4, stronger emphasis1/4 and partial
focus with outside_weight.8 achieves3/4. Strong ignore/full focus remain0/4.
All replays exact. Removing the random-subspace restriction helps, but preserving
one forecast half while reproducing a large context counterfactual in the other
remains difficult under radius.2. This is a bounded-site/budget negative, not a
model-wide impossibility theorem. Do not silently replace this requested control
with full-horizon input gating or splice target forecasts into the output.

## D20 — response reuse across new requests, doses and periods

Artifact: `runs/dynamic_d20_response_transfer`, 40.4s, four new contexts seed123.
Only the original level-all profile was retained. New long-width(+.15), period18
seasonality(+.15), and short-level(-.1) requests made **zero fresh site searches**.
Width and negative level pass8/8 combined; off-grid seasonality fails4/4
(32–40% error), for8/12 overall. All replays exact. Reusing a response is useful
and cheap to initialize, but does not eliminate target-dependent norm limits.
Selection is labeled online transfer, with no new held-out validation claim.

## Closure and later-session handoff

The final focused regression suite passes57 tests. All completed steering runs
have protocol, per-case results and findings files; failures D03/D15/D17 retain
their failure findings. Portable copies, source hashes and scientific figures
are in `examples/dynamic_steering/`. Model weights and larger learned profiles
remain under local `runs/.../profiles`; the API can train/save replacements.

Use `docs/dynamic_steering.md` and the download-free custom-model example to start
new integrations. D07/D08/D16 are the larger synthetic suites, D15b/D17b the
real-source transfer checks, D09/D10 the strict context-invariance audits,
D11 the expanded-batch rollout check, and D18/D19 the selective-context limit.
All successful forecasts are actual native inferences with input/internal edits.
No universal-model guarantee, semantic-neuron claim, forecast-accuracy gain or
uncertainty-calibration claim follows from these experiments. Useful next work
would be source-disjoint utility/coverage evaluation, upstream-site restrictions,
multivariate contracts, and stronger selective-context controllers with explicit
cost/perturbation budgets. Do not rerun broad expensive sweeps without a new
hypothesis; the small proofs and negative results above should guide the choice.

Final independent artifact audit: 115 file hashes checked, **1,125 steering
cases** and **96 sensitivity pairs** recomputed from saved arrays. Maximum
published-error discrepancy is 1.11e-16; recorded native replay discrepancy is 0.
The audit verifies errors, success flags and crossing rates separately from the
controller, and preserves its machine-readable output in
`examples/dynamic_steering/artifact_audit.json`. These counts include exploratory
and repeated-context experiments and must not be treated as independent samples
for an aggregate success-rate claim.
