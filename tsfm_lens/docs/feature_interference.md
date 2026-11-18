# Controlled feature-interference audit

This standalone audit asks whether changing one known periodic input feature
changes the model forecast's *untouched* frequency component, and whether model
comparisons change between pure and mixed compositions. It complements
[window robustness](window_robustness.md), which tests geometric measurement
choices. Neither workflow establishes hidden feature selectivity or internal
causal mediation; linear representation probes require separate validation.

Use the environment in which `tsfm_lens` is installed and run from `tsfm_lens/`.
The shipped [mock CPU example](../configs/examples/feature_interference/mock_cpu.yaml)
uses four generating seeds and two registered mock models. It prospectively
selects exact two-sided sign probability as the primary endpoint; four seeds
make that comparison descriptive. The nine permutations apply only to secondary
historical mean comparisons. The same YAML contains commented native
model replacements and prospective sample-size suggestions. Copy it before
configuring a native study; the standalone example is not a pipeline preset.
For new studies the example enables `design.native_background: true`, so a ninth
constant-background input is forecast directly. The default remains false for
compatibility with existing eight-cell artifacts.

Use a fresh disjoint seed range for a replication and record it before inference.
This audit generates synthetic data and never reads a sealed private corpus.
The first version supports two distinct integer periods greater than two, both
tiling the context and horizon. It checks the five-column harmonic design for
full rank, conditioning, and orthogonality; aliased frequencies are refused.

Prepare immutable inputs and protocol before running any model:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python run_feature_interference.py \
  --config configs/examples/feature_interference/mock_cpu.yaml \
  --prepare-only --out runs/feature_interference_sign_prepared
```

Then forecast into a separate new output directory:

```bash
python run_feature_interference.py \
  --prepared runs/feature_interference_sign_prepared \
  --device cpu --out runs/feature_interference_sign_forecasts
```

CPU is the default device. An explicit CUDA device uses one model at a time,
with a 6 GiB allocator cap; use only an assigned visible GPU. The CLI internally
caps torch/BLAS to one CPU thread and forecast batches to at most eight. It uses
the existing adapter registry and explicit `ModelConfig` entries. Cached models
only: network downloads are disabled. Native forecast repeatability must pass;
sampled forecasts require a separate draw-matched protocol and are refused when
the native repeat differs. The library leaves resource control with its caller.
The CLI also checks an estimated corpus/forecast/bootstrap working-array budget
of 2048 MiB (`--memory-limit-mb`); this is an estimate, not an operating-system
memory limit or the model allocator limit.

Each seed supplies eight matched pure/mixed inputs: four mixed phase-factorial
cells and two cells for each pure component. With `native_background: true`,
the ninth input contains only the same constant offset, with exactly zero true
component futures. Amplitudes, offset and the mixed-energy denominator remain
matched across all cells. Target φ refers to the first configured period, ψ to the second.
Both intervention directions and untouched-phase cells are averaged within the
seed; seeds, not rows or horizon positions, are resampled.

Analyze existing artifacts without loading a model:

```bash
python run_feature_interference.py \
  --config configs/examples/feature_interference/mock_cpu.yaml \
  --corpus runs/feature_interference_sign_prepared/corpus.npz \
  --metadata runs/feature_interference_sign_prepared/metadata.json \
  --predictions mock_patch=runs/feature_interference_sign_forecasts/mock_patch_predictions.npz \
  --predictions mock_wave=runs/feature_interference_sign_forecasts/mock_wave_predictions.npz \
  --out runs/feature_interference_sign_saved_analysis
```

The corpus NPZ requires `context [N,C]`, `component_future [N,2,H]`, and
`row_ids [N]`. Each forecast NPZ requires `prediction [N,H]`, its own matching
`row_ids [N]`, and a Unicode scalar `corpus_sha256` equal to the SHA256 of the
**exact supplied corpus NPZ bytes**. Native inference writes this binding before
analysis. A rewritten or repacked corpus needs a newly established binding;
retain the exact prepared corpus file alongside its forecasts.

IDs are explicit strings
`generator_seed:condition:cell_phi:cell_psi`. Metadata uses the prepared schema:
`seed`, `generator_seed`, `condition` (`mixed`, `pure_phi`, `pure_psi`, plus
`background` only when enabled),
`cell_phi`, `cell_psi`, `phi`, `psi`, `a`, `b`, `offset`, and
`scale=sqrt((a*a+b*b)/2)`. Shapes alone cannot establish identity: missing,
duplicated or reordered artifact IDs are refused. There is no automatic legacy
row-order fallback.
IDs name factorial cells and intentionally recur across context lengths, periods,
and amplitude ranges. The separate corpus hash prevents accidentally attaching
same-shaped forecasts from another physical input corpus. Hashes preserve a
producer's declared binding; they are not cryptographic proof that model inference
was performed. Array-API callers must establish forecast/input provenance
themselves.

New native prediction artifacts also record `model_name`, a digest of the
effective model configuration, and a checkpoint-identifier digest. The latter
hashes the identifier string, not checkpoint weight bytes. These declarations
are included in the analysis provenance.

`report.html` is self-contained and works offline. It includes the complete
protocol and artifact hashes; all model-pair effect intervals and multiplicity
corrections; pointwise component-response intervals; original-unit and scaled
error channels; and fixed waveform examples from all four mixed cells of the
first generating seed. `results.json` and `seed_contrasts.npz` support further
review. The source artifacts and prepared directory remain unchanged.
Saved-forecast analysis does not by itself prove the supplied configuration was
registered before forecasting; its provenance states that timing limitation.

The portable [native example report](../../examples/feature_interference/report.html)
ships the fresh 64-seed native audit's exact corpus and forecast bytes, metadata,
original protocol and producer hashes. No checkpoints or activations are included.
Its original registered endpoint was **mean**, with the stated symmetry
assumption. The later report retains that endpoint and labels the added exact
sign comparisons secondary; it is not prospective sign-primary confirmation.
Producer inference and later analysis source hashes are recorded separately.
Reproduce that saved analysis on CPU, without loading model weights:

```bash
python run_feature_interference.py \
  --config ../examples/feature_interference/config.yaml \
  --corpus ../examples/feature_interference/corpus.npz \
  --metadata ../examples/feature_interference/metadata.json \
  --predictions bolt=../examples/feature_interference/bolt_predictions.npz \
  --predictions timesfm=../examples/feature_interference/timesfm_predictions.npz \
  --producer-provenance ../examples/feature_interference/producer_analysis_provenance.json \
  --out runs/feature_interference_portable_replay
```

The supplied producer provenance verifies its neighboring original `protocol.json`
and all supplied input/forecast hashes. A changed endpoint or protocol is visibly
labeled an alternate-plan reanalysis rather than prospective confirmation.

Signed contrasts are **later-listed model minus earlier-listed model** in YAML
order. Every model pair has four fixed contrasts: untouched phase change after
changing φ, untouched phase change after changing ψ, mixed scaled MSE, and the
mixed-minus-equal-weight-pure model gap. All pairs share one complete Holm
family for each endpoint type. Untestable endpoints retain p=1. Declare
`inference.primary_estimand` before preparation:

- `sign_probability`: the new example's choice. It tests
  `q=P(D>0 | D!=0, jointly identifiable)` against 0.5 using an exact **two-sided**
  binomial test. Its Clopper-Pearson confidence interval is pointwise, not
  multiplicity-adjusted. Positive, negative, tied, and excluded seeds remain
  explicit. It does not establish expected-MSE superiority.
- `mean`: the core default for compatibility with historical protocols. Its
  mean intervals are approximate pointwise bootstrap intervals conditional on
  fixed native forecasts. Its sign-flip p values require symmetry or exchangeable
  paired labels; they are not distribution-free tests of mean zero under
  arbitrary skew.

Mean and exact sign comparisons receive separate complete Holm corrections.
Only the predeclared family controls primary statistical status; there is no
OR rule or secondary-endpoint rescue. The report retains the actual signed mean
and median alongside q, and flags when mean and majority sign point in opposite
directions. Such disagreement is useful information, not an error or a generic
model-winner label. The old one-sided `sign_sensitivity` remains a secondary
compatibility diagnostic.

For array/JSON consumers, `selected_primary` is the canonical chosen endpoint.
`mean_comparisons` and `sign_comparisons` expose both families. The historical
`primary` field remains the mean family, regardless of selection, to preserve
its previous numerical meaning. Changing endpoints after inspecting forecasts
is an alternate-estimand reanalysis, not prospective primary confirmation.
Specifically, `model_by_composition` is the mixed model gap minus one half of
the sum of the pure φ and pure ψ model gaps. It is not a difference between
native-minus-component-sum gains. The tool reports statistical effects without
a minimum practical threshold or discovery gate; significance alone is not
evidence of a discovery or practical relevance.

`inference_resolution` describes the selected endpoint. Exact sign resolution
depends on the number of nonzero seeds: the minimum two-sided p value is
`min(1, 2**(1-n_nonzero))`, independent of the permutation budget. The report
retains each comparison's count and floor, plus the family bound. Four mock
seeds cannot reject even before multiplicity correction.
`mean_inference_resolution` separately retains the Monte Carlo plus-one floor
`1/(n_permutations+1)` and complete-family bound
`min(1, family_size/(n_permutations+1))`. For example, three models give 12 mean
claims; nine permutations cannot yield a significant Holm result. Runs whose
selected family cannot reject at alpha 0.05 are visibly labeled descriptive.

Every composition uses the **same mixed-energy denominator** `(a²+b²)/2`,
including pure inputs. Phase is undefined below the configured relative true
amplitude floor; coefficient changes remain measurable. Any unidentifiable
matched cell excludes that entire seed bundle from the corresponding paired
phase contrast. Model-dependent exclusions can change the complete-case
estimand; undefined phase is not evidence of absent coupling.
Per-model phase intervals use each model's identifiable seeds. Subtracting those
marginal means is not a paired comparison: each model-pair phase effect uses
only jointly identifiable bundles. Results retain valid/excluded seed IDs and
the two model means restricted to that paired subset. Statistical helpers require
one-dimensional seed vectors, reject infinities, and allow explicit NaN only as
undefined phase, rather than flattening repeated rows into extra seeds.

Offset and out-of-basis
forecast errors are retained in the exact orthogonal MSE decomposition. Its
signed algebraic amplitude×phase term is not the cross-frequency input
interference statistic.

With native background enabled, the component sum is
`f(pure_phi) + f(pure_psi) - f(background)`, using the actual model forecast for
the ninth cell. The report includes background forecast error and signed bias as
descriptive diagnostics. This subtraction preserves affine forecast bias rather
than doubling it; it does not substitute the known offset for a model response.
The main four comparisons still use the same mixed and pure conditions, so
their definitions and multiplicity family do not change. Noise support is not
part of this version.

With the default `native_background: false`, the original oracle forecast sum
`f(pure_phi) + f(pure_psi) - KNOWN_offset` remains supported. This is **not**
`f(offset)`: it doubles a constant forecast bias and can make squared error four
times worse. Both branches require controlled pure-component input decomposition
and are composition diagnostics, not generic deployable forecasting upgrades.
A positive model-by-composition interaction alone does not show a model ranking
flip; inspect the separately reported mixed and pure gap intervals. Input
coupling can arise from additive frequency mixing and does not demonstrate
hidden nonlinear feature interaction, unique features, or monosemanticity.
