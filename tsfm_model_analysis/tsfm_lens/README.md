# tsfm-lens

Layered, architecture-agnostic comparison of time-series foundation model
internals. Built to answer, for a pair like **TimesFM** (decoder-only,
32-step patch tokens) and **Chronos** (T5 encoder-decoder, per-timestep
quantized tokens): *where do these models compute similar things, where do
they differ, on which kinds of data — and which of those findings survive
statistical scrutiny and a held-out benchmark?*

It consumes labeled benchmarks produced by the companion `tsfm_benchmark`
package (sealed corpora with per-sample provenance and a public/private
split) and produces one self-contained interactive HTML report.

## The layered method

No single measurement supports the claim "these models share structure."
Each level asks a stronger question and exists because of the previous
level's limitation:

| Level | Question | Method | Limitation it inherits |
|---|---|---|---|
| L0 | Who is better, where? | MASE / sMAPE / pinball per family, paired bootstrap tests, Holm-corrected | Behavioral only |
| Screen | Which of *this* model's own layers are worth further analysis? | Residual-trajectory work+bend geometry (default), or coverage/factor-emergence | Cheap proxy for interestingness, not interestingness itself |
| Profile | What is in each model? | Effective dimensionality, family-probe decodability, CKA-to-input per layer | Per-model, descriptive |
| Lens | Where in depth does the forecast form? | Skip lens (layer-l states patched into the final block, decoded with the model's own head) + tuned ridge readout; crystallization depth | Depth-resolved, not component-resolved |
| L1 | Do representations share geometry? | Linear CKA (global + per-family) with series-bootstrap CIs, RSA | Correlational |
| L2 | Is the shared geometry linearly translatable *beyond the input*? | Ridge stitching vs an **input-feature baseline**; CI on the gain | Still not causal |
| L3 | Where is structure causally carried? | Corruption sensitivity fingerprints + within-model activation patching (per layer **and per window**) | Within-model causality, compared across models |
| Attention | Which heads look where, and which matter? | Lag-profile head taxonomy, periodicity heads (induction-head analog), head/MLP mean-ablation ΔMASE, first-step decoder cross-attention | Pattern support varies by architecture |
| L4 | How does each model organize the data? | Activation clustering with approximate labels, AMI across models | Descriptive |
| Exemplars | What does the difference look like? | Per-family case studies: forecasts, per-series lens curves, window attention maps | Illustrative, not statistical |
| Confirm | Which dev findings are real? | One-shot re-test of dev hypotheses on a sealed **private** corpus | The gold standard |

Three design points are load-bearing:

- **Time alignment.** Token states are pooled onto shared fixed-width time
  windows (default 32 steps) via overlap-weighted pooling matrices built
  from each adapter's declared `token_time_spans`. Everything cross-model
  operates on `[series, window, dim]` tensors, so patch-based and
  per-step tokenizations become comparable.
- **The stitching baseline.** Both models saw the same input, so "layer A
  predicts layer B" is partly trivial. A hand-crafted input-feature probe
  (raw window, FFT magnitudes, summary stats) predicts the same targets,
  and only the *gain* above it is reported as evidence of shared learned
  structure.
- **Exploration vs confirmation.** Everything computed on the dev corpus
  is exploratory — many comparisons are looked at, so significant results
  are hypotheses. The confirm stage tests them exactly once on a sealed
  private corpus and reports verdicts; that section of the report is the
  evidence, the rest is context.

## Statistical treatment

The resampling unit is always the **series**: windows within a series are
strongly dependent, both models score the same series, and corruptions are
applied to the same series. Concretely:

- L0 family comparisons are **paired** bootstrap tests of per-series MASE
  differences, Holm-corrected across families; a "strength" requires a
  corrected p below alpha *and* a CI excluding zero. Ratios are kept as
  effect sizes.
- L1 CKA at the peak pair uses a **cluster bootstrap** (resample series,
  keep their windows together); family-conditioned CKA gets per-family CIs.
- L2 reports a CI on the best stitching **gain** by resampling validation
  series with the fitted probes held fixed — evaluation uncertainty,
  deliberately not refit variability.
- L3 keeps per-series sensitivity deltas so fingerprint agreement gets a
  **paired** cluster bootstrap (the same series resample applied to both
  models); behavioral deltas carry CIs.
- Profile probe accuracies are CIs over held-out series against a
  majority-class chance line. Cluster AMI gets a series-bootstrap CI.
- Bootstrap p-values are approximate and floored at 1/n_boot; they are
  decision aids, not precision instruments. `stats.n_boot_heavy` caps the
  expensive statistics separately from the cheap ones.

## Install

```bash
pip install -r requirements.txt          # core
pip install chronos-forecasting timesfm          # the models you compare (timesfm>=2.0; see below)
pip install umap-learn                   # optional, nicer cluster maps
```

## Quickstart

Smoke run (no downloads, no GPU, ~1 minute) — validates every stage
including confirmation and produces a real report from two mock
architectures:

```bash
python run.py --config configs/smoke.yaml
open runs/smoke/report.html
```

Real comparison:

```bash
# 1. point data.path at a tsfm_benchmark PUBLIC (dev) directory
# 2. verify token-time alignment on your installed library versions:
python run.py --config configs/default.yaml --check-alignment timesfm
python run.py --config configs/default.yaml --check-alignment chronos
# 3. explore freely on dev:
python run.py --config configs/default.yaml
# 4. when analysis is FROZEN, enable confirm with the PRIVATE directory
#    and run the one-shot confirmation:
python run.py --config configs/default.yaml --stages confirm,report
```

Stage selection and reruns:

```bash
python run.py --config configs/default.yaml --stages extract,l1,report
python run.py --config configs/default.yaml --stages l2 --force l2
```

Stages skip themselves when their artifacts exist; the report renders
whatever subset ran. Every stage has an `enabled` flag and its own knobs in
the YAML (`configs/default.yaml` documents all of them inline).

## Confirmation discipline

The private corpus is a consumable. Run the confirm stage once, after all
exploration is frozen; it refuses to overwrite existing confirmation
artifacts, requires seal verification by default
(`confirm.require_seal: true`), and logs loudly when it runs. If the
private set is ever peeked at, regenerate a fresh private epoch with
`tsfm_benchmark` rather than reusing it. Findings that were not
pre-registered as dev hypotheses cannot be rescued by the confirm stage —
that is by design.

## GPU notes

Extraction and forecasting run batched under autocast (bf16 by default);
activations are stored float16 in a chunked zarr store; CKA and ridge
solves are closed-form on-device in fp32. Models are loaded lazily and
released between stages (`run.keep_models_loaded: true` to keep both
resident). The main VRAM knobs are `models[*].batch_size`,
`capture_layer_stride`, and the `max_rows` caps in `l1`/`l2`/`internals`.
Bootstraps add analysis-time CPU/GPU cost only, never extraction cost;
tune `stats.n_boot` / `stats.n_boot_heavy` if analysis stages feel slow.

## The alignment caveat (read this once)

Cross-architecture comparison is only as valid as the token-to-time
mapping. Adapters *declare* spans; libraries *change*. The
`--check-alignment` command runs an empirical impulse test: an input
impulse in window *w* must perturb aligned window *w* most, and the command
prints the per-layer diagonal-hit fraction. Values well below 1.0 at late
layers are normal (attention mixes positions); near-zero everywhere means
the declared spans are wrong for your installed version — fix the adapter
before trusting any cross-model number. `alignment.sanity_check: true`
also runs a spot check during extraction.

Tested adapter targets: `chronos-forecasting>=1.2` with `chronos-t5-*`
checkpoints, `chronos-forecasting>=1.4` with `chronos-bolt-*` checkpoints
(patch geometry is read from the checkpoint config; forecasts are
deterministic, so its L3 curves carry no sampling noise), and the `timesfm`
PyPI package's 2.5-only API (`>=2.0`, which dropped the old
`TimesFmHparams`/`TimesFmCheckpoint`/`TimesFm` class entirely, with no
Python>=3.12-compatible release that keeps it) with
`google/timesfm-2.5-200m-pytorch`. TimesFM's attention runs through a fused
SDPA kernel by default with no exposed weights, so `attention_patterns`
temporarily swaps in the library's own unfused dot-product math to recover
them (same output, same query/key/value, restored right after). Its
feed-forward block has no wrapping MLP submodule (`mlp_info` finds
nothing), so MLP mean-ablation is unavailable for this checkpoint family;
head-level ablation still works via the attention output projection. An older
`timesfm` install that still exposes the Hparams API would need the
class-based loader instead of the current adapter.

## Extending

**New model — bring your own HF checkpoint.** Nothing downstream changes in
principle; four real adapters (Chronos-T5, Chronos-Bolt, TimesFM 2.5,
Chronos-2, Sundial) have proven this out, and the one real shared-code gap
a fourth architecture ever surfaced (a capture hook that assumed `tuple`
outputs) was fixed once in `extraction/hooks.py`, not per-adapter.

1. Subclass `ModelAdapter` (`tsfm_lens/models/base.py`): implement `load`,
   `module`, `prepare`, `forward`, `token_time_spans`, `predict`; override
   `postprocess_tokens`/`token_slice` if the sequence carries specials or
   padding. `chronos_bolt_adapter.py` is the compact worked example;
   `sundial_adapter.py` is the most recent one and shows how to route around
   a checkpoint's own broken `.generate()`/cache path without touching any
   shared file.
2. Register it in `tsfm_lens/models/__init__.py`.
3. `python run.py --config <your config> --discover-layers` to pick a
   `layer_regex` over its residual stream.
4. `python run.py --config <your config> --check-alignment <name>` — read
   the full per-layer table, not just whether it ran (see the alignment
   caveat below).
5. Drop it into a config next to any existing model and run the pipeline
   normally — `layer_screen` (below) and every downstream stage work
   unmodified.

**Layer screening.** Before the expensive per-layer analyses (Profile, L1,
SAE), the `layer_screen` stage (`analysis/layer_screen.py`, method
`work_bend` by default) scores each model's own layers as a cheap
residual-trajectory proxy for "worth a closer look," and
`sae.targets: auto` consumes its selection. It runs automatically right
after `extract`; `run_layer_screen_bakeoff.py` is the standalone,
null-controlled comparison that chose `work_bend` over the alternatives.

**SAE phase.** `sae.enabled: true` trains a real `TopKSAE`
(`sae/models.py`/`train.py`) straight from the store's activations, with a
real eval harness (`sae/eval.py`: reconstruction fidelity, dead-feature
rate, forecast preservation) and a ground-truth feature-alignment score
(`sae/ground_truth.py`). `sae/crosscoder.py` is a feasibility-tested (not
yet a pipeline stage) joint dictionary across two models' activations, and
`sae/matching.py` correlates two independently-trained dictionaries'
activation profiles to find cross-model feature partners without one.
`extraction.hooks.token_patch` is the shared intervention primitive
feature ablation and L3 patching both reuse.

## Design decisions and caveats

- **CKA is correlational** and reported as such; the report's framing keeps
  L1 claims geometric, L2 claims linear-translatability, L3 claims causal.
- **Stitching direction matters** (A→B and B→A both reported); R² is
  variance-weighted on a held-out **series** split, never a window split.
  The L2 gain CI holds fitted probes fixed, so it quantifies how stable the
  number is on new series, not how stable refitting would be.
- **Patching is within-model.** Cross-model activation transplants are not
  meaningful without a learned mapping; what is compared is each model's
  restoration-by-depth curve on a shared relative-depth axis.
- **Chronos (sampled) forecasts add noise** to its L3 curves that TimesFM
  and Chronos-Bolt (both deterministic) do not have.
- **Fingerprint comparison interpolates over relative depth**, which
  assumes depth fraction is the right axis to compare a 50-layer decoder
  against a 12-block encoder; it is a reasonable convention, not a fact.
- **The AMI bootstrap resamples label pairs**, not the clustering itself;
  it reflects assignment stability, not KMeans initialization variance.
- **Cluster labels are approximate by construction** (majority family +
  salient signal statistics); they are hypotheses for reading the maps.
- **A smoke-mode confirm is mechanically real but distributionally
  identical** to dev (same generators, shifted seed); genuine confirmation
  value comes from sealed private corpora.
- Metrics (MASE, sMAPE, pinball) are implemented in ~20 lines rather than
  importing a large evaluation framework; formulas are in
  `analysis/l0_behavioral.py` where they can be audited. Statistical
  machinery is similarly ~80 audited lines in `analysis/stats.py` built on
  numpy, because the bootstrap designs (cluster, paired) are the substance
  and generic CI libraries obscure them.

## Layout

```
tsfm_lens/
  config.py            typed YAML config, per-stage enables and defaults
  data.py              sealed-corpus loader (+ jsonl fallback, smoke generator)
  models/              ModelAdapter contract; TimesFM/Chronos/Chronos-Bolt/
                       Chronos-2/Sundial/mock adapters, conformance + capability-
                       matrix generators
  extraction/          hooks, time alignment (+ impulse check), zarr store
  analysis/            stats (bootstrap/Holm), l0 behavioral, layer_screen
                       (cheap per-model layer scoring, feeds sae.targets),
                       internals profile, lens (skip + tuned forecast lens,
                       + spectral_lens's frequency-domain crystallization
                       depth), l1 geometry, l2 stitching, l3
                       perturbation+patching (per window), attention (lag
                       profiles, head/MLP ablation, cross-attn), clustering,
                       exemplars (case studies), confirm (private benchmark)
  sae/                 TopKSAE baseline (models/train/eval/ground_truth),
                       cross-model matching, crosscoder (feasibility-tested,
                       not yet a pipeline stage)
  report/              single-file interactive HTML report
  pipeline.py          stage DAG, artifact skipping, dependency resolution
run.py                 CLI
run_layer_screen_bakeoff.py, run_crosscoder_feasibility.py,
run_spectral_lens.py, run_meta_report.py, ...   standalone scripts that
                       reuse an already-extracted run's config/data/store
                       for probes that aren't (yet, or ever) pipeline stages
configs/               default.yaml (real pair), smoke.yaml (mocks)
tests/test_smoke.py    full pipeline end-to-end + confirmation verdict test
```
