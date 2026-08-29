# Functionality Summary — what this repo does, and how usable it is

> **Scope.** A short orientation for someone deciding whether to use this repo, and
> for whom. `CLAUDE.md` is the architecture-and-doctrine reference (why things are
> built the way they are, and which invariants are load-bearing); `ROADMAP.md` is the
> forward plan and the research record. This file is neither — it is the one-page
> answer to *"what can I actually do with this today?"*
>
> Written 2026-08-28. Where a number here could go stale (stage counts, test counts),
> the authoritative source is named beside it.

---

## 1. In one paragraph

Two independently installable halves. **`tsfm_benchmark`** generates synthetic
time-series corpora with exact ground truth, audits them for leakage against real
reference data, seals them with hash manifests, and separately validates that a
generated corpus is actually diverse rather than 5,000 copies of the same shape.
**`tsfm_lens`** takes two (or, now, one or many) time-series foundation model
checkpoints, runs them through a 17-stage layered analysis — behavioral, cost,
representational, causal, attention, sparse-autoencoder — and renders one
self-contained interactive HTML report in which every cross-model claim is either on
an axis both models genuinely share or is printed next to the measured size of the
asymmetry. The pipeline's distinguishing property is not breadth; it is that the
report's conclusions are *derived from artifacts by printed rules* rather than
written as sentences, and that a claim which cannot be supported renders as a stated
absence rather than being quietly dropped.

## 2. What you can do with it today

| I want to… | Command | Needs |
|---|---|---|
| Kick the tyres end to end | `python run.py --config configs/smoke.yaml` | nothing — CPU, mocks, no downloads, ~20 min |
| Analyze **one** model on its own | `--config configs/smoke_solo.yaml` | comparison stages self-drop with a stated reason |
| Compare **two** real checkpoints | `--config configs/medium_run_chronos_base.yaml` | GPU + HF checkpoints |
| Compare **three or more** | `--config configs/smoke_panel.yaml` | all C(n,2) pairs, one designated reference |
| Analyze a checkpoint **with no adapter written** | `adapter: generic_hf` + `--probe-adapter <name>` | probes 4 seams, refuses rather than guesses |
| Check an environment before a long run | `python run.py --doctor` | loads models, runs the alignment gate |
| Find the right config | `python run.py --list-configs` | prints all 46 by run shape, with purpose |
| Verify a finished run is still reproducible | `python run.py --verify-provenance <run_dir>` | no model load |
| Generate a benchmark corpus | `example_runs/run_full.py --config configs/example.yaml --references monash` | CPU only |
| Prove that corpus is diverse | `example_runs/run_validation.py --corpus … [--enforce-gates]` | CPU only |
| Compare N finished runs | `python run_meta_report.py --runs a,b,c` | reads artifacts, re-runs nothing |

Plus ~28 standalone `run_*.py` studies (scaling ladder, layer-screen bake-off,
crosscoder ladder, seasonality circuit, spectral lens, error fingerprinting,
agreement, parameter/horizon/context sweeps) that operate on **already-extracted
runs at zero forward passes** — that "reads existing artifacts, loads no checkpoint"
contract is the repo's most reusable design idea.

## 3. The analysis stack

17 pipeline stages (`pipeline.stage_names()` is the authority). Each exists because
of the previous one's limitation, and each states its evidence class in the report:

- **`extract`** — activations captured on a shared `[series, window, dim]` axis via
  measured token→time spans, stored chunked in zarr, gated by an impulse-alignment
  check that raises rather than warns.
- **`budget` / `frontend`** — parameters, FLOPs (**measured** with torch's
  `FlopCounterMode`, so it works on unseen architectures), latency, VRAM; and what
  each tokenizer does to the input before any layer.
- **`layer_screen`** — which of a model's own layers are worth expensive analysis.
- **`l0`** — accuracy per data family, paired bootstrap, Holm-corrected across
  **every model pair jointly**, plus quantile calibration. *(behavioral)*
- **`internals` / `lens`** — effective dimensionality, probe decodability, and where
  in depth the forecast crystallizes. *(descriptive / depth-resolved)*
- **`l1`** — linear CKA between every layer pair, against a shuffled-series null.
  *(geometric — correlational)*
- **`l2`** — ridge stitching, reported only as the **gain over an input-feature
  baseline**, because both models saw the same input. *(translatable)*
- **`l3`** — corruption-sensitivity fingerprints and within-model activation
  patching, per layer and per window. *(causal within-model)*
- **`attention` / `cluster` / `sae` / `exemplars`** — head taxonomy and ablation at
  two lag resolutions, partition agreement, sparse dictionaries with ground-truth
  feature alignment, and worked case studies.
- **`register` / `confirm`** — dev findings are frozen, then tested **once** against
  a sealed private corpus. This is the only part of the report that is confirmation
  rather than exploration, and the private corpus is treated as a consumable.

Five architectures have hand-written adapters (TimesFM 2.5, Chronos-T5,
Chronos-Bolt, Chronos-2, Sundial), plus a zero-code `generic_hf` probe path verified
end to end on a checkpoint nobody wrote an adapter for. Capability **tiers are
derived from what an adapter implements**, never declared, and a tier-0 black box
that only forecasts still produces a real (smaller) report.

## 4. What makes it trustworthy

These are the things worth copying even if you never run the pipeline:

1. **Negative results are kept.** The flagship crosscoder lost against its own
   pre-registered decision rule and the loss is written up rather than re-searched.
   A cross-model reliability heuristic lost to a free baseline in 10 of 11 cases and
   was deliberately *not* wired in.
2. **Every headline number has a floor beside it.** Untrained-weight twins,
   shuffled-series nulls, label-permutation nulls, repeat-run noise floors,
   `p`-floors from the bootstrap itself. The report's scorecard prints the rule that
   decided each verdict, so a reader can disagree with the arithmetic rather than
   only with the English.
3. **Failures are recorded as traps with their cost.** `CLAUDE.md` §11 is ~39 entries
   of "this looked fine and was wrong, here is the mechanism." Several would
   otherwise have been rediscovered repeatedly.
4. **Missing capability degrades loudly and by name.** A dropped section says *why*
   (tier, run shape, absent artifact), never "artifacts missing".

## 5. Honest usability assessment

**Strong:** one command to a complete report; a preflight doctor; CI on three
dependency pinnings; 100+ analysis tests plus a benchmark suite; a glossary, per-stage
"what this tells you" docs, a worked example that is itself a regression test, and a
failure-mode gallery.

**Rough edges, in the order they will bite you:**

- **Config sprawl.** ~46 configs in one flat directory, most of them frozen
  experiment records rather than starting points. `python run.py --list-configs`
  now prints them grouped by run shape with each file's own purpose line, derived
  from the files rather than from an index that could go stale. The four you
  usually want are `smoke.yaml`, `smoke_solo.yaml`, `smoke_panel.yaml`,
  `default.yaml`.
- **The full test suite takes ~3 hours**, dominated by `test_smoke.py` (~20 min).
  Run `tests/test_smoke.py` first and the rest in the background.
- **Real-checkpoint runs are not one-button** — they need a GPU, a working
  `timesfm`/`chronos-forecasting`/`transformers` install, and several load-bearing
  version pins (`zarr<3`, `datasets<3`) whose absence fails in non-obvious ways.
  `DEPENDENCIES.md` and `--doctor` exist precisely for this.
- **Univariate only, by decision.** A cross-series attention axis has no time
  interval and cannot be pooled onto the shared window axis; multivariate models are
  analyzed as univariate ones with a stated unmeasured axis.
- **Chronos-T5's decoder is uncaptured** — a depth-located claim about it is a claim
  about its encoder, and the report auto-qualifies it with the measured captured-FLOP
  fraction.

**Who it is for.** Someone comparing TSFM checkpoints who cares more about whether a
comparison is *fair* than about getting a number quickly. It is not a
one-line-import library, and it does not pretend to be.
