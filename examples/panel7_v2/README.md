# Example run: seven TSFMs, full pipeline, confirmed once on held-out data

This folder is one complete, real TSFM-Lens run on a **seven-model panel**,
published so you can see what the pipeline produces before running it yourself.
It is the run the paper's case study is built on (it supersedes the four-model
run in `../concept_atlas_v2/`, which is kept unchanged). It holds:

- **`report.html`**: the self-contained interactive report, rendered after the
  one-shot confirmation. Download it and open it in a browser; GitHub does not
  render it.
- **`run/`**: the lightweight artifacts the report was built from, in the same
  layout as a real run directory, plus `confirm/confirmation.json`,
  `hypotheses.json` and `reliability_panel7.json`. The large ones are left out;
  see below.
- **`config.yaml`**: the config (a copy of `tsfm_lens/configs/panel7_v2.yaml`).

| | |
|---|---|
| Models | TimesFM 2.5 (231M), Chronos-2 (119M), Sundial (base, 128M), Chronos-Bolt (small, 48M), Timer (84M, `generic_hf`), Time-MoE (50M checkpoint, 113M total, `generic_hf`), Chronos-T5-Base (201M, encoder captured) |
| Corpus | `benchmark_large_v2`: 965 dev series (exploration); 971 sealed private series, v2 epoch 3 (confirmation) |
| Context / horizon | 480 / 64 steps (480 is a multiple of Timer's 96-step token) |
| SAE targets | 56 layers (TimesFM 10, Chronos-2 9, Sundial 9, Chronos-Bolt 4, Timer 6, Time-MoE 9, Chronos-T5-Base 9); 55 measured, Chronos-2 block 11 withheld by the reach probe |
| Null | `sae.ablation_null: profile_matched`, 16 null draws per feature, empirical chance rate on |
| Stages | all 19, from `corpus` to `report`, then `register` and `confirm` |
| Hardware | 1x NVIDIA RTX A5000 |
| Recorded stage time | 75,899.62 s (21.1 h): `sae` 14,614 s, `concepts` 55,608 s (shared-input agreement 45,189 s) |
| Report | 240 findings, 123 multiplicity families, 4,238 comparisons |

## How it was produced

```bash
# 1. Build the corpus (from the repo root; CPU only)
PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_full.py \
    --config tsfm_benchmark/configs/large_run_v2.yaml \
    --out benchmark_large_v2 --references monash --reference-limit 120

# 2. Run every stage on the dev split (from tsfm_lens/; one GPU)
cd tsfm_lens
python run.py --config configs/panel7_v2.yaml
```

The private split is v2 **epoch 3** (971 series after the leakage gate). An
earlier mint, v2 epoch 1, was rejected before any model saw it because 716 of
its 961 series were identical to series in the v1 epoch-1 split that an earlier
confirmation had consumed; epoch seeds depend only on (seed, epoch), so the
epoch number is shared across corpus versions that reuse generator specs.
Epoch 3 has 0 shared sample hashes with every other split of either corpus
version.

**The first confirm attempt crashed before writing any statistic and was
restarted.** Every private battery finished, then the claim ledger raised a
`KeyError` because the family map did not yet hold the `reliability_u1` family
(a bug in a code path no test had exercised: concept claims and U1 claims
registered together). `confirmation.json` was never written, so the stage's own
`repeated_look` guard did not fire, and no private number was read or fed back.
The registry (sha256 `ff6c4f9e...`) and every frozen spec are unchanged; the
restart differs only by the crash fix. This is one test of the frozen registry
run twice for compute reasons, not a second look. The artifact records
`repeated_look: false`. Full account: ROADMAP section 38.2 on the `dev`
branch.

After the confirm, a defect in the artifact's own text was found: 19 of the 21
non-confirmed single-feature causal claims carry a `non_replication_reading` of
"untestable", which is wrong (they were tested and not detected; the adaptive
procedure reaches the Holm floor). The verdicts are unaffected. The report
recomputes the reading from stored fields at render time and
`confirmation.json` was not rewritten, so that one field in the artifact still
carries the old sentence.

## What the run found

Each result names its **evidence class**. *Held-out* means the claim was
registered on the dev split and tested exactly once on the sealed private split.
Everything else is exploratory, on the dev split.

### Which model forecasts best (behavioral)

| Model | Median MASE (dev) | Params |
|---|---|---|
| Chronos-2 | 1.2002 | 119M |
| TimesFM | 1.2560 | 231M |
| Sundial | 1.4131 | 128M |
| Chronos-T5-Base | 1.4231 | 201M |
| Chronos-Bolt | 1.5487 | 48M |
| Timer | 1.9306 | 84M |
| Time-MoE | 2.8642 | 113M |
| naive / seasonal naive | 2.7593 / 1.9819 | – |

- **Held-out:** Chronos-2's dev-registered advantage over TimesFM on `mixture`
  series confirms: 0.2085 [0.1189, 0.3133], Holm-corrected p 0.0002, n 290.
  Overall on the private split the paired difference is -0.0696
  [-0.1293, -0.0081], p 0.0252, n 971 (positive would favour TimesFM).
- Time-MoE's median MASE is above the naive baseline's, and Timer's is about the
  seasonal naive's. Timer and Time-MoE output a point forecast only.

### Do the models represent inputs alike? (geometric)

- **Held-out:** the TimesFM xf.4 / Chronos-2 block 7 layer pair has linear CKA
  0.4257 on dev (shuffled-series null 0.0383) and 0.4109 [0.4014, 0.4266] on
  private data; it replicates.
- Seven similarity metrics over the 21 model pairs agree only weakly
  (Kendall's W 0.2544, p 0.0095).

### Do they react to damaged input the same way? (causal, within each model)

- **Held-out:** 6 of 9 TimesFM-versus-Chronos-2 depth-profile agreements
  replicate (overall rank correlation 0.5239 [0.5040, 0.5442]). The 3 that do
  not: `frequency_shift` and `smooth` are stronger on private data, and
  `level_shift` flips sign (-0.1882 to 0.1086 [-0.0252, 0.1594]).

### Concepts: what the models learn and share (SAE features + ablation)

- **Causal features.** 442 (feature, channel) cells clear the profile-matched
  null against 178.75 expected by the empirical chance rate (2.4727x) over 55
  measured targets, with 194 pooled causal features. By model: Chronos-Bolt
  5.6986x, Time-MoE 4.0109x, TimesFM 3.1445x, Chronos-2 2.9381x, Sundial
  2.1562x, Chronos-T5-Base 1.6429x, **Timer 1.0169x (0 of its 6 targets
  significant: at chance)**.
- **Atlas.** 11 concepts, 9 multi-model, none spanning more than 3 of the 7
  models; 9 of 11 seed-stable. Cross-model mixing is *below* chance (9
  multi-model concepts against a null mean of 10.755, p_below 0.0249; mean
  model purity 0.5909 against 0.4741): the atlas is **segregated by model**.
- **Shared-input causal agreement.** 1,290 tests: 7 same causal effect, 22 acts
  differently, 9 shape only, 45 level only, 133 no specific agreement, 1,074
  not scorable. All 7 "same" verdicts are Chronos-2 and Chronos-Bolt, in one
  atlas concept. A known-answer test of this rung on a planted pair of
  architectures (`FINDINGS.md` MN-30) shows it is specific (0 of 9 false
  "same") but insensitive (it finds a shared single-direction concept in 0.4 of
  seeds), so an absence of "same" across architectures is not evidence of
  difference.

### Held-out confirmation, one look (private v2 epoch 3, 971 series)

43 of 82 replicable claims confirm; 6 are not testable; 33 are not confirmed.

| Family | Confirmed / tested |
|---|---|
| L0 accuracy (Chronos-2 vs TimesFM, `mixture`) | 1/1 |
| L1 peak CKA pair | replicates |
| L3 fingerprint agreement | 6/9 |
| Concept transfer (same inputs) | **20/20** |
| Single-feature causal effects | **10/31** (1 not testable) |
| Multi-model atlas concepts | 0/3 (the 1 confirmed atlas claim is single-model) |
| Shared-input agreement (L5) | **0/6** (5 not testable) |
| Structure ("no concept in all models"; "most prominent features are causally null") | 2/2 |
| U1 reliability from internals (Timer, Time-MoE) | 2/2 (predictive, not causal) |

- Transfer: Holm maximum 0.009995 (the floor); private AUC 0.9194-0.9696. The
  20 claims come from 2 concepts. This is the third private corpus on which 20
  frozen transfer claims confirm.
- Causal: by model, confirmed/registered TimesFM 3/4, Time-MoE 2/7, Sundial
  2/4, Chronos-Bolt 2/6, Chronos-2 1/7, Chronos-T5-Base 0/4.
- Structure: 0.835 of 927 prominent features are causally null (one-sided lower
  bound 0.8166; dev 0.8102).
- U1: log-MASE Spearman gain from SAE-family activations over the free
  quantile-width baseline, refit on private data (n 935): Timer 0.0446
  [0.0292, 0.0605], Time-MoE 0.011 [0.0032, 0.0194]. Routing (U2) is negative
  on dev and was not registered.
- **Reading:** models share what they select, not (demonstrably) what they do
  with it.

### Known caveats in this run

- The dev U1/U2 numbers are in `reliability_panel7.json` (n = 941 series,
  1000 bootstraps over series).
- Real-derived series (weather, ETT, electricity, traffic) may overlap the
  models' own pretraining data. This cannot be verified.
- 12 of 33 concept parts are provenance-driven: their top series come mostly
  from one real-derived generator.
- Timer's concepts sit on targets whose ablation clears are not above chance.

## What is in `run/`

The same layout as `../concept_atlas_v2/run/`, plus:

| Path | What it holds |
|---|---|
| `confirm/confirmation.json` | the one-time held-out test (K2): all claim families |
| `hypotheses.json` | the frozen dev registry |
| `reliability_panel7.json` | dev U1/U2: failure prediction and routing from internals |

`corpus/card.md` is absent (the run did not write one).

**Left out for size** (regenerate by running the config):
- `activations.zarr` and the per-target SAE weights and ablation batteries;
- `sae/meta.json` (9.3 MB);
- `sae/shared_input_agreement.json` (11.5 MB);
- `sae/atlas_transfer.json` (1.2 MB);
- `report_preconfirm.html` (the pre-confirm render);
- the `.npz` arrays behind figures.

Every number in the report comes from those full artifacts. The report embeds
everything its figures need, so it opens on its own.
