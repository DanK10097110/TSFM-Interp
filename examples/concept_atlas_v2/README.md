# Example run: four TSFMs, full pipeline, confirmed on held-out data

This folder is one complete, real TSFM-Lens run, published so you can see what
the pipeline produces before running it yourself. It holds:

- **`report.html`**: the self-contained interactive report. Download it and
  open it in a browser; GitHub does not render it.
- **`run/`**: the lightweight artifacts the report was built from, in the same
  layout as a real run directory. The large ones are left out; see below.
- **`config.yaml`**: the config, with paths for the current repo layout.

| | |
|---|---|
| Models | TimesFM 2.5 (200M), Chronos-2, Sundial (base 128M), Chronos-Bolt (small) |
| Corpus | `benchmark_large_v2`: 965 dev series (exploration), 800 sealed private series (confirmation) |
| Context / horizon | 512 / 64 steps |
| Stages | all 19, from `corpus` to `report` |
| SAE targets | 29 layers across the four models (chosen by `layer_screen`) |
| Hardware | 1× NVIDIA RTX A5000 |
| Report | 18 sections, 126 findings |

## How it was produced

```bash
# 1. Build the corpus (from the repo root; CPU only)
PYTHONPATH=. python3 tsfm_benchmark/example_runs/run_full.py \
    --config tsfm_benchmark/configs/large_run_v2.yaml \
    --out benchmark_large_v2 --references monash --reference-limit 120

# 2. Run every stage (from tsfm_lens/; one GPU)
cd tsfm_lens
python run.py --config configs/concept_atlas_v2.yaml
```

`run/config_resolved.yaml` is the config exactly as it ran. Its corpus paths
(`../../benchmark_large_v2/...`) predate the repo's folder flattening.
`config.yaml` is the same config with current paths (`../benchmark_large_v2/...`).

The compute stages ran at commit `141d7bd`. The report was re-rendered at
`a6a1d59`, after three report-text fixes and a concept-family naming fix. Those
fixes change only titles and sentences, never numbers. The re-render is why
`run/run_manifest.json` records `a6a1d59`.

The concepts stage dominates the runtime. Its shared-input agreement step
alone ran 1223 cross-model tests in about 3 hours.

## What the run found

Each result names its **evidence class**, which is how strong a claim it
supports. *Held-out* means the claim was registered on the dev split and then
tested exactly once on the sealed private split, which no earlier run had
looked at. Everything else is exploratory, on the dev split.

### Which model forecasts best (behavioral)

| Model | Median MASE (dev) | Params | Latency / series |
|---|---|---|---|
| Chronos-2 | 1.1251 | 119M | ~21 ms |
| TimesFM | 1.1829 | 231M | ~38 ms |
| Sundial | 1.3335 | 128M | ~7 ms |
| Chronos-Bolt | 1.4913 | 48M | ~14 ms |
| naive / seasonal naive | 2.6829 / 2.0280 | – | – |

MASE is the forecast error divided by that of a naive forecast, so lower is
better. The table's last row applies the same metric to the two simple
baselines themselves.

- **Held-out:** Chronos-2 beats TimesFM overall on the private split. The effect
  is −0.1478 [−0.2106, −0.0883], p 0.0005 (positive would favour TimesFM). Its
  dev-registered advantage on `mixture` series also confirms: 0.3073
  [0.2085, 0.4126], Holm-corrected p 0.0005.
- Sundial's 80% forecast intervals cover only 37% of outcomes. It is the one
  poorly calibrated model.
- Chronos-2's edge over Chronos-Bolt costs about 5.4× the compute.

### Do the models represent inputs alike? (geometric, linear-translatable)

- **Held-out:** the most similar layer pair between TimesFM and Chronos-2
  (TimesFM block 4, Chronos-2 block 7) has linear CKA 0.4262 on dev and 0.4253
  [0.4162, 0.4423] on private data. The shuffled-series null is 0.037.
- Every ordered model pair's activations can be linearly predicted from the
  other's better than from input features alone (all 12 L2 findings). This
  shows shared structure, but not causation.
- Seven similarity metrics do **not** agree on which pair is closest
  (Kendall's W 0.28, p 0.07). CKA ranks Sundial–Chronos-Bolt first (0.928);
  error agreement ranks TimesFM–Chronos-2 first (0.947). Neither ranking is
  wrong: each metric answers a different question.

### Do they react to damaged input the same way? (causal, within each model)

- **Held-out:** 9 of 9 claims about where in depth TimesFM and Chronos-2 react
  to each kind of input corruption replicate. The overall rank correlation is
  0.5098 [0.4889, 0.5273].
- Level shifts move every model's forecast the most. Dropout or spikes move
  them the least.

### Concepts: what the models learn and share (sparse autoencoder features + ablation)

> **Correction (2026-09-30): this section was measured with a lenient null.**
> The random-direction null used in this run removed one uniform, batch-averaged
> amount from every token. A feature's ablation removes its full activation where
> it fires hardest, so strong features cleared that null on size alone. A
> known-answer forecaster with planted concepts caught the problem: 17–25% of
> features on a layer with no planted effect cleared, on 5 of 5 seeds.
>
> This same run was re-scored under a **profile-matched** null, in which each null
> draw removes the feature's own per-token amount along a random direction:
> - causal features: 293 → **110** (3.06× the empirical chance rate; 20 of 28
>   layers significant);
> - tight concepts: 27 → **7** (3 multi-model);
> - convergent concepts: 13 → **0**;
> - features clearing no channel: 56% → **84%**.
>
> Treat the numbers below as upper bounds. The profile-matched null is the
> default for new runs, and the original lenient null remains available as
> `sae.ablation_null: mean_magnitude`. Full account: `FINDINGS.md` MN-29 and the
> "re-measured" lines on each affected entry.

An SAE (sparse autoencoder) splits each chosen layer into features. Each
feature is then removed on the series it fires on, and the forecast change is
measured against a random-direction null. This gives causal, within-model
evidence.

- **293 features** in the four models measurably move the forecast.
- **Concept families.** Grouped by what they *do* to the forecast, they fall
  into 9 readable families covering 288 of 293 features. The families are:
  Level lowerers, Level raisers & Accuracy improvers, Volatility dampeners,
  Trend boosters & Accuracy improvers, Seasonality amplifiers, Trend boosters &
  Volatility amplifiers, Level raisers & Trend boosters, Trend dampeners, and
  High-frequency shifters.
  - They are a readable way to divide a **continuum**, not discovered clusters.
    A column-shuffled control produces as many families (p 1.0).
  - Each family's members lean toward one model more than chance would
    predict.
- **Tight concepts.** A stricter clustering finds 27 tight concepts. 22 of 27
  reproduce across independently trained SAE seeds.
  - None spans all four models.
  - When features in different models share an effect, it usually comes from
    *different* inputs: 13 convergent concepts versus 2 fully shared.
- **Held-out:** 20 of 20 registered claims that another model's dictionary picks
  out the same series as a concept replicate on the private split.
  - Holm-corrected p ≤ 0.009995; private AUC 0.9708–0.9951.
  - The 20 claims come from only 2 concepts, so this is two concepts
    replicating, not twenty independent ones.
- **Same input, compared effects.** On 1223 cross-model tests, features in
  different models were ablated on the *same* series:
  - only 420 tests were scorable (both sides clear their own nulls);
  - of those, 11 show the same causal effect and 64 (15.2%) act differently;
  - the rest agree only in level or shape, or show no specific agreement.

### Known caveats in this run

- The private split was not tested for exchangeability with the dev split. No
  `cross_split.json` was built for this corpus, and the report states this as
  a skip reason.
- Real-derived series (weather, ETT, electricity, traffic) may overlap the
  models' own pretraining data. This cannot be verified.
- 27 of 76 concept parts are provenance-driven: their top series come mostly
  from one real-derived generator.

## What is in `run/`

| Path | Stage | What it holds |
|---|---|---|
| `config_resolved.yaml`, `run_manifest.json` | – | config as run; per-stage fingerprints, package versions, checkpoint revisions |
| `tiers.json`, `shapes.json`, `routing.json` | gates | capability tier per model; run shape (panel); routing |
| `corpus/card.{json,md}` | corpus | composition and the audit trust ladder |
| `alignment/alignment_check.json` | extract | token-to-time alignment gate record |
| `budget/model_budget.json` | budget | params by role, FLOPs, latency, VRAM |
| `frontend/frontend.json` | frontend | scale equivariance, NaN handling, staleness |
| `layer_screen/selection.json` | layer_screen | which layers got the expensive analyses |
| `l0/*.json`, `l0/metrics.parquet` | l0 | per-series MASE/sMAPE/pinball, paired bootstrap, calibration, noise floor |
| `internals/profile.json` | internals | effective dimension, decodability, CKA-to-input per layer |
| `lens/lens.json` | lens | crystallization depth per model |
| `l1/meta.json`, `l1/cka_sae.json` | l1 | best CKA pair with CI and null, depth curve, RSA |
| `l2/stitching.json` | l2 | stitching gain over the input-feature baseline |
| `l3/{meta,patching}.json` | l3 | corruption fingerprints, activation patching |
| `attention/meta.json` | attention | lag profiles, periodicity heads, head ablation |
| `clustering/*.json` | cluster | activation clusters, AMI across models |
| `exemplars/exemplars.json` | exemplars | the worked case studies |
| `fairness/card.json` | report | what each model was and was not given |
| `sae/concept_*.json`, `sae/concepts.json`, `sae/transfer.json`, `sae/descriptions.json` | sae, concepts | concepts, atlas, families, stability, profiles, transfer, descriptions |
| `hypotheses.json` | register | the frozen dev hypotheses |
| `confirm/confirmation.json` | confirm | the one-time held-out test |
| `report/findings.json`, `report/results.csv` | report | every finding with evidence class and caveat |
| `report/model_similarity.json` | report | the 7 pair-similarity metrics and their rank consensus |

**Left out for size** (regenerate by running the config):
- `activations.zarr` (1.7 GB);
- SAE weights and per-target ablation batteries (3.5 GB);
- `sae/meta.json` (5 MB);
- `sae/shared_input_agreement.json` (10.6 MB);
- `sae/atlas_transfer.json` (1.2 MB);
- the `.npz` arrays behind figures.

Every number in the report comes from those full artifacts. The report embeds
everything its figures need, so it opens on its own.
