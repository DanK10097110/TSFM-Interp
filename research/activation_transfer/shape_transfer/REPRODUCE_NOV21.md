# Reproducing the 21 November studies

Run from the repository root in the environment used for TSFM-Lens with its pinned model weights, local GiftEval/Monash sources and public panel SAE checkpoints available. The original data, feature draws and most ordinary-state capture prerequisites are prepared by the earlier research drivers. The study scripts use the existing forecast adapters and write physical effect arrays under each study's `results/` directory. Large clean states and Jacobians go to named scratch directories in `/tmp`. [Earlier 480-step material and study-window setup](../replication/README.md) · [Eight-lead derivatives and initial studies](REPRODUCE.md) · [Additional daily/traffic cohort preparation](fresh_cohort/README.md) · [Blind-frequency cohort and 16-lead calibration](blind_frequency/README.md)

The following shell excerpts use the same package prefix. The decoders run on CPU; several independent model jobs can be batched or started separately. NumPy/PyTorch and the number of CPU threads affect exact low-amplitude floating-point values. The new quantile confirmation explicitly matches the calibration and holdout reference archives’ reduction order. Timer’s pinned checkpoint uses trusted local model code and a writable Hugging Face module cache.

```bash
export PYTHONPATH=.:tsfm_lens:tsfm_benchmark
export HF_HUB_OFFLINE=1
export HF_MODULES_CACHE=/tmp/tsfm_requests_hf_modules
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
p=research.activation_transfer.shape_transfer
```

## Native source controls, depths and projected directions

The depth stage includes the original 416 calibration/evaluation histories plus 256 additional evaluation histories. The 1,024 train rows contribute site norms and PCA directions. The five receiver sites per architecture are block outputs over the matching 32-observation data window. The independent seed dictionaries in `response_subspace` are the refits from the original boundary study.

```bash
for m in bolt timesfm; do
  python -m "$p.depth.geometry" "$m"
  python -m "$p.depth.intervene" "$m"
done
for d in 0 1 3 4 5; do python -m "$p.depth.intervene" bolt --depth "$d"; done
for d in 0 5 10 15 19; do python -m "$p.depth.intervene" timesfm --depth "$d"; done
python -m "$p.depth.summarize"
for m in bolt timesfm; do python -m "$p.depth.paired_sites" "$m"; done
python -m "$p.depth.score_paired"

python -m "$p.self_transfer.run" prepare
for m in bolt timesfm timer; do python -m "$p.self_transfer.run" "$m"; done
python -m "$p.self_transfer.score"
python -m "$p.self_transfer.figure"
python -m "$p.self_transfer.audit"

for twin in pretrained seed7513 seed9749; do python -m "$p.random_receiver.run" "$twin"; done
python -m "$p.random_receiver.summarize"
```

The Timer self-control checks against the earlier Timer validation archive. Its point forecast is degenerate across quantiles, so only median/point transfer studies include it. The random-receiver audit independently reconstructs one seeded checkpoint. [Timer prerequisites and revision](third_receiver/README.md) · [Random numerical checks](random_receiver/README.md)

## Quantiles and unseen deciles

The public ordinary path is rerun at nine native deciles, while the inverse only receives gradients for the anchors specified by the corresponding arm. Train the independent quantile interpolators on historical donor removals. The 128+128 blind-frequency cohort supplies the separate normal-score confirmation. The current source and adaptive recipient use the *same* offered anchors in the same-information comparison.

```bash
for m in bolt timesfm; do python -m "$p.quantiles.geometry" "$m"; done
for s in bolt timesfm; do python -m "$p.quantiles.intervene" "$s"; done
python -m "$p.quantiles.summarize"
for s in bolt timesfm; do python -m "$p.quantiles.interior" "$s"; done
python -m "$p.quantiles.score_interior"
for m in bolt timesfm; do python -m "$p.quantiles.normal_score" geometry "$m"; done
for s in bolt timesfm; do python -m "$p.quantiles.normal_score" run "$s" --batch 24; done
python -m "$p.quantiles.score_normal"
python -m "$p.quantiles.five_interp"

for m in bolt timesfm; do
  python -m "$p.response_subspace.run" "$m"
  for s in 0 1 2; do python -m "$p.response_subspace.run" "$m" --seed "$s"; done
  python -m "$p.response_subspace.dose" "$m"
done
python -m "$p.response_subspace.summarize"
python -m "$p.response_subspace.score_dose"
```

## Full-dose feedback, interaction and horizon tests

The machine selector preserves its exclusion list and normalized-input hash. The VM controller's scalar search and correction are prospective on those 256 unseen machines. The stored factorial arrays include both unpatched and same-batch, zero-displacement reference differences. The interaction scorer consistently uses the latter. Run the nested feedback jobs after the parent depth stage; run the lead-placement jobs after the blind-frequency and historical continuation prerequisites.

```bash
python -m "$p.depth.feedback.data" data
for m in bolt timesfm; do python -m "$p.depth.feedback.data" "$m"; done
for d in 0 10 19; do python -m "$p.depth.feedback.run" "$d"; done
python -m "$p.depth.feedback.score"
python -m "$p.depth.feedback.curvature"
python -m "$p.depth.feedback.last_block"
python -m "$p.depth.feedback.score_last_block"

for s in bolt timesfm; do
  for t in native bolt timesfm; do python -m "$p.interactions.run" "$s" "$t"; done
done
python -m "$p.interactions.score"
shards=$(mktemp -d)
for c in original extra; do
  for k in 0 1 2; do python -m "$p.interactions.conditional" "$c" --parts 3 --part "$k" --out "$shards"; done
  python -m "$p.interactions.conditional" "$c" --parts 3 --merge-from "$shards"
done
python -m "$p.interactions.conditional" blind
python -m "$p.interactions.score_conditional"

for m in bolt timesfm; do python -m "$p.lead_budget.run" geometry "$m"; done
for s in bolt timesfm; do python -m "$p.lead_budget.run" run "$s" --batch 24; done
python -m "$p.lead_budget.score"
python -m "$p.lead_budget.residual"
python -m "$p.lead_budget.figure"
python -m "$p.forecast_loss.run"
python -m "$p.forecast_loss.strata"
```

## Independently implemented studies and checks

The individual study reports contain their own cohort locks, costs, scripts, model replays and mutation checks. [Reduced calibration budget](calibration_budget/README.md) · [Retrieval using dense or sparse donor states](state_predictor/README.md) · [Active target feature shortlists](feature_groups/README.md) · [Moving the forecast origin](rolling/README.md) · [Independent off-support confirmation](rolling/fixed_2622/README.md) · [Historical named-pair interventions](level_pair/README.md) · [Blinded finite-dictionary selection](dictionary_prior/README.md) · [Independent SAE fits and prospective gate replication](dictionary_prior/refit/README.md) · [Common input perturbations](input_bridge/README.md) · [Actual cross-checkpoint writes passed through the receiving SAE](dictionary_prior/receiver_encoding/README.md) · [Independent scientific review](method_review/REVIEW.md)

```bash
python -m "$p.cohort_audit"
python -m "$p.check_plants"
python -m pytest -q research/activation_transfer tsfm_lens/tests/test_smoke.py \
  tsfm_lens/tests/test_contrast_steering.py tsfm_lens/tests/test_steering_api.py \
  tsfm_lens/tests/test_steering.py tsfm_lens/tests/test_sae.py
```

The combined suite collected **177 tests**. All passed across disjoint file partitions (47, 25, 33, 38 and 34). One existing TSFM-Lens smoke test returned a value and produced a pytest warning. Twenty-two independently planted defects each caused the corresponding known-answer test to fail. [Known-answer mutation results](plant_results.json) · [Unique series and exact-input overlap across the named cohorts](cohort_audit.json) · [Isolated-copy replay driver](replay_20261121.py)

Ten isolated whole-model jobs in the first two snapshots copied the research Python sources into separate writable roots, reused the same trained checkpoints and prior measurement caches, reran physical forward passes and compared every archived array. All compared values were bit-identical on this machine. These reruns check implementation replay with fixed inputs. [Seven original jobs](replay_20261121.json) · [Three follow-up jobs](replay_20261121_followup.json)

```bash
python -m "$p.replay_20261121" prepare
for job in pair_depth self_timer inner_dist distribution rotated factorial feedback; do
  python -m "$p.replay_20261121" "$job"
done
python -m "$p.replay_20261121" finish
followup=$(mktemp -d)
python -m "$p.replay_20261121" prepare --work "$followup"
for job in conditional lead_placement last_block; do
  python -m "$p.replay_20261121" "$job" --work "$followup"
done
python -m "$p.replay_20261121" finish_followup --work "$followup"
```


The third snapshot covers the prospective gate study, the TimesFM equal-length control and the realized-future reconstruction. The latter repeats scoring and raw outcome extraction; the other two repeat physical interventions. Its input archives are fixed in all cases. [Final isolated replay receipts](replay_20261121_final.json)

```bash
last=$(mktemp -d)
python -m "$p.replay_20261121" prepare --work "$last"
for job in gate_confirmation gate_norms future_outcomes; do
  python -m "$p.replay_20261121" "$job" --work "$last"
done
python -m "$p.replay_20261121" finish_final --work "$last"
```


Render and check the selected Markdown reports with `python -m "$p.review_reports_20261121"`. This checks that all linked local files and figures exist and are visible to Git, then reconciles the exact-array replay receipts. Sixteen isolated replay jobs compared 317 arrays across 19 archives. Every value matched, including in the job that reconstructs observed futures and rescores outcomes. [Rendered-report verification record](validation_20261121.json)

The final replay covers the TimesFM-donor physical panel and one Bolt-donor seed comparing rotated, history-matched decoder spans with fixed random spans, and all four assignments of intact active spans to different histories. Regenerate it with `python -m "$p.replay_20261121" prepare --work "$last"`, then run `decoder_span`, `span_control`, `swap_support` and `finish_span` with the same work directory. [Exact within-span replay](replay_20261121_span.json)
