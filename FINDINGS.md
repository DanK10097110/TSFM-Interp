# FINDINGS.md — What this project has learned about TSFM internals

The curated results ledger of the TSFM interpretability project. It records what
has been learned about **whether different time-series foundation models learn
shared or unique internal representations, where in depth and when**, what each
model is like, and which methods do and do not work, including confirmed
negatives. Each entry is a claim with exact numbers, an evidence class, a way to
reproduce it, and a pointer to the full record.

`ROADMAP.md` is the lab notebook: every attempt, dated, in full. It lives on the
`dev` branch only; the `Ref.` lines below cite its sections. This file is the
distilled result set. It exists so that design decisions can be reasoned about,
and new reports written, without mining a 37k-line notebook.

---

## How to maintain this file (read before adding anything)

These instructions are binding for every future session, human or agent.

1. **When to add.** Whenever a Findings block is written into `ROADMAP.md` (the
   orchestrating session does this after reviewing an agent's results), decide
   whether it contains a result worth keeping here. If it does, add or update
   the entry **in the same commit**. Also add results that surface as side
   notes while testing, e.g. "metric X is confounded by Y" or "model Z does
   something odd". Background agents report; only the orchestrating session
   edits this file.
2. **What qualifies.** Major or interesting results that stand on their own for
   an outside reader, whether or not they are novel:
   - a fact about a model;
   - a cross-model comparison;
   - a method that works;
   - a method that does not work, with the measurement that killed it.

   **Confirmed negatives are first-class entries** (for example "the
   crosscoder does not beat independent SAEs" or "activation-only feature
   detection is confounded"). Do **not** add process history, plumbing bugs
   without an external lesson, or intermediate numbers that a later entry
   supersedes.
3. **Entry format.** Every entry has these fields:
   - **ID**: a section prefix plus a number, never reused. Prefixes:
     `SH` shared vs unique, `CA` causal comparison, `DE` depth/where-when,
     `PM` per-model, `MP` method positive, `MN` method negative,
     `BM` benchmark.
   - **Claim**: exact numbers, quoted as recorded. Do not round numbers that
     are quoted in `ROADMAP.md`.
   - **Models / layers.**
   - **Evidence**: the class, one of geometric (L1), linear-translatable
     (L2), causal within-model (L3), correlational, descriptive, behavioral,
     illustrative, or held-out confirmed. Never let a correlational number
     read as causal.
   - **Status**: exploratory (dev) / replicated / held-out confirmed /
     negative (closed) / superseded / open.
   - **Reproduce**: run directory, artifact, driver or config, and commit
     where known.
   - **Ref**: the `ROADMAP.md` § (and approximate line) or `CLAUDE_FULL.md`
     §11.N.
   - **Score**: **N/5** plus one line on *why* it is interesting.
     - 5 = answers a research question or decides a design, and is useful
       outside this repo.
     - 4 = strong result or general lesson.
     - 3 = solid, narrower result.
     - 2 = niche but worth knowing.
     - 1 = kept for completeness.
4. **Corrections.** When a later measurement changes an entry, **correct it in
   place**. Change its Status to `superseded` or amend the claim, and add a
   dated `Correction:` line naming what changed and why. Never delete an entry,
   and never silently rewrite a number.
5. **Proof.** Every number must be traceable to a run artifact or a
   `ROADMAP.md` Findings block. Before quoting a number from memory, grep the
   artifact. `runs/` is gitignored, so name the run directory **and** the
   `ROADMAP.md` § that records the number.
6. **Scope notes.**
   - Everything is on the dev corpus unless it says *held-out confirmed*.
   - "TimesFM" means TimesFM 2.5-200M.
   - Chronos-T5 claims are about its **encoder only**, since the decoder is
     not captured.
   - Chronos-2 claims are about its **TIME axis only**; GROUP attention is out
     of scope.
   - Depth claims carry the coverage caveat (DE-04, DE-05).
7. **Keep the headline table (§A) in sync** when a 5/5 entry is added or
   changes status.

---

## A. Headline results at a glance

| ID | One line | Score |
|---|---|---|
| SH-01 | Five architecturally unrelated TSFMs share above-null representational geometry (peak CKA 0.38–0.47 vs nulls 0.02–0.16) | 5 |
| SH-05 | Geometric similarity and data-organization similarity (CKA vs clustering AMI) rank model pairs differently, replicated on 3 panels | 5 |
| SH-06 | Pair similarity splits into a geometric camp (Chronos-2/Bolt) and a behavioral/causal camp (TimesFM/Chronos-2); Kendall's W 0.27 over 7 metrics | 5 |
| SH-10 | Joint crosscoder loses to independent post-hoc-matched SAEs on its pre-registered rule (0.2787 vs 0.3204) | 5 |
| SH-14 | Cross-model concept transfer works when matched on shared **inputs** (not causal fingerprints); ~46–56% of concepts transfer | 5 |
| SH-15 | Sharing a causal-effect profile goes with *less* shared input selectivity (OR 0.345, p 0.027) | 5 |
| SH-16 | No concept is shared by all 4 models; the multi-model pattern is **convergent** effect from different inputs. **Under the matched null (MN-29) the convergent class vanishes (13 → 0 on v2); superseded pending the power check** | 5 |
| MN-30 | The L5 'same causal effect' rung is specific (0/9 decoys) but insensitive on a known answer (0.4 / 0.0 on planted shared concepts across architectures); cross-architecture absences are uninformative |  5 |
| MN-32 | A subspace redesign of the shared-input agreement test fails its pre-fixed gate (sens 0.5 / 0.375, 0/14 false agree); even the planted distributed subspace does not clear a size-matched null, so per-series cross-architecture effect agreement is not measurable at these effect sizes |  4 |
| MN-33 | Family-presence claim fails its pre-fixed known-answer gate (best held-out sensitivity 0.42, 0 false claims); universal effect families stay descriptive, not registered |  3 |
| SH-22 | At 7 models under the matched null, the atlas is **segregated by model** (cross-model mixing below chance, p 0.025). The full same-effect verdict (7/1290) is one Chronos-2 ↔ Chronos-Bolt concept, but 1053 tests are underpowered and partial agreement crosses families (level only 35, shape only 9); the strict rung's sensitivity is unmeasured | 5 |
| SH-18 | Held-out: 19/20 searched and **20/20 frozen-feature** concept-transfer claims confirm on fresh sealed private epochs; **20/20 again at 7 models on a fourth private draw, and on V3 private + 800 raw real windows (20/20 each)** | 5 |
| CA-11 | Held-out (K2, 7 models): 10/31 single-feature causal claims confirm; **0** cross-model causal claims do (atlas 0/3 multi-model, L5 0/6); models share what they select, not demonstrably what they do | 5 |
| SH-19 | Unit-level correspondence (CCM) NO-GO: low-rank TSFM representations defeat the rotation control | 5 |
| SH-24 | Input control for CONCAT geometry (v3 dev): residualizing each layer on its window's 32 z-scored input values keeps **all 21 pairs above a residual null**, but median peak CKA falls 0.6206 → 0.3993; the 0.93–0.94 Chronos-2/Sundial/Bolt peaks are first-layer input encoding in part | 5 |
| CA-06 | Measured on shared series with matched floors, cross-model causal "disagreement" shrinks from 18/19 to 12/288 (12 of 89 scorable) | 4 |
| CA-03 | Seasonality circuit: TimesFM uses 1 head, Chronos-T5-Base a non-additive 5-head circuit | 5 |
| DE-03 | Chronos-2's skip lens is a silent no-op (the head reads only forecast placeholders), so the flat curve means nothing | 5 |
| DE-04 | Cross-model depth axes do not align: Chronos-T5's encoder is only 0.478 of its stack | 5 |
| DE-07 | 5-rung Chronos-T5 scaling ladder: only decodability scales monotonically | 5 |
| PM-08 | `random_init` twins are not one null: TimesFM's is an identity stack, Chronos-2/Bolt's are numerically dead | 5 |
| MN-01 | CKA/L2 cannot detect lineage: an untrained same-architecture pair looks *more* similar than a real related pair | 5 |
| MN-05 | 93% of SAE "top feature" headlines were provenance/residualization artifacts | 5 |
| MN-06 | A grounded small-LLM narrator keeps fabricating comparisons; blacklist guards do not converge | 5 |
| MN-08 | Injection-space SAE "roles" have negative silhouette in ablation space at 13/13 targets | 5 |
| MN-17 | Generator-counterfactual concept mediation NO-GO twice (0/98, 0/151 survive BH) | 5 |
| MP-01 | Token→time spans can be *measured* (IoU 1.0 on 6 checkpoints), enabling a zero-code adapter | 5 |
| MP-04 | A single-seed SAE "gold ranking" flips a bake-off verdict; averaging 3 seeds fixes it | 5 |
| MP-05 | SAE feature death is a property of the layer's geometry (alive atoms ∝ effective dim, exponent 1.02) | 5 |
| MN-21 | Reviving dead SAE features *hurts* causal alignment at 5/5 TimesFM depths | 5 |
| MN-29 | The battery's legacy null is lenient: known-answer gate fails on 5/5 seeds (FPR 0.17–0.25), passes on 5/5 with a profile-matched null; on v2, causal features 293 → 110 (13.0× → 3.06× empirical chance), per-target concepts 9 → 0, convergent atlas concepts 13 → 0 | 5 |
| MN-38 | Mean versus concat changes raw TimesFM-pair CKA by 0.30–0.43 (held-out); concat preserves order, while preferred measurement depends on window, rank and task (scope correction: MP-12) | 5 |

---

## B. Shared vs unique representations

### B.1 Geometry and linear translatability

#### SH-01 · Above-null shared geometry across five unrelated architectures
- **Claim.** Peak linear CKA, each measured against its own run's
  shuffled-series null:

  | Pair | Peak CKA [CI] | Null |
  |---|---|---|
  | TimesFM ↔ Chronos-T5-Base | 0.381 | — |
  | TimesFM ↔ Chronos-2 | 0.43 [0.42, 0.46] | ≈0.04 |
  | TimesFM ↔ Sundial | 0.381 [0.359, 0.410] | 0.022 |
  | TimesFM ↔ Timer (zero-code `generic_hf`) | 0.4494054317474365 [0.4112, 0.5066] | 0.0630 |
  | Lag-Llama ↔ Chronos-T5-Small | 0.4681587815284729 [0.4240, 0.5350] | 0.1565 [0.1437, 0.1705] |

  Lag-Llama is an 8-block RoPE decoder over 92 lag/calendar features, and
  its pair peaks at `transformer.h.5` ↔ `encoder.block.4`.
- **Models.** TimesFM, Chronos-T5-Base/Small, Chronos-2, Sundial, Timer, Lag-Llama.
- **Evidence.** Geometric (L1). **Status:** exploratory (dev); each result is a single corpus and checkpoint pair.
- **Reproduce.** `configs/{chronos2_phase4_check,sundial_phase4_check,generic_hf_timer,lag_llama_vs_chronos}.yaml` → `runs/*/l1/meta.json`.
- **Pooling caveat (2026-10-14, MN-38).** These peaks use window-mean pooling, which understates TimesFM's similarity to 16-step and 1-step models. On 512 v3 dev series with lossless window concatenation, TimesFM's peaks are 0.3866–0.7348 (vs 0.3137–0.4967 under the mean on the same series): Chronos-2 0.7348, Chronos-Bolt 0.6543, Chronos-T5-Base 0.6428, Sundial 0.6363. Above-null sharing holds and is stronger; the levels and pair ranking above are properties of the mean statistic.
- **Update (2026-10-17, ROADMAP §47.1, §47.3).** With concatenated windows and a 1000-permutation null, all 21 pairs of the
  7-model panel exceed their nulls on v2 dev and v3 dev (512 series each), and on a fresh sealed v3 private epoch
  (registered, run once; 21/21, Holm 0.02097902097902098). TimesFM's peaks: v2 0.3575086295604706–0.6353724598884583, v3
  0.37477120757102966–0.7119318842887878; TimesFM/Chronos-2 private 0.7070513688986635.
- **Ref.** ROADMAP §9 (~7500–7650), §16 E3(d) (~11160), §34 D2 (~33030). §44.5.
- **Score. 5/5.** This is the central evidence that TSFMs converge on a common representational geometry, and it generalizes to a model integrated with zero adapter code. Read it with SH-05, SH-06 and MN-01: geometry alone is not evidence of shared *function* or lineage.

#### SH-02 · Above-null shared structure holds at matched depth, not in "best pair vs best pair" comparisons
- **Claim.**
  - Comparing the real run's global best pair with the untrained-twin run's
    global best pair made L1/L2 look null: p=0.474 and p=0.532; L2
    TimesFM→Chronos 0.318 real vs 0.388 null.
  - At the **matched layer index**, the L2 null floor is 0.062/0.046 and the
    real gain beats both by +0.351/+0.367 (p=0.002 each), at 8 of 12 Chronos
    blocks.
  - L1 beats both architecture-only floors at every middle depth and loses
    only at the extremes.
- **Models.** TimesFM ↔ Chronos-T5-Base, with random-init twins.
- **Evidence.** Geometric / linear-translatable, null-corrected. **Status:** corrected in place (2026-08-07).
- **Reproduce.** `analysis/null_baseline.py`; `configs/null_timesfm_random.yaml`, `null_chronos_random.yaml`.
- **Ref.** ROADMAP §13 (~8409–8430, ~8654–8667); §16 E9.
- **Score. 4/5.** A general design rule for null baselines: compare at matched depth and read the whole curve; a peak-vs-peak comparison can manufacture a false null.

#### SH-03 · Stitching is asymmetric: reading *into* TimesFM is easier than reading out of it
- **Claim.** L2 best gain over the input-feature baseline:

  | Pair | → direction | ← direction |
  |---|---|---|
  | TimesFM ↔ Chronos-T5-Base | 0.3179 | 0.4132 |
  | TimesFM ↔ Chronos-2 | 0.3827 | 0.4800 |
  | Chronos-T5-Base ↔ Chronos-2 (within family) | 0.4490 | 0.4526 |

  The within-family pair is the only near-symmetric one. The pattern
  reproduces on `full_report_run_large` (0.3614/0.4570, 0.5035/0.5164,
  0.4929/0.4981).
- **Evidence.** Linear-translatable (L2). **Status:** replicated on 2 corpora.
- **Reproduce.** `runs/full_report_run_3model/l2/stitching.json`, `runs/full_report_run_large/l2/stitching.json`.
- **Ref.** ROADMAP §24.5 (~18781–18787, ~19421).
- **Score. 3/5.** A reproducible directional asymmetry that has not been explained mechanistically.

#### SH-04 · Doubling Chronos's encoder depth: behavior is flat, and two "shared structure" metrics move in opposite directions
- **Claim.** Going from Chronos-T5-Small to Base against a fixed TimesFM:
  - MASE barely moves (2.365→2.355).
  - The pre-registered "TimesFM better on `random_parametric`" result
    **replicates on the private split** at both sizes (gap 0.454/0.455,
    p=0.002).
  - Peak CKA rises 0.276→0.381, and L2 gain rises in both directions
    (0.287→0.318; 0.335→0.413).
  - Clustering AMI *falls*, 0.705→0.538.
- **Evidence.** Behavioral (held-out confirmed for MASE) / geometric / descriptive. **Status:** MASE confirmed; the rest exploratory.
- **Reproduce.** `configs/medium_run.yaml` vs `configs/medium_run_chronos_base.yaml`.
- **Ref.** ROADMAP §5.5 (~2117–2197).
- **Score. 4/5.** A first demonstration that geometric and clustering similarity can disagree on the *direction* of change; DE-07 shows the trend is not monotone at 5 sizes.

#### SH-05 · Geometric similarity (CKA) and data-organization similarity (clustering AMI) rank pairs differently, on 3 panels
- **Claim.** In each panel the closest pair by CKA is not the closest pair by AMI:

  | Panel | Pair | CKA | AMI |
  |---|---|---|---|
  | 3-model (`full_report_run_3model`) | Chronos-T5-Base × Chronos-2 | **0.7400** (highest) | 0.5628 (lowest) |
  | | TimesFM × Chronos-2 | 0.4297 | 0.6057 |
  | `full_report_run_large` | Chronos pair | 0.6975 (closest) | 0.4977 [0.4804, 0.5200] (lowest, non-overlapping CIs) |
  | | TimesFM × Chronos-2 | — | 0.6484 [0.6224, 0.6967] |
  | 4-model | Chronos-2 × Chronos-Bolt | **0.8830** (null 0.002) | 0.6203 (not the highest) |
  | | TimesFM × Chronos-Bolt | 0.3983 | 0.6497 |
  | | TimesFM × Chronos-2 | 0.4362 | 0.6484 |

- **Evidence.** Geometric vs descriptive (two classes on purpose). **Status:** replicated on 3 panels; not tested on private data.
- **Reproduce.** `runs/{full_report_run_3model,full_report_run_large,full_report_run_4model}/l1/meta.json`, `l4/`/`clustering/comparison.json`.
- **Ref.** ROADMAP §24.5 (~18774–18799, ~19416–19433), §26 F (~23422–23454).
- **Score. 5/5.** A single similarity number would tell the model-comparison story backwards; "similar geometry" and "organizes data alike" are separate axes.

#### SH-06 · Pair similarity splits into a geometric camp and a behavioral/causal camp (7 metrics, never pooled)
- **Claim.** On the 4-model panel, two pairs top different metrics.

  **TimesFM/Chronos-2 is closest on behavior and causal-effect metrics:**
  - error agreement 0.9508137661557489;
  - atlas causal co-membership 0.9504574076274207;
  - 3 shared concepts.

  **Chronos-2/Chronos-Bolt is closest on geometry and input metrics:**
  - CKA 0.883013129234314;
  - stitching 0.7092976272106171;
  - input-transfer R_rel 0.8546867367576367.

  **Yet Chronos-2/Chronos-Bolt ranks last** on error agreement
  (0.7200189762792106) and co-membership (0.47258979206049145).

  **Consensus across metrics is weak:**
  - Kendall's W over all 7 metrics is 0.26866549088771313 (p 0.0765).
  - Over representation-only metrics it is 0.5301587301587302 (p 0.1449).

  **The split is not a difficulty artifact.** Controlling for
  seasonal-naive difficulty keeps TimesFM/Chronos-2 error agreement at
  0.8918, while every Bolt pair drops to 0.32–0.49.
- **Evidence.** Geometric + correlational, rank-consensus only. **Status:** exploratory (dev).
- **Reproduce.** `runs/full_report_run_4model/report/model_similarity.json` (`analysis/model_similarity.py`); `report/findings.json` claim `compare.2`.
- **Ref.** ROADMAP §37.11 P8 Q3 (~36955–36972).
- **Score. 5/5.** Representation geometry tracks architecture family, while causal behavior tracks how a model errs. "Which model is most like which" depends on the metric, so no single similarity score should be reported.

#### SH-07 · The seasonality-heavy vs autocorrelation-heavy structural correlates differ by architecture
- **Claim.** After excluding provenance-confounded fields:
  - Every headline structural correlate is a seasonality or autocorrelation
    property.
  - TimesFM's matched features shift with depth from *which* period
    (`seasonal_period_dominant` 29→25→20→15→8 over blocks 2/6/10/16/18) to
    *how many* (`n_seasonalities` 10→10→24→29→33).
  - Chronos-T5-Base crosses more weakly.
  - Chronos-2 never crosses: `ar_coeff_sum` leads at all 3 layers (17/19/19).
- **Evidence.** Correlational (post-residualization ground-truth alignment). **Status:** exploratory; qualitatively reproduced on the 4-model panel.
- **Reproduce.** `runs/full_report_run_large/sae/meta.json`.
- **Ref.** ROADMAP §26 D1 (~22488–22499), §26 E (~23086–23098).
- **Score. 3/5.** A per-model "what the features track" profile; correlational only.

#### SH-24 · Shared geometry survives an input control, but the early-layer peaks are partly shared input encoding
- **Claim.** On the v3 CONCAT rows (`runs/panel7_v3_concat`, 512 series × 15 windows, recorded CONCAT peak layer pair per model pair; known-answer reproduction of all 21 `concat_peak` values, max |Δ| 2.618150207744918e-08):
  - **Input CKA by depth** (input = the window's 32 values z-scored by the 480-step context mean/std, CONCAT of 32 scalar positions): TimesFM 0.531 at `stacked_xf.0` falling to 0.051 at `stacked_xf.18`; Chronos-2 0.894 (block 0) → 0.243 (block 11); Sundial 0.840 (layer 0), minimum 0.099 (layer 6); Chronos-Bolt 0.913 (block 0), minimum 0.496 (block 5); Timer 0.268 (layer 0), minimum 0.094; Time-MoE 0.118 (layer 0), maximum 0.436 (layer 10); Chronos-T5-Base 0.429 (block 0), maximum 0.686 (block 5).
  - **Input-residualized CKA** (per-position ridge on the 32 input values, 5-fold cross-fitted by series, λ = 1e-3·n_train; null = 200 whole-series permutations): **21/21 pairs above the residual null p95** (p 0.004975124378109453 each, the floor). Median peak CKA 0.6206148514354821 → 0.39933235386171145 (median retained fraction 0.7420791857700478; residual range 0.09517507196642126–0.8196844306110395).
  - Pairs with Chronos-T5-Base or with TimesFM's first layer retain 0.374–0.631 of their CKA (e.g. TimesFM/Chronos-2 0.7119319005414653 → 0.3463902538665664; Sundial/Chronos-T5-Base 0.6727 → 0.2513). The first-layer Chronos-2/Sundial/Chronos-Bolt peaks (0.9259–0.9422; input CKA of both ends 0.84–0.91) fall to 0.6962–0.8197. Seven pairs, all with Timer or Time-MoE, retain ≥ 0.957 (Sundial/Time-MoE 0.7404 → 0.7413).
  - Raw (unnormalized) input is not an informative control: CKA(raw input, layer) ≤ 0.025 everywhere because a few high-variance series dominate the Gram.
- **Reading.** Cross-model geometric similarity is not only a re-encoding of the input, but its size is: the headline 0.93–0.94 peaks sit at first layers and are partly shared input encoding. The control is linear and uses only the window's own values, so residual CKA is an upper bound on input-free sharing. The peaks were selected with the input present (selected-peak null, as in L1).
- **Models / layers.** All 7; the recorded v3 CONCAT peak layer pair per pair.
- **Evidence.** Geometric (L1), exploratory (dev). **Status:** exploratory.
- **Reproduce.** `tsfm_lens/run_input_control_cka.py --config configs/panel7_v3_concat.yaml` (commit ed33275) → `/common/users/djk255/tsfm_runs/input_control_cka/input_control.json` (copy in `paper/findings/figures/data/`).
- **Ref.** ROADMAP §47.6.
- **Score. 5/5.** Answers the first question a reviewer asks of any cross-model CKA claim, and corrects the size of the headline geometry result.

### B.2 Features, dictionaries and concepts

#### SH-10 · The joint crosscoder loses to independently trained, post-hoc-matched SAEs (pre-registered)
- **Claim.**
  - On the pre-registered scorecard (ground-truth alignment of shared atoms),
    the joint crosscoder V1 (dict 1024, k48) scores **0.2787** vs V0
    (independent per-model SAEs, matched post hoc) **0.3204**: diff −0.0417,
    CI [−0.0693, −0.0137], p=0.002, 46,382 rows.
  - V2 (BatchTopK) also loses, at all 3 seeds.
  - Pipeline wiring stays gated shut.
- **Models.** TimesFM `stacked_xf.4` ↔ Chronos-T5-Base `encoder.block.10` (the peak-CKA pair).
- **Evidence.** Correlational, pre-registered decision rule. **Status:** negative (closed) 2026-08-18.
- **Reproduce.** `sae/crosscoder.py`, `sae/crosscoder_eval.py::score_variant`, `run_crosscoder_ladder.py`, `configs/crosscoder_stage0_gate.yaml`.
- **Ref.** ROADMAP §6.2.1 (~4073–4121, Stage 4 ~6125–6239); §22.1.
- **Score. 5/5.** Decides a design question with value outside this repo: compare models through independent dictionaries plus post-hoc matching, not joint training.

#### SH-11 · The crosscoder's "shared fraction" is meaningless without a same-seed untrained-twin floor, and the floor itself is seed-fragile
- **Claim.**
  - At 4608 rows, the real pair's `frac_shared` of **0.8453** is *below*
    TimesFM-vs-its-own-random-twin at **0.9738**: diff −0.1285, CI
    [−0.1601, −0.1007].
  - At 46,382 rows over 3 seeds, the real pair's value is flat (0.749–0.777)
    while the twin floor swings 0.696–0.856. The real pair clears at seed 0
    (+0.081), ties at seed 1 and reverses at seed 2 (−0.107).
  - β-confirmation separately shows that most "model-specific" atoms are
    shrunk shared atoms: 77.70% → **93.5%** shared.
- **Evidence.** Correlational vs architecture-only null. **Status:** negative (closed).
- **Reproduce.** `runs/medium_run_chronos_base/crosscoder/ladder.json`; Stage 4 sweep.
- **Ref.** ROADMAP §6.2.1 Stage 1 (~5234–5306), Stage 2/4 (~5960–5974, ~6134–6219).
- **Score. 5/5.** A general rule for any cross-model "shared dictionary" metric: quote it only beside an architecture-matched null measured at more than one seed.

#### SH-12 · Post-hoc matching of independent SAE dictionaries finds real matched pairs
- **Claim.**
  - The matcher combines activation-profile correlation, series-level
    Jaccard and ground-truth agreement.
  - It pairs 40 of TimesFM's 50 ground-truth-matched candidate features with
    a Chronos-T5-Base partner; 10 are unmatched.
  - The best pair is TimesFM f2872 ↔ Chronos f2312, both named
    `archetype_trend_dominant` (ρ 0.648/0.527), with profile correlation
    0.831 and Jaccard 0.818.
- **Evidence.** Correlational. **Status:** exploratory. Note that the field is an archetype dummy, which MN-05 later showed to be a confound class.
- **Reproduce.** `sae/matching.py::match_cross_model_features`; `runs/medium_run_chronos_base_feature_ablation_check/sae/`.
- **Ref.** ROADMAP §16 E16 (~10358–10443).
- **Score. 3/5.** A working, cheap alternative to the crosscoder, though matched on a field later shown to be provenance-heavy.

#### SH-13 · Cross-model SAE *role* matching does not beat its nulls
- **Claim.**
  - TimesFM `stacked_xf.18` × Chronos-T5-Base `encoder.block.10`: match
    rate 0.8333333333333334 against untrained-twin floors of exactly 1.0 for
    both models. This is the same shape as SH-11.
  - With a population null on `full_report_run_large_revived`, the legacy
    fixed-threshold 100% match rate becomes 0/7, 1/7 and 0/7 across the three
    pairs.
  - Only 1 of 21 matches clears: TimesFM ↔ Chronos-2 "far-horizon disperser",
    cosine 0.951 vs p95 0.925.
- **Evidence.** Correlational, null-calibrated. **Status:** negative (closed); superseded by input-space transfer (SH-14).
- **Reproduce.** `sae/role_matching.py::role_correspondence_table`, `permutation_null_cosine`; `runs/full_report_run_large(_revived)/sae/roles.json`; `runs/null_*_random`.
- **Ref.** ROADMAP §25.25 (~21604–21721), §26 D1 (~22838–22916).
- **Score. 4/5.** The second independent mechanism where apparent SAE sharing fails an architecture-only null.

#### SH-14 · Concept transfer works when matched on shared *inputs*, against a stratum-matched null
- **Claim.**
  - The test asks whether some feature in the other model separates a
    concept's top-k firing series, against random series subsets matched on
    archetype/family strata.
  - A *uniform* null was tried first and gave a false 30/30.
  - On 30 concepts × 284 concept-destination pairs: forward 160/284
    (**56.34%**), reciprocal 131/284 (**46.13%**).
  - 11/30 concepts reach all 3 other models, and 8/30 reach none.
  - The lowest pair is Chronos-2×Sundial (31.2%), which agrees with their
    opposite L3 specializations and low CKA. The highest is
    Chronos-Bolt×TimesFM (78.6%).
  - At atlas level with BH per ordered pair and leg: 511 tests, 316
    uncorrected reciprocal, **288 FDR-reciprocal** (float64 ranks).
- **Evidence.** Correlational (input agreement). **Status:** exploratory; byte-reproducible at a fixed seed. Held-out confirmation is in SH-18.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** Atlas-transfer tests fall from 1581 to 431, and reciprocal-FDR transfers from 1223 to 372 (86% of tests reciprocal in both runs). Input-level transfer of the surviving causal features holds at the same rate; the base is smaller.
- **Reproduce.** `sae/transfer.py::run_transfer`; `runs/full_report_run_4model/sae/transfer.json`, `sae/atlas_transfer.json`.
- **Ref.** ROADMAP §30.2 (~25702–25884), §37.6 (~36018–36025).
- **Score. 5/5.** Establishes the method: match concepts on what features respond to, and treat the causal fingerprint only as a description (see MN-09).

#### SH-15 · Sharing a causal-effect profile goes with *less*, not more, shared input selectivity
- **Claim.**
  - Over 120 (concept, source, destination) triples, input transfer (FDR)
    succeeds in **38/56 = 0.679** of triples where the destination holds an
    effect-space atlas member.
  - It succeeds in **55/64 = 0.859** where the destination does not.
  - Odds ratio 0.345, Fisher two-sided p **0.0275**. With float16 ranks it
    read OR 0.487, p 0.097.
- **Evidence.** Correlational. **Status:** exploratory (dev).
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** Not recomputed directly. The profile null leaves 3 'shared' and 0 'convergent' atlas concepts, too few to repeat the OR test (legacy v2 base: 27 concepts).
- **Reproduce.** `sae/atlas_transfer.json` × `sae/concept_atlas.json`, `runs/full_report_run_4model`.
- **Ref.** ROADMAP §37.6 (~36026–36044).
- **Score. 5/5.** "Same effect on the forecast" and "responds to the same inputs" are different claims and can anti-correlate. This drives the sharing taxonomy in SH-16.

#### SH-16 · No concept is shared by all four models; multi-model concepts are mostly *convergent*
- **Claim.**
  - No atlas concept has causal members in all 4 models:
    `compare.1` "0 of 19 shared by all 4".
  - Sharing classes over 19 concepts: **convergent 7** (same effect,
    different inputs), shared 4, single-model 5, partially shared 3.
  - Restricted to seed-stable concepts: convergent 7, shared 3, single-model
    3, partially shared 1.
  - Every stable 3-model concept is convergent.
  - The atlas has 19 concepts from 201 pooled causal features (63 assigned;
    min_cosine 0.9, complete linkage, ≥3 members).
- **Evidence.** Descriptive taxonomy over causal (ablation) and correlational (top-k overlap, hypergeometric + BH) parts. **Status:** exploratory (dev).
- **Replication (2026-09-29).** `runs/concept_atlas_v2` (v2 corpus, fixed Sundial; ROADMAP §37.11c), 29 targets, 27 tight concepts: again no 4-model concept (n_models 1:11, 2:15, 3:1, 4:0); sharing classes convergent 13, single-model 11, shared 2, partially shared 1 (stable only: 9 / 10 / 2 / 1). Convergent stays the most common multi-model class. Still dev.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** The v2 atlas shrinks from 27 concepts (16 multi-model, from 293 pooled causal features) to **7 (3 multi-model, from 110)**. Sharing classes: convergent 13 → **0**, single-model 11 → 4, shared 2 → 3, partially shared 1 → 0. **The 'mostly convergent' headline does not survive the matched null**: the convergent class was built from features the lenient null admitted. 'No concept spans all four models' still holds, but on a far smaller base. Status of this entry: superseded pending the power check and the 7-model run.
- **7-model panel (2026-10-01, `runs/panel7_v2_dev`, 7 models, profile-matched null, ROADMAP §38.3.4).** 11 atlas concepts from 194 pooled causal features, 9 multi-model; **none spans more than 3 of the 7 models** (models per concept 1:2, 2:4, 3:5). Sharing classes convergent 5, shared 2, partially shared 2, single-model 2 (stable only 4 / 2 / 1 / 2). But cross-model mixing is *below* chance (SH-22), so 'convergent' is no more common than random assignment would produce. 'No concept spans the whole panel' holds; 'mostly convergent' is not restored.
- **Held-out (2026-10-02, K2, CA-11).** Registered claim `no_concept_in_all_models` **confirmed** on private v2 epoch 3: the private atlas has 10 concepts, at most 3 of 7 models per concept. 'No concept spans the panel' is now held-out confirmed (structural). 'Mostly convergent' was deliberately not registered (low-powered, no private counterpart).
- **Reproduce.** `runs/full_report_run_4model/sae/concept_profiles.json`, `sae/concept_atlas.json`; `report/findings.json` `compare.1`.
- **Ref.** ROADMAP §37.3, §37.11 P8 (~36929–36941).
- **Score. 5/5.** The headline answer to the founding question on this panel: TSFMs converge on shared *forecast adjustments* far more than on shared *feature detectors*.

#### SH-17 · Seed-stability ceiling: 14/19 atlas concepts reproduce; every model pair sits below its within-model ceiling
- **Claim.**
  - With 3 replicate SAEs per target (GPU), 14/19 = 0.737 concepts are
    stable, clearing the pre-registered 25% bar.
  - Within-model ceilings: Chronos-2 0.964, Chronos-Bolt 1.0, Sundial 0.955,
    TimesFM 0.870.
  - Relative transfer R_rel = R/√(ceil_A·ceil_B) peaks at 0.873
    (Chronos-Bolt→Chronos-2).
  - **Sundial is the weakest source in all 3 of its outbound pairs**
    (0.439–0.698).
- **Evidence.** Descriptive (reproducibility) + correlational. **Status:** exploratory. CPU-trained replicates flip 2 concepts' status (MN-27), so GPU replicates are canonical.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** All 7 profile-null atlas concepts are seed-stable (7/7), against 22/27 under legacy. Stability rises as the set shrinks to its strongest members.
- **7-model panel (2026-10-01, `runs/panel7_v2_dev`, 7 models, profile-matched null, ROADMAP §38.3.4).** 9/11 atlas concepts seed-stable (0.8181818181818182), 3 replicate SAEs per target.
- **Reproduce.** `sae/concept_stability.json` (`concepts.n_sae_seeds: 3`), `runs/full_report_run_4model`.
- **Ref.** ROADMAP §37.5 (~35788–35937).
- **Score. 4/5.** Gives cross-model transfer the "two runs of the same model" denominator that convergent-learning claims need.

#### SH-18 · Held-out confirmation: concept transfer replicates on fresh sealed private epochs, and so do the specific frozen feature pairs
- **Claim.**
  - **P7 (search mode, epoch 1, 961 series).**
    - 20 claims were registered from dev only (top reciprocal-FDR, stable,
      by dev AUC margin).
    - **19/20 confirmed** under Holm (m 20, n_null 2000, minimum Holm p
      0.009995). Private AUC ranged 0.7966–0.9993.
    - The failure was Chronos-Bolt b.4 c6 → TimesFM xf.10: reverse AUC 0.521
      vs p95 0.569, Holm 0.257871.
    - Only **6/20** private best features equal the dev features, so P7
      confirms "the destination layer has *some* selecting feature".
  - **P7b (frozen mode, epoch 2, 963 series).**
    - Dev's destination feature is frozen and scored against a
      single-feature null.
    - **20/20 confirmed**, Holm p 0.009995 each. Forward AUC 0.7228–0.9991;
      reverse 0.8483–0.9990.
    - P7's only failure confirms when frozen: f5278 forward 0.8223 vs p95
      0.6155, reverse 0.9429 vs 0.6474. The search had picked an unstable
      feature, f4664.
  - **Where the claims come from.**
    - Concept 6, "dominant lowers seasonal" (Chronos-2/Bolt/Sundial): 8
      claims. The same destination features recur: TimesFM xf.2 f3290 and
      Sundial l.3 f731.
    - Concept 4 (Chronos-2 + TimesFM, lowers level): 8 claims. Its top
      series are 17–20/20 from the real-derived `mixture` generator in 3/4
      parts, so it is provenance-heavy.
    - Concept 0, "strong raises seasonal": 4 claims. It is labelled
      convergent, yet features in other models select its series.
  - **Replication on a different corpus (2026-09-29).** `runs/concept_atlas_v2` (v2 corpus, fixed Sundial; ROADMAP §37.11c), P7 search mode on the unpeeked v2 private split (800 series, epoch 0, first look): **20/20 confirmed**, Holm max 0.009995, private AUC 0.9708–0.9951; 6/20 private best features equal dev's. Caveat: the 20 claims come from 2 concepts (16 from concept 0, Chronos-2 + TimesFM "strong lowers level"; 4 from concept 9, Chronos-2 + Sundial).
  - **Fourth private draw, 7 models (2026-10-02, K2, v2 epoch 3, 971 series, first look; CA-11).** **20/20 confirmed** in search mode, Holm max 0.009995002498750623. Private AUC 0.9194269190325972–0.9696109358569927, reverse 0.8053101997896951–0.9981598317560463; 8/20 private best features equal dev's. The claims come from TimesFM xf.18 concept 1 (14) and Sundial l.6 concept 3 (6), with destinations in every other family (Timer 6, TimesFM 5, Sundial 3, Chronos-2 2, Chronos-T5-Base 2, Time-MoE 1, Chronos-Bolt 1). Reproduce: `runs/panel7_v2_dev/confirm/confirmation.json` (`concept_replication`), private v2 epoch 3 (971 series), registry sha256 `ff6c4f9e…`, k3wt 597848a.
  - **V3 (2026-10-10, CA-12).** `panel7_v3_dev` on `benchmark_v3/private_epoch1` (1024 series, first look): **20/20 confirmed** (Holm max 0.009995002498750623; private AUC 0.843–0.99); on 800 raw real windows (external leg, overlap 0/0, not counted in the verdict) **20/20**, AUC 0.757–0.991. Sources: Sundial 14, Chronos-2 5, Time-MoE 1.
  - **Other held-out results from the same epoch-1 confirmation.** Peak-CKA
    pair CKA 0.44 [0.43, 0.45]; 7/9 L3 fingerprint agreements (overall ρ
    0.50); Chronos-2 > others on `mixture` (ΔMASE 0.2222, p 0.0005).
- **Evidence.** **Held-out confirmed** (private split) for a *correlational* claim: the same series are grouped by both models. This is not causal. **Status:** confirmed.
- **Reproduce.** `runs/full_report_run_4model_epoch1/confirm/confirmation.json` (`concept_replication`; ids repaired offline, originals in `*.pre_id_fix.json`), `runs/full_report_run_4model_epoch2/confirm/confirmation.json`; `analysis/confirm.py::_replicate_registered_concepts`, `sae/transfer.py::transfer_one_fixed_feature`; `concepts.transfer_claim_mode`; commits `b812a2a`, `411ea25`.
- **Ref.** ROADMAP §37.10 P7 (~36775–36844), P7b (~36845–36885).
- **Score. 5/5.** The only concept-level claims in the project that clear the gold-standard bar. Specific feature pairs in different architectures select the same unseen series.

#### SH-19 · Unit-level correspondence (Convergent Core Mass) closes NO-GO: TSFM representations are too low-rank for rotation controls
- **Claim.**
  - Uncontrolled neuron-level CCM looked strongly positive: 0.24–0.54, with
    every CI excluding 0. Examples: TimesFM/Chronos-2 0.2471 at CKA 0.436;
    Chronos-2/Bolt 0.5396 at CKA 0.883.
  - A Haar rotation of *real* activations barely moves it (one basis) or
    *raises* it (both bases):

    | Level / pair | Real | B rotated | Both rotated |
    |---|---|---|---|
    | N: TimesFM xf4 / C2 b6 | 0.2380 | 0.2131 | 0.2614 |
    | N: C2 b0 / Bolt b0 | 0.5311 | 0.5082 | 0.6131 |
    | F: C2 b6 / Bolt b4 | 0.1476 | 0.1282 | 0.3313 |
    | F: Sundial l3 / TimesFM xf18 | 0.1471 | 0.1228 | 0.2947 |

  - SAE atoms are no more one-to-one matched across models than random
    directions are.
  - Cause: participation ratio is as low as 1.54, and the top 10 PCs carry
    37–92% of variance, so every direction correlates with the same dominant
    components.
- **Evidence.** Geometric / method. **Status:** negative (closed) 2026-09-26. Not merged; recovery commit `046b96c`.
- **Reproduce.** Branch `worktree-agent-addcbd551b7d8b676` (`analysis/alignment_spectrum.py`, `run_alignment_spectrum.py`); the rotation control reads the store only.
- **Ref.** ROADMAP §35.14 (~34833–34896).
- **Score. 5/5.** A controlled negative on "do TSFMs learn the same *units*", plus a general trap: rotation-invariance controls assume isotropy, which low-rank representations violate.

#### SH-20 · Per-target concepts are mostly non-modular
- **Claim.**
  - With a minimum cluster size of 3 applied *before* the silhouette
    comparison, 13 targets yield only **4 concepts**, and **11/13 targets
    are non-modular**.
  - The old unconstrained metric's top four silhouettes (0.717, 0.674,
    0.636, 0.607) were all `[big, 1]` splits.
- **Evidence.** Descriptive (clustering over causal vectors). **Status:** confirmed; this motivated pooling features across models into the atlas.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** v2 per-target concepts go from 9 to **0** (28/28 measured targets non-modular). Causal features per target are 0–8; 24/28 have fewer than 6, and the 5 with ≥6 have a min cluster < 3 at every admissible k. About 62% of profile clears are chance-level (110 clearing features vs a 67.88 chance expectation). This is a **low-power result, not a measured absence**: k = 8 rows × 16 null draws. Power check (ROADMAP §38.1.8, round 6): raising (k, n_null) to (16, 32) or (32, 32) does not help. It gives 3 targets with ≥6 causal features, and 0 or 1 targets admitting a k. The binding limit is feature sparsity: 178/672 candidates never fire and are unscorable at any setting.
- **7-model panel (2026-10-01, `runs/panel7_v2_dev`, 7 models, profile-matched null, ROADMAP §38.3.4).** 10 per-target concepts on 5 of 56 targets (Chronos-2 block 9, Sundial layers 5 and 8, Timer layers 3 and 7; 2 each); 50 non-modular. **Timer's 4 concepts sit on targets whose clears are not above empirical chance** (BH q 0.272 and 0.163; Timer overall 1.0169× chance, PM-17), so the concepts step should condition on per-target significance. K2 registers only from the 22 BH-significant targets.
- **Reproduce.** `sae/concepts.py::_sweep_k`; `runs/full_report_run_4model/sae/concepts.json` (`concept_min_members` 1 vs 3).
- **Ref.** ROADMAP §32.1, §32.5 D, §32.13 (~29179–29247).
- **Score. 4/5.** A real result about how modular TSFM causal features are (mostly not, at this dictionary size), plus a general silhouette trap.

#### SH-21 · Causal channel repertoires are nested: models differ in degree, not in kind
- **Claim.**

  | Model | Roles | Roles with causal effect | Channels moved | Dominant channel (null units) |
  |---|---|---|---|---|
  | TimesFM | 32 | 29 | 7 | near-horizon (2.78) |
  | Chronos-2 | 21 | 14 | 5 | near-horizon (0.27) |
  | Sundial | 21 | 18 | 4 | far-horizon (1.39) |
  | Chronos-Bolt | 14 | 8 | 3 | far-horizon (1.69) |

  - All 4 models move `horizon_shape_far`.
  - Only 3 of 9 channels are unique to one model: level→TimesFM,
    flatness→Sundial, spectral_centroid→Chronos-2.
  - TimesFM and Chronos-2 share a dominant channel but differ tenfold in
    strength (2.78 vs 0.27).
- **Evidence.** Causal within-model, aggregated. **Status:** exploratory.
- **Reproduce.** `report/derived.py::sae_causal_repertoire`; `run_sae_compare.py`; `runs/full_report_run_4model`.
- **Ref.** ROADMAP §28.3 (~23993–24023).
- **Score. 4/5.** Recasts "what is each model better at" as how strongly, not which, forecast properties its features control.

---

#### SH-23 · Cross-model results barely move with the alignment window (32, 64, 96 steps)
- **Claim.** `panel7_v3_dev`, re-pooling the stored 32-step windows exactly into 64 and 96 steps (same SAEs, no retraining). Atlas transfer reciprocal-FDR rate 0.8266 / 0.8009 / 0.7987 (467 tests); per-test verdict agreement with the 32-step run 0.9272 (64) and 0.9165 (96). Peak CKA over 21 model pairs changes by a median +0.0127 (64) and +0.0175 (96), range −0.0566 to +0.2247; the peak layer pair is the same at all three windows for 9 of 21 pairs.
- **Reading.** Shared-input transfer and the size of geometric similarity do not depend on the 32-step convention; *which* layer pair peaks often does, so peak-location claims are window-dependent.
- **Evidence.** Descriptive, exploratory (dev). **Status:** open.
- **Reproduce.** `runs/panel7_v3_dev/sae/window_sensitivity.json` (V3-E2, ROADMAP §41.1 "Window sensitivity").
- **Ref.** ROADMAP §41.6.
- **Score. 3/5.** Answers the obvious reviewer question about the window choice.

#### SH-22 · At 7 models, the atlas is segregated by model, and the full same-effect verdict is reached only inside the Chronos family; the strict rung is underpowered
- **Claim.**
  - The pooled atlas has more structure than chance: 11 concepts vs a column-shuffle null p95 of 5.049999999999983 (p 0.004975124378109453).
  - But its clusters are **purer by model than random assignment**: 9 multi-model concepts vs a null mean of 10.755 (p_below 0.024875621890547265); mean model purity 0.5909090909090909 vs 0.4741287878787878 (p_above 0.024875621890547265). Verdict `segregated by model`. The coarser families (9) are also segregated by model and no more numerous than null (p 0.6766169154228856).
  - Shared-input causal agreement over 1290 tests: same causal effect 7, acts differently 22, shape only 9, level only 45, no specific agreement 133, not scorable 1074.
  - **All 7 'same causal effect' verdicts are Chronos-2 ↔ Chronos-Bolt, in one atlas concept** ('mild raises dispersion'). Six of them share the Chronos-Bolt block-4 feature set, so this is one lineage-shared concept, not seven.
- **Correction (2026-10-01, same day): the L5 result is mostly undecided, not a measured difference.** 1053 of the 1074 not-scorable tests fail because at least one side's single-feature ablation does not clear its own null on the ~24 shared series (526 neither side, 434 only the source clears, 93 only the destination clears; 21 have no side record). Across families, 172 tests are scorable: level only 35, shape only 9, acts differently 19, no specific agreement 109, same causal effect 0. Within the Chronos family, 44 of 233 are scorable: same causal effect 7, level only 10, acts differently 3, no specific agreement 24. Partial agreement does cross families. At the profile level, atlas concept 6 ('shifts the level', Chronos-2 / Chronos-Bolt / Sundial) and concept 4 (with Timer) are classed shared. The 'same causal effect' verdict requires level AND shape to beat both matched floors, with single-feature ablations, so its cross-architecture sensitivity is unmeasured. Next: a known-answer test of the rung across two different planted architectures, and whole-concept (feature-set) ablation, before K2 spends the private epoch (ROADMAP §38.3.4).
- **Known-answer check (2026-10-01, MN-30).** On a planted pair of different architectures, the rung finds a truly shared concept in 0.4 (single direction) and 0.0 (distributed) of seeds, with 0/9 false 'same'. **The within-Chronos-only pattern is therefore a sensitivity limit of the rung, not evidence that sharing follows lineage.**
- **Held-out (2026-10-02, K2, CA-11).** No L5 agreement claim confirms (0/6 tested, 5 not testable). The lone Chronos-2 ↔ Chronos-Bolt 'same' claim that was testable reproduces its verdict but misses Holm (p 0.011, p_holm 0.0659). No multi-model atlas concept confirms (0/3; only 1/3 member pairs keep cosine ≥ 0.9). 'No concept in all models' confirms: the private atlas has 10 concepts, at most 3 of 7 models each. Status: the segregation and 'no universal concept' readings are held-out consistent; every specific cross-model causal claim is not confirmed.
- **Models / layers.** TimesFM, Chronos-2, Sundial, Chronos-Bolt, Timer, Time-MoE, Chronos-T5-Base (encoder); 56 targets, 55 measured.
- **Evidence.** Descriptive (atlas structure vs nulls) + causal within-model on shared inputs (L5, each side against its own null and matched floors). **Status:** exploratory (dev); K2 held-out: no cross-model causal claim confirmed (CA-11).
- **Reproduce.** `runs/panel7_v2_dev/sae/concept_atlas.json` (`null`), `sae/shared_input_agreement.json`, `sae/concept_stage.json`; `configs/panel7_v2.yaml` at dev 4ace6a4.
- **Ref.** ROADMAP §38.3.4.
- **Score. 5/5.** Answers the founding sharing question on the widest panel: causal forecast adjustments are mostly model-specific, and the one clear case of the same effect on the same inputs follows model lineage.

#### SH-25 · Both models retain the spectral features at their actual forecast-head inputs despite different composition failures
- **Claim.** A linear spectral probe, fit on 96 old generating seeds with
  regularization chosen on 48 validation seeds, generalizes without refitting
  to 128 fresh two-frequency bundles. Reconstructed future MSE, normalized by
  the same mixture variance as the native forecast, is **0.0005480985483314407**
  [0.00040073265909734875, 0.0007349060599206458] for Bolt and
  **0.0006623374741218504** [0.0004980500019688435, 0.0008819125012721511]
  for TimesFM; native errors are **0.2044275523140093** and
  **0.3512594243442453**. Both period32 and period16 coefficient pairs are
  recoverable. This is extra supervised, task-specific decoding, **not a fair
  zero-shot forecast improvement or a causal repair**; a known-frequency input
  harmonic estimator is essentially exact.
- **Models / layers.** Chronos-Bolt-small `output_patch_embedding` input
  (the actual decoder query); TimesFM 2.5 `output_projection_point` input at
  the last read position. Bolt's decoder attends all encoder states; REG is
  not its sole readout surface. Hooks change native forecasts by exactly zero.
- **Evidence.** Held-out predictive / linear recoverability.
- **Status.** Replicated on fresh synthetic seeds; not sealed-private confirmation.
- **Reproduce.** `experiments/feature_search/head_readout.py`;
  `runs/feature_search_20261111_native_head_readout_fp64/`
  (frozen weights/scaler); source-cutpoint hashes in
  `runs/feature_search_20261111_native_head_capture/`.
- **Ref.** ROADMAP §48.12.
- **Score. 4/5.** Separates retained task features from their native use and
  avoids diagnosing a forecast failure as missing representation.

## C. Causal comparison (within-model causal evidence compared across models)

#### CA-01 · Corruption-sensitivity agreement is pair-specific, not a general TimesFM-vs-rest anti-correlation
- **Claim.** L3 fingerprint agreement between TimesFM and each other model:

  | Pair | Overall ρ | `noise` alone |
  |---|---|---|
  | TimesFM vs Chronos-T5-Base | ≈−0.90 | anti-correlated |
  | TimesFM vs Chronos-2 | −0.75 [−0.76, −0.73] | ρ −0.39 |
  | TimesFM vs Sundial | **+0.517** [0.451, 0.560] | ρ −0.80 |

  `noise` is anti-correlated in every pair. TimesFM and Chronos-T5-Small
  both react *least* to `detrend` of all 9 corruptions (0.151/0.310), even
  though its perturbation (3.38) is not small. That is consistent with both
  forecasting from the recent local level.
- **Evidence.** Causal within-model (fingerprints); correlational (agreement). **Status:** exploratory. Epoch-1 private confirm: TimesFM/Chronos-2 overall ρ 0.50, with 7/9 registered corruptions replicating.
- **Reproduce.** `configs/{chronos2,sundial}_phase4_check.yaml` → `l3/meta.json`; `runs/medium_run/l3/sensitivity.npz`.
- **Ref.** ROADMAP §9 (~7518–7632), §5.5 (~1971–1998).
- **Score. 4/5.** A regularity treated as general turned out to be pair-specific once a third decoder-only model was added.

#### CA-02 · Noise-sensitivity agreement flips sign with dose
- **Claim.** TimesFM vs Chronos-T5-Base depth-profile agreement by noise
  level:

  | SNR | ρ |
  |---|---|
  | 20 dB | +0.397 |
  | 12 dB | −0.439 |
  | 6 dB → −3 dB | −0.852 → −0.764 |

  Each model's peak layer never moves across the sweep: TimesFM's first
  captured layer and Chronos's last.
- **Evidence.** Causal within-model / correlational. **Status:** exploratory.
- **Reproduce.** `run_noise_snr_sweep.py`; `runs/noise_snr_sweep.json`.
- **Ref.** ROADMAP §7 (~7101–7157).
- **Score. 3/5.** A single "these models disagree on noise" number is really a dose-response curve.

#### CA-03 · Seasonality circuit: TimesFM needs 1 head; Chronos-T5-Base needs a non-additive 5-head circuit
- **Claim.**
  - The minimal sufficient and necessary head set for `deseasonalize`
    restoration is **1 head** for TimesFM (p=0.04) and **5 heads** for
    Chronos-T5-Base (p=0.008).
  - Path patching conserves exactly for TimesFM. It does not for Chronos's
    set: mean |gap| 0.0238 vs mean |effect| 0.0108, a genuine nonlinear
    interaction.
- **Evidence.** Causal within-model. **Status:** exploratory (closed 2026-08-23).
- **Reproduce.** `analysis/seasonality_circuit.py`; `runs/medium_run_chronos_base` (isolated copy); report section "Circuit".
- **Ref.** ROADMAP §20 H8 (~14147–14956); §23.3 F1 (~17440).
- **Score. 5/5.** The clearest component-level mechanistic contrast in the project: the same capability is implemented by one head in one model and a distributed interacting circuit in another.

#### CA-04 · Heads that *look* periodic are not the heads that *carry* periodicity
- **Claim.** The rank correlation between attention-mass-at-seasonal-lags and causal seasonal-power loss under single-head ablation is TimesFM ρ=0.026 (p 0.64, n=320) and Chronos-T5-Base ρ=−0.118 (p 0.16, n=144). The top candidates do not overlap.
- **Evidence.** Causal vs descriptive. **Status:** exploratory.
- **Reproduce.** `analysis/seasonality_circuit.py::score_single_head_effects`.
- **Ref.** ROADMAP §20 H8 Stage 1 (~14268–14300).
- **Score. 4/5.** A warning for attention-pattern taxonomies in general: attention mass is not causal responsibility.

#### CA-05 · Steering a ground-truth-matched feature works as predicted in TimesFM and backwards in Chronos-T5-Base, twice
- **Claim.**
  - Steering the best `seasonal_amplitude_max` feature by ±2σ moved the
    forecast's seasonal band in the predicted direction for TimesFM:
    f6148, ρ −0.33; f1481, ρ 0.22.
  - It moved it in the **opposite** direction for Chronos-T5-Base: f5340,
    ρ 0.42; f32, ρ −0.44.
  - Each result held in two independent dictionaries (~97% dead vs ~20%
    dead).
- **Evidence.** Causal within-model (one feature per model per run). **Status:** replicated twice.
- **Reproduce.** `sae/eval.py::feature_steering_effects`; `configs/medium_run_chronos_base_feature_steering_check.yaml`, `configs/sae_revival.yaml`.
- **Ref.** ROADMAP §16 E14 (~10029–10295), §23.2 C1 (~17248–17398).
- **Score. 4/5.** "Correlates with a ground-truth factor" and "causally controls that property" can dissociate, and they do so by model.

#### CA-06 · On shared series with matched floors, cross-model causal "disagreement" mostly disappears
- **Claim.**
  - The old activation-matched role comparison gave 1 same role vs **18
    "fires together, acts differently"**. That included Chronos-2↔Bolt,
    whose identically named roles scored cosine −0.120. Its successor
    finding `sae.6` reads 14 of 15 scorable pairs act differently.
  - P5b ablates both sides on the *same* series U, and each side must clear
    its own null and beat activation-matched random-feature floors.
  - Over 288 reciprocal-FDR tests it finds: same causal effect **6**,
    level only 15, shape only 13, **no specific agreement 43**, **acts
    differently 12**, not scorable 199.
  - **Correction (2026-09-27).** The first published counts were 5 / 15 / 8 /
    32 / 9 / 219. They changed after a seeding fix to the own-effect null for
    sampled models (Sundial). All 177 tests with no Sundial side are
    byte-identical before and after. Of the 89 scorable tests, 12 act
    differently. ROADMAP §37.8.
- **Replication (2026-09-29).** `runs/concept_atlas_v2` (v2 corpus, fixed Sundial; ROADMAP §37.11c), 1223 reciprocal-FDR tests: same causal effect 11, level only 52, shape only 29, no specific agreement 264, acts differently 64, not scorable 803. Of 420 scorable, 64 (15.2%) act differently (reference 12/89 = 13.5%). Runtime 10854.9 s.
- **Evidence.** Causal within-model, compared on shared inputs. **Status:** exploratory; corrected 2026-09-27; replicated on the v2 corpus 2026-09-29.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** v2 agreement verdicts go from same 11 / level-only 52 / shape-only 29 / no specific agreement 264 / acts differently 64 / not scorable 803 (1223 tests) to **7 / 4 / 8 / 28 / 10 / 315 (372 tests)**, with the matched null now also at the agreement step's own-effect gate. 'Acts differently' stays rare (10). Not-scorable rises from 66% to 85%, because fewer sides clear their own matched null on U.
- **Reproduce.** `runs/full_report_run_4model/sae/shared_input_agreement.json` (`sae/shared_input_agreement.py`, commit `fc37ec0`; the pre-fix artifact is `shared_input_agreement.pre_seed_fix.json`).
- **Ref.** ROADMAP §27.3 (~23844–23866), §37.8 P5b (~36309–36447).
- **Score. 4/5.** Most of the "models act differently" headline came from comparing effects on different inputs. Hold the input fixed before comparing causal effects across models.

#### CA-07 · Most causal SAE features change forecast *shape*, not only level
- **Claim.**
  - Per candidate, the median share of the effect that is a level shift is
    **0.535** (n=436), and only 20.6% of candidates exceed 0.9.
  - Per-model medians: TimesFM 0.687, Sundial 0.542, Chronos-2 0.467,
    Chronos-Bolt 0.297.
  - A P4 re-run gives 0.5652.
  - Shape-causal candidates beyond a level-removed null: TimesFM 46/61,
    Chronos-2 54/61, Sundial 52/53, Chronos-Bolt 26/26.
  - The top shape channel differs by model: TimesFM near-horizon, Chronos-2
    dispersion, Sundial far-horizon, Chronos-Bolt MASE.
- **Evidence.** Causal within-model. **Status:** exploratory.
- **Reproduce.** `runs/full_report_run_4model/sae/*_ablation.json` (`shape_channels`, level share).
- **Ref.** ROADMAP §37.3, §37.7 (~36135–36174).
- **Score. 4/5.** Corrects the "a causal fingerprint is one signed scalar" reading of a pooled PC1 (98.3%). A pooled-variance summary and per-item dominance answer different questions.

#### CA-08 · Injection overstates causal role relative to own-site ablation
- **Claim.**
  - Summed over 13 targets: 1394 injection clearing cells vs 942 ablation
    clearing cells.
  - Sundial `model.layers.7` has the 2nd-highest injection count (117) and
    the lowest ablation count (47).
  - Ablation exceeds injection only at Chronos-2 blocks 6 and 8.
- **Evidence.** Causal within-model (two interventions). **Status:** confirmed on run.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** Not re-measured. Its own-site ablation arm used the legacy null, so the size of the injection-vs-ablation gap is uncertain. Read the ablation side as an upper bound.
- **Reproduce.** `sae/response.py::feature_response_fingerprints` vs `feature_ablation_fingerprints`.
- **Ref.** ROADMAP §27.3 (~23809–23823).
- **Score. 4/5.** Steering a direction in and removing a feature where it fires are different questions with different answers.

#### CA-09 · Single-direction robustness differs about 7× between models
- **Claim.**
  - Within each model, the fraction of the hidden state removed predicts
    forecast movement (Spearman TimesFM +0.747, Chronos-Bolt +0.697,
    Chronos-2 +0.367, Sundial +0.269).
  - At a similar removed fraction (Sundial 15.8% vs TimesFM 18.5%), the
    median effect is 0.040 vs 0.275.
- **Evidence.** Causal within-model. **Status:** exploratory. Whether depth, architecture or dictionary quality drives it is undetermined.
- **Reproduce.** `runs/full_report_run_4model/sae/*_ablation.json`.
- **Ref.** ROADMAP §32.7b(3) (~28537–28545).
- **Score. 3/5.**

#### CA-10 · Causal effect space is a continuum; six broad "families" tile it but do not beat a shuffle null
- **Claim.**
  - Pooled causal SAE features (201, 9 ablation channels) have a final-partition silhouette of
    0.31–0.345 for every family count from 3 to 16. There are no crisp clusters.
  - A parsimonious cut gives 6 families covering 196 of 201 features, each spanning all 4 models:
    Long-range steerers, Level raisers, Seasonality dampeners, Near-term steerers, Volatility
    dampeners, Volatility amplifiers (the two horizon titles were corrected in R2: those channels
    are distances, not directions).
  - They do not beat a column-shuffle null: p_n_families 0.9851, p_frac_assigned 0.8358.
  - Membership is more model-segregated than chance (purity p = 0.0348).
  - The tight complete-linkage atlas leaves 69% of features unassigned.
- **Correction (2026-09-29, titles only).** The horizon-distance channels clear in every family, so titles led by them named what families share. Titles now lead with directed channels (commit f2ed2a9); membership, sizes and nulls are unchanged (only `families[].title` differs in `concept_families.json`). Reference-run titles are now: Accuracy improvers (59), Level raisers (46), Seasonality dampeners (29), Volatility amplifiers & Seasonality amplifiers (28), Volatility dampeners (21), Volatility amplifiers & Trend dampeners (13).
- **Evidence.** Descriptive. **Status:** exploratory; replicated on the v2 corpus.
- **Replication (2026-09-29).** `runs/concept_atlas_v2` (v2 corpus, fixed Sundial; ROADMAP §37.11c): 293 causal features → 9 families, 288/293 (0.9829) assigned; null again not beaten (p_n_families 1.0, p_frac_assigned 0.9851); "segregated by model"; 3 tight concepts split. Two families' titles were near-identical ("Near-term steerers & Long-range steerers" / "Long-range steerers & Near-term steerers"); fixed the same day (titles lead with directed channels, commit f2ed2a9). v2 family titles are now: Level lowerers (80), Level raisers & Accuracy improvers (61), Volatility dampeners (48), Trend boosters & Accuracy improvers (23), Seasonality amplifiers (17), Trend boosters & Volatility amplifiers (12), Level raisers & Trend boosters (18), Trend dampeners (18), High-frequency shifters (11).
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** Families: 5 at threshold 0.85 (92/110 assigned), p_n_families 0.3781094527363184, p_frac_assigned 0.38308457711442784; still 'segregated by model', and still not beating the shuffle null. The continuum reading survives.
- **Universal effect types at 7 models (2026-10-07, `panel7_v2_dev`, profile-matched null; read from `examples/panel7_v2/run/sae/concept_families.json`).** Of 9 families over 194 causal features, three have causal members in **all seven** models: Level lowerers 44 (Chronos-2 9, Chronos-Bolt 2, Chronos-T5-Base 6, Sundial 10, Time-MoE 4, Timer 6, TimesFM 7), Level raisers 45 (10, 3, 5, 6, 3, 4, 14), Seasonality dampeners 25 (5, 5, 2, 9, 1, 2, 1); Volatility dampeners span 6 models, Mixed-effect II 6, Mixed-effect I and Trend boosters 5, Seasonality amplifiers and Mixed-effect III 4. Membership follows each feature's own measured effect, so "every model has level-raising / level-lowering / seasonality-damping causal features" holds whether or not the partition beats its shuffle null. Descriptive, exploratory, not registered; Timer's members carry its chance-level clear rate (PM-17). Ref. ROADMAP §40 (Findings 2026-10-07).
- **Reproduce.** `sae/concept_families.py::run_concept_families` on `runs/full_report_run_4model` → `sae/concept_families.json`.
- **Ref.** ROADMAP §37.11b (incl. R2).
- **Score. 4/5.** It reframes "concepts" in effect space as a readable tiling rather than discovered units. It also explains why tight concepts are small and why near neighbours fall outside them.

#### CA-11 · Held-out: 10 of 31 single-feature causal claims confirm; no cross-model causal claim does
- **Claim.** K2, the first private-split test of causal concept claims, was run once on 7 models. Each Holm family is per claim type.
  - **Single-feature causal effects: 10/31 tested confirm** (32 registered; 1 not testable because the feature fires on no private series). Each confirmed claim has p 0.001 after the adaptive redraw to 1000 draws, and p_holm 0.031 (the floor, 31/1001).
  - By model, confirmed/registered: TimesFM 3/4, Time-MoE 2/7, Sundial 2/4, Chronos-Bolt 2/6, Chronos-2 1/7, Chronos-T5-Base 0/4.
  - By channel: horizon_shape_near 4/8, mase 3/8, spectral_centroid 2/4, dispersion 1/3, seasonal 0/4, level 0/4.
  - 19 of the 21 failures were not even at the 200-draw p floor. There is 1 sign flip (Chronos-T5-Base b.10 f195).
  - **Multi-model atlas concepts: 0/3** (concepts 6, 8, 9). Centroid cosine still beats the null (p_centroid 0.0005–0.007), but only 1/3 member pairs reach cosine 0.9 (rule fails, p_holm 0.1349). The one atlas claim that confirms is single-model: Chronos-2 b.5/b.7/b.8 (pair fraction 1.0, centroid 0.9588, p_holm 0.008).
  - **Shared-input causal agreement (L5): 0/6 tested** (11 registered; 5 not testable because one side does not clear its own null on the private shared series). The best case, Chronos-Bolt b.4 → Chronos-2 b.7 'same causal effect', reproduces the verdict (p 0.011) but misses Holm (0.0659). Two 'acts differently' verdicts reproduce (Holm 0.1598, 0.1918).
  - Same look: concept transfer 20/20 (SH-18), structure 2/2 (SH-16, MN-14), U1 2/2 (MN-28), L0 1/1, L1 replicates, and L3 6/9 (`frequency_shift` and `smooth` are stronger on private; `level_shift` flips sign, −0.188168449197861 → 0.10862299465240642). In total, 43 of 82 replicable claims confirm and 6 are not testable.
- **Reading of the atlas failures (2026-10-07).** All three multi-model concepts have private centroid cosines 0.9554–0.9883 against null p95 0.8544–0.9067 (p_centroid 0.0005–0.0070); they fail only the registered conjunctive rule's second leg (≥ half of member pairs at cosine ≥ 0.9; with 3 members, 1 of 3 pairs did, p_pair 0.045–0.064). The registered verdict (0/3) stands; "the average effect profile of a multi-model concept reproduces" is a post hoc reading of one component and needs its own registration (ROADMAP §40).
- **Caveat on the artifact's text.** 19 non-confirmed claims carry the reading "untestable", which is wrong. The MDE was computed from each claim's own 200 draws, but the adaptive procedure can reach the Holm floor, so these claims were tested and not detected. The verdicts are unaffected, and the reading is now recomputed at render time.
- **Models / layers.** TimesFM, Chronos-2, Sundial, Chronos-Bolt, Timer, Time-MoE, Chronos-T5-Base (encoder); profile-matched null (MN-29).
- **Evidence.** **Held-out confirmed** (causal within-model, for the 10 single-feature claims). Held-out *not* confirmed for every cross-model causal claim. **Status:** held-out confirmed / negative on private.
- **Reproduce.** `runs/panel7_v2_dev/confirm/confirmation.json` (`concept_replication`), private v2 epoch 3 (971 series), registry sha256 `ff6c4f9e…`, k3wt 597848a.
- **Ref.** ROADMAP §38.2 (Findings — K2 CONFIRMED ONCE).
- **Score. 5/5.** This is the gold-standard answer to "do the causal concept claims hold up". The individual within-model causal features partly do (about 1 in 3). The cross-model causal sharing claims do not, while input-level sharing replicates on a fourth private draw: models share what they select, not demonstrably what they do with it.

#### CA-12 · Held-out V3 (3× corpus, 7 models): 11 of 24 single-feature causal claims confirm; transfer 20/20 on private and 20/20 on raw real windows
- **Claim.** `panel7_v3_dev`, confirmed once on `benchmark_v3/private_epoch1` (1024 series, first look; 111 registered, 63 replicable): **41 of 63 confirm, 3 not testable.**
  - **Single-feature causal: 11/24 tested** (27 registered; 3 fire on no private series). Confirmed/tested by model: Chronos-2 4/6, Chronos-Bolt 5/13, Sundial 1/1, Timer 1/4. Confirmed p_holm 0.023976–0.044955. K2 (CA-11) was 10/31.
  - **Concept transfer 20/20** (Holm max 0.009995002498750623); **external raw real windows (800, overlap 0/0 with dev and private, not counted in the verdict) 20/20**, AUC 0.757–0.991.
  - Structure 2/2 (no concept spans all 7 models; causally-null share of prominent features 0.9186991869918699); U1 reliability 2/3 (Time-MoE, Timer); L0 1/1 (Chronos-2 better on `mixture`, 0.30262611941041717 [0.17646094397473408, 0.4447032489712877]); L1 replicates (CKA 0.4435 dev, 0.4491 [0.4353, 0.4641] private); L3 4/9 by the strict CI rule, 8/9 same sign, `level_shift` flips sign again (−0.164 → 0.252; K2 −0.188 → 0.109).
  - No cross-model causal claim was registrable: atlas-centroid 0 (all 4 atlas concepts excluded by the pre-set per-target BH gate, ROADMAP §42.1).
- **Reading.** The K2 pattern holds on a 3× larger corpus with more real sources: what models *select* is shared and replicates on synthetic and raw real data; within-model single-feature effects replicate about half the time; shared causal *effects* remain undemonstrated.
- **Models / layers.** All 7 (Chronos-T5-Base encoder only); profile-matched cov null.
- **Evidence.** **Held-out confirmed** (causal within-model for the 11; correlational for transfer; external real leg is external validity, not confirmation). Private exchangeability was computed after the look (model-free): energy p 0.994503, TOST 22/22 (MP-09); private audit `not_recorded`. **Status:** held-out confirmed.
- **Reproduce.** `runs/panel7_v3_dev/confirm/confirmation.json`, registry sha256 `67b91269…`, config `configs/panel7_v3.yaml` (commit 877ca70).
- **Ref.** ROADMAP §41.6.
- **Score. 4/5.** Independent replication of CA-11's answer on a new corpus and on raw real windows.

---

#### CA-13 · Local forecast-head effects replicate, but the stronger retained-feature comparison fails its gate
- **Claim.** Fresh128 generating bundles, period16 phase tangents at actual
  forecast-head inputs: native-minus-frozen-probe off-target energy is
  **0.05987707943785135** [0.04375830966984287, 0.0802239206957035] for
  Bolt and **0.4939784532781031** [0.325254252594046, 0.7167902722611047]
  for TimesFM. Their difference is **0.4341013738402517**
  [0.269918309536019, 0.6470654426926147]; all Holm3 p-values **0.0015**.
  **The complete cross-model claim fails:** TimesFM's frozen-probe off-target
  relative RMS is **0.11326241938620228**, exceeding the fixed0.10 retained
  feature gate. Bolt passes at **0.09855118286690241**. No rescue or retuning.
  Internal interventions reproduce natural local output derivatives and self
  patches are exactly0, but this does not establish a shared isolated feature
  used differently by the two heads.
- **Models / layers.** Bolt `output_patch_embedding` input; TimesFM
  `output_projection_point` input, actual native read positions.
- **Evidence.** Causal within-model (L3) for measured local intervention
  responses; predictive for the frozen probe. Stronger retained-feature
  comparison is not established.
- **Status.** Negative (closed) for full cross-model gate; effect sizes
  replicated on fresh synthetic seeds, not sealed benchmark confirmation.
- **Reproduce.** `runs/feature_search_20261111_local_tangent_replication1659000/`,
  including plots with the failed gate; `experiments/feature_search/local_tangent.py`.
- **Ref.** ROADMAP §48.13.
- **Score. 3/5.** Demonstrates why significant intervention contrasts alone
  cannot certify that a cleanly retained concept accounts for model differences.

#### CA-14 · An anchored composition edit repairs a late TimesFM suffix on fresh seeds
- **Claim.** On16 fresh discovery seeds, replacing the state after15 of20
  blocks with an amplitude-weighted, background-anchored combination of natural
  component states reduces normalized MSE from **0.4625987461492669** to
  **0.036757892047086**. The mean of two geometry-matched random controls is
  **0.36724197569672923**; both gain contrasts are positive16/16 with
  Holm8 p **0.000244140625** across the four-cut discovery search. Suffix
  composition-residual energy is **0.03119729414172619**; aggregate recovery
  relative to the actual component-sum reference is **0.9274894544128434**.
- **Models / layers.** TimesFM2.5-200M, after15 blocks (zero-based index14),
  full native four-token states, periods32/16, context128/horizon64. Only
  candidates0/5/10/15 were searched;20 is a late control. Cut10 fails random
  geometry and stays untestable;0/5 fail their corrected tests.
- **Evidence.** Causal within-model (L3), structured state intervention
  independently replicated with oracle-separated input components. The edit is an
  explicit off-manifold composition hypothesis, not an isolated feature.
- **Status.** Replicated on64 fresh synthetic seeds, with all frozen gates
  passed and no excluded seeds. Native MSE **0.463167541269782** falls to
  **0.03884115741909176**, versus random mean **0.4001224368955292**.
  Native gain **0.4243263838506902** is positive61/64, Holm2 p
  **4.742842403537839e-15**; random advantage **0.3612812794764374**
  is positive64/64, Holm2 p **2.168404344971009e-19**. Suffix energy
  **0.03340652363661104**, component reference **0.0037141944288743354**,
  and gap recovery **0.9235461810611629** meet the fixed practical gates.
  Sign tests support conditional majority; mean gates are empirical, not
  population mean-superiority tests. Discovery effects were subject to selection
  optimism. The original execution exceeded its
  memory cap; a corrected execution preserves all29 arrays bitwise and every
  scientific result exactly while meeting the same cap. It is not a second
  scientific sample. No harmful-layer origin or cross-model causal identity
  follows; success concerns the earliest tested cut clearing every discovery
  gate. Cut10 remains unresolved under its failed random-control validity gate.
- **Reproduce.** `experiments/feature_search/composition_restoration.py`;
  `runs/feature_search_stage49_restoration3600000_memorysafe/`, with linked
  original failed execution and independent cached statistics audit; fresh
  `runs/feature_search_stage49_restoration3700000/` and
  `experiments/feature_search/composition_restoration_replication.py`.
- **Ref.** ROADMAP §49.14–49.17.
- **Score. 4/5.** A controlled, independently replicated state intervention
  moves beyond feature decodability to actual forecasting behavior.
- **Correction (2026-11-12).** Fresh64 replication replaces planned status;
  the discovery sample and its resource-correction history remain visible.

## D. Where and when: depth, crystallization, coverage

#### DE-01 · Whole-layer patching hides a strong, depth-intensifying recency effect
- **Claim.**
  - TimesFM's whole-layer restoration is flat across depth (`level_shift`
    0.036→0.035).
  - Per window, restoration concentrates in window 15 of 16, the one just
    before the horizon.
  - For `noise`, window 15 climbs 0.068→0.223→0.286→0.370→0.412 from layer
    0 to 16, while window 0 falls 0.091→0.009.
  - Chronos-T5 shows the same late-window concentration.
- **Correction (2026-10-17, ROADMAP §47.5).** `runs/medium_run` no longer exists, so 0.068->0.412 cannot be reproduced. Re-measured on v2 dev, TimesFM only, 320 family-stratified series (pipeline path verified: the n=16 re-run reproduces `panel7_v2_dev/l3/patching.npz` with max |Δ| 0.0): noise, window 14 of 15 (just before the horizon), layers 0/4/8/12/16: 0.4267186224460602, 0.5525148510932922, 0.6117260456085205, 0.5992766618728638, 0.6312912702560425; window 0: -0.19078321754932404 -> 6.814002699684352e-05. Concentration in the last window and its growth with depth hold (peak at layer 8 then flat); the levels are higher than the medium run's. Reproduce: `run_dagger_remeasure.py patching` -> `runs/dagger_remeasure/patching.json` (commit e8f4012).
- **Evidence.** Causal within-model (L3 patching). **Status:** exploratory (n≈300).
- **Reproduce.** `runs/medium_run/l3/patching.npz` (`restoration_windows_TimesFM`).
- **Ref.** ROADMAP §5.5 (~1907–1939).
- **Score. 4/5.** Averaging over positions can erase the causal story; patch per window.

#### DE-02 · Frequency-domain lens: TimesFM front-loads trend and never settles seasonality; Chronos resolves both late
- **Claim.**
  - TimesFM's trend crystallizes at relative depth 0.111, but its seasonal
    band never crystallizes (final error 0.0274, 2.07× the deepest layer's).
  - Chronos-T5-Base crystallizes trend at 0.909 and seasonal at 1.0.
  - n=10 series.
- **Evidence.** Depth-resolved lens. **Status:** exploratory (small n). Read with DE-04: Chronos's "1.0" is only 0.478 of its stack.
- **Reproduce.** `run_spectral_lens.py --run runs/medium_run_chronos_base`; `runs/spectral_lens.json`.
- **Ref.** ROADMAP §16 E13 (~9953–10028).
- **Score. 4/5.** A time-series-native version of "front-loads vs accumulates".

#### DE-03 · Chronos-2's skip lens is a silent no-op, because the head reads only forecast placeholders
- **Claim.**
  - The skip-lens MASE was byte-identical (1.564247…) at all 12 layers, with
    crystallization depth 0.0 everywhere, a "perfect" flat curve.
  - The cause is `hidden_states[:, -num_output_patches:]`: the head reads
    trailing placeholder tokens, while patches write leading context tokens.
  - The corrected reach-by-depth is 0.737/0.643/0.282/0.095/0.000 at blocks
    0/3/6/9/11.
- **Evidence.** Causal within-model (the corrected measurement). **Status:** confirmed. The skip lens is withheld via `forecast_reads_patched_positions() = False`.
- **Reproduce.** `models/contrib/chronos2_adapter.py`; `analysis/lens.py`; `runs/full_report_run_3model/lens/`.
- **Ref.** ROADMAP §24.7(1) (~18978–18999), §25(22); CLAUDE_FULL §11.42.
- **Score. 5/5.** A clean, plausible curve nearly got published as "Chronos-2's forecast is formed at layer 0", which is the opposite of the truth. Always run the self-patch = 0 and cross-patch ≠ 0 controls.

#### DE-04 · Cross-architecture depth axes do not align
- **Claim.**
  - On a block-fraction axis (whole stack, including uncaptured parts),
    Chronos-T5-Base's last captured encoder block is at
    **0.4782608695652174**.
  - TimesFM at stride 2 tops out at 0.9473684210526315.
  - The comparable-depth overlap is only 50.48%.
  - Re-running the E9 null tests on the corrected axis moved no verdicts.
- **Evidence.** Descriptive. **Status:** confirmed (wired 2026-08-18; `alignment.depth_axis: block`).
- **Reproduce.** `analysis/depth_axis.py`; `runs/medium_run_chronos_base`.
- **Ref.** ROADMAP §17.1 G-I, §18 F1 (~12417–12716).
- **Score. 5/5.** Qualifies every cross-model "where in depth" claim between encoder-decoder and decoder-only TSFMs.

#### DE-05 · Captured-compute coverage: "100% of blocks captured" can be the least-observed model
- **Claim.**

  | Model | Share of forecast FLOPs captured |
  |---|---|
  | TimesFM (stride 2) | 42.525% |
  | Chronos-T5-Base (100% of encoder blocks) | **14.35%** (decoder × 20 samples never seen) |
  | Sundial | ~23% (flow-matching head not captured) |
  | Chronos-Bolt | ~79% |
  | Chronos-2 | 97.5% |

  A "compute by depth" chart normalized by each model's own captured pass
  showed Chronos at 1.0 and TimesFM at 0.851, which inverts the true
  ordering.
- **Evidence.** Descriptive (measured with `FlopCounterMode`). **Status:** confirmed. Depth claims below 90% coverage are auto-qualified.
- **Reproduce.** `budget/model_budget.json::coverage`; `runs/medium_run_chronos_base`, `runs/full_report_run_4model`.
- **Ref.** ROADMAP §17.1 G-II, §18 F4, §23.2 B1 (~17194–17246); CLAUDE_FULL §12.
- **Score. 5/5.** Use FLOP fraction of the full inference pass, never block counts, as the headline coverage number.

#### DE-06 · Effective dimensionality predicts two interpretability proxies in opposite directions
- **Claim.**
  - Pooled over 8 (run, model) groups and depth-controlled, lower effective
    dimensionality goes with a more linearly readable forecast (tuned-lens
    R²: ρ −0.686 [−0.871, −0.329]).
  - Higher effective dimensionality goes with better family decodability
    (ρ +0.653 [+0.176, +0.859]).
  - L3 fingerprint entropy is a tighter proxy for decodability (ρ +0.674
    [+0.247, +0.823]).
  - Within Chronos-T5-Base the relation flips sign (effective dim
    3.42→13.93 as R² goes 0.433→0.581), so the pooled rule picked the worst
    layer (MN-11).
- **Evidence.** Correlational (n_groups 8). **Status:** exploratory.
- **Reproduce.** `analysis/layer_selection.py::run_layer_selection_study`; `runs/layer_selection_study.json`.
- **Ref.** ROADMAP §6.1 (~2347–2433), §6.2 (~3648–3671).
- **Score. 4/5.** No single "is this layer interesting" score exists. Forecast-readability and concept-decodability trade off.

#### DE-07 · Chronos-T5 scaling ladder (8.4M→709M): only decodability scales monotonically
- **Claim.**
  - Across tiny/mini/small/base/large, only family-probe decodability is
    monotone and significant (ρ 1.0, p 0.0167 = the n=5 floor).
  - The rest are non-monotone:
    - L1 peak CKA ρ 0.5, p 0.45;
    - L2 ρ 0.8/0.2;
    - AMI ρ 0.1;
    - crystallization depth ρ 0.7;
    - effective dimension ρ −0.7/−0.3;
    - MASE ρ −0.6, p 0.35 (bigger is not monotonically more accurate).
  - This refutes the 2-point pattern in SH-04 as a trend.
- **Evidence.** Mixed per metric. **Status:** exploratory (one family, one corpus).
- **Reproduce.** `configs/scaling_ladder_chronos.yaml`, `run_scaling_ladder.py`; `runs/ladder/scaling_ladder.json`.
- **Ref.** ROADMAP §20 H1 (~13883–14146).
- **Score. 5/5.** A same-architecture, same-corpus scaling study of interpretability properties for a TSFM, and a caution against reading 2-point comparisons as trends.

#### DE-08 · Which layers are "interesting" depends almost entirely on the selector
- **Claim.**
  - A specification curve over `work_bend`/`coverage`/`factor_emergence`
    finds 0/3 selected layers robust for every model. Jaccard vs baseline:
    TimesFM 0.0/0.0/0.0; Chronos-T5-Base 0.5/0.0/0.2; Chronos-2
    0.5/0.0/0.0.
  - By contrast, L0 strength-set and lens-crystallization claims were fully
    robust.
  - Separately, a spacing rule wrongly applied to a cross-layer change score
    rejected TimesFM layer 19, the top scorer (1.767), and admitted layers
    2 and 6, which scored below the mean. The selection changed from
    `[2,6,9,15,18]` to `[9,10,15,18,19]`.
- **Evidence.** Descriptive. **Status:** confirmed; `min_gap` now defaults to 1.
- **Reproduce.** `analysis/spec_curve.py`, `run_spec_curve.py`; `runs/full_report_run_large_revived/spec_curve/results.json`; `layer_screen.py::select_work_bend`.
- **Ref.** ROADMAP §34 A2 (~30777–30817), §31.1 (~27104–27140).
- **Score. 4/5.** Any claim keyed on "the important layers" must name its selector and check alternatives.

#### DE-09 · Time-MoE's window alignment decays to chance past layer 4, with zero backward leakage (architectural, not bad spans)
- **Claim.**
  - Per-layer impulse checks in float32, normalization off, at 0.04/0.2/0.4/1.0× the calibrated 0.25 amplitude.
    - Backward leakage is exactly 0.00e+00 in every layer and amplitude cell.
    - Across all 48 cells the argmax never lands earlier; every miss lands in a later window.
    - Spans have IoU 1.000 against the declared ones.
  - Diagonal hit fraction at 0.01: layers 0–4 1.00; layers 5/6/7 0.60/0.60/0.40; layer 11 0.33. At 0.05 and above, layers 5–11 are 0.07–0.20, which is chance.
  - Sundial reference at 0.25, layers 0–11: 1.00, 1.00, 1.00, 0.93, 0.53, 0.33, 0.27, 0.13, 0.27, 0.27, 0.27, 0.20. Its backward leakage of 3.5e-3 to 1.5e-2 comes entirely from its global z-score normalization.
  - Time-MoE decays much sooner than Sundial: at 0.01 Sundial is 1.00 through layer 10.
  - Method trap: with z-score normalization on, bf16 showed "backward leakage" of 0.06–0.6. Z-scoring rescales every position when one point changes, so leakage must be measured with normalization off.
- **Evidence.** Descriptive (instrument validation). **Status:** measured, one checkpoint (Maple728/TimeMoE-50M) at context 480.
- **Reproduce.** `scratchpad`-only diagnostic (`diag_align.py`, recorded in ROADMAP §38.3); the `panel7_v2` extract alignment table.
- **Ref.** ROADMAP §38.3 (K3 onboarding review); CLAUDE_FULL §11.22.
- **Score. 3/5.** The extract gate reads only the shallowest layer and passes. Window-level claims on Time-MoE layers ≥5 (per-window patching, window-resolved cross-model depth) are weak and must say so.

---

#### D-42C · A handful of SAE features move the forecast as much as the SAE error term (31 targets, descriptive)
- **Claim.** On 3 hand-picked `panel7_v3_dev` targets, zeroing the 5 most prominent features changes the forecast by a share 0.6649 (TimesFM stacked_xf.6), 0.5933 (Chronos-Bolt encoder.block.3) and 0.4068 (Sundial model.layers.11) of feature-plus-error-term change, and by 0.2187 / 0.2594 / 0.0862 context sd more than activation-matched random 5-feature sets. TimesFM top-20 removal moves the forecast 10.14 sd (off-manifold; its 0.9864 share is not a coverage claim).
- **Attribution-ranked, 31 targets (2026-10-10, ROADMAP §42.5, `runs/panel7_v3_sec42/sae/completeness.json`).** Median share N=1 0.5669, **N=5 0.7392** [0.0926, 0.9044], N=20 0.7545; the set beats activation-matched random sets on 30/31 targets at N=1 and N=5 (median excess 0.1332, 0.1661 context sd), 31/31 at N=20. Exceptions: Chronos-2 (N=5 0.09–0.33) and Time-MoE layers.11 at N=5. Per-target p is floored at 1/17 (16 draws).
- **Evidence.** Descriptive (31 targets, attribution ranking); the 3-target original is illustrative. **Status:** open (exploratory, dev).
- **Reproduce.** `run_completeness.py` (merge of §42 C), `sae/completeness.py`.
- **Ref.** ROADMAP §42.2, §42.5.
- **Score. 3/5.** Bears on "what is the main cause of a prediction": on most targets the forecast change is concentrated in about 5 atoms.

## E. Per-model character

#### PM-01 · Scale equivariance: TimesFM is exact; Chronos-T5 is off by four orders of magnitude; Sundial is far worse
- **Claim.** Residual in context-scale units:

  | Model | Residual |
  |---|---|
  | TimesFM | 1.15e-06 at 1000×; 0.0000 on the 4-model panel |
  | Chronos-T5-Base | 0.435 at 1000×; 0.450 at 0.001× |
  | Chronos-2 | 0.0086 |
  | Chronos-Bolt | 0.0080 |
  | Sundial | **38.4318** [30.35, 48.28] at 0.001× |

- **Evidence.** Behavioral (tier-0). **Status:** exploratory.
- **Reproduce.** `analysis/scale_equivariance.py`; `runs/frontend_check/frontend/frontend.json`; `runs/full_report_run_4model/report/findings.json` `frontend.1–4`.
- **Ref.** ROADMAP §16 E17 (~10562–10590).
- **Score. 4/5.** Explicit per-patch normalization buys scale invariance. Quantized tokenizers and Sundial's head do not have it.

#### PM-02 · NaN handling splits by family
- **Claim.**
  - TimesFM and Sundial **propagate** injected NaNs to the forecast in 3/3
    positions, with no error raised.
  - Chronos-T5-Base, Chronos-2 and Chronos-Bolt **handle** them (0/3
    non-finite).
  - Chronos's tokenizer clips 15.625% of series to its quantization bound at
    least once.
- **Evidence.** Behavioral. **Status:** exploratory.
- **Reproduce.** `analysis/nan_handling.py`, `quantization_resolution.py`; `configs/frontend_check.yaml`; 4-model `frontend.*`.
- **Ref.** ROADMAP §16 E17 (~10585–10624).
- **Score. 3/5.** A practical robustness difference for messy real inputs.

#### PM-03 · Long context: TimesFM keeps improving; Chronos-T5-Base degrades past ~384
- **Claim.**
  - On context 128→512, TimesFM's MASE falls monotonically, reaching 0.623
    [0.608, 0.637] at 512.
  - Chronos-T5-Base is best at 384 (0.673) and worse at 512 (0.690).
- **Evidence.** Behavioral. **Status:** exploratory (one recipe, one seed).
- **Reproduce.** `run_context_scaling_sweep.py --run runs/medium_run_chronos_base --min-context 128 --max-context 512 --n-points 6 --n-series 64`.
- **Ref.** ROADMAP §16 E20 (~10681–10743).
- **Score. 4/5.** A dose-response answer to "does the model use long context".

#### PM-04 · TimesFM patch-phase aliasing is real but modest; its output patch is 128 steps
- **Claim.**
  - Shifting the context start over 0..31 steps gives a per-series MASE CV of
    0.0232 [0.0181, 0.0287], an aggregate swing of about 2% (2.2955–2.3695).
    Individual series reach a CV of 0.125.
  - This confirms the period-32 anomaly: TimesFM 0.847 vs Chronos 0.663,
    the only period where TimesFM lost.
  - TimesFM 2.5 emits **128-step** output patches, and
    `predict(H=128)[:, :64]` is bit-identical to `predict(H=64)`.
- **Evidence.** Behavioral; architecture fact. **Status:** exploratory / confirmed.
- **Reproduce.** `analysis/phase_sensitivity.py`; `runs/medium_run_chronos_base/phase_sensitivity_sweep.json`; `timesfm_2p5_base.py:89-90`.
- **Ref.** ROADMAP §7 (~7030–7046), §16 E17 (~10498–10521), §33.1 (~29794–29825).
- **Score. 3/5.**

#### PM-05 · Accuracy, cost and observability: Chronos-2 is the most accurate, cheapest and best-observed; winners vary by family
- **Claim.**
  - **3-model panel.**
    - Chronos-2 MASE 1.6774 vs TimesFM 1.7345 and Chronos-T5-Base 1.9948.
    - Chronos-2 costs 119.5M params, 69.2 GFLOPs and 22.9 ms, with 97.5%
      captured.
    - Chronos-T5-Base is least accurate at 78× the FLOPs and 50× the latency.
    - TimesFM's edge on `random_parametric`/`sequential_par` **confirms on
      private data** (+0.169 [+0.098, +0.238], p 0.0005).
  - **4-model panel.**
    - Chronos-2 beats Sundial on 4 families and Chronos-Bolt on
      `random_parametric`.
    - Chronos-Bolt beats TimesFM on `mixture`.
    - Chronos-2's edge over Chronos-Bolt (1.68 vs 2.13) costs 5.4× the
      compute (8.65 vs 1.59 GFLOPs/series).
    - Chronos-2 > others on `mixture` is held-out confirmed (SH-18).
  - **v2 corpus (2026-09-29).** `runs/concept_atlas_v2` (v2 corpus, fixed Sundial; ROADMAP §37.11c): dev median MASE Chronos-2 1.1251, TimesFM 1.1829, Sundial 1.3335, Chronos-Bolt 1.4913 (naive 2.6829, seasonal naive 2.0280). Held-out on v2 private (800 series): overall TimesFM-vs-Chronos-2 effect −0.1478 [−0.2106, −0.0883], p 0.0005, where positive favours TimesFM, so Chronos-2 is better; the registered claim that Chronos-2 is better on `mixture` confirms again (0.3073 [0.2085, 0.4126], p_holm 0.0005, n 240).
- **Evidence.** Behavioral + descriptive. **Status:** partly held-out confirmed.
- **Reproduce.** `runs/full_report_run_large/{budget/model_budget.json,l0/metrics.parquet,confirm/confirmation.json}`; 4-model `l0/summary.json`, `report/findings.json` `l0.*`, `budget.*`.
- **Ref.** ROADMAP §24.7(20) (~19395–19453).
- **Score. 4/5.** Any accuracy ranking of these models must carry its compute confound.

#### PM-06 · Calibration: Sundial is badly under-covering
- **Claim.**

  | Model | Max reliability gap | 80% interval coverage |
  |---|---|---|
  | TimesFM | 0.023 | 0.786 |
  | Chronos-2 | 0.036 | 0.750 |
  | Chronos-Bolt | 0.046 | 0.723 |
  | Sundial | **0.253** | **0.374** |

- **Evidence.** Behavioral. **Status:** exploratory.
- **Reproduce.** `runs/full_report_run_4model/report/findings.json` `l0.7–l0.10`.
- **Ref.** CLAUDE.md §6.1 (L0 calibration).
- **Score. 3/5.**

#### PM-07 · Models differ ~5.4× in how readily they revert to the mean on identical inputs
- **Claim.**
  - 49.3% of 1297 panels have a "flat" forecast (sd < 10% of context sd).
    Context lag-1 autocorrelation explains this (ρ 0.626, p 5.7e-142), and
    flat forecasts score *better* (MASE 0.821 vs 1.440).
  - On the same series, the raw forecast-sd ratio differs by model: TimesFM
    0.299 (34.9% flat), Sundial 0.110, Chronos-Bolt 0.098, Chronos-2 0.055
    (60.6% flat).
  - The SAE adds only 2.4 points of flatness.
- **Evidence.** Behavioral. **Status:** confirmed on run.
- **Reproduce.** `report/derived.py::flatness_population`; `runs/full_report_run_4model/sae/*_ablation.json` (`unpatched`).
- **Ref.** ROADMAP §32.7 (~28250–28479).
- **Re-check (2026-09-27).** Preprocessing is ruled out: native library forecasts match the adapters to 0.002–0.028 context sd, and flat fractions match. Flat is 2–4% on periodic contexts vs 49–61% on non-periodic ones. The overall share is driven by the real-derived tier (BM-06). Ref. ROADMAP §32.7 addendum.
- **Score. 4/5.** Flat forecasts are mostly optimal behavior on noise-like context, and they expose a real, unstated per-model difference in reversion to the mean.

#### PM-15 · The Sundial adapter feeds raw-scale input (checkpoint normalization bypassed)
- **Claim.**
  - Sundial's remote code normalizes per series only in `generate()`. The adapter calls `forward()`,
    which defaults to `revin=False`, and whose own revin branch is broken for num_samples > 1. So
    Sundial never saw normalized input.
  - Symptoms:
    - scale-equivariance residual 38.4318 context-sd units, vs ≤ 0.0086 for the other three models;
    - 100% flat forecasts (MASE 4.134) on large-valued electricity series, where the other models are
      at 24–32%;
    - rescaling or z-scoring removes it.
  - On the small-valued v1 corpus the effect is small: median MASE 1.3552 → 1.3111, 80% coverage
    0.3761 → 0.3325. Sundial's under-coverage (PM-06) is real, not this defect.
- **Evidence.** Behavioral, adapter-level (verified at source). **Status:** fixed 2026-09-28.
  - With the checkpoint's `generate()` rule (std + 1e-5), Sundial's residual is 2.9e-05 at ×1000 and
    0.029955 at ×0.001 (was 3.3741 / 38.4318). v2 electricity flat share 1.0 → 0.08.
  - The same defect was in `generic_hf` on Timer: 49.7334 / 3.8053 → 0.0198 / 0.0214.
  - Existing runs' Sundial activations and concept results predate the fix.
  - **In-pipeline check (2026-09-29).** `runs/concept_atlas_v2` (v2 corpus, fixed Sundial; ROADMAP §37.11c): frontend residual 3.302e-05 (×1000) / 0.03318 (×0.001), vs TimesFM 1.965e-06, Chronos-2 0.009253, Chronos-Bolt 0.008653 at ×0.001.
  - **Correction (2026-09-29).** This entry was first filed as PM-10, an ID already in use (Chronos-T5 decoder attention). Renumbered to PM-15; IDs are never reused.
- **Reproduce.** Scratch scripts `sundial_scale.py`, `sundial_revin.py`, `sundial_seqpar.py` (session scratchpad); `frontend` findings in `runs/full_report_run_4model`.
- **Ref.** ROADMAP §32.7 D1 (S1 bullet).
- **Score. 4/5.** A silent adapter-level input bug that the pipeline's own frontend diagnostic caught numerically, but that nobody read as a bug. It invalidates Sundial on any large-scale corpus.

#### PM-08 · `random_init` twins are not one null condition
- **Claim.**
  - **TimesFM's twin is an exact identity stack.** `RMSNorm` zero-inits
    `scale` with no `1 + scale` term, so 140 of 232 tensors are zero and all
    20 layers are bit-identical. Yet it is the *most* causally reachable
    twin: 15.6% forecast change, and relative reach 0.18687 at all 5 layers.
  - **Chronos-2 and Chronos-Bolt twins are numerically dead.** Their
    changes are 6.5e-08 and 9.3e-08 of forecast scale, a per-series-scale
    constant.
  - **Sundial's twin is live**, at 9.0% (reach 0.68–0.87).
  - A causal-agreement floor from twins exists for only 1 of 6 pairs.
- **Evidence.** Causal within-model / architecture fact (verified at source). **Status:** confirmed. The within-run matched-feature floor (CA-06) replaced twins.
- **Reproduce.** Adapter loads with `random_init: true`; `analysis/response_reach.py::reach_probe`; `configs/null_4model_causal_floor.yaml`, `configs/null_timesfm_random.yaml`.
- **Ref.** ROADMAP §29 (~25076–25230), §37.8 P5a (~36271–36308); CLAUDE_FULL §11.56.
- **Score. 5/5.** "Random weights" means different things per architecture. TimesFM's twin is a valid geometric floor but an embedding-only causal floor; the Chronos twins cannot carry a causal battery.

#### PM-09 · Sundial: fast position mixing, sampled head, and rarely scorable as a causal destination
- **Claim.**
  - The alignment hit fraction is 1.00 at every layer at 0.02–0.05×
    amplitude, but decays by mid-depth at 0.25×, with zero future leakage.
    This is attributed to plain-residual, no-QK-norm blocks.
  - The flow-matching head is unseeded: two bare `predict()` calls differ by
    up to 0.4975 (Chronos-T5-Small: 0.124).
  - As a shared-input-agreement destination, only **1 of 71** tests is
    scorable. This is unexplained.
  - **Correction (2026-09-27).** The 1-of-71 figure was an instrument defect,
    not a Sundial property. P5b's own-effect null seeded Sundial's sampled head
    differently from its baseline, inflating the null (level p95 0.1430 vs
    0.029–0.059 in Sundial's own battery). After the fix (merged `8a1b253`),
    Sundial is scorable as a destination in 12/71 (null p95 0.0236), still the
    lowest of the four destinations (others 18–28). ROADMAP §37.8.
- **Evidence.** Descriptive / causal. **Status:** confirmed; the destination-scorability bullet is superseded (instrument defect).
- **Reproduce.** `--check-alignment` on Sundial configs; `models/conformance.py::_seeded_predict`; `sae/shared_input_agreement.json`.
- **Ref.** CLAUDE_FULL §11.22, §11.50; ROADMAP §37.8 (~36367).
- **Score. 3/5.**

#### PM-10 · Chronos-T5's decoder attends most to the latest step; TimesFM's causal mask is exact
- **Claim.** TimesFM's `future_mass` is exactly 0.0 at every head. Chronos-T5's first-step cross-attention peaks at lag 0 in every layer.
- **Evidence.** Descriptive. **Status:** exploratory.
- **Reproduce.** `runs/medium_run` attention artifacts.
- **Ref.** ROADMAP §5.5 (~1897–1906).
- **Score. 2/5.** A sanity check.

#### PM-11 · TimesFM has a causally load-bearing "random walk" concept, but it is not seed-stable
- **Claim.**
  - Concept C17 at TimesFM `stacked_xf.6`: structural ρ 0.833 with
    `has_random_walk`; 19/20 top series are `random_walk_drift` (24.78×
    enrichment); p 0.000999.
  - Ablating it worsens MASE by 4.04× the null p95.
  - It fails P2's stability test at both replicates.
- **Evidence.** Correlational + causal within-model. **Status:** exploratory (unstable).
- **Reproduce.** `runs/full_report_run_4model/sae/concept_profiles.json` (Q2).
- **Ref.** ROADMAP §37.11 P8 (~36938–36954).
- **Score. 3/5.** A vivid feature story that the reproducibility gate correctly refuses to promote.

#### PM-12 · A model's own quantile width is a better error predictor than cross-model disagreement
- **Claim.**
  - Own-width vs error Spearman: TimesFM **0.8665**, Chronos-T5-Base
    **0.7709**.
  - Cross-model disagreement reaches Spearman 0.716 (a 5.9× MASE spread
    across deciles), but it loses to own width in **10 of 11** model-runs,
    with 1 inconclusive and 0 favoring disagreement.
- **Evidence.** Behavioral. **Status:** confirmed (dev); disagreement is closed as a heuristic.
- **Reproduce.** `analysis/calibration.py::reliability_from_own_width`; `analysis/agreement.py`, `run_agreement.py`.
- **Ref.** ROADMAP §20 H4 (~13789–13882), §23.3 E1 (~17705–17784).
- **Score. 4/5.** A practitioner's result: "run two models and check agreement" is worse than the free quantile band.

#### PM-13 · Chronos-2's cross-series GROUP attention is unmeasured by design
- **Claim.** All analysis is `[series, window, dim]`. GROUP attention has no time interval, so everything reported about Chronos-2 concerns its TIME axis. How much computation lives on GROUP is unknown.
- **Evidence.** Scope statement. **Status:** by design (2026-08-12).
- **Ref.** CLAUDE_FULL §12.
- **Score. 3/5.** A caveat that must travel with every Chronos-2 claim.

#### PM-14 · SAE activation magnitudes are not comparable across models without a normalizer
- **Claim.**
  - Median hidden-state norm ranges from 2.62 (Chronos-2 b.8) to 143.25
    (TimesFM xf.10), a **55×** spread.
  - "Activation 0.31" and "23.13" were 11.8% and 43.8% of their hidden
    states.
- **Evidence.** Descriptive. **Status:** fixed (renders % of a typical hidden state).
- **Reproduce.** `median_hidden_norm` in `runs/full_report_run_4model/sae/meta.json`.
- **Ref.** ROADMAP §32.7b(2) (~28516–28535).
- **Score. 3/5.**

#### PM-16 · Timer returns all-NaN forecasts and activations on a constant context; the other six panel models do not
- **Claim.**
  - Setup: context 480, horizon 64, constants 0.0, 1.0, 1e3 and 1 + 1e-7 noise, with and without autocast.
  - `thuml/timer-base-84m` (via `generic_hf`) gives 100% NaN forecasts and NaN activations in all 8/8 captured layers, in all four cases.
  - Measured on 1 + N(0, sd): NaN for sd ≤ 1e-4, finite from sd = 1e-3.
  - TimesFM, Chronos-2, Sundial, Chronos-Bolt and Time-MoE (with `input_normalization: zscore`) return the constant, within 2e-7 of it.
  - Chronos-T5-Base is finite but off by 0.0015–0.0029 in units of |constant|+1.
  - Consequence: 5 constant Monash windows in the SAE real-data augmentation poisoned the first optimizer step and crashed the 7-model run (ROADMAP §38.3).
  - Consequence: before the fix, L0 silently averaged over a different series set per model when a model returned NaN (pandas skips NaN), with CI [nan, nan] and a finite p. The same happened in the skip lens (NaN curves read as "never converges").
- **Evidence.** Behavioral. **Status:** measured.
  - `frontend` now probes constant contexts and renders a loud row.
  - `--doctor` warns.
  - L0 and the lens drop non-finite series from every model and record them.
  - SAE training drops non-finite real-data rows.
  - Model inputs are never altered by default: an epsilon large enough to help (≥ 1e-3 at level 1) would materially rewrite the input.
  - Still unguarded: predictions stored raw and read by L3/attention/agreement, where NaN is mostly skipped via nanmean. Relevant only if a constant series enters a sampled corpus.
- **Reproduce.** `analysis/constant_context.py`, `frontend.constant_context`, `tests/test_constant_context.py`.
- **Ref.** ROADMAP §38.3 (7-model run crash).
- **Score. 3/5.** A zero-code adapter can pass the alignment gate and still fail on a trivial input. "Loads and aligns" is not "handles every input".

---

#### PM-17 · Timer's SAE features do not clear the ablation null above chance
- **Claim.** On the 7-model panel, Timer's 6 targets give 30 clearing cells against 29.5 expected by empirical chance (1.0169×), and 0 of 6 targets are BH-significant. Every other model is above chance (1.6429× Chronos-T5-Base to 5.6986× Chronos-Bolt). Timer's targets do reach the forecast (relative reach 0.0659104–0.260557), so this is not an unreachable layer: its features' ablations are not more forecast-specific than random directions of the same size.
- **Models / layers.** Timer (`thuml/timer-base-84m`, `generic_hf`), model.layers 1, 2, 3, 4, 6, 7.
- **Evidence.** Causal within-model (ablation vs profile-matched null), descriptive aggregate. **Status:** exploratory (dev).
- **Reproduce.** `runs/panel7_v2_dev/sae/Timer/*_ablation.json` (`empirical_chance`); clears from the run log (to be rendered by `chance-render`).
- **Ref.** ROADMAP §38.3.4.
- **Score. 3/5.** A per-model fact that changes how Timer's concepts are read, and a reason to gate concepts on per-target significance.

#### PM-18 · Controlled feature composition reverses TimesFM versus Bolt's ranking; their failure modes differ
- **Claim.** On 128 fresh seed bundles, each with paired pure and mixed
  period32/period16 signals, the TimesFM-minus-Bolt mixed-error gap is
  **0.146831872030236** [0.11511752415202106, 0.1790967268823472], while the
  model-by-composition contrast is **0.25178730121876025**
  [0.21538921012423637, 0.2885231015590052]. Changing only one input phase
  changes the other predicted phase more in TimesFM: excess disturbances
  **0.3279038512503415** [0.24031824798550505, 0.4230899215990888] and
  **0.4534599782467891** [0.38325726931122517, 0.5251240350920741] radians.
  All four predeclared tests pass Holm4 **0.002** and their practical thresholds.
  Resampling keeps all eight factorial/pure cells of a seed together.
- **Composition diagnostic.** Forecasting the known pure components separately
  and summing with the exact known offset correction reduces TimesFM MSE from
  **0.3512594243442453** to **0.0030202935401052824**, removing a fraction
  **0.9914015302344021** [0.9902879922367146, 0.9923149662433601]. It does
  not help Bolt on this draw. This uses an **oracle input decomposition** and
  exact offset, not a generic forecast upgrade or `f(offset)` baseline.
- **Fresh context/model extension.** 64 new bundles, contexts128/256/512,
  horizon32: TimesFM mixed MSE **0.349466573709151 / 0.34946656956483124 /
  0.34946649909290317**; Bolt **0.21823886495623107 / 0.13606338388518485 /
  0.041825578693870505**; Chronos-2 **0.0009742264718476344 /
  0.0006608511550697683 / 0.0005744582875454149**. All six predeclared
  composition claims pass Holm6 **0.003**. These signals repeat identical
  32-step patches: TimesFM's invariance is specific to this design, not a
  general claim about long context. All MSE values use the same
  `(a²+b²)/2` denominator across the pure and mixed conditions of each seed.
- **Models / layers.** TimesFM 2.5-200M, Chronos-Bolt-small, Chronos-2; native forecasts.
- **Noise robustness.** Another 64 fresh bundles, context512/horizon32, with
  Gaussian context noise at10% of clean mixture RMS and clean conditional-mean
  future truth: TimesFM's native-minus-component-sum gain is
  **0.565583824301698** [0.4789526340549909, 0.6539884940384483], and its
  native mixed-loss excess over Bolt is **0.523860819463454**
  [0.4332454357804728, 0.6155645624105813]. The four fixed clean/noisy claims
  all pass Holm4 **0.002**. This extension subtracts **actual**
  `f(background)`, not an exact-offset oracle. Noise is shared across component
  counterfactuals; known decomposition is still oracle information. Native
  normalization scales can change, so this remains behavioral evidence.
- **Evidence.** Behavioral, controlled input intervention; no hidden-mechanism
  attribution. SH-25 separately establishes task-feature recoverability.
- **Status.** Prospective fresh-synthetic replication and extension; no sealed
  benchmark-private look. Restricted to this two-frequency task family.
- **Reproduce.** `experiments/feature_search/mixture_replication.py`,
  `mixture_context_expansion.py`, `mixture_composition_audit.py`;
  `runs/feature_search_20261111_mixture_replication1215000/`,
  `runs/feature_search_20261111_mixture_composition1215000/`,
  `runs/feature_search_20261111_mixture_context1326000/`,
  `runs/feature_search_20261111_mixture_noise1770000/`;
  noise driver `experiments/feature_search/mixture_noise_background.py`.
  The separate native public-CLI check is portable in
  `examples/feature_interference/` (saved forecasts, exact inputs and report).
- **Ref.** ROADMAP §48.10–48.11, §48.15, §48.21.
- **Score. 4/5.** A reproducible task-specific model comparison with feature
  interference, component controls and a third model, rather than a CKA ranking.
  Composition interference itself has prior art
  (https://arxiv.org/abs/2511.15324); see ROADMAP §48.17 for the contribution's scope.

#### PM-19 · Repeated patches delimit TimesFM's context benefit on two-frequency mixtures
- **Claim.** In a prospective32-bundle comparison with an identical future
  origin and short contexts that are exact suffixes of long contexts,
  TimesFM's mean context128→512 benefit for periods31/33 minus its benefit
  for period32 is **0.17250893269492243**
  [0.1367600676769811, 0.21923424793946084], p **0.0005**, passing the
  frozen0.05 practical threshold. A second period16 component is always
  present. At period31 mixed loss falls **0.13002088179741342 →
  0.003840171334537932**; at period32 it is **0.37043001525656616 →
  0.3704300549180914**. Loss uses common theoretical component energy.
  Identical32-step patches give identical value vectors in the inspected
  TimesFM architecture, explaining why duplicating them cannot add temporal
  distinction. This does not explain why its head forecasts a pure repeated
  sine well but a mixture poorly. Nearby periods also change normalization,
  frequency preference and finite-window structure; no sole cause is proven.
- **Models / layers.** TimesFM 2.5-200M, Bolt-small behavioral control;
  TimesFM first/middle/final residual token variation and installed source.
- **Evidence.** Behavioral, descriptive and source-level conditional argument;
  not a causal isolation of patch size.
- **Status.** Prospective fresh-synthetic challenge passed; restricted task family.
- **Reproduce.** `experiments/feature_search/patch_context_challenge.py`,
  `patch_homogeneity.py`; `runs/feature_search_20261111_patch_context1548000/`,
  `runs/feature_search_20261111_patch_homogeneity/`.
- **Ref.** ROADMAP §48.14.
- **Score. 4/5.** Turns a context-length anomaly into a prospective conditional
  prediction, while keeping the architecture explanation's limits explicit.

#### PM-20 · The strong two-frequency failure does not generalize to the tested trend–seasonality task
- **Claim.** A fresh 64-bundle, three-model discovery screen at periods24/48,
  context192/horizon48 finds **0/12** contrasts above the fixed absolute
  practical threshold0.05, even though11 have Holm12 p **0.003**. The largest
  native gap is Bolt-minus-TimesFM at period48: **0.032891776989840735**
  [0.028450752459306894, 0.0377026899694327]. At period24, native mixed MSE
  is **0.0032415266684128687** for TimesFM versus **0.016850267849793904**
  for Bolt. Units are the common sum of component future variances, including
  trend. All actual-background component-sum gains are below the advancement
  threshold. This does not establish equivalence or absence of small effects.
- **Models / layers.** TimesFM 2.5-200M, Chronos-Bolt-small, Chronos-2;
  native predictions, no hidden captures.
- **Evidence.** Behavioral; paired synthetic discovery screen.
- **Status.** Negative (closed) under the prospective practical gate;
  no selected follow-up or sealed-private confirmation.
- **Reproduce.** `experiments/feature_search/trend_seasonality.py`,
  `trend_seasonality_protocol.json`;
  `runs/feature_search_20261111_trend_seasonality1881000/`.
- **Ref.** ROADMAP §48.16.
- **Score. 3/5.** Bounds the positive finding's scope and prevents spending
  more compute on tiny but statistically significant differences.

#### PM-21 · Native linear head skips carry most local cross-frequency response in the tested mixture
- **Claim.** Exact head replay on128 cached seed bundles agrees with native
  forecasts to normalized RMS **1.1616196852667529e-07** (Bolt) and
  **1.5304380828017896e-07** (TimesFM). For period16 phase perturbations,
  the linear skip's signed projection onto the total off-target derivative is
  **0.8688605152258402** [0.8304050378746096, 0.9088376396896608] for Bolt
  and **0.9572140281831615** [0.9377662236464296, 0.9745467932949905] for
  TimesFM. Nonlinear shares are **0.1311390609148922** and
  **0.042786387793319745**. Normalization and numerical replay residuals
  are explicitly retained; delta0.02 sensitivity agrees with primary0.01.
  These are signed attribution shares, not fractions of forecast error or
  independent branch energies. Incoming states can already mix the concepts:
  this does not establish where that mixing originates or suggest a repair.
- **Models / layers.** Bolt-small `output_patch_embedding`; TimesFM2.5-200M
  `output_projection_point`; Chronos-2 `output_patch_embedding` with its shared
  nonlinear inverse; native median forecast selection and normalization.
- **Evidence.** Descriptive native computation decomposition along observed
  input phase changes; no learned probe or isolated causal-feature assertion.
- **Accurate comparator extension.** On the identical128 seeds and common
  analytic target-phase strength denominator, Chronos-2's total off-frequency
  energy ratio is **0.0009365699279938675**
  [0.0007697780421198571, 0.0011184857798916235], versus Bolt
  **0.069589415082317** and TimesFM **0.5068068289233191**. Chronos-2's
  skip/nonlinear ratios are **0.0009901870057601352** /
  **0.0004963239142425883**, giving the frozen descriptive label of weak
  branch mixing. Its nonlinear `sinh` inverse requires a common secant
  allocation through the SUM, not separate inverses of each branch.
  This allocation is explicit and not claimed unique; native controls and
  replay gates pass.
- **Fresh replication.** On128 new bundles, native off-target energy ratios
  are TimesFM **0.5261452802980068**, Bolt **0.06792029434615932**, and
  Chronos-2 **0.0010493469166017904**. TimesFM-minus-Chronos-2 difference
  is **0.5250959333814048**, approximate95% paired interval
  [0.3280945788163841, 0.7723828712240296]; Bolt-minus-Chronos-2 is
  **0.06687094742955753** [0.052205034722263094, 0.08340006809074582].
  Both frozen practical gates pass. Each has126 positive and2 negative
  paired energy numerators: q **0.984375**, exact pointwise95% interval
  [0.944697409977551, 0.9981021068782607], global Holm2 exact two-sided
  sign p **9.706056854739628e-35**. This establishes conditional majority
  direction; the energy-ratio intervals remain approximate. Chronos-2's weak
  branch-mixing description persists. No probe or hidden intervention was used.
- **Status.** Prospective fresh-synthetic replication of the two model
  comparisons; branch decomposition remains descriptive native computation,
  not an isolated causal-feature or sealed-private confirmation claim.
- **Reproduce.** `runs/feature_search_20261111_head_branch_audit1659000/`,
  including protocol, selected tensor hashes and branch/bootstrap arrays;
  `experiments/feature_search/head_branch_audit.py`;
  `runs/feature_search_20261111_chronos2_head_audit1659000/`,
  `runs/feature_search_20261111_native_head_replication2225000/`.
- **Ref.** ROADMAP §48.17, §48.19–48.20; CA-13 remains failed under its separate gate.
- **Score. 4/5.** Locates the measured output response in a simple native
  readout path and discourages attributing it to head nonlinearity without evidence.

#### PM-22 · A broader frequency screen exposes opposite TimesFM and Bolt weaknesses
- **Claim.** On 16 fresh matched bundles at context128/horizon64, normalized
  mixture MSE for periods32/16 is Bolt **0.3820556874818121**, TimesFM
  **0.5869264504828446**, Chronos-2 **0.0010072802166847731**. For periods40/20
  it is Bolt **0.9985357819308737**, TimesFM **0.00045863979466698586** and
  Chronos-2 **0.000867662034843415**. At context512 on40/20, Bolt remains
  **0.4748967708490257**; the other two are **4.763951575016e-05** and
  **0.0005000873729892579**. Periods24/12 and48/24 are low-error controls.
  Thirty-nine of the complete108 exact-sign/point-mean comparisons pass;
  recurring effects are not necessarily same-direction model superiority.
- **Models / layers.** Native TimesFM2.5-200M, Bolt-small, Chronos-2 forecasts;
  six period pairs, contexts128/512, no hidden activation pooling.
- **Evidence.** Behavioral exploratory screen with matched inputs and global
  Holm108. Conditional sign evidence differs from a test of expected loss.
- **Fresh replication.** On64 new bundles, the primary context128 native
  ranking-flip gate fails: the32/16 gap has q **0.5625**, Holm p **1.0**.
  The separate off-response reversal passes: TF-minus-Bolt differences are
  **0.4341239800184107** at32/16 and **-0.1813287680828609** at40/20;
  their crossover **-0.6154527481012717** has Holm p **9.90309863193456e-12**.
  TimesFM32 actual component-sum S/L is **0.006784148048216241**
  [0.005734700793525599,0.00837327135483489], while Bolt40 is
  **0.8502795746095388** [0.7679059716968856,0.9382502269311893]. Bolt40
  already has substantial pure-component errors. Its component-sum gain is
  not detected at128, and becomes negative at512. These are different observed
  failure patterns, not proof of exclusive internal mechanisms.
- **Partial phase-response follow-up.** On the fixed40/20 amplitude grid,
  Bolt native normalized MSE is **0.9937586478672578**; expected phase-axis
  error is **0.9436013883016936**, including **0.7026315836996115**
  gain/rotation error and **0.2409698046020821** wrong-time-character error.
  Joint phase error is **0.02573335521746873**. Low-order patterns are stable,
  but off-grid coverage fails (MN-43), so this is a partial behavioral
  description with preprocessing included, not a complete internal explanation.
- **Status.** Mixed fresh-synthetic replication: feature-response reversal
  and TimesFM composition gain replicate; primary accuracy-ranking flip fails.
  Context512 sensitivity does not rescue it. Ratio intervals are pointwise;
  sign evidence concerns conditional majority, not expected-loss superiority.
- **Reproduce.** `experiments/feature_search/frequency_generalization.py`;
  `runs/feature_search_stage49_frequency3200000/` and prepared sibling.
  Fresh check: `experiments/feature_search/frequency_crossover_replication.py`,
  `runs/feature_search_stage49_crossover3400000/` and prepared sibling.
  Partial response: `examples/feature_search_stage49/phase_orbit/`.
- **Ref.** ROADMAP §49.4–49.5, §49.7, §49.12.
- **Score. 4/5.** A concrete model-specific reversal motivates a falsifiable
  mechanism comparison rather than a universal model or similarity ranking.

#### PM-23 · TimesFM's homogeneous mixture failure survives a common normalizer
- **Claim.** Source-exact one-token replay reproduces16 inspected seeds × six
  native64-step conditions with maximum waveform RMS/common-scale
  **5.3707417253144065e-06**. Identical patch values make QK routing unable
  to distinguish positions in this regime. Native composition-residual energy
  is **0.5526366314127467**; fixed-mixture-normalizer residual energy remains
  **0.6739155886423474**, radial remainder **0.5285770016700883**, with cross
  energy **-0.6498545646648604**. Radial effects and nonadditivity substantially
  cancel. These are algebraic components, not fractions of causal blame.
- **Models / layers.** TimesFM2.5-200M, exact32/16 repeated patches,
  context128/horizon64; complete pointwise stack and native forecast head.
- **Evidence.** Descriptive/source-verified computation replay and an explicit
  common-normalizer counterfactual. Its lower-RMS pure branches are off-manifold.
- **Status.** Exploratory. Rejects a solely affine radial-bias account of that
  counterfactual; does not establish normalization is irrelevant, isolate a
  responsible layer, or extend the homogeneous theorem to noisy contexts.
- **Reproduce.** `experiments/feature_search/homogeneous_normalizer.py`;
  `runs/feature_search_stage49_homogeneous_pilot/`, calibration sibling.
- **Ref.** ROADMAP §49.11, §49.13; earlier patch boundary §48.14 /PM-19.
- **Score. 4/5.** A cheap source-faithful reduction narrows mechanistic
  explanations and makes controlled internal tests feasible on CPU.

## F. Method results — positives (what works, and was adopted)

#### MP-01 · Token→time spans can be measured, and a zero-code adapter follows
- **Claim.**
  - Impulse-based span discovery reproduces every hand-written adapter's
    spans exactly (mean IoU 1.0, 0 flagged). That covers 6 checkpoints and
    4 tokenization styles.
  - On `thuml/timer-base-84m` it recovers 96-step spans from measurement
    alone, and the `generic_hf` pipeline then runs end to end (12 sections,
    36 findings).
  - The first refusal statistic ("diffuseness") would have refused 3 of 5
    correct adapters (0.794/0.675/0.613), because it scales with token
    count. It was replaced by peak-to-pedestal contrast.
- **Evidence.** Descriptive (method validation). **Status:** confirmed.
- **Reproduce.** `extraction/span_discovery.py`; `run.py --discover-spans`, `--probe-adapter`; `models/generic_hf_adapter.py`.
- **Ref.** ROADMAP §16 E3 (~10855–11306); CLAUDE_FULL §11.33.
- **Score. 5/5.** Turns "add a TSFM" from adapter-writing into a measurement.

#### MP-02 · Alignment gates must be normalized by their attainable ceiling
- **Claim.** With 96-step tokens on a 32-step window, the diagonal-hit ceiling is exactly 1/3. Timer scored 0.3333 against a 0.5 bar, a false refusal of a provably correct map. The gate now uses `hits / resolvable_hit_ceiling`, which is a no-op for every earlier model.
- **Evidence.** Descriptive. **Status:** confirmed.
- **Reproduce.** `extraction/alignment.py::resolvable_hit_ceiling`.
- **Ref.** ROADMAP §16 E3(d) (~11218–11245); CLAUDE_FULL §11.35.
- **Score. 3/5.**

#### MP-03 · AuxK revives dead SAE/crosscoder atoms 10–13× on real activations, though a synthetic bed showed it as a wash
- **Claim.**
  - At a fixed 1280-atom dictionary, the crosscoder goes from 80 alive atoms
    (93.75% dead) to 835 alive with `aux_coef` 0.03125, and 1072 alive
    (16.2% dead) at 0.25.
  - Fidelity *rises* at the same time (TimesFM 0.4913→0.6028).
  - Bigger dictionaries and 10× more data do not fix death (MN-20).
  - The synthetic bed read dead 0.272→0.266.
  - The adopted recipe is `aux_k` + `dict_size_policy: search` +
    `min_train_steps`. It takes TimesFM xf.18 to 0.207±0.026 dead (dict
    2048) and Chronos-T5-Base b.6 to 0.219±0.042 (dict 192).
- **Evidence.** Method, verified on real checkpoints. **Status:** adopted, and config-scoped to preserve old runs.
- **Reproduce.** `run_crosscoder_stage0.py --grid full`; `runs/crosscoder_stage0_bigdata/crosscoder_stage0.json`; `configs/sae_revival.yaml`.
- **Ref.** ROADMAP §6.2.1 Stage 0 (~4332–4485), §23.2 A1 (~16892–17034).
- **Score. 4/5.** A mechanism can be inert on a synthetic bed and decisive on real data. See MN-21 for what revival costs.

#### MP-04 · A single-seed SAE "gold ranking" is noisy enough to flip a bake-off verdict
- **Claim.**
  - Re-extracting with a different seed flipped `work_bend`'s `beats_random`
    on 2 of 3 architectures, even though the selected layers were
    bit-identical.
  - Averaging the gold ranking over 3 seeded SAEs made every verdict agree
    across seeds.
  - `work_bend` then matched the oracle on Chronos-T5-Small (recall 1.00) and
    beat both nulls on Sundial. On TimesFM, neither it nor the oracle beat
    uniform stride, because the gold mass sits at the edges.
- **Evidence.** Method. **Status:** fixed; `work_bend` is the production selector.
- **Reproduce.** `analysis/layer_screen_bakeoff.py::build_gold_ranking`; `runs/layer_screen_bakeoff_v4(_seed1).json`.
- **Ref.** ROADMAP §6.1.1 (~2697–3290).
- **Score. 5/5.** Any SAE-derived reference used to validate a cheaper method needs replicate seeds.

#### MP-05 · SAE feature death is set by layer geometry, not the recipe
- **Claim.**
  - With an identical recipe (dict 6144, k 32, 8608 rows), Chronos-2 is
    0.108–0.134 dead while Chronos-T5-Base is 0.951–0.973 dead (~9×).
  - Across 45 cells, alive-atom count vs layer effective dim has ρ
    **+0.5842** (p 2.5e-05), a log-log exponent of **1.02**, and R² 0.3885.
  - For dict size the figures are ρ −0.2010 (n.s.) and exponent −1.128.
  - Within TimesFM, ρ is +0.900.
  - Only 3/45 cells are dictionary-bound.
  - After the revival recipe, the Chronos family is under 1% dead while
    TimesFM and Sundial stay at 11.8–29.6%.
- **Evidence.** Descriptive / correlational (partly circular with PCA-based effective dim). **Status:** replicated on 2 corpora; the controlled follow-up has not been run.
- **Reproduce.** `sae/meta.json` × `internals/profile.json` over nine runs; `runs/full_report_run_{3model,large,4model}/sae/meta.json`.
- **Ref.** ROADMAP §23.2 A1 (~17399–17424), §25.16(c) (~20339–20432), §26 F (~23412).
- **Score. 5/5.** One fixed SAE recipe is not equal grounds across TSFMs. Effective dim, which the pipeline computes for free, predicts the dictionary it can support.

#### MP-06 · Ground-truth alignment of SAE features is a real signal, when gated by a permutation null
- **Claim.**
  - After revival, 154/10240 TimesFM features and 115/6144 Chronos-T5-Base
    features match a ground-truth field.
  - The best match is Chronos f↔`has_intermittency`, ρ **0.881**.
  - The caveat is that raw ρ does not separate real from random weights: an
    untrained twin scores 0.565 vs 0.567. Only the gap over the permutation
    null does.
- **Evidence.** Correlational. **Status:** exploratory.
- **Reproduce.** `sae/ground_truth.py`; `runs/medium_run_chronos_base/sae/`, `runs/sae_revival`.
- **Ref.** ROADMAP §6.2 (~3630–3647), §25.20 (~20748–20917).
- **Score. 3/5.**

#### MP-07 · The causal ablation battery discriminates: SAE features clear a random-direction null well beyond chance
- **Claim.**
  - At Chronos-T5-Base b.10, 72 (feature, channel) cells clear vs 17.55
    expected, with self-patch exactly 0.0 and cross-patch 0.7082.
  - Random-control nominations clear at about half the rate of real rules
    (4/12 vs 8–9/12).
  - Across the 4-model battery, clearing cells run 2.49×–6.61× chance.
- **Evidence.** Causal within-model. **Status:** confirmed. The first pass applied to one model only, because of an autocast defect (MN-24).
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** The battery's `0.05 × cells` chance line is not a chance rate (effect is a mean over k rows vs the p95 of pooled per-row null values). The leave-one-draw-out empirical rate is 0.024605 (legacy) / 0.019711 (profile) per cell. Against it, legacy clears 1348 cells vs 103.6875 expected (13.0006×) and profile 254 vs 83.0625 (3.0579×), with 20/28 targets BH-significant under profile (27/28 legacy). The battery still discriminates, but the legacy null inflated clears about 5-fold (1348 → 254), because it removed a size-mismatched amount.
- **Feature-level chance on V3 (2026-10-13, ROADMAP §43.1).** Counting features (>= 1 of 9 channels) against the draw-level chance rate, which keeps the channels' dependence: `panel7_v3_dev` cov 91/967 vs 70.4375 expected (1.2919×, Poisson-binomial p 0.004505097802029455), iso 238 vs 67.0 (3.5522×); `panel7_v3_sec42` cov 112/1118 vs 80.25 (1.3956×, p 8.720751822035558e-05). The independent-channel product (111.19 for v3 cov) overstates chance; reading it as chance ("below chance") is wrong. `run_feature_chance_test.py`.
- **Reproduce.** `run_stage2_response_fingerprint.py`; `sae/response.py::feature_ablation_fingerprints`.
- **Ref.** ROADMAP §25.23 (~21263–21422), §27.3, §43.1.
- **Score. 4/5.** The foundation of every causal concept claim.

#### MP-08 · Reach gates need a relative threshold
- **Claim.** `relative_reach = cross_delta / forecast_scale > 1e-3` separates dead twins (≤6e-06 and ~3e-05) from live ones (≥0.1125). It sits 29× above the highest dead twin and 112× below the lowest live one. The old absolute `1e-12` gate read TimesFM's identity twin as dead.
- **Evidence.** Method. **Status:** adopted (`concepts.min_relative_reach`).
- **Reproduce.** `sae/response.py::reach_probe`.
- **Ref.** ROADMAP §37.8 P5a (~36271–36308); CLAUDE_FULL §11.56.
- **Score. 3/5.**

#### MP-09 · Dev and private benchmark splits are exchangeable, which is what the confirm stage relies on
- **Claim.**
  - `benchmark_large` public_dev vs private_test: energy distance p 0.5117,
    TOST equivalent on 21/22 catch22 features (`CO_trev_1_num`
    inconclusive), 5 near-duplicates out of 933,155, and Cramér's V ≤ 0.092.
  - Fresh epochs pass the same checks. Epoch 1: energy p 0.908046, 5/927365
    near-duplicates. Epoch 2: energy p 0.922539, 5/929295.
- **V2 exchangeability (benchmark_large_v2 public_dev vs private_epoch3, the 7-model confirmation split; `benchmark_large_v2/private_epoch3/cross_split.json`, ROADMAP ~39267):** energy p 0.951024, TOST 20/22, 6 cross-split near-duplicates of 937,015; dev effective dimension 5.367, private 5.57. (Added 2026-10-17: the paper quotes these.)
- **V3 exchangeability (computed 2026-10-12, after the confirm look, model-free; `benchmark_v3/cross_split_epoch1/cross_split.json`): dev 2876 vs private epoch 1 2868 sealed series (confirm draws 1024 of them via `confirm.max_series`); energy distance p 0.994503 (2000 permutations); TOST 22/22 catch22 features equivalent at 0.2 sd (Holm); tier composition p 0.971975; 9 cross-split DTW near-duplicate pairs of 8,248,368 (threshold 0.97); centroid distance 0.0705.
- **Evidence.** Held-out validation. **Status:** confirmed.
- **Reproduce.** `benchmark_validation/cross_split.py`; `run_validation.py --compare-splits`.
- **Ref.** ROADMAP §34 B2/B3 (~31618–31810), §37.10.
- **Score. 3/5.**

---

#### MP-10 · Causal features keep their effect direction under mean and resample ablation; which features clear depends on the intervention (2026-10-17: retitled after resample and clean arms; was "Causal features survive mean ablation when they are sparse; the verdict is intervention-dependent only for dense features")
- **Claim.** `panel7_v3_dev`, the 91 features clearing >= 1 channel under the on-manifold (cov) null with zero ablation, re-run with the atom set to its token-level corpus mean (null matched to the mean removal, same rows/seeds/draws; zero rerun reproduced 182/182 features exactly):
  - **69/91 still clear** under cov (70/91 iso); clearing cells 170 → 137.
  - Effect direction is preserved: zero-vs-mean cosine median 0.9968869277075498 (IQR 0.9765–0.9998).
  - Survival depends on density: token mu/top < 0.1, 56/62; >= 0.1, 13/29 (Spearman rho −0.4348, p 1.66e-05). Losses by model: Chronos-T5-Base 2/7 and Time-MoE 0/3 survive.
  - Mean ablation creates almost no new clears: 1 of 91 sampled non-clearing features.
- **Evidence.** Causal within-model (dev, exploratory). **Status:** measured once.
- **Reproduce.** `tsfm_lens/run_ablation_mode_check.py plan|launch|aggregate` → `runs/ablation_mode_check/ablation_mode_check.json`; `feature_ablation_fingerprints(intervention="mean")` (commit 67685d7).
- **Ref.** ROADMAP §43.2.
- **Score. 3/5.** Answers the standard "zero ablation is off-manifold" objection for most features.
- **Addendum (2026-10-17, ROADMAP §47.4): resample and clean-activation ablation.** On a regenerated v3 dev run that
  reproduces the recorded one bit-exactly (50/50 SAE training histories identical; the same 91 clearing feature indices;
  zero control 182/182 exact), the same 91 features under (a) resample ablation (activation at the matching token of
  another firing series; 90 scorable) and (b) subtracting the feature's decoder contribution from the clean activation,
  each with its own matched null:
  - Resample: 58/90 clear cov (69/90 iso); cells 166 -> 118; cosine to zero median 0.99078334224017; density split
    43/61 sparse vs 15/29 dense; 2/91 new clears in the sample.
  - Clean: 49/91 cov (63/91 iso); cells 170 -> 113; cosine median 0.9193719121363682 (q25 0.6717053797281081); density
    split 29/62 sparse vs 20/29 dense; 8/91 new clears in the sample (flag, not decomposed).
  - Across mean, resample and clean: 37/91 clear cov under all three, 79/91 under at least one, 12/91 under none.
  - Reading: mean and resample keep the effect direction; the clean intervention, which leaves the reconstruction error
    in place, agrees less. Most zero-ablation features keep an effect under some on-manifold intervention, but which
    ones survive depends on the intervention. **Status:** measured once (dev, exploratory).
  - Reproduce: `tsfm_lens/run_ablcheck_regen.py`, `run_ablcheck_compare.py`, `run_ablation_mode_check.py --arms
    mean,resample,clean` -> `runs/ablation_mode_check_v2/ablation_mode_check.json`, `runs/panel7_v3_ablcheck/repro.json`
    (commits b99c4ff, 02caf5c).

#### MP-11 · Training a fine-token model's SAE on its own tokens instead of window means fits the intervention space and preserves the forecast far better; comparison stays defined
- **Claim.** 512 v3 dev series (410 train / 102 held-out), 3 seeds per arm, matched optimizer steps, pre-registered
  rules; W = window-mean training (current library), T = the model's own tokens. TimesFM (tokens = windows) is the
  control: training matrices byte-identical, no metric differs beyond noise (D3).
  - Token FVU (held-out): Chronos-T5-Base block 3 0.2835459862397514 -> 0.04417716197644572 (noise
    0.019284019597174983), block 11 0.5469249152105949 -> 0.11733299141061297 (noise 0.05942832328811091);
    Chronos-Bolt block 2 0.4093 -> 0.1466, block 5 0.8422 -> 0.2349.
  - Token-level forecast preservation |dMASE|: Chronos-T5-Base block 3 0.30693666140238446 -> 0.04895734786987305,
    block 11 1.0201536019643147 -> 0.11304529507954915; Chronos-Bolt within noise (weakly powered).
  - Dead features fall (Chronos-T5-Base block 3 0.328125 -> 0.04513888888888889); features clearing the causal null
    rise (of 64: Chronos-T5-Base 2.67 -> 5.67 and 2.0 -> 6.0; Chronos-Bolt block 5 0.33 -> 5.33).
  - Costs: top-series overlap with a fixed TimesFM dictionary falls (Chronos-Bolt block 2 BH fraction
    0.3044093113906048 -> 0.24164541774501724, noise 0.018545472483144396; Chronos-T5-Base within noise); within-window
    slope is NOT more decodable from pooled T codes (D4 fails; Chronos-T5-Base block 3 R2 0.2546269088868454 vs
    0.2944049866869174, noise 0.05677055407107734), although T recovers a planted within-window direction W cannot
    (mock: |cos| 0.9881351590156555 vs 0.44555026292800903); a T dictionary applied to window means is off-distribution
    (Chronos-T5-Base window FVU 0.6133 vs 0.1069).
- **Models / layers.** Chronos-T5-Base encoder.block.3/.11, Chronos-Bolt encoder.block.2/.5, TimesFM stacked_xf.0/.18.
- **Evidence.** Descriptive (fit) and causal within-model (forecast preservation, ablation yield); dev, exploratory.
  **Status:** measured once; pre-registered verdict "recommend token training as the default"; not yet in the library
  (`sae/train.py` trains and persists codes from window means).
- **Reproduce.** `tsfm_lens/run_token_sae_check.py` -> `runs/token_sae_check/main/{cells,table.json,decisions.json}`
  (commit f9f3f11).
- **Ref.** ROADMAP §46, §46.1.
- **Score. 4/5.** Window-mean SAE training is a hidden handicap for 1-step models; fixing it changes fit, dead rate and
  causal yield, while cross-model comparison stays defined.

---

#### MP-12 · Window robustness needs matched extent, feature budget, and a separate task-information check
- **Claim.** On the same cached dev series and fixed middle layers, window
  concatenation is not a uniform similarity boost: at W128, TimesFM/Bolt raw
  CKA mean→concat is **0.30325622517858053 → 0.34915321043261105**, but
  PCA8 CKA is **0.2331308306660635 → 0.1338747277636627** and PCA32 is
  **0.26600981566893317 → 0.22532351841408854**. PCA is fit on training
  series only. Equal rank still leaves different spectra. Complete-context
  concatenation is exactly invariant to arbitrary window partition (max
  difference **0.0**); window-row CKA changes the observational unit.
- **Task check.** On fresh phase pairs, W32 Bolt mean-minus-concat phase-vector
  MSE at PCA8 is **0.43163991774494487**
  [0.40522112183278375, 0.4581688715844688], Holm4 **0.0004**. Raising the
  PCA budget substantially closes it, so this is not literal absence of phase
  information. The geometry ranking differs across the mixed-period and pure
  phase corpora; neither mean nor concat is a universal winner.
- **Models / layers.** Fixed TimesFM10/Bolt3/Chronos-2 block6 on 128 common
  cached dev series (64 train/64 test), identical trailing384-step extent;
  phase information check uses 48 fresh held-out generating-seed bundles.
- **Evidence.** Descriptive geometric and held-out predictive. The cached
  null grid is unadjusted/descriptive and does not condition on input/family.
- **Status.** Implemented, opt-in; existing pipeline defaults unchanged.
- **Reproduce.** `run_window_robustness.py`; `docs/window_robustness.md`;
  `runs/feature_search_20261111_window_cached_report/report.html` and
  `runs/feature_search_20261111_window_phase993000/`.
- **Ref.** ROADMAP §48.8.
- **Score. 4/5.** Makes pooling-dependent conclusions auditable without model
  inference and prevents window size from silently changing the comparison.

## G. Method results — confirmed negatives (what does not work, and why)

#### MN-01 · Representational similarity cannot detect model lineage or provenance
- **Claim.**
  - An untrained same-architecture pair (random-init Chronos-T5-Small vs
    random-init Base) shows CKA **0.878** and L2 gain 0.834/0.940.
  - A real, lineage-related trained pair (Chronos-T5-Small vs Base) shows
    CKA 0.734 and gain 0.637.
  - The untrained pair wins: L1 diff −0.1431, L2 diff −0.3029, p 0.0005.
- **Evidence.** Geometric / linear-translatable. **Status:** negative (closed) 2026-08-10.
- **Reproduce.** `configs/distill_positive_chronos_small_base.yaml`, `distill_negative_random_architecture.yaml`; `run_distillation_detection_test.py`.
- **Ref.** ROADMAP §6.3 (~6387–6443), §13 (~8262–8274).
- **Score. 5/5.** Real stakes for IP and provenance disputes: similarity numbers can reflect nothing but shared architecture.

#### MN-02 · Behavioral error-fingerprinting cannot detect lineage either
- **Claim.**
  - The magnitude channel ranks two random twins highest: 0.923 corrected
    (0.9719 before correction).
  - A genuine fine-tuned parent/child pair reads 0.594 [0.395, 0.745],
    *below* that null.
  - The shape channel separates trained from random (lineage 0.828 vs twins
    0.115), but not lineage from a shared recipe (a sibling pair reads
    0.878).
  - TimesFM vs its own twin scores 0.4977 while Chronos vs its twin scores
    0.1358, so the floor is model-specific.
- **Evidence.** Correlational, with controls. **Status:** negative (closed) for magnitude; shape stays confounded with recipe.
- **Reproduce.** `analysis/error_fingerprint.py`, `run_error_fingerprint.py`, `run_finetune_child.py`; `runs/lineage_pair`, `runs/distill_*`, `runs/null_*_random`.
- **Ref.** ROADMAP §6.3.1 (~6608–6712, corrected in place per §23.3 D2), §23.3 D2 (~17453–17685).
- **Score. 4/5.** A second, independent falsification of lineage detection, on an unrelated signal.

#### MN-03 · Crosscoder: joint training buys nothing
See SH-10 and SH-11. **5/5.**

#### MN-04 · SAE feature names from an argmax over ground-truth fields elect corpus provenance
- **Claim.**
  - `tier_realism_stress` is the most common name at 17–21 of ~51 rows per
    target (68% on one run).
  - Provenance dummies are defined on all 965 series, while structural
    fields are defined on only 374–555, so the argmax compares different
    sample sizes.
  - The 34 atoms sharing that one name are near-orthogonal (mean cosine
    0.029), so the naming scheme is the defect, not the dictionary.
- **Evidence.** Descriptive. **Status:** fixed by separating and residualizing (next entry).
- **Reproduce.** `runs/full_report_run_large/sae/meta.json`; `run_stage1_exit_check.py`.
- **Ref.** ROADMAP §25.1 (~19587–19649), §25.22.
- **Score. 4/5.** A trap for any probing pipeline on a corpus with a provenance split.

#### MN-05 · The residualization fix made it worse: 93% of headline cards were float-rounding noise
- **Claim.**
  - Before: 445/566 matched features (78.6%) and 11/11 headline rows named a
    provenance dummy.
  - Residualizing on provenance annihilates three binary flags that are set
    by exactly one archetype (oof R² 1.0000; the residual sd is ~1/1200 of
    the original).
  - Spearman then ranks the rounding noise as signal: `has_intermittency`
    ρ +0.203 raw became +0.615 "residualized".
  - 82/88 cards (93%) used these fields.
  - The same lesson recurred later: residualizing on archetype flagged 48/55
    concept parts as provenance-driven.
- **Evidence.** Method. **Status:** fixed (`min_residual_scale` 0.01; fields listed in `fields_not_separable_from_provenance`; residualize only on true confounds).
- **Reproduce.** `sae/ground_truth.py::best_ground_truth_matches_separated`; `runs/full_report_run_large/sae/meta.json`.
- **Ref.** ROADMAP §26 A1–A3 (~21889–21999), §37.11; CLAUDE_FULL §11.48.
- **Score. 5/5.** A control that explains *everything* is as dangerous as one that explains nothing, and its failure looks like a strong positive result.

#### MN-06 · A grounded small LLM narrator fabricates comparisons, and blacklist guards do not converge
- **Claim.**
  - Qwen2.5-1.5B was restricted to a measured evidence packet with a
    reject/retry guard.
  - Four consecutive guard-strengthening passes each produced a *new* class
    of unsupported claim:
    - synonyms;
    - misattribution to the other model (8/25 solo sentences at one point);
    - invented comparisons to the corpus median;
    - rankings on pairs with 0 scored comparisons.
  - Final pass: 4/6 generated summaries passed every guard and 2 of those
    were wrong, vs 6/6 deterministic sentences correct.
  - Acceptance must be measured per state: comparative 38–55%, solo 85–97%.
  - Accepted descriptions dropped effect sizes and added only lexical
    variation.
- **Evidence.** Method (generative reliability). **Status:** negative (closed). Deterministic composition is the default, and the feature narrator was removed.
- **Reproduce.** `sae/compare.py::check_synthesis_text`; `run_sae_compare.py`; `runs/full_report_run_4model/sae/comparison.json`.
- **Ref.** ROADMAP §26 C, §28.4–§28.18 (~24025–25073), §32.7c–d; CLAUDE_FULL §11.53.
- **Score. 5/5.** Transferable to any "LLM explains this feature" pipeline. For the "nothing was measured" state, use an allowlist, or skip generation.

#### MN-07 · Causal fingerprints cannot be a cross-model matching key (low-dimensional on the four-model run; ~5–6 of 9 dimensions on the seven-model runs)
- **Claim.**
  - The 9-channel ablation fingerprint has participation-ratio dimension
    2.09. The 64-step delta curve has dimension 1.04, with PC1 at 98.3%.
  - The mutual-nearest-neighbour match rate (37.6%) sits inside its null
    p95 (50.0%).
  - The full curve does worse: 15.3% vs 13.1% null at cosine 0.99.
- **Correction (2026-10-17, ROADMAP §47.5).** PR 2.09 reproduces exactly on the four-model run (2.0915412316960187, 201 features) but does not carry over. v3: 5.575534873891118 (covariance null, 91 features; PC1 share 0.2828) and 5.996530057329665 (isotropic, 238 features). v2: 1.0508376999675946, driven by one entry of 269.8 (null p95 near 0; PC1 0.9754); with unit-normalized rows (the geometry the atlas clusters) v2 is 6.383117072160696. The 9-channel fingerprint is therefore about 5-6 dimensional on the seven-model runs, not nearly one-dimensional; the closed negative (fingerprints as a cross-model matching key) rests on the match-rate test, not on this PR. Reproduce: `runs/dagger_remeasure/items_v2_v3.json` (commit e8f4012).
- **Evidence.** Causal fingerprints, compared cross-model. **Status:** negative (closed). Input-space transfer (SH-14) replaced it. CA-07 shows that the *per-item* effects are not only a level shift.
- **Reproduce.** Measured on `runs/full_report_run_4model/sae/*_ablation.json`.
- **Ref.** ROADMAP §30.2 (~25645–25701).
- **Score. 4/5.** What a feature *does* can be degenerate across features even when what it *detects* is not.

#### MN-08 · Grouping features is intervention-relative: injection-space roles fail in ablation space
- **Claim.**
  - Projecting injection-clustered roles into ablation space gives mean
    silhouette **−0.235**, negative at 13/13 targets.
  - Re-clustering natively gives **+0.450**. Example: TimesFM xf.16 goes
    from −0.326 to 0.717.
- **Evidence.** Causal within-model. **Status:** confirmed; roles were superseded by ablation-space concepts.
- **Reproduce.** `sae/concepts.py::build_concept_matrix`; `runs/full_report_run_4model`.
- **Ref.** ROADMAP §30.1 (~25535–25642).
- **Score. 5/5.** A feature "role" is not intrinsic; it depends on which intervention defines it.

#### MN-09 · Naming a cluster by its argmax channel fails a permutation null
- **Claim.**
  - Member-agrees-with-cluster-name rates are 39.3% vs null p95 43.8% for
    old roles, and 30.3% vs 30.3% for new concepts.
  - A few channels dominate the population's magnitude, which gave 21
    distinct names for 201 features.
  - Contrastive z-ranked names give 101/201.
- **Evidence.** Correlational. **Status:** negative (closed).
- **Reproduce.** `sae/concepts.py::compose_name`.
- **Ref.** ROADMAP §30.1 M2, §30.7 (~26416–26533).
- **Score. 3/5.**

#### MN-10 · Silhouette rewards peeling off a single outlier
See SH-20. **4/5.**

#### MN-11 · A pooled cross-model layer rule fails as a within-model selector
- **Claim.** The rule from DE-06 picked Chronos-T5-Base `encoder.block.0`, the layer with the *lowest* tuned-lens R² (0.433) of 12.
- **Evidence.** Correlational. **Status:** negative (closed).
- **Reproduce.** `runs/medium_run_chronos_base`; `analysis/layer_selection.py`.
- **Ref.** ROADMAP §6.2 (~3648–3671).
- **Score. 4/5.** Pooled cross-architecture statistics do not transfer to single-model rules.

#### MN-12 · `factor_emergence` layer selection fails because meta fields are readable at layer 0
- **Claim.**
  - Recall@budget was 0.00 on Chronos-T5-Small.
  - Provenance and archetype fields are linearly readable from the first
    layer of every architecture.
  - A peak-weighting fix that worked on the synthetic plant made TimesFM
    worse (0.20→0.00).
- **Evidence.** Method. **Status:** negative (closed).
- **Reproduce.** `analysis/layer_screen.py::select_factor_emergence`; `runs/layer_screen_bakeoff_v2.json`.
- **Ref.** ROADMAP §6.1.1 (~2761–3067); CLAUDE_FULL §11.18.
- **Score. 3/5.** A fix validated only on synthetic data can be the mirror-image wrong fix on real data.

#### MN-13 · Extending the causal horizon from 64 to 128 steps adds nothing
- **Claim.**
  - The new band (steps 64–127) correlates with the old band at Pearson
    0.949 and Spearman 0.977, matching the pre-registered kill criterion.
  - Clearing cells are unchanged (53 vs 53).
  - `horizon_shape_far` discriminability falls 30% (2.7727→1.9340).
- **Evidence.** Causal within-model. **Status:** negative (closed).
- **Reproduce.** `runs/full_report_run_4model/sae/TimesFM/stacked_xf_10`; §33.6 acceptance.
- **Ref.** ROADMAP §33 (~29774–30119).
- **Score. 4/5.**

#### MN-14 · Most visually prominent "interesting features" were causally null
- **Claim.**
  - Of 565 rendered SAE candidates, **364 (64.4%)** cleared 0 of 9 causal
    channels, covering 724/1297 panels.
  - The share clearing nothing is 54.5–71.1% per model.
- **Evidence.** Causal within-model. **Status:** fixed (null panels collapsed); held-out confirmed (2026-10-02).
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** On v2 (672 candidates) the share clearing 0/9 channels is 0.5639880952380952 under the legacy null and **0.8363095238095238** under the profile-matched null (scorable only: 0.4068825910931174 → 0.7773279352226721). By model: Chronos-2 0.389 → 0.799, Chronos-Bolt 0.611 → 0.847, Sundial 0.454 → 0.843, TimesFM 0.754 → 0.850. The claim strengthens: most activation-prominent features are causally null.
- **Held-out (2026-10-02, K2, CA-11).** Registered claim `majority_prominent_features_causally_null` **confirmed** on private v2 epoch 3, 7 models: private causally-null share 0.835 of 927 features (one-sided 95% lower bound 0.8166 > 0.5; dev 0.8102; p_holm 0.0005). Status: held-out confirmed.
- **Reproduce.** `runs/full_report_run_4model/sae/*_ablation.json` (`n_channels_clearing`).
- **Ref.** ROADMAP §32.7b (~28480–28675).
- **Score. 4/5.** Detecting features by activation alone is not evidence that they matter: most high-activation features do nothing causally.

#### MN-15 · The SAE's own reconstruction error moves the forecast 1.7–3.5× more than removing a feature
- **Claim.**
  - Median reconstruction cost is 0.139 context sd, against a median feature
    effect of 0.0395.
  - 23.0% of panels have a feature effect below 1/10 of the reconstruction
    cost.
- **Correction (2026-10-17, ROADMAP §47.5).** The 3.5× is a four-model, lenient-null number. Same definition (max over horizon, all panels, no clearing gate; four-model reproduces exactly, 0.03947468804168946 vs 0.13896064178421366) on the seven-model runs: v2 effect median 0.09650917675554359 vs reconstruction 0.21386207843123203 (ratio 2.2159766109386854, 2801 panels; 0.13423777222420563 of panels below a tenth); v3 0.15631556301374533 vs 0.2604575236137563 (ratio 1.6662289959628231, 2722 panels). The direction holds; the factor is 1.7-2.2 on the current runs. Reproduce: `tsfm_lens/run_dagger_remeasure.py json-items` -> `runs/dagger_remeasure/items_v2_v3.json` (commit e8f4012).
- **Evidence.** Causal within-model. **Status:** confirmed on run.
- **Reproduce.** `unpatched` array in `runs/full_report_run_4model/sae/*_ablation.json`.
- **Ref.** ROADMAP §32.7b I4 (~28590–28645).
- **Score. 4/5.** A fidelity ceiling on how much any per-feature causal story can carry.

#### MN-16 · The forecast-preservation SAE check has a tokenizer-granularity confound
- **Claim.**
  - When the token width is finer than the 32-step window, window-level
    patching broadcasts one vector over 32 tokens.
  - Chronos-T5-Base shows window ΔMASE +3.8972 (sd 0.242) vs token ΔMASE
    +0.2457 (sd 0.2113).
  - Chronos-Bolt b.4 has the best fidelity in its run (0.955), yet window
    ΔMASE +0.378 vs token −0.135.
  - The earlier headline "reconstruction beats the clean forecast" (−0.346)
    is **retracted**: it replicated in 1/5 seeds at −0.112. The TimesFM
    5-seed floor is +0.1704 ± 0.1212.
- **Evidence.** Causal within-model. **Status:** the retraction stands; the admission gate reads token granularity.
- **Reproduce.** `run_sae_repeat_variance.py`; `runs/medium_run_chronos_base/sae/repeat_variance.json`.
- **Ref.** ROADMAP §6.2 (~3781–3804), §13 (~8437–8533), §31.2 (~27141–27176).
- **Score. 4/5.** A single-seed SAE metric can be off by 2–24× its own noise floor.

#### MN-17 · Generator-counterfactual concept mediation: NO-GO twice
- **Claim.**
  - The design uses draw-neutral dose ladders on 5 seeded generator knobs,
    against a matched random-feature null.
  - First try (4 targets): **0/98** survive BH. 14 respond uncorrected vs
    ≤4.9 expected (binomial p 0.000376).
  - Registered retry (9 more targets, pooled BH, n_null 10000): **0/151**,
    with minimum q 0.4548, and 11 vs ≤7.55 uncorrected (p 0.1366).
  - A knob moves a large fraction of *any* alive feature about as much as it
    moves the targeted concept, so what is missing is specificity, not
    signal.
- **Evidence.** Correlational input-response test. **Status:** negative (closed). Mediation (P6b) was not built, and the knob claim family is empty.
- **Reproduce.** `concepts.cf_enabled`; `sae/counterfactual.py::run_counterfactual_response`; `build_pipeline/counterfactual.py`; `sae/counterfactual_response.json`.
- **Ref.** ROADMAP §37.9 (~36600–36722).
- **Score. 5/5.** A confirmed negative on validating SAE concepts against synthetic ground-truth knobs, useful to anyone trying the same.

#### MN-18 · Absence of agreement is not disagreement
- **Claim.** P5b's first pass called 109 of 288 tests "acts differently" because it failed an upper-tail (p95) floor. With a lower-tail (p05) test on both statistics, the count is **9**; 32 are "no specific agreement".
- **Evidence.** Method. **Status:** fixed.
- **Re-measured under the profile-matched null (2026-09-30, MN-29; v2 copy, same features and seed).** The lower-tail rule is unchanged. Under the profile null, 10 of 372 tests are 'acts differently' (legacy 64 of 1223).
- **Reproduce.** `sae/shared_input_agreement.py`.
- **Ref.** ROADMAP §37.8 (~36309–36447); CLAUDE.md §8.
- **Score. 4/5.**

#### MN-19 · "Same inputs" must be a top-k overlap test, not a whole-series correlation
- **Claim.**
  - Two sparse SAE features can have ρ 0.80 with zero shared top-20 series.
  - ρ 0.09 passed BH at n=965.
  - Cross-fitted residualization split a binary field's ties by fold,
    moving `has_random_walk` from ρ 0.833 to 0.328.
- **Evidence.** Method. **Status:** fixed (hypergeometric top-k overlap + BH).
- **Reproduce.** `sae/concept_profiles.py`; commits `384c7cd`, `f9b2372`, `8feb22f`.
- **Ref.** ROADMAP §37.11 (~36913–36932).
- **Score. 4/5.**

#### MN-20 · More dictionary or more data does not fix SAE death, and more real data makes it worse
- **Claim.**
  - Going from 112 to 448 to 1120 offered atoms yields 51→61→68 alive.
  - 10× more rows leaves the dead rate flat (0.961→0.938) while fidelity
    improves.
  - Real Monash rows have *lower* effective dimension than the synthetic
    corpus (1.76 vs 2.66 for TimesFM) and a higher dead rate (0.9755 vs
    0.9593).
- **Evidence.** Method. **Status:** negative (closed).
- **Reproduce.** `runs/crosscoder_stage0_bigdata/crosscoder_stage0.json`; `runs/sae_corpus_diversity_check.json`.
- **Ref.** ROADMAP §6.2.1 (~4396–4438), §23.2 A1 (~16386–16424).
- **Score. 3/5.**

#### MN-21 · Reviving dead features hurts ground-truth alignment in TimesFM at every depth
- **Claim.**
  - Dead vs revived recipe at TimesFM xf.2/6/10/16/18: revived − dead is
    −0.030, −0.080, −0.082, −0.132, −0.114.
  - The size of the hurt tracks effective dim (ρ 0.9, one-sided p 0.0417).
  - Chronos-T5-Base is neutral or slightly helped (+0.2294→+0.2362).
    Chronos-2 is already alive.
  - Across 11 targets, the margin over null fell at 9, and the fall is
    predicted by growth in alive atoms (r −0.9686; doubling alive atoms
    gives ×0.757 margin).
  - The causal (ablation) signal fell much less (×0.65–0.76), or rose
    (Chronos-2 b.6 2.27→4.06× chance).
- **Evidence.** Correlational vs causal within-model. **Status:** exploratory (one seed per depth).
- **Reproduce.** `configs/sae_revival.yaml`, `configs/full_report_run_large_revived.yaml`; `runs/sae_q2_depth_sweep/`, `runs/full_report_run_large_revived/sae/`.
- **Ref.** ROADMAP §22.0 (~15704–16072), §26 E (~23114–23231).
- **Score. 5/5.** Inverts the default "dead features are pure pathology" assumption. Judge liveness tradeoffs on the causal channel, not the correlational one.

#### MN-22 · Uncalibrated corruption charts show perturbation energy, not sensitivity; cross-model rankings survive calibration
- **Claim.**
  - Uncalibrated, `level_shift` dominates by 18.22× / 10.55×. Calibrated by
    input energy, that drops to 5.18× / 3.01×, and Chronos's top corruption
    flips to `frequency_shift`.
  - Chronos-T5-Base remains more sensitive than TimesFM on 8/9 corruptions
    either way.
- **Evidence.** Causal within-model. **Status:** confirmed.
- **Reproduce.** `runs/medium_run_chronos_base_l3_input_energy` vs `runs/medium_run_chronos_base`.
- **Ref.** ROADMAP §23.2 A2 (~17128–17193).
- **Score. 3/5.**

#### MN-23 · MASE's denominator floors out on intermittent series
- **Claim.** At intermittency 0.8, about 80% of targets are zero and MASE appears to *improve*. This propagates into every MASE-based metric (L0, ΔMASE, L3, SAE preservation).
- **Evidence.** Metric caveat. **Status:** floored `_mase_scale`; a standing caveat for zero-heavy families.
- **Reproduce.** `analysis/stats.py::_mase_scale`; `runs/param_sweep_intermittency_rate.json`.
- **Ref.** ROADMAP §7 (~7058–7086).
- **Score. 3/5.**

#### MN-24 · A tiny precision mismatch never moved an aggregate but reorganized 71% of a partition
- **Claim.**
  - The clean cache was captured under bf16 autocast while predictions ran
    in fp32.
  - As a result, TimesFM's self-patch read 0.002144730417057872, not 0.
  - Aggregates moved by at most 0.66% of CI width.
  - But 29/41 (71%) of role assignments at TimesFM xf.10 changed.
  - The bf16 Chronos models were inert, so the control passed by
    architectural accident.
- **Evidence.** Method. **Status:** fixed (`capture_raw_tokens(autocast=False)`).
- **Reproduce.** `extraction/extract.py::capture_raw_tokens`.
- **Ref.** ROADMAP §25.23, §26 D1 (~22622–22831); CLAUDE_FULL §11.49.
- **Score. 4/5.** Aggregate noise-floor checks do not certify clustering-derived artifacts.

#### MN-25 · The zero-code adapter path fails on most real TSFM checkpoints before it starts
- **Claim.**
  - 2 of 8 probed checkpoints resolved (Timer, TimeMoE).
  - 5/8 crashed inside `transformers.AutoConfig`: Toto, MOMENT, Lag-Llama,
    Moirai and TTM all ship standalone packages.
  - VisionTS has no loadable checkpoint.
  - Separately, argmax span discovery cannot see Lag-Llama's 84 disjoint lag
    reads (contiguity 1.000).
- **Evidence.** Descriptive. **Status:** open (known limit).
- **Reproduce.** `run_probe_sweep.py`; `docs/probe_sweep.json`.
- **Ref.** ROADMAP §34 D1 (~32739–32848), D2 (~32996–33029); CLAUDE_FULL §11.58.
- **Score. 3/5.**

#### MN-26 · Other closed negatives, recorded briefly
- **GPD tail-fit p-values.** Only 2/16 fell within a factor of 2 of exact
  5000-draw p, so they were dropped for adaptive redraw. §37.6. 2/5.
- **Fine-tuning plasticity vs the layer selector.** No significant
  correlation (n=6; ρ −0.771, p 0.072). §20 H6. 2/5.
- **UMAP for the 9-D concept space.** PCA PC1–3 already carry 75.0%, so
  UMAP was not used. §32.14. 2/5.
- **Rescaling ablation overlays.** Panels under 1 px went only
  77.7%→70.8%; effects are genuinely small (median 0.0395 context sd).
  §32.6. 2/5.
- **Misfit detector.** It compared member-to-centroid against pairwise
  cosine (a +0.248 gap), reading "no misfits" when there were 10/192. §32.13b. 2/5.
- **Raw-scale heatmap.** It hid cleared effects; raw spread across channels
  is 1285× vs 2.9× in null units. §31.4. 2/5.

#### MN-27 · CPU- vs GPU-trained SAE replicates are not interchangeable
- **Claim.** CPU retraining with the same seeds kept 14/19 stable but flipped concepts 14 and 16, with different per-model ceilings. Float64 re-ranking of the GPU replicates reproduced the original exactly, so the difference is training nondeterminism.
- **Evidence.** Descriptive. **Status:** GPU replicates are canonical.
- **Reproduce.** `sae/concept_stability.json` on GPU vs CPU copies.
- **Ref.** ROADMAP §37.5 (~35883–35892).
- **Score. 3/5.**

#### MN-28 · Looking inside does not predict forecast failure beyond the model's own quantile width (dev, v2)
- **Claim.**
  - **Setup (U1).** Per model, a series' log MASE and its failure flag (MASE above the seasonal-naive MASE) are predicted from free output-only features: quantile width, flatness and catch22 of the context.
  - **U1 result.** Adding internals (SAE concept-family activations at the last window, per-series lens convergence depth, residual norm at the crystallization layer) gives no gain for any model. Cross-fitted, 936 series (29 MASE-unreliable excluded), series bootstrap n_boot 1000.
  - **U1 gain, log-MASE Spearman:**
    - Chronos-2 −0.004538525487737699 [−0.011114460635546608, 0.001579303153024544];
    - Chronos-Bolt 0.0015074468712488187 [−0.00431583261828489, 0.007117558727123265];
    - Sundial 0.004392437244468139 [−0.002947812538624474, 0.011878917521050776];
    - TimesFM −0.004004029536078035 [−0.01042833860707762, 0.001947542083900313].
  - **U1 gain, failure AUROC:**
    - Chronos-2 0.0009112349914237594 [−0.021583010754270764, 0.02161872865773572];
    - Chronos-Bolt 0.017620172849398652 [−0.00303947104788612, 0.039507799948734226];
    - Sundial 0.002595258692529301 [−0.01689299387157657, 0.022535002261753423];
    - TimesFM 0.005844155844155874 [−0.008952987027037031, 0.018686576182679344].
  - **Why there is no room.** The baseline alone reaches Spearman 0.8095725334786167–0.8390565740367988.
  - **U2 (route each series to the model with the lowest predicted MASE).** Realized mean MASE:
    - baseline routing 1.6841631168977191;
    - baseline+internals routing 1.6973776593724759;
    - best single model (Chronos-2, chosen on the same rows) 1.6908591055335143;
    - oracle 1.470673442284903.
  - **U2 gap (baseline − internals):** −0.013214542474756893 [−0.02965822172605901, 0.0026436959894803815], p 0.122. Internals routing is, if anything, worse.
- **Evidence.** Predictive (behavioral), dev only. **Status:** confirmed negative on dev.
  - Chronos-2 had only the SAE group (skip lens unavailable, DE-03). TimesFM lacked the crystallization norm (its mean lens never crystallizes).
  - Lens depth is near-degenerate on several models: TimesFM 104/965 converge; Sundial 838/965 sit at depth 1.0.
  - The failure task is imbalanced (rate 0.094–0.174).
  - CIs hold the out-of-fold predictions fixed.
  - One of 36 per-group CIs excludes 0 (the Chronos-Bolt crystallization-norm AUROC gain). Given the multiplicity, it is not claimed.
- **7-model panel (2026-10-01, `runs/panel7_v2_dev`, ROADMAP §38.4.4).** Routing stays negative: internals routing 1.9638372652500882 vs baseline routing 1.947559678401894 (gap p 0.258), and both lose to always using Chronos-2 (1.8559481862647405; p 0.001). U1 is a **narrow partial positive**: SAE-family activations add to the free baseline's error ranking for the two weakest models only (Timer +0.030310590743556798 [+0.018647995847319663, +0.04301877629706014], Time-MoE +0.011873989992855383 [+0.004531001655895828, +0.01920766804668001]); no failure-AUROC gain excludes 0. Both are registered for the K2 private look; still dev. Robust to fold assignment: across 5 fold seeds the gains are 0.030310590743556798–0.036080231129527296 (Timer) and 0.010632868442613463–0.015195058383651538 (Time-MoE), with every CI excluding 0; no other model's CI excludes 0 at any seed.
- **Held-out (2026-10-02, K2, CA-11).** Both registered U1 claims **confirm** when the frozen procedure (SAE checkpoints, families, baseline, folds, seed) is refit on private v2 epoch 3 (n 935, 36 MASE-unreliable excluded). Log-MASE Spearman gain: Timer 0.0446 [0.0292, 0.0605], p_holm 0.002 (baseline 0.7568 → 0.8013); Time-MoE 0.011 [0.0032, 0.0194], p_holm 0.012 (0.835 → 0.846). The Timer private gain exceeds every dev fold seed (0.0303–0.0361). Predictive, not causal. The entry's headline negative still holds for the four strongest models and for routing; the narrow positive for the two weakest models is held-out confirmed.
- **Reproduce.** `python run_reliability_from_internals.py --run <copy of runs/concept_atlas_v2 with lens rerun at lens.depth_max_series=965> --out reliability_v2.json`; `analysis/reliability_from_internals.py`.
- **Ref.** ROADMAP §38.4.4.
- **Score. 4/5.** Same verdict as PM-12, now for internals: for "when should I distrust this forecast", the free quantile band already carries what the internals carry. A clean practitioner-facing negative, and it bounds claims of practical use from interpretability.


#### MN-29 · The ablation battery's legacy null is lenient: a known-answer forecaster fails the gate with it and passes with a profile-matched null
- **Claim.**
  - **The flaw.** The legacy random-direction null (`sae/response.py`, `null_magnitude`) removes ONE uniform amount from every token: the chunk's mean |z| over all tokens and rows, zeros included, averaged over the chunk's features. The feature ablation removes z_f(t)·w_f, which is large exactly on the feature's own top-firing rows. A strong or dense atom therefore clears the null on size alone. The same sizing is used by the shared-input agreement step's `own_effect_null`.
  - **Known-answer test.** The planted forecaster `mock_planted` was run at dose 1, seeds 0–4. Seed 0 tuned `entanglement_min`; seeds 1–4 are held out. The gate (fixed before any data) is control-layer FPR ≤ 0.10 and sensitivity ≥ 0.5.
    - Legacy null: **stop on all 5 seeds**. Per-cell FPR 0.2483974358974359, 0.20353982300884957, 0.21862348178137653, 0.22468354430379747, 0.16818181818181818; per-feature FPR 0.338–0.465.
    - Opt-in `sae.ablation_null: profile_matched` (each null draw removes, at every token, |z_f(t)|·‖w_f‖ along a random decoded direction; the removal profile is the feature's own and only the direction is random): **pass on all 5**. Per-cell FPR 0.0, 0.0, 0.002844950213371266, 0.0, 0.0. Sensitivity 1.0, 0.7, 0.8333333333333334, 0.8888888888888888, 0.9.
  - **Seed 0 downstream, legacy vs profile:**
    - sub-null decoys cleared 4/4 vs 0/4;
    - input-only decoys cleared (raw) 2/4 vs 0/4;
    - atlas ARI 0.3357664233576642 vs 0.46938775510204084;
    - sharing-class accuracy 0.36363636363636365 vs 0.7142857142857143;
    - transfer accuracy 0.6153846153846154 vs 0.4444444444444444;
    - control-layer atlas rows 151 vs 28.
  - **Real data, one target.** On `concept_atlas_v2` TimesFM `stacked_xf.12` (18 scorable candidates, same seed, on a copy):
    - the legacy re-run reproduced the recorded artifact exactly: 29 clearing cells, 9 features;
    - profile-matched: **9 cells, 4 features, against a chance level of 8.1 cells** (0.05 × 162).
- **Evidence.** Known-answer instrument validation (planted ground truth), plus one real target.
- **Status:** the flaw is confirmed. Its real-data extent is open: a full v2 re-score under the matched null is running.
- **Real-data extent (2026-09-30, v2 copy; ROADMAP §38.1.8).**
  - Clearing features 293 → 110, cells 1348 → 254.
  - Against the leave-one-draw-out empirical chance rate (≈2% per cell, not the nominal 5%), that is 13.0006× → 3.0579× chance, with 20/28 targets BH-significant under the matched null.
  - Per-target concepts 9 → 0 (low power at k = 8).
  - Atlas 27 → 7, convergent 13 → 0.
  - A covariance-shaped ('on-manifold') matched null clears even fewer (106 cells, 1.1277× chance), so the isotropic matched null is not over-strict.
  - The affected entries (MP-07, MN-14, SH-14–17, SH-20, CA-06, CA-08, CA-10, MN-18) each carry a 're-measured' line. **Until it lands, every causal-feature count and every downstream concept number in B.2/C was measured with the lenient null and is an upper bound.** This includes MN-14's 64.4%, the per-target concept counts, the atlas, transfer, agreement (CA-06) and sharing classes.
- **Caveat.** In the planted world, a random decoded direction overlaps the planted span more than a p_perp atom does, so the matched null may be over-strict there (sensitivity falls to 0.7–0.9, and weak true effects are never detected).
- **A second mislabel (2026-10-01, `runs/panel7_v2_dev`, 7 models, profile-matched null, ROADMAP §38.3.4).** The battery's log line called `0.05 × cells` 'expected by chance'. The measured leave-one-draw-out rate is 0.003–0.04 per cell, so the 7-model run read as 442 / 459.9 = 0.96× chance in the log; against the empirical expectation it is 442 / 178.75 = **2.4727×**. The report never rendered the comparison. Fixed by labelling both numbers and rendering clears vs empirical chance per target (branch `chance-render`).
- **Reproduce.** `python run_known_answer.py --config configs/known_answer.yaml --null mean_magnitude|profile_matched` (seeds 0–4, dose 1); `sae/response.py::_profile_matched_null_replacement`; `tests/test_*profile*`.
- **Ref.** ROADMAP §38.1.7.
- **Score. 5/5.** The known-answer test did its job: it caught a lenient statistic behind the project's causal claims, and the held-out seeds show the fix.
---

#### MN-30 · The shared-input causal-agreement rung (L5) misses most truly shared concepts across architectures, and can call a shared concept "acts differently"
- **Claim.**
  - Known answer: two `mock_planted` architectures that differ in width (64 vs 96), depth (5 vs 7) and residual basis (random rotation), with planted shared and decoy concepts, 5 scored seeds.
  - The current rung reads a shared single-direction concept "same causal effect" in 0.4 of seeds, and a shared concept spread over 3–4 directions in 0.0.
  - No variant reaches the pre-fixed gate (sensitivity ≥ 0.5 on both): set ablation 0.2 / 0.0, a 2× larger U 0.6 / 0.0, a 4× larger U 0.4 / 0.0, partial agreement as its own rung 0.6 / 0.2.
  - Decoys are never called "same" (0/9 under every variant). So "same" is specific but insensitive, and an absence of "same" across architectures is uninformative.
  - A shared, same-sign pure-dispersion concept reads "acts differently" in 3/10 directed tests (diagnosis open).
- **Diagnosis of the false disagreement (2026-10-01, branch `l5-known-answer` 0953ff3).**
  - In every false call on the shared pure-dispersion concept, statistic (i) fires (level concordance, e.g. −0.999 against both floors' p05 of about −0.49 and −0.46), while `level` clears its own null on NEITHER side. Statistic (ii) stays high (0.887–1.0).
  - Level concordance was compared on two sub-null residuals that happen to be deterministically opposite in sign. There is no sign inversion on the dispersion channel.
  - The truly shared a and b cases never read "acts differently" (0/10 each).
  - Fix (opt-in `concepts.agreement_require_defined_firing`): a below-p05 statistic counts only if it is defined, meaning (i) needs `level` to clear on both sides, and (ii) needs both sides to clear a shape-mask channel.
  - With the fix, false disagreement goes 3/30 → 0/30 (dose 1) and 4/30 → 0/30 (dose 2). The opposite-effect decoy is still caught (2/10, unchanged), and the gate's sensitivity numbers are unchanged.
  - **Real data:** 4 of `panel7_v2_dev`'s 22 'acts differently' tests have an undefined firing statistic. The other 18 are genuine: level clears on both sides with negative concordance (−0.347 to −0.727).
- **Models / layers.** Mock pair (ArchA `blocks.2`, ArchB `blocks.4`).
- **Evidence.** Known-answer validation of a method (planted ground truth, decoys, held-out seeds). **Status:** measured; diagnosis of the false disagreement open.
- **Held-out (2026-10-02, K2, CA-11).** Consistent with the insensitivity: of 11 registered L5 claims, 5 are not testable on private data (one side does not clear its own null) and 0/6 confirm.
- **Reproduce.** `run_l5_known_answer.py`, `configs/l5_known_answer.yaml` (branch `l5-known-answer`, f832798); `runs/l5ka/scored_dose{1,2}/l5_known_answer_aggregate.json`.
- **Ref.** ROADMAP §38.3.4 (L5 known-answer bullet).
- **Score. 5/5.** Decides how to read every cross-model "same causal effect" claim: positives stand, negatives do not, and SH-22's "only the Chronos family" is a sensitivity limit, not a finding about lineage.

#### MN-31 · Causal MASE blame separates "fires on bad series" from "causes bad forecasts", but single-feature *help* attribution fails its known-answer gate at k = 32
- **Claim.** Forecast-repair R0 used a planted mock with harmful, helpful, decoy and side-effect concepts whose sign on MASE holds by construction. Dose 1, seeds 0–4, k = 32, BH over about 136 features.
  - **Pass:** harmful blame 4/5; decoy 5/5 (corr_MASE +0.076 to +0.197, yet |causal ΔMASE| ≤ 0.0057, p_bh ≥ 0.97); held-out zero-edit of harmful 5/5 (firing test series −0.0274 to −0.0922, CIs exclude 0); side-effect harm on weak-firing test series 5/5 (+0.0047 to +0.0256).
  - **Fail:** helpful blame 2/5 against 0.8 needed. The helpful atom ranks 2nd–6th by |effect| with the right sign and a CI excluding 0 (+0.0356 to +0.0779), but z ≈ 2.2–3.6 does not survive BH over the dictionary. One seed failed SAE recovery (cosine 0.798).
  - The null is not biased positive: the median null mean over features is about 0.
  - Unplanted split atoms carry large shares of the planted effects (e.g. −0.081 vs the planted −0.155), so real-data blame likely needs atom sets.
  - All-ones edits change the forecast by exactly 0.0 in every cell.
- **R0b (2026-10-02, pre-registered: fresh seeds 5–9, n 1200, k 64, a two-stage screen to a 16-feature BH family).** The gate fails again, with the failure moved to the other concept: helpful 5/5, **harmful 2/5** (stage-1 z 0.63–1.55 in 3 seeds; stage-2 ΔMASE always negative, e.g. −0.0404 [−0.0609, −0.0220] at p_bh 0.471). The decoy's literal 0/5 is a pre-registration error: a causal screen drops a zero-effect decoy by design, and the decoy was never blamed. **Held-out zero-edits of the harmful atom lower test MASE in 10/10 seeds across R0 and R0b** (R0b −0.0111 to −0.0904, all CIs exclude 0). Per the pre-registration, §39 stopped at R0. Reading: single-feature blame significance against an equal-energy random-direction null is at its power limit, while held-out repair itself works on the known answer. Re-basing the gate on held-out improvement is a new design that awaits a user decision.
- **Models / layers.** `mock_planted` (`vocabulary: repair`), blocks.2.
- **Evidence.** Known-answer validation (causal within-model on a planted model; held-out edits behavioral). **Status:** negative (closed at R0; closed for good by the user 2026-10-03): R0 and R0b both fail their pre-registered gates.
- **Reproduce.** `python run_repair_known_answer.py --config configs/repair_known_answer.yaml` (commit 5a8077f); R0b: `--config configs/repair_known_answer_r0b.yaml --two-stage` (commit 6c94c66); `analysis/repair_blame.py`, `analysis/repair_edit.py`.
- **Ref.** ROADMAP §39.7–39.8.
- **Score. 3/5.** Blaming a feature for errors needs causal ablation, not correlation: the decoy correlates with error in 5/5 seeds and has no effect. Crediting a single feature for *good* forecasts is underpowered at realistic sizes, and that bounds the "what causes good predictions" half of the question.

#### MN-32 · A subspace redesign of the shared-input agreement test also fails its pre-fixed known-answer gate; the limit is the effect-to-removed-amount ratio, not member selection
- **Claim.** V3-C (ROADMAP §41): subspace ablation of a concept's member set on both sides, a subspace-matched null (an isometric random frame, same per-token removed amount), series-bootstrap CIs against matched floors, TOST reported. Gate fixed before any result (commit 0b0d723): sensitivity ≥ 0.8 for each planted shared concept, pooled false agree ≤ 0.05, seeds 1–4 held out, dose 1 primary.
  - Held out, dose 1: main sens 0.5 (single direction) / 0.375 (distributed), false agree 0/14; V2 (k = 100) 0.5 / 0.375; V3 contrast subspace 0.0 / 0.0. Dose 2: main 0.5 / 0.5, V2 0.5 / 0.375, V3 0.0 / 0.0. **Gate FAIL on every variant.** False differ on shared concepts 0/16 everywhere; the opposite-effect twin reads `differ` 0.25 (main, V2).
  - **Ceiling (seed 0, dose 1):** ablating the answer-key planted distributed subspace itself does not clear its size-matched null (ArchA real 0.0078 vs null q95 0.0169, rank-p 0.467; ArchB 0.0080 vs 0.0164, rank-p 0.267), while the planted single direction does (0.0160 vs 0.0048; 0.0180 vs 0.0042). Single-concept misses (seeds 3, 4) come from SAE atoms mixing the concept with benign mass (null q95 ≈ 4× larger).
- **Reading.** Per-series "same causal effect on the same inputs" across architectures is not measurable by size-matched ablation at these effect sizes; positives stay specific (0/14 false agree). Cross-model effect sharing is therefore tested at the concept-profile level instead (V3-B centroid agreement, universal-family presence).
- **Evidence.** Known-answer validation of a method. **Status:** negative (closed by the §40 stop rule).
- **Reproduce.** `run_l5_known_answer.py` then `run_l5_subspace_known_answer.py` with `configs/l5_known_answer_v3.yaml` (merge e72e93e); gate in `analysis/l5_subspace_gate.py`.
- **Ref.** ROADMAP §41.2.
- **Score. 4/5.** Closes the question MN-30 left open: the instrument, not the data size, is the limit.

#### MN-33 · "Model m has causal features of effect family F beyond chance" fails its pre-fixed known-answer gate under both nulls
- **Claim.** V3-B family-presence claim, gate committed before any result (27e8e8b): K1-scale `mock_planted` (600 series, dose 1), family centroids from recovered planted atoms, held-out seeds 1–4, one-shot confirm on the sealed private_test split; sensitivity ≥ 0.8 over planted (family, model) pairs, false-claim ≤ 0.05 over absent pairs plus a loud same-shaped decoy null.
  - `profile_matched`, count statistic: sensitivity 0.1667 (6/36), false 1/45. `profile_matched_cov`: 0.0556 (2/36), false 0/54. V1 (min_cosine 0.7): 0.0278, false 0/28. V2 (at-least-one statistic): 0.4167 (15/36), false 0/45. **All FAIL.**
  - Diagnosis: about 10 testable features per pair on a 64-series private sample, and random directions already put 16–40% of features at cosine ≥ 0.5 to a family centroid; 11/33 pairs had raw p < 0.05 but only 6 survived Holm.
- **Reading.** Specific but underpowered. "Every model has level-raising, level-lowering and seasonality-damping causal features" (CA-10 addendum) stays a descriptive, exploratory statement; it is not registered in V3 (`panel7_v3.yaml` has `register_family_presence_claims: false`).
- **Evidence.** Known-answer validation of a method. **Status:** negative (closed: two declared variants tried).
- **Reproduce.** `run_family_presence_known_answer.py`, `configs/family_presence_known_answer.yaml`, gate `analysis/family_presence_gate.py` (merge 729b739).
- **Ref.** ROADMAP §41.4.
- **Score. 3/5.**

#### MN-34 · Better SAE reconstruction does not go with clearer feature causality
- **Claim.** Across targets, held-out reconstruction fidelity vs the fraction of scorable candidate features with at least one channel clearing the ablation null: V3 (`panel7_v3_dev`, primary null `profile_matched_cov`) ρ = −0.326 (p = 0.022, n = 49); v2 (`panel7_v2_dev`) ρ = −0.219 (p = 0.108, n = 55). Forecast-preservation |ΔMASE| vs the same fraction: V3 ρ = −0.237 (p = 0.102), v2 ρ = −0.091 (p = 0.508). V3 trained on 2.5× the rows of v2 and raised median held-out fidelity from 0.749 to 0.82; SAE admission went from 5/56 to 8/49.
- **Reading.** No evidence that scaling SAE training data would make feature causality clearer; the binding limit is the per-feature battery (8 series, 16 null directions) and candidate selection. Depth and dictionary size are uncontrolled confounds, so the negative sign is not interpreted. Basis for dropping reconstruction fidelity to a loose sanity floor (ROADMAP §42).
- **Admission consequence (2026-10-10, ROADMAP §42.5).** Re-evaluated without retraining under a 0.5 fidelity floor with forecast preservation on 128 held-out series (`admission_sec42`): 35 of 50 V3 targets admitted (legacy 8); flips False→True 28, True→False 1 (Chronos-T5-Base encoder.block.5, ΔMASE 0.3474 at 128 series).
- **Evidence.** Descriptive, exploratory (dev artifacts, no new compute). **Status:** open.
- **Reproduce.** Spearman over `runs/<run>/sae/meta.json` (`reconstruction_fidelity_heldout`, admission checks) and `sae/<model>/<layer>_ablation.json` (`candidates[*].n_channels_clearing`, `scorable`).
- **Ref.** ROADMAP §41.5, §42.
- **Score. 3/5.**

#### MN-35 · Attribution to the horizon-mean point forecast cannot rank shape concepts and fails its pre-fixed gate
- **Claim.** Ranking SAE atoms by |activation × ∂(mean point forecast over the horizon)/∂activation| recovers planted K1 concepts in its top 20 at 0.25, 0.3077, 0.2222, 0.25 on held-out seeds 1–4 (mean 0.2575, bar 0.8): **FAIL**. Level concepts are recovered (3/3, 4/4, 1/1, 3/3); trend, seasonal and dispersion almost never (seed 1: trend 0/6, dispersion 0/2). The gradient is exact (first-order vs real ablation Pearson ≥ 0.99999). On the same seeds, after the fact: per-horizon-step attribution 0.8990, MASE attribution 0.7511, the legacy candidate union 0.4850, random 0.1026.
- **Reading.** A scalar that averages the horizon cancels shape effects; the summary, not the gradient, is the problem. The per-step and MASE variants are re-gated on unseen seeds 5–8 before use (ROADMAP §42.1).
- **Re-gate (unseen seeds 5–8, ROADMAP §42.3).** Per-step attribution (`attribution_horizon_l1`) 0.9, 0.625, 0.8, 1.0, mean 0.83125: **PASS**; MASE attribution mean 0.7375: FAIL; legacy union 0.625; random 0.15. Its dispersion recovery relies on the mock's point-visible dispersion shape and says nothing about quantile-only dispersion in real models.
- **Evidence.** Known-answer validation of a method. **Status:** negative (mean-summary rule); per-step variant adopted after passing on unseen seeds.
- **Reproduce.** `run_attribution_known_answer.py`, `configs/attribution_known_answer.yaml`, `sae/attribution.py` (merge 26588d5).
- **Ref.** ROADMAP §42.1.
- **Score. 3/5.**

#### MN-36 · A powered per-feature test (more series, more null draws, multiplicity control) fails its gate; the simple p95 rule at 8 series is the better-calibrated design
- **Claim.** On K1 (`mock_planted`, cov null, held-out seeds 1–4) a per-feature test with 32 series, 64 null draws, a predictive-t p per channel, Holm within feature and BH within target reaches sensitivity 27/45 = 0.6 (FPR 7/264): **FAIL** against 0.8. The legacy rule (≥ 1 channel above the null's pooled p95) scores 41/45 = 0.9111 at 8 series × 16 draws (FPR 5/2299 cells) and 34/45 = 0.7556 at 32 × 64. The literal rank-p version cannot reject at all (Holm floor 9/65 = 0.1385).
- **Reading.** Planted effects sit at ~2–6 null sd and the null is common-mode across series, so more rows barely tighten it while channel and feature multiplicity raise the bar. V3's scarcity of causal features is not a per-feature sample-size problem; candidate choice and on-manifold effect size are the levers. Corrects the premise of ROADMAP §42.
- **Correction (2026-10-10, MN-37).** Candidate choice turned out not to be a lever under the on-manifold null either; on-manifold effect size is.
- **Evidence.** Known-answer validation of a method. **Status:** negative (closed).
- **Reproduce.** `run_powered_known_answer.py`, `configs/known_answer_powered.yaml`, `sae/powered.py` (merge d85f590); JSONs in `tsfm_lens/runs/sec42_results/`.
- **Ref.** ROADMAP §42.4.
- **Score. 4/5.**

#### MN-37 · Gradient-attribution candidates do not raise the per-feature causal rate under the on-manifold null
- **Claim.** `panel7_v3_sec42` (V3 SAEs, 38 targets, legacy p95 rule at 8 series × 16 draws): adding the top-20 `attribution_horizon_l1` atoms per target raises cov-causal features 78 → 112 by testing 620 more candidates. Per scorable candidate on the 31 differentiable targets: cov null attribution-nominated 48/562 = 0.0854 vs random-rule 24/219 = 0.1096 (Fisher OR 0.759, p 0.335); isotropic null 168/562 = 0.2989 vs 42/219 = 0.1918 (OR 1.797, p 0.0022). Feature-level cov count stays below independent chance (0.898; dev 0.818).
- **Reading.** Atoms with large forecast gradients move the forecast more than a random isotropic direction but not more than a covariance-matched one. Neither per-feature sample size (MN-36) nor candidate choice is the lever; on-manifold effect size is. The rule passed its K1 gate (MN-35), so the gate did not predict real-model behavior; why is not measured.
- **Evidence.** Causal within-model (battery), exploratory (dev). **Status:** negative (closed).
- **Reproduce.** `configs/panel7_v3_sec42.yaml`, `build_sec42_run_dir.py`, `run_admission_reeval.py` (merge of 4c50518); `runs/panel7_v3_sec42/sae/*/*_ablation.json` (`candidates[*].rules`, `n_channels_clearing`, `by_null.profile_matched`).
- **Ref.** ROADMAP §42.5.
- **Score. 4/5.** Closes the "test better features" route with a measurement.

#### MN-38 · Window-mean pooling hides within-window order and understates cross-model similarity by up to 0.33 CKA on dev (0.30–0.43 held-out); concatenating a window's tokens fixes it
- **Claim.** On v3 dev series (four disjoint sets: PILOT 128, MAIN 512, CONFIRM 512, VALIDATE 512):
  - Averaging a window's tokens discards a large share of token variation (within-window share 0.40-0.76 for the
    16-step and 1-step models; Chronos-T5-Base 0.70-0.76 at every depth) and, with it, within-window ORDER. On
    CONFIRM, pooled/fine decodability of window summaries falls to 0.699 (Time-MoE layer 0) and 0.773 (Chronos-T5-Base
    block 0); slope decodes at 0.13 pooled vs 0.96 fine (Chronos-T5-Base block 0). Level and spread are kept.
  - On identical rows, mean pooling is fair (|Δ peak CKA| < 0.05, rho ≥ 0.9) for only 10 of 21 pairs (CONFIRM).
    Input control: CKA of the input with TimesFM layer 0 is 0.1497 (window mean), 0.4322 (4 sub-means), 0.5372
    (32 values).
  - Lossless CONCAT (tokens concatenated in time order; Gram sums) on VALIDATE, all layers: TimesFM vs Chronos-2
    0.4522 → 0.7348, vs Chronos-Bolt 0.3877 → 0.6543, vs Chronos-T5-Base 0.3137 → 0.6428, vs Sundial 0.4967 → 0.6363;
    pairs with Timer fall (Timer vs Chronos-T5-Base 0.2886 → 0.1851); rank rho over 21 pairs 0.8026. Sub-window
    means at 8 steps do not converge for Chronos-T5-Base (0.5321 vs CONCAT 0.6216), so 1-step models need full CONCAT.
  - A mean-plus-trend rule repairs decodability (15/15 ratios ≥ 0.993) but its CKA depends on the block weight
    (spread up to 0.2630), so it is not a similarity representation.
- **Reading.** Window-level CKA under mean pooling is biased toward dissimilarity between coarse-token (TimesFM) and
  fine-token models. Shared geometry is stronger than reported, and mostly at early, input-encoding layers. Pair
  rankings by CKA partly reflect tokenization.
- **Scope correction (2026-11-11; MP-12).** The recorded loss of within-window
  order and the held-out raw CKA contrasts stand. "Understates" and "fixes"
  above refer to this particular order-preserving representation and study,
  not an intrinsic ground-truth similarity or a universally preferable metric.
  Matching PCA rank can reverse a raw mean-versus-concat ordering, and changing
  window length changes the row-level estimand even with identical series and
  crop. Higher CKA alone does not establish more shared task information.
  Use the window robustness diagnostic and held-out concept probes before
  extending this conclusion to another window, layer, task or feature budget.
- **Evidence.** Descriptive / geometric; pre-registered criteria (ROADMAP §44–44.4). **Status:** held-out confirmed
  (ROADMAP §47.3, fresh sealed epoch; before that confirmed on a fresh dev set); the fix is in the library as `alignment.window_rep: concat`, the default
  since ROADMAP §44.6 (recorded runs pin `mean`).
- **Addendum (ROADMAP §45.1, 2026-10-16): the pipeline on the full v3 dev run.** Re-extraction reproduces all 21
  recorded MEAN peaks exactly (max |Δ| 0.0). On the same 512 series, CONCAT vs MEAN peak CKA: TimesFM vs Chronos-2
  0.7119318842887878 vs 0.4411318302154541, vs Chronos-T5-Base 0.6248822212219238 vs 0.31856971979141235, vs
  Chronos-Bolt 0.6245514154434204 vs 0.3722766041755676, vs Sundial 0.6043637990951538 vs 0.470203697681427, vs Timer
  0.37477120757102966 vs 0.37477126717567444. Spearman of 21 peaks 0.764935064935065; TimesFM/Chronos-2 7th vs 13th.
  The L1 CONCAT null is 5 permutations (mean/min/max), weaker than §44.5's 40. Reproduce:
  `configs/panel7_v3_concat.yaml` (extract,l1), `run_v3_concat_compare.py`.
- **Addendum (ROADMAP §47.3, 2026-10-17): HELD-OUT CONFIRMED.** Registered before data (commit 1c1a7f4) and tested once on
  a fresh sealed v3 private epoch (epoch 2, 512 series, 0 hash overlap with 13 splits). H-C1: all 21 pairs' CONCAT CKA at
  the dev layer pairs exceeds the 1000-permutation shuffled-series null (each p 0.000999000999000999, Holm
  0.02097902097902098); TimesFM/Chronos-2 0.7070513688986635 (dev 0.7119318842887878). H-C2: averaging lowers TimesFM's CKA
  at the same layers and rows by 0.3215409540598726 (Chronos-2), 0.4333057196446368 (Chronos-T5-Base), 0.3005587061032854
  (Chronos-Bolt), 0.3029117490758303 (Sundial), each bootstrap CI above 0.2697, Holm p 0.0007998400319936012. Reproduce:
  `tsfm_lens/run_concat_confirm.py`, `configs/panel7_v3_concat_confirm_e2.yaml`.
- **Addendum (ROADMAP §47.1, 2026-10-17): v2 and a 1000-permutation null.** The CONCAT null now uses 1000
  whole-series permutations with a p95 and the best pair a series-bootstrap CI; on v3 the peaks are unchanged (max |Δ| 0.0)
  and every pair has p = 0.000999000999000999. On 512 v2 dev series, CONCAT vs MEAN on the same rows: TimesFM vs Chronos-2
  0.6353724598884583 vs 0.43068113923072815, vs Chronos-T5-Base 0.507454514503479 vs 0.34261298179626465, vs Chronos-Bolt
  0.5459014177322388 vs 0.3827388882637024, vs Sundial 0.5461937785148621 vs 0.42134612798690796; Timer vs
  Chronos-T5-Base falls 0.38444724678993225 -> 0.250899076461792. Reproduce: `configs/panel7_v2_concat.yaml`,
  `configs/panel7_v2_concat_meancheck.yaml`, `configs/panel7_v3_concat_null.yaml` (commits cb26f52, 5ab8ac4).
- **Addendum (ROADMAP §47.2, 2026-10-17): stitching.** On the same 512 v3 series, with L2's protocol (reproduced to
  2.98e-08), concatenating the SOURCE window raises stitching gain over the input baseline from 0.4980521354254505 to
  0.5536236378315946 on average at the 21 CONCAT peak pairs (both directions), most into TimesFM (Chronos-T5-Base ->
  TimesFM 0.3070 -> 0.7879). Pair ranking is kept (Spearman over 21 pairs 0.9246753246753248; 0.9662337662337662 at the
  recorded L2 pairs). Concatenated TARGETS of 12- and 32-token models (12288/24576 dims) are underdetermined at 5760
  training rows (gain -0.11537519500112832 vs MEAN). Reproduce: `tsfm_lens/run_concat_stitching.py` (commit aca7a1e).
- **Reproduce.** `tsfm_lens/run_pooling_check.py` (pilot/main/confirm JSONs in `runs/pooling_check/`),
  `tsfm_lens/run_concat_check.py` (`runs/concat_check/`), `analysis/window_concat.py`.
- **Ref.** ROADMAP §44.2, §44.3a, §44.5.
- **Score. 5/5.** A large, replicated dependence on temporal representation;
  the dated scope correction separates order preservation from a universal
  preference for one similarity measurement.

---

#### MN-39 · A bounded frozen-decoder hierarchy loses stability and does not improve task organization
- **Claim.** On a fresh controlled seasonality corpus, a frozen-level-one,
  SAE++-inspired decoder hierarchy with four parents fails all three prospective
  promotion criteria. For 256 training-selected atoms from the historical
  Chronos-Bolt dictionary, three hierarchy seeds have pairwise ARI
  **0.09744900235251616 / 0.022057249581602746 / 0.10284922591492561**.
  Held-out coarse balanced accuracy is **0.90625 / 0.8229166666666666 /
  0.9895833333333334**, versus cosine grouping **0.9166666666666666**.
  Fine period accuracy is **0.7916666666666666 / 0.625 / 0.7083333333333334**,
  versus cosine **0.7708333333333334**, size-matched random **0.9375**, and
  uncompressed codes **1.0**. Uncompressed coarse accuracy is also **1.0**.
  The selected atoms cover **0.6938634399308558** of training activation
  frequency. Optimization is finite and decreasing; every atom is assigned.
- **Models / layers.** Chronos-Bolt-small, `encoder.block.3`, historical SAE;
  48 held-out generating seeds, each with paired absent/present seasonality.
- **Evidence.** Descriptive / held-out predictive, exploratory. This tests a
  bounded frozen-dictionary transfer ablation, **not full joint SAE++**. The
  context/horizon changes from historical 480/64 to 128/32; pooling matches
  the historical 32-step window means. No causal or monosemanticity claim.
- **Status.** Negative screen (closed without scaling, 2026-11-11).
- **Reproduce.** `tsfm_lens/experiments/feature_search/hierarchy_poc.py`;
  `tsfm_lens/runs/feature_search_20261111_hierarchy/results.json`;
  checkpoint `runs/panel7_v3_ablcheck/sae/Chronos-Bolt/encoder_block_3.pt`.
- **Ref.** ROADMAP §48.1–48.2.
- **Score. 3/5.** A cheap falsifiable screen prevents expensive hierarchy
  training from being justified by reconstruction alone.

#### MN-40 · A mean-zero null does not justify paired sign flips without symmetry
- **Claim.** An instrument calibration using the actual production statistic
  finds Holm4 family rejection in **8/128** symmetric Gaussian null families
  (**0.0625**, exact95% binomial interval
  [0.027366684886370826, 0.11943678339385401]). In deliberately skewed
  mean-zero families, where the statistic's symmetry assumption is violated,
  rejection is **44/128** (**0.34375**). Each family has64 independent
  seed-level contrasts and999 Monte Carlo sign flips. Mean-zero and
  exchangeable/symmetric differences are distinct assumptions. The symmetric
  sample is too small to certify5% calibration; the stress case demonstrates
  the danger of claiming distribution-free mean inference.
- **Models / layers.** No models; statistical machinery only.
- **Evidence.** Illustrative simulation with known population nulls and a
  deliberately invalid-assumption decoy; no TSFM performance conclusion.
- **Status.** Negative (closed) for the claim that mean zero suffices.
  Historical model contrasts retain their stated conditional assumptions.
- **Different endpoint.** New protocols can instead select exact two-sided
  inference on positive-sign probability, with ties/exclusions and mean effects
  shown separately. In an asymmetric q=.5 simulation with mean−1, the observed
  corrected count is0/128, but the analytical family error is
  **0.03219153531930352**. Zero is a low-tail draw, not proof of zero error;
  the separate post-hoc audit records its probability
  **0.015172085364807403**. This endpoint does not test the mean.
- **Reproduce.** `runs/feature_search_20261111_statistics_calibration1992000/`,
  `runs/feature_search_20261111_sign_null_calibration2114000/`,
  saved function snapshots and per-replicate results.
- **Ref.** ROADMAP §48.18.
- **Score. 3/5.** Motivates an explicitly different exact sign-probability
  endpoint and prevents mistaking conditional mean p-values for universal guarantees.

#### MN-41 · Derivative-based failure prediction does not survive a stronger input control
- **Claim.** Off-frequency energy alone clears no frozen promotion gate on
  the128-seed evaluation cohort. Bolt's joint off/on-target derivative ridge
  augmentation initially gives **0.3068546604934841** relative prediction-loss
  improvement, with exploratory Holm6 sign p **0.004979984371356317**.
  A predeclared stronger input RBF challenge reverses this: input-only prediction
  MSE **0.00793532001987993**, joint **0.00922707377887333**, on-only
  **0.010098940806783127**. Joint gain is **-0.001291753758993399**,
  pointwise95% interval [-0.003004891461113323,0.00026984535158859103].
- **Models / layers.** All three native models in the first analysis; only
  positive Bolt candidate in the RBF challenge. Input descriptors and fixed
  native derivatives; no new GPU inference.
- **Evidence.** Predictive exploratory analysis: discovery1659000 fits/CV,
  evaluation2225000 fixed. Both cohorts were previously inspected.
- **Status.** Negative (closed) for this incremental-prediction design; no
  new native replication. Does not establish that interference is causally irrelevant.
- **Reproduce.** `experiments/feature_search/stage49_failure_prediction.py`;
  `runs/feature_search_20261111_stage49_failure_prediction_v3/` and
  `runs/feature_search_20261111_stage49_rbf_challenge/`.
- **Ref.** ROADMAP §49.2.
- **Score. 3/5.** Prevents a simple input-baseline underfit from becoming a
  claimed mechanistic explanation of individual forecasting failures.

#### MN-42 · A residual-update clamp pilot fails its complete localization gate
- **Claim.** No tested TimesFM site advances. At block19, off-energy falls
  from **0.1303457303564841** to **0.08028764842129353** and intended relative
  RMS error improves by **0.017882898460587382**. But13/16 favorable signs
  versus native and random give Holm8 p **0.170166015625**. The practical
  effect alone does not satisfy the registered gate. Baseline forecasts are
  unchanged, and all-condition MSE is **0.30075443735853646** native versus
  **0.3007761068835905** clamped, so this is not forecasting repair.
- **Models / layers.** TimesFM2.5-200M blocks[0,9,15,19], all native positions,
  16 fresh bundles3100000, periods32/16, context128/horizon32.
- **Evidence.** Exploratory within-model response intervention, with exact
  identity guards, output-clamp ceilings and matched-random edit energy.
- **Status.** Negative (closed) under the promotion rule; not evidence of zero
  layer effect. No larger sample or threshold change follows this pilot.
- **Reproduce.** `experiments/feature_search/stage49_residual_update.py`;
  `runs/stage49_timesfm_residual_update3100000_retry1/`.
- **Ref.** ROADMAP §49.3.
- **Score. 3/5.** Separates an interesting local effect from a defensible
  localized-feature claim and from actual forecasting improvement.

#### MN-43 · Low-order phase-response features do not fully explain the difficult cases
- **Claim.** In a fixed8×8 to16×16 phase-grid check, TimesFM32/16 has0/4
  stable low-order bundles and0/8 off-grid passes; Bolt32/16 has2/4 and0/8.
  Their mode identities remain unresolved at this budget. Bolt40/20 has4/4
  stability and high-order-energy passes, but0/8 off-grid passes: waveform
  RMS/shared-scale ranges **0.06061027936225086–0.14363814021190519**, above
  the frozen0.05 limit. Its stable low-order pattern is only partial.
  TimesFM40/20 and Chronos-2 at both pairs pass all three checks, but no
  originally eligible cross-model comparison achieves complete coverage.
- **Models / layers.** Native TimesFM2.5, Bolt-small and Chronos-2 forecasts,
  context128/horizon64, eight fixed amplitude/scale bundles; no hidden states.
- **Evidence.** Behavioral/descriptive mathematical-resolution validation.
  Full-waveform phase DFT avoids temporal-basis leakage; separate nested and
  off-grid checks test phase aliasing and predictive coverage. No population
  p-value belongs to this deterministic grid.
- **Status.** Negative (closed) for complete low-order explanations at this
  budget. Stable40/20 patterns may be described with their coverage limit;
  no hidden-feature expansion, higher order or finer grid is promoted.
  Original failed pilot controls remain recorded; the amended follow-up's
  fixed-batch native controls pass exactly.
- **Reproduce.** `experiments/feature_search/phase_orbit_alias.py`;
  `runs/feature_search_stage49_phase_alias3500000/`, with linked coarse pilot,
  separate FP64 replay and fixed-batch control artifacts.
- **Ref.** ROADMAP §49.8–49.10, §49.12.
- **Score. 3/5.** Prevents a visually simple low-order feature diagram from
  becoming an unsupported explanation; separates stable partial structure
  from adequate task coverage.

#### MN-44 · Native forecasts do not preserve near-equivalent linear task recovery
- **Claim.** Under SH-25's same fitting specification, output-decoder scaled
  future MSE is **0.022370863219791523** for Bolt and **0.07411271997270756**
  for TimesFM, versus frozen head-input decoder **0.0005480985483314407**
  and **0.0006623374741218504**. Paired excesses are **0.02182276467146008**
  [0.01975334769485844,0.023993157166713287] and **0.07345038249858571**
  [0.06463308883673971,0.08314720863694158]. Both models fail the0.01
  absolute/excess practical gates; no nonlinear fallback follows.
- **Models / layers.** Bolt-small and TimesFM2.5, native32-step output versus
  actual head input; same96/48 old split and128 inspected evaluation bundles.
- **Evidence.** Correlational (exploratory linear recoverability). Intervals are pointwise
  seed bootstraps, not new confirmation. Different feature dimensions and
  finite-sample fitting remain possible contributors to the gap.
- **Status.** Negative (closed) for near-equivalent output recovery under this
  specification. It does not demonstrate information destruction or absence.
  The decoder uses supervised labels and known frequencies, not a fair
  zero-shot forecasting correction. Original caches lack embedded row IDs;
  physical row/split/prediction checks do not erase that provenance limitation.
- **Reproduce.** `experiments/feature_search/forecast_recoverability.py`;
  `runs/feature_search_stage49_forecast_recoverability/`.
- **Ref.** ROADMAP §49.11, §49.13; SH-25.
- **Score. 3/5.** Distinguishes retained head-input task information from what
  a fixed linear diagnostic can recover after the native forecast mapping.

#### MN-45 · A tiny cross-fitted linear subspace does not preserve the composition repair
- **Claim.** On the16 reused discovery seeds, uncentered channel-SVD ranks1/2/4
  yield normalized losses **0.29508030518868866**, **0.1952377910849517**
  and **0.1385365731017681**, versus full-edit **0.036757892047086**.
  They recover **0.39338273758107306**, **0.6278424263167613** and
  **0.7609936198600142** of the full edit's native-loss improvement.
  Every rank fails the frozen loss≤0.06 and recovery≥0.8 gates. Rank4
  beats its matched-random mean by **0.23509182549852548** and the
  mean-only edit by **0.14481047365507416**, which cannot rescue those failures.
- **Models / layers.** TimesFM2.5, after15 of20 blocks (index14), native
  four-token states, periods32/16, context128/horizon64. Opposite seed-parity
  folds fit8 seeds and evaluate8; tokens remain dependent within each seed.
- **Evidence.** Causal within-model (L3), exploratory compression of the
  oracle-component state intervention. No forecast targets enter fitting;
  held-out corrections still require separated component inputs.
- **Status.** Negative (closed) for the fixed ranks1/2/4 recipe. All16 seeds
  and all three ranks pass geometry/execution checks; none is dropped. No
  p-values on this inspected cohort, no rank escalation or fresh follow-up.
  This does not rule out all low-dimensional/nonlinear mechanisms and does
  not weaken CA-14's independent full-state restoration replication.
- **Reproduce.** `experiments/feature_search/composition_compactness.py`;
  `runs/feature_search_stage49_compactness3600000/`, with sibling
  `feature_search_stage49_compactness3600000_independent_audit/` replaying
  both fitted projectors, all96 random edits, actual state geometry and losses.
- **Ref.** ROADMAP §49.18–49.19; CA-14.
- **Score. 3/5.** Prevents a strong whole-state intervention from being
  prematurely attributed to a few clean linear features.

## H. Benchmark trust

#### BM-01 · Near-duplicate matchers disagree ~70× on the same corpus
- **Claim.** On the demo corpus (5 planted duplicates), redundancy is **0.072%** with `dtw` (the actual CLI default) and **4.835%** with `xcorr` (the default according to the docstring).
- **Status:** open. **Reproduce.** `run_validation.py --method dtw|xcorr`. **Ref.** ROADMAP §34 B4 (~31890–31904).
- **Score. 3/5.** A redundancy number means nothing without its matcher.

#### BM-02 · A fixed per-group diversity threshold fails small groups on pure noise
- **Claim.** The null near-collision fraction is 0.3933 at n=5, 0.25 at n=8 and 0.20 at n=10, against a 0.20 bar. The gate is now skipped below n=20, with a third `None` state.
- **Status:** fixed. **Reproduce.** `benchmark_validation/gates.py`; `tests/test_gates_calibration.py`. **Ref.** ROADMAP §34 B4 (~31922–31938).
- **Score. 2/5.**

#### BM-03 · Length-inheriting generators can silently empty a family
- **Claim.**
  - `block_bootstrap` drew series of 27–152 points, against 576 needed.
  - Every audit passed, and the downstream filter dropped them all.
  - With `min_length` on, only `weather` (~600 series) is usable, because
    long hourly Monash domains fail to load.
- **Status:** fixed (opt-in `min_length`). **Reproduce.** `sources.py::_filter_min_length`; `tests/test_source_min_length.py`. **Ref.** ROADMAP §24.7(17–18) (~21281–21339); CLAUDE_FULL §11.46.
- **Score. 3/5.** Check the per-family length range, not only the count.

#### BM-04 · Golden-hash regeneration may be numpy-version-fragile
- **Claim.** The golden-hash test fails deterministically in a fresh `numpy==2.1.0` environment with no code change. The suspect is `rng.choice(replace=False)`.
- **Status:** open. **Ref.** CLAUDE_FULL §11.13.
- **Score. 3/5.** "Bit-exact" needs a pinned numpy.

#### BM-05 · The counterfactual identity check caught a corruption-replay bug
- **Claim.** 244/298 dose-1.0 regenerations failed to reproduce the original, because the post-generation corruption chain was not replayed. After the fix, 0/1113 fail.
- **Status:** fixed. **Reproduce.** `build_pipeline/counterfactual.py::_replay_transforms`. **Ref.** ROADMAP §37.9 (~36611–36619).
- **Score. 2/5.** An example of why "dose 1.0 must equal the original" is a mandatory free test.

#### BM-06 · ~42% of the dev corpus is weakly predictable, intermittent real-derived data, and it drives the ~50% flat-forecast share
- **Claim.**
  - The real-derived tier is 410/965 series, almost all Monash `weather` plus ETT. Its structure:
    `block_bootstrap` 61% zeros (lag1 0.212); `sequential_par` 36% zeros (lag1 0.115); `mixture`
    lag1 0.241.
  - Flat forecasts there are 0.67–1.0 of series, vs 0 on structured synthetic archetypes.
  - Preprocessing is not responsible. Native APIs match the adapters to 0.002–0.028 context sd.
  - Point forecasts are central estimates, so they are flat by design on such data. Sundial single
    sample sd ratio 0.42 vs median 0.16; the single sample is worse (MASE 1.472).
- **Evidence.** Behavioral / descriptive. **Status:** measured on dev, orchestrator scratch scripts; not in the pipeline.
- **Reproduce.** Stored predictions in `runs/full_report_run_4model` by generator; corpus task list in `configs/large_run.yaml`.
- **Ref.** ROADMAP §32.7 addendum (2026-09-27); PM-07; MN-14.
- **Remedy (D1, opt-in).** `configs/large_run_v2.yaml` adds `autogluon/chronos_datasets` electricity_hourly and traffic twins for each real-derived generator. The naive-flat proxy (lag1 < 0.2 and not periodic) halves corpus-wide, 0.2093 → 0.1067. Actual model flat share on dev falls correspondingly, v1 → v2: TimesFM 0.4943 → 0.3254, Chronos-2 0.4912 → 0.3202, Sundial 0.3917 → 0.2715, Chronos-Bolt 0.4881 → 0.3171. Sundial's sequential_par flat share rises 0.20 → 0.72; diagnosed as an adapter defect (PM-15). No full pipeline rerun on v2 yet. Ref. ROADMAP §32.7 D1.
- **Score. 4/5.** The ~50% flat share and small ablation effects are partly a property of the corpus, not of the models. Rebalancing the real-derived source would change what the causal battery can see.

#### PM-24 · Real-background units changes preserve these two models' responses
- **Claim.** On12 heldout hourly sources (4 each electricity/M4/pedestrians),
  the worst fixed scale/offset forecast-or-response RMS, averaged by dataset,
  is Bolt **5.488865236960517e-7**, **3.1393979675288033e-7**,
  **3.227940600501552e-7**, and TimesFM **2.280941964267713e-6**,
  **1.4082511017083496e-6**, **7.435093066571062e-7**, in source past-SD
  units. Scales0.1/10 and offsets-5/+5 expose no practical defect against the
  prespecified0.1 gate. This does not establish exact global equivariance.
- **Models / layers.** Chronos-Bolt small and TimesFM2.5-200M; native forecasts,
  context128/horizon32, relative seasonal amplitude0.5, period24 hours.
- **Evidence.** Behavioral, descriptive.
- **Status.** Exploratory (dev); scaling-defect screen closed without expansion.
- **Reproduce.** `experiments/feature_search/real_background_poc.py`;
  `runs/real_background_poc492306/{bolt,timesfm}.json`, experiment4;
  `examples/feature_search_stage49/real_background/index.html`.
- **Ref.** ROADMAP §49.24–49.25.
- **Score. 3/5.** Rules out practical units changes as the explanation for
  response variation on these particular real-background pilot inputs.

#### MN-46 · Good tiny-SAE state reconstruction misses controlled activation responses
- **Claim.** Three matched64-atom TopK8 refits (150 steps,12 discovery sources)
  achieve heldout state NMSE Bolt **0.05188297480344772**,
  **0.052068207412958145**, **0.05264982953667641**, but response NMSE
  **0.6410754323005676**, **0.6525039076805115**, **0.6279963850975037**.
  TimesFM state NMSE is **0.1314772069454193**, **0.13562437891960144**,
  **0.1317705512046814**, with response NMSE **0.5753476023674011**,
  **0.5692651867866516**, **0.5553540587425232**. Every refit fails the frozen
  response<=0.25 gate despite passing state reconstruction. Controlled
  forecast-effect NMSE also exceeds0.1. Feature permutation and duplication
  conserve decoded states within **7.62939453125e-6**.
- **Models / layers.** Chronos-Bolt small final encoder block and TimesFM2.5
  final transformer block; all native tokens including Bolt REG. Three cached
  hourly datasets, source-disjoint12 evaluation sources. Positive seasonal
  addition at0.5 past-SD, period24; no SAE seed/width/budget selection.
- **Evidence.** Descriptive geometric reconstruction and within-model patches;
  oracle native-normalizer closure also fails, limiting functional attribution.
- **Status.** Negative (closed) for this tiny recipe; exploratory evidence.
  This is not evidence that SAEs generally fail or that model semantics are
  unstable. Response-sensitive fitting or greater capacity was not tested.
- **Reproduce.** `experiments/feature_search/real_background_poc.py::sae_audit`;
  `runs/real_background_poc492306/{bolt,timesfm}.json`, experiment6;
  `examples/feature_search_stage49/real_background/index.html`.
- **Ref.** ROADMAP §49.24–49.25.
- **Score. 4/5.** A concrete counterexample to treating global reconstruction
  quality as sufficient validation for controlled feature-response comparisons.

#### PM-25 · A separated weak component exposes a model-dependent distraction boundary
- **Claim.** With target amplitude0.5/bin5 fixed and far distractor bin17 at
  amplitudes0.5/2/8, TimesFM check response NMSE is **0.12283654886009954**,
  **0.5504058311228635**, **1.8385632140476604**. A past-only K4 Fourier
  partition yields **0.004979990196191095** at all three amplitudes. Bolt
  instead has native errors **1.1023462011450356**, **2.030500181193551**,
  **1.279860610934934**, with separated error **0.7478962472416841**.
  For a nearby bin6 distractor at amplitude8, TimesFM changes
  **1.4235090872171847**→**1.4456124856595713**: the fixed split keeps that
  pair in one group. The far adapter's amplitude invariance is structural
  because target-group inputs are identical; its accuracy is the measurement.
- **Models / layers.** Chronos-Bolt small and TimesFM2.5-200M native point
  forecasts;128 history/32 horizon, fixed noise0.05, no internal intervention.
 16 fresh generating seeds split8 discovery/8 check; signs paired within seed.
- **Evidence.** Behavioral, descriptive; no isolated internal mechanism claim.
- **Status.** Exploratory (dev), candidate for fresh-source replication.
  TimesFM's far-amplitude sensitivity has the same positive direction in both
  halves; no multiplicity-corrected significance or broad generalization claim.
- **Reproduce.** `experiments/feature_search/fourier_poc.py`;
  `runs/fourier_poc4929000/`; portable `fourier_poc/{bolt,timesfm}_summary.json`
  under `examples/feature_search_stage49/`, responses and distractor contrasts.
- **Ref.** ROADMAP §49.29–49.30.
- **Score. 4/5.** Separates a model that forecasts the isolated contribution
  accurately from one that remains inaccurate after the same decomposition.

#### PM-26 · Controlled-response fidelity can improve while real forecast accuracy deteriorates
- **Claim.** TimesFM's native→K4 known-addition response NMSE on four check
  electricity sources is **0.7354096312950661**→**0.23021885330862987**;
  on four pedestrian sources **0.6644351168534955**→**0.23514514961125638**.
  All four sources in each check group and all four discovery sources in each
  group improve response. However, unmodified total-future normalized loss
  increases **0.6073912219985821**→**1.425829712079506** and
  **0.12325481761696377**→**1.6635653671014523**, respectively.
  Bolt check response instead worsens **0.7356323719635418**→
  **0.9769994295733234** and **0.6810495039385699**→**2.108136046437345**.
- **Models / layers.** TimesFM2.5-200M and Chronos-Bolt small point forecasts;
  real hourly histories with paired +/-0.5 past-SD sinusoidal additions,
  period128/5. Exact reconstruction-preserving Fourier split, no true latent
  decomposition, training, SAE or internal patch.
- **Evidence.** Behavioral, descriptive. Source-disjoint discovery/check,
  conditional on these small datasets; pretraining overlap is unknown.
- **Status.** Exploratory (dev). M4 has a similar TimesFM check pattern but
  fails discovery/check response-direction consistency and is not included in
  the two-domain consistency claim. No general superiority or causality claim.
- **Reproduce.** `experiments/feature_search/fourier_report.py`;
  `runs/fourier_poc4929000/`; `examples/feature_search_stage49/fourier_poc/`.
- **Ref.** ROADMAP §49.29–49.30.
- **Score. 4/5.** Demonstrates why component-response consistency should be
  measured alongside forecast quality rather than used as its substitute.

#### PM-27 · Matched training coverage can dominate extra optimization on unseen combinations
- **Claim.** Across six fresh training corpora with paired initializations,
  the small transformer's unseen-pair NMSE is **0.6374173760414124** at400
  steps and **0.6398251255353292** at800 when the pair combinations are
  withheld. Adding those combinations, with the same example count and exactly
  matched per-component exposures, yields **0.015170028277983269** and
  **0.006212268335123857**. The MLP repeats a smaller gap at800 steps:
  **0.010820749215781689**→**0.0018902896748234828**. All six paired corpora
  improve in both architectures at both checkpoints. Ridge already solves the
  withheld task with **0.001306674792431295** error; a neural-normalization-
  matched ridge control also remains accurate.
- **Models / layers.** Small trained MLP and two-layer patch transformer;
  four sinusoidal frequencies,128 history/32 horizon,1,024 training examples
  and256 latent test blocks per corpus. Equal neural optimizer/data budgets;
  no pretrained TSFM modification or internal activation intervention.
- **Evidence.** Controlled training intervention: combination coverage within
  fixed architecture. Six independent corpus+initialization pairs; test rows
  are not independent training replications. Ridge is a solvability reference,
  not an optimizer-matched architecture experiment.
- **Status.** Exploratory replication on fresh corpora. Supports a persistent
  learned composition failure within this budget; does not identify pretrained
  TSFM training exposure or rule out improvement with much longer optimization.
- **Reproduce.** `experiments/feature_search/coverage_replication.py` and
  `coverage_normalization_control.py`; `runs/coverage_replication493900/`,
  `runs/coverage_normalization493900/`; portable coherence follow-up report.
- **Ref.** ROADMAP §49.31–49.33.
- **Score. 4/5.** Establishes a concrete mechanism that can generate strong
  component familiarity alongside poor combination generalization.

#### PM-28 · Fourier-response rescue has a sharp grouping and off-grid boundary
- **Claim.** On64 fresh seeds, TimesFM's strong far-distractor response NMSE
  changes **3.079471319358797**→**0.004152199204811288** with four-band Fourier
  splitting. For nearby on-grid components, merely moving one band boundary
  changes **1.7095258110167177**→**0.0036124372813698823**, still using four
  parts. But moving frequencies off-grid reverses the benefit: near native
  **1.975004373423026**→FFT **29.435810463708094**, far
  **4.165856284653323**→**9.740471878910146**. Bolt also worsens off-grid.
  A history-only fitted two-tone reference achieves **0.008172170252036145**
  near and **0.0031745014192145794** far, without supplied frequencies.
- **Models / layers.** Chronos-Bolt small and TimesFM2.5-200M native forecasts;
  target amplitude0.5, distractor8,128 history/32 horizon. Frequencies5/6/17
  on-grid and5.35/6.35/17.35 off-grid. No true decomposition supplied to the
  Fourier adapter; the classical reference assumes two tones.
- **Evidence.** Behavioral and controlled input transformation. The on-grid
  far improvement passes the fixed12-test Holm family; grouping/off-grid
  mechanisms are exploratory on64 fresh seeds. Normalizer self-identity and
  forecast repeat gates pass exactly. Independent arithmetic audit passes.
- **Status.** Grounded dev replication extends PM-25 and limits its scope.
  Decomposition count alone is insufficient; paired normalization drift is
  insufficient to explain the native failure. More history helps TimesFM in
  the tested16-seed subset but worsens Bolt's far case in all16. No universal
  adapter benefit, hidden mechanism, or architecture-only attribution.
- **Reproduce.** `experiments/feature_search/coherence_study.py`,
  `coherence_analysis.py`, `coherence_classical.py`;
  `runs/coherence_replication4932000/`; coherence follow-up report.
- **Ref.** ROADMAP §49.31–49.34.
- **Score. 4/5.** A strong negative control shows that an apparent component
  interpretation can depend on the analysis representation itself.

#### PM-29 · The real response/forecast tradeoff replicates on pedestrians, with narrower scope
- **Claim.** On16 fresh pedestrian TEST sources, TimesFM native→FFT response
  NMSE improves **0.6834002141596086**→**0.16499201896090102**, while actual
  future loss worsens **0.16289278445396405**→**1.6260539730301482**. All16
  sources show both directions; Holm p **0.0001373291015625** for each.
  Electricity response improvement does not pass (Holm p
  **0.1152191162109375**, mean interval **[-0.12700144198376273,
  0.6248026350943895]**), although its forecast harm does. Bolt pedestrian
  response instead worsens **0.7210832771225903**→**2.4515692026859535**.
- **Models / layers.** Same two native forecasters and past-only four-band
  decomposition; known +/-0.5 past-SD additions, no oracle real components.
- **Evidence.** Fixed primary family and fresh source-level evaluation;
  bootstrap intervals are conditional on datasets. Source temporal dependence
  and unknown pretraining overlap limit population generalization.
- **Status.** Replicates PM-26's pedestrian tradeoff; narrows its earlier
  two-domain pilot scope. Suppressing high-band forecasts recovers only
  **0.006212028283677725** pedestrian TimesFM loss, so that particular noise
  explanation cannot account for the **1.4631611885761842** total penalty.
- **Reproduce.** `runs/coherence_replication4932000/analysis.json` primary
  family and controls; `examples/feature_search_stage49/coherence_followup/`.
- **Ref.** ROADMAP §49.34.
- **Score. 4/5.** Better component-response fidelity is demonstrably compatible
  with worse real forecasting, even after expanding beyond the tiny pilot.

#### PM-30 · The fixed perturbation selector screen does not establish incremental practical value
- **Claim.** Added response diagnostics worsen ridge TEST loss versus the
  matched input-only selector on electricity **0.5710651275237811**→
  **0.65187409462335**, M4 **0.11545850429947696**→**0.12296266560858424**,
  and pedestrians **0.15995633314817082**→**0.1651241191327528**; traffic ties.
  RBF diagnostics worsen electricity/M4, tie pedestrians, and improve traffic
  by **0.01719967661651505**, driven by only2/16 sources (sign p **0.25**).
- **Models / layers.** Select Bolt or TimesFM using fixed ridge/RBF predictors;
  input descriptors versus input+past-only perturbation responses.64 FIT and64
  untouched TEST sources,16/domain; global/domain constants and past32-step
  hindcast baselines included. No test-selected model or hyperparameter.
- **Evidence.** Exploratory source-disjoint prediction; TEST choice invariance
  under arbitrary TEST-future label changes is tested. Full paired intervals
  and losses retained. Traffic mean is heavy-tailed, so domains are separate.
- **Status.** Negative/unsupported for this fixed recipe. This is not proof
  that all diagnostics fail or that small observed differences are significant.
- **Reproduce.** `coherence_analysis.py::selectors`; replication
  `selector_rows.json`, `analysis.json` selector contrasts, portable report.
- **Ref.** ROADMAP §49.34.
- **Score. 3/5.** Prevents promoting an interpretable response metric into a
  model-selection claim without incremental predictive evidence.

#### PM-31 · Pure hidden composition gives a modest Bolt gain, not a general TSFM solution
- **Claim.** FIT-selected pure Bolt first-block/depth2 fusion improves held-out
  synthetic normalized loss by **0.04554777148291069**, interval
  **[0.03945377595869312, 0.05159511970435278]**, and beats matched random edits
  by **0.014407703650315172**. TimesFM's selected half-native-state blend gains
  **0.0027575090146978944**, interval **[-0.009879762380357215,
  0.013874104216795776]**. Geometry-selected pure TimesFM fusion instead harms
  forecasting: gain **-0.007938986098528698**, interval
  **[-0.014037585629773879, -0.0017337445038410355]**.
- **Models / layers.** Bolt-small blocks0/1/3, TimesFM2.5 blocks0/5/10; adaptive
  continuous-frequency tones plus intact residual; affine zero-state correction.
  Pilot-FIT geometry weights frozen before replication. Full-context native
  normalization metadata remain shared even in pure fusion.
- **Evidence.** L3 within-model intervention and source-held-out behavior;
  16 TEST seed groups, five dependent conditions each. Exact identity gates pass.
  Conditional bootstrap intervals are exploratory, not corrected confirmation.
- **Status.** Modest conditional Bolt positive; TimesFM selected blend is
  inconclusive. Better hidden reconstruction alone is not a forecasting criterion.
- **Reproduce.** `selective_analysis.py`; `selective_replication4942000/analysis.json`;
  `examples/feature_search_stage49/selective_composition/`.
- **Ref.** ROADMAP §49.35–49.37.
- **Score. 4/5.** Separates causal usefulness from geometric reconstructibility.

#### PM-32 · A dominant component can conceal failure to reconstruct the remainder
- **Claim.** First-block/depth1 equal composition on off-grid dominant mixtures
  has mean FP64 whole-state error **0.0022665692188878588** (Bolt) and
  **0.00793706386934049** (TimesFM), yet error relative to full-minus-dominant
  state is **2.2292954023482277** and **3.685396283813229**. Balanced mixtures give
  remaining-relative errors **2.4911155737257302** and **0.26719512599310846**.
  Despite closer TimesFM reconstruction, its balanced future loss worsens
  **0.327477538734416**→**0.33736003512872703**; Bolt improves
  **0.7459936128206015**→**0.6915583213036312**.
- **Models / layers.** Same frozen pair, first block, pure equal-weight fusion,
  residual retained; full-context normalizer. All native token positions.
- **Evidence.** L1 geometry plus L3 suffix intervention, descriptive paired means
  over16 held-out seeds; dominant-only and balanced controls. The measured
  remainder is not assumed to be an oracle representation of the weak component.
- **Status.** Replicates the pilot denominator warning; model differences concern
  normalized within-model errors, not coordinate correspondence across models.
- **Reproduce.** `selective_diagnostics.py`; replication saved states,
  `marginal_geometry.json`, `analysis.json`; selective composition report.
- **Ref.** ROADMAP §49.37.
- **Score. 4/5.** A good whole-state score can hide precisely the contribution an
  interpretability experiment is meant to recover.

#### PM-33 · Local smoothness and transferable behavior do not guarantee correct component semantics
- **Claim.** At synthetic perturbation radii **0.001/0.01/0.1**, Bolt first-block
  sensitivity is **0.5246727590065028/0.5243779804855171/0.5201747462844821**;
  TimesFM is **0.7031406556959651/0.7030663185533745/0.6959231112928421**.
  Curvature shrinks with radius. On16 real TEST receivers, Bolt near-daily
  first-block/radius0.1 effect transport NMSE is **0.024698876363800082** near
  versus **0.1678951031062752** far. All **192/192** oracle and semantic gates
  pass across repeated cuts/radii/donors. At19.2 hours, **96/96** oracle gates
  pass but **0/96** native-signal semantic gates pass.
- **Models / layers.** Local tests on both models, native and locked normalization;
  transfer within each model from FIT donor to TEST receiver, three cuts.
- **Evidence.** L1 local finite differences and L3 donor-effect intervention.
  Transfer oracle is the receiver's actual counterfactual state with matching
  normalization; semantic gate separately requires known-signal response NMSE
  at most0.5. Repeated intervention rows are not independent sources.
- **Status.** Local averages favor smoothness in sampled directions; neither
  global smoothness nor cross-model semantic equivalence is established.
  The fixed second frequency is an exploratory follow-up on the same sources.
- **Reproduce.** `selective_transfer.py`, `selective_frequency_transfer.py`,
  `selective_transfer_analysis.py`; selective composition report and raw controls.
- **Ref.** ROADMAP §49.36–49.37; earlier oracle-normalizer controls in §49.34.
- **Score. 4/5.** Reproducible transport can preserve the model's error; semantic
  accuracy must be checked independently of patch validity and transfer fidelity.

#### PM-34 · Selective tones plus a modelled residual work, but extraction and depth matter
- **Claim.** FIT-selected hybrid TEST synthetic losses are
  **0.006031990799018737** (Bolt) and **0.004931439430488167** (TimesFM), with
  native-relative gains **1.0682909210254592** and **0.23171231421643068**.
  The conservative tree splits only **3/16** moderate-injection real contexts,
  versus **16/16** with the declared fixed stronger injection. At amplitude8,
  depth1 hybrid future losses are **0.3186258362659885** versus native
  **15.135874660784449** (Bolt), and **0.37227180446010033** versus
  **1.4132528995863665** (TimesFM), in original-history variance units.
  Requested depth3 gives **0.36046953460557996** and **0.4211007133624285**.
- **Models / layers.** Classical continuation of past-fitted continuous-frequency
  tones; native model predicts the intact residual. No oracle components or
  future labels enter extraction. Maximum three recursive residual peels.
- **Evidence.** Held-out synthetic behavior and exploratory controlled additions
  to16 real TEST sources, four per dataset. Strong additions share sources with
  the main study and do not increase independent n. Raw real pretraining overlap
  is unknown. No best TEST depth becomes a selection rule.
  At fixed19.2-hour period/phase, TimesFM native response NMSE improves
  **0.22965478913847118**→**0.03153184616947545** as addition amplitude2→8;
  this is amplitude/context dependence, not categorical frequency blindness.
- **Status.** Positive bounded feasibility; does not establish natural-data
  forecasting gains or internal additive representations. Abstention is an
  extraction boundary, not a negative result about an intervention never applied.
- **Reproduce.** `selective_tree.py`, `selective_strong_signal.py`,
  `selective_strong_audit.py`; replication `strong_signal_scores.json`, report.
- **Ref.** ROADMAP §49.35–49.37; contrasts earlier fixed-band PM-28.
- **Score. 4/5.** Tests the intended limited-decomposition adapter and shows why
  both residual preservation and an explicit abstention audit are necessary.

#### PM-35 · A continuation prior can cause attenuation without failed pattern learning
- **Claim.** In matched tiny-model training, changing only the unobservable
  survival probability of a future tone from0 to0.5 to1 changes predicted tone
  gain accordingly. At probability0.5, two paired MLP seeds give
  **0.49643564224243164/0.49768996238708496** and two transformer seeds give
  **0.49440494179725647/0.4973456859588623**. At probability1 the gains are
  **0.9938386082649231/0.994098961353302** and
  **0.994004487991333/0.996536910533905** respectively.
- **Models / layers.** Small MLP and two-layer transformer; identical observed
  histories, initialization, batches and200-step budget across survival arms.
  Exact conditional-mean supervision equals expected two-branch MSE gradients.
- **Evidence.** Controlled training intervention in two toy architectures;
  512 training histories,256 held-out histories, two training seeds. Both pass
  the prespecified conditional-mean error<=0.15 adequacy gate. Analytic marginal
  risk and irreducible-risk decomposition independently replayed.
- **Status.** Exploratory mechanism demonstration, not attribution of pretrained
  TSFM failures. Test histories do not increase independent training-seed n.
  The native Bolt and TimesFM outputs used in these studies are medians; this
  toy conditional-mean MSE result does not identify their training objectives.
  Initialization control: MLP tone-decoding errors
  **0.006750625085256504/0.006583728204450735** are already below p0-trained
  **0.016827709793355333/0.01536392071152469**. Transformer initialization
  **0.4061008965454886/0.45874250688843426** also beats p0-trained
  **0.47648969898855154/0.5042714721036291**. Recoverability alone therefore
  does not show that the p0 objective learned a forecasting concept.
  **Prediction-functional control:** matched MSE/absolute-loss training on the
  same two-future distribution gives different optimal behavior. Across two
  tiny architectures/two initializations, survival probability0.25 yields MSE
  gains **0.24789963829063225–0.24967727238006154** versus absolute-loss
  **0.0032150687063610115–0.004985426555465776**. At0.75 the respective gains
  are **0.7490467598869157–0.7529284809960199** and
  **0.9777613686899456–0.9897283558827841**. At0.5, absolute-loss gains vary
  **0.1109961110676092–0.2680873488456798** while remaining close to the
  nonunique median set. A low median gain need not encode a low survival prior.
  This is a toy objective demonstration, not a diagnosis of pretrained training.
- **Reproduce.** `prior_training.py`, `prior_training_analysis.py`;
  `runs/prior_training_pilot4951000/analysis.json` and raw predictions;
  `prior_objective_control.py`, `runs/prior_objective_control4963000/knowledge.json`
  (24 matched arms, two tests/two plants,96 independent arithmetic checks).
- **Ref.** ROADMAP §49.38.
- **Score. 3/5.** Shows why attenuated forecasts alone cannot diagnose excessive
  randomness assumptions or an inability to encode periodic structure.

#### PM-36 · Cross-model predictive usefulness exists in the synthetic pilot, but outputs can suffice
- **Claim.** Predicting TimesFM's forecast, input-only loss
  **0.026855664954461247** falls to **0.01769764292154032** with Bolt hidden
  states, but falls further to **0.013847275247863784** with Bolt forecasts.
  Reverse-direction losses are **0.02398174312703629**, **0.020036055751863173**,
  and **0.010813336181818452**. For predicting Bolt's fixed fusion response,
  input+TimesFM hidden improves **0.0024933167683625517**→**0.0021698780749397047**;
  input+TimesFM forecast gives **0.0024418541423520124**.
- **Models / layers.** Frozen Bolt/TimesFM half-depth full-token states; target
  fusion is first-block/depth1 equal pure composition. Train-only transforms and
  grouped inner selection; strong input features include the full history.
- **Evidence.** Source-held-out predictive comparison on16 synthetic TEST seeds
  from earlier studies, not fresh confirmation or causal cross-model alignment.
  Source-shuffled nulls and rotation invariance checked. Stable CV tie handling
  was required to avoid roundoff-induced changes in selected predictors.
- **Status.** Exploratory positive for bounded predictive usefulness; not general
  semantic equivalence. Real and failure-prediction endpoints are mixed/negative;
  constant-label endpoints cannot demonstrate discrimination.
  **Fresh replication (2026-11-16):**64 new sources, frozen original FIT probes,
  all three primary contrasts pass. Input→input+source-hidden losses are
  **0.023864581292049478→0.015733820374062967**,
  **0.01912109637372958→0.01604583723607949**, and
  **0.002156430411168464→0.001852114482009267**, in the directions above.
  Exploratory gains adding hidden to input+source-forecast are
  **0.0011738879565609288/0.0008663409304243716/0.0001372677512059746**;
  all source intervals are positive. This measures finite-data readout utility,
  not new information beyond the input or equivalent semantics.
  **Generic-feature follow-up:** Random Fourier/ReLU input features plus source
  forecasts give TimesFM-emulation loss
  **0.012719115560900612/0.012726500254948755**, beating trained-hidden
  **0.013207666764194471**. Reverse-direction ReLU also matches trained hidden.
  For Bolt fusion-response prediction, trained-hidden loss
  **0.0020740497112315586** remains below random
  **0.0022476849828418953/0.0022478083458015554**. This post-hoc reuse narrows
  the interesting trained-feature advantage to the intervention-response target;
  new-condition generalization required a separate test. Width/readout budgets
  match, not network architecture or pretraining cost.
  **Fresh condition-shift test:**20 new FIT/32 TEST sources and five conditions,
  with exactly96 FIT rows, retain the trained-state advantage for Bolt fusion
  response. Trained loss **0.002255216479176824** is below random Fourier
  **0.0029625792146836614** and random ReLU **0.002444654328383262**;
  relative reductions **23.876584700280104%/7.74906484761468%** pass both
  frozen≥5% criteria and positive97.5% paired-source intervals. Old frozen
  probes extrapolate poorly to the shifted conditions; the positive is for
  matched-budget local refitting, not a universal cross-model translator.
  Excluding the anchor descriptively reduces the trained-hidden advantage over random ReLU to
  **3.584563021955512%**, with its interval crossing zero. The primary aggregate
  passes, but the trained-feature advantage is not uniform across shifts.
- **Reproduce.** `comparability_pilot.py`, `comparability_core.py`;
  `runs/comparability_pilot4951000/results.json`, protocol/amendment and controls;
  `comparability_replication4952000` preserves frozen probes and raw predictions.
  `comparability_random4953000` records generic-feature controls; original
  replication audit detects25 plants and replays60 intervals; new controls
  detect11 plants and replay120 arithmetic comparisons. Fresh
  `comparability_conditions4962000` catches29 plants and independently replays
  422 intervals and both primary decisions.
- **Ref.** ROADMAP §49.38,49.40.
- **Score. 4/5.** Replaces an all-or-nothing comparability claim with a concrete,
  baseline-relative test and exposes when hidden access is unnecessary.

#### PM-37 · Continuation gains depend on future stability, and native models can use change cues
- **Claim.** With identical histories and equal probability of persistent,
  decaying, drifting and stopped futures, TEST native/hybrid losses are Bolt
  **0.7044685896391737/0.546102676503224** and TimesFM
  **0.4974279871980144/0.5385035414665008**. TimesFM's hybrid becomes preferable
  above persistence probability **0.8058511886716185**, holding the other future
  probabilities equal. Visible decay/cessation cues improve both native models
  on **4/4** TEST seeds; drift helps TimesFM **4/4**, Bolt **2/4**.
- **Models / layers.** Frozen native forecasts versus fitted-tone continuation
  plus native residual; period19.2, amplitude8,128-context/32-horizon, cue onset−32.
- **Evidence.** Exploratory controlled behavioral pilot, eight seeds/four TEST.
  Identical histories are deduplicated; singleton calls pass exact repeat and
  self-patch gates. Independent score replay and planted faults pass.
- **Status.** A counterexample to categorical claims that models ignore visible
  changes. Crossover is a hypothetical risk boundary, NOT an inferred training
  prior. The tree accepts cued decay/drift despite harmful continuation and
  abstains on cessation; past extraction validity does not ensure persistence.
  **Fresh replication:**64 seeds,32 TEST, two periods and five cue leads.
  A FIT-only stability veto fails to beat always-native by the declared source
  sign-test screen in either model. Against always-hybrid it wins30/32 Bolt
  and32/32 TimesFM; Bolt's mean gain **0.024316141847413635** has interval
  **[-0.012248115026278854,0.05250180964470645]**, so majority wins do not
  establish population-mean improvement. TimesFM gain is
  **0.11701714686534168**. Recent32-step refitting beats full-history fitted
  tones on visible-changing cases32/32 sources at both periods, with gains
  **0.12318180842823585/0.11615768959478193**. This diagnoses a continuation
  assumption in the adapter, not the pretrained training data.
- **Reproduce.** `persistence_*`; `runs/persistence_pilot4951000/singleton_geometry/`;
  three knowledge.json files preserve failed batching, calibration and recovery.
  `persistence_replication4953000` independently replays35840 scores,35840 cue
  comparisons,8960 risks and six primary means; twenty tests/nine new plants
  plus a score plant pass/catch.
- **Ref.** ROADMAP §49.38,49.40.
- **Score. 4/5.** Separates unavailable future information from ignored observable
  evidence, and identifies an adapter assumption that can fail.

#### PM-38 · Response decoding is normalization-sensitive; local nonlinear interactions can still be smooth
- **Claim.** First-block coefficient-decoding NMSE native→background-locked is
  Bolt **0.4996478849685851**→**0.012752598094436797**, TimesFM
  **0.4723127043833158**→**0.01359228639593964**; raw-input baseline
  **2.9406281795183107e-07**. Native forecast chord curvature shrinks with radius
  on **48/48** Bolt and **47/48** TimesFM paths; paths share eight sources.
- **Models / layers.** First/quarter/half blocks; fixed-phase frequency paths near
 24h/19.2h, amplitudes0.1/0.5/2, native and anchored native normalization.
  Four-point interaction differences are nonzero even with normalization locked.
- **Evidence.** Instrumentation pilot: four synthetic backgrounds and four reused
  real FIT sources. Seven tests/plants and21600 independent arithmetic checks.
- **Status.** Exploratory. Decoder improvement is not information recovery until
  omitted native normalization metadata is controlled. **Follow-up (2026-11-16):**
  normalized-target decoding with correct history-only inverse reduces native
  first-block errors to **0.009669252562936726/0.030581905688274888** (Bolt/TimesFM),
  improving8/8 sources per model. That inverse explicitly uses observed background
  Fourier coefficients; matched controls receive the same information. Its
  training-mean controls score **0.8039847473064277/0.8448163768887251**. The
  endpoint route without that coefficient correction gives hidden errors
  **0.1776317107733182/0.20469722023739187** versus matched inputs
  **0.2009918735389694/0.1998988078602276**. These controls support a decoder-
  parameterization confound, not blanket hidden superiority or hidden-only
  information recovery. Small-radius smoothness
  does not imply additive composition, semantic invariance or global smoothness.
  The crossed-phase fresh-background follow-up is recorded below.
  **Focused expansion:**16 fresh synthetic backgrounds×four phases, actual
  RMS radii0.01/0.005/0.002. All16 source averages show decreasing curvature in
  every tested model/policy/axis/endpoint setting; five individual phase curves
  are exceptions. Native forecast frequency-curvature medians decrease
  **0.040190041464709246→0.01285684887688486** (Bolt) and
  **0.0873284666297234→0.01742486071070292** (TimesFM). Yet19.2h response NMSE
  remains **1.398347641932045/2.3657490304769673**. Local smoothness coexists
  with inaccurate component response and small calibrated nonlinear interactions.
- **Reproduce.** `response_surface_*`; `runs/response_surface_pilot4951000/knowledge.json`,
  raw arrays, analysis and report; `response_surface_normalization4953000`
  (eight tests/plants,7920 independent arithmetic checks across144 fits).
  `response_surface_focused4954100` adds45824 arithmetic checks and56 source-
  summary checks; the larger full grid is parked.
- **Ref.** ROADMAP §49.38–49.40.
- **Score. 4/5.** Supplies competing explanations for apparent representational
  messiness and prevents conflating smooth nonlinearity with instability.

#### PM-39 · Conservative amplitude attenuation alone does not explain the measured TSFM responses
- **Claim.** For19.2h additions at amplitude0.1, mean response NMSE is
  **3.390003549579674** (Bolt) and **2.1344526166056355** (TimesFM). Optimally
  fitting a separate scalar in[0,1] to each response still leaves normalized
  residual **2.337825752154055** and **0.9575308104404547**. A free-phase
  same-frequency plane still leaves off-frequency error **0.832073982853724**
  and **0.40932874401172437**.
- **Models / layers.** Native forecasts from the response-surface pilot; also
  evaluated under locked normalization. Error units are known-signal energy.
- **Evidence.** Descriptive explanatory projections, eight backgrounds (four
  synthetic, four reused real FIT sources). Independent original-score agreement
  for192 rows; exact error partition and planted-bound check pass.
- **Status.** Negative for a sufficient scalar-conservatism explanation in this
  pilot, not a diagnosis of training data. Rowwise projections are optimistic
  explanatory bounds, not deployable calibrations. At amplitude2, bounded
  residuals decrease to **0.40867184504571824/0.10954368175736659**.
- **Reproduce.** `attenuation_analysis.py`;
  `runs/attenuation_explanation4951000/knowledge.json` and raw response bank.
- **Ref.** ROADMAP §49.38–49.39; contrast controlled-training PM-35.
- **Score. 4/5.** Distinguishes a plausible prior mechanism from a complete
  explanation; waveform and background effects remain to be explained.

#### PM-40 · Natural Fourier-hybrid gains vary by domain, and short hindcasts can miss them
- **Claim.** On four new electricity TEST sources, depth1 hybrid changes native
  MSE from **0.32830079848935756** to **0.3230221699359904** for Bolt and
  **0.6192148434100198** to **0.35038952478445495** for TimesFM. On four traffic
  sources it worsens **0.266875622142397**→**0.5146018681958637** and
  **0.2869937946993105**→**0.4171051966105375**. TimesFM's two-hindcast selector
  rejects all electricity hybrids, missing three improvements and one tie.
- **Models / layers.** Native Bolt/TimesFM versus past-fitted tone continuation
  plus native residual,128-context/32-horizon; unmodified real data.
- **Evidence.** Exploratory behavioral source holdout:26 sources across four
  domains, half FIT/TEST; only one pedestrian TEST source, no interval there.
  Own-history normalization at each origin; main future excluded from selection.
- **Status.** **Fresh replication:**64 new TEST sources/domain,192 total.
  TimesFM electricity improves **0.5515747684009784→0.4676360158302627**;
  gain **0.08393875257071565**, six-comparison interval
  **[0.015307078093565,0.1583223135327461]**,24 wins/35 ties/5 losses.
  Bolt electricity does not improve; traffic is harmful in both, with gains
  **-0.1539340790284484/-0.1346564054134044** and adjusted intervals below zero.
  Neither M4 model establishes improvement. Seasonal-naive24 electricity mean
  MSE **0.3993750427653811** is lower than the hybrids; its paired difference
  from TimesFM hybrid remains uncertain. No universal improvement or failure.
  Pretraining overlap unknown; no TEST-best depth deployed. The smaller pilot's
  selector false-rejection observation is not a universal selector failure.
  **Normalization-path control:** on the same64 electricity sources, TimesFM
  ordinary/locked hybrid gains are **0.08393870612092993/0.08201179427098884**
  after consistent singleton replay. Locking forward/inverse normalization to
  full-history values therefore preserves the observed positive; their small
  difference is uncertain, not demonstrated equivalent. Bolt remains negative
  in mean for both. This is a reused-cohort controlled intervention, not fresh
  confirmation or a training diagnosis. Old batch4/singleton rounding changes
  no source-gain sign and is bounded explicitly.
- **Reproduce.** `natural_validation_*`;
  `runs/natural_validation_pilot4951001/knowledge.json`, `analysis.json`,
  `verification.json` (24 tests and8 amendment plants).
  `natural_validation_replication4955000` adds17 tests/21 plants and independent
  exact MSE/selector replay in audit.json. `persistence_electricity_lock4957200`
  adds384-loss/2304-partition/six-contrast/256-drift-bound independent replay.
- **Ref.** ROADMAP §49.40.
- **Score. 4/5.** Separates synthetic persistent-tone feasibility from natural
  continuation uncertainty and tests a plausible past-only selection remedy.

#### PM-41 · Some off-tone distortion aligns with background modulation, but a simple template is insufficient
- **Claim.** At19.2h/amplitude0.1, response error outside the added-frequency
  plane is **0.832073982853724/0.40932874401172425** for Bolt/TimesFM.
  Fitting constant/trend and then the unperturbed forecast reduces the residual
  to **0.3345255393285157/0.19192169094791175**. The background template beats
  the95th percentile of19 time-permuted templates on **6/8** and **5/8** sources.
  Locked-normalizer residuals remain **0.32135854483142834/0.18529909782956655**.
- **Models / layers.** Bolt/TimesFM native and background-locked forecast responses.
- **Evidence.** Descriptive per-response fitted projections on eight existing
  pilot backgrounds. One test/three plants and192 arithmetic replays pass.
- **Status.** Partial descriptive explanation, not a learned prediction method
  or causal mechanism. Short-horizon correlated bases and projection order limit
  interpretation; simple background modulation leaves substantial residual error.
- **Reproduce.** `background_modulation.py`;
  `runs/background_modulation4956100/knowledge.json`, projections and verification.
- **Ref.** ROADMAP §49.40; follows PM-39.
- **Score. 3/5.** Narrows the alternatives to scalar attenuation without treating
  every nonlinear response as incoherent or as a normalizer artifact.


### Steering research additions (2026-11-16)

#### PM-42 · Real Fourier-hybrid benefits extend beyond the extracted frequency
- **Claim.** On64 new electricity sources, TimesFM hybrid gain
  **0.08393875257071565** partitions into **0.03509242848460109** in the
  past-fitted sine/cosine plane and **0.04884632408611454** outside it.
  **0.5168608174353616** of the actual forecast-update energy is off-plane.
  Traffic losses chiefly occur in the tone plane: gains
  **-0.13961662375115133/-0.12706543331818337** for Bolt/TimesFM.
- **Models / layers.** Native versus depth1 hybrid forecast outputs; same192-source
  natural replication, with the frequency frozen from each context's past.
- **Evidence.** Descriptive exact orthogonal error/update partitions,384 paired
  model/source rows;270 extraction-eligible. One test/three plants and raw replay.
- **Status.** The benefit is not solely a repaired tone amplitude or phase.
  The modelled residual changes other forecast components. This does not identify
  a true decomposition of natural futures or a causal training mechanism.
- **Reproduce.** `natural_gain_location.py`;
  `runs/natural_gain_location4956300/knowledge.json`, partitions and verification.
- **Ref.** ROADMAP §49.40; follows PM-40.
- **Score. 4/5.** Makes the observed real-data gain more interpretable and guards
  against attributing every Fourier-adapter improvement to one extracted concept.

#### PM-43 · Tokenization can create input discontinuities without arbitrary encoder behavior
- **Claim.** Chronos-T5-tiny has a one-token change for raw-history pairs only
  **7.573261841801273e-09** RMS apart on average across eight sources.
  Relative jumps average **0.04599361338361545** at embeddings and
  **0.02180104422917018** at the final normalized encoder output. On continuous
  embedding paths, first-step probability curvature shrinks
  **0.18482274082093753→0.03314285486742809** as radius0.25→0.03125.
- **Models / layers.** Cached Chronos-T5-tiny, encoder and deterministic first
  decoder-step probabilities; one architecture-matched random initialization.
- **Evidence.** Controlled descriptive architecture pilot, eight synthetic
  backgrounds. Exact same-token, repeat and embedding-replay gates;720 arithmetic
  checks. Three tests/five plants include a precision-resolution guard.
- **Status.** Known frontend discontinuity is distinct from internal incoherence.
  Random output probabilities are nearly uniform;32 probability-curvature cells
  are precision-limited and cannot support a smoothness comparison. Embedding
  interpolation is off-manifold; no multi-step forecasting or semantic claim.
  **Fresh paired16-source replication at tiny/small sizes:** input distance
  **4.837472450240249e-09**, final-encoder relative jumps
  **0.023675344465227722/0.02218872810943274**, first-step probability TV
  **0.002784460601826859/0.0029375174143024202**. All16 pretrained curves per
  model shrink between largest/smallest embedding radii; full four-radius
  monotonicity holds13/16 and15/16. Random probability ratios remain unresolved.
- **Reproduce.** `token_boundary.py`, `token_boundary_analysis.py`;
  `runs/token_boundary_pilot4956000/audit/knowledge.json`;
  `token_boundary_replication4961000_tiny` and `_small`, each1440 audit checks.
- **Ref.** ROADMAP §49.40.
- **Score. 3/5.** Prevents a tokenizer artifact or numerical floor from being
  misreported as general representational messiness.

#### PM-44 · Other marginal quantiles do not rescue weak-tone response failures
- **Claim.** Weak19.2h median response NMSEs are
  **2.5675212280679705/1.9717594022240545** for Bolt/TimesFM; averaging nine
  native central quantiles gives **2.5120652628795126/1.9689040160935263**.
  Even per-response oracle-best quantiles leave
  **2.161041132950336/1.801664427855226**; locked-normalizer oracle errors are
  **2.0141060449313395/1.8573412859313922**.
- **Models / layers.** Nine native forecast quantiles; both native point outputs
  are medians. Eight reused backgrounds, four phases, two periods, three amplitudes.
- **Evidence.** Descriptive controlled-response pilot. Exact median/native replay,
  repeat and normalization gates; two tests/three plants,3072 arithmetic checks.
- **Status.** Negative for a sufficient median-versus-other-quantile explanation
  in these weak-tone cases. Width changes do not establish calibrated uncertainty;
  marginal quantiles do not define joint forecast paths. Oracle selection is
  explanatory only. Stronger tones have much lower errors.
- **Reproduce.** `quantile_response.py`;
  `runs/quantile_response4956200/knowledge.json`, raw quantiles and verification.
- **Ref.** ROADMAP §49.40; complements PM-39 rather than attributing training data.
- **Score. 4/5.** Tests an alternative explanation arising from the actual native
  output functional, instead of diagnosing a model from its median alone.

#### PM-45 · Some incorrect weak-tone responses have a transferable phase structure
- **Claim.** A two-coordinate sine/cosine response map fitted on the other15
  backgrounds predicts Bolt's period19.2 weak-tone response better than ideal
  continuation on16/16 sources: phase-energy-pooled NMSE
  **1.3906871921613242→0.5068692957225513**. TimesFM improves in mean
  **2.3537098800938168→1.999695124522527**, but only8/16 sources improve.
  Per-source fitted maps leave **0.2314247904913106/0.9018266927220593**;
  these are optimistic explanatory ceilings, not held-out predictions.
- **Models / layers.** Bolt-small/TimesFM2.5 median forecasts;16 reused fresh-
  replication backgrounds, four phases, amplitude0.1, periods24 and19.2.
- **Evidence.** Descriptive response prediction; no new native inference.
  Exact odd/even/off-frequency energy decomposition, DC-leakage sensitivity,
  source-stratified bootstrap;4672 independent checks, maximum discrepancy
  **8.929789570134933e-15**. Model response prediction is not true-future accuracy.
- **Status.** Systematic phase response is a bounded alternative to undifferentiated
  messiness. Four phases at one amplitude do not establish an LTI system or a
  global response law; background dependence and off-frequency errors remain.
  Fresh-source validation is a separate experiment, not implied by this reuse.
  **Frozen-map validation:**16 new sources at four phases and nearby frequencies
  retain a period19.2 advantage over old-fitted scalar attenuation. Bolt scalar
  **0.7779709336359619→0.5306089803018366** and TimesFM
  **1.9283313395848056→1.7485319484160944**. Paired mean gains
  **0.2473619533341253/0.17979939116871133** have family4-adjusted intervals
  **[0.1632782914560016,0.3182855976544267]** and
  **[0.125550594671179,0.23033485864196618]**. Both identity comparisons also
  retain positive adjusted intervals. Thus simple attenuation alone is
  insufficient to describe these responses; a frozen phase map adds predictive
  value. Intervals condition on the four equally weighted background types.
  **Oblique-phase check:** eight additional angles on the same16 transfer
  backgrounds retain scalar-minus-map gains **0.24816757777846288** for
  Bolt and **0.18448195097911668** for TimesFM, with positive family4-adjusted
  intervals. Coefficients remain byte-identical. These are new phase inputs,
  not independent new backgrounds; no universal phase-law claim follows.
- **Reproduce.** `response_phase_study.py`, `response_phase_audit.py`;
  `runs/response_phase4954200/knowledge.json`, raw arithmetic and bootstrap indices;
  `response_phase_transfer4954300` contains400 inputs, frozen zero/scalar/
  complex/full-map controls, fresh responses and10860 independent score checks.
  `response_phase_oblique4954500` adds144 native inputs and5164 independent
  arithmetic checks, with six tests/eight planted regressions detected.
- **Ref.** ROADMAP §49.40; follows PM-38/39/41.
- **Score. 4/5.** Separates predictable wrong behavior from faithful continuation.

#### PM-46 · Native FFN selectivity is model-dependent and does not require an SAE
- **Claim.** In the first-block RMS/four-feature synthetic TEST slice, Bolt's
  native FFN coordinates classify five controlled perturbation types with
  accuracy **0.825**, versus **0.525** averaged over three structured rotations,
  **0.5750000000000001** for FIT PCA and **0.925** for input coordinates.
  TimesFM's corresponding accuracies are **0.7/0.7833333333333333/0.75/0.925**.
  The full registered first/half-depth, mean/RMS and feature-budget table is
  retained; this informative slice is exploratory, not a selected primary test.
- **Models / layers.** Actual post-activation inputs to Bolt `DenseReluDense.wo`
  and TimesFM `ff1`, plus matched residual captures. Sixteen fresh synthetic
  backgrounds split8/8 and eight reused real backgrounds split4/4; five
  equal-input-RMS perturbation types with phase/sign variants.
- **Evidence.** Held-out perturbation selectivity under background-locked
  normalization; no SAE and no selected-unit causal intervention yet. FIT-only
  selection, source grouping, exact replay and intervention reach gates pass.
  Eleven tests/16 plants and independent replay of270 feature records/4680
  source summaries pass, maximum metric error **9.836575998178887e-14**.
- **Status.** Native coordinates can be useful, especially early Bolt, but this
  does not establish general monosemanticity. Structured rotations are not Haar;
  PCA has a different candidate count; raw input itself is highly informative.
  Small reused real cohorts establish no fresh real-data generalization.
  **Causal follow-up:** four FIT-selected early Bolt units, reset to background
  values, remove **0.006158264488730194** of the ramp/noise response on eight
  fresh sources,95% interval **[−0.00486514095782705,0.017475451719261043]**.
  Their advantage over assigned activity-matched controls is
  **0.012473076834747427**, interval **[0.0021393454858827585,
  0.022147086103784748]**; controls themselves increase the response on average.
  This small relative advantage does not establish strong selective removal.
  Actual edit and output reach are audited separately from full-FFN reach.
  **The larger causal test did not replicate that advantage.** On32 fresh
  sources with four frozen control sets crossed with every source, selected
  signed removal before subtracting controls is **0.0008338539038024147**, interval
  **[−0.00640342426749434,0.007862991279733313]**. The paired advantage is
  **0.0003116219633811186**, interval
  **[−0.006925097142577513,0.007340577240027061]**. Both intervals lie within
  the preregistered descriptive±0.02 band. This bounds this particular k4 edit;
  it does not show that all native neurons are causally irrelevant. FIT activity
  matching also fails to equalize actual held-out edit energy: selected target
  edits have **2.7226714950868156×** the mean-control projected energy. All160
  selected edits are nonzero and reach the output. The negative concerns aligned
  response removal, not absence of any causal output change.
- **Reproduce.** `ffn_selectivity_study.py`, `ffn_selectivity_analysis.py`;
  `runs/ffn_selectivity_pilot4961000/knowledge.json`, `compact_record.json`,
  `independent_audit.json`. Native capture480 cells/model,37.56/90.35seconds.
  `ffn_selectivity_causal4963000` preserves384 Bolt cells, exact edit support,
  nine tests/15 plants and240 event/567 source-summary independent replays.
  `ffn_selectivity_causal_replication4964000` adds1088 Bolt cells and independent
  replay of800 events/392 source summaries, maxeventerror3.3306690738754696e-16.
- **Ref.** ROADMAP §49.40; addresses the scope distinction in §49.39.
- **Score. 3/5.** Tests genuine neurons rather than extrapolating a monosemanticity
  claim from residual dimensions or successful sparse reconstruction.

#### PM-47 · Limited Fourier separation fails on a chronological weather holdout
- **Claim.** Fixed depth1 increases equal-station normalized MSE from
  **0.4729334720068331→0.6547303222032483** for Bolt and
  **0.4503466537272577→0.5064548545682812** for TimesFM on October2026
  airport temperatures. Only **1/8 and3/8** station means improve. Seasonal24
  MSE is **0.6059731631636294**; both native models have lower observed means.
- **Models / data.** Exact cached Bolt/TimesFM revisions predate observations
  (2025-11-21/2025-10-02). Eight fixed US airports, four fixed origins,
  history128/horizon32;30 complete futures, JFK two and other stations four.
  The fixed rule accepts18 histories and abstains on12. No threshold tuning.
- **Evidence.** Behavioral chronological-observation holdout, descriptive pilot.
  Source weighting is by station, not window. Official GHCNh source/archive
  hashes, checkpoint chronology, target-read prohibition, exact singleton
  repeats/abstention identity and independent score replay are preserved.
  Five native-protocol tests/11 plants plus an isolated score-corruption plant;
  replay240 window losses/64 station losses/eight intervals.
- **Status.** Negative for a general Fourier benefit on these genuinely later
  observations; not a population theorem. Shared weather limits station
  independence. Archived observations are not a real-time vintage, null quality
  flags mean unflagged rather than verified QC completion, and familiar
  locations/climates may occur in training. A zero-residual physical-temperature
  baseline discards DC and is not a meaningful claim that native residuals are
  necessary; it remains recorded as a diagnostic only.
  **Saved-array follow-up:** restoring mean/last residual levels gives errors
  **0.8649854790083853/1.1309893084967446**, still worse than native/hybrid.
  Weather hybrid gains partition into history-fitted tone-plane/off-plane
  **−0.021876413106958606/−0.15992043708945658** for Bolt and
  **−0.023748947453231563/−0.03235925338779193** for TimesFM. Much damage
  lies outside the peeled frequency; this does not attribute all harm to bad
  analytic continuation. Three tests/four plants and independent replay pass.
  **Joint normalization intervention:** locking residual forward/inverse scales
  to the full history worsens Bolt hybrid MSE to **0.8431448147928856** on all
  eight station means. TimesFM becomes **0.4763265507043724**, but improves only
  two stations and its paired interval crosses zero. Residual renormalization
  alone is therefore not a general explanation or repair of the weather failure.
  Four tests/five plants and independent replay of180 losses/1080 partition
  scalars pass; the forward and inverse contributions are not separated.
- **Reproduce.** `postrelease_weather_prepare.py` and related pilot modules;
  `runs/postrelease_weather_202610/ghcnh_oct2026/knowledge.json` for provenance,
  `runs/postrelease_weather_pilot202610/knowledge.json` for forecasts/audits.
- **Ref.** ROADMAP §49.40; complements the public-source electricity positive PM-40.
- **Score. 4/5.** Tests the frozen idea on observations that postdate the weights,
  while retaining an informative negative and a strong simple baseline.

#### PM-48 · Longer-period waveform fits explain a little more weak-tone response, but no unique oscillator
- **Claim.** With old16-source-only selection, Bolt's response to a19.2-period
  tone is better predicted by a24-period output basis: fresh response NMSE
  **0.5306089803018366→0.48412325715727056**. TimesFM selects22 and improves
  **1.748531948416094→1.726079453649708**. The old mean32×2 response waveform
  performs better still (**0.45748844616145007/1.6466122728733559**), at greater
  fitted capacity. Period24 controls select24 and receive no frequency-shift gain.
- **Models / data.** Native median response, weak amplitude0.1, old16 FIT and
  reused fresh16 TEST backgrounds/four phases,32-step horizon; no new inference.
- **Evidence.** Saved-array descriptive follow-up. Nine output-period candidates,
  source-CV selection entirely on old responses, all FIT/TEST curves retained.
  Eight tests/nine plants and768 independent checks pass, maximum discrepancy
  **2.3314683517128287e-15**.
- **Status.** Conditional waveform distortion, not identified frequency snapping
  or training-prior causality. With matched DC/trend nuisance terms Bolt selects
 22 instead of24; TimesFM's added shift gain becomes uncertain. Runner-up gaps
  are small, the horizon contains few cycles, and effects differ by background
  type. Good coefficient conditioning does not identify a unique frequency.
- **Reproduce.** `response_phase_frequency4954400/knowledge.json`, frozen FIT
  selection, full curves, source scores, figures and independent audit.
- **Ref.** ROADMAP §49.40; follows PM-45.
- **Score. 3/5.** Tests a concrete phase-distortion alternative while exposing its
  finite-horizon and nuisance ambiguity.

#### PM-49 · Shared position-dependent state structure explains much more Bolt variance than TimesFM variance
- **Claim.** A source-weighted FIT position-conditioned mean reduces TEST error
  relative to the global FIT mean by **80.30466851357155%** for Bolt but only
  **10.107015663060814%** for TimesFM. Corresponding error ratios are
  **0.19695331486428444/0.8989298433693919**. Existing PCA top8/top16 coefficient
  reconstructions are substantially better: Bolt
  **0.021846093615440033/0.009173964053706064**; TimesFM
  **0.08610490644031232/0.04259929649012923**.
- **Models / layers.** First-block full-token states, frozen dictionary cohort:
  32 FIT/32 TEST sources,112 rows each;16 synthetic andfour per real domain.
- **Evidence.** Post-hoc geometric compression diagnostic, no native inference
  or dictionary training. Five tests/five plants,204 independent metric checks;
  maximum discrepancy **4.395839963453344e-16**.
- **Status.** Fixed means cannot respond to input perturbations, so good global
  reconstruction alone does not establish preservation of useful input changes.
  The baseline may contain common content/normalization/token roles; this is
  not causal attribution to positional embeddings. It does not explain all PCA
  gains. PCA uses a full fitted basis with adaptive sparse coefficients and
  greater stored capacity, not a single fixed rank8/rank16 subspace.
- **Reproduce.** `runs/comparability_position_mean20261116/knowledge.json`,
  source scores, parameter counts, gates and independent audit.
- **Ref.** ROADMAP §49.40; qualifies the dictionary reconstruction comparison.
- **Score. 3/5.** Provides a static-state baseline that separates compression from
  responsiveness and exposes a model-dependent nuisance contribution.

#### PM-50 · Sparse state reconstruction and weak-response preservation are different objectives
- **Claim.** At k16, PCA whole-state NMSE is **0.009173964124244276** for Bolt
  and **0.042599296711000684** for TimesFM, while saved sinusoid-response NMSE
  is **0.3008273079816649/0.41102390528013866**. TopK response errors are
  **0.3720378072540668/0.40271957749287796**, nested-inspired
  **0.3003756530838683/0.4071664697199135**, averaged over three seeds.
  Low state reconstruction error therefore does not guarantee small normalized
  error on weak state differences. These metrics have distinct denominators.
- **Models / layers.** First-block Bolt/TimesFM, width512/1280; native axes,
  rotations, full FIT PCA with adaptive signed top-k coefficients, TopK and a
  Matryoshka-inspired nested-prefix variant. k8/16;32 FIT/32 TEST sources.
- **Evidence.**44 fits,44 native stages, four stability stages and two saved
  response audits. Actual activity is matched; neural methods use500 steps and
  three seeds. Nested has1.25× forward dense-matmul cost at equal steps/parameters.
  Nineteen tests/19 plants,8890 arithmetic checks,880 native identity gates.
- **Status.** No uniform SAE advantage. PCA k16 native point distortion is
  **0.01136767773111914/0.028067500051962135** over20 receivers; TimesFM k8
  TopK beats PCA (**0.06998571582496164** versus **0.10808140090652474**).
  Preservation is not forecast improvement or semantic validity. PCA stores a
  full fitted basis; signed and nonnegative codes are different constraints.
  **2048/3520** causal response cells have invalid random geometry and cannot
  support null-relative semantic claims. Exact base replay finds119/120 zero
  TimesFM TopK-k8 edits;478/480 cached ± edits were zero, but these are proxies, not certificates for
  unsaved singleton branches. Independent ± random edits are not calibrated
  smooth-response nulls. Earlier stronger null wording is explicitly superseded.
- **Reproduce.** `dictionary_pilot*`, `dictionary_core.py`, `dictionary_review.py`;
  `runs/dictionary_pilot4951000/recovery_exit_monitor/knowledge.json`,
  `comparison_summary.json`, `validity_audit/` and `response_audit_*`.
  Initial monitor failure and recovery provenance remain saved.
- **Ref.** ROADMAP §49.40; follows PM-49 and complements genuine-neuron PM-46.
- **Score. 4/5.** Supplies matched no-SAE/modern-SAE baselines and catches an
  interpretability-control failure before treating zero effects as model findings.

#### PM-51 · Sparse encoders can introduce much more roughness than the states they encode
- **Claim.** Across frozen PCA/TopK/nested maps at k8/16, the ratio of mean
  decoded actual-path chord RMS to mean raw-state chord RMS is **76–130×**
  for Bolt and **106–189×** for TimesFM under native normalization. With
  normalization locked, the ranges are **79–121×/129–231×**. These are
  ratios of source means over balanced paths, not average pathwise ratios.
- **Evidence.** Sixteen saved synthetic backgrounds, first-block residual states,
  768 matched-radius paths, one frozen dictionary seed, no fitting or native
  inference. All24 sparse-arm means exceed their raw-state counterparts; this
  need not hold for every individual source or path. Isolating the dictionary
  on a straight activation chord leaves substantial roughness; identity and
  fixed-first-k linear PCA have isolated chord RMS at roundoff
  (maximum **1.5983290058905532e-15**). Adaptive top-k PCA is nonlinear.
- **Interpretation.** Apparent feature jaggedness can arise in the measuring
  dictionary. It cannot automatically be attributed to the model. Actual native
  paths also have nonzero curvature. The isolated activation interpolation is
  off-manifold, and compression changes fidelity and secant responses; this
  is not a ranking of semantic quality. FP64 changes dictionary arithmetic
  over saved FP32 states, not native-model precision.
- **Validation.** Six tests/six corrected plants,36 independently replayed arms,
  6480 source-summary checks;90.856seconds,1.667GB peak, one CPU thread.
  An initially ineffective planted mutation is retained as a failed validation
  attempt. Resolution-floor sensitivity and absolute chord sizes are saved.
  **Fixed-support control:** with the same weights, freezing TopK membership
  at each actual path midpoint removes **99.30–99.79%** of mean actual-path
  chord RMS across six model/method cells. Keeping ReLU live or freezing its
  central gate gives identical saved-path metrics, isolating TopK membership
  switching in this design. Isolated fixed-map chord RMS is at roundoff.
  Hidden perturbation-response RMS error falls **61.58–82.37%**; mean state
  reconstruction error slightly worsens in five cells and improves in TimesFM
  TopK. All16 sources improve response error in every cell. These gates are
  chosen locally at each path's midpoint, not a deployable global encoder.
  Four tests/four plants,2898 summary checks and18-arm dense replay pass;
  50.887seconds,1.068GB peak, no fitting or native calls.
- **Reproduce.** `runs/dictionary_smoothness4965000/leading_findings.json`,
  `knowledge.json`, `compact_record.json`, independent replay and aggregation audit.
  `dictionary_smoothness_fixed_support4965100` contains the controlled gate
  comparison and corrects the parent's active-support/ReLU wording explicitly.
- **Ref.** ROADMAP §49.40; follows PM-38/50.
- **Score. 4/5.** Separates measurement-induced nonlinearity from model geometry.

#### PM-52 · Training a dictionary to preserve responses changes what it preserves
- **Claim.** A paired-response loss added to Bolt TopK-k16 reduces held-out
  sinusoid response NMSE **0.33706543732597316→0.08178803929115522** and
  noise response NMSE **0.6499851374406904→0.2761959152644845**, at the cost
  of global-state NMSE **0.013896715445253071→0.02355438870740306**.
  On16 focused sources with held-out directions/phases, response NMSE falls
  **0.4779692525911392→0.1367897436126779**. Every32 old TEST source and
  every16 focused source improves in the corresponding response comparison.
- **Design.** One new seed4965000, frozen lambda0/lambda1, same initialization,
  batches, architecture,500 steps and computed branches. Source-weighted
  central-difference reconstruction is normalized with FIT-only scales/floors.
  Frozen adaptive PCA-k16 gives global/sine/noise/focused NMSE
  **0.00917/0.30083/0.61419/0.44724**. Measured lambda1 activity is
  **15.99757**, not exactly16. Six tests and323 independent arithmetic checks
  pass;22.25seconds,0.984GB peak, one thread.
- **Scope.** This is a response-aware-inspired pilot, not a TSAE reproduction.
  Old TEST data are reused; the focused bank was excluded from fitting and
  selection. Hidden-response fidelity describes the model, not true semantics
  or forecast accuracy. The descriptive expansion gate passed; a separate
  bounded native preservation check is complete, with no lambda search.
  **Native preservation:** on20 baseline receivers, distortion is lambda0
  **0.027193170003341927**, lambda1 **0.02480812127788782**, PCA
  **0.011367679684409307**. On15 paired receivers, source-mean sinusoid
  NMSE is **0.14779665980988985/0.19236556912378608/0.14294983813373244**;
  noise NMSE **0.7082152774328037/0.8738374357450586/0.6469003467085652**.
  Lambda1 improves12/15 sine and11/15 noise receivers, but adverse tails
  reverse the means. Noise ratio-of-total-energies improves
  **0.536958287827822→0.27442507632856705**; it does not replace the frozen
  source-average primary. No denominator falls below its recorded floor.
  Baseline error uses history variance, while paired error uses native output
  response energy; neither measures forecast accuracy against future truth.
  All identity/restoration gates, nine independent exact forecast replays and
  792 arithmetic checks pass.399 cells take33.8405seconds,1.110GB peak.
  **Saved-array tail audit:** hidden-response NMSE improves for all15 actual
  singleton sources in each direction, including adverse output cases. Two
  low-native-response synthetic sources account for96.62% of positive sine
  deterioration; two real sources account for99.17% of positive noise
  deterioration. Noise mean error/history variance rises
  **0.007429496410932423→0.014622474951728765**, demonstrating genuine
  adverse output-error increases, not solely small denominators. Sine mean
  error/history variance instead falls **0.00420397959319133→
  0.0021108072573565806**. Net source-mean worsening is86.61%/94.09%
  orthogonal to the native sine/noise response, respectively. Raw pooled
  energies depend on source units; history-normalized noise pooling worsens
  **0.5133076997963011→1.0102742592023026**. Top1/top2 influence removal
  reverses means but is descriptive, not an exclusion rule or replacement
  primary. Four tests/600 independent FP64 identities pass;0.463seconds,
  no native calls or fitting. No downstream Jacobian was measured.
  Native and objective branches closed.
- **Reproduce.** `runs/dictionary_response4965000/knowledge.json`, paired arm
  records, comparison and arithmetic audit; `dictionary_response_native4965000`
  contains the independent native stage and per-source outputs.
  `dictionary_response_influence4965000` preserves all sources, domain
  summaries, energy identities and the influence analysis.
- **Ref.** ROADMAP §49.40; follows PM-50/51.
- **Score. 4/5.** Demonstrates an objective-dependent reconstruction/response tradeoff.

#### PM-53 · A longer forecast window strengthens a frequency-distortion description, even for strong tones
- **Claim.** For Bolt period19.2 inputs, OLD-source-selected output-frequency
  maps improve on same-frequency maps at both amplitudes and horizons. At64
  steps, period22 is selected with or without matched DC/trend nuisance terms;
  all period24 controls retain24. Tone-only64-step NMSE changes are
  **0.6671436361279975→0.4718520703236281** at amplitude0.1 and
  **0.8618611110237165→0.5910433543110412** at amplitude2. Family4-adjusted
  gain intervals are **[0.14272620350980894,0.23566626897020315]** and
  **[0.2506674391046784,0.2928226919160863]**.
- **Scope.**16 OLD FIT/16 reused TEST backgrounds,544 fixed inputs, native
  normalization and a finite nine-period grid. The32-step selection depends
  on amplitude and nuisance terms. Strong amplitude also changes normalization;
  this does not uniquely identify an SNR, training-prior, oscillator or patch
  mechanism. The scores predict model responses, not forecasting truth.
- **Recovery.** The original historical-replay gate compared changed batch
  membership/row geometry and failed before scientific evaluation; its missing
  discrepancy values remain a limitation. Four-case preflight replays original
  geometry exactly and reproduces a changed-geometry effect. An outcome-blind
  amendment preserves strict current64/current32 same-batch prefix equality,
  single-forward gates and measured historical drift bounds. No autoregression.
  Independent1210 score/replay and1544 roundoff checks pass; maximum audited
  rebatching NMSE change **1.9054664079920514e-06**, below its bound
  **8.08821007460502e-06**. Native recovery takes42.844seconds,1.048GB peak.
- **Reproduce.** `runs/response_phase_horizon_recovery4954700/knowledge.json`,
  original failed parent, `response_phase_horizon_replay4954700`, amendment,
  exact query records, frozen fits, raw outputs and independent audits.
- **Ref.** ROADMAP §49.40; follows PM-45/48. Native branch closed.
- **Score. 3/5.** Extends the waveform observation beyond a weak, short-window setting.

#### CA-15 · Adaptive native-head steering works without SAE features on fresh contexts
- **Claim.** With frozen radius0.2,ridge1e-5,six correction steps,dose0.2,
  interval widening,narrowing and seasonal-prior tasks pass full-output relative
  error<0.1 on64/64 fresh contexts per task for both models, and64/64 on each
  of those tasks in a second off-grid-period distribution. Fresh mean errors
  (widen,narrow,season) are Bolt **0.01906498080258097,
  0.011064170357698487,0.012995774410960347** and TimesFM
  **0.0015741310043839983,0.0023598210753626066,0.0007871052444355581**.
  Isolated upper-quantile control is harder:57/64 and63/64 fresh,45/64 and60/64
  off-grid. All2,048 one/six-step replication interventions independently replay
  through full models; normalized discrepancies≤2.384185791015625e-06.
- **Models / layers.** Chronos-Bolt-small native decoder-head input;
  TimesFM2.5-200M final-token point-head input; context128,horizon32,nine quantiles.
- **Evidence.** Causal within-model (L3), fresh-seed replication with discovery-
  selected settings frozen before evaluation. Additional twelve-source real
  checks support external validity but use reused sources with unknown
  pretraining overlap. Not sealed private confirmation.
- **Status.** Replicated within the stated adaptive-control tasks. No claim of
  universal semantic vectors, monosemantic neurons, forecast improvement or
  calibrated uncertainty. Quantile ordering is a separate reported diagnostic.
- **Reproduce.** `tsfm_lens/tsfm_lens/experiments/steering_control.py`;
  `runs/steering_e06_*_fresh64`, `runs/steering_e07_*_offgrid64`;
  portable arrays/audits/figures in `examples/steering/`;
  API `tsfm_lens.steering.ForecastSteerer`.
- **Ref.** ROADMAP §50.1–50.2; `docs/steering_research_log.md` E00–E10.
- **Score. 4/5.** Establishes a practical control baseline that disentangles
  output controllability from the stronger claim of semantic feature discovery.

#### MN-47 · Lower SAE reconstruction error does not consistently improve selective steering
- **Claim.** In a512-atom,k16,600-step,24-context Bolt pilot, held-out
  reconstruction NMSE is TopK **0.05970584973692894**, AbsTopK
  **0.05588775873184204**, BatchTopK **0.0515652634203434**. Yet64-atom
  seasonal-control errors at dose0.2 are **0.06595,0.06934,0.06858** respectively
  (rounded means; exact rows in artifacts). BatchTopK test L0 **21.25** versus
  **16** for the other arms additionally prevents a matched-test-sparsity claim.
  Single atoms fail the0.1 full-output criterion on every tested task/context.
- **Models / layers.** Chronos-Bolt-small native decoder-head input.
- **Evidence.** Descriptive dictionary fidelity and causal within-model steering;
  matched optimizer/init/training budget; one training seed, fresh test contexts.
- **Status.** Exploratory negative for adopting these variants on this evidence;
  not a general rejection of signed/adaptive SAEs. Modern variants remain opt-in.
- **Reproduce.** `sae/modern.py`; `runs/steering_e02_topk`,
  `runs/steering_e02_abstopk`, `runs/steering_e02_batchtopk`; portable summaries
  in corresponding `examples/steering/` directories.
- **Ref.** ROADMAP §50.2; `docs/steering_research_log.md` E02.
- **Score. 3/5.** A concrete TSFM example of why reconstruction and actual
  selective control need separate evaluation and realized-sparsity checks.


#### CA-16 · Bolt encoder-output control supports width and season, but isolated upper-quantile control fails the tested budget
- **Claim.** On16 fresh contexts, radius0.2 of the full encoder-state Frobenius
  norm,ridge1e-5,six nonlinear correction steps, upper-only control passes0/16
  for REG-only, shared-token and fully independent token edits. Mean errors are
  **0.2570767497950713,0.242367760175313,0.27089506942840846**. Widening passes
  **15/16,14/16,13/16**; season passes **15/16,16/16,16/16**. Original-Jacobian
  optimal upper-error means are **0.4195139834384104,0.31967113879732045,
  0.21209865207669704**; these bound only the initial linearized problem.
- **Models / layers.** Chronos-Bolt-small complete final encoder output,
  including REG; actual frozen decoder and head downstream;128 context,32 horizon.
- **Evidence.** Causal within-model (L3), fresh synthetic replication of a
  four-context discovery; all scopes/settings frozen and no exclusions.
- **Status.** Replicated restricted negative for upper-only at this site,
  norm budget and optimizer, with positive width/season control. Other budgets,
  sites and optimizers remain untested; head/encoder norm units are not directly
  comparable. No global impossibility claim.
- **Reproduce.** `experiments/steering_upstream.py`;
  `runs/steering_e09_upstream_fresh16`, portable raw results/arrays in
  `examples/steering/e09_upstream_fresh16`.
- **Ref.** ROADMAP §50.3–50.4; `docs/steering_research_log.md` E05,E05b,E09.
- **Score. 4/5.** Shows that steering feasibility depends on the allowed
  intervention site and geometry; late-head success alone is insufficient.

#### CA-17 · Automatic steering discovery generalizes across registered forecast architectures, with target-dependent limits
- **Claim.** A shared no-grad steering engine automatically discovers real
  internal/input interventions on Bolt, TimesFM and point-only Timer. Larger
  fresh-context suites achieve **141/168**, **156/168**, and **103/112** learned
  successes respectively (≤10% full-output error, radius0.2); explicit context
  preprocessing is excluded. Frozen synthetic profiles pass **60/72** Bolt and
  **68/72** TimesFM real-source cases with local response refinement. All native
  replays are exact. Large clipping and some selective edits fail.
- **Models / surfaces.** Automatically observed parameter-module inputs; no
  model/layer names in the engine. H64 for Bolt/TimesFM and H96 for Timer.
  A small Chronos2 pilot succeeds on only1/4 learned requests: registration
  portability is broader than reliable steering capability.
- **Evidence.** Eight new synthetic contexts per larger model suite, separate
  calibration/validation;12 reused real evaluation sources per transfer audit.
  D11 additionally controls Bolt's repeated 1→9 expanded-batch H96 rollout on
  two fresh contexts (8/8 learned cases). No future labels or weight training.
- **Mechanism.** Signed-effect screening avoids inactive branches. For a narrow
  standard Linear module, its exact input row space removes invisible directions:
  paired Timer successes rise16/28→26/28, then103/112 on fresh contexts.
- **Status.** Useful adaptive control, not universal support, semantic-feature
  discovery, a fixed vector, forecasting improvement or calibrated uncertainty.
  Real-source pretraining overlap is unknown; most successes occur near readouts.
- **Reproduce.** `steering/dynamic.py`; `experiments/dynamic_steering_study.py`;
  `experiments/dynamic_real_audit.py`; `examples/dynamic_steering/` portable raw
  arrays, protocols, findings and scientific figures.
- **Ref.** ROADMAP §51; `docs/dynamic_steering_research_log.md` D07/D08/D11/D14–D17b.
- **Score. 4/5.** A working architecture-independent intervention API with
  explicit negative results, replay audits and saved-profile transfer.

#### CA-18 · Exact context removal is operationally possible; selective-horizon context focus remains budget-limited
- **Claim.** Explicit ignore/focus preprocessing makes forecasts exactly
  invariant to removed values in **64/64** Bolt/TimesFM sensitivity cases.
  Amplifying deviations increases output sensitivity in **27/32**, so nonlinear
  emphasis is not universally monotonic. Reproducing context counterfactuals
  only in one forecast half while protecting the other is harder: expanding
  Bolt control from rank128 to512 improves successes **5/28→12/28**, while
  strong suppression/focus still fails the radius0.2 budget.
- **Evidence.** Eight fresh contexts/model for input-policy audits; four matched
  contexts across seven selective-horizon requests for the span comparison.
  All forecasts are real model inference, with exact intervention replay.
- **Status.** Operational input relevance semantics, explicitly distinct from
  attention control or learned internal steering. The partial-horizon negative
  is restricted to tested sites/budgets; no global impossibility follows.
- **Reproduce.** `experiments/dynamic_context_audit.py`; D09/D10/D18/D19 raw
  artifacts in `examples/dynamic_steering/` and local `runs/dynamic_*`.
- **Ref.** ROADMAP §51; `docs/dynamic_steering.md` context-policy contract.
- **Score. 4/5.** Separates a useful exact preprocessing capability from the
  stronger internal/selective causal claim it cannot automatically establish.
