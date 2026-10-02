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
| SH-22 | At 7 models under the matched null, the atlas is **segregated by model** (cross-model mixing below chance, p 0.025). The full same-effect verdict (7/1290) is one Chronos-2 ↔ Chronos-Bolt concept, but 1053 tests are underpowered and partial agreement crosses families (level only 35, shape only 9); the strict rung's sensitivity is unmeasured | 5 |
| SH-18 | Held-out: 19/20 searched and **20/20 frozen-feature** concept-transfer claims confirm on fresh sealed private epochs; **20/20 again at 7 models on a fourth private draw** | 5 |
| CA-11 | Held-out (K2, 7 models): 10/31 single-feature causal claims confirm; **0** cross-model causal claims do (atlas 0/3 multi-model, L5 0/6); models share what they select, not demonstrably what they do | 5 |
| SH-19 | Unit-level correspondence (CCM) NO-GO: low-rank TSFM representations defeat the rotation control | 5 |
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
- **Ref.** ROADMAP §9 (~7500–7650), §16 E3(d) (~11160), §34 D2 (~33030).
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
- **Caveat on the artifact's text.** 19 non-confirmed claims carry the reading "untestable", which is wrong. The MDE was computed from each claim's own 200 draws, but the adaptive procedure can reach the Holm floor, so these claims were tested and not detected. The verdicts are unaffected, and the reading is now recomputed at render time.
- **Models / layers.** TimesFM, Chronos-2, Sundial, Chronos-Bolt, Timer, Time-MoE, Chronos-T5-Base (encoder); profile-matched null (MN-29).
- **Evidence.** **Held-out confirmed** (causal within-model, for the 10 single-feature claims). Held-out *not* confirmed for every cross-model causal claim. **Status:** held-out confirmed / negative on private.
- **Reproduce.** `runs/panel7_v2_dev/confirm/confirmation.json` (`concept_replication`), private v2 epoch 3 (971 series), registry sha256 `ff6c4f9e…`, k3wt 597848a.
- **Ref.** ROADMAP §38.2 (Findings — K2 CONFIRMED ONCE).
- **Score. 5/5.** This is the gold-standard answer to "do the causal concept claims hold up". The individual within-model causal features partly do (about 1 in 3). The cross-model causal sharing claims do not, while input-level sharing replicates on a fourth private draw: models share what they select, not demonstrably what they do with it.

---

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
- **Reproduce.** `run_stage2_response_fingerprint.py`; `sae/response.py::feature_ablation_fingerprints`.
- **Ref.** ROADMAP §25.23 (~21263–21422), §27.3.
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
- **Evidence.** Held-out validation. **Status:** confirmed.
- **Reproduce.** `benchmark_validation/cross_split.py`; `run_validation.py --compare-splits`.
- **Ref.** ROADMAP §34 B2/B3 (~31618–31810), §37.10.
- **Score. 3/5.**

---

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

#### MN-07 · Causal fingerprints are nearly one-dimensional, so they cannot be a cross-model matching key
- **Claim.**
  - The 9-channel ablation fingerprint has participation-ratio dimension
    2.09. The 64-step delta curve has dimension 1.04, with PC1 at 98.3%.
  - The mutual-nearest-neighbour match rate (37.6%) sits inside its null
    p95 (50.0%).
  - The full curve does worse: 15.3% vs 13.1% null at cosine 0.99.
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

#### MN-15 · The SAE's own reconstruction error moves the forecast ~3.5× more than removing a feature
- **Claim.**
  - Median reconstruction cost is 0.139 context sd, against a median feature
    effect of 0.0395.
  - 23.0% of panels have a feature effect below 1/10 of the reconstruction
    cost.
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
- **Evidence.** Known-answer validation (causal within-model on a planted model; held-out edits behavioral). **Status:** negative (closed at R0): R0 and R0b both fail their pre-registered gates; continuing needs a new design decision.
- **Reproduce.** `python run_repair_known_answer.py --config configs/repair_known_answer.yaml` (commit 5a8077f); R0b: `--config configs/repair_known_answer_r0b.yaml --two-stage` (commit 6c94c66); `analysis/repair_blame.py`, `analysis/repair_edit.py`.
- **Ref.** ROADMAP §39.7–39.8.
- **Score. 3/5.** Blaming a feature for errors needs causal ablation, not correlation: the decoy correlates with error in 5/5 seeds and has no effect. Crediting a single feature for *good* forecasts is underpowered at realistic sizes, and that bounds the "what causes good predictions" half of the question.

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
