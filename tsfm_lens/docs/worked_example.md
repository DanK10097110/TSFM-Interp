# Reading a report, end to end — one worked example

`ROADMAP.md` §21 J3's second half. The glossary (README, and collapsed under
the report's own preamble) defines the vocabulary; the per-stage docs
(`tsfm_lens/stage_docs.py`) say what each stage can and cannot tell you. Neither
teaches the skill this page teaches: **how to actually read a finished report
and end up with the right conclusion**, including which numbers to ignore.

It is written against one real run — `runs/medium_run_chronos_base`,
`google/timesfm-2.5-200m-pytorch` vs. `amazon/chronos-t5-base`, on a
288-series public-dev corpus — and quotes that run's actual artifacts, not
illustrative values. Every number below can be checked against the JSON files
named beside it. Where a number has since been superseded, that is said in
place rather than quietly updated; a worked example that silently tracks the
latest run stops being a worked example.

Read it with the report open. It follows the report's own section order.

---

## 0. Before any result: the fairness card

**Artifact:** `fairness/card.json` · **Report section:** "The fairness card"

This section renders first and unconditionally, before a single quality
number. It exists because every cross-model comparison in the rest of the
report is confounded by something, and the honest move is to state the size of
each confound up front rather than footnote it later.

| Axis | TimesFM | Chronos-T5-Base | Asymmetry |
|---|---|---|---|
| Parameters | 231.3M | 201.4M | 1.15× |
| FLOPs per forward (per series) | 7.42 GFLOPs | 96.84 GFLOPs | **13.04×** |
| Captured FLOP fraction | 42.5% | **14.4%** | 28.2 pt gap |
| Depth axis (`block`) | caps at 0.947 | caps at **0.478** | 50.5% overlap |
| Forecast determinism | deterministic | **±0.160 MASE** | one sampled, one not |

**What to take from it, in order of how much it will change your reading:**

1. **±0.160 MASE.** This is Chronos-T5-Base's *noise floor* — how much its MASE
   moves between two identical repeat runs, because it decodes by sampling.
   Any Chronos delta smaller than roughly twice that is not a result. Keep this
   number in your head; §1 below is where it immediately bites.
2. **14.4%.** The captured layers perform 14.4% of what a full Chronos forecast
   costs — its decoder runs 20 times per forecast and is never captured. Every
   depth-located statement about Chronos in this report is a statement about
   one-seventh of its computation. The report appends that qualifier
   automatically to such findings, but knowing *why* changes how much weight
   you give them.
3. **13.04×.** Chronos spends thirteen times the compute per forward pass.
   This does not weaken the behavioral result in §1 — it strengthens it in one
   direction and should make you suspicious in the other. If TimesFM wins on
   quality, it wins while spending a thirteenth of the compute. If Chronos had
   won, "it also cost 13× more" would be the first thing to say.
4. **50.5% overlap.** The two models' depth axes barely half-overlap, because
   Chronos's captured surface is its encoder only. Curves plotted against
   relative depth can be compared *in shape* over the overlapping range; a
   claim about *where* something happens in one model versus the other is
   carrying an offset nothing here measures away.

Three rows read "not yet measured" (finest resolvable lag, declared training
exposure, capability intersection). That is deliberate: an unmeasured
asymmetry is rendered as unmeasured rather than omitted, so its absence is
visible instead of invisible.

---

## 0.5. The corpus card — trust the benchmark before trusting any model result

**Artifact:** `corpus/card.json` (`ROADMAP.md` §34 items B1–C2) · **Report section:** "The corpus card" · **Evidence class:** descriptive

This section renders second, still before any model result, and for a
reason worth sitting with: **every number in every section below this one is
a claim about `benchmark_medium/public_dev`, and this is the one section that
asks whether that corpus itself deserves the trust the rest of the report
spends on it.**

`benchmark_medium/public_dev` is a sealed corpus (`epoch 0`, global digest
`6337c592a7bc…`) of **288 series across 3 families** — `random_parametric`
(188), `mixture` (60), `parametric` (40) — split by tier into **228
`synthetic`** series (no real data touched at all) and **60
`realism_stress`** series (real-derived, so they inherit a real corpus's
*distribution* even though every instance still had to pass the leakage
gate — the tier distinction §4.1 and the glossary's "leakage" entry both
draw). The largest single family, `random_parametric`, is 65.3% of the
corpus (`quality.largest_family_share`) — worth knowing before reading any
per-family statistic elsewhere in this doc as if every family carried equal
weight.

**The corpus trust ladder renders seven rows for this corpus, and the
honest reading is that none of them currently say "clears":**

| Claim | Verdict | Why |
|---|---|---|
| No series is a copy of a real reference series | `not_recorded` | manifest predates item B1's audit-persistence fix |
| No within-split near-duplicates | `not_recorded` | same reason |
| Dev and private splits share no series | `not_recorded` | same reason |
| Private split looks like the dev split | `inconclusive` | no equivalence test has been run (item B3) |
| A model can't win here by memorizing one shape (effective dimensionality) | `not_run` | no `benchmark_validation` report configured for this corpus |
| Series vary in more than one way | `not_run` | same reason |
| Models weren't trained on this data | `not_verifiable` | no mechanism anywhere can inspect a checkpoint's own training data |

This is not a defect in the section — it is the section doing exactly its
job. `benchmark_medium/public_dev` was built before item B1 existed, so its
manifest's `extra` block is empty; the correct rendering of "an audit that
never ran" is `not_recorded`, never a false `clean`, which is the same
`CLAUDE.md` §11.37 discipline ("absent and degenerate are different states,
and a threshold that cannot tell them apart reports the dangerous one as
the safe one") every other section in this report is built around. **Read
this the way you would read a `not yet measured` fairness row two sections
up: the honest answer to "is this benchmark corpus leak-free and diverse"
is currently *we have not checked*, not *yes*.** Nothing below this line
stops being useful for that — L0's MASE numbers, L1's CKA, L3's causal
fingerprints are all real measurements of what these two models do on this
corpus — but "this corpus" is exactly the object this section just told you
has an open trust question, and a reader who skips this section would never
learn that.

The representative-series gallery (one context window per family, sampled
with `np.random.default_rng` rather than a head slice — §11.38's fix, so a
corpus written grouped by generator doesn't silently show only its
first-sorting family) and the diversity figure both degrade the same
honest way: the gallery always renders (it needs no audit), while the
diversity figure reads `"not available (no corpus.validation_report
configured)"` rather than a redundancy fraction or an effective
dimensionality, because `benchmark_validation` was never run against this
particular corpus. A UMAP embedding is never computed here either way
(invariant 4 — diversity is measured in catch22 feature space, never on
UMAP coordinates, and this section will say so even when a validation
report *is* configured).

---

## 1. Behavioral profile — the only section that needs no interpretability at all

**Artifacts:** `l0/summary.json`, `l0/metrics.parquet` · **Evidence class:** behavioral

Overall MASE: **TimesFM 1.953, Chronos-T5-Base 2.251**. Lower is better; 1.0
means "no better than a naive same-scale baseline," so both models are
*worse than naive* on this corpus in aggregate. That is a property of the
benchmark, not a scandal — the synthetic corpus contains regimes (pure noise,
intermittent bursts) where a naive forecast is genuinely hard to beat.

Per family is where the actual content is:

| Family | TimesFM MASE | Chronos MASE | Gap |
|---|---|---|---|
| `mixture` | 1.172 | 1.182 | **0.010** |
| `parametric` | 2.961 | 2.743 | 0.218, favors Chronos |
| `random_parametric` | 2.000 | 2.500 | 0.500, favors TimesFM |

**The finding I would not act on is the first row.** A 0.010 MASE gap on
`mixture` is sixteen times *smaller* than Chronos's own ±0.160 repeat-run
noise floor from the fairness card. Run this pipeline again with a different
sampling seed and that gap could easily reverse. It is not a small effect; it
is *no measurement at all*, and the only reason to notice it is to say so
explicitly. Two models tying is a legitimate result — "TimesFM is very
slightly better at mixtures" is not.

**The finding I would act on is the third row.** 0.500 MASE is roughly 3× the
noise floor, and it is the one that survives everything downstream (§7).

Note also `mase_reliability`: 16 of 576 model-series pairs were **excluded**
from MASE because their scale denominator was too small to make the ratio
meaningful. They are excluded and counted, not silently kept — a near-constant
series can otherwise produce a MASE in the hundreds and dominate a family
average by itself.

---

## 2. Cost and capacity — the question a practitioner asks first

**Artifact:** `budget/model_budget.json`

The fairness card's numbers come from here, with the detail behind them:
TimesFM's 231.3M parameters split 196.7M body / 32.8M head / 1.8M front-end
across 20 identical 9.84M-parameter blocks. FLOPs are **measured** with
torch's `FlopCounterMode` during a real forward pass, not derived from a
parameter count — which is why the number works for architectures nobody has
written a formula for.

The panel that matters is L0 MASE re-plotted against measured FLOPs. Read
§1's result on that axis and it changes register: TimesFM is not merely better
overall on this corpus, it is better at 7.42 GFLOPs against 96.84.

One caveat stated in the section itself: `role_split_is_heuristic: true`. The
body/head/front-end split is a regex over module names, and it is honest about
being one. The *total* is exact; the split is a reading aid.

---

## 3. Model internals and the forecast lens — descriptive, then depth-resolved

**Artifacts:** `internals/profile.json`, `lens/lens.json` · **Evidence class:** descriptive

Internals is per-model and makes no cross-model claim: effective
dimensionality per layer, family-probe decodability per layer, CKA-to-input
per layer. Read it as each model's own profile.

The lens is the one place the report answers "where does the forecast actually
form." TimesFM's final MASE through the skip lens is **1.790** — close to but
not identical to §1's 1.953, because the lens runs on a smaller sampled batch;
that difference is a sample-size artifact and not a finding. The curve rises
from 2.976 at layer 0 (95% CI [2.118, 3.965]) toward the final value.

**Read the CIs here, not the means.** Layer 0's [2.118, 3.965] and layer 10's
[1.595, 2.993] overlap almost entirely. The *shape* of the curve — monotone
improvement, most of it early — is the finding. "The forecast crystallizes at
layer *k* specifically" is not supported by these intervals, and the
crystallization depth should be read as "somewhere in this region," which is
also how the section states it.

---

## 4. Representational geometry (L1) — the first cross-model number, and the weakest

**Artifact:** `l1/meta.json` · **Evidence class:** geometric (correlational)

Peak linear CKA: **0.381, 95% CI [0.363, 0.408]**, at TimesFM `stacked_xf.4` ↔
Chronos `encoder.block.10`. The shuffled-series null is **0.036 [0.035,
0.037]**.

Four things to read here, in order:

1. **0.381 vs. a 0.036 null is decisive** — the two models genuinely organize
   this benchmark alike at some pair of layers. The CIs do not come close to
   touching.
2. **It is correlational, and one specific confound is not removed by that
   null.** Both models read the same input. A shuffled-series null rules out
   coincidence; it does not rule out "any two networks of this shape agree
   this much before either is trained." The repo has a separate
   *untrained-twin floor* for exactly that, and against it this run's peak CKA
   **beats Chronos's own floor (0.174) and trails TimesFM's own (0.599)** —
   two decisive verdicts pointing opposite ways, depending on which floor you
   check. That is the honest state of this number, and it is why L2 exists.
3. **The per-family CKAs disagree with each other**: `parametric` 0.710,
   `random_parametric` 0.610, `mixture` 0.478. The global 0.381 is not "the"
   similarity — it is a peak over a heterogeneous set, and the family
   breakdown is more informative than the headline.
4. **Do not read the peak's location as a cross-model claim.** It sits at
   relative depth 0.211 for TimesFM and 0.435 for Chronos. Under the fairness
   card's 50.5% axis overlap, "TimesFM's early layers match Chronos's late
   encoder blocks" is a statement about two axes that are not calibrated
   against each other.

---

## 5. Stitching probes (L2) — the same question, one rung stronger

**Artifact:** `l2/stitching.json` · **Evidence class:** linearly-translatable

Best gain, Chronos→TimesFM: **0.413 [0.360, 0.461]** (raw R² 0.499).
Best gain, TimesFM→Chronos: **0.318 [0.293, 0.342]** (raw R² 0.545).

**The gain is the number; the raw R² is not.** Note that the direction with
the *higher* raw R² (0.545) has the *lower* gain (0.318) — a plain
input-feature probe already explains most of that R², so most of it is "both
models read the same series," not shared learned structure. Anyone quoting
0.545 as evidence of cross-model similarity would be quoting mostly the
benchmark.

**The two directions are separate measurements, not one number reported
twice.** Stitching is not symmetric: predicting TimesFM's layer 18 from
Chronos's block 10 is an easier problem than the reverse, and both are
reported because the asymmetry is itself informative.

The same untrained-twin caution from §4 applies, with a sharper history: an
earlier session compared this run's peak gain against the null run's *global*
peak — which sits at block 0, where an untrained network trivially predicts
its own untrained twin because both are still close to raw input — and
concluded, wrongly, that the gain was indistinguishable from null. Compared at
the *matching layer index*, real gain exceeds the floor by +0.351 (p=0.002).
**The lesson generalizes: when comparing against a null run, match the layer,
not the peak.**

---

## 6. Perturbation and patching (L3) — where the overall number is the misleading one

**Artifact:** `l3/meta.json` · **Evidence class:** causal within model

Overall cross-model fingerprint agreement: **0.356 [0.330, 0.385]**. Read
alone, that says "moderate agreement." Per corruption, it says something else
entirely:

| Corruption | Agreement | Reading |
|---|---|---|
| `level_shift` | **+0.964** | near-identical depth profiles |
| `deseasonalize` | +0.893 | strong agreement |
| `detrend` | +0.813 [0.344, 0.909] | agreement, but a very wide CI |
| `dropout` | +0.603 [0.396, 0.730] | agreement, wide CI |
| `frequency_shift` | −0.178 | essentially unrelated |
| `warp` | −0.428 | opposed |
| `noise` | −0.875 | **opposed** |
| `smooth` | −0.875 | opposed |
| `spike` | −0.953 | **opposed** |

The two models carry *structural* corruptions (level shifts, seasonality,
trend) at similar depths and carry *local, high-frequency* ones at opposite
depths. Averaging a set that ranges from +0.96 to −0.95 produces 0.356, a
number that describes none of the rows. **Quote the breakdown; the overall
number is arithmetic, not a finding.**

Two more things this section will tempt you into:

- **The sensitivity bars are not calibrated across corruptions.**
  `level_shift` is a permanent multi-σ step over 40% of the series;
  `spike` touches ~0.6% of timesteps. A taller bar for `level_shift` says
  the perturbation is larger, not that the model is more sensitive. Compare
  each corruption's bar *across models*, never one corruption's bar against
  another's.
- **`detrend`'s CI is [0.344, 0.909].** Its point estimate belongs in the
  "agrees" group; its interval spans from "barely" to "strongly." It is one
  more run away from moving.

Patching (`l3/patching.npz`) is the causal half, and its claims are *within
one model*. What gets compared across models is each model's own
restoration-by-depth curve — never an activation transplanted between them,
which would be meaningless without a learned mapping.

---

## 7. Private benchmark confirmation — the only section that is evidence

**Artifact:** `confirm/confirmation.json` · **Evidence class:** confirmatory

Everything above is exploratory. Many comparisons were looked at; some were
significant; that is what looking at many comparisons does. This section tests
pre-registered hypotheses **exactly once** against a sealed private corpus
drawn from a disjoint seed range.

**One hypothesis was registered, and it confirmed.** From §1's third row —
TimesFM favored on `random_parametric` (dev ratio 0.777):

- 286 private series, 186 in the tested family
- mean per-series MASE difference **0.455, 95% CI [0.234, 0.675]**
- p = 0.002, Holm-corrected p = 0.002, **confirmed: true**
- overall (all families): 0.345 [0.196, 0.510], positive favors TimesFM

The CKA replication check is the other half: dev 0.381, private **0.374
[0.355, 0.408]**, `replicates: true`. The geometric result is not an artifact
of the dev corpus.

**How to state the conclusion of this whole report, honestly:**

> On this benchmark, TimesFM forecasts `random_parametric` series better than
> Chronos-T5-Base by ~0.46 MASE — pre-registered and confirmed on held-out
> private data — while using roughly a thirteenth of the per-forward compute.
> The two models share real representational geometry (CKA 0.381 vs. a 0.036
> null, replicating at 0.374 on private data), most strongly on `parametric`
> series, and carry structural corruptions at similar depths while carrying
> local high-frequency ones at opposite depths. Chronos's depth-located
> results describe ~14% of its actual computation. The `mixture` family shows
> no difference between the models.

Every clause there is either confirmatory, or a caveat, or an exploratory
number quoted with its null beside it. Nothing in it depends on a delta below
a noise floor, on an overall number that averages over sign changes, or on a
raw R² where a gain was the honest quantity.

---

## 7.5. Sparse features and causal roles (SAE) — not on this run, and why that matters

**Artifact:** `sae/meta.json`. (This run has no role/channel artifacts — see below.)

This run trained a baseline `TopKSAE` on both models (`sae/meta.json` above),
but predates the causal channel battery, named "roles," and cross-model role
matching this section describes today (`ROADMAP.md` §25's Component A/B/C
work) — so its own report renders no roles table, no role×model matrix, and
no "CKA in SAE feature space" subsection, and this page cannot quote numbers
for them from this run without fabricating data that was never computed here.
That is itself the lesson worth stating explicitly: **an artifact's absence
on a specific run is not evidence the pipeline lacks the capability** — check
`stage_docs.py`'s `sae` entry (or this repo's current `ROADMAP.md` §25.24/
§25.25) for what the stage can do *today*, and check the run directory's own
files for what that *particular* run actually has, before concluding either
way. The real numbers for the causal-channel and role-matching work — a
reach of 0.708, 23 of 39 candidates clearing at least one channel, and a
cross-model match rate of 0.833 that does *not* clear its own untrained-twin
floor of 1.0 — are quoted from `runs/full_report_run_large`, a different,
newer run, in `ROADMAP.md` §25.23 and §25.25; they are not reproduced here
because reproducing them against this page's own run would require re-running
that run's SAE stage with the new machinery, which has not been done.

---

## What this example is not

- **Not a claim about TimesFM and Chronos in general.** One corpus, one
  checkpoint pair, one context length, one horizon. A confirmed finding is
  confirmed *on this benchmark*.
- **Not current forever.** Numbers here are from one recorded run;
  `ROADMAP.md` §5's Findings block is the live research record. If a number
  here disagrees with one there, that one is newer.
- **Not a substitute for the per-figure notes.** Every chart in the report
  carries its own collapsed "what is this chart" block with purpose, how to
  read a value, and where it misleads. This page teaches the reading posture;
  those blocks carry the specifics.
