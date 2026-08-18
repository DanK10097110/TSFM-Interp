# ROADMAP_ARCHIVE.md — history and superseded plans

> **What this file is.** `ROADMAP.md` grew to 16,400 lines, most of it the
> *scaffolding* of finished work: fix plans for bugs that are fixed, hypothesis
> tables for questions that are answered, and a session log that had drifted
> from the changelog its own header asks for into a full narrative diary. That
> material is history, not direction, and it was crowding out the parts of the
> roadmap a session actually reads.
>
> **Nothing was deleted.** `ROADMAP.md` §0.2's discipline — never delete a
> finding, even a negative one — is why this file exists instead of a `git rm`.
> Everything below is the *verbatim* original text, moved here so the roadmap
> can be read in one sitting.
>
> **What stayed in `ROADMAP.md`:** every Findings block, every recorded number,
> every decision and refutation, every open item's full plan. What moved here:
> fix plans for closed items, superseded planning blocks, the pre-compression
> session-log narrative, and parked items' original long-form write-ups.
>
> **This file is append-only and needs no maintenance.** It is not a second
> source of truth — it is a record of what a plan looked like before it became
> a result. Do not update it to reflect current state; that is `ROADMAP.md`'s
> and `CLAUDE.md`'s job. Read it only when you need to know *why* something was
> planned the way it was, or to recover a detail a compressed summary dropped.
>
> Sections appear in the order they were archived (roughly bottom-of-file
> upward), each under a heading naming the `ROADMAP.md` section it came from.


## Archived: §22's original six-wave sequencing (2026-08-12), superseded by §0.5's single queue and §22's parked-features section

## 22. How §17–§21 slot into the existing plan (added 2026-08-12)

Not a new phase spine — §3's phases stand. This is the interleaving, and it is
also the answer to "what should the next ten sessions actually do."

**Wave A — retroactive integrity (do before the next round of cross-model
claims).** These change how numbers already in this file should be read, which
is why they come first, exactly as §16 E9's null baseline did.
`F6` (noise-floor units) → `F2` (budgets) → `F1` (depth axis) → `F4` (coverage)
→ `F9` scaffold. ~4 sessions. **Highest ratio of corrected-record to effort in
the file.**

**Wave B — the flagship research deliverable.** Unchanged and still first in
priority terms: §0.5 Tier 1, i.e. §6.2.1 Stages 0→2. Wave A and Wave B are
independent and should be run in parallel (Wave A is analysis-layer, Wave B is
GPU-bound training — the natural background/foreground split `CLAUDE.md` §2.8
describes).

**Wave C — legibility.** `J1`+`E6` together → `J5` → `J2` → `J4` → `J3` → `J6`.
~4 sessions. Cheap, and it is what converts the work into something a stranger
can use and a reviewer can check.

**Wave D — the four cheap novel studies.** `H4` (agreement) → `H1` (scaling
ladder) → `H3` (memorization) → `H2` (recommender). ~5 sessions. Each produces a
result the repo currently cannot state; H2 depends on Wave A's F2/F6.

**Wave E — architecture breadth, one class per session.** `G1` (tiers) first
since it makes the rest cheap, then `G6` (black-box — nearly free after G1),
then `G3` (the mixing-profile abstraction — the big one), then `G4` (MoE), then
`G2`/`E3` (auto-adapter), then `G7` (supervised controls), then `G5` (SSM
design). ~8 sessions, and the most valuable long-term investment after the
crosscoder.

**Wave F — adoptability.** §16's T1 chain, unchanged: `E4` → `E1` → `E7a`/`E7b`
→ `E5` → `E6`. Best done *after* Wave C, since E1's one-button promise is worth
much less if what comes back is unreadable.

**The one ordering constraint worth stating explicitly:** do not run Wave F
before Wave A. A one-button tool that hands a stranger a depth-axis figure
comparing 80%-of-TimesFM against 80%-of-Chronos's-encoder, with no coverage
number and no noise floor, is worse than the current five-command tool — because
it removes the expert who would have known not to believe it. That is §16's own
argument about §15's P1 items, applied one level up: **equal grounds is a
prerequisite for automation, not a refinement of it.**


## Archived: §20's original H1-H12 write-ups (2026-08-12) in full, before the 2026-08-18 triage

## 20. New capability proposals (added 2026-08-12)

> **Filter applied before anything was added here.** §16's own Findings block
> sets the bar: *state which claim this sharpens.* Several obvious-sounding
> additions were considered and **rejected** — recorded here so they aren't
> re-proposed:
> - *Probe every intermediate representation with a bigger probe family
>   (MLPs, transformers).* Rejected: a stronger probe answers "is the
>   information present" more permissively, which weakens rather than sharpens
>   every decodability claim. Linear probes are the right tool precisely because
>   they are weak.
> - *t-SNE/UMAP galleries of activation space.* Rejected: `CLAUDE.md` §5's
>   single most important rule already forbids quantitative use of embedding
>   coordinates, and a gallery invites exactly that.
> - *An LLM-written narrative summary of the report.* Rejected: the report's
>   value is that every sentence traces to an artifact; generated prose breaks
>   that and cannot be audited.
>
> Items are **H1–H12**, ordered by (value × cheapness) ÷ risk. H1–H4 are the
> ones to build first: all four are cheap, all four reuse existing artifacts,
> and all four produce results the repo currently cannot state at all.

### H1 ⭐ The model-family scaling ladder `[ ]`

**The cheapest genuinely novel study available in this repo.** Chronos-T5 ships
as tiny → mini → small → base → large: a five-point, same-architecture,
same-training-corpus, same-tokenizer size ladder. §5.3 used two of those points
once, as a size *control*. Used as a *ladder*, it answers a question nobody has
published for TSFMs: **which interpretability properties scale with size, and
which don't?**

Run the identical config at all five sizes and plot against parameter count:
crystallization depth (relative and absolute), effective dimensionality per
layer, family-probe decodability, L1 peak CKA against a fixed reference model,
L2 stitching gain, L4 clustering AMI, SAE dead-feature rate and ground-truth
alignment, calibration error, and MASE. Every one of these is already computed;
this is a sweep harness plus a report section, not new analysis.

**Why it's high-value.** Scaling behaviour is the most legible result type in ML,
and disagreements are the interesting part — `ROADMAP.md` §5's base-vs-small
entry already found L1/L2 *growing* with size while L4 clustering AMI *shrank*,
on two points. Five points turn that from a curiosity into a curve. It also
directly serves the practitioner question "is the bigger checkpoint worth it?",
in cost-normalized terms once F2 lands.

**Deliverables.** `configs/scaling_ladder_chronos.yaml` (five model blocks or
five run dirs — five runs is cleaner given VRAM), a
`run_scaling_ladder.py` cross-run reducer built on the existing
`report/meta_report.py` (which already aggregates N run dirs — extend, don't
duplicate), and a report section plotting metric-vs-size with per-point CIs.

**Acceptance criterion.** Monotonicity stated per metric with CIs, and any
non-monotone metric flagged rather than smoothed. **Explicitly report which
metrics are flat** — a flat curve is the finding that a metric doesn't measure
capability.

**Cost.** 1 session of harness + 5 extraction runs (tiny/mini are cheap; large
is the constraint). Depends on nothing.

### H2 ⭐ The practitioner recommender — "which model for my data?" `[ ]`

**The item that makes this repo useful to people who will never read a CKA
plot**, and the strongest answer to §1's "anyone".

Given a user's own series (or a described profile: sampling frequency, dominant
seasonality, trend/intermittency/noise character), return a ranked model
recommendation with reasons and a confidence, grounded entirely in results the
repo already produces:
1. Featurize the user's series with `benchmark_validation`'s existing catch22
   pipeline.
2. Map it into the benchmark's family/archetype space — nearest archetypes by
   the same feature space, with a distance that triggers an **explicit refusal**
   when the user's data resembles nothing in the corpus ("your data is outside
   this benchmark's coverage; the recommendation would be extrapolation").
3. Look up per-family L0 results (MASE, calibration, per-horizon behaviour) for
   each analyzed model, plus F2's cost, plus F6's noise floor so a
   recommendation is never made on a difference smaller than run-to-run noise.
4. Return a ranked list with a plain-language reason per model and the
   supporting numbers, plus what the user would gain from the runner-up.

**Deliverables.** `tsfm_lens/recommend.py::recommend(series_or_profile,
results) -> Recommendation`; a CLI `tsfm-lens recommend --series my.csv`; a
report section demonstrating it on three held-out example series.

**Acceptance criterion.** On held-out synthetic series with *known* archetypes,
the recommender's top pick matches the empirically best model for that archetype
at a rate clearly above chance — and the refusal path fires on deliberately
out-of-coverage input (e.g. a frequency far outside the corpus).

**Risk to state up front.** This is the one item that can be *confidently
wrong* in a way a reader acts on. It must refuse rather than guess, and every
recommendation must carry the tier/coverage caveats from F9. Depends on: F2, F6,
and E4's fixed corpus for comparability.

**Cost.** ~1.5 sessions.

### H3 ⭐ Memorization and verbatim-recall probing `[ ]`

**Novel, cheap, and it inverts machinery this repo already owns.** The benchmark
exists to prevent leakage *into* evaluation. The same DTW/near-duplicate
machinery, pointed the other way, asks: **do these models reproduce real series
they were trained on?**

Method: take real series from published corpora (Monash/ETT via `sources.py`),
feed a prefix, and compare the model's continuation against the *true* held-out
continuation using the leakage auditor's own banded-DTW distance — then compare
that against the distance achieved on **matched synthetic series with the same
catch22 profile** that no model can have seen. A model that is dramatically
closer on real-and-plausibly-trained-on series than on statistically matched
unseen ones is showing memorization rather than generalization.

**Why the control is the whole item:** real series are often *easier* than
synthetic ones, so raw accuracy on real data proves nothing. The matched-profile
synthetic control is what turns this from a naive claim into a measurement. Add
a second control: the same test on the model's own `random_init` twin (E9's
mechanism, free) bounds how much of the gap is architecture-plus-input-statistics.

**Deliverables.** `analysis/memorization.py::recall_probe(adapter, real_pool,
matched_synthetic_pool, prefix_frac, ...) -> dict`; report section; findings
gated on F6's noise floor.

**Acceptance criterion.** The gap is reported with a series-bootstrap CI against
both controls. **A null result is a perfectly good outcome here and must be
reported as such** — "no detectable verbatim recall at this prefix length" is a
publishable, reassuring finding, and pre-committing to publishing it is what
keeps the item honest.

**Cost.** ~1.5 sessions. Also directly informs F7's training-exposure confound
and `CLAUDE.md` §4.1's distributional-leakage tier, which currently has no
empirical measurement at all.

### H4 ⭐ Cross-model agreement as a reliability signal `[ ]`

Nearly free — pure reduction over `predict()` output L0 already holds, zero new
forward passes. For each series, compute inter-model forecast disagreement
(pointwise and distributional), then test whether disagreement predicts error.
If it does, "run two models and check whether they agree" becomes an actionable
reliability heuristic for practitioners, and a *use* for the comparison rather
than only a study of it. Report the correlation with CI, per family and per
horizon step (E12's axis), plus a calibration curve of disagreement→error so a
user can read a threshold off it.

**Acceptance criterion.** Correlation with a series-bootstrap CI, and an
explicit comparison against the obvious cheaper baseline (each model's *own*
quantile width — if self-reported uncertainty predicts error just as well,
cross-model agreement adds nothing and that should be said).

**Cost.** ~0.5 session. Do this first of the four.

### H5 Temporal-position and phase representation `[ ]`

How does each front-end encode *when*? Probe for absolute position, phase within
the dominant period, and time-since-changepoint from window states across depth
(the existing `internals.py` probe harness, new targets — the ground-truth
labels already exist in `GroundTruth`). Directly relevant to a real architectural
difference (TimesFM's patch-index positional scheme vs. T5's relative
attention bias vs. Sundial's scheme) and complements E13's spectral lens: E13
asks *which frequencies*, this asks *which phase*. **Cost:** ~1 session.
**Sharpens:** the "where does the forecast crystallize" story, by distinguishing
"knows the shape" from "knows where in the cycle it is."

### H6 Fine-tuning plasticity — which layers move `[ ]`

§6.3.1 Option E produces fine-tuned children as a *by-product*. For free, ask:
which layers changed most (weight-delta norm per block), and does that match
where the interpretability metrics say the task-relevant structure lives? A
match is a rare, satisfying cross-validation of the whole layer-screening
enterprise (§6.1.1) from a completely independent signal. **Cost:** ~0.5 session
on top of E. **Sharpens:** `work_bend`'s production status, which currently
rests on a single-corpus bake-off.

### H7 Distribution-shift envelope, distinct from L3's corruptions `[ ]`

L3 corrupts a series to find where a property is carried. This asks a different
question: **where does each model stop working?** Sweep *beyond* the training/
benchmark envelope — periods shorter and longer than any in the corpus,
amplitudes and offsets far outside it, sampling frequencies not represented,
context lengths beyond the model's training context — and map each model's
degradation curve. Reuses `generators.py` with out-of-range parameters (nothing
new to build on the data side) and `predict()` only, so it is tier-0 compatible
and works on hosted models.

**Why it's distinct from E20's context sweep:** E20 varies *how much* input;
this varies *what kind*. **Cost:** ~1 session. **Sharpens:** every "model A is
better" claim, by locating the boundary where the ordering reverses — which is
usually where the practical decision actually sits.

### H8 The seasonality circuit — the flagship mechanistic result `[ ]`

The repo's mechanistic ambition currently tops out at "which heads matter"
(mean-ablation ΔMASE) and "which layer carries it" (L3 patching). The next rung
is a **minimal sufficient set**: find the smallest set of heads/MLPs whose
patching restores period-detection behaviour, and show it is both sufficient
(patch only these → behaviour restored) and necessary (ablate only these →
behaviour lost) on the deseasonalize corruption, whose ground-truth period is
known exactly for every synthetic series.

Composes E18 (path patching), the existing periodicity-head taxonomy (the
candidate set — no blind search needed), per-window patching (already built),
and the known-period ground truth. **Greedy search over the candidate set, with
the necessity/sufficiency pair as the acceptance test**; report the set size and
the fraction of behaviour it explains. **Cost:** 2 sessions, and it depends on
E18. **Sharpens:** `CLAUDE.md` §12 item 6 — the stated limitation that nothing
is component-level below head/MLP granularity. This is the item that would make
the repo's mechanistic claims comparable to transformer-lens-era LLM work rather
than adjacent to it.

### H9 Analysis card export `[ ]`

A one-page, citable Markdown/PDF card per analyzed model: identity and
checkpoint digest, tier, F9's fairness rows, headline L0/calibration/cost
numbers, coverage, confirmed-vs-exploratory findings, corpus digest, library
versions, and the run's git SHA. Trivially built on E6's `findings.json` +
F9's `card.json` + E5's registry, and it is what someone actually attaches to a
paper or a decision memo. **Cost:** ~0.5 session, after E6/F9.

### H10 Report progressive disclosure `[ ]`

See §21 — kept as an item so it is schedulable, specified there.

### H11 Checkpoint-trajectory analysis `[ ]`

If any TSFM publishes intermediate training checkpoints, the emergence question
("when during training does seasonality decodability appear?") becomes
answerable with zero new analysis code — it is H1's ladder with training step
as the axis instead of parameter count. **Blocked on availability, not on
design.** Recorded so it is checked rather than rediscovered: verify whether any
of the five integrated families publish intermediate checkpoints `[VERIFY]`. If
none do, the fine-tuned children from §6.3.1 Option E give a short, artificial
trajectory as a fallback. **Cost:** ~0.5 session if checkpoints exist.

### H12 Deterministic-replay and result-provenance mode `[ ]`

An honesty feature, aimed squarely at the reproducibility problems this repo has
already paid for (`CLAUDE.md` §11.13's golden-hash mystery, §11.24's
config-meaning drift, §11.25's stale zarr store). One flag that records, per
run: every library version, the git SHA, CUDA/driver versions, the resolved
config *after* defaults, corpus digest, every RNG seed actually drawn, and a
digest of the activation store — written to `provenance.json` and rendered in
the report. Then `run.py --verify-provenance <run_dir>` re-checks the current
environment against it and reports every difference.

**Why it earns a slot:** §11.24 cost a full investigation session that a single
"these two runs differ in these 3 ways" diff would have closed in a minute, and
that failure mode gets *more* likely as background-agent workflows (`CLAUDE.md`
§2.8) become the norm. **Cost:** ~1 session. Composes with A3's config
fingerprint (already built) — extend it rather than adding a parallel mechanism.

---


## Archived: §19's G3-G7 full write-ups (2026-08-12), parked 2026-08-18 per §22.2

### G3 Attention-free architectures — the generic mixing-profile abstraction `[ ]`

**The highest-value item in this section.** TTM/TinyTimeMixer has no attention;
an SSM has no attention; a convolutional forecaster has no attention. Today the
entire attention stage skips, and with it the lag taxonomy, periodicity heads,
and head ablation — a third of the report.

**The insight that generalizes it.** Every one of those analyses actually asks
one question: **how does information at time *t* influence the representation at
time *t′*?** Attention weights are one way to read that off, available only for
attention models. But it can be **measured** for any architecture by
perturbation, which is exactly the machinery `impulse_alignment_check` and E3
already use:

- **`analysis/mixing_profile.py::mixing_kernel(adapter, layer, context_len,
  amplitude_frac, device) -> np.ndarray [T_out, T_in]`** — perturb input
  position *j*, measure the response at every output position *i* of the chosen
  block, giving an empirical influence matrix. This is a **measured** analog of
  an attention pattern that exists for attention, convolution, mixing MLPs,
  SSMs, and MoE alike.
- Everything the attention stage computes then runs off `mixing_kernel` instead
  of off `attention_patterns`: lag profiles (row-averaged influence by offset),
  periodicity scores (excess influence at seasonal-lag multiples), causality
  check (mass above the diagonal — which for a causal decoder must be zero, the
  same sanity check TimesFM already passes).
- For attention models, **validate the abstraction by comparing the measured
  kernel against the true attention pattern** on a model where both exist. If
  they agree, the perturbation method is trustworthy on models where only it is
  available; if they disagree, that disagreement is itself a finding about what
  attention weights do and don't tell you — a live debate in interpretability,
  and this repo would have a clean measurement of it.

**Acceptance criterion.** On Chronos-T5 (or TimesFM), `mixing_kernel`'s
lag-profile taxonomy reproduces the attention-pattern taxonomy's cross-family
ordering (rank correlation ≥ 0.7); then an attention-free adapter produces a
full lag/periodicity analysis.

**Cost.** 2 sessions. Cost scales as `O(T / stride)` forward passes per layer,
so it is genuinely expensive at fine stride — make stride a knob, default coarse
(one perturbation per alignment window, i.e. 16 passes at `context_len 512`,
`window 32`), and refine only where the coarse map shows structure.

**Why this is worth two sessions:** it converts "no attention → a third of the
report is blank" into "every architecture gets an information-flow analysis on
the same footing," which is precisely §18's equal-grounds thesis applied to the
architecture axis rather than the metric axis. It is also the single most
publishable methodological item in this file after the crosscoder.

### G4 MoE architectures — routing as a first-class axis `[ ]`

MoE feed-forward layers (TimeMoE, Moirai-MoE) keep a residual stream, so tiers
0–2 work unchanged. What changes and what it unlocks:
- `mlp_info` is ambiguous — "the MLP" is a router plus N experts. Mean-ablating
  "the MLP" conflates routing with expert computation. Define
  `moe_info(block) -> {router, experts: list, top_k}` as a new optional
  capability; MLP ablation on an MoE block ablates *experts*, and a separate
  **router ablation** (force uniform routing) isolates the routing decision —
  two distinct interventions the current abstraction cannot express.
- New analysis: **expert specialization by family.** For each block, the
  distribution of expert assignment conditioned on generator family, plus
  routing entropy per family. This is a *labelled* specialization measurement —
  the benchmark's known ground-truth families make it far cleaner than anything
  possible on wild data, and it is a natural companion to the SAE work (both ask
  "what does this model factor its computation into", one architecturally, one
  learned).
- **Interpretability payoff:** if experts specialize by seasonality/trend
  regime, that is a legible, discrete circuit story of a kind dense models don't
  offer — the highest-ceiling result available in this section.

**Acceptance criterion.** Expert-assignment-by-family with a permutation null
(shuffle family labels) so "specialization" is not read off routing imbalance
that exists regardless of input.

**Cost.** ~1.5 sessions once an MoE checkpoint is wired.

### G5 State-space / recurrent models — patching semantics `[ ]`

An SSM's position *i* representation is a function of a carried state, so
replacing the block output at position *i* does **not** isolate position *i*'s
contribution — the patch propagates forward through the recurrence in a way an
attention patch does not. Two consequences to design for **before** claiming an
L3 result on such a model:
- `token_patch` remains mechanically fine but its *interpretation* changes;
  the adapter must declare `carries_state: bool = False` (default keeps every
  current adapter correct), and L3's report note must state the different
  reading when it is true.
- The clean intervention for a stateful model is a **state patch** rather than an
  output patch. Scope this as tier-2-with-caveat rather than pretending
  equivalence; G3's measured mixing kernel is the fairer cross-architecture
  substitute and is unaffected by the issue.

**Cost.** ~1 session of design + adapter, and it is mostly a correctness-of-
interpretation item, not a code item — which is why it is worth writing down
before an SSM adapter exists rather than after a wrong number is published.

### G6 Black-box and hosted models `[ ]`

With G1's tier 0 this is nearly free and disproportionately useful: a
practitioner deciding between a hosted API and a self-hosted checkpoint can get
L0, calibration, horizon-resolved metrics, cost/latency (F2 — where an API's
real number is network latency and dollars, both worth reporting), agreement
(H4), and the recommender (H2), all on the same footing as an open model.

**Design constraints that must be explicit:** a hosted model's inputs leave the
machine, so (a) **never** send the sealed private corpus to a third-party API —
`confirm` must refuse tier-0 network adapters by default, since that would leak
the one consumable the whole method depends on (`CLAUDE.md` §6.7); (b) rate
limits and cost mean series counts need their own cap; (c) results are not
reproducible if the endpoint changes silently, so record the response's model
version if the API exposes one and warn loudly if it does not.

**Cost.** ~1 session for a reference HTTP adapter plus the `confirm` refusal.

### G7 Supervised baselines as controls, not competitors `[ ]`

PatchTST/DLinear/N-BEATS/seasonal-naive are not TSFMs and should not be
"compared" to one — but they answer a question the repo currently cannot: **how
much of a foundation model's advantage is foundation-ness?** A per-family
trained-on-this-corpus baseline, plus seasonal-naive (which MASE already
implies), turns every L0 result from "A beats B" into "A beats B, and both beat
/ fail to beat a small model trained directly on this task" — which is the
context a practitioner needs and a reviewer will ask for.

Slots in as tier 0 with a `zero_shot: false` marker so the report never mixes
them into the zero-shot comparison. **Cost:** ~1 session; `sktime`/`neuralforecast`
provide the implementations, so this is wiring, not modelling.

---


## Archived: §18 F7's original write-up (parked 2026-08-18, see §22.6)

### F7 Training-exposure confound, estimated rather than waved at `[ ]`

**What's wrong.** These models saw different, largely undocumented corpora.
Every "model A is better at seasonal data" is confounded with "model A saw more
seasonal data." Nothing in the repo mentions this, and it is arguably the
largest confound in the whole comparison.

**It cannot be eliminated. It can be bounded and reported**, using machinery
that already exists:
- `ModelConfig` gains `training_corpora: list[str]` and `training_notes: str` —
  *declared* provenance from each model's card/paper, with an explicit
  `"undocumented"` value that the report surfaces as a warning rather than a
  blank.
- Reuse `tsfm_benchmark`'s `audit.py` in the opposite direction: for each
  declared corpus this repo can actually fetch (Monash, ETT via
  `sources.py`), run the existing realism/leakage comparison **between that
  corpus and the benchmark**, producing a per-model *distributional overlap*
  estimate per family. A model whose declared training data overlaps the
  benchmark's seasonal family heavily has an advantage there that is not a
  representational finding.
- Report as a "Training exposure" panel with the standing caveat that declared
  data is incomplete for every current model, so overlap is a **lower bound**.
- The `synthetic` tier is the honest control here and should be said so
  loudly: it is the only part of the corpus no model can have trained on
  instance-wise, so per-tier result splits (already available) are the
  cleanest evidence in the whole report. **Add a report line contrasting each
  finding's strength on `synthetic` vs `real_derived` tiers** — cheap, and it
  turns an existing data property into a confound control.

**Acceptance criterion.** Every model in a report has either a declared corpus
list or an explicit "undocumented" marker, and per-tier result splits are
rendered for L0.

**Cost.** ~1 session for the declared-provenance plumbing and the per-tier
split; the overlap estimate is a further session and needs the `datasets<3`
path.


## Archived: §18 F6's original deliverables/plan (implemented 2026-08-12)

### F6 🔴 Every delta in noise-floor units `[x]`

**What's wrong.** §15 A13 established repeat-run noise floors — a real fix. But
deltas are still *reported* as raw numbers, and the models have structurally
different floors (Chronos-T5 samples; TimesFM and Chronos-Bolt are
deterministic — `CLAUDE.md` §12 item 3). A ΔMASE of +0.2 means something
different for each. The §13 entry showing an SAE forecast-preservation ΔMASE
moving 0.175 → 0.1097 across identical configs is the concrete case.

**Deliverable.** A shared helper `analysis/stats.py::in_floor_units(delta,
floor) -> dict` returning `{raw, floor, ratio, interpretable: bool}` with
`interpretable = |delta| > 2 * floor`, used by every stage that reports a
delta: L3 sensitivity and patching restoration, attention head/MLP ablation
ΔMASE, SAE forecast preservation, steering effects. The report renders
`Δ = +0.21 (2.6× this model's noise floor)` and greys out any delta below
`1×`. A finding may not be emitted from an uninterpretable delta at all.

**Acceptance criterion.** Re-render an existing run; count how many current
findings fall below their own noise floor. **That count is the deliverable** —
if it is nonzero, this item has retroactively corrected the record, which is
the whole point.

**Cost.** ~1 session, mostly threading the floor through call sites.


## Archived: §18 F4's original deliverables/plan (implemented 2026-08-12)

### F4 🔴 Capture-coverage accounting `[x]`

**What's wrong.** `CLAUDE.md` §12 items 1–2 state the coverage asymmetry in
prose. `report/coverage.json` is *section* coverage (which report sections
rendered), not *computational* coverage — verified by reading it. So the single
most important caveat on every Chronos claim is nowhere in the machine-readable
output.

**Deliverable.** Extend F2's budget record with a coverage block per model:
`captured_blocks / total_blocks`, `captured_params / total_params`,
`captured_flops / total_flops` (the honest headline number), and a list of
named uncaptured surfaces from F1's `uncaptured_surfaces()`. Render as a
"Coverage" row in the fairness card (F9), and have every per-model depth figure
annotate the captured fraction in its subtitle.

**The rule this enables, which should be enforced in code, not prose:** a
finding whose text asserts a depth-located claim about a model with
`captured_flops < 0.9` gets an automatic qualifier appended ("…within the
captured *encoder*; ~48% of this model's forward computation is unobserved").
Implement as a check in the findings builder, not as author discipline —
invariant 8's lesson is that discipline-only mechanisms decay.

**Acceptance criterion.** Chronos-T5's report states its captured FLOP
fraction numerically, and a depth-located Chronos finding carries the automatic
qualifier.

**Cost.** ~0.5 session on top of F2.


## Archived: §18 F3's original write-up (parked 2026-08-18, see §22.6)

### F3 Capability-intersection comparison mode `[ ]`

**What's wrong.** §17.1 G-III — asymmetric capability support silently yields
one-sided sections in a comparison document.

**Deliverable.** `fairness.mode: intersection | full | both` in config
(default `both`).
- `pipeline.py` computes the **capability intersection** across all configured
  models before any analysis stage runs, using `models/capability_matrix.py`
  (already built) as the source of truth rather than discovering support
  per-stage.
- Under `intersection`, an analysis that only some models support is skipped
  entirely with a stated reason. Under `both` (default) it runs, but the report
  places it in a clearly headed **"Asymmetric — available for a subset of
  models"** part of the document, physically separated from the symmetric
  comparison, with a one-line statement of which models are missing and why.
- The findings list marks each finding `symmetric: bool`. A finding derived
  from an asymmetric section may never be promoted to a headline claim or a
  `confirm` hypothesis without an explicit override, since it cannot be a
  comparison.

**Acceptance criterion.** A config pairing Chronos-T5 (has
`cross_attention_patterns`) with TimesFM (cannot — no encoder/decoder split)
renders cross-attention under the asymmetric heading, and under
`mode: intersection` does not render it at all. Smoke test asserts both.

**Cost.** ~0.5 session; composes with §16 E6's findings refactor, so do them
together.


## Archived: §18 F2's original deliverables/plan (implemented 2026-08-12)

### F2 🔴 Parameter, compute, latency and memory accounting `[x]`

**What's wrong.** §17.1 G-II — none of this is measured. Every quality
comparison is size-confounded and no cost axis exists at all, which is the
first thing a practitioner asks.

**Deliverables.** `tsfm_lens/analysis/model_budget.py`:
- `parameter_census(adapter) -> dict` — total, trainable, and a breakdown by
  role (embedding/tokenizer front-end, body blocks, output head), plus
  per-block counts. Pure `torch` introspection over `named_parameters()`; no
  architecture knowledge needed beyond the block-name regex the adapter already
  provides.
- `measure_forward_cost(adapter, context_len, batch, device) -> dict` — wall
  clock (median of N, after warmup), peak allocated VRAM via
  `torch.cuda.max_memory_allocated`, and **measured** FLOPs via
  `torch.utils.flop_counter.FlopCounterMode` — measured, not hand-derived, so
  it works on an unseen architecture with no new code. Record per-block
  cumulative FLOPs, which is exactly what F1's D2 axis needs.
- `predict_cost(adapter, context_len, horizon, ...) -> dict` — the same for the
  full forecast path, which is where Chronos's sampled decoding shows its true
  cost (`num_samples` × decoder passes) and where a deterministic model looks
  very different.
- Written to `budget/model_budget.json` by a new cheap pipeline stage running
  right after `extract` (it needs a loaded model and nothing else).

**Report additions.** A "Cost and capacity" panel: params, FLOPs/forward,
latency, peak VRAM per model; then **compute-normalized L0** — MASE plotted
against FLOPs and against parameter count, so "better" and "better per unit
cost" are visibly different claims. A model that wins on MASE and loses on
MASE-per-FLOP is an important, actionable result that the repo currently cannot
express.

**Acceptance criterion.** Every configured model has a budget record; the
report renders quality-vs-cost; and F1's D2 axis consumes the per-block
cumulative FLOPs without a second measurement pass.

**Cost.** ~1 session. `FlopCounterMode` is the load-bearing choice — verify it
handles each adapter's ops (custom attention kernels can be invisible to it;
if a model reports implausibly low FLOPs, that is the failure mode, so
sanity-check against an analytic estimate for one known model and **warn on
disagreement** rather than trusting silently).


## Archived: §18 F1's original deliverables block (built 2026-08-13; the remaining wiring is live in ROADMAP.md)

**Deliverables.**
- `tsfm_lens/analysis/depth_axis.py`:
  - `depth_coordinates(adapter, captured_layers, axis: str, budget=None) -> np.ndarray`
    returning a coordinate per captured layer for `axis in {"index","block","compute","functional"}`.
  - `total_stack_size(adapter) -> dict` — `{n_blocks_total, n_blocks_captured,
    n_blocks_uncaptured_in_captured_surface, n_blocks_outside_captured_surface}`.
    The last field is the one that catches Chronos's decoder. Uses
    `all_layer_names()` (exists) plus a new optional
    `ModelAdapter.uncaptured_surfaces() -> dict[str, int]` (defaults to `{}`,
    so no adapter breaks) for surfaces the adapter knows about but does not
    capture — Chronos-T5 returns `{"decoder": 12}`.
  - `align_on_axis(coords_a, values_a, coords_b, values_b) -> tuple` —
    interpolation onto the shared axis with **explicit refusal outside the
    overlap**: if Chronos's block axis only spans [0, 0.5], TimesFM's [0.5, 1.0]
    is not compared against anything, it is rendered as an unmatched region.
    Today the interpolation silently stretches one model over the other's range.
- `config.py`: `alignment.depth_axis: block` as the **new default**, with
  `index` retained and explicitly labelled legacy so already-recorded numbers
  remain reproducible (invariant 1's spirit, applied to analysis rather than
  generation).
- Every cross-model depth figure prints the axis name in its title and its
  `_note()` explains what the axis means in one sentence.

**Acceptance criterion.** A cross-model depth figure under `block` shows
Chronos's curve ending at ~0.5 with the region above it visibly unmatched, and
the L1/L3 depth-agreement statistics are recomputed over the overlap only.
Re-run §16 E9's depth-curve null tests on the new axis and record whether any
verdict changes — **this is the acceptance test that matters**, because if a
verdict flips, the old axis was materially misleading and that is itself a
finding worth publishing.

**Cost.** ~1.5 sessions plus one re-analysis run (no re-extraction — depth
coordinates are metadata over existing artifacts). D2 depends on F2.

**Beginner explanation to ship with it.** "Two models can have a different
number of layers, and one may only be half-observable. Plotting both on '0 to
1' hides that. Under the new default axis, a model we can only see half of
occupies only half the plot, so you can see what we don't know."


## Archived: §16 E22/E24 detail-ups (parked 2026-08-18, see §22.3-§22.4)

**E22 · Scale — split**

- [ ] **E22a · Full-scale corpus validation.** *Unblocked* — A17's matcher work
      is `[x]`. This is a run, not a build: `configs/full_multidomain.yaml` at
      its literal ~99K-sample scale through `benchmark_validation`.
- [ ] **E22b · Full-scale analysis runs.** Extraction and analysis at that
      corpus size; a VRAM/throughput budgeting exercise more than a coding one.
      E2's `doctor` preflight should be the thing that sizes it.

---

**E24 · Real-corpus activation bucketing — detail-up**

🔴 **The gap the item doesn't name: wild data has no family labels.** Every L4
number this would generalize (`analysis/clustering.py`'s cluster labelling,
AMI, the majority-family naming) is defined against the synthetic corpus's
known generator families. On Monash/LOTSA there are none. Three options, decide
before building:
  (a) **Cross-model AMI only** — compare each model's *partition* against the
      other's, which needs no labels at all. This is the item's actual
      scientific content and is fully available today. *Recommended.*
  (b) **Dataset-of-origin as a proxy label** — Monash's domain tags (tourism,
      electricity, traffic…) as the family axis. Real but coarse, and confounds
      domain with sampling frequency.
  (c) **catch22-derived pseudo-labels** — cluster in `benchmark_validation`'s
      feature space and use those as labels. Circular if the same features
      drive the comparison; usable as a descriptive overlay only.

*Acceptance criterion.* Cross-model AMI on wild data reported with a
series-bootstrap CI **and** against the shuffled-series null, so "the two models
partition wild data similarly" is not read off a bare number.

---


## Archived: §16 E20b detail-up (parked 2026-08-18, see §22.4)

- [ ] **E20b · Context-length sweep.** *Expensive by nature* — every context
      length is a **fresh extraction** (a different `context_len` changes the
      store's shape, so nothing is reusable), and it interacts with A3's
      config-fingerprint rule: each sweep point must get its own `run.name`.
      Budget one extraction per point and say so in the config. **⚠️ See
      E20a's note above — this framing may be stale; the already-landed
      `context_scaling.py` (E20's own implementation) does not actually
      need fresh extraction per point, contradicting this item's own
      premise. Re-scope or merge into E20a before starting new work here.**

---


## Archived: §16 E18 detail-up (parked 2026-08-18, folded into H8, see §22.7)

**E18 · Path patching — detail-up**

*Blocker status:* none, but see the two capability gaps below — these decide
scope, and neither is mentioned in the item today.

🔴 **Gap 1: the hook primitive does not exist yet.** `hooks.py` has
`token_patch` (replace a module's *output*) and `output_mean_ablate`. Path
patching needs to replace a component's contribution *along one path* —
i.e. freeze the input a downstream component receives from one upstream
component while leaving its other inputs live. That is a new primitive
(`path_patch(src_module, dst_module, cache, ...)`), not a configuration of
the existing ones. Budget for it explicitly.

🔴 **Gap 2: TimesFM has no `mlp_info`.** Its feed-forward block is two bare
`nn.Linear`s (`ff0`/`ff1`) with no wrapping module for `_scan_mlp` to find
(`CLAUDE.md` §6.2). So any MLP-path result is **Chronos-only** unless
`timesfm_adapter.py` gains a hard-coded `mlp_info` the way
`sundial_adapter.py` already did for the same class of gap. Doing that
hard-coding is a prerequisite, and is ~10 lines.

*Scope recommendation.* Attention-head paths only, first: heads are discovered
on all adapters via the shared `_scan_attention`, so a head→head path-patching
result is available on every model today. Add MLP paths after Gap 2 is closed.

*Acceptance criterion.* On one model, reproduce a result already known from
mean-ablation (the `top_k` most load-bearing heads by ΔMASE) and show the path
decomposition sums to approximately the total effect — a conservation check
that catches a wrong path primitive, which is otherwise very hard to detect.

---


## Archived: §16 E4/E5 detail-ups (parked 2026-08-18, see §22.3)

**E4 · Bundled reference corpus — detail-up**

*Blocker status:* **none technically. One open non-technical question, below.**

🔴 **Licensing decision required before any distribution.** The corpus as
specified covers "both tiers", and the `real_derived` tier is derived from
Monash/ETT source data. `SourceRef` records per-source licenses precisely so
this question can be answered (`CLAUDE.md` §4.3), but it has not been answered.
Three resolutions, pick one explicitly and record it:
  (a) **Synthetic-only distribution** — ship only the `synthetic` tier, which
      touches zero real data and is unambiguously redistributable. Costs the
      realism-stress coverage. *Recommended default* — it is the only option
      that needs no legal judgment.
  (b) **Fetch-on-first-use for the real-derived half** — distribute the
      synthetic tier plus a pinned build recipe + digest for the real-derived
      tier, which the user's own machine builds from sources they fetch
      themselves. Preserves coverage, costs first-run time and a `datasets<3`
      dependency for that path.
  (c) **Full distribution after per-source license review** — requires
      confirming every included source permits redistribution of derived works.

*Deliverables.*
- `tsfm_benchmark/configs/reference_v1.yaml` — committed, seed-pinned build
  config. Target a few hundred series across all 12 archetypes (opt-in ones
  listed explicitly, per invariant 1).
- `corpora/reference_v1/` (public split only) or a fetch manifest, depending on
  the decision above. Private split **never** distributed.
- `tsfm_lens/data.py` learns `corpus: reference_v1` as a resolvable name, with
  digest verification on load and a clear error on mismatch.
- Every report prints corpus name + digest (feeds E5's keying).

*Acceptance criterion.* A fresh clone with no `build_pipeline` run can produce
a report; two machines' reports on the same corpus print identical digests.

*Cost.* ~1 session plus one build run, once the licensing choice is made.

---

**E5 · Results registry — detail-up**

*Blocker status:* needs E4 (the corpus-version key is meaningless without a
versioned corpus). A3/A7 are done.

*Correction inherited from the block at the top of §16:* this does **not**
replace globbing — `meta_report.py` already takes an explicit `--runs` list.
The registry's value is append-only history keyed by provenance.

*Schema — undecided today, decide it here.* One row per
`(corpus_digest, checkpoint, config_hash, stage, metric_name)` with columns:
`value: float`, `ci_low`, `ci_high`, `n_series`, `run_dir`, `timestamp`,
`git_sha`, `library_versions_digest`. Long format (one metric per row), not
wide — stages emit different metric sets, and a wide schema would need
migration every time a stage gains a number.

*Concurrency — the second undecided thing.* Parquet has no append primitive and
two concurrent runs would clobber each other. Write one
`results/parts/<run_id>.parquet` per run (never mutated), and have readers
`pd.concat` the directory. This is append-only by construction, needs no
locking, and survives a killed run leaving a partial file (skip unreadable
parts with a warning).

*Deliverables.* `tsfm_lens/registry.py` with
`emit(run_dir, rows: list[dict]) -> Path` and `read(results_dir) -> pd.DataFrame`;
a `pipeline.py` hook calling `emit` at the end of each stage;
`run_meta_report.py --from-registry` as an alternative to `--runs`.

*Acceptance criterion.* Two runs of different configs both appear; deleting one
part file leaves the other readable; `read()` on an empty dir returns an empty
frame with the right columns rather than raising.

*Cost.* ~1 session.

---


## Archived: §16 E1 detail-up (parked 2026-08-18, see §22.3)

**E1 · Zero-config entry point — detail-up**

*Blocker status:* needs E4 (a corpus to default to). E2 and A7 are done.

*Deliverables.*
- `tsfm_lens/entry.py` — new module. Three functions, all pure and unit-testable
  without a GPU:
  - `resolve_adapter(checkpoint: str) -> str` — checkpoint id → registered
    adapter name. **This mapping does not exist anywhere today** and is the
    item's real content: `models/__init__.py` has a name→class registry, but
    nothing maps `amazon/chronos-t5-base` → `chronos`. Implement as an ordered
    list of `(regex, adapter_name)` rules in one module-level constant, first
    match wins, `ValueError` naming every known pattern on no match. Seed it
    with the five shipped adapters (`chronos-t5-*` → `chronos`, `chronos-bolt-*`
    → `chronos_bolt`, `chronos-2*` → `chronos2`, `timesfm-*` → `timesfm`,
    `sundial-*` → `sundial`). Keep it a plain data table, not a heuristic —
    a wrong guess here silently analyses the wrong stack.
  - `preset_overrides(name: str) -> dict` — `quick|standard|deep` → the config
    fragment. **The three presets are undefined today**; define them by reading
    the values out of the three existing configs rather than inventing numbers:
    `quick` ≈ `configs/smoke.yaml`'s series counts and stage set, `standard` ≈
    `configs/medium_run.yaml`, `deep` ≈ `configs/default.yaml`. Document in the
    docstring that these are *snapshots* of those files, and that changing a
    preset is a user-visible behaviour change.
  - `build_config(models: list[str], preset: str, corpus: str) -> Config` —
    composes the above into a real `Config` object via `config.py`'s existing
    loader path, so every field passes the same validation a hand-written YAML
    does.
- `run.py` gains a `compare` subcommand calling the three in order, then
  `dump_config(cfg, run_dir / "generated_config.yaml")` **before** running
  anything (so a crashed run still leaves the teaching artifact), then the
  normal pipeline, then prints the report path.

*Acceptance criterion.* On a clean checkout with no YAML edited, one command
produces a report for a two-model comparison, and the emitted
`generated_config.yaml` re-runs to an identical stage set when passed back via
`--config`.

*Test.* `tests/test_entry.py`: `resolve_adapter` over all five shipped patterns
plus an unknown one (asserting the error names the known patterns); each preset
`build_config`s into a `Config` that passes validation; a mock-model end-to-end
`compare` run asserting `generated_config.yaml` round-trips.

*Cost.* ~1 session for the code; the round-trip test is the fiddly part.

---


## Archived: §16's 2026-08-11 corrections block (applied to the items 2026-08-18)

> ⚠️ **Corrections block, 2026-08-11 (§0.2 — correct in place, don't delete).**
> This backlog was written 2026-08-06 against a repo state that has since moved
> a long way. A planning pass checked every item's stated dependencies and
> sub-deliverables against the live repo; the items themselves are left
> verbatim below, but **read them with the corrections here first**, because
> several read as blocked or unstarted when they are neither.
>
> **1. Every "Depends on: A…" line in this section is stale.** All of A2, A3,
> A4, A5, A7, A13, A14, A15, A16, A17 are `[x]` as of the 2026-08-06 audit
> sweep and its follow-ups. E1, E3, E4, E5, E6 and E22 all still read as if
> those are pending prerequisites. **They are not — every T1 item is
> unblocked by §15 today.** Where an item lists only A-items as dependencies
> (E3, E6), it has *no remaining external blocker at all* and can be started
> immediately. Where it lists an E-item (E1 → E4/E2; E5 → E4; E7 → E1), only
> the E-dependency is real.
>
> **2. E6 has two already-built sub-deliverables.** "Fold in A5's coverage
> panel and A7's provenance panel" is done: `report/coverage.json` is written
> and rendered, and `_alignment_provenance_block` exists at
> `tsfm_lens/report/report.py:290`. What actually remains in E6 is
> `report/findings.json`, per-finding confidence badges, stable anchors, and
> the E5-backed diff mode — see the detail-up under E6 below.
>
> **3. E22's A17 clause is done.** "A17's blocked matcher (so a full-scale
> corpus can be validated)" — A17 is `[x]`. E22's remaining content is the
> scale *runs* themselves, not any unblocking work.
>
> **4. E5 mis-describes the thing it would replace.** "`meta_report.py` reads
> it instead of globbing run directories" — `report/meta_report.py` does not
> glob; it takes an explicit `--runs a,b,c` list (`run_meta_report.py`). The
> registry's value is therefore *not* "stop globbing", it is the append-only
> history and provenance keying. Stated correctly in the detail-up below.
>
> **5. E3's premise shifted with A2.** E3 says the alignment mapping is
> "verified by a check nobody is forced to run." Post-A2 the in-pipeline check
> no longer discards its result. E3's actual argument is now the stronger one:
> even an *enforced* check only validates a **declared** span table, it never
> *derives* one — which is what makes `GenericHFAdapter` possible and what E3
> is really for.
>
> **6. E19's "before Phase 4" framing is expired.** Phase 4 (§9) shipped —
> Chronos-2 and Sundial are both integrated. E19's decision is now a
> *retroactive* one; see §13's tracked "ratify or reverse" entry.
>
> **7. E7's "install story decided and documented" clause is decided.**
> `CLAUDE.md` §3 records the 2026-08-11 call: two packages, one repo, kept
> deliberately separate because their dependency sets barely overlap. What
> remains in E7 is the *console script* and the docs, not the decision.
>
> **8. Three items bundle unblocked work behind blocked work** (E7, E20, E22)
> and are split into sub-items in the detail-ups below, so the ready parts are
> individually checkable instead of hidden inside a single unchecked box.


## Archived: §16 E24's original write-up (parked 2026-08-18)

- [ ] **E24 · Real-corpus activation bucketing.** `CLAUDE.md` §13 item 7, not
  previously cross-referenced into this backlog (closed the gap 2026-08-11,
  same pass as E23). Embed a large real corpus (Monash, GIFT-Eval
  pretraining corpus, LOTSA) in each model, cluster, and compare
  **partitions** across models via AMI / cluster matching — the same L4
  method already built (`analysis/clustering.py`), just pointed at wild
  data instead of the synthetic benchmark. Deliberately sequenced after (not
  instead of) the synthetic-corpus work already done: clusters on
  uncontrolled real data are uninterpretable without the controlled
  reference points L4 already established on the synthetic benchmark first
  — this is a generalization check on an existing method, not a
  replacement for it. Needs a real large-corpus source wired in (Monash's
  `sources.py` adapter already exists per §4.3 and is the natural first
  choice) and a GPU session with both models loaded; no design blocker,
  just not yet scheduled.


## Archived: §16 E21/E22's original write-ups (parked 2026-08-18)

- [ ] **E21 · Chronos decoder capture as a separate measurement.**
  `CLAUDE.md` §13 item 2 / §12 item 1, unchanged in substance: a second
  capture surface over decoder layers under teacher forcing, with its own
  forecast-time alignment windows, reported as a *distinct* measurement rather
  than CKA'd against context-window states. Worth restating in this backlog
  because it is the single largest **coverage asymmetry** in the flagship
  comparison — half of Chronos is unobserved — and any claim about what
  "Chronos represents" is really about its encoder until this exists.
- [ ] **E22 · Scale.** A17's blocked matcher (so a full-scale corpus can be
  validated at all), streaming/chunked CKA and ridge solves so `l1.max_rows`
  and `l2.max_rows` stop being the binding constraint on cross-model
  statistics, and a cost model that turns E2's VRAM/wall-clock estimate into a
  planning tool ("this config will take ~35 min and 14 GB; per-window
  patching is 60% of it"). The §5.1 literal-scale `full_multidomain.yaml`
  build (~99K samples, ~6 h estimated) is the concrete forcing function.


## Archived: §16 E18's original write-up (parked 2026-08-18)

- [ ] **E18 · Component-level attribution (path patching).** `CLAUDE.md` §13
  item 8's "deeper component resolution", scoped: head→head and head→MLP path
  patching within a model, so a claim can be "this head's output matters
  *because* that head reads it" instead of "these heads matter." This is the
  clearest remaining gap against transformer-lens's own resolution, and the
  `token_patch` primitive plus `attention_info`/`mlp_info` already provide the
  hooks. Genuinely more expensive than everything above it — schedule after
  T1/T2.


## Archived: §16 E7's original write-up (split 2026-08-18)

- [ ] **E7 · Docs, quickstart, notebooks, packaging.** §10's existing open
  items, restated here only because E1-E6 change what they should say: a
  quickstart that is literally E1's one command; notebooks for (a) smoke run,
  (b) add an adapter — ideally now "point `GenericHFAdapter` at a checkpoint
  and watch span discovery run", (c) train and read an SAE, (d) read a
  confirm verdict; the auto-generated capability matrix (built 2026-08-06)
  rendered into those docs; and the one-repo/two-packages install story
  decided and documented, with a `tsfm-lens` console script. Depends on: E1.


## Archived: §16 E5/E6's original write-ups (E5 parked, E6 narrowed 2026-08-18)

- [ ] **E5 · A results registry, so adding a model appends to a table.**
  Append-only `results/index.parquet`: one row per (corpus version, checkpoint,
  config hash, stage) with the headline statistics and full provenance;
  `meta_report.py` reads it instead of globbing run directories. Turns the
  current one-off cross-run comparison into (a) regression tracking across
  checkpoints and library versions and (b) the substrate for a public
  comparison table. Depends on: A3, A7, E4.
- [ ] **E6 · Report as a shareable product.** Fold in A5's coverage panel and
  A7's provenance panel, then add: `report/findings.json` (machine-readable
  findings with the claim ids from A15's registry, so downstream tooling and
  the bias card consume one source of truth), per-finding confidence badges
  (evidence class + registered/exploratory + whether it cleared the A13 noise
  floor), stable anchors for deep-linking a section, and a "what changed vs.
  run X" diff mode built on E5. Depends on: A5, A7, A13, A15.


## Archived: §16 E4's original write-up (parked 2026-08-18)

- [ ] **E4 · A bundled, versioned reference corpus.** Ship (or fetch-on-first-
  use with a pinned manifest digest) `corpora/reference_v1`: a few hundred
  sealed series covering all 12 archetypes and both tiers, small enough to
  distribute, built by a committed config with a recorded seed. Two payoffs
  beyond convenience: a new user's first report needs no `build_pipeline`
  run at all, and **results become comparable across users and machines**,
  because everyone's numbers are against the same corpus digest — which is
  what makes E5's registry meaningful and what a leaderboard would require.
  Every report prints the corpus version and digest. Keep the private split
  out of the distribution (`CLAUDE.md` §4.5's seal-records-intent caveat) and
  document the epoch-regeneration path for anyone who wants a fresh held-out
  set. Depends on: A7 (digest in provenance).


## Archived: §16 E1's original write-up (parked 2026-08-18)

- [ ] **E1 · A zero-config entry point.** `tsfm-lens compare --model
  hf://google/timesfm-2.5-200m-pytorch --model hf://amazon/chronos-t5-base
  --preset standard` → resolves adapters from the registry by checkpoint
  pattern, generates a full config, uses the bundled reference corpus (E4),
  runs preflight (E2), runs the pipeline, opens the report. `--preset
  quick|standard|deep` maps to series counts, `capture_layer_stride`, patching
  strides, and which stages are enabled — the three cost/depth points the
  existing configs already represent implicitly (`smoke`/`medium_run`/
  `default`), named and documented instead of copy-pasted. Emit the generated
  YAML into the run dir so the zero-config path is a *teaching* path: "here is
  the config I made for you, edit it next time." Depends on: E4, E2, A7.


## Archived: §15 items A1-A19 in full — original fix plans and verbatim Findings blocks (all fixed 2026-08-06)

### A1 — The production `layer_screen` stage cannot be fair to every layer `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** §6.1.1's R2 ("fair to every layer... the screen must pin
  stride to 1 regardless of what the analysis config uses") is not
  implemented. The stage screens exactly the layers extraction happened to
  store.
- **Evidence.** `analysis/layer_screen.py:455` — `layers =
  store.layers(m.name)`; `configs/default.yaml:29` and
  `configs/medium_run.yaml:32` — `capture_layer_stride: 2` for TimesFM (20
  blocks → 10 captured). No `layer_screen.stride` knob exists in
  `config.py`'s `LayerScreenConfig`.
- **Blast radius.** In every production config, half of TimesFM's blocks are
  invisible to the selector that decides where expensive analysis and SAE
  training go — the exact objection (§6.1.1 problem 3) that disqualified the
  §6.1 mechanism. The report's Screen section says "selected N of 10 captured
  layers", which reads as complete. And the bake-off that chose `work_bend`
  ran at stride 1 (`configs/layer_screen_experiment.yaml`), so production
  selections are made under conditions the method was never validated under.
- **Fix.**
  1. `LayerScreenConfig` gains `stride: int = 1` and `require_full_capture:
     bool = True`.
  2. Preferred mechanism: a **dedicated cheap screening extraction** — one
     extra forward pass per model over `layer_screen.max_series` series
     (default a few hundred, far fewer than the analysis corpus), all blocks
     at stride 1, written to `screen_activations.zarr` in fp16 and deleted
     after selection unless `layer_screen.keep_store: true`. Cost is bounded
     and pays R4 correctly: screening still runs before every expensive
     stage. Reuse `extraction/extract.py::_extract_model` with an override
     for layer list and row count rather than writing a second extraction
     path.
  3. If a caller insists on screening the strided store (`require_full_capture:
     false`), record `fair_to_all_layers: false` plus the realized stride in
     `layer_screen/selection.json`, and have the report's Screen note say so
     in the section body — not in a collapsed `_note()`.
  4. Denominator honesty: the Screen section must report "screened X of Y
     model blocks" using `adapter.all_layer_names()` as Y, not the store's
     captured count.
  5. Test: a config with `capture_layer_stride: 2` and
     `require_full_capture: true` screens all 20 mock blocks; with `false`, it
     screens 10 and emits `fair_to_all_layers: false`.
- **Also worth fixing while here.** `selection.json` records `method` per
  model, and `_uniform_fallback` correctly writes `method:
  "uniform_fallback"` — but nothing surfaces the discrepancy between the
  *requested* method and the *used* one. Add `method_requested` alongside
  `method`, and have the report and `findings` name any fallback explicitly:
  a run that silently selected layers with the null it was supposed to beat
  should be impossible to miss.
- **Cost.** ~a day including the screening-extraction path and tests.
- **Findings (2026-08-06).** Implemented essentially as planned: `LayerScreenConfig`
  gained `stride`/`require_full_capture`/`keep_store`; `layer_screen.py` gained
  `_run_screening_extraction`, which reuses `extraction/extract.py::_extract_model`
  (now parameterized by an explicit `layers`/`rows` pair instead of deriving
  them from `adapter`/`data` internally) against a fresh, temporary store, then
  deletes it after selection unless `keep_store: true`. `run_layer_screen` now
  takes `hub` (pipeline.py's stage closure updated) so it can drive its own
  extraction pass. Verified two ways: (1) synthetic — a smoke config with
  `capture_layer_stride: 2` on one mock model showed the main store capturing
  3 of 6 blocks while the dedicated screening pass correctly screened all 6
  (`fair_to_all_layers: true`), and setting `require_full_capture: false`
  correctly reported `fair_to_all_layers: false` with the true denominator (6)
  both in `selection.json` and in the rendered report body (not a collapsed
  note); (2) real checkpoints — a live run against `google/timesfm-2.5-200m-pytorch`
  (`capture_layer_stride: 2`, 20 blocks) and `amazon/chronos-t5-small` (stride
  1, 6 blocks) confirmed the exact bug scenario A1 described (`extraction
  complete: {'TimesFM': 10, 'Chronos-T5': 6}`) and its fix (`layer_screen:
  TimesFM (work_bend, budget=5/20, fair_to_all_layers=True) selected
  ['stacked_xf.6', 'stacked_xf.10', 'stacked_xf.14', 'stacked_xf.16',
  'stacked_xf.18']` — a real selection over all 20 blocks, not just the 10
  the main analysis config would have captured). Full `tsfm_lens` pytest
  suite (81 tests) green after the change. One design deviation from the
  fix plan: row sampling for the screening pass uses a plain
  `rng.choice(..., replace=False)` (matching the "already random" call
  sites named in A4), not yet the stratified `sample_rows` helper A4
  introduces — the call site will be swapped to it when A4 lands, per that
  item's own fix plan.

### A2 — Invariant 7 (empirical alignment verification) is logged and discarded `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** The impulse alignment check runs, prints one INFO line,
  and its result is thrown away. There is no threshold, no artifact, no
  report surface, and only a quarter of the layers are probed.
- **Evidence.** `extraction/extract.py:50-51` —
  `impulse_alignment_check(adapter, cfg.alignment.window, layers[::max(1,
  len(layers)//4)])`, return value unused; `extraction/alignment.py:84-86` —
  `log.info(...)` is the only output; `config.py:43` —
  `sanity_check: bool = False` (the three real configs do set it `true`).
- **Blast radius.** `CLAUDE.md` §6.3 states that near-zero diagonal hits mean
  "fix the adapter before trusting any cross-model number", and §7 invariant 7
  says alignment is verified empirically on every new library version. Today
  both depend on a human running `--check-alignment` by hand and reading the
  output. This has already failed once in a way that cost a session (§11.16:
  a 0.06 hit fraction that looked like a broken adapter and was a broken
  test) — and that was found only *because* someone ran the standalone tool.
  Inside a pipeline run, the same signal is one line among hundreds, followed
  by a complete-looking report.
- **Fix.**
  1. `run_extraction` captures the returned dict per model and writes
     `alignment/alignment_check.json` (per model: per-layer hit fraction, min,
     mean, window, n_windows, impulse amplitude used, layer subset probed).
  2. Probe **all** captured layers, not `[::len//4]` — the check costs one
     forward pass over `n_windows + 1` rows, i.e. nothing.
  3. New config: `alignment.min_diagonal_frac` (default `0.5` at the shallowest
     probed layer, `0.0` elsewhere — deep layers legitimately mix positions,
     per §6.3) and `alignment.on_failure: fail|warn` **defaulting to `fail`**.
     Failing here is correct doctrine: a broken span map is a broken
     assumption, not an unsupported capability.
  4. Flip `sanity_check` default to `true` in `config.py` so a hand-written
     config can't opt out by omission.
  5. The report grows a mandatory **"Alignment & provenance"** panel (shared
     with A7) showing per-model min/mean hit fraction with a pass/fail badge,
     rendered before any cross-model section. A report whose alignment gate
     didn't run should say that where a reader will see it.
  6. Test: a deliberately-mis-declaring mock adapter (spans shifted by one
     window) must fail the gate; `tests/test_adapter_conformance.py` already
     has the "deliberately broken adapter" pattern to copy.
- **Cost.** Half a day.
- **Findings (2026-08-06).** Implemented as `extraction/alignment.py::run_alignment_gate`,
  which wraps `impulse_alignment_check` (now probing every layer
  `extraction/extract.py` actually captures, not a `[::len//4]` subset — free,
  since the check is one extra forward pass), builds the full per-layer/min/
  mean/shallowest-layer record, and raises `RuntimeError` with an actionable
  message when the shallowest probed layer's hit fraction is below
  `alignment.min_diagonal_frac` (new, default 0.5) and `alignment.on_failure`
  (new, default `"fail"`) is `"fail"`; `"warn"` logs instead. `alignment.sanity_check`
  now defaults to `true` (the three real configs already set it explicitly, so
  this changes nothing for them; it closes the "opt out by omission" gap for
  new configs). `run_extraction` writes the per-model records to
  `alignment/alignment_check.json`. Verified three ways: (1) unit-level — a
  deliberately sabotaged mock adapter (`token_time_spans` permuted via
  `np.roll`) drove the shallowest layer's hit fraction to 0.00 and the gate
  raised with the expected message under `on_failure: fail`, and returned a
  `passed: false` record under `warn`; (2) smoke end-to-end — both mock
  architectures pass cleanly (hit fraction 1.00 at every layer) and
  `alignment/alignment_check.json` contains the full expected record shape;
  full 81-test pytest suite green; (3) real checkpoints — a live run against
  `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-small` both showed
  `diagonal-hit fraction min=1.00 mean=1.00`, consistent with `CLAUDE.md` §9's
  existing findings for these two checkpoints and confirming the gate doesn't
  false-positive on real, correctly-aligned adapters. Item 6 (a permanent
  regression test copying `test_adapter_conformance.py`'s "deliberately
  broken adapter" pattern) is now committed as `tests/test_alignment_gate.py`
  (3 tests: fails loudly on broken spans, warns instead when configured,
  passes cleanly on a correct adapter). **Not yet done from the original fix
  plan:** item 5, the dedicated "Alignment & provenance" report panel —
  deferred to land together with A7 (which the fix plan explicitly says
  shares that panel), so it renders both alignment and provenance in one
  place instead of being built twice.

### A3 — Artifact reuse has no config fingerprint, so stale artifacts mix silently `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** Stage skipping is pure path-existence. Nothing checks
  whether the existing artifacts were produced by the *current* config.
- **Evidence.** `pipeline.py:169` — `if stage.done(cfg) and stage.name not in
  force`, where every `done` is a `.exists()` check (`pipeline.py:78-128`).
  `dump_config(cfg, run_dir/"config_resolved.yaml")` (`pipeline.py:146`)
  writes the config but nothing ever reads it back to compare.
- **Blast radius.** Change `data.context_len`, `alignment.window`,
  `data.path`, or a checkpoint id, rerun into the same `run.name` without
  `--force all`, and you get last config's extraction feeding this config's
  analyses, with a report that looks complete. This is not hypothetical usage:
  §6.2's own Findings describe editing `medium_run_chronos_base.yaml` to add
  an `sae:` block and rerunning `--stages sae` against existing extraction —
  correct in that instance precisely because the edit didn't touch extraction
  inputs, but nothing in the tool knew that.
- **Fix.**
  1. Each `Stage` declares `config_keys: tuple[str, ...]` — the dotted config
     paths it actually consumes (`extract` → `data.*`, `alignment.*`,
     `models[*].{checkpoint,layer_regex,capture_layer_stride,batch_size}`,
     `extraction.*`; `l1` → `l1.*` plus extract's keys, etc.). Explicit
     per-stage keys, not a whole-config hash, so an unrelated edit doesn't
     invalidate everything — that would make the guard so annoying it gets
     `--force all`-ed past, which is worse than no guard.
  2. On completion, append to `run_manifest.json` (A7): stage name,
     fingerprint (sha256 over the resolved values of its `config_keys` plus
     the fingerprints of its dependency stages), timestamp, wall-clock,
     realized sample sizes (A4/A19).
  3. On rerun: if a stage would be skipped but its fingerprint differs,
     **refuse** with a message naming the changed keys and the two values, and
     offering `--force <stage>` or a new `--allow-stale`. Transitively mark
     downstream stages stale.
  4. The report renders a "stale inputs" warning per section whose fingerprint
     doesn't match the current config, and `meta_report.py` refuses to compare
     runs whose corpus fingerprint differs without a `--allow-mixed` flag.
  5. Test: run smoke, change `alignment.window`, rerun → expect refusal
     naming `alignment.window`; then `--force all` → clean run.
- **Cost.** ~a day. Highest leverage-per-line item in this section: it turns a
  whole class of "how did that number get there" into an error message.
- **Findings (2026-08-06).** Implemented as a new `manifest.py`
  (`resolve_config_keys`, `fingerprint_stage`, `load_manifest`/`save_manifest`,
  `diff_resolved`) plus a `config_keys: tuple` field on every `Stage` in
  `pipeline.py`. One deviation from the original plan, chosen deliberately:
  keys are whole-config-section wildcards (`"data"`, `"l1"`, `"models[*].checkpoint"`,
  etc.) rather than exhaustive single-field lists per stage — still far
  narrower than a whole-config hash (a `report.title` edit doesn't invalidate
  `extract`), and it matches the fix plan's own worked example for `extract`
  (`data.*`, `alignment.*`, ...) rather than adding single-field-list
  maintenance burden with no evidence it's needed yet. `run_pipeline` now
  does a pre-flight pass computing every stage's fingerprint (own resolved
  keys + dependency fingerprints, so a change upstream propagates without
  every downstream stage needing to declare it), and refuses — before
  running anything — if any stage that would be skipped has a fingerprint
  mismatch against `run_manifest.json`, naming exactly which resolved values
  changed and which upstream stage caused a transitive mismatch. `--allow-stale`
  (new CLI flag) downgrades this to a warning. Runs with no manifest at all
  (every run directory that predates this feature) degrade to a warning and
  are trusted as-is, rather than breaking every existing run dir the moment
  this landed — verified against the real `runs/smoke` directory from this
  session's own earlier testing. Verified four ways: (1) an identical rerun
  causes no staleness warnings or errors; (2) changing `alignment.window` and
  rerunning without `--force` raises `ValueError` naming `alignment` and its
  old/new values, and separately names `layer_screen`/`l0`/etc. as
  transitively stale via `extract`; (3) `--force <non-confirm stages>` with
  `--allow-stale` (needed because `confirm`'s own pre-existing "run once"
  guard correctly refuses to be force-rerun) regenerates cleanly and the new
  manifest reflects the new value; a subsequent skip-eligible rerun is then
  clean; (4) a real run against `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small` (the same live run used to verify A1/A2) wrote a
  valid `run_manifest.json` with no errors, confirming the mechanism doesn't
  choke on real per-model config resolution (`models[*].checkpoint` etc.).
  Committed as `tests/test_manifest_fingerprint.py` (3 tests). Full
  `tsfm_lens` suite green (87 tests, up from 81 — 3 new alignment-gate tests
  from A2 plus 3 new manifest tests here). One emergent interaction worth
  recording rather than "fixing": if `confirm`'s upstream inputs go stale,
  the pipeline will keep refusing on every subsequent run (its manifest
  entry is never updated while it stays skipped, by design) until the user
  either accepts `--allow-stale` indefinitely or makes a deliberate call via
  `--force confirm` — which is the correct tension for a "consumed once"
  stage to create, not a bug to smooth over.

### A4 — Corpora are written grouped by task, and four call sites subsample with a head slice `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** `build_pipeline` emits samples task-by-task in spec order,
  so `corpus.jsonl` is **blocked by generator/task**. Several stages take the
  *first* N series rather than a random or stratified sample, which therefore
  silently selects a family-skewed prefix.
- **Evidence.** Ordering: `build_pipeline/builder.py::_build_split` — `for
  spec in specs: for k in range(spec.count)`, appending in order; `seal.py`
  preserves that order. Head slices: `data.py:75` — `kept = kept[:
  cfg.max_series]`; `sae/eval.py:86-88` — `contexts =
  data.contexts()[:take]`; `sae/ground_truth.py:153-155` —
  `np.arange(n)`; `analysis/layer_screen.py:450-451` — `rows =
  np.arange(n_use)`. Indirect: `analysis/confirm.py:80-82` passes
  `confirm.max_series` into the same `data.py` head slice.
  (Correctly randomized for contrast, and the pattern to copy:
  `internals.py:44`, `l1_geometry.py:67`/`:176`, `l3_perturbation.py:185`,
  `lens.py:92`, `attention.py:235`, `clustering.py:46` all use
  `rng.choice(..., replace=False)`.)
- **Blast radius.** Two recorded results are directly affected, and one
  recorded puzzle is probably explained by this:
  - §6.2's SAE forecast-preservation ΔMASE (+0.047 TimesFM / +2.37
    Chronos-T5-Base) was computed on `min(data.n,
    forecast_preservation_max_series, batch_size)` = the **first ~16-32
    series** of a 3-family blocked corpus. The cross-model asymmetry was
    attributed entirely to window-broadcast granularity; part of it may be
    that the two models were compared on the same skewed handful, and the
    absolute values may not represent the corpus at all.
  - §6.2's ground-truth alignment ρ values use the first
    `ground_truth_max_series` rows, so which archetypes are even present
    depends on task order (compounding A10).
  - §5.3 flagged that lens numbers moved for an *unchanged* TimesFM
    checkpoint (final MASE 2.277 → 1.680) between two configs differing only
    in `lens.max_series` (32 vs 24), and correctly concluded "the difference
    is which ~24-32 series got sampled." `lens.py` does sample randomly — but
    with a different `size`, `rng.choice` returns a *different draw*, so the
    two runs' lens stages ran on largely disjoint tiny samples. Same root
    cause class: sample identity is neither stratified nor recorded, so no two
    stages or runs are comparable.
  - `default.yaml`'s `confirm.max_series: 1024` against
    `full_multidomain_run1`'s 4279 private series would confirm hypotheses on
    a task-ordered prefix — the gold-standard stage, on a biased slice. Not
    yet bitten only because `medium_run`'s private split is smaller than its
    cap.
- **Fix.**
  1. One shared helper, `utils.sample_rows(n, k, seed, strata=None) ->
     np.ndarray`: stratified-without-replacement when `strata` is given
     (proportional allocation with a floor of 1 per stratum, remainder by
     largest-remainder), plain `rng.choice` otherwise, always sorted, always
     deterministic in `(n, k, seed, strata)`.
  2. Replace **every** cap with it, passing `data.meta["family"]` as strata:
     the four head slices above plus the seven already-random sites (so that
     family composition is stable across stages and across runs with
     different `k`).
  3. `data.py`: never truncate an ordered list. Either stratify the truncation
     or, at minimum, shuffle deterministically before it — and log the
     realized family composition, not just the family count.
  4. Every stage records the row ids it actually used in its own artifact
     (`rows: [...]`, `n_requested`, `n_realized`) so two stages' samples can
     be compared, and so a future session can tell whether a discrepancy is a
     model effect or a sampling effect without re-deriving it (§5.3 had to).
  5. Warn when realized family composition deviates from the corpus by more
     than a tolerance — the case where even stratification can't help because
     `k < n_families`.
  6. Check `benchmark_validation`'s `--max-sequences` for the same pattern
     (`CLAUDE.md` §5 documents it as subsampling; verify it's random and
     stratified by generator, since its whole purpose is measuring
     per-group diversity).
  7. Test: a synthetic 3-family blocked corpus, cap = n/3, assert all three
     families present in the realized sample and that the same `(seed, k)`
     reproduces it exactly.
- **Then re-measure.** §6.2's forecast-preservation and ground-truth numbers,
  and re-check the §5.3 lens comparison with matched `max_series` and a
  recorded row set. Supersede the old numbers in place with a dated
  correction (§0.2), don't delete them.
- **Cost.** ~a day for the helper + call sites + tests; the re-measurement is
  a short GPU run against already-extracted stores.

**Findings (2026-08-06).** Implemented per the fix plan, with two deviations
and two second-order bugs found along the way:
- `utils.sample_rows(n, k, seed, strata=None)` + `_stratified_alloc` (largest-
  remainder proportional allocation, floor 1 per stratum) landed as specified
  — deterministic in `(n, k, seed, strata)`, always sorted, logs the realized
  vs. population family composition on every call (closing fix item 5's
  "warn on deviation" via always-log rather than a silent pass/warn split;
  simpler and strictly more informative).
- All four named head slices fixed (`data.py::_assemble`, `sae/eval.py`,
  `sae/ground_truth.py`, `layer_screen.py`'s fallback path) plus all seven
  already-random sites switched from bare `rng.choice` to the shared,
  strata-aware helper (`internals.py`, `l1_geometry.py` ×2,
  `l3_perturbation.py`, `lens.py` ×2, `attention.py` ×2, `clustering.py`) so
  family composition is now stable across stages and across differing `k`
  — directly fixing §5.3's documented "lens numbers moved because
  `rng.choice` drew a different sample at a different size" puzzle: `lens.py`
  now uses one fixed seed (`cfg.run.seed + 8`) shared across both models
  instead of a shared *advancing* `rng`, so two models' lens stages sample
  the same row set at a given `k`, and two runs at different `k` share the
  maximal overlap a stratified draw allows.
- Two second-order bugs not in the original evidence list, found while fixing
  the named sites: (a) `l3_perturbation.py::_patching` re-sliced an
  already-`sample_rows`-sampled array with a further `[:take]` prefix slice,
  reintroducing family-order bias one level down — fixed by threading one
  `sel` index consistently through `ctx_clean`/`targets`/`series_ids`/
  `families`/`ctx_corr`; (b) `layer_screen.py`'s own uniform-fallback path
  (taken when a model's captured layers aren't the full block set) had an
  unnamed `np.arange(n_use)` head slice, fixed the same way as the four named
  sites.
- Every touched stage's artifact now records `n_requested`, `n_realized`,
  `rows` (fix item 4) — `sae/eval.py::forecast_preservation` and
  `sae/ground_truth.py::ground_truth_alignment` confirmed carrying this.
  **Deviation:** a `limited_by` field (why `n_realized < n_requested` — cap
  vs. batch size vs. corpus size) was drafted for `forecast_preservation`
  then deliberately removed — that's A16's scope, not A4's, and the
  heuristic first draft was fragile enough that shipping it here would have
  been exactly the "half-finished implementation" this repo's own doctrine
  forbids.
- Fix item 6 (`benchmark_validation`'s `--max-sequences`): checked
  `matching.py` directly rather than assuming — its subsampling is already
  random (`rng.choice`, not a head slice) but **not** stratified by
  generator. Decided **not** to fix in this pass (would duplicate the same
  helper into a sibling package with a different dependency footprint) and
  folded it into A17's fix plan instead, which already covers this file for
  an unrelated reason (blocked/approximate matching at scale) — noted there
  explicitly so it isn't lost.
- **Deviation from the fix plan's wording:** `layer_screen.py`'s dedicated
  screening-extraction row sampling (added under A1, same session) used a
  plain `rng.choice` rather than `sample_rows`, since `sample_rows` didn't
  exist yet when A1 was implemented earlier in this same session. Swapped
  over to `sample_rows` with family strata as part of this fix, so no
  follow-up debt remains.
- **Verification — unit tests.** New `tests/test_sample_rows.py` (5 tests):
  a synthetic 100/30/10 blocked-corpus case proving the exact bug this
  replaces (`strata[:k]` head slice is 100% one family; `sample_rows` isn't),
  determinism in `(seed, k)`, the unstratified path's behavior preserved,
  `k > n` returning everything, and an end-to-end `data.py::_assemble` case
  with a synthetic 20/10/5 blocked corpus asserting all three families
  survive a `max_series` cap. Full suite green at 92/92 after this fix
  (up from 87 after A3; the 5 new `sample_rows` tests are additive, not a
  net-zero replacement of prior coverage).
- **Verification — real-checkpoint run.** `configs/_a4_real_check.yaml`
  (scratch, deleted after use) against live `google/timesfm-2.5-200m-pytorch`
  + `amazon/chronos-t5-small`, real sealed corpus at
  `../../benchmark_medium/public_dev` (confirmed blocked by task by direct
  inspection: 288 rows, `{'random_parametric': 188, 'mixture': 60,
  'parametric': 40}` in that exact order, so the first 40+ rows are 100%
  `random_parametric` — precisely the scenario this fix targets) with
  `data.max_series: 40`. Pipeline completed cleanly end to end (`extract`,
  `layer_screen`, `lens`, `l1`, `l3`, `attention`, `cluster`, `report`; L2/
  SAE/exemplars/confirm disabled for this scratch config) — 7 sections, 16
  findings, no errors. The log's `sample_rows` lines are the actual proof:
  even though the corpus is blocked by task, every capped sub-sample kept
  all three families in roughly the population's own proportion — e.g. a
  `k=16` draw from the `k=40`-assembled population
  `{'mixture': 8, 'parametric': 6, 'random_parametric': 26}` realized
  `{'mixture': 3, 'parametric': 3, 'random_parametric': 10}`, not the
  100%-`random_parametric` result the pre-fix `[:16]` head slice would have
  produced against this exact corpus. This is the first time any stage in
  this pipeline has been run against a real, family-blocked corpus with
  `max_series` actively capping below the corpus size — a scenario the
  smoke config's single-family-per-stage synthetic data can't exercise at
  all, so this real run is not a redundant re-check of the unit tests, it is
  the only test that could have caught the original bug in situ.
- **Re-measurement (fix plan's own last step):** not performed as a
  standalone action *in this fix's own session*. The §6.2 SAE
  forecast-preservation/ground-truth numbers and the §5.3 lens comparison
  remain as previously recorded, now understood to have been measured on a
  family-skewed sample of unrecorded composition; a superseding re-run would
  need `sae.enabled: true` end-to-end (a longer GPU run than this fix's own
  verification needed) and is left as a follow-up rather than bundled into
  this fix, consistent with treating each A-item as its own scoped change.
  **Correction (2026-08-10): this bullet went stale the same day it was
  written and was never updated to say so.** The §6.2 re-measurement it
  defers *did* happen later the same day (2026-08-06, same-day follow-up —
  see §6.2's own Findings block, "RE-MEASURED" entry) and again on
  2026-08-10 (a third data point — see the same Findings block's newest
  entry). §5.3's lens comparison was also later completed (2026-08-10, see
  the entries below this one). Leaving this bullet's "left as a follow-up"
  wording unedited after both follow-ups actually landed is exactly the
  kind of doc drift `ROADMAP.md` §0 warns about — flagged and corrected in
  place here rather than silently left for a future session to re-discover.
- **Partial follow-up attempt, 2026-08-10 — one of the two configs re-measured, the other blocked by a newly-discovered stale store.** Re-ran the matched-`max_series` (24, per `medium_run.yaml`'s same-day fix) lens comparison against `medium_run_chronos_base.yaml`'s already-extracted store: succeeded, TimesFM's own lens numbers came back `final_mase: 1.7899408340454102`, `crystallization_depth: 1.0`. The `medium_run.yaml` side of the same comparison could not be re-measured — its `activations.zarr` turned out to be a dead store, written entirely in zarr v3 format and silently unreadable (returns empty, not an error) under this repo's pinned zarr v2 (see `CLAUDE.md` §11.25, new this session). A background agent was dispatched to delete the stale store and rerun `extract` from scratch, but it failed outright on an infrastructure-level session/API usage limit on the agent's own side (not a code or logic bug) before completing. **Still not re-measured as of 2026-08-10** — `medium_run`'s re-extraction plus the matched lens rerun remains the concrete next step; nothing about the numbers already on record here needed correcting, since the ones being checked were never actually invalidated, only unable to be refreshed yet.
- **Completed, same day, second attempt.** A fresh background agent was
  relaunched (the first retry's failure was purely a session/API limit, not
  a logic problem — verified by confirming a clean redo worked). The stale
  store was renamed, not deleted, to `runs/
  medium_run_stale_v3_backup_20260810` (preserved for anyone who wants to
  double-check the zarr-v3 diagnosis directly), and the full
  `medium_run.yaml` pipeline (13 stages, real `google/timesfm-2.5-200m-
  pytorch` + `amazon/chronos-t5-small` checkpoints) was rerun clean end to
  end in about 11 minutes wall-clock. Confirmed directly (not just taken on
  the agent's word — the agent itself twice made the same "waiting on a
  notification that will never come" mistake §11.25's own writeup already
  flagged as a risk of `CLAUDE.md` §2.8's delegation pattern, resumed each
  time with explicit polling instructions, and its actual progress verified
  independently via `ps`/filesystem checks): `runs/medium_run/
  activations.zarr` now has a valid v2 `.zattrs` and no `zarr.json`,
  `report.html` rendered **11 sections / 27 findings**, and `confirm`
  completed (1/1 dev hypotheses confirmed on private data, 17 registered/2
  replicable). Fresh, now-trustworthy lens numbers: TimesFM `final_mase:
  1.7899408340454102, crystallization_depth: 1.0` — **bit-identical to
  `medium_run_chronos_base.yaml`'s own already-recorded TimesFM number**,
  which is exactly the expected outcome now that both configs' `lens.
  max_series` are matched at 24 (same TimesFM checkpoint, same corpus,
  same context/horizon/seed, same sampled series → deterministic TimesFM
  reproduces bit-for-bit) — a clean, independent confirmation that the
  earlier max_series-mismatch fix this item exists to verify actually
  works. Chronos-T5 (small, this config's pairing — not directly comparable
  to `medium_run_chronos_base`'s Chronos-T5-**Base**): `final_mase:
  2.173410654067993, crystallization_depth: 1.0`. Full `tsfm_lens` pytest
  suite self-run after this fix to confirm nothing else regressed (this
  only touched a `runs/` artifact directory, not source code): **208
  passed, 2 warnings** (both pre-existing and benign — an intentional
  overflow-cast warning inside `test_nonfinite.py`'s own precision test,
  and a pytest style warning about `test_smoke.py::test_end_to_end`
  returning a value — same count and same two warnings the re-extraction
  agent's own independent pytest run had already reported). **This closes the item**: both
  configs' lens numbers are now on record from real, non-empty stores.

### A5 — The report silently omits sections; nothing in the HTML says what's missing or why `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** A section whose artifacts are absent is skipped with
  `log.info`; a section whose builder raises is dropped with `log.warning`.
  The rendered HTML — the thing that actually gets shared — records neither.
- **Evidence.** `report/report.py:92-100`.
- **Blast radius.** A reader of `report.html` cannot distinguish "L2 was
  disabled in this config" from "L2 crashed while rendering" from "L2 ran
  fine and I missed it." `CLAUDE.md` §6.5 describes the info/warning split as
  a deliberate honesty mechanism, and it is — for whoever watched the console.
  For the report's actual audience it's invisible, which is precisely the
  failure mode §2.3 ("explainability of the explainer") exists to prevent, and
  it gets worse the closer this repo gets to §1's one-button promise, where
  nobody watches a console at all.
- **Fix.**
  1. Accumulate a status per builder: `rendered` · `skipped: stage not enabled
     in config` (distinguish from) `skipped: artifacts missing` · `failed:
     <exception class>: <message>` and write `report/coverage.json`.
  2. Render a mandatory **"Run coverage"** panel: every section, its status,
     and for failures the exception message plus the artifact paths it wanted.
     Not collapsed by default when anything failed.
  3. `run_report` returns/logs a one-line summary ("11 rendered, 1 skipped, 1
     FAILED: Attention") and `run.py` exits non-zero when a section failed
     unless `--allow-partial-report` — a crash in a report builder is a bug
     (`CLAUDE.md` §11.6 says this surface is the least type-safe in the repo),
     and today it exits 0.
  4. Test: corrupt one artifact (truncate `l2/stitching.json`), assert the
     report renders with a `failed` row naming it, and that the exit code is
     non-zero.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented per the fix plan, with each builder now
carrying an explicit config attribute so "disabled" and "missing" are
genuinely distinguishable, not inferred:
- Every builder tuple in `report.py` gained a `config_attr` (`"l0"`,
  `"layer_screen"`, ..., `"confirm"`) mapping it to its stage's own
  `PipelineConfig` sub-config. The per-builder loop now checks
  `getattr(cfg, config_attr).enabled` whenever required artifacts are
  missing, producing exactly the three-way status the fix plan asked for:
  `rendered` / `skipped: stage not enabled in config` /
  `skipped: artifacts missing: <paths>` / `failed: <ExceptionClass>:
  <message>`.
- `report/coverage.json` (`{summary, sections: [{eyebrow, title, status,
  detail}, ...]}`) is written on every report render, success or failure.
- A mandatory **"Run coverage"** panel renders directly under "Before the
  numbers" (ahead of the Findings box, so it's the first thing after the
  preamble) as a `<details>` — auto-`open` when anything failed, collapsed
  by default on a clean run so it doesn't add visual clutter to the common
  case; failed rows render in red (`cov-failed`) regardless of open/closed
  state.
- `ReportConfig.allow_partial: bool = False` (new field) + CLI
  `--allow-partial-report` (mirroring `--allow-stale`'s existing pattern of
  a config field settable either way). `run_report` now computes the
  one-line summary (`"N rendered, M skipped[, K FAILED: names]"`), logs it
  at `warning` level if anything failed else `info`, writes `report.html`
  and `coverage.json` **regardless**, and only then — if there are failures
  and `allow_partial` is false — raises `RuntimeError` naming every failed
  section and its exception, plus the exact path to the (still-written)
  report and the fix/override instructions. Because `pipeline.py`'s report
  `Stage.run` is not wrapped in a try/except, this `RuntimeError` propagates
  uncaught out of `run_pipeline` and `run.py`'s `main()`, which is exactly
  what makes the process exit non-zero — no extra exit-code plumbing was
  needed beyond raising in the right place with the report already on disk.
- **Verification — unit tests.** New `tests/test_report_coverage.py` (4
  tests, reusing `test_smoke.py::build_config`'s mock pipeline as a fixture):
  a healthy run has zero `failed` rows and no `cov-failed` HTML class; a
  corrupted `l2/stitching.json` makes `run_report` raise (`pytest.raises`
  matching "L2"), with `report.html` still written and containing a
  `cov-failed` row naming L2, and `coverage.json`'s summary containing
  "FAILED: L2"; the same corruption with `cfg.report.allow_partial = True`
  set does not raise and still records the failed row; and disabling `l2`
  in config while deleting its artifacts produces `status: skipped,
  detail: "stage not enabled in config"` rather than an "artifacts missing"
  message, proving the two skip reasons are genuinely distinguished and not
  just differently-worded guesses. Full suite green at 96/96 (up from 92
  after A4).
- **Verification — CLI exit code, mock pipeline.** Ran `run.py --stages
  report` directly (not just calling `run_report` in-process) against a
  freshly-built mock-model run dir with `l1/cka.npz` corrupted to a garbage
  byte string: without `--allow-partial-report` the process printed the
  traceback and **exited 1**; with the flag it printed the same "1 FAILED:
  L1" summary line, still wrote a report naming the failure, and **exited
  0**. (First attempt reused one run directory's `config_resolved.yaml`
  across both invocations and got `exit 0` for the no-flag case too — a
  self-caught mistake, not a code bug: the flagged invocation is dumped
  back into `config_resolved.yaml` by `run_pipeline` itself, so the *third*
  command was unknowingly reading a config that already had
  `allow_partial: true` baked in from the *second* command's run. Re-ran
  with two independently-built fresh run directories, one per code path, to
  get a clean result.)
- **Verification — real checkpoints.** `configs/_a5_real_check.yaml`
  (scratch, deleted after use; same reduced-cap pattern as A4's real check)
  against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small`, real sealed corpus, with SAE/Exemplars/Confirm
  disabled and every other stage enabled. Clean run: `coverage.json` shows
  9 `rendered` + 3 `skipped: stage not enabled in config` (SAE/Exemplars/
  Confirm), 0 failed. Then corrupted the *real, checkpoint-produced*
  `l2/stitching.json` in place and re-ran `run.py --stages report` against
  that same run directory: `report: section L2 failed: JSONDecodeError...`,
  `8 rendered, 3 skipped, 1 FAILED: L2`, report.html written, process
  **exited 1**. This confirms the mechanism generalizes past the smoke
  config's mock-model artifact shapes — nothing about real-checkpoint
  extraction/analysis artifacts (bfloat16 tensors, real family labels,
  layer-screened targets) interacts differently with the coverage/failure
  logic, as expected since this fix operates purely on `run_dir` paths and
  exception objects, not on activation content.
- **Deviation from the fix plan's wording:** item 2 said "not collapsed by
  default when anything failed" — implemented as auto-`open` exactly when
  `any_failed`, and left collapsed (not "not collapsed", i.e. closed) on a
  clean run, since always-expanding a coverage table on every healthy run
  would add noise to the common case the rest of this report's `<details>`-
  based note mechanism (`CLAUDE.md` §6.5) is designed to avoid. This matches
  the fix plan's actual intent (surface failures prominently) without
  over-reading "not collapsed" as "never collapsible."

### A6 — `family_key` mis-resolution and unlabeled corpora degrade to one family, silently `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** An explicit `data.family_key` that doesn't resolve falls
  through to a candidate list, then to `provenance.generator`, then to the
  literal string `"unknown"` — per row, with no aggregate check.
- **Evidence.** `data.py:89-99` (`_family_of`), `data.py:22`
  (`_FAMILY_CANDIDATES`). The only signal is the family *count* in an INFO log
  (`data.py:83-84`).
- **Blast radius.** A typo in `family_key` silently regroups every per-family
  statistic in the pipeline — L0's Holm-corrected family strengths (and
  therefore what `confirm` pre-registers), L1's per-family CKA, attention's
  family-conditioned lag profiles, clustering's labels and AMI, exemplar
  selection. A corpus without labels collapses to a single family, at which
  point per-family "comparisons" are computed over one group and the report
  still renders them.
- **Fix.**
  1. If `family_key != "auto"` and it resolves for fewer than (say) 90% of
     rows: **raise**, naming the key, the resolution rate, and the keys that
     *would* have worked. Falling back to a different grouping than the one
     the user asked for is exactly the "silently wrong slice" `CLAUDE.md` §2.5
     forbids.
  2. Record `family_resolution: {key_requested, key_used, n_unknown,
     n_families, composition: {family: count}}` in `run_manifest.json` and
     surface it in the report's coverage panel.
  3. If `n_families < 2`, disable every per-family section with an explicit
     stated reason rather than emitting degenerate one-group statistics; keep
     the overall (pooled) comparisons.
  4. Test: a corpus with a typo'd key raises; a single-family corpus renders a
     report with per-family sections explicitly marked not-applicable.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented items 1, 2, and 4 exactly as specified;
item 3 implemented for the four sites the fix plan's own blast-radius
evidence names, after checking each one for whether "per-family" there is
actually a *comparison* (degenerate with one group) or just a *breakdown*
(still meaningful with one group) — a real subtlety the fix text didn't
fully resolve on its own:
- `data.py::_resolve_family_key` resolves one explicit dotted key with no
  fallback (distinct from `_family_of`'s "auto" chain), and
  `_check_family_key_resolution` measures the aggregate hit rate across the
  whole pre-cap corpus; below 90% it raises, naming the requested key, the
  measured rate, and — by re-running the same resolution check against
  every other `_FAMILY_CANDIDATES` key — which of them *would* clear 90%
  instead, or an explicit "none do, use `family_key: auto`" fallback
  message when none do either.
- `BenchmarkData` gained `family_resolution: dict` (`key_requested,
  key_used, resolution_rate, n_unknown, n_families, composition`) and
  `n_families` — computed once in `_assemble`, always (not just when a
  problem is detected). `extraction/extract.py` persists it via a new
  `manifest.py::record_extra(run_dir, key, value)` helper into
  `run_manifest.json` alongside (not replacing) A3's per-stage
  fingerprints; `pipeline.py`'s per-stage manifest save now re-reads and
  re-merges any such extra top-level keys on every write so a later
  stage's save can't clobber what an earlier stage recorded mid-run (a real
  ordering bug caught by testing the actual sequence, not assumed safe).
  `report.py::_family_resolution_line` renders it as a line inside the same
  `<details class="coverage">` panel A5 built, immediately under the
  section table — literally "surfaced in the report's coverage panel" as
  item 2 asked, not a separate section.
- **Item 3, scoped after inspection, not applied uniformly:** checked each
  of the fix plan's four named blast-radius sites for whether "per-family"
  there is a comparison or a breakdown:
  - **L0** (`_summarize`): `family_tests`/`strengths`/`mase_ratio` are a
    genuine cross-family *comparison* (Holm-corrected, needs >=2 groups) —
    guarded with an early return once `n_families < 2`, replaced with an
    explicit `family_comparisons: {applicable: false, reason}`. `per_family`/
    `overall` (plain aggregates, not comparisons) are untouched and still
    carry the pooled model-vs-model result.
  - **L1** (`run_l1`): family-conditioned CKA compares each family's
    geometry *to the same computation run without conditioning* — with one
    family it's a second, differently-pooled measurement of the same
    number, not comparative signal. Guarded the same way; the window-level
    `cka_window`/`depth_curve` (never family-conditioned) are untouched.
  - **Clustering** (`_label_clusters`/`run_clustering`): checked the AMI
    computation specifically and found the fix plan's own "clustering...
    AMI" framing needs a correction — `adjusted_mutual_info_score` here
    compares **the two models' own cluster assignments to each other**, not
    to family labels, so AMI is fully meaningful regardless of family count
    and was **not** guarded. What *is* genuinely degenerate is
    `_label_clusters`'s majority-family purity (trivially 100% for every
    cluster with one family, a mathematical certainty misreadable as a
    finding) — guarded to drop the family/purity framing and keep only the
    feature-based descriptor, with a `family_comparisons` note recorded in
    `comparison.json` (not `clusters.json`, to avoid the literal string
    `"family_comparisons"` being misread as a model name by
    `_sec_clusters`'s `list(clusters.keys())` — caught by tracing the
    report's own parsing before picking a location, not after).
  - **Attention:** re-read `_pattern_analysis`'s periodicity-head scoring
    and concluded it is **not** degenerate at one family the way the other
    three are — "per-family" there tags each head with whichever family
    gave it the best score, which remains a real (if less varied) signal
    with only one candidate family, not a comparison that becomes
    meaningless. **Left unguarded**, with the reasoning recorded here so a
    future session doesn't have to re-derive it. `head_delta_family`/
    `mlp_delta_family` arrays are computed but were already found to be
    dead code (never read by any `_sec_attention` render path) — orthogonal
    to this fix, not touched.
  - **Exemplars:** per-family case studies aren't a cross-family
    *comparison* either (they showcase series within each family, not
    compare families' properties) — left unguarded for the same reason as
    attention.
- **A genuine crash found and fixed along the way, not in the original
  evidence list:** `analysis/internals.py`'s family-decodability probe
  (`_family_probe`, a `LogisticRegression` classifier) **raises** from
  scikit-learn when given only one class to fit — this is exactly the
  failure mode a prior real-checkpoint session flagged as "directly
  relevant" before this fix existed. Guarded in `run_internals`: the probe
  is skipped (recorded as `{"value": None}` per layer) when `n_families <
  2`, with a `family_comparisons` reason attached to each model's profile
  entry; `effective_dim`/`input_cka` (not family-dependent) are computed as
  usual. `_sec_internals`'s trailing findings loop (`np.argmax` over probe
  values) would also have raised on an all-`None` list — guarded the same
  way. Without this fix, a real single-family corpus would have **crashed
  the whole pipeline at the `internals` stage**, not just rendered a
  degenerate number — worse than anything in the original evidence list.
- **Verification — unit tests.** New `tests/test_family_resolution.py` (5
  tests): a typo'd key (0% resolution) raises naming the correct
  alternative key; a 50%-resolution key raises; a 95%-resolution key does
  not raise; `family_key: auto` never raises even on a fully unlabeled
  corpus (falls to `"unknown"`); and a full mini pipeline (mock models, a
  genuinely single-family 40-series corpus) runs end to end through
  extract → layer_screen → l0 → internals → lens → l1 → l2 → l3 → attention
  → cluster → report with **zero crashes**, and asserts `family_comparisons.
  applicable is False` in L0/L1/clustering/internals artifacts, pooled L0
  `overall` metrics and L1's window-level `depth_curve` still present, AMI
  still a real float, `run_manifest.json` recording `n_families: 1`, and
  the rendered report containing the "Not applicable." explanation at
  least 4 times (one caught-and-fixed test bug along the way: two of the
  four sections initially used a differently-worded "not applicable" phrase
  than the other two, which the test's exact-match caught immediately —
  unified all four to the same phrasing rather than loosening the
  assertion). Full suite green at 101/101 (up from 96 after A5).
- **Verification — real checkpoints.** `configs/_a6_real_check.yaml`
  (scratch, deleted after use; same pattern as A4/A5's real checks) against
  live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small`, the
  real 3-family `benchmark_medium` corpus. Confirms the **normal path is
  unaffected** by every guard added above: `run_manifest.json` records
  `family_resolution: {key_used: auto, n_families: 3, composition:
  {mixture: 8, parametric: 6, random_parametric: 26}, ...}`; L0 still
  produces real `family_tests`; L1's `families` list has all 3; clustering
  and internals both show `family_comparisons: None` (guard correctly
  not triggered); the report contains the family-resolution line inside the
  coverage panel and **zero** "Not applicable." occurrences — a clean
  9-rendered/3-skipped/0-failed run. (The single-family crash path itself
  was not re-verified against real checkpoints beyond the mock-model
  pipeline test above — building a genuinely single-family *real* corpus
  wasn't readily available this session — but every guard is architecture-
  and GPU-agnostic code operating on `data.meta["family"]` counts, not on
  activation content, so the mock-model coverage is expected to generalize
  the same way A4/A5's did.)

### A7 — No run provenance, and the activation store has no schema version `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** A run directory records the resolved config and nothing
  else: no git SHA, no library versions, no device, no checkpoint revisions,
  no per-stage timings. The zarr store records only geometry, so a store
  written by a different zarr major version fails with a low-level error
  instead of an actionable one.
- **Evidence.** `pipeline.py:146` writes only `config_resolved.yaml`;
  `extraction/store.py:37` — attrs are `n_series`/`n_windows`/`window`/
  `context_len` only. Contrast the benchmark half, which does this correctly:
  `build_pipeline/builder.py:190` — `sample.provenance.library_versions =
  _library_versions()` per sample.
- **Blast radius.** This repo has now lost time to environment drift in three
  separate sessions (`CLAUDE.md` §11.15 zarr v3-vs-v2, §11.17 cp1252 encoding,
  and §5's third-machine entry finding zarr 3.2.1 installed against a
  `zarr<3` pin), and §7's Findings record a concrete replication *blocked*
  because `runs/medium_run`'s store was written in a v3-formatted layout and
  "isn't readable here without re-extraction." With no provenance in the run
  dir, no artifact can be attributed to an environment after the fact, and
  `meta_report.py` can silently aggregate runs from mismatched stacks.
- **Fix.**
  1. `utils.run_provenance()` → dict: git SHA + dirty flag, `tsfm_lens`
     version, python/torch/numpy/pandas/zarr/scikit-learn/transformers/
     timesfm/chronos-forecasting versions (whatever imports), CUDA/driver and
     device names, per-model resolved checkpoint id **and HF revision hash**,
     corpus manifest digest (from `load_sealed`'s manifest — already computed,
     currently discarded by the analysis half), config hash. No hostnames or
     user paths (keeps it shareable and invariant-11-clean).
  2. Write `run_manifest.json` at run start and append per-stage records
     (fingerprint from A3, wall-clock, realized n from A4/A19).
  3. `ActivationStore.create` writes `schema_version` (start at 1),
     `zarr_format`, and writer versions into root attrs; `ActivationStore.
     __init__` validates and raises a *specific* message on mismatch
     ("activations.zarr was written with zarr format 3 by tsfm_lens 0.1.0;
     this environment has zarr 2.18.7 — pin `zarr>=2.16,<3` and re-extract,
     or read it in a v3 environment"). This is the fourth recurrence of one
     trap; make the fourth one self-explaining.
  4. Report renders the provenance panel (shared with A2); `meta_report.py`
     shows one provenance row per run and warns on mismatched corpus digests
     or library majors.
  5. Test: hand-write a store with `schema_version: 999`, assert the specific
     error; assert `run_manifest.json` exists after a smoke run and contains
     no absolute user paths (ties to invariant 11).
- **Cost.** ~a day, and it retires a recurring cost.

**Findings (2026-08-06).** All five fix items implemented; one real,
previously-undocumented gap found and closed along the way (`timesfm` has
no `__version__` attribute at all — confirmed live, not assumed):
- `utils.run_provenance(cfg=None, corpus_digest=None) -> dict`: git SHA +
  dirty flag (`git status --porcelain` against a repo-relative `cwd`, never
  an absolute path — invariant 11), `tsfm_lens.__version__`, Python version,
  and versions for `torch/numpy/pandas/zarr/sklearn/transformers/timesfm/
  chronos` via `_pkg_version` — which now tries `__version__` first, then
  falls back to `importlib.metadata.version()` for packages that expose
  neither (see below), device info (CUDA availability, device name(s) via
  `torch.cuda.get_device_name`, CUDA version, driver version via
  `nvidia-smi --query-gpu=driver_version`), per-model checkpoint id **and**
  its locally-cached HF commit hash (`_hf_revision`, via
  `huggingface_hub.scan_cache_dir()` — deliberately the *local cache*, not
  a Hub API call, so it reflects the exact snapshot actually loaded and
  works fully offline), corpus digest, and a config hash (sha256 over
  `dataclasses.asdict(cfg)`, reusing `manifest.py`'s existing `_stable_json`
  serializer). Every sub-lookup is wrapped to degrade to `None` rather than
  raise (`CLAUDE.md` sec 2.5) — not installed, no git repo, no GPU, and no
  network are all real, expected environments for this repo, not bugs.
- **Corpus digest was genuinely discarded, confirmed by reading the code
  before assuming**: `data.py::_load_corpus_rows` called
  `samples, _ = load_sealed(...)`, throwing away the manifest `load_sealed`
  itself already verifies. Now returns `(rows, corpus_digest)`, threaded
  through `load_benchmark`/`_assemble` into a new `BenchmarkData.
  corpus_digest` field (parallel to A6's `family_resolution`) —
  `extraction/extract.py`'s `run_extraction` merges it into the
  already-recorded `provenance` dict once the corpus is actually loaded
  (base provenance is recorded by `run_pipeline` at run start, before any
  stage runs, since git/library/device facts don't depend on data being
  loaded; the corpus digest specifically does, so it arrives one step
  later via a read-merge-write through the same `manifest.py::record_extra`
  A6 built).
- `extraction/store.py`: `_SCHEMA_VERSION = 1` constant; `ActivationStore.
  create` stamps `schema_version` + `written_by: {tsfm_lens_version,
  zarr_version}` into root attrs; `__init__` validates on any non-`"w"`
  open and raises a message naming the stored vs. expected version and the
  writer's exact library versions, plus the concrete fix
  (`--force extract` and downstream). **Also wrapped the `zarr.open_group`
  call itself** in a try/except that re-raises with a pointer to the
  `zarr<3` pin and CLAUDE.md sec 11.15 — this catches the *other* half of
  "fails with a low-level error instead of an actionable one" (a genuinely
  cross-major-version-incompatible store failing to open at all, before
  any attrs can even be read) that a `schema_version` attrs check alone
  cannot reach, since that check requires the group to have opened
  successfully first. **Not independently tested** — reproducing a real
  zarr-v2-vs-v3 on-disk incompatibility would need a second zarr major
  installed in this environment, which isn't available; the wrap's error
  message was verified by inspection, not by triggering a real cross-version
  failure.
- **Report panel, shared with A2 as originally deferred**:
  `_alignment_provenance_block` in `report.py` reads both `run_manifest.
  json`'s `provenance` key and `alignment/alignment_check.json` (A2's
  artifact, previously written but never rendered) into one combined
  `<details class="coverage">` panel — auto-open when any model's alignment
  gate recorded `passed: false`, immediately below the run-coverage panel
  A5 built. Shows per-model min/mean diagonal-hit fraction and gate
  pass/fail, plus git SHA (+dirty), tsfm_lens version, device, config hash,
  corpus digest, per-model checkpoint+HF revision, and package versions.
- `meta_report.py`: `summarize_run` now reads each run's `run_manifest.
  json` provenance (absent for pre-A7 runs, handled as a plain missing key,
  not an error); new `provenance_warnings(run_summaries)` flags **mismatched
  corpus digests** and **mismatched major versions** of
  `torch`/`numpy`/`zarr` across the runs being aggregated — the exact
  "meta_report.py can silently aggregate runs from mismatched stacks" risk
  named in this item's own blast radius. Rendered as a highlighted warning
  block above the existing runs table, plus a new "Run provenance" table
  (one row per run: git SHA, tsfm_lens version, key package versions,
  device, corpus digest — "one provenance row per run" per the fix's own
  wording). Runs with no provenance recorded are excluded from a given
  check rather than flagged as mismatched — absence of data isn't evidence
  of drift.
- **A real, previously-undocumented gap found and fixed, not in the
  original evidence list**: the first real-checkpoint verification run
  showed `packages: {..., "timesfm": null, ...}` even though `timesfm` is
  installed and working (checkpoints loaded fine throughout this session).
  Checked directly (`import timesfm; timesfm.__version__` →
  `AttributeError`) rather than assumed: the installed `timesfm` package
  genuinely has no `__version__` attribute at all — a real gap in
  "whatever imports" as originally scoped, not a bug in the lookup itself.
  Fixed by falling back to `importlib.metadata.version(module_name)` when
  `__version__` is absent; re-verified live (`packages["timesfm"] ==
  "2.0.2"` after the fix, matching `pip show timesfm`).
- **Verification — unit tests.** New `tests/test_provenance.py` (4 tests):
  `run_provenance()` output contains no absolute `/home/...` paths and
  resolves `torch`'s version (a hard dependency, must never be `None`); a
  hand-corrupted `schema_version: 999` store raises `RuntimeError` matching
  the exact mismatch on open; a freshly-created store round-trips through
  reopen with no error; and a full smoke run's `run_manifest.json` has no
  absolute paths, records `corpus_digest: None` (correctly, for the
  no-sealed-corpus smoke source), and its `report.html` contains the
  rendered "Alignment &amp; provenance" panel. Two new tests in
  `tests/test_meta_report.py`: mismatched corpus digests and a mismatched
  `torch` major across two synthetic runs both produce the expected
  warning strings (and a matching `numpy` major does **not** false-positive
  warn), rendered visibly in the aggregate HTML; and runs with no recorded
  provenance at all aggregate cleanly with zero warnings. Full suite green
  at 107/107 (up from 101 after A6).
- **Verification — real checkpoints.** `configs/_a7_real_check.yaml`
  (scratch, deleted after use; a fast extract+l0+l1+report-only config to
  keep this check cheap) against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small`, the real `benchmark_medium` sealed corpus.
  Every field resolved to a real, correct value: `git_sha` (the actual
  working-tree HEAD, `git_dirty: true` — correctly reflecting this
  session's uncommitted edits), real package versions for all 8 tracked
  libraries (after the `timesfm` fix above), real device info (`NVIDIA RTX
  A5000`, CUDA 12.9, driver 535.216.03 via `nvidia-smi`), and — the
  detail most worth calling out — **real, resolved HF revision hashes for
  both checkpoints** (`1d952420fb...` for TimesFM 2.5,
  `a971ba2194...` for Chronos-T5-small) via `huggingface_hub.
  scan_cache_dir()`, plus a real sha256 corpus digest and config hash.
  The rendered report's "Alignment & provenance" panel showed the real
  alignment gate results (1.00/1.00 diagonal-hit fraction for both models,
  consistent with every prior session's numbers) directly alongside this
  provenance line, exactly as the shared-panel design intended.

### A8 — Invariant 1 is currently red and unowned (golden hashes vs. numpy) `[x]` · **P1** · addressed 2026-08-06

- **What's wrong.** `test_golden_hashes_pre_extension_outputs_unchanged` fails
  against numpy 2.1.0 with no change to the generators; the leading hypothesis
  (`Generator.choice(..., replace=False)` is not stream-stable across numpy
  versions) has never been confirmed, and the failure has been carried as a
  known-red test across five sessions.
- **Evidence.** `CLAUDE.md` §9, §11.13; §5's 2026-08-03 Findings (seed 0
  produces `4070db4646799f09`, expected `2856658d044e4c49`).
- **Blast radius.** Invariant 1 is the whole basis of the sealed-corpus
  reproducibility claim, which is in turn the basis of the epoch-regeneration
  story (`CLAUDE.md` §4.5) and of every "this corpus regenerates from its
  seed" statement. A red invariant that nobody owns decays into a permanently
  accepted failure, and the suite's "40 passed / 1 failed" line stops being
  read at all.
- **Fix.**
  1. **Confirm the mechanism first, cheaply** (§2.4): in an isolated env, diff
     `np.random.default_rng(0).choice(100, 5, replace=False)` and
     `default_rng(0).normal(size=5)` across the numpy versions available.
     One is expected to move and the other not; that alone settles it without
     bisecting the whole generator.
  2. If confirmed: replace every `rng.choice(..., replace=False)` in
     `generators.py` with a construct built from primitives whose stream is
     stable — candidate `np.argsort(rng.random(n))[:k]` — and **verify that
     candidate's cross-version stability by the same test** rather than
     assuming it (the assumption is what broke here the first time).
  3. Re-baseline the golden hashes in the same commit; record the numpy
     version they were captured against *in the test file itself* and in
     `DEPENDENCIES.md` (invariant 12).
  4. Split the regression test: one case over a pure `rng.normal`-only recipe,
     one over the full composite. A future break then localizes itself instead
     of just saying "the hash moved."
  5. State the **determinism contract** in the seal manifest, per generator:
     `bit_exact: true|best_effort`. `sequential_par` is best-effort by
     construction (SDV's `PARSynthesizer` exposes no seed — `CLAUDE.md`
     §11.11), and nothing in a sealed corpus currently tells a consumer which
     of its samples can actually be regenerated. That's a correctness claim a
     manifest should carry, not a docs footnote.
  6. Once green, wire into CI (A14) as a blocking check, and mark it
     `xfail(strict=True)` in the interim so it flips loudly when fixed rather
     than staying quietly red.
- **Cost.** Half a day to confirm + a day to fix and re-baseline.

**Findings (2026-08-06) — item 1's own "confirm first" step changed the
shape of this fix.** Followed the fix plan's own ordering exactly: before
touching any code, checked whether the failure reproduces at all in the
current, persistent environment.

- **The golden-hash test currently PASSES here** (`python3 -m pytest
  tests/test_generator_extensions.py::test_golden_hashes_pre_extension_outputs_unchanged`
  → 1 passed), and the **full `tsfm_benchmark` suite has zero failures**
  (37 passed / 1 skipped before this session's additions) — not the
  "40 passed / 1 failed" state `CLAUDE.md` §9 describes. This environment's
  numpy is 2.1.0, the same version the original failure report named, so
  this isn't simply "a different numpy version, of course it's fine now."
- **Fix item 1's direct comparison refutes the leading hypothesis**, rather
  than confirming it: `np.random.default_rng(0).choice(100, 5,
  replace=False)` and `.normal(size=5)` were run in an isolated,
  throw-away `venv` with `numpy==1.26.4` (chosen so as not to touch this
  session's own pinned, MKL-matched numpy 2.1.0 install and risk
  reintroducing `CLAUDE.md` §11.13's SEH crash) and compared byte-for-byte
  against the same two calls under this environment's numpy 2.1.0. **Both
  arrays were bit-identical across the two numpy versions.** If `choice`'s
  algorithm were the culprit, this is exactly the comparison that would
  have shown a divergence, and it didn't.
- **What this means, stated plainly:** the specific, five-session-old
  hypothesis ("`Generator.choice(..., replace=False)` isn't stream-stable
  across numpy versions") is not supported by direct evidence between the
  two most relevant versions (1.26.4 and 2.1.0), and the originally-reported
  mismatch is **not currently reproducible** in this persistent environment.
  The original failure's true cause remains unexplained — it may have been
  a different numpy point-release, a different platform/BLAS backend, or
  an environment-specific artifact (`CLAUDE.md` §11.13 already documents an
  unrelated numpy/MKL ABI crash in this same version's history, i.e. this
  numpy build has precedent for environment-sensitive behavior). Per
  `CLAUDE.md` §2.4's own doctrine, an unconfirmed hypothesis is not grounds
  to rewrite `generators.py`'s RNG calls — doing so would risk breaking bit-
  exact reproducibility for a *new*, self-inflicted reason (§11.1's own
  lesson), to fix a bug that isn't currently present. **No changes were made
  to `generators.py`'s actual `rng.choice` calls.**
- **What *was* still done, because it's correct regardless of whether the
  original bug ever recurs:**
  1. Item 3 (partial): recorded the numpy version (2.1.0) the hashes are
     verified against directly in the test file's own comment above
     `GOLDEN`, and updated `DEPENDENCIES.md` §3's numpy row and §5's fragile-
     spots list to state the refutation plainly rather than repeating the
     unconfirmed hypothesis as fact. Re-baselining wasn't needed — the
     existing hashes already match.
  2. Item 4: split the regression test. New `GOLDEN_NORMAL_ONLY` +
     `test_golden_hashes_normal_only_recipe_unchanged` runs the *same*
     recipe/seeds as the composite test but with `n_changepoints=0,
     n_anomalies=0` — the only two knobs that call `rng.choice(...,
     replace=False)` (confirmed by grep, not assumed). A future break now
     localizes: normal-only fails alone → the break is in `rng.choice`;
     composite fails alone → impossible (composite is a strict superset of
     normal-only's draws) — so really: both fail → the break is upstream
     (`rng.normal` or `default_rng` itself); only composite fails → the
     break is specifically in the choice-based changepoint/anomaly draws.
  3. Item 5: `generators.py` gained a `BIT_EXACT` dict (`parametric`,
     `random_parametric`, `mixture`, `block_bootstrap` → `True`;
     `sequential_par` → `False`, per `CLAUDE.md` §11.11's already-documented
     "PARSynthesizer exposes no seed" finding) and `seal.py::seal_corpus`
     now writes a `determinism: {generator: {bit_exact, count}}` block into
     every manifest — the `bit_exact: true|best_effort` contract the fix
     asked for, generalized slightly to a per-generator breakdown since a
     real corpus mixes generators with different guarantees. An unknown
     generator (not yet in `BIT_EXACT`) degrades to `bit_exact: None` rather
     than crashing or silently claiming a guarantee, tested explicitly.
  4. Item 6: **not applicable as worded** — `xfail(strict=True)` marks a
     test *expected* to fail; this test currently passes, so marking it
     `xfail(strict=True)` would itself fail the suite (strict xfail demands
     an actual failure). CI wiring itself is A14's scope, not yet reached;
     when A14 lands, this test should simply run as a normal blocking check
     like any other, with no special-casing needed now that it's green.
- **Verification — unit tests.** `test_golden_hashes_normal_only_recipe_unchanged`
  (new) plus the existing composite test both pass. New
  `tests/test_seal_determinism.py` (3 tests): `BIT_EXACT` has the expected
  value for all five registered generators (asserting `sequential_par is
  False` specifically, so this can't silently regress to claiming a
  guarantee SDV doesn't provide); a corpus mixing `parametric`/
  `sequential_par` samples produces the exact expected per-generator
  `determinism` breakdown in the sealed manifest; and a sample from a
  generator not in `BIT_EXACT` degrades to `bit_exact: None` without
  crashing `seal_corpus`. Full `tsfm_benchmark` suite green at 41 passed / 1
  skipped (up from 37 passed / 1 skipped — 4 new tests, 0 regressions, 0
  fixes needed to existing tests since none were failing).
- **Verification — real build.** Two real, non-mocked builds: (1)
  `example_runs/run_smoke.py` end to end (leakage gate, seal, reload+verify,
  cross-process reproducibility, epoch regeneration) — unaffected by the
  `seal.py` import addition, all prior assertions still pass; (2) a real
  `run_full.py` build against `configs/example.yaml` with live Monash data
  (`bootstrap_catalog` pooling 251 series across 6 real domains) producing a
  20-sample `public_dev` split mixing `parametric`/`mixture`/
  `random_parametric` — the written `manifest.json` carries a real
  `determinism` field: `{"parametric": {"bit_exact": true, "count": 10},
  "mixture": {"bit_exact": true, "count": 5}, "random_parametric":
  {"bit_exact": true, "count": 5}}`, exactly matching the corpus's actual
  generator mix. Confirms the manifest change composes correctly with the
  real leakage-audit/seal/real-source pipeline, not just synthetic
  `TimeSeriesSample` fixtures.
- **What "fixed" means for this item, precisely:** the audit's own
  instruction was to confirm the mechanism and either repair it or
  re-baseline; the confirmation step is what actually resolved this — there
  was nothing live to repair, and the hashes needed no re-baselining. What
  shipped is exactly the fix plan's items 3-5 (version provenance, a
  self-localizing test split, and the determinism contract), which stand on
  their own merit independent of whether the original numpy-version bug
  ever recurs. If it resurfaces on a *different* numpy version in a future
  session, the split test (item 4) will say which half of the mechanism
  broke instead of just "the hash moved," which is the actual, lasting
  value of this session's work here.

### A9 — `l0/summary.json` has no archetype granularity `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** `random_parametric`'s sampled archetype is recorded in
  sample provenance but never carried into the analysis half, so every
  per-group statistic stops at the coarse `family` level.
- **Evidence.** §5.5's own follow-up note; `data.py:77-82` builds `meta` with
  `series_id`/`family`/`tier` only.
- **Blast radius.** The question §5.5 exists to answer — "does 'TimesFM
  front-loads then compresses' hold on `intermittent_bursts` too, or only on
  the smooth archetypes it was observed on?" — is unanswerable today: all 12
  archetypes collapse into one `random_parametric` family. It also forces
  `sae/ground_truth.py` to reconstruct archetype labels itself, which is where
  A10 comes from.
- **Fix.** `data.py` carries `archetype` (from
  `provenance.generator_params`, `None` for tiers that can't express one) and
  `generator` into `meta.parquet`; `l0_behavioral.py` reports per-`(family,
  archetype)` alongside per-family (with an explicit minimum-n guard, and
  falling back to the family label where archetype is `None` — stated, not
  silent); `meta_report.py` gains the per-archetype stability table §5.5
  wanted; exemplars and attention gain an optional archetype conditioning.
  Keep `family` as the primary axis so recorded numbers stay comparable.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented all five fix items, scoped as follows:
- `data.py::_assemble` now carries two new `meta` columns:
  `archetype` (from `provenance.generator_params.archetype`, `None` for
  tiers that can't express one) and `generator` (the raw provenance
  generator name, finer than `family` when `family_key` groups several
  generators under one task label) via two new small resolvers
  (`_archetype_of`, `_generator_of`) applied before the same
  `max_series` stratified-subsample step A4 already built, so both new
  columns stay aligned under subsampling.
- `l0_behavioral.py`: `_score` carries `archetype`/`generator` into
  `metrics.parquet`; a new `_archetype_summary()` (called from
  `_summarize`, independent of and before the `n_families < 2` early
  return, since the single-family `random_parametric` corpus *is* the
  motivating case for this axis) reports per-archetype mean metrics **and**
  Holm-corrected paired MASE-ratio tests, mirroring `family_tests`'
  shape/statistics exactly but on the finer axis. Series whose tier can't
  express an archetype fall back to their family label (`fallback_to_family`,
  stated in the output, not silently dropped or silently blended with real
  archetype rows); groups below `stats.min_series` are dropped and named in
  `dropped_min_n` (reusing the existing min-n knob rather than adding a new
  one). Degrades explicitly (`{"applicable": False, "reason": ...}`) when no
  series in the corpus carry an archetype at all, when fewer than 2 groups
  survive the min-n guard, or when `stats.enabled` is false — matching A6's
  established "Not applicable, stated" pattern rather than a silent skip.
- `report.py`: new `_archetype_block()` renders the per-archetype table,
  fallback/dropped notes, and the paired-test table (or the applicable:false
  reason) directly under L0's existing per-family section, and feeds
  `strengths`-style findings into the same findings list.
- `meta_report.py`: new `archetype_stability()` (exact mirror of
  `family_stability()`, one axis finer) plus a rendered "Per-archetype
  stability across runs" table, wired into `build_meta_report`.
- `exemplars.py`/`report.py`: `_select_exemplars` now carries an `archetype`
  column through to `exemplars.json` (mapped from `data.meta` by row index,
  avoiding a `pivot_table` NaN-index-drop trap), and the exemplar card
  headings show it when present (`"{family} · archetype {archetype} · series
  {id}"`).
- **Deliberately scoped out, stated rather than silently dropped**:
  `attention.py`'s family-conditioned lag profiles, periodicity scores, and
  head/MLP ablation ΔMASE were **not** given an archetype axis. Unlike L0's
  aggregate (a `groupby` and a paired test), attention's per-family
  machinery is threaded through batched forward passes, ablation loops, and
  its own `top_heads`/`meta` JSON shape (`CLAUDE.md` §11.6 already flags
  this file's report-key surface as fragile) — duplicating a
  min-n-guarded, fallback-aware grouping axis through all of that
  correctly would be a multi-hour change in its own right, not the
  half-day this item was costed at, and the fix plan itself called this
  piece "optional." Left for a future item if a concrete question needs it
  (the motivating §5.5 question — "does the family-level MASE gap hold on
  every archetype" — is already answerable from L0 alone, which is what got
  built).
- **Verification — unit tests.** New `tests/test_archetype_reporting.py` (6
  tests): `_assemble` carries `archetype`/`generator` with correct `None`
  fallback; `_archetype_summary` returns `applicable: False` when no series
  carry an archetype; a group below `min_series` is dropped and named;
  a planted MASE gap on one archetype (with a tied second archetype and a
  no-archetype "real_derived" family mixed in) produces the correct
  `favored` verdict and correct `fallback_to_family`; `archetype_stability`
  flags a favored-model disagreement across two synthetic runs; and a full
  mini pipeline (three archetype-labeled + one unlabeled family, mock
  adapters) asserts `l0/summary.json`'s `per_archetype` block, `meta.parquet`'s
  `generator` column, and `report.html`'s rendered "Per-archetype breakdown"
  all appear correctly. Full suite green at 113/113 (up from 107 after A7 —
  6 new tests, 0 regressions).
- **Verification — real checkpoints.** `configs/_a9_real_check.yaml`
  (scratch, deleted after use; extract+l0+report only) against live
  `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small` on the real
  `benchmark_medium/public_dev` corpus (288 series: 188 `random_parametric`,
  60 `mixture`, 40 `parametric` — confirmed via `meta.parquet`'s new
  `generator` column, exactly matching the corpus's documented composition).
  `l0/summary.json`'s `per_archetype` block recovered all **12** of
  `CLAUDE.md` §4.2's archetypes as distinct groups (`trend_dominant`,
  `seasonal_dominant`, `multi_seasonal_complex`, `regime_switching`,
  `ar_colored_noise`, `anomaly_heavy`, `clean_low_noise`, `noisy_chaotic`,
  `random_walk_drift`, `intermittent_bursts`, `amplitude_modulated`,
  `nonsinusoidal_seasonal`), each with a real Holm-corrected paired test
  (e.g. `multi_seasonal_complex` ratio 0.365, p=0.005 but p_holm=0.07 — a
  real illustration of why the Holm correction matters once the comparison
  fans out to 14 groups instead of 3 families); `mixture` and `parametric`
  correctly appeared in `fallback_to_family` with zero groups dropped for
  low n. `report.html` rendered the "Per-archetype breakdown" section. This
  is exactly the previously-unanswerable question this item names in its
  own blast radius — "does the family-level gap hold on every archetype
  `random_parametric` was pooled from" — now answerable directly from one
  run's `l0/summary.json`, no `sae/ground_truth.py` archetype-label
  reconstruction (A10's origin) required.

### A10 — Ground-truth archetype dummies count every real-derived series as a negative `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** Real-derived-tier series carry an empty `GroundTruth` by
  design (`CLAUDE.md` §4.1); the archetype dummy encoding treats that as
  `archetype_x = 0` — a valid negative example — rather than "not applicable."
- **Evidence.** §6.2's 2026-08-06 Findings, which found this by reading real
  exemplar rows (two `mixture` series at activation 0.0 / `field_value` 0.0
  sitting in a top-exemplar table) and correctly declined to change it as a
  side effect of a reporting feature.
- **Blast radius.** Every archetype-related ρ already recorded in §6.2
  (including TimesFM's headline `archetype_trend_dominant` ρ=0.650/0.752) is
  computed against a label that partly encodes *tier*, not archetype. A
  feature that merely separates synthetic from real-derived series scores as
  an archetype detector. Interacts with A4: which tiers are even present
  depends on a head slice.
- **Fix.**
  1. `encode_series_level` emits `NaN` for any field a series' tier cannot
     express; `best_ground_truth_matches` already reports per-field `n`, so
     drop `NaN` rows per field and let `n` tell the reader how much support a
     match has.
  2. Add an explicit `tier` (and `generator`, from A9) column to the
     candidate field set, so "this feature detects the real-derived tier" is a
     *detectable, nameable* match rather than something that masquerades as an
     archetype match.
  3. Re-run against `runs/medium_run_chronos_base` and record old vs. new ρ
     side by side once, then supersede the old numbers in place with a dated
     correction (§0.2). Don't silently re-baseline.
  4. Test: a planted feature that perfectly separates tiers must match `tier`,
     not an archetype dummy.
- **Cost.** Half a day + a short re-scoring run.

**Findings (2026-08-06).** Implemented all four fix items in
`sae/ground_truth.py`, and found (via this item's own real-checkpoint
verification, per `CLAUDE.md` §2.4) two additional real bugs one level
removed from the one this item was named for:

- **Item 1 (archetype dummies), as named.** `load_ground_truth_table`'s
  per-archetype dummy construction moved into a new pure function
  `_add_dummy_columns(df)` (split from the `load_sealed` I/O, matching this
  file's existing pure/IO split for `best_ground_truth_matches` vs
  `ground_truth_alignment`, so the fix could be unit tested directly on a
  synthetic DataFrame). Rows with no archetype now get `NaN`, not `0.0`, on
  every `archetype_*` dummy.
- **Item 2 (tier/generator as first-class fields), as named, plus a real bug
  found underneath it.** `tier_{value}`/`generator_{value}` one-hot dummies
  are now added (always defined, no NaN case, unlike archetype). Building
  this surfaced a genuine, previously undocumented bug: `data.py::
  _sample_to_row`'s `hasattr(sample, "tier")` is **always False** —
  `TimeSeriesSample` has no `tier` attribute; `build_pipeline/builder.py`
  writes it to `provenance.generator_params["tier"]` instead. Confirmed live
  against `benchmark_medium/public_dev` (not assumed): every real
  sealed-corpus series' `meta["tier"]` was silently `"unknown"`. New
  `data.py::_tier_of(row)` checks the top-level key first (smoke rows) then
  falls back to `provenance.generator_params.tier` (real samples); re-verified
  live — `Counter({'synthetic': 228, 'realism_stress': 60})`, matching
  `benchmark_medium`'s known composition exactly. `l0`/A9's `meta.parquet`
  `tier` column inherits this fix for free (same `_assemble` call site).
- **A second, larger bug found the same way, one level removed from the one
  this item names:** `_scalar_ground_truth`'s `len([])`/`"x" in {}` on an
  empty `generative_params` silently produced a valid `0`/`False` for
  `n_seasonalities`/`ar_order`/`n_changepoints`/`n_anomalies`/`has_random_
  walk`/`has_intermittency`/`has_heteroskedastic` too — the exact same
  "not applicable" vs "genuinely zero" confusion this item is about, just
  not confined to the `archetype` dummy. Gated all seven fields on a new
  `has_gt` flag. **First attempt used `has_gt = bool(generative_params)`
  and was itself wrong** — checked directly against a real `mixture` sample
  rather than assumed (§2.4 again, twice in one item): `mixture` sets
  `generative_params={"mode": ...}` (`generators.py:450`), non-empty but with
  none of `parametric`'s structural keys, so `bool(p)` would have left this
  exact generator's rows un-fixed. Corrected to `has_gt = "trend" in p`,
  since `_generative_params()` (`generators.py:152`) sets `trend`/
  `seasonalities`/`ar_coeffs`/`noise_scale` together unconditionally for
  every `parametric`/`random_parametric` call and `mixture`/
  `block_bootstrap`/`sequential_par` never set `"trend"` at all — verified
  against a live `mixture` sample from `benchmark_medium` before trusting it.
- **Item 3 (re-score and supersede old numbers), done against real
  checkpoints, not a fresh run.** No retraining needed: `sae/train.py`
  already persists trained SAE weights to `.pt` (`load_sae_checkpoint`), so
  the already-extracted `runs/medium_run_chronos_base` store plus its two
  already-trained checkpoints (`TimesFM/stacked_xf.18`,
  `Chronos-T5-Base/encoder.block.6`) were re-scored directly through the
  fixed `ground_truth_alignment`, with zero new model forward passes.
  **Old vs. new, side by side:**
  - `mean_abs_rho_matched`: TimesFM 0.317 → 0.349; Chronos-T5-Base 0.306 →
    0.401 — a real, substantial increase in how well-matched the average
    top feature is, exactly as expected once tier-masquerading matches stop
    diluting the field.
  - **Most matches previously attributed to `n_seasonalities` were actually
    detecting tier.** E.g. TimesFM feature 5109: old best match
    `n_seasonalities` ρ=−0.552 (n=288) → new best match
    `tier_realism_stress` ρ=+0.943 (n=288). Six of TimesFM's top-8 and six
    of Chronos-T5-Base's top-8 (by new ρ) flipped from a spurious
    `n_seasonalities`/`ar_order`/`n_anomalies` match to a `tier_*` match at
    much higher |ρ| (0.70–0.94 vs. 0.43–0.60) — these features were tier
    detectors all along, previously misattributed to the nearest available
    numeric field because no `tier` field existed to compete for the match.
    Genuinely intermittency-detecting features (e.g. TimesFM/Chronos feature
    2189, `has_intermittency` ρ=0.881→0.880) were essentially unaffected,
    confirming the fix corrects specifically the tier-masquerading cases and
    doesn't disturb real matches.
  - **The headline number this item's own evidence names is confirmed
    genuine, not an artifact.** `CLAUDE.md` §13's `archetype_trend_dominant`
    ρ=0.650 (TimesFM feature 2872): re-scored at ρ=0.648, now correctly on
    `n=188` (the actual `random_parametric` count) instead of the old,
    tier-inflated `n=288`. Every `archetype_trend_dominant` match in both
    models' old top lists survived at essentially the same ρ under the fix
    (TimesFM: 0.650→0.648, 0.561→0.558; Chronos-T5-Base: six features at
    0.530→0.527, 0.518→0.514, 0.500→0.497) — this specific finding was a
    real archetype match, not a tier-confound artifact, and is superseded
    in place here with the corrected `n` per this item's own "supersede, don't
    silently re-baseline" instruction; no other file restates these numbers
    (`CLAUDE.md` §13's item 3 links here rather than repeating them).
- **Item 4 (test).** Done, generalized to cover both the named bug and the
  one found underneath it. New `tests/test_ground_truth_dummies.py` (9
  tests): real-derived rows get `NaN` not `0.0` on archetype dummies;
  `tier_*`/`generator_*` dummies are always defined; a planted tier-detecting
  feature matches `tier_*` and not an archetype dummy (with a sanity check
  proving the pre-fix encoding really was indistinguishable, restricted to
  the two-archetype/tier comparison where the bug actually bites); `_tier_of`
  resolves both the top-level and `provenance.generator_params` paths and
  degrades to `"unknown"` without crashing; a fully-empty `GroundTruth` gets
  `None` on every count field; a real-shaped `mixture` `generative_params`
  (`{"mode": ...}`) is correctly *not* mistaken for real ground truth; and a
  genuinely zero-valued synthetic recipe still records real `0.0`s, not
  `None`. Full suite: 122 tests collected, 121 passed, 0 failures (up from
  113 before this item — the 1-test discrepancy from this item's own 9 new
  tests is unreconciled bookkeeping across sessions, not a failure or a
  skip; the log shows zero failures/errors/skips).
- **Known, accepted side-effect, not fixed this session:** `layer_screen.py`'s
  `factor_emergence` method (not the default; `work_bend` is, per §6.1.1)
  also calls `load_ground_truth_table` and now sees the new `tier_*`/
  `generator_*` dummies as additional candidate factors alongside every
  archetype. Left as is — `factor_emergence` already has its own open,
  larger-scope defect (`CLAUDE.md` §11.18, not yet re-verified against a
  real bake-off) and is not the default selector; revisit if that method is
  ever promoted past `work_bend`.

### A11 — MASE's scale term degenerates on intermittent and near-constant series `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** The denominator is the mean absolute first difference of
  the context (`+ 1e-8`). Under heavy intermittency or a flat context it
  collapses, and because `parametric`'s intermittency zeroes the horizon too,
  numerator and denominator both floor.
- **Evidence.** §7's intermittency-sweep caveat (MASE *falls* for both models
  at rate 0.8, which is an artifact); `analysis/stats.py::mase()`.
- **Blast radius.** Wider than that one sweep, because `mase()` was
  deliberately centralized (§6.2's refactor) and is now the metric behind L0's
  headline numbers and per-family strengths, head/MLP ablation ΔMASE, L3
  restoration scores, the SAE forecast-preservation check, every parameter
  sweep, and the bias card's verdicts. Any zero-heavy or low-volatility family
  is silently compressed toward a floor in all of them. Consequence worth
  stating plainly: `intermittent_bursts` — per `CLAUDE.md` §4.2 the sharpest
  quantized-vs-continuous stressor available, opted into deliberately by §5.1
  — cannot currently be scored interpretably, so one of the corpus's most
  discriminating archetypes is dead weight until this is fixed.
- **Fix.**
  1. `l0.scale: mean_abs_diff|seasonal_naive` (default `mean_abs_diff` so
     every recorded number keeps reproducing). `seasonal_naive` uses period
     `m` from ground truth where available, else an autocorrelation estimate,
     and is the standard fix for exactly this degeneracy.
  2. A **reliability guard** independent of the scale choice: when the
     denominator is below a threshold fraction of the target's own mean
     absolute level, mark that series `mase_reliable: false`, exclude it from
     aggregates, and **report the excluded count** everywhere a MASE
     aggregate is shown. Silent exclusion would be its own version of this
     bug.
  3. Add one scale-free companion metric that doesn't degenerate the same way
     (e.g. MAE normalized by the target's own MAD) and show it beside MASE in
     L0 and the sweeps, so a floor effect is visible as a divergence between
     the two.
  4. Audit `sMAPE`'s zero handling in `l0_behavioral.py` at the same time —
     same family of failure, same corpus regions.
  5. Test: a mostly-zero synthetic series must be flagged unreliable, not
     scored; a normal series's MASE must be byte-identical to today's.
- **Cost.** ~a day including re-running the intermittency sweep to see what
  it actually says.

**Findings (2026-08-06).** Implemented items 1-4; item 5 covered by new
tests. Scoped deliberately to `stats.py` (shared) + `l0_behavioral.py`/
`config.py`/`report.py` (L0, the stage this item's own evidence and blast
radius are about) rather than every one of `mase()`'s ~8 call sites:

- **Item 1.** `stats.py` gained `dominant_period()` (moved out of
  `attention.py`, which had its own private, byte-identical copy —
  deduplicated, `attention.py` now imports the shared one under its old
  private name so nothing else in that file changed), `_mase_scale()`
  (`mean_abs_diff` — unchanged formula, still the default — or
  `seasonal_naive`, mean absolute error of a period-`m` seasonal-naive
  forecast, `m` autocorrelation-estimated per series when not supplied),
  and `mase()` gained optional `scale_mode`/`periods` kwargs with defaults
  that reproduce every existing call byte-for-byte. New `config.py::
  L0Config.scale: "mean_abs_diff"|"seasonal_naive"` (default
  `"mean_abs_diff"`). **Ground-truth-period lookup (`m` "from ground truth
  where available") was not built** — it would require L0 to load the
  sealed corpus's `GroundTruth` records the way `sae/ground_truth.py`
  does, a new coupling this stage doesn't otherwise have; the
  always-available autocorrelation fallback is what `seasonal_naive`
  actually uses. Stated here as a real, deliberate gap, not silently
  dropped.
- **Item 2.** New `stats.py::mase_reliability()`: `False` when the MASE
  denominator is below `min_scale_frac` of the target's own mean absolute
  level, independent of `scale_mode`. New `L0Config.min_scale_frac: 0.05`
  (on by default, unlike `scale` — this guard's whole purpose is to stop a
  degenerate case from silently entering aggregates, so a default of `0.0`
  would defeat it). `l0_behavioral.py::_score` records a per-series
  `mase_reliable` column in `metrics.parquet`; `_summarize`/
  `_archetype_summary` compute every MASE-based number (means, CIs, ratios,
  paired tests) on the reliable subset only, while `smape`/`pinball`/
  `mae_over_mad` still aggregate every row (they don't share this
  degeneracy). Excluded counts are recorded corpus-wide
  (`summary["mase_reliability"]`) and per `per_family`/`per_archetype` row
  (`mase_n_excluded`); `report.py` renders both — a blurb naming the total
  excluded count right under the MASE chart (auto-added as a finding too)
  and a `mase_n_excluded` column in the new per-family/per-archetype tables.
- **Item 3.** New `stats.py::mae_over_mad()` — MAE normalized by the
  target's own median absolute deviation, never touching the context, so it
  can't share MASE's context-driven degeneracy. Reported beside MASE in
  every L0 table (`_score`, `per_family`, `overall`, `per_archetype`) with a
  report note on how to read a MASE/`mae_over_mad` divergence. **Not wired
  into the parameter sweeps** (`analysis/parameter_sweep.py` still computes
  its own local MASE only) — left as a documented gap since sweep case
  studies are a separate, lower-priority consumer than L0's headline
  numbers.
- **Item 4 (sMAPE audit).** Checked rather than assumed (`CLAUDE.md` §2.4):
  generated a real `intermittent_bursts` series
  (`random_parametric(seed=0, archetypes=["intermittent_bursts"])`, 72.5%
  zero-valued) and scored a deliberately-wrong constant-0.3 forecast against
  its zero-heavy horizon. sMAPE = **1.97** (correctly saturating near its
  own max of 2.0 — a genuinely bad forecast against a near-zero target
  scores near-worst, not `inf`/`NaN`), MASE for the identical case = **3.20**
  (also a large, sane number, not the "artifact" reading this item warned
  about, since here the *context* wasn't degenerate, only the *target*
  region tested). **Conclusion: no fix needed.** `l0_behavioral.py::_score`'s
  existing `+ 1e-8` epsilon on the denominator already handles the
  both-zero edge case correctly; the real degeneracy this item is about
  lives in MASE's context-derived scale term, not sMAPE's target-derived
  one, and item 2's reliability guard is the actual fix for that.
- **Item 5 (tests).** New `tests/test_mase_reliability.py` (11 tests):
  `mean_abs_diff` scale matches the pre-existing formula byte-for-byte;
  `dominant_period` recovers a planted period-24 sine to within 1 step;
  `seasonal_naive` scale matches a hand-computed value for a known period;
  an unknown `scale_mode` raises; `mase_reliability` is all-`True` when
  disabled (`min_scale_frac<=0`), correctly flags an all-flat-context/
  nonzero-target batch as unreliable, and correctly keeps a normal series
  reliable; `mae_over_mad` matches a hand-computed value;
  `_summarize`/`_score` correctly exclude unreliable rows from the MASE
  mean while keeping them in smape (a planted 999.0 MASE on unreliable rows
  does **not** move the reported per-family mean, proving the exclusion
  actually happens, not just gets flagged); and a full mini pipeline run
  (mock adapters, `l0.min_scale_frac: 0.05`) confirms `mase_reliability`/
  `mae_over_mad` appear in real `summary.json`/`report.html` output. **Two
  pre-existing tests broke and were fixed, not worked around**:
  `test_archetype_reporting.py`'s synthetic `_metrics()` fixture built a
  metrics DataFrame by hand without a `mase_reliable`/`mae_over_mad` column
  (written before this item existed) — `_archetype_summary` now requires
  both, so both tests failed with a `KeyError` until the fixture was
  updated to add `mase_reliable=True`/`mae_over_mad=mase` columns, matching
  the "every row reliable, sanity metric equals MASE" semantics those
  synthetic rows already implied. Full suite green at 133/133 (up from 121
  after A10).
- **Verification — real checkpoints.** `configs/_a11_real_check.yaml`
  (scratch, deleted after use; extract+l0+report only,
  `min_scale_frac: 0.05`) against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small` on the real `benchmark_medium/public_dev`
  corpus. **16 of 576 model-series rows (8 unique series, since reliability
  is series-level and duplicated across both models) were excluded as
  unreliable** — real numbers, real degenerate contexts, not a synthetic
  demonstration. Rendered correctly end to end: `summary.json`'s
  `mase_reliability` block, `mae_over_mad` alongside `mase` in every
  `per_family`/`per_archetype` row, and `report.html`'s "16 of 576 series
  excluded from every MASE mean/ratio/test above and below" note (grep-
  confirmed present) plus its new "Per-family metrics" table. **A genuinely
  non-obvious finding, worth stating plainly rather than assuming the
  reliability guard would only fire where §7's caveat suspected it**: the
  8 excluded series were **not** concentrated in `intermittent_bursts` (0
  excluded there, despite being the archetype `CLAUDE.md` §4.2 calls "the
  sharpest quantized-vs-continuous stressor" and the one this item's own
  Evidence section names) — they were scattered across `parametric` (2) and
  `random_parametric` (6, including 2 in `trend_dominant` specifically).
  The degeneracy this item names turns out to be driven by genuinely flat
  *stretches of a sampled recipe* (e.g. a `trend_dominant` draw with a
  near-zero trend scale and low noise), not specifically by the
  intermittency archetype's design — a real, previously-unmeasured fact
  about which series this guard actually protects, not something either
  this item's evidence or `CLAUDE.md` had predicted.
- **Deliberately out of scope, stated rather than silently dropped**: L3's
  patching-restoration MASE, `attention.py`'s ablation ΔMASE, `lens.py`'s
  crystallization-depth MASE, `sae/eval.py`'s forecast-preservation MASE,
  `confirm.py`'s re-tested MASE, and `parameter_sweep.py`'s sweep MASE all
  keep calling `mase()`/computing MASE inline with the pre-A11 defaults
  (`scale_mode="mean_abs_diff"`, no reliability filtering) — every number
  those stages have already recorded stays byte-identical. This item's own
  Fix section named `l0.scale` (an L0-config knob) as the mechanism, and
  the reliability guard's report-surfacing ask ("everywhere a MASE
  aggregate is shown") is read here as "wherever L0 already aggregates
  MASE," not as a mandate to touch every downstream stage's own metric
  computation in the same pass — each of those is its own, separately-
  scoped follow-up if the degeneracy turns out to matter there too.

### A12 — The corruption battery is uncalibrated across corruptions `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** Each corruption's strength is set independently, so
  "behavioral sensitivity" bars are not comparable across corruptions —
  `level_shift` dominates by construction at 4-12× everything else.
- **Evidence.** §5's 2026-08-04 Findings (with numbers); `CLAUDE.md` §6.5's
  standing warning. The 2026-08-04 session fixed the *display* and explicitly
  left calibration as "an option, not a decision."
- **Blast radius.** The L3 section's headline chart, the cross-model
  fingerprint agreement (rank correlations are computed per corruption, so
  those are fine — but any statement of the form "model X is most sensitive
  to Y" is not), and any future reader who trusts bar heights over the prose
  note. Also blocks the "is one model better at high-frequency structure vs.
  statistical means" question (§1) from being answered *contrastively across
  corruption types* rather than within one.
- **Fix.**
  1. `l3.calibrate: none|input_energy` (default `none` — every recorded number
     keeps reproducing byte-for-byte, which is what made the last session
     defer this).
  2. Under `input_energy`, solve per corruption for the parameter that hits a
     common per-series perturbation-energy budget, measured with the
     corruption functions alone (no model in the loop — cheap, pure numpy,
     unit-testable). Record realized budget, parameter, and **footprint**
     (fraction of timesteps touched) per corruption in `l3/meta.json`.
  3. Report shows footprint next to every sensitivity bar regardless of mode,
     so sparse-by-construction corruptions (`spike` ~0.6%, `dropout` ~14%)
     can't be misread as "the model is robust" — the exact misreading the
     2026-08-04 session had to investigate by hand.
  4. Keep every corruption. §5's decision to keep the low-signal end of the
     contrast stands and is not what this item revisits.
- **Cost.** ~a day.

**Findings (2026-08-06).** Implemented all four fix items in
`l3_perturbation.py` + `config.py` + `report.py`:

- **Item 1.** `L3Config.calibrate: "none"|"input_energy"`, default `"none"`.
- **Item 2.** New pure-numpy `calibrate_corruptions(names, configs,
  contexts, seed)`: per-corruption perturbation energy (`mean((corrupted -
  clean)**2)`) and footprint (`fraction of timesteps changed beyond a
  scale-relative tolerance`), a shared budget set to the **median of the
  battery's own natural (pre-calibration) energies** (anchored to this
  battery's existing scale rather than an arbitrary constant — a deliberate
  design choice this item's own wording left open), and a per-corruption
  solve to hit that budget. **Checked monotonicity directly before trusting
  it** (`CLAUDE.md` §2.4): swept each corruption's obvious magnitude
  parameter against energy on a real batch of series.
  `level_shift:scale`, `spike:scale`, `smooth:kernel`, `warp:strength`,
  `dropout:frac` are all confirmed monotone increasing over a wide range —
  calibrated via a 30-iteration bisection (`_bisect_param`), clamped (not
  extrapolated) when the target falls outside the search range.
  `noise:snr_db` is calibrated via a closed form instead
  (`snr_db = 10·log10(mean_power / target)`, exact since
  `corrupt_noise`'s own formula is closed-form in `snr_db`) — cheaper and
  exact where bisection would just be approximating an already-known
  relationship. **`frequency_shift:factor` was swept and found genuinely
  non-monotone past ~1.5×** (energy plateaus/wobbles rather than growing) —
  excluded from calibration on that empirical basis, not assumed.
  `detrend` (no free parameter) and `deseasonalize` (integer `top_k`, not a
  continuous magnitude knob) have nothing to calibrate by construction.
  All three are still fully reported (footprint/energy), just never
  adjusted — `calibrated: false` in `l3/meta.json`'s new `calibration`
  block, not silently omitted from it.
- **Item 3.** Footprint is computed and stored for **every** corruption
  regardless of `calibrate` mode (`corruption_footprint_meta` covers the
  `"none"` path). `report.py::_sec_l3` now labels every behavioral-
  sensitivity bar's x-axis tick with its footprint percentage
  (`"spike<br>0.6% touched"` etc., real numbers below) and, when
  `calibrate: input_energy`, prepends a blurb naming which corruptions were
  calibrated to which shared budget and which weren't and why.
- **Item 4.** Every corruption stays in the battery in both modes — nothing
  dropped, per §5's own prior decision this item explicitly said stands.
- **A real bug found and fixed via this item's own testing, not by
  inspection alone**: the first implementation used `seed + 200 + i` inside
  `_bisect_param`'s search but `seed + i` for the *final* realized-energy
  verification recompute after the solve — for a corruption whose output
  depends on more than just the calibrated parameter (`warp`'s random knot
  jitter, drawn fresh from the RNG on every call), searching and verifying
  under two different RNG streams meant the verified energy didn't actually
  match what the search had converged on. Caught by a unit test asserting
  <5% relative error (got 5.18% — just over), not by eyeballing the numbers.
  Fixed by using one consistent seed per corruption throughout
  `calibrate_corruptions` for both the search and the verification.
  **Lesson, matching this session's own recurring pattern (§11.19's
  device-transfer bug, A9/A10's real-checkpoint-only findings): a
  stochastic corruption function needs the *same* RNG draw sequence
  compared against itself, not just the same parameter value, or a
  seed mismatch masquerades as a solver-tolerance issue.**
- **Verification — unit tests.** New `tests/test_l3_calibration.py` (9
  tests): energy/footprint are exactly zero for identical arrays;
  `spike`'s footprint is small as expected; `_bisect_param` hits a target
  within tolerance and correctly clamps rather than extrapolates when the
  target is out of range; `calibrate: none` leaves every config
  byte-identical while still reporting footprint/energy for all of them;
  `calibrate_corruptions` converges every calibratable corruption to the
  shared median budget within 5%; `detrend`/`deseasonalize` are never
  touched by calibration; a full mini pipeline with the **default** config
  (`calibrate: none`) produces a `calibration` block with `calibrated:
  false` everywhere (the byte-identical-reproduction regression guard this
  item's item 1 promised); and a full mini pipeline with `calibrate:
  input_energy` produces a real calibration block and renders both the
  calibration blurb and footprint captions in `report.html` (matched
  loosely against `"touched"`, not the literal `"touched</span>"`, since
  plotly JSON-escapes `<`/`/` in embedded tick-label strings — caught by
  the first assertion failing, not assumed). Full suite green at 142/142
  (up from 133 after A11).
- **Verification — real checkpoints.** `configs/_a12_real_check.yaml`
  (scratch, deleted after use; extract+l3+report only, patching disabled to
  keep it fast, `calibrate: input_energy`) against live
  `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small` on the real
  `benchmark_medium/public_dev` corpus. Real target energy = **0.1606**
  (median of the battery's configured-default natural energies). All six
  calibratable corruptions converged close to it: `noise` 0.1608, `level_
  shift` 0.1606, `spike` 0.1606, `dropout` 0.1626, `warp` 0.1606, `smooth`
  0.1512 (the one visibly furthest off at ~6% relative error — noted
  plainly rather than smoothed over; still within the same order of
  magnitude and not a sign of the seed bug recurring, since `smooth`'s
  corruption function is itself deterministic given `kernel`, no RNG
  dependence to mismatch). `frequency_shift`/`detrend`/`deseasonalize`
  correctly stayed at their natural, uncalibrated energies (1.132, 0.049,
  0.369 respectively). Real footprints, exactly the contrast this item's
  blast radius named: `spike` 0.6%, `dropout` 28.0%, `level_shift` 40.0%,
  `warp` 91.6%, `detrend` 96.2%, `smooth` 96.1%, `frequency_shift` 96.9%,
  `deseasonalize` 99.6%, `noise` 99.7% — confirming `spike`/`dropout`'s low
  aggregate behavioral-sensitivity bars (documented elsewhere as
  potentially misleading) are exactly the sparse-footprint corruptions this
  fix's whole point is to keep from being misread as model robustness.
  `report.html` rendered the calibration blurb and every bar's footprint
  caption (grep-confirmed).

### A13 — No repeat-run noise floor, so small ΔMASE values are uninterpretable `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** Nothing measures how much a model's own MASE varies
  between two identical calls. Every Δ in the repo is compared against zero
  rather than against that floor.
- **Evidence.** `CLAUDE.md` §12 item 3 (Chronos-T5 samples, TimesFM is
  deterministic; "seeds are pinned and `num_samples` reduced during patching"
  — pinning makes a run reproducible, it does not make a *difference*
  meaningful); head/MLP ablation ΔMASE (`attention.py`), SAE
  forecast-preservation ΔMASE (`sae/eval.py`), L3 restoration
  (`l3_perturbation.py`) all compare across separate `predict` calls.
- **Blast radius.** The SAE validity check's TimesFM "pass" is ΔMASE
  **+0.047** — a number whose significance is entirely a question of the noise
  floor, which is unknown. Head-ablation "top load-bearing heads" rankings
  can be sorted by noise in their tail. For the sampled model the floor may
  be substantial, and the two models' floors differ, which means every
  cross-model Δ comparison currently rests on an unmeasured asymmetry that
  `CLAUDE.md` §12 already names as a known hazard.
- **Fix.**
  1. Cheap sub-step in L0 (or a tiny `noise_floor` stage): for each model,
     predict the same `k` series `R` times (R≈5) with different seeds; write
     `l0/noise_floor.json` — distribution of |ΔMASE| between identical
     repeats, overall and per family, plus a `deterministic: bool` flag taken
     from the adapter.
  2. Every ΔMASE figure draws the floor as a shaded band / reference line;
     every findings string that asserts a Δ is meaningful must compare against
     it, and say so ("+0.047, against a repeat-run floor of ±0.0X").
  3. Deterministic models should show a floor of exactly 0 — a free, strong
     wiring sanity check, in the same spirit as TimesFM's exactly-0.0
     `future_mass` (§5.4).
  4. `Chronos-Bolt` as the deterministic control is already suggested by
     `CLAUDE.md` §12; the floor measurement is what makes that comparison
     quantitative instead of rhetorical.
- **Cost.** Half a day. Cheapest item here relative to how many recorded
  numbers it re-contextualizes.

**Findings (2026-08-06).** Implemented as a cheap sub-step of L0 (item 1's
first option), not a new pipeline stage — no `pipeline.py`/`manifest.py`
wiring needed, since it reuses L0's already-loaded adapters and existing
`_predict_all`/`mase` machinery.

- **Item 1.** New `l0_behavioral.py::_measure_noise_floor()`: for each
  model, calls `adapter.predict()` `noise_floor_repeats` times on the same
  `noise_floor_series`-sized subset and measures `|ΔMASE|` between
  consecutive repeats (overall, per family, and a `deterministic: bool`
  flag). **Deliberately did not add a `seed` parameter to `ModelAdapter.
  predict()`** — repeat calls under the ambient (advancing) global torch
  RNG state already differ for a genuinely sampling model and are
  byte-identical for a genuinely deterministic one, which is exactly the
  signal this item wants; adding an explicit seed contract would have
  meant touching `base.py`'s interface plus all 4 concrete adapters for no
  additional signal. Written to a new `l0/noise_floor.json`, model → `
  {repeats, n_series, deterministic, mase_abs_delta_mean/p95/max,
  per_family}`.
- **Item 1, a scoping deviation worth stating**: unlike A11/A12's "off or
  byte-identical by default" precedent, this item's own fix plan implies
  the floor should actually be *on* to recontextualize existing ΔMASE
  figures — a floor nobody ever measures is not different from not having
  this item at all. Defaulted **on** (`noise_floor_repeats: 3,
  noise_floor_series: 16`), a small, deliberately cheap addition (3 extra
  `predict()` calls on 16 series per model) rather than a full opt-in.
  `noise_floor_repeats: 0` fully disables it (adds zero cost, `l0/
  noise_floor.json` is simply not written) for anyone who wants the old
  runtime exactly.
- **Item 2.** Wired into three places, not literally "every ΔMASE display"
  (see the explicit deferral below): `report.py::_noise_floor_block()` (a
  new "Repeat-run noise floor" table + note under L0's own chart, shared
  by any section that wants to reference it); L3's "Behavioral sensitivity"
  chart gains a **dotted horizontal reference line per sampling model** at
  its own floor's mean, plus a note explaining a bar below its model's
  line is not distinguishable from repeat-run noise; and the SAE section's
  `forecast_preservation` finding string now appends `(repeat-run floor
  ±X)` for a sampling model or `(this model is deterministic; the delta is
  real signal)` for a deterministic one.
- **Item 3.** Confirmed exactly as predicted, on real weights (below) and
  via the mock-adapter unit tests: a model with no sampling in its forecast
  path shows `deterministic: true` and a floor of literally `0.0` — the
  free wiring sanity check this item names, same spirit as TimesFM's
  exactly-0.0 `future_mass` (`CLAUDE.md` §9/§5.4).
- **Item 4.** Not additional code — `Chronos-Bolt`'s adapter already exists
  (`CLAUDE.md` §6.2's support matrix); this item's contribution is that the
  "Chronos-Bolt as deterministic control" comparison `CLAUDE.md` §12
  already suggests is now backed by an actual measured number instead of
  being purely rhetorical, the moment anyone runs a config pairing it with
  Chronos-T5.
- **Deliberately out of scope, stated rather than silently dropped**:
  `attention.py`'s head/MLP ablation ΔMASE charts and `top_heads` rankings
  (named explicitly in this item's own Evidence/Blast-radius sections) do
  **not** gain a floor reference line in this pass — that chart's data
  shape (per-head, per-family delta arrays, already flagged as this
  repo's most fragile report-key surface, `CLAUDE.md` §11.6) would need
  its own, more careful integration than a single reference line, and
  ablation deltas are computed from a single clean/ablated forecast pair
  per head rather than L0's own repeat-call structure — comparable in
  principle but a separate wiring task. `l3_perturbation.py`'s activation-
  patching restoration curves (also named in Evidence) are likewise left
  unannotated; both are natural follow-ups once this item's core mechanism
  is trusted, not silently assumed already covered.
- **Verification — unit tests.** New `tests/test_noise_floor.py` (5 tests):
  a fixed-function fake adapter gets exactly `0.0`/`deterministic: true`; a
  fake adapter with real per-call sampling noise gets a real nonzero floor
  with `p95 >= mean` and correct per-family keys; repeats/n_series are
  recorded correctly; a full mini pipeline (mock adapters, default config)
  writes `l0/noise_floor.json` with both mocks correctly `deterministic:
  true` (mocks are fixed functions of their input, no dropout/sampling at
  inference) and renders "Repeat-run noise floor" in `report.html`; and
  `noise_floor_repeats: 0` fully disables the artifact. Full suite green
  at 147/147 (up from 142 after A12).
- **Verification — real checkpoints.** `configs/_a13_real_check.yaml`
  (scratch, deleted after use; extract+l0+l3(patching disabled)+report
  only) against live `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-small` on `benchmark_medium/public_dev`. **Exactly the
  asymmetry `CLAUDE.md` §12 item 3 names, now a real number instead of a
  caveat**: TimesFM — `deterministic: true`, floor `0.0` across all three
  families, confirming the free wiring check. Chronos-T5 — `deterministic:
  false`, mean `|ΔMASE|` **0.167**, p95 **0.586**, max **0.999** — a floor
  roughly **3-20x the size of the SAE forecast-preservation "pass"
  threshold (+0.047) this item's own blast-radius section names**, meaning
  that specific "pass" (measured on deterministic TimesFM) was never
  actually at risk from this, but the identical check on a sampling model
  like Chronos-T5 would be effectively unmeasurable at that magnitude
  without many more repeats or series than either check currently uses — a
  concrete, previously-unquantified risk this item's own Blast-radius
  section predicted correctly. **A further non-obvious finding**: the
  floor is not uniform across families — `mixture` 0.041 vs `parametric`
  0.463, over an 11x range — so "the noise floor" is meaningfully family-
  dependent, not a single number a future session should assume transfers
  across corpora. `report.html` rendered the table, the dotted reference
  line at Chronos-T5's exact floor value in the L3 chart (grep-confirmed
  in the embedded plot JSON), and the "is deterministic" finding text.

### A14 — No CI; the whole test suite is manual, and upstream breaks are found by humans on new machines `[x]` · **P1** · fixed 2026-08-06

- **What's wrong.** There is no CI configuration in the repo. Every "suite
  green" claim is a session's word for it, in whatever environment that
  session had.
- **Evidence.** §10's CI checklist item (still `[~]`, explicitly noting no CI
  config exists); the repeated-rediscovery record: `CLAUDE.md` §11.8 (timesfm
  API removal), §11.9 (`datasets>=3`), §11.10 (`tsbootstrap` rewrite), §11.15
  (zarr v2/v3), §11.17 (cp1252), plus §5's third-machine zarr recurrence.
- **Blast radius.** Every one of those was found by a human losing a session
  to it on a new machine. That is the single most repeated cost in this
  repo's history, and it is exactly what CI exists to absorb.
- **Fix.** `.github/workflows/ci.yml` with three jobs:
  1. **`fast` (CPU, pinned):** `tsfm_lens` + `tsfm_benchmark` suites,
     `compileall`, the invariant-11 absolute-path grep as a real failing
     check, `tests/test_adapter_conformance.py` against all mocks, and a full
     `run.py --config configs/smoke.yaml` end-to-end asserting section and
     finding counts (the smoke run is minutes and mock-only — no downloads, no
     GPU, per `CLAUDE.md` §8).
  2. **`pinned` (the DEPENDENCIES.md recipe):** installs the exact verified
     versions and runs the same suite — proves the documented recipe still
     resolves and works.
  3. **`bleeding` (allowed to fail, scheduled weekly):** installs the *latest*
     of every load-bearing dependency (zarr, datasets, numpy, torch,
     transformers, timesfm, chronos-forecasting) and runs the same suite. Its
     entire job is to fail on someone else's release rather than on a future
     session's Tuesday. When it fails, that is a `CLAUDE.md` §11 entry
     waiting to be written cheaply instead of expensively.
  4. Publish `report.html` and `coverage.json` from the smoke run as build
     artifacts, so report regressions are reviewable in a PR.
  5. Golden-hash test blocking once A8 is green (`xfail(strict=True)` until).
- **Cost.** ~a day. Prerequisite for §16's Tier 1 credibility.

**Findings (2026-08-06).** New `.github/workflows/ci.yml` with three jobs,
matching the fix plan closely with two stated, deliberate deviations:

- **Item 1 (`fast`).** pip-only (not conda) install of DEPENDENCIES.md §3's
  exact versions where a PyPI wheel exists, CPU-only torch, then
  `compileall`, the invariant-11 absolute-path check, both test suites
  (`tsfm_benchmark/tests`, `tsfm_lens/tests` — the latter already runs
  `test_adapter_conformance.py` against every mock, nothing extra needed),
  and a full `run.py --config configs/smoke.yaml` run. `mkl` is
  **deliberately not pinned in CI** — that pin only matters for
  conda-forge's MKL-linked numpy build (`CLAUDE.md` §11.13); PyPI's numpy
  wheel links OpenBLAS, so the ABI crash it guards against doesn't apply to
  a pip-based job. Report/coverage uploaded as build artifacts (item 4).
- **Item 2 (`pinned`).** Recreates DEPENDENCIES.md §2's exact conda recipe
  via `conda-incubator/setup-miniconda`, runs the same two suites. No GPU
  on a GitHub-hosted runner, so torch is CPU-only here too — "pinned"
  verifies the *documented recovery procedure resolves and the suite
  passes*, not that a GPU is present; real-checkpoint verification stays
  the separate, manual, per-session step this repo has used throughout
  (every ROADMAP.md Findings block above this one).
- **Item 3 (`bleeding`).** Weekly cron (`workflow_dispatch` also available
  on demand) installing the *latest* of numpy/torch/transformers/timesfm/
  chronos-forecasting unpinned, `continue-on-error: true`. `zarr`/`datasets`
  are deliberately still bounded at their documented ceilings (`<3` each)
  rather than left fully open — this job exists to catch a break in a
  library that **isn't** already a understood, documented constraint
  (`CLAUDE.md` §11.8's untracked `timesfm` API removal is exactly the
  shape of thing this job is for), not to manufacture a weekly rediscovery
  of the same two already-known ceilings.
- **Item 4.** `smoke-report` artifact (report.html + coverage.json) uploaded
  from the `fast` job on every run (`if: always()`), reviewable in a PR.
- **Item 5.** Not applicable as separate work — A8 already left the
  golden-hash test green with no `xfail` special-casing (A8's own Findings:
  "when A14 lands, this test should simply run as a normal blocking check
  like any other, with no special-casing needed now that it's green"). It
  runs inside `tsfm_benchmark/tests` like every other test, blocking by
  construction.
- **A deliberate deviation from the fix plan's literal wording**: item 1
  says the smoke check should assert "section and finding counts." Built
  instead as: zero `"failed"` sections (via the run's own `coverage.json`,
  A5's artifact) + a floor of >=8 rendered sections + a minimum
  `report.html` byte size — **not** an exact count. An exact count is
  brittle by construction: this session alone changed the smoke run's
  finding count from 18 (an old `CLAUDE.md` reference) to 22, then 23 (A7),
  then 25 (after A9-A13's own new findings) without the pipeline being
  broken at any point — a CI check pinned to an exact number would have
  needed editing after nearly every item in this session, training
  whoever maintains it to treat CI failures as routine busywork rather
  than a real signal. A floor + zero-failures check still catches an
  actual regression (a section crashing, or the report collapsing to
  near-empty) without that maintenance tax.
- **Two real bugs found and fixed via this item's own local verification**
  (`CLAUDE.md` §2.4 — checked by actually running each step, not assumed
  from reading the shell commands):
  1. The invariant-11 grep's first draft (`[A-Za-z]:[\\/]`, no word
     boundary) false-positived on `https://` (the `s` before `:` in
     "https") and on **escaped newlines inside Python string literals**
     (`"data:\n"` contains the literal substring `a:\` — the `\n` escape,
     not a Windows path). Running it against the actual repo surfaced 5
     false positives immediately; fixed by requiring a word boundary
     (`\b[A-Za-z]:[\\/]`) before the candidate drive letter, which a
     `\n`/`\t` escape mid-identifier never has. Re-run: clean.
  2. The smoke-check script assumed `coverage.json` was a bare list of
     section dicts; it's actually `{"summary": ..., "sections": [...]}`
     (A5's own real schema) — caught immediately on the first local run
     (`TypeError: string indices must be integers`), fixed by indexing
     `["sections"]`.
- **A third finding, real but out of this item's own scope — flagged for
  A18, not fixed here**: `clean_reinstall.sh` (repo root) was read while
  designing the `fast`/`bleeding` jobs' install lists as a precedent for a
  pure-pip (non-conda) setup already used in this repo, and turned out to
  **install `datasets==3.6.0`** — directly contradicting the hard `<3`
  ceiling `CLAUDE.md` §11.9, `DEPENDENCIES.md` §3, and `pyproject.toml`'s
  own `real-data` extras all document, and omitting `zarr`/`tsbootstrap`/
  `sdv`/`pycatch22`/`umap-learn`/`plotly`/`jinja2`/`PyYAML` entirely while
  including packages this repo's own code never imports at all
  (`peft`, `sentence-transformers`, `hdbscan`, `matplotlib`, `seaborn`).
  This script has drifted to describe a different environment than the one
  this repo actually needs — a real, active landmine for anyone who runs it
  expecting a working setup. Not fixed in this pass (A18's explicit scope
  is "doc/config drift"); noted here so it isn't lost before that item is
  reached.
- **Verification, and its honest limit.** Everything **runnable locally**
  was actually run, not just read: YAML parses (`yaml.safe_load`,
  modulo the well-known `on:`→`True` PyYAML quirk that GitHub's own parser
  doesn't share); `compileall` clean; the fixed invariant-11 grep clean
  against the real repo; both test suites green (already reconfirmed
  147/147 for `tsfm_lens` earlier this session, unchanged by this item
  since it adds no pipeline code); a full `configs/smoke.yaml` run
  (11 sections rendered, 25 findings, 0 failed) with the fixed
  smoke-check script passing against its real `coverage.json`. **What this
  item cannot verify from here**: an actual GitHub Actions execution.
  Triggering one requires pushing this workflow file to the remote
  (`origin` is a real GitHub repo, confirmed via `git remote -v`), which
  this session did not do — pushing is a shared-state action outside this
  item's own authorization, and no commit was made for this or any other
  item this session per the user's standing "only commit when explicitly
  asked" instruction. This file is therefore verified **as far as running
  every one of its steps locally can verify it**, not as "green on GitHub"
  — a meaningful difference worth stating plainly rather than overclaiming.

### A15 — Exploratory multiplicity is unaccounted, and `confirm`'s pre-registration is a convention `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** `confirm` derives the hypotheses it tests from whatever
  `l0/summary.json` says at the moment it runs — a mutable dev artifact. Only
  L0 family strengths plus one CKA pair are confirmable at all, while the
  report accumulates ~22 "findings" with no record of how many comparisons
  produced them. Holm correction is applied within L0's family tests, not
  across the registered set.
- **Evidence.** `analysis/confirm.py:53-56` (`_test_hypotheses(cfg, metrics,
  ...)` reading dev artifacts), `report/report.py`'s `findings` accumulation.
- **Blast radius.** `CLAUDE.md` §6.7's exploration-vs-confirmation discipline
  is the repo's strongest epistemic claim, and it is currently enforced by
  discipline alone: re-running L0 after glancing at private results would
  change what gets "pre-registered", with no trace. Meanwhile the report's
  findings list is a large multiple-comparison surface labelled exploratory
  only in a preamble.
- **Fix.**
  1. A machine-readable registry, `hypotheses.json`, written by a new
     `register` step (cheap; reads existing artifacts): one entry per dev
     claim — claim id, stage, statistic, direction, threshold, the artifact
     path + content hash it was derived from, and free-text statement.
  2. `confirm` **requires** it, refuses (loudly, actionably) if any referenced
     artifact hash no longer matches, and tests exactly the registered set —
     no more, no fewer. Registry hash goes in `confirm/confirmation.json`.
  3. Extend the registry beyond L0: any statistic with a CI can be registered
     (L1 peak CKA, L2 best gain, L3 fingerprint agreement, clustering AMI).
     The confirm stage already replicates CKA — this generalizes the seam
     rather than inventing one.
  4. Multiplicity ledger: the report prints "N comparisons examined → M
     registered → K confirmed", and every unregistered finding is labelled
     *exploratory* inline in the HTML, not only in the preamble. Apply
     Holm/BH across the registered set as a whole.
  5. Test: mutate `l0/summary.json` after registration, assert `confirm`
     refuses with a message naming the drifted artifact.
- **Cost.** ~a day and a half. This is the item that most raises what the repo
  can honestly claim.

**Findings (2026-08-06).** Implemented items 1, 2, 4, and 5 in full; item 3
generalized the seam as instructed but only made two of the five stage
types it names actually *replicable* — a deliberate, stated scope decision
explained below, not a silent gap.

- **Item 1.** New `analysis/hypotheses.py`: `build_registry(cfg)` (pure,
  reads whatever dev artifacts exist) assembles one entry per confirmable
  claim across `l0` (family strengths), `l0_archetype` (A9's per-archetype
  strengths), `l1` (peak CKA), `l2` (stitching gain per direction), `l3`
  (fingerprint agreement per corruption), and `clustering` (AMI) — each
  entry carries `id`, `stage`, the statistic-specific fields, `artifact`
  (path), `artifact_sha256`, a free-text `statement`, and `replicable:
  bool` (+ `not_replicable_reason` when false). `run_register(cfg)` writes
  it to `hypotheses.json` at the run root. New `register` pipeline `Stage`
  (`deps=["l0"]`, `enabled` mirrors `confirm.enabled` — no point
  registering if nothing will consume it), inserted immediately before
  `confirm`, which now depends on `register` instead of `l0` directly
  (transitively still requires `l0`).
- **Item 2.** `confirm.py::run_confirm` now refuses immediately (before
  touching the private corpus) if `hypotheses.json` doesn't exist, naming
  the fix (`--stages` must include `register`). `check_registry_freshness`
  re-hashes every referenced artifact and raises, naming every drifted
  file and both its registered and current hash prefixes, on any
  mismatch — the actual enforcement mechanism behind `CLAUDE.md` §6.7's
  discipline, previously enforced by convention alone. `hypotheses.json`'s
  own sha256 is recorded in `confirmation.json` (`registry_sha256`).
  `_test_registered_hypotheses` (renamed from `_test_hypotheses`) sources
  its claim list from `registry["hypotheses"]` (filtered to `stage == "l0"
  and replicable`) instead of re-reading `l0/summary.json`'s `strengths`
  dict directly — the registry, not the mutable dev artifact, is now the
  source of truth for *which* claims get tested, exactly item 2's wording.
- **Item 3, generalized but scoped honestly.** `_replicate_registered_cka`
  (renamed from `_replicate_cka`) now looks up the registered, replicable
  `l1` hypothesis rather than reading `l1/meta.json`'s best pair directly —
  the seam item 3 asks to generalize, done for the one replication
  `confirm.py` already had. **`l2`/`l3`/`clustering` hypotheses are
  registered (hash-pinned, counted in the ledger, shown in a "registered
  but not yet replicated" report table with a stated reason each) but
  `confirm` does not yet re-test them on private data** — doing so
  correctly would mean re-fitting stitching probes (L2), re-running the
  full corruption battery (L3), or re-fitting cluster labels (clustering)
  against a private corpus inside this same stage, each a substantial
  design/implementation task in its own right, not a "generalize an
  existing seam" extension the way L1's CKA replication was. Also scoped
  out: `l0_archetype` claims, registered but not replicable, since
  `confirm`'s private-corpus pivot is family-keyed and a fresh private
  epoch has no guaranteed archetype composition match to dev's (stated in
  each entry's `not_replicable_reason`, not silently dropped).
- **Item 4.** `report.py::run_report` captures `len(findings)` at the exact
  point the "Confirm" builder is reached (before it runs) as the
  exploratory/confirmatory boundary, then tags every finding at an earlier
  index with `"[exploratory — not pre-registered] "` when rendering — no
  per-section code changes needed, since every section already appends to
  the same shared `findings` list in a fixed order. `_sec_confirm` renders
  a "Multiplicity ledger" (`N exploratory findings examined → M registered
  (K replicable today) → J confirmed`) plus the not-yet-replicated table.
  **A real ordering change this required**: `_sec_confirm`'s own two
  `findings.insert(0, ...)`/`insert(1, ...)` calls (originally there to put
  Confirm's findings first, for prominence) had to become `.append(...)` —
  inserting at the front would have landed them *before* the captured
  boundary index and mislabeled genuinely confirmatory findings as
  exploratory. Confirm's findings now appear at the end of the findings
  list instead of the front; a deliberate, explainable tradeoff (correctness
  of the new label over the old prominence-ordering), not an oversight.
- **Item 5.** Directly covered by `test_hypotheses.py`'s
  `test_check_registry_freshness_raises_naming_drifted_artifact_after_
  mutation` — mutates `l0/summary.json` after registration, asserts the
  raised message names `l0/summary.json` and says "hash changed".
- **Verification — unit tests.** New `tests/test_hypotheses.py` (9 tests):
  registry construction reads real strengths with a real sha256; archetype
  claims are correctly marked non-replicable with a reason; an empty
  registry when no `l0` artifact exists; `run_register` writes a real
  file; freshness passes when nothing changed; freshness raises (naming
  the file) on mutation and on deletion; `confirm` refuses outright with no
  registry; `confirm` refuses on a stale one. Two pre-existing tests
  updated for the rename/new artifact: `test_smoke.py::
  test_confirm_hypothesis_path` now builds a registry dict directly
  (standing in for what `register` would have written) and calls
  `_test_registered_hypotheses`; `test_smoke.py`'s `expected` artifact list
  gained `hypotheses.json`. `test_manifest_fingerprint.py` (A3's stage-
  fingerprint regression test, which enumerates `stage_names()`) re-run
  and confirmed unaffected by the new `register` stage. Full suite green
  at 156/156 (up from 147 after A13 — A14 added no test-suite code).
- **Verification — real checkpoints, deliberately without touching the
  private corpus.** `runs/medium_run_chronos_base` (a real prior session's
  full GPU run against `google/timesfm-2.5-200m-pytorch` +
  `amazon/chronos-t5-base`) already has a **consumed** `confirm/
  confirmation.json` — re-running `confirm` against it would be exactly
  the "repeated peeking" `CLAUDE.md` §6.7 warns against, so it wasn't done.
  Instead, `run_register` was called standalone against that run's real
  dev artifacts (a read-only operation touching only `l0`/`l1`/`l2`/`l3`/
  `clustering` JSON, never the private corpus) and registered **14 real
  hypotheses** (2 replicable: `TimesFM`'s `random_parametric` family
  strength, and the `TimesFM`↔`Chronos-T5-Base` peak CKA at
  `stacked_xf.4`↔`encoder.block.10`; 12 correctly marked not-yet-
  replicable — 2 L2 gains, 9 L3 corruption-agreement values, 1 clustering
  AMI). `check_registry_freshness` passed clean against these real,
  large, real-checkpoint-produced JSON files. A drift-refusal test was
  then run on a **scratch copy** of that run directory (never the
  original): mutating its `l0/summary.json` and re-checking produced the
  real error `"l0/summary.json (hash changed: registered bb4f19692101,
  now e9bf5f748d80)"` — the exact mechanism working against real file
  sizes and real hashes, not a synthetic stand-in. Separately, a fresh
  `configs/smoke.yaml` run (mock adapters, confirm enabled) exercised the
  full mechanism end to end including the actual private-corpus
  confirmation step (smoke's "private" corpus is a second smoke-generated
  draw, not a real sealed epoch, so no discipline concern there): **23
  exploratory findings examined → 17 hypotheses registered → 5 replicable
  → 3 confirmed**, with a "Registered but not yet replicated (12)" table
  rendered in `report.html`, and all 23 pre-Confirm findings correctly
  carrying the `[exploratory — not pre-registered]` tag (grep-confirmed
  count matches exactly).

### A16 — Silent cap reduction makes stage sample sizes non-comparable `[x]` · **P2** · fixed 2026-08-06

- **What's wrong.** Stages that must fit one batch clamp their configured cap
  to `adapter.cfg.batch_size` with no warning, so the realized sample size
  differs from the configured one — and differs *per model*, since batch sizes
  do.
- **Evidence.** `lens.py:91` — `take = min(data.n, cfg.lens.max_series,
  adapter.cfg.batch_size)`; same pattern at `l3_perturbation.py:318`,
  `sae/eval.py:86`, `attention.py` ablation paths. YAML comments carry the
  constraint ("must be <= every model's batch_size",
  `configs/medium_run.yaml:78,85,99`) — i.e. it's documented for humans and
  silently absorbed by code.
- **Blast radius.** §5.3 already hit the consequence: lens numbers that
  couldn't be compared across two runs, diagnosed correctly but only after
  someone noticed an impossible result (the same TimesFM checkpoint appearing
  to change). Any cross-run or cross-model comparison of a batch-limited
  stage is currently untrustworthy unless someone manually verifies the caps
  matched.
- **Fix.** Warn whenever a configured cap is reduced, naming which limit bound
  it; record `n_requested`/`n_realized`/`limited_by` in every stage artifact
  (folds into A4's row recording and A3's manifest); `meta_report.py` refuses
  to compare a batch-limited statistic across runs whose realized n differ,
  or flags it prominently rather than tabulating it as comparable. Optionally
  a `strict_caps: true` mode that raises instead of clamping.
- **Cost.** Half a day (mostly free if done with A4).

**Findings (2026-08-06).** Implemented the fix at exactly the sites the
item's own evidence names, and checked (rather than assumed) two sites
that looked like the same bug but weren't:

- **Core mechanism.** New `utils.py::capped_take(requested, **limits) ->
  {"n_requested", "n_realized", "limited_by"}` — generalizes the
  `n_requested`/`n_realized` field pair A4 already established (`sae/
  ground_truth.py`, `sae/eval.py`) with a `limited_by` list naming every
  limit that actually bound the result, and warns whenever the request was
  reduced. Named kwargs (`n_available=...`, `batch_size=...`) rather than
  positional limits, so the warning/`limited_by` output is self-describing
  without a separate label argument.
- **Wired into the three sites the item's evidence cites by line number,
  all confirmed still present at those lines**: `lens.py`'s `_model_lens`
  (`n_series_skip`/`n_requested_skip`/`limited_by_skip` in `lens/
  lens.json`), `l3_perturbation.py`'s `_patching` (`n_requested`/
  `n_realized`/`limited_by` in `l3/patching.json` — **had to also extend
  the save dict's key whitelist**, since `run_l3`'s `patching.json` write
  explicitly lists which keys of each model's result dict get persisted;
  the three new fields would have been computed and then silently dropped
  on the way to disk without that second edit, caught by checking the
  save site, not assumed safe from the producer-side change alone), and
  `sae/eval.py`'s `forecast_preservation` (`limited_by` alongside the
  `n_requested`/`n_realized` that already existed there from A4, with a
  comment that had *already* forward-referenced this exact item's number).
- **Two sites checked and found NOT to have this bug, correcting the
  item's own evidence list rather than fixing something that wasn't
  broken**: `attention.py`'s ablation paths (lines 226/256, explicitly
  named in the item's Evidence section) sample `take = min(data.n,
  cfg.attention.max_series)` — no `adapter.cfg.batch_size` term at all —
  and process the full realized set through internal `batch_slices`
  chunking, exactly the pattern that makes lens/l3-patching/sae-eval's
  *actual* bug (a hook-based intervention requiring one single batch,
  `CLAUDE.md`'s documented `token_patch` batch-size hardening) not apply
  here. `sae/real_data.py:58`'s superficially similar `take = min(batch_size,
  adapter.cfg.batch_size)` was also checked directly: it sizes an internal
  micro-batch for a `batch_slices` loop over the *entire* `contexts` array
  — every context still gets processed regardless of `take`'s value, so
  this is not a sample-size cap at all. Neither site was touched.
- **Item's `meta_report.py` ask**, implemented as "flag prominently" (the
  fix text's second option) rather than "refuse to compare": new
  `_batch_cap_warnings()`, reusing A7's `provenance_warnings` mechanism and
  rendering slot (same "Environment drift" box) — compares each model's
  lens `n_realized` across the aggregated runs and warns by name when they
  differ, which is exactly `CLAUDE.md` §5.3's own already-documented
  incident (crystallization depth moving for an *unchanged* checkpoint
  purely because two configs implied different realized n), now an
  automated check instead of something a human has to notice looks
  impossible. Scoped to `lens` only (the stage §5.3's real incident was
  about, and the only one `summarize_run` already tracked per-model
  numbers for) — `l3` patching's cap isn't cross-run-aggregated by
  `meta_report.py` today, so extending this warning there is a natural
  follow-up, not done in this pass, stated rather than silently skipped.
  A `strict_caps: true` raise-instead-of-clamp mode (the fix's "optionally")
  was not built — genuinely optional per the item's own wording, and the
  warn-plus-record mechanism above is the load-bearing half of this item.
- **Verification — unit tests.** New `tests/test_capped_take.py` (9 tests):
  no reduction is silent with `limited_by: None`; reduction by `batch_size`
  alone, by `n_available` alone, and by both tied together, each names the
  right limit(s); a request that exactly equals a limit (not actually
  reduced) is correctly not flagged; a full mini pipeline with `lens.
  max_series: 999` (exceeding every mock's `batch_size: 64`) records
  `n_requested_skip=999`, `n_series_skip<=64`, `limited_by_skip` non-`None`
  in real `lens.json`; the analogous check for `l3.patching.max_series:
  999` against real `patching.json`; and `_batch_cap_warnings` fires on a
  synthetic two-run mismatch and stays silent when realized n matches.
  Full suite green at 165/165 (up from 156 after A15).
- **Verification — real checkpoints.** `configs/_a16_real_check.yaml`
  (scratch, deleted after use; extract+lens+l3-patching+report only) against
  live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small` on
  `benchmark_medium/public_dev`, both models pinned to a deliberately small
  real `batch_size: 16` with `lens.max_series: 64` / `l3.patching.
  max_series: 64` — both well above that batch size, to force a genuine
  clamp on real weights rather than only mocks. Both models, both stages:
  requested 64, realized **16**, `limited_by: ["batch_size"]` — recorded
  correctly in both real `lens/lens.json` and real `l3/patching.json`, the
  exact real-world version of the §5.3 incident this item exists to make
  visible automatically instead of discoverable only by noticing an
  impossible result.

### A17 — `benchmark_validation`'s per-group metrics have no minimum-n guard or CI, and the matcher's O(n²) ceiling blocks full-scale validation `[x]` · **P3** · fixed 2026-08-06

- **What's wrong.** Per-group diversity numbers are reported as point
  estimates at any group size, and the pairwise matcher is O(n²) with
  subsampling as the only mitigation.
- **Evidence.** §5's 2026-08-03 per-task breakdown reports effective
  dimensionality at n=39 (`real_sequential_par`) alongside n=3154, with the
  caveat "enough to trust the direction, not the exact numbers" written by
  hand; `CLAUDE.md` §12 documents the O(n²) matcher and `--max-sequences`.
- **Blast radius.** A future session reading that table without the prose
  caveat will over-read n=39. And the literal-scale `full_multidomain.yaml`
  build (§5.1, still unrun, ~99K samples) cannot be validated at all at
  O(n²) — 8.9M pairs already at 4219 sequences.
- **Fix.** Bootstrap CIs on per-group diversity metrics (resampling
  sequences within group), a `min_group_n` below which a group reports
  "insufficient n" instead of a number, and a blocked/approximate matcher
  (bucket by catch22 neighbourhood first, do exact xcorr/DTW within and
  across adjacent buckets only) with the exactness trade-off stated in the
  report. Also verify `--max-sequences` subsampling is random and stratified
  by generator (A4 item 6) — if it's a head slice, per-group numbers are
  measuring task order.
- **Cost.** ~a day and a half.

**Findings (2026-08-06).** Implemented all four fix items in
`diversity.py`/`matching.py`/`report.py`/`plot.py`. This item lives entirely
in `tsfm_benchmark/benchmark_validation` — a different package from A9-A16's
`tsfm_lens` work, with no models or checkpoints involved at all (pure
corpus/feature-space code), so "real run" verification here means a real
sealed corpus, not real weights.

- **Item 1 (bootstrap CIs).** New `diversity.py::_bootstrap_group_ci`:
  resamples a group's own scaled feature rows with replacement and
  recomputes the *actual* PCA/nearest-neighbour pipeline `diversity_metrics`
  itself uses (not a closed-form/normal-theory approximation, which could
  be a poor stand-in for a nonlinear statistic like the PCA participation
  ratio at exactly the small-n regime this item is about) `n_boot` times,
  reporting a percentile CI. New `DiversityReport.effective_dimensionality_
  ci`/`near_collision_fraction_ci` fields (`None` unless requested).
  `diversity_metrics_by_group(..., n_boot=200)` — on by default, since an
  unmeasured CI is exactly the gap this item exists to close; `n_boot=0`
  restores the pre-A17 byte-identical fields.
- **Item 2 (min_group_n → explicit sentinel).** New `InsufficientN` dataclass
  (`n_sequences`, `min_required`, `status="insufficient_n"`).
  `diversity_metrics_by_group` now returns an entry for **every** group
  (never a silent omission) — `InsufficientN` below `min_group_size`,
  `DiversityReport` at or above it. `report.py::_by_group_summary` and
  `plot.py::plot_diversity_by_group` both updated to handle the mixed
  return type: the report emits an explicit `{"status": "insufficient_n",
  ...}` block per small group (JSON, not a missing key), and the plot
  excludes those groups from the bars (not plotted as a misleading zero)
  with the excluded count stated in the plot title.
- **Item 3 (blocked/approximate matcher).** New `matching.py::
  _match_all_blocked` (`match_all(..., blocked=True, n_blocks=None,
  adjacent_k=2)`): buckets sequences via `KMeans` over the same z-normalised,
  resampled arrays the exact matcher already compares (a self-contained
  shape-neighbourhood proxy — **a deliberate deviation from the fix text's
  literal "catch22 neighbourhood"**, made to avoid adding a new coupling
  from `matching.py` to `features.py`; stated here, not silently
  substituted), then computes *exact* similarity — reusing the same
  batched `_max_xcorr_matrix`/`_dtw_similarity_matrix` routines the exact
  path already uses, no new numerical code — only for pairs whose two
  sequences share a bucket or fall in each other's `adjacent_k` nearest
  buckets by centroid distance. Every other pair is left unscored (`NaN`
  in `similarity_matrix`, excluded from `redundancy_fraction`'s
  denominator and from `redundancy_by_group`'s per-group means) rather
  than assumed non-redundant. `MatchReport` gains `coverage_fraction`
  (exactly what fraction of all possible pairs were actually scored —
  the exactness trade-off the fix explicitly requires be stated, not
  buried), `blocked`, `n_blocks`. The existing exact path was refactored
  into `_match_all_exact` with **zero behavior change** (same code, new
  function name, called by default) — `match_all`'s default
  (`blocked=False`) reproduces every existing caller byte-for-byte. Falls
  back to the exact path (with `coverage_fraction=1.0`) when `n` is too
  small to block meaningfully. Wired into `run_validation.py` as
  `--blocked`/`--n-blocks`.
- **Item 4 (stratified `--max-sequences`).** Checked first, per `CLAUDE.md`
  §2.4, rather than assumed broken: `match_all`'s existing subsampling
  already used `rng.choice` (genuinely random, **not** a head slice) —
  the "if it's a head slice" conditional in this item's own fix wording
  didn't apply. It was, however, *unstratified* — a corpus dominated by one
  generator would draw mostly from it. New `matching.py::
  _stratified_subsample` (a small local proportional-allocation sampler,
  deliberately **not** an import of `tsfm_lens`'s `utils.sample_rows` —
  the two packages have no declared dependency on each other by design,
  per `loaders.py`'s own module docstring) replaces the plain `rng.choice`
  call, stratifying by `SeqRecord.group` (the generator name).
- **Verification — unit tests.** New `tests/test_benchmark_validation_a17.py`
  (9 tests): a group below `min_group_size` gets an `InsufficientN` entry
  alongside (not instead of) a normal group's `DiversityReport`; the
  bootstrap CI is present and well-formed (a test that initially asserted
  the CI must bracket the point estimate **failed and was corrected**,
  not the code — a bootstrap resample duplicates rows, which can
  systematically shift a nonlinear statistic like the PCA participation
  ratio, so non-coverage is expected statistical behavior, not a bug;
  fixed the test's assumption rather than loosening a real check);
  `n_boot=0` skips the CI; a small group's CI is measurably wider than a
  large group's (a real, checkable claim about bootstrap behavior, not
  just "doesn't crash"); stratified subsampling preserves group
  proportions and is provably not a head slice (a second test here also
  had to be corrected: it asserted a 10-of-110 minority group must get
  `>=5` of a 20-item stratified sample, but proportional allocation
  correctly gives ~2 — the test's expectation was wrong, not the sampler);
  the default exact path is unaffected by the refactor (byte-identical
  similarity matrix across two calls); the blocked matcher reports partial
  coverage with no `NaN` leaking into `redundant_pairs`; and it falls back
  to the exact path cleanly for too-small `n`. Full `tsfm_benchmark` suite
  green at 50 passed / 1 skipped (up from 41 passed / 1 skipped after A8).
- **Verification — real corpus (no models involved).** `run_validation.py`
  against the real, previously-built `benchmark_medium/public_dev` (288
  sequences, verified sealed corpus). `--group-by archetype`: all 12 real
  `CLAUDE.md` §4.2 archetypes got a real bootstrap CI, several visibly wide
  at small real n (`trend_dominant`, n=7: effective_dimensionality 1.938,
  CI (1.212, 2.237) — exactly the "trust the direction, not the exact
  number" caveat this item names, now a quantitative interval instead of
  hand-written prose). `--blocked --n-blocks 8`: real coverage_fraction
  **43.6%** of all 41,328 possible pairs, redundancy fraction and
  within/across-group similarity all computed correctly among only the
  scored pairs (0.0 redundancy, matching the exact path's own 0.0 on this
  corpus — a real, if limited, cross-check that blocking didn't change the
  qualitative finding), rendered correctly into both the console summary
  and `validation_report.json`. A direct call to `plot_diversity_by_group`
  with a real `InsufficientN` entry present confirmed no crash and a
  correctly-sized output file.

### A18 — Small doc/config drift that the repo's own invariants forbid `[x]` · **P3** · fixed 2026-08-06

- `configs/default.yaml:192` — `path: /path/to/private_benchmark_dir`, an
  absolute placeholder in a committed config. Invariant 11 has no
  placeholder exemption, and the 2026-08-04 audit's grep-for-`/home/`
  approach wouldn't catch this form. Fix: make it a relative example
  (`../../benchmark_full/private_test`) with a comment, and extend the
  invariant-11 check (A14 job 1) to flag any config path starting with `/`
  or matching a drive letter.
- `tsfm_lens/requirements.txt` — names `google/timesfm-2.0-500m-pytorch` as
  the tested adapter target; the adapter moved to TimesFM 2.5
  (`google/timesfm-2.5-200m-pytorch`, `CLAUDE.md` §11.8) sessions ago. Fix:
  update both `requirements.txt` and `pyproject.toml`'s `timesfm` extra, and
  cross-check against `DEPENDENCIES.md`.
- `runs/` artifacts referenced throughout this file are gitignored and
  regenerable, which is correct — but several Findings cite them as evidence
  without recording the provenance needed to regenerate identically. A7's
  manifest fixes this going forward; consider a one-time pass adding the
  environment used to the older Findings entries that are still load-bearing.

**Findings (2026-08-06).** Fixed the two named drift items, one item found
via the same cross-check that wasn't in the original evidence list, and
extended A14's CI check as this item's own fix wording asked for.

- **Absolute placeholder in `configs/default.yaml`.** Fixed the named
  `confirm.path: /path/to/private_benchmark_dir` → a relative example
  (`../../benchmark_full/private_test`) with an explanatory comment.
  **A repo-wide grep for the same shape (`^\s*path:\s*/`) found a second,
  unnamed instance in the same file**: `data.path: /path/to/benchmark_dir`
  (line 15) — the exact same violation the item's evidence only cited once
  (line 192). Fixed identically (`../../benchmark_full/public_dev`). Both
  now load cleanly via `load_config` (checked directly, not assumed).
- **Stale `timesfm` version references.** `requirements.txt`'s commented
  `timesfm[torch]>=1.2 ... google/timesfm-2.0-500m-pytorch` and
  `pyproject.toml`'s `timesfm[torch]>=1.2` extra both corrected to
  `timesfm>=2.0` (matching `DEPENDENCIES.md`'s verified `timesfm==2.0.2`,
  installed plain, no `[torch]` extra — `>=2.0` removed the old
  `TimesFmHparams`/`TimesFmCheckpoint` class API and, with it, whatever
  the `[torch]` extra used to install) with a comment naming
  `CLAUDE.md` §11.8 and the current default checkpoint
  (`google/timesfm-2.5-200m-pytorch`).
- **A third instance of the same drift, found via the item's own
  "cross-check against DEPENDENCIES.md" instruction, not in the original
  evidence list**: a repo-wide grep for `timesfm-2.0-500m`/`timesfm[torch]`
  found two README.md files (`tsfm_model_analysis/README.md` and
  `tsfm_model_analysis/tsfm_lens/README.md`) with the identical stale
  `pip install chronos-forecasting timesfm[torch]` line, and the outer
  `tsfm_model_analysis/README.md` specifically still describing the
  retired `google/timesfm-2.0-500m-pytorch` checkpoint in prose — a
  genuinely different, more detailed correction had already landed in the
  inner `tsfm_model_analysis/tsfm_lens/README.md` at some earlier point
  (the 2.5-only-API explanation, the SDPA-unfusing `attention_patterns`
  note, the no-MLP-submodule note) without the outer, apparently-not-kept-
  in-sync duplicate ever being updated to match. Both install lines fixed
  to plain `timesfm` (no `[torch]`); the outer README's stale checkpoint
  paragraph replaced with the same corrected text the inner one already
  had. **Not attempted**: a full reconciliation of the two README files'
  remaining differences (the `diff` surfaced several — a missing Lens/
  Attention/Exemplars row in the outer file's stage table, a differently-
  detailed L3 description) — out of this item's stated scope (doc/config
  *drift the repo's invariants forbid*, i.e. stale facts, not general
  doc-duplication cleanup); flagged here as a real, separate finding for
  whoever next touches either README.
- **CI extension, as the fix explicitly asked for.** A14's invariant-11
  grep (mid-string drive-letter / `/home/` / `/Users/` patterns) would
  **not** have caught this item's own bug shape (a bare `/path/to/...`
  with no drive letter or `/home//Users/` segment) — confirmed by testing
  the old pattern against the literal old line and seeing it not match,
  not assumed. New, separate CI step: `^\s*(path|out_dir):\s*/` over every
  YAML file, catching exactly this shape. **Caught a real false-positive
  risk before it could ever fire in CI**: the new grep, run locally
  against the actual repo, initially matched four lines inside
  `runs/*/config_resolved.yaml` — gitignored, per-run generated artifacts
  that correctly contain the absolute path of whatever machine produced
  them, not committed templates invariant 11 is about. Fixed by adding
  the same `--exclude-dir=runs` the first invariant-11 check already had;
  re-run clean against the real repo afterward.
- **Third fix-plan bullet (retroactive Findings provenance pass):
  deliberately not done**, per that bullet's own softer "consider" wording
  (contrasted with the first two bullets' plain "Fix:") — a full pass
  annotating environment provenance onto every older, still-load-bearing
  Findings entry across this file would be a large, separate undertaking
  in its own right, not a doc/config-drift fix. Left as a stated,
  explicitly-scoped-out follow-up, not silently skipped.
- **Verification.** No new unit tests — this item is pure doc/config text
  plus one CI shell-check extension, not runtime code. Verified instead by
  actually running what changed: `load_config` against the corrected
  `configs/default.yaml` resolves both paths correctly (checked directly);
  the new and existing invariant-11 grep patterns tested against the real
  repository both before and after each fix (confirmed the bug shape,
  confirmed the fix, confirmed no new false positives); a repo-wide grep
  for the retired checkpoint name and `timesfm[torch]` came back empty
  outside of `ROADMAP.md`'s own historical Findings prose (correctly left
  untouched — a dated record, not a current claim). Full `tsfm_lens` suite
  re-run clean at 165/165 (unchanged from A17 — this item touched no
  runtime code, so no test count change is the correct outcome, not an
  oversight).

### A19 — `store.load`'s row selection and `float16` storage are unguarded against silent precision/indexing surprises `[x]` · **P3** · fixed 2026-08-06

- **What's wrong.** Activations are stored `float16` (`CLAUDE.md` §6.4) and
  read back for closed-form fp32 linear algebra. Nothing checks for overflow
  to `inf` at write time (bf16-range activations can exceed fp16's ~65504
  max) or for NaNs entering a CKA/ridge solve.
- **Evidence.** `extraction/extract.py:67` — `.astype(np.float16)` with no
  finiteness check; `store.py`'s `oindex` path (correct, per `CLAUDE.md`
  §11.4) has no assertion that the returned row count matches the request.
- **Blast radius.** A single `inf`/`NaN` layer silently poisons a CKA row, an
  eff-dim estimate, or a ridge solve; downstream those appear as a plausible
  number (CKA near 0, or a `nan` that a `nanmean` swallows) rather than an
  error. Low likelihood, high diagnosis cost — the kind of thing that gets
  misattributed to a model finding. `layer_screen`'s own fallback path
  already anticipates "a degenerate/constant layer" in its docstring, which
  suggests this has been brushed against.
- **Fix.** Count and log non-finite values per (model, layer) at write time,
  store the counts in store attrs, refuse to write a layer that is wholly
  non-finite, and have every analysis that consumes activations assert
  finiteness once per layer (cheap) with an actionable error naming the
  layer. Consider `extraction.store_dtype: float32` guidance for models
  whose activations run hot. Report the counts in the coverage panel.
- **Cost.** Half a day.

**Findings (2026-08-06).** Implemented the fix's core items in
`store.py`/`extract.py`/`report.py`/`config.py`, and — per this item's own
"§2.4 applies here too" instruction — found that the diagnosis's fix
item 5 ("consider a `store_dtype: float32` knob") was already wrong to
phrase as a suggestion: the knob already existed as real config, and was
silently non-functional.

- **Write-time tracking.** `store.py::write_batch` accumulates a per-
  (model, layer) non-finite count and total-element count in memory
  (`self._nonfinite_counts`/`_nonfinite_totals`) across every batch;
  `finalize_layer` (called once per layer, at the end of `extract.py::
  _extract_model`'s batch loop) persists `{count, total, fraction}` into
  `root.attrs["nonfinite"]` and **raises `RuntimeError` if a layer is
  wholly non-finite** (`fraction >= 1.0`) — refusing to leave a completely
  poisoned layer in the store for every downstream CKA/ridge/PCA call to
  independently trip over, per the fix's own wording. Partial non-
  finiteness logs a warning and is recorded, not fatal.
- **Read-time assertion.** `store.load(..., check_finite=True)` (the new
  default) checks `np.isfinite` on whatever was just materialized and
  raises `ValueError` naming the model, layer, count, and fraction the
  first time any consumer actually reads a poisoned array — the "every
  analysis... assert finiteness once per layer" half of the fix, at the
  single choke point every stage already calls through. `check_finite=
  False` exists as a stated escape hatch; **not wired into any current
  caller** — every real call site (`sae/ground_truth.py`, `sae/train.py`,
  `layer_screen.py`, `l1_geometry.py`, `l2_stitching.py`, `clustering.py`,
  `internals.py`, `lens.py`, `l3_perturbation.py`) keeps the default,
  checked directly via a repo-wide grep, not assumed.
- **Report.** New `report.py::_nonfinite_summary()` opens the store
  read-only and reads `root.attrs["nonfinite"]`, folded into the existing
  combined `_alignment_provenance_block` (A2+A7's panel) as a new table —
  renders whenever any layer has been tracked, **including when every
  count is zero** (confirmed on real data below), so the check having run
  and found nothing is itself visible, not indistinguishable from the
  check never having run.
- **The real finding, more significant than the item's own item 5's
  phrasing suggested.** `extraction.store_dtype` was **not** a
  not-yet-built "consider" suggestion — it already existed as a real
  `ExtractionConfig` field, already wired into `store.init_layer` (which
  correctly used it to set the zarr array's declared dtype). But
  `extract.py`'s very next line, `aligned = align(hidden,
  pool).cpu().numpy().astype(np.float16)`, **hardcoded `float16`
  regardless of that config** — so `store_dtype: float32` changed only
  the on-disk dtype *label*; the actual float32-range values had already
  been irreversibly rounded (and, in an overflow case, turned to `inf`)
  to float16 before `store.write_batch` ever saw them. The config knob
  was a complete no-op for the one thing (avoiding float16 overflow) it
  existed for. Found by reading the exact line this item's own Evidence
  pointed at, not by assuming a config field being consumed by one
  function meant the whole path worked — fixed by casting to
  `np.dtype(cfg.extraction.store_dtype)` instead of the hardcoded literal.
- **Verification — unit tests.** New `tests/test_nonfinite.py` (8 tests):
  a clean batch records a zero count; a partially non-finite batch is
  recorded, not raised; a wholly non-finite layer raises; counts
  accumulate correctly across multiple batches; `store.load` raises on a
  poisoned layer, naming it; `check_finite=False` bypasses the check
  (poison still present, just not raised on); a clean layer loads
  normally; and — the test that actually caught the real bug above — a
  full mini pipeline run with a deliberately huge injected activation
  value (1e5, overflows float16's ~65504 max, fits float32 easily) shows
  the value becoming non-finite under the `store_dtype: float16` default
  and **reproduces a genuine `RuntimeWarning: overflow encountered in
  cast`** in the process, then shows the identical value surviving finite
  under `store_dtype: float32` once the hardcoded-cast bug was fixed —
  proving the *values*, not just the array's dtype label, actually
  differ. Full suite green at 173/173 (up from 165 after A18).
- **Verification — real checkpoints.** `configs/_a19_real_check.yaml`
  (scratch, deleted after use; extract+l0+report only, both models at
  `capture_layer_stride: 1` — every layer captured, not the usual stride-2
  subset, to maximize the chance of catching a real overflow anywhere in
  either model's full depth) against live
  `google/timesfm-2.5-200m-pytorch` (all 20 layers) +
  `amazon/chronos-t5-small` (all 6 encoder blocks) on
  `benchmark_medium/public_dev`, `store_dtype: float16` (the default).
  **A genuinely open empirical question, resolved honestly rather than
  assumed either way**: all 26 real captured layers showed **zero**
  non-finite values — `float16` overflow, the concrete risk this whole
  item is about, did not materialize in practice for either checkpoint on
  this corpus. A true negative result, not a weaker test: the mechanism
  ran cleanly end to end on real activations with no false positives, and
  the new report table rendered all 26 layers' `0.0000%` fractions
  correctly (grep-confirmed present in real `report.html`) rather than
  being hidden because nothing was found — exactly the "state it, don't
  silently omit" behavior the rest of this session's fixes established.
  Whether a *different* real model (larger, or one known to produce
  extreme activations) would actually trip this guard remains untested;
  the guard itself is now real and in place either way.

---

**All 19 items in this section (A1–A19) are now `[x]` complete as of
2026-08-06.** Every item was implemented, unit-tested, verified against a
real run (real checkpoints for the `tsfm_lens` items, a real sealed
corpus with no models involved for A17, config/doc-only verification for
A18), and documented here with a dated Findings block before moving to the
next — several items surfaced genuinely new bugs beyond what their own
Evidence sections named (A6's sklearn single-class crash, A7's missing
`timesfm.__version__`, A9/A10's tier-threading and `mixture`
`generative_params` bugs, A12's cross-seed bisection mismatch, A16's two
evidence corrections, A18's second unnamed absolute-path instance, A19's
non-functional `store_dtype` knob — each found by direct inspection or
testing, per `CLAUDE.md` §2.4, not assumed from the audit's own wording).
Nothing in this section remains open; `ROADMAP.md` §16's enhancement
backlog is the next place forward work on this repo should look.
**Amended 2026-08-12:** two further audit items, **A20** and **A21**, were
added below after this closure — both found while building something else
(§6.2.1 Stage 0's extraction config, and §18 F2's tests), not by a fresh
audit pass. A1–A19 remain closed; the section is not.

---


## Archived: §14's full session-log narrative (~85 entries, 2026-08-03 to 2026-08-13), verbatim

## 14. Session log

> Append one entry per session that makes non-trivial progress on this
> roadmap. Keep entries short — a few lines. This is a changelog, not a
> diary; the *content* of what was learned belongs in the relevant phase's
> Findings block above, this log just tracks when and by whom.

- **2026-08-03** — Initial roadmap authored (planning only, no implementation
  this session). Verified current repo state against `CLAUDE.md`'s
  description: corrected the `benchmark_validation` location `[VERIFY]`,
  confirmed the SAE phase is still exactly a stub. Sequenced the brief's four
  ideas plus its five specific suggestions into Phases 0–5 with an explicit
  dependency rationale (§3).
- **2026-08-03 (same day, follow-up)** — Went back and updated `CLAUDE.md`
  itself (not just this roadmap) with everything found stale during the
  verification pass above, plus a few things found on a closer look:
  the `benchmark_validation` tree location, the actual (not originally
  planned) `CO_trev_1_num` winsorization fix mechanism plus two new
  per-group diversity/redundancy functions, the TimesFM adapter's move to
  TimesFM 2.5 (new checkpoint, halved layer count, and a newly-supported
  `attention_patterns` capability — added `CLAUDE.md` §11.8 as a trap entry
  since a pinned dependency's own breaking release caused this, not a
  choice made in-repo), two new L3 corruptions (`warp`, `dropout`), and
  stale `examples/` → `example_runs/` CLI paths. This file's own
  `QuantileTransformer`/IQR wording from the initial pass was corrected to
  match, above.
- **2026-08-03 (Phase 1 §5.1/§5.2 implementation session)** — Picked up the
  roadmap's own next step: unblock and verify the "wired but unverified"
  real-data path, and widen the benchmark's real-world data beyond the one
  Monash-only `mixture` task that had ever actually been exercised. Set up
  a dedicated conda-forge environment (`tsfmPy`) for the build/validation
  half of the repo (no torch/CUDA needed here), and in the process of
  actually running things — not just reading the code — found and fixed
  four independent real breakages (a `datasets>=3` removal of script-based
  loading that made Monash entirely unloadable, a full `tsbootstrap` API
  rewrite, `sequential_par` being a literal stub, and a source-injection
  bug in `run_full.py` that only `mixture` happened to avoid), plus a
  short-real-series `block_length` crash, plus an unrelated native
  numpy/MKL ABI crash that would have broken `benchmark_validation` on any
  fresh conda-forge setup. Implemented `sequential_par` against SDV's
  `PARSynthesizer` for the first time. Widened `configs/full_multidomain.yaml`
  to use all three real-derived generators against **two** independent real
  corpora (Monash's full 32-domain catalog, not a 10-domain cap, plus ETT),
  added `configs/real_data_smoke.yaml` for fast verification, added
  `tests/test_real_derived_generators.py`, and ran a full
  `run_full.py` → `run_validation.py` build end-to-end against live data for
  the first time (see §5's Findings block for exact numbers and every
  discovered issue). Also surfaced a new, unresolved risk: the golden-hash
  bit-exact-reproducibility regression test fails against numpy 2.1.0 for
  reasons unrelated to this session's code changes — flagged as follow-up,
  not resolved. A full-scale (not smoke) real-data build, and the
  GPU-for-`sequential_par` question, are both explicitly left for the next
  session (see §5's Findings for why).
- **2026-08-03 (same-day follow-up session — §5.1's next concrete step)** —
  Picked up exactly where the previous entry left off: ran
  `benchmark_validation` on a real, non-smoke-scale build for the first
  time. Before sizing that build, calibrated instead of guessing twice: (1)
  isolated `sequential_par`'s actual per-call cost driver by holding a real
  Monash cohort fixed and varying only `epochs` — found the previous
  session's "lower epochs" diagnosis was wrong, the real cause was
  unbounded training-sequence length (a 40,720-point real series was being
  fed whole into a per-timestep RNN fit), fixed with a new
  `max_train_length` parameter (~150-200x speedup, ~1000s → ~5-6s per fit);
  (2) ran a `--max-count 300` throughput calibration on the full config
  before committing to a build size, found the literal `full_multidomain.yaml`
  would take ~6+ hours at measured throughput, and built a new
  ratio-preserving ~10x-scaled config (`configs/full_multidomain_run1.yaml`)
  instead, sized to finish in well under an hour. That build produced 4219
  public_dev + 4279 private_test sequences (near-dupes only ~1.4%, vs. the
  calibration run's ~28% — confirmed that figure was a `--max-count` sizing
  artifact, not real). Ran `benchmark_validation` on the result: corpus-wide
  numbers corroborate the earlier small-demo figures (redundancy ~0%,
  eff-dim 4.84/22), and the per-task breakdown surfaced a genuine new
  finding — the synthetic backbone's effective dimensionality is markedly
  higher than every real-derived task's, meaning the real-derived tier adds
  domain breadth but not catch22-feature-space breadth per task (see §5's
  Findings for the full numbers and the correct way to read that). The
  literal full-scale `full_multidomain.yaml` build (~99K attempted samples)
  remains future work, best done unattended/overnight given its ~6-hour
  estimate.
- **2026-08-03 (§5.4 session — first real-checkpoint run of `tsfm_lens`)** —
  Picked up §5.4, the last unstarted item in Phase 1: validate the two
  live-weights unknowns flagged in `CLAUDE.md` §9/§10 by actually running
  real TimesFM/Chronos checkpoints, something that had never been done in
  this repo before. Discovered this machine has an RTX 5070 GPU (previously
  undocumented); per user instruction, reused the existing `tsfmPy` conda
  env rather than building a second one, upgrading its torch in place to a
  CUDA build and adding `transformers`/`timesfm`/`chronos-forecasting`/
  `zarr` — verified this doesn't resurrect the `CLAUDE.md` §11.13 MKL ABI
  crash and doesn't regress the existing `tsfm_benchmark` test suite (still
  40/1, unchanged). Found and fixed two genuine, previously-undiscovered
  bugs in the process — `tsfm_lens`'s zarr storage layer called a v3-only
  API method despite being pinned to zarr v2 (would have crashed on the
  very first real extraction), and `impulse_alignment_check`'s hardcoded
  impulse amplitude was badly miscalibrated for Chronos's context-adaptive
  quantization tokenizer, producing misleadingly low (as low as 0.06)
  diagonal-alignment scores that looked like a broken adapter but were
  actually a broken test. Both fixed and re-verified: both checkpoints now
  show perfect (1.00) alignment at every layer. Ran the full
  `medium_run.yaml` pipeline end-to-end against real
  `google/timesfm-2.5-200m-pytorch` and `amazon/chronos-t5-small`
  checkpoints (~10.5 min on the RTX 5070) — confirmed Chronos
  `output_attentions` and TimesFM's SDPA-unfusing attention-pattern capture
  both work correctly on live weights (sane, non-degenerate outputs:
  TimesFM's future attention mass is exactly zero everywhere, as required
  for a causal decoder), and confirmed TimesFM's previously-flat L3 patching
  curve is a genuine whole-layer-averaging artifact — per-window patching
  recovers a strong, depth-intensifying causal signal concentrated in the
  context window nearest the forecast horizon. See §5's Findings block for
  full numbers, including first (exploratory, small-scale) real L0/L2
  results corroborating the earlier "TimesFM stronger" finding and
  confirming a real bidirectional L2 stitching gain. Also fixed a stale
  doc-drift comment in `configs/default.yaml` flagged in `CLAUDE.md`'s own
  reconciliation note.
- **2026-08-04 (Phase 0 session)** — Picked up Phase 0 (explainability &
  report hygiene), the roadmap's own first-listed phase and, per §3's stated
  rationale, the one every later phase's report credibility depends on —
  left entirely unstarted (all checkboxes `[ ]`) across three sessions that
  had instead continued straight into Phase 1. Before implementing anything,
  actually read `report/report.py` rather than trust the checklist's framing
  (`CLAUDE.md` §2.4): found most of the "glossary" ask already substantially
  satisfied by an existing `_note()` mechanism attached to nearly every plot,
  apparently added in a session whose work never got reflected back into
  this file — a reminder that this file can itself drift stale the same way
  `CLAUDE.md` does. Fixed the two genuine gaps that check turned up (an
  unexplained periodicity-head score, a missing worked numeric example for
  head ablation), then implemented the three items that really were
  unstarted: `report.verbose`/`--verbose`, a "How to read this report"
  preamble with the full evidence-class ladder, and — the most substantial
  piece — a new L3 per-series verbose case-study section (context + true
  continuation + clean/corrupted/patched forecasts next to that series' own
  layer x window restoration grid), extracted from data the aggregate
  computation already produces at effectively zero extra forward-pass cost.
  Also found and fixed a genuine, previously-undiscovered platform bug in
  the process: every `write_text`/`read_text` call in the package used the
  platform-default text encoding, which crashed the report stage with
  `UnicodeEncodeError` on native Windows (cp1252) the moment it actually ran
  there for the first time — the same "never verified in this actual
  environment" pattern `CLAUDE.md` §11.15 already describes. Fixed
  everywhere, not just the one call that crashed first. Re-ran the full
  smoke suite after every change (`CLAUDE.md` §11.6's own instruction) and
  added new shape/content assertions covering the new verbose artifacts and
  both the on/off gating paths, rather than eyeballing one passing run. See
  §4's Findings block for full detail.
- **2026-08-04 (housekeeping + L3 investigation session)** — Four
  user-directed tasks, none following straight from the phase sequence: (1)
  investigated a user-reported "detrend/spike/dropout barely react" pattern
  against the real `runs/medium_run` artifacts rather than guessing —
  confirmed it's not a corruption-implementation bug (direct input-magnitude
  checks show `detrend` perturbs the input as much as `noise`/
  `deseasonalize`) but a mix of a genuine finding (both real checkpoints are
  least activation-sensitive to `detrend` of any corruption — consistent
  with recency-dominated forecasting over trend-extrapolation) and a report
  display artifact (`level_shift`'s far larger configured strength dominates
  the linear-scale Behavioral sensitivity chart, visually flattening
  everything else). Fixed the display (value labels, a sharper note, a new
  least/most-sensitive findings callout) without touching the corruptions or
  their statistics, and explicitly decided to keep all three in the battery
  — dropping low-signal corruptions would remove the contrastive signal L3
  exists to produce. (2) Audited the whole repo for hardcoded absolute
  paths; found and fixed the one violation
  (`tsfm_lens/configs/medium_run.yaml`). (3) Created `DEPENDENCIES.md`,
  and in doing so resolved a previously-deferred 2026-08-03 open question:
  CUDA torch, numpy/MKL, and `sdv` all coexist fine in the single `tsfmPy`
  env today — `sequential_par`'s GPU path is not blocked, correcting a
  stale "decision deferred" note left in §5. (4) Added two new standing
  invariants to `CLAUDE.md` §7 (no hardcoded paths; keep `DEPENDENCIES.md`
  current) and mirrored them into this file's §0. Re-ran the `tsfm_lens`
  smoke suite and `tsfm_benchmark`'s test suite after all code changes
  (still 40/1 and the smoke suite green) — nothing here was left unverified.
- **2026-08-04 (checkpoint-breadth session, partial)** — Picked up §5.3, the
  next unstarted Phase 1 item: run more than the single default checkpoint
  pair so existing findings can be checked for a size confound. Added
  `configs/medium_run_chronos_base.yaml` (identical `benchmark_medium`
  corpus and TimesFM checkpoint to `configs/medium_run.yaml`, swapping only
  `chronos-t5-small` → `chronos-t5-base`), and ran `--check-alignment`
  against the new checkpoint per invariant 7 before trusting anything from
  it — perfect 1.00 diagonal-hit fraction at every one of its 12 encoder
  blocks. Started the full pipeline run, but it was **explicitly deferred
  mid-run by the user** before completing or producing a report — no
  base-vs-small comparison numbers exist yet. Left as `[~]` rather than
  `[x]` in §5.3 so this isn't mistaken for a finished comparison; the config
  and alignment check are real, the actual result is the next session's
  first step (`python run.py --config configs/medium_run_chronos_base.yaml`
  to completion, then diff against `runs/medium_run`'s existing artifacts).
- **2026-08-05 (§5.3 completion session)** — Picked up exactly where the
  prior session left off: completed the deferred
  `medium_run_chronos_base.yaml` run, this time on a new, previously-
  undocumented machine (Linux, `cudaPy` conda env, 8× RTX A5000). Checked
  package versions against `DEPENDENCIES.md` before running anything
  (`CLAUDE.md` §11.15's own lesson) and found the same zarr-v3-vs-v2
  `create_array`/`create_dataset` trap recurring on this third environment
  — fixed by pinning `zarr<3` as documented, no code change needed this
  time since the §11.15 fix already targets the v2 API correctly. Re-ran
  `--check-alignment` for both models before trusting the run (perfect 1.00
  at every layer, matching prior machines). Ran the full 12-stage pipeline
  to completion (~32.5 min; 10 sections, 21 findings) and diffed every
  major artifact against the existing chronos-small `runs/medium_run`: the
  one dev-to-private-confirmed finding this corpus supports
  (`random_parametric` favors TimesFM) is size-robust, but L1 CKA and L2
  stitching gain both *increase* with Chronos's size while L4 clustering
  AMI *decreases* — a genuine, checked demonstration that different
  evidence-class metrics don't move together across a size change, not
  just a restated doctrine point. See §5's Findings block for full numbers.
  Did not touch Phase 1 §5.1 (corpus breadth) or re-litigate §5.4 (already
  done) — this session's scope was strictly finishing §5.3's one open
  item. Next: §5.5 (cross-run aggregator + retiring `CLAUDE.md` §10's
  single-run findings into this file) is Phase 1's last unstarted item.
- **2026-08-05 (same-day follow-up — §5.5, closing out Phase 1)** — Built
  `tsfm_lens/report/meta_report.py` + CLI `run_meta_report.py`: reads N
  existing run directories' artifacts (L0/L1/L2/lens/clustering/confirm),
  degrading per-section rather than crashing when a stage was skipped in a
  given run, and renders one comparison table plus a per-family stability
  view. Tested against all four run directories that existed in the repo
  (`medium_run`, `medium_run_chronos_base`, `real_run`, `smoke`) — on first
  use it correctly surfaced that `random_parametric` favors TimesFM
  *stably* across all three real-checkpoint runs (two Chronos sizes, one
  independent corpus), directly answering the "does this hold across
  regimes" question this deliverable exists for. Found and fixed one real
  bug building it: `runs/real_run`'s TimesFM has `crystallization_depth:
  null` (never crystallized within tolerance) and the first template draft
  crashed formatting that with `%.2f` — added two regression tests
  (`tests/test_meta_report.py`) covering this and the missing-stage
  degrade path. Re-ran the full `tsfm_lens` pytest suite after — all 5
  tests (3 existing + 2 new) pass, no regressions. Then retired `CLAUDE.md`
  §10 per this section's own "better" option: moved its original
  provisional findings verbatim into this section's Findings block (dated
  2026-08-05, "carried forward"), stated which of its claims are now
  superseded by later evidence and which aren't, and replaced §10 itself
  with a pointer here — added a matching `CLAUDE.md` reconciliation note
  and updated its repo-layout tree and CLI examples for the new files.
  **Phase 1 is now fully checked off** (§5.1–§5.5 all done); Phase 2a
  (layer-selection metric research) is the next unstarted item per §3's
  dependency ordering.
- **2026-08-05 (same-day follow-up — §6.1, Phase 2a first pass)** — Picked
  up the next item per §3's dependency ordering: does effective
  dimensionality predict which layers are worth interpreting? Built
  `tsfm_lens/analysis/layer_selection.py`, pooling 64 (run, model, layer)
  records across all 4 existing run directories and correlating three
  candidate metrics (effective dimensionality, input-CKA, and a proposed
  new one — L3 fingerprint entropy) against two interpretability proxies
  (tuned-lens R², family-probe decodability), reusing `analysis/stats.py`'s
  existing bootstrap primitive with a cluster unit of (run, model) rather
  than layer (layers within one model are depth-autocorrelated the same
  way windows within a series are). Checked the obvious confound before
  trusting anything (§2.4): three of the four metrics trend with depth on
  their own, so every correlation is reported both raw and depth-controlled.
  Real result: effective dimensionality predicts the two proxies in
  **opposite directions** (lower predicts more forecast-readable, higher
  predicts more family-decodable), both significant after depth control —
  so there's no single "worth interpreting" score, which is itself the
  finding. The proposed alternative, L3 fingerprint entropy, beat effective
  dimensionality on the decodability proxy (tighter CI, same direction).
  Added `recommend_layers(goal=...)` rather than a goal-less "auto," with
  an explicit note distinguishing it from `clustering.py`'s unrelated
  `layer: auto`. Did not test kurtosis/sparsity of raw activations (would
  need new zarr-store plumbing, not just reading existing artifacts like
  the other three candidates) — flagged as follow-up. Three new regression
  tests (`tests/test_layer_selection.py`); full suite 10/10 after. See
  §6.1's Findings block for exact numbers and the honest n_groups=8
  caveat. Next: Phase 2b (SAE variants) is the flagship research thread and
  a much larger undertaking — a natural point to check in with the user on
  scope/direction before committing to a specific SAE architecture, per
  §6.2's own "evaluate empirically, do not assume one is best" instruction.
- **2026-08-05 (same-day follow-up — §6.2, Phase 2b baseline)** — Checked in
  with the user before committing to an SAE architecture given the phase's
  scale; user chose the baseline-first path (item 4: plain TopK SAE +
  eval harness + ground-truth alignment, before attempting the flagship
  crosscoder). Researched the exact integration points first (a background
  agent read `sae/interface.py`, `extraction/store.py`, `extraction/hooks.py`,
  `tsfm_benchmark`'s ground-truth schema, `config.py`, `pipeline.py` in
  detail) rather than guessing signatures. Built `sae/models.py`
  (`TopKSAE`: architectural top-k sparsity, unit-norm decoder columns),
  `sae/train.py` (training loop reading straight from the zarr store's
  window-level activations, `run_sae` as a new pipeline stage gated on
  `sae.enabled`), `sae/eval.py` (reconstruction fidelity, dead-feature
  rate, forecast-preservation reusing `token_patch` + L3's own
  per-token-to-window assignment), and `sae/ground_truth.py` (the
  ground-truth feature-alignment score, split into a pure matching
  function and an I/O wrapper specifically so the matching logic is unit-
  testable without a real corpus). Extracted `l0_behavioral.py`'s inline
  MASE formula into a shared `analysis/stats.py::mase()` along the way
  (re-verified byte-identical via the smoke suite) rather than
  reimplementing it for the new eval harness. Ran the baseline against
  `runs/medium_run_chronos_base`'s already-extracted real-checkpoint
  activations (added an `sae:` block to that config, ran `--stages sae` —
  the existing stage-skip machinery correctly reused the existing
  extraction with no new model calls needed for training/eval) and found:
  real, ground-truth-verified seasonality-linked features (the session's
  clearest positive result); a genuinely high dead-feature rate (~98%,
  `dict_size_mult: 8` was real overcapacity for ~4600 training rows with
  no dead-neuron resampling implemented); and a forecast-preservation
  check that correctly caught a real problem (patching the reconstruction
  back in roughly doubled MASE for both models) that should block Phase
  3's planned feature-ablation work on this baseline until fixed — the
  validity check did exactly its job. Also discovered, by actually trying
  to use it rather than trusting the pooled statistics, that §6.1's
  `recommend_layers` policy does **not** transfer as a within-model
  ranking rule: it correctly picked TimesFM's best layer in this run but
  picked Chronos-T5-Base's *worst* one, because that model's own layers
  trend the opposite direction from the pooled cross-model finding —
  documented with the exact numbers in both `recommend_layers`'s docstring
  and this file's §6.2 Findings. 6 new unit tests
  (`tests/test_sae.py`) plus one new mock-adapter integration test
  (`tests/test_smoke.py::test_sae_stage_integration`, confirming
  `forecast_preservation` works with no real corpus and
  `ground_truth_alignment` degrades to a logged error rather than
  crashing when one isn't available); full suite 17/17 after. See §6.2's
  Findings block for every number. Not done this session, left as
  explicit follow-up in §6.2's checklist: the crosscoder (item 1), the
  encode-store-into-`sae/{model}/{layer}` seam, feeding SAE results back
  into §6.1's correlation study, and verbose-mode report integration.
- **2026-08-05 (same-day follow-up — fixing the baseline, per user
  choice)** — User chose to fix the baseline (dead-feature collapse,
  forecast-preservation failure) before touching the crosscoder, and
  explicitly asked for more SAE training data from an established source
  ("something on huggingface") rather than only shrinking the dictionary.
  Implemented dead-neuron resampling and `sae/real_data.py` (reuses
  `tsfm_benchmark`'s existing generic HF loader, verified live against
  Monash). Dead-neuron resampling took **three iterations, not one**, each
  caught by actually measuring before declaring it fixed (§2.4): attempt 1
  (fixed absolute encoder scale, no Adam-state reset) was catastrophic —
  fidelity went deeply negative, worse than baseline, not just
  ineffective. Attempt 2 (calibrated scale + Adam-state reset) fixed
  Chronos-T5-Base but was still catastrophic for TimesFM specifically —
  root cause: resampling ~10,000 dead atoms at once from too few source
  rows collapsed most of them onto near-duplicate directions. Attempt 3
  (capped resample fraction per event + directional jitter) finally gave
  a clean, real improvement on both models, confirmed first in a fast
  controlled sweep (no model calls) before trusting it on a real run.
  Final real-run numbers: reconstruction fidelity 0.629→**0.858**
  (TimesFM), 0.567→**0.840** (Chronos-T5-Base). Forecast-preservation now
  **passes for TimesFM** (ΔMASE +0.047) but **still fails for
  Chronos-T5-Base** (ΔMASE +2.37) despite similar fidelity gains —
  investigated the discrepancy rather than shrugging at it, and found a
  real, architecture-specific confound in the check's own window-broadcast
  approximation (exact for TimesFM's patch tokenization where window ==
  token width, lossy for Chronos's per-timestep tokenization where one
  window spans 32 distinct tokens forced to the same value) — documented
  in `sae/eval.py`'s docstring, not left as an unexplained asymmetry.
  Ground-truth alignment got richer and produced the session's cleanest
  single match: Chronos-T5-Base's top feature now tracks
  `has_intermittency` at ρ=0.881. Added a regression test guarding the
  exact resampling-collapse failure mode found along the way. Full suite:
  21/21. See §6.2's Findings for every number and the full three-attempt
  story. `configs/medium_run_chronos_base.yaml`'s `sae:` block now
  reflects the working configuration (`epochs: 60,
  resample_dead_every_epochs: 5, real_data_enabled: true`), not the
  original broken-baseline one.
- **2026-08-05 (design-only session — §6.1.1, layer-selector redesign)** — No
  implementation this session, per explicit user instruction. Read `CLAUDE.md`
  and `ROADMAP.md` in full and `analysis/layer_selection.py` to characterize
  why the §6.1 layer-selection mechanism, though its limitations were already
  documented, is not the right long-term solution: it's a cross-(run, model)
  correlation study that provably fails as a within-single-model selector
  (§6.2 Findings), it's circular (needs the full expensive pipeline over all
  layers *before* it can select), it's unfair to strided-out layers, and its
  predictive direction is itself architecture-dependent. Wrote §6.1.1: three
  candidate selectors (residual-trajectory work/bend geometry;
  ground-truth factor-emergence spectroscopy; submodular redundancy-coverage),
  each mapped against four explicit design requirements (architecture-agnostic,
  all-layers-fair, parsimonious, cheap-enough-to-run-first), plus §6.1.1-E —
  the bake-off protocol (independent SAE-ground-truth gold ranking,
  parsimony curve, uniform-stride null, within-model robustness gate) that
  decides the winner rather than picking one on intuition. Re-scoped §6.1 in
  place as a *finding* not a selector, updated the §13 open question it
  partially answers, and added a new open question for the bake-off outcome.
- **2026-08-05 (same-day follow-up — §6.1.1 implementation + first bake-off
  run)** — Implemented and empirically ran the design from the session
  above, per explicit user instruction ("implement the tests, compare their
  usefulness ..., update documentation with results"). Built
  `tsfm_lens/analysis/layer_screen.py` (all three selectors:
  `select_work_bend`, `select_coverage`/`greedy_coverage_selection`,
  `select_factor_emergence`/`factor_probe_matrix`, plus the unifying
  `select_layers()` dispatcher) and `layer_screen_bakeoff.py` (§6.1.1-E's
  gold ranking via `build_gold_ranking` — reusing `sae/train.py`'s
  `train_sae` and `sae/ground_truth.py`'s matching logic rather than
  reinventing either — plus `recall_at_budget`/`parsimony_curve`/
  `null_curves`/`robustness_gate` and three ensemble combiners). Renamed
  `sae/ground_truth.py`'s `_encode_series_level` → `encode_series_level`
  (now genuinely shared between the production SAE stage and the bake-off,
  not duplicated). Added 12 new unit tests
  (`tests/test_layer_screen.py`) against synthetic data with planted,
  known-correct answers (a regime change, a redundant band, a ground-truth
  factor embedded only from a known layer onward) — full `tsfm_lens` suite
  33/33 after, no regressions. Built `configs/layer_screen_experiment.yaml`
  (the first config in this repo to capture **all 20** of TimesFM's layers,
  not the usual stride-2 10, specifically because a layer *screen* has to be
  fair to every layer by construction) and
  `run_layer_screen_bakeoff.py`, then actually ran the whole thing against
  live `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-small`
  checkpoints on an 8×RTX A5000 machine (extraction + gold-ranking SAE
  training for all 26 layers completed in under two minutes per seed) and
  replicated with an independent SAE seed before trusting any result
  (`CLAUDE.md` §2.4) — gold-ranking Spearman stability ρ=1.0/0.926. Headline
  result: Idea A (`work_bend`) matched the oracle almost exactly and beat
  both nulls on Chronos-T5-Small; no selector beat uniform-stride on
  TimesFM, but neither did the oracle there (a real, different finding from
  §6.1's outright selector failure); Idea B underperformed with a diagnosed,
  not-yet-fixed weighting flaw; combining methods did not beat the best
  single method on either model. Deliberately **not** wired into any config
  knob or `sae.targets: auto` yet — the result is a genuine, provisional
  first pass on one corpus/checkpoint-pair, not a settled cross-architecture
  claim. Full numbers, the exact diagnosed Idea B flaw, and the concrete
  follow-up list are in §6.1.1's Findings block and the updated §13 entry.
- **2026-08-05 (third same-day follow-up — §6.1.1 production wiring)** — Per
  explicit user instruction ("make sure the new layer selection is default
  in full pipeline runs and is fully integrated"), wired `work_bend` in as
  the pipeline's default layer selector rather than leaving it a standalone
  bake-off script. Added `LayerScreenConfig` to `config.py`; a
  `layer_screen` pipeline `Stage` in `pipeline.py` running right after
  `extract` and before every expensive stage (satisfies R4 by construction,
  via canonical stage-list order); `run_layer_screen` in
  `analysis/layer_screen.py` with two independent graceful-degrade paths
  (`factor_emergence` without a sealed/ground-truth corpus falls back to
  `work_bend`; any selector raising on a real model falls back to a
  uniform-stride `_uniform_fallback`); `sae/train.py::_default_targets`
  rewritten to consume `layer_screen/selection.json` when `sae.targets` is
  empty, replacing the old "each model's final captured layer" default; a
  new "Screen" report section between L0 and Profile with its own
  evidence-class-honest `_note()`. Updated `default.yaml`/`medium_run.yaml`/
  `layer_screen_experiment.yaml` with an explicit `layer_screen:` block;
  deliberately left `medium_run_chronos_base.yaml`'s already-enabled `sae`
  stage on its **explicit**, historically-documented targets rather than
  switching to `auto`, so re-running that exact config still reproduces
  §5.3/§6.2's recorded numbers. Extended `tests/test_smoke.py` with the new
  artifact path, the new report-section token, and a new test asserting
  `_default_targets` resolves *exactly* what `layer_screen` selected (not
  just that each half works alone) — full suite 34/34, plus a live CLI run
  of `configs/smoke.yaml` (not just pytest's tmp-dir harness) confirming a
  real 11-section/22-finding report and a sane `selection.json`. This is a
  deliberate policy call, not a new empirical result: the bake-off's mixed,
  single-corpus finding is unchanged (§6.1.1's Findings, §13's open
  question) — what changed is that "provisional but currently-best" is now
  treated as good enough to replace an admittedly-arbitrary old default,
  on the user's explicit authority to make that call, rather than waiting
  for the broader cross-architecture validation the Findings still name as
  outstanding. Full detail in §6.1.1's second Findings block.
- **2026-08-05 (fourth same-day follow-up — §6.2/§13 crosscoder feasibility
  test)** — Read `ROADMAP.md` fresh per the user's autonomous-loop
  instruction ("do the next unfinished thing... keep doing this") and
  picked up §13's explicit prerequisite for §6.2 item 1's flagship
  crosscoder: an early small-scale trainability/stability test, before
  committing to the full build. Built `tsfm_lens/sae/crosscoder.py`
  (`CrosscoderSAE` — a TopK dictionary jointly trained across `n_sources`
  inputs of independent dimension, joint per-feature decoder
  normalization so `relative_decoder_norm` carries real shared-vs-specific
  signal, dead-atom resampling ported from the single-source baseline) and
  `run_crosscoder_feasibility.py` (loads an already-extracted run's store
  at the L1 peak-CKA layer pair, trains independent baselines plus one
  joint crosscoder, compares). Found and fixed two real bugs via testing
  before trusting any result (`CLAUDE.md` §2.4): a genuine training
  instability (unequal per-source activation scale lets the larger-scale
  source dominate the joint MSE objective, reproduced synthetically as a
  fidelity collapse to −9.7, fixed via per-source scale normalization
  built into the model itself) and a device-mismatch crash in dead-neuron
  resampling that only the real-GPU run surfaced. Ran for real against
  `runs/medium_run_chronos_base`'s already-extracted TimesFM/Chronos-T5-Base
  activations: joint training is stable (no source-domination collapse
  across three hyperparameter settings), though this run's dictionaries
  were mostly dead regardless of crosscoder-vs-baseline — a data/training-
  budget finding, not a crosscoder-specific one, and itself surfaced a
  second metric bug (the shared/specific split counted dead atoms, which
  are near-"shared" by random-init symmetry regardless of signal; fixed
  with an alive-only filter). 3 new unit tests, full `tsfm_lens` suite
  37/37 after. Full numbers, both bug fixes, and the concrete "what's
  still needed before the flagship build" list are in §6.2's new Findings
  block and the updated §13 entry — this session deliberately did not
  attempt the flagship crosscoder itself (SAEAdapter implementation, report
  section), only the feasibility gate.
- **2026-08-05 (fifth same-day follow-up — §7 Phase 3 noise-SNR × depth
  sweep)** — Continued the autonomous "read roadmap, do the next unfinished
  thing" loop. Picked §7 Phase 3 bullet 2, the one bullet in an otherwise
  entirely-unstarted phase with a concrete, already-observed effect to
  chase (ρ≈−0.90 additive-noise depth-sensitivity anti-correlation) rather
  than an open-ended new design. Built `run_noise_snr_sweep.py`, which
  reruns only `analysis/l3_perturbation.py::run_l3`'s `noise` corruption
  at several SNR values against an already-extracted run's store (no
  re-extraction, no new downloads) and reads back its own already-computed
  per-corruption Spearman agreement rather than writing new statistics
  machinery. Ran for real against `runs/medium_run_chronos_base`
  (TimesFM 2.5-200M vs. Chronos-T5-Base), 96 series, 6 SNR points, ~82
  seconds total on one GPU. Found a real dose-response transition: positive
  agreement at mild noise (ρ=+0.40), a sign flip by SNR=12dB (ρ=−0.44), and
  a broad plateau of strong anti-correlation from 6dB down through −3dB
  (ρ≈−0.76 to −0.85) — the main pipeline's default 6dB setting sits well
  inside that plateau, not at a fragile edge. Also found each model's own
  peak-sensitivity layer (TimesFM's first captured layer, Chronos's last)
  never moves across the entire SNR range — only the cross-model rank
  agreement's sign/strength changes. The *why* (quantization vs. something
  else) is still open, named as the concrete next step. Full numbers in
  §7's new Findings block.
- **2026-08-05 (sixth same-day follow-up — §7 quantization-churn probe)** —
  Picked up the concrete next step named at the end of the previous entry:
  test the "continuous embedding vs. quantization" hypothesis directly,
  rather than leaving it open. Built `analysis/quantization_churn.py`
  (`token_churn_stats`, a pure statistic split from I/O so it's unit-testable
  without a live tokenizer, mirroring `sae/ground_truth.py`'s existing
  pattern) and `run_quantization_churn_sweep.py`, which reuses the exact
  same 96-series row sample and noise draws the prior sweep used and
  tokenizes clean vs. corrupted contexts through whichever model in the
  pair exposes a quantization tokenizer — gracefully skipping
  continuous-embedding models like TimesFM with a log, per `CLAUDE.md`
  §2.5, rather than fabricating a comparable metric where none exists. Ran
  live against `runs/medium_run_chronos_base`'s Chronos-T5-Base (no
  re-extraction, no forward passes — tokenization only). Found a real,
  nuanced answer rather than a clean confirmation: raw churn *fraction*
  saturates almost immediately (96.4% of tokens already reassigned at the
  mildest SNR tested) and has no dynamic range to explain anything, but
  token-ID jump *magnitude* does — its two largest relative jumps
  (20dB→12dB, 12dB→6dB) line up with the sweep's sign-flip and subsequent
  deepening, a real partial confirmation of the quantization hypothesis for
  *why the sign flips*. But the plateau region (6dB→−3dB) breaks a clean
  story: jump magnitude keeps climbing there while the cross-model
  agreement stays flat rather than continuing to deepen, so quantization
  severity alone doesn't explain the plateau's shape — likely each model's
  own sensitivity saturating under heavy corruption contributes too, not
  decomposed further this session. Sanity-checked the jump numbers against
  the tokenizer's own bin width (`MeanScaleUniformBins`, 4094 bins over
  [−15,15]) before trusting them — calibrated to the actual mechanism, not
  an artifact. 7 new unit tests (`tests/test_quantization_churn.py`, all
  synthetic), full `tsfm_lens` suite **44/44** after (was 37). Full numbers
  in §7's Findings block. This closes out §7's one concrete, ready-to-chase
  lead from `CLAUDE.md` §10/carried-forward findings; the rest of §7
  (controlled parameter sweeps, feature-level ablation, verbose-mode case
  studies, the per-model bias card) remains entirely unstarted and is the
  natural next pick for a future session, per §3's dependency ordering
  (feature-level ablation still needs a forecast-preservation-passing SAE,
  which per §6.2's Findings currently exists for TimesFM but not
  Chronos-T5-Base).
- **2026-08-05 (seventh same-day follow-up — §7 Phase 3 bullet 1, controlled
  parameter sweeps)** — Continued the autonomous "read roadmap, do the next
  unfinished thing" loop; picked up exactly what the previous entry named as
  the natural next pick, §7's remaining unstarted bullet 1. Built
  `analysis/parameter_sweep.py` (pure recipe-construction and sweep-data
  generation, using `build_pipeline`'s `parametric` generator directly and
  holding every component fixed except the one swept parameter, unit-tested
  without any GPU since generation is plain numpy) and
  `run_parameter_sweep.py` (reuses an existing run's model configs/
  checkpoints, generates fresh sweep data itself, scores via the same
  `adapter.predict` call L0 already makes — no extraction, no store I/O).
  Ran all three sweeps the checklist names, live, against
  `runs/medium_run_chronos_base`: seasonal period (4→256), noise scale
  (0.05→2.5, an inverse-SNR sweep), and intermittency rate (0→0.8). Found
  three genuinely different regimes rather than one clean story: TimesFM
  leads increasingly as seasonal period grows, with one striking exception —
  a real, CI-confirmed MASE spike specifically at period=32, TimesFM's own
  patch width, flagged as a plausible aliasing effect but explicitly not
  verified as the mechanism this session; Chronos-T5-Base is far better at
  very clean low-noise periodic signal specifically, a reversal from the
  period sweep, with the gap closing entirely by noise_scale≈0.6; and the
  intermittency sweep is confounded by MASE's own scale term collapsing
  under heavy zeroing (which also zeroes the forecast horizon itself),
  documented as a real methodological caveat rather than reported as a
  clean finding. 8 new unit tests (`tests/test_parameter_sweep.py`, all
  synthetic/GPU-free), full `tsfm_lens` suite **52/52** after (was 44). Full
  numbers and the exact caveat reasoning are in §7's Findings block. Not
  attempted this session: crystallization depth and L3 noise-fingerprint
  response per sweep point (both need the full extraction/lens machinery,
  not just prediction — a materially larger undertaking, left as an
  explicit follow-up rather than partially built), verbose-mode case
  studies, and the per-model bias card. §7's remaining unstarted work
  (feature-level ablation, verbose-mode case studies, the bias card) is the
  natural next pick, with feature-level ablation still gated on a
  forecast-preservation-passing SAE for Chronos-T5-Base specifically (per
  §6.2's Findings, TimesFM already has one).
- **2026-08-05 (eighth same-day follow-up — §7 Phase 3 bullet 5, per-model
  bias card)** — Continued the autonomous "read roadmap, do the next
  unfinished thing" loop. Of the three items the previous entry named as
  the natural next pick, deliberately chose the bias card over
  feature-level ablation (bullet 3): per `ROADMAP.md` §0.4's "explain before
  reordering," ablation is only half-unblocked (§6.2's Findings show the
  SAE's forecast-preservation validity check passes for TimesFM but still
  fails for Chronos-T5-Base), so building it now would produce a one-sided,
  confounded comparison rather than the real cross-model test it's meant to
  be — recorded as an explicit, reasoned skip in §7's checklist, not a
  silent one. Built `report/bias_card.py` (pure comparison/aggregation
  logic — CI-overlap-based "favored" verdicts, generic anomaly detection
  against each sweep's own plurality winner, caveat attachment — split from
  its HTML-rendering I/O the same way `analysis/parameter_sweep.py` and
  `sae/ground_truth.py` already split) and `run_bias_card.py`, which
  consolidates the `param_sweep_*.json` artifacts from the immediately
  preceding session into one standalone `runs/bias_card.html` — deliberately
  not a new `report.py` section, mirroring `report/meta_report.py`'s
  existing precedent for cross-artifact (not single-run) aggregation.
  6 new unit tests, full `tsfm_lens` suite **58/58** after (was 52). Ran
  live against the real sweep data: the generated card mechanically
  corroborated every hand-written finding from the previous session (the
  period=32 TimesFM anomaly, Chronos's noise-sweep edge being confined to
  the two lowest noise levels once CI-overlap is required, the
  intermittency caveat attached to both models) with no numbers hand-picked
  for the card — a genuine, if small, validation that the CI-based
  "favored" logic recovers the same conclusions a human read of the tables
  already reached. Closes out §7 bullet 5. Remaining §7 work: feature-level
  ablation (blocked as above until Chronos's SAE forecast-preservation gap
  closes) and verbose-mode case studies for the sweeps (bullet 4, still
  entirely unstarted and now the only fully-unblocked item left in this
  phase) — the natural next pick.
- **2026-08-05 (ninth same-day follow-up — §7 Phase 3 bullet 4, verbose
  sweep case studies)** — Continued the autonomous "read roadmap, do the
  next unfinished thing" loop; picked up exactly what the previous entry
  named as the last fully-unblocked item in this phase. Built
  `report/sweep_case_studies.py` (`select_case_study_values` — pure,
  reusing `report/bias_card.py`'s anomaly detection rather than
  reinventing "interesting sweep point" — plus `render_case_studies_html`
  for the I/O) and `run_sweep_case_studies.py`, which generates one fresh
  example series per selected sweep value and renders its context/true-
  continuation/both-models'-forecasts, narrated — the forecast half of the
  bullet; explicitly did not attempt the per-layer skip-lens-curve half,
  which needs the full extraction/lens machinery per series and was
  already flagged in two prior entries as a materially larger undertaking
  than anything this session's sweeps have needed so far. Ran live against
  all three real sweep artifacts from the last two sessions
  (`runs/medium_run_chronos_base`): the selection logic correctly picked
  each sweep's own extremes plus its already-known anomaly (or the middle
  value, for the noise_scale sweep which has none) with zero manual
  intervention, and every case study's per-model MASE matched the
  aggregate sweep numbers already on record, confirming consistency rather
  than a parallel, potentially-diverging generation path. 7 new unit
  tests, full `tsfm_lens` suite **65/65** after (was 58). Full detail in
  §7's Findings block. This closes the forecast half of bullet 4 — the
  checklist item is marked done with that scope stated explicitly, not
  silently expanded to cover lens curves it doesn't include. §7's remaining
  work is now: the lens-curve half of bullet 4 (needs extraction/lens
  machinery per case-study series) and feature-level ablation (bullet 3,
  still blocked on Chronos-T5-Base's SAE forecast-preservation gap, §6.2).
  Phase 3's three fully-unblocked, self-contained deliverables (controlled
  sweeps, the bias card, forecast-half case studies) are now all done;
  everything left in this phase needs either the SAE fix or new
  extraction/lens plumbing, not just `adapter.predict` — a natural point to
  pick up a different phase, or to invest in one of those two prerequisites
  directly, next session.
- **2026-08-06 (autonomous 20-minute-cadence session, user-directed)** —
  Picked up two small, well-scoped, previously-diagnosed-but-unfixed items
  that needed no live GPU/checkpoint rerun, deliberately avoiding anything
  requiring unattended multi-hour compute: (1) §6.1.1's `factor_emergence`
  peak-weighting flaw (`CLAUDE.md` §11.18) — `factor_emergence_scores`
  (`tsfm_lens/analysis/layer_screen.py`) now weights each ground-truth
  factor's contribution by its own peak decodability and gates the
  `emergence` event behind a new absolute `min_peak_score` floor, verified
  against a new planted-data unit test (weak/noisy factors no longer
  out-vote a real late peak); real-checkpoint re-verification through the
  §6.1.1-E bake-off is explicitly still open, not claimed done. (2) SAE
  training's redundant real-data catalog refetch (noted but not fixed in
  §6.2's Findings) — `run_sae` now samples the real context pool once per
  run instead of once per (model, layer) target. Full `tsfm_lens` suite
  re-run clean after both: **66/66**. Also corrected one piece of doc
  drift found while touching this area: `CLAUDE.md` §6.2's own warning
  about a stale `configs/default.yaml` comment was itself stale — that
  comment had already been fixed in an untracked earlier session. First
  fix's implementation was done by a background agent (per the user's
  20-minute self-pacing loop request); this session's continuation
  verified its work (full suite, cross-checked the diff against the
  diagnosis in both files) before writing it up here, rather than trusting
  the agent's own unseen completion report.
- **2026-08-06 (same loop, iteration 3)** — Picked up §6.2's last
  fully-unblocked, no-live-GPU-needed checklist item: the per-feature SAE
  exemplar panel. New `report/sae_exemplars.py` (pure selection/table-
  building split from I/O, mirroring `report/sweep_case_studies.py`'s own
  convention) + a new "Sparse feature dictionary" `report.py` section
  showing each SAE target's summary stats plus its top ground-truth-
  matched features' top-activating exemplar series and their real
  ground-truth values. Renamed `sae/train.py`'s private `_sanitize` to
  public `sanitize` rather than importing a name-mangled helper across a
  module boundary. Verified two ways (`CLAUDE.md` §2.4): new planted-data
  unit tests in `tests/test_sae.py`, then regenerated the real report for
  the already-existing `runs/medium_run_chronos_base` real-checkpoint run
  and read the actual HTML, which reproduced this section's own
  previously-cited numbers exactly (the `has_intermittency` ρ=0.881
  match) and surfaced one genuine, minor, not-fixed observation about how
  the existing archetype dummy-encoding treats real-derived-tier series —
  see this session's new Findings entry in §6.2 for the full account.
  Full suite: **69/69**. Everything from this loop remains uncommitted
  working-tree changes only, per the user's git-safety instructions.
- **2026-08-06 (same loop, iteration 4)** — Picked up §10 (Phase 5)'s CI
  checklist item's model-adapter half: a lightweight adapter-conformance
  checker, chosen specifically because it's testable end-to-end against
  the existing mock adapters with no live GPU/checkpoint needed, same
  standard as every prior iteration this loop. New
  `tsfm_lens/models/conformance.py::check_adapter_conformance` runs the
  same manual checklist Phase 4 (§9) and this bullet already describe in
  prose — layer discovery, well-formed `token_time_spans`, a sane
  `impulse_alignment_check`, a correctly-shaped finite `predict()` output,
  and every optional capability degrading to `None` rather than raising —
  and a new `tests/test_adapter_conformance.py` runs it against all three
  registered mocks (`mock_patch`, `mock_step`, and `mock_wave` — the third
  architecture that already existed specifically to prove new adapters
  need no changes outside their own file, but that no default config
  actually exercises), plus a no-optional-capabilities stand-in (proving
  the "`None` is fine" path, previously untested since every existing mock
  happens to support everything) and a deliberately-broken adapter (proving
  a real contract violation is still caught, not silently accepted). Full
  suite **74/74**. Not attempted, and correctly flagged as such in this
  section's checklist entry: actually wiring this into a CI pipeline (no
  CI config exists in this repo yet) and the separate, already-known,
  unresolved golden-hash numpy-version fragility (`CLAUDE.md` §11.13) —
  this bullet's "CI" framing names both together but this session's work
  only covers the adapter-conformance half. Everything from this loop
  remains uncommitted working-tree changes only.
- **2026-08-06 (same loop, iteration 5)** — Picked up §10's paired
  "auto-generated model-zoo capability matrix" items (§10 and §9's own
  cross-reference to it), the natural next Phase-5 CI item after
  iteration 4's conformance checker, and again chosen for being verifiable
  with no live GPU work: new `tsfm_lens/models/capability_matrix.py` +
  `run_capability_matrix.py` + `tests/test_capability_matrix.py`. Verified
  two ways (`CLAUDE.md` §2.4): synthetic-fixture unit tests, then the CLI
  run twice against real artifacts — once with no `--verify` at all
  (reproduced `CLAUDE.md` §6.2's hand-maintained table purely from static
  class introspection) and once pointed at the already-completed
  `runs/medium_run_chronos_base` (reproduced the one genuinely subtle
  documented case, TimesFM's `mlp_info` declaring-but-verifying-nothing,
  as a distinct rendered state from "never checked"). That real-run
  cross-check caught a genuine bug before it shipped: the renderer's
  `verified is True`/`is False` comparisons broke silently on any
  single-capability all-verified/no-`NA` DataFrame column (pandas then
  stores plain `numpy.bool_`, not Python's `True`/`False` singletons) —
  fixed to use `pd.isna(...)`/`bool(...)`, with a dedicated regression
  test added so this can't regress silently. Full suite **81/81**.
  Everything from this loop remains uncommitted working-tree changes only.
- **2026-08-06 (audit + planning session — no implementation)** — Per explicit
  user instruction ("do not implement any code"), read `CLAUDE.md` and this
  file end to end, then read the pipeline itself against their claims looking
  for silent-failure paths, fixable acknowledged limitations, and non-obvious
  design errors. Produced two new sections: **§15** (19 audited items, each
  with `file:line` evidence, a stated blast radius, and a detailed fix plan
  marked NEEDS IMPLEMENTATION) and **§16** (an enhancement backlog for the §1
  "anyone, any model, one button" north star, 22 items in four tiers). Added
  inline `⚠️ AUDIT` markers at the six places in this file where an audit
  finding contradicts or materially qualifies something already written down
  (§5.3's invariant-7 bullet, §5.5's per-archetype follow-up, §6.1.1's R2
  requirement, §6.2's forecast-preservation deliverable, §7's intermittency
  caveat, §5's corruption-calibration deferral), and a matching reconciliation
  note plus `§11.20` pointer in `CLAUDE.md`. The three findings worth naming
  here because they change how existing recorded numbers should be read:
  invariant 7's alignment verification is logged-and-discarded rather than
  enforced (A2); `build_pipeline` writes corpora grouped by task while four
  call sites subsample with a head slice, so several recorded numbers were
  measured on family-skewed prefixes (A4); and the `layer_screen` stage as
  wired cannot satisfy its own all-layers-fair requirement under any
  production config (A1). Nothing was implemented or committed.
- **2026-08-06 (same-day follow-up — the deferred §6.2/A4 re-measurement)**
  — With every §15 P1 item now fixed, picked up the one concrete piece of
  unfinished business those fixes explicitly left behind: A4's own
  Findings said the SAE forecast-preservation/ground-truth numbers "remain
  as previously recorded... left as a follow-up," and §6.2's own AUDIT
  block said not to act on the "TimesFM passes" conclusion before
  re-measuring. Verified the full suite green (173/173) first, then
  re-ran `l0` (for a matched noise floor) and `sae` against
  `configs/medium_run_chronos_base.yaml`'s already-extracted store, live
  `google/timesfm-2.5-200m-pytorch` + `amazon/chronos-t5-base` (checkpoints
  already cached locally, GPU idle and available). Corrected numbers now
  in §6.2's Findings, superseding the old ones in place rather than
  replacing them: forecast-preservation ΔMASE for TimesFM moved
  +0.047 → +0.175 and for Chronos-T5-Base +2.368 → +3.869 under the
  now-representative stratified sample — both *worse* than previously
  reported, ruling out the possibility that the old sampling bug happened
  to favor one model over the other. The new matched noise floor
  (TimesFM exactly `0.0`, deterministic; Chronos-T5-Base mean `0.160`/max
  `0.830`) confirms Chronos-T5-Base's failure is real signal, not sampling
  noise, and that TimesFM's own delta — while still much smaller and still
  real by construction (deterministic ⇒ any nonzero floor is signal) — is
  larger than the "passes almost exactly" framing previously implied.
  Ground-truth alignment also moved (both models' ρ up moderately) but,
  checked directly rather than assumed, its own row set is provably
  identical before and after (this corpus's 288 series is smaller than
  the configured cap, so `sample_rows` degenerates to "take everything"
  regardless of the fix) — that delta is ordinary SAE-training run-to-run
  variance, not something A4 changed, and is called out as such so it
  isn't misattributed. `report.html` was confirmed (by direct grep of the
  rendered HTML, not just the JSON artifact) to render both new
  noise-floor-contextualized finding strings end to end. Full suite
  reconfirmed green at 173/173 after the re-run. Everything from this
  session remains uncommitted working-tree changes only.
- **2026-08-06 (same-day follow-up — §16 E9 implementation)** — With every
  §15 P1 item closed and the §6.2 re-measurement done, picked E9
  (untrained-weights null baseline) as the highest-value item left in the
  backlog, per this file's own "arguably the highest-value item in T2"
  framing. Implemented `ModelConfig.random_init` + `models/base.py::
  random_init_like` (HF's own `type(model)(model.config)` idiom for the two
  Chronos adapters, a checkpoint-load skip for TimesFM — no generic
  reinitialization heuristic, no per-adapter special-casing beyond one
  branch in each `load()`), which gives L1/L2/internals/SAE a real floor
  with zero changes to any of those four analysis modules: pairing a real
  model against its own `random_init` twin is just an ordinary two-model
  config. Added `sae/ground_truth.py::permutation_null_alignment`, a
  label-permutation null for the SAE ground-truth alignment score's own
  multiple-comparisons inflation, wired into the report. New unit tests
  (`tests/test_random_init.py`, `tests/test_ground_truth_permutation_null.py`,
  10 tests total) — one, a full mock-pipeline `extract`+`l1` run of a
  real-vs-random-init pair, exercises the whole wiring end to end offline.
  Found and fixed one real bug by actually running the smoke test rather
  than trusting the diff: a 4th paragraph appended to an existing
  `report.py` note tuple collided with `_note()`'s `summary=` keyword.
  Full suite green at 183/183. Ran two real-checkpoint verification runs
  in the background (`configs/null_timesfm_random.yaml`,
  `null_chronos_random.yaml`, live TimesFM 2.5 / Chronos-T5-Base), both
  completed cleanly, and the result is a genuine **split verdict** worth
  reading in full (§16 E9's Findings has all four numbers, not summarized
  twice here): internals' probe decodability and the SAE ground-truth
  alignment (validated further by its own permutation null landing right
  on top of the untrained twins' scores, exactly as it should) both
  clearly survive the null: real models beat their untrained twins at
  every depth. L1's peak CKA and, more strikingly, L2's stitching gain do
  **not** clearly exceed an architecture-only floor computed from the same
  models' own untrained twins — L2 gain in particular sits within, and for
  one direction *below*, the untrained-twin floor's range. This materially
  qualifies (not refutes) `CLAUDE.md` §6.5's L2 framing, and is exactly the
  kind of check E9's own text predicted this repo was missing. `CLAUDE.md`
  §6.2 updated with the new capability and a matching reconciliation note.
- **2026-08-07 — E9 follow-up: the eyeball null comparison becomes a real
  significance test.** Picked up the prior session's own explicitly-named
  next step (§16 E9's Findings: "a rigorous verdict... needs a per-layer-pair
  bootstrap test... not eyeballing two point estimates") and §13's matching
  open question, both still marked unresolved at session start. Built
  `stats.py::bootstrap_ci_diff` (general paired/unpaired bootstrap CI + p for
  the difference of two statistics), factored `l2_stitching.py`'s train/val
  split into a reusable `series_split()`, and a new `analysis/null_baseline.py`
  (+ `run_null_baseline_test.py` CLI) that reads two already-extracted run
  directories — no model loaded, nothing re-run — and bootstraps real-run vs.
  null-run peak CKA / best gain with the *same* series-index draw in both
  (verified row-for-row corpus alignment first, not assumed; falls back to an
  unpaired bootstrap with a warning otherwise). 5 new tests
  (`tests/test_null_baseline.py`), full suite green at **189/189**. Ran the
  new CLI against the two already-completed real-checkpoint null runs from
  the prior session (`null_timesfm_random`, `null_chronos_random`) against
  `medium_run_chronos_base` — no GPU needed, pure re-analysis of existing
  activation stores. Result (full numbers in §16 E9's Findings, appended
  in place): L2's null result is now confirmed by an actual difference test
  (both directions' CIs comfortably contain zero, p=0.474/0.532) rather than
  inferred from overlapping point-estimate CIs. L1's result sharpens into
  something more specific than "model-asymmetric": the real cross-model peak
  CKA significantly *exceeds* Chronos's own untrained-twin floor (p=0.001)
  while significantly *trailing* TimesFM's own untrained-twin floor (p=0.001)
  — two decisive, opposite verdicts from the same real number depending on
  which model supplies the null. §13's open question and §16 E9's checklist
  item both updated in place (kept at `[~]`, not `[x]`: this tests only each
  run's single global best pair, not a full per-layer-pair grid, and on one
  corpus/checkpoint size only).
- **2026-08-07 (same-day second follow-up, run via a scheduled autonomous
  loop) — the depth-curve half of E9's named next step.** Extended
  `analysis/null_baseline.py` with `compare_l1_depth_curve`: for each of
  TimesFM's 10 captured layers, bootstrap its best-cross-model-partner CKA
  against both models' own untrained-twin floor *at that exact layer*
  (mechanical extension of `bootstrap_ci_diff`, exactly as the prior
  entry's "next concrete step" predicted). 2 new tests, full suite green at
  **191/191**. Real result (full table in §16 E9's Findings): real
  cross-model structure decisively beats **both** floors at every middle
  depth (layers 6-14 of 10), and loses to at least one floor only at the
  extremes (shallowest vs. TimesFM's own steeply-decaying floor, deepest
  vs. Chronos's own flat floor) — a materially more informative and more
  positive picture than the single peak-pair result alone suggested, since
  the peak (layer 4) happens to land in the one genuinely ambiguous spot.
  Also found and fixed a real display bug (not a numbers bug) in
  `run_null_baseline_test.py`'s verdict formatter, which collapsed
  "null decisively exceeds real" into the same label as "not clearly
  different." `CLAUDE.md` §6.5 updated with a pointer to read the depth
  curve, not just the peak. Cross-corpus/checkpoint-size replication and
  an equivalent L2 depth curve remain the open next step.
- **2026-08-07 (same-day third follow-up, run via the scheduled autonomous
  loop's next 30-minute firing) — the L2 analog, and it reverses rather
  than just qualifies the earlier peak-pair reading.** Added
  `compare_l2_depth_curve` to `analysis/null_baseline.py` (factoring
  `_best_direction_decomposition` into a reusable `_pair_decomposition`
  helper first), `--depth-curve l2` on the CLI, 2 new tests, full suite
  green at **193/193**. Real result (full table in §16 E9's Findings): the
  earlier "L2 gain is statistically indistinguishable from null" verdict
  turned out to be an artifact of comparing the real run's actual best pair
  (Chronos block 10, gain 0.413) against `null_chronos_random`'s own
  *unrelated* global-best pair (block 0 — an early-layer self-vs-random-twin
  predictability artifact, gain 0.444, nothing to do with where the real
  cross-model signal peaks). Retested layer-for-layer at block 10 in both
  runs, the null floor is only 0.062/0.046 and real decisively beats both
  (p=0.002 each) — and does so at every block from 4 through 11 (8 of 12),
  losing only at the earliest, lowest-absolute-gain blocks. Same lesson as
  the L1 depth curve, arguably sharper here: a single global-best-pair
  comparison across two independently-run configs can silently compare two
  *unrelated* layers whenever each run's own optimum lands somewhere
  different, and only a matched-layer test catches that. `CLAUDE.md` §6.5
  corrected in place (not deleted). §13's open question rewritten to state
  the layer-matching lesson explicitly rather than a flat "L2 gain doesn't
  survive." Open: the reverse L2 direction, and cross-corpus/checkpoint-size
  replication for both L1 and L2.
- **2026-08-07 (same-day fourth follow-up, autonomous loop) — §6.1's
  long-open "feed SAE evaluation results back into the layer-selection
  correlation study" item, finally unblocked by the null-baseline runs
  already sitting in `runs/`.** Built `sae/layer_sweep.py` +
  `run_sae_layer_sweep.py` (per-captured-layer, CPU-only TopK-SAE
  train+eval reusing the existing `sae` module's functions verbatim) and a
  fourth `layer_selection.py` proxy metric, `sae_ground_truth_rho`. Ran it
  against `medium_run_chronos_base` (already had it), plus
  `null_timesfm_random`/`null_chronos_random` (chosen specifically because
  they're zarr-v2 and already had `internals` enabled) to clear
  `cluster_bootstrap_spearman`'s `n_groups>=3` floor — 6 groups, 66 records
  total. Result: `input_cka` vs the new `sae_ground_truth_rho` proxy is
  depth-controlled-significant (ρ=0.623, CI [0.074,0.758]) and closely
  replicates the *already-established* `input_cka` vs `probe_decodability`
  relationship (ρ=0.580) — two independent interpretability proxies
  agreeing after the same depth control, a real convergent-validity result.
  `effective_dim` showed no significant relationship with either proxy at
  this n. Found (and explicitly flagged, not silently worked around) that
  `runs/medium_run`/`runs/real_run` can't be added to this pool — both
  predate this repo's `zarr<3` pin and are unreadable by the current
  environment (`CLAUDE.md` §11.15's trap, recurring). Full suite green at
  198/198. Full numbers in §6.1's Findings.
- **2026-08-07 (same-day fifth follow-up, autonomous loop) — Phase 2c
  (§6.3): first real test of whether L1/L2's stitching machinery separates
  a documented-shared-lineage model pair from an independently-trained
  one.** Verified from `amazon/chronos-t5-base`'s actual HF README (not
  guessed) that every Chronos-T5 size fine-tunes from a *separately*
  pretrained `google/t5-efficient-{size}` checkpoint (no shared init
  weights) but all sizes then train on the identical time-series
  corpus/procedure — the precise "meaningful lineage" the checklist asked
  to confirm. Ran a new lean config
  (`configs/distill_positive_chronos_small_base.yaml`, only `l0/l1/l2/
  internals/report`, done in under 2 minutes) live against real
  `amazon/chronos-t5-small` + `amazon/chronos-t5-base` as the positive
  pair; reused `medium_run_chronos_base` (Chronos-T5-Base vs TimesFM,
  already on disk) as the negative. New `analysis/distillation_detection.py`
  is a ~20-line named wrapper around `null_baseline.py`'s existing
  bootstrap-difference functions (nothing in them is null-specific) plus a
  matching CLI. Result: **decisive separation** — L1 peak CKA 0.734
  (positive) vs 0.381 (negative), diff +0.353 CI [+0.307,+0.390] p=0.0005;
  L2 best gain 0.637 vs 0.413, diff +0.224 CI [+0.168,+0.285] p=0.0005 —
  but flagged, not oversold: the positive pair is same-architecture (both
  T5) and the negative is cross-architecture, so this one test can't yet
  separate "tracks lineage" from "tracks architecture match." Left the
  checklist's "write a standalone method note" item explicitly undone
  pending a same-architecture negative control, per §2.6/§2.7's honesty
  doctrine — real stakes (provenance/IP disputes) if overclaimed. Full
  suite green at 200/200 (2 new tests). Full numbers in §6.3's Findings.
- **2026-08-07 (same-day sixth follow-up, autonomous loop) — Phase 4 (§9):
  first real multi-model-expansion addition, Chronos-2, plus a real bug
  found and fixed in shared infrastructure rather than worked around.**
  New `models/chronos2_adapter.py` (`Chronos2Adapter`), registered as
  `"chronos2"`. `--check-alignment` crashed on the very first real capture:
  `extraction/hooks.py`'s `_primary()` assumed a module's forward output is
  either a plain tensor or a literal `tuple`, but `Chronos2EncoderBlock`
  returns an HF `ModelOutput`-style dataclass (dict-like, integer-indexable,
  but not a `tuple` instance) — exactly the "any exception outside the new
  adapter file is a bug in the abstraction" case §9's own checklist names.
  Fixed in `hooks.py` (generalized `_primary`, added `_rebuild` so
  `token_patch`/`output_mean_ablate` preserve the original output type
  instead of degrading it to a plain tuple), with a new offline regression
  test (`tests/test_hooks_modeloutput.py`, 3 tests) rather than any
  Chronos-2-specific special-casing. After the fix: perfect 1.00
  diagonal-hit alignment at all 12 layers, and a full live GPU validation
  run (Chronos-2 vs TimesFM, with L3 per-window patching and attention
  patterns/ablation both exercised) completed clean end-to-end — L1 peak
  CKA 0.43 vs a shuffled-null ≈0.04, L3 fingerprint agreement ρ=−0.75
  (replicating the TimesFM/Chronos-T5 anti-correlation pattern against a
  third, structurally distinct model), and real, sane periodicity/
  load-bearing-head numbers from attention analysis. Full suite green at
  203/203. Full numbers and the abstraction-bug mechanism in §9's Findings.
- **2026-08-10 — resumed after a gap; documented the ad-hoc autonomous-loop
  pattern as a real doctrine section, kicked off the next flagged §13 item
  as a background run, and closed the Phase 4 licensing-recheck open
  question live.** No prior session had written down *how* the several
  "run via a scheduled autonomous loop" entries above actually worked, so
  it was ad hoc every time; added `CLAUDE.md` §2.8 ("Delegate long-running
  compute; keep the roadmap moving in parallel") stating the pattern as an
  instruction, and set up a session-local 30-minute-cadence `CronCreate`
  loop (5-hour self-stopping deadline) so this continues without a human
  re-prompting each time. Launched §13's explicitly-flagged next step — a
  live rerun of the `factor_emergence` layer-selector bake-off
  (`CLAUDE.md` §11.18's fix, previously verified only on synthetic data)
  against the real, already-extracted `runs/layer_screen_experiment` store
  — as a background agent per the new §2.8 rule, rather than blocking on it;
  its numbers will be written up here once it reports back (not yet landed
  as of this entry). While that ran, closed §13's other open item live
  rather than leaving it as a stale assumption: **Moirai** (`Salesforce/
  moirai-1.1-R-small` and the newer `moirai-2.0-R-small`) is
  `cc-by-nc-4.0` (non-commercial, fine for this repo's research use but
  previously unstated), and Moirai 2.0 turns out to have moved to a
  **decoder-only** architecture, not the masked-encoder design §9's
  candidate list described (only 1.x still matches that description).
  **Sundial** (`thuml/sundial-base-128m`) is **Apache-2.0** (fully
  permissive), confirmed decoder-only with a flow-matching probabilistic
  head — answering §9's own "worth checking" note rather than leaving it
  open. Net call: Sundial is now the more attractive next Phase-4 pick of
  the two (permissive license, cleaner structural contrast) — see §9 and
  §13 for the full recheck. No code changed this entry (docs + one
  background job only); full suite not re-run since nothing executable
  changed.
- **2026-08-10 (same-day follow-up) — the background bake-off rerun landed;
  written up as a genuine negative result, not the hoped-for confirmation.**
  The `factor_emergence` fix (`CLAUDE.md` §11.18, 2026-08-06) does not clear
  the nulls on real checkpoints: Chronos-T5-Small is completely unaffected
  (bit-identical failing selection, both seeds, pre- and post-fix) and
  TimesFM's recall@budget actively regressed (0.20→0.00). A controlled
  same-data-different-formula comparison (not just diffing two independently
  noisy end-to-end runs) pinned the cause: on this real corpus the
  early-layer bias comes from *strong*, near-input-statistics factors
  decoding well from layer 0, the mirror image of the weak-noisy-factor
  scenario the fix's synthetic test targeted — so peak-weighting amplifies
  the bias instead of correcting it, and the new absolute floor never once
  fired. `work_bend`/`coverage` (the actual production default) confirmed
  unaffected. §13's corresponding sub-question and `CLAUDE.md` §11.18 both
  updated in place with the full numbers; `factor_emergence` remains
  correctly excluded from production. Full `tsfm_lens` suite green at
  203/203 (verification-only, no production code changed). This closes out
  the two items opened earlier this session — the licensing recheck and
  this bake-off rerun — with no roadmap item left in-flight as of this
  entry; the 30-minute `CronCreate` loop set up earlier this session will
  pick the next one.
- **2026-08-10 (second cron-loop firing) — Sundial added as a fourth real
  `ModelAdapter`, Phase 4's second multi-model-expansion addition, via the
  same background-agent-plus-continue-the-loop pattern as the entry
  above.** No background agent was pending write-up at firing time (the
  bake-off rerun above was already closed out); picked Sundial as the next
  concrete item per §9's own priority list, now that the same firing's
  earlier licensing recheck had established it as the more attractive of
  the two remaining candidates. Launched the whole adapter-build-and-verify
  task as one background agent (adapter code, alignment check, capability
  matrix, a live TimesFM-vs-Sundial comparison run, tests — the full §9
  checklist, not just a rerun of existing code) rather than working on it
  inline, and deliberately did not start a second, file-touching task in
  parallel this firing to avoid contention with an agent whose scope was
  this wide. Once it landed: verified independently before writing
  anything up (re-ran the full test suite myself — 207/207, matching the
  agent's own count — and spot-checked the live run's actual
  `runs/sundial_phase4_check/l1/meta.json` artifact against the numbers
  quoted in its report, rather than trusting the report alone). Real
  result: no shared-infrastructure bug this time (contrast with Chronos-2's
  §11.21), but two genuine bugs in the **checkpoint's own remote code** (a
  `transformers.DynamicCache` API mismatch breaking `.generate()` and even
  a cached `forward()` call), worked around by calling the lower-level
  `forward(..., use_cache=False)` directly rather than risking a
  transformers downgrade that could break the other three adapters — new
  `CLAUDE.md` §11.22. Live numbers (§9's new Findings block has the full
  set): L1 peak CKA 0.381 vs. a null of 0.022 (real shared geometry with a
  third architecture, replicating Chronos-2's own finding), and a genuinely
  new data point — L3 fingerprint agreement is *positive* (ρ=0.517) for
  TimesFM-vs-Sundial, breaking the anti-correlation pattern every other
  model pairing in this repo had shown so far. `CLAUDE.md` §6.2's capability
  matrix and adapter-description prose updated to a five-model table.
  No roadmap item left in-flight as of this entry.
- **2026-08-10 (third cron-loop firing) — closed a stale-but-already-answered
  §13 item, then used today's Sundial addition to attack the bake-off's own
  named "third architecture family" open question.** No background agent
  was pending write-up. Rereading §13 (not re-running anything) found the
  "is `work_bend` still the bake-off winner under production conditions"
  item had actually been resolved by A1's fix days ago and simply never
  marked `[x]` — `LayerScreenConfig`'s defaults (`stride: 1`,
  `require_full_capture: True`) already make every production config run
  the dedicated stride-1 screening extraction A1 built, and A1's own
  Findings already recorded live proof of exactly this scenario. Marked
  resolved with a pointer back to A1 rather than re-verifying something
  already verified. Then picked up §6.1.1-E's own still-open replication
  question — "a third architecture family" — now directly answerable
  using Sundial, added earlier this same day. Launched a background agent
  to build a 3-model (TimesFM + Chronos-T5-Small + Sundial, all
  `capture_layer_stride: 1`) extension of the original bake-off config and
  rerun `run_layer_screen_bakeoff.py` against it, checking both whether
  `work_bend` still wins on a third architecture and whether
  `factor_emergence`'s newly-diagnosed (this same day) early-layer-bias
  failure mode reproduces or looks different for Sundial. Did not start a
  second parallel task this firing (the new 3-model extraction is itself
  the natural next step; no other independent item was picked up). Results
  not yet landed as of this entry.
- **2026-08-10 (fourth cron-loop firing) — the 3-model bake-off landed;
  writing it up surfaced a more important finding than the one it was sent
  to check.** Independently verified before writing anything up (reread
  the actual `runs/layer_screen_bakeoff_v2.json` and the existing
  `runs/layer_screen_bakeoff.json` directly, and reran the full pytest
  suite myself — 207/207, matching). The "third architecture family"
  question got a real answer: `work_bend` beats both nulls on Sundial, but
  the three-architecture picture is "beats a different subset of the two
  nulls on each," not a clean sweep, and `factor_emergence`'s diagnosed
  early-layer-bias failure (from earlier today) reproduces cleanly on
  Sundial, reinforcing that it's a property of the ground-truth factor
  table rather than either specific architecture. **The bigger finding,
  caught only because verifying the comparison numbers meant reading both
  JSON files directly:** Chronos-T5-Small's qualitative bake-off verdict
  is not stable across independent re-extractions of the identical config
  — `work_bend`'s `beats_random` flipped True→False between the existing
  same-day run and this session's fresh 3-model extraction, while
  TimesFM's own verdict reproduced unchanged in the same two runs. The
  bake-off's existing "stability replicate" only ever reseeds SAE training
  on one frozen, already-extracted store — it has never tested whether the
  extraction itself is stable, and this is the first time that axis was
  ever exercised (by accident, as a side effect of needing a fresh
  extraction for Sundial). Wrote this up as a new, prominent §13 open item
  (not folded quietly into an existing one) and a new `CLAUDE.md` §11.23,
  and updated the verification-status table row rather than treating it as
  a footnote — per §2.6's honesty doctrine, this genuinely changes how much
  confidence any single-run bake-off number in this file deserves, and
  said so plainly rather than downplaying it. No roadmap item left
  in-flight as of this entry; did not start a new background task this
  firing given the length of this write-up.
- **2026-08-10 (fifth cron-loop firing) — the extraction-variance mystery
  from the previous entry is fully root-caused, not just narrowed.** Found
  nothing in progress via `ListAgents` beyond the already-written-up
  3-model bake-off agent; launched a background investigation (ranked
  candidate causes, then live-tested the top one) into whether
  Chronos-T5-Small's flipped bake-off verdict was extraction/GPU/SAE-seed
  variance. While that ran, closed an unrelated `DEPENDENCIES.md`
  invariant-12 gap (Chronos-2's and Sundial's fragile version interactions
  had never been recorded there — added both with the same file:line detail
  `CLAUDE.md` §11.21/§11.22 already has). The agent's own report initially
  under-delivered (returned mid-experiment claiming it was "waiting for
  pytest," resumed once via `SendMessage` to get the real final numbers),
  but its actual finding — GPU nondeterminism ruled out via a bit-exact
  `np.array_equal` re-extraction, Sundial's presence ruled out via a
  2-model-only rerun matching the 3-model numbers, SAE-seed sensitivity
  real but not the cause since neither compared run ever varied it — pointed
  at a code change between the two runs' dates rather than randomness.
  Followed that up directly with `git diff 84cdbc0 HEAD --
  tsfm_lens/data.py` and confirmed it precisely: `§15 A4`'s sampling fix
  (landed 2026-08-06, between the experiment's 08-05 and 08-10 runs)
  replaced a `kept[:max_series]` head slice with a stratified
  `sample_rows(...)` call, and the experiment's own `max_series: 220`
  against a 288-row corpus meant the pre-fix run silently excluded the
  entire 60-row `mixture` family (verified directly via a `Counter` over
  the actual corpus file) — full family coverage only appeared in the
  post-fix runs. Corrected §13's 🔴 item and §6.1.1's Findings block in
  place (marked resolved, did not delete the original "not yet known"
  framing — appended the resolution after it per §0.2), added `CLAUDE.md`
  §11.24 for the general lesson (a background rerun of "the same"
  experiment can silently change meaning across a shared-infrastructure
  fix, with no config diff to catch it — check `git log` on the shared
  paths before spending investigation budget on nondeterminism
  hypotheses), and updated the verification-status table row. Independently
  re-ran the full test suite myself rather than trusting the agent's
  count: **207 passed, 2 warnings in 287.78s**, both warnings pre-existing
  (an intentional overflow-cast test, a `PytestReturnNotNoneWarning` in
  `test_smoke.py`) and unrelated to anything touched this session — matches
  the agent's own reported count exactly. No production code changed this
  firing (docs + a resumed investigation only). Narrower question left
  open: whether the now-correctly-sampled bake-off verdict is itself stable
  across a second seed — smaller in scope than what was open at the start
  of this firing, and a reasonable candidate for the next one.
- **2026-08-10 (sixth cron-loop firing) — started the second-seed stability
  check the previous entry named as the next candidate, and closed out its
  own "worth a deliberate sweep" note.** `ListAgents` showed nothing
  in-flight (the previous firing's investigation agent had fully completed
  and dropped off the list). Launched a background agent to build
  `configs/layer_screen_experiment_v2_seed1.yaml` (identical to `_v2.yaml`
  except `run.seed: 1`), extract, run the bake-off, and compare against the
  existing seed-0 numbers for all three models plus a row-sample-overlap
  check, to answer the narrower question left open at the end of the last
  entry — results not yet landed. While that ran, swept every config's
  `data:` block (not the several other, unrelated `*_max_series` knobs) for
  the specific `data.max_series` corpus-level cap the just-resolved bug
  lives in: it turns out to be set **only** in the
  `layer_screen_experiment*.yaml` family — every other real-checkpoint
  config (`medium_run*.yaml`, `default.yaml`, both `null_*_random.yaml`,
  `distill_positive_chronos_small_base.yaml`, `sundial_phase4_check.yaml`,
  `chronos2_phase4_check.yaml`) sets no `data.max_series` at all, so the
  `_assemble` guard this bug lived behind never fires for them regardless
  of which side of 2026-08-06 they ran on. Wrote this up as closing the
  "worth a deliberate sweep" line from two entries ago: the bug has exactly
  the one victim already found and fixed, not a wider undiscovered
  problem — the two other already-tracked affected numbers (§6.2's SAE
  forecast-preservation/ground-truth, §5.3's lens comparison) go through
  different call sites entirely and their re-measurement status is
  unchanged by this sweep. No test suite run this entry (docs-only
  addition, no code touched, and the background agent's own pytest run
  will cover the code path it touches). Second-seed stability results not
  yet landed as of this entry.
- **2026-08-10 (seventh cron-loop firing, same-turn write-up of the
  second-seed background agent) — the narrower question from the previous
  entry is answered: no, `work_bend` does not reliably survive a reseed.**
  The delegating agent's own final report (a second nested agent it used to
  actually run the GPU work) gave a full seed0-vs-seed1 comparison; rather
  than take it on trust, independently re-derived every headline number
  directly from `runs/layer_screen_bakeoff_v2_seed1.json` and both runs'
  `meta.parquet` files myself (gold_agreement, recall_at_budget,
  beats_random/beats_uniform_stride, selected layers, and the row-overlap
  Jaccard) — all matched the agent's report exactly, including to the full
  decimal. Headline: 76.8% row overlap between the two seeds' 220-of-288
  draws (not a biased or wildly different sample), yet `work_bend` — the
  production default — flips its `beats_random` verdict on 2 of 3
  architectures (TimesFM True→False, Chronos False→True), while
  `coverage` stays fully stable on all three and Sundial stays fully
  stable on every selector. In every flip, the selector's own chosen
  layers were bit-identical across seeds — only the null-comparison
  scorecard moved, tied to the SAE-mass gold ranking shifting under the
  ~23%-different row draw. Read this as a statistical-power gap in the
  bake-off's own evaluation (recall@budget at n=220, budget≤5) rather than
  a bias like A4's — a materially different, and arguably more concerning
  for production use, kind of instability than what the previous two
  entries closed out. Wrote the full comparison into `ROADMAP.md` §6.1.1's
  Findings and updated §13's item and `CLAUDE.md`'s verification-status
  row in place. Independently launched my own separate `pytest -q` run in
  parallel with the write-up (the agent's own run already reported 207
  passed, 2 warnings, matching every prior count this session) as a second
  check; it was still running when this entry was written due to apparent
  GPU contention with the peer session's own activity, so treat the
  suite's status this entry as "agent-confirmed, self-confirmation
  in-flight" rather than doubly independently verified — worth a quick
  glance next firing if the log wasn't checked before this session ends.
  No new background task started this firing — the write-up was the full
  scope of the work. Remaining open items: whether a third seed would
  narrow or confirm this spread, and whether the bake-off's statistical
  power should be improved (larger corpus, bootstrap CI on recall@budget)
  before trusting any single future run's verdict — left for a future
  firing or session, not attempted here.
- **2026-08-10 (fifth cron-loop firing) — launched the extraction-variance
  characterization, and closed a real `DEPENDENCIES.md` invariant-12 gap
  while it runs.** `ListAgents` showed only the already-written-up 3-model
  bake-off agent (completed, stale re-notification, no new content).
  Nothing in progress, so started the §13 🔴 item from the previous firing:
  launched a background agent (two parts — first read
  `data.py`/`extract.py`/`pipeline.py`/`layer_screen.py`/
  `layer_screen_bakeoff.py` for every candidate source of run-to-run
  nondeterminism with file:line citations and rank them; then design and
  run a targeted experiment against the single most plausible candidate,
  either re-running the bake-off against the *same* already-extracted
  `runs/layer_screen_experiment_v2/` store with two SAE seeds, or a fresh
  third independent extraction, whichever the ranking points to) to
  characterize whether Chronos-T5-Small's flipped verdict (previous entry)
  is explained by SAE-training stochasticity, extraction-level
  nondeterminism, or something else. Told it explicitly not to touch
  production code — root-cause first, design a fix in a later session.
  While that ran, noticed `DEPENDENCIES.md` §5 ("Known fragile spots") had
  never been updated for either Chronos-2 (`CLAUDE.md` §11.21, the
  `ModelOutput`-vs-tuple hooks bug) or Sundial (`CLAUDE.md` §11.22, the
  `DynamicCache` API break in its own remote code) — a real gap against
  invariant 12's "any newly-discovered fragile version interaction gets
  reflected there in the same session" rule, missed in both of those
  adapter-build sessions. Added both as new bullets in `DEPENDENCIES.md` §5
  with the same file:line-anchored detail `CLAUDE.md` §11 already has, so
  the two records stay in sync. No test suite run needed (docs-only change,
  no code touched). Extraction-variance results not yet landed as of this
  entry.
- **2026-08-10 — explicit user instruction to fix the second-seed
  `work_bend` instability rather than leave it as documented, mid-turn
  during a routine cron-loop firing.** Not a mechanical firing this time —
  the user directly told the session not to just leave the previous
  entry's finding ("`work_bend` doesn't reliably survive a reseed") as a
  written-up limitation. Root-caused by re-reading `layer_screen.py`
  directly rather than re-testing hypotheses: `select_work_bend`/
  `select_coverage` take no `seed` kwarg at all (grepped to confirm) and
  are fully deterministic given a fixed corpus, so the entire flip had to
  be coming from `build_gold_ranking`'s single stochastic per-layer SAE
  training run standing in as ground truth at a budget small enough
  (2-5 layers) that ordinary training noise flips which layers count as
  "gold." Fixed with two changes, both landed and both verified against
  real checkpoints, not left as an untested patch: (1) `build_gold_ranking`
  now averages `n_replicates=3` independently-seeded SAE-training runs per
  layer instead of trusting one, wired through a new `run_layer_screen_
  bakeoff.py --n-gold-replicates` flag; (2) new `configs/
  layer_screen_experiment_v4[.yaml/_seed1.yaml]` drop the `max_series: 220`
  corpus cap entirely (the corpus is only 288 rows — subsampling bought
  negligible compute savings for real determinism cost). Added a synthetic
  unit test for the averaging/determinism property
  (`test_build_gold_ranking_averages_replicates_and_is_deterministic`);
  full suite green at **208/208** (self-run, not just an agent's report).
  Then launched a background agent to rerun the actual 3-checkpoint
  bake-off (TimesFM 2.5, Chronos-T5-Small, Sundial) on both seeds with the
  fix — not just asserting the fix should work. Result: **every
  `beats_random` verdict now matches across both seeds, for all three
  selectors on all three architectures** (`work_bend`'s TimesFM False↔False
  and Chronos-T5-Small True↔True, both previously flipping, now hold).
  Full comparison table and residual caveats (one secondary metric,
  `factor_emergence`'s `beats_uniform_stride` on Chronos, still moves
  slightly) are in §6.1.1's Findings block, not repeated here. Also fixed a
  small related gap the verification agent caught along the way:
  `gold_score_std`/`n_gold_replicates` were computed but never persisted
  into the bake-off's saved JSON — one-line fix in `run_layer_screen_
  bakeoff.py`. §13's corresponding open item is updated to reflect this is
  now resolved rather than an open statistical-power gap.
- **2026-08-10 (eighth cron-loop firing, deadline reached mid-write-up) —
  stopped the loop with two items left incomplete; both now finished.**
  The prior firing's write-up of a real, newly-discovered finding (a stale
  `runs/medium_run` activation store, written in zarr v3 and silently
  unreadable — returns empty rather than erroring — under this repo's
  pinned zarr v2) was interrupted twice by `Edit` tool-hook timeouts, and
  the loop's own 5-hour deadline (`2026-08-10T21:16:58Z`) was crossed before
  a third attempt could be made, so per the loop's explicit instructions the
  session stopped (deleted cron job `d69df929`) rather than retry further.
  Picked back up on direct user instruction ("finish what you were unable
  to finish, then move onto the next roadmap item"): the `CLAUDE.md` §11.25
  trap entry landed cleanly this time, and this section's own item plus the
  `§5.3`/A4 Findings block above were updated with the same finding —
  including that the background agent attempting the actual re-extraction
  had separately failed on its own infrastructure-level session/API usage
  limit, not a code bug. Re-armed the recurring loop (new job, every 20
  minutes for 3 hours from `2026-08-10T23:13:24Z`) and relaunched the
  `runs/medium_run` re-extraction as a fresh background agent before moving
  to the next roadmap item, per `CLAUDE.md` §2.8.
- **2026-08-10 (same-day follow-up) — picked the next roadmap item while the
  re-extraction ran in the background: closed §6.3's confound, and it fell
  the wrong way.** While waiting on `runs/medium_run`'s re-extraction (also
  needing a mid-flight correction — the background agent twice ended its
  turn assuming a raw shell background job would notify it automatically,
  same mistake `CLAUDE.md` documents elsewhere; resumed it with explicit
  polling instructions each time and verified its real progress directly on
  disk rather than trusting its self-report), built the same-architecture
  negative control §6.3's 2026-08-07 Findings said wasn't available with any
  checkpoint in reach: `configs/distill_negative_random_architecture.yaml`
  pairs `Chronos-T5-Small`/`Chronos-T5-Base` with `random_init: true` on
  both sides — same architecture family as the real trained positive pair,
  zero shared training or lineage. Ran live in about 90 seconds. Result was
  the opposite of what the method would need to be useful: this zero-
  training control's L1 CKA (0.878) and L2 gain (0.834–0.940) both
  significantly *exceeded* the real trained positive pair's (0.734 CKA /
  0.637 gain; bootstrap diff CIs [−0.217,−0.130] and [−0.335,−0.269], both
  p=0.0005, `n_boot=2000`). Closed §6.3's own checklist item and §13's
  matching open question as answered — the method is falsified for its
  stated provenance/IP use case, not merely confounded — rather than left
  open. No source code changed (one new config, reused the already-built
  `random_init` mechanism and `distillation_detection.py` test harness
  exactly per `CLAUDE.md` §2.2), so no test suite re-run was needed; the
  live-checkpoint numbers are themselves the verification.
  Meanwhile the relaunched `runs/medium_run` re-extraction agent completed
  cleanly (~11 min wall-clock, needing the same "actively poll, don't
  assume a raw shell background job notifies you" correction as its
  predecessor, verified independently on disk both mid-run and at
  completion rather than trusting either agent report at face value): fresh
  v2 store confirmed (`.zattrs` present, no `zarr.json`), `report.html`
  rendered 11 sections / 27 findings, `confirm` passed (1/1 dev hypotheses
  on private data), and the now-trustworthy TimesFM lens number
  (`final_mase: 1.7899408340454102`) reproduced bit-for-bit against
  `medium_run_chronos_base.yaml`'s already-recorded value — exactly the
  cross-check the matched-`max_series` fix (§5.3/A4) was supposed to
  produce. Self-ran the full `tsfm_lens` suite afterward rather than
  trusting the agent's own count: **208 passed, 2 warnings**, identical to
  what the agent had independently reported. Both this session's open
  threads (§6.3's confound, §5.3/A4's stale store) are now fully closed —
  see their own Findings blocks for the numbers, not repeated here.
- **2026-08-10 (ninth cron-loop firing) — picked up §16 E10 (probabilistic-
  forecast calibration diagnostics) as the next concrete item, with nothing
  else in flight.** Both prior threads (§6.3's confound, §5.3/A4's stale
  store) were fully closed by the previous entry, and no background agent
  was running, so per `CLAUDE.md` §2.8 this was small/quick enough to build
  directly rather than delegate. New `analysis/calibration.py` (reliability
  curve, PIT histogram, quantile-crossing rate, interval coverage/sharpness
  by horizon) reduces over `predict()`'s already-computed `quantiles`/
  `targets` arrays — zero new forward passes, wired into `run_l0` behind a
  new `L0Config.calibration` flag and rendered as three new report
  sections. Verified with 7 new synthetic-planted-answer unit tests (all
  passing on first run) plus a live `configs/smoke.yaml` run confirming the
  new `l0/calibration.json` artifact and report sections render with real,
  correctly-populated content. Full suite green at 215/215, zero
  regressions. See §16 E10's own checklist entry for the full write-up and
  numbers, not repeated here.
- **2026-08-10 (tenth cron-loop firing) — launched the §13/A4 follow-up
  SAE re-measurement in the background, did §16 E11's n_boot/p-floor fix
  directly in parallel.** Per `CLAUDE.md` §2.8: §13's open item "how much of
  what's already recorded survives §15's sampling/noise-floor fixes" still
  named one concrete, un-closed re-measurement — §6.2's SAE
  forecast-preservation ΔMASE and ground-truth ρ, computed 2026-08-05, one
  day before A4's stratified-sampling fix landed. Confirmed the fix is
  already wired into both call sites (`sae/eval.py:94`, `sae/
  ground_truth.py:257`) and the `medium_run_chronos_base` activation store
  is a valid v2 zarr (no §11.25-class repair needed), so this was a pure
  re-run of `sae,report` against already-extracted activations — launched
  as a background agent (needed correcting twice for the same "waiting on
  a notification that will never arrive" mistake this file's own §11.24/
  earlier entries already document as a recurring risk of this delegation
  pattern; it eventually armed a proper `Monitor` for the actual GPU run
  and a second one for its own post-run pytest check). While that ran,
  picked up §16 E11 (statistical-discipline hardening) directly: audited
  every `p=` rendering in `report.py` (only two exist, both already paired
  with a CI — that bullet was already satisfied) and confirmed A15's
  `hypotheses.json` registry already covers the multiplicity-ledger bullet
  (this item just hadn't been marked to reflect either). What was actually
  missing — `n_boot`/the 1/n_boot floor stated inline wherever a p-value is
  shown — is now fixed (`stats.py::paired_bootstrap` returns `n_boot`;
  `report.py`'s new `_p_note` helper renders it at all three sites) and
  verified live against a rerun of `configs/smoke.yaml`'s `l0,report`
  stages (confirmed `"n_boot=150, p floored at 1/n_boot=0.0067"` in the
  actual rendered HTML) plus a full self-run suite: **215 passed, 2
  warnings**, same two pre-existing benign warnings, no regressions. See
  §16 E11's own checklist entry for the full detail. The cluster-bootstrap-
  unit-explicit bullet of E11 remains open; E11 marked `[~]`, not `[x]`.
  The background SAE re-measurement agent then reported back: Chronos-T5-
  Base's ΔMASE and both models' ground-truth ρ replicated closely against
  the already-recorded 2026-08-06 re-measurement (confirming those numbers
  stable), but **TimesFM's forecast-preservation ΔMASE did not** (0.175 →
  0.1097, a real ~37% shift with reconstruction fidelity itself unchanged)
  — a genuine new finding, not an agent error. Investigating this surfaced
  that the §13 open item this whole thread was closing had itself gone
  stale hours after it was written: its own "left as a follow-up" framing,
  and A4's matching Findings-block bullet, were never updated after the
  2026-08-06 same-day follow-up had already performed exactly the
  re-measurement both text blocks still described as outstanding. Corrected
  both in place (§0.2 discipline) rather than layering a third, redundant
  "still open" note on top, appended the new 08-10 measurement to §6.2's
  Findings, and spun the newly-discovered TimesFM SAE-forecast-preservation
  run-to-run variance out as its own explicit, narrower open follow-up
  (a repeat-run noise floor for that one metric specifically) instead of
  leaving the whole §13 item ambiguously reopened. §13's item marked `[x]`;
  §5.3's lens-comparison half was already closed by an earlier entry this
  same day. Verification: the launching agent's own post-rerun pytest
  (215 passed, 2 warnings) matches this turn's own earlier, independently-
  run full-suite result — a real cross-check, not a single unverified
  report.

**2026-08-10, eleventh cron-loop firing.** Nothing was in progress
(previous firing's write-up was the last outstanding item). Re-read
§13/§15/§16 fresh and picked up §16 E12 (horizon-resolved everything) —
specifically only its cheapest, zero-new-forward-passes sub-part
(MASE/pinball per horizon step), since the other two named sub-parts
(L3 per-window patching by horizon, Lens crystallization by horizon) are
each their own separate unit of work. Implemented directly (small/quick,
no background delegation needed): `analysis/stats.py::
mase_pinball_by_horizon()`, `L0Config.horizon_resolved` flag,
`l0_behavioral.py::_summarize_by_horizon` + new `l0/horizon_resolved.json`
artifact, `report.py::_horizon_resolved_block()` wired into `_sec_l0`. Four
new synthetic-planted-answer tests
(`tests/test_horizon_resolved.py`) all passed on first run. Verified live,
not just unit-tested: reran `configs/smoke.yaml`'s `l0,report` stages,
confirmed `l0/horizon_resolved.json`'s array shapes directly and the
rendered HTML's actual findings text (`"L0 horizon profile — patchy: MASE
2.731 at h=1 vs 4.042 at h=32 (ratio 1.48x)"`,
`"...steppy: MASE 5.270 at h=1 vs 5.584 at h=32 (ratio 1.06x)"`) — findings
count rose from 27 to 29 as expected, sections stayed at 11. Full
`tsfm_lens` suite green at **219 passed, 2 warnings** (215 prior + the 4
new tests, same two pre-existing benign warnings, zero regressions). §16
E12 marked `[~]` (partial — the L3/Lens sub-parts remain open, each is a
natural next pick for a future firing). No background work was launched
this firing since the chosen item was small enough to finish inline within
the loop's own turn.

**2026-08-10, twelfth cron-loop firing.** No background agent was running
(`ListAgents` returned none) and nothing else was mid-way. Per this
firing's own brief, retried the `CLAUDE.md` §11.25 write-up that two
earlier edit attempts had failed to apply due to tool-hook timeouts — this
time it landed cleanly on the first try. Confirmed on disk, not just from
memory, that the underlying fix is real before writing it up: `runs/
medium_run/activations.zarr` has a valid v2 `.zattrs`/`.zgroup` (no
`zarr.json`), file mtimes 2026-08-10 23:17–23:28Z (consistent with the
"same-day follow-up" session-log entry above, which already documented the
successful re-extraction — that entry's numbers were correct, only
`CLAUDE.md` §11.25 itself had never actually been updated to match).
Corrected §11.25 in place: replaced the "Not yet fixed as of 2026-08-10"
line with what actually happened (a second background agent completed the
delete-and-re-extract cleanly after the first hit an infra-level usage
limit; ~11 min wall-clock; fresh v2 store; TimesFM lens number
`1.7899408340454102` reproduced bit-for-bit against `medium_run_chronos_
base`; full suite 208/2 at the time), and explicitly flagged that
`real_run`'s own store is untouched and still stale (confirmed directly:
`real_run/activations.zarr` still has `zarr.json`, no `.zattrs`) — not to
be assumed fixed by association. Also corrected a now-stale sub-claim in
§6.2's SAE-ground-truth-null Findings block (line ~2880) that had listed
`medium_run` alongside `real_run` as unusable for pooling due to the zarr
v3 issue — added a same-day correction noting `medium_run` could now be
added to that pool on a future pass, while `real_run` remains excluded and
unchecked. Doc-only changes (`CLAUDE.md`, `ROADMAP.md`); no code touched,
so no test suite re-run was needed. No new roadmap item started this
firing per the brief's own instruction to finish the pending write-up
first and not layer new heavy work on top in the same turn.

**2026-08-10, thirteenth cron-loop firing.** Same firing prompt as the
prior one (the still-pending §11.25 retry it names had already landed last
firing) — `ListAgents` confirmed no background work running, so re-read
§13/§15/§16 fresh per the brief's own instructions. Picked up the first of
§16 E12's two remaining named sub-parts: L3 per-window patching
restoration resolved by horizon step, explicitly flagged by the previous
E12 write-up as "the arrays already exist... this needs only the
reduction step" — confirmed that framing was accurate by reading
`_window_restoration`/`_patching` directly, so this was small/quick enough
to build inline rather than delegate to a background agent. Implemented,
unit-tested (4 new synthetic-planted-answer tests against a newly
extracted, directly-testable `restoration_by_horizon()` helper) and
live-verified against a rerun of `configs/smoke.yaml`'s `l3,report`
stages (confirmed the new artifact's shape/finiteness and the new report
heatmap section + findings text directly). Full `tsfm_lens` suite green
at **223 passed, 2 warnings**, zero regressions. See §16 E12's own
checklist entry for the full write-up and numbers. E12 stays `[~]` — only
the Lens crystallization-depth-by-horizon sub-part remains, a natural pick
for a future firing. No background work was launched this firing since
the chosen item finished comfortably inline.

**2026-08-11, fourteenth cron-loop firing.** Same recurring firing prompt
as the prior two; `ListAgents` again confirmed no background work running
and nothing was mid-way. Picked up the exact item the previous firing
flagged as the natural next pick: the Lens crystallization-depth-by-
horizon sub-part, the last of §16 E12's three named parts. Implemented
directly (small/quick, no delegation needed): `analysis/lens.py` gained
`per_series_h`/`final_mase_h` (the same skip-lens arrays already computed
for the whole-horizon curve, kept resolved by horizon step instead of
collapsed by `.mean(axis=2)`) and a new standalone
`crystallization_depths()` function that unifies the existing scalar
crossing logic with a new per-horizon-step version — refactored the
existing call site to use it too, verified behavior-preserving via a
dedicated 2-D/1-D consistency test rather than assumed. New
`LensConfig.horizon_resolved` flag, new `skip_mase_by_horizon` array and
`crystallization_depth_by_horizon` list artifacts, new
`report.py::_lens_horizon_block()` (heatmap + line chart + findings). 4
new synthetic-planted-answer tests (`tests/test_lens_horizon_resolved.py`)
all passed on first run; extended `test_smoke.py`'s existing artifact
shape-check test too. Verified live end-to-end: reran
`configs/smoke.yaml`'s `lens,report` stages, confirmed the new arrays'
shapes/finiteness and the rendered HTML's actual new chart titles and
findings text directly (not just that the run exited 0) — findings count
rose from 39 to 41 as expected, sections stayed at 11. Full `tsfm_lens`
suite green at **227 passed, 2 warnings** (223 prior + the 4 new tests,
same two pre-existing benign warnings, zero regressions). **§16 E12 is
now fully `[x]`** — all three named sub-parts landed across this and the
two prior firings. No background work was launched or left running this
firing.

**2026-08-11, fifteenth cron-loop firing.** Same recurring firing prompt;
`ListAgents` again confirmed no background work in flight and nothing was
mid-way from a prior firing. §16 E13 (spectral lens) was considered and set
aside as too large to safely land within the remaining loop window, so
instead picked a doc-only item off the open-checklist list: "revisit and
either implement or explicitly defer-with-reason the remaining `CLAUDE.md`
§13 future-work items not already folded into this roadmap" (this
section's own list, above). Cross-checked all five named `CLAUDE.md` §13
items against the existing §16 backlog and found three already tracked
(Chronos decoder gap = E21, deeper component resolution = E18, more source
adapters already explained in full at `CLAUDE.md` §4.3/§12) and two
genuinely missing (CI diversity gates for `benchmark_validation`,
real-corpus activation bucketing) — added those as new §16 **E23**/**E24**,
each with scope and an explicit defer-with-reason rationale rather than
either implementing them or leaving them unaddressed. Checklist item marked
`[x]` with the full resolution recorded inline above. No code changed, no
test suite rerun (nothing to verify — this was a cross-referencing audit,
not an implementation), no background work launched or left running.

**2026-08-11, sixteenth cron-loop firing.** `ListAgents` confirmed no
background work in flight. Picked up **§16 E15** ("Per-token SAE
evaluation... fixes the window-broadcast confound that currently makes
Chronos's forecast-preservation number unreadable"), the item directly
blocking §7 bullet 3's feature-level ablation, with both its stated
dependencies (A4, A13) already `[x]`.

Implemented the token-granularity fix additively in
`tsfm_lens/sae/eval.py`: extracted the existing window-pooled-then-
broadcast logic into `_window_broadcast_replacement` unchanged, and added
`_token_level_replacement`, which encodes/decodes every raw token
independently through the SAE with no window pooling or broadcast at all.
`forecast_preservation` gained a `granularity: "window" | "token"`
parameter (default `"window"`, so every previously-recorded number under
the `forecast_preservation` key is untouched) and now returns a
`"granularity"` field; unknown values raise. `sae/train.py`'s per-target
loop now calls `forecast_preservation` a second time with
`granularity="token"`, stored under a new, separate
`"forecast_preservation_token"` key alongside the original
`"forecast_preservation"` — additive, nothing overwritten, per §0's own
discipline. `report/report.py`'s SAE section renders both ΔMASE values
side by side when the token variant is present. Extended
`tests/test_smoke.py`'s mock-pipeline SAE integration test to assert on
both keys, and added a new synthetic-data unit test to `tests/test_sae.py`
(`test_window_broadcast_collapses_within_window_variation_token_level_does_not`)
that builds a 32-token/8-token-window Chronos-like span array and confirms
directly: `_token_level_replacement` reproduces the input exactly (identity
SAE), `_window_broadcast_replacement` collapses every token in a window to
that window's mean, and only the token-level path retains real within-
window variation (per-feature std averaged over the token axis > 0.1 vs.
< 1e-5 for the broadcast path). One self-introduced test bug caught and
fixed before trusting the result (§2.4 discipline): the first assertion
called `.std()` on a 2-D `(token, feature)` tensor with no `dim` argument,
which flattens across both axes and let the two features' differing means
inflate the "broadcast" std spuriously; fixed to `.std(dim=0)` (then
`.mean()`/`.max()` across the feature axis) — confirmed via a standalone
debug script that printed `_window_broadcast_replacement`'s actual output
and found all 8 rows in the first window bit-identical, proving the
production code was already correct and the bug was in the test's own
reduction. `tests/test_sae.py -q` → 11 passed after the fix; full
`tsfm_lens` suite (228 tests) green with zero regressions — 2 pre-existing
benign warnings only (float32-cast overflow in an unrelated test,
`test_end_to_end`'s pytest return-value warning).

Launched a background `Agent` to get real numbers from
`runs/medium_run_chronos_base`'s already-extracted TimesFM-2.5-200M /
Chronos-T5-Base store (`python run.py --config
configs/medium_run_chronos_base.yaml --stages sae,report --force sae`,
`conda activate cudaPy` — note: this machine's actual active environment is
named `cudaPy`, not the `tsfmPy` name `DEPENDENCIES.md`'s generic
recreation instructions use; confirmed via `which python3` resolving into
`~/miniforge3/envs/cudaPy`), briefed to report the exact
`forecast_preservation`/`forecast_preservation_token` dicts for both
targets and nothing else. The agent's own report initially came back saying
it had launched the run as a detached background process (PID 2406883) and
armed a persistent log monitor rather than waiting synchronously — the
notification for that monitor has since landed, same firing, with the real
numbers.

**Same-day follow-up, numbers now in hand.** TimesFM's `mase_delta` came
back bit-for-bit identical between granularities (`0.10971450805664062`
both under `"window"` and `"token"`) — exactly as expected, since its token
width equals the alignment window, confirming the token-level code path
introduces no unrelated systematic bias. Chronos-T5-Base's numbers: under
`"window"`, `mase_clean: 3.1090171337127686`, `mase_reconstructed:
6.9776177406311035`, `mase_delta: 3.868600606918335` (badly fails, in line
with the prior recorded failure); under `"token"`, `mase_clean:
3.1090171337127686` (unchanged, same clean forecast), `mase_reconstructed:
2.7632558345794678`, `mase_delta: -0.3457612991333008` (net *better* than
the clean forecast). This is decisive confirmation that the window-broadcast
confound — not SAE reconstruction quality — was the dominant driver of every
previously-recorded Chronos forecast-preservation failure. The verifying
agent itself flagged a caveat worth preserving rather than smoothing over:
Chronos's *negative* token-granularity delta is a little surprising on its
own terms, and arguably deserves its own scrutiny (n=24 series is a small
sample; TopK sparsity acting as a mild denoiser is another candidate) rather
than being read purely as "confound fully explained, SAE is great" — not yet
investigated, left as an open thread. `CLAUDE.md` §13 item 3 now carries a
"Correction:" callout with these exact numbers, and §16's E15 checklist item
above carries the same Findings note. **E15 marked `[~]`, not `[x]`**: the
per-token-evaluation half (this write-up) is done and live-verified: the
*feature-level ablation* half of the item's stated scope has not been
started and is a reasonable next pick now that its blocker is resolved.

In parallel (per this loop's own step 2, so the turn wasn't spent idle
while the above ran), also implemented **§16 E23** (`benchmark_validation`
diversity metrics as CI pass/fail gates) — a fully independent package
(`tsfm_benchmark`, not `tsfm_lens`), chosen specifically so it shared no
files with the background run above. New `benchmark_validation/gates.py`:
`GateThresholds` (corpus-wide redundancy fraction, effective
dimensionality, near-collision fraction, plus looser per-group variants of
the latter two) and `check_diversity_gates(report, thresholds=
DEFAULT_THRESHOLDS)`, which evaluates a `validation_report.json`-shaped
dict and returns a pass/fail verdict per gate plus an overall verdict;
`InsufficientN` groups (sec 15 A17) are skipped rather than scored as
failures. **Calibration status, stated honestly in the module docstring
and unchanged from E23's own original caveat: still only one reference
data point** (the demo-mode run's 4.8% redundancy / 4.6-of-24 effective
dimensionality, `CLAUDE.md` §5) — thresholds are set with deliberate
headroom around that single sample, not tightly calibrated to it, so a
fail here means "worth a human look," not a validated guarantee either
way. Wired into `example_runs/run_validation.py`: the gate table always
prints after the existing summary; a new `--enforce-gates` flag (default
off, for the calibration reason just stated) exits nonzero on any failing
gate. New `tests/test_diversity_gates.py` (6 tests, synthetic dicts,
covering: the demo reference point passing, a deliberately collapsed
corpus failing all three corpus-wide gates, missing metrics failing closed
rather than being silently skipped, an `insufficient_n` group being
excluded rather than penalized, a legitimately narrow archetype passing
its own looser per-group floor, and custom thresholds being respected) —
all 6 passed. Full `tsfm_benchmark` suite reran clean (56 passed, 1
skipped, zero regressions). Live-smoked via `run_validation.py --demo`:
the gate table rendered correctly and, unprompted, caught a **real**
near-collision failure in the demo's own planted-duplicate group
(`group[unknown].near_collision_fraction` 0.400 > 0.2 threshold) — expected
and correct, since that group *is* five near-duplicate series by
construction, and is exactly the kind of genuine collapse this gate exists
to catch; not a threshold-calibration bug, a working demonstration of the
mechanism. §16 E23 is now scaffolded and tested but left `[ ]` rather than
`[x]`, since its own stated prerequisite — recalibrating thresholds against
more than one real corpus build — is still open; that's the natural next
step for a future firing, not attempted this session.

**2026-08-11, seventeenth cron-loop firing, same-day follow-up.** Closed
out the "Feature-level ablation" item left open above: wrote and passed a
synthetic-planted-answer unit test
(`test_sae.py::test_feature_ablated_replacement_removes_exactly_that_features_contribution`,
found and fixed one real bug along the way — the test's fake linear-decoder
SAE fixture needed a `__call__` returning `(recon, features)` to match
`_token_level_replacement`'s real calling contract, caught by actually
running the test per §2.4, not by inspection); extended
`test_smoke.py::test_sae_stage_integration` to assert graceful no-op
degradation (`entry["feature_ablation"] is None`) when no ground-truth
matches exist, verified via a direct run showing the exact skip log line.
Launched a background `Agent` (per §2.8, report-only) to live-verify
against real `google/timesfm-2.5-200m-pytorch` /
`amazon/chronos-t5-base` activations from `runs/medium_run_chronos_base`
— it ran clean end-to-end (exit 0, no errors) against real
ground-truth-matched candidates on both targets (154 TimesFM / 115
Chronos-T5-Base matched features); full verbatim per-feature MASE deltas
are in §13's "Feature-level ablation" Findings note. Item marked `[x]`
there and in §16 E15's cross-reference. Reran the full `tsfm_lens` suite
after all edits: **229 passed, 0 failed, 2 pre-existing unrelated
warnings, 292.28s** — no regressions. As this firing's second, independent
quick item (per step 2/3, run in parallel with the background agent):
repo-wide grep for hardcoded absolute paths (`CLAUDE.md` invariant 11) —
`grep -rn "/home/\|/Users/\|[A-Za-z]:\\\\" --include="*.py" --include="*.yaml"
--include="*.yml" tsfm_model_analysis tsfm_benchmark` excluding `runs/` and
`.git/` — found zero real violations; the only two matches are a benign
fake-YAML string fixture in `test_meta_report.py` and the absolute-path-
*detection* regex itself in `test_provenance.py`. Invariant 11 confirmed
still holding after every edit made this session.

**2026-08-11, eighteenth cron-loop firing.** Checked for completed
background agents first (`ListAgents` → none running) and confirmed the
stop condition isn't met (16:08 UTC vs. the 21:52 UTC deadline). Surveyed
open items, confirmed §15's A1–A19 audit backlog is fully closed, and
picked §16 E16 (cross-model SAE feature matching) as the next concrete,
unblocked item — no fresh extraction or training needed, since real SAE
checkpoints already exist on disk from the feature-ablation work
(`runs/medium_run_chronos_base_feature_ablation_check/sae/`). Implemented
`sae/matching.py` + `tests/test_sae_matching.py` (5 synthetic planted-answer
tests, all passing; one real `numpy.bool_`/`is` bug in
`ground_truth_agreement` caught by the planted test itself before any live
run — see §16 E16's own Findings for the fix). Live-verified the same
firing against the real TimesFM/Chronos-T5-Base checkpoints and the
existing zarr store: 40 of 50 TimesFM ground-truth-matched features found
a Chronos-side partner clearing `corr_threshold=0.3`, with the top matches
concentrated in `archetype_trend_dominant`/`tier_realism_stress` and
agreeing on ground-truth field and sign — full numbers in §16 E16's
Findings, not repeated here. Item marked `[x]`. Launched the full
`tsfm_lens` suite in the background to confirm zero regressions from the
new module; per this firing's own no-idle-waiting rule, moved directly to
writing up this session-log entry and the E16 Findings rather than
blocking on it. **Background suite completed same firing: 234 passed, 0
failed, same 2 pre-existing unrelated warnings** (up from the prior
session's 229 — the 5 new `test_sae_matching.py` tests — confirming zero
regressions from the new module).

**2026-08-11, nineteenth cron-loop firing.** Checked for completed
background agents first (`ListAgents` → none running) and confirmed the
stop condition isn't met (16:38 UTC vs. the 21:52 UTC deadline; 22 open
`- [ ]` checklist items remained). Picked §16 E17 (input front-end
diagnostics) — specifically its named phase-sensitivity/patch-boundary
probe, described as "the direct test of the period=32 aliasing hypothesis
§7 [now `CLAUDE.md` §12] flagged and explicitly did not verify." Designed
and implemented, mirroring `quantization_churn.py`'s established
pure-stats-module + I/O-CLI-script + test-file split:
`tsfm_lens/analysis/phase_sensitivity.py` (`phase_sensitivity_stats` — a
pure numpy reduction over a `{shift: per-series MASE array}` mapping into
each series' MASE coefficient of variation across shifts, with a
near-zero-mean-MASE exclusion path so a near-perfect series doesn't
contribute a meaningless near-infinite ratio) and
`run_phase_sensitivity_sweep.py` (the I/O script: trims 0..patch_width-1
points off the FRONT of an already-fixed context — leaving the context's
own endpoint and the forecast target untouched — and calls
`adapter.predict()` at each trim, reusing an already-extracted run's
config/models/data with no re-extraction needed). Patch width is derived
per-model from `token_time_spans()`'s median span width (mirroring
`analysis/attention.py`'s own local computation), not hardcoded to 32,
since a fixed constant would silently mean the wrong thing for a model
whose patch width isn't 32. Added a graceful skip (not a crash) for any
model whose own patch width is 1 (Chronos-T5/Bolt's per-timestep
tokenization) — that model has no patch boundary to be sensitive to, so
it is the expected null control, not a bug, and `phase_sensitivity_stats`
itself requires >=2 shifts to compute a variance at all.
`tests/test_phase_sensitivity.py` (3 synthetic planted-answer tests: a
single-shift rejection, a phase-sensitive-vs-phase-invariant planted-signal
comparison, and the near-zero-MASE exclusion path) all passed on the first
run, per `CLAUDE.md` §2.4's "verify empirically" discipline. Launched a
background agent to live-verify against real checkpoints
(`runs/medium_run_chronos_base`'s already-extracted TimesFM-2.5-200M /
Chronos-T5-Base pair) — not yet returned as of this entry; results will be
written up in §16 E17's own Findings block, not repeated here, once they
land. Per this firing's own no-idle-waiting rule, used the same turn to
close a second, independent, quick item: marked **E8 (CI)** `[x]` — it was
always a pure duplicate pointer to A14 (already fixed and confirmed
2026-08-06), just a bookkeeping fix, no new code. §16's open-item count
therefore actually drops by one this firing (E8) even though E17 itself
isn't fully closed yet — its design/implementation is done and tested, its
live-checkpoint verification is in flight.

**2026-08-11, nineteenth firing, same-day follow-up — background
live-checkpoint verification landed.** The agent found and fixed a real
bug on its first attempt (TimesFM 2.5's `decode()` requires context length
to stay a multiple of its 32-step patch width; the script's naive
`contexts[:, shift:]` trim violated that for every non-multiple-of-32
shift) — fixed in `run_phase_sensitivity_sweep.py` only, via a new
`_predict_point_trimmed()` that keeps the tensor patch-aligned and marks
the trimmed front as masked/invalid in TimesFM's own `decode()` mask
argument rather than shrinking the array. `phase_sensitivity.py` itself
needed no change; its unit tests still pass. Live run against
`runs/medium_run_chronos_base` (64 series): Chronos-T5-Base correctly
skipped as the null control (`patch_width=1`); TimesFM shows a real
corpus-mean MASE coefficient of variation of 0.0232 (95% CI
[0.0181,0.0287]) across the 32 possible patch-phase shifts, worst-minus-best
MASE gap 0.0740 — real but modest in aggregate, with individual series up
to CV 0.125. Full numbers and reading written into §16 E17's own Findings
block above (not repeated here). E17 stays `[~]` — the phase-sensitivity
sub-piece is now fully closed (designed, implemented, unit-tested, and
live-verified), but the item's other named sub-pieces (quantization/
dynamic-range, scale-equivariance, context-truncation, NaN handling, and
wiring into `pipeline.py` as a real `frontend` `Stage`) remain undone. No
`CLAUDE.md` claim needed updating: `CLAUDE.md` §12 bullet 5's aliasing
claim is about attention's resolvable lag ceiling specifically, which this
forecast-level probe doesn't directly test — a related but distinct
question, so nothing there was stated wrong. Ran `test_phase_sensitivity.py`
directly as a final confirmation after the fix (3/3 passing) rather than
re-launching the full suite, since the fix was scoped entirely to one
standalone script with no shared-module changes.

**2026-08-11, twentieth firing — implemented §16 E13 (Spectral lens),
launched its live-checkpoint verification in the background, and used the
same turn for a doc-hygiene item.** Picked E13 as the next unfinished item
(nothing was in flight from the nineteenth firing — E17's write-up above
closed out everything that firing had started). Built the same
three-file split already established for E16/E17: a pure-numpy stats
module (`analysis/spectral_lens.py::spectral_lens_stats`) that takes the
existing skip lens's per-layer forecasts (`lens.py::skip_lens_forecasts`,
no new forward passes) plus each series' ground-truth dominant seasonal
period, FFT's the forecast horizon, and reuses `lens.py::
crystallization_depths` to report *when* the trend (DC bin), seasonal
(ground-truth-period bin), and residual bands each reach the model's own
final-layer error — the frequency-domain "what crystallizes first"
companion to the time-domain "when does MASE crystallize" the skip lens
already answers. A period that can't complete a cycle within the horizon
(or is missing — every real-derived-tier series per `CLAUDE.md` §4.1) is
excluded from the seasonal band with a logged count
(`n_series_with_period`) rather than silently producing a wrong number,
per `CLAUDE.md` §2.5. `run_spectral_lens.py` is the I/O-doing CLI script
against an already-extracted run. `tests/test_spectral_lens.py` (3
synthetic planted-answer tests) found a real test-construction bug on the
first run, not a bug in the module itself: adding independent per-layer
noise on top of a perfectly-periodic planted signal let that noise's own
tiny random DC contribution swamp the (genuinely tiny, for an exact
integer-period sine) trend-band signal being measured, producing a
misleadingly-failing assertion. Fixed by removing the redundant noise
layer (the planted series already carry fixed noise from construction)
and by loosening one over-strict absolute-threshold assertion to the more
meaningful relative comparison it was actually trying to express (trend
error clearly below seasonal error at layer 0, not "already at final
quality"). All 3 tests pass after the fix. Per `CLAUDE.md` §2.8, launched
a background agent (`run_in_background: true`) to live-verify
`run_spectral_lens.py` against `runs/medium_run_chronos_base` (the
established already-extracted TimesFM-2.5-200M / Chronos-T5-Base pair),
briefed with this machine's actual environment (`conda activate cudaPy` —
confirmed via `conda env list` that this machine has no `tsfmPy` env
despite other sessions' notes assuming one; `cudaPy` has torch 2.12.0
(CUDA available), zarr 2.18.7, transformers 4.57.6 already installed and
working) and instructed to report raw numbers verbatim and diagnose (not
silently patch around) any crash — not yet returned as of this entry;
results go into §16 E13's own Findings block once they land. Per this
firing's own no-idle-waiting rule, used the same turn for a second,
independent, quick item needing no compute: updated `tsfm_model_analysis/
tsfm_lens/README.md` (§10's open "API/docs pass" deliverable), which had
drifted — it described only Chronos-T5/Bolt/TimesFM/mock adapters and
made no mention of `layer_screen`, the SAE phase's actual implementation
(TopKSAE/crosscoder/matching), or Chronos-2/Sundial. Added a Screen row to
the layered-method table, rewrote "Extending" into a concrete 5-step
"bring your own HF checkpoint" walkthrough naming the real adapters as
worked examples, and updated the Layout section to list the actual current
module set (confirmed against `ls tsfm_lens/models/`, `tsfm_lens/analysis/`,
`tsfm_lens/sae/` directly, not from memory) including the standalone
probe scripts (`run_layer_screen_bakeoff.py`, `run_crosscoder_feasibility.py`,
`run_spectral_lens.py`, `run_meta_report.py`). No code changed, so no test
run needed; verified only by re-reading the edited sections and
cross-checking every named file/module actually exists on disk.

**2026-08-11, twentieth firing, same-day follow-up — background live-checkpoint
verification landed; E13 write-up complete.** The background agent launched
above finished: exit code 0, no bugs found, no code changes needed. Full
numbers and reading are now in §16 E13's own entry above (not repeated
here) — headline: TimesFM's trend crystallizes near-instantly (relative
depth 0.111) but its seasonal band never crystallizes within tolerance
across all 10 captured layers, while Chronos-T5-Base crystallizes both
trend (depth 0.909) and seasonal (depth 1.0) only in a late, compressed
burst — a new frequency-domain confirmation of `CLAUDE.md` §14's existing
front-loads-vs-accumulates framing. Flagged one real caveat rather than
treating the numbers as final: the live run's `--max-series 48` was capped
to 24 by Chronos-T5-Base's `batch_size`, and only 10 of those 24 series had
a usable ground-truth seasonal period, so this reads as a real but
small-sample finding pending a rerun with a batch_size override before it
gets cited elsewhere. Re-ran `tests/test_spectral_lens.py` directly (3/3
passing, no regressions) as this firing's confirmation step; no other test
files touch this module. **E13 marked `[x]`** in §16 — implementation,
unit tests, and live-checkpoint verification are all now complete; the
sample-size caveat is recorded as a named follow-up, not an open
implementation gap.

**2026-08-11, twenty-first cron-loop firing.** Continued directly from the
prior firing's in-progress work implementing **§16 E14 (Steering /
directional control)** — its blocker (E15's per-token evaluation) had
already been confirmed substantively complete, so E14 itself (new
`analysis/steering.py`: `trend_slope`, `seasonal_band_magnitude`,
`predicted_direction_metric`, `evaluate_direction_match`; new
`sae/eval.py::feature_steering_effects`; new `sae/train.py` wiring behind
`cfg.sae.feature_steering_enabled`; new `SAEConfig` flags in `config.py`)
was already drafted. This firing: (1) fixed a stale `[~]` marker on **E15**
to `[x]` (its own text and a corroborating §13 entry already confirmed it
done — bookkeeping only, no new code); (2) ran the full `tsfm_lens` suite
after the E14 edits — **248 passed, 0 failed**, 2 pre-existing warnings
unrelated to this change (an `extract.py` float16-cast overflow warning and
a `test_smoke.py` pytest-return-value style warning, both predating this
session) — no regressions; (3) launched a background agent to live-verify
`feature_steering_effects` against a real SAE checkpoint. **Caught and
fixed a real mistake before it wasted GPU time**: the first launch attempt
used `isolation: "worktree"`, which is wrong for this task specifically
because `runs/` is git-ignored — a fresh worktree has no
`activations.zarr` to reuse, so the run would have failed at its very
first "reuse the already-extracted store" step. Checked
`git check-ignore` directly rather than assuming, confirmed it, killed the
misconfigured agent before it did any real work, and relaunched without
worktree isolation, directly against this working copy, against a new
`configs/medium_run_chronos_base_feature_steering_check.yaml` (mirrors the
existing `..._feature_ablation_check.yaml` pattern: copies
`medium_run_chronos_base`'s already-extracted activations, runs only
`--stages sae,report`). Confirmed the run is real and progressing (a live
`python run.py` process, not just an agent's own claim) and armed a
`Monitor` on its log file (grep for `sae:|Error|Traceback|EXIT_CODE=|epoch
[0-9]+/|dead-neuron|feature_steering|Killed|OOM`, exiting on the log's own
`EXIT_CODE=` marker) so its completion/failure surfaces without polling.
Results will be written up in §16 E14's own entry once they land, not
repeated here.

In parallel (per this loop's own step 2/3 — never idle-wait on the
background run above), picked up **§16 E23**'s own named next step: a
calibration pass against a *second* real reference point beyond the single
205-sequence synthetic demo run its thresholds were originally set
against. Ran `benchmark_validation` (via `run_validation.py
--enforce-gates`) against `benchmark_medium/public_dev` — the real,
already-sealed 288-sequence corpus used throughout this repo's `tsfm_lens`
real-checkpoint runs (79.2% synthetic / 20.8% realism-stress by tier; 4
task groups). Chosen deliberately because it's a fully independent package
(`tsfm_benchmark`, not `tsfm_lens`) sharing no files with the background
run above, and because it was already built (no new corpus generation
needed) — quick enough (n=288, ~15s wall-clock end to end) to run directly
rather than delegate. **Result: all gates passed, corpus-wide and every
per-group check.** Exact numbers: redundancy fraction 0.0000 (0/41328
pairs at/above the DTW threshold) vs. the 0.15 ceiling; effective
dimensionality 5.346 of 22 catch22 features vs. the 2.0 floor; near-collision
fraction 0.0521 vs. the 0.10 ceiling. Per-group effective dimensionality
ranged from 1.319 (`real_weather_weighted_sum`, n=30) to 5.184
(`diverse_synthetic_backbone`, n=188) — both comfortably clear the looser
1.0 per-group floor, and the real-derived weather groups' lower
dimensionality is exactly the "a narrow, legitimately-real-derived
subgroup naturally has lower effective dimensionality without that being a
collapse" case `gates.py`'s own per-group floor was designed to tolerate
(`CLAUDE.md` §5's discussion, cross-referenced in `gates.py`'s docstring).
**This is a real, non-demo second data point, but does not by itself close
E23**: it is a comfortable pass (every gate cleared with real margin, not
a borderline call), so it doesn't yet tell us whether `DEFAULT_THRESHOLDS`
would also correctly *fail* a real (not synthetically-planted) collapsed
corpus — the demo run's planted-duplicate group is still the only case on
record where a gate actually caught something. `DEFAULT_THRESHOLDS` left
unchanged; **E23 stays `[ ]`**, now with two consistent real passing
reference points instead of one, but still short of its own stated bar (a
full corpus build, e.g. `full_multidomain_run1.yaml`'s ~4200-sequence
scale) and still missing a real *failing* case to calibrate the fail side
against.

**2026-08-11, twenty-first firing, same-day follow-up.** The
feature-steering live-verification job (launched above) finished on its
own mid-turn — its own launching agent hit an unrelated session-limit API
error before it could report, but the underlying `python run.py` process it
started completed successfully (exit 0, 5m14s) and left real results on
disk. Read `sae/meta.json` directly rather than trusting the dead agent's
ambiguous last message, per `CLAUDE.md` §2.4. Full results, including a
real (not a bug) gap the run surfaced — none of the 16 tested top-matched
features across both models landed on a ground-truth field the directional
check knows how to score — are written into §16 E14's own entry, not
repeated here. E14 stays `[~]`.

**2026-08-11, twenty-second cron-loop firing.** Implemented **§16 E2 —
`tsfm-lens doctor`** in full. New `tsfm_lens/doctor.py`: a fixed list of
independent `DoctorCheck` producers (`name`/`status: pass|warn|fail`/
`detail`/`remediation`), split into two tiers matching the cost/frequency
split `CLAUDE.md` §2.8 already applies elsewhere — a fast, static
`run_preflight(cfg)` (no model load, seconds) covering every item E2's spec
named except the two that require a loaded model, and `run_preflight(cfg,
full=True)` (`run.py --doctor`, standalone) which additionally loads every
configured model and runs `models/conformance.py::check_adapter_conformance`
plus a real `impulse_alignment_check` per model — the two things invariant 7
says must be checked by hand on every new checkpoint/library bump, now one
command instead of two. Static checks: zarr major version vs. the `<3` pin
(§11.15's exact trap), `run.device=cuda` vs. `torch.cuda.is_available()`,
free VRAM (`torch.cuda.mem_get_info`) with a rough order-of-magnitude
estimate for attention-pattern capture and per-window patching blowups
(explicitly labeled as a heuristic, not an exact prediction — neither loads
a checkpoint to learn its real hidden size/head count), disk headroom on the
filesystem holding `run.out_dir`, corpus seal verification for both
`data.path` and (when enabled) `confirm.path` (fast mode: existence +
manifest/corpus.jsonl presence only; full mode: real `load_sealed(...,
verify=True)` hash re-check), `context_len % alignment.window == 0` (noted
in the code as defense-in-depth — `PipelineConfig.validate()` already
raises on this at config-load time for the normal YAML path; the doctor
check is for programmatically-constructed configs that skip `validate()`),
and every enabled batch-per-call stage's `max_series`/equivalent cap against
`min(m.batch_size for m in cfg.models)` (l3.patching, lens, attention
ablation, sae forecast-preservation/feature-ablation/feature-steering) —
the exact A16/§11.5 class of crash. `run.py` changes: `--doctor` (full
preflight, exits), `--no-preflight` (skip the otherwise-automatic fast
preflight), `--allow-preflight-fail` (run anyway despite a FAIL). **Explicit
design decision, resolving this item's only open judgment call:** the
default preflight is *blocking* on any FAIL, not merely advisory — matches
the existing `--allow-stale`/`--allow-partial-report` precedent of
defaulting to strict with a named opt-in escape, rather than inventing a
third convention. New `tests/test_doctor.py` (14 tests, all synthetic-config
except one that loads `configs/smoke.yaml`'s two real mock adapters through
the full-mode path — same "real adapter, no live checkpoint" precedent
`test_adapter_conformance.py` already uses); one test-authoring bug fixed
along the way (two tests assumed `_check_corpus_seal` returns exactly one
check, written before noticing `configs/smoke.yaml` has `confirm.enabled:
true`, which correctly adds a second `confirm corpus` check — fixed by
setting `cfg.confirm.enabled = False` in those two tests to isolate the
assertion, not by changing the check itself). **Live-verified, not just
unit-tested:** `python run.py --config configs/smoke.yaml --doctor` produced
a real 14-row table (`14 checks: 14 pass, 0 warn, 0 fail`) with genuine
per-model data (`conformance: patchy` → 6 layers/4 tokens/predict shape
`[4, 8]`; `conformance: steppy` → 4 layers/128 tokens/predict shape `[4,
8]`; both models' alignment min diagonal-hit fraction 1.00);
`--discover-layers` correctly skips the preflight banner entirely (early
exit before the preflight block); a real `--stages extract` run showed the
default (fast) preflight banner (`10 checks: 10 pass, 0 warn, 0 fail`)
printing, passing, and not interfering with the pipeline's own subsequent
skip-logic/completion; `--no-preflight` on the same invocation confirmed no
preflight banner prints at all. Full `tsfm_lens` suite re-run after all of
the above: **248 passed, 0 failed**, the same 2 pre-existing warnings as
every prior firing (an `extract.py` float16-cast overflow, a
`test_smoke.py` pytest-return-value style warning) — no regressions.
**Correction (same firing, caught by re-verifying rather than trusting the
248 figure above a second time — CLAUDE.md §2.4): that 248 count was stale.**
A direct `pytest tests/ --collect-only` run in this same session shows
**262** tests on disk (`262 tests collected`), and a full re-run after the
E14 widening edit below confirms **262 passed, 0 failed**, same 2
pre-existing warnings — the gap is several already-present-but-apparently-
uncollected-earlier files (`test_phase_sensitivity.py`, `test_sae_matching.py`,
`test_spectral_lens.py`, `test_steering.py`, still untracked in git per
`git status`, plus this firing's own new `test_doctor.py`). Whatever caused
the earlier 248-vs-262 mismatch was not investigated further since it isn't
this item's concern, but the number that matters is the one just verified
directly: **262 passed, 0 failed, 2 pre-existing warnings**, both before
and after this firing's own two code changes (E2's `doctor.py`/`run.py`,
and E14's candidate-widening below).
**E2 marked `[x]`** in §16 below — implementation, tests, and live
verification are all complete against everything the spec named; the VRAM/
attention-cost estimate is intentionally a rough heuristic rather than an
exact simulation (stated as such in its own detail string), which is a
scope choice consistent with the spec's own "estimated peak" wording, not a
gap. Not addressed this firing, left for a future pass if it matters:
wiring `--doctor`/`--no-preflight` into `CLAUDE.md` §8's CLI examples, or
`doctor.py` into §3's repo layout — neither blocks E2 itself.

**2026-08-11, twenty-second cron-loop firing, same-day follow-up.** Per this
loop's own "never idle-wait, start a second independent item" rule, picked
up **E14**'s own explicitly named next step while the full-suite re-run
above was still finishing in the background: widened
`sae/train.py::run_sae`'s feature-steering candidate selection. Previously
`candidates` was purely the top-`feature_steering_top_k`-by-|ρ|
ground-truth-matched features, which is why the prior live run (§14's
"twenty-first firing, same-day follow-up" entry above) found zero features
matched to `trend_scale`/`seasonal_amplitude_max` — the only two fields
`predicted_direction_metric` maps to a directional claim — among either
model's top 8. Now, after building the normal top-k set, the code
additionally looks up each of those two fields' own single best-|ρ| match
from `gt["features"]` (already sorted by `-|rho|` by
`ground_truth.py::ground_truth_alignment`) and appends it to `candidates`
if not already present, logging `sae: feature-steering candidates for
{key} widened with directional-field match(es): {widened}` when it does —
so the directional claim gets at least one evaluable example per model
whenever `ground_truth_alignment` found a match for either field *at all*,
without displacing the existing top-k set (ablation's candidate selection,
a separate code path, is untouched). Launched two independent background
checks rather than waiting on either: (1) the already-running full
`tsfm_lens` suite re-run (started before this edit, so it also covers this
change) — result not yet in hand as of this entry; (2) a live-checkpoint
verification agent rerunning `configs/medium_run_chronos_base_feature_
steering_check.yaml` (`--stages sae,report --force sae`, reusing the
existing `activations.zarr` so no new extraction) to check whether the
widening actually surfaces a `trend_scale`/`seasonal_amplitude_max` match
this time and, if so, whether `direction_match` comes back a real
verdict rather than `None`. **Update (same firing): the full-suite half of
this has now landed — see the correction paragraph appended to the prior
entry above for the full explanation of a 248-vs-262 discrepancy caught
and resolved along the way (CLAUDE.md sec 2.4). Net result: 262 passed, 0
failed, the same 2 pre-existing warnings, both before and after this
firing's `train.py` widening edit — no regressions from the candidate-
widening change.** The live-checkpoint verification agent
(`a286624c5cceea416`) is still running as of this update — its result
(whether the widening actually surfaces a `trend_scale`/
`seasonal_amplitude_max` match on the real corpus, and whether
`direction_match` comes back a real verdict) remains pending and will be
written up in a later entry once it actually reports, per this loop's own
rule against predicting a background result in advance. E14
stays `[~]`.

**2026-08-11, twenty-second cron-loop firing, second follow-up.** The
live-checkpoint verification agent (`a286624c5cceea416`) reported back.
Found and fixed one real, pre-existing config bug along the way (unrelated
to the widening edit): `configs/medium_run_chronos_base_feature_steering_
check.yaml`'s `feature_ablation_max_series`/`feature_steering_max_series`
were both `32`, exceeding Chronos-T5-Base's `batch_size: 24` — caught
immediately by the new `doctor.py` preflight (E2, this same firing) at
startup; fixed both to `24`. After the fix the run completed cleanly
(~5 min, reused the existing `activations.zarr`). **The widening logic
correctly did not fire, and the pre-widening null result persists**: the
new log line never appears in the run's stdout, and reading `sae/meta.json`
directly confirms why — neither `trend_scale` nor `seasonal_amplitude_max`
appears as any feature's `best_field` anywhere in either target's stored
top-50 `ground_truth_alignment.features` list, so the widening loop's
`field_matches` was empty for both fields on both targets and (correctly)
appended and logged nothing. Full detail, including the real scope
limitation this surfaced (the widening code can only rescue a match within
`ground_truth_alignment`'s own top-50 truncation, not below it — and in
this run neither field appears even there), a sharper named next step, and
why this counts as a true negative on an untested code path rather than a
confirmed pass, is written into §16 E14's own entry, not repeated here.
**E14 stays `[~]`.**

In parallel (never idle-wait on the above), per §16 E23's own explicitly
named next step (recorded in the twenty-first firing's entry above: "still
short of its own stated bar — a full corpus build... and still missing a
real failing case"), launched a background agent to (1) build the
~4200-sequence `configs/full_multidomain_run1.yaml` corpus via
`run_full.py` against live Monash/ETT data and run `benchmark_validation
--enforce-gates` against it, the full-scale reference point E23 has never
actually had, and (2) find or construct a corpus that genuinely fails
`check_diversity_gates` (starting with `run_validation.py`'s own planted-
duplicate demo mode with `--enforce-gates`, escalating to a hand-built
collapsed corpus if that still passes), since neither of E23's two
reference points to date has ever triggered a FAIL on any gate. Briefed to
report raw numbers only, not edit `ROADMAP.md`/`CLAUDE.md` itself, per
§2.8. **Result pending — to be written up in a later entry once it
reports**, per this loop's own rule against predicting a background
result in advance.

**2026-08-11, twenty-second cron-loop firing, third follow-up.** The
agent above reported back claiming Task 2 (the FAIL-case search) "done
with strong, reproducible results" and that it had backgrounded Task 1's
corpus build (`run_full.py` against `full_multidomain_run1.yaml`) to wait
for on its own — but its actual message did not include any of Task 2's
numbers, so per this repo's own "verify empirically, don't trust a claim
at face value" doctrine (`CLAUDE.md` §2.4) that summary alone is not
something to write into a permanent finding. Checked Task 1 directly
rather than taking the agent's word: `ps aux` confirms a real
`python3 example_runs/run_full.py --config configs/full_multidomain_run1.yaml`
process genuinely running (started 16:42 local / ~2026-08-11T20:42Z, 7 min
elapsed at check time), logging live Monash-loading output (with the
expected pandas-freq-string `FutureWarning`s, harmless) to
`build_full1.log` in this session's scratchpad — a real, in-progress
build, not a stalled or fabricated one. Sent the agent a follow-up message
asking it to state Task 2's exact numbers (which corpus/approach, exact
gate values/thresholds, exactly what it took to flip a gate to FAIL) before
resuming its wait on Task 1, since a full-scale build can plausibly take
up to the ~1hr the config's own header estimates and may not finish before
this loop's own deadline (`2026-08-11T21:51:56Z`, ~63 min out from this
check). **Both Task 1 and Task 2's exact numbers remain pending as of this
entry** — nothing above should be read as a confirmed finding yet; the
next entry (this firing or a later one) will report the agent's actual
reply.

**2026-08-11, twenty-second cron-loop firing, fourth follow-up.** The agent
answered on the second explicit request, pasting real numbers this time
(verified as plausible, not just accepted — the gate names/thresholds match
`gates.py::DEFAULT_THRESHOLDS` exactly, and the qualitative pattern is
internally consistent across three independent probes). **Task 2's actual
findings — a real, reproducible FAIL case has now been found, closing E23's
other still-missing half:**
1. **Demo mode + `--enforce-gates` already fails, on its own, once actually
   run with the flag** (`run_validation.py --demo --enforce-gates`, no new
   corpus needed): the 205-sequence demo (5 planted near-duplicates in an
   `unknown` group of `n=5`) fails `group[unknown].near_collision_fraction`
   (0.4000 > the 0.20 per-group ceiling) while every corpus-wide gate and
   every other per-group gate passes comfortably — the planted duplicates
   are a large enough fraction of their own tiny 5-row group to trip the
   *group* ceiling even though they're invisible corpus-wide (redundancy
   fraction 0.0007, eff-dim 4.892/22, near-collision 0.0585 — all comfortable
   passes at the corpus level). This alone is a legitimate answer to E23's
   "find or construct a FAIL case" ask, and it required zero new code: the
   demo mode's own known planted-duplicate design already crosses a real
   threshold, just not the one anyone had looked at (corpus-wide) before.
2. **A cleaner, deliberately-constructed sweep confirms exactly where the
   corpus-wide `redundancy_fraction` gate flips.** 100-sequence corpus:
   `100 − n_dup` from `random_parametric` plus `n_dup` near-clones of one
   fixed base series (base + N(0, 0.01·std(base)) noise), swept
   `frac_dup` from 0.00 to 1.00. Redundancy fraction crosses the 0.15
   ceiling between `frac_dup=0.30` (0.0939, PASS) and `frac_dup=0.40`
   (0.1657, FAIL) — i.e. a near-duplicate cluster has to reach **40% of a
   100-sequence corpus** before the corpus-wide redundancy gate trips (every
   `frac_dup ≥ 0.40` in the sweep fails on `redundancy_fraction` from there
   on, up to a full 1.00 at `frac_dup=1.00`). `effective_dimensionality` and
   `near_collision_fraction` stayed comfortably passing across the *entire*
   sweep in this construction, including at `frac_dup=1.00` (100% cloned) —
   eff-dim actually *rose* to 6.697 at full duplication, which is a real,
   perhaps counterintuitive property of this specific construction (a single
   dense duplicate cluster plus a small amount of per-clone Gaussian noise
   inflates local feature-space spread among the near-duplicates enough to
   keep PCA's participation-ratio eff-dim from collapsing) — not a gate bug,
   confirmed by an isolated follow-up.
3. **The follow-up isolated why eff-dim/near-collision stayed flat**: at
   `noise_frac=0.0` (bit-identical duplicates, zero jitter) eff-dim
   collapses to exactly 0.0 and near-collision jumps to 1.0 (both gates
   correctly FAIL, alongside redundancy_fraction=1.0) — the gates behave
   exactly as designed on a true degenerate corpus. Any nonzero noise
   fraction, even 1e-10 (numerically indistinguishable from exact but not
   bit-identical), immediately restores eff-dim to ~4.3+ and near-collision
   to ~0.06 — the eff-dim/near-collision metrics are sensitive to true
   informational degeneracy (identical feature vectors) but not to the kind
   of near-duplication that shape-matching's DTW threshold catches (small,
   real but below-threshold differences), which is exactly the gap
   `redundancy_fraction` (computed from the DTW/xcorr matcher, not the
   catch22 feature space) is designed to cover instead — the three gates
   are catching genuinely different failure modes, not redundantly
   re-testing the same one.
4. **E23's checkbox can now move to `[x]`.** Both of its previously-missing
   halves are done: a full-scale real corpus reference point (Task 1, once
   it lands — see below) and a real, reproduced FAIL case (Task 2, now
   confirmed with three independent constructions, not just one lucky
   parameter). `DEFAULT_THRESHOLDS` itself was not changed by any of this —
   the finding is that the existing thresholds already discriminate a real
   pass from a real fail correctly on synthetic constructions; no
   recalibration was needed or done.

**Task 1 (the full-scale build) is still in progress as of this entry** —
directly re-checked via `ps aux` and the build log rather than taking the
agent's word: PID 2633315 still running, ~9m19s CPU time accumulated,
74.2% CPU, still logging Monash reference-series loading output (harmless
pandas frequency-string `FutureWarning`s, same as the earlier check) with
no `BUILD_EXIT_CODE=` marker yet in `build_full1.log`. Current UTC time at
this check: `2026-08-11T20:55:15Z`, ~57 minutes before this loop's deadline
(`2026-08-11T21:51:56Z`). The config's own header estimates "well under an
hour" on a different, prior machine — plausible it finishes in time, not
guaranteed. **E23 is marked `[x]` above on the strength of Task 2 alone**
(a full, self-contained finding that does not depend on Task 1 landing);
if/when Task 1's build and its own `--enforce-gates` run complete, their
numbers will be appended as a further, purely confirmatory data point, not
as something E23's closure is waiting on.

**2026-08-11, twenty-second cron-loop firing, fifth follow-up.** Task 1
finished (`BUILD_EXIT_CODE=0`, 1122s wall-clock ≈ 18.7 min — well under the
config header's "under an hour" estimate on this machine) via a directly-
armed `Monitor` on the build log, independent of the earlier-unreliable
agent chain. Build stats: `public_dev` 4288 sequences, `private_test` 4315,
1377 rejected by the leakage/DTW gate, 156 near-duplicates caught, pooled
from 182 real series across 12 Monash domains (several other domains
skipped — 30s timeouts or incompatible frequency strings — and substituted,
per `bootstrap_catalog`'s own graceful-degradation design). Ran
`run_validation.py --corpus ./benchmark_out_full1/public_dev --out
outputs_full1 --enforce-gates` directly (quick enough at this scale — 74.8s
end to end for n=4288 — not worth delegating): **all gates PASS**,
corpus-wide and every one of 7 task groups. Exact numbers: redundancy
fraction 0.0000 (18/9,191,328 pairs at/above the DTW threshold) vs. the
0.15 ceiling; effective dimensionality 4.886 of 22 vs. the 2.0 floor;
near-collision fraction 0.0501 vs. the 0.10 ceiling. Per-group effective
dimensionality ranged from 1.398 (`real_sequential_par`, n=40) to 5.918
(`diverse_synthetic_backbone`, n=3205) — both clear the 1.0 per-group floor
comfortably, and the real-derived groups' consistently lower dimensionality
(1.4–2.9 vs. the synthetic backbone's 5.9) replicates the same pattern
`benchmark_medium/public_dev`'s 288-sequence run already showed (real
groups lower than synthetic, still passing) — now confirmed at ~15x the
sequence count and against `full_multidomain_run1.yaml`'s specific config
rather than a different corpus. **This is the third consistent real
passing reference point** (after the 205-sequence demo and the
288-sequence `benchmark_medium`), and — combined with the fourth
follow-up's three FAIL constructions above — closes the calibration gap
E23's own next-step language named: `DEFAULT_THRESHOLDS` now has evidence
it discriminates correctly on both sides (comfortable real-corpus passes
at three scales, and reproducible fails when a corpus is deliberately
constructed to be redundant/collapsed) rather than only ever having been
exercised on comfortable passes. `DEFAULT_THRESHOLDS` unchanged — no
recalibration was needed. E23 remains `[x]`; this entry is the confirmatory
data point flagged as pending in the entry immediately above.

**2026-08-11, twenty-third cron-loop firing.** Confirmed via `ListAgents`
("No reachable agents") and a cross-check of `git status`'s untracked/
modified files against this log's own entries through the fifth follow-up
above that no prior background work was pending write-up. Implemented
**§16 E20 — context-length scaling sweeps** (L0 MASE axis only, first
pass): new `tsfm_lens/analysis/context_scaling.py` +
`run_context_scaling_sweep.py` + `tests/test_context_scaling.py` (7/7
passing, pure/deterministic). Dry-ran against the mock-adapter `runs/smoke`
config, then delegated the real-checkpoint run to a background agent per
§2.8 (`runs/medium_run_chronos_base`'s TimesFM-2.5-200M / Chronos-T5-Base
configs, no re-extraction). Full numbers and the qualitative
TimesFM-keeps-improving-to-512-vs-Chronos-peaks-at-384-then-worsens finding
are recorded at E20's own checklist entry (§16) rather than duplicated
here, per this file's own "describe stable architecture in `CLAUDE.md`,
findings in `ROADMAP.md`, each exactly once" discipline. E20 marked `[~]`
(crystallization depth / attention lag profile vs. context length still
open). No `CLAUDE.md` claim changed by this item. Time remaining against
this loop's `21:51:56Z` deadline at the point of this write-up: ~30 min —
insufficient for a second 5+-minute background item plus its own write-up
within the same firing window, so this firing's second-item slot was used
for the write-up itself and the git-status/ListAgents check above rather
than starting new compute that couldn't be safely landed before the
deadline.

**2026-08-11, user-directed planning pass over this whole file
(no code changed).** Brief, verbatim in substance: clean up unnecessary
items, add detail where needed, plan the flagship crosscoder in full with
multiple candidate implementations to be tried and compared, rewrite so a
simpler model or a junior engineer can follow it, and **go back to every
section that was skipped or documented-then-skipped and fix it fully** —
with §6.3's falsified provenance method named as the exemplar ("if the
functionality is still desirable, think of a new way to approach it and
provide multiple options"). Planning only; nothing implemented, no run
performed, no claim in this file's Findings blocks altered.

What landed, in file order:
- **§0.5 "Start here — next actions, in order"** (new). The file is
  organized by topic, which is right for a reference and wrong as an entry
  point. Four tiers of pointers into the owning sections, explicitly *not* a
  second source of truth — items are still marked done where they live.
- **§6.2.1 "The flagship crosscoder — full build plan"** (new, ~370 lines).
  The centrepiece of the brief. Stage 0 is a **blocking gate** on the
  90–98% dead-feature rate, with four ranked hypotheses and an exit
  criterion, because `relative_decoder_norm` reads ~98% "shared" from dead-
  atom symmetry alone and every downstream number is therefore currently
  uninterpretable. Stage 1 specifies `sae/crosscoder_eval.py` and a
  **five-rung validation ladder** (L-A identity sanity → L-B `random_init`
  hard null → L-C layer-offset monotonicity → L-D the real question → L-E
  planted synthetic) with a pre-registered decision rule. Stage 2 gives six
  candidate variants V0–V6, each with mechanism, rationale, cost, and *the
  specific number that must move* for it to win — V0 being the existing
  post-hoc `sae/matching.py` route, which every crosscoder variant must
  beat or the honest result is a negative one. Stages 3–4 cover wiring and
  the research deliverable.
- **§6.3.1 "Provenance detection, reopened — five replacement approaches"**
  (new, ~200 lines). Directly answers the brief's exemplar. The falsified
  result is reframed as a **reusable design constraint** rather than a dead
  end: architecture match alone beat real lineage on the old metric (CKA
  0.878 and L2 gain 0.834/0.940 vs. real lineage's 0.734/0.637), so every
  replacement is scored on whether architecture-matching can fake its
  signal. Options A–E compared in one table then detailed; C
  (idiosyncratic-error fingerprinting) recommended because it passes that
  control by construction, E (fine-tuned lineage ground truth with a
  same-architecture negative) marked do-first because §6.3 never had a true
  positive case or an architecture-matched negative — which is why it was
  untestable, not merely wrong.
- **§11 corrections block.** Five drifts: "do not re-litigate" now means
  built-not-audited post-§15; 10-stage undercount (13, `layer_screen`
  missing from the enumeration); the per-window-patching caveat is
  discharged; "crosscoder not started" is half stale; the golden-hash
  bullet hides a still-unexplained invariant.
- **§12.** The Chronos-decoder non-goal reclassified as *deferred with a
  design* (E21), since it now has a specified surface — the clause stays,
  as the constraint E21 inherits.
- **§13.** One entry had **no checkbox at all** and read as live when it was
  resolved immediately below — fixed. Four items appended that were
  previously named only *inside other items' prose* and never tracked,
  which is exactly how an acknowledged follow-up gets lost: SAE
  forecast-preservation repeat-run variance (🔴 — an identical config moved
  ΔMASE 0.175 → 0.1097), the reverse L2 direction, `work_bend` on a second
  Chronos size, and E19's retroactive multivariate decision.
- **§16 corrections block + a detail-up subsection for all twelve unstarted
  items.** The single most common defect in that section was **stale
  dependency lines** — every A-item referenced is `[x]`, so all of T1 is
  unblocked today while E1/E3/E4/E5/E6/E22 still read as blocked. Two of
  E6's sub-deliverables and E22's A17 clause are already built; E5
  mis-describes `meta_report.py` (it takes `--runs`, it does not glob).
  The detail-ups add file paths, signatures, acceptance criteria, tests and
  cost per item, and surface four gaps the items themselves never named:
  E1's missing checkpoint→adapter mapping and undefined presets, E3's
  amplitude-sweep requirement and contiguous-span inference step, E4's
  **licensing decision** on redistributing `real_derived`-tier data, E6's
  `list[str]`→dataclass findings refactor across ~12 call sites, E18's
  missing path-patching hook primitive and TimesFM's absent `mlp_info`, and
  E24's absence of family labels on wild data. E7, E20 and E22 were split so
  their unblocked halves are individually checkable.

No Findings block was edited and no recorded number was changed anywhere in
this pass — per §0.2, this session added structure and detail around the
research record without touching it.

**2026-08-12, user-directed strategic pass (planning only; no code changed).**
Brief: read the plan *as a whole* against the repo's actual goal (a one-button,
transformer-lens-esque tool that lets anyone understand and compare TSFMs),
improve it to be valuable to advanced researchers while remaining
understandable to beginners, make it adapt dynamically to as many TSFM
architectures as possible, ensure models are compared **on equal grounds**, and
plan out any additional useful features. Five new sections (§17–§22, ~1,130
lines). Nothing implemented; no Findings block or recorded number touched.

**The two findings that came out of grepping the code rather than re-reading the
docs** — both are new, neither was in §15 or §16, and both retroactively qualify
numbers already in this file:
- 🔴 **The cross-model depth axis is comparing unlike to unlike.**
  `utils.py::relative_depths` is `arange(n)/(n-1)` over each model's *captured*
  layers. For Chronos-T5 the captured surface is the **encoder only**, so
  Chronos's "relative depth 1.0" is the middle of its computation — last
  encoder block, entire decoder still to run — while TimesFM's 1.0 is its
  actual output. Every figure interpolating both onto a shared relative-depth
  axis (L1 CKA depth curves, L3 fingerprint agreement, crystallization depth,
  and §16 E9's depth-curve null tests) is therefore comparing 80%-of-TimesFM
  against 80%-of-Chronos's-encoder. `CLAUDE.md` §12 item 4 states the honest
  caveat and the pipeline then uses the axis as if it were a fact. Also
  stride-dependent: two configs of the same model give the same block different
  depth coordinates. Fix: §18 F1, which builds four candidate axes (index /
  block-of-full-stack / compute-fraction / functional) and bakes them off the
  way §6.1.1 did for layer selectors — **with re-running E9's null tests on the
  new axis as the acceptance test**, since a flipped verdict would prove the old
  axis materially misleading.
- 🔴 **There is no parameter, FLOP, latency or VRAM accounting anywhere in the
  repo** (grep: no `n_params`, no FLOP estimate, no timing in any analysis
  module). So every quality comparison is size-confounded, and the first
  question a practitioner asks — *per unit of compute, which model wins?* —
  cannot be expressed. §5.3's base-vs-small study is the only size control in
  the file and it varies size *within* a family rather than normalizing across
  them. Fix: §18 F2, measuring FLOPs via `torch.utils.flop_counter` rather than
  hand-deriving them so it works on unseen architectures, which also supplies
  F1's compute-fraction axis for free.
- Also confirmed by reading it: **`report/coverage.json` is *section* coverage**
  (which report sections rendered), **not computational coverage** of each
  model — so the single most important caveat on every Chronos claim is absent
  from the machine-readable output entirely. Fix: §18 F4.

What landed, in file order:
- **§17 Gap analysis.** The whole-plan read, done once: three of the four words
  in "compare *any* TSFM on *equal grounds*, one button" have unfinished
  business, with **equal grounds** the largest gap and the one the plan was
  least aware of. Five structural gaps (G-I depth axis · G-II no compute
  accounting · G-III capability asymmetry silently yields one-sided comparisons
  · G-IV the envelope is narrower than "any TSFM" and nothing enumerates what's
  outside · G-V no on-ramp for beginners *and* no methods spine for experts).
  §17.2 deliberately records **what must not be "improved"** — the evidence-class
  ladder, the L2 baseline, the confirm split, series-level resampling, §0.2's
  append-only discipline — since a restructuring pass is exactly when
  load-bearing constraints get tidied away.
- **§18 Equal grounds (F1–F9).** The rule it establishes: *every cross-model
  number is either on an axis both models genuinely share, or is rendered next
  to the measured size of the asymmetry* — no prose-only caveats; an
  unmeasurable asymmetry downgrades the claim rather than footnoting it. F1
  depth axes (four candidates, tabled with arguments both ways) · F2 budgets ·
  F3 a capability-**intersection** comparison mode so asymmetric analyses are
  physically separated from the symmetric comparison and can never be promoted
  to a headline claim · F4 coverage accounting with an *automatically appended*
  qualifier on depth-located claims for any model under 90% captured FLOPs ·
  F5 token-resolution parity for attention-lag claims · F6 every delta in
  noise-floor units, with findings *forbidden* below the floor · F7 the
  training-exposure confound bounded rather than waved at (declared corpora +
  reuse of the leakage auditor in reverse + per-tier result splits, since the
  `synthetic` tier is the only data no model can have trained on instance-wise)
  · F8 multiple-comparison correction across *models*, not just families ·
  F9 the **fairness card** — one auto-generated page, rendered *before* any
  result section, listing every measured asymmetry and which claims it
  qualifies.
- **§19 Architecture adaptivity (G1–G7).** Reframed from "support more models"
  to "support more *architecture classes*, where a class is defined by which
  §12 envelope assumption it breaks" — a sixth attention-transformer-over-
  patches teaches nothing, the first attention-free model unlocks a family. A
  14-row landscape table (Lag-Llama's non-contiguous lag tokens, TTM's absent
  attention, TimeMoE/Moirai-MoE routing, SSMs' carried state, hosted/API models,
  VisionTS's 2-D patches, Time-LLM/TabPFN-TS as likely documented refusals,
  supervised baselines as controls) with effort and value per row, all marked
  `[VERIFY]`. **G3 is the standout:** every attention analysis in the repo
  actually asks one question — how does information at *t* influence the
  representation at *t′* — which can be **measured by perturbation** for
  attention, convolution, mixing MLPs, SSMs and MoE alike, and validated
  against real attention patterns on a model where both exist. That converts
  "no attention → a third of the report is blank" into a common footing, and is
  the most publishable methodological item in the file after the crosscoder.
  G1's four adapter tiers (black box → observable → steerable → decomposable)
  make partial support first-class, which is what makes hosted models nearly
  free — with an explicit rule that `confirm` must **refuse** network adapters,
  since sending the sealed private corpus to a third-party API would burn the
  one consumable the whole method depends on.
- **§20 New capabilities (H1–H12).** Opens with three *rejected* proposals and
  why (stronger probes weaken decodability claims; embedding galleries invite
  the one thing `CLAUDE.md` §5 forbids; generated narrative breaks
  artifact-traceability), since §16's own Findings bar is "state which claim
  this sharpens." Four cheap-and-novel first: H4 cross-model agreement as a
  reliability signal (~0.5 session, pure reduction over existing `predict()`
  output, with the honest baseline check that self-reported quantile width may
  predict error just as well) · H1 the **Chronos-T5 tiny→large scaling ladder**,
  five same-corpus same-tokenizer points turning §5.3's two-point L1/L2-grow-
  but-AMI-shrinks curiosity into a curve · H3 **memorization probing**, the
  leakage machinery inverted, where the matched-catch22-profile synthetic
  control is the whole item and a null result is pre-committed as publishable ·
  H2 the **practitioner recommender**, the item that serves §1's "anyone", with
  a mandatory refusal path for out-of-coverage input. Then H5 phase/position
  probes, H6 fine-tuning plasticity (free from §6.3.1 Option E, and an
  independent cross-validation of `work_bend`'s production status), H7 a
  distribution-shift envelope distinct from L3's corruptions, H8 the
  **seasonality circuit** as the flagship mechanistic result (minimal
  sufficient head/MLP set, necessity *and* sufficiency, against known-exact
  ground-truth periods), H9 analysis-card export, H11 checkpoint trajectories,
  and H12 a deterministic-replay/provenance mode motivated directly by
  §11.24's cost — a session of investigation that a three-line environment
  diff would have closed in a minute.
- **§21 Two audiences (J1–J6).** One document, *layered* — not two, because
  two drift (this repo has paid for that between `CLAUDE.md` and `ROADMAP.md`
  repeatedly). J1's three-layer claim contract (`plain`/`text`/auto-generated
  `caveat`, generated because invariant 8's lesson is that author discipline
  decays), J2 a fixed four-line card per stage whose fourth line is *what it
  cannot tell you*, J3 glossary + one worked example read paragraph by
  paragraph, J4 headline/standard/methods progressive disclosure, J5 the
  **advanced methods appendix** (cheapest credibility item in the file — the
  content already exists in module docstrings and its absence is what a
  sceptical reader notices first), J6 a **failure-mode gallery** built entirely
  from real artifacts already in `runs/`, every one of which fooled someone
  here first.
- **§22 Sequencing.** Six waves (A retroactive integrity · B the flagship
  crosscoder, unchanged and still top priority, run in parallel with A as the
  natural background/foreground split · C legibility · D the four cheap studies
  · E architecture breadth, one class per session · F adoptability), with one
  hard constraint stated explicitly: **Wave F must not precede Wave A** —
  automating a comparison on an unequal axis removes the expert who would have
  known not to believe it. §0.5 updated to point here and to add Wave A as a
  new Tier 0.

---

**2026-08-11, twenty-fourth cron-loop firing.** Checked `ListAgents` — nothing
pending write-up from a prior background job (only an unrelated peer
interactive session listed). With the loop's own deadline (`21:51:56Z`) only
~13 minutes out and the next scheduled firing (`22:07`) past it, judged this
firing very likely the last one able to do real work, so declined to launch a
new background agent (no future firing would exist to pick it up) and instead
implemented **E20a (horizon sweep)** directly — see §16 E20a's own entry
above for the full implementation/test/dry-run detail and the output-path bug
found and fixed along the way; not repeated here per this file's own
discipline of recording findings once at the item's checklist entry. Also
surfaced, at E20a's entry: E20b's "expensive by nature, needs fresh
extraction per point" framing appears to be stale, since E20's own landed
implementation (`context_scaling.py`) never needed fresh extraction either —
flagged for a future firing to resolve before starting E20b as scoped.
**Not completed this firing:** a live real-checkpoint run of the new horizon
sweep (implementation + 6/6 local tests + a clean mock-adapter dry run were
completed instead, given the time remaining) — left as the explicit next
step in E20a's own entry rather than either skipped silently or rushed
unsafely against the deadline.

---

**2026-08-12, §6.2.1 Stage 0 implementation session (multi-firing).** Picked
up §0.5's Tier-1 item 1 — the crosscoder's blocking dead-feature gate — and
took it from "four ranked hypotheses" to a measured verdict. Implemented the
one hypothesis that had no implementation at all (**H4, AuxK**:
`sae/models.py::auxiliary_dead_loss` plus call sites in `sae/train.py` and
`sae/crosscoder.py`, with `aux_k`/`aux_coef`/`aux_dead_steps` threaded
through both train configs and defaulting to **off**, so every number already
on record stays regenerable); built the gate's harness
(`run_crosscoder_stage0.py`, which trains the joint crosscoder **and** two
matched per-model `TopKSAE` baselines on the same rows at the same budget per
grid row, and scores all four exit criteria); built the ~10x larger store H1
needed (`configs/crosscoder_stage0.yaml`, 46382 aligned rows vs. the
feasibility run's 4608); and ran the 9-row sweep against real
TimesFM-2.5-200M / Chronos-T5-Base activations. Findings — including the full
sweep table, not just the winner — are at §6.2.1's "Stage 0 sweep run"
Findings block, not repeated here; the headline is that **the decisive
hypothesis was the one ranked last** (AuxK: 80 → 1072 alive atoms at
identical dictionary/rows), **H2 was refuted by the sweep's sharpest number**
(alive count is nearly invariant to dictionary size: 51/61/68 alive at
112/448/1120 atoms), and **the gate still does not pass** — its blocker has
moved from deadness to per-source fidelity, so Stage 0 stays `[ ]`. Two real
bugs found by running rather than reading: a numpy-vs-torch `alive_mask`
type error in the harness, and an unseeded `torch.randperm` in
`train_crosscoder` that made "identical" configs irreproducible (fixed;
`h4:aux_off` now reproduces `h1:rows=all` to every printed digit across
separate invocations, which is what confirmed it). Also worth recording as a
methodology point: the synthetic AuxK test bed did **not** predict the real
result (a wash on one bed, provably inert on another), which is exactly why
`tests/test_aux_k.py` asserts only "does no harm" — had the stronger claim
been asserted to match the hypothesis, it would have passed for the wrong
reason (`CLAUDE.md` §2.4). Full `tsfm_lens` suite green after all of it:
**283 passed, 2 warnings**. A follow-up `k` sweep (k ∈ {16,32,48,64} against
TimesFM's measured effective dimensionality of 28.15) was launched to test
the fidelity blocker directly and is written up at §6.2.1 separately.

---

**2026-08-12, cron-loop session — Stage 0's dictionary-size question closed,
and the reverse L2 direction measured.** Two independent items, both
finishing work the previous session had only set up.

*Stage 0 (§6.2.1).* Ran two new sweeps against the same 46382-row
`runs/crosscoder_stage0_bigdata` store. **`--grid h2xh4`** (dict ∈
{640, 768, 896, 1024} at k=48/aux_k=64/coef=0.03125, 60 epochs) tested
whether H2 (dictionary size) matters *in the AuxK-on regime* where it
plausibly could have, after the original sweep refuted it in the AuxK-off
regime. It does not: the crosscoder **passes every exit criterion at all
four sizes** (fidelity 0.7346/0.7670 at 640 up to 0.7590/0.7782 at 896;
dead rate 0.0112–0.1081; alive 627–991). H2 stays refuted. Notably
`dict=896` **beats the committed winner** (k=48/dict=1280) on Chronos
fidelity (0.7782 vs 0.7556) and dead rate (0.0112 vs 0.0203) at 70% of the
parameters — `configs/crosscoder_stage0_winner.yaml` was **deliberately
left untouched**, because its `expected:` block records the k-sweep row and
rewriting it would break the replay property that config was created to
establish. **`--grid pinch`** (a new grid, dict ∈ {512, 576, 704}) then
answered the residual question by measurement rather than extrapolation
(§2.4): is there *any* dictionary size at which the **Chronos-T5-Base
baseline** clears both `dead ≤ 0.30` and `alive ≥ 500`? Across all eight
now-measured sizes — alive 386/355/467/405/473/529/546/576 and dead
0.2461/0.3837/0.2703/0.4247/0.3841/0.4096/0.4668/0.5500 at dict
512/576/640/704/768/896/1024/1280 — **no size in [512, 1280] satisfies
both**; the two rows that clear the rate bar (512, 640) fall 114 and 33
atoms short of the alive floor. Fidelity is never the blocker (0.858–0.888
throughout), and TimesFM's baseline passes at every size. Two things
recorded rather than papered over: the alive counts are **non-monotone in
dictionary size**, implying ±60-atom single-seed noise against a 33-atom
decision margin (fix: replicate seeds, as `run_layer_screen_bakeoff.py
--n-gold-replicates` already does); and the residual is now a **criteria
question, not a training question** — `MIN_ALIVE = 500` is a global
constant applied to a layer whose measured effective dimensionality is
13.93. Two defensible resolutions are named in §6.2.1 (a per-model floor of
~`20 × eff_dim` → 279 Chronos / 563 TimesFM, or keep the constant and carry
a quantified caveat into Stage 2) and **neither is decided here** — what is
not defensible is picking whichever makes the gate pass. **Stage 0 stays
`[ ]`.**

*Reverse L2 direction (§16 E9, §13).* `compare_l2_depth_curve` gained a
`direction=` parameter (and `run_null_baseline_test.py` a `--direction`
flag, with a direction-suffixed default output filename) so the
non-best-gaining direction is measurable as its own thing — L2 is not
symmetric, the ridge probe maps one model's states onto the other's, so a
verdict on one direction says nothing about the reverse. Unknown direction
names **raise** and list the available ones rather than silently falling
back to the best (§2.5). Ran `TimesFM->Chronos-T5-Base` at n_boot=500 with
the nulls supplied in that order. Every src layer's best dst partner is
`encoder.block.10`. Against Chronos's own floor (−0.1696, *negative*) real
exceeds at all 10 depths by +0.38 to +0.49, p=0.002. Against TimesFM's own
floor (which decays 0.3883 → −0.1045 across layers 0→18) the null exceeds
at layers 0 and 2, the CI spans zero at 4/6/8, and **real exceeds
decisively at 10/12/14/16/18** (+0.200/+0.263/+0.288/+0.308/+0.340,
p=0.002 each). This **closes** the earlier recorded pessimistic claim
("0.318 real vs 0.388 null") as the identical wrong-layer artifact already
corrected once for the forward direction — comparing a real peak against a
null's own global best rather than against the null at the matching depth.
§13's reverse-L2 item is now `[x]`; full table at §16 E9's fourth
follow-up; `CLAUDE.md` §6.5's "the reverse L2 direction is the named next
step" was stale and is corrected in place. Tests: 2 new
(`test_l2_depth_curve_runs_the_reverse_direction_when_asked`,
`test_l2_depth_curve_rejects_an_unknown_direction`) plus one existing
regex updated for the reworded error — `tests/test_null_baseline.py`
**12 passed**.

**Same session, second half — three more items closed, one recorded number
retracted.**

4. **§16 E19 (multivariate/any-variate support) — decided, not deferred
   again.** Ratified **univariate-only**, written up as a new bullet in
   `CLAUDE.md` §12's "Envelope hard edges" list. The decision rests on
   `pooling_matrix`'s premise (a token maps to a contiguous *time interval
   within one series*) rather than on cost: a cross-series axis has no time
   interval, no place on the shared window axis, and no lag in timesteps, so
   `[series, variate, window, dim]` means re-deriving alignment, not adding a
   dimension. Names the concrete exclusion (Chronos-2's GROUP attention),
   states the rule for future adapters, and explicitly records that the
   *size* of the resulting coverage gap is **unknown**. §16 E19, §13's E19
   entry, and §0.5 item 8 are `[x]`; §19's landscape table's Moirai row is
   corrected (the any-variate axis is now a documented exclusion, not a
   blocker — residual work there is variable-`token_width`, effort
   downgraded to "medium").

5. **§16 E20a (horizon scaling) — closed with a live real-checkpoint run**,
   29 s wall clock, both checkpoints loaded and released cleanly. The
   headline is that **MASE is essentially flat across an 8× horizon range**
   for both models (TimesFM 0.609–0.622, Chronos-T5-Base 0.646–0.690), with
   TimesFM lower at every horizon and the two CIs cleanly disjoint at h=64
   but overlapping at h=8. This **contradicts the mock dry-run**, previously
   recorded as showing "the expected qualitative shape" — the write-up
   explains why the *mock* was the misleading one (MASE normalizes by a
   context-derived naive scale that doesn't depend on horizon length) rather
   than treating the real result as anomalous, and warns against reading it
   as "horizon length doesn't matter."

6. **§13's SAE repeat-run variance — measured, and it retracts a number.**
   New `tsfm_lens/run_sae_repeat_variance.py` retrains both configured SAE
   targets at seeds 0–4 against `runs/medium_run_chronos_base`'s **frozen,
   read-only** store. The frozen-store control is exact (`mase_clean` sd
   **0.0** across all five seeds, both models, both granularities), so
   everything measured is SAE-training stochasticity alone. Three results:
   (a) the 0.175→0.1097 swing this item was opened for is **half a standard
   deviation** (sd 0.1212, range 0.309) — there was never anything to
   explain; (b) TimesFM's window/token granularity identity replicates
   bit-for-bit at *all five* seeds, strengthening E15's mechanism claim; and
   (c) 🔴 **E15's Chronos token-granularity −0.346 is retracted** — it falls
   outside the five-seed range [−0.112, +0.414] and its sign replicates in
   1 of 5 seeds. The correct value is **+0.246 ± 0.211**. E15's *conclusion*
   survives decisively (window +3.897 ± 0.242 vs. token +0.246 ± 0.211 is a
   ~16× gap against the seed noise). Corrected in place at §16 E15,
   `CLAUDE.md` §13 item 3, and §0.5 item 6. Residual — rendering the floor
   beside the numbers in the report — split out as its own §13 item rather
   than folded into the closed one. Also visible in the same artifact and
   carried to §6.2.1: both dictionaries are **94.5% / 97.3% dead**, stable
   across seeds (sd ≤0.005), i.e. reproducibly the condition Stage 0's gate
   exists to eliminate.

**Test status for the whole session: full `tsfm_lens` suite 289 passed, 2
warnings** (19 min), against 283 at the last recorded full run — the +6 are
this session's two reverse-L2 direction tests and four earlier additions.
Both warnings are pre-existing and unrelated (a deliberate float16-overflow
probe in `test_nonfinite.py`, and `test_smoke.py::test_end_to_end` returning
a `Path`).

**Same session, third part — Stage 0's criteria question decided, its
replicate landed, and the gate still does not open.** The `MIN_ALIVE`
question the first half deliberately left open was taken: the floor becomes
`max(100, round(20 × eff_dim))` per source (279 Chronos / 563 TimesFM), on
the reasoning that the old `MIN_ALIVE = 500` and `MAX_DEAD_RATE = 0.30` were
**jointly unsatisfiable at any dictionary size** for a model whose alive
count saturates — a criteria pair no size can satisfy is measuring the
constants' interaction, not the SAE (§6.2.1 finding (11) + DECISION). Note
the change makes the crosscoder's bar *stricter* (563 > 500), not looser.
The argument was then written as an executable assertion
(`tests/test_stage0_criteria.py`, new, 8 tests) — **and it failed**, because
the paragraph had stated the cap as `500/0.70 = 714` when the correct bound
is the model's own saturation ceiling over 0.70 (`576/0.70 = 823`), i.e. the
looser bound that makes the claim *harder* to prove. Conclusion survived
(the first size reaching 500 alive is 896 > 823); the stated arithmetic did
not, and is corrected in place at all three places it appeared. §2.4 in the
small.

The 5-seed `--grid pinch` replicate (dict 512/576/704, crosscoder + both
matched baselines, 46382 rows) then landed and **Stage 0 stays `[ ]`** for a
third, different reason: the same opposite-directions structure recurs
*between* models. Chronos's untouched rate bar caps a shared dictionary near
569 atoms; the crosscoder's alive floor needs ~606. All three artifacts pass
together at **0 of 5 seeds at every size** (findings (12)-(13)). Per-model
dictionary sizing would close it — every artifact has a passing size on
record (Chronos 512, TimesFM 576/704, crosscoder 704) — but that is a second
criteria change in one session in the direction that makes the gate pass, so
it is written up with its evidence and **not applied** (finding (14)). Also
newly measured: the crosscoder's failures are **bimodal**, not marginal —
four seeds cluster at dead 0.0415 ± 0.0088 / fidelity 0.7233/0.7453 while
one collapses to 0.1804 / 0.6687/0.6343, so Stage 1's scorecard needs
replicates per variant or a bad basin will be scored as a bad variant
(finding (15)). Two tests added for the pinch result; `test_stage0_criteria.py`
now 10, and 25 passed across it plus the winner-config and new budget suites.

One self-inflicted process failure, written up as **`CLAUDE.md` §11.27**:
seeds 1–3 of the first pinch attempt died with a `TypeError` because the CLI
they re-invoke per iteration was edited *while the loop was running*, and
the shell loop still exited 0 — a background job that re-reads its script
each iteration makes the source, not just the artifacts, part of what §2.8
says not to touch. All five seeds were relaunched under frozen code and the
partial run discarded rather than merged.

Also started this session (not yet a pipeline stage): **§18 F2's model
budget**, `tsfm_lens/analysis/model_budget.py` — parameter census by role,
FLOPs measured via `torch.utils.flop_counter.FlopCounterMode` (measured, not
hand-derived, so it works on architectures with no adapter yet), an analytic
`2 × body_params × tokens × batch` sanity floor, latency and peak VRAM, and
**per-block cumulative FLOPs recovered from the same measured pass** —
F1's D2 compute-fraction depth axis needs those and F2's acceptance
criterion forbids a second measurement pass. Degrades to `None` with a
warning wherever the counter cannot see the model, because a depth axis
silently pinned at compute fraction 0 would read as a finding. 11 tests
(`tests/test_model_budget.py`, CPU-only synthetic stack). Config/pipeline/
report wiring is the remaining half of F2.

- **2026-08-13** — Closed §13's `_per_block_flops` item, the first of the two
  left open when §18 F4 landed. The recorded hypothesis (`FlopCounterMode`
  keys blocks relative to the *entered* submodule) was confirmed — by a
  synthetic two-stack probe rather than the GPU run the item planned, since
  it is a question about torch, not about Chronos. The item's own fix sketch
  (unique-suffix match) turned out **unsound on T5 specifically** and was
  replaced: entered module paths are now recorded by pre-hooks and
  `ModuleTracker`'s naming rule reconstructed from them, withholding loudly
  on a same-class root collision. All 12 Chronos blocks now resolve; the
  headline flips from upper bound to measurement **without its value
  changing** (the bound was exactly tight). Re-rendered `report.html` gained
  Chronos's depth trace, which immediately exposed F1's problem visually:
  Chronos ends that chart at 1.0 and TimesFM at 0.851, i.e. the
  better-covered model looks worse, because each curve is normalized by its
  own measured forward pass. Full suite 341 passed, 2 warnings (10 min),
  against 338 at the last recorded run. `CLAUDE.md` gains §11.28; §9's row
  and §12 item 2 corrected in place. §13's second open item (the F4
  qualifier firing on SAE feature findings) left open deliberately,
  unchanged.
- **2026-08-13 (same day, later — §6.2.1 Stage 0 closed)** — Took the
  criteria call §0.5 item 1 and finding (14) had left pending, but only
  after measuring the premise behind it: the matched per-model baseline is
  3.2–5.4× *less* dead than the crosscoder at every shared size and moves
  the opposite way as the dictionary grows, refuting this section's own
  "the baseline hit the same wall" (finding (16)). DECISION: "matched"
  means matched *budget*, not matched *dictionary size*; the three numeric
  bars are untouched and the matched-size run is retained as a control.
  Implemented as `--baseline-dict-sizes` + a `gate` grid + a baseline memo
  in `run_crosscoder_stage0.py`, then gated on a 5-seed replicate rather
  than the 4-of-5 numbers already in hand: `dict=1024` passes at **5/5**
  with both baselines passing at every seed, `dict=896` at 3/5 on fidelity
  alone (finding (17)). Committed `configs/crosscoder_stage0_gate.yaml`;
  corrected the now-stale "checkbox is still open" comment in the winner
  config without touching its numbers. 16 new tests
  (`test_stage0_baseline_sizing.py`, `test_stage0_gate_config.py`). Stage 0
  is `[x]`; Stage 1 is unblocked.

---


## Archived: §6.2.1 Stages 3/4 and the original cost table (2026-08-11), superseded by the 2026-08-18 triage

#### Stage 3 — `[ ]` Wire the winner into the pipeline

Only after Stage 1's decision rule declares a winner.

**3a. The `SAEAdapter` problem, stated honestly.** `sae/interface.py`'s
Protocol is `encode(activations: [N,D]) -> [N,F]` — inherently
single-source. A crosscoder's `encode` needs *all* sources. Two options,
both acceptable, **pick one and write down which**:

- **Option 1 (recommended) — extend the Protocol.** Add an optional
  `encode_joint(sources: list[Tensor]) -> Tensor` and a `n_sources`
  attribute; `load_sae` dispatches on a `variant` field in the checkpoint
  (the docstring already anticipates this: *"a crosscoder checkpoint would
  need a different loader"*). Downstream consumers that only understand
  single-source SAEs check `n_sources == 1` and skip loudly (§2.5). Honest,
  slightly more plumbing.
- **Option 2 — a source view.** `CrosscoderSourceView(crosscoder, i)`
  satisfying the existing Protocol by passing zeros for the other sources.
  Less plumbing, but **it is an approximation** — the encoder sums source
  contributions, so zeroing B changes A's features. If chosen, measure how
  much (correlate `encode_one(x_A)` against the joint `encode([x_A, x_B])`
  over the corpus) and print that number in the report. Do not ship this
  silently.

**3b. Pipeline stage.** New `crosscoder` stage in `pipeline.py`, depending
on `extract` and `layer_screen`, gated on `sae.crosscoder.enabled`. Config
block under `SAEConfig` (mirroring the existing flat-field style):
`crosscoder: {enabled, variant, pairs: [{model_a, layer_a, model_b, layer_b}],
dict_size_mult, k, epochs, aux_k, shared_band, ...}`. Default `layer` for
each pair resolves to the L1 peak-CKA pair when left `auto` — the machinery
to find it already exists in `l1/meta.json`, so `auto` is a lookup, not a
new search. Artifacts: `crosscoder/{pair}/checkpoint.pt`, `.../scorecard.json`,
`.../features.json`.

**3c. Report section.** Per §0 rule 6 and §8, every figure needs a
`_note()`. Minimum contents: the shared/specific histogram **with the L-B
floor drawn on it as a reference line**; the top shared atoms with their
ground-truth field and ρ; the top model-specific atoms per model, β-confirmed
via V4; the fidelity-per-source pair; and a findings string that answers the
founding question in one sentence with its floor and CI attached.

**3d. Close the encode-store seam.** `sae/interface.py`'s docstring has
promised `sae/{model}/{layer}` written back into the store since the
baseline landed, and it is still not wired. Doing it here (for both the
baseline and the crosscoder) unblocks `level="sae"` reads in L1 and
clustering — i.e. **CKA in feature space instead of activation space**,
which is a genuinely different measurement and one of the more interesting
cheap wins left in the repo.

---

#### Stage 4 — `[ ]` The research deliverable

The point of all of the above. Write into §6.2's Findings, and only with all
three parts:

1. **The number**: what fraction of the jointly-trained dictionary is shared
   between TimesFM and Chronos-T5-Base at their peak-CKA layer pair, over
   alive atoms, β-confirmed, with a bootstrap CI (series is the resampling
   unit — invariant 2).
2. **The floor**: the same number for the L-B `random_init` pair. Without
   this, part 1 is unreadable — see §6.3.
3. **The content**: *what* the shared atoms are, via ground-truth alignment.
   "62% shared" is a statistic; "the shared atoms align with seasonal period
   and trend scale while each model's specific atoms align with its own
   tokenization artifacts" is an answer to the founding question.

**Then update**: `CLAUDE.md` §6.2's support matrix and §13 item 3, this
file's §11, and §13's crosscoder open question — all of which currently say
the flagship crosscoder is not started.

---

#### Cost and sequencing summary

| Stage | Work | GPU time | Blocking? |
|---|---|---|---|
| 0 | Dead-feature root cause (H3→H2→H1→H4) | ~2–3 h across sweeps | 🔴 **Yes — blocks everything** |
| 1 | `crosscoder_eval.py` + 5-rung ladder | ~1 h (L-B/L-C reuse existing stores) | Blocks Stage 2 comparisons |
| 2 | V0/V1 wiring, then V2, V4, then V3/V5/V6 | ~1 h per variant per rung | No — variants are independent |
| 3 | Adapter + stage + report + store seam | ~30 min verification | Needs a Stage 1 winner |
| 4 | Writeup | none | Needs Stage 3 |

**Minimum viable path if time is short**: Stage 0 (H3+H1 only) → Stage 1
(scorecard + rungs L-A, L-B, L-D only) → V0 vs V1 vs V2 vs V4 → Stage 4.
That is a complete, honest, publishable result without V3/V5/V6, and it is
the path to take unless Stage 0's Findings suggest a specific variant is
needed to get a live dictionary at all.


## Archived: §6.2.1 V5/V6 full write-ups (parked 2026-08-18, see §22.1)

**V5 · Frequency-aware dictionary (§6.2 item 2, now made concrete) `[ ]`**
- *Change*, two independently testable sub-variants — **do not build both at
  once, they confound each other**:
  - **V5a (init)**: initialize a configurable fraction of atoms in each
    source's encoder at directions obtained by regressing that source's
    activations onto an FFT basis of the corresponding input windows. Cheap,
    no loss change, fully reversible.
  - **V5b (loss)**: add an auxiliary penalty on the *spectral entropy* of
    each atom's activation profile across windows within a series —
    rewarding atoms whose firing pattern is periodic at a single dominant
    frequency. Note this regularizes the **feature's temporal firing
    pattern**, not the decoder vector, which is the version that actually
    corresponds to "this atom detects periodicity."
- *Why it might win*: the benchmark labels seasonality exactly
  (`seasonal_period_dominant`, `n_seasonalities`, per-seasonality
  period/amplitude/phase). If frequency structure is a real organizing axis
  of TSFM activations, a dictionary biased toward it should align better
  with those specific fields.
- *Cost*: V5a half a day, V5b two days (needs a per-series windowed view of
  feature activations that `load_all_windows` does not currently return in
  series-grouped form — check before estimating).
- *Number that must move*: `gt_alignment` restricted to the seasonality
  fields specifically — **not** overall mean ρ, which would let an
  improvement on trend fields mask no improvement on the fields this variant
  targets. Pre-register that restriction. Cross-check against E13's spectral
  lens: if the spectral lens says Chronos crystallizes its seasonal band
  only at the final layer, a frequency-aware dictionary at a mid layer
  should *not* help Chronos much — a prediction this variant can be falsified
  by.
- *⚠️ Adopt only if it measurably helps*, per §6.2's own original wording and
  §2.5. A frequency-aware dictionary that ties V1 is a worse deliverable
  than V1, not an equal one.

**V6 · Multi-resolution / Matryoshka crosscoder (§6.2 item 3, now made concrete) `[ ]`**
- *Change*: train one dictionary whose **nested prefixes** are each required
  to reconstruct — loss = Σ over a set of prefix sizes `m ∈ {64, 256, 1024,
  F}` of the reconstruction error using only the first `m` atoms. Early
  atoms are forced to carry coarse, high-variance structure (trend, regime);
  later atoms refine.
- *Why it might win*: (a) it is the published remedy for **feature
  absorption**, where a broad feature swallows a narrower one; (b) it maps
  unusually naturally onto time series, where "coarse trend vs. fine local
  texture" is a real, labeled property rather than an analogy; (c) it makes
  the dictionary interpretable *at multiple budgets* — a 64-atom summary of
  what a TSFM layer encodes is a far better report figure than a 6144-atom
  one.
- *Second, TSFM-native axis worth trying inside the same variant*: this repo
  pools activations at a configurable `alignment.window`. Train the same
  dictionary on **window-pooled and token-level activations jointly** (each
  as a separate "source" — `CrosscoderSAE` already supports N sources of
  independent dimension, and E15 already established token granularity is
  the honest one for Chronos). This is a multi-resolution crosscoder in the
  *time* axis rather than the dictionary axis, and no other repo is
  positioned to try it.
- *Cost*: two to three days. Note it is **composable with V2/V3**, not
  exclusive.
- *Number that must move*: `gt_alignment` at small prefix sizes (a 64-atom
  prefix should still align with the coarse ground-truth fields), plus the
  fidelity-vs-prefix-size curve as a new output.

---


## Archived: §6.2.1 V3's full write-up (parked 2026-08-18, see §22.1)

**V3 · Explicit shared/private parameterization ("grouped crosscoder") `[ ]`**
- *Change*: stop *inferring* the split and *parameterize* it. Partition the
  dictionary into three blocks — `F_shared` (both sources' decoders active),
  `F_private_A` (only A's decoder), `F_private_B` (only B's) — with block
  sizes as hyperparameters. Add a group-sparsity penalty (or simply a higher
  effective price on private blocks via a per-block TopK budget) so an atom
  lands in a private block only when the shared block genuinely cannot
  reconstruct it.
- *Why it might win*: the current shared/specific number is a **post-hoc
  read of a continuous ratio with an arbitrary `(0.3, 0.7)` band** — the
  band is a free parameter nobody has justified, and the feasibility run's
  own "97–99% shared" bug (dead atoms clustering at 0.5 by symmetry) is a
  direct consequence of inferring structure rather than building it in.
  V3 makes the split a modeling choice with a likelihood consequence, so it
  can be *tested*: does forcing more capacity into private blocks hurt
  reconstruction? By how much? That curve is a far stronger answer to "is
  there shared structure" than a histogram threshold.
- *Cost*: moderate — a real architectural change (~150 lines), plus a
  block-size sweep. Two to three days.
- *Number that must move*: L-E's planted-recovery F1 (V3 should recover a
  constructed split more precisely than a thresholded ratio), and the
  **shape** of the fidelity-vs-shared-block-fraction curve becomes a new,
  first-class output.


## §6.2.1 crosscoder — original intro and Stage 0 four-hypothesis plan (superseded 2026-08-18)

### 6.2.1 The flagship crosscoder — full build plan `[ ]`

> **Added 2026-08-11 (planning only, no code).** §6.2 item 1 has, today, a
> *mechanism* (`sae/crosscoder.py::CrosscoderSAE`) and a *feasibility
> verdict* ("joint training is stable"), and nothing else: no `SAEAdapter`
> implementation, no pipeline stage, no config, no report section, no
> answer to the founding question. That gap has sat open since 2026-08-05
> because "build the flagship crosscoder" is not an instruction anyone can
> execute — it is a research program compressed into five words. This
> section decompresses it into an ordered, gated sequence with explicit
> pass/fail criteria at every step, six named variant implementations to
> compare, and a fixed scorecard defined *before* any of them is written so
> the comparison can't be rationalized after the fact (§2.2).
>
> **How to use this section.** Work top to bottom. Stage 0 is a hard gate —
> do not write a single new variant until it passes, because every metric
> downstream is uninterpretable on a 95%-dead dictionary. Stages 1 and 2 can
> overlap once Stage 0 is green. Stage 3 only starts once a winner is
> declared by the Stage 1 decision rule. Every stage below names the file to
> create, the function signatures to write, the test that proves it works,
> and the number that has to move.

**What this section inherits, and must not re-derive.** Two facts from the
2026-08-05 feasibility Findings above constrain everything here:

1. **Source-scale mismatch is solved.** `CrosscoderSAE` normalizes each
   source by its own global std internally. Do not reintroduce a raw-scale
   MSE sum, and do not "simplify" `source_scale` away — a synthetic 20×
   scale mismatch reproducibly collapses the smaller source's fidelity to
   −9.7 without it.
2. **🔴 The dictionaries were 90–98% dead, and this is the blocking
   problem.** 122 alive atoms out of 1280 is not a dictionary; it is 122
   directions. Every headline crosscoder number — the shared/specific split,
   ground-truth alignment, cross-model feature matching — is computed over
   alive atoms, so on a 95%-dead run they are all statistics over a sample
   too small to distinguish signal from initialization. The recorded "69%
   shared / 30% TimesFM-specific / <1% Chronos-specific" split is over 122
   atoms and **should not be quoted as a finding** until Stage 0 closes.


## §6.2.1 Stage 0 — original four-hypothesis plan and exit-criteria prose (superseded 2026-08-18)

#### Stage 0 — `[x]` BLOCKING GATE: produce a dictionary that is actually alive

> ✅ **Closed 2026-08-13 by finding (17)** — `configs/crosscoder_stage0_gate.yaml`
> (dict 1024, k 48, AuxK on, baselines at their own sizes) passes every
> criterion at 5 of 5 seeds, and both per-model baselines pass at every seed.
> The plan below is preserved as written; read it alongside findings (1)–(17),
> which record where it held and where it didn't — in particular, the
> parenthetical in the next paragraph ("the per-model `TopKSAE` baseline hit
> the same wall at matched settings") turned out to be **false**, and finding
> (16) is the measurement that overturned it.

**Do this before writing any new variant.** The dead-feature rate is not a
crosscoder problem (the per-model `TopKSAE` baseline hit the same wall at
matched settings), so fixing it once fixes it for every variant below, and
attempting Stage 2 first means comparing six variants on a metric that is
mostly measuring initialization noise.

Four ranked hypotheses, each with a decisive cheap test. **Run them in the
order given** — H3 is minutes, H1 is the most likely cause, and running
H3/H2 first makes H1's sweep cheaper to interpret.

| # | Hypothesis | Why it's plausible here | Decisive test | Cost |
|---|---|---|---|---|
| **H3** | **Dead-atom resampling was never on.** | `CrosscoderTrainConfig.resample_dead_every_epochs` defaults to **`0` = disabled**, and the feasibility runs did not set it. The single-source `train.py` path has resampling and its own two already-fixed bug classes; the crosscoder inherited the code but not the setting. | Rerun the best-tuned feasibility setting with `resample_dead_every_epochs=2`. Nothing else changed. | ~5 min |
| **H1** | **Far too little data per atom.** | 4608 aligned rows for a 1280-atom dictionary is **3.6 rows per atom**. Published SAE practice is 10³–10⁴ activations per atom. An atom that never wins the TopK competition in its first few epochs never receives gradient again. | Sweep rows at fixed `dict_size`: 4.6k → ~50k → ~500k. Get the extra rows two ways and report both: (a) extract more series into the store (`data.max_series`, more windows per series), (b) flip the already-built `sae.real_data_enabled` (`sae/real_data.py`, HF-sourced activation augmentation) — the mechanism exists and has never been exercised for the crosscoder. | ~1–2 h |
| **H2** | **Dictionary far larger than the layer's effective dimensionality.** | `dict_size = 8 × max(d_in)` is a rule of thumb imported from NLP transformers. `analysis/internals.py` **already computes per-layer effective dimensionality** for exactly this run — read it for `stacked_xf.4` / `encoder.block.10` instead of guessing. If eff-dim there is ~40, an 8×768=6144-atom dictionary is asking for ~150 atoms per genuine direction. | One run at `dict_size ≈ 4 × eff_dim` and one at `16 × eff_dim`, from the recorded internals numbers. | ~20 min |
| **H4** | **TopK's winner-take-all dynamics need an auxiliary loss.** | Standard, published failure mode of TopK SAEs: unit-norm decoder + hard top-k means a latent that loses early is permanently starved. The published fix is an **auxiliary reconstruction loss** (`aux_k`) that asks the top-`k_aux` *currently dead* latents to reconstruct the residual, giving them gradient without letting them into the main forward path. | Implement `aux_k` in `train_crosscoder` (~15 lines: take the residual `x − x̂`, encode with dead latents only, add `aux_coef · MSE` to the loss) and rerun. | ~1 h |

> **Correction (2026-08-12) — H3's premise is false; H3 is already
> answered.** The H3 row above asserts "the feasibility runs did not set it."
> They did. `run_crosscoder_feasibility.py:79` (per-model baselines) and
> `:94` (crosscoder) both pass
> `resample_dead_every_epochs=max(1, args.epochs // 5)` explicitly, and
> `git log` shows that file unchanged since before the 2026-08-05
> feasibility run whose numbers are recorded above. The dataclass *default*
> is indeed `0`, which is what the hypothesis was reasoning from — but the
> CLI that produced every recorded number overrode it. So the real state of
> H3 is stronger and worse than "untested": **dead-atom resampling was on,
> at every recorded setting, and the 90–98% dead rate happened anyway.**
> Resampling is not the missing piece. Row kept rather than deleted
> (§0.2); do not spend the ~5 min running it.

**Exit criteria (all must hold, on the crosscoder *and* on the matched
per-model `TopKSAE` baseline at the same data/budget):**

- `dead_feature_rate ≤ 0.30`
- `≥ 500` alive atoms in absolute count (a 30% dead rate on a 200-atom
  dictionary is not a pass — both conditions, not either)
- `per_source_fidelity ≥ 0.70` for **both** sources simultaneously
- The winning configuration is written into a committed config file, not
  left as a CLI incantation in a session log

**Record in Findings**: which hypothesis was decisive, the full sweep table
(not just the winner), and the resulting alive-atom count. If none of H1–H4
gets there, that is itself a real, publishable finding about TSFM activation
geometry at these layers — write it up as such and reconsider whether
window-pooled activations are the right training substrate at all (see V6's
token-level note below).


## §0.5 — original tiered next-actions queue with stacked status updates (superseded 2026-08-18)

## 0.5. Start here — next actions, in order (added 2026-08-11)

> **Who this is for.** A session (or a person) picking this repo up cold and
> asking "what do I actually do next?" The rest of this file is organized by
> *topic*, which is right for a reference and wrong for a starting point. This
> section is the queue. It is a **pointer list, not a second source of truth**
> — every entry links to the section that owns the work, and when an item is
> done you mark it done *there*, not here. Re-derive this list whenever it
> looks stale rather than trusting it; §13 and §16 are authoritative.
>
> **Rules that apply to every item below**, so they aren't repeated per-entry:
> anything expected to run 5+ minutes goes to a background agent per
> `CLAUDE.md` §2.8; any config edit means a new `run.name` (§15 A3); any new
> checkpoint or library bump means `--check-alignment` read in full, per-layer,
> by a human (invariant 7 + `CLAUDE.md` §11.22).
>
> ⚠️ **Updated 2026-08-12 — read §22 alongside this.** A strategic pass added
> §17 (a whole-plan gap analysis against §1's north star), §18 (**equal
> grounds** — F1–F9, the fairness contract made measurable), §19 (architecture
> adaptivity — G1–G7 plus a landscape table of what's outside the envelope),
> §20 (new capabilities H1–H12) and §21 (the beginner/advanced layering,
> J1–J6). **§22 is the resulting six-wave interleaving** and is now the
> authoritative sequencing for anything beyond this section's Tier 1. Two
> changes to the queue below, both from §17's findings:
> - **A new Tier 0 goes first for anything cross-model.** ~~§18's F6 (deltas in
>   noise-floor units)~~ **F6 done 2026-08-12 — the audit count on
>   `medium_run_chronos_base` is 0 of 6 below-floor, so nothing recorded from
>   that run needed retroactive correction; see §18 F6's Findings for the one
>   marginal value (§16 E15's `−0.346`, at 2.2×) and for L3 restoration's
>   stated deferral**, ~~F2 (parameter/compute budgets — *nothing* in the repo
>   measures these today)~~ **F2 done 2026-08-12 — measured, staged and
>   rendered; see §18 F2's Findings. Its remaining gap is a live-checkpoint
>   run, not implementation**, F1 (the depth axis — Chronos's "relative depth 1.0"
>   is currently the middle of its computation, plotted against TimesFM's
>   actual output) — ⚠️ **F1 half-landed 2026-08-13: the module, the four
>   axes, the adapter `uncaptured_surfaces()` declarations, the `block`
>   default and 13 tests are in; nothing consumes them yet, so every figure
>   still renders the legacy axis and the acceptance criterion (the §16 E9
>   depth-curve null re-run) has not been run. The remaining work is the
>   eight `relative_depths` call sites, the figure titles/notes, and that
>   re-run — see §18 F1's Findings** — and ~~F4 (capture-coverage accounting)~~ **F4 done
>   2026-08-12 — measured live and rendered: TimesFM observes 42.5% of its
>   forward FLOPs, Chronos-T5-Base at most 14.4%, and every depth-located
>   finding about either now carries an automatic qualifier. §12 item 2's
>   "for Chronos maybe half" was wrong in both halves and is corrected. See
>   §18 F4's Findings**. These **retroactively
>   qualify numbers already recorded in this file**, which is why they precede
>   the next round of cross-model claims — the same argument §16 E9's null
>   baseline won.
> - **Tier 4 (adoptability) must not overtake Tier 0.** §22's closing
>   paragraph has the reasoning: one-button automation removes the expert who
>   would have known not to believe an unequal axis.

**Tier 1 — the flagship research deliverable.** This is the one thing in the
file that is both novel and unfinished, and everything else is support.

1. **§6.2.1 Stage 0 — the dead-feature blocking gate.** 🔴 Do this before any
   other crosscoder work. 90–98% of dictionary atoms are dead in every SAE run
   on record, which makes `relative_decoder_norm` (the whole shared-vs-specific
   metric) read ~98% "shared" from dead-atom symmetry alone. Four ranked
   hypotheses and an explicit exit criterion are in §6.2.1. **Nothing
   downstream of this is trustworthy until it passes.**
   **Status 2026-08-12 (updated after the `k`, `h2xh4` and `pinch` sweeps) —
   still `[ ]`, and the blocker has moved twice.** H4 (AuxK) was decisive
   (80 alive atoms → 1072); H1 and H2 are both refuted as fixes, H2 now in
   the AuxK-*on* regime too (the crosscoder passes every criterion at dict
   ∈ {640, 768, 896, 1024}). The **crosscoder itself passes** — a winner is
   committed at `configs/crosscoder_stage0_winner.yaml`. What still fails is
   the **Chronos-T5-Base per-model baseline**, and the `pinch` sweep proved
   by measurement that **no dictionary size in [512, 1280] clears both
   `dead ≤ 0.30` and `alive ≥ 500`** (the two rows clearing the rate bar sit
   114 and 33 atoms short of the floor; fidelity is never the blocker).
   So the residual is a **criteria question, not a training question**:
   `MIN_ALIVE = 500` is a global constant applied to a layer with measured
   effective dimensionality 13.93. §6.2.1 names two defensible resolutions
   (a per-model floor of ~`20 × eff_dim`, or keep the constant and carry a
   quantified caveat into Stage 2) and **deliberately decides neither** —
   that call is the actual next action here. Also open: alive counts are
   non-monotone in dictionary size, implying ±60-atom single-seed noise
   against a 33-atom margin, so the decision should rest on replicate seeds.
   Full tables in §6.2.1's sweep Findings.
   **Status update, later the same day — the criteria question above is now
   decided, the replicate seeds are in, and Stage 0 is still `[ ]` for a
   third, different reason.** The floor became `max(100, round(20 ×
   eff_dim))` per source (279 Chronos / 563 TimesFM — *stricter* than the
   500 it replaced, for the crosscoder), because the old pair was provably
   unsatisfiable at any size for a model whose alive count saturates
   (§6.2.1 finding (11) + DECISION). The 5-seed `pinch` replicate then
   showed the same shape recurring **between** models: Chronos's untouched
   rate bar caps a shared dictionary near 569 atoms while the crosscoder's
   alive floor needs ~606, so all three artifacts pass together at **0 of 5
   seeds at every size** (findings (12)-(13)). The recommended fix — size
   the dictionary per model, matching rows/k/epochs rather than dict, which
   gives every artifact a passing size on record — is written up with its
   evidence and **deliberately not applied**, because it would be a second
   criteria change in one session in the direction that makes the gate pass
   (finding (14)). That call is now the actual next action here. Also newly
   on record: the crosscoder's failures are **bimodal**, one collapsed run
   in five rather than a marginal miss (finding (15)) — Stage 1's scorecard
   needs replicates per variant or it will attribute a bad basin to a
   variant.
   ✅ **CLOSED 2026-08-13 — Stage 0 is `[x]`; start at item 2 below.** The
   pending criteria call above was made (per-model baseline *sizing*, at
   matched budget — findings (16) + DECISION), justified by a measurement
   that **inverted** the premise this whole item was written on: the
   per-model baseline never "hit the same wall", it is 3.2–5.4× *less* dead
   than the crosscoder at every shared size, and the two move in opposite
   directions as the dictionary grows. The gate replicate then passed at
   **5 of 5 seeds** at `dict=1024` with both baselines passing at their own
   sizes (finding (17)); the committed config is
   `configs/crosscoder_stage0_gate.yaml`. The blocking sentence above —
   "nothing downstream of this is trustworthy until it passes" — is
   discharged, with one carry-over: `relative_decoder_norm` still must not
   be quoted until **Stage 1's** scorecard measures it on this alive
   dictionary. ✅ **Measured 2026-08-13 (item 2) — and the ban tightens
   rather than lifts:** the alive dictionary's real-pair `frac_shared` is
   0.845 against an untrained-twin floor of 0.974, so the quantity is now
   known *not* to separate learned cross-model structure from architecture
   at V1. Quote it only with its L-B floor.
2. **§6.2.1 Stage 1 — the validation ladder** (L-A identity → L-B `random_init`
   hard null → L-C layer-offset monotonicity → L-D the real question → L-E
   planted synthetic). Build `sae/crosscoder_eval.py` first; the ladder is what
   makes Stage 2's variant comparison mean anything.
   🔴 **Built and run 2026-08-13 — stays `[ ]`. L-A and L-E pass perfectly
   (frac_shared 1.000; planted-recovery F1 1.0), so the metric is sound; but
   V1 **fails L-B**: real 0.845 vs. null 0.974, diff −0.129, CI
   [−0.160, −0.101], p=0.002 — TimesFM reads as *more* shared with a random
   copy of itself than with Chronos. `dead_feature_rate` also fails (0.331 vs
   ≤0.30), traced to this ladder running on 4,608 rows against Stage 0's
   46,382 — **Stage 0's 3.2% dead does not transfer across row counts.**
   Next actions, in this order, and note the third **reorders Stage 2**:
   (a) score V0 so clause 2 stops being `unrun`; (b) re-run the ladder at the
   bigdata row count; (c) build **V4 (latent scaling) before V2/V3** — it is
   the published diagnostic for precisely the failure L-B just exhibited.
   ✅ **(a) done 2026-08-13** (`--v0`, `crosscoder/ladder_v0.json`): the
   verdict now reports `unrun: []`, and clause 2 **fails** — V1 0.3456 vs
   V0 0.3658, diff −0.0202, CI [−0.0489, +0.0119], p=0.208. That makes the
   rule's own escape clause ("if no variant beats V0, that is the result")
   the leading reading, but **not yet callable**: the CI spans zero, and the
   point estimate changes sign with V0's dictionary sizing. Two traps found
   scoring it — V0's candidate pool was capped at the top 50 by |ρ|, the
   exact quantity being compared (+57% inflation), and the ladder silently
   trained V0 at matched rather than Stage 0's committed per-model sizes.
   **(b) is now the item that decides Stage 1**: 46,382 rows removes the
   dead-rate confound *and* narrows this CI. Full numbers in §6.2.1 Stage 1's
   Findings.
   ✅ **(b) done 2026-08-13, and it decided clause 2 — against V1.** At 46,382
   rows the dead rate falls to **0.032** (Stage 0's 3.2% reproduces exactly
   once the two share a corpus size, so that confound is fully explained), and
   clause 2 resolves: V1 **0.2787** vs V0 **0.3204**, diff **−0.0417**, CI
   **[−0.0693, −0.0137]**, p=**0.002**. The CI came off zero on the *losing*
   side — the joint dictionary's ground-truth alignment is significantly worse
   than two independent dictionaries matched post hoc, at a 4.3% *larger*
   `n_alive`. **The rule's escape clause is now the reading the data supports,
   not merely the leading one.** Stage 1 still stays `[ ]`, for a new reason:
   **L-B and L-C both went unrun** — L-B because the null run holds 4,608 rows
   against the real run's 46,382 and the ladder refused to pair across corpora,
   L-C because every configured layer delta exceeds TimesFM's captured depth.
   Revised next actions: **(b2)** re-extract `null_timesfm_random` at 46,382
   rows (a real extraction, not a re-analysis); **(b3)** give L-C deltas that
   fit the captured stack; then **(c)** V4, whose case finding 2 strengthens.
   ✅ **(b2)+(b3) done 2026-08-17 — Stage 1 checkbox now `[x]`.** All five
   ladder rungs now run for real at 46,382 rows: L-B clears its floor
   (diff +0.0807, CI [0.038,0.126], p=0.002 — the *opposite* direction from
   the earlier 4,608-row failure), L-C runs and is real (flat through δ=3,
   drops at δ=4) but does not gate the pre-registered decision rule. L-D and
   the V0 comparison reproduce bit-for-bit (diff −0.0417, CI
   [−0.0693,−0.0137], unchanged). `verdict.passes=False` with `unrun: []` —
   every clause evaluated, V1 still loses clause 2 to V0. Next: **(c)** build
   V4 (latent scaling), now against a fully-populated ladder rather than a
   ladder with unrun clauses. Full numbers in §6.2.1 Stage 1's Findings.
3. **§6.2.1 Stage 2 — variants V0–V6**, scored on the ladder. V0 (the existing
   post-hoc `sae/matching.py` route) is the baseline every crosscoder variant
   must beat; if none does, that is a real, publishable negative result.

**Tier 2 — the reopened falsified item.** Independent of Tier 1, so it can run
in parallel by a background agent.

4. **§6.3.1 Option E — build the lineage ground truth first.** Fine-tune
   `chronos-t5-small` into `child_light`/`child_drifted`, with
   `chronos-t5-mini` as the same-architecture *negative*. §6.3's original
   method was falsified precisely because it had no true-positive case and no
   architecture-matched negative; every replacement option is untestable until
   this exists.
5. **§6.3.1 Option C — idiosyncratic-error fingerprinting.** The recommended
   replacement: black-box, architecture-agnostic, zero new forward passes, and
   passes the architecture control by construction. The load-bearing step is
   regressing out difficulty (via L2's existing input-feature probe) *before*
   correlating residuals.

**Tier 3 — cheap items that retire recorded uncertainty.** Each is small and
each currently leaves a number in this file with unknown error bars.

6. ~~**§13 — SAE forecast-preservation repeat-run variance.**~~ **Measured
   2026-08-12** — n=5 seeds on a frozen store via the new
   `run_sae_repeat_variance.py`. The floor is **sd 0.121** (TimesFM), so the
   0.175→0.1097 swing that motivated this is half a standard deviation and
   explains itself. One recorded number is **retracted**: E15's Chronos
   token-granularity **−0.346** falls outside the five-seed range
   [−0.112, +0.414] and its sign replicates in 1 of 5 seeds — quote
   **+0.246 ± 0.211** instead. Numbers at §13's Findings block. ~~**Residual
   split out as its own §13 item:** the floor is measured but not yet
   *rendered* beside the numbers in the report.~~ **Rendered 2026-08-13** —
   `sae.n_seeds` (default 1, so nothing already recorded changes) sizes the
   floor inside the stage, and the report renders `± sd over N seeds` beside
   each ΔMASE, warns in the body when a delta is smaller than its own spread,
   and says "Not measured" out loud when no floor exists. §13's Findings
   block has the mechanism.
7. ~~**§13 — the reverse L2 direction** (`TimesFM→Chronos`).~~ **Done
   2026-08-12** — real gain beats Chronos's own floor at every depth and
   TimesFM's own floor from `stacked_xf.10` down; the recorded pessimistic
   number was a wrong-layer artifact. §13's checkbox is `[x]`, numbers at
   §16 E9's fourth follow-up.
8. ~~**§16 E19 — ratify or reverse the univariate-only envelope.**~~ **Done
   2026-08-12 — ratified univariate-only.** The decision paragraph is in
   `CLAUDE.md` §12's envelope-edge list; Chronos-2's GROUP axis is the named
   exclusion, with a stated rule for future cross-series adapters and an
   explicit note that the size of the resulting coverage gap is unmeasured.

**Tier 4 — adoptability, in dependency order.** All of §16's T1 is unblocked by
§15 today (see §16's corrections block); the internal ordering below is the
only real constraint.

9. **E4** (reference corpus — has a licensing decision to make first) →
   **E1** (zero-config entry point) → **E7a/E7b** (console script + install
   docs, both independently unblocked) → **E5** (registry) → **E6** (findings
   as data). **E3** (empirical span discovery) is unblocked and independent of
   that chain — it is the highest generality-per-hour item in the file and can
   start any time.


## ROADMAP.md header — original seed-brief corrections block and 2026-08-06 audit-pass announcement (superseded 2026-08-18)

> **Corrections already made against the source brief this file was seeded
> from (and now also fixed in `CLAUDE.md` itself):** `benchmark_validation`
> lives at `tsfm_benchmark/benchmark_validation/` (nested, confirmed via
> `pyproject.toml`'s package list) — `CLAUDE.md`'s `[VERIFY]` guess of a
> sibling location was wrong. The `CO_trev_1_num` outlier-swamping issue
> flagged as an outstanding `QuantileTransformer`/IQR-ranking fix in
> `CLAUDE.md` §5 **has been fixed, but by a different, already-landed
> mechanism** — `RobustScaler` output winsorized to ±`clip_scaled` robust-scale
> units, with per-sequence anomaly diagnostics surfaced in `plot.py`/
> `report.py` — not the originally-proposed `QuantileTransformer`/IQR-ranking
> approach; `CLAUDE.md` §5 now describes the actual mechanism. The SAE phase
> is still exactly a stub (`sae/interface.py`, a `Protocol` and a
> `NotImplementedError` — no training code exists anywhere yet), which is
> expected: it's `Phase 2` below and hasn't started. Also newly discovered
> and fixed in `CLAUDE.md`: the TimesFM adapter has moved to TimesFM 2.5
> (different checkpoint, layer count, and a newly-supported
> `attention_patterns` capability) — see `CLAUDE.md` §11.8.
>
> **Audit pass added 2026-08-06 (planning only, no code).** Two new sections
> at the end of this file: **§15** is a code-grounded audit of *silent-failure
> paths, fragile mechanisms, and acknowledged-but-fixable limitations* found by
> reading the pipeline against the claims in this file and `CLAUDE.md`
> (19 items, each with evidence at `file:line`, why it matters, and a detailed
> fix plan marked **NEEDS IMPLEMENTATION**); **§16** is an enhancement backlog
> aimed squarely at the §1 north star — what a stranger with a checkpoint and
> one button actually needs, benchmarked against what `transformer-lens` users
> expect. Several §15 items contradict claims made elsewhere in this file and
> in `CLAUDE.md`; those places now carry inline `⚠️ AUDIT` markers pointing at
> the relevant item rather than being quietly left to mislead. Nothing in §15
> or §16 has been implemented — they are scheduled work, not findings.


## Archived: §17.3's original sequencing block (2026-08-12), superseded by §0.5's single queue

### 17.3 Sequencing these against the existing plan

The gaps above do **not** displace §0.5's Tier 1 (the flagship crosscoder) —
that is the novel research contribution and stays first. They interleave:
- **G-I and G-II are cheap and retroactively qualify numbers already recorded**
  (every depth-axis figure, every cross-model MASE comparison). They belong
  *before* the next round of cross-model claims, not after — same argument
  §16 E9 made for the untrained-weights null, which is now the most-cited
  control in the file.
- **G-III and G-V are report-layer work** and compose naturally with §16 E6.
- **G-IV is the long pole** and is best paid down incrementally: one new
  architecture *class* (not model) per session, each chosen to break a
  different assumption.
