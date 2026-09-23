"""Single source of truth for "what does each pipeline stage tell you?"

`ROADMAP.md` sec 21 J2: every stage answers a specific question, by a
specific mechanism, and — the line most tooling omits — cannot tell you
something specific either. Both the HTML report (`report/report.py`) and
the README's generated stage table (`render_stage_docs.py`) render directly
from `STAGE_DOCS` below, so the two surfaces cannot drift apart the way two
independently hand-maintained descriptions of the same stage always
eventually do.

Content here is consolidated, not re-authored, from two places that already
went through review: `CLAUDE.md` sec 6.1's stage table (question/method/
inherited limitation) and sec 6.5's per-stage prose, and this module's own
sibling `report/report.py`'s existing `_note()` blocks (the "Purpose" /
"Reading values" / "Limitations" text already shown under each figure).
Every stage `pipeline.stage_names()` can produce has an entry here — a
missing one is a loud `KeyError` from `get()`, not a blank box in the
report (`CLAUDE.md` sec 2.5).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageDoc:
    """The four fixed lines `ROADMAP.md` sec 21 J2 asks for, per stage.

    `cannot_tell` is the one that matters: every stage's real epistemic
    limitation, not a softened or omitted version of it (`CLAUDE.md` sec 2.6).
    """
    question: str
    how: str
    good_bad: str
    cannot_tell: str


# Keys are `pipeline.stage_names()`'s own names. A few report-section
# `config_attr` names differ from the pipeline stage name for historical
# reasons (the clustering stage is named "cluster" in `pipeline.py` but its
# config section and report builder both say "clustering") — resolved by
# `get()` below rather than by duplicating an entry.
_ALIASES = {"clustering": "cluster", "l4": "cluster"}


STAGE_DOCS: dict = {
    "corpus": StageDoc(
        question=("Can the benchmark corpus itself be trusted -- is it free of copied "
                   "reference series, free of near-duplicates, and diverse enough that a "
                   "model can't do well here just by memorizing one shape?"),
        how=("Reads the sealed corpus's own manifest (ROADMAP.md sec 34 item B1's audit "
             "block, when present) and, optionally, an existing `benchmark_validation` "
             "report, and renders both as a fixed trust ladder of named claims -- never "
             "recomputes a leakage or diversity check itself, and never imports "
             "`benchmark_validation`."),
        good_bad=("Good: every trust-ladder row reads `measured` with a real reference "
                   "and a stated verdict. Bad: a row reads `not_checked` -- the corpus "
                   "was built with `--references none`, so every candidate passed the "
                   "leakage gate trivially, which is NOT the same as a corpus that was "
                   "checked and found clean -- or `not_recorded`, meaning no audit ever "
                   "ran at all."),
        cannot_tell=("Two rows are always unresolvable and are rendered that way rather "
                      "than omitted: whether the private split's distribution genuinely "
                      "matches the dev split's (no mechanism in this repo compares them), "
                      "and whether any model being compared was trained on data shaped "
                      "like this corpus (no audit anywhere can inspect a checkpoint's own "
                      "training data). A `measured` row is evidence the corpus is sound "
                      "by the checks that exist -- it is not proof no other check would "
                      "find a problem.")),
    "extract": StageDoc(
        question=("Has each model's internal computation actually been captured, and "
                   "lined up correctly with real time, so every later stage has something "
                   "trustworthy to work from?"),
        how=("Runs each model once over the benchmark series, hooks its internal layers "
             "to save their activations, and pools each model's own tokens onto a shared "
             "time-window axis so architectures that read one timestep per token and "
             "architectures that read dozens per token become directly comparable."),
        good_bad=("Good: the alignment check's diagonal-hit fraction sits near 1.0 at "
                   "every captured layer, and no non-finite (inf/NaN) activations are "
                   "recorded. Bad: the alignment gate fails outright — the declared "
                   "mapping between token position and true time is wrong for this "
                   "checkpoint or library version — or a layer's stored activations "
                   "contain non-finite values that would silently poison every "
                   "downstream computation that reads it."),
        cannot_tell=("This stage doesn't evaluate or compare anything — it only says "
                      "whether each model's computation was captured and time-"
                      "aligned correctly. A passing alignment check means the token-to-"
                      "time mapping is trustworthy; it says nothing about whether the "
                      "captured activations are interesting, whether the models are "
                      "similar, or whether any later stage will find anything at all.")),
    "budget": StageDoc(
        question="What does each model cost to run, and is its forecast quality bought with more compute?",
        how=("Runs a handful of extra forward (and, optionally, full-forecast) passes "
             "through each model while measuring parameter counts, FLOPs, wall-clock "
             "latency, and peak GPU memory, then re-plots each model's L0 accuracy "
             "against those costs instead of looking at accuracy alone."),
        good_bad=("Good: FLOPs are measured directly (not falling back to an "
                   "unmeasured or 'suspicious_low' reading) and the model sits down-"
                   "and-to-the-left on the compute-normalized chart — lower error for "
                   "less compute. Bad: a `flops_sanity: suspicious_low` verdict (the "
                   "measurement can't see a fused or custom kernel and is undercounting "
                   "compute), or the L0 stage didn't run, so cost can't be normalized "
                   "by quality at all."),
        cannot_tell=("Cost is measured only at this run's own context length, batch, "
                      "and horizon — it does not generalize to a different sequence "
                      "length, and FLOPs are not the same thing as latency (a model "
                      "with fewer FLOPs can still be slower on real hardware). It also "
                      "only tells you the price of a forward pass, never whether that "
                      "price bought anything mechanistically interesting — that's what "
                      "every later stage is for.")),
    "frontend": StageDoc(
        question=("What does each model do to its input BEFORE any layer runs -- how "
                   "coarsely does it quantize the series, does scaling the input scale "
                   "the forecast back out cleanly, how does accuracy degrade as recent "
                   "context goes missing, and what happens when a context value is "
                   "NaN?"),
        how=("Reads each re-quantizing tokenizer's own bin geometry to score how coarse "
             "one quantization step is relative to a series' own amplitude; calls "
             "`predict()` on the input scaled up/down by large factors and compares the "
             "rescaled-back forecast to the original; calls `predict()` with the most "
             "recent context progressively withheld (a staleness scenario, not a front-"
             "trim) to trace a degradation curve; and injects NaN at the front, middle, "
             "and back of the context to see whether `predict()` errors, silently "
             "produces a non-finite forecast, or genuinely handles it."),
        good_bad=("Good: a re-quantizing tokenizer's bin width is a small fraction of "
                   "the series' own amplitude with little saturating clipping, near-zero "
                   "scale-equivariance residual, graceful (not cliff-shaped) accuracy "
                   "degradation as recent context is withheld, and a NaN verdict of "
                   "'handled' or an explicit, clean error rather than a silently "
                   "corrupted forecast. Bad: heavy clipping or a large quantization "
                   "step relative to signal amplitude, a scale-equivariance residual "
                   "that grows with the scale factor, a sharp cliff at a specific "
                   "context length, or a 'propagates' NaN verdict -- a non-finite input "
                   "silently becomes a non-finite forecast with no error to flag it."),
        cannot_tell=("Every diagnostic here is about the FRONT DOOR only -- what happens "
                      "before or around a forward pass, never what happens inside one "
                      "(that's every other stage's job). Quantization resolution is "
                      "meaningful only for a re-quantizing tokenizer and is reported as "
                      "'not applicable', never a fabricated zero, for a continuous-"
                      "embedding architecture. NaN handling is checked at only a "
                      "handful of hand-placed positions and one missing-fraction, not "
                      "exhaustively; and none of these four probes says anything about "
                      "forecast quality on ordinary, well-formed input -- L0 is what "
                      "answers that.")),
    "layer_screen": StageDoc(
        question=("Which of this model's own layers are worth spending the expensive "
                   "stages (Lens, L1, L2, L3, attention, SAE) on?"),
        how=("Scores every layer a model actually captured with a cheap, architecture-"
             "agnostic proxy for 'interesting residual-stream activity' — by default, "
             "how much a layer's own trajectory through activation space bends and "
             "does work — and flags the highest scorers within a compute budget."),
        good_bad=("Good: the selected layer(s) coincide with, or precede, the layers "
                   "where L1/L2/L3/Lens later find the strongest real signal — the "
                   "cheap proxy pointed at something genuine. Bad: a selection sitting "
                   "at the very first or very last layer with no real signal there, or "
                   "a run where the report has to state the method fell back to a "
                   "uniform-stride null because its own selection failed."),
        cannot_tell=("This is a cheap proxy for interestingness, not interestingness "
                      "itself — it won a single-corpus, single-checkpoint-pair bake-off, "
                      "not a validated rule across architectures. It also only screens "
                      "the layers `capture_layer_stride` actually captured, so at a "
                      "stride greater than 1 it cannot see, and cannot recommend, half "
                      "the model's blocks; it answers 'which captured layer looks most "
                      "active,' not 'which is the best layer in the whole model.'")),
    "l0": StageDoc(
        question="Which model actually forecasts more accurately, and on what kinds of data?",
        how=("Runs every model's native forecasting procedure on every benchmark "
             "series, scores each series' error against a naive same-scale baseline, "
             "and statistically compares the paired per-series differences within each "
             "data family, correcting for testing many families -- and, when more than "
             "two models are configured, many model pairs -- at once."),
        good_bad=("Good: a family where the paired difference's confidence interval "
                   "clearly excludes zero after correction — a real, replicable "
                   "accuracy edge on that kind of data. Bad: every family's interval "
                   "spans zero (no reliable difference survives correction), or a gap "
                   "that looks real in the bar chart but isn't backed by a significant "
                   "test."),
        cannot_tell=("This is purely behavioral: it can tell you which model wins and "
                      "where, but nothing about why, or what mechanism inside either "
                      "model produces the gap — that's what every stage after this one "
                      "exists to investigate. It also measures point-forecast accuracy "
                      "only; a model can score well here and still have badly "
                      "miscalibrated uncertainty (a separate calibration diagnostic "
                      "covers that, but it is not this headline number). And it is the "
                      "ONLY stage that compares more than two models: everything after "
                      "it compares the designated pair only, so with three models the "
                      "extra pairs are unexamined there rather than weakly "
                      "evidenced.")),
    "internals": StageDoc(
        question="What does each model's own internal representation look like at each depth, on its own terms?",
        how=("For each layer of each model separately, measures how many effective "
             "dimensions its activations actually use, how close its representation "
             "still is to simple statistics of the raw input, and how well a simple "
             "classifier can read the data's family straight out of it."),
        good_bad=("Good: a clear expand-then-compress effective-dimensionality curve, "
                   "a family-decodability curve that rises well above chance at some "
                   "depth, and a representation that visibly moves away from raw-input "
                   "statistics with depth. Bad: a flat effective-dimensionality curve "
                   "or a family-probe accuracy that never clears chance — either means "
                   "this per-model view has nothing to say about that model's depth "
                   "behavior."),
        cannot_tell=("Everything here is per-model and descriptive — it never directly "
                      "compares models against each other (L1 exists for "
                      "that). 'Decodable' does not mean 'used by the forecast': a "
                      "layer can carry perfect family information the model itself "
                      "never reads out. It's also limited to the window-pooled "
                      "representation, so structure that lives only within a window "
                      "and not across windows is invisible to it.")),
    "lens": StageDoc(
        question="At what depth does each model's forecast actually take its final shape?",
        how=("Feeds each layer's own activations into the model's normal output "
             "pathway, as if that layer were the last one, to see how good a forecast "
             "it already produces; separately, fits a small linear readout per layer "
             "to check whether the forecast is already recoverable there even if the "
             "model's own head can't decode it yet."),
        good_bad=("Good: a curve that reaches close to final accuracy well before the "
                   "last layer ('crystallizes early'), consistent with a similarly "
                   "early rise in the tuned-lens readout. Bad: a flat, uninformative "
                   "curve where the tuned lens also shows nothing until the very last "
                   "layer — the forecast genuinely needs the model's full depth — or "
                   "the two disagree sharply, itself informative about miscalibrated "
                   "early layers."),
        cannot_tell=("The direct (skip-lens) readout can be misleadingly pessimistic "
                      "about early layers for reasons unrelated to information content "
                      "— the model's final block was never trained to decode them — "
                      "which is exactly why the tuned-lens readout exists alongside it. "
                      "Even together, both only detect *linearly* recoverable "
                      "structure; a forecast could be present in a layer in a form no "
                      "linear readout can see.")),
    "l1": StageDoc(
        question="Do the models represent this data similarly at all?",
        how=("Compares every pair of layers, one from each model, by how similarly "
             "they organize the same set of series geometrically — a comparison that "
             "ignores each layer's own arbitrary rotation and scale — and checks the "
             "result against a shuffled-series control."),
        good_bad=("Good: a peak similarity that sits clearly above the shuffled-"
                   "series null, especially one that follows a sensible pattern by "
                   "depth (early layers matching early layers). Bad: a similarity "
                   "score that's high everywhere and barely rises above the null — a "
                   "sign the number mostly reflects that the models see the same "
                   "input, not that they've learned anything in common."),
        cannot_tell=("This is correlational: a high score never establishes that the "
                      "two models compute the same thing, only that their "
                      "representations are geometrically similar — which two under-"
                      "trained or too-simple networks can also produce for reasons "
                      "that have nothing to do with shared learning. It says nothing "
                      "about mechanism; L2 and L3 exist specifically because this "
                      "stage can't answer that.")),
    "l2": StageDoc(
        question=("Can one model's layer actually be translated into the other's, "
                   "beyond what's explained by both simply seeing the same input?"),
        how=("Fits a simple linear map from one model's layer activations to the "
             "other's, compares how well it predicts against a hand-built baseline "
             "that only ever sees the raw input (never the other model), and reports "
             "the gap between the two as the real evidence."),
        good_bad=("Good: the best pair's gain over the input baseline has a "
                   "confidence interval that clearly sits above zero — real evidence "
                   "the source layer predicts the target layer better than raw input "
                   "alone could. Bad: that interval includes zero — the apparent "
                   "shared structure could just be an artifact of the models seeing "
                   "the same series."),
        cannot_tell=("Only the gain over the input baseline counts as evidence here — "
                      "raw predictive accuracy alone is untrustworthy, since both "
                      "models process the same input and would score well on that "
                      "basis alone. This also only detects *linear* translatability "
                      "(a genuine nonlinear correspondence between two layers can "
                      "score zero gain), and it is still not a causal claim: a good "
                      "stitch means the two layers' information is compatible, not "
                      "that either model's forecast actually depends on it.")),
    "l3": StageDoc(
        question=("Where, causally, does each model actually carry a specific "
                   "structural property of the data — and does damaging it there hurt "
                   "the forecast?"),
        how=("Corrupts one structural property of the input at a time (trend, "
             "seasonality, added noise, and so on), measures how much each layer's "
             "activations shift because of it, then splices the clean, uncorrupted "
             "activations back into an otherwise-corrupted forward pass, one layer at "
             "a time, to see how much of the clean forecast that single splice "
             "restores."),
        good_bad=("Good: a restoration curve that rises toward 1.0 at some depth for "
                   "a given corruption — that layer causally carries the fix for that "
                   "property — and two models whose sensitivity fingerprints peak at "
                   "matching relative depths for the same corruption. Bad: a "
                   "restoration curve that never clears a small fraction anywhere, or "
                   "a corruption whose measured sensitivity is small only because it "
                   "touches a tiny fraction of the input's timesteps in the first "
                   "place, not because the model is robust to it."),
        cannot_tell=("The activation-shift fingerprints are a magnitude-of-change "
                      "measure by themselves, not a causal one — a layer can shift a "
                      "lot without that shift ever affecting the final forecast, which "
                      "is exactly why the patching curve exists alongside it. Even the "
                      "patching curve is causal only *within* one model: this stage "
                      "never transplants activations between models, so any "
                      "cross-model comparison here is two separately-measured within-"
                      "model curves side by side, never a joint causal test.")),
    "attention": StageDoc(
        question="Which attention heads look where in time, and which of them (or which MLP blocks) actually matter to the forecast?",
        how=("Averages each head's attention weight as a function of how far back in "
             "time it looks, flags heads whose attention concentrates near seasonal-"
             "period multiples, and separately re-runs the forecast with individual "
             "heads or MLP blocks zeroed out to see how much accuracy degrades."),
        good_bad=("Good: a head with a clean, repeating attention stripe at a real "
                   "seasonal lag (a periodicity head) whose removal via ablation "
                   "measurably hurts accuracy on exactly that kind of data — pattern "
                   "and causal importance agreeing. Bad: no head or MLP block's "
                   "removal moves accuracy at all, or the architecture reports "
                   "'unsupported' because it exposes no attention weights to inspect."),
        cannot_tell=("The lag-profile and periodicity-head numbers are unconditional "
                      "averages over sampled series, so a head that's periodic only on "
                      "some data families and flat on others can look unremarkable in "
                      "the averaged view even though it's doing real, family-specific "
                      "work. Pattern support (whether attention weights or ablation "
                      "are even available) varies by architecture, so a missing or "
                      "'unsupported' result here is a capability gap, not evidence the "
                      "model lacks that structure.")),
    "cluster": StageDoc(
        question="How does each model organize the whole benchmark on its own terms, and do the models group the data the same way?",
        how=("Projects each model's activations, at its own side of the strongest "
             "cross-model layer pair, down to a small number of dimensions, clusters "
             "them, labels each cluster by what kind of data dominates it, and "
             "compares how much each pair's groupings agree using a chance-"
             "corrected overlap score."),
        good_bad=("Good: clean, well-separated clusters that line up with real, "
                   "interpretable properties of the data, and a partition-agreement "
                   "score meaningfully above chance between a pair of models. Bad: a "
                   "single smeared blob with no real separation, or cluster labels "
                   "that don't correspond to anything a human would recognize as a "
                   "coherent group."),
        cannot_tell=("The 2D map is a visualization only — apparent distances or gaps "
                      "between clusters on the plot are not quantitative and should "
                      "never be read as a diversity or separation measure. Cluster "
                      "labels are approximate (majority family plus salient "
                      "statistics), not ground truth, and everything here is "
                      "descriptive: it says how each model happens to partition the "
                      "data, not why, and not whether that partition is causally "
                      "meaningful to either model's forecast.")),
    "seasonality_circuit": StageDoc(
        question=("What is the smallest set of a model's own attention heads that is "
                   "causally sufficient and necessary to carry its seasonal forecasting, "
                   "and does each head's effect act independently of the others in that "
                   "set, or do they interact?"),
        how=("Scores every candidate head's own effect on a seasonal-power metric under "
             "the `deseasonalize` corruption, greedily grows a minimal set by always "
             "adding the head that most raises restoration, checks that set against a "
             "random-same-size-set null (the mandatory acceptance bar), then — only for "
             "a set that clears the null — path-patches inside it: noising one head "
             "alone, then again with every other set member frozen clean, then again "
             "re-injecting just the isolated delta into each other member one at a time, "
             "to see whether the total effect is the sum of its parts."),
        good_bad=("Good: a small selected set whose restoration clearly beats the "
                   "random-set null with a real confidence-interval margin, and (for a "
                   "multi-head set) a path-decomposition whose parts sum close to the "
                   "whole. Bad: a selected set that never separates from the null (the "
                   "circuit isn't well captured by attention heads alone, or this "
                   "particular greedy search missed it), or a large conservation gap, "
                   "which does not necessarily mean the method is broken — it can mean "
                   "the heads found genuinely interact nonlinearly rather than "
                   "contributing independent, additive paths."),
        cannot_tell=("This is within-model causal evidence only (invariant 5) — nothing "
                      "here compares one model's circuit to another's, and a set that is "
                      "*sufficient* is not proven *unique*: a greedy search finds a small "
                      "set, never provably the smallest, and a different search order "
                      "could find a different set of the same size. The scope is "
                      "attention heads only — an MLP-mediated circuit component would be "
                      "invisible to this analysis entirely, and it is a single benchmark "
                      "corpus's own seasonal families, not a claim about seasonality in "
                      "general.")),
    "sae": StageDoc(
        question="Can each model's layer be decomposed into a small number of individually interpretable features, do those features actually matter to the forecast, and do the two models' features play the same role?",
        how=("Trains a sparse dictionary that reconstructs a layer's activations from "
             "only a handful of active 'feature' directions at a time, then checks "
             "reconstruction quality, how many learned features ever fire at all, "
             "whether swapping in the reconstruction changes the forecast, and "
             "whether any feature's activation tracks a known ground-truth property "
             "of the data (like a trend order or a seasonal period). A causal channel "
             "battery then patches each alive feature's direction into a clean forward "
             "pass and measures a real, per-channel response (not just correlation with "
             "a label), scored against a random-direction null. Features are clustered "
             "into named 'roles' by their shared response and structural signature, and "
             "each model's roles are matched against the other model's by cosine "
             "similarity of their null-normalized response fingerprints, checked against "
             "an untrained-twin floor."),
        good_bad=("Good: high reconstruction fidelity, a low dead-feature rate, "
                   "forecast-preservation deltas that stay within the SAE's own "
                   "seed-to-seed noise floor, and features whose activation clearly "
                   "tracks a real ground-truth field well above what random label "
                   "shuffling would produce by chance. For the causal layer: a channel "
                   "response that clears the random-direction null, and a cross-model "
                   "role match rate that clears its own untrained-twin floor (not just "
                   "zero). Bad: the large majority of features never firing at all (a "
                   "common failure mode), a forecast-preservation delta far outside the "
                   "noise floor, a ground-truth alignment score that isn't meaningfully "
                   "above its own permutation-null control, or a role match rate that "
                   "looks large but sits below the untrained-twin floor — architecture "
                   "match alone can produce that, not shared learned structure."),
        cannot_tell=("A feature firing on a particular kind of series is illustrative "
                      "correlation unless the causal channel battery has confirmed a "
                      "real patched response for that feature — check the channel result, "
                      "not just the ground-truth alignment score, before reading any "
                      "feature as meaningful. A cross-model role match is a geometric "
                      "correspondence in response space, not evidence the two models use "
                      "that role the same way causally — and on the one real pair checked "
                      "so far, the match rate did not clear its own untrained-twin floor, "
                      "so 'the models share this feature' is not yet an established claim "
                      "for any pair. Only the small alive fraction of the dictionary "
                      "(often under 10%) can ever show up as an example, and the headline "
                      "alignment number is inflated by searching many candidate "
                      "ground-truth fields per feature, so it must always be read "
                      "next to its permutation-null control, never on its own.")),
    "concepts": StageDoc(
        question="Which features does each model's dictionary actually use when it forecasts, do they group into named concepts, and does another model group the same series the same way?",
        how=("For every trained dictionary, zeroes each candidate feature out of the "
             "model's own reconstruction on the series where that feature fires "
             "hardest, and measures nine properties of the forecast against a "
             "random-direction null. Features that move something are clustered by "
             "what they move into concepts (or the dictionary is declared "
             "non-modular when no clean grouping exists). Each concept's top series "
             "are then checked in every other model: does some feature there "
             "separate the same series better than random series of the same kind "
             "would, and does that feature's own top series come back the other "
             "way? Finally every concept is described in a sentence composed only "
             "from what was measured. Separately, every causal feature across every "
             "model is also pooled into one shared space and clustered directly "
             "against each other, so a group there can span several models even "
             "when a dictionary's own per-target clustering above found no clean "
             "grouping within it."),
        good_bad=("Good: features whose removal moves the forecast well above the "
                   "null, concepts with several members each, and transfers that clear "
                   "the matched null in both directions. Bad: most targets declared "
                   "non-modular (the dictionary's causal features do not group), or "
                   "transfer rates that are high only because the matched null is weak "
                   "for that kind of series."),
        cannot_tell=("A transfer that clears its null says another model separates "
                      "the same series, not that it uses the feature the same way "
                      "causally. The rates are uncorrected for the many tests made, and "
                      "nothing yet says how often a concept would transfer to a second "
                      "dictionary trained on the SAME model, so a rate cannot be read "
                      "as high or low on its own. Everything here is measured on the "
                      "development corpus only.")),
    "exemplars": StageDoc(
        question="What does an aggregate difference between models actually look like on one real series?",
        how=("Picks a handful of concrete series per data family — specifically ones "
             "where the models' forecasts disagree the most — and shows each "
             "one's context, true continuation, both forecasts, per-layer lens "
             "curves, and pooled attention pattern side by side."),
        good_bad=("Good: a case that makes an aggregate statistic from an earlier "
                   "section tangible — you can see exactly where one model's forecast "
                   "goes wrong on a concrete series. Bad: a case that reads as an "
                   "ordinary, unremarkable series despite being flagged as the most "
                   "divergent — worth treating skeptically rather than trusting the "
                   "selection blindly."),
        cannot_tell=("These series are deliberately chosen for maximal disagreement, "
                      "not for being typical or representative — this stage cannot "
                      "tell you how often such a disagreement happens, only that it "
                      "can happen and roughly what it looks like when it does. It is "
                      "illustrative evidence only, the same evidence class as a hand-"
                      "picked example in a paper, not a statistical claim.")),
    "register": StageDoc(
        question="Which of the (many) exploratory findings from the dev corpus are worth spending the one-shot private confirmation on?",
        how=("Reads the dev-corpus findings this run has already produced and freezes "
             "a small, explicit list of testable hypotheses — plus a content hash of "
             "exactly which dev artifact each one came from — before the private "
             "(held-out) corpus is ever touched."),
        good_bad=("Good: a registry whose hypotheses map cleanly onto real dev "
                   "findings with clear pass/fail criteria. Bad: a registry that "
                   "can't be built at all because a dev artifact it depends on "
                   "doesn't exist — which correctly blocks confirmation rather than "
                   "letting it run against nothing."),
        cannot_tell=("This stage provides no evidence itself about which model is "
                      "better or how they compare — it only decides, once, which "
                      "questions get to be asked of the private data. It cannot "
                      "rescue a finding that wasn't registered here; anything not "
                      "pre-registered cannot later be confirmed, by design.")),
    "confirm": StageDoc(
        question="Which of the exploratory dev-corpus findings are actually real, rather than an artifact of looking at many comparisons?",
        how=("Re-runs exactly the pre-registered hypotheses, exactly once, on a "
             "sealed corpus that no exploratory analysis in this run has ever seen, "
             "and reports which ones hold up."),
        good_bad=("Good: a hypothesis that replicates on the private corpus with the "
                   "same direction and a confidence interval excluding zero after "
                   "correction — the strongest evidence this pipeline can produce. "
                   "Bad: a dev finding that fails to replicate (a real and expected "
                   "outcome some of the time, since dev findings are exploratory by "
                   "construction), or a hypothesis marked 'untestable' because this "
                   "stage doesn't yet re-test that kind of claim on private data."),
        cannot_tell=("This section is the actual evidence, but only for the specific "
                      "hypotheses that were pre-registered — it says nothing about "
                      "any exploratory finding that wasn't registered. It also cannot "
                      "be re-run: if the private corpus is ever peeked at, this "
                      "evidence is spent and a fresh held-out corpus is required. A "
                      "'not confirmed' verdict does not mean the dev finding was "
                      "wrong, only that it didn't survive being tested exactly once, "
                      "out of sample.")),
    "spec_curve": StageDoc(
        question="Do this run's headline claims survive plausible variation in the "
                 "analysis's own knobs -- the depth axis, the attention resolution "
                 "mode, the L0 error scale, the bootstrap sample count, which corpus "
                 "rows were sampled, which layer-screening method was used -- or does "
                 "the reported number depend on one particular, somewhat arbitrary "
                 "analysis choice?",
        how=("Re-derives each headline claim (an L0 family-strength verdict, a lens "
             "crystallization state, a layer-screen selection, an attention "
             "periodicity head) from this run's own already-written activation store "
             "and predictions, one knob at a time, and reports what fraction of the "
             "applicable grid preserves the baseline verdict. No model is reloaded "
             "and no forward pass runs -- this is a standalone reducer over existing "
             "artifacts, not a pipeline stage, so it can be run against any "
             "already-completed run without re-extracting anything."),
        good_bad=("Good: a claim whose robust_frac is 1.0 across every applicable "
                   "cell -- the verdict does not depend on which of these knobs was "
                   "chosen. Bad: a robust_frac well below 1.0, meaning the headline "
                   "number in this report is a property of one particular analysis "
                   "configuration rather than of the models being compared -- read "
                   "which specific knob flips it before trusting the headline claim "
                   "on its own."),
        cannot_tell=("This is a robustness diagnostic, not a significance test -- "
                      "`evidence_class: descriptive` throughout, and a claim clearing "
                      "every cell has not been shown significant here, only stable "
                      "under these particular perturbations. The grid is deliberately "
                      "narrow (six knobs needing no re-extraction, swept one at a time "
                      "by default) and a knob not applicable to a given claim is "
                      "excluded from its denominator rather than counted as robust, so "
                      "a perfect score describes only the cells that could be checked. "
                      "The layer-screen family recomputes against this run's own main "
                      "activation store rather than the dedicated stride-1, "
                      "every-block store the production `layer_screen` stage uses and "
                      "then deletes by default, so every cell in that family is marked "
                      "`fair_to_all_layers: false` -- a real, stated deviation from the "
                      "production selection's own fairness guarantee, not the same "
                      "measurement re-run.")),
    "report": StageDoc(
        question="Given everything the other stages found, is this a fair comparison, and what's actually solid enough to act on?",
        how=("Collects every stage's artifacts into one document, states upfront "
             "which measured asymmetries between each pair of models (parameters, "
             "compute, coverage, forecast determinism) apply before any result is "
             "read, and marks every section as rendered, skipped, or failed rather "
             "than silently omitting anything."),
        good_bad=("Good: every enabled stage's section actually rendered (no "
                   "'failed' rows in the coverage panel) and the fairness card names "
                   "its asymmetries with real measured values rather than 'not yet "
                   "measured'. Bad: any section marked 'failed' (a bug, not an "
                   "expected degrade) or a fairness row that couldn't be measured, "
                   "which should downgrade how much weight a depth-located claim in "
                   "that section deserves."),
        cannot_tell=("The report presents what the pipeline measured; it adds no new "
                      "evidence of its own. A 'rendered' status only means a section "
                      "produced output without error, not that a result inside it is "
                      "strong — that's for the evidence-class ladder shown once at "
                      "the top (geometric / translatable / causal-within-model / "
                      "descriptive / illustrative / behavioral) to decide, section by "
                      "section.")),
}


def get(name: str) -> StageDoc:
    """Look up a stage's doc by its `pipeline.stage_names()` name or a known alias.

    Raises `KeyError` (never a silent default or a blank render) for a name
    with no entry — `CLAUDE.md` sec 2.5's "degrade loudly" applies to
    documentation coverage exactly as it does to a broken adapter assumption:
    a stage nobody wrote these four lines for should fail visibly, not
    render an empty box next to every other section's real content.
    """
    if name in STAGE_DOCS:
        return STAGE_DOCS[name]
    if name in _ALIASES:
        return STAGE_DOCS[_ALIASES[name]]
    raise KeyError(
        f"no StageDoc for stage {name!r} -- add one to "
        f"tsfm_lens/stage_docs.py:STAGE_DOCS (ROADMAP.md sec 21 J2)")


def render_markdown(names) -> str:
    """Render `names` (in order) as a Markdown section, one subsection per stage.

    The single function both `render_stage_docs.py` (the README generator)
    and its own drift-check test call, so the README's rendered text is
    always a pure function of `STAGE_DOCS` -- never hand-copied.
    """
    lines = ["## What each stage tells you", "",
             "Four fixed questions per stage (`ROADMAP.md` sec 21 J2), generated "
             "from `tsfm_lens/stage_docs.py` by `render_stage_docs.py` -- edit that "
             "file, not this section, and re-run the script to update it.", ""]
    for name in names:
        doc = get(name)
        lines += [
            f"### `{name}`", "",
            f"**Question.** {doc.question}", "",
            f"**How.** {doc.how}", "",
            f"**Good vs. bad result.** {doc.good_bad}", "",
            f"**What it cannot tell you.** {doc.cannot_tell}", "",
        ]
    return "\n".join(lines).rstrip() + "\n"
