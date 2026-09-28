# Contributing a new model adapter

You have a time-series foundation model you want compared against the ones
already in this repo. This page is the path from "I have a checkpoint" to
"I have a rendered report." It is process only — every fact that could
change (which tier unlocks which stage, a capability name, a default
checkpoint, a count of registered adapters) lives in the *generated*
`tsfm_lens/ADAPTERS.md` (a subdirectory of this same directory, sibling to
this `docs/` directory — not "up" from here), never in this file, so this
page cannot go stale the way a hand-written list of tiers or stages would
(`CLAUDE.md` §11.34 — a hand-maintained fact is a claim checked nowhere).
Read `tsfm_lens/ADAPTERS.md` first if you haven't; it also carries a "How to
read this file" section that explains what each verification step actually
measures, why a refusal from one of them is a valid, honest result rather
than a bug to route around, why a shared bug found while adding your model
gets fixed in the shared code rather than worked around in your adapter
file, and the licensing convention to follow. Nothing below repeats that
content — it is the companion to this page, not optional background reading.

**Run every command on this page from the directory that directly contains
this repo's own `README.md`, `run.py` and `configs/`** — the parent
directory of the `docs/` folder this file lives in, and a sibling of the
`tsfm_lens/` package directory named above.

## Step 0 — check whether the reconnaissance is already done

Before running anything, read `docs/probe_sweep.md` (in this same directory)
— the regenerable record of which candidate checkpoints the zero-code path
below has already been tried against, and why the ones that didn't resolve
didn't (a refused gate, or a **crash before this repo's own code ever ran at
all** — e.g. the checkpoint id doesn't exist, or its `config.json` isn't in a
form `transformers.AutoConfig` recognizes — which is a different and
important distinction from a considered refusal; see that file's own
header). Someone may have already done this for your checkpoint or one
shaped like it. Do this *before* Step 1, not after — a probe run costs
network time and produces a raw traceback for the "crash before our code
ran" case, which is expected but easy to mistake for a broken config if you
hit it cold.

## Decision tree

**Before writing any adapter code at all, try the zero-code path.** You need
a config file — copy `configs/generic_hf_timer.yaml` (this repo's own
worked example of exactly this path) and change only its `checkpoint:` and
model `name:` fields to point at your model; nothing else in this page
specifies config syntax, and that file is the canonical starting point. Then
run:

```
python run.py --config <your config> --probe-adapter <name>
```

This measures — never guesses — the four things every adapter needs: which
forward argument carries the series, which modules form the block stack,
how each token maps onto time, and where the forecast comes out. Three
outcomes, not two: **(a)** all four resolve — you are very likely done, no
adapter file, no registration, nothing to maintain; **(b)** a named,
considered refusal — it names exactly what it could not safely infer, and
that refusal *is* the specification for the adapter you write next, not a
dead end; **(c)** a raw crash from a library several frames below this
repo's own code (a nonexistent checkpoint id, an unrecognized
`config.json`) — this is Step 0's third outcome, not a bug in your config;
`docs/probe_sweep.md` has real examples of this exact shape.

**If the probe genuinely can't reach your checkpoint** (outcome (b) or (c)
above, or it needs a covariate, or it's multivariate), write a contrib
adapter:

```
python run.py --new-adapter <name> --checkpoint <your-checkpoint-id>
```

This scaffolds `tsfm_lens/models/contrib/<name>_adapter.py` (a copy of
`models/TEMPLATE_adapter.py` with your name and checkpoint filled in) and a
matching `configs/smoke_<name>.yaml`, and prints five commands to run next, in
order, each annotated with what a good result looks like at that point.
Follow them in the order printed — they are ordered so that the very first
one already passes, against the scaffold's own tier-0 stand-in, before you
have written a single line against your real checkpoint. That first pass is
not a formality: it tells you the scaffolding, registration and config are
correct, so any failure after you start filling in real code is about your
adapter, not about the machinery around it.

From there: replace the inherited `load()`/`predict()` with calls into your
real checkpoint, then climb tiers by uncommenting the template's own
tier-by-tier blocks (each one is a working example, not just a stub — read
its comments), re-running `--check-adapter <name>` after each tier. Stop
climbing whenever your architecture genuinely has nothing further to expose;
a lower tier reached honestly is a complete, valid contribution, not an
unfinished one.

Run the five printed commands **in the order printed**, not ahead of where
your adapter actually is: commands 3 and 4 assume tier 1 is implemented, and
running either one against a still-tier-0 scaffold raises a raw
`CapabilityUnavailable` traceback rather than a clean message — a real, known
rough edge (unlike `--check-adapter`, which degrades to a typed `n/a` row for
the identical situation), not a sign anything is broken. Wait until you've
uncommented the tier-1 methods before running commands 3 and 4.

## What a refusal means

Several checks in this pipeline are designed to refuse rather than guess —
a diffuse impulse response, tokens that read disjoint lags instead of one
contiguous span, a capability your architecture genuinely doesn't have. None
of these are failures on your part or gaps you need to route around. `tsfm_lens/ADAPTERS.md`'s
narrative explains the specific mechanism for each check; the general
principle is that a refusal with a named, measured reason is worth more to
this repo than a plausible-looking guess that turns out to be wrong two
stages downstream.

## One architecture class per contribution, and where a shared bug gets fixed

Prefer a model that breaks a *different* assumption from what's already
registered over one that looks architecturally identical to an existing
adapter — the second kind confirms something already confirmed several
times over. If your model's own checkpoint code, or this repo's shared
extraction/hooking code, turns out to have a bug that only your architecture
exercises, fix it in the shared module rather than working around it in your
adapter file — the next contributor's architecture inherits the fix for free
that way, instead of re-discovering the same gap.

## Licensing

Record your checkpoint's license on the model config, the same way the
benchmark-generation half of this repo already records a license per real
data source. The convention and its reasoning are in `tsfm_lens/ADAPTERS.md`'s
narrative — read it there rather than here, so this page never states a
convention that has since moved.

## Where the facts live

- **`tsfm_lens/ADAPTERS.md`** (generated — run `render_adapter_docs.py` to refresh it,
  `--check` to confirm it isn't stale): every currently-registered adapter,
  its tier, its declared capabilities, its default checkpoint, and the
  narrative explaining all of the above in detail.
- **`docs/probe_sweep.md`** (generated — `run_probe_sweep.py` to refresh):
  which candidate checkpoints the zero-code path has already been tried
  against, and the outcome.
- **`docs/worked_example.md`**: not about adapters — a section-by-section
  reading of one finished report. Useful once your adapter is registered and
  you want to know what the numbers it produces actually mean.
- **`python run.py --check-adapter <name>`**: the one command that runs every
  verification step in dependency order and tells you, for your specific
  adapter, right now, what still needs work.

Nothing else about "how to add a model" should need to exist. If you find
yourself reading adapter source files to answer a question this page and
`tsfm_lens/ADAPTERS.md` don't answer, that is a gap in one of those two documents —
not a sign you're missing some third, unwritten source of truth.
