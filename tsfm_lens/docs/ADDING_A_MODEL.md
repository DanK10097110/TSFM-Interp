# Adding a model

You have a time-series foundation model you want compared against the ones
already in this repo. This page covers three things: the `ModelAdapter`
contract every model implements, the path from "I have a checkpoint" to "I
have a rendered report," and a worked case study of the single most common
class of adapter mistake (silently feeding a model the wrong input scale).
Every fact here was checked against the code in this checkout; where a fact
could change (a capability name, a default checkpoint, a count of registered
adapters), this page points at the *generated*
[`tsfm_lens/ADAPTERS.md`](../tsfm_lens/ADAPTERS.md) instead of repeating it by hand.

## 1. The `ModelAdapter` contract (`tsfm_lens/models/base.py`)

Every model enters the pipeline through a `ModelAdapter` subclass, registered
in `models/__init__.py` (built-in) or auto-discovered from
`models/contrib/*.py` (see §2 below). **Tiers are derived from what a
subclass actually overrides, never declared** — `capability_tier()` inspects
the class itself, because a hand-set tier integer is a claim checked nowhere
(the same lesson `CLAUDE.md` applies everywhere else: a label that isn't
tested against the code it describes goes stale).

**Required on every adapter** (tier 0, "black box" — the floor; concrete
methods that raise `CapabilityUnavailable` by default, not `abstractmethod`,
so a black-box model is expressible at all):

- `load()` — instantiate the model.
- `predict(contexts, horizon, quantiles) -> {"point": [B,H], "quantiles": [B,H,Q]}` —
  forecast, in the original scale.

**Tier 1 ("observable") adds:**

- `module` (the `nn.Module` hooks attach to), `prepare(contexts)` (raw
  contexts → model-ready input), `forward(prepared)` (one pass that fires
  capture hooks), and `token_time_spans()` (`[n_tokens, 2]`, each token's
  `(start, end)` in context steps — the single piece of information that
  makes patch-based and per-step tokenizations comparable).
- `default_layer_regex` — anchor it (e.g. `^blocks\.\d+$`); it's matched with
  `re.search`, a substring match, so an unanchored pattern over-matches.

**Tier 2 ("steerable")** requires nothing extra to *implement* — it's a
declaration: `single_pass_context: bool = True` (the default) means the
whole context is processed in one forward pass, so a cached clean state can
be written back into a corrupted run at a fixed token position
(`hooks.token_patch`'s assumption). Every adapter written so far is
single-pass.

**Tier 3 ("decomposable")** adds `attention_info()` and
`attention_patterns()` (both must be overridden). Tier 3 deliberately does
**not** require `mlp_info()` — TimesFM's feed-forward block is two bare
`nn.Linear`s with no wrapping module to name, so requiring it would wrongly
demote the model the attention taxonomy works best on.

| Tier | Name | Requires |
|---|---|---|
| 0 | black box | `load`, `predict` |
| 1 | observable | + `module`, `prepare`, `forward`, `token_time_spans` |
| 2 | steerable | + `single_pass_context` (declaration, defaults True) |
| 3 | decomposable | + `attention_info`, `attention_patterns` |

**Fully optional**, each defaulting to `None`/a safe no-op so an adapter that
doesn't implement it degrades gracefully rather than crashing anything
downstream: `mlp_info`, `cross_attention_patterns` (encoder-decoder only),
`time_localization` (only meaningful for an adapter that *measures* its own
spans — see `generic_hf`), `token_ids` (only meaningful for a re-quantizing
tokenizer, used for probe-amplitude calibration), `postprocess_tokens`/
`token_slice` (override when the captured sequence carries specials or
padding `token_time_spans` doesn't describe), `forecast_reads_patched_positions`
(default `True`; override to `False` if the forecast head reads positions
that carry no time span, so patching-based measurements decline instead of
silently measuring a no-op), and `uncaptured_surfaces()` (declare blocks the
layer regex never matches — Chronos-T5's decoder is the canonical example).

### `CapabilityUnavailable` vs `NotTimeLocalized`

Both are raised by adapter methods, but they mean opposite things and must
never be conflated:

- **`CapabilityUnavailable`** (`NotImplementedError` subclass) — a fact
  **about the adapter**: it never claimed to expose this capability at its
  current tier. The pipeline routes around it.
- **`NotTimeLocalized`** (`ValueError` subclass) — a **measured verdict about
  the model**: an adapter that *derives* its own token→time map (only
  `generic_hf` does this) measured its impulse response and found it too
  diffuse, or found tokens reading disjoint lags rather than one contiguous
  interval, to pool onto a window axis at all. The pipeline catches this one
  specifically and routes that model to `l0`/`budget`/`report` only — not a
  workaround to defeat, a correct and honest response to a real property of
  the model. A hand-written adapter that *declares* its spans never raises
  this; only `GenericHFAdapter` (`measures_own_spans = True`) can.

## 2. The path from checkpoint to report

**Step 0 — check whether it's already been tried.** Read
[`probe_sweep.md`](probe_sweep.md) (generated by `run_probe_sweep.py`): which
candidate checkpoints the zero-code path below has already been tried
against, and why the ones that didn't resolve didn't.

**Step 1 — try the zero-code path first, before writing any adapter code.**
Copy `configs/generic_hf_timer.yaml` (this repo's own worked example — see
also `configs/examples/generic_hf_timer.yaml`), point `checkpoint:` at your
model, then:

```bash
python run.py --config <your config> --probe-adapter <name>
```

This *measures* — never guesses — the four things every adapter needs: which
forward argument carries the series, which modules form the block stack, how
each token maps onto time, and where the forecast comes out. Three outcomes:
**(a)** all four resolve — you're very likely done, no adapter file, nothing
to maintain; **(b)** a named, considered refusal — it names exactly what it
couldn't safely infer, which *is* the specification for the hand-written
adapter you write next; **(c)** a raw crash several frames below this repo's
own code (a nonexistent checkpoint id, an unrecognized `config.json`) — not a
bug in your config.

**Step 2 — if the probe can't reach your checkpoint, write a contrib
adapter.** Copy [`tsfm_lens/models/TEMPLATE_adapter.py`](../tsfm_lens/models/TEMPLATE_adapter.py)
into `tsfm_lens/models/contrib/<name>_adapter.py`. Give it a module-level
`ADAPTER_NAME = "<name>"` string literal — this is how it's discovered: a
static `ast.parse` scan of `models/contrib/*.py`
(`tsfm_lens/models/contrib/__init__.py`) reads `ADAPTER_NAME`/`ADAPTER_CLASS`
via the AST, never `exec`/`import`-and-inspect, so a broken contrib file
can't crash the registry scan for every other adapter. Equivalently, scaffold
it automatically:

```bash
python run.py --new-adapter <name> --checkpoint <your-checkpoint-id>
```

This writes the file plus a matching `configs/smoke_<name>.yaml` (paired
against `mock_patch`) and prints five commands to run **in order** — the
first already passes against the scaffold's own tier-0 stand-in, before you
write a line of real code, so a later failure is about your adapter, not the
scaffolding. Replace the inherited `load()`/`predict()` with your real
checkpoint, then climb tiers by uncommenting the template's tier-by-tier
blocks (each is a working example, not a stub), re-running `--check-adapter`
after each tier. Stop climbing whenever your architecture genuinely has
nothing further to expose — a lower tier reached honestly is a complete
contribution.

**Step 3 — verify.**

```bash
python run.py --config <your config> --check-adapter <name>
python run.py --config <your config> --check-alignment <name>
```

`--check-adapter` runs every check below in dependency order and prints one
typed row (`pass`/`warn`/`fail`/`not_applicable`) per check; it exits
nonzero iff any row fails.

**Step 4 — regenerate the docs.**

```bash
python render_adapter_docs.py
```

This rewrites `tsfm_lens/ADAPTERS.md`. `--check` (no args needed) confirms it
isn't stale — CI runs this.

## 3. Checklist — what adapter authors get wrong, and how to check it

- **Input normalization / scaling.** See the case study below. Check with
  the `frontend` stage's scale-equivariance residual: the report flags a
  value above `max(1.0, 10x the peers' median)`.
- **Sampled decoders.** A model whose `predict()` samples (Chronos-T5,
  Sundial) needs a **seeded** `predict()`, and any two calls being compared
  must use the **same** seed. Free control: predict twice with no patch at
  all and require bit-identical output.
- **Precision.** Forward passes run under bf16 autocast by default. The
  clean cache used for patching (`extraction.hooks`/`capture_raw_tokens`)
  runs with autocast **off**, specifically to match `predict()`'s precision
  — a mismatch here is a silent source of spurious patching noise.
- **Batch chunking.** `hooks.token_patch` (the one intervention primitive
  used by L3, the lens, the SAE, and the concepts stage) raises if the model
  chunks its batch internally. Keep `max_series <= batch_size` everywhere in
  your config.
- **Layer regex.** Anchor `default_layer_regex` (e.g. `^blocks\.\d+$`) — it's
  matched with `re.search`, a substring match, so an unanchored pattern can
  match more than the block stack.
- **Token→time spans.** `token_time_spans()` must give contiguous, per-token
  `(start, end)` intervals. `extraction/span_discovery.py` *measures* them
  independently and the alignment gate cross-checks the two. Patch-based
  models have coarse native lag resolution (TimesFM: one 32-step patch).
- **Time localization.** A model whose impulse response is not
  time-localized is routed to `l0`/`budget`/`report` only — see
  `NotTimeLocalized` above.
- **Encoder-decoder models.** Only the encoder may be capturable in practice
  (Chronos-T5's decoder is not captured here). Declare the uncaptured
  surface via `uncaptured_surfaces()` so the `block` depth axis reflects it,
  and say in your adapter's docstring that depth claims are claims about the
  encoder.
- **Reach controls for patching.** Self-patch (a layer patched into itself)
  must give exactly `0.0`; a cross-layer patch must give a nonzero change.
  Both are free, and both are run by `models/conformance.py` — a clean, flat,
  plausible-looking curve with neither control checked can mean the patch
  never reached anywhere the forecast head reads.
- **`random_init` null twins.** `random_init_like()` gives a valid floor for
  geometry and decodability, but it is **not automatically a causal floor**:
  TimesFM's twin is an exact identity stack (its RMSNorm scale is
  zero-initialized), and the Chronos-2/Chronos-Bolt twins give numerically
  dead forecasts. Check reach against a twin before running a causal battery
  on it.
- **Output conventions.** `predict()` returns `{"point": [B,H], "quantiles":
  [B,H,Q]}` in the *original* scale — denormalize before returning, using the
  *same* per-row stats `prepare()`/`forward()` normalized with (see the
  Sundial case study: recomputing stats a second time is how a normalization
  bug hides).
- **Upstream breakage.** After any library or checkpoint bump, rerun
  `--discover-layers`, `--check-alignment`, and `--check-adapter`. Remote
  code (a checkpoint's own `modeling_*.py`) changes silently between
  revisions.
- **Tests to add.** A mock inner module with a planted, known answer (e.g.
  scale-dependent output for the equivariance test) plus a decoy (a constant
  context must not produce NaN). Use `tests/test_sundial_adapter.py` as the
  template — it plants exactly this class of bug and confirms the fix.

## 4. Case study: the Sundial normalization bug

*(`thuml/sundial-base-128m`; verified against `tsfm_lens/models/sundial_adapter.py`,
which documents this in detail in its own module docstring.)*

**What happened.** HF checkpoints often normalize inputs only inside their
own `generate()`/pipeline path, not inside `forward()` itself. Sundial
z-scores each series (`revin`) inside its remote `generate()` code
(`ts_generation_mixin.py`), on by default there — but `SundialForCausalLM.forward()`
itself defaults to `revin=False`. This repo's adapter calls `forward()`
directly (`.generate()` was broken on the installed `transformers` version —
a `DynamicCache.seen_tokens` incompatibility — and remains unused for a
second, independent reason below), so **every Sundial forecast and captured
activation, before the fix, was computed on raw-scale input**: the
checkpoint's own intended normalization never ran.

Compounding it: the remote `forward(revin=True)` branch — the seemingly
obvious fix — is itself broken for `num_samples > 1` (a broadcast error,
`predictions * stdev + means` fails to match shapes), and uses a *different*
rule from `generate()`'s own (an absolute `1e-2` floor on the std instead of
an additive `1e-5`), which independently breaks scale-equivariance for
small-magnitude series.

**Symptoms.** The `frontend` stage's scale-equivariance residual read 38.4318
context-sd units for Sundial against ≤0.0086 for every other model in
`runs/full_report_run_4model` — a huge number that, without this diagnosis,
reads as a property of the model rather than a bug. At the input level:
10x/100x-rescaling a context collapsed Sundial's forecast to flat (a
0.002 forecast/context-sd ratio, MASE 4.13) where manually z-scoring the
input first did not. On the electricity `sequential_par` benchmark series,
100% of Sundial's forecasts were flat.

**Fix.** Reproduce the checkpoint's *own* `generate()` rule manually, in
exactly one place (`prepare()`): `mu = x.mean(1)`,
`sd = x.std(1, unbiased=False) + 1e-5`, normalize by `(x - mu) / sd`. `prepare()`
returns the normalized tensor **plus its per-row `mu`/`sd`** (not a bare
tensor), so `predict()` denormalizes with the *same* stats `forward()`
normalized with — never a second, independently-computed
`contexts.mean()/.std()` call that could silently drift from the first.
Because both `forward()` (capture) and `predict()` (and anything patched
through `hooks.token_patch`, whose clean cache also goes through
`prepare()` + `forward()`) route through this one function, capture,
forecasting, and patching all see an identical normalized input — there is
no second code path into the backbone that could see raw-scale input
instead.

**Numbers after the fix** (same run, same corpus):

| Metric | Before | After |
|---|---:|---:|
| Sundial scale-equivariance residual | 38.4318 | 0.029955 |
| v1 median MASE | 1.3583 | 1.3109 |
| Flat-forecast share, electricity `sequential_par` | 1.0 | 0.08 |

The **same class of bug** was independently found in the zero-code
`generic_hf` adapter, applied to Timer: its scale-equivariance residual went
from 49.7334/3.8053 to 0.0198/0.0214 after the same fix pattern (normalize
once, denormalize with the same stats, in the one function every code path
routes through).

**The lesson:** when you call a model below its public API (`forward()`
instead of `.generate()`, a bare `nn.Module` instead of a pipeline wrapper),
you inherit the responsibility to reproduce **everything** the public API
did to the input and output — not just the forward pass itself. A model's
own normalization is exactly this kind of thing: easy to miss because the
model still runs and produces a plausible-looking (if degenerate) forecast.

## 5. One architecture class per contribution

Prefer a model that breaks a *different* assumption from what's already
registered over one that looks architecturally identical to an existing
adapter — the second kind confirms something already confirmed. If your
model's own checkpoint code, or this repo's shared extraction/hooking code,
turns out to have a bug that only your architecture exercises, **fix it in
the shared module**, not with a per-adapter workaround — the next
contributor's architecture inherits the fix for free.

## 6. Licensing

Record your checkpoint's license on the model config, the same way the
benchmark-generation half of this repo records a license per real data
source (`SourceRef`). See [`tsfm_lens/ADAPTERS.md`](../tsfm_lens/ADAPTERS.md) for the
convention in more detail and worked examples (e.g. Lag-Llama, Apache 2.0).

## Where the facts live

- **[`tsfm_lens/ADAPTERS.md`](../tsfm_lens/ADAPTERS.md)** (generated — `render_adapter_docs.py`
  to refresh, `--check` to confirm it isn't stale): every currently-registered
  adapter, its tier, declared capabilities, default checkpoint, and a
  narrative explaining what each verification step measures.
- **[`probe_sweep.md`](probe_sweep.md)** (generated — `run_probe_sweep.py`):
  which candidate checkpoints the zero-code path has already been tried
  against.
- **[`worked_example.md`](worked_example.md)**: not about adapters — a
  section-by-section reading of one finished report.
- **`python run.py --check-adapter <name>`**: the one command that runs
  every verification step in dependency order for your specific adapter.

If you find yourself reading adapter source files to answer a question
neither this page nor `ADAPTERS.md` answers, that's a gap in one of those two
documents, not a sign of some third, unwritten source of truth.
