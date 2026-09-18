"""Derived `ADAPTERS.md` (`ROADMAP.md` sec 34.6 Item E2).

Everything below is derived from the running registry -- `models.ADAPTERS`,
`models.CONTRIB_REGISTRY`, `models.discovery_errors`, and each adapter
class's own `capability_tier()` -- never hand-maintained, following the same
no-drift discipline as `stage_docs.py`/`glossary.py` (one source of truth,
two rendering surfaces cannot exist here since there is only one: this file
IS the surface, checked stale-or-fresh by `render_adapter_docs.py --check`).

A hand-written `configs/README.md`-style index was already rejected once in
this repo (`run.py`'s own module docstring: "a claim checked nowhere") for
the identical reason a hand-written adapter table would be: nothing forces
it to be updated when an adapter's tier changes, gains a capability, or a
contrib file breaks. Deriving it costs nothing per adapter and loading no
checkpoint -- every value here comes from a class object or an already-run
discovery scan, same evidence-class discipline as `capability_matrix.py`
(`CLAUDE.md` sec 2.6): this is the STATIC, declared-only half of that split,
never "verified against a live run."
"""

from __future__ import annotations

import inspect
import re
from typing import Dict, List, Optional, Tuple

_CHECKPOINT_LITERAL = re.compile(r'cfg\.checkpoint or "([^"]+)"')

# E2.2's narrative half. Deliberately hand-written and deliberately about
# *process*, never a fact that could drift (ROADMAP.md sec 34.6 Item E2.2 /
# E5's own rule: "every fact that could drift belongs in E2's derived file,
# not here" -- restated in reverse for this module, since this IS E2's
# derived file: no number, tier, count or capability name is hardcoded into
# this prose; every one of those appears only in the generated table above.
_NARRATIVE = """\
## How to read this file

**The four probes, and what each one actually tells you.** A contributor
never has to write all four by hand -- `run.py --check-adapter <model>`
(Item E4) runs them in this order and turns each into one row:

1. **`conformance`** (`models.conformance.check_adapter_conformance`) --
   the cheapest, most load-bearing check. It patches a layer's own clean
   state into itself and requires the forecast move by *exactly* zero
   (`CLAUDE.md` sec 11.42's patch-identity control), then patches an
   earlier layer in and requires a *nonzero* move (the patch-reach
   control). A model whose forecast head samples (`predict()` returns a
   different answer on two identical calls) needs both calls seeded the
   same way, or this check reports a false failure that looks like a
   position bug (`CLAUDE.md` sec 11.50) -- the checklist already does this
   for you.
2. **`check_alignment`** (`extraction.alignment.run_alignment_gate`) --
   an impulse probe asking whether perturbing window *w* of the input
   perturbs pooled window *w* of the activations most. A diagonal-hit
   fraction near the check's own measured *ceiling* (not literally 1.0 --
   some architectures cannot reach 1.0 even when correct, `CLAUDE.md`
   sec 11.35) at shallow layers, decaying at deeper ones, is healthy:
   attention mixes positions by design. Near-zero at every layer means the
   declared `token_time_spans()` do not match what the model actually
   reads.
3. **`discover_spans`** (`extraction.span_discovery.discover_spans` /
   `compare_declared`) -- *measures* each token's time span from its own
   impulse response instead of trusting what the adapter declares, and
   reports the IoU between the two. A low IoU with a real, contiguous
   measured span means `token_time_spans()` is simply wrong and should be
   fixed. A **refusal** (`SpanDiscovery.refusal_reason()` returns non-None)
   is a different, third outcome -- see below.
4. **Capability rows** (`models.capability_matrix.declared_capabilities`) --
   one row per optional method (`attention_info`, `attention_patterns`,
   `mlp_info`, `cross_attention_patterns`). `not_applicable` here is not a
   failure: it means this adapter's tier does not claim the capability, or
   the architecture genuinely has nothing to hook there (TimesFM's own
   `mlp_info` is a real, permanent `None` -- its feed-forward block is two
   bare `nn.Linear`s with no wrapping module, `CLAUDE.md` sec 6.2's support
   matrix).

**Why a refusal is a result, not a bug to work around.** `discover_spans`
can raise `NotTimeLocalized` -- a *measured* verdict that this model's
impulse response is too diffuse to pool onto a window axis, or that its
tokens read a set of disjoint lags rather than one contiguous interval
(`CLAUDE.md` sec 12's envelope edge). Routing that model to `l0`-only
analysis is the correct, honest response, not a workaround to defeat. Do
not catch `ValueError` broadly to make a refusal go away -- `NotTimeLocalized`
is a `ValueError` subclass specifically so it can be told apart from an
ordinary bug (`CapabilityUnavailable`, by contrast, is a `NotImplementedError`
subclass: a fact about what the adapter author wrote, never a verdict about
the model).

**One architecture class per contribution.** A seventh
attention-over-contiguous-patches transformer confirms generality this repo
has already confirmed several times over; a model that breaks a *different*
envelope assumption (non-contiguous lag tokens, no attention at all, a
non-causal encoder, a robust per-series scaler front end -- `ROADMAP.md`
§19/D2's own ordering) is worth far more per session. If your model looks
architecturally identical to one already registered, check whether
`generic_hf` (the zero-code probe path) already reaches it before writing a
new file at all.

**Why a bug found while adding a new adapter gets fixed in the shared
module, not worked around locally.** `extraction/hooks.py`'s tuple-only
output check (`CLAUDE.md` sec 11.21) and the alignment probe's fixed
impulse amplitude (sec 11.16/11.26) were both found this way: a new
architecture exercised an assumption three prior adapters had never
stressed. The fix in both cases landed in the shared module precisely so
the *next* new architecture inherits it for free. A per-adapter workaround
for a shared-infrastructure bug just defers the same investigation to
whoever adds the next model.

**What each tier costs to implement and what it buys.** Tier 0 needs only
`load()`/`predict()` -- a hosted or API-only model with no accessible
internals still gets `l0`/`budget`/`report`/`confirm`. Tier 1 adds
`module`/`prepare`/`forward`/`token_time_spans` -- the whole activation
pipeline: extraction, L1/L2 geometry, L3 patching, clustering, SAE. Tier 2
additionally declares `single_pass_context` (the whole context is
processed in one forward pass, which every within-model patch assumes).
Tier 3 additionally overrides **both** `attention_info` and
`attention_patterns` -- head-level analysis and attention taxonomy. The
tier is *derived* from what a subclass actually overrides, never a number
you set by hand (`CLAUDE.md` sec 6.2: a hand-set integer is a claim checked
nowhere).

**Licensing.** Record a `license` field on the model config, mirroring the
benchmark generation package's own `SourceRef` discipline for real-data
sources -- so a checkpoint's terms travel with the config that names it,
not only with whatever page it was downloaded from.
"""


def default_checkpoint(adapter_cls: type) -> Optional[str]:
    """The literal default checkpoint id in `adapter_cls.load`, if any.

    Every real adapter in this repo declares its default the same way --
    `self.cfg.checkpoint or "<repo/id>"` -- so a source regex finds it
    without instantiating or loading anything. Returns None for a mock (no
    checkpoint at all) or a contrib adapter that requires an explicit
    `checkpoint:` in its config (`CLAUDE.md`'s `GenericHFAdapter` precedent:
    "there is no default to guess at" is itself a valid, statable answer,
    not a gap in this function).
    """
    try:
        source = inspect.getsource(adapter_cls.load)
    except (OSError, TypeError):
        return None
    match = _CHECKPOINT_LITERAL.search(source)
    return match.group(1) if match else None


def stages_unlocked_by_tier(tier: int) -> List[str]:
    """Every stage `pipeline.stage_names()` can produce at this tier or below.

    Inverts `pipeline._STAGE_MIN_TIER` (the same dict `doctor.py`'s own tier
    gate reads) rather than re-deriving the tier table by hand -- a second,
    hand-copied version of that mapping is exactly the kind of duplicate
    `CLAUDE.md` sec 2.1/sec 11.24 warns can drift the moment one of the two
    is edited alone.
    """
    from .pipeline import _STAGE_MIN_TIER, stage_names

    return [name for name in stage_names() if _STAGE_MIN_TIER.get(name, 1) <= tier]


def adapter_row(name: str, adapter_cls: type, *,
                is_contrib: bool = False, module_path: str = "") -> dict:
    """One resolved adapter's full row -- no checkpoint loaded, class only."""
    from .models.capability_matrix import CAPABILITY_METHODS, declared_capabilities
    from .models.base import TIER_NAMES

    tier = adapter_cls.capability_tier()
    declared = declared_capabilities(adapter_cls)
    return {
        "name": name,
        "status": "ok",
        "source": module_path if is_contrib else f"models.{adapter_cls.__module__.rsplit('.', 1)[-1]}",
        "contrib": is_contrib,
        "class_name": adapter_cls.__name__,
        "tier": tier,
        "tier_name": TIER_NAMES[tier],
        "default_checkpoint": default_checkpoint(adapter_cls),
        "stages_unlocked": stages_unlocked_by_tier(tier),
        "declared_capabilities": [m for m in CAPABILITY_METHODS if declared[f"{m}_declared"]],
    }


def build_adapter_table() -> Tuple[List[dict], List[dict]]:
    """`(rows, error_rows)` for every name the registry currently knows about.

    `rows` covers every built-in and every contrib adapter that resolves
    cleanly. `error_rows` covers contrib files discovery already flagged
    (`models.discovery_errors` -- a missing literal, an unparseable file, a
    name collision) PLUS any contrib adapter whose class fails to actually
    import when this function tries it (its own top-level code, or a heavy
    library it needs, may not be installed in whatever environment
    generates this doc) -- named by file rather than silently dropped
    (`CLAUDE.md` sec 2.5).
    """
    from . import models as models_pkg
    from .models import contrib as contrib_pkg

    rows: List[dict] = []
    for name, adapter_cls in sorted(models_pkg.ADAPTERS.items()):
        rows.append(adapter_row(name, adapter_cls))

    error_rows: List[dict] = list(models_pkg.discovery_errors)
    for name, (module_path, class_name) in sorted(models_pkg.CONTRIB_REGISTRY.items()):
        try:
            adapter_cls = contrib_pkg.import_contrib_class(module_path, class_name)
        except Exception as exc:
            error_rows.append({"file": module_path.rsplit(".", 1)[-1] + ".py",
                               "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(adapter_row(name, adapter_cls, is_contrib=True, module_path=module_path))
    return rows, error_rows


def render_markdown(rows: List[dict], error_rows: List[dict]) -> str:
    """The full `ADAPTERS.md` body -- a standalone generated file, not a
    splice into a hand-authored one, since (unlike `stage_docs.py`'s report
    section) nothing else renders this content live; the file itself is the
    only surface."""
    lines = [
        "<!-- GENERATED by render_adapter_docs.py from tsfm_lens/adapter_docs.py "
        "-- do not hand-edit; run render_adapter_docs.py to regenerate. -->",
        "# Model adapters",
        "",
        "Every adapter currently registered in this checkout, derived from the "
        "running registry (`models.ADAPTERS` + `models.CONTRIB_REGISTRY`) -- "
        "nothing below is hand-maintained, so it cannot go stale the way a "
        "hand-written index would (`CLAUDE.md` sec 11.34).",
        "",
        "**Adding your own** (`ROADMAP.md` sec 34.6 Item E1): copy "
        "`tsfm_lens/models/TEMPLATE_adapter.py` into `tsfm_lens/models/contrib/"
        "<name>_adapter.py`, fill in `load()`/`predict()`, then run "
        "`python run.py --config <your config> --check-adapter <name>` "
        "(sec 34.6 Item E4) for a one-command checklist of everything below.",
        "",
        "| adapter | source | tier | default checkpoint | declared capabilities | "
        "stages unlocked |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        checkpoint = row["default_checkpoint"] or "_(none -- set `checkpoint:` in config)_"
        capabilities = ", ".join(row["declared_capabilities"]) or "_none_"
        stages = ", ".join(row["stages_unlocked"])
        source = f"contrib/{row['source'].rsplit('.', 1)[-1]}.py" if row["contrib"] else row["source"]
        lines.append(f"| `{row['name']}` | {source} | {row['tier']} "
                     f"({row['tier_name']}) | {checkpoint} | {capabilities} | {stages} |")

    lines += ["", "## Contrib files that failed to register", ""]
    if not error_rows:
        lines.append("_None -- every file in `models/contrib/` registered cleanly._")
    else:
        lines.append("| file | problem |")
        lines.append("|---|---|")
        for err in error_rows:
            lines.append(f"| `{err['file']}` | {err['error']} |")
    lines.append("")
    lines.append(_NARRATIVE)
    return "\n".join(lines)
