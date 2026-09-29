"""`run.py --new-adapter NAME --checkpoint ID` (`ROADMAP.md` sec 34.6 Item E3).

Scaffolds a copy-paste-ready contrib adapter (by substitution over
`TEMPLATE_adapter.py`, never a second, hand-written generator -- `CLAUDE.md`
sec 2.2) plus a config pairing it against `mock_patch`, and returns the five
commands a contributor runs next, in dependency order (matching E4.2's own
check ordering), each with what a good result looks like at that point.

Pairing against `mock_patch` rather than another mock is deliberate (E3's
own spec): `mock_patch` is tier 3, so as the contributed adapter climbs
tiers, this SAME config starts exercising l1/l2/attention/... with no edits
needed -- the comparison stages gate on `min(tier_a, tier_b)`, so nothing
about the config has to change when the adapter's tier does.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

_PKG_DIR = Path(__file__).resolve().parent  # tsfm_lens/tsfm_lens
TEMPLATE_PATH = _PKG_DIR / "models" / "TEMPLATE_adapter.py"


class ScaffoldError(ValueError):
    """A refusal naming exactly what is wrong -- never a partial write."""


@dataclass
class ScaffoldResult:
    name: str
    checkpoint: str
    class_name: str
    adapter_rel_path: str
    config_rel_path: str
    commands: List[str] = field(default_factory=list)


def class_name_for(name: str) -> str:
    """`my_model` -> `MyModelAdapter`; already-suffixed names are not doubled."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", name) if p]
    camel = "".join(p[:1].upper() + p[1:] for p in parts) or "New"
    if not camel.endswith("Adapter"):
        camel += "Adapter"
    return camel


def validate_name(name: str, known_adapters: Set[str]) -> None:
    """Failure modes 2 and 3 from E3's own list, checked before any file write.

    `ADAPTER_NAME` becomes a module-level string a class name is also
    derived from, so it must be a plain Python identifier -- a hyphenated or
    digit-leading name would produce a contrib file that fails to import,
    which is a worse failure than refusing up front and naming why.
    """
    if not name or not name.isidentifier() or keyword.iskeyword(name):
        raise ScaffoldError(
            f"'{name}' is not a valid Python identifier -- it becomes a module-level "
            f"ADAPTER_NAME and part of a class name, so it must be letters/digits/"
            f"underscore only, not start with a digit, and not be a Python keyword")
    if name in known_adapters:
        raise ScaffoldError(
            f"adapter name '{name}' is already registered -- pick a different name "
            f"(currently registered: {sorted(known_adapters)})")


def render_adapter_source(name: str, checkpoint: str, class_name: str) -> str:
    """Substitute over `TEMPLATE_adapter.py`'s own text -- never regenerate it."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    docstring = (
        f'"""Scaffolded for checkpoint "{checkpoint}" (ROADMAP.md sec 34.6 Item E3).\n\n'
        f'    Still a `MockBlackBoxAdapter` clone -- it forecasts and nothing else is\n'
        f'    real yet. Override `load()` to actually fetch/load "{checkpoint}" (the\n'
        f'    `self.cfg.checkpoint or "{checkpoint}"` convention every real adapter in\n'
        f'    this repo uses, so a config can still override it) and `predict()` to\n'
        f'    call it, then climb the tiers commented in below.\n'
        f'    """'
    )
    # "TemplateAdapter" is replaced EVERYWHERE, not just at its declaration --
    # the template's own module docstring mentions it in prose too ("As
    # shipped, this file runs. `TemplateAdapter` subclasses..."), and leaving
    # that occurrence untouched would ship a file whose class is renamed but
    # whose own docstring still talks about a class that no longer exists.
    if template.count("TemplateAdapter") < 3:
        raise ScaffoldError(
            "TEMPLATE_adapter.py no longer mentions 'TemplateAdapter' at least 3 times "
            "(class decl + ADAPTER_CLASS + docstring prose) -- this scaffolder's "
            "substitutions are out of sync with the template; fix scaffold_adapter.py "
            "before regenerating")
    source = template.replace("TemplateAdapter", class_name)

    replacements = [
        ('ADAPTER_NAME = "template"', f'ADAPTER_NAME = "{name}"'),
        ('"""Rename this class (and update ADAPTER_CLASS above) before real use."""',
         docstring),
    ]
    for old, new in replacements:
        if old not in source:
            # Defensive, not a normal user-facing failure (CLAUDE.md sec 2.5):
            # if the template's own text ever changes, a silent no-op
            # substitution here would ship a file still named
            # TemplateAdapter/"template" -- fail loudly instead of writing
            # something that looks scaffolded but registers under the wrong
            # name.
            raise ScaffoldError(
                f"TEMPLATE_adapter.py no longer contains the expected text {old!r} -- "
                f"this scaffolder's substitutions are out of sync with the template; "
                f"fix scaffold_adapter.py before regenerating")
        source = source.replace(old, new, 1)
    return source


def render_config_source(name: str, checkpoint: str) -> str:
    """`configs/smoke_<name>.yaml`, paired against `mock_patch` (see module docstring).

    Every path here is relative (`out_dir: runs`), matching `CLAUDE.md`
    invariant 11 -- nothing in a generated config may hardcode an absolute
    path.
    """
    return f"""# Scaffolded by `run.py --new-adapter {name} --checkpoint {checkpoint}`
# (ROADMAP.md sec 34.6 Item E3). Pairs the new adapter against `mock_patch`
# (tier 3) rather than another mock: as '{name}' climbs tiers this SAME
# config starts exercising l1/l2/attention/... with no edits needed, since
# the comparison stages gate on min(tier_a, tier_b).
#
# As shipped ('{name}' is still a MockBlackBoxAdapter clone) this reaches
# only a tier-0 report -- see the numbered commands `--new-adapter` printed.

run:
  name: smoke_{name}
  out_dir: runs
  device: cpu
  dtype: float32
  seed: 0

data:
  source: smoke
  context_len: 128
  horizon: 32
  smoke_series_per_family: 20

alignment:
  window: 32

models:
  - name: patchy
    adapter: mock_patch
    batch_size: 64
  - name: {name}
    adapter: {name}
    batch_size: 64
    checkpoint: "{checkpoint}"

l0:
  calibration: true

budget:
  measure_predict: true

report:
  verbose: false
"""


def command_plan(name: str) -> List[str]:
    """The five commands E3 asks for, in E4.2's own dependency order.

    Ordered so that running them exactly as printed, from the as-shipped
    tier-0 scaffold, never ends at an unexplained stack trace (E3's own
    acceptance bar): commands 1 and 2 are true RIGHT NOW, against the
    pristine scaffold; commands 3-5 are each labelled with the tier they
    need, so a contributor knows which ones to skip until they've climbed
    that far rather than discovering it from a `CapabilityUnavailable`.
    """
    cfg = f"configs/smoke_{name}.yaml"
    return [
        (f"1. python run.py --config {cfg} --check-adapter {name}\n"
         f"   Good result now (tier 0): 'overall: pass', every other row "
         f"printed as '[n/a ]' (status not_applicable) -- that is correct, "
         f"not a failure (E4.3)."),
        (f"2. python run.py --config {cfg}\n"
         f"   Good result now: a rendered runs/smoke_{name}/report.html naming "
         f"'{name}' -- confirms the scaffold and config work before you touch "
         f"real adapter code."),
        (f"3. python run.py --config {cfg} --discover-layers {name} --contains block\n"
         f"   Good result once you've filled in module/prepare/forward (tier 1): "
         f"your real block-stack module names, to anchor your layer regex against "
         f"(CLAUDE.md sec 11.30 -- an unanchored regex silently merges two stacks)."),
        (f"4. python run.py --config {cfg} --check-alignment {name}\n"
         f"   Good result once token_time_spans is real: a diagonal-hit fraction "
         f"near this check's own printed ceiling at shallow layers, decaying at "
         f"deeper ones is healthy; near-zero everywhere means the declared spans "
         f"are wrong (CLAUDE.md sec 11.22/sec 6.3)."),
        (f"5. python run.py --config {cfg} --check-adapter {name}\n"
         f"   Good result once every capability you implemented is in: "
         f"'overall: pass' again, now with those rows reading pass/warn instead "
         f"of not_applicable."),
    ]


def scaffold(name: str, checkpoint: str, known_adapters: Set[str],
             contrib_dir: Path, configs_dir: Path) -> ScaffoldResult:
    """Write the adapter file and its config, refusing before writing either.

    Both refusal checks (name validity/collision, then file existence) run
    before any write, so a mid-pair failure never leaves one file written
    and the other missing (E3's "refuses to overwrite... naming it" failure
    mode, applied to the pair as a unit).
    """
    validate_name(name, known_adapters)
    class_name = class_name_for(name)
    adapter_path = contrib_dir / f"{name}_adapter.py"
    config_path = configs_dir / f"smoke_{name}.yaml"
    for path in (adapter_path, config_path):
        if path.exists():
            raise ScaffoldError(
                f"refusing to overwrite existing file: {path} -- delete it first "
                f"if you really mean to regenerate it")

    adapter_source = render_adapter_source(name, checkpoint, class_name)
    config_source = render_config_source(name, checkpoint)

    contrib_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(adapter_source, encoding="utf-8")
    config_path.write_text(config_source, encoding="utf-8")

    return ScaffoldResult(
        name=name, checkpoint=checkpoint, class_name=class_name,
        adapter_rel_path=f"models/contrib/{name}_adapter.py",
        config_rel_path=f"configs/smoke_{name}.yaml",
        commands=command_plan(name),
    )
