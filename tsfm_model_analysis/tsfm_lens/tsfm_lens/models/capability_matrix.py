"""Auto-generated `ModelAdapter` capability matrix (ROADMAP.md §10, Phase 5's
"auto-generated model-zoo capability matrix" + "auto-generate the capability
matrix from the registry" checklist items).

Two layers, kept explicitly distinct so a "declares this method" signal is
never mistaken for "actually returned something at runtime for a specific
checkpoint" (`CLAUDE.md` §2.6's evidence-class discipline):

- `declared_capabilities` -- static: does the adapter *class* override the
  base `ModelAdapter` default for each optional capability? Free, needs no
  loaded model, but only tells you the adapter *tries*; e.g. TimesFM
  overrides `mlp_info` (it calls the shared `_scan_mlp` helper) but that
  scan finds nothing for TimesFM's bare two-`nn.Linear` feed-forward block
  (`CLAUDE.md` §6.2) -- a fact only a real run can surface.
- `verified_capabilities_from_run` -- reads an already-completed run's
  `attention/meta.json` + `attention/arrays.npz` and reports what actually
  produced usable output for one model in that run. Deliberately reuses
  existing artifacts (e.g. `runs/medium_run_chronos_base`) rather than
  loading any checkpoint itself, so building and testing this module needs
  no live GPU work.

`build_capability_matrix` merges both into one table, one row per
registered adapter; `render_capability_matrix_markdown` renders it for docs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import TIER_NAMES, ModelAdapter

CAPABILITY_METHODS = ["attention_info", "mlp_info", "attention_patterns",
                      "cross_attention_patterns"]


def declared_capabilities(adapter_cls: type) -> dict:
    """Which optional capabilities this adapter *class* overrides, statically.

    Comparing the class's own method object to `ModelAdapter`'s catches the
    case every adapter in this repo actually uses (subclass defines its own
    method); a subclass that reassigned the identical base implementation
    via some other indirection would slip through unnoticed, but none does.
    """
    return {f"{m}_declared": getattr(adapter_cls, m) is not getattr(ModelAdapter, m)
            for m in CAPABILITY_METHODS}


def verified_capabilities_from_run(run_dir, model_name: str) -> Optional[dict]:
    """What actually produced usable output for `model_name` in a completed run.

    Returns `None` if the run has no `attention`-stage artifacts at all (the
    stage was disabled, or the run predates it) rather than a dict of all
    `False`, so a caller can tell "not checked" apart from "checked and
    found nothing."
    """
    from ..utils import load_json

    meta_path = Path(run_dir) / "attention" / "meta.json"
    arrays_path = Path(run_dir) / "attention" / "arrays.npz"
    if not meta_path.exists() or not arrays_path.exists():
        return None
    meta = load_json(meta_path)
    if model_name not in meta:
        return None
    files = set(np.load(arrays_path).files)
    return {
        "attention_info_verified": f"head_delta_{model_name}" in files,
        "mlp_info_verified": f"mlp_delta_{model_name}" in files,
        "attention_patterns_verified": f"lag_profile_{model_name}" in files,
        "cross_attention_patterns_verified": f"cross_profile_{model_name}" in files,
    }


def build_capability_matrix(registry: dict, verify: Optional[dict] = None) -> pd.DataFrame:
    """One row per registered adapter.

    `verify` is an optional `{adapter_name: (run_dir, model_name)}` map --
    only adapters actually exercised by some already-completed run can be
    empirically checked, so this is opt-in per adapter rather than assumed
    for the whole registry. Verified columns are left as `pd.NA` (not
    `False`) for any adapter not in `verify`, so "never checked" stays
    visually distinct from "checked, capability declared but produced
    nothing" -- exactly the TimesFM/`mlp_info` case this module exists to
    surface honestly.
    """
    verify = verify or {}
    rows = []
    for adapter_name, adapter_cls in sorted(registry.items()):
        # Tier first, because it is the coarse answer the per-capability
        # columns then explain: it is what the pipeline actually gates on
        # (`ROADMAP.md` sec 19 G1), while the columns say which specific
        # method is why.
        row = {"adapter": adapter_name, "tier": adapter_cls.capability_tier(),
               "tier_name": TIER_NAMES[adapter_cls.capability_tier()]}
        row.update(declared_capabilities(adapter_cls))
        verified = None
        if adapter_name in verify:
            run_dir, model_name = verify[adapter_name]
            verified = verified_capabilities_from_run(run_dir, model_name)
        for m in CAPABILITY_METHODS:
            row[f"{m}_verified"] = verified[f"{m}_verified"] if verified else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def render_capability_matrix_markdown(df: pd.DataFrame) -> str:
    """Render as a Markdown table: ✅ verified, 🔶 declared-only (unverified), ❌ neither."""
    lines = ["| adapter | tier | " + " | ".join(CAPABILITY_METHODS) + " |",
             "|---" * (len(CAPABILITY_METHODS) + 2) + "|"]
    for _, row in df.iterrows():
        cells = [str(row["adapter"]), f'{row["tier"]} ({row["tier_name"]})']
        for m in CAPABILITY_METHODS:
            verified, declared = row[f"{m}_verified"], bool(row[f"{m}_declared"])
            # `verified` may come back as a numpy bool (not Python's `True`/
            # `False` singletons) once a DataFrame column has no NA rows to
            # keep it `object`-dtype, so compare by value via pd.isna/bool(),
            # never by `is` -- an earlier version of this used `is` and
            # silently misrendered exactly this all-verified/no-NA case.
            if pd.isna(verified):
                cells.append("🔶 declared, unverified" if declared else "❌")
            elif bool(verified):
                cells.append("✅ verified")
            else:
                cells.append("⚠️ declared, found nothing" if declared else "❌")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
