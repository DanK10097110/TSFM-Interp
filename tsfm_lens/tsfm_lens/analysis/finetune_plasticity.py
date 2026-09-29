"""Fine-tuning plasticity: which blocks move, and does that match where the
interpretability metrics say the task-relevant structure lives (ROADMAP.md
sec 22.5 H6).

`experiments/finetune_child.py` produces fine-tuned children of
`amazon/chronos-t5-small` (`ROADMAP.md` sec 6.3.1 Option E) as a by-product of
an unrelated provenance-detection experiment. For free, this module asks a
second question of the same artifact: per-block weight-delta norm against the
parent checkpoint is a training signal completely independent of every
activation-based method elsewhere in this repo (CKA, the layer-screen
bake-off's SAE-mass gold ranking, decodability probes) -- no forward pass, no
benchmark corpus, just two state dicts. If the blocks that moved most during
fine-tuning are also the blocks `layer_screen`'s `work_bend` selector (or the
bake-off's own gold ranking) already flagged as task-relevant, that is a rare
cross-validation of the whole layer-screening enterprise from a source that
could not have been circular with it.

Deliberately narrow in scope, matching this item's own ~0.5-session budget:
no new fine-tuning runs, no new corpus, no pipeline stage. Reads two already-
saved HF checkpoints and (optionally) an existing layer-screen bake-off JSON.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch


def block_weight_delta_norms(parent_state_dict: dict, child_state_dict: dict,
                             block_pattern: str = r"^(encoder|decoder)\.block\.(\d+)\.") -> dict:
    """Per-block L2 weight-delta norm between a parent checkpoint and a fine-tuned child.

    `block_pattern` must have the block index as its *last* capturing group;
    the stack name (`encoder`/`decoder` for a T5-family model) is read from
    the first group when present. Parameters whose name does not match
    `block_pattern` (embeddings, final layer norm, LM head, relative-position
    biases outside a block) are summed separately under `"other"` so the
    per-block totals are not silently missing mass -- their sum plus every
    per-block norm accounts for the whole state dict.

    Every parameter shared by both dicts must match in shape (a genuine
    architecture mismatch, e.g. comparing across checkpoint families, is a
    real error and raises rather than silently skipping the parameter). A
    name present in only one dict is skipped with a note in `"skipped"`,
    since a fine-tune that added/renamed a parameter (unlikely for a plain
    HF `from_pretrained`/`save_pretrained` round trip, but not this
    function's job to assume) should not silently understate the delta.

    `per_block` keys are `"{stack}.{idx}"` strings (e.g. `"encoder.3"`), not
    tuples -- the whole return value must round-trip through `json.dumps`
    unchanged, since this is a report-only tool whose artifact is a JSON file
    like every other standalone analysis module in this repo.
    """
    pattern = re.compile(block_pattern)
    per_block: dict = {}
    other = 0.0
    skipped = []
    for name, parent_tensor in parent_state_dict.items():
        if name not in child_state_dict:
            skipped.append(name)
            continue
        child_tensor = child_state_dict[name]
        if parent_tensor.shape != child_tensor.shape:
            raise ValueError(
                f"shape mismatch for '{name}': parent {tuple(parent_tensor.shape)} "
                f"vs child {tuple(child_tensor.shape)} -- these are not the same "
                f"architecture, weight-delta comparison is meaningless")
        delta_norm = torch.linalg.vector_norm(
            (child_tensor.float() - parent_tensor.float()).reshape(-1)).item()
        m = pattern.match(name)
        if m is None:
            other += delta_norm
            continue
        stack = m.group(1) if m.lastindex and m.lastindex >= 2 else "block"
        idx = int(m.group(m.lastindex))
        key = f"{stack}.{idx}"
        per_block[key] = per_block.get(key, 0.0) + delta_norm
    for name in child_state_dict:
        if name not in parent_state_dict:
            skipped.append(name)
    return {"per_block": per_block, "other": other, "skipped": skipped}


def block_norms_for_stack(delta: dict, stack: str) -> list:
    """`[norm_block_0, norm_block_1, ...]` for one stack, in block-index order.

    Raises if `stack` has no blocks in `delta["per_block"]` -- an empty
    result would otherwise silently look like "the stack exists and has
    zero plasticity" rather than "the stack was never captured."
    """
    entries = []
    for key, v in delta["per_block"].items():
        s, _, idx_str = key.rpartition(".")
        if s == stack:
            entries.append((int(idx_str), v))
    if not entries:
        raise ValueError(f"no blocks found for stack '{stack}' in this delta "
                         f"-- check block_pattern against this checkpoint's "
                         f"actual parameter names")
    entries.sort(key=lambda t: t[0])
    n = entries[-1][0] + 1
    out = [0.0] * n
    for idx, v in entries:
        out[idx] = v
    return out


def plasticity_vs_interpretability_rank_correlation(
        weight_deltas: list, interpretability_scores: list) -> dict:
    """Spearman correlation between per-block weight-delta norm and an
    independent interpretability score (e.g. `layer_screen`'s `work_bend`
    `score_per_layer`, or the bake-off's SAE-mass `gold_score`) over the
    same block indices.

    Degrades to `rho: None` with a stated reason (CLAUDE.md sec 2.5) rather
    than raising or fabricating a coefficient when there are fewer than 2
    blocks -- too few points for a rank correlation to mean anything.
    """
    from scipy.stats import spearmanr

    n = min(len(weight_deltas), len(interpretability_scores))
    if n < 2:
        return {"rho": None, "n_blocks": n,
                "reason": "fewer than 2 blocks to correlate over"}
    a = np.asarray(weight_deltas[:n], dtype=float)
    b = np.asarray(interpretability_scores[:n], dtype=float)
    rho, pval = spearmanr(a, b)
    return {"rho": float(rho), "p_value": float(pval), "n_blocks": n}


def plasticity_report(parent_checkpoint: str, child_checkpoints: dict,
                      bakeoff_entry: Optional[dict] = None,
                      block_pattern: str = r"^(encoder|decoder)\.block\.(\d+)\.") -> dict:
    """End-to-end: load one parent + N children, compute per-block deltas for
    each, and (if `bakeoff_entry` is given -- one model's entry from a
    `layer_screen_bakeoff.py`-produced JSON) correlate the encoder-stack
    deltas against that entry's `gold_score` and every selector's
    `score_per_layer`.

    `child_checkpoints` maps a label (e.g. `"child_light"`) to a checkpoint
    path/id. Only `transformers` is imported here (deferred), so calling
    this from a context with no GPU/network beyond the local checkpoint
    paths already on disk works exactly like the rest of this repo's
    report-only tools (`ROADMAP.md` sec 6.1's "reads existing run artifacts,
    no model load" family) -- except this one genuinely does load two small
    CPU-resident state dicts, which is why it stays out of the pipeline
    proper (`CLAUDE.md` sec 2.8's short-vs-background split: this is short).
    """
    from transformers import T5ForConditionalGeneration

    parent_sd = T5ForConditionalGeneration.from_pretrained(parent_checkpoint).state_dict()
    children = {}
    for label, path in child_checkpoints.items():
        child_sd = T5ForConditionalGeneration.from_pretrained(path).state_dict()
        delta = block_weight_delta_norms(parent_sd, child_sd, block_pattern)
        entry = {"delta": delta,
                 "encoder_norms": block_norms_for_stack(delta, "encoder")}
        try:
            entry["decoder_norms"] = block_norms_for_stack(delta, "decoder")
        except ValueError:
            entry["decoder_norms"] = None
        if bakeoff_entry is not None:
            entry["vs_gold_score"] = plasticity_vs_interpretability_rank_correlation(
                entry["encoder_norms"], bakeoff_entry["gold_score"])
            entry["vs_selectors"] = {
                method: plasticity_vs_interpretability_rank_correlation(
                    entry["encoder_norms"], sel["score_per_layer"])
                for method, sel in bakeoff_entry.get("selections", {}).items()}
        children[label] = entry
    return {"parent": parent_checkpoint, "children": children}
