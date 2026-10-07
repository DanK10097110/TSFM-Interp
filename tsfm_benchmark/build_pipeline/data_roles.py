"""The corpus data-roles card (ROADMAP sec 41, V3-A): what each kind of series is for.

A reader of a result needs to know, per series, whether it carries ground truth (synthetic),
inherits a real source's distribution (real_derived) or is raw external data (external_real),
and what that role may and may not be used for. This module computes that from the sealed
splits themselves, never from the config, so the card cannot drift from the corpus: it counts
each role per split, lists the real sources behind them, and refuses to write a card when a
sample's role is missing, unknown, or contradicts its generator. The ``used for`` / ``never
used for`` text is a fixed table taken from ROADMAP sec 41 and is the same for every corpus.

``sae_augment`` and ``leakage_reference`` are roles of data *around* the corpus (SAE training
rows and the leakage audit's reference set); they are not stored in it, so the card lists them
with the count the manifests can attest to (the audit's reference series) or ``None``.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any

from .schema import REAL_DERIVED_GENERATORS, ROLES, SYNTHETIC_GENERATORS, TimeSeriesSample
from .seal import load_sealed

ROLE_USE: dict[str, dict[str, str]] = {
    "synthetic": {
        "what": "parametric / random_parametric generators; touches no real data, exact ground truth",
        "used_for": "causal discovery, ground-truth naming, every registered claim",
        "never_used_for": "nothing excluded",
    },
    "real_derived": {
        "what": "mixture / block_bootstrap / sequential_par over real source series; inherits the source distribution",
        "used_for": "realism, per-tier replication of claims",
        "never_used_for": "ground-truth naming",
    },
    "external_real": {
        "what": "raw windows cut unchanged from an external benchmark (GIFT-Eval); no generator, no ground truth",
        "used_for": "one external-validity leg of confirm, reported separately",
        "never_used_for": "discovery, registration, SAE training",
    },
}
NON_SAMPLE_USE: dict[str, dict[str, str]] = {
    "sae_augment": {
        "what": "raw Monash windows (sae.real_data_*), outside this corpus",
        "used_for": "SAE training rows only",
        "never_used_for": "any statistic",
    },
    "leakage_reference": {
        "what": "the real reference series the leakage audit compares against (Monash archive)",
        "used_for": "leakage audit only",
        "never_used_for": "anything else",
    },
}


def role_violations(samples: list[TimeSeriesSample]) -> list[str]:
    """Samples whose role is missing, unknown, or inconsistent with their generator."""
    bad: list[str] = []
    for s in samples:
        gen = s.provenance.generator
        if s.role not in ROLES:
            bad.append(f"{s.sample_id}: role {s.role!r} is not one of {ROLES}")
        elif s.role == "synthetic" and gen not in SYNTHETIC_GENERATORS:
            bad.append(f"{s.sample_id}: role synthetic but generator {gen}")
        elif s.role == "real_derived" and gen not in REAL_DERIVED_GENERATORS:
            bad.append(f"{s.sample_id}: role real_derived but generator {gen}")
        elif s.role == "external_real" and gen != "gifteval_window":
            bad.append(f"{s.sample_id}: role external_real but generator {gen}")
        elif gen == "gifteval_window" and s.role != "external_real":
            bad.append(f"{s.sample_id}: gifteval_window with role {s.role}")
    return bad


def _role_block(samples: list[TimeSeriesSample]) -> dict[str, Any]:
    sources: Counter = Counter()
    generators: Counter = Counter()
    tasks: Counter = Counter()
    for s in samples:
        generators[s.provenance.generator] += 1
        tasks[s.provenance.generator_params.get("task_name", "unknown")] += 1
        for corpus in {r.corpus for r in s.provenance.source_refs}:
            sources[corpus] += 1
    lengths = [len(s.values) for s in samples]
    return {"count": len(samples), "generators": dict(sorted(generators.items())), "tasks": dict(sorted(tasks.items())),
            "sources": dict(sorted(sources.items())), "length_min": min(lengths), "length_max": max(lengths)}


def build_roles_card(splits: dict[str, str]) -> dict[str, Any]:
    """Compute the card from sealed split directories ``{split_name: directory}``.

    Every split is loaded with ``load_sealed(verify=True)``, so a tampered corpus fails here, and
    every sample must pass ``role_violations``; otherwise a ``ValueError`` lists the offenders.
    """
    card: dict[str, Any] = {"schema_version": 1, "splits": {}, "roles": {}, "non_sample_roles": {}}
    totals: dict[str, list[TimeSeriesSample]] = defaultdict(list)
    references = None
    for split, directory in splits.items():
        samples, manifest = load_sealed(directory, verify=True)
        bad = role_violations(samples)
        if bad:
            raise ValueError(f"split '{split}' has {len(bad)} role violations, first: {bad[:3]}")
        by_role: dict[str, list[TimeSeriesSample]] = defaultdict(list)
        for s in samples:
            by_role[s.role].append(s)
            totals[s.role].append(s)
        card["splits"][split] = {"n_samples": len(samples), "global_digest": manifest["global_digest"],
                                 "by_role": {r: _role_block(v) for r, v in sorted(by_role.items())}}
        gate = (manifest.get("extra", {}).get("audit", {}) or {}).get("gate")
        if gate and references is None and gate.get("gate_effective"):
            references = {"references": gate.get("references"), "n_series": gate.get("reference_n_series")}
    for role in ROLES:
        if role in totals:
            card["roles"][role] = {**ROLE_USE[role], **_role_block(totals[role])}
        else:
            card["roles"][role] = {**ROLE_USE[role], "count": 0}
    card["non_sample_roles"] = {
        "sae_augment": {**NON_SAMPLE_USE["sae_augment"], "count": None},
        "leakage_reference": {**NON_SAMPLE_USE["leakage_reference"], "count": (references or {}).get("n_series"),
                              "references": (references or {}).get("references")}}
    return card


def render_markdown(card: dict[str, Any]) -> str:
    """Markdown rendering of ``build_roles_card`` output, one table per concern."""
    lines = ["# Corpus data roles", "",
             "Every series carries exactly one role. Counts are read from the sealed splits.", "",
             "| Role | Count | What it is | Used for | Never used for |", "|---|---|---|---|---|"]
    for role, block in card["roles"].items():
        lines.append(f"| `{role}` | {block['count']} | {block['what']} | {block['used_for']} | {block['never_used_for']} |")
    for role, block in card["non_sample_roles"].items():
        count = block["count"] if block["count"] is not None else "not stored here"
        lines.append(f"| `{role}` | {count} | {block['what']} | {block['used_for']} | {block['never_used_for']} |")
    lines += ["", "## Per split", "", "| Split | Role | Count | Length min-max |", "|---|---|---|---|"]
    for split, sblock in card["splits"].items():
        for role, b in sblock["by_role"].items():
            lines.append(f"| {split} | `{role}` | {b['count']} | {b['length_min']}-{b['length_max']} |")
    lines += ["", "## Sources per role", ""]
    for role, block in card["roles"].items():
        if block.get("sources"):
            lines += [f"### `{role}`", ""] + [f"- {corpus}: {n}" for corpus, n in block["sources"].items()] + [""]
        elif block["count"]:
            lines += [f"### `{role}`", "", "- no real source (generated from parameters)", ""]
    return "\n".join(lines) + "\n"


def write_roles_card(splits: dict[str, str], out_dir: str) -> dict[str, Any]:
    """Write ``data_roles.json`` and ``data_roles.md`` into ``out_dir`` and return the card."""
    card = build_roles_card(splits)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data_roles.json"), "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    with open(os.path.join(out_dir, "data_roles.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(card))
    return card
