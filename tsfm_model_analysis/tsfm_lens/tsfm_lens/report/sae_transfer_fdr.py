"""ROADMAP.md sec 37 P3 -- rendering for the FDR-controlled halves of
`sae/transfer.py`: the per-target transfer test's `p`/`p_bh`/`reciprocal_fdr`
keys (additive to `sae/transfer.json`, item 4) and the new, atlas-level
`run_atlas_transfer` (`sae/atlas_transfer.json`, item 3).

One public entry point, `transfer_fdr_block(run_dir, cfg) -> str`, called
right after `sae_concept_atlas.py::atlas_block` (`sae_concepts.py`'s own
concept-atlas section) since both read the same `sae/concept_atlas.json`
concept identities and this module answers the question that section's own
docstring poses but does not measure: two features can share a concept
(same causal EFFECT profile) while firing on different series, or transfer
(same INPUT selectivity) while sitting in different concepts -- "effect
space" and "input space" are independent claims, and the 2x2 cross-check
below is the first place in the report that puts a number on how often they
actually agree.

Two small pieces, matching the artifact's own two questions:
  1. one row per ordered (source model, destination model) pair --
     `sae/atlas_transfer.json`'s own `pair_summary`, already a pure
     reduction over its `tests` list, rendered as-is (no recomputation).
  2. the 2x2 cross-check (`sae/atlas_transfer.json::cross_check_2x2`),
     over every (concept, ordered model pair) cell that was actually
     TESTED -- an untested cell contributes to neither axis, per
     `CLAUDE.md` sec 11.37's three-states discipline (never conflate
     "untested" with "No").

No model name, architecture family, or `cfg.models[i]` index appears
anywhere in this module -- every label is read from the artifact's own
`src_model`/`dst_model` strings, mirroring `report/derived.py::
bottom_line_rows`'s own adaptivity contract (`tests/test_concept_atlas.py`'s
`_stripped_source`/`_BANNED_NAMES` pattern is reused directly by this
module's own test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .derived import load_json_or_none

__all__ = ["transfer_fdr_data", "transfer_fdr_block"]


def transfer_fdr_data(run_dir: Path) -> Optional[dict]:
    """`-> {"transfer": {...} | None, "atlas_transfer": {...} | None}` or
    `None` when NEITHER `sae/transfer.json` nor `sae/atlas_transfer.json`
    exists -- a pure loader, no recomputation; every field rendered below is
    read straight off one of the two artifacts."""
    run_dir = Path(run_dir)
    transfer = load_json_or_none(run_dir / "sae" / "transfer.json")
    atlas_transfer = load_json_or_none(run_dir / "sae" / "atlas_transfer.json")
    if transfer is None and atlas_transfer is None:
        return None
    return {"transfer": transfer, "atlas_transfer": atlas_transfer}


def _per_target_pair_table(transfer: dict) -> pd.DataFrame:
    """One row per ordered (src_model, dst_model) pair over
    `sae/transfer.json::pairs` -- tests, uncorrected reciprocal count, and
    FDR-reciprocal count, the per-target sibling of
    `sae/atlas_transfer.json::pair_summary`."""
    pairs = transfer.get("pairs") or []
    by_pair: dict = {}
    for p in pairs:
        key = (p.get("src_model"), p.get("dst_model"))
        rec = by_pair.setdefault(key, {"src_model": key[0], "dst_model": key[1],
                                       "n_tests": 0, "n_uncorrected_reciprocal": 0,
                                       "n_fdr_reciprocal": 0})
        rec["n_tests"] += 1
        rec["n_uncorrected_reciprocal"] += int(bool(p.get("reciprocal")))
        rec["n_fdr_reciprocal"] += int(bool(p.get("reciprocal_fdr")))
    rows = [by_pair[k] for k in sorted(by_pair)]
    return pd.DataFrame(rows)


def _pair_summary_frame(pair_summary: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "source model": r["src_model"], "destination model": r["dst_model"],
        "tests": r["n_tests"], "uncorrected reciprocal": r["n_uncorrected_reciprocal"],
        "FDR reciprocal": r["n_fdr_reciprocal"],
    } for r in sorted(pair_summary, key=lambda r: (r["src_model"], r["dst_model"]))])


def _cross_check_table(cross_check: dict) -> pd.DataFrame:
    labels = {"effect_yes_input_yes": "spans effect space, transfers (input space)",
             "effect_yes_input_no": "spans effect space, does not transfer",
             "effect_no_input_yes": "does not span effect space, transfers",
             "effect_no_input_no": "does not span effect space, does not transfer"}
    return pd.DataFrame([{"cell": labels[k], "count": int(cross_check.get(k, 0))}
                         for k in labels])


def transfer_fdr_block(run_dir: Path, cfg) -> str:
    """The FDR-controlled transfer tables, or `""` when neither artifact
    exists. Renders (a) the per-target `run_transfer` pair table with its
    new FDR column, (b) the atlas-level `run_atlas_transfer` pair table, and
    (c) the 2x2 effect-space-vs-input-space cross-check."""
    # Lazy, per `sae_concept_atlas.py`'s own import-cycle note (CLAUDE.md
    # sec 11.52).
    from .report import _frag, _note, _table

    data = transfer_fdr_data(Path(run_dir))
    if data is None:
        return ""

    transfer = data["transfer"]
    atlas_transfer = data["atlas_transfer"]
    inner = "<h5>Cross-model transfer — significance and FDR control</h5>"

    if transfer and transfer.get("pairs"):
        tbl = _per_target_pair_table(transfer)
        tbl = tbl.rename(columns={"src_model": "source model", "dst_model": "destination model",
                                  "n_tests": "tests", "n_uncorrected_reciprocal": "uncorrected reciprocal",
                                  "n_fdr_reciprocal": "FDR reciprocal"})
        inner += "<h6>Per-target concept transfer, by ordered model pair</h6>"
        inner += _table(tbl)
        inner += _note(
            "For each ordered (source, destination) model pair, how many "
            "per-target concept transfer tests ran, how many cleared the raw "
            "p95 null in both directions (\"uncorrected reciprocal\"), and how "
            "many still clear once each leg is Benjamini-Hochberg corrected "
            f"within its own (pair, leg) family at q={transfer.get('fdr_q')} "
            "(\"FDR reciprocal\").",
            "A test's exact permutation p is floored at "
            f"1/{int(transfer.get('n_null_draws', 0)) + 1}; where every null "
            f"draw fell short of the observed statistic, `p_method="
            f"{transfer.get('p_method')!r}` extrapolates a finer p instead of "
            "reporting that same floor for every such test, which is what "
            "would otherwise make BH unable to separate them.",
            "FDR reciprocal is always <= uncorrected reciprocal: BH can only "
            "remove survivors relative to the raw p95 comparison, never add "
            "them. Read this table beside the multiplicity ledger, which "
            "records each (pair, leg) as its own correction family rather "
            "than pooling them.")

    if atlas_transfer and atlas_transfer.get("pair_summary"):
        inner += "<h6>Concept-atlas transfer, by ordered model pair</h6>"
        inner += _table(_pair_summary_frame(atlas_transfer["pair_summary"]))
        inner += _note(
            "The same transfer test, over the cross-model ATLAS's own "
            "concepts instead of each target's own per-target clusters: for "
            "every atlas concept's part at one model, does another model's "
            "dictionary select the same top-firing series.",
            "Built from the identical draw-building and scoring path as the "
            "per-target table above (`sae/transfer.py::transfer_one`), just "
            "over a different source unit -- an atlas concept's part at one "
            "target, rather than that target's own concept.",
            "A concept with only one model's worth of members contributes no "
            "row here at all (there is no other model to test against); read "
            "`sae/atlas_transfer.json::concept_summary` for which concepts "
            "were skipped for that reason.")

        cross_check = atlas_transfer.get("cross_check_2x2") or {}
        total = sum(int(v) for v in cross_check.values())
        if total:
            inner += "<h6>Effect space vs. input space</h6>"
            inner += _table(_cross_check_table(cross_check))
            yes_yes = int(cross_check.get("effect_yes_input_yes", 0))
            yes_no = int(cross_check.get("effect_yes_input_no", 0))
            effect_yes_total = yes_yes + yes_no
            inner += _note(
                "For every (atlas concept, ordered model pair) cell that was "
                "actually tested: does the concept span both models in "
                "EFFECT space (both hold >=1 member of the same causal-"
                "profile concept), and does it also transfer in INPUT space "
                "(FDR-reciprocal at some tested target pair)?",
                (f"Of the {effect_yes_total} cell(s) that span both models in "
                 f"effect space, {yes_yes} also transfer in input space and "
                 f"{yes_no} do not -- a concept's shared causal profile does "
                 "not by itself guarantee its members were selected by "
                 "similar inputs." if effect_yes_total else
                 "No tested cell spans both models in effect space on this "
                 "run, so this cross-check has nothing to compare yet."),
                "Every cell here was TESTED; a (concept, pair) combination "
                "with no persisted features at one of the two models "
                "contributes to neither axis rather than counting as a "
                "measured \"No\" (CLAUDE.md sec 11.37) -- so these four "
                "counts do not sum to every concept times every pair, only "
                "to every pair actually examined.")

    return inner
