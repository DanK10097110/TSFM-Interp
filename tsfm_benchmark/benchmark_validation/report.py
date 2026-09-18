"""Assemble the validation report.

Collects the matching, feature, and diversity results into one JSON-serialisable
record and prints a short human summary. The similarity matrix itself is omitted
from the JSON by default because it is O(n^2); the bucketed profile and the
flagged redundant pairs carry the actionable signal.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .diversity import DiversityReport, InsufficientN
from .features import FeatureMatrix
from .loaders import SeqRecord
from .matching import MatchReport


def _composition(records: list[SeqRecord]) -> dict[str, Any]:
    """Actual counts realized in the corpus -- by tier, task, generator, and domain."""
    total = len(records) or 1
    tier_counts = Counter(r.tier for r in records)
    task_counts = Counter(r.task for r in records)
    group_counts = Counter(r.group for r in records)
    domain_counts: Counter[str] = Counter()
    for r in records:
        domain_counts.update(r.domains)

    return {
        "n_sequences": len(records),
        "by_tier": {k: {"count": v, "fraction": round(v / total, 4)} for k, v in tier_counts.most_common()},
        "by_task": {k: v for k, v in task_counts.most_common()},
        "by_generator": {k: v for k, v in group_counts.most_common()},
        "by_domain": dict(domain_counts.most_common()),
    }


def _by_group_summary(by_group: dict[str, Any]) -> dict[str, Any]:
    """Per-group headline numbers, plus which feature(s) actually drive each group's variance.

    ``diversity_metrics_by_group`` scales every group against one scaler fit
    on the whole corpus (so groups stay comparable), which has a real
    consequence worth surfacing directly: if one feature is nearly constant
    across most of the corpus (tiny global IQR) but a minority group
    legitimately varies on it, that single feature can dominate the group's
    scaled variance and make it look collapsed in every *other* dimension by
    comparison. ``top_variance_feature`` names the feature so that's
    diagnosable instead of just a suspicious-looking number.

    A group below the minimum-n threshold (sec 15 A17) is an ``InsufficientN``
    entry, not a ``DiversityReport`` -- surfaced explicitly with its own
    status rather than being coerced into the same numeric shape (which
    would either crash on the missing attributes or, worse, silently print
    as zeros).
    """
    out: dict[str, Any] = {}
    for g, d in by_group.items():
        if isinstance(d, InsufficientN):
            out[g] = {"status": d.status, "n_sequences": d.n_sequences,
                      "min_required": d.min_required}
            continue
        out[g] = {
            "n_sequences": d.n_sequences,
            "effective_dimensionality": d.effective_dimensionality,
            "effective_dimensionality_ci": d.effective_dimensionality_ci,
            "total_variance": d.total_variance,
            "nn_distance_mean": d.nn_distance_mean,
            "near_collision_fraction": d.near_collision_fraction,
            "near_collision_fraction_ci": d.near_collision_fraction_ci,
            "top_variance_feature": d.feature_variance_ranking[0] if d.feature_variance_ranking else None,
        }
    return out


def build_report(
    match: MatchReport,
    fm: FeatureMatrix,
    diversity: DiversityReport,
    embed_method: str,
    records: list[SeqRecord] | None = None,
    diversity_by_group: dict[str, Any] | None = None,
    diversity_by_group_key: str | None = None,
    redundancy_by_group_result: dict[str, Any] | None = None,
    corpus_digest: str | None = None,
    split_role: str | None = None,
) -> dict[str, Any]:
    """Combine all stage outputs into one serialisable dictionary.

    ``records`` (the same list passed to matching/feature extraction) is
    optional only for backward compatibility; passing it fills in the
    ``composition`` section with the ratios actually realized in the corpus,
    which is the only place in this report that reflects the pipeline's
    build-time labels (tier/task/generator/domain) rather than a downstream
    metric derived from the raw values. ``diversity_by_group`` (see
    ``diversity_metrics_by_group``) and ``redundancy_by_group_result`` (see
    ``redundancy_by_group``) are the per-group counterparts of the global
    ``diversity``/``matching`` numbers -- pass them to surface a subgroup that
    has collapsed even though the corpus-wide numbers look fine. ``corpus_digest``
    (ROADMAP.md sec 34 item C1.3) is the sealed corpus's own manifest
    ``global_digest``, when known -- carried through so a downstream
    ``tsfm_lens`` corpus card can refuse to pair this report with a
    different corpus that happens to share its file path. ``None`` for a
    corpus with no sealed manifest (e.g. ``--demo`` mode); a report with no
    digest is not proof of tampering, only that nothing could be checked.

    ``split_role`` (ROADMAP.md sec 34 item B2.1) -- pass ``"private"`` when
    validating the sealed private corpus. This records ``model_free: true``
    (a statement, not a new computation -- see ``cross_split.py``'s module
    docstring for the argument that nothing in this package ever involves a
    model) and redacts the one field in this report that would otherwise
    carry private sample ids: ``matching.top_redundant_pairs``. Every other
    field here is already an aggregate statistic (a fraction, a count, a
    bucketed histogram) and needs no redaction. ``"public"``/``None``/
    anything else leaves the report exactly as it has always been --
    existing callers are unaffected.
    """
    report: dict[str, Any] = {
        "n_sequences": diversity.n_sequences,
        "corpus_digest": corpus_digest,
        "matching": {
            "method": match.method,
            "redundancy_fraction": round(match.redundancy_fraction, 5),
            "n_redundant_pairs": len(match.redundant_pairs),
            "top_redundant_pairs": match.redundant_pairs[:10],
            "bucket_edges": match.bucket_edges,
            "bucket_counts": match.bucket_counts,
            "blocked": match.blocked,
            "n_blocks": match.n_blocks,
            "coverage_fraction": match.coverage_fraction,
        },
        "features": {
            "set": "catch24" if len(fm.feature_names) == 24 else "catch22",
            "n_features": len(fm.feature_names),
            "n_sequences_imputed": fm.n_imputed,
            "n_values_winsorized": fm.n_clipped,
        },
        "diversity": {
            "effective_dimensionality": diversity.effective_dimensionality,
            "total_variance": diversity.total_variance,
            "nn_distance_mean": diversity.nn_distance_mean,
            "nn_distance_min": diversity.nn_distance_min,
            "nn_distance_p05": diversity.nn_distance_p05,
            "near_collision_fraction": diversity.near_collision_fraction,
            "top_varying_features": diversity.feature_variance_ranking[:8],
        },
        "embedding_method": embed_method,
    }
    if records is not None:
        report["composition"] = _composition(records)
    if diversity_by_group is not None:
        report["diversity_by_group"] = {"by": diversity_by_group_key or "unknown", "groups": _by_group_summary(diversity_by_group)}
    if redundancy_by_group_result is not None:
        report["redundancy_by_group"] = redundancy_by_group_result
    if split_role == "private":
        report["split_role"] = "private"
        report["model_free"] = True
        report["per_sample_identifiers_omitted"] = True
        report["matching"]["top_redundant_pairs"] = []
    return report


def save_report(report: dict[str, Any], path: str) -> str:
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    return path


def print_summary(report: dict[str, Any]) -> None:
    m, f, d = report["matching"], report["features"], report["diversity"]
    if "composition" in report:
        c = report["composition"]
        tier_str = ", ".join(f"{k}={v['count']} ({v['fraction']:.1%})" for k, v in c["by_tier"].items())
        print(f"composition by tier  : {tier_str}")
        print(f"composition by task  : {c['by_task']}")
        if c["by_domain"]:
            print(f"real domains used    : {c['by_domain']}")
    print(f"sequences            : {report['n_sequences']}")
    print(f"feature set          : {f['set']} ({f['n_features']} features), imputed {f['n_sequences_imputed']}, winsorized {f.get('n_values_winsorized', 0)} values")
    print(f"match method         : {m['method']}" + (f" (blocked, {m['n_blocks']} shape-clusters, "
                                                       f"coverage={m['coverage_fraction']:.1%} of all pairs)"
                                                       if m.get("blocked") else ""))
    print(f"redundancy fraction  : {m['redundancy_fraction']}  ({m['n_redundant_pairs']} pairs >= threshold"
         + (", among scored pairs only" if m.get("blocked") else "") + ")")
    print(f"effective dimensions : {d['effective_dimensionality']} of {f['n_features']}")
    print(f"NN dist mean/min/p05 : {d['nn_distance_mean']} / {d['nn_distance_min']} / {d['nn_distance_p05']}")
    print(f"near-collision frac  : {d['near_collision_fraction']}")
    print(f"embedding            : {report['embedding_method']}")
    if m["top_redundant_pairs"]:
        print("top redundant pairs  :")
        for a, b, s in m["top_redundant_pairs"][:5]:
            print(f"   {a}  ~  {b}   sim={s:.4f}")
    if "diversity_by_group" in report:
        dg = report["diversity_by_group"]
        print(f"diversity by {dg['by']:<8} : (effective_dim [CI] / total_var / near_collision_frac, n, top variance feature)")
        insufficient = {g: v for g, v in dg["groups"].items() if v.get("status") == "insufficient_n"}
        sufficient = {g: v for g, v in dg["groups"].items() if v.get("status") != "insufficient_n"}
        for g, v in sorted(sufficient.items(), key=lambda kv: kv[1]["effective_dimensionality"]):
            top_feat = f"{v['top_variance_feature'][0]}={v['top_variance_feature'][1]:.1f}" if v["top_variance_feature"] else "-"
            eff_ci = f" {v['effective_dimensionality_ci']}" if v.get("effective_dimensionality_ci") else ""
            print(f"   {g:<32} {v['effective_dimensionality']:>6.3f}{eff_ci} / {v['total_variance']:>10.3f} / {v['near_collision_fraction']:.3f}   (n={v['n_sequences']:<4d} top={top_feat})")
        for g, v in sorted(insufficient.items()):
            print(f"   {g:<32} insufficient n={v['n_sequences']} (< {v['min_required']} required) -- no metrics reported")
    if "redundancy_by_group" in report:
        rg = report["redundancy_by_group"]
        print(f"within-group similarity : {rg['within_group_mean_similarity']}")
        print(f"across-group similarity : {rg['across_group_mean_similarity']}")
