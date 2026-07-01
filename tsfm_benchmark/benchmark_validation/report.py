"""Assemble the validation report.

Collects the matching, feature, and diversity results into one JSON-serialisable
record and prints a short human summary. The similarity matrix itself is omitted
from the JSON by default because it is O(n^2); the bucketed profile and the
flagged redundant pairs carry the actionable signal.
"""

from __future__ import annotations

import json
from typing import Any

from .diversity import DiversityReport
from .features import FeatureMatrix
from .matching import MatchReport


def build_report(match: MatchReport, fm: FeatureMatrix, diversity: DiversityReport, embed_method: str) -> dict[str, Any]:
    """Combine all stage outputs into one serialisable dictionary."""
    return {
        "n_sequences": diversity.n_sequences,
        "matching": {
            "method": match.method,
            "redundancy_fraction": round(match.redundancy_fraction, 5),
            "n_redundant_pairs": len(match.redundant_pairs),
            "top_redundant_pairs": match.redundant_pairs[:10],
            "bucket_edges": match.bucket_edges,
            "bucket_counts": match.bucket_counts,
        },
        "features": {
            "set": "catch24" if len(fm.feature_names) == 24 else "catch22",
            "n_features": len(fm.feature_names),
            "n_sequences_imputed": fm.n_imputed,
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


def save_report(report: dict[str, Any], path: str) -> str:
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    return path


def print_summary(report: dict[str, Any]) -> None:
    m, f, d = report["matching"], report["features"], report["diversity"]
    print(f"sequences            : {report['n_sequences']}")
    print(f"feature set          : {f['set']} ({f['n_features']} features), imputed {f['n_sequences_imputed']}")
    print(f"match method         : {m['method']}")
    print(f"redundancy fraction  : {m['redundancy_fraction']}  ({m['n_redundant_pairs']} pairs >= threshold)")
    print(f"effective dimensions : {d['effective_dimensionality']} of {f['n_features']}")
    print(f"NN dist mean/min/p05 : {d['nn_distance_mean']} / {d['nn_distance_min']} / {d['nn_distance_p05']}")
    print(f"near-collision frac  : {d['near_collision_fraction']}")
    print(f"embedding            : {report['embedding_method']}")
    if m["top_redundant_pairs"]:
        print("top redundant pairs  :")
        for a, b, s in m["top_redundant_pairs"][:5]:
            print(f"   {a}  ~  {b}   sim={s:.4f}")
