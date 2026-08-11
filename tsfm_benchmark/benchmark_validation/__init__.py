"""Validation helpers for benchmark analysis and reporting."""

from .diversity import DiversityReport, InsufficientN, diversity_metrics, diversity_metrics_by_group
from .embedding import embed_3d
from .features import FeatureMatrix, extract_features
from .gates import DEFAULT_THRESHOLDS, GateThresholds, check_diversity_gates, print_gate_summary
from .loaders import SeqRecord, from_arrays, from_samples, from_sealed
from .matching import MatchReport, match_all, redundancy_by_group
from .plot import (
    plot_diversity_by_group,
    plot_domain_composition,
    plot_embedding,
    plot_example_sequences,
    plot_feature_anomalies,
    plot_feature_variance,
    plot_group_composition,
    plot_redundancy_histogram,
    plot_top_redundant_pairs,
    plot_value_length_distribution,
)
from .report import build_report, print_summary, save_report

__all__ = [
    "DiversityReport",
    "InsufficientN",
    "FeatureMatrix",
    "MatchReport",
    "SeqRecord",
    "DEFAULT_THRESHOLDS",
    "GateThresholds",
    "build_report",
    "check_diversity_gates",
    "diversity_metrics",
    "diversity_metrics_by_group",
    "embed_3d",
    "extract_features",
    "from_arrays",
    "from_samples",
    "from_sealed",
    "match_all",
    "plot_diversity_by_group",
    "plot_domain_composition",
    "plot_embedding",
    "plot_example_sequences",
    "plot_feature_anomalies",
    "plot_feature_variance",
    "plot_group_composition",
    "plot_redundancy_histogram",
    "plot_top_redundant_pairs",
    "plot_value_length_distribution",
    "print_gate_summary",
    "print_summary",
    "redundancy_by_group",
    "save_report",
]
