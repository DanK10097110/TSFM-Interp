"""Validation helpers for benchmark analysis and reporting."""

from .diversity import DiversityReport, diversity_metrics
from .embedding import embed_3d
from .features import FeatureMatrix, extract_features
from .loaders import SeqRecord, from_arrays, from_samples, from_sealed
from .matching import MatchReport, match_all
from .plot import plot_embedding
from .report import build_report, print_summary, save_report

__all__ = [
    "DiversityReport",
    "FeatureMatrix",
    "MatchReport",
    "SeqRecord",
    "build_report",
    "diversity_metrics",
    "embed_3d",
    "extract_features",
    "from_arrays",
    "from_samples",
    "from_sealed",
    "match_all",
    "plot_embedding",
    "print_summary",
    "save_report",
]
