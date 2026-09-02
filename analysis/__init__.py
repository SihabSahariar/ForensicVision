"""Heuristic degradation analysis.

Every score produced here is a *heuristic indicator* derived from classical
image statistics, not the output of a validated classification model. See
:data:`app.constants.HEURISTIC_DISCLAIMER`.
"""

from analysis.analyzer import (
    AnalysisReport,
    DegradationAnalyzer,
    analyze_array,
    analyze_image,
)
from analysis.base import ANALYZER_VERSION, MetricResult

__all__ = [
    "AnalysisReport",
    "DegradationAnalyzer",
    "MetricResult",
    "analyze_image",
    "analyze_array",
    "ANALYZER_VERSION",
]
