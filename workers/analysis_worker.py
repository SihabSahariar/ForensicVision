"""Degradation-analysis worker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from analysis.analyzer import AnalysisReport, DegradationAnalyzer
from core.image_io import ImageData
from workers.base import BaseWorker

logger = logging.getLogger(__name__)

__all__ = ["AnalysisWorker"]


class AnalysisWorker(BaseWorker):
    """Runs the degradation analyzer off the GUI thread."""

    description = "Analysing image"

    def __init__(
        self,
        image: ImageData,
        source_path: Optional[Path] = None,
        roi: Optional[Dict[str, Any]] = None,
        analyzer: Optional[DegradationAnalyzer] = None,
    ) -> None:
        """Create the worker.

        Args:
            image: Image to analyse.
            source_path: Original path, used for container-level evidence.
            roi: ROI descriptor when analysing a region.
            analyzer: Injected analyzer; a default is created when omitted.
        """
        super().__init__()
        self._image = image
        self._source_path = source_path
        self._roi = roi
        self._analyzer = analyzer or DegradationAnalyzer()

    def execute(self) -> AnalysisReport:
        """Run every indicator and return the report."""
        self.report_status("Running degradation analysis...")
        return self._analyzer.analyze(
            self._image,
            source_path=self._source_path,
            roi=self._roi,
            progress=self.report,
            cancelled=self.is_cancelled,
        )
