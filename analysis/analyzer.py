"""Degradation analysis orchestrator.

Runs every registered indicator and assembles a :class:`AnalysisReport`. All
results are explicitly labelled as heuristic indicators - none of them is the
output of a validated classifier, and the UI and PDF report repeat that
qualifier wherever the numbers appear.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from analysis.base import ANALYZER_VERSION, MetricResult
from analysis.blur import analyze_blur, analyze_motion_blur
from analysis.contrast import analyze_contrast
from analysis.exposure import analyze_exposure
from analysis.haze import analyze_haze
from analysis.jpeg import analyze_jpeg
from analysis.noise import analyze_noise
from analysis.resolution import analyze_resolution
from app.constants import (
    DEGRADATION_ACTION_THRESHOLD,
    DEGRADATION_ORDER,
    HEURISTIC_DISCLAIMER,
    DegradationKey,
)
from core.image_io import ImageData

logger = logging.getLogger(__name__)

__all__ = ["AnalysisReport", "DegradationAnalyzer", "analyze_image"]

ProgressCallback = Callable[[int, str], None]


@dataclass
class AnalysisReport:
    """The complete set of degradation indicators for one image.

    Attributes:
        metrics: Results keyed by :class:`app.constants.DegradationKey`.
        image_shape: ``(height, width, channels)`` of the analysed image.
        source_path: Path of the analysed file, when known.
        roi: ROI descriptor when the analysis was region-limited.
        duration_s: Wall-clock analysis time.
        created_at: ISO-8601 UTC timestamp.
        analyzer_version: Version of the scoring formulas used.
    """

    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    image_shape: tuple = (0, 0, 0)
    source_path: Optional[str] = None
    roi: Optional[Dict[str, Any]] = None
    duration_s: float = 0.0
    created_at: str = ""
    analyzer_version: str = ANALYZER_VERSION

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------- accessors
    def score(self, key: str) -> float:
        """Return the normalised score for ``key``, or 0 when absent."""
        metric = self.metrics.get(key)
        return metric.score if metric is not None else 0.0

    def get(self, key: str) -> Optional[MetricResult]:
        """Return the :class:`MetricResult` for ``key``."""
        return self.metrics.get(key)

    def scores(self) -> Dict[str, float]:
        """Return ``{key: score}`` in canonical display order."""
        return {
            key: self.metrics[key].score
            for key in DEGRADATION_ORDER
            if key in self.metrics
        }

    def ordered(self) -> List[MetricResult]:
        """Return metrics in canonical display order."""
        return [self.metrics[key] for key in DEGRADATION_ORDER if key in self.metrics]

    def actionable(
        self, threshold: float = DEGRADATION_ACTION_THRESHOLD
    ) -> List[MetricResult]:
        """Return metrics above ``threshold``, most severe first."""
        candidates = [m for m in self.metrics.values() if m.score >= threshold]
        return sorted(candidates, key=lambda m: m.score, reverse=True)

    @property
    def dominant(self) -> Optional[MetricResult]:
        """The single most severe indicator, if any were computed."""
        if not self.metrics:
            return None
        return max(self.metrics.values(), key=lambda m: m.score)

    def summary_line(self) -> str:
        """Return a one-line summary for the status bar."""
        actionable = self.actionable()
        if not actionable:
            return "No degradation indicator exceeded the action threshold."
        parts = [f"{m.label} {m.percent}" for m in actionable[:4]]
        return "Dominant indicators: " + ", ".join(parts)

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping including the disclaimer."""
        return {
            "schema": "forensicvision.analysis/1",
            "analyzer_version": self.analyzer_version,
            "created_at": self.created_at,
            "duration_s": round(self.duration_s, 4),
            "image_shape": list(self.image_shape),
            "source_path": self.source_path,
            "roi": self.roi,
            "disclaimer": HEURISTIC_DISCLAIMER,
            "scores": {k: round(v, 4) for k, v in self.scores().items()},
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisReport":
        """Rebuild a report from :meth:`to_dict` output."""
        report = cls(
            image_shape=tuple(data.get("image_shape", (0, 0, 0))),
            source_path=data.get("source_path"),
            roi=data.get("roi"),
            duration_s=float(data.get("duration_s", 0.0)),
            created_at=data.get("created_at", ""),
            analyzer_version=data.get("analyzer_version", ANALYZER_VERSION),
        )
        for key, payload in (data.get("metrics") or {}).items():
            report.metrics[key] = MetricResult(
                key=payload.get("key", key),
                label=payload.get("label", key),
                score=float(payload.get("score", 0.0)),
                method=payload.get("method", ""),
                measurements=payload.get("measurements", {}),
                notes=list(payload.get("notes", [])),
                reference=payload.get("reference", ""),
            )
        return report


class DegradationAnalyzer:
    """Runs the individual indicators over an image.

    The analyzer is stateless and thread-safe; a single instance may be shared
    between the GUI and worker threads.
    """

    #: Weight of each indicator when computing an overall quality impression.
    OVERALL_WEIGHTS: Dict[str, float] = {
        DegradationKey.BLUR.value: 0.25,
        DegradationKey.NOISE.value: 0.18,
        DegradationKey.JPEG.value: 0.16,
        DegradationKey.LOW_RESOLUTION.value: 0.16,
        DegradationKey.LOW_CONTRAST.value: 0.09,
        DegradationKey.UNDEREXPOSURE.value: 0.06,
        DegradationKey.OVEREXPOSURE.value: 0.05,
        DegradationKey.HAZE.value: 0.05,
    }

    def analyze(
        self,
        image: ImageData,
        source_path: Optional[Path] = None,
        roi: Optional[Dict[str, Any]] = None,
        progress: Optional[ProgressCallback] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> AnalysisReport:
        """Analyse ``image`` and return a populated report.

        Args:
            image: Image to analyse.
            source_path: Original file path, used for container-level evidence
                such as JPEG quantisation tables. Defaults to
                ``image.source_path``.
            roi: ROI descriptor recorded in the report when the array passed in
                is a region crop.
            progress: Optional ``(percent, message)`` callback.
            cancelled: Optional predicate polled between indicators.

        Returns:
            The completed :class:`AnalysisReport`.
        """
        started = time.perf_counter()
        pixels = image.pixels
        path = source_path or image.source_path

        report = AnalysisReport(
            image_shape=image.shape,
            source_path=str(path) if path else None,
            roi=roi,
        )

        steps: List[tuple] = [
            ("Blur", lambda: [analyze_blur(pixels)]),
            ("Motion blur", lambda: [analyze_motion_blur(pixels)]),
            ("Noise", lambda: [analyze_noise(pixels)]),
            ("JPEG artefacts", lambda: [analyze_jpeg(pixels, path)]),
            ("Resolution", lambda: [analyze_resolution(pixels)]),
            ("Exposure", lambda: list(analyze_exposure(pixels))),
            ("Contrast", lambda: [analyze_contrast(pixels)]),
            ("Haze", lambda: [analyze_haze(pixels)]),
        ]

        total = len(steps)
        for index, (name, runner) in enumerate(steps):
            if cancelled is not None and cancelled():
                logger.info("Analysis cancelled after %d/%d steps", index, total)
                break
            if progress is not None:
                progress(int(index * 100 / total), f"Analysing: {name}")
            try:
                for result in runner():
                    report.metrics[result.key] = result
            except Exception:
                logger.exception("Indicator '%s' failed; continuing", name)

        if progress is not None:
            progress(100, "Analysis complete")

        report.duration_s = time.perf_counter() - started
        logger.info(
            "Analysis complete in %.2fs - %s",
            report.duration_s,
            report.summary_line(),
        )
        return report

    def overall_severity(self, report: AnalysisReport) -> float:
        """Return a single weighted severity figure in ``[0, 1]``.

        This is a convenience summary for sorting batches; it is deliberately
        not shown as a headline "quality score" in the UI, because a single
        number invites over-interpretation.
        """
        total_weight = 0.0
        accumulated = 0.0
        for key, weight in self.OVERALL_WEIGHTS.items():
            metric = report.metrics.get(key)
            if metric is None:
                continue
            accumulated += weight * metric.score
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return float(accumulated / total_weight)


#: Shared stateless instance.
_default_analyzer = DegradationAnalyzer()


def analyze_image(
    image: ImageData,
    source_path: Optional[Path] = None,
    roi: Optional[Dict[str, Any]] = None,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> AnalysisReport:
    """Convenience wrapper around the shared :class:`DegradationAnalyzer`."""
    return _default_analyzer.analyze(
        image, source_path=source_path, roi=roi, progress=progress, cancelled=cancelled
    )


def analyze_array(array: np.ndarray, source_path: Optional[Path] = None) -> AnalysisReport:
    """Analyse a bare numpy array (used by ROI and batch paths)."""
    return analyze_image(ImageData(pixels=array, source_path=source_path))
