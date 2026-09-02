"""Shared types and helpers for the degradation analyzers.

Every analyzer returns a :class:`MetricResult` carrying both a normalised
severity score *and* the raw measurements it was derived from. The raw values
are what an examiner needs in order to challenge or reproduce a finding; the
normalised score exists only to drive the UI and the pipeline recommender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

__all__ = [
    "MetricResult",
    "clamp01",
    "linear_map",
    "log_map",
    "to_gray",
    "downsample_for_analysis",
    "ANALYZER_VERSION",
]

#: Bumped whenever a scoring formula changes, so stored results stay comparable.
ANALYZER_VERSION: str = "1.0.0"

#: Longest edge used for analysis; larger images are area-averaged down. This
#: keeps measurements stable across resolutions and bounds run time.
ANALYSIS_MAX_EDGE: int = 1600


@dataclass
class MetricResult:
    """One degradation indicator.

    Attributes:
        key: Canonical metric key, see :class:`app.constants.DegradationKey`.
        label: Human readable name.
        score: Normalised severity in ``[0, 1]``; higher means more degraded.
        method: Short description of the estimator used.
        measurements: Raw numeric outputs keyed by name.
        notes: Free-text observations shown in the detail dialog.
        reference: Literature reference for the estimator, when applicable.
    """

    key: str
    label: str
    score: float
    method: str = ""
    measurements: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    reference: str = ""

    def __post_init__(self) -> None:
        self.score = clamp01(self.score)

    @property
    def percent(self) -> int:
        """Score expressed as an integer percentage."""
        return int(round(self.score * 100))

    @property
    def severity(self) -> str:
        """Coarse severity band used for colour coding."""
        if self.score >= 0.66:
            return "high"
        if self.score >= 0.35:
            return "medium"
        return "low"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "key": self.key,
            "label": self.label,
            "score": round(self.score, 4),
            "percent": self.percent,
            "severity": self.severity,
            "method": self.method,
            "measurements": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.measurements.items()
            },
            "notes": list(self.notes),
            "reference": self.reference,
        }

    def detail_text(self) -> str:
        """Render a plain-text technical breakdown for the detail dialog."""
        lines = [
            f"{self.label}",
            "=" * max(8, len(self.label)),
            "",
            f"Severity score : {self.percent} / 100  ({self.severity})",
            f"Estimator      : {self.method or 'n/a'}",
        ]
        if self.reference:
            lines.append(f"Reference      : {self.reference}")
        lines.append("")
        lines.append("Raw measurements")
        lines.append("-" * 16)
        if self.measurements:
            width = max(len(str(k)) for k in self.measurements)
            for key, value in self.measurements.items():
                if isinstance(value, float):
                    rendered = f"{value:.6g}"
                else:
                    rendered = str(value)
                lines.append(f"  {str(key).ljust(width)} : {rendered}")
        else:
            lines.append("  (none)")
        if self.notes:
            lines.append("")
            lines.append("Observations")
            lines.append("-" * 12)
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``, mapping NaN to 0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return max(0.0, min(1.0, number))


def linear_map(value: float, low: float, high: float, invert: bool = False) -> float:
    """Map ``value`` from ``[low, high]`` onto ``[0, 1]``.

    Args:
        value: Raw measurement.
        low: Measurement that maps to 0 (or 1 when ``invert``).
        high: Measurement that maps to 1 (or 0 when ``invert``).
        invert: Reverse the mapping direction.
    """
    if high == low:
        return 0.0
    ratio = (float(value) - low) / (high - low)
    ratio = clamp01(ratio)
    return 1.0 - ratio if invert else ratio


def log_map(value: float, low: float, high: float, invert: bool = False) -> float:
    """Logarithmic variant of :func:`linear_map` for wide-dynamic measurements."""
    epsilon = 1e-12
    v = math.log10(max(float(value), epsilon))
    lo = math.log10(max(low, epsilon))
    hi = math.log10(max(high, epsilon))
    return linear_map(v, lo, hi, invert=invert)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return an ``HxW`` ``float32`` luminance plane in ``[0, 1]``.

    Accepts ``uint8``/``uint16``/float arrays with 1, 3 or 4 channels.
    """
    array = image
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    elif array.dtype == np.uint16:
        array = array.astype(np.float32) / 65535.0
    else:
        array = array.astype(np.float32)
        if array.max(initial=0.0) > 1.5:
            array = array / 255.0

    if array.ndim == 2:
        return np.clip(array, 0.0, 1.0)
    if array.shape[2] >= 3:
        rgb = np.ascontiguousarray(array[..., :3])
        return np.clip(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 0.0, 1.0)
    return np.clip(array[..., 0], 0.0, 1.0)


def downsample_for_analysis(
    image: np.ndarray, max_edge: int = ANALYSIS_MAX_EDGE
) -> np.ndarray:
    """Area-average ``image`` down so its longest edge is ``max_edge``.

    Analysis on very large frames is both slow and biased: block and noise
    statistics change with sampling density. Downsampling to a fixed working
    size makes scores comparable between a 640x480 CCTV frame and a 45 MP
    scan, and the raw measurements record the working size used.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
