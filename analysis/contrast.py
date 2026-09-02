"""Contrast and dynamic-range analysis."""

from __future__ import annotations

import logging

import numpy as np

from analysis.base import (
    MetricResult,
    clamp01,
    downsample_for_analysis,
    linear_map,
    to_gray,
)
from app.constants import DegradationKey

logger = logging.getLogger(__name__)

__all__ = ["analyze_contrast", "histogram_entropy"]


def histogram_entropy(gray: np.ndarray, bins: int = 256) -> float:
    """Return the Shannon entropy of the luminance histogram, in bits.

    A value near 8 indicates the full 8-bit range is exercised; values below
    about 5 indicate a compressed tonal range.
    """
    histogram, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    total = histogram.sum()
    if total == 0:
        return 0.0
    probabilities = histogram.astype(np.float64) / float(total)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def analyze_contrast(image: np.ndarray) -> MetricResult:
    """Estimate how far the image falls short of using its tonal range.

    Args:
        image: RGB or grayscale array.

    Returns:
        A :class:`~analysis.base.MetricResult` for
        :data:`DegradationKey.LOW_CONTRAST`.
    """
    working = downsample_for_analysis(image)
    gray = to_gray(working)

    p01 = float(np.percentile(gray, 1.0))
    p99 = float(np.percentile(gray, 99.0))
    dynamic_range = max(0.0, p99 - p01)
    rms_contrast = float(np.std(gray))
    entropy = histogram_entropy(gray)

    occupied_bins = int(np.count_nonzero(np.histogram(gray, bins=256, range=(0, 1))[0]))
    occupancy = occupied_bins / 256.0

    score_range = linear_map(dynamic_range, 0.85, 0.18)
    score_rms = linear_map(rms_contrast, 0.22, 0.045)
    score_entropy = linear_map(entropy, 7.2, 4.0)
    score = clamp01(0.45 * score_range + 0.35 * score_rms + 0.20 * score_entropy)

    notes = []
    if dynamic_range < 0.35:
        notes.append(
            f"The 1st-99th percentile spread covers only {dynamic_range * 255:.0f} "
            "of 255 levels."
        )
    if occupancy < 0.35:
        notes.append(
            f"Only {occupied_bins} of 256 luminance levels are populated; "
            "stretching contrast will produce visible posterisation."
        )
    if entropy < 5.0:
        notes.append("Low histogram entropy - limited tonal information present.")

    return MetricResult(
        key=DegradationKey.LOW_CONTRAST.value,
        label="Low Contrast",
        score=score,
        method=(
            "Weighted 1-99 percentile dynamic range (0.45), RMS contrast (0.35) "
            "and histogram entropy (0.20)"
        ),
        measurements={
            "percentile_01": p01,
            "percentile_99": p99,
            "dynamic_range": dynamic_range,
            "dynamic_range_8bit_levels": dynamic_range * 255.0,
            "rms_contrast": rms_contrast,
            "histogram_entropy_bits": entropy,
            "occupied_levels": occupied_bins,
            "level_occupancy": occupancy,
            "score_range_component": score_range,
            "score_rms_component": score_rms,
            "score_entropy_component": score_entropy,
        },
        notes=notes,
    )
