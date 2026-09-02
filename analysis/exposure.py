"""Exposure and clipping analysis.

Reports under-exposure and over-exposure as two separate indicators because
they call for opposite corrections and can coexist in a single high-dynamic
range scene (crushed shadows plus blown highlights).

Clipping is measured on the *maximum* channel for highlights and the *minimum*
channel for shadows, so a single blown colour channel is not hidden by the
luminance average.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

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

__all__ = ["analyze_exposure", "clipping_statistics"]

#: 8-bit sample values at or beyond which a channel counts as clipped.
_SHADOW_CLIP = 2.0 / 255.0
_HIGHLIGHT_CLIP = 253.0 / 255.0


def _normalised_channels(image: np.ndarray) -> np.ndarray:
    """Return an ``HxWxC`` float array in ``[0, 1]`` (alpha dropped)."""
    array = image
    if array.ndim == 2:
        array = array[..., None]
    array = array[..., : min(3, array.shape[2])]
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    result = array.astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def clipping_statistics(image: np.ndarray) -> Dict[str, float]:
    """Return shadow/highlight clipping fractions and channel extremes."""
    channels = _normalised_channels(image)
    channel_max = channels.max(axis=2)
    channel_min = channels.min(axis=2)

    highlight = float(np.mean(channel_max >= _HIGHLIGHT_CLIP))
    shadow = float(np.mean(channel_min <= _SHADOW_CLIP))

    per_channel: Dict[str, float] = {}
    names = ("red", "green", "blue")[: channels.shape[2]]
    for index, name in enumerate(names):
        plane = channels[..., index]
        per_channel[f"{name}_clipped_high"] = float(np.mean(plane >= _HIGHLIGHT_CLIP))
        per_channel[f"{name}_clipped_low"] = float(np.mean(plane <= _SHADOW_CLIP))

    stats = {
        "highlight_clipped_fraction": highlight,
        "shadow_clipped_fraction": shadow,
    }
    stats.update(per_channel)
    return stats


def analyze_exposure(image: np.ndarray) -> Tuple[MetricResult, MetricResult]:
    """Estimate under- and over-exposure severity.

    Args:
        image: RGB or grayscale array of any supported dtype.

    Returns:
        ``(underexposure_result, overexposure_result)``.
    """
    working = downsample_for_analysis(image)
    gray = to_gray(working)
    stats = clipping_statistics(working)

    mean_luma = float(gray.mean())
    median_luma = float(np.median(gray))
    p05 = float(np.percentile(gray, 5.0))
    p95 = float(np.percentile(gray, 95.0))

    highlight = stats["highlight_clipped_fraction"]
    shadow = stats["shadow_clipped_fraction"]

    # Reference points: a well-exposed frame sits around 0.35-0.55 mean luma.
    dark_component = linear_map(median_luma, 0.34, 0.06)
    shadow_component = linear_map(shadow, 0.02, 0.35)
    under_score = clamp01(0.7 * dark_component + 0.3 * shadow_component)

    bright_component = linear_map(median_luma, 0.62, 0.93)
    highlight_component = linear_map(highlight, 0.01, 0.25)
    over_score = clamp01(0.55 * bright_component + 0.45 * highlight_component)

    under_notes = []
    over_notes = []
    if shadow > 0.10:
        under_notes.append(
            f"{shadow * 100:.1f}% of pixels have at least one channel at or below "
            "the black point; detail in those areas is not recoverable."
        )
    if under_score > 0.5:
        under_notes.append(
            "Brightening will amplify sensor noise proportionally; consider "
            "denoising before or after exposure correction."
        )
    if highlight > 0.05:
        over_notes.append(
            f"{highlight * 100:.1f}% of pixels have at least one channel at or "
            "above the white point; those samples carry no recoverable data."
        )
    for name in ("red", "green", "blue"):
        key = f"{name}_clipped_high"
        if stats.get(key, 0.0) > highlight * 1.5 and stats.get(key, 0.0) > 0.03:
            over_notes.append(
                f"The {name} channel clips notably more than the others - "
                "possible colour cast or illuminant saturation."
            )

    shared = {
        "mean_luminance": mean_luma,
        "median_luminance": median_luma,
        "percentile_05": p05,
        "percentile_95": p95,
    }
    shared.update(stats)

    under = MetricResult(
        key=DegradationKey.UNDEREXPOSURE.value,
        label="Underexposure",
        score=under_score,
        method=(
            "Median luminance mapped against a 0.34-0.06 reference band, "
            "combined with the shadow-clipped pixel fraction"
        ),
        measurements=dict(shared, score_dark_component=dark_component,
                          score_shadow_component=shadow_component),
        notes=under_notes,
    )
    over = MetricResult(
        key=DegradationKey.OVEREXPOSURE.value,
        label="Overexposure",
        score=over_score,
        method=(
            "Median luminance mapped against a 0.62-0.93 reference band, "
            "combined with the highlight-clipped pixel fraction"
        ),
        measurements=dict(shared, score_bright_component=bright_component,
                          score_highlight_component=highlight_component),
        notes=over_notes,
    )
    return under, over
