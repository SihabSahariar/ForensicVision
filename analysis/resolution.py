"""Resolution adequacy analysis.

Two distinct questions are answered:

1. **Absolute resolution** - is the frame simply small? A 320x240 CCTV capture
   has a hard ceiling on the detail it can contain. This drives the score,
   because it is what determines whether super-resolution is worth running.
2. **Oversampling** - was the frame interpolated up from something smaller?
   Natural image spectra fall off steadily but never vanish, because sensor
   noise and aliasing keep some energy right up to Nyquist. Interpolation
   leaves a near-empty outer band instead. Comparing the outer annulus
   (0.75-1.0 of Nyquist) with a mid annulus (0.25-0.5) detects that cliff.

The distinction matters operationally: case 1 benefits from super-resolution;
case 2 does not, because the pixels are already interpolated. Oversampling is
therefore reported as a warning note rather than being folded into the score,
so an examiner is not misled by an impressive-looking pixel count.
"""

from __future__ import annotations

import logging
from typing import Tuple

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

__all__ = ["analyze_resolution", "oversampling_ratio"]

#: Long-edge lengths bracketing the absolute-resolution score. 1200 px is the
#: point at which resolution stops being the limiting factor for most forensic
#: detail tasks; below 240 px almost nothing is recoverable.
_SMALL_EDGE = 240
_ADEQUATE_EDGE = 1200

#: Outer-to-mid annulus energy ratio below which the frame looks interpolated.
#: Measured separation on natural and synthetic content: a natively sampled
#: frame sits around 1e-1, while 2x-4x interpolated enlargements sit between
#: 2e-4 and 7e-3, so 1e-2 separates them with roughly a decade of margin either
#: side. Very aggressive enlargement (6x and beyond) with a ringing-heavy
#: kernel such as Lanczos can push energy back into the outer band and escape
#: detection; that is a deliberate false-negative bias, since the absolute
#: resolution score already dominates in those cases.
_OVERSAMPLING_THRESHOLD = 1.0e-2


def oversampling_ratio(gray: np.ndarray) -> float:
    """Return the outer-to-mid annulus spectral energy ratio.

    Values below :data:`_OVERSAMPLING_THRESHOLD` indicate the frame was
    interpolated up from a smaller original, because genuine sampling always
    leaves some noise energy near Nyquist.

    Args:
        gray: ``HxW`` float image in ``[0, 1]``.

    Returns:
        The energy ratio; 1.0 is returned for images too small to measure.
    """
    height, width = gray.shape[:2]
    if height < 64 or width < 64:
        return 1.0

    windowed = gray * np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(windowed))).astype(np.float64) ** 2

    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)

    mid = power[(radius >= 0.25) & (radius < 0.5)]
    outer = power[(radius >= 0.75) & (radius <= 1.0)]
    if mid.size == 0 or outer.size == 0:
        return 1.0
    mid_mean = float(mid.mean())
    if mid_mean <= 0.0:
        return 1.0
    return float(outer.mean() / mid_mean)


def analyze_resolution(image: np.ndarray) -> MetricResult:
    """Estimate whether the frame's resolution limits recoverable detail.

    Args:
        image: RGB or grayscale array.

    Returns:
        A :class:`~analysis.base.MetricResult` for
        :data:`DegradationKey.LOW_RESOLUTION`.
    """
    height, width = image.shape[:2]
    long_edge = max(height, width)
    short_edge = min(height, width)
    megapixels = (height * width) / 1_000_000.0

    # Oversampling must be measured on the native grid: downsampling would
    # itself remove the outer band the estimator is looking for.
    gray_native = to_gray(image)
    ratio = oversampling_ratio(gray_native)
    oversampled = ratio < _OVERSAMPLING_THRESHOLD

    score = clamp01(linear_map(long_edge, _ADEQUATE_EDGE, _SMALL_EDGE))

    notes = []
    if long_edge < 640:
        notes.append(
            f"Frame long edge is {long_edge} px - typical of a low-resolution "
            "surveillance capture or a heavily cropped region."
        )
    if oversampled:
        notes.append(
            "Spectral energy stops abruptly well short of Nyquist "
            f"(outer/mid annulus ratio {ratio:.2e}). The frame appears to have "
            "been interpolated up from a smaller original, so its pixel count "
            "overstates the information it carries. Super-resolution applied to "
            "an already-interpolated frame adds no new measured detail."
        )
    if short_edge < 64:
        notes.append(
            "Short edge below 64 px; most restoration networks will need "
            "padding and results will be dominated by the learned prior."
        )

    return MetricResult(
        key=DegradationKey.LOW_RESOLUTION.value,
        label="Low Resolution",
        score=score,
        method=(
            "Absolute long-edge length mapped over 1200-240 px. Oversampling is "
            "detected separately from the outer/mid spectral annulus ratio and "
            "reported as a note rather than folded into the score."
        ),
        measurements={
            "width": width,
            "height": height,
            "long_edge": long_edge,
            "short_edge": short_edge,
            "megapixels": round(megapixels, 4),
            "oversampling_ratio": ratio,
            "appears_upscaled": oversampled,
        },
        notes=notes,
    )
