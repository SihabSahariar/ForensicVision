"""Noise estimation (luminance and chroma).

Two independent estimators are used:

* **Immerkaer's fast estimator** - convolution with a 3x3 Laplacian-of-Laplacian
  mask whose response to a smooth signal is zero, so the mean absolute response
  is proportional to the noise standard deviation.
* **Wavelet MAD** - the median absolute deviation of the finest diagonal detail
  band, a standard robust sigma estimator that is insensitive to edges.

Chroma noise is measured separately in YCrCb because sensor and codec noise is
frequently far stronger in the chroma planes, and the appropriate remedy
differs.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
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

__all__ = ["analyze_noise", "estimate_sigma_immerkaer", "estimate_sigma_mad"]

#: Immerkaer's mask: zero response to first- and second-order polynomials.
_IMMERKAER_MASK = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32
)


def estimate_sigma_immerkaer(plane: np.ndarray) -> float:
    """Estimate the noise standard deviation of a single plane.

    Args:
        plane: ``HxW`` float array in ``[0, 1]``.

    Returns:
        Estimated sigma in the same units as ``plane``.

    Reference:
        J. Immerkaer, "Fast noise variance estimation", CVIU 64(2), 1996.
    """
    height, width = plane.shape[:2]
    if height < 4 or width < 4:
        return 0.0
    response = cv2.filter2D(plane, cv2.CV_32F, _IMMERKAER_MASK)
    interior = response[1:-1, 1:-1]
    scale = np.sqrt(np.pi / 2.0) / (6.0 * (width - 2) * (height - 2))
    return float(scale * np.abs(interior).sum())


def estimate_sigma_mad(plane: np.ndarray) -> float:
    """Estimate noise sigma from the finest diagonal wavelet detail band.

    A single-level Haar decomposition is used, implemented directly with
    slicing so that PyWavelets is not a hard dependency.

    Args:
        plane: ``HxW`` float array in ``[0, 1]``.
    """
    height, width = plane.shape[:2]
    if height < 4 or width < 4:
        return 0.0
    even_rows = plane[0 : height - height % 2 : 2, 0 : width - width % 2 : 2]
    odd_rows = plane[1 : height - height % 2 : 2, 0 : width - width % 2 : 2]
    even_cols = plane[0 : height - height % 2 : 2, 1 : width - width % 2 : 2]
    odd_cols = plane[1 : height - height % 2 : 2, 1 : width - width % 2 : 2]
    diagonal = (even_rows - odd_rows - even_cols + odd_cols) / 2.0
    median_abs = float(np.median(np.abs(diagonal)))
    return median_abs / 0.6745


def _edge_free_mask(gray: np.ndarray) -> np.ndarray:
    """Return a boolean mask of low-gradient (flat) regions.

    Noise statistics measured over edges are inflated by scene content, so the
    reported sigma is also computed restricted to flat areas.
    """
    gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) ** 2
    gradient += cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) ** 2
    gradient = cv2.GaussianBlur(gradient, (0, 0), 2.0)
    if gradient.size == 0:
        return np.zeros_like(gray, dtype=bool)
    threshold = float(np.percentile(gradient, 35.0))
    return gradient <= max(threshold, 1e-8)


def analyze_noise(image: np.ndarray) -> MetricResult:
    """Estimate luminance and chroma noise severity.

    Args:
        image: RGB or grayscale array of any supported dtype.

    Returns:
        A :class:`~analysis.base.MetricResult` for :data:`DegradationKey.NOISE`.
    """
    working = downsample_for_analysis(image)
    gray = to_gray(working)

    sigma_immerkaer = estimate_sigma_immerkaer(gray)
    sigma_mad = estimate_sigma_mad(gray)

    flat_mask = _edge_free_mask(gray)
    flat_fraction = float(flat_mask.mean()) if flat_mask.size else 0.0
    if flat_fraction > 0.02:
        flat_values = gray[flat_mask]
        sigma_flat = float(np.std(flat_values - np.mean(flat_values)))
    else:
        sigma_flat = sigma_mad

    sigma_luma = float(np.median([sigma_immerkaer, sigma_mad]))

    sigma_cr = sigma_cb = 0.0
    if working.ndim == 3 and working.shape[2] >= 3:
        rgb = working[..., :3]
        if rgb.dtype == np.uint8:
            rgb_float = rgb.astype(np.float32) / 255.0
        elif rgb.dtype == np.uint16:
            rgb_float = rgb.astype(np.float32) / 65535.0
        else:
            rgb_float = np.clip(rgb.astype(np.float32), 0.0, 1.0)
        ycrcb = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2YCrCb)
        sigma_cr = estimate_sigma_immerkaer(np.ascontiguousarray(ycrcb[..., 1]))
        sigma_cb = estimate_sigma_immerkaer(np.ascontiguousarray(ycrcb[..., 2]))

    sigma_chroma = max(sigma_cr, sigma_cb)

    # 8-bit reference points: sigma 0.004 (~1/255) is essentially clean;
    # sigma 0.09 (~23/255) is heavy sensor or low-light noise.
    score_luma = linear_map(sigma_luma, 0.004, 0.075)
    score_chroma = linear_map(sigma_chroma, 0.004, 0.055)
    score = clamp01(max(score_luma, 0.65 * score_luma + 0.55 * score_chroma))

    notes = []
    if sigma_chroma > sigma_luma * 1.4 and sigma_chroma > 0.01:
        notes.append(
            "Chroma noise exceeds luminance noise - typical of low-light sensor "
            "gain or aggressive chroma subsampling."
        )
    if flat_fraction < 0.05:
        notes.append(
            "Few flat regions were available; the sigma estimate may include "
            "scene texture."
        )
    if sigma_luma > 0.06:
        notes.append("Noise level is high enough to mask fine detail.")

    return MetricResult(
        key=DegradationKey.NOISE.value,
        label="Noise",
        score=score,
        method=(
            "Median of Immerkaer fast variance and Haar-MAD sigma for luminance; "
            "Immerkaer sigma on Cr/Cb for chroma"
        ),
        reference="Immerkaer, CVIU 1996; Donoho & Johnstone, Biometrika 1994",
        measurements={
            "sigma_luma": sigma_luma,
            "sigma_immerkaer": sigma_immerkaer,
            "sigma_mad": sigma_mad,
            "sigma_flat_regions": sigma_flat,
            "sigma_cr": sigma_cr,
            "sigma_cb": sigma_cb,
            "sigma_chroma": sigma_chroma,
            "sigma_luma_8bit_equivalent": sigma_luma * 255.0,
            "flat_region_fraction": flat_fraction,
            "score_luma_component": score_luma,
            "score_chroma_component": score_chroma,
        },
        notes=notes,
    )
