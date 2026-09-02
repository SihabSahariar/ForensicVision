"""Blur and motion-blur estimation.

Three independent estimators are combined so a single failure mode (e.g. a
low-texture scene fooling a variance measure) does not dominate the result:

1. **Perceptual blur** - Crete et al. (2007), a no-reference metric that
   compares the image's neighbour differences before and after a controlled
   re-blur. Robust and bounded in ``[0, 1]``.
2. **Laplacian variance** - the classical focus measure, log-mapped.
3. **Spectral slope** - the fraction of spectral energy above a normalised
   radius; genuinely sharp images retain high-frequency energy.

Motion blur is detected from the anisotropy of directional gradient energy:
linear motion suppresses gradients along the direction of travel while leaving
the perpendicular direction intact.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

from analysis.base import (
    MetricResult,
    clamp01,
    downsample_for_analysis,
    linear_map,
    log_map,
    to_gray,
)
from app.constants import DegradationKey

logger = logging.getLogger(__name__)

__all__ = ["analyze_blur", "analyze_motion_blur", "perceptual_blur", "laplacian_variance"]


def laplacian_variance(gray: np.ndarray) -> float:
    """Return the variance of the Laplacian - the classical focus measure.

    Args:
        gray: ``HxW`` float image in ``[0, 1]``.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(laplacian.var())


def tenengrad(gray: np.ndarray) -> float:
    """Return the Tenengrad focus measure (mean squared Sobel gradient)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def perceptual_blur(gray: np.ndarray) -> Tuple[float, float, float]:
    """Compute the Crete no-reference perceptual blur metric.

    The image is re-blurred with a 9-tap box filter along each axis. In an
    already-blurred image the extra blur changes neighbour differences very
    little, so the ratio of "lost variation" is small and the metric is high.

    Args:
        gray: ``HxW`` float image in ``[0, 1]``.

    Returns:
        ``(blur, blur_horizontal, blur_vertical)``, each in ``[0, 1]``.

    Reference:
        F. Crete et al., "The blur effect: perception and estimation with a new
        no-reference perceptual blur metric", SPIE HVEI 2007.
    """
    if gray.ndim != 2 or gray.size == 0:
        return 0.0, 0.0, 0.0

    kernel = np.ones((1, 9), dtype=np.float32) / 9.0
    blurred_h = cv2.filter2D(gray, cv2.CV_32F, kernel)
    blurred_v = cv2.filter2D(gray, cv2.CV_32F, kernel.T)

    def _axis(original: np.ndarray, blurred: np.ndarray, axis: int) -> float:
        d_original = np.abs(np.diff(original, axis=axis))
        d_blurred = np.abs(np.diff(blurred, axis=axis))
        variation = np.maximum(0.0, d_original - d_blurred)
        total = float(d_original.sum())
        if total <= 1e-9:
            return 0.0
        return clamp01((total - float(variation.sum())) / total)

    blur_h = _axis(gray, blurred_h, axis=1)
    blur_v = _axis(gray, blurred_v, axis=0)
    return max(blur_h, blur_v), blur_h, blur_v


def spectral_high_frequency_ratio(gray: np.ndarray, cutoff: float = 0.25) -> float:
    """Return the share of spectral energy above ``cutoff`` x Nyquist.

    Args:
        gray: ``HxW`` float image in ``[0, 1]``.
        cutoff: Normalised radius, 0 at DC and 1 at the spectrum corner.
    """
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return 0.0

    windowed = gray * np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(windowed)))
    power = spectrum.astype(np.float64) ** 2

    cy, cx = height / 2.0, width / 2.0
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)

    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    high = float(power[radius >= cutoff].sum())
    return clamp01(high / total)


def _directional_energy(gray: np.ndarray, bins: int = 18) -> np.ndarray:
    """Return gradient energy sampled over ``bins`` orientations in [0, pi)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.linspace(0.0, np.pi, bins, endpoint=False)
    energies = np.empty(bins, dtype=np.float64)
    for index, theta in enumerate(angles):
        directional = gx * np.cos(theta) + gy * np.sin(theta)
        energies[index] = float(np.mean(directional.astype(np.float64) ** 2))
    return energies


def analyze_blur(image: np.ndarray) -> MetricResult:
    """Estimate overall blur severity.

    Args:
        image: RGB or grayscale array of any supported dtype.

    Returns:
        A :class:`~analysis.base.MetricResult` for :data:`DegradationKey.BLUR`.
    """
    working = downsample_for_analysis(image)
    gray = to_gray(working)

    blur_perceptual, blur_h, blur_v = perceptual_blur(gray)
    lap_var = laplacian_variance(gray)
    ten = tenengrad(gray)
    hf_ratio = spectral_high_frequency_ratio(gray)

    # Laplacian variance of a well-focused 8-bit photo normalised to [0,1]
    # typically lands between 1e-3 and 3e-2; 1e-5 is unmistakably defocused.
    score_laplacian = log_map(lap_var, 1e-5, 8e-3, invert=True)
    # A sharp natural image keeps roughly 8-25% of its energy above 0.25 fs.
    score_spectral = linear_map(hf_ratio, 0.015, 0.16, invert=True)

    score = clamp01(
        0.55 * blur_perceptual + 0.25 * score_laplacian + 0.20 * score_spectral
    )

    notes = []
    if lap_var < 1e-4:
        notes.append(
            "Very low Laplacian variance - consistent with heavy defocus or a "
            "low-texture scene. Inspect a detailed region before concluding."
        )
    if hf_ratio < 0.02:
        notes.append(
            "Almost no energy above a quarter of the sampling frequency; the "
            "frame carries little recoverable fine detail."
        )
    if blur_perceptual > 0.75:
        notes.append("Perceptual blur metric is in the strongly-blurred band.")

    return MetricResult(
        key=DegradationKey.BLUR.value,
        label="Blur",
        score=score,
        method=(
            "Weighted combination of Crete perceptual blur (0.55), log-mapped "
            "Laplacian variance (0.25) and spectral high-frequency ratio (0.20)"
        ),
        reference="Crete et al., SPIE HVEI 2007; Pech-Pacheco et al., ICPR 2000",
        measurements={
            "perceptual_blur": blur_perceptual,
            "perceptual_blur_horizontal": blur_h,
            "perceptual_blur_vertical": blur_v,
            "laplacian_variance": lap_var,
            "tenengrad": ten,
            "high_frequency_ratio": hf_ratio,
            "score_laplacian_component": score_laplacian,
            "score_spectral_component": score_spectral,
            "working_size": f"{gray.shape[1]}x{gray.shape[0]}",
        },
        notes=notes,
    )


def analyze_motion_blur(image: np.ndarray) -> MetricResult:
    """Estimate directional (motion) blur severity and its dominant angle.

    Linear motion attenuates image gradients along the direction of travel. The
    estimator samples gradient energy over 18 orientations and reports the
    anisotropy together with the orientation of minimum energy, which
    corresponds to the motion direction.

    Args:
        image: RGB or grayscale array.

    Returns:
        A :class:`~analysis.base.MetricResult` for
        :data:`DegradationKey.MOTION_BLUR`.
    """
    working = downsample_for_analysis(image)
    gray = to_gray(working)

    energies = _directional_energy(gray)
    if energies.size == 0 or float(energies.max()) <= 1e-12:
        return MetricResult(
            key=DegradationKey.MOTION_BLUR.value,
            label="Motion Blur",
            score=0.0,
            method="Directional gradient-energy anisotropy",
            measurements={"gradient_energy": 0.0},
            notes=["Image carries no measurable gradient energy."],
        )

    e_max = float(energies.max())
    e_min = float(energies.min())
    anisotropy = clamp01((e_max - e_min) / (e_max + e_min + 1e-12))
    min_index = int(np.argmin(energies))
    angle_deg = float(min_index * 180.0 / energies.size)

    blur_perceptual, _, _ = perceptual_blur(gray)

    # Motion blur requires both overall softness *and* a directional signature.
    # Isotropic defocus produces low anisotropy; sharp textured images with
    # directional content produce high anisotropy but low blur.
    directional = linear_map(anisotropy, 0.18, 0.62)
    score = clamp01(directional * linear_map(blur_perceptual, 0.30, 0.80))

    notes = []
    if score >= 0.45:
        notes.append(
            f"Directional signature consistent with linear motion at "
            f"approximately {angle_deg:.0f} deg from horizontal."
        )
        notes.append(
            "Angle is an estimate from gradient anisotropy, not a measured "
            "camera trajectory."
        )
    elif anisotropy > 0.5 and blur_perceptual < 0.35:
        notes.append(
            "Strong directional structure but the frame is sharp - most likely "
            "oriented scene content (fencing, blinds, text) rather than motion."
        )

    return MetricResult(
        key=DegradationKey.MOTION_BLUR.value,
        label="Motion Blur",
        score=score,
        method=(
            "Anisotropy of directional gradient energy over 18 orientations, "
            "gated by the perceptual blur level"
        ),
        measurements={
            "anisotropy": anisotropy,
            "estimated_angle_deg": angle_deg,
            "energy_max": e_max,
            "energy_min": e_min,
            "perceptual_blur": blur_perceptual,
        },
        notes=notes,
    )
