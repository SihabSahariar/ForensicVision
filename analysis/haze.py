"""Haze / atmospheric-veiling estimation.

Uses the dark channel prior (He, Sun & Tang, CVPR 2009): in a haze-free
outdoor image most local patches contain at least one colour channel with a
very low value, so the "dark channel" is near zero. Airlight adds a roughly
uniform offset to every channel, lifting the dark channel.

The dark channel alone is confounded by legitimately bright scenes (snow, a
white wall, an over-exposed frame), so four signatures must agree before the
indicator fires, combined as a weighted geometric mean:

1. the **lower quartile** of the dark channel is elevated - airlight lifts the
   whole distribution, whereas a bright sky lifts only its upper tail;
2. mean **saturation** is low - scattering washes colour out;
3. **relative** local contrast is compressed - veiling scales scene contrast
   and adds a constant, so contrast measured against the local mean collapses;
4. the estimated **transmission is spatially uniform** - this is what separates
   a veil from defocus or motion blur, which depress contrast just as much but
   leave transmission tracking scene depth.

The indicator is additionally suppressed in proportion to the clipped-pixel
fraction, because the prior has no colour information to work with in a
blown-out frame. This is an explicitly heuristic indicator.
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

__all__ = ["analyze_haze", "dark_channel", "estimate_airlight"]


def dark_channel(rgb: np.ndarray, patch: int = 15) -> np.ndarray:
    """Return the dark channel of an RGB image in ``[0, 1]``.

    Args:
        rgb: ``HxWx3`` float array in ``[0, 1]``.
        patch: Local minimum-filter window size.
    """
    minimum = np.min(rgb, axis=2)
    size = max(3, patch | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(minimum, kernel)


def estimate_airlight(rgb: np.ndarray, dark: np.ndarray, top: float = 0.001) -> np.ndarray:
    """Estimate the atmospheric light from the brightest dark-channel pixels.

    Args:
        rgb: ``HxWx3`` float array in ``[0, 1]``.
        dark: Dark channel from :func:`dark_channel`.
        top: Fraction of the brightest dark-channel pixels to consider.

    Returns:
        A length-3 array with the per-channel airlight estimate.
    """
    flat_dark = dark.ravel()
    count = max(1, int(flat_dark.size * top))
    indices = np.argpartition(flat_dark, -count)[-count:]
    candidates = rgb.reshape(-1, 3)[indices]
    luminance = candidates @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    best = candidates[int(np.argmax(luminance))]
    return np.asarray(best, dtype=np.float32)


def _to_rgb_float(image: np.ndarray) -> np.ndarray:
    array = image
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array[..., :3]
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def analyze_haze(image: np.ndarray) -> MetricResult:
    """Estimate atmospheric haze severity.

    Args:
        image: RGB or grayscale array.

    Returns:
        A :class:`~analysis.base.MetricResult` for :data:`DegradationKey.HAZE`.
    """
    working = downsample_for_analysis(image, 1024)
    rgb = _to_rgb_float(working)
    gray = to_gray(working)

    # Blown highlights are not haze: a clipped sample carries no colour
    # information and would otherwise inflate the dark channel. Exclude
    # near-saturated pixels from the statistic.
    valid = rgb.max(axis=2) < 0.985
    dark = dark_channel(rgb, patch=15)
    dark_values = dark[valid] if valid.any() else dark.ravel()
    dark_mean = float(dark_values.mean())
    dark_p90 = float(np.percentile(dark_values, 90.0))
    # The lower quartile is the discriminating statistic. He et al. observe
    # that a haze-free outdoor image keeps most of its dark channel near zero;
    # a bright sky lifts only the *upper* tail, whereas airlight lifts the
    # whole distribution including its darkest patches.
    dark_p25 = float(np.percentile(dark_values, 25.0))
    clipped_fraction = float(1.0 - valid.mean())

    airlight = estimate_airlight(rgb, dark)
    airlight_luma = float(airlight @ np.array([0.299, 0.587, 0.114], dtype=np.float32))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation_mean = float(hsv[..., 1].mean())

    # Local contrast within 15x15 windows, expressed *relative* to the local
    # mean. Veiling multiplies scene contrast by the transmission and adds a
    # constant airlight term, so relative contrast collapses. An absolute
    # measure would instead confuse a hazy frame with a merely smooth one.
    mean_local = cv2.blur(gray, (15, 15))
    mean_sq_local = cv2.blur(gray * gray, (15, 15))
    local_variance = np.maximum(0.0, mean_sq_local - mean_local * mean_local)
    local_sigma = np.sqrt(local_variance)
    local_contrast = float(local_sigma.mean())
    relative_contrast = float((local_sigma / np.maximum(mean_local, 0.05)).mean())

    # Transmission map from the DCP model; a spatially uniform transmission is
    # the signature of a genuine veil rather than local scene content.
    normalised = np.clip(rgb / np.maximum(airlight[None, None, :], 1e-3), 0.0, 3.0)
    transmission = 1.0 - 0.95 * dark_channel(normalised.astype(np.float32), patch=15)
    transmission_std = float(transmission.std())

    was_gray = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)

    score_dark = linear_map(dark_p25, 0.12, 0.50)
    score_saturation = linear_map(saturation_mean, 0.32, 0.07)
    score_contrast = linear_map(relative_contrast, 0.09, 0.02)
    # A genuine atmospheric veil is spatially smooth, so the estimated
    # transmission varies little. Defocus and motion blur also depress relative
    # contrast, but leave transmission tracking scene depth and albedo. This
    # term is what separates "hazy" from "merely soft".
    score_uniform = linear_map(transmission_std, 0.16, 0.05)

    # Atmospheric scattering produces every appearance signature at once: an
    # airlight-lifted dark channel, washed-out colour and compressed relative
    # contrast. A weighted sum would let a single bright but sharp, saturated
    # scene (snow, a white wall, an over-exposed frame) score as hazy, so those
    # three are combined as a weighted geometric mean - any one being absent
    # suppresses the result.
    if was_gray:
        # Saturation carries no information for a monochrome source.
        appearance = (score_dark ** 0.60) * (score_contrast ** 0.40)
    else:
        appearance = (
            (score_dark ** 0.45)
            * (score_saturation ** 0.30)
            * (score_contrast ** 0.25)
        )

    # Transmission uniformity is applied as a *gate* rather than as another
    # geometric term. Defocus and motion blur reproduce all three appearance
    # signatures - they lift the dark channel through the local minimum filter
    # and they flatten relative contrast - so a term that merely nudges the
    # result is not enough to separate "hazy" from "soft". A veil is uniform;
    # blur leaves transmission tracking scene depth and albedo, and that is the
    # discriminating measurement.
    base_score = appearance * score_uniform

    # A frame that is largely clipped is an exposure failure, not a haze
    # observation: the dark channel prior has no colour information to work
    # with. Suppress the indicator rather than reporting a confident number.
    suppression = clamp01(1.0 - max(0.0, clipped_fraction - 0.05) / 0.45)
    score = clamp01(base_score * suppression)

    notes = []
    if suppression < 0.95:
        notes.append(
            "Haze indicator suppressed because a large share of the frame is "
            "clipped; over-exposure is the more parsimonious explanation."
        )
    if was_gray:
        notes.append(
            "Source is monochrome; the saturation component of the haze prior "
            "is unavailable and the score rests on the dark channel and local "
            "contrast only."
        )
    if clipped_fraction > 0.10:
        notes.append(
            f"{clipped_fraction * 100:.1f}% of pixels are near-saturated and "
            "were excluded from the dark-channel statistic."
        )
    if dark_p25 > 0.35:
        notes.append(
            "Even the darkest quartile of the dark channel is elevated, which "
            "is the characteristic signature of an airlight veil rather than of "
            "a merely bright scene."
        )
    if score_dark > 0.5 and score_uniform < 0.35:
        notes.append(
            "The dark channel is elevated but the estimated transmission varies "
            "strongly across the frame. That pattern fits scene content or "
            "optical blur better than a uniform atmospheric veil, and the "
            "indicator has been reduced accordingly. A genuinely localised fog "
            "bank would show the same pattern - inspect the frame directly."
        )
    if score > 0.5 and saturation_mean > 0.35:
        notes.append(
            "Elevated dark channel with retained saturation - the frame may "
            "simply contain large bright surfaces rather than haze."
        )

    return MetricResult(
        key=DegradationKey.HAZE.value,
        label="Haze",
        score=score,
        method=(
            "Weighted geometric mean of the dark-channel lower quartile (0.45), "
            "mean saturation (0.25), relative local contrast (0.20) and "
            "transmission uniformity (0.20), suppressed in proportion to the "
            "clipped-pixel fraction"
        ),
        reference="He, Sun & Tang, 'Single Image Haze Removal Using Dark "
        "Channel Prior', CVPR 2009",
        measurements={
            "dark_channel_mean": dark_mean,
            "dark_channel_p25": dark_p25,
            "dark_channel_p90": dark_p90,
            "near_saturated_fraction": clipped_fraction,
            "clipping_suppression_factor": suppression,
            "airlight_r": float(airlight[0]),
            "airlight_g": float(airlight[1]),
            "airlight_b": float(airlight[2]),
            "airlight_luminance": airlight_luma,
            "saturation_mean": saturation_mean,
            "local_contrast": local_contrast,
            "relative_local_contrast": relative_contrast,
            "transmission_std": transmission_std,
            "monochrome_source": was_gray,
            "score_dark_component": score_dark,
            "score_saturation_component": score_saturation,
            "score_contrast_component": score_contrast,
            "score_uniformity_component": score_uniform,
        },
        notes=notes,
    )
