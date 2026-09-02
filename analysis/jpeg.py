"""JPEG compression artefact analysis.

Measures three things:

* **Blockiness** - the excess of pixel differences across 8x8 block boundaries
  relative to differences inside blocks (Wang, Bovik & Evan, ICIP 2000).
* **Ringing** - high-frequency oscillation in the neighbourhood of strong
  edges, the visual signature of quantised high-frequency DCT coefficients.
* **Container evidence** - when the source file is a JPEG, its quantisation
  tables are read directly and an IJG-equivalent quality factor is derived.
  This is measured evidence rather than a heuristic and is weighted highest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

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

__all__ = ["analyze_jpeg", "blockiness", "estimate_jpeg_quality"]

#: The IJG standard luminance quantisation table at quality 50.
_STD_LUMA_QTABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)

#: Natural (row-major) position of each coefficient in JPEG zigzag order.
#: Quantisation tables are stored in the file - and returned by Pillow - in
#: zigzag order, so they must be permuted before comparing element-wise with
#: the standard table above.
_ZIGZAG_TO_NATURAL = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ],
    dtype=np.intp,
)


def dezigzag(table: np.ndarray) -> np.ndarray:
    """Reorder a 64-element zigzag-ordered table into an 8x8 natural array."""
    natural = np.empty(64, dtype=np.float64)
    natural[_ZIGZAG_TO_NATURAL] = np.asarray(table, dtype=np.float64).ravel()[:64]
    return natural.reshape(8, 8)


def blockiness(gray: np.ndarray, block: int = 8) -> Tuple[float, float, float]:
    """Measure 8x8 blocking strength.

    The estimator is *phase-selective*. Comparing the block-boundary phase
    against the average of all interior positions reports a false positive on
    any scene built from axis-aligned edges - architecture, screenshots,
    documents, fencing - because those edges land on the boundary phase as
    often as anywhere else. Instead, the mean absolute neighbour difference is
    computed separately for each of the ``block`` phases and the boundary phase
    is compared against the *median* of the others. Genuine JPEG blocking
    elevates exactly one phase; scene structure elevates them all.

    Args:
        gray: ``HxW`` float image in ``[0, 1]``.
        block: Block size; JPEG uses 8.

    Returns:
        ``(blockiness, horizontal_ratio, vertical_ratio)``.
    """
    height, width = gray.shape[:2]
    if height < block * 3 or width < block * 3:
        return 0.0, 1.0, 1.0

    # Difference index j holds |x[j+1] - x[j]|; a JPEG block boundary sits
    # between samples block-1 and block, i.e. at difference index block-1.
    d_h = np.abs(np.diff(gray, axis=1))
    d_v = np.abs(np.diff(gray, axis=0))

    def _phase_ratio(differences: np.ndarray, axis: int) -> Tuple[float, float]:
        """Return ``(ratio, boundary_mean)`` for one axis."""
        length = differences.shape[axis]
        if length < block * 2:
            return 1.0, 0.0
        means = []
        for phase in range(block):
            indices = np.arange(phase, length, block)
            if indices.size == 0:
                continue
            selected = (
                differences[:, indices] if axis == 1 else differences[indices, :]
            )
            means.append(float(selected.mean()))
        if len(means) < block:
            return 1.0, 0.0

        boundary_mean = means[block - 1]
        others = means[: block - 1]
        reference = float(np.median(others))

        # A near-flat image drives the reference to the quantisation floor,
        # where the ratio explodes on numerical noise alone. One 8-bit level is
        # the smallest difference that can carry real signal.
        floor = 1.0 / 255.0
        return boundary_mean / max(reference, floor), boundary_mean

    ratio_h, boundary_h = _phase_ratio(d_h, axis=1)
    ratio_v, boundary_v = _phase_ratio(d_v, axis=0)

    # JPEG quantises a square 8x8 lattice, so genuine blocking raises *both*
    # axes. Taking the maximum would let single-axis structure through: a
    # horizontally motion-blurred frame has its horizontal differences smoothed
    # away, leaving the untouched vertical scene structure to dominate. The
    # geometric mean requires both axes to agree while still tolerating the
    # asymmetry a real anisotropic degradation introduces.
    combined = float(np.sqrt(max(ratio_h, 0.0) * max(ratio_v, 0.0)))

    # Damp the result when the absolute boundary discontinuity is itself below
    # one 8-bit level - real blocking is visible, not sub-LSB.
    floor = 1.0 / 255.0
    strength = min(1.0, max(boundary_h, boundary_v) / floor)
    excess = (combined - 1.0) * strength
    return max(0.0, excess), ratio_h, ratio_v


def ringing_strength(gray: np.ndarray) -> float:
    """Estimate ringing energy in the neighbourhood of strong edges.

    High-pass energy is measured in a dilated band around Canny edges but
    excluding the edges themselves, normalised by the overall high-pass energy.
    """
    height, width = gray.shape[:2]
    if height < 32 or width < 32:
        return 0.0

    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, 60, 160)
    if edges.max() == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_edge = cv2.dilate(edges, kernel, iterations=1) > 0
    on_edge = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    band = near_edge & ~on_edge
    if band.sum() < 64:
        return 0.0

    high_pass = gray - cv2.GaussianBlur(gray, (0, 0), 1.4)
    band_energy = float(np.mean(np.abs(high_pass[band])))
    flat = ~near_edge
    flat_energy = float(np.mean(np.abs(high_pass[flat]))) if flat.sum() > 64 else 0.0
    return float(max(0.0, band_energy - flat_energy))


def estimate_jpeg_quality(path: Path) -> Optional[Dict[str, float]]:
    """Read a JPEG's quantisation tables and derive an IJG quality factor.

    Args:
        path: Path to the candidate JPEG file.

    Returns:
        A mapping with ``quality``, ``luma_mean_step`` and ``table_count``, or
        ``None`` when the file is not a JPEG or has no readable tables.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None

    try:
        with Image.open(path) as img:
            if (img.format or "").upper() != "JPEG":
                return None
            tables = getattr(img, "quantization", None)
            if not tables:
                return None
            first_key = 0 if 0 in tables else sorted(tables)[0]
            luma = dezigzag(tables[first_key])
    except Exception:
        logger.debug("Could not read quantisation tables from %s", path, exc_info=True)
        return None

    # Invert the IJG scaling rule: scale = 5000/Q for Q<50 else 200-2Q, with
    # table = clip(round((std*scale + 50)/100), 1, 255).
    with np.errstate(divide="ignore", invalid="ignore"):
        scales = (luma * 100.0 - 50.0) / _STD_LUMA_QTABLE
    usable = scales[np.isfinite(scales) & (scales > 0)]
    if usable.size == 0:
        return None
    scale = float(np.median(usable))

    if scale <= 0:
        quality = 100.0
    elif scale < 100.0:
        quality = (200.0 - scale) / 2.0
    else:
        quality = 5000.0 / scale
    quality = float(np.clip(quality, 1.0, 100.0))

    return {
        "quality": quality,
        "luma_mean_step": float(luma.mean()),
        "table_count": float(len(tables)),
    }


def analyze_jpeg(
    image: np.ndarray, source_path: Optional[Path] = None
) -> MetricResult:
    """Estimate JPEG compression artefact severity.

    Args:
        image: RGB or grayscale array.
        source_path: Original file, used to read quantisation tables when the
            evidence is itself a JPEG.

    Returns:
        A :class:`~analysis.base.MetricResult` for :data:`DegradationKey.JPEG`.
    """
    # Blocking is a pixel-grid phenomenon: resampling destroys it, so unlike the
    # other analyzers this one measures the image at native resolution when it
    # is small enough, and records when it could not.
    height, width = image.shape[:2]
    resampled = max(height, width) > 3000
    working = downsample_for_analysis(image, 3000) if resampled else image
    gray = to_gray(working)

    excess, ratio_h, ratio_v = blockiness(gray)
    ringing = ringing_strength(gray)

    score_block = linear_map(excess, 0.06, 0.85)
    score_ringing = linear_map(ringing, 0.002, 0.03)

    measurements: Dict[str, float] = {
        "blockiness_excess": excess,
        "boundary_ratio_horizontal": ratio_h,
        "boundary_ratio_vertical": ratio_v,
        "ringing_strength": ringing,
        "score_blockiness_component": score_block,
        "score_ringing_component": score_ringing,
    }
    notes = []

    score = clamp01(0.7 * score_block + 0.3 * score_ringing)

    quality_info = estimate_jpeg_quality(source_path) if source_path else None
    if quality_info is not None:
        quality = quality_info["quality"]
        measurements["container_jpeg_quality"] = quality
        measurements["quantisation_luma_mean_step"] = quality_info["luma_mean_step"]
        # Quality 95+ is visually lossless; 40 and below is heavily quantised.
        score_container = linear_map(quality, 95.0, 35.0)
        measurements["score_container_component"] = score_container
        score = clamp01(0.55 * score_container + 0.30 * score_block + 0.15 * score_ringing)
        notes.append(
            f"Source container is JPEG with an estimated IJG quality factor of "
            f"{quality:.0f} (derived from the embedded quantisation tables)."
        )
        if quality < 60:
            notes.append(
                "Quantisation is coarse; fine detail has been discarded at "
                "encode time and cannot be recovered, only plausibly filled in."
            )
    else:
        notes.append(
            "No JPEG quantisation tables were available; the score rests on "
            "pixel-domain blocking and ringing measurements only."
        )

    if resampled:
        notes.append(
            "Image exceeded 3000 px and was resampled for analysis, which "
            "attenuates block-grid measurements."
        )
    if excess > 0.4:
        notes.append("A pronounced 8x8 discontinuity grid is present.")

    measurements["working_size"] = f"{gray.shape[1]}x{gray.shape[0]}"

    return MetricResult(
        key=DegradationKey.JPEG.value,
        label="JPEG Artifacts",
        score=score,
        method=(
            "Block-boundary discontinuity ratio and near-edge ringing energy, "
            "combined with the container's quantisation tables when available"
        ),
        reference="Wang, Bovik & Evan, ICIP 2000; IJG quantisation scaling rule",
        measurements=measurements,
        notes=notes,
    )
