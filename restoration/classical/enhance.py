"""Classical tone, colour, denoise and dehaze operators.

All functions take and return ``HxWx3`` ``float32`` RGB arrays in ``[0, 1]``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "clahe",
    "gamma_correct",
    "auto_levels",
    "nlm_denoise",
    "bilateral_denoise",
    "dark_channel_dehaze",
    "jpeg_deblock",
    "lanczos_resize",
]


# --------------------------------------------------------------------------- #
# Tone
# --------------------------------------------------------------------------- #

def clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel.

    Operating in CIE L*a*b* keeps hue and saturation unchanged, so the result
    is a tone remap rather than a colour transform.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        clip_limit: Contrast limit; higher values equalise more aggressively
            and amplify noise.
        tile_grid: Number of tiles along each axis.
    """
    grid = max(1, int(tile_grid))
    lab = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2LAB)
    lightness = np.clip(lab[..., 0] * (255.0 / 100.0), 0, 255).astype(np.uint8)
    operator = cv2.createCLAHE(
        clipLimit=max(0.1, float(clip_limit)), tileGridSize=(grid, grid)
    )
    lab[..., 0] = operator.apply(lightness).astype(np.float32) * (100.0 / 255.0)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0.0, 1.0).astype(np.float32)


def gamma_correct(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply a power-law tone curve.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        gamma: Exponent; values below 1 brighten, above 1 darken.
    """
    gamma = max(0.05, float(gamma))
    return np.clip(np.power(np.clip(image, 0.0, 1.0), gamma), 0.0, 1.0).astype(
        np.float32
    )


def auto_levels(
    image: np.ndarray, low_percentile: float = 0.5, high_percentile: float = 99.5
) -> np.ndarray:
    """Stretch luminance so the given percentiles map to black and white.

    The stretch is computed on luminance and applied as a single gain/offset to
    all three channels, so the colour balance of the source is preserved.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        low_percentile: Luminance percentile mapped to 0.
        high_percentile: Luminance percentile mapped to 1.
    """
    luminance = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2GRAY)
    low = float(np.percentile(luminance, max(0.0, low_percentile)))
    high = float(np.percentile(luminance, min(100.0, high_percentile)))
    if high - low < 1e-4:
        return image.astype(np.float32)
    scaled = (image - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Denoise
# --------------------------------------------------------------------------- #

def nlm_denoise(
    image: np.ndarray,
    strength: float = 6.0,
    chroma_strength: float = 8.0,
    template: int = 7,
    search: int = 21,
) -> np.ndarray:
    """Non-local means denoising via OpenCV.

    NLM averages pixels whose surrounding patches are similar, which preserves
    repeating structure such as text and brickwork far better than a local
    smoother. It cannot invent detail: every output sample is a weighted mean
    of measured input samples.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        strength: Luminance filter strength ``h``.
        chroma_strength: Chrominance filter strength.
        template: Patch edge length (odd).
        search: Search-window edge length (odd).
    """
    as_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    result = cv2.fastNlMeansDenoisingColored(
        as_uint8,
        None,
        float(max(0.1, strength)),
        float(max(0.1, chroma_strength)),
        int(max(3, template) | 1),
        int(max(5, search) | 1),
    )
    return (result.astype(np.float32) / 255.0).astype(np.float32)


def bilateral_denoise(
    image: np.ndarray, diameter: int = 7, sigma_color: float = 0.08,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """Edge-preserving bilateral smoothing.

    Faster than NLM and useful when the noise is mild, at the cost of a
    characteristic "watercolour" flattening of fine texture at high settings.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        diameter: Neighbourhood diameter.
        sigma_color: Colour sigma in ``[0, 1]`` units.
        sigma_space: Spatial sigma in pixels.
    """
    return np.clip(
        cv2.bilateralFilter(
            np.ascontiguousarray(image, dtype=np.float32),
            int(max(3, diameter)),
            float(max(0.001, sigma_color)),
            float(max(0.1, sigma_space)),
        ),
        0.0,
        1.0,
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# JPEG artefacts
# --------------------------------------------------------------------------- #

def jpeg_deblock(image: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Suppress 8x8 blocking with a boundary-selective smoother.

    A guided smoothing pass is computed over the whole frame, then blended in
    only along the JPEG block grid. Interior detail is therefore untouched,
    which distinguishes this from simply blurring the image.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        strength: Blend weight at block boundaries, in ``[0, 1]``.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return image.astype(np.float32)

    height, width = image.shape[:2]
    smoothed = cv2.bilateralFilter(
        np.ascontiguousarray(image, dtype=np.float32), 5, 0.06, 4.0
    )

    # Weight map: 1 on the block grid, tapering over two pixels either side.
    grid = np.zeros((height, width), dtype=np.float32)
    grid[7::8, :] = 1.0
    grid[8::8, :] = 1.0
    grid[:, 7::8] = 1.0
    grid[:, 8::8] = 1.0
    grid = cv2.GaussianBlur(grid, (0, 0), 0.9)
    grid = np.clip(grid / max(grid.max(), 1e-6), 0.0, 1.0) * strength

    weight = grid[..., None]
    return np.clip(image * (1.0 - weight) + smoothed * weight, 0.0, 1.0).astype(
        np.float32
    )


# --------------------------------------------------------------------------- #
# Dehaze
# --------------------------------------------------------------------------- #

def dark_channel_dehaze(
    image: np.ndarray,
    omega: float = 0.90,
    patch: int = 15,
    t_min: float = 0.12,
    guided_radius: int = 40,
    guided_epsilon: float = 1e-3,
) -> np.ndarray:
    """Single-image dehazing via the dark channel prior.

    Estimates the airlight and a transmission map, refines the map with a
    guided filter, then inverts the atmospheric scattering model
    ``I = J*t + A*(1-t)`` for the scene radiance ``J``.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        omega: Fraction of haze removed; below 1 keeps some aerial perspective.
        patch: Dark-channel minimum-filter window.
        t_min: Transmission floor, which bounds noise amplification in the
            densest regions.
        guided_radius: Guided-filter radius used to refine transmission.
        guided_epsilon: Guided-filter regularisation.

    Reference:
        He, Sun & Tang, CVPR 2009.
    """
    rgb = np.clip(image, 0.0, 1.0).astype(np.float32)
    size = max(3, int(patch) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    dark = cv2.erode(np.min(rgb, axis=2), kernel)

    # Airlight: brightest pixel among the top 0.1% of the dark channel.
    flat = dark.ravel()
    count = max(1, int(flat.size * 0.001))
    indices = np.argpartition(flat, -count)[-count:]
    candidates = rgb.reshape(-1, 3)[indices]
    airlight = candidates[
        int(np.argmax(candidates @ np.array([0.299, 0.587, 0.114], np.float32)))
    ]
    airlight = np.maximum(airlight, 0.05).astype(np.float32)

    normalised = np.clip(rgb / airlight[None, None, :], 0.0, 3.0)
    transmission = 1.0 - float(np.clip(omega, 0.05, 1.0)) * cv2.erode(
        np.min(normalised, axis=2), kernel
    )

    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    transmission = _guided_filter(
        guide, transmission, int(max(1, guided_radius)), float(guided_epsilon)
    )
    transmission = np.clip(transmission, float(np.clip(t_min, 0.02, 0.9)), 1.0)

    recovered = (rgb - airlight[None, None, :]) / transmission[
        ..., None
    ] + airlight[None, None, :]
    return np.clip(recovered, 0.0, 1.0).astype(np.float32)


def _guided_filter(
    guide: np.ndarray, source: np.ndarray, radius: int, epsilon: float
) -> np.ndarray:
    """Single-channel guided filter (He, Sun & Tang, ECCV 2010)."""
    window = (2 * radius + 1, 2 * radius + 1)
    mean_guide = cv2.blur(guide, window)
    mean_source = cv2.blur(source, window)
    mean_product = cv2.blur(guide * source, window)
    covariance = mean_product - mean_guide * mean_source

    mean_guide_sq = cv2.blur(guide * guide, window)
    variance = mean_guide_sq - mean_guide * mean_guide

    a = covariance / (variance + epsilon)
    b = mean_source - a * mean_guide

    mean_a = cv2.blur(a, window)
    mean_b = cv2.blur(b, window)
    return (mean_a * guide + mean_b).astype(np.float32)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #

def lanczos_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Resample by ``scale`` using a Lanczos-4 kernel.

    Lanczos is a windowed-sinc interpolator: it reconstructs the band-limited
    signal implied by the existing samples. It therefore produces a larger
    pixel grid without inventing any new spatial frequencies - the honest
    baseline against which a learned super-resolver should be compared.

    Args:
        image: ``HxWx3`` RGB float array in ``[0, 1]``.
        scale: Output scale factor.
    """
    scale = float(max(0.05, scale))
    height, width = image.shape[:2]
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_LANCZOS4 if scale >= 1.0 else cv2.INTER_AREA
    return np.clip(
        cv2.resize(image, target, interpolation=interpolation), 0.0, 1.0
    ).astype(np.float32)
