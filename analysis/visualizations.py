"""Forensic visualisation tools.

Every function here renders an *analytical view* of an image. None of them
modifies evidence, and none should be presented as an enhanced image: they are
instruments for reading what is already in the data. The GUI labels each view
accordingly.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import cv2
import numpy as np

from analysis.base import to_gray
from core.image_utils import ensure_uint8_rgb
from gui.theme import Palette

logger = logging.getLogger(__name__)

__all__ = ["VISUALIZATIONS", "render_visualization", "VISUALIZATION_LABELS"]


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit RGB view of ``image``."""
    return ensure_uint8_rgb(image)[..., :3]


def _gray_to_rgb(plane: np.ndarray) -> np.ndarray:
    """Stack a single plane into an RGB image."""
    if plane.dtype != np.uint8:
        plane = np.clip(plane, 0, 255).astype(np.uint8)
    return np.stack([plane] * 3, axis=-1)


def _plot_histogram(
    channels: Dict[str, np.ndarray],
    width: int = 900,
    height: int = 420,
    log_scale: bool = False,
) -> np.ndarray:
    """Render histogram curves onto a dark canvas.

    Drawn with OpenCV rather than matplotlib so the visualisation appears
    instantly in the viewer and needs no extra backend on a headless build.
    """
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    canvas[:] = (int(Palette.VIEWPORT[5:7], 16),
                 int(Palette.VIEWPORT[3:5], 16),
                 int(Palette.VIEWPORT[1:3], 16))[::-1]

    grid_colour = (44, 50, 61)
    for fraction in range(1, 4):
        y = int(height * fraction / 4)
        cv2.line(canvas, (0, y), (width, y), grid_colour, 1)
    for fraction in range(1, 4):
        x = int(width * fraction / 4)
        cv2.line(canvas, (x, 0), (x, height), grid_colour, 1)

    colours = {
        "red": Palette.CH_R, "green": Palette.CH_G,
        "blue": Palette.CH_B, "luminance": Palette.CH_K,
    }

    peak = 1.0
    computed: Dict[str, np.ndarray] = {}
    for name, plane in channels.items():
        histogram, _ = np.histogram(plane, bins=256, range=(0, 256))
        values = histogram.astype(np.float64)
        if log_scale:
            values = np.log10(values + 1.0)
        computed[name] = values
        peak = max(peak, float(values.max()))

    for name, values in computed.items():
        hex_colour = colours.get(name, Palette.CH_K).lstrip("#")
        colour = (
            int(hex_colour[4:6], 16), int(hex_colour[2:4], 16), int(hex_colour[0:2], 16)
        )
        points = []
        for index, value in enumerate(values):
            x = int(index * (width - 1) / 255)
            y = int(height - 1 - (value / peak) * (height - 24))
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points, np.int32)], False, colour, 2, cv2.LINE_AA)

    label = "Logarithmic count" if log_scale else "Linear count"
    cv2.putText(canvas, f"0                                    Level                                    255",
                (10, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (154, 163, 178), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (154, 163, 178), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Individual visualisations
# --------------------------------------------------------------------------- #

def rgb_histogram(image: np.ndarray) -> np.ndarray:
    """Per-channel RGB histogram."""
    rgb = _as_rgb(image)
    return _plot_histogram(
        {"red": rgb[..., 0], "green": rgb[..., 1], "blue": rgb[..., 2]}
    )


def luminance_histogram(image: np.ndarray) -> np.ndarray:
    """Luminance histogram on a logarithmic count axis."""
    gray = (to_gray(image) * 255.0).astype(np.uint8)
    return _plot_histogram({"luminance": gray}, log_scale=True)


def grayscale_view(image: np.ndarray) -> np.ndarray:
    """BT.601 luminance rendering."""
    return _gray_to_rgb((to_gray(image) * 255.0).round())


def edge_map(image: np.ndarray) -> np.ndarray:
    """Canny edge map."""
    gray = (to_gray(image) * 255.0).astype(np.uint8)
    return _gray_to_rgb(cv2.Canny(gray, 60, 160))


def high_pass(image: np.ndarray) -> np.ndarray:
    """High-pass residual, centred at mid grey.

    Reveals sharpening halos, resampling ringing and splice boundaries that are
    invisible at normal display gain.
    """
    gray = to_gray(image)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    scaled = np.clip(residual * 6.0 + 0.5, 0.0, 1.0) * 255.0
    return _gray_to_rgb(scaled.round())


def noise_residual(image: np.ndarray) -> np.ndarray:
    """Median-filter residual, amplified.

    Structure visible here that does not follow scene content can indicate
    local processing or compositing.
    """
    gray = (to_gray(image) * 255.0).astype(np.uint8)
    denoised = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, denoised).astype(np.float32)
    scaled = np.clip(residual * 10.0, 0, 255)
    return _gray_to_rgb(scaled.round())


def saturation_map(image: np.ndarray) -> np.ndarray:
    """HSV saturation rendered as a heatmap."""
    rgb = _as_rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    coloured = cv2.applyColorMap(hsv[..., 1], cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def exposure_map(image: np.ndarray) -> np.ndarray:
    """Luminance zones rendered with a perceptual colour map."""
    gray = (to_gray(image) * 255.0).astype(np.uint8)
    coloured = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def clipping_map(image: np.ndarray) -> np.ndarray:
    """Highlight clipped samples: red for blown, blue for crushed.

    Clipped samples carry no recoverable information; anything that appears in
    these regions after enhancement is necessarily synthesised.
    """
    rgb = _as_rgb(image)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)

    gray = (to_gray(image) * 255.0).astype(np.uint8)
    canvas = np.stack([gray // 2] * 3, axis=-1)

    blown = channel_max >= 253
    crushed = channel_min <= 2
    canvas[blown] = (230, 70, 70)
    canvas[crushed] = (70, 110, 230)
    return canvas


def frequency_spectrum(image: np.ndarray) -> np.ndarray:
    """Log-magnitude Fourier spectrum, DC centred.

    Regular bright spikes indicate periodic structure (sensor pattern, halftone
    screens, JPEG block grids); an abrupt circular cut-off indicates the image
    was interpolated up from a smaller original.
    """
    gray = to_gray(image)
    height, width = gray.shape[:2]
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray * window)))
    magnitude = np.log1p(spectrum)
    normalised = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    coloured = cv2.applyColorMap(normalised.astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def error_level_analysis(image: np.ndarray) -> np.ndarray:
    """Error Level Analysis: difference after a controlled JPEG re-encode.

    Regions that have been through a different number of compression cycles
    respond differently to re-encoding. ELA is suggestive only - it is heavily
    confounded by local contrast and texture, and is not proof of manipulation.
    """
    rgb = _as_rgb(image)
    encoded = cv2.imencode(
        ".jpg", rgb[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
    difference = cv2.absdiff(rgb, decoded).astype(np.float32)
    peak = float(difference.max())
    scale = 255.0 / peak if peak > 0 else 1.0
    return np.clip(difference * scale, 0, 255).astype(np.uint8)


#: Registry of visualisation name -> renderer.
VISUALIZATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rgb_histogram": rgb_histogram,
    "luminance_histogram": luminance_histogram,
    "grayscale": grayscale_view,
    "edge_map": edge_map,
    "high_pass": high_pass,
    "noise_residual": noise_residual,
    "saturation_map": saturation_map,
    "exposure_map": exposure_map,
    "clipping_map": clipping_map,
    "frequency_spectrum": frequency_spectrum,
    "error_level_analysis": error_level_analysis,
}

#: Menu labels for each visualisation.
VISUALIZATION_LABELS: Dict[str, str] = {
    "rgb_histogram": "RGB Histogram",
    "luminance_histogram": "Luminance Histogram",
    "grayscale": "Grayscale",
    "edge_map": "Edge Map",
    "high_pass": "High Pass",
    "noise_residual": "Noise Residual",
    "saturation_map": "Saturation Map",
    "exposure_map": "Exposure Map",
    "clipping_map": "Clipping Map",
    "frequency_spectrum": "Frequency Spectrum",
    "error_level_analysis": "Error Level Analysis",
}

#: Per-visualisation caveats shown alongside the rendering.
VISUALIZATION_NOTES: Dict[str, str] = {
    "high_pass": (
        "High-pass residual amplified 6x. Halos around edges indicate prior "
        "sharpening; regular ripples indicate resampling."
    ),
    "noise_residual": (
        "Median-filter residual amplified 10x. Areas whose noise structure "
        "differs from their surroundings warrant closer inspection."
    ),
    "clipping_map": (
        "Red: at least one channel at the white point. Blue: at least one "
        "channel at the black point. These samples carry no recoverable "
        "information."
    ),
    "frequency_spectrum": (
        "Log-magnitude Fourier spectrum. A sharp circular cut-off indicates "
        "the frame was interpolated up from a smaller original."
    ),
    "error_level_analysis": (
        "ELA is suggestive only. It is strongly influenced by local contrast "
        "and texture and must not be treated as evidence of manipulation."
    ),
}


def render_visualization(name: str, image: np.ndarray) -> np.ndarray:
    """Render the named visualisation.

    Args:
        name: A key from :data:`VISUALIZATIONS`.
        image: Source array; never modified.

    Returns:
        An 8-bit RGB rendering.

    Raises:
        KeyError: The visualisation is unknown.
    """
    renderer = VISUALIZATIONS[name]
    return renderer(image)
