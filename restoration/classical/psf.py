"""Point-spread-function models used by the deconvolution operators."""

from __future__ import annotations

import numpy as np

__all__ = ["gaussian_psf", "motion_psf", "disk_psf", "normalise_psf"]


def normalise_psf(psf: np.ndarray) -> np.ndarray:
    """Return ``psf`` scaled to unit sum (energy preserving)."""
    total = float(psf.sum())
    if total <= 0.0:
        result = np.zeros_like(psf, dtype=np.float32)
        centre = tuple(dimension // 2 for dimension in psf.shape)
        result[centre] = 1.0
        return result
    return (psf / total).astype(np.float32)


def gaussian_psf(sigma: float, size: int = 0) -> np.ndarray:
    """Return an isotropic Gaussian PSF - the model for defocus-like softness.

    Args:
        sigma: Standard deviation in pixels.
        size: Kernel edge length; derived from ``sigma`` when 0.
    """
    sigma = max(0.1, float(sigma))
    if size <= 0:
        size = int(2 * np.ceil(3.0 * sigma) + 1)
    size = max(3, size | 1)
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel_1d = np.exp(-(axis ** 2) / (2.0 * sigma * sigma))
    return normalise_psf(np.outer(kernel_1d, kernel_1d))


def motion_psf(length: float, angle_deg: float) -> np.ndarray:
    """Return a linear-motion PSF.

    The kernel is drawn by supersampling the motion segment, which avoids the
    staircase aliasing a naive line-draw produces at oblique angles.

    Args:
        length: Motion extent in pixels.
        angle_deg: Direction of travel, degrees anticlockwise from horizontal.
    """
    length = max(1.0, float(length))
    size = int(2 * np.ceil(length / 2.0) + 1)
    size = max(3, size | 1)
    kernel = np.zeros((size, size), dtype=np.float32)

    centre = (size - 1) / 2.0
    theta = np.deg2rad(float(angle_deg))
    dx, dy = np.cos(theta), -np.sin(theta)

    samples = max(64, int(length * 16))
    for offset in np.linspace(-length / 2.0, length / 2.0, samples):
        x = centre + dx * offset
        y = centre + dy * offset
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if 0 <= yy < size and 0 <= xx < size:
                    kernel[yy, xx] += wy * wx

    return normalise_psf(kernel)


def disk_psf(radius: float) -> np.ndarray:
    """Return a uniform disk PSF - the physical model for circular defocus.

    Args:
        radius: Blur-circle radius in pixels.
    """
    radius = max(0.5, float(radius))
    size = int(2 * np.ceil(radius) + 1)
    size = max(3, size | 1)
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Supersample the boundary for a smooth edge.
    distance = np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)
    kernel = np.clip(radius + 0.5 - distance, 0.0, 1.0)
    return normalise_psf(kernel)
