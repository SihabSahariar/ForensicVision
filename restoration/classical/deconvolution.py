"""Deterministic deconvolution.

Both algorithms here invert an explicit, examiner-chosen blur model. Unlike a
learned deblurring network they cannot introduce structures that are not
derivable from the measured samples: the output is a constrained inverse of the
input under the stated PSF. They will, however, amplify noise and can produce
ringing, and neither is capable of recovering spatial frequencies that the blur
drove to zero.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["richardson_lucy", "wiener_deconvolution", "unsharp_mask"]


def _to_odd_pad(image: np.ndarray, psf: np.ndarray) -> int:
    """Return the border width needed to avoid wrap-around artefacts."""
    return max(psf.shape[0], psf.shape[1])


def richardson_lucy(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 20,
    damping: float = 0.0,
    progress: Optional[Callable[[int, str], None]] = None,
) -> np.ndarray:
    """Richardson-Lucy deconvolution of a float image in ``[0, 1]``.

    The algorithm is the maximum-likelihood solution for Poisson-distributed
    observations and is iterative: each step multiplies the current estimate by
    the correlation of the PSF with the ratio between the observation and the
    current re-blurred estimate.

    Args:
        image: ``HxW`` or ``HxWxC`` float array in ``[0, 1]``.
        psf: Unit-sum point spread function.
        iterations: Number of RL iterations. More iterations sharpen further
            but amplify noise and ringing without bound.
        damping: Blend factor in ``[0, 1)`` applied to each update, trading
            sharpness for noise stability. 0 is the classical algorithm.
        progress: Optional ``(percent, message)`` callback.

    Returns:
        The deconvolved image, same shape as ``image``.

    Reference:
        Richardson, JOSA 1972; Lucy, AJ 1974.
    """
    iterations = max(1, int(iterations))
    damping = float(np.clip(damping, 0.0, 0.95))
    border = _to_odd_pad(image, psf)

    padded = cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_REFLECT_101
    ).astype(np.float32)
    if padded.ndim == 2:
        padded = padded[..., None]

    psf_flipped = psf[::-1, ::-1].copy()
    epsilon = 1e-7
    estimate = np.clip(padded, epsilon, None).copy()

    for step in range(iterations):
        reblurred = np.empty_like(estimate)
        for channel in range(estimate.shape[2]):
            reblurred[..., channel] = cv2.filter2D(
                estimate[..., channel], -1, psf, borderType=cv2.BORDER_REFLECT_101
            )
        np.maximum(reblurred, epsilon, out=reblurred)
        ratio = padded / reblurred

        correction = np.empty_like(ratio)
        for channel in range(ratio.shape[2]):
            correction[..., channel] = cv2.filter2D(
                ratio[..., channel], -1, psf_flipped,
                borderType=cv2.BORDER_REFLECT_101,
            )

        if damping > 0.0:
            correction = 1.0 + (correction - 1.0) * (1.0 - damping)

        estimate *= correction
        np.clip(estimate, 0.0, 1.5, out=estimate)

        if progress is not None:
            progress(
                int((step + 1) * 100 / iterations),
                f"Richardson-Lucy iteration {step + 1}/{iterations}",
            )

    result = estimate[border:-border, border:-border]
    if image.ndim == 2:
        result = result[..., 0]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def wiener_deconvolution(
    image: np.ndarray, psf: np.ndarray, noise_to_signal: float = 0.01
) -> np.ndarray:
    """Regularised inverse filtering in the frequency domain.

    The Wiener filter is the linear minimum-mean-square-error inverse under a
    stationary noise assumption. It is a single non-iterative pass, so it is far
    faster than Richardson-Lucy and cannot diverge, but it is also softer.

    Args:
        image: ``HxW`` or ``HxWxC`` float array in ``[0, 1]``.
        psf: Unit-sum point spread function.
        noise_to_signal: The regularisation term. Larger values suppress noise
            amplification at the cost of sharpness.

    Returns:
        The deconvolved image, same shape as ``image``.
    """
    noise_to_signal = float(max(1e-6, noise_to_signal))
    border = _to_odd_pad(image, psf)
    padded = cv2.copyMakeBorder(
        image, border, border, border, border, cv2.BORDER_REFLECT_101
    ).astype(np.float32)
    if padded.ndim == 2:
        padded = padded[..., None]

    height, width = padded.shape[:2]

    # Embed the PSF in a full-size, origin-centred array so its transform has
    # zero phase; otherwise the result is shifted by half the kernel size.
    otf_kernel = np.zeros((height, width), dtype=np.float32)
    kh, kw = psf.shape
    otf_kernel[:kh, :kw] = psf
    otf_kernel = np.roll(otf_kernel, -(kh // 2), axis=0)
    otf_kernel = np.roll(otf_kernel, -(kw // 2), axis=1)
    otf = np.fft.rfft2(otf_kernel)

    denominator = (otf.conj() * otf).real + noise_to_signal
    filter_kernel = otf.conj() / denominator

    result = np.empty_like(padded)
    for channel in range(padded.shape[2]):
        spectrum = np.fft.rfft2(padded[..., channel])
        restored = np.fft.irfft2(spectrum * filter_kernel, s=(height, width))
        result[..., channel] = restored

    result = result[border:-border, border:-border]
    if image.ndim == 2:
        result = result[..., 0]
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def unsharp_mask(
    image: np.ndarray, radius: float = 1.5, amount: float = 0.8, threshold: float = 0.0
) -> np.ndarray:
    """Classical unsharp masking.

    This increases acutance by boosting existing local contrast. It does not
    recover lost detail and cannot add information; over-application produces
    halos at edges that must not be mistaken for scene content.

    Args:
        image: ``HxWxC`` float array in ``[0, 1]``.
        radius: Gaussian radius of the blur used to form the mask.
        amount: Strength of the boost.
        threshold: Minimum local contrast (in ``[0, 1]``) before sharpening is
            applied, which prevents amplifying flat-area noise.
    """
    radius = max(0.1, float(radius))
    blurred = cv2.GaussianBlur(image, (0, 0), radius, borderType=cv2.BORDER_REFLECT_101)
    detail = image - blurred
    if threshold > 0.0:
        mask = (np.abs(detail) >= float(threshold)).astype(np.float32)
        detail = detail * mask
    return np.clip(image + float(amount) * detail, 0.0, 1.0).astype(np.float32)
