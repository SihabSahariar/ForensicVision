"""Tiled inference with feathered blending and out-of-memory backoff.

Large evidence images routinely exceed the VRAM of a mid-range GPU. This module
splits the input into overlapping tiles, processes each independently and
recombines them with a raised-cosine weight ramp so no seam is visible in the
output - which matters here because a seam would be an artefact introduced by
the tool, indistinguishable at a glance from an artefact in the evidence.

If the device runs out of memory the tile size is halved and the whole pass is
retried, down to :data:`app.constants.MIN_TILE_SIZE`.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from app.constants import MIN_TILE_SIZE
from core.exceptions import OperationCancelled, OutOfMemoryError

logger = logging.getLogger(__name__)

__all__ = [
    "tiled_process",
    "pad_to_multiple",
    "is_out_of_memory",
    "estimate_tile_size",
]

TileFunction = Callable[[np.ndarray], np.ndarray]
ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


def is_out_of_memory(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` represents a device out-of-memory event."""
    try:
        import torch  # noqa: PLC0415

        if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
            return True
    except Exception:  # pragma: no cover - torch optional
        pass
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "cublas_status_alloc_failed" in text
    )


def pad_to_multiple(
    image: np.ndarray, multiple: int, mode: str = "reflect"
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad ``image`` so both spatial dimensions are multiples of ``multiple``.

    Reflection padding is used because zero padding creates a hard synthetic
    edge that several architectures turn into a visible border artefact.

    Args:
        image: ``HxWxC`` array.
        multiple: Required divisor; 1 disables padding.
        mode: Numpy pad mode.

    Returns:
        ``(padded_image, (pad_height, pad_width))``.
    """
    if multiple <= 1:
        return image, (0, 0)
    height, width = image.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    # ``reflect`` requires the pad to be smaller than the dimension.
    effective = mode if (pad_h < height and pad_w < width) else "edge"
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=effective)
    return padded, (pad_h, pad_w)


def _blend_window(height: int, width: int, ramp: int) -> np.ndarray:
    """Return a raised-cosine weight window for one tile.

    The window is 1 in the tile interior and falls smoothly to near 0 across
    ``ramp`` pixels at each edge, so overlapping tiles sum to a flat response.
    """
    def _axis(length: int) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        span = min(ramp, length // 2)
        if span <= 0:
            return weights
        taper = 0.5 - 0.5 * np.cos(
            np.linspace(0.0, np.pi, span + 2, dtype=np.float32)[1:-1]
        )
        weights[:span] = taper
        weights[-span:] = taper[::-1]
        return np.maximum(weights, 1e-3)

    return np.outer(_axis(height), _axis(width)).astype(np.float32)


def estimate_tile_size(
    available_mb: float, scale: int = 1, channels: int = 3, safety: float = 0.35
) -> int:
    """Suggest a tile edge length that should fit in ``available_mb``.

    The estimate assumes a network holds on the order of 40 float32 feature
    maps at tile resolution; it is intentionally conservative because the cost
    of an over-large tile is a full OOM retry.

    Args:
        available_mb: Free device memory in MiB.
        scale: Output scale factor of the model.
        channels: Channel count.
        safety: Fraction of free memory to actually target.

    Returns:
        A tile edge length, clamped to at least :data:`MIN_TILE_SIZE`.
    """
    budget_bytes = max(0.0, available_mb) * 1024 * 1024 * safety
    per_pixel = 4.0 * channels * 40.0 * (1.0 + scale * scale)
    if per_pixel <= 0:
        return 512
    pixels = budget_bytes / per_pixel
    edge = int(np.sqrt(max(pixels, 1.0)))
    edge = max(MIN_TILE_SIZE, min(1024, (edge // 32) * 32))
    return edge


def tiled_process(
    image: np.ndarray,
    function: TileFunction,
    scale: int = 1,
    tile_size: int = 512,
    overlap: int = 32,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCheck] = None,
    auto_reduce: bool = True,
    message: str = "Processing",
) -> np.ndarray:
    """Run ``function`` over ``image`` in overlapping tiles.

    Args:
        image: ``HxWx3`` ``float32`` array in ``[0, 1]``.
        function: Callable applied to each tile, returning a tile scaled by
            ``scale``.
        scale: Output scale factor of ``function``.
        tile_size: Tile edge length in input pixels; 0 or negative processes
            the whole image in one pass.
        overlap: Overlap between neighbouring tiles, in input pixels.
        progress: Optional ``(percent, message)`` callback.
        cancelled: Optional predicate polled between tiles.
        auto_reduce: Halve the tile size and retry after an OOM.
        message: Progress message prefix.

    Returns:
        The recombined ``(H*scale)x(W*scale)x3`` output.

    Raises:
        OperationCancelled: ``cancelled`` returned ``True``.
        OutOfMemoryError: Memory was exhausted even at the minimum tile size.
    """
    height, width = image.shape[:2]

    if tile_size <= 0 or (tile_size >= width and tile_size >= height):
        if progress is not None:
            progress(5, f"{message} (single pass)")
        try:
            result = function(image)
        except Exception as exc:
            if not (is_out_of_memory(exc) and auto_reduce):
                raise
            # Fall back to tiling. The fallback size must be strictly smaller
            # than the image, otherwise this branch is re-entered and recurses
            # without bound on an image smaller than the nominal tile size.
            fallback = min(512, max(width, height) // 2)
            fallback = (fallback // 32) * 32
            if fallback < MIN_TILE_SIZE:
                raise OutOfMemoryError(
                    "The device ran out of memory on an image too small to "
                    f"tile further ({width}x{height}). Try the CPU device."
                ) from exc
            logger.warning(
                "Out of memory in single-pass mode; retrying with %d px tiles",
                fallback,
            )
            _free_device_memory()
            return tiled_process(
                image,
                function,
                scale=scale,
                tile_size=fallback,
                overlap=overlap,
                progress=progress,
                cancelled=cancelled,
                auto_reduce=auto_reduce,
                message=message,
            )
        if progress is not None:
            progress(100, f"{message} complete")
        return result

    current_tile = max(MIN_TILE_SIZE, int(tile_size))
    while True:
        try:
            return _run_tiles(
                image,
                function,
                scale=scale,
                tile_size=current_tile,
                overlap=overlap,
                progress=progress,
                cancelled=cancelled,
                message=message,
            )
        except OperationCancelled:
            raise
        except Exception as exc:
            if not (auto_reduce and is_out_of_memory(exc)):
                raise
            _free_device_memory()
            if current_tile <= MIN_TILE_SIZE:
                raise OutOfMemoryError(
                    "The device ran out of memory even at the minimum tile size "
                    f"({MIN_TILE_SIZE} px). Try the CPU device, or process a "
                    "smaller region of interest."
                ) from exc
            current_tile = max(MIN_TILE_SIZE, current_tile // 2)
            logger.warning(
                "Out of memory; retrying with tile size %d px", current_tile
            )
            if progress is not None:
                progress(0, f"Out of memory - retrying at {current_tile} px tiles")


def _run_tiles(
    image: np.ndarray,
    function: TileFunction,
    scale: int,
    tile_size: int,
    overlap: int,
    progress: Optional[ProgressCallback],
    cancelled: Optional[CancelCheck],
    message: str,
) -> np.ndarray:
    """Execute one full tiled pass at a fixed tile size."""
    height, width, channels = image.shape
    overlap = max(0, min(int(overlap), tile_size // 2 - 1)) if tile_size > 2 else 0
    stride = max(1, tile_size - overlap)

    xs = list(range(0, max(1, width - overlap), stride))
    ys = list(range(0, max(1, height - overlap), stride))
    # Drop starts that would produce a tile fully inside the previous one.
    xs = [x for x in xs if x == 0 or x < width - overlap]
    ys = [y for y in ys if y == 0 or y < height - overlap]

    out_h, out_w = height * scale, width * scale
    accumulator = np.zeros((out_h, out_w, channels), dtype=np.float32)
    weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

    total = len(xs) * len(ys)
    done = 0
    logger.debug(
        "Tiled pass: %dx%d image, %d tiles of %d px (overlap %d, scale %d)",
        width, height, total, tile_size, overlap, scale,
    )

    for y0 in ys:
        for x0 in xs:
            if cancelled is not None and cancelled():
                raise OperationCancelled("Processing cancelled by the user")

            x1 = min(x0 + tile_size, width)
            y1 = min(y0 + tile_size, height)
            # Keep tiles full-size where possible so networks with a minimum
            # receptive field behave consistently near the right/bottom edges.
            x0e = max(0, x1 - tile_size)
            y0e = max(0, y1 - tile_size)

            tile = np.ascontiguousarray(image[y0e:y1, x0e:x1])
            output = function(tile)

            expected = ((y1 - y0e) * scale, (x1 - x0e) * scale)
            if output.shape[:2] != expected:
                raise ValueError(
                    f"Tile function returned {output.shape[:2]}, expected {expected}"
                )

            window = _blend_window(
                output.shape[0], output.shape[1], max(4, overlap * scale // 2)
            )[..., None]

            oy, ox = y0e * scale, x0e * scale
            accumulator[oy : oy + output.shape[0], ox : ox + output.shape[1]] += (
                output.astype(np.float32) * window
            )
            weights[oy : oy + output.shape[0], ox : ox + output.shape[1]] += window

            done += 1
            if progress is not None:
                percent = int(done * 100 / max(1, total))
                progress(percent, f"{message}: tile {done}/{total}")

    np.maximum(weights, 1e-6, out=weights)
    result = accumulator / weights
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _free_device_memory() -> None:
    """Best-effort release of cached CUDA memory between retries."""
    try:
        from core.device import empty_cache  # noqa: PLC0415

        empty_cache()
    except Exception:  # pragma: no cover - defensive
        pass
