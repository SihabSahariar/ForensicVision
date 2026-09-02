"""Numpy <-> Qt conversion helpers and small image utilities.

This is the only module in :mod:`core` that imports PyQt5; it is kept separate
so the rest of the engine remains GUI-free.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

from core.image_io import ImageData

logger = logging.getLogger(__name__)

__all__ = [
    "numpy_to_qimage",
    "image_to_qimage",
    "image_to_qpixmap",
    "qimage_to_numpy",
    "make_thumbnail",
    "resize_to_fit",
    "ensure_uint8_rgb",
    "sample_pixel",
]


def ensure_uint8_rgb(array: np.ndarray) -> np.ndarray:
    """Coerce any supported array into contiguous 8-bit RGB or RGBA.

    Args:
        array: ``HxW``, ``HxWx3`` or ``HxWx4`` array of any supported dtype.

    Returns:
        A C-contiguous ``uint8`` array with 3 or 4 channels.
    """
    data = array
    if data.dtype == np.uint16:
        data = (data.astype(np.float32) / 257.0).round().astype(np.uint8)
    elif data.dtype in (np.float32, np.float64):
        data = (np.clip(data, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)

    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    elif data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    elif data.shape[2] == 2:
        gray = data[..., 0]
        data = np.dstack([gray, gray, gray, data[..., 1]])
    return np.ascontiguousarray(data)


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convert a numpy array to a deep-copied :class:`QImage`.

    The copy is deliberate: Qt does not take ownership of the numpy buffer, and
    a view into a temporary array is the single most common source of crashes
    in PyQt image code.
    """
    data = ensure_uint8_rgb(array)
    height, width = data.shape[:2]
    if data.shape[2] == 4:
        fmt = QImage.Format_RGBA8888
        bytes_per_line = 4 * width
    else:
        fmt = QImage.Format_RGB888
        bytes_per_line = 3 * width
    image = QImage(data.data, width, height, bytes_per_line, fmt)
    return image.copy()


def image_to_qimage(image: ImageData) -> QImage:
    """Convert an :class:`~core.image_io.ImageData` to a :class:`QImage`."""
    return numpy_to_qimage(image.pixels)


def image_to_qpixmap(image: ImageData) -> QPixmap:
    """Convert an :class:`~core.image_io.ImageData` to a :class:`QPixmap`."""
    return QPixmap.fromImage(image_to_qimage(image))


def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    """Convert a :class:`QImage` to an ``HxWx3``/``HxWx4`` ``uint8`` array."""
    has_alpha = qimage.hasAlphaChannel()
    fmt = QImage.Format_RGBA8888 if has_alpha else QImage.Format_RGB888
    converted = qimage.convertToFormat(fmt)
    width, height = converted.width(), converted.height()
    channels = 4 if has_alpha else 3
    ptr = converted.constBits()
    ptr.setsize(converted.byteCount())
    stride = converted.bytesPerLine()
    buffer = np.frombuffer(ptr, dtype=np.uint8, count=height * stride)
    array = buffer.reshape(height, stride)[:, : width * channels]
    return np.ascontiguousarray(array.reshape(height, width, channels))


def resize_to_fit(array: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale ``array`` so its longest edge is at most ``max_edge``.

    Images already smaller than ``max_edge`` are returned unchanged (no
    upscaling, which would fabricate detail).
    """
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_edge or longest == 0:
        return array
    scale = max_edge / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(array, new_size, interpolation=cv2.INTER_AREA)


def make_thumbnail(image: ImageData, max_edge: int = 256) -> np.ndarray:
    """Return an 8-bit RGB thumbnail of ``image`` for lists and reports."""
    return resize_to_fit(ensure_uint8_rgb(image.pixels), max_edge)


def sample_pixel(
    image: ImageData, x: int, y: int
) -> Optional[Tuple[Tuple[int, ...], int, Tuple[int, int, int]]]:
    """Sample one pixel for the viewer's readout.

    Args:
        image: Source image.
        x: Column index in image coordinates.
        y: Row index in image coordinates.

    Returns:
        ``(rgb_or_rgba, gray, hsv)`` with 8-bit component values, or ``None``
        when the coordinates fall outside the image.
    """
    if not (0 <= x < image.width and 0 <= y < image.height):
        return None
    data = ensure_uint8_rgb(image.pixels)
    px = data[y, x]
    rgb = tuple(int(v) for v in px[:3])
    components = tuple(int(v) for v in px)
    gray = int(round(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]))
    hsv_pixel = cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2HSV)[0][0]
    hsv = (int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2]))
    return components, gray, hsv
