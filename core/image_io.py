"""Image loading and saving with forensic fidelity guarantees.

Design rules:

* Pixel data is kept in its native bit depth (``uint8``/``uint16``) whenever the
  container supports it. Nothing is silently converted to 8-bit.
* Channel order in memory is **RGB / RGBA / grayscale** - never OpenCV's BGR.
  Conversions to BGR happen only at the OpenCV call boundary.
* Alpha is preserved end to end.
* Source evidence is never rewritten. Exports always target a new path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from app.constants import SUPPORTED_IMAGE_EXTENSIONS
from core.exceptions import ImageLoadError, ImageSaveError, UnsupportedFormatError

logger = logging.getLogger(__name__)

__all__ = ["ImageData", "load_image", "save_image", "is_supported_path"]


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #

@dataclass
class ImageData:
    """An in-memory image plus the provenance needed to round-trip it.

    Attributes:
        pixels: ``HxW`` (gray), ``HxWx3`` (RGB) or ``HxWx4`` (RGBA) array with
            dtype ``uint8``, ``uint16`` or ``float32``.
        source_path: Where the image came from, if it was read from disk.
        source_format: Upper-case container name, e.g. ``"JPEG"``.
        icc_profile: Raw ICC bytes when the decoder exposed them.
        extra: Free-form decoder metadata (EXIF blobs, DPI, ...).
    """

    pixels: np.ndarray
    source_path: Optional[Path] = None
    source_format: Optional[str] = None
    icc_profile: Optional[bytes] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- geometry ----------------------------------------------------------- #
    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.pixels.shape[1])

    @property
    def channels(self) -> int:
        """Number of channels (1, 3 or 4)."""
        return 1 if self.pixels.ndim == 2 else int(self.pixels.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """``(height, width, channels)`` triple."""
        return self.height, self.width, self.channels

    @property
    def dtype(self) -> np.dtype:
        """Underlying numpy dtype."""
        return self.pixels.dtype

    @property
    def bit_depth(self) -> int:
        """Bits per channel implied by :attr:`dtype`."""
        if self.pixels.dtype == np.uint8:
            return 8
        if self.pixels.dtype == np.uint16:
            return 16
        return 32

    @property
    def has_alpha(self) -> bool:
        """Whether an alpha channel is present."""
        return self.channels == 4

    @property
    def is_gray(self) -> bool:
        """Whether the image is single-channel."""
        return self.channels == 1

    @property
    def max_value(self) -> float:
        """Maximum representable sample value for :attr:`dtype`."""
        if self.pixels.dtype == np.uint8:
            return 255.0
        if self.pixels.dtype == np.uint16:
            return 65535.0
        return 1.0

    # -- conversions -------------------------------------------------------- #
    def copy(self) -> "ImageData":
        """Return a deep copy, sharing no buffers with ``self``."""
        return replace(self, pixels=self.pixels.copy(), extra=dict(self.extra))

    def with_pixels(self, pixels: np.ndarray) -> "ImageData":
        """Return a copy of ``self`` carrying ``pixels`` instead.

        Provenance fields (source path/format/ICC) are preserved so a derivative
        can still report what it was derived from.
        """
        return replace(self, pixels=pixels, extra=dict(self.extra))

    def split_alpha(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split colour and alpha planes.

        Returns:
            ``(colour, alpha)`` where ``alpha`` is ``None`` for opaque images.
        """
        if self.has_alpha:
            return self.pixels[..., :3], self.pixels[..., 3]
        return self.pixels, None

    def to_float_rgb(self) -> np.ndarray:
        """Return an ``HxWx3`` ``float32`` array normalised to ``[0, 1]``.

        Grayscale input is replicated across three channels and alpha is
        dropped; use :meth:`split_alpha` first if alpha must be retained.
        """
        colour, _ = self.split_alpha()
        arr = colour.astype(np.float32) / float(self.max_value)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return np.clip(arr, 0.0, 1.0)

    def to_uint8_rgb(self) -> np.ndarray:
        """Return an ``HxWx3`` ``uint8`` RGB view suitable for display."""
        arr = self.to_float_rgb()
        return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def to_gray_float(self) -> np.ndarray:
        """Return an ``HxW`` ``float32`` luminance plane in ``[0, 1]``."""
        colour, _ = self.split_alpha()
        arr = colour.astype(np.float32) / float(self.max_value)
        if arr.ndim == 2:
            return np.clip(arr, 0.0, 1.0)
        return np.clip(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 0.0, 1.0)

    def describe(self) -> Dict[str, Any]:
        """Return a small dictionary used by the metadata panel and reports."""
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "dtype": str(self.dtype),
            "has_alpha": self.has_alpha,
            "format": self.source_format,
            "megapixels": round(self.width * self.height / 1_000_000.0, 3),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def is_supported_path(path: os.PathLike | str) -> bool:
    """Return ``True`` when ``path`` has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _format_from_suffix(suffix: str) -> str:
    mapping = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".jpe": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }
    return mapping.get(suffix.lower(), suffix.lstrip(".").upper() or "UNKNOWN")


def _imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    """``cv2.imread`` replacement that tolerates non-ASCII Windows paths."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, flags)


def _bgr_to_rgb(array: np.ndarray) -> np.ndarray:
    """Reorder an OpenCV-decoded array into RGB/RGBA channel order."""
    if array.ndim == 2:
        return array
    if array.shape[2] == 3:
        return array[..., ::-1].copy()
    if array.shape[2] == 4:
        rgb = array[..., 2::-1]
        alpha = array[..., 3:4]
        return np.concatenate([rgb, alpha], axis=-1).copy()
    return array


def _rgb_to_bgr(array: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_bgr_to_rgb`."""
    return _bgr_to_rgb(array)


def _load_with_pillow(path: Path) -> Optional[ImageData]:
    """Fallback decoder used when OpenCV refuses a file."""
    try:
        from PIL import Image  # noqa: PLC0415 - optional path
    except ImportError:  # pragma: no cover - Pillow is a hard requirement
        return None
    try:
        with Image.open(path) as img:
            img.load()
            icc = img.info.get("icc_profile")
            mode = img.mode
            if mode not in ("L", "RGB", "RGBA", "I;16", "I;16B"):
                img = img.convert("RGBA" if "A" in mode else "RGB")
                mode = img.mode
            array = np.array(img)
            extra: Dict[str, Any] = {}
            if "dpi" in img.info:
                extra["dpi"] = img.info["dpi"]
            return ImageData(
                pixels=array,
                source_path=path,
                source_format=(img.format or _format_from_suffix(path.suffix)).upper(),
                icc_profile=icc,
                extra=extra,
            )
    except Exception:
        logger.debug("Pillow could not decode %s", path, exc_info=True)
        return None


def load_image(path: os.PathLike | str, allow_unsupported: bool = False) -> ImageData:
    """Decode an image from disk preserving bit depth and alpha.

    Args:
        path: File to read.
        allow_unsupported: Attempt decoding even for unknown extensions.

    Returns:
        The decoded :class:`ImageData`.

    Raises:
        UnsupportedFormatError: Extension is outside the supported set.
        ImageLoadError: The file is missing, empty or undecodable.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ImageLoadError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ImageLoadError(f"Not a regular file: {file_path}")
    if not allow_unsupported and not is_supported_path(file_path):
        raise UnsupportedFormatError(
            f"Unsupported image extension '{file_path.suffix}'. Supported: "
            + ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        )

    raw = _imread_unicode(file_path, cv2.IMREAD_UNCHANGED)
    if raw is not None and raw.size > 0:
        pixels = _bgr_to_rgb(raw)
        if pixels.ndim == 3 and pixels.shape[2] == 2:
            # Gray + alpha: OpenCV reports this rarely; normalise to RGBA.
            gray = pixels[..., 0]
            alpha = pixels[..., 1]
            pixels = np.dstack([gray, gray, gray, alpha])
        image = ImageData(
            pixels=np.ascontiguousarray(pixels),
            source_path=file_path,
            source_format=_format_from_suffix(file_path.suffix),
        )
        logger.debug(
            "Loaded %s via OpenCV: %dx%d, %d ch, %d-bit",
            file_path.name,
            image.width,
            image.height,
            image.channels,
            image.bit_depth,
        )
        return image

    fallback = _load_with_pillow(file_path)
    if fallback is not None:
        logger.debug("Loaded %s via Pillow fallback", file_path.name)
        return fallback

    raise ImageLoadError(f"Could not decode image: {file_path}")


def save_image(
    image: ImageData,
    path: os.PathLike | str,
    *,
    jpeg_quality: int = 98,
    png_compression: int = 3,
    webp_quality: int = 100,
    overwrite: bool = False,
) -> Path:
    """Write ``image`` to ``path``.

    Lossless containers (PNG/TIFF/BMP) receive the pixel data unchanged,
    including 16-bit samples and alpha. JPEG and WebP inevitably resample; the
    defaults are chosen to minimise added loss.

    Args:
        image: Image to write.
        path: Destination file. Its extension selects the container.
        jpeg_quality: Quality for ``.jpg``/``.jpeg`` output (1-100).
        png_compression: PNG compression level (0-9); 3 balances size/speed.
        webp_quality: WebP quality; 100 selects near-lossless.
        overwrite: Permit replacing an existing file.

    Returns:
        The path that was written.

    Raises:
        ImageSaveError: The destination exists (without ``overwrite``) or the
            encoder failed.
    """
    dest = Path(path)
    if dest.exists() and not overwrite:
        raise ImageSaveError(f"Refusing to overwrite existing file: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    suffix = dest.suffix.lower()
    pixels = image.pixels

    if suffix in (".jpg", ".jpeg", ".jpe"):
        if pixels.dtype != np.uint8:
            pixels = _to_uint8(pixels)
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            logger.warning("JPEG cannot store alpha; dropping alpha channel")
            pixels = pixels[..., :3]
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    elif suffix == ".webp":
        if pixels.dtype != np.uint8:
            pixels = _to_uint8(pixels)
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(webp_quality)]
    elif suffix in (".tif", ".tiff", ".bmp"):
        params = []
        if suffix == ".bmp" and pixels.dtype != np.uint8:
            pixels = _to_uint8(pixels)
    else:
        raise ImageSaveError(f"Unsupported output extension: {dest.suffix}")

    encoded = _rgb_to_bgr(pixels)
    ok, buffer = cv2.imencode(suffix, encoded, params)
    if not ok:
        raise ImageSaveError(f"Encoder rejected image for {dest}")
    try:
        buffer.tofile(str(dest))
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise ImageSaveError(f"Could not write {dest}: {exc}") from exc

    logger.info("Wrote %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Down-convert ``uint16``/float data to ``uint8`` with correct scaling."""
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.uint16:
        return (array.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
    return (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
