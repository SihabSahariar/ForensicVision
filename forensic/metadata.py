"""Evidence metadata extraction.

Metadata is **read** from evidence and recorded; it is never stripped from or
rewritten into the original file. Derivatives are written without inherited
EXIF by default so that camera metadata is not silently attributed to an
algorithmically modified image; the provenance record carries the linkage
instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["FileMetadata", "extract_metadata", "format_gps"]

#: EXIF tags surfaced prominently in the metadata panel.
_HIGHLIGHT_TAGS: Tuple[str, ...] = (
    "Image Make",
    "Image Model",
    "Image Software",
    "Image DateTime",
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "EXIF LensModel",
    "EXIF FNumber",
    "EXIF ExposureTime",
    "EXIF ISOSpeedRatings",
    "EXIF FocalLength",
    "EXIF Flash",
    "EXIF WhiteBalance",
    "Image Orientation",
    "Image XResolution",
    "Image YResolution",
)


@dataclass
class FileMetadata:
    """Everything known about an evidence file that is not pixel data.

    Attributes:
        path: Absolute path to the file that was inspected.
        filename: Base name of the file.
        size_bytes: File size on disk.
        mtime: Filesystem modification time (UTC).
        container: Detected container format, e.g. ``"JPEG"``.
        width: Pixel width, if decodable.
        height: Pixel height, if decodable.
        channels: Channel count, if decodable.
        bit_depth: Bits per channel, if decodable.
        exif: Flattened EXIF tag mapping.
        gps: Decoded GPS fields, empty when absent.
        icc_profile_present: Whether an embedded ICC profile was found.
        warnings: Non-fatal issues encountered during extraction.
    """

    path: str = ""
    filename: str = ""
    size_bytes: int = 0
    mtime: Optional[datetime] = None
    container: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    exif: Dict[str, str] = field(default_factory=dict)
    gps: Dict[str, str] = field(default_factory=dict)
    icc_profile_present: bool = False
    warnings: List[str] = field(default_factory=list)

    # -- presentation helpers ---------------------------------------------- #
    @property
    def dimensions(self) -> str:
        """Return ``"1920 x 1080"`` or ``"unknown"``."""
        if self.width and self.height:
            return f"{self.width} x {self.height}"
        return "unknown"

    @property
    def megapixels(self) -> Optional[float]:
        """Return the pixel count in megapixels, if known."""
        if self.width and self.height:
            return round(self.width * self.height / 1_000_000.0, 2)
        return None

    def size_human(self) -> str:
        """Return the file size in human-friendly units."""
        return human_size(self.size_bytes)

    def highlights(self) -> List[Tuple[str, str]]:
        """Return the commonly-cited EXIF fields present in this file."""
        rows: List[Tuple[str, str]] = []
        for tag in _HIGHLIGHT_TAGS:
            if tag in self.exif:
                label = tag.split(" ", 1)[-1]
                rows.append((label, self.exif[tag]))
        return rows

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "path": self.path,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime.isoformat() if self.mtime else None,
            "container": self.container,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "exif": dict(self.exif),
            "gps": dict(self.gps),
            "icc_profile_present": self.icc_profile_present,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileMetadata":
        """Rebuild from :meth:`to_dict` output."""
        mtime = data.get("mtime")
        return cls(
            path=data.get("path", ""),
            filename=data.get("filename", ""),
            size_bytes=int(data.get("size_bytes", 0) or 0),
            mtime=datetime.fromisoformat(mtime) if mtime else None,
            container=data.get("container", ""),
            width=data.get("width"),
            height=data.get("height"),
            channels=data.get("channels"),
            bit_depth=data.get("bit_depth"),
            exif=dict(data.get("exif", {})),
            gps=dict(data.get("gps", {})),
            icc_profile_present=bool(data.get("icc_profile_present", False)),
            warnings=list(data.get("warnings", [])),
        )


def human_size(num_bytes: int) -> str:
    """Format a byte count with binary units."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"  # pragma: no cover - unreachable


def _ratio_to_float(value: Any) -> Optional[float]:
    """Convert an exifread ratio (or number) to ``float``."""
    try:
        if hasattr(value, "num") and hasattr(value, "den"):
            return float(value.num) / float(value.den or 1)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_degrees(values: Any) -> Optional[float]:
    """Convert an EXIF degrees/minutes/seconds triple to decimal degrees."""
    try:
        parts = [_ratio_to_float(v) for v in values]
        if len(parts) != 3 or any(p is None for p in parts):
            return None
        degrees, minutes, seconds = parts  # type: ignore[misc]
        return degrees + minutes / 60.0 + seconds / 3600.0
    except TypeError:
        return None


def format_gps(gps: Dict[str, str]) -> str:
    """Return a compact ``lat, lon`` string, or an empty string."""
    lat = gps.get("latitude")
    lon = gps.get("longitude")
    if lat and lon:
        return f"{lat}, {lon}"
    return ""


def _extract_exif(path: Path, metadata: FileMetadata) -> None:
    """Populate ``metadata.exif`` and ``metadata.gps`` using exifread."""
    try:
        import exifread  # noqa: PLC0415 - optional dependency
    except ImportError:
        metadata.warnings.append("exifread not installed; EXIF not extracted")
        return

    try:
        with path.open("rb") as handle:
            tags = exifread.process_file(handle, details=False, strict=False)
    except Exception as exc:
        metadata.warnings.append(f"EXIF parse failed: {exc}")
        logger.debug("EXIF parsing failed for %s", path, exc_info=True)
        return

    gps_raw: Dict[str, Any] = {}
    for key, value in tags.items():
        if key in ("JPEGThumbnail", "TIFFThumbnail", "Filename", "EXIF MakerNote"):
            continue
        text = str(value).strip()
        if not text:
            continue
        metadata.exif[key] = text[:512]
        if key.startswith("GPS "):
            gps_raw[key] = value

    _decode_gps(gps_raw, metadata)


def _decode_gps(gps_raw: Dict[str, Any], metadata: FileMetadata) -> None:
    """Decode GPS EXIF tags into decimal degrees."""
    if not gps_raw:
        return
    try:
        lat_tag = gps_raw.get("GPS GPSLatitude")
        lat_ref = str(gps_raw.get("GPS GPSLatitudeRef", "")).strip()
        lon_tag = gps_raw.get("GPS GPSLongitude")
        lon_ref = str(gps_raw.get("GPS GPSLongitudeRef", "")).strip()

        if lat_tag is not None and lon_tag is not None:
            lat = _dms_to_degrees(lat_tag.values)
            lon = _dms_to_degrees(lon_tag.values)
            if lat is not None and lon is not None:
                if lat_ref.upper().startswith("S"):
                    lat = -lat
                if lon_ref.upper().startswith("W"):
                    lon = -lon
                metadata.gps["latitude"] = f"{lat:.6f}"
                metadata.gps["longitude"] = f"{lon:.6f}"

        alt_tag = gps_raw.get("GPS GPSAltitude")
        if alt_tag is not None:
            altitude = _ratio_to_float(alt_tag.values[0])
            if altitude is not None:
                metadata.gps["altitude_m"] = f"{altitude:.2f}"

        stamp = gps_raw.get("GPS GPSDateStamp")
        if stamp is not None:
            metadata.gps["date_stamp"] = str(stamp)
    except Exception as exc:  # pragma: no cover - malformed EXIF
        metadata.warnings.append(f"GPS decode failed: {exc}")


def _extract_image_properties(path: Path, metadata: FileMetadata) -> None:
    """Fill in dimensions/bit depth without fully decoding when possible."""
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as img:
            metadata.container = (img.format or "").upper()
            metadata.width, metadata.height = img.size
            mode_channels = {
                "1": 1, "L": 1, "P": 1, "I": 1, "F": 1,
                "I;16": 1, "I;16B": 1, "I;16L": 1,
                "LA": 2, "RGB": 3, "YCbCr": 3, "HSV": 3, "LAB": 3,
                "RGBA": 4, "CMYK": 4, "RGBX": 4,
            }
            metadata.channels = mode_channels.get(img.mode, 3)
            metadata.bit_depth = 16 if img.mode.startswith("I;16") or img.mode == "I" else 8
            metadata.icc_profile_present = bool(img.info.get("icc_profile"))
            return
    except Exception:
        logger.debug("Pillow property probe failed for %s", path, exc_info=True)

    try:
        from core.image_io import load_image  # noqa: PLC0415 - avoid cycle at import

        image = load_image(path, allow_unsupported=True)
        metadata.width = image.width
        metadata.height = image.height
        metadata.channels = image.channels
        metadata.bit_depth = image.bit_depth
        metadata.container = image.source_format or metadata.container
    except Exception as exc:
        metadata.warnings.append(f"Could not read image properties: {exc}")


def extract_metadata(path: os.PathLike | str) -> FileMetadata:
    """Extract file, image and EXIF metadata for ``path``.

    The function is deliberately fault-tolerant: any sub-extractor that fails
    records a warning rather than aborting, because partial metadata is still
    evidentially useful.

    Args:
        path: File to inspect.

    Returns:
        A populated :class:`FileMetadata`.
    """
    file_path = Path(path)
    metadata = FileMetadata(path=str(file_path), filename=file_path.name)

    try:
        stat = file_path.stat()
        metadata.size_bytes = stat.st_size
        metadata.mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    except OSError as exc:
        metadata.warnings.append(f"stat() failed: {exc}")
        return metadata

    if not metadata.container:
        metadata.container = file_path.suffix.lstrip(".").upper()

    _extract_image_properties(file_path, metadata)
    _extract_exif(file_path, metadata)

    logger.debug(
        "Metadata for %s: %s, %d EXIF tags, GPS=%s",
        file_path.name,
        metadata.dimensions,
        len(metadata.exif),
        bool(metadata.gps),
    )
    return metadata
