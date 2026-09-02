"""Model weight installation.

Policy (specification S15/S43): weights are **never** downloaded silently. Every
download is initiated explicitly by the investigator from the Model Manager,
which first shows the URL, the licence and the size. Downloads verify SHA-256
where upstream publishes one, and always land atomically via a temporary file so
a partial transfer can never be mistaken for an installed model.

Installing from a local file is supported as a first-class path, for air-gapped
deployments and for models whose upstream distribution is not a direct URL.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from app.paths import ensure_dir, weights_dir
from app.version import APP_NAME, APP_VERSION
from core.exceptions import ForensicVisionError, OperationCancelled
from restoration.base import WeightSpec

logger = logging.getLogger(__name__)

__all__ = [
    "WeightInstallError",
    "DownloadResult",
    "download_weight",
    "install_from_file",
    "verify_weight",
    "probe_url",
    "remove_weight",
]

#: ``(bytes_done, bytes_total)`` progress callback.
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]

_CHUNK = 256 * 1024
_USER_AGENT = f"{APP_NAME}/{APP_VERSION}"


class WeightInstallError(ForensicVisionError):
    """Raised when a weight file cannot be installed or verified."""


@dataclass
class DownloadResult:
    """Outcome of a weight installation."""

    path: Path
    size_bytes: int
    sha256: str
    verified: bool
    source: str

    def summary(self) -> str:
        """Return a one-line description for the log and the audit trail."""
        state = "verified" if self.verified else "UNVERIFIED (no published digest)"
        return (
            f"{self.path.name} - {self.size_bytes / (1024 * 1024):.1f} MiB, "
            f"sha256 {self.sha256[:16]}, {state}, from {self.source}"
        )


def _sha256_of(path: Path, progress: Optional[ProgressCallback] = None) -> str:
    """Return the SHA-256 of ``path``, streaming the file."""
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


def probe_url(url: str, timeout: float = 15.0) -> Tuple[bool, int, str]:
    """Check that ``url`` is reachable without downloading it.

    Args:
        url: Location to probe.
        timeout: Socket timeout in seconds.

    Returns:
        ``(reachable, size_bytes, message)``. ``size_bytes`` is 0 when the
        server does not advertise a content length.
    """
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            size = int(length) if length and length.isdigit() else 0
            return True, size, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # Some release hosts reject HEAD but serve GET; report it as such.
        return False, 0, f"HTTP {exc.code} {exc.reason}"
    except Exception as exc:
        return False, 0, str(exc)


def download_weight(
    spec: WeightSpec,
    destination_dir: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCheck] = None,
    timeout: float = 60.0,
) -> DownloadResult:
    """Download the weight file described by ``spec``.

    Args:
        spec: The weight to fetch; must carry a URL.
        destination_dir: Target directory; defaults to the configured weights
            directory.
        progress: Optional ``(done, total)`` callback.
        cancelled: Optional predicate polled during transfer.
        timeout: Socket timeout in seconds.

    Returns:
        A :class:`DownloadResult`.

    Raises:
        WeightInstallError: No URL, transfer failure, or digest mismatch.
        OperationCancelled: ``cancelled`` returned ``True``.
    """
    if not spec.url:
        raise WeightInstallError(
            f"'{spec.filename}' has no published direct download URL. Obtain it "
            f"from {spec.source or 'the upstream project'} and install it with "
            "'Install from file...' in the Model Manager."
        )

    target_dir = ensure_dir(Path(destination_dir) if destination_dir else weights_dir())
    final_path = target_dir / spec.filename
    temp_path = final_path.with_suffix(final_path.suffix + ".part")

    logger.info("Downloading %s from %s", spec.filename, spec.url)
    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            total = int(length) if length and length.isdigit() else spec.size_bytes
            done = 0
            digest = hashlib.sha256()
            with temp_path.open("wb") as handle:
                while True:
                    if cancelled is not None and cancelled():
                        raise OperationCancelled(
                            f"Download of {spec.filename} cancelled"
                        )
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
    except OperationCancelled:
        temp_path.unlink(missing_ok=True)
        raise
    except urllib.error.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise WeightInstallError(
            f"Server returned HTTP {exc.code} ({exc.reason}) for {spec.url}. "
            "The upstream release may have moved; install the file manually "
            "with 'Install from file...'."
        ) from exc
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise WeightInstallError(f"Download of {spec.filename} failed: {exc}") from exc

    actual_sha = digest.hexdigest()
    if spec.sha256 and actual_sha.lower() != spec.sha256.lower():
        temp_path.unlink(missing_ok=True)
        raise WeightInstallError(
            f"Digest mismatch for {spec.filename}.\n"
            f"Expected sha256 {spec.sha256}\n"
            f"Received sha256 {actual_sha}\n"
            "The file was discarded. Do not use weights that fail verification."
        )

    size = temp_path.stat().st_size
    if size == 0:
        temp_path.unlink(missing_ok=True)
        raise WeightInstallError(f"Downloaded file {spec.filename} is empty")

    temp_path.replace(final_path)
    result = DownloadResult(
        path=final_path,
        size_bytes=size,
        sha256=actual_sha,
        verified=bool(spec.sha256),
        source=spec.url,
    )
    logger.info("Installed weight: %s", result.summary())
    return result


def install_from_file(
    spec: WeightSpec,
    source: Path,
    destination_dir: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> DownloadResult:
    """Install a locally-obtained weight file.

    Args:
        spec: The weight being installed, used for the target filename and the
            expected digest.
        source: Path to the file the investigator obtained.
        destination_dir: Target directory; defaults to the weights directory.
        progress: Optional ``(done, total)`` callback used while hashing.

    Returns:
        A :class:`DownloadResult`.

    Raises:
        WeightInstallError: The source is missing or fails verification.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise WeightInstallError(f"File not found: {source_path}")

    target_dir = ensure_dir(Path(destination_dir) if destination_dir else weights_dir())
    final_path = target_dir / spec.filename
    temp_path = final_path.with_suffix(final_path.suffix + ".part")

    try:
        shutil.copy2(source_path, temp_path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise WeightInstallError(f"Could not copy {source_path}: {exc}") from exc

    actual_sha = _sha256_of(temp_path, progress)
    if spec.sha256 and actual_sha.lower() != spec.sha256.lower():
        temp_path.unlink(missing_ok=True)
        raise WeightInstallError(
            f"Digest mismatch for {spec.filename}.\n"
            f"Expected sha256 {spec.sha256}\n"
            f"Received sha256 {actual_sha}\n"
            "The file was not installed."
        )

    temp_path.replace(final_path)
    result = DownloadResult(
        path=final_path,
        size_bytes=final_path.stat().st_size,
        sha256=actual_sha,
        verified=bool(spec.sha256),
        source=str(source_path),
    )
    logger.info("Installed weight from file: %s", result.summary())
    return result


def verify_weight(spec: WeightSpec, directory: Optional[Path] = None) -> Tuple[bool, str]:
    """Re-hash an installed weight file and compare with the published digest.

    Returns:
        ``(ok, message)``. When upstream publishes no digest, ``ok`` is ``True``
        and the message states that the file could not be verified.
    """
    target_dir = Path(directory) if directory else weights_dir()
    path = target_dir / spec.filename
    if not path.is_file():
        return False, f"{spec.filename} is not installed"
    actual = _sha256_of(path)
    if not spec.sha256:
        return True, (
            f"{spec.filename} present (sha256 {actual[:16]}). Upstream publishes "
            "no digest for this file, so its authenticity cannot be verified "
            "automatically."
        )
    if actual.lower() == spec.sha256.lower():
        return True, f"{spec.filename} verified (sha256 {actual[:16]})"
    return False, (
        f"{spec.filename} FAILED verification: expected {spec.sha256[:16]}, "
        f"found {actual[:16]}"
    )


def remove_weight(spec: WeightSpec, directory: Optional[Path] = None) -> bool:
    """Delete an installed weight file.

    Returns:
        ``True`` when a file was removed.
    """
    target_dir = Path(directory) if directory else weights_dir()
    path = target_dir / spec.filename
    if not path.is_file():
        return False
    path.unlink()
    logger.info("Removed weight file %s", path)
    return True


def installed_size(directory: Optional[Path] = None) -> int:
    """Return the total size in bytes of everything in the weights directory."""
    target_dir = Path(directory) if directory else weights_dir()
    if not target_dir.is_dir():
        return 0
    return sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
