"""Cryptographic hashing for evidence integrity.

Policy
------
* SHA-256 is the primary integrity mechanism.
* SHA-512 is computed alongside it for defence in depth.
* MD5 is computed **only** for cross-referencing with legacy case-management
  systems. It is collision-broken and must never be used to assert integrity;
  every API here labels it accordingly.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from app.constants import HASH_ALGORITHMS, HASH_CHUNK_SIZE, PRIMARY_HASH_ALGORITHM
from core.exceptions import IntegrityError

logger = logging.getLogger(__name__)

__all__ = [
    "HashSet",
    "hash_file",
    "hash_bytes",
    "hash_array",
    "verify_file",
    "MD5_ADVISORY",
]

MD5_ADVISORY: str = (
    "MD5 is provided for legacy cross-reference only. It is cryptographically "
    "broken and is not used by ForensicVision for integrity verification."
)


@dataclass(frozen=True)
class HashSet:
    """The digests computed for one artefact.

    Attributes:
        sha256: Primary integrity digest.
        sha512: Secondary integrity digest.
        md5: Legacy reference digest (see :data:`MD5_ADVISORY`).
        size_bytes: Size of the hashed byte stream.
    """

    sha256: str
    sha512: str
    md5: str
    size_bytes: int = 0

    @property
    def primary(self) -> str:
        """The digest treated as authoritative."""
        return self.sha256

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable mapping."""
        return {
            "sha256": self.sha256,
            "sha512": self.sha512,
            "md5": self.md5,
            "size_bytes": self.size_bytes,
            "primary_algorithm": PRIMARY_HASH_ALGORITHM,
            "md5_advisory": MD5_ADVISORY,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "HashSet":
        """Rebuild a :class:`HashSet` from :meth:`to_dict` output."""
        return cls(
            sha256=str(data.get("sha256", "")),
            sha512=str(data.get("sha512", "")),
            md5=str(data.get("md5", "")),
            size_bytes=int(data.get("size_bytes", 0) or 0),
        )

    def matches(self, other: "HashSet") -> bool:
        """Compare against ``other`` using the primary digest only."""
        return bool(self.sha256) and self.sha256 == other.sha256

    def short(self, length: int = 16) -> str:
        """Return a truncated SHA-256 for compact display."""
        return self.sha256[:length]


def _new_digests() -> Dict[str, "hashlib._Hash"]:
    return {name: hashlib.new(name) for name in HASH_ALGORITHMS}


def hash_file(
    path: os.PathLike | str,
    progress: Optional[Callable[[int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> HashSet:
    """Compute all configured digests for a file in a single streamed pass.

    Args:
        path: File to hash.
        progress: Optional callback receiving ``(bytes_done, bytes_total)``.
        cancelled: Optional predicate; hashing aborts when it returns ``True``.

    Returns:
        The computed :class:`HashSet`.

    Raises:
        IntegrityError: The file is unreadable, or hashing was cancelled.
    """
    file_path = Path(path)
    try:
        total = file_path.stat().st_size
    except OSError as exc:
        raise IntegrityError(f"Cannot stat {file_path}: {exc}") from exc

    digests = _new_digests()
    done = 0
    try:
        with file_path.open("rb") as handle:
            while True:
                if cancelled is not None and cancelled():
                    raise IntegrityError(f"Hashing cancelled for {file_path}")
                chunk = handle.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                for digest in digests.values():
                    digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except OSError as exc:
        raise IntegrityError(f"Cannot read {file_path}: {exc}") from exc

    result = HashSet(
        sha256=digests["sha256"].hexdigest(),
        sha512=digests["sha512"].hexdigest(),
        md5=digests["md5"].hexdigest(),
        size_bytes=done,
    )
    logger.info("SHA-256 %s  %s (%d bytes)", result.short(), file_path.name, done)
    return result


def hash_bytes(data: bytes) -> HashSet:
    """Compute all configured digests for an in-memory byte string."""
    digests = _new_digests()
    for digest in digests.values():
        digest.update(data)
    return HashSet(
        sha256=digests["sha256"].hexdigest(),
        sha512=digests["sha512"].hexdigest(),
        md5=digests["md5"].hexdigest(),
        size_bytes=len(data),
    )


def hash_array(array: np.ndarray) -> HashSet:
    """Hash raw pixel data.

    This is a *content* hash of the decoded samples, independent of container
    and metadata. It lets the application prove that two derivatives are
    pixel-identical even when written to different formats.
    """
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.shape}|{contiguous.dtype}".encode("utf-8")
    digests = _new_digests()
    for digest in digests.values():
        digest.update(header)
        digest.update(contiguous.tobytes())
    return HashSet(
        sha256=digests["sha256"].hexdigest(),
        sha512=digests["sha512"].hexdigest(),
        md5=digests["md5"].hexdigest(),
        size_bytes=int(contiguous.nbytes),
    )


def verify_file(path: os.PathLike | str, expected: HashSet) -> bool:
    """Re-hash ``path`` and compare against ``expected``.

    Only the primary (SHA-256) digest decides the result; a SHA-512 mismatch is
    logged as a critical inconsistency because it indicates either corruption
    or a stored-record problem.
    """
    actual = hash_file(path)
    ok = actual.sha256 == expected.sha256
    if not ok:
        logger.error(
            "Integrity check FAILED for %s: expected %s, got %s",
            path,
            expected.short(),
            actual.short(),
        )
    elif expected.sha512 and actual.sha512 != expected.sha512:
        logger.critical(
            "SHA-256 matched but SHA-512 differs for %s - stored record suspect",
            path,
        )
        return False
    return ok


def hash_many(
    paths: Iterable[os.PathLike | str],
) -> Dict[str, HashSet]:
    """Hash several files, returning a mapping keyed by string path."""
    return {str(path): hash_file(path) for path in paths}
