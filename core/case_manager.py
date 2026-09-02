"""Case lifecycle and evidence import.

A case is a self-contained directory::

    CASE-0001/
        case.json            manifest (human readable)
        case.db              SQLite database
        evidence/original/   byte-exact copies of imported files
        derivatives/         enhanced images + provenance sidecars
        analysis/            exported analysis documents
        reports/             generated PDFs
        metadata/            extracted metadata documents
        logs/                per-case operation log

Import performs, in order: copy -> hash -> metadata -> register -> protect.
The original bytes are copied *before* anything else touches the file, and the
hash is computed from the stored copy so the recorded digest describes exactly
what the case holds.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.constants import (
    CASE_DB_FILENAME,
    CASE_ID_PREFIX,
    CASE_MANIFEST_FILENAME,
    CASE_SUBDIRS,
)
from app.version import APP_VERSION
from core.exceptions import CaseError, EvidenceError
from core.image_io import ImageData, is_supported_path, load_image
from database.database import Database
from database.models import Case, Derivative, Evidence
from database.repository import CaseRepository
from forensic.hashing import HashSet, hash_file
from forensic.metadata import FileMetadata, extract_metadata
from forensic.safe_mode import SafeModeGuard, get_guard

logger = logging.getLogger(__name__)

__all__ = ["CaseManager", "ImportResult", "next_case_id"]

_CASE_ID_RE = re.compile(r"^CASE-(\d+)$", re.IGNORECASE)


@dataclass
class ImportResult:
    """Outcome of importing one evidence file."""

    evidence: Evidence
    hashes: HashSet
    metadata: FileMetadata
    stored_path: Path
    duplicate_of: Optional[Evidence] = None

    @property
    def is_duplicate(self) -> bool:
        """Whether an identical file was already registered in this case."""
        return self.duplicate_of is not None


def next_case_id(parent: Path) -> str:
    """Return the next unused sequential case identifier under ``parent``.

    Args:
        parent: Directory that holds case folders.

    Returns:
        A string such as ``"CASE-0003"``.
    """
    highest = 0
    if parent.exists():
        for child in parent.iterdir():
            if not child.is_dir():
                continue
            match = _CASE_ID_RE.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"{CASE_ID_PREFIX}{highest + 1:04d}"


def _sanitise_component(name: str) -> str:
    """Return ``name`` reduced to characters safe for every target filesystem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "unnamed"


class CaseManager:
    """Owns an open case: its directory layout, database and safe-mode guard.

    The manager is intentionally GUI-free so batch tooling and tests can drive
    the same code path the GUI uses.
    """

    def __init__(
        self,
        root: Path,
        database: Database,
        case_row: Case,
        guard: Optional[SafeModeGuard] = None,
    ) -> None:
        self._root = Path(root)
        self._db = database
        self._repo = CaseRepository(database)
        self._case = case_row
        self._guard = guard or get_guard()

    # ------------------------------------------------------------- properties
    @property
    def root(self) -> Path:
        """Case directory."""
        return self._root

    @property
    def case(self) -> Case:
        """The case database row (detached)."""
        return self._case

    @property
    def case_id(self) -> str:
        """Case identifier, e.g. ``"CASE-0001"``."""
        return self._case.case_id

    @property
    def case_pk(self) -> int:
        """Case primary key."""
        return int(self._case.id)

    @property
    def repository(self) -> CaseRepository:
        """Repository bound to this case's database."""
        return self._repo

    @property
    def database(self) -> Database:
        """The underlying database handle."""
        return self._db

    @property
    def guard(self) -> SafeModeGuard:
        """The safe-mode guard in force for this case."""
        return self._guard

    # -- directories -------------------------------------------------------- #
    @property
    def evidence_dir(self) -> Path:
        """Directory holding byte-exact originals."""
        return self._root / "evidence" / "original"

    @property
    def derivatives_dir(self) -> Path:
        """Directory holding enhanced derivatives."""
        return self._root / "derivatives"

    @property
    def analysis_dir(self) -> Path:
        """Directory holding exported analysis documents."""
        return self._root / "analysis"

    @property
    def reports_dir(self) -> Path:
        """Directory holding generated reports."""
        return self._root / "reports"

    @property
    def metadata_dir(self) -> Path:
        """Directory holding extracted metadata documents."""
        return self._root / "metadata"

    @property
    def logs_dir(self) -> Path:
        """Directory holding the per-case operation log."""
        return self._root / "logs"

    # ---------------------------------------------------------- construction
    @classmethod
    def create(
        cls,
        parent: Path,
        case_id: Optional[str] = None,
        title: str = "",
        investigator: str = "",
        organisation: str = "",
        description: str = "",
        guard: Optional[SafeModeGuard] = None,
    ) -> "CaseManager":
        """Create a new case directory, database and manifest.

        Args:
            parent: Directory in which the case folder is created.
            case_id: Explicit identifier; auto-generated when omitted.
            title: Short case title.
            investigator: Examiner name recorded in reports.
            organisation: Organisation recorded in reports.
            description: Free-form case description.
            guard: Safe-mode guard to bind (defaults to the global guard).

        Returns:
            An open :class:`CaseManager`.

        Raises:
            CaseError: The target directory already exists or is unusable.
        """
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        identifier = case_id or next_case_id(parent)
        root = parent / _sanitise_component(identifier)

        if root.exists() and any(root.iterdir()):
            raise CaseError(f"Case directory already exists and is not empty: {root}")

        try:
            for sub in CASE_SUBDIRS:
                (root / sub).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CaseError(f"Could not create case layout at {root}: {exc}") from exc

        database = Database(root / CASE_DB_FILENAME)
        database.create_all()
        repo = CaseRepository(database)
        case_row = repo.create_case(
            case_id=identifier,
            title=title,
            investigator=investigator,
            organisation=organisation,
            description=description,
            root_path=str(root),
            safe_mode_default=(guard or get_guard()).enabled,
        )

        manager = cls(root, database, case_row, guard)
        manager._write_manifest()
        manager._repo.add_audit(
            manager.case_pk,
            action="case.create",
            target=identifier,
            detail=f"Case created at {root}",
            actor=investigator,
            safe_mode=manager._guard.enabled,
        )
        logger.info("Case %s created at %s", identifier, root)
        return manager

    @classmethod
    def open(
        cls, root: Path, guard: Optional[SafeModeGuard] = None
    ) -> "CaseManager":
        """Open an existing case directory.

        Args:
            root: The case folder (the one containing ``case.db``).
            guard: Safe-mode guard to bind.

        Raises:
            CaseError: The directory is not a valid case.
        """
        root = Path(root)
        db_path = root / CASE_DB_FILENAME
        if not db_path.exists():
            raise CaseError(f"No case database found in {root}")

        database = Database(db_path)
        database.create_all()  # tolerate schema additions between versions
        repo = CaseRepository(database)
        case_row = repo.get_case()
        if case_row is None:
            raise CaseError(f"Case database in {root} contains no case record")

        for sub in CASE_SUBDIRS:
            (root / sub).mkdir(parents=True, exist_ok=True)

        manager = cls(root, database, case_row, guard)
        manager._reprotect_evidence()
        manager._repo.add_audit(
            manager.case_pk,
            action="case.open",
            target=case_row.case_id,
            detail=f"Case opened from {root}",
            safe_mode=manager._guard.enabled,
        )
        logger.info("Case %s opened from %s", case_row.case_id, root)
        return manager

    def close(self) -> None:
        """Flush the manifest and release database connections."""
        try:
            self._write_manifest()
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("Could not update case manifest on close")
        self._db.dispose()
        logger.info("Case %s closed", self.case_id)

    # ------------------------------------------------------------- manifest
    def _write_manifest(self) -> Path:
        """Write the human-readable ``case.json`` manifest."""
        counts = self._repo.counts(self.case_pk)
        manifest: Dict[str, Any] = {
            "schema": "forensicvision.case/1",
            "case_id": self._case.case_id,
            "title": self._case.title,
            "investigator": self._case.investigator,
            "organisation": self._case.organisation,
            "description": self._case.description,
            "created_at": self._case.created_at.isoformat()
            if self._case.created_at
            else "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "application_version": APP_VERSION,
            "safe_mode": self._guard.enabled,
            "counts": counts,
            "layout": list(CASE_SUBDIRS),
        }
        path = self._root / CASE_MANIFEST_FILENAME
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return path

    def refresh_case_row(self) -> None:
        """Reload the cached case row from the database."""
        row = self._repo.get_case(self._case.case_id)
        if row is not None:
            self._case = row

    def update_details(self, **fields: Any) -> None:
        """Update case metadata and rewrite the manifest."""
        self._repo.update_case(self.case_pk, **fields)
        self.refresh_case_row()
        self._write_manifest()

    # ---------------------------------------------------------------- import
    def import_evidence(
        self,
        source: Path,
        notes: str = "",
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> ImportResult:
        """Import one image as evidence.

        The workflow is: validate -> copy original bytes -> hash the stored
        copy -> extract metadata -> register in SQLite -> write-protect.

        Args:
            source: File to import.
            notes: Investigator notes stored with the evidence row.
            progress: Optional ``(done, total)`` callback used while hashing.

        Returns:
            An :class:`ImportResult`. When the same content is already present,
            ``duplicate_of`` is set and no second copy is stored.

        Raises:
            EvidenceError: The source is missing, unsupported or uncopyable.
        """
        src = Path(source)
        if not src.is_file():
            raise EvidenceError(f"Evidence file not found: {src}")
        if not is_supported_path(src):
            raise EvidenceError(
                f"Unsupported evidence format '{src.suffix}': {src.name}"
            )

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        stored_path = self._unique_destination(self.evidence_dir, src.name)

        try:
            shutil.copy2(src, stored_path)
        except OSError as exc:
            raise EvidenceError(f"Could not copy evidence into case: {exc}") from exc

        try:
            hashes = hash_file(stored_path, progress=progress)
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            raise EvidenceError(f"Hashing failed for {src.name}: {exc}") from exc

        existing = self._repo.find_evidence_by_hash(self.case_pk, hashes.sha256)
        if existing is not None:
            stored_path.unlink(missing_ok=True)
            logger.info(
                "Evidence %s duplicates existing item #%s; import skipped",
                src.name,
                existing.id,
            )
            self._repo.add_audit(
                self.case_pk,
                action="evidence.duplicate",
                target=src.name,
                detail=f"Identical content already registered as #{existing.id}",
                safe_mode=self._guard.enabled,
            )
            metadata = extract_metadata(Path(existing.stored_path))
            return ImportResult(
                evidence=existing,
                hashes=hashes,
                metadata=metadata,
                stored_path=Path(existing.stored_path),
                duplicate_of=existing,
            )

        metadata = extract_metadata(stored_path)
        metadata_doc = metadata.to_dict()
        self._write_metadata_document(stored_path.stem, metadata_doc, hashes)

        evidence = self._repo.add_evidence(
            case_pk=self.case_pk,
            original_filename=src.name,
            original_path=str(src),
            stored_path=str(stored_path),
            sha256=hashes.sha256,
            sha512=hashes.sha512,
            md5=hashes.md5,
            size_bytes=hashes.size_bytes,
            image_format=metadata.container,
            width=metadata.width or 0,
            height=metadata.height or 0,
            channels=metadata.channels or 0,
            bit_depth=metadata.bit_depth or 8,
            notes=notes,
            file_metadata=metadata_doc,
        )

        self._guard.protect_file(stored_path)
        self._repo.add_audit(
            self.case_pk,
            action="evidence.import",
            target=src.name,
            detail=(
                f"Imported from {src} | sha256={hashes.sha256} | "
                f"{metadata.dimensions} | {hashes.size_bytes} bytes"
            ),
            actor=self._case.investigator,
            safe_mode=self._guard.enabled,
        )
        self._write_manifest()
        return ImportResult(
            evidence=evidence,
            hashes=hashes,
            metadata=metadata,
            stored_path=stored_path,
        )

    def import_many(
        self,
        sources: List[Path],
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[ImportResult]:
        """Import several files, continuing past individual failures.

        Args:
            sources: Files to import.
            progress: Optional ``(index, total, filename)`` callback.

        Returns:
            Results for the files that imported successfully.
        """
        results: List[ImportResult] = []
        total = len(sources)
        for index, source in enumerate(sources):
            if progress is not None:
                progress(index, total, Path(source).name)
            try:
                results.append(self.import_evidence(Path(source)))
            except Exception as exc:
                logger.error("Import failed for %s: %s", source, exc)
        if progress is not None:
            progress(total, total, "")
        return results

    # ----------------------------------------------------------- derivatives
    def derivative_path(
        self, evidence: Evidence, suffix: str, extension: str = ".png"
    ) -> Path:
        """Build a non-colliding derivative path for ``evidence``.

        Args:
            evidence: Source evidence row.
            suffix: Operation tag included in the filename, e.g. ``"nafnet"``.
            extension: Output container extension.
        """
        stem = Path(evidence.stored_path).stem
        name = f"{_sanitise_component(stem)}_{_sanitise_component(suffix)}{extension}"
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)
        return self._unique_destination(self.derivatives_dir, name)

    @staticmethod
    def _unique_destination(directory: Path, filename: str) -> Path:
        """Return a path in ``directory`` that does not yet exist."""
        candidate = directory / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        index = 1
        while True:
            candidate = directory / f"{stem}_{index:03d}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    # --------------------------------------------------------------- helpers
    def load_evidence_image(self, evidence: Evidence) -> ImageData:
        """Decode the stored original for ``evidence``."""
        return load_image(Path(evidence.stored_path))

    def load_derivative_image(self, derivative: Derivative) -> ImageData:
        """Decode a stored derivative."""
        return load_image(Path(derivative.path))

    def verify_evidence(self, evidence: Evidence) -> bool:
        """Re-hash stored evidence and compare with the recorded digest."""
        stored = Path(evidence.stored_path)
        if not stored.exists():
            logger.error("Evidence file missing: %s", stored)
            return False
        actual = hash_file(stored)
        ok = actual.sha256 == evidence.sha256
        self._repo.add_audit(
            self.case_pk,
            action="evidence.verify",
            target=evidence.original_filename,
            detail=("PASS" if ok else f"FAIL expected {evidence.sha256}"),
            safe_mode=self._guard.enabled,
        )
        return ok

    def verify_all_evidence(self) -> Dict[str, bool]:
        """Verify every evidence item; returns ``{filename: passed}``."""
        return {
            item.original_filename: self.verify_evidence(item)
            for item in self._repo.list_evidence(self.case_pk)
        }

    def _reprotect_evidence(self) -> None:
        """Re-apply read-only protection to all stored originals."""
        for item in self._repo.list_evidence(self.case_pk):
            path = Path(item.stored_path)
            if path.exists():
                self._guard.protect_file(path)

    def _write_metadata_document(
        self, stem: str, metadata: Dict[str, Any], hashes: HashSet
    ) -> Path:
        """Persist the extracted metadata alongside its digests."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        path = self.metadata_dir / f"{_sanitise_component(stem)}.metadata.json"
        document = {
            "schema": "forensicvision.metadata/1",
            "case_id": self.case_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "hashes": hashes.to_dict(),
            "metadata": metadata,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False, default=str)
        return path

    # ------------------------------------------------------------- accessors
    def list_evidence(self) -> List[Evidence]:
        """Return every evidence row for this case."""
        return self._repo.list_evidence(self.case_pk)

    def list_derivatives(self, evidence_id: Optional[int] = None) -> List[Derivative]:
        """Return derivative rows, optionally filtered by evidence."""
        return self._repo.list_derivatives(self.case_pk, evidence_id)

    def counts(self) -> Dict[str, int]:
        """Return the case's item counts."""
        return self._repo.counts(self.case_pk)

    def audit(self, action: str, target: str = "", detail: str = "") -> None:
        """Append an audit entry for this case."""
        self._repo.add_audit(
            self.case_pk,
            action=action,
            target=target,
            detail=detail,
            actor=self._case.investigator,
            safe_mode=self._guard.enabled,
        )
