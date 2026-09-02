"""Evidence-import worker.

Importing performs a byte copy plus three full-file hash passes, which is
IO-bound and slow enough on large TIFFs to freeze the GUI if run inline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

from core.case_manager import CaseManager, ImportResult
from core.exceptions import OperationCancelled
from workers.base import BaseWorker

logger = logging.getLogger(__name__)

__all__ = ["ImportWorker"]


class ImportWorker(BaseWorker):
    """Imports one or more files into a case."""

    description = "Importing evidence"

    def __init__(self, case: CaseManager, paths: Sequence[Path]) -> None:
        super().__init__()
        self._case = case
        self._paths = [Path(p) for p in paths]

    def execute(self) -> List[ImportResult]:
        """Copy, hash, extract metadata and register each file."""
        results: List[ImportResult] = []
        total = len(self._paths)
        failures: List[str] = []

        for index, path in enumerate(self._paths):
            if self.is_cancelled():
                raise OperationCancelled("Import cancelled")

            base = int(index * 100 / total)
            span = max(1, int(100 / total))
            self.report(base, f"Importing {path.name} ({index + 1}/{total})")
            self.report_status(f"Hashing {path.name}...")

            def hash_progress(done: int, size: int, _base=base, _span=span) -> None:
                if size > 0:
                    self.report(_base + int(done * _span / size), f"Hashing {path.name}")

            try:
                results.append(
                    self._case.import_evidence(path, progress=hash_progress)
                )
            except Exception as exc:
                logger.error("Import failed for %s: %s", path, exc)
                failures.append(f"{path.name}: {exc}")

        self.report(100, "Import complete")
        if failures and not results:
            raise RuntimeError(
                "No files could be imported:\n" + "\n".join(failures)
            )
        if failures:
            self.report_status(
                f"Imported {len(results)} file(s); {len(failures)} failed"
            )
        return results
