"""PDF report generation worker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.case_manager import CaseManager
from workers.base import BaseWorker

logger = logging.getLogger(__name__)

__all__ = ["ReportWorker"]


class ReportWorker(BaseWorker):
    """Builds a forensic PDF report off the GUI thread."""

    description = "Generating report"

    def __init__(
        self,
        case: CaseManager,
        context: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """Create the worker.

        Args:
            case: The case being reported on.
            context: Report content assembled by the report dialog.
            output_path: Destination PDF path.
        """
        super().__init__()
        self._case = case
        self._context = context
        self._output_path = Path(output_path)

    def execute(self) -> Path:
        """Render the PDF and register it with the case."""
        from reports.pdf_report import ForensicReportBuilder

        self.report_status("Building report...")
        builder = ForensicReportBuilder(self._case)
        path = builder.build(
            self._context,
            self._output_path,
            progress=self.report,
            cancelled=self.is_cancelled,
        )

        from forensic.hashing import hash_file

        self.report(96, "Hashing report")
        hashes = hash_file(path)
        self._case.repository.add_report(
            case_pk=self._case.case_pk,
            evidence_id=self._context.get("evidence_id"),
            path=str(path),
            kind="pdf",
            sha256=hashes.sha256,
            size_bytes=hashes.size_bytes,
            author=self._context.get("investigator", ""),
        )
        self._case.audit(
            action="report.generate",
            target=path.name,
            detail=f"sha256={hashes.sha256}",
        )
        self.report(100, "Report complete")
        return path
