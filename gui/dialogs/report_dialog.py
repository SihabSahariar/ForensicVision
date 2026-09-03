"""Report generation dialog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from analysis.analyzer import AnalysisReport
from core.case_manager import CaseManager
from core.image_io import ImageData
from database.models import Derivative, Evidence
from forensic.metadata import extract_metadata
from forensic.provenance import environment_snapshot
from gui.comparison_viewer import DifferenceMode, compute_difference
from gui.utils import open_with_default_application, show_error
from gui.widgets.common import BannerLabel, SectionLabel
from restoration.registry import ModelRegistry
from workers.report_worker import ReportWorker

logger = logging.getLogger(__name__)

__all__ = ["ReportDialog"]


class ReportDialog(QDialog):
    """Collects report options and runs the report worker."""

    def __init__(
        self,
        case: CaseManager,
        evidence: Evidence,
        derivative: Optional[Derivative] = None,
        report: Optional[AnalysisReport] = None,
        original: Optional[ImageData] = None,
        enhanced: Optional[ImageData] = None,
        difference_statistics: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate forensic report")
        self.setMinimumWidth(620)

        self._case = case
        self._evidence = evidence
        self._derivative = derivative
        self._analysis = report
        self._original = original
        self._enhanced = enhanced
        self._difference_statistics = difference_statistics or {}
        self._worker: Optional[ReportWorker] = None
        self._output: Optional[Path] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("Forensic report")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        form = QFormLayout()
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._investigator = QLineEdit(self._case.case.investigator)
        form.addRow("Investigator", self._investigator)

        self._organisation = QLineEdit(self._case.case.organisation)
        form.addRow("Organisation", self._organisation)

        default_name = (
            f"{self._case.case_id}_{Path(self._evidence.stored_path).stem}_report.pdf"
        )
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        self._path_edit = QLineEdit(str(self._case.reports_dir / default_name))
        path_row.addWidget(self._path_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        form.addRow("Output", path_row)

        layout.addLayout(form)

        layout.addWidget(SectionLabel("Include"))
        self._include_metadata = QCheckBox("Full EXIF metadata listing")
        self._include_metadata.setChecked(True)
        layout.addWidget(self._include_metadata)

        self._include_images = QCheckBox("Before / after images")
        self._include_images.setChecked(self._original is not None)
        self._include_images.setEnabled(self._original is not None)
        layout.addWidget(self._include_images)

        self._include_difference = QCheckBox("Difference analysis")
        has_pair = self._original is not None and self._enhanced is not None
        self._include_difference.setChecked(has_pair)
        self._include_difference.setEnabled(has_pair)
        layout.addWidget(self._include_difference)

        self._include_audit = QCheckBox("Audit trail extract")
        self._include_audit.setChecked(True)
        layout.addWidget(self._include_audit)

        layout.addWidget(SectionLabel("Examiner's notes on limitations"))
        self._limitations = QPlainTextEdit()
        self._limitations.setPlaceholderText(
            "Optional. Anything specific to this examination that a reader "
            "must know in order to interpret the results correctly."
        )
        self._limitations.setMaximumHeight(90)
        layout.addWidget(self._limitations)

        layout.addWidget(
            BannerLabel(
                "The report always includes the mandatory disclaimer, the "
                "heuristic-indicator caveat, the full model licensing and "
                "provenance, and the per-step hash chain. These cannot be "
                "disabled."
            )
        )

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setProperty("role", "hint")
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._generate = QPushButton("Generate report")
        self._generate.setProperty("accent", True)
        self._generate.setDefault(True)
        self._generate.clicked.connect(self._on_generate)
        buttons.addButton(self._generate, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---------------------------------------------------------------- helpers
    def _browse(self) -> None:
        """Choose the output path."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save report as", self._path_edit.text(), "PDF (*.pdf)"
        )
        if path:
            self._path_edit.setText(path)

    def _build_context(self) -> Dict[str, Any]:
        """Assemble everything the report builder needs."""
        context: Dict[str, Any] = {
            "evidence": self._evidence,
            "evidence_id": self._evidence.id,
            "derivative": self._derivative,
            "investigator": self._investigator.text().strip(),
            "organisation": self._organisation.text().strip(),
            "custom_limitations": self._limitations.toPlainText().strip(),
            "include_audit": self._include_audit.isChecked(),
            "environment": environment_snapshot(),
        }

        if self._include_metadata.isChecked():
            context["metadata"] = self._evidence.file_metadata or extract_metadata(
                Path(self._evidence.stored_path)
            ).to_dict()

        if self._analysis is not None:
            context["analysis"] = self._analysis.to_dict()
        else:
            stored = self._case.repository.latest_analysis(
                evidence_id=self._evidence.id
            )
            if stored is not None:
                context["analysis"] = stored.details

        if self._derivative is not None:
            context["pipeline"] = self._derivative.pipeline
            # Not the same question as "was a network involved": Zero-DCE is
            # neural and cannot synthesise. Prefer what the run actually
            # recorded, and fall back to the pipeline's own models.
            provenance = self._derivative.provenance or {}
            if "may_synthesise" in provenance:
                context["may_synthesise"] = bool(provenance["may_synthesise"])
            else:
                context["may_synthesise"] = any(
                    entry.get("may_synthesise")
                    for entry in self._collect_models(self._derivative.pipeline)
                )
            context["models"] = self._collect_models(self._derivative.pipeline)

        context["history"] = self._case.repository.list_steps(
            self._case.case_pk, self._evidence.id
        )

        if self._include_images.isChecked():
            if self._original is not None:
                context["original_image"] = self._original.pixels
            if self._enhanced is not None:
                context["enhanced_image"] = self._enhanced.pixels

        if (
            self._include_difference.isChecked()
            and self._original is not None
            and self._enhanced is not None
        ):
            visual, statistics = compute_difference(
                self._original.pixels, self._enhanced.pixels,
                DifferenceMode.AMPLIFIED,
            )
            context["difference_image"] = visual
            context["difference_statistics"] = statistics
            context["difference_label"] = DifferenceMode.LABELS[
                DifferenceMode.AMPLIFIED
            ]
        elif self._difference_statistics:
            context["difference_statistics"] = self._difference_statistics

        return context

    @staticmethod
    def _collect_models(pipeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return full model descriptions for every step in ``pipeline``."""
        models: List[Dict[str, Any]] = []
        seen = set()
        for step in (pipeline or {}).get("steps", []):
            name = step.get("model")
            if not name or name in seen:
                continue
            seen.add(name)
            info = ModelRegistry.info(name)
            if info is not None:
                models.append(info.to_dict())
        return models

    # --------------------------------------------------------------- handlers
    def _on_generate(self) -> None:
        """Validate and start the report worker."""
        output = Path(self._path_edit.text().strip())
        if not output.name:
            QMessageBox.warning(self, "Output required", "Choose an output path.")
            return
        if output.suffix.lower() != ".pdf":
            output = output.with_suffix(".pdf")

        try:
            context = self._build_context()
        except Exception as exc:
            logger.exception("Could not assemble the report context")
            show_error(self, "Report failed", str(exc))
            return

        self._output = output
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._generate.setEnabled(False)
        self._status.setText("Generating...")

        worker = ReportWorker(self._case, context, output)
        worker.progress.connect(self._on_progress)
        worker.finished_work.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.cancelled_work.connect(self._on_cancelled)
        self._worker = worker
        worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        if message:
            self._status.setText(message)

    def _on_finished(self, path: Path) -> None:
        self._worker = None
        self._progress.setVisible(False)
        self._generate.setEnabled(True)
        answer = QMessageBox.information(
            self, "Report generated",
            f"The report was written to:\n{path}\n\n"
            "Its SHA-256 has been recorded in the case database.",
            QMessageBox.Open | QMessageBox.Close,
            QMessageBox.Open,
        )
        if answer == QMessageBox.Open:
            open_with_default_application(path)
        self.accept()

    def _on_error(self, message: str, detail: str) -> None:
        self._worker = None
        self._progress.setVisible(False)
        self._generate.setEnabled(True)
        self._status.setText("Failed")
        show_error(self, "Report failed", message, detail)

    def _on_cancelled(self) -> None:
        self._worker = None
        self._progress.setVisible(False)
        self._generate.setEnabled(True)
        self._status.setText("Cancelled")

    def closeEvent(self, event) -> None:
        """Stop the worker if the dialog is closed mid-run."""
        if self._worker is not None:
            self._worker.stop_and_wait(5000)
        super().closeEvent(event)
