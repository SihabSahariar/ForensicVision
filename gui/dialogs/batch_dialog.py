"""Batch processing dialog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from core.case_manager import CaseManager
from gui.theme import Palette
from gui.utils import reveal_in_file_manager, show_error
from gui.widgets.common import BannerLabel, SectionLabel
from restoration.pipeline import Pipeline
from workers.batch_worker import BatchItemResult, BatchSummary, BatchWorker, discover_images

logger = logging.getLogger(__name__)

__all__ = ["BatchDialog"]

_STATUS_COLOURS = {
    "ok": Palette.OK,
    "error": Palette.ERROR,
    "skipped": Palette.FG_2,
    "cancelled": Palette.WARN,
    "pending": Palette.FG_1,
}


class BatchDialog(QDialog):
    """Runs the full workflow over a folder of images."""

    def __init__(
        self,
        case: CaseManager,
        config: AppConfig,
        parent: Optional[QWidget] = None,
        staged_pipeline: Optional[Pipeline] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch processing")
        self.setMinimumSize(880, 620)

        self._case = case
        self._config = config
        self._staged = staged_pipeline
        self._paths: List[Path] = []
        self._worker: Optional[BatchWorker] = None
        self._row_for: dict = {}

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("Batch processing")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        layout.addWidget(
            BannerLabel(
                "Every file is imported into the case, hashed, analysed and "
                "processed through the same recorded pipeline the interactive "
                "workflow uses. Derivatives, provenance and history are written "
                "exactly as for a single image."
            )
        )

        form = QFormLayout()
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Folder containing images to process")
        folder_row.addWidget(self._folder_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse)
        form.addRow("Source folder", folder_row)

        self._recursive = QCheckBox("Include subfolders")
        form.addRow("", self._recursive)
        self._recursive.toggled.connect(self._rescan)

        self._pipeline_combo = QComboBox()
        self._pipeline_combo.addItem("Per-image automatic pipeline", "auto")
        if self._staged is not None and self._staged.enabled_steps:
            self._pipeline_combo.addItem(
                f"Staged pipeline ({len(self._staged.enabled_steps)} steps)", "staged"
            )
        form.addRow("Pipeline", self._pipeline_combo)

        export_row = QHBoxLayout()
        export_row.setSpacing(6)
        self._export_edit = QLineEdit()
        self._export_edit.setPlaceholderText(
            "Optional: also copy derivatives to this folder"
        )
        export_row.addWidget(self._export_edit, 1)
        export_browse = QPushButton("Browse...")
        export_browse.clicked.connect(self._browse_export)
        export_row.addWidget(export_browse)
        form.addRow("Export copies to", export_row)

        layout.addLayout(form)

        layout.addWidget(SectionLabel("Files"))
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["File", "Status", "Pipeline", "Detail"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self._table, 1)

        self._summary = QLabel("No folder selected.")
        self._summary.setProperty("role", "hint")
        layout.addWidget(self._summary)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._current = QLabel("")
        self._current.setProperty("role", "mono")
        layout.addWidget(self._current)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._start = QPushButton("Start")
        self._start.setProperty("accent", True)
        self._start.setEnabled(False)
        self._start.clicked.connect(self._on_start)
        controls.addWidget(self._start)

        self._pause = QPushButton("Pause")
        self._pause.setEnabled(False)
        self._pause.clicked.connect(self._on_pause)
        controls.addWidget(self._pause)

        self._cancel = QPushButton("Cancel run")
        self._cancel.setEnabled(False)
        self._cancel.setProperty("destructive", True)
        self._cancel.clicked.connect(self._on_cancel)
        controls.addWidget(self._cancel)
        controls.addStretch(1)

        open_output = QPushButton("Open derivatives folder")
        open_output.clicked.connect(
            lambda: reveal_in_file_manager(self._case.derivatives_dir)
        )
        controls.addWidget(open_output)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        controls.addWidget(buttons)
        layout.addLayout(controls)

    # ---------------------------------------------------------------- helpers
    def _browse_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select source folder")
        if directory:
            self._folder_edit.setText(directory)
            self._rescan()

    def _browse_export(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select export folder")
        if directory:
            self._export_edit.setText(directory)

    def _rescan(self) -> None:
        """Find images in the chosen folder and populate the table."""
        folder = self._folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            self._paths = []
            self._table.setRowCount(0)
            self._summary.setText("No folder selected.")
            self._start.setEnabled(False)
            return

        self._paths = discover_images(Path(folder), self._recursive.isChecked())
        self._table.setRowCount(len(self._paths))
        self._row_for = {}
        for row, path in enumerate(self._paths):
            self._row_for[str(path)] = row
            self._table.setItem(row, 0, QTableWidgetItem(path.name))
            status = QTableWidgetItem("pending")
            status.setForeground(QBrush(QColor(_STATUS_COLOURS["pending"])))
            self._table.setItem(row, 1, status)
            self._table.setItem(row, 2, QTableWidgetItem(""))
            self._table.setItem(row, 3, QTableWidgetItem(""))

        self._summary.setText(f"{len(self._paths)} image(s) found.")
        self._start.setEnabled(bool(self._paths))

    # --------------------------------------------------------------- handlers
    def _on_start(self) -> None:
        """Confirm and start the batch worker."""
        if not self._paths:
            return

        use_staged = self._pipeline_combo.currentData() == "staged"
        pipeline = self._staged if use_staged else None
        description = (
            " -> ".join(s.display_name for s in pipeline.enabled_steps)
            if pipeline else "an automatically chosen pipeline per image"
        )

        answer = QMessageBox.question(
            self, "Start batch run",
            f"Process {len(self._paths)} file(s) using {description}?\n\n"
            f"Every file will be imported into case {self._case.case_id}, "
            "hashed and recorded.",
            QMessageBox.Cancel | QMessageBox.Yes, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return

        export_dir = self._export_edit.text().strip()
        worker = BatchWorker(
            case=self._case,
            paths=self._paths,
            fixed_pipeline=pipeline,
            device=self._config.device,
            fp16=self._config.use_fp16,
            export_dir=Path(export_dir) if export_dir else None,
        )
        worker.progress.connect(self._on_progress)
        worker.item_started.connect(self._on_item_started)
        worker.item_finished.connect(self._on_item_finished)
        worker.finished_work.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.cancelled_work.connect(self._on_cancelled)

        self._worker = worker
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._start.setEnabled(False)
        self._pause.setEnabled(True)
        self._cancel.setEnabled(True)
        worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        self._current.setText(message)

    def _on_item_started(
        self, index: int, total: int, filename: str, pipeline: str
    ) -> None:
        """Show which file is running and the pipeline chosen for it."""
        row = index if index < self._table.rowCount() else -1
        if row >= 0:
            item = QTableWidgetItem("running")
            item.setForeground(QBrush(QColor(Palette.ACCENT)))
            self._table.setItem(row, 1, item)
            self._table.setItem(row, 2, QTableWidgetItem(pipeline))
            self._table.scrollToItem(self._table.item(row, 0))
        self._current.setText(f"{filename}  |  {pipeline}")

    def _on_item_finished(self, item: BatchItemResult) -> None:
        """Update the table row for a completed file."""
        row = self._row_for.get(str(item.path))
        if row is None:
            return
        status = QTableWidgetItem(item.status)
        status.setForeground(
            QBrush(QColor(_STATUS_COLOURS.get(item.status, Palette.FG_0)))
        )
        self._table.setItem(row, 1, status)
        self._table.setItem(row, 2, QTableWidgetItem(item.pipeline_summary))
        detail = item.message
        if item.status == "ok" and item.output_path is not None:
            detail = f"{item.output_path.name}  ({item.duration_s:.1f}s)"
        self._table.setItem(row, 3, QTableWidgetItem(detail))

    def _on_pause(self) -> None:
        """Toggle pause."""
        if self._worker is None:
            return
        if self._worker.is_paused:
            self._worker.resume()
            self._pause.setText("Pause")
            self._current.setText("Resuming...")
        else:
            self._worker.pause()
            self._pause.setText("Resume")
            self._current.setText("Paused - will stop after the current file.")

    def _on_cancel(self) -> None:
        """Cancel the run."""
        if self._worker is not None:
            self._worker.resume()
            self._worker.cancel()
            self._current.setText("Cancelling...")

    def _reset_controls(self) -> None:
        self._worker = None
        self._progress.setVisible(False)
        self._start.setEnabled(bool(self._paths))
        self._pause.setEnabled(False)
        self._pause.setText("Pause")
        self._cancel.setEnabled(False)

    def _on_finished(self, summary: BatchSummary) -> None:
        self._reset_controls()
        self._summary.setText(
            f"{summary.completed} processed, {summary.skipped} skipped, "
            f"{summary.failed} failed in {summary.total_duration_s:.1f}s"
        )
        QMessageBox.information(
            self, "Batch complete",
            f"Processed : {summary.completed}\n"
            f"Skipped   : {summary.skipped}\n"
            f"Failed    : {summary.failed}\n"
            f"Duration  : {summary.total_duration_s:.1f} s\n\n"
            f"Derivatives were written to:\n{self._case.derivatives_dir}",
        )

    def _on_error(self, message: str, detail: str) -> None:
        self._reset_controls()
        show_error(self, "Batch failed", message, detail)

    def _on_cancelled(self) -> None:
        self._reset_controls()
        self._current.setText("Cancelled")

    def closeEvent(self, event) -> None:
        """Stop the batch before closing."""
        if self._worker is not None:
            answer = QMessageBox.question(
                self, "Batch running",
                "Cancel the batch run and close?",
                QMessageBox.Cancel | QMessageBox.Yes, QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._worker.resume()
            self._worker.stop_and_wait(10000)
        super().closeEvent(event)
