"""Model Manager dialog.

Shows every registered model with its task, kind, version, licence, repository,
weight size and installation status. Downloads are always explicit: the URL,
licence and size are shown, and the user must press Install. Nothing is fetched
in the background (S15, S43).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants import MODEL_STATUS_LABELS, ModelKind, ModelStatus
from app.paths import weights_dir
from restoration.base import ModelInfo, WeightSpec
from restoration.registry import ModelRegistry
from restoration.weights import installed_size, probe_url, remove_weight, verify_weight
from gui.theme import Palette
from gui.utils import reveal_in_file_manager, show_error
from gui.widgets.common import BannerLabel, SectionLabel
from workers.download_worker import DownloadWorker, InstallFileWorker

logger = logging.getLogger(__name__)

__all__ = ["ModelManagerDialog"]

_COLUMNS = ("Model", "Task", "Kind", "Version", "Status", "Size", "Licence")

_STATUS_COLOURS = {
    ModelStatus.INSTALLED.value: Palette.OK,
    ModelStatus.MISSING_WEIGHTS.value: Palette.WARN,
    ModelStatus.MISSING_DEPENDENCY.value: Palette.WARN,
    ModelStatus.NOT_INTEGRATED.value: Palette.FG_2,
}


class ModelManagerDialog(QDialog):
    """Browse, install, verify and remove restoration models.

    Signals:
        modelsChanged: Emitted after any installation change, so the
            restoration panel can refresh its availability display.
    """

    modelsChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Manager")
        self.setMinimumSize(1040, 680)
        self._rows: List[dict] = []
        self._worker: Optional[object] = None
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("Restoration models")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        layout.addWidget(
            BannerLabel(
                "Model weights are never downloaded automatically. Review each "
                "model's licence and source before installing it, and confirm "
                "that its terms permit your intended use - several of these "
                "weights are restricted to non-commercial research."
            )
        )

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter, 1)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(_COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        splitter.addWidget(self._table)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 6, 0, 0)
        detail_layout.setSpacing(6)
        detail_layout.addWidget(SectionLabel("Model detail"))
        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 11px;"
        )
        detail_layout.addWidget(self._detail, 1)
        splitter.addWidget(detail_widget)
        splitter.setSizes([380, 240])

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setProperty("role", "hint")
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_label)

        actions = QHBoxLayout()
        actions.setSpacing(6)

        self._install_button = QPushButton("Install")
        self._install_button.setProperty("accent", True)
        self._install_button.clicked.connect(self._install_selected)
        actions.addWidget(self._install_button)

        self._install_file_button = QPushButton("Install from file...")
        self._install_file_button.clicked.connect(self._install_from_file)
        actions.addWidget(self._install_file_button)

        self._verify_button = QPushButton("Verify")
        self._verify_button.clicked.connect(self._verify_selected)
        actions.addWidget(self._verify_button)

        self._remove_button = QPushButton("Remove weights")
        self._remove_button.setProperty("destructive", True)
        self._remove_button.clicked.connect(self._remove_selected)
        actions.addWidget(self._remove_button)

        self._repo_button = QPushButton("Open repository")
        self._repo_button.clicked.connect(self._open_repository)
        actions.addWidget(self._repo_button)

        actions.addStretch(1)

        self._cancel_button = QPushButton("Cancel download")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._cancel_worker)
        actions.addWidget(self._cancel_button)

        layout.addLayout(actions)

        footer = QHBoxLayout()
        self._footer_label = QLabel("")
        self._footer_label.setProperty("role", "hint")
        footer.addWidget(self._footer_label, 1)

        folder_button = QPushButton("Open weights folder")
        folder_button.clicked.connect(lambda: reveal_in_file_manager(weights_dir()))
        footer.addWidget(folder_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        footer.addWidget(buttons)
        layout.addLayout(footer)

    # ------------------------------------------------------------------ public
    def refresh(self) -> None:
        """Reload the table from the registry."""
        selected = self._selected_name()
        self._rows = ModelRegistry.status_table()
        self._table.setRowCount(len(self._rows))

        for row_index, row in enumerate(self._rows):
            info: ModelInfo = row["info"]
            size = "-"
            if info.weights:
                total = sum(w.size_bytes for w in info.weights)
                size = f"{total / (1024 * 1024):.0f} MiB" if total else "unknown"

            values = (
                info.display_name,
                row["task_label"],
                "Classical" if info.kind == ModelKind.CLASSICAL.value else "Neural",
                info.version,
                row["status_label"],
                size,
                info.license_name.split(";")[0][:44],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 4:
                    item.setForeground(
                        QBrush(QColor(_STATUS_COLOURS.get(row["status"], Palette.FG_0)))
                    )
                if column == 0 and info.may_synthesise:
                    item.setToolTip(
                        "This model can synthesise image content that is not "
                        "present in the source evidence."
                    )
                self._table.setItem(row_index, column, item)

        self._update_footer()
        if selected:
            self._select_by_name(selected)
        elif self._rows:
            self._table.selectRow(0)
        self._on_selection_changed()

    def _update_footer(self) -> None:
        """Update the summary line under the table."""
        installed = sum(
            1 for row in self._rows if row["status"] == ModelStatus.INSTALLED.value
        )
        total_bytes = installed_size()
        self._footer_label.setText(
            f"{installed} of {len(self._rows)} models ready  |  "
            f"weights folder: {weights_dir()}  ({total_bytes / (1024 * 1024):.0f} MiB)"
        )

    # ---------------------------------------------------------------- helpers
    def _selected_row(self) -> Optional[dict]:
        """Return the selected table row's backing record."""
        index = self._table.currentRow()
        if 0 <= index < len(self._rows):
            return self._rows[index]
        return None

    def _selected_name(self) -> str:
        """Return the selected model's registry key."""
        row = self._selected_row()
        return row["name"] if row else ""

    def _select_by_name(self, name: str) -> None:
        """Select the row for ``name``."""
        for index, row in enumerate(self._rows):
            if row["name"] == name:
                self._table.selectRow(index)
                return

    def _primary_spec(self, info: ModelInfo) -> Optional[WeightSpec]:
        """Return the first required weight spec."""
        for spec in info.weights:
            if spec.required:
                return spec
        return None

    # --------------------------------------------------------------- handlers
    def _on_selection_changed(self) -> None:
        """Render the detail pane and enable the applicable actions."""
        row = self._selected_row()
        if row is None:
            self._detail.clear()
            for button in (
                self._install_button, self._install_file_button,
                self._verify_button, self._remove_button, self._repo_button,
            ):
                button.setEnabled(False)
            return

        info: ModelInfo = row["info"]
        spec = self._primary_spec(info)
        installed = row["status"] == ModelStatus.INSTALLED.value
        integrated = row["status"] != ModelStatus.NOT_INTEGRATED.value

        self._install_button.setEnabled(
            integrated and not installed and bool(spec and spec.url)
        )
        self._install_file_button.setEnabled(integrated and bool(spec))
        self._verify_button.setEnabled(bool(spec))
        self._remove_button.setEnabled(
            bool(spec) and (weights_dir() / spec.filename).is_file()
        )
        self._repo_button.setEnabled(bool(info.repository))

        lines = [
            info.display_name,
            "=" * len(info.display_name),
            "",
            f"Registry key : {info.name}",
            f"Task         : {info.task_label}",
            f"Kind         : {'Classical (deterministic)' if info.kind == ModelKind.CLASSICAL.value else 'Neural (learned prior)'}",
            f"Version      : {info.version}",
            f"Scale        : x{info.scale}",
            f"Status       : {row['status_label']}",
        ]
        if row["reason"]:
            lines.append(f"Detail       : {row['reason']}")
        lines += [
            "",
            f"Authors      : {info.authors or 'n/a'}",
            f"Licence      : {info.license_name or 'n/a'}",
            f"Repository   : {info.repository or 'n/a'}",
            f"Paper        : {info.paper or 'n/a'}",
            "",
            "DESCRIPTION",
            "-" * 11,
            info.description or "(none)",
            "",
            "WHAT IT DOES TO THE PIXELS",
            "-" * 26,
            info.method or "(not documented)",
        ]
        if info.may_synthesise:
            lines += [
                "",
                "*** This model can synthesise image content that is not "
                "present in the source evidence. ***",
            ]
        if info.notes:
            lines += ["", "NOTES", "-" * 5, info.notes]

        if info.weights:
            lines += ["", "WEIGHT FILES", "-" * 12]
            for weight in info.weights:
                path = weights_dir() / weight.filename
                lines += [
                    f"  {weight.filename}",
                    f"    installed : {'yes - ' + str(path) if path.is_file() else 'no'}",
                    f"    size      : {weight.size_human()}",
                    f"    licence   : {weight.license_name or 'see model licence'}",
                    f"    url       : {weight.url or '(no direct download published)'}",
                    f"    source    : {weight.source or 'n/a'}",
                    f"    sha256    : {weight.sha256 or '(not published upstream)'}",
                ]

        self._detail.setPlainText("\n".join(lines))

    def _install_selected(self) -> None:
        """Confirm, then download the selected model's weights."""
        row = self._selected_row()
        if row is None:
            return
        info: ModelInfo = row["info"]
        spec = self._primary_spec(info)
        if spec is None or not spec.url:
            QMessageBox.information(
                self,
                "Manual installation required",
                f"{info.display_name} has no direct download URL.\n\n"
                f"Obtain '{spec.filename if spec else 'the weight file'}' from:\n"
                f"{spec.source if spec else info.repository}\n\n"
                "Then use 'Install from file...'.",
            )
            return

        confirm = QMessageBox(self)
        confirm.setIcon(QMessageBox.Question)
        confirm.setWindowTitle("Install model weights")
        confirm.setText(f"Download weights for <b>{info.display_name}</b>?")
        confirm.setInformativeText(
            f"File:     {spec.filename}\n"
            f"Size:     {spec.size_human()}\n"
            f"Licence:  {spec.license_name or info.license_name}\n"
            f"From:     {spec.url}\n\n"
            + (
                f"The download will be verified against the published SHA-256 "
                f"({spec.sha256[:16]}...)."
                if spec.sha256
                else "Upstream publishes no SHA-256 for this file, so its "
                "authenticity cannot be verified automatically. The digest of "
                "whatever is received will be recorded."
            )
        )
        confirm.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        confirm.button(QMessageBox.Ok).setText("Download")
        confirm.setDefaultButton(QMessageBox.Cancel)
        if confirm.exec_() != QMessageBox.Ok:
            return

        reachable, size, message = probe_url(spec.url)
        if not reachable:
            QMessageBox.warning(
                self,
                "Source unreachable",
                f"Could not reach the download URL:\n{spec.url}\n\n{message}\n\n"
                "The upstream release may have moved. Obtain the file manually "
                "and use 'Install from file...'.",
            )
            return

        worker = DownloadWorker(spec)
        self._start_worker(worker, f"Downloading {spec.filename}")

    def _install_from_file(self) -> None:
        """Install a locally-obtained weight file."""
        row = self._selected_row()
        if row is None:
            return
        info: ModelInfo = row["info"]
        spec = self._primary_spec(info)
        if spec is None:
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {spec.filename}",
            "",
            "Model weights (*.pth *.pt *.ckpt *.safetensors *.bin);;All files (*)",
        )
        if not path:
            return

        worker = InstallFileWorker(spec, Path(path))
        self._start_worker(worker, f"Installing {spec.filename}")

    def _verify_selected(self) -> None:
        """Re-hash an installed weight file and report the result."""
        row = self._selected_row()
        if row is None:
            return
        spec = self._primary_spec(row["info"])
        if spec is None:
            return
        ok, message = verify_weight(spec)
        if ok:
            QMessageBox.information(self, "Verification", message)
        else:
            QMessageBox.warning(self, "Verification failed", message)

    def _remove_selected(self) -> None:
        """Delete an installed weight file after confirmation."""
        row = self._selected_row()
        if row is None:
            return
        info: ModelInfo = row["info"]
        spec = self._primary_spec(info)
        if spec is None:
            return
        answer = QMessageBox.question(
            self,
            "Remove weights",
            f"Delete {spec.filename} from the weights folder?\n\n"
            f"{info.display_name} will become unavailable until it is "
            "reinstalled. Existing derivatives and their records are unaffected.",
            QMessageBox.Cancel | QMessageBox.Yes,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        model = ModelRegistry.try_get(info.name)
        if model is not None:
            model.unload()
        if remove_weight(spec):
            logger.info("Removed weights for %s", info.name)
        self.refresh()
        self.modelsChanged.emit()

    def _open_repository(self) -> None:
        """Open the model's upstream repository in a browser."""
        row = self._selected_row()
        if row is None:
            return
        url = row["info"].repository
        if url:
            QDesktopServices.openUrl(QUrl(url))

    # ----------------------------------------------------------------- worker
    def _start_worker(self, worker, label: str) -> None:
        """Run a download/install worker with progress feedback."""
        self._worker = worker
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._progress_label.setVisible(True)
        self._progress_label.setText(label)
        self._cancel_button.setVisible(True)
        self._set_actions_enabled(False)

        worker.progress.connect(self._on_worker_progress)
        worker.finished_work.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        worker.cancelled_work.connect(self._on_worker_cancelled)
        worker.start()

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self._install_button, self._install_file_button,
            self._verify_button, self._remove_button, self._table,
        ):
            button.setEnabled(enabled)

    def _on_worker_progress(self, percent: int, message: str) -> None:
        self._progress.setValue(percent)
        if message:
            self._progress_label.setText(message)

    def _finish_worker(self) -> None:
        self._progress.setVisible(False)
        self._progress_label.setVisible(False)
        self._cancel_button.setVisible(False)
        self._set_actions_enabled(True)
        self._worker = None
        self.refresh()
        self.modelsChanged.emit()

    def _on_worker_finished(self, result) -> None:
        summary = getattr(result, "summary", lambda: str(result))()
        self._finish_worker()
        QMessageBox.information(self, "Installation complete", summary)

    def _on_worker_error(self, message: str, detail: str) -> None:
        self._finish_worker()
        show_error(self, "Installation failed", message, detail)

    def _on_worker_cancelled(self) -> None:
        self._finish_worker()
        logger.info("Weight installation cancelled")

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def closeEvent(self, event) -> None:
        """Stop any running download before closing."""
        if self._worker is not None:
            self._worker.stop_and_wait(5000)
        super().closeEvent(event)
