"""Application log dock.

Subscribes to the in-memory log handler. Records arrive on whichever thread
emitted them, so the listener marshals to the GUI thread through a queued
signal connection rather than touching widgets directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.logging_setup import LogRecordEntry, get_log_file_path, get_memory_handler
from gui.theme import Palette

logger = logging.getLogger(__name__)

__all__ = ["LogPanel"]

_LEVEL_COLOURS = {
    "DEBUG": Palette.FG_2,
    "INFO": Palette.FG_0,
    "WARNING": Palette.WARN,
    "ERROR": Palette.ERROR,
    "CRITICAL": Palette.ERROR,
}

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class LogPanel(QWidget):
    """Live view of the application log with filtering and export."""

    #: Internal bridge so worker-thread records reach the GUI thread safely.
    _recordArrived = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._min_level = logging.INFO
        self._filter_text = ""
        self._autoscroll = True
        self._build_ui()

        self._recordArrived.connect(self._append_record, Qt.QueuedConnection)
        self._handler = get_memory_handler()
        for entry in self._handler.snapshot():
            self._append_record(entry)
        self._handler.add_listener(self._on_record)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._level_box = QComboBox()
        self._level_box.addItems(_LEVELS)
        self._level_box.setCurrentText("INFO")
        self._level_box.currentTextChanged.connect(self._on_level_changed)
        self._level_box.setToolTip("Minimum level to display")
        controls.addWidget(self._level_box)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter text...")
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        controls.addWidget(self._filter_edit, 1)

        self._autoscroll_box = QCheckBox("Follow")
        self._autoscroll_box.setChecked(True)
        self._autoscroll_box.toggled.connect(self._on_autoscroll_toggled)
        controls.addWidget(self._autoscroll_box)

        layout.addLayout(controls)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(20000)
        self._view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._view.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 11px;"
        )
        layout.addWidget(self._view, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        export_button = QPushButton("Export log...")
        export_button.clicked.connect(self._export)
        buttons.addWidget(export_button)

        open_button = QPushButton("Open log file location")
        open_button.clicked.connect(self._open_location)
        buttons.addWidget(open_button)

        clear_button = QPushButton("Clear view")
        clear_button.setToolTip(
            "Clears only this view. The log file on disk is never truncated."
        )
        clear_button.clicked.connect(self._view.clear)
        buttons.addWidget(clear_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    # ---------------------------------------------------------------- records
    def _on_record(self, entry: LogRecordEntry) -> None:
        """Handler-thread callback; hand the record to the GUI thread."""
        self._recordArrived.emit(entry)

    @pyqtSlot(object)
    def _append_record(self, entry: LogRecordEntry) -> None:
        """Append one record to the view, honouring the current filters."""
        if entry.levelno < self._min_level:
            return
        text = entry.formatted()
        if self._filter_text and self._filter_text not in text.lower():
            return

        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(_LEVEL_COLOURS.get(entry.level, Palette.FG_0)))
        cursor.insertText(text + "\n", fmt)

        if self._autoscroll:
            self._view.verticalScrollBar().setValue(
                self._view.verticalScrollBar().maximum()
            )

    def _rebuild(self) -> None:
        """Re-render the buffer after a filter change."""
        self._view.clear()
        for entry in self._handler.snapshot():
            self._append_record(entry)

    # --------------------------------------------------------------- handlers
    def _on_level_changed(self, level_name: str) -> None:
        self._min_level = getattr(logging, level_name, logging.INFO)
        self._rebuild()

    def _on_filter_changed(self, text: str) -> None:
        self._filter_text = text.strip().lower()
        self._rebuild()

    def _on_autoscroll_toggled(self, enabled: bool) -> None:
        self._autoscroll = bool(enabled)

    def _export(self) -> None:
        """Write the visible log to a text file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export application log", "forensicvision_log.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._view.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", f"Could not write the log:\n{exc}")
            return
        logger.info("Application log exported to %s", path)

    def _open_location(self) -> None:
        """Reveal the rotating log file in the OS file manager."""
        path = get_log_file_path()
        if path is None:
            QMessageBox.information(self, "Log file", "No log file is active.")
            return
        from gui.utils import reveal_in_file_manager

        reveal_in_file_manager(path)

    def closeEvent(self, event) -> None:
        """Detach the log listener when the panel is destroyed."""
        try:
            self._handler.remove_listener(self._on_record)
        except Exception:  # pragma: no cover - defensive
            pass
        super().closeEvent(event)
