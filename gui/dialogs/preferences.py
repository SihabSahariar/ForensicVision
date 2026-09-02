"""Preferences dialog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.constants import MIN_TILE_SIZE
from core.device import get_device_report
from gui.widgets.common import BannerLabel
from ocr.engine import available_engines

logger = logging.getLogger(__name__)

__all__ = ["PreferencesDialog"]


class PreferencesDialog(QDialog):
    """Edits the process-wide :class:`~app.config.AppConfig`."""

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(620)
        self._config = config
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # -- processing -------------------------------------------------------
        processing = QGroupBox("Processing")
        processing_form = QFormLayout(processing)
        processing_form.setSpacing(8)

        self._device = QComboBox()
        self._device.addItem("Automatic (prefer GPU)", "auto")
        self._device.addItem("CUDA (GPU)", "cuda")
        self._device.addItem("CPU only", "cpu")
        processing_form.addRow("Compute device", self._device)

        report = get_device_report()
        self._cuda_index = QSpinBox()
        self._cuda_index.setRange(0, max(0, report.gpu_count - 1))
        self._cuda_index.setEnabled(report.gpu_count > 1)
        self._cuda_index.setToolTip(
            "Which CUDA device to use when several are installed."
            if report.gpu_count > 1
            else "Only one CUDA device is present."
        )
        processing_form.addRow("CUDA device index", self._cuda_index)

        self._fp16 = QCheckBox(
            "Use half precision (FP16) on CUDA where the model supports it"
        )
        processing_form.addRow("", self._fp16)

        self._tile_size = QSpinBox()
        self._tile_size.setRange(0, 4096)
        self._tile_size.setSingleStep(64)
        self._tile_size.setSpecialValueText("Disabled (whole image)")
        self._tile_size.setToolTip(
            f"Tile edge length for large images. Minimum effective size is "
            f"{MIN_TILE_SIZE} px."
        )
        processing_form.addRow("Tile size (px)", self._tile_size)

        self._tile_overlap = QSpinBox()
        self._tile_overlap.setRange(0, 256)
        self._tile_overlap.setSingleStep(8)
        processing_form.addRow("Tile overlap (px)", self._tile_overlap)

        self._auto_reduce = QCheckBox(
            "Halve the tile size and retry automatically after an out-of-memory error"
        )
        processing_form.addRow("", self._auto_reduce)

        layout.addWidget(processing)

        # -- forensic ---------------------------------------------------------
        forensic = QGroupBox("Forensic")
        forensic_form = QFormLayout(forensic)
        forensic_form.setSpacing(8)

        self._safe_mode = QCheckBox("Enable Forensic Safe Mode by default")
        forensic_form.addRow("", self._safe_mode)

        self._confirm_synthesis = QCheckBox(
            "Confirm before running any operation that can synthesise content"
        )
        forensic_form.addRow("", self._confirm_synthesis)

        self._allow_download = QCheckBox(
            "Permit model-weight downloads from the Model Manager"
        )
        forensic_form.addRow("", self._allow_download)

        layout.addWidget(forensic)

        # -- folders ----------------------------------------------------------
        folders = QGroupBox("Folders")
        folders_form = QFormLayout(folders)
        folders_form.setSpacing(8)

        cases_row = QHBoxLayout()
        cases_row.setSpacing(6)
        self._cases_root = QLineEdit()
        cases_row.addWidget(self._cases_root, 1)
        cases_browse = QPushButton("Browse...")
        cases_browse.clicked.connect(
            lambda: self._browse_into(self._cases_root, "Select the cases folder")
        )
        cases_row.addWidget(cases_browse)
        folders_form.addRow("Cases folder", cases_row)

        weights_row = QHBoxLayout()
        weights_row.setSpacing(6)
        self._weights_root = QLineEdit()
        weights_row.addWidget(self._weights_root, 1)
        weights_browse = QPushButton("Browse...")
        weights_browse.clicked.connect(
            lambda: self._browse_into(self._weights_root, "Select the weights folder")
        )
        weights_row.addWidget(weights_browse)
        folders_form.addRow("Model weights", weights_row)

        layout.addWidget(folders)

        # -- OCR --------------------------------------------------------------
        ocr = QGroupBox("OCR (optional)")
        ocr_form = QFormLayout(ocr)
        ocr_form.setSpacing(8)

        self._ocr_engine = QComboBox()
        self._ocr_engine.addItem("Automatic", "auto")
        for engine in available_engines():
            self._ocr_engine.addItem(engine, engine)
        self._ocr_engine.addItem("Disabled", "none")
        ocr_form.addRow("Preferred engine", self._ocr_engine)

        tesseract_row = QHBoxLayout()
        tesseract_row.setSpacing(6)
        self._tesseract = QLineEdit()
        self._tesseract.setPlaceholderText("Leave empty to use PATH")
        tesseract_row.addWidget(self._tesseract, 1)
        tesseract_browse = QPushButton("Browse...")
        tesseract_browse.clicked.connect(self._browse_tesseract)
        tesseract_row.addWidget(tesseract_browse)
        ocr_form.addRow("Tesseract binary", tesseract_row)

        layout.addWidget(ocr)

        layout.addWidget(
            BannerLabel(
                "Folder changes take effect for newly created cases and newly "
                "installed weights. Existing cases keep their own directories."
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---------------------------------------------------------------- helpers
    def _browse_into(self, field: QLineEdit, caption: str) -> None:
        directory = QFileDialog.getExistingDirectory(self, caption, field.text())
        if directory:
            field.setText(directory)

    def _browse_tesseract(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the tesseract executable", self._tesseract.text()
        )
        if path:
            self._tesseract.setText(path)

    def _load(self) -> None:
        """Populate the controls from the configuration."""
        config = self._config
        index = self._device.findData(config.device)
        self._device.setCurrentIndex(max(0, index))
        self._cuda_index.setValue(config.cuda_index)
        self._fp16.setChecked(config.use_fp16)
        self._tile_size.setValue(config.tile_size)
        self._tile_overlap.setValue(config.tile_overlap)
        self._auto_reduce.setChecked(config.auto_reduce_tile)
        self._safe_mode.setChecked(config.safe_mode)
        self._confirm_synthesis.setChecked(config.confirm_synthesis)
        self._allow_download.setChecked(config.allow_model_download)
        self._cases_root.setText(config.cases_root)
        self._weights_root.setText(config.weights_root)
        ocr_index = self._ocr_engine.findData(config.ocr_engine)
        self._ocr_engine.setCurrentIndex(max(0, ocr_index))
        self._tesseract.setText(config.tesseract_cmd)

    def _on_accept(self) -> None:
        """Write the controls back into the configuration."""
        config = self._config
        config.device = self._device.currentData()
        config.cuda_index = self._cuda_index.value()
        config.use_fp16 = self._fp16.isChecked()
        config.tile_size = self._tile_size.value()
        config.tile_overlap = self._tile_overlap.value()
        config.auto_reduce_tile = self._auto_reduce.isChecked()
        config.safe_mode = self._safe_mode.isChecked()
        config.confirm_synthesis = self._confirm_synthesis.isChecked()
        config.allow_model_download = self._allow_download.isChecked()
        config.cases_root = self._cases_root.text().strip() or config.cases_root
        config.weights_root = self._weights_root.text().strip() or config.weights_root
        config.ocr_engine = self._ocr_engine.currentData()
        config.tesseract_cmd = self._tesseract.text().strip()
        self.accept()
