"""OCR dialog: compare readings from the original and the derivative."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.constants import OCR_DISCLAIMER
from core.image_io import ImageData
from gui.widgets.common import BannerLabel, SectionLabel
from ocr.engine import OcrEngine, OcrResult, available_engines
from workers.base import FunctionWorker

logger = logging.getLogger(__name__)

__all__ = ["OcrDialog"]


class OcrDialog(QDialog):
    """Runs OCR on the original and (optionally) the enhanced image."""

    def __init__(
        self,
        original: Optional[ImageData],
        enhanced: Optional[ImageData] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("OCR interpretation")
        self.setMinimumSize(720, 560)
        self._original = original
        self._enhanced = enhanced
        self._worker: Optional[FunctionWorker] = None
        self._results = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("Machine-generated OCR interpretation")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        layout.addWidget(BannerLabel(OCR_DISCLAIMER))

        engines = available_engines()
        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("Engine:"))
        self._engine_combo = QComboBox()
        if engines:
            self._engine_combo.addItem("Automatic", "auto")
            for engine in engines:
                self._engine_combo.addItem(engine, engine)
        else:
            self._engine_combo.addItem("No engine installed", "none")
            self._engine_combo.setEnabled(False)
        controls.addWidget(self._engine_combo)

        controls.addWidget(QLabel("Upscale:"))
        self._upscale = QSpinBox()
        self._upscale.setRange(1, 6)
        self._upscale.setValue(2)
        self._upscale.setToolTip(
            "Interpolated enlargement applied to the OCR input only. It never "
            "affects the displayed image or any exported derivative."
        )
        controls.addWidget(self._upscale)

        self._preprocess = QCheckBox("Contrast normalisation")
        self._preprocess.setChecked(True)
        controls.addWidget(self._preprocess)
        controls.addStretch(1)

        self._run_button = QPushButton("Run OCR")
        self._run_button.setProperty("accent", True)
        self._run_button.setEnabled(bool(engines))
        self._run_button.clicked.connect(self._on_run)
        controls.addWidget(self._run_button)
        layout.addLayout(controls)

        if not engines:
            layout.addWidget(
                BannerLabel(
                    "No OCR engine is installed. Install one of:\n"
                    "  pip install pytesseract   (plus the Tesseract binary on PATH)\n"
                    "  pip install paddleocr paddlepaddle\n\n"
                    "OCR is optional; the rest of the application is unaffected."
                )
            )

        layout.addWidget(SectionLabel("OCR before  (original evidence)"))
        self._before = QPlainTextEdit()
        self._before.setReadOnly(True)
        self._before.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 12px;"
        )
        layout.addWidget(self._before, 1)

        layout.addWidget(SectionLabel("OCR after  (enhanced derivative)"))
        self._after = QPlainTextEdit()
        self._after.setReadOnly(True)
        self._after.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 12px;"
        )
        if self._enhanced is None:
            self._after.setPlainText(
                "(no enhanced derivative is loaded - run a restoration first)"
            )
        layout.addWidget(self._after, 1)

        self._status = QLabel("")
        self._status.setProperty("role", "hint")
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ public
    @property
    def results(self) -> dict:
        """Readings keyed by ``"original"`` / ``"enhanced"``."""
        return dict(self._results)

    # --------------------------------------------------------------- handlers
    def _on_run(self) -> None:
        """Run OCR on both images off the GUI thread."""
        if self._original is None and self._enhanced is None:
            return

        engine_name = self._engine_combo.currentData() or "auto"
        preprocess = self._preprocess.isChecked()
        upscale = self._upscale.value()
        original = self._original
        enhanced = self._enhanced

        def work() -> dict:
            engine = OcrEngine(engine_name)
            output = {}
            if original is not None:
                output["original"] = engine.read(
                    original.pixels, preprocess=preprocess, upscale=upscale
                )
            if enhanced is not None:
                output["enhanced"] = engine.read(
                    enhanced.pixels, preprocess=preprocess, upscale=upscale
                )
            return output

        self._run_button.setEnabled(False)
        self._status.setText("Running OCR...")

        worker = FunctionWorker(work, description="Running OCR")
        worker.finished_work.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    def _on_finished(self, results: dict) -> None:
        """Render both readings."""
        self._worker = None
        self._run_button.setEnabled(True)
        self._results = results

        before: Optional[OcrResult] = results.get("original")
        after: Optional[OcrResult] = results.get("enhanced")

        if before is not None:
            self._before.setPlainText(self._format(before))
        if after is not None:
            self._after.setPlainText(self._format(after))

        if before is not None and after is not None and before.ok and after.ok:
            if before.text.strip() != after.text.strip():
                self._status.setText(
                    "The readings differ. A difference means the enhancement "
                    "changed what the OCR engine saw - not that the enhanced "
                    "reading is correct."
                )
            else:
                self._status.setText("Both readings agree.")
        else:
            self._status.setText("OCR complete.")

    @staticmethod
    def _format(result: OcrResult) -> str:
        """Render an OCR result with its confidence detail."""
        if not result.ok:
            return f"(error: {result.error})"
        lines = [result.summary(), ""]
        lines.append(f"--- engine: {result.engine}, "
                     f"mean confidence {result.mean_confidence * 100:.0f}% ---")
        for text, confidence in result.lines[:60]:
            lines.append(f"  {confidence * 100:5.1f}%  {text}")
        return "\n".join(lines)

    def _on_error(self, message: str, detail: str) -> None:
        self._worker = None
        self._run_button.setEnabled(True)
        self._status.setText("OCR failed")
        QMessageBox.warning(self, "OCR failed", message)

    def closeEvent(self, event) -> None:
        """Stop the worker before closing."""
        if self._worker is not None:
            self._worker.stop_and_wait(4000)
        super().closeEvent(event)
