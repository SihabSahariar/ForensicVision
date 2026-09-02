"""Detection dialog for locating regions of interest.

Two sources: general object detection (Ultralytics YOLO) and face detection
(OpenCV YuNet). Neither identifies anyone - they locate regions an examiner may
want to look at, and the face landmarks exist only to drive the alignment that
face restoration requires.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.image_io import ImageData
from detection.face import FaceDetection, YuNetDetector
from detection.face import detector_available as face_detector_available
from detection.yolo import COCO_GROUPS, Detection, ObjectDetector, detector_available
from gui.roi_tools import ROI
from gui.theme import Palette
from gui.widgets.common import BannerLabel, SectionLabel
from workers.base import FunctionWorker

logger = logging.getLogger(__name__)

__all__ = ["DetectDialog"]

#: Inter-ocular distance below which a face carries very little information.
_LOW_INFORMATION_IOD = 30.0

AnyDetection = Union[Detection, FaceDetection]


class DetectDialog(QDialog):
    """Runs detection and turns a result into an ROI.

    Signals:
        roiChosen: ``(ROI)`` when a detection is selected for examination.
        faceRestoreRequested: ``(ROI)`` when the user asks to restore a face.
    """

    roiChosen = pyqtSignal(object)
    faceRestoreRequested = pyqtSignal(object)

    def __init__(self, image: ImageData, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detect regions of interest")
        self.setMinimumSize(820, 580)
        self._image = image
        self._detections: List[AnyDetection] = []
        self._worker: Optional[FunctionWorker] = None
        self._object_detector = ObjectDetector()
        self._face_detector = YuNetDetector()
        self._build_ui()
        self._on_source_changed()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("Detection")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        layout.addWidget(
            BannerLabel(
                "Detection locates regions worth examining. Class labels are "
                "the detector's estimate and are not identifications. Face "
                "landmarks are used only to compute an alignment transform; "
                "nothing here performs recognition or matching."
            )
        )

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        source_row.addWidget(QLabel("Detect:"))
        self._source = QComboBox()
        self._source.addItem("Objects (YOLO)", "objects")
        self._source.addItem("Faces (YuNet)", "faces")
        self._source.currentIndexChanged.connect(self._on_source_changed)
        source_row.addWidget(self._source)

        source_row.addWidget(QLabel("Minimum confidence:"))
        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.05, 0.95)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(0.25)
        source_row.addWidget(self._confidence)
        source_row.addStretch(1)

        self._run_button = QPushButton("Detect")
        self._run_button.setProperty("accent", True)
        self._run_button.clicked.connect(self._on_detect)
        source_row.addWidget(self._run_button)
        layout.addLayout(source_row)

        # Source-specific options.
        self._options = QStackedWidget()

        object_page = QWidget()
        object_layout = QHBoxLayout(object_page)
        object_layout.setContentsMargins(0, 0, 0, 0)
        object_layout.setSpacing(8)
        self._groups = {}
        for group in COCO_GROUPS:
            box = QCheckBox(group.title())
            box.setChecked(group in ("person", "vehicle"))
            self._groups[group] = box
            object_layout.addWidget(box)
        self._all_classes = QCheckBox("All classes")
        object_layout.addWidget(self._all_classes)
        object_layout.addStretch(1)
        self._options.addWidget(object_page)

        face_page = QWidget()
        face_layout = QHBoxLayout(face_page)
        face_layout.setContentsMargins(0, 0, 0, 0)
        face_layout.setSpacing(8)
        face_layout.addWidget(QLabel("Minimum face size (px):"))
        self._min_face = QSpinBox()
        self._min_face.setRange(6, 512)
        self._min_face.setValue(YuNetDetector.DEFAULT_MIN_SIZE)
        self._min_face.setSingleStep(2)
        self._min_face.setToolTip(
            "Detections smaller than this are discarded. The default is "
            "deliberately low: small faces are the normal case in surveillance "
            "stills, and a generous filter would hide the very detections you "
            "are looking for."
        )
        face_layout.addWidget(self._min_face)
        face_layout.addStretch(1)
        self._options.addWidget(face_page)

        layout.addWidget(self._options)

        self._availability = BannerLabel("")
        self._availability.setVisible(False)
        layout.addWidget(self._availability)

        layout.addWidget(SectionLabel("Detections"))
        self._table = QTableWidget(0, 4)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.itemDoubleClicked.connect(lambda _item: self._on_use_roi())
        self._table.itemSelectionChanged.connect(self._update_buttons)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self._table, 1)

        self._status = QLabel("")
        self._status.setProperty("role", "hint")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)

        self._restore_button = QPushButton("Enhance Face...")
        self._restore_button.setToolTip(
            "Select this face as the region of interest and open the face "
            "restoration model."
        )
        self._restore_button.setVisible(False)
        self._restore_button.setEnabled(False)
        self._restore_button.clicked.connect(self._on_restore_face)
        buttons.addButton(self._restore_button, QDialogButtonBox.ActionRole)

        self._use_button = QPushButton("Use as region of interest")
        self._use_button.setProperty("accent", True)
        self._use_button.setEnabled(False)
        self._use_button.clicked.connect(self._on_use_roi)
        buttons.addButton(self._use_button, QDialogButtonBox.AcceptRole)

        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---------------------------------------------------------------- helpers
    @property
    def _mode(self) -> str:
        """Currently selected detection source."""
        return self._source.currentData()

    def _on_source_changed(self) -> None:
        """Swap the options page and re-check availability."""
        faces = self._mode == "faces"
        self._options.setCurrentIndex(1 if faces else 0)
        self._restore_button.setVisible(faces)
        self._table.setHorizontalHeaderLabels(
            ["Face", "Confidence", "Size", "Inter-ocular"]
            if faces
            else ["Label", "Confidence", "Size", "Position"]
        )
        self._table.setRowCount(0)
        self._detections = []
        self._status.setText("")
        self._update_buttons()

        if faces:
            available, reason = face_detector_available()
        else:
            available = detector_available()
            reason = (
                ""
                if available
                else "The 'ultralytics' package is not installed.\n\n"
                "Install it with:  pip install ultralytics\n\n"
                "Object detection is optional; the rest of the application is "
                "unaffected."
            )

        self._run_button.setEnabled(available)
        self._availability.setVisible(not available)
        if not available:
            self._availability.setText(reason)

    def _update_buttons(self) -> None:
        """Enable the action buttons when a row is selected."""
        has_selection = self._table.currentRow() >= 0
        self._use_button.setEnabled(has_selection)
        self._restore_button.setEnabled(has_selection and self._mode == "faces")

    # --------------------------------------------------------------- handlers
    def _on_detect(self) -> None:
        """Run the selected detector off the GUI thread."""
        confidence = self._confidence.value()
        pixels = self._image.pixels

        if self._mode == "faces":
            detector = self._face_detector
            detector.unload()
            detector._score_threshold = confidence  # noqa: SLF001
            minimum = self._min_face.value()

            def work() -> List[AnyDetection]:
                return list(detector.detect(pixels, min_size=minimum))

            message = "Detecting faces..."
        else:
            groups = None
            if not self._all_classes.isChecked():
                groups = [n for n, b in self._groups.items() if b.isChecked()]
                if not groups:
                    QMessageBox.information(
                        self, "Select a class group",
                        "Choose at least one class group, or tick 'All classes'.",
                    )
                    return
            detector = self._object_detector

            def work() -> List[AnyDetection]:
                return list(
                    detector.detect(pixels, confidence=confidence, groups=groups)
                )

            message = (
                "Detecting objects... the first run downloads the detector "
                "weights."
            )

        self._run_button.setEnabled(False)
        self._status.setText(message)
        worker = FunctionWorker(work, description="Detecting")
        worker.finished_work.connect(self._on_finished)
        worker.error.connect(self._on_error)
        self._worker = worker
        worker.start()

    def _on_finished(self, detections: List[AnyDetection]) -> None:
        """Populate the results table."""
        self._worker = None
        self._run_button.setEnabled(True)
        self._detections = detections
        faces = self._mode == "faces"

        self._table.setRowCount(len(detections))
        low_information = 0
        for row, detection in enumerate(detections):
            if faces:
                iod = detection.inter_ocular_distance
                values = [
                    f"Face {row + 1}",
                    f"{detection.confidence * 100:.0f}%",
                    f"{detection.width} x {detection.height}",
                    f"{iod:.0f} px",
                ]
            else:
                values = [
                    detection.label,
                    f"{detection.confidence * 100:.0f}%",
                    f"{detection.width} x {detection.height}",
                    f"({detection.box[0]}, {detection.box[1]})",
                ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if faces and column == 3 and detection.inter_ocular_distance < _LOW_INFORMATION_IOD:
                    from PyQt5.QtGui import QBrush, QColor

                    item.setForeground(QBrush(QColor(Palette.WARN)))
                    item.setToolTip(
                        "Very few pixels between the eyes. A restored face "
                        "would be dominated by the learned prior rather than "
                        "by measured detail."
                    )
                    low_information += 1
                self._table.setItem(row, column, item)

        if not detections:
            self._status.setText("No detections.")
        elif faces:
            text = (
                f"{len(detections)} face(s). Double-click a row to use it as a "
                "region of interest."
            )
            if low_information:
                text += (
                    f"  {low_information} of them have an inter-ocular distance "
                    f"below {_LOW_INFORMATION_IOD:.0f} px - restoration of "
                    "those would be almost entirely synthesised."
                )
            self._status.setText(text)
        else:
            self._status.setText(
                f"{len(detections)} detection(s). Double-click a row to use it "
                "as a region of interest."
            )
        self._update_buttons()

    def _on_error(self, message: str, detail: str) -> None:
        self._worker = None
        self._run_button.setEnabled(True)
        self._status.setText("Detection failed")
        QMessageBox.warning(self, "Detection failed", message)

    def _selected(self) -> Optional[AnyDetection]:
        """Return the selected detection."""
        row = self._table.currentRow()
        if 0 <= row < len(self._detections):
            return self._detections[row]
        return None

    def _on_use_roi(self) -> None:
        """Emit the selected detection as an ROI."""
        detection = self._selected()
        if detection is None:
            return
        self.roiChosen.emit(detection.to_roi())
        self.accept()

    def _on_restore_face(self) -> None:
        """Ask the main window to open face restoration for this face."""
        detection = self._selected()
        if detection is None:
            return
        if detection.inter_ocular_distance < _LOW_INFORMATION_IOD:
            answer = QMessageBox.warning(
                self,
                "Very little facial information",
                f"This face has only {detection.inter_ocular_distance:.0f} "
                "pixels between the eyes.\n\n"
                "A restoration would be reconstructed almost entirely from the "
                "model's learned prior over faces, not from measured detail in "
                "this frame. The result may look convincing and still bear no "
                "relation to the person depicted.\n\n"
                "Continue?",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
        self.faceRestoreRequested.emit(detection.to_roi())
        self.accept()

    def closeEvent(self, event) -> None:
        """Stop detection before closing."""
        if self._worker is not None:
            self._worker.stop_and_wait(5000)
        super().closeEvent(event)
