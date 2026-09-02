"""New-case dialog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from core.case_manager import next_case_id
from gui.widgets.common import BannerLabel

logger = logging.getLogger(__name__)

__all__ = ["NewCaseDialog"]


class NewCaseDialog(QDialog):
    """Collects the details needed to create a case."""

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New case")
        self.setMinimumWidth(560)
        self._config = config
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("Create a forensic case")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        form = QFormLayout()
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        location = QHBoxLayout()
        location.setSpacing(6)
        self._location_edit = QLineEdit(str(self._config.cases_path))
        location.addWidget(self._location_edit, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        location.addWidget(browse)
        form.addRow("Cases folder", location)

        self._case_id_edit = QLineEdit(next_case_id(self._config.cases_path))
        self._case_id_edit.setToolTip(
            "Used as the folder name; keep it filesystem-safe."
        )
        form.addRow("Case ID", self._case_id_edit)

        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("e.g. Forecourt CCTV - vehicle identification")
        form.addRow("Title", self._title_edit)

        self._investigator_edit = QLineEdit()
        self._investigator_edit.setPlaceholderText("Examiner name recorded in reports")
        form.addRow("Investigator", self._investigator_edit)

        self._organisation_edit = QLineEdit()
        form.addRow("Organisation", self._organisation_edit)

        self._description_edit = QPlainTextEdit()
        self._description_edit.setPlaceholderText(
            "Scope of examination, request reference, any relevant background."
        )
        self._description_edit.setMaximumHeight(90)
        form.addRow("Description", self._description_edit)

        layout.addLayout(form)

        self._safe_mode_box = QCheckBox("Enable Forensic Safe Mode for this case")
        self._safe_mode_box.setChecked(True)
        layout.addWidget(self._safe_mode_box)

        layout.addWidget(
            BannerLabel(
                "Imported originals are copied into the case, hashed with "
                "SHA-256/512, and write-protected. Every operation is recorded "
                "in the case database and audit trail."
            )
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Create case")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ public
    @property
    def cases_root(self) -> Path:
        """Parent directory chosen for the case."""
        return Path(self._location_edit.text().strip())

    @property
    def case_id(self) -> str:
        """Chosen case identifier."""
        return self._case_id_edit.text().strip()

    @property
    def title(self) -> str:
        """Case title."""
        return self._title_edit.text().strip()

    @property
    def investigator(self) -> str:
        """Investigator name."""
        return self._investigator_edit.text().strip()

    @property
    def organisation(self) -> str:
        """Organisation name."""
        return self._organisation_edit.text().strip()

    @property
    def description(self) -> str:
        """Case description."""
        return self._description_edit.toPlainText().strip()

    @property
    def safe_mode(self) -> bool:
        """Whether safe mode should be enabled."""
        return self._safe_mode_box.isChecked()

    # --------------------------------------------------------------- handlers
    def _browse(self) -> None:
        """Choose the parent directory for cases."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select the folder that will hold cases", self._location_edit.text()
        )
        if directory:
            self._location_edit.setText(directory)
            self._case_id_edit.setText(next_case_id(Path(directory)))

    def _on_accept(self) -> None:
        """Validate before closing."""
        from PyQt5.QtWidgets import QMessageBox

        if not self.case_id:
            QMessageBox.warning(self, "Case ID required", "Enter a case identifier.")
            return
        target = self.cases_root / self.case_id
        if target.exists() and any(target.iterdir()):
            QMessageBox.warning(
                self,
                "Folder in use",
                f"{target} already exists and is not empty.\n\n"
                "Choose a different case ID or open the existing case instead.",
            )
            return
        self.accept()
