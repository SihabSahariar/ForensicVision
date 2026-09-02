"""Pipeline review and editing dialog.

Shown before every auto-restoration run. The investigator can reorder, disable,
remove and re-parameterise steps; nothing executes until Run is pressed. This
is the review gate the specification requires (S18).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants import MODEL_KIND_LABELS, ModelKind
from restoration.auto_engine import Recommendation
from restoration.base import ModelInfo
from restoration.pipeline import Pipeline, PipelineStep
from restoration.registry import ModelRegistry
from gui.restoration_panel import ParameterEditor
from gui.theme import Palette
from gui.widgets.common import BannerLabel, SectionLabel

logger = logging.getLogger(__name__)

__all__ = ["PipelineEditorDialog"]


class PipelineEditorDialog(QDialog):
    """Review, edit and approve a restoration pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        recommendation: Optional[Recommendation] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recommended pipeline" if recommendation else "Edit pipeline")
        self.setMinimumSize(880, 620)
        self._pipeline = pipeline.copy()
        self._recommendation = recommendation
        self._build_ui()
        self._refresh_list()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel(
            "Review the proposed pipeline"
            if self._recommendation
            else "Edit the pipeline"
        )
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        subtitle = QLabel(
            "Nothing runs until you press Run. Steps can be reordered, "
            "disabled, removed or re-parameterised."
        )
        subtitle.setProperty("role", "hint")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # ---- left: step list ------------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(SectionLabel("Steps"))

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["#", "Operation", "Kind"])
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        left_layout.addWidget(self._tree, 1)

        controls = QHBoxLayout()
        controls.setSpacing(4)
        for label, slot, tip in (
            ("Up", self._move_up, "Move the selected step earlier"),
            ("Down", self._move_down, "Move the selected step later"),
            ("Remove", self._remove, "Remove the selected step"),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            button.setToolTip(tip)
            controls.addWidget(button)
        controls.addStretch(1)
        left_layout.addLayout(controls)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        self._add_combo = QComboBox()
        self._populate_add_combo()
        add_row.addWidget(self._add_combo, 1)
        add_button = QPushButton("Add step")
        add_button.clicked.connect(self._add_step)
        add_row.addWidget(add_button)
        left_layout.addLayout(add_row)

        splitter.addWidget(left)

        # ---- right: parameters and rationale --------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(6)

        right_layout.addWidget(SectionLabel("Step parameters"))
        self._model_label = QLabel("Select a step")
        self._model_label.setWordWrap(True)
        self._model_label.setProperty("role", "hint")
        right_layout.addWidget(self._model_label)

        self._editor = ParameterEditor()
        right_layout.addWidget(self._editor)

        apply_button = QPushButton("Apply parameters to this step")
        apply_button.clicked.connect(self._apply_parameters)
        right_layout.addWidget(apply_button)

        right_layout.addWidget(SectionLabel("Rationale"))
        self._rationale = QPlainTextEdit()
        self._rationale.setReadOnly(True)
        self._rationale.setStyleSheet("font-size: 11px;")
        if self._recommendation is not None:
            self._rationale.setPlainText(self._recommendation.rationale_text())
        else:
            self._rationale.setPlainText(self._pipeline.rationale or "(no rationale)")
        right_layout.addWidget(self._rationale, 1)

        splitter.addWidget(right)
        splitter.setSizes([420, 460])

        self._warning_banner = BannerLabel("")
        self._warning_banner.setVisible(False)
        layout.addWidget(self._warning_banner)

        self._issues_label = QLabel("")
        self._issues_label.setProperty("role", "warning")
        self._issues_label.setWordWrap(True)
        layout.addWidget(self._issues_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._run_button = QPushButton("Run pipeline")
        self._run_button.setProperty("accent", True)
        self._run_button.setDefault(True)
        self._run_button.clicked.connect(self._on_run)
        buttons.addButton(self._run_button, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_add_combo(self) -> None:
        """Fill the add-step combo with every available model."""
        self._add_combo.clear()
        for info in ModelRegistry.infos():
            model = ModelRegistry.try_get(info.name)
            if model is None or not model.availability().ok:
                continue
            marker = "" if info.kind == ModelKind.CLASSICAL.value else " [neural]"
            self._add_combo.addItem(
                f"{info.task_label}: {info.display_name}{marker}", info.name
            )

    # ------------------------------------------------------------------ public
    @property
    def pipeline(self) -> Pipeline:
        """The edited pipeline."""
        return self._pipeline

    # ---------------------------------------------------------------- refresh
    def _refresh_list(self) -> None:
        """Rebuild the step list from the pipeline."""
        self._tree.blockSignals(True)
        self._tree.clear()
        for index, step in enumerate(self._pipeline.steps, start=1):
            info = step.info()
            kind = MODEL_KIND_LABELS.get(info.kind, "") if info else "unknown"
            item = QTreeWidgetItem([str(index), step.describe(), kind.split(" ")[0]])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if step.enabled else Qt.Unchecked)
            if step.may_synthesise:
                item.setForeground(1, QBrush(QColor(Palette.WARN)))
            if step.note:
                item.setToolTip(1, step.note)
            self._tree.addTopLevelItem(item)
        self._tree.blockSignals(False)
        self._update_validation()

    def _update_validation(self) -> None:
        """Re-run validation and update the warning banners."""
        issues = self._pipeline.validate()
        self._issues_label.setText(
            "\n".join(f"- {issue}" for issue in issues) if issues else ""
        )
        blocking = [
            issue for issue in issues
            if "not registered" in issue or "cannot run" in issue
            or "no enabled steps" in issue.lower()
            or "Weight file" in issue or "not installed" in issue
        ]
        self._run_button.setEnabled(not blocking)

        if self._pipeline.may_synthesise:
            self._warning_banner.setText(
                "This pipeline contains a generative step. The output may "
                "include structures that are not present in the source "
                "evidence, and must be reported as a derivative representation."
            )
            self._warning_banner.setVisible(True)
        else:
            self._warning_banner.setText(
                "All steps are deterministic signal processing. This pipeline "
                "cannot introduce structures absent from the measured samples."
            )
            self._warning_banner.setVisible(True)

    def _selected_index(self) -> int:
        """Return the selected row, or -1."""
        items = self._tree.selectedItems()
        if not items:
            return -1
        return self._tree.indexOfTopLevelItem(items[0])

    # --------------------------------------------------------------- handlers
    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Toggle a step's enabled flag."""
        if column != 0:
            return
        index = self._tree.indexOfTopLevelItem(item)
        if 0 <= index < len(self._pipeline.steps):
            self._pipeline.steps[index].enabled = item.checkState(0) == Qt.Checked
            self._update_validation()

    def _on_selection_changed(self) -> None:
        """Load the selected step's parameters into the editor."""
        index = self._selected_index()
        if index < 0:
            self._editor.set_specs([])
            self._model_label.setText("Select a step")
            return
        step = self._pipeline.steps[index]
        info = step.info()
        if info is None:
            self._editor.set_specs([])
            self._model_label.setText(
                f"'{step.model_name}' is not registered in this installation."
            )
            return
        self._model_label.setText(
            f"<b>{info.display_name}</b><br>{info.method or info.description}"
        )
        self._editor.set_specs(list(info.parameters))
        self._apply_values_to_editor(step.parameters)

    def _apply_values_to_editor(self, values: Dict[str, Any]) -> None:
        """Push stored values into the freshly built controls."""
        from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

        for name, widget in self._editor._widgets.items():  # noqa: SLF001
            if name not in values:
                continue
            value = values[name]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                position = widget.findData(value)
                if position >= 0:
                    widget.setCurrentIndex(position)
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))

    def _apply_parameters(self) -> None:
        """Store the editor's values back onto the selected step."""
        index = self._selected_index()
        if index < 0:
            return
        self._pipeline.steps[index].parameters.update(self._editor.values())
        self._refresh_list()
        self._tree.setCurrentItem(self._tree.topLevelItem(index))

    def _move_up(self) -> None:
        """Move the selected step one position earlier."""
        index = self._selected_index()
        if index > 0 and self._pipeline.move(index, -1):
            self._refresh_list()
            self._tree.setCurrentItem(self._tree.topLevelItem(index - 1))

    def _move_down(self) -> None:
        """Move the selected step one position later."""
        index = self._selected_index()
        if index >= 0 and self._pipeline.move(index, 1):
            self._refresh_list()
            self._tree.setCurrentItem(self._tree.topLevelItem(index + 1))

    def _remove(self) -> None:
        """Remove the selected step."""
        index = self._selected_index()
        if index >= 0 and self._pipeline.remove(index):
            self._refresh_list()

    def _add_step(self) -> None:
        """Append the model selected in the combo."""
        name = self._add_combo.currentData()
        if not name:
            return
        info = ModelRegistry.info(name)
        parameters = info.default_parameters() if info else {}
        self._pipeline.add(PipelineStep(model_name=name, parameters=parameters))
        self._refresh_list()
        self._tree.setCurrentItem(
            self._tree.topLevelItem(len(self._pipeline.steps) - 1)
        )

    def _on_run(self) -> None:
        """Validate and accept."""
        issues = self._pipeline.validate()
        blocking = [
            issue for issue in issues
            if "not registered" in issue or "cannot run" in issue
            or "no enabled steps" in issue.lower()
        ]
        if blocking:
            QMessageBox.warning(
                self, "Pipeline cannot run", "\n".join(f"- {i}" for i in blocking)
            )
            return
        self.accept()
