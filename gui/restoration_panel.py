"""Manual restoration dock.

One collapsible section per task. Each section lists the models registered for
that task, generates its parameter controls from the model's
:class:`~restoration.base.ParamSpec` declarations, and shows availability
honestly: an unavailable model is selectable but its Run button is replaced by
an Install prompt, never by a substitute result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.constants import MODEL_KIND_LABELS, TASK_LABELS, TASK_ORDER, ModelKind
from restoration.base import ModelInfo, ParamSpec
from restoration.pipeline import Pipeline, PipelineStep
from restoration.registry import ModelRegistry
from gui.theme import Palette
from gui.widgets.common import CollapsibleSection, HLine, SectionLabel

logger = logging.getLogger(__name__)

__all__ = ["RestorationPanel", "TaskSection"]


class ParameterEditor(QWidget):
    """Builds and reads the controls for one model's parameters."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(5)
        self._layout.setLabelAlignment(Qt.AlignLeft)
        self._layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._widgets: Dict[str, QWidget] = {}
        self._specs: Dict[str, ParamSpec] = {}

    def set_specs(self, specs: List[ParamSpec]) -> None:
        """Rebuild the controls for ``specs``."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._widgets.clear()
        self._specs.clear()

        for spec in specs:
            widget = self._build_widget(spec)
            if widget is None:
                continue
            if spec.help_text:
                widget.setToolTip(spec.help_text)
            label = QLabel(spec.label)
            if spec.help_text:
                label.setToolTip(spec.help_text)
            self._layout.addRow(label, widget)
            self._widgets[spec.name] = widget
            self._specs[spec.name] = spec

        self.setVisible(bool(self._widgets))

    def _build_widget(self, spec: ParamSpec) -> Optional[QWidget]:
        """Create the control matching ``spec.kind``."""
        if spec.kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(spec.default))
            return box
        if spec.kind == "choice":
            combo = QComboBox()
            for value, label in spec.choices:
                combo.addItem(label, value)
            index = combo.findData(spec.default)
            combo.setCurrentIndex(max(0, index))
            return combo
        if spec.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(spec.minimum), int(spec.maximum))
            spin.setSingleStep(max(1, int(spec.step)))
            spin.setValue(int(spec.default))
            return spin
        spin = QDoubleSpinBox()
        spin.setRange(float(spec.minimum), float(spec.maximum))
        spin.setSingleStep(float(spec.step))
        spin.setDecimals(4 if spec.step < 0.01 else 2)
        spin.setValue(float(spec.default))
        return spin

    def values(self) -> Dict[str, Any]:
        """Return the current parameter values."""
        result: Dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentData()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                result[name] = widget.value()
        return result


class TaskSection(CollapsibleSection):
    """A collapsible group of models for one restoration task.

    Signals:
        runRequested: ``(model_name, parameters)``
        previewRequested: ``(model_name, parameters)``
        addToPipelineRequested: ``(model_name, parameters)``
        installRequested: ``(model_name)``
    """

    runRequested = pyqtSignal(str, dict)
    previewRequested = pyqtSignal(str, dict)
    addToPipelineRequested = pyqtSignal(str, dict)
    installRequested = pyqtSignal(str)

    def __init__(self, task: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(TASK_LABELS.get(task, task), expanded=False, parent=parent)
        self._task = task
        self._infos: List[ModelInfo] = []
        self._build()

    def _build(self) -> None:
        self._model_combo = QComboBox()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.add_widget(self._model_combo)

        self._kind_label = QLabel("")
        self._kind_label.setProperty("role", "hint")
        self._kind_label.setWordWrap(True)
        self.add_widget(self._kind_label)

        self._method_label = QLabel("")
        self._method_label.setProperty("role", "hint")
        self._method_label.setWordWrap(True)
        self.add_widget(self._method_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self.add_widget(self._status_label)

        self._editor = ParameterEditor()
        self.add_widget(self._editor)

        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        self._preview_button = QPushButton("Preview")
        self._preview_button.setToolTip(
            "Run on the visible region only, without writing a derivative."
        )
        self._preview_button.clicked.connect(
            lambda: self.previewRequested.emit(self.current_model(), self.parameters())
        )
        buttons.addWidget(self._preview_button)

        self._add_button = QPushButton("Add to pipeline")
        self._add_button.clicked.connect(
            lambda: self.addToPipelineRequested.emit(
                self.current_model(), self.parameters()
            )
        )
        buttons.addWidget(self._add_button)

        self._run_button = QPushButton("Run")
        self._run_button.setProperty("accent", True)
        self._run_button.clicked.connect(
            lambda: self.runRequested.emit(self.current_model(), self.parameters())
        )
        buttons.addWidget(self._run_button)

        self._install_button = QPushButton("Install model")
        self._install_button.setVisible(False)
        self._install_button.clicked.connect(
            lambda: self.installRequested.emit(self.current_model())
        )
        buttons.addWidget(self._install_button)

        self.add_layout(buttons)

    # ------------------------------------------------------------------ public
    def refresh(self) -> None:
        """Reload the model list and availability for this task."""
        current = self.current_model()
        self._infos = ModelRegistry.by_task(self._task)
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for info in self._infos:
            marker = "" if info.kind == ModelKind.CLASSICAL.value else "  [neural]"
            self._model_combo.addItem(f"{info.display_name}{marker}", info.name)
        index = self._model_combo.findData(current)
        self._model_combo.setCurrentIndex(max(0, index))
        self._model_combo.blockSignals(False)
        self._on_model_changed()

    @property
    def task(self) -> str:
        """The task this section covers."""
        return self._task

    @property
    def has_models(self) -> bool:
        """Whether any model is registered for this task."""
        return bool(self._infos)

    def current_model(self) -> str:
        """Registry key of the selected model."""
        return self._model_combo.currentData() or ""

    def current_info(self) -> Optional[ModelInfo]:
        """The selected model's :class:`ModelInfo`."""
        name = self.current_model()
        return ModelRegistry.info(name) if name else None

    def parameters(self) -> Dict[str, Any]:
        """Current parameter values for the selected model."""
        return self._editor.values()

    # --------------------------------------------------------------- handlers
    def _on_model_changed(self) -> None:
        """Update descriptions, parameters and availability."""
        info = self.current_info()
        if info is None:
            self._editor.set_specs([])
            self._status_label.setText("")
            self._method_label.setText("")
            self._kind_label.setText("")
            for button in (self._run_button, self._preview_button, self._add_button):
                button.setEnabled(False)
            self._install_button.setVisible(False)
            return

        self._editor.set_specs(list(info.parameters))
        self._kind_label.setText(
            f"{MODEL_KIND_LABELS.get(info.kind, info.kind)}"
            + (f"  -  v{info.version}" if info.version else "")
        )
        self._method_label.setText(info.method or info.description)

        model = ModelRegistry.try_get(info.name)
        state = model.availability() if model else None
        available = bool(state and state.ok)

        for button in (self._run_button, self._preview_button, self._add_button):
            button.setEnabled(available)
        self._install_button.setVisible(not available)

        if available:
            if info.may_synthesise:
                self._status_label.setProperty("role", "warning")
                self._status_label.setText(
                    "Ready. This model may synthesise image content."
                )
            else:
                self._status_label.setProperty("role", "ok")
                self._status_label.setText(
                    "Ready. Deterministic - cannot invent image content."
                )
        else:
            self._status_label.setProperty("role", "error")
            self._status_label.setText(state.reason if state else "Unavailable.")
        from gui.theme import refresh_style

        refresh_style(self._status_label)


class RestorationPanel(QWidget):
    """Dock hosting the auto-enhance button and every task section.

    Signals:
        autoEnhanceRequested: The user pressed Auto Enhance.
        runRequested: ``(model_name, parameters)``
        previewRequested: ``(model_name, parameters)``
        addToPipelineRequested: ``(model_name, parameters)``
        installRequested: ``(model_name)``
        runPipelineRequested: The user asked to run the staged pipeline.
        editPipelineRequested: The user asked to open the pipeline editor.
    """

    autoEnhanceRequested = pyqtSignal()
    runRequested = pyqtSignal(str, dict)
    previewRequested = pyqtSignal(str, dict)
    addToPipelineRequested = pyqtSignal(str, dict)
    installRequested = pyqtSignal(str)
    runPipelineRequested = pyqtSignal()
    editPipelineRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._sections: List[TaskSection] = []
        self._pipeline = Pipeline(name="Staged pipeline")
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(10, 10, 10, 6)
        header_layout.setSpacing(6)

        self._auto_button = QPushButton("Auto Enhance  (E)")
        self._auto_button.setProperty("accent", True)
        self._auto_button.setMinimumHeight(30)
        self._auto_button.setToolTip(
            "Analyse the image, propose a pipeline, and show it for review "
            "before anything runs."
        )
        self._auto_button.clicked.connect(self.autoEnhanceRequested.emit)
        header_layout.addWidget(self._auto_button)
        header_layout.addWidget(HLine())
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(4)
        scroll.setWidget(container)

        for task in TASK_ORDER:
            section = TaskSection(task)
            section.runRequested.connect(self.runRequested.emit)
            section.previewRequested.connect(self.previewRequested.emit)
            section.addToPipelineRequested.connect(self._on_add_to_pipeline)
            section.installRequested.connect(self.installRequested.emit)
            self._sections.append(section)
            layout.addWidget(section)

        layout.addStretch(1)

        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 10)
        footer_layout.setSpacing(5)
        footer_layout.addWidget(HLine())
        footer_layout.addWidget(SectionLabel("Staged pipeline"))

        self._pipeline_label = QLabel("(empty)")
        self._pipeline_label.setWordWrap(True)
        self._pipeline_label.setProperty("role", "mono")
        footer_layout.addWidget(self._pipeline_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(5)
        self._edit_button = QPushButton("Edit pipeline...")
        self._edit_button.clicked.connect(self.editPipelineRequested.emit)
        buttons.addWidget(self._edit_button)

        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self.clear_pipeline)
        buttons.addWidget(self._clear_button)

        self._run_pipeline_button = QPushButton("RUN")
        self._run_pipeline_button.setProperty("accent", True)
        self._run_pipeline_button.setEnabled(False)
        self._run_pipeline_button.clicked.connect(self.runPipelineRequested.emit)
        buttons.addWidget(self._run_pipeline_button)
        footer_layout.addLayout(buttons)

        outer.addWidget(footer)

    # ------------------------------------------------------------------ public
    def refresh_models(self) -> None:
        """Reload every section's model list and availability."""
        for section in self._sections:
            section.refresh()
            section.setVisible(section.has_models)

    def focus_task(self, task: str) -> None:
        """Expand the section for ``task`` and collapse the others.

        Used when the investigator arrives from a detection result and should
        be looking at one specific operation.
        """
        for section in self._sections:
            match = section.task == task
            section.set_expanded(match)
            if match:
                section.setFocus()

    @property
    def pipeline(self) -> Pipeline:
        """The staged pipeline."""
        return self._pipeline

    def set_pipeline(self, pipeline: Pipeline) -> None:
        """Replace the staged pipeline."""
        self._pipeline = pipeline
        self._update_pipeline_label()

    def clear_pipeline(self) -> None:
        """Discard every staged step."""
        self._pipeline = Pipeline(name="Staged pipeline")
        self._update_pipeline_label()

    def set_busy(self, busy: bool) -> None:
        """Disable action buttons while an operation is running."""
        self._auto_button.setEnabled(not busy)
        self._run_pipeline_button.setEnabled(
            not busy and bool(self._pipeline.enabled_steps)
        )
        for section in self._sections:
            section.setEnabled(not busy)

    # --------------------------------------------------------------- handlers
    def _on_add_to_pipeline(self, model_name: str, parameters: dict) -> None:
        """Append a configured step to the staged pipeline."""
        if not model_name:
            return
        self._pipeline.add(
            PipelineStep(model_name=model_name, parameters=dict(parameters))
        )
        self._update_pipeline_label()
        self.addToPipelineRequested.emit(model_name, parameters)

    def _update_pipeline_label(self) -> None:
        """Refresh the staged-pipeline summary."""
        steps = self._pipeline.enabled_steps
        if not steps:
            self._pipeline_label.setText("(empty)")
            self._run_pipeline_button.setEnabled(False)
            return
        lines = [
            f"{index}. {step.describe()}" for index, step in enumerate(steps, start=1)
        ]
        if self._pipeline.may_synthesise:
            lines.append("")
            lines.append("Includes a generative step.")
        self._pipeline_label.setText("\n".join(lines))
        self._run_pipeline_button.setEnabled(True)
