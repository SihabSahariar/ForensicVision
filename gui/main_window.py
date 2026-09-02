"""ForensicVision main window.

Hosts the menus, toolbar, dock widgets and the central viewer/comparison stack,
and orchestrates the worker threads. All long-running work is delegated to
:mod:`workers`; this module only wires signals and updates widgets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QSettings, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from analysis.analyzer import AnalysisReport
from analysis.visualizations import (
    VISUALIZATION_LABELS,
    VISUALIZATION_NOTES,
    render_visualization,
)
from app.config import AppConfig, get_config, save_config
from app.constants import (
    IMAGE_FILE_FILTER,
    LOSSLESS_EXPORT_FORMATS,
    SETTINGS_GEOMETRY,
    SETTINGS_STATE,
    STATUS_MESSAGE_TIMEOUT_MS,
    DegradationKey,
    TaskType,
)
from app.version import APP_NAME, APP_VERSION, build_string
from core.case_manager import CaseManager
from core.device import get_device_report, refresh_device_report
from core.exceptions import ForensicVisionError
from core.image_io import ImageData, load_image, save_image
from database.models import Derivative, Evidence
from forensic.hashing import HashSet, hash_file
from forensic.metadata import extract_metadata
from forensic.safe_mode import SAFE_MODE_OFF_WARNING, get_guard
from restoration.auto_engine import AutoRestorationEngine
from restoration.pipeline import Pipeline, PipelineStep
from restoration.registry import ModelRegistry
from gui.analysis_panel import AnalysisPanel
from gui.case_explorer import CaseExplorer
from gui.comparison_viewer import ComparisonViewer
from gui.dialogs.new_case import NewCaseDialog
from gui.dialogs.pipeline_editor import PipelineEditorDialog
from gui.history_panel import HistoryPanel
from gui.inspector import InspectorPanel, InspectorTab
from gui.image_viewer import ImageViewer, PixelProbe
from gui.log_panel import LogPanel
from gui.metadata_panel import MetadataPanel
from gui.model_manager import ModelManagerDialog
from gui.restoration_panel import RestorationPanel
from gui.roi_tools import ROI, ROIType
from gui.theme import Palette, refresh_style
from gui.utils import (
    confirm_synthesis,
    open_with_default_application,
    reveal_in_file_manager,
    show_error,
)
from gui.widgets.common import BannerLabel
from workers.analysis_worker import AnalysisWorker
from workers.import_worker import ImportWorker
from workers.restoration_worker import RestorationWorker

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """The application's main window."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        super().__init__()
        self._config = config or get_config()
        self._guard = get_guard(self._config.safe_mode)
        self._settings = QSettings("ForensicVision", "ForensicVision")

        self._case: Optional[CaseManager] = None
        self._evidence: Optional[Evidence] = None
        self._derivative: Optional[Derivative] = None
        self._original_image: Optional[ImageData] = None
        self._current_image: Optional[ImageData] = None
        self._enhanced_image: Optional[ImageData] = None
        self._report: Optional[AnalysisReport] = None
        self._worker: Optional[Any] = None
        self._visualization_active: Optional[str] = None
        self._last_probe: Optional[PixelProbe] = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setMinimumSize(1180, 720)
        self.setDockOptions(
            QMainWindow.AnimatedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AllowTabbedDocks
        )

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._connect_signals()

        self._restore_layout()
        self._refresh_models()
        self._update_actions()
        self._start_device_timer()

        logger.info("%s ready", build_string())

    # ================================================================== build
    def _build_central(self) -> None:
        """Create the central viewer/comparison tab stack."""
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        viewer_page = QWidget()
        viewer_layout = QVBoxLayout(viewer_page)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)

        self._visualization_banner = BannerLabel("")
        self._visualization_banner.setVisible(False)
        viewer_layout.addWidget(self._visualization_banner)

        self.viewer = ImageViewer()
        viewer_layout.addWidget(self.viewer, 1)
        self._tabs.addTab(viewer_page, "Viewer")

        self.comparison = ComparisonViewer()
        self._tabs.addTab(self.comparison, "Compare")

        self.setCentralWidget(self._tabs)

    def _build_docks(self) -> None:
        """Create the single tabbed inspector dock.

        One dock rather than six. The panels are the same, but they no longer
        compete for screen space with each other or with the image.
        """
        self.inspector = InspectorPanel()

        # Keep the individual panels reachable as attributes: the rest of the
        # window, and the tests, address them directly.
        self.case_explorer = self.inspector.case_explorer
        self.metadata_panel = self.inspector.metadata_panel
        self.analysis_panel = self.inspector.analysis_panel
        self.restoration_panel = self.inspector.restoration_panel
        self.history_panel = self.inspector.history_panel
        self.log_panel = self.inspector.log_panel

        self._inspector_dock = self._make_dock(
            "Inspector", self.inspector, Qt.RightDockWidgetArea
        )
        self.resizeDocks([self._inspector_dock], [380], Qt.Horizontal)

    def _make_dock(
        self, title: str, widget: QWidget, area: Qt.DockWidgetArea
    ) -> QDockWidget:
        """Wrap ``widget`` in a dock and attach it."""
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{title.replace(' ', '_').lower()}")
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(area, dock)
        return dock

    # --------------------------------------------------------------- actions
    def _build_actions(self) -> None:
        """Create every QAction with its shortcut."""

        def action(
            text: str,
            slot,
            shortcut: str = "",
            tip: str = "",
            checkable: bool = False,
        ) -> QAction:
            item = QAction(text, self)
            if shortcut:
                item.setShortcut(QKeySequence(shortcut))
            if tip:
                item.setToolTip(tip)
                item.setStatusTip(tip)
            item.setCheckable(checkable)
            if checkable:
                item.toggled.connect(slot)
            else:
                item.triggered.connect(slot)
            return item

        def set_initial_state(item: QAction, checked: bool) -> None:
            """Set a checkable action's starting state without firing its slot.

            The slots reach widgets that later build steps have not created
            yet, and an exception raised inside a Qt slot aborts the process
            rather than propagating.
            """
            item.blockSignals(True)
            item.setChecked(checked)
            item.blockSignals(False)

        # -- File / case ------------------------------------------------------
        self.act_new_case = action(
            "&New Case...", self._on_new_case, "Ctrl+N", "Create a forensic case"
        )
        self.act_open_case = action(
            "&Open Case...", self._on_open_case, "Ctrl+Shift+O",
            "Open an existing case folder",
        )
        self.act_close_case = action("&Close Case", self._on_close_case)
        self.act_import = action(
            "&Import Evidence...", self._on_import, "Ctrl+O",
            "Copy, hash and register image evidence",
        )
        self.act_open_image = action(
            "Open Image (no case)...", self._on_open_image_standalone,
            "Ctrl+Shift+I", "Inspect an image without creating a case",
        )
        self.act_export = action(
            "&Export Derivative...", self._on_export, "Ctrl+S",
            "Write the enhanced image to a new file",
        )
        self.act_export_as = action(
            "Export Derivative &As...", self._on_export_as, "Ctrl+Shift+S"
        )
        self.act_quit = action("E&xit", self.close, "Ctrl+Q")

        # -- Analysis ---------------------------------------------------------
        self.act_analyse = action(
            "&Analyse Image", self._on_analyse, "A",
            "Run every degradation indicator",
        )
        self.act_analyse_roi = action(
            "Analyse &ROI", self._on_analyse_roi, "Shift+A",
            "Run the analysis on the selected region only",
        )
        self.act_verify = action(
            "&Verify Evidence Integrity", self._on_verify_evidence
        )

        # -- Restoration ------------------------------------------------------
        self.act_auto = action(
            "Auto &Enhance", self._on_auto_enhance, "E",
            "Analyse, propose a pipeline, and show it for review",
        )
        self.act_run_pipeline = action(
            "&Run Staged Pipeline", self._on_run_staged_pipeline, "Ctrl+R"
        )
        self.act_edit_pipeline = action(
            "Edit &Pipeline...", self._on_edit_pipeline
        )
        self.act_enhance_roi = action(
            "Enhance &ROI", self._on_enhance_roi, "Shift+E"
        )
        self.act_reset_derivative = action(
            "Return to &Original", self._on_show_original
        )

        # -- View -------------------------------------------------------------
        self.act_fit = action("&Fit to Window", self._on_fit, "F")
        self.act_reset_view = action("&Reset View", self._on_reset_view, "R")
        self.act_zoom_in = action("Zoom &In", self.viewer.zoom_in, "Ctrl++")
        self.act_zoom_out = action("Zoom &Out", self.viewer.zoom_out, "Ctrl+-")
        self.act_zoom_100 = action("100%", lambda: self._set_zoom(1.0), "1")
        self.act_zoom_200 = action("200%", lambda: self._set_zoom(2.0), "2")
        self.act_zoom_400 = action("400%", lambda: self._set_zoom(4.0), "4")
        self.act_zoom_800 = action("800%", lambda: self._set_zoom(8.0), "8")
        self.act_zoom_50 = action("50%", lambda: self._set_zoom(0.5))
        self.act_zoom_25 = action("25%", lambda: self._set_zoom(0.25))
        self.act_crosshair = action(
            "Show &Crosshair", self._on_toggle_crosshair, "C", checkable=True
        )

        # -- inspector ---------------------------------------------------- #
        self.act_toggle_inspector = action(
            "Show &Inspector", self._on_toggle_inspector, "F9",
            "Show or hide the inspector panel", checkable=True,
        )
        set_initial_state(self.act_toggle_inspector, True)

        self.act_focus_mode = action(
            "&Focus Mode", self._on_toggle_focus_mode, "F11",
            "Hide every panel and give the whole window to the image",
            checkable=True,
        )

        self.act_next_tab = action(
            "Next Inspector Tab", lambda: self.inspector.next_tab(1), "Ctrl+Tab"
        )
        self.act_prev_tab = action(
            "Previous Inspector Tab", lambda: self.inspector.next_tab(-1),
            "Ctrl+Shift+Tab",
        )

        # One shortcut per tab: Alt+1..Alt+5.
        self._inspector_actions: List[QAction] = []
        for position, (tab_id, label, tooltip) in enumerate(
            InspectorTab.ORDER, start=1
        ):
            item = action(
                f"&{position}  {label}",
                lambda _checked=False, key=tab_id: self._show_inspector_tab(key),
                f"Alt+{position}",
                tooltip,
            )
            self._inspector_actions.append(item)
        self.act_compare = action(
            "&Compare Original / Enhanced", self._on_compare, "Ctrl+D"
        )
        self.act_reset_layout = action("Reset &Layout", self._on_reset_layout)

        # -- ROI --------------------------------------------------------------
        self._roi_group = QActionGroup(self)
        self._roi_group.setExclusive(True)
        self.act_roi_none = action(
            "No Selection Tool", lambda checked: self._set_roi_mode(None),
            "Esc", checkable=True,
        )
        self.act_roi_rect = action(
            "&Rectangle ROI", lambda checked: self._set_roi_mode(ROIType.RECTANGLE),
            "Ctrl+1", checkable=True,
        )
        self.act_roi_ellipse = action(
            "&Ellipse ROI", lambda checked: self._set_roi_mode(ROIType.ELLIPSE),
            "Ctrl+2", checkable=True,
        )
        self.act_roi_polygon = action(
            "&Polygon ROI", lambda checked: self._set_roi_mode(ROIType.POLYGON),
            "Ctrl+3", checkable=True,
        )
        self.act_roi_freehand = action(
            "&Freehand ROI", lambda checked: self._set_roi_mode(ROIType.FREEHAND),
            "Ctrl+4", checkable=True,
        )
        for item in (
            self.act_roi_none, self.act_roi_rect, self.act_roi_ellipse,
            self.act_roi_polygon, self.act_roi_freehand,
        ):
            self._roi_group.addAction(item)
        set_initial_state(self.act_roi_none, True)

        self.act_roi_crop = action("&Crop to ROI", self._on_crop_roi)
        self.act_roi_export = action("&Export ROI...", self._on_export_roi)
        self.act_roi_clear = action("C&lear ROI", self._on_clear_roi)

        # -- Tools ------------------------------------------------------------
        self.act_model_manager = action(
            "&Model Manager...", self._on_model_manager, "Ctrl+M"
        )
        self.act_batch = action("&Batch Processing...", self._on_batch)
        self.act_safe_mode = action(
            "&Forensic Safe Mode", self._on_toggle_safe_mode, checkable=True
        )
        set_initial_state(self.act_safe_mode, self._guard.enabled)
        self.act_ocr = action("&OCR Region...", self._on_ocr)
        self.act_detect = action("&Detect Objects...", self._on_detect)
        self.act_report = action(
            "&Generate Report...", self._on_report, "Ctrl+P"
        )
        self.act_preferences = action("&Preferences...", self._on_preferences)

        # -- Help -------------------------------------------------------------
        self.act_about = action("&About ForensicVision", self._on_about)
        self.act_shortcuts = action("&Keyboard Shortcuts", self._on_shortcuts)
        self.act_limitations = action(
            "&Limitations and Disclaimer", self._on_limitations
        )

        # -- Undo / redo of the *view*, never of evidence ----------------------
        self.act_undo = action(
            "&Undo View Change", self._on_undo, "Ctrl+Z",
            "Step back through viewed images. Evidence is never modified.",
        )
        self.act_redo = action(
            "&Redo View Change", self._on_redo, "Ctrl+Shift+Z"
        )
        self._view_history: List[ImageData] = []
        self._view_position = -1

    def _build_menus(self) -> None:
        """Assemble the menu bar."""
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.act_new_case)
        file_menu.addAction(self.act_open_case)
        file_menu.addAction(self.act_close_case)
        file_menu.addSeparator()
        file_menu.addAction(self.act_import)
        file_menu.addAction(self.act_open_image)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export)
        file_menu.addAction(self.act_export_as)
        file_menu.addSeparator()
        self._recent_menu = file_menu.addMenu("Recent Cases")
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        case_menu = bar.addMenu("&Case")
        case_menu.addAction(self.act_new_case)
        case_menu.addAction(self.act_open_case)
        case_menu.addAction(self.act_close_case)
        case_menu.addSeparator()
        case_menu.addAction(self.act_report)

        evidence_menu = bar.addMenu("&Evidence")
        evidence_menu.addAction(self.act_import)
        evidence_menu.addAction(self.act_verify)
        evidence_menu.addSeparator()
        evidence_menu.addAction(self.act_roi_none)
        evidence_menu.addAction(self.act_roi_rect)
        evidence_menu.addAction(self.act_roi_ellipse)
        evidence_menu.addAction(self.act_roi_polygon)
        evidence_menu.addAction(self.act_roi_freehand)
        evidence_menu.addSeparator()
        evidence_menu.addAction(self.act_roi_crop)
        evidence_menu.addAction(self.act_roi_export)
        evidence_menu.addAction(self.act_roi_clear)

        analysis_menu = bar.addMenu("&Analysis")
        analysis_menu.addAction(self.act_analyse)
        analysis_menu.addAction(self.act_analyse_roi)
        analysis_menu.addSeparator()
        visual_menu = analysis_menu.addMenu("Forensic &Visualisations")
        for key, label in VISUALIZATION_LABELS.items():
            item = QAction(label, self)
            item.triggered.connect(
                lambda _checked=False, name=key: self._show_visualization(name)
            )
            visual_menu.addAction(item)
        visual_menu.addSeparator()
        clear_visual = QAction("Clear Visualisation", self)
        clear_visual.triggered.connect(self._clear_visualization)
        visual_menu.addAction(clear_visual)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.act_ocr)
        analysis_menu.addAction(self.act_detect)

        restoration_menu = bar.addMenu("&Restoration")
        restoration_menu.addAction(self.act_auto)
        restoration_menu.addAction(self.act_edit_pipeline)
        restoration_menu.addAction(self.act_run_pipeline)
        restoration_menu.addSeparator()
        restoration_menu.addAction(self.act_enhance_roi)
        restoration_menu.addAction(self.act_reset_derivative)

        view_menu = bar.addMenu("&View")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_reset_view)
        view_menu.addSeparator()
        for item in (
            self.act_zoom_25, self.act_zoom_50, self.act_zoom_100,
            self.act_zoom_200, self.act_zoom_400, self.act_zoom_800,
        ):
            view_menu.addAction(item)
        view_menu.addSeparator()
        view_menu.addAction(self.act_zoom_in)
        view_menu.addAction(self.act_zoom_out)
        view_menu.addSeparator()
        view_menu.addAction(self.act_crosshair)
        view_menu.addAction(self.act_compare)
        view_menu.addSeparator()
        view_menu.addAction(self.act_next_tab)
        view_menu.addAction(self.act_prev_tab)
        view_menu.addSeparator()
        panels_menu = view_menu.addMenu("&Panels")
        for action in self._inspector_actions:
            panels_menu.addAction(action)
        panels_menu.addSeparator()
        panels_menu.addAction(self.act_toggle_inspector)
        panels_menu.addAction(self.act_focus_mode)
        view_menu.addSeparator()
        view_menu.addAction(self.act_undo)
        view_menu.addAction(self.act_redo)
        view_menu.addSeparator()
        view_menu.addAction(self.act_reset_layout)

        tools_menu = bar.addMenu("&Tools")
        tools_menu.addAction(self.act_model_manager)
        tools_menu.addAction(self.act_batch)
        tools_menu.addSeparator()
        tools_menu.addAction(self.act_safe_mode)
        tools_menu.addSeparator()
        tools_menu.addAction(self.act_preferences)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction(self.act_shortcuts)
        help_menu.addAction(self.act_limitations)
        help_menu.addSeparator()
        help_menu.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        """Assemble the main toolbar."""
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)

        toolbar.addAction(self.act_import)
        toolbar.addAction(self.act_export)
        toolbar.addSeparator()
        toolbar.addAction(self.act_analyse)
        toolbar.addAction(self.act_auto)
        toolbar.addSeparator()
        toolbar.addAction(self.act_compare)
        toolbar.addAction(self.act_report)
        toolbar.addSeparator()
        toolbar.addAction(self.act_roi_rect)
        toolbar.addAction(self.act_roi_clear)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._cancel_worker)
        toolbar.addWidget(self._cancel_button)

    def _build_statusbar(self) -> None:
        """Assemble the status bar."""
        status = QStatusBar()
        self.setStatusBar(status)

        self._safe_mode_label = QLabel(self._guard.status_text())
        self._safe_mode_label.setProperty("role", "safemode")
        status.addPermanentWidget(self._safe_mode_label)

        self._device_label = QLabel(get_device_report().summary_line())
        status.addPermanentWidget(self._device_label)

        self._model_label = QLabel("Model: -")
        status.addPermanentWidget(self._model_label)

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setVisible(False)
        status.addPermanentWidget(self._progress)

        self._pixel_label = QLabel("")
        self._pixel_label.setProperty("role", "mono")
        self._pixel_label.setMinimumWidth(340)
        status.addWidget(self._pixel_label)

        self._update_safe_mode_display()

    def _connect_signals(self) -> None:
        """Wire panel signals to handlers."""
        self.viewer.pixelProbed.connect(self._on_pixel_probed)
        self.viewer.contextMenuRequested.connect(self._on_viewer_context_menu)
        for viewer in (
            self.comparison.left_viewer, self.comparison.right_viewer,
            self.comparison.difference_viewer,
        ):
            viewer.contextMenuRequested.connect(self._on_viewer_context_menu)
        self.viewer.cursorLeft.connect(lambda: self._pixel_label.setText(""))
        self.viewer.zoomChanged.connect(self._on_zoom_changed)
        self.viewer.roiCreated.connect(self._on_roi_created)
        self.viewer.roiCleared.connect(lambda: self._update_actions())
        self.comparison.pixelProbed.connect(self._on_pixel_probed)

        self.case_explorer.evidenceSelected.connect(self._on_evidence_selected)
        self.case_explorer.derivativeSelected.connect(self._on_derivative_selected)
        self.case_explorer.importRequested.connect(self._on_import)
        self.case_explorer.compareRequested.connect(self._on_compare_derivative)
        self.case_explorer.revealRequested.connect(reveal_in_file_manager)
        self.case_explorer.reportSelected.connect(open_with_default_application)

        self.analysis_panel.analyseRequested.connect(self._on_analyse)
        self.metadata_panel.verifyRequested.connect(self._on_verify_evidence)

        self.restoration_panel.autoEnhanceRequested.connect(self._on_auto_enhance)
        self.restoration_panel.runRequested.connect(self._on_run_single_model)
        self.restoration_panel.previewRequested.connect(self._on_preview_model)
        self.restoration_panel.installRequested.connect(self._on_install_model)
        self.restoration_panel.runPipelineRequested.connect(
            self._on_run_staged_pipeline
        )
        self.restoration_panel.editPipelineRequested.connect(self._on_edit_pipeline)

        self.history_panel.derivativeSelected.connect(self._on_derivative_selected)
        self.history_panel.compareRequested.connect(self._on_compare_derivative)
        self.history_panel.revealRequested.connect(reveal_in_file_manager)
        self.inspector.tabChanged.connect(self._on_inspector_tab_changed)

    def _start_device_timer(self) -> None:
        """Refresh the VRAM readout periodically."""
        self._device_timer = QTimer(self)
        self._device_timer.setInterval(4000)
        self._device_timer.timeout.connect(self._update_device_label)
        self._device_timer.start()

    # ================================================================ helpers
    def _update_device_label(self) -> None:
        """Refresh the device/VRAM status text."""
        try:
            self._device_label.setText(
                refresh_device_report().summary_line(self._config.cuda_index)
            )
        except Exception:  # pragma: no cover - defensive
            pass

    def _update_safe_mode_display(self) -> None:
        """Reflect the guard state in the status bar."""
        self._safe_mode_label.setText(self._guard.status_text())
        self._safe_mode_label.setProperty(
            "role", "safemode" if self._guard.enabled else "warning"
        )
        self._safe_mode_label.setToolTip(self._guard.describe())
        refresh_style(self._safe_mode_label)

    def _refresh_models(self) -> None:
        """Reload the restoration panel's model availability."""
        self.restoration_panel.refresh_models()
        available = len(ModelRegistry.available())
        total = len(ModelRegistry.infos())
        self._model_label.setText(f"Models: {available}/{total} ready")

    def _update_actions(self) -> None:
        """Enable/disable actions based on the current state."""
        has_case = self._case is not None
        has_image = self._current_image is not None
        has_evidence = self._evidence is not None
        has_enhanced = self._enhanced_image is not None
        has_roi = self.viewer.roi is not None
        busy = self._worker is not None

        for act in (self.act_close_case, self.act_import, self.act_batch):
            act.setEnabled(has_case and not busy)
        self.act_report.setEnabled(has_case and has_evidence and not busy)
        for act in (self.act_analyse, self.act_auto):
            act.setEnabled(has_image and not busy)
        self.act_analyse_roi.setEnabled(has_image and has_roi and not busy)
        self.act_enhance_roi.setEnabled(
            has_image and has_roi and has_evidence and not busy
        )
        for act in (self.act_export, self.act_export_as):
            act.setEnabled(has_enhanced or has_image)
        self.act_compare.setEnabled(has_enhanced)
        self.act_verify.setEnabled(has_evidence and not busy)
        self.act_reset_derivative.setEnabled(self._original_image is not None)
        for act in (self.act_roi_crop, self.act_roi_export, self.act_roi_clear):
            act.setEnabled(has_roi)
        self.act_run_pipeline.setEnabled(
            has_image and bool(self.restoration_panel.pipeline.enabled_steps) and not busy
        )
        self.act_undo.setEnabled(self._view_position > 0)
        self.act_redo.setEnabled(
            0 <= self._view_position < len(self._view_history) - 1
        )
        self.restoration_panel.set_busy(busy)
        self.analysis_panel.set_busy(busy)

    def _status(self, message: str, timeout: int = STATUS_MESSAGE_TIMEOUT_MS) -> None:
        """Show a transient status-bar message."""
        self.statusBar().showMessage(message, timeout)

    def _set_image(
        self,
        image: Optional[ImageData],
        record_history: bool = True,
        keep_view: bool = False,
    ) -> None:
        """Display ``image`` in the main viewer."""
        self._current_image = image
        self.viewer.set_image(image, keep_view=keep_view)
        if image is not None and record_history:
            self._push_history(image)
        self._update_actions()

    def _push_history(self, image: ImageData) -> None:
        """Record a viewed image for the view-only undo stack."""
        del self._view_history[self._view_position + 1:]
        self._view_history.append(image)
        if len(self._view_history) > 24:
            self._view_history.pop(0)
        self._view_position = len(self._view_history) - 1

    # ================================================================ workers
    def _run_worker(
        self,
        worker,
        on_finished,
        description: str = "",
        show_progress: bool = True,
    ) -> None:
        """Start ``worker`` and wire its signals to the status bar."""
        if self._worker is not None:
            QMessageBox.information(
                self, "Busy",
                "Another operation is already running. Wait for it to finish "
                "or cancel it first.",
            )
            return

        self._worker = worker
        self._progress.setVisible(show_progress)
        self._progress.setValue(0)
        self._cancel_button.setVisible(True)
        self._status(description or getattr(worker, "description", "Working..."), 0)
        self._update_actions()

        def cleanup() -> None:
            self._worker = None
            self._progress.setVisible(False)
            self._cancel_button.setVisible(False)
            self._update_actions()

        def handle_finished(result) -> None:
            cleanup()
            try:
                on_finished(result)
            except Exception as exc:
                logger.exception("Result handler failed")
                show_error(self, "Operation failed", str(exc))

        def handle_error(message: str, detail: str) -> None:
            cleanup()
            self._status("Operation failed", STATUS_MESSAGE_TIMEOUT_MS)
            show_error(self, "Operation failed", message, detail)

        def handle_cancelled() -> None:
            cleanup()
            self._status("Operation cancelled")

        worker.progress.connect(self._on_worker_progress)
        worker.status.connect(lambda text: self._status(text, 0))
        worker.finished_work.connect(handle_finished)
        worker.error.connect(handle_error)
        worker.cancelled_work.connect(handle_cancelled)
        worker.start()

    @pyqtSlot(int, str)
    def _on_worker_progress(self, percent: int, message: str) -> None:
        """Update the progress bar and status text."""
        self._progress.setValue(percent)
        if message:
            self._status(message, 0)

    def _cancel_worker(self) -> None:
        """Ask the running worker to stop."""
        if self._worker is not None:
            self._worker.cancel()
            self._status("Cancelling...")

    # ============================================================ case events
    def _on_new_case(self) -> None:
        """Create a new case."""
        dialog = NewCaseDialog(self._config, self)
        if dialog.exec_() != NewCaseDialog.Accepted:
            return
        try:
            self._guard.set_enabled(dialog.safe_mode)
            case = CaseManager.create(
                parent=dialog.cases_root,
                case_id=dialog.case_id,
                title=dialog.title,
                investigator=dialog.investigator,
                organisation=dialog.organisation,
                description=dialog.description,
                guard=self._guard,
            )
        except ForensicVisionError as exc:
            show_error(self, "Could not create case", str(exc))
            return
        self._adopt_case(case)
        self._status(f"Case {case.case_id} created")

    def _on_open_case(self) -> None:
        """Open an existing case folder."""
        directory = QFileDialog.getExistingDirectory(
            self, "Open case folder", str(self._config.cases_path)
        )
        if directory:
            self._open_case_path(Path(directory))

    def _open_case_path(self, path: Path) -> None:
        """Open the case at ``path``."""
        try:
            case = CaseManager.open(path, guard=self._guard)
        except ForensicVisionError as exc:
            show_error(self, "Could not open case", str(exc))
            return
        self._adopt_case(case)
        self._status(f"Case {case.case_id} opened")

    def _adopt_case(self, case: CaseManager) -> None:
        """Bind the window to ``case``."""
        if self._case is not None:
            self._case.close()
        self._case = case
        self._evidence = None
        self._derivative = None
        self._set_image(None, record_history=False)
        self._enhanced_image = None
        self._original_image = None
        self._report = None

        self.case_explorer.set_case(case)
        self.history_panel.set_case(case)
        self.analysis_panel.clear()
        self.metadata_panel.clear()
        self.comparison.set_images(None, None)

        self.setWindowTitle(
            f"{APP_NAME} {APP_VERSION}  -  {case.case_id}"
            + (f"  -  {case.case.title}" if case.case.title else "")
        )

        self._config.push_recent_case(str(case.root))
        save_config(self._config)
        self._rebuild_recent_menu()
        self._update_actions()

    def _on_close_case(self) -> None:
        """Close the open case."""
        if self._case is None:
            return
        self._case.close()
        self._case = None
        self._evidence = None
        self._derivative = None
        self._set_image(None, record_history=False)
        self._enhanced_image = None
        self.case_explorer.set_case(None)
        self.history_panel.set_case(None)
        self.analysis_panel.clear()
        self.metadata_panel.clear()
        self.comparison.set_images(None, None)
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self._update_actions()

    def _rebuild_recent_menu(self) -> None:
        """Refresh the recent-cases submenu."""
        self._recent_menu.clear()
        if not self._config.recent_cases:
            empty = QAction("(none)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for entry in self._config.recent_cases:
            item = QAction(entry, self)
            item.triggered.connect(
                lambda _checked=False, p=entry: self._open_case_path(Path(p))
            )
            self._recent_menu.addAction(item)

    # ======================================================== evidence events
    def _on_import(self) -> None:
        """Import one or more evidence files."""
        if self._case is None:
            QMessageBox.information(
                self, "No case open",
                "Create or open a case before importing evidence.\n\n"
                "Evidence is copied into the case, hashed and registered so "
                "the chain of custody is recorded.",
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import evidence", "", IMAGE_FILE_FILTER
        )
        if not paths:
            return

        worker = ImportWorker(self._case, [Path(p) for p in paths])
        self._run_worker(worker, self._on_import_finished, "Importing evidence...")

    def _on_import_finished(self, results: List) -> None:
        """Refresh after an import and select the first new item."""
        self.case_explorer.refresh()
        duplicates = [r for r in results if r.is_duplicate]
        if duplicates:
            names = "\n".join(f"  - {r.evidence.original_filename}" for r in duplicates)
            QMessageBox.information(
                self, "Duplicate content detected",
                f"{len(duplicates)} file(s) had content identical to evidence "
                f"already in this case and were not stored twice:\n\n{names}",
            )
        if results:
            self._load_evidence(results[0].evidence)
        self._status(f"Imported {len(results)} file(s)")

    def _on_open_image_standalone(self) -> None:
        """Open an image for inspection without a case."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", IMAGE_FILE_FILTER
        )
        if path:
            self._load_standalone_path(Path(path), notify=True)

    def _load_standalone_path(self, path: Path, notify: bool = False) -> None:
        """Load ``path`` for inspection without registering it as evidence.

        Args:
            path: Image to open.
            notify: Show the explanatory dialog. Suppressed on startup, where
                the user chose this mode deliberately on the command line.
        """
        try:
            image = load_image(Path(path))
        except ForensicVisionError as exc:
            show_error(self, "Could not open image", str(exc))
            return

        self._evidence = None
        self._derivative = None
        self._original_image = image
        self._enhanced_image = None
        self._report = None
        self._set_image(image)
        self.analysis_panel.clear()
        self.comparison.set_images(image, None)

        metadata = extract_metadata(Path(path))
        hashes = hash_file(Path(path))
        self.metadata_panel.set_metadata(metadata, hashes, image, Path(path).name)

        if notify:
            QMessageBox.information(
                self, "Opened outside a case",
                "This image was opened for inspection only. It has not been "
                "copied into a case, and derivatives cannot be recorded with "
                "provenance until you create a case and import it.",
            )
        self._status(f"Opened {Path(path).name} (no case)")
        self._update_actions()

    def _on_evidence_selected(self, evidence_id: int) -> None:
        """Load the evidence chosen in the explorer."""
        if self._case is None:
            return
        evidence = self._case.repository.get_evidence(evidence_id)
        if evidence is not None:
            self._load_evidence(evidence)

    def _load_evidence(self, evidence: Evidence) -> None:
        """Display an evidence item and its metadata."""
        if self._case is None:
            return
        try:
            image = self._case.load_evidence_image(evidence)
        except ForensicVisionError as exc:
            show_error(self, "Could not load evidence", str(exc))
            return

        self._evidence = evidence
        self._derivative = None
        self._original_image = image
        self._enhanced_image = None
        self._report = None
        self._view_history.clear()
        self._view_position = -1
        self._clear_visualization()

        self._set_image(image)
        self.comparison.set_images(image, None)
        self.history_panel.set_evidence(evidence)
        self.analysis_panel.clear()

        metadata = extract_metadata(Path(evidence.stored_path))
        hashes = HashSet(
            sha256=evidence.sha256, sha512=evidence.sha512,
            md5=evidence.md5, size_bytes=evidence.size_bytes,
        )
        self.metadata_panel.set_metadata(
            metadata, hashes, image, evidence.original_filename
        )

        stored = self._case.repository.latest_analysis(evidence_id=evidence.id)
        if stored is not None:
            try:
                self._report = AnalysisReport.from_dict(stored.details)
                self.analysis_panel.set_report(self._report)
            except Exception:
                logger.debug("Stored analysis could not be restored", exc_info=True)

        self._status(f"{evidence.original_filename}  -  {evidence.dimensions}")
        self._update_actions()

    def _on_derivative_selected(self, derivative_id: int) -> None:
        """Display a derivative alongside its original."""
        if self._case is None:
            return
        derivative = self._case.repository.get_derivative(derivative_id)
        if derivative is None:
            return
        try:
            image = self._case.load_derivative_image(derivative)
        except ForensicVisionError as exc:
            show_error(self, "Could not load derivative", str(exc))
            return

        if self._evidence is None or self._evidence.id != derivative.evidence_id:
            evidence = self._case.repository.get_evidence(derivative.evidence_id)
            if evidence is not None:
                self._load_evidence(evidence)

        self._derivative = derivative
        self._enhanced_image = image
        self._set_image(image)
        self.comparison.set_images(self._original_image, image)

        metadata = extract_metadata(Path(derivative.path))
        hashes = HashSet(
            sha256=derivative.sha256, sha512=derivative.sha512,
            md5=derivative.md5, size_bytes=derivative.size_bytes,
        )
        self.metadata_panel.set_metadata(
            metadata, hashes, image, Path(derivative.path).name
        )
        self._status(f"Derivative: {Path(derivative.path).name}")
        self._update_actions()

    def _on_verify_evidence(self) -> None:
        """Re-hash the current evidence and report the outcome."""
        if self._case is None or self._evidence is None:
            return
        ok = self._case.verify_evidence(self._evidence)
        if ok:
            QMessageBox.information(
                self, "Integrity verified",
                f"{self._evidence.original_filename}\n\n"
                f"The stored file still matches its recorded SHA-256:\n"
                f"{self._evidence.sha256}",
            )
        else:
            QMessageBox.critical(
                self, "INTEGRITY CHECK FAILED",
                f"{self._evidence.original_filename} no longer matches its "
                f"recorded SHA-256.\n\nExpected:\n{self._evidence.sha256}\n\n"
                "The stored evidence has changed since import. Do not rely on "
                "any derivative produced from it.",
            )

    # ======================================================== analysis events
    def _on_analyse(self) -> None:
        """Analyse the currently displayed image."""
        if self._current_image is None:
            return
        source = (
            Path(self._evidence.stored_path) if self._evidence is not None else None
        )
        worker = AnalysisWorker(self._current_image, source_path=source)
        self._run_worker(worker, self._on_analysis_finished, "Analysing image...")

    def _on_analyse_roi(self) -> None:
        """Analyse the selected region only."""
        roi = self.viewer.roi
        if roi is None or self._current_image is None:
            return
        crop = roi.crop(self._current_image.pixels)
        if crop.size == 0:
            return
        worker = AnalysisWorker(
            ImageData(pixels=crop), roi=roi.to_dict()
        )
        self._run_worker(
            worker, self._on_analysis_finished, f"Analysing {roi.describe()}..."
        )

    def _on_analysis_finished(self, report: AnalysisReport) -> None:
        """Store and display an analysis result."""
        self._report = report
        self.analysis_panel.set_report(report)

        if self._case is not None and self._evidence is not None:
            self._case.repository.add_analysis(
                case_pk=self._case.case_pk,
                evidence_id=self._evidence.id,
                derivative_id=self._derivative.id if self._derivative else None,
                scores=report.scores(),
                details=report.to_dict(),
                analyzer_version=report.analyzer_version,
                roi=report.roi,
            )
            self._case.audit(
                action="analysis.run",
                target=self._evidence.original_filename,
                detail=report.summary_line(),
            )
            self.case_explorer.refresh()

        self.inspector.show_tab(InspectorTab.ANALYSIS)
        self._status(report.summary_line())

    # ===================================================== restoration events
    def _on_auto_enhance(self) -> None:
        """Analyse if needed, then propose a pipeline for review."""
        if self._current_image is None:
            return
        if self._report is None:
            source = (
                Path(self._evidence.stored_path) if self._evidence is not None else None
            )
            worker = AnalysisWorker(self._current_image, source_path=source)
            self._run_worker(
                worker,
                lambda report: (
                    self._on_analysis_finished(report),
                    self._propose_pipeline(report),
                ),
                "Analysing before auto enhance...",
            )
            return
        self._propose_pipeline(self._report)

    def _propose_pipeline(self, report: AnalysisReport) -> None:
        """Build a recommendation and show the review dialog."""
        engine = AutoRestorationEngine()
        recommendation = engine.recommend(report)

        if recommendation.is_empty and not recommendation.skipped:
            QMessageBox.information(
                self, "No restoration indicated",
                "No degradation indicator exceeded the action threshold.\n\n"
                "Enhancement that is not indicated by the measurements adds "
                "risk without adding information. You can still stage "
                "operations manually from the Restoration panel.",
            )
            return

        dialog = PipelineEditorDialog(recommendation.pipeline, recommendation, self)
        if dialog.exec_() != PipelineEditorDialog.Accepted:
            self._status("Pipeline review cancelled")
            return

        self.restoration_panel.set_pipeline(dialog.pipeline)
        self._execute_pipeline(dialog.pipeline)

    def _on_edit_pipeline(self) -> None:
        """Open the pipeline editor on the staged pipeline."""
        dialog = PipelineEditorDialog(self.restoration_panel.pipeline, None, self)
        if dialog.exec_() == PipelineEditorDialog.Accepted:
            self.restoration_panel.set_pipeline(dialog.pipeline)
            self._execute_pipeline(dialog.pipeline)
        else:
            self.restoration_panel.set_pipeline(dialog.pipeline)

    def _on_run_staged_pipeline(self) -> None:
        """Run the pipeline staged in the restoration panel."""
        pipeline = self.restoration_panel.pipeline
        if not pipeline.enabled_steps:
            QMessageBox.information(
                self, "Nothing staged",
                "Add at least one operation to the pipeline first.",
            )
            return
        self._execute_pipeline(pipeline)

    def _on_run_single_model(self, model_name: str, parameters: dict) -> None:
        """Run one model as a single-step pipeline."""
        if not model_name:
            return
        pipeline = Pipeline(name=f"Single step: {model_name}")
        pipeline.add(PipelineStep(model_name=model_name, parameters=dict(parameters)))
        self._execute_pipeline(pipeline)

    def _on_preview_model(self, model_name: str, parameters: dict) -> None:
        """Run one model on the ROI (or a centre crop) without persisting."""
        if self._current_image is None or not model_name:
            return

        roi = self.viewer.roi
        if roi is not None and roi.is_valid():
            crop = roi.crop(self._current_image.pixels)
        else:
            height, width = self._current_image.pixels.shape[:2]
            size = min(512, height, width)
            top = (height - size) // 2
            left = (width - size) // 2
            crop = self._current_image.pixels[top:top + size, left:left + size].copy()

        pipeline = Pipeline(name=f"Preview: {model_name}")
        pipeline.add(PipelineStep(model_name=model_name, parameters=dict(parameters)))

        if not self._confirm_pipeline_synthesis(pipeline):
            return

        worker = RestorationWorker(
            image=ImageData(pixels=crop),
            pipeline=pipeline,
            persist=False,
            device=self._config.device,
            fp16=self._config.use_fp16,
        )
        self._run_worker(
            worker, self._on_preview_finished, f"Previewing {model_name}..."
        )

    def _on_preview_finished(self, outcome) -> None:
        """Show a preview result in the comparison tab."""
        self._enhanced_image = None
        self.comparison.set_images(self._preview_source(), outcome.image)
        self._tabs.setCurrentWidget(self.comparison)
        self.comparison.set_mode(ComparisonViewer.MODE_SIDE_BY_SIDE)
        self._status(
            "Preview only - no derivative was written and nothing was recorded."
        )

    def _preview_source(self) -> Optional[ImageData]:
        """Return the crop the preview was run on, for side-by-side display.

        Mirrors the crop selection in :meth:`_on_preview_model` so the pair
        shown to the investigator is genuinely before/after of the same region.
        """
        if self._current_image is None:
            return None
        roi = self.viewer.roi
        if roi is not None and roi.is_valid():
            return ImageData(pixels=roi.crop(self._current_image.pixels))
        height, width = self._current_image.pixels.shape[:2]
        size = min(512, height, width)
        top = (height - size) // 2
        left = (width - size) // 2
        return ImageData(
            pixels=self._current_image.pixels[top:top + size, left:left + size].copy()
        )

    def _confirm_pipeline_synthesis(self, pipeline: Pipeline) -> bool:
        """Ask for confirmation when a pipeline can synthesise content.

        Face restoration gets its own, blunter warning: it does not sharpen a
        face, it replaces one, and the failure mode is a convincing face that
        belongs to nobody.
        """
        if not pipeline.may_synthesise or not self._config.confirm_synthesis:
            return True

        generative = [
            step.display_name for step in pipeline.enabled_steps if step.may_synthesise
        ]
        face_steps = [
            step for step in pipeline.enabled_steps
            if step.task == TaskType.FACE_RESTORATION.value
        ]

        if face_steps:
            answer = QMessageBox.warning(
                self,
                "Face restoration synthesises facial detail",
                "<b>This operation replaces the face, it does not sharpen "
                "it.</b><br><br>"
                "The output is a face reconstructed from a learned prior over "
                "human faces, conditioned on the degraded input. It routinely "
                "invents features that are absent from the source - including "
                "eyewear, facial hair, apparent age and face shape - and it "
                "does so convincingly.<br><br>"
                "It must never be used for identification. Any report "
                "including the result must state that the facial detail is "
                "synthesised.<br><br>"
                "Proceed?",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return False
            if len(generative) == len(face_steps):
                return True

        return confirm_synthesis(
            self, ", ".join(generative),
            "Compare the result against the original and against a classical "
            "baseline before relying on any recovered detail.",
        )

    def _execute_pipeline(
        self, pipeline: Pipeline, roi_mask: Optional[np.ndarray] = None
    ) -> None:
        """Validate, confirm and run ``pipeline`` against the current image."""
        if self._current_image is None:
            return

        issues = pipeline.validate()
        blocking = [
            issue for issue in issues
            if "not registered" in issue or "cannot run" in issue
            or "no enabled steps" in issue.lower()
        ]
        if blocking:
            QMessageBox.warning(
                self, "Pipeline cannot run",
                "\n".join(f"- {issue}" for issue in blocking),
            )
            return

        if not self._confirm_pipeline_synthesis(pipeline):
            return

        roi = self.viewer.roi
        worker = RestorationWorker(
            image=self._current_image,
            pipeline=pipeline,
            case=self._case,
            evidence=self._evidence,
            parent_derivative=self._derivative,
            roi_mask=roi_mask,
            roi_descriptor=roi.to_dict() if (roi_mask is not None and roi) else None,
            device=self._config.device,
            fp16=self._config.use_fp16,
            persist=self._case is not None and self._evidence is not None,
        )
        self._run_worker(
            worker, self._on_restoration_finished,
            f"Running {len(pipeline.enabled_steps)} step(s)...",
        )

    def _on_restoration_finished(self, outcome) -> None:
        """Display and register a completed restoration."""
        self._enhanced_image = outcome.image
        self._derivative = outcome.derivative_row
        self._set_image(outcome.image)
        self.comparison.set_images(self._original_image, outcome.image)
        self._tabs.setCurrentWidget(self.comparison)

        if outcome.output_path is not None:
            self.case_explorer.refresh()
            self.history_panel.refresh()
            metadata = extract_metadata(outcome.output_path)
            hashes = hash_file(outcome.output_path)
            self.metadata_panel.set_metadata(
                metadata, hashes, outcome.image, outcome.output_path.name
            )
            self._status(
                f"Derivative written: {outcome.output_path.name}  "
                f"(sha256 {hashes.short()})"
            )
        else:
            self._status("Restoration complete - not persisted (no case open)")

        self._report = None
        self.analysis_panel.clear()
        self._update_actions()

    def _on_enhance_roi(self) -> None:
        """Run the staged pipeline restricted to the ROI."""
        roi = self.viewer.roi
        if roi is None or self._current_image is None:
            return
        pipeline = self.restoration_panel.pipeline
        if not pipeline.enabled_steps:
            QMessageBox.information(
                self, "Nothing staged",
                "Stage at least one operation before enhancing a region.",
            )
            return
        if pipeline.total_scale != 1:
            QMessageBox.warning(
                self, "Region enhancement unavailable",
                "The staged pipeline changes the image geometry, which cannot "
                "be applied to a region only.\n\nRemove the super-resolution "
                "step, or crop to the ROI first and enhance the crop.",
            )
            return
        mask = roi.to_mask(self._current_image.width, self._current_image.height)
        self._execute_pipeline(pipeline, roi_mask=mask)

    def _on_show_original(self) -> None:
        """Return the viewer to the original evidence."""
        if self._original_image is None:
            return
        self._derivative = None
        self._enhanced_image = None
        self._clear_visualization()
        self._set_image(self._original_image)
        self._status("Showing original evidence")

    def _on_install_model(self, model_name: str) -> None:
        """Open the Model Manager focused on ``model_name``."""
        dialog = ModelManagerDialog(self)
        dialog.modelsChanged.connect(self._refresh_models)
        dialog._select_by_name(model_name)  # noqa: SLF001 - intentional focus
        dialog.exec_()
        self._refresh_models()

    # ============================================================= view events
    def _set_zoom(self, factor: float) -> None:
        """Apply a preset zoom to the active viewer."""
        self._active_viewer().set_zoom(factor)

    def _active_viewer(self) -> ImageViewer:
        """Return the viewer the user is currently looking at."""
        if self._tabs.currentWidget() is self.comparison:
            if self.comparison.mode == ComparisonViewer.MODE_DIFFERENCE:
                return self.comparison.difference_viewer
            return self.comparison.left_viewer
        return self.viewer

    def _on_fit(self) -> None:
        self._active_viewer().fit_to_window()

    def _on_reset_view(self) -> None:
        self._active_viewer().reset_view()

    def _on_toggle_crosshair(self, enabled: bool) -> None:
        for viewer in (
            self.viewer, self.comparison.left_viewer,
            self.comparison.right_viewer, self.comparison.difference_viewer,
        ):
            viewer.set_crosshair_enabled(enabled)

    # ------------------------------------------------------------- inspector
    def _show_inspector_tab(self, tab_id: str) -> None:
        """Reveal the inspector and bring ``tab_id`` to the front."""
        if not self._inspector_dock.isVisible():
            self._inspector_dock.show()
            self.act_toggle_inspector.setChecked(True)
        self.inspector.show_tab(tab_id)

    def _on_toggle_inspector(self, visible: bool) -> None:
        """Show or hide the inspector dock."""
        self._inspector_dock.setVisible(visible)
        if visible and self.act_focus_mode.isChecked():
            self.act_focus_mode.setChecked(False)

    def _on_inspector_tab_changed(self, tab_id: str) -> None:
        """Refresh a tab's contents when it becomes visible.

        The panels are no longer all on screen at once, so a tab that was
        hidden while work completed needs a refresh when the user returns to
        it. Refreshing on demand also avoids rebuilding trees nobody is
        looking at.
        """
        if tab_id == InspectorTab.HISTORY and self._case is not None:
            self.history_panel.refresh()
        elif tab_id == InspectorTab.CASE and self._case is not None:
            self.case_explorer.refresh()

    def _on_toggle_focus_mode(self, enabled: bool) -> None:
        """Give the whole window to the image, or restore the panels."""
        self._inspector_dock.setVisible(not enabled)
        self.act_toggle_inspector.blockSignals(True)
        self.act_toggle_inspector.setChecked(not enabled)
        self.act_toggle_inspector.blockSignals(False)
        self._status(
            "Focus mode on - press F11 to bring the inspector back"
            if enabled else "Focus mode off"
        )

    # --------------------------------------------------------- context menus
    def _on_viewer_context_menu(self, position) -> None:
        """Build and show the image viewer's context menu.

        Most actions live here rather than in a permanently visible panel, so
        the panels can stay closed and the image can have the space.
        """
        menu = QMenu(self)
        has_image = self._current_image is not None
        has_roi = self.viewer.roi is not None

        analyse = menu.addMenu("Analyse")
        analyse.addAction(self.act_analyse)
        analyse.addAction(self.act_analyse_roi)
        analyse.setEnabled(has_image)

        visuals = menu.addMenu("Forensic visualisation")
        for key, label in VISUALIZATION_LABELS.items():
            item = visuals.addAction(label)
            item.triggered.connect(
                lambda _checked=False, name=key: self._show_visualization(name)
            )
        visuals.addSeparator()
        visuals.addAction("Clear visualisation", self._clear_visualization)
        visuals.setEnabled(has_image)

        menu.addSeparator()
        menu.addAction(self.act_auto)
        menu.addAction(self.act_run_pipeline)
        menu.addAction(self.act_enhance_roi)
        menu.addAction(self.act_edit_pipeline)

        menu.addSeparator()
        selection = menu.addMenu("Selection tool")
        for item in (
            self.act_roi_none, self.act_roi_rect, self.act_roi_ellipse,
            self.act_roi_polygon, self.act_roi_freehand,
        ):
            selection.addAction(item)

        region = menu.addMenu("Region")
        region.addAction(self.act_roi_crop)
        region.addAction(self.act_roi_export)
        region.addAction(self.act_roi_clear)
        region.setEnabled(has_roi)
        if has_roi:
            zoom_to = region.addAction("Zoom to region")
            zoom_to.triggered.connect(
                lambda: self.viewer.zoom_to_roi(self.viewer.roi)
            )

        menu.addSeparator()
        view = menu.addMenu("View")
        view.addAction(self.act_fit)
        view.addAction(self.act_reset_view)
        view.addSeparator()
        for item in (
            self.act_zoom_25, self.act_zoom_50, self.act_zoom_100,
            self.act_zoom_200, self.act_zoom_400, self.act_zoom_800,
        ):
            view.addAction(item)
        view.addSeparator()
        view.addAction(self.act_crosshair)

        menu.addAction(self.act_compare)
        menu.addAction(self.act_reset_derivative)

        menu.addSeparator()
        if self._last_probe is not None:
            probe = self._last_probe
            copy_menu = menu.addMenu(f"Copy pixel ({probe.x}, {probe.y})")
            copy_menu.addAction(
                f"Coordinates  {probe.x}, {probe.y}",
                lambda: self._copy_text(f"{probe.x}, {probe.y}"),
            )
            rgb = ", ".join(str(v) for v in probe.rgb)
            copy_menu.addAction(
                f"RGB  {rgb}", lambda: self._copy_text(rgb)
            )
            copy_menu.addAction(
                f"Hex  {self._probe_hex(probe)}",
                lambda: self._copy_text(self._probe_hex(probe)),
            )

        menu.addAction(self.act_detect)
        menu.addAction(self.act_ocr)
        menu.addSeparator()
        menu.addAction(self.act_export)
        menu.addSeparator()
        menu.addAction(self.act_toggle_inspector)
        menu.addAction(self.act_focus_mode)

        menu.exec_(position)

    @staticmethod
    def _probe_hex(probe: PixelProbe) -> str:
        """Return a probe's colour as a hex triplet."""
        return "#{:02X}{:02X}{:02X}".format(*probe.rgb)

    def _copy_text(self, text: str) -> None:
        """Put ``text`` on the clipboard and confirm in the status bar."""
        QApplication.clipboard().setText(text)
        self._status(f"Copied: {text}")

    def _on_compare(self) -> None:
        """Switch to the comparison tab."""
        if self._enhanced_image is None:
            QMessageBox.information(
                self, "Nothing to compare",
                "Run a restoration first, or select a derivative in the case "
                "explorer.",
            )
            return
        self.comparison.set_images(self._original_image, self._enhanced_image)
        self._tabs.setCurrentWidget(self.comparison)

    def _on_compare_derivative(self, derivative_id: int) -> None:
        """Load a derivative and switch to comparison."""
        self._on_derivative_selected(derivative_id)
        self._tabs.setCurrentWidget(self.comparison)

    def _on_zoom_changed(self, factor: float) -> None:
        """Reflect the zoom level in the status bar."""
        self._status(f"Zoom {factor * 100:.0f}%", 1500)

    def _on_pixel_probed(self, probe: PixelProbe) -> None:
        """Render the pixel readout."""
        self._last_probe = probe
        components = probe.components
        rgb = f"{components[0]}, {components[1]}, {components[2]}"
        alpha = f"  A: {probe.alpha}" if probe.alpha is not None else ""
        self._pixel_label.setText(
            f"X: {probe.x}   Y: {probe.y}   RGB: {rgb}{alpha}   "
            f"Gray: {probe.gray}   HSV: {probe.hsv[0]}, {probe.hsv[1]}, {probe.hsv[2]}"
        )

    def _on_undo(self) -> None:
        """Step back through viewed images."""
        if self._view_position > 0:
            self._view_position -= 1
            self.viewer.set_image(self._view_history[self._view_position])
            self._current_image = self._view_history[self._view_position]
            self._update_actions()

    def _on_redo(self) -> None:
        """Step forward through viewed images."""
        if 0 <= self._view_position < len(self._view_history) - 1:
            self._view_position += 1
            self.viewer.set_image(self._view_history[self._view_position])
            self._current_image = self._view_history[self._view_position]
            self._update_actions()

    # ============================================================== ROI events
    def _set_roi_mode(self, mode: Optional[ROIType]) -> None:
        """Switch the viewer's ROI drawing tool."""
        self.viewer.set_roi_mode(mode)
        if mode is None:
            self._status("Selection tool off")
        else:
            self._status(f"{mode.value.title()} selection: drag on the image")

    def _on_roi_created(self, roi: ROI) -> None:
        """React to a completed ROI."""
        self._status(roi.describe())
        self._update_actions()

    def _on_clear_roi(self) -> None:
        """Remove the active ROI."""
        self.viewer.clear_roi()
        self._update_actions()

    def _on_crop_roi(self) -> None:
        """Show the ROI crop in the viewer (evidence is untouched)."""
        roi = self.viewer.roi
        if roi is None or self._current_image is None:
            return
        crop = roi.crop(self._current_image.pixels)
        if crop.size == 0:
            return
        self._set_image(self._current_image.with_pixels(crop))
        self.viewer.clear_roi(emit=False)
        self._status(
            f"Cropped view to {crop.shape[1]} x {crop.shape[0]} - "
            "the stored evidence is unchanged"
        )

    def _on_export_roi(self) -> None:
        """Write the ROI crop to a new file."""
        roi = self.viewer.roi
        if roi is None or self._current_image is None:
            return
        crop = roi.crop(self._current_image.pixels)
        if crop.size == 0:
            return
        default = "roi_export.png"
        if self._evidence is not None:
            default = f"{Path(self._evidence.stored_path).stem}_roi.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export region of interest", default,
            "PNG (*.png);;TIFF (*.tif);;All files (*)",
        )
        if not path:
            return
        try:
            save_image(self._current_image.with_pixels(crop), Path(path), overwrite=True)
        except ForensicVisionError as exc:
            show_error(self, "Export failed", str(exc))
            return
        if self._case is not None:
            self._case.audit(
                action="roi.export", target=Path(path).name, detail=roi.describe()
            )
        self._status(f"Region exported to {Path(path).name}")

    # ==================================================== visualisation events
    def _show_visualization(self, name: str) -> None:
        """Render a forensic visualisation into the main viewer."""
        if self._current_image is None:
            return
        try:
            rendered = render_visualization(name, self._current_image.pixels)
        except Exception as exc:
            logger.exception("Visualisation %s failed", name)
            show_error(self, "Visualisation failed", str(exc))
            return

        self._visualization_active = name
        label = VISUALIZATION_LABELS.get(name, name)
        note = VISUALIZATION_NOTES.get(name, "")
        self._visualization_banner.setText(
            f"ANALYTICAL VIEW - {label}. This is a visualisation of the "
            f"evidence, not an enhanced image, and must not be exported as a "
            f"derivative.{(' ' + note) if note else ''}"
        )
        self._visualization_banner.setVisible(True)
        self.viewer.set_image(ImageData(pixels=rendered))
        self._tabs.setCurrentIndex(0)
        self._status(f"Visualisation: {label}")

    def _clear_visualization(self) -> None:
        """Leave visualisation mode and restore the working image."""
        self._visualization_banner.setVisible(False)
        if self._visualization_active is None:
            return
        self._visualization_active = None
        restore = self._enhanced_image or self._original_image
        if restore is not None:
            self.viewer.set_image(restore)
            self._current_image = restore

    # ============================================================ tools events
    def _on_model_manager(self) -> None:
        """Open the Model Manager."""
        dialog = ModelManagerDialog(self)
        dialog.modelsChanged.connect(self._refresh_models)
        dialog.exec_()
        self._refresh_models()

    def _on_batch(self) -> None:
        """Open the batch processing dialog."""
        if self._case is None:
            QMessageBox.information(
                self, "No case open", "Open a case before batch processing."
            )
            return
        from gui.dialogs.batch_dialog import BatchDialog

        dialog = BatchDialog(self._case, self._config, self)
        dialog.exec_()
        self.case_explorer.refresh()

    def _on_report(self) -> None:
        """Open the report dialog."""
        if self._case is None or self._evidence is None:
            QMessageBox.information(
                self, "Nothing to report",
                "Open a case and select an evidence item first.",
            )
            return
        from gui.dialogs.report_dialog import ReportDialog

        dialog = ReportDialog(
            case=self._case,
            evidence=self._evidence,
            derivative=self._derivative,
            report=self._report,
            original=self._original_image,
            enhanced=self._enhanced_image,
            difference_statistics=self.comparison.difference_statistics(),
            parent=self,
        )
        dialog.exec_()
        self.case_explorer.refresh()

    def _on_ocr(self) -> None:
        """Open the OCR dialog for the ROI or full frame."""
        if self._current_image is None:
            return
        from gui.dialogs.ocr_dialog import OcrDialog

        roi = self.viewer.roi
        original = self._original_image
        enhanced = self._enhanced_image

        if roi is not None and roi.is_valid():
            if original is not None:
                original = ImageData(pixels=roi.crop(original.pixels))
            # The ROI is expressed in the *original's* coordinates. Cropping the
            # enhanced image with it is only valid when the geometry matches;
            # after super-resolution it does not, so the full frame is passed
            # instead of a silently misaligned crop.
            if enhanced is not None:
                same_geometry = (
                    self._original_image is not None
                    and enhanced.shape[:2] == self._original_image.shape[:2]
                )
                if same_geometry:
                    enhanced = ImageData(pixels=roi.crop(enhanced.pixels))

        OcrDialog(original, enhanced, self).exec_()

    def _on_detect(self) -> None:
        """Open the object-detection dialog."""
        if self._current_image is None:
            return
        from gui.dialogs.detect_dialog import DetectDialog

        dialog = DetectDialog(self._current_image, self)
        dialog.roiChosen.connect(self._on_detection_roi)
        dialog.faceRestoreRequested.connect(self._on_face_restore_requested)
        dialog.exec_()

    def _on_detection_roi(self, roi: ROI) -> None:
        """Apply an ROI produced by detection."""
        self.viewer.set_roi(roi)
        self.viewer.zoom_to_roi(roi)
        self._tabs.setCurrentIndex(0)
        self._status(f"Region selected from detection: {roi.describe()}")
        self._update_actions()

    def _on_face_restore_requested(self, roi: ROI) -> None:
        """Select the face and open the restoration panel focused on it."""
        self._on_detection_roi(roi)
        self.restoration_panel.focus_task(TaskType.FACE_RESTORATION.value)
        self.inspector.show_tab(InspectorTab.RESTORE)
        self._status(
            "Face selected. Review the model's warning before running - face "
            "restoration synthesises facial detail."
        )

    def _on_toggle_safe_mode(self, enabled: bool) -> None:
        """Enable or disable Forensic Safe Mode."""
        if not enabled:
            answer = QMessageBox.warning(
                self, "Disable Forensic Safe Mode",
                SAFE_MODE_OFF_WARNING,
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                self.act_safe_mode.blockSignals(True)
                self.act_safe_mode.setChecked(True)
                self.act_safe_mode.blockSignals(False)
                return

        self._guard.set_enabled(enabled)
        self._config.safe_mode = enabled
        save_config(self._config)
        self._update_safe_mode_display()
        if self._case is not None:
            self._case.audit(
                action="safemode.toggle",
                detail=f"Forensic Safe Mode {'enabled' if enabled else 'DISABLED'}",
            )
        self._status(self._guard.status_text())

    def _on_preferences(self) -> None:
        """Open the preferences dialog."""
        from gui.dialogs.preferences import PreferencesDialog

        dialog = PreferencesDialog(self._config, self)
        if dialog.exec_() == PreferencesDialog.Accepted:
            save_config(self._config)
            self._refresh_models()
            self._update_device_label()
            self._status("Preferences saved")

    # ============================================================= export
    def _on_export(self) -> None:
        """Export the displayed derivative to a chosen path."""
        self._export_image(ask_format=False)

    def _on_export_as(self) -> None:
        """Export with an explicit format choice."""
        self._export_image(ask_format=True)

    def _export_image(self, ask_format: bool) -> None:
        """Write the current image to a new file, never over the original."""
        image = self._enhanced_image or self._current_image
        if image is None:
            return

        if self._evidence is not None:
            default = f"{Path(self._evidence.stored_path).stem}_export.png"
        else:
            default = "forensicvision_export.png"

        filters = (
            "PNG (*.png);;TIFF (*.tif *.tiff);;BMP (*.bmp);;"
            "WebP (*.webp);;JPEG (*.jpg *.jpeg)"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export derivative", default, filters
        )
        if not path:
            return

        destination = Path(path)
        try:
            self._guard.assert_can_write(destination)
        except ForensicVisionError as exc:
            show_error(self, "Blocked by Forensic Safe Mode", str(exc))
            return

        if destination.suffix.lower() not in LOSSLESS_EXPORT_FORMATS:
            answer = QMessageBox.warning(
                self, "Lossy export format",
                f"{destination.suffix.upper()} is a lossy format. The exported "
                "file will not be bit-identical to what you see, and its hash "
                "will not match the recorded derivative hash.\n\n"
                "Export as PNG or TIFF for evidential use.",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            save_image(image, destination, overwrite=True)
        except ForensicVisionError as exc:
            show_error(self, "Export failed", str(exc))
            return

        hashes = hash_file(destination)
        if self._case is not None:
            self._case.audit(
                action="derivative.export",
                target=destination.name,
                detail=f"sha256={hashes.sha256} | {destination}",
            )
        self._status(f"Exported {destination.name}  (sha256 {hashes.short()})")

    # =============================================================== help
    def _on_about(self) -> None:
        """Show the about box."""
        report = get_device_report()
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<h3>{APP_NAME} {APP_VERSION}</h3>"
            "<p>Forensic image analysis and enhancement workstation.</p>"
            f"<p style='color:#9aa3b2'>{build_string()}<br>"
            f"PyTorch {report.torch_version or 'not installed'}"
            f"{' / CUDA ' + report.cuda_version if report.cuda_available else ''}<br>"
            f"{report.device_label()}</p>"
            "<p style='color:#d99a2b'>Algorithmic enhancement produces derivative "
            "representations. AI-based restoration may infer or synthesise "
            "structures not present in the source evidence.</p>"
        )

    def _on_shortcuts(self) -> None:
        """List the keyboard shortcuts."""
        rows = [
            ("Ctrl+N", "New case"),
            ("Ctrl+Shift+O", "Open case"),
            ("Ctrl+O", "Import evidence"),
            ("Ctrl+Shift+I", "Open image without a case"),
            ("Ctrl+S", "Export derivative"),
            ("Ctrl+Shift+S", "Export derivative as..."),
            ("A", "Analyse image"),
            ("Shift+A", "Analyse ROI"),
            ("E", "Auto enhance"),
            ("Shift+E", "Enhance ROI"),
            ("Ctrl+R", "Run staged pipeline"),
            ("Ctrl+D", "Compare original / enhanced"),
            ("Ctrl+P", "Generate report"),
            ("Ctrl+M", "Model manager"),
            ("F", "Fit to window"),
            ("R", "Reset view"),
            ("1 / 2 / 4 / 8", "Zoom 100% / 200% / 400% / 800%"),
            ("Ctrl+ + / -", "Zoom in / out"),
            ("Mouse wheel", "Zoom under cursor"),
            ("Middle drag / Space+drag", "Pan"),
            ("C", "Toggle crosshair"),
            ("Right-click on the image", "Context menu - most actions live here"),
            ("F9", "Show / hide the inspector"),
            ("F11", "Focus mode - give the whole window to the image"),
            ("Alt+1 .. Alt+5", "Case / Analysis / Restore / History / Log tab"),
            ("Ctrl+Tab", "Cycle inspector tabs"),
            ("Ctrl+1..4", "Rectangle / ellipse / polygon / freehand ROI"),
            ("Esc", "Cancel ROI drawing"),
            ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo view change"),
        ]
        body = "".join(
            f"<tr><td style='padding-right:18px'><b>{key}</b></td>"
            f"<td>{description}</td></tr>"
            for key, description in rows
        )
        QMessageBox.information(
            self, "Keyboard shortcuts",
            f"<table>{body}</table>"
            "<p style='color:#9aa3b2'>Undo/redo affects the view only. "
            "Original evidence is never modified.</p>",
        )

    def _on_limitations(self) -> None:
        """Show the limitations and disclaimer text."""
        from app.constants import (
            FORENSIC_REPORT_DISCLAIMER,
            HEURISTIC_DISCLAIMER,
            OCR_DISCLAIMER,
            SYNTHESIS_WARNING,
        )

        QMessageBox.information(
            self, "Limitations and disclaimer",
            f"<h4>Enhancement</h4><p>{FORENSIC_REPORT_DISCLAIMER}</p>"
            f"<h4>Generative models</h4><p>{SYNTHESIS_WARNING}</p>"
            f"<h4>Analysis indicators</h4><p>{HEURISTIC_DISCLAIMER}</p>"
            f"<h4>OCR</h4><p>{OCR_DISCLAIMER}</p>",
        )

    # =============================================================== layout
    def _restore_layout(self) -> None:
        """Restore the saved window geometry and dock arrangement."""
        geometry = self._settings.value(SETTINGS_GEOMETRY)
        state = self._settings.value(SETTINGS_STATE)
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)

    def _save_layout(self) -> None:
        """Persist the window geometry and dock arrangement."""
        self._settings.setValue(SETTINGS_GEOMETRY, self.saveGeometry())
        self._settings.setValue(SETTINGS_STATE, self.saveState())

    def _on_reset_layout(self) -> None:
        """Restore the default dock arrangement."""
        self._settings.remove(SETTINGS_GEOMETRY)
        self._settings.remove(SETTINGS_STATE)
        self._inspector_dock.setFloating(False)
        self._inspector_dock.show()
        self.addDockWidget(Qt.RightDockWidgetArea, self._inspector_dock)
        self.resizeDocks([self._inspector_dock], [380], Qt.Horizontal)
        self.inspector.show_tab(InspectorTab.CASE)
        self.act_focus_mode.setChecked(False)
        self._status("Layout reset")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop workers, save the layout and close the case cleanly."""
        if self._worker is not None:
            answer = QMessageBox.question(
                self, "Operation in progress",
                "An operation is still running. Cancel it and exit?",
                QMessageBox.Cancel | QMessageBox.Yes, QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._worker.stop_and_wait(8000)

        self._save_layout()
        try:
            ModelRegistry.unload_all()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Error unloading models on exit")
        if self._case is not None:
            self._case.close()
        save_config(self._config)
        logger.info("%s closed", APP_NAME)
        super().closeEvent(event)
