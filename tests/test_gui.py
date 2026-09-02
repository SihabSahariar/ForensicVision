"""GUI tests: dialogs, panels and the worker-thread plumbing.

These construct real widgets against a real ``QApplication``. They avoid the
network and avoid any dialog that would block, so they run unattended.

The window is never shown: ``QWidget`` construction and signal wiring work
without a visible window, and on a headless machine Qt's ``offscreen``
platform is selected automatically by the fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from app.config import AppConfig
from app.constants import ModelKind, TaskType
from core.image_io import ImageData


@pytest.fixture(scope="session")
def qapp():
    """A session-wide QApplication, offscreen when there is no display."""
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    if not os.environ.get("DISPLAY") and os.name != "nt":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QApplication([])
    yield application


@pytest.fixture()
def gui_config(tmp_path) -> AppConfig:
    """A configuration pointing at throw-away directories."""
    return AppConfig(
        cases_root=str(tmp_path / "cases"),
        weights_root=str(tmp_path / "weights"),
        device="cpu",
        use_fp16=False,
        confirm_synthesis=False,
    )


@pytest.fixture()
def scene_image() -> ImageData:
    """A small RGB test image."""
    from scripts.make_sample import build_scene

    return ImageData(pixels=build_scene(240, 168, seed=5))


class TestImageViewer:
    """The pan/zoom canvas."""

    def test_zoom_and_fit(self, qapp, scene_image) -> None:
        """Zoom presets apply and fit-to-window resets the mode."""
        from gui.image_viewer import ImageViewer

        viewer = ImageViewer()
        viewer.resize(400, 300)
        viewer.set_image(scene_image)

        viewer.set_zoom(4.0)
        assert viewer.zoom == pytest.approx(4.0)
        assert viewer.fit_mode is False

        viewer.fit_to_window()
        assert viewer.fit_mode is True

    def test_zoom_is_clamped(self, qapp, scene_image) -> None:
        """Zoom cannot exceed the configured bounds."""
        from app.constants import MAX_ZOOM, MIN_ZOOM
        from gui.image_viewer import ImageViewer

        viewer = ImageViewer()
        viewer.set_image(scene_image)
        viewer.set_zoom(10_000.0)
        assert viewer.zoom <= MAX_ZOOM
        viewer.set_zoom(1e-6)
        assert viewer.zoom >= MIN_ZOOM

    def test_pixel_probe_reads_the_source_array(self, qapp) -> None:
        """The readout samples the source data, not the rendered pixmap."""
        from gui.image_viewer import ImageViewer

        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[4, 6] = (200, 100, 50)
        viewer = ImageViewer()
        viewer.set_image(ImageData(pixels=pixels))

        probes = []
        viewer.pixelProbed.connect(probes.append)
        from PyQt5.QtCore import QPointF

        viewer._emit_probe(QPointF(6.5, 4.5))
        assert probes
        assert probes[0].rgb == (200, 100, 50)
        assert probes[0].x == 6 and probes[0].y == 4

    def test_roi_round_trip(self, qapp, scene_image) -> None:
        """A programmatic ROI is stored and cleared."""
        from gui.image_viewer import ImageViewer
        from gui.roi_tools import ROI

        viewer = ImageViewer()
        viewer.set_image(scene_image)
        roi = ROI.from_box(10, 20, 40, 30, label="Test")
        viewer.set_roi(roi)
        assert viewer.roi is not None
        assert viewer.roi.bounding_box() == (10, 20, 40, 30)

        viewer.clear_roi()
        assert viewer.roi is None

    def test_clearing_the_image(self, qapp, scene_image) -> None:
        """Clearing removes the image without error."""
        from gui.image_viewer import ImageViewer

        viewer = ImageViewer()
        viewer.set_image(scene_image)
        assert viewer.has_image
        viewer.clear()
        assert not viewer.has_image


class TestComparisonViewer:
    """Before/after comparison."""

    def test_scale_normalisation(self, qapp, scene_image) -> None:
        """A 2x derivative is shown at the same field of view as the original.

        Copying the zoom verbatim would display the enlarged derivative at a
        different apparent scale, which defeats the point of the comparison.
        """
        import cv2

        from gui.comparison_viewer import ComparisonViewer

        viewer = ComparisonViewer()
        viewer.resize(800, 400)
        enlarged = ImageData(
            pixels=cv2.resize(
                scene_image.pixels,
                (scene_image.width * 2, scene_image.height * 2),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        viewer.set_images(scene_image, enlarged)

        left_zoom = viewer.left_viewer.zoom
        right_zoom = viewer.right_viewer.zoom
        assert right_zoom == pytest.approx(left_zoom / 2.0, rel=0.02)

    def test_scale_normalisation_survives_a_resize(
        self, qapp, scene_image
    ) -> None:
        """Resizing must not let the two panes drift out of scale.

        Only the left viewer stays in fit mode; the right is driven from it.
        A refit on resize therefore has to re-emit ``viewChanged``, or the
        right pane keeps a zoom computed for the old viewport - which is what
        happened whenever the Compare tab was first shown at a size other
        than the one it had while hidden.
        """
        import cv2

        from gui.comparison_viewer import ComparisonViewer

        viewer = ComparisonViewer()
        viewer.resize(400, 300)
        viewer.show()
        qapp.processEvents()
        enlarged = ImageData(
            pixels=cv2.resize(
                scene_image.pixels,
                (scene_image.width * 2, scene_image.height * 2),
                interpolation=cv2.INTER_NEAREST,
            )
        )
        viewer.set_images(scene_image, enlarged)
        qapp.processEvents()
        before = viewer.left_viewer.zoom

        viewer.resize(1000, 700)
        qapp.processEvents()

        # Guard the guard: a resize that changed nothing would pass trivially.
        assert viewer.left_viewer.zoom != pytest.approx(before, rel=1e-3)
        assert viewer.right_viewer.zoom == pytest.approx(
            viewer.left_viewer.zoom / 2.0, rel=0.02
        )
        viewer.close()

    def test_difference_modes_all_render(self, qapp, scene_image) -> None:
        """Every difference visualisation produces an image and statistics."""
        import cv2

        from gui.comparison_viewer import DifferenceMode, compute_difference

        modified = cv2.GaussianBlur(scene_image.pixels, (7, 7), 0)
        for mode in DifferenceMode.LABELS:
            visual, statistics = compute_difference(
                scene_image.pixels, modified, mode
            )
            assert visual.ndim == 3 and visual.shape[2] == 3, mode
            assert "psnr_db" in statistics

    def test_difference_of_identical_images_is_zero(
        self, qapp, scene_image
    ) -> None:
        """Comparing an image with itself reports no change."""
        from gui.comparison_viewer import DifferenceMode, compute_difference

        _, statistics = compute_difference(
            scene_image.pixels, scene_image.pixels, DifferenceMode.ABSOLUTE
        )
        assert statistics["mean_absolute_difference"] == 0.0
        assert statistics["psnr_db"] > 90


class TestAnalysisPanel:
    """The degradation dock."""

    def test_populates_every_bar(self, qapp, scene_image) -> None:
        """Every indicator gets a bar and a score."""
        from analysis import analyze_image
        from app.constants import DEGRADATION_ORDER
        from gui.analysis_panel import AnalysisPanel

        panel = AnalysisPanel()
        panel.set_report(analyze_image(scene_image))
        assert panel.report is not None
        for key in DEGRADATION_ORDER:
            assert key in panel._bars
            assert panel._bars[key].score >= 0.0

    def test_clear_resets(self, qapp, scene_image) -> None:
        """Clearing removes the report and blanks the bars."""
        from analysis import analyze_image
        from gui.analysis_panel import AnalysisPanel

        panel = AnalysisPanel()
        panel.set_report(analyze_image(scene_image))
        panel.clear()
        assert panel.report is None


class TestRestorationPanel:
    """The manual restoration dock."""

    def test_sections_populate(self, qapp, registry) -> None:
        """Every task with a registered model gets a visible section."""
        from gui.restoration_panel import RestorationPanel

        panel = RestorationPanel()
        panel.refresh_models()
        populated = [s for s in panel._sections if s.has_models]
        assert len(populated) >= 8

    def test_focus_task_expands_one_section(self, qapp, registry) -> None:
        """Focusing a task expands it and collapses the rest."""
        from gui.restoration_panel import RestorationPanel

        panel = RestorationPanel()
        panel.refresh_models()
        panel.focus_task(TaskType.DENOISE.value)

        expanded = [s.task for s in panel._sections if s._expanded]
        assert expanded == [TaskType.DENOISE.value]

    def test_staging_a_step(self, qapp, registry) -> None:
        """Adding to the pipeline updates the staged summary."""
        from gui.restoration_panel import RestorationPanel

        panel = RestorationPanel()
        panel.refresh_models()
        assert not panel.pipeline.enabled_steps

        panel._on_add_to_pipeline("clahe", {"clip_limit": 3.0})
        assert len(panel.pipeline.enabled_steps) == 1
        assert "CLAHE" in panel._pipeline_label.text()

        panel.clear_pipeline()
        assert not panel.pipeline.enabled_steps

    def test_unavailable_model_offers_install(self, qapp, registry) -> None:
        """A model without weights shows Install rather than Run."""
        from gui.restoration_panel import TaskSection

        section = TaskSection(TaskType.DEBLUR.value)
        section.refresh()
        index = section._model_combo.findData("nafnet_deblur")
        if index < 0:
            pytest.skip("NAFNet is not registered")
        section._model_combo.setCurrentIndex(index)

        assert section._run_button.isEnabled() is False
        # isVisible() is False for any widget whose top-level window was never
        # shown, so isHidden() is what reflects the explicit show/hide state.
        assert section._install_button.isHidden() is False


class TestPipelineEditor:
    """The review gate."""

    def test_lists_and_reorders(self, qapp, registry) -> None:
        """Steps are listed and can be moved."""
        from gui.dialogs.pipeline_editor import PipelineEditorDialog
        from restoration.pipeline import Pipeline, PipelineStep

        pipeline = Pipeline(
            steps=[PipelineStep("clahe"), PipelineStep("unsharp")]
        )
        dialog = PipelineEditorDialog(pipeline)
        assert dialog._tree.topLevelItemCount() == 2

        dialog._tree.setCurrentItem(dialog._tree.topLevelItem(0))
        dialog._move_down()
        assert dialog.pipeline.steps[0].model_name == "unsharp"

    def test_disabling_a_step(self, qapp, registry) -> None:
        """Unchecking a row disables that step."""
        from PyQt5.QtCore import Qt

        from gui.dialogs.pipeline_editor import PipelineEditorDialog
        from restoration.pipeline import Pipeline, PipelineStep

        dialog = PipelineEditorDialog(
            Pipeline(steps=[PipelineStep("clahe"), PipelineStep("unsharp")])
        )
        item = dialog._tree.topLevelItem(0)
        item.setCheckState(0, Qt.Unchecked)
        assert len(dialog.pipeline.enabled_steps) == 1

    def test_generative_pipeline_is_flagged(self, qapp, registry) -> None:
        """A generative step raises the warning banner."""
        from gui.dialogs.pipeline_editor import PipelineEditorDialog
        from restoration.pipeline import Pipeline, PipelineStep

        dialog = PipelineEditorDialog(
            Pipeline(steps=[PipelineStep("realesrgan_x4plus")])
        )
        assert dialog._warning_banner.isVisible() or dialog._warning_banner.text()
        assert "generative" in dialog._warning_banner.text().lower()

    def test_classical_pipeline_says_so(self, qapp, registry) -> None:
        """An all-classical pipeline gets the reassuring banner instead."""
        from gui.dialogs.pipeline_editor import PipelineEditorDialog
        from restoration.pipeline import Pipeline, PipelineStep

        dialog = PipelineEditorDialog(Pipeline(steps=[PipelineStep("clahe")]))
        assert "deterministic" in dialog._warning_banner.text().lower()


class TestModelManager:
    """The installation dialog."""

    def test_lists_every_registered_model(self, qapp, registry) -> None:
        """The table has one row per model."""
        from gui.model_manager import ModelManagerDialog

        dialog = ModelManagerDialog()
        assert len(dialog._rows) == len(registry.infos())
        assert dialog._table.rowCount() == len(dialog._rows)

    def test_detail_pane_shows_licence_and_source(self, qapp, registry) -> None:
        """Selecting a neural model shows its provenance."""
        from gui.model_manager import ModelManagerDialog

        dialog = ModelManagerDialog()
        dialog._select_by_name("realesrgan_x4plus")
        text = dialog._detail.toPlainText()
        assert "Licence" in text
        assert "Repository" in text
        assert "github.com" in text
        assert "synthesise" in text.lower()

    def test_unintegrated_model_cannot_be_installed(
        self, qapp, registry
    ) -> None:
        """An unintegrated model offers no Install action."""
        from gui.model_manager import ModelManagerDialog

        dialog = ModelManagerDialog()
        dialog._select_by_name("lama")
        assert dialog._install_button.isEnabled() is False


class TestOcrDialog:
    """OCR degrades cleanly when no engine is installed."""

    def test_reports_missing_engine(self, qapp, scene_image, monkeypatch) -> None:
        """With no engine the Run button is disabled and the fix is stated."""
        monkeypatch.setattr("gui.dialogs.ocr_dialog.available_engines", lambda: [])
        from gui.dialogs.ocr_dialog import OcrDialog

        dialog = OcrDialog(scene_image, None)
        assert dialog._run_button.isEnabled() is False

        banners = [
            widget.text()
            for widget in dialog.findChildren(type(dialog._status))
            if "pip install" in widget.text()
        ]
        assert banners, "the dialog must say how to install an engine"

    def test_disclaimer_is_present(self, qapp, scene_image, monkeypatch) -> None:
        """The OCR caveat is shown regardless of engine availability."""
        from app.constants import OCR_DISCLAIMER

        monkeypatch.setattr("gui.dialogs.ocr_dialog.available_engines", lambda: [])
        from gui.dialogs.ocr_dialog import OcrDialog

        dialog = OcrDialog(scene_image, None)
        texts = " ".join(
            widget.text() for widget in dialog.findChildren(type(dialog._status))
        )
        assert OCR_DISCLAIMER[:40] in texts


class TestDetectDialog:
    """Detection source switching and availability reporting."""

    def test_face_source_reports_availability(self, qapp, scene_image) -> None:
        """Switching to faces reports whether the detector can run."""
        from detection.face import detector_available
        from gui.dialogs.detect_dialog import DetectDialog

        dialog = DetectDialog(scene_image)
        dialog._source.setCurrentIndex(dialog._source.findData("faces"))

        available, _ = detector_available()
        assert dialog._run_button.isEnabled() is available
        if not available:
            assert dialog._availability.text()
            assert dialog._availability.isHidden() is False

    def test_face_columns_are_labelled(self, qapp, scene_image) -> None:
        """The face table reports the inter-ocular measurement."""
        from gui.dialogs.detect_dialog import DetectDialog

        dialog = DetectDialog(scene_image)
        dialog._source.setCurrentIndex(dialog._source.findData("faces"))
        headers = [
            dialog._table.horizontalHeaderItem(i).text()
            for i in range(dialog._table.columnCount())
        ]
        assert "Inter-ocular" in headers

    def test_minimum_face_default_is_permissive(self, qapp, scene_image) -> None:
        """The size filter does not hide small surveillance faces by default."""
        from detection.face import YuNetDetector
        from gui.dialogs.detect_dialog import DetectDialog

        dialog = DetectDialog(scene_image)
        assert dialog._min_face.value() == YuNetDetector.DEFAULT_MIN_SIZE
        assert dialog._min_face.value() <= 16


class TestBatchDialog:
    """Batch discovery and controls."""

    def test_discovers_supported_images(
        self, qapp, gui_config, tmp_path, case
    ) -> None:
        """Only supported extensions are listed."""
        import cv2

        from gui.dialogs.batch_dialog import BatchDialog

        folder = tmp_path / "batch"
        folder.mkdir()
        for name in ("a.png", "b.jpg", "c.tif"):
            cv2.imwrite(str(folder / name), np.zeros((32, 32, 3), np.uint8))
        (folder / "notes.txt").write_text("ignore me")

        dialog = BatchDialog(case, gui_config)
        dialog._folder_edit.setText(str(folder))
        dialog._rescan()

        assert len(dialog._paths) == 3
        assert dialog._start.isEnabled()

    def test_empty_folder_disables_start(
        self, qapp, gui_config, tmp_path, case
    ) -> None:
        """Nothing to process means Start stays disabled."""
        from gui.dialogs.batch_dialog import BatchDialog

        folder = tmp_path / "empty"
        folder.mkdir()
        dialog = BatchDialog(case, gui_config)
        dialog._folder_edit.setText(str(folder))
        dialog._rescan()
        assert dialog._start.isEnabled() is False


class TestReportDialog:
    """Report options and context assembly."""

    def test_context_carries_the_mandatory_fields(
        self, qapp, case, sample_png
    ) -> None:
        """The assembled context includes evidence, history and environment."""
        from gui.dialogs.report_dialog import ReportDialog

        evidence = case.import_evidence(sample_png).evidence
        dialog = ReportDialog(case=case, evidence=evidence)
        context = dialog._build_context()

        assert context["evidence"] is evidence
        assert "history" in context
        assert context["environment"]["application_version"]

    def test_difference_disabled_without_a_pair(
        self, qapp, case, sample_png
    ) -> None:
        """Difference analysis is offered only when both images exist."""
        from gui.dialogs.report_dialog import ReportDialog

        evidence = case.import_evidence(sample_png).evidence
        dialog = ReportDialog(case=case, evidence=evidence)
        assert dialog._include_difference.isEnabled() is False


class TestMainWindow:
    """Top-level wiring."""

    def test_single_inspector_dock(self, qapp, gui_config) -> None:
        """The window has exactly one dock, hosting every panel.

        Six separate docks consumed roughly 600 px of width and 220 px of
        height between them, squeezing the image and clipping the taller
        panels. Consolidating them is what returns that space to the image.
        """
        from PyQt5.QtWidgets import QDockWidget
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        docks = window.findChildren(QDockWidget)
        assert len(docks) == 1, [d.windowTitle() for d in docks]
        assert docks[0] is window._inspector_dock

        for panel in (
            window.case_explorer, window.metadata_panel, window.analysis_panel,
            window.restoration_panel, window.history_panel, window.log_panel,
        ):
            assert panel is not None
        window.close()

    def test_every_panel_reachable_by_tab(self, qapp, gui_config) -> None:
        """Each panel can be brought to the front by its tab."""
        from gui.inspector import InspectorTab
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        for tab_id, _label, _tip in InspectorTab.ORDER:
            window.inspector.show_tab(tab_id)
            assert window.inspector.current_tab == tab_id
        window.close()

    def test_focus_mode_hides_the_inspector(self, qapp, gui_config) -> None:
        """Focus mode gives the whole window to the image."""
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        assert window._inspector_dock.isHidden() is False

        window.act_focus_mode.setChecked(True)
        assert window._inspector_dock.isHidden() is True
        assert window.act_toggle_inspector.isChecked() is False

        window.act_focus_mode.setChecked(False)
        assert window._inspector_dock.isHidden() is False
        window.close()

    def test_inspector_toggle(self, qapp, gui_config) -> None:
        """The inspector can be hidden and shown independently."""
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        window.act_toggle_inspector.setChecked(False)
        assert window._inspector_dock.isHidden() is True
        window.act_toggle_inspector.setChecked(True)
        assert window._inspector_dock.isHidden() is False
        window.close()

    def test_showing_a_tab_reveals_a_hidden_inspector(
        self, qapp, gui_config
    ) -> None:
        """Jumping to a panel un-hides the dock rather than doing nothing."""
        from gui.inspector import InspectorTab
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        window.act_toggle_inspector.setChecked(False)
        window._show_inspector_tab(InspectorTab.ANALYSIS)
        assert window._inspector_dock.isHidden() is False
        assert window.inspector.current_tab == InspectorTab.ANALYSIS
        window.close()

    def test_actions_disabled_without_a_case(self, qapp, gui_config) -> None:
        """Case-dependent actions start disabled."""
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        assert window.act_import.isEnabled() is False
        assert window.act_report.isEnabled() is False
        assert window.act_analyse.isEnabled() is False
        window.close()

    def test_safe_mode_shown_in_status_bar(self, qapp, gui_config) -> None:
        """The safe-mode state is visible, not buried in a menu."""
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        assert "SAFE MODE" in window._safe_mode_label.text().upper()
        window.close()

    def test_visualisations_do_not_alter_the_working_image(
        self, qapp, gui_config, sample_png
    ) -> None:
        """An analytical view is displayed, then cleanly reverted."""
        from analysis.visualizations import VISUALIZATIONS
        from gui.main_window import MainWindow

        window = MainWindow(gui_config)
        window._load_standalone_path(Path(sample_png), notify=False)
        original = window._original_image.pixels.copy()

        for name in VISUALIZATIONS:
            window._show_visualization(name)
            assert window._visualization_banner.isHidden() is False, name
            assert "ANALYTICAL VIEW" in window._visualization_banner.text()

        window._clear_visualization()
        assert np.array_equal(window._original_image.pixels, original)
        assert window._visualization_banner.isHidden() is True
        window.close()
