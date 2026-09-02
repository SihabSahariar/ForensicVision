"""Before / after comparison and difference visualisation.

Four modes:

* **Side by side** - two viewers with locked zoom, pan and cursor.
* **Split** - one image revealed left of a draggable divider, the other right.
* **Overlay** - a slider cross-fades between the two.
* **Difference** - analytical views of what actually changed.

Difference views are explicitly labelled as analytical visualisations, not
evidence: an amplified difference map is a rendering choice, and its apparent
intensity depends entirely on the gain applied.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.image_io import ImageData
from core.image_utils import ensure_uint8_rgb
from gui.image_viewer import ImageViewer, PixelProbe
from gui.theme import Palette
from gui.widgets.common import BannerLabel, SectionLabel

logger = logging.getLogger(__name__)

__all__ = ["ComparisonViewer", "DifferenceMode", "compute_difference"]


class DifferenceMode:
    """Available difference visualisations."""

    ABSOLUTE = "absolute"
    GRAYSCALE = "grayscale"
    AMPLIFIED = "amplified"
    EDGE = "edge"
    HEATMAP = "heatmap"

    LABELS = {
        ABSOLUTE: "Absolute difference (RGB)",
        GRAYSCALE: "Grayscale difference",
        AMPLIFIED: "Amplified difference (x8)",
        EDGE: "Edge difference",
        HEATMAP: "Difference heatmap",
    }


DIFFERENCE_DISCLAIMER = (
    "Analytical visualisation, not evidence. Difference maps are rendered with "
    "a chosen gain and colour mapping; their apparent intensity is a display "
    "choice. Where the two images differ in size the original is resampled for "
    "comparison, which itself introduces small differences."
)


def compute_difference(
    original: np.ndarray, enhanced: np.ndarray, mode: str = DifferenceMode.ABSOLUTE
) -> Tuple[np.ndarray, dict]:
    """Compute a difference visualisation between two images.

    Args:
        original: Reference image (any supported dtype).
        enhanced: Comparison image.
        mode: One of the :class:`DifferenceMode` values.

    Returns:
        ``(visualisation_uint8_rgb, statistics)``.
    """
    left = ensure_uint8_rgb(original)[..., :3]
    right = ensure_uint8_rgb(enhanced)[..., :3]

    resampled = False
    if left.shape[:2] != right.shape[:2]:
        # Compare at the larger geometry so detail added by super-resolution is
        # visible rather than being averaged away.
        target = (max(left.shape[1], right.shape[1]), max(left.shape[0], right.shape[0]))
        if left.shape[1] != target[0] or left.shape[0] != target[1]:
            left = cv2.resize(left, target, interpolation=cv2.INTER_LANCZOS4)
            resampled = True
        if right.shape[1] != target[0] or right.shape[0] != target[1]:
            right = cv2.resize(right, target, interpolation=cv2.INTER_LANCZOS4)
            resampled = True

    absolute = cv2.absdiff(left, right)
    gray = cv2.cvtColor(absolute, cv2.COLOR_RGB2GRAY)

    statistics = {
        "mean_absolute_difference": float(absolute.mean()),
        "max_absolute_difference": int(absolute.max()),
        "changed_pixel_fraction": float((gray > 2).mean()),
        "rms_difference": float(np.sqrt(np.mean(absolute.astype(np.float64) ** 2))),
        "resampled_for_comparison": resampled,
    }
    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    statistics["psnr_db"] = (
        99.0 if mse < 1e-9 else float(10.0 * np.log10(255.0 ** 2 / mse))
    )

    if mode == DifferenceMode.ABSOLUTE:
        return absolute, statistics
    if mode == DifferenceMode.GRAYSCALE:
        return np.stack([gray] * 3, axis=-1), statistics
    if mode == DifferenceMode.AMPLIFIED:
        amplified = np.clip(absolute.astype(np.int32) * 8, 0, 255).astype(np.uint8)
        return amplified, statistics
    if mode == DifferenceMode.EDGE:
        left_edges = cv2.Canny(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), 60, 160)
        right_edges = cv2.Canny(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY), 60, 160)
        # Red: edges only in the original. Green: edges only in the derivative.
        visual = np.zeros_like(left)
        visual[..., 0] = cv2.subtract(left_edges, right_edges)
        visual[..., 1] = cv2.subtract(right_edges, left_edges)
        visual[..., 2] = cv2.bitwise_and(left_edges, right_edges)
        return visual, statistics
    if mode == DifferenceMode.HEATMAP:
        normalised = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        coloured = cv2.applyColorMap(normalised.astype(np.uint8), cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB), statistics

    return absolute, statistics


class _SplitOverlayView(QWidget):
    """Renders two images with a split or cross-fade, sharing one viewer."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._original: Optional[ImageData] = None
        self._enhanced: Optional[ImageData] = None
        self._position = 0.5
        self._mode = "split"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.viewer = ImageViewer()
        layout.addWidget(self.viewer, 1)

        controls = QHBoxLayout()
        controls.setContentsMargins(8, 0, 8, 6)
        controls.setSpacing(8)
        self._left_label = QLabel("Original")
        self._left_label.setProperty("role", "hint")
        controls.addWidget(self._left_label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(500)
        self._slider.valueChanged.connect(self._on_slider)
        controls.addWidget(self._slider, 1)

        self._right_label = QLabel("Enhanced")
        self._right_label.setProperty("role", "hint")
        controls.addWidget(self._right_label)
        layout.addLayout(controls)

    def set_images(
        self, original: Optional[ImageData], enhanced: Optional[ImageData]
    ) -> None:
        """Set the pair being compared."""
        self._original = original
        self._enhanced = enhanced
        self._render()

    def set_mode(self, mode: str) -> None:
        """Switch between ``"split"`` and ``"overlay"``."""
        self._mode = mode
        self._render()

    def _on_slider(self, value: int) -> None:
        self._position = value / 1000.0
        self._render(keep_view=True)

    def _render(self, keep_view: bool = False) -> None:
        """Composite the two images according to the current mode."""
        if self._original is None or self._enhanced is None:
            self.viewer.set_image(None)
            return

        left = ensure_uint8_rgb(self._original.pixels)[..., :3]
        right = ensure_uint8_rgb(self._enhanced.pixels)[..., :3]
        if left.shape[:2] != right.shape[:2]:
            target = (right.shape[1], right.shape[0])
            left = cv2.resize(left, target, interpolation=cv2.INTER_LANCZOS4)

        if self._mode == "overlay":
            alpha = self._position
            composite = cv2.addWeighted(left, 1.0 - alpha, right, alpha, 0.0)
        else:
            composite = right.copy()
            boundary = int(round(self._position * composite.shape[1]))
            boundary = max(0, min(composite.shape[1], boundary))
            composite[:, :boundary] = left[:, :boundary]
            if 0 < boundary < composite.shape[1]:
                line = QColorLine(Palette.ACCENT)
                composite[:, max(0, boundary - 1):boundary + 1] = line

        self.viewer.set_image(
            ImageData(pixels=composite), keep_view=keep_view
        )


def QColorLine(hex_colour: str) -> np.ndarray:
    """Return the RGB triple for ``hex_colour`` as a uint8 array."""
    value = hex_colour.lstrip("#")
    return np.array(
        [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)], dtype=np.uint8
    )


class ComparisonViewer(QWidget):
    """Before/after comparison with synchronised navigation.

    Signals:
        pixelProbed: ``(PixelProbe)`` forwarded from the active viewer.
        statisticsChanged: ``(dict)`` after a difference is computed.
    """

    pixelProbed = pyqtSignal(object)
    statisticsChanged = pyqtSignal(dict)

    MODE_SIDE_BY_SIDE = "side_by_side"
    MODE_SPLIT = "split"
    MODE_OVERLAY = "overlay"
    MODE_DIFFERENCE = "difference"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._original: Optional[ImageData] = None
        self._enhanced: Optional[ImageData] = None
        self._syncing = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(8)

        toolbar_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Side by side", self.MODE_SIDE_BY_SIDE)
        self._mode_combo.addItem("Split view", self.MODE_SPLIT)
        self._mode_combo.addItem("Overlay", self.MODE_OVERLAY)
        self._mode_combo.addItem("Difference", self.MODE_DIFFERENCE)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar_layout.addWidget(self._mode_combo)

        self._difference_combo = QComboBox()
        for key, label in DifferenceMode.LABELS.items():
            self._difference_combo.addItem(label, key)
        self._difference_combo.currentIndexChanged.connect(self._refresh_difference)
        self._difference_combo.setVisible(False)
        toolbar_layout.addWidget(self._difference_combo)

        self._stats_label = QLabel("")
        self._stats_label.setProperty("role", "mono")
        toolbar_layout.addWidget(self._stats_label, 1)

        layout.addWidget(toolbar)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)

        # Side-by-side page
        self._side_page = QWidget()
        side_layout = QVBoxLayout(self._side_page)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(2)

        labels = QHBoxLayout()
        labels.setContentsMargins(8, 2, 8, 2)
        left_label = QLabel("ORIGINAL")
        left_label.setProperty("role", "section")
        right_label = QLabel("ENHANCED")
        right_label.setProperty("role", "section")
        labels.addWidget(left_label, 1)
        labels.addWidget(right_label, 1)
        side_layout.addLayout(labels)

        self._splitter = QSplitter(Qt.Horizontal)
        self.left_viewer = ImageViewer()
        self.right_viewer = ImageViewer()
        self._splitter.addWidget(self.left_viewer)
        self._splitter.addWidget(self.right_viewer)
        self._splitter.setSizes([1, 1])
        side_layout.addWidget(self._splitter, 1)
        self._stack.addWidget(self._side_page)

        # Split / overlay page
        self._blend_page = _SplitOverlayView()
        self._stack.addWidget(self._blend_page)

        # Difference page
        self._difference_page = QWidget()
        difference_layout = QVBoxLayout(self._difference_page)
        difference_layout.setContentsMargins(0, 0, 0, 0)
        difference_layout.setSpacing(4)
        self.difference_viewer = ImageViewer()
        difference_layout.addWidget(self.difference_viewer, 1)
        self._difference_banner = BannerLabel(DIFFERENCE_DISCLAIMER)
        difference_layout.addWidget(self._difference_banner)
        self._stack.addWidget(self._difference_page)

        self._connect_sync()

    def _connect_sync(self) -> None:
        """Lock the two side-by-side viewers together."""
        self.left_viewer.viewChanged.connect(
            lambda: self._sync_from(self.left_viewer, self.right_viewer)
        )
        self.right_viewer.viewChanged.connect(
            lambda: self._sync_from(self.right_viewer, self.left_viewer)
        )
        for viewer in (self.left_viewer, self.right_viewer,
                       self.difference_viewer, self._blend_page.viewer):
            viewer.pixelProbed.connect(self.pixelProbed.emit)

    @staticmethod
    def _relative_scale(source: ImageViewer, target: ImageViewer) -> float:
        """Zoom multiplier making ``target`` show ``source``'s field of view.

        After super-resolution the two images have different pixel dimensions,
        so copying the zoom factor verbatim would show the enlarged derivative
        at a different apparent scale from the original - which defeats the
        purpose of a side-by-side comparison.

        A viewer of width ``W`` at zoom ``Z`` displays ``W/Z`` image pixels, so
        it covers the fraction ``W / (Z * image_width)`` of its image. Equating
        that fraction across the pair gives
        ``Z_target = Z_source * source_width / target_width``. Scrollbar values
        are in viewport pixels and cancel out exactly, so they transfer
        unchanged.
        """
        source_image = source.image
        target_image = target.image
        if source_image is None or target_image is None:
            return 1.0
        if target_image.width <= 0:
            return 1.0
        return source_image.width / float(target_image.width)

    def _sync_from(self, source: ImageViewer, target: ImageViewer) -> None:
        """Mirror ``source``'s view state onto ``target``, matching scale."""
        if self._syncing:
            return
        self._syncing = True
        try:
            zoom, horizontal, vertical = source.view_state()
            target.apply_view_state(
                (zoom * self._relative_scale(source, target), horizontal, vertical)
            )
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ public
    def set_images(
        self, original: Optional[ImageData], enhanced: Optional[ImageData]
    ) -> None:
        """Set the pair being compared and refresh every mode."""
        self._original = original
        self._enhanced = enhanced

        self.left_viewer.set_image(original)
        self.right_viewer.set_image(enhanced)
        self._blend_page.set_images(original, enhanced)
        self._refresh_difference()

        if original is not None and enhanced is not None:
            # Fit the original, then derive the derivative's view from it so
            # both panes open showing the same region at the same apparent
            # scale, whatever the size difference between them.
            self.left_viewer.fit_to_window()
            self._sync_from(self.left_viewer, self.right_viewer)

    @property
    def mode(self) -> str:
        """The active comparison mode."""
        return self._mode_combo.currentData()

    def set_mode(self, mode: str) -> None:
        """Switch to ``mode``."""
        index = self._mode_combo.findData(mode)
        if index >= 0:
            self._mode_combo.setCurrentIndex(index)

    def difference_statistics(self) -> dict:
        """Return the statistics from the most recent difference computation."""
        return getattr(self, "_statistics", {})

    def current_difference_image(self) -> Optional[np.ndarray]:
        """Return the rendered difference visualisation, if any."""
        return getattr(self, "_difference_image", None)

    # --------------------------------------------------------------- handlers
    def _on_mode_changed(self) -> None:
        """Show the page matching the selected mode."""
        mode = self.mode
        self._difference_combo.setVisible(mode == self.MODE_DIFFERENCE)
        if mode == self.MODE_SIDE_BY_SIDE:
            self._stack.setCurrentWidget(self._side_page)
        elif mode in (self.MODE_SPLIT, self.MODE_OVERLAY):
            self._blend_page.set_mode(
                "split" if mode == self.MODE_SPLIT else "overlay"
            )
            self._stack.setCurrentWidget(self._blend_page)
        else:
            self._stack.setCurrentWidget(self._difference_page)
            self._refresh_difference()

    def _refresh_difference(self) -> None:
        """Recompute and display the selected difference visualisation."""
        if self._original is None or self._enhanced is None:
            self.difference_viewer.set_image(None)
            self._stats_label.setText("")
            return

        mode = self._difference_combo.currentData() or DifferenceMode.ABSOLUTE
        try:
            visual, statistics = compute_difference(
                self._original.pixels, self._enhanced.pixels, mode
            )
        except Exception:
            logger.exception("Difference computation failed")
            return

        self._difference_image = visual
        self._statistics = statistics
        self.difference_viewer.set_image(ImageData(pixels=visual), keep_view=True)

        self._stats_label.setText(
            f"mean |diff| {statistics['mean_absolute_difference']:.2f}   "
            f"max {statistics['max_absolute_difference']}   "
            f"changed {statistics['changed_pixel_fraction'] * 100:.1f}%   "
            f"PSNR {statistics['psnr_db']:.2f} dB"
        )
        self.statisticsChanged.emit(statistics)
