"""High-precision image viewer built on QGraphicsView.

Forensic design decisions:

* Above 200% zoom the pixmap is drawn with **nearest-neighbour** sampling so an
  examiner inspects real samples rather than an interpolated reconstruction.
* Pixel readout is taken from the source array, not from the rendered pixmap.
* ROIs are overlay items; they never modify the displayed image.
* The view exposes its transform state so two viewers can be locked together
  for before/after comparison.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from app.constants import MAX_ZOOM, MIN_ZOOM, ZOOM_STEP
from core.image_io import ImageData
from core.image_utils import image_to_qpixmap, sample_pixel
from gui.roi_tools import ROI, ROIOverlayItem, ROIType, make_overlay_item
from gui.theme import Palette

logger = logging.getLogger(__name__)

__all__ = ["ImageViewer", "PixelProbe"]

#: Zoom factor at and above which nearest-neighbour sampling is used.
_PIXEL_EXACT_THRESHOLD = 2.0


class PixelProbe:
    """Immutable pixel readout emitted by :attr:`ImageViewer.pixelProbed`."""

    __slots__ = ("x", "y", "components", "gray", "hsv")

    def __init__(
        self,
        x: int,
        y: int,
        components: Tuple[int, ...],
        gray: int,
        hsv: Tuple[int, int, int],
    ) -> None:
        self.x = x
        self.y = y
        self.components = components
        self.gray = gray
        self.hsv = hsv

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """The first three components as an RGB triple."""
        return tuple(self.components[:3])  # type: ignore[return-value]

    @property
    def alpha(self) -> Optional[int]:
        """Alpha component when present."""
        return self.components[3] if len(self.components) > 3 else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PixelProbe(x={self.x}, y={self.y}, rgb={self.rgb})"


class ImageViewer(QGraphicsView):
    """Pan/zoom image canvas with pixel inspection and ROI drawing.

    Signals:
        pixelProbed: Emitted with a :class:`PixelProbe` while hovering.
        cursorLeft: Emitted when the cursor leaves the image area.
        zoomChanged: Emitted with the new scale factor.
        viewChanged: Emitted after any pan or zoom (used for view syncing).
        roiCreated: Emitted with a finished :class:`~gui.roi_tools.ROI`.
        roiCleared: Emitted when the active ROI is removed.
    """

    pixelProbed = pyqtSignal(object)
    cursorLeft = pyqtSignal()
    zoomChanged = pyqtSignal(float)
    viewChanged = pyqtSignal()
    roiCreated = pyqtSignal(object)
    roiCleared = pyqtSignal()
    #: ``(global_position)`` when the user asks for the context menu. The
    #: viewer does not build the menu itself - the main window owns the
    #: actions, so it populates one and shows it at this position.
    contextMenuRequested = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        self._image: Optional[ImageData] = None
        self._zoom: float = 1.0
        self._fit_mode: bool = True
        self._panning: bool = False
        self._space_held: bool = False
        self._pan_origin = QPoint()
        self._crosshair_enabled = False
        self._cursor_scene_pos: Optional[QPointF] = None
        self._syncing = False

        # ROI state
        self._roi_mode: Optional[ROIType] = None
        self._roi_item: Optional[ROIOverlayItem] = None
        self._roi_drawing = False
        self._roi_points: List[Tuple[float, float]] = []
        self._active_roi: Optional[ROI] = None

        self._configure_view()

    # ------------------------------------------------------------------ setup
    def _configure_view(self) -> None:
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setBackgroundBrush(QBrush(QColor(Palette.VIEWPORT)))
        self.setFrameShape(QGraphicsView.NoFrame)

        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self._on_scroll)

    # ------------------------------------------------------------------ image
    @property
    def image(self) -> Optional[ImageData]:
        """The currently displayed image, if any."""
        return self._image

    @property
    def has_image(self) -> bool:
        """Whether an image is loaded."""
        return self._image is not None

    def set_image(self, image: Optional[ImageData], keep_view: bool = False) -> None:
        """Display ``image``.

        Args:
            image: Image to show, or ``None`` to clear the canvas.
            keep_view: Preserve the current zoom/pan (used when toggling between
                analytic visualisations of the same frame).
        """
        previous_transform = self.transform()
        previous_h = self.horizontalScrollBar().value()
        previous_v = self.verticalScrollBar().value()
        same_geometry = (
            self._image is not None
            and image is not None
            and self._image.shape[:2] == image.shape[:2]
        )

        self._image = image
        self.clear_roi(emit=False)

        if image is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(QRectF())
            self.resetTransform()
            self._zoom = 1.0
            self.viewport().update()
            return

        pixmap = image_to_qpixmap(image)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        if keep_view and same_geometry:
            self.setTransform(previous_transform)
            self.horizontalScrollBar().setValue(previous_h)
            self.verticalScrollBar().setValue(previous_v)
            self._update_sampling_mode()
        else:
            self.fit_to_window()

    def clear(self) -> None:
        """Remove the current image and any ROI."""
        self.set_image(None)

    # ------------------------------------------------------------------- zoom
    @property
    def zoom(self) -> float:
        """Current scale factor (1.0 == 100%)."""
        return self._zoom

    @property
    def fit_mode(self) -> bool:
        """Whether the view rescales to fit on resize."""
        return self._fit_mode

    def set_zoom(self, factor: float, anchor_under_mouse: bool = False) -> None:
        """Set an absolute zoom factor, clamped to the supported range."""
        if not self.has_image:
            return
        factor = max(MIN_ZOOM, min(MAX_ZOOM, float(factor)))
        if abs(factor - self._zoom) < 1e-9:
            return

        self.setTransformationAnchor(
            QGraphicsView.AnchorUnderMouse
            if anchor_under_mouse
            else QGraphicsView.AnchorViewCenter
        )
        self.resetTransform()
        self.scale(factor, factor)
        self._zoom = factor
        self._fit_mode = False
        self._update_sampling_mode()
        self._update_roi_pen()
        self.zoomChanged.emit(self._zoom)
        self._emit_view_changed()

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        self.set_zoom(self._zoom * ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        self.set_zoom(self._zoom / ZOOM_STEP)

    def fit_to_window(self) -> None:
        """Scale the image so it fits entirely inside the viewport."""
        if not self.has_image:
            return
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom = float(self.transform().m11())
        self._fit_mode = True
        self._update_sampling_mode()
        self._update_roi_pen()
        self.zoomChanged.emit(self._zoom)
        self._emit_view_changed()

    def reset_view(self) -> None:
        """Return to fit-to-window (bound to ``R``)."""
        self.fit_to_window()

    def zoom_to_roi(self, roi: ROI, margin: float = 0.08) -> None:
        """Zoom so that ``roi`` fills the viewport with a small margin."""
        if not self.has_image or not roi.is_valid():
            return
        x, y, w, h = roi.bounding_box()
        pad_x, pad_y = w * margin, h * margin
        target = QRectF(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.fitInView(target, Qt.KeepAspectRatio)
        self._zoom = float(self.transform().m11())
        self._fit_mode = False
        self._update_sampling_mode()
        self._update_roi_pen()
        self.zoomChanged.emit(self._zoom)
        self._emit_view_changed()

    def _update_sampling_mode(self) -> None:
        """Switch between smooth and pixel-exact rendering.

        Below 200% smoothing avoids aliasing artefacts that could be mistaken
        for image content; at or above 200% the examiner must see raw samples.
        """
        mode = (
            Qt.FastTransformation
            if self._zoom >= _PIXEL_EXACT_THRESHOLD
            else Qt.SmoothTransformation
        )
        if self._pixmap_item.transformationMode() != mode:
            self._pixmap_item.setTransformationMode(mode)

    # -------------------------------------------------------------- view sync
    def view_state(self) -> Tuple[float, int, int]:
        """Return ``(zoom, h_scroll, v_scroll)`` for mirroring onto a peer."""
        return (
            self._zoom,
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value(),
        )

    def apply_view_state(self, state: Tuple[float, int, int]) -> None:
        """Apply a peer's :meth:`view_state` without re-emitting sync signals."""
        if self._syncing:
            return
        zoom, h_value, v_value = state
        self._syncing = True
        try:
            if abs(zoom - self._zoom) > 1e-9:
                self.setTransformationAnchor(QGraphicsView.NoAnchor)
                self.resetTransform()
                self.scale(zoom, zoom)
                self._zoom = zoom
                self._fit_mode = False
                self._update_sampling_mode()
                self._update_roi_pen()
            self.horizontalScrollBar().setValue(h_value)
            self.verticalScrollBar().setValue(v_value)
        finally:
            self._syncing = False

    def _emit_view_changed(self) -> None:
        if not self._syncing:
            self.viewChanged.emit()

    def _on_scroll(self, _value: int) -> None:
        self._emit_view_changed()

    # ------------------------------------------------------------------- ROI
    @property
    def roi(self) -> Optional[ROI]:
        """The active ROI, if one has been drawn."""
        return self._active_roi

    def set_roi_mode(self, mode: Optional[ROIType]) -> None:
        """Enter or leave ROI drawing mode.

        Args:
            mode: The geometry to draw, or ``None`` to return to navigation.
        """
        self._roi_mode = mode
        self._roi_drawing = False
        self._roi_points = []
        if mode is None:
            self.viewport().setCursor(Qt.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CrossCursor)

    def set_roi(self, roi: Optional[ROI]) -> None:
        """Display ``roi`` programmatically (e.g. from object detection)."""
        self.clear_roi(emit=False)
        if roi is None or not roi.is_valid():
            return
        self._active_roi = roi
        self._roi_item = make_overlay_item(roi)
        self._roi_item.set_view_scale(self._zoom)
        self._scene.addItem(self._roi_item)

    def clear_roi(self, emit: bool = True) -> None:
        """Remove the active ROI overlay."""
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
        had_roi = self._active_roi is not None
        self._active_roi = None
        self._roi_points = []
        self._roi_drawing = False
        if emit and had_roi:
            self.roiCleared.emit()

    def _update_roi_pen(self) -> None:
        if self._roi_item is not None:
            self._roi_item.set_view_scale(self._zoom)

    def _finish_roi(self) -> None:
        """Commit the in-progress ROI and notify listeners."""
        if not self._roi_points or self._roi_mode is None:
            return
        roi = ROI(roi_type=self._roi_mode, points=list(self._roi_points))
        self._roi_drawing = False
        self._roi_points = []
        if not roi.is_valid():
            self.clear_roi(emit=False)
            return
        self._active_roi = roi
        if self._roi_item is not None:
            self._roi_item.update_roi(roi)
        self.roiCreated.emit(roi)

    def _preview_roi(self, points: List[Tuple[float, float]]) -> None:
        roi = ROI(roi_type=self._roi_mode or ROIType.RECTANGLE, points=points)
        if self._roi_item is None:
            self._roi_item = make_overlay_item(roi)
            self._scene.addItem(self._roi_item)
        else:
            self._roi_item.update_roi(roi)
        self._roi_item.set_view_scale(self._zoom)

    # --------------------------------------------------------------- crosshair
    def set_crosshair_enabled(self, enabled: bool) -> None:
        """Toggle the cursor-following crosshair."""
        self._crosshair_enabled = bool(enabled)
        self.viewport().update()

    @property
    def crosshair_enabled(self) -> bool:
        """Whether the crosshair overlay is active."""
        return self._crosshair_enabled

    # -------------------------------------------------------------- rendering
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """Paint the viewport background and an alpha checkerboard."""
        painter.fillRect(rect, QColor(Palette.VIEWPORT))
        if self._image is None or not self._image.has_alpha:
            return
        scene_rect = self._scene.sceneRect().intersected(rect)
        if scene_rect.isEmpty():
            return
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.fillRect(scene_rect, QColor("#4a4a4a"))
        cell = max(8.0, 8.0 / max(self._zoom, 1e-6))
        painter.setBrush(QBrush(QColor("#5c5c5c")))
        y = scene_rect.top() - (scene_rect.top() % (cell * 2))
        row = 0
        while y < scene_rect.bottom():
            x = scene_rect.left() - (scene_rect.left() % (cell * 2))
            col = 0
            while x < scene_rect.right():
                if (row + col) % 2 == 0:
                    painter.drawRect(QRectF(x, y, cell, cell))
                x += cell
                col += 1
            y += cell
            row += 1
        painter.restore()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Paint the crosshair overlay above the image."""
        if not self._crosshair_enabled or self._cursor_scene_pos is None:
            return
        pen = QPen(QColor(Palette.CROSSHAIR))
        pen.setWidthF(1.0 / max(self._zoom, 1e-6))
        painter.setPen(pen)
        pos = self._cursor_scene_pos
        painter.drawLine(QPointF(rect.left(), pos.y()), QPointF(rect.right(), pos.y()))
        painter.drawLine(QPointF(pos.x(), rect.top()), QPointF(pos.x(), rect.bottom()))

    # ----------------------------------------------------------------- events
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom under the cursor; Ctrl+wheel scrolls instead."""
        if not self.has_image:
            return
        if event.modifiers() & Qt.ControlModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        self.set_zoom(self._zoom * factor, anchor_under_mouse=True)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start panning or ROI drawing."""
        if not self.has_image:
            super().mousePressEvent(event)
            return

        start_pan = event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._space_held
        )
        if start_pan:
            self._panning = True
            self._pan_origin = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._roi_mode is not None:
            point = self._scene_point(event.pos())
            if self._roi_mode in (ROIType.RECTANGLE, ROIType.ELLIPSE, ROIType.FREEHAND):
                self._roi_drawing = True
                self._roi_points = [point]
                self._preview_roi([point, point])
            else:  # polygon: click to add a vertex, double-click/right-click to close
                if not self._roi_drawing:
                    self._roi_drawing = True
                    self._roi_points = [point]
                else:
                    self._roi_points.append(point)
                self._preview_roi(self._roi_points + [point])
            event.accept()
            return

        if event.button() == Qt.RightButton:
            # While a polygon is being drawn, right-click closes it. Otherwise
            # it opens the context menu, which is where most actions now live.
            if self._roi_mode == ROIType.POLYGON and self._roi_drawing:
                if len(self._roi_points) >= 3:
                    self._finish_roi()
                event.accept()
                return
            self.contextMenuRequested.emit(event.globalPos())
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Pan, extend an ROI, and emit the pixel readout."""
        if self._panning:
            delta = event.pos() - self._pan_origin
            self._pan_origin = event.pos()
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())
        self._cursor_scene_pos = scene_pos
        if self._crosshair_enabled:
            self.viewport().update()

        if self._roi_drawing and self._roi_mode is not None:
            point = self._scene_point(event.pos())
            if self._roi_mode in (ROIType.RECTANGLE, ROIType.ELLIPSE):
                self._preview_roi([self._roi_points[0], point])
            elif self._roi_mode == ROIType.FREEHAND:
                last = self._roi_points[-1]
                if abs(last[0] - point[0]) + abs(last[1] - point[1]) >= 1.0:
                    self._roi_points.append(point)
                self._preview_roi(self._roi_points)
            else:  # polygon rubber band
                self._preview_roi(self._roi_points + [point])

        self._emit_probe(scene_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Finish panning or close a drag-drawn ROI."""
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.viewport().setCursor(
                Qt.CrossCursor if self._roi_mode is not None else Qt.ArrowCursor
            )
            self._emit_view_changed()
            event.accept()
            return

        if (
            event.button() == Qt.LeftButton
            and self._roi_drawing
            and self._roi_mode in (ROIType.RECTANGLE, ROIType.ELLIPSE, ROIType.FREEHAND)
        ):
            point = self._scene_point(event.pos())
            if self._roi_mode in (ROIType.RECTANGLE, ROIType.ELLIPSE):
                self._roi_points = [self._roi_points[0], point]
            self._finish_roi()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Close a polygon ROI on double click."""
        if self._roi_mode == ROIType.POLYGON and self._roi_drawing:
            if len(self._roi_points) >= 3:
                self._finish_roi()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle view shortcuts (space to pan, R/F/1/2, Esc to cancel ROI)."""
        key = event.key()
        if key == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        if key == Qt.Key_Escape:
            if self._roi_drawing:
                self.clear_roi()
            else:
                self.set_roi_mode(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Release the space-pan modifier."""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self.viewport().setCursor(
                    Qt.CrossCursor if self._roi_mode is not None else Qt.ArrowCursor
                )
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Clear the readout when the pointer leaves the viewport."""
        self._cursor_scene_pos = None
        if self._crosshair_enabled:
            self.viewport().update()
        self.cursorLeft.emit()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        """Keep the image fitted while in fit mode.

        The refit changes the view as much as an explicit zoom does, so it must
        emit ``viewChanged`` too. A synced peer has ``_fit_mode`` False (see
        :meth:`apply_view_state`) and so never refits itself; without this
        signal the comparison panes silently drift apart whenever the window is
        resized or the Compare tab is shown for the first time.
        """
        super().resizeEvent(event)
        if self._fit_mode and self.has_image:
            rect = self._scene.sceneRect()
            if not rect.isEmpty():
                self.fitInView(rect, Qt.KeepAspectRatio)
                self._zoom = float(self.transform().m11())
                self._update_sampling_mode()
                self._update_roi_pen()
                self.zoomChanged.emit(self._zoom)
                self._emit_view_changed()

    # -------------------------------------------------------------- internals
    def _scene_point(self, viewport_pos: QPoint) -> Tuple[float, float]:
        """Map a viewport position to clamped image coordinates."""
        scene_pos = self.mapToScene(viewport_pos)
        if self._image is None:
            return float(scene_pos.x()), float(scene_pos.y())
        x = max(0.0, min(float(self._image.width), float(scene_pos.x())))
        y = max(0.0, min(float(self._image.height), float(scene_pos.y())))
        return x, y

    def _emit_probe(self, scene_pos: QPointF) -> None:
        """Sample the source array at ``scene_pos`` and emit the readout."""
        if self._image is None:
            return
        x = int(np.floor(scene_pos.x()))
        y = int(np.floor(scene_pos.y()))
        sample = sample_pixel(self._image, x, y)
        if sample is None:
            self.cursorLeft.emit()
            return
        components, gray, hsv = sample
        self.pixelProbed.emit(PixelProbe(x, y, components, gray, hsv))
