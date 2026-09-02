"""Region-of-interest model and interactive graphics items.

An ROI never mutates the underlying evidence. It is purely a coordinate
description plus a rasterisation helper used to extract or mask sub-regions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
)

from gui.theme import Palette

logger = logging.getLogger(__name__)

__all__ = ["ROIType", "ROI", "ROIOverlayItem", "make_overlay_item"]


class ROIType(str, Enum):
    """Supported region-of-interest geometries."""

    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "freehand"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


ROI_TYPE_LABELS = {
    ROIType.RECTANGLE.value: "Rectangle",
    ROIType.ELLIPSE.value: "Ellipse",
    ROIType.POLYGON.value: "Polygon",
    ROIType.FREEHAND.value: "Freehand",
}


@dataclass
class ROI:
    """A region of interest in *image* pixel coordinates.

    Attributes:
        roi_type: Geometry kind.
        points: For rectangle/ellipse, two opposite corners; for polygon and
            freehand, the ordered vertex list.
        label: Investigator-supplied name, e.g. ``"License plate"``.
    """

    roi_type: ROIType
    points: List[Tuple[float, float]] = field(default_factory=list)
    label: str = ""

    # -- geometry ----------------------------------------------------------- #
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Return the integer bounding box as ``(x, y, width, height)``."""
        if not self.points:
            return 0, 0, 0, 0
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x0, y0 = int(np.floor(min(xs))), int(np.floor(min(ys)))
        x1, y1 = int(np.ceil(max(xs))), int(np.ceil(max(ys)))
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    def clipped_box(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Return the bounding box clamped to an image of ``width`` x ``height``."""
        x, y, w, h = self.bounding_box()
        x0 = max(0, min(x, width))
        y0 = max(0, min(y, height))
        x1 = max(0, min(x + w, width))
        y1 = max(0, min(y + h, height))
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    def is_valid(self, min_size: int = 2) -> bool:
        """Whether the ROI encloses a usable area."""
        _, _, w, h = self.bounding_box()
        if self.roi_type in (ROIType.POLYGON, ROIType.FREEHAND):
            return len(self.points) >= 3 and w >= min_size and h >= min_size
        return w >= min_size and h >= min_size

    @property
    def area(self) -> int:
        """Bounding-box area in pixels."""
        _, _, w, h = self.bounding_box()
        return w * h

    # -- rasterisation ------------------------------------------------------ #
    def to_mask(self, width: int, height: int) -> np.ndarray:
        """Rasterise the ROI into a ``uint8`` mask (0 or 255).

        Args:
            width: Mask width, normally the image width.
            height: Mask height, normally the image height.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if not self.points:
            return mask

        if self.roi_type == ROIType.RECTANGLE:
            x, y, w, h = self.clipped_box(width, height)
            if w > 0 and h > 0:
                mask[y : y + h, x : x + w] = 255
        elif self.roi_type == ROIType.ELLIPSE:
            x, y, w, h = self.bounding_box()
            centre = (int(x + w / 2), int(y + h / 2))
            axes = (max(1, int(w / 2)), max(1, int(h / 2)))
            cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, thickness=-1)
        else:
            pts = np.array([[int(round(px)), int(round(py))] for px, py in self.points],
                           dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def crop(self, array: np.ndarray) -> np.ndarray:
        """Return the ROI's bounding-box crop of ``array`` (a copy)."""
        height, width = array.shape[:2]
        x, y, w, h = self.clipped_box(width, height)
        if w <= 0 or h <= 0:
            return array[0:0, 0:0].copy()
        return array[y : y + h, x : x + w].copy()

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "type": self.roi_type.value,
            "points": [[float(px), float(py)] for px, py in self.points],
            "label": self.label,
            "bbox": list(self.bounding_box()),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ROI":
        """Rebuild an ROI from :meth:`to_dict` output."""
        return cls(
            roi_type=ROIType(data.get("type", ROIType.RECTANGLE.value)),
            points=[(float(p[0]), float(p[1])) for p in data.get("points", [])],
            label=data.get("label", ""),
        )

    @classmethod
    def from_box(cls, x: float, y: float, w: float, h: float, label: str = "") -> "ROI":
        """Convenience constructor for an axis-aligned rectangle."""
        return cls(
            roi_type=ROIType.RECTANGLE,
            points=[(x, y), (x + w, y + h)],
            label=label,
        )

    def describe(self) -> str:
        """Return a one-line summary for status bars and history entries."""
        x, y, w, h = self.bounding_box()
        name = self.label or ROI_TYPE_LABELS.get(self.roi_type.value, "ROI")
        return f"{name}: {w} x {h} px at ({x}, {y})"


# --------------------------------------------------------------------------- #
# Graphics items
# --------------------------------------------------------------------------- #

class ROIOverlayItem(QGraphicsPathItem):
    """Renders an :class:`ROI` on top of the image at constant screen width.

    The pen width is recomputed from the view scale so the outline stays a
    hairline at 800% zoom instead of swallowing the pixels under inspection.
    """

    def __init__(self, roi: ROI, parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._roi = roi
        self._view_scale = 1.0
        self.setZValue(100)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setBrush(QBrush(QColor(Palette.ROI).lighter(100).darker(100)))
        brush_colour = QColor(Palette.ROI)
        brush_colour.setAlpha(28)
        self.setBrush(QBrush(brush_colour))
        self._apply_pen()
        self.update_roi(roi)

    # -- public API --------------------------------------------------------- #
    @property
    def roi(self) -> ROI:
        """The ROI currently displayed."""
        return self._roi

    def update_roi(self, roi: ROI) -> None:
        """Replace the displayed geometry."""
        self._roi = roi
        self.setPath(build_path(roi))

    def set_view_scale(self, scale: float) -> None:
        """Inform the item of the current view zoom so it can thin its pen."""
        self._view_scale = max(scale, 1e-6)
        self._apply_pen()

    # -- internals ---------------------------------------------------------- #
    def _apply_pen(self) -> None:
        pen = QPen(QColor(Palette.ROI))
        pen.setWidthF(1.6 / self._view_scale)
        pen.setCosmetic(False)
        pen.setJoinStyle(Qt.MiterJoin)
        self.setPen(pen)


def build_path(roi: ROI) -> QPainterPath:
    """Convert an :class:`ROI` into a :class:`QPainterPath`."""
    path = QPainterPath()
    if not roi.points:
        return path

    if roi.roi_type == ROIType.RECTANGLE:
        x, y, w, h = roi.bounding_box()
        path.addRect(QRectF(x, y, w, h))
    elif roi.roi_type == ROIType.ELLIPSE:
        x, y, w, h = roi.bounding_box()
        path.addEllipse(QRectF(x, y, w, h))
    else:
        first = roi.points[0]
        path.moveTo(QPointF(first[0], first[1]))
        for px, py in roi.points[1:]:
            path.lineTo(QPointF(px, py))
        if len(roi.points) >= 3:
            path.closeSubpath()
    return path


def make_overlay_item(roi: ROI) -> ROIOverlayItem:
    """Create an overlay item for ``roi``.

    Rectangles and ellipses could use the dedicated Qt item classes, but a
    single path-based item keeps pen scaling and hit testing uniform.
    """
    return ROIOverlayItem(roi)


def rect_from_points(start: Sequence[float], end: Sequence[float]) -> ROI:
    """Build a rectangle ROI from two drag endpoints."""
    return ROI(
        roi_type=ROIType.RECTANGLE,
        points=[(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))],
    )


# Keep the unused-but-documented Qt item classes importable for future
# specialised handles (resize grips) without re-importing at call sites.
__all__ += ["QGraphicsRectItem", "QGraphicsEllipseItem", "build_path", "rect_from_points",
            "ROI_TYPE_LABELS"]
