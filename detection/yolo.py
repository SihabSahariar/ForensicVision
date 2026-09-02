"""Optional YOLO object detection.

Detection exists to help an examiner *find* a region worth examining - a
vehicle, a person, a plate - not to make assertions about identity. Class
labels are the detector's opinion and are labelled as such throughout.

The model weights are downloaded by Ultralytics on first use, so detection is
gated behind an explicit user action and reports clearly when weights are
absent (S26).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.paths import weights_dir
from core.image_utils import ensure_uint8_rgb

logger = logging.getLogger(__name__)

__all__ = ["Detection", "ObjectDetector", "detector_available", "COCO_GROUPS"]

#: Coarse groupings used to filter the COCO class list to forensic relevance.
COCO_GROUPS: Dict[str, Tuple[str, ...]] = {
    "person": ("person",),
    "vehicle": ("car", "motorcycle", "bus", "truck", "bicycle", "train", "boat"),
    "readable": ("book", "laptop", "cell phone", "tv", "clock"),
}


@dataclass
class Detection:
    """One detected object."""

    label: str
    confidence: float
    box: Tuple[int, int, int, int]
    group: str = ""

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        """Box area in pixels."""
        return max(0, self.width) * max(0, self.height)

    def describe(self) -> str:
        """Return a one-line description for the results table."""
        return (
            f"{self.label} ({self.confidence * 100:.0f}%) "
            f"{self.width} x {self.height} px at ({self.box[0]}, {self.box[1]})"
        )

    def to_roi(self):
        """Return this detection as an :class:`~gui.roi_tools.ROI`."""
        from gui.roi_tools import ROI

        return ROI.from_box(
            self.box[0], self.box[1], self.width, self.height, label=self.label
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "label": self.label,
            "group": self.group,
            "confidence": round(self.confidence, 4),
            "box": list(self.box),
        }


def detector_available() -> bool:
    """Whether the ``ultralytics`` package is importable."""
    try:
        import ultralytics  # noqa: PLC0415, F401

        return True
    except Exception:
        return False


class ObjectDetector:
    """Wraps an Ultralytics YOLO model.

    The detector is entirely optional: :meth:`detect` reports the missing
    dependency rather than raising, and the GUI surfaces that message.
    """

    #: Default checkpoint; small enough to download quickly, accurate enough
    #: to locate people and vehicles in surveillance frames.
    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(self, model_name: str = "") -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    @property
    def model_path(self) -> Path:
        """Where the checkpoint is stored."""
        return weights_dir() / self._model_name

    @property
    def is_installed(self) -> bool:
        """Whether the checkpoint is already present locally."""
        return self.model_path.is_file()

    def load(self) -> None:
        """Load the detector, downloading the checkpoint if permitted.

        Raises:
            RuntimeError: ``ultralytics`` is not installed.
        """
        if self._model is not None:
            return
        if not detector_available():
            raise RuntimeError(
                "Object detection requires the 'ultralytics' package. "
                "Install it with: pip install ultralytics"
            )
        from ultralytics import YOLO  # noqa: PLC0415

        target = self.model_path if self.is_installed else self._model_name
        logger.info("Loading detector %s", target)
        self._model = YOLO(str(target))

        # Keep the checkpoint inside the application's weights folder so it is
        # visible in the Model Manager rather than hidden in a cache.
        if not self.is_installed:
            try:
                source = Path(getattr(self._model, "ckpt_path", "") or "")
                if source.is_file():
                    import shutil

                    weights_dir().mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, self.model_path)
            except Exception:  # pragma: no cover - best effort
                logger.debug("Could not relocate detector weights", exc_info=True)

    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
        max_detections: int = 100,
        groups: Optional[List[str]] = None,
    ) -> List[Detection]:
        """Detect objects in ``image``.

        Args:
            image: Frame to search.
            confidence: Minimum confidence to report.
            max_detections: Cap on returned boxes.
            groups: Restrict to these :data:`COCO_GROUPS` keys.

        Returns:
            Detections sorted by confidence, highest first.
        """
        self.load()
        rgb = ensure_uint8_rgb(image)[..., :3]

        results = self._model.predict(  # type: ignore[union-attr]
            source=rgb, conf=float(confidence), max_det=int(max_detections),
            verbose=False,
        )

        allowed: Optional[set] = None
        if groups:
            allowed = set()
            for group in groups:
                allowed.update(COCO_GROUPS.get(group, ()))

        detections: List[Detection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    x1, y1, x2, y2 = (
                        float(v) for v in box.xyxy[0].tolist()
                    )
                    class_index = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                except Exception:  # pragma: no cover - defensive
                    continue
                label = str(names.get(class_index, class_index))
                if allowed is not None and label not in allowed:
                    continue
                group = next(
                    (name for name, members in COCO_GROUPS.items() if label in members),
                    "other",
                )
                detections.append(
                    Detection(
                        label=label,
                        confidence=score,
                        box=(int(x1), int(y1), int(x2), int(y2)),
                        group=group,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info("Detected %d object(s)", len(detections))
        return detections

    def unload(self) -> None:
        """Release the detector."""
        self._model = None
