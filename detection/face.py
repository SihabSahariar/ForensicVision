"""Face detection, alignment and paste-back.

Face restoration networks are trained on faces warped into a canonical frame:
FFHQ-aligned, 512x512, eyes and mouth corners at fixed positions. Feeding them
an arbitrary crop produces plausible but geometrically wrong output, so the
alignment stage is not optional - it is the difference between a usable result
and a confident fabrication.

Detection uses **YuNet**, which is part of OpenCV's own API
(``cv2.FaceDetectorYN``) and emits exactly the five landmarks the canonical
template needs: right eye, left eye, nose tip, right mouth corner, left mouth
corner, in that order. The model is a 230 KiB ONNX file from the official
OpenCV Zoo.

Nothing here identifies anyone. Detection locates a region; the landmarks
exist solely to compute a warp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.paths import weights_dir
from core.image_utils import ensure_uint8_rgb

logger = logging.getLogger(__name__)

__all__ = [
    "FaceDetection",
    "AlignedFace",
    "FaceAligner",
    "YuNetDetector",
    "YUNET_WEIGHT",
    "FFHQ_TEMPLATE_512",
    "detector_available",
]

#: Canonical five-point template for a 512x512 FFHQ-aligned face, in the order
#: YuNet reports landmarks: right eye, left eye, nose tip, right mouth corner,
#: left mouth corner. "Right" is the subject's right, which appears on the
#: left of the image - hence the ascending x for entries 0 and 1.
FFHQ_TEMPLATE_512: np.ndarray = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def _yunet_weight_spec():
    """Return the :class:`~restoration.base.WeightSpec` for the YuNet model."""
    from restoration.base import WeightSpec

    return WeightSpec(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        size_bytes=232_589,
        license_name="MIT (OpenCV Zoo)",
        source="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    )


#: Weight specification, so the Model Manager installs it like any other.
YUNET_WEIGHT = _yunet_weight_spec()


def detector_available() -> Tuple[bool, str]:
    """Report whether face detection can run.

    Returns:
        ``(available, reason)``. ``reason`` is empty when available.
    """
    if not hasattr(cv2, "FaceDetectorYN"):
        return False, (
            f"OpenCV {cv2.__version__} does not provide cv2.FaceDetectorYN. "
            "Upgrade to opencv-python >= 4.5.4."
        )
    path = weights_dir() / YUNET_WEIGHT.filename
    if not path.is_file():
        return False, (
            f"The YuNet face detector ({YUNET_WEIGHT.filename}, 227 KiB) is not "
            "installed. Install it from Tools > Model Manager."
        )
    return True, ""


@dataclass
class FaceDetection:
    """One detected face with its five alignment landmarks."""

    box: Tuple[int, int, int, int]
    landmarks: np.ndarray
    confidence: float

    @property
    def width(self) -> int:
        """Detection box width in pixels."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        """Detection box height in pixels."""
        return self.box[3] - self.box[1]

    @property
    def inter_ocular_distance(self) -> float:
        """Distance between the eye landmarks, in source pixels.

        This is the honest measure of how much facial information the source
        actually contains. Below roughly 30 px, restoration output is dominated
        by the learned prior rather than by the evidence.
        """
        return float(np.linalg.norm(self.landmarks[1] - self.landmarks[0]))

    def describe(self) -> str:
        """Return a one-line summary for the results table."""
        return (
            f"face ({self.confidence * 100:.0f}%) {self.width}x{self.height} px, "
            f"inter-ocular {self.inter_ocular_distance:.0f} px"
        )

    def to_roi(self):
        """Return the detection box as an :class:`~gui.roi_tools.ROI`."""
        from gui.roi_tools import ROI

        return ROI.from_box(
            self.box[0], self.box[1], self.width, self.height, label="Face"
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "box": list(self.box),
            "landmarks": self.landmarks.tolist(),
            "confidence": round(self.confidence, 4),
            "inter_ocular_distance": round(self.inter_ocular_distance, 2),
        }


@dataclass
class AlignedFace:
    """A face warped into the canonical frame, plus the inverse transform."""

    image: np.ndarray
    affine: np.ndarray
    detection: FaceDetection
    face_size: int = 512

    @property
    def inverse_affine(self) -> np.ndarray:
        """The transform mapping the canonical frame back to source pixels."""
        return cv2.invertAffineTransform(self.affine)


class YuNetDetector:
    """Face detector wrapping ``cv2.FaceDetectorYN``."""

    def __init__(
        self,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 500,
    ) -> None:
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._detector = None

    @property
    def model_path(self) -> Path:
        """Location of the ONNX model."""
        return weights_dir() / YUNET_WEIGHT.filename

    def load(self) -> None:
        """Create the underlying detector.

        Raises:
            RuntimeError: OpenCV lacks the API, or the model is not installed.
        """
        if self._detector is not None:
            return
        available, reason = detector_available()
        if not available:
            raise RuntimeError(reason)
        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320),
            self._score_threshold, self._nms_threshold, self._top_k,
        )
        logger.info("YuNet face detector loaded from %s", self.model_path)

    #: Smallest box edge reported by default.
    #:
    #: Deliberately low. Small faces are the normal case in surveillance
    #: stills - a 128 px frame puts a face at roughly 23 px - and a size filter
    #: generous enough to look tidy would silently hide exactly the detections
    #: an examiner is looking for. The honest behaviour is to report the face
    #: and let the inter-ocular measurement say how little information it
    #: carries, not to pretend there is no face there.
    DEFAULT_MIN_SIZE: int = 12

    def detect(
        self, image: np.ndarray, min_size: Optional[int] = None
    ) -> List[FaceDetection]:
        """Detect faces in ``image``.

        The frame is upscaled before detection when it is small - the detector
        has a minimum practical face size, and refusing to look is worse than
        looking at an interpolated copy. Coordinates are mapped back to source
        pixels, so every returned landmark is in the original coordinate
        system.

        Args:
            image: RGB frame of any supported dtype.
            min_size: Discard detections whose box edge is below this; defaults
                to :attr:`DEFAULT_MIN_SIZE`.

        Returns:
            Detections ordered by confidence, highest first.
        """
        min_size = self.DEFAULT_MIN_SIZE if min_size is None else int(min_size)
        self.load()
        rgb = ensure_uint8_rgb(image)[..., :3]
        bgr = rgb[..., ::-1].copy()

        # YuNet is trained around 320x320; very small frames benefit from a
        # detection-only enlargement.
        height, width = bgr.shape[:2]
        scale = 1.0
        longest = max(height, width)
        if longest < 640:
            scale = min(4.0, 640.0 / max(1, longest))
            bgr = cv2.resize(
                bgr, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, raw = self._detector.detect(bgr)
        if raw is None:
            return []

        detections: List[FaceDetection] = []
        for row in raw:
            x, y, box_w, box_h = row[0:4] / scale
            landmarks = row[4:14].reshape(5, 2) / scale
            confidence = float(row[14])
            if box_w < min_size or box_h < min_size:
                continue
            detections.append(
                FaceDetection(
                    box=(
                        int(round(x)), int(round(y)),
                        int(round(x + box_w)), int(round(y + box_h)),
                    ),
                    landmarks=landmarks.astype(np.float32),
                    confidence=confidence,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.info("Detected %d face(s)", len(detections))
        return detections

    def unload(self) -> None:
        """Release the detector."""
        self._detector = None


class FaceAligner:
    """Warps faces into the canonical frame and pastes results back."""

    def __init__(self, face_size: int = 512) -> None:
        self._face_size = int(face_size)
        self._template = FFHQ_TEMPLATE_512 * (self._face_size / 512.0)

    @property
    def face_size(self) -> int:
        """Edge length of the canonical frame."""
        return self._face_size

    def align(
        self, image: np.ndarray, detection: FaceDetection
    ) -> Optional[AlignedFace]:
        """Warp one detected face into the canonical frame.

        Args:
            image: Source RGB frame.
            detection: The face to align.

        Returns:
            An :class:`AlignedFace`, or ``None`` when a stable transform could
            not be estimated.
        """
        rgb = ensure_uint8_rgb(image)[..., :3]
        affine, _ = cv2.estimateAffinePartial2D(
            detection.landmarks, self._template, method=cv2.LMEDS
        )
        if affine is None:
            logger.warning("Could not estimate an alignment transform")
            return None

        warped = cv2.warpAffine(
            rgb, affine, (self._face_size, self._face_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(135, 133, 132),
        )
        return AlignedFace(
            image=warped, affine=affine, detection=detection,
            face_size=self._face_size,
        )

    def paste_back(
        self,
        canvas: np.ndarray,
        restored_face: np.ndarray,
        aligned: AlignedFace,
        upscale: float = 1.0,
    ) -> np.ndarray:
        """Blend a restored face back into the frame.

        The seam is feathered proportionally to the face area, so the boundary
        of the restored region does not itself become an artefact that could be
        mistaken for image content.

        Args:
            canvas: Destination frame, RGB uint8.
            restored_face: The restored canonical-frame face.
            aligned: The alignment that produced it.
            upscale: Scale factor between the source frame and ``canvas``.

        Returns:
            A new frame with the face blended in. ``canvas`` is not modified.
        """
        output = ensure_uint8_rgb(canvas)[..., :3].astype(np.float32)
        height, width = output.shape[:2]

        inverse = aligned.inverse_affine.copy()
        inverse[:, 0:2] *= upscale
        inverse[:, 2] *= upscale

        face = ensure_uint8_rgb(restored_face)[..., :3].astype(np.float32)
        if face.shape[0] != aligned.face_size:
            face = cv2.resize(
                face, (aligned.face_size, aligned.face_size),
                interpolation=cv2.INTER_LINEAR,
            )

        warped_face = cv2.warpAffine(face, inverse, (width, height))

        mask = np.ones((aligned.face_size, aligned.face_size), dtype=np.float32)
        warped_mask = cv2.warpAffine(mask, inverse, (width, height))

        # Erode away the interpolation fringe at the warped border.
        erosion = max(2, int(2 * upscale))
        warped_mask = cv2.erode(
            warped_mask, np.ones((erosion, erosion), np.uint8)
        )

        area = float(warped_mask.sum())
        if area <= 0:
            return output.astype(np.uint8)

        edge = max(2, int(area ** 0.5) // 20)
        warped_mask = cv2.erode(
            warped_mask, np.ones((edge * 2, edge * 2), np.uint8)
        )
        blur = edge * 2 + 1
        soft_mask = cv2.GaussianBlur(warped_mask, (blur, blur), 0)[..., None]
        soft_mask = np.clip(soft_mask, 0.0, 1.0)

        blended = soft_mask * warped_face + (1.0 - soft_mask) * output
        return np.clip(blended, 0, 255).astype(np.uint8)
