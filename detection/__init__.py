"""Optional detection used to locate regions of interest.

Two detectors, neither of which identifies anyone:

* :mod:`detection.yolo` - general object detection (person, vehicle, ...) via
  Ultralytics. Class labels are the detector's estimate, not identifications.
* :mod:`detection.face` - face detection and canonical alignment via OpenCV's
  YuNet. Its landmarks exist solely to compute the warp that face restoration
  requires; they are not used for recognition or matching.
"""

from detection.face import (
    FFHQ_TEMPLATE_512,
    AlignedFace,
    FaceAligner,
    FaceDetection,
    YuNetDetector,
    YUNET_WEIGHT,
)
from detection.face import detector_available as face_detector_available
from detection.yolo import COCO_GROUPS, Detection, ObjectDetector, detector_available

__all__ = [
    "Detection",
    "ObjectDetector",
    "detector_available",
    "COCO_GROUPS",
    "FaceDetection",
    "AlignedFace",
    "FaceAligner",
    "YuNetDetector",
    "YUNET_WEIGHT",
    "FFHQ_TEMPLATE_512",
    "face_detector_available",
]
