"""Static constants shared across the ForensicVision application.

This module deliberately avoids importing PyQt5 or any heavy third-party
dependency so that it can be consumed by headless components (CLI tools, batch
runners, unit tests) without pulling in a GUI stack.
"""

from __future__ import annotations

from enum import Enum

# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #

#: Extensions accepted when importing evidence.
SUPPORTED_IMAGE_EXTENSIONS: tuple = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

#: Qt file-dialog filter built from :data:`SUPPORTED_IMAGE_EXTENSIONS`.
IMAGE_FILE_FILTER: str = (
    "Images ("
    + " ".join("*" + ext for ext in SUPPORTED_IMAGE_EXTENSIONS)
    + ");;JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;"
    "BMP (*.bmp);;WebP (*.webp);;All files (*)"
)

#: Formats that keep pixel data bit-exact when written.
LOSSLESS_EXPORT_FORMATS: tuple = (".png", ".tif", ".tiff", ".bmp")

# --------------------------------------------------------------------------- #
# Case layout
# --------------------------------------------------------------------------- #

CASE_SUBDIRS: tuple = (
    "evidence/original",
    "derivatives",
    "analysis",
    "reports",
    "metadata",
    "logs",
)

CASE_ID_PREFIX: str = "CASE-"
CASE_DB_FILENAME: str = "case.db"
CASE_MANIFEST_FILENAME: str = "case.json"

# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

#: Algorithms computed for every evidence item and every derivative.
HASH_ALGORITHMS: tuple = ("sha256", "sha512", "md5")

#: The algorithm treated as authoritative for integrity checks. MD5 is retained
#: only for cross-referencing with legacy tooling and must never be relied upon.
PRIMARY_HASH_ALGORITHM: str = "sha256"

#: Chunk size used when streaming files through hash functions.
HASH_CHUNK_SIZE: int = 1024 * 1024


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

class DegradationKey(str, Enum):
    """Canonical keys returned by the degradation analyzer."""

    BLUR = "blur"
    MOTION_BLUR = "motion_blur"
    NOISE = "noise"
    JPEG = "jpeg"
    LOW_RESOLUTION = "low_resolution"
    UNDEREXPOSURE = "underexposure"
    OVEREXPOSURE = "overexposure"
    LOW_CONTRAST = "low_contrast"
    HAZE = "haze"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Human-readable labels for the analysis panel.
DEGRADATION_LABELS: dict = {
    DegradationKey.BLUR.value: "Blur",
    DegradationKey.MOTION_BLUR.value: "Motion Blur",
    DegradationKey.NOISE.value: "Noise",
    DegradationKey.JPEG.value: "JPEG Artifacts",
    DegradationKey.LOW_RESOLUTION.value: "Low Resolution",
    DegradationKey.UNDEREXPOSURE.value: "Underexposure",
    DegradationKey.OVEREXPOSURE.value: "Overexposure",
    DegradationKey.LOW_CONTRAST.value: "Low Contrast",
    DegradationKey.HAZE.value: "Haze",
}

#: Order used when rendering the analysis dock.
DEGRADATION_ORDER: tuple = tuple(DEGRADATION_LABELS.keys())

#: Threshold above which a degradation is considered actionable by the
#: auto-restoration engine.
DEGRADATION_ACTION_THRESHOLD: float = 0.45

#: Mandatory qualifier attached to every analysis result surfaced in the UI or
#: in generated reports. Required by the forensic honesty policy.
HEURISTIC_DISCLAIMER: str = (
    "Heuristic indicator - derived from classical image statistics, not from a "
    "validated classification model. Values are relative severity estimates on "
    "a 0-100 scale and must be interpreted by a qualified examiner."
)


# --------------------------------------------------------------------------- #
# Restoration
# --------------------------------------------------------------------------- #

class TaskType(str, Enum):
    """Restoration task categories used for model discovery and routing."""

    SUPER_RESOLUTION = "super_resolution"
    DEBLUR = "deblur"
    DENOISE = "denoise"
    JPEG_ARTIFACT = "jpeg_artifact"
    FACE_RESTORATION = "face_restoration"
    INPAINTING = "inpainting"
    DEHAZE = "dehaze"
    EXPOSURE = "exposure"
    CONTRAST = "contrast"
    SHARPEN = "sharpen"
    GENERIC = "generic"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


TASK_LABELS: dict = {
    TaskType.SUPER_RESOLUTION.value: "Super Resolution",
    TaskType.DEBLUR.value: "Deblur",
    TaskType.DENOISE.value: "Denoise",
    TaskType.JPEG_ARTIFACT.value: "JPEG Artifact Removal",
    TaskType.FACE_RESTORATION.value: "Face Restoration",
    TaskType.INPAINTING.value: "Inpainting",
    TaskType.DEHAZE.value: "Dehaze",
    TaskType.EXPOSURE.value: "Exposure Correction",
    TaskType.CONTRAST.value: "Contrast Enhancement",
    TaskType.SHARPEN.value: "Sharpening",
    TaskType.GENERIC.value: "General Restoration",
}

#: Order in which tasks are presented in the restoration panel.
TASK_ORDER: tuple = (
    TaskType.JPEG_ARTIFACT.value,
    TaskType.DENOISE.value,
    TaskType.DEBLUR.value,
    TaskType.SUPER_RESOLUTION.value,
    TaskType.FACE_RESTORATION.value,
    TaskType.DEHAZE.value,
    TaskType.EXPOSURE.value,
    TaskType.CONTRAST.value,
    TaskType.SHARPEN.value,
    TaskType.INPAINTING.value,
    TaskType.GENERIC.value,
)


class ModelKind(str, Enum):
    """Distinguishes learned models from deterministic classical algorithms.

    This separation is a forensic requirement: an examiner must always be able
    to tell whether an operation could have *synthesised* image content.
    """

    #: Deep-learning model with trained weights. May hallucinate detail.
    NEURAL = "neural"
    #: Deterministic signal-processing algorithm. Cannot invent structures that
    #: are not derivable from the input samples.
    CLASSICAL = "classical"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


MODEL_KIND_LABELS: dict = {
    ModelKind.NEURAL.value: "Neural (learned prior)",
    ModelKind.CLASSICAL.value: "Classical (deterministic DSP)",
}


class ModelStatus(str, Enum):
    """Installation status of a restoration model."""

    #: Ready to run right now.
    INSTALLED = "installed"
    #: Adapter exists, weights are absent - offer an explicit download.
    MISSING_WEIGHTS = "missing_weights"
    #: Adapter exists, a required Python package is absent.
    MISSING_DEPENDENCY = "missing_dependency"
    #: Adapter is a documented stub; upstream integration is not implemented.
    NOT_INTEGRATED = "not_integrated"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


MODEL_STATUS_LABELS: dict = {
    ModelStatus.INSTALLED.value: "Installed",
    ModelStatus.MISSING_WEIGHTS.value: "Weights missing",
    ModelStatus.MISSING_DEPENDENCY.value: "Dependency missing",
    ModelStatus.NOT_INTEGRATED.value: "Not integrated",
}


#: Warning shown before any operation that can synthesise image content.
SYNTHESIS_WARNING: str = (
    "This operation uses a generative or learned prior. It may introduce or "
    "synthesise structures - including facial features, characters and edges - "
    "that are not present in the source evidence. The output is a derivative "
    "visualisation and is not proof of the original scene content."
)

#: Legally-oriented disclaimer embedded in every generated report (spec S33).
FORENSIC_REPORT_DISCLAIMER: str = (
    "Algorithmic image enhancement modifies image data. AI-based restoration "
    "may infer or synthesize structures that are not directly represented in "
    "the source image. Enhanced imagery is a derivative representation and "
    "should not automatically be interpreted as an exact recovery of "
    "information absent from the original evidence."
)

#: Disclaimer attached to every OCR result.
OCR_DISCLAIMER: str = (
    "Machine-generated OCR interpretation. Character recognition performed on "
    "an algorithmically enhanced image reflects the enhanced derivative, not "
    "the original evidence, and must not be treated as proof of the true "
    "content of a degraded region."
)


# --------------------------------------------------------------------------- #
# Processing / devices
# --------------------------------------------------------------------------- #

DEFAULT_TILE_SIZE: int = 512
DEFAULT_TILE_OVERLAP: int = 32
MIN_TILE_SIZE: int = 64

#: Pixel count above which tiled inference is used unconditionally.
TILING_PIXEL_THRESHOLD: int = 1024 * 1024


class DeviceKind(str, Enum):
    """Compute backends supported by the restoration engine."""

    CPU = "cpu"
    CUDA = "cuda"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

ZOOM_PRESETS: tuple = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MIN_ZOOM: float = 0.02
MAX_ZOOM: float = 64.0
ZOOM_STEP: float = 1.25

SETTINGS_GEOMETRY: str = "MainWindow/geometry"
SETTINGS_STATE: str = "MainWindow/windowState"
SETTINGS_LAST_CASE: str = "Session/lastCasePath"
SETTINGS_SAFE_MODE: str = "Forensic/safeMode"
SETTINGS_DEVICE: str = "Processing/device"
SETTINGS_FP16: str = "Processing/fp16"
SETTINGS_TILE_SIZE: str = "Processing/tileSize"

STATUS_MESSAGE_TIMEOUT_MS: int = 6000
