"""Exception hierarchy for ForensicVision.

Every failure that can reasonably be surfaced to an investigator derives from
:class:`ForensicVisionError` so the GUI can present a single, consistent error
dialog while still distinguishing recoverable conditions.
"""

from __future__ import annotations

__all__ = [
    "ForensicVisionError",
    "ImageLoadError",
    "ImageSaveError",
    "UnsupportedFormatError",
    "CaseError",
    "EvidenceError",
    "IntegrityError",
    "SafeModeViolation",
    "ModelError",
    "ModelNotAvailableError",
    "WeightsMissingError",
    "DependencyMissingError",
    "InferenceError",
    "OutOfMemoryError",
    "PipelineError",
    "ReportError",
    "OperationCancelled",
]


class ForensicVisionError(Exception):
    """Base class for all application-specific errors."""


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #

class ImageLoadError(ForensicVisionError):
    """Raised when an image file cannot be decoded."""


class ImageSaveError(ForensicVisionError):
    """Raised when an image cannot be written to disk."""


class UnsupportedFormatError(ImageLoadError):
    """Raised for file extensions outside the supported set."""


# --------------------------------------------------------------------------- #
# Case / evidence
# --------------------------------------------------------------------------- #

class CaseError(ForensicVisionError):
    """Raised for case creation, opening or layout problems."""


class EvidenceError(ForensicVisionError):
    """Raised when evidence import or lookup fails."""


class IntegrityError(ForensicVisionError):
    """Raised when a stored hash does not match the file on disk."""


class SafeModeViolation(ForensicVisionError):
    """Raised when an operation is blocked by Forensic Safe Mode."""


# --------------------------------------------------------------------------- #
# Models / inference
# --------------------------------------------------------------------------- #

class ModelError(ForensicVisionError):
    """Base class for restoration-model failures."""


class ModelNotAvailableError(ModelError):
    """Raised when a requested model is not usable on this installation."""


class WeightsMissingError(ModelNotAvailableError):
    """Raised when a neural model has no weights installed.

    The GUI turns this into an actionable *Install model* prompt rather than
    producing any substitute output.
    """

    def __init__(self, model_name: str, message: str = "") -> None:
        self.model_name = model_name
        super().__init__(
            message or f"Weights for '{model_name}' are not installed."
        )


class DependencyMissingError(ModelNotAvailableError):
    """Raised when an optional Python package required by a model is absent."""

    def __init__(self, package: str, message: str = "") -> None:
        self.package = package
        super().__init__(message or f"Required package '{package}' is not installed.")


class InferenceError(ModelError):
    """Raised when a model fails during ``process``."""


class OutOfMemoryError(InferenceError):
    """Raised when the compute device runs out of memory."""


# --------------------------------------------------------------------------- #
# Pipelines / reporting
# --------------------------------------------------------------------------- #

class PipelineError(ForensicVisionError):
    """Raised when a restoration pipeline is invalid or fails mid-run."""


class ReportError(ForensicVisionError):
    """Raised when report generation fails."""


class OperationCancelled(ForensicVisionError):
    """Raised inside workers when the user cancels a long-running operation."""
