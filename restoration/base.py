"""Restoration model interface.

Every restoration operation - learned or classical - implements
:class:`RestorationModel`. The interface is deliberately narrow so the GUI, the
pipeline runner and any future CLI or video front-end all drive models the same
way.

Contract
--------
* :meth:`RestorationModel.process` receives and returns an ``HxWx3`` ``float32``
  RGB array in ``[0, 1]``. Alpha is handled by the pipeline, not by models.
* :meth:`RestorationModel.availability` must report honestly whether the model
  can run *right now*. A model that cannot run raises rather than returning a
  substitute result - never a fabricated output.
* :attr:`ModelInfo.may_synthesise` marks operations capable of inventing image
  content. The GUI warns before running these and the provenance record and
  report both carry the flag.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.constants import ModelKind, ModelStatus, TaskType
from core.exceptions import ModelNotAvailableError

logger = logging.getLogger(__name__)

__all__ = [
    "ParamSpec",
    "WeightSpec",
    "ModelInfo",
    "Availability",
    "RestorationModel",
    "ProgressReporter",
]

#: ``(percent, message)`` progress callback used by long-running operations.
ProgressReporter = Callable[[int, str], None]


@dataclass(frozen=True)
class ParamSpec:
    """Describes one user-adjustable parameter so the GUI can build controls.

    Attributes:
        name: Keyword passed to :meth:`RestorationModel.process`.
        label: Human readable control label.
        kind: ``"float"``, ``"int"``, ``"bool"`` or ``"choice"``.
        default: Default value.
        minimum: Lower bound for numeric parameters.
        maximum: Upper bound for numeric parameters.
        step: Increment for numeric parameters.
        choices: Allowed ``(value, label)`` pairs for ``"choice"`` parameters.
        help_text: Tooltip shown next to the control.
    """

    name: str
    label: str
    kind: str = "float"
    default: Any = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    choices: Sequence[Tuple[Any, str]] = field(default_factory=tuple)
    help_text: str = ""

    def clamp(self, value: Any) -> Any:
        """Coerce ``value`` into this parameter's declared domain."""
        if self.kind == "bool":
            return bool(value)
        if self.kind == "choice":
            allowed = [choice[0] for choice in self.choices]
            return value if value in allowed else self.default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        number = max(self.minimum, min(self.maximum, number))
        return int(round(number)) if self.kind == "int" else number


@dataclass(frozen=True)
class WeightSpec:
    """A weight file a neural model needs.

    Attributes:
        filename: Name the file is stored under in the weights directory.
        url: Canonical download location.
        sha256: Expected digest; empty when upstream publishes none.
        size_bytes: Approximate download size, for the Model Manager.
        license_name: Licence covering the *weights* (often stricter than the
            code licence).
        source: Human readable provenance of the weights.
        required: Whether the model is unusable without this file.
    """

    filename: str
    url: str = ""
    sha256: str = ""
    size_bytes: int = 0
    license_name: str = ""
    source: str = ""
    required: bool = True

    def size_human(self) -> str:
        """Return the download size in MiB, or ``"unknown"``."""
        if self.size_bytes <= 0:
            return "unknown"
        return f"{self.size_bytes / (1024 * 1024):.1f} MiB"


@dataclass(frozen=True)
class ModelInfo:
    """Static description of a restoration model.

    This is what the Model Manager renders and what the provenance record and
    PDF report cite, so licence and source fields are mandatory in practice for
    anything with downloadable weights.
    """

    name: str
    display_name: str
    task: str
    kind: str = ModelKind.CLASSICAL.value
    version: str = "1.0"
    description: str = ""
    #: Short note on what the operation actually does to the pixels.
    method: str = ""
    license_name: str = ""
    repository: str = ""
    paper: str = ""
    authors: str = ""
    weights: Sequence[WeightSpec] = field(default_factory=tuple)
    parameters: Sequence[ParamSpec] = field(default_factory=tuple)
    #: Fixed output scale factor; 1 for same-size operations.
    scale: int = 1
    supports_fp16: bool = False
    supports_tiling: bool = False
    requires_packages: Sequence[str] = field(default_factory=tuple)
    #: Whether the operation can introduce content absent from the input.
    may_synthesise: bool = False
    #: Multiple of which the input dimensions must be padded before inference.
    size_multiple: int = 1
    notes: str = ""

    @property
    def is_neural(self) -> bool:
        """Whether this model uses learned weights."""
        return self.kind == ModelKind.NEURAL.value

    @property
    def task_label(self) -> str:
        """Human readable task name."""
        from app.constants import TASK_LABELS  # local import avoids a cycle

        return TASK_LABELS.get(self.task, self.task)

    def parameter(self, name: str) -> Optional[ParamSpec]:
        """Return the :class:`ParamSpec` named ``name``."""
        for spec in self.parameters:
            if spec.name == name:
                return spec
        return None

    def default_parameters(self) -> Dict[str, Any]:
        """Return a mapping of every parameter's default value."""
        return {spec.name: spec.default for spec in self.parameters}

    def sanitise(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp ``params`` to the declared domains, filling in defaults."""
        result = self.default_parameters()
        for spec in self.parameters:
            if spec.name in params:
                result[spec.name] = spec.clamp(params[spec.name])
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description for reports and provenance."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "task": self.task,
            "task_label": self.task_label,
            "kind": self.kind,
            "version": self.version,
            "description": self.description,
            "method": self.method,
            "license": self.license_name,
            "repository": self.repository,
            "paper": self.paper,
            "authors": self.authors,
            "scale": self.scale,
            "may_synthesise": self.may_synthesise,
            "weights": [
                {
                    "filename": w.filename,
                    "sha256": w.sha256,
                    "license": w.license_name,
                    "source": w.source,
                    "size_bytes": w.size_bytes,
                }
                for w in self.weights
            ],
        }


@dataclass(frozen=True)
class Availability:
    """Whether a model can run, and why not when it cannot."""

    status: str
    reason: str = ""
    missing_weights: Sequence[str] = field(default_factory=tuple)
    missing_packages: Sequence[str] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether the model is ready to run."""
        return self.status == ModelStatus.INSTALLED.value

    @property
    def label(self) -> str:
        """Short status label for the Model Manager table."""
        from app.constants import MODEL_STATUS_LABELS  # local import

        return MODEL_STATUS_LABELS.get(self.status, self.status)


class RestorationModel(ABC):
    """Base class for every restoration operation.

    Subclasses implement :meth:`_process` and, for neural models, :meth:`_load`
    and :meth:`_unload`. The public :meth:`process` handles availability
    checking, parameter sanitising and input validation.
    """

    #: Populated by each subclass.
    info: ModelInfo

    def __init__(self, weights_dir: Optional[Path] = None) -> None:
        from app.paths import weights_dir as default_weights_dir

        self._weights_dir = Path(weights_dir) if weights_dir else default_weights_dir()
        self._loaded = False
        self._device = "cpu"
        self._fp16 = False

    # ------------------------------------------------------------- properties
    @property
    def name(self) -> str:
        """Registry key for this model."""
        return self.info.name

    @property
    def version(self) -> str:
        """Model version string."""
        return self.info.version

    @property
    def loaded(self) -> bool:
        """Whether weights are currently resident."""
        return self._loaded

    @property
    def device(self) -> str:
        """Device the model is currently loaded on."""
        return self._device

    @property
    def weights_dir(self) -> Path:
        """Directory searched for this model's weight files."""
        return self._weights_dir

    def weight_path(self, spec: WeightSpec) -> Path:
        """Return the on-disk location for ``spec``."""
        return self._weights_dir / spec.filename

    # ----------------------------------------------------------- availability
    def availability(self) -> Availability:
        """Report whether this model can run right now.

        Returns:
            An :class:`Availability` describing the current state.
        """
        missing_packages = [
            package
            for package in self.info.requires_packages
            if not _package_available(package)
        ]
        if missing_packages:
            return Availability(
                status=ModelStatus.MISSING_DEPENDENCY.value,
                reason=(
                    "Required Python package(s) not installed: "
                    + ", ".join(missing_packages)
                ),
                missing_packages=tuple(missing_packages),
            )

        missing_weights = [
            spec.filename
            for spec in self.info.weights
            if spec.required and not self.weight_path(spec).is_file()
        ]
        if missing_weights:
            return Availability(
                status=ModelStatus.MISSING_WEIGHTS.value,
                reason=(
                    f"Weight file(s) not installed: {', '.join(missing_weights)}. "
                    "Install them from Tools > Model Manager."
                ),
                missing_weights=tuple(missing_weights),
            )

        return Availability(status=ModelStatus.INSTALLED.value)

    def require_available(self) -> None:
        """Raise :class:`ModelNotAvailableError` when the model cannot run."""
        state = self.availability()
        if not state.ok:
            raise ModelNotAvailableError(state.reason)

    # ------------------------------------------------------------- life cycle
    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        """Prepare the model for inference.

        Args:
            device: ``"cpu"`` or ``"cuda[:n]"``.
            fp16: Request half precision where supported.

        Raises:
            ModelNotAvailableError: Weights or dependencies are missing.
        """
        self.require_available()
        if self._loaded and self._device == device and self._fp16 == fp16:
            return
        if self._loaded:
            self.unload()
        self._device = device
        self._fp16 = bool(fp16 and self.info.supports_fp16 and device.startswith("cuda"))
        self._load()
        self._loaded = True
        logger.info(
            "Loaded %s v%s on %s%s",
            self.info.display_name,
            self.info.version,
            device,
            " (fp16)" if self._fp16 else "",
        )

    def unload(self) -> None:
        """Release weights and device memory."""
        if not self._loaded:
            return
        try:
            self._unload()
        finally:
            self._loaded = False
            logger.debug("Unloaded %s", self.info.display_name)

    # ---------------------------------------------------------------- process
    def process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        """Run the operation on ``image``.

        Args:
            image: ``HxWx3`` ``float32`` RGB array in ``[0, 1]``.
            progress: Optional ``(percent, message)`` callback.
            **params: Model parameters; unknown keys are ignored and known ones
                are clamped to their declared domain.

        Returns:
            An ``HxWx3`` ``float32`` RGB array in ``[0, 1]``. Spatial size is
            multiplied by :attr:`ModelInfo.scale`.

        Raises:
            ModelNotAvailableError: The model cannot run in this installation.
            ValueError: ``image`` has the wrong shape or dtype.
        """
        self.require_available()
        array = _validate_input(image)
        clean = self.info.sanitise(params)
        if not self._loaded:
            self.load(self._device, self._fp16)
        result = self._process(array, progress=progress, **clean)
        return _validate_output(result)

    def metadata(self) -> Dict[str, Any]:
        """Return a description of this model instance for provenance records."""
        state = self.availability()
        payload = self.info.to_dict()
        payload.update(
            {
                "status": state.status,
                "status_reason": state.reason,
                "loaded": self._loaded,
                "device": self._device,
                "fp16": self._fp16,
                "weights_dir": str(self._weights_dir),
                "installed_weights": {
                    spec.filename: str(self.weight_path(spec))
                    for spec in self.info.weights
                    if self.weight_path(spec).is_file()
                },
            }
        )
        return payload

    def weights_digest(self) -> str:
        """Return the SHA-256 declared for the primary weight file, if any."""
        for spec in self.info.weights:
            if spec.required and spec.sha256:
                return spec.sha256
        return ""

    # ------------------------------------------------------------- subclasses
    def _load(self) -> None:
        """Subclass hook: bring weights into memory. Default is a no-op."""

    def _unload(self) -> None:
        """Subclass hook: release resources. Default is a no-op."""

    @abstractmethod
    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        """Subclass hook performing the actual transformation."""
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.info.name!r} loaded={self._loaded}>"


def _package_available(package: str) -> bool:
    """Return ``True`` when ``package`` can be imported."""
    import importlib.util

    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):  # pragma: no cover - odd import states
        return False


def _validate_input(image: np.ndarray) -> np.ndarray:
    """Validate and normalise a model input array."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Model input must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Model input must be HxWx3 RGB; got shape {image.shape}"
        )
    if image.size == 0:
        raise ValueError("Model input is empty")
    array = image
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.ascontiguousarray(np.clip(array, 0.0, 1.0))


def _validate_output(image: np.ndarray) -> np.ndarray:
    """Validate a model output array."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Model output must be an HxWx3 RGB array")
    return np.ascontiguousarray(np.clip(image.astype(np.float32), 0.0, 1.0))
