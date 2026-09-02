"""Restoration pipelines: ordered sequences of operations with provenance.

A pipeline is the unit the investigator reviews and approves before anything
runs. Each step records its own input and output digests, so the chain from
original evidence to final derivative is verifiable step by step rather than
only end to end.

Alpha handling: models operate on RGB only. The runner splits alpha off, runs
the colour pipeline, and rescales alpha to match any change in geometry using
nearest-neighbour sampling - alpha is a mask, and interpolating it would
fabricate partial transparency that was never in the source.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from app.constants import ModelKind, TaskType
from core.exceptions import ModelNotAvailableError, OperationCancelled, PipelineError
from core.image_io import ImageData
from forensic.hashing import HashSet, hash_array
from restoration.base import ModelInfo, RestorationModel
from restoration.registry import ModelRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineStep",
    "Pipeline",
    "StepResult",
    "PipelineResult",
    "PipelineRunner",
]

ProgressCallback = Callable[[int, str], None]
CancelCheck = Callable[[], bool]


@dataclass
class PipelineStep:
    """One operation in a pipeline.

    Attributes:
        model_name: Registry key of the model to run.
        parameters: Parameters passed to :meth:`RestorationModel.process`.
        enabled: Disabled steps are skipped but retained for the record.
        note: Investigator's justification for including this step.
    """

    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    note: str = ""

    def info(self) -> Optional[ModelInfo]:
        """Return the registered :class:`ModelInfo`, if the model exists."""
        return ModelRegistry.info(self.model_name)

    @property
    def display_name(self) -> str:
        """Human readable model name, falling back to the registry key."""
        info = self.info()
        return info.display_name if info else self.model_name

    @property
    def task(self) -> str:
        """The task this step performs."""
        info = self.info()
        return info.task if info else TaskType.GENERIC.value

    @property
    def may_synthesise(self) -> bool:
        """Whether this step can introduce content absent from the input."""
        info = self.info()
        return bool(info and info.may_synthesise)

    @property
    def scale(self) -> int:
        """Output scale factor of this step."""
        info = self.info()
        if info is None:
            return 1
        # A model may expose a user-selectable scale that differs from its
        # native one; the parameter wins because that is what will be produced.
        return int(self.parameters.get("scale", info.scale))

    def describe(self) -> str:
        """Return a compact one-line description including key parameters."""
        info = self.info()
        name = info.display_name if info else self.model_name
        if self.scale > 1:
            name = f"{name} x{self.scale}"
        interesting = {
            key: value
            for key, value in self.parameters.items()
            if key not in ("scale", "tile_size", "tile_overlap", "cancelled")
        }
        if interesting:
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(interesting.items()))
            return f"{name} ({rendered})"
        return name

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        info = self.info()
        return {
            "model": self.model_name,
            "display_name": self.display_name,
            "task": self.task,
            "kind": info.kind if info else "",
            "version": info.version if info else "",
            "parameters": dict(self.parameters),
            "enabled": self.enabled,
            "note": self.note,
            "may_synthesise": self.may_synthesise,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        """Rebuild a step from :meth:`to_dict` output."""
        return cls(
            model_name=data.get("model", ""),
            parameters=dict(data.get("parameters", {})),
            enabled=bool(data.get("enabled", True)),
            note=data.get("note", ""),
        )


@dataclass
class Pipeline:
    """An ordered, reviewable sequence of restoration steps."""

    steps: List[PipelineStep] = field(default_factory=list)
    name: str = "Custom pipeline"
    rationale: str = ""

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def enabled_steps(self) -> List[PipelineStep]:
        """Steps that will actually execute."""
        return [step for step in self.steps if step.enabled]

    @property
    def total_scale(self) -> int:
        """Product of every enabled step's scale factor."""
        total = 1
        for step in self.enabled_steps:
            total *= max(1, step.scale)
        return total

    @property
    def may_synthesise(self) -> bool:
        """Whether any enabled step can synthesise content."""
        return any(step.may_synthesise for step in self.enabled_steps)

    def add(self, step: PipelineStep) -> "Pipeline":
        """Append ``step`` and return ``self`` for chaining."""
        self.steps.append(step)
        return self

    def move(self, index: int, delta: int) -> bool:
        """Move the step at ``index`` by ``delta`` positions."""
        target = index + delta
        if not (0 <= index < len(self.steps) and 0 <= target < len(self.steps)):
            return False
        self.steps[index], self.steps[target] = self.steps[target], self.steps[index]
        return True

    def remove(self, index: int) -> bool:
        """Remove the step at ``index``."""
        if 0 <= index < len(self.steps):
            del self.steps[index]
            return True
        return False

    def validate(self) -> List[str]:
        """Return a list of problems that would prevent or degrade execution.

        An empty list means the pipeline is ready to run.
        """
        issues: List[str] = []
        if not self.enabled_steps:
            issues.append("The pipeline contains no enabled steps.")

        for index, step in enumerate(self.enabled_steps, start=1):
            info = step.info()
            if info is None:
                issues.append(f"Step {index}: model '{step.model_name}' is not registered.")
                continue
            model = ModelRegistry.try_get(step.model_name)
            if model is None:
                issues.append(f"Step {index}: model '{step.model_name}' could not be created.")
                continue
            state = model.availability()
            if not state.ok:
                issues.append(f"Step {index} ({info.display_name}): {state.reason}")

        # Ordering advice: super-resolution should come last, because running it
        # first forces every later model to work on interpolated pixels and
        # multiplies their cost.
        tasks = [step.task for step in self.enabled_steps]
        if TaskType.SUPER_RESOLUTION.value in tasks:
            sr_index = tasks.index(TaskType.SUPER_RESOLUTION.value)
            after = tasks[sr_index + 1:]
            cleanup = {
                TaskType.JPEG_ARTIFACT.value,
                TaskType.DENOISE.value,
                TaskType.DEBLUR.value,
            }
            if cleanup.intersection(after):
                issues.append(
                    "Super-resolution is scheduled before a cleanup step. "
                    "Artefact removal and denoising work better at native "
                    "resolution; consider moving super-resolution to the end."
                )

        if self.total_scale > 16:
            issues.append(
                f"Combined scale factor is x{self.total_scale}, which will "
                "produce a very large output and is rarely evidentially useful."
            )
        return issues

    def describe(self) -> str:
        """Return a multi-line human readable summary."""
        if not self.steps:
            return "(empty pipeline)"
        lines = []
        for index, step in enumerate(self.enabled_steps, start=1):
            marker = " [may synthesise]" if step.may_synthesise else ""
            lines.append(f"{index}. {step.describe()}{marker}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "rationale": self.rationale,
            "total_scale": self.total_scale,
            "may_synthesise": self.may_synthesise,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Rebuild a pipeline from :meth:`to_dict` output."""
        return cls(
            steps=[PipelineStep.from_dict(s) for s in data.get("steps", [])],
            name=data.get("name", "Custom pipeline"),
            rationale=data.get("rationale", ""),
        )

    def copy(self) -> "Pipeline":
        """Return a deep copy."""
        return Pipeline.from_dict(self.to_dict())


@dataclass
class StepResult:
    """The record of one executed step."""

    index: int
    step: PipelineStep
    model_info: Optional[ModelInfo]
    input_hashes: HashSet
    output_hashes: HashSet
    input_shape: tuple
    output_shape: tuple
    duration_s: float
    device: str
    started_at: str
    status: str = "ok"
    message: str = ""

    @property
    def input_dimensions(self) -> str:
        """``"W x H"`` of the step input."""
        return f"{self.input_shape[1]} x {self.input_shape[0]}"

    @property
    def output_dimensions(self) -> str:
        """``"W x H"`` of the step output."""
        return f"{self.output_shape[1]} x {self.output_shape[0]}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "index": self.index,
            "step": self.step.to_dict(),
            "model": self.model_info.to_dict() if self.model_info else {},
            "input_sha256": self.input_hashes.sha256,
            "output_sha256": self.output_hashes.sha256,
            "input_dimensions": self.input_dimensions,
            "output_dimensions": self.output_dimensions,
            "duration_s": round(self.duration_s, 4),
            "device": self.device,
            "started_at": self.started_at,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class PipelineResult:
    """The outcome of running a pipeline."""

    image: ImageData
    steps: List[StepResult] = field(default_factory=list)
    run_id: str = ""
    pipeline: Optional[Pipeline] = None
    total_duration_s: float = 0.0
    device: str = "cpu"
    cancelled: bool = False

    @property
    def succeeded(self) -> bool:
        """Whether every step completed without error."""
        return not self.cancelled and all(s.status == "ok" for s in self.steps)

    @property
    def may_synthesise(self) -> bool:
        """Whether any executed step could synthesise content."""
        return any(
            s.model_info.may_synthesise for s in self.steps if s.model_info is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline.to_dict() if self.pipeline else {},
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_s": round(self.total_duration_s, 4),
            "device": self.device,
            "cancelled": self.cancelled,
            "may_synthesise": self.may_synthesise,
        }


class PipelineRunner:
    """Executes pipelines, tracking hashes and timings for every step."""

    def __init__(
        self,
        device: str = "auto",
        fp16: bool = True,
        unload_between_steps: bool = False,
    ) -> None:
        """Create a runner.

        Args:
            device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
            fp16: Request half precision where a model supports it.
            unload_between_steps: Free each model's memory after its step. Use
                on memory-constrained GPUs; costs reload time per step.
        """
        self._device = device
        self._fp16 = fp16
        self._unload_between_steps = unload_between_steps

    def run(
        self,
        image: ImageData,
        pipeline: Pipeline,
        progress: Optional[ProgressCallback] = None,
        cancelled: Optional[CancelCheck] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """Execute ``pipeline`` against ``image``.

        Args:
            image: Source image; never modified.
            pipeline: The approved pipeline.
            progress: Optional ``(percent, message)`` callback.
            cancelled: Optional predicate polled between and within steps.
            roi_mask: Optional ``HxW`` mask limiting where results are applied.
                Steps that change geometry cannot be masked, and the runner
                raises rather than silently ignoring the restriction.

        Returns:
            A :class:`PipelineResult` carrying the derivative and the record.

        Raises:
            PipelineError: The pipeline is invalid or a step failed.
            OperationCancelled: The operation was cancelled.
        """
        steps = pipeline.enabled_steps
        if not steps:
            raise PipelineError("The pipeline contains no enabled steps.")

        from core.device import resolve_torch_device

        resolved_device = self._resolve_device_label()
        run_id = str(uuid.uuid4())
        started_total = time.perf_counter()

        colour, alpha = image.split_alpha()
        working = self._to_float_rgb(colour, image.max_value)
        original_shape = working.shape

        if roi_mask is not None and pipeline.total_scale != 1:
            raise PipelineError(
                "A region-limited run cannot include steps that change image "
                "geometry. Either remove the super-resolution step or run the "
                "pipeline on the full frame."
            )

        results: List[StepResult] = []
        total_steps = len(steps)

        for index, step in enumerate(steps):
            if cancelled is not None and cancelled():
                raise OperationCancelled("Pipeline cancelled by the user")

            base_percent = int(index * 100 / total_steps)
            span = int(100 / total_steps)

            def step_progress(percent: int, message: str) -> None:
                if progress is not None:
                    overall = base_percent + int(percent * span / 100)
                    progress(
                        min(99, overall),
                        f"Step {index + 1}/{total_steps} - {message}",
                    )

            if progress is not None:
                progress(base_percent, f"Step {index + 1}/{total_steps} - {step.describe()}")

            model = self._acquire(step)
            input_hashes = hash_array(working)
            input_shape = working.shape
            step_started = time.perf_counter()
            started_at = datetime.now(timezone.utc).isoformat()

            try:
                parameters = dict(step.parameters)
                parameters["cancelled"] = cancelled
                output = model.process(working, progress=step_progress, **parameters)
            except OperationCancelled:
                raise
            except Exception as exc:
                logger.exception("Step %d (%s) failed", index + 1, step.model_name)
                raise PipelineError(
                    f"Step {index + 1} ({step.display_name}) failed: {exc}"
                ) from exc
            finally:
                if self._unload_between_steps:
                    model.unload()

            duration = time.perf_counter() - step_started
            results.append(
                StepResult(
                    index=index,
                    step=step,
                    model_info=step.info(),
                    input_hashes=input_hashes,
                    output_hashes=hash_array(output),
                    input_shape=input_shape,
                    output_shape=output.shape,
                    duration_s=duration,
                    device=model.device,
                    started_at=started_at,
                )
            )
            logger.info(
                "Step %d/%d %s: %s -> %s in %.2fs",
                index + 1, total_steps, step.display_name,
                f"{input_shape[1]}x{input_shape[0]}",
                f"{output.shape[1]}x{output.shape[0]}",
                duration,
            )
            working = output

        if roi_mask is not None:
            working = self._apply_mask(
                original=self._to_float_rgb(colour, image.max_value),
                restored=working,
                mask=roi_mask,
            )

        result_image = self._compose(image, working, alpha)

        if progress is not None:
            progress(100, "Pipeline complete")

        return PipelineResult(
            image=result_image,
            steps=results,
            run_id=run_id,
            pipeline=pipeline,
            total_duration_s=time.perf_counter() - started_total,
            device=resolved_device,
        )

    # ---------------------------------------------------------------- helpers
    def _resolve_device_label(self) -> str:
        """Return the device string that models will be loaded onto."""
        if self._device == "cpu":
            return "cpu"
        from core.device import get_device_report

        report = get_device_report()
        return "cuda" if report.has_gpu else "cpu"

    def _acquire(self, step: PipelineStep) -> RestorationModel:
        """Fetch and load the model for ``step``."""
        model = ModelRegistry.try_get(step.model_name)
        if model is None:
            raise PipelineError(f"Model '{step.model_name}' is not registered.")
        state = model.availability()
        if not state.ok:
            raise ModelNotAvailableError(
                f"{step.display_name} cannot run: {state.reason}"
            )
        model.load(device=self._resolve_device_label(), fp16=self._fp16)
        return model

    @staticmethod
    def _to_float_rgb(colour: np.ndarray, max_value: float) -> np.ndarray:
        """Normalise a colour plane to ``HxWx3`` ``float32`` in ``[0, 1]``."""
        array = colour.astype(np.float32) / float(max_value)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return np.clip(array, 0.0, 1.0)

    @staticmethod
    def _apply_mask(
        original: np.ndarray, restored: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Blend ``restored`` into ``original`` only where ``mask`` is set.

        The mask edge is feathered by two pixels so the boundary of a
        region-limited enhancement does not itself become a visible artefact
        that could be mistaken for image content.
        """
        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(
                mask, (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        weight = (mask.astype(np.float32) / 255.0)
        weight = cv2.GaussianBlur(weight, (0, 0), 1.5)[..., None]
        return np.clip(original * (1.0 - weight) + restored * weight, 0.0, 1.0)

    @staticmethod
    def _compose(
        source: ImageData, colour: np.ndarray, alpha: Optional[np.ndarray]
    ) -> ImageData:
        """Rebuild an :class:`ImageData` from processed colour and source alpha.

        The output keeps the source bit depth: a 16-bit original yields a
        16-bit derivative, so a restoration pass never silently costs precision.
        """
        max_value = source.max_value
        if source.dtype == np.uint8:
            pixels = (np.clip(colour, 0, 1) * 255.0).round().astype(np.uint8)
        elif source.dtype == np.uint16:
            pixels = (np.clip(colour, 0, 1) * 65535.0).round().astype(np.uint16)
        else:
            pixels = np.clip(colour, 0.0, 1.0).astype(np.float32)

        if alpha is not None:
            if alpha.shape[:2] != pixels.shape[:2]:
                # Alpha is a mask: resample it without interpolation so no
                # partial transparency is invented at the edges.
                alpha = cv2.resize(
                    alpha, (pixels.shape[1], pixels.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            alpha = alpha.astype(pixels.dtype)
            pixels = np.dstack([pixels, alpha])

        return ImageData(
            pixels=np.ascontiguousarray(pixels),
            source_path=source.source_path,
            source_format=source.source_format,
            icc_profile=source.icc_profile,
            extra=dict(source.extra),
        )
