"""Restoration pipeline worker.

Runs a pipeline off the GUI thread and, when a case is supplied, writes the
derivative, its provenance sidecar and the database records as part of the same
unit of work - so a derivative can never exist on disk without its record.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.constants import LOSSLESS_EXPORT_FORMATS
from core.case_manager import CaseManager
from core.image_io import ImageData, save_image
from database.models import Derivative, Evidence
from forensic.hashing import HashSet, hash_array, hash_file
from forensic.provenance import ProvenanceRecord, write_sidecar
from restoration.pipeline import Pipeline, PipelineResult, PipelineRunner
from workers.base import BaseWorker

logger = logging.getLogger(__name__)

__all__ = [
    "RestorationWorker",
    "RestorationOutcome",
    "persist_restoration",
    "operation_label",
]


class RestorationOutcome:
    """What a completed restoration produced.

    Attributes:
        result: The raw pipeline result including the derivative image.
        derivative_row: The database row, when the run was case-backed.
        output_path: Where the derivative was written, if it was.
        provenance: The provenance record attached to the derivative.
    """

    __slots__ = ("result", "derivative_row", "output_path", "provenance")

    def __init__(
        self,
        result: PipelineResult,
        derivative_row: Optional[Derivative] = None,
        output_path: Optional[Path] = None,
        provenance: Optional[ProvenanceRecord] = None,
    ) -> None:
        self.result = result
        self.derivative_row = derivative_row
        self.output_path = output_path
        self.provenance = provenance

    @property
    def image(self) -> ImageData:
        """The restored image."""
        return self.result.image


class RestorationWorker(BaseWorker):
    """Executes a pipeline and optionally persists the derivative."""

    description = "Running restoration pipeline"

    def __init__(
        self,
        image: ImageData,
        pipeline: Pipeline,
        case: Optional[CaseManager] = None,
        evidence: Optional[Evidence] = None,
        parent_derivative: Optional[Derivative] = None,
        roi_mask: Optional[np.ndarray] = None,
        roi_descriptor: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        fp16: bool = True,
        persist: bool = True,
        output_format: str = ".png",
    ) -> None:
        """Create the worker.

        Args:
            image: Source image.
            pipeline: The approved pipeline.
            case: Case to record the derivative in; ``None`` runs transiently.
            evidence: Evidence row the derivative descends from.
            parent_derivative: Derivative this one was chained from.
            roi_mask: Optional mask limiting where results are applied.
            roi_descriptor: Serialisable ROI description for provenance.
            device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
            fp16: Use half precision where supported.
            persist: Write the derivative and register it.
            output_format: Container extension for the derivative.
        """
        super().__init__()
        self._image = image
        self._pipeline = pipeline
        self._case = case
        self._evidence = evidence
        self._parent = parent_derivative
        self._roi_mask = roi_mask
        self._roi_descriptor = roi_descriptor
        self._device = device
        self._fp16 = fp16
        self._persist = persist and case is not None and evidence is not None
        self._output_format = output_format

    def execute(self) -> RestorationOutcome:
        """Run the pipeline, then persist the derivative when requested."""
        runner = PipelineRunner(device=self._device, fp16=self._fp16)
        self.report_status(f"Running {len(self._pipeline.enabled_steps)} step(s)...")

        result = runner.run(
            self._image,
            self._pipeline,
            progress=self.report,
            cancelled=self.is_cancelled,
            roi_mask=self._roi_mask,
        )

        if not self._persist:
            return RestorationOutcome(result=result)

        self.report(96, "Writing derivative")
        return self._persist_result(result)

    # ---------------------------------------------------------------- persist
    def _persist_result(self, result: PipelineResult) -> RestorationOutcome:
        """Write the derivative, its sidecar and every database record."""
        assert self._case is not None and self._evidence is not None
        return persist_restoration(
            result=result,
            source_image=self._image,
            pipeline=self._pipeline,
            case=self._case,
            evidence=self._evidence,
            parent_derivative=self._parent,
            roi_descriptor=self._roi_descriptor,
            output_format=self._output_format,
        )

    # ---------------------------------------------------------------- helpers
    def _operation_label(self) -> str:
        """Return a short operation name for the derivative row."""
        return operation_label(self._pipeline)


def _suffix_for(pipeline: Pipeline, roi: Optional[Dict[str, Any]]) -> str:
    """Build a filename tag summarising ``pipeline``."""
    parts = [step.model_name for step in pipeline.enabled_steps][:3]
    tag = "_".join(parts) if parts else "derivative"
    return f"roi_{tag}" if roi else tag


def operation_label(pipeline: Pipeline) -> str:
    """Return a short operation name for a derivative row."""
    steps = pipeline.enabled_steps
    if len(steps) == 1:
        return steps[0].task
    return "pipeline"


def persist_restoration(
    result: PipelineResult,
    source_image: ImageData,
    pipeline: Pipeline,
    case: CaseManager,
    evidence: Evidence,
    parent_derivative: Optional[Derivative] = None,
    roi_descriptor: Optional[Dict[str, Any]] = None,
    output_format: str = ".png",
) -> RestorationOutcome:
    """Write a derivative and register every associated record.

    Shared by :class:`RestorationWorker` and the batch runner so that a
    derivative on disk always has a matching sidecar, derivative row, per-step
    history and audit entry - there is only one code path that can create one.

    Args:
        result: The completed pipeline result.
        source_image: The image the pipeline was run against.
        pipeline: The pipeline that produced ``result``.
        case: Case to register the derivative in.
        evidence: Evidence the derivative descends from.
        parent_derivative: Derivative this one was chained from, if any.
        roi_descriptor: Serialisable ROI description for the provenance record.
        output_format: Container extension for the derivative.

    Returns:
        A :class:`RestorationOutcome`.
    """
    suffix = _suffix_for(pipeline, roi_descriptor)
    extension = output_format
    if extension not in LOSSLESS_EXPORT_FORMATS:
        logger.warning(
            "Derivative format %s is lossy; falling back to PNG so the recorded "
            "hash describes the produced pixels exactly",
            extension,
        )
        extension = ".png"

    output_path = case.derivative_path(evidence, suffix, extension)
    save_image(result.image, output_path)

    file_hashes = hash_file(output_path)
    if parent_derivative is not None:
        input_hashes = HashSet(
            sha256=parent_derivative.sha256,
            sha512=parent_derivative.sha512,
            md5=parent_derivative.md5,
            size_bytes=parent_derivative.size_bytes,
        )
    else:
        input_hashes = HashSet(
            sha256=evidence.sha256,
            sha512=evidence.sha512,
            md5=evidence.md5,
            size_bytes=evidence.size_bytes,
        )

    from core.device import get_device_report

    device_report = get_device_report()
    gpu_name = ""
    primary = device_report.primary_gpu()
    if result.device.startswith("cuda") and primary is not None:
        gpu_name = primary.name

    last_step = result.steps[-1] if result.steps else None
    model_chain = " -> ".join(s.step.display_name for s in result.steps)

    provenance = ProvenanceRecord.build(
        input_hashes=input_hashes,
        output_hashes=file_hashes,
        operation=operation_label(pipeline),
        model=model_chain,
        model_version=(
            last_step.model_info.version
            if last_step is not None and last_step.model_info is not None
            else ""
        ),
        model_kind=result.model_kind,
        parameters={
            f"step_{i + 1}_{s.step.model_name}": s.step.parameters
            for i, s in enumerate(result.steps)
        },
        device="CUDA" if result.device.startswith("cuda") else "CPU",
        gpu=gpu_name,
        duration_s=result.total_duration_s,
        input_dimensions=f"{source_image.width} x {source_image.height}",
        output_dimensions=f"{result.image.width} x {result.image.height}",
        may_synthesise=result.may_synthesise,
        pipeline=[s.to_dict() for s in result.steps],
        roi=roi_descriptor,
        case_id=case.case_id,
        evidence_filename=evidence.original_filename,
    )
    write_sidecar(provenance, output_path)

    repo = case.repository
    derivative = repo.add_derivative(
        case_pk=case.case_pk,
        evidence_id=evidence.id,
        parent_derivative_id=parent_derivative.id if parent_derivative else None,
        path=str(output_path),
        label=output_path.stem,
        sha256=file_hashes.sha256,
        sha512=file_hashes.sha512,
        md5=file_hashes.md5,
        size_bytes=file_hashes.size_bytes,
        width=result.image.width,
        height=result.image.height,
        channels=result.image.channels,
        bit_depth=result.image.bit_depth,
        operation=operation_label(pipeline),
        model_name=model_chain,
        model_version="",
        model_kind=result.model_kind,
        parameters={
            f"step_{i + 1}": s.step.to_dict() for i, s in enumerate(result.steps)
        },
        provenance=provenance.to_dict(),
        pipeline=pipeline.to_dict(),
    )

    for step_result in result.steps:
        repo.add_step(
            case_pk=case.case_pk,
            evidence_id=evidence.id,
            derivative_id=derivative.id,
            run_id=result.run_id,
            sequence=step_result.index,
            operation=step_result.step.task,
            model_name=step_result.step.model_name,
            model_version=(
                step_result.model_info.version if step_result.model_info else ""
            ),
            model_kind=(
                step_result.model_info.kind if step_result.model_info else ""
            ),
            parameters=step_result.step.parameters,
            input_sha256=step_result.input_hashes.sha256,
            output_sha256=step_result.output_hashes.sha256,
            input_size=step_result.input_dimensions,
            output_size=step_result.output_dimensions,
            device=step_result.device,
            duration_s=step_result.duration_s,
            status=step_result.status,
            message=step_result.message,
        )

    case.audit(
        action="derivative.create",
        target=output_path.name,
        detail=(
            f"pipeline={model_chain} | "
            f"input_sha256={input_hashes.sha256} | "
            f"output_sha256={file_hashes.sha256} | "
            f"synthesis={result.may_synthesise}"
        ),
    )

    logger.info(
        "Derivative registered: %s (sha256 %s)", output_path.name, file_hashes.short()
    )
    return RestorationOutcome(
        result=result,
        derivative_row=derivative,
        output_path=output_path,
        provenance=provenance,
    )
