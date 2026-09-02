"""Provenance records for derivative images.

Every derivative written by ForensicVision is accompanied by a JSON sidecar and
an equivalent database row describing exactly how it was produced. The record
is designed so that an independent examiner can reproduce the transformation
or, at minimum, evaluate what the transformation could have introduced.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.constants import FORENSIC_REPORT_DISCLAIMER, ModelKind
from app.version import APP_NAME, APP_VERSION
from forensic.hashing import HashSet

logger = logging.getLogger(__name__)

__all__ = ["ProvenanceRecord", "environment_snapshot", "write_sidecar"]


def environment_snapshot() -> Dict[str, Any]:
    """Capture the software/hardware environment for reproducibility.

    All heavy imports are guarded so this works on an ML-free installation.
    """
    snapshot: Dict[str, Any] = {
        "application": APP_NAME,
        "application_version": APP_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }

    try:
        import numpy  # noqa: PLC0415

        snapshot["numpy_version"] = numpy.__version__
    except Exception:  # pragma: no cover
        snapshot["numpy_version"] = ""

    try:
        import cv2  # noqa: PLC0415

        snapshot["opencv_version"] = cv2.__version__
    except Exception:  # pragma: no cover
        snapshot["opencv_version"] = ""

    try:
        import torch  # noqa: PLC0415

        snapshot["pytorch_version"] = torch.__version__
        snapshot["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        if snapshot["cuda_available"]:
            snapshot["gpu"] = torch.cuda.get_device_name(0)
            cudnn = torch.backends.cudnn.version()
            snapshot["cudnn_version"] = str(cudnn) if cudnn else ""
    except Exception:  # pragma: no cover - torch optional
        snapshot["pytorch_version"] = ""
        snapshot["cuda_version"] = ""
        snapshot["cuda_available"] = False

    return snapshot


@dataclass
class ProvenanceRecord:
    """A complete description of one derivative's origin.

    Attributes:
        input_sha256: Digest of the exact bytes that were read as input.
        output_sha256: Digest of the written derivative.
        operation: Task category, e.g. ``"deblur"``.
        model: Model or algorithm identifier.
        model_version: Model version string.
        model_kind: ``"neural"`` or ``"classical"``.
        parameters: Every user- and engine-supplied parameter.
        device: ``"CUDA"`` or ``"CPU"``.
        timestamp: UTC time the operation completed.
        pipeline: Ordered list of step descriptors when part of a pipeline.
        may_synthesise: Whether the operation can invent image content.
        roi: ROI descriptor when the operation was region-limited.
    """

    input_sha256: str = ""
    output_sha256: str = ""
    input_sha512: str = ""
    output_sha512: str = ""
    input_md5: str = ""
    output_md5: str = ""
    operation: str = ""
    model: str = ""
    model_version: str = ""
    model_kind: str = ModelKind.CLASSICAL.value
    model_license: str = ""
    model_source: str = ""
    weights_sha256: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    device: str = "CPU"
    gpu: str = ""
    duration_s: float = 0.0
    input_dimensions: str = ""
    output_dimensions: str = ""
    timestamp: str = ""
    pipeline: List[Dict[str, Any]] = field(default_factory=list)
    may_synthesise: bool = False
    roi: Optional[Dict[str, Any]] = None
    environment: Dict[str, Any] = field(default_factory=environment_snapshot)
    case_id: str = ""
    evidence_filename: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------- convenience
    @classmethod
    def build(
        cls,
        *,
        input_hashes: HashSet,
        output_hashes: HashSet,
        operation: str,
        model: str,
        model_version: str = "",
        model_kind: str = ModelKind.CLASSICAL.value,
        parameters: Optional[Dict[str, Any]] = None,
        device: str = "CPU",
        gpu: str = "",
        duration_s: float = 0.0,
        input_dimensions: str = "",
        output_dimensions: str = "",
        may_synthesise: bool = False,
        pipeline: Optional[List[Dict[str, Any]]] = None,
        roi: Optional[Dict[str, Any]] = None,
        case_id: str = "",
        evidence_filename: str = "",
        model_license: str = "",
        model_source: str = "",
        weights_sha256: str = "",
    ) -> "ProvenanceRecord":
        """Construct a record from hash sets and operation metadata."""
        return cls(
            input_sha256=input_hashes.sha256,
            output_sha256=output_hashes.sha256,
            input_sha512=input_hashes.sha512,
            output_sha512=output_hashes.sha512,
            input_md5=input_hashes.md5,
            output_md5=output_hashes.md5,
            operation=operation,
            model=model,
            model_version=model_version,
            model_kind=model_kind,
            model_license=model_license,
            model_source=model_source,
            weights_sha256=weights_sha256,
            parameters=dict(parameters or {}),
            device=device,
            gpu=gpu,
            duration_s=round(float(duration_s), 4),
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
            may_synthesise=bool(may_synthesise),
            pipeline=list(pipeline or []),
            roi=roi,
            case_id=case_id,
            evidence_filename=evidence_filename,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON document written to the sidecar."""
        return {
            "schema": "forensicvision.provenance/1",
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "input_sha512": self.input_sha512,
            "output_sha512": self.output_sha512,
            "input_md5": self.input_md5,
            "output_md5": self.output_md5,
            "operation": self.operation,
            "model": self.model,
            "model_version": self.model_version,
            "model_kind": self.model_kind,
            "model_license": self.model_license,
            "model_source": self.model_source,
            "weights_sha256": self.weights_sha256,
            "parameters": self.parameters,
            "device": self.device,
            "gpu": self.gpu,
            "duration_s": self.duration_s,
            "input_dimensions": self.input_dimensions,
            "output_dimensions": self.output_dimensions,
            "timestamp": self.timestamp,
            "pipeline": self.pipeline,
            "may_synthesise": self.may_synthesise,
            "roi": self.roi,
            "case_id": self.case_id,
            "evidence_filename": self.evidence_filename,
            "notes": self.notes,
            "environment": self.environment,
            "disclaimer": FORENSIC_REPORT_DISCLAIMER,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Rebuild a record from :meth:`to_dict` output."""
        record = cls()
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return record

    def summary_lines(self) -> List[str]:
        """Return short lines for the history detail view."""
        lines = [
            f"Operation:  {self.operation}",
            f"Model:      {self.model} {self.model_version}".rstrip(),
            f"Kind:       {self.model_kind}",
            f"Device:     {self.device}" + (f" ({self.gpu})" if self.gpu else ""),
            f"Duration:   {self.duration_s:.3f} s",
            f"Input:      {self.input_dimensions}  sha256 {self.input_sha256[:16]}",
            f"Output:     {self.output_dimensions}  sha256 {self.output_sha256[:16]}",
            f"Timestamp:  {self.timestamp}",
        ]
        if self.may_synthesise:
            lines.append("WARNING:    This operation may synthesise image content.")
        return lines


def write_sidecar(record: ProvenanceRecord, image_path: Path) -> Path:
    """Write ``record`` next to ``image_path`` as ``<name>.provenance.json``.

    Args:
        record: The provenance document.
        image_path: Path of the derivative image.

    Returns:
        Path of the written sidecar.
    """
    sidecar = image_path.with_suffix(image_path.suffix + ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, ensure_ascii=False)
    logger.info("Provenance sidecar written: %s", sidecar.name)
    return sidecar
