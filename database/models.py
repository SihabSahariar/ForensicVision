"""SQLAlchemy ORM models for the per-case forensic database.

Every case carries its own SQLite file inside the case directory, so a case
folder is self-contained and can be archived or handed over as a unit.

Schema notes
------------
* ``Evidence`` rows are append-only in spirit: the application never rewrites
  the stored original, and Forensic Safe Mode blocks deletion.
* ``Derivative`` rows form a tree via ``parent_derivative_id``, which is what
  the Processing History panel renders.
* ``AuditEvent`` is the tamper-evidence trail: each row records the safe-mode
  state at the time of the action.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "Case",
    "Evidence",
    "Derivative",
    "AnalysisRecord",
    "ProcessingStep",
    "ReportRecord",
    "AuditEvent",
    "utcnow",
]


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ForensicVision tables."""


class JsonMixin:
    """Helpers for columns that store JSON documents as text."""

    @staticmethod
    def _dump(value: Optional[Dict[str, Any]]) -> str:
        if not value:
            return "{}"
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _load(value: Optional[str]) -> Dict[str, Any]:
        if not value:
            return {}
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {"value": loaded}
        except (TypeError, ValueError):
            return {}


class Case(Base, JsonMixin):
    """A forensic case: the root container for evidence and derivatives."""

    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    investigator: Mapped[str] = mapped_column(String(255), default="")
    organisation: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    root_path: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )
    app_version: Mapped[str] = mapped_column(String(32), default="")
    safe_mode_default: Mapped[bool] = mapped_column(Boolean, default=True)

    evidence: Mapped[List["Evidence"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", lazy="selectin"
    )
    reports: Mapped[List["ReportRecord"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", lazy="selectin"
    )
    audit_events: Mapped[List["AuditEvent"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", lazy="select"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Case {self.case_id!r} title={self.title!r}>"


class Evidence(Base, JsonMixin):
    """An imported original image, preserved byte-for-byte."""

    __tablename__ = "evidence"
    __table_args__ = (UniqueConstraint("case_pk", "sha256", name="uq_case_sha256"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)

    original_filename: Mapped[str] = mapped_column(String(512))
    original_path: Mapped[str] = mapped_column(Text, default="")
    stored_path: Mapped[str] = mapped_column(Text)
    imported_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    sha256: Mapped[str] = mapped_column(String(64), index=True)
    sha512: Mapped[str] = mapped_column(String(128), default="")
    md5: Mapped[str] = mapped_column(String(32), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    image_format: Mapped[str] = mapped_column(String(16), default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    channels: Mapped[int] = mapped_column(Integer, default=0)
    bit_depth: Mapped[int] = mapped_column(Integer, default=8)

    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    notes: Mapped[str] = mapped_column(Text, default="")

    case: Mapped["Case"] = relationship(back_populates="evidence")
    derivatives: Mapped[List["Derivative"]] = relationship(
        back_populates="evidence", cascade="all, delete-orphan", lazy="selectin"
    )
    analyses: Mapped[List["AnalysisRecord"]] = relationship(
        back_populates="evidence", cascade="all, delete-orphan", lazy="selectin"
    )
    steps: Mapped[List["ProcessingStep"]] = relationship(
        back_populates="evidence", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def file_metadata(self) -> Dict[str, Any]:
        """Decoded metadata document."""
        return self._load(self.metadata_json)

    def set_file_metadata(self, value: Dict[str, Any]) -> None:
        """Store ``value`` as the metadata document."""
        self.metadata_json = self._dump(value)

    @property
    def dimensions(self) -> str:
        """Return ``"W x H"`` for display."""
        return f"{self.width} x {self.height}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Evidence {self.original_filename!r} sha256={self.sha256[:12]}>"


class Derivative(Base, JsonMixin):
    """An enhanced image produced from evidence or another derivative."""

    __tablename__ = "derivatives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    evidence_id: Mapped[int] = mapped_column(ForeignKey("evidence.id"), index=True)
    parent_derivative_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("derivatives.id"), nullable=True
    )

    path: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    sha256: Mapped[str] = mapped_column(String(64), index=True)
    sha512: Mapped[str] = mapped_column(String(128), default="")
    md5: Mapped[str] = mapped_column(String(32), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    channels: Mapped[int] = mapped_column(Integer, default=0)
    bit_depth: Mapped[int] = mapped_column(Integer, default=8)

    operation: Mapped[str] = mapped_column(String(64), default="")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    model_kind: Mapped[str] = mapped_column(String(16), default="")
    parameters_json: Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    pipeline_json: Mapped[str] = mapped_column(Text, default="{}")

    evidence: Mapped["Evidence"] = relationship(back_populates="derivatives")
    children: Mapped[List["Derivative"]] = relationship(
        back_populates="parent", lazy="selectin"
    )
    parent: Mapped[Optional["Derivative"]] = relationship(
        back_populates="children", remote_side=[id]
    )
    analyses: Mapped[List["AnalysisRecord"]] = relationship(
        back_populates="derivative", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def parameters(self) -> Dict[str, Any]:
        """Decoded model parameters."""
        return self._load(self.parameters_json)

    def set_parameters(self, value: Dict[str, Any]) -> None:
        """Store ``value`` as the model parameters."""
        self.parameters_json = self._dump(value)

    @property
    def provenance(self) -> Dict[str, Any]:
        """Decoded provenance document."""
        return self._load(self.provenance_json)

    def set_provenance(self, value: Dict[str, Any]) -> None:
        """Store ``value`` as the provenance document."""
        self.provenance_json = self._dump(value)

    @property
    def pipeline(self) -> Dict[str, Any]:
        """Decoded pipeline description."""
        return self._load(self.pipeline_json)

    def set_pipeline(self, value: Dict[str, Any]) -> None:
        """Store ``value`` as the pipeline description."""
        self.pipeline_json = self._dump(value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Derivative {self.label!r} op={self.operation!r}>"


class AnalysisRecord(Base, JsonMixin):
    """A degradation-analysis result for evidence or a derivative."""

    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    evidence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("evidence.id"), nullable=True, index=True
    )
    derivative_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("derivatives.id"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    analyzer_version: Mapped[str] = mapped_column(String(32), default="")
    roi_json: Mapped[str] = mapped_column(Text, default="{}")
    scores_json: Mapped[str] = mapped_column(Text, default="{}")
    details_json: Mapped[str] = mapped_column(Text, default="{}")

    evidence: Mapped[Optional["Evidence"]] = relationship(back_populates="analyses")
    derivative: Mapped[Optional["Derivative"]] = relationship(back_populates="analyses")

    @property
    def scores(self) -> Dict[str, Any]:
        """Decoded normalised degradation scores."""
        return self._load(self.scores_json)

    def set_scores(self, value: Dict[str, Any]) -> None:
        """Store normalised degradation scores."""
        self.scores_json = self._dump(value)

    @property
    def details(self) -> Dict[str, Any]:
        """Decoded per-metric technical details."""
        return self._load(self.details_json)

    def set_details(self, value: Dict[str, Any]) -> None:
        """Store per-metric technical details."""
        self.details_json = self._dump(value)


class ProcessingStep(Base, JsonMixin):
    """One executed operation, recorded for the processing history panel."""

    __tablename__ = "processing_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    evidence_id: Mapped[int] = mapped_column(ForeignKey("evidence.id"), index=True)
    derivative_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("derivatives.id"), nullable=True
    )
    run_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=0)

    operation: Mapped[str] = mapped_column(String(64), default="")
    model_name: Mapped[str] = mapped_column(String(128), default="")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    model_kind: Mapped[str] = mapped_column(String(16), default="")
    parameters_json: Mapped[str] = mapped_column(Text, default="{}")

    input_sha256: Mapped[str] = mapped_column(String(64), default="")
    output_sha256: Mapped[str] = mapped_column(String(64), default="")
    input_size: Mapped[str] = mapped_column(String(32), default="")
    output_size: Mapped[str] = mapped_column(String(32), default="")

    device: Mapped[str] = mapped_column(String(64), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    message: Mapped[str] = mapped_column(Text, default="")

    evidence: Mapped["Evidence"] = relationship(back_populates="steps")

    @property
    def parameters(self) -> Dict[str, Any]:
        """Decoded operation parameters."""
        return self._load(self.parameters_json)

    def set_parameters(self, value: Dict[str, Any]) -> None:
        """Store operation parameters."""
        self.parameters_json = self._dump(value)


class ReportRecord(Base, JsonMixin):
    """A generated report artefact and its digest."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    evidence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("evidence.id"), nullable=True
    )
    path: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(32), default="pdf")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    sha256: Mapped[str] = mapped_column(String(64), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    author: Mapped[str] = mapped_column(String(255), default="")

    case: Mapped["Case"] = relationship(back_populates="reports")


class AuditEvent(Base, JsonMixin):
    """An immutable audit-trail entry."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(255), default="")
    target: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[str] = mapped_column(Text, default="")
    safe_mode: Mapped[bool] = mapped_column(Boolean, default=True)
    app_version: Mapped[str] = mapped_column(String(32), default="")

    case: Mapped["Case"] = relationship(back_populates="audit_events")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AuditEvent {self.action!r} at {self.timestamp}>"
