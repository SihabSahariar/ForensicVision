"""Repository layer: all database access goes through these functions.

Keeping SQL/ORM usage in one module means the GUI never holds a live ORM
session, which avoids cross-thread session misuse - the single most common
source of bugs when workers write results back.

Every function opens its own short-lived session and returns detached objects
or plain dictionaries.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select

from app.version import APP_VERSION
from database.database import Database
from database.models import (
    AnalysisRecord,
    AuditEvent,
    Case,
    Derivative,
    Evidence,
    ProcessingStep,
    ReportRecord,
)

logger = logging.getLogger(__name__)

__all__ = ["CaseRepository"]


class CaseRepository:
    """CRUD operations scoped to a single case database."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------ cases
    def create_case(
        self,
        case_id: str,
        title: str = "",
        investigator: str = "",
        organisation: str = "",
        description: str = "",
        root_path: str = "",
        safe_mode_default: bool = True,
    ) -> Case:
        """Insert a new case row and return it (detached)."""
        with self._db.session() as session:
            case = Case(
                case_id=case_id,
                title=title,
                investigator=investigator,
                organisation=organisation,
                description=description,
                root_path=root_path,
                app_version=APP_VERSION,
                safe_mode_default=safe_mode_default,
            )
            session.add(case)
            session.flush()
            session.expunge_all()
            logger.info("Created case %s (pk=%s)", case_id, case.id)
            return case

    def get_case(self, case_id: Optional[str] = None) -> Optional[Case]:
        """Return a case by identifier, or the only case in the file."""
        with self._db.read_session() as session:
            stmt = select(Case)
            if case_id:
                stmt = stmt.where(Case.case_id == case_id)
            case = session.execute(stmt.limit(1)).scalar_one_or_none()
            if case is not None:
                session.expunge(case)
            return case

    def update_case(self, case_pk: int, **fields: Any) -> None:
        """Apply ``fields`` to the case row identified by ``case_pk``."""
        with self._db.session() as session:
            case = session.get(Case, case_pk)
            if case is None:
                return
            for key, value in fields.items():
                if hasattr(case, key):
                    setattr(case, key, value)

    # --------------------------------------------------------------- evidence
    def add_evidence(self, **fields: Any) -> Evidence:
        """Insert an evidence row; ``fields`` map directly to model columns."""
        metadata = fields.pop("file_metadata", None)
        with self._db.session() as session:
            evidence = Evidence(**fields)
            if metadata is not None:
                evidence.set_file_metadata(metadata)
            session.add(evidence)
            session.flush()
            session.expunge_all()
            logger.info(
                "Registered evidence %s (sha256 %s)",
                evidence.original_filename,
                evidence.sha256[:16],
            )
            return evidence

    def list_evidence(self, case_pk: int) -> List[Evidence]:
        """Return all evidence rows for a case, oldest first."""
        with self._db.read_session() as session:
            rows = (
                session.execute(
                    select(Evidence)
                    .where(Evidence.case_pk == case_pk)
                    .order_by(Evidence.imported_at.asc(), Evidence.id.asc())
                )
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return list(rows)

    def get_evidence(self, evidence_id: int) -> Optional[Evidence]:
        """Return one evidence row by primary key."""
        with self._db.read_session() as session:
            row = session.get(Evidence, evidence_id)
            if row is not None:
                session.expunge(row)
            return row

    def find_evidence_by_hash(self, case_pk: int, sha256: str) -> Optional[Evidence]:
        """Return an existing evidence row with the same content digest."""
        with self._db.read_session() as session:
            row = session.execute(
                select(Evidence).where(
                    Evidence.case_pk == case_pk, Evidence.sha256 == sha256
                )
            ).scalar_one_or_none()
            if row is not None:
                session.expunge(row)
            return row

    def update_evidence(self, evidence_id: int, **fields: Any) -> None:
        """Apply ``fields`` to an evidence row (notes, metadata, ...)."""
        metadata = fields.pop("file_metadata", None)
        with self._db.session() as session:
            row = session.get(Evidence, evidence_id)
            if row is None:
                return
            if metadata is not None:
                row.set_file_metadata(metadata)
            for key, value in fields.items():
                if hasattr(row, key):
                    setattr(row, key, value)

    # ------------------------------------------------------------ derivatives
    def add_derivative(self, **fields: Any) -> Derivative:
        """Insert a derivative row and return it (detached)."""
        parameters = fields.pop("parameters", None)
        provenance = fields.pop("provenance", None)
        pipeline = fields.pop("pipeline", None)
        with self._db.session() as session:
            derivative = Derivative(**fields)
            if parameters is not None:
                derivative.set_parameters(parameters)
            if provenance is not None:
                derivative.set_provenance(provenance)
            if pipeline is not None:
                derivative.set_pipeline(pipeline)
            session.add(derivative)
            session.flush()
            session.expunge_all()
            logger.info(
                "Registered derivative %s (sha256 %s)",
                derivative.label or derivative.path,
                derivative.sha256[:16],
            )
            return derivative

    def list_derivatives(
        self, case_pk: int, evidence_id: Optional[int] = None
    ) -> List[Derivative]:
        """Return derivatives for a case, optionally filtered by evidence."""
        with self._db.read_session() as session:
            stmt = select(Derivative).where(Derivative.case_pk == case_pk)
            if evidence_id is not None:
                stmt = stmt.where(Derivative.evidence_id == evidence_id)
            rows = (
                session.execute(stmt.order_by(Derivative.created_at.asc()))
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return list(rows)

    def get_derivative(self, derivative_id: int) -> Optional[Derivative]:
        """Return one derivative row by primary key."""
        with self._db.read_session() as session:
            row = session.get(Derivative, derivative_id)
            if row is not None:
                session.expunge(row)
            return row

    # --------------------------------------------------------------- analysis
    def add_analysis(
        self,
        case_pk: int,
        scores: Dict[str, Any],
        details: Dict[str, Any],
        evidence_id: Optional[int] = None,
        derivative_id: Optional[int] = None,
        analyzer_version: str = "",
        roi: Optional[Dict[str, Any]] = None,
    ) -> AnalysisRecord:
        """Persist a degradation-analysis result."""
        with self._db.session() as session:
            record = AnalysisRecord(
                case_pk=case_pk,
                evidence_id=evidence_id,
                derivative_id=derivative_id,
                analyzer_version=analyzer_version,
            )
            record.set_scores(scores)
            record.set_details(details)
            if roi:
                record.roi_json = AnalysisRecord._dump(roi)
            session.add(record)
            session.flush()
            session.expunge_all()
            return record

    def latest_analysis(
        self, evidence_id: Optional[int] = None, derivative_id: Optional[int] = None
    ) -> Optional[AnalysisRecord]:
        """Return the most recent analysis for a target, if any."""
        with self._db.read_session() as session:
            stmt = select(AnalysisRecord)
            if evidence_id is not None:
                stmt = stmt.where(AnalysisRecord.evidence_id == evidence_id)
            if derivative_id is not None:
                stmt = stmt.where(AnalysisRecord.derivative_id == derivative_id)
            row = session.execute(
                stmt.order_by(AnalysisRecord.created_at.desc()).limit(1)
            ).scalar_one_or_none()
            if row is not None:
                session.expunge(row)
            return row

    # ---------------------------------------------------------------- history
    def add_step(self, **fields: Any) -> ProcessingStep:
        """Record one executed processing step."""
        parameters = fields.pop("parameters", None)
        with self._db.session() as session:
            step = ProcessingStep(**fields)
            if parameters is not None:
                step.set_parameters(parameters)
            session.add(step)
            session.flush()
            session.expunge_all()
            return step

    def list_steps(
        self, case_pk: int, evidence_id: Optional[int] = None
    ) -> List[ProcessingStep]:
        """Return processing steps ordered by run and sequence."""
        with self._db.read_session() as session:
            stmt = select(ProcessingStep).where(ProcessingStep.case_pk == case_pk)
            if evidence_id is not None:
                stmt = stmt.where(ProcessingStep.evidence_id == evidence_id)
            rows = (
                session.execute(
                    stmt.order_by(
                        ProcessingStep.started_at.asc(), ProcessingStep.sequence.asc()
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return list(rows)

    # ---------------------------------------------------------------- reports
    def add_report(self, **fields: Any) -> ReportRecord:
        """Record a generated report artefact."""
        with self._db.session() as session:
            report = ReportRecord(**fields)
            session.add(report)
            session.flush()
            session.expunge_all()
            return report

    def list_reports(self, case_pk: int) -> List[ReportRecord]:
        """Return report rows for a case, newest first."""
        with self._db.read_session() as session:
            rows = (
                session.execute(
                    select(ReportRecord)
                    .where(ReportRecord.case_pk == case_pk)
                    .order_by(ReportRecord.created_at.desc())
                )
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return list(rows)

    # ------------------------------------------------------------------ audit
    def add_audit(
        self,
        case_pk: int,
        action: str,
        target: str = "",
        detail: str = "",
        actor: str = "",
        safe_mode: bool = True,
    ) -> None:
        """Append an audit-trail entry.

        Audit writes must never break the operation they describe, so failures
        are logged rather than raised.
        """
        try:
            with self._db.session() as session:
                session.add(
                    AuditEvent(
                        case_pk=case_pk,
                        action=action,
                        target=target,
                        detail=detail,
                        actor=actor,
                        safe_mode=safe_mode,
                        app_version=APP_VERSION,
                    )
                )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to write audit event %s", action)

    def list_audit(self, case_pk: int, limit: int = 500) -> List[AuditEvent]:
        """Return the most recent audit entries, newest first."""
        with self._db.read_session() as session:
            rows = (
                session.execute(
                    select(AuditEvent)
                    .where(AuditEvent.case_pk == case_pk)
                    .order_by(AuditEvent.timestamp.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
            for row in rows:
                session.expunge(row)
            return list(rows)

    # ------------------------------------------------------------- statistics
    def counts(self, case_pk: int) -> Dict[str, int]:
        """Return row counts used by the case summary header."""
        with self._db.read_session() as session:
            def _count(model, extra=None) -> int:
                stmt = select(func.count()).select_from(model).where(
                    model.case_pk == case_pk
                )
                if extra is not None:
                    stmt = stmt.where(extra)
                return int(session.execute(stmt).scalar_one())

            return {
                "evidence": _count(Evidence),
                "derivatives": _count(Derivative),
                "analyses": _count(AnalysisRecord),
                "reports": _count(ReportRecord),
                "steps": _count(ProcessingStep),
            }
