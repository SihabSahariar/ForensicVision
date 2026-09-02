"""Forensic PDF report generation with ReportLab.

Report structure follows the specification (S32): case information, evidence
information, hashes, metadata, image analysis, restoration pipeline, model
information, processing parameters, before/after, difference analysis,
processing history, and the limitations/disclaimer section.

The disclaimer is mandatory and is rendered on its own page as well as in the
footer of every page, so a printed extract can never lose it.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.constants import (
    FORENSIC_REPORT_DISCLAIMER,
    HEURISTIC_DISCLAIMER,
    OCR_DISCLAIMER,
    SYNTHESIS_WARNING,
)
from app.version import APP_NAME, APP_VERSION, build_string
from core.case_manager import CaseManager
from core.exceptions import OperationCancelled, ReportError
from core.image_utils import ensure_uint8_rgb, resize_to_fit
from forensic.hashing import MD5_ADVISORY
from forensic.provenance import environment_snapshot

logger = logging.getLogger(__name__)

__all__ = ["ForensicReportBuilder"]

ProgressCallback = Callable[[int, str], None]

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

_INK = colors.HexColor("#14171d")
_MUTED = colors.HexColor("#4a5261")
_RULE = colors.HexColor("#c3c9d4")
_ACCENT = colors.HexColor("#23557f")
_WARN_BG = colors.HexColor("#fdf4e3")
_WARN_BORDER = colors.HexColor("#c8863c")
_TABLE_HEAD = colors.HexColor("#eceff4")


def _styles() -> Dict[str, ParagraphStyle]:
    """Build the paragraph styles used throughout the report."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FVTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=_INK, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "FVSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10.5, leading=14, textColor=_MUTED, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "FVH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=_ACCENT,
            spaceBefore=14, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "FVH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=_INK,
            spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "FVBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=13.5, textColor=_INK, alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "FVSmall", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=_MUTED,
        ),
        "mono": ParagraphStyle(
            "FVMono", parent=base["Normal"], fontName="Courier",
            fontSize=7.5, leading=10, textColor=_INK,
        ),
        "warn": ParagraphStyle(
            "FVWarn", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12.5, textColor=_INK,
            borderColor=_WARN_BORDER, borderWidth=0.8, borderPadding=7,
            backColor=_WARN_BG, spaceBefore=6, spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "FVCaption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8, leading=10.5, textColor=_MUTED, spaceBefore=2,
            spaceAfter=8,
        ),
    }


def _escape(value: Any) -> str:
    """Escape text for ReportLab's mini-HTML paragraph parser."""
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


class ForensicReportBuilder:
    """Renders a case/evidence report to PDF."""

    PAGE_SIZE = A4
    MARGIN = 18 * mm

    def __init__(self, case: CaseManager) -> None:
        self._case = case
        self._styles = _styles()
        self._case_label = case.case_id

    # ------------------------------------------------------------------ build
    def build(
        self,
        context: Dict[str, Any],
        output_path: Path,
        progress: Optional[ProgressCallback] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> Path:
        """Render the report.

        Args:
            context: Content assembled by the report dialog.
            output_path: Destination PDF.
            progress: Optional ``(percent, message)`` callback.
            cancelled: Optional predicate polled between sections.

        Returns:
            The written path.

        Raises:
            ReportError: Rendering failed.
            OperationCancelled: The user aborted.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        story: List[Any] = []
        sections: Sequence[Tuple[str, Callable[[List[Any], Dict[str, Any]], None]]] = (
            ("Cover", self._section_cover),
            ("Case information", self._section_case),
            ("Evidence information", self._section_evidence),
            ("Hashes", self._section_hashes),
            ("Metadata", self._section_metadata),
            ("Image analysis", self._section_analysis),
            ("Restoration pipeline", self._section_pipeline),
            ("Model information", self._section_models),
            ("Before / after", self._section_before_after),
            ("Difference analysis", self._section_difference),
            ("Processing history", self._section_history),
            ("OCR", self._section_ocr),
            ("Audit trail", self._section_audit),
            ("Limitations", self._section_limitations),
        )

        total = len(sections)
        for index, (name, renderer) in enumerate(sections):
            if cancelled is not None and cancelled():
                raise OperationCancelled("Report generation cancelled")
            if progress is not None:
                progress(int(index * 92 / total), f"Rendering: {name}")
            try:
                renderer(story, context)
            except Exception:
                logger.exception("Report section '%s' failed", name)
                story.append(
                    Paragraph(
                        f"[Section '{_escape(name)}' could not be rendered; see "
                        "the application log.]",
                        self._styles["warn"],
                    )
                )

        if progress is not None:
            progress(94, "Writing PDF")

        try:
            document = self._make_document(output_path, context)
            document.build(story)
        except Exception as exc:
            raise ReportError(f"Could not write the report: {exc}") from exc

        logger.info("Report written: %s", output_path)
        return output_path

    def _make_document(
        self, path: Path, context: Dict[str, Any]
    ) -> BaseDocTemplate:
        """Create the document template with header/footer painting."""
        document = BaseDocTemplate(
            str(path),
            pagesize=self.PAGE_SIZE,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=self.MARGIN,
            bottomMargin=self.MARGIN + 10 * mm,
            title=f"{self._case_label} forensic report",
            author=context.get("investigator", APP_NAME),
            subject="Forensic image analysis and enhancement report",
            creator=build_string(),
        )
        frame = Frame(
            document.leftMargin, document.bottomMargin,
            document.width, document.height, id="body",
        )
        document.addPageTemplates(
            [PageTemplate(id="main", frames=[frame], onPage=self._paint_page)]
        )
        return document

    def _paint_page(self, canvas, document) -> None:
        """Draw the running header and footer."""
        canvas.saveState()
        width, height = self.PAGE_SIZE

        canvas.setStrokeColor(_RULE)
        canvas.setLineWidth(0.5)
        canvas.line(
            self.MARGIN, height - self.MARGIN + 6,
            width - self.MARGIN, height - self.MARGIN + 6,
        )
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(_MUTED)
        canvas.drawString(
            self.MARGIN, height - self.MARGIN + 10,
            f"{APP_NAME} forensic report  -  {self._case_label}",
        )
        canvas.drawRightString(
            width - self.MARGIN, height - self.MARGIN + 10,
            f"Page {canvas.getPageNumber()}",
        )

        footer_y = self.MARGIN + 2 * mm
        canvas.line(self.MARGIN, footer_y + 16, width - self.MARGIN, footer_y + 16)
        canvas.setFont("Helvetica-Oblique", 6.6)
        canvas.setFillColor(_MUTED)
        canvas.drawString(
            self.MARGIN, footer_y + 8,
            "Enhanced imagery is a derivative representation. AI-based "
            "restoration may infer or synthesise structures not present in the "
            "source evidence.",
        )
        canvas.drawString(
            self.MARGIN, footer_y,
            f"Generated by {build_string()}",
        )
        canvas.restoreState()

    # --------------------------------------------------------------- helpers
    def _table(
        self,
        rows: Sequence[Tuple[str, Any]],
        key_width: float = 46 * mm,
        mono_values: bool = False,
    ) -> Table:
        """Build a two-column key/value table."""
        style = self._styles["mono" if mono_values else "body"]
        data = [
            [
                Paragraph(f"<b>{_escape(key)}</b>", self._styles["body"]),
                Paragraph(_escape(value), style),
            ]
            for key, value in rows
        ]
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        table = Table(data, colWidths=[key_width, available - key_width])
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.25, _RULE),
                ]
            )
        )
        return table

    def _grid(
        self, header: Sequence[str], rows: Sequence[Sequence[Any]],
        widths: Optional[Sequence[float]] = None,
    ) -> Table:
        """Build a bordered grid table."""
        data = [[Paragraph(f"<b>{_escape(h)}</b>", self._styles["small"])
                 for h in header]]
        for row in rows:
            data.append(
                [Paragraph(_escape(cell), self._styles["small"]) for cell in row]
            )
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        if widths is None:
            widths = [available / len(header)] * len(header)
        table = Table(data, colWidths=list(widths), repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEAD),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.25, _RULE),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def _image_flowable(
        self, array: np.ndarray, max_width: float, max_height: float = 90 * mm
    ) -> Optional[RLImage]:
        """Embed a numpy array as a PNG flowable scaled to fit."""
        try:
            rgb = ensure_uint8_rgb(array)[..., :3]
            rgb = resize_to_fit(rgb, 1600)
            from PIL import Image as PILImage

            buffer = io.BytesIO()
            PILImage.fromarray(rgb).save(buffer, format="PNG")
            buffer.seek(0)

            height, width = rgb.shape[:2]
            scale = min(max_width / width, max_height / height)
            return RLImage(buffer, width=width * scale, height=height * scale)
        except Exception:
            logger.exception("Could not embed image in report")
            return None

    # -------------------------------------------------------------- sections
    def _section_cover(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Title block and the mandatory disclaimer."""
        case = self._case.case
        story.append(Paragraph("Forensic Image Analysis Report", self._styles["title"]))
        story.append(
            Paragraph(
                f"{_escape(case.case_id)}"
                + (f" &nbsp;|&nbsp; {_escape(case.title)}" if case.title else ""),
                self._styles["subtitle"],
            )
        )
        story.append(
            self._table(
                [
                    ("Report generated", datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S UTC")),
                    ("Investigator", context.get("investigator") or case.investigator
                     or "(not recorded)"),
                    ("Organisation", context.get("organisation") or case.organisation
                     or "(not recorded)"),
                    ("Application", build_string()),
                    ("Forensic Safe Mode",
                     "ENABLED" if self._case.guard.enabled else "DISABLED"),
                ]
            )
        )
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("Mandatory disclaimer", self._styles["h2"]))
        story.append(Paragraph(_escape(FORENSIC_REPORT_DISCLAIMER), self._styles["warn"]))
        if context.get("may_synthesise"):
            story.append(Paragraph(_escape(SYNTHESIS_WARNING), self._styles["warn"]))

    def _section_case(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Case metadata."""
        case = self._case.case
        counts = self._case.counts()
        story.append(Paragraph("1. Case information", self._styles["h1"]))
        story.append(
            self._table(
                [
                    ("Case ID", case.case_id),
                    ("Title", case.title or "(none)"),
                    ("Created", case.created_at),
                    ("Case directory", str(self._case.root)),
                    ("Evidence items", counts.get("evidence", 0)),
                    ("Derivatives", counts.get("derivatives", 0)),
                    ("Analyses recorded", counts.get("analyses", 0)),
                    ("Processing steps", counts.get("steps", 0)),
                ]
            )
        )
        if case.description:
            story.append(Paragraph("Description", self._styles["h2"]))
            story.append(Paragraph(_escape(case.description), self._styles["body"]))

    def _section_evidence(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Evidence identification."""
        evidence = context.get("evidence")
        story.append(Paragraph("2. Evidence information", self._styles["h1"]))
        if evidence is None:
            story.append(Paragraph("No evidence item selected.", self._styles["body"]))
            return
        story.append(
            self._table(
                [
                    ("Filename", evidence.original_filename),
                    ("Source path", evidence.original_path or "(not recorded)"),
                    ("Stored copy", evidence.stored_path),
                    ("Imported (UTC)", evidence.imported_at),
                    ("Format", evidence.image_format),
                    ("Dimensions", f"{evidence.width} x {evidence.height} px"),
                    ("Channels", evidence.channels),
                    ("Bit depth", f"{evidence.bit_depth} bits/channel"),
                    ("File size", f"{evidence.size_bytes:,} bytes"),
                    ("Notes", evidence.notes or "(none)"),
                ]
            )
        )

    def _section_hashes(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Digest listing for evidence and derivative."""
        story.append(Paragraph("3. Cryptographic hashes", self._styles["h1"]))
        evidence = context.get("evidence")
        derivative = context.get("derivative")

        if evidence is not None:
            story.append(Paragraph("Original evidence", self._styles["h2"]))
            story.append(
                self._table(
                    [
                        ("SHA-256 (primary)", evidence.sha256),
                        ("SHA-512", evidence.sha512),
                        ("MD5 (legacy)", evidence.md5),
                    ],
                    mono_values=True,
                )
            )
        if derivative is not None:
            story.append(Paragraph("Derivative", self._styles["h2"]))
            story.append(
                self._table(
                    [
                        ("File", Path(derivative.path).name),
                        ("SHA-256 (primary)", derivative.sha256),
                        ("SHA-512", derivative.sha512),
                        ("MD5 (legacy)", derivative.md5),
                    ],
                    mono_values=True,
                )
            )
        story.append(Paragraph(_escape(MD5_ADVISORY), self._styles["caption"]))

    def _section_metadata(self, story: List[Any], context: Dict[str, Any]) -> None:
        """EXIF and file metadata."""
        story.append(Paragraph("4. Metadata", self._styles["h1"]))
        metadata = context.get("metadata")
        if not metadata:
            story.append(
                Paragraph("No metadata was extracted.", self._styles["body"])
            )
            return

        gps = metadata.get("gps") or {}
        if gps:
            story.append(Paragraph("GPS", self._styles["h2"]))
            story.append(
                self._table([(k.replace("_", " ").title(), v) for k, v in gps.items()])
            )

        exif = metadata.get("exif") or {}
        story.append(
            Paragraph(f"EXIF tags ({len(exif)} recorded)", self._styles["h2"])
        )
        if exif:
            rows = [(key, value) for key, value in sorted(exif.items())]
            story.append(
                self._grid(
                    ["Tag", "Value"], rows,
                    widths=[62 * mm, self.PAGE_SIZE[0] - 2 * self.MARGIN - 62 * mm],
                )
            )
        else:
            story.append(
                Paragraph(
                    "No EXIF metadata is present in this file.",
                    self._styles["body"],
                )
            )

        warnings = metadata.get("warnings") or []
        if warnings:
            story.append(Paragraph("Extraction warnings", self._styles["h2"]))
            for warning in warnings:
                story.append(Paragraph(f"- {_escape(warning)}", self._styles["body"]))

    def _section_analysis(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Degradation indicators."""
        story.append(PageBreak())
        story.append(Paragraph("5. Image analysis", self._styles["h1"]))
        report = context.get("analysis")
        if not report:
            story.append(
                Paragraph("No analysis was recorded for this item.", self._styles["body"])
            )
            return

        story.append(Paragraph(_escape(HEURISTIC_DISCLAIMER), self._styles["warn"]))

        metrics = report.get("metrics", {})
        order = report.get("scores", {}).keys()
        rows = []
        for key in order:
            metric = metrics.get(key, {})
            rows.append(
                [
                    metric.get("label", key),
                    f"{metric.get('percent', 0)} / 100",
                    metric.get("severity", ""),
                    metric.get("method", ""),
                ]
            )
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        story.append(
            self._grid(
                ["Indicator", "Score", "Band", "Estimator"], rows,
                widths=[32 * mm, 18 * mm, 16 * mm, available - 66 * mm],
            )
        )

        story.append(Paragraph("Detailed measurements", self._styles["h2"]))
        for key in order:
            metric = metrics.get(key, {})
            measurements = metric.get("measurements", {})
            if not measurements:
                continue
            story.append(
                Paragraph(
                    f"<b>{_escape(metric.get('label', key))}</b> "
                    f"- {metric.get('percent', 0)}/100",
                    self._styles["body"],
                )
            )
            rows = [(str(k), str(v)) for k, v in measurements.items()]
            story.append(self._table(rows, key_width=58 * mm, mono_values=True))
            for note in metric.get("notes", []):
                story.append(Paragraph(f"- {_escape(note)}", self._styles["caption"]))
            story.append(Spacer(1, 3 * mm))

    def _section_pipeline(self, story: List[Any], context: Dict[str, Any]) -> None:
        """The executed restoration pipeline."""
        story.append(PageBreak())
        story.append(Paragraph("6. Restoration pipeline", self._styles["h1"]))
        pipeline = context.get("pipeline")
        if not pipeline or not pipeline.get("steps"):
            story.append(
                Paragraph(
                    "No restoration was applied to this evidence item.",
                    self._styles["body"],
                )
            )
            return

        steps = [s for s in pipeline["steps"] if s.get("enabled", True)]
        rows = []
        for index, step in enumerate(steps, start=1):
            rows.append(
                [
                    str(index),
                    step.get("display_name", step.get("model", "")),
                    step.get("task", ""),
                    step.get("kind", ""),
                    "yes" if step.get("may_synthesise") else "no",
                ]
            )
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        story.append(
            self._grid(
                ["#", "Operation", "Task", "Kind", "May synthesise"], rows,
                widths=[10 * mm, available - 90 * mm, 30 * mm, 24 * mm, 26 * mm],
            )
        )

        if pipeline.get("rationale"):
            story.append(Paragraph("Rationale", self._styles["h2"]))
            for line in str(pipeline["rationale"]).splitlines():
                if line.strip():
                    story.append(Paragraph(_escape(line), self._styles["body"]))

        story.append(Paragraph("7. Processing parameters", self._styles["h1"]))
        for index, step in enumerate(steps, start=1):
            parameters = step.get("parameters", {})
            story.append(
                Paragraph(
                    f"<b>Step {index}: {_escape(step.get('display_name', ''))}</b>",
                    self._styles["body"],
                )
            )
            if parameters:
                story.append(
                    self._table(
                        [(str(k), str(v)) for k, v in sorted(parameters.items())],
                        key_width=52 * mm, mono_values=True,
                    )
                )
            else:
                story.append(
                    Paragraph("(no adjustable parameters)", self._styles["caption"])
                )
            if step.get("note"):
                story.append(Paragraph(_escape(step["note"]), self._styles["caption"]))
            story.append(Spacer(1, 2 * mm))

    @staticmethod
    def _models_from_pipeline(pipeline: Any) -> List[Dict[str, Any]]:
        """Describe every distinct model named by ``pipeline``.

        Section 6 lists the pipeline's steps, so section 8 must describe those
        same models or the report contradicts itself. Deriving them here rather
        than relying on the caller means a CLI or service front end produces the
        same provenance section the GUI does.
        """
        from restoration.registry import ModelRegistry

        models: List[Dict[str, Any]] = []
        seen = set()
        for step in (pipeline or {}).get("steps", []):
            name = step.get("model")
            if not name or name in seen:
                continue
            seen.add(name)
            info = ModelRegistry.info(name)
            if info is not None:
                models.append(info.to_dict())
        return models

    def _section_models(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Model provenance and licensing."""
        story.append(Paragraph("8. Model information", self._styles["h1"]))
        models = context.get("models")
        if not models:
            models = self._models_from_pipeline(context.get("pipeline"))
        if not models:
            story.append(Paragraph("No models were used.", self._styles["body"]))
        else:
            for info in models:
                rows = [
                    ("Model", info.get("display_name", "")),
                    ("Version", info.get("version", "")),
                    ("Task", info.get("task_label", info.get("task", ""))),
                    ("Kind", info.get("kind", "")),
                    ("Authors", info.get("authors", "")),
                    ("Licence", info.get("license", "")),
                    ("Repository", info.get("repository", "")),
                    ("Paper", info.get("paper", "")),
                    ("May synthesise",
                     "YES" if info.get("may_synthesise") else "no"),
                ]
                weights = info.get("weights") or []
                for weight in weights:
                    rows.append(
                        (
                            f"Weights: {weight.get('filename', '')}",
                            f"licence {weight.get('license', 'n/a')}; "
                            f"sha256 {weight.get('sha256') or '(not published)'}",
                        )
                    )
                story.append(KeepTogether([self._table(rows), Spacer(1, 4 * mm)]))
                if info.get("method"):
                    story.append(
                        Paragraph(_escape(info["method"]), self._styles["caption"])
                    )

        environment = context.get("environment") or environment_snapshot()
        story.append(Paragraph("Execution environment", self._styles["h2"]))
        story.append(
            self._table([(str(k), str(v)) for k, v in environment.items()])
        )

    def _section_before_after(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Side-by-side thumbnails of the original and derivative."""
        original = context.get("original_image")
        enhanced = context.get("enhanced_image")
        if original is None and enhanced is None:
            return

        story.append(PageBreak())
        story.append(Paragraph("9. Before / after", self._styles["h1"]))
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        half = (available - 6 * mm) / 2

        cells = []
        headers = []
        if original is not None:
            flowable = self._image_flowable(original, half, 95 * mm)
            if flowable is not None:
                cells.append(flowable)
                headers.append("ORIGINAL EVIDENCE")
        if enhanced is not None:
            flowable = self._image_flowable(enhanced, half, 95 * mm)
            if flowable is not None:
                cells.append(flowable)
                headers.append("ENHANCED DERIVATIVE")

        if not cells:
            return

        header_row = [
            Paragraph(f"<b>{_escape(h)}</b>", self._styles["small"]) for h in headers
        ]
        table = Table(
            [header_row, cells],
            colWidths=[half] * len(cells),
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ]
            )
        )
        story.append(table)
        story.append(
            Paragraph(
                "Images are reproduced at reduced resolution for the report. "
                "The full-resolution derivative and its hash are recorded in "
                "the case directory.",
                self._styles["caption"],
            )
        )

    def _section_difference(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Difference map and statistics."""
        difference = context.get("difference_image")
        statistics = context.get("difference_statistics") or {}
        if difference is None and not statistics:
            return

        story.append(Paragraph("10. Difference analysis", self._styles["h1"]))
        story.append(
            Paragraph(
                "Analytical visualisation, not evidence. Difference maps are "
                "rendered with a chosen gain and colour mapping; their apparent "
                "intensity is a display choice.",
                self._styles["warn"],
            )
        )

        if difference is not None:
            available = self.PAGE_SIZE[0] - 2 * self.MARGIN
            flowable = self._image_flowable(difference, available, 95 * mm)
            if flowable is not None:
                story.append(flowable)
                story.append(
                    Paragraph(
                        _escape(context.get("difference_label", "Absolute difference")),
                        self._styles["caption"],
                    )
                )

        if statistics:
            story.append(
                self._table(
                    [
                        (str(key).replace("_", " ").title(), str(value))
                        for key, value in statistics.items()
                    ]
                )
            )

    def _section_history(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Per-step processing history with input/output digests."""
        story.append(PageBreak())
        story.append(Paragraph("11. Processing history", self._styles["h1"]))
        steps = context.get("history") or []
        if not steps:
            story.append(
                Paragraph("No processing steps are recorded.", self._styles["body"])
            )
            return

        story.append(
            Paragraph(
                "Each step records the digest of exactly what entered and left "
                "it, so the chain from original to derivative is verifiable "
                "step by step.",
                self._styles["body"],
            )
        )
        for step in steps:
            rows = [
                ("Sequence", step.sequence + 1),
                ("Operation", step.operation),
                ("Model", f"{step.model_name} {step.model_version}".strip()),
                ("Kind", step.model_kind),
                ("Device", step.device),
                ("Started (UTC)", step.started_at),
                ("Duration", f"{step.duration_s:.3f} s"),
                ("Input size", step.input_size),
                ("Input SHA-256", step.input_sha256),
                ("Output size", step.output_size),
                ("Output SHA-256", step.output_sha256),
                ("Status", step.status),
            ]
            story.append(
                KeepTogether(
                    [
                        self._table(rows, key_width=40 * mm, mono_values=True),
                        Spacer(1, 4 * mm),
                    ]
                )
            )

    def _section_ocr(self, story: List[Any], context: Dict[str, Any]) -> None:
        """OCR readings, when any were recorded."""
        ocr = context.get("ocr")
        if not ocr:
            return
        story.append(Paragraph("12. OCR interpretation", self._styles["h1"]))
        story.append(Paragraph(_escape(OCR_DISCLAIMER), self._styles["warn"]))
        rows = []
        for entry in ocr:
            rows.append(
                [
                    entry.get("source", ""),
                    entry.get("engine", ""),
                    f"{entry.get('mean_confidence', 0.0) * 100:.0f}%",
                    entry.get("text", ""),
                ]
            )
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        story.append(
            self._grid(
                ["Source", "Engine", "Confidence", "Reading"], rows,
                widths=[28 * mm, 24 * mm, 22 * mm, available - 74 * mm],
            )
        )

    def _section_audit(self, story: List[Any], context: Dict[str, Any]) -> None:
        """Audit-trail extract."""
        if not context.get("include_audit", True):
            return
        story.append(Paragraph("13. Audit trail", self._styles["h1"]))
        try:
            events = self._case.repository.list_audit(self._case.case_pk, limit=250)
        except Exception:
            logger.exception("Could not read the audit trail")
            return
        if not events:
            story.append(Paragraph("No audit entries.", self._styles["body"]))
            return

        rows = [
            [
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                event.action,
                event.target,
                "on" if event.safe_mode else "OFF",
            ]
            for event in reversed(events)
        ]
        available = self.PAGE_SIZE[0] - 2 * self.MARGIN
        story.append(
            self._grid(
                ["Timestamp (UTC)", "Action", "Target", "Safe mode"], rows,
                widths=[34 * mm, 34 * mm, available - 86 * mm, 18 * mm],
            )
        )

    def _section_limitations(self, story: List[Any], context: Dict[str, Any]) -> None:
        """The mandatory limitations and disclaimer section."""
        story.append(PageBreak())
        story.append(Paragraph("14. Limitations and disclaimer", self._styles["h1"]))

        story.append(Paragraph("Enhancement", self._styles["h2"]))
        story.append(Paragraph(_escape(FORENSIC_REPORT_DISCLAIMER), self._styles["warn"]))

        story.append(Paragraph("Generative models", self._styles["h2"]))
        story.append(Paragraph(_escape(SYNTHESIS_WARNING), self._styles["body"]))

        story.append(Paragraph("Analysis indicators", self._styles["h2"]))
        story.append(Paragraph(_escape(HEURISTIC_DISCLAIMER), self._styles["body"]))

        story.append(Paragraph("Specific limitations", self._styles["h2"]))
        limitations = [
            "Samples clipped at the black or white point at capture carry no "
            "recoverable information. Detail appearing in those regions after "
            "enhancement is synthesised.",
            "Spatial frequencies removed by blur, downsampling or lossy "
            "compression are not present in the file. Any operation that "
            "appears to restore them is inferring them.",
            "Super-resolution applied to an already-interpolated image adds no "
            "measured detail.",
            "Difference maps and other analytical visualisations depend on the "
            "gain and colour mapping chosen for display.",
            "OCR readings reflect the image supplied to the OCR engine. A "
            "reading taken from an enhanced derivative is not evidence of the "
            "characters present in the original.",
            "Object detection class labels are the detector's estimate and are "
            "not identifications.",
        ]
        for item in limitations:
            story.append(Paragraph(f"- {_escape(item)}", self._styles["body"]))

        if context.get("custom_limitations"):
            story.append(Paragraph("Examiner's notes", self._styles["h2"]))
            for line in str(context["custom_limitations"]).splitlines():
                if line.strip():
                    story.append(Paragraph(_escape(line), self._styles["body"]))

        story.append(Spacer(1, 6 * mm))
        story.append(
            Paragraph(
                f"Report produced by {_escape(build_string())} on "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.",
                self._styles["caption"],
            )
        )
