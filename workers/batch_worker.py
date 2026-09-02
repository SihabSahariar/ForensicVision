"""Batch processing worker.

Processes a folder of images through import -> analyse -> restore -> export,
supporting pause and cancel. Runs sequentially on one worker thread because the
bottleneck is GPU inference, which serialises anyway; parallelism here would
only increase peak VRAM and the chance of an out-of-memory failure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PyQt5.QtCore import QMutex, QMutexLocker, pyqtSignal

from analysis.analyzer import AnalysisReport, DegradationAnalyzer
from app.constants import SUPPORTED_IMAGE_EXTENSIONS
from core.case_manager import CaseManager
from core.exceptions import OperationCancelled
from core.image_io import load_image
from restoration.auto_engine import AutoRestorationEngine
from restoration.pipeline import Pipeline
from restoration.pipeline import PipelineRunner
from workers.base import BaseWorker
from workers.restoration_worker import persist_restoration

logger = logging.getLogger(__name__)

__all__ = ["BatchWorker", "BatchItemResult", "BatchSummary", "discover_images"]


def discover_images(folder: Path, recursive: bool = False) -> List[Path]:
    """Return supported image files in ``folder``, sorted by name."""
    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in sorted(Path(folder).glob(pattern))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return found


@dataclass
class BatchItemResult:
    """Outcome for one file in a batch run."""

    path: Path
    status: str = "pending"
    message: str = ""
    output_path: Optional[Path] = None
    pipeline_summary: str = ""
    duration_s: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchSummary:
    """Aggregate outcome of a batch run."""

    items: List[BatchItemResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    cancelled: bool = False

    @property
    def completed(self) -> int:
        """Number of files processed successfully."""
        return sum(1 for item in self.items if item.status == "ok")

    @property
    def failed(self) -> int:
        """Number of files that failed."""
        return sum(1 for item in self.items if item.status == "error")

    @property
    def skipped(self) -> int:
        """Number of files skipped (e.g. no restoration indicated)."""
        return sum(1 for item in self.items if item.status == "skipped")


class BatchWorker(BaseWorker):
    """Runs the full workflow over a list of files."""

    description = "Batch processing"

    #: ``(index, total, filename, pipeline_text)`` for the batch dialog.
    item_started = pyqtSignal(int, int, str, str)
    #: ``(BatchItemResult)`` as each file completes.
    item_finished = pyqtSignal(object)

    def __init__(
        self,
        case: CaseManager,
        paths: Sequence[Path],
        fixed_pipeline: Optional[Pipeline] = None,
        device: str = "auto",
        fp16: bool = True,
        export_dir: Optional[Path] = None,
        auto_threshold: float = 0.45,
    ) -> None:
        """Create the worker.

        Args:
            case: Case to import into and record against.
            paths: Files to process.
            fixed_pipeline: Apply this pipeline to every file. When ``None``,
                each file gets its own pipeline from the auto engine.
            device: Compute device preference.
            fp16: Use half precision where supported.
            export_dir: Additional directory to copy derivatives into.
            auto_threshold: Action threshold for the auto engine.
        """
        super().__init__()
        self._case = case
        self._paths = [Path(p) for p in paths]
        self._fixed_pipeline = fixed_pipeline
        self._device = device
        self._fp16 = fp16
        self._export_dir = Path(export_dir) if export_dir else None
        self._analyzer = DegradationAnalyzer()
        self._engine = AutoRestorationEngine(threshold=auto_threshold)
        self._pause_mutex = QMutex()
        self._paused = False

    # ------------------------------------------------------------------ pause
    def pause(self) -> None:
        """Pause after the current file completes."""
        with QMutexLocker(self._pause_mutex):
            self._paused = True

    def resume(self) -> None:
        """Resume a paused run."""
        with QMutexLocker(self._pause_mutex):
            self._paused = False

    @property
    def is_paused(self) -> bool:
        """Whether the run is currently paused."""
        with QMutexLocker(self._pause_mutex):
            return self._paused

    def _wait_while_paused(self) -> None:
        """Block while paused, checking for cancellation."""
        while self.is_paused:
            if self.is_cancelled():
                raise OperationCancelled("Batch cancelled")
            self.msleep(150)

    # --------------------------------------------------------------- execute
    def execute(self) -> BatchSummary:
        """Process every file, continuing past individual failures."""
        summary = BatchSummary()
        started = time.perf_counter()
        total = len(self._paths)

        for index, path in enumerate(self._paths):
            if self.is_cancelled():
                summary.cancelled = True
                break
            self._wait_while_paused()

            item = BatchItemResult(path=path)
            item_started = time.perf_counter()
            self.report(
                int(index * 100 / max(1, total)),
                f"{path.name} ({index + 1}/{total})",
            )

            try:
                self._process_one(path, item, index, total)
            except OperationCancelled:
                summary.cancelled = True
                item.status = "cancelled"
                summary.items.append(item)
                break
            except Exception as exc:
                logger.exception("Batch item failed: %s", path)
                item.status = "error"
                item.message = str(exc)

            item.duration_s = time.perf_counter() - item_started
            summary.items.append(item)
            self.item_finished.emit(item)

        summary.total_duration_s = time.perf_counter() - started
        self.report(100, "Batch complete")
        logger.info(
            "Batch finished: %d ok, %d failed, %d skipped in %.1fs",
            summary.completed, summary.failed, summary.skipped,
            summary.total_duration_s,
        )
        return summary

    # ---------------------------------------------------------------- helpers
    def _process_one(
        self, path: Path, item: BatchItemResult, index: int, total: int
    ) -> None:
        """Import, analyse, restore and record one file."""
        import_result = self._case.import_evidence(path)
        evidence = import_result.evidence
        image = self._case.load_evidence_image(evidence)

        report = self._analyzer.analyze(
            image,
            source_path=Path(evidence.stored_path),
            cancelled=self.is_cancelled,
        )
        item.scores = report.scores()

        self._case.repository.add_analysis(
            case_pk=self._case.case_pk,
            evidence_id=evidence.id,
            scores=report.scores(),
            details=report.to_dict(),
            analyzer_version=report.analyzer_version,
        )

        if self._fixed_pipeline is not None:
            pipeline = self._fixed_pipeline.copy()
        else:
            pipeline = self._engine.recommend(report).pipeline

        item.pipeline_summary = " -> ".join(
            step.display_name for step in pipeline.enabled_steps
        ) or "(no operation indicated)"
        self.item_started.emit(index, total, path.name, item.pipeline_summary)

        if not pipeline.enabled_steps:
            item.status = "skipped"
            item.message = "No degradation indicator exceeded the action threshold."
            return

        # Run inline: this worker is already off the GUI thread, and the
        # derivative is persisted through the same function the interactive
        # path uses, so batch and single-image runs produce identical records.
        runner = PipelineRunner(device=self._device, fp16=self._fp16)
        result = runner.run(image, pipeline, cancelled=self.is_cancelled)
        outcome = persist_restoration(
            result=result,
            source_image=image,
            pipeline=pipeline,
            case=self._case,
            evidence=evidence,
        )

        item.status = "ok"
        item.output_path = outcome.output_path

        if self._export_dir is not None and outcome.output_path is not None:
            import shutil

            self._export_dir.mkdir(parents=True, exist_ok=True)
            destination = self._export_dir / outcome.output_path.name
            shutil.copy2(outcome.output_path, destination)
