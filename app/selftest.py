"""Functional self-test.

``--check`` reports what is *present*. This module verifies what actually
*works*: it registers the models, runs a classical operator, runs the analyzer,
creates a throw-away case, imports and hashes an image, executes a restoration
pipeline, writes a derivative with its provenance sidecar, and renders a PDF.

It exists mainly for verifying a frozen build. A PyInstaller bundle can start,
report a healthy environment and still be broken in ways that only appear when
work is attempted - a missing hidden import, an unbundled data file, a
read-only weights directory. The most dangerous of those failures are silent:
a broken ``torch`` makes the application report "PyTorch not installed" and
fall back to the classical operators, which looks like a legitimate CPU-only
configuration rather than a defect.

Everything runs in a temporary directory and is removed afterwards.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["SelfTestResult", "run_self_test"]


@dataclass
class SelfTestResult:
    """Outcome of one functional check."""

    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False

    @property
    def marker(self) -> str:
        """Three-character status marker for the report."""
        if self.skipped:
            return "--  "
        return "OK  " if self.passed else "FAIL"


@dataclass
class SelfTestReport:
    """Aggregate of every check."""

    results: List[SelfTestResult] = field(default_factory=list)

    def add(
        self, name: str, passed: bool, detail: str = "", skipped: bool = False
    ) -> None:
        """Record one outcome."""
        self.results.append(SelfTestResult(name, passed, detail, skipped))

    @property
    def failures(self) -> List[SelfTestResult]:
        """Checks that ran and failed."""
        return [r for r in self.results if not r.passed and not r.skipped]

    @property
    def skipped(self) -> List[SelfTestResult]:
        """Checks that could not run."""
        return [r for r in self.results if r.skipped]

    @property
    def ok(self) -> bool:
        """Whether every check that ran passed."""
        return not self.failures


def run_self_test(verbose: bool = True) -> SelfTestReport:
    """Exercise the application end to end in a temporary directory.

    Args:
        verbose: Print each result as it completes.

    Returns:
        The completed :class:`SelfTestReport`.
    """
    report = SelfTestReport()
    workspace = Path(tempfile.mkdtemp(prefix="forensicvision_selftest_"))

    def announce(result: SelfTestResult) -> None:
        if verbose:
            print(f"  {result.marker}{result.name:38s} {result.detail}")

    def step(name: str, function: Callable[[], str]) -> bool:
        """Run one check, converting an exception into a failure."""
        try:
            detail = function()
        except Exception as exc:
            report.add(name, False, f"{type(exc).__name__}: {exc}")
            logger.debug("Self-test step %r failed", name, exc_info=True)
            announce(report.results[-1])
            return False
        report.add(name, True, detail)
        announce(report.results[-1])
        return True

    try:
        import numpy as np

        from app.paths import is_frozen, resource_root, weights_dir

        state: dict = {}

        # ---------------------------------------------------------- packaging
        step(
            "Running frozen" if is_frozen() else "Running from source",
            lambda: f"resources at {resource_root()}",
        )

        def _stylesheet() -> str:
            from gui.theme import load_stylesheet

            sheet = load_stylesheet()
            if len(sheet) < 2000:
                raise RuntimeError("stylesheet missing or truncated")
            return f"{len(sheet)} characters"

        step("Stylesheet resource", _stylesheet)

        def _docs() -> str:
            path = resource_root() / "docs" / "LIMITATIONS.md"
            if not path.is_file():
                raise RuntimeError(f"not bundled: {path}")
            return path.name

        step("Documentation resource", _docs)

        def _weights_writable() -> str:
            directory = weights_dir()
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".selftest"
            probe.write_bytes(b"x")
            probe.unlink()
            return str(directory)

        step("Weights directory writable", _weights_writable)

        # ----------------------------------------------------------- registry
        def _registry() -> str:
            from restoration import REGISTRATION_REPORT, register_all_models
            from restoration.registry import ModelRegistry

            total = register_all_models()
            broken = {
                name: info.get("error", "")
                for name, info in REGISTRATION_REPORT.items()
                if info.get("status") != "ok"
            }
            if broken:
                raise RuntimeError(f"model families failed to import: {broken}")
            state["registry"] = ModelRegistry
            return f"{total} models across {len(ModelRegistry.tasks())} tasks"

        if not step("Model registry", _registry):
            return report

        registry = state["registry"]

        # -------------------------------------------------------------- torch
        def _torch() -> str:
            from core.device import get_device_report

            device_report = get_device_report()
            if not device_report.torch_available:
                raise RuntimeError(
                    f"PyTorch did not import: {device_report.error}. In a "
                    "frozen build this usually means a torch submodule was "
                    "excluded; see docs/PACKAGING.md."
                )
            state["device"] = "cuda" if device_report.has_gpu else "cpu"
            return (
                f"{device_report.torch_version} on "
                f"{device_report.device_label()}"
            )

        torch_ok = step("PyTorch import", _torch)
        device = state.get("device", "cpu")

        # ----------------------------------------------------- image pipeline
        from scripts.make_sample import build_scene

        scene = build_scene(320, 224, seed=5)

        def _classical() -> str:
            model = registry.get("clahe")
            output = model.process(scene.astype(np.float32) / 255.0)
            if output.shape[:2] != scene.shape[:2]:
                raise RuntimeError("unexpected output geometry")
            return f"CLAHE -> {output.shape}"

        step("Classical operator", _classical)

        def _analysis() -> str:
            from analysis import analyze_image
            from core.image_io import ImageData

            result = analyze_image(ImageData(pixels=scene))
            if len(result.metrics) != 9:
                raise RuntimeError(f"only {len(result.metrics)} indicators ran")
            state["analysis"] = result
            return f"{len(result.metrics)} indicators"

        step("Degradation analysis", _analysis)

        # ------------------------------------------------------ case workflow
        def _case() -> str:
            import cv2

            from core.case_manager import CaseManager
            from forensic.safe_mode import SafeModeGuard

            source = workspace / "evidence.png"
            cv2.imwrite(str(source), scene[..., ::-1])

            case = CaseManager.create(
                parent=workspace / "cases", case_id="CASE-0001",
                title="Self-test", investigator="Self-test",
                guard=SafeModeGuard(enabled=True),
            )
            evidence = case.import_evidence(source).evidence
            if len(evidence.sha256) != 64:
                raise RuntimeError("evidence was not hashed")
            state["case"] = case
            state["evidence"] = evidence
            return f"imported, sha256 {evidence.sha256[:16]}"

        if not step("Case creation and evidence import", _case):
            return report

        case = state["case"]
        evidence = state["evidence"]

        step(
            "Evidence integrity verification",
            lambda: "SHA-256 re-verified"
            if case.verify_evidence(evidence)
            else _raise("digest mismatch"),
        )

        # ----------------------------------------------------- neural, if any
        def _neural() -> str:
            from restoration.pipeline import Pipeline, PipelineStep

            candidates = [
                info.name for info in registry.available()
                if info.kind == "neural"
            ]
            if not candidates:
                raise _Skip(
                    "no neural model installed - install one from the Model "
                    "Manager to exercise this path"
                )
            name = candidates[0]
            model = registry.get(name)
            model.load(device=device, fp16=False)
            try:
                output = model.process(
                    scene.astype(np.float32) / 255.0, tile_size=256
                )
            finally:
                model.unload()
            state["neural"] = name
            return f"{name} on {device} -> {output.shape}"

        _step_allowing_skip(report, announce, "Neural inference", _neural)

        # ---------------------------------------------- derivative + reporting
        def _derivative() -> str:
            from restoration.pipeline import Pipeline, PipelineRunner, PipelineStep
            from workers.restoration_worker import persist_restoration

            image = case.load_evidence_image(evidence)
            pipeline = Pipeline(steps=[PipelineStep("clahe")])
            run = PipelineRunner(device="cpu").run(image, pipeline)
            outcome = persist_restoration(
                result=run, source_image=image, pipeline=pipeline,
                case=case, evidence=evidence,
            )
            sidecar = outcome.output_path.with_suffix(
                outcome.output_path.suffix + ".provenance.json"
            )
            if not sidecar.is_file():
                raise RuntimeError("provenance sidecar was not written")
            state["outcome"] = outcome
            state["image"] = image
            return f"{outcome.output_path.name} + sidecar"

        derivative_ok = step("Derivative and provenance", _derivative)

        def _report() -> str:
            from reports.pdf_report import ForensicReportBuilder

            outcome = state["outcome"]
            path = case.reports_dir / "selftest.pdf"
            ForensicReportBuilder(case).build(
                {
                    "evidence": evidence,
                    "evidence_id": evidence.id,
                    "derivative": outcome.derivative_row,
                    "analysis": state["analysis"].to_dict(),
                    "pipeline": outcome.derivative_row.pipeline,
                    "history": case.repository.list_steps(
                        case.case_pk, evidence.id
                    ),
                    "original_image": state["image"].pixels,
                    "enhanced_image": outcome.image.pixels,
                    "investigator": "Self-test",
                },
                path,
            )
            if path.read_bytes()[:5] != b"%PDF-":
                raise RuntimeError("output is not a PDF")
            return f"{path.stat().st_size // 1024} KiB"

        if derivative_ok:
            step("PDF report generation", _report)

        try:
            case.close()
        except Exception:  # pragma: no cover - cleanup only
            pass

    except Exception:
        report.add("Self-test harness", False, traceback.format_exc(limit=3))
        announce(report.results[-1])
    finally:
        _remove_tree(workspace)

    return report


class _Skip(Exception):
    """Raised by a check that cannot run in this environment."""


def _raise(message: str) -> str:
    """Raise ``RuntimeError``; used to keep lambdas expression-only."""
    raise RuntimeError(message)


def _step_allowing_skip(report, announce, name: str, function) -> None:
    """Run a check that may legitimately be skipped."""
    try:
        detail = function()
    except _Skip as skip:
        report.add(name, True, str(skip), skipped=True)
        announce(report.results[-1])
        return
    except Exception as exc:
        report.add(name, False, f"{type(exc).__name__}: {exc}")
        announce(report.results[-1])
        return
    report.add(name, True, detail)
    announce(report.results[-1])


def _remove_tree(path: Path) -> None:
    """Delete a directory tree, clearing read-only evidence flags first."""
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    os.chmod(child, 0o666)
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # pragma: no cover - cleanup only
        pass
