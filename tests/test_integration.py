"""End-to-end integration test of the specified MVP workflow.

Exercises the exact chain the specification requires (S42, S47)::

    Create case -> Import -> Hash -> Analyse -> Recommend pipeline
    -> Restore -> Derivative -> Hash derivative -> Difference
    -> Processing history -> PDF report

The whole chain runs headless, so it also serves as the smoke test for a
machine with no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.analyzer import analyze_image
from app.constants import DegradationKey, ModelKind
from core.image_io import ImageData, load_image
from forensic.hashing import hash_file
from restoration.auto_engine import AutoRestorationEngine
from restoration.pipeline import Pipeline, PipelineRunner, PipelineStep
from workers.restoration_worker import persist_restoration


@pytest.fixture()
def cctv_evidence(tmp_path: Path) -> Path:
    """A degraded, CCTV-like JPEG: small, soft, noisy and compressed."""
    from scripts.make_sample import build_scene, degrade

    scene = build_scene(720, 480, seed=11)
    degraded = degrade(scene, "cctv")
    path = tmp_path / "forecourt_camera_04.jpg"
    cv2.imwrite(str(path), degraded[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 28])
    return path


class TestMvpWorkflow:
    """The complete specified workflow, in order."""

    def test_full_workflow(
        self, case, cctv_evidence: Path, registry, tmp_path: Path
    ) -> None:
        """Run every stage and assert the forensic invariants at each one."""
        # -- 1. Import evidence -------------------------------------------- #
        import_result = case.import_evidence(cctv_evidence, notes="Test import")
        evidence = import_result.evidence
        stored = Path(evidence.stored_path)

        assert stored.is_file()
        assert stored.parent == case.evidence_dir
        # The original file is untouched; the case holds a byte-exact copy.
        assert stored.read_bytes() == cctv_evidence.read_bytes()

        # -- 2. Hashes ------------------------------------------------------ #
        assert len(evidence.sha256) == 64
        assert len(evidence.sha512) == 128
        assert len(evidence.md5) == 32
        assert hash_file(stored).sha256 == evidence.sha256
        assert case.verify_evidence(evidence) is True

        # -- 3. Originals are write-protected ------------------------------- #
        with pytest.raises(PermissionError):
            stored.open("ab")

        # -- 4. Load and analyse -------------------------------------------- #
        image = case.load_evidence_image(evidence)
        assert image.width == evidence.width and image.height == evidence.height

        report = analyze_image(image, source_path=stored)
        assert report.metrics
        assert report.score(DegradationKey.JPEG.value) > 0.4
        assert report.score(DegradationKey.LOW_RESOLUTION.value) > 0.4

        case.repository.add_analysis(
            case_pk=case.case_pk,
            evidence_id=evidence.id,
            scores=report.scores(),
            details=report.to_dict(),
            analyzer_version=report.analyzer_version,
        )
        assert case.repository.latest_analysis(evidence_id=evidence.id) is not None

        # -- 5. Recommend a pipeline for review ----------------------------- #
        recommendation = AutoRestorationEngine().recommend(report)
        pipeline = recommendation.pipeline
        assert pipeline.enabled_steps, recommendation.rationale_text()
        assert not [
            issue for issue in pipeline.validate()
            if "cannot run" in issue or "not registered" in issue
        ]
        # Every proposed step must be justified.
        for item in recommendation.steps:
            assert item.reason

        # -- 6. Run the pipeline -------------------------------------------- #
        result = PipelineRunner(device="cpu", fp16=False).run(image, pipeline)
        assert result.succeeded
        assert len(result.steps) == len(pipeline.enabled_steps)

        # The hash chain must be continuous from step to step.
        for earlier, later in zip(result.steps, result.steps[1:]):
            assert earlier.output_hashes.sha256 == later.input_hashes.sha256

        # -- 7. Persist the derivative -------------------------------------- #
        outcome = persist_restoration(
            result=result,
            source_image=image,
            pipeline=pipeline,
            case=case,
            evidence=evidence,
        )
        derivative = outcome.derivative_row
        output_path = outcome.output_path

        assert derivative is not None and output_path is not None
        assert output_path.is_file()
        assert output_path.parent == case.derivatives_dir
        # Derivatives are written losslessly so the recorded hash describes
        # exactly the pixels that were produced.
        assert output_path.suffix == ".png"

        # -- 8. Hash the derivative ----------------------------------------- #
        derivative_hashes = hash_file(output_path)
        assert derivative_hashes.sha256 == derivative.sha256
        assert derivative.sha256 != evidence.sha256

        # -- 9. Provenance sidecar ------------------------------------------ #
        sidecar = output_path.with_suffix(output_path.suffix + ".provenance.json")
        assert sidecar.is_file()
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        assert provenance["input_sha256"] == evidence.sha256
        assert provenance["output_sha256"] == derivative_hashes.sha256
        assert provenance["case_id"] == case.case_id
        assert provenance["disclaimer"]
        assert provenance["pipeline"]
        assert provenance["environment"]["application_version"]

        # -- 10. Processing history ----------------------------------------- #
        steps = case.repository.list_steps(case.case_pk, evidence.id)
        assert len(steps) == len(result.steps)
        assert steps[0].input_sha256
        assert steps[-1].output_sha256
        for step in steps:
            assert step.model_name
            assert step.device

        # -- 11. Difference analysis ---------------------------------------- #
        from gui.comparison_viewer import DifferenceMode, compute_difference

        visual, statistics = compute_difference(
            image.pixels, outcome.image.pixels, DifferenceMode.AMPLIFIED
        )
        assert visual.shape[2] == 3
        assert statistics["mean_absolute_difference"] > 0
        assert "psnr_db" in statistics

        # -- 12. Audit trail ------------------------------------------------ #
        actions = {event.action for event in case.repository.list_audit(case.case_pk)}
        assert {"case.create", "evidence.import", "derivative.create"} <= actions

        # -- 13. PDF report -------------------------------------------------- #
        from reports.pdf_report import ForensicReportBuilder

        context = {
            "evidence": evidence,
            "evidence_id": evidence.id,
            "derivative": derivative,
            "investigator": "Test Runner",
            "organisation": "ForensicVision CI",
            "metadata": evidence.file_metadata,
            "analysis": report.to_dict(),
            "pipeline": derivative.pipeline,
            "may_synthesise": result.may_synthesise,
            "models": [
                info.to_dict()
                for info in (step.info() for step in pipeline.enabled_steps)
                if info is not None
            ],
            "history": steps,
            "original_image": image.pixels,
            "enhanced_image": outcome.image.pixels,
            "difference_image": visual,
            "difference_statistics": statistics,
            "include_audit": True,
            "custom_limitations": "Synthetic test material.",
        }

        report_path = case.reports_dir / "integration_report.pdf"
        written = ForensicReportBuilder(case).build(context, report_path)

        assert written.is_file()
        assert written.stat().st_size > 20_000
        header = written.read_bytes()[:5]
        assert header == b"%PDF-", header

        # -- 14. Report registration ---------------------------------------- #
        report_hashes = hash_file(written)
        case.repository.add_report(
            case_pk=case.case_pk,
            evidence_id=evidence.id,
            path=str(written),
            kind="pdf",
            sha256=report_hashes.sha256,
            size_bytes=report_hashes.size_bytes,
            author="Test Runner",
        )
        assert case.repository.list_reports(case.case_pk)

        # -- 15. Evidence is still intact ------------------------------------ #
        assert case.verify_evidence(evidence) is True
        assert hash_file(stored).sha256 == evidence.sha256

        counts = case.counts()
        assert counts["evidence"] == 1
        assert counts["derivatives"] == 1
        assert counts["reports"] == 1
        assert counts["steps"] == len(result.steps)


class TestForensicInvariants:
    """Properties that must hold no matter what the user does."""

    def test_derivative_never_overwrites_evidence(
        self, case, cctv_evidence: Path, registry
    ) -> None:
        """Repeated runs produce distinct derivatives, never touching the original."""
        evidence = case.import_evidence(cctv_evidence).evidence
        original_digest = hash_file(Path(evidence.stored_path)).sha256
        image = case.load_evidence_image(evidence)

        pipeline = Pipeline(steps=[PipelineStep("clahe")])
        paths = []
        for _ in range(3):
            result = PipelineRunner(device="cpu").run(image, pipeline)
            outcome = persist_restoration(
                result=result, source_image=image, pipeline=pipeline,
                case=case, evidence=evidence,
            )
            paths.append(outcome.output_path)

        assert len({str(p) for p in paths}) == 3
        assert hash_file(Path(evidence.stored_path)).sha256 == original_digest

    def test_classical_pipeline_is_reproducible(
        self, case, cctv_evidence: Path, registry
    ) -> None:
        """A deterministic pipeline produces bit-identical output twice."""
        evidence = case.import_evidence(cctv_evidence).evidence
        image = case.load_evidence_image(evidence)
        pipeline = Pipeline(
            steps=[
                PipelineStep("deblock", {"strength": 0.6}),
                PipelineStep("clahe", {"clip_limit": 2.0, "tile_grid": 8}),
                PipelineStep("lanczos", {"scale": 2}),
            ]
        )
        runner = PipelineRunner(device="cpu")
        first = runner.run(image, pipeline)
        second = runner.run(image, pipeline)
        assert (
            first.steps[-1].output_hashes.sha256
            == second.steps[-1].output_hashes.sha256
        )

    def test_derivative_chain_is_traceable(
        self, case, cctv_evidence: Path, registry
    ) -> None:
        """A derivative of a derivative records its parent."""
        evidence = case.import_evidence(cctv_evidence).evidence
        image = case.load_evidence_image(evidence)

        first_pipeline = Pipeline(steps=[PipelineStep("deblock")])
        first_result = PipelineRunner(device="cpu").run(image, first_pipeline)
        first = persist_restoration(
            result=first_result, source_image=image, pipeline=first_pipeline,
            case=case, evidence=evidence,
        )

        second_pipeline = Pipeline(steps=[PipelineStep("clahe")])
        second_result = PipelineRunner(device="cpu").run(
            first.image, second_pipeline
        )
        second = persist_restoration(
            result=second_result, source_image=first.image,
            pipeline=second_pipeline, case=case, evidence=evidence,
            parent_derivative=first.derivative_row,
        )

        assert second.derivative_row.parent_derivative_id == first.derivative_row.id
        assert second.provenance.input_sha256 == first.derivative_row.sha256

    def test_report_contains_mandatory_disclaimer(
        self, case, cctv_evidence: Path
    ) -> None:
        """The specified disclaimer text reaches the rendered PDF."""
        from reports.pdf_report import ForensicReportBuilder

        evidence = case.import_evidence(cctv_evidence).evidence
        path = case.reports_dir / "disclaimer_check.pdf"
        ForensicReportBuilder(case).build(
            {"evidence": evidence, "evidence_id": evidence.id}, path
        )

        # Extract the text so the assertion covers what a reader actually sees,
        # not merely that the constant exists in the codebase.
        try:
            from pypdf import PdfReader
        except ImportError:
            pytest.skip("pypdf is not installed; cannot read back the PDF text")

        text = " ".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        normalised = " ".join(text.split())
        assert "may infer or synthesize structures" in normalised
        assert "derivative representation" in normalised

    def test_application_starts_without_ml(self, monkeypatch) -> None:
        """The engine still registers classical models when torch is absent.

        The specification requires the application to start when no ML stack is
        available (S42), so a missing torch must degrade the model list rather
        than break start-up.
        """
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is unavailable in this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

        from restoration.classical.models import LanczosUpscaleModel

        model = LanczosUpscaleModel()
        assert model.availability().ok
        output = model.process(
            np.zeros((32, 32, 3), dtype=np.float32), scale=2
        )
        assert output.shape == (64, 64, 3)


class TestFunctionalSelfTest:
    """The ``--self-test`` harness.

    This is what verifies a packaged build, so it has to be trustworthy in its
    own right: it must exercise real work, and it must fail when something is
    genuinely broken rather than reporting success regardless.
    """

    def test_all_checks_pass_from_source(self, registry) -> None:
        """Every functional check passes in a working source checkout."""
        from app.selftest import run_self_test

        report = run_self_test(verbose=False)
        assert report.ok, [
            (r.name, r.detail) for r in report.failures
        ]

    def test_covers_the_whole_workflow(self, registry) -> None:
        """The harness exercises analysis, restoration and reporting."""
        from app.selftest import run_self_test

        names = {r.name for r in run_self_test(verbose=False).results}
        for expected in (
            "Model registry",
            "Classical operator",
            "Degradation analysis",
            "Case creation and evidence import",
            "Evidence integrity verification",
            "Derivative and provenance",
            "PDF report generation",
        ):
            assert expected in names, expected

    def test_detects_a_broken_registry(self, registry, monkeypatch) -> None:
        """A model family that fails to import is reported, not swallowed.

        Without this, a packaging error that drops a model family would be
        invisible: the build would still start and still restore images with
        the classical operators.
        """
        import restoration

        def broken(replace: bool = False) -> int:
            restoration.REGISTRATION_REPORT.clear()
            restoration.REGISTRATION_REPORT["realesrgan"] = {
                "status": "error", "error": "simulated packaging failure"
            }
            return 0

        monkeypatch.setattr(restoration, "register_all_models", broken)

        from app.selftest import run_self_test

        report = run_self_test(verbose=False)
        assert not report.ok
        assert any("registry" in r.name.lower() for r in report.failures)

    def test_leaves_no_temporary_files(self, registry) -> None:
        """The harness cleans up after itself."""
        import tempfile
        from pathlib import Path

        from app.selftest import run_self_test

        before = set(Path(tempfile.gettempdir()).glob("forensicvision_selftest_*"))
        run_self_test(verbose=False)
        after = set(Path(tempfile.gettempdir()).glob("forensicvision_selftest_*"))
        assert after == before


class TestReportSelfConsistency:
    """The report must not describe the same run two different ways."""

    def test_models_are_derived_when_the_caller_omits_them(
        self, registry, tmp_path
    ) -> None:
        """Section 8 describes the pipeline's models with no ``models`` key.

        Model collection used to live only in the GUI's report dialog, so any
        other front end got a section 8 reading "No models were used" directly
        under a section 6 listing neural steps.
        """
        from reports.pdf_report import ForensicReportBuilder

        pipeline = {
            "steps": [
                {"model": "clahe", "params": {}},
                {"model": "unsharp", "params": {}},
                {"model": "clahe", "params": {}},
            ]
        }

        models = ForensicReportBuilder._models_from_pipeline(pipeline)

        names = [entry["name"] for entry in models]
        assert names == ["clahe", "unsharp"], "duplicates must collapse"
        assert all(entry.get("license") for entry in models)

    def test_an_empty_pipeline_still_reports_no_models(self, registry) -> None:
        """A classical-free, step-free run is honestly described as such."""
        from reports.pdf_report import ForensicReportBuilder

        assert ForensicReportBuilder._models_from_pipeline({"steps": []}) == []
        assert ForensicReportBuilder._models_from_pipeline(None) == []
