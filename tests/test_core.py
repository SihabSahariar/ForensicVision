"""Tests for image I/O, case management and the database layer."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.exceptions import CaseError, EvidenceError, ImageSaveError, UnsupportedFormatError
from core.image_io import ImageData, is_supported_path, load_image, save_image


class TestImageIO:
    """Loading and saving with bit-depth and alpha fidelity."""

    def test_round_trip_uint8(self, tmp_path: Path) -> None:
        """8-bit RGB survives a PNG round trip bit-exactly."""
        pixels = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
        path = tmp_path / "image.png"
        save_image(ImageData(pixels=pixels), path)
        loaded = load_image(path)
        assert np.array_equal(loaded.pixels, pixels)
        assert loaded.bit_depth == 8

    def test_round_trip_uint16(self, tmp_path: Path) -> None:
        """16-bit data is preserved rather than silently truncated to 8-bit."""
        pixels = (np.random.default_rng(1).random((30, 40, 3)) * 65535).astype(np.uint16)
        path = tmp_path / "image16.png"
        save_image(ImageData(pixels=pixels), path)
        loaded = load_image(path)
        assert loaded.dtype == np.uint16
        assert loaded.bit_depth == 16
        assert np.array_equal(loaded.pixels, pixels)

    def test_alpha_preserved(self, tmp_path: Path) -> None:
        """An alpha channel survives the round trip."""
        pixels = (np.random.default_rng(2).random((20, 25, 4)) * 255).astype(np.uint8)
        path = tmp_path / "rgba.png"
        save_image(ImageData(pixels=pixels), path)
        loaded = load_image(path)
        assert loaded.has_alpha
        assert loaded.channels == 4
        assert np.array_equal(loaded.pixels, pixels)

    def test_channel_order_is_rgb(self, tmp_path: Path) -> None:
        """Loading returns RGB, not OpenCV's BGR."""
        pixels = np.zeros((8, 8, 3), dtype=np.uint8)
        pixels[..., 0] = 255  # pure red in RGB
        path = tmp_path / "red.png"
        save_image(ImageData(pixels=pixels), path)
        loaded = load_image(path)
        assert loaded.pixels[0, 0, 0] == 255
        assert loaded.pixels[0, 0, 1] == 0
        assert loaded.pixels[0, 0, 2] == 0

    def test_grayscale(self, tmp_path: Path) -> None:
        """Single-channel images stay single-channel."""
        pixels = (np.random.default_rng(3).random((16, 16)) * 255).astype(np.uint8)
        path = tmp_path / "gray.png"
        save_image(ImageData(pixels=pixels), path)
        loaded = load_image(path)
        assert loaded.is_gray
        assert loaded.channels == 1

    def test_refuses_overwrite_by_default(self, tmp_path: Path) -> None:
        """Existing files are not clobbered without an explicit flag."""
        pixels = np.zeros((4, 4, 3), dtype=np.uint8)
        path = tmp_path / "image.png"
        save_image(ImageData(pixels=pixels), path)
        with pytest.raises(ImageSaveError):
            save_image(ImageData(pixels=pixels), path)
        save_image(ImageData(pixels=pixels), path, overwrite=True)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """An unknown extension is rejected."""
        path = tmp_path / "file.xyz"
        path.write_bytes(b"nonsense")
        with pytest.raises(UnsupportedFormatError):
            load_image(path)

    def test_supported_path_predicate(self) -> None:
        """The extension predicate is case-insensitive."""
        assert is_supported_path("a.JPG")
        assert is_supported_path("a.tiff")
        assert not is_supported_path("a.pdf")

    def test_conversions(self) -> None:
        """Float and grayscale conversions produce the documented ranges."""
        pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
        image = ImageData(pixels=pixels)
        as_float = image.to_float_rgb()
        assert as_float.dtype == np.float32
        assert 0.0 <= as_float.min() and as_float.max() <= 1.0
        gray = image.to_gray_float()
        assert gray.shape == (10, 10)
        assert abs(float(gray.mean()) - 128 / 255) < 0.01

    def test_describe(self) -> None:
        """The description dictionary carries the expected keys."""
        image = ImageData(pixels=np.zeros((7, 11, 3), dtype=np.uint8))
        described = image.describe()
        assert described["width"] == 11
        assert described["height"] == 7
        assert described["bit_depth"] == 8


class TestCaseManagement:
    """Case creation, import and verification."""

    def test_layout_created(self, case) -> None:
        """Every required subdirectory exists."""
        from app.constants import CASE_SUBDIRS

        for sub in CASE_SUBDIRS:
            assert (case.root / sub).is_dir()
        assert (case.root / "case.db").is_file()
        assert (case.root / "case.json").is_file()

    def test_case_id_sequence(self, tmp_path: Path) -> None:
        """Case identifiers increment."""
        from core.case_manager import next_case_id

        parent = tmp_path / "cases"
        parent.mkdir()
        assert next_case_id(parent) == "CASE-0001"
        (parent / "CASE-0001").mkdir()
        (parent / "CASE-0007").mkdir()
        assert next_case_id(parent) == "CASE-0008"

    def test_refuses_existing_non_empty(self, tmp_path: Path) -> None:
        """Creating over a populated folder is refused."""
        from core.case_manager import CaseManager

        parent = tmp_path / "cases"
        target = parent / "CASE-0001"
        target.mkdir(parents=True)
        (target / "stray.txt").write_text("x")
        with pytest.raises(CaseError):
            CaseManager.create(parent=parent, case_id="CASE-0001")

    def test_import_copies_and_hashes(self, case, sample_png: Path) -> None:
        """Import stores a copy and records the digest of that copy."""
        from forensic.hashing import hash_file

        result = case.import_evidence(sample_png)
        stored = Path(result.evidence.stored_path)
        assert stored.is_file()
        assert stored.parent == case.evidence_dir
        assert result.hashes.sha256 == hash_file(stored).sha256
        assert result.evidence.width == 480
        assert result.evidence.height == 320

    def test_original_is_write_protected(self, case, sample_png: Path) -> None:
        """The stored original is read-only under safe mode."""
        result = case.import_evidence(sample_png)
        with pytest.raises(PermissionError):
            Path(result.evidence.stored_path).open("wb")

    def test_duplicate_content_detected(self, case, sample_png: Path) -> None:
        """Re-importing identical content does not store a second copy."""
        first = case.import_evidence(sample_png)
        second = case.import_evidence(sample_png)
        assert second.is_duplicate
        assert second.evidence.id == first.evidence.id
        assert len(list(case.evidence_dir.iterdir())) == 1

    def test_verify_evidence(self, case, sample_png: Path) -> None:
        """Verification passes for untouched evidence."""
        result = case.import_evidence(sample_png)
        assert case.verify_evidence(result.evidence) is True

    def test_unsupported_import_rejected(self, case, tmp_path: Path) -> None:
        """A non-image file is refused."""
        path = tmp_path / "notes.txt"
        path.write_text("not an image")
        with pytest.raises(EvidenceError):
            case.import_evidence(path)

    def test_metadata_document_written(self, case, sample_png: Path) -> None:
        """A metadata sidecar is written into the case."""
        case.import_evidence(sample_png)
        documents = list(case.metadata_dir.glob("*.metadata.json"))
        assert documents

    def test_reopen(self, case, sample_png: Path) -> None:
        """A case can be closed and reopened with its evidence intact."""
        from core.case_manager import CaseManager

        case.import_evidence(sample_png)
        root = case.root
        case_id = case.case_id
        case.close()

        reopened = CaseManager.open(root, guard=case.guard)
        try:
            assert reopened.case_id == case_id
            assert len(reopened.list_evidence()) == 1
        finally:
            reopened.close()

    def test_audit_trail(self, case, sample_png: Path) -> None:
        """Import writes an audit entry."""
        case.import_evidence(sample_png)
        events = case.repository.list_audit(case.case_pk)
        actions = {event.action for event in events}
        assert "case.create" in actions
        assert "evidence.import" in actions


class TestDatabase:
    """Repository behaviour."""

    def test_counts(self, case, sample_png: Path) -> None:
        """Row counts reflect what was inserted."""
        assert case.counts()["evidence"] == 0
        case.import_evidence(sample_png)
        assert case.counts()["evidence"] == 1

    def test_derivative_registration(self, case, sample_png: Path) -> None:
        """A derivative row links back to its evidence."""
        evidence = case.import_evidence(sample_png).evidence
        derivative = case.repository.add_derivative(
            case_pk=case.case_pk,
            evidence_id=evidence.id,
            path=str(case.derivatives_dir / "d.png"),
            sha256="a" * 64,
            operation="deblur",
            model_name="Test",
            parameters={"strength": 1.0},
        )
        assert derivative.id is not None
        assert derivative.parameters == {"strength": 1.0}
        assert len(case.list_derivatives(evidence.id)) == 1

    def test_analysis_round_trip(self, case, sample_png: Path) -> None:
        """Analysis scores and details survive storage."""
        evidence = case.import_evidence(sample_png).evidence
        case.repository.add_analysis(
            case_pk=case.case_pk,
            evidence_id=evidence.id,
            scores={"blur": 0.5},
            details={"metrics": {}},
            analyzer_version="1.0.0",
        )
        latest = case.repository.latest_analysis(evidence_id=evidence.id)
        assert latest is not None
        assert latest.scores["blur"] == 0.5

    def test_steps_ordered(self, case, sample_png: Path) -> None:
        """Processing steps come back in sequence order."""
        evidence = case.import_evidence(sample_png).evidence
        for index in range(3):
            case.repository.add_step(
                case_pk=case.case_pk,
                evidence_id=evidence.id,
                run_id="run",
                sequence=index,
                operation="test",
                model_name=f"model{index}",
            )
        steps = case.repository.list_steps(case.case_pk, evidence.id)
        assert [s.sequence for s in steps] == [0, 1, 2]
