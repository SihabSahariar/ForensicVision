"""Tests for hashing, metadata extraction, provenance and safe mode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.exceptions import IntegrityError, SafeModeViolation
from forensic.hashing import HashSet, hash_array, hash_bytes, hash_file, verify_file
from forensic.metadata import extract_metadata, human_size
from forensic.provenance import ProvenanceRecord, environment_snapshot, write_sidecar
from forensic.safe_mode import SafeModeGuard


class TestHashing:
    """Digest computation and verification."""

    def test_known_vector(self) -> None:
        """Digests match the published values for the empty string."""
        result = hash_bytes(b"")
        assert result.sha256 == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert result.md5 == "d41d8cd98f00b204e9800998ecf8427e"
        assert result.sha512.startswith("cf83e1357eefb8bd")

    def test_abc_vector(self) -> None:
        """SHA-256 of 'abc' matches the FIPS 180-4 example."""
        assert hash_bytes(b"abc").sha256 == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_file_matches_bytes(self, tmp_path: Path) -> None:
        """Hashing a file agrees with hashing its bytes."""
        payload = b"forensic evidence" * 5000
        path = tmp_path / "blob.bin"
        path.write_bytes(payload)
        assert hash_file(path).sha256 == hash_bytes(payload).sha256

    def test_size_recorded(self, tmp_path: Path) -> None:
        """The byte count is captured alongside the digests."""
        path = tmp_path / "blob.bin"
        path.write_bytes(b"x" * 1234)
        assert hash_file(path).size_bytes == 1234

    def test_verify_detects_change(self, tmp_path: Path) -> None:
        """A modified file fails verification."""
        path = tmp_path / "blob.bin"
        path.write_bytes(b"original")
        recorded = hash_file(path)
        assert verify_file(path, recorded) is True
        path.write_bytes(b"tampered")
        assert verify_file(path, recorded) is False

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Hashing a missing file raises rather than returning a null digest."""
        with pytest.raises(IntegrityError):
            hash_file(tmp_path / "does_not_exist.bin")

    def test_array_hash_is_content_addressed(self) -> None:
        """Identical pixels hash identically; a single changed sample does not."""
        first = np.zeros((16, 16, 3), dtype=np.uint8)
        second = first.copy()
        assert hash_array(first).sha256 == hash_array(second).sha256
        second[8, 8, 0] = 1
        assert hash_array(first).sha256 != hash_array(second).sha256

    def test_array_hash_distinguishes_shape(self) -> None:
        """Reshaping the same bytes yields a different content hash."""
        flat = np.zeros((32, 8, 3), dtype=np.uint8)
        tall = np.zeros((8, 32, 3), dtype=np.uint8)
        assert hash_array(flat).sha256 != hash_array(tall).sha256

    def test_serialisation_round_trip(self) -> None:
        """A HashSet survives a dict round trip."""
        original = hash_bytes(b"round trip")
        restored = HashSet.from_dict(original.to_dict())
        assert restored.sha256 == original.sha256
        assert restored.matches(original)

    def test_progress_callback(self, tmp_path: Path) -> None:
        """The progress callback reports monotonically increasing byte counts."""
        path = tmp_path / "big.bin"
        path.write_bytes(b"a" * (3 * 1024 * 1024))
        seen = []
        hash_file(path, progress=lambda done, total: seen.append(done))
        assert seen == sorted(seen)
        assert seen[-1] == path.stat().st_size

    def test_cancellation(self, tmp_path: Path) -> None:
        """A cancelled hash raises instead of returning a partial digest."""
        path = tmp_path / "big.bin"
        path.write_bytes(b"a" * (3 * 1024 * 1024))
        with pytest.raises(IntegrityError):
            hash_file(path, cancelled=lambda: True)


class TestMetadata:
    """Metadata extraction."""

    def test_basic_properties(self, sample_png: Path) -> None:
        """Dimensions and container are read back correctly."""
        metadata = extract_metadata(sample_png)
        assert metadata.filename == "sample.png"
        assert metadata.width == 480
        assert metadata.height == 320
        assert metadata.container == "PNG"
        assert metadata.size_bytes > 0

    def test_serialisation_round_trip(self, sample_png: Path) -> None:
        """Metadata survives a dict round trip."""
        from forensic.metadata import FileMetadata

        original = extract_metadata(sample_png)
        restored = FileMetadata.from_dict(original.to_dict())
        assert restored.width == original.width
        assert restored.filename == original.filename

    def test_missing_file_is_tolerated(self, tmp_path: Path) -> None:
        """A missing file yields warnings rather than an exception."""
        metadata = extract_metadata(tmp_path / "nope.jpg")
        assert metadata.warnings

    def test_human_size(self) -> None:
        """Byte counts render in binary units."""
        assert human_size(0) == "0 B"
        assert human_size(1536).startswith("1.50 KiB")


class TestProvenance:
    """Provenance records and sidecars."""

    def test_round_trip(self) -> None:
        """A record survives a dict round trip."""
        record = ProvenanceRecord.build(
            input_hashes=hash_bytes(b"in"),
            output_hashes=hash_bytes(b"out"),
            operation="deblur",
            model="Test Model",
            model_version="1.0",
            may_synthesise=True,
        )
        restored = ProvenanceRecord.from_dict(record.to_dict())
        assert restored.operation == "deblur"
        assert restored.may_synthesise is True
        assert restored.input_sha256 == hash_bytes(b"in").sha256

    def test_disclaimer_present(self) -> None:
        """Every serialised record carries the mandatory disclaimer."""
        from app.constants import FORENSIC_REPORT_DISCLAIMER

        record = ProvenanceRecord.build(
            input_hashes=hash_bytes(b"a"),
            output_hashes=hash_bytes(b"b"),
            operation="test",
            model="Test",
        )
        assert record.to_dict()["disclaimer"] == FORENSIC_REPORT_DISCLAIMER

    def test_sidecar_written(self, tmp_path: Path) -> None:
        """The sidecar lands next to the image and parses as JSON."""
        image_path = tmp_path / "derivative.png"
        image_path.write_bytes(b"not really a png")
        record = ProvenanceRecord.build(
            input_hashes=hash_bytes(b"a"),
            output_hashes=hash_bytes(b"b"),
            operation="test",
            model="Test",
        )
        sidecar = write_sidecar(record, image_path)
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["operation"] == "test"

    def test_environment_snapshot(self) -> None:
        """The environment snapshot records the application version."""
        from app.version import APP_VERSION

        snapshot = environment_snapshot()
        assert snapshot["application_version"] == APP_VERSION
        assert "python_version" in snapshot


class TestSafeMode:
    """Forensic Safe Mode enforcement."""

    def test_protect_makes_file_readonly(self, tmp_path: Path) -> None:
        """A protected file cannot be opened for writing."""
        path = tmp_path / "evidence.bin"
        path.write_bytes(b"evidence")
        guard = SafeModeGuard(enabled=True)
        guard.protect_file(path)
        assert guard.is_protected(path)
        with pytest.raises(PermissionError):
            path.open("wb")

    def test_write_to_original_is_refused(self, tmp_path: Path) -> None:
        """assert_can_write refuses a registered original."""
        path = tmp_path / "evidence.bin"
        path.write_bytes(b"evidence")
        guard = SafeModeGuard(enabled=True)
        guard.protect_file(path)
        with pytest.raises(SafeModeViolation):
            guard.assert_can_write(path)

    def test_other_paths_are_writable(self, tmp_path: Path) -> None:
        """Non-evidence paths are unaffected."""
        guard = SafeModeGuard(enabled=True)
        guard.assert_can_write(tmp_path / "derivative.png")

    def test_delete_refused(self) -> None:
        """Deletion is blocked while safe mode is on."""
        guard = SafeModeGuard(enabled=True)
        with pytest.raises(SafeModeViolation):
            guard.assert_can_delete("evidence")

    def test_history_immutable(self) -> None:
        """History modification is blocked while safe mode is on."""
        guard = SafeModeGuard(enabled=True)
        with pytest.raises(SafeModeViolation):
            guard.assert_can_modify_history()

    def test_unprotect_requires_safe_mode_off(self, tmp_path: Path) -> None:
        """Protection cannot be lifted while safe mode is enabled."""
        path = tmp_path / "evidence.bin"
        path.write_bytes(b"evidence")
        guard = SafeModeGuard(enabled=True)
        guard.protect_file(path)
        with pytest.raises(SafeModeViolation):
            guard.unprotect_file(path)
        guard.set_enabled(False)
        assert guard.unprotect_file(path) is True

    def test_listeners_notified(self) -> None:
        """State changes reach registered listeners."""
        guard = SafeModeGuard(enabled=True)
        seen = []
        guard.add_listener(seen.append)
        guard.set_enabled(False)
        guard.set_enabled(True)
        assert seen == [False, True]
