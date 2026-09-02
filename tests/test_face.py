"""Tests for face detection, alignment and CodeFormer restoration.

The subject is ``skimage.data.astronaut()`` - a NASA public-domain photograph
that is a long-standing benchmark image in the image-processing literature, and
is already installed as part of scikit-image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.constants import ModelKind, ModelStatus, TaskType
from app.paths import weights_dir
from core.exceptions import InferenceError, ModelNotAvailableError
from detection.face import (
    FFHQ_TEMPLATE_512,
    FaceAligner,
    YUNET_WEIGHT,
    YuNetDetector,
    detector_available,
)


@pytest.fixture(scope="module")
def portrait() -> np.ndarray:
    """A 512x512 RGB photograph containing one clearly visible face."""
    skimage_data = pytest.importorskip("skimage.data")
    return skimage_data.astronaut()


@pytest.fixture(scope="module")
def degraded(portrait: np.ndarray) -> np.ndarray:
    """The same frame degraded the way a surveillance still is.

    Downscale to 128 px, blur, add sensor noise, then compress hard.
    """
    small = cv2.resize(portrait, (128, 128), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0.8)
    rng = np.random.default_rng(0)
    small = np.clip(
        small.astype(np.float32) + rng.normal(0, 6, small.shape), 0, 255
    ).astype(np.uint8)
    _, buffer = cv2.imencode(
        ".jpg", small[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 35]
    )
    small = cv2.imdecode(buffer, cv2.IMREAD_COLOR)[..., ::-1]
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)


yunet_installed = pytest.mark.skipif(
    not (weights_dir() / YUNET_WEIGHT.filename).is_file(),
    reason="YuNet detector weights are not installed",
)
codeformer_installed = pytest.mark.skipif(
    not (weights_dir() / "codeformer.pth").is_file(),
    reason="CodeFormer weights are not installed",
)


class TestAvailability:
    """Honest reporting when the detector is absent."""

    def test_reports_reason_when_unavailable(self, monkeypatch, tmp_path) -> None:
        """A missing model file yields an actionable message, not a crash."""
        monkeypatch.setattr("detection.face.weights_dir", lambda: tmp_path)
        available, reason = detector_available()
        assert available is False
        assert "Model Manager" in reason

    def test_weight_spec_is_declared(self) -> None:
        """The detector model is declared like any other weight."""
        assert YUNET_WEIGHT.filename.endswith(".onnx")
        assert YUNET_WEIGHT.url.startswith("https://")
        assert YUNET_WEIGHT.license_name


class TestTemplate:
    """The canonical alignment template."""

    def test_shape_and_order(self) -> None:
        """Five points, ordered to match YuNet's landmark output."""
        assert FFHQ_TEMPLATE_512.shape == (5, 2)
        # Entry 0 is the subject's right eye, which sits left in the image.
        assert FFHQ_TEMPLATE_512[0][0] < FFHQ_TEMPLATE_512[1][0]
        # Mouth corners sit below the eyes.
        assert FFHQ_TEMPLATE_512[3][1] > FFHQ_TEMPLATE_512[0][1]
        # The nose sits between the eyes horizontally.
        assert (
            FFHQ_TEMPLATE_512[0][0]
            < FFHQ_TEMPLATE_512[2][0]
            < FFHQ_TEMPLATE_512[1][0]
        )


@yunet_installed
class TestDetection:
    """YuNet face detection."""

    def test_finds_the_face(self, portrait: np.ndarray) -> None:
        """The detector locates exactly one face with five landmarks."""
        detections = YuNetDetector().detect(portrait)
        assert len(detections) == 1
        face = detections[0]
        assert face.confidence > 0.5
        assert face.landmarks.shape == (5, 2)
        assert face.width > 40 and face.height > 40

    def test_landmarks_are_anatomically_ordered(self, portrait: np.ndarray) -> None:
        """Landmarks come back in the order the template expects."""
        face = YuNetDetector().detect(portrait)[0]
        right_eye, left_eye, nose, right_mouth, left_mouth = face.landmarks
        assert right_eye[0] < left_eye[0]
        assert nose[1] > right_eye[1]
        assert right_mouth[1] > nose[1]
        assert right_mouth[0] < left_mouth[0]

    def test_landmarks_lie_inside_the_box(self, portrait: np.ndarray) -> None:
        """Every landmark falls within the reported detection box."""
        face = YuNetDetector().detect(portrait)[0]
        x0, y0, x1, y1 = face.box
        margin = 4
        for x, y in face.landmarks:
            assert x0 - margin <= x <= x1 + margin
            assert y0 - margin <= y <= y1 + margin

    def test_inter_ocular_distance(self, portrait: np.ndarray) -> None:
        """The information measure is positive and consistent with the box."""
        face = YuNetDetector().detect(portrait)[0]
        assert 0 < face.inter_ocular_distance < face.width

    def test_finds_face_in_degraded_frame(self, degraded: np.ndarray) -> None:
        """Detection still succeeds on a heavily degraded frame."""
        detections = YuNetDetector().detect(degraded)
        assert len(detections) >= 1

    def test_no_face_in_a_blank_frame(self) -> None:
        """A frame with no face yields no detections rather than a guess."""
        blank = np.full((256, 256, 3), 128, dtype=np.uint8)
        assert YuNetDetector().detect(blank) == []

    def test_roi_conversion(self, portrait: np.ndarray) -> None:
        """A detection converts to a usable ROI."""
        roi = YuNetDetector().detect(portrait)[0].to_roi()
        assert roi.is_valid()
        assert roi.label == "Face"


@yunet_installed
class TestAlignment:
    """Warping into and out of the canonical frame."""

    def test_align_produces_canonical_frame(self, portrait: np.ndarray) -> None:
        """The aligned face is square and the declared size."""
        face = YuNetDetector().detect(portrait)[0]
        aligned = FaceAligner(512).align(portrait, face)
        assert aligned is not None
        assert aligned.image.shape == (512, 512, 3)

    def test_landmarks_land_on_the_template(self, portrait: np.ndarray) -> None:
        """Applying the transform maps the landmarks onto the template.

        This is the property that makes the restoration geometrically valid;
        without it the network sees a face in the wrong frame.
        """
        face = YuNetDetector().detect(portrait)[0]
        aligned = FaceAligner(512).align(portrait, face)
        assert aligned is not None

        homogeneous = np.hstack(
            [face.landmarks, np.ones((5, 1), dtype=np.float32)]
        )
        mapped = homogeneous @ aligned.affine.T
        error = np.linalg.norm(mapped - FFHQ_TEMPLATE_512, axis=1)
        # A similarity transform cannot fit five points exactly; a few pixels
        # of residual in a 512 px frame is the expected fit quality.
        assert error.max() < 25.0, error.tolist()

    @pytest.mark.parametrize("face_size", [256, 512])
    def test_face_size_scales_template(
        self, portrait: np.ndarray, face_size: int
    ) -> None:
        """A different canonical size scales the template consistently."""
        face = YuNetDetector().detect(portrait)[0]
        aligned = FaceAligner(face_size).align(portrait, face)
        assert aligned is not None
        assert aligned.image.shape == (face_size, face_size, 3)

    def test_paste_back_round_trip(self, portrait: np.ndarray) -> None:
        """Pasting an unmodified aligned face back reproduces the source."""
        face = YuNetDetector().detect(portrait)[0]
        aligner = FaceAligner(512)
        aligned = aligner.align(portrait, face)
        assert aligned is not None

        restored = aligner.paste_back(portrait, aligned.image, aligned)
        error = np.abs(restored.astype(float) - portrait.astype(float))
        # Only resampling error should remain.
        assert error.mean() < 2.0, error.mean()

    def test_paste_back_leaves_the_rest_untouched(
        self, portrait: np.ndarray
    ) -> None:
        """Blending affects only the face region."""
        face = YuNetDetector().detect(portrait)[0]
        aligner = FaceAligner(512)
        aligned = aligner.align(portrait, face)
        assert aligned is not None

        # Replace the aligned face with flat magenta, then check a far corner.
        loud = np.full_like(aligned.image, 255)
        loud[..., 1] = 0
        result = aligner.paste_back(portrait, loud, aligned)
        corner = np.abs(
            result[-40:, -40:].astype(float) - portrait[-40:, -40:].astype(float)
        )
        assert corner.max() < 1.0

    def test_inverse_transform_is_consistent(self, portrait: np.ndarray) -> None:
        """The inverse affine undoes the forward one."""
        face = YuNetDetector().detect(portrait)[0]
        aligned = FaceAligner(512).align(portrait, face)
        assert aligned is not None

        forward = np.vstack([aligned.affine, [0, 0, 1]])
        inverse = np.vstack([aligned.inverse_affine, [0, 0, 1]])
        identity = forward @ inverse
        assert np.allclose(identity, np.eye(3), atol=1e-3)


class TestCodeFormerAdapter:
    """The adapter's declarations and guard rails."""

    def test_registered_and_marked_synthetic(self, registry) -> None:
        """CodeFormer is registered as a neural, synthesising model."""
        info = registry.info("codeformer")
        assert info is not None
        assert info.task == TaskType.FACE_RESTORATION.value
        assert info.kind == ModelKind.NEURAL.value
        assert info.may_synthesise is True

    def test_declares_both_weight_files(self, registry) -> None:
        """It declares the network *and* the detector it depends on."""
        info = registry.info("codeformer")
        filenames = {spec.filename for spec in info.weights}
        assert "codeformer.pth" in filenames
        assert YUNET_WEIGHT.filename in filenames

    def test_licence_records_non_commercial_restriction(self, registry) -> None:
        """The S-Lab non-commercial restriction is surfaced, not buried."""
        info = registry.info("codeformer")
        assert "NON-COMMERCIAL" in info.license_name.upper()

    def test_warning_text_is_explicit(self, registry) -> None:
        """The method text states that the face is generated, not recovered."""
        info = registry.info("codeformer")
        text = info.method.lower()
        assert "synthesis" in text or "synthesised" in text or "generates" in text
        assert "identification" in text

    def test_gfpgan_still_reports_its_blocker(self, registry) -> None:
        """GFPGAN remains declared, unavailable and explained."""
        model = registry.try_get("gfpgan")
        assert model is not None
        state = model.availability()
        assert state.status == ModelStatus.NOT_INTEGRATED.value
        assert "StyleGAN2" in state.reason
        assert "CodeFormer" in state.reason
        with pytest.raises(ModelNotAvailableError):
            model.process(np.zeros((16, 16, 3), dtype=np.float32))


@pytest.fixture(scope="module")
def model(registry):
    """A loaded CodeFormer instance on the CPU, shared across the module."""
    instance = registry.get("codeformer")
    instance.load(device="cpu", fp16=False)
    yield instance
    instance.unload()


@yunet_installed
@codeformer_installed
class TestCodeFormerRestoration:
    """End-to-end restoration. Requires both weight files."""

    def test_no_face_raises_rather_than_no_op(self, model) -> None:
        """A frame without a face raises a clear error instead of returning it.

        Silently returning the input would leave a derivative in the case whose
        provenance claims a face was restored.
        """
        blank = np.full((256, 256, 3), 0.5, dtype=np.float32)
        with pytest.raises(InferenceError) as excinfo:
            model.process(blank)
        assert "no face" in str(excinfo.value).lower()

    def test_restores_and_preserves_geometry(self, model, degraded) -> None:
        """Output keeps the input's size, dtype and range."""
        source = degraded.astype(np.float32) / 255.0
        output = model.process(source, fidelity=0.7)
        assert output.shape == source.shape
        assert output.dtype == np.float32
        assert 0.0 <= output.min() and output.max() <= 1.0

    def test_records_the_faces_it_changed(self, model, degraded) -> None:
        """Each restored face is recorded with its information measure."""
        model.process(degraded.astype(np.float32) / 255.0, fidelity=0.7)
        faces = model.last_faces
        assert len(faces) >= 1
        record = faces[0]
        assert "inter_ocular_distance" in record
        assert "low_information" in record
        assert record["fidelity"] == pytest.approx(0.7)

    def test_only_the_face_region_changes(self, model, degraded) -> None:
        """Restoration is confined to the face; the background is untouched."""
        source = degraded.astype(np.float32) / 255.0
        output = model.process(source, fidelity=0.7)
        corner = np.abs(output[-48:, -48:] - source[-48:, -48:])
        assert corner.max() < 0.02, corner.max()

    def test_fidelity_changes_the_result(self, model, degraded) -> None:
        """The fidelity weight measurably steers the output.

        Higher fidelity must stay closer to the measured input - that is the
        control an examiner uses to see how much of the result is the prior.
        """
        source = degraded.astype(np.float32) / 255.0
        low = model.process(source, fidelity=0.0)
        high = model.process(source, fidelity=1.0)

        assert not np.allclose(low, high, atol=1e-3)
        distance_low = float(np.mean(np.abs(low - source)))
        distance_high = float(np.mean(np.abs(high - source)))
        assert distance_high < distance_low, (distance_low, distance_high)

    def test_output_is_sharper_than_input(self, model, degraded) -> None:
        """The restored face carries far more high-frequency content.

        That extra content is exactly what the synthesis warning is about: it
        was generated, not recovered.
        """
        source = degraded.astype(np.float32) / 255.0
        output = model.process(source, fidelity=0.5)

        def detail(array: np.ndarray) -> float:
            gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_32F).var())

        assert detail(output) > detail(source) * 3.0
