"""Tests for the degradation analyzers.

Each analyzer is checked for *responsiveness* (a stronger degradation must
produce a higher score) and for *specificity* (a degradation must not strongly
trigger unrelated indicators). Absolute score values are deliberately not
asserted, because they are calibration choices rather than ground truth.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.analyzer import AnalysisReport, DegradationAnalyzer, analyze_image
from analysis.base import MetricResult, clamp01, linear_map, log_map, to_gray
from analysis.blur import analyze_blur, analyze_motion_blur, perceptual_blur
from analysis.contrast import analyze_contrast
from analysis.exposure import analyze_exposure
from analysis.haze import analyze_haze
from analysis.jpeg import analyze_jpeg, blockiness, dezigzag, estimate_jpeg_quality
from analysis.noise import analyze_noise, estimate_sigma_immerkaer
from analysis.resolution import analyze_resolution, oversampling_ratio
from app.constants import DegradationKey
from core.image_io import ImageData


@pytest.fixture(scope="module")
def base_scene() -> np.ndarray:
    """A textured scene large enough for every estimator."""
    from scripts.make_sample import build_scene

    return build_scene(640, 448, seed=3)


class TestHelpers:
    """Scoring helpers."""

    def test_clamp01(self) -> None:
        """Values are clamped and NaN maps to zero."""
        assert clamp01(-1.0) == 0.0
        assert clamp01(2.0) == 1.0
        assert clamp01(float("nan")) == 0.0

    def test_linear_map_direction(self) -> None:
        """Mapping respects the declared endpoints and inversion."""
        assert linear_map(0.0, 0.0, 1.0) == 0.0
        assert linear_map(1.0, 0.0, 1.0) == 1.0
        assert linear_map(0.0, 0.0, 1.0, invert=True) == 1.0

    def test_log_map_monotonic(self) -> None:
        """The logarithmic map is monotonic across decades."""
        values = [log_map(v, 1e-5, 1e-1) for v in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
        assert values == sorted(values)

    def test_to_gray_range(self, base_scene: np.ndarray) -> None:
        """Luminance is single-channel and normalised."""
        gray = to_gray(base_scene)
        assert gray.ndim == 2
        assert 0.0 <= gray.min() and gray.max() <= 1.0


class TestBlur:
    """Blur and motion-blur estimation."""

    def test_monotonic_in_blur_radius(self, base_scene: np.ndarray) -> None:
        """More Gaussian blur yields a higher score."""
        scores = []
        for kernel in (1, 5, 11, 21):
            blurred = (
                cv2.GaussianBlur(base_scene, (kernel, kernel), 0)
                if kernel > 1 else base_scene
            )
            scores.append(analyze_blur(blurred).score)
        assert scores == sorted(scores), scores
        assert scores[-1] - scores[0] > 0.3

    def test_perceptual_blur_bounded(self, base_scene: np.ndarray) -> None:
        """The Crete metric stays inside [0, 1]."""
        value, horizontal, vertical = perceptual_blur(to_gray(base_scene))
        assert 0.0 <= value <= 1.0
        assert 0.0 <= horizontal <= 1.0
        assert 0.0 <= vertical <= 1.0

    def test_motion_blur_detected(self, base_scene: np.ndarray) -> None:
        """A long linear blur scores higher than an isotropic one."""
        length = 21
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        motion = cv2.filter2D(base_scene, -1, kernel)
        isotropic = cv2.GaussianBlur(base_scene, (21, 21), 0)
        assert analyze_motion_blur(motion).score > analyze_motion_blur(isotropic).score

    def test_motion_angle_estimate(self, base_scene: np.ndarray) -> None:
        """The reported angle is near the true horizontal motion direction."""
        length = 21
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        result = analyze_motion_blur(cv2.filter2D(base_scene, -1, kernel))
        angle = result.measurements["estimated_angle_deg"]
        assert min(angle, 180 - angle) < 25.0, angle

    def test_measurements_recorded(self, base_scene: np.ndarray) -> None:
        """Raw measurements accompany the score."""
        result = analyze_blur(base_scene)
        assert "laplacian_variance" in result.measurements
        assert "perceptual_blur" in result.measurements
        assert result.method and result.reference


class TestNoise:
    """Noise estimation."""

    def test_monotonic_in_sigma(self, base_scene: np.ndarray) -> None:
        """More additive noise yields a higher score."""
        rng = np.random.default_rng(0)
        scores = []
        for sigma in (0, 4, 12, 28):
            noisy = np.clip(
                base_scene.astype(np.float32) + rng.normal(0, sigma, base_scene.shape),
                0, 255,
            ).astype(np.uint8)
            scores.append(analyze_noise(noisy).score)
        assert scores == sorted(scores), scores

    def test_sigma_estimate_tracks_truth(self, base_scene: np.ndarray) -> None:
        """The Immerkaer estimate tracks the injected sigma."""
        rng = np.random.default_rng(1)
        noisy = np.clip(
            base_scene.astype(np.float32) + rng.normal(0, 20, base_scene.shape), 0, 255
        ).astype(np.uint8)
        estimated = analyze_noise(noisy).measurements["sigma_luma_8bit_equivalent"]
        # Scene texture inflates the estimate; require the right order of
        # magnitude rather than an exact match.
        assert 8.0 < estimated < 40.0, estimated

    def test_flat_image_is_clean(self) -> None:
        """A perfectly flat image reports essentially no noise."""
        flat = np.full((128, 128, 3), 120, dtype=np.uint8)
        assert analyze_noise(flat).score < 0.1
        assert estimate_sigma_immerkaer(to_gray(flat)) < 1e-3


class TestJpeg:
    """JPEG artefact estimation."""

    def test_dezigzag_identity(self) -> None:
        """De-zigzagging maps position 0 to the DC coefficient."""
        table = list(range(64))
        natural = dezigzag(table)
        assert natural.shape == (8, 8)
        assert natural[0, 0] == 0
        assert natural[0, 1] == 1
        assert natural[1, 0] == 2

    def test_quality_recovery(self, base_scene: np.ndarray, tmp_path: Path) -> None:
        """The encoder's quality factor is recovered from the tables."""
        for quality in (90, 70, 40):
            path = tmp_path / f"q{quality}.jpg"
            cv2.imwrite(
                str(path), base_scene[..., ::-1],
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            info = estimate_jpeg_quality(path)
            assert info is not None
            assert abs(info["quality"] - quality) < 6, (quality, info["quality"])

    def test_quality_none_for_png(self, sample_png: Path) -> None:
        """A PNG has no quantisation tables."""
        assert estimate_jpeg_quality(sample_png) is None

    def test_monotonic_in_compression(
        self, base_scene: np.ndarray, tmp_path: Path
    ) -> None:
        """Harsher compression yields a higher score."""
        scores = []
        for quality in (95, 60, 20):
            path = tmp_path / f"c{quality}.jpg"
            cv2.imwrite(
                str(path), base_scene[..., ::-1],
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            decoded = cv2.imread(str(path))[..., ::-1]
            scores.append(analyze_jpeg(decoded, path).score)
        assert scores == sorted(scores), scores

    def test_blockiness_separates_compressed_from_clean(
        self, base_scene: np.ndarray, tmp_path: Path
    ) -> None:
        """Compressing the *same* content multiplies the blockiness measure.

        The absolute value on uncompressed content is content-dependent, so the
        meaningful property is the separation between a frame and a compressed
        copy of itself.
        """
        clean, _, _ = blockiness(to_gray(base_scene))
        path = tmp_path / "compressed.jpg"
        cv2.imwrite(
            str(path), base_scene[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 25]
        )
        compressed, _, _ = blockiness(
            to_gray(cv2.imread(str(path))[..., ::-1])
        )
        assert compressed > clean * 3.0, (clean, compressed)

    @pytest.mark.parametrize(
        "name",
        ["flat", "noise", "gradient", "blurred", "coarse_grid"],
    )
    def test_no_false_blocking_on_unblocked_content(
        self, name: str, base_scene: np.ndarray
    ) -> None:
        """Content without an 8-pixel periodicity reports no blocking.

        ``coarse_grid`` is the case a naive boundary-versus-interior estimator
        fails: hard axis-aligned edges at a non-8 pitch land on the block phase
        as often as anywhere else.
        """
        rng = np.random.default_rng(0)
        if name == "flat":
            image = cv2.GaussianBlur(
                np.full((256, 256, 3), 100, dtype=np.uint8), (31, 31), 0
            )
        elif name == "noise":
            image = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
        elif name == "gradient":
            row = np.linspace(0, 255, 256, dtype=np.uint8)
            image = np.tile(row[None, :, None], (256, 1, 3))
        elif name == "blurred":
            image = cv2.GaussianBlur(base_scene, (15, 15), 0)
        else:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            for offset in range(0, 256, 30):
                cv2.line(image, (offset, 0), (offset, 255), (255, 255, 255), 2)
                cv2.line(image, (0, offset), (255, offset), (255, 255, 255), 2)

        excess, _, _ = blockiness(to_gray(image))
        assert excess < 0.2, (name, excess)


class TestResolution:
    """Resolution adequacy."""

    def test_small_scores_higher(self, base_scene: np.ndarray) -> None:
        """A smaller frame scores higher on the low-resolution indicator."""
        small = cv2.resize(base_scene, (160, 112), interpolation=cv2.INTER_AREA)
        assert analyze_resolution(small).score > analyze_resolution(base_scene).score

    @pytest.mark.parametrize("factor", [2, 3, 4])
    @pytest.mark.parametrize(
        "interpolation", [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR]
    )
    def test_upscaling_detected(
        self, base_scene: np.ndarray, factor: int, interpolation: int
    ) -> None:
        """Interpolated enlargements are flagged across kernels and factors.

        The 640x448 working size is comfortably inside the estimator's reliable
        range; see :meth:`test_detection_limit_is_documented` for the boundary.
        """
        small = cv2.resize(
            base_scene,
            (640 // factor, 448 // factor),
            interpolation=cv2.INTER_AREA,
        )
        upscaled = cv2.resize(small, (640, 448), interpolation=interpolation)
        result = analyze_resolution(upscaled)
        assert result.measurements["appears_upscaled"] is True, (
            factor, result.measurements["oversampling_ratio"]
        )

    def test_native_frame_not_flagged(self, base_scene: np.ndarray) -> None:
        """A natively sampled frame is not reported as interpolated."""
        assert analyze_resolution(base_scene).measurements["appears_upscaled"] is False

    @pytest.mark.parametrize(
        "path",
        sorted(p for p in Path("samples").glob("sample_*") if "upscal" not in p.stem),
    )
    def test_no_false_positive_on_degraded_natives(self, path: Path) -> None:
        """No natively captured frame - however degraded - is called upscaled.

        A false positive here would attach a misleading provenance claim to
        genuine evidence, so the threshold is deliberately biased towards
        missing aggressive enlargements rather than inventing them.
        """
        from core.image_io import load_image

        result = analyze_resolution(load_image(path).pixels)
        assert result.measurements["appears_upscaled"] is False, path.name

    def test_detection_limit_is_documented(self, base_scene: np.ndarray) -> None:
        """Record the estimator's known blind spot.

        Blur and interpolation are both low-pass operations, so a heavily
        blurred native frame and an aggressively enlarged one are not separable
        by outer-band energy alone. This test pins the boundary so a future
        change to the threshold cannot silently move it.
        """
        heavily_blurred = cv2.GaussianBlur(base_scene, (21, 21), 0)
        native_ratio = oversampling_ratio(to_gray(heavily_blurred))

        small = cv2.resize(base_scene, (160, 112), interpolation=cv2.INTER_AREA)
        upscaled_4x = cv2.resize(small, (640, 448), interpolation=cv2.INTER_CUBIC)
        upscaled_ratio = oversampling_ratio(to_gray(upscaled_4x))

        # The two populations are close enough that only a conservative
        # threshold is defensible.
        assert native_ratio > upscaled_ratio

    def test_oversampling_ratio_separates(self, base_scene: np.ndarray) -> None:
        """The annulus ratio separates native from interpolated by a decade."""
        small = cv2.resize(base_scene, (160, 112), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(small, (640, 448), interpolation=cv2.INTER_CUBIC)
        native = oversampling_ratio(to_gray(base_scene))
        interpolated = oversampling_ratio(to_gray(upscaled))
        assert native > interpolated * 10, (native, interpolated)


class TestExposure:
    """Exposure and clipping."""

    def test_dark_and_bright(self, base_scene: np.ndarray) -> None:
        """Darkening raises underexposure; brightening raises overexposure."""
        dark = np.clip(base_scene.astype(np.float32) * 0.2, 0, 255).astype(np.uint8)
        bright = np.clip(
            base_scene.astype(np.float32) * 2.4 + 40, 0, 255
        ).astype(np.uint8)

        under_dark, over_dark = analyze_exposure(dark)
        under_bright, over_bright = analyze_exposure(bright)

        assert under_dark.score > 0.4
        assert over_dark.score < 0.2
        assert over_bright.score > 0.4
        assert under_bright.score < 0.2

    def test_clipping_fractions(self) -> None:
        """Clipping fractions are measured on the extreme channels."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50] = 255
        under, over = analyze_exposure(image)
        assert abs(over.measurements["highlight_clipped_fraction"] - 0.5) < 0.02
        assert abs(under.measurements["shadow_clipped_fraction"] - 0.5) < 0.02


class TestContrastAndHaze:
    """Contrast and haze indicators."""

    def test_low_contrast_detected(self, base_scene: np.ndarray) -> None:
        """Compressing the tonal range raises the low-contrast score."""
        compressed = (base_scene.astype(np.float32) * 0.25 + 96).astype(np.uint8)
        assert analyze_contrast(compressed).score > analyze_contrast(base_scene).score

    def test_haze_detected(self, base_scene: np.ndarray) -> None:
        """A synthetic airlight veil raises the haze score."""
        depth = np.linspace(0.4, 0.85, base_scene.shape[0], dtype=np.float32)
        depth = depth[:, None, None]
        airlight = np.array([230.0, 234.0, 240.0], dtype=np.float32)
        hazy = np.clip(
            base_scene.astype(np.float32) * (1 - depth) + airlight * depth, 0, 255
        ).astype(np.uint8)
        assert analyze_haze(hazy).score > 0.6
        assert analyze_haze(base_scene).score < 0.45

    def test_bright_image_is_not_hazy(self, base_scene: np.ndarray) -> None:
        """Over-exposure alone must not read as haze."""
        bright = np.clip(
            base_scene.astype(np.float32) * 2.4 + 40, 0, 255
        ).astype(np.uint8)
        assert analyze_haze(bright).score < 0.45

    def test_blur_is_not_hazy(self, base_scene: np.ndarray) -> None:
        """Optical blur alone must not read as haze."""
        blurred = cv2.GaussianBlur(base_scene, (15, 15), 0)
        assert analyze_haze(blurred).score < 0.45


class TestAnalyzer:
    """The orchestrator."""

    def test_all_indicators_present(self, base_scene: np.ndarray) -> None:
        """Every declared indicator is produced."""
        from app.constants import DEGRADATION_ORDER

        report = analyze_image(ImageData(pixels=base_scene))
        for key in DEGRADATION_ORDER:
            assert key in report.metrics, key

    def test_scores_bounded(self, base_scene: np.ndarray) -> None:
        """Every score lies inside [0, 1]."""
        report = analyze_image(ImageData(pixels=base_scene))
        for metric in report.metrics.values():
            assert 0.0 <= metric.score <= 1.0

    def test_serialisation_round_trip(self, base_scene: np.ndarray) -> None:
        """A report survives a dict round trip."""
        original = analyze_image(ImageData(pixels=base_scene))
        restored = AnalysisReport.from_dict(original.to_dict())
        assert set(restored.metrics) == set(original.metrics)
        assert abs(
            restored.score(DegradationKey.BLUR.value)
            - original.score(DegradationKey.BLUR.value)
        ) < 1e-4

    def test_disclaimer_in_output(self, base_scene: np.ndarray) -> None:
        """The heuristic caveat is embedded in the serialised report."""
        from app.constants import HEURISTIC_DISCLAIMER

        report = analyze_image(ImageData(pixels=base_scene))
        assert report.to_dict()["disclaimer"] == HEURISTIC_DISCLAIMER

    def test_cancellation(self, base_scene: np.ndarray) -> None:
        """A cancelled analysis stops early instead of completing."""
        report = analyze_image(
            ImageData(pixels=base_scene), cancelled=lambda: True
        )
        assert len(report.metrics) < 9

    def test_progress_reported(self, base_scene: np.ndarray) -> None:
        """Progress is reported monotonically and reaches 100."""
        seen = []
        analyze_image(
            ImageData(pixels=base_scene),
            progress=lambda percent, message: seen.append(percent),
        )
        assert seen == sorted(seen)
        assert seen[-1] == 100

    def test_overall_severity(self, base_scene: np.ndarray) -> None:
        """The weighted summary is bounded."""
        analyzer = DegradationAnalyzer()
        report = analyzer.analyze(ImageData(pixels=base_scene))
        assert 0.0 <= analyzer.overall_severity(report) <= 1.0

    def test_tiny_image_does_not_crash(self) -> None:
        """A very small frame is analysed without raising."""
        tiny = np.full((12, 14, 3), 128, dtype=np.uint8)
        report = analyze_image(ImageData(pixels=tiny))
        assert report.metrics

    def test_grayscale_input(self, base_scene: np.ndarray) -> None:
        """A monochrome frame is handled by every indicator."""
        gray = cv2.cvtColor(base_scene, cv2.COLOR_RGB2GRAY)
        report = analyze_image(ImageData(pixels=gray))
        assert report.metrics
        haze = report.get(DegradationKey.HAZE.value)
        assert haze is not None
        assert haze.measurements["monochrome_source"] is True
