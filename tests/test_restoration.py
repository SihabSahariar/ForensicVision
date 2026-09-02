"""Tests for the restoration engine: registry, tiling, pipelines, auto engine."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.analyzer import analyze_image
from app.constants import ModelKind, ModelStatus, TaskType
from core.exceptions import ModelNotAvailableError, OperationCancelled, PipelineError
from core.image_io import ImageData
from restoration.auto_engine import AutoRestorationEngine
from restoration.base import ModelInfo, ParamSpec, WeightSpec
from restoration.classical import deconvolution, enhance, psf
from restoration.pipeline import Pipeline, PipelineRunner, PipelineStep
from restoration.registry import ModelRegistry
from restoration.tiling import estimate_tile_size, pad_to_multiple, tiled_process


@pytest.fixture(scope="module")
def scene_float() -> np.ndarray:
    """A float RGB test scene in [0, 1]."""
    from scripts.make_sample import build_scene

    return build_scene(320, 224, seed=5).astype(np.float32) / 255.0


class TestParamSpec:
    """Parameter declaration and clamping."""

    def test_clamps_numeric(self) -> None:
        """Numeric values are clamped into range."""
        spec = ParamSpec("x", "X", kind="float", default=1.0, minimum=0.0, maximum=2.0)
        assert spec.clamp(5.0) == 2.0
        assert spec.clamp(-1.0) == 0.0
        assert spec.clamp(1.5) == 1.5

    def test_int_rounds(self) -> None:
        """Integer parameters round."""
        spec = ParamSpec("n", "N", kind="int", default=4, minimum=1, maximum=10)
        assert spec.clamp(4.6) == 5

    def test_choice_falls_back(self) -> None:
        """An invalid choice falls back to the default."""
        spec = ParamSpec(
            "s", "S", kind="choice", default=2, choices=((2, "2x"), (4, "4x"))
        )
        assert spec.clamp(4) == 4
        assert spec.clamp(99) == 2

    def test_bool(self) -> None:
        """Boolean parameters coerce."""
        spec = ParamSpec("b", "B", kind="bool", default=False)
        assert spec.clamp(1) is True

    def test_sanitise_fills_defaults(self) -> None:
        """Missing parameters are filled from the defaults."""
        info = ModelInfo(
            name="t", display_name="T", task=TaskType.GENERIC.value,
            parameters=(
                ParamSpec("a", "A", default=1.0, minimum=0.0, maximum=2.0),
                ParamSpec("b", "B", default=0.5, minimum=0.0, maximum=1.0),
            ),
        )
        assert info.sanitise({"a": 9.0}) == {"a": 2.0, "b": 0.5}


class TestRegistry:
    """Model registration and discovery."""

    def test_classical_models_registered(self, registry) -> None:
        """Every classical operator is present and usable."""
        for name in ("lanczos", "richardson_lucy", "wiener", "unsharp", "nlm",
                     "bilateral", "deblock", "dcp_dehaze", "exposure", "clahe"):
            model = registry.try_get(name)
            assert model is not None, name
            assert model.availability().ok, name

    def test_tasks_covered(self, registry) -> None:
        """Every core task has at least one registered model."""
        for task in (
            TaskType.SUPER_RESOLUTION.value, TaskType.DEBLUR.value,
            TaskType.DENOISE.value, TaskType.JPEG_ARTIFACT.value,
        ):
            assert registry.by_task(task), task

    def test_unknown_model_raises(self, registry) -> None:
        """Requesting an unregistered model raises."""
        with pytest.raises(ModelNotAvailableError):
            registry.get("no_such_model")

    def test_status_table_complete(self, registry) -> None:
        """Every model reports a status with the required fields."""
        rows = registry.status_table()
        assert rows
        for row in rows:
            assert row["status"] in {s.value for s in ModelStatus}
            assert row["display_name"]
            assert row["kind"] in {k.value for k in ModelKind}

    def test_classical_models_never_synthesise(self, registry) -> None:
        """No deterministic operator is flagged as generative."""
        for info in registry.infos():
            if info.kind == ModelKind.CLASSICAL.value:
                assert info.may_synthesise is False, info.name

    def test_neural_models_declare_licence(self, registry) -> None:
        """Every neural model carries licence and repository provenance."""
        for info in registry.infos():
            if info.kind == ModelKind.NEURAL.value:
                assert info.license_name, info.name
                assert info.repository, info.name

    @pytest.mark.parametrize("name", ["gfpgan", "lama"])
    def test_not_integrated_models_refuse_to_run(self, registry, name: str) -> None:
        """A declared-but-unintegrated model raises instead of returning output.

        CodeFormer is deliberately absent from this list: it is now fully
        integrated, and its behaviour is covered by ``tests/test_face.py``.
        """
        model = registry.try_get(name)
        assert model is not None
        state = model.availability()
        assert state.status == ModelStatus.NOT_INTEGRATED.value
        assert state.reason, "an unintegrated model must explain why"
        with pytest.raises(ModelNotAvailableError):
            model.process(np.zeros((16, 16, 3), dtype=np.float32))

    def test_every_declared_model_is_either_usable_or_explained(
        self, registry
    ) -> None:
        """No model is ever silently unavailable.

        Whatever a model's state, the investigator must get either a working
        model or a specific reason it cannot run - never a blank status or a
        substitute result.
        """
        for row in registry.status_table():
            if row["status"] == ModelStatus.INSTALLED.value:
                continue
            assert row["reason"], row["name"]


class TestClassicalOperators:
    """The deterministic operators."""

    def test_all_preserve_shape_and_range(self, registry, scene_float) -> None:
        """Every classical operator returns a valid image."""
        for info in registry.infos():
            if info.kind != ModelKind.CLASSICAL.value:
                continue
            model = registry.get(info.name)
            output = model.process(scene_float)
            assert output.dtype == np.float32, info.name
            assert output.ndim == 3 and output.shape[2] == 3, info.name
            assert 0.0 <= output.min() and output.max() <= 1.0, info.name
            expected = (
                scene_float.shape[0] * info.scale,
                scene_float.shape[1] * info.scale,
            )
            if info.name != "lanczos":
                assert output.shape[:2] == expected, info.name

    def test_lanczos_scales(self, registry, scene_float) -> None:
        """Lanczos honours the requested scale factor."""
        model = registry.get("lanczos")
        for scale in (2, 3, 4):
            output = model.process(scene_float, scale=scale)
            assert output.shape[0] == scene_float.shape[0] * scale
            assert output.shape[1] == scene_float.shape[1] * scale

    @staticmethod
    def _psnr(a: np.ndarray, b: np.ndarray) -> float:
        """Peak signal-to-noise ratio between two float images in [0, 1]."""
        mse = float(np.mean((a - b) ** 2))
        return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))

    @staticmethod
    def _blur(truth: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply ``kernel`` to ``truth`` with reflected borders."""
        return cv2.filter2D(truth, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    @staticmethod
    def _kernel(kind: str) -> np.ndarray:
        """Return the PSF used by the deconvolution tests."""
        if kind == "gaussian":
            return psf.gaussian_psf(2.0)
        if kind == "motion":
            return psf.motion_psf(15.0, 0.0)
        return psf.disk_psf(3.0)

    @pytest.mark.parametrize("kind", ["gaussian", "motion", "disk"])
    def test_richardson_lucy_recovers_all_psf_types(self, kind: str) -> None:
        """Richardson-Lucy improves PSNR for every supported blur model.

        A full-size scene is used deliberately: a small, already band-limited
        frame loses very little to a modest blur, so there is almost nothing
        for deconvolution to recover and the test would measure noise.
        """
        from scripts.make_sample import build_scene

        truth = build_scene(960, 640, seed=5).astype(np.float32) / 255.0
        kernel = self._kernel(kind)
        blurred = self._blur(truth, kernel)
        baseline = self._psnr(blurred, truth)
        restored = deconvolution.richardson_lucy(blurred, kernel, iterations=25)
        assert self._psnr(restored, truth) > baseline + 1.0, kind

    @pytest.mark.parametrize("kind", ["gaussian", "motion"])
    def test_wiener_recovers_invertible_psfs(self, kind: str) -> None:
        """Wiener improves PSNR for blur models without transfer-function zeros."""
        from scripts.make_sample import build_scene

        truth = build_scene(960, 640, seed=5).astype(np.float32) / 255.0
        kernel = self._kernel(kind)
        blurred = self._blur(truth, kernel)
        baseline = self._psnr(blurred, truth)
        restored = deconvolution.wiener_deconvolution(blurred, kernel, 0.002)
        assert self._psnr(restored, truth) > baseline + 1.0, kind

    def test_wiener_unsuitable_for_disk_is_documented(self, registry) -> None:
        """The disk-PSF limitation is stated in the model's own documentation.

        A pillbox transfer function has exact zeros. Wiener cannot invert those
        frequencies and amplifies noise there instead, so it can score *worse*
        than the blurred input. That is a property of the algorithm rather than
        a defect, and the examiner has to be told - so the guidance is asserted
        here alongside the measurement that motivates it.
        """
        from scripts.make_sample import build_scene

        truth = build_scene(960, 640, seed=5).astype(np.float32) / 255.0
        kernel = psf.disk_psf(3.0)
        blurred = self._blur(truth, kernel)

        wiener = deconvolution.wiener_deconvolution(blurred, kernel, 0.002)
        lucy = deconvolution.richardson_lucy(blurred, kernel, iterations=25)
        assert self._psnr(lucy, truth) > self._psnr(wiener, truth)

        info = registry.info("wiener")
        assert info is not None
        assert "defocus" in info.method.lower()
        assert "richardson-lucy" in info.method.lower()

    def test_richardson_lucy_noise_amplification(self) -> None:
        """RL on noisy input peaks then degrades - the documented behaviour."""
        from scripts.make_sample import build_scene

        truth = build_scene(480, 336, seed=5).astype(np.float32) / 255.0
        kernel = psf.gaussian_psf(2.0)
        rng = np.random.default_rng(0)
        noisy = np.clip(
            cv2.filter2D(truth, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
            + rng.normal(0, 0.01, truth.shape),
            0, 1,
        ).astype(np.float32)

        scores = [
            self._psnr(
                deconvolution.richardson_lucy(noisy, kernel, iterations=n), truth
            )
            for n in (10, 40, 160)
        ]
        assert scores[2] < max(scores), scores

    def test_psf_kernels_normalised(self) -> None:
        """Every PSF has unit sum, so deconvolution preserves brightness."""
        for kernel in (
            psf.gaussian_psf(2.0), psf.motion_psf(11, 30.0), psf.disk_psf(4.0)
        ):
            assert abs(float(kernel.sum()) - 1.0) < 1e-5
            assert kernel.shape[0] % 2 == 1

    def test_denoise_reduces_noise(self, scene_float) -> None:
        """Non-local means lowers the estimated noise sigma."""
        from analysis.noise import estimate_sigma_immerkaer
        from analysis.base import to_gray

        rng = np.random.default_rng(0)
        noisy = np.clip(
            scene_float + rng.normal(0, 0.06, scene_float.shape), 0, 1
        ).astype(np.float32)
        denoised = enhance.nlm_denoise(noisy, 10.0, 10.0)
        assert estimate_sigma_immerkaer(to_gray(denoised)) < estimate_sigma_immerkaer(
            to_gray(noisy)
        )

    def test_dehaze_restores_contrast(self, scene_float) -> None:
        """Dehazing a synthetic veil raises the dynamic range."""
        depth = np.full(scene_float.shape[:2] + (1,), 0.6, dtype=np.float32)
        airlight = np.array([0.9, 0.92, 0.94], dtype=np.float32)
        hazy = np.clip(
            scene_float * (1 - depth) + airlight * depth, 0, 1
        ).astype(np.float32)
        restored = enhance.dark_channel_dehaze(hazy)
        assert float(restored.std()) > float(hazy.std())

    def test_rejects_wrong_shape(self, registry, scene_float) -> None:
        """A non-RGB input is rejected rather than silently coerced."""
        model = registry.get("unsharp")
        with pytest.raises(ValueError):
            model.process(scene_float[..., 0])


class TestTiling:
    """Tiled execution."""

    def test_identity_reconstructs(self) -> None:
        """Tiling an identity function reproduces the input."""
        image = np.random.default_rng(0).random((200, 317, 3)).astype(np.float32)
        result = tiled_process(image, lambda t: t, scale=1, tile_size=64, overlap=16)
        assert result.shape == image.shape
        assert float(np.abs(result - image).max()) < 1e-5

    @pytest.mark.parametrize("scale", [1, 2, 4])
    def test_scaled_geometry(self, scale: int) -> None:
        """Output geometry follows the declared scale factor."""
        image = np.random.default_rng(1).random((96, 128, 3)).astype(np.float32)

        def upscale(tile: np.ndarray) -> np.ndarray:
            return cv2.resize(
                tile,
                (tile.shape[1] * scale, tile.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )

        result = tiled_process(image, upscale, scale=scale, tile_size=48, overlap=8)
        assert result.shape == (96 * scale, 128 * scale, 3)

    def test_matches_whole_image(self) -> None:
        """Tiled output closely matches a single-pass result."""
        image = np.random.default_rng(2).random((160, 200, 3)).astype(np.float32)

        def blur(tile: np.ndarray) -> np.ndarray:
            return cv2.GaussianBlur(tile, (0, 0), 1.2)

        tiled = tiled_process(image, blur, tile_size=64, overlap=24)
        whole = blur(image)
        assert float(np.abs(tiled - whole).mean()) < 0.02

    def test_cancellation(self) -> None:
        """A cancelled tiled run raises."""
        image = np.zeros((128, 128, 3), dtype=np.float32)
        with pytest.raises(OperationCancelled):
            tiled_process(
                image, lambda t: t, tile_size=32, cancelled=lambda: True
            )

    def test_progress_reaches_completion(self) -> None:
        """Progress is reported and ends at 100."""
        image = np.zeros((128, 128, 3), dtype=np.float32)
        seen = []
        tiled_process(
            image, lambda t: t, tile_size=64,
            progress=lambda percent, message: seen.append(percent),
        )
        assert seen and seen[-1] == 100

    def test_oom_backoff(self) -> None:
        """An out-of-memory error triggers a retry at a smaller tile size."""
        image = np.zeros((256, 256, 3), dtype=np.float32)
        sizes_seen = []

        def flaky(tile: np.ndarray) -> np.ndarray:
            sizes_seen.append(max(tile.shape[:2]))
            if max(tile.shape[:2]) > 64:
                raise RuntimeError("CUDA out of memory")
            return tile

        result = tiled_process(
            image, flaky, tile_size=256, overlap=8, auto_reduce=True
        )
        assert result.shape == image.shape
        assert max(sizes_seen) > 64 and min(sizes_seen) <= 64

    def test_pad_to_multiple(self) -> None:
        """Padding rounds up to the requested multiple."""
        image = np.zeros((37, 53, 3), dtype=np.float32)
        padded, (pad_h, pad_w) = pad_to_multiple(image, 8)
        assert padded.shape[0] % 8 == 0 and padded.shape[1] % 8 == 0
        assert pad_h == 3 and pad_w == 3
        unchanged, pads = pad_to_multiple(image, 1)
        assert pads == (0, 0) and unchanged.shape == image.shape

    def test_estimate_tile_size_bounded(self) -> None:
        """Tile estimates stay within the supported range."""
        assert 64 <= estimate_tile_size(1024) <= 1024
        assert estimate_tile_size(0) >= 64


class TestPipeline:
    """Pipeline construction, validation and execution."""

    def test_empty_pipeline_invalid(self) -> None:
        """An empty pipeline reports an issue and refuses to run."""
        pipeline = Pipeline()
        assert pipeline.validate()
        with pytest.raises(PipelineError):
            PipelineRunner(device="cpu").run(
                ImageData(pixels=np.zeros((8, 8, 3), dtype=np.uint8)), pipeline
            )

    def test_unknown_model_reported(self, registry) -> None:
        """A missing model is named in the validation output."""
        pipeline = Pipeline(steps=[PipelineStep(model_name="nope")])
        issues = pipeline.validate()
        assert any("nope" in issue for issue in issues)

    def test_total_scale(self, registry) -> None:
        """Scale factors multiply across steps."""
        pipeline = Pipeline(
            steps=[
                PipelineStep("lanczos", {"scale": 2}),
                PipelineStep("lanczos", {"scale": 2}),
            ]
        )
        assert pipeline.total_scale == 4

    def test_ordering_advice(self, registry) -> None:
        """Super-resolution before cleanup produces an ordering warning."""
        pipeline = Pipeline(
            steps=[PipelineStep("lanczos", {"scale": 2}), PipelineStep("nlm")]
        )
        assert any("Super-resolution" in issue for issue in pipeline.validate())

    def test_disabled_steps_skipped(self, registry, scene_float) -> None:
        """A disabled step is retained but not executed."""
        pipeline = Pipeline(
            steps=[
                PipelineStep("unsharp", enabled=False),
                PipelineStep("clahe", enabled=True),
            ]
        )
        assert len(pipeline.steps) == 2
        assert len(pipeline.enabled_steps) == 1

    def test_run_records_hash_chain(self, registry, scene_float) -> None:
        """Each step's output digest becomes the next step's input digest."""
        pipeline = Pipeline(
            steps=[PipelineStep("clahe"), PipelineStep("unsharp"),
                   PipelineStep("lanczos", {"scale": 2})]
        )
        image = ImageData(pixels=(scene_float * 255).astype(np.uint8))
        result = PipelineRunner(device="cpu").run(image, pipeline)

        assert len(result.steps) == 3
        for earlier, later in zip(result.steps, result.steps[1:]):
            assert earlier.output_hashes.sha256 == later.input_hashes.sha256
        assert result.image.width == image.width * 2
        assert result.succeeded

    def test_classical_pipeline_not_synthetic(self, registry, scene_float) -> None:
        """A classical-only pipeline is not flagged as generative."""
        pipeline = Pipeline(steps=[PipelineStep("clahe"), PipelineStep("nlm")])
        assert pipeline.may_synthesise is False

    def test_alpha_preserved(self, registry) -> None:
        """Alpha survives a pipeline that changes geometry."""
        rgba = np.zeros((32, 40, 4), dtype=np.uint8)
        rgba[..., :3] = 120
        rgba[..., 3] = 200
        pipeline = Pipeline(steps=[PipelineStep("lanczos", {"scale": 2})])
        result = PipelineRunner(device="cpu").run(ImageData(pixels=rgba), pipeline)
        assert result.image.has_alpha
        assert result.image.width == 80
        assert int(result.image.pixels[..., 3].mean()) == 200

    def test_bit_depth_preserved(self, registry) -> None:
        """A 16-bit source yields a 16-bit derivative."""
        pixels = (np.random.default_rng(0).random((32, 32, 3)) * 65535).astype(np.uint16)
        pipeline = Pipeline(steps=[PipelineStep("unsharp")])
        result = PipelineRunner(device="cpu").run(ImageData(pixels=pixels), pipeline)
        assert result.image.dtype == np.uint16

    def test_roi_mask_limits_change(self, registry, scene_float) -> None:
        """A masked run leaves pixels outside the mask essentially unchanged."""
        image = ImageData(pixels=(scene_float * 255).astype(np.uint8))
        mask = np.zeros(image.pixels.shape[:2], dtype=np.uint8)
        mask[40:120, 40:120] = 255

        pipeline = Pipeline(steps=[PipelineStep("clahe", {"clip_limit": 8.0})])
        result = PipelineRunner(device="cpu").run(image, pipeline, roi_mask=mask)

        outside = np.abs(
            result.image.pixels[:20, :20].astype(int) - image.pixels[:20, :20].astype(int)
        )
        inside = np.abs(
            result.image.pixels[60:100, 60:100].astype(int)
            - image.pixels[60:100, 60:100].astype(int)
        )
        assert outside.mean() < 1.0
        assert inside.mean() > outside.mean()

    def test_roi_with_scaling_refused(self, registry, scene_float) -> None:
        """A geometry-changing pipeline cannot be region-limited."""
        image = ImageData(pixels=(scene_float * 255).astype(np.uint8))
        mask = np.zeros(image.pixels.shape[:2], dtype=np.uint8)
        mask[10:50, 10:50] = 255
        pipeline = Pipeline(steps=[PipelineStep("lanczos", {"scale": 2})])
        with pytest.raises(PipelineError):
            PipelineRunner(device="cpu").run(image, pipeline, roi_mask=mask)

    def test_serialisation_round_trip(self, registry) -> None:
        """A pipeline survives a dict round trip."""
        original = Pipeline(
            steps=[PipelineStep("clahe", {"clip_limit": 3.0}, note="test")],
            name="Test", rationale="because",
        )
        restored = Pipeline.from_dict(original.to_dict())
        assert restored.name == "Test"
        assert restored.steps[0].parameters["clip_limit"] == 3.0
        assert restored.steps[0].note == "test"

    def test_move_and_remove(self, registry) -> None:
        """Steps can be reordered and removed."""
        pipeline = Pipeline(
            steps=[PipelineStep("clahe"), PipelineStep("unsharp"), PipelineStep("nlm")]
        )
        assert pipeline.move(0, 1)
        assert pipeline.steps[0].model_name == "unsharp"
        assert not pipeline.move(0, -1)
        assert pipeline.remove(0)
        assert len(pipeline.steps) == 2


class TestAutoEngine:
    """Pipeline recommendation."""

    def test_jpeg_triggers_artifact_removal(self, registry, tmp_path: Path) -> None:
        """A heavily compressed frame gets an artefact-removal step first."""
        from scripts.make_sample import build_scene

        scene = build_scene(480, 336, seed=5)
        path = tmp_path / "low.jpg"
        cv2.imwrite(str(path), scene[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 15])
        image = ImageData(pixels=cv2.imread(str(path))[..., ::-1], source_path=path)

        report = analyze_image(image, source_path=path)
        recommendation = AutoRestorationEngine().recommend(report)
        tasks = [step.task for step in recommendation.pipeline.enabled_steps]
        assert TaskType.JPEG_ARTIFACT.value in tasks
        assert tasks[0] == TaskType.JPEG_ARTIFACT.value

    def test_low_resolution_triggers_super_resolution_last(
        self, registry, tmp_path: Path
    ) -> None:
        """Super-resolution is proposed and placed at the end."""
        from scripts.make_sample import build_scene

        small = cv2.resize(
            build_scene(480, 336, seed=5), (150, 105), interpolation=cv2.INTER_AREA
        )
        report = analyze_image(ImageData(pixels=small))
        recommendation = AutoRestorationEngine().recommend(report)
        tasks = [step.task for step in recommendation.pipeline.enabled_steps]
        assert TaskType.SUPER_RESOLUTION.value in tasks
        assert tasks[-1] == TaskType.SUPER_RESOLUTION.value

    def test_upscaled_source_gets_no_super_resolution(
        self, registry
    ) -> None:
        """An already-interpolated frame is not proposed for enlargement."""
        from scripts.make_sample import build_scene

        # A 2x enlargement sits comfortably inside the oversampling
        # estimator's reliable range; see the resolution analyzer tests for
        # where that range ends.
        scene = build_scene(720, 504, seed=5)
        small = cv2.resize(scene, (360, 252), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(small, (720, 504), interpolation=cv2.INTER_CUBIC)

        report = analyze_image(ImageData(pixels=upscaled))
        recommendation = AutoRestorationEngine().recommend(report)
        tasks = [step.task for step in recommendation.pipeline.enabled_steps]
        assert TaskType.SUPER_RESOLUTION.value not in tasks
        assert any("interpolated" in w for w in recommendation.warnings)

    def test_clean_image_gets_empty_pipeline(self, registry) -> None:
        """A clean, adequately sized frame gets no operations."""
        from scripts.make_sample import build_scene

        scene = build_scene(1400, 980, seed=5)
        report = analyze_image(ImageData(pixels=scene))
        recommendation = AutoRestorationEngine().recommend(report)
        assert recommendation.is_empty, recommendation.pipeline.describe()

    def test_every_step_has_a_reason(self, registry) -> None:
        """Each recommended step carries a justification."""
        from scripts.make_sample import build_scene

        small = cv2.resize(
            build_scene(480, 336, seed=5), (150, 105), interpolation=cv2.INTER_AREA
        )
        recommendation = AutoRestorationEngine().recommend(
            analyze_image(ImageData(pixels=small))
        )
        for item in recommendation.steps:
            assert item.reason
            assert item.trigger_key
        assert recommendation.rationale_text()

    def test_prefer_classical(self, registry) -> None:
        """The classical preference avoids learned models."""
        from scripts.make_sample import build_scene

        small = cv2.resize(
            build_scene(480, 336, seed=5), (150, 105), interpolation=cv2.INTER_AREA
        )
        recommendation = AutoRestorationEngine(prefer_classical=True).recommend(
            analyze_image(ImageData(pixels=small))
        )
        for step in recommendation.pipeline.enabled_steps:
            info = step.info()
            assert info is not None
            assert info.kind == ModelKind.CLASSICAL.value, info.name
        assert recommendation.pipeline.may_synthesise is False

    def test_generative_pipeline_warns(self, registry) -> None:
        """A pipeline containing a learned model carries a warning."""
        from scripts.make_sample import build_scene

        small = cv2.resize(
            build_scene(480, 336, seed=5), (150, 105), interpolation=cv2.INTER_AREA
        )
        recommendation = AutoRestorationEngine().recommend(
            analyze_image(ImageData(pixels=small))
        )
        if recommendation.pipeline.may_synthesise:
            assert any("generative" in w for w in recommendation.warnings)


class TestNAFNet:
    """NAFNet adapter.

    Upstream publishes NAFNet weights only through Google Drive, and no
    attributable direct-download host exists, so these tests cannot verify
    compatibility with the *published* checkpoint. What they do verify is that
    the architecture's published configurations reproduce the published
    parameter counts exactly, and that the adapter's config-inference and
    load path round-trips a checkpoint of that shape.
    """

    #: Published parameter counts for the upstream configurations.
    PUBLISHED = {
        (32, 1, (1, 1, 1, 28), (1, 1, 1, 1)): 17.11,
        (64, 1, (1, 1, 1, 28), (1, 1, 1, 1)): 67.89,
    }

    @pytest.mark.parametrize("config,expected", list(PUBLISHED.items()))
    def test_parameter_counts_match_published(self, config, expected) -> None:
        """Each published configuration reproduces its published size."""
        from restoration.nafnet.arch import NAFNet

        width, middle, encoder, decoder = config
        network = NAFNet(3, width, middle, list(encoder), list(decoder))
        millions = sum(p.numel() for p in network.parameters()) / 1e6
        assert abs(millions - expected) < 0.01, (config, millions, expected)

    @pytest.mark.parametrize(
        "config",
        [
            (32, 1, [1, 1, 1, 28], [1, 1, 1, 1]),
            (64, 12, [2, 2, 4, 8], [2, 2, 2, 2]),
            (16, 2, [1, 2], [2, 1]),
        ],
    )
    def test_config_inference_round_trips(self, config) -> None:
        """The layout is recovered from a checkpoint's key structure alone.

        This is what lets the adapter load whichever NAFNet variant the
        investigator installs by hand, rather than only a hard-coded one.
        """
        from restoration.nafnet.arch import NAFNet, infer_config_from_state_dict

        width, middle, encoder, decoder = config
        network = NAFNet(3, width, middle, encoder, decoder)
        recovered = infer_config_from_state_dict(network.state_dict())

        assert recovered["width"] == width
        assert recovered["middle_blk_num"] == middle
        assert recovered["enc_blk_nums"] == encoder
        assert recovered["dec_blk_nums"] == decoder
        assert recovered["img_channel"] == 3

    def test_inferred_config_rebuilds_a_loadable_network(self, tmp_path) -> None:
        """A checkpoint written from one network loads into the inferred one."""
        import torch

        from restoration.nafnet.arch import NAFNet, infer_config_from_state_dict
        from restoration.torch_base import load_state_dict

        original = NAFNet(3, 16, 2, [1, 2], [2, 1])
        path = tmp_path / "nafnet_variant.pth"
        torch.save({"params": original.state_dict()}, path)

        state = load_state_dict(path)
        rebuilt = NAFNet(**infer_config_from_state_dict(state))
        missing, unexpected = rebuilt.load_state_dict(state, strict=False)
        assert missing == [] and unexpected == []

    def test_adapter_uses_the_installed_checkpoint_shape(self, tmp_path) -> None:
        """The adapter builds its network from the installed file, not a guess."""
        import torch

        from restoration.nafnet.arch import NAFNet
        from restoration.nafnet.model import NAFNetDeblur

        model = NAFNetDeblur(weights_dir=tmp_path)
        # An unusual layout that no fallback would produce.
        odd = NAFNet(3, 16, 3, [1, 2], [2, 1])
        torch.save(
            {"params": odd.state_dict()},
            tmp_path / model.info.weights[0].filename,
        )

        built = model.build_network()
        assert sum(p.numel() for p in built.parameters()) == sum(
            p.numel() for p in odd.parameters()
        )

    def test_rejects_a_non_nafnet_checkpoint(self) -> None:
        """Inference refuses a state dict from a different architecture."""
        from restoration.nafnet.arch import infer_config_from_state_dict

        with pytest.raises(ValueError):
            infer_config_from_state_dict({"conv_first.weight": None})

    def test_declares_manual_installation(self, registry) -> None:
        """The adapter states that weights must be installed by hand."""
        info = registry.info("nafnet_deblur")
        assert info is not None
        spec = info.weights[0]
        assert spec.url == "", "no invented mirror URL should be declared"
        assert spec.source, "the upstream location must still be recorded"
        assert "manual" in info.notes.lower()
