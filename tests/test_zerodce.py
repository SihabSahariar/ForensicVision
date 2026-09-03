"""Tests for the Zero-DCE and Zero-DCE++ low-light adapters.

Two of these tests carry more weight than the rest.

``test_curve_is_monotonic_over_the_whole_domain`` and
``test_output_is_monotone_in_its_own_input_pixel`` are what justify declaring
these models ``may_synthesise=False`` while their kind is ``neural``. If either
ever fails, that declaration is wrong and the classification must change before
the models ship.

``test_matches_the_upstream_forward_pass`` pins this reimplementation against a
transcription of the published forward pass, so a refactor cannot quietly
change what the published weights compute.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.constants import ModelKind, TaskType

torch = pytest.importorskip("torch")

MODEL_NAMES = ("zerodce", "zerodce_pp")

#: Exactly the parameter names in the published checkpoints, so a rename here
#: would be caught before anyone tries to load real weights.
ZERODCE_KEYS = tuple(
    f"e_conv{i}.{suffix}" for i in range(1, 8) for suffix in ("weight", "bias")
)
ZERODCE_PP_KEYS = tuple(
    f"e_conv{i}.{block}.{suffix}"
    for i in range(1, 8)
    for block in ("depth_conv", "point_conv")
    for suffix in ("weight", "bias")
)


@pytest.fixture(scope="module")
def scene_float() -> np.ndarray:
    """A float RGB test scene in [0, 1]."""
    from scripts.make_sample import build_scene

    return build_scene(160, 112, seed=5).astype(np.float32) / 255.0


def _weights_ready(registry, name: str) -> bool:
    model = registry.try_get(name)
    return model is not None and model.availability().ok


class TestRegistration:
    """How the two models present themselves to the rest of the application."""

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_registered_as_neural_exposure_models(self, registry, name: str) -> None:
        """Both are exposure-task models backed by learned weights."""
        info = registry.info(name)
        assert info is not None, f"{name} is not registered"
        assert info.task == TaskType.EXPOSURE.value
        assert info.kind == ModelKind.NEURAL.value
        assert info.is_neural

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_declared_as_incapable_of_synthesis(self, registry, name: str) -> None:
        """The one neural family that cannot invent content says so.

        This is the assertion the rest of the file exists to justify.
        """
        info = registry.info(name)
        assert info.may_synthesise is False

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_weights_are_attributed_and_verifiable(self, registry, name: str) -> None:
        """Every weight file carries a URL, a digest and its licence.

        The upstream licence is non-commercial; stating it is not optional,
        because the Model Manager shows it before anyone downloads anything.
        """
        info = registry.info(name)
        assert info.weights, f"{name} declares no weight file"
        for spec in info.weights:
            assert spec.url, "no download URL"
            assert len(spec.sha256) == 64, "digest must be a full SHA-256"
            assert spec.size_bytes > 0
            # A sub-megabyte checkpoint must not render as "0 MiB", which reads
            # as "nothing to download".
            assert spec.size_human() not in {"unknown", "0 MiB", "0.0 MiB"}
            assert "NC" in spec.license_name or "non-commercial" in spec.license_name
            assert spec.source
        assert info.paper and info.repository and info.authors

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_method_text_explains_the_mechanism(self, registry, name: str) -> None:
        """The method string is printed in reports, so it must be specific."""
        info = registry.info(name)
        assert "curve" in info.method.lower()
        assert "monoton" in info.method.lower()
        # The known caveats belong with the model, not in a commit message.
        assert "noise" in info.notes.lower()


class TestCurveProperties:
    """The mathematics that makes these models non-synthesising."""

    def test_curve_is_monotonic_over_the_whole_domain(self) -> None:
        """LE(x; r) never decreases, for any r the network can emit.

        ``r`` is a tanh output, so it is bounded to [-1, 1]. The derivative
        1 + r(2x - 1) has minimum 0 over that box, so the curve - and any
        composition of it - is monotonically non-decreasing. A monotone map
        creates no new local extrema, which is why the network cannot introduce
        an edge that the input does not contain.
        """
        from restoration.zerodce.arch import enhance

        x = torch.linspace(0.0, 1.0, 1001).view(1, 1, 1, -1)
        worst = 0.0
        for value in np.linspace(-1.0, 1.0, 101):
            curve = torch.full_like(x, float(value))
            worst = min(worst, float(torch.diff(enhance(x, curve), dim=-1).min()))
        assert worst >= -1e-7, f"curve decreased somewhere (min slope {worst})"

    def test_curve_preserves_the_unit_interval(self) -> None:
        """The eight-fold composition never leaves [0, 1]."""
        from restoration.zerodce.arch import enhance

        x = torch.linspace(0.0, 1.0, 501).view(1, 1, 1, -1)
        for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            out = enhance(x, torch.full_like(x, value))
            assert float(out.min()) >= -1e-6
            assert float(out.max()) <= 1.0 + 1e-6

    def test_enhance_accepts_one_map_or_a_sequence(self) -> None:
        """Zero-DCE++ shares one map; Zero-DCE emits eight."""
        from restoration.zerodce.arch import ITERATIONS, enhance

        x = torch.rand(1, 3, 4, 4)
        single = torch.full_like(x, 0.3)
        assert torch.allclose(enhance(x, single), enhance(x, [single] * ITERATIONS))


class TestArchitecture:
    """Structure, checked without needing the published weights."""

    def test_zerodce_parameter_names_match_the_checkpoint(self) -> None:
        """Layer naming is what makes official weights load unmodified."""
        from restoration.zerodce.arch import ZeroDCE

        assert tuple(ZeroDCE().state_dict().keys()) == ZERODCE_KEYS

    def test_zerodce_pp_parameter_names_match_the_checkpoint(self) -> None:
        """Depth-separable blocks keep upstream's inner attribute names."""
        from restoration.zerodce.arch import ZeroDCEPlusPlus

        assert tuple(ZeroDCEPlusPlus().state_dict().keys()) == ZERODCE_PP_KEYS

    def test_published_parameter_counts(self) -> None:
        """79.4k and 10.6k parameters, as published."""
        from restoration.zerodce.arch import ZeroDCE, ZeroDCEPlusPlus

        assert sum(p.numel() for p in ZeroDCE().parameters()) == 79_416
        assert sum(p.numel() for p in ZeroDCEPlusPlus().parameters()) == 10_561

    @pytest.mark.parametrize("scale", [1, 4, 8, 12])
    def test_matches_the_upstream_forward_pass(self, scale: int) -> None:
        """Our size-based resizing equals upstream's scale-factor version.

        Upstream crops the input to a multiple of ``scale_factor``; cropping
        evidence is not acceptable, so this implementation resizes by explicit
        target size instead. On an input that *is* a multiple the two must agree
        exactly - otherwise the published weights are computing something else.
        """
        import torch.nn as nn
        import torch.nn.functional as F

        from restoration.zerodce.arch import ZeroDCEPlusPlus

        torch.manual_seed(7)
        net = ZeroDCEPlusPlus(scale_factor=scale).eval()

        def upstream(module, x):
            relu = module.relu
            if scale == 1:
                source = x
            else:
                source = F.interpolate(
                    x, scale_factor=1 / scale, mode="bilinear",
                    recompute_scale_factor=True,
                )
            x1 = relu(module.e_conv1(source))
            x2 = relu(module.e_conv2(x1))
            x3 = relu(module.e_conv3(x2))
            x4 = relu(module.e_conv4(x3))
            x5 = relu(module.e_conv5(torch.cat([x3, x4], 1)))
            x6 = relu(module.e_conv6(torch.cat([x2, x5], 1)))
            curve = torch.tanh(module.e_conv7(torch.cat([x1, x6], 1)))
            if scale != 1:
                curve = nn.UpsamplingBilinear2d(scale_factor=scale)(curve)
            for _ in range(8):
                x = x + curve * (torch.pow(x, 2) - x)
            return x

        # 96 x 120 is divisible by every scale factor under test.
        sample = torch.rand(1, 3, 96, 120)
        with torch.inference_mode():
            assert torch.allclose(net(sample), upstream(net, sample), atol=1e-6)

    def test_odd_sizes_are_preserved_not_cropped(self) -> None:
        """Evidence keeps its dimensions whatever the curve resolution."""
        from restoration.zerodce.arch import ZeroDCEPlusPlus

        net = ZeroDCEPlusPlus(scale_factor=12).eval()
        sample = torch.rand(1, 3, 101, 137)
        with torch.inference_mode():
            assert net(sample).shape == sample.shape

    def test_tiny_inputs_do_not_collapse_the_curve_map(self) -> None:
        """An image smaller than the scale factor still produces a curve map."""
        from restoration.zerodce.arch import ZeroDCEPlusPlus

        net = ZeroDCEPlusPlus(scale_factor=12).eval()
        sample = torch.rand(1, 3, 5, 7)
        with torch.inference_mode():
            assert net(sample).shape == sample.shape


class TestWithPublishedWeights:
    """Behaviour with the real checkpoints; skipped when they are absent."""

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_checkpoint_loads_with_no_key_mismatch(self, registry, name: str) -> None:
        """Zero missing, zero unexpected - the project's loading standard."""
        if not _weights_ready(registry, name):
            pytest.skip(f"{name} weights are not installed")

        from restoration.torch_base import load_state_dict

        model = registry.get(name)
        network = model.build_network()
        state = load_state_dict(model.primary_weight_path())
        missing, unexpected = network.load_state_dict(state, strict=False)
        assert not missing, f"missing keys: {missing[:5]}"
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_brightens_an_underexposed_frame(self, registry, name: str) -> None:
        """The whole point: a dark frame comes out lighter, and in range."""
        if not _weights_ready(registry, name):
            pytest.skip(f"{name} weights are not installed")

        model = registry.get(name)
        rng = np.random.default_rng(0)
        dark = (rng.random((64, 64, 3)).astype(np.float32) * 0.18)
        model.load("cpu")
        try:
            out = model.process(dark)
        finally:
            model.unload()

        assert out.shape == dark.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.mean() > dark.mean() * 1.5

    @pytest.mark.parametrize("name", MODEL_NAMES)
    def test_strength_zero_is_a_no_op(self, registry, name: str) -> None:
        """The adapter-side blend reaches the identity at 0."""
        if not _weights_ready(registry, name):
            pytest.skip(f"{name} weights are not installed")

        model = registry.get(name)
        rng = np.random.default_rng(1)
        source = (rng.random((32, 32, 3)).astype(np.float32) * 0.25)
        model.load("cpu")
        try:
            out = model.process(source, strength=0.0)
        finally:
            model.unload()
        assert np.allclose(out, source, atol=1e-5)

    def test_output_is_monotone_in_its_own_input_pixel(self, registry) -> None:
        """With the published weights, each pixel's response never decreases.

        The curve map is fixed by the input, then the response is swept across
        the whole intensity range. A non-decreasing response is what rules out
        the network painting structure into a pixel: whatever it does, it does
        by choosing a monotone curve for that pixel.
        """
        if not _weights_ready(registry, "zerodce_pp"):
            pytest.skip("zerodce_pp weights are not installed")

        from restoration.zerodce.arch import enhance

        model = registry.get("zerodce_pp")
        model.load("cpu")
        try:
            network = model.network
            network.scale_factor = 12
            # load() honours the configured device rather than the argument, so
            # follow the network rather than assuming CPU.
            device = next(network.parameters()).device
            rng = np.random.default_rng(2)
            base = (rng.random((32, 32, 3)).astype(np.float32) * 0.3)
            with torch.inference_mode():
                tensor = torch.from_numpy(base.transpose(2, 0, 1)[None]).to(device)
                curve = network.curve_map(tensor)
                responses = [
                    enhance(torch.full_like(tensor, float(level)), curve)
                    for level in torch.linspace(0.0, 1.0, 48)
                ]
            stack = torch.stack(responses)
        finally:
            model.unload()

        slope = float(torch.diff(stack, dim=0).min())
        assert slope >= -1e-6, f"response decreased somewhere (min slope {slope})"

    def test_flat_regions_gain_shading_but_not_texture(self, registry) -> None:
        """The honest caveat, pinned as a measurement.

        A spatially varying curve can introduce low-frequency shading across a
        region that was uniform. What it cannot introduce is texture: the step
        between neighbouring pixels stays far below one 8-bit level. Both halves
        matter, so both are asserted.
        """
        if not _weights_ready(registry, "zerodce_pp"):
            pytest.skip("zerodce_pp weights are not installed")

        model = registry.get("zerodce_pp")
        model.load("cpu")
        try:
            flat = np.full((128, 128, 3), 0.12, dtype=np.float32)
            out = model.process(flat, scale_factor=12)
        finally:
            model.unload()

        grey = out.mean(axis=2)
        largest_step = max(
            float(np.abs(np.diff(grey, axis=0)).max()),
            float(np.abs(np.diff(grey, axis=1)).max()),
        )
        assert largest_step * 255 < 2.0, (
            "adjacent pixels differ by more than two 8-bit levels on a flat "
            "patch, which would be texture rather than shading"
        )
        assert float(out.max() - out.min()) > 0.0, (
            "a perfectly uniform output would mean the curve map is constant, "
            "and the shading caveat in the documentation would be overstated"
        )


class TestAutoEngineIntegration:
    """The engine must only reach for a low-light model when it is dark."""

    @staticmethod
    def _report(pixels: np.ndarray):
        from analysis import analyze_image
        from core.image_io import ImageData

        return analyze_image(ImageData(pixels=pixels))

    def test_underexposure_prefers_the_low_light_models(self, registry) -> None:
        """Zero-DCE outranks the fixed gamma curve on a dark frame."""
        from restoration.auto_engine import AutoRestorationEngine

        dark = np.full((96, 128, 3), 14, dtype=np.uint8)
        engine = AutoRestorationEngine()
        names = [
            info.name
            for info in engine._candidates(
                TaskType.EXPOSURE.value, self._report(dark)
            )
        ]
        assert names[:2] == ["zerodce_pp", "zerodce"], names
        assert "exposure" in names, "the classical fallback must stay available"

    def test_overexposure_does_not_offer_a_low_light_model(self, registry) -> None:
        """A blown-out frame gets the classical tone mapping and nothing else.

        Zero-DCE is trained exclusively on low-light photography; proposing it
        for an overexposed frame would be a learned model applied outside the
        distribution it was fitted to.
        """
        from restoration.auto_engine import AutoRestorationEngine

        bright = np.full((96, 128, 3), 250, dtype=np.uint8)
        engine = AutoRestorationEngine()
        names = [
            info.name
            for info in engine._candidates(
                TaskType.EXPOSURE.value, self._report(bright)
            )
        ]
        assert names[0] == "exposure", names

    def test_no_report_falls_back_to_the_classical_ordering(self, registry) -> None:
        """Without an analysis there is no direction to reason about."""
        from restoration.auto_engine import AutoRestorationEngine

        engine = AutoRestorationEngine()
        names = [info.name for info in engine._candidates(TaskType.EXPOSURE.value)]
        assert names[0] == "exposure", names

    def test_gamma_is_not_written_onto_a_model_that_has_no_gamma(
        self, registry
    ) -> None:
        """A parameter with no effect must not enter the provenance record."""
        from restoration.auto_engine import AutoRestorationEngine

        dark = np.full((96, 128, 3), 14, dtype=np.uint8)
        engine = AutoRestorationEngine()
        params = engine._parameters_for(
            registry.info("zerodce_pp"),
            self._report(dark),
            TaskType.EXPOSURE.value,
            4,
        )
        assert "gamma" not in params
        assert set(params) == {"scale_factor", "strength"}

    def test_a_noisy_dark_frame_warns_about_amplification(self, registry) -> None:
        """Brightening shadows brightens their noise; say so up front."""
        if not _weights_ready(registry, "zerodce_pp"):
            pytest.skip("zerodce_pp weights are not installed")

        from restoration.auto_engine import AutoRestorationEngine

        rng = np.random.default_rng(4)
        noisy_dark = np.clip(
            18 + rng.normal(0, 22, (128, 160, 3)), 0, 255
        ).astype(np.uint8)
        recommendation = AutoRestorationEngine().recommend(self._report(noisy_dark))

        assert any(
            step.model_name in {"zerodce", "zerodce_pp"}
            for step in recommendation.pipeline.enabled_steps
        ), "expected a low-light model on a dark frame"
        assert any(
            "noise" in warning.lower() and "shadow" in warning.lower()
            for warning in recommendation.warnings
        ), recommendation.warnings


class TestKindIsNotSynthesis:
    """The two axes are independent, and the record must keep them apart."""

    def test_pipeline_reports_kind_and_synthesis_separately(
        self, registry, scene_float
    ) -> None:
        """A neural, non-synthesising pipeline records exactly that.

        Before Zero-DCE existed, ``model_kind`` was written as a proxy for
        ``may_synthesise``. That would now record a run of a trained network as
        "classical", which is a false statement in a derivative's own record.
        """
        if not _weights_ready(registry, "zerodce_pp"):
            pytest.skip("zerodce_pp weights are not installed")

        from core.image_io import ImageData
        from restoration.pipeline import Pipeline, PipelineRunner, PipelineStep

        pipeline = Pipeline(name="Low light")
        pipeline.add(PipelineStep("zerodce_pp", {"scale_factor": 12}))
        dark = ImageData(pixels=(scene_float * 0.2 * 255).astype(np.uint8))
        result = PipelineRunner(device="cpu").run(dark, pipeline)

        assert result.model_kind == ModelKind.NEURAL.value
        assert result.may_synthesise is False

    def test_a_classical_pipeline_is_still_classical(
        self, registry, scene_float
    ) -> None:
        """The change must not relabel deterministic operators."""
        from core.image_io import ImageData
        from restoration.pipeline import Pipeline, PipelineRunner, PipelineStep

        pipeline = Pipeline(name="Classical")
        pipeline.add(PipelineStep("clahe", {}))
        source = ImageData(pixels=(scene_float * 255).astype(np.uint8))
        result = PipelineRunner(device="cpu").run(source, pipeline)

        assert result.model_kind == ModelKind.CLASSICAL.value
        assert result.may_synthesise is False
