"""Zero-DCE and Zero-DCE++ low-light adapters.

These two are the only *neural* models in the registry declared
``may_synthesise=False``, and the reason is structural rather than a judgement
call. Both networks output tone-curve coefficients, not pixels: the output at a
pixel is a monotonically non-decreasing function of that pixel's own input
value, so no learned image prior can paint an edge, a character or a face into
the result. See :mod:`restoration.zerodce.arch` for the derivation and
``tests/test_zerodce.py`` for the assertions.

That is not the same as "safe". The curve varies spatially, so the operation
can introduce **low-frequency** shading and colour differences across a region
that was uniform in the original, and it amplifies shadow noise like any other
brightening. Both effects are stated in :attr:`ModelInfo.notes` and measured in
the test suite.
"""

from __future__ import annotations

import logging

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, ParamSpec, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["ZeroDCEModel", "ZeroDCEPlusModel", "register_zerodce"]

_AUTHORS = (
    "Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, "
    "Sam Kwong, Runmin Cong"
)
_LICENSE = (
    "CC BY-NC 4.0 - academic research use only, non-commercial "
    "(upstream code and weights); this adapter Apache-2.0"
)

_METHOD = (
    "Estimates the coefficients of a pixel-wise tone curve and applies it eight "
    "times: LE(x) = x + r*x*(x-1) with r in [-1, 1]. The derivative "
    "1 + r(2x-1) is non-negative over the whole domain, so each pixel's output "
    "is a monotonically non-decreasing function of that pixel's own input "
    "value. The network selects the curve; it does not generate pixels, and it "
    "cannot introduce an edge, a character or a facial feature that the input "
    "does not contain. Trained without any reference image, using only "
    "non-reference losses on exposure, colour constancy, spatial consistency "
    "and curve smoothness."
)

_NOTES = (
    "Two limits worth reporting alongside any result. The curve map is spatially "
    "varying, so the operation can introduce low-frequency shading or colour "
    "differences across a region that was uniform in the source. And it "
    "brightens shadows without denoising them, so sensor noise in the darkest "
    "areas is amplified along with the signal - run a denoiser first when the "
    "noise indicator is also high."
)


class ZeroDCEModel(TorchRestorationModel):
    """Zero-DCE: eight full-resolution curve maps (CVPR 2020)."""

    info = ModelInfo(
        name="zerodce",
        display_name="Zero-DCE (low light)",
        task=TaskType.EXPOSURE.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description=(
            "Zero-reference low-light enhancement by pixel-wise curve estimation."
        ),
        method=_METHOD,
        license_name=_LICENSE,
        repository="https://github.com/Li-Chongyi/Zero-DCE",
        paper=(
            "Guo et al., 'Zero-Reference Deep Curve Estimation for Low-Light "
            "Image Enhancement', CVPR 2020"
        ),
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="zerodce_epoch99.pth",
                url=(
                    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/"
                    "master/Zero-DCE_code/snapshots/Epoch99.pth"
                ),
                sha256=(
                    "a4395acb874f320375d9704997cef874"
                    "eaaaaa26a1777ceb29a92b70f74c3612"
                ),
                size_bytes=320_017,
                license_name=_LICENSE,
                source="https://github.com/Li-Chongyi/Zero-DCE (snapshots/Epoch99.pth)",
            ),
        ),
        parameters=(
            ParamSpec(
                name="strength", label="Strength", kind="float",
                default=1.0, minimum=0.0, maximum=1.0, step=0.05,
                help_text=(
                    "Blend between the input (0) and the full enhancement (1). "
                    "An adapter-side blend, not part of the published model; it "
                    "preserves the per-pixel monotonicity."
                ),
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=False,
        notes=(
            _NOTES + " Curves are estimated at full resolution, so a large frame "
            "is processed in overlapping tiles and each tile gets its own "
            "curves; on very large images prefer Zero-DCE++, which sees the "
            "whole frame in one pass."
        ),
    )

    def build_network(self):
        """Instantiate the published 32-channel configuration."""
        require_torch()
        from restoration.zerodce.arch import ZeroDCE

        return ZeroDCE(width=32)

    def run_network(self, tensor, **params):
        """Enhance ``tensor``, optionally blending back towards the input."""
        from core.exceptions import InferenceError

        if self._network is None:  # pragma: no cover - guarded by process()
            raise InferenceError("Network is not loaded")
        output = self._network(tensor)
        strength = float(params.get("strength", 1.0))
        if strength >= 1.0:
            return output
        return tensor + strength * (output - tensor)


class ZeroDCEPlusModel(TorchRestorationModel):
    """Zero-DCE++: depth-separable, one shared curve map (TPAMI 2021)."""

    info = ModelInfo(
        name="zerodce_pp",
        display_name="Zero-DCE++ (low light)",
        task=TaskType.EXPOSURE.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description=(
            "Accelerated Zero-DCE: 10.6k parameters, one shared curve map "
            "estimated at reduced resolution."
        ),
        method=(
            _METHOD
            + " This variant shares a single curve map across all eight "
            "iterations and estimates it on a downscaled copy that is "
            "bilinearly upsampled, which constrains the curve map to low "
            "spatial frequencies."
        ),
        license_name=_LICENSE,
        repository="https://github.com/Li-Chongyi/Zero-DCE_extension",
        paper=(
            "Li, Guo & Loy, 'Learning to Enhance Low-Light Image via "
            "Zero-Reference Deep Curve Estimation', IEEE TPAMI 2021"
        ),
        authors="Chongyi Li, Chunle Guo, Chen Change Loy",
        weights=(
            WeightSpec(
                filename="zerodce_pp_epoch99.pth",
                url=(
                    "https://raw.githubusercontent.com/Li-Chongyi/"
                    "Zero-DCE_extension/main/Zero-DCE%2B%2B/"
                    "snapshots_Zero_DCE%2B%2B/Epoch99.pth"
                ),
                sha256=(
                    "ca8855b90df9a80fa4195a831f33d347"
                    "6b1964f787eb70602797c773067f3b84"
                ),
                size_bytes=52_395,
                license_name=_LICENSE,
                source=(
                    "https://github.com/Li-Chongyi/Zero-DCE_extension "
                    "(Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth)"
                ),
            ),
        ),
        parameters=(
            ParamSpec(
                name="scale_factor", label="Curve resolution", kind="choice",
                default=12,
                choices=(
                    (1, "Full resolution (slowest)"),
                    (4, "1/4 resolution"),
                    (8, "1/8 resolution"),
                    (12, "1/12 resolution (published default)"),
                ),
                help_text=(
                    "Resolution at which the curve map is estimated before "
                    "bilinear upsampling. Higher factors are faster and give a "
                    "smoother, more global correction."
                ),
            ),
            ParamSpec(
                name="strength", label="Strength", kind="float",
                default=1.0, minimum=0.0, maximum=1.0, step=0.05,
                help_text=(
                    "Blend between the input (0) and the full enhancement (1). "
                    "An adapter-side blend, not part of the published model; it "
                    "preserves the per-pixel monotonicity."
                ),
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=False,
        requires_packages=("torch",),
        may_synthesise=False,
        notes=(
            _NOTES + " Tiling is disabled: at the published 1/12 curve "
            "resolution the activations are small enough to process a full "
            "frame in one pass, and doing so avoids per-tile exposure "
            "differences entirely."
        ),
    )

    def build_network(self):
        """Instantiate the published configuration.

        ``scale_factor`` is a runtime parameter rather than a build-time one, so
        the network is constructed at 1 and the attribute is set per call in
        :meth:`run_network`. It is read only inside the forward pass and holds
        no state, so this is safe.
        """
        require_torch()
        from restoration.zerodce.arch import ZeroDCEPlusPlus

        return ZeroDCEPlusPlus(width=32, scale_factor=1)

    def run_network(self, tensor, **params):
        """Enhance ``tensor`` at the requested curve resolution."""
        from core.exceptions import InferenceError

        if self._network is None:  # pragma: no cover - guarded by process()
            raise InferenceError("Network is not loaded")
        self._network.scale_factor = int(params.get("scale_factor", 12))
        output = self._network(tensor)
        strength = float(params.get("strength", 1.0))
        if strength >= 1.0:
            return output
        return tensor + strength * (output - tensor)


def register_zerodce(replace: bool = False) -> int:
    """Register the Zero-DCE variants."""
    count = 0
    for model_class in (ZeroDCEModel, ZeroDCEPlusModel):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
