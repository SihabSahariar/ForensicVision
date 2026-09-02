"""Real-ESRGAN adapters.

Three variants are exposed. All are GAN-trained generators: they hallucinate
plausible high-frequency detail, which is precisely why the adapters set
``may_synthesise=True`` and the UI warns before running them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, ParamSpec, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["RealESRGANx4", "RealESRGANx2", "RealESRGANAnime", "register_realesrgan"]

_REPO = "https://github.com/xinntao/Real-ESRGAN"
_PAPER = "Wang et al., 'Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data', ICCVW 2021"
_AUTHORS = "Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan"
_CODE_LICENSE = "BSD-3-Clause (upstream code); this adapter Apache-2.0"
_WEIGHT_LICENSE = "BSD-3-Clause - see the Real-ESRGAN repository"

_SYNTHESIS_NOTE = (
    "GAN-trained generator. It produces detail that is statistically plausible "
    "for the training distribution rather than measured from this frame. "
    "Characters, facial features and textures in the output may be invented. "
    "Compare against the Lanczos Upscale baseline before relying on any "
    "recovered detail."
)


class _RealESRGANBase(TorchRestorationModel):
    """Shared implementation for the RRDBNet-based variants."""

    #: Set by subclasses.
    num_block: int = 23
    net_scale: int = 4

    def build_network(self):
        """Instantiate :class:`~restoration.realesrgan.arch.RRDBNet`."""
        require_torch()
        from restoration.realesrgan.arch import RRDBNet

        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=self.net_scale,
            num_feat=64,
            num_block=self.num_block,
            num_grow_ch=32,
        )

    def _process(
        self,
        image: np.ndarray,
        progress=None,
        **params: Any,
    ) -> np.ndarray:
        """Run the network, then resample to the requested output scale.

        The checkpoint fixes the network's native scale. When the examiner asks
        for a different factor the network runs at its native scale and the
        result is resampled with Lanczos, which is stated in the provenance
        record rather than being silently absorbed.
        """
        requested = int(params.pop("scale", self.net_scale))
        result = super()._process(image, progress=progress, **params)

        if requested != self.net_scale:
            from restoration.classical.enhance import lanczos_resize

            factor = requested / float(self.net_scale)
            logger.info(
                "Resampling %s output by %.3f to reach the requested x%d",
                self.info.display_name, factor, requested,
            )
            result = lanczos_resize(result, factor)
        return result


class RealESRGANx4(_RealESRGANBase):
    """The general-purpose x4 model."""

    num_block = 23
    net_scale = 4

    info = ModelInfo(
        name="realesrgan_x4plus",
        display_name="Real-ESRGAN x4plus",
        task=TaskType.SUPER_RESOLUTION.value,
        kind=ModelKind.NEURAL.value,
        version="0.1.0",
        description="General-purpose 4x blind super-resolution for real-world degradations.",
        method=_SYNTHESIS_NOTE,
        license_name=_CODE_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="RealESRGAN_x4plus.pth",
                url=f"{_REPO}/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                size_bytes=67_040_989,
                license_name=_WEIGHT_LICENSE,
                source=f"{_REPO}/releases/tag/v0.1.0",
            ),
        ),
        parameters=(
            ParamSpec(
                name="scale", label="Output scale", kind="choice", default=4,
                choices=((2, "2x (resampled from 4x)"), (3, "3x (resampled from 4x)"),
                         (4, "4x (native)")),
                help_text="Non-native scales run the network at 4x then Lanczos-resample.",
            ),
        ),
        scale=4,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=1,
        notes="Native scale is 4x; other factors are reached by resampling.",
    )


class RealESRGANx2(_RealESRGANBase):
    """The x2 model - preferred when only a modest enlargement is needed."""

    num_block = 23
    net_scale = 2

    info = ModelInfo(
        name="realesrgan_x2plus",
        display_name="Real-ESRGAN x2plus",
        task=TaskType.SUPER_RESOLUTION.value,
        kind=ModelKind.NEURAL.value,
        version="0.2.1",
        description="General-purpose 2x blind super-resolution.",
        method=_SYNTHESIS_NOTE,
        license_name=_CODE_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="RealESRGAN_x2plus.pth",
                url=f"{_REPO}/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                size_bytes=67_061_725,
                license_name=_WEIGHT_LICENSE,
                source=f"{_REPO}/releases/tag/v0.2.1",
            ),
        ),
        parameters=(
            ParamSpec(
                name="scale", label="Output scale", kind="choice", default=2,
                choices=((2, "2x (native)"), (4, "4x (resampled from 2x)")),
            ),
        ),
        scale=2,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        #: The x2 variant pixel-unshuffles by 2, so input dimensions must be even.
        size_multiple=2,
    )


class RealESRGANAnime(_RealESRGANBase):
    """The 6-block variant: much faster, tuned for flat illustration content."""

    num_block = 6
    net_scale = 4

    info = ModelInfo(
        name="realesrgan_anime6b",
        display_name="Real-ESRGAN x4 (6-block)",
        task=TaskType.SUPER_RESOLUTION.value,
        kind=ModelKind.NEURAL.value,
        version="0.2.2.4",
        description=(
            "Compact 6-block variant. Roughly four times faster than x4plus and "
            "trained for illustration content; on photographic evidence it "
            "tends to flatten texture."
        ),
        method=_SYNTHESIS_NOTE,
        license_name=_CODE_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="RealESRGAN_x4plus_anime_6B.pth",
                url=(
                    f"{_REPO}/releases/download/v0.2.2.4/"
                    "RealESRGAN_x4plus_anime_6B.pth"
                ),
                size_bytes=17_938_799,
                license_name=_WEIGHT_LICENSE,
                source=f"{_REPO}/releases/tag/v0.2.2.4",
            ),
        ),
        parameters=(
            ParamSpec(
                name="scale", label="Output scale", kind="choice", default=4,
                choices=((2, "2x (resampled)"), (4, "4x (native)")),
            ),
        ),
        scale=4,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        notes="Trained on illustration data; verify carefully on photographic evidence.",
    )


def register_realesrgan(replace: bool = False) -> int:
    """Register every Real-ESRGAN variant."""
    count = 0
    for model_class in (RealESRGANx4, RealESRGANx2, RealESRGANAnime):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
