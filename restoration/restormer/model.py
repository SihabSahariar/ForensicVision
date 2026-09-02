"""Restormer adapters for motion deblurring, defocus deblurring and denoising.

All three share one architecture and differ only in the checkpoint, so a single
base class parameterised by weight file and task covers them.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["register_restormer"]

_REPO = "https://github.com/swz30/Restormer"
_PAPER = "Zamir et al., 'Restormer: Efficient Transformer for High-Resolution Image Restoration', CVPR 2022"
_AUTHORS = "Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao"
_LICENSE = (
    "ACADEMIC / non-commercial research use only (upstream Restormer licence). "
    "Review the upstream terms before any commercial or operational deployment."
)

_METHOD = (
    "Transformer trained end to end to invert a specific degradation family. "
    "The output is the network's learned estimate of the clean image, not a "
    "measured recovery: fine structure it produces is drawn from the training "
    "distribution and may not correspond to the original scene."
)


class _RestormerBase(TorchRestorationModel):
    """Shared Restormer plumbing; subclasses only supply :attr:`info`."""

    #: Set True for the defocus checkpoint, which uses the dual-pixel skip.
    dual_pixel_task: bool = False

    def build_network(self):
        """Instantiate the Restormer network with the published configuration."""
        require_torch()
        from restoration.restormer.arch import Restormer

        return Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=(4, 6, 6, 8),
            num_refinement_blocks=4,
            heads=(1, 2, 4, 8),
            ffn_expansion_factor=2.66,
            bias=False,
            layer_norm_type="WithBias",
            dual_pixel_task=self.dual_pixel_task,
        )


def _weights(filename: str, size: int) -> tuple:
    return (
        WeightSpec(
            filename=filename,
            url=f"{_REPO}/releases/download/v1.0/{filename}",
            size_bytes=size,
            license_name=_LICENSE,
            source=f"{_REPO}/releases/tag/v1.0",
        ),
    )


class RestormerMotionDeblur(_RestormerBase):
    """Motion deblurring checkpoint (GoPro/HIDE training data)."""

    info = ModelInfo(
        name="restormer_motion_deblur",
        display_name="Restormer Motion Deblur",
        task=TaskType.DEBLUR.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Removes camera- and subject-motion blur.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=_weights("motion_deblurring.pth", 104_745_285),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
        notes="Trained on dynamic-scene motion blur; least effective on pure defocus.",
    )


class RestormerDefocusDeblur(_RestormerBase):
    """Single-image defocus deblurring checkpoint."""

    dual_pixel_task = False

    info = ModelInfo(
        name="restormer_defocus_deblur",
        display_name="Restormer Defocus Deblur",
        task=TaskType.DEBLUR.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Removes out-of-focus blur from a single image.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=_weights("single_image_defocus_deblurring.pth", 104_745_285),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
        notes="Use for defocus; prefer the motion checkpoint for directional blur.",
    )


class RestormerDenoise(_RestormerBase):
    """Real-image denoising checkpoint (SIDD training data)."""

    info = ModelInfo(
        name="restormer_denoise",
        display_name="Restormer Denoise",
        task=TaskType.DENOISE.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Removes real sensor noise, trained on the SIDD dataset.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=_weights("real_denoising.pth", 104_637_509),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
        notes=(
            "Trained on smartphone sensor noise; performance on other sensors "
            "and on codec noise is not guaranteed."
        ),
    )


def register_restormer(replace: bool = False) -> int:
    """Register every Restormer variant."""
    count = 0
    for model_class in (RestormerMotionDeblur, RestormerDefocusDeblur, RestormerDenoise):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
