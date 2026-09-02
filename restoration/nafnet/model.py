"""NAFNet adapters.

Upstream distributes NAFNet checkpoints through Google Drive rather than as
direct release assets, so these adapters declare no download URL. The Model
Manager therefore shows them as *Weights missing* with the upstream location,
and the investigator installs the file with **Install from file...**.

This is deliberate: ForensicVision will not invent a mirror URL, because a
weight file whose provenance cannot be stated is not admissible tooling.
"""

from __future__ import annotations

import logging
from typing import Sequence

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["register_nafnet"]

_REPO = "https://github.com/megvii-research/NAFNet"
_PAPER = "Chen et al., 'Simple Baselines for Image Restoration', ECCV 2022"
_AUTHORS = "Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, Jian Sun"
_LICENSE = (
    "MIT (upstream code). Released weights are provided for non-commercial "
    "research; review the upstream terms before operational deployment."
)
_SOURCE = f"{_REPO}#results-and-pre-trained-models"

_METHOD = (
    "Activation-free UNet trained end to end on a specific degradation family. "
    "Its output is a learned estimate of the clean image; detail it introduces "
    "comes from the training distribution rather than from measurements of "
    "this frame."
)


class _NAFNetBase(TorchRestorationModel):
    """Shared NAFNet plumbing.

    Upstream ships several block layouts (GoPro uses ``[1, 1, 1, 28]`` with one
    middle block, SIDD uses ``[2, 2, 4, 8]`` with twelve) and distributes the
    ``.pth`` files without their YAML configs. Because the investigator installs
    these files by hand, the adapter reads the layout back out of the checkpoint
    rather than assuming one - so any published NAFNet variant loads correctly.

    :attr:`fallback_width` and friends are used only when no checkpoint is
    present, e.g. when the Model Manager reports on an uninstalled model.
    """

    fallback_width: int = 32
    fallback_middle: int = 1
    fallback_enc: Sequence[int] = (1, 1, 1, 28)
    fallback_dec: Sequence[int] = (1, 1, 1, 1)

    def build_network(self):
        """Instantiate NAFNet, shaping it from the installed checkpoint."""
        require_torch()
        from restoration.nafnet.arch import NAFNet, infer_config_from_state_dict
        from restoration.torch_base import load_state_dict

        path = self.primary_weight_path()
        if path.is_file():
            try:
                config = infer_config_from_state_dict(load_state_dict(path))
                logger.info(
                    "NAFNet configuration read from %s: width=%d enc=%s middle=%d dec=%s",
                    path.name,
                    config["width"],
                    config["enc_blk_nums"],
                    config["middle_blk_num"],
                    config["dec_blk_nums"],
                )
                return NAFNet(**config)
            except Exception:
                logger.warning(
                    "Could not infer the NAFNet layout from %s; using the "
                    "documented default for this variant",
                    path.name,
                    exc_info=True,
                )

        return NAFNet(
            img_channel=3,
            width=self.fallback_width,
            middle_blk_num=self.fallback_middle,
            enc_blk_nums=list(self.fallback_enc),
            dec_blk_nums=list(self.fallback_dec),
        )


class NAFNetDeblur(_NAFNetBase):
    """GoPro-trained motion deblurring (width 32)."""

    fallback_width = 32
    fallback_middle = 1
    fallback_enc = (1, 1, 1, 28)
    fallback_dec = (1, 1, 1, 1)

    info = ModelInfo(
        name="nafnet_deblur",
        display_name="NAFNet Deblur (GoPro w32)",
        task=TaskType.DEBLUR.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Motion deblurring trained on the GoPro dataset.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="NAFNet-GoPro-width32.pth",
                url="",  # upstream publishes Google Drive links only
                size_bytes=68_000_000,
                license_name=_LICENSE,
                source=_SOURCE,
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=16,
        notes=(
            "Weights must be downloaded manually from the upstream repository "
            "and installed with 'Install from file...' in the Model Manager."
        ),
    )


class NAFNetDeblurLarge(_NAFNetBase):
    """GoPro-trained motion deblurring (width 64) - stronger and slower."""

    fallback_width = 64
    fallback_middle = 1
    fallback_enc = (1, 1, 1, 28)
    fallback_dec = (1, 1, 1, 1)

    info = ModelInfo(
        name="nafnet_deblur_large",
        display_name="NAFNet Deblur (GoPro w64)",
        task=TaskType.DEBLUR.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Wider NAFNet motion-deblurring variant.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="NAFNet-GoPro-width64.pth",
                url="",
                size_bytes=270_000_000,
                license_name=_LICENSE,
                source=_SOURCE,
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=16,
        notes="Weights must be installed manually; see the upstream repository.",
    )


class NAFNetDenoise(_NAFNetBase):
    """SIDD-trained real-image denoising (width 32)."""

    fallback_width = 32
    fallback_middle = 12
    fallback_enc = (2, 2, 4, 8)
    fallback_dec = (2, 2, 2, 2)

    info = ModelInfo(
        name="nafnet_denoise",
        display_name="NAFNet Denoise (SIDD w32)",
        task=TaskType.DENOISE.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Real sensor-noise removal trained on the SIDD dataset.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="NAFNet-SIDD-width32.pth",
                url="",
                size_bytes=68_000_000,
                license_name=_LICENSE,
                source=_SOURCE,
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=16,
        notes="Weights must be installed manually; see the upstream repository.",
    )


def register_nafnet(replace: bool = False) -> int:
    """Register the NAFNet variants."""
    count = 0
    for model_class in (NAFNetDeblur, NAFNetDeblurLarge, NAFNetDenoise):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
