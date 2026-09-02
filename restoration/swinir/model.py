"""SwinIR adapters for super-resolution, denoising and JPEG artefact removal."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, ParamSpec, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["register_swinir"]

_REPO = "https://github.com/JingyunLiang/SwinIR"
_RELEASE = f"{_REPO}/releases/download/v0.0"
_PAPER = "Liang et al., 'SwinIR: Image Restoration Using Swin Transformer', ICCVW 2021"
_AUTHORS = "Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte"
_LICENSE = "Apache-2.0 (upstream code and weights)"

_METHOD = (
    "Shifted-window transformer trained for one restoration task. Its output "
    "is a learned reconstruction: detail it adds reflects the training "
    "distribution rather than measurements of this frame."
)


class _SwinIRBase(TorchRestorationModel):
    """Shared SwinIR plumbing; subclasses supply the network configuration."""

    #: Keyword arguments forwarded to :class:`~restoration.swinir.arch.SwinIR`.
    network_config: Dict[str, Any] = {}

    def build_network(self):
        """Instantiate SwinIR with this variant's published configuration."""
        require_torch()
        from restoration.swinir.arch import SwinIR

        return SwinIR(**self.network_config)


class SwinIRClassicalSR(_SwinIRBase):
    """Classical (bicubic-degradation) x4 super-resolution."""

    network_config = {
        "upscale": 4, "in_chans": 3, "img_size": 64, "window_size": 8,
        "img_range": 1.0, "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": "pixelshuffle", "resi_connection": "1conv",
    }

    info = ModelInfo(
        name="swinir_classical_sr",
        display_name="SwinIR Classical SR x4",
        task=TaskType.SUPER_RESOLUTION.value,
        kind=ModelKind.NEURAL.value,
        version="0.0",
        description=(
            "PSNR-oriented 4x super-resolution trained on bicubic downsampling."
        ),
        method=(
            _METHOD
            + " Trained on clean bicubic downsampling, so it is the more "
            "conservative choice on lightly degraded sources; it performs "
            "poorly on compressed or noisy CCTV frames."
        ),
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
                url=f"{_RELEASE}/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
                size_bytes=67_903_000,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v0.0",
            ),
        ),
        scale=4,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
    )


class SwinIRRealSR(_SwinIRBase):
    """Real-world (blind) x4 super-resolution, GAN-trained large model."""

    network_config = {
        "upscale": 4, "in_chans": 3, "img_size": 64, "window_size": 8,
        "img_range": 1.0, "depths": [6] * 9, "embed_dim": 240,
        "num_heads": [8] * 9, "mlp_ratio": 2,
        "upsampler": "nearest+conv", "resi_connection": "3conv",
    }

    info = ModelInfo(
        name="swinir_real_sr",
        display_name="SwinIR Real-World SR x4 (GAN)",
        task=TaskType.SUPER_RESOLUTION.value,
        kind=ModelKind.NEURAL.value,
        version="0.0",
        description="Blind 4x super-resolution for real-world degradations.",
        method=(
            _METHOD
            + " This is the GAN-trained large variant: it produces the most "
            "convincing texture and is correspondingly the most likely to "
            "invent detail. Compare against Lanczos before relying on it."
        ),
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                url=f"{_RELEASE}/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
                size_bytes=142_500_000,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v0.0",
            ),
        ),
        scale=4,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
    )


class SwinIRColorDenoise(_SwinIRBase):
    """Colour denoising at a nominal sigma of 25/255."""

    network_config = {
        "upscale": 1, "in_chans": 3, "img_size": 128, "window_size": 8,
        "img_range": 1.0, "depths": [6] * 6, "embed_dim": 180,
        "num_heads": [6] * 6, "mlp_ratio": 2,
        "upsampler": "", "resi_connection": "1conv",
    }

    info = ModelInfo(
        name="swinir_denoise",
        display_name="SwinIR Colour Denoise (sigma 25)",
        task=TaskType.DENOISE.value,
        kind=ModelKind.NEURAL.value,
        version="0.0",
        description="Colour denoising trained for additive Gaussian noise at sigma 25.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
                url=f"{_RELEASE}/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
                size_bytes=122_900_000,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v0.0",
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
        notes="Tuned for sigma 25/255; over-smooths cleaner sources.",
    )


class SwinIRJpegCar(_SwinIRBase):
    """Compression-artefact removal at a nominal JPEG quality of 40."""

    network_config = {
        "upscale": 1, "in_chans": 1, "img_size": 126, "window_size": 7,
        "img_range": 255.0, "depths": [6] * 6, "embed_dim": 180,
        "num_heads": [6] * 6, "mlp_ratio": 2,
        "upsampler": "", "resi_connection": "1conv",
    }

    info = ModelInfo(
        name="swinir_car",
        display_name="SwinIR JPEG Artefact Removal (q40)",
        task=TaskType.JPEG_ARTIFACT.value,
        kind=ModelKind.NEURAL.value,
        version="0.0",
        description="Compression-artefact removal trained at JPEG quality 40.",
        method=(
            _METHOD
            + " The published checkpoint is single-channel, so ForensicVision "
            "applies it to the luminance channel and leaves chroma untouched - "
            "which matches how JPEG quantises the two planes differently."
        ),
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
                url=f"{_RELEASE}/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth",
                size_bytes=102_900_000,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v0.0",
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=7,
        notes="Luminance-only model; prefer FBCNN for strongly coloured artefacts.",
    )

    def run_network(self, tensor, **params: Any):
        """Apply the luminance-only network, preserving the chroma planes."""
        import torch  # noqa: PLC0415

        if self._network is None:  # pragma: no cover - guarded by process()
            raise RuntimeError("Network is not loaded")

        # BT.601 luma/chroma, matching JPEG's own colour transform.
        r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        y_restored = self._network(y)

        r_out = y_restored + 1.402 * cr
        g_out = y_restored - 0.344136 * cb - 0.714136 * cr
        b_out = y_restored + 1.772 * cb
        return torch.cat([r_out, g_out, b_out], dim=1)


_SWINIR_MODELS = (
    SwinIRClassicalSR,
    SwinIRRealSR,
    SwinIRColorDenoise,
    SwinIRJpegCar,
)


def register_swinir(replace: bool = False) -> int:
    """Register the SwinIR variants."""
    count = 0
    for model_class in _SWINIR_MODELS:
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
