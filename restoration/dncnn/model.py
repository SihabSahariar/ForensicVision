"""DnCNN denoising adapters."""

from __future__ import annotations

import logging

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["register_dncnn"]

_REPO = "https://github.com/cszn/KAIR"
_PAPER = "Zhang et al., 'Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising', IEEE TIP 2017"
_AUTHORS = "Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, Lei Zhang"
_LICENSE = "MIT (upstream KAIR); this adapter Apache-2.0"

_METHOD = (
    "Convolutional network trained to predict and subtract additive noise. "
    "It is discriminative rather than generative: it has no learned image "
    "prior to draw new structures from, which makes it one of the more "
    "conservative learned options. It can still erase genuine fine texture "
    "that resembles noise."
)


class DnCNNColorBlind(TorchRestorationModel):
    """Blind colour denoiser: 20 layers, no batch norm, no noise-level input."""

    info = ModelInfo(
        name="dncnn_color_blind",
        display_name="DnCNN (colour, blind)",
        task=TaskType.DENOISE.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Blind additive-noise removal for colour images.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="dncnn_color_blind.pth",
                url=f"{_REPO}/releases/download/v1.0/dncnn_color_blind.pth",
                size_bytes=2_681_083,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v1.0",
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=False,
        notes=(
            "Trained on additive white Gaussian noise. Sensor noise, codec "
            "noise and film grain are not AWGN, so results vary."
        ),
    )

    def build_network(self):
        """Instantiate the 20-layer blind configuration."""
        require_torch()
        from restoration.dncnn.arch import DnCNN

        return DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R")


class DnCNNColorLevel(TorchRestorationModel):
    """Noise-level-specific colour denoiser (sigma 25), 17 layers with BN."""

    info = ModelInfo(
        name="dncnn_color_25",
        display_name="DnCNN (colour, sigma 25)",
        task=TaskType.DENOISE.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Denoiser tuned for additive Gaussian noise at sigma = 25/255.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="dncnn_25.pth",
                url=f"{_REPO}/releases/download/v1.0/dncnn_25.pth",
                size_bytes=2_237_795,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v1.0",
            ),
        ),
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=False,
        notes=(
            "This checkpoint is grayscale (single channel); ForensicVision "
            "applies it per colour channel. Use the blind model when the noise "
            "level is unknown."
        ),
    )

    def build_network(self):
        """Instantiate the 17-layer configuration.

        The published ``dncnn_25`` checkpoint carries no batch-norm parameters
        (its convolutions sit at ``model.0, model.2, ... model.32``), so the
        plain Conv+ReLU variant is the matching architecture despite the
        original paper describing batch norm.
        """
        require_torch()
        from restoration.dncnn.arch import DnCNN

        return DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode="R")

    def run_network(self, tensor, **params):
        """Apply the single-channel network to each colour channel in turn."""
        import torch  # noqa: PLC0415

        if self._network is None:  # pragma: no cover - guarded by process()
            raise RuntimeError("Network is not loaded")
        channels = [
            self._network(tensor[:, index : index + 1]) for index in range(tensor.shape[1])
        ]
        return torch.cat(channels, dim=1)


def register_dncnn(replace: bool = False) -> int:
    """Register the DnCNN variants."""
    count = 0
    for model_class in (DnCNNColorBlind, DnCNNColorLevel):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
