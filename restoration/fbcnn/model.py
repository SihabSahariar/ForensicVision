"""FBCNN JPEG-artefact removal adapters."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, ParamSpec, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch

logger = logging.getLogger(__name__)

__all__ = ["FBCNNColor", "FBCNNGray", "register_fbcnn"]

_REPO = "https://github.com/jiaxi-jiang/FBCNN"
_PAPER = "Jiang, Zhang & Timofte, 'Towards Flexible Blind JPEG Artifacts Removal', ICCV 2021"
_AUTHORS = "Jiaxi Jiang, Kai Zhang, Radu Timofte"
_LICENSE = "Apache-2.0 (upstream code and weights)"

_METHOD = (
    "Predicts the compressed image's quality factor and conditions the decoder "
    "on it. Removes blocking and ringing and reconstructs plausible detail in "
    "the frequency bands quantisation discarded - that reconstruction is "
    "inferred from training data, not recovered from the file."
)


class _FBCNNBase(TorchRestorationModel):
    """Shared FBCNN plumbing including the quality-factor override."""

    in_channels: int = 3

    def build_network(self):
        """Instantiate the FBCNN network."""
        require_torch()
        from restoration.fbcnn.arch import FBCNN

        return FBCNN(
            in_nc=self.in_channels,
            out_nc=self.in_channels,
            nc=(64, 128, 256, 512),
            nb=4,
        )

    def run_network(self, tensor, **params: Any):
        """Run FBCNN, optionally overriding the predicted quality factor."""
        import torch  # noqa: PLC0415

        if self._network is None:  # pragma: no cover - guarded by process()
            raise RuntimeError("Network is not loaded")

        if self.in_channels == 1:
            # The grayscale checkpoint is applied per colour channel.
            outputs = []
            for index in range(tensor.shape[1]):
                plane = tensor[:, index : index + 1]
                out, _ = self._network(plane, self._qf_tensor(plane, params))
                outputs.append(out)
            return torch.cat(outputs, dim=1)

        output, predicted = self._network(tensor, self._qf_tensor(tensor, params))
        self._last_predicted_qf = float(predicted.flatten()[0].item())
        return output

    def _qf_tensor(self, reference, params: dict):
        """Build the quality-factor override tensor, or ``None`` for blind mode."""
        import torch  # noqa: PLC0415

        if not params.get("override_quality", False):
            return None
        quality = float(params.get("quality_factor", 50.0)) / 100.0
        return torch.tensor(
            [[quality]], dtype=reference.dtype, device=reference.device
        )

    def _process(
        self, image: np.ndarray, progress=None, **params: Any
    ) -> np.ndarray:
        self._last_predicted_qf = None
        result = super()._process(image, progress=progress, **params)
        if self._last_predicted_qf is not None:
            logger.info(
                "%s predicted a source quality factor of %.0f/100",
                self.info.display_name,
                self._last_predicted_qf * 100.0,
            )
        return result

    @property
    def last_predicted_quality(self) -> Optional[float]:
        """Quality factor (0-100) the network predicted on the last run."""
        value = getattr(self, "_last_predicted_qf", None)
        return None if value is None else value * 100.0


_QF_PARAMS = (
    ParamSpec(
        name="override_quality",
        label="Override predicted quality",
        kind="bool",
        default=False,
        help_text=(
            "By default the network estimates the source quality factor. "
            "Enable this to state it explicitly and see how sensitive the "
            "result is to that assumption."
        ),
    ),
    ParamSpec(
        name="quality_factor",
        label="Assumed quality factor",
        kind="int",
        default=50,
        minimum=1,
        maximum=100,
        step=1,
        help_text="Lower values apply stronger artefact removal. Used only when the override is enabled.",
    ),
)


class FBCNNColor(_FBCNNBase):
    """Colour JPEG artefact removal."""

    in_channels = 3

    info = ModelInfo(
        name="fbcnn_color",
        display_name="FBCNN (colour)",
        task=TaskType.JPEG_ARTIFACT.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Blind JPEG artefact removal with a predicted quality factor.",
        method=_METHOD,
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="fbcnn_color.pth",
                url=f"{_REPO}/releases/download/v1.0/fbcnn_color.pth",
                size_bytes=287_780_000,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v1.0",
            ),
        ),
        parameters=_QF_PARAMS,
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
    )


class FBCNNGray(_FBCNNBase):
    """Grayscale JPEG artefact removal, applied per colour channel."""

    in_channels = 1

    info = ModelInfo(
        name="fbcnn_gray",
        display_name="FBCNN (grayscale)",
        task=TaskType.JPEG_ARTIFACT.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Grayscale JPEG artefact removal; applied to each colour channel.",
        method=_METHOD
        + " Applying the grayscale model per channel ignores chroma "
        "subsampling, so the colour model is preferred for colour evidence.",
        license_name=_LICENSE,
        repository=_REPO,
        paper=_PAPER,
        authors=_AUTHORS,
        weights=(
            WeightSpec(
                filename="fbcnn_gray.pth",
                url=f"{_REPO}/releases/download/v1.0/fbcnn_gray.pth",
                size_bytes=287_700_000,
                license_name=_LICENSE,
                source=f"{_REPO}/releases/tag/v1.0",
            ),
        ),
        parameters=_QF_PARAMS,
        scale=1,
        supports_fp16=True,
        supports_tiling=True,
        requires_packages=("torch",),
        may_synthesise=True,
        size_multiple=8,
        notes="Prefer the colour model unless the evidence is genuinely monochrome.",
    )


def register_fbcnn(replace: bool = False) -> int:
    """Register the FBCNN variants."""
    count = 0
    for model_class in (FBCNNColor, FBCNNGray):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
