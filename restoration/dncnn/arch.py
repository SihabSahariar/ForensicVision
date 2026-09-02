"""DnCNN: residual learning of deep CNN for image denoising.

Implemented to match the layer naming of the KAIR reference implementation
(``model.N.weight``), so the published ``dncnn_*.pth`` checkpoints load
unchanged.

The network predicts the *noise* and subtracts it, which is why the final
activation is absent and the forward pass ends with ``x - n``.

Reference:
    Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
    for Image Denoising", IEEE TIP 2017. Upstream code (cszn/KAIR) is MIT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DnCNN"]


class DnCNN(nn.Module):
    """Plain residual denoising CNN.

    Args:
        in_nc: Input channels (1 for grayscale, 3 for colour).
        out_nc: Output channels.
        nc: Feature width.
        nb: Total convolution count. Blind models use 20 without batch norm;
            noise-level-specific models use 17 with batch norm.
        act_mode: ``"R"`` for Conv+ReLU (no batch norm) or ``"BR"`` for
            Conv+BatchNorm+ReLU, matching the KAIR mode strings.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: int = 64,
        nb: int = 20,
        act_mode: str = "R",
    ) -> None:
        super().__init__()
        use_batch_norm = "B" in act_mode.upper()

        layers: list = [
            nn.Conv2d(in_nc, nc, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(nb - 2):
            layers.append(nn.Conv2d(nc, nc, kernel_size=3, padding=1, bias=True))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(nc, momentum=0.9, eps=1e-4, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(nc, out_nc, kernel_size=3, padding=1, bias=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` minus the predicted noise residual."""
        return x - self.model(x)
