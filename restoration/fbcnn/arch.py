"""FBCNN: Flexible Blind Convolutional Neural Network for JPEG restoration.

FBCNN predicts the quality factor of the compressed input and feeds that
estimate back into the decoder through per-channel gamma/beta modulation. The
prediction can also be overridden, which lets an examiner sweep the assumed
quality and observe how sensitive the result is to that assumption - useful
evidence in itself about how much of the output is inference.

Layer naming follows the reference implementation, which is built on KAIR's
``basicblock`` helpers, so the published ``fbcnn_color.pth`` and
``fbcnn_gray.pth`` checkpoints load unchanged.

Reference:
    Jiang, Zhang & Timofte, "Towards Flexible Blind JPEG Artifacts Removal",
    ICCV 2021. Upstream code is Apache-2.0.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["FBCNN", "ResBlock", "QFAttention"]


def _conv_sequence(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    mode: str = "CRC",
) -> nn.Module:
    """Build a small conv/activation stack from a KAIR-style mode string.

    ``C`` is a convolution, ``T`` a transposed convolution and ``R`` a ReLU.
    A single-element stack is returned unwrapped, exactly as KAIR's
    ``sequential`` helper does, which is what keeps checkpoint keys aligned.
    """
    layers: list = []
    for token in mode:
        if token == "C":
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            )
        elif token == "T":
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                )
            )
        elif token == "R":
            layers.append(nn.ReLU(inplace=True))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported mode token: {token}")
    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Conv-ReLU-Conv residual block."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        self.res = _conv_sequence(in_channels, out_channels, mode="CRC")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the block's residual to its input."""
        return x + self.res(x)


class QFAttention(nn.Module):
    """Residual block whose output is modulated by the quality-factor embedding."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        self.res = _conv_sequence(in_channels, out_channels, mode="CRC")

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Scale and shift the residual by the per-channel gamma/beta."""
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x + gamma * self.res(x) + beta


class FBCNN(nn.Module):
    """The FBCNN encoder-decoder with quality-factor prediction.

    Args:
        in_nc: Input channels (3 for the colour model, 1 for grayscale).
        out_nc: Output channels.
        nc: Channel widths at the four scales.
        nb: Residual blocks per stage.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: Sequence[int] = (64, 128, 256, 512),
        nb: int = 4,
    ) -> None:
        super().__init__()
        self.nc = list(nc)
        self.nb = nb

        self.m_head = _conv_sequence(in_nc, nc[0], mode="C")

        def _down(level: int) -> nn.Sequential:
            return nn.Sequential(
                *[ResBlock(nc[level], nc[level]) for _ in range(nb)],
                _conv_sequence(
                    nc[level], nc[level + 1], kernel_size=2, stride=2, padding=0, mode="C"
                ),
            )

        self.m_down1 = _down(0)
        self.m_down2 = _down(1)
        self.m_down3 = _down(2)

        self.m_body_encoder = nn.Sequential(
            *[ResBlock(nc[3], nc[3]) for _ in range(nb)]
        )
        self.m_body_decoder = nn.Sequential(
            *[ResBlock(nc[3], nc[3]) for _ in range(nb)]
        )

        def _up(from_level: int, to_level: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    _conv_sequence(
                        nc[from_level], nc[to_level], kernel_size=2, stride=2,
                        padding=0, mode="T",
                    ),
                    *[QFAttention(nc[to_level], nc[to_level]) for _ in range(nb)],
                ]
            )

        self.m_up3 = _up(3, 2)
        self.m_up2 = _up(2, 1)
        self.m_up1 = _up(1, 0)

        self.m_tail = _conv_sequence(nc[0], out_nc, mode="C")

        self.qf_pred = nn.Sequential(
            *[ResBlock(nc[3], nc[3]) for _ in range(nb)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(nc[3], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self.qf_embed = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.to_gamma_3 = nn.Sequential(nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = nn.Sequential(nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = nn.Sequential(nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = nn.Sequential(nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = nn.Sequential(nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = nn.Sequential(nn.Linear(512, nc[0]), nn.Tanh())

    def forward(
        self, x: torch.Tensor, qf_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restore ``x`` and return ``(restored, predicted_quality_factor)``.

        Args:
            x: ``Bx3xHxW`` (or ``Bx1xHxW``) input in ``[0, 1]``.
            qf_input: Optional ``Bx1`` tensor overriding the predicted quality
                factor, where 0 is the worst quality and 1 the best.
        """
        height, width = x.size()[-2:]
        pad_bottom = int(math.ceil(height / 8) * 8 - height)
        pad_right = int(math.ceil(width / 8) * 8 - width)
        if pad_bottom or pad_right:
            x = nn.ReplicationPad2d((0, pad_right, 0, pad_bottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        encoded = self.m_body_encoder(x4)
        qf = self.qf_pred(encoded)
        decoded = self.m_body_decoder(encoded)

        embedding = self.qf_embed(qf_input if qf_input is not None else qf)
        gamma_3, beta_3 = self.to_gamma_3(embedding), self.to_beta_3(embedding)
        gamma_2, beta_2 = self.to_gamma_2(embedding), self.to_beta_2(embedding)
        gamma_1, beta_1 = self.to_gamma_1(embedding), self.to_beta_1(embedding)

        out = decoded + x4
        out = self.m_up3[0](out)
        for index in range(self.nb):
            out = self.m_up3[index + 1](out, gamma_3, beta_3)

        out = out + x3
        out = self.m_up2[0](out)
        for index in range(self.nb):
            out = self.m_up2[index + 1](out, gamma_2, beta_2)

        out = out + x2
        out = self.m_up1[0](out)
        for index in range(self.nb):
            out = self.m_up1[index + 1](out, gamma_1, beta_1)

        out = out + x1
        out = self.m_tail(out)
        return out[..., :height, :width], qf
