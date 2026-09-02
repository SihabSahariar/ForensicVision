"""Real-ESRGAN generator (RRDBNet).

Implemented natively rather than depending on the ``basicsr``/``realesrgan``
PyPI packages, which are unmaintained and fail against torchvision >= 0.17.

Layer names deliberately mirror the upstream BasicSR implementation
(``conv_first``, ``body.N.rdbM.convK``, ``conv_body``, ``conv_up1/2``,
``conv_hr``, ``conv_last``) so official ``RealESRGAN_*.pth`` checkpoints load
without any key remapping.

Reference:
    Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with
    Pure Synthetic Data", ICCVW 2021. Upstream code is BSD-3-Clause.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RRDBNet", "ResidualDenseBlock", "RRDB", "SRVGGNetCompact"]


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Inverse of ``nn.PixelShuffle``: fold space into channels.

    Args:
        x: ``BxCxHxW`` tensor; ``H`` and ``W`` must be divisible by ``scale``.
        scale: Downsampling factor.
    """
    batch, channels, height, width = x.size()
    out_channels = channels * (scale ** 2)
    assert height % scale == 0 and width % scale == 0
    out_h, out_w = height // scale, width // scale
    view = x.view(batch, channels, out_h, scale, out_w, scale)
    return view.permute(0, 1, 3, 5, 2, 4).reshape(batch, out_channels, out_h, out_w)


class ResidualDenseBlock(nn.Module):
    """Five-convolution residual dense block with growth channels."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dense block with the upstream 0.2 residual scaling."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block: three stacked dense blocks."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the three dense blocks with an outer residual connection."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """The Real-ESRGAN / ESRGAN generator.

    Args:
        num_in_ch: Input channel count (3 for RGB).
        num_out_ch: Output channel count.
        scale: Output scale factor; 1, 2 and 4 are supported. Scales below 4
            are realised by pixel-unshuffling the input first, exactly as
            upstream does, so the checkpoint shapes match.
        num_feat: Base feature width.
        num_block: Number of RRDB blocks (23 for the standard models, 6 for
            the anime variant).
        num_grow_ch: Dense-block growth channels.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve ``x`` by :attr:`scale`."""
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # The network always upsamples by 4; smaller scales were absorbed by
        # the pixel-unshuffle above.
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """The compact VGG-style generator used by ``realesr-general-x4v3``.

    Much smaller and faster than :class:`RRDBNet`, at some cost in detail.
    Upstream layer naming (``body.N``) is preserved for checkpoint
    compatibility.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve ``x``, adding a nearest-neighbour skip connection."""
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base
