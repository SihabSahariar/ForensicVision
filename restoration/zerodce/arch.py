"""Zero-DCE and Zero-DCE++ curve-estimation networks.

Both networks are reimplemented here with upstream layer naming
(``e_conv1``..``e_conv7``, and ``depth_conv``/``point_conv`` inside each
Zero-DCE++ block) so the official checkpoints load with no key remapping.

What these networks output is unusual and worth stating plainly, because it
decides how the rest of the application treats them: **they do not output an
image**. They output the coefficients of a tone curve, and the curve is then
applied to the input. The enhancement step is

.. math:: \\mathrm{LE}(x; r) = x + r\\,x\\,(x - 1)

applied eight times, with :math:`r = \\tanh(\\cdot) \\in [-1, 1]` estimated per
pixel and per channel. Its derivative is :math:`1 + r(2x - 1)`, which over
:math:`x \\in [0, 1]`, :math:`r \\in [-1, 1]` has minimum 0 and maximum 2 - so
the map is monotonically non-decreasing, and so is any composition of it. A
pixel's output therefore depends on that pixel's own value through a monotone
function; the network chooses *which* monotone function, it does not paint
pixels. See :func:`enhance` and ``tests/test_zerodce.py``, which assert this.

* **Zero-DCE** predicts eight separate curve maps (24 output channels) at full
  resolution.
* **Zero-DCE++** predicts one shared curve map (3 output channels) with
  depth-separable convolutions, optionally at a reduced resolution that is
  bilinearly upsampled - which makes it far cheaper and constrains the curve
  map to low spatial frequencies.

References:
    Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image
    Enhancement", CVPR 2020.
    Li, Guo & Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep
    Curve Estimation", IEEE TPAMI 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ZeroDCE", "ZeroDCEPlusPlus", "CSDNTem", "enhance", "ITERATIONS"]

#: Number of times the curve is applied. Fixed by both published architectures.
ITERATIONS: int = 8


def enhance(x: torch.Tensor, curves) -> torch.Tensor:
    """Apply the light-enhancement curve ``ITERATIONS`` times.

    Args:
        x: ``Nx3xHxW`` input in ``[0, 1]``.
        curves: Either one ``Nx3xHxW`` curve map reused for every iteration
            (Zero-DCE++), or a sequence of ``ITERATIONS`` such maps (Zero-DCE).

    Returns:
        The enhanced tensor, same shape as ``x``.
    """
    if torch.is_tensor(curves):
        curves = [curves] * ITERATIONS
    for curve in curves:
        x = x + curve * (torch.pow(x, 2) - x)
    return x


class ZeroDCE(nn.Module):
    """Zero-DCE: seven plain 3x3 convolutions predicting eight curve maps.

    Symmetric skip concatenation joins conv3+conv4, conv2+conv5 and conv1+conv6,
    which is why ``e_conv5``..``e_conv7`` take ``2 * width`` input channels.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(width, width, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(width, width, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(width, width, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(width * 2, width, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(width * 2, width, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(width * 2, 3 * ITERATIONS, 3, 1, 1, bias=True)

    def curve_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw ``Nx24xHxW`` curve parameters in ``[-1, 1]``."""
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance ``x``; returns a tensor of the same shape."""
        maps = self.curve_maps(x)
        return enhance(x, torch.split(maps, 3, dim=1))


class CSDNTem(nn.Module):
    """Depth-separable convolution block used throughout Zero-DCE++.

    A depthwise 3x3 followed by a pointwise 1x1. The attribute names match the
    published checkpoint, which stores ``depth_conv`` and ``point_conv``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the depthwise then the pointwise convolution."""
        return self.point_conv(self.depth_conv(x))


class ZeroDCEPlusPlus(nn.Module):
    """Zero-DCE++: depth-separable, one shared curve map, optional downscaling.

    ``scale_factor`` estimates the curve map on an input reduced by that factor
    and bilinearly upsamples it back. Upstream's reference script crops the
    input to a multiple of ``scale_factor``; cropping evidence is not acceptable
    here, so this implementation resizes by explicit target size instead. For an
    input whose dimensions *are* a multiple the two are numerically identical -
    ``tests/test_zerodce.py`` asserts that against a transcription of the
    upstream forward pass.
    """

    def __init__(self, width: int = 32, scale_factor: int = 1) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = int(scale_factor)
        self.e_conv1 = CSDNTem(3, width)
        self.e_conv2 = CSDNTem(width, width)
        self.e_conv3 = CSDNTem(width, width)
        self.e_conv4 = CSDNTem(width, width)
        self.e_conv5 = CSDNTem(width * 2, width)
        self.e_conv6 = CSDNTem(width * 2, width)
        self.e_conv7 = CSDNTem(width * 2, 3)

    def curve_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the single ``Nx3xHxW`` curve map for ``x`` in ``[-1, 1]``."""
        height, width = x.shape[-2:]
        scale = max(1, self.scale_factor)

        if scale == 1:
            source = x
        else:
            source = F.interpolate(
                x,
                size=(max(1, height // scale), max(1, width // scale)),
                mode="bilinear",
                align_corners=False,
            )

        x1 = self.relu(self.e_conv1(source))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        if scale != 1:
            # align_corners=True matches upstream's nn.UpsamplingBilinear2d.
            curve = F.interpolate(
                curve, size=(height, width), mode="bilinear", align_corners=True
            )
        return curve

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance ``x``; returns a tensor of the same shape."""
        return enhance(x, self.curve_map(x))
