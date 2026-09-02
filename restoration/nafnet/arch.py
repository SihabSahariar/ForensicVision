"""NAFNet: Nonlinear Activation Free Network for image restoration.

The architecture replaces activations with a "SimpleGate" (an element-wise
product of two channel halves) and channel attention with a single pooled 1x1
convolution, which makes it unusually cheap for its restoration quality.

Layer names mirror the reference implementation (``intro``, ``encoders.N.M``,
``downs.N``, ``middle_blks.N``, ``ups.N``, ``decoders.N.M``, ``ending``) so the
published NAFNet checkpoints load unchanged.

Reference:
    Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.
    Upstream code and weights are MIT-licensed but the released weights are
    distributed for non-commercial research - see THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "NAFNet",
    "NAFBlock",
    "LayerNorm2d",
    "SimpleGate",
    "infer_config_from_state_dict",
]


class LayerNorm2d(nn.Module):
    """Layer normalisation over the channel axis of a 4-D feature map.

    Parameters are registered as flat ``weight``/``bias`` vectors to match the
    upstream checkpoint layout.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` per position across channels, then scale and shift."""
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * y + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split the channels in half and multiply - NAFNet's activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the element-wise product of the two channel halves."""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """One NAFNet block: gated depth-wise conv plus a gated feed-forward stage."""

    def __init__(
        self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0
    ) -> None:
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        # Simplified channel attention: global pool then a 1x1 projection.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply both residual stages with their learned scaling factors."""
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """UNet-shaped NAFNet.

    Args:
        img_channel: Input/output channel count.
        width: Base feature width.
        middle_blk_num: Blocks at the bottleneck.
        enc_blk_nums: Blocks in each encoder stage.
        dec_blk_nums: Blocks in each decoder stage.
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: Sequence[int] = (2, 2, 4, 8),
        dec_blk_nums: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()

        self.intro = nn.Conv2d(
            img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.ending = nn.Conv2d(
            width, img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Right/bottom-pad ``x`` so both dimensions divide by the stage count."""
        _, _, height, width = x.size()
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Restore ``inp``; padding is handled internally and then cropped."""
        _, _, height, width = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :height, :width]


def infer_config_from_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a NAFNet configuration from a checkpoint's key structure.

    Upstream publishes several block layouts (GoPro uses ``[1, 1, 1, 28]`` with
    a single middle block; SIDD uses ``[2, 2, 4, 8]`` with twelve) and the files
    are distributed without their YAML configs. Reading the shape back out of
    the checkpoint means any published NAFNet variant loads correctly, instead
    of only the two or three whose layouts were hard-coded.

    Args:
        state: A flat ``{name: tensor}`` checkpoint mapping.

    Returns:
        Keyword arguments for :class:`NAFNet`.

    Raises:
        ValueError: The mapping does not look like a NAFNet checkpoint.
    """
    intro = state.get("intro.weight")
    if intro is None:
        raise ValueError(
            "Checkpoint has no 'intro.weight'; it does not look like a NAFNet model."
        )
    width = int(intro.shape[0])
    img_channel = int(intro.shape[1])

    def _stage_counts(prefix: str) -> List[int]:
        pattern = re.compile(rf"^{prefix}\.(\d+)\.(\d+)\.")
        seen: Dict[int, int] = {}
        for key in state:
            match = pattern.match(key)
            if match:
                stage, block = int(match.group(1)), int(match.group(2))
                seen[stage] = max(seen.get(stage, -1), block)
        return [seen[i] + 1 for i in sorted(seen)]

    def _flat_count(prefix: str) -> int:
        pattern = re.compile(rf"^{prefix}\.(\d+)\.")
        highest = -1
        for key in state:
            match = pattern.match(key)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest + 1

    enc = _stage_counts("encoders")
    dec = _stage_counts("decoders")
    middle = _flat_count("middle_blks")

    if not enc or not dec:
        raise ValueError("Checkpoint has no encoder/decoder blocks; not a NAFNet model.")

    return {
        "img_channel": img_channel,
        "width": width,
        "middle_blk_num": middle,
        "enc_blk_nums": enc,
        "dec_blk_nums": dec,
    }
