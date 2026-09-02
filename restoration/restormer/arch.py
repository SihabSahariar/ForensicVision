"""Restormer: efficient transformer for high-resolution image restoration.

Implemented natively so that neither ``basicsr`` nor ``einops`` is required;
the ``rearrange`` calls in the reference implementation are expressed here as
plain ``reshape``/``permute``, which is exactly equivalent for these patterns.

Layer names mirror the upstream implementation (``patch_embed``,
``encoder_level1..3``, ``latent``, ``decoder_level1..3``, ``refinement``,
``output``) so official Restormer checkpoints load unchanged.

Reference:
    Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image
    Restoration", CVPR 2022. Upstream code and weights are ACADEMIC/
    non-commercial licensed - see THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Restormer"]


def _to_3d(x: torch.Tensor) -> torch.Tensor:
    """``BxCxHxW`` -> ``Bx(HW)xC``."""
    return x.flatten(2).transpose(1, 2)


def _to_4d(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``Bx(HW)xC`` -> ``BxCxHxW``."""
    batch, _, channels = x.shape
    return x.transpose(1, 2).reshape(batch, channels, height, width)


class BiasFreeLayerNorm(nn.Module):
    """Layer norm without a mean subtraction or bias term."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise by the standard deviation along the channel axis."""
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """Standard layer norm over the channel axis."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise and apply the affine transform."""
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Channel-axis layer norm applied to a 4-D feature map."""

    def __init__(self, dim: int, layer_norm_type: str = "WithBias") -> None:
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body: nn.Module = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` over its channel dimension."""
        height, width = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), height, width)


class FeedForward(nn.Module):
    """Gated depth-wise convolutional feed-forward network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated depth-wise FFN."""
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """Multi-DConv-head transposed attention (MDTA).

    Attention is computed across *channels* rather than spatial positions, so
    cost is linear in pixel count - which is what makes the architecture usable
    on full-resolution evidence images.
    """

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel-wise attention over ``x``."""
        batch, channels, height, width = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = channels // self.num_heads
        shape = (batch, self.num_heads, head_dim, height * width)
        q = q.reshape(shape)
        k = k.reshape(shape)
        v = v.reshape(shape)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        out = attention @ v
        out = out.reshape(batch, channels, height, width)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    """One Restormer block: MDTA followed by GDFN, both residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feed-forward with residual connections."""
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class OverlapPatchEmbed(nn.Module):
    """3x3 convolutional patch embedding (overlapping, stride 1)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 48, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the image into the feature space."""
        return self.proj(x)


class Downsample(nn.Module):
    """Halve spatial size and double channels via pixel-unshuffle."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample ``x`` by a factor of two."""
        return self.body(x)


class Upsample(nn.Module):
    """Double spatial size and halve channels via pixel-shuffle."""

    def __init__(self, n_feat: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` by a factor of two."""
        return self.body(x)


class Restormer(nn.Module):
    """The Restormer encoder-decoder.

    Args:
        inp_channels: Input channel count.
        out_channels: Output channel count.
        dim: Base feature width.
        num_blocks: Transformer blocks at each of the four scales.
        num_refinement_blocks: Blocks in the final refinement stage.
        heads: Attention heads at each scale.
        ffn_expansion_factor: GDFN hidden expansion.
        bias: Whether convolutions carry a bias term.
        layer_norm_type: ``"WithBias"`` or ``"BiasFree"``.
        dual_pixel_task: Enable the dual-pixel defocus skip connection.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        dual_pixel_task: bool = False,
    ) -> None:
        super().__init__()

        def _stage(width: int, head_count: int, count: int) -> nn.Sequential:
            return nn.Sequential(
                *[
                    TransformerBlock(
                        dim=width,
                        num_heads=head_count,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        layer_norm_type=layer_norm_type,
                    )
                    for _ in range(count)
                ]
            )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        self.encoder_level1 = _stage(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = _stage(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = _stage(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = _stage(dim * 8, heads[3], num_blocks[3])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = _stage(dim * 4, heads[2], num_blocks[2])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = _stage(dim * 2, heads[1], num_blocks[1])

        self.up2_1 = Upsample(dim * 2)
        # Level 1 of the decoder keeps 2*dim channels: the skip connection is
        # concatenated without a channel reduction, matching upstream.
        self.decoder_level1 = _stage(dim * 2, heads[0], num_blocks[0])
        self.refinement = _stage(dim * 2, heads[0], num_refinement_blocks)

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.output = nn.Conv2d(
            dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        """Restore ``inp_img``; spatial size must be a multiple of 8."""
        enc1_in = self.patch_embed(inp_img)
        enc1 = self.encoder_level1(enc1_in)

        enc2 = self.encoder_level2(self.down1_2(enc1))
        enc3 = self.encoder_level3(self.down2_3(enc2))
        latent = self.latent(self.down3_4(enc3))

        dec3 = self.up4_3(latent)
        dec3 = self.reduce_chan_level3(torch.cat([dec3, enc3], 1))
        dec3 = self.decoder_level3(dec3)

        dec2 = self.up3_2(dec3)
        dec2 = self.reduce_chan_level2(torch.cat([dec2, enc2], 1))
        dec2 = self.decoder_level2(dec2)

        dec1 = self.up2_1(dec2)
        dec1 = self.decoder_level1(torch.cat([dec1, enc1], 1))
        dec1 = self.refinement(dec1)

        if self.dual_pixel_task:
            dec1 = dec1 + self.skip_conv(enc1_in)
            return self.output(dec1)
        return self.output(dec1) + inp_img
