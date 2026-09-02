"""CodeFormer: blind face restoration with a codebook lookup transformer.

Three parts:

1. A **VQGAN autoencoder** whose discrete codebook holds a learned vocabulary
   of high-quality face patches.
2. A **transformer** that predicts, from the degraded input's features, which
   codebook entry belongs at each of the 16x16 latent positions. This is the
   step that makes the method robust to heavy degradation - and the step that
   makes its output a *reconstruction from a prior over faces* rather than a
   recovery of the subject.
3. **Controllable feature transformation** blocks that mix encoder features
   back into the decoder, weighted by a fidelity parameter ``w``. At ``w = 0``
   the output is pure codebook reconstruction (highest quality, lowest
   fidelity to the input); at ``w = 1`` the encoder's own features dominate
   (closest to the input, least "restored").

Layer naming follows the reference implementation (``encoder.blocks.N``,
``quantize.embedding``, ``generator.blocks.N``, ``ft_layers.N``,
``idx_pred_layer``, ``fuse_convs_dict.<size>``) so the published
``codeformer.pth`` loads unchanged.

Reference:
    Zhou, Chan, Li & Loy, "Towards Robust Blind Face Restoration with Codebook
    Lookup Transformer", NeurIPS 2022. Upstream code and weights are covered by
    the S-Lab License 1.0 (non-commercial research use only).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CodeFormerNet", "VQAutoEncoder"]


def normalize(in_channels: int) -> nn.GroupNorm:
    """Return the group norm used throughout the VQGAN."""
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def swish(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation, written explicitly to match the reference."""
    return x * torch.sigmoid(x)


# --------------------------------------------------------------------------- #
# VQGAN building blocks
# --------------------------------------------------------------------------- #

class VectorQuantizer(nn.Module):
    """The learned discrete codebook."""

    def __init__(self, codebook_size: int, emb_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)

    def get_codebook_feat(
        self, indices: torch.Tensor, shape: Sequence[int]
    ) -> torch.Tensor:
        """Look up codebook entries and reshape them to a feature map.

        Args:
            indices: Flat entry indices.
            shape: Target ``(B, H, W, C)`` before the channel permutation.
        """
        indices = indices.view(-1, 1)
        one_hot = torch.zeros(
            indices.shape[0], self.codebook_size, device=indices.device
        )
        one_hot.scatter_(1, indices, 1)
        quantised = torch.matmul(one_hot.float(), self.embedding.weight)
        if shape is not None:
            quantised = quantised.view(shape).permute(0, 3, 1, 2).contiguous()
        return quantised


class ResBlock(nn.Module):
    """Group-normalised residual block."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
        self.norm2 = normalize(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, self.out_channels, 1, 1, 0)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Apply the block with its skip connection."""
        x = self.conv1(swish(self.norm1(x_in)))
        x = self.conv2(swish(self.norm2(x)))
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in


class AttnBlock(nn.Module):
    """Single-head spatial self-attention over a feature map."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend spatially and add the result to the input."""
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)

        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        weights = torch.bmm(q, k) * (int(channels) ** -0.5)
        weights = F.softmax(weights, dim=2)

        v = v.reshape(batch, channels, height * width)
        h = torch.bmm(v, weights.permute(0, 2, 1))
        h = h.reshape(batch, channels, height, width)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    """Stride-2 convolution with asymmetric padding, as upstream."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Halve the spatial size."""
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    """Nearest-neighbour upsample followed by a convolution."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double the spatial size."""
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    """VQGAN encoder: image -> latent feature map."""

    def __init__(
        self,
        in_channels: int,
        nf: int,
        emb_dim: int,
        ch_mult: Sequence[int],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: Sequence[int],
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = [nn.Conv2d(in_channels, nf, 3, 1, 1)]

        current_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = nf
        for level in range(len(ch_mult)):
            block_in = nf * in_ch_mult[level]
            block_out = nf * ch_mult[level]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in, block_out))
                block_in = block_out
                if current_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if level != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                current_res //= 2

        blocks.append(ResBlock(block_in, block_in))
        blocks.append(AttnBlock(block_in))
        blocks.append(ResBlock(block_in, block_in))
        blocks.append(normalize(block_in))
        blocks.append(nn.Conv2d(block_in, emb_dim, 3, 1, 1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` to the latent feature map."""
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    """VQGAN decoder: latent feature map -> image."""

    def __init__(
        self,
        nf: int,
        emb_dim: int,
        ch_mult: Sequence[int],
        res_blocks: int,
        img_size: int,
        attn_resolutions: Sequence[int],
    ) -> None:
        super().__init__()
        block_in = nf * ch_mult[-1]
        current_res = img_size // 2 ** (len(ch_mult) - 1)

        blocks: List[nn.Module] = [
            nn.Conv2d(emb_dim, block_in, 3, 1, 1),
            ResBlock(block_in, block_in),
            AttnBlock(block_in),
            ResBlock(block_in, block_in),
        ]

        for level in reversed(range(len(ch_mult))):
            block_out = nf * ch_mult[level]
            for _ in range(res_blocks):
                blocks.append(ResBlock(block_in, block_out))
                block_in = block_out
                if current_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))
            if level != 0:
                blocks.append(Upsample(block_in))
                current_res *= 2

        blocks.append(normalize(block_in))
        blocks.append(nn.Conv2d(block_in, 3, 3, 1, 1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode ``x`` back to an image."""
        for block in self.blocks:
            x = block(x)
        return x


class VQAutoEncoder(nn.Module):
    """Encoder + codebook + generator, matching the upstream module layout."""

    def __init__(
        self,
        img_size: int = 512,
        nf: int = 64,
        ch_mult: Sequence[int] = (1, 2, 2, 4, 4, 8),
        res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        codebook_size: int = 1024,
        emb_dim: int = 256,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = 3
        self.nf = nf
        self.embed_dim = emb_dim
        self.ch_mult = list(ch_mult)
        self.resolution = img_size
        self.n_blocks = res_blocks
        self.codebook_size = codebook_size
        self.attn_resolutions = list(attn_resolutions)

        self.encoder = Encoder(
            self.in_channels, nf, emb_dim, self.ch_mult, res_blocks,
            self.resolution, self.attn_resolutions,
        )
        self.quantize = VectorQuantizer(codebook_size, emb_dim, beta)
        self.generator = Generator(
            nf, emb_dim, self.ch_mult, res_blocks, self.resolution,
            self.attn_resolutions,
        )


# --------------------------------------------------------------------------- #
# Transformer
# --------------------------------------------------------------------------- #

class TransformerSALayer(nn.Module):
    """Pre-norm self-attention layer with a GELU feed-forward block."""

    def __init__(
        self,
        embed_dim: int,
        nhead: int = 8,
        dim_mlp: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, target: torch.Tensor, query_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply self-attention and the feed-forward block, both residual."""
        normed = self.norm1(target)
        query = key = normed if query_pos is None else normed + query_pos
        attended = self.self_attn(query, key, value=normed)[0]
        target = target + self.dropout1(attended)

        normed = self.norm2(target)
        projected = self.linear2(self.dropout(F.gelu(self.linear1(normed))))
        return target + self.dropout2(projected)


class FuseSftBlock(nn.Module):
    """Controllable feature transformation: scale/shift the decoder features."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(
        self, enc_feat: torch.Tensor, dec_feat: torch.Tensor, w: float = 1.0
    ) -> torch.Tensor:
        """Blend encoder detail into the decoder stream, weighted by ``w``."""
        fused = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(fused)
        shift = self.shift(fused)
        return dec_feat + w * (dec_feat * scale + shift)


class CodeFormerNet(VQAutoEncoder):
    """The full CodeFormer model."""

    def __init__(
        self,
        dim_embd: int = 512,
        n_head: int = 8,
        n_layers: int = 9,
        codebook_size: int = 1024,
        latent_size: int = 256,
        connect_list: Sequence[str] = ("32", "64", "128", "256"),
    ) -> None:
        super().__init__(
            img_size=512, nf=64, ch_mult=(1, 2, 2, 4, 4, 8), res_blocks=2,
            attn_resolutions=(16,), codebook_size=codebook_size,
        )

        self.connect_list = list(connect_list)
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        self.ft_layers = nn.Sequential(
            *[
                TransformerSALayer(
                    embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp,
                    dropout=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False),
        )

        self.channels: Dict[str, int] = {
            "16": 512, "32": 256, "64": 256, "128": 128, "256": 128, "512": 64,
        }
        #: Encoder block index whose output feeds each skip connection.
        self.fuse_encoder_block: Dict[str, int] = {
            "512": 2, "256": 5, "128": 8, "64": 11, "32": 14, "16": 18,
        }
        #: Generator block index after which each skip is fused back in.
        self.fuse_generator_block: Dict[str, int] = {
            "16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21,
        }

        self.fuse_convs_dict = nn.ModuleDict()
        for size in self.connect_list:
            channels = self.channels[size]
            self.fuse_convs_dict[size] = FuseSftBlock(channels, channels)

    def forward(
        self, x: torch.Tensor, w: float = 0.5, adain: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Restore a batch of aligned faces.

        Args:
            x: ``Bx3x512x512`` tensor in ``[-1, 1]``.
            w: Fidelity weight. 0 favours codebook quality, 1 favours
                similarity to the input.
            adain: Match the codebook features' statistics to the input's.

        Returns:
            ``(restored, logits)`` where ``restored`` is in ``[-1, 1]``.
        """
        # ---- encoder, capturing the skip features -------------------------
        encoder_features: Dict[str, torch.Tensor] = {}
        capture = {self.fuse_encoder_block[size] for size in self.connect_list}
        for index, block in enumerate(self.encoder.blocks):
            x = block(x)
            if index in capture:
                encoder_features[str(x.shape[-1])] = x.clone()

        latent = x

        # ---- transformer predicts codebook indices -------------------------
        position = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        query = self.feat_emb(latent.flatten(2).permute(2, 0, 1))
        for layer in self.ft_layers:
            query = layer(query, query_pos=position)

        logits = self.idx_pred_layer(query).permute(1, 0, 2)

        # ---- codebook lookup ----------------------------------------------
        _, top_index = torch.topk(F.softmax(logits, dim=2), 1, dim=2)
        quantised = self.quantize.get_codebook_feat(
            top_index, shape=[x.shape[0], 16, 16, 256]
        )
        if adain:
            quantised = _adaptive_instance_normalisation(quantised, latent)

        # ---- generator, fusing encoder detail back in ----------------------
        out = quantised
        fuse = {self.fuse_generator_block[size] for size in self.connect_list}
        for index, block in enumerate(self.generator.blocks):
            out = block(out)
            if index in fuse and w > 0:
                size = str(out.shape[-1])
                if size in self.fuse_convs_dict:
                    out = self.fuse_convs_dict[size](
                        encoder_features[size].detach(), out, w
                    )

        return out, logits


def _calc_mean_std(
    feat: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel mean and standard deviation of a feature map."""
    size = feat.size()
    batch, channels = size[:2]
    variance = feat.view(batch, channels, -1).var(dim=2) + eps
    std = variance.sqrt().view(batch, channels, 1, 1)
    mean = feat.view(batch, channels, -1).mean(dim=2).view(batch, channels, 1, 1)
    return mean, std


def _adaptive_instance_normalisation(
    content: torch.Tensor, style: torch.Tensor
) -> torch.Tensor:
    """Match ``content``'s channel statistics to ``style``'s."""
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    normalised = (content - content_mean.expand(content.size())) / content_std.expand(
        content.size()
    )
    return normalised * style_std.expand(content.size()) + style_mean.expand(
        content.size()
    )
