"""SwinIR: image restoration using Swin Transformer.

Implemented natively (no ``timm`` dependency) with layer names matching the
reference implementation, so the published ``00N_*.pth`` checkpoints load
unchanged.

The ``timm`` helpers the original imports - ``DropPath``, ``trunc_normal_`` and
``to_2tuple`` - are reproduced here; only ``DropPath`` has any effect at
inference time, and that effect is the identity in ``eval`` mode.

Reference:
    Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCVW
    2021. Upstream code is Apache-2.0.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwinIR"]


def to_2tuple(value) -> Tuple[int, int]:
    """Return ``value`` as a pair, duplicating scalars."""
    if isinstance(value, (tuple, list)):
        return tuple(value)  # type: ignore[return-value]
    return (value, value)


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """In-place truncated normal initialisation (inference-irrelevant)."""
    with torch.no_grad():
        tensor.normal_(0.0, std).clamp_(-2 * std, 2 * std)
    return tensor


class DropPath(nn.Module):
    """Stochastic depth. Identity during evaluation, which is all we use."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged in eval mode; drop paths during training."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()


class Mlp(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network."""
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Split ``BxHxWxC`` into ``(B*nW)x(ws*ws)xC`` windows."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch, height // window_size, window_size,
        width // window_size, window_size, channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    """Inverse of :func:`window_partition`."""
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(
        batch, height // window_size, width // window_size,
        window_size, window_size, -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with a relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Attend within each window, optionally applying a shift mask."""
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            windows = mask.shape[0]
            attn = attn.view(
                batch // windows, windows, self.num_heads, tokens, tokens
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """One (shifted-)window transformer block."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        attn_mask = self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, resolution: Tuple[int, int]) -> torch.Tensor:
        """Build the attention mask that isolates shifted-window regions."""
        height, width = resolution
        img_mask = torch.zeros((1, height, width, 1))
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in slices:
            for w_slice in slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Apply windowed attention and the feed-forward block."""
        height, width = x_size
        batch, _, channels = x.shape

        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted = x

        x_windows = window_partition(shifted, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)

        # The cached mask is built for the training resolution; a different
        # inference tile size needs its own mask.
        if self.input_resolution == x_size:
            mask = self.attn_mask
        elif self.shift_size > 0:
            mask = self.calculate_mask(x_size).to(x.device)
        else:
            mask = None

        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attn_windows, self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        x = x.view(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    """Flatten a feature map into a token sequence, with optional norm."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``BxCxHxW`` -> ``Bx(HW)xC``."""
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Inverse of :class:`PatchEmbed`."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """``Bx(HW)xC`` -> ``BxCxHxW``."""
        batch, _, _ = x.shape
        return x.transpose(1, 2).view(batch, self.embed_dim, x_size[0], x_size[1])


class BasicLayer(nn.Module):
    """A stack of :class:`SwinTransformerBlock` at one resolution."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Run every block at ``x_size``."""
        for block in self.blocks:
            x = block(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: a :class:`BasicLayer` plus a conv."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
        img_size: int = 224,
        patch_size: int = 4,
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown residual connection: {resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0,
            embed_dim=dim, norm_layer=None,
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        """Apply the transformer group, then the residual convolution."""
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )


class Upsample(nn.Sequential):
    """Pixel-shuffle upsampler for scales that are powers of two, or 3."""

    def __init__(self, scale: int, num_feat: int) -> None:
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*layers)


class UpsampleOneStep(nn.Sequential):
    """Single-step upsampler used by the lightweight SwinIR variants."""

    def __init__(
        self, scale: int, num_feat: int, num_out_ch: int,
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        super().__init__(
            nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )


class SwinIR(nn.Module):
    """SwinIR for super-resolution, denoising and compression-artefact removal.

    Args:
        img_size: Training patch size; only affects cached masks.
        patch_size: Always 1 in SwinIR.
        in_chans: Input channels.
        embed_dim: Token width.
        depths: Blocks per RSTB stage.
        num_heads: Attention heads per stage.
        window_size: Attention window edge.
        mlp_ratio: Feed-forward expansion.
        upscale: Output scale (1 for denoise/CAR).
        img_range: Input scaling; 1.0 or 255.0 depending on the checkpoint.
        upsampler: ``"pixelshuffle"``, ``"pixelshuffledirect"``,
            ``"nearest+conv"`` or ``""`` for same-size tasks.
        resi_connection: ``"1conv"`` or ``"3conv"``.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer=nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "",
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            self.layers.append(
                RSTB(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[index],
                    num_heads=num_heads[index],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:index]): sum(depths[: index + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                )
            )
        self.norm = norm_layer(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        else:
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif self.upsampler == "nearest+conv":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise linear and norm layers (irrelevant once weights load)."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Reflect-pad ``x`` so both dimensions are a window-size multiple."""
        _, _, height, width = x.size()
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), "reflect")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer body."""
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore ``x``; padding and mean normalisation are handled here."""
        height, width = x.shape[2], x.shape[3]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest"))
            )
            if self.upscale == 4:
                x = self.lrelu(
                    self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest"))
                )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            first = self.conv_first(x)
            body = self.conv_after_body(self.forward_features(first)) + first
            x = x + self.conv_last(body)

        x = x / self.img_range + self.mean
        return x[:, :, : height * self.upscale, : width * self.upscale]
