"""Proposed neural-to-latent adaptor and its controlled component variants.

The default constructor reproduces the original five-stage architecture. The
``dilations`` and attention arguments are exposed so component ablations can be
selected from ``config.json`` without maintaining a second model copy.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """Residual local/multiscale temporal block with optional self-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        use_attention: bool = False,
        attention_heads: int = 8,
        channel_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        if out_channels % 8 != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by 8 for GroupNorm"
            )
        if use_attention and out_channels % attention_heads != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by "
                f"attention_heads={attention_heads}"
            )
        if channel_kernel_size < 1 or temporal_kernel_size < 1:
            raise ValueError(
                "channel_kernel_size and temporal_kernel_size must be positive, "
                f"got {channel_kernel_size} and {temporal_kernel_size}"
            )

        # This projection changes features and also sees a local 3-step window.
        self.channel_mixer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=channel_kernel_size,
            stride=1,
            padding="same",
        )
        self.temporal_conv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=temporal_kernel_size,
            padding="same",
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act = nn.Tanh()

        self.use_attention = bool(use_attention)
        if self.use_attention:
            self.att_norm = nn.LayerNorm(out_channels)
            self.attn = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=attention_heads,
                batch_first=True,
            )

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.channel_mixer(x)
        x = self.temporal_conv(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.use_attention:
            # [B, C, T] -> [B, T, C]: attention acts across time steps.
            x_time_first = self.att_norm(x.permute(0, 2, 1))
            attention_output, _ = self.attn(
                x_time_first,
                x_time_first,
                x_time_first,
                need_weights=False,
            )
            x = x + attention_output.permute(0, 2, 1)
        return x + residual


class Adaptor_model(nn.Module):
    """Progressive multiscale-convolution and self-attention adaptor.

    Shape contract:
        input:  ``[B, T_neural, neu_F_dim]``
        output: ``[B, latent_F_dim, latent_T_dim]``

    Defaults reproduce the proposed model: hidden widths
    ``[2048, 1024, 512, 256]``, dilations ``[1, 2, 4, 8, 1]``, and temporal
    self-attention in blocks ``[3, 4, 5]``.
    """

    def __init__(
        self,
        neu_F_dim: int = 4667,
        latent_T_dim: int = 52,
        latent_F_dim: int = 64,
        hidden_widths: Sequence[int] = (2048, 1024, 512, 256),
        dilations: Sequence[int] = (1, 2, 4, 8, 1),
        use_attention: bool = True,
        attention_blocks: Sequence[int] = (3, 4, 5),
        attention_heads: int = 8,
        channel_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        sigma_rescale: float = 0.06,
    ) -> None:
        super().__init__()
        hidden_widths = tuple(int(width) for width in hidden_widths)
        dilations = tuple(int(dilation) for dilation in dilations)
        attention_blocks = tuple(int(index) for index in attention_blocks)
        num_blocks = len(hidden_widths) + 1

        if len(dilations) != num_blocks:
            raise ValueError(
                f"dilations must contain {num_blocks} values, got {len(dilations)}"
            )
        if any(dilation < 1 for dilation in dilations):
            raise ValueError(f"all dilations must be >= 1, got {dilations}")
        invalid_attention_blocks = [
            index for index in attention_blocks if not 1 <= index <= num_blocks
        ]
        if invalid_attention_blocks:
            raise ValueError(
                f"attention_blocks must be within [1, {num_blocks}], got "
                f"{invalid_attention_blocks}"
            )
        if sigma_rescale <= 0:
            raise ValueError(f"sigma_rescale must be positive, got {sigma_rescale}")

        self.target_time = int(latent_T_dim)
        self.sigma_rescale = float(sigma_rescale)
        self.dilations = dilations
        self.attention_blocks = attention_blocks if use_attention else tuple()
        self.num_blocks = num_blocks

        channel_widths = (int(neu_F_dim), *hidden_widths, int(latent_F_dim))
        attention_block_set = set(self.attention_blocks)
        for block_index in range(1, num_blocks + 1):
            block = ResBlock(
                in_channels=channel_widths[block_index - 1],
                out_channels=channel_widths[block_index],
                dilation=dilations[block_index - 1],
                use_attention=block_index in attention_block_set,
                attention_heads=int(attention_heads),
                channel_kernel_size=int(channel_kernel_size),
                temporal_kernel_size=int(temporal_kernel_size),
            )
            # Preserve block1..block5 state_dict names from the original model.
            setattr(self, f"block{block_index}", block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input [B, T, F], got shape {tuple(x.shape)}")
        x = x.permute(0, 2, 1)
        for block_index in range(1, self.num_blocks + 1):
            x = getattr(self, f"block{block_index}")(x)

        # Late alignment keeps all temporal modeling on the neural time grid.
        x = F.interpolate(
            x, size=self.target_time, mode="linear", align_corners=False
        )
        return x / self.sigma_rescale


if __name__ == "__main__":
    output = Adaptor_model()(torch.randn(2, 90, 4667))
    print(output.shape)
