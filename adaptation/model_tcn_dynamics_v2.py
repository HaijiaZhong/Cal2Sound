"""Calcium-informed recurrent TCN for neural-to-latent adaptation.

This model preserves the residual dilated-TCN baseline and inserts a lightweight
calcium-aware recurrent refinement module between the input projection and TCN.

Default proposed model:
    neural input
      -> baseline-compatible input projection
      -> CalciumGRURefinement (unidirectional or bidirectional)
      -> unchanged residual dilated TCN
      -> output projection
      -> temporal interpolation

The switches are intentionally designed for clean ablations:
    use_recurrent_refinement=False
        -> pure TCN baseline architecture
    use_recurrent_refinement=True, use_calcium_innovation=False
        -> TCN + vanilla GRU refinement
    use_recurrent_refinement=True, use_calcium_innovation=True
        -> proposed TCN + calcium-informed GRU refinement

Input:
    [B, T_neural, neu_F_dim]

Output:
    [B, latent_F_dim, latent_T_dim]
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    """Non-causal constant-width residual TCN block.

    Kept identical to model_tcn.py so that the TCN backbone is not changed.
    """

    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float = 0.1,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if channels % norm_groups != 0:
            raise ValueError(
                f"channels={channels} must be divisible by norm_groups={norm_groups}"
            )
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        layers: list[nn.Module] = []
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(norm_groups, channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _logit(probability: float) -> float:
    """Numerically stable scalar logit used for initialization."""
    eps = 1e-6
    probability = min(max(float(probability), eps), 1.0 - eps)
    return math.log(probability / (1.0 - probability))


class CalciumGRURefinement(nn.Module):
    """Calcium-informed recurrent refinement in the projected neural space.

    Let z_t be the projected calcium feature at imaging frame t.  A lightweight
    AR(1)-style persistence model is used to form a calcium innovation:

        e_t = z_t - gamma * z_{t-1}

    where gamma is a learnable channel-wise retention coefficient constrained
    to (gamma_min, gamma_max).

    The GRU receives:
        z_t + innovation_scale * e_t

    instead of only z_t.  Thus the recurrent state can jointly use the observed
    calcium level and the part that is not explained by calcium persistence.

    The GRU output is added back through a learnable residual gate beta:

        z'_t = z_t + beta * GRU(...)

    Initializing beta to a small value keeps the proposed network close to the
    already-tested TCN baseline at the beginning of optimization.

    Setting use_calcium_innovation=False yields a vanilla GRU refinement with
    the same hidden size and recurrent backbone, which is useful as a direct
    ablation of the calcium-specific inductive bias.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.1,
        gru_layers: int = 1,
        bidirectional: bool = False,
        use_calcium_innovation: bool = True,
        decay_init: float = 0.95,
        gamma_min: float = 0.50,
        gamma_max: float = 0.999,
        innovation_scale_init: float = 0.10,
        residual_scale_init: float = 0.10,
    ) -> None:
        super().__init__()

        channels = int(channels)
        gru_layers = int(gru_layers)

        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if bidirectional and channels % 2 != 0:
            raise ValueError(
                "channels must be even when bidirectional=True so that the "
                "concatenated GRU directions preserve the channel width"
            )
        if gru_layers < 1:
            raise ValueError(f"gru_layers must be >= 1, got {gru_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not 0.0 <= gamma_min < gamma_max <= 1.0:
            raise ValueError(
                "gamma bounds must satisfy 0 <= gamma_min < gamma_max <= 1"
            )
        if not gamma_min < decay_init < gamma_max:
            raise ValueError(
                f"decay_init={decay_init} must lie strictly inside "
                f"({gamma_min}, {gamma_max})"
            )

        self.channels = channels
        self.use_calcium_innovation = bool(use_calcium_innovation)
        self.bidirectional = bool(bidirectional)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

        # Parameterize gamma inside a bounded interval for stable AR-like
        # persistence. One gamma is learned per projected neural channel.
        normalized_init = (
            (float(decay_init) - self.gamma_min)
            / (self.gamma_max - self.gamma_min)
        )
        self.decay_logits = nn.Parameter(
            torch.full((channels,), _logit(normalized_init))
        )

        # Learnable strength of the calcium innovation entering the GRU.
        # A scalar is deliberately used to keep the added physics-informed
        # component lightweight and easy to interpret.
        self.innovation_scale = nn.Parameter(
            torch.tensor(float(innovation_scale_init))
        )

        self.input_norm = nn.LayerNorm(channels)
        # For a bidirectional GRU, use half the channels per direction so the
        # concatenated output remains C-dimensional and can be residual-added
        # to the original projected calcium feature z_t.
        hidden_size = channels // 2 if self.bidirectional else channels
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=float(dropout) if gru_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        self.output_norm = nn.LayerNorm(channels)

        # Residual insertion starts close to the original TCN baseline.
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def calcium_retention(self) -> torch.Tensor:
        """Return channel-wise gamma values constrained to the configured range."""
        unit_interval = torch.sigmoid(self.decay_logits)
        return self.gamma_min + (
            self.gamma_max - self.gamma_min
        ) * unit_interval

    def calcium_innovation(self, sequence: torch.Tensor) -> torch.Tensor:
        """Compute AR(1)-style innovation for [B, T, C] projected features."""
        if sequence.ndim != 3:
            raise ValueError(
                f"expected projected sequence [B, T, C], got {tuple(sequence.shape)}"
            )

        gamma = self.calcium_retention().view(1, 1, -1)

        innovation = torch.empty_like(sequence)
        # No previous observation is available for the first frame.
        innovation[:, :1] = sequence[:, :1]
        if sequence.size(1) > 1:
            innovation[:, 1:] = (
                sequence[:, 1:] - gamma * sequence[:, :-1]
            )
        return innovation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine projected features x with shape [B, C, T]."""
        if x.ndim != 3:
            raise ValueError(f"expected input [B, C, T], got {tuple(x.shape)}")
        if x.size(1) != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.size(1)}"
            )

        sequence = x.transpose(1, 2)  # [B, T, C]

        if self.use_calcium_innovation:
            innovation = self.calcium_innovation(sequence)
            recurrent_input = (
                sequence + self.innovation_scale * innovation
            )
        else:
            # Direct vanilla-GRU control: no calcium AR innovation is supplied.
            recurrent_input = sequence

        recurrent_input = self.input_norm(recurrent_input)
        recurrent_output, _ = self.gru(recurrent_input)
        recurrent_output = self.output_norm(recurrent_output)

        refined = sequence + self.residual_scale * recurrent_output
        return refined.transpose(1, 2)


class Adaptor_model(nn.Module):
    """Calcium-informed GRU refinement followed by the original TCN backbone."""

    def __init__(
        self,
        neu_F_dim: int = 4667,
        latent_T_dim: int = 52,
        latent_F_dim: int = 64,
        channels: int = 256,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        norm_groups: int = 8,
        sigma_rescale: float = 0.06,
        # Ablation controls.
        use_recurrent_refinement: bool = True,
        use_calcium_innovation: bool = True,
        # Calcium-GRU controls. The default remains unidirectional so existing
        # 2026-08-31 V2 checkpoints/configs remain structurally reproducible.
        gru_layers: int = 1,
        bidirectional: bool = False,
        decay_init: float = 0.95,
        gamma_min: float = 0.50,
        gamma_max: float = 0.999,
        innovation_scale_init: float = 0.10,
        recurrent_residual_scale_init: float = 0.10,
    ) -> None:
        super().__init__()

        self.neu_F_dim = int(neu_F_dim)
        channels = int(channels)
        dilations = tuple(int(dilation) for dilation in dilations)

        if not dilations:
            raise ValueError("dilations must not be empty")
        if sigma_rescale <= 0:
            raise ValueError(
                f"sigma_rescale must be positive, got {sigma_rescale}"
            )

        self.target_time = int(latent_T_dim)
        self.sigma_rescale = float(sigma_rescale)
        self.use_recurrent_refinement = bool(use_recurrent_refinement)
        self.use_calcium_innovation = bool(use_calcium_innovation)
        self.bidirectional = bool(bidirectional)

        # Exactly the same input stage as model_tcn.py.
        self.input_projection = nn.Sequential(
            nn.Conv1d(self.neu_F_dim, channels, kernel_size=1),
            nn.GroupNorm(int(norm_groups), channels),
            nn.GELU(),
        )

        if self.use_recurrent_refinement:
            self.recurrent_refinement = CalciumGRURefinement(
                channels=channels,
                dropout=float(dropout),
                gru_layers=int(gru_layers),
                bidirectional=self.bidirectional,
                use_calcium_innovation=self.use_calcium_innovation,
                decay_init=float(decay_init),
                gamma_min=float(gamma_min),
                gamma_max=float(gamma_max),
                innovation_scale_init=float(innovation_scale_init),
                residual_scale_init=float(recurrent_residual_scale_init),
            )
        else:
            self.recurrent_refinement = nn.Identity()

        # Exactly the same residual dilated-TCN backbone as model_tcn.py.
        self.temporal_network = nn.Sequential(
            *[
                TemporalBlock(
                    channels=channels,
                    dilation=dilation,
                    dropout=float(dropout),
                    norm_groups=int(norm_groups),
                )
                for dilation in dilations
            ]
        )

        # Exactly the same output stage as model_tcn.py.
        self.output_projection = nn.Conv1d(
            channels, int(latent_F_dim), kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B, T_neural, neu_F_dim] -> [B, latent_F_dim, latent_T_dim]."""
        if x.ndim != 3:
            raise ValueError(
                f"expected input [B, T, F], got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.neu_F_dim:
            raise ValueError(
                f"expected input feature dimension {self.neu_F_dim}, "
                f"got {x.size(-1)}"
            )
        if x.size(1) < 1:
            raise ValueError(
                "input time dimension must contain at least one frame"
            )

        x = self.input_projection(x.transpose(1, 2))
        x = self.recurrent_refinement(x)
        x = self.temporal_network(x)
        x = self.output_projection(x)
        x = F.interpolate(
            x,
            size=self.target_time,
            mode="linear",
            align_corners=False,
        )
        return x / self.sigma_rescale

    def learned_calcium_retention(self) -> torch.Tensor | None:
        """Convenience accessor for inspecting learned gamma after training."""
        if not self.use_recurrent_refinement:
            return None
        if not self.use_calcium_innovation:
            return None
        return self.recurrent_refinement.calcium_retention()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    x = torch.randn(2, 90, 4667)

    proposed = Adaptor_model()
    y = proposed(x)
    assert y.shape == (2, 64, 52), y.shape

    vanilla_gru_control = Adaptor_model(
        use_recurrent_refinement=True,
        use_calcium_innovation=False,
    )
    y_gru = vanilla_gru_control(x)
    assert y_gru.shape == (2, 64, 52), y_gru.shape

    pure_tcn_control = Adaptor_model(
        use_recurrent_refinement=False,
    )
    y_tcn = pure_tcn_control(x)
    assert y_tcn.shape == (2, 64, 52), y_tcn.shape

    gamma = proposed.learned_calcium_retention()
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Proposed parameters: {_parameter_count(proposed):,}")
    print(f"TCN-control parameters: {_parameter_count(pure_tcn_control):,}")
    print(
        "Initial calcium retention gamma: "
        f"mean={gamma.mean().item():.4f}, "
        f"min={gamma.min().item():.4f}, "
        f"max={gamma.max().item():.4f}"
    )
