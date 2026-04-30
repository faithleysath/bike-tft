"""Adaptive-only Graph WaveNet-style temporal convolution model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecasting_models.agcrn_nyc_gwnet_v1.model import DilatedTemporalBlock


@dataclass(frozen=True)
class AdaptiveGraphWaveNetConfig:
    """Configuration for the adaptive-only NYC Graph WaveNet variant."""

    num_nodes: int = 883
    input_dim: int = 38
    output_dim: int = 2
    horizon: int = 12
    embed_dim: int = 10
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 128
    end_channels: int = 256
    blocks: int = 2
    layers: int = 3
    kernel_size: int = 2
    graph_order: int = 2
    dropout: float = 0.1


class AdaptiveGraphWaveNet(nn.Module):
    """Graph WaveNet-style model using only learned adaptive station support."""

    def __init__(self, config: AdaptiveGraphWaveNetConfig) -> None:
        super().__init__()
        self.config = config
        self.node_embeddings = nn.Parameter(torch.randn(config.num_nodes, config.embed_dim))
        self.start_conv = nn.Conv2d(config.input_dim, config.residual_channels, kernel_size=(1, 1))
        blocks: list[DilatedTemporalBlock] = []
        for _block in range(config.blocks):
            for layer in range(config.layers):
                blocks.append(
                    DilatedTemporalBlock(
                        residual_channels=config.residual_channels,
                        dilation_channels=config.dilation_channels,
                        skip_channels=config.skip_channels,
                        kernel_size=config.kernel_size,
                        dilation=2**layer,
                        graph_order=config.graph_order,
                        dropout=config.dropout,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.end_conv_1 = nn.Conv2d(config.skip_channels, config.end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(config.end_channels, config.horizon * config.output_dim, kernel_size=(1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable parameters."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def relation_weight_dict(self) -> dict[str, float]:
        """Return a graph-weight payload compatible with earlier summaries."""
        return {"adaptive": 1.0}

    def adaptive_support(self, *, dtype: torch.dtype) -> torch.Tensor:
        """Build learned adaptive graph support."""
        support = F.softmax(F.relu(self.node_embeddings @ self.node_embeddings.T), dim=1)
        return support.to(dtype=dtype)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Predict future departures and arrivals.

        Args:
            source: Tensor shaped [batch, lag, nodes, input_dim].

        Returns:
            Tensor shaped [batch, horizon, nodes, output_dim].
        """
        if source.shape[2] != self.config.num_nodes or source.shape[3] != self.config.input_dim:
            raise ValueError(
                f"Expected input shape [B,T,{self.config.num_nodes},{self.config.input_dim}], got {tuple(source.shape)}"
            )
        x = source.permute(0, 3, 2, 1).contiguous()
        x = self.start_conv(x)
        support = self.adaptive_support(dtype=x.dtype)
        skip_total: torch.Tensor | None = None
        for block in self.blocks:
            x, skip = block(x, support)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("AdaptiveGraphWaveNet has no temporal blocks")
        output = F.relu(skip_total)
        output = F.relu(self.end_conv_1(output))
        output = self.end_conv_2(output)
        output = output[..., -1]
        output = output.reshape(-1, self.config.horizon, self.config.output_dim, self.config.num_nodes)
        return output.permute(0, 1, 3, 2).contiguous()

