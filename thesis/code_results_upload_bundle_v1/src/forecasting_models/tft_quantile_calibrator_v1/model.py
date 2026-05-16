"""TFT-style quantile model for station-level multi-horizon forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TFTQuantileConfig:
    """Configuration for the lightweight TFT-style quantile model."""

    num_nodes: int
    input_dim: int
    future_time_dim: int
    output_dim: int = 2
    quantile_count: int = 3
    horizon: int = 12
    hidden_dim: int = 32
    station_embed_dim: int = 16
    attention_heads: int = 4
    dropout: float = 0.1


class GatedFeatureProjection(nn.Module):
    """Feature projection with a simple gate, similar in spirit to TFT gating blocks."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to hidden space."""
        projected = torch.tanh(self.value(x)) * torch.sigmoid(self.gate(x))
        return self.norm(self.dropout(projected))


class TFTQuantileModel(nn.Module):
    """A compact temporal fusion quantile model over dense station panels."""

    def __init__(self, config: TFTQuantileConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.config = config
        self.input_projection = GatedFeatureProjection(config.input_dim, config.hidden_dim, config.dropout)
        self.encoder = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.future_projection = nn.Sequential(
            nn.Linear(config.future_time_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.station_embedding = nn.Embedding(config.num_nodes, config.station_embed_dim)
        self.static_projection = nn.Linear(config.station_embed_dim, config.hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.post_attention_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.quantile_head = nn.Linear(config.hidden_dim, config.output_dim * config.quantile_count)

    def forward(
        self,
        x: torch.Tensor,
        future_time: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return sorted quantiles and optionally attention weights.

        Quantiles are shaped [B, H, N, output_dim, quantile_count]. When
        requested, attention weights are shaped [B, N, heads, H, lag].
        """
        batch_size, lag, node_count, input_dim = x.shape
        if node_count != self.config.num_nodes:
            raise ValueError(f"Expected {self.config.num_nodes} nodes, got {node_count}")
        if input_dim != self.config.input_dim:
            raise ValueError(f"Expected input_dim={self.config.input_dim}, got {input_dim}")
        if future_time.shape[1] != self.config.horizon:
            raise ValueError(f"Expected horizon={self.config.horizon}, got {future_time.shape[1]}")

        station_ids = torch.arange(node_count, device=x.device)
        static_context = self.static_projection(self.station_embedding(station_ids))

        series = x.permute(0, 2, 1, 3).reshape(batch_size * node_count, lag, input_dim)
        encoded_input = self.input_projection(series)
        encoded_history, _state = self.encoder(encoded_input)

        future = self.future_projection(future_time)
        future = future[:, None, :, :].expand(batch_size, node_count, self.config.horizon, self.config.hidden_dim)
        static = static_context[None, :, None, :].expand(batch_size, node_count, self.config.horizon, self.config.hidden_dim)
        queries = (future + static).reshape(batch_size * node_count, self.config.horizon, self.config.hidden_dim)

        attended, weights = self.attention(
            queries,
            encoded_history,
            encoded_history,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        fused = self.post_attention_gate(torch.cat([queries, attended], dim=-1))
        raw = self.quantile_head(fused)
        raw = raw.reshape(
            batch_size,
            node_count,
            self.config.horizon,
            self.config.output_dim,
            self.config.quantile_count,
        )
        raw = raw.permute(0, 2, 1, 3, 4).contiguous()

        q10 = raw[..., 0:1]
        q50 = q10 + F.softplus(raw[..., 1:2])
        q90 = q50 + F.softplus(raw[..., 2:3])
        quantiles = torch.cat([q10, q50, q90], dim=-1)
        if not return_attention:
            return quantiles
        if weights is None:
            raise RuntimeError("Attention weights were not returned by MultiheadAttention")
        weights = weights.reshape(
            batch_size,
            node_count,
            self.config.attention_heads,
            self.config.horizon,
            lag,
        )
        return quantiles, weights
