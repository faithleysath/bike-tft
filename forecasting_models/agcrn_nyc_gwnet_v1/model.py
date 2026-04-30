"""Graph WaveNet-style temporal convolution model with weak OD relation support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forecasting_models.agcrn_nyc_relational_v1.model import RELATION_NAMES, normalized_init_logits


@dataclass(frozen=True)
class GraphWaveNetConfig:
    """Configuration for the NYC Graph WaveNet variant."""

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
    adaptive_init_weight: float = 0.95
    od_forward_init_weight: float = 0.025
    od_reverse_init_weight: float = 0.025


def load_relation_graphs(path: str | Path, station_ids: list[str]) -> tuple[torch.Tensor, dict[str, object]]:
    """Load and validate OD relation graph supports."""
    arrays = np.load(path, allow_pickle=False)
    required = {"od_forward_support", "od_reverse_support", "station_ids", "metadata_json"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"Relation graph artifact is missing array(s): {sorted(missing)}")

    graph_station_ids = [str(item) for item in arrays["station_ids"].tolist()]
    if graph_station_ids != [str(item) for item in station_ids]:
        raise ValueError("Relation graph station_ids do not match the training bundle station_ids")

    od_forward = arrays["od_forward_support"].astype(np.float32)
    od_reverse = arrays["od_reverse_support"].astype(np.float32)
    expected_shape = (len(station_ids), len(station_ids))
    if od_forward.shape != expected_shape or od_reverse.shape != expected_shape:
        raise ValueError(
            f"Expected relation supports shaped {expected_shape}, got {od_forward.shape} and {od_reverse.shape}"
        )
    metadata = json.loads(str(arrays["metadata_json"].item()))
    supports = np.stack([od_forward, od_reverse], axis=0)
    return torch.from_numpy(supports), metadata


def graph_nconv(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Apply node convolution over a support matrix.

    Args:
        x: Tensor shaped [batch, channels, nodes, time].
        support: Tensor shaped [nodes, nodes], row-normalized as output node by input node.
    """
    return torch.einsum("bcmt,nm->bcnt", x, support).contiguous()


class GraphConv(nn.Module):
    """Graph convolution over powers of a fused support matrix."""

    def __init__(self, channels: int, output_channels: int, graph_order: int, dropout: float) -> None:
        super().__init__()
        if graph_order < 1:
            raise ValueError("graph_order must be at least 1")
        self.graph_order = graph_order
        self.dropout = dropout
        self.mlp = nn.Conv2d((graph_order + 1) * channels, output_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Run graph convolution over [I, A, A^2, ...]."""
        outputs = [x]
        x_k = x
        for _order in range(1, self.graph_order + 1):
            x_k = graph_nconv(x_k, support)
            outputs.append(x_k)
        output = torch.cat(outputs, dim=1)
        output = self.mlp(output)
        return F.dropout(output, p=self.dropout, training=self.training)


class DilatedTemporalBlock(nn.Module):
    """One gated dilated temporal convolution block with graph convolution."""

    def __init__(
        self,
        *,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        graph_order: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.filter_conv = nn.Conv2d(
            residual_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        self.gate_conv = nn.Conv2d(
            residual_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        self.graph_conv = GraphConv(dilation_channels, residual_channels, graph_order, dropout)
        self.skip_conv = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(residual_channels)

    def temporal_conv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        """Run left-padded causal temporal convolution."""
        padded = F.pad(x, (self.left_padding, 0, 0, 0))
        return conv(padded)

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return updated residual stream and skip contribution."""
        residual = x
        gated = torch.tanh(self.temporal_conv(self.filter_conv, x)) * torch.sigmoid(self.temporal_conv(self.gate_conv, x))
        skip = self.skip_conv(gated)
        output = self.graph_conv(gated, support)
        output = self.norm(output + residual)
        return output, skip


class GraphWaveNet(nn.Module):
    """Graph WaveNet-style model for multi-step station dep/arr forecasting."""

    def __init__(self, config: GraphWaveNetConfig, relation_supports: torch.Tensor) -> None:
        super().__init__()
        if relation_supports.shape != (2, config.num_nodes, config.num_nodes):
            raise ValueError(
                f"Expected relation supports shape (2, {config.num_nodes}, {config.num_nodes}), got {tuple(relation_supports.shape)}"
            )
        self.config = config
        self.node_embeddings = nn.Parameter(torch.randn(config.num_nodes, config.embed_dim))
        self.relation_logits = nn.Parameter(
            normalized_init_logits(
                (
                    config.adaptive_init_weight,
                    config.od_forward_init_weight,
                    config.od_reverse_init_weight,
                )
            )
        )
        self.register_buffer("relation_supports", relation_supports.to(dtype=torch.float32))
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
        """Initialize trainable parameters while preserving relation priors."""
        for name, parameter in self.named_parameters():
            if name == "relation_logits":
                continue
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def relation_weight_tensor(self) -> torch.Tensor:
        """Return softmax-normalized relation weights."""
        return torch.softmax(self.relation_logits, dim=0)

    def relation_weight_dict(self) -> dict[str, float]:
        """Return relation weights as a JSON-ready dictionary."""
        values = self.relation_weight_tensor().detach().cpu().numpy().astype(float)
        return {name: float(value) for name, value in zip(RELATION_NAMES, values, strict=True)}

    def fused_support(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build weakly fused adaptive and OD support."""
        adaptive = F.softmax(F.relu(self.node_embeddings @ self.node_embeddings.T), dim=1)
        od_forward = self.relation_supports[0].to(device=device, dtype=dtype)
        od_reverse = self.relation_supports[1].to(device=device, dtype=dtype)
        relation_stack = torch.stack((adaptive.to(dtype=dtype), od_forward, od_reverse), dim=0)
        return torch.einsum("r,rnm->nm", self.relation_weight_tensor().to(dtype=dtype), relation_stack)

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
        support = self.fused_support(device=x.device, dtype=x.dtype)
        skip_total: torch.Tensor | None = None
        for block in self.blocks:
            x, skip = block(x, support)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("GraphWaveNet has no temporal blocks")
        output = F.relu(skip_total)
        output = F.relu(self.end_conv_1(output))
        output = self.end_conv_2(output)
        output = output[..., -1]
        output = output.reshape(-1, self.config.horizon, self.config.output_dim, self.config.num_nodes)
        return output.permute(0, 1, 3, 2).contiguous()

