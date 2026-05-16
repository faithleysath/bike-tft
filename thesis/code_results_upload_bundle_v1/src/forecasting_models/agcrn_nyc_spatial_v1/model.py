"""AGCRN with a fixed geographic distance graph fused into graph convolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialAGCRNConfig:
    """Configuration for the spatial-prior AGCRN variant."""

    num_nodes: int = 883
    input_dim: int = 38
    output_dim: int = 2
    horizon: int = 12
    embed_dim: int = 10
    rnn_units: int = 64
    num_layers: int = 2
    support_count: int = 2
    spatial_top_k: int = 20
    spatial_sigma_km: float | None = None
    spatial_init_mix: float = 0.5
    spatial_mode: str = "fused"


def haversine_distance_km(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Build pairwise station distances in kilometers."""
    radius_km = 6371.0088
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)
    delta_lat = lat_rad[:, None] - lat_rad[None, :]
    delta_lng = lng_rad[:, None] - lng_rad[None, :]
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(delta_lng / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return (radius_km * c).astype(np.float32)


def build_distance_support(
    station_static_path: str | Path,
    station_ids: list[str],
    *,
    top_k: int = 20,
    sigma_km: float | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Create a row-normalized kNN Gaussian support from station coordinates."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    static = pd.read_csv(station_static_path, dtype={"station_id": str})
    required = {"station_id", "station_lat", "station_lng"}
    missing = required.difference(static.columns)
    if missing:
        raise ValueError(f"Station static table is missing columns: {sorted(missing)}")

    order = pd.DataFrame({"station_id": [str(item) for item in station_ids]})
    ordered = order.merge(static.loc[:, ["station_id", "station_lat", "station_lng"]], on="station_id", how="left")
    if ordered[["station_lat", "station_lng"]].isna().any().any():
        missing_ids = ordered.loc[ordered["station_lat"].isna(), "station_id"].head(10).tolist()
        raise ValueError(f"Station static table is missing coordinate rows for station ids: {missing_ids}")

    lat = ordered["station_lat"].to_numpy(dtype=np.float64, copy=False)
    lng = ordered["station_lng"].to_numpy(dtype=np.float64, copy=False)
    distances = haversine_distance_km(lat, lng)
    node_count = len(station_ids)
    neighbor_count = min(top_k, node_count - 1)
    if neighbor_count < 1:
        raise ValueError("At least two stations are required to build a spatial graph")

    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    neighbor_indices = np.argpartition(masked, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    neighbor_distances = np.take_along_axis(masked, neighbor_indices, axis=1)
    if sigma_km is None:
        positive = neighbor_distances[np.isfinite(neighbor_distances) & (neighbor_distances > 0)]
        sigma = float(np.median(positive)) if len(positive) else 1.0
    else:
        sigma = float(sigma_km)
    if sigma <= 0:
        raise ValueError("sigma_km must be positive")

    weights = np.exp(-((neighbor_distances / sigma) ** 2)).astype(np.float32)
    support = np.zeros((node_count, node_count), dtype=np.float32)
    rows = np.arange(node_count)[:, None]
    support[rows, neighbor_indices] = weights
    row_sums = support.sum(axis=1, keepdims=True)
    support = np.divide(support, row_sums, out=np.zeros_like(support), where=row_sums > 0)
    metadata = {
        "node_count": int(node_count),
        "spatial_top_k": int(neighbor_count),
        "spatial_sigma_km": float(sigma),
        "mean_neighbor_distance_km": float(neighbor_distances.mean()),
        "median_neighbor_distance_km": float(np.median(neighbor_distances)),
    }
    return torch.from_numpy(support), metadata


def logit(value: float) -> float:
    """Return the logit transform for a probability in (0, 1)."""
    if not 0.0 < value < 1.0:
        raise ValueError("value must be in (0, 1)")
    return float(np.log(value / (1.0 - value)))


class SpatialAdaptiveGraphConv(nn.Module):
    """Adaptive AGCRN graph convolution with distance-prior support fusion."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        support_count: int,
        embed_dim: int,
        spatial_init_mix: float,
        spatial_mode: str,
    ) -> None:
        super().__init__()
        if spatial_mode not in {"fused", "separate"}:
            raise ValueError("spatial_mode must be 'fused' or 'separate'")
        if spatial_mode == "fused" and support_count < 2:
            raise ValueError("fused mode requires identity and fused supports")
        if spatial_mode == "separate" and support_count < 3:
            raise ValueError("separate mode requires identity, adaptive, and spatial supports")
        self.support_count = support_count
        self.spatial_mode = spatial_mode
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, support_count, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))
        self.spatial_mix_logit = nn.Parameter(torch.tensor(logit(spatial_init_mix), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
        spatial_support: torch.Tensor,
    ) -> torch.Tensor:
        """Run graph convolution over identity and adaptive-spatial fused supports."""
        node_count = node_embeddings.shape[0]
        adaptive_support = F.softmax(F.relu(node_embeddings @ node_embeddings.T), dim=1)
        spatial_support = spatial_support.to(device=x.device, dtype=x.dtype)
        if self.spatial_mode == "fused":
            spatial_mix = torch.sigmoid(self.spatial_mix_logit)
            graph_support = (1.0 - spatial_mix) * adaptive_support + spatial_mix * spatial_support
            support_set = [
                torch.eye(node_count, device=x.device, dtype=x.dtype),
                graph_support,
            ]
        else:
            graph_support = adaptive_support
            support_set = [
                torch.eye(node_count, device=x.device, dtype=x.dtype),
                adaptive_support,
                spatial_support,
            ]
        while len(support_set) < self.support_count:
            support_set.append((2 * graph_support) @ support_set[-1] - support_set[-2])
        supports = torch.stack(support_set[: self.support_count], dim=0)

        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class SpatialAGCRNCell(nn.Module):
    """GRU-style recurrent cell using spatial-prior graph convolutions."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        support_count: int,
        embed_dim: int,
        spatial_init_mix: float,
        spatial_mode: str,
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = SpatialAdaptiveGraphConv(
            dim_in + dim_out,
            2 * dim_out,
            support_count,
            embed_dim,
            spatial_init_mix,
            spatial_mode,
        )
        self.update = SpatialAdaptiveGraphConv(
            dim_in + dim_out,
            dim_out,
            support_count,
            embed_dim,
            spatial_init_mix,
            spatial_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        spatial_support: torch.Tensor,
    ) -> torch.Tensor:
        state = state.to(device=x.device, dtype=x.dtype)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, spatial_support))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, spatial_support))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device, dtype=dtype)


class SpatialAVWDCRNN(nn.Module):
    """Stacked adaptive graph recurrent encoder with spatial support."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        support_count: int,
        embed_dim: int,
        num_layers: int,
        spatial_init_mix: float,
        spatial_mode: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.node_num = node_num
        self.input_dim = dim_in
        self.cells = nn.ModuleList(
            [
                SpatialAGCRNCell(
                    node_num,
                    dim_in if layer == 0 else dim_out,
                    dim_out,
                    support_count,
                    embed_dim,
                    spatial_init_mix,
                    spatial_mode,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
        spatial_support: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a sequence shaped [batch, time, nodes, features]."""
        if x.shape[2] != self.node_num or x.shape[3] != self.input_dim:
            raise ValueError(f"Expected input shape [B,T,{self.node_num},{self.input_dim}], got {tuple(x.shape)}")

        current_inputs = x
        batch_size = x.shape[0]
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size, device=x.device, dtype=x.dtype)
            inner_states = []
            for step in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, step, :, :], state, node_embeddings, spatial_support)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs


class SpatialAGCRN(nn.Module):
    """AGCRN model that fuses adaptive station relations with a distance graph."""

    def __init__(self, config: SpatialAGCRNConfig, spatial_support: torch.Tensor) -> None:
        super().__init__()
        if spatial_support.shape != (config.num_nodes, config.num_nodes):
            raise ValueError(
                f"Expected spatial support shape ({config.num_nodes}, {config.num_nodes}), got {tuple(spatial_support.shape)}"
            )
        self.config = config
        self.node_embeddings = nn.Parameter(torch.randn(config.num_nodes, config.embed_dim))
        self.register_buffer("spatial_support", spatial_support.to(dtype=torch.float32))
        self.encoder = SpatialAVWDCRNN(
            node_num=config.num_nodes,
            dim_in=config.input_dim,
            dim_out=config.rnn_units,
            support_count=config.support_count,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            spatial_init_mix=config.spatial_init_mix,
            spatial_mode=config.spatial_mode,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.horizon * config.output_dim,
            kernel_size=(1, config.rnn_units),
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable parameters."""
        for name, parameter in self.named_parameters():
            if name.endswith("spatial_mix_logit"):
                continue
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Predict future departures and arrivals."""
        output = self.encoder(source, self.node_embeddings, self.spatial_support)
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(
            -1,
            self.config.horizon,
            self.config.output_dim,
            self.config.num_nodes,
        )
        return output.permute(0, 1, 3, 2).contiguous()
