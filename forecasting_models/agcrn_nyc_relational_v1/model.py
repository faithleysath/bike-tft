"""AGCRN with training-period OD relation supports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RELATION_NAMES = ("adaptive", "od_forward", "od_reverse")


@dataclass(frozen=True)
class RelationalAGCRNConfig:
    """Configuration for the relational NYC AGCRN variant."""

    num_nodes: int = 883
    input_dim: int = 38
    output_dim: int = 2
    horizon: int = 12
    embed_dim: int = 10
    rnn_units: int = 64
    num_layers: int = 2
    support_count: int = 2
    relation_mode: str = "fused"
    adaptive_init_weight: float = 0.70
    od_forward_init_weight: float = 0.15
    od_reverse_init_weight: float = 0.15


def normalized_init_logits(weights: tuple[float, float, float]) -> torch.Tensor:
    """Convert positive initial weights into softmax logits."""
    values = np.asarray(weights, dtype=np.float32)
    if np.any(values <= 0):
        raise ValueError("Initial relation weights must be positive")
    values = values / values.sum()
    return torch.log(torch.as_tensor(values, dtype=torch.float32))


def load_relation_graphs(path: str | Path, station_ids: list[str]) -> tuple[torch.Tensor, dict[str, object]]:
    """Load and validate OD relation graph supports."""
    arrays = np.load(path)
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


class RelationalAdaptiveGraphConv(nn.Module):
    """Node-adaptive graph convolution with optional OD relation supports."""

    def __init__(self, dim_in: int, dim_out: int, support_count: int, embed_dim: int, relation_mode: str) -> None:
        super().__init__()
        if relation_mode not in {"fused", "separate"}:
            raise ValueError("relation_mode must be 'fused' or 'separate'")
        if relation_mode == "fused" and support_count < 2:
            raise ValueError("fused mode requires at least identity and fused relation supports")
        if relation_mode == "separate" and support_count < 4:
            raise ValueError("separate mode requires identity, adaptive, OD forward, and OD reverse supports")
        self.support_count = support_count
        self.relation_mode = relation_mode
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, support_count, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))

    def build_supports(
        self,
        *,
        node_embeddings: torch.Tensor,
        relation_supports: torch.Tensor,
        relation_weights: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the support tensor used by graph convolution."""
        node_count = node_embeddings.shape[0]
        adaptive_support = F.softmax(F.relu(node_embeddings @ node_embeddings.T), dim=1)
        od_forward = relation_supports[0].to(device=device, dtype=dtype)
        od_reverse = relation_supports[1].to(device=device, dtype=dtype)
        identity = torch.eye(node_count, device=device, dtype=dtype)

        if self.relation_mode == "fused":
            relation_stack = torch.stack((adaptive_support, od_forward, od_reverse), dim=0)
            graph_support = torch.einsum("r,rnm->nm", relation_weights.to(dtype=dtype), relation_stack)
            support_set = [identity, graph_support]
        else:
            weighted_adaptive = relation_weights[0].to(dtype=dtype) * adaptive_support
            weighted_forward = relation_weights[1].to(dtype=dtype) * od_forward
            weighted_reverse = relation_weights[2].to(dtype=dtype) * od_reverse
            graph_support = weighted_adaptive + weighted_forward + weighted_reverse
            support_set = [identity, weighted_adaptive, weighted_forward, weighted_reverse]

        while len(support_set) < self.support_count:
            support_set.append((2 * graph_support) @ support_set[-1] - support_set[-2])
        return torch.stack(support_set[: self.support_count], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
        relation_supports: torch.Tensor,
        relation_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run relation-aware adaptive graph convolution."""
        supports = self.build_supports(
            node_embeddings=node_embeddings,
            relation_supports=relation_supports,
            relation_weights=relation_weights,
            device=x.device,
            dtype=x.dtype,
        )
        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class RelationalAGCRNCell(nn.Module):
    """GRU-style recurrent cell using relation-aware graph convolutions."""

    def __init__(self, node_num: int, dim_in: int, dim_out: int, support_count: int, embed_dim: int, relation_mode: str) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = RelationalAdaptiveGraphConv(dim_in + dim_out, 2 * dim_out, support_count, embed_dim, relation_mode)
        self.update = RelationalAdaptiveGraphConv(dim_in + dim_out, dim_out, support_count, embed_dim, relation_mode)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        node_embeddings: torch.Tensor,
        relation_supports: torch.Tensor,
        relation_weights: torch.Tensor,
    ) -> torch.Tensor:
        state = state.to(device=x.device, dtype=x.dtype)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, relation_supports, relation_weights))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, relation_supports, relation_weights))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create an all-zero hidden state."""
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device, dtype=dtype)


class RelationalAVWDCRNN(nn.Module):
    """Stacked adaptive graph recurrent encoder with OD relation supports."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        support_count: int,
        embed_dim: int,
        num_layers: int,
        relation_mode: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.node_num = node_num
        self.input_dim = dim_in
        self.cells = nn.ModuleList(
            [
                RelationalAGCRNCell(
                    node_num,
                    dim_in if layer == 0 else dim_out,
                    dim_out,
                    support_count,
                    embed_dim,
                    relation_mode,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        node_embeddings: torch.Tensor,
        relation_supports: torch.Tensor,
        relation_weights: torch.Tensor,
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
                state = cell(
                    current_inputs[:, step, :, :],
                    state,
                    node_embeddings,
                    relation_supports,
                    relation_weights,
                )
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs


class RelationalAGCRN(nn.Module):
    """AGCRN model that adds training-period OD relations to adaptive supports."""

    def __init__(self, config: RelationalAGCRNConfig, relation_supports: torch.Tensor) -> None:
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
        self.encoder = RelationalAVWDCRNN(
            node_num=config.num_nodes,
            dim_in=config.input_dim,
            dim_out=config.rnn_units,
            support_count=config.support_count,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            relation_mode=config.relation_mode,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.horizon * config.output_dim,
            kernel_size=(1, config.rnn_units),
            bias=True,
        )
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

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Predict future departures and arrivals."""
        output = self.encoder(
            source,
            self.node_embeddings,
            self.relation_supports,
            self.relation_weight_tensor(),
        )
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(
            -1,
            self.config.horizon,
            self.config.output_dim,
            self.config.num_nodes,
        )
        return output.permute(0, 1, 3, 2).contiguous()

