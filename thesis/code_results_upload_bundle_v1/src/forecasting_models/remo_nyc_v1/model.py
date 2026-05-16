"""Paper-faithful ReMo-style hypergraph relational model for NYC forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReMoConfig:
    """Configuration for the ReMo NYC variant."""

    num_nodes: int = 883
    input_dim: int = 38
    output_dim: int = 2
    horizon: int = 12
    hidden_dim: int = 64
    node_embed_dim: int = 16
    num_views: int = 2
    num_hyperedges: int = 16
    num_relation_types: int = 4
    dropout: float = 0.1


class MultiRangeTemporalConv(nn.Module):
    """Multi-kernel temporal convolution module used before relational modeling."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        branch_channels = hidden_dim // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(input_dim, branch_channels, kernel_size=(1, kernel), padding=(0, kernel - 1))
                for kernel in (2, 3, 6, 12)
            ]
        )
        self.proj = nn.Conv2d(branch_channels * 4, hidden_dim, kernel_size=(1, 1))
        self.dropout = dropout

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Encode source shaped [B, T, N, F] into [B, N, C]."""
        x = source.permute(0, 3, 2, 1).contiguous()
        outputs = []
        for conv in self.branches:
            branch = conv(x)
            outputs.append(branch[..., -source.shape[1] :])
        hidden = F.gelu(self.proj(torch.cat(outputs, dim=1)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden[..., -1].transpose(1, 2).contiguous()


class RelationalModelingBlock(nn.Module):
    """Multi-view hypergraph constructor and relation-aware message passing."""

    def __init__(self, config: ReMoConfig) -> None:
        super().__init__()
        self.config = config
        self.node_embeddings = nn.Parameter(torch.randn(config.num_nodes, config.node_embed_dim))
        self.view_queries = nn.Parameter(torch.randn(config.num_views, config.num_hyperedges, config.node_embed_dim))
        self.temporal_to_membership = nn.Linear(config.hidden_dim, config.num_views * config.num_hyperedges)
        self.node_to_edge = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.edge_type_score = nn.Linear(config.hidden_dim, config.num_relation_types)
        self.relation_transforms = nn.ModuleList(
            [nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_relation_types)]
        )
        self.edge_to_node = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def construct_membership(self, node_hidden: torch.Tensor) -> torch.Tensor:
        """Construct multi-view soft hypergraph memberships [B,V,N,K]."""
        batch = node_hidden.shape[0]
        static_logits = torch.einsum("ne,vke->vnk", self.node_embeddings, self.view_queries)
        dynamic_logits = self.temporal_to_membership(node_hidden).reshape(
            batch,
            self.config.num_nodes,
            self.config.num_views,
            self.config.num_hyperedges,
        )
        logits = static_logits.unsqueeze(0).permute(0, 1, 2, 3) + dynamic_logits.permute(0, 2, 1, 3)
        return torch.softmax(logits, dim=-1)

    def forward(self, node_hidden: torch.Tensor) -> torch.Tensor:
        """Run relational hypergraph message passing over node hidden states."""
        membership = self.construct_membership(node_hidden)
        edge_denominator = membership.sum(dim=2).clamp_min(1e-6)
        edge_hidden = torch.einsum("bvnk,bnc->bvkc", membership, self.node_to_edge(node_hidden)) / edge_denominator.unsqueeze(-1)
        relation_weight = torch.softmax(self.edge_type_score(edge_hidden), dim=-1)
        relation_messages = []
        for index, transform in enumerate(self.relation_transforms):
            relation_messages.append(transform(edge_hidden) * relation_weight[..., index : index + 1])
        edge_hidden = torch.stack(relation_messages, dim=0).sum(dim=0)
        node_denominator = membership.sum(dim=-1).clamp_min(1e-6)
        node_message = torch.einsum("bvnk,bvkc->bvnc", membership, edge_hidden) / node_denominator.unsqueeze(-1)
        node_message = node_message.mean(dim=1)
        return self.norm(node_hidden + F.gelu(self.edge_to_node(node_message)))


class ReMoNYC(nn.Module):
    """ReMo-style model for dep/arr station forecasting."""

    def __init__(self, config: ReMoConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal = MultiRangeTemporalConv(config.input_dim, config.hidden_dim, config.dropout)
        self.relational = RelationalModelingBlock(config)
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.horizon * config.output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable parameters."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Predict future dep/arr values from source shaped [B,T,N,F]."""
        if source.shape[2] != self.config.num_nodes or source.shape[3] != self.config.input_dim:
            raise ValueError(
                f"Expected source shape [B,T,{self.config.num_nodes},{self.config.input_dim}], got {tuple(source.shape)}"
            )
        hidden = self.temporal(source)
        hidden = self.relational(hidden)
        output = self.output(hidden).reshape(source.shape[0], self.config.num_nodes, self.config.horizon, self.config.output_dim)
        return output.permute(0, 2, 1, 3).contiguous()

