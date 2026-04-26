"""Project-local AGCRN modules adapted from the original paper implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AGCRNConfig:
    """Configuration for the NYC AGCRN model."""

    num_nodes: int = 200
    input_dim: int = 38
    output_dim: int = 2
    horizon: int = 12
    embed_dim: int = 10
    rnn_units: int = 64
    num_layers: int = 2
    cheb_k: int = 2


class AdaptiveGraphConv(nn.Module):
    """Node-adaptive graph convolution from AGCRN."""

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, dim_out))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Run adaptive graph convolution.

        Args:
            x: Tensor shaped ``[batch, nodes, channels]``.
            node_embeddings: Tensor shaped ``[nodes, embed_dim]``.
        """
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(node_embeddings @ node_embeddings.T), dim=1)
        support_set = [torch.eye(node_num, device=x.device, dtype=x.dtype), supports]
        for _ in range(2, self.cheb_k):
            support_set.append((2 * supports) @ support_set[-1] - support_set[-2])
        supports = torch.stack(support_set[: self.cheb_k], dim=0)

        weights = torch.einsum("nd,dkio->nkio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        return torch.einsum("bnki,nkio->bno", x_g, weights) + bias


class AGCRNCell(nn.Module):
    """GRU-style recurrent cell with adaptive graph convolutions."""

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AdaptiveGraphConv(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AdaptiveGraphConv(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        state = state.to(device=x.device, dtype=x.dtype)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device, dtype=dtype)


class AVWDCRNN(nn.Module):
    """Stacked adaptive graph recurrent encoder."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        embed_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in if layer == 0 else dim_out, dim_out, cheb_k, embed_dim) for layer in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode a sequence shaped ``[batch, time, nodes, features]``."""
        if x.shape[2] != self.node_num or x.shape[3] != self.input_dim:
            raise ValueError(
                f"Expected input shape [B,T,{self.node_num},{self.input_dim}], got {tuple(x.shape)}"
            )

        current_inputs = x
        batch_size = x.shape[0]
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size, device=x.device, dtype=x.dtype)
            inner_states = []
            for step in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, step, :, :], state, node_embeddings)
                inner_states.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs


class AGCRN(nn.Module):
    """AGCRN model for multi-step, per-station dep/arr forecasting."""

    def __init__(self, config: AGCRNConfig) -> None:
        super().__init__()
        self.config = config
        self.node_embeddings = nn.Parameter(torch.randn(config.num_nodes, config.embed_dim))
        self.encoder = AVWDCRNN(
            node_num=config.num_nodes,
            dim_in=config.input_dim,
            dim_out=config.rnn_units,
            cheb_k=config.cheb_k,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
        )
        self.end_conv = nn.Conv2d(
            in_channels=1,
            out_channels=config.horizon * config.output_dim,
            kernel_size=(1, config.rnn_units),
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters similarly to the original implementation."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Predict future departures and arrivals.

        Args:
            source: Tensor shaped ``[batch, lag, nodes, input_dim]``.

        Returns:
            Tensor shaped ``[batch, horizon, nodes, 2]`` where channel 0 is dep
            and channel 1 is arr.
        """
        output = self.encoder(source, self.node_embeddings)
        output = output[:, -1:, :, :]
        output = self.end_conv(output)
        output = output.squeeze(-1).reshape(
            -1,
            self.config.horizon,
            self.config.output_dim,
            self.config.num_nodes,
        )
        return output.permute(0, 1, 3, 2).contiguous()
