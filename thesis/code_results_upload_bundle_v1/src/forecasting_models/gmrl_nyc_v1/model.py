"""Project-local GMRL modules adapted from the IJCAI 2023 implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG2PI = math.log(2 * math.pi)


def make_dilations(lag: int) -> tuple[int, ...]:
    """Build dilations that reduce ``lag`` timesteps to one output state."""
    if lag < 2:
        raise ValueError("lag must be at least 2")
    remaining = lag - 1
    dilation = 1
    values: list[int] = []
    while remaining > 0:
        value = min(dilation, remaining)
        values.append(value)
        remaining -= value
        dilation *= 2
    return tuple(values)


@dataclass(frozen=True)
class GMRLConfig:
    """Configuration for the NYC GMRL variant."""

    num_nodes: int = 883
    num_sources: int = 2
    input_dim: int = 1
    output_dim: int = 1
    lag: int = 12
    horizon: int = 12
    num_components: int = 17
    hidden_channels: int = 24
    kernel_size: int = 2
    use_hra: bool = True
    gmre_chunk_size: int = 2048

    @property
    def dilations(self) -> tuple[int, ...]:
        return make_dilations(self.lag)


class GMRE(nn.Module):
    """Gaussian Mixture Representation Extractor with chunked posterior evaluation."""

    def __init__(
        self,
        *,
        num_components: int,
        channels: int,
        num_nodes: int,
        num_sources: int,
        current_time: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if num_components <= 0:
            raise ValueError("num_components must be positive")
        if current_time <= 0:
            raise ValueError("current_time must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.num_components = num_components
        self.chunk_size = chunk_size
        self.in_features = num_nodes * num_sources * current_time
        self.alpha = nn.Sequential(
            nn.Conv1d(in_channels=self.in_features, out_channels=num_components, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.sigma = nn.Conv1d(in_channels=self.in_features, out_channels=num_components, kernel_size=1)
        self.mu = nn.Conv1d(in_channels=self.in_features, out_channels=num_components, kernel_size=1)

    def _chunked_cluster_norm(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize target by the most likely Gaussian component without materializing B*C*N*K."""
        batch, channels, point_count = target.shape
        norm_chunks: list[torch.Tensor] = []
        weighted_log_sum = torch.zeros_like(alpha)
        first_item_sum = target.new_tensor(0.0)
        item_count = 0
        log_alpha = torch.log(alpha.clamp_min(1e-12))

        for start in range(0, point_count, self.chunk_size):
            end = min(start + self.chunk_size, point_count)
            target_chunk = target[:, :, start:end].unsqueeze(-1)
            log_component_prob = (
                -torch.log(sigma).unsqueeze(2)
                - 0.5 * LOG2PI
                - 0.5 * torch.pow((target_chunk - mu.unsqueeze(2)) / sigma.unsqueeze(2), 2)
            )
            log_prob = log_component_prob + log_alpha.unsqueeze(2)
            weighted_log = log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True)
            labels = torch.argmax(weighted_log, dim=-1)
            selected_mu = torch.gather(mu, 2, labels)
            selected_sigma = torch.gather(sigma, 2, labels)
            norm_chunks.append((target[:, :, start:end] - selected_mu) / (selected_sigma + 1e-5))

            weighted_log_sum = weighted_log_sum + weighted_log.sum(dim=2)
            first_item_sum = first_item_sum + (weighted_log.exp() * log_component_prob).sum()
            item_count += weighted_log.numel()

        norm = torch.cat(norm_chunks, dim=2)
        q_z = weighted_log_sum / point_count
        kl_loss = F.kl_div(q_z, alpha, reduction="mean")
        first_item = first_item_sum / item_count
        feature_loss = kl_loss - first_item
        return norm, feature_loss

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cluster-normalized representation and regularization loss."""
        batch, channels, sources, nodes, time_steps = source.shape
        hidden = source.reshape(batch, channels, -1).permute(0, 2, 1)
        if hidden.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} flattened points, got {hidden.shape[1]}")
        alpha = self.alpha(hidden).permute(0, 2, 1)
        sigma = torch.exp(torch.clamp(self.sigma(hidden), min=-6.0, max=6.0)).permute(0, 2, 1).clamp_min(1e-5)
        mu = self.mu(hidden).permute(0, 2, 1)
        target = hidden.permute(0, 2, 1)
        norm, feature_loss = self._chunked_cluster_norm(target, mu, sigma, alpha)
        return norm.reshape(batch, channels, sources, nodes, time_steps), feature_loss


class ResidualBlock(nn.Module):
    """One GMRL residual block: GMRE followed by gated temporal convolution."""

    def __init__(
        self,
        *,
        num_components: int,
        num_nodes: int,
        num_sources: int,
        channels: int,
        current_time: int,
        dilation: int,
        kernel_size: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if current_time - dilation * (kernel_size - 1) <= 0:
            raise ValueError(
                f"dilation={dilation} and kernel_size={kernel_size} do not fit current_time={current_time}"
            )
        self.num_sources = num_sources
        self.gmre = GMRE(
            num_components=num_components,
            channels=channels,
            num_nodes=num_nodes,
            num_sources=num_sources,
            current_time=current_time,
            chunk_size=chunk_size,
        )
        self.filter_conv = nn.Conv3d(
            in_channels=2 * channels,
            out_channels=num_sources * channels,
            kernel_size=(num_sources, 1, kernel_size),
            dilation=(1, 1, dilation),
        )
        self.gate_conv = nn.Conv3d(
            in_channels=2 * channels,
            out_channels=num_sources * channels,
            kernel_size=(num_sources, 1, kernel_size),
            dilation=(1, 1, dilation),
        )
        self.residual_conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 1, 1))
        self.skip_conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 1, 1))

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return updated residual stream, skip stream, and GMRE regularization loss."""
        gmre_source, feature_loss = self.gmre(source)
        hidden = torch.cat((gmre_source, source), dim=1)
        filtered = torch.tanh(self.filter_conv(hidden))
        gated = torch.sigmoid(self.gate_conv(hidden))
        batch, _channels, _one_source_axis, nodes, time_steps = filtered.shape
        output = (filtered * gated).reshape(batch, -1, self.num_sources, nodes, time_steps)
        return self.residual_conv(output), self.skip_conv(output), feature_loss


class HRA(nn.Module):
    """Hidden Representation Augmenter from the original GMRL architecture."""

    def __init__(
        self,
        *,
        num_nodes: int,
        num_sources: int,
        horizon: int,
        output_dim: int,
        channels: int,
        final_time: int,
        use_hra: bool,
    ) -> None:
        super().__init__()
        self.use_hra = use_hra
        self.final_time = final_time
        if use_hra:
            self.memory_count = 8
            self.memory_dim = channels
            self.flat_hidden = num_sources * num_nodes * channels * final_time
            self.memory = nn.Parameter(torch.empty(self.memory_count, self.memory_dim))
            self.query_weight = nn.Parameter(torch.empty(self.flat_hidden, self.memory_dim))
            self.fc_weight = nn.Parameter(torch.empty(self.memory_dim, self.flat_hidden))
            self.out_layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(in_channels=channels + self.memory_dim, out_channels=channels, kernel_size=(1, 1, 1)),
                nn.ReLU(),
                nn.Conv3d(in_channels=channels, out_channels=horizon * output_dim, kernel_size=(1, 1, 1)),
            )
        else:
            self.out_layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 1, 1)),
                nn.ReLU(),
                nn.Conv3d(in_channels=channels, out_channels=horizon * output_dim, kernel_size=(1, 1, 1)),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize HRA memory parameters."""
        if self.use_hra:
            nn.init.xavier_normal_(self.memory)
            nn.init.xavier_normal_(self.query_weight)
            nn.init.xavier_normal_(self.fc_weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict future tensor values from the accumulated skip representation."""
        batch, channels, sources, nodes, time_steps = hidden.shape
        if time_steps != self.final_time:
            raise ValueError(f"Expected final_time={self.final_time}, got {time_steps}")
        if self.use_hra:
            query = hidden.reshape(batch, -1) @ self.query_weight
            attention = torch.softmax(query @ self.memory.T, dim=1)
            prototype = attention @ self.memory
            memory_hidden = (prototype @ self.fc_weight).reshape(batch, channels, sources, nodes, time_steps)
            hidden = torch.cat((hidden, memory_hidden), dim=1)
        return self.out_layer(hidden)


class GMRL(nn.Module):
    """Gaussian Mixture Representation Learning model for dep/arr forecasting."""

    def __init__(self, config: GMRLConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.hidden_channels
        self.input_projection = nn.Conv3d(in_channels=config.input_dim, out_channels=channels, kernel_size=(1, 1, 1))
        self.temporal_embedding = nn.Parameter(torch.empty(channels, config.lag))
        self.location_embedding = nn.Parameter(torch.empty(channels, config.num_nodes))
        self.source_embedding = nn.Parameter(torch.empty(channels, config.num_sources))
        self.embedding_projection = nn.Sequential(
            nn.Conv3d(in_channels=2 * channels, out_channels=2 * channels, kernel_size=(1, 1, 1)),
            nn.ReLU(),
        )

        residual_channels = 2 * channels
        current_time = config.lag
        blocks: list[ResidualBlock] = []
        for dilation in config.dilations:
            blocks.append(
                ResidualBlock(
                    num_components=config.num_components,
                    num_nodes=config.num_nodes,
                    num_sources=config.num_sources,
                    channels=residual_channels,
                    current_time=current_time,
                    dilation=dilation,
                    kernel_size=config.kernel_size,
                    chunk_size=config.gmre_chunk_size,
                )
            )
            current_time -= dilation * (config.kernel_size - 1)
        self.residual_blocks = nn.ModuleList(blocks)
        self.hra = HRA(
            num_nodes=config.num_nodes,
            num_sources=config.num_sources,
            horizon=config.horizon,
            output_dim=config.output_dim,
            channels=residual_channels,
            final_time=current_time,
            use_hra=config.use_hra,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize model parameters."""
        nn.init.xavier_normal_(self.temporal_embedding)
        nn.init.xavier_normal_(self.location_embedding)
        nn.init.xavier_normal_(self.source_embedding)
        for name, parameter in self.named_parameters():
            if name.startswith(("temporal_embedding", "location_embedding", "source_embedding", "hra.")):
                continue
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.uniform_(parameter)

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future source values.

        Args:
            source: Tensor shaped [batch, lag, nodes, sources, input_dim].

        Returns:
            Prediction shaped [batch, horizon, nodes, sources, output_dim] and GMRE regularization loss.
        """
        expected = (self.config.lag, self.config.num_nodes, self.config.num_sources, self.config.input_dim)
        if tuple(source.shape[1:]) != expected:
            raise ValueError(f"Expected input shape [B,{expected}], got {tuple(source.shape)}")
        hidden = source.permute(0, 4, 3, 2, 1).contiguous()
        hidden = self.input_projection(hidden)
        batch, channels, sources, nodes, time_steps = hidden.shape
        ttse = (
            self.temporal_embedding.reshape(1, channels, 1, 1, time_steps)
            + self.location_embedding.reshape(1, channels, 1, nodes, 1)
            + self.source_embedding.reshape(1, channels, sources, 1, 1)
        )
        hidden = self.embedding_projection(torch.cat((hidden, ttse.expand(batch, -1, -1, -1, -1)), dim=1))

        skip_total: torch.Tensor | None = None
        feature_losses: list[torch.Tensor] = []
        for block in self.residual_blocks:
            residual = hidden
            hidden, skip, feature_loss = block(hidden)
            hidden = hidden + residual[..., -hidden.shape[-1] :]
            skip_total = skip if skip_total is None else skip + skip_total[..., -skip.shape[-1] :]
            feature_losses.append(feature_loss)
        if skip_total is None:
            raise RuntimeError("GMRL has no residual blocks")

        output = self.hra(skip_total)
        if output.shape[-1] != 1:
            output = output[..., -1:]
        output = output.reshape(
            source.shape[0],
            self.config.horizon,
            self.config.output_dim,
            self.config.num_sources,
            self.config.num_nodes,
            1,
        )
        output = output[..., 0].permute(0, 1, 4, 3, 2).contiguous()
        feature_loss = torch.stack(feature_losses).mean()
        return output, feature_loss
