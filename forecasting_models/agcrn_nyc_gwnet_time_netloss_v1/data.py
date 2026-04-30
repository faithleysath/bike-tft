"""Time-aware objective data loading for Graph WaveNet experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from forecasting_models.agcrn_nyc.data import StandardScaler, load_bundle_arrays, split_window_starts
from forecasting_models.agcrn_nyc_objective_v1.data import DEFAULT_BUNDLE, TARGET_MODES, transform_targets


def build_timestamp_features(timestamps: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build deterministic target-time features for every timestamp."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    hour = index.hour.to_numpy(dtype=np.float32)
    day_of_week = index.dayofweek.to_numpy(dtype=np.float32)
    month = (index.month.to_numpy(dtype=np.float32) - 1.0).astype(np.float32)
    is_weekend = (day_of_week >= 5).astype(np.float32)
    features = np.stack(
        [
            np.sin(2.0 * np.pi * hour / 24.0),
            np.cos(2.0 * np.pi * hour / 24.0),
            np.sin(2.0 * np.pi * day_of_week / 7.0),
            np.cos(2.0 * np.pi * day_of_week / 7.0),
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
            is_weekend,
        ],
        axis=-1,
    )
    names = [
        "target_hour_sin",
        "target_hour_cos",
        "target_day_of_week_sin",
        "target_day_of_week_cos",
        "target_month_sin",
        "target_month_cos",
        "target_is_weekend",
    ]
    return features.astype(np.float32), names


class TimeAwareObjectiveWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset with target-time features for each forecast horizon step."""

    def __init__(
        self,
        features: np.ndarray,
        future_time_features: np.ndarray,
        model_targets: np.ndarray,
        raw_targets: np.ndarray,
        baselines: np.ndarray,
        *,
        starts: np.ndarray,
        lag: int,
        horizon: int,
    ) -> None:
        self.features = features
        self.future_time_features = future_time_features
        self.model_targets = model_targets
        self.raw_targets = raw_targets
        self.baselines = baselines
        self.starts = starts.astype(np.int64)
        self.lag = lag
        self.horizon = horizon

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        target_start = start + self.lag
        target_end = target_start + self.horizon
        x = self.features[start : start + self.lag]
        future_time = self.future_time_features[target_start:target_end]
        y_model = self.model_targets[target_start:target_end]
        y_raw = self.raw_targets[target_start:target_end]
        baseline = self.baselines[target_start:target_end]
        return (
            torch.from_numpy(x),
            torch.from_numpy(future_time),
            torch.from_numpy(y_model),
            torch.from_numpy(y_raw),
            torch.from_numpy(baseline),
        )


@dataclass(frozen=True)
class TimeAwareObjectiveBundleData:
    """Loaded and normalized data for time-aware Graph WaveNet training."""

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    target_mode: str
    seasonal_lag: int
    feature_names: list[str]
    future_time_feature_names: list[str]
    station_ids: list[str]
    timestamps: list[str]
    window_counts: dict[str, int]
    input_dim: int
    future_time_dim: int
    num_nodes: int
    net_loss_scale: float

    def inverse_model_tensor(self, values: torch.Tensor, baselines: torch.Tensor) -> torch.Tensor:
        """Convert model-space predictions or labels back to original count space."""
        unscaled = self.target_scaler.inverse_transform_tensor(values)
        if self.target_mode == "raw":
            return unscaled.clamp_min(0)
        if self.target_mode == "log1p":
            return torch.expm1(unscaled).clamp_min(0)
        if self.target_mode == "seasonal_residual":
            return (unscaled + baselines.to(device=values.device, dtype=values.dtype)).clamp_min(0)
        raise ValueError(f"Unsupported target_mode: {self.target_mode}")


def make_time_aware_dataloaders(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    *,
    target_mode: str = "raw",
    seasonal_lag: int = 168,
    lag: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> TimeAwareObjectiveBundleData:
    """Load the NYC bundle and create chronological loaders with future target-time features."""
    arrays = load_bundle_arrays(bundle_path)
    features = arrays["features"].astype(np.float32)
    timestamps = [str(item) for item in arrays["timestamps"].tolist()]
    future_time_features, future_time_feature_names = build_timestamp_features(timestamps)
    raw_targets = np.concatenate(
        [arrays["target_dep"].astype(np.float32), arrays["target_arr"].astype(np.float32)],
        axis=-1,
    )
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}")
    model_targets, baselines = transform_targets(raw_targets, target_mode=target_mode, seasonal_lag=seasonal_lag)
    time_count, num_nodes, input_dim = features.shape
    train_starts, val_starts, test_starts = split_window_starts(
        time_count=time_count,
        lag=lag,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_input_end = int(train_starts[-1] + lag + 1)
    train_target_start = lag
    train_target_end = int(train_starts[-1] + lag + horizon + 1)
    feature_scaler = StandardScaler.fit(features[:train_input_end])
    target_scaler = StandardScaler.fit(model_targets[train_target_start:train_target_end])
    train_raw_targets = raw_targets[train_target_start:train_target_end]
    train_net = train_raw_targets[..., 1] - train_raw_targets[..., 0]
    mean_abs_train_net = float(np.mean(np.abs(train_net)))
    net_loss_scale = float(max(mean_abs_train_net * 5.0, 10.0))
    features = feature_scaler.transform(features)
    model_targets = target_scaler.transform(model_targets)

    train_dataset = TimeAwareObjectiveWindowDataset(
        features,
        future_time_features,
        model_targets,
        raw_targets,
        baselines,
        starts=train_starts,
        lag=lag,
        horizon=horizon,
    )
    val_dataset = TimeAwareObjectiveWindowDataset(
        features,
        future_time_features,
        model_targets,
        raw_targets,
        baselines,
        starts=val_starts,
        lag=lag,
        horizon=horizon,
    )
    test_dataset = TimeAwareObjectiveWindowDataset(
        features,
        future_time_features,
        model_targets,
        raw_targets,
        baselines,
        starts=test_starts,
        lag=lag,
        horizon=horizon,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return TimeAwareObjectiveBundleData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_mode=target_mode,
        seasonal_lag=seasonal_lag,
        feature_names=[str(item) for item in arrays["feature_names"].tolist()],
        future_time_feature_names=future_time_feature_names,
        station_ids=[str(item) for item in arrays["station_ids"].tolist()],
        timestamps=timestamps,
        window_counts={
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
            "total": len(train_dataset) + len(val_dataset) + len(test_dataset),
        },
        input_dim=input_dim,
        future_time_dim=future_time_features.shape[-1],
        num_nodes=num_nodes,
        net_loss_scale=net_loss_scale,
    )
