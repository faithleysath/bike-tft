"""Objective-aware data loading for NYC AGCRN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from forecasting_models.agcrn_nyc.data import StandardScaler, load_bundle_arrays, split_window_starts


DEFAULT_BUNDLE = Path("dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz")
TARGET_MODES = ("raw", "log1p", "seasonal_residual")


class ObjectiveWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset with model-space target, raw target, and baseline arrays."""

    def __init__(
        self,
        features: np.ndarray,
        model_targets: np.ndarray,
        raw_targets: np.ndarray,
        baselines: np.ndarray,
        *,
        starts: np.ndarray,
        lag: int,
        horizon: int,
    ) -> None:
        self.features = features
        self.model_targets = model_targets
        self.raw_targets = raw_targets
        self.baselines = baselines
        self.starts = starts.astype(np.int64)
        self.lag = lag
        self.horizon = horizon

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        target_start = start + self.lag
        target_end = target_start + self.horizon
        x = self.features[start : start + self.lag]
        y_model = self.model_targets[target_start:target_end]
        y_raw = self.raw_targets[target_start:target_end]
        baseline = self.baselines[target_start:target_end]
        return torch.from_numpy(x), torch.from_numpy(y_model), torch.from_numpy(y_raw), torch.from_numpy(baseline)


@dataclass(frozen=True)
class ObjectiveBundleData:
    """Loaded and normalized data for objective ablation training."""

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    target_mode: str
    seasonal_lag: int
    feature_names: list[str]
    station_ids: list[str]
    timestamps: list[str]
    window_counts: dict[str, int]
    input_dim: int
    num_nodes: int

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


def build_seasonal_baseline(raw_targets: np.ndarray, *, seasonal_lag: int = 168, fallback_lag: int = 24) -> np.ndarray:
    """Build a no-leakage seasonal count baseline from prior observed target values."""
    if seasonal_lag <= 0 or fallback_lag <= 0:
        raise ValueError("seasonal_lag and fallback_lag must be positive")
    baseline = np.zeros_like(raw_targets, dtype=np.float32)
    time_count = raw_targets.shape[0]
    for index in range(time_count):
        if index >= seasonal_lag:
            source = index - seasonal_lag
        elif index >= fallback_lag:
            source = index - fallback_lag
        elif index > 0:
            source = index - 1
        else:
            continue
        baseline[index] = raw_targets[source]
    return baseline.astype(np.float32)


def transform_targets(
    raw_targets: np.ndarray,
    *,
    target_mode: str,
    seasonal_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model-space targets before scaling and count-space baselines."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}")
    baselines = np.zeros_like(raw_targets, dtype=np.float32)
    if target_mode == "raw":
        return raw_targets.astype(np.float32), baselines
    if target_mode == "log1p":
        return np.log1p(raw_targets).astype(np.float32), baselines
    baselines = build_seasonal_baseline(raw_targets, seasonal_lag=seasonal_lag)
    return (raw_targets - baselines).astype(np.float32), baselines


def make_objective_dataloaders(
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
) -> ObjectiveBundleData:
    """Load the NYC bundle and create chronological dataloaders for objective ablations."""
    arrays = load_bundle_arrays(bundle_path)
    features = arrays["features"].astype(np.float32)
    raw_targets = np.concatenate(
        [arrays["target_dep"].astype(np.float32), arrays["target_arr"].astype(np.float32)],
        axis=-1,
    )
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
    features = feature_scaler.transform(features)
    model_targets = target_scaler.transform(model_targets)

    train_dataset = ObjectiveWindowDataset(
        features,
        model_targets,
        raw_targets,
        baselines,
        starts=train_starts,
        lag=lag,
        horizon=horizon,
    )
    val_dataset = ObjectiveWindowDataset(
        features,
        model_targets,
        raw_targets,
        baselines,
        starts=val_starts,
        lag=lag,
        horizon=horizon,
    )
    test_dataset = ObjectiveWindowDataset(
        features,
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
    return ObjectiveBundleData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_mode=target_mode,
        seasonal_lag=seasonal_lag,
        feature_names=[str(item) for item in arrays["feature_names"].tolist()],
        station_ids=[str(item) for item in arrays["station_ids"].tolist()],
        timestamps=[str(item) for item in arrays["timestamps"].tolist()],
        window_counts={
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
            "total": len(train_dataset) + len(val_dataset) + len(test_dataset),
        },
        input_dim=input_dim,
        num_nodes=num_nodes,
    )

