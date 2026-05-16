"""Data loading utilities for the NYC AGCRN bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_BUNDLE = Path("dataset/preprocessing/processed/nyc/nyc_agcrn_bundle.npz")


@dataclass(frozen=True)
class StandardScaler:
    """Per-channel standard scaler for numpy arrays and torch tensors."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        mean = values.mean(axis=tuple(range(values.ndim - 1)), keepdims=True).astype(np.float32)
        std = values.std(axis=tuple(range(values.ndim - 1)), keepdims=True).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * std + mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": np.squeeze(self.mean).astype(float).tolist(),
            "std": np.squeeze(self.std).astype(float).tolist(),
        }


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset over normalized feature and target arrays."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        starts: np.ndarray,
        lag: int,
        horizon: int,
    ) -> None:
        self.features = features
        self.targets = targets
        self.starts = starts.astype(np.int64)
        self.lag = lag
        self.horizon = horizon

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        x = self.features[start : start + self.lag]
        y = self.targets[start + self.lag : start + self.lag + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass(frozen=True)
class BundleData:
    """Loaded and normalized NYC AGCRN data."""

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_names: list[str]
    station_ids: list[str]
    timestamps: list[str]
    window_counts: dict[str, int]
    input_dim: int
    num_nodes: int


def split_window_starts(
    *,
    time_count: int,
    lag: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build chronological train/validation/test window start indices."""
    total = time_count - lag - horizon + 1
    if total <= 0:
        raise ValueError("Not enough timesteps to build windows")
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if train_count <= 0 or val_count <= 0 or train_count + val_count >= total:
        raise ValueError("Invalid split ratios for available windows")
    starts = np.arange(total, dtype=np.int64)
    train = starts[:train_count]
    val = starts[train_count : train_count + val_count]
    test = starts[train_count + val_count :]
    return train, val, test


def load_bundle_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load bundle arrays without pickle."""
    bundle = np.load(path)
    required = {"features", "target_dep", "target_arr", "timestamps", "station_ids", "feature_names"}
    missing = required.difference(bundle.files)
    if missing:
        raise ValueError(f"NYC bundle is missing array(s): {sorted(missing)}")
    return {key: bundle[key] for key in bundle.files}


def make_dataloaders(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    *,
    lag: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> BundleData:
    """Load the NYC bundle and create chronological dataloaders."""
    arrays = load_bundle_arrays(bundle_path)
    features = arrays["features"].astype(np.float32)
    targets = np.concatenate(
        [arrays["target_dep"].astype(np.float32), arrays["target_arr"].astype(np.float32)],
        axis=-1,
    )
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
    target_scaler = StandardScaler.fit(targets[train_target_start:train_target_end])
    features = feature_scaler.transform(features)
    targets = target_scaler.transform(targets)

    train_dataset = WindowDataset(features, targets, starts=train_starts, lag=lag, horizon=horizon)
    val_dataset = WindowDataset(features, targets, starts=val_starts, lag=lag, horizon=horizon)
    test_dataset = WindowDataset(features, targets, starts=test_starts, lag=lag, horizon=horizon)
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
    return BundleData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
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
