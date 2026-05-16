"""Data loading for the NYC GMRL adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from forecasting_models.agcrn_nyc.data import StandardScaler, load_bundle_arrays, split_window_starts


DEFAULT_BUNDLE = Path("dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz")
TARGET_MODES = ("raw", "log1p")


class GMRLWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Sliding-window tensor dataset shaped for GMRL."""

    def __init__(
        self,
        model_values: np.ndarray,
        raw_values: np.ndarray,
        *,
        starts: np.ndarray,
        lag: int,
        horizon: int,
    ) -> None:
        self.model_values = model_values
        self.raw_values = raw_values
        self.starts = starts.astype(np.int64)
        self.lag = lag
        self.horizon = horizon

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        target_start = start + self.lag
        target_end = target_start + self.horizon
        x = self.model_values[start : start + self.lag, :, :, None]
        y_model = self.model_values[target_start:target_end, :, :, None]
        y_raw = self.raw_values[target_start:target_end]
        return torch.from_numpy(x), torch.from_numpy(y_model), torch.from_numpy(y_raw)


@dataclass(frozen=True)
class GMRLBundleData:
    """Loaded and normalized data for GMRL training."""

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    target_scaler: StandardScaler
    target_mode: str
    station_ids: list[str]
    timestamps: list[str]
    window_counts: dict[str, int]
    num_nodes: int
    num_sources: int
    input_dim: int

    def inverse_model_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Convert model-space values back to original count space."""
        restore_last_dim = values.ndim == 5 and values.shape[-1] == 1
        if restore_last_dim:
            values = values.squeeze(-1)
        unscaled = self.target_scaler.inverse_transform_tensor(values)
        if self.target_mode == "raw":
            restored = unscaled.clamp_min(0)
            return restored.unsqueeze(-1) if restore_last_dim else restored
        if self.target_mode == "log1p":
            restored = torch.expm1(unscaled).clamp_min(0)
            return restored.unsqueeze(-1) if restore_last_dim else restored
        raise ValueError(f"Unsupported target_mode: {self.target_mode}")


def transform_targets(raw_targets: np.ndarray, target_mode: str) -> np.ndarray:
    """Transform raw count targets into model space."""
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}")
    if target_mode == "raw":
        return raw_targets.astype(np.float32)
    return np.log1p(raw_targets).astype(np.float32)


def make_gmrl_dataloaders(
    bundle_path: str | Path = DEFAULT_BUNDLE,
    *,
    target_mode: str = "log1p",
    lag: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> GMRLBundleData:
    """Load the NYC bundle and create chronological dataloaders for GMRL."""
    arrays = load_bundle_arrays(bundle_path)
    raw_targets = np.concatenate(
        [arrays["target_dep"].astype(np.float32), arrays["target_arr"].astype(np.float32)],
        axis=-1,
    )
    model_values = transform_targets(raw_targets, target_mode)
    time_count, num_nodes, num_sources = model_values.shape
    train_starts, val_starts, test_starts = split_window_starts(
        time_count=time_count,
        lag=lag,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_target_start = lag
    train_target_end = int(train_starts[-1] + lag + horizon + 1)
    target_scaler = StandardScaler.fit(model_values[train_target_start:train_target_end])
    model_values = target_scaler.transform(model_values)

    train_dataset = GMRLWindowDataset(model_values, raw_targets, starts=train_starts, lag=lag, horizon=horizon)
    val_dataset = GMRLWindowDataset(model_values, raw_targets, starts=val_starts, lag=lag, horizon=horizon)
    test_dataset = GMRLWindowDataset(model_values, raw_targets, starts=test_starts, lag=lag, horizon=horizon)
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
    return GMRLBundleData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=target_scaler,
        target_mode=target_mode,
        station_ids=[str(item) for item in arrays["station_ids"].tolist()],
        timestamps=[str(item) for item in arrays["timestamps"].tolist()],
        window_counts={
            "train": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
            "total": len(train_dataset) + len(val_dataset) + len(test_dataset),
        },
        num_nodes=num_nodes,
        num_sources=num_sources,
        input_dim=1,
    )
