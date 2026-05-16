"""Shared utilities for external-paper NYC reproduction runs."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from forecasting_models.agcrn_nyc.data import load_bundle_arrays, split_window_starts
from forecasting_models.agcrn_nyc.train import compute_metrics, write_horizon_metrics
from forecasting_models.agcrn_nyc_objective_v1.data import ObjectiveBundleData


DEFAULT_BUNDLE = Path("dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz")
DEFAULT_RELATION_GRAPHS = Path("dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k20.npz")


def set_seed(seed: int) -> None:
    """Make stochastic training behavior reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_batches[T](loader: Iterable[T], limit: int | None) -> Iterable[T]:
    """Yield at most ``limit`` batches from a dataloader."""
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write training history CSV."""
    if not rows:
        raise ValueError("Cannot write empty history")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_relation_support(path: str | Path, station_ids: list[str]) -> np.ndarray:
    """Load a row-normalized OD support for models that need an initial graph."""
    arrays = np.load(path, allow_pickle=False)
    required = {"od_forward_support", "station_ids"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"Relation graph artifact is missing array(s): {sorted(missing)}")
    graph_station_ids = [str(item) for item in arrays["station_ids"].tolist()]
    if graph_station_ids != [str(item) for item in station_ids]:
        raise ValueError("Relation graph station_ids do not match the training bundle station_ids")
    support = arrays["od_forward_support"].astype(np.float32)
    if support.shape != (len(station_ids), len(station_ids)):
        raise ValueError(f"Invalid support shape: {support.shape}")
    support = support.copy()
    np.fill_diagonal(support, np.maximum(np.diag(support), 1.0))
    row_sum = support.sum(axis=1, keepdims=True)
    support = support / np.where(row_sum <= 1e-6, 1.0, row_sum)
    return support.astype(np.float32)


def make_esg_static_feature(
    bundle_path: str | Path,
    *,
    target_mode: str,
    lag: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> np.ndarray:
    """Build ESG's historical node feature matrix from training-period dep/arr targets."""
    arrays = load_bundle_arrays(bundle_path)
    raw_targets = np.concatenate(
        [arrays["target_dep"].astype(np.float32), arrays["target_arr"].astype(np.float32)],
        axis=-1,
    )
    values = np.log1p(raw_targets) if target_mode == "log1p" else raw_targets
    train_starts, _val_starts, _test_starts = split_window_starts(
        time_count=values.shape[0],
        lag=lag,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_target_end = int(train_starts[-1] + lag + horizon + 1)
    train_values = values[:train_target_end]
    mean = train_values.mean()
    std = train_values.std()
    if std < 1e-6:
        std = 1.0
    standardized = ((train_values - mean) / std).astype(np.float32)
    return np.transpose(standardized, (0, 2, 1)).reshape(-1, standardized.shape[1]).astype(np.float32)


@torch.no_grad()
def collect_objective_predictions(
    model: torch.nn.Module,
    data: ObjectiveBundleData,
    *,
    device: torch.device,
    limit_batches: int | None,
    call_model,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect inverse-transformed predictions from an ObjectiveBundleData test loader."""
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in limited_batches(data.test_loader, limit_batches):
        x, _y, raw_y, baseline = batch
        x = x.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        pred = call_model(model, x)
        preds.append(data.inverse_model_tensor(pred, baseline).cpu())
        labels.append(raw_y.cpu())
    if not preds:
        raise RuntimeError("No test batches were processed")
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def write_metrics(output_dir: Path, pred: torch.Tensor, true: torch.Tensor) -> dict[str, object]:
    """Compute and write standard dep/arr metrics."""
    metrics, horizon_rows = compute_metrics(pred, true)
    write_horizon_metrics(output_dir / "test_horizon_metrics.csv", horizon_rows)
    return metrics

