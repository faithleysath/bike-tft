#!/usr/bin/env python3
"""Train AGCRN on the stage 2 bike-sharing bundle."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGCRN_UPSTREAM_ROOT = Path(__file__).resolve().parent / "agcrn_upstream"
if str(AGCRN_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AGCRN_UPSTREAM_ROOT))

AGCRN = importlib.import_module("model.AGCRN").AGCRN


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AGCRN on the stage 2 shared-bike bundle."
    )
    parser.add_argument(
        "--bundle",
        default="data/processed/stage_02_feature_enrichment/agcrn_stage3_bundle.npz",
        help="Path to the stage 2 AGCRN bundle.",
    )
    parser.add_argument(
        "--split-manifest",
        default="data/processed/stage_02_feature_enrichment/split_manifest.json",
        help="Path to the shared train/validation/test split manifest.",
    )
    parser.add_argument(
        "--feature-manifest",
        default="data/processed/stage_02_feature_enrichment/feature_manifest.json",
        help="Path to the feature manifest paired with the bundle.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/stage_03_baselines_and_ablation/agcrn_dep_default",
        help="Directory used for checkpoints, metrics, and training logs.",
    )
    parser.add_argument(
        "--target-key",
        choices=("target_dep", "target_arr"),
        default="target_dep",
        help="Which target tensor in the bundle should be optimized.",
    )
    parser.add_argument("--lag", type=int, default=12, help="Encoder sequence length.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cheb-k", type=int, default=2)
    parser.add_argument(
        "--loss",
        choices=("mae", "mse"),
        default="mae",
        help="Training loss applied on normalized targets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection: auto, cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience measured in epochs without validation improvement.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Optional max train batches per epoch for smoke runs.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Optional max validation batches per epoch for smoke runs.",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=None,
        help="Optional max test batches during the final evaluation.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    """Choose a torch device from CLI input."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model_parameters(model: nn.Module) -> None:
    """Match the upstream AGCRN parameter initialization scheme."""
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
        else:
            nn.init.uniform_(parameter)


def timestamp_index(timestamps: pd.DatetimeIndex, value: str) -> int:
    """Return the integer position of a split boundary timestamp."""
    match = np.where(timestamps == pd.Timestamp(value))[0]
    if len(match) != 1:
        raise ValueError(f"Timestamp {value!r} not found exactly once in bundle timestamps")
    return int(match[0])


@dataclass(frozen=True)
class TensorScaler:
    """Per-feature input scaler and scalar target scaler."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    def inverse_target(self, value: torch.Tensor) -> torch.Tensor:
        """Undo target normalization on a torch tensor."""
        mean = torch.as_tensor(self.target_mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.target_std, device=value.device, dtype=value.dtype)
        return value * std + mean

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        """Serialize scaler stats into NumPy arrays."""
        return {
            "feature_mean": self.feature_mean.astype(np.float32),
            "feature_std": self.feature_std.astype(np.float32),
            "target_mean": np.asarray([self.target_mean], dtype=np.float32),
            "target_std": np.asarray([self.target_std], dtype=np.float32),
        }


class BundleWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Windowed access into the dense [T, N, D] bike-sharing bundle."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        target: np.ndarray,
        start_indices: np.ndarray,
        lag: int,
        horizon: int,
        scaler: TensorScaler,
    ) -> None:
        self.features = features
        self.target = target
        self.start_indices = start_indices.astype(np.int64, copy=False)
        self.lag = lag
        self.horizon = horizon
        self.scaler = scaler

    def __len__(self) -> int:
        return int(len(self.start_indices))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.start_indices[index])
        x = self.features[start : start + self.lag]
        y = self.target[start + self.lag : start + self.lag + self.horizon]
        x_scaled = (x - self.scaler.feature_mean) / self.scaler.feature_std
        y_scaled = (y - self.scaler.target_mean) / self.scaler.target_std
        return (
            torch.from_numpy(np.asarray(x_scaled, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_scaled, dtype=np.float32)),
        )


def build_window_starts(
    *,
    timestamps: pd.DatetimeIndex,
    split_manifest: dict[str, Any],
    split_name: str,
    lag: int,
    horizon: int,
) -> np.ndarray:
    """Generate sample start indices whose targets fall inside the requested split."""
    split_payload = split_manifest["splits"][split_name]
    target_start_idx = timestamp_index(timestamps, split_payload["start"])
    target_end_idx = timestamp_index(timestamps, split_payload["end"])
    start_min = max(0, target_start_idx - lag)
    start_max = target_end_idx - lag - horizon + 1
    if start_max < start_min:
        return np.empty(0, dtype=np.int64)
    return np.arange(start_min, start_max + 1, dtype=np.int64)


def fit_scaler(
    *,
    features: np.ndarray,
    target: np.ndarray,
    timestamps: pd.DatetimeIndex,
    split_manifest: dict[str, Any],
) -> TensorScaler:
    """Fit feature and target normalizers on the training time range only."""
    train_end = timestamp_index(timestamps, split_manifest["splits"]["train"]["end"])
    train_features = features[: train_end + 1]
    train_target = target[: train_end + 1]
    feature_mean = train_features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = train_features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std).astype(np.float32)
    target_mean = float(train_target.mean(dtype=np.float64))
    target_std = float(train_target.std(dtype=np.float64))
    if target_std == 0:
        target_std = 1.0
    return TensorScaler(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )


def build_loaders(
    *,
    features: np.ndarray,
    target: np.ndarray,
    timestamps: pd.DatetimeIndex,
    split_manifest: dict[str, Any],
    lag: int,
    horizon: int,
    batch_size: int,
    num_workers: int,
    scaler: TensorScaler,
) -> tuple[dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]], dict[str, int]]:
    """Create stage 3 train/validation/test dataloaders from the bundle."""
    datasets: dict[str, BundleWindowDataset] = {}
    counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        starts = build_window_starts(
            timestamps=timestamps,
            split_manifest=split_manifest,
            split_name=split_name,
            lag=lag,
            horizon=horizon,
        )
        datasets[split_name] = BundleWindowDataset(
            features=features,
            target=target,
            start_indices=starts,
            lag=lag,
            horizon=horizon,
            scaler=scaler,
        )
        counts[split_name] = len(starts)
        if counts[split_name] == 0:
            raise ValueError(f"No window samples available for split {split_name!r}")

    loaders: dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
    }
    return loaders, counts


def iter_limited(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    limit_batches: int | None,
) -> Iterator[tuple[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Yield loader batches with an optional hard stop for smoke runs."""
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        yield batch_index, batch


def run_epoch(
    *,
    model: Any,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[float, int]:
    """Run one training or evaluation epoch on normalized targets."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    batches = 0
    for _, (data, target) in iter_limited(loader, limit_batches):
        data = data.to(device)
        target = target.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        output = model(data, target, teacher_forcing_ratio=0.0)
        loss = criterion(output, target)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    if batches == 0:
        raise ValueError("Epoch ran zero batches")
    return total_loss / batches, batches


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    """Compute aggregate and per-horizon regression metrics."""
    diff = pred - true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mask = np.abs(true) > 1e-6
    if mask.any():
        mape = float(np.mean(np.abs(diff[mask] / true[mask])))
    else:
        mape = float("nan")

    horizon_rows: list[dict[str, float | int]] = []
    for horizon_index in range(pred.shape[1]):
        pred_h = pred[:, horizon_index, ...]
        true_h = true[:, horizon_index, ...]
        diff_h = pred_h - true_h
        mask_h = np.abs(true_h) > 1e-6
        mape_h = float(np.mean(np.abs(diff_h[mask_h] / true_h[mask_h]))) if mask_h.any() else float("nan")
        horizon_rows.append(
            {
                "horizon": horizon_index + 1,
                "mae": float(np.mean(np.abs(diff_h))),
                "rmse": float(np.sqrt(np.mean(np.square(diff_h)))),
                "mape": mape_h,
            }
        )
    return {
        "aggregate": {"mae": mae, "rmse": rmse, "mape": mape},
        "per_horizon": horizon_rows,
    }


def evaluate_loader(
    *,
    model: Any,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    scaler: TensorScaler,
    device: torch.device,
    limit_batches: int | None,
) -> dict[str, Any]:
    """Run inference and compute metrics on the original target scale."""
    model.eval()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    with torch.no_grad():
        for _, (data, target) in iter_limited(loader, limit_batches):
            data = data.to(device)
            target = target.to(device)
            output = model(data, target, teacher_forcing_ratio=0.0)
            output = scaler.inverse_target(output).cpu().numpy()
            target = scaler.inverse_target(target).cpu().numpy()
            predictions.append(output)
            truths.append(target)
    if not predictions:
        raise ValueError("Evaluation ran zero batches")
    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(truths, axis=0)
    metrics = compute_metrics(pred, true)
    metrics["sample_count"] = int(pred.shape[0])
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON to disk."""
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Train AGCRN on the stage 2 bike-sharing bundle and save run artifacts."""
    args = parse_args()
    if args.lag < 1 or args.horizon < 1:
        raise SystemExit("--lag and --horizon must both be positive")
    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--batch-size and --epochs must both be positive")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")

    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = np.load(project_path(args.bundle), allow_pickle=False)
    split_manifest = json.loads(project_path(args.split_manifest).read_text(encoding="utf-8"))
    feature_manifest = json.loads(project_path(args.feature_manifest).read_text(encoding="utf-8"))
    feature_names = bundle["feature_names"].astype(str).tolist()
    expected_features = feature_manifest["bundle_feature_order"]
    if feature_names != expected_features:
        raise ValueError("Bundle feature_names do not match feature_manifest bundle_feature_order")

    features = bundle["features"].astype(np.float32)
    target = bundle[args.target_key].astype(np.float32)
    timestamps = pd.DatetimeIndex(pd.to_datetime(bundle["timestamps"]))
    node_ids = bundle["node_ids"].astype(str).tolist()

    scaler = fit_scaler(
        features=features,
        target=target,
        timestamps=timestamps,
        split_manifest=split_manifest,
    )
    loaders, window_counts = build_loaders(
        features=features,
        target=target,
        timestamps=timestamps,
        split_manifest=split_manifest,
        lag=args.lag,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=scaler,
    )

    device = resolve_device(args.device)
    model_args = SimpleNamespace(
        num_nodes=len(node_ids),
        input_dim=features.shape[2],
        output_dim=target.shape[2],
        horizon=args.horizon,
        num_layers=args.num_layers,
        rnn_units=args.rnn_units,
        cheb_k=args.cheb_k,
        embed_dim=args.embed_dim,
        default_graph=True,
    )
    model = AGCRN(model_args).to(device)
    initialize_model_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion: nn.Module
    if args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = output_dir / "best_model.pt"

    print(
        "Training AGCRN on stage 3 bundle:",
        json.dumps(
            {
                "device": str(device),
                "target_key": args.target_key,
                "windows": window_counts,
                "input_dim": int(features.shape[2]),
                "num_nodes": len(node_ids),
            },
            ensure_ascii=True,
        ),
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_batches = run_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            limit_batches=args.limit_train_batches,
        )
        val_loss, val_batches = run_epoch(
            model=model,
            loader=loaders["validation"],
            optimizer=None,
            criterion=criterion,
            device=device,
            limit_batches=args.limit_val_batches,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_batches": train_batches,
                "val_batches": val_batches,
            }
        )
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise ValueError(
                f"Encountered non-finite loss values at epoch {epoch}: "
                f"train_loss={train_loss}, val_loss={val_loss}"
            )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"(train_batches={train_batches}, val_batches={val_batches})"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "target_key": args.target_key,
                "feature_names": feature_names,
                "node_ids": node_ids,
                "model_args": vars(model_args),
                "cli_args": vars(args),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stop after {epoch} epochs without validation improvement")
                break

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    validation_metrics = evaluate_loader(
        model=model,
        loader=loaders["validation"],
        scaler=scaler,
        device=device,
        limit_batches=args.limit_val_batches,
    )
    test_metrics = evaluate_loader(
        model=model,
        loader=loaders["test"],
        scaler=scaler,
        device=device,
        limit_batches=args.limit_test_batches,
    )

    pd.DataFrame(history_rows).to_csv(output_dir / "train_history.csv", index=False)
    pd.DataFrame(test_metrics["per_horizon"]).to_csv(output_dir / "test_horizon_metrics.csv", index=False)
    scaler_payload = scaler.to_npz_payload()
    np.savez_compressed(
        output_dir / "scalers.npz",
        feature_mean=scaler_payload["feature_mean"],
        feature_std=scaler_payload["feature_std"],
        target_mean=scaler_payload["target_mean"],
        target_std=scaler_payload["target_std"],
    )
    write_json(
        output_dir / "metrics_summary.json",
        {
            "device": str(device),
            "target_key": args.target_key,
            "feature_dim": int(features.shape[2]),
            "num_nodes": len(node_ids),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "window_counts": window_counts,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "limit_test_batches": args.limit_test_batches,
            "validation_metrics": validation_metrics["aggregate"],
            "test_metrics": test_metrics["aggregate"],
        },
    )
    write_json(
        output_dir / "hparams.json",
        {
            **vars(args),
            "resolved_device": str(device),
            "feature_names": feature_names,
            "num_nodes": len(node_ids),
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "test_metrics": test_metrics["aggregate"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
