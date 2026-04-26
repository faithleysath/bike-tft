"""Train AGCRN on the model-ready NYC Citi Bike bundle."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .data import DEFAULT_BUNDLE, BundleData, make_dataloaders
from .model import AGCRN, AGCRNConfig


TARGET_NAMES = ("dep", "arr")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC AGCRN for dep+arr forecasting.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/agcrn_nyc/runs/default")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cheb-k", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Make stochastic training behavior reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    """Resolve the requested training device."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def limited_batches(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    limit: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Yield at most ``limit`` batches from a dataloader."""
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def train_epoch(
    model: AGCRN,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    limit_batches: int | None,
) -> float:
    """Run one training epoch."""
    model.train()
    losses: list[float] = []
    for x, y in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("No training batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: AGCRN,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss_fn: nn.Module,
    device: torch.device,
    limit_batches: int | None,
) -> float:
    """Evaluate normalized MAE loss."""
    model.eval()
    losses: list[float] = []
    for x, y in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        losses.append(float(loss_fn(pred, y).item()))
    if not losses:
        raise RuntimeError("No evaluation batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def collect_predictions(
    model: AGCRN,
    data: BundleData,
    *,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect inverse-scaled predictions and labels from the test set."""
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, y in limited_batches(data.test_loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        preds.append(data.target_scaler.inverse_transform_tensor(pred).cpu())
        labels.append(data.target_scaler.inverse_transform_tensor(y).cpu())
    if not preds:
        raise RuntimeError("No test batches were processed")
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def metric_payload(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    """Compute MAE/RMSE/MAPE for tensors in original units."""
    error = pred - true
    mae = torch.mean(torch.abs(error)).item()
    rmse = torch.sqrt(torch.mean(error * error)).item()
    mask = torch.abs(true) > 1e-6
    if bool(mask.any()):
        mape = torch.mean(torch.abs(error[mask] / true[mask])).item()
    else:
        mape = float("nan")
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def compute_metrics(pred: torch.Tensor, true: torch.Tensor) -> tuple[dict[str, object], list[dict[str, float | int | str]]]:
    """Compute aggregate and horizon metrics for dep and arr targets."""
    target_metrics = {
        name: metric_payload(pred[..., index], true[..., index])
        for index, name in enumerate(TARGET_NAMES)
    }
    average_metrics = metric_payload(pred, true)
    rows: list[dict[str, float | int | str]] = []
    for horizon_index in range(pred.shape[1]):
        for target_index, target_name in enumerate(TARGET_NAMES):
            metrics = metric_payload(pred[:, horizon_index, :, target_index], true[:, horizon_index, :, target_index])
            rows.append({"horizon": horizon_index + 1, "target": target_name, **metrics})
    return {"targets": target_metrics, "average": average_metrics}, rows


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write train history CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def write_horizon_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write per-horizon metrics CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon", "target", "mae", "rmse", "mape"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Train and evaluate the NYC AGCRN model."""
    try:
        args = parse_args()
        set_seed(args.seed)
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = make_dataloaders(
            args.bundle,
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        config = AGCRNConfig(
            num_nodes=data.num_nodes,
            input_dim=data.input_dim,
            output_dim=2,
            horizon=args.horizon,
            embed_dim=args.embed_dim,
            rnn_units=args.rnn_units,
            num_layers=args.num_layers,
            cheb_k=args.cheb_k,
        )
        model = AGCRN(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
        loss_fn = nn.L1Loss()

        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            started_at = time.time()
            train_loss = train_epoch(
                model,
                data.train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                limit_batches=args.limit_train_batches,
            )
            val_loss = evaluate_loss(
                model,
                data.val_loader,
                loss_fn=loss_fn,
                device=device,
                limit_batches=args.limit_val_batches,
            )
            seconds = time.time() - started_at
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "seconds": seconds,
                }
            )
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} seconds={seconds:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epoch(s).")
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a best model state")

        model.load_state_dict(best_state)
        checkpoint_path = output_dir / "best_model.pt"
        torch.save(
            {
                "model_state": best_state,
                "model_config": asdict(config),
                "feature_scaler": data.feature_scaler.to_dict(),
                "target_scaler": data.target_scaler.to_dict(),
                "feature_names": data.feature_names,
                "station_ids": data.station_ids,
                "args": vars(args),
            },
            checkpoint_path,
        )

        pred, true = collect_predictions(
            model,
            data,
            device=device,
            limit_batches=args.limit_test_batches,
        )
        metrics, horizon_rows = compute_metrics(pred, true)
        summary = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "window_counts": data.window_counts,
            "model_config": asdict(config),
            "metrics": metrics,
            "limits": {
                "train": args.limit_train_batches,
                "validation": args.limit_val_batches,
                "test": args.limit_test_batches,
            },
        }
        write_history(output_dir / "train_history.csv", history)
        write_horizon_metrics(output_dir / "test_horizon_metrics.csv", horizon_rows)
        (output_dir / "metrics_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {checkpoint_path}")
        print(f"Wrote {output_dir / 'metrics_summary.json'}")
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
