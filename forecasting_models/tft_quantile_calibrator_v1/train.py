"""Train a TFT-style quantile calibrator on NYC Citi Bike data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from forecasting_models.agcrn_nyc.train import compute_metrics, resolve_device, write_horizon_metrics
from forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.data import DEFAULT_BUNDLE, TimeAwareObjectiveBundleData, make_time_aware_dataloaders

from .model import TFTQuantileConfig, TFTQuantileModel


QUANTILES = (0.1, 0.5, 0.9)
TARGET_NAMES = ("dep", "arr")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a TFT-style NYC quantile calibrator.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/tft_quantile_calibrator_v1/runs/default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--station-embed-dim", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()
    if args.hidden_dim <= 0 or args.station_embed_dim <= 0:
        parser.error("--hidden-dim and --station-embed-dim must be positive")
    if args.hidden_dim % args.attention_heads != 0:
        parser.error("--hidden-dim must be divisible by --attention-heads")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    return args


def set_seed(seed: int) -> None:
    """Make stochastic training behavior reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_batches[T](loader: Iterable[T], limit: int | None) -> Iterable[T]:
    """Yield at most limit batches."""
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...] = QUANTILES) -> torch.Tensor:
    """Compute mean pinball loss for sorted quantile predictions."""
    losses = []
    for index, quantile in enumerate(quantiles):
        error = target - pred[..., index]
        losses.append(torch.maximum((quantile - 1.0) * error, quantile * error))
    return torch.mean(torch.stack(losses, dim=-1))


def inverse_quantiles(values: torch.Tensor, data: TimeAwareObjectiveBundleData) -> torch.Tensor:
    """Convert model-space quantiles to original count scale."""
    mean = torch.as_tensor(data.target_scaler.mean, dtype=values.dtype, device=values.device).unsqueeze(-1)
    std = torch.as_tensor(data.target_scaler.std, dtype=values.dtype, device=values.device).unsqueeze(-1)
    unscaled = values * std + mean
    if data.target_mode == "raw":
        return unscaled.clamp_min(0)
    if data.target_mode == "log1p":
        return torch.expm1(unscaled).clamp_min(0)
    raise ValueError(f"Unsupported target_mode for quantile model: {data.target_mode}")


def train_epoch(
    model: TFTQuantileModel,
    loader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None,
) -> float:
    """Run one training epoch."""
    model.train()
    losses: list[float] = []
    for x, future_time, y, _raw_y, _baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, future_time)
        loss = pinball_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("No training batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: TFTQuantileModel,
    loader,
    *,
    device: torch.device,
    limit_batches: int | None,
) -> float:
    """Evaluate model-space pinball loss."""
    model.eval()
    losses: list[float] = []
    for x, future_time, y, _raw_y, _baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        losses.append(float(pinball_loss(model(x, future_time), y).item()))
    if not losses:
        raise RuntimeError("No evaluation batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def collect_quantiles(
    model: TFTQuantileModel,
    data: TimeAwareObjectiveBundleData,
    *,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect inverse-transformed quantiles, raw labels, model-space quantiles, and model-space labels."""
    model.eval()
    quantiles: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model_space_quantiles: list[torch.Tensor] = []
    model_space_labels: list[torch.Tensor] = []
    for x, future_time, y, raw_y, _baseline in limited_batches(data.test_loader, limit_batches):
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        pred = model(x, future_time)
        quantiles.append(inverse_quantiles(pred, data).cpu())
        model_space_quantiles.append(pred.cpu())
        model_space_labels.append(y.cpu())
        labels.append(raw_y.cpu())
    if not quantiles:
        raise RuntimeError("No test batches were processed")
    return (
        torch.cat(quantiles, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(model_space_quantiles, dim=0),
        torch.cat(model_space_labels, dim=0),
    )


def interval_metrics(quantiles: torch.Tensor, true: torch.Tensor) -> dict[str, object]:
    """Compute PICP and interval width for q10-q90."""
    q10 = quantiles[..., 0]
    q90 = quantiles[..., 2]
    covered = (true >= q10) & (true <= q90)
    width = q90 - q10
    target_metrics = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        target_metrics[target_name] = {
            "picp_80": float(covered[..., target_index].float().mean().item()),
            "interval_width": float(width[..., target_index].mean().item()),
        }
    target_metrics["average"] = {
        "picp_80": float(covered.float().mean().item()),
        "interval_width": float(width.mean().item()),
    }
    return target_metrics


def quantile_horizon_rows(quantiles: torch.Tensor, true: torch.Tensor) -> list[dict[str, float | int | str]]:
    """Build horizon-level q50 and interval metrics."""
    rows: list[dict[str, float | int | str]] = []
    q50 = quantiles[..., 1]
    q10 = quantiles[..., 0]
    q90 = quantiles[..., 2]
    for horizon_index in range(quantiles.shape[1]):
        for target_index, target_name in enumerate(TARGET_NAMES):
            pred = q50[:, horizon_index, :, target_index]
            label = true[:, horizon_index, :, target_index]
            error = pred - label
            mask = torch.abs(label) > 1e-6
            mape = torch.mean(torch.abs(error[mask] / label[mask])).item() if bool(mask.any()) else float("nan")
            covered = (label >= q10[:, horizon_index, :, target_index]) & (label <= q90[:, horizon_index, :, target_index])
            rows.append(
                {
                    "horizon": horizon_index + 1,
                    "target": target_name,
                    "q50_mae": float(torch.mean(torch.abs(error)).item()),
                    "q50_rmse": float(torch.sqrt(torch.mean(error * error)).item()),
                    "q50_mape": float(mape),
                    "picp_80": float(covered.float().mean().item()),
                    "interval_width": float((q90[:, horizon_index, :, target_index] - q10[:, horizon_index, :, target_index]).mean().item()),
                }
            )
    return rows


def write_quantile_horizon_metrics(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write quantile horizon metrics CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["horizon", "target", "q50_mae", "q50_rmse", "q50_mape", "picp_80", "interval_width"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write train history CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_pinball_loss", "val_pinball_loss", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Train and evaluate the quantile model."""
    try:
        args = parse_args()
        set_seed(args.seed)
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = make_time_aware_dataloaders(
            args.bundle,
            target_mode="log1p",
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        config = TFTQuantileConfig(
            num_nodes=data.num_nodes,
            input_dim=data.input_dim,
            future_time_dim=data.future_time_dim,
            output_dim=2,
            quantile_count=len(QUANTILES),
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            station_embed_dim=args.station_embed_dim,
            attention_heads=args.attention_heads,
            dropout=args.dropout,
        )
        model = TFTQuantileModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
                device=device,
                limit_batches=args.limit_train_batches,
            )
            val_loss = evaluate_loss(
                model,
                data.val_loader,
                device=device,
                limit_batches=args.limit_val_batches,
            )
            seconds = time.time() - started_at
            history.append(
                {
                    "epoch": epoch,
                    "train_pinball_loss": train_loss,
                    "val_pinball_loss": val_loss,
                    "seconds": seconds,
                }
            )
            print(f"epoch={epoch} train_pinball={train_loss:.6f} val_pinball={val_loss:.6f} seconds={seconds:.2f}")
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
                "target_mode": data.target_mode,
                "future_time_feature_names": data.future_time_feature_names,
                "feature_names": data.feature_names,
                "station_ids": data.station_ids,
                "quantiles": list(QUANTILES),
                "args": vars(args),
            },
            checkpoint_path,
        )

        quantiles, true, model_space_quantiles, model_space_true = collect_quantiles(
            model,
            data,
            device=device,
            limit_batches=args.limit_test_batches,
        )
        q50 = quantiles[..., 1]
        q50_metrics, q50_horizon_rows = compute_metrics(q50, true)
        interval = interval_metrics(quantiles, true)
        quantile_rows = quantile_horizon_rows(quantiles, true)
        summary = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "best_epoch": best_epoch,
            "best_val_pinball_loss": best_val_loss,
            "window_counts": data.window_counts,
            "model_config": asdict(config),
            "quantiles": list(QUANTILES),
            "target_mode": data.target_mode,
            "q50_metrics": q50_metrics,
            "interval_metrics": interval,
            "test_model_space_pinball_loss": float(pinball_loss(model_space_quantiles, model_space_true).item()),
            "limits": {
                "train": args.limit_train_batches,
                "validation": args.limit_val_batches,
                "test": args.limit_test_batches,
            },
        }
        write_history(output_dir / "train_history.csv", history)
        write_horizon_metrics(output_dir / "test_q50_horizon_metrics.csv", q50_horizon_rows)
        write_quantile_horizon_metrics(output_dir / "test_quantile_horizon_metrics.csv", quantile_rows)
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
