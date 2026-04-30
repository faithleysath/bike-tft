"""Train the GMRL adaptation on NYC Citi Bike dep/arr tensors."""

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
import torch.nn as nn

from forecasting_models.agcrn_nyc.train import compute_metrics, resolve_device, write_horizon_metrics

from .data import DEFAULT_BUNDLE, GMRLBundleData, make_gmrl_dataloaders
from .model import GMRL, GMRLConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC GMRL for dep+arr forecasting.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/gmrl_nyc_v1/runs/default")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--loss-type", choices=("mae", "mse"), default="mae")
    parser.add_argument("--feature-loss-weight", type=float, default=0.0)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-components", type=int, default=17)
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--disable-hra", action="store_true")
    parser.add_argument("--gmre-chunk-size", type=int, default=2048)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()
    if args.feature_loss_weight < 0:
        parser.error("--feature-loss-weight must be non-negative")
    if args.num_components <= 0 or args.hidden_channels <= 0:
        parser.error("--num-components and --hidden-channels must be positive")
    return args


def set_seed(seed: int) -> None:
    """Make stochastic training behavior reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_batches(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    limit: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield at most limit batches from a dataloader."""
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def prediction_loss(pred: torch.Tensor, true: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Compute model-space prediction loss."""
    if loss_type == "mae":
        return torch.mean(torch.abs(pred - true))
    if loss_type == "mse":
        return nn.functional.mse_loss(pred, true)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def train_epoch(
    model: GMRL,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    feature_loss_weight: float,
    limit_batches: int | None,
) -> tuple[float, float]:
    """Run one training epoch."""
    model.train()
    losses: list[float] = []
    feature_losses: list[float] = []
    for x, y, _raw_y in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred, feature_loss = model(x)
        pred_loss = prediction_loss(pred, y, loss_type)
        loss = pred_loss if feature_loss_weight == 0 else pred_loss + feature_loss_weight * feature_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(float(pred_loss.item()))
        feature_losses.append(float(feature_loss.item()))
    if not losses:
        raise RuntimeError("No training batches were processed")
    return float(np.mean(losses)), float(np.mean(feature_losses))


@torch.no_grad()
def evaluate_loss(
    model: GMRL,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    loss_type: str,
    limit_batches: int | None,
) -> float:
    """Evaluate model-space prediction loss."""
    model.eval()
    losses: list[float] = []
    for x, y, _raw_y in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred, _feature_loss = model(x)
        losses.append(float(prediction_loss(pred, y, loss_type).item()))
    if not losses:
        raise RuntimeError("No evaluation batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def collect_predictions(
    model: GMRL,
    data: GMRLBundleData,
    *,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect inverse-transformed predictions and raw labels from the test set."""
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, _y, raw_y in limited_batches(data.test_loader, limit_batches):
        x = x.to(device, non_blocking=True)
        pred, _feature_loss = model(x)
        preds.append(data.inverse_model_tensor(pred).squeeze(-1).cpu())
        labels.append(raw_y.cpu())
    if not preds:
        raise RuntimeError("No test batches were processed")
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write train history CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "feature_loss", "val_loss", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Train and evaluate GMRL."""
    try:
        args = parse_args()
        set_seed(args.seed)
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = make_gmrl_dataloaders(
            args.bundle,
            target_mode=args.target_mode,
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        config = GMRLConfig(
            num_nodes=data.num_nodes,
            num_sources=data.num_sources,
            input_dim=data.input_dim,
            output_dim=1,
            lag=args.lag,
            horizon=args.horizon,
            num_components=args.num_components,
            hidden_channels=args.hidden_channels,
            kernel_size=args.kernel_size,
            use_hra=not args.disable_hra,
            gmre_chunk_size=args.gmre_chunk_size,
        )
        model = GMRL(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            started_at = time.time()
            train_loss, feature_loss = train_epoch(
                model,
                data.train_loader,
                optimizer=optimizer,
                device=device,
                loss_type=args.loss_type,
                feature_loss_weight=args.feature_loss_weight,
                limit_batches=args.limit_train_batches,
            )
            val_loss = evaluate_loss(
                model,
                data.val_loader,
                device=device,
                loss_type=args.loss_type,
                limit_batches=args.limit_val_batches,
            )
            seconds = time.time() - started_at
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "feature_loss": feature_loss,
                    "val_loss": val_loss,
                    "seconds": seconds,
                }
            )
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"feature_loss={feature_loss:.6f} val_loss={val_loss:.6f} seconds={seconds:.2f}",
                flush=True,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} stale epoch(s).", flush=True)
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a best model state")

        model.load_state_dict(best_state)
        checkpoint_path = output_dir / "best_model.pt"
        torch.save(
            {
                "model_state": best_state,
                "model_config": asdict(config),
                "target_scaler": data.target_scaler.to_dict(),
                "target_mode": data.target_mode,
                "station_ids": data.station_ids,
                "args": vars(args),
            },
            checkpoint_path,
        )

        pred, true = collect_predictions(model, data, device=device, limit_batches=args.limit_test_batches)
        metrics, horizon_rows = compute_metrics(pred, true)
        summary = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "window_counts": data.window_counts,
            "model_config": asdict(config),
            "target_mode": data.target_mode,
            "loss": {
                "loss_type": args.loss_type,
                "feature_loss_weight": args.feature_loss_weight,
            },
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
        print(f"Wrote {checkpoint_path}", flush=True)
        print(f"Wrote {output_dir / 'metrics_summary.json'}", flush=True)
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
