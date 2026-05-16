"""Train combined-target ablations for the NYC relational AGCRN model."""

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
from forecasting_models.agcrn_nyc_relational_v1.model import (
    RelationalAGCRN,
    RelationalAGCRNConfig,
    load_relation_graphs,
)

from .data import DEFAULT_BUNDLE, TARGET_MODES, ObjectiveBundleData, make_objective_dataloaders


DEFAULT_RELATION_GRAPHS = Path("dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k20.npz")
LOSS_TYPES = ("mae", "huber", "weighted_mae")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC relational AGCRN combined objective ablations.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--relation-graphs", default=DEFAULT_RELATION_GRAPHS.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/agcrn_nyc_objective_v2/runs/default")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="raw")
    parser.add_argument("--loss-type", choices=LOSS_TYPES, default="mae")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--nonzero-weight", type=float, default=2.0)
    parser.add_argument("--seasonal-lag", type=int, default=168)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--support-count", type=int, default=2)
    parser.add_argument("--relation-mode", choices=("fused", "separate"), default="fused")
    parser.add_argument("--adaptive-init-weight", type=float, default=0.95)
    parser.add_argument("--od-forward-init-weight", type=float, default=0.025)
    parser.add_argument("--od-reverse-init-weight", type=float, default=0.025)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()
    if args.huber_beta <= 0:
        parser.error("--huber-beta must be positive")
    if args.nonzero_weight <= 0:
        parser.error("--nonzero-weight must be positive")
    if args.seasonal_lag <= 0:
        parser.error("--seasonal-lag must be positive")
    if args.relation_mode == "fused" and args.support_count < 2:
        parser.error("--support-count must be at least 2 for fused mode")
    if args.relation_mode == "separate" and args.support_count < 4:
        parser.error("--support-count must be at least 4 for separate mode")
    if args.adaptive_init_weight <= 0 or args.od_forward_init_weight <= 0 or args.od_reverse_init_weight <= 0:
        parser.error("relation initial weights must be positive")
    return args


def set_seed(seed: int) -> None:
    """Make stochastic training behavior reproducible where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_batches(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    limit: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield at most limit batches from a dataloader."""
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def objective_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    raw_target: torch.Tensor,
    *,
    loss_type: str,
    huber_beta: float,
    nonzero_weight: float,
) -> torch.Tensor:
    """Compute model-space objective loss."""
    if loss_type == "mae":
        return torch.mean(torch.abs(pred - target))
    if loss_type == "huber":
        return torch.nn.functional.smooth_l1_loss(pred, target, beta=huber_beta, reduction="mean")
    if loss_type == "weighted_mae":
        error = torch.abs(pred - target)
        weights = torch.where(raw_target > 0, torch.full_like(error, nonzero_weight), torch.ones_like(error))
        return torch.sum(error * weights) / torch.clamp_min(torch.sum(weights), 1.0)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def train_epoch(
    model: RelationalAGCRN,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    huber_beta: float,
    nonzero_weight: float,
    limit_batches: int | None,
) -> float:
    """Run one training epoch."""
    model.train()
    losses: list[float] = []
    for x, y, raw_y, _baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw_y = raw_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = objective_loss(
            pred,
            y,
            raw_y,
            loss_type=loss_type,
            huber_beta=huber_beta,
            nonzero_weight=nonzero_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("No training batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: RelationalAGCRN,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    loss_type: str,
    huber_beta: float,
    nonzero_weight: float,
    limit_batches: int | None,
) -> float:
    """Evaluate model-space objective loss."""
    model.eval()
    losses: list[float] = []
    for x, y, raw_y, _baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw_y = raw_y.to(device, non_blocking=True)
        pred = model(x)
        loss = objective_loss(
            pred,
            y,
            raw_y,
            loss_type=loss_type,
            huber_beta=huber_beta,
            nonzero_weight=nonzero_weight,
        )
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("No evaluation batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def collect_predictions(
    model: RelationalAGCRN,
    data: ObjectiveBundleData,
    *,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect inverse-transformed predictions and raw labels from the test set."""
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, _y, raw_y, baseline in limited_batches(data.test_loader, limit_batches):
        x = x.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        pred = model(x)
        preds.append(data.inverse_model_tensor(pred, baseline).cpu())
        labels.append(raw_y.cpu())
    if not preds:
        raise RuntimeError("No test batches were processed")
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write train history CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "seconds"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Train and evaluate the objective ablation."""
    try:
        args = parse_args()
        set_seed(args.seed)
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = make_objective_dataloaders(
            args.bundle,
            target_mode=args.target_mode,
            seasonal_lag=args.seasonal_lag,
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        relation_supports, relation_metadata = load_relation_graphs(args.relation_graphs, data.station_ids)
        config = RelationalAGCRNConfig(
            num_nodes=data.num_nodes,
            input_dim=data.input_dim,
            output_dim=2,
            horizon=args.horizon,
            embed_dim=args.embed_dim,
            rnn_units=args.rnn_units,
            num_layers=args.num_layers,
            support_count=args.support_count,
            relation_mode=args.relation_mode,
            adaptive_init_weight=args.adaptive_init_weight,
            od_forward_init_weight=args.od_forward_init_weight,
            od_reverse_init_weight=args.od_reverse_init_weight,
        )
        model = RelationalAGCRN(config, relation_supports).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

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
                loss_type=args.loss_type,
                huber_beta=args.huber_beta,
                nonzero_weight=args.nonzero_weight,
                limit_batches=args.limit_train_batches,
            )
            val_loss = evaluate_loss(
                model,
                data.val_loader,
                device=device,
                loss_type=args.loss_type,
                huber_beta=args.huber_beta,
                nonzero_weight=args.nonzero_weight,
                limit_batches=args.limit_val_batches,
            )
            seconds = time.time() - started_at
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": seconds})
            weights = model.relation_weight_dict()
            print(
                "epoch="
                f"{epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} seconds={seconds:.2f} "
                f"weights=adaptive:{weights['adaptive']:.3f},od_fwd:{weights['od_forward']:.3f},od_rev:{weights['od_reverse']:.3f}"
            )
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
                "relation_metadata": relation_metadata,
                "relation_weights": model.relation_weight_dict(),
                "feature_scaler": data.feature_scaler.to_dict(),
                "target_scaler": data.target_scaler.to_dict(),
                "target_mode": data.target_mode,
                "seasonal_lag": data.seasonal_lag,
                "feature_names": data.feature_names,
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
            "relation_metadata": relation_metadata,
            "relation_weights": model.relation_weight_dict(),
            "target_mode": data.target_mode,
            "seasonal_lag": data.seasonal_lag,
            "loss": {
                "loss_type": args.loss_type,
                "huber_beta": args.huber_beta,
                "nonzero_weight": args.nonzero_weight,
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
        print(f"Wrote {checkpoint_path}")
        print(f"Wrote {output_dir / 'metrics_summary.json'}")
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
