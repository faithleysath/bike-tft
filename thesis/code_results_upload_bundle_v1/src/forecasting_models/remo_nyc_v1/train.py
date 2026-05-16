"""Train the ReMo-style NYC hypergraph relational model."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from forecasting_models.agcrn_nyc.train import resolve_device
from forecasting_models.agcrn_nyc_objective_v1.data import make_objective_dataloaders
from forecasting_models.paper_repro_common import (
    DEFAULT_BUNDLE,
    collect_objective_predictions,
    limited_batches,
    set_seed,
    write_history,
    write_json,
    write_metrics,
)

from .model import ReMoConfig, ReMoNYC


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC ReMo for dep+arr forecasting.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/remo_nyc_v1/runs/default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=("raw", "log1p", "seasonal_residual"), default="log1p")
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--node-embed-dim", type=int, default=16)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--num-hyperedges", type=int, default=16)
    parser.add_argument("--num-relation-types", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def train_epoch(model: ReMoNYC, loader, *, optimizer, device, limit_batches: int | None) -> float:
    """Run one ReMo training epoch."""
    model.train()
    losses: list[float] = []
    for x, y, _raw_y, _baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.mean(torch.abs(pred - y))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("No training batches were processed")
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(model: ReMoNYC, loader, *, device, limit_batches: int | None) -> float:
    """Evaluate model-space MAE."""
    model.eval()
    losses: list[float] = []
    for x, y, _raw_y, _baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        losses.append(float(torch.mean(torch.abs(pred - y)).item()))
    if not losses:
        raise RuntimeError("No evaluation batches were processed")
    return float(np.mean(losses))


def call_model(model: ReMoNYC, x: torch.Tensor) -> torch.Tensor:
    """Inference wrapper for common prediction collection."""
    return model(x)


def main() -> int:
    """Train and evaluate ReMo."""
    try:
        args = parse_args()
        set_seed(args.seed)
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = make_objective_dataloaders(
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
        config = ReMoConfig(
            num_nodes=data.num_nodes,
            input_dim=data.input_dim,
            output_dim=2,
            horizon=args.horizon,
            hidden_dim=args.hidden_dim,
            node_embed_dim=args.node_embed_dim,
            num_views=args.num_views,
            num_hyperedges=args.num_hyperedges,
            num_relation_types=args.num_relation_types,
            dropout=args.dropout,
        )
        model = ReMoNYC(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            started_at = time.time()
            train_loss = train_epoch(model, data.train_loader, optimizer=optimizer, device=device, limit_batches=args.limit_train_batches)
            val_loss = evaluate_loss(model, data.val_loader, device=device, limit_batches=args.limit_val_batches)
            seconds = time.time() - started_at
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": seconds})
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} seconds={seconds:.2f}", flush=True)
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
        pred, true = collect_objective_predictions(model, data, device=device, limit_batches=args.limit_test_batches, call_model=call_model)
        metrics = write_metrics(output_dir, pred, true)
        checkpoint_path = output_dir / "best_model.pt"
        torch.save({"model_state": best_state, "model_config": asdict(config), "args": vars(args), "station_ids": data.station_ids}, checkpoint_path)
        write_history(output_dir / "train_history.csv", history)
        write_json(
            output_dir / "metrics_summary.json",
            {
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "paper": "ReMo, IJCAI 2023",
                "source": "paper implementation; no official public repository found on IJCAI page",
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "window_counts": data.window_counts,
                "model_config": asdict(config),
                "target_mode": data.target_mode,
                "metrics": metrics,
                "limits": {"train": args.limit_train_batches, "validation": args.limit_val_batches, "test": args.limit_test_batches},
            },
        )
        print(f"Wrote {checkpoint_path}", flush=True)
        print(f"Wrote {output_dir / 'metrics_summary.json'}", flush=True)
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
