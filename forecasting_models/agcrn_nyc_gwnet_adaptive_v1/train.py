"""Train adaptive-only Graph WaveNet on NYC Citi Bike log1p targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from forecasting_models.agcrn_nyc.train import compute_metrics, resolve_device, write_horizon_metrics
from forecasting_models.agcrn_nyc_objective_v1.data import DEFAULT_BUNDLE, make_objective_dataloaders
from forecasting_models.agcrn_nyc_gwnet_v1.train import (
    collect_predictions,
    evaluate_loss,
    set_seed,
    train_epoch,
    write_history,
)

from .model import AdaptiveGraphWaveNet, AdaptiveGraphWaveNetConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train adaptive-only NYC Graph WaveNet for dep+arr forecasting.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/agcrn_nyc_gwnet_adaptive_v1/runs/default")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--residual-channels", type=int, default=32)
    parser.add_argument("--dilation-channels", type=int, default=32)
    parser.add_argument("--skip-channels", type=int, default=128)
    parser.add_argument("--end-channels", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=2)
    parser.add_argument("--graph-order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()
    if args.blocks <= 0 or args.layers <= 0:
        parser.error("--blocks and --layers must be positive")
    if args.kernel_size <= 1:
        parser.error("--kernel-size must be greater than 1")
    if args.graph_order <= 0:
        parser.error("--graph-order must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    return args


def main() -> int:
    """Train and evaluate adaptive-only Graph WaveNet."""
    try:
        args = parse_args()
        set_seed(args.seed)
        device = resolve_device(args.device)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = make_objective_dataloaders(
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
        config = AdaptiveGraphWaveNetConfig(
            num_nodes=data.num_nodes,
            input_dim=data.input_dim,
            output_dim=2,
            horizon=args.horizon,
            embed_dim=args.embed_dim,
            residual_channels=args.residual_channels,
            dilation_channels=args.dilation_channels,
            skip_channels=args.skip_channels,
            end_channels=args.end_channels,
            blocks=args.blocks,
            layers=args.layers,
            kernel_size=args.kernel_size,
            graph_order=args.graph_order,
            dropout=args.dropout,
        )
        model = AdaptiveGraphWaveNet(config).to(device)
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
                limit_batches=args.limit_train_batches,
            )
            val_loss = evaluate_loss(
                model,
                data.val_loader,
                device=device,
                limit_batches=args.limit_val_batches,
            )
            seconds = time.time() - started_at
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": seconds})
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
                "graph": "adaptive_only",
                "feature_scaler": data.feature_scaler.to_dict(),
                "target_scaler": data.target_scaler.to_dict(),
                "target_mode": data.target_mode,
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
            "graph": "adaptive_only",
            "target_mode": data.target_mode,
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

