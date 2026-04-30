"""Train relational AGCRN on the NYC Citi Bike bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

from forecasting_models.agcrn_nyc.data import BundleData, make_dataloaders
from forecasting_models.agcrn_nyc.train import (
    collect_predictions,
    compute_metrics,
    evaluate_loss,
    resolve_device,
    set_seed,
    train_epoch,
    write_history,
    write_horizon_metrics,
)

from .model import RelationalAGCRN, RelationalAGCRNConfig, load_relation_graphs


DEFAULT_BUNDLE = Path("dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz")
DEFAULT_RELATION_GRAPHS = Path("dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC relational AGCRN for dep+arr forecasting.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--relation-graphs", default=DEFAULT_RELATION_GRAPHS.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/agcrn_nyc_relational_v1/runs/default")
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
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--support-count", type=int, default=2)
    parser.add_argument("--relation-mode", choices=("fused", "separate"), default="fused")
    parser.add_argument("--adaptive-init-weight", type=float, default=0.70)
    parser.add_argument("--od-forward-init-weight", type=float, default=0.15)
    parser.add_argument("--od-reverse-init-weight", type=float, default=0.15)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()
    if args.relation_mode == "fused" and args.support_count < 2:
        parser.error("--support-count must be at least 2 for fused mode")
    if args.relation_mode == "separate" and args.support_count < 4:
        parser.error("--support-count must be at least 4 for separate mode")
    init_sum = args.adaptive_init_weight + args.od_forward_init_weight + args.od_reverse_init_weight
    if args.adaptive_init_weight <= 0 or args.od_forward_init_weight <= 0 or args.od_reverse_init_weight <= 0:
        parser.error("relation initial weights must be positive")
    if init_sum <= 0:
        parser.error("relation initial weights must sum to a positive value")
    return args


def save_checkpoint(
    path: Path,
    *,
    model: RelationalAGCRN,
    best_state: dict[str, torch.Tensor],
    config: RelationalAGCRNConfig,
    relation_metadata: dict[str, object],
    data: BundleData,
    args: argparse.Namespace,
) -> None:
    """Write the best checkpoint with scalers and relation metadata."""
    torch.save(
        {
            "model_state": best_state,
            "model_config": asdict(config),
            "relation_metadata": relation_metadata,
            "relation_weights": model.relation_weight_dict(),
            "feature_scaler": data.feature_scaler.to_dict(),
            "target_scaler": data.target_scaler.to_dict(),
            "feature_names": data.feature_names,
            "station_ids": data.station_ids,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    """Train and evaluate the relational NYC AGCRN model."""
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
        save_checkpoint(
            checkpoint_path,
            model=model,
            best_state=best_state,
            config=config,
            relation_metadata=relation_metadata,
            data=data,
            args=args,
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
