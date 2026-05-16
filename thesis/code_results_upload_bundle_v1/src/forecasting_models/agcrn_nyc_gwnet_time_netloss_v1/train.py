"""Train time-aware Graph WaveNet with an auxiliary net-flow objective."""

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

from forecasting_models.agcrn_nyc.train import compute_metrics, metric_payload, resolve_device, write_horizon_metrics
from .data import DEFAULT_BUNDLE, TimeAwareObjectiveBundleData, make_time_aware_dataloaders

from .model import GraphWaveNet, GraphWaveNetConfig, load_relation_graphs


DEFAULT_RELATION_GRAPHS = Path("dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k20.npz")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train NYC time-aware Graph WaveNet for dep+arr forecasting.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE.as_posix())
    parser.add_argument("--relation-graphs", default=DEFAULT_RELATION_GRAPHS.as_posix())
    parser.add_argument("--output-dir", default="forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/default")
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
    parser.add_argument("--future-time-embed-dim", type=int, default=16)
    parser.add_argument(
        "--net-loss-weight",
        type=float,
        default=0.10,
        help="Weight for count-space net-flow MAE auxiliary loss.",
    )
    parser.add_argument(
        "--net-loss-count-cap",
        type=float,
        default=200.0,
        help="Per-target count cap used only inside the auxiliary net-flow loss for numerical stability.",
    )
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
    if args.blocks <= 0 or args.layers <= 0:
        parser.error("--blocks and --layers must be positive")
    if args.kernel_size <= 1:
        parser.error("--kernel-size must be greater than 1")
    if args.graph_order <= 0:
        parser.error("--graph-order must be positive")
    if args.future_time_embed_dim <= 0:
        parser.error("--future-time-embed-dim must be positive")
    if args.net_loss_weight < 0:
        parser.error("--net-loss-weight must be non-negative")
    if args.net_loss_count_cap <= 0:
        parser.error("--net-loss-count-cap must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
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
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    limit: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield at most limit batches from a dataloader."""
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def compute_training_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    raw_y: torch.Tensor,
    baseline: torch.Tensor,
    data: TimeAwareObjectiveBundleData,
    *,
    net_loss_weight: float,
    net_loss_count_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute model-space dep/arr MAE plus count-space net-flow auxiliary loss."""
    base_loss = torch.mean(torch.abs(pred - y))
    if net_loss_weight == 0:
        zero = base_loss.new_tensor(0.0)
        return base_loss, base_loss.detach(), zero
    pred_counts = inverse_model_tensor_for_loss(pred, baseline, data, count_cap=net_loss_count_cap)
    net_pred = pred_counts[..., 1] - pred_counts[..., 0]
    net_true = raw_y.to(device=pred.device, dtype=pred.dtype)[..., 1] - raw_y.to(device=pred.device, dtype=pred.dtype)[..., 0]
    net_loss = torch.mean(torch.abs(net_pred - net_true)) / data.net_loss_scale
    return base_loss + net_loss_weight * net_loss, base_loss.detach(), net_loss.detach()


def inverse_model_tensor_for_loss(
    values: torch.Tensor,
    baselines: torch.Tensor,
    data: TimeAwareObjectiveBundleData,
    *,
    count_cap: float,
) -> torch.Tensor:
    """Convert model-space predictions to bounded count space for auxiliary loss."""
    unscaled = data.target_scaler.inverse_transform_tensor(values)
    if data.target_mode == "raw":
        counts = unscaled
    elif data.target_mode == "log1p":
        counts = torch.expm1(torch.clamp(unscaled, min=0.0, max=float(np.log1p(count_cap))))
    elif data.target_mode == "seasonal_residual":
        counts = unscaled + baselines.to(device=values.device, dtype=values.dtype)
    else:
        raise ValueError(f"Unsupported target_mode: {data.target_mode}")
    return torch.clamp(counts, min=0.0, max=count_cap)


def train_epoch(
    model: GraphWaveNet,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    data: TimeAwareObjectiveBundleData,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    net_loss_weight: float,
    net_loss_count_cap: float,
    limit_batches: int | None,
) -> dict[str, float]:
    """Run one training epoch using model-space MAE and net-flow auxiliary loss."""
    model.train()
    losses: list[float] = []
    base_losses: list[float] = []
    net_losses: list[float] = []
    for x, future_time, y, raw_y, baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw_y = raw_y.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, future_time)
        loss, base_loss, net_loss = compute_training_loss(
            pred,
            y,
            raw_y,
            baseline,
            data,
            net_loss_weight=net_loss_weight,
            net_loss_count_cap=net_loss_count_cap,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
        base_losses.append(float(base_loss.item()))
        net_losses.append(float(net_loss.item()))
    if not losses:
        raise RuntimeError("No training batches were processed")
    return {
        "loss": float(np.mean(losses)),
        "base_loss": float(np.mean(base_losses)),
        "net_loss": float(np.mean(net_losses)),
    }


@torch.no_grad()
def evaluate_loss(
    model: GraphWaveNet,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    data: TimeAwareObjectiveBundleData,
    device: torch.device,
    net_loss_weight: float,
    net_loss_count_cap: float,
    limit_batches: int | None,
) -> dict[str, float]:
    """Evaluate model-space MAE and net-flow auxiliary loss."""
    model.eval()
    losses: list[float] = []
    base_losses: list[float] = []
    net_losses: list[float] = []
    for x, future_time, y, raw_y, baseline in limited_batches(loader, limit_batches):
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw_y = raw_y.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        pred = model(x, future_time)
        loss, base_loss, net_loss = compute_training_loss(
            pred,
            y,
            raw_y,
            baseline,
            data,
            net_loss_weight=net_loss_weight,
            net_loss_count_cap=net_loss_count_cap,
        )
        losses.append(float(loss.item()))
        base_losses.append(float(base_loss.item()))
        net_losses.append(float(net_loss.item()))
    if not losses:
        raise RuntimeError("No evaluation batches were processed")
    return {
        "loss": float(np.mean(losses)),
        "base_loss": float(np.mean(base_losses)),
        "net_loss": float(np.mean(net_losses)),
    }


@torch.no_grad()
def collect_predictions(
    model: GraphWaveNet,
    data: TimeAwareObjectiveBundleData,
    *,
    device: torch.device,
    limit_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect inverse-transformed predictions and raw labels from the test set."""
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for x, future_time, _y, raw_y, baseline in limited_batches(data.test_loader, limit_batches):
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        pred = model(x, future_time)
        preds.append(data.inverse_model_tensor(pred, baseline).cpu())
        labels.append(raw_y.cpu())
    if not preds:
        raise RuntimeError("No test batches were processed")
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write train history CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_base_loss",
                "train_net_loss",
                "val_loss",
                "val_base_loss",
                "val_net_loss",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_net_flow_metrics(pred: torch.Tensor, true: torch.Tensor) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Compute aggregate and horizon metrics for net_flow = arr - dep."""
    pred_net = pred[..., 1] - pred[..., 0]
    true_net = true[..., 1] - true[..., 0]
    rows: list[dict[str, float | int | str]] = []
    for horizon_index in range(pred.shape[1]):
        metrics = metric_payload(pred_net[:, horizon_index], true_net[:, horizon_index])
        rows.append({"horizon": horizon_index + 1, "target": "net_flow", **metrics})
    return metric_payload(pred_net, true_net), rows


def main() -> int:
    """Train and evaluate Graph WaveNet."""
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
        relation_supports, relation_metadata = load_relation_graphs(args.relation_graphs, data.station_ids)
        config = GraphWaveNetConfig(
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
            future_time_dim=data.future_time_dim,
            future_time_embed_dim=args.future_time_embed_dim,
            adaptive_init_weight=args.adaptive_init_weight,
            od_forward_init_weight=args.od_forward_init_weight,
            od_reverse_init_weight=args.od_reverse_init_weight,
        )
        model = GraphWaveNet(config, relation_supports).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, args.epochs + 1):
            started_at = time.time()
            train_stats = train_epoch(
                model,
                data.train_loader,
                data=data,
                optimizer=optimizer,
                device=device,
                net_loss_weight=args.net_loss_weight,
                net_loss_count_cap=args.net_loss_count_cap,
                limit_batches=args.limit_train_batches,
            )
            val_stats = evaluate_loss(
                model,
                data.val_loader,
                data=data,
                device=device,
                net_loss_weight=args.net_loss_weight,
                net_loss_count_cap=args.net_loss_count_cap,
                limit_batches=args.limit_val_batches,
            )
            seconds = time.time() - started_at
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "train_base_loss": train_stats["base_loss"],
                    "train_net_loss": train_stats["net_loss"],
                    "val_loss": val_stats["loss"],
                    "val_base_loss": val_stats["base_loss"],
                    "val_net_loss": val_stats["net_loss"],
                    "seconds": seconds,
                }
            )
            weights = model.relation_weight_dict()
            print(
                "epoch="
                f"{epoch} train_loss={train_stats['loss']:.6f} val_loss={val_stats['loss']:.6f} "
                f"train_net={train_stats['net_loss']:.6f} val_net={val_stats['net_loss']:.6f} seconds={seconds:.2f} "
                f"weights=adaptive:{weights['adaptive']:.3f},od_fwd:{weights['od_forward']:.3f},od_rev:{weights['od_reverse']:.3f}"
            )
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
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
                "future_time_feature_names": data.future_time_feature_names,
                "net_loss_scale": data.net_loss_scale,
                "net_loss_weight": args.net_loss_weight,
                "net_loss_count_cap": args.net_loss_count_cap,
                "feature_names": data.feature_names,
                "station_ids": data.station_ids,
                "args": vars(args),
            },
            checkpoint_path,
        )

        pred, true = collect_predictions(model, data, device=device, limit_batches=args.limit_test_batches)
        metrics, horizon_rows = compute_metrics(pred, true)
        net_flow_metrics, net_flow_horizon_rows = compute_net_flow_metrics(pred, true)
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
            "future_time_feature_names": data.future_time_feature_names,
            "net_loss_scale": data.net_loss_scale,
            "net_loss_weight": args.net_loss_weight,
            "net_loss_count_cap": args.net_loss_count_cap,
            "metrics": metrics,
            "net_flow_metrics": net_flow_metrics,
            "limits": {
                "train": args.limit_train_batches,
                "validation": args.limit_val_batches,
                "test": args.limit_test_batches,
            },
        }
        write_history(output_dir / "train_history.csv", history)
        write_horizon_metrics(output_dir / "test_horizon_metrics.csv", horizon_rows)
        write_horizon_metrics(output_dir / "test_net_flow_horizon_metrics.csv", net_flow_horizon_rows)
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
