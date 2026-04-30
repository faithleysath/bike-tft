"""Export Graph WaveNet test-split forecasts for rebalancing algorithms."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forecasting_models.agcrn_nyc.data import load_bundle_arrays, split_window_starts
from forecasting_models.agcrn_nyc.train import resolve_device
from .data import make_time_aware_dataloaders

from .model import GraphWaveNet, GraphWaveNetConfig, load_relation_graphs
from .train import DEFAULT_RELATION_GRAPHS


DEFAULT_CHECKPOINT = Path("forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/default/best_model.pt")
DEFAULT_OUTPUT = Path("forecasting_models/agcrn_nyc_gwnet_time_netloss_v1/runs/default/test_forecasts_for_rebalancing.parquet")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export Graph WaveNet forecasts for rebalancing.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--bundle", default="dataset/preprocessing/processed/nyc_top883/nyc_agcrn_bundle.npz")
    parser.add_argument("--relation-graphs", default=DEFAULT_RELATION_GRAPHS.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, relation_graphs: str | Path, station_ids: list[str], device: torch.device) -> GraphWaveNet:
    """Load a trained Graph WaveNet checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GraphWaveNetConfig(**checkpoint["model_config"])
    relation_supports, _metadata = load_relation_graphs(relation_graphs, station_ids)
    model = GraphWaveNet(config, relation_supports)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_test_predictions(model: GraphWaveNet, data, device: torch.device) -> np.ndarray:
    """Return inverse-transformed test predictions shaped [windows, horizon, nodes, 2]."""
    predictions: list[torch.Tensor] = []
    for x, future_time, _y, _raw_y, baseline in data.test_loader:
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        baseline = baseline.to(device, non_blocking=True)
        pred = model(x, future_time)
        predictions.append(data.inverse_model_tensor(pred, baseline).cpu())
    if not predictions:
        raise RuntimeError("No test batches were exported")
    return torch.cat(predictions, dim=0).numpy().astype(np.float32)


def build_forecast_frame(
    *,
    predictions: np.ndarray,
    timestamps: list[str],
    test_starts: np.ndarray,
    lag: int,
    horizon: int,
    node_count: int,
) -> pd.DataFrame:
    """Build a dense forecast table accepted by rebalancing forecast mode."""
    window_count = predictions.shape[0]
    if predictions.shape != (window_count, horizon, node_count, 2):
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
    if window_count != len(test_starts):
        raise ValueError(f"Prediction count {window_count} does not match test starts {len(test_starts)}")

    timestamp_index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    decision_indices = test_starts + lag - 1
    target_indices = test_starts[:, None] + lag + np.arange(horizon, dtype=np.int64)[None, :]
    row_count = window_count * horizon * node_count

    decision_ts = np.repeat(timestamp_index[decision_indices].to_numpy(dtype="datetime64[ns]"), horizon * node_count)
    target_ts = np.repeat(timestamp_index[target_indices.reshape(-1)].to_numpy(dtype="datetime64[ns]"), node_count)
    node_idx = np.tile(np.arange(node_count, dtype=np.int32), window_count * horizon)
    net_flow_pred = (predictions[..., 1] - predictions[..., 0]).reshape(row_count).astype(np.float32)

    return pd.DataFrame(
        {
            "decision_ts": decision_ts,
            "target_ts": target_ts,
            "node_idx": node_idx,
            "net_flow_pred": net_flow_pred,
        }
    )


def main() -> int:
    """CLI entrypoint."""
    try:
        args = parse_args()
        device = resolve_device(args.device)
        data = make_time_aware_dataloaders(
            args.bundle,
            target_mode="log1p",
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model = load_model(args.checkpoint, args.relation_graphs, data.station_ids, device)
        predictions = collect_test_predictions(model, data, device)
        arrays = load_bundle_arrays(args.bundle)
        _train_starts, _val_starts, test_starts = split_window_starts(
            time_count=len(arrays["timestamps"]),
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        forecast = build_forecast_frame(
            predictions=predictions,
            timestamps=[str(item) for item in arrays["timestamps"].tolist()],
            test_starts=test_starts,
            lag=args.lag,
            horizon=args.horizon,
            node_count=data.num_nodes,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_parquet(output_path, index=False)
        metadata = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "checkpoint": str(args.checkpoint),
            "bundle": str(args.bundle),
            "relation_graphs": str(args.relation_graphs),
            "output": str(output_path),
            "row_count": int(len(forecast)),
            "window_count": int(predictions.shape[0]),
            "horizon": int(args.horizon),
            "node_count": int(data.num_nodes),
            "columns": list(forecast.columns),
            "decision_start": str(forecast["decision_ts"].iloc[0]),
            "decision_end": str(forecast["decision_ts"].iloc[-1]),
            "target_start": str(forecast["target_ts"].iloc[0]),
            "target_end": str(forecast["target_ts"].iloc[-1]),
        }
        metadata_path = output_path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
