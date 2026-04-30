"""Export TFT-style quantile forecasts for risk-aware rebalancing."""

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
from forecasting_models.agcrn_nyc_gwnet_time_netloss_v1.data import make_time_aware_dataloaders
from forecasting_models.tft_quantile_calibrator_v1.model import TFTQuantileConfig, TFTQuantileModel
from forecasting_models.tft_quantile_calibrator_v1.train import inverse_quantiles


DEFAULT_CHECKPOINT = Path("forecasting_models/tft_quantile_calibrator_v1/runs/default/best_model.pt")
DEFAULT_OUTPUT = Path("forecasting_models/tft_quantile_calibrator_v1/runs/default/test_quantile_forecasts_for_rebalancing.parquet")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export TFT-style quantile forecasts for rebalancing.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--bundle", default="dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_agcrn_bundle.npz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, device: torch.device) -> TFTQuantileModel:
    """Load a trained quantile model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TFTQuantileConfig(**checkpoint["model_config"])
    model = TFTQuantileModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_test_quantiles(model: TFTQuantileModel, data, device: torch.device, limit_test_batches: int | None) -> np.ndarray:
    """Return inverse-transformed test quantiles shaped [windows, horizon, nodes, 2, 3]."""
    predictions: list[torch.Tensor] = []
    for batch_index, (x, future_time, _y, _raw_y, _baseline) in enumerate(data.test_loader):
        if limit_test_batches is not None and batch_index >= limit_test_batches:
            break
        x = x.to(device, non_blocking=True)
        future_time = future_time.to(device, non_blocking=True)
        pred = model(x, future_time)
        predictions.append(inverse_quantiles(pred, data).cpu())
    if not predictions:
        raise RuntimeError("No test batches were exported")
    return torch.cat(predictions, dim=0).numpy().astype(np.float32)


def build_forecast_frame(
    *,
    quantiles: np.ndarray,
    timestamps: list[str],
    test_starts: np.ndarray,
    lag: int,
    horizon: int,
    node_count: int,
) -> pd.DataFrame:
    """Build a dense quantile forecast table accepted by rebalancing forecast mode."""
    window_count = quantiles.shape[0]
    if quantiles.shape != (window_count, horizon, node_count, 2, 3):
        raise ValueError(f"Unexpected quantile shape: {quantiles.shape}")
    if window_count != len(test_starts):
        raise ValueError(f"Prediction count {window_count} does not match test starts {len(test_starts)}")

    timestamp_index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    decision_indices = test_starts + lag - 1
    target_indices = test_starts[:, None] + lag + np.arange(horizon, dtype=np.int64)[None, :]
    row_count = window_count * horizon * node_count

    dep_q10 = quantiles[..., 0, 0].reshape(row_count).astype(np.float32)
    dep_q50 = quantiles[..., 0, 1].reshape(row_count).astype(np.float32)
    dep_q90 = quantiles[..., 0, 2].reshape(row_count).astype(np.float32)
    arr_q10 = quantiles[..., 1, 0].reshape(row_count).astype(np.float32)
    arr_q50 = quantiles[..., 1, 1].reshape(row_count).astype(np.float32)
    arr_q90 = quantiles[..., 1, 2].reshape(row_count).astype(np.float32)
    net_flow_q10 = (arr_q10 - dep_q90).astype(np.float32)
    net_flow_q50 = (arr_q50 - dep_q50).astype(np.float32)
    net_flow_q90 = (arr_q90 - dep_q10).astype(np.float32)

    return pd.DataFrame(
        {
            "decision_ts": np.repeat(timestamp_index[decision_indices].to_numpy(dtype="datetime64[ns]"), horizon * node_count),
            "target_ts": np.repeat(timestamp_index[target_indices.reshape(-1)].to_numpy(dtype="datetime64[ns]"), node_count),
            "node_idx": np.tile(np.arange(node_count, dtype=np.int32), window_count * horizon),
            "dep_q10": dep_q10,
            "dep_q50": dep_q50,
            "dep_q90": dep_q90,
            "arr_q10": arr_q10,
            "arr_q50": arr_q50,
            "arr_q90": arr_q90,
            "net_flow_q10": net_flow_q10,
            "net_flow_q50": net_flow_q50,
            "net_flow_q90": net_flow_q90,
            "net_flow_pred": net_flow_q50,
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
        model = load_model(args.checkpoint, device)
        quantiles = collect_test_quantiles(model, data, device, args.limit_test_batches)
        arrays = load_bundle_arrays(args.bundle)
        _train_starts, _val_starts, test_starts = split_window_starts(
            time_count=len(arrays["timestamps"]),
            lag=args.lag,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        if args.limit_test_batches is not None:
            test_starts = test_starts[: quantiles.shape[0]]
        forecast = build_forecast_frame(
            quantiles=quantiles,
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
            "output": str(output_path),
            "row_count": int(len(forecast)),
            "window_count": int(quantiles.shape[0]),
            "horizon": int(args.horizon),
            "node_count": int(data.num_nodes),
            "columns": list(forecast.columns),
            "decision_start": str(forecast["decision_ts"].iloc[0]),
            "decision_end": str(forecast["decision_ts"].iloc[-1]),
            "target_start": str(forecast["target_ts"].iloc[0]),
            "target_end": str(forecast["target_ts"].iloc[-1]),
            "risk_columns": {
                "median": "net_flow_q50",
                "conservative": "net_flow_q10",
                "aggressive": "net_flow_q90",
            },
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

