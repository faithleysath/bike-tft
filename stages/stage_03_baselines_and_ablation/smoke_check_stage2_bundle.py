#!/usr/bin/env python3
"""Validate that the stage 2 AGCRN bundle can be sliced into stage 3 windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smoke-check the stage 2 AGCRN bundle without training a model."
    )
    parser.add_argument(
        "--bundle",
        default="data/processed/stage_02_feature_enrichment/agcrn_stage3_bundle.npz",
        help="Path to the stage 2 AGCRN bundle.",
    )
    parser.add_argument("--lag", type=int, default=12, help="Encoder window length.")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon.")
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional JSON report path for the smoke-check summary.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def main() -> None:
    """Load the stage 2 bundle and verify stage 3 window slicing assumptions."""
    args = parse_args()
    if args.lag < 1 or args.horizon < 1:
        raise SystemExit("--lag and --horizon must both be positive")

    bundle_path = project_path(args.bundle)
    payload = np.load(bundle_path, allow_pickle=False)
    features = payload["features"]
    target_dep = payload["target_dep"]
    target_arr = payload["target_arr"]
    timestamps = payload["timestamps"]
    node_ids = payload["node_ids"]
    feature_names = payload["feature_names"]

    if features.ndim != 3:
        raise ValueError(f"Expected features to be 3D, found shape {features.shape}")
    if target_dep.ndim != 3 or target_arr.ndim != 3:
        raise ValueError("Expected both targets to be 3D arrays")
    if features.shape[0] != target_dep.shape[0] or features.shape[0] != target_arr.shape[0]:
        raise ValueError("Feature and target timestamp axes do not match")
    if features.shape[1] != target_dep.shape[1] or features.shape[1] != target_arr.shape[1]:
        raise ValueError("Feature and target node axes do not match")
    if features.shape[1] != len(node_ids):
        raise ValueError("node_ids length does not match the node axis in features")
    if features.shape[2] != len(feature_names):
        raise ValueError("feature_names length does not match the feature axis")

    sample_count = features.shape[0] - args.lag - args.horizon + 1
    if sample_count <= 0:
        raise ValueError("Bundle is too short for the requested lag/horizon configuration")

    sample_indices = sorted({0, sample_count // 2, sample_count - 1})
    x_preview = np.stack([features[index : index + args.lag] for index in sample_indices], axis=0)
    y_dep_preview = np.stack(
        [target_dep[index + args.lag : index + args.lag + args.horizon] for index in sample_indices],
        axis=0,
    )
    y_arr_preview = np.stack(
        [target_arr[index + args.lag : index + args.lag + args.horizon] for index in sample_indices],
        axis=0,
    )

    if x_preview.shape[1:] != (args.lag, features.shape[1], features.shape[2]):
        raise ValueError(f"Unexpected X preview shape: {x_preview.shape}")
    if y_dep_preview.shape[1:] != (args.horizon, features.shape[1], 1):
        raise ValueError(f"Unexpected dep target preview shape: {y_dep_preview.shape}")
    if y_arr_preview.shape[1:] != (args.horizon, features.shape[1], 1):
        raise ValueError(f"Unexpected arr target preview shape: {y_arr_preview.shape}")

    report = {
        "bundle": str(bundle_path),
        "lag": args.lag,
        "horizon": args.horizon,
        "timestamps": int(features.shape[0]),
        "nodes": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "sample_count": int(sample_count),
        "expected_x_shape": [int(sample_count), args.lag, int(features.shape[1]), int(features.shape[2])],
        "expected_y_shape": [int(sample_count), args.horizon, int(features.shape[1]), 1],
        "preview_x_shape": [int(value) for value in x_preview.shape],
        "preview_y_dep_shape": [int(value) for value in y_dep_preview.shape],
        "preview_y_arr_shape": [int(value) for value in y_arr_preview.shape],
        "first_timestamp": str(timestamps[0]),
        "last_timestamp": str(timestamps[-1]),
    }

    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.report_json is not None:
        report_path = project_path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
