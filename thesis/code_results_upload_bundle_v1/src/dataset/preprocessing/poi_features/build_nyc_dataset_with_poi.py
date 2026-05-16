#!/usr/bin/env python3
"""Create an additive NYC processed dataset version with static POI features."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_BASE_DIR: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_v2")
DEFAULT_POI_FEATURES: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_station_poi_features_500m.csv")
DEFAULT_OUTPUT_DIR: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_poi_v1")


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Append station-level POI features to an existing NYC bundle.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR.as_posix())
    parser.add_argument("--poi-features", default=DEFAULT_POI_FEATURES.as_posix())
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed outputs.")
    return parser.parse_args()


def ensure_output_dir(output_dir: Path, *, force: bool) -> None:
    """Create output dir and guard against accidental overwrites."""
    outputs = [
        output_dir / "nyc_station_hour_panel.parquet",
        output_dir / "nyc_station_static_features.csv",
        output_dir / "nyc_agcrn_bundle.npz",
        output_dir / "nyc_dataset_manifest.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        names = ", ".join(path.as_posix() for path in existing)
        raise RuntimeError(f"Refusing to overwrite existing output(s): {names}. Use --force.")
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    base_dir = project_path(args.base_dir)
    poi_path = project_path(args.poi_features)
    output_dir = project_path(args.output_dir)
    ensure_output_dir(output_dir, force=args.force)

    base_panel_path = base_dir / "nyc_station_hour_panel.parquet"
    base_static_path = base_dir / "nyc_station_static_features.csv"
    base_bundle_path = base_dir / "nyc_agcrn_bundle.npz"
    base_manifest_path = base_dir / "nyc_dataset_manifest.json"
    for path in [base_panel_path, base_static_path, base_bundle_path, base_manifest_path, poi_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    poi = pd.read_csv(poi_path, dtype={"station_id": "string"})
    poi_columns = [column for column in poi.columns if column.startswith("poi_")]
    if not poi_columns:
        raise ValueError(f"No POI feature columns found in {poi_path}")
    poi = poi.loc[:, ["node_idx", "station_id", *poi_columns]].copy()
    poi["station_id"] = poi["station_id"].astype(str)

    static = pd.read_csv(base_static_path, dtype={"station_id": "string"})
    static["station_id"] = static["station_id"].astype(str)
    static_with_poi = static.merge(
        poi.drop(columns=["station_id"]),
        on="node_idx",
        how="left",
        validate="one_to_one",
    )
    if static_with_poi[poi_columns].isna().any().any():
        raise ValueError("Some stations did not receive POI features")
    static_with_poi.to_csv(output_dir / "nyc_station_static_features.csv", index=False)

    panel = pd.read_parquet(base_panel_path)
    panel["station_id"] = panel["station_id"].astype(str)
    panel_with_poi = panel.merge(
        poi.drop(columns=["station_id"]),
        on="node_idx",
        how="left",
        validate="many_to_one",
    )
    if panel_with_poi[poi_columns].isna().any().any():
        raise ValueError("Some panel rows did not receive POI features")
    panel_with_poi.to_parquet(output_dir / "nyc_station_hour_panel.parquet", index=False)

    arrays = np.load(base_bundle_path, allow_pickle=False)
    features = arrays["features"].astype(np.float32)
    station_ids = [str(item) for item in arrays["station_ids"].tolist()]
    poi_sorted = poi.sort_values("node_idx", kind="stable").reset_index(drop=True)
    if len(poi_sorted) != len(station_ids) or poi_sorted["node_idx"].to_numpy().tolist() != list(range(len(station_ids))):
        raise ValueError("POI node_idx order does not match bundle station_ids")
    poi_matrix = poi_sorted[poi_columns].to_numpy(dtype=np.float32)
    poi_tensor = np.broadcast_to(poi_matrix[None, :, :], (features.shape[0], features.shape[1], len(poi_columns))).copy()
    new_features = np.concatenate([features, poi_tensor], axis=-1).astype(np.float32)
    base_feature_names = [str(item) for item in arrays["feature_names"].tolist()]
    feature_names = np.asarray([*base_feature_names, *poi_columns])
    np.savez_compressed(
        output_dir / "nyc_agcrn_bundle.npz",
        features=new_features,
        target_dep=arrays["target_dep"],
        target_arr=arrays["target_arr"],
        target_inventory=arrays["target_inventory"],
        timestamps=arrays["timestamps"],
        station_ids=arrays["station_ids"],
        feature_names=feature_names,
    )

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_dataset": base_manifest,
        "source": {
            "base_dir": base_dir.as_posix(),
            "poi_features": poi_path.as_posix(),
        },
        "outputs": {
            "panel": (output_dir / "nyc_station_hour_panel.parquet").as_posix(),
            "station_static": (output_dir / "nyc_station_static_features.csv").as_posix(),
            "bundle": (output_dir / "nyc_agcrn_bundle.npz").as_posix(),
        },
        "panel": {
            "rows": int(len(panel_with_poi)),
            "time_count": int(len(pd.unique(panel_with_poi["ts"]))),
            "station_count": int(len(static_with_poi)),
        },
        "features": {
            "base_feature_count": int(features.shape[-1]),
            "poi_feature_count": len(poi_columns),
            "feature_count": int(new_features.shape[-1]),
            "poi_feature_names": poi_columns,
            "targets": ["target_dep", "target_arr", "target_inventory"],
        },
        "bundle_shapes": {
            "features": list(new_features.shape),
            "target_dep": list(arrays["target_dep"].shape),
            "target_arr": list(arrays["target_arr"].shape),
            "target_inventory": list(arrays["target_inventory"].shape),
            "timestamps": list(arrays["timestamps"].shape),
            "station_ids": list(arrays["station_ids"].shape),
            "feature_names": list(feature_names.shape),
        },
        "notes": {
            "version_reason": "Add station-level OpenStreetMap POI radius features for thesis alignment and ablation.",
            "poi_snapshot": "POI features are static and reflect the downloaded OSM snapshot, not a historical 2022 POI snapshot.",
        },
    }
    (output_dir / "nyc_dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    poi_manifest = poi_path.with_suffix(".manifest.json")
    if poi_manifest.exists():
        manifest_copy_path = output_dir / poi_manifest.name
        if poi_manifest.resolve() != manifest_copy_path.resolve():
            shutil.copy2(poi_manifest, manifest_copy_path)
    print(json.dumps({"output_dir": output_dir.as_posix(), "feature_count": int(new_features.shape[-1])}, indent=2))


if __name__ == "__main__":
    main()
