#!/usr/bin/env python3
"""Aggregate cached OSM POIs into per-station radius features."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_STATIONS: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_v2/nyc_station_static_features.csv")
DEFAULT_POI_DIR: Final[Path] = Path("dataset/data_sources/nyc_poi/raw/osm_nyc_poi")
DEFAULT_OUTPUT: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_station_poi_features_500m.csv")
EARTH_RADIUS_M: Final[float] = 6_371_000.0


@dataclass(frozen=True)
class PoiPoint:
    """One category-specific POI point."""

    category: str
    osm_type: str
    osm_id: int
    lat: float
    lng: float


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build station-level POI count features.")
    parser.add_argument("--stations", default=DEFAULT_STATIONS.as_posix())
    parser.add_argument("--poi-dir", default=DEFAULT_POI_DIR.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--radius-m", type=float, default=500.0)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def element_point(element: dict[str, object]) -> tuple[float, float] | None:
    """Extract a representative point from an OSM element."""
    lat = element.get("lat")
    lng = element.get("lon")
    if lat is not None and lng is not None:
        return float(lat), float(lng)
    center = element.get("center")
    if isinstance(center, dict) and center.get("lat") is not None and center.get("lon") is not None:
        return float(center["lat"]), float(center["lon"])
    return None


def load_category_points(poi_dir: Path) -> dict[str, list[PoiPoint]]:
    """Load cached category JSON files into point lists."""
    category_points: dict[str, list[PoiPoint]] = {}
    for path in sorted(poi_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        category = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        points: list[PoiPoint] = []
        seen: set[tuple[str, int]] = set()
        for element in payload.get("elements", []):
            if not isinstance(element, dict):
                continue
            osm_type = str(element.get("type", ""))
            osm_id_raw = element.get("id")
            if osm_id_raw is None:
                continue
            osm_id = int(osm_id_raw)
            key = (osm_type, osm_id)
            if key in seen:
                continue
            point = element_point(element)
            if point is None:
                continue
            seen.add(key)
            points.append(PoiPoint(category=category, osm_type=osm_type, osm_id=osm_id, lat=point[0], lng=point[1]))
        category_points[category] = points
    if not category_points:
        raise FileNotFoundError(f"No category JSON files found under {poi_dir}")
    return category_points


def haversine_matrix_m(station_lat: np.ndarray, station_lng: np.ndarray, poi_lat: np.ndarray, poi_lng: np.ndarray) -> np.ndarray:
    """Compute pairwise station-to-POI distances in meters."""
    station_lat_rad = np.deg2rad(station_lat)[:, None]
    station_lng_rad = np.deg2rad(station_lng)[:, None]
    poi_lat_rad = np.deg2rad(poi_lat)[None, :]
    poi_lng_rad = np.deg2rad(poi_lng)[None, :]
    dlat = poi_lat_rad - station_lat_rad
    dlng = poi_lng_rad - station_lng_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(station_lat_rad) * np.cos(poi_lat_rad) * np.sin(dlng / 2.0) ** 2
    return (2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float32)


def aggregate_category(
    stations: pd.DataFrame,
    points: list[PoiPoint],
    *,
    radius_m: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return count within radius and nearest distance for one POI category."""
    station_lat = stations["station_lat"].to_numpy(dtype=np.float64)
    station_lng = stations["station_lng"].to_numpy(dtype=np.float64)
    counts = np.zeros(len(stations), dtype=np.int32)
    nearest = np.full(len(stations), np.inf, dtype=np.float32)
    if not points:
        return counts, np.full(len(stations), radius_m, dtype=np.float32)
    poi_lat_all = np.asarray([point.lat for point in points], dtype=np.float64)
    poi_lng_all = np.asarray([point.lng for point in points], dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        distances = haversine_matrix_m(station_lat, station_lng, poi_lat_all[start:end], poi_lng_all[start:end])
        counts += np.count_nonzero(distances <= radius_m, axis=1).astype(np.int32)
        nearest = np.minimum(nearest, distances.min(axis=1))
    nearest = np.where(np.isfinite(nearest), nearest, radius_m).astype(np.float32)
    return counts, nearest


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    stations_path = project_path(args.stations)
    poi_dir = project_path(args.poi_dir)
    output_path = project_path(args.output)
    manifest_path = output_path.with_suffix(".manifest.json")
    if (output_path.exists() or manifest_path.exists()) and not args.force:
        raise RuntimeError(f"Refusing to overwrite existing POI output: {output_path}. Use --force.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(stations_path, dtype={"station_id": "string"})
    required = {"node_idx", "station_id", "station_lat", "station_lng"}
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError(f"Station file is missing columns: {sorted(missing)}")
    stations = stations.sort_values("node_idx", kind="stable").reset_index(drop=True)
    category_points = load_category_points(poi_dir)
    result = stations.loc[:, ["node_idx", "station_id", "station_lat", "station_lng"]].copy()
    category_summary: dict[str, dict[str, float | int]] = {}
    area_km2 = math.pi * (args.radius_m / 1000.0) ** 2

    total_counts = np.zeros(len(stations), dtype=np.int32)
    for category, points in sorted(category_points.items()):
        counts, nearest = aggregate_category(stations, points, radius_m=args.radius_m, chunk_size=args.chunk_size)
        result[f"poi_{category}_count_500m"] = counts.astype(np.int32)
        result[f"poi_{category}_density_per_km2_500m"] = (counts.astype(np.float32) / area_km2).astype(np.float32)
        result[f"poi_{category}_nearest_m"] = nearest.astype(np.float32)
        total_counts += counts
        category_summary[category] = {
            "raw_point_count": len(points),
            "station_count_mean": float(counts.mean()),
            "station_count_max": int(counts.max(initial=0)),
            "station_count_nonzero": int(np.count_nonzero(counts)),
        }
        print(json.dumps({"category": category, **category_summary[category]}))

    result["poi_total_count_500m"] = total_counts.astype(np.int32)
    result["poi_total_density_per_km2_500m"] = (total_counts.astype(np.float32) / area_km2).astype(np.float32)
    result.to_csv(output_path, index=False)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "stations": stations_path.as_posix(),
            "poi_dir": poi_dir.as_posix(),
            "provider": "OpenStreetMap via Overpass API",
            "license": "Open Database License (ODbL)",
        },
        "parameters": {
            "radius_m": args.radius_m,
            "chunk_size": args.chunk_size,
        },
        "outputs": {
            "station_poi_features": output_path.as_posix(),
        },
        "station_count": int(len(result)),
        "feature_columns": [column for column in result.columns if column.startswith("poi_")],
        "category_summary": category_summary,
        "notes": {
            "poi_snapshot": "POIs reflect the downloaded OpenStreetMap snapshot, not a historical 2022 POI snapshot.",
            "geometry": "Ways and relations are represented by Overpass center points before radius aggregation.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
