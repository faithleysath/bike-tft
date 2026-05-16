# POI Feature Builders

This directory contains scripts that convert cached OpenStreetMap POIs into station-level static features.

Recommended thesis version:

```bash
uv run python dataset/preprocessing/poi_features/build_station_poi_features.py \
  --poi-dir dataset/data_sources/nyc_poi/raw/osm_nyc_poi_20260429 \
  --output dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_station_poi_features_500m.csv

uv run python dataset/preprocessing/poi_features/build_nyc_dataset_with_poi.py \
  --base-dir dataset/preprocessing/processed/nyc_top883_v2 \
  --poi-features dataset/preprocessing/processed/nyc_top883_poi_v1/nyc_station_poi_features_500m.csv \
  --output-dir dataset/preprocessing/processed/nyc_top883_poi_v1
```

The scripts are additive and refuse to overwrite existing processed outputs unless `--force` is passed.

