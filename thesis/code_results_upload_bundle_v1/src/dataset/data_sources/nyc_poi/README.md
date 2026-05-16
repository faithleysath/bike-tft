# NYC POI Data Source

This directory stores OpenStreetMap POI pulls used as static station-context features.

The intended thesis use is limited:

- POI data is a static urban-context proxy, not a historical 2022 snapshot.
- Raw OSM JSON is cached by category so the feature build can be reproduced.
- Station-level features are generated separately under a new processed dataset version, for example `nyc_top883_poi_v1`, instead of overwriting existing datasets.

## Download

```bash
uv run python dataset/data_sources/nyc_poi/download_osm_poi_overpass.py \
  --output-dir dataset/data_sources/nyc_poi/raw/osm_nyc_poi_20260429
```

The default bounding box covers New York City and nearby Citi Bike station areas.
The script queries categories separately to reduce Overpass timeout risk.

## Categories

- `food`: restaurants, cafes, bars, pubs, fast food.
- `transit`: subway entrances, railway stations, public transport stations/platforms.
- `office`: OSM `office=*` plus office/commercial buildings.
- `education`: schools, universities, colleges, kindergartens.
- `healthcare`: hospitals, clinics, doctors, pharmacies.
- `retail`: OSM `shop=*`.
- `leisure`: parks, museums, attractions, fitness/sports venues.

