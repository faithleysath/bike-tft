# Citi Bike -> TFT MVP

This is a minimal pipeline to get a station-level multi-horizon forecasting workflow running quickly.

## What this MVP solves

Use Citi Bike trip history data to build a **station-hour panel** and train a **Temporal Fusion Transformer (TFT)** to predict:

- target: `dep_count`
- granularity: `station_id x hour`
- horizon: next `6` hours by default

This is intentionally minimal:

- no weather yet
- no POI yet
- no inventory yet
- no rebalancing yet

The goal is to make the training route work first. Later, your own campus data can be remapped to the same schema.

## Expected raw input

One or more Citi Bike monthly CSV files with columns such as:

- `ride_id`
- `rideable_type`
- `started_at`
- `ended_at`
- `start_station_name`
- `start_station_id`
- `end_station_name`
- `end_station_id`
- `start_lat`
- `start_lng`
- `end_lat`
- `end_lng`
- `member_casual`

## Recommended first run

Start small:

- 1 to 3 months of trip CSVs
- hourly aggregation
- top 100 to 200 most active stations
- predict next 6 hours
- encoder length = 7 days = 168 hours

This is usually small enough for a laptop and large enough to validate the full pipeline.

## 1) Create the station-hour panel

```bash
python preprocess_citibike.py \
  --input ./data/raw \
  --output-dir ./data/processed \
  --freq 1H \
  --top-n-stations 200
```

Outputs:

- `station_hour_panel.parquet`
- `station_meta.parquet`
- `summary.csv`

Key columns in `station_hour_panel.parquet`:

- `ts`
- `station_id`
- `station_name`
- `station_lat`
- `station_lng`
- `dep_count`
- `arr_count`
- `net_flow`
- `dep_member_count`
- `dep_casual_count`
- `dep_classic_count`
- `dep_electric_count`
- `hour`
- `day_of_week`
- `day_of_month`
- `month`
- `week_of_year`
- `is_weekend`
- `hour_sin`, `hour_cos`
- `dow_sin`, `dow_cos`
- `time_idx`

## 2) Train TFT

```bash
python train_tft.py \
  --data ./data/processed/station_hour_panel.parquet \
  --output-dir ./runs/citibike_tft \
  --target dep_count \
  --max-encoder-length 168 \
  --max-prediction-length 6 \
  --validation-horizon 168 \
  --batch-size 128 \
  --max-epochs 15
```

Artifacts:

- training logs under `runs/citibike_tft/logs`
- best checkpoint under `runs/citibike_tft/checkpoints`
- dataset object under `runs/citibike_tft/timeseries_dataset.pkl`

## Minimal modeling choices

### Target

`dep_count` is the number of trips that start from a station in an hour.

Why start with departures instead of inventory?

- trip data directly supervises departures
- no need to reconstruct station stock first
- easier to validate the time-series pipeline

Later you can switch or extend to:

- `arr_count`
- multi-target (`dep_count`, `arr_count`)
- inferred inventory / stock
- campus station demand

### Features in this MVP

Static features:

- `station_id`
- `station_name`
- `station_lat`
- `station_lng`

Known future features:

- `time_idx`
- `hour`
- `day_of_week`
- `day_of_month`
- `month`
- `week_of_year`
- `is_weekend`
- `hour_sin`, `hour_cos`
- `dow_sin`, `dow_cos`

Unknown historical features:

- `dep_count`
- `arr_count`
- `net_flow`
- member/casual split
- classic/electric split

## How to map this to your campus project later

Your future campus table only needs to match the same idea:

- one row per `station_id x time_slot`
- one integer `time_idx`
- target column such as `dep_count`
- static station features
- known future time features
- unknown past demand / stock features

Then you can keep the same `train_tft.py` with only small column changes.

## Suggested next upgrades after the MVP works

1. Add holiday and school-calendar features.
2. Add weather as known future covariates.
3. Add station neighborhood / POI / zone labels as static features.
4. Add inventory or inferred stock to support rebalancing.
5. Add a simple LightGBM baseline and compare against TFT.

## Practical advice

- Do not use all stations first.
- Do not use minute-level granularity first.
- Do not add too many features before the pipeline trains once.
- Get one clean end-to-end run first.

## Environment

Suggested core packages:

- pandas
- pyarrow
- numpy
- torch
- lightning
- pytorch-forecasting

You can install them manually or adapt them to your current environment.
