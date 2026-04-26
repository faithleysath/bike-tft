# NYC Weather

## 数据源

- 数据源：Open-Meteo Historical Weather API
- API 地址：https://archive-api.open-meteo.com/v1/archive
- 坐标：`40.7128, -74.0060`
- 时区：`America/New_York`
- 时间范围：`2022-01-01` 到 `2023-01-02`

## 字段

```text
temperature_2m
apparent_temperature
relative_humidity_2m
precipitation
rain
snowfall
cloud_cover
wind_speed_10m
wind_gusts_10m
weather_code
```

## 下载

```bash
uv run python dataset/data_sources/nyc_weather/download_open_meteo_weather.py
```

默认输出到：

```text
dataset/data_sources/nyc_weather/raw/open_meteo_nyc_hourly_20220101_20230102.raw.csv
dataset/data_sources/nyc_weather/raw/open_meteo_nyc_hourly_20220101_20230102.meta.json
```
