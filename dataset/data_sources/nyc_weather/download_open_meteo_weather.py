#!/usr/bin/env python3
"""Download hourly Open-Meteo weather data for New York City."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = Path("dataset/data_sources/nyc_weather/raw")
OPEN_METEO_ARCHIVE_URL: Final[str] = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_LATITUDE: Final[float] = 40.7128
DEFAULT_LONGITUDE: Final[float] = -74.0060
DEFAULT_START_DATE: Final[str] = "2022-01-01"
DEFAULT_END_DATE: Final[str] = "2023-01-02"
DEFAULT_TIMEZONE: Final[str] = "America/New_York"
DEFAULT_PREFIX: Final[str] = "open_meteo_nyc_hourly_20220101_20230102"
DEFAULT_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_RETRIES: Final[int] = 3
HOURLY_FIELDS: Final[list[str]] = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "weather_code",
]


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_iso_date(value: str) -> date:
    """Parse a YYYY-MM-DD date argument."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value!r}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download hourly historical weather from Open-Meteo."
    )
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--start-date", type=parse_iso_date, default=parse_iso_date(DEFAULT_START_DATE))
    parser.add_argument("--end-date", type=parse_iso_date, default=parse_iso_date(DEFAULT_END_DATE))
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Directory used to store the downloaded raw CSV and metadata JSON.",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output files if they already exist.",
    )
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("--end-date must be greater than or equal to --start-date")
    return args


def build_request_url(args: argparse.Namespace, *, response_format: str) -> str:
    """Build an Open-Meteo archive request URL."""
    params = {
        "latitude": f"{args.latitude:.4f}",
        "longitude": f"{args.longitude:.4f}",
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": args.timezone,
        "format": response_format,
    }
    return f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}"


def ensure_target_paths(csv_path: Path, meta_path: Path, force: bool) -> None:
    """Reject accidental overwrites unless --force is present."""
    existing = [path for path in (csv_path, meta_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"Refusing to overwrite existing output file(s): {names}. Use --force to replace them."
        )


def fetch_url_bytes(url: str, retries: int = DEFAULT_RETRIES) -> bytes:
    """Fetch bytes from an HTTPS endpoint with simple retry handling."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response:  # nosec: fixed HTTPS URL
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 5))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def decode_bytes(payload: bytes, *, label: str) -> str:
    """Decode UTF-8 API payloads with a helpful error message."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Failed to decode {label} payload as UTF-8") from exc


def parse_csv_sections(csv_text: str) -> tuple[list[str], list[str], list[str], list[list[str]]]:
    """Split Open-Meteo CSV output into metadata header/value rows and data rows."""
    rows = [row for row in csv.reader(StringIO(csv_text)) if any(cell.strip() for cell in row)]
    if len(rows) < 4:
        raise RuntimeError("Open-Meteo CSV payload is shorter than expected")

    metadata_header = [cell.strip() for cell in rows[0]]
    metadata_values = [cell.strip() for cell in rows[1]]
    data_header = [cell.strip() for cell in rows[2]]
    data_rows = [[cell.strip() for cell in row] for row in rows[3:]]
    if not data_rows:
        raise RuntimeError("Open-Meteo CSV payload contains no hourly rows")
    return metadata_header, metadata_values, data_header, data_rows


def validate_csv_payload(
    data_header: list[str],
    data_rows: list[list[str]],
    *,
    expected_start_date: date,
    expected_end_date: date,
) -> None:
    """Enforce the expected CSV shape and hourly time coverage."""
    if "time" not in data_header:
        raise RuntimeError("Downloaded CSV is missing the 'time' column")

    time_index = data_header.index("time")
    first_time = datetime.fromisoformat(data_rows[0][time_index])
    last_time = datetime.fromisoformat(data_rows[-1][time_index])
    expected_start = datetime.fromisoformat(f"{expected_start_date.isoformat()}T00:00")
    expected_end_min = datetime.fromisoformat(f"{expected_end_date.isoformat()}T23:00")

    if first_time != expected_start:
        raise RuntimeError(
            f"Unexpected first hourly timestamp: {first_time.isoformat(timespec='minutes')}"
        )
    if last_time < expected_end_min:
        raise RuntimeError(
            f"Downloaded CSV stops too early: {last_time.isoformat(timespec='minutes')}"
        )


def validate_metadata_payload(metadata_payload: dict[str, object], expected_timezone: str) -> None:
    """Verify metadata returned by Open-Meteo matches the requested timezone."""
    timezone = metadata_payload.get("timezone")
    if timezone != expected_timezone:
        raise RuntimeError(
            f"Unexpected Open-Meteo timezone: {timezone!r} (expected {expected_timezone!r})"
        )


def compute_sha256(payload: bytes) -> str:
    """Compute a hex SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write bytes to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Serialize JSON metadata to disk atomically."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_atomic(path, serialized)


def main() -> int:
    """Entrypoint for downloading Open-Meteo hourly weather data."""
    try:
        args = parse_args()
        output_dir = project_path(args.output_dir)
        csv_path = output_dir / f"{args.prefix}.raw.csv"
        meta_path = output_dir / f"{args.prefix}.meta.json"

        ensure_target_paths(csv_path, meta_path, force=args.force)

        csv_url = build_request_url(args, response_format="csv")
        json_url = build_request_url(args, response_format="json")

        print(f"Downloading Open-Meteo CSV to {csv_path}")
        csv_bytes = fetch_url_bytes(csv_url)
        if not csv_bytes:
            raise RuntimeError("Downloaded CSV payload is empty")

        print("Downloading Open-Meteo metadata payload")
        metadata_bytes = fetch_url_bytes(json_url)
        metadata_payload = json.loads(decode_bytes(metadata_bytes, label="JSON"))
        if not isinstance(metadata_payload, dict):
            raise RuntimeError("Open-Meteo JSON payload is not an object")

        validate_metadata_payload(metadata_payload, args.timezone)

        csv_text = decode_bytes(csv_bytes, label="CSV")
        metadata_header, metadata_values, data_header, data_rows = parse_csv_sections(csv_text)
        validate_csv_payload(
            data_header,
            data_rows,
            expected_start_date=args.start_date,
            expected_end_date=args.end_date,
        )

        sha256 = compute_sha256(csv_bytes)
        units = metadata_payload.get("hourly_units")
        if not isinstance(units, dict):
            raise RuntimeError("Open-Meteo JSON payload is missing hourly_units metadata")

        write_bytes_atomic(csv_path, csv_bytes)

        metadata_document: dict[str, object] = {
            "source": "open-meteo",
            "request_url": csv_url,
            "metadata_request_url": json_url,
            "downloaded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "input_coordinates": {
                "latitude": args.latitude,
                "longitude": args.longitude,
            },
            "returned_coordinates": {
                "latitude": metadata_payload.get("latitude"),
                "longitude": metadata_payload.get("longitude"),
            },
            "timezone": metadata_payload.get("timezone"),
            "timezone_abbreviation": metadata_payload.get("timezone_abbreviation"),
            "utc_offset_seconds": metadata_payload.get("utc_offset_seconds"),
            "hourly_fields": HOURLY_FIELDS,
            "hourly_units": units,
            "csv_metadata_header": metadata_header,
            "csv_metadata_values": metadata_values,
            "csv_header": data_header,
            "raw_file_size_bytes": len(csv_bytes),
            "sha256": sha256,
        }
        write_json_atomic(meta_path, metadata_document)

        print(f"Wrote {csv_path}")
        print(f"Wrote {meta_path}")
        return 0
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
