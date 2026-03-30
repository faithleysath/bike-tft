#!/usr/bin/env python3
"""Inspect CSV headers and sample field metadata into a JSON file."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NULL_MARKERS = {"", "na", "n/a", "nan", "null", "none"}
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass
class FieldStats:
    """Collect sampled metadata for a single CSV field."""

    name: str
    position: int
    semantic_role: str
    type_counter: Counter[str] = field(default_factory=Counter)
    non_empty_count: int = 0
    empty_count: int = 0
    unique_values: set[str] = field(default_factory=set)
    example_values: list[str] = field(default_factory=list)

    def add(self, value: str) -> None:
        """Add a sampled value to this field's statistics."""
        normalized = value.strip()
        if normalized.lower() in NULL_MARKERS:
            self.empty_count += 1
            return

        self.non_empty_count += 1
        inferred_type = "string" if self.semantic_role == "identifier" else infer_value_type(normalized)
        self.type_counter[inferred_type] += 1
        self.unique_values.add(normalized)
        if normalized not in self.example_values and len(self.example_values) < 5:
            self.example_values.append(normalized)

    def to_dict(self) -> dict[str, Any]:
        """Convert the sampled metadata to a JSON-serializable dict."""
        sample_count = self.non_empty_count + self.empty_count
        inferred_type = most_common_or_default(self.type_counter, default="unknown")
        return {
            "name": self.name,
            "position": self.position,
            "semantic_role": self.semantic_role,
            "recommended_storage_type": recommended_storage_type(self.semantic_role),
            "inferred_value_type": inferred_type,
            "observed_value_types": sorted(self.type_counter),
            "sample_non_empty_count": self.non_empty_count,
            "sample_empty_count": self.empty_count,
            "sample_fill_rate": round(self.non_empty_count / sample_count, 6) if sample_count else None,
            "sample_unique_value_count": len(self.unique_values),
            "example_values": self.example_values,
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect CSV files and export field metadata as JSON."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/stage_01_citibike_mvp/citi-bike-nyc",
        help="Directory containing CSV files to inspect.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern used to find input CSV files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/stage_01_citibike_mvp/citibike_csv_field_metadata.json",
        help="Path to the JSON metadata output file.",
    )
    parser.add_argument(
        "--sample-rows-per-file",
        type=int,
        default=1000,
        help="How many data rows to sample from each CSV file.",
    )
    return parser.parse_args()


def list_csv_files(input_dir: Path, pattern: str) -> list[Path]:
    """List CSV files that match the requested pattern."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_dir} with pattern {pattern!r}")
    return files


def read_header(path: Path) -> list[str]:
    """Read the header row from a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def sample_rows(path: Path, limit: int) -> Iterable[list[str]]:
    """Yield up to ``limit`` rows from a CSV file, excluding the header."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            yield row


def infer_value_type(value: str) -> str:
    """Infer a coarse type for a sampled CSV value."""
    if looks_like_datetime(value):
        return "datetime"
    if looks_like_int(value):
        return "integer"
    if looks_like_float(value):
        return "float"
    return "string"


def looks_like_datetime(value: str) -> bool:
    """Check whether a value matches a basic datetime format."""
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def looks_like_int(value: str) -> bool:
    """Check whether a value can be parsed as an integer."""
    return bool(INT_RE.fullmatch(value))


def looks_like_float(value: str) -> bool:
    """Check whether a value can be parsed as a float."""
    if ":" in value:
        return False
    return bool(FLOAT_RE.fullmatch(value))


def infer_semantic_role(column_name: str) -> str:
    """Infer a semantic role from the column name."""
    if column_name.endswith("_at"):
        return "timestamp"
    if column_name.endswith("_id"):
        return "identifier"
    if column_name.endswith("_lat") or column_name.endswith("_lng"):
        return "coordinate"
    if column_name.endswith("_name"):
        return "name"
    return "attribute"


def recommended_storage_type(semantic_role: str) -> str:
    """Recommend a downstream storage type from the semantic role."""
    if semantic_role == "timestamp":
        return "datetime"
    if semantic_role == "coordinate":
        return "float"
    return "string"


def build_schema_variants(files: list[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    """Group files by header layout and return the canonical header."""
    variants: dict[tuple[str, ...], list[str]] = {}
    for path in files:
        header = tuple(read_header(path))
        variants.setdefault(header, []).append(path.name)

    canonical_header = list(next(iter(variants)))
    payload = [
        {
            "columns": list(header),
            "column_count": len(header),
            "file_count": len(paths),
            "files": sorted(paths),
        }
        for header, paths in variants.items()
    ]
    payload.sort(key=lambda item: (-item["file_count"], item["columns"]))
    return canonical_header, payload


def collect_field_stats(files: list[Path], header: list[str], sample_limit: int) -> list[FieldStats]:
    """Collect sampled metadata for each field across all input files."""
    stats = [
        FieldStats(
            name=name,
            position=index,
            semantic_role=infer_semantic_role(name),
        )
        for index, name in enumerate(header)
    ]

    for path in files:
        for row in sample_rows(path, sample_limit):
            for index, field_stat in enumerate(stats):
                value = row[index] if index < len(row) else ""
                field_stat.add(value)

    return stats


def most_common_or_default(counter: Counter[str], default: str) -> str:
    """Return the most common key in a counter, or a fallback."""
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def build_metadata(
    input_dir: Path,
    pattern: str,
    files: list[Path],
    header: list[str],
    schema_variants: list[dict[str, Any]],
    field_stats: list[FieldStats],
    sample_rows_per_file: int,
) -> dict[str, Any]:
    """Build the JSON payload for CSV field metadata."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": input_dir.as_posix(),
        "pattern": pattern,
        "file_count": len(files),
        "files": [path.name for path in files],
        "canonical_header": header,
        "column_count": len(header),
        "sample_rows_per_file": sample_rows_per_file,
        "schema_variants": schema_variants,
        "fields": [field_stat.to_dict() for field_stat in field_stats],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write metadata JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run the CSV metadata extraction workflow."""
    args = parse_args()
    input_dir = project_path(args.input_dir)
    output_path = project_path(args.output)

    files = list_csv_files(input_dir, args.pattern)
    canonical_header, schema_variants = build_schema_variants(files)
    field_stats = collect_field_stats(files, canonical_header, args.sample_rows_per_file)
    payload = build_metadata(
        input_dir=input_dir,
        pattern=args.pattern,
        files=files,
        header=canonical_header,
        schema_variants=schema_variants,
        field_stats=field_stats,
        sample_rows_per_file=args.sample_rows_per_file,
    )
    write_json(output_path, payload)

    print(f"Inspected {len(files)} file(s)")
    print(f"Wrote metadata JSON to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
