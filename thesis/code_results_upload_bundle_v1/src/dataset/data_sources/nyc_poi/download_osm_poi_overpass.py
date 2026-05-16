#!/usr/bin/env python3
"""Download NYC OpenStreetMap POIs by category through the Overpass API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = Path("dataset/data_sources/nyc_poi/raw/osm_nyc_poi")
DEFAULT_ENDPOINT: Final[str] = "https://overpass-api.de/api/interpreter"
DEFAULT_BBOX: Final[str] = "40.45,-74.30,40.95,-73.65"


@dataclass(frozen=True)
class TagFilter:
    """One OSM tag filter used in an Overpass query."""

    key: str
    values: tuple[str, ...] | None = None


CATEGORY_FILTERS: Final[dict[str, tuple[TagFilter, ...]]] = {
    "food": (
        TagFilter("amenity", ("restaurant", "cafe", "fast_food", "bar", "pub", "food_court")),
    ),
    "transit": (
        TagFilter("railway", ("station", "subway_entrance", "halt", "tram_stop")),
        TagFilter("public_transport", ("station", "platform", "stop_position")),
        TagFilter("station", ("subway", "light_rail")),
    ),
    "office": (
        TagFilter("office", None),
        TagFilter("building", ("office", "commercial")),
    ),
    "education": (
        TagFilter("amenity", ("school", "university", "college", "kindergarten")),
    ),
    "healthcare": (
        TagFilter("amenity", ("hospital", "clinic", "doctors", "pharmacy", "dentist")),
        TagFilter("healthcare", None),
    ),
    "retail": (
        TagFilter("shop", None),
    ),
    "leisure": (
        TagFilter("leisure", ("park", "fitness_centre", "sports_centre", "garden", "playground")),
        TagFilter("tourism", ("museum", "attraction", "gallery", "hotel")),
        TagFilter("amenity", ("theatre", "cinema", "arts_centre", "community_centre")),
    ),
}


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download NYC OSM POIs from Overpass by category.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--bbox",
        default=DEFAULT_BBOX,
        help="Overpass bbox as south,west,north,east. Default covers NYC and nearby Citi Bike stations.",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Overpass query timeout in seconds.")
    parser.add_argument("--request-timeout", type=int, default=240, help="HTTP request timeout in seconds.")
    parser.add_argument("--sleep-seconds", type=float, default=3.0, help="Delay between category requests.")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORY_FILTERS),
        default=sorted(CATEGORY_FILTERS),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing category JSON files.")
    return parser.parse_args()


def overpass_filter(filter_spec: TagFilter, bbox: str) -> str:
    """Return node/way/relation query lines for one tag filter."""
    if filter_spec.values:
        escaped = "|".join(urllib.parse.quote(value, safe="") for value in filter_spec.values)
        selector = f'["{filter_spec.key}"~"^({escaped})$"]'
    else:
        selector = f'["{filter_spec.key}"]'
    return "\n".join(
        [
            f"  node{selector}({bbox});",
            f"  way{selector}({bbox});",
            f"  relation{selector}({bbox});",
        ]
    )


def build_query(category: str, *, bbox: str, timeout: int) -> str:
    """Build one Overpass QL query for a category."""
    lines: list[str] = [f"[out:json][timeout:{timeout}];", "("]
    for filter_spec in CATEGORY_FILTERS[category]:
        lines.append(overpass_filter(filter_spec, bbox))
    lines.extend([");", "out center tags;"])
    return "\n".join(lines)


def post_overpass(endpoint: str, query: str, *, request_timeout: int) -> dict[str, object]:
    """POST a query to Overpass and return parsed JSON."""
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": "bike-tft-thesis-poi-builder/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Overpass HTTP {exc.code}: {detail[:1000]}") from exc


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": {
            "provider": "OpenStreetMap via Overpass API",
            "endpoint": args.endpoint,
            "license": "Open Database License (ODbL)",
        },
        "parameters": {
            "bbox": args.bbox,
            "timeout": args.timeout,
            "categories": args.categories,
        },
        "category_files": {},
    }

    for index, category in enumerate(args.categories):
        output_path = output_dir / f"{category}.json"
        query_path = output_dir / f"{category}.overpassql"
        query = build_query(category, bbox=args.bbox, timeout=args.timeout)
        query_path.write_text(query + "\n", encoding="utf-8")
        if output_path.exists() and not args.force:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            status = "existing"
        else:
            if index > 0 and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            payload = post_overpass(args.endpoint, query, request_timeout=args.request_timeout)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            status = "downloaded"
        element_count = len(payload.get("elements", [])) if isinstance(payload, dict) else 0
        manifest["category_files"][category] = {
            "json": output_path.relative_to(PROJECT_ROOT).as_posix()
            if output_path.is_relative_to(PROJECT_ROOT)
            else output_path.as_posix(),
            "query": query_path.relative_to(PROJECT_ROOT).as_posix()
            if query_path.is_relative_to(PROJECT_ROOT)
            else query_path.as_posix(),
            "status": status,
            "element_count": element_count,
        }
        print(json.dumps({"category": category, "status": status, "element_count": element_count}))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

