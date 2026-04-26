#!/usr/bin/env python3
"""Download a Kaggle dataset into the NYC Citi Bike raw data directory."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR: Final[Path] = Path("dataset/data_sources/nyc_citibike_orders/raw")
KAGGLE_OWNER_SLUG_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)$"
)
KAGGLE_DATASET_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"^https?://www\.kaggle\.com/datasets/(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)(?:[/?#].*)?$"
)
KAGGLE_JSON_LD_URL_RE: Final[re.Pattern[str]] = re.compile(
    r'"url":"https://www\.kaggle\.com/(?P<owner>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)"'
)


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths no matter where the script is launched from."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset into the NYC Citi Bike raw data directory."
    )
    parser.add_argument(
        "dataset",
        help=(
            "Kaggle dataset reference. Accepts either owner/slug or a Kaggle dataset URL. "
            "Hashed dataset URLs are resolved automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help=(
            "Root directory for downloaded raw data "
            f"(default: {DEFAULT_OUTPUT_DIR.as_posix()})."
        ),
    )
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unzip the downloaded dataset after download (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing dataset directory and download again.",
    )
    return parser.parse_args()


def resolve_dataset_ref(dataset: str) -> str:
    """Resolve a Kaggle owner/slug from user input."""
    value = dataset.strip().rstrip("/")
    if match := KAGGLE_OWNER_SLUG_RE.fullmatch(value):
        return f"{match.group('owner')}/{match.group('slug')}"

    if match := KAGGLE_DATASET_URL_RE.fullmatch(value):
        return f"{match.group('owner')}/{match.group('slug')}"

    if value.startswith("http://") or value.startswith("https://"):
        return resolve_hashed_kaggle_url(value)

    raise ValueError(
        "Unsupported dataset reference. Use a Kaggle dataset URL or owner/slug, "
        "for example leonczarlinski/citi-bike-nyc."
    )


def resolve_hashed_kaggle_url(url: str) -> str:
    """Resolve a hashed Kaggle dataset URL by reading its public HTML metadata."""
    try:
        with urlopen(url) as response:  # nosec: Kaggle HTTPS URL provided by the user
            html = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise ValueError(f"Could not open Kaggle dataset page: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"Could not open Kaggle dataset page: {exc.reason}") from exc

    match = KAGGLE_JSON_LD_URL_RE.search(html)
    if not match:
        raise ValueError(
            "Failed to resolve the dataset owner/slug from the Kaggle page. "
            "Please pass owner/slug directly."
        )

    return f"{match.group('owner')}/{match.group('slug')}"


def build_target_dir(output_root: Path, dataset_ref: str) -> Path:
    """Build the on-disk download directory for a Kaggle dataset."""
    _, slug = dataset_ref.split("/", maxsplit=1)
    return output_root / slug


def ensure_credentials_hint() -> None:
    """Fail fast with a clear message when Kaggle credentials are missing."""
    home = Path.home()
    candidate_files = [
        home / ".kaggle" / "kaggle.json",
        home / ".config" / "kaggle" / "kaggle.json",
        home / ".kaggle" / "access_token",
        home / ".kaggle" / "access_token.txt",
    ]
    has_file = any(path.is_file() for path in candidate_files)
    has_env = bool(
        (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or os.environ.get("KAGGLE_API_TOKEN")
    )

    if has_file or has_env:
        return

    raise RuntimeError(
        "Kaggle credentials not found. Configure ~/.kaggle/kaggle.json, "
        "~/.kaggle/access_token, or set KAGGLE_USERNAME/KAGGLE_KEY or "
        "KAGGLE_API_TOKEN before downloading."
    )


def prepare_target_dir(target_dir: Path, force: bool) -> None:
    """Create an empty target directory, optionally replacing existing data."""
    if target_dir.exists():
        if not force:
            if any(target_dir.iterdir()):
                print(
                    f"Dataset directory already exists and is not empty: {target_dir}",
                    file=sys.stderr,
                )
                print("Use --force to replace it.", file=sys.stderr)
                raise SystemExit(2)
            return

        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)


def download_dataset(dataset_ref: str, target_dir: Path, extract: bool) -> None:
    """Authenticate with Kaggle and download the dataset files."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=dataset_ref,
        path=target_dir.as_posix(),
        unzip=extract,
        quiet=False,
    )


def main() -> int:
    """Entrypoint for downloading Kaggle datasets."""
    try:
        args = parse_args()
        dataset_ref = resolve_dataset_ref(args.dataset)
        output_root = project_path(args.output_dir)
        target_dir = build_target_dir(output_root, dataset_ref)

        ensure_credentials_hint()
        prepare_target_dir(target_dir, force=args.force)

        print(f"Resolved dataset: {dataset_ref}")
        print(f"Download directory: {target_dir}")
        print(f"Extract archives: {args.extract}")

        download_dataset(dataset_ref, target_dir, extract=args.extract)
        print("Download complete.")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
