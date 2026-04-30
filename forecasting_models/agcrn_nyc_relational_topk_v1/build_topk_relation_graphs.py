#!/usr/bin/env python3
"""Build sparse top-k OD relation graphs from the training-period dense OD artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_BASE_GRAPH: Final[Path] = Path("dataset/preprocessing/processed/nyc_top883_relation_graphs_v1.npz")
DEFAULT_OUTPUT_TEMPLATE: Final[str] = "dataset/preprocessing/processed/nyc_top883_relation_graphs_topk_v1_k{top_k}.npz"


def project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths from any launch directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build row-wise top-k OD relation graph supports.")
    parser.add_argument("--base-graph", default=DEFAULT_BASE_GRAPH.as_posix())
    parser.add_argument("--output", default=None, help="Output .npz path. Defaults to a k-specific processed path.")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Keep same-station OD counts. Defaults to excluding them because identity support already models self-loops.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing graph artifact.")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a dense support matrix, leaving all-zero rows as zeros."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float32), where=row_sums > 0).astype(np.float32)


def topk_counts(counts: np.ndarray, *, top_k: int, include_self: bool) -> tuple[np.ndarray, dict[str, object]]:
    """Keep the largest positive counts per row and zero all other entries."""
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"OD counts must be a square matrix, got shape {counts.shape}")
    node_count = counts.shape[0]
    neighbor_count = min(top_k, node_count if include_self else node_count - 1)
    if neighbor_count <= 0:
        raise ValueError("top_k leaves no eligible neighbors")

    working = counts.astype(np.float32, copy=True)
    if not include_self:
        np.fill_diagonal(working, 0.0)

    sparse = np.zeros_like(working, dtype=np.float32)
    for row_index in range(node_count):
        row = working[row_index]
        positive = np.flatnonzero(row > 0)
        if len(positive) == 0:
            continue
        if len(positive) <= neighbor_count:
            keep = positive
        else:
            partition = np.argpartition(row[positive], kth=len(positive) - neighbor_count)[-neighbor_count:]
            keep = positive[partition]
        sparse[row_index, keep] = row[keep]

    row_nonzero = np.count_nonzero(sparse, axis=1)
    eligible_mass = float(working.sum())
    retained_mass = float(sparse.sum())
    nonzero_row_counts = row_nonzero[row_nonzero > 0]
    metadata = {
        "node_count": int(node_count),
        "requested_top_k": int(top_k),
        "effective_top_k": int(neighbor_count),
        "include_self": bool(include_self),
        "nonzero_edges": int(np.count_nonzero(sparse)),
        "density": float(np.count_nonzero(sparse) / float(sparse.size)),
        "zero_row_count": int(np.count_nonzero(row_nonzero == 0)),
        "min_nonzero_per_nonzero_row": int(nonzero_row_counts.min()) if len(nonzero_row_counts) else 0,
        "max_nonzero_per_row": int(row_nonzero.max(initial=0)),
        "mean_nonzero_per_row": float(row_nonzero.mean()),
        "eligible_trip_mass": eligible_mass,
        "retained_trip_mass": retained_mass,
        "retained_trip_mass_share": float(retained_mass / eligible_mass) if eligible_mass > 0 else 0.0,
    }
    return sparse, metadata


def load_base_graph(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load dense OD counts, station IDs, and metadata from the base graph artifact."""
    arrays = np.load(path, allow_pickle=False)
    required = {"od_counts", "station_ids", "metadata_json"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"Base relation graph is missing array(s): {sorted(missing)}")
    od_counts = arrays["od_counts"]
    station_ids = arrays["station_ids"]
    metadata = json.loads(str(arrays["metadata_json"].item()))
    return od_counts, station_ids, metadata


def output_path_for(args: argparse.Namespace) -> Path:
    """Resolve the output path for the requested k."""
    if args.output is not None:
        return project_path(args.output)
    return project_path(DEFAULT_OUTPUT_TEMPLATE.format(top_k=args.top_k))


def main() -> int:
    """Build and write a top-k relation graph artifact."""
    try:
        args = parse_args()
        base_path = project_path(args.base_graph)
        output_path = output_path_for(args)
        if output_path.exists() and not args.force:
            raise RuntimeError(f"Refusing to overwrite existing output: {output_path}. Use --force.")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        od_counts, station_ids, base_metadata = load_base_graph(base_path)
        forward_counts, forward_metadata = topk_counts(
            od_counts,
            top_k=args.top_k,
            include_self=args.include_self,
        )
        reverse_counts, reverse_metadata = topk_counts(
            od_counts.T,
            top_k=args.top_k,
            include_self=args.include_self,
        )
        od_forward_support = row_normalize(forward_counts)
        od_reverse_support = row_normalize(reverse_counts)

        metadata = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source": {
                "base_graph": base_path.as_posix(),
                "base_graph_metadata": base_metadata,
            },
            "parameters": {
                "top_k": int(args.top_k),
                "include_self": bool(args.include_self),
            },
            "forward_graph": forward_metadata,
            "reverse_graph": reverse_metadata,
            "leakage_rule": "Top-k graphs are derived from the base training-period OD graph; no validation or test trips are added.",
        }
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
        np.savez_compressed(
            output_path,
            od_forward_support=od_forward_support,
            od_reverse_support=od_reverse_support,
            od_counts=od_counts.astype(np.int32, copy=False),
            od_forward_topk_counts=forward_counts.astype(np.int32),
            od_reverse_topk_counts=reverse_counts.astype(np.int32),
            station_ids=station_ids,
            metadata_json=np.asarray(metadata_text),
        )
        print(f"Wrote {output_path}")
        print(
            json.dumps(
                {
                    "top_k": args.top_k,
                    "include_self": args.include_self,
                    "forward": forward_metadata,
                    "reverse": reverse_metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
