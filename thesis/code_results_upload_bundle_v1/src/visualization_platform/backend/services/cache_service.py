"""JSON cache helpers for decision-level backend responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import CACHE_DIR

CACHE_VERSION = "v2"


class CacheService:
    """Store per-decision API responses on disk."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, *, ts: str, model: str, algorithm: str, cap: int) -> str:
        """Build a stable cache key."""
        digest = hashlib.sha1(f"{CACHE_VERSION}|{ts}|{model}|{algorithm}|{cap}".encode("utf-8")).hexdigest()[:16]
        return f"{digest}.json"

    def read(self, *, ts: str, model: str, algorithm: str, cap: int) -> dict[str, Any] | None:
        """Read a cached decision response if present."""
        path = self.cache_dir / self.key(ts=ts, model=model, algorithm=algorithm, cap=cap)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, *, ts: str, model: str, algorithm: str, cap: int, payload: dict[str, Any]) -> None:
        """Write a cached decision response."""
        path = self.cache_dir / self.key(ts=ts, model=model, algorithm=algorithm, cap=cap)
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
