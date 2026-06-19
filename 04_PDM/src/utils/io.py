"""
Small filesystem / serialization helpers.
=========================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exceptions import ManifestError


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, raising a clear error if missing/malformed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ManifestError(f"Malformed JSON at {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dict to JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def read_manifest(path: Path) -> list[str]:
    """Read a newline-delimited manifest of relative file stems.

    Raises ``ManifestError`` if the file is missing or empty so callers fail
    fast with an actionable message instead of an empty DataLoader.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(
            f"Manifest not found: {path}. Run the preprocessing/patch scripts first."
        )
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ManifestError(f"Manifest is empty: {path}")
    return lines
