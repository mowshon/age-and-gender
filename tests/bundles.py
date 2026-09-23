"""Helpers for building valid and deliberately broken model bundles."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODELS = ROOT / "src/age_and_gender/models"
CONVERSION_BUNDLE = ROOT / "tools/conversion/artifacts/v1"

Mutation = Callable[[dict[str, Any]], None]


def package_manifest() -> dict[str, Any]:
    """Return the installed bundle's manifest as a fresh mutable document."""
    return json.loads((PACKAGE_MODELS / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(destination: Path, manifest: Mapping[str, Any]) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def manifest_only(destination: Path, mutate: Mutation | None = None) -> Path:
    """Write just a manifest, for failures that are detected before any artifact."""
    manifest = package_manifest()
    if mutate is not None:
        mutate(manifest)
    return write_manifest(destination, manifest)


def full_bundle(destination: Path, mutate: Mutation | None = None) -> Path:
    """Copy the installed bundle, optionally rewriting its manifest."""
    shutil.copytree(PACKAGE_MODELS, destination)
    if mutate is not None:
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        mutate(manifest)
        write_manifest(destination, manifest)
    return destination


def corrupt_bytes(path: Path, offset: int = -1) -> None:
    """Flip one byte in place, keeping the file size unchanged."""
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))
