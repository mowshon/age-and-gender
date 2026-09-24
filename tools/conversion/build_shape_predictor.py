#!/usr/bin/env python3
"""Export the pinned shape predictor and package it as a NumPy artifact.

Maintainer-only, PR-8 feasibility tooling. Hashes the source ``.dat``,
refuses anything but the pinned model, then runs
``age_and_gender_export_shape_predictor`` on *that same file* into a private
temporary directory and packages its flat binary + JSON output (see
``export_shape_predictor.cpp``) into a single ``.npz`` plus a
``manifest.json`` that :mod:`numpy_shape_predictor` loads. Running the
exporter here, rather than accepting a pre-exported directory, is what makes
the manifest's ``source.sha256`` a statement about the packaged parameters
instead of about an unrelated file. Follows ``build_onnx.py``'s provenance
conventions (source hash, converter revision, tool versions) without
depending on its ONNX-specific code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_NUM_PARTS = 5
SOURCE_SHA256 = "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_u32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u4")


def load_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


# The files that determine the artifact's bytes. The NumPy runtime modules
# and the probe are covered separately by the validation report's
# ``code_revision`` (see validate_shape_predictor.py).
CONVERTER_SOURCES = (
    "tools/conversion/export_shape_predictor.cpp",
    "tools/conversion/build_shape_predictor.py",
)


def converter_revision() -> str:
    digest = hashlib.sha256()
    for relative in CONVERTER_SOURCES:
        digest.update(relative.encode("ascii"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def build(exporter: Path, source_model: Path, output_dir: Path) -> dict[str, Any]:
    source_sha256 = sha256_file(source_model)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(f"unexpected source model SHA-256: {source_sha256}")
    with tempfile.TemporaryDirectory(prefix="shape-predictor-export-") as export_dir:
        subprocess.run(
            [str(exporter), "--model", str(source_model), "--output-dir", export_dir],
            check=True,
        )
        # Re-hash after export so a file swapped mid-build cannot be attributed.
        if sha256_file(source_model) != source_sha256:
            raise ValueError(f"{source_model} changed while it was being exported")
        return _package(Path(export_dir), source_model, source_sha256, output_dir)


def _package(
    export_dir: Path, source_model: Path, source_sha256: str, output_dir: Path
) -> dict[str, Any]:
    raw_manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    num_parts = int(raw_manifest["num_parts"])
    if num_parts != EXPECTED_NUM_PARTS:
        raise ValueError(f"expected a {EXPECTED_NUM_PARTS}-point predictor, got {num_parts}")
    num_cascades = int(raw_manifest["num_cascades"])
    cascade_num_features: list[int] = raw_manifest["cascade_num_features"]
    cascade_tree_splits: list[list[int]] = raw_manifest["cascade_tree_splits"]
    if len(cascade_num_features) != num_cascades or len(cascade_tree_splits) != num_cascades:
        raise ValueError("manifest cascade counts are inconsistent")

    initial_shape = load_f32(export_dir / "initial_shape.f32")
    if initial_shape.size != 2 * num_parts:
        raise ValueError("initial_shape size does not match num_parts")

    anchor_idx = load_u32(export_dir / "anchor_idx.u32")
    deltas_flat = load_f32(export_dir / "deltas.f32")
    total_features = sum(cascade_num_features)
    if anchor_idx.size != total_features or deltas_flat.size != 2 * total_features:
        raise ValueError("anchor_idx/deltas size does not match cascade_num_features")
    deltas = deltas_flat.reshape(total_features, 2)

    splits_idx1 = load_u32(export_dir / "splits_idx1.u32")
    splits_idx2 = load_u32(export_dir / "splits_idx2.u32")
    splits_thresh = load_f32(export_dir / "splits_thresh.f32")
    leaves_flat = load_f32(export_dir / "leaves.f32")

    total_splits = sum(sum(tree_splits) for tree_splits in cascade_tree_splits)
    total_leaves = sum(
        sum(count + 1 for count in tree_splits) for tree_splits in cascade_tree_splits
    )
    if (
        splits_idx1.size != total_splits
        or splits_idx2.size != total_splits
        or splits_thresh.size != total_splits
    ):
        raise ValueError("splits arrays do not match cascade_tree_splits")
    if leaves_flat.size != total_leaves * 2 * num_parts:
        raise ValueError("leaves array does not match cascade_tree_splits")
    leaves = leaves_flat.reshape(total_leaves, 2 * num_parts)

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "shape-predictor-v1.npz"
    np.savez(
        npz_path,
        initial_shape=initial_shape.reshape(num_parts, 2),
        anchor_idx=anchor_idx,
        deltas=deltas,
        splits_idx1=splits_idx1,
        splits_idx2=splits_idx2,
        splits_thresh=splits_thresh,
        leaves=leaves,
    )

    manifest = {
        "schema_version": 1,
        "bundle_id": "age-and-gender-shape-predictor-v1",
        "converter_revision": converter_revision(),
        "dlib": {"version": "19.20.0", "source": "tools/vendor/dlib"},
        "tool_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "source": {
            "filename": source_model.name,
            "sha256": source_sha256,
            "bytes": source_model.stat().st_size,
        },
        "artifact": {
            "filename": npz_path.name,
            "sha256": sha256_file(npz_path),
            "bytes": npz_path.stat().st_size,
        },
        "num_parts": num_parts,
        "num_cascades": num_cascades,
        "cascade_num_features": cascade_num_features,
        "cascade_tree_splits": cascade_tree_splits,
        "chip_extraction": {
            "canonical_points": [
                [0.8595674595992, 0.2134981538014],
                [0.6460604764104, 0.2289674387677],
                [0.1205750620789, 0.2137274526848],
                [0.3340850613712, 0.2290642403242],
                [0.4901123135679, 0.6277975316475],
            ],
            "padding": 0.2,
            "gender_size": 32,
            "age_size": 64,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exporter",
        type=Path,
        default=ROOT / "build/conversion/age_and_gender_export_shape_predictor",
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=ROOT / "src/age_and_gender/models/shape_predictor_5_face_landmarks.dat",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(args.exporter.resolve(), args.source_model.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
