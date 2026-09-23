#!/usr/bin/env python3
"""Decode a canonical input, run the legacy oracle, and hash every artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_HASHES = {
    "dnn_age_predictor_v1.dat": "4b78d4d7055e22620e362884b5551caa9379080277338aa2d4cdfc592f0e9fa3",
    "dnn_gender_classifier_v1.dat": (
        "85453d6f6585c8e02ada95929956783c780dc04dcec5bdfd14af82f15c99ba41"
    ),
    "shape_predictor_5_face_landmarks.dat": (
        "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888"
    ),
}

CANONICAL_INPUTS = {
    "test-image.jpg": {
        "source_sha256": "0a3eb36690b3ae319939e534b6c4cd00def8045af8c12d62829d6d36ec7746f4",
        "rgb_sha256": "a4a49f6bdf3b93b56cafd4421d19314cb8e8f76f184496a07a220f6020d4f8b9",
        "size": [1100, 825],
    },
    "test-image-2.jpg": {
        "source_sha256": "e060bdc12cd53020c104ad305324c69832ec4720c2709e525be5975aa7877db9",
        "rgb_sha256": "94e936c478ed256ca699233d019af72f964019ffa4e6fbcabcf9b2504d4fc9ed",
        "size": [634, 435],
    },
    "dogs.jpg": {
        "source_sha256": "66e22f8c3bd3b8f876ad9158caaa064992d2b56a5681d9c6efa163a59db7ed03",
        "rgb_sha256": "594e39d3ecee106f1fce1e21d307e81e6f609ba1e5e3f5b25d81d7559734e246",
        "size": [900, 916],
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def machine_metadata() -> dict[str, Any]:
    cpu_flags = ""
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text(encoding="utf-8").splitlines():
            if line.startswith("flags") or line.startswith("Features"):
                cpu_flags = line.partition(":")[2].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_flags": cpu_flags.split(),
        "environment_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument(
        "--box",
        action="append",
        default=[],
        help="Explicit top,right,bottom,left rectangle; repeat to preserve ordering",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    expected_input = CANONICAL_INPUTS.get(args.image.name)
    if expected_input is None:
        raise RuntimeError(f"unregistered input: {args.image.name}")
    check_hash(args.image, expected_input["source_sha256"])
    for model_name, expected_hash in MODEL_HASHES.items():
        check_hash(args.models / model_name, expected_hash)

    args.output.mkdir(parents=True, exist_ok=False)
    with Image.open(args.image) as image:
        rgb_image = image.convert("RGB")
        if list(rgb_image.size) != expected_input["size"]:
            raise RuntimeError(f"unexpected dimensions for {args.image}")
        rgb = rgb_image.tobytes()
    if sha256_bytes(rgb) != expected_input["rgb_sha256"]:
        raise RuntimeError("decoded RGB differs from the frozen Pillow 12.3.0 reference")
    raw_path = args.output / "input.rgb"
    raw_path.write_bytes(rgb)

    command = [
        str(args.oracle.resolve()),
        "--image",
        str(raw_path.resolve()),
        "--width",
        str(expected_input["size"][0]),
        "--height",
        str(expected_input["size"][1]),
        "--models",
        str(args.models.resolve()),
        "--output",
        str(args.output.resolve()),
        "--benchmark-runs",
        str(args.benchmark_runs),
    ]
    for box in args.box:
        command.extend(("--box", box))
    subprocess.run(command, cwd=root, check=True)

    oracle_version = subprocess.check_output(
        [str(args.oracle.resolve()), "--version"], text=True
    ).splitlines()
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "reference-manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "repository": {
            "revision": git_value(root, "rev-parse", "HEAD"),
            "dirty": bool(git_value(root, "status", "--short")),
        },
        "input": {
            "source": str(args.image.resolve().relative_to(root)),
            **expected_input,
            "decoder": f"Pillow {Image.__version__}",
        },
        "models": {
            name: {"sha256": digest, "bytes": (args.models / name).stat().st_size}
            for name, digest in MODEL_HASHES.items()
        },
        "oracle": {
            "source": "tools/legacy/oracle.cpp",
            "network_definitions": "tools/network_definitions.h",
            "build_recipe": "tools/legacy/CMakeLists.txt",
            "source_sha256": sha256_file(root / "tools/legacy/oracle.cpp"),
            "network_definitions_sha256": sha256_file(
                root / "tools/network_definitions.h"
            ),
            "build_recipe_sha256": sha256_file(root / "tools/legacy/CMakeLists.txt"),
            "root_cmake_sha256": sha256_file(
                root / "tools/legacy/original-extension/CMakeLists.txt"
            ),
            "executable_sha256": sha256_file(args.oracle),
            "version_output": oracle_version,
            "command": command,
        },
        "machine": machine_metadata(),
        "artifacts": artifacts,
    }
    (args.output / "reference-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
