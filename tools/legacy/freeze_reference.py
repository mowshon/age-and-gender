#!/usr/bin/env python3
"""Create a deterministic checked golden from one oracle reference directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(
        (args.reference_dir / "oracle-report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (args.reference_dir / "reference-manifest.json").read_text(encoding="utf-8")
    )
    faces = []
    for face in report["faces"]:
        artifacts = {}
        for name, metadata in face["artifacts"].items():
            artifact_path = args.reference_dir / metadata["path"]
            artifact_sha256 = sha256_file(artifact_path)
            golden_name = f"{artifact_sha256}{artifact_path.suffix}"
            golden_path = args.output.parent / "artifacts" / golden_name
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            if not golden_path.exists():
                shutil.copyfile(artifact_path, golden_path)
            elif sha256_file(golden_path) != artifact_sha256:
                raise RuntimeError(f"content-addressed artifact is corrupt: {golden_path}")
            artifacts[name] = {
                **metadata,
                "sha256": artifact_sha256,
                "golden_path": f"artifacts/{golden_name}",
            }
        faces.append(
            {
                "rectangle": face["rectangle"],
                "landmarks": face["landmarks"],
                "artifacts": artifacts,
                "gender_logits": face["gender_logits"],
                "gender_probabilities": face["gender_probabilities"],
                "age_logits": face["age_logits"],
                "age_probabilities": face["age_probabilities"],
                "age_expectation": face["age_expectation"],
                "result": face["result"],
            }
        )
    golden = {
        "schema_version": 1,
        "repository_revision": manifest["repository"]["revision"],
        "input": manifest["input"],
        "models": manifest["models"],
        "oracle": {
            key: manifest["oracle"][key]
            for key in (
                "source",
                "build_recipe",
                "source_sha256",
                "build_recipe_sha256",
                "root_cmake_sha256",
                "executable_sha256",
                "version_output",
            )
        },
        "machine": manifest["machine"],
        "box_mode": report["box_mode"],
        "input_boxes_trbl": report["input_boxes_trbl"],
        "models_runtime": report["models"],
        "faces": faces,
        "results": report["results"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
