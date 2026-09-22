#!/usr/bin/env python3
"""Compare two independently generated oracle reference directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first_report = json.loads((args.first / "oracle-report.json").read_text())
    second_report = json.loads((args.second / "oracle-report.json").read_text())
    stable_report_fields = ("models", "input_boxes_trbl", "faces", "results")
    mismatched_fields = [
        field
        for field in stable_report_fields
        if first_report[field] != second_report[field]
    ]
    first_artifact_names = {
        path.name
        for path in args.first.iterdir()
        if path.suffix in {".rgb", ".f32"}
    }
    second_artifact_names = {
        path.name
        for path in args.second.iterdir()
        if path.suffix in {".rgb", ".f32"}
    }
    artifact_names = sorted(first_artifact_names & second_artifact_names)
    artifact_hashes = {
        name: {
            "first": sha256_file(args.first / name),
            "second": sha256_file(args.second / name),
        }
        for name in artifact_names
    }
    mismatched_artifacts = [
        name for name, hashes in artifact_hashes.items() if len(set(hashes.values())) != 1
    ]
    result = {
        "schema_version": 1,
        "first_manifest": json.loads(
            (args.first / "reference-manifest.json").read_text(encoding="utf-8")
        ),
        "second_manifest": json.loads(
            (args.second / "reference-manifest.json").read_text(encoding="utf-8")
        ),
        "compared_report_fields": list(stable_report_fields),
        "mismatched_report_fields": mismatched_fields,
        "artifact_hashes": artifact_hashes,
        "first_only_artifacts": sorted(first_artifact_names - second_artifact_names),
        "second_only_artifacts": sorted(second_artifact_names - first_artifact_names),
        "mismatched_artifacts": mismatched_artifacts,
        "stable": not (
            mismatched_fields
            or mismatched_artifacts
            or first_artifact_names != second_artifact_names
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["stable"]:
        raise SystemExit("legacy references are not stable")


if __name__ == "__main__":
    main()
