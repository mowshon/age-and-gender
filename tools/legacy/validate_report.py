#!/usr/bin/env python3
"""Validate canonical oracle output against the checked fixture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"legacy report validation failed: {message}")


def validate_probabilities(values: list[float], size: int, label: str) -> None:
    if len(values) != size:
        fail(f"{label} shape: expected {size}, got {len(values)}")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        fail(f"{label} contains a non-finite or out-of-range value")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6, rel_tol=1e-4):
        fail(f"{label} does not sum to one: {sum(values)}")


def compare_vector(
    expected: list[float],
    actual: list[float],
    *,
    image: str,
    face_index: int,
    model: str,
    atol: float,
    rtol: float,
) -> None:
    if len(expected) != len(actual):
        fail(
            f"{image} face {face_index} {model}: expected shape ({len(expected)},), "
            f"actual shape ({len(actual)},), max error unavailable, first mismatch 0"
        )
    maximum_error = 0.0
    first_mismatch = None
    for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
        error = abs(expected_value - actual_value)
        maximum_error = max(maximum_error, error)
        if (
            first_mismatch is None
            and (
                not math.isfinite(expected_value)
                or not math.isfinite(actual_value)
                or error > atol + rtol * abs(expected_value)
            )
        ):
            first_mismatch = index
    if first_mismatch is not None:
        fail(
            f"{image} face {face_index} {model}: shape ({len(expected)},), "
            f"max error {maximum_error}, first mismatch {first_mismatch}"
        )


def compare_exact_artifact(
    expected_path: Path,
    actual_path: Path,
    *,
    image: str,
    face_index: int,
    model: str,
    dtype: str,
    shape: list[int],
) -> None:
    expected_bytes = expected_path.read_bytes()
    actual_bytes = actual_path.read_bytes()
    if expected_bytes == actual_bytes:
        return
    if dtype == "uint8":
        expected_values = expected_bytes
        actual_values = actual_bytes
    else:
        expected_values = [value[0] for value in struct.iter_unpack("<f", expected_bytes)]
        actual_values = [value[0] for value in struct.iter_unpack("<f", actual_bytes)]
    first_mismatch = next(
        (
            index
            for index, values in enumerate(zip(expected_values, actual_values))
            if values[0] != values[1]
        ),
        min(len(expected_values), len(actual_values)),
    )
    maximum_error = max(
        (
            abs(float(expected) - float(actual))
            for expected, actual in zip(expected_values, actual_values)
        ),
        default=math.inf,
    )
    fail(
        f"{image} face {face_index} {model}: expected shape {tuple(shape)}, "
        f"actual shape {tuple(shape)}, max error {maximum_error}, "
        f"first mismatch {first_mismatch}"
    )


def expected_input(manifest: dict[str, Any], rgb_sha256: str) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["inputs"]
        if item["decoded_rgb_sha256"] == rgb_sha256
    ]
    if len(matches) != 1:
        fail(f"no unique fixture for decoded RGB SHA-256 {rgb_sha256}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/legacy/manifest.json"),
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw_path = args.report.parent / "input.rgb"
    if not raw_path.exists():
        raw_path = Path(report["input"]["path"])
    rgb_sha256 = sha256_file(raw_path)
    fixture = expected_input(manifest, rgb_sha256)
    expected = fixture["expected"]
    golden_path = Path(fixture["golden"])
    if report["box_mode"] == "explicit":
        matching_cases = [
            case
            for case in manifest["explicit_box_cases"]
            if case["input"] == fixture["path"]
            and case["boxes_trbl"] == report["input_boxes_trbl"]
        ]
        if len(matching_cases) != 1:
            fail("no unique explicit-box fixture matches the report")
        expected = matching_cases[0]["expected"]
        golden_path = Path(matching_cases[0]["golden"])
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    reference_manifest_path = args.report.parent / "reference-manifest.json"
    if not reference_manifest_path.exists():
        fail("reference-manifest.json is missing beside the report")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    for key in ("source_sha256", "build_recipe_sha256", "root_cmake_sha256"):
        if reference_manifest["oracle"][key] != golden["oracle"][key]:
            fail(f"oracle provenance {key} differs from golden")
    if reference_manifest["models"] != golden["models"]:
        fail("model provenance differs from golden")
    if report["models"] != golden["models_runtime"]:
        fail("serialized model input means differ from golden")
    if rgb_sha256 != fixture["decoded_rgb_sha256"]:
        fail("decoded RGB SHA-256 differs")
    if len(report["faces"]) != len(expected):
        fail(f"expected {len(expected)} faces, got {len(report['faces'])}")
    if len(golden["faces"]) != len(report["faces"]):
        fail("golden and actual face counts differ")

    for index, (actual, expected_face, golden_face) in enumerate(
        zip(report["faces"], expected, golden["faces"])
    ):
        if actual["rectangle"] != expected_face["rectangle"]:
            fail(f"face {index} rectangle differs")
        if actual["landmarks"] != expected_face["landmarks"]:
            fail(f"face {index} landmarks differ")
        if actual["result"] != expected_face["result"]:
            fail(f"face {index} public result differs")
        if (
            "age_expectation" in expected_face
            and abs(actual["age_expectation"] - expected_face["age_expectation"]) > 1e-4
        ):
            fail(f"face {index} age expectation differs")
        if abs(actual["age_expectation"] - golden_face["age_expectation"]) > 1e-4:
            fail(f"face {index} age expectation differs from golden")
        validate_probabilities(
            actual["gender_probabilities"], 2, f"face {index} gender probabilities"
        )
        validate_probabilities(
            actual["age_probabilities"], 81, f"face {index} age probabilities"
        )
        compare_vector(
            golden_face["gender_logits"],
            actual["gender_logits"],
            image=fixture["path"],
            face_index=index,
            model="gender logits",
            atol=1e-5,
            rtol=1e-4,
        )
        compare_vector(
            golden_face["age_logits"],
            actual["age_logits"],
            image=fixture["path"],
            face_index=index,
            model="age logits",
            atol=1e-5,
            rtol=1e-4,
        )
        compare_vector(
            golden_face["gender_probabilities"],
            actual["gender_probabilities"],
            image=fixture["path"],
            face_index=index,
            model="gender probabilities",
            atol=1e-6,
            rtol=1e-4,
        )
        compare_vector(
            golden_face["age_probabilities"],
            actual["age_probabilities"],
            image=fixture["path"],
            face_index=index,
            model="age probabilities",
            atol=1e-6,
            rtol=1e-4,
        )
        if set(actual["artifacts"]) != set(golden_face["artifacts"]):
            fail(f"face {index} artifact names differ from golden")
        for artifact_name, artifact in actual["artifacts"].items():
            artifact_path = args.report.parent / artifact["path"]
            element_count = math.prod(artifact["shape"])
            bytes_per_element = {"uint8": 1, "float32-le": 4}.get(artifact["dtype"])
            if bytes_per_element is None:
                fail(f"{artifact_path.name} has unsupported dtype {artifact['dtype']}")
            expected_bytes = element_count * bytes_per_element
            if artifact_path.stat().st_size != expected_bytes:
                fail(f"{artifact_path.name} has the wrong byte count")
            expected_artifact = golden_face["artifacts"][artifact_name]
            if {
                key: artifact[key] for key in ("dtype", "shape")
            } != {key: expected_artifact[key] for key in ("dtype", "shape")}:
                fail(f"{artifact_path.name} metadata differs from golden")
            if sha256_file(artifact_path) != expected_artifact["sha256"]:
                compare_exact_artifact(
                    golden_path.parent / expected_artifact["golden_path"],
                    artifact_path,
                    image=fixture["path"],
                    face_index=index,
                    model=artifact_name,
                    dtype=artifact["dtype"],
                    shape=artifact["shape"],
                )

    if report["results"] != [item["result"] for item in expected]:
        fail("top-level public results differ")
    print(f"legacy report passed for {len(report['faces'])} face(s)")


if __name__ == "__main__":
    main()
