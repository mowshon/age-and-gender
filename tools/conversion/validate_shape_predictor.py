#!/usr/bin/env python3
"""Validate the NumPy shape-predictor/chip-extraction port.

Maintainer-only, PR-8 feasibility tooling. Writes one JSON report whose
top-level ``passed`` is true only if **every** section below ran and passed
(a section that cannot run is a failure, not a skip):

1. ``frozen_corpus``: exact landmarks and chips for every face in
   ``tests/fixtures/legacy/*.golden.json`` (spec/PR-1.md's exact-equality
   thresholds). A missing corpus image is a failure.
2. ``cascade_stages``: bit-exact per-cascade feature values and shape
   estimates against ``age_and_gender_probe_shape_predictor`` (``--probe``
   is required). With ``--write-stage-fixture`` the oracle's trace is also
   written as a compact fixture the CI test replays without the probe.
3. ``regression_cases``: every case in ``regression-cases.json`` — synthetic
   inputs on which an earlier version of the port disagreed with dlib — must
   now match live ``dlib-bin`` exactly.
4. ``live_dlib_synthetic``: an exact landmark + both-chip sweep against live
   ``dlib-bin`` over reproducible boundary/deep-pyramid cases
   (:func:`shape_predictor_evidence.synthetic_case`). Every failure is
   recorded as a ``(seed, index)`` pair that regenerates the input.
5. ``chip_geometry``: bit-exact ``chip_details`` rect/angle against
   ``dlib.get_face_chip_details`` on random landmark sets, directly testing
   the ported similarity transform/SVD rather than only its rounded outputs.
6. ``chip_fast_path``: zero-angle rectangles with fractional and integral
   sizes, covering ``extract_image_chip``'s raw-copy bypass decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import dlib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "tools/conversion"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from numpy_chip_extraction import (  # noqa: E402
    ChipDetails,
    extract_image_chip,
    get_face_chip_details,
    pyramid_depth,
)
from numpy_shape_predictor import (  # noqa: E402
    ShapePredictor,
    _evaluate_tree,
    _extract_feature_pixel_values,
    _grayscale_intensity,
    load_shape_predictor_artifact,
)
from shape_predictor_evidence import (  # noqa: E402
    ARTIFACT_DIR,
    SOURCE_MODEL,
    STAGE_TRACE_FIXTURE,
    artifact_sha256,
    code_revision,
    environment_info,
    load_regression_cases,
    synthetic_case,
)
from tests.golden import golden_documents  # noqa: E402

CHIP_SIZES = {"gender": 32, "age": 64}
PADDING = 0.2
MAX_LISTED_FAILURES = 50


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(np.array(image.convert("RGB")))


def _section(checked: int, failures: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "checked": checked,
        "failure_count": len(failures),
        "failures": failures[:MAX_LISTED_FAILURES],
        "passed": checked > 0 and not failures,
        **extra,
    }


# -- live-dlib comparison of one (image, rectangle) case ---------------------

_WORKER: dict[str, Any] = {}


def _init_worker(artifact_dir: str) -> None:
    _WORKER["numpy"] = load_shape_predictor_artifact(artifact_dir)
    _WORKER["dlib"] = dlib.shape_predictor(str(SOURCE_MODEL))


def compare_case(
    predictor: ShapePredictor,
    dlib_predictor: dlib.shape_predictor,
    image: np.ndarray,
    rectangle: tuple[int, int, int, int],
) -> tuple[str | None, int]:
    """Exact landmark + both-chip comparison with live dlib.

    Returns a mismatch description (``None`` if exact) and the deepest
    pyramid descent the case's chips exercised.
    """
    dlib_shape = dlib_predictor(image, dlib.rectangle(*rectangle))
    dlib_landmarks = np.array(
        [[dlib_shape.part(k).x, dlib_shape.part(k).y] for k in range(dlib_shape.num_parts)]
    )
    numpy_landmarks = predictor(image, rectangle)
    if not np.array_equal(dlib_landmarks, numpy_landmarks):
        return f"landmarks {numpy_landmarks.tolist()} != dlib {dlib_landmarks.tolist()}", 0
    deepest = 0
    for task, size in CHIP_SIZES.items():
        dlib_chip = np.asarray(
            dlib.extract_image_chip(image, dlib.get_face_chip_details(dlib_shape, size, PADDING))
        )
        details = get_face_chip_details(numpy_landmarks.astype(np.float64), size, PADDING)
        deepest = max(deepest, pyramid_depth(details)[0])
        chip = extract_image_chip(image, details)
        if not np.array_equal(dlib_chip, chip):
            differing = int((dlib_chip != chip).sum())
            return f"{task} chip differs in {differing} channel values", deepest
    return None, deepest


def _run_synthetic(job: tuple[int, int]) -> tuple[int, bool, int, str | None]:
    seed, index = job
    image, rectangle = synthetic_case(seed, index)
    left, top, right, bottom = rectangle
    boundary = left < 0 or top < 0 or right >= image.shape[1] or bottom >= image.shape[0]
    problem, depth = compare_case(_WORKER["numpy"], _WORKER["dlib"], image, rectangle)
    return index, boundary, depth, problem


# -- sections -----------------------------------------------------------------


def check_frozen_corpus(predictor: ShapePredictor) -> dict[str, Any]:
    failures: list[str] = []
    documents = golden_documents()
    faces_checked = 0
    for document in documents:
        source_path = ROOT / document.source_path
        if not source_path.is_file():
            failures.append(f"{document.name}: source image {document.source_path} is missing")
            continue
        array = _read_rgb(source_path)
        for face in document.faces:
            faces_checked += 1
            landmarks = predictor(array, tuple(face.rectangle))
            if landmarks.tolist() != face.landmarks:
                failures.append(
                    f"{document.name}:{face.index} landmarks {landmarks.tolist()} != "
                    f"golden {face.landmarks}"
                )
                continue
            for task, size in CHIP_SIZES.items():
                details = get_face_chip_details(landmarks.astype(np.float64), size, PADDING)
                chip = extract_image_chip(array, details)
                golden_chip = face.chip(task)
                if not np.array_equal(chip, golden_chip):
                    diff = int((chip != golden_chip).sum())
                    failures.append(
                        f"{document.name}:{face.index} {task} chip differs in {diff} values"
                    )
    return _section(faces_checked, failures, documents_checked=len(documents))


def check_cascade_stages(
    predictor: ShapePredictor, probe: Path, work_dir: Path, fixture_path: Path | None
) -> dict[str, Any]:
    """Per-cascade stage agreement against the compiled dlib stage-trace probe."""
    failures: list[str] = []
    fixture_faces: list[dict[str, Any]] = []
    stages_checked = 0
    for document in golden_documents():
        source_path = ROOT / document.source_path
        if not document.faces:
            continue
        if not source_path.is_file():
            failures.append(f"{document.name}: source image {document.source_path} is missing")
            continue
        array = _read_rgb(source_path)
        raw_path = work_dir / f"{document.name}.rgb"
        array.tofile(raw_path)
        height, width = array.shape[:2]
        intensity = _grayscale_intensity(array)
        for face in document.faces:
            left, top, right, bottom = face.rectangle
            stage_dir = work_dir / f"{document.name}-face{face.index}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    str(probe),
                    "--model",
                    str(SOURCE_MODEL),
                    "--image",
                    str(raw_path),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--rect",
                    f"{left},{top},{right},{bottom}",
                    "--output-dir",
                    str(stage_dir),
                ],
                check=True,
            )
            initial_shape = predictor._initial_shape
            current_shape = initial_shape.astype(np.float32).copy()
            fixture_stages: list[dict[str, str]] = []
            for stage_index, cascade in enumerate(predictor._cascades):
                features = _extract_feature_pixel_values(
                    intensity, left, top, right, bottom, current_shape, initial_shape, cascade
                )
                oracle_features = np.fromfile(
                    stage_dir / f"stage-{stage_index}-features.f32", dtype="<f4"
                )
                if not np.array_equal(features, oracle_features):
                    failures.append(
                        f"{document.name}:{face.index} cascade {stage_index} features differ"
                    )
                for idx1, idx2, thresh, leaves in zip(
                    cascade.tree_splits_idx1,
                    cascade.tree_splits_idx2,
                    cascade.tree_splits_thresh,
                    cascade.tree_leaves,
                    strict=True,
                ):
                    leaf = _evaluate_tree(features, idx1, idx2, thresh, leaves)
                    current_shape = (current_shape + leaf.reshape(predictor.num_parts, 2)).astype(
                        np.float32
                    )
                oracle_shape = np.fromfile(
                    stage_dir / f"stage-{stage_index}-shape.f32", dtype="<f4"
                ).reshape(predictor.num_parts, 2)
                if not np.array_equal(current_shape, oracle_shape):
                    failures.append(
                        f"{document.name}:{face.index} cascade {stage_index} shape differs"
                    )
                fixture_stages.append(
                    {
                        "features_sha256": hashlib.sha256(
                            oracle_features.astype("<f4").tobytes()
                        ).hexdigest(),
                        "shape_f32_hex": oracle_shape.astype("<f4").tobytes().hex(),
                    }
                )
                stages_checked += 1
            fixture_faces.append(
                {
                    "document": document.name,
                    "face": face.index,
                    "rectangle": [left, top, right, bottom],
                    "stages": fixture_stages,
                }
            )
    if fixture_path is not None and not failures:
        fixture = {
            "schema_version": 1,
            "description": (
                "Per-cascade dlib oracle trace (age_and_gender_probe_shape_predictor) for "
                "every frozen-corpus face: SHA-256 of each stage's little-endian float32 "
                "feature values and the float32 shape estimate after that stage."
            ),
            "artifact_sha256": artifact_sha256(),
            "faces": fixture_faces,
        }
        fixture_path.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    return _section(stages_checked, failures, faces_traced=len(fixture_faces))


def check_regression_cases(predictor: ShapePredictor) -> dict[str, Any]:
    dlib_predictor = dlib.shape_predictor(str(SOURCE_MODEL))
    failures: list[str] = []
    cases = load_regression_cases()
    for case in cases:
        image, rectangle = synthetic_case(case["seed"], case["index"])
        problem, _ = compare_case(predictor, dlib_predictor, image, rectangle)
        if problem is not None:
            failures.append(f"seed={case['seed']} index={case['index']} {rectangle}: {problem}")
    return _section(len(cases), failures)


def check_live_dlib_synthetic(
    artifact_dir: Path, seed: int, count: int, workers: int
) -> dict[str, Any]:
    failures: list[str] = []
    failing_cases: list[dict[str, int]] = []
    boundary_cases = 0
    depth_counts: dict[int, int] = {}
    jobs = [(seed, index) for index in range(count)]
    with multiprocessing.Pool(
        workers, initializer=_init_worker, initargs=(str(artifact_dir),)
    ) as pool:
        for index, boundary, depth, problem in pool.imap_unordered(
            _run_synthetic, jobs, chunksize=16
        ):
            boundary_cases += boundary
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            if problem is not None:
                failing_cases.append({"seed": seed, "index": index})
                failures.append(f"seed={seed} index={index}: {problem}")
    failing_cases.sort(key=lambda case: case["index"])
    return _section(
        count,
        sorted(failures),
        seed=seed,
        boundary_cases=boundary_cases,
        max_pyramid_depth=max(depth_counts, default=0),
        cases_by_max_pyramid_depth={
            str(depth): depth_counts[depth] for depth in sorted(depth_counts)
        },
        failing_cases=failing_cases,
    )


def check_chip_geometry(seed: int, count: int) -> dict[str, Any]:
    """Bit-exact ``chip_details`` (rect and angle) against ``dlib.get_face_chip_details``."""
    rng = np.random.default_rng(seed)
    failures: list[str] = []
    for _ in range(count):
        base = rng.uniform(-200, 1500, 2)
        spread = rng.uniform(5, 800) * rng.uniform(0.05, 1)
        points = np.rint(base + rng.normal(size=(5, 2)) * spread).astype(np.int64)
        shape = dlib.full_object_detection(
            dlib.rectangle(0, 0, 10, 10), [dlib.point(int(x), int(y)) for x, y in points]
        )
        for size in CHIP_SIZES.values():
            expected = dlib.get_face_chip_details(shape, size, PADDING)
            actual = get_face_chip_details(points.astype(np.float64), size, PADDING)
            expected_rect = (
                expected.rect.left(),
                expected.rect.top(),
                expected.rect.right(),
                expected.rect.bottom(),
            )
            if expected_rect != actual.rect or expected.angle != actual.angle:
                failures.append(
                    f"points={points.tolist()} size={size}: rect {actual.rect} angle "
                    f"{actual.angle!r} != dlib {expected_rect} {expected.angle!r}"
                )
    return _section(count * len(CHIP_SIZES), failures, seed=seed)


FAST_PATH_RECTS = (
    # (rect, rows, cols): integral heights take dlib's raw-copy bypass,
    # fractional ones must interpolate (review finding: (0.2, 0.2, 2.4, 2.4)).
    ((0.2, 0.2, 2.4, 2.4), 3, 3),
    ((0.2, 0.2, 2.2, 2.2), 3, 3),
    ((0.6, 0.6, 2.6, 2.6), 3, 3),
    ((0.5, 0.5, 2.5, 2.5), 3, 3),
    ((-1.3, 4.2, 1.3, 7.0), 3, 3),
    ((3.5, 3.5, 5.5, 5.5), 3, 3),
    ((10.0, 12.0, 41.0, 43.0), 32, 32),
    ((10.25, 12.0, 41.25, 43.0), 32, 32),
    ((10.3, 12.1, 41.2, 43.3), 32, 32),
)


def check_chip_fast_path(seed: int) -> dict[str, Any]:
    image = np.random.default_rng(seed).integers(0, 256, size=(60, 70, 3), dtype=np.uint8)
    failures: list[str] = []
    for rect, rows, cols in FAST_PATH_RECTS:
        expected = np.asarray(
            dlib.extract_image_chip(
                image, dlib.chip_details(dlib.drectangle(*rect), dlib.chip_dims(rows, cols))
            )
        )
        actual = extract_image_chip(image, ChipDetails(rect=rect, angle=0.0, rows=rows, cols=cols))
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            failures.append(f"rect={rect} {rows}x{cols}: chip differs from dlib")
    return _section(len(FAST_PATH_RECTS), failures)


# -- driver -------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--probe",
        type=Path,
        required=True,
        help="Path to age_and_gender_probe_shape_predictor (the cascade-stage oracle).",
    )
    parser.add_argument("--synthetic-cases", type=int, default=10000)
    parser.add_argument("--synthetic-seed", type=int, default=2026)
    parser.add_argument("--geometry-cases", type=int, default=50000)
    parser.add_argument("--geometry-seed", type=int, default=2027)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--write-stage-fixture",
        action="store_true",
        help=f"Also write the oracle stage trace to {STAGE_TRACE_FIXTURE.relative_to(ROOT)}.",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = load_shape_predictor_artifact(args.artifact)

    report: dict[str, Any] = {
        "schema_version": 2,
        "code_revision": code_revision(),
        "artifact_sha256": artifact_sha256(),
        "environment": environment_info(),
    }
    sections: dict[str, dict[str, Any]] = {}
    sections["frozen_corpus"] = check_frozen_corpus(predictor)
    with tempfile.TemporaryDirectory(prefix="shape-predictor-stages-") as tmp:
        sections["cascade_stages"] = check_cascade_stages(
            predictor,
            args.probe,
            Path(tmp),
            STAGE_TRACE_FIXTURE if args.write_stage_fixture else None,
        )
    sections["regression_cases"] = check_regression_cases(predictor)
    sections["live_dlib_synthetic"] = check_live_dlib_synthetic(
        args.artifact, args.synthetic_seed, args.synthetic_cases, args.workers
    )
    sections["chip_geometry"] = check_chip_geometry(args.geometry_seed, args.geometry_cases)
    sections["chip_fast_path"] = check_chip_fast_path(args.geometry_seed)

    report["sections"] = sections
    report["passed"] = all(section["passed"] is True for section in sections.values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
