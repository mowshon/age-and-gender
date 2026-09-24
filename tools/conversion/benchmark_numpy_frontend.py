#!/usr/bin/env python3
"""Time the NumPy shape-predictor/chip-extraction port against the wheel.

Maintainer-only, PR-8 feasibility tooling: spec/PR-8.md's "Initial
feasibility deliverable" item 4 asks to "time this path against the wheel
frontend, using the PR-6 methodology." Follows spec/PR-6.md's steady-state
protocol (10 warmups, 100 measured iterations, five repeats by default) and
``benchmarks/benchmark_pipeline.py``'s ``time_case``/``summarize`` shape,
duplicated locally rather than imported so this tool and the shipped-package
benchmark suite stay independently runnable.

Each wheel/NumPy pair times *equivalent work* from identical inputs (the same
decoded RGB array and rectangle, or the same landmarks):

- ``landmarks``: one shape-predictor call.
- ``chip_gender``/``chip_age``: ``get_face_chip_details`` **and**
  ``extract_image_chip`` inside the timed call on both sides (the wheel's
  ``FaceFrontend._chip`` does both, so the NumPy side must too).
- ``face_frontend``: landmarks followed by both chips — the whole per-face
  work this port would replace.

The report records machine, dependency, and thread metadata, plus the
prototype ``code_revision`` so a stale benchmark is detectable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import dlib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "tools/conversion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from numpy_chip_extraction import extract_image_chip, get_face_chip_details  # noqa: E402
from numpy_shape_predictor import load_shape_predictor_artifact  # noqa: E402
from shape_predictor_evidence import (  # noqa: E402
    ARTIFACT_DIR,
    artifact_sha256,
    code_revision,
    environment_info,
)

from age_and_gender._faces import FaceFrontend  # noqa: E402
from age_and_gender._models import bundled_models  # noqa: E402

CHIP_SIZES = {"gender": 32, "age": 64}
PADDING = 0.2
CASES = ("landmarks", "chip_gender", "chip_age", "face_frontend")


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"count": 0}
    ordered = sorted(samples_ms)
    return {
        "count": len(ordered),
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": _percentile(ordered, 0.95),
        "max_ms": ordered[-1],
        "stdev_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def time_case(fn, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    per_repeat_medians = []
    pooled: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000.0)
        per_repeat_medians.append(statistics.median(samples))
        pooled.extend(samples)
    summary = summarize(pooled)
    summary["repeats"] = repeats
    summary["warmup_per_repeat"] = warmup
    summary["iterations_per_repeat"] = iterations
    summary["per_repeat_median_ms"] = per_repeat_medians
    return summary


def load_reference_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    with Image.open(ROOT / "example/test-image.jpg") as image:
        array = np.ascontiguousarray(np.array(image.convert("RGB")))
    # spec/INVESTIGATION.md's first reference face.
    return array, (419, 266, 506, 352)


def run(warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    image, rectangle = load_reference_image()

    frontend = FaceFrontend(bundled_models())
    numpy_predictor = load_shape_predictor_artifact(ARTIFACT_DIR)
    dlib_rect = dlib.rectangle(*rectangle)

    dlib_shape = frontend._predictor(image, dlib_rect)
    numpy_landmarks = numpy_predictor(image, rectangle)
    wheel_landmarks = [[dlib_shape.part(k).x, dlib_shape.part(k).y] for k in range(5)]
    if numpy_landmarks.tolist() != wheel_landmarks:
        raise RuntimeError("NumPy and wheel landmarks differ; timings would not be comparable")
    landmarks_f64 = numpy_landmarks.astype(np.float64)
    for size in CHIP_SIZES.values():
        wheel_chip = frontend._chip(image, dlib_shape, size)
        numpy_chip = extract_image_chip(image, get_face_chip_details(landmarks_f64, size, PADDING))
        if not np.array_equal(wheel_chip, numpy_chip):
            raise RuntimeError(f"NumPy and wheel {size}px chips differ")

    def numpy_chip(size: int) -> np.ndarray:
        return extract_image_chip(image, get_face_chip_details(landmarks_f64, size, PADDING))

    def wheel_face() -> None:
        shape = frontend._predictor(image, dlib_rect)
        for size in CHIP_SIZES.values():
            frontend._chip(image, shape, size)

    def numpy_face() -> None:
        landmarks = numpy_predictor(image, rectangle).astype(np.float64)
        for size in CHIP_SIZES.values():
            extract_image_chip(image, get_face_chip_details(landmarks, size, PADDING))

    pairs = {
        "landmarks": (
            lambda: frontend._predictor(image, dlib_rect),
            lambda: numpy_predictor(image, rectangle),
        ),
        "chip_gender": (
            lambda: frontend._chip(image, dlib_shape, CHIP_SIZES["gender"]),
            lambda: numpy_chip(CHIP_SIZES["gender"]),
        ),
        "chip_age": (
            lambda: frontend._chip(image, dlib_shape, CHIP_SIZES["age"]),
            lambda: numpy_chip(CHIP_SIZES["age"]),
        ),
        "face_frontend": (wheel_face, numpy_face),
    }
    cases: dict[str, Any] = {}
    ratios: dict[str, float] = {}
    for name in CASES:
        wheel_fn, numpy_fn = pairs[name]
        cases[f"{name}_wheel"] = time_case(wheel_fn, warmup, iterations, repeats)
        cases[f"{name}_numpy"] = time_case(numpy_fn, warmup, iterations, repeats)
        ratios[f"{name}_numpy_over_wheel"] = (
            cases[f"{name}_numpy"]["median_ms"] / cases[f"{name}_wheel"]["median_ms"]
        )

    return {
        "schema_version": 2,
        "code_revision": code_revision(),
        "artifact_sha256": artifact_sha256(),
        "protocol": {
            "warmup_per_repeat": warmup,
            "iterations_per_repeat": iterations,
            "repeats": repeats,
            "input": "example/test-image.jpg, rectangle (419, 266, 506, 352)",
            "timer": "time.perf_counter, per call, in-process",
        },
        "environment": environment_info(),
        "cases": cases,
        "ratios": ratios,
    }


def markdown_table(report: dict[str, Any]) -> str:
    labels = {
        "landmarks": "Landmark prediction",
        "chip_gender": "Gender chip (32x32, details + extraction)",
        "chip_age": "Age chip (64x64, details + extraction)",
        "face_frontend": "Per-face total (landmarks + both chips)",
    }
    lines = [
        "| Stage | dlib-bin median (p95) | NumPy port median (p95) | Median ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in CASES:
        wheel = report["cases"][f"{name}_wheel"]
        numpy_case = report["cases"][f"{name}_numpy"]
        ratio = report["ratios"][f"{name}_numpy_over_wheel"]
        lines.append(
            f"| {labels[name]} | {wheel['median_ms']:.3f} ms ({wheel['p95_ms']:.3f}) | "
            f"{numpy_case['median_ms']:.3f} ms ({numpy_case['p95_ms']:.3f}) | "
            f"**{ratio:.1f}x** |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--quick", action="store_true", help="Fast dev sanity pass; not a performance claim."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Also write the summary table (for the feasibility report) to this file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warmup, iterations, repeats = args.warmup, args.iterations, args.repeats
    if args.quick:
        warmup, iterations, repeats = 2, 10, 1
    report = run(warmup, iterations, repeats)
    if args.quick:
        report["note"] = "--quick: not a performance claim, see benchmarks/README.md's convention"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = markdown_table(report)
    if args.markdown is not None:
        args.markdown.write_text(table + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
