#!/usr/bin/env python3
"""Independent-session oversubscription check for candidate ONNX Runtime thread counts.

spec/PR-6.md's "Optimization order" step 5 asks to "compare one-worker and
multiple-worker scenarios to avoid oversubscription" before tuning ONNX
Runtime intra/inter-op threads. `tools/conversion/validate_conversion.py`'s
thread-count axis (`THREAD_CANDIDATES`/`SELECTED_THREAD_SETTING`) answers
whether a higher `intra_op_num_threads` changes numerical results; it does
not, and cannot by itself, answer whether raising it regresses throughput
once *independent* sessions run concurrently (a small container or a
multi-process deployment, for example). This script answers that second
question, on real `AgeAndGender` instances, not raw ONNX Runtime sessions.

`AgeAndGender.predict()` serializes calls on *one* instance with an internal
lock (see `api.py`), so `tests/integration/test_api.py`'s `ConcurrencyTests`
— which use one shared instance — never exercise two independently
multi-threaded ONNX Runtime sessions running at the same time; they check
correctness under that lock, not oversubscription. This script instead
builds several independent `AgeAndGender` instances (each with its own
session, as a multi-worker/multi-process deployment would have) and drives
each one from its own thread concurrently, at each candidate thread count.

Usage::

    venv/bin/python benchmarks/concurrency_benchmark.py \\
      --output benchmarks/results/concurrency-report.json

Each (thread setting, worker count, face count) cell is measured as
aggregate calls/second across all workers for a fixed wall-clock duration
(``--duration``, default 2 seconds) after a short warmup, rather than a fixed
iteration count, so more workers do not simply multiply the run time.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402
from tests.bundles import full_bundle  # noqa: E402
from tests.golden import golden_documents, golden_images  # noqa: E402
from tools.conversion.validate_conversion import THREAD_CANDIDATES  # noqa: E402

import age_and_gender._models as _models_module  # noqa: E402
from age_and_gender import AgeAndGender  # noqa: E402
from age_and_gender._images import as_rgb_array  # noqa: E402

SCHEMA_VERSION = 1
TEST_IMAGE = ROOT / "example/test-image.jpg"  # 5 faces, detected
WORKER_COUNTS = (1, 4)
FACE_COUNTS = (5, 32)


@contextlib.contextmanager
def _allow_runtime(threads: dict[str, int]):
    """Temporarily accept ``threads`` as a valid bundle runtime contract.

    ``_models.py``'s ``SUPPORTED_RUNTIME`` deliberately locks the *shipped*
    bundle to the parity-validated (1, 1) setting (see that module's
    docstring); this benchmark needs real ``AgeAndGender``/``NeuralNetwork``
    sessions at other candidate thread counts so it measures the actual
    production code path rather than a hand-rolled stand-in. The patch is
    local to this process, lasts only for the ``from_model_dir()`` call inside
    the ``with`` block, and never touches the shipped manifest, any file on
    disk, or any other process; the resulting session's own thread settings
    are fixed at construction time, so nothing about the shipped default
    changes once this context manager exits.
    """
    candidate = {
        "provider": "CPUExecutionProvider",
        "graph_optimization_level": "ORT_DISABLE_ALL",
        **threads,
    }
    with mock.patch.object(_models_module, "SUPPORTED_RUNTIME", candidate):
        yield


def _load_array(path: Path) -> Any:
    with Image.open(path) as image:
        return as_rgb_array(image.convert("RGB"))


def _explicit_boxes(count: int) -> tuple[Any, list[list[int]], list[dict[str, Any]]]:
    """``count`` faces via the golden explicit-box fixture, tiled and sliced.

    Reuses ``tests/parity/test_batching.py::EndToEndChunkBoundaryTests``'s
    approach: repeated identical boxes on the same source image produce
    repeated identical results, so this needs no extra fixture beyond the
    already-frozen ``explicit-boxes.golden.json``.
    """
    document = next(doc for doc in golden_documents() if doc.name == "explicit-boxes.golden.json")
    image = _load_array(ROOT / document.source_path)
    boxes = document.input_boxes_trbl
    expected = [face.result for face in document.faces]
    repeats = math.ceil(count / len(boxes))
    return image, (boxes * repeats)[:count], (expected * repeats)[:count]


def _make_bundle_with_threads(tmp_root: Path, name: str, threads: dict[str, int]) -> Path:
    """A throwaway copy of the installed bundle with a candidate thread setting.

    `_models.py`'s `SUPPORTED_RUNTIME` locks the *shipped* bundle to (1, 1);
    this does not touch that contract, it only lets this maintainer-only
    benchmark load a structurally-identical bundle with a different `runtime`
    block via the normal `from_model_dir()` path, so the measured sessions are
    real `NeuralNetwork`/`AgeAndGender` sessions, not a hand-rolled stand-in.
    """

    def set_runtime(manifest: dict[str, Any]) -> None:
        manifest["runtime"] = {
            "provider": "CPUExecutionProvider",
            "graph_optimization_level": "ORT_DISABLE_ALL",
            **threads,
        }

    return full_bundle(tmp_root / name, mutate=set_runtime)


def _run_workers(
    make_predictor,
    scenario_call,
    worker_count: int,
    warmup_calls: int,
    duration_s: float,
) -> dict[str, Any]:
    """Run ``worker_count`` independent predictors concurrently for ``duration_s``.

    Each worker gets its own predictor instance (its own ONNX Runtime
    sessions) and its own thread, so this measures the same kind of
    concurrency a multi-worker/multi-process deployment would produce, not
    contention inside one shared, lock-serialized instance.
    """
    predictors = [make_predictor() for _ in range(worker_count)]
    for predictor in predictors:
        for _ in range(warmup_calls):
            scenario_call(predictor)

    counts = [0] * worker_count
    stop = threading.Event()

    def worker(index: int) -> None:
        predictor = predictors[index]
        local = 0
        while not stop.is_set():
            scenario_call(predictor)
            local += 1
        counts[index] = local

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [pool.submit(worker, index) for index in range(worker_count)]
        start = time.perf_counter()
        time.sleep(duration_s)
        stop.set()
        for future in futures:
            future.result()
        elapsed = time.perf_counter() - start

    total_calls = sum(counts)
    return {
        "worker_count": worker_count,
        "elapsed_s": elapsed,
        "total_calls": total_calls,
        "calls_per_worker": counts,
        "calls_per_second": total_calls / elapsed if elapsed > 0 else None,
    }


def measure(
    tmp_root: Path,
    warmup_calls: int,
    duration_s: float,
    worker_counts: tuple[int, ...],
    face_counts: tuple[int, ...],
    thread_candidates: dict[str, dict[str, int]],
) -> dict[str, Any]:
    five_face_image = _load_array(TEST_IMAGE)
    thirty_two_image, thirty_two_boxes, thirty_two_expected = _explicit_boxes(32)
    scenarios: dict[int, Any] = {
        5: lambda predictor: predictor.predict(five_face_image),
        32: lambda predictor: predictor.predict(thirty_two_image, thirty_two_boxes),
    }
    # Correctness is not this script's job (tests/parity/test_batching.py
    # already covers it end to end), but a silently-broken scenario would
    # make every throughput number meaningless, so check it once up front.
    reference = AgeAndGender()
    expected_five = [face.result for face in golden_images()["test-image.golden.json"]]
    if reference.predict(five_face_image) != expected_five:
        raise RuntimeError("5-face scenario does not match the golden fixture")
    if reference.predict(thirty_two_image, thirty_two_boxes) != thirty_two_expected:
        raise RuntimeError("32-face scenario does not match the golden fixture")

    results: dict[str, Any] = {}
    for name, threads in thread_candidates.items():
        bundle_dir = _make_bundle_with_threads(tmp_root, f"bundle-{name}", threads)

        def make_predictor(bundle_dir=bundle_dir, threads=threads) -> AgeAndGender:
            with _allow_runtime(threads):
                return AgeAndGender.from_model_dir(bundle_dir)

        by_faces: dict[str, Any] = {}
        for faces in face_counts:
            scenario_call = scenarios[faces]
            by_workers = {}
            for worker_count in worker_counts:
                cell = _run_workers(
                    make_predictor, scenario_call, worker_count, warmup_calls, duration_s
                )
                cell["faces_per_second"] = (
                    cell["calls_per_second"] * faces
                    if cell["calls_per_second"] is not None
                    else None
                )
                by_workers[str(worker_count)] = cell
            by_faces[str(faces)] = by_workers
        results[name] = {"thread_settings": threads, "faces": by_faces}
    return results


def environment_info() -> dict[str, Any]:
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        cpu_model = platform.processor() or "unknown"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "note": (
            "Captured on a development machine, not a dedicated benchmark host; "
            "see benchmarks/README.md's Machine caveat. Throughput numbers here "
            "are relative (across thread/worker settings on one machine), not an "
            "absolute SLA."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-calls", type=int, default=2)
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds measured per cell.")
    parser.add_argument(
        "--worker-counts", type=int, nargs="+", default=list(WORKER_COUNTS)
    )
    parser.add_argument("--face-counts", type=int, nargs="+", default=list(FACE_COUNTS))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink duration/warmup for a fast development sanity pass; not a report number.",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.warmup_calls = 1
        args.duration = 0.3
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="age-and-gender-concurrency-") as tmp:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_unix": time.time(),
            "environment": environment_info(),
            "warmup_calls": args.warmup_calls,
            "duration_s": args.duration,
            "worker_counts": args.worker_counts,
            "face_counts": args.face_counts,
            "thread_candidates": THREAD_CANDIDATES,
            "results": measure(
                Path(tmp),
                args.warmup_calls,
                args.duration,
                tuple(args.worker_counts),
                tuple(args.face_counts),
                THREAD_CANDIDATES,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
