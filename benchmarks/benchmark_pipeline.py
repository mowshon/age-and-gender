#!/usr/bin/env python3
"""Machine-readable performance report for the Python inference pipeline.

spec/PR-6.md requires measuring before optimizing and reporting the same
categories old and new: fresh-process import and model/session initialization,
first vs. warmed prediction, each pipeline stage, batch-size thresholds, and
steady-state behavior (to catch accidental reinitialization or growth).

This script measures the *new* Python package only. The legacy baseline it is
compared against was captured the same way spec/PR-1.md's oracle tooling always
has: `tools/legacy/oracle.cpp` run through `benchmarks/legacy_baseline.py`, and
is read back from `benchmarks/results/linux-x86_64-python310.json` plus the
single-face oracle report alongside it (see benchmarks/README.md). Both sides
use the same decoded RGB arrays, the same source weights, and the same machine.

Usage::

    venv/bin/python benchmarks/benchmark_pipeline.py --output benchmarks/results/report.json

Run ``--help`` for the full set of tunables. Defaults follow spec/PR-6.md's
"at least 10 warmups and 100 measured iterations per case, repeated across five
runs"; pass ``--quick`` during development to shrink all three for a fast
sanity pass whose numbers are not meant to gate anything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import dlib  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image  # noqa: E402

import age_and_gender  # noqa: E402
from age_and_gender._faces import AGE_CHIP_SIZE, GENDER_CHIP_SIZE, FaceFrontend  # noqa: E402
from age_and_gender._images import as_rgb_array  # noqa: E402
from age_and_gender._inference import DEFAULT_MAX_BATCH_SIZE, prepare_batch  # noqa: E402
from age_and_gender._models import bundled_models  # noqa: E402
from age_and_gender._postprocess import face_predictions  # noqa: E402

SCHEMA_VERSION = 1
TEST_IMAGE = ROOT / "example/test-image.jpg"  # 5 faces, detected
TEST_IMAGE_2 = ROOT / "example/test-image-2.jpg"  # 2 faces, detected
NO_FACE_IMAGE = ROOT / "libs/dlib/examples/faces/dogs.jpg"  # 0 faces
# spec/INVESTIGATION.md's frozen reference boxes for test-image.jpg, [L,T,R,B].
SINGLE_FACE_BOX_LTRB = [419, 266, 506, 352]
LEGACY_RESULTS_DIR = ROOT / "benchmarks/results"


def _ltrb_to_trbl(box) -> list[int]:
    """``AgeAndGender.predict()``'s box argument keeps the legacy TRBL order,
    but ``FaceFrontend.detect()`` (used here to build synthetic explicit-box
    scenarios) returns ``[left, top, right, bottom]``; passing LTRB straight
    into ``predict()`` would silently reinterpret it as a different box.
    """
    left, top, right, bottom = box
    return [top, right, bottom, left]

# -- timing helpers ---------------------------------------------------------


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
    """Run ``fn()`` under the warmup/iterations/repeats protocol spec/PR-6.md asks for.

    Each of ``repeats`` independent runs performs its own ``warmup`` untimed
    calls followed by ``iterations`` timed ones; the per-repeat medians are
    reported alongside one summary pooled over every timed sample, so a
    maintainer can see both the typical case and run-to-run noise.
    """
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


# -- environment --------------------------------------------------------


def environment_info(warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    bundle = bundled_models()
    runtime = bundle.runtime
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
        "package_version": age_and_gender.__version__,
        "dependency_versions": {
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "pillow": Image.__version__,
        },
        "environment_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "runtime_settings": {
            "provider": runtime.provider,
            "graph_optimization_level": runtime.graph_optimization_level,
            "intra_op_num_threads": runtime.intra_op_num_threads,
            "inter_op_num_threads": runtime.inter_op_num_threads,
            "default_max_batch_size": DEFAULT_MAX_BATCH_SIZE,
        },
        "warmup_per_repeat": warmup,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "note": (
            "Captured on a development machine, not a dedicated benchmark host; "
            "see benchmarks/README.md before treating this as the spec/PR-6.md "
            "controlled-hardware performance gate."
        ),
    }


# -- cold start (fresh process) ------------------------------------------

_COLD_START_SCRIPT = """
import json
import sys
import time

t0 = time.perf_counter()
import age_and_gender
from age_and_gender._images import as_rgb_array
t_import = time.perf_counter()

predictor = age_and_gender.AgeAndGender()
t_construct = time.perf_counter()

predictor._engine.age.ensure_loaded()
predictor._engine.gender.ensure_loaded()
predictor._ensure_frontend()
t_init = time.perf_counter()

from PIL import Image
with Image.open(sys.argv[1]) as image:
    array = as_rgb_array(image.convert("RGB"))
t_decode = time.perf_counter()

first = predictor.predict(array)
t_first = time.perf_counter()
second = predictor.predict(array)
t_second = time.perf_counter()

print(json.dumps({
    "import_ms": (t_import - t0) * 1000.0,
    "construct_ms": (t_construct - t_import) * 1000.0,
    "model_session_init_ms": (t_init - t_construct) * 1000.0,
    "decode_and_validate_ms": (t_decode - t_init) * 1000.0,
    "first_predict_ms": (t_first - t_decode) * 1000.0,
    "second_predict_ms": (t_second - t_first) * 1000.0,
    "faces": len(first),
}))
"""


def measure_cold_start(runs: int) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    samples: list[dict[str, float]] = []
    for _ in range(runs):
        result = subprocess.run(
            [sys.executable, "-c", _COLD_START_SCRIPT, str(TEST_IMAGE)],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        samples.append(json.loads(result.stdout))
    fields = [key for key in samples[0] if key != "faces"]
    return {
        "runs": runs,
        "faces": samples[0]["faces"],
        **{field: summarize([sample[field] for sample in samples]) for field in fields},
    }


# -- fixtures -------------------------------------------------------------


def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return as_rgb_array(image.convert("RGB"))


# -- stage-level timing -----------------------------------------------------


def measure_stages(predictor, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    frontend: FaceFrontend = predictor._ensure_frontend()
    engine = predictor._engine
    image = _load_array(TEST_IMAGE)
    pil_image = Image.open(TEST_IMAGE).convert("RGB")
    detections = frontend.detect(image)
    extractions = frontend.extract(image)
    detection_rect = dlib.rectangle(*detections[0])
    shape = frontend._predictor(image, detection_rect)
    gender_chip = extractions[0].gender_chip
    age_chip = extractions[0].age_chip
    gender_chips = [extraction.gender_chip for extraction in extractions]
    age_chips = [extraction.age_chip for extraction in extractions]
    gender_tensor = prepare_batch(gender_chips, engine.gender.spec)
    age_tensor = prepare_batch(age_chips, engine.age.spec)
    engine.gender.ensure_loaded()
    engine.age.ensure_loaded()
    gender_probabilities = engine.gender.run(gender_tensor)
    age_probabilities = engine.age.run(age_tensor)

    case = lambda fn: time_case(fn, warmup, iterations, repeats)  # noqa: E731
    return {
        "image_validate_copy_from_pil_ms": case(lambda: as_rgb_array(pil_image)),
        "image_validate_copy_array_passthrough_ms": case(lambda: as_rgb_array(image)),
        "detection_5_faces_ms": case(lambda: frontend.detect(image)),
        "landmarks_one_face_ms": case(lambda: frontend._predictor(image, detection_rect)),
        "chip_extract_gender_one_face_ms": case(
            lambda: frontend._chip(image, shape, GENDER_CHIP_SIZE)
        ),
        "chip_extract_age_one_face_ms": case(lambda: frontend._chip(image, shape, AGE_CHIP_SIZE)),
        "normalization_gender_5_chips_ms": case(
            lambda: prepare_batch(gender_chips, engine.gender.spec)
        ),
        "normalization_age_5_chips_ms": case(lambda: prepare_batch(age_chips, engine.age.spec)),
        "network_gender_run_5_chips_ms": case(lambda: engine.gender.run(gender_tensor)),
        "network_age_run_5_chips_ms": case(lambda: engine.age.run(age_tensor)),
        "postprocess_5_faces_ms": case(
            lambda: face_predictions(
                [extraction.rectangle for extraction in extractions],
                gender_probabilities,
                age_probabilities,
                labels=engine.gender.spec.labels,
                age_weights=engine.age.spec.age_weights,
            )
        ),
        "single_gender_chip": gender_chip.shape,
        "single_age_chip": age_chip.shape,
    }


# -- batch-size sweep ---------------------------------------------------


def _observed_call_batch_sizes(network, chips: list) -> list[int]:
    """Run ``network.probabilities(chips)`` once and record each underlying
    ``session.run()`` call's batch size, by spying on the session directly.

    ``NeuralNetwork.run()`` always makes exactly one ONNX Runtime call for
    whatever tensor it is given; only ``probabilities()`` applies the bounded
    chunking spec/PR-6.md asks for (see ``_inference.py``). Timing ``run()``
    on a pre-built batch-64 tensor, as an earlier version of this function
    did, therefore never exercised chunking at all — it measured one
    unbounded batch-64 call and then *computed* a "chunked_calls" figure from
    arithmetic rather than *observing* what chunking actually did. This helper
    calls the real chunking entry point and returns the batch size of every
    ONNX Runtime call it actually made, so the sweep below can report observed
    behavior instead of an assumed one.
    """
    network.ensure_loaded()
    session = network._session
    original_run = session.run
    sizes: list[int] = []

    def spy(names, feed):
        sizes.append(feed[network.spec.input_name].shape[0])
        return original_run(names, feed)

    session.run = spy  # type: ignore[method-assign]
    try:
        network.probabilities(chips)
    finally:
        session.run = original_run
    return sizes


def measure_batch_sweep(predictor, warmup: int, iterations: int, repeats: int) -> dict[str, Any]:
    engine = predictor._engine
    frontend = predictor._ensure_frontend()
    image = _load_array(TEST_IMAGE)
    extraction = frontend.extract(image)[0]
    result: dict[str, Any] = {}
    for task, chip, network in (
        ("gender", extraction.gender_chip, engine.gender),
        ("age", extraction.age_chip, engine.age),
    ):
        network.ensure_loaded()
        sizes = {}
        for batch_size in (1, 2, 8, 32, 64):
            chips = [chip] * batch_size
            # Timed through the real chunking entry point (`probabilities()`),
            # not the lower-level `run()`, so batch sizes above the chunk
            # bound (currently only 64) are actually split here, not just
            # timed as one oversized call.
            timing = time_case(
                lambda c=chips, n=network: n.probabilities(c), warmup, iterations, repeats
            )
            timing["faces_per_second"] = (
                batch_size / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else None
            )
            observed = _observed_call_batch_sizes(network, chips)
            timing["chunked_calls"] = len(observed)
            timing["observed_call_batch_sizes"] = observed
            sizes[str(batch_size)] = timing
        result[task] = sizes
    return result


# -- full-image scenarios -------------------------------------------------


def _explicit_boxes_from(image: np.ndarray, frontend: FaceFrontend, repeat: int = 1) -> list:
    """Detected boxes, converted to the public API's TRBL order and repeated."""
    boxes = [_ltrb_to_trbl(box) for box in frontend.detect(image)]
    return boxes * repeat


def measure_full_image_scenarios(
    predictor, warmup: int, iterations: int, repeats: int
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    frontend = predictor._ensure_frontend()

    def add(name: str, image: np.ndarray, boxes) -> None:
        # A first call outside the timed loop to report separately, matching
        # spec/PR-6.md's "first prediction and warmed prediction separately".
        first_start = time.perf_counter()
        first_result = predictor.predict(image, boxes)
        first_ms = (time.perf_counter() - first_start) * 1000.0
        timing = time_case(lambda: predictor.predict(image, boxes), warmup, iterations, repeats)
        faces = len(first_result)
        timing["faces"] = faces
        timing["first_call_ms"] = first_ms
        timing["faces_per_second"] = (
            faces / (timing["median_ms"] / 1000.0) if faces and timing["median_ms"] > 0 else None
        )
        scenarios[name] = timing

    if NO_FACE_IMAGE.is_file():
        add("zero_faces_detect", _load_array(NO_FACE_IMAGE), None)
    test_image = _load_array(TEST_IMAGE)
    test_image_2 = _load_array(TEST_IMAGE_2)
    add("five_faces_detect", test_image, None)
    add("two_faces_detect", test_image_2, None)
    add("one_face_explicit", test_image, [_ltrb_to_trbl(SINGLE_FACE_BOX_LTRB)])
    detected_boxes = _explicit_boxes_from(test_image, frontend, repeat=1)
    add("five_faces_explicit", test_image, detected_boxes)
    many_boxes = _explicit_boxes_from(test_image, frontend, repeat=10)  # 50 faces
    add("many_faces_explicit_50", test_image, many_boxes)
    return scenarios


# -- steady state / resource lifecycle -------------------------------------


def measure_steady_state(predictor, iterations: int) -> dict[str, Any]:
    image = _load_array(TEST_IMAGE)
    predictor.predict(image)  # ensure everything is already warm
    engine = predictor._engine
    age_session_before = engine.age._session
    gender_session_before = engine.gender._session
    frontend_before = predictor._frontend
    bundle_before = predictor._bundle

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        predictor.predict(image)
        samples.append((time.perf_counter() - start) * 1000.0)

    half = max(1, iterations // 10)
    first_median = statistics.median(samples[:half])
    last_median = statistics.median(samples[-half:])
    return {
        "iterations": iterations,
        "age_session_reused": engine.age._session is age_session_before,
        "gender_session_reused": engine.gender._session is gender_session_before,
        "frontend_reused": predictor._frontend is frontend_before,
        "bundle_reused": predictor._bundle is bundle_before,
        "bundled_models_cache_info": bundled_models.cache_info()._asdict(),
        "first_window_median_ms": first_median,
        "last_window_median_ms": last_median,
        "growth_ratio": (last_median / first_median) if first_median else None,
    }


# -- memory / model counts --------------------------------------------------


def measure_memory_and_counts(predictor) -> dict[str, Any]:
    engine = predictor._engine
    frontend = predictor._ensure_frontend()
    return {
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "ru_maxrss is KiB on Linux; this is process-lifetime peak, not this call's delta",
        "age_sessions": 1 if engine.age.is_loaded else 0,
        "gender_sessions": 1 if engine.gender.is_loaded else 0,
        "detectors": 1 if frontend is not None else 0,
        "shape_predictors": 1 if frontend is not None else 0,
    }


# -- legacy comparison --------------------------------------------------


def _median(values: list[float]) -> float:
    return statistics.median(values)


def compare_to_legacy(full_image_scenarios: dict[str, Any]) -> dict[str, Any] | None:
    """Compare warmed medians against the frozen legacy oracle measurements.

    ``benchmarks/results/linux-x86_64-python310.json`` already carries the
    5-face (test-image.jpg) and 2-face (test-image-2.jpg) legacy warm totals
    from spec/PR-1.md's oracle. The 1-face case has no equivalent in that file
    (neither example image has exactly one face); see
    benchmarks/results/legacy-single-face-oracle-report.json, captured the same
    way with `--box` selecting spec/INVESTIGATION.md's first reference face
    instead of running the detector, and benchmarks/README.md for the exact
    command.
    """
    baseline_path = LEGACY_RESULTS_DIR / "linux-x86_64-python310.json"
    single_face_path = LEGACY_RESULTS_DIR / "legacy-single-face-oracle-report.json"
    if not baseline_path.is_file():
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    workloads = {Path(w["input"]["source"]).name: w for w in baseline["workloads"]}
    cases = []

    def add_case(label: str, faces: int, legacy_median_ms: float, new_key: str) -> None:
        new_timing = full_image_scenarios.get(new_key)
        if new_timing is None:
            return
        new_median = new_timing["median_ms"]
        ratio = new_median / legacy_median_ms if legacy_median_ms else None
        cases.append(
            {
                "label": label,
                "faces": faces,
                "legacy_warm_median_ms": legacy_median_ms,
                "new_warm_median_ms": new_median,
                "ratio_new_over_legacy": ratio,
                "regression_pct": (ratio - 1.0) * 100.0 if ratio is not None else None,
                "within_10_percent_regression_gate": (ratio is not None and ratio <= 1.10),
            }
        )

    if "test-image.jpg" in workloads:
        add_case(
            "five_faces (test-image.jpg, detected)",
            5,
            _median(workloads["test-image.jpg"]["warm_total_ms"]),
            "five_faces_detect",
        )
    if "test-image-2.jpg" in workloads:
        add_case(
            "two_faces (test-image-2.jpg, detected)",
            2,
            _median(workloads["test-image-2.jpg"]["warm_total_ms"]),
            "two_faces_detect",
        )
    if single_face_path.is_file():
        single = json.loads(single_face_path.read_text(encoding="utf-8"))
        add_case(
            "one_face (test-image.jpg, explicit box)",
            1,
            _median(single["timing_ms"]["warm_total"]),
            "one_face_explicit",
        )
    return {
        "acceptance_gate": (
            "spec/PR-6.md: warmed median one-face and five-face latency should not "
            "regress by more than 10% versus the measured legacy public baseline"
        ),
        "cases": cases,
    }


# -- main -------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictor = age_and_gender.AgeAndGender()
    predictor.predict(_load_array(TEST_IMAGE))  # warm everything before measuring

    full_image_scenarios = measure_full_image_scenarios(
        predictor, args.warmup, args.iterations, args.repeats
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "environment": environment_info(args.warmup, args.iterations, args.repeats),
        "cold_start": measure_cold_start(args.cold_runs),
        "stages": measure_stages(predictor, args.warmup, args.iterations, args.repeats),
        "batch_size_sweep": measure_batch_sweep(
            predictor, args.warmup, args.iterations, args.repeats
        ),
        "full_image_scenarios": full_image_scenarios,
        "steady_state": measure_steady_state(predictor, args.steady_state_iterations),
        "memory_and_model_counts": measure_memory_and_counts(predictor),
    }
    report["legacy_comparison"] = compare_to_legacy(full_image_scenarios)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cold-runs", type=int, default=5)
    parser.add_argument("--steady-state-iterations", type=int, default=200)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shrink every count for a fast development sanity pass; numbers "
        "from this mode do not satisfy spec/PR-6.md's measurement protocol "
        "and must not be used to evaluate the 10%% regression gate.",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.warmup, args.iterations, args.repeats = 2, 10, 1
        args.cold_runs = 2
        args.steady_state_iterations = 20
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
