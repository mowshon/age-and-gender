"""Shared provenance and case generation for the PR-8 feasibility evidence.

Maintainer-only. ``validate_shape_predictor.py``, ``benchmark_numpy_frontend.py``
and ``tests/parity/test_numpy_shape_predictor_feasibility.py`` all import this
module so that:

- every checked-in report records the exact prototype source revision it was
  produced from, and the CI test can reject a report that is stale relative
  to the current sources (:func:`code_revision`);
- every synthetic case is individually reproducible from ``(seed, index)``,
  so any mismatch a sweep finds can be retained as a regression fixture
  rather than only described (:func:`synthetic_case`).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "tools/conversion/artifacts/shape-predictor-v1"
SOURCE_MODEL = ROOT / "src/age_and_gender/models/shape_predictor_5_face_landmarks.dat"
REGRESSION_CASES = ARTIFACT_DIR / "regression-cases.json"
STAGE_TRACE_FIXTURE = ARTIFACT_DIR / "stage-trace.json"

# Every source whose behavior the validation/benchmark evidence depends on.
PROTOTYPE_SOURCES = (
    "tools/conversion/dlib_linalg.py",
    "tools/conversion/numpy_shape_predictor.py",
    "tools/conversion/numpy_chip_extraction.py",
    "tools/conversion/probe_shape_predictor.cpp",
    "tools/conversion/shape_predictor_evidence.py",
    "tools/conversion/validate_shape_predictor.py",
    "tools/conversion/benchmark_numpy_frontend.py",
)


def code_revision() -> str:
    """SHA-256 over :data:`PROTOTYPE_SOURCES` (paths and bytes)."""
    digest = hashlib.sha256()
    for relative in PROTOTYPE_SOURCES:
        digest.update(relative.encode("ascii"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def artifact_sha256() -> str:
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    return manifest["artifact"]["sha256"]


def synthetic_case(seed: int, index: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """One deterministic (noise image, square rectangle) pair.

    Biased toward the edge cases spec/PR-8.md names: rectangles extending
    outside the image (zero-fill boundary handling) and rectangles much larger
    than either output chip (multi-level ``pyramid_down<2>`` descent). Noise
    images are deliberate: every sampled pixel differs from its neighbors, so
    any off-by-one-pixel geometry error changes the output.
    """
    rng = np.random.default_rng([seed, index])
    height = int(rng.integers(40, 1400))
    width = int(rng.integers(40, 1400))
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    left = int(rng.integers(-150, width))
    top = int(rng.integers(-150, height))
    size = int(rng.integers(10, min(width, height) + 300))
    return image, (left, top, left + size, top + size)


def load_regression_cases() -> list[dict[str, Any]]:
    return json.loads(REGRESSION_CASES.read_text(encoding="utf-8"))["cases"]


def environment_info() -> dict[str, Any]:
    """Machine/dependency/thread metadata, following benchmarks/benchmark_pipeline.py."""
    import dlib
    from PIL import Image

    def read_first(path: str, prefix: str | None = None) -> str | None:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            if prefix is None:
                return line.strip()
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return None

    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        numpy_blas = f"{blas.get('name')} {blas.get('version')}"
    except Exception:  # best-effort metadata only
        numpy_blas = "unknown"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": read_first("/proc/cpuinfo", "model name") or platform.processor(),
        "logical_cores": os.cpu_count(),
        "cpu_governor": read_first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "dependency_versions": {
            "numpy": np.__version__,
            "numpy_blas": numpy_blas,
            "dlib": dlib.__version__,
            "dlib_use_blas": bool(getattr(dlib, "DLIB_USE_BLAS", False)),
            "dlib_use_lapack": bool(getattr(dlib, "DLIB_USE_LAPACK", False)),
            "pillow": Image.__version__,
        },
        "environment_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "timed_code_threads": (
            "both timed paths are single-threaded: dlib's shape predictor and "
            "chip extraction run on the calling thread, and the NumPy port uses "
            "only elementwise ufuncs (no BLAS calls)"
        ),
    }
