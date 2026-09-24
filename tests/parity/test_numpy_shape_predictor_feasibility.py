"""Parity and evidence checks for the PR-8 NumPy shape-predictor/chip-extraction port.

Loads the checked-in ``tools/conversion/artifacts/shape-predictor-v1``
artifact directly (no compilation needed at test time). This is the CI lane
for ``tools/conversion/validate_shape_predictor.py``: it covers every kind of
check the validator runs, at CI-sized volume —

- the frozen PR-1 corpus (a missing corpus image fails, it does not skip);
- every cascade stage, replayed from the dlib probe's checked-in trace;
- the retained regression cases, plus a small fresh synthetic sweep, against
  live ``dlib-bin`` (a runtime dependency, so always installed);
- bit-exact ``chip_details`` geometry and the zero-angle raw-copy decision;
- input validation and artifact integrity checks;
- that every checked-in report was produced from the current sources.

This is feasibility-prototype coverage for spec/PR-8.md, not the shipped
runtime: ``src/age_and_gender/_faces.py`` still uses ``dlib-bin``,
unconditionally. Nothing here is imported by, or gates, that module.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import dlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tools/conversion"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_shape_predictor  # noqa: E402
import validate_shape_predictor as validator  # noqa: E402
from numpy_chip_extraction import (  # noqa: E402
    ChipDetails,
    extract_image_chip,
    get_face_chip_details,
)
from numpy_shape_predictor import (  # noqa: E402
    ArtifactError,
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
    load_regression_cases,
    synthetic_case,
)

from tests.golden import golden_documents  # noqa: E402

VALIDATION_REPORT = ARTIFACT_DIR / "shape-predictor-report.json"
BENCHMARK_REPORT = ARTIFACT_DIR / "numpy-frontend-benchmark.json"
REQUIRED_SECTIONS = {
    "frozen_corpus",
    "cascade_stages",
    "regression_cases",
    "live_dlib_synthetic",
    "chip_geometry",
    "chip_fast_path",
}
# A different seed from the checked-in report's sweep, so CI adds coverage.
CI_SYNTHETIC_SEED = 31337
CI_SYNTHETIC_CASES = 24
CI_GEOMETRY_CASES = 1500


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class NumpyShapePredictorFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = load_shape_predictor_artifact(ARTIFACT_DIR)
        cls.dlib_predictor = dlib.shape_predictor(str(SOURCE_MODEL))
        cls.documents = golden_documents()

    def _image(self, document) -> np.ndarray:
        source_path = ROOT / document.source_path
        self.assertTrue(source_path.is_file(), f"corpus image {document.source_path} is missing")
        return validator._read_rgb(source_path)

    def test_corpus_is_the_frozen_one(self) -> None:
        self.assertEqual(len(self.documents), 4)
        self.assertEqual(sum(len(document.faces) for document in self.documents), 11)

    def test_exact_landmarks_and_chips_on_every_frozen_face(self) -> None:
        for document in self.documents:
            array = self._image(document)
            for face in document.faces:
                with self.subTest(face=face.label):
                    landmarks = self.predictor(array, tuple(face.rectangle))
                    self.assertEqual(landmarks.tolist(), face.landmarks)
                    for task, size in validator.CHIP_SIZES.items():
                        details = get_face_chip_details(landmarks.astype(np.float64), size, 0.2)
                        chip = extract_image_chip(array, details)
                        self.assertTrue(np.array_equal(chip, face.chip(task)), f"{task} chip")

    def test_every_cascade_stage_matches_the_dlib_oracle_trace(self) -> None:
        fixture = _load_json(STAGE_TRACE_FIXTURE)
        self.assertEqual(fixture["artifact_sha256"], artifact_sha256())
        documents = {document.name: document for document in self.documents}
        expected_faces = sum(len(document.faces) for document in self.documents)
        self.assertEqual(len(fixture["faces"]), expected_faces)
        stages = 0
        for traced in fixture["faces"]:
            document = documents[traced["document"]]
            intensity = _grayscale_intensity(self._image(document))
            left, top, right, bottom = traced["rectangle"]
            initial = self.predictor._initial_shape
            shape = initial.astype(np.float32).copy()
            self.assertEqual(len(traced["stages"]), len(self.predictor._cascades))
            for index, (cascade, oracle) in enumerate(
                zip(self.predictor._cascades, traced["stages"], strict=True)
            ):
                features = _extract_feature_pixel_values(
                    intensity, left, top, right, bottom, shape, initial, cascade
                )
                self.assertEqual(
                    hashlib.sha256(features.astype("<f4").tobytes()).hexdigest(),
                    oracle["features_sha256"],
                    f"{traced['document']}:{traced['face']} cascade {index} features",
                )
                for idx1, idx2, thresh, leaves in zip(
                    cascade.tree_splits_idx1,
                    cascade.tree_splits_idx2,
                    cascade.tree_splits_thresh,
                    cascade.tree_leaves,
                    strict=True,
                ):
                    leaf = _evaluate_tree(features, idx1, idx2, thresh, leaves)
                    shape = (shape + leaf.reshape(self.predictor.num_parts, 2)).astype(np.float32)
                self.assertEqual(
                    shape.astype("<f4").tobytes().hex(),
                    oracle["shape_f32_hex"],
                    f"{traced['document']}:{traced['face']} cascade {index} shape",
                )
                stages += 1
        self.assertEqual(stages, expected_faces * len(self.predictor._cascades))

    def test_retained_regression_cases_match_live_dlib(self) -> None:
        cases = load_regression_cases()
        self.assertGreater(len(cases), 0)
        for case in cases:
            image, rectangle = synthetic_case(case["seed"], case["index"])
            with self.subTest(seed=case["seed"], index=case["index"]):
                problem, _ = validator.compare_case(
                    self.predictor, self.dlib_predictor, image, rectangle
                )
                self.assertIsNone(problem)

    def test_fresh_synthetic_sweep_matches_live_dlib(self) -> None:
        for index in range(CI_SYNTHETIC_CASES):
            image, rectangle = synthetic_case(CI_SYNTHETIC_SEED, index)
            with self.subTest(index=index, rectangle=rectangle):
                problem, _ = validator.compare_case(
                    self.predictor, self.dlib_predictor, image, rectangle
                )
                self.assertIsNone(problem)

    def test_chip_details_geometry_is_bit_exact(self) -> None:
        section = validator.check_chip_geometry(seed=CI_SYNTHETIC_SEED, count=CI_GEOMETRY_CASES)
        self.assertTrue(section["passed"], section["failures"])

    def test_zero_angle_fast_path_matches_dlib(self) -> None:
        section = validator.check_chip_fast_path(seed=CI_SYNTHETIC_SEED)
        self.assertTrue(section["passed"], section["failures"])
        # The review's reproducer: fractional size must interpolate, not copy.
        image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        rect = (0.2, 0.2, 2.4, 2.4)
        expected = np.asarray(
            dlib.extract_image_chip(
                image, dlib.chip_details(dlib.drectangle(*rect), dlib.chip_dims(3, 3))
            )
        )
        actual = extract_image_chip(image, ChipDetails(rect=rect, angle=0.0, rows=3, cols=3))
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.array_equal(actual, image[0:3, 0:3]))

    def test_invalid_chip_parameters_raise_instead_of_hanging(self) -> None:
        landmarks = np.array(self.documents[1].faces[0].landmarks, dtype=np.float64)
        for size in (0, -1, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                get_face_chip_details(landmarks, size)
        with self.assertRaises(ValueError):
            get_face_chip_details(landmarks, 32, padding=-0.1)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        for details in (
            ChipDetails(rect=(0.0, 0.0, 5.0, 5.0), angle=0.1, rows=0, cols=0),
            ChipDetails(rect=(5.0, 5.0, 0.0, 0.0), angle=0.1, rows=4, cols=4),
        ):
            with self.subTest(details=details), self.assertRaises(ValueError):
                extract_image_chip(image, details)

    def test_loader_rejects_tampered_artifacts(self) -> None:
        def tampered(edit) -> Path:
            directory = Path(tempfile.mkdtemp(prefix="shape-predictor-tamper-"))
            self.addCleanup(shutil.rmtree, directory)
            for name in ("manifest.json", "shape-predictor-v1.npz"):
                shutil.copy2(ARTIFACT_DIR / name, directory / name)
            manifest = _load_json(directory / "manifest.json")
            edit(directory, manifest)
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return directory

        def drop_cascade(_directory: Path, manifest: dict) -> None:
            manifest["cascade_num_features"].pop()
            manifest["cascade_tree_splits"].pop()

        def wrong_count(_directory: Path, manifest: dict) -> None:
            manifest["num_cascades"] += 1

        def wrong_schema(_directory: Path, manifest: dict) -> None:
            manifest["schema_version"] = 99

        def corrupt_npz(directory: Path, _manifest: dict) -> None:
            payload = bytearray((directory / "shape-predictor-v1.npz").read_bytes())
            payload[len(payload) // 2] ^= 0xFF
            (directory / "shape-predictor-v1.npz").write_bytes(bytes(payload))

        for edit in (drop_cascade, wrong_count, wrong_schema, corrupt_npz):
            with self.subTest(edit=edit.__name__), self.assertRaises(ArtifactError):
                load_shape_predictor_artifact(tampered(edit))

    def test_checked_in_evidence_is_current(self) -> None:
        manifest = _load_json(ARTIFACT_DIR / "manifest.json")
        self.assertEqual(manifest["converter_revision"], build_shape_predictor.converter_revision())
        self.assertEqual(manifest["source"]["sha256"], build_shape_predictor.SOURCE_SHA256)

        report = _load_json(VALIDATION_REPORT)
        self.assertEqual(report["code_revision"], code_revision(), "rerun the validator")
        self.assertEqual(report["artifact_sha256"], artifact_sha256())
        self.assertEqual(set(report["sections"]), REQUIRED_SECTIONS)
        for name, section in report["sections"].items():
            self.assertIs(section["passed"], True, name)
            self.assertEqual(section["failure_count"], 0, name)
        self.assertIs(report["passed"], True)

        benchmark = _load_json(BENCHMARK_REPORT)
        self.assertEqual(benchmark["code_revision"], code_revision(), "rerun the benchmark")
        self.assertEqual(benchmark["artifact_sha256"], artifact_sha256())
        protocol = benchmark["protocol"]
        self.assertGreaterEqual(protocol["warmup_per_repeat"], 10)
        self.assertGreaterEqual(protocol["iterations_per_repeat"], 100)
        self.assertGreaterEqual(protocol["repeats"], 5)
        self.assertNotIn("note", benchmark, "a --quick run is not performance evidence")


if __name__ == "__main__":
    unittest.main()
