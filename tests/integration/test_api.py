"""Behavioral tests for the public ``AgeAndGender`` class.

Complements tests/parity/test_end_to_end.py (exact result equality against the
frozen oracle): this module checks laziness, construction modes, repeated and
concurrent use, and the ``from_model_dir`` factory, using the real bundled
models throughout rather than mocks.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from age_and_gender import AgeAndGender
from age_and_gender._images import as_rgb_array
from tests.bundles import PACKAGE_MODELS, full_bundle
from tests.golden import golden_documents, golden_images

ROOT = Path(__file__).resolve().parents[2]


def _load_test_image() -> np.ndarray:
    with Image.open(ROOT / "example/test-image.jpg") as image:
        return as_rgb_array(image.convert("RGB"))


class ConstructionTests(unittest.TestCase):
    def test_default_constructor_performs_no_heavy_io(self) -> None:
        predictor = AgeAndGender()
        self.assertIsNone(predictor._frontend)
        self.assertFalse(predictor._engine.age.is_loaded)
        self.assertFalse(predictor._engine.gender.is_loaded)

    def test_first_predict_call_builds_state_lazily(self) -> None:
        predictor = AgeAndGender()
        predictor.predict(_load_test_image())
        self.assertIsNotNone(predictor._frontend)
        self.assertTrue(predictor._engine.age.is_loaded)
        self.assertTrue(predictor._engine.gender.is_loaded)

    def test_repr_names_the_backing_bundle(self) -> None:
        self.assertIn("age_and_gender.models", repr(AgeAndGender()))

    def test_from_model_dir_uses_only_the_given_bundle(self) -> None:
        """Matching output alone would not distinguish real replacement from a
        no-op, since `full_bundle()` copies the installed models byte for
        byte; every component's `.bundle.origin` is checked directly so this
        fails if a network or the frontend silently keeps using the default.
        """
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            predictor = AgeAndGender.from_model_dir(bundle_dir)
            self.assertEqual(predictor._bundle.origin, str(bundle_dir))
            self.assertEqual(predictor._engine.age.bundle.origin, str(bundle_dir))
            self.assertEqual(predictor._engine.gender.bundle.origin, str(bundle_dir))
            expected = [face.result for face in golden_images()["test-image.golden.json"]]
            self.assertEqual(predictor.predict(_load_test_image()), expected)
            self.assertEqual(predictor._frontend.bundle.origin, str(bundle_dir))

    def test_from_model_dir_missing_manifest_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            AgeAndGender.from_model_dir(Path(directory) / "does-not-exist")

    def test_from_model_dir_incomplete_bundle_names_the_missing_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            manifest_path = bundle_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["shape_predictor"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape_predictor"):
                AgeAndGender.from_model_dir(bundle_dir)

    def test_from_model_dir_missing_artifact_fails_immediately_not_lazily(self) -> None:
        """`load_bundle()` only validates the manifest JSON; an artifact file
        it names but that is actually missing from disk must be caught right
        here, not deferred to whatever `predict()` call first happens to need
        that particular model.
        """
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            (bundle_dir / "age-v1.onnx").unlink()
            with self.assertRaises(FileNotFoundError):
                AgeAndGender.from_model_dir(bundle_dir)

    def test_from_model_dir_corrupt_artifact_fails_immediately_not_lazily(self) -> None:
        """Loading is structural, not hash-based, so the artifact must fail
        the backend parser rather than merely differ from a recorded digest.
        Replacing the whole file makes that failure deterministic.
        """
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            (bundle_dir / "shape_predictor_5_face_landmarks.dat").write_bytes(os.urandom(2048))
            with self.assertRaises(ValueError):
                AgeAndGender.from_model_dir(bundle_dir)


class PredictionBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = AgeAndGender()

    def test_no_face_image_returns_empty_list(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(self.predictor.predict(image), [])

    def test_repeated_predictions_reuse_state_and_agree(self) -> None:
        image = _load_test_image()
        first = self.predictor.predict(image)
        second = self.predictor.predict(image)
        third = self.predictor.predict(image)
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_predict_does_not_mutate_the_input_image(self) -> None:
        image = _load_test_image()
        before = image.copy()
        self.predictor.predict(image)
        self.assertTrue(np.array_equal(image, before))

    def test_explicit_boxes_bypass_detection(self) -> None:
        predictor = AgeAndGender()
        predictor.predict(_load_test_image())  # build the frontend first
        spy = mock.Mock(
            side_effect=AssertionError("the detector must not run when explicit boxes are given")
        )
        predictor._frontend._detector = spy
        explicit = next(doc for doc in golden_documents() if doc.box_mode == "explicit")
        with Image.open(ROOT / explicit.source_path) as image:
            array = as_rgb_array(image.convert("RGB"))
        results = predictor.predict(array, explicit.input_boxes_trbl)
        self.assertEqual(spy.call_count, 0)
        self.assertEqual(results, [face.result for face in explicit.faces])


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_predictions_on_one_instance_are_all_correct(self) -> None:
        predictor = AgeAndGender()
        image = _load_test_image()
        expected = [face.result for face in golden_images()["test-image.golden.json"]]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(predictor.predict, image) for _ in range(16)]
            results = [future.result() for future in futures]
        for result in results:
            self.assertEqual(result, expected)

    def test_predictions_on_one_instance_are_serialized_not_interleaved(self) -> None:
        """Proves the lock actually excludes concurrent entry, rather than the
        threads coincidentally producing correct results despite racing.
        """
        predictor = AgeAndGender()
        image = _load_test_image()
        predictor.predict(image)  # build state up front
        in_flight = threading.Event()
        concurrent_entry = threading.Event()
        real_extract = predictor._frontend.extract

        def guarded_extract(*args: object, **kwargs: object) -> object:
            if in_flight.is_set():
                concurrent_entry.set()
            in_flight.set()
            try:
                return real_extract(*args, **kwargs)
            finally:
                in_flight.clear()

        predictor._frontend.extract = guarded_extract
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(predictor.predict, image) for _ in range(16)]
            for future in futures:
                future.result()
        self.assertFalse(
            concurrent_entry.is_set(), "two predict() calls ran their core concurrently"
        )


class BundledModelsPathTests(unittest.TestCase):
    def test_package_models_directory_is_a_valid_bundle(self) -> None:
        """Sanity check for the fixture the construction tests copy from."""
        self.assertTrue((PACKAGE_MODELS / "manifest.json").is_file())


class WorkingDirectoryIndependenceTests(unittest.TestCase):
    def test_predict_works_from_outside_the_repository_working_directory(self) -> None:
        array = _load_test_image()
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            original_cwd = os.getcwd()
            try:
                os.chdir(os.path.expanduser("~"))
                predictor = AgeAndGender()
                predictor.load_shape_predictor(bundle_dir / "shape_predictor_5_face_landmarks.dat")
                predictor.load_dnn_gender_classifier(bundle_dir / "gender-v1.onnx")
                predictor.load_dnn_age_predictor(bundle_dir / "age-v1.onnx")
                results = predictor.predict(array)
            finally:
                os.chdir(original_cwd)
        expected = [face.result for face in golden_images()["test-image.golden.json"]]
        self.assertEqual(results, expected)


if __name__ == "__main__":
    unittest.main()
