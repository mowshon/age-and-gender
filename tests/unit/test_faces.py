from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import dlib
import numpy as np

from age_and_gender._faces import AGE_CHIP_SIZE, GENDER_CHIP_SIZE, FaceExtraction, FaceFrontend
from age_and_gender._images import as_rgb_array
from age_and_gender._models import bundled_models, load_bundle
from tests.bundles import full_bundle, write_manifest


@lru_cache(maxsize=1)
def _tiny_incompatible_predictor_bytes() -> bytes:
    """A real, valid dlib shape predictor that returns 3 parts, not 5.

    Trained from scratch on a handful of synthetic images with minimal trainer
    settings so it runs in milliseconds; this is a genuine incompatible model,
    not a corrupt file, so it exercises the runtime part-count check rather
    than the bundle's own hash/size verification.
    """
    rng = np.random.default_rng(0)
    images = []
    detections = []
    for _ in range(6):
        image = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
        rect = dlib.rectangle(2, 2, 37, 37)
        parts = dlib.points([dlib.point(10, 10), dlib.point(20, 20), dlib.point(30, 15)])
        images.append(image)
        detections.append([dlib.full_object_detection(rect, parts)])

    options = dlib.shape_predictor_training_options()
    options.oversampling_amount = 1
    options.nu = 0.5
    options.tree_depth = 2
    options.cascade_depth = 2
    options.feature_pool_size = 20
    options.num_test_splits = 2
    options.be_verbose = False
    predictor = dlib.train_shape_predictor(images, detections, options)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "predictor.dat"
        predictor.save(str(path))
        return path.read_bytes()


def _bundle_with_replaced_predictor(destination: Path, payload: bytes) -> Path:
    """Copy the installed bundle, swapping the predictor bytes but not its manifest."""
    bundle_dir = full_bundle(destination)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    filename = manifest["shape_predictor"]["artifact"]["filename"]
    (bundle_dir / filename).write_bytes(payload)
    manifest["shape_predictor"]["artifact"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["shape_predictor"]["artifact"]["bytes"] = len(payload)
    write_manifest(bundle_dir, manifest)
    return bundle_dir


class FaceFrontendConstructionTests(unittest.TestCase):
    def test_loads_from_the_installed_bundle(self) -> None:
        frontend = FaceFrontend(bundled_models())
        self.assertIs(frontend.bundle, bundled_models())

    def test_corrupt_shape_predictor_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            garbage = os.urandom(2048)
            bundle_dir = _bundle_with_replaced_predictor(Path(directory) / "bundle", garbage)
            bundle = load_bundle(bundle_dir)
            with self.assertRaises(ValueError):
                FaceFrontend(bundle)

    def test_incompatible_landmark_model_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = _tiny_incompatible_predictor_bytes()
            bundle_dir = _bundle_with_replaced_predictor(Path(directory) / "bundle", payload)
            bundle = load_bundle(bundle_dir)
            with self.assertRaisesRegex(ValueError, "3 landmark parts"):
                FaceFrontend(bundle)


class FaceFrontendUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frontend = FaceFrontend(bundled_models())

    def test_no_face_image_detects_and_extracts_nothing(self) -> None:
        image = as_rgb_array(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(self.frontend.detect(image), [])
        self.assertEqual(self.frontend.extract(image), [])

    def test_extract_returns_chips_of_the_declared_sizes(self) -> None:
        image = as_rgb_array(np.zeros((64, 64, 3), dtype=np.uint8))
        boxes = [[4, 4, 60, 60]]
        extractions = self.frontend.extract(image, boxes)
        self.assertEqual(len(extractions), 1)
        extraction = extractions[0]
        self.assertIsInstance(extraction, FaceExtraction)
        self.assertEqual(extraction.rectangle, [4, 4, 60, 60])
        self.assertEqual(len(extraction.landmarks), 5)
        self.assertEqual(extraction.gender_chip.shape, (GENDER_CHIP_SIZE, GENDER_CHIP_SIZE, 3))
        self.assertEqual(extraction.age_chip.shape, (AGE_CHIP_SIZE, AGE_CHIP_SIZE, 3))
        self.assertEqual(extraction.gender_chip.dtype, np.uint8)
        self.assertEqual(extraction.age_chip.dtype, np.uint8)

    def test_crop_chips_match_extract_with_the_whole_image_box(self) -> None:
        rng = np.random.default_rng(0)
        face = as_rgb_array(rng.integers(0, 256, size=(90, 70, 3), dtype=np.uint8))
        (extraction,) = self.frontend.extract(face, [[0, 0, 69, 89]])
        gender_chip, age_chip = self.frontend.crop_chips(face, (GENDER_CHIP_SIZE, AGE_CHIP_SIZE))
        self.assertTrue(np.array_equal(gender_chip, extraction.gender_chip))
        self.assertTrue(np.array_equal(age_chip, extraction.age_chip))
        (only_age,) = self.frontend.crop_chips(face, (AGE_CHIP_SIZE,))
        self.assertTrue(np.array_equal(only_age, extraction.age_chip))

    def test_unrepresentable_box_raises_value_error(self) -> None:
        image = as_rgb_array(np.zeros((64, 64, 3), dtype=np.uint8))
        huge = 10**30
        with self.assertRaises(ValueError):
            self.frontend.extract(image, [[0, 0, huge, huge]])

    def test_extract_does_not_mutate_the_input_image(self) -> None:
        image = as_rgb_array(np.zeros((64, 64, 3), dtype=np.uint8))
        before = image.copy()
        self.frontend.extract(image, [[4, 4, 60, 60]])
        self.assertTrue(np.array_equal(image, before))


if __name__ == "__main__":
    unittest.main()
