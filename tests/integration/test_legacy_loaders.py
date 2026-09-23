"""``load_shape_predictor``/``load_dnn_age_predictor``/``load_dnn_gender_classifier``.

Covers the compatibility boundary spec/PR-5.md draws for the three loader
methods. The two neural loaders only accept an ``.onnx`` file with a sibling
``manifest.json`` naming it for that task — there is no hash-based
recognition of the original ``.dat`` weights, since loading is validated
structurally, not cryptographically (see spec/PR-3.md's review follow-ups and
``_models.py``'s module docstring). ``load_shape_predictor`` is unaffected: it
loads any compatible ``.dat`` file directly through dlib, as before.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from age_and_gender import AgeAndGender
from age_and_gender._images import as_rgb_array
from tests.bundles import full_bundle
from tests.golden import golden_images

ROOT = Path(__file__).resolve().parents[2]
LEGACY_MODELS = ROOT / "example/models"
AGE_DAT = LEGACY_MODELS / "dnn_age_predictor_v1.dat"
SHAPE_DAT = LEGACY_MODELS / "shape_predictor_5_face_landmarks.dat"
EXPECTED_RESULTS = [face.result for face in golden_images()["test-image.golden.json"]]


def _test_image_array():
    with Image.open(ROOT / "example/test-image.jpg") as image:
        return as_rgb_array(image.convert("RGB"))


class NeuralLoaderTests(unittest.TestCase):
    def test_explicit_onnx_with_sibling_manifest_is_accepted(self) -> None:
        """Matching output alone would not distinguish real replacement from a
        no-op, since `full_bundle()` copies the installed models byte for
        byte; `.bundle.origin` and network object identity are checked
        directly so this fails if the loader silently kept the default.
        """
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            predictor = AgeAndGender()
            original_age = predictor._engine.age
            predictor.load_dnn_age_predictor(bundle_dir / "age-v1.onnx")
            self.assertIsNot(predictor._engine.age, original_age)
            self.assertEqual(predictor._engine.age.bundle.origin, str(bundle_dir))
            self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)

    def test_explicit_onnx_with_sibling_manifest_is_accepted_for_gender_too(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            predictor = AgeAndGender()
            predictor.load_dnn_gender_classifier(bundle_dir / "gender-v1.onnx")
            self.assertEqual(predictor._engine.gender.bundle.origin, str(bundle_dir))
            self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)

    def test_explicit_onnx_without_a_matching_manifest_entry_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            # This file exists and its sibling manifest is valid, but the
            # manifest names it as the *gender* model, not age.
            predictor = AgeAndGender()
            with self.assertRaisesRegex(ValueError, "does not name this file as the age model"):
                predictor.load_dnn_age_predictor(bundle_dir / "gender-v1.onnx")

    def test_renamed_onnx_no_longer_matches_its_manifest_entry(self) -> None:
        """Unlike the old hash-based recognition, resolution is now purely by
        the manifest's declared filename: a copy under a different name is
        not the file the manifest names, even though its bytes are identical.
        """
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            renamed = bundle_dir / "renamed-age-model.onnx"
            renamed.write_bytes((bundle_dir / "age-v1.onnx").read_bytes())
            predictor = AgeAndGender()
            with self.assertRaisesRegex(ValueError, "does not name this file as the age model"):
                predictor.load_dnn_age_predictor(renamed)

    def test_explicit_onnx_with_no_manifest_at_all_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lone = Path(directory) / "age-v1.onnx"
            lone.write_bytes(b"not really an onnx graph")
            predictor = AgeAndGender()
            with self.assertRaises(FileNotFoundError):
                predictor.load_dnn_age_predictor(lone)

    def test_missing_path_raises_file_not_found(self) -> None:
        predictor = AgeAndGender()
        with self.assertRaises(FileNotFoundError):
            predictor.load_dnn_age_predictor(ROOT / "no-such-file.onnx")

    @unittest.skipUnless(
        AGE_DAT.is_file(), "example/models/*.dat is not available in this checkout"
    )
    def test_raw_dat_weights_are_refused_with_conversion_guidance(self) -> None:
        """The original dlib .dat weights cannot be loaded directly: there is
        no runtime path from a proprietary dlib serialization to ONNX.
        """
        predictor = AgeAndGender()
        with self.assertRaisesRegex(ValueError, "only .onnx files are accepted") as caught:
            predictor.load_dnn_age_predictor(AGE_DAT)
        self.assertIn("tools/conversion", str(caught.exception))

    def test_failed_load_leaves_the_previous_model_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = full_bundle(Path(directory) / "bundle")
            predictor = AgeAndGender()
            predictor.predict(_test_image_array())  # load the default session
            with self.assertRaises(ValueError):
                # names the gender model, not age
                predictor.load_dnn_age_predictor(bundle_dir / "gender-v1.onnx")
            self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)


@unittest.skipUnless(SHAPE_DAT.is_file(), "example/models/*.dat is not available in this checkout")
class ShapePredictorLoaderTests(unittest.TestCase):
    def test_known_file_loads_directly_and_predict_still_matches(self) -> None:
        predictor = AgeAndGender()
        predictor.load_shape_predictor(SHAPE_DAT)
        self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)

    def test_loading_before_first_predict_avoids_the_default_frontend(self) -> None:
        predictor = AgeAndGender()
        predictor.load_shape_predictor(SHAPE_DAT)
        self.assertIsNone(predictor._frontend)
        self.assertIsNotNone(predictor._predictor_override)
        self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)

    def test_loading_after_first_predict_replaces_the_live_frontend(self) -> None:
        predictor = AgeAndGender()
        predictor.predict(_test_image_array())
        self.assertIsNotNone(predictor._frontend)
        predictor.load_shape_predictor(SHAPE_DAT)
        self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)

    def test_missing_path_raises_file_not_found(self) -> None:
        predictor = AgeAndGender()
        with self.assertRaises(FileNotFoundError):
            predictor.load_shape_predictor(ROOT / "no-such-file.dat")

    def test_corrupt_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt-shape.dat"
            corrupt.write_bytes(b"not a dlib shape predictor")
            predictor = AgeAndGender()
            with self.assertRaises(ValueError):
                predictor.load_shape_predictor(corrupt)

    def test_neural_dat_is_not_a_shape_predictor(self) -> None:
        predictor = AgeAndGender()
        with self.assertRaises(ValueError):
            predictor.load_shape_predictor(AGE_DAT)

    def test_failed_load_leaves_the_previous_predictor_usable(self) -> None:
        predictor = AgeAndGender()
        predictor.predict(_test_image_array())  # build the default frontend
        with self.assertRaises(ValueError):
            predictor.load_shape_predictor(AGE_DAT)
        self.assertEqual(predictor.predict(_test_image_array()), EXPECTED_RESULTS)


if __name__ == "__main__":
    unittest.main()
