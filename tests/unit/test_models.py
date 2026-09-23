from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from age_and_gender import _models
from age_and_gender._models import (
    ModelBundle,
    bundled_models,
    digest_file,
    load_bundle,
)
from tests.bundles import (
    CONVERSION_BUNDLE,
    PACKAGE_MODELS,
    ROOT,
    corrupt_bytes,
    full_bundle,
    manifest_only,
    package_manifest,
    write_manifest,
)

LEGACY_AGE_SHA = "4b78d4d7055e22620e362884b5551caa9379080277338aa2d4cdfc592f0e9fa3"
LEGACY_GENDER_SHA = "85453d6f6585c8e02ada95929956783c780dc04dcec5bdfd14af82f15c99ba41"
LEGACY_SHAPE_SHA = "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888"

# An audit hook records every file the interpreter opens while the package is
# imported. Run in a child process so the import is genuinely the first one.
IMPORT_PROBE = """
import json, sys

opened = []
sys.addaudithook(lambda event, args: opened.append(str(args[0])) if event == "open" else None)

import age_and_gender

print(json.dumps({
    "version": age_and_gender.__version__,
    "model_files": [
        path
        for path in opened
        if path.endswith((".onnx", ".dat")) or path.endswith("manifest.json")
    ],
}))
"""


class TempBundleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.tmp = Path(self._directory.name)

    def assertRejects(self, bundle: Path, *fragments: str) -> None:
        with self.assertRaises(ValueError) as caught:
            load_bundle(bundle)
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)


class BundledResourceTests(unittest.TestCase):
    def test_installed_bundle_is_resolved_and_cached(self) -> None:
        bundle = bundled_models()
        self.assertIs(bundle, bundled_models())
        self.assertEqual(bundle.bundle_id, "age-and-gender-v1")
        self.assertEqual(bundle.age.chip_size, 64)
        self.assertEqual(bundle.age.classes, 81)
        self.assertEqual(bundle.gender.chip_size, 32)
        self.assertEqual(bundle.gender.labels, ("female", "male"))
        self.assertEqual(bundle.shape_predictor.parts, 5)

    def test_resolution_does_not_depend_on_the_working_directory(self) -> None:
        """importlib.resources, not Path.cwd(), decides where the models are."""
        bundle = bundled_models()
        self.assertNotIn(str(Path.cwd()), bundle.origin)
        with bundle.shape_predictor_file() as path:
            self.assertTrue(path.is_file())
            self.assertEqual(digest_file(path), LEGACY_SHAPE_SHA)

    def test_runtime_block_is_the_validated_configuration(self) -> None:
        runtime = bundled_models().runtime
        self.assertEqual(runtime.provider, "CPUExecutionProvider")
        self.assertEqual(runtime.graph_optimization_level, "ORT_DISABLE_ALL")
        self.assertEqual(runtime.intra_op_num_threads, 1)
        self.assertEqual(runtime.inter_op_num_threads, 1)

    def test_normalization_and_age_weights_come_from_the_converter(self) -> None:
        bundle = bundled_models()
        for spec in (bundle.age, bundle.gender):
            self.assertEqual(spec.normalization.means, (122.781998, 117.000999, 104.297997))
            self.assertEqual(spec.normalization.scale, 1.0 / 256.0)
        weights = bundle.age.age_weights
        assert weights is not None
        self.assertEqual(weights[0], 0.25)
        self.assertEqual(weights[1:], tuple(float(index) for index in range(1, 81)))

    def test_model_bytes_match_the_manifest(self) -> None:
        bundle = load_bundle(PACKAGE_MODELS)
        payload = bundle.model_bytes("age")
        self.assertEqual(len(payload), bundle.age.size_bytes)
        self.assertEqual(_models.digest_bytes(payload), bundle.age.sha256)

    def test_importing_the_package_opens_no_model_resource(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
        completed = subprocess.run(
            [sys.executable, "-c", IMPORT_PROBE],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(ROOT),
        )
        result = json.loads(completed.stdout)
        self.assertEqual(result["model_files"], [])
        self.assertTrue(result["version"])

    def test_notices_ship_with_the_models_and_match_their_digests(self) -> None:
        manifest = package_manifest()
        notices = (
            manifest["license"]["notice"],
            manifest["shape_predictor"]["license"]["notice"],
        )
        self.assertEqual(len({notice["path"] for notice in notices}), 2)
        for notice in notices:
            path = PACKAGE_MODELS / notice["path"]
            self.assertTrue(path.is_file(), notice["path"])
            self.assertEqual(digest_file(path), notice["sha256"], notice["path"])

    def test_package_manifest_agrees_with_the_conversion_bundle(self) -> None:
        """The package bundle is assembled from PR-2's artifacts, not re-derived."""
        conversion = json.loads((CONVERSION_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        package = package_manifest()
        self.assertEqual(package["bundle_kind"], "package")
        for key, value in conversion.items():
            self.assertEqual(package[key], value, key)


class LegacySourceMappingTests(unittest.TestCase):
    def test_known_source_hashes_map_to_their_bundle_role(self) -> None:
        bundle = bundled_models()
        self.assertEqual(bundle.source_role(LEGACY_AGE_SHA), "age")
        self.assertEqual(bundle.source_role(LEGACY_GENDER_SHA), "gender")
        self.assertEqual(bundle.source_role(LEGACY_SHAPE_SHA), "shape_predictor")
        self.assertEqual(bundle.source_role(LEGACY_AGE_SHA.upper()), "age")

    def test_unknown_source_hashes_do_not_select_a_default(self) -> None:
        bundle = bundled_models()
        self.assertIsNone(bundle.source_role("0" * 64))
        self.assertEqual(
            set(bundle.source_digests),
            {LEGACY_AGE_SHA, LEGACY_GENDER_SHA, LEGACY_SHAPE_SHA},
        )

    def test_source_digests_is_a_copy(self) -> None:
        bundle = bundled_models()
        digests = bundle.source_digests
        digests["0" * 64] = "age"  # type: ignore[index]
        self.assertIsNone(bundle.source_role("0" * 64))


class CorruptResourceTests(TempBundleTestCase):
    def test_flipped_model_byte_is_rejected(self) -> None:
        bundle = full_bundle(self.tmp / "flipped")
        corrupt_bytes(bundle / "age-v1.onnx")
        with self.assertRaises(ValueError) as caught:
            load_bundle(bundle).model_bytes("age")
        self.assertIn("sha256", str(caught.exception))

    def test_truncated_model_is_rejected_by_size(self) -> None:
        bundle = full_bundle(self.tmp / "short")
        path = bundle / "gender-v1.onnx"
        path.write_bytes(path.read_bytes()[:-16])
        with self.assertRaises(ValueError) as caught:
            load_bundle(bundle).model_bytes("gender")
        self.assertIn("bytes", str(caught.exception))

    def test_corrupt_shape_predictor_is_rejected(self) -> None:
        bundle = full_bundle(self.tmp / "predictor")
        corrupt_bytes(bundle / "shape_predictor_5_face_landmarks.dat")
        loaded = load_bundle(bundle)
        with self.assertRaises(ValueError) as caught, loaded.shape_predictor_file():
            pass
        self.assertIn("sha256", str(caught.exception))

    def test_a_model_changed_after_a_good_read_is_rejected(self) -> None:
        """A bundle that verified once must not hand out changed bytes later."""
        path = full_bundle(self.tmp / "changed")
        bundle = load_bundle(path)
        good = bundle.model_bytes("age")
        corrupt_bytes(path / "age-v1.onnx")
        with self.assertRaises(ValueError) as caught:
            bundle.model_bytes("age")
        self.assertIn("sha256", str(caught.exception))
        self.assertEqual(_models.digest_bytes(good), bundle.age.sha256)

    def test_a_predictor_changed_after_a_good_read_is_rejected(self) -> None:
        path = full_bundle(self.tmp / "changed-predictor")
        bundle = load_bundle(path)
        with bundle.shape_predictor_file() as first:
            self.assertTrue(first.is_file())
        corrupt_bytes(path / "shape_predictor_5_face_landmarks.dat")
        with self.assertRaises(ValueError), bundle.shape_predictor_file():
            pass

    def test_missing_artifact_raises_file_not_found(self) -> None:
        bundle = full_bundle(self.tmp / "missing")
        (bundle / "age-v1.onnx").unlink()
        with self.assertRaises(FileNotFoundError):
            load_bundle(bundle).model_bytes("age")

    def test_missing_manifest_raises_file_not_found(self) -> None:
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_bundle(empty)

    def test_unreadable_manifest_names_the_bundle(self) -> None:
        broken = self.tmp / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("{not json", encoding="utf-8")
        self.assertRejects(broken, "not valid JSON")


class UnsupportedManifestTests(TempBundleTestCase):
    def test_newer_schema_version_is_refused(self) -> None:
        self.assertRejects(
            manifest_only(self.tmp / "schema", lambda m: m.__setitem__("schema_version", 2)),
            "schema_version",
        )

    def test_conversion_bundle_is_not_a_package_bundle(self) -> None:
        """PR-2's artifact directory has no landmark model, and says so clearly."""
        self.assertRejects(CONVERSION_BUNDLE, "bundle_kind", "build_bundle.py")

    def test_missing_runtime_block_is_refused(self) -> None:
        self.assertRejects(
            manifest_only(self.tmp / "no-runtime", lambda m: m.pop("runtime")),
            "no runtime block",
        )

    def test_different_runtime_options_are_refused(self) -> None:
        def relax(manifest: dict) -> None:
            manifest["runtime"]["graph_optimization_level"] = "ORT_ENABLE_ALL"

        self.assertRejects(
            manifest_only(self.tmp / "optimized", relax), "differs from the validated"
        )

    def test_extra_threads_are_refused(self) -> None:
        def widen(manifest: dict) -> None:
            manifest["runtime"]["intra_op_num_threads"] = 4

        self.assertRejects(manifest_only(self.tmp / "threads", widen), "runtime")

    def test_mislabelled_task_is_refused(self) -> None:
        def swap(manifest: dict) -> None:
            manifest["models"]["age"]["task"] = "gender"

        self.assertRejects(manifest_only(self.tmp / "task", swap), "declares task")

    def test_wrong_input_shape_is_refused(self) -> None:
        def resize(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["shape"] = ["N", 3, 32, 32]

        self.assertRejects(manifest_only(self.tmp / "shape", resize), "input.shape")

    def test_wrong_class_count_is_refused(self) -> None:
        def reclassify(manifest: dict) -> None:
            manifest["models"]["gender"]["output"]["shape"] = ["N", 3]

        self.assertRejects(manifest_only(self.tmp / "classes", reclassify), "output.shape")

    def test_non_float_graphs_are_refused(self) -> None:
        def quantize(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["dtype"] = "uint8"

        self.assertRejects(manifest_only(self.tmp / "dtype", quantize), "float32")

    def test_logit_outputs_are_refused(self) -> None:
        def strip(manifest: dict) -> None:
            manifest["models"]["gender"]["output"]["softmax_in_graph"] = False

        self.assertRejects(manifest_only(self.tmp / "logits", strip), "softmax")

    def test_changed_normalization_is_refused(self) -> None:
        def rescale(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["normalization"]["scale"] = 1 / 255

        self.assertRejects(manifest_only(self.tmp / "scale", rescale), "division by 256")

    def test_missing_means_are_refused(self) -> None:
        def drop(manifest: dict) -> None:
            manifest["models"]["gender"]["input"]["normalization"]["means"] = [1.0, 2.0]

        self.assertRejects(manifest_only(self.tmp / "means", drop), "three numbers")

    def test_bgr_inputs_are_refused(self) -> None:
        def flip(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["color_order"] = "BGR"

        self.assertRejects(manifest_only(self.tmp / "bgr", flip), "color_order")

    def test_reordered_gender_labels_are_refused(self) -> None:
        def reorder(manifest: dict) -> None:
            manifest["models"]["gender"]["labels"] = ["male", "female"]

        self.assertRejects(manifest_only(self.tmp / "labels", reorder), "labels")

    def test_changed_age_class_weights_are_refused(self) -> None:
        def shift(manifest: dict) -> None:
            manifest["models"]["age"]["age_weights"][0] = 0.0

        self.assertRejects(manifest_only(self.tmp / "weights", shift), "class weights")

    def test_short_age_class_weights_are_refused(self) -> None:
        def truncate(manifest: dict) -> None:
            manifest["models"]["age"]["age_weights"] = [0.25, 1, 2]

        self.assertRejects(manifest_only(self.tmp / "short-weights", truncate), "81")

    def test_missing_model_section_is_refused(self) -> None:
        self.assertRejects(
            manifest_only(self.tmp / "no-gender", lambda m: m["models"].pop("gender")),
            "models.gender",
        )

    def test_missing_shape_predictor_is_refused(self) -> None:
        self.assertRejects(
            manifest_only(self.tmp / "no-predictor", lambda m: m.pop("shape_predictor")),
            "shape_predictor",
        )

    def test_non_five_point_predictor_is_refused(self) -> None:
        def sixty_eight(manifest: dict) -> None:
            manifest["shape_predictor"]["parts"] = 68

        self.assertRejects(manifest_only(self.tmp / "parts", sixty_eight), "five-point")

    def test_manifest_must_be_an_object(self) -> None:
        directory = self.tmp / "list"
        directory.mkdir()
        (directory / "manifest.json").write_text("[]", encoding="utf-8")
        self.assertRejects(directory, "JSON object")

    def test_malformed_artifact_entry_is_refused(self) -> None:
        def blank(manifest: dict) -> None:
            manifest["models"]["age"]["artifact"]["sha256"] = None

        self.assertRejects(manifest_only(self.tmp / "artifact", blank), "artifact.sha256")

    def test_rejected_manifests_do_not_disturb_the_installed_bundle(self) -> None:
        with self.assertRaises(ValueError):
            load_bundle(manifest_only(self.tmp / "bad", lambda m: m.pop("runtime")))
        self.assertIsInstance(bundled_models(), ModelBundle)
        self.assertEqual(bundled_models().bundle_id, "age-and-gender-v1")


class ManifestRoundTripTests(TempBundleTestCase):
    def test_an_unmodified_copy_still_loads(self) -> None:
        bundle = load_bundle(write_manifest(self.tmp / "copy", package_manifest()))
        self.assertEqual(bundle.bundle_id, "age-and-gender-v1")
        self.assertEqual(bundle.origin, str(self.tmp / "copy"))


if __name__ == "__main__":
    unittest.main()
