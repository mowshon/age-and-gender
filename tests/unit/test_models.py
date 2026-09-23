from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from age_and_gender._models import ModelBundle, bundled_models, load_bundle
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            self.assertGreater(path.stat().st_size, 0)

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

    def test_model_bytes_are_read_from_the_artifact_file(self) -> None:
        bundle = load_bundle(PACKAGE_MODELS)
        payload = bundle.model_bytes("age")
        self.assertEqual(payload, (PACKAGE_MODELS / bundle.age.filename).read_bytes())

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
        """A repo-content check: the manifest's recorded notice digests still
        match the shipped notice files. This is unrelated to model loading
        (notices are never read by `_models.py` at runtime, by design — see
        spec/PR-3.md's review follow-ups), so it hashes directly rather than
        through any package API.
        """
        manifest = package_manifest()
        notices = (
            manifest["license"]["notice"],
            manifest["shape_predictor"]["license"]["notice"],
        )
        self.assertEqual(len({notice["path"] for notice in notices}), 2)
        for notice in notices:
            path = PACKAGE_MODELS / notice["path"]
            self.assertTrue(path.is_file(), notice["path"])
            self.assertEqual(_sha256(path), notice["sha256"], notice["path"])

    def test_package_manifest_agrees_with_the_conversion_bundle(self) -> None:
        """The package bundle is assembled from PR-2's artifacts, not re-derived."""
        conversion = json.loads((CONVERSION_BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        package = package_manifest()
        self.assertEqual(package["bundle_kind"], "package")
        for key, value in conversion.items():
            self.assertEqual(package[key], value, key)


class UnverifiedArtifactContentTests(TempBundleTestCase):
    """Loading is structural, not cryptographic: a bundle only has to satisfy
    the manifest's declared task/shape/normalization contract, not match a
    recorded hash. These tests guard that decision directly, so a future
    change cannot silently reintroduce a hash gate without a test noticing.
    """

    def test_model_bytes_are_returned_even_when_corrupted(self) -> None:
        bundle_dir = full_bundle(self.tmp / "flipped")
        corrupt_bytes(bundle_dir / "age-v1.onnx")
        corrupted = (bundle_dir / "age-v1.onnx").read_bytes()
        self.assertEqual(load_bundle(bundle_dir).model_bytes("age"), corrupted)

    def test_model_bytes_are_returned_even_when_truncated(self) -> None:
        bundle_dir = full_bundle(self.tmp / "short")
        path = bundle_dir / "gender-v1.onnx"
        path.write_bytes(path.read_bytes()[:-16])
        self.assertEqual(load_bundle(bundle_dir).model_bytes("gender"), path.read_bytes())

    def test_shape_predictor_file_is_yielded_even_when_corrupted(self) -> None:
        bundle_dir = full_bundle(self.tmp / "predictor")
        corrupt_bytes(bundle_dir / "shape_predictor_5_face_landmarks.dat")
        with load_bundle(bundle_dir).shape_predictor_file() as path:
            self.assertTrue(path.is_file())

    def test_model_bytes_reflect_the_current_file_not_a_cached_read(self) -> None:
        bundle_dir = full_bundle(self.tmp / "changed")
        bundle = load_bundle(bundle_dir)
        original = bundle.model_bytes("age")
        corrupt_bytes(bundle_dir / "age-v1.onnx")
        changed = bundle.model_bytes("age")
        self.assertNotEqual(original, changed)
        self.assertEqual(changed, (bundle_dir / "age-v1.onnx").read_bytes())

    def test_missing_artifact_still_raises_file_not_found(self) -> None:
        """Dropping hash verification does not drop existence checking."""
        bundle_dir = full_bundle(self.tmp / "missing")
        (bundle_dir / "age-v1.onnx").unlink()
        with self.assertRaises(FileNotFoundError):
            load_bundle(bundle_dir).model_bytes("age")

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


class RelaxedArtifactMetadataTests(TempBundleTestCase):
    """A caller pointing this at their own converted model should not have to
    fabricate hash/provenance metadata just to satisfy the schema.
    """

    def test_artifact_without_sha256_or_bytes_still_loads(self) -> None:
        def strip(manifest: dict) -> None:
            del manifest["models"]["age"]["artifact"]["sha256"]
            del manifest["models"]["age"]["artifact"]["bytes"]

        bundle_dir = full_bundle(self.tmp / "no-hash", strip)
        bundle = load_bundle(bundle_dir)
        self.assertIsNone(bundle.age.sha256)
        self.assertIsNone(bundle.age.size_bytes)
        self.assertTrue(bundle.model_bytes("age"))

    def test_missing_source_block_still_loads(self) -> None:
        def drop(manifest: dict) -> None:
            del manifest["models"]["age"]["source"]
            del manifest["shape_predictor"]["source"]

        bundle_dir = full_bundle(self.tmp / "no-source", drop)
        bundle = load_bundle(bundle_dir)
        self.assertIsNone(bundle.age.source_filename)
        self.assertIsNone(bundle.shape_predictor.source_filename)


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

    def test_null_age_class_weight_is_refused_as_a_value_error(self) -> None:
        """A `float(None)` inside the parser would raise a raw TypeError
        instead of the documented ValueError; the length/type check must
        catch this before any element is converted.
        """

        def nullify(manifest: dict) -> None:
            manifest["models"]["age"]["age_weights"][5] = None

        self.assertRejects(manifest_only(self.tmp / "null-weight", nullify), "class weights")

    def test_boolean_age_class_weight_is_refused(self) -> None:
        def booleanize(manifest: dict) -> None:
            manifest["models"]["age"]["age_weights"][5] = True

        self.assertRejects(manifest_only(self.tmp / "bool-weight", booleanize), "class weights")

    def test_path_traversal_in_artifact_filename_is_refused(self) -> None:
        def escape(manifest: dict) -> None:
            manifest["models"]["age"]["artifact"]["filename"] = "../outside.onnx"

        self.assertRejects(manifest_only(self.tmp / "traversal", escape), "not a plain filename")

    def test_absolute_artifact_filename_is_refused(self) -> None:
        def absolute(manifest: dict) -> None:
            manifest["models"]["age"]["artifact"]["filename"] = "/etc/passwd"

        self.assertRejects(manifest_only(self.tmp / "absolute", absolute), "not a plain filename")

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

    def test_missing_artifact_filename_is_refused(self) -> None:
        def blank(manifest: dict) -> None:
            del manifest["models"]["age"]["artifact"]["filename"]

        self.assertRejects(manifest_only(self.tmp / "artifact", blank), "artifact.filename")

    def test_malformed_artifact_filename_is_refused(self) -> None:
        def wrong_type(manifest: dict) -> None:
            manifest["models"]["age"]["artifact"]["filename"] = 123

        self.assertRejects(
            manifest_only(self.tmp / "artifact-type", wrong_type), "artifact.filename"
        )

    def test_malformed_artifact_sha256_is_refused_when_present(self) -> None:
        """sha256 is optional, but a present-and-malformed value is still an error."""

        def wrong_type(manifest: dict) -> None:
            manifest["models"]["age"]["artifact"]["sha256"] = 123

        self.assertRejects(manifest_only(self.tmp / "sha-type", wrong_type), "artifact.sha256")

    def test_null_input_shape_is_refused_as_a_value_error(self) -> None:
        """`list(None)` would raise a raw TypeError; a null shape must be
        reported the same documented way as a missing or wrong-length one.
        """

        def nullify(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["shape"] = None

        self.assertRejects(manifest_only(self.tmp / "null-shape", nullify), "input.shape")

    def test_non_list_output_shape_is_refused_as_a_value_error(self) -> None:
        def wrong_type(manifest: dict) -> None:
            manifest["models"]["gender"]["output"]["shape"] = 7

        self.assertRejects(manifest_only(self.tmp / "int-shape", wrong_type), "output.shape")

    def test_non_list_gender_labels_is_refused_as_a_value_error(self) -> None:
        """`list(labels or [])` would raise a raw TypeError for a truthy
        non-iterable value such as a bare int.
        """

        def wrong_type(manifest: dict) -> None:
            manifest["models"]["gender"]["labels"] = 7

        self.assertRejects(manifest_only(self.tmp / "int-labels", wrong_type), "labels")

    def test_nan_normalization_mean_is_refused(self) -> None:
        def nanify(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["normalization"]["means"][0] = float("nan")

        self.assertRejects(manifest_only(self.tmp / "nan-mean", nanify), "finite")

    def test_infinite_normalization_mean_is_refused(self) -> None:
        def infinitize(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["normalization"]["means"][1] = float("inf")

        self.assertRejects(manifest_only(self.tmp / "inf-mean", infinitize), "finite")

    def test_float32_overflowing_normalization_mean_is_refused(self) -> None:
        """Finite as float64 but `inf` once cast to float32, the dtype every
        network actually computes in.
        """

        def overflow(manifest: dict) -> None:
            manifest["models"]["age"]["input"]["normalization"]["means"][2] = 1e40

        self.assertRejects(manifest_only(self.tmp / "overflow-mean", overflow), "finite")

    def test_rejected_manifests_do_not_disturb_the_installed_bundle(self) -> None:
        with self.assertRaises(ValueError):
            load_bundle(manifest_only(self.tmp / "bad", lambda m: m.pop("runtime")))
        self.assertIsInstance(bundled_models(), ModelBundle)
        self.assertEqual(bundled_models().bundle_id, "age-and-gender-v1")


class SymlinkContainmentTests(TempBundleTestCase):
    """`_check_flat_filename()` only rejects traversal spelled out in the
    manifest's filename string; a flat name that is itself a symlink pointing
    outside the bundle directory must be refused too, or `from_model_dir()`'s
    "backed entirely by this directory" guarantee would not hold.
    """

    def test_symlinked_neural_artifact_outside_the_bundle_is_refused(self) -> None:
        bundle_dir = full_bundle(self.tmp / "symlinked-age")
        target = self.tmp / "outside.onnx"
        target.write_bytes(b"not a real model, just needs to exist")
        artifact = bundle_dir / "age-v1.onnx"
        artifact.unlink()
        artifact.symlink_to(target)
        with self.assertRaises(ValueError) as caught:
            load_bundle(bundle_dir).model_bytes("age")
        self.assertIn("outside the bundle directory", str(caught.exception))

    def test_symlinked_shape_predictor_outside_the_bundle_is_refused(self) -> None:
        bundle_dir = full_bundle(self.tmp / "symlinked-predictor")
        target = self.tmp / "outside.dat"
        target.write_bytes(b"not a real predictor, just needs to exist")
        artifact = bundle_dir / "shape_predictor_5_face_landmarks.dat"
        artifact.unlink()
        artifact.symlink_to(target)
        with (
            self.assertRaises(ValueError) as caught,
            load_bundle(bundle_dir).shape_predictor_file(),
        ):
            pass
        self.assertIn("outside the bundle directory", str(caught.exception))

    def test_symlink_that_stays_inside_the_bundle_is_accepted(self) -> None:
        """A symlink is only refused for escaping the bundle root, not for
        existing at all: this would be a false positive if it were rejected.
        """
        bundle_dir = full_bundle(self.tmp / "internal-symlink")
        artifact = bundle_dir / "age-v1.onnx"
        real_bytes = artifact.read_bytes()
        renamed = bundle_dir / "age-v1-real.onnx"
        artifact.rename(renamed)
        artifact.symlink_to(renamed)
        self.assertEqual(load_bundle(bundle_dir).model_bytes("age"), real_bytes)


class ManifestRoundTripTests(TempBundleTestCase):
    def test_an_unmodified_copy_still_loads(self) -> None:
        bundle = load_bundle(write_manifest(self.tmp / "copy", package_manifest()))
        self.assertEqual(bundle.bundle_id, "age-and-gender-v1")
        self.assertEqual(bundle.origin, str(self.tmp / "copy"))


if __name__ == "__main__":
    unittest.main()
