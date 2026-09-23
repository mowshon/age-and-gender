from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from age_and_gender import _models
from age_and_gender._inference import InferenceEngine, NeuralNetwork, prepare_batch
from age_and_gender._models import bundled_models, load_bundle
from tests.bundles import corrupt_bytes, full_bundle
from tests.golden import golden_faces


class StubSession:
    """Stands in for a session so invalid model output can be exercised."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        return [self.output]


class StubTensor:
    """One graph input or output, shaped like onnxruntime's NodeArg."""

    def __init__(self, name: str, shape: list, tensor_type: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class StubGraph:
    """A session whose declared signature can be posed without building a graph."""

    def __init__(self, inputs: list[StubTensor], outputs: list[StubTensor]) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self) -> list[StubTensor]:
        return self._inputs

    def get_outputs(self) -> list[StubTensor]:
        return self._outputs


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = bundled_models()

    def test_uint8_chips_become_contiguous_float32_nchw(self) -> None:
        chips = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        batch = prepare_batch(chips, self.bundle.gender)
        self.assertEqual(batch.dtype, np.float32)
        self.assertEqual(batch.shape, (2, 3, 32, 32))
        self.assertTrue(batch.flags["C_CONTIGUOUS"])

    def test_channel_means_are_subtracted_per_channel(self) -> None:
        chips = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        batch = prepare_batch(chips, self.bundle.gender)
        means = np.asarray(self.bundle.gender.normalization.means, dtype=np.float32)
        for channel in range(3):
            self.assertEqual(batch[0, channel].min(), np.float32(-means[channel] / 256.0))
            self.assertEqual(batch[0, channel].max(), np.float32(-means[channel] / 256.0))

    def test_a_sequence_of_chips_is_stacked_in_order(self) -> None:
        first = np.full((32, 32, 3), 7, dtype=np.uint8)
        second = np.full((32, 32, 3), 9, dtype=np.uint8)
        batch = prepare_batch([first, second], self.bundle.gender)
        self.assertGreater(batch[1].min(), batch[0].min())

    def test_empty_input_keeps_the_batch_axis(self) -> None:
        self.assertEqual(prepare_batch([], self.bundle.age).shape, (0, 3, 64, 64))

    def test_chips_are_not_mutated(self) -> None:
        chips = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
        prepare_batch(chips, self.bundle.gender)
        self.assertTrue((chips == 128).all())

    def test_wrong_dtype_is_rejected(self) -> None:
        chips = np.zeros((1, 32, 32, 3), dtype=np.float32)
        with self.assertRaises(ValueError) as caught:
            prepare_batch(chips, self.bundle.gender)
        self.assertIn("uint8", str(caught.exception))

    def test_wrong_chip_size_is_rejected(self) -> None:
        chips = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        with self.assertRaises(ValueError) as caught:
            prepare_batch(chips, self.bundle.gender)
        self.assertIn("[N, 32, 32, 3]", str(caught.exception))

    def test_unbatched_chip_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prepare_batch(np.zeros((32, 32, 3), dtype=np.uint8), self.bundle.gender)


class SessionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = bundled_models()
        self.face = golden_faces()[0]

    def test_sessions_are_created_lazily_and_reused(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        self.assertFalse(network.is_loaded)
        network.probabilities([self.face.gender_chip])
        session = network._session
        self.assertIsNotNone(session)
        for _ in range(3):
            network.probabilities([self.face.gender_chip])
        self.assertIs(network._session, session)

    def test_predictions_neither_reread_nor_rehash_the_model(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        calls: list[int] = []
        original = _models.digest_bytes
        _models.digest_bytes = lambda payload: (calls.append(len(payload)), original(payload))[1]
        try:
            for _ in range(5):
                network.probabilities([self.face.gender_chip])
        finally:
            _models.digest_bytes = original
        self.assertEqual(len(calls), 1, "the weights were re-read during prediction")

    def test_zero_chips_never_build_a_session(self) -> None:
        engine = InferenceEngine(self.bundle)
        for network in (engine.age, engine.gender):
            probabilities = network.probabilities([])
            self.assertEqual(probabilities.shape, (0, network.spec.classes))
            self.assertEqual(probabilities.dtype, np.float32)
            self.assertFalse(network.is_loaded)

    def test_session_runs_on_the_declared_provider(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        network.ensure_loaded()
        self.assertEqual(network._session.get_providers(), ["CPUExecutionProvider"])

    def test_inputs_must_be_prepared_float32_nchw(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        with self.assertRaises(ValueError):
            network.run(np.zeros((1, 3, 32, 32), dtype=np.float64))
        with self.assertRaises(ValueError):
            network.run(np.zeros((1, 32, 32, 3), dtype=np.float32))
        with self.assertRaises(TypeError):
            network.run([[0.0]])  # type: ignore[arg-type]

    def test_non_contiguous_inputs_are_rejected(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        prepared = prepare_batch([self.face.gender_chip] * 3, self.bundle.gender)
        with self.assertRaises(ValueError) as caught:
            network.run(prepared[::2])
        self.assertIn("contiguous", str(caught.exception))

    def test_non_finite_model_output_is_rejected(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        network._session = StubSession(np.asarray([[np.nan, 0.5]], dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            network.probabilities([self.face.gender_chip])
        self.assertIn("non-finite", str(caught.exception))

    def test_out_of_range_model_output_is_rejected(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        network._session = StubSession(np.asarray([[-0.2, 1.2]], dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            network.probabilities([self.face.gender_chip])
        self.assertIn("[0, 1]", str(caught.exception))

    def test_unnormalized_model_output_is_rejected(self) -> None:
        network = NeuralNetwork(self.bundle, "age")
        network._session = StubSession(np.full((1, 81), 0.5, dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            network.probabilities([self.face.age_chip])
        self.assertIn("softmax", str(caught.exception))

    def test_wrong_output_width_is_rejected(self) -> None:
        network = NeuralNetwork(self.bundle, "age")
        network._session = StubSession(np.asarray([[0.5, 0.5]], dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            network.probabilities([self.face.age_chip])
        self.assertIn("81", str(caught.exception))

    def test_a_short_batch_of_rows_is_rejected(self) -> None:
        """Rows are matched to faces by position, so a missing row is fatal."""
        network = NeuralNetwork(self.bundle, "gender")
        network._session = StubSession(np.asarray([[0.25, 0.75]], dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            network.probabilities([self.face.gender_chip] * 2)
        self.assertIn("1 rows for 2 chips", str(caught.exception))

    def test_a_long_batch_of_rows_is_rejected(self) -> None:
        network = NeuralNetwork(self.bundle, "gender")
        network._session = StubSession(np.full((3, 2), 0.5, dtype=np.float32))
        with self.assertRaises(ValueError) as caught:
            network.probabilities([self.face.gender_chip])
        self.assertIn("3 rows for 1 chips", str(caught.exception))


class GraphSignatureTests(unittest.TestCase):
    """The declared signature is checked before a session is ever used."""

    def setUp(self) -> None:
        self.network = NeuralNetwork(bundled_models(), "gender")

    def check(self, inputs: list[StubTensor], outputs: list[StubTensor]) -> str:
        with self.assertRaises(ValueError) as caught:
            self.network._check_signature(StubGraph(inputs, outputs))
        return str(caught.exception)

    def test_the_packaged_graph_passes(self) -> None:
        self.network._check_signature(
            StubGraph(
                [StubTensor("images", ["N", 3, 32, 32])],
                [StubTensor("probabilities", ["N", 2])],
            )
        )

    def test_a_fixed_input_batch_is_rejected(self) -> None:
        message = self.check(
            [StubTensor("images", [1, 3, 32, 32])],
            [StubTensor("probabilities", ["N", 2])],
        )
        self.assertIn("fixed batch dimension", message)

    def test_a_fixed_output_batch_is_rejected(self) -> None:
        message = self.check(
            [StubTensor("images", ["N", 3, 32, 32])],
            [StubTensor("probabilities", [1, 2])],
        )
        self.assertIn("fixed batch dimension", message)

    def test_an_unrelated_output_batch_symbol_is_rejected(self) -> None:
        message = self.check(
            [StubTensor("images", ["N", 3, 32, 32])],
            [StubTensor("probabilities", ["M", 2])],
        )
        self.assertIn("one row per input chip", message)

    def test_a_mismatched_class_count_is_rejected(self) -> None:
        message = self.check(
            [StubTensor("images", ["N", 3, 32, 32])],
            [StubTensor("probabilities", ["N", 81])],
        )
        self.assertIn("[N, 2]", message)

    def test_a_mismatched_chip_size_is_rejected(self) -> None:
        message = self.check(
            [StubTensor("images", ["N", 3, 64, 64])],
            [StubTensor("probabilities", ["N", 2])],
        )
        self.assertIn("[N, 3, 32, 32]", message)


class BrokenBundleSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.tmp = Path(self._directory.name)

    def test_swapped_age_and_gender_graphs_are_detected(self) -> None:
        """Hashes alone cannot catch this: the manifest names the wrong graph."""

        def swap(manifest: dict) -> None:
            age = manifest["models"]["age"]["artifact"]
            gender = manifest["models"]["gender"]["artifact"]
            manifest["models"]["age"]["artifact"] = gender
            manifest["models"]["gender"]["artifact"] = age

        bundle = load_bundle(full_bundle(self.tmp / "swapped", swap))
        with self.assertRaises(ValueError) as caught:
            NeuralNetwork(bundle, "age").ensure_loaded()
        message = str(caught.exception)
        self.assertIn("gender-v1.onnx", message)
        self.assertIn("64", message)

    def test_unparsable_graph_names_the_task(self) -> None:
        """A file that hashes correctly but is not a graph still fails cleanly."""
        bundle_path = self.tmp / "garbage"
        payload = b"not an onnx graph"

        def repoint(manifest: dict) -> None:
            artifact = manifest["models"]["gender"]["artifact"]
            artifact["bytes"] = len(payload)
            artifact["sha256"] = hashlib.sha256(payload).hexdigest()

        full_bundle(bundle_path, repoint)
        (bundle_path / "gender-v1.onnx").write_bytes(payload)
        bundle = load_bundle(bundle_path)
        with self.assertRaises(ValueError) as caught:
            NeuralNetwork(bundle, "gender").ensure_loaded()
        self.assertIn("gender model", str(caught.exception))


class ReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.tmp = Path(self._directory.name)
        self.engine = InferenceEngine(bundled_models())
        self.face = golden_faces()[0]

    def test_successful_replacement_swaps_one_task_only(self) -> None:
        gender_before = self.engine.gender
        replacement = load_bundle(full_bundle(self.tmp / "same"))
        installed = self.engine.replace("age", replacement)
        self.assertIs(self.engine.age, installed)
        self.assertTrue(installed.is_loaded)
        self.assertIs(self.engine.gender, gender_before)
        np.testing.assert_allclose(
            self.engine.age.probabilities([self.face.age_chip]),
            self.face.age_probabilities[None, :],
            atol=1e-6,
            rtol=1e-4,
        )

    def test_failed_replacement_leaves_the_working_model_in_place(self) -> None:
        original = self.engine.age
        original.ensure_loaded()
        session = original._session
        broken = full_bundle(self.tmp / "broken")
        corrupt_bytes(broken / "age-v1.onnx")
        with self.assertRaises(ValueError):
            self.engine.replace("age", load_bundle(broken))
        self.assertIs(self.engine.age, original)
        self.assertIs(original._session, session)
        np.testing.assert_allclose(
            self.engine.age.probabilities([self.face.age_chip]),
            self.face.age_probabilities[None, :],
            atol=1e-6,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
