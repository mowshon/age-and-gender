"""Bounded-batch chunking parity.

``NeuralNetwork.probabilities()`` splits a call larger than its
``max_batch_size`` into several ordered ONNX Runtime calls instead of handing
one unbounded batch to the session (see ``_inference.py``). This module checks
that splitting is invisible to the result: probabilities, the public age/gender
output, and face ordering all stay the same regardless of where the chunk
boundaries fall, and that chunking never re-reads weights or builds a second
session.

The already-known "batching is not always safe" trap is the individual chip
*extraction* dlib performs (``get_face_chips`` sharing a crop/pyramid across a
batch); that regression is guarded by
tests/parity/test_frontend.py::BatchingTrapRegressionTests and is unrelated to
the ONNX-side batching this module covers, which only ever chunks *already
individually extracted* chips.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from age_and_gender import AgeAndGender
from age_and_gender._images import as_rgb_array
from age_and_gender._inference import DEFAULT_MAX_BATCH_SIZE, InferenceEngine, NeuralNetwork
from age_and_gender._models import bundled_models, load_bundle
from age_and_gender._postprocess import face_predictions
from tests.bundles import PACKAGE_MODELS
from tests.golden import golden_documents, golden_faces

PROBABILITY_TOLERANCE = {"atol": 1e-6, "rtol": 1e-4}
ROOT = Path(__file__).resolve().parents[2]


class DefaultBoundTests(unittest.TestCase):
    """The shipped default bounds each runtime call."""

    def test_default_batch_size_is_32(self) -> None:
        self.assertEqual(DEFAULT_MAX_BATCH_SIZE, 32)

    def test_networks_use_the_default_bound_unless_overridden(self) -> None:
        engine = InferenceEngine(bundled_models())
        self.assertEqual(engine.age.max_batch_size, DEFAULT_MAX_BATCH_SIZE)
        self.assertEqual(engine.gender.max_batch_size, DEFAULT_MAX_BATCH_SIZE)

    def test_non_positive_bound_is_rejected(self) -> None:
        for invalid in (0, -1):
            with self.assertRaises(ValueError):
                NeuralNetwork(bundled_models(), "gender", max_batch_size=invalid)

    def test_non_integer_bound_is_rejected(self) -> None:
        # 1.5 would fail confusingly deep inside range()/slicing; True is an
        # int subclass that would silently become a batch size of 1.
        for invalid in (1.5, True, "32", None):
            with self.assertRaises(TypeError):
                NeuralNetwork(bundled_models(), "gender", max_batch_size=invalid)


class ChunkingParityTests(unittest.TestCase):
    """A chunked call must return exactly what one unbounded call would."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = bundled_models()
        # 11 real faces, repeated, gives distinct-valued chips spanning several
        # chunk boundaries rather than one chip repeated N times, so a bug that
        # swapped or dropped a row inside the chunking loop would show up as a
        # wrong face, not just a right value in the wrong number of rows.
        cls.faces = golden_faces() * 3  # 33 faces: one past the default bound of 32

    def chips(self, task: str) -> list[np.ndarray]:
        return [face.chip(task) for face in self.faces]

    def unbounded(self, task: str) -> np.ndarray:
        network = NeuralNetwork(self.bundle, task, max_batch_size=len(self.faces))
        return network.probabilities(self.chips(task))

    def test_chunked_probabilities_match_one_unbounded_call(self) -> None:
        for task in ("age", "gender"):
            with self.subTest(task=task):
                expected = self.unbounded(task)
                for bound in (1, 2, 3, 7, 32, 33):
                    network = NeuralNetwork(self.bundle, task, max_batch_size=bound)
                    actual = network.probabilities(self.chips(task))
                    self.assertEqual(actual.shape, expected.shape)
                    np.testing.assert_allclose(actual, expected, **PROBABILITY_TOLERANCE)

    def test_chunk_boundary_does_not_reorder_faces(self) -> None:
        """Repeats the corpus in a scrambled order so a swapped chunk boundary
        would misattribute a face's result to a neighboring row.
        """
        scrambled = list(reversed(golden_faces())) + golden_faces()[:5]
        for task in ("age", "gender"):
            with self.subTest(task=task):
                chips = [face.chip(task) for face in scrambled]
                full_batch = NeuralNetwork(self.bundle, task, max_batch_size=len(chips))
                small_chunks = NeuralNetwork(self.bundle, task, max_batch_size=4)
                np.testing.assert_allclose(
                    small_chunks.probabilities(chips),
                    full_batch.probabilities(chips),
                    **PROBABILITY_TOLERANCE,
                )

    def test_public_results_are_unaffected_by_chunk_boundaries(self) -> None:
        gender = NeuralNetwork(self.bundle, "gender", max_batch_size=5)
        age = NeuralNetwork(self.bundle, "age", max_batch_size=5)
        results = face_predictions(
            [face.rectangle for face in self.faces],
            gender.probabilities(self.chips("gender")),
            age.probabilities(self.chips("age")),
            labels=self.bundle.gender.labels,
            age_weights=self.bundle.age.age_weights,
        )
        expected = [face.result for face in self.faces]
        self.assertEqual(results, expected)

    def test_exactly_one_bound_worth_of_chips_takes_a_single_call(self) -> None:
        network = NeuralNetwork(self.bundle, "gender", max_batch_size=4)
        network.ensure_loaded()
        calls: list[int] = []
        original_run = network._session.run

        def spy(names, feed):
            calls.append(feed[network.spec.input_name].shape[0])
            return original_run(names, feed)

        network._session.run = spy  # type: ignore[method-assign]
        network.probabilities(self.chips("gender")[:4])
        self.assertEqual(calls, [4])

    def test_one_more_than_the_bound_splits_into_two_calls(self) -> None:
        network = NeuralNetwork(self.bundle, "gender", max_batch_size=4)
        network.ensure_loaded()
        calls: list[int] = []
        original_run = network._session.run

        def spy(names, feed):
            calls.append(feed[network.spec.input_name].shape[0])
            return original_run(names, feed)

        network._session.run = spy  # type: ignore[method-assign]
        network.probabilities(self.chips("gender")[:5])
        self.assertEqual(calls, [4, 1])


class EndToEndChunkBoundaryTests(unittest.TestCase):
    """The public ``AgeAndGender.predict()`` path at the chunk boundary.

    The classes above chunk pre-extracted chips directly through
    ``NeuralNetwork``/``face_predictions()``; a bug confined to how
    ``api.py``'s ``predict()`` wires those pieces together (for example,
    reusing the wrong extraction list, or breaking chunking's ordering
    together with ``FaceFrontend.extract()``'s own ordering) would not
    necessarily show up there. This drives the real public entry point at
    face counts straddling ``DEFAULT_MAX_BATCH_SIZE`` (32) on both sides.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = AgeAndGender()
        document = next(
            doc for doc in golden_documents() if doc.name == "explicit-boxes.golden.json"
        )
        with Image.open(ROOT / document.source_path) as image:
            cls.image = as_rgb_array(image.convert("RGB"))
        cls.boxes = document.input_boxes_trbl
        cls.expected = [face.result for face in document.faces]
        assert cls.boxes and len(cls.boxes) == len(cls.expected)

    def _tile(self, count: int) -> tuple[list, list]:
        """Repeat the golden explicit-box fixture out to exactly ``count`` faces.

        Every repeated box is identical, and extraction has no cross-face
        dependency (each face's landmarks/chips/network run only from its own
        rectangle), so the expected result for a duplicated box is simply the
        golden result duplicated in the same order.
        """
        repeats = math.ceil(count / len(self.boxes))
        return (self.boxes * repeats)[:count], (self.expected * repeats)[:count]

    def test_predict_matches_expected_at_chunk_boundaries(self) -> None:
        for count in (
            DEFAULT_MAX_BATCH_SIZE,
            DEFAULT_MAX_BATCH_SIZE + 1,
            DEFAULT_MAX_BATCH_SIZE * 2,
            DEFAULT_MAX_BATCH_SIZE * 2 + 1,
        ):
            with self.subTest(faces=count):
                boxes, expected = self._tile(count)
                results = self.predictor.predict(self.image, boxes)
                self.assertEqual(results, expected)


class ChunkingResourceLifecycleTests(unittest.TestCase):
    """Chunking must stay a pure Python-side split: one session, one load."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = bundled_models()
        cls.faces = golden_faces() * 4  # 44 chips: several chunks at bound 4

    def test_chunked_calls_reuse_the_one_session(self) -> None:
        network = NeuralNetwork(self.bundle, "gender", max_batch_size=4)
        self.assertFalse(network.is_loaded)
        network.probabilities([face.gender_chip for face in self.faces])
        session = network._session
        self.assertIsNotNone(session)
        network.probabilities([face.gender_chip for face in self.faces[:9]])
        self.assertIs(network._session, session)

    def test_chunking_does_not_reread_model_bytes(self) -> None:
        """Uses a fresh `load_bundle()` instance rather than the process-wide
        cached `bundled_models()` singleton (mirrors
        tests/unit/test_inference.py::test_predictions_do_not_reread_the_model),
        so monkeypatching `model_bytes` here cannot leak into other tests.
        """
        bundle = load_bundle(PACKAGE_MODELS)
        network = NeuralNetwork(bundle, "age", max_batch_size=3)
        calls: list[str] = []
        original = bundle.model_bytes

        def spy(task: str) -> bytes:
            calls.append(task)
            return original(task)

        bundle.model_bytes = spy  # type: ignore[method-assign]
        network.probabilities([face.age_chip for face in self.faces])
        self.assertEqual(len(calls), 1, "chunking re-read the model weights")

    def test_zero_chips_stay_a_single_no_op_call(self) -> None:
        network = NeuralNetwork(self.bundle, "gender", max_batch_size=4)
        probabilities = network.probabilities([])
        self.assertEqual(probabilities.shape, (0, network.spec.classes))
        self.assertFalse(network.is_loaded)


if __name__ == "__main__":
    unittest.main()
