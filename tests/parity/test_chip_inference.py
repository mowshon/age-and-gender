"""Chip-level parity of the packaged runtime against frozen fixtures."""

from __future__ import annotations

import unittest

import numpy as np

from age_and_gender._inference import InferenceEngine, prepare_batch
from age_and_gender._models import bundled_models
from age_and_gender._postprocess import age_expectations, face_predictions
from tests.golden import GoldenFace, golden_faces, golden_images
from tests.parity.legacy_contract import compare_stage

PROBABILITY_TOLERANCE = {"atol": 1e-6, "rtol": 1e-4}
AGE_EXPECTATION_ATOL = 1e-4


class ChipInferenceTests(unittest.TestCase):
    """Run chip inference through the installed bundle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = bundled_models()
        cls.engine = InferenceEngine(cls.bundle)
        cls.images = golden_images()
        cls.faces = golden_faces()

    def probabilities(self, task: str, faces: list[GoldenFace]) -> np.ndarray:
        network = self.engine.network(task)
        return network.probabilities([face.chip(task) for face in faces])

    def test_corpus_is_the_frozen_one(self) -> None:
        self.assertEqual(len(self.images), 3)
        self.assertEqual(len(self.faces), 11)

    def test_preprocessing_reproduces_the_frozen_input_tensors(self) -> None:
        for face in self.faces:
            for task in ("age", "gender"):
                prepared = prepare_batch([face.chip(task)], self.bundle.spec(task))
                expected = face.input_tensor(task)
                self.assertEqual(prepared.dtype, expected.dtype, face.label)
                self.assertEqual(prepared.shape, expected.shape, face.label)
                self.assertTrue(
                    np.array_equal(prepared, expected),
                    f"{face.label} {task} input tensor is not bit-identical",
                )

    def test_probabilities_match_the_oracle(self) -> None:
        for task in ("age", "gender"):
            for face in self.faces:
                actual = self.probabilities(task, [face])
                expected = face.probabilities(task)[None, :]
                compare_stage(
                    image=face.image,
                    face_index=face.index,
                    model=task,
                    expected=expected.ravel().tolist(),
                    actual=actual.ravel().tolist(),
                    expected_shape=expected.shape,
                    actual_shape=actual.shape,
                    **PROBABILITY_TOLERANCE,
                )

    def test_age_expectation_matches_the_oracle(self) -> None:
        probabilities = self.probabilities("age", self.faces)
        expectations = age_expectations(probabilities, self.bundle.age.age_weights)
        for face, expectation in zip(self.faces, expectations):
            self.assertLessEqual(
                abs(float(expectation) - face.age_expectation),
                AGE_EXPECTATION_ATOL,
                face.label,
            )

    def test_public_results_match_the_oracle_exactly(self) -> None:
        for name, faces in self.images.items():
            results = face_predictions(
                [face.rectangle for face in faces],
                self.probabilities("gender", faces),
                self.probabilities("age", faces),
                labels=self.bundle.gender.labels,
                age_weights=self.bundle.age.age_weights,
            )
            self.assertEqual(results, [face.result for face in faces], name)

    def test_batching_faces_does_not_change_public_results(self) -> None:
        batched = face_predictions(
            [face.rectangle for face in self.faces],
            self.probabilities("gender", self.faces),
            self.probabilities("age", self.faces),
            labels=self.bundle.gender.labels,
            age_weights=self.bundle.age.age_weights,
        )
        individual = [
            face_predictions(
                [face.rectangle],
                self.probabilities("gender", [face]),
                self.probabilities("age", [face]),
                labels=self.bundle.gender.labels,
                age_weights=self.bundle.age.age_weights,
            )[0]
            for face in self.faces
        ]
        self.assertEqual(batched, individual)

    def test_duplicate_boxes_keep_their_order_and_results(self) -> None:
        """The explicit-box fixture repeats a face; both copies must survive."""
        faces = self.images["explicit-boxes.golden.json"]
        rectangles = [face.rectangle for face in faces]
        self.assertGreater(len(rectangles), len({tuple(box) for box in rectangles}))
        results = face_predictions(
            rectangles,
            self.probabilities("gender", faces),
            self.probabilities("age", faces),
            labels=self.bundle.gender.labels,
            age_weights=self.bundle.age.age_weights,
        )
        self.assertEqual(results, [face.result for face in faces])

    def test_one_session_per_model_serves_every_image(self) -> None:
        age, gender = self.engine.age, self.engine.gender
        self.probabilities("age", self.faces[:1])
        self.probabilities("gender", self.faces[:1])
        self.assertTrue(age.is_loaded and gender.is_loaded)
        sessions = (age._session, gender._session)
        for faces in self.images.values():
            self.probabilities("age", faces)
            self.probabilities("gender", faces)
        self.assertIs(self.engine.age, age)
        self.assertIs(self.engine.gender, gender)
        self.assertEqual((age._session, gender._session), sessions)


if __name__ == "__main__":
    unittest.main()
