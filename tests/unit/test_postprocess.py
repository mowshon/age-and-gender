from __future__ import annotations

import math
import unittest

import numpy as np

from age_and_gender._models import bundled_models
from age_and_gender._postprocess import (
    age_expectations,
    age_predictions,
    confidence_percents,
    face_predictions,
    gender_predictions,
    round_ages,
)
from tests.golden import golden_faces
from tests.parity.legacy_contract import age_expectation, confidence_percent, gender_result
from tests.parity.legacy_contract import round_age as reference_round_age

AGE_WEIGHTS = (0.25, *range(1, 81))
GENDER_LABELS = ("female", "male")
SEED = 20260922


def probability_rows(count: int, classes: int, seed: int = SEED) -> np.ndarray:
    """Deterministic normalized float32 probability rows."""
    raw = np.random.default_rng(seed).random((count, classes)).astype(np.float32)
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


class AgeReductionTests(unittest.TestCase):
    def test_matches_the_frozen_scalar_contract_bit_for_bit(self) -> None:
        rows = probability_rows(64, 81)
        computed = age_expectations(rows, AGE_WEIGHTS)
        for index, row in enumerate(rows):
            expected = age_expectation([float(value) for value in row])
            self.assertEqual(float(computed[index]), expected, f"row {index}")

    def test_matches_the_oracle_expectations(self) -> None:
        for face in golden_faces():
            computed = age_expectations(face.age_probabilities[None, :], AGE_WEIGHTS)
            self.assertAlmostEqual(float(computed[0]), face.age_expectation, delta=1e-4)

    def test_sequential_float32_differs_from_a_dot_product(self) -> None:
        """Why the class loop exists: reassociating the sum is not equivalent."""
        rows = probability_rows(64, 81)
        sequential = age_expectations(rows, AGE_WEIGHTS)
        dotted = rows @ np.asarray(AGE_WEIGHTS, dtype=np.float32)
        self.assertTrue((sequential != dotted).any())
        for index, row in enumerate(rows):
            self.assertEqual(float(sequential[index]), age_expectation([float(v) for v in row]))

    def test_class_zero_counts_as_a_quarter_year(self) -> None:
        certain = np.zeros((1, 81), dtype=np.float32)
        certain[0, 0] = np.float32(1.0)
        self.assertEqual(float(age_expectations(certain, AGE_WEIGHTS)[0]), 0.25)

    def test_expectation_is_not_the_most_likely_class(self) -> None:
        rows = probability_rows(64, 81)
        ages = [prediction["value"] for prediction in age_predictions(rows, AGE_WEIGHTS)]
        self.assertNotEqual(ages, [int(index) for index in rows.argmax(axis=1)])

    def test_weight_count_must_match_the_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            age_expectations(np.zeros((1, 80), dtype=np.float32), AGE_WEIGHTS)


class AgeRoundingTests(unittest.TestCase):
    def test_half_values_round_away_from_zero(self) -> None:
        cases = {0.0: 0, 0.25: 0, 0.5: 1, 24.5: 25, 25.5: 26, 25.4999: 25, 80.5: 81}
        expectations = np.asarray(list(cases), dtype=np.float32)
        self.assertEqual(round_ages(expectations), list(cases.values()))

    def test_matches_the_frozen_scalar_rounding(self) -> None:
        expectations = np.asarray(
            [0.0, 0.4999999, 0.5, 1.4999999, 1.5, 26.499998, 26.5, 79.999992],
            dtype=np.float32,
        )
        self.assertEqual(
            round_ages(expectations),
            [reference_round_age(float(value)) for value in expectations],
        )

    def test_the_half_is_added_in_float64(self) -> None:
        """float32(x + 0.5) rounds a second time and crosses the boundary here.

        This is the only half-boundary in the age range where the two differ:
        above it, ``x + 0.5`` stays representable at the same float32 exponent,
        so a float32 add happens to give the right answer anyway.
        """
        value = np.nextafter(np.float32(0.5), np.float32(0.0))
        self.assertEqual(round_ages(np.asarray([value]))[0], 0)
        self.assertEqual(reference_round_age(float(value)), 0)
        # What the implementation must not do:
        self.assertEqual(math.floor(float(np.float32(float(value) + 0.5))), 1)

    def test_returns_built_in_integers(self) -> None:
        for age in round_ages(np.asarray([25.5], dtype=np.float32)):
            self.assertIs(type(age), int)

    def test_negative_expectations_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            round_ages(np.asarray([-0.5], dtype=np.float32))


class ConfidenceTests(unittest.TestCase):
    def test_matches_the_frozen_scalar_contract(self) -> None:
        values = np.asarray(
            [0.0, 0.005, 0.29, 0.44, 0.83, 0.845, 0.85, 0.999999, 1.0], dtype=np.float32
        )
        self.assertEqual(
            confidence_percents(values),
            [confidence_percent(float(value)) for value in values],
        )

    def test_the_percent_multiply_stays_in_float32(self) -> None:
        """A float64 multiply floors this probability one percent lower."""
        probability = np.float32(0.8499999642372131)
        self.assertEqual(confidence_percents([probability]), [85])
        self.assertEqual(int(np.floor(float(probability) * 100.0)), 84)

    def test_boundaries_around_a_whole_percent(self) -> None:
        exact = np.float32(0.5)
        below = np.nextafter(exact, np.float32(0.0))
        self.assertEqual(confidence_percents([below, exact]), [49, 50])

    def test_returns_built_in_integers(self) -> None:
        for value in confidence_percents(np.asarray([1.0], dtype=np.float32)):
            self.assertIs(type(value), int)

    def test_values_outside_the_unit_interval_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            confidence_percents(np.asarray([1.5], dtype=np.float32))
        with self.assertRaises(ValueError):
            confidence_percents(np.asarray([np.nan], dtype=np.float32))


class GenderTests(unittest.TestCase):
    def test_matches_the_frozen_scalar_contract(self) -> None:
        rows = probability_rows(32, 2)
        predictions = gender_predictions(rows, GENDER_LABELS)
        for index, row in enumerate(rows):
            label, confidence = gender_result([float(value) for value in row])
            self.assertEqual(predictions[index], {"value": label, "confidence": confidence})

    def test_equal_probabilities_select_female(self) -> None:
        tied = np.full((1, 2), 0.5, dtype=np.float32)
        self.assertEqual(
            gender_predictions(tied, GENDER_LABELS),
            [{"value": "female", "confidence": 50}],
        )

    def test_a_single_step_above_female_selects_male(self) -> None:
        female = np.float32(0.5)
        male = np.nextafter(female, np.float32(1.0))
        rows = np.asarray([[female, male]], dtype=np.float32)
        self.assertEqual(gender_predictions(rows, GENDER_LABELS)[0]["value"], "male")

    def test_returns_built_in_types(self) -> None:
        prediction = gender_predictions(
            np.asarray([[0.25, 0.75]], dtype=np.float32), GENDER_LABELS
        )[0]
        self.assertIs(type(prediction["value"]), str)
        self.assertIs(type(prediction["confidence"]), int)

    def test_label_count_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            gender_predictions(np.zeros((1, 2), dtype=np.float32), ("female",))

    def test_probability_width_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            gender_predictions(np.zeros((1, 3), dtype=np.float32), GENDER_LABELS)


class ResultAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = bundled_models()

    def test_golden_faces_reproduce_the_frozen_results(self) -> None:
        faces = golden_faces()
        results = face_predictions(
            [face.rectangle for face in faces],
            np.stack([face.gender_probabilities for face in faces]),
            np.stack([face.age_probabilities for face in faces]),
            labels=self.bundle.gender.labels,
            age_weights=self.bundle.age.age_weights,
        )
        self.assertEqual(results, [face.result for face in faces])

    def test_key_order_matches_the_original_extension(self) -> None:
        faces = golden_faces()[:1]
        result = face_predictions(
            [faces[0].rectangle],
            faces[0].gender_probabilities[None, :],
            faces[0].age_probabilities[None, :],
            labels=GENDER_LABELS,
            age_weights=AGE_WEIGHTS,
        )[0]
        self.assertEqual(list(result), ["gender", "age", "face"])
        self.assertTrue(all(type(value) is int for value in result["face"]))

    def test_no_faces_produce_an_empty_list(self) -> None:
        self.assertEqual(
            face_predictions(
                [],
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 81), dtype=np.float32),
                labels=GENDER_LABELS,
                age_weights=AGE_WEIGHTS,
            ),
            [],
        )

    def test_mismatched_face_counts_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            face_predictions(
                [[0, 0, 1, 1]],
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 81), dtype=np.float32),
                labels=GENDER_LABELS,
                age_weights=AGE_WEIGHTS,
            )
        self.assertIn("same faces", str(caught.exception))

    def test_malformed_rectangles_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            face_predictions(
                [[0, 0, 1]],
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 81), dtype=np.float32),
                labels=GENDER_LABELS,
                age_weights=AGE_WEIGHTS,
            )


if __name__ == "__main__":
    unittest.main()
