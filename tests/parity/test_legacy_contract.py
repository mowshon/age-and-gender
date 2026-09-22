from __future__ import annotations

import math
import unittest

from tests.parity.legacy_contract import (
    StageMismatch,
    age_expectation,
    compare_stage,
    confidence_percent,
    convert_boxes,
    float32,
    gender_result,
    normalize_rgb_hwc,
    round_age,
)


class PostprocessingTests(unittest.TestCase):
    def test_age_rounding_boundaries_are_half_away_from_zero(self) -> None:
        self.assertEqual(round_age(math.nextafter(20.5, 0.0)), 20)
        self.assertEqual(round_age(20.5), 21)
        self.assertEqual(round_age(math.nextafter(20.5, math.inf)), 21)

    def test_gender_tie_selects_female(self) -> None:
        self.assertEqual(gender_result([0.5, 0.5]), ("female", 50))
        self.assertEqual(gender_result([0.49, 0.51]), ("male", 51))

    def test_confidence_uses_float32_then_floor(self) -> None:
        boundary = float32(0.3)
        self.assertEqual(confidence_percent(float32(boundary - 2**-24)), 29)
        self.assertEqual(confidence_percent(boundary), 30)
        self.assertEqual(confidence_percent(float32(boundary + 2**-24)), 30)

    def test_age_class_zero_has_quarter_year_weight(self) -> None:
        probabilities = [0.0] * 81
        probabilities[0] = 1.0
        self.assertEqual(age_expectation(probabilities), 0.25)

    def test_age_is_expectation_not_argmax(self) -> None:
        probabilities = [0.0] * 81
        probabilities[10] = 0.51
        probabilities[80] = 0.49
        self.assertEqual(round_age(age_expectation(probabilities)), 44)

    def test_age_reduction_is_sequential_float32(self) -> None:
        probabilities = [float32(1.0 / 81.0)] * 81
        sequential = age_expectation(probabilities)
        double_precision = 0.25 * probabilities[0] + sum(
            index * probabilities[index] for index in range(1, 81)
        )
        self.assertNotEqual(sequential, double_precision)


class InputContractTests(unittest.TestCase):
    def test_empty_boxes_mean_detection(self) -> None:
        self.assertIsNone(convert_boxes([]))

    def test_boxes_are_inclusive_and_ordered_without_clipping(self) -> None:
        boxes = [(2, 9, 8, -3), (2, 9, 8, -3), (0, 4, 3, 1)]
        self.assertEqual(
            convert_boxes(boxes),
            [[-3, 2, 9, 8], [-3, 2, 9, 8], [1, 0, 4, 3]],
        )

    def test_normalization_is_rgb_nchw_float32(self) -> None:
        actual = normalize_rgb_hwc(bytes([0, 1, 2, 255, 254, 253]), 1, 2)
        expected = [
            float32((float32(0) - float32(122.781998)) / 256.0),
            float32((float32(255) - float32(122.781998)) / 256.0),
            float32((float32(1) - float32(117.000999)) / 256.0),
            float32((float32(254) - float32(117.000999)) / 256.0),
            float32((float32(2) - float32(104.297997)) / 256.0),
            float32((float32(253) - float32(104.297997)) / 256.0),
        ]
        self.assertEqual(actual, expected)


class StageComparisonTests(unittest.TestCase):
    def test_declared_shape_must_match_value_count(self) -> None:
        with self.assertRaises(StageMismatch):
            compare_stage(
                image="fixture.rgb",
                face_index=0,
                model="logits",
                expected=[1.0],
                actual=[1.0],
                expected_shape=(1, 2),
                actual_shape=(1, 2),
                atol=0,
                rtol=0,
            )

    def test_mismatch_identifies_stage_and_first_value(self) -> None:
        with self.assertRaises(StageMismatch) as caught:
            compare_stage(
                image="fixture.rgb",
                face_index=2,
                model="age probabilities",
                expected=[0.2, 0.8],
                actual=[0.2, 0.7],
                expected_shape=(1, 2),
                actual_shape=(1, 2),
                atol=1e-6,
                rtol=1e-4,
            )
        self.assertEqual(caught.exception.first_mismatch, 1)
        self.assertIn("fixture.rgb face 2 age probabilities", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
