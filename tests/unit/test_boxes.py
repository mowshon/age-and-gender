from __future__ import annotations

import unittest

import numpy as np

from age_and_gender._images import parse_boxes


class AutoDetectionSentinelTests(unittest.TestCase):
    def test_none_means_auto_detect(self) -> None:
        self.assertIsNone(parse_boxes(None))

    def test_empty_list_means_auto_detect(self) -> None:
        self.assertIsNone(parse_boxes([]))

    def test_empty_tuple_means_auto_detect(self) -> None:
        self.assertIsNone(parse_boxes(()))

    def test_empty_numpy_array_means_auto_detect(self) -> None:
        self.assertIsNone(parse_boxes(np.empty((0, 4), dtype=np.int64)))


class ConversionTests(unittest.TestCase):
    def test_single_box_converts_trbl_to_inclusive_ltrb(self) -> None:
        # (top, right, bottom, left) -> [left, top, right, bottom]
        self.assertEqual(parse_boxes([(29, 189, 101, 117)]), [[117, 29, 189, 101]])

    def test_order_and_duplicates_are_preserved(self) -> None:
        boxes = [(69, 525, 141, 453), (29, 189, 101, 117), (69, 525, 141, 453)]
        self.assertEqual(
            parse_boxes(boxes),
            [[453, 69, 525, 141], [117, 29, 189, 101], [453, 69, 525, 141]],
        )

    def test_out_of_frame_coordinates_are_kept_not_clipped(self) -> None:
        self.assertEqual(parse_boxes([(-20, 100, 100, -20)]), [[-20, -20, 100, 100]])

    def test_lists_and_tuples_both_work(self) -> None:
        self.assertEqual(parse_boxes([[29, 189, 101, 117]]), [[117, 29, 189, 101]])

    def test_numpy_integer_coordinates_are_accepted(self) -> None:
        boxes = np.array([[29, 189, 101, 117]], dtype=np.int32)
        self.assertEqual(parse_boxes(boxes), [[117, 29, 189, 101]])

    def test_python_and_numpy_integers_can_mix_within_one_box(self) -> None:
        box = (np.int64(29), 189, np.int16(101), 117)
        self.assertEqual(parse_boxes([box]), [[117, 29, 189, 101]])


class MalformedBoxTests(unittest.TestCase):
    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_boxes([(29, 189, 101)])
        with self.assertRaises(ValueError):
            parse_boxes([(29, 189, 101, 117, 0)])

    def test_non_sequence_box_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            parse_boxes([42])

    def test_non_sequence_boxes_argument_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            parse_boxes(42)  # type: ignore[arg-type]

    def test_float_coordinates_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            parse_boxes([(29.0, 189, 101, 117)])

    def test_bool_coordinates_are_rejected_despite_being_ints(self) -> None:
        with self.assertRaises(TypeError):
            parse_boxes([(True, 189, 101, 117)])

    def test_numpy_bool_coordinates_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            parse_boxes([(np.bool_(True), 189, 101, 117)])

    def test_string_coordinates_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            parse_boxes([("29", 189, 101, 117)])


class DegenerateGeometryTests(unittest.TestCase):
    def test_left_equal_to_right_is_rejected(self) -> None:
        # top=0, right=10, bottom=10, left=10 -> left == right
        with self.assertRaises(ValueError):
            parse_boxes([(0, 10, 10, 10)])

    def test_top_equal_to_bottom_is_rejected(self) -> None:
        # top=10, right=10, bottom=10, left=0 -> top == bottom
        with self.assertRaises(ValueError):
            parse_boxes([(10, 10, 10, 0)])

    def test_left_greater_than_right_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_boxes([(0, 5, 10, 20)])

    def test_top_greater_than_bottom_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_boxes([(20, 20, 5, 0)])


class BackendRangeTests(unittest.TestCase):
    """A box the pure-Python validator accepts must also be constructible by
    the backend rectangle type on every supported platform, including Windows
    where dlib's C long is 32-bit, not just on the platform running the test.
    """

    def test_coordinate_far_beyond_a_32_bit_long_is_rejected(self) -> None:
        huge = 10**30
        with self.assertRaises(ValueError):
            parse_boxes([(0, huge, huge, 0)])

    def test_negative_coordinate_far_beyond_a_32_bit_long_is_rejected(self) -> None:
        huge = -(10**30)
        with self.assertRaises(ValueError):
            parse_boxes([(huge, 100, 100, huge)])

    def test_coordinate_just_past_the_32_bit_long_boundary_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_boxes([(0, 2**31, 2**31, 0)])

    def test_coordinate_at_the_32_bit_long_boundary_is_accepted(self) -> None:
        limit = 2**31 - 1
        self.assertEqual(parse_boxes([(0, limit, limit, 0)]), [[0, 0, limit, limit]])


if __name__ == "__main__":
    unittest.main()
