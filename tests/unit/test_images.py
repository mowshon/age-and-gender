from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from age_and_gender._images import as_rgb_array


def rgb_image(width: int = 6, height: int = 4) -> Image.Image:
    array = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    return Image.fromarray(array, mode="RGB")


def rgb_array(width: int = 6, height: int = 4) -> np.ndarray:
    return np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)


class PillowInputTests(unittest.TestCase):
    def test_rgb_image_becomes_hwc_uint8_array(self) -> None:
        image = rgb_image(width=5, height=3)
        array = as_rgb_array(image)
        self.assertEqual(array.dtype, np.uint8)
        self.assertEqual(array.shape, (3, 5, 3))
        self.assertTrue(np.array_equal(array, np.asarray(image)))

    def test_grayscale_mode_is_rejected(self) -> None:
        image = rgb_image().convert("L")
        with self.assertRaises(ValueError):
            as_rgb_array(image)

    def test_rgba_mode_is_rejected_rather_than_dropping_alpha(self) -> None:
        image = rgb_image().convert("RGBA")
        with self.assertRaises(ValueError):
            as_rgb_array(image)

    def test_palette_mode_is_rejected(self) -> None:
        image = rgb_image().convert("P")
        with self.assertRaises(ValueError):
            as_rgb_array(image)


class NumpyInputTests(unittest.TestCase):
    def test_contiguous_uint8_array_round_trips(self) -> None:
        source = rgb_array()
        array = as_rgb_array(source)
        self.assertTrue(np.array_equal(array, source))
        self.assertEqual(array.dtype, np.uint8)

    def test_contiguous_array_is_not_copied(self) -> None:
        source = rgb_array()
        array = as_rgb_array(source)
        self.assertIs(array, source)

    def test_non_contiguous_view_is_normalized_without_mutating_the_source(self) -> None:
        source = rgb_array(width=8)
        view = source[:, ::-1]
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        before = source.copy()
        array = as_rgb_array(view)
        self.assertTrue(array.flags["C_CONTIGUOUS"])
        self.assertTrue(np.array_equal(array, view))
        self.assertTrue(np.array_equal(source, before))

    def test_read_only_array_is_accepted_and_left_read_only(self) -> None:
        source = rgb_array()
        source.setflags(write=False)
        array = as_rgb_array(source)
        self.assertTrue(np.array_equal(array, source))

    def test_float_dtype_is_rejected_rather_than_scaled(self) -> None:
        source = (rgb_array().astype(np.float32)) / 255.0
        with self.assertRaises(ValueError):
            as_rgb_array(source)

    def test_int32_dtype_is_rejected(self) -> None:
        source = rgb_array().astype(np.int32)
        with self.assertRaises(ValueError):
            as_rgb_array(source)

    def test_grayscale_shape_is_rejected(self) -> None:
        source = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            as_rgb_array(source)

    def test_four_channel_shape_is_rejected_rather_than_dropping_alpha(self) -> None:
        source = np.zeros((4, 4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            as_rgb_array(source)

    def test_zero_width_is_rejected(self) -> None:
        source = np.zeros((4, 0, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            as_rgb_array(source)

    def test_zero_height_is_rejected(self) -> None:
        source = np.zeros((0, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            as_rgb_array(source)


class UnsupportedInputTests(unittest.TestCase):
    def test_a_plain_list_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            as_rgb_array([[0, 0, 0]])

    def test_a_string_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            as_rgb_array("photo.jpg")


if __name__ == "__main__":
    unittest.main()
