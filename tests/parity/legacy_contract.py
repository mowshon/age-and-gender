"""Executable scalar and stage rules frozen from the C++ implementation."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass


def float32(value: float) -> float:
    """Round a Python number to IEEE-754 binary32."""
    return struct.unpack("=f", struct.pack("=f", value))[0]


def age_expectation(probabilities: Sequence[float]) -> float:
    if len(probabilities) != 81:
        raise ValueError(f"age probabilities must contain 81 values, got {len(probabilities)}")
    estimate = float32(float32(0.25) * float32(probabilities[0]))
    for index in range(1, 81):
        term = float32(float32(index) * float32(probabilities[index]))
        estimate = float32(estimate + term)
    return estimate


def round_age(expectation: float) -> int:
    """Match std::lround for the nonnegative age estimate domain."""
    if not math.isfinite(expectation) or expectation < 0:
        raise ValueError("age expectation must be finite and nonnegative")
    return math.floor(expectation + 0.5)


def confidence_percent(probability: float) -> int:
    value = float32(probability)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be finite and between zero and one")
    return math.floor(float32(value * float32(100.0)))


def gender_result(probabilities: Sequence[float]) -> tuple[str, int]:
    if len(probabilities) != 2:
        raise ValueError("gender probabilities must contain two values")
    female, male = (float32(value) for value in probabilities)
    selected = male if female < male else female
    return ("male" if female < male else "female", confidence_percent(selected))


def convert_boxes(boxes: Sequence[Sequence[int]]) -> list[list[int]] | None:
    """Convert legacy TRBL input to inclusive LTRB; empty means detection."""
    if not boxes:
        return None
    converted: list[list[int]] = []
    for index, box in enumerate(boxes):
        if len(box) != 4 or any(type(value) is not int for value in box):
            raise ValueError(f"box {index} must contain four integers")
        top, right, bottom, left = box
        if left > right or top > bottom:
            raise ValueError(f"box {index} has invalid inclusive geometry")
        converted.append([left, top, right, bottom])
    return converted


def normalize_rgb_hwc(rgb: bytes, height: int, width: int) -> list[float]:
    """Return legacy planar NCHW values for one tightly packed RGB image."""
    if len(rgb) != height * width * 3:
        raise ValueError("RGB byte count does not match height*width*3")
    means = (float32(122.781998), float32(117.000999), float32(104.297997))
    plane_size = height * width
    output = [0.0] * (3 * plane_size)
    for pixel in range(plane_size):
        for channel in range(3):
            difference = float32(float32(rgb[pixel * 3 + channel]) - means[channel])
            output[channel * plane_size + pixel] = float32(difference / 256.0)
    return output


@dataclass(frozen=True)
class StageMismatch(AssertionError):
    image: str
    face_index: int
    model: str
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    maximum_error: float
    first_mismatch: int | None

    def __str__(self) -> str:
        return (
            f"{self.image} face {self.face_index} {self.model}: "
            f"shape {self.expected_shape} != {self.actual_shape}; "
            f"max error {self.maximum_error}; first mismatch {self.first_mismatch}"
        )


def compare_stage(
    *,
    image: str,
    face_index: int,
    model: str,
    expected: Sequence[float],
    actual: Sequence[float],
    expected_shape: tuple[int, ...],
    actual_shape: tuple[int, ...],
    atol: float,
    rtol: float,
) -> None:
    first_mismatch: int | None = None
    maximum_error = 0.0
    expected_size = math.prod(expected_shape)
    actual_size = math.prod(actual_shape)
    if (
        expected_shape == actual_shape
        and len(expected) == len(actual)
        and len(expected) == expected_size
        and len(actual) == actual_size
    ):
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            error = abs(expected_value - actual_value)
            maximum_error = max(maximum_error, error)
            if (
                first_mismatch is None
                and (
                    not math.isfinite(expected_value)
                    or not math.isfinite(actual_value)
                    or error > atol + rtol * abs(expected_value)
                )
            ):
                first_mismatch = index
    else:
        first_mismatch = 0
    if first_mismatch is not None:
        raise StageMismatch(
            image,
            face_index,
            model,
            expected_shape,
            actual_shape,
            maximum_error,
            first_mismatch,
        )
