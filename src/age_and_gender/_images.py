"""RGB image and face-rectangle validation.

Nothing here imports :mod:`dlib`: this module only validates and normalizes the
caller's inputs into plain NumPy arrays and ``[left, top, right, bottom]``
integer rectangles. :mod:`age_and_gender._faces` is the only place those
rectangles are turned into backend objects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ._types import Rectangle

__all__ = ["as_face_array", "as_rgb_array", "parse_boxes"]

_RECTANGLE_LENGTH = 4
# A pre-cropped face is its own rectangle, so it must satisfy the same strict
# left < right and top < bottom rule as an explicit box.
_MIN_FACE_SIDE = 2
# dlib's rectangle stores each coordinate in a C++ long. This module does not
# import dlib, so it cannot query the backend's actual limit, and that limit is
# platform-dependent: a C long is 64-bit on Linux/macOS but only 32-bit on
# Windows, a platform this project supports. Bounding by signed-32-bit range
# keeps a box that passes this check constructible on every supported
# platform, generously covering any real image coordinate.
_MAX_COORDINATE_MAGNITUDE = 2**31 - 1


def as_rgb_array(photo: Image.Image | np.ndarray) -> NDArray[np.uint8]:
    """Validate an image and return it as an H x W x 3 uint8 RGB array.

    Args:
        photo: An RGB :class:`PIL.Image.Image`, or an ``[H, W, 3]`` uint8 NumPy
            array. Callers convert other Pillow modes explicitly, for example
            with ``image.convert("RGB")``; this function does not guess a
            color conversion.

    Returns:
        A C-contiguous ``[H, W, 3]`` uint8 array. An input array that is already
        a contiguous uint8 RGB array is returned unmodified, including a
        read-only one; it is never mutated. A Pillow image is always decoded
        into a new array.

    Raises:
        TypeError: `photo` is neither a Pillow image nor a NumPy array.
        ValueError: The image is not RGB, not uint8, not three-dimensional with
            three channels, or empty.
    """
    if isinstance(photo, Image.Image):
        if photo.mode != "RGB":
            raise ValueError(
                f"image mode must be RGB, got {photo.mode!r}; convert it first, "
                'for example with image.convert("RGB")'
            )
        array = np.asarray(photo, dtype=np.uint8)
    elif isinstance(photo, np.ndarray):
        if photo.dtype != np.uint8:
            raise ValueError(
                f"image array must be uint8, got dtype {photo.dtype}; scaling or "
                "casting the values is left to the caller"
            )
        if photo.ndim != 3 or photo.shape[2] != 3:
            raise ValueError(f"image array must have shape [H, W, 3], got {photo.shape}")
        # A copy only happens for a non-contiguous view; an already-contiguous
        # array, read-only or not, is returned as the same object.
        array = np.ascontiguousarray(photo)
    else:
        raise TypeError(f"photo must be a PIL.Image.Image or a NumPy array, got {type(photo)!r}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"image must be nonempty, got shape {array.shape}")
    return array


def as_face_array(face: Image.Image | np.ndarray) -> NDArray[np.uint8]:
    """Validate an already-cropped face and return it as an RGB array.

    Args:
        face: One face, as accepted by :func:`as_rgb_array`.

    Returns:
        The array :func:`as_rgb_array` returns.

    Raises:
        TypeError: `face` is neither a Pillow image nor a NumPy array.
        ValueError: `face` fails :func:`as_rgb_array` validation, or is
            narrower or shorter than two pixels, which as a rectangle would
            have degenerate geometry.
    """
    array = as_rgb_array(face)
    if array.shape[0] < _MIN_FACE_SIDE or array.shape[1] < _MIN_FACE_SIDE:
        raise ValueError(
            f"face must be at least {_MIN_FACE_SIDE} x {_MIN_FACE_SIDE} pixels, "
            f"got shape {array.shape}"
        )
    return array


def parse_boxes(face_bounding_boxes: Sequence[Sequence[int]] | None) -> list[Rectangle] | None:
    """Validate legacy boxes and convert them to inclusive rectangles.

    Args:
        face_bounding_boxes: `None`, or a sequence of four-integer
            ``(top, right, bottom, left)`` boxes, matching the original
            extension's input order. `None` or an empty sequence both mean
            "run detection instead of using explicit boxes".

    Returns:
        `None` when detection should run. Otherwise, one ``[left, top, right,
        bottom]`` rectangle per input box, in the same order, with duplicates
        preserved and out-of-frame coordinates kept as given.

    Raises:
        TypeError: `face_bounding_boxes`, or one of its boxes, is not a
            sequence, or a coordinate is not a Python or NumPy integer
            (booleans are rejected even though ``bool`` is an ``int`` subclass).
        ValueError: A box does not contain exactly four values, its geometry is
            degenerate (requires ``left < right`` and ``top < bottom``), or a
            coordinate is too large to be representable by the backend
            rectangle type on every supported platform.
    """
    if face_bounding_boxes is None:
        return None
    # len() avoids NumPy arrays raising on an ambiguous `if boxes:` truth test.
    try:
        box_count = len(face_bounding_boxes)
    except TypeError as error:
        raise TypeError(
            f"face_bounding_boxes must be a sequence of boxes, got {type(face_bounding_boxes)!r}"
        ) from error
    if box_count == 0:
        return None
    return [_parse_box(box, index) for index, box in enumerate(face_bounding_boxes)]


def _parse_box(box: Sequence[int], index: int) -> Rectangle:
    try:
        length = len(box)
    except TypeError as error:
        raise TypeError(f"box {index} must be a sequence of four integers") from error
    if length != _RECTANGLE_LENGTH:
        raise ValueError(f"box {index} must contain four values, got {length}")
    top, right, bottom, left = (_as_integer(value, index) for value in box)
    if left >= right or top >= bottom:
        raise ValueError(
            f"box {index} has degenerate geometry (top={top}, right={right}, "
            f"bottom={bottom}, left={left}); requires left < right and top < bottom"
        )
    return [left, top, right, bottom]


def _as_integer(value: object, index: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"box {index} coordinates must be integers, not bool")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"box {index} coordinates must be Python or NumPy integers, got {type(value)!r}"
        )
    coordinate = int(value)
    if abs(coordinate) > _MAX_COORDINATE_MAGNITUDE:
        raise ValueError(
            f"box {index} coordinate {coordinate} exceeds the backend's representable "
            f"range (+/-{_MAX_COORDINATE_MAGNITUDE})"
        )
    return coordinate
