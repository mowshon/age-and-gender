"""Dlib-backed detection, landmark alignment, and per-face chip extraction.

This module is the only place that imports :mod:`dlib` (installed as the
``dlib-bin`` wheel) or constructs its objects. Callers outside it never see a
``dlib.rectangle`` or ``dlib.full_object_detection``: they pass and receive
plain ``[left, top, right, bottom]`` rectangles and uint8 chip arrays.

Both chips for a face are extracted independently, one
:func:`dlib.extract_image_chip` call each, from the original image. dlib's
batched ``get_face_chips`` shares a crop/pyramid across the whole batch and is
*not* pixel-equivalent to individual extraction; see spec/PR-4.md's "known
batching trap" for a measured counterexample on this repository's own example
image. Do not introduce it as a "safe" optimization here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import dlib
import numpy as np

from ._models import ModelBundle
from ._types import Rectangle

__all__ = [
    "AGE_CHIP_SIZE",
    "CHIP_PADDING",
    "GENDER_CHIP_SIZE",
    "FaceExtraction",
    "FaceFrontend",
]

# The legacy extension calls extract_image_chip(image, get_face_chip_details(shape,
# size), chip) with dlib's default padding, at these two sizes. Changing either
# value changes every downstream age/gender result.
CHIP_PADDING: Final = 0.2
GENDER_CHIP_SIZE: Final = 32
AGE_CHIP_SIZE: Final = 64
_REQUIRED_LANDMARK_PARTS: Final = 5
_DETECTOR_UPSAMPLE_NUM_TIMES: Final = 0

# A shape predictor's output part count depends only on how it was trained, not
# on the image or box content, so a throwaway probe is enough to validate it
# once at load time rather than on every face.
_PROBE_IMAGE: Final = np.zeros((16, 16, 3), dtype=np.uint8)
_PROBE_RECTANGLE: Final = dlib.rectangle(0, 0, 15, 15)


@dataclass(frozen=True, slots=True)
class FaceExtraction:
    """One face's rectangle, landmarks, and both independently aligned chips."""

    rectangle: Rectangle
    landmarks: list[list[int]]
    gender_chip: np.ndarray
    age_chip: np.ndarray


class FaceFrontend:
    """A reusable dlib HOG detector and five-point landmark predictor.

    Both are expensive to construct and stateless (the detector) or read-only
    (the predictor) once loaded, so one instance is built and reused across
    predictions rather than rebuilt per call.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        """Load the detector and the bundle's verified landmark model.

        Args:
            bundle: Validated bundle supplying the five-point landmark model.

        Raises:
            FileNotFoundError: The bundle is missing the landmark artifact.
            ValueError: The artifact fails its hash check, cannot be loaded as
                a shape predictor, or does not produce five landmark parts.
        """
        self._bundle = bundle
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = _load_predictor(bundle)

    def __repr__(self) -> str:
        return f"FaceFrontend(bundle={self._bundle.origin!r})"

    @property
    def bundle(self) -> ModelBundle:
        """The bundle this frontend's landmark model was resolved from."""
        return self._bundle

    def detect(self, image: np.ndarray) -> list[Rectangle]:
        """Detect faces with zero upsampling, in the detector's own order.

        Args:
            image: A C-contiguous ``[H, W, 3]`` uint8 RGB array, as returned by
                :func:`age_and_gender._images.as_rgb_array`.

        Returns:
            Inclusive ``[left, top, right, bottom]`` rectangles, one per
            detection, in detector order.
        """
        detections = self._detector(image, _DETECTOR_UPSAMPLE_NUM_TIMES)
        return [_rectangle_of(detection) for detection in detections]

    def extract(
        self, image: np.ndarray, boxes: Sequence[Rectangle] | None = None
    ) -> list[FaceExtraction]:
        """Resolve faces, then extract both chips for each one.

        Args:
            image: A C-contiguous ``[H, W, 3]`` uint8 RGB array, as returned by
                :func:`age_and_gender._images.as_rgb_array`.
            boxes: Already-validated ``[left, top, right, bottom]`` rectangles
                from :func:`age_and_gender._images.parse_boxes`, or `None`/empty
                to detect faces instead. Rectangles are used exactly as given,
                including ones that extend outside the image; they are not
                clipped or reordered.

        Returns:
            One :class:`FaceExtraction` per face, preserving detector/box
            order. A rectangle whose landmark model returns no parts is
            skipped, matching the original extension; the bundled five-point
            model always returns five, so this is a defensive no-op today.

        Raises:
            ValueError: A box's coordinates cannot be represented by the
                backend rectangle type.
        """
        pairs = self._resolve(image, boxes)
        extractions: list[FaceExtraction] = []
        for rectangle, detection in pairs:
            shape = self._predictor(image, detection)
            if shape.num_parts == 0:
                continue
            extractions.append(
                FaceExtraction(
                    rectangle=rectangle,
                    landmarks=_landmarks_of(shape),
                    gender_chip=self._chip(image, shape, GENDER_CHIP_SIZE),
                    age_chip=self._chip(image, shape, AGE_CHIP_SIZE),
                )
            )
        return extractions

    def _resolve(
        self, image: np.ndarray, boxes: Sequence[Rectangle] | None
    ) -> list[tuple[Rectangle, dlib.rectangle]]:
        if not boxes:
            detections = self._detector(image, _DETECTOR_UPSAMPLE_NUM_TIMES)
            return [(_rectangle_of(detection), detection) for detection in detections]
        resolved: list[tuple[Rectangle, dlib.rectangle]] = []
        for index, box in enumerate(boxes):
            left, top, right, bottom = (int(value) for value in box)
            try:
                detection = dlib.rectangle(left, top, right, bottom)
            except (TypeError, OverflowError) as error:
                raise ValueError(
                    f"box {index} {list(box)!r} is not representable by the backend "
                    f"rectangle type: {error}"
                ) from error
            resolved.append(([left, top, right, bottom], detection))
        return resolved

    def _chip(self, image: np.ndarray, shape: dlib.full_object_detection, size: int) -> np.ndarray:
        details = dlib.get_face_chip_details(shape, size, CHIP_PADDING)
        chip = dlib.extract_image_chip(image, details)
        return np.ascontiguousarray(chip)


def _rectangle_of(detection: dlib.rectangle) -> Rectangle:
    return [detection.left(), detection.top(), detection.right(), detection.bottom()]


def _landmarks_of(shape: dlib.full_object_detection) -> list[list[int]]:
    return [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]


def _load_predictor(bundle: ModelBundle) -> dlib.shape_predictor:
    spec = bundle.shape_predictor
    with bundle.shape_predictor_file() as path:
        try:
            predictor = dlib.shape_predictor(str(path))
        except Exception as error:
            raise ValueError(
                f"{bundle.origin}: could not load {spec.filename} as a shape predictor: {error}"
            ) from error
    parts = predictor(_PROBE_IMAGE, _PROBE_RECTANGLE).num_parts
    if parts != _REQUIRED_LANDMARK_PARTS:
        raise ValueError(
            f"{bundle.origin}: {spec.filename} produces {parts} landmark parts, "
            f"this release requires {_REQUIRED_LANDMARK_PARTS}"
        )
    return predictor
