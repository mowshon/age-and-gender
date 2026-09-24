"""Dlib-backed detection, landmark alignment, and per-face chip extraction.

This module is the only place that imports :mod:`dlib` (installed as the
``dlib-bin`` wheel) or constructs its objects. Callers outside it never see a
``dlib.rectangle`` or ``dlib.full_object_detection``: they pass and receive
plain ``[left, top, right, bottom]`` rectangles and uint8 chip arrays.

Both chips for a face are extracted independently from the original image.
Dlib's batched ``get_face_chips`` is not pixel-equivalent; the counterexample
is guarded by ``tests/parity/test_frontend.py::BatchingTrapRegressionTests``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import dlib
import numpy as np
from numpy.typing import NDArray

from ._models import ModelBundle
from ._types import Rectangle

__all__ = [
    "AGE_CHIP_SIZE",
    "CHIP_PADDING",
    "GENDER_CHIP_SIZE",
    "FaceExtraction",
    "FaceFrontend",
    "load_predictor",
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
    gender_chip: NDArray[np.uint8]
    age_chip: NDArray[np.uint8]


class FaceFrontend:
    """A reusable dlib HOG detector and five-point landmark predictor.

    Both are expensive to construct and stateless (the detector) or read-only
    (the predictor) once loaded, so one instance is built and reused across
    predictions rather than rebuilt per call.
    """

    def __init__(
        self, bundle: ModelBundle, *, predictor: dlib.shape_predictor | None = None
    ) -> None:
        """Load the detector and a validated five-point landmark predictor.

        Args:
            bundle: Validated bundle supplying the five-point landmark model,
                unless `predictor` is given explicitly.
            predictor: An already-validated five-point predictor to install
                instead of the bundle's own, for example one produced by
                :func:`load_predictor`. Skips reading the bundle's landmark
                artifact entirely, so a caller that is about to replace it
                does not pay for a load that is immediately discarded.

        Raises:
            FileNotFoundError: `predictor` is not given and the bundle is
                missing the landmark artifact.
            ValueError: `predictor` is not given and the bundle's landmark
                artifact cannot be loaded as a shape predictor, or does not
                produce five landmark parts.
        """
        self._bundle = bundle
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = predictor if predictor is not None else _load_predictor(bundle)

    def __repr__(self) -> str:
        return f"FaceFrontend(bundle={self._bundle.origin!r})"

    @property
    def bundle(self) -> ModelBundle:
        """The bundle this frontend's landmark model was resolved from."""
        return self._bundle

    def replace_predictor(self, predictor: dlib.shape_predictor) -> None:
        """Install an already-validated five-point predictor.

        The detector is unaffected: it is stateless and not tied to any
        particular landmark model.

        Args:
            predictor: A predictor that has already been confirmed to produce
                five landmark parts, for example via :func:`load_predictor`.
        """
        self._predictor = predictor

    def detect(self, image: NDArray[np.uint8]) -> list[Rectangle]:
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
        self, image: NDArray[np.uint8], boxes: Sequence[Rectangle] | None = None
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
        self, image: NDArray[np.uint8], boxes: Sequence[Rectangle] | None
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

    def _chip(
        self, image: NDArray[np.uint8], shape: dlib.full_object_detection, size: int
    ) -> NDArray[np.uint8]:
        details = dlib.get_face_chip_details(shape, size, CHIP_PADDING)
        chip = dlib.extract_image_chip(image, details)
        return np.ascontiguousarray(chip)


def _rectangle_of(detection: dlib.rectangle) -> Rectangle:
    return [detection.left(), detection.top(), detection.right(), detection.bottom()]


def _landmarks_of(shape: dlib.full_object_detection) -> list[list[int]]:
    return [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]


def _probe_parts(predictor: dlib.shape_predictor) -> int:
    return int(predictor(_PROBE_IMAGE, _PROBE_RECTANGLE).num_parts)


def _load_predictor(bundle: ModelBundle) -> dlib.shape_predictor:
    spec = bundle.shape_predictor
    with bundle.shape_predictor_file() as path:
        try:
            predictor = dlib.shape_predictor(str(path))
        except Exception as error:
            raise ValueError(
                f"{bundle.origin}: could not load {spec.filename} as a shape predictor: {error}"
            ) from error
    parts = _probe_parts(predictor)
    if parts != _REQUIRED_LANDMARK_PARTS:
        raise ValueError(
            f"{bundle.origin}: {spec.filename} produces {parts} landmark parts, "
            f"this release requires {_REQUIRED_LANDMARK_PARTS}"
        )
    return predictor


def load_predictor(path: str | os.PathLike[str]) -> dlib.shape_predictor:
    """Load and validate an explicit five-point landmark model file.

    Unlike the bundle-backed loading :class:`FaceFrontend` does internally,
    this reads `path` directly: dlib deserializes a shape predictor natively,
    with no ONNX conversion step, so any compatible file works, not only the
    bundled one. This is what the public API's ``load_shape_predictor``
    compatibility method uses.

    Args:
        path: Path to a dlib five-point shape-predictor ``.dat`` file.

    Returns:
        A predictor confirmed to produce five landmark parts.

    Raises:
        FileNotFoundError: `path` does not exist.
        ValueError: The file cannot be loaded as a shape predictor, or
            produces a different number of landmark parts.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"no shape predictor at {file_path}")
    try:
        predictor = dlib.shape_predictor(str(file_path))
    except Exception as error:
        raise ValueError(f"{file_path}: could not load as a shape predictor: {error}") from error
    parts = _probe_parts(predictor)
    if parts != _REQUIRED_LANDMARK_PARTS:
        raise ValueError(
            f"{file_path}: produces {parts} landmark parts, this release requires "
            f"{_REQUIRED_LANDMARK_PARTS}"
        )
    return predictor
