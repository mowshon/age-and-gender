"""Frontend equality of the dlib-bin frontend against the PR-1 oracle fixtures.

Every image, box, landmark, and chip here is compared for exact equality, not
tolerance: the legacy contract requires the detector, landmark predictor, and
individual chip extraction to reproduce the oracle bit for bit (spec/PR-1.md's
parity thresholds table). A tolerance would hide a real frontend drift.
"""

from __future__ import annotations

import hashlib
import struct
import unittest
from pathlib import Path
from unittest import mock

import dlib
import numpy as np
from PIL import Image

from age_and_gender._faces import AGE_CHIP_SIZE, CHIP_PADDING, GENDER_CHIP_SIZE, FaceFrontend
from age_and_gender._images import as_rgb_array, parse_boxes
from age_and_gender._models import bundled_models
from tests.golden import GoldenDocument, golden_documents

ROOT = Path(__file__).resolve().parents[2]


def image_available(document: GoldenDocument) -> bool:
    """Whether the document's source image exists in this checkout/sdist.

    ``dogs.golden.json`` sources from ``tools/vendor/dlib/examples/faces/dogs.jpg``,
    which is deliberately not part of the sdist (its manifest entry notes it
    "must not be redistributed separately"; see tools/legacy/README.md). Tests
    that need a document's pixels skip it, rather than fail, when its image is
    unavailable, so this module still runs from a built sdist that lacks
    ``libs/``.
    """
    return (ROOT / document.source_path).is_file()


def decode(document: GoldenDocument) -> np.ndarray:
    with Image.open(ROOT / document.source_path) as image:
        array = as_rgb_array(image.convert("RGB"))
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    if digest != document.rgb_sha256:
        raise AssertionError(
            f"{document.source_path}: decoded RGB {digest} != frozen {document.rgb_sha256}"
        )
    return array


class FrontendParityTests(unittest.TestCase):
    """Runs the installed dlib-bin frontend, not the historical C++ oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = bundled_models()
        cls.frontend = FaceFrontend(cls.bundle)
        cls.documents = golden_documents()

    def test_corpus_is_the_frozen_one(self) -> None:
        self.assertEqual(len(self.documents), 4)
        self.assertEqual(sum(len(document.faces) for document in self.documents), 11)

    def resolve_boxes(self, document: GoldenDocument) -> list[list[int]] | None:
        if document.box_mode == "detect":
            self.assertEqual(document.input_boxes_trbl, [])
            return None
        return parse_boxes(document.input_boxes_trbl)

    def test_exact_rectangles_landmarks_and_chips_on_every_document(self) -> None:
        for document in self.documents:
            with self.subTest(document=document.name):
                if not image_available(document):
                    self.skipTest(f"{document.source_path} is not available in this checkout")
                image = decode(document)
                boxes = self.resolve_boxes(document)
                extractions = self.frontend.extract(image, boxes)
                self.assertEqual(len(extractions), len(document.faces), "face count")
                for extraction, face in zip(extractions, document.faces):
                    label = face.label
                    self.assertEqual(extraction.rectangle, face.rectangle, f"{label}: rectangle")
                    self.assertEqual(extraction.landmarks, face.landmarks, f"{label}: landmarks")
                    self.assertTrue(
                        np.array_equal(extraction.gender_chip, face.gender_chip),
                        f"{label}: gender chip differs",
                    )
                    self.assertTrue(
                        np.array_equal(extraction.age_chip, face.age_chip),
                        f"{label}: age chip differs",
                    )
                    gender_shape = (GENDER_CHIP_SIZE, GENDER_CHIP_SIZE, 3)
                    age_shape = (AGE_CHIP_SIZE, AGE_CHIP_SIZE, 3)
                    self.assertEqual(extraction.gender_chip.shape, gender_shape)
                    self.assertEqual(extraction.age_chip.shape, age_shape)
                    self.assertEqual(extraction.gender_chip.dtype, np.uint8)
                    self.assertEqual(extraction.age_chip.dtype, np.uint8)

    def test_detection_is_zero_upsampled_and_keeps_detector_order(self) -> None:
        for document in self.documents:
            if document.box_mode != "detect":
                continue
            with self.subTest(document=document.name):
                if not image_available(document):
                    self.skipTest(f"{document.source_path} is not available in this checkout")
                image = decode(document)
                rectangles = self.frontend.detect(image)
                self.assertEqual(
                    rectangles,
                    [face.rectangle for face in document.faces],
                    "detector rectangles/order",
                )

    def test_explicit_boxes_bypass_detection_and_keep_order_and_duplicates(self) -> None:
        explicit = next(doc for doc in self.documents if doc.box_mode == "explicit")
        image = decode(explicit)
        boxes = parse_boxes(explicit.input_boxes_trbl)
        self.assertIsNotNone(boxes)
        self.assertEqual(len(boxes), len(explicit.faces))
        extractions = self.frontend.extract(image, boxes)
        self.assertEqual([e.rectangle for e in extractions], [f.rectangle for f in explicit.faces])
        # The fixture itself repeats one box and includes an out-of-frame one;
        # this asserts the corpus still exercises both, not just that we handled it.
        rectangles = [tuple(box) for box in boxes]
        self.assertGreater(len(rectangles), len(set(rectangles)), "fixture should repeat a box")
        out_of_frame = any(box[0] < 0 or box[1] < 0 for box in boxes)
        self.assertTrue(out_of_frame, "fixture should be out of frame")

    def test_explicit_boxes_never_invoke_the_detector(self) -> None:
        """The rectangle-equality test above proves correct output; this proves
        detection is actually skipped, rather than run and coincidentally
        agreeing, by replacing the detector with a spy that fails if called.
        """
        explicit = next(doc for doc in self.documents if doc.box_mode == "explicit")
        image = decode(explicit)
        boxes = parse_boxes(explicit.input_boxes_trbl)
        frontend = FaceFrontend(self.bundle)
        spy = mock.Mock(
            side_effect=AssertionError("the detector must not run when explicit boxes are given")
        )
        # Reaching into the private attribute is this test's only job: proving
        # non-invocation, which has no public observation point.
        frontend._detector = spy
        extractions = frontend.extract(image, boxes)
        self.assertEqual(spy.call_count, 0)
        self.assertEqual([e.rectangle for e in extractions], [f.rectangle for f in explicit.faces])

    def test_no_face_image_returns_empty(self) -> None:
        no_face = next(
            doc for doc in self.documents if doc.faces == [] and doc.box_mode == "detect"
        )
        if not image_available(no_face):
            self.skipTest(f"{no_face.source_path} is not available in this checkout")
        image = decode(no_face)
        self.assertEqual(self.frontend.detect(image), [])
        self.assertEqual(self.frontend.extract(image), [])


class PaddingPrecisionAndRouteEquivalenceTests(unittest.TestCase):
    """Retained evidence for two claims tools/legacy/frontend-comparison.md
    makes: the padding argument is honored at double, not float32, precision,
    and the get_face_chip convenience route agrees with the production route
    (get_face_chip_details + extract_image_chip) on the frozen corpus. Without
    a test, either claim could silently go stale on a future dlib-bin version.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = bundled_models()
        cls.documents = golden_documents()

    def test_padding_argument_is_honored_at_double_not_float32_precision(self) -> None:
        document = next(doc for doc in self.documents if doc.name == "test-image.golden.json")
        face = document.faces[0]
        shape = dlib.full_object_detection(
            dlib.rectangle(*face.rectangle),
            dlib.points([dlib.point(x, y) for x, y in face.landmarks]),
        )
        float32_padding = struct.unpack("=f", struct.pack("=f", CHIP_PADDING))[0]
        self.assertNotEqual(
            float32_padding, CHIP_PADDING, "fixture needs a padding value that truncates"
        )

        double_details = dlib.get_face_chip_details(shape, AGE_CHIP_SIZE, CHIP_PADDING)
        truncated_details = dlib.get_face_chip_details(shape, AGE_CHIP_SIZE, float32_padding)
        # If the binding silently narrowed the argument to float32, passing the
        # already-float32-truncated value would round-trip to the same double
        # and these would be indistinguishable; they are not.
        self.assertNotEqual(double_details.rect.left(), truncated_details.rect.left())

    def test_get_face_chip_convenience_route_matches_the_production_route(self) -> None:
        document = next(doc for doc in self.documents if doc.name == "test-image.golden.json")
        image = decode(document)
        for face in document.faces:
            shape = dlib.full_object_detection(
                dlib.rectangle(*face.rectangle),
                dlib.points([dlib.point(x, y) for x, y in face.landmarks]),
            )
            for size, expected_chip in (
                (GENDER_CHIP_SIZE, face.gender_chip),
                (AGE_CHIP_SIZE, face.age_chip),
            ):
                convenience_chip = np.ascontiguousarray(
                    dlib.get_face_chip(image, shape, size=size, padding=CHIP_PADDING)
                )
                self.assertTrue(
                    np.array_equal(convenience_chip, expected_chip),
                    f"{face.label} size {size}: get_face_chip disagrees with the golden chip",
                )


class BatchingTrapRegressionTests(unittest.TestCase):
    """Guards the individual-crop semantics spec/PR-4.md calls out by name.

    On tests/fixtures/legacy's first image, batching the 32x32 extraction with
    dlib's get_face_chips changes 1540, 1665, 1796, 0, and 1599 channel values
    (out of 3072) across the five faces, versus extracting each chip alone. This
    proves the production frontend keeps individual extraction rather than an
    unsafe "batch for speed" shortcut, and that the counterexample this repo
    documents is still reproducible on the pinned dlib-bin wheel.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = bundled_models()
        cls.frontend = FaceFrontend(cls.bundle)
        cls.documents = golden_documents()

    def test_batched_extraction_disagrees_with_individual_extraction(self) -> None:
        document = next(doc for doc in self.documents if doc.name == "test-image.golden.json")
        image = decode(document)

        # Reconstruct each face's shape from the production frontend's own public
        # output (rectangle + landmarks), so this test exercises the same shapes
        # the real extraction path computed rather than a hand-built stand-in.
        extractions = self.frontend.extract(image)
        shapes = [
            dlib.full_object_detection(
                dlib.rectangle(*extraction.rectangle),
                dlib.points([dlib.point(x, y) for x, y in extraction.landmarks]),
            )
            for extraction in extractions
        ]

        batched = dlib.get_face_chips(
            image, dlib.full_object_detections(shapes), size=GENDER_CHIP_SIZE, padding=CHIP_PADDING
        )
        differing_counts = [
            int((np.asarray(batch_chip) != extraction.gender_chip).sum())
            for batch_chip, extraction in zip(batched, extractions)
        ]
        self.assertEqual(differing_counts, [1540, 1665, 1796, 0, 1599])
        self.assertTrue(any(count > 0 for count in differing_counts))

        # The production path itself must keep matching the individually
        # extracted (golden) chips, not the batched ones.
        for extraction, face in zip(extractions, document.faces):
            self.assertTrue(np.array_equal(extraction.gender_chip, face.gender_chip))


if __name__ == "__main__":
    unittest.main()
