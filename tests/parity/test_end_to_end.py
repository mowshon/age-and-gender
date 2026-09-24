"""End-to-end ``AgeAndGender.predict()`` parity against the frozen oracle.

Unlike tests/parity/test_frontend.py (frontend only) and
tests/parity/test_chip_inference.py (pre-extracted chips only), this drives
the full public pipeline: a raw decoded image in, the compatible result
dictionaries out, compared for exact equality.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from PIL import Image

from age_and_gender import AgeAndGender
from age_and_gender._images import as_rgb_array
from tests.golden import GoldenDocument, golden_documents

ROOT = Path(__file__).resolve().parents[2]


def decode(document: GoldenDocument) -> Image.Image:
    return Image.open(ROOT / document.source_path)


class EndToEndParityTests(unittest.TestCase):
    """Runs the public ``AgeAndGender`` class, not the internal modules directly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = AgeAndGender()
        cls.documents = golden_documents()

    def test_corpus_is_the_frozen_one(self) -> None:
        self.assertEqual(len(self.documents), 3)
        self.assertEqual(sum(len(document.faces) for document in self.documents), 11)

    def resolve_boxes(self, document: GoldenDocument) -> list[list[int]] | None:
        if document.box_mode == "detect":
            self.assertEqual(document.input_boxes_trbl, [])
            return None
        return document.input_boxes_trbl

    def test_exact_results_on_every_document(self) -> None:
        for document in self.documents:
            with self.subTest(document=document.name):
                with decode(document) as image:
                    array = as_rgb_array(image.convert("RGB"))
                boxes = self.resolve_boxes(document)
                results = self.predictor.predict(array, boxes)
                expected = [face.result for face in document.faces]
                self.assertEqual(results, expected)

    def test_pillow_image_input_matches_array_input(self) -> None:
        document = next(doc for doc in self.documents if doc.name == "test-image.golden.json")
        with decode(document) as image:
            from_pillow = self.predictor.predict(image.convert("RGB"))
        with decode(document) as image:
            from_array = self.predictor.predict(as_rgb_array(image.convert("RGB")))
        self.assertEqual(from_pillow, from_array)
        self.assertEqual(from_pillow, [face.result for face in document.faces])

    def test_positional_and_keyword_calls_agree(self) -> None:
        explicit = next(doc for doc in self.documents if doc.box_mode == "explicit")
        with decode(explicit) as image:
            array = as_rgb_array(image.convert("RGB"))
        positional = self.predictor.predict(array, explicit.input_boxes_trbl)
        keyword = self.predictor.predict(
            photo_numpy_array=array, face_bounding_boxes=explicit.input_boxes_trbl
        )
        self.assertEqual(positional, keyword)
        self.assertEqual(positional, [face.result for face in explicit.faces])

    def test_empty_box_list_behaves_like_omitted_boxes(self) -> None:
        document = next(doc for doc in self.documents if doc.box_mode == "detect")
        with decode(document) as image:
            array = as_rgb_array(image.convert("RGB"))
        self.assertEqual(self.predictor.predict(array, []), self.predictor.predict(array, None))
        self.assertEqual(self.predictor.predict(array), self.predictor.predict(array, None))

    def test_results_are_json_serializable_without_a_custom_encoder(self) -> None:
        document = next(doc for doc in self.documents if doc.name == "test-image.golden.json")
        with decode(document) as image:
            results = self.predictor.predict(image.convert("RGB"))
        serialized = json.dumps(results)
        self.assertEqual(json.loads(serialized), results)
        for face in results:
            self.assertIsInstance(face["age"]["value"], int)
            self.assertIsInstance(face["age"]["confidence"], int)
            self.assertIsInstance(face["gender"]["confidence"], int)
            self.assertIsInstance(face["gender"]["value"], str)


if __name__ == "__main__":
    unittest.main()
