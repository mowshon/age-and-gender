"""Use ``face_recognition``'s detector instead of this package's bundled one.

``face_recognition`` is optional: this package does not install, require, or
recommend it. Its declared ``dlib`` requirement is satisfied by the official
source-only ``dlib`` distribution, not the ``dlib-bin`` wheels this package
uses, so installing both in the same environment risks a source build (the
compiler requirement this package otherwise avoids) and a module conflict.
Install it in a *separate* virtual environment from this package's, or accept
that risk knowingly, before running this example:

    python -m pip install face_recognition numpy

``face_recognition.face_locations()`` returns ``(top, right, bottom, left)``
boxes, the same order and inclusive endpoints ``AgeAndGender.predict()``
expects for ``face_bounding_boxes``, so they are passed through unchanged.

Run from anywhere; paths default to the files next to this script:

    python example/example-with-face-recognition.py
    python example/example-with-face-recognition.py --image path/to/photo.jpg --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from age_and_gender import AgeAndGender

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=HERE / "test-image-2.jpg")
    parser.add_argument("--output", type=Path, default=HERE / "result-2.jpg")
    parser.add_argument("--show", action="store_true", help="Also open a preview window.")
    args = parser.parse_args()

    # Uses the models installed with the package; no explicit paths needed.
    # See example.py's comment for why the two neural loaders only accept
    # converted .onnx files.
    predictor = AgeAndGender()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
        face_bounding_boxes = face_recognition.face_locations(
            np.asarray(image),
            model="hog",  # "hog" for CPU, "cnn" for GPU (NVIDIA with CUDA)
        )
        results = predictor.predict(image, face_bounding_boxes)

        font = ImageFont.truetype(str(HERE / "Acme-Regular.ttf"), 15)
        draw = ImageDraw.Draw(image)
        for face in results:
            left, top, right, bottom = face["face"]
            gender = face["gender"]["value"].title()
            gender_confidence = face["gender"]["confidence"]
            age = face["age"]["value"]
            age_confidence = face["age"]["confidence"]

            draw.rectangle([(left, top), (right, bottom)], outline="red", width=5)
            draw.text(
                (left - 10, bottom + 10),
                f"{gender} (~{gender_confidence}%)\n{age} y.o. (~{age_confidence}%).",
                fill="red",
                font=font,
                align="center",
            )

        image.save(args.output)
        print(f"{len(results)} face(s) found; annotated image saved to {args.output}")
        if args.show:
            image.show()


if __name__ == "__main__":
    main()
