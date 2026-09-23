"""Detect faces in an image and annotate each with its predicted age and gender.

Run from anywhere; paths default to the files next to this script:

    python example/example.py
    python example/example.py --image path/to/photo.jpg --output out.jpg --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from age_and_gender import AgeAndGender

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=HERE / "test-image.jpg")
    parser.add_argument("--output", type=Path, default=HERE / "result.jpg")
    parser.add_argument("--show", action="store_true", help="Also open a preview window.")
    args = parser.parse_args()

    # Uses the models installed with the package; no explicit paths needed.
    # The original dnn_age_predictor_v1.dat/dnn_gender_classifier_v1.dat
    # cannot be loaded directly any more (see spec/PR-5.md's "Design change"
    # section): they are a proprietary dlib format with no runtime ONNX
    # reader, so load_dnn_age_predictor/load_dnn_gender_classifier now only
    # accept a converted .onnx file (see tools/conversion/README.md). A custom
    # five-point landmark model can still be loaded directly, with
    # predictor.load_shape_predictor("shape_predictor_5_face_landmarks.dat").
    predictor = AgeAndGender()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
        results = predictor.predict(image)

        font = ImageFont.truetype(str(HERE / "Acme-Regular.ttf"), 20)
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
                fill="white",
                font=font,
                align="center",
            )

        image.save(args.output)
        print(f"{len(results)} face(s) found; annotated image saved to {args.output}")
        if args.show:
            image.show()


if __name__ == "__main__":
    main()
