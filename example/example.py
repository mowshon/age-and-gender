"""Predict age and gender, print the results as JSON, and save an annotated image.

    python example/example.py
    python example/example.py photo.jpg --output annotated.jpg
    python example/example.py photo.jpg --models-dir path/to/bundle > result.json

JSON goes to stdout; status messages go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from age_and_gender import AgeAndGender

HERE = Path(__file__).resolve().parent

# Pillow looks bare font file names up in the system font directories.
SYSTEM_FONTS = ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the first installed system font, or Pillow's built-in one."""
    for name in SYSTEM_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def main() -> None:
    """Run the example."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", type=Path, default=HERE / "test-image.jpg")
    parser.add_argument("--output", type=Path, default=HERE / "result.jpg")
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="model bundle directory with manifest.json (default: the bundled models)",
    )
    args = parser.parse_args()

    # Without --models-dir, the models installed with the package are used.
    predictor = AgeAndGender.from_model_dir(args.models_dir) if args.models_dir else AgeAndGender()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    results = predictor.predict(image)

    draw = ImageDraw.Draw(image)
    font = load_font(max(16, image.width // 50))
    for face in results:
        left, top, right, bottom = face["face"]
        gender, age = face["gender"], face["age"]
        label = (
            f"{gender['value'].title()} ({gender['confidence']}%)\n"
            f"{age['value']} y.o. ({age['confidence']}%)"
        )
        draw.rectangle((left, top, right, bottom), outline="red", width=3)
        draw.text(
            (left, bottom + 6), label, font=font, fill="white", stroke_width=2, stroke_fill="black"
        )
    image.save(args.output)

    print(json.dumps(results, indent=2))
    print(f"{len(results)} face(s) found; annotated image saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
