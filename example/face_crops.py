"""Estimate age and gender for faces another detector has already cropped.

    python example/face_crops.py
    python example/face_crops.py crop-1.png crop-2.png

Each input image is one face, standing in for the crop an external detector
(OpenCV, MediaPipe, RetinaFace, ...) returns, for example
``frame[top:bottom, left:right]``. This package's own detector never runs on
these crops. JSON goes to stdout; status messages go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from age_and_gender import AgeAndGender

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Run the example."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "faces", nargs="*", type=Path, default=[HERE / "face-1.png", HERE / "face-2.png"]
    )
    args = parser.parse_args()

    predictor = AgeAndGender()
    results = {}
    for path in args.faces:
        # Detectors hand over pixel arrays, so load each crop as one. The
        # example crops are RGBA PNGs; the package only accepts RGB.
        with Image.open(path) as source:
            face = np.asarray(source.convert("RGB"))
        results[path.name] = {
            # Both estimates at once.
            "predict_face": predictor.predict_face(face),
            # Or just the one you need: each runs only its own model.
            "gender": predictor.gender(face),
            "age": predictor.age(face),
        }

    print(json.dumps(results, indent=2))
    print(f"{len(results)} face crop(s) processed without face detection", file=sys.stderr)


if __name__ == "__main__":
    main()
