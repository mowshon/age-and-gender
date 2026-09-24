"""Estimate age and gender for faces another detector has already cropped.

face-1.png and face-2.png stand in for the crops an external detector
returns, for example ``frame[top:bottom, left:right]``. This package's own
detector never runs on them.

    python example/face_crops.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from age_and_gender import AgeAndGender

HERE = Path(__file__).resolve().parent

predictor = AgeAndGender()

for name in ("face-1.png", "face-2.png"):
    # The example crops are RGBA PNGs; the package only accepts RGB.
    with Image.open(HERE / name) as source:
        face = np.asarray(source.convert("RGB"))

    print(name)
    # Both estimates at once.
    print("  predict_face:", predictor.predict_face(face))
    # Or just the one you need: each runs only its own model.
    print("  gender:      ", predictor.gender(face))
    print("  age:         ", predictor.age(face))
