"""Age and gender estimation from face images.

Importing this package resolves no model resources and runs no inference: the
bundled models are read the first time a network is actually used.

::

    from age_and_gender import AgeAndGender
    from PIL import Image

    predictor = AgeAndGender()
    with Image.open("photo.jpg") as image:
        results = predictor.predict(image.convert("RGB"))
"""

from __future__ import annotations

from ._types import AgePrediction, FacePrediction, GenderPrediction, Rectangle
from .api import AgeAndGender

__all__ = [
    "AgeAndGender",
    "AgePrediction",
    "FacePrediction",
    "GenderPrediction",
    "Rectangle",
    "__version__",
]

__version__ = "2.0.0"
