"""Age and gender estimation from face images.

Importing this package resolves no model resources and runs no inference: the
bundled models are read the first time a network is actually used.

The public ``AgeAndGender`` class is added by the API work that builds on this
runtime; until then the package exposes its result types and version.
"""

from __future__ import annotations

from ._types import AgePrediction, FacePrediction, GenderPrediction, Rectangle

__all__ = [
    "AgePrediction",
    "FacePrediction",
    "GenderPrediction",
    "Rectangle",
    "__version__",
]

__version__ = "2.0.0a0"
