"""Result types and small shared aliases for the public contract."""

from __future__ import annotations

from typing import Literal, TypedDict

__all__ = [
    "AgePrediction",
    "FacePrediction",
    "GenderPrediction",
    "Rectangle",
    "Task",
]

Task = Literal["age", "gender"]
"""Neural task identifier, used as the manifest key and the model role."""

Rectangle = list[int]
"""Face rectangle as ``[left, top, right, bottom]`` with inclusive endpoints."""


class GenderPrediction(TypedDict):
    """Gender label and its floored percentage confidence."""

    value: str
    confidence: int


class AgePrediction(TypedDict):
    """Rounded expected age in years and its floored percentage confidence."""

    value: int
    confidence: int


class FacePrediction(TypedDict):
    """One face of a prediction result.

    Key order matches the original C++ extension: gender, age, then face.
    """

    gender: GenderPrediction
    age: AgePrediction
    face: Rectangle
