"""Legacy-exact conversion from probability vectors to the public result.

Every rule here reproduces the original C++ arithmetic, including where it
rounds. The age expectation is accumulated sequentially in float32 because a
reduction that sums in a different order, or at a different precision, can move
the result across the half-year boundary that ``std::lround`` decides on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._types import AgePrediction, FacePrediction, GenderPrediction, Rectangle

__all__ = [
    "age_expectations",
    "age_predictions",
    "confidence_percents",
    "face_predictions",
    "gender_predictions",
    "round_ages",
]


def age_expectations(
    probabilities: np.ndarray, weights: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Return the float32 expected age of each row.

    Args:
        probabilities: ``[N, 81]`` float32 class probabilities.
        weights: The 81 class weights, starting with 0.25 for class zero.

    Returns:
        A ``[N]`` float32 array of expected ages, before rounding.

    Raises:
        ValueError: The probabilities do not match the weight count.
    """
    values = _as_probabilities(probabilities, len(weights), "age")
    class_weights = np.asarray(weights, dtype=np.float32)
    # The accumulator stays float32 and each class is added in class order, as
    # the C++ loop does. A dot product over the class axis is mathematically the
    # same sum but is free to reassociate it, which is not bit-compatible.
    estimate = np.multiply(class_weights[0], values[:, 0], dtype=np.float32)
    for index in range(1, class_weights.shape[0]):
        estimate += np.multiply(class_weights[index], values[:, index], dtype=np.float32)
    return estimate


def round_ages(expectations: np.ndarray) -> list[int]:
    """Round expected ages the way ``std::lround`` does for nonnegative values.

    The ``+ 0.5`` is evaluated in float64 so it does not introduce a second
    float32 rounding on top of the accumulated estimate.

    Args:
        expectations: ``[N]`` float32 expected ages.

    Returns:
        Built-in integers in the same order.

    Raises:
        ValueError: An expectation is not finite and nonnegative.
    """
    estimates = np.asarray(expectations, dtype=np.float32).astype(np.float64)
    if not np.isfinite(estimates).all() or (estimates < 0.0).any():
        raise ValueError("age expectations must be finite and nonnegative")
    return [int(value) for value in np.floor(estimates + 0.5)]


def confidence_percents(probabilities: np.ndarray | Sequence[float]) -> list[int]:
    """Convert probabilities to the legacy floored percentage confidences.

    Args:
        probabilities: Values in ``[0, 1]``.

    Returns:
        Built-in integers, each ``floor(float32(p * 100))``.

    Raises:
        ValueError: A value is not a finite probability.
    """
    values = np.asarray(probabilities, dtype=np.float32)
    if values.size and not (
        np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 1.0
    ):
        raise ValueError("confidences require finite probabilities between zero and one")
    # The multiply is float32, matching the C++ expression; the floor of a
    # float32 is exact, so only that multiply can move a value across a percent.
    return [int(value) for value in np.floor(values * np.float32(100.0))]


def age_predictions(
    probabilities: np.ndarray, weights: Sequence[float] | np.ndarray
) -> list[AgePrediction]:
    """Build the age half of the result for each row.

    The reported confidence is the largest class probability, not a probability
    of the rounded expected age.

    Args:
        probabilities: ``[N, 81]`` float32 class probabilities.
        weights: The 81 class weights.

    Returns:
        One mapping per row, in input order.
    """
    values = _as_probabilities(probabilities, len(weights), "age")
    ages = round_ages(age_expectations(values, weights))
    confidences = confidence_percents(values.max(axis=1))
    return [{"value": age, "confidence": confidence} for age, confidence in zip(ages, confidences)]


def gender_predictions(probabilities: np.ndarray, labels: Sequence[str]) -> list[GenderPrediction]:
    """Build the gender half of the result for each row.

    Args:
        probabilities: ``[N, 2]`` float32 probabilities ordered female, male.
        labels: The two labels, ordered to match the columns.

    Returns:
        One mapping per row, in input order. Exactly equal probabilities select
        the first label, as the original comparison does.

    Raises:
        ValueError: The probabilities or labels do not have two columns.
    """
    if len(labels) != 2:
        raise ValueError(f"gender needs exactly two labels, got {len(labels)}")
    values = _as_probabilities(probabilities, 2, "gender")
    chose_second = values[:, 0] < values[:, 1]
    selected = np.where(chose_second, values[:, 1], values[:, 0])
    confidences = confidence_percents(selected)
    return [
        {"value": str(labels[1] if second else labels[0]), "confidence": confidence}
        for second, confidence in zip(chose_second, confidences)
    ]


def face_predictions(
    rectangles: Sequence[Sequence[int]],
    gender_probabilities: np.ndarray,
    age_probabilities: np.ndarray,
    *,
    labels: Sequence[str],
    age_weights: Sequence[float] | np.ndarray,
) -> list[FacePrediction]:
    """Assemble the public result list.

    Args:
        rectangles: Inclusive ``[left, top, right, bottom]`` boxes in face order.
        gender_probabilities: ``[N, 2]`` float32 probabilities.
        age_probabilities: ``[N, 81]`` float32 probabilities.
        labels: Gender labels ordered to match the gender columns.
        age_weights: The 81 age class weights.

    Returns:
        One dictionary per face, with ``gender``, ``age``, and ``face`` keys in
        the order the original extension inserted them.

    Raises:
        ValueError: The rectangles and probability rows disagree in length.
    """
    genders = gender_predictions(gender_probabilities, labels)
    ages = age_predictions(age_probabilities, age_weights)
    if not len(rectangles) == len(genders) == len(ages):
        raise ValueError(
            f"{len(rectangles)} rectangles, {len(genders)} gender rows and "
            f"{len(ages)} age rows must describe the same faces"
        )
    return [
        {"gender": gender, "age": age, "face": _rectangle(rectangle)}
        for rectangle, gender, age in zip(rectangles, genders, ages)
    ]


def _rectangle(rectangle: Sequence[int]) -> Rectangle:
    if len(rectangle) != 4:
        raise ValueError(f"a face rectangle needs four values, got {len(rectangle)}")
    return [int(value) for value in rectangle]


def _as_probabilities(probabilities: np.ndarray, classes: int, task: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != classes:
        raise ValueError(f"{task} probabilities must have shape [N, {classes}], got {values.shape}")
    return values
