"""The public ``AgeAndGender`` class.

The class ties image and face handling, ONNX inference, and compatible
postprocessing together behind the original calling convention. The default
constructor uses the models bundled with the package, while
``from_model_dir()`` accepts an explicit bundle.

Neural model loaders accept an ``.onnx`` graph with a sibling
``manifest.json`` naming it for the requested task. Neural dlib ``.dat`` files
are not supported. The landmark loader accepts a compatible dlib ``.dat``
file directly.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ._faces import FaceFrontend, load_predictor
from ._images import as_face_array, as_rgb_array, parse_boxes
from ._inference import InferenceEngine, NeuralNetwork
from ._models import ModelBundle, bundled_models, load_bundle
from ._postprocess import age_predictions, face_predictions, gender_predictions
from ._types import AgePrediction, FaceAttributes, FacePrediction, GenderPrediction, Task

if TYPE_CHECKING:
    # Only imported for type checking: api.py itself never touches dlib
    # objects, it just holds a predictor _faces.py validated and passes it
    # back in, keeping _faces.py the sole module that imports dlib at runtime.
    import dlib

__all__ = ["AgeAndGender"]

_ONNX_SUFFIX = ".onnx"


class AgeAndGender:
    """Detects faces and estimates each one's age and gender.

    ``AgeAndGender()`` works immediately, offline, using the models installed
    with this package. Model resources are resolved eagerly but their bytes
    are only read and loaded into a session or predictor the first time they
    are actually needed, by :meth:`predict` or by one of the ``load_*``
    methods below.

    Calls on one instance are thread-safe but serialized by an internal lock:
    concurrent callers block rather than race, so a prediction never observes
    a model mid-replacement. Independent instances share no state and can be
    used by independent worker threads or processes without that
    serialization, at the cost of each holding its own copy of the loaded
    models in memory.
    """

    def __init__(self) -> None:
        """Build a predictor backed by the models installed with this package.

        Raises:
            FileNotFoundError: The installed package is missing its model
                resources.
            ValueError: The installed manifest is malformed.
        """
        self._init_from_bundle(bundled_models())

    @classmethod
    def from_model_dir(cls, directory: str | os.PathLike[str]) -> Self:
        """Build a predictor from an explicit model bundle directory.

        Unlike the zero-configuration constructor, this loads and structurally
        validates every model immediately rather than on first use: an
        explicitly named bundle is meant to fail here, not deep inside a
        later :meth:`predict` call, if it turns out to be missing or
        incompatible.

        Args:
            directory: Directory holding ``manifest.json`` plus the age and
                gender ONNX graphs and the five-point landmark model it names.

        Returns:
            A predictor backed entirely by ``directory``'s models; the
            package's own bundled models are not used.

        Raises:
            FileNotFoundError: `directory` or its manifest does not exist, or
                one of the artifacts the manifest names is missing.
            ValueError: The manifest is malformed, is missing the age model,
                the gender model, or the shape predictor, or one of those
                artifacts fails to load as the declared kind of model.
        """
        bundle = load_bundle(directory)
        instance = cls.__new__(cls)
        instance._init_from_bundle(bundle, eager=True)
        return instance

    def _init_from_bundle(self, bundle: ModelBundle, *, eager: bool = False) -> None:
        self._lock = threading.Lock()
        self._bundle = bundle
        self._engine = InferenceEngine(bundle)
        self._frontend: FaceFrontend | None = None
        self._predictor_override: dlib.shape_predictor | None = None
        if eager:
            self._engine.age.ensure_loaded()
            self._engine.gender.ensure_loaded()
            self._frontend = FaceFrontend(bundle)

    def __repr__(self) -> str:
        """Return a debugging representation naming the backing bundle."""
        return f"AgeAndGender(bundle={self._bundle.origin!r})"

    # ------------------------------------------------------------------
    # Legacy-compatible loaders
    # ------------------------------------------------------------------

    def load_shape_predictor(self, shape_predictor_path: str | os.PathLike[str]) -> None:
        """Replace the five-point landmark model with an explicit file.

        Args:
            shape_predictor_path: Path to a dlib five-point shape-predictor
                ``.dat`` file, for example the original
                ``shape_predictor_5_face_landmarks.dat``. Loaded directly
                through dlib; any compatible file works.

        Raises:
            FileNotFoundError: The path does not exist.
            ValueError: The file cannot be loaded as a shape predictor, or
                does not produce five landmark parts.
        """
        predictor = load_predictor(shape_predictor_path)
        with self._lock:
            if self._frontend is not None:
                self._frontend.replace_predictor(predictor)
            else:
                self._predictor_override = predictor

    def load_dnn_age_predictor(self, dnn_age_predictor_path: str | os.PathLike[str]) -> None:
        """Replace the age model with an explicit ``.onnx`` file.

        Args:
            dnn_age_predictor_path: Path to an ``.onnx`` file with a sibling
                ``manifest.json`` naming it as the age model. The original
                dlib ``dnn_age_predictor_v1.dat`` is not accepted.

        Raises:
            FileNotFoundError: The path does not exist.
            ValueError: The path is not an ``.onnx`` file, or its sibling
                manifest does not identify it as the age model.
        """
        self._load_neural(dnn_age_predictor_path, "age")

    def load_dnn_gender_classifier(
        self, dnn_gender_classifier_path: str | os.PathLike[str]
    ) -> None:
        """Replace the gender model with an explicit ``.onnx`` file.

        Args:
            dnn_gender_classifier_path: Path to an ``.onnx`` file with a
                sibling ``manifest.json`` naming it as the gender model. The
                original dlib ``dnn_gender_classifier_v1.dat`` is not accepted.

        Raises:
            FileNotFoundError: The path does not exist.
            ValueError: The path is not an ``.onnx`` file, or its sibling
                manifest does not identify it as the gender model.
        """
        self._load_neural(dnn_gender_classifier_path, "gender")

    def _load_neural(self, path: str | os.PathLike[str], task: Task) -> None:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"no {task} model at {file_path}")
        if file_path.suffix.lower() != _ONNX_SUFFIX:
            raise ValueError(
                f"{file_path}: only .onnx files are accepted; the original dlib "
                ".dat weights cannot be loaded. Pass an .onnx model whose sibling "
                "manifest.json names it for this task."
            )
        bundle = _explicit_onnx_bundle(file_path, task)
        with self._lock:
            # NeuralNetwork.ensure_loaded() inside replace() validates the
            # candidate session before anything is swapped, so a failure here
            # leaves the previous model in place.
            self._engine.replace(task, bundle)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        photo_numpy_array: Image.Image | np.ndarray,
        face_bounding_boxes: Sequence[Sequence[int]] | None = None,
    ) -> list[FacePrediction]:
        """Detect faces and estimate each one's age and gender.

        Args:
            photo_numpy_array: An RGB :class:`PIL.Image.Image`, or an
                ``[H, W, 3]`` uint8 NumPy array. The parameter keeps its
                original name for keyword-argument compatibility.
            face_bounding_boxes: `None` or an empty sequence runs detection.
                Otherwise, a sequence of four-integer ``(top, right, bottom,
                left)`` boxes, the original extension's order and inclusive
                endpoints; supplied boxes are used exactly as given, including
                duplicates and out-of-frame coordinates, and detection does
                not run.

        Returns:
            One dictionary per face, in detector/box order:
            ``{"gender": {"value": str, "confidence": int}, "age": {"value":
            int, "confidence": int}, "face": [left, top, right, bottom]}``.
            An empty list when no faces are available.

        Raises:
            TypeError: `photo_numpy_array` or a box coordinate has the wrong
                type.
            ValueError: The image or a box fails validation.
        """
        image = as_rgb_array(photo_numpy_array)
        boxes = parse_boxes(face_bounding_boxes)
        with self._lock:
            frontend = self._ensure_frontend()
            extractions = frontend.extract(image, boxes)
            if not extractions:
                return []
            gender_network = self._engine.gender
            age_network = self._engine.age
            gender_probabilities = gender_network.probabilities(
                [extraction.gender_chip for extraction in extractions]
            )
            age_probabilities = age_network.probabilities(
                [extraction.age_chip for extraction in extractions]
            )
            gender_labels = gender_network.spec.labels
            age_weights = age_network.spec.age_weights
            assert gender_labels is not None  # a gender spec always carries labels
            assert age_weights is not None  # an age spec always carries class weights
            return face_predictions(
                [extraction.rectangle for extraction in extractions],
                gender_probabilities,
                age_probabilities,
                labels=gender_labels,
                age_weights=age_weights,
            )

    # ------------------------------------------------------------------
    # Already-cropped faces
    # ------------------------------------------------------------------

    def predict_face(self, face: Image.Image | np.ndarray) -> FaceAttributes:
        """Estimate age and gender for one face another detector already cropped.

        Detection does not run: the whole image is taken as the face
        rectangle, then landmarked and aligned exactly as :meth:`predict`
        aligns a detected face, so the result equals ``predict(face, [(0,
        width - 1, height - 1, 0)])[0]`` without its ``"face"`` key. Crop
        roughly where a face detector's box would sit; the landmark model
        tolerates a modest margin either way, but alignment, and with it the
        estimate, degrades on crops that are mostly background or cut
        features off.

        Args:
            face: One face as an RGB :class:`PIL.Image.Image` or an ``[H, W,
                3]`` uint8 NumPy array, at least 2 x 2 pixels.

        Returns:
            ``{"gender": {"value": str, "confidence": int}, "age": {"value":
            int, "confidence": int}}``.

        Raises:
            TypeError: `face` is neither a Pillow image nor a NumPy array.
            ValueError: `face` is not an RGB uint8 image of at least 2 x 2
                pixels.
        """
        image = as_face_array(face)
        with self._lock:
            gender_network = self._engine.gender
            age_network = self._engine.age
            gender_chip, age_chip = self._ensure_frontend().crop_chips(
                image, (gender_network.spec.chip_size, age_network.spec.chip_size)
            )
            return {
                "gender": _gender_of(gender_network, gender_chip),
                "age": _age_of(age_network, age_chip),
            }

    def gender(self, face: Image.Image | np.ndarray) -> GenderPrediction:
        """Estimate the gender of one face another detector already cropped.

        Extracts only the gender chip and runs only the gender model, which
        is loaded on first use; the age model is not loaded. The result equals
        ``predict_face(face)["gender"]``.

        Args:
            face: One face, as accepted by :meth:`predict_face`.

        Returns:
            ``{"value": str, "confidence": int}``.

        Raises:
            TypeError: `face` is neither a Pillow image nor a NumPy array.
            ValueError: `face` is not an RGB uint8 image of at least 2 x 2
                pixels.
        """
        image = as_face_array(face)
        with self._lock:
            network = self._engine.gender
            (chip,) = self._ensure_frontend().crop_chips(image, (network.spec.chip_size,))
            return _gender_of(network, chip)

    def age(self, face: Image.Image | np.ndarray) -> AgePrediction:
        """Estimate the age of one face another detector already cropped.

        Extracts only the age chip and runs only the age model, which is
        loaded on first use; the gender model is not loaded. The result equals
        ``predict_face(face)["age"]``.

        Args:
            face: One face, as accepted by :meth:`predict_face`.

        Returns:
            ``{"value": int, "confidence": int}``.

        Raises:
            TypeError: `face` is neither a Pillow image nor a NumPy array.
            ValueError: `face` is not an RGB uint8 image of at least 2 x 2
                pixels.
        """
        image = as_face_array(face)
        with self._lock:
            network = self._engine.age
            (chip,) = self._ensure_frontend().crop_chips(image, (network.spec.chip_size,))
            return _age_of(network, chip)

    def _ensure_frontend(self) -> FaceFrontend:
        if self._frontend is None:
            self._frontend = FaceFrontend(self._bundle, predictor=self._predictor_override)
        return self._frontend


def _gender_of(network: NeuralNetwork, chip: NDArray[np.uint8]) -> GenderPrediction:
    labels = network.spec.labels
    assert labels is not None  # a gender spec always carries labels
    return gender_predictions(network.probabilities([chip]), labels)[0]


def _age_of(network: NeuralNetwork, chip: NDArray[np.uint8]) -> AgePrediction:
    age_weights = network.spec.age_weights
    assert age_weights is not None  # an age spec always carries class weights
    return age_predictions(network.probabilities([chip]), age_weights)[0]


def _explicit_onnx_bundle(file_path: Path, task: Task) -> ModelBundle:
    bundle = load_bundle(file_path.parent)
    spec = bundle.spec(task)
    if spec.filename != file_path.name:
        raise ValueError(
            f"{file_path}: sibling {file_path.parent / 'manifest.json'} does not "
            f"name this file as the {task} model (it names {spec.filename!r})"
        )
    return bundle
