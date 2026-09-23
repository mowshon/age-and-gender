"""Bundled model resources, manifest validation, and legacy source mapping.

A *bundle* is a directory holding ``manifest.json``, the two converted ONNX
networks, the five-point landmark model, and their license notices. The package
installs one such bundle; :func:`load_bundle` accepts an equivalent directory
produced by the maintainer conversion tooling.

Nothing here is executed at import time: resolving the installed bundle reads
the manifest, and model bytes are read and hash-checked the first time a network
is actually loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final

from ._types import Task

__all__ = [
    "ModelBundle",
    "NeuralModelSpec",
    "Normalization",
    "RuntimeSpec",
    "ShapePredictorSpec",
    "bundled_models",
    "digest_bytes",
    "digest_file",
    "load_bundle",
]

MANIFEST_NAME: Final = "manifest.json"
SUPPORTED_SCHEMA_VERSION: Final = 1
SUPPORTED_BUNDLE_KIND: Final = "package"
TASKS: Final[tuple[Task, ...]] = ("age", "gender")

# The package owns 32x32 gender and 64x64 age chip extraction and the 81-class
# age reduction, so a bundle that declares anything else is not loadable by this
# release even if its graph is otherwise well formed.
CHIP_SIZES: Final[Mapping[Task, int]] = {"age": 64, "gender": 32}
CLASS_COUNTS: Final[Mapping[Task, int]] = {"age": 81, "gender": 2}
GENDER_LABELS: Final = ("female", "male")
NORMALIZATION_SCALE: Final = 1.0 / 256.0

# Parity was established with exactly these session options and does not hold at
# the other graph optimization levels; see tools/conversion/README.md for the
# measured comparison. A bundle that names different options was validated
# against a different contract, so it is rejected instead of silently run.
SUPPORTED_RUNTIME: Final[Mapping[str, Any]] = {
    "provider": "CPUExecutionProvider",
    "graph_optimization_level": "ORT_DISABLE_ALL",
    "intra_op_num_threads": 1,
    "inter_op_num_threads": 1,
}

_READ_BLOCK: Final = 1024 * 1024


def digest_bytes(payload: bytes) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of a file.

    Args:
        path: File to hash. Read in blocks, so large models do not have to be
            held in memory.

    Raises:
        FileNotFoundError: The path does not exist.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Normalization:
    """Input normalization contract recorded by the converter."""

    means: tuple[float, float, float]
    scale: float


@dataclass(frozen=True, slots=True)
class NeuralModelSpec:
    """Validated description of one converted network inside a bundle."""

    task: Task
    filename: str
    sha256: str
    size_bytes: int
    chip_size: int
    classes: int
    input_name: str
    output_name: str
    normalization: Normalization
    source_filename: str
    source_sha256: str
    labels: tuple[str, ...] | None = None
    age_weights: tuple[float, ...] | None = None

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Per-sample input shape as ``(channels, height, width)``."""
        return (3, self.chip_size, self.chip_size)


@dataclass(frozen=True, slots=True)
class ShapePredictorSpec:
    """Validated description of the bundled five-point landmark model."""

    filename: str
    sha256: str
    size_bytes: int
    parts: int
    source_filename: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """ONNX Runtime session options the bundle's parity evidence was gathered with."""

    provider: str
    graph_optimization_level: str
    intra_op_num_threads: int
    inter_op_num_threads: int


class ModelBundle:
    """A validated set of model resources resolved from one location.

    The manifest is parsed and checked when the bundle is created. Artifact
    bytes are read and hash-checked whenever they are handed out, and are not
    retained afterwards. Reads happen when a model is loaded, never per
    prediction, so a live session neither reopens nor rehashes its weights.
    """

    def __init__(self, root: Traversable, manifest: Mapping[str, Any], origin: str) -> None:
        self._root = root
        self._manifest = manifest
        self._origin = origin
        self._bundle_id = str(manifest["bundle_id"])
        self._runtime = _parse_runtime(manifest, origin)
        self._specs: dict[Task, NeuralModelSpec] = {
            task: _parse_neural_model(manifest, task, origin) for task in TASKS
        }
        self._shape_predictor = _parse_shape_predictor(manifest, origin)
        self._source_roles: dict[str, str] = {
            **{spec.source_sha256: task for task, spec in self._specs.items()},
            self._shape_predictor.source_sha256: "shape_predictor",
        }

    def __repr__(self) -> str:
        return f"ModelBundle(bundle_id={self._bundle_id!r}, origin={self._origin!r})"

    @property
    def bundle_id(self) -> str:
        """Identifier the converter stamped into the manifest."""
        return self._bundle_id

    @property
    def origin(self) -> str:
        """Human-readable location the bundle was resolved from."""
        return self._origin

    @property
    def manifest(self) -> Mapping[str, Any]:
        """The parsed manifest document."""
        return self._manifest

    @property
    def runtime(self) -> RuntimeSpec:
        """Session options every network in this bundle must be run with."""
        return self._runtime

    @property
    def age(self) -> NeuralModelSpec:
        """Specification of the bundled age network."""
        return self._specs["age"]

    @property
    def gender(self) -> NeuralModelSpec:
        """Specification of the bundled gender network."""
        return self._specs["gender"]

    @property
    def shape_predictor(self) -> ShapePredictorSpec:
        """Specification of the bundled five-point landmark model."""
        return self._shape_predictor

    def spec(self, task: Task) -> NeuralModelSpec:
        """Return the specification for ``task``.

        Raises:
            KeyError: ``task`` is not one of the supported neural tasks.
        """
        return self._specs[task]

    def model_bytes(self, task: Task) -> bytes:
        """Read and verify the serialized ONNX graph for ``task``.

        The bytes returned are always the ones that were just hashed, so a
        second session built from this bundle cannot pick up a file that changed
        after an earlier read.

        Args:
            task: ``"age"`` or ``"gender"``.

        Returns:
            The exact bytes recorded in the manifest.

        Raises:
            FileNotFoundError: The artifact is missing from the bundle.
            ValueError: The artifact's size or digest does not match the manifest.
        """
        spec = self._specs[task]
        return self._read_verified(spec.filename, spec.sha256, spec.size_bytes)

    @contextmanager
    def shape_predictor_file(self) -> Iterator[Path]:
        """Yield a filesystem path to the verified five-point landmark model.

        dlib deserializes from a path rather than from bytes, so the resource is
        materialized for the duration of the context. The path is only valid
        inside the ``with`` block: an installed package may have to extract it
        from a zip import, and the extracted copy is removed on exit. Each entry
        verifies the file it is about to yield, so a later call cannot hand out
        a path whose contents changed since an earlier one.

        Raises:
            FileNotFoundError: The artifact is missing from the bundle.
            ValueError: The artifact's size or digest does not match the manifest.
        """
        spec = self._shape_predictor
        resource = self._resource(spec.filename)
        with resources.as_file(resource) as path:
            size = path.stat().st_size
            if size != spec.size_bytes:
                raise ValueError(
                    f"{self._origin}: {spec.filename} is {size} bytes, "
                    f"manifest records {spec.size_bytes}"
                )
            digest = digest_file(path)
            if digest != spec.sha256:
                raise ValueError(
                    f"{self._origin}: {spec.filename} has sha256 {digest}, "
                    f"manifest records {spec.sha256}"
                )
            yield path

    @property
    def source_digests(self) -> Mapping[str, str]:
        """Map legacy ``.dat`` SHA-256 digests to the bundle role they satisfy.

        Roles are ``"age"``, ``"gender"``, and ``"shape_predictor"``. PR-5's
        loader methods use this to accept the original model files by content
        rather than by filename.
        """
        return dict(self._source_roles)

    def source_role(self, sha256: str) -> str | None:
        """Return the bundle role a legacy source digest maps to, or ``None``."""
        return self._source_roles.get(sha256.lower())

    def _resource(self, name: str) -> Traversable:
        return self._root.joinpath(name)

    def _read_verified(self, name: str, sha256: str, size_bytes: int) -> bytes:
        # Whatever is returned is hashed first. Remembering that a name was once
        # verified would let a later read of a changed file through unchecked,
        # and the read only happens when a model is loaded, so there is nothing
        # to gain by skipping it.
        payload = self._resource(name).read_bytes()
        if len(payload) != size_bytes:
            raise ValueError(
                f"{self._origin}: {name} is {len(payload)} bytes, manifest records {size_bytes}"
            )
        digest = digest_bytes(payload)
        if digest != sha256:
            raise ValueError(
                f"{self._origin}: {name} has sha256 {digest}, manifest records {sha256}"
            )
        return payload


def load_bundle(directory: str | os.PathLike[str]) -> ModelBundle:
    """Load and validate an explicit model bundle directory.

    Args:
        directory: Directory holding ``manifest.json`` and the artifacts it names.

    Returns:
        The validated bundle.

    Raises:
        FileNotFoundError: The directory or its manifest does not exist.
        ValueError: The manifest is malformed or describes an unsupported bundle.
    """
    root = Path(directory)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in model bundle {root}")
    return _build(root, str(root))


@lru_cache(maxsize=1)
def bundled_models() -> ModelBundle:
    """Return the bundle installed with the package.

    Resolution uses :mod:`importlib.resources`, so it works from a wheel
    installed anywhere and does not depend on the current working directory.

    Raises:
        FileNotFoundError: The installed package is missing its model resources.
        ValueError: The installed manifest is malformed.
    """
    root = resources.files(__package__).joinpath("models")
    return _build(root, f"{__package__}.models")


def _build(root: Traversable, origin: str) -> ModelBundle:
    try:
        document = root.joinpath(MANIFEST_NAME).read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{origin}: no {MANIFEST_NAME}") from error
    try:
        manifest = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError(f"{origin}: {MANIFEST_NAME} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{origin}: {MANIFEST_NAME} must contain a JSON object")
    _check_bundle_identity(manifest, origin)
    return ModelBundle(root, manifest, origin)


def _check_bundle_identity(manifest: Mapping[str, Any], origin: str) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{origin}: manifest schema_version {schema_version!r} is not supported; "
            f"this release reads version {SUPPORTED_SCHEMA_VERSION}"
        )
    kind = manifest.get("bundle_kind")
    if kind != SUPPORTED_BUNDLE_KIND:
        raise ValueError(
            f"{origin}: manifest bundle_kind {kind!r} is not a package bundle; "
            "build one with tools/conversion/build_bundle.py"
        )
    if not isinstance(manifest.get("bundle_id"), str) or not manifest["bundle_id"]:
        raise ValueError(f"{origin}: manifest is missing a bundle_id")


def _parse_runtime(manifest: Mapping[str, Any], origin: str) -> RuntimeSpec:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(
            f"{origin}: manifest has no runtime block; sessions are built from the "
            "options the bundle was validated with, never from runtime defaults"
        )
    if runtime != dict(SUPPORTED_RUNTIME):
        raise ValueError(
            f"{origin}: manifest runtime {runtime!r} differs from the validated "
            f"configuration {dict(SUPPORTED_RUNTIME)!r}"
        )
    return RuntimeSpec(
        provider=runtime["provider"],
        graph_optimization_level=runtime["graph_optimization_level"],
        intra_op_num_threads=runtime["intra_op_num_threads"],
        inter_op_num_threads=runtime["inter_op_num_threads"],
    )


def _section(manifest: Mapping[str, Any], origin: str, *path: str) -> Mapping[str, Any]:
    node: Any = manifest
    for index, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            location = ".".join(path[: index + 1])
            raise ValueError(f"{origin}: manifest is missing {location}")
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError(f"{origin}: manifest {'.'.join(path)} must be an object")
    return node


def _artifact(section: Mapping[str, Any], origin: str, label: str) -> tuple[str, str, int]:
    for key, expected in (("filename", str), ("sha256", str), ("bytes", int)):
        value = section.get(key)
        if not isinstance(value, expected) or isinstance(value, bool):
            raise ValueError(f"{origin}: manifest {label}.{key} is missing or malformed")
    return str(section["filename"]), str(section["sha256"]).lower(), int(section["bytes"])


def _parse_neural_model(manifest: Mapping[str, Any], task: Task, origin: str) -> NeuralModelSpec:
    model = _section(manifest, origin, "models", task)
    if model.get("task") != task:
        raise ValueError(f"{origin}: manifest models.{task} declares task {model.get('task')!r}")
    filename, sha256, size_bytes = _artifact(
        _section(manifest, origin, "models", task, "artifact"), origin, f"models.{task}.artifact"
    )
    source_filename, source_sha256, _ = _artifact(
        _section(manifest, origin, "models", task, "source"), origin, f"models.{task}.source"
    )
    chip_size = _check_graph_signature(manifest, task, origin)
    normalization = _parse_normalization(manifest, task, origin)
    labels = _parse_labels(model, task, origin)
    age_weights = _parse_age_weights(model, task, origin)
    return NeuralModelSpec(
        task=task,
        filename=filename,
        sha256=sha256,
        size_bytes=size_bytes,
        chip_size=chip_size,
        classes=CLASS_COUNTS[task],
        input_name=str(_section(manifest, origin, "models", task, "input")["name"]),
        output_name=str(_section(manifest, origin, "models", task, "output")["name"]),
        normalization=normalization,
        source_filename=source_filename,
        source_sha256=source_sha256,
        labels=labels,
        age_weights=age_weights,
    )


def _check_graph_signature(manifest: Mapping[str, Any], task: Task, origin: str) -> int:
    """Validate the declared tensor signature and return the chip size."""
    model_input = _section(manifest, origin, "models", task, "input")
    model_output = _section(manifest, origin, "models", task, "output")
    for section, label in ((model_input, "input"), (model_output, "output")):
        if section.get("dtype") != "float32":
            raise ValueError(f"{origin}: manifest models.{task}.{label}.dtype must be float32")
        if not isinstance(section.get("name"), str) or not section["name"]:
            raise ValueError(f"{origin}: manifest models.{task}.{label}.name is missing")
    if model_input.get("color_order") != "RGB":
        raise ValueError(f"{origin}: manifest models.{task}.input.color_order must be RGB")
    if model_output.get("softmax_in_graph") is not True:
        raise ValueError(
            f"{origin}: manifest models.{task}.output must apply softmax in the graph; "
            "the package consumes probabilities, not logits"
        )
    size = CHIP_SIZES[task]
    expected_input = ["N", 3, size, size]
    if list(model_input.get("shape", [])) != expected_input:
        raise ValueError(
            f"{origin}: manifest models.{task}.input.shape must be {expected_input}, "
            f"got {model_input.get('shape')!r}"
        )
    expected_output = ["N", CLASS_COUNTS[task]]
    if list(model_output.get("shape", [])) != expected_output:
        raise ValueError(
            f"{origin}: manifest models.{task}.output.shape must be {expected_output}, "
            f"got {model_output.get('shape')!r}"
        )
    return size


def _parse_normalization(manifest: Mapping[str, Any], task: Task, origin: str) -> Normalization:
    section = _section(manifest, origin, "models", task, "input", "normalization")
    means = section.get("means")
    if (
        not isinstance(means, Sequence)
        or isinstance(means, (str, bytes))
        or len(means) != 3
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in means
        )
    ):
        raise ValueError(
            f"{origin}: manifest models.{task}.input.normalization.means must be three numbers"
        )
    scale = section.get("scale")
    if scale != NORMALIZATION_SCALE:
        raise ValueError(
            f"{origin}: manifest models.{task} normalization scale {scale!r} is not "
            f"{NORMALIZATION_SCALE} (division by 256)"
        )
    return Normalization(
        means=(float(means[0]), float(means[1]), float(means[2])),
        scale=float(scale),
    )


def _parse_labels(model: Mapping[str, Any], task: Task, origin: str) -> tuple[str, ...] | None:
    if task != "gender":
        return None
    labels = model.get("labels")
    if list(labels or []) != list(GENDER_LABELS):
        raise ValueError(
            f"{origin}: manifest models.gender.labels must be {list(GENDER_LABELS)}, got {labels!r}"
        )
    return GENDER_LABELS


def _parse_age_weights(
    model: Mapping[str, Any], task: Task, origin: str
) -> tuple[float, ...] | None:
    if task != "age":
        return None
    weights = model.get("age_weights")
    if (
        not isinstance(weights, Sequence)
        or isinstance(weights, (str, bytes))
        or len(weights) != CLASS_COUNTS["age"]
    ):
        raise ValueError(
            f"{origin}: manifest models.age.age_weights must list "
            f"{CLASS_COUNTS['age']} class weights"
        )
    # Class zero represents a quarter of a year rather than zero, and every other
    # class weight is its own index. A bundle that disagrees would change every
    # age this package reports.
    expected = (0.25, *range(1, CLASS_COUNTS["age"]))
    if tuple(float(value) for value in weights) != tuple(float(value) for value in expected):
        raise ValueError(
            f"{origin}: manifest models.age.age_weights do not match the legacy "
            "class weights (0.25 followed by 1..80)"
        )
    return tuple(float(value) for value in weights)


def _parse_shape_predictor(manifest: Mapping[str, Any], origin: str) -> ShapePredictorSpec:
    section = _section(manifest, origin, "shape_predictor")
    if section.get("format") != "dlib-shape-predictor":
        raise ValueError(
            f"{origin}: manifest shape_predictor.format {section.get('format')!r} is not "
            "a dlib shape predictor"
        )
    if section.get("parts") != 5:
        raise ValueError(
            f"{origin}: manifest shape_predictor.parts must be 5; the alignment "
            "contract is the five-point model"
        )
    filename, sha256, size_bytes = _artifact(
        _section(manifest, origin, "shape_predictor", "artifact"),
        origin,
        "shape_predictor.artifact",
    )
    source_filename, source_sha256, _ = _artifact(
        _section(manifest, origin, "shape_predictor", "source"),
        origin,
        "shape_predictor.source",
    )
    return ShapePredictorSpec(
        filename=filename,
        sha256=sha256,
        size_bytes=size_bytes,
        parts=5,
        source_filename=source_filename,
        source_sha256=source_sha256,
    )
