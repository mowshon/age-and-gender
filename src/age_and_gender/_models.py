"""Bundled model resources and manifest validation.

A *bundle* is a directory holding ``manifest.json``, the two converted ONNX
networks, and the five-point landmark model. The package installs one such
bundle; :func:`load_bundle` accepts any directory with a manifest in the same
schema, whether produced by the maintainer conversion tooling or hand-authored
for a custom model.

Nothing here is executed at import time: resolving the installed bundle reads
the manifest, and model bytes are read the first time a network is actually
loaded.

Validation here is structural, not cryptographic: the manifest must describe a
compatible task, tensor shape/dtype, normalization contract, and labels, and
:mod:`age_and_gender._inference` separately checks the loaded ONNX graph's own
declared signature against that same manifest. There is no SHA-256/byte-size
check of artifact contents against the manifest — a bundle only has to be
structurally compatible, not byte-identical to some known-good copy, so a
caller can point this at their own model files without regenerating a
hash-locked manifest for them. A manifest may still record a ``source`` block
naming what a model was converted from, purely as an informational note; it is
never required and never used to gate loading.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._types import Task

__all__ = [
    "ModelBundle",
    "NeuralModelSpec",
    "Normalization",
    "RuntimeSpec",
    "ShapePredictorSpec",
    "bundled_models",
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


@dataclass(frozen=True, slots=True)
class Normalization:
    """Input normalization contract recorded by the converter."""

    means: tuple[float, float, float]
    scale: float


@dataclass(frozen=True, slots=True)
class NeuralModelSpec:
    """Validated description of one converted network inside a bundle.

    ``sha256``/``size_bytes`` and ``source_filename``/``source_sha256`` are
    informational only, carried over from the manifest when present (the
    package's own shipped manifest records them for provenance); none of them
    are checked against the artifact's actual bytes at load time.
    """

    task: Task
    filename: str
    chip_size: int
    classes: int
    input_name: str
    output_name: str
    normalization: Normalization
    sha256: str | None = None
    size_bytes: int | None = None
    source_filename: str | None = None
    source_sha256: str | None = None
    labels: tuple[str, ...] | None = None
    age_weights: tuple[float, ...] | None = None

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Per-sample input shape as ``(channels, height, width)``."""
        return (3, self.chip_size, self.chip_size)


@dataclass(frozen=True, slots=True)
class ShapePredictorSpec:
    """Validated description of the bundled five-point landmark model.

    ``sha256``/``size_bytes`` and ``source_filename``/``source_sha256`` are
    informational only; see :class:`NeuralModelSpec`.
    """

    filename: str
    parts: int
    sha256: str | None = None
    size_bytes: int | None = None
    source_filename: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """ONNX Runtime session options the bundle's parity evidence was gathered with."""

    provider: str
    graph_optimization_level: str
    intra_op_num_threads: int
    inter_op_num_threads: int


class ModelBundle:
    """A structurally validated set of model resources resolved from one location.

    The manifest is parsed and structurally checked when the bundle is
    created: task, tensor shape/dtype, normalization, and labels. Artifact
    bytes are read fresh from disk whenever they are handed out, never cached
    or retained afterwards, but not hash-verified against the manifest — a
    bundle only has to be structurally compatible, checked by the ONNX Runtime
    session or dlib predictor that actually loads it. Reads happen when a
    model is loaded, never per prediction, so a live session never reopens its
    weights.
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
        """Read the serialized ONNX graph for ``task``.

        Bytes are read fresh from the bundle every time a model is loaded,
        never cached, so a file that changed on disk is always picked up.
        Structural compatibility is checked separately, by the ONNX Runtime
        session this payload builds (see ``_inference.py``'s signature check)
        — there is no hash comparison here.

        Args:
            task: ``"age"`` or ``"gender"``.

        Returns:
            The artifact's current bytes.

        Raises:
            FileNotFoundError: The artifact is missing from the bundle.
        """
        spec = self._specs[task]
        return self._resource(spec.filename).read_bytes()

    @contextmanager
    def shape_predictor_file(self) -> Iterator[Path]:
        """Yield a filesystem path to the five-point landmark model.

        dlib deserializes from a path rather than from bytes, so the resource is
        materialized for the duration of the context. The path is only valid
        inside the ``with`` block: an installed package may have to extract it
        from a zip import, and the extracted copy is removed on exit.
        Structural compatibility (producing five landmark parts) is checked
        separately, by ``_faces.py`` when it actually loads the predictor.

        Raises:
            FileNotFoundError: The artifact is missing from the bundle.
        """
        spec = self._shape_predictor
        resource = self._resource(spec.filename)
        with resources.as_file(resource) as path:
            if not path.is_file():
                raise FileNotFoundError(f"{self._origin}: {spec.filename} not found")
            yield path

    def _resource(self, name: str) -> Traversable:
        resource = self._root.joinpath(name)
        if isinstance(resource, Path):
            _check_contained(resource, self._root, self._origin, name)
        return resource


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


def _artifact(
    section: Mapping[str, Any], origin: str, label: str
) -> tuple[str, str | None, int | None]:
    """Parse an artifact's filename, and its sha256/bytes if present.

    ``filename`` is the only field this release actually uses to locate and
    load a model; ``sha256``/``bytes`` are carried through as informational
    metadata when a manifest happens to record them (the shipped manifest
    does, for provenance), but are optional and never checked against the
    artifact's actual bytes.
    """
    filename = section.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{origin}: manifest {label}.filename is missing or malformed")
    _check_flat_filename(filename, origin, label)
    sha256 = section.get("sha256")
    if sha256 is not None and not (isinstance(sha256, str) and sha256):
        raise ValueError(f"{origin}: manifest {label}.sha256 is malformed")
    size_bytes = section.get("bytes")
    if size_bytes is not None and (not isinstance(size_bytes, int) or isinstance(size_bytes, bool)):
        raise ValueError(f"{origin}: manifest {label}.bytes is malformed")
    return filename, (str(sha256).lower() if sha256 is not None else None), size_bytes


def _check_flat_filename(filename: str, origin: str, label: str) -> None:
    """Reject anything that could resolve outside the bundle directory.

    Every artifact this manifest format names lives directly under the bundle
    root (the shipped manifest and build_bundle.py both only ever write flat
    names). `_resource()` joins this filename onto the bundle root without
    further checks, so an absolute path or a `..` component here would let a
    corrupt or malicious manifest read a file the caller never intended to
    expose, breaking `from_model_dir()`'s "backed entirely by this directory"
    contract.
    """
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError(
            f"{origin}: manifest {label}.filename {filename!r} is not a plain filename"
        )


def _check_contained(resource: Path, root: Traversable, origin: str, name: str) -> None:
    """Reject a resource that resolves outside the bundle root.

    `_check_flat_filename()` only rejects traversal spelled out in the
    manifest's `filename` string; a flat, innocent-looking name can still be a
    symlink on disk that points elsewhere. A directory-backed bundle root is a
    real `Path`, so its resolved target can be checked directly; this is a
    no-op for the installed package's own (non-symlinked) resources and for
    any other `Traversable` implementation that isn't filesystem-backed.
    """
    try:
        resolved = resource.resolve(strict=True)
    except OSError:
        return
    root_path = root if isinstance(root, Path) else Path(str(root))
    if not resolved.is_relative_to(root_path.resolve()):
        raise ValueError(
            f"{origin}: {name} resolves outside the bundle directory "
            "(a symlink escaping the bundle is not accepted)"
        )


def _optional_section(manifest: Mapping[str, Any], *path: str) -> Mapping[str, Any] | None:
    """Like `_section()`, but returns `None` instead of raising when absent.

    Used for the `source` block, which is informational provenance rather
    than something this release requires every bundle to declare.
    """
    node: Any = manifest
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, dict) else None


def _parse_neural_model(manifest: Mapping[str, Any], task: Task, origin: str) -> NeuralModelSpec:
    model = _section(manifest, origin, "models", task)
    if model.get("task") != task:
        raise ValueError(f"{origin}: manifest models.{task} declares task {model.get('task')!r}")
    filename, sha256, size_bytes = _artifact(
        _section(manifest, origin, "models", task, "artifact"), origin, f"models.{task}.artifact"
    )
    source = _optional_section(manifest, "models", task, "source")
    if source is not None:
        source_filename, source_sha256, _ = _artifact(source, origin, f"models.{task}.source")
    else:
        source_filename = source_sha256 = None
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
    if _as_list_or_none(model_input.get("shape")) != expected_input:
        raise ValueError(
            f"{origin}: manifest models.{task}.input.shape must be {expected_input}, "
            f"got {model_input.get('shape')!r}"
        )
    expected_output = ["N", CLASS_COUNTS[task]]
    if _as_list_or_none(model_output.get("shape")) != expected_output:
        raise ValueError(
            f"{origin}: manifest models.{task}.output.shape must be {expected_output}, "
            f"got {model_output.get('shape')!r}"
        )
    return size


def _as_list_or_none(value: Any) -> list[Any] | None:
    """Return `value` if it is already a list, `[]` for a missing/null field.

    Anything else (a bare number, bool, string, or object) is not a shape or
    label list under any manifest this format allows, so it is reported by
    the caller's own mismatch message instead of being coerced through
    `list()`, which raises a bare `TypeError` on a non-iterable value such as
    an int or `null`.
    """
    if value is None:
        return []
    return value if isinstance(value, list) else None


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
    with np.errstate(over="ignore"):
        means_are_finite = all(
            math.isfinite(value) and np.isfinite(np.float32(value)) for value in means
        )
    if not means_are_finite:
        raise ValueError(
            f"{origin}: manifest models.{task}.input.normalization.means must be finite "
            "and representable as float32"
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
    if _as_list_or_none(labels) != list(GENDER_LABELS):
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
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in weights
        )
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
    source = _optional_section(manifest, "shape_predictor", "source")
    if source is not None:
        source_filename, source_sha256, _ = _artifact(source, origin, "shape_predictor.source")
    else:
        source_filename = source_sha256 = None
    return ShapePredictorSpec(
        filename=filename,
        sha256=sha256,
        size_bytes=size_bytes,
        parts=5,
        source_filename=source_filename,
        source_sha256=source_sha256,
    )
