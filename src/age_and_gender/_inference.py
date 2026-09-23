"""ONNX Runtime sessions and the float32 tensor preparation they consume.

Sessions are built from the options recorded in the bundle manifest, created
lazily on first use, and reused for every later prediction. Parity with the
legacy C++ networks depends on both the session options and the exact order of
the preprocessing arithmetic, so neither is inferred from runtime defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import onnxruntime as ort

from ._models import TASKS, ModelBundle, NeuralModelSpec
from ._types import Task

__all__ = ["InferenceEngine", "NeuralNetwork", "prepare_batch"]

GRAPH_OPTIMIZATION_LEVELS: Final[dict[str, ort.GraphOptimizationLevel]] = {
    "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
_PROBABILITY_SUM_TOLERANCE: Final = 1e-3


def prepare_batch(chips: np.ndarray | Sequence[np.ndarray], spec: NeuralModelSpec) -> np.ndarray:
    """Turn aligned uint8 RGB chips into the network's float32 NCHW input.

    Args:
        chips: One ``[N, size, size, 3]`` uint8 array, or a sequence of
            ``[size, size, 3]`` uint8 chips in face order.
        spec: Model specification supplying the chip size and channel means.

    Returns:
        A C-contiguous ``[N, 3, size, size]`` float32 array.

    Raises:
        ValueError: The chips have the wrong dtype, rank, or spatial size.
    """
    batch = _stack_chips(chips, spec)
    # Byte to float32 is exact, the mean subtraction is the only rounding step,
    # and 1/256 is a power of two, so dividing by 256 only adjusts the exponent.
    # Reordering these steps would change the last bit of some inputs.
    values = batch.astype(np.float32)
    values -= np.asarray(spec.normalization.means, dtype=np.float32)
    values /= np.float32(256.0)
    return np.ascontiguousarray(values.transpose(0, 3, 1, 2))


def _stack_chips(chips: np.ndarray | Sequence[np.ndarray], spec: NeuralModelSpec) -> np.ndarray:
    size = spec.chip_size
    if isinstance(chips, np.ndarray):
        batch = chips
    elif len(chips) == 0:
        batch = np.empty((0, size, size, 3), dtype=np.uint8)
    else:
        batch = np.stack(list(chips))
    if batch.dtype != np.uint8:
        raise ValueError(f"{spec.task} chips must be uint8, got {batch.dtype}")
    if batch.ndim != 4 or batch.shape[1:] != (size, size, 3):
        raise ValueError(
            f"{spec.task} chips must have shape [N, {size}, {size}, 3], got {batch.shape}"
        )
    return batch


class NeuralNetwork:
    """One converted network and the session that runs it.

    The specification and session options are fixed when the object is created.
    The session itself is built on first use and then reused, so weights are
    never re-read and the graph is never re-optimized per prediction.
    """

    def __init__(self, bundle: ModelBundle, task: Task) -> None:
        self._bundle = bundle
        self._spec = bundle.spec(task)
        self._runtime = bundle.runtime
        self._session: ort.InferenceSession | None = None

    def __repr__(self) -> str:
        state = "loaded" if self._session is not None else "not loaded"
        return f"NeuralNetwork(task={self._spec.task!r}, {state})"

    @property
    def spec(self) -> NeuralModelSpec:
        """Validated description of the model this network runs."""
        return self._spec

    @property
    def task(self) -> Task:
        """The task this network was created for."""
        return self._spec.task

    @property
    def bundle(self) -> ModelBundle:
        """The bundle the model was resolved from."""
        return self._bundle

    @property
    def is_loaded(self) -> bool:
        """Whether the session has been created."""
        return self._session is not None

    def ensure_loaded(self) -> None:
        """Create and validate the session if it does not exist yet.

        Raises:
            FileNotFoundError: The bundle is missing the model artifact.
            ValueError: The artifact fails its hash check, cannot be loaded, or
                does not match the graph signature the manifest declares.
        """
        if self._session is None:
            self._session = self._create_session()

    def probabilities(self, chips: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """Run the network over aligned uint8 chips.

        Args:
            chips: Chips in face order, as accepted by :func:`prepare_batch`.

        Returns:
            A ``[N, classes]`` float32 array of probabilities, in face order.

        Raises:
            ValueError: The chips or the resulting probabilities are invalid.
        """
        return self.run(prepare_batch(chips, self._spec))

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run the network over a prepared input tensor.

        Args:
            inputs: C-contiguous ``[N, 3, size, size]`` float32 array.

        Returns:
            A ``[N, classes]`` float32 array of probabilities.

        Raises:
            ValueError: The tensor or the returned probabilities are invalid.
        """
        self._check_inputs(inputs)
        if inputs.shape[0] == 0:
            # No faces means no work: an empty batch would still build the
            # session and is rejected by some execution providers.
            return np.zeros((0, self._spec.classes), dtype=np.float32)
        self.ensure_loaded()
        assert self._session is not None  # narrowed by ensure_loaded
        outputs = self._session.run([self._spec.output_name], {self._spec.input_name: inputs})[0]
        self._check_probabilities(outputs, inputs.shape[0])
        return outputs

    def _create_session(self) -> ort.InferenceSession:
        spec = self._spec
        runtime = self._runtime
        level = GRAPH_OPTIMIZATION_LEVELS.get(runtime.graph_optimization_level)
        if level is None:
            raise ValueError(
                f"{self._bundle.origin}: unknown graph optimization level "
                f"{runtime.graph_optimization_level!r}"
            )
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = runtime.intra_op_num_threads
        options.inter_op_num_threads = runtime.inter_op_num_threads
        payload = self._bundle.model_bytes(spec.task)
        try:
            session = ort.InferenceSession(
                payload, sess_options=options, providers=[runtime.provider]
            )
        except Exception as error:
            raise ValueError(
                f"{self._bundle.origin}: {spec.filename} could not be loaded as the "
                f"{spec.task} model: {error}"
            ) from error
        providers = session.get_providers()
        if providers != [runtime.provider]:
            raise ValueError(
                f"{self._bundle.origin}: {spec.task} session runs on {providers}, "
                f"not the validated {[runtime.provider]}"
            )
        self._check_signature(session)
        return session

    def _check_signature(self, session: ort.InferenceSession) -> None:
        spec = self._spec
        origin = f"{self._bundle.origin}: {spec.filename}"
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"{origin} has {len(inputs)} inputs and {len(outputs)} outputs; "
                "the supported graph has exactly one of each"
            )
        if inputs[0].name != spec.input_name or outputs[0].name != spec.output_name:
            raise ValueError(
                f"{origin} exposes {inputs[0].name!r}/{outputs[0].name!r}, "
                f"manifest declares {spec.input_name!r}/{spec.output_name!r}"
            )
        if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
            raise ValueError(f"{origin} is not a float32 graph")
        if list(inputs[0].shape[1:]) != list(spec.input_shape):
            raise ValueError(
                f"{origin} takes {inputs[0].shape}, manifest declares "
                f"[N, {', '.join(str(value) for value in spec.input_shape)}]"
            )
        if len(outputs[0].shape) != 2 or outputs[0].shape[1] != spec.classes:
            raise ValueError(
                f"{origin} returns {outputs[0].shape}, manifest declares "
                f"[N, {spec.classes}] for the {spec.task} task"
            )
        # The manifest declares a symbolic batch on both tensors. A graph with a
        # fixed batch, or one whose output batch is unrelated to its input
        # batch, cannot honour the "one row per face, in face order" contract.
        input_batch, output_batch = inputs[0].shape[0], outputs[0].shape[0]
        if not isinstance(input_batch, str) or not isinstance(output_batch, str):
            raise ValueError(
                f"{origin} has a fixed batch dimension ({inputs[0].shape} -> "
                f"{outputs[0].shape}); the manifest declares a dynamic batch"
            )
        if input_batch != output_batch:
            raise ValueError(
                f"{origin} maps batch {input_batch!r} to {output_batch!r}; the "
                "output must carry one row per input chip"
            )

    def _check_inputs(self, inputs: np.ndarray) -> None:
        spec = self._spec
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"{spec.task} inputs must be a NumPy array")
        if inputs.dtype != np.float32:
            raise ValueError(f"{spec.task} inputs must be float32, got {inputs.dtype}")
        if inputs.ndim != 4 or inputs.shape[1:] != spec.input_shape:
            raise ValueError(
                f"{spec.task} inputs must have shape [N, "
                f"{', '.join(str(value) for value in spec.input_shape)}], "
                f"got {inputs.shape}"
            )
        if not inputs.flags["C_CONTIGUOUS"]:
            raise ValueError(f"{spec.task} inputs must be C-contiguous")

    def _check_probabilities(self, outputs: np.ndarray, rows: int) -> None:
        spec = self._spec
        if outputs.dtype != np.float32 or outputs.ndim != 2 or outputs.shape[1] != spec.classes:
            raise ValueError(
                f"{spec.task} model returned {outputs.dtype} {outputs.shape}, "
                f"expected float32 [{rows}, {spec.classes}]"
            )
        # Results are matched to faces by position, so a short or long batch
        # would silently attach one face's prediction to another.
        if outputs.shape[0] != rows:
            raise ValueError(f"{spec.task} model returned {outputs.shape[0]} rows for {rows} chips")
        if not np.isfinite(outputs).all():
            raise ValueError(f"{spec.task} model returned non-finite probabilities")
        if outputs.size and (outputs.min() < 0.0 or outputs.max() > 1.0):
            raise ValueError(f"{spec.task} model returned values outside [0, 1]")
        sums = outputs.sum(axis=1, dtype=np.float64)
        if outputs.size and np.abs(sums - 1.0).max() > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                f"{spec.task} model rows do not sum to one; the graph must apply "
                "softmax and return probabilities"
            )


class InferenceEngine:
    """The age and gender networks of one bundle, replaceable one at a time."""

    def __init__(self, bundle: ModelBundle) -> None:
        self._networks: dict[Task, NeuralNetwork] = {
            task: NeuralNetwork(bundle, task) for task in TASKS
        }

    def network(self, task: Task) -> NeuralNetwork:
        """Return the network currently serving ``task``."""
        return self._networks[task]

    @property
    def age(self) -> NeuralNetwork:
        """The network currently serving the age task."""
        return self._networks["age"]

    @property
    def gender(self) -> NeuralNetwork:
        """The network currently serving the gender task."""
        return self._networks["gender"]

    def replace(self, task: Task, bundle: ModelBundle) -> NeuralNetwork:
        """Swap in ``bundle``'s model for ``task`` once it is fully validated.

        The replacement session is created before anything is swapped, so a
        failed load leaves the previously validated model and its live session
        in place.

        Args:
            task: Task whose model should be replaced.
            bundle: Validated bundle supplying the replacement.

        Returns:
            The installed network.

        Raises:
            FileNotFoundError: The bundle is missing the model artifact.
            ValueError: The replacement fails validation; nothing is replaced.
        """
        candidate = NeuralNetwork(bundle, task)
        candidate.ensure_loaded()
        self._networks[task] = candidate
        return candidate
