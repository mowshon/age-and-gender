#!/usr/bin/env python3
"""Build the two release ONNX graphs from dlib's trusted XML export."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODELS = {
    "age": {
        "name": "dnn_age_predictor_v1.dat",
        "sha256": "4b78d4d7055e22620e362884b5551caa9379080277338aa2d4cdfc592f0e9fa3",
        "input_size": 64,
        "classes": 81,
    },
    "gender": {
        "name": "dnn_gender_classifier_v1.dat",
        "sha256": "85453d6f6585c8e02ada95929956783c780dc04dcec5bdfd14af82f15c99ba41",
        "input_size": 32,
        "classes": 2,
    },
}
EXPECTED_LAYERS = {
    "age": [
        "loss_multiclass_log", "fc", "avg_pool", "relu", "add_prev",
        "affine_con", "con", "relu", "affine_con", "con", "tag:1",
        "relu", "add_prev", "avg_pool", "skip:1", "tag:2", "affine_con",
        "con", "relu", "affine_con", "con", "tag:1", "relu", "add_prev",
        "affine_con", "con", "relu", "affine_con", "con", "tag:1", "relu",
        "add_prev", "avg_pool", "skip:1", "tag:2", "affine_con", "con",
        "relu", "affine_con", "con", "tag:1", "relu", "add_prev",
        "affine_con", "con", "relu", "affine_con", "con", "tag:1",
        "max_pool", "relu", "affine_con", "con", "input_rgb_image",
    ],
    "gender": [
        "loss_multiclass_log", "fc", "multiply", "relu", "fc", "multiply",
        "avg_pool", "relu", "affine_con", "con", "relu", "affine_con",
        "con", "avg_pool", "relu", "affine_con", "con", "relu",
        "affine_con", "con", "input_rgb_image_sized",
    ],
}
MEANS = [122.781998, 117.000999, 104.297997]
OPSET = 18
IR_VERSION = 10
# Parity was established with graph optimization disabled; see the conversion
# report for the measurements that rejected the other levels. Consumers must
# create sessions with exactly these options, so they belong in the bundle
# rather than only in prose.
RUNTIME = {
    "provider": "CPUExecutionProvider",
    "graph_optimization_level": "ORT_DISABLE_ALL",
    "intra_op_num_threads": 1,
    "inter_op_num_threads": 1,
}


@dataclass(frozen=True)
class Value:
    name: str
    shape: tuple[int, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"unexpected SHA-256 for {path}: {actual}")


def layer_kind(layer: ET.Element) -> str:
    layer_type = layer.attrib["type"]
    if layer_type in {"tag", "skip"}:
        return f"{layer_type}:{layer.attrib['id']}"
    children = list(layer)
    if len(children) != 1:
        raise ValueError(f"layer {layer.attrib['idx']} must have exactly one definition")
    return children[0].tag


def parse_floats(element: ET.Element) -> np.ndarray:
    values = np.fromstring(element.text or "", dtype=np.float32, sep=" ")
    if values.size == 0:
        raise ValueError(f"{element.tag} has no parameters")
    return values


class GraphBuilder:
    def __init__(self, task: str, size: int, classes: int) -> None:
        self.task = task
        self.size = size
        self.classes = classes
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.tags: dict[str, Value] = {}

    def initializer(self, name: str, values: np.ndarray) -> str:
        array = np.ascontiguousarray(values)
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def unary(self, op: str, value: Value, index: str) -> Value:
        output = f"layer_{index}_{op.lower()}"
        self.nodes.append(helper.make_node(op, [value.name], [output], name=output))
        return Value(output, value.shape)

    def convolution(self, value: Value, definition: ET.Element, index: str) -> Value:
        channels, height, width = value.shape
        outputs = int(definition.attrib["num_filters"])
        kernel_h = int(definition.attrib["nr"])
        kernel_w = int(definition.attrib["nc"])
        stride_h = int(definition.attrib["stride_y"])
        stride_w = int(definition.attrib["stride_x"])
        pad_h = int(definition.attrib["padding_y"])
        pad_w = int(definition.attrib["padding_x"])
        parameters = parse_floats(definition)
        weight_count = outputs * channels * kernel_h * kernel_w
        if parameters.size != weight_count + outputs:
            raise ValueError(
                f"layer {index} convolution expected {weight_count + outputs} "
                f"parameters, got {parameters.size}"
            )
        weights = parameters[:weight_count].reshape(
            outputs, channels, kernel_h, kernel_w
        )
        bias = parameters[weight_count:]
        output = f"layer_{index}_conv"
        self.nodes.append(
            helper.make_node(
                "Conv",
                [
                    value.name,
                    self.initializer(f"layer_{index}_weights", weights),
                    self.initializer(f"layer_{index}_bias", bias),
                ],
                [output],
                name=output,
                kernel_shape=[kernel_h, kernel_w],
                strides=[stride_h, stride_w],
                pads=[pad_h, pad_w, pad_h, pad_w],
            )
        )
        output_h = 1 + (height + 2 * pad_h - kernel_h) // stride_h
        output_w = 1 + (width + 2 * pad_w - kernel_w) // stride_w
        return Value(output, (outputs, output_h, output_w))

    def affine(self, value: Value, definition: ET.Element, index: str) -> Value:
        channels, _, _ = value.shape
        parameters = parse_floats(definition)
        if parameters.size != 2 * channels:
            raise ValueError(
                f"layer {index} affine expected {2 * channels} parameters, "
                f"got {parameters.size}"
            )
        broadcast_shape = (1, channels, 1, 1)
        scaled = f"layer_{index}_affine_scale"
        output = f"layer_{index}_affine"
        self.nodes.append(
            helper.make_node(
                "Mul",
                [
                    value.name,
                    self.initializer(
                        f"layer_{index}_scale", parameters[:channels].reshape(broadcast_shape)
                    ),
                ],
                [scaled],
                name=scaled,
            )
        )
        self.nodes.append(
            helper.make_node(
                "Add",
                [
                    scaled,
                    self.initializer(
                        f"layer_{index}_offset", parameters[channels:].reshape(broadcast_shape)
                    ),
                ],
                [output],
                name=output,
            )
        )
        return Value(output, value.shape)

    def pool(self, value: Value, definition: ET.Element, index: str) -> Value:
        kernel_h = int(definition.attrib["nr"])
        kernel_w = int(definition.attrib["nc"])
        if kernel_h == 0 and kernel_w == 0:
            output = f"layer_{index}_global_average_pool"
            self.nodes.append(
                helper.make_node(
                    "GlobalAveragePool", [value.name], [output], name=output
                )
            )
            return Value(output, (value.shape[0], 1, 1))

        stride_h = int(definition.attrib["stride_y"])
        stride_w = int(definition.attrib["stride_x"])
        pad_h = int(definition.attrib["padding_y"])
        pad_w = int(definition.attrib["padding_x"])
        op = "MaxPool" if definition.tag == "max_pool" else "AveragePool"
        output = f"layer_{index}_{op.lower()}"
        attributes: dict[str, Any] = {
            "kernel_shape": [kernel_h, kernel_w],
            "strides": [stride_h, stride_w],
            "pads": [pad_h, pad_w, pad_h, pad_w],
            "ceil_mode": 0,
        }
        if op == "AveragePool":
            attributes["count_include_pad"] = 0
        self.nodes.append(
            helper.make_node(op, [value.name], [output], name=output, **attributes)
        )
        channels, height, width = value.shape
        output_h = 1 + (height + 2 * pad_h - kernel_h) // stride_h
        output_w = 1 + (width + 2 * pad_w - kernel_w) // stride_w
        return Value(output, (channels, output_h, output_w))

    def fully_connected(
        self, value: Value, definition: ET.Element, index: str
    ) -> Value:
        outputs = int(definition.attrib["num_outputs"])
        inputs = int(np.prod(value.shape))
        parameters = parse_floats(definition)
        if parameters.size != inputs * outputs + outputs:
            raise ValueError(
                f"layer {index} FC expected {inputs * outputs + outputs} parameters, "
                f"got {parameters.size}"
            )
        flattened = f"layer_{index}_flatten"
        output = "logits" if outputs == self.classes else f"layer_{index}_fc"
        self.nodes.append(
            helper.make_node("Flatten", [value.name], [flattened], name=flattened, axis=1)
        )
        self.nodes.append(
            helper.make_node(
                "Gemm",
                [
                    flattened,
                    self.initializer(
                        f"layer_{index}_weights",
                        parameters[: inputs * outputs].reshape(inputs, outputs),
                    ),
                    self.initializer(
                        f"layer_{index}_bias", parameters[inputs * outputs :]
                    ),
                ],
                [output],
                name=f"layer_{index}_fc",
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0,
            )
        )
        return Value(output, (outputs, 1, 1))

    def pad_to(self, value: Value, target: tuple[int, int, int], name: str) -> Value:
        differences = tuple(end - start for start, end in zip(value.shape, target))
        if any(difference < 0 for difference in differences):
            raise ValueError(f"cannot pad {value.shape} to {target}")
        if not any(differences):
            return value
        output = f"{name}_padded"
        pads = np.asarray([0, 0, 0, 0, 0, *differences], dtype=np.int64)
        self.nodes.append(
            helper.make_node(
                "Pad",
                [value.name, self.initializer(f"{name}_pads", pads)],
                [output],
                name=output,
                mode="constant",
            )
        )
        return Value(output, target)

    def add_previous(self, value: Value, tag_id: str, index: str) -> Value:
        tagged = self.tags.get(tag_id)
        if tagged is None:
            raise ValueError(f"layer {index} references unavailable tag {tag_id}")
        target = tuple(max(left, right) for left, right in zip(value.shape, tagged.shape))
        left = self.pad_to(value, target, f"layer_{index}_left")
        right = self.pad_to(tagged, target, f"layer_{index}_right")
        output = f"layer_{index}_residual_add"
        self.nodes.append(
            helper.make_node("Add", [left.name, right.name], [output], name=output)
        )
        return Value(output, target)

    def build(self, layers: list[ET.Element]) -> onnx.ModelProto:
        current: Value | None = None
        for layer in reversed(layers):
            index = layer.attrib["idx"]
            layer_type = layer.attrib["type"]
            if layer_type == "input":
                definition = next(iter(layer))
                means = [float(definition.attrib[channel]) for channel in ("r", "g", "b")]
                if not np.array_equal(
                    np.asarray(means, dtype=np.float32), np.asarray(MEANS, dtype=np.float32)
                ):
                    raise ValueError(f"unexpected {self.task} input means: {means}")
                if definition.tag == "input_rgb_image_sized" and (
                    int(definition.attrib["nr"]) != self.size
                    or int(definition.attrib["nc"]) != self.size
                ):
                    raise ValueError(f"unexpected {self.task} input dimensions")
                current = Value("images", (3, self.size, self.size))
                continue
            if layer_type == "loss":
                continue
            if current is None:
                raise ValueError("network does not end in an input layer")
            if layer_type == "tag":
                self.tags[layer.attrib["id"]] = current
                continue
            if layer_type == "skip":
                tag_id = layer.attrib["id"]
                if tag_id not in self.tags:
                    raise ValueError(f"layer {index} skips to unavailable tag {tag_id}")
                current = self.tags[tag_id]
                continue

            definition = next(iter(layer))
            if definition.tag == "con":
                current = self.convolution(current, definition, index)
            elif definition.tag == "affine_con":
                current = self.affine(current, definition, index)
            elif definition.tag == "relu":
                current = self.unary("Relu", current, index)
            elif definition.tag in {"max_pool", "avg_pool"}:
                current = self.pool(current, definition, index)
            elif definition.tag == "multiply":
                scalar = np.float32(definition.attrib["val"])
                if scalar != np.float32(0.5):
                    raise ValueError(f"unexpected multiply value at layer {index}: {scalar}")
                output = f"layer_{index}_multiply"
                self.nodes.append(
                    helper.make_node(
                        "Mul",
                        [
                            current.name,
                            self.initializer(
                                f"layer_{index}_scalar", np.asarray(scalar, dtype=np.float32)
                            ),
                        ],
                        [output],
                        name=output,
                    )
                )
                current = Value(output, current.shape)
            elif definition.tag == "fc":
                current = self.fully_connected(current, definition, index)
            elif definition.tag == "add_prev":
                current = self.add_previous(current, definition.attrib["tag"], index)
            else:
                raise ValueError(f"unsupported layer {index}: {definition.tag}")

        if current is None or current.name != "logits":
            raise ValueError(f"{self.task} graph did not terminate in expected logits")
        self.nodes.append(
            helper.make_node(
                "Softmax", [current.name], ["probabilities"], name="probabilities", axis=1
            )
        )
        graph = helper.make_graph(
            self.nodes,
            f"age-and-gender-{self.task}-v1",
            [
                helper.make_tensor_value_info(
                    "images", TensorProto.FLOAT, ["N", 3, self.size, self.size]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "probabilities", TensorProto.FLOAT, ["N", self.classes]
                )
            ],
            initializer=self.initializers,
            value_info=[
                helper.make_tensor_value_info(
                    "logits", TensorProto.FLOAT, ["N", self.classes]
                )
            ],
        )
        model = helper.make_model(
            graph,
            producer_name="age-and-gender",
            producer_version="1",
            domain="age-and-gender",
            model_version=1,
            opset_imports=[helper.make_opsetid("", OPSET)],
            ir_version=IR_VERSION,
        )
        model.doc_string = (
            f"{self.task} classifier converted from the original dlib weights; "
            "input is already-normalized RGB NCHW float32."
        )
        onnx.checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(inferred, full_check=True)
        return inferred


def load_layers(path: Path, task: str) -> list[ET.Element]:
    layers = list(ET.parse(path).getroot())
    actual = [layer_kind(layer) for layer in layers]
    if actual != EXPECTED_LAYERS[task]:
        raise ValueError(f"unexpected {task} architecture:\n{actual}")
    indices = [int(layer.attrib["idx"]) for layer in layers]
    if indices != list(range(len(layers))):
        raise ValueError(f"{task} layer indices are not contiguous")
    return layers


def save_model(model: onnx.ModelProto, path: Path) -> None:
    path.write_bytes(model.SerializeToString(deterministic=True))
    loaded = onnx.load(path, load_external_data=False)
    if any(initializer.data_location for initializer in loaded.graph.initializer):
        raise ValueError(f"{path} unexpectedly uses external tensor data")
    onnx.checker.check_model(loaded, full_check=True)


def converter_revision() -> str:
    digest = hashlib.sha256()
    for relative in (
        "tools/network_definitions.h",
        "tools/conversion/export_dlib.cpp",
        "tools/conversion/build_onnx.py",
        "tools/conversion/notices/Cydral-age-gender-models.md",
    ):
        digest.update(relative.encode("ascii"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def make_manifest(output: Path, source_dir: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for task, contract in SOURCE_MODELS.items():
        source = source_dir / contract["name"]
        target = output / f"{task}-v1.onnx"
        models[task] = {
            "task": task,
            "source": {
                "filename": contract["name"],
                "sha256": contract["sha256"],
                "bytes": source.stat().st_size,
            },
            "artifact": {
                "filename": target.name,
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            },
            "input": {
                "name": "images",
                "dtype": "float32",
                "shape": ["N", 3, contract["input_size"], contract["input_size"]],
                "color_order": "RGB",
                "normalization": {"means": MEANS, "scale": 0.00390625},
            },
            "output": {
                "name": "probabilities",
                "dtype": "float32",
                "shape": ["N", contract["classes"]],
                "softmax_in_graph": True,
            },
        }
    models["gender"]["labels"] = ["female", "male"]
    models["age"]["age_weights"] = [0.25, *range(1, 81)]
    return {
        "schema_version": 1,
        "bundle_id": "age-and-gender-v1",
        "converter_revision": converter_revision(),
        "dlib": {"version": "19.20.0", "source": "tools/vendor/dlib"},
        "onnx": {"version": onnx.__version__, "opset": OPSET, "ir_version": IR_VERSION},
        "runtime": RUNTIME,
        "tool_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": importlib.metadata.version("onnxruntime"),
        },
        "preprocessing": {
            "face_chip_padding": 0.2,
            "chips_extracted_individually": True,
            "normalization_formula": "(RGB_byte - channel_mean) / 256.0",
        },
        "graph_policy": {
            "affine_folded": False,
            "reduced_precision": False,
            "external_data": False,
            "dynamic_positive_batch": True,
            "residual_padding": "trailing channels, rows, and columns; top-left origin preserved",
        },
        "license": {
            "source": "https://github.com/davisking/dlib-models",
            "attribution": "Cydral Technology",
            "spdx": "CC0-1.0",
            "notice": {
                "path": "notices/Cydral-age-gender-models.md",
                "sha256": sha256_file(output / "notices/Cydral-age-gender-models.md"),
            },
        },
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--source-models", type=Path, default=ROOT / "models")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_models.resolve()
    exporter = args.exporter.resolve()
    work_dir = args.work_dir.resolve()
    output = args.output.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    notice_dir = output / "notices"
    notice_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "tools/conversion/notices/Cydral-age-gender-models.md",
        notice_dir / "Cydral-age-gender-models.md",
    )

    sources: dict[str, Path] = {}
    for task, contract in SOURCE_MODELS.items():
        source = source_dir / contract["name"]
        require_hash(source, str(contract["sha256"]))
        sources[task] = source

    xml_paths = {task: work_dir / f"{task}.xml" for task in SOURCE_MODELS}
    subprocess.run(
        [
            str(exporter),
            "--age-model", str(sources["age"]),
            "--gender-model", str(sources["gender"]),
            "--age-xml", str(xml_paths["age"]),
            "--gender-xml", str(xml_paths["gender"]),
        ],
        check=True,
    )

    for task, contract in SOURCE_MODELS.items():
        layers = load_layers(xml_paths[task], task)
        model = GraphBuilder(
            task, int(contract["input_size"]), int(contract["classes"])
        ).build(layers)
        save_model(model, output / f"{task}-v1.onnx")

    manifest = make_manifest(output, source_dir)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
