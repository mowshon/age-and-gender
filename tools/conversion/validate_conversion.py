#!/usr/bin/env python3
"""Validate converted ONNX networks against frozen and live dlib references."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The legacy rounding rules are owned by PR-1. Importing them here keeps one
# authority for the public contract instead of a second copy that can drift.
from tests.parity.legacy_contract import (  # noqa: E402
    age_expectation,
    confidence_percent,
    gender_result,
    round_age,
)

FIXTURES = ROOT / "tests/fixtures/legacy"
MODELS = {
    "age": {
        "source": "dnn_age_predictor_v1.dat",
        "size": 64,
        "classes": 81,
        "logits_tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "probability_tolerance": {"atol": 1e-6, "rtol": 1e-4},
    },
    "gender": {
        "source": "dnn_gender_classifier_v1.dat",
        "size": 32,
        "classes": 2,
        "logits_tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "probability_tolerance": {"atol": 1e-6, "rtol": 1e-4},
    },
}
MEANS = np.asarray([122.781998, 117.000999, 104.297997], dtype=np.float32)
BATCH_SIZES = (1, 2, 7, 32)
AGE_EXPECTATION_ATOL = 1e-4
# PR-1 proposed atol=1e-5 for "intermediate logits / floating activations" and
# stated the floating thresholds still had to be ratified against measurement.
# Measured here on the deep 256-channel stage (`layer_3_relu`, values to 42.03):
#
#   max|dlib - exact float64| = 5.10e-05
#   max|onnx - exact float64| = 3.28e-05   <- the converted graph is the closer one
#   max|onnx - dlib|          = 4.96e-05
#
# That is float32 accumulation-order noise over ~2300 multiply-accumulates
# (about 13 ULP at that magnitude), and it lands on post-ReLU near-zero elements
# where rtol contributes nothing. An atol of 1e-5 is below the noise floor of the
# reference itself. This tolerance applies only to internal activations; logits,
# probabilities and public results keep PR-1's thresholds unchanged. A structural
# conversion error moves these tensors by orders of magnitude more than this.
ACTIVATION_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}
MEASURED_ACTIVATION_NOISE = {
    "stage": "age:layer_3_relu",
    "dlib_vs_exact_float64": 5.1041e-05,
    "onnx_vs_exact_float64": 3.2791e-05,
    "onnx_vs_dlib": 4.9591e-05,
    "stage_value_range": [0.0, 42.0283],
}
SELECTED_SETTING = "disabled"
OPTIMIZATION_LEVELS = {
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
THREAD_SETTINGS = {"intra_op_num_threads": 1, "inter_op_num_threads": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_tensor_names(model: onnx.ModelProto) -> dict[int, str]:
    """Map each dlib layer index to the converted tensor the builder named for it."""
    names: dict[int, str] = {}
    for node in model.graph.node:
        if not node.name.startswith("layer_"):
            continue
        index = node.name.split("_")[1]
        if index.isdigit():
            names[int(index)] = node.output[0]
    return names


def add_debug_outputs(model_path: Path, stages: tuple[int, ...] = ()) -> bytes:
    """Expose logits, and optionally per-stage tensors, as extra graph outputs.

    Extra outputs pin tensors that graph optimization would otherwise fuse away,
    so an instrumented graph is only representative of the shipped graph when
    optimization is disabled.
    """
    model = onnx.load(model_path, load_external_data=False)
    probabilities = model.graph.output[0]
    classes = probabilities.type.tensor_type.shape.dim[1].dim_value
    model.graph.output.append(
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", classes])
    )
    if stages:
        names = stage_tensor_names(model)
        # Reuse the inferred value_info so each extra output carries the shape
        # the checker requires, rather than an unshaped tensor type.
        inferred = {
            value.name: value
            for value in shape_inference.infer_shapes(model, strict_mode=True).graph.value_info
        }
        for index in stages:
            if index not in names:
                raise ValueError(f"no converted tensor for dlib layer {index}")
            name = names[index]
            if name == "logits":
                continue
            if name not in inferred:
                raise ValueError(f"shape inference did not describe stage tensor {name}")
            model.graph.output.append(inferred[name])
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString(deterministic=True)


def make_session(model: bytes, optimization: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    options.intra_op_num_threads = THREAD_SETTINGS["intra_op_num_threads"]
    options.inter_op_num_threads = THREAD_SETTINGS["inter_op_num_threads"]
    return ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])


def max_error(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))


def max_violation(expected: np.ndarray, actual: np.ndarray, tolerance: dict[str, float]) -> float:
    """Return how far the worst element exceeds its own allowance.

    A plain maximum absolute error does not predict pass or fail, because the
    criterion is ``atol + rtol*|expected|`` and the tightest allowances sit
    where ``|expected|`` is smallest. Negative values mean the check passed with
    that much headroom.
    """
    expected64 = expected.astype(np.float64)
    difference = np.abs(expected64 - actual.astype(np.float64))
    allowance = tolerance["atol"] + tolerance["rtol"] * np.abs(expected64)
    return float(np.max(difference - allowance))


class Comparison:
    """Accumulate the worst error and the worst tolerance violation of a stage."""

    def __init__(self, tolerance: dict[str, float]) -> None:
        self.tolerance = tolerance
        self.max_absolute = 0.0
        self.max_violation = -float("inf")
        self.failures: list[str] = []

    def add(self, label: str, expected: np.ndarray, actual: np.ndarray) -> None:
        if expected.shape != actual.shape:
            self.failures.append(f"{label}: shape {actual.shape} != {expected.shape}")
            self.max_violation = float("inf")
            return
        if not np.isfinite(actual).all():
            self.failures.append(f"{label}: non-finite values")
            self.max_violation = float("inf")
            return
        self.max_absolute = max(self.max_absolute, max_error(expected, actual))
        violation = max_violation(expected, actual, self.tolerance)
        self.max_violation = max(self.max_violation, violation)
        if violation > 0:
            index = np.unravel_index(
                int(np.argmax(np.abs(expected - actual) - self.tolerance["atol"]
                              - self.tolerance["rtol"] * np.abs(expected))),
                expected.shape,
            )
            location = tuple(int(value) for value in index)
            self.failures.append(
                f"{label}: violation {violation:.6e} at {location}, "
                f"expected {expected[location]}, actual {actual[location]}"
            )

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_absolute_error": self.max_absolute,
            "max_tolerance_violation": (
                None if self.max_violation == -float("inf") else self.max_violation
            ),
            "failures": self.failures[:5],
        }


def age_public(probabilities: np.ndarray) -> tuple[int, int, float]:
    values = [float(value) for value in probabilities]
    expectation = age_expectation(values)
    return round_age(expectation), confidence_percent(max(values)), expectation


def gender_public(probabilities: np.ndarray) -> tuple[str, int]:
    return gender_result([float(value) for value in probabilities])


def validate_manifest(bundle: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("bundle_id") != "age-and-gender-v1":
        raise ValueError("unsupported conversion manifest")
    # The graph contract is the opset and IR version. The onnx library version is
    # recorded under tool_versions for provenance and is deliberately not pinned
    # here, so a checked-in bundle stays verifiable on a newer toolchain.
    if manifest["onnx"]["opset"] != 18 or manifest["onnx"]["ir_version"] != 10:
        raise ValueError("manifest declares an unexpected graph contract")
    runtime = manifest.get("runtime")
    expected_runtime = {
        "provider": "CPUExecutionProvider",
        "graph_optimization_level": "ORT_DISABLE_ALL",
        **THREAD_SETTINGS,
    }
    if runtime != expected_runtime:
        raise ValueError(f"manifest runtime contract must be {expected_runtime}, got {runtime}")
    for task in MODELS:
        artifact = manifest["models"][task]["artifact"]
        path = bundle / artifact["filename"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"{task} ONNX artifact does not match manifest")
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(inferred, full_check=True)
        if len(model.opset_import) != 1 or model.opset_import[0].version != 18:
            raise ValueError(f"{task} model has an unexpected opset")
        if model.ir_version != 10:
            raise ValueError(f"{task} model has an unexpected IR version")


def fixture_cases(
    task: str,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    cases = []
    for golden_path in sorted(FIXTURES.glob("*.golden.json")):
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        for face_index, face in enumerate(golden["faces"]):
            artifact = face["artifacts"][f"{task}_input"]
            tensor_path = FIXTURES / artifact["golden_path"]
            tensor = np.fromfile(tensor_path, dtype="<f4").reshape(artifact["shape"])
            cases.append(
                (
                    f"{golden_path.name}:face-{face_index}",
                    tensor,
                    np.asarray(face[f"{task}_logits"], dtype=np.float32)[None, :],
                    np.asarray(face[f"{task}_probabilities"], dtype=np.float32)[None, :],
                    face,
                )
            )
    return cases


def fixture_chips(task: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return distinct real chips with the tensors the oracle derived from them."""
    key = f"{task}_chip"
    chips: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for golden_path in sorted(FIXTURES.glob("*.golden.json")):
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        for face_index, face in enumerate(golden["faces"]):
            chip = face["artifacts"][key]
            if chip["sha256"] in chips:
                continue
            tensor = face["artifacts"][f"{task}_input"]
            chips[chip["sha256"]] = (
                f"{golden_path.name}:face-{face_index}",
                np.fromfile(FIXTURES / chip["golden_path"], dtype=np.uint8).reshape(chip["shape"]),
                np.fromfile(FIXTURES / tensor["golden_path"], dtype="<f4").reshape(tensor["shape"]),
            )
    return list(chips.values())


def validate_fixtures(task: str, session: ort.InferenceSession) -> dict[str, Any]:
    contract = MODELS[task]
    logits_check = Comparison(contract["logits_tolerance"])
    probability_check = Comparison(contract["probability_tolerance"])
    public_mismatches: list[str] = []
    expectation_error = 0.0
    cases = fixture_cases(task)
    for label, tensor, expected_logits, expected_probabilities, face in cases:
        probabilities, logits = session.run(["probabilities", "logits"], {"images": tensor})
        logits_check.add(f"{label}:logits", expected_logits, logits)
        probability_check.add(f"{label}:probabilities", expected_probabilities, probabilities)
        result = face["result"]
        if task == "age":
            age, confidence, expectation = age_public(probabilities[0])
            if age != result["age"]["value"] or confidence != result["age"]["confidence"]:
                public_mismatches.append(f"{label}: age {age}/{confidence}")
            # The oracle's own C++ expectation is the reference, not a Python
            # re-derivation of it from the same probabilities.
            difference = abs(expectation - float(face["age_expectation"]))
            expectation_error = max(expectation_error, difference)
            if difference > AGE_EXPECTATION_ATOL:
                public_mismatches.append(f"{label}: age expectation drift {difference:.6e}")
        else:
            value, confidence = gender_public(probabilities[0])
            if value != result["gender"]["value"] or confidence != result["gender"]["confidence"]:
                public_mismatches.append(f"{label}: gender {value}/{confidence}")
    summary = {
        "cases": len(cases),
        "distinct_chips": len(fixture_chips(task)),
        "logits": logits_check.summary(),
        "probabilities": probability_check.summary(),
        "public_mismatches": public_mismatches[:5],
        "public_passed": not public_mismatches,
    }
    if task == "age":
        summary["max_age_expectation_error"] = expectation_error
    return summary


def synthetic_images(size: int, count: int) -> np.ndarray:
    ramp = np.arange(size * size * 3, dtype=np.uint32).reshape(size, size, 3) % 256
    rng = np.random.default_rng(20260922)
    seeds = [
        np.zeros((size, size, 3), dtype=np.uint8),
        np.full((size, size, 3), 255, dtype=np.uint8),
        ramp.astype(np.uint8),
        rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8),
    ]
    while len(seeds) < count:
        seeds.append(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8))
    return np.stack(seeds[:count])


def normalize(images: np.ndarray) -> np.ndarray:
    nchw = images.transpose(0, 3, 1, 2).astype(np.float32)
    return np.asarray((nchw - MEANS.reshape(1, 3, 1, 1)) / np.float32(256.0), dtype=np.float32)


def run_probe(
    probe: Path,
    source_models: Path,
    work_dir: Path,
    task: str,
    images: np.ndarray,
    stages_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    contract = MODELS[task]
    count = images.shape[0]
    prefix = work_dir / f"{task}-batch-{count}-{hashlib.sha256(images.tobytes()).hexdigest()[:12]}"
    rgb_path = prefix.with_suffix(".rgb")
    logits_path = prefix.with_suffix(".logits.f32")
    probabilities_path = prefix.with_suffix(".probabilities.f32")
    images.tofile(rgb_path)
    command = [
        str(probe),
        "--task", task,
        "--model", str(source_models / contract["source"]),
        "--images", str(rgb_path),
        "--count", str(count),
        "--logits", str(logits_path),
        "--probabilities", str(probabilities_path),
    ]
    if stages_dir is not None:
        stages_dir.mkdir(parents=True, exist_ok=True)
        command.extend(("--stages-dir", str(stages_dir)))
    subprocess.run(command, check=True)
    shape = (count, contract["classes"])
    stages: dict[int, np.ndarray] = {}
    if stages_dir is not None:
        for entry in json.loads((stages_dir / "stages.json").read_text(encoding="utf-8")):
            stages[int(entry["layer"])] = np.fromfile(
                stages_dir / entry["file"], dtype="<f4"
            ).reshape(entry["shape"])
    return (
        np.fromfile(logits_path, dtype="<f4").reshape(shape),
        np.fromfile(probabilities_path, dtype="<f4").reshape(shape),
        stages,
    )


def validate_synthetic(
    task: str,
    session: ort.InferenceSession,
    probe: Path,
    source_models: Path,
    work_dir: Path,
) -> dict[str, Any]:
    contract = MODELS[task]
    logits_check = Comparison(contract["logits_tolerance"])
    probability_check = Comparison(contract["probability_tolerance"])
    batch_check = Comparison(contract["probability_tolerance"])
    public_mismatches: list[str] = []
    for count in BATCH_SIZES:
        images = synthetic_images(contract["size"], count)
        tensor = normalize(images)
        expected_logits, expected_probabilities, _ = run_probe(
            probe, source_models, work_dir, task, images
        )
        probabilities, logits = session.run(["probabilities", "logits"], {"images": tensor})
        label = f"synthetic:batch-{count}"
        logits_check.add(f"{label}:logits", expected_logits, logits)
        probability_check.add(f"{label}:probabilities", expected_probabilities, probabilities)
        individual = np.concatenate(
            [session.run(["probabilities"], {"images": sample[None, :]})[0] for sample in tensor]
        )
        batch_check.add(f"{label}:batch-vs-single", probabilities, individual)
        for index in range(count):
            if task == "age":
                batched = age_public(probabilities[index])[:2]
                single = age_public(individual[index])[:2]
                reference = age_public(expected_probabilities[index])[:2]
            else:
                batched = gender_public(probabilities[index])
                single = gender_public(individual[index])
                reference = gender_public(expected_probabilities[index])
            if batched != reference:
                public_mismatches.append(f"{label}:{index}: {batched} != oracle {reference}")
            if batched != single:
                public_mismatches.append(f"{label}:{index}: batch {batched} != single {single}")
    return {
        "batch_sizes": list(BATCH_SIZES),
        "logits": logits_check.summary(),
        "probabilities": probability_check.summary(),
        "batch_vs_single": batch_check.summary(),
        "public_mismatches": public_mismatches[:5],
        "public_passed": not public_mismatches,
    }


def validate_stages(
    task: str,
    bundle: Path,
    probe: Path,
    source_models: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Compare every exported dlib stage tensor with the converted graph.

    This localizes a conversion error to a block instead of only reporting that
    the final logits drifted, and it is what checks the residual branches.
    """
    contract = MODELS[task]
    chips = fixture_chips(task)
    if not chips:
        raise ValueError(f"no frozen {task} chips available for stage comparison")
    label = chips[0][0]
    images = np.stack([chip for _, chip, _ in chips])
    tensors = np.concatenate([tensor for _, _, tensor in chips])
    stages_dir = work_dir / f"{task}-stages"
    _, _, stage_tensors = run_probe(
        probe, source_models, work_dir, task, images, stages_dir=stages_dir
    )
    indices = tuple(sorted(stage_tensors))
    session = make_session(
        add_debug_outputs(bundle / f"{task}-v1.onnx", indices),
        OPTIMIZATION_LEVELS[SELECTED_SETTING],
    )
    model = onnx.load(bundle / f"{task}-v1.onnx", load_external_data=False)
    names = stage_tensor_names(model)
    check = Comparison(ACTIVATION_TOLERANCE)
    per_stage = []
    for index in indices:
        expected = stage_tensors[index]
        actual = session.run([names[index]], {"images": tensors})[0]
        actual = actual.reshape(expected.shape)
        # The final logits keep PR-1's logits threshold; the internal activations
        # are held to the measured float32 noise floor instead.
        is_logits = names[index] == "logits"
        tolerance = contract["logits_tolerance"] if is_logits else ACTIVATION_TOLERANCE
        stage_check = Comparison(tolerance)
        stage_check.add(f"{task}:layer-{index}", expected, actual)
        check.add(f"{task}:layer-{index}", expected, actual)
        per_stage.append(
            {
                "layer": index,
                "tensor": names[index],
                "shape": list(expected.shape),
                "tolerance": "logits" if is_logits else "activation",
                **stage_check.summary(),
            }
        )
    return {
        "chips": len(chips),
        "first_chip": label,
        "note": (
            "Stage tensors are pinned as extra graph outputs, which suppresses "
            "fusion, so stages are compared only at the selected setting."
        ),
        "activation_tolerance": ACTIVATION_TOLERANCE,
        "combined": check.summary(),
        "stages": per_stage,
    }


def evaluate_setting(
    name: str,
    level: ort.GraphOptimizationLevel,
    bundle: Path,
    probe: Path,
    source_models: Path,
    work_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"selected": name == SELECTED_SETTING, "models": {}}
    for task in MODELS:
        session = make_session(add_debug_outputs(bundle / f"{task}-v1.onnx"), level)
        fixtures = validate_fixtures(task, session)
        synthetic = validate_synthetic(task, session, probe, source_models, work_dir)
        result["models"][task] = {"fixtures": fixtures, "synthetic": synthetic}
    result["logits_passed"] = all(
        model[group]["logits"]["passed"]
        for model in result["models"].values()
        for group in ("fixtures", "synthetic")
    )
    result["probabilities_passed"] = all(
        model[group]["probabilities"]["passed"]
        for model in result["models"].values()
        for group in ("fixtures", "synthetic")
    )
    result["public_passed"] = all(
        model[group]["public_passed"]
        for model in result["models"].values()
        for group in ("fixtures", "synthetic")
    )
    result["passed"] = (
        result["logits_passed"] and result["probabilities_passed"] and result["public_passed"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source-models", type=Path, default=ROOT / "example/models")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    probe = args.probe.resolve()
    source_models = args.source_models.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(bundle, manifest)

    settings = {
        name: evaluate_setting(name, level, bundle, probe, source_models, work_dir)
        for name, level in OPTIMIZATION_LEVELS.items()
    }
    stages = {
        task: validate_stages(task, bundle, probe, source_models, work_dir)
        for task in MODELS
    }

    selected = settings[SELECTED_SETTING]
    stages_passed = all(entry["combined"]["passed"] for entry in stages.values())
    report: dict[str, Any] = {
        "schema_version": 2,
        "bundle_id": manifest["bundle_id"],
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "provider": "CPUExecutionProvider",
        "thread_settings": THREAD_SETTINGS,
        "selected_setting": SELECTED_SETTING,
        "tolerances": {
            task: {
                "logits": MODELS[task]["logits_tolerance"],
                "probabilities": MODELS[task]["probability_tolerance"],
                "age_expectation_atol": AGE_EXPECTATION_ATOL,
                "internal_activations": ACTIVATION_TOLERANCE,
            }
            for task in MODELS
        },
        "activation_tolerance_evidence": MEASURED_ACTIVATION_NOISE,
        "corpus": {
            "real_faces": len(fixture_cases("age")),
            "distinct_age_chips": len(fixture_chips("age")),
            "distinct_gender_chips": len(fixture_chips("gender")),
            "source_images": sorted(path.name for path in FIXTURES.glob("*.golden.json")),
            "synthetic_batch_sizes": list(BATCH_SIZES),
            "limitation": (
                "PR-1 defers the representative 30-image/100-face corpus to release "
                "hardening, so the real-chip evidence here is narrow. Synthetic chips "
                "and the live dlib probe carry the rest of the numerical coverage."
            ),
        },
        "settings": settings,
        "stages": stages,
        "stages_passed": stages_passed,
        "passed": bool(selected["passed"] and stages_passed),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        print(f"conversion validation FAILED; see {report_path}", file=sys.stderr)
        return 1
    rejected = sorted(name for name, entry in settings.items() if not entry["passed"])
    print(f"conversion validation passed at '{SELECTED_SETTING}'; rejected: {rejected or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
