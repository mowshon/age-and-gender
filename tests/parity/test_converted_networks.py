from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tools/conversion/artifacts/v1"

# `onnx` is maintainer conversion tooling rather than a runtime dependency of the
# published package, so this parity lane skips instead of failing where only the
# user-facing dependencies are installed.
try:
    import onnx
    import onnxruntime as ort  # noqa: F401 - availability is part of the guard
    from tools.conversion.validate_conversion import (
        ACTIVATION_TOLERANCE,
        MODELS,
        OPTIMIZATION_LEVELS,
        SELECTED_SETTING,
        SELECTED_THREAD_SETTING,
        THREAD_CANDIDATES,
        THREAD_SETTINGS,
        Comparison,
        add_debug_outputs,
        fixture_cases,
        fixture_chips,
        make_session,
        sha256_file,
        validate_fixtures,
        validate_manifest,
    )

    CONVERSION_TOOLING = True
except ImportError as error:  # pragma: no cover - environment dependent
    CONVERSION_TOOLING = False
    IMPORT_ERROR = error


@unittest.skipUnless(CONVERSION_TOOLING, "conversion tooling (onnx) is not installed")
class ConvertedNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        cls.sessions = {
            task: make_session(
                add_debug_outputs(BUNDLE / f"{task}-v1.onnx"),
                OPTIMIZATION_LEVELS[SELECTED_SETTING],
            )
            for task in MODELS
        }

    def test_manifest_artifacts_and_graph_contracts(self) -> None:
        self.assertEqual(self.manifest["bundle_id"], "age-and-gender-v1")
        self.assertEqual(self.manifest["onnx"]["opset"], 18)
        self.assertEqual(self.manifest["onnx"]["ir_version"], 10)
        for task, contract in MODELS.items():
            artifact = self.manifest["models"][task]["artifact"]
            path = BUNDLE / artifact["filename"]
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(sha256_file(path), artifact["sha256"])
            model = onnx.load(path, load_external_data=False)
            onnx.checker.check_model(model, full_check=True)
            self.assertEqual(model.graph.input[0].name, "images")
            self.assertEqual(model.graph.output[0].name, "probabilities")
            self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "N")
            self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "N")
            self.assertNotIn("BatchNormalization", {node.op_type for node in model.graph.node})
            self.assertEqual(
                model.graph.output[0].type.tensor_type.shape.dim[1].dim_value,
                contract["classes"],
            )

    def test_manifest_pins_the_session_options_parity_depends_on(self) -> None:
        """PR-3 builds sessions from this block, so it has to be in the bundle."""
        self.assertEqual(
            self.manifest["runtime"],
            {
                "provider": "CPUExecutionProvider",
                "graph_optimization_level": "ORT_DISABLE_ALL",
                **THREAD_SETTINGS,
            },
        )
        validate_manifest(BUNDLE, self.manifest)

    def test_frozen_fixture_parity(self) -> None:
        for task, session in self.sessions.items():
            result = validate_fixtures(task, session)
            self.assertEqual(result["cases"], 11)
            self.assertEqual(result["distinct_chips"], 8)
            self.assertTrue(result["logits"]["passed"], result["logits"]["failures"])
            self.assertTrue(result["probabilities"]["passed"], result["probabilities"]["failures"])
            self.assertTrue(result["public_passed"], result["public_mismatches"])

    def test_age_expectation_matches_the_oracle_value(self) -> None:
        result = validate_fixtures("age", self.sessions["age"])
        self.assertLess(result["max_age_expectation_error"], 1e-4)

    def test_dynamic_batches_match_frozen_outputs(self) -> None:
        for task, session in self.sessions.items():
            _, tensor, expected_logits, expected_probabilities, _ = fixture_cases(task)[0]
            for count in (1, 2, 7, 32):
                batch = np.repeat(tensor, count, axis=0)
                probabilities, logits = session.run(
                    ["probabilities", "logits"], {"images": batch}
                )
                logit_check = Comparison(MODELS[task]["logits_tolerance"])
                logit_check.add(
                    f"{task}:batch-{count}:logits",
                    np.repeat(expected_logits, count, axis=0),
                    logits,
                )
                self.assertTrue(logit_check.passed, logit_check.failures)
                probability_check = Comparison(MODELS[task]["probability_tolerance"])
                probability_check.add(
                    f"{task}:batch-{count}:probabilities",
                    np.repeat(expected_probabilities, count, axis=0),
                    probabilities,
                )
                self.assertTrue(probability_check.passed, probability_check.failures)

    def test_heterogeneous_batch_matches_individual_inference(self) -> None:
        """Every distinct real chip in one batch, against the same chips alone."""
        for task, session in self.sessions.items():
            tensors = np.concatenate([tensor for _, _, tensor in fixture_chips(task)])
            batched = session.run(["probabilities"], {"images": tensors})[0]
            individual = np.concatenate(
                [session.run(["probabilities"], {"images": row[None, :]})[0] for row in tensors]
            )
            check = Comparison(MODELS[task]["probability_tolerance"])
            check.add(f"{task}:batch-vs-single", batched, individual)
            self.assertTrue(check.passed, check.failures)

    def test_checked_conversion_report(self) -> None:
        report = json.loads(
            (BUNDLE / "conversion-report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["schema_version"], 3)
        self.assertTrue(report["passed"])
        self.assertTrue(report["stages_passed"])
        self.assertEqual(report["selected_setting"], SELECTED_SETTING)
        self.assertTrue(report["settings"][SELECTED_SETTING]["passed"])
        self.assertEqual(
            report["tolerances"]["age"]["internal_activations"], ACTIVATION_TOLERANCE
        )
        for task in MODELS:
            self.assertEqual(report["stages"][task]["chips"], 8)
            self.assertEqual(
                report["settings"][SELECTED_SETTING]["models"][task]["synthetic"][
                    "batch_sizes"
                ],
                [1, 2, 7, 32],
            )

    def test_checked_conversion_report_thread_investigation(self) -> None:
        """spec/PR-6.md's thread-count axis (mirrors the optimization-level one)."""
        report = json.loads(
            (BUNDLE / "conversion-report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["selected_thread_setting"], SELECTED_THREAD_SETTING)
        self.assertEqual(report["thread_candidates"], THREAD_CANDIDATES)
        self.assertEqual(set(report["thread_variants"]), set(THREAD_CANDIDATES))
        self.assertEqual(set(report["thread_stages"]), set(THREAD_CANDIDATES))
        for task in MODELS:
            self.assertEqual(
                report["thread_variants"][SELECTED_THREAD_SETTING]["thread_settings"],
                THREAD_SETTINGS,
            )
            self.assertEqual(
                report["thread_stages"][SELECTED_THREAD_SETTING][task]["thread_settings"],
                THREAD_SETTINGS,
            )
        # Shipping a higher default is still an explicit, separate decision
        # (see SUPPORTED_RUNTIME in _models.py); this only checks that the
        # investigation itself ran and recorded a real pass/fail per candidate.
        self.assertIsInstance(report["thread_investigation_passed"], bool)

    def test_report_records_measured_rejections_rather_than_prose(self) -> None:
        report = json.loads(
            (BUNDLE / "conversion-report.json").read_text(encoding="utf-8")
        )
        rejected = {
            name for name, entry in report["settings"].items() if not entry["passed"]
        }
        self.assertEqual(rejected, {"basic", "extended", "all"})
        for name in rejected:
            entry = report["settings"][name]
            # Every rejection must be backed by a recorded failing comparison, and
            # must be a logits-only failure: the shipped outputs still agree.
            self.assertFalse(entry["logits_passed"])
            self.assertTrue(entry["probabilities_passed"])
            self.assertTrue(entry["public_passed"])
            failures = [
                failure
                for task in MODELS
                for group in ("fixtures", "synthetic")
                for failure in entry["models"][task][group]["logits"]["failures"]
            ]
            self.assertTrue(failures, f"{name} was rejected without a recorded failure")


if __name__ == "__main__":
    unittest.main()
