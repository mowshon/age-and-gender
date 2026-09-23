"""Build, install, and run the wheel outside the checkout.

Source-tree tests cannot show that the model resources are actually packaged, or
that resolving them is independent of the working directory, so this module
builds the distributions, installs the wheel into a separate environment, and
runs chip inference from an unrelated directory with no index available.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import unittest
import zipfile
from collections.abc import Callable
from pathlib import Path

from tests.golden import golden_images

ROOT = Path(__file__).resolve().parents[2]
NATIVE_SUFFIXES = (".so", ".pyd", ".dylib", ".dll", ".a", ".lib")
BUILD_TIMEOUT = 900
MODEL_RESOURCES = (
    "models/manifest.json",
    "models/age-v1.onnx",
    "models/gender-v1.onnx",
    "models/shape_predictor_5_face_landmarks.dat",
)
NOTICES = (
    "models/notices/Cydral-age-gender-models.md",
    "models/notices/dlib-shape-predictor-5-face-landmarks.md",
)

CHILD_PROGRAM = """
import json, sys
from pathlib import Path

import numpy as np

import age_and_gender
from age_and_gender._inference import InferenceEngine
from age_and_gender._models import bundled_models
from age_and_gender._postprocess import face_predictions

request = json.loads(Path(sys.argv[1]).read_text())
bundle = bundled_models()
engine = InferenceEngine(bundle)

def chips(task):
    return [
        np.fromfile(face[task + "_chip"], dtype=np.uint8).reshape(face[task + "_shape"])
        for face in request["faces"]
    ]

results = face_predictions(
    [face["rectangle"] for face in request["faces"]],
    engine.gender.probabilities(chips("gender")),
    engine.age.probabilities(chips("age")),
    labels=bundle.gender.labels,
    age_weights=bundle.age.age_weights,
)
print(json.dumps({
    "version": age_and_gender.__version__,
    "package_file": age_and_gender.__file__,
    "bundle_origin": bundle.origin,
    "sys_path": [entry for entry in sys.path if entry],
    "cwd": str(Path.cwd()),
    "results": results,
    "empty": engine.age.probabilities([]).shape,
}))
"""

# Unlike CHILD_PROGRAM above, this drives the real detector/landmark/alignment
# path (age_and_gender._faces, backed by dlib) and Pillow decoding, starting
# from a raw image rather than pre-extracted chips. CHILD_PROGRAM alone cannot
# show the binary-only installed dlib-bin/Pillow wheels actually work: it never
# imports either.
FRONTEND_CHILD_PROGRAM = """
import json, sys
from pathlib import Path

from PIL import Image

import age_and_gender
from age_and_gender._faces import FaceFrontend
from age_and_gender._images import as_rgb_array
from age_and_gender._inference import InferenceEngine
from age_and_gender._models import bundled_models
from age_and_gender._postprocess import face_predictions

image_path = Path(sys.argv[1])
bundle = bundled_models()
frontend = FaceFrontend(bundle)
engine = InferenceEngine(bundle)

with Image.open(image_path) as image:
    array = as_rgb_array(image.convert("RGB"))
extractions = frontend.extract(array)

results = face_predictions(
    [extraction.rectangle for extraction in extractions],
    engine.gender.probabilities([extraction.gender_chip for extraction in extractions]),
    engine.age.probabilities([extraction.age_chip for extraction in extractions]),
    labels=bundle.gender.labels,
    age_weights=bundle.age.age_weights,
)
print(json.dumps({
    "version": age_and_gender.__version__,
    "results": results,
}))
"""


# Exercises the public AgeAndGender class exactly as spec/PR-5.md's target
# surface shows it: zero-configuration construction, the three loader methods
# (the two neural ones against an explicit .onnx bundle, since the original
# .dat weights are not loadable directly), and both predict() call styles.
# CHILD_PROGRAM and FRONTEND_CHILD_PROGRAM above only reach the internal
# modules directly; this is the only check in the suite that imports
# age_and_gender.AgeAndGender from a binary-only install.
API_CHILD_PROGRAM = """
import json, sys
from pathlib import Path

from PIL import Image

from age_and_gender import AgeAndGender

image_path = Path(sys.argv[1])
bundle_dir = Path(sys.argv[2])

zero_config = AgeAndGender()
with Image.open(image_path) as image:
    zero_config_results = zero_config.predict(image.convert("RGB"))

explicit = AgeAndGender()
explicit.load_shape_predictor(bundle_dir / "shape_predictor_5_face_landmarks.dat")
explicit.load_dnn_gender_classifier(bundle_dir / "gender-v1.onnx")
explicit.load_dnn_age_predictor(bundle_dir / "age-v1.onnx")
with Image.open(image_path) as image:
    positional_results = explicit.predict(image.convert("RGB"))
with Image.open(image_path) as image:
    keyword_results = explicit.predict(photo_numpy_array=image.convert("RGB"))

print(json.dumps({
    "zero_config_results": zero_config_results,
    "positional_results": positional_results,
    "keyword_results": keyword_results,
}))
"""


# spec/PR-7.md: "Model loading must not write into site-packages." Zero
# configuration, one prediction, nothing else: any write attempt into the
# read-only package directory this program is run against would surface as an
# OSError/PermissionError here rather than as a passing, silently-ignored call.
READ_ONLY_CHILD_PROGRAM = """
import json, sys
from pathlib import Path

from PIL import Image

from age_and_gender import AgeAndGender

image_path = Path(sys.argv[1])
predictor = AgeAndGender()
with Image.open(image_path) as image:
    results = predictor.predict(image.convert("RGB"))
print(json.dumps({"results": results}))
"""


def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
        **kwargs,  # type: ignore[arg-type]
    )


@unittest.skipUnless(
    (ROOT / "pyproject.toml").is_file(), "distribution tests need the project source"
)
class PackagedDistributionTests(unittest.TestCase):
    """One build and one installation shared by every check in this module."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import build  # noqa: F401
        except ImportError as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"the build frontend is not installed: {error}") from error
        cls._workspace = tempfile.TemporaryDirectory()
        cls.workspace = Path(cls._workspace.name)
        cls.dist = cls.workspace / "dist"
        # --no-isolation keeps the build offline: hatchling comes from this
        # environment instead of being downloaded into a throwaway one.
        _run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(cls.dist), str(ROOT)],
            cwd=str(cls.workspace),
        )
        wheels = sorted(cls.dist.glob("*.whl"))
        archives = sorted(cls.dist.glob("*.tar.gz"))
        assert len(wheels) == 1 and len(archives) == 1, (wheels, archives)
        cls.wheel = wheels[0]
        cls.sdist = archives[0]
        cls.wheel_names = zipfile.ZipFile(cls.wheel).namelist()
        with tarfile.open(cls.sdist) as archive:
            cls.sdist_names = archive.getnames()
        cls.environment = cls._install(cls.workspace / "env", cls.wheel)
        cls.output = cls._infer(cls.environment)
        cls.frontend_output = cls._infer_frontend(cls.environment)
        cls.api_output = cls._infer_api(cls.environment)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._workspace.cleanup()

    @classmethod
    def _install(cls, prefix: Path, wheel: Path) -> Path:
        """Create an environment outside the checkout and install the wheel offline."""
        _run([sys.executable, "-m", "venv", str(prefix)])
        python = prefix / ("Scripts" if os.name == "nt" else "bin") / "python"
        site = Path(
            _run(
                [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"]
            ).stdout.strip()
        )
        # NumPy and ONNX Runtime are added from this project's environment
        # instead of being downloaded, so the whole check stays offline. The
        # path is appended after the new environment's own site-packages, so the
        # installed wheel is what provides age_and_gender.
        (site / "_project_dependencies.pth").write_text(
            f"{sysconfig.get_paths()['purelib']}\n", encoding="utf-8"
        )
        _run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)])
        return python

    @classmethod
    def _run_isolated(cls, python: Path, sandbox: Path, program: Path, *args: str) -> dict:
        """Run `program` with no compiler, no CMake, and no index reachable."""
        empty_path = cls.workspace / "empty-path"
        empty_path.mkdir(exist_ok=True)
        environment = {
            "HOME": str(cls.workspace),
            "PATH": str(empty_path),
            "PYTHONNOUSERSITE": "1",
            "PIP_NO_INDEX": "1",
        }
        if "SYSTEMROOT" in os.environ:  # pragma: no cover - Windows only
            environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        completed = _run(
            [str(python), str(program), *args],
            cwd=str(sandbox),
            env=environment,
        )
        return json.loads(completed.stdout)

    @classmethod
    def _infer(cls, python: Path) -> dict:
        """Run chip inference from an unrelated directory with a minimal environment."""
        sandbox = cls.workspace / "elsewhere"
        sandbox.mkdir()
        # The chips are copied out of the checkout so the child reads nothing
        # from the source tree except the wheel it has installed.
        faces = golden_images()["test-image.golden.json"]
        request: dict[str, list[dict]] = {"faces": []}
        for face in faces:
            entry = {"rectangle": face.rectangle}
            for task in ("age", "gender"):
                chip = face.chip(task)
                path = sandbox / f"face-{face.index}-{task}.rgb"
                path.write_bytes(chip.tobytes())
                entry[f"{task}_chip"] = str(path)
                entry[f"{task}_shape"] = list(chip.shape)
            request["faces"].append(entry)
        request_path = sandbox / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        program = sandbox / "run_inference.py"
        program.write_text(CHILD_PROGRAM, encoding="utf-8")
        return cls._run_isolated(python, sandbox, program, str(request_path))

    @classmethod
    def _infer_frontend(cls, python: Path) -> dict:
        """Run detection, landmark alignment, and inference from a raw image.

        Unlike `_infer`, this starts from JPEG bytes rather than pre-extracted
        chips, so it is the only check that actually exercises the installed
        dlib-bin/Pillow wheels from the binary-only install.
        """
        sandbox = cls.workspace / "elsewhere-frontend"
        sandbox.mkdir()
        # The image bytes are copied out of the checkout so the child reads
        # nothing from the source tree except the wheel it has installed.
        image_path = sandbox / "test-image.jpg"
        image_path.write_bytes((ROOT / "example/test-image.jpg").read_bytes())
        program = sandbox / "run_frontend.py"
        program.write_text(FRONTEND_CHILD_PROGRAM, encoding="utf-8")
        return cls._run_isolated(python, sandbox, program, str(image_path))

    @classmethod
    def _infer_api(cls, python: Path) -> dict:
        """Run the public AgeAndGender class, including the explicit loaders.

        Copies the image and, as an explicit bundle directory, the package's
        own installed models (``src/age_and_gender/models/``, not
        ``example/models/``: the latter's legacy ``.dat`` duplicates are
        excluded from the sdist, but the former is the package's own data and
        always ships) out of the checkout, so the child reads nothing from the
        source tree except the wheel it has installed.
        """
        sandbox = cls.workspace / "elsewhere-api"
        sandbox.mkdir()
        image_path = sandbox / "test-image.jpg"
        image_path.write_bytes((ROOT / "example/test-image.jpg").read_bytes())
        bundle_dir = sandbox / "bundle"
        bundle_dir.mkdir()
        for name in (
            "manifest.json",
            "age-v1.onnx",
            "gender-v1.onnx",
            "shape_predictor_5_face_landmarks.dat",
        ):
            (bundle_dir / name).write_bytes(
                (ROOT / "src/age_and_gender/models" / name).read_bytes()
            )
        program = sandbox / "run_api.py"
        program.write_text(API_CHILD_PROGRAM, encoding="utf-8")
        return cls._run_isolated(python, sandbox, program, str(image_path), str(bundle_dir))

    def test_wheel_is_pure_python_and_has_no_native_extension(self) -> None:
        self.assertTrue(self.wheel.name.endswith("-py3-none-any.whl"), self.wheel.name)
        native = [name for name in self.wheel_names if name.lower().endswith(NATIVE_SUFFIXES)]
        self.assertEqual(native, [])
        self.assertNotIn("age_and_gender.py", self.wheel_names)
        top_level = {name.split("/")[0] for name in self.wheel_names}
        self.assertEqual(
            top_level, {"age_and_gender", f"age_and_gender-{self.output['version']}.dist-info"}
        )

    def test_wheel_carries_the_model_resources(self) -> None:
        for name in (*MODEL_RESOURCES, *NOTICES, "py.typed"):
            self.assertIn(f"age_and_gender/{name}", self.wheel_names)

    def test_wheel_resources_match_their_recorded_digests(self) -> None:
        with zipfile.ZipFile(self.wheel) as archive:
            self._check_digests(
                lambda name: archive.read(f"age_and_gender/{name}"),
            )

    def test_source_distribution_carries_the_model_resources(self) -> None:
        stem = self.sdist.name[: -len(".tar.gz")]
        for name in (*MODEL_RESOURCES, *NOTICES):
            self.assertIn(f"{stem}/src/age_and_gender/{name}", self.sdist_names)
        self.assertIn(f"{stem}/pyproject.toml", self.sdist_names)
        self.assertFalse(
            [name for name in self.sdist_names if "/libs/" in name or name.endswith("setup.py")],
            "the sdist must not ship the historical native build",
        )

    def test_source_distribution_carries_what_test_frontend_needs(self) -> None:
        """tests/parity/test_frontend.py decodes these images to exercise the
        real detector/landmark path; a sdist without them can't run the tests
        it ships (github.com/mowshon/age-and-gender parity gap, fixed here).
        """
        stem = self.sdist.name[: -len(".tar.gz")]
        for name in ("example/test-image.jpg", "example/test-image-2.jpg"):
            self.assertIn(f"{stem}/{name}", self.sdist_names)
        # example/models/ duplicates the legacy .dat files already bundled as
        # ONNX/manifest resources; it must not be pulled in just to reach the
        # two images above.
        self.assertFalse(
            [name for name in self.sdist_names if "/example/models/" in name],
            "the sdist must not ship the legacy .dat duplicates under example/models/",
        )

    def test_source_distribution_resources_match_their_recorded_digests(self) -> None:
        stem = self.sdist.name[: -len(".tar.gz")]
        with tarfile.open(self.sdist) as archive:

            def read(name: str) -> bytes:
                member = archive.extractfile(f"{stem}/src/age_and_gender/{name}")
                assert member is not None, name
                return member.read()

            self._check_digests(read)

    def _check_digests(self, read: Callable[[str], bytes]) -> None:
        """Verify the shipped artifacts and notices against the shipped manifest."""
        manifest = json.loads(read("models/manifest.json"))
        recorded = {
            manifest["models"]["age"]["artifact"]["filename"]: manifest["models"]["age"][
                "artifact"
            ],
            manifest["models"]["gender"]["artifact"]["filename"]: manifest["models"]["gender"][
                "artifact"
            ],
            manifest["shape_predictor"]["artifact"]["filename"]: manifest["shape_predictor"][
                "artifact"
            ],
            manifest["license"]["notice"]["path"]: manifest["license"]["notice"],
            manifest["shape_predictor"]["license"]["notice"]["path"]: manifest["shape_predictor"][
                "license"
            ]["notice"],
        }
        self.assertEqual(len(recorded), 5)
        for name, entry in recorded.items():
            payload = read(f"models/{name}")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"], name)
            if "bytes" in entry:
                self.assertEqual(len(payload), entry["bytes"], name)

    def test_runtime_dependencies_stay_minimal(self) -> None:
        stem = self.wheel.name.split("-py3-none-any")[0]
        with zipfile.ZipFile(self.wheel) as archive:
            metadata = archive.read(f"{stem}.dist-info/METADATA").decode("utf-8")
        required = [
            line.split(":", 1)[1].strip()
            for line in metadata.splitlines()
            if line.startswith("Requires-Dist:") and "extra ==" not in line
        ]
        self.assertTrue(any(name.startswith("numpy") for name in required), required)
        self.assertTrue(any(name.startswith("onnxruntime") for name in required), required)
        excluded_prefixes = (
            "onnx ",
            "onnx=",
            "torch",
            "caffe",
            "face-recognition",
            "face_recognition",
        )
        for excluded in excluded_prefixes:
            self.assertFalse(
                [name for name in required if name.lower().startswith(excluded)], required
            )

    def test_a_wheel_can_be_built_from_the_source_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            extracted = Path(directory)
            with tarfile.open(self.sdist) as archive:
                archive.extractall(extracted, filter="data")
            source = extracted / self.sdist.name[: -len(".tar.gz")]
            _run(
                [sys.executable, "-m", "build", "--wheel", "--no-isolation", str(source)],
                cwd=str(extracted),
            )
            rebuilt = sorted((source / "dist").glob("*.whl"))
            self.assertEqual(len(rebuilt), 1)
            self.assertEqual(rebuilt[0].name, self.wheel.name)
            self.assertEqual(
                sorted(zipfile.ZipFile(rebuilt[0]).namelist()), sorted(self.wheel_names)
            )

    def test_installed_package_runs_from_an_unrelated_directory(self) -> None:
        package_file = Path(self.output["package_file"])
        self.assertNotIn(str(ROOT), str(package_file))
        self.assertIn("site-packages", str(package_file))
        self.assertNotIn(str(ROOT), self.output["cwd"])

    def test_no_checkout_source_is_on_the_installed_runtime_path(self) -> None:
        """Neither the package sources, tools/, nor the CMake build tree is reachable.

        The project environment's site-packages is under the checkout and is
        deliberately allowed: it is how this test supplies NumPy and ONNX
        Runtime without an index.
        """
        dependencies = str(Path(sysconfig.get_paths()["purelib"]).resolve())
        forbidden = [str(ROOT), *(str(ROOT / name) for name in ("src", "tools", "build"))]
        for entry in self.output["sys_path"]:
            resolved = str(Path(entry).resolve())
            if resolved == dependencies or resolved.startswith(f"{dependencies}{os.sep}"):
                continue
            for prefix in forbidden:
                self.assertFalse(
                    resolved == prefix or resolved.startswith(f"{prefix}{os.sep}"),
                    f"{entry} points back into {prefix}",
                )

    def test_bundled_models_resolve_through_the_installed_package(self) -> None:
        self.assertEqual(self.output["bundle_origin"], "age_and_gender.models")
        self.assertEqual(self.output["empty"], [0, 81])

    def test_offline_inference_reproduces_the_frozen_results(self) -> None:
        expected = [face.result for face in golden_images()["test-image.golden.json"]]
        self.assertEqual(self.output["results"], expected)

    def test_offline_frontend_reproduces_the_frozen_results_from_a_raw_image(self) -> None:
        """The binary-only installed dlib-bin and Pillow wheels actually work:
        detection, landmark prediction, and individual chip alignment all ran
        in the child process, starting from JPEG bytes, not pre-extracted chips.
        """
        expected = [face.result for face in golden_images()["test-image.golden.json"]]
        self.assertEqual(self.frontend_output["results"], expected)

    def test_public_api_reproduces_the_frozen_results_from_the_installed_wheel(self) -> None:
        """spec/PR-5.md's "built-wheel integration using real models" check:
        the public AgeAndGender class, not the internal modules, driven end to
        end from a binary-only install.
        """
        expected = [face.result for face in golden_images()["test-image.golden.json"]]
        self.assertEqual(self.api_output["zero_config_results"], expected)

    def test_explicit_loaders_reproduce_the_frozen_results_from_the_installed_wheel(self) -> None:
        """The three loader calls, against an explicit bundle directory built
        from the installed wheel's own models, still reproduce the exact
        results, positionally and by keyword.
        """
        expected = [face.result for face in golden_images()["test-image.golden.json"]]
        self.assertEqual(self.api_output["positional_results"], expected)
        self.assertEqual(self.api_output["keyword_results"], expected)

    @unittest.skipIf(
        os.name == "nt",
        "chmod does not model a read-only directory on Windows the way it does on POSIX",
    )
    def test_predicts_with_a_read_only_installed_package_directory(self) -> None:
        """spec/PR-7.md: "Model loading must not write into site-packages."

        Installs into its own environment, separate from the class-shared one
        every other test in this module reads, so making it read-only cannot
        affect any other test's fixtures or their cleanup.
        """
        with tempfile.TemporaryDirectory() as workspace_name:
            workspace = Path(workspace_name)
            python = self._install(workspace / "env", self.wheel)
            locate = "import age_and_gender, os; print(os.path.dirname(age_and_gender.__file__))"
            package_dir = Path(_run([str(python), "-c", locate]).stdout.strip())
            self.assertTrue(package_dir.is_dir())
            try:
                self._chmod_tree_read_only(package_dir)
                sandbox = workspace / "elsewhere-read-only"
                sandbox.mkdir()
                image_path = sandbox / "test-image.jpg"
                image_path.write_bytes((ROOT / "example/test-image.jpg").read_bytes())
                program = sandbox / "run_read_only.py"
                program.write_text(READ_ONLY_CHILD_PROGRAM, encoding="utf-8")
                output = self._run_isolated(python, sandbox, program, str(image_path))
            finally:
                # Restore write permissions so TemporaryDirectory cleanup, and
                # this environment's own removal above, can delete the files.
                self._chmod_tree_writable(package_dir)
            expected = [face.result for face in golden_images()["test-image.golden.json"]]
            self.assertEqual(output["results"], expected)

    @staticmethod
    def _chmod_tree_read_only(root: Path) -> None:
        for path in [root, *root.rglob("*")]:
            path.chmod(0o555 if path.is_dir() else 0o444)

    @staticmethod
    def _chmod_tree_writable(root: Path) -> None:
        for path in [root, *root.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)


if __name__ == "__main__":
    unittest.main()
