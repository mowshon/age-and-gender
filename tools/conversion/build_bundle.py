#!/usr/bin/env python3
"""Assemble the installed model bundle from converted artifacts and source models.

The conversion bundle in ``tools/conversion/artifacts/v1`` covers only the two
converted neural networks. The package additionally ships the original
five-point landmark model, which is loaded directly by dlib and is therefore
copied rather than converted. This script joins the two into the directory that
``src/age_and_gender/models`` installs, and writes the package manifest the
runtime validates against.

It is maintainer tooling: nothing in the installed package runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERSION_BUNDLE = ROOT / "tools/conversion/artifacts/v1"
DEFAULT_SHAPE_PREDICTOR = ROOT / "example/models/shape_predictor_5_face_landmarks.dat"
DEFAULT_OUTPUT = ROOT / "src/age_and_gender/models"

# The exact five-point predictor PR-1 froze its landmark fixtures against. A
# different file with the same name is a different model, so the bytes are
# pinned rather than the filename.
SHAPE_PREDICTOR_SHA256 = "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888"
SHAPE_PREDICTOR_BYTES = 9150489
SHAPE_PREDICTOR_PARTS = 5
SHAPE_PREDICTOR_NOTICE = "notices/dlib-shape-predictor-5-face-landmarks.md"
SHAPE_PREDICTOR_NOTICE_TEXT = """# Five-point face landmark model notice

`shape_predictor_5_face_landmarks.dat` is redistributed unchanged from the
`davisking/dlib-models` project, where it is dedicated to the public domain
under CC0 1.0.

Source: <https://github.com/davisking/dlib-models>

License: <https://creativecommons.org/publicdomain/zero/1.0/>
"""

TASKS = ("age", "gender")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_conversion_bundle(bundle: Path) -> dict[str, Any]:
    """Return the conversion manifest after re-verifying every artifact it names."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("bundle_id") != "age-and-gender-v1":
        raise SystemExit(f"{bundle} does not hold a supported conversion manifest")
    if "shape_predictor" in manifest:
        raise SystemExit(f"{bundle} already looks like a package bundle")
    for task in TASKS:
        artifact = manifest["models"][task]["artifact"]
        path = bundle / artifact["filename"]
        if path.stat().st_size != artifact["bytes"] or sha256_file(path) != artifact["sha256"]:
            raise SystemExit(f"{path} does not match the conversion manifest")
    notice = manifest["license"]["notice"]
    notice_path = bundle / notice["path"]
    if sha256_file(notice_path) != notice["sha256"]:
        raise SystemExit(f"{notice_path} does not match the conversion manifest")
    return manifest


def build(conversion_bundle: Path, shape_predictor: Path, output: Path) -> dict[str, Any]:
    """Write the package bundle and return the manifest that was written."""
    manifest = check_conversion_bundle(conversion_bundle)

    digest = sha256_file(shape_predictor)
    if digest != SHAPE_PREDICTOR_SHA256:
        raise SystemExit(
            f"{shape_predictor} has sha256 {digest}, expected {SHAPE_PREDICTOR_SHA256}"
        )

    # The predictor keeps its own basename, so a file that happens to be named
    # like one of the graphs would overwrite it and leave a manifest whose
    # hashes no longer describe the packaged files.
    predictor_name = shape_predictor.name
    taken = {manifest["models"][task]["artifact"]["filename"] for task in TASKS}
    taken.add("manifest.json")
    if predictor_name in taken:
        raise SystemExit(
            f"{shape_predictor} would be written as {predictor_name}, which the "
            "bundle already uses; rename the file before packaging it"
        )

    if output.exists():
        shutil.rmtree(output)
    (output / "notices").mkdir(parents=True)

    for task in TASKS:
        filename = manifest["models"][task]["artifact"]["filename"]
        shutil.copyfile(conversion_bundle / filename, output / filename)
    conversion_notice = manifest["license"]["notice"]["path"]
    shutil.copyfile(conversion_bundle / conversion_notice, output / conversion_notice)

    shutil.copyfile(shape_predictor, output / predictor_name)
    (output / SHAPE_PREDICTOR_NOTICE).write_text(SHAPE_PREDICTOR_NOTICE_TEXT, encoding="utf-8")

    manifest["bundle_kind"] = "package"
    manifest["shape_predictor"] = {
        "artifact": {
            "filename": predictor_name,
            "sha256": digest,
            "bytes": shape_predictor.stat().st_size,
        },
        "format": "dlib-shape-predictor",
        "license": {
            "attribution": "Davis E. King",
            "notice": {
                "path": SHAPE_PREDICTOR_NOTICE,
                "sha256": hashlib.sha256(SHAPE_PREDICTOR_NOTICE_TEXT.encode("utf-8")).hexdigest(),
            },
            "source": "https://github.com/davisking/dlib-models",
            "spdx": "CC0-1.0",
        },
        "parts": SHAPE_PREDICTOR_PARTS,
        # The landmark model is loaded by dlib as-is, so the shipped artifact and
        # the legacy source file the user may pass to load_shape_predictor() are
        # the same bytes.
        "source": {
            "filename": predictor_name,
            "sha256": digest,
            "bytes": shape_predictor.stat().st_size,
        },
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "manifest.json").write_text(serialized, encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-bundle", type=Path, default=DEFAULT_CONVERSION_BUNDLE)
    parser.add_argument("--shape-predictor", type=Path, default=DEFAULT_SHAPE_PREDICTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    build(arguments.conversion_bundle, arguments.shape_predictor, arguments.output)
    print(f"wrote package bundle to {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
