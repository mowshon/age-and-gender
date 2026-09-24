#!/usr/bin/env python3
"""Verify a binary-only ``age-and-gender`` install offline, from the checkout.

Run this against an *installed* package (a wheel built by ``python -m
build``, installed with ``pip install --only-binary=:all:``), not against
this repository's ``src/`` tree — the point is to prove the installed
artifact works, with no source checkout on ``sys.path`` ahead of it. This
repository is only used as a source of known-good fixtures (example images,
the frozen compatibility corpus, and the ``.dat`` shape predictor, which is
byte-identical to the original dlib-models download).

Network access is not required; run this under a network namespace or
firewall to confirm that, per spec/PR-7.md's "Prove compiler-free
installation" gate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def check_bundled_prediction(predictor) -> None:
    with Image.open(REPO_ROOT / "example/test-image.jpg") as image:
        results = predictor.predict(image.convert("RGB"))
    if not results:
        raise SystemExit("bundled-model prediction found no faces on the known example image")
    print(f"bundled prediction: {len(results)} face(s), first = {results[0]!r}")


def check_shape_predictor_dat_loading(predictor) -> None:
    dat_path = REPO_ROOT / "src/age_and_gender/models/shape_predictor_5_face_landmarks.dat"
    predictor.load_shape_predictor(str(dat_path))
    with Image.open(REPO_ROOT / "example/test-image.jpg") as image:
        results = predictor.predict(image.convert("RGB"))
    if not results:
        raise SystemExit("prediction after loading the .dat shape predictor found no faces")
    print(f"shape predictor .dat loading: {len(results)} face(s)")


def check_explicit_box_prediction(predictor) -> None:
    with Image.open(REPO_ROOT / "example/test-image.jpg") as image:
        array = np.asarray(image.convert("RGB"))
    results = predictor.predict(array, [(266, 506, 352, 419)])
    if len(results) != 1:
        raise SystemExit(f"explicit-box prediction returned {len(results)} face(s), expected 1")
    print(f"explicit-box prediction: {results[0]!r}")


def check_neural_dat_rejected(predictor) -> None:
    # Rejection happens before the file is read, so a placeholder stands in
    # for the original weights.
    with tempfile.TemporaryDirectory() as directory:
        dat_path = Path(directory) / "dnn_age_predictor_v1.dat"
        dat_path.write_bytes(b"dlib serialized network")
        try:
            predictor.load_dnn_age_predictor(str(dat_path))
        except ValueError:
            print("neural .dat loader correctly rejected the legacy dlib weight file")
            return
    raise SystemExit("load_dnn_age_predictor accepted a legacy .dat file; it must not")


def check_compatibility_corpus(predictor) -> int:
    from age_and_gender._images import as_rgb_array

    manifest = json.loads((REPO_ROOT / "tests/fixtures/legacy/manifest.json").read_text())
    checked = 0
    for case in manifest["inputs"]:
        path = REPO_ROOT / case["path"]
        if not path.is_file():
            print(f"skip (not present in this checkout): {case['path']}")
            continue
        with Image.open(path) as image:
            array = as_rgb_array(image.convert("RGB"))
        actual = predictor.predict(array)
        expected = [face["result"] for face in case["expected"]]
        if actual != expected:
            raise SystemExit(f"{case['path']}: expected {expected!r}, got {actual!r}")
        checked += 1
    for case in manifest["explicit_box_cases"]:
        path = REPO_ROOT / case["input"]
        with Image.open(path) as image:
            array = as_rgb_array(image.convert("RGB"))
        actual = predictor.predict(array, case["boxes_trbl"])
        expected = [face["result"] for face in case["expected"]]
        if actual != expected:
            raise SystemExit(f"{case['name']}: expected {expected!r}, got {actual!r}")
        checked += 1
    print(f"compatibility corpus: {checked} document(s) matched exactly")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-corpus",
        type=int,
        default=1,
        help="Fail if fewer than this many compatibility-corpus documents were checked.",
    )
    args = parser.parse_args()

    from age_and_gender import AgeAndGender

    predictor = AgeAndGender()
    check_bundled_prediction(predictor)
    check_explicit_box_prediction(predictor)
    check_neural_dat_rejected(predictor)
    checked = check_compatibility_corpus(predictor)
    if checked < args.require_corpus:
        raise SystemExit(
            f"only {checked} compatibility-corpus document(s) were available; "
            f"expected at least {args.require_corpus}"
        )
    # Mutates the shared predictor; keep it last so every prior check runs
    # against the untouched bundled models.
    check_shape_predictor_dat_loading(predictor)
    print("all offline smoke checks passed")


if __name__ == "__main__":
    main()
