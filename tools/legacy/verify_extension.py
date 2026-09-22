#!/usr/bin/env python3
"""Compare the original extension's public output with an oracle report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from age_and_gender import AgeAndGender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--box", action="append", default=[])
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    input_metadata = report["input"]
    raw_path = args.report.parent / "input.rgb"
    if not raw_path.exists():
        raw_path = Path(input_metadata["path"])
    raw = np.fromfile(raw_path, dtype=np.uint8).reshape(
        input_metadata["height"], input_metadata["width"], 3
    )
    report_boxes = report.get("input_boxes_trbl", [])
    boxes = [[int(value) for value in box.split(",")] for box in args.box]
    if boxes and boxes != report_boxes:
        raise SystemExit("command-line boxes differ from the oracle report")
    boxes = boxes or report_boxes

    predictor = AgeAndGender()
    predictor.load_shape_predictor(str(args.models / "shape_predictor_5_face_landmarks.dat"))
    predictor.load_dnn_gender_classifier(str(args.models / "dnn_gender_classifier_v1.dat"))
    predictor.load_dnn_age_predictor(str(args.models / "dnn_age_predictor_v1.dat"))
    actual = predictor.predict(raw, boxes)
    expected = report["results"]
    if actual != expected:
        raise SystemExit(
            "original extension and standalone oracle differ:\n"
            f"expected: {expected!r}\nactual:   {actual!r}"
        )
    print(f"extension parity passed for {len(actual)} face(s)")


if __name__ == "__main__":
    main()
