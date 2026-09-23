"""Loaders for the PR-1 frozen oracle fixtures, without conversion tooling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures/legacy"


@dataclass(frozen=True)
class GoldenFace:
    """One frozen face: its chips, input tensors, probabilities, and result."""

    image: str
    index: int
    rectangle: list[int]
    landmarks: list[list[int]]
    gender_chip: np.ndarray
    age_chip: np.ndarray
    gender_input: np.ndarray
    age_input: np.ndarray
    gender_probabilities: np.ndarray
    age_probabilities: np.ndarray
    age_expectation: float
    result: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.image}:face-{self.index}"

    def chip(self, task: str) -> np.ndarray:
        return self.age_chip if task == "age" else self.gender_chip

    def input_tensor(self, task: str) -> np.ndarray:
        return self.age_input if task == "age" else self.gender_input

    def probabilities(self, task: str) -> np.ndarray:
        return self.age_probabilities if task == "age" else self.gender_probabilities


@dataclass(frozen=True)
class GoldenDocument:
    """One frozen oracle run: its source image, box mode, and faces."""

    name: str
    source_path: str
    source_sha256: str
    rgb_sha256: str
    size: tuple[int, int]
    box_mode: str
    input_boxes_trbl: list[list[int]]
    faces: list[GoldenFace]


def _array(face: dict[str, Any], key: str, dtype: str) -> np.ndarray:
    artifact = face["artifacts"][key]
    data = np.fromfile(FIXTURES / artifact["golden_path"], dtype=dtype)
    return data.reshape(artifact["shape"])


def _faces(document_name: str, golden: dict[str, Any]) -> list[GoldenFace]:
    return [
        GoldenFace(
            image=document_name,
            index=index,
            rectangle=list(face["rectangle"]),
            landmarks=[list(point) for point in face["landmarks"]],
            gender_chip=_array(face, "gender_chip", np.uint8),
            age_chip=_array(face, "age_chip", np.uint8),
            gender_input=_array(face, "gender_input", "<f4"),
            age_input=_array(face, "age_input", "<f4"),
            gender_probabilities=np.asarray(face["gender_probabilities"], dtype=np.float32),
            age_probabilities=np.asarray(face["age_probabilities"], dtype=np.float32),
            age_expectation=float(face["age_expectation"]),
            result=face["result"],
        )
        for index, face in enumerate(golden["faces"])
    ]


def golden_documents() -> list[GoldenDocument]:
    """Return every frozen oracle run, including its source image and box mode."""
    documents = []
    for path in sorted(FIXTURES.glob("*.golden.json")):
        golden = json.loads(path.read_text(encoding="utf-8"))
        documents.append(
            GoldenDocument(
                name=path.name,
                source_path=golden["input"]["source"],
                source_sha256=golden["input"]["source_sha256"],
                rgb_sha256=golden["input"]["rgb_sha256"],
                size=tuple(golden["input"]["size"]),
                box_mode=golden["box_mode"],
                input_boxes_trbl=[list(box) for box in golden["input_boxes_trbl"]],
                faces=_faces(path.name, golden),
            )
        )
    return documents


def golden_images() -> dict[str, list[GoldenFace]]:
    """Return every frozen image's faces, keyed by golden file name."""
    return {document.name: document.faces for document in golden_documents()}


def golden_faces() -> list[GoldenFace]:
    """Return every frozen face across all images, in file and face order."""
    return [face for faces in golden_images().values() for face in faces]
