"""Pure NumPy port of ``dlib::shape_predictor::operator()``.

Maintainer-only feasibility prototype for spec/PR-8.md; not part of the
shipped runtime (``src/age_and_gender`` still uses ``dlib-bin`` for this).

Ported line-for-line from ``tools/vendor/dlib/dlib/image_processing/
shape_predictor.h``, verified against that source rather than the abstract
docs, because several details are easy to get numerically wrong:

- The per-cascade feature-pixel transform maps a trained relative offset
  (``deltas``) through the *linear* part only of a similarity transform
  between the initial and current shape estimate
  (``find_tform_between_shapes(initial, current).get_m()``,
  shape_predictor.h:264), then through an *affine* map from the unit square
  to the face rectangle's own inclusive corners
  (``unnormalizing_tform``, shape_predictor.h:219-233) — not a single
  combined transform.
- The feature "pixel value" is grayscale intensity computed as
  ``(r + g + b) // 3`` with integer truncating division
  (dlib/pixel.h's ``assign_pixel_helpers::assign`` grayscale-from-rgb
  overload), not a luminance-weighted formula.
- Final landmark coordinates round **half away from zero, per axis**
  (``floor(x + 0.5)``, matching dlib's floating-to-integral
  ``vector_assign_helper`` specialization), not Python/NumPy's
  round-half-to-even.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dlib_linalg import find_affine_transform, find_similarity_transform

__all__ = ["ArtifactError", "ShapePredictor", "load_shape_predictor_artifact"]


@dataclass(frozen=True, slots=True)
class _Cascade:
    anchor_idx: np.ndarray  # (num_features,) uint32
    deltas: np.ndarray  # (num_features, 2) float32
    tree_splits_idx1: list[np.ndarray]  # per tree: (num_splits,) uint32
    tree_splits_idx2: list[np.ndarray]
    tree_splits_thresh: list[np.ndarray]  # per tree: (num_splits,) float32
    tree_leaves: list[np.ndarray]  # per tree: (num_leaves, 2*num_parts) float32


class ShapePredictor:
    """A loaded, ready-to-run port of one dlib 5-point ``shape_predictor``."""

    def __init__(self, initial_shape: np.ndarray, cascades: list[_Cascade]) -> None:
        self._initial_shape = initial_shape  # (num_parts, 2) float32
        self._cascades = cascades
        self.num_parts = initial_shape.shape[0]

    def __call__(self, image: np.ndarray, rectangle: tuple[int, int, int, int]) -> np.ndarray:
        """Predict landmarks for one face.

        Args:
            image: ``[H, W, 3]`` uint8 RGB array.
            rectangle: Inclusive ``(left, top, right, bottom)``.

        Returns:
            ``(num_parts, 2)`` int64 array of ``[x, y]`` landmark coordinates.
        """
        left, top, right, bottom = rectangle
        # dlib's current_shape is a matrix<float,0,1>: kept in float32 for the
        # whole cascade, never widened, so this must match bit for bit rather
        # than accumulate in float64 (a real, measured 3-unit intensity
        # divergence was observed on this repo's own corpus before this fix).
        current_shape = self._initial_shape.astype(np.float32).copy()
        intensity = _grayscale_intensity(image)
        for cascade in self._cascades:
            features = _extract_feature_pixel_values(
                intensity, left, top, right, bottom, current_shape, self._initial_shape, cascade
            )
            for idx1, idx2, thresh, leaves in zip(
                cascade.tree_splits_idx1,
                cascade.tree_splits_idx2,
                cascade.tree_splits_thresh,
                cascade.tree_leaves,
                strict=True,
            ):
                leaf = _evaluate_tree(features, idx1, idx2, thresh, leaves)
                current_shape = (current_shape + leaf.reshape(self.num_parts, 2)).astype(np.float32)

        return _unnormalize(current_shape, left, top, right, bottom)


def _grayscale_intensity(image: np.ndarray) -> np.ndarray:
    """``(r + g + b) // 3`` with integer truncating division, per pixel."""
    channels = image.astype(np.uint32)
    total = channels[..., 0] + channels[..., 1] + channels[..., 2]
    return (total // 3).astype(np.float32)


def _find_tform_between_shapes_linear(from_shape: np.ndarray, to_shape: np.ndarray) -> np.ndarray:
    """``matrix_cast<float>(find_tform_between_shapes(from, to).get_m())``.

    Port of shape_predictor.h:175-197 plus the cast at line 264.

    The Umeyama fit runs in double precision (dlib's ``dlib::vector<double,2>``
    accumulators widen the float32 shape points) using dlib's own scalar
    SVD (:mod:`dlib_linalg`), then the linear part is cast to float32 since it
    is immediately multiplied against float32 ``deltas``.
    """
    if from_shape.shape[0] == 1:
        return np.eye(2, dtype=np.float32)
    m, _ = find_similarity_transform(
        [(float(x), float(y)) for x, y in from_shape.tolist()],
        [(float(x), float(y)) for x, y in to_shape.tolist()],
    )
    return np.array(m, dtype=np.float64).astype(np.float32)


def _unnormalizing_affine(
    left: int, top: int, right: int, bottom: int
) -> tuple[list[list[float]], list[float]]:
    """``unnormalizing_tform(rect)`` (shape_predictor.h:219-233): returns ``(m, b)``.

    Maps ``(0,0)->tl``, ``(1,0)->tr``, ``(1,1)->br`` through dlib's general
    ``find_affine_transform`` (``Q * pinv(P)`` via its own SVD), not the
    closed form ``m = diag(width, height)``: dlib's pinv solve leaves
    last-bit residue (including tiny nonzero off-diagonal terms), and a
    closed form that is "more exact" than dlib disagrees with it whenever a
    feature point lands on a ``.5`` rounding boundary. The corners pass
    through ``vector<float,2>`` in dlib, hence the float32 round trip.
    """

    def as_float32_point(x: int, y: int) -> tuple[float, float]:
        return float(np.float32(x)), float(np.float32(y))

    to_points = [
        as_float32_point(left, top),
        as_float32_point(right, top),
        as_float32_point(right, bottom),
    ]
    from_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    return find_affine_transform(from_points, to_points)


def _extract_feature_pixel_values(
    intensity: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    current_shape: np.ndarray,
    initial_shape: np.ndarray,
    cascade: _Cascade,
) -> np.ndarray:
    """Port of ``impl::extract_feature_pixel_values`` (shape_predictor.h:237-281)."""
    tform = _find_tform_between_shapes_linear(initial_shape, current_shape)  # float32
    # point_transform_affine is always double precision in dlib.
    m, b = _unnormalizing_affine(left, top, right, bottom)

    anchors = current_shape[cascade.anchor_idx]  # (num_features, 2) float32
    # dlib computes tform*deltas[i] as an independent 2x2-matrix-times-vector
    # product for each feature, in a plain sequential loop (shape_predictor.h:
    # 271-279) - two float32 multiplies and one add per output component, no
    # reduction over more than 2 terms. A batched (2,2)@(2,N) NumPy matmul
    # here is mathematically equivalent but NOT guaranteed bit-identical: it
    # is dispatched to BLAS, which may use fused multiply-add or different
    # instruction scheduling than a plain scalar loop. That one-ULP-scale
    # difference was observed in practice landing exactly on a feature whose
    # projected coordinate sat within ~1e-6 of a pixel's ``x.5`` rounding
    # boundary, flipping which pixel got sampled and cascading into a
    # completely different final landmark. Explicit elementwise arithmetic
    # (never routed through BLAS) avoids that ambiguity.
    dx, dy = cascade.deltas[:, 0], cascade.deltas[:, 1]
    proj_x = tform[0, 0] * dx + tform[0, 1] * dy
    proj_y = tform[1, 0] * dx + tform[1, 1] * dy
    # (num_features, 2) float32, normalized space
    projected = np.stack([proj_x, proj_y], axis=1) + anchors
    # point_transform_affine::operator(): (m*p) + b in double, elementwise.
    proj64_x, proj64_y = projected[:, 0].astype(np.float64), projected[:, 1].astype(np.float64)
    image_x = m[0][0] * proj64_x + m[0][1] * proj64_y + b[0]
    image_y = m[1][0] * proj64_x + m[1][1] * proj64_y + b[1]
    points = np.stack([image_x, image_y], axis=1)  # image space, double

    # dlib assigns a computed dlib::vector<float,2> into a point (long,2),
    # which rounds half away from zero per axis (see module docstring).
    px = np.floor(points[:, 0] + 0.5).astype(np.int64)
    py = np.floor(points[:, 1] + 0.5).astype(np.int64)

    height, width = intensity.shape[:2]
    in_bounds = (px >= 0) & (py >= 0) & (px < width) & (py < height)
    values = np.zeros(points.shape[0], dtype=np.float32)
    clipped_x = np.clip(px, 0, width - 1)
    clipped_y = np.clip(py, 0, height - 1)
    sampled = intensity[clipped_y, clipped_x]
    values[in_bounds] = sampled[in_bounds]
    return values


def _evaluate_tree(
    features: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    thresh: np.ndarray,
    leaves: np.ndarray,
) -> np.ndarray:
    """Port of ``impl::regression_tree::operator()`` (shape_predictor.h:63-89).

    The split test subtracts and compares in float32
    (``(float)feature_pixel_values[idx1] - (float)feature_pixel_values[idx2] >
    splits[i].thresh``): ``features``/``thresh`` are kept float32 by the
    caller, so plain NumPy scalar arithmetic here stays float32 too, rather
    than widening through a Python ``float()`` cast.
    """
    num_splits = idx1.shape[0]
    if num_splits == 0:
        return leaves[0]
    node = 0
    while node < num_splits:
        go_left = bool(features[idx1[node]] - features[idx2[node]] > thresh[node])
        node = 2 * node + 1 if go_left else 2 * node + 2
    return leaves[node - num_splits]


def _unnormalize(
    current_shape: np.ndarray, left: int, top: int, right: int, bottom: int
) -> np.ndarray:
    m, b = _unnormalizing_affine(left, top, right, bottom)
    shape64_x = current_shape[:, 0].astype(np.float64)
    shape64_y = current_shape[:, 1].astype(np.float64)
    image_x = m[0][0] * shape64_x + m[0][1] * shape64_y + b[0]
    image_y = m[1][0] * shape64_x + m[1][1] * shape64_y + b[1]
    points = np.stack([image_x, image_y], axis=1)
    x = np.floor(points[:, 0] + 0.5).astype(np.int64)
    y = np.floor(points[:, 1] + 0.5).astype(np.int64)
    return np.stack([x, y], axis=1)


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_BUNDLE_ID = "age-and-gender-shape-predictor-v1"
_ARRAY_DTYPES = {
    "initial_shape": np.dtype("<f4"),
    "anchor_idx": np.dtype("<u4"),
    "deltas": np.dtype("<f4"),
    "splits_idx1": np.dtype("<u4"),
    "splits_idx2": np.dtype("<u4"),
    "splits_thresh": np.dtype("<f4"),
    "leaves": np.dtype("<f4"),
}


class ArtifactError(ValueError):
    """The artifact directory is not a valid, intact shape-predictor bundle."""


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def load_shape_predictor_artifact(artifact_dir: str | Path) -> ShapePredictor:
    """Load and fully validate a bundle produced by :mod:`build_shape_predictor`.

    Checks the manifest schema/bundle id, the ``.npz`` SHA-256 and byte size
    recorded in the manifest, every array's dtype and shape, that the
    declared cascade/tree layout consumes each flat array exactly, and that
    every stored index is in range. Anything else raises
    :class:`ArtifactError` rather than silently truncating or misreading.
    """
    import json

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
        f"unsupported schema_version {manifest.get('schema_version')!r}",
    )
    _require(
        manifest.get("bundle_id") == ARTIFACT_BUNDLE_ID,
        f"unexpected bundle_id {manifest.get('bundle_id')!r}",
    )
    artifact = manifest["artifact"]
    npz_path = artifact_dir / artifact["filename"]
    _require(npz_path.parent == artifact_dir, "artifact filename must not leave the bundle")
    _require(
        npz_path.stat().st_size == artifact["bytes"],
        f"{npz_path.name}: size differs from manifest",
    )
    _require(_sha256_file(npz_path) == artifact["sha256"], f"{npz_path.name}: SHA-256 mismatch")

    num_parts = manifest["num_parts"]
    num_cascades = manifest["num_cascades"]
    cascade_num_features: list[int] = manifest["cascade_num_features"]
    cascade_tree_splits: list[list[int]] = manifest["cascade_tree_splits"]
    _require(isinstance(num_parts, int) and num_parts > 0, "num_parts must be a positive int")
    _require(
        len(cascade_num_features) == num_cascades and len(cascade_tree_splits) == num_cascades,
        "cascade_num_features/cascade_tree_splits do not match num_cascades",
    )

    with np.load(npz_path, allow_pickle=False) as data:
        _require(set(data.files) == set(_ARRAY_DTYPES), f"unexpected arrays {sorted(data.files)}")
        arrays = {name: data[name] for name in _ARRAY_DTYPES}
    for name, dtype in _ARRAY_DTYPES.items():
        _require(arrays[name].dtype == dtype, f"{name}: dtype {arrays[name].dtype}, want {dtype}")

    total_features = sum(cascade_num_features)
    total_splits = sum(sum(tree_splits) for tree_splits in cascade_tree_splits)
    total_leaves = sum(len(tree_splits) + sum(tree_splits) for tree_splits in cascade_tree_splits)
    expected_shapes = {
        "initial_shape": (num_parts, 2),
        "anchor_idx": (total_features,),
        "deltas": (total_features, 2),
        "splits_idx1": (total_splits,),
        "splits_idx2": (total_splits,),
        "splits_thresh": (total_splits,),
        "leaves": (total_leaves, 2 * num_parts),
    }
    for name, shape in expected_shapes.items():
        _require(arrays[name].shape == shape, f"{name}: shape {arrays[name].shape}, want {shape}")
    _require(bool((arrays["anchor_idx"] < num_parts).all()), "anchor_idx references a missing part")

    initial_shape = arrays["initial_shape"]
    cascades: list[_Cascade] = []
    feature_offset = 0
    split_offset = 0
    leaf_offset = 0
    for num_features, tree_splits in zip(cascade_num_features, cascade_tree_splits, strict=True):
        cascade_anchor_idx = arrays["anchor_idx"][feature_offset : feature_offset + num_features]
        cascade_deltas = arrays["deltas"][feature_offset : feature_offset + num_features]
        feature_offset += num_features

        tree_idx1: list[np.ndarray] = []
        tree_idx2: list[np.ndarray] = []
        tree_thresh: list[np.ndarray] = []
        tree_leaves: list[np.ndarray] = []
        for num_splits in tree_splits:
            idx1 = arrays["splits_idx1"][split_offset : split_offset + num_splits]
            idx2 = arrays["splits_idx2"][split_offset : split_offset + num_splits]
            _require(
                bool((idx1 < num_features).all() and (idx2 < num_features).all()),
                "split feature index out of range for its cascade",
            )
            tree_idx1.append(idx1)
            tree_idx2.append(idx2)
            tree_thresh.append(arrays["splits_thresh"][split_offset : split_offset + num_splits])
            split_offset += num_splits

            num_leaves = num_splits + 1
            tree_leaves.append(arrays["leaves"][leaf_offset : leaf_offset + num_leaves])
            leaf_offset += num_leaves

        cascades.append(
            _Cascade(
                anchor_idx=cascade_anchor_idx,
                deltas=cascade_deltas,
                tree_splits_idx1=tree_idx1,
                tree_splits_idx2=tree_idx2,
                tree_splits_thresh=tree_thresh,
                tree_leaves=tree_leaves,
            )
        )
    _require(
        (feature_offset, split_offset, leaf_offset) == (total_features, total_splits, total_leaves),
        "cascade layout does not consume the stored arrays exactly",
    )

    return ShapePredictor(initial_shape, cascades)
