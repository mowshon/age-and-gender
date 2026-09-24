"""Pure NumPy port of dlib's ``get_face_chip_details`` + ``extract_image_chip``.

Maintainer-only feasibility prototype for spec/PR-8.md; not part of the
shipped runtime. Ported from
``tools/vendor/dlib/dlib/image_transforms/interpolation.h`` and
``tools/vendor/dlib/dlib/image_transforms/image_pyramid.h``'s
``pyramid_down<2>`` (= ``pyramid_down_2_1``), verified line-by-line against
that source. Notable, easy-to-miss details:

- The extraction rectangle is a *chip-sized box scaled by the similarity
  transform's scale factor*, centered on where the transform maps the chip's
  own center — not a tight box around the landmarks
  (``chip_details``'s point-correspondence constructor, interpolation.h:1608-
  1637).
- When the chip is much smaller than its source rectangle, dlib pre-
  downsamples through an image pyramid before bilinear sampling, rather than
  bilinear-sampling the full-resolution crop directly, "since at a high
  enough downsampling amount [bilinear] would effectively turn into nearest
  neighbor interpolation" (interpolation.h:1795-1799). The pyramid step
  itself is a real 5-tap ``[1,4,6,4,1]`` separable filter followed by 2:1
  decimation with an odd, off-center formula (``point_down(p) = p/2 -
  (1.25, 0.75)``, image_pyramid.h:159), not a naive box average.
- Only ONE chip is ever extracted per call in this codebase
  (``_faces.py``'s per-face-per-size individual extraction), which lets this
  port skip building and storing the full pyramid array
  ``extract_image_chips`` keeps for a batch: it applies the downsample
  operator exactly as many times as dlib's per-chip level selection asks for
  and keeps only that level (see :func:`_build_source_image`).
- All transform geometry (similarity/affine fits, rotations, rect
  arithmetic) uses :mod:`dlib_linalg`'s scalar ports of dlib's own routines,
  not NumPy/LAPACK equivalents, so that ``chip_details`` and the sampling map
  are bit-identical to dlib's; see that module's docstring for why.
- Bilinear-interpolated channel values are assigned into the uint8 output via
  ``static_cast<unsigned char>``, which **truncates** the fractional part
  (matrix_utilities.h's ``vector_to_pixel_helper<P,2>`` for rgb pixels) —
  it does not round to nearest. Out-of-bounds samples (no edge clamping) are
  replaced with black (0), not clamped to the nearest valid pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from dlib_linalg import find_affine_transform, find_similarity_transform, rotate_point

__all__ = ["ChipDetails", "extract_image_chip", "get_face_chip_details", "pyramid_depth"]

# Canonical 5-point template, normalized [0,1] face space, before padding
# (interpolation.h:1964-1968). Order matches the bundled 5-point predictor's
# landmark order (part 0..4).
_CANONICAL_POINTS = np.array(
    [
        [0.8595674595992, 0.2134981538014],
        [0.6460604764104, 0.2289674387677],
        [0.1205750620789, 0.2137274526848],
        [0.3340850613712, 0.2290642403242],
        [0.4901123135679, 0.6277975316475],
    ]
)


@dataclass(frozen=True, slots=True)
class ChipDetails:
    """Mirrors dlib's ``chip_details`` public fields (rect/angle/rows/cols)."""

    rect: tuple[float, float, float, float]  # (left, top, right, bottom), inclusive, double
    angle: float
    rows: int
    cols: int

    def size(self) -> int:
        return self.rows * self.cols


def get_face_chip_details(landmarks: np.ndarray, size: int, padding: float = 0.2) -> ChipDetails:
    """Port of ``get_face_chip_details`` for a 5-point detection (interpolation.h:1942-2034).

    Args:
        landmarks: ``(5, 2)`` array of ``[x, y]`` landmark coordinates (the
            same order the bundled 5-point predictor returns).
        size: Output chip size (chip is always square: ``size x size``).
        padding: Face-chip padding, matching dlib's default parameter.

    Returns:
        The chip's extraction geometry.
    """
    if landmarks.shape != (5, 2):
        raise ValueError(f"expected 5 landmark points, got shape {landmarks.shape}")
    # dlib: DLIB_CASSERT(padding >= 0 && size > 0). Without this, size 0
    # reaches extract_image_chip's pyramid loop, which never terminates.
    if isinstance(size, bool) or int(size) != size or size <= 0:
        raise ValueError(f"chip size must be a positive integer, got {size!r}")
    if not padding >= 0:
        raise ValueError(f"padding must be >= 0, got {padding!r}")
    size = int(size)
    # dlib: `p = (padding+p)/(2*padding+1); from_points.push_back(p*size)`.
    # `padding + p` is a scalar-plus-matrix expression, so the division that
    # follows is dlib's matrix/scalar, i.e. a multiply by the reciprocal.
    reciprocal = 1.0 / (2 * padding + 1)
    chip_points = [
        ((padding + x) * reciprocal * size, (padding + y) * reciprocal * size)
        for x, y in _CANONICAL_POINTS.tolist()
    ]
    img_points = [(float(x), float(y)) for x, y in landmarks.tolist()]

    # chip_details' point-correspondence constructor (interpolation.h:1608-1637),
    # evaluated with dlib's own scalar arithmetic (see dlib_linalg).
    m, b = find_similarity_transform(chip_points, img_points)
    # p = get_m() * (1, 0), elementwise exactly as dlib's matrix-vector product.
    p_x = m[0][0] * 1.0 + m[0][1] * 0.0
    p_y = m[1][0] * 1.0 + m[1][1] * 0.0
    angle = math.atan2(p_y, p_x)
    scale = math.sqrt(p_x * p_x + p_y * p_y)  # dlib::vector::length, not hypot

    half = size / 2.0
    center_x = (m[0][0] * half + m[0][1] * half) + b[0]
    center_y = (m[1][0] * half + m[1][1] * half) + b[1]
    # centered_drect(center, size*scale, size*scale) (drectangle.h:389-399)
    width = size * scale - 1
    height = size * scale - 1
    rect = (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )
    return ChipDetails(rect=rect, angle=angle, rows=size, cols=size)


def _rect_is_empty(rect: tuple[float, float, float, float]) -> bool:
    """``drectangle::is_empty`` (drectangle.h:119-120)."""
    left, top, right, bottom = rect
    return top > bottom or left > right


def _rect_width(rect: tuple[float, float, float, float]) -> float:
    return 0.0 if _rect_is_empty(rect) else rect[2] - rect[0] + 1


def _rect_height(rect: tuple[float, float, float, float]) -> float:
    return 0.0 if _rect_is_empty(rect) else rect[3] - rect[1] + 1


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    """``drectangle::area`` (drectangle.h:95-117): zero exactly when ``is_empty()``."""
    return _rect_width(rect) * _rect_height(rect)


def _rect_down(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """``pyramid_down_2_1::rect_down`` via ``point_down`` (image_pyramid.h:154-160)."""
    left, top, right, bottom = rect
    return (left / 2.0 - 1.25, top / 2.0 - 0.75, right / 2.0 - 1.25, bottom / 2.0 - 0.75)


def _rect_center(rect: tuple[float, float, float, float]) -> tuple[float, float]:
    """``center(drectangle)`` (drectangle.h:299-307)."""
    return (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2


def _round_rect_half_up(rect: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """``drectangle::operator rectangle`` (drectangle.h:64-71)."""
    left, top, right, bottom = rect
    return (
        math.floor(left + 0.5),
        math.floor(top + 0.5),
        math.floor(right + 0.5),
        math.floor(bottom + 0.5),
    )


def _pyramid_down_rgb(image: np.ndarray) -> np.ndarray:
    """Port of ``pyramid_down_2_1::operator()`` for RGB images (image_pyramid.h:340-465).

    A 5x5 Gaussian ([1,4,6,4,1] separable), applied with the row filter first
    (producing an intermediate at half horizontal resolution) then the column
    filter (half vertical resolution), all in integer arithmetic, then a
    truncating divide by 256. Returns an empty (0, 0, 3) array when either
    input dimension is <= 8, matching dlib's own degenerate-input behavior.
    """
    rows, cols = image.shape[:2]
    if rows <= 8 or cols <= 8:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    values = image.astype(np.uint32)
    temp_cols = (cols - 3) // 2
    # Row filter: for output column c, taps at input columns 2c..2c+4.
    row_out = np.empty((rows, temp_cols, 3), dtype=np.uint32)
    for c in range(temp_cols):
        oc = 2 * c
        row_out[:, c, :] = (
            values[:, oc, :]
            + 4 * values[:, oc + 1, :]
            + 6 * values[:, oc + 2, :]
            + 4 * values[:, oc + 3, :]
            + values[:, oc + 4, :]
        )

    temp_rows = row_out.shape[0]
    down_rows = (temp_rows - 3) // 2
    down = np.empty((down_rows, temp_cols, 3), dtype=np.uint8)
    dr = 0
    r = 2
    while r < temp_rows - 2:
        combined = (
            row_out[r - 2, :, :].astype(np.uint32)
            + 4 * row_out[r - 1, :, :]
            + 6 * row_out[r, :, :]
            + 4 * row_out[r + 1, :, :]
            + row_out[r + 2, :, :]
        )
        down[dr, :, :] = (combined // 256).astype(np.uint8)
        dr += 1
        r += 2
    return down


def _build_source_image(
    image: np.ndarray, bounding_box: tuple[int, int, int, int], depth: int
) -> np.ndarray:
    """Crops to ``bounding_box`` then applies ``depth`` pyramid descents.

    Only the selected level is needed: with exactly one chip per call, every
    other pyramid level ``extract_image_chips`` builds is unused (see module
    docstring).
    """
    left, top, right, bottom = bounding_box
    height, width = image.shape[:2]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, width - 1)
    bottom = min(bottom, height - 1)
    current = image[top : bottom + 1, left : right + 1, :]
    for _ in range(depth):
        current = _pyramid_down_rgb(current)
    return current


def pyramid_depth(details: ChipDetails) -> tuple[int, float]:
    """Pyramid depth and crop border ``grow`` (interpolation.h:1801-1823).

    One unconditional ``rect_down``, then further descents while the rect's
    area still exceeds the chip size.
    """
    depth = 0
    grow = 2.0
    descended = _rect_down(details.rect)
    while _rect_area(descended) > details.size():
        descended = _rect_down(descended)
        depth += 1
        grow = grow * 2 + 2
    return depth, grow


def extract_image_chip(image: np.ndarray, details: ChipDetails) -> np.ndarray:
    """Port of ``extract_image_chip``/``extract_image_chips`` (interpolation.h:1761-1938).

    Args:
        image: ``[H, W, 3]`` uint8 RGB array.
        details: As returned by :func:`get_face_chip_details`.

    Returns:
        ``[details.rows, details.cols, 3]`` uint8 RGB chip.

    Raises:
        ValueError: ``details`` has a zero-sized chip or an empty rectangle
            (dlib's own ``extract_image_chips`` precondition).
    """
    height, width = image.shape[:2]
    rows, cols = details.rows, details.cols
    if details.size() == 0 or _rect_is_empty(details.rect):
        raise ValueError(f"invalid chip details: {rows}x{cols} chip of rect {details.rect}")

    # Fast-path bypass: no rotation/scaling requested at all
    # (interpolation.h:1910-1914). dlib compares the integer chip dimensions
    # against the *floating-point* drectangle height/width, and only then
    # converts the rect to an integer rectangle for the raw copy.
    if (
        details.angle == 0
        and rows == _rect_height(details.rect)
        and cols == _rect_width(details.rect)
    ):
        return _basic_extract_image_chip(image, _round_rect_half_up(details.rect))

    corners = (
        (details.rect[0], details.rect[1]),
        (details.rect[2], details.rect[1]),
        (details.rect[0], details.rect[3]),
        (details.rect[2], details.rect[3]),
    )

    depth, grow = pyramid_depth(details)

    center = _rect_center(details.rect)
    rotated = [rotate_point(center, corner, details.angle) for corner in corners]
    rot_rect = (
        min(p[0] for p in rotated),
        min(p[1] for p in rotated),
        max(p[0] for p in rotated),
        max(p[1] for p in rotated),
    )
    grown = (rot_rect[0] - grow, rot_rect[1] - grow, rot_rect[2] + grow, rot_rect[3] + grow)
    clipped = (
        max(grown[0], 0.0),
        max(grown[1], 0.0),
        min(grown[2], float(width - 1)),
        min(grown[3], float(height - 1)),
    )
    # `rectangle bounding_box; bounding_box += drect`: an empty drect leaves
    # the default empty rectangle (0, 0, -1, -1) in place.
    if _rect_is_empty(clipped):
        bounding_box = (0, 0, -1, -1)
    else:
        bounding_box = _round_rect_half_up(clipped)
        if bounding_box[0] > bounding_box[2] or bounding_box[1] > bounding_box[3]:
            bounding_box = (0, 0, -1, -1)

    # Per-chip level selection (interpolation.h:1852-1868), evaluated on the
    # bounding-box-local rect exactly as dlib does rather than assumed equal
    # to `depth - 1`: translation can move a rect_down area across the
    # threshold by a rounding step.
    local_rect = (
        details.rect[0] - bounding_box[0],
        details.rect[1] - bounding_box[1],
        details.rect[2] - bounding_box[0],
        details.rect[3] - bounding_box[1],
    )
    level = -1
    while _rect_area(_rect_down(local_rect)) > details.size():
        level += 1
        local_rect = _rect_down(local_rect)
    if level >= depth:
        raise AssertionError(f"pyramid level {level} exceeds built depth {depth}")

    source_image = _build_source_image(image, bounding_box, level + 1)

    local_center = _rect_center(local_rect)
    to_points = [
        rotate_point(local_center, (local_rect[0], local_rect[1]), details.angle),
        rotate_point(local_center, (local_rect[2], local_rect[1]), details.angle),
        rotate_point(local_center, (local_rect[0], local_rect[3]), details.angle),
    ]
    from_points = [(0.0, 0.0), (cols - 1.0, 0.0), (0.0, rows - 1.0)]
    m, b = find_affine_transform(from_points, to_points)

    return _sample_bilinear(source_image, m, b, rows, cols)


def _basic_extract_image_chip(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """Port of ``impl::basic_extract_image_chip`` (interpolation.h:1718-1756).

    The output takes the integer rectangle's own dimensions, as dlib's
    ``vchip.set_size(location.height(), location.width())`` does.
    """
    left, top, right, bottom = rect
    height, width = image.shape[:2]
    rows, cols = max(bottom - top + 1, 0), max(right - left + 1, 0)
    chip = np.zeros((rows, cols, 3), dtype=np.uint8)
    area_left, area_top = max(left, 0), max(top, 0)
    area_right, area_bottom = min(right, width - 1), min(bottom, height - 1)
    if area_left > area_right or area_top > area_bottom:
        return chip
    chip_left, chip_top = area_left - left, area_top - top
    chip_right, chip_bottom = area_right - left, area_bottom - top
    chip[chip_top : chip_bottom + 1, chip_left : chip_right + 1, :] = image[
        area_top : area_bottom + 1, area_left : area_right + 1, :
    ]
    return chip


def _sample_bilinear(
    source: np.ndarray, m: list[list[float]], b: list[float], rows: int, cols: int
) -> np.ndarray:
    """Port of ``transform_image`` + ``interpolate_bilinear`` (interpolation.h:229-266, 392-423)."""
    chip = np.zeros((rows, cols, 3), dtype=np.uint8)
    if source.shape[0] == 0 or source.shape[1] == 0:
        return chip
    src_h, src_w = source.shape[:2]

    cc, rr = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
    # Explicit elementwise form rather than a batched (rows*cols, 2) @ (2, 2)
    # matmul: dlib calls point_transform_affine::operator() once per output
    # pixel (transform_image, interpolation.h:415-422), each a standalone
    # 2x2-matrix-times-vector product. Routing the same math through a single
    # large NumPy/BLAS matmul is mathematically equivalent but not guaranteed
    # bit-identical (observed divergence, see numpy_shape_predictor's
    # _extract_feature_pixel_values docstring); this keeps every pixel's
    # transform an independent 2-term sum, matching dlib's own evaluation.
    px = m[0][0] * cc + m[0][1] * rr + b[0]
    py = m[1][0] * cc + m[1][1] * rr + b[1]
    left = np.floor(px).astype(np.int64)
    top = np.floor(py).astype(np.int64)
    right = left + 1
    bottom = top + 1
    valid = (left >= 0) & (top >= 0) & (right < src_w) & (bottom < src_h)

    left_c = np.clip(left, 0, src_w - 1)
    top_c = np.clip(top, 0, src_h - 1)
    right_c = np.clip(right, 0, src_w - 1)
    bottom_c = np.clip(bottom, 0, src_h - 1)

    lr_frac = (px - left)[..., None]
    tb_frac = (py - top)[..., None]

    source_f = source.astype(np.float64)
    tl = source_f[top_c, left_c]
    tr = source_f[top_c, right_c]
    bl = source_f[bottom_c, left_c]
    br = source_f[bottom_c, right_c]

    interpolated = (1 - tb_frac) * ((1 - lr_frac) * tl + lr_frac * tr) + tb_frac * (
        (1 - lr_frac) * bl + lr_frac * br
    )
    # static_cast<unsigned char> truncates toward zero; values are always
    # non-negative convex combinations of uint8 inputs, so this is floor().
    truncated = np.trunc(interpolated).astype(np.uint8)
    chip[valid] = truncated[valid]
    return chip
