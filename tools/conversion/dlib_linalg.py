"""Scalar ports of the dlib geometry/linear-algebra routines the PR-8 port needs.

Maintainer-only feasibility prototype for spec/PR-8.md; not part of the
shipped runtime. Everything here is plain Python ``float`` arithmetic
(IEEE-754 double, one rounding per operation, no fused multiply-add), which
is exactly what dlib's own scalar C++ loops compute when built — as both
``dlib-bin`` and this repository's probe binaries are — without
``DLIB_USE_LAPACK``/``DLIB_USE_BLAS``. NumPy/LAPACK equivalents
(``numpy.linalg.svd``/``pinv``/``det``, ``np.hypot``, BLAS ``matmul``) compute
the same mathematics through different operation orders and disagree with
dlib in the last bit on most inputs; that is not good enough for the exact
parity rule in spec/PR-8.md, because those last bits routinely feed a
``floor(x + 0.5)`` or a ``static_cast<unsigned char>`` downstream.

Ported from ``tools/vendor/dlib/dlib/``:

- :func:`svd4`: ``matrix/matrix_la.h``'s non-LAPACK ``svd4`` (Golub-Reinsch,
  translated literally including its ``goto`` control flow).
- :func:`pinv`: ``pinv``/``pinv_helper`` (same file).
- :func:`find_affine_transform`, :func:`find_similarity_transform`:
  ``geometry/point_transforms.h``.
- Division of a dlib *matrix expression* by a scalar is multiplication by
  its reciprocal (``matrix/matrix.h``'s ``operator/``), while division of a
  ``dlib::vector`` or a scalar is true division; the ports keep that
  distinction wherever dlib's own code has it. Likewise a scalar factor of
  a matrix product is applied *after* the product (``(r*v)*c``), because
  dlib's expression templates move it there.
- :func:`matmul`: every dlib matrix product used here evaluates each output
  element as a sequential dot product (``matrix_multiply_helper::eval``,
  ``matrix/matrix.h``; the blocked path in ``matrix_default_mul.h`` is only
  taken for matrices far larger than any here).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

__all__ = [
    "det2",
    "find_affine_transform",
    "find_similarity_transform",
    "matmul",
    "pinv",
    "rotate_point",
    "svd4",
]

Matrix = list[list[float]]
Point = tuple[float, float]

_DOUBLE_EPSILON = sys.float_info.epsilon
_DOUBLE_MIN = sys.float_info.min


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """``a * b`` with dlib's element order: ``t = a[r][0]*b[0][c]; t += a[r][i]*b[i][c]``."""
    inner = len(b)
    result = []
    for row in a:
        out_row = []
        for c in range(len(b[0])):
            total = row[0] * b[0][c]
            for i in range(1, inner):
                total += row[i] * b[i][c]
            out_row.append(total)
        result.append(out_row)
    return result


def transpose(a: Matrix) -> Matrix:
    return [list(column) for column in zip(*a, strict=True)]


def det2(m: Matrix) -> float:
    """dlib's closed-form 2x2 determinant (``matrix_la.h``'s ``det`` specialization)."""
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def svd4(a: Matrix) -> tuple[Matrix, list[float], Matrix]:
    """``svd4(SVD_SKINNY_U, withv=true, a, u, q, v)`` for an ``m x n`` matrix, ``m >= n``.

    Returns ``(u, q, v)`` with ``a == u @ diag(q) @ v.T``; ``u`` is ``m x n``.
    Raises ``ArithmeticError`` where dlib would return a nonzero convergence
    error code (it never does for the well-conditioned 2x2/3x3 inputs here).
    """
    m = len(a)
    n = len(a[0])
    if m < n:
        raise ValueError("svd4 requires rows >= columns")
    eps = _DOUBLE_EPSILON
    tol = _DOUBLE_MIN / eps

    e = [0.0] * n
    q = [0.0] * n
    u = [[float(value) for value in row] for row in a]
    v = [[0.0] * n for _ in range(n)]
    failed_at: int | None = None

    # Householder's reduction to bidiagonal form.
    g = x = 0.0
    l = 0  # noqa: E741 - dlib's own variable name, kept for line-by-line review
    for i in range(n):
        e[i] = g
        s = 0.0
        l = i + 1  # noqa: E741
        for j in range(i, m):
            s += u[j][i] * u[j][i]
        if s < tol:
            g = 0.0
        else:
            f = u[i][i]
            g = math.sqrt(s) if f < 0 else -math.sqrt(s)
            h = f * g - s
            u[i][i] = f - g
            for j in range(l, n):
                s = 0.0
                for k in range(i, m):
                    s += u[k][i] * u[k][j]
                f = s / h
                for k in range(i, m):
                    u[k][j] += f * u[k][i]
        q[i] = g
        s = 0.0
        for j in range(l, n):
            s += u[i][j] * u[i][j]
        if s < tol:
            g = 0.0
        else:
            f = u[i][i + 1]
            g = math.sqrt(s) if f < 0 else -math.sqrt(s)
            h = f * g - s
            u[i][i + 1] = f - g
            for j in range(l, n):
                e[j] = u[i][j] / h
            for j in range(l, m):
                s = 0.0
                for k in range(l, n):
                    s += u[j][k] * u[i][k]
                for k in range(l, n):
                    u[j][k] += s * e[k]
        y = abs(q[i]) + abs(e[i])
        if y > x:
            x = y

    # Accumulation of right-hand transformations.
    for i in range(n - 1, -1, -1):
        if g != 0.0:
            h = u[i][i + 1] * g
            for j in range(l, n):
                v[j][i] = u[i][j] / h
            for j in range(l, n):
                s = 0.0
                for k in range(l, n):
                    s += u[i][k] * v[k][j]
                for k in range(l, n):
                    v[k][j] += s * v[k][i]
        for j in range(l, n):
            v[i][j] = v[j][i] = 0.0
        v[i][i] = 1.0
        g = e[i]
        l = i  # noqa: E741

    # Accumulation of left-hand transformations (u is m x n: the "pad to
    # full" loop dlib runs first is empty for SVD_SKINNY_U).
    for i in range(n - 1, -1, -1):
        l = i + 1  # noqa: E741
        g = q[i]
        for j in range(l, n):
            u[i][j] = 0.0
        if g != 0.0:
            h = u[i][i] * g
            for j in range(l, n):
                s = 0.0
                for k in range(l, m):
                    s += u[k][i] * u[k][j]
                f = s / h
                for k in range(i, m):
                    u[k][j] += f * u[k][i]
            for j in range(i, m):
                u[j][i] /= g
        else:
            for j in range(i, m):
                u[j][i] = 0.0
        u[i][i] += 1.0

    # Diagonalization of the bidiagonal form.
    eps *= x
    for k in range(n - 1, -1, -1):
        iteration = 0
        while True:  # test_f_splitting
            cancel = True
            for l in range(k, -1, -1):  # noqa: E741
                if abs(e[l]) <= eps:
                    cancel = False
                    break
                if abs(q[l - 1]) <= eps:
                    break
            if cancel:
                c = 0.0
                s = 1.0
                l1 = l - 1
                for i in range(l, k + 1):
                    f = s * e[i]
                    e[i] *= c
                    if abs(f) <= eps:
                        break
                    g = q[i]
                    h = q[i] = math.sqrt(f * f + g * g)
                    c = g / h
                    s = -f / h
                    for j in range(m):
                        y = u[j][l1]
                        z = u[j][i]
                        u[j][l1] = y * c + z * s
                        u[j][i] = -y * s + z * c

            # test_f_convergence
            z = q[k]
            if l == k:
                break  # convergence
            iteration += 1
            if iteration > 300:
                failed_at = k
                break
            x = q[l]
            y = q[k - 1]
            g = e[k - 1]
            h = e[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
            g = math.sqrt(f * f + 1.0)
            f = ((x - z) * (x + z) + h * (y / ((f - g) if f < 0 else (f + g)) - h)) / x

            # Next QR transformation.
            c = s = 1.0
            for i in range(l + 1, k + 1):
                g = e[i]
                y = q[i]
                h = s * g
                g *= c
                e[i - 1] = z = math.sqrt(f * f + h * h)
                c = f / z
                s = h / z
                f = x * c + g * s
                g = -x * s + g * c
                h = y * s
                y *= c
                for j in range(n):
                    x = v[j][i - 1]
                    z = v[j][i]
                    v[j][i - 1] = x * c + z * s
                    v[j][i] = -x * s + z * c
                q[i - 1] = z = math.sqrt(f * f + h * h)
                if z != 0:
                    c = f / z
                    s = h / z
                f = c * g + s * y
                x = -s * g + c * y
                for j in range(m):
                    y = u[j][i - 1]
                    z = u[j][i]
                    u[j][i - 1] = y * c + z * s
                    u[j][i] = -y * s + z * c
            e[l] = 0.0
            e[k] = f
            q[k] = x
        if failed_at is not None:
            # dlib's `break` out of the whole k loop on non-convergence.
            break
        # convergence
        if z < 0.0:
            q[k] = -z
            for j in range(n):
                v[j][k] = -v[j][k]

    if failed_at is not None:
        raise ArithmeticError(f"svd4 failed to converge at singular value {failed_at}")
    return u, q, v


def pinv(m: Matrix) -> Matrix:
    """``pinv(m)`` with the default ``tol = 0`` (``pinv_helper`` via ``svd3``)."""
    rows, cols = len(m), len(m[0])
    if cols > rows:
        return transpose(pinv(transpose(m)))
    u, w, v = svd4(m)
    # (machine_eps*max(nr, nc))*max(w), evaluated left to right as in dlib.
    eps = _DOUBLE_EPSILON * max(rows, cols) * max(w)
    # round_zeros(w, eps) then reciprocal(): 1/x for nonzero x, else 0.
    inverted = [0.0 if (-eps < value < eps) else value for value in w]
    inverted = [1.0 / value if value != 0 else 0.0 for value in inverted]
    # tmp(scale_columns(v, inverted)) * trans(u)
    scaled = [[v[r][c] * inverted[c] for c in range(cols)] for r in range(cols)]
    return matmul(scaled, transpose(u))


def find_affine_transform(
    from_points: Sequence[Point], to_points: Sequence[Point]
) -> tuple[Matrix, list[float]]:
    """``find_affine_transform`` (point_transforms.h:287-317): returns ``(m, b)``."""
    if len(from_points) != len(to_points) or len(from_points) < 3:
        raise ValueError("find_affine_transform needs >= 3 matching point pairs")
    p = [
        [float(point[0]) for point in from_points],
        [float(point[1]) for point in from_points],
        [1.0] * len(from_points),
    ]
    q = [
        [float(point[0]) for point in to_points],
        [float(point[1]) for point in to_points],
    ]
    solved = matmul(q, pinv(p))
    m = [[solved[0][0], solved[0][1]], [solved[1][0], solved[1][1]]]
    b = [solved[0][2], solved[1][2]]
    return m, b


def find_similarity_transform(
    from_points: Sequence[Point], to_points: Sequence[Point]
) -> tuple[Matrix, list[float]]:
    """``find_similarity_transform`` (point_transforms.h:321-382): returns ``(m, b)``.

    Inputs are converted to double first, as dlib's ``dlib::vector<double,2>``
    accumulators do for float32 and integer point types alike.
    """
    count = len(from_points)
    if count != len(to_points) or count < 2:
        raise ValueError("find_similarity_transform needs >= 2 matching point pairs")
    from_xy = [(float(point[0]), float(point[1])) for point in from_points]
    to_xy = [(float(point[0]), float(point[1])) for point in to_points]

    mean_from_x = mean_from_y = mean_to_x = mean_to_y = 0.0
    for (fx, fy), (tx, ty) in zip(from_xy, to_xy, strict=True):
        mean_from_x += fx
        mean_from_y += fy
        mean_to_x += tx
        mean_to_y += ty
    divisor = float(count)
    mean_from_x /= divisor
    mean_from_y /= divisor
    mean_to_x /= divisor
    mean_to_y /= divisor

    sigma_from = 0.0
    cov = [[0.0, 0.0], [0.0, 0.0]]
    for (fx, fy), (tx, ty) in zip(from_xy, to_xy, strict=True):
        dfx, dfy = fx - mean_from_x, fy - mean_from_y
        dtx, dty = tx - mean_to_x, ty - mean_to_y
        sigma_from += dfx * dfx + dfy * dfy
        # (to - mean_to) * trans(from - mean_from): a rank-1 outer product.
        cov[0][0] += dtx * dfx
        cov[0][1] += dtx * dfy
        cov[1][0] += dty * dfx
        cov[1][1] += dty * dfy
    sigma_from /= divisor
    # `cov /= n` on a dlib *matrix* is `cov * (1/n)`: dlib implements
    # matrix/scalar as multiplication by the reciprocal (matrix.h's
    # operator/ -> matrix_mul_scal_exp(m, one/s)), unlike the true divisions
    # above on dlib::vector and double.
    reciprocal = 1.0 / divisor
    cov = [[value * reciprocal for value in row] for row in cov]

    u, singular, v = svd4(cov)
    d = [[singular[0], 0.0], [0.0, singular[1]]]
    s = [[1.0, 0.0], [0.0, 1.0]]
    det_cov = det2(cov)
    if det_cov < 0 or (det_cov == 0 and det2(u) * det2(v) < 0):
        if d[1][1] < d[0][0]:
            s[1][1] = -1.0
        else:
            s[0][0] = -1.0

    r = matmul(matmul(u, s), transpose(v))
    c = 1.0
    if sigma_from != 0:
        ds = matmul(d, s)
        c = 1.0 / sigma_from * (ds[0][0] + ds[1][1])
    cr = [[value * c for value in row] for row in r]
    # `mean_to - c*r*mean_from`: dlib's expression templates percolate the
    # scalar outside the product (matrix.h's matrix_mul_scal_exp operator*
    # overloads), so this evaluates as `(r*mean_from)*c`, not `(c*r)*mean_from`.
    b = [
        mean_to_x - (r[0][0] * mean_from_x + r[0][1] * mean_from_y) * c,
        mean_to_y - (r[1][0] * mean_from_x + r[1][1] * mean_from_y) * c,
    ]
    return cr, b


def rotate_point(center: Point, point: Point, angle: float) -> Point:
    """``rotate_point<double>`` via ``point_rotator`` (point_transforms.h:22-49, 678-687)."""
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    px = point[0] - center[0]
    py = point[1] - center[1]
    x = cos_angle * px - sin_angle * py
    y = sin_angle * px + cos_angle * py
    return x + center[0], y + center[1]
