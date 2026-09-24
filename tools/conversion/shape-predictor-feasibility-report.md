# PR-8 feasibility report: shape predictor and chip extraction in NumPy

This is the "Initial feasibility deliverable" spec/PR-8.md asks for before
committing to the full dlib-removal track (its units A–D). It does **not**
port the HOG detector (unit C) and does **not** touch
`src/age_and_gender/_faces.py` or any shipped dependency — everything here is
maintainer-only prototype code under `tools/conversion/`, exercised by
`tests/parity/test_numpy_shape_predictor_feasibility.py` and
`tools/conversion/validate_shape_predictor.py`.

## What was built

| Piece | File | Role |
| --- | --- | --- |
| Trained-parameter exporter | `export_shape_predictor.cpp` | Deserializes `shape_predictor_5_face_landmarks.dat` (replicating `dlib::deserialize(shape_predictor&, istream&)`'s exact field sequence) and dumps `initial_shape`/`forests`/`anchor_idx`/`deltas` to flat little-endian binaries + a JSON index. |
| Stage-trace oracle | `probe_shape_predictor.cpp` | Inlines `shape_predictor::operator()`'s loop, calling dlib's own `dlib::impl::` helpers directly (not a reimplementation), dumping per-cascade `current_shape`/`feature_pixel_values`, final landmarks, `chip_details`, and final chip pixels. |
| Artifact builder | `build_shape_predictor.py` | Hashes the source `.dat`, refuses anything but the pinned model, runs the exporter on that same file, and packages `artifacts/shape-predictor-v1/` (`.npz` + `manifest.json`). |
| dlib numerics | `dlib_linalg.py` | Scalar ports of dlib's own `svd4`, `pinv`, `find_similarity_transform`, `find_affine_transform`, `rotate_point`, and small-matrix products, in dlib's exact operation order. |
| NumPy shape predictor | `numpy_shape_predictor.py` | Port of the cascade-regression-tree landmark predictor, plus a loader that verifies the artifact's schema, SHA-256, dtypes, shapes, layout, and index ranges. |
| NumPy chip extraction | `numpy_chip_extraction.py` | Port of `get_face_chip_details` + `extract_image_chip`, including the `pyramid_down<2>` decimation filter and the zero-angle raw-copy bypass. |
| Evidence helpers | `shape_predictor_evidence.py` | Source `code_revision`, reproducible `(seed, index)` synthetic cases, environment metadata. |
| Validator | `validate_shape_predictor.py` | Six checks (below); writes `shape-predictor-report.json` and the CI stage-trace fixture `stage-trace.json`. |
| Benchmark | `benchmark_numpy_frontend.py` | Times equivalent NumPy vs `dlib-bin` work under spec/PR-6.md's protocol; writes `numpy-frontend-benchmark.json`. |

## Correctness result: exact

`shape-predictor-report.json` (checked in; top-level `passed` is true only if
every section ran and passed, and the CI test rejects it if it is stale
relative to the current sources):

| Check | Volume | Mismatches |
| --- | ---: | ---: |
| Frozen PR-1 corpus: landmarks and both chips, exact | 11 faces, 22 chips | 0 |
| Per-cascade stages vs the C++ oracle, bit-exact features and shapes | 165 (face, cascade) pairs | 0 |
| Retained regression cases vs live `dlib-bin` | 5 | 0 |
| Live `dlib-bin` sweep: landmarks + both chips (8,651 boundary cases; pyramid depth up to 5) | 10,000 | 0 |
| `chip_details` rect and angle, bit-exact vs `dlib.get_face_chip_details` | 100,000 | 0 |
| Zero-angle raw-copy decision (fractional and integral rect sizes) | 9 | 0 |

Separately from the report, the same live-dlib comparison was run on 30,000
cases (seed 2026, indices 0–29,999) against both the previous and the
current code: **5 mismatches before, 0 after**. Those 5 inputs are retained
in `regression-cases.json` and checked by the validator and CI.

The CI test (`tests/parity/test_numpy_shape_predictor_feasibility.py`) runs
without the probe. It replays the oracle's stage trace from `stage-trace.json`,
reruns the frozen corpus (a missing corpus image fails the test), checks the
regression cases, a fresh 24-case synthetic sweep, 1,500 bit-exact
`chip_details` geometries and the raw-copy decision, and verifies input
validation and artifact tamper rejection.

### What it took: three floating-point problems, all fixed

**1. BLAS/matmul non-associativity (fixed in the first version).** Each
cascade's feature projection was first computed as one batched NumPy
`matmul`. dlib computes it per feature as an independent 2-term scalar sum
(`shape_predictor.h:271-279`). The two are mathematically equivalent but not
bit-identical, and a one-ULP difference at a `.5` rounding boundary sampled
the wrong pixel, which then compounded into different landmarks. Every
repeated small transform is now explicit elementwise arithmetic.

**2. `current_shape` precision (fixed in the first version).** dlib keeps
the shape estimate in float32 throughout the cascade. Accumulating in float64
caused ~1e-6 stage error, which the bit-exact stage trace catches.

**3. Not dlib's arithmetic (fixed in this revision).** The first version
used `numpy.linalg.svd`/`pinv`/`det`, `np.hypot`, and a closed-form
`diag(width, height)` for `unnormalizing_tform`. The earlier report called
the resulting residual an "unfixable-in-scope" SVD-algorithm difference
affecting roughly 1 in 4,000 rectangles. That was wrong on both counts:

- *It was not rare at the level that matters.* On random landmark sets,
  `chip_details` differed from dlib's in the last bit **93% of the time**.
  Only the final rounding hid it, so most last-bit differences did not change
  an output, while ~1 in 6,000 cases changed a landmark or chip pixel.
- *It was fixable.* `dlib-bin` 20.0.1 is built without LAPACK/BLAS
  (`dlib.DLIB_USE_LAPACK == False`), as are this repository's probes, so
  dlib runs its own scalar Golub–Reinsch `svd4`. Python floats perform
  exactly the same IEEE-754 double operations as those scalar C++ loops, so a
  literal port in the same operation order reproduces them bit for bit.
  `dlib_linalg.py` is that port. Getting it exact also needed three
  expression-template details that are easy to miss:
  - dlib's `matrix / scalar` is multiplication by the reciprocal, so
    `cov /= n` and the canonical template's `(padding + p) / (2*padding + 1)`
    both multiply by `1/x`.
  - A scalar factor of a matrix product moves outside the product, so
    `c*r*mean_from` is evaluated as `(r*mean_from)*c`.
  - `c = 1.0/sigma_from * trace(d*s)`, not `trace / sigma_from`.

  The similarity fit, `unnormalizing_tform`'s `Q*pinv(P)` (which has tiny
  nonzero off-diagonal residue in dlib, so the closed form was "more exact"
  than dlib and therefore wrong), rotations (`point_rotator` with libm
  `sin`/`cos`, not NumPy's possibly SIMD versions), and `length()` (`sqrt`,
  not `hypot`) all now go through these ports.

Other defects fixed in this revision, each covered by a test:

- The zero-angle raw-copy bypass compared the chip size against a *rounded*
  rect. dlib compares against the floating-point rect, so fractional rects
  such as `(0.2, 0.2, 2.4, 2.4)` must interpolate.
- Rect emptiness now follows `drectangle::is_empty()` (`l > r || t > b`),
  and the pyramid level is selected on the bounding-box-local rect exactly
  as dlib does, instead of being assumed equal to the depth.
- `get_face_chip_details` now enforces dlib's `size > 0 && padding >= 0`
  precondition. Previously a size of 0 made the pyramid loop run forever.
  `extract_image_chip` rejects zero-sized or empty-rect chip details.

**Portability caveat.** Exactness relies on the oracle using scalar, non-FMA
IEEE double arithmetic, as `dlib-bin` 20.0.1 does on x86-64 (checked: no FMA
instructions in its extension module). A dlib build with LAPACK enabled, or
one compiled with FMA contraction, would need its own verification. Unit D's
"all supported CPU platforms" requirement applies here directly.

## Performance result: substantially slower

`numpy-frontend-benchmark.json` (checked in). The run followed spec/PR-6.md's
steady-state protocol: 10 warmups, 100 measured iterations, five repeats,
in-process `perf_counter` per call, same decoded RGB array and rectangle
(`example/test-image.jpg`, face `(419, 266, 506, 352)`) on both sides.
Machine: AMD Ryzen 5 3600 (12 logical cores, `schedutil` governor), Linux
6.8, Python 3.13.15 (GCC 11.4), NumPy 2.5.3 (scipy-openblas 0.3.34),
`dlib-bin` 20.0.1 without BLAS/LAPACK. No thread environment variables were
set, and both timed paths are single-threaded. This is the repository's
shared development machine, not a dedicated benchmark host (see
`benchmarks/README.md`'s caveat). Per-repeat medians agreed within ~2%.

Both sides of each row time the same work. The chip rows include
`get_face_chip_details` on both sides, because the wheel's
`FaceFrontend._chip` computes chip details inside the call; the previous
revision's NumPy chip timings excluded that step. The benchmark refuses to
run unless both paths produce identical landmarks and chips for the input.

| Stage | dlib-bin median (p95) | NumPy port median (p95) | Median ratio |
| --- | ---: | ---: | ---: |
| Landmark prediction | 0.431 ms (0.604) | 44.833 ms (51.586) | **104.0x** |
| Gender chip (32x32, details + extraction) | 0.042 ms (0.048) | 1.995 ms (2.491) | **47.3x** |
| Age chip (64x64, details + extraction) | 0.063 ms (0.072) | 0.738 ms (0.778) | **11.7x** |
| Per-face total (landmarks + both chips) | 0.541 ms (0.622) | 48.544 ms (55.673) | **89.7x** |

The per-face total (landmarks plus both chips, i.e. all the work this port
would replace per face) is roughly 90x slower. dlib's landmark predictor is
compiled C++ evaluating 15 cascades × 500 trees as tight binary-tree descents,
while the NumPy port pays Python-level overhead for each of those 7,500 tree
traversals per face (`_evaluate_tree`'s loop has no vectorizable structure,
because each node depends on the previous comparison). The 32×32 gender chip is
slower than the 64×64 age chip because its source rectangle needs one more
pyramid level, and each `_pyramid_down_rgb` pass loops over output columns
in Python. The scalar `dlib_linalg` ports cost about 0.1 ms per face, which
is negligible next to tree traversal. This prototype answers the correctness
question, not the final performance question.

## Code-size / complexity assessment

| Component | Real logic (excl. weights/boilerplate) | Status here |
| --- | ---: | --- |
| Shape predictor cascade (`shape_predictor.h`) | ~150 lines | **Ported, validated exactly** |
| Chip extraction (`interpolation.h`'s relevant slice) | ~250 lines | **Ported, validated exactly** |
| Pyramid decimation filter (`image_pyramid.h`) | ~100 lines | **Ported, validated exactly** (part of the above) |
| dlib numerics (`matrix_la.h`'s `svd4`/`pinv`, `point_transforms.h` fits) | ~250 lines | **Ported, validated exactly** (`dlib_linalg.py`) |
| HOG face detector (`frontal_face_detector.h` + `scan_fhog_pyramid.h` + `fhog.h`, minus the embedded weight blob) | **~4,000+ lines** | **Not attempted** |

The detector is the dominant remaining cost by a wide margin — both in
source size and, per `benchmarks/report-pr6.md`'s own measurement, in
runtime share ("detection is roughly 3/4 of whole-image latency"). It also
has no analog to this report's "export trained parameters, then port a
well-specified numeric algorithm" strategy: dlib's `object_detector<
scan_fhog_pyramid<pyramid_down<6>>>` combines five independently-trained
HOG filters, a finer six-level pyramid than chip extraction's two-level one,
FHOG feature extraction, a separable-filter SVD decomposition, and
overlap-based non-max suppression — each a nontrivial numerical subsystem in
its own right, several of which (FHOG binning, the pyramid ratio) are more
naturally vectorizable than the landmark cascade's tree traversal but still
represent a much larger, higher-risk port. Its separable-filter
decomposition also goes through dlib's SVD, so `dlib_linalg.py` would be
reused there. The lesson from this report applies to the detector too:
port dlib's own arithmetic, and do not substitute library equivalents.

## Recommendation

**Feasibility for units A/B (landmark model, chip extraction): confirmed for
correctness.** Both are portable to Python/NumPy with bit-identical parity
against `dlib-bin` on everything checked: the frozen corpus, every cascade
stage, 100,000 chip geometries, 10,000 boundary and deep-pyramid synthetic
cases in the report, and a separate 30,000-case sweep, with zero mismatches
and no known residual. Two lessons generalize to any further dlib port here:

- Do not substitute NumPy/LAPACK/BLAS for dlib's numerics, even where they
  are "more accurate". Port dlib's own scalar routines in its own operation
  order, including its expression-template evaluation order.
- Test the unrounded intermediates (such as `chip_details`) bit for bit.
  Comparing only rounded outputs hid a 93% last-bit disagreement behind an
  apparent 1-in-several-thousand failure rate.

**Performance for A/B is the open question.** A ~90x slower per-face frontend
is very unlikely to be acceptable as-is. Removing dlib entirely rules out
keeping dlib's detector while using the NumPy landmark step, so the whole
frontend would have to clear spec/PR-6.md's 10% regression gate. Any
decision to go past this feasibility stage should budget real optimization
work for units A/B (for example batched tree evaluation or a compiled
kernel), each re-proven against this validator, not just a correctness port.

**Feasibility for unit C (HOG detector): not evaluated here.** Given its
size and algorithmic breadth relative to units A/B, it is the largest
remaining risk and effort if this track proceeds. Per spec/PR-8.md's
decision rule, this is recorded as an explicit non-attempt, not a silent
gap: **a full go/no-go for the complete dlib-free track cannot be made from
this report alone.** The next step before committing to unit D's
`_faces.py` replacement is a follow-up feasibility pass scoped to the
detector, with its own prototype, correctness sweep, and timing.
