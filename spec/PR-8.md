# PR-8 — Conditional follow-up: remove dlib from the runtime

**Status:** conditional scope, not required for the recommended compiler-free
package in PR-7.

**Depends on:** PR-1 reference fixtures and the narrow frontend boundary from
PR-4; integrates after the working package exists.

**Risk:** very high; split implementation into smaller reviewed PRs after the
initial feasibility experiment.

## When this work is needed

Use this track if “all on Python side” means **no runtime dlib dependency**,
including prebuilt wheels. PR-1–PR-7 remove project-owned native code and end-user
compilation, but intentionally retain dlib's matching detector/alignment behavior
through a Python package dependency.

This track can still use NumPy and ONNX Runtime wheels. Requiring no native code
anywhere, including those dependencies, is a different performance constraint
and would require a separate implementation plan.

## Why replacing the two CNNs is insufficient

Three remaining frontend components affect every result:

1. Embedded dlib HOG detector: a trained five-filter detector with pyramid,
   coordinate mapping, scoring, suppression, and ordering behavior.
2. Five-point landmark `.dat`: a cascade of regression trees with shape-relative
   pixel sampling and ordered floating-point updates.
3. Chip extraction: landmark similarity transform, padding, image-pyramid
   downsampling, bilinear interpolation, and integer pixel conversion.

A modern ONNX face detector with five landmark outputs changes boxes and crops.
It may be faster or detect more faces, but it does not satisfy exact legacy
compatibility. External boxes alone do not remove the need for matching landmarks
and chip extraction.

## Initial feasibility deliverable

Before committing to a full port, implement a small reference prototype that:

- Exports the original landmark tree parameters to a versioned NumPy-friendly
  artifact through maintainer tooling.
- Runs landmark prediction for the supplied external boxes in Python/NumPy and
  compares every cascade stage with the dlib oracle.
- Ports chip details and single-chip extraction for the fixture images and
  compares all uint8 pixels, including pyramid-downsampled and boundary cases.
- Times this path against the wheel frontend, using the PR-6 methodology.
- Produces an explicit go/no-go report for correctness, CPU cost, maintenance
  complexity, and the remaining detector work.

Unknown arbitrary `.dat` reading is not necessary for a bundled known-model port:
exported immutable parameters avoid having to maintain a general dlib binary
parser. If direct format compatibility becomes a requirement, scope and test that
parser separately rather than folding it into the numerical port.

## Follow-on implementation units if feasible

### A. Landmark model and predictor

- Export initial shape, cascade forests, split indices/thresholds, leaf vectors,
  anchor indices, and shape-relative deltas with exact dtypes and source hashes.
- Follow `libs/dlib/dlib/image_processing/shape_predictor.h`, including feature
  pixel extraction and rectangle normalizing transforms.
- Preserve RGB intensity computation, out-of-image samples, float32 stage/tree
  accumulation order, and integer landmark coordinate conversion.
- Vectorize across faces/trees only where it preserves the oracle updates.
- Validate all stage tensors and final integer landmarks against PR-1.

### B. Alignment and chip extraction

- Port the five-point canonical template and similarity transform.
- Preserve `chip_details` rectangle/angle calculations and inclusive-coordinate
  conventions; verify floating and integer geometry separately.
- Port pyramid-down filtering/origins and individual crop bounding logic, then
  bilinear sampling and uint8 assignment rules.
- Validate border padding, high-downsample ratios, rotations, odd dimensions,
  and both chip sizes with exact array equality.
- Use no generic Pillow/OpenCV resize substitution unless proven byte-identical
  for every supported branch and fixture.

### C. Embedded HOG detector

- Export the pinned learned detector rather than using a newly trained model.
- Port RGB gradient selection, dlib FHOG bins/normalization, `pyramid_down<6>`,
  learned filter evaluation, score thresholds, rectangle mapping, overlap tests,
  and tie/order behavior.
- Relevant sources include `image_processing/frontal_face_detector.h`,
  `scan_fhog_pyramid.h`, `object_detector.h`, `image_transforms/fhog.h`, and
  `image_transforms/image_pyramid.h` under the pinned dlib tree.
- Benchmark feature extraction/filter evaluation independently. Python loops
  over all pixels/windows are unlikely to meet the performance objective;
  vectorize or use standard ONNX operators where faithful and practical.
- Generic scikit-image HOG/OpenCV detectors are not presumed equivalent.

### D. Integrate and remove the dependency

- Run exact frontend and end-to-end parity across all supported CPU platforms.
- Re-run binary-only package installation and all performance/resource checks.
- Replace the `_faces.py` implementation behind the existing boundary; keep the
  public API/result contract.
- Remove `dlib-bin` only after the new default fully supports auto-detection and
  caller-supplied boxes. Do not call a boxes-only partial implementation complete.
- Update asset manifests, attribution, package size, platform matrix, and model
  loader documentation. `load_shape_predictor` still has to load any
  compatible five-point predictor directly (no hash lookup, no manifest —
  see spec/PR-5.md's "Design change" section) if its runtime representation
  changes.

## Acceptance and decision rule

- Exact rectangles/order, landmarks, chips, and public dictionaries under the
  same PR-1 contract; no relaxed “visually similar” test.
- No `import dlib`, native extension build, or hidden dlib subprocess at runtime.
- Numerical/performance reports show the consequence of the port explicitly.
- A feasibility failure is recorded as such. It does not silently authorize
  changing face models or weakening parity.

The recommended first release remains PR-7 because it directly addresses the
installation problem with substantially less numerical reimplementation. This
conditional track makes the remaining work explicit if eliminating the native
dlib runtime itself is also a hard requirement.

## Implementation status

The **initial feasibility deliverable only** was delivered on `pr-8`:
maintainer-only prototype tooling under `tools/conversion/`
(`export_shape_predictor.cpp`, `probe_shape_predictor.cpp`,
`build_shape_predictor.py`, `dlib_linalg.py`, `numpy_shape_predictor.py`,
`numpy_chip_extraction.py`, `shape_predictor_evidence.py`,
`validate_shape_predictor.py`, `benchmark_numpy_frontend.py`), exercised by
`tests/parity/test_numpy_shape_predictor_feasibility.py`. Results are in
`tools/conversion/shape-predictor-feasibility-report.md`.

**Follow-on implementation units A/B/C/D (landmark model, chip extraction,
HOG detector, `_faces.py` integration) were not attempted.** This PR does not
change `src/age_and_gender/`, `pyproject.toml`, or any shipped dependency.
Per this file's own framing ("split implementation into smaller reviewed PRs
after the initial feasibility experiment"), that was the intended scope
boundary, not a partial implementation of the larger track.

Initial feasibility deliverable, item by item:

- [x] Export the landmark tree parameters to a versioned NumPy artifact.
      The builder runs the exporter on the hashed, pinned source file itself.
      The loader verifies schema, artifact SHA-256, dtypes, shapes, cascade
      layout, and index ranges.
- [x] Landmark prediction in Python/NumPy compared with the dlib oracle at
      every cascade stage: 165/165 (face, cascade) pairs bit-exact against
      the C++ probe, and replayed in CI from the checked-in `stage-trace.json`.
- [x] Chip details and single-chip extraction compared on all uint8 pixels:
      exact on the frozen corpus, 10,000 boundary/deep-pyramid synthetic
      cases (depth up to 5), 5 retained regression cases, and 100,000
      bit-exact `chip_details` geometries, plus a separate 30,000-case sweep
      with zero mismatches.
- [x] Timed against the wheel frontend with the PR-6 protocol (10 warmups ×
      100 iterations × 5 repeats, equivalent work on both sides, machine,
      dependency, and thread metadata recorded).
- [x] Explicit go/no-go report: go on correctness for units A/B; the
      ~90x-slower per-face frontend is an open performance problem; unit C
      was not evaluated, so no go/no-go for the whole track.

Acceptance and decision rule, scoped to what this PR covers:

- [x] Exact landmarks and chips under the PR-1 contract, with no tolerance
      and no known residual. Rectangles/order and public dictionaries
      depend on the detector and `_faces.py` integration (units C/D) and are
      **not** covered by this PR.
- [x] The prototype imports no dlib. Only the maintainer-only validator,
      benchmark, and parity test import `dlib` to compare against it.
- [x] `shape-predictor-report.json` and `numpy-frontend-benchmark.json`
      state the numerical and performance consequences explicitly. The CI
      test rejects either report if it is stale relative to the current
      prototype sources, or if any validation section is missing or failed.
- [x] The HOG detector (unit C) is recorded as an explicit non-attempt, not
      a silent gap.

Exactness depends on the oracle using scalar, non-FMA double arithmetic, as
`dlib-bin` 20.0.1 does on x86-64 (built without LAPACK/BLAS). Unit D's
"all supported CPU platforms" parity run must re-establish this for every
platform wheel.
