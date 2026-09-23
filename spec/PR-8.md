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
