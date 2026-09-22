# PR-1 — Freeze legacy behavior and build a reference oracle

**Depends on:** none.

**Risk:** medium; enables every later parity claim.

**Deliverable:** reproducible reference inference, stage fixtures, and executable
compatibility rules.

## Goal

Make “same result as the old C++ interface” measurable before changing runtimes.
Use the current source and the exact local model hashes from
[INVESTIGATION.md](INVESTIGATION.md), not README screenshots or another dlib
example program.

## Work

1. Add a maintainer-only oracle under `tools/legacy/`:
   - Preserve the exact network aliases from `src/main.cpp`. PR-2 moved these
     into the shared `tools/network_definitions.h` so the oracle and the ONNX
     exporter cannot drift apart; the reference manifests record that header's
     SHA-256 alongside `oracle.cpp`.
   - Use vendored dlib 19.20.0 initially, CPU float32, without fast-math/CUDA.
   - Read lossless/raw RGB input and optional explicit rectangles.
   - Export detected rectangles in order, landmarks, individual 32/64 chips,
     normalized tensors, network logits/probabilities, age expectation, and
     final result dictionaries.
   - Preserve the legacy per-call subnet copy in the end-to-end timing path;
     instrumented tensor export is a separate untimed path.
   - Keep the legacy extension runnable in a pinned compatible environment so
     its public outputs can be checked against the standalone oracle. If modern
     Python cannot build pybind11 2.5, use a pinned historical build environment
     for this check rather than changing production semantics.
2. Record a reference manifest: repository revision, model and input SHA-256,
   compiler flags/version, CPU features, OS, image decoder/version, dlib version,
   thread settings, and oracle source/build recipe.
3. Add fixtures under `tests/fixtures/legacy/`:
   - Original two example images and the exact decoded RGB arrays used by both
     paths; reference existing images rather than duplicating large payloads
     unnecessarily.
   - Auto-detection, explicit boxes, reversed box order, duplicate boxes, empty
     box list, no-face images, and in-frame/out-of-frame rectangles.
   - A representative frozen corpus of at least 30 images and 100 faces including
     rotations, small/large faces, multiple face scales, image borders, and
     different illumination. Record source and redistribution status of added
     fixtures; external corpus artifacts may be content-addressed in CI.
   - Deterministic synthetic images/chips and postprocessing vectors for edge
     cases. They complement real images, not replace them.
4. Add stage-comparison helpers and a machine-readable report. A failed stage
   should name image, face index, model, expected/actual shape, maximum error,
   and first mismatch.
5. Capture a repeatable baseline for model load, cold inference, warm inference,
   frontend time, CNN time, and peak memory. Save machine metadata with results.

## Compatibility rules to freeze

- RGB uint8 input and original image coordinates.
- No detector upsampling in the default path.
- Input `(top, right, bottom, left)`; output `[left, top, right, bottom]`,
  inclusive endpoints and stable order.
- Empty external boxes trigger detection.
- Independent source-image crops at 32 and 64, padding 0.2.
- Actual serialized input means, division by 256, NCHW float32.
- Age expectation with class-zero weight 0.25 and sequential float32 accumulation.
- Half-away-from-zero age rounding; floored float32 percentage confidences.
- Gender class order female/male, with ties selecting female.
- Ordinary Python scalar/list/dictionary return types.

Missing-model behavior and malformed native-memory inputs need clean new errors,
not emulation of crashes. Keep these separately marked as intentional validation
improvements.

## Parity thresholds

Use the following initial **release thresholds**, fixed before conversion:

| Stage | Required agreement |
| --- | --- |
| Detector boxes/count/order | Exact |
| Landmark coordinates | Exact |
| Individual uint8 chips | Exact array equality |
| Normalized input float32 tensors | Exact, including dtype/layout |
| Intermediate logits / floating activations | `atol=1e-5`, `rtol=1e-4`, finite values |
| Softmax probability vectors | `atol=1e-6`, `rtol=1e-4`, valid shape/range and sums |
| Pre-rounding expected age | Absolute difference <= `1e-4` |
| Public result values and ordering | Exact; **no ±1-year or ±1%-point exception** |

These floating thresholds are proposals to ratify using the reference stability
study, not measurements from the investigation. PR-2 ratified one of them: deep
internal activations are compared at `atol=1e-4`, because the measured float32
accumulation noise of the reference itself is 5.10e-05 at the deepest stage. The
logits, probability, age-expectation and public-result rows are unchanged. If identical legacy runs on
different machines exceed them, establish a documented reference CPU lane and
investigate before relaxing any requirement. A tensor passing its tolerance
never overrides a public-output mismatch.

Test synthetic ties and values immediately below/at/above age half-integers and
confidence-percent integer boundaries. Include probability vectors where the
expected age differs from the most likely class, and where float32 sequential
reduction differs from a float64 or pairwise reduction.

## Proposed files

- `tools/legacy/README.md`, `tools/legacy/CMakeLists.txt`, oracle source and pinned
  environment recipe.
- `tests/fixtures/legacy/manifest.json`, raw tensors/chips and expected JSON.
- `tests/parity/` comparison helpers and oracle tests.
- `benchmarks/legacy_baseline.py` and baseline result artifact.

## Acceptance

- [ ] A clean maintainer environment can rebuild and run the oracle.
- [ ] Oracle and original extension agree on intended valid-input fixtures.
- [ ] Every reference artifact is traceable to source/model/input hashes.
- [ ] Repeated runs are stable in the defined reference lane.
- [ ] Tests cover rounding, color/layout, rectangles, and empty-box semantics.
- [ ] Baseline report distinguishes measured timings from setup/instrumentation.
- [ ] Public Python package installation is not made dependent on oracle tools.

**Stop condition:** if a reliable reference cannot be established, investigate
that first; do not generate “golden” results from the replacement implementation.

## Implementation status

Delivered on `refactor/pr-1-legacy-oracle`, with two follow-ups applied in PR-2:
the network aliases moved to the shared `tools/network_definitions.h`, and the
goldens were re-frozen so their provenance blocks match the current oracle
source. The re-freeze changed only manifest metadata; every chip, tensor, logit,
probability, landmark and public result is byte-identical, and the rebuilt
oracle binary hashes the same as the pre-refactor build.

Acceptance against the list above:

- [x] A clean maintainer environment can rebuild and run the oracle.
- [x] Oracle and original extension agree on intended valid-input fixtures.
- [x] Every reference artifact is traceable to source/model/input hashes.
- [x] Repeated runs are stable in the defined reference lane.
- [x] Tests cover rounding, color/layout, rectangles, and empty-box semantics.
- [x] Baseline report distinguishes measured timings from setup/instrumentation.
- [x] Public Python package installation is not made dependent on oracle tools.

The corpus item in the work list is **not** met.
`tests/fixtures/legacy/manifest.json` records
`representative_30_image_100_face_corpus` as deferred to release hardening. The
frozen set is 11 faces over 3 images, which collapses to 8 distinct chips once
the duplicate and repeated boxes are removed. Every later parity claim inherits
that limit, so PR-4 should widen it before release rather than treating the
frozen set as sufficient evidence.
