# PR-4 — Preserve detection, landmarks, and face-chip alignment

**Depends on:** PR-1; production module integrates into PR-3's package scaffold.

**Risk:** medium/high because preprocessing directly controls model outputs.

**Deliverable:** Python frontend with exact stage parity against legacy fixtures.

## Goal

Move image handling and frontend orchestration into Python while keeping the
same learned detector, landmark predictor, and individual chip extraction.

The investigation successfully compared the current `dlib-bin==20.0.1` frontend
to vendored dlib 19.20 for seven faces. This PR expands that smoke evidence into
a real compatibility suite.

## Work

1. Implement `_images.py`:
   - Accept RGB Pillow images and H×W×3 uint8 NumPy arrays.
   - Reject empty/incorrect dimensions, non-RGB modes, and non-uint8 arrays with
     helpful errors. Do not silently multiply floats by 255, wrap integers to
     bytes, discard alpha, or swap RGB/BGR.
   - Normalize non-contiguous views using `np.ascontiguousarray` only when needed.
   - Keep input arrays unmodified, including read-only arrays.
   - Do not auto-rotate EXIF, resize, or change color profiles in the parity path;
     image decoding/conversion remains explicit at the call boundary.
2. Implement box parsing:
   - Input tuples/lists are `(top, right, bottom, left)` integers, exactly four
     values. Accept Python/NumPy integral values, excluding booleans.
   - Preserve order and duplicates; return dlib inclusive rectangles.
   - Require nondegenerate geometry (`left < right`, `top < bottom`) and values
     representable by the backend. Raise a clear error for malformed boxes.
   - Retain valid out-of-frame coordinates instead of clipping them.
   - `None` or an empty sequence means auto-detection. Avoid ambiguous NumPy
     truth-value checks; inspect length explicitly.
3. Implement `_faces.py` with `dlib-bin`, imported as `dlib`:
   - Load the same five-point model and require five landmark parts for the
     supported bundle format.
   - Initialize and reuse the frontal HOG detector.
   - Call detector with explicit zero upsampling and unchanged default threshold.
   - Bypass detection when nonempty boxes are supplied.
   - Preserve detector/box order through landmarks and both chip lists.
4. Extract chips **one face at a time, from the original image**, at size 32 and
   size 64, with padding 0.2 explicitly specified.
   - Prefer `get_face_chip_details(..., size=..., padding=0.2)` followed by
     `extract_image_chip` if that best matches the C++ double-precision padding
     path. The convenience `get_face_chip` binding uses a float padding
     parameter; compare both routes rather than assuming exact equivalence.
   - Preserve dlib's bilinear interpolation, pyramid behavior, black boundary
     handling, and uint8 conversion.
   - Never derive the 32 chip by resizing the 64 chip or vice versa.
   - Never replace alignment with rectangular crop-and-resize.
5. Expose a small internal result containing rectangles and corresponding chip
   arrays. Keep backend objects out of the public API.

## Known batching trap

On the first example image, batching 32×32 extraction with `get_face_chips`
changed **1540, 1665, 1796, 0, and 1599** channel values respectively compared
with individual extraction. The batch routine shares a crop/pyramid and is not
a safe performance substitution. Keep individual extraction even when neural
inference is batched later.

## Tests

- Direct equality of boxes, landmark arrays, and both uint8 chips for the complete
  PR-1 corpus. Compare pixels, not just hashes.
- Empty/no-face input results, externally supplied boxes, ordering, duplicates,
  partially out-of-frame faces, and mixed face scales in one image.
- Read-only and sliced/non-contiguous arrays; no mutation of caller inputs.
- Errors for grayscale/RGBA inputs, malformed shapes/dtypes, invalid rectangles,
  corrupt shape predictors, and incompatible landmark models.
- A regression fixture demonstrating that the production extraction path keeps
  the individual-crop semantics, including the known batching counterexample.
- Per-supported-platform frontend parity with the pinned wheel version.

## Proposed files

- `src/age_and_gender/_images.py`, `_faces.py`.
- `tests/unit/test_images.py`, `test_boxes.py`.
- `tests/parity/test_frontend.py`.
- Frontend comparison report referencing the oracle manifest and wheel versions.

## Acceptance

- [ ] Exact frontend equality on the frozen corpus.
- [ ] Default HOG behavior remains zero-upsampled with original ordering.
- [ ] Both crops use padding 0.2 and independent individual extraction.
- [ ] Valid boxes are not clipped/reordered and inputs are not mutated.
- [ ] No `face_recognition` or substitute detector/landmark model is required.
- [ ] Clean binary-only install of the selected dlib wheel passes frontend smoke.

**Stop condition:** if the newest dlib wheel changes crops/detections, first
identify the cause and evaluate another wheel version with current Python
support. Do not silently accept changed detections to obtain newer dependencies.
If no wheel satisfies the contract, revisit the frontend plan before PR-7.

## Implementation status

Delivered on `pr-4`. `src/age_and_gender/_images.py` validates RGB Pillow images
and `[H, W, 3]` uint8 NumPy arrays and parses legacy `(top, right, bottom,
left)` boxes into inclusive `[left, top, right, bottom]` rectangles, without
importing `dlib`. `src/age_and_gender/_faces.py` is the only module that
imports `dlib`: `FaceFrontend` loads the bundle's five-point landmark model and
the frontal HOG detector once, validates the loaded predictor actually returns
five parts (a throwaway-image probe at construction time, since a shape
predictor's part count is fixed by training rather than by its input), and
exposes `detect()` and `extract()`. Both chip sizes are extracted individually
via `get_face_chip_details(shape, size, 0.2)` + `extract_image_chip`, never
`get_face_chips`; `FaceExtraction` returns plain rectangles, landmarks, and
uint8 chip arrays, with no backend object crossing the module boundary.

`dlib-bin==20.0.1` and `Pillow>=12.3.0,<13` are now runtime dependencies.
`dlib-bin` is pinned exactly, not ranged, because `tools/legacy/frontend-
comparison.md` records what was actually compared against the oracle; widening
it means rerunning that comparison, per this file's stop condition.

### Measured parity

`tests/parity/test_frontend.py` runs the installed frontend against every
fixture in `tests/fixtures/legacy/*.golden.json` — the same 4 documents / 11
faces PR-1 froze and PR-3 already uses for chip-level neural parity. Every
rectangle, every landmark point, and every one of the 22 chips matched exactly;
see `tools/legacy/frontend-comparison.md` for the full report, including the
reproduced batching counterexample (`[1540, 1665, 1796, 0, 1599]` differing
channel values, matching `spec/INVESTIGATION.md`'s recorded numbers) and the
padding-precision check confirming `get_face_chip_details`'s padding argument
is honored at full `double` precision on this wheel, matching the vendored
dlib 19.20 C++ signature the oracle calls.

New tests: `tests/unit/test_images.py`, `tests/unit/test_boxes.py`,
`tests/unit/test_faces.py` (corrupt and incompatible-landmark-model bundles,
built from a real dlib shape predictor trained from scratch on synthetic data
rather than a downloaded fixture, so no new binary asset is needed to exercise
that path), and `tests/parity/test_frontend.py`. The suite is 196 tests and
`ruff check .` is clean.

### Acceptance against the list above

- [x] Exact frontend equality on the frozen corpus.
- [x] Default HOG behavior remains zero-upsampled with original ordering.
- [x] Both crops use padding 0.2 and independent individual extraction.
- [x] Valid boxes are not clipped/reordered and inputs are not mutated.
- [x] No `face_recognition` or substitute detector/landmark model is required.
- [x] Clean binary-only install of the selected dlib wheel passes frontend smoke.

### Deviations and carried limits

- **Corpus size is unchanged from PR-1.** `tools/legacy/README.md`'s "Current
  fixture scope" section already explains why: the only additional face images
  vendored in this tree are `libs/dlib/examples/faces/*.jpg`, real
  identifiable people whose redistribution rights are unclear, and that section
  explicitly rules out assembling a corpus from them. The 30-image/100-face
  target `spec/PR-1.md` and `spec/INVESTIGATION.md` describe stays deferred to
  release hardening (PR-7), to be built from independently licensed images
  rather than by reusing unvetted vendored assets under schedule pressure.
- **Only Linux x86-64 CPython 3.13.15 was exercised.** The "per-supported-
  platform frontend parity" bullet in this file's Tests section is a
  cross-platform CI matrix concern; PR-3 carried the same limit for the ONNX
  runtime side, and PR-7 owns the actual multi-platform install/parity gate for
  both.
- **`FaceExtraction` carries landmarks, not just rectangles and chips.** The
  Work section's "small internal result containing rectangles and corresponding
  chip arrays" did not name landmarks explicitly, but the Tests section
  requires direct landmark-array equality against the oracle, and the
  alternative (reaching into `FaceFrontend`'s private predictor from the test
  module) would have broken the "keep backend objects out of the public API"
  boundary from the other direction. Landmarks are plain `[x, y]` int pairs,
  not a `dlib` type.

### Review follow-ups

An external review of this branch raised five findings. Four were reproduced
and fixed here; the fifth (corpus size) is addressed below rather than fixed,
because it is a scope decision, not a bug, and forcing it through in this pass
would have meant either reusing the unvetted vendored images this file already
rules out, or a second substantial fixture-generation change bundled into a
bug-fix pass.

1. **The "binary-only install passes frontend smoke" acceptance box was
   checked without a test that could fail on it (high).** The integration
   suite's child program (`tests/integration/test_package_resources.py`)
   imported `_inference`/`_models`/`_postprocess` against pre-extracted chips,
   never `_faces`, `dlib`, or `Pillow`; the offline binary-only install could
   have been missing a working `dlib-bin` entirely and this suite would still
   pass. Added `FRONTEND_CHILD_PROGRAM` and `_infer_frontend`, which decode a
   copied `example/test-image.jpg` with Pillow and run `FaceFrontend.extract`
   inside the same isolated, `--no-deps`, no-index, empty-`PATH` environment,
   then compare the resulting public dictionaries against the frozen oracle
   results (`test_offline_frontend_reproduces_the_frozen_results_from_a_raw_
   image`). This is now the only check that actually exercises the installed
   `dlib-bin`/Pillow wheels rather than pre-extracted chips.
2. **`parse_boxes()` accepted coordinates it did not actually validate
   (medium).** `[(0, 10**30, 10**30, 0)]` passed `_images.py`'s validation
   silently; only `FaceFrontend._resolve()`'s `dlib.rectangle()` call would
   eventually reject it, contradicting this file's Work section ("values
   representable by the backend"). `_images.py` cannot import `dlib` to ask its
   actual limit, and that limit is platform-dependent (C `long` is 64-bit on
   Linux/macOS, 32-bit on Windows, a supported platform), so `_as_integer` now
   rejects any coordinate outside signed-32-bit range directly, before `_faces
   .py` is ever reached. `_faces.py`'s own `TypeError`/`OverflowError` handling
   is kept as defense in depth for callers that construct boxes without going
   through `parse_boxes()`.
3. **The sdist could not run the test module it ships (medium).** `pyproject
   .toml`'s sdist selection carried `tests/` but neither `example/` nor
   `libs/`, while `tests/parity/test_frontend.py` decodes `example/test-
   image.jpg`/`test-image-2.jpg` and (for the no-face case)
   `libs/dlib/examples/faces/dogs.jpg` to exercise the real detector path,
   unlike the pre-extracted-chip parity tests PR-1/PR-3 added. Fixed by adding
   just those two example images (not all of `example/`, which also carries
   the 20 MiB legacy `.dat` duplicates under `example/models/`) to the sdist's
   `only-include`, and by making the three tests that touch `dogs.jpg` skip
   with a clear reason instead of failing when it is absent — `dogs.jpg` stays
   out of the sdist deliberately, since its own manifest entry says it "must
   not be redistributed separately." A new integration test,
   `test_source_distribution_carries_what_test_frontend_needs`, guards both
   halves of this fix (the two images are present; `example/models/` is not).
4. **Two claims in `tools/legacy/frontend-comparison.md` had no retained,
   re-runnable test (low).** The padding-precision and `get_face_chip`-route-
   equivalence claims were one-time interactive checks during development, not
   part of the test suite; a future `dlib-bin` version could silently falsify
   either and nothing would fail. Added
   `PaddingPrecisionAndRouteEquivalenceTests` to `tests/parity/test_frontend
   .py`, retaining both checks. Also added
   `test_explicit_boxes_never_invoke_the_detector`, a spy-based test proving
   the detector is actually skipped for explicit boxes rather than run and
   coincidentally agreeing (the existing test only checked output equality).
   Also fixed the report's own contradiction: it said `dlib-bin==20.0.1` was
   "the only version available" in the same sentence that named three
   available versions; it is the *newest* (`LATEST`) one, which is what was
   actually compared.
5. **Corpus size is unchanged from PR-1 (high, not fixed here).** This review
   confirmed the finding is accurate — 4 documents / 11 faces, same as PR-1 —
   and confirmed (by building `build/legacy-oracle-pr2` and running
   `tools/legacy/run_reference.py` against derived crops/rotations/rescales of
   the two already-licensed example images) that a licensing-safe expansion is
   feasible: transforms of images already covered by this repository's own
   license carry no new redistribution question, unlike the vendored
   `libs/dlib/examples/faces/*.jpg` set this file already rules out. That is a
   new fixture-generation change — new oracle runs, new golden JSON and binary
   artifacts, updates to `tools/legacy/run_reference.py`'s canonical-input
   allowlist and `tests/fixtures/legacy/manifest.json` — not a bug fix, so it
   was not bundled into this review-response pass. It remains open, tracked
   here and in `tools/legacy/README.md`'s "Current fixture scope" section.
