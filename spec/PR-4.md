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
