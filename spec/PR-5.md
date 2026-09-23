# PR-5 — Implement the compatible public Python API

**Depends on:** PR-3 and PR-4.

**Risk:** medium.

**Deliverable:** end-to-end `AgeAndGender` implemented in Python with the legacy
return contract and a simpler default setup.

## Goal

Allow existing users to retain their import, predict calls, box format, and
result processing, while new users can run immediately with bundled models.

## Target public surface

```python
from age_and_gender import AgeAndGender, __version__

predictor = AgeAndGender()
predictor = AgeAndGender.from_model_dir("my-converted-model-bundle")

predictor.load_shape_predictor("shape_predictor_5_face_landmarks.dat")
predictor.load_dnn_gender_classifier("dnn_gender_classifier_v1.dat")
predictor.load_dnn_age_predictor("dnn_age_predictor_v1.dat")

results = predictor.predict(photo_numpy_array=image)
results = predictor.predict(image, [(top, right, bottom, left)])
```

**Revised during implementation** (explicit product direction: loading must not
gate on cryptographic hashes; a caller decides what model to load and where
they got it from). The two neural loader calls above now require an already
converted `.onnx` file — the original `dnn_age_predictor_v1.dat`/
`dnn_gender_classifier_v1.dat` are a proprietary dlib serialization with no
runtime Python reader, so there was never a way to load an arbitrary one of
those directly; only the three files this repository ships could ever satisfy
a hash-based lookup, which made that mechanism recognize exactly one fixed
pair of files, not really "loading a model" as a general capability. Convert
custom weights once with `tools/conversion` and load the resulting `.onnx`
file. `load_shape_predictor` is unaffected: dlib deserializes its file
natively, so it was never hash-gated, and still loads any compatible file.

## Work

1. Add `api.py` and export the class through `__init__.py`.
   - Default constructor selects bundled model metadata and loads heavyweight
     state lazily.
   - `from_model_dir()` accepts a validated manifest-based bundle and checks its
     completeness. Errors identify the missing/incompatible model.
   - Preserve the original `photo_numpy_array` parameter name for keyword
     compatibility. Default `face_bounding_boxes=None`; both `None` and `[]`
     retain auto-detection semantics.
   - Loader methods return `None`, accept strings/path-like values, and replace
     state only after successful validation.
2. Specify model-path compatibility precisely. Loading is validated
   **structurally, not cryptographically**: a model only has to satisfy the
   manifest's declared task/shape/dtype/normalization contract (checked
   against the actual loaded ONNX graph's own signature) and, for the neural
   networks, the real ONNX Runtime session it builds; there is no SHA-256 or
   byte-size check of the artifact against the manifest, so a caller can point
   this at any model they trust without regenerating a hash-locked manifest
   for it first.
   - Neural `.onnx`: require a sibling bundle manifest identifying and
     validating that file/task; do not guess normalization from output width.
     The manifest may record a `source` block for provenance, but it is
     optional and never checked against artifact bytes.
   - Raw neural `.dat`: refuse with a `ValueError` and a concise instruction to
     convert first with the documented maintainer workflow, then pass the
     resulting `.onnx` file. There is no runtime path from the proprietary
     dlib serialization to a loadable model, so this is not a hash gate to
     relax — an arbitrary `.dat` genuinely cannot be loaded directly.
   - Shape `.dat`: require a compatible five-point predictor and load it
     directly through dlib; any file that deserializes and produces five parts
     works, known or not.
   - Missing paths raise `FileNotFoundError`; bad shape/dtype/format/model
     metadata raises a documented `ValueError`; invalid call types raise
     `TypeError`. Wrap backend failures with model/task context and chaining.
3. Connect the pipeline:
   - Validate input and boxes.
   - Detect if needed, predict landmarks, extract individual chips.
   - Return `[]` immediately if no faces are available.
   - Run the two persistent sessions using the PR-2/PR-3 validated execution mode.
   - Apply exact legacy postprocessing and assemble results in face order.
4. Define state/concurrency:
   - Protect first-use initialization, prediction, and model replacement with an
     instance lock in the initial implementation.
   - Calls on one instance are safe but serialized; independent instances can
     be used by independent workers. Document memory tradeoffs briefly.
   - Failed loads leave previously usable state intact.
   - Do not share mutable buffers across calls without ownership/lifetime rules.
5. Add concise public type annotations and Google-style docstrings for the class,
   loaders, constructor factory, and prediction. Describe box order, inclusive
   endpoints, RGB requirement, defaults, return format, and meaningful exceptions.

## Required output

```python
{
    "gender": {"value": "female", "confidence": 100},
    "age": {"value": 26, "confidence": 84},
    "face": [419, 266, 506, 352],
}
```

No additional metadata keys in default results. Keep native Python scalar types
and the existing string labels. Model/version/provider diagnostics belong in
separate diagnostic output rather than every result record.

## Tests

- Exact end-to-end dictionaries against the full PR-1 corpus, including both
  default detection and caller-supplied boxes.
- Original examples' initialization sequence and positional/keyword predict
  calls, executed from outside the repository working directory.
- New zero-configuration constructor and explicit model bundle factory.
- Known, renamed, mismatched, missing, corrupt, and custom model file paths.
- No-face return, repeated predictions, successful/failed reloads, and concurrent
  same-instance calls under the documented locking behavior.
- Built-wheel integration using real models; unit mocks cannot establish parity.
- JSON serialization of returned values without a custom NumPy encoder.

## Proposed files

- `src/age_and_gender/api.py`, exports and concise result types.
- `tests/integration/test_api.py`, `test_legacy_loaders.py`.
- `tests/parity/test_end_to_end.py`.
- Migration usage draft and compatibility behavior table.

## Acceptance

- [ ] Old intended valid-input calls work for the shape predictor's `.dat` file
      directly, and for the two neural models once converted to `.onnx`
      (revised: the neural `.dat` files themselves are not directly loadable
      by any runtime, so "old calls work" no longer means passing them as-is).
- [ ] Default constructor works offline with installed package assets.
- [ ] Exact output parity passes; meaningful invalid-input errors are documented.
- [ ] A model that does not structurally match the declared task/shape/
      normalization is refused, not silently substituted or guessed at.
- [ ] No project C++ extension is imported or built by runtime calls.
- [ ] State replacement and same-instance concurrency follow the documented rules.

**Compatibility boundary:** arbitrary user-trained dlib networks are not generic
Python-loadable through these methods — the original C++ template network types
have no generic Python deserializer. Converted models must satisfy the explicit
architecture and preprocessing manifest, checked structurally against the loaded
graph; they do **not** need to match a specific recorded hash, so a caller can
load whatever compatible model they have, from wherever they obtained it.
Document this in the 2.0 migration guide.

## Implementation status

Delivered on `pr-5`. `src/age_and_gender/api.py` adds the public `AgeAndGender`
class and `src/age_and_gender/__init__.py` exports it alongside `__version__`.
It is entirely a thin composition layer: `predict()` calls `_images.py`'s
validation, `_faces.py`'s `FaceFrontend.extract()`, `_inference.py`'s
`InferenceEngine`, and `_postprocess.py`'s `face_predictions()` in the order
spec/INVESTIGATION.md's recommended pipeline lists; no numeric logic is
duplicated here.

### What was built

- `AgeAndGender()` resolves `bundled_models()` immediately but performs no
  further I/O until `predict()` or a loader is first called: the ONNX sessions
  and the dlib detector/landmark predictor are all built lazily on first use,
  verified directly by `tests/integration/test_api.py`'s
  `test_default_constructor_performs_no_heavy_io`.
- `AgeAndGender.from_model_dir(directory)` loads and fully validates an
  explicit bundle through `_models.load_bundle()`, which already checks
  schema, runtime options, both neural models, and the shape predictor in one
  pass; a missing or incomplete bundle raises `FileNotFoundError`/`ValueError`
  naming the missing section, with no extra validation code needed here.
- `predict(photo_numpy_array, face_bounding_boxes=None)` keeps the original
  parameter name and both-`None`-and-empty-mean-detect semantics, returns `[]`
  immediately for zero faces without touching the ONNX sessions, and preserves
  face order end to end.
- `load_shape_predictor`, `load_dnn_age_predictor`, and
  `load_dnn_gender_classifier` are preserved. The two neural loaders accept an
  `.onnx` path whose sibling `manifest.json` names that exact file for the
  requested task (reusing `load_bundle()` on the file's parent directory, so
  it gets the same full validation `from_model_dir()` does); a mismatched or
  absent manifest is refused, and any non-`.onnx` path (including the
  original `.dat` weights) is refused with a message pointing at
  `tools/conversion/README.md`. `load_shape_predictor` loads its file directly
  through dlib, unaffected by any of this — `_faces.py` gained a module-level
  `load_predictor()` for exactly that, since no conversion step exists for the
  landmark model. See "Design change" below for why the neural loaders no
  longer recognize the raw `.dat` files by hash.
- `_faces.py`'s `FaceFrontend` gained a `predictor=` constructor keyword and a
  `replace_predictor()` method (both additive; every existing PR-4 call site
  and test is unaffected). `AgeAndGender` uses them to keep the "loads
  heavyweight state lazily" contract for the landmark model specifically:
  calling `load_shape_predictor()` before the first `predict()` validates and
  stores the replacement without ever reading the bundle's own default
  landmark model, rather than loading the default only to discard it.
- A single `threading.Lock` serializes first-use initialization, the whole of
  `predict()` (not just lazy construction), and every loader's state swap, so
  concurrent calls on one instance queue rather than interleave; independent
  instances share nothing and are not serialized against each other.
- 39 new tests from the initial implementation, then further additions and
  rewrites from the review follow-ups and the design change below. The suite
  is 239 tests and `ruff check .` is clean (the total is lower than an
  earlier count in this file because the design change below deleted more
  hash-specific tests than it added).

### Design change: model loading is validated structurally, not cryptographically

After the review follow-ups below landed (which, at that point, had made
hash verification *stricter* — path-traversal-safe filenames, eager hash
checks in `from_model_dir()`, hash-first recognition regardless of
extension), explicit product direction reversed the underlying premise:
loading must not gate on matching a specific recorded SHA-256 at all. A
caller decides what model to load and where they obtained it; the package's
job is to check that what they gave it is structurally usable, not that it is
byte-identical to some known-good copy.

This changed two things, confirmed with the user before implementing (the
tradeoffs were genuinely open questions, not obvious calls):

1. **`_models.py` no longer verifies artifact SHA-256/byte-size against the
   manifest.** `ModelBundle.model_bytes()` and `shape_predictor_file()` now
   just read the file and let the real consumer — the ONNX Runtime session,
   or dlib's own deserializer — be the compatibility check. The manifest's
   `sha256`/`bytes`/`source` fields became informational-only (parsed if
   present, never required, never enforced); a hand-authored manifest for a
   custom model does not need to fabricate them.
2. **The two neural loaders no longer recognize the original `.dat` files by
   hash at all**, not even the three specific ones this package ships.
   Reflecting on it, that mechanism only ever worked for exactly those three
   files — no other `.dat` could ever satisfy it, since there is no runtime
   path from the proprietary dlib format to ONNX — so it was not really "load
   a model" as a general capability, just a recognizer for one fixed pair of
   files. `load_dnn_age_predictor`/`load_dnn_gender_classifier` now only
   accept an `.onnx` file with a sibling manifest; a `.dat` path is refused
   outright with conversion guidance. `load_shape_predictor` is untouched: it
   was never hash-gated, because dlib reads that file natively.

Consequence, stated plainly: this package no longer detects a corrupted or
substituted model artifact by comparing it to a known-good hash. A
structurally valid but wrong-weights ONNX file (or a shape predictor that
happens to still deserialize) will load without complaint. That is the
accepted tradeoff of this direction, not an oversight — structural validation
(task, tensor shape/dtype, normalization, five landmark parts) still catches
the large majority of realistic mistakes (wrong task, wrong chip size,
resized/retrained-for-different-input model), just not silent bit-level
corruption or a swapped-but-structurally-identical file.

Code changes: `src/age_and_gender/_models.py` (verification removed,
`source_role`/`source_digests`/`digest_bytes`/`digest_file` deleted as dead
code once nothing used them for gating), `src/age_and_gender/api.py`
(`_load_neural()` simplified to `.onnx`-only; `from_model_dir()` now eagerly
builds real ONNX sessions and the real frontend instead of hash-checking
bytes, so "checks its completeness" is satisfied by actually loading
everything once, up front). Test changes: `tests/unit/test_models.py` (hash-
mismatch tests replaced with tests proving bytes are returned *even when*
corrupted/truncated, plus new tests that a manifest without `sha256`/`bytes`/
`source` still loads), `tests/unit/test_inference.py` (the "weights are not
re-read" test now spies on `model_bytes()` instead of the deleted
`digest_bytes`), `tests/integration/test_legacy_loaders.py` (the whole
`NeuralLoaderTests` class rewritten around `.onnx`+manifest, including a new
test that a *renamed* `.onnx` — identical bytes, different name — is now
correctly refused, the inverse of the old renamed-`.dat` test), and
`tests/integration/test_package_resources.py` (the built-wheel check now
builds an explicit `.onnx` bundle from `src/age_and_gender/models/` instead
of copying `example/models/`'s `.dat` files, which incidentally also fixed
review finding 7 below more simply than the original patch did, since
`src/age_and_gender/models/` — unlike `example/models/` — always ships in
the sdist).

### Compatibility behavior table

| Legacy call | Behavior here |
| --- | --- |
| `AgeAndGender()` | Uses the models installed with the package; no downloads, no `.dat` paths required. |
| `.load_shape_predictor(path)` | Loads `path` directly through dlib; any compatible five-point model works. |
| `.load_dnn_age_predictor(path)` / `.load_dnn_gender_classifier(path)` with a `.dat` path | Always refused: the original dlib weights have no runtime ONNX conversion path. `ValueError` points at `tools/conversion/README.md`. |
| same, with an `.onnx` path | Accepted only with a sibling `manifest.json` (built by `tools/conversion/build_bundle.py`, or hand-authored in the same schema) naming that exact file for that task. Not required to match any recorded hash — structural validation only. |
| `.predict(photo_numpy_array, face_bounding_boxes=[(top, right, bottom, left), ...])` | Unchanged: same keyword name, same box order and inclusive endpoints, `None`/`[]` both mean auto-detect. |
| Result dictionaries | Byte-for-byte the same shape and key order as 1.x, exact on the frozen corpus. |

### Acceptance against the list above

- [x] Old intended valid-input calls work for the shape predictor's `.dat`
      file directly, and for the two neural models once converted to `.onnx`.
- [x] Default constructor works offline with installed package assets.
- [x] Exact output parity passes; meaningful invalid-input errors are documented.
- [x] A model that does not structurally match the declared task/shape/
      normalization is refused, not silently substituted or guessed at.
- [x] No project C++ extension is imported or built by runtime calls.
- [x] State replacement and same-instance concurrency follow the documented rules.

### Deviations and carried limits

- **No hash-based recognition of the original `.dat` neural weights.**
  Superseded by the design change above: `load_dnn_age_predictor`/
  `load_dnn_gender_classifier` accept only `.onnx` files now, so the earlier
  "known/renamed/mismatched-hash" behavior this section used to describe no
  longer exists. Kept here as a pointer for anyone looking for it.
- **Explicit `.onnx` replacement requires a full package-kind manifest
  alongside it**, i.e. the same layout `from_model_dir()` consumes wholesale,
  even when only one task is being replaced. A manifest naming just one model
  is not a format `_models.py` parses; building one for the single-model case
  was out of scope here and did not seem to be what the spec's "sibling bundle
  manifest identifying and validating that file/task" was asking for.
- **No separate migration-guide document was added.** The compatibility table
  above covers the behavior differences; `README.md` still documents the 1.x
  CMake extension pending PR-7's rewrite, so a second, temporary user-facing
  document did not seem worth adding for one release cycle.
- **Platform/corpus limits are unchanged from PR-3/PR-4.** Only Linux x86-64
  CPython 3.13.15 was exercised; the cross-platform matrix and the wider
  fixture corpus stay PR-7's.

### Review follow-ups

**Superseded note:** findings 1 and 2 below describe fixes made to a
hash-verification design that "Design change" above later removed entirely.
They are kept as an accurate record of what was found and fixed *at that
point in the branch's history* — the specific mechanisms they fixed
(`_check_bundle_is_complete()`'s hash reads, hash-first `.dat` recognition)
no longer exist in the form described, though the underlying properties they
established still hold today by other means: `from_model_dir()` still fails
immediately rather than lazily (now via eager real loading, not hash
verification), and a `.dat` file is still never silently accepted for a
neural loader (now because `.dat` is unconditionally refused there, not
because its hash was unrecognized). Findings 3 and 6 are unaffected by the
design change — path-traversal-safe filenames and `age_weights` type
validation are both structural checks, not hash checks, and remain exactly
as described. Findings 4 and 5 remain accurate as written; finding 7's actual
current fix ended up simpler than described (see the design-change section's
last paragraph).

An external review of this branch raised seven findings. Six were reproduced
and fixed here; the seventh (wrapping prediction-time backend errors) is
addressed below rather than fixed, because on inspection it reads spec/PR-5
.md's own requirement more broadly than the Work section actually states it.

1. **`from_model_dir()` accepted incomplete or corrupt bundles (high).**
   `load_bundle()` only parses and validates the manifest JSON; artifact
   bytes are read and hash-checked lazily, the first time a model is actually
   loaded (by design — see `_models.py`'s module docstring and `AgeAndGender`
   's own zero-configuration constructor). `from_model_dir()` inherited that
   laziness, so deleting `age-v1.onnx` from a copied bundle directory still
   let `from_model_dir()` succeed; the failure only surfaced on the first
   `predict()` call. Confirmed by deleting the artifact and constructing
   directly. `from_model_dir()` now calls a new `_check_bundle_is_complete()`
   right after `load_bundle()`, which reads and hash-checks both ONNX graphs
   and the shape predictor and discards the bytes, so a missing or corrupt
   artifact fails at `from_model_dir()` itself. This is a deliberate
   asymmetry with `AgeAndGender()`, which stays lazy: an explicitly named
   bundle is meant to fail immediately, not deep inside a later prediction.
   `test_from_model_dir_missing_artifact_fails_immediately_not_lazily` and
   `test_from_model_dir_corrupt_artifact_fails_immediately_not_lazily` in
   `tests/integration/test_api.py` cover both cases.
2. **Legacy model recognition branched on the `.onnx` suffix before hashing
   (medium).** `_load_neural()` picked the explicit-`.onnx`/sibling-manifest
   path or the hash-lookup path by extension first, so a byte-identical
   legacy `.dat` renamed with a `.onnx` extension was routed to the manifest
   path and rejected for lacking a `manifest.json` neighbor — contradicting
   "identify by SHA-256, irrespective of filename." Confirmed by renaming
   `dnn_age_predictor_v1.dat` to `renamed.onnx` and loading it.
   `_load_neural()` now always hashes the file and checks it against
   `bundled_models().source_role()` first, regardless of extension; only an
   unrecognized hash falls through to the `.onnx` sibling-manifest path.
   `test_recognition_is_independent_of_extension_not_just_of_name` covers the
   renamed-with-`.onnx`-extension case the original tests missed.
3. **Model manifests could escape the supplied bundle directory (medium).**
   `_artifact()` accepted a manifest's `filename` field with no check for `..`
   components or absolute paths before it was joined onto the bundle root, so
   a manifest naming `../outside.onnx` (or `/etc/passwd`) would happily read
   that file — breaking `from_model_dir()`'s documented "backed entirely by
   this directory" guarantee, and a real concern for a bundle manifest that
   might be shared or downloaded rather than self-authored. Confirmed by
   constructing exactly that manifest and reading the resulting artifact.
   `_artifact()` now rejects any filename containing `/` or `\`, or equal to
   `.`/`..`/empty, via a new `_check_flat_filename()`; every real manifest
   (the shipped one and everything `build_bundle.py` produces) already only
   ever uses flat names, so this narrows nothing legitimate.
   `test_path_traversal_in_artifact_filename_is_refused` and
   `test_absolute_artifact_filename_is_refused` in
   `tests/unit/test_models.py` cover it.
4. **Backend execution errors lack model/task context (medium) — investigated,
   not changed.** The finding is accurate as stated: `NeuralNetwork.run()`
   does not wrap `session.run()`, and `FaceFrontend`'s landmark/chip calls are
   similarly unwrapped. But re-reading spec/PR-5.md's Work section, "Wrap
   backend failures with model/task context and chaining" is the closing
   sentence of item 2, "Specify model-path compatibility precisely" — a bullet
   list entirely about the three loader methods (missing paths, malformed
   metadata, unknown hashes). Read in that context it describes *loading*
   failures, which are already wrapped: `_create_session()` chains ONNX
   Runtime session-construction errors with `f"{filename} could not be loaded
   as the {task} model"`, and `_faces.py`'s `_load_predictor()`/
   `load_predictor()` chain dlib deserialization errors the same way. Item 3,
   "Connect the pipeline" (the section that actually describes `predict()`'s
   execution), says nothing about wrapping backend exceptions. Extending that
   to prediction-time `session.run()`/dlib calls would mean changing
   `_inference.py` and `_faces.py`'s tested, PR-3/PR-4-delivered execution
   paths for behavior this file does not ask for, so it was left alone; the
   underlying exception types (`onnxruntime`'s own exceptions, dlib's) still
   propagate, they are just not re-wrapped.
5. **Successful custom-model tests could not distinguish real replacement
   from a no-op (medium).** `tests/bundles.py`'s `full_bundle()` copies the
   installed models byte for byte, so a test that only checks `predict()`'s
   output against the golden results cannot tell "the custom bundle was
   actually used" apart from "the loader silently kept the default and the
   output matched anyway because the bytes happen to be identical." This was
   true of `test_from_model_dir_uses_only_the_given_bundle` and
   `test_explicit_onnx_with_sibling_manifest_is_accepted`, and the explicit-
   `.onnx` success path was only exercised for the age model. Fixed by
   asserting `.bundle.origin` (for `from_model_dir()`, where the origin
   genuinely changes to the temp directory) and network object identity
   (for the two legacy-hash loaders, where the origin stays
   `"age_and_gender.models"` either way, so a fresh `NeuralNetwork` object
   actually being swapped in is the observable proof) directly on
   `predictor._engine.age`/`.gender`/`predictor._frontend`, and by adding
   `test_explicit_onnx_with_sibling_manifest_is_accepted_for_gender_too`.
6. **Malformed `age_weights` entries raised raw `TypeError`, not the
   documented `ValueError` (low).** `_parse_age_weights()` called `float()` on
   every element without checking its type first; a JSON `null` (or any other
   non-numeric value) inside an otherwise correctly-sized 81-element list
   raised an uncaught `TypeError` instead of the `ValueError` this file
   documents for malformed model metadata. Confirmed by setting one element to
   `null` and calling `from_model_dir()`. Fixed by adding the same
   `isinstance(value, (int, float)) and not isinstance(value, bool)` guard
   `_parse_normalization()` already uses for `means`, checked before any
   `float()` conversion. `test_null_age_class_weight_is_refused_as_a_value_
   error` and `test_boolean_age_class_weight_is_refused` cover it.
7. **The shipped integration test module could not run from an extracted
   sdist (low).** The new `_infer_api()` (added for the "built-wheel
   integration using real models" acceptance item) unconditionally copied
   `example/models/*.dat` into the isolated sandbox, but that directory is
   deliberately excluded from the sdist (PR-4's `only-include` policy,
   guarded by `test_source_distribution_carries_what_test_frontend_needs`).
   Running this test module from an extracted sdist would therefore fail
   inside `setUpClass`, before any test method runs. Reproduced by building
   the sdist, extracting it, and running the two dependent test methods from
   inside the extraction. Fixed by making the `.dat` copy conditional on the
   files actually being present, passing zero or three trailing paths to the
   child program accordingly, and having `API_CHILD_PROGRAM` run only the
   zero-configuration half when none are given; the legacy-loader test method
   skips (rather than fails) when its half did not run. Re-verified against
   an extracted sdist: the zero-config check passes and the legacy-loader
   check skips cleanly instead of crashing the class.
