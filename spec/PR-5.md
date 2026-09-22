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

The three `.dat` loader calls above remain supported for the supplied model
hashes. Neural `.dat` loading selects an already converted bundled equivalent;
the shape predictor loads directly. Explain that distinction in API docs.

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
2. Specify model-path compatibility precisely:
   - Known age/gender `.dat`: identify by SHA-256, irrespective of filename, then
     resolve the corresponding installed ONNX model without network access.
   - Renamed files with identical bytes work; a familiar filename with different
     bytes does not select defaults.
   - Unknown neural `.dat`: raise `ValueError` with the source hash and a concise
     instruction to use the documented maintainer conversion workflow.
   - Explicit neural `.onnx`: require a sibling bundle manifest identifying and
     validating that file/task; do not guess normalization from output width.
   - Shape `.dat`: require a compatible five-point predictor and load it directly.
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

- [ ] Old intended valid-input calls work for the three supplied model files.
- [ ] Default constructor works offline with installed package assets.
- [ ] Exact output parity passes; meaningful invalid-input errors are documented.
- [ ] Unknown `.dat` weights cannot silently fall back to bundled predictions.
- [ ] No project C++ extension is imported or built by runtime calls.
- [ ] State replacement and same-instance concurrency follow the documented rules.

**Compatibility boundary:** arbitrary user-trained dlib networks are not generic
Python-loadable through these methods. Converted models must satisfy the explicit
architecture and preprocessing manifest; document this in the 2.0 migration guide.
