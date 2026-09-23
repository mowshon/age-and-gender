# PR-3 — Build the Python package, bundled models, and inference runtime

**Depends on:** PR-2 and PR-1's postprocessing contract.

**Consumes from PR-2:** `tools/conversion/artifacts/v1/`. The bundle manifest
carries a `runtime` block (`CPUExecutionProvider`, `ORT_DISABLE_ALL`,
single-threaded intra/inter-op). Build ONNX Runtime sessions from that block and
fail loudly if it is absent or different, rather than relying on runtime
defaults. Parity was established with those options and does not hold at the
other graph optimization levels; see `tools/conversion/README.md` for the
measured comparison.

**Risk:** medium.

**Deliverable:** installable Python package with independently tested chip-level
inference and packaged model resources.

## Goal

Replace project-owned native-extension packaging with a conventional Python
package and reusable ONNX sessions. Keep model preparation out of installation.

## Work

### Package scaffold

- Use the existing `venv/` for project development and invoke tooling through
  `venv/bin/python`; see `AGENTS.md` and the roadmap's environment section.
- Add PEP 517/518/621 `pyproject.toml` with Hatchling and explicit
  `src/age_and_gender/` wheel inclusion.
- Keep the distribution identity `age-and-gender` and import `age_and_gender`;
  underscores and hyphens normalize to the same PyPI project. Use a 2.0 prerelease
  version during migration because supported Python/platform/model-format
  expectations change.
- Start with CPython >=3.12 and the candidate dependencies in the investigation;
  validate exact versions before accepting constraints.
- Add `py.typed`, result TypedDicts, formatter/linter configuration, and a test
  configuration that excludes vendored upstream tests.
- Ensure the new PEP 517 build selects only the Python package. The old setup.py
  can remain temporarily as historical source, but must not participate in this
  build or install an identically named extension beside the package. Remove
  obsolete build files in PR-7 after parity passes.
- Keep conversion/oracle/test tools out of runtime wheel dependencies.

### Model resources

- Bundle `age-v1.onnx`, `gender-v1.onnx`, the five-point `.dat`, manifest, and
  required notices inside `src/age_and_gender/models/`.
- Include these resources in both wheel and source distribution. Building a
  source distribution should not fetch or convert models.
- Resolve with `importlib.resources`; respect extraction-context lifetime where
  a library requires a path. Do not assume `Path.cwd()` is the project root.
- Validate hashes once per load, together with manifest schema, task, graph
  signature, normalization contract, and labels.
- Expose internal resolution of bundled models, explicit manifest-based bundles,
  and the exact known legacy `.dat` hash mappings used by PR-5.
- Do not hash the entire model or reopen it on every prediction.
- Models are available offline immediately after package installation; no
  implicit model downloader is needed for the initial release.

### ONNX session lifecycle

- Create one CPU session per loaded age/gender model and reuse it.
- Construct sessions lazily on first model use, or when an explicit loader needs
  validation. Importing `age_and_gender` must not load weights or perform I/O
  beyond normal imports.
- Set provider, graph optimization level, and thread settings explicitly from
  the PR-2 validated configuration. Do not silently switch to GPU.
- Validate contiguous float32 `[N,3,H,W]` inputs and finite `[N,C]` probabilities.
- Return early for zero chips rather than submitting an empty batch.
- Keep session metadata immutable after successful initialization. A failed
  replacement leaves the old validated model/session available.

### Numeric processing

- Convert chip uint8 RGB to float32 before subtracting means, divide by exactly
  256, transpose NHWC to NCHW, and make contiguous only where required.
- Use the serialized float32 mean values recorded by the converter.
- Implement age reduction sequentially over 81 classes with a float32
  accumulator. Vectorizing over faces is fine; a NumPy dot product over classes
  is not automatically equivalent.
- Apply legacy half-away-from-zero rounding and float32 confidence multiplication
  followed by floor. Use a higher-precision `+0.5` when emulating nonnegative
  `lround`, avoiding an additional float32 rounding at that step.
- Map equal gender probabilities to female. Return built-in `int` and `str`
  values, not NumPy scalar types.

## Proposed files

```text
pyproject.toml
src/age_and_gender/__init__.py
src/age_and_gender/_models.py
src/age_and_gender/_inference.py
src/age_and_gender/_postprocess.py
src/age_and_gender/_types.py
src/age_and_gender/py.typed
src/age_and_gender/models/...
tests/unit/test_models.py
tests/unit/test_postprocess.py
tests/parity/test_chip_inference.py
tests/integration/test_package_resources.py
```

## Tests

- Actual ONNX inference on oracle chips, plus synthetic postprocessing boundaries.
- Known/unknown source hashes, corrupt resources, unsupported manifests, swapped
  age/gender graphs, wrong dimensions/dtypes, and invalid probability outputs.
- Multiple inference calls reuse sessions; reloading a model replaces one session
  atomically. No-face input does not construct/invoke neural sessions unnecessarily.
- Build wheel, install outside the checkout, change working directory, and run
  chip inference offline. Also build a wheel from the source distribution.
- Confirm imports and model lookup work without `libs/` or CMake on the runtime
  path. Source-tree-only tests are insufficient.

## Acceptance

- [ ] Project wheel is `py3-none-any` and has no project native extension.
- [ ] Packaged models are present and verified in wheel and source distribution.
- [ ] Chip-level parity and postprocessing boundary tests pass.
- [ ] Sessions/weights are not recreated per call.
- [ ] Model loading/inference performs no network access or conversion.
- [ ] Runtime dependencies exclude ONNX tooling, Torch, Caffe, and `face_recognition`.
- [ ] Public docstrings use concise Google-style Args/Returns/Raises sections
  only where useful; non-obvious float32 behavior has focused comments.

## Implementation status

Delivered on `PR-3`. The package lives in `src/age_and_gender/` and installs the
PR-2 artifacts as package data.

### What was built

- `pyproject.toml` with Hatchling, `requires-python >=3.12`, version `2.0.0a0`
  read from `__init__.py`, explicit wheel/sdist selection, and the pytest and
  Ruff configuration. The 1.x `setup.py` and `CMakeLists.txt` stay in the tree
  but take no part in the PEP 517 build; PR-7 deletes them.
- `tools/conversion/build_bundle.py` assembles `src/age_and_gender/models/` from
  `tools/conversion/artifacts/v1/` plus the pinned five-point landmark model. The
  package manifest is the conversion manifest with `"bundle_kind": "package"` and
  a new `shape_predictor` block added; a test asserts every carried-over field is
  unchanged, so the bundle cannot drift from PR-2's evidence.
- `_models.py` resolves the bundle through `importlib.resources`, validates the
  manifest (schema, bundle kind, runtime block, task, graph signature,
  normalization, labels, age class weights, landmark part count), verifies
  artifact size and SHA-256 once per load, hands out the landmark model through a
  context manager that owns the extraction lifetime, and exposes the legacy
  `.dat` digest-to-role mapping PR-5 needs.
- `_inference.py` builds one CPU session per model from the manifest's `runtime`
  block, lazily and then reused; validates the graph signature against the
  manifest and the execution provider against the declared one; prepares
  contiguous float32 `[N, 3, H, W]` inputs; returns early for zero chips without
  constructing a session; and replaces a model only after the replacement has
  loaded and validated.
- `_postprocess.py` implements the frozen arithmetic: sequential float32 age
  reduction over the 81 classes, float64 `+ 0.5` before the floor, float32
  percentage confidences, female on a tie, built-in `int`/`str` results, and the
  original `gender`/`age`/`face` key order.
- 118 new tests across `tests/unit/`, `tests/parity/test_chip_inference.py`, and
  `tests/integration/test_package_resources.py`. The suite is 137 tests and
  `ruff check .` is clean.

**Updated after PR-5.** The "verifies artifact size and SHA-256 once per
load" and "legacy `.dat` digest-to-role mapping" behavior described above was
later removed by explicit product direction: loading is validated
structurally (manifest schema, task, graph signature, normalization,
labels — all still enforced exactly as built here), not cryptographically.
`ModelBundle.model_bytes()`/`shape_predictor_file()` now just read the file;
`source_role()`/`source_digests`/`digest_bytes()`/`digest_file()` were
deleted once nothing used them for gating. See spec/PR-5.md's "Design change"
section for the full rationale and consequences. Everything else in this
section (session lifecycle, numeric processing, package scaffold) is
unaffected.

### Measured parity

Chip inference through the installed bundle, against all 11 frozen oracle faces:

| Stage | Result |
| --- | --- |
| Normalized input tensors | bit-identical to the frozen `.f32` artifacts |
| Age probabilities | max abs error 2.13e-06 (threshold `atol=1e-6, rtol=1e-4`) |
| Gender probabilities | max abs error 2.61e-08 |
| Age expectation | max abs error 1.53e-05 (threshold 1e-4) |
| Public results | exact on every image, including duplicate boxes |

### Dependency re-resolution

PR-2's evidence was gathered on NumPy 2.3.3 and ONNX Runtime 1.23.0. Both ends of
the proposed range were run on CPython 3.13.15:

| | numpy 2.3.3 / onnxruntime 1.23.0 | numpy 2.5.3 / onnxruntime 1.30.0 |
| --- | --- | --- |
| Full suite | 137 passed | 137 passed |
| Age / gender max probability error | 2.13e-06 / 2.61e-08 | 2.13e-06 / 2.61e-08 |
| Age expectation max error | 1.53e-05 | 1.53e-05 |
| Public results | exact | exact |

The two runs agree to the last bit, so the constraints are
`numpy>=2.3.3,<3` and `onnxruntime>=1.23.0,<1.31`: both bounds are measured
rather than inherited. The project environment is left on the newer pair.

### Acceptance against the list above

- [x] Project wheel is `py3-none-any` and has no project native extension.
- [x] Packaged models are present and verified in wheel and source distribution.
- [x] Chip-level parity and postprocessing boundary tests pass.
- [x] Sessions/weights are not recreated per call.
- [x] Model loading/inference performs no network access or conversion.
- [x] Runtime dependencies exclude ONNX tooling, Torch, Caffe, and `face_recognition`.
- [x] Public docstrings use concise Google-style Args/Returns/Raises sections
  only where useful; non-obvious float32 behavior has focused comments.

The offline claim is evidenced rather than proven absolutely: the integration
test builds with `--no-isolation`, installs with `--no-index`, and runs the child
process with `PIP_NO_INDEX=1` and an empty `PATH`, so no index, compiler, or
CMake is reachable. It does not assert at the socket level that nothing dials
out.

### Deviations and carried limits

- **Frontend dependencies are not declared yet.** `dlib-bin` and Pillow belong to
  PR-4's `_faces.py`/`_images.py`; declaring them here would put an untested,
  unused requirement into the wheel. PR-4 adds them to `dependencies` when the
  code that imports them lands. The public `AgeAndGender` class is PR-5's.
- **Only CPython 3.13.15 was exercised.** `requires-python >=3.12` follows the
  investigation's proposed baseline. The 3.12/3.13/3.14 matrix and the
  cross-platform wheel installation checks are PR-7's.
- **Long description deferred.** `readme` is intentionally not wired into the
  project metadata while the published README still documents the 1.x CMake
  extension; PR-7 rewrites it.
- **Formatter is configured, not enforced repo-wide.** `ruff format` was applied
  to the files this PR owns. Running it across PR-1 and PR-2's maintainer tooling
  would reformat 9 more files for no functional gain, so that is left to PR-7.
  Introducing `ruff check` did require small cosmetic edits in
  `tools/conversion/build_onnx.py`, `validate_conversion.py`,
  `tools/legacy/run_reference.py` and `tests/parity/test_converted_networks.py`
  (import order, line wrapping, one unused unpack); no conversion logic changed
  and the conversion parity suite still passes. `setup.py` and `example/` are
  excluded from lint as historical source that PR-5 and PR-7 replace.
- **The bundle's runtime block is required to equal the validated configuration.**
  A bundle naming a different provider, optimization level, or thread count is
  refused rather than run, because PR-2's parity evidence does not cover those
  settings. Widening that is PR-6's measured call.
- **Corpus limit is inherited.** Chip parity still rests on 11 faces over 3
  images, 8 distinct chips. PR-4 owns widening it; nothing here improves that
  evidence.

### Review follow-ups

An external review of this branch raised five findings. All five were
reproduced and fixed here; none changed a numeric result, and the parity table
above still holds.

1. **Reused bundles could return unverified bytes (medium).** `ModelBundle`
   remembered which filenames it had verified, but still reread the file on each
   call, so a bundle whose artifact changed after one good read handed the new
   bytes to a second session unchecked. Verification is now unconditional on
   every read, for both the ONNX graphs and the landmark model. Nothing is
   hashed per prediction — the read only happens when a model is loaded — and
   `tests/unit/test_inference.py` now asserts that directly instead of asserting
   a hash-once shortcut that was the bug.
2. **The batch dimension was not enforced (medium).** Session validation ignored
   dimension zero, so a graph with a fixed `[1, ...]` batch satisfied a manifest
   declaring `[N, ...]`, and the returned row count was never compared with the
   input row count. `_check_signature` now requires a symbolic batch on both
   tensors and requires the two symbols to match, and `run()` rejects any result
   whose row count differs from the chip count. Verified against a real ONNX
   graph rewritten to a literal batch of 1, not only against a stub.
3. **`build_bundle.py` could write an invalid bundle (low).** The landmark model
   was copied under its own basename with no collision check, so a correct
   predictor file named `age-v1.onnx` silently overwrote the age graph and the
   command still reported success. The builder now refuses a predictor whose
   basename collides with a graph or with `manifest.json`.
4. **The rounding regression test did not exercise its boundary (low).** The
   test used `np.float32(26.499998)`, where the float64 and float32 additions
   both give 26. A sweep of every half-boundary in the age range shows exactly
   one value where they differ — the float32 step below 0.5, which gives 0
   correctly and 1 with a float32 add — and the test now uses it. The
   implementation was already correct; only the guard was weak.
5. **Packaged notice hashes were recorded but never checked (low).** The
   manifest records a digest for both license notices and nothing verified them.
   The unit suite now checks both against the installed bundle, and the
   integration suite verifies every artifact *and* both notices against the
   manifest **inside** the built wheel and the built sdist, so the check covers
   what ships rather than what is in the working tree. Deliberately not added:
   runtime enforcement in `_models.py`. Hashing legal notices during model load
   would make a damaged or absent notice break inference, which is a worse
   failure than the one it prevents; CI is the right place for that check.
