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
