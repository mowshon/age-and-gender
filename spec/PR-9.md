# PR-9 — Cleanup: remove legacy tooling and internal-plan references

**Depends on:** PR-8 merged. Commit the pending `example/` cleanup in the
`pr-8` working tree first, so this PR's diff contains only the cleanup.

**Risk:** low for users. `predict()` results, the public API, the bundled model
files, and the dependency pins do not change. The only runtime edits are two
error messages, comments, type annotations, and one unreachable code path.
Medium for maintainers: the ONNX conversion pipeline and the C++ reference
oracle leave the tree and can only be recovered from Git history.

**Deliverable:** a repository that holds only what `src/age_and_gender` needs
to be built, tested, and released. It has no C/C++ sources, no vendored dlib,
no conversion or oracle tooling, no benchmark harness, no local source models,
and no references to `spec/` outside `spec/`.

## Goal

The migration replaced the compiled 1.x extension with the same trained
networks run through ONNX Runtime. That work is finished and verified. The
tooling that produced and proved the conversion, together with the benchmark
harness used to justify the performance work, now makes up 2,397 of the 2,494
tracked files. The package uses none of it.

PR-9 removes that tooling and keeps the evidence that still matters in the test
suite. It also rewrites public docs and code comments to describe the package
as it is, not the internal plan that produced it.

## Findings

Investigated on 2026-09-24 against `63cc4fc` (branch `pr-8`), using `venv/`.

### 1. The conversion pipeline cannot stand on its own

`build_onnx.py` does not read the `.dat` files itself. It reads XML written by a
C++ exporter, and the exporter builds only against the vendored dlib source:

```text
models/dnn_{age_predictor,gender_classifier}_v1.dat    root models/, Git-ignored
  → age_and_gender_export_dlib                          tools/conversion/export_dlib.cpp
      compiled with tools/vendor/dlib/.../source.cpp    vendored dlib 19.20, 35 MB
      network types from tools/network_definitions.h    shared with tools/legacy/oracle.cpp
  → XML holding every layer parameter
  → build_onnx.py            → age-v1.onnx, gender-v1.onnx
  → validate_conversion.py   also needs the C++ probe_dlib (same vendored dlib)
  → build_bundle.py          → src/age_and_gender/models/
```

The `dlib-bin` Python wheel cannot replace the exporter. Its bindings expose a
fixed set of built-in models (the HOG detector, shape predictors, and two
specific CNNs), not general `dnn` network deserialization. A dlib DNN `.dat`
file can only be read by C++ code compiled with the exact network type.

Keeping `tools/conversion/` would therefore mean keeping `tools/vendor/`,
`tools/network_definitions.h`, and a CMake toolchain. **Decision: remove all of
`tools/`.**

### 2. The ONNX models already reproduce the `.dat` results

- The shipped `src/age_and_gender/models/age-v1.onnx` and `gender-v1.onnx` are
  byte-identical to `tools/conversion/artifacts/v1/` (SHA-256 `4f9f0075…` and
  `d4c4116c…`). That bundle's `conversion-report.json` records
  `"passed": true` for logits, probabilities, stage tensors, and public results
  against dlib.
- `tests/fixtures/legacy/` holds the frozen outputs of the original C++
  pipeline: chips, input tensors, probabilities, age expectations, and public
  result dictionaries. Three test files check the installed package against
  them, and none of them imports anything from `tools/`:

| Test file | What it enforces |
| --- | --- |
| `tests/parity/test_chip_inference.py` | Frozen chips through the shipped ONNX: bit-identical input tensors, probabilities within `atol=1e-6, rtol=1e-4`, age expectation within `1e-4`, exact public results |
| `tests/parity/test_end_to_end.py` | Raw image through `AgeAndGender.predict()`: dictionaries exactly equal to the 1.x extension's |
| `tests/parity/test_frontend.py` | `dlib-bin` detector boxes, landmarks, and both chip sizes, bit-exact |

- Baseline: `venv/bin/python -m pytest` gives **273 passed**. The three files
  above alone give 23 passed.

The claim that the ONNX models work exactly like the `.dat` files is therefore
established on the frozen corpus, and CI keeps enforcing it after `tools/` is
gone.

One capability is lost: re-running the conversion or regenerating the fixtures
now requires checking out an older commit. That is accepted, and the PR
description records the last commit that contains `tools/`.

### 3. Root `models/` is only a tooling input

Root `models/` is untracked (ignored by `/models`) and holds 20 MB: the three
original dlib-models downloads. Only `tools/conversion/` and `tools/legacy/`
read it. The package ships its own copy of the landmark model (same SHA-256,
`c4b1e980…`) and the two converted networks. Nothing in `src/`, `tests/`,
`benchmarks/`, `example/`, or CI refers to root `models/`.

### 4. Type check

`mypy` on `src/age_and_gender` with `--ignore-missing-imports` (dlib ships no
stubs) reports 2 errors. `--strict` adds 3 more. All of them are annotation
defects, not runtime bugs; §4 of the Work section lists them.

## Work

### 1. Delete

| Path | Contents | Why it can go |
| --- | --- | --- |
| `tools/vendor/` | Vendored dlib 19.20: 2,157 files, 35 MB | Only compiled by the C++ tools below |
| `tools/legacy/` | C++ reference oracle, reference-freezing scripts, frozen 1.x extension source with pybind11 2.5: 199 files | Its output, the frozen fixtures, stays in `tests/` |
| `tools/conversion/` | C++ exporter and probes, `build_onnx.py`, `validate_conversion.py`, `build_bundle.py`, the PR-8 NumPy shape-predictor prototype, `artifacts/`: 30 files, 18 MB | The shipped ONNX files are its final output |
| `tools/network_definitions.h` | dlib network types for the C++ tools | Only used by the C++ tools |
| `tools/ci/` | CI smoke test | Moved, see §2 |
| `tests/parity/test_converted_networks.py` | 9 tests of `tools/conversion/artifacts/v1` through `tools.conversion` | Runtime parity is covered by the tests in Findings §2 |
| `tests/parity/test_numpy_shape_predictor_feasibility.py` | 10 tests of the PR-8 prototype | Prototype removed |
| `tests/fixtures/legacy/dogs.golden.json` | No-face golden whose image is `tools/vendor/dlib/examples/faces/dogs.jpg` | The image cannot be redistributed on its own; a synthetic no-face image replaces it (§3) |
| `tests/fixtures/legacy/stability-linux-x86_64.json` | Oracle run-to-run comparison | Only needed to regenerate goldens with the oracle; nothing reads it |
| `benchmarks/` | Pipeline and concurrency benchmarks, the legacy-baseline script, the internal PR-6 report, and captured results: 9 files | Not needed to build, test, or run the package. Two of its scripts also depend on `tools/` or `dogs.jpg` |

Also delete these local, untracked directories: `rm -rf models/ build/`.
`build/` holds only CMake output from the removed tools (`conversion/`,
`legacy-oracle*/`, and `reference/`).

### 2. Keep the CI smoke test, outside `tools/`

- `git mv tools/ci/smoke_install.py .github/scripts/smoke_install.py`. Only the
  workflows use it, and `REPO_ROOT = parents[2]` still resolves to the
  repository root.
- Update its four invocations: three in `ci.yml` (`install-smoke` twice,
  `glibc-floor` once) and one in `release.yml` (`verify-testpypi`).
- Every remaining corpus image is in the checkout, so a missing input becomes a
  failure instead of a skip. Raise the `--require-corpus` default to the
  corpus size (3 documents).
- Remove the `spec/PR-7.md` quote from its docstring.

### 3. Tests that depended on `tools/`

- `tests/bundles.py`: delete `CONVERSION_BUNDLE`.
- `tests/unit/test_models.py`:
  - Delete `test_package_manifest_agrees_with_the_conversion_bundle`.
  - Rewrite `test_conversion_bundle_is_not_a_package_bundle` as
    `test_unknown_bundle_kind_is_refused`, using
    `manifest_only(self.tmp / "kind", lambda m: m.__setitem__("bundle_kind", "conversion"))`
    and asserting that `"bundle_kind"` is in the message.
- `tests/integration/test_legacy_loaders.py`: rename
  `test_raw_dat_weights_are_refused_with_conversion_guidance` to
  `test_raw_dat_weights_are_refused`. Assert that the new message (§4) mentions
  `manifest.json` rather than `tools/conversion`.
- No-face coverage: `tests/integration/test_api.py::test_no_face_image_returns_empty_list`
  and `tests/parity/test_frontend.py::test_no_face_image_returns_empty` should
  build a synthetic image in the test and assert `[]`, as
  `tests/unit/test_faces.py` already does with `np.zeros((64, 64, 3), np.uint8)`.
  Use a larger image (for example 480×640) so the detector scans more than one
  window. Remove `NO_FACE_IMAGE` and the skip branches.
- Corpus size: `test_corpus_is_the_frozen_one` in `test_chip_inference.py` and
  `test_end_to_end.py` now expects **3 documents** and still **11 faces**.
- Remove the `image_available()` helpers and their skips from
  `test_frontend.py` and `test_end_to_end.py`. Every remaining golden's source
  image (`example/test-image.jpg`, `example/test-image-2.jpg`) is in both the
  repository and the sdist, so a missing image is now a failure.
- `tests/fixtures/legacy/manifest.json`:
  - Drop the `dogs.jpg` input and the `stability_report` key, and set `status`
    to `"frozen"`.
  - Reword `required_cases.no_face` to "synthetic uniform images in the tests".
  - Reword `representative_30_image_100_face_corpus` to "not collected".
  - After these edits the file names no removed path and no PR.
- Leave the three remaining `*.golden.json` files byte-for-byte. Their `oracle`
  blocks name `tools/legacy/...` as the provenance of the frozen values, which
  is still accurate history. Rewriting them would make them something other
  than the recorded oracle output.
- `tests/integration/test_package_resources.py::test_no_checkout_source_is_on_the_installed_runtime_path`:
  drop `"tools"` from the forbidden prefixes and from its docstring.

Expected result: 253 tests (273 − 9 − 10 − 1) and **0 skipped**, both locally
and in every CI cell. The skip lane that ran only when `onnx` was installed no
longer exists.

### 4. Runtime source (`src/age_and_gender`)

Nothing changes in behavior except the text of two error messages.

**Error messages that point at removed tooling**

- `api.py`, `_load_neural()`: replace "Convert them first with tools/conversion
  (see tools/conversion/README.md) and pass the resulting .onnx file, alongside
  its manifest.json" with guidance that stands on its own. For example:
  `"{path}: only .onnx files are accepted; the original dlib .dat weights cannot
  be loaded. Pass an .onnx model whose sibling manifest.json names it for this
  task."`
- `_models.py`, `_check_bundle_identity()`: replace "build one with
  tools/conversion/build_bundle.py" with the expected value:
  `"manifest bundle_kind {kind!r} is not supported; expected 'package'"`.

**Shipped manifest**

In `src/age_and_gender/models/manifest.json`, remove two provenance fields that
point at deleted files: `converter_revision` (a hash of the deleted converter
sources) and `dlib.source` (`"tools/vendor/dlib"`). Keep `dlib.version`, the
`source` hashes of the original `.dat` downloads, and every field the runtime
validates. Do not touch the `.onnx` and `.dat` files.

**Comments and docstrings** (rules in §6)

| Location | Now | Change |
| --- | --- | --- |
| `api.py` module docstring | 31 lines, with conversion-tooling paths | About 10 lines: what the class ties together, plus the loader boundary (`.onnx` and a manifest for the neural models, `.dat` for the landmark model) |
| `api.py`, `load_dnn_*` docstrings | "see the module docstring for why and how to convert it" | State that the `.dat` is not accepted; no conversion instructions |
| `_faces.py` module docstring | Cites spec/PR-4.md's "known batching trap" | Point to `tests/parity/test_frontend.py::BatchingTrapRegressionTests` |
| `_inference.py`, `DEFAULT_MAX_BATCH_SIZE` comment | 16 lines citing spec/PR-6.md and `validate_conversion.py` | 2–3 lines: it bounds the size of one ONNX Runtime call, and chunking never changes results because rows are independent (`tests/parity/test_batching.py`) |
| `_inference.py`, `NeuralNetwork.probabilities` docstring | 23 lines | Summary, `Args`/`Returns`/`Raises`, and one sentence about chunking |
| `_models.py` module docstring | 23 lines | About 10 lines: what a bundle is, and that validation is structural, not hash-based |
| `_models.py`, `SUPPORTED_RUNTIME` comment | Points to `tools/conversion/README.md` | Give the reasons inline. Parity was verified only with graph optimization disabled. Execution stays single-threaded because more intra-op threads were measured to reduce throughput when several instances run at once |
| `_models.py`: `NeuralModelSpec`, `ModelBundle`, `model_bytes`, `shape_predictor_file`, `_artifact`, `_check_flat_filename`, `_check_contained`, `_as_list_or_none` | Multi-paragraph docstrings that each restate "structural, not cryptographic" | Say it once in the module docstring; each function keeps one or two lines of why |

**Types** (Findings §4)

| Location | Problem | Fix |
| --- | --- | --- |
| `_models.py:444-448` and `:609-613` | `source_filename`/`source_sha256` are inferred as `str`, then assigned `None` (mypy `assignment`) | Annotate them as `str \| None` before the branch |
| `_inference.py:210`, `NeuralNetwork.run` | Returns ONNX Runtime output typed `Any` | Annotate the local and the return as `NDArray[np.float32]`; `_check_probabilities` already enforces the dtype |
| `_postprocess.py:50`, `age_expectations` | Returns `Any` from `np.multiply(..., dtype=np.float32)` | Annotate `estimate: NDArray[np.float32]` |
| `_faces.py:206`, `_probe_parts` | Returns dlib's untyped `num_parts` as `int` | `return int(...)` |
| `api.py:164`, `from_model_dir` | Annotated `-> AgeAndGender` but builds `cls.__new__(cls)` | `-> Self` (`typing.Self`) |
| Array-returning helpers | Bare `np.ndarray` where the dtype is guaranteed | `NDArray[np.uint8]` for images and chips (`as_rgb_array`, `FaceExtraction.*_chip`); `NDArray[np.float32]` for tensors and probabilities (`prepare_batch`, `NeuralNetwork.probabilities`/`run`, `age_expectations`) |

The public result types were checked against what
`_postprocess.face_predictions()` builds, and they are correct:

- Every `confidence` and the age `value` is a built-in `int`, not a NumPy
  scalar. The gender `value` is a `str`.
- `face` is a `list[int]`.
- Keys are in the order gender, age, face.

`_types.py` stays as it is.

**Dead code**

`_inference.GRAPH_OPTIMIZATION_LEVELS` maps four levels, but
`_models._parse_runtime()` rejects every runtime block except `ORT_DISABLE_ALL`.
Three of the entries, and the `level is None` branch in `_create_session()`,
can never run. Replace them with the single supported level. The rejection
itself stays covered by `tests/unit/test_models.py`, which refuses a manifest
with `ORT_ENABLE_ALL`.

### 5. Configuration

**`pyproject.toml`**

- `dependencies` comment: drop spec/PR-3, spec/PR-4, and
  `tools/legacy/frontend-comparison.md`. Keep the reason for the exact
  `dlib-bin` pin: boxes, landmarks, and chips are bit-exact with this wheel
  (`tests/parity/test_frontend.py`), so re-run parity before widening it.
- `[tool.hatch.build.targets.wheel]` comment: drop the remark about the 1.x
  `setup.py` and CMake sources, which no longer exist.
- `[tool.hatch.build.targets.sdist]` comment: drop the `tools/vendor/` and
  `dogs.jpg` remarks.
- `[tool.pytest.ini_options]`: drop the comment about vendored sources and
  remove `"tools"` from `norecursedirs`.
- `[tool.ruff]`:
  - Set `src = ["src", "tests"]`.
  - Remove `tools/vendor` and `tools/legacy/original-extension` from
    `extend-exclude`, along with their comment.
  - Replace the `"tools/**"` per-file ignore with `".github/scripts/**"`, and
    remove the `"benchmarks/**"` one.

**`.gitignore`**

Remove the CMake entries (`CMakeLists.txt.user` through `_deps`) and `/models`.
The result:

```gitignore
.idea
build
dist/
*.egg-info/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
/venv/
```

**`.github/workflows/ci.yml`**

- Delete the `conversion-parity` job, which only ran the deleted
  `test_converted_networks.py`.
- Delete the `legacy-tooling` job, which builds the deleted C++ tools.
- Change the smoke-test path to `.github/scripts/smoke_install.py` (three
  places).
- Comments:
  - Drop the spec/PR-1.md and spec/PR-7.md quotes.
  - Update the `workflow_call` comment, which lists "conversion parity" and
    "the maintainer-tooling build".
  - Update the `test` job comment about skipped conversion checks.

**`.github/workflows/release.yml`**

- Change the smoke-test path (one place).
- Remove "and maintainer-tooling check" and the spec/PR-7.md quote from the
  comments.

### 6. Documentation and comments outside `spec/`

These rules apply to every comment, docstring, and Markdown file outside
`spec/`:

- Do not reference `spec/`, PR numbers, or the migration plan. State the reason
  itself, or point to the test that enforces it.
- Do not reference removed paths (`tools/`, root `models/`, `libs/`,
  `dogs.jpg`).
- Comments explain why: the compatibility details a reader could otherwise
  break, such as float32 accumulation order, individual chip extraction,
  inclusive rectangles, and the TRBL box order. They do not narrate how the
  code got here.
- Docstrings are Google style: a one-line summary, then `Args`/`Returns`/`Raises`
  where they add information. Keep module docstrings to about 10 lines.

Per file:

- **`README.md`**
  - *Supported platforms*: drop "(see `spec/INVESTIGATION.md` …)" and the
    sentence that links PR-8.
  - *Bundled and custom models*: replace the paragraph that starts "Converting
    a different trained network is not currently automated". The new text says
    the bundled ONNX files are the original weights, converted once. It
    describes a custom bundle as an `.onnx` graph plus a `manifest.json`
    modeled on the shipped `src/age_and_gender/models/manifest.json`, and lists
    what the manifest must declare:
    - the task and the input/output names;
    - float32 RGB input `[N, 3, S, S]` and output `[N, classes]`;
    - channel means with a 1/256 scale, and softmax inside the graph;
    - labels `["female", "male"]` and the 81 age weights;
    - the runtime block.
  - *What "same results" means*: replace the `spec/README.md` pointer with a
    pointer to `tests/parity/`.
  - *Performance*: remove the latency table and every `benchmarks/` link;
    once the harness is gone, nothing in the repository reproduces those
    numbers. Replace them with one or two sentences without figures: the
    ONNX Runtime sessions and the landmark model are loaded once and reused,
    so a `predict()` call no longer copies the image and both networks the way
    1.x did.
  - *Development*: drop the `spec/README.md` link and the "Maintainer-only
    tooling … lives under `tools/`" paragraph.
- **`AGENTS.md`**: drop the legacy-oracle bullet and the "Migration
  specification" section (see Open decisions).
- **Test code**: fix every hit of the second verification grep. The spec/PR
  references are in `tests/golden.py`, `test_batching.py`,
  `test_chip_inference.py`, `test_end_to_end.py`, `test_frontend.py`,
  `test_models.py`, `test_api.py`, `test_legacy_loaders.py`, and
  `test_package_resources.py`.
- **Model notices** (`src/age_and_gender/models/notices/`): checked; no
  changes needed.

### 7. `spec/`

Add PR-9 to the "Read first" list in `spec/README.md`. Make no other edits to
`spec/`. The older specs describe `tools/` and root `models/` layouts that this
PR removes; that is expected for an internal record of the plan. The layout
below supersedes the "Final repository layout" in `spec/README.md`.

## Final layout

```text
age-and-gender/
├── .github/
│   ├── scripts/smoke_install.py
│   └── workflows/                 ci.yml, release.yml
├── AGENTS.md
├── LICENSE
├── README.md
├── pyproject.toml
├── .gitignore
├── example/
│   ├── example.py
│   ├── result.jpg
│   ├── test-image.jpg
│   └── test-image-2.jpg
├── spec/                          internal plan; only the PR-9 index entry changes
├── src/
│   └── age_and_gender/            unchanged file set
│       └── models/                age-v1.onnx, gender-v1.onnx,
│                                  shape_predictor_5_face_landmarks.dat,
│                                  manifest.json, notices/
├── tests/
│   ├── fixtures/legacy/           3 golden files, artifacts/, manifest.json
│   ├── unit/
│   ├── integration/
│   └── parity/
└── venv/                          local, Git-ignored
```

## Out of scope

- Any change to `predict()` results, public names or signatures, the bundled
  model files, dependency pins, or the supported platform matrix.
- The dlib-free runtime track (PR-8 units A–D). It stops here; restarting it
  would begin by restoring the prototype and the oracle from Git history.
- Rewriting Git history to shrink the repository. The deleted files stay in
  history.
- Regenerating frozen fixtures.
- Adding a type checker to CI. PR-9 fixes the current findings; enforcing them
  is a follow-up.

## Open decisions (default first)

1. **Spec pointer in `AGENTS.md`.** Default: remove it, because the brief covers
   every Markdown file outside `spec/`. Alternative: keep one line, since
   coding agents read `AGENTS.md` and users do not.
2. **Visibility of `spec/`.** `spec/` and `AGENTS.md` are tracked, so they are
   public on GitHub. If the plan is meant to stay private, untrack them
   (`.gitignore` plus `git rm --cached`) in a separate change. Default: not
   part of PR-9.

## Verification

```bash
# No references to removed paths (frozen provenance data excluded)
git grep -n -I -E 'tools/|tools\.(conversion|legacy)|network_definitions|vendor/dlib|dogs\.jpg' \
  -- ':!spec' ':!tests/fixtures/legacy/*.golden.json'

# No references to the internal plan
git grep -n -I -i -P 'spec/|\bPR-[0-9]|\bpr[0-9]' -- ':!spec'

# No C/C++ or CMake left in the tree
git ls-files | grep -E '\.(c|cc|cpp|h|hpp)$|CMakeLists\.txt$'

venv/bin/python -m ruff check .
venv/bin/python -m pytest -q    # 253 passed, 0 skipped

# Types: a one-off run in pipx's isolated environment, not in venv/
pipx run --python "$(readlink -f venv/bin/python)" --spec mypy mypy \
  --python-executable venv/bin/python --ignore-missing-imports --strict \
  src/age_and_gender
```

The three `git` commands must print nothing. Before this PR, the first two print
79 and 92 lines outside `tools/`, and the third lists 1,744 files, all under
`tools/`.

## Acceptance

- [ ] `tools/`, `benchmarks/`, and root `models/` no longer exist, and no C,
  C++, CMake, or vendored third-party source is tracked.
- [ ] All three verification `git` commands print nothing.
- [ ] `ruff check .` passes.
- [ ] `pytest` passes with no skips, including the three parity files from
  Findings §2 on the 3-document, 11-face corpus.
- [ ] `mypy --strict` reports no errors in `src/age_and_gender`, apart from the
  missing dlib stubs.
- [ ] The wheel's `.onnx` and `.dat` files are byte-identical to before.
  `models/manifest.json` differs only by the two removed provenance fields.
- [ ] No CI job builds C++ or installs `onnx`. Every remaining job passes,
  including install-smoke from `.github/scripts/smoke_install.py`.
- [ ] `README.md` and `AGENTS.md` describe only what exists in the repository;
  the README makes no performance claim with figures.
- [ ] The PR description records the SHA of the last commit that still contains
  `tools/`, for anyone who needs the conversion pipeline or the oracle again.
