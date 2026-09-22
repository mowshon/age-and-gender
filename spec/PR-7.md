# PR-7 — Validate installation, remove native build infrastructure, release

**Depends on:** PR-1 through PR-6 passing.

**Risk:** medium; cross-platform packaging is the final user-facing requirement.

**Deliverable:** modern, documented 2.0 package with compiler-free supported
installations and preserved output behavior.

## Goal

A developer installs the package with pip, loads a local image, and predicts
offline. The published wheel has no dependency on this repository's C++ build,
system GUI development libraries, or a model conversion step.

## Installation and compatibility matrix

Start with standard-GIL CPython **3.12, 3.13, and 3.14** on:

| Platform | Candidate minimum / architecture |
| --- | --- |
| Linux | glibc 2.28+, x86-64 |
| Linux | glibc 2.28+, ARM64 |
| Windows | x86-64; verify actual supported OS release in CI/runtime docs |
| macOS | macOS 14+, ARM64 |

This matrix comes from the observed ONNX Runtime/dlib-bin wheel intersection.
Every advertised cell still needs the full dependency resolver and real inference
tests. If a cell cannot be validated, omit it from the supported release matrix.
Do not assume a universal project wheel makes native dependencies universal.

## Work

### 1. Build and distribution verification

- Build a wheel and source distribution through the new PEP 517 backend.
- Build a second wheel from the source distribution in isolation.
- Inspect archives: include Python modules, typing marker, model assets, notices,
  metadata; exclude vendored C++, temporary probes, test data, and build outputs
  from the runtime wheel.
- Set a compressed wheel size budget of 100 MiB and report measured size. If
  unexpectedly large, examine duplication before redesigning model delivery.
- Test installed resources outside the checkout and with a read-only package
  directory. Model loading must not write into site-packages.
- Validate metadata/README rendering and distribution identity; retain existing
  MIT project license and include separate model/dependency notices as needed.

### 2. Prove compiler-free installation

In fresh platform environments, with no project checkout/native build tools
available to the install path:

```bash
python -m pip install --only-binary=:all: age-and-gender
python -m pip check
```

For prerelease CI, use the built wheel path and normal dependency resolution;
for release staging, use the candidate artifact index with explicit version.
Also smoke-test the ordinary documented `pip install` command. Retain logs
showing every dependency resolved to a wheel and no source-build subprocess ran.

After installation, block network access and run bundled-model prediction,
legacy known-`.dat` loading, explicit-box prediction, and the compatibility
corpus. Headless environments must work without X11 development packages.

Test unsupported platform resolution deliberately so documentation does not
suggest users fix a missing wheel by installing a compiler. No implicit fallback
from `dlib-bin` to official source-only `dlib` is permitted in packaging logic.

### 3. CI and dependency policy

- Run pull-request workflows for both `refactoring` and `master` targets. Merge
  individual implementation PRs into `refactoring`, then merge the validated
  migration into `master` through a final PR.
- Run lint/format checks, unit tests, model-signature checks, frontend/ONNX parity,
  built-wheel integration, and binary-only installation smoke tests.
- Keep frozen model/tensor fixtures with hashes; cache external corpus artifacts
  by hash. Normal tests should not need to build the legacy C++ oracle.
- Keep a separate maintainer job that can regenerate/export/validate models using
  the pinned legacy reference. Make its environment reproducible before cleanup.
- Test minimum allowed dependencies and latest allowed resolved dependencies.
  Pin conversion locks and reference-lane locks; update them deliberately.
- For dependency upgrades, compare full results before broadening version bounds.
  In particular, newer dlib or runtime kernels can change inference behavior.
- Use a release workflow with build/test/artifact inspection before publication.

### 4. Remove legacy application build requirements

After all parity and install gates pass:

- Remove root `CMakeLists.txt`, custom CMake `setup.py`, `src/main.cpp`, and
  application vendoring under `libs/`.
- First preserve the oracle/export source and a content-pinned method of obtaining
  its required dlib reference source (immutable archive/hash or retained historical
  revision). A mutable upstream download is not adequate reproducibility.
- Keep any necessary native maintainer tools isolated under `tools/`; they must
  not enter the runtime build or source-distribution build requirements.
- Consolidate tracked model assets so the wheel ships one shape model and two
  converted networks. Preserve source `.dat` retrieval/hashes for maintainers.
- Leave user-downloaded `models/` contents alone. Add appropriate local-model,
  virtual-environment, cache, and generated-artifact ignore rules.
- Update stale Python/C++ platform classifiers and obsolete installation commands.

### 5. User documentation and examples

- Replace source compilation instructions with the one-command installation path
  and a truthful tested platform/Python matrix.
- Add the working `AgeAndGender()` example; fix the old missing `data` assignment.
- Show explicit RGB conversion, NumPy inputs, external boxes, inclusive output
  coordinates, no-face results, and reuse of one predictor for multiple images.
- Document bundled offline models, explicit model directories, and known-hash
  legacy `.dat` compatibility. Explain how maintainers convert custom compatible
  networks; users should not expect arbitrary `.dat` loading.
- Remove `face_recognition` from the default installation story. Explain its
  coordinate convention without pulling in a competing `dlib` distribution.
- Explain that the Python package uses prebuilt native dependency wheels; do not
  advertise it as entirely native-code-free.
- Include a concise behavior migration table, model attributions, benchmark
  results, and the precise scope of the parity guarantee.
- Make examples resolve their local inputs relative to their script or explicit
  command-line paths, and allow headless execution without `img.show()`.
- Apply concise Google-style docstrings and comments to authored code; avoid
  carrying long template/tutorial explanations into simple runtime functions.

## Acceptance

- [ ] Every advertised platform/Python cell installs from wheels and runs offline.
- [ ] Full exact-output parity passes for the selected CPU runtime configuration.
- [ ] Default users need no CMake, compiler, local `libs/`, or manual model download.
- [ ] Wheel and source distribution contain the required assets and build cleanly.
- [ ] Legacy reference/export tooling remains reproducible and isolated.
- [ ] Root native application sources and build dependencies are removed only
  after their replacement has passed verification.
- [ ] Examples and migration docs match the installed API.
- [ ] Benchmarks support all published performance claims.

**Release readiness:** when these checks pass, prepare the versioned release
artifacts and changelog for publication.
