# Legacy reference oracle

This directory is maintainer-only. It rebuilds the dlib 19.20.0 age/gender
pipeline independently of the Python package and freezes each observable stage.
It is not imported by, packaged with, or required by the public runtime.

The executable reads exact HWC RGB bytes, uses zero detector upsampling, extracts
the 32 and 64 chips independently with dlib's C++ default padding of 0.2, and
exports chips, normalized NCHW float32 tensors, logits, probabilities, landmarks,
age expectation, and public results. Its timed path retains the original
per-call softmax subnet copies. Instrumented export runs separately and is never
reported as inference time.

## Build

Use a clean CPU-only build without fast-math or CUDA:

```bash
cmake -S tools/legacy -B build/legacy-oracle -DCMAKE_BUILD_TYPE=Release
cmake --build build/legacy-oracle --parallel 2
build/legacy-oracle/age_and_gender_legacy_oracle --version
```

The target directly compiles the repository's vendored dlib. It does not use the
system dlib package and does not add a dependency to `setup.py`.

The network aliases live in `tools/network_definitions.h`, shared with the
conversion exporter in `tools/conversion/` so the oracle and the converter can
never deserialize through divergent copies of the architecture. The frozen
goldens record that header's SHA-256 alongside `oracle.cpp`, so a change to
either one is visible in every reference manifest.

## Generate a reference

The historical lane is CPython 3.10 with the exact packages in
`requirements-python310.txt`. Keep it separate from the project `venv/`:

```bash
python3.10 -m venv /tmp/age-gender-legacy
/tmp/age-gender-legacy/bin/python -m pip install -r tools/legacy/requirements-python310.txt
/tmp/age-gender-legacy/bin/python tools/legacy/run_reference.py \
  example/test-image.jpg \
  --oracle build/legacy-oracle/age_and_gender_legacy_oracle \
  --models example/models \
  --output build/reference/test-image
```

`run_reference.py` rejects unknown canonical inputs, changed model files, source
image drift, and decoded-RGB drift. The resulting manifest records repository,
decoder, compiler/build, model, input, artifact, machine, CPU-feature, and thread
metadata. The `.rgb` and `.f32` files are raw little-endian arrays whose shapes
are declared by `oracle-report.json` (chips are HWC RGB; tensors are NCHW).

Validate canonical geometry, probabilities, artifact sizes, age expectations,
and public output against the checked manifest:

```bash
/tmp/age-gender-legacy/bin/python tools/legacy/validate_report.py \
  build/reference/test-image/oracle-report.json
```

Checked goldens retain exact chip/tensor hashes and float vectors while excluding
non-deterministic timings. Regenerate one only from a validated legacy run:

```bash
venv/bin/python tools/legacy/freeze_reference.py \
  build/reference/test-image \
  --output tests/fixtures/legacy/test-image.golden.json
```

Capture a baseline with per-workload machine and provenance metadata:

```bash
venv/bin/python benchmarks/legacy_baseline.py \
  build/reference/test-image/oracle-report.json \
  build/reference/test-image-2/oracle-report.json \
  --output benchmarks/results/linux-x86_64-python310.json
```

Two independently generated references can be checked without comparing their
non-deterministic timing fields:

```bash
venv/bin/python tools/legacy/compare_references.py \
  build/reference/test-image-run-1 build/reference/test-image-run-2 \
  --output tests/fixtures/legacy/stability-linux-x86_64.json
```

Pass `--box top,right,bottom,left` repeatedly for explicit rectangles. Omitting
all boxes exercises auto-detection; this is also the legacy empty-list behavior.

Use the `--box=...` form for rectangles with negative coordinates, otherwise
`argparse` reads the leading minus sign as another option. The frozen
out-of-frame case is generated with:

```bash
--box=69,525,141,453 --box=29,189,101,117 --box=69,525,141,453 --box=-20,100,100,-20
```

## Check the original extension

Build the original extension source in the same historical environment, then
compare its public dictionaries with the standalone report. The reference build
disables dlib's unused JPEG/PNG loaders so those development headers are not part
of the historical lane; `src/main.cpp` always receives decoded NumPy bytes.

```bash
cmake -S . -B /tmp/age-gender-extension \
  -DPYTHON_EXECUTABLE=/tmp/age-gender-legacy/bin/python \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/tmp/age-gender-extension/lib \
  -DAGE_GENDER_IMAGE_IO=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/age-gender-extension --parallel 2
PYTHONPATH=/tmp/age-gender-extension/lib \
  /tmp/age-gender-legacy/bin/python tools/legacy/verify_extension.py \
  build/reference/test-image/oracle-report.json --models example/models
```

This check intentionally compares only public output. Stage files come from the
standalone oracle because the original pybind11 API does not expose them.

## Current fixture scope

The checked conversion-gate fixtures cover the two original example images,
seven detected faces, a no-face image, and explicit reversed, duplicate, and
out-of-frame rectangles. They are intentionally small enough to run while
developing the ONNX converter. The exact chips, normalized inputs, logits,
probabilities, age expectations, and public results are sufficient to localize
conversion drift and unblock PR-2.

A larger, independently licensed evaluation corpus is deferred to release
hardening. It is not required to begin model conversion and must not be assembled
from vendored photographs whose redistribution rights are unclear. Do not
promote README screenshots or replacement-runtime output into goldens.
