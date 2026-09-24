# age-and-gender

Age and gender estimation from face images, with **no compiler required**.
`pip install` pulls in prebuilt wheels only; there is no CMake, no C++
compiler, and no model download at install or import time.

![Detected faces annotated with predicted age and gender](https://raw.githubusercontent.com/mowshon/age-and-gender/master/example/result.jpg)

© [Bill Gates family](https://www.businessinsider.com/microsoft-bill-melinda-gates-drive-daughter-to-school-2019-4)

## Installation

```bash
python -m pip install age-and-gender
```

That's it — the age and gender networks and the five-point face landmark
model are installed with the package, and prediction works offline
immediately afterward.

### Supported platforms

| Platform | Minimum |
| --- | --- |
| Linux | glibc 2.28+, x86-64 or ARM64 |
| Windows | x86-64 |
| macOS | 14+, Apple Silicon (ARM64) |

Supported Python versions are **3.12, 3.13, and 3.14**, standard-GIL builds.
This matrix follows the wheel availability of this package's two native
dependencies, ONNX Runtime and [`dlib-bin`](https://pypi.org/project/dlib-bin/)
(see `spec/INVESTIGATION.md` for how it was derived). It is not a claim that
every other combination fails — only that these are the ones this project
builds and installs from wheels only. Precisely what `.github/workflows/ci.yml`
proves for each cell, so this list isn't broader than the evidence:

- Every OS/Python cell installs the built wheel with `--only-binary=:all:`
  and runs prediction, including the compatibility corpus.
- The `glibc-floor` job additionally installs and runs inside the exact
  `manylinux_2_28` container images ONNX Runtime and `dlib-bin` build their
  wheels against, on x86-64 and ARM64 — the only jobs that test the
  advertised **glibc 2.28** floor itself; the Linux cells above run on
  whatever newer glibc the `ubuntu-latest`/`ubuntu-24.04-arm` runner images
  ship.
- Prediction needing no network access is verified by actually blocking
  outbound access (an unprivileged network namespace) on Linux only,
  best-effort; Windows and macOS run the same smoke test without a network
  block, which shows it doesn't happen to need one but isn't proof it
  couldn't.

The package is *Python-only* in the sense that there is no project C++
extension or build step left for you to run — but NumPy, Pillow, ONNX
Runtime, and `dlib-bin` still ship native code in their own wheels, the same
way most numerical Python packages do. This is not a claim that the
dependency tree is entirely free of compiled code; see
[PR-8](https://github.com/mowshon/age-and-gender/blob/master/spec/PR-8.md) for the conditional, currently unimplemented follow-up
that investigates removing the dlib dependency itself.

## Quick example

```python
from age_and_gender import AgeAndGender
from PIL import Image

predictor = AgeAndGender()

with Image.open("photo.jpg") as image:
    # The model only accepts RGB; convert explicitly rather than relying on
    # however the source file happens to be encoded.
    results = predictor.predict(image.convert("RGB"))

print(results)
```

```python
[
    {
        "gender": {"value": "female", "confidence": 100},
        "age": {"value": 26, "confidence": 84},
        "face": [419, 266, 506, 352],
    },
    # ... one dictionary per detected face, in detection order
]
```

`face` is `[left, top, right, bottom]`, inclusive pixel coordinates. An image
with no detected faces returns `[]`.

### Runnable example

[`example/example.py`](https://github.com/mowshon/age-and-gender/blob/master/example/example.py)
prints the results as JSON and saves a copy of the image with each face boxed
and labelled:

```bash
python example/example.py                                   # bundled sample photo
python example/example.py photo.jpg --output annotated.jpg
python example/example.py photo.jpg --models-dir path/to/bundle > result.json
```

`--models-dir` loads a model bundle directory (see
[Bundled and custom models](#bundled-and-custom-models)) instead of the models
installed with the package. Labels use a system font, falling back to Pillow's
built-in font.

### NumPy array input

`predict()` also accepts an `[H, W, 3]` uint8 RGB array directly, instead of a
`PIL.Image.Image`:

```python
import numpy as np

array = np.asarray(image.convert("RGB"))
results = predictor.predict(array)
```

### Explicit face boxes

Pass `face_bounding_boxes` to skip detection and score specific regions
instead — for example, boxes from another face detector. Boxes use
`(top, right, bottom, left)`, matching the
[`face_recognition`](https://github.com/ageitgey/face_recognition) library's
`face_locations()` convention (not installed by this package; see below), and
are used exactly as given, including duplicates or coordinates that fall
outside the image:

```python
results = predictor.predict(
    image.convert("RGB"),
    face_bounding_boxes=[
        (266, 506, 352, 419),  # top, right, bottom, left
    ],
)
```

Omitting `face_bounding_boxes`, or passing `None` or an empty sequence, runs
the bundled face detector.

### Reusing one predictor

Constructing `AgeAndGender()` does no I/O; models are loaded lazily on first
use and then kept in memory. Build one instance and reuse it across images
rather than constructing a new one per call:

```python
predictor = AgeAndGender()
for path in image_paths:
    with Image.open(path) as image:
        print(predictor.predict(image.convert("RGB")))
```

Calls on one instance are thread-safe (serialized by an internal lock).
Independent instances — one per worker thread or process — avoid that
serialization entirely, at the cost of each holding its own copy of the
loaded models.

### Using a different detector

The bundled dlib HOG detector is adequate for most images, but any detector
that returns `(top, right, bottom, left)` boxes can supply
`face_bounding_boxes` instead — for example,
[`face_recognition`](https://github.com/ageitgey/face_recognition)'s
`face_locations()`, which uses that exact convention.

**This package does not install, require, or recommend `face_recognition`.**
Its declared `dlib` requirement is satisfied by the official source-only
`dlib` distribution, not the `dlib-bin` wheels this package uses — installing
both risks a source build (the compiler requirement this package exists to
avoid) and a module conflict. If you already depend on `face_recognition` for
an unrelated reason, its boxes pass through unchanged; installing it in a
separate virtual environment from this package avoids the conflict entirely.

```python
import face_recognition
import numpy as np

rgb = image.convert("RGB")
boxes = face_recognition.face_locations(np.asarray(rgb), model="hog")
results = predictor.predict(rgb, face_bounding_boxes=boxes)
```

## Bundled and custom models

`AgeAndGender()` uses the age, gender, and landmark models installed with the
package — nothing to download. To use different models instead:

```python
predictor = AgeAndGender()
predictor.load_shape_predictor("path/to/shape_predictor_5_face_landmarks.dat")
```

`load_shape_predictor` loads any compatible dlib five-point `.dat` file
directly. The neural loaders are different: `load_dnn_age_predictor` and
`load_dnn_gender_classifier` accept an `.onnx` file from a *complete model
bundle directory* — one with a `manifest.json` that names it for that task and
also names a shape predictor, alongside the other task's `.onnx` file — **not**
the original `dnn_age_predictor_v1.dat`/`dnn_gender_classifier_v1.dat` files.
Those are a proprietary dlib serialization format with no runtime ONNX reader;
there is no way to load one directly, by design.

```python
predictor.load_dnn_age_predictor("path/to/bundle/age-v1.onnx")
```

For a whole bundle at once, use `AgeAndGender.from_model_dir("path/to/bundle")`,
which validates every model immediately instead of on first use. Loading is
purely structural — the manifest's declared shape, dtype, and normalization
are checked, not the file's hash — so any bundle built to that contract works,
regardless of where its weights came from.

**Converting a different trained network is not currently automated.** The
maintainer pipeline in [`tools/conversion`](https://github.com/mowshon/age-and-gender/blob/master/tools/conversion/README.md)
converts *this project's own* two original dlib weight files specifically: its
`build_onnx.py` step checks the input `.dat` files against their known SHA-256
hashes and refuses anything else, and its raw output is not yet a loadable
bundle until a follow-up step (`build_bundle.py`) adds a shape predictor to
the manifest. Building a bundle for genuinely different weights means
hand-authoring an `.onnx` graph and a `manifest.json` that matches the
documented contract (task, tensor shape/dtype, normalization, and label
order); the tooling above is a worked example of that contract, not a
converter for arbitrary input.

## Migrating from 1.x

| | 1.x | 2.0 |
| --- | --- | --- |
| Install | `git clone` + `python setup.py install` (needs CMake, a C++ compiler, and X11/JPEG/PNG dev headers) | `pip install age-and-gender` (wheels only) |
| Models | Download three `.dat` files yourself and load them by path | Bundled with the package; load automatically |
| `AgeAndGender()` | Needed all three `load_*` calls before `predict()` would work | Works immediately; `load_*` methods are optional overrides |
| `load_dnn_age_predictor`/`load_dnn_gender_classifier` | Accepted the original dlib `.dat` files | Accept a converted `.onnx` file + manifest only (see above) |
| `load_shape_predictor` | Accepted a dlib `.dat` file | Unchanged |
| `predict()` | Same signature, same result schema, same box convention | Unchanged |
| Supported Python | 2.7, 3.4–3.8 (per the old classifiers) | 3.12, 3.13, 3.14 |

`predict()`'s inputs, box convention, and result dictionaries are unchanged;
existing calling code does not need to change, only how the package is
installed and how a *custom* model would be loaded.

### What "same results" means

The age and gender networks are the original trained weights, converted to
ONNX rather than retrained; detection, landmarks, and chip alignment run
through the same dlib algorithms as 1.x, via the `dlib-bin` wheel instead of
a compiled extension. This package's test suite requires exact agreement
with the original C++ implementation — face count, order, rectangles, the
integer age, the gender string, and the integer confidences — on a frozen
compatibility corpus, not just "close enough" agreement. It is a tested
release contract on that corpus, not a claim of bit-identical floating point
on every possible image, CPU, or execution provider: a probability-to-integer
boundary (a rounded age, a floored confidence percentage) can in principle
land differently for an image this project has not measured. A change that
fails that exact-agreement gate blocks release rather than shipping with a
weakened guarantee. See `spec/README.md`'s "Shared acceptance contract" for
the full, precise wording.

## Performance

The Python runtime is faster than the original compiled extension on every
measured case — the old code copied the whole input image and both neural
network subnets on *every* `predict()` call, which this package's persistent
ONNX Runtime sessions avoid entirely:

| Case | Faces | Legacy | 2.0 | Change |
| --- | ---: | ---: | ---: | ---: |
| Single explicit box | 1 | 46.28 ms | 4.80 ms | -90% |
| Detected | 2 | 116.35 ms | 28.61 ms | -75% |
| Detected | 5 | 325.18 ms | 82.77 ms | -75% |

Full methodology, machine caveats, and additional measurements (batch-size
sweep, cold start, concurrency) are in [`benchmarks/README.md`](https://github.com/mowshon/age-and-gender/blob/master/benchmarks/README.md)
and [`benchmarks/report-pr6.md`](https://github.com/mowshon/age-and-gender/blob/master/benchmarks/report-pr6.md); raw data is in
[`benchmarks/results/`](https://github.com/mowshon/age-and-gender/tree/master/benchmarks/results/). These are one development
machine's numbers, reported as reproducible relative comparisons, not a
certified benchmark on dedicated hardware.

## Model attributions and licenses

This project's own code is MIT-licensed; see [`LICENSE`](https://github.com/mowshon/age-and-gender/blob/master/LICENSE). The
bundled model weights are third-party and carry their own notices, reproduced
in [`src/age_and_gender/models/notices/`](https://github.com/mowshon/age-and-gender/tree/master/src/age_and_gender/models/notices/):

- The age and gender network weights were contributed by **Cydral
  Technology** to [`davisking/dlib-models`](https://github.com/davisking/dlib-models)
  and dedicated to the public domain under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). This
  package ships them converted to ONNX; the weights themselves are
  unchanged.
- The five-point face landmark model (`shape_predictor_5_face_landmarks.dat`)
  is redistributed unchanged from the same project, by Davis E. King, also
  under CC0 1.0.

## Development

See [`AGENTS.md`](https://github.com/mowshon/age-and-gender/blob/master/AGENTS.md) for the project's development environment, and
[`spec/README.md`](https://github.com/mowshon/age-and-gender/blob/master/spec/README.md) for the full migration specification this
package was built against, including the acceptance contract that "same
results" is tested against.

```bash
venv/bin/python -m pytest
venv/bin/python -m ruff check .
```

Maintainer-only tooling — the historical C++ reference oracle, the ONNX
conversion pipeline, and the frozen extension source kept for provenance —
lives under [`tools/`](https://github.com/mowshon/age-and-gender/tree/master/tools/) and is documented there; none of it is part of
the published package's build or runtime.

## Changelog

**2.0.0**
- Rebuilt as a pure Python package: no CMake, no C++ compiler, no vendored
  dlib source in the install path. The age and gender networks now run
  through ONNX Runtime; detection, landmarks, and alignment use the
  [`dlib-bin`](https://pypi.org/project/dlib-bin/) wheel.
- Models are bundled with the package; no manual download.
- `load_dnn_age_predictor`/`load_dnn_gender_classifier` now accept converted
  `.onnx` bundles instead of the original dlib `.dat` files (see
  "Migrating from 1.x" above).
- Added `AgeAndGender.from_model_dir()` for loading a complete custom model
  bundle.
- `predict()`'s inputs, box convention, and result schema are unchanged from
  1.x.

**1.0.1**
- `predict(pillow_img)` requires a `PIL.Image` object.
- `predict(pillow_img, face_bounding_boxes)` accepts an optional list of
  face boxes; omitting it runs detection.

**1.0.0**
- Initial release.
