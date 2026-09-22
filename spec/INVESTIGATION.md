# Investigation: age-and-gender Python migration

## 1. Executive findings

**Yes: developers can install a Python package and load/run these models without
building this project's C++ code.** It requires moving custom neural-network
inference to a standard runtime, rather than just replacing the pybind11 calls.

The recommended design is:

```text
Pillow / RGB uint8 NumPy image
    → Python validation
    → existing dlib HOG detector, or caller-provided boxes
    → existing five-point landmark model
    → individually extracted 32×32 and 64×64 aligned RGB chips
    → Python float32 NCHW preprocessing
    → persistent ONNX Runtime CPU sessions, same converted weights
    → Python legacy-compatible postprocessing
    → existing list[dict] result
```

Key findings:

- The repository's application implementation is **one C++ extension**. There is
  no existing Python inference module to modernize in place.
- The `.dat` files are dlib serialization, not generic ONNX, Torch, TensorFlow,
  Caffe, or OpenCV models. The age/gender network types are defined in C++ templates.
- Standard dlib Python bindings expose the detector and shape predictor, but not
  these two custom classifiers or a generic loader for arbitrary dlib networks.
- Simply depending on `dlib` does not solve installation: the current official
  PyPI release inspected has a source archive and **zero wheels**.
- `dlib-bin` has current wheels and worked locally. It preserves the frontend
  much more directly than adopting a different detector/landmark model.
- C++ default chip padding is **0.2**, different from the Python
  `get_face_chip` helper's **0.25**. This matters to both networks.
- Batched crop extraction is **not pixel-equivalent** to individual crop
  extraction on the repository's first example. Keep crops individual initially.
- Small floating-point changes can alter floored confidence percentages or rounded
  ages. Output parity needs a reference oracle and intermediate fixtures.
- The README's displayed output is not a reliable golden fixture for the current
  code and images: the investigation's reference run returned different boxes,
  ordering, and some ages/confidences.

## 2. Repository inventory and installation problems

| Location | Finding |
| --- | --- |
| `src/main.cpp` | 283 lines: network definitions, image conversion, detection/alignment, postprocessing, pybind11 exports |
| `setup.py` | Version 1.0.1; custom `build_ext` invokes CMake; runtime requirements are only Pillow and NumPy |
| `CMakeLists.txt` | Builds `src/main.cpp` and all of dlib; unconditionally links pthread, X11, JPEG, PNG |
| `libs/dlib/` | Vendored dlib **19.20.0**, including examples/tools and its own embedded pybind11 |
| `libs/pybind11/` | Vendored pybind11 **2.5.0** |
| `example/` | Two source photos, visualization scripts/results, a font, and a tracked copy of all three models |
| `models/` | User-downloaded, untracked models; byte-identical to the tracked example copies |
| `spec/` | Empty before this investigation |
| Tests / CI / modern build metadata | No project-level tests, CI workflow, or `pyproject.toml` found; vendored upstream tests are not project coverage |

The old installation requires CMake, a C++ compiler, platform link dependencies,
and a compatible Python C API. It imports `distutils.version`, which is removed
from the standard library in Python 3.12. Even if a setuptools compatibility
layer supplies it, this remains obsolete build infrastructure. The build's
platform-specific system libraries contradict the broad platform classifiers.
The application does not need a display, yet its build links X11.

Documentation lists Python 2.7–3.8; package classifiers additionally list 2.6.
The first README usage snippet calls `data` without constructing `AgeAndGender`.
The optional `face_recognition` example also pulls in an additional model package
and the source-built `dlib` distribution. That should not remain the default
installation recipe.

### Source anchors

- Build requirements: `setup.py:18–61`, `setup.py:76–82`, `CMakeLists.txt:12–22`.
- Vendored versions: `libs/dlib/dlib/CMakeLists.txt:21–25` and
  `libs/pybind11/include/pybind11/detail/common.h:95–97`.
- Networks: `src/main.cpp:18–124`.
- Public behavior: `src/main.cpp:128–282`.

## 3. Downloaded model identity

These sizes and SHA-256 digests were measured locally. Each file under `models/`
matches the same-named file under `example/models/`.

| Model | Bytes | Purpose |
| --- | ---: | --- |
| `dnn_age_predictor_v1.dat` | 10,790,013 | 81-way age classifier with expected-age postprocessing |
| `dnn_gender_classifier_v1.dat` | 530,926 | Two-class gender classifier |
| `shape_predictor_5_face_landmarks.dat` | 9,150,489 | Five-point regression-tree landmark predictor |

```text
dnn_age_predictor_v1.dat
4b78d4d7055e22620e362884b5551caa9379080277338aa2d4cdfc592f0e9fa3

dnn_gender_classifier_v1.dat
85453d6f6585c8e02ada95929956783c780dc04dcec5bdfd14af82f15c99ba41

shape_predictor_5_face_landmarks.dat
c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888
```

Total original model size is **20,471,428 bytes** (~19.52 MiB). This is small
enough to make bundling the shape model and converted networks a practical
default; measure the actual wheel after conversion. Do not ship duplicate old
and converted neural networks without a specific need.

The upstream dlib-models README attributes the age/gender networks to Cydral
Technology under CC0-1.0. The five-point model is covered by Davis King's
public-domain statement. Preserve upstream model descriptions and notices in
the distribution. The 68-point model has different licensing and is not involved.

The HOG detector is a **fourth learned component**, embedded in dlib's
`frontal_face_detector.h`, not an additional downloaded file. It consists of five
HOG filters and uses `scan_fhog_pyramid<pyramid_down<6>>`. Any dlib-free plan must
account for this component too.

## 4. Exact legacy inference contract

### Model loading and state

`AgeAndGender()` holds three models and a detector. `load_shape_predictor()` both
deserializes the landmark model and initializes the detector. The two neural
loaders deserialize into their respective C++ template network types.

There are no clear loaded-state guards. Calling prediction too early is not a
useful compatibility behavior to preserve; the new package should initialize
bundled defaults or raise a documented model error.

### Input and boxes

The exported method is `predict(photo_numpy_array, face_bounding_boxes=[])`.
Despite that name, the examples pass RGB Pillow images, which pybind11 converts
to an unsigned-byte NumPy array. `from_numpy()` copies each RGB pixel into a
dlib matrix. It assumes a three-dimensional array and accesses channels 0–2
without robust channel-count validation.

For intended valid inputs:

- RGB channel order, H×W×3 unsigned bytes.
- Nonempty external boxes use **(top, right, bottom, left)**.
- They are translated directly into dlib inclusive `(left, top, right, bottom)`
  rectangles. Preserve input order and duplicate boxes.
- Empty boxes mean **run detection**, not “return zero faces.”
- Detection uses `detector(in)` with no preliminary image upsampling; Python
  must use the equivalent zero-upsampling path.
- Output rectangles are **[left, top, right, bottom]**, retaining inclusive dlib
  coordinates. Do not switch silently to exclusive image-slice endpoints.
- The old wrapper does not clip external rectangles before landmark prediction.
- Faces with no landmark parts are skipped, though the shipped model returns five.

Invalid array shapes, accidental float-to-byte casts, malformed boxes, and native
out-of-bounds accesses are not behaviors to emulate. Specify clean validation.
Adding `None` as another detection sentinel is a compatible extension; retaining
the `photo_numpy_array` keyword avoids breaking existing keyword calls.

### Alignment

For every face, the same predicted five landmarks produce **two independent
chips from the original image**:

1. Gender: `get_face_chip_details(shape, 32)` → `extract_image_chip`.
2. Age: `get_face_chip_details(shape, 64)` → `extract_image_chip`.

Default padding is `0.2`, and the default interpolation is bilinear. Chip
extraction also builds a downsampling pyramid where appropriate. It is not
equivalent to cropping a bounding rectangle and calling Pillow resize, nor to
resizing one model's chip into the other model's size.

Five-point template locations, normalization, and transforms are in
`libs/dlib/dlib/image_transforms/interpolation.h:1942–2033`; the pyramid path is
at `1766–1876`.

### Input normalization

Both loaded networks reported these float32 mean values:

```text
R = 122.781998    G = 117.000999    B = 104.297997
```

They correspond to the serialized float32 versions of `[122.782, 117.001,
104.298]`. Input processing is planar **NCHW float32**, with:

```text
(RGB_byte - channel_mean) / 256.0
```

It is not BGR, not division by 255, and not ImageNet standard deviation
normalization. Export actual serialized means rather than trusting defaults for
all possible model files. Source: `libs/dlib/dlib/dnn/input.h:90–143`.

### Age model

The source calls it ResNet-10. Treat the actual template graph as authoritative:
it has five two-convolution residual blocks, a stem convolution, and an 81-output
fully connected layer. A stock framework ResNet is not a drop-in equivalent.

Input-to-output structure for 64×64 chips:

| Stage | Channels / spatial size | Details |
| --- | --- | --- |
| Input | 3 / 64×64 | RGB normalization above |
| Stem convolution | 64 / 29×29 | 7×7, stride 2, **zero padding**; affine, ReLU |
| Max pool | 64 / 14×14 | 3×3, stride 2, zero padding |
| Residual 64 | 64 / 14×14 | Two 3×3 stride-1 convolutions, padding 1 |
| Down residual 128 | 128 / 7×7 | Main branch 6×6; skip branch 7×7; dlib zero-extends before addition |
| Residual 128 | 128 / 7×7 | Two stride-1 convolutions |
| Down residual 256 | 256 / 3×3 | Skip channels zero-extended from 128 to 256 |
| Residual 256 | 256 / 3×3 | Two stride-1 convolutions |
| Global average pool, FC | 256 → 81 | Softmax added for inference |

The spatial sizes above follow the inspected layer definitions and floor output
formula; PR-2 must confirm them against captured intermediate tensors.

`add_prev` is not ordinary broadcasting: it allocates the maximum extent in each
dimension and treats missing coordinates of either operand as zero. This is
critical for downsampling blocks. See `layers.h:2205–2215` and
`cuda/cpu_dlib.cpp:279–334`.

Age postprocessing is:

```text
estimate = float32(0.25 * p[0])
for i = 1 .. 80, in order:
    estimate = float32(estimate + float32(i * p[i]))
age = std::lround(estimate)
confidence = floor(float32(max(p) * 100.0))
```

Class zero represents **0.25 years**. Age is an expectation, not `argmax`.
`std::lround` rounds half away from zero, unlike Python `round`/NumPy `rint`.
For this nonnegative estimate, use `math.floor(float(estimate) + 0.5)` after
preserving float32 accumulation; do not introduce another float32 rounding in
the `+0.5` step. Confidence is the largest class probability, not a calibrated
probability of the rounded expected age.

### Gender model

32×32 RGB → two affine/ReLU convolution pairs (32 channels, then 64), each
followed by 2×2 average pooling → 0.5 multiplier → FC(16) → ReLU → 0.5
multiplier → FC(2) → softmax.

The two **loaded** multiplier values were both 0.5. These are inference-time
scalings and cannot simply be removed as if dropout had no effect.

Class 0 is `female`; class 1 is `male`. Exactly tied probabilities return
`female`. Confidence is `floor(float32(selected_probability * 100.0))`.

### Output

```python
[
    {
        "gender": {"value": "female", "confidence": 100},
        "age": {"value": 26, "confidence": 84},
        "face": [419, 266, 506, 352],
    },
]
```

Return ordinary Python strings, integers, dictionaries, and lists. Preserve
face ordering. The current source inserts `gender`, then `age`, then `face`;
retaining that order is inexpensive even though dictionary equality ignores it.

## 5. Executed probes and their limits

### Environment

- Linux x86-64, glibc 2.35; Python 3.10.15.
- GCC 11.4.0; CMake 3.22.1 available.
- Temporary environment under `/tmp/opencode/age-gender-investigation`.
- Binary-only install succeeded for `dlib-bin==20.0.1`, `numpy==2.2.6`, and
  `Pillow==12.3.0`; `pip check` reported no broken requirements.
- NumPy 2.2.6 was selected because this local Python is 3.10, not because it is
  the newest NumPy release.

A temporary standalone C++ program reconstructed the network aliases and
inference operations from `src/main.cpp`, linking the vendored dlib 19.20 source
with `-std=c++14 -O2 -DDLIB_NO_GUI_SUPPORT` and pthread, without BLAS/CUDA.
It successfully deserialized all three local models and ran inference.

Pillow decoded each JPEG once; identical raw RGB bytes went to the C++ and Python
paths. This avoids accidentally measuring different JPEG decoders. The Python
comparison ran the current dlib detector with zero upsampling, landmarks, and
individual chips with explicit padding 0.2.

**Result:** all seven face boxes, their ordering, all five landmarks per face,
and 14 individually extracted chip FNV-1a-64 checksums matched. Checksums are
strong smoke evidence; PR-1/PR-4 require direct array comparisons on retained
fixtures. This was a reconstructed standalone core, not a successful installation
or execution of the original pybind11 extension.

### Reference outputs observed

Each row is in detector return order. These are local C++ probe observations,
not yet approved cross-platform golden fixtures.

| Image | Face `[L,T,R,B]` | Age | Age confidence | Gender | Gender confidence |
| --- | --- | ---: | ---: | --- | ---: |
| `test-image.jpg` | `[419,266,506,352]` | 26 | 84 | female | 100 |
| `test-image.jpg` | `[780,112,883,215]` | 58 | 46 | male | 99 |
| `test-image.jpg` | `[595,135,699,238]` | 19 | 72 | male | 98 |
| `test-image.jpg` | `[227,198,314,285]` | 62 | 53 | female | 99 |
| `test-image.jpg` | `[352,544,438,630]` | 25 | 89 | female | 100 |
| `test-image-2.jpg` | `[117,29,189,101]` | 60 | 22 | male | 99 |
| `test-image-2.jpg` | `[453,69,525,141]` | 51 | 86 | female | 99 |

Source image SHA-256:

```text
example/test-image.jpg      (1100×825)
0a3eb36690b3ae319939e534b6c4cd00def8045af8c12d62829d6d36ec7746f4

example/test-image-2.jpg    (634×435)
e060bdc12cd53020c104ad305324c69832ec4720c2709e525be5975aa7877db9
```

### Crop-batching counterexample

For `test-image.jpg`, comparing Python `get_face_chips()` against five individual
`get_face_chip()` calls with the same shapes and padding produced this many
different channel values per 32×32 RGB chip:

```text
[1540, 1665, 1796, 0, 1599]  # out of 3072 channel values per chip
```

The 64×64 chips matched on this image, and both sizes matched on the second
image. The implementation constructs a shared crop/pyramid for a batch; grouping
chips can change its origin/downsampling behavior. Therefore, batch neural
inference only after individual chips have been extracted. Do not generalize
the matching 64×64 observation to all images.

### What has not been proved

- Neither network has been converted to ONNX in this investigation. **PR-2 has
  since done so**; see `spec/PR-2.md` for the delivered result.
- No ONNX-vs-dlib probabilities or complete new-package outputs were compared.
  **PR-2 compared probabilities, logits and intermediate stages**; complete
  new-package outputs remain PR-5's.
- No broad image corpus, Windows, macOS, ARM64, or Python 3.12–3.14 execution
  was tested locally.
- No speedup has been measured.
- No generic Python dlib serializer/deserializer or dlib-free frontend has been
  implemented.

PR-1 converts the temporary experiments into reproducible, reviewed maintainer
tooling and durable fixtures. Temporary probe paths are not build dependencies.

## 6. Runtime choices

| Approach | Installation | Same models/results? | Assessment |
| --- | --- | --- | --- |
| Wrap everything using `pip install dlib` | Current official package builds from source | Custom age/gender bindings still missing | Does not solve either core problem |
| Use `dlib-bin` alone | Wheels on a defined platform matrix | Has frontend, but no custom age/gender API | Useful frontend dependency, insufficient alone |
| Publish wheels of this existing extension | Easy for supported wheel tags | Closest to existing code | Solves end-user builds but retains project C++ ownership |
| ONNX networks + wheel-provided dlib frontend | Python API; binary dependencies | Reuses weights and alignment; numerical parity must be tested | **Recommended** |
| ONNX networks + different detector/landmarks | Usually wheel-installable | Face set/crops change, so outputs change | Separate behavior change, not compatibility migration |
| PyTorch recreation of the same graph | Wheels; much heavier dependency tree | Possible with the same careful conversion | No need for training framework in inference package |
| Python/NumPy dlib-format reader and full frontend | Large custom numerical implementation | Possible in principle; extensive validation needed | Conditional dlib-free project, PR-8 |

OpenCV `readNet`, `torch.load`, and `onnxruntime.InferenceSession` do not directly
interpret these dlib network serialization files. Renaming a `.dat` file does
not convert it. The landmark model is a regression-tree cascade, not another CNN
that can be passed through the same convolution-network converter.

### Conversion path

Use the existing network types in a maintainer-only executable to deserialize
the models and export network structure and parameters. Vendored dlib already
provides `net_to_xml()` and a `tools/convert_dlib_nets_to_caffe/` example proving
that these layer descriptions/weights are inspectable. Build a direct ONNX
converter for the exact required layer set, rather than introducing Caffe as
another build dependency.

Important exporter detail: in this dlib version, `affine_::get_layer_params()`
returns an empty tensor. A naive `visit_layer_parameters()` dump loses affine
scale and bias. `affine_::to_xml()` does expose them. `net_to_xml()` sets precision
to nine significant digits, suitable for float32 round-tripping; still verify
every extracted parameter against an independent reference and record locale.
See `layers.h:2008–2180`, `dnn/utilities.h:101–112`.

Prefer standard ONNX operations and an explicit, validated opset/IR version.
Latest ONNX tooling need not imply emitting its newest default IR/opset, which
may exceed the selected ONNX Runtime's support.

## 7. Dependency and platform research

The following are the latest stable versions returned by live PyPI JSON metadata
on **2026-09-22**. They are a dated starting point, not an assertion that a full
combination has already passed parity or installation tests.

| Dependency | Observed version | Python requirement | Proposed role |
| --- | --- | --- | --- |
| NumPy | 2.5.3 | >=3.12 | Runtime tensors and preprocessing |
| Pillow | 12.3.0 | >=3.10 | Runtime Pillow input support / example image decoding |
| ONNX Runtime | 1.30.0 | >=3.11 | CPU inference runtime |
| dlib-bin | 20.0.1 | No metadata minimum; wheels cp310–cp314 | Detector, landmarks, alignment |
| dlib (official) | 20.0.1 | No metadata minimum | Oracle/reference only; latest release has zero wheels |
| ONNX | 1.23.0 | >=3.10 | Conversion/validation tooling only |
| Hatchling | 1.32.4 | >=3.10 | Python package build backend |
| pytest | 9.1.1 | >=3.10 | Tests |
| Ruff | 0.16.8 | >=3.7 | Lint/format and selected docstring checks |
| PyTorch | 2.14.0 | >=3.10 | Evaluated, not required by recommended design |

**Proposed baseline: CPython 3.12–3.14, standard GIL builds.** Python 3.12 permits
the newest NumPy; choosing a lower minimum requires a separate dependency lane.
Recheck versions at implementation time and run the resolver and parity suite
before committing constraints. Keep exact reproducible CI/conversion lockfiles.

Proposed initial runtime constraints after validation:

```text
numpy>=2.3.3,<3
Pillow>=12.3.0,<13
onnxruntime>=1.23.0,<1.24
dlib-bin==20.0.1
```

**Updated after PR-2.** The table above records the newest versions PyPI
advertised on the investigation date. The versions the project environment
actually resolved, and that the conversion bundle was validated against, are
NumPy 2.3.3, onnx 1.19.0 and ONNX Runtime 1.23.0 on CPython 3.13.15. The
constraints above now name the tested runtime rather than an untested newer one,
because the conversion parity evidence in
`tools/conversion/artifacts/v1/conversion-report.json` was gathered there. PR-3
should re-resolve, re-run the parity suite, and widen or move these pins on the
strength of that run instead of inheriting either set of numbers unverified.

The tight frontend/runtime constraints are deliberate parity controls. Widen
them after testing; dependency upgrades should regenerate an explicit parity
report. Do not perpetually freeze all tooling or claim every future minor is
known-compatible.

### Candidate wheel intersection

Observed ONNX Runtime 1.30.0 wheels cover Linux glibc 2.28+ x86-64/ARM64,
Windows x86-64/ARM64, and macOS 14+ ARM64. Observed dlib-bin 20.0.1 wheels cover
Linux glibc/musl x86-64/ARM64, Windows x86-64, and macOS 14+ ARM64.

The candidate **intersection** for the proposed release is therefore:

- Linux glibc 2.28+ x86-64 and ARM64.
- Windows x86-64.
- macOS 14+ Apple Silicon.

Verify NumPy, Pillow, and all transitive wheels as well. The selected versions
do not establish support for Intel macOS, Windows ARM64, Alpine/musl, 32-bit
systems, PyPy, or free-threaded CPython. Available wheel filenames are a
candidate matrix, not successful execution evidence. Advertise only CI-tested
combinations; narrowing the matrix is preferable to accidental compilation.

`dlib-bin` and `dlib` are different distribution names exporting the same module.
Do not install both. `face-recognition==1.3.0` declares `dlib>=19.7`, which is not
satisfied in pip metadata by installing `dlib-bin`. Making it a required or
default extra can reintroduce a source build and module conflict. Continue to
accept its box convention without depending on that package.

## 8. Target package design

```text
pyproject.toml
src/age_and_gender/
    __init__.py
    api.py                 # Public class and compatibility methods
    _images.py             # RGB and rectangle validation
    _faces.py              # Dlib frontend, kept behind a narrow boundary
    _inference.py          # ONNX sessions and float32 tensor preparation
    _postprocess.py        # Exact legacy rounding / result conversion
    _models.py             # Resources, manifests, hash-based legacy mapping
    _types.py              # Result TypedDicts, small shared types
    py.typed
    models/
        manifest.json
        age-v1.onnx
        gender-v1.onnx
        shape_predictor_5_face_landmarks.dat
        notices/
tools/legacy/              # Maintainer oracle, isolated from package build
tools/conversion/          # Maintainer export and ONNX generation
tests/
    fixtures/
    unit/
    parity/
    integration/
benchmarks/
```

Do not create a generalized backend/plugin framework for this migration. Keep
one CPU execution path with an isolated frontend boundary and simple model
metadata. `onnx` belongs to conversion tooling; `onnxruntime` belongs to users.

### Model availability and old loader calls

Bundle converted ONNX networks and the original five-point model. Load resources
using `importlib.resources`, including correct lifetime management if a native
API needs a temporary extracted file. Do not resolve relative to the process
working directory. No import-time downloads, conversion, or model inference.

Preserve `load_shape_predictor`, `load_dnn_age_predictor`, and
`load_dnn_gender_classifier` names. Known legacy neural `.dat` files can continue
to work by **hashing their contents and selecting their preconverted bundled
equivalent**. This is not direct `.dat` inference. Unknown `.dat` weights require
the documented maintainer converter and must fail clearly instead of silently
using defaults. Shape `.dat` loading remains direct through dlib.

Support explicit converted model bundles with manifests. Validate task, shape,
dtype, normalization, label order, and hashes when loading. A file with 81 outputs
alone is not enough to identify a compatible age model.

### Public API

- `AgeAndGender()` uses bundled models lazily.
- `AgeAndGender.from_model_dir(path)` loads a validated explicit bundle.
- Existing loader methods return `None` and replace the selected model only
  after successful validation.
- `predict(photo_numpy_array, face_bounding_boxes=None)` retains the original
  keyword and recognizes both `None` and an empty sequence as auto-detection.
- RGB Pillow images and H×W×3 uint8 arrays are the supported initial inputs.
  Non-RGB Pillow inputs and non-uint8 arrays get helpful validation errors rather
  than silent color/range conversion. Examples show `.convert("RGB")`.
- Out-of-frame but ordered nondegenerate integer rectangles retain old dlib
  behavior; no implicit clipping. Invalid geometry/types raise clear errors.
- Default output is exactly the old schema. Optional future raw probabilities
  must be separate from that default return contract.

## 9. Parity, performance, and release strategy

Establish a deterministic CPU oracle before touching packaging. Freeze raw RGB
inputs, boxes, landmarks, individual chips, input tensors, logits/probabilities,
expected ages, and final dictionaries. Compare each stage to localize failures.
Retain the old weights and oracle for conversion maintenance independently of
the user-facing package.

Exact final outputs are required on the compatibility corpus. Intermediate
floating-point outputs use small fixed tolerances defined in PR-1, with explicit
boundary tests. There is no blanket “within one year/one percent is the same”
exception. CPU is the initial supported provider; GPU, reduced precision, and
quantization are separate behavior/performance projects.

Current opportunities identified by inspection:

1. The old `predict()` creates two softmax networks and copies both network
   subnets on **every call**, even before determining whether there are faces.
   Persistent ONNX sessions remove this work.
2. The old path copies every image pixel into a dlib matrix. Python NumPy image
   views can avoid that application-level loop; verify backend copy behavior.
3. CNN inference runs twice per face. Batch already-extracted chips, subject to
   parity validation and bounded batch sizes.
4. Allocate contiguous float32 model inputs once per batch, and preserve float32
   arithmetic. Vectorize across faces while retaining ordered age reduction.
5. Separate cold start, warm prediction, detector, crop extraction, and network
   time. The HOG frontend may dominate after CNN optimization.
6. Tune CPU thread counts only after measuring small-batch overhead and
   application-level concurrency. Avoid nested-thread oversubscription.

Do not batch crop extraction, substitute a detector, add detector upsampling,
resize source images, fuse away affine layers, quantize, or use lower precision
as an unvalidated default optimization.

## 10. Risks and explicit decisions

| Risk / decision | Resolution |
| --- | --- |
| ONNX numerical parity is unproven | **Cleared by PR-2** on a narrow corpus: stage-by-stage tensors, probabilities and public results agree with the oracle |
| Conversion evidence rests on 8 distinct real chips | PR-1 deferred the 30-image/100-face corpus; PR-4 must widen it before release |
| New frontend dlib may drift on other images | PR-4 runs a larger corpus; choose a tested wheel version, not an untested upgrade |
| Probability boundaries change public integers | Exact final-output checks plus float32/rounding boundary fixtures |
| Third-party dlib wheel maintenance / platform gaps | Pin validated distribution, test binary-only installs, publish truthful platform matrix |
| Model/resource size | Bundle once; measure wheel; no duplicate age/gender `.dat` payload |
| Arbitrary legacy custom `.dat` files | Explicit conversion workflow; hash-based compatibility only for known models |
| Future dlib-free requirement | Conditional PR-8; do not disguise model substitution as equivalence |
| Removal of legacy sources loses reproducibility | Preserve pinned oracle/export recipe before removing `libs/` from the application tree |
| README screenshots treated as truth | Regenerate fixtures from executable behavior and original decoded inputs |

## 11. External references

Accessed 2026-09-22. Pin release/commit revisions in the implementation's model
and conversion manifests rather than relying on mutable `master` links.

- [Upstream models and license descriptions](https://github.com/davisking/dlib-models)
- [Official dlib release metadata](https://pypi.org/pypi/dlib/json)
- [dlib-bin metadata and wheel list](https://pypi.org/pypi/dlib-bin/json)
- [dlib wheel build project](https://github.com/alesanfra/dlib-wheels)
- [ONNX Runtime metadata](https://pypi.org/pypi/onnxruntime/json)
- [ONNX metadata](https://pypi.org/pypi/onnx/json)
- [NumPy metadata](https://pypi.org/pypi/numpy/json)
- [Pillow metadata](https://pypi.org/pypi/Pillow/json)
- [Hatchling metadata](https://pypi.org/pypi/hatchling/json)
- [pytest metadata](https://pypi.org/pypi/pytest/json)
- [Ruff metadata](https://pypi.org/pypi/ruff/json)
- [PyTorch metadata](https://pypi.org/pypi/torch/json)
- [face-recognition dependency metadata](https://pypi.org/pypi/face-recognition/json)
- [Current Python chip helper implementation](https://github.com/davisking/dlib/blob/master/tools/python/src/numpy_returns.cpp)
