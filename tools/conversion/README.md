# Dlib-to-ONNX conversion

This maintainer-only pipeline converts the repository's original dlib 19.20.0
age and gender weights into self-contained opset-18 ONNX graphs. It is not a
runtime dependency of the Python package.

The C++ exporter owns deserialization through the exact shared network aliases
in `tools/network_definitions.h`. Its fixed-locale XML contains every parameter,
including affine values omitted by dlib's generic parameter visitor. The Python
builder accepts only the two reviewed layer sequences and known source hashes.

## Rebuild

The source weights are read from the Git-ignored root `models/` directory.
Download [`dnn_age_predictor_v1.dat.bz2`](https://github.com/davisking/dlib-models/raw/master/age-predictor/dnn_age_predictor_v1.dat.bz2),
[`dnn_gender_classifier_v1.dat.bz2`](https://github.com/davisking/dlib-models/raw/master/gender-classifier/dnn_gender_classifier_v1.dat.bz2),
and [`shape_predictor_5_face_landmarks.dat.bz2`](https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2)
from `davisking/dlib-models` and unpack the `.dat` files there. The tools check
them against their known SHA-256 hashes.

Use the project environment and a CPU-only dlib build:

```bash
venv/bin/python -m pip install -r tools/conversion/requirements.txt
cmake -S tools/conversion -B build/conversion -DCMAKE_BUILD_TYPE=Release
cmake --build build/conversion --parallel 2
venv/bin/python tools/conversion/build_onnx.py \
  --exporter build/conversion/age_and_gender_export_dlib \
  --work-dir build/conversion/xml \
  --output tools/conversion/artifacts/v1
```

Run the frozen-fixture, stage, and live synthetic-reference checks:

```bash
venv/bin/python tools/conversion/validate_conversion.py \
  --bundle tools/conversion/artifacts/v1 \
  --probe build/conversion/age_and_gender_probe_dlib \
  --source-models models \
  --work-dir build/conversion/validation \
  --report tools/conversion/artifacts/v1/conversion-report.json
venv/bin/python -m pytest tests/parity/test_converted_networks.py
```

`validate_conversion.py` exits non-zero and records the failing comparisons in
the report when any gate fails. It is the only producer of that report; nothing
in it is written by hand.

## Reproducibility

Re-run `build_onnx.py` into another empty output directory and compare. Both
`.onnx` files are byte-identical across runs and across environments, because
the graph content carries no timestamps and tensors are serialized
deterministically.

`manifest.json` additionally records `tool_versions` (Python, NumPy, onnx,
onnxruntime) and a `converter_revision` that hashes the converter sources. Those
fields are provenance, so manifest byte-identity only holds within one
environment. A clean-environment re-export is verified by comparing the two
`.onnx` hashes and every manifest field outside `tool_versions`.

Measured on a second, independent environment:

| | project env | clean env |
| --- | --- | --- |
| Python | 3.13.15 | 3.10.15 |
| NumPy | 2.3.3 | 2.2.6 |
| onnx | 1.19.0 | 1.23.0 |
| `age-v1.onnx` / `gender-v1.onnx` | \- | **byte-identical** |

Only `onnx.version` and `tool_versions` differed. The graph contract fields
(`opset`, `ir_version`) matched, which is why `validate_conversion.py` checks the
contract rather than the onnx library version: pinning the version would reject
this bundle on any other toolchain.

## Package bundle

`build_bundle.py` assembles what the wheel installs. It re-verifies every
artifact the conversion manifest names, copies the two graphs and their notice,
adds the original five-point landmark model (pinned by SHA-256, copied rather
than converted, because dlib loads it directly), and writes the package
manifest:

```bash
venv/bin/python tools/conversion/build_bundle.py
```

The package manifest is the conversion manifest plus `"bundle_kind": "package"`
and a `shape_predictor` block; every other field is carried over byte for byte,
which `tests/unit/test_models.py` checks. Rebuild it whenever a new conversion
bundle is produced. The runtime reads this manifest, not the one under
`artifacts/`, and refuses the conversion directory by name because it has no
landmark model.

## Validated runtime setting

Version 1 uses `CPUExecutionProvider` with
`GraphOptimizationLevel.ORT_DISABLE_ALL` and single-threaded intra/inter-op
execution. These options are recorded in the bundle's `manifest.json` under
`runtime`, because parity depends on them and PR-3 builds its sessions from the
manifest rather than from this document.

`validate_conversion.py` measures all four optimization levels on every run and
writes the outcome per level. As checked in:

| Setting | Logits | Probabilities | Public results |
| --- | --- | --- | --- |
| `disabled` (selected) | pass | pass | pass |
| `basic` | **fail** | pass | pass |
| `extended` | **fail** | pass | pass |
| `all` | **fail** | pass | pass |

Every rejection is a logits-only failure against PR-1's `atol=1e-5, rtol=1e-4`;
the shipped `probabilities` output and the public age/gender results agree with
the oracle at every level. The initial release still selects `disabled` so the
strictest frozen threshold is met end to end. Revisiting that trade-off, and the
throughput it costs, belongs to PR-6, which owns measured performance work.

Note that the instrumented graph used for validation pins `logits` as an extra
output, which itself suppresses some fusion. The measurement is therefore of the
instrumented graph, not exactly of the shipped one.

`conversion-report.json`'s `onnxruntime` field records the version the report
was actually generated with. Regenerate the report (rerun the command above)
after upgrading the pinned `onnxruntime` dependency, rather than trusting a
report captured under an older version; this file was last regenerated under
`onnxruntime` 1.30.0 (the version PR-6's `benchmarks/benchmark_pipeline.py`
also measured with), superseding an earlier capture under 1.23.0.

## Thread-count investigation

Version 1's `runtime` block also fixes single-threaded ONNX Runtime execution
(`intra_op_num_threads`/`inter_op_num_threads`: 1). `validate_conversion.py`
sweeps a second axis alongside the graph-optimization one above —
`THREAD_CANDIDATES`/`SELECTED_THREAD_SETTING`, mirroring
`OPTIMIZATION_LEVELS`/`SELECTED_SETTING` — running the same frozen-fixture,
stage, and live synthetic-reference checks at each candidate
`intra_op_num_threads` value, with the graph optimization level held at the
shipped `disabled` setting. As checked in (`conversion-report.json`'s
`thread_variants`/`thread_stages`):

| Setting | Logits | Probabilities | Public results | Stage tensors |
| --- | --- | --- | --- | --- |
| `1x1` (shipped) | pass | pass | pass | pass |
| `2x1` | pass | pass | pass | pass |
| `4x1` | pass | pass | pass | pass |
| `6x1` | pass | pass | pass | pass |

Every candidate is bit-identical to `1x1` on the full validated corpus (frozen
real chips, synthetic batches at sizes 1/2/7/32, and every exported stage
tensor) — `ORT_DISABLE_ALL` keeps per-sample computation thread-count
independent and only parallelizes across batch rows, so raising the thread
count changes nothing PR-1's tolerances gate. This is the numerical half of
the spec/PR-6.md thread investigation; the runtime shipped default is
unchanged, and this alone would not be sufficient reason to change it. See the
next paragraph.

**Numerical parity is not the whole decision.** spec/PR-6.md separately
requires comparing one-worker and multiple-worker scenarios "to avoid
oversubscription": `AgeAndGender.predict()` serializes calls on *one*
instance with an internal lock, so a single shared instance (what the
existing `tests/integration/test_api.py::ConcurrencyTests` exercise) can never
oversubscribe a CPU by itself — but this package explicitly supports multiple
*independent* instances/processes, each with its own multi-threaded ONNX
Runtime session, and that scenario can. `benchmarks/concurrency_benchmark.py`
measures exactly that: real `AgeAndGender` instances, one to four running
concurrently from independent threads, at each thread candidate, on 5- and
32-face scenarios. Measured on this repository's development machine (AMD
Ryzen 5 3600, 12 logical cores; see `benchmarks/report-pr6.md`'s thread/
concurrency section for the full table and machine caveat):

- One worker: raising `intra_op_num_threads` helps the 32-face scenario
  (8.7 → ~11–12 calls/s from `1x1` to `4x1`/`6x1`) and is roughly flat or
  slightly worse for the 5-face scenario.
- Four independent concurrent workers: raising `intra_op_num_threads`
  regresses both scenarios past `2x1` — 5-face throughput drops from 39.3
  calls/s at `1x1` to 20.3 calls/s at `6x1`; 32-face throughput drops from a
  `2x1` peak of 21.4 calls/s to 9.3 calls/s at `6x1`, below the `1x1` baseline
  of 19.0.

**Decision: the shipped default stays `1x1`.** A higher thread count is
numerically safe in isolation but measurably regresses the concurrent-caller
throughput this package is explicitly designed to support once more than one
or two independent sessions run at once, for a single-worker gain that is
modest end to end (detection, not the CNNs, dominates whole-image latency;
see `benchmarks/report-pr6.md`). This mirrors how the graph-optimization
rejections above are handled: a candidate that measurably regresses something
this package's own contract cares about is documented and rejected here
instead of silently left unmeasured, per spec/PR-6.md's "If parity permits no
further numeric optimization, ship the simpler passing configuration with the
measured report" — read together with its own oversubscription requirement,
this is the same escape hatch applied to a throughput regression instead of a
parity failure. `SUPPORTED_RUNTIME` in `_models.py`, the shipped
`manifest.json` files, and this document's "Validated runtime setting" table
above are unchanged.

## Stage comparison

`probe_dlib.cpp` exports intermediate dlib tensors and `validate_conversion.py`
compares each one against the matching converted tensor, so a conversion error
is localized to a block instead of only showing up in the final logits. This is
what checks the residual branches: the pooled skip branches (`layer_32`,
`layer_13`) and the zero-extended additions that follow them.

dlib disables `get_output()` on any layer that an in-place layer is stacked on,
so the exported indices are the tops of in-place chains. Each one maps onto the
converted tensor the builder named `layer_<index>_...`.

Internal activations are compared at `atol=1e-4, rtol=1e-4` rather than PR-1's
proposed `atol=1e-5`. PR-1 stated that its floating thresholds were proposals to
ratify against measurement; the measurement on the deepest 256-channel stage is:

```text
max|dlib - exact float64| = 5.10e-05
max|onnx - exact float64| = 3.28e-05   <- the converted graph is the closer one
max|onnx - dlib|          = 4.96e-05
stage value range          [0.00, 42.03]
```

That is float32 accumulation-order noise across roughly 2300 multiply-accumulates,
about 13 ULP at that magnitude, landing on post-ReLU near-zero elements where
`rtol` contributes nothing. An `atol` of 1e-5 sits below the noise floor of the
reference implementation itself. Logits, probabilities, the age expectation and
the public results all keep PR-1's original thresholds unchanged. A structural
conversion error moves these tensors by orders of magnitude more than this.

## Corpus and its limits

The frozen real-chip corpus is 11 faces over 3 images, which is 8 distinct chips
once the duplicate and repeated boxes in `explicit-boxes.golden.json` are
collapsed. PR-1's fixture manifest explicitly defers the representative
30-image/100-face corpus to release hardening. Synthetic black, white,
channel-ramp and seeded-random chips at batch sizes 1, 2, 7 and 32, checked
against the live dlib probe, carry the rest of the numerical coverage.

The conversion evidence is correspondingly narrow on real faces. Widening it is
PR-4's frontend corpus work; the numbers in the report should be read with that
in mind.

## PR-8 feasibility prototype: shape predictor and chip extraction in NumPy

This is separate, unrelated tooling that happens to live alongside the ONNX
conversion pipeline above: a maintainer-only investigation into whether
dlib's 5-point shape predictor and face-chip extraction can be ported to
pure Python/NumPy (spec/PR-8.md's "Initial feasibility deliverable"). It is
not a runtime dependency, not part of the shipped package, and does not
change anything the ONNX conversion pipeline above produces or validates.

Rebuild the two new C++ tools alongside the existing ones (same CMake
project):

```bash
cmake -S tools/conversion -B build/conversion -DCMAKE_BUILD_TYPE=Release
cmake --build build/conversion --parallel 2 \
  --target age_and_gender_export_shape_predictor age_and_gender_probe_shape_predictor
```

Export the bundled shape predictor's trained parameters and package them.
The builder hashes the source `.dat`, refuses anything but the pinned model,
and runs the exporter on that same file itself, so the manifest's source
hash always describes the packaged parameters:

```bash
venv/bin/python tools/conversion/build_shape_predictor.py \
  --exporter build/conversion/age_and_gender_export_shape_predictor \
  --output-dir tools/conversion/artifacts/shape-predictor-v1
```

Run the full validator. `--probe` is required: the report's top-level
`passed` is true only if every section ran and passed — frozen-corpus exact
match, bit-for-bit per-cascade stage agreement with the compiled oracle, the
retained regression cases (`regression-cases.json`), a reproducible
live-`dlib-bin` synthetic sweep (boundary and deep-pyramid cases), bit-exact
`chip_details` geometry, and the zero-angle raw-copy decision.
`--write-stage-fixture` also refreshes `stage-trace.json`, the oracle trace
the CI test replays without the probe:

```bash
venv/bin/python tools/conversion/validate_shape_predictor.py \
  --probe build/conversion/age_and_gender_probe_shape_predictor \
  --write-stage-fixture \
  --report tools/conversion/artifacts/shape-predictor-v1/shape-predictor-report.json
venv/bin/python -m pytest tests/parity/test_numpy_shape_predictor_feasibility.py
```

Time the NumPy path against the wheel (spec/PR-6.md protocol by default:
10 warmups, 100 iterations, five repeats; `--markdown` writes the summary
table used in the feasibility report):

```bash
venv/bin/python tools/conversion/benchmark_numpy_frontend.py \
  --output tools/conversion/artifacts/shape-predictor-v1/numpy-frontend-benchmark.json \
  --markdown build/conversion/numpy-frontend-benchmark.md
```

Every checked-in report records the prototype `code_revision`, and the CI
test fails if any of them is stale relative to the current sources, so rerun
the validator and benchmark after changing any file listed in
`shape_predictor_evidence.PROTOTYPE_SOURCES`.

See `shape-predictor-feasibility-report.md` for the results and go/no-go
discussion.
