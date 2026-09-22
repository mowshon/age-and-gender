# Dlib-to-ONNX conversion

This maintainer-only pipeline converts the repository's original dlib 19.20.0
age and gender weights into self-contained opset-18 ONNX graphs. It is not a
runtime dependency of the Python package.

The C++ exporter owns deserialization through the exact shared network aliases
in `tools/network_definitions.h`. Its fixed-locale XML contains every parameter,
including affine values omitted by dlib's generic parameter visitor. The Python
builder accepts only the two reviewed layer sequences and known source hashes.

## Rebuild

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
  --source-models example/models \
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
