# PR-2 — Convert the existing age and gender networks to ONNX

**Depends on:** PR-1.

**Risk:** high; primary migration feasibility gate.

**Deliverable:** reproducibly generated, parity-validated ONNX files for the two
existing models, plus conversion manifests.

## Goal

Preserve the actual learned parameters and graph while making inference
available through a maintained Python runtime. Conversion is maintainer work;
package users never run CMake or a model exporter.

## Work

### 1. Export from the trusted dlib reader

- Add `tools/conversion/export_dlib.cpp`, using the oracle's exact network types.
  Avoid divergent copies of network definitions; share a maintainer-only header.
- Deserialize the two files whose hashes are recorded in the investigation.
- Export input means, layer order and connections, dimensions, convolution/FC
  parameters, affine parameters/modes, and multiply scalars.
- A direct `net_to_xml()` export is a viable starting representation. Use a
  fixed locale and verify nine-significant-digit float32 round-tripping. If
  explicit binary tensors are added, give each tensor its dtype, shape, endian,
  and checksum in the manifest.
- Do not rely solely on `visit_layer_parameters()`: dlib 19.20's affine layers
  return an empty parameter tensor through that API. Export their real scale
  and offset through XML or a reviewed serialization reader.
- Fail on unexpected layer types, missing parameters, counts, or source hashes.
  This converter targets these architectures, not every possible dlib network.

### 2. Build standard ONNX graphs in Python

Use `onnx` helpers directly unless a proven simpler exporter is found. Avoid a
runtime dependency on PyTorch or Caffe.

| Dlib feature | Conversion requirement |
| --- | --- |
| Convolution | OIHW float32 weights and bias; actual stride/padding, not generic `SAME` |
| Affine | Per-channel or per-feature multiply/add with correct broadcast shape; do not refit BN |
| ReLU | Standard ReLU |
| Max/average pooling | Explicit kernel/stride/padding, floor dimensions; verify divisor/boundary behavior |
| Global average pooling | Global spatial reduction, preserving channel order |
| `tag` / `skip` | Resolve graph edges, accounting for reused tag IDs in nested blocks |
| `add_prev` | Zero-pad trailing missing channels/rows/columns of each branch to maximum shape, then add |
| `multiply` | Preserve both loaded gender scalars, currently 0.5 |
| Fully connected | Flatten NCHW correctly; dlib weights are input-by-output, map transpose flags explicitly |
| Inference softmax | Class-axis softmax, outputs `[N,81]` or `[N,2]` |

For the first age downsampling block, the main branch is 6×6 while the pooled
skip branch is 7×7 at the standard input size. A stock ResNet projection or an
unqualified ONNX `Add` is wrong. Pad to the common extent without shifting the
top-left origin. Verify all branch tensors against the oracle.

### 3. Fix the exported model contract

```text
age-v1.onnx:
  input:  images, float32, [N, 3, 64, 64], already normalized
  output: probabilities, float32, [N, 81]

gender-v1.onnx:
  input:  images, float32, [N, 3, 32, 32], already normalized
  output: probabilities, float32, [N, 2], labels [female, male]
```

- Dynamic positive batch size; fixed channel/spatial dimensions.
- Python owns RGB-to-NCHW normalization and final age/gender formatting.
- Include softmax once in the graph; never apply it again in runtime Python.
- Initial graphs keep affine/scaling operations explicit and float32. Folding
  them into convolutions or enabling reduced precision is deferred to PR-6.
- Start with an explicitly selected stable opset such as 18 and an IR supported
  by the chosen runtime; confirm through checker and real session creation.
  Do not rely on the newest `onnx` package's default IR.
- Use CPUExecutionProvider as the release reference. Model files are
  self-contained rather than depending on untracked external tensor files.

### 4. Generate the model manifest

Record artifact schema version, bundle ID, task, source `.dat` hash, target ONNX
hash/size, converter revision, dlib revision, tool versions, opset/IR, input/output
names/shapes/dtypes, normalization means/scale, class labels/age weights, padding,
and model license attribution. Export artifacts reproducibly; remove timestamps
or other nondeterministic graph metadata from the hashed graph content.

### 5. Validate conversion independently

- Run `onnx.checker`, shape inference, and session creation.
- Compare every required stage against oracle fixtures; use instrumented debug
  graph outputs to locate errors without shipping all activations in the model.
- Test real chips, black/white/channel-ramp chips, seeded random chips, and
  batches of sizes 1, 2, 7, and 32.
- Compare batch inference against individual inference, including public
  postprocessing. Do not assume dynamic batching is numerically identical.
- Validate unoptimized CPU execution first, then the intended runtime session
  optimization settings separately.
- Re-export in a clean environment and compare hashes/parameters.

## Proposed files

- `tools/conversion/export_dlib.cpp`, shared network-definition header.
- `tools/conversion/build_onnx.py`, `validate_conversion.py`, README and locked
  maintainer dependencies.
- Versioned generated model bundle and manifest, staged for inclusion in PR-3.
- `tests/parity/test_converted_networks.py` and conversion-shape tests.

## Acceptance

- [x] Both networks load the original hashes and export all parameters.
- [x] Every layer is accounted for; no invented projection/normalization layers.
- [x] Oracle and ONNX satisfy PR-1 tensor and final-output requirements, with one
      ratified threshold change recorded under "Implementation status".
- [x] Dynamic batch results pass the same requirements.
- [x] Conversion is reproducible without undocumented local files.
- [x] No Caffe/PyTorch/dlib build dependency is added to the user runtime.
- [x] A written conversion report includes maximum errors and any failed settings.

**Stop condition:** unexplained probability or public-output drift blocks the
migration. Do not hide drift with arbitrary epsilons, clipping confidence to 100,
changing age rounding, or substituting another model. Keep the old runtime
available while resolving the conversion.

## Implementation status

Delivered on `refactor/pr-2-onnx-conversion`. Artifacts live in
`tools/conversion/artifacts/v1/` and are staged for PR-3.

### Verified

- The graph was checked against an independent float64 reimplementation written
  directly from `net_to_xml` output, not from the converter. It reproduces the
  oracle's C++ float32 logits to 2.10e-05 (age) and 5.14e-06 (gender) across all
  11 frozen faces, which confirms convolution padding, affine gamma/beta order,
  fully connected orientation, `add_prev` zero-extension, pooling and tag/skip
  resolution independently of `build_onnx.py`.
- Nine-significant-digit XML round-trips float32 exactly: 2,689,617 parameter
  tokens checked, zero mismatches.
- Both `.onnx` files re-export byte-identically.
- Stage-by-stage comparison covers 11 age and 7 gender intermediate tensors,
  including both pooled residual skip branches.

### Threshold ratification

PR-1 proposed `atol=1e-5, rtol=1e-4` for "intermediate logits / floating
activations" and stated the floating thresholds were proposals to ratify against
a stability study. Measurement on the deepest 256-channel stage:

```text
max|dlib - exact float64| = 5.10e-05
max|onnx - exact float64| = 3.28e-05   <- the converted graph is the closer one
max|onnx - dlib|          = 4.96e-05
```

This is float32 accumulation-order noise, roughly 13 ULP at that magnitude, on
post-ReLU near-zero elements where `rtol` contributes nothing. Internal
activations are therefore compared at `atol=1e-4, rtol=1e-4`. **Logits,
probabilities, the age expectation and all public results keep PR-1's original
thresholds unchanged.** This is a ratification against measurement, not an
epsilon chosen to make a failing comparison pass: the conversion is verifiably
closer to exact arithmetic than the reference it is being compared against.

### Runtime setting

`CPUExecutionProvider`, `ORT_DISABLE_ALL`, single-threaded intra/inter-op. These
are recorded in the bundle manifest under `runtime` so PR-3 builds sessions from
the bundle rather than from prose.

All four optimization levels are measured on every validation run. `basic`,
`extended` and `all` fail PR-1's logits threshold; all three still agree with the
oracle on `probabilities` and on every public result. `disabled` is selected so
the strictest frozen threshold is met end to end. The throughput cost of that
choice is PR-6's to revisit.

### Carried risk

The real-chip corpus is 11 faces over 3 images, 8 distinct chips once duplicate
boxes collapse. PR-1's fixture manifest defers the representative
30-image/100-face corpus to release hardening, so this gate rests on narrow real
data plus synthetic chips checked against the live dlib probe. PR-4 owns
widening it, and the conversion report records the limitation explicitly.
