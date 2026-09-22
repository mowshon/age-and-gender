# PR-6 — Optimize measured bottlenecks while retaining parity

**Depends on:** PR-5 and PR-1 baseline.

**Risk:** medium/high for changes to floating-point kernels or batching.

**Deliverable:** reproducible performance report and only validated improvements.

## Goal

Make the Python package efficient without changing the model results. The old
code has clear avoidable work, but no speedup magnitude was measured during the
investigation.

## Benchmark first

Add `benchmarks/benchmark_pipeline.py` with machine-readable JSON output. Measure:

- Fresh-process import and model/session initialization separately.
- First prediction and warmed prediction separately.
- Image validation/copy, detection, landmarks, each chip size, normalization,
  each network, and postprocessing.
- Full images with zero, one, five, and many faces; explicit-box calls separately
  from auto-detection; chip batches of 1, 2, 8, and 32.
- Median and p95 latency, faces/second, peak RSS, and model/session counts.
- Repeated steady-state calls to detect accidental reinitialization or growth.

Use identical decoded RGB arrays, rectangles, source weights, hardware, and
execution conditions for old and new implementations. Report CPU, OS, compiler,
Python/dependency versions, thread counts, batch size, provider, graph settings,
warmup/sample counts, and power/thermal conditions where available.

For steady-state numbers, use at least 10 warmups and 100 measured iterations per
case, repeated across five runs on a dedicated machine. Cold-start samples must
use fresh processes. Keep instrumented tensor-dump mode out of timed loops.

Compare the legacy public path including per-call subnet copies; if a standalone
oracle approximates it, label that benchmark separately from the actual extension.
Do not conflate different detector face counts with inference speed improvements.

## Optimization order

1. Verify the PR-3 lifecycle eliminates per-predict model loads, session creation,
   and the old two-subnet copies.
2. Remove redundant application-level image/chip copies; keep immutable model
   metadata and reuse safe internal buffers only with explicit ownership.
3. Batch **already individually extracted** same-size chips in bounded batches.
   - Preserve input ordering when joining chunks.
   - Benchmark batch-size thresholds; one face need not take a batch-setup path.
   - Bound memory on high-face-count images and document the default maximum.
   - Validate batched vs individual raw probabilities and public outputs.
4. Vectorize float32 preprocessing across faces and preserve sequential class
   accumulation for age. The 81-step age reduction is cheap relative to CNNs.
5. Tune ONNX Runtime intra/inter-op threads and sequential/parallel execution.
   Compare one-worker and multiple-worker scenarios to avoid oversubscription.
6. Evaluate graph optimizations/fusions one setting at a time. Keep the current
   graph settings if an ostensibly faster setting changes public output.

The original templates use affine and multiply operations whose rearrangement
can change rounding. Explicit affine folding, new CPU kernels, and model graph
fusion require the complete parity suite, not just equivalent algebra.

## Constraints

- Preserve individual chip extraction. The known `get_face_chips` counterexample
  is a required regression fixture.
- Default input size, padding, detector threshold, detector upsampling, weights,
  labels, and postprocessing stay fixed.
- Initial default remains float32 CPU. FP16, INT8, quantization, GPU providers,
  and alternate face models require separate opt-in feature work and evidence.
- Do not change the existing empty-box behavior to bypass detection for benchmarks.
- Run the whole parity corpus after each numeric optimization. A boundary mismatch
  disqualifies that optimization from the compatibility default.

## Performance acceptance

- Mandatory: exact final-output parity and the existing tensor requirements.
- Mandatory: model/session construction is absent from steady-state prediction.
- Mandatory: on the named reference machine, warmed median one-face and five-face
  latency should not regress by more than 10% versus the measured legacy public
  baseline; repeat the benchmark to distinguish noise. Investigate regressions
  and keep the last passing implementation if needed.
- Target: a material measured CNN throughput gain for multi-face input; 2× is a
  useful investigation target, **not a pre-established claim or universal SLA**.
- Report cold-start and memory tradeoffs even when warmed throughput improves.
- Performance gates run on controlled hardware, not timing-sensitive assertions
  on shared CI runners. Ordinary CI can check session reuse and bounded behavior.

If parity permits no further numeric optimization, ship the simpler passing
configuration with the measured report. Installation simplification is already
a meaningful result; do not weaken output compatibility to invent a speed win.

## Proposed files

- `benchmarks/benchmark_pipeline.py`, benchmark README and result schema.
- Changes to `_inference.py` / `_postprocess.py` only where supported by profiling.
- `tests/parity/test_batching.py` and targeted resource-lifecycle tests.
- Before/after report with raw measurements and final selected defaults.

## Acceptance

- [ ] Benchmark commands and environment reproduce the report.
- [ ] Default and all supported batch sizes/chunk boundaries pass parity.
- [ ] Selected thread/optimization defaults have measured justification.
- [ ] No unbounded model reloads, leaked buffers, or input mutation.
- [ ] Release notes state observed workload-specific gains and limitations.
