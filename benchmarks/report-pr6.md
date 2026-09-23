# PR-6 performance report

Machine: this repository's development VM (AMD Ryzen 5 3600, 12 logical
cores, Linux 6.8, CPython 3.13.15) — **not** a dedicated benchmark host; see
`benchmarks/README.md`'s "Machine caveat". All numbers below are reproducible
with the commands in that file and are recorded in full, with per-repeat
medians and p95s, in
`benchmarks/results/python-pipeline-linux-x86_64-python313.json`
(`--warmup 10 --iterations 50 --repeats 3 --cold-runs 5
--steady-state-iterations 200`), and, for the thread/concurrency section
below, in `tools/conversion/artifacts/v1/conversion-report.json` and
`benchmarks/results/concurrency-report.json`.

## Starting point

PR-3/PR-5 already did most of the structural work spec/PR-6.md's "Optimization
order" asks for: persistent ONNX Runtime sessions (`_inference.py`'s
`NeuralNetwork.ensure_loaded()`), no per-call subnet copies, no redundant
image/chip copies (`_images.as_rgb_array` already returns a contiguous input
unmodified; `_faces.FaceFrontend` extracts each chip once), and vectorized
per-face preprocessing across a whole batch at once (`prepare_batch()`). That
left two things this PR actually changed:

1. **Bounded batch chunking** (`_inference.py`): `NeuralNetwork.probabilities()`
   handed ONNX Runtime one unbounded batch — every face in the image, in one
   call. A very high face count would build one correspondingly large float32
   tensor and hand it to the session in a single call. `probabilities()` now
   splits any call over `DEFAULT_MAX_BATCH_SIZE` (32) into ordered,
   concatenated chunks of at most that many chips. 32 was not picked freely:
   it is the largest batch size `tools/conversion/validate_conversion.py` and
   `tests/parity/test_converted_networks.py::test_dynamic_batches_match_frozen_outputs`
   already validate numerically, so the chunk boundary carries existing parity
   evidence. `tests/parity/test_batching.py` adds chunk-boundary-specific
   coverage: chunked calls reproduce one unbounded call bit-for-bit-adjacent
   (within PR-1's probability tolerance) at bounds 1/2/3/7/32/33, chunk splits
   never reorder faces, the public `face_predictions()` output is unaffected,
   and chunking never creates a second session or re-reads model weights.
2. **This benchmark suite** (`benchmarks/benchmark_pipeline.py`), which is the
   deliverable spec/PR-6.md asks for regardless of what else changes.

Nothing else in `_inference.py`/`_postprocess.py` changed. Per spec/PR-6.md's
own escape hatch ("If parity permits no further numeric optimization, ship the
simpler passing configuration with the measured report"), the measurements
below explain why: the pipeline's cost is dominated by dlib's HOG detector, not
by the CNNs this PR could safely touch, and both runtime-configuration axes
spec/PR-6.md calls out — graph optimization level and thread count — were
measured and rejected for the shipped default rather than left unmeasured;
see "Evaluated but not changed" below.

## Old vs. new: the acceptance gate

spec/PR-6.md: *"warmed median one-face and five-face latency should not
regress by more than 10% versus the measured legacy public baseline."*

| Case | Faces | Legacy warm median | New warm median | Change |
| --- | ---: | ---: | ---: | ---: |
| `test-image.jpg`, explicit single box | 1 | 46.28 ms | 4.80 ms | **-90%** |
| `test-image-2.jpg`, detected | 2 | 116.35 ms | 28.61 ms | **-75%** |
| `test-image.jpg`, detected | 5 | 325.18 ms | 82.77 ms | **-75%** |

The legacy numbers are spec/PR-1.md's frozen oracle measurements (5- and
2-face cases) plus one new single-face oracle capture, both explained in
`benchmarks/README.md`. All three cases clear the 10% regression gate by a
wide margin — this was expected: the legacy path re-copies the entire input
image and re-copies both subnets on every `predict()` call (see
spec/INVESTIGATION.md item 1), and PR-3 already removed both per-call costs
before this PR started. This PR's own contribution (bounded chunking) does not
change single- or few-face latency at all — chunking only activates above 32
faces — so this gate was already met going into PR-6; it is reported here
because spec/PR-6.md requires it regardless of which PR produced the win.

**What this table is, precisely.** Two caveats spec/PR-6.md itself calls for,
stated explicitly rather than left implicit:

- **Oracle, not the compiled extension.** "Legacy warm median" here is
  `tools/legacy/oracle.cpp`'s standalone C++ reproduction of the original
  pybind11 extension's `predict()` path (same network types, same per-call
  subnet copies, same frontend) — not a timed run of the actual compiled
  `age_and_gender` C++ extension itself. spec/PR-6.md allows this explicitly
  ("if a standalone oracle approximates it, label that benchmark separately
  from the actual extension"); this paragraph is that label.
- **Reduced protocol, not the certified gate.** This capture used 50
  iterations × 3 repeats on this repository's shared development VM, not
  spec/PR-6.md's "at least 10 warmups and 100 measured iterations... repeated
  across five runs on a dedicated machine." The margins above (75–90%) are far
  wider than run-to-run noise at this iteration count could plausibly explain
  — see `per_repeat_median_ms` in the linked JSON for the actual spread — but
  that is evidence toward the gate, not a substitute for the controlled-
  hardware, full-protocol run spec/PR-6.md's acceptance section requires
  before this is treated as a certified pass. See "Limitations" below.

## Where the time actually goes

| Stage (5-face `test-image.jpg`) | Median |
| --- | ---: |
| Detection (`FaceFrontend.detect`) | 63.92 ms |
| Age network, 5 chips (`NeuralNetwork.run`) | 10.63 ms |
| Gender network, 5 chips | 3.41 ms |
| Landmarks, one face | 0.41 ms |
| Postprocessing, 5 faces | 0.18 ms |
| Chip extraction, one face (both sizes) | 0.11 ms |
| Image validation (already-contiguous array) | 0.001 ms |

Full-image comparison, same source image, explicit boxes vs. detection:

| Scenario | Faces | Median | Faces/second |
| --- | ---: | ---: | ---: |
| Detect, 0 faces (`dogs.jpg`, 900x916) | 0 | 60.80 ms | — |
| Detect, 5 faces (`test-image.jpg`, 1100x825) | 5 | 82.77 ms | 60.4 |
| Detect, 2 faces (`test-image-2.jpg`, 634x435) | 2 | 28.61 ms | 69.9 |
| Explicit boxes, 1 face | 1 | 4.80 ms | 208.2 |
| Explicit boxes, 5 faces (detector skipped) | 5 | 19.00 ms | 263.2 |
| Explicit boxes, 50 faces (detector skipped) | 50 | 184.68 ms | 270.7 |

Detection cost tracks image *pixel count*, not face count (the 0-face,
900x916 `dogs.jpg` costs more detector time than the 2-face, 634x435 image).
Skipping it (explicit boxes) drops 5-face latency from 82.77 ms to 19.00 ms —
detection is roughly 3/4 of whole-image latency on these examples, exactly the
"HOG frontend may dominate after CNN optimization" spec/INVESTIGATION.md
predicted. This is also why bounded chunking's effect is invisible at
end-to-end scale for these example images: the CNNs it touches are already the
smaller cost, and explicit-box calls (which skip detection) scale near-linearly
with face count regardless (4.80 ms/face at 1 face, 3.80 ms/face at 5, 3.69
ms/face at 50), which is what "chunking doesn't change results or blow up
memory past 32 faces" is supposed to look like, not a throughput win by itself.

## Batch-size sweep (`NeuralNetwork.probabilities()`, the real chunking entry point)

An earlier version of this sweep timed `NeuralNetwork.run()` directly on a
pre-built batch-64 tensor and then *computed* a "chunked calls" figure from
`ceil(64 / 32)` arithmetic — it never actually called the chunking code
(`probabilities()`), since `run()` always makes exactly one ONNX Runtime call
for whatever tensor it is handed. That meant the batch-64 row measured one
unbounded batch-64 call, not chunking, and the report's "two 32-row calls"
claim was inferred, not observed. The sweep below instead times
`probabilities()` directly — the same method `predict()` calls — and observes
each ONNX Runtime call's actual batch size by spying on the session
(`benchmark_pipeline.py::_observed_call_batch_sizes`), so "chunked calls" and
"observed call batch sizes" below are what happened, not what the boundary
arithmetic implies should happen.

| Batch | Gender median | Gender faces/s | Age median | Age faces/s | Chunked calls | Observed call batch sizes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.74 ms | 1357 | 2.26 ms | 443 | 1 | `[1]` |
| 2 | 1.42 ms | 1411 | 4.42 ms | 453 | 1 | `[2]` |
| 8 | 5.55 ms | 1442 | 17.66 ms | 453 | 1 | `[8]` |
| 32 | 22.04 ms | 1452 | 69.30 ms | 462 | 1 | `[32]` |
| 64 | 44.38 ms | 1442 | 137.06 ms | 467 | 2 | `[32, 32]` |

Throughput is essentially flat from batch 1 to 32 (single-threaded execution;
see below), which is expected and fine — batching's win here is fewer
Python/ONNX-Runtime call crossings for a multi-face image, not sub-linear
per-sample cost. The batch-64 row is confirmed, not assumed, to be two
size-32 ONNX Runtime calls; it costs about what two separate 32-batches would
(compare 2 × the batch-32 median against the batch-64 median for each task),
i.e. chunking adds no measurable overhead beyond the extra call itself. Note
what this bounds and what it does not: chunking bounds the size of each
individual ONNX Runtime call and its input tensor, not the total memory
`predict()` holds for one image — `api.py`'s `predict()` still extracts and
retains every face's chips up front before calling `probabilities()` (see
`_inference.py`'s `DEFAULT_MAX_BATCH_SIZE` docstring); a 32x32x3/64x64x3 uint8
chip is small (3–12 KiB), so this scales linearly and modestly with face
count regardless of the chunk bound, but it is not itself bounded by
`DEFAULT_MAX_BATCH_SIZE`.

## Steady state and resource lifecycle

200 repeated `predict()` calls on one predictor: `age`/`gender` sessions,
frontend, and the process-wide bundle cache (`bundled_models.cache_info()`
stayed at 1 hit / 1 miss throughout) were all the same *objects* at the end as
at the start (identity-checked, `is`, not just equal), and
`measure_memory_and_counts()` counted exactly one live age session, one
gender session, one detector, and one shape predictor — no accidental
reinitialization was observed. First-window vs. last-window median latency
ratio was 0.977 (flat, not the multi-x slowdown accidental reinitialization or
a growing internal buffer would produce). Peak RSS for the whole run was
180,528 KiB, but that figure is `ru_maxrss` sampled once after the run — the
process's lifetime peak, not a before/after delta for the run itself — so it
is reported as one data point, not evidence of "no memory growth" on its own;
the reinitialization/identity checks above are what actually rule out
reinitialization, and they say nothing about heap fragmentation or allocator
behavior below the object-identity level.

## Cold start

Fresh-process medians, 5 subprocesses: import 111.2 ms, `AgeAndGender()`
construction 2.3 ms (does no I/O by design), model+session+frontend
initialization 315.1 ms, first `predict()` call after that 85.9 ms, second
call 86.1 ms — consistent with "first" already being close to "warmed"
because initialization (the actually expensive lazy step) already ran before
the timed first call. Total cold path (import → construct → init → first
predict) is roughly 515 ms.

## Evaluated but not changed

**Graph optimization level.** Already evaluated in PR-2/PR-3
(`tools/conversion/README.md`'s "Validated runtime setting" table): `basic`,
`extended`, and `all` all fail the internal logits tolerance
(`atol=1e-5, rtol=1e-4`) that the shared acceptance contract requires for
intermediate tensors, even though the shipped `probabilities` output and
public results still agreed with the oracle on that narrow corpus. Per
spec/PR-6.md — *"Keep the current graph settings if an ostensibly faster
setting changes public output"* combined with the acceptance contract's
intermediate-tensor requirement — `ORT_DISABLE_ALL` stays. This PR did not
re-run that evaluation; it had already happened and nothing here changes its
inputs.

**Thread count and concurrency.** An earlier draft of this report described a
one-off, non-shipped spot check here (11 frozen faces, one machine) and left
the full investigation as a follow-up. That investigation is now done, as
checked-in, reproducible tooling rather than a one-off, and the result is a
documented rejection, the same way the graph-optimization axis above is
handled — not a gap.

*Numerical parity.* `tools/conversion/validate_conversion.py`'s
`THREAD_SETTINGS` was previously a constant, never varied; it now has a
thread-count axis (`THREAD_CANDIDATES`/`SELECTED_THREAD_SETTING`) mirroring
`OPTIMIZATION_LEVELS`/`SELECTED_SETTING`, run over the *same* frozen-fixture,
stage, and synthetic-reference checks as the optimization axis — not just the
11-face corpus. As checked into
`tools/conversion/artifacts/v1/conversion-report.json`'s `thread_variants`/
`thread_stages` (see `tools/conversion/README.md`'s "Thread-count
investigation" section for the full table): `1x1` (shipped), `2x1`, `4x1`, and
`6x1` are all bit-identical on every check. `ORT_DISABLE_ALL` keeps
per-sample computation thread-count-independent and only parallelizes across
batch rows, confirming the earlier spot check on the full validated corpus.

*Concurrent-caller throughput.* Numerical parity alone does not settle
whether to ship a higher thread count: spec/PR-6.md separately requires
comparing one-worker and multiple-worker scenarios to avoid oversubscription.
`ConcurrencyTests` (`tests/integration/test_api.py`) is **not** that
comparison and an earlier draft of this report was wrong to cite it as one:
`AgeAndGender.predict()` serializes every call on one instance behind an
internal lock (`api.py`), so those tests — which share one instance — prove
correctness under that lock, never two independently multi-threaded ONNX
Runtime sessions running at the same time. This package explicitly supports
multiple *independent* instances/processes instead, each with its own
session, and that is the scenario that can actually oversubscribe a CPU.
`benchmarks/concurrency_benchmark.py` measures it directly: real
`AgeAndGender` instances (one to four, each independent, each on its own
thread) at every thread candidate, on 5- and 32-face scenarios, aggregate
calls/second over a fixed wall-clock window. Measured on this machine (AMD
Ryzen 5 3600, 12 logical cores; `benchmarks/results/concurrency-report.json`
has the full data):

| Faces | Threads | 1 worker (calls/s) | 4 workers (calls/s) |
| ---: | --- | ---: | ---: |
| 5 | `1x1` | 11.49 | 39.26 |
| 5 | `2x1` | 12.06 | 38.47 |
| 5 | `4x1` | 11.72 | 26.31 |
| 5 | `6x1` | 10.75 | 20.25 |
| 32 | `1x1` | 8.71 | 18.99 |
| 32 | `2x1` | 10.67 | 21.44 |
| 32 | `4x1` | 11.93 | 13.57 |
| 32 | `6x1` | 11.40 | 9.32 |

One worker gets a real gain from more threads on the 32-face scenario (8.71 →
up to 11.93 calls/s) and is flat-to-slightly-worse on the 5-face scenario.
Four independent concurrent workers — the oversubscription scenario
spec/PR-6.md asks about — regress past `2x1` on both scenarios: 5-face
throughput falls from 39.26 calls/s at `1x1` to 20.25 at `6x1`; 32-face
throughput peaks at `2x1` (21.44) and falls to 9.32 at `6x1`, *below* the
`1x1` baseline of 18.99. This closely reproduces the pattern an earlier,
non-shipped spot check for this report found by hand on the same machine.

**Decision: `SUPPORTED_RUNTIME` in `_models.py` stays `1x1`.** A higher
thread count is numerically safe in isolation but measurably regresses
concurrent-caller throughput once more than one or two independent sessions
run at once — exactly the scenario this package is designed to support — for
a single-worker gain that would also be modest end to end, since detection
(not the CNNs) dominates whole-image latency (see "Where the time actually
goes" above). Per spec/PR-6.md's escape hatch ("if parity permits no further
numeric optimization, ship the simpler passing configuration with the
measured report"), applied here to a throughput regression rather than a
parity failure: the shipped manifests, `SUPPORTED_RUNTIME`, and
`tools/conversion/README.md`'s "Validated runtime setting" table are
unchanged.

## Limitations

- Single development machine, not spec/PR-6.md's "dedicated machine" /
  "reference machine"; see `benchmarks/README.md`. None of the numbers in
  this report — including the acceptance-gate table and the thread/
  concurrency table — are the certified controlled-hardware run spec/PR-6.md's
  acceptance section describes; see "What this table is, precisely" above.
- The reduced iteration count actually used for `benchmark_pipeline.py`'s
  numbers (50 iterations × 3 repeats, not the documented default of 100 × 5)
  trades some statistical confidence for a runtime that fits this session;
  every full-image and batch-sweep median above is still a median over 150
  timed calls (50 × 3), not a single sample, and `per_repeat_median_ms` in the
  JSON report shows run-to-run spread.
- The concurrency benchmark measures a fixed 2-second wall-clock window per
  cell (`benchmarks/concurrency_benchmark.py --duration`, not a fixed
  iteration count); that window trades some statistical confidence for a
  bounded total run time the same way the reduced iteration count above does,
  and has not been repeated across independent runs to characterize noise.
- No Windows/macOS/ARM64 measurement; spec/PR-7.md owns the cross-platform
  installation matrix.
- `faces/second` for `zero_faces_detect` is undefined (no faces), left `null`
  rather than reported as a misleading value.
- The thread-count numerical-parity result is validated against the same
  corpus the graph-optimization axis uses (11 real faces, 8 distinct chips,
  synthetic batches, and every exported stage tensor — see
  `tools/conversion/README.md`'s "Corpus and its limits"), which is itself a
  narrow real-face corpus per PR-1's own deferral of a representative
  30-image/100-face set; it is not a claim about arbitrary unseen images.
