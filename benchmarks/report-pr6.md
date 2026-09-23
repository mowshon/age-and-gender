# PR-6 performance report

Machine: this repository's development VM (AMD Ryzen 5 3600, 12 logical
cores, Linux 6.8, CPython 3.13.15) — **not** a dedicated benchmark host; see
`benchmarks/README.md`'s "Machine caveat". All numbers below are reproducible
with the commands in that file and are recorded in full, with per-repeat
medians and p95s, in
`benchmarks/results/python-pipeline-linux-x86_64-python313.json`
(`--warmup 10 --iterations 50 --repeats 3 --cold-runs 5
--steady-state-iterations 100`).

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
by the CNNs this PR could safely touch, and the two runtime-configuration axes
spec/PR-6.md calls out (graph optimization level, thread count) were either
already evaluated and rejected (graph fusion) or are evidenced-but-deferred
(thread count) rather than blindly left alone — see below.

## Old vs. new: the acceptance gate

spec/PR-6.md: *"warmed median one-face and five-face latency should not
regress by more than 10% versus the measured legacy public baseline."*

| Case | Faces | Legacy warm median | New warm median | Change |
| --- | ---: | ---: | ---: | ---: |
| `test-image.jpg`, explicit single box | 1 | 46.28 ms | 4.80 ms | **-90%** |
| `test-image-2.jpg`, detected | 2 | 116.35 ms | 28.90 ms | **-75%** |
| `test-image.jpg`, detected | 5 | 325.18 ms | 82.55 ms | **-75%** |

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

## Where the time actually goes

| Stage (5-face `test-image.jpg`) | Median |
| --- | ---: |
| Detection (`FaceFrontend.detect`) | 63.34 ms |
| Age network, 5 chips (`NeuralNetwork.run`) | 10.64 ms |
| Gender network, 5 chips | 3.39 ms |
| Landmarks, one face | 0.41 ms |
| Postprocessing, 5 faces | 0.18 ms |
| Chip extraction, one face (both sizes) | 0.11 ms |
| Image validation (already-contiguous array) | 0.0006 ms |

Full-image comparison, same source image, explicit boxes vs. detection:

| Scenario | Faces | Median | Faces/second |
| --- | ---: | ---: | ---: |
| Detect, 0 faces (`dogs.jpg`, 900x916) | 0 | 60.70 ms | — |
| Detect, 5 faces (`test-image.jpg`, 1100x825) | 5 | 82.55 ms | 60.6 |
| Detect, 2 faces (`test-image-2.jpg`, 634x435) | 2 | 28.90 ms | 69.2 |
| Explicit boxes, 1 face | 1 | 4.80 ms | 208.4 |
| Explicit boxes, 5 faces (detector skipped) | 5 | 18.74 ms | 266.8 |
| Explicit boxes, 50 faces (detector skipped) | 50 | 173.21 ms | 288.7 |

Detection cost tracks image *pixel count*, not face count (the 0-face,
900x916 `dogs.jpg` costs more detector time than the 2-face, 634x435 image).
Skipping it (explicit boxes) drops 5-face latency from 82.55 ms to 18.74 ms —
detection is roughly 3/4 of whole-image latency on these examples, exactly the
"HOG frontend may dominate after CNN optimization" spec/INVESTIGATION.md
predicted. This is also why bounded chunking's effect is invisible at
end-to-end scale for these example images: the CNNs it touches are already the
smaller cost, and explicit-box calls (which skip detection) scale near-linearly
with face count regardless (4.80 ms/face at 1 face, 3.75 ms/face at 5, 3.46
ms/face at 50), which is what "chunking doesn't change results or blow up
memory past 32 faces" is supposed to look like, not a throughput win by itself.

## Batch-size sweep (`NeuralNetwork.run`, prepared tensor only)

| Batch | Gender median | Gender faces/s | Age median | Age faces/s | Chunked calls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.70 ms | 1430 | 2.23 ms | 449 | 1 |
| 2 | 1.37 ms | 1462 | 4.38 ms | 457 | 1 |
| 8 | 5.45 ms | 1469 | 16.85 ms | 475 | 1 |
| 32 | 21.93 ms | 1459 | 66.62 ms | 480 | 1 |
| 64 | 47.50 ms | 1348 | 136.56 ms | 469 | 2 (32+32) |

Throughput is essentially flat from batch 1 to 32 (single-threaded execution;
see below), which is expected and fine — batching's win here is fewer
Python/ONNX-Runtime call crossings for a multi-face image, not sub-linear
per-sample cost. The batch-64 row is one bounded-chunking call (two
size-32 ONNX Runtime calls); it costs about what two separate 32-batches would,
i.e. chunking adds no measurable overhead beyond the extra call itself.

## Steady state and resource lifecycle

100 repeated `predict()` calls on one predictor: `age`/`gender` sessions,
frontend, and the process-wide bundle cache (`bundled_models.cache_info()`
stayed at 1 hit / 1 miss throughout) were all the same objects at the end as
at the start — no reinitialization, no growth. First-window vs. last-window
median latency ratio was 0.985 (noise, not growth). Peak RSS for the whole
run was 220,192 KiB. Exactly one age session, one gender session, one
detector, and one shape predictor existed throughout, regardless of face
count or repeat count.

## Cold start

Fresh-process medians, 5 subprocesses: import 128.7 ms, `AgeAndGender()`
construction 2.6 ms (does no I/O by design), model+session+frontend
initialization 331.9 ms, first `predict()` call after that 92.3 ms, second
call 85.0 ms — consistent with "first" already being close to "warmed"
because initialization (the actually expensive lazy step) already ran before
the timed first call. Total cold path (import → construct → init → first
predict) is roughly 555 ms.

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

**Thread count.** Not previously measured at any value other than 1/1
(`tools/conversion/validate_conversion.py`'s `THREAD_SETTINGS` is a constant,
never varied). A one-off investigation for this report found two things:
first, `intra_op_num_threads` does buy real throughput on larger batches at
`ORT_DISABLE_ALL` (raw session, age model, batch 32: 65.9 ms at 1 thread vs.
41.4 ms at 6 threads on this machine); second, unlike the graph-optimization
axis, it did not change the result at all on the frozen 11-face corpus
(`max|output(threads=N) - output(threads=1)| == 0.0` for N in 2/4/6, both
tasks) — `ORT_DISABLE_ALL` appears to keep per-sample computation
thread-count-independent and only parallelizes across independent batch rows.
That is a genuinely promising, evidenced lead, but it is **not** shipped here:
raising it is a manifest/`SUPPORTED_RUNTIME` contract change
(`_models.py`), which `tools/conversion/build_bundle.py` and
`tools/conversion/validate_conversion.py` would need to validate across the
*full* parity corpus (not an 11-face spot check) before it can replace the
shipped default, and it needs the "avoid nested-thread oversubscription"
comparison spec/PR-6.md asks for — this package's own `ConcurrencyTests`
already exercises multiple `predict()` calls from independent threads, and
each with its own multi-threaded session could oversubscribe a small
container. The end-to-end benefit would also be modest for these examples,
since detection (not the CNNs) dominates whole-image latency. Flagged as a
follow-up rather than attempted here.

## Limitations

- Single development machine, not spec/PR-6.md's "dedicated machine" /
  "reference machine"; see `benchmarks/README.md`.
- The reduced iteration count actually used (50 iterations × 3 repeats, not
  the documented default of 100 × 5) trades some statistical confidence for a
  runtime that fits this session; every full-image and batch-sweep median
  above is still a median over 150 timed calls (50 × 3), not a single sample,
  and `per_repeat_median_ms` in the JSON report shows run-to-run spread.
- No Windows/macOS/ARM64 measurement; spec/PR-7.md owns the cross-platform
  installation matrix.
- `faces/second` for `zero_faces_detect` is undefined (no faces), left `null`
  rather than reported as a misleading value.
- The thread-count finding above is a single-session, single-machine,
  11-chip spot check, not a corpus-wide validated result; treat it as a lead,
  not a measured claim about the shipped configuration.
