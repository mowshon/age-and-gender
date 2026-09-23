# Benchmarks

`benchmark_pipeline.py` measures the installed Python package end to end and
stage by stage, and produces one machine-readable JSON report. It exists to
satisfy spec/PR-6.md's "benchmark first" requirement: every optimization in
that PR is only justified by a number this script (or the frozen legacy
baseline it compares against) produced.

## Running it

```bash
venv/bin/python benchmarks/benchmark_pipeline.py \
  --output benchmarks/results/python-pipeline-<platform>.json
```

Defaults follow spec/PR-6.md's measurement protocol: 10 warmups and 100
measured iterations per case, repeated across 5 independent runs, plus 5 fresh
subprocesses for cold-start numbers and 200 steady-state calls. That full
protocol takes several minutes. Pass `--quick` for a fast development sanity
pass (2 warmups, 10 iterations, 1 repeat) — **numbers from `--quick` must not
be used to evaluate the 10% regression gate or reported as a performance
claim**; the script labels them as such in `environment.note` regardless, but
`--quick` makes the departure from the protocol explicit at the call site too.
Every count actually used is recorded in the report's `environment` block, so
a report is self-describing even if these defaults change later.

The checked-in `benchmarks/results/python-pipeline-linux-x86_64-python313.json`
was captured with `--iterations 50 --repeats 3` (a reduced but still
multi-repeat, multi-hundred-sample capture) rather than the full defaults, on
this repository's development machine — see "Machine caveat" below.

## What it measures

One JSON report with these top-level sections:

- `environment` — platform, CPU, Python/dependency versions, the ONNX Runtime
  session options and default batch bound actually in effect, and the
  warmup/iteration/repeat counts used for every case in the report.
- `cold_start` — a fresh interpreter's import time, `AgeAndGender()`
  construction (cheap; it does no I/O), model/session initialization, image
  decode+validation, and the first vs. second `predict()` call, each timed
  separately, averaged over several fresh subprocesses.
- `stages` — image validation, detection, landmark prediction, each chip size's
  extraction, normalization, each network's `run()`, and postprocessing,
  timed in isolation via the internal modules (not just inferred from the
  full-pipeline total).
- `batch_size_sweep` — each network's `run()` latency and faces/second at
  batch sizes 1, 2, 8, 32, and 64 (one past the default chunk bound), so the
  effect of chunking at the boundary is directly visible.
- `full_image_scenarios` — whole-image `predict()` latency for 0/1/2/5/50
  faces, explicit boxes and auto-detection kept separate, first call vs.
  warmed-median latency, and faces/second.
- `steady_state` — 100+ repeated calls on one predictor, checking that the age
  and gender sessions, the frontend, and the bundled-models cache are the same
  objects at the end as at the start (no accidental reinitialization), and
  comparing the first and last measurement windows for latency growth.
- `memory_and_model_counts` — peak RSS (`ru_maxrss`, KiB, Linux) and a count of
  live sessions/detectors/predictors, which should never exceed one each.
- `legacy_comparison` — the new warmed medians against the frozen legacy
  oracle numbers for the same face counts; see below for where those numbers
  come from.

All latency figures are `median_ms`/`p95_ms`/`mean_ms`/`min_ms`/`max_ms` plus
`per_repeat_median_ms`, so both the typical case and run-to-run noise are
visible without re-running anything.

## Legacy comparison inputs

`compare_to_legacy()` reads two pre-existing measurements rather than
re-running the C++ extension itself (the legacy build is maintainer-only
tooling; see spec/PR-1.md and `tools/legacy/`):

- `benchmarks/results/linux-x86_64-python310.json` — spec/PR-1.md's frozen
  legacy oracle report, already covering `test-image.jpg` (5 faces) and
  `test-image-2.jpg` (2 faces), produced by `legacy_baseline.py` from
  `tools/legacy/oracle.cpp` runs.
- `benchmarks/results/legacy-single-face-oracle-report.json` — neither example
  image has exactly one face, so spec/PR-6.md's "one-face" acceptance case has
  no equivalent in the file above. This is the same oracle binary, run once
  more with an explicit single box (spec/INVESTIGATION.md's first reference
  face, `[419, 266, 506, 352]` in `[L, T, R, B]`) instead of the detector, so
  the timing reflects one real face's frontend + CNN cost rather than a
  detector-driven whole image with one face in it. Reproduce it with:

  ```bash
  # from a fresh build/legacy-oracle/ per tools/legacy/README.md
  ./build/legacy-oracle/age_and_gender_legacy_oracle \
    --image build/reference/test-image-v4/input.rgb \
    --width 1100 --height 825 \
    --models example/models \
    --output build/reference/test-image-single-face \
    --benchmark-runs 5 \
    --box 266,506,352,419  # legacy top,right,bottom,left order
  cp build/reference/test-image-single-face/oracle-report.json \
    benchmarks/results/legacy-single-face-oracle-report.json
  ```

  `build/reference/test-image-v4/input.rgb` is produced by
  `tools/legacy/freeze_reference.py` (see `tools/legacy/README.md`); this
  capture reused the executable and raw RGB input already pinned by the
  checked-in baseline (its `executable_sha256` matched the freshly rebuilt
  binary byte for byte) and additionally confirmed the reported result
  (`female`, age 26, confidence 84) matches
  `tests/fixtures/legacy/test-image.golden.json`'s first face, so the box
  really did select the same face the golden fixtures use.

Both comparisons are old and new results on the same decoded RGB bytes, the
same source weights, and the same machine, per spec/PR-6.md's "Benchmark
first" requirement.

## Machine caveat

spec/PR-6.md's acceptance criteria are meant to run "on the named reference
machine" / "controlled hardware, not timing-sensitive assertions on shared CI
runners." This repository has no such dedicated benchmark host; every report
under `results/` was captured on a shared development machine and says so in
its own `environment.note` field. Treat the relative numbers (new vs. legacy,
one batch size vs. another) as informative and reproducible; treat the
absolute milliseconds as one machine's sample, to be re-measured before they
gate a release.

## Result files

- `linux-x86_64-python310.json` — spec/PR-1.md's frozen legacy oracle baseline
  (whole-image, 5- and 2-face cases). Not regenerated by this PR.
- `legacy-single-face-oracle-report.json` — the one-face legacy oracle capture
  described above.
- `python-pipeline-linux-x86_64-python313.json` — this PR's Python package
  report, `schema_version: 1` (this document + the section list above is the
  schema).
