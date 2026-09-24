# Python migration specification

Investigated on **2026-09-22**, against commit
`e5c912f6ba739f30a45c04208b6d16500e4488cd` (package version 1.0.1).

## Recommendation

Build a Python-owned package that runs the **existing trained age and gender
networks through ONNX Runtime**, and uses **a prebuilt dlib Python wheel** for
the existing detector, landmarks, and face alignment. Convert the two neural
networks once in maintainer tooling, and distribute the converted models with
the package. Ordinary developers should only need:

```bash
python -m pip install age-and-gender
```

```python
from age_and_gender import AgeAndGender
from PIL import Image

predictor = AgeAndGender()
with Image.open("photo.jpg") as image:
    results = predictor.predict(image.convert("RGB"))
```

This is the **target API**, not functionality implemented by this specification.
The import name and result dictionaries stay compatible with the old interface.

### Meaning of “Python-side”

The recommended release has Python application code, a platform-independent
project wheel, no vendored `libs/`, and no project C++ extension or end-user
compiler requirement. NumPy, Pillow, ONNX Runtime, and the proposed `dlib-bin`
dependency still contain native code delivered in wheels. `dlib-bin` is a
third-party wheel distribution; it imports as `dlib`.

This interpretation addresses the stated installation problem. If the requirement
is **no dlib at runtime at all**, the detector and landmark/alignment algorithms
also need replacement or a faithful port. [PR-8](PR-8.md) specifies that conditional
follow-up. Replacing them with a different face model cannot be described as
preserving the old results. A literally native-code-free dependency tree would
also exclude NumPy and ONNX Runtime and is outside the recommended architecture.

## Read first

- [Investigation, evidence, and architecture](INVESTIGATION.md)
- [PR-1 — Freeze the legacy behavior and reference oracle](PR-1.md)
- [PR-2 — Convert the existing neural networks to ONNX](PR-2.md)
- [PR-3 — Build the Python package and model runtime](PR-3.md)
- [PR-4 — Preserve detection, landmarks, and alignment](PR-4.md)
- [PR-5 — Implement the compatible public Python API](PR-5.md)
- [PR-6 — Optimize measured bottlenecks without changing results](PR-6.md)
- [PR-7 — Validate installation, remove native build infrastructure, release](PR-7.md)
- [PR-8 — Conditional dlib-free runtime investigation and implementation gates](PR-8.md)
- [PR-9 — Cleanup: remove legacy tooling and internal-plan references](PR-9.md)

## Project development environment

Use the existing repository-local **`venv/`**, verified as Python **3.13.15**.
This interpreter is within the proposed supported Python range.

Run project commands from the repository root with the explicit interpreter:

```bash
venv/bin/python --version
venv/bin/python -m pip --version
```

Once the relevant development tools and package scaffold are introduced by the
implementation PRs, use the same interpreter for them:

```bash
venv/bin/python -m pytest
venv/bin/python -m ruff check .
venv/bin/python -m build
```

Shell activation with `source venv/bin/activate` is optional; explicit interpreter
paths also work across independent terminal sessions. Reuse this environment for
project dependencies, tests, and Python conversion tooling. Legacy reproduction
uses a separate pinned environment only where PR-1 requires historical tooling.
`AGENTS.md` records this rule for future coding sessions.

## Final repository layout

Target after PR-7; the optional PR-8 can change frontend assets/implementation:

```text
age-and-gender/
├── AGENTS.md
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/                 # Tests, wheel installation, release validation
├── src/
│   └── age_and_gender/            # Installable Python package
│       ├── __init__.py
│       ├── api.py
│       ├── _images.py
│       ├── _faces.py
│       ├── _inference.py
│       ├── _postprocess.py
│       ├── _models.py
│       ├── _types.py
│       ├── py.typed
│       └── models/               # Included in the installed package
│           ├── manifest.json
│           ├── age-v1.onnx
│           ├── gender-v1.onnx
│           ├── shape_predictor_5_face_landmarks.dat
│           └── notices/
├── tests/
│   ├── fixtures/
│   │   └── legacy/
│   ├── unit/
│   ├── parity/
│   └── integration/
├── tools/                        # Maintainer-only reference and conversion tools
│   ├── legacy/
│   └── conversion/
├── benchmarks/                   # Reproducible performance measurements
├── example/                      # Existing examples, updated for the Python API
├── spec/                         # Investigation and PR specifications
├── models/                       # Existing local source models, Git-ignored
└── venv/                         # Existing local Python environment, Git-ignored
```

New main directories are `src/age_and_gender/`, `tests/`, `tools/`, `benchmarks/`,
and `.github/workflows/`. Existing `example/`, `spec/`, `models/`, and `venv/` keep
their roles. Generated `build/`, `dist/`, and test caches are local artifacts.

The two model directories serve different purposes: root `models/` holds the
original local downloads; `src/age_and_gender/models/` holds the validated assets
distributed to users. Maintainer C++ reference/export programs live under
`tools/`, outside the package's installation path. Root `libs/`, `CMakeLists.txt`,
the custom `setup.py`, and `src/main.cpp` are removed in PR-7 after parity passes.

## Integration-branch workflow

Use **`refactoring`** as the integration branch, based on the current default
branch **`master`**. Each implementation PR has its own short-lived branch and
sets its base/target to `refactoring`:

```text
master
└── refactoring
    ├── refactor/pr-1-legacy-oracle      → PR into refactoring
    ├── refactor/pr-2-onnx-conversion    → PR into refactoring
    ├── refactor/pr-3-python-runtime     → PR into refactoring
    ├── refactor/pr-4-face-pipeline      → PR into refactoring
    ├── refactor/pr-5-public-api         → PR into refactoring
    ├── refactor/pr-6-performance        → PR into refactoring
    └── refactor/pr-7-release           → PR into refactoring

refactoring                            → final PR into master
```

Create the integration branch from `master`:

```bash
git switch master
git switch -c refactoring
```

Create the first implementation branch from it:

```bash
git switch -c refactor/pr-1-legacy-oracle
```

For later dependent work, start from the updated `refactoring` branch after the
prerequisite PR has merged. For example, PR-2 starts after PR-1 is integrated.
Independent development can follow the dependency graph below, but dependent
PRs should not merge before their prerequisites. This avoids repeatedly including
unmerged prerequisite changes in each PR's diff.

Configure CI to run on pull requests targeting `refactoring` as well as `master`.
Keep the same parity and installation checks on the integration branch. Once
PR-7 passes, open one final migration PR from `refactoring` into `master`.
Conditional PR-8 can be a follow-up after that release.

`.gitignore` previously listed `/spec` and `AGENTS.md`, which kept the
specifications and the agent instructions local-only. **PR-2 removed both entries**,
so `spec/` and `AGENTS.md` are tracked and travel with pull requests. Root
`/models` and `/venv/` remain local-only; the package's nested model assets are
tracked.

## Execution order

```text
PR-1 ── PR-2 ── PR-3 ──┐
  └──── PR-4 ───────────┼── PR-5 ── PR-6 ── PR-7
                       └─────────────────── PR-8 (conditional follow-up)
```

PR-4 can be developed against the oracle while conversion is underway. PR-3 owns
the package/build scaffold; PR-4's production module lands into that scaffold.
There must be a working, parity-validated candidate before deleting native sources.
ONNX conversion and frontend parity are early feasibility gates, not cleanup work
to leave until the end.

## Shared acceptance contract

1. The actual downloaded model weights are reused; no retraining or substitute
   age/gender models.
2. Exact public-output agreement on the frozen compatibility corpus: face count,
   order, rectangles, age integer, gender string, and integer confidences.
3. Intermediate probability tensors satisfy the explicit tolerances in PR-1.
   A passing tensor tolerance does not excuse a changed public integer or label.
4. All advertised platforms pass binary-only installation and offline inference
   from the built wheel in a clean environment.
5. Default prediction requires neither model downloads nor source compilation.
6. Supported Python/dependency versions are decided by tested wheel availability
   and parity, using the latest stable versions where they pass.
7. Performance changes are accompanied by reproducible measurements and the same
   parity checks. No unmeasured speed claims.
8. Type hints and concise Google-style public docstrings; implementation comments
   explain compatibility subtleties rather than narrating each line.

“Same results” is a tested release contract, not a claim of bit-identical floating
point on every possible image, CPU, and execution provider. The probability-to-
integer boundaries make that universal guarantee unrealistic for a runtime
change. A failed parity gate blocks release rather than silently weakening this
contract.

## Investigation status

Source inspection, model hashes, live dependency metadata, a legacy C++ inference
probe, and Python frontend comparisons are complete. The local examples produced
seven detected faces with matching frontend checksums. ONNX conversion,
full-package parity, cross-platform execution, and performance improvements remain
implementation work. The PR files are specifications, not claims those steps
have already passed.
