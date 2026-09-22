# Project environment

- Use the existing project virtual environment at `venv/` for Python development.
- Run Python as `venv/bin/python` and pip as `venv/bin/python -m pip` from the
  repository root. Use this interpreter for tests, linting, builds, and conversion
  scripts too, for example `venv/bin/python -m pytest`.
- Reuse this environment rather than creating another one or installing project
  dependencies into system Python.
- The environment was verified as Python 3.13.15. Keep `venv/` out of Git and
  package artifacts.
- Historical legacy-oracle reproduction may use its separately pinned environment
  as specified in `spec/PR-1.md`; keep it isolated from the project environment.

# Migration specification

- Start with `spec/README.md` for the migration sequence, target layout, and
  integration-branch workflow.
- The specifications describe planned work; verify each PR's acceptance criteria
  before treating its implementation as complete.
