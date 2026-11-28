# GitHub Actions

`ci.yml` runs on every push and pull request (Ubuntu, Python 3.12):

1. install the project with its `dev` extras, constrained by `requirements.lock`;
2. `pip check`;
3. `pytest -q`.

Tests that need local-only inputs (raw Loghub logs, Phase-1 artifacts under `data/processed/`, GPU
libraries such as `torch`) skip with a reason, so a clean clone passes. Expected on a clean clone:
474 passed, 57 skipped.
