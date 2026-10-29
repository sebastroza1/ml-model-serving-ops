# Reproduction Guide

1. Create the Python 3.12 environment from `README.md`.
2. Run the training command to regenerate `artifacts/model.joblib` and `metadata.json`.
3. Run `ruff check .`, `pytest`, and `pip check`.
4. Start Uvicorn and query all five endpoints.
5. Build the Docker image only when the Docker daemon is available.
6. Apply Kubernetes manifests only to a local test cluster and record that result separately.

The fixed generator seed, package pins, artifact metadata, API tests, and version response provide the reproducibility boundary for this experiment.
